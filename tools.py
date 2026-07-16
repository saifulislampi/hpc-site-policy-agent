"""Bounded, site-aware web discovery tools for HPC documentation."""

from __future__ import annotations

import io
import ipaddress
import re
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import urldefrag, urljoin, urlparse

import httpx
import trafilatura
from bs4 import BeautifulSoup
from ddgs import DDGS
from ddgs.exceptions import DDGSException
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, ValidationError
from pypdf import PdfReader

from discovery import (
    classify_source,
    derive_site_identity,
    generate_discovery_queries,
    policy_matches,
    repair_query,
    root_matches,
    topic_matches,
)
from models import ToolCall, ToolDefinition, ToolExecutionError, ToolResult
from schemas import (
    CandidateSource,
    DiscoveryCoverage,
    DiscoverySelection,
    DocumentBlock,
    DocumentLink,
    DocumentSection,
    FetchedDocument,
    FinishDiscoveryArgs,
    SiteIdentity,
    TopicCoverage,
    TopicName,
)


SearchBackend = Callable[[str, int], list[dict[str, str]]]
PageBackend = Callable[[str], dict[str, Any]]
EventCallback = Callable[[str, dict[str, Any]], None]


class StrictArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SearchArgs(StrictArgs):
    query: str = Field(min_length=3, max_length=500)


class FetchArgs(StrictArgs):
    url: HttpUrl


@dataclass(slots=True)
class RegisteredTool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any]], dict[str, Any]]
    terminal: bool = False

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )


class ScoutTools:
    def __init__(
        self,
        *,
        allowed_domains: list[str],
        site_identity: SiteIdentity | None = None,
        search_results: int = 8,
        max_page_chars: int = 20_000,
        search_budget: int = 12,
        page_budget: int = 8,
        search_backend: SearchBackend | None = None,
        page_backend: PageBackend | None = None,
    ) -> None:
        if not allowed_domains:
            raise ValueError("At least one approved domain is required.")

        self.allowed_domains = [self._normalize_domain(x) for x in allowed_domains]
        self.site_identity = site_identity or derive_site_identity(
            display_name=self.allowed_domains[0],
            organization_domains=self.allowed_domains,
        )
        self.search_results = search_results
        self.max_page_chars = max_page_chars
        self.search_budget = search_budget
        self.page_budget = page_budget
        self._search_backend = search_backend or self._ddgs_search
        self._page_backend = page_backend or self._http_fetch
        self._event_callback: EventCallback | None = None

        self.documents: dict[str, FetchedDocument] = {}
        self.rejected_documents: dict[str, FetchedDocument] = {}
        self.candidates: dict[str, CandidateSource] = {}
        self.canonical_root: str | None = None
        self.query_log: list[dict[str, Any]] = []
        self.links_followed: list[str] = []
        self.selected_urls: set[str] = set()
        self._attempted_urls: set[str] = set()
        self._searches_used = 0
        self._pages_used = 0
        self._topic_queries: dict[TopicName, int] = {
            "submission_policy": 0,
            "networking_policy": 0,
        }
        self._topic_pages: dict[TopicName, set[str]] = {
            "submission_policy": set(),
            "networking_policy": set(),
        }
        self._topic_evidence: dict[TopicName, set[str]] = {
            "submission_policy": set(),
            "networking_policy": set(),
        }

        self._tools = {
            "search_web": RegisteredTool(
                name="search_web",
                description=(
                    "Search approved institutional domains. Queries are repaired to "
                    "include the target-site alias, and sibling-site results are rejected."
                ),
                parameters=SearchArgs.model_json_schema(),
                handler=self._search_web,
            ),
            "fetch_page": RegisteredTool(
                name="fetch_page",
                description=(
                    "Fetch one ranked target-site or explicitly applicable organization "
                    "page. Sibling-site pages are deterministically rejected."
                ),
                parameters=FetchArgs.model_json_schema(),
                handler=self._fetch_page,
            ),
            "finish_discovery": RegisteredTool(
                name="finish_discovery",
                description=(
                    "Finish only after submission and networking coverage are resolved. "
                    "Every selected source must already be fetched and site-eligible."
                ),
                parameters=FinishDiscoveryArgs.model_json_schema(),
                handler=self._finish_discovery,
                terminal=True,
            ),
        }

    def set_event_callback(self, callback: EventCallback) -> None:
        self._event_callback = callback

    def definitions(self) -> list[ToolDefinition]:
        return [tool.definition() for tool in self._tools.values()]

    def bootstrap_discovery(self, *, keywords: list[str]) -> dict[str, Any]:
        """Locate a canonical root, crawl topic links, then run focused searches."""

        queries = generate_discovery_queries(self.site_identity)
        for query in queries["canonical_root"]:
            self._ranked_search(query, objective="canonical_root", generated=True)

        root_candidate = self._best_root_candidate()
        if root_candidate is not None:
            result = self._fetch_page({"url": str(root_candidate.url)})
            if result.get("ok"):
                self.canonical_root = str(root_candidate.url)
                self._emit(
                    "canonical_root",
                    {
                        "url": self.canonical_root,
                        "scope": root_candidate.classification.site_scope,
                        "score": root_candidate.classification.score,
                    },
                )
                self._crawl_canonical_root()

        for objective in ("submission_policy", "networking_policy"):
            for query in queries[objective]:
                self._ranked_search(
                    query,
                    objective=objective,
                    generated=True,
                )
            self._fetch_topic_candidates(
                objective,
                limit=3 if objective == "submission_policy" else 2,
            )

        for query in queries["operational_policy"]:
            self._ranked_search(
                query,
                objective="operational_policy",
                generated=True,
            )
        self._fetch_policy_candidates(limit=3)

        for keyword in keywords:
            if self._searches_used >= self.search_budget:
                break
            topic = self._infer_topic(keyword)
            self._ranked_search(keyword, objective=topic, generated=False)
            if topic is not None:
                self._fetch_topic_candidates(topic, limit=1)

        self._fetch_sibling_control()

        coverage = self.discovery_coverage()
        context = {
            "canonical_root": self.canonical_root,
            "queries": self.query_log,
            "coverage": coverage.model_dump(mode="json"),
            "fetched_documents": [
                self._document_context(document)
                for document in self.documents.values()
            ],
            "top_candidates": [
                {
                    "url": str(item.url),
                    "title": item.title,
                    "snippet": item.snippet[:500],
                    "scope": item.classification.site_scope,
                    "score": item.classification.score,
                    "fetched": item.fetched,
                }
                for item in self._ranked_candidates(eligible_only=True)[:10]
            ],
            "rejected_candidates": [
                {
                    "url": str(item.url),
                    "title": item.title,
                    "scope": item.classification.site_scope,
                    "score": item.classification.score,
                    "reason": item.rejection_reason,
                }
                for item in self._ranked_candidates(eligible_only=False)
                if item.classification.site_scope in {"sibling", "unrelated"}
            ][:10],
        }
        self._emit("coverage", coverage.model_dump(mode="json"))
        return context

    def execute(self, call: ToolCall) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            raise ToolExecutionError(f"Unknown tool requested: {call.name}")

        try:
            output = tool.handler(call.arguments)
        except (ValueError, ValidationError, httpx.HTTPError) as exc:
            output = {"ok": False, "error": str(exc)}

        return ToolResult(
            call_id=call.call_id,
            name=call.name,
            output=output,
            terminal=tool.terminal and output.get("ok", False),
        )

    def selected_documents(self, selection: DiscoverySelection) -> list[FetchedDocument]:
        documents: list[FetchedDocument] = []
        for raw_url in selection.source_urls:
            url = self._canonical_url(str(raw_url))
            document = self.documents.get(url)
            if document is None:
                raise ToolExecutionError(
                    f"Selected source was not fetched during discovery: {url}"
                )
            if document.classification.site_scope not in {
                "target_site",
                "organization_general",
            }:
                raise ToolExecutionError(
                    f"Selected source is not eligible for extraction: {url}"
                )
            if (
                document.classification.site_scope == "organization_general"
                and not self._organization_page_applies(document)
            ):
                raise ToolExecutionError(
                    f"Organization-general source does not explicitly apply to target: {url}"
                )
            documents.append(document)
        return documents

    def partial_selection(self, *, reason: str) -> DiscoverySelection:
        """Finish from deterministic crawl state when the model loop cannot finish."""

        coverage = self.discovery_coverage()
        topics = []
        for item in (coverage.submission_policy, coverage.networking_policy):
            if item.status == "not_investigated":
                item = item.model_copy(
                    update={
                        "status": "search_exhausted",
                        "notes": [*item.notes, f"Agent discovery ended early: {reason}"],
                    }
                )
            topics.append(item)
        coverage = coverage.model_copy(
            update={
                "submission_policy": topics[0],
                "networking_policy": topics[1],
            }
        )
        selected = list(self._coverage_support_urls(coverage))
        ranked_documents = sorted(
            self.documents.values(),
            key=lambda item: item.classification.score,
            reverse=True,
        )
        selected.extend(
            str(document.url)
            for document in ranked_documents
            if document.classification.site_scope == "target_site"
        )
        selected = list(dict.fromkeys(selected))[:10]
        self.selected_urls.update(selected)
        for url in selected:
            candidate = self.candidates.get(url)
            if candidate is not None:
                candidate.selected = True
        unanswered = [
            item.topic
            for item in topics
            if item.status != "evidence_found"
        ]
        selection = DiscoverySelection(
            source_urls=selected,
            summary=(
                "Deterministic discovery state was retained after the model loop "
                f"ended early: {reason}"
            ),
            unanswered_topics=unanswered,
            canonical_root=self.canonical_root,
            coverage=coverage,
            termination_reason="partial_discovery_fallback",
        )
        self._emit(
            "selection",
            {
                "selected_urls": selected,
                "termination_reason": selection.termination_reason,
                "coverage": coverage.model_dump(mode="json"),
                "fallback_reason": reason,
            },
        )
        return selection

    def corpus_documents(self) -> list[FetchedDocument]:
        """Return eligible web pages plus retained sibling negative controls."""

        eligible = [
            document
            for document in self.documents.values()
            if document.classification.site_scope == "target_site"
            or (
                document.classification.site_scope == "organization_general"
                and self._organization_page_applies(document)
            )
        ]
        siblings = [
            document
            for document in self.rejected_documents.values()
            if document.classification.site_scope == "sibling"
        ]
        return [*eligible, *siblings]

    def discovery_coverage(self) -> DiscoveryCoverage:
        return DiscoveryCoverage(
            canonical_root_found=self.canonical_root is not None,
            canonical_root=self.canonical_root,
            submission_policy=self._topic_coverage("submission_policy"),
            networking_policy=self._topic_coverage("networking_policy"),
        )

    def discovery_metrics(self) -> dict[str, int]:
        """Return deterministic request counters for traces and evaluations."""

        return {
            "search_requests": self._searches_used,
            "page_requests": self._pages_used,
            "search_budget": self.search_budget,
            "page_budget": self.page_budget,
            "candidate_count": len(self.candidates),
            "fetched_target_documents": sum(
                document.classification.site_scope == "target_site"
                for document in self.documents.values()
            ),
            "fetched_eligible_documents": len(self.documents),
            "rejected_fetched_documents": len(self.rejected_documents),
        }

    def _search_web(self, raw: dict[str, Any]) -> dict[str, Any]:
        args = SearchArgs.model_validate(raw)
        topic = self._infer_topic(args.query)
        return self._ranked_search(args.query, objective=topic, generated=False)

    def _ranked_search(
        self,
        query: str,
        *,
        objective: str | TopicName | None,
        generated: bool,
    ) -> dict[str, Any]:
        repaired_query, repaired = repair_query(query, self.site_identity)
        if self._searches_used >= self.search_budget:
            return {
                "ok": False,
                "error": f"Search budget exhausted at {self.search_budget} queries.",
                "query": repaired_query,
            }

        self._searches_used += 1
        if objective in {"submission_policy", "networking_policy"}:
            self._topic_queries[objective] += 1
        try:
            raw_results = self._search_backend(
                repaired_query,
                max(self.search_results * 3, self.search_results),
            )
        except (DDGSException, httpx.HTTPError) as exc:
            log_entry = {
                "query": repaired_query,
                "original_query": query,
                "repaired": repaired,
                "generated": generated,
                "objective": objective,
                "result_count": 0,
                "error": str(exc),
            }
            self.query_log.append(log_entry)
            self._emit("query", log_entry)
            return {
                "ok": False,
                "query": repaired_query,
                "query_repaired": repaired,
                "error": f"Search request failed: {exc}",
            }
        seen: set[str] = set()
        current: list[CandidateSource] = []
        for item in raw_results or []:
            raw_url = item.get("href") or item.get("url")
            if not raw_url:
                continue
            try:
                url = self._validate_url(raw_url)
            except ValueError:
                continue
            if url in seen:
                continue
            seen.add(url)
            title = str(item.get("title") or "")[:500]
            snippet = str(item.get("body") or item.get("snippet") or "")[:1000]
            classification = classify_source(
                identity=self.site_identity,
                url=url,
                title=title,
                text=snippet,
                linked_from_canonical_root=False,
            )
            rejection_reason = self._rejection_reason(classification.site_scope)
            candidate = CandidateSource(
                url=url,
                title=title,
                snippet=snippet,
                classification=classification,
                fetched=url in self.documents or url in self.rejected_documents,
                selected=url in self.selected_urls,
                rejection_reason=rejection_reason,
            )
            self._upsert_candidate(candidate)
            current.append(self.candidates[url])
            self._emit(
                "source_classification",
                self.candidates[url].model_dump(mode="json"),
            )

        current.sort(
            key=lambda item: self._candidate_rank(item, objective),
            reverse=True,
        )
        log_entry = {
            "query": repaired_query,
            "original_query": query,
            "repaired": repaired,
            "generated": generated,
            "objective": objective,
            "result_count": len(current),
        }
        self.query_log.append(log_entry)
        self._emit("query", log_entry)
        self._emit(
            "candidate_rankings",
            {
                "query": repaired_query,
                "candidates": [
                    {
                        "url": str(item.url),
                        "scope": item.classification.site_scope,
                        "score": item.classification.score,
                        "rank_score": self._candidate_rank(item, objective),
                        "rejection_reason": item.rejection_reason,
                    }
                    for item in current
                ],
            },
        )
        eligible = [
            item
            for item in current
            if item.classification.site_scope in {
                "target_site",
                "organization_general",
            }
        ][: self.search_results]
        rejected = [
            {
                "url": str(item.url),
                "title": item.title,
                "scope": item.classification.site_scope,
                "score": item.classification.score,
                "rejection_reason": item.rejection_reason,
            }
            for item in current
            if item.classification.site_scope in {"sibling", "unrelated"}
        ]
        return {
            "ok": True,
            "query": repaired_query,
            "query_repaired": repaired,
            "approved_domains": self.allowed_domains,
            "results": [item.model_dump(mode="json") for item in eligible],
            "rejected_results": rejected,
        }

    def _fetch_page(self, raw: dict[str, Any]) -> dict[str, Any]:
        args = FetchArgs.model_validate(raw)
        url = self._validate_url(str(args.url))
        preliminary = self.candidates.get(url)
        if preliminary is None:
            classification = classify_source(
                identity=self.site_identity,
                url=url,
            )
            preliminary = CandidateSource(
                url=url,
                title="",
                snippet="",
                classification=classification,
                rejection_reason=self._rejection_reason(classification.site_scope),
            )
            self._upsert_candidate(preliminary)

        if preliminary.classification.site_scope == "unrelated":
            return {
                "ok": False,
                "rejected": True,
                "url": url,
                "scope": preliminary.classification.site_scope,
                "error": preliminary.rejection_reason,
            }
        if url in self.documents:
            return self._document_tool_output(self.documents[url], cached=True)
        if url in self._attempted_urls:
            return {
                "ok": False,
                "url": url,
                "error": "URL was already attempted during this run.",
                "cached": True,
            }
        if self._pages_used >= self.page_budget:
            return {
                "ok": False,
                "error": f"Page budget exhausted at {self.page_budget} pages.",
                "url": url,
            }

        self._pages_used += 1
        self._attempted_urls.add(url)
        try:
            raw_page = self._page_backend(url)
        except httpx.HTTPError as exc:
            self._emit(
                "page_fetch_failed",
                {"url": url, "error": str(exc)},
            )
            return {
                "ok": False,
                "url": url,
                "error": f"Page request failed: {exc}",
            }
        final_url = self._validate_url(str(raw_page.get("url") or url))
        content_type = str(raw_page.get("content_type") or "text/html").lower()

        if "application/pdf" in content_type or final_url.lower().endswith(".pdf"):
            content = raw_page.get("content") or b""
            full_text = self._extract_pdf(content)
            title = final_url.rsplit("/", 1)[-1]
            links: list[DocumentLink] = []
            sections = [
                DocumentSection(
                    heading=title,
                    heading_path=[title],
                    text=full_text,
                    links=[],
                    blocks=[DocumentBlock(kind="text", text=full_text)],
                )
            ]
            main_heading = title
        else:
            html = str(raw_page.get("text") or "")
            soup = BeautifulSoup(html, "html.parser")
            title = soup.title.get_text(" ", strip=True) if soup.title else final_url
            main_heading_tag = soup.find("h1")
            main_heading = (
                main_heading_tag.get_text(" ", strip=True)
                if main_heading_tag
                else title
            )
            full_text = trafilatura.extract(
                html,
                include_links=False,
                include_images=False,
                favor_precision=True,
            ) or self._fallback_text(soup)
            links = self._extract_links(soup, final_url)
            sections = self._extract_sections(soup, final_url)

        full_text = re.sub(r"\n{3,}", "\n\n", full_text).strip()
        if not full_text:
            raise ValueError("No readable document text was extracted.")

        linked_from_root = final_url in self.links_followed
        classification = classify_source(
            identity=self.site_identity,
            url=final_url,
            title=title,
            heading=main_heading,
            text=full_text,
            linked_from_canonical_root=linked_from_root,
        )
        relevant_text = self._relevant_text(sections, full_text)
        document = FetchedDocument(
            url=final_url,
            title=title[:500],
            text=full_text[: self.max_page_chars],
            links=links[:200],
            sections=sections[:100],
            relevant_text=relevant_text,
            text_truncated=len(full_text) > self.max_page_chars,
            classification=classification,
        )
        rejection_reason = self._rejection_reason(classification.site_scope)
        candidate = CandidateSource(
            url=final_url,
            title=title[:500],
            snippet=relevant_text[:1000],
            classification=classification,
            fetched=True,
            selected=final_url in self.selected_urls,
            rejection_reason=rejection_reason,
        )
        self._upsert_candidate(candidate)
        if classification.site_scope in {"sibling", "unrelated"}:
            self.rejected_documents[final_url] = document
            self._emit(
                "source_rejected",
                {
                    "url": final_url,
                    "scope": classification.site_scope,
                    "reason": rejection_reason,
                },
            )
            return {
                "ok": False,
                "rejected": True,
                "url": final_url,
                "scope": classification.site_scope,
                "error": rejection_reason,
            }

        self.documents[final_url] = document
        self._record_page_coverage(document)
        self._emit(
            "page_fetched",
            {
                "url": final_url,
                "scope": classification.site_scope,
                "score": classification.score,
                "text_truncated": document.text_truncated,
                "section_count": len(document.sections),
                "relevant_sections_preserved": bool(document.relevant_text),
            },
        )
        return self._document_tool_output(document, cached=False)

    def _finish_discovery(self, raw: dict[str, Any]) -> dict[str, Any]:
        args = FinishDiscoveryArgs.model_validate(raw)
        coverage = self.discovery_coverage()
        incomplete = [
            item.topic
            for item in (coverage.submission_policy, coverage.networking_policy)
            if item.status == "not_investigated"
        ]
        if incomplete:
            return {
                "ok": False,
                "error": "Discovery coverage is incomplete for: " + ", ".join(incomplete),
                "coverage": coverage.model_dump(mode="json"),
            }

        selected: list[str] = []
        rejected: list[dict[str, str]] = []
        for raw_url in args.source_urls:
            url = self._canonical_url(str(raw_url))
            document = self.documents.get(url)
            if document is None:
                rejected.append({"url": url, "reason": "source was not fetched"})
                continue
            if document.classification.site_scope not in {
                "target_site",
                "organization_general",
            }:
                rejected.append(
                    {"url": url, "reason": "source is not site-eligible"}
                )
                continue
            if (
                document.classification.site_scope == "organization_general"
                and not self._organization_page_applies(document)
            ):
                rejected.append(
                    {
                        "url": url,
                        "reason": "organization-general page does not explicitly apply",
                    }
                )
                continue
            selected.append(url)

        if not selected:
            selected = [
                url
                for url, document in self.documents.items()
                if document.classification.site_scope == "target_site"
            ][:4]
        selected = list(
            dict.fromkeys([*self._coverage_support_urls(coverage), *selected])
        )
        failure_is_explicit = any(
            item.status == "search_exhausted"
            for item in (coverage.submission_policy, coverage.networking_policy)
        )
        if not selected and not failure_is_explicit:
            return {
                "ok": False,
                "error": "No fetched target-site documents are eligible for extraction.",
                "rejected": rejected,
            }

        selected = list(dict.fromkeys(selected))[:10]
        self.selected_urls.update(selected)
        for url in selected:
            candidate = self.candidates.get(url)
            if candidate is not None:
                candidate.selected = True
        termination_reason = self._termination_reason(coverage)
        selection = DiscoverySelection(
            source_urls=selected,
            summary=args.summary,
            unanswered_topics=args.unanswered_topics,
            canonical_root=self.canonical_root,
            coverage=coverage,
            termination_reason=termination_reason,
        )
        self._emit(
            "selection",
            {
                "selected_urls": selected,
                "rejected": rejected,
                "termination_reason": termination_reason,
                "coverage": coverage.model_dump(mode="json"),
                "metrics": self.discovery_metrics(),
            },
        )
        return {
            "ok": True,
            "selection": selection.model_dump(mode="json"),
            "rejected": rejected,
        }

    def _best_root_candidate(self) -> CandidateSource | None:
        eligible = [
            item
            for item in self.candidates.values()
            if item.classification.site_scope == "target_site"
        ]
        if not eligible:
            return None
        return max(
            eligible,
            key=lambda item: item.classification.score
            + 4 * root_matches(f"{item.title} {urlparse(str(item.url)).path}"),
        )

    def _crawl_canonical_root(self) -> None:
        if self.canonical_root is None:
            return
        root = self.documents.get(self.canonical_root)
        if root is None:
            return
        ranked_links: list[tuple[float, DocumentLink, TopicName | None]] = []
        for link in root.links:
            url = self._canonical_url(str(link.url))
            existing = self.candidates.get(url)
            classification = classify_source(
                identity=self.site_identity,
                url=url,
                title=(existing.title if existing is not None else link.text),
                text=(existing.snippet if existing is not None else ""),
                linked_from_canonical_root=True,
            )
            topic = self._infer_topic(f"{link.text} {urlparse(url).path}")
            candidate = CandidateSource(
                url=url,
                title=(existing.title if existing is not None else link.text),
                snippet=(
                    existing.snippet
                    if existing is not None
                    else "Linked from canonical target-site root."
                ),
                classification=classification,
                fetched=url in self.documents,
                rejection_reason=self._rejection_reason(classification.site_scope),
            )
            self._upsert_candidate(candidate)
            if classification.site_scope != "target_site" or topic is None:
                continue
            rank = classification.score + 5 * topic_matches(
                f"{link.text} {urlparse(url).path}", topic
            )
            ranked_links.append((rank, link, topic))

        ranked_links.sort(key=lambda item: item[0], reverse=True)
        per_topic: dict[TopicName, int] = {
            "submission_policy": 0,
            "networking_policy": 0,
        }
        for _rank, link, topic in ranked_links:
            if topic is None or per_topic[topic] >= 2:
                continue
            url = self._canonical_url(str(link.url))
            if url in self.documents or self._pages_used >= self.page_budget:
                continue
            self.links_followed.append(url)
            self._emit(
                "link_followed",
                {"from": self.canonical_root, "to": url, "topic": topic},
            )
            result = self._fetch_page({"url": url})
            if result.get("ok"):
                per_topic[topic] += 1

    def _fetch_topic_candidates(self, topic: TopicName, *, limit: int) -> None:
        """Fetch several strong pages, continuing past topic-mismatched results."""

        candidates = [
            item
            for item in self.candidates.values()
            if item.classification.site_scope == "target_site"
            and topic_matches(
                f"{item.title} {item.snippet} {urlparse(str(item.url)).path}",
                topic,
            )
            and str(item.url) not in self.documents
            and str(item.url) not in self.rejected_documents
            and str(item.url) not in self._attempted_urls
        ]
        candidates.sort(
            key=lambda item: self._candidate_rank(item, topic), reverse=True
        )
        supported = 0
        attempts = 0
        maximum_attempts = max(limit * 2, limit)
        for candidate in candidates:
            if (
                supported >= limit
                or attempts >= maximum_attempts
                or self._pages_used >= self.page_budget
            ):
                break
            attempts += 1
            result = self._fetch_page({"url": str(candidate.url)})
            if not result.get("ok"):
                continue
            document = self.documents.get(str(result.get("url")))
            if document is not None and self._document_supports_topic(document, topic):
                supported += 1
            else:
                self._emit(
                    "topic_mismatch",
                    {
                        "requested_url": str(candidate.url),
                        "final_url": result.get("url"),
                        "topic": topic,
                    },
                )

    def _fetch_policy_candidates(self, *, limit: int) -> None:
        candidates = [
            item
            for item in self.candidates.values()
            if item.classification.site_scope == "target_site"
            and policy_matches(
                f"{item.title} {item.snippet} {urlparse(str(item.url)).path}"
            )
            and str(item.url) not in self.documents
            and str(item.url) not in self.rejected_documents
            and str(item.url) not in self._attempted_urls
        ]
        candidates.sort(
            key=lambda item: self._candidate_rank(item, "operational_policy"),
            reverse=True,
        )
        fetched = 0
        for candidate in candidates:
            if fetched >= limit or self._pages_used >= self.page_budget:
                break
            result = self._fetch_page({"url": str(candidate.url)})
            if result.get("ok"):
                fetched += 1

    def _fetch_sibling_control(self) -> None:
        """Retain one sibling page so retrieval exclusion is measurable."""

        if self._pages_used >= self.page_budget:
            return
        candidates = [
            item
            for item in self.candidates.values()
            if item.classification.site_scope == "sibling"
            and str(item.url) not in self.rejected_documents
            and topic_matches(
                f"{item.title} {item.snippet} {urlparse(str(item.url)).path}",
                "submission_policy",
            )
        ]
        if not candidates:
            return
        candidate = max(
            candidates,
            key=lambda item: self._candidate_rank(item, "submission_policy"),
        )
        self._emit("sibling_control_fetch", {"url": str(candidate.url)})
        try:
            self._fetch_page({"url": str(candidate.url)})
        except KeyError:
            # Small mocked backends may intentionally omit negative-control pages.
            self._emit(
                "page_fetch_failed",
                {"url": str(candidate.url), "error": "mock page unavailable"},
            )

    def _record_page_coverage(self, document: FetchedDocument) -> None:
        if document.classification.site_scope != "target_site":
            return
        searchable = (
            f"{document.title}\n{urlparse(str(document.url)).path}\n"
            f"{document.relevant_text}"
        ).lower()
        for topic in ("submission_policy", "networking_policy"):
            if topic_matches(searchable, topic):
                self._topic_pages[topic].add(str(document.url))

        if self._document_supports_topic(document, "submission_policy"):
            self._topic_evidence["submission_policy"].add(str(document.url))

        if self._document_supports_topic(document, "networking_policy"):
            self._topic_evidence["networking_policy"].add(str(document.url))

    @staticmethod
    def _document_supports_topic(
        document: FetchedDocument,
        topic: TopicName,
    ) -> bool:
        searchable = f"{document.title}\n{document.text}".lower()
        if topic == "submission_policy":
            signals = (
                "#sbatch",
                " sbatch ",
                "--partition",
                "mandatory sbatch",
                "submit work to a slurm",
                "job submission script",
                "queues (partitions)",
            )
            return any(signal in searchable for signal in signals)
        signals = (
            "firewall",
            "tcp port",
            "port range",
            "outbound network",
            "network egress",
            "socket policy",
            "compute-to-compute",
            "compute to compute",
        )
        return any(signal in searchable for signal in signals)

    def _topic_coverage(self, topic: TopicName) -> TopicCoverage:
        pages = self._topic_pages[topic]
        evidence = self._topic_evidence[topic]
        queries = self._topic_queries[topic]
        notes: list[str] = []
        if evidence:
            status = "evidence_found"
            notes.append("Target-site topic evidence was found.")
        elif self.canonical_root and len(pages) >= 2 and queries >= 2:
            status = "documentation_silent"
            notes.append(
                "Multiple target-site pages and focused searches contained no explicit policy."
            )
        elif queries >= 2 and not self.canonical_root:
            status = "search_exhausted"
            notes.append(
                "Focused searches were exhausted without adequate target-site pages."
            )
        elif queries >= 2 and self._searches_used >= self.search_budget:
            status = "search_exhausted"
            notes.append("The search budget ended before adequate topic coverage.")
        else:
            status = "not_investigated"
            notes.append("Topic coverage is not yet sufficient to finish discovery.")
        return TopicCoverage(
            topic=topic,
            status=status,
            target_site_pages_examined=len(pages),
            queries_attempted=queries,
            evidence_urls=sorted(evidence),
            notes=notes,
        )

    def _candidate_rank(
        self,
        candidate: CandidateSource,
        objective: str | TopicName | None,
    ) -> float:
        value = f"{candidate.title} {urlparse(str(candidate.url)).path}"
        rank = candidate.classification.score
        if objective == "canonical_root":
            rank += 4 * root_matches(value)
        elif objective in {"submission_policy", "networking_policy"}:
            rank += 5 * topic_matches(value, objective)
            rank += 0.5 * topic_matches(candidate.snippet, objective)
        elif objective == "operational_policy":
            rank += 5 * policy_matches(value)
            rank += 0.5 * policy_matches(candidate.snippet)
        return rank

    def _ranked_candidates(self, *, eligible_only: bool) -> list[CandidateSource]:
        values = list(self.candidates.values())
        if eligible_only:
            values = [
                item
                for item in values
                if item.classification.site_scope
                in {"target_site", "organization_general"}
            ]
        return sorted(
            values,
            key=lambda item: item.classification.score,
            reverse=True,
        )

    def _upsert_candidate(self, candidate: CandidateSource) -> None:
        url = self._canonical_url(str(candidate.url))
        existing = self.candidates.get(url)
        if existing is None or candidate.classification.score >= existing.classification.score:
            if existing is not None:
                candidate.fetched = candidate.fetched or existing.fetched
                candidate.selected = candidate.selected or existing.selected
            self.candidates[url] = candidate

    def _document_tool_output(
        self,
        document: FetchedDocument,
        *,
        cached: bool,
    ) -> dict[str, Any]:
        ranked_links = sorted(
            document.links,
            key=lambda link: max(
                topic_matches(f"{link.text} {link.url}", "submission_policy"),
                topic_matches(f"{link.text} {link.url}", "networking_policy"),
            ),
            reverse=True,
        )
        ranked_sections = sorted(
            document.sections,
            key=lambda section: sum(
                topic_matches(f"{section.heading} {section.text}", topic)
                for topic in self._topic_pages
            ),
            reverse=True,
        )
        return {
            "ok": True,
            "url": str(document.url),
            "title": document.title,
            "site_scope": document.classification.site_scope,
            "score": document.classification.score,
            "reasons": document.classification.reasons,
            "relevant_text": document.relevant_text,
            "relevant_sections": [
                section.model_dump(mode="json") for section in ranked_sections[:6]
            ],
            "links": [link.model_dump(mode="json") for link in ranked_links[:30]],
            "text_truncated": document.text_truncated,
            "cached": cached,
        }

    def _document_context(self, document: FetchedDocument) -> dict[str, Any]:
        return {
            "url": str(document.url),
            "title": document.title,
            "scope": document.classification.site_scope,
            "score": document.classification.score,
            "relevant_text": document.relevant_text[:4000],
            "text_truncated": document.text_truncated,
        }

    def _coverage_support_urls(self, coverage: DiscoveryCoverage) -> list[str]:
        urls: list[str] = []
        for item in (coverage.submission_policy, coverage.networking_policy):
            urls.extend(str(url) for url in item.evidence_urls)
            if item.status == "documentation_silent":
                urls.extend(sorted(self._topic_pages[item.topic])[:2])
        return [url for url in dict.fromkeys(urls) if url in self.documents]

    def _relevant_text(
        self,
        sections: list[DocumentSection],
        full_text: str,
    ) -> str:
        ranked: list[tuple[int, DocumentSection]] = []
        for section in sections:
            value = f"{section.heading} {section.text}"
            score = sum(topic_matches(value, topic) for topic in self._topic_pages)
            if score:
                ranked.append((score, section))
        ranked.sort(key=lambda item: item[0], reverse=True)
        if not ranked:
            return full_text[: min(self.max_page_chars, 6000)]
        chunks: list[str] = []
        length = 0
        for _score, section in ranked:
            chunk = f"## {section.heading}\n{section.text}".strip()
            if chunk in chunks:
                continue
            chunks.append(chunk)
            length += len(chunk)
            if length >= 8000 or len(chunks) >= 8:
                break
        return "\n\n".join(chunks)[:8000]

    def _extract_sections(
        self,
        soup: BeautifulSoup,
        base_url: str,
    ) -> list[DocumentSection]:
        content = soup.find("article") or soup.find("main") or soup.body or soup
        elements = content.find_all(
            ["h1", "h2", "h3", "h4", "p", "pre", "li", "table"]
        )
        sections: list[DocumentSection] = []
        heading = "Overview"
        heading_path = [heading]
        heading_stack: list[str] = []
        blocks: list[DocumentBlock] = []
        links: list[str] = []

        def flush() -> None:
            unique_blocks: list[DocumentBlock] = []
            seen_blocks: set[tuple[str, str]] = set()
            for block in blocks:
                key = (block.kind, block.text)
                if block.text and key not in seen_blocks:
                    seen_blocks.add(key)
                    unique_blocks.append(block)
            text = "\n\n".join(block.text for block in unique_blocks).strip()
            if text:
                sections.append(
                    DocumentSection(
                        heading=heading[:500],
                        heading_path=[item[:500] for item in heading_path],
                        text=text,
                        links=list(dict.fromkeys(links))[:50],
                        blocks=unique_blocks,
                    )
                )

        for element in elements:
            if element.name in {"h1", "h2", "h3", "h4"}:
                flush()
                heading = element.get_text(" ", strip=True) or "Untitled section"
                level = int(element.name[1])
                heading_stack = heading_stack[: level - 1]
                while len(heading_stack) < level - 1:
                    heading_stack.append("Untitled section")
                heading_stack.append(heading)
                heading_path = list(heading_stack)
                blocks = []
                links = []
                continue
            if element.name != "table" and element.find_parent("table") is not None:
                continue
            if element.name != "pre" and element.find_parent("pre") is not None:
                continue
            if element.name == "table":
                text = self._table_to_markdown(element)
                kind = "table"
            elif element.name == "pre":
                text = element.get_text("\n", strip=True)
                kind = "code"
            else:
                text = element.get_text(" ", strip=True)
                kind = "text"
            if text:
                blocks.append(DocumentBlock(kind=kind, text=text))
            for anchor in element.find_all("a", href=True):
                try:
                    links.append(self._validate_url(urljoin(base_url, anchor["href"])))
                except ValueError:
                    continue
        flush()
        return sections

    @staticmethod
    def _table_to_markdown(table: Any) -> str:
        rows: list[list[str]] = []
        for row in table.find_all("tr"):
            cells = [
                cell.get_text(" ", strip=True).replace("|", "\\|")
                for cell in row.find_all(["th", "td"], recursive=False)
            ]
            if cells:
                rows.append(cells)
        if not rows:
            return ""
        width = max(len(row) for row in rows)
        normalized = [row + [""] * (width - len(row)) for row in rows]
        header = normalized[0]
        body = normalized[1:]
        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join("---" for _ in range(width)) + " |",
        ]
        lines.extend("| " + " | ".join(row) + " |" for row in body)
        return "\n".join(lines)

    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> list[DocumentLink]:
        links: list[DocumentLink] = []
        seen: set[str] = set()
        for anchor in soup.find_all("a", href=True):
            candidate = urljoin(base_url, anchor["href"])
            try:
                candidate = self._validate_url(candidate)
            except ValueError:
                continue
            if candidate in seen:
                continue
            seen.add(candidate)
            links.append(
                DocumentLink(
                    url=candidate,
                    text=anchor.get_text(" ", strip=True)[:500],
                )
            )
        return links

    def _organization_page_applies(self, document: FetchedDocument) -> bool:
        text = f"{document.title}\n{document.relevant_text}".lower()
        organization_patterns = (
            r"\bapplies\s+to\s+all\s+clusters\b",
            r"\bapplies\s+to\s+all\s+systems\b",
            r"\bfor\s+all\s+clusters\b",
            r"\bfor\s+all\s+systems\b",
            r"\borganization-wide\s+policy\b",
        )
        if any(self._has_non_negated_match(text, pattern) for pattern in organization_patterns):
            return True

        for alias in self.site_identity.aliases:
            escaped = re.escape(alias.lower())
            explicit_patterns = (
                rf"\bapplies\s+to\s+{escaped}\b",
                rf"\bfor\s+{escaped}\s+(?:users?|jobs?|systems?|clusters?)\b",
                rf"\b{escaped}\s+(?:users?|jobs?|systems?|clusters?)\s+(?:must|should|may|are)\b",
            )
            if any(
                self._has_non_negated_match(text, pattern)
                for pattern in explicit_patterns
            ):
                return True
        return False

    @staticmethod
    def _has_non_negated_match(text: str, pattern: str) -> bool:
        for match in re.finditer(pattern, text):
            prefix = text[max(0, match.start() - 60) : match.start()]
            if not re.search(r"\b(?:not|never|doesn['’]?t|does\s+not|no)\b", prefix):
                return True
        return False

    def _termination_reason(self, coverage: DiscoveryCoverage) -> str:
        statuses = {
            coverage.submission_policy.status,
            coverage.networking_policy.status,
        }
        if "search_exhausted" in statuses:
            return "search_budget_exhausted_with_discovery_failure"
        if "documentation_silent" in statuses:
            return "target_site_search_complete_with_documentation_silence"
        return "target_site_evidence_found_for_both_topics"

    @staticmethod
    def _rejection_reason(scope: str) -> str | None:
        if scope == "sibling":
            return "Sibling or other-site documentation cannot establish target policy."
        if scope == "unrelated":
            return "No target-site or organization-wide policy evidence."
        return None

    @staticmethod
    def _infer_topic(value: str) -> TopicName | None:
        submission = topic_matches(value, "submission_policy")
        networking = topic_matches(value, "networking_policy")
        if submission == networking == 0:
            return None
        return "submission_policy" if submission >= networking else "networking_policy"

    def _emit(self, event: str, payload: dict[str, Any]) -> None:
        if self._event_callback is not None:
            self._event_callback(event, payload)

    def _ddgs_search(self, query: str, max_results: int) -> list[dict[str, str]]:
        return list(DDGS(timeout=10).text(query, max_results=max_results) or [])

    @staticmethod
    def _http_fetch(url: str) -> dict[str, Any]:
        headers = {
            "User-Agent": (
                "HPCPolicyScout/0.2 (+research documentation discovery; "
                "contact the repository owner for details)"
            )
        }
        with httpx.Client(timeout=20, follow_redirects=True, headers=headers) as client:
            response = client.get(url)
            response.raise_for_status()
        return {
            "url": str(response.url),
            "content_type": response.headers.get("content-type", ""),
            "text": response.text,
            "content": response.content,
        }

    def _extract_pdf(self, content: bytes) -> str:
        reader = PdfReader(io.BytesIO(content))
        pages = []
        for page in reader.pages[:100]:
            pages.append(page.extract_text() or "")
            if sum(len(x) for x in pages) >= self.max_page_chars:
                break
        return "\n\n".join(pages)

    @staticmethod
    def _fallback_text(soup: BeautifulSoup) -> str:
        for tag in soup(["script", "style", "noscript", "svg", "nav", "footer"]):
            tag.decompose()
        return soup.get_text("\n", strip=True)

    def _validate_url(self, raw_url: str) -> str:
        url = self._canonical_url(raw_url)
        parsed = urlparse(url)
        if parsed.scheme != "https":
            raise ValueError("Only HTTPS URLs are allowed.")
        if parsed.port not in {None, 443}:
            raise ValueError("Non-standard URL ports are not allowed.")
        hostname = (parsed.hostname or "").lower().rstrip(".")
        if not hostname:
            raise ValueError("URL has no hostname.")
        try:
            ipaddress.ip_address(hostname)
        except ValueError:
            pass
        else:
            raise ValueError("Literal IP addresses are not allowed.")
        if not any(
            hostname == domain or hostname.endswith("." + domain)
            for domain in self.allowed_domains
        ):
            raise ValueError(f"Domain is not approved: {hostname}")
        return url

    @staticmethod
    def _canonical_url(raw_url: str) -> str:
        clean, _fragment = urldefrag(raw_url.strip())
        return clean

    @staticmethod
    def _normalize_domain(domain: str) -> str:
        parsed = urlparse(domain if "://" in domain else f"https://{domain}")
        hostname = (parsed.hostname or "").lower().strip().lstrip(".").rstrip(".")
        if not hostname:
            raise ValueError(f"Invalid approved domain: {domain}")
        return hostname
