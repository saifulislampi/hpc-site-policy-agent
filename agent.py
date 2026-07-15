"""The bounded discovery loop and deterministic extraction orchestration."""

from __future__ import annotations

import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from models import AgentError, ModelTurn, ToolResult
from prompts import (
    DISCOVERY_SYSTEM_PROMPT,
    EXTRACTION_SYSTEM_PROMPT,
    build_discovery_goal,
    build_extraction_input,
)
from providers.base import BaseProvider
from reporting import RunArtifacts, build_run_artifacts
from corpus import CorpusStore, LexicalRetriever, build_corpus_records
from schemas import (
    ChunkerConfiguration,
    DiscoverySelection,
    ExtractedPolicy,
    FieldRetrieval,
)
from tools import ScoutTools


class JsonlRunLogger:
    def __init__(self, *, log_dir: Path, run_id: str) -> None:
        log_dir.mkdir(parents=True, exist_ok=True)
        self.path = log_dir / f"{run_id}.jsonl"

    def write(self, event: str, payload: dict[str, Any]) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **payload,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class HPCPolicyScoutAgent:
    def __init__(
        self,
        *,
        provider: BaseProvider,
        tools: ScoutTools,
        max_steps: int = 10,
        log_dir: str | Path = "logs",
        corpus_dir: str | Path = "corpora/default",
        refresh_corpus: bool = False,
        chunk_chars: int = 1800,
        retrieval_top_k: int = 5,
    ) -> None:
        self.provider = provider
        self.tools = tools
        self.max_steps = max_steps
        self.corpus_store = CorpusStore(corpus_dir)
        self.refresh_corpus = refresh_corpus
        self.chunk_chars = chunk_chars
        self.retrieval_top_k = retrieval_top_k
        self.run_id = str(uuid.uuid4())
        self.logger = JsonlRunLogger(log_dir=Path(log_dir), run_id=self.run_id)
        self._model_requests = 0
        self._discovery_input_tokens = 0
        self._discovery_output_tokens = 0
        self._started_at = time.monotonic()
        if hasattr(self.tools, "set_event_callback"):
            self.tools.set_event_callback(self._record_tool_event)

    def run(
        self,
        *,
        site_name: str,
        site_id: str,
        discovery_report_reference: str,
        keywords: list[str],
        allowed_domains: list[str],
    ) -> RunArtifacts:
        run_timestamp = datetime.now().astimezone()
        self.logger.write(
            "run_started",
            {
                "run_id": self.run_id,
                "site_name": site_name,
                "keywords": keywords,
                "allowed_domains": allowed_domains,
            },
        )
        self._progress(f"Starting discovery for {site_name}")

        discovery_context = None
        if hasattr(self.tools, "bootstrap_discovery"):
            self._progress("Locating and crawling target-site documentation")
            discovery_context = self.tools.bootstrap_discovery(keywords=keywords)

        selection = self._run_discovery(
            site_name=site_name,
            keywords=keywords,
            allowed_domains=allowed_domains,
            discovery_context=discovery_context,
        )
        selected_documents = self.tools.selected_documents(selection)

        self.logger.write(
            "discovery_complete",
            {
                "selection": selection.model_dump(mode="json"),
                "document_count": len(selected_documents),
                "metrics": self._run_metrics(selection=selection),
            },
        )

        self._progress("Chunking and merging discovered web pages into the corpus")
        corpus_documents = (
            self.tools.corpus_documents()
            if hasattr(self.tools, "corpus_documents")
            else selected_documents
        )
        now = datetime.now(timezone.utc)
        stored_documents, stored_chunks = build_corpus_records(
            corpus_documents,
            maximum_chars=self.chunk_chars,
            fetched_at=now,
        )
        snapshot = self.corpus_store.merge(
            corpus_id=site_id,
            site_identity=self.tools.site_identity,
            incoming_documents=stored_documents,
            incoming_chunks=stored_chunks,
            chunker=ChunkerConfiguration(maximum_chars=self.chunk_chars),
            refresh=self.refresh_corpus,
            now=now,
        )
        sibling_chunks = sum(
            chunk.site_scope == "sibling" for chunk in snapshot.chunks
        )
        self._progress(
            f"Corpus ready: {len(snapshot.chunks)} reusable chunks "
            f"({sibling_chunks} sibling chunks retained)"
        )

        retrievals = LexicalRetriever(snapshot.chunks).retrieve_all(
            allowed_scopes={"target_site", "organization_general"},
            top_k=self.retrieval_top_k,
        )
        retrieved_siblings = sum(
            hit.chunk.site_scope == "sibling"
            for retrieval in retrievals.values()
            for hit in retrieval.hits
        )
        self._progress(
            f"Retrieved chunks for {len(retrievals)} fields: "
            f"{sibling_chunks} sibling chunks, {retrieved_siblings} retrieved"
        )
        self.logger.write(
            "retrieval_complete",
            {
                "corpus_id": snapshot.manifest.corpus_id,
                "corpus_fingerprint": snapshot.manifest.corpus_fingerprint,
                "sibling_chunks": sibling_chunks,
                "retrieved_sibling_chunks": retrieved_siblings,
                "fields": {
                    field: retrieval.model_dump(mode="json")
                    for field, retrieval in retrievals.items()
                },
            },
        )

        extraction_input = build_extraction_input(
            site_name=site_name,
            retrievals=retrievals,
            discovery_summary=selection.summary,
            unanswered_topics=selection.unanswered_topics,
            discovery_coverage=selection.coverage,
        )
        self._progress(
            "Extracting policy report from "
            f"{sum(len(item.hits) for item in retrievals.values())} "
            "field-ranked chunks"
        )
        report = self.provider.extract_report(
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
            user_prompt=extraction_input,
        )
        report.discovery_coverage = selection.coverage
        citation_audit = self._validate_report_evidence(report, retrievals)
        self.logger.write("retrieval_citation_audit", citation_audit)
        self._progress("Structured extraction completed")

        metrics = self._run_metrics(selection=selection, extraction_complete=True)
        artifacts = build_run_artifacts(
            extracted=report,
            tools=self.tools,
            run_id=self.run_id,
            provider=getattr(self.provider, "provider_name", "unknown"),
            model=getattr(self.provider, "model", "unknown"),
            timestamp=run_timestamp,
            site_id=site_id,
            discovery_report_reference=discovery_report_reference,
            termination_reason=selection.termination_reason,
            metrics=metrics,
            corpus_snapshot=snapshot,
            retrievals=retrievals,
            citation_audit=citation_audit,
            corpus_manifest_reference=str(self.corpus_store.manifest_path),
        )
        self._progress("Built discovery-report and candidate site-policy artifacts")

        self.logger.write(
            "run_complete",
            {
                "run_id": self.run_id,
                "metrics": metrics,
                "discovery_report": artifacts.discovery_report.model_dump(mode="json"),
                "site_policy": artifacts.site_policy.model_dump(mode="json"),
            },
        )
        return artifacts

    def _run_discovery(
        self,
        *,
        site_name: str,
        keywords: list[str],
        allowed_domains: list[str],
        discovery_context: dict[str, Any] | None = None,
    ) -> DiscoverySelection:
        goal = build_discovery_goal(
            site_name=site_name,
            keywords=keywords,
            allowed_domains=allowed_domains,
            site_identity=getattr(self.tools, "site_identity", None),
            discovery_context=discovery_context,
        )
        self._progress("Waiting for the discovery model to select sources")
        turn = self.provider.start_agent(
            system_prompt=DISCOVERY_SYSTEM_PROMPT,
            user_prompt=goal,
            tools=self.tools.definitions(),
        )

        for step in range(1, self.max_steps + 1):
            self._log_turn(step, turn)
            if not turn.tool_calls:
                raise AgentError(
                    "Discovery model returned no tool call. It must finish using "
                    "finish_discovery. Model text: " + (turn.text or "<empty>")
                )

            results: list[ToolResult] = []
            for call in turn.tool_calls:
                result = self.tools.execute(call)
                self.logger.write(
                    "tool_result",
                    {
                        "step": step,
                        "tool": call.name,
                        "arguments": call.arguments,
                        "result_summary": self._summarize_tool_output(result.output),
                        "terminal": result.terminal,
                    },
                )

                if result.terminal:
                    selection_data = result.output.get("selection")
                    if selection_data is None:
                        raise AgentError("Terminal discovery tool returned no selection.")
                    return DiscoverySelection.model_validate(selection_data)

                results.append(result)

            if step == self.max_steps:
                break

            force_tool = (
                "finish_discovery" if step == self.max_steps - 1 else None
            )
            action = "finish discovery" if force_tool else "choose the next action"
            self._progress(f"Waiting for the discovery model to {action}")
            turn = self.provider.continue_agent(
                tool_results=results,
                force_tool=force_tool,
            )

        raise AgentError(f"Discovery exceeded the maximum of {self.max_steps} steps.")

    @staticmethod
    def _validate_report_evidence(
        report: ExtractedPolicy,
        retrievals: dict[str, FieldRetrieval],
    ) -> dict[str, Any]:
        retrieved_urls = {
            str(hit.chunk.source_url)
            for retrieval in retrievals.values()
            for hit in retrieval.hits
        }
        report_urls = {str(source.url) for source in report.sources}
        unexpected_sources = report_urls - retrieved_urls
        if unexpected_sources:
            raise AgentError(
                "Final report cited sources outside the retrieved chunk set: "
                + ", ".join(sorted(unexpected_sources))
            )
        audit: dict[str, Any] = {}
        for policy in (report.slurm_policy, report.network_policy):
            for field, finding in policy.__dict__.items():
                retrieval = retrievals[field]
                hits = {hit.chunk.chunk_id: hit for hit in retrieval.hits}
                cited: set[str] = set()
                for evidence in finding.evidence:
                    hit = hits.get(evidence.chunk_id)
                    if hit is None:
                        raise AgentError(
                            f"Evidence for {field} cites chunk {evidence.chunk_id} "
                            "outside that field's retrieved list."
                        )
                    if str(evidence.source_url) != str(hit.chunk.source_url):
                        raise AgentError(
                            f"Evidence URL for {field} does not match chunk "
                            f"{evidence.chunk_id}."
                        )
                    if evidence.quote not in hit.chunk.text:
                        raise AgentError(
                            f"Evidence quote for {field} is not a literal substring "
                            f"of chunk {evidence.chunk_id}."
                        )
                    cited.add(evidence.chunk_id)
                audit[field] = {
                    "retrieved": [
                        {
                            "chunk_id": hit.chunk.chunk_id,
                            "score": hit.score,
                            "cited": hit.chunk.chunk_id in cited,
                        }
                        for hit in retrieval.hits
                    ],
                    "retrieved_but_uncited": [
                        hit.chunk.chunk_id
                        for hit in retrieval.hits
                        if hit.chunk.chunk_id not in cited
                    ],
                }
        return audit

    def _log_turn(self, step: int, turn: ModelTurn) -> None:
        self._model_requests += 1
        self._discovery_input_tokens += turn.input_tokens or 0
        self._discovery_output_tokens += turn.output_tokens or 0
        self.logger.write(
            "model_turn",
            {
                "step": step,
                "response_id": turn.response_id,
                "text": turn.text,
                "tool_calls": [
                    {"name": call.name, "arguments": call.arguments}
                    for call in turn.tool_calls
                ],
                "input_tokens": turn.input_tokens,
                "output_tokens": turn.output_tokens,
            },
        )
        tool_names = ", ".join(call.name for call in turn.tool_calls) or "no tool"
        self._progress(f"Discovery model turn {step} completed: {tool_names}")

    def _record_tool_event(self, event: str, payload: dict[str, Any]) -> None:
        self.logger.write(event, payload)
        metrics = (
            self.tools.discovery_metrics()
            if hasattr(self.tools, "discovery_metrics")
            else {}
        )
        if event == "query":
            count = metrics.get("search_requests", "?")
            budget = metrics.get("search_budget", "?")
            suffix = (
                f"failed: {payload['error']}"
                if payload.get("error")
                else f"{payload.get('result_count', 0)} results"
            )
            self._progress(
                f"Search {count}/{budget}: {payload.get('query', '')} ({suffix})"
            )
        elif event == "canonical_root":
            self._progress(f"Canonical root: {payload.get('url')}")
        elif event == "link_followed":
            self._progress(f"Following {payload.get('topic')} link: {payload.get('to')}")
        elif event == "page_fetched":
            count = metrics.get("page_requests", "?")
            budget = metrics.get("page_budget", "?")
            self._progress(f"Fetched page {count}/{budget}: {payload.get('url')}")
        elif event == "page_fetch_failed":
            self._progress(f"Skipped failed page: {payload.get('url')}")
        elif event == "sibling_control_fetch":
            self._progress(
                f"Retaining sibling negative-control page: {payload.get('url')}"
            )
        elif event == "coverage":
            submission = payload.get("submission_policy", {}).get("status")
            networking = payload.get("networking_policy", {}).get("status")
            self._progress(
                f"Coverage: submission={submission}, networking={networking}"
            )
        elif event == "selection":
            self._progress(
                f"Selected {len(payload.get('selected_urls', []))} documents; "
                f"termination={payload.get('termination_reason')}"
            )

    def _progress(self, message: str) -> None:
        elapsed = time.monotonic() - self._started_at
        print(f"[{elapsed:6.1f}s] {message}", file=sys.stderr, flush=True)

    def _run_metrics(
        self,
        *,
        selection: DiscoverySelection,
        extraction_complete: bool = False,
    ) -> dict[str, Any]:
        tool_metrics = (
            self.tools.discovery_metrics()
            if hasattr(self.tools, "discovery_metrics")
            else {}
        )
        extraction_usage = (
            getattr(self.provider, "last_extraction_usage", {})
            if extraction_complete
            else {}
        )
        extraction_input = extraction_usage.get("input_tokens") or 0
        extraction_output = extraction_usage.get("output_tokens") or 0
        return {
            **tool_metrics,
            "model_requests": self._model_requests + int(extraction_complete),
            "discovery_input_tokens": self._discovery_input_tokens,
            "discovery_output_tokens": self._discovery_output_tokens,
            "extraction_input_tokens": extraction_input,
            "extraction_output_tokens": extraction_output,
            "total_input_tokens": self._discovery_input_tokens + extraction_input,
            "total_output_tokens": self._discovery_output_tokens + extraction_output,
            "termination_reason": selection.termination_reason,
        }

    @staticmethod
    def _summarize_tool_output(output: dict[str, Any]) -> dict[str, Any]:
        summary = dict(output)
        if isinstance(summary.get("text"), str):
            summary["text"] = summary["text"][:500] + "..."
        if isinstance(summary.get("links"), list):
            summary["links"] = summary["links"][:10]
        return summary
