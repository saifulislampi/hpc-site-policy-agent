"""Transient scope-filtered BM25 retrieval over persisted corpus chunks."""

from __future__ import annotations

import math
import re
from collections import Counter

from schemas import CorpusChunk, FieldRetrieval, RetrievalHit
from schemas.discovery import SiteScope


FIELD_QUERY_VARIANTS: dict[str, list[str]] = {
    "scheduler": ["scheduler workload manager Slurm", "batch scheduler"],
    "submit_command": ["submit command sbatch job script", "submitting a job"],
    "submission_options": [
        "mandatory minimum SBATCH fields account partition",
        "job script nodes tasks CPUs memory time job name",
        "#SBATCH --account --partition --time --mem",
    ],
    "account_allocation_policy": [
        "allocation account project credits service units",
        "SBATCH account required mybalance",
    ],
    "default_partition": ["default partition queue", "no partition specified"],
    "partitions": [
        "Slurm partitions queues limits walltime",
        "shared wholenode wide highmem gpu ai debug queue",
        "partition maximum nodes maximum duration charging factor",
    ],
    "walltime_policy": ["walltime time limit maximum duration", "maximum job time"],
    "memory_policy": ["memory request --mem limit per node", "memory per core"],
    "job_size_policy": ["job size maximum nodes cores tasks", "maximum job size"],
    "charging_model": [
        "charging accounting service units credits allocation cost",
        "SU charge cores memory node exclusive",
    ],
    "purge_policy": [
        "purge retention deletion expiration files scratch policy",
        "scratch files removed days",
    ],
    "cost_traps": [
        "cost warning charge multiplier waste credits policy FAQ",
        "unexpected SU charges memory whole node highmem",
    ],
    "manager_worker_connectivity": [
        "manager worker compute login connectivity TCP firewall",
        "compute node connect login node",
    ],
    "worker_worker_connectivity": [
        "worker compute node to node connectivity TCP firewall",
        "inter-node application network policy",
    ],
    "published_port_range": [
        "published TCP UDP port range firewall application ports",
        "allowed listening ports",
    ],
    "manager_address_guidance": [
        "manager address hostname interface bind login compute",
        "head node address hostname",
    ],
    "login_node_socket_policy": [
        "login node socket service listening policy",
        "services on login nodes prohibited",
    ],
    "outbound_compute_network": [
        "outbound compute node internet network egress",
        "compute nodes access external network",
    ],
}

FIELD_QUERIES = {
    field: " | ".join(variants)
    for field, variants in FIELD_QUERY_VARIANTS.items()
}

WHOLE_SECTION_FIELDS = frozenset({"charging_model", "purge_policy", "cost_traps"})
NETWORK_FIELDS = frozenset(
    {
        "manager_worker_connectivity",
        "worker_worker_connectivity",
        "published_port_range",
        "manager_address_guidance",
        "login_node_socket_policy",
        "outbound_compute_network",
    }
)


class LexicalRetriever:
    def __init__(self, chunks: list[CorpusChunk]) -> None:
        self.chunks = chunks

    def retrieve_all(
        self,
        *,
        allowed_scopes: set[SiteScope],
        top_k: int,
    ) -> dict[str, FieldRetrieval]:
        return {
            field: self.retrieve(
                field=field,
                query=query,
                allowed_scopes=allowed_scopes,
                top_k=top_k,
            )
            for field, query in FIELD_QUERIES.items()
        }

    def retrieve(
        self,
        *,
        field: str,
        query: str,
        allowed_scopes: set[SiteScope],
        top_k: int,
    ) -> FieldRetrieval:
        scoped = [chunk for chunk in self.chunks if chunk.site_scope in allowed_scopes]
        eligible = _deduplicate_content(scoped)
        excluded: dict[SiteScope, int] = {}
        for chunk in self.chunks:
            if chunk.site_scope not in allowed_scopes:
                excluded[chunk.site_scope] = excluded.get(chunk.site_scope, 0) + 1
        variants = FIELD_QUERY_VARIANTS.get(field, [query])
        variant_scores = [_bm25(variant, eligible) for variant in variants]
        scores = [
            max(values) + 0.25 * sum(values)
            for values in zip(*variant_scores)
        ] if eligible else []
        ranked: list[tuple[float, CorpusChunk]] = []
        for chunk, score in zip(eligible, scores):
            heading = " ".join(chunk.heading_path).lower()
            searchable = f"{heading}\n{chunk.text[:1000]}".lower()
            if _looks_like_navigation(chunk):
                score *= 0.05
            if field in {"partitions", "default_partition"}:
                if not any(term in searchable for term in ("partition", "queue")):
                    continue
            if field == "scheduler" and not any(
                term in searchable for term in ("slurm", "scheduler")
            ):
                continue
            if field == "submit_command" and "sbatch" not in searchable:
                continue
            if field == "submission_options" and not any(
                term in searchable
                for term in ("#sbatch", "--account", "--partition", " sbatch ")
            ):
                continue
            if field in NETWORK_FIELDS and chunk.block_type == "table":
                if field != "published_port_range" or "port" not in searchable:
                    continue
            if field in WHOLE_SECTION_FIELDS and chunk.block_type == "section":
                if any(term in heading for term in ("polic", "faq", "frequently")):
                    score = max(score, 0.01)
            if field == "submission_options" and (
                "#SBATCH" in chunk.text
                or "mandatory sbatch" in chunk.text.lower()
                or "job submission" in heading
            ):
                score += 2.0
            if score > 0:
                ranked.append((score, chunk))
        ranked.sort(key=lambda item: (item[0], item[1].chunk_id), reverse=True)
        limit = max(top_k, 8) if field in WHOLE_SECTION_FIELDS else top_k
        return FieldRetrieval(
            field=field,
            query=query,
            allowed_scopes=sorted(allowed_scopes),
            hits=[
                RetrievalHit(chunk=chunk, score=round(score, 6))
                for score, chunk in ranked[:limit]
            ],
            excluded_scope_counts=excluded,
        )


def _bm25(query: str, chunks: list[CorpusChunk]) -> list[float]:
    if not chunks:
        return []
    tokenized = [
        _tokens(" ".join([chunk.title, *chunk.heading_path, chunk.text]))
        for chunk in chunks
    ]
    query_terms = set(_tokens(query))
    document_frequency = {
        term: sum(term in set(tokens) for tokens in tokenized) for term in query_terms
    }
    average_length = sum(len(tokens) for tokens in tokenized) / len(tokenized)
    k1 = 1.5
    b = 0.75
    scores: list[float] = []
    for tokens in tokenized:
        counts = Counter(tokens)
        length = len(tokens)
        score = 0.0
        for term in query_terms:
            frequency = counts[term]
            if not frequency:
                continue
            df = document_frequency[term]
            inverse_frequency = math.log(1 + (len(chunks) - df + 0.5) / (df + 0.5))
            denominator = frequency + k1 * (
                1 - b + b * length / max(average_length, 1)
            )
            score += inverse_frequency * frequency * (k1 + 1) / denominator
        scores.append(score)
    return scores


def _tokens(value: str) -> list[str]:
    return re.findall(r"--?[a-z][a-z0-9-]*|[a-z0-9]+", value.lower())


def _deduplicate_content(chunks: list[CorpusChunk]) -> list[CorpusChunk]:
    """Keep one provenance-bearing copy of byte-identical chunk text."""

    result: list[CorpusChunk] = []
    seen: set[str] = set()
    for chunk in sorted(
        chunks,
        key=lambda item: (
            item.site_scope != "target_site",
            str(item.source_url),
            item.chunk_id,
        ),
    ):
        if chunk.content_hash in seen:
            continue
        seen.add(chunk.content_hash)
        result.append(chunk)
    return result


def _looks_like_navigation(chunk: CorpusChunk) -> bool:
    text = chunk.text.lower()
    signals = (
        "rcac resources rcac resources",
        "rcac blogs rcac blogs",
        "archive archive",
        "categories categories",
        "week 1 week 1",
        "all software audio/visualization",
    )
    return sum(signal in text for signal in signals) >= 2
