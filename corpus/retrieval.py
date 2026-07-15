"""Transient scope-filtered BM25 retrieval over persisted corpus chunks."""

from __future__ import annotations

import math
import re
from collections import Counter

from schemas import CorpusChunk, FieldRetrieval, RetrievalHit
from schemas.discovery import SiteScope


FIELD_QUERIES: dict[str, str] = {
    "scheduler": "scheduler workload manager Slurm",
    "submit_command": "submit command sbatch job script",
    "submission_options": "mandatory minimum SBATCH fields options account partition nodes tasks CPUs memory time job name",
    "account_allocation_policy": "allocation account project credits service units charging",
    "default_partition": "default partition queue",
    "partitions": "partition queue limits maximum nodes walltime shared whole node GPU AI debug",
    "walltime_policy": "walltime time limit maximum duration",
    "memory_policy": "memory request mem limit per node",
    "job_size_policy": "job size node core task maximum minimum limit",
    "charging_model": "charging accounting service units credits allocation cost",
    "purge_policy": "purge retention deletion expiration files scratch policy",
    "cost_traps": "cost warning charge multiplier waste credits policy FAQ",
    "manager_worker_connectivity": "manager worker compute login connectivity TCP firewall",
    "worker_worker_connectivity": "worker compute node to node connectivity TCP firewall",
    "published_port_range": "published TCP UDP port range firewall application ports",
    "manager_address_guidance": "manager address hostname interface bind login compute",
    "login_node_socket_policy": "login node socket service listening policy",
    "outbound_compute_network": "outbound compute node internet network egress",
}

WHOLE_SECTION_FIELDS = frozenset({"charging_model", "purge_policy", "cost_traps"})


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
        eligible = [chunk for chunk in self.chunks if chunk.site_scope in allowed_scopes]
        excluded: dict[SiteScope, int] = {}
        for chunk in self.chunks:
            if chunk.site_scope not in allowed_scopes:
                excluded[chunk.site_scope] = excluded.get(chunk.site_scope, 0) + 1
        scores = _bm25(query, eligible)
        ranked: list[tuple[float, CorpusChunk]] = []
        for chunk, score in zip(eligible, scores):
            heading = " ".join(chunk.heading_path).lower()
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

