"""Build exact, non-overlapping evidence spans for grouped extraction."""

from __future__ import annotations

from dataclasses import dataclass

from schemas import CorpusChunk, DiscoveryCoverage, FieldRetrieval


@dataclass(frozen=True)
class GroundedReference:
    field: str
    evidence_ref: str
    chunk: CorpusChunk
    quote: str


@dataclass(frozen=True)
class GroundedExtractionContext:
    prompt: str
    references: dict[str, dict[str, GroundedReference]]
    unique_chunks: int
    field_references: int


def build_grounded_context(
    *,
    site_name: str,
    group_name: str,
    fields: tuple[str, ...],
    retrievals: dict[str, FieldRetrieval],
    discovery_summary: str,
    unanswered_topics: list[str],
    discovery_coverage: DiscoveryCoverage,
) -> GroundedExtractionContext:
    chunks: dict[str, CorpusChunk] = {}
    spans_by_chunk: dict[str, list[str]] = {}
    references: dict[str, dict[str, GroundedReference]] = {}
    maps: list[str] = []
    field_reference_count = 0

    for field in fields:
        retrieval = retrievals[field]
        references[field] = {}
        lines = [f"--- FIELD {field} EVIDENCE START ---", f"QUERY: {retrieval.query}"]
        if not retrieval.hits:
            lines.append("NO EVIDENCE SPANS RETRIEVED")
        counter = 0
        for hit in retrieval.hits:
            chunk = hit.chunk
            chunks.setdefault(chunk.chunk_id, chunk)
            spans = spans_by_chunk.setdefault(chunk.chunk_id, _literal_spans(chunk.text))
            global_chunk = list(chunks).index(chunk.chunk_id) + 1
            for span_index, quote in enumerate(spans, start=1):
                counter += 1
                evidence_ref = f"{field}:E{counter}"
                reference = GroundedReference(
                    field=field,
                    evidence_ref=evidence_ref,
                    chunk=chunk,
                    quote=quote,
                )
                references[field][evidence_ref] = reference
                lines.append(
                    f"EVIDENCE REF: {evidence_ref} -> G{global_chunk}:S{span_index} "
                    f"(retrieval score {hit.score})"
                )
                field_reference_count += 1
        lines.append(f"--- FIELD {field} EVIDENCE END ---")
        maps.append("\n".join(lines))

    library: list[str] = ["--- EXACT EVIDENCE SPAN LIBRARY START ---"]
    for global_index, chunk in enumerate(chunks.values(), start=1):
        library.extend(
            [
                f"--- GLOBAL CHUNK G{global_index} START ---",
                f"URL: {chunk.source_url}",
                f"TITLE: {chunk.title}",
                f"HEADING PATH: {' > '.join(chunk.heading_path)}",
                f"SITE SCOPE: {chunk.site_scope}",
            ]
        )
        for span_index, quote in enumerate(spans_by_chunk[chunk.chunk_id], start=1):
            library.extend(
                [
                    f"[G{global_index}:S{span_index} START]",
                    quote,
                    f"[G{global_index}:S{span_index} END]",
                ]
            )
        library.append(f"--- GLOBAL CHUNK G{global_index} END ---")
    library.append("--- EXACT EVIDENCE SPAN LIBRARY END ---")

    prompt = "\n\n".join(
        [
            f"SITE: {site_name}",
            f"EXTRACTION GROUP: {group_name}",
            "RETURN ONLY THESE FIELDS: " + ", ".join(fields),
            f"DISCOVERY SUMMARY: {discovery_summary}",
            "DETERMINISTIC DISCOVERY COVERAGE:\n"
            + discovery_coverage.model_dump_json(indent=2),
            "DISCOVERY UNANSWERED TOPICS:\n"
            + ("\n".join(f"- {item}" for item in unanswered_topics) or "- none"),
            *maps,
            "\n".join(library),
        ]
    )
    return GroundedExtractionContext(
        prompt=prompt,
        references=references,
        unique_chunks=len(chunks),
        field_references=field_reference_count,
    )


def _literal_spans(text: str, *, maximum_chars: int = 1000) -> list[str]:
    """Return exact substrings with no overlap; tables remain row-addressable."""

    candidates = [part.strip() for part in text.split("\n\n") if part.strip()]
    if not candidates:
        candidates = [text.strip()] if text.strip() else []
    spans: list[str] = []
    for candidate in candidates:
        remaining = candidate
        while len(remaining) > maximum_chars:
            split_at = remaining.rfind("\n", 0, maximum_chars + 1)
            if split_at < maximum_chars // 2:
                split_at = remaining.rfind(" ", 0, maximum_chars + 1)
            if split_at < maximum_chars // 2:
                split_at = maximum_chars
            span = remaining[:split_at].strip()
            if span:
                spans.append(span)
            remaining = remaining[split_at:].strip()
        if remaining:
            spans.append(remaining)
    return spans
