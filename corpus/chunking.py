"""Deterministic zero-overlap, heading-aware document chunking."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime

from schemas import CorpusChunk, CorpusDocument, FetchedDocument


WHOLE_SECTION_TERMS = frozenset(
    {"policy", "policies", "faq", "faqs", "frequently asked questions"}
)


def build_corpus_records(
    documents: list[FetchedDocument],
    *,
    maximum_chars: int,
    fetched_at: datetime,
) -> tuple[list[CorpusDocument], list[CorpusChunk]]:
    corpus_documents: list[CorpusDocument] = []
    corpus_chunks: list[CorpusChunk] = []
    for document in documents:
        if str(document.url).lower().endswith(".pdf"):
            continue
        document_payload = {
            "url": str(document.url),
            "title": document.title,
            "sections": [section.model_dump(mode="json") for section in document.sections],
        }
        content_hash = _hash_text(
            json.dumps(document_payload, ensure_ascii=False, sort_keys=True)
        )
        document_id = "D_" + _hash_text(str(document.url))[:16]
        corpus_documents.append(
            CorpusDocument(
                document_id=document_id,
                source_url=document.url,
                title=document.title,
                site_scope=document.classification.site_scope,
                trust_level=document.classification.trust_level,
                content_hash=content_hash,
                fetched_at=fetched_at,
                text_truncated=document.text_truncated,
            )
        )
        ordinal = 0
        for section in document.sections:
            heading_path = section.heading_path or [section.heading]
            if _whole_section(heading_path):
                ordinal += 1
                corpus_chunks.append(
                    _chunk(
                        document=document,
                        document_id=document_id,
                        heading_path=heading_path,
                        block_type="section",
                        text=section.text,
                        ordinal=ordinal,
                    )
                )
                continue

            pending_text: list[str] = []

            def flush_pending() -> None:
                nonlocal ordinal
                combined = "\n\n".join(pending_text).strip()
                pending_text.clear()
                for piece in _split_text(combined, maximum_chars=maximum_chars):
                    ordinal += 1
                    corpus_chunks.append(
                        _chunk(
                            document=document,
                            document_id=document_id,
                            heading_path=heading_path,
                            block_type="text",
                            text=piece,
                            ordinal=ordinal,
                        )
                    )

            for block in section.blocks:
                if block.kind == "table":
                    flush_pending()
                    ordinal += 1
                    corpus_chunks.append(
                        _chunk(
                            document=document,
                            document_id=document_id,
                            heading_path=heading_path,
                            block_type="table",
                            text=block.text,
                            ordinal=ordinal,
                        )
                    )
                    continue
                if pending_text and (
                    sum(len(item) for item in pending_text) + len(block.text)
                    > maximum_chars
                ):
                    flush_pending()
                pending_text.append(block.text)
            flush_pending()

        if not document.sections and document.relevant_text:
            for ordinal, piece in enumerate(
                _split_text(document.relevant_text, maximum_chars=maximum_chars),
                start=1,
            ):
                corpus_chunks.append(
                    _chunk(
                        document=document,
                        document_id=document_id,
                        heading_path=[document.title],
                        block_type="text",
                        text=piece,
                        ordinal=ordinal,
                    )
                )
    unique_chunks = {chunk.chunk_id: chunk for chunk in corpus_chunks}
    return corpus_documents, list(unique_chunks.values())


def _chunk(
    *,
    document: FetchedDocument,
    document_id: str,
    heading_path: list[str],
    block_type: str,
    text: str,
    ordinal: int,
) -> CorpusChunk:
    clean = text.strip()
    identity = json.dumps(
        {
            "document_id": document_id,
            "heading_path": heading_path,
            "block_type": block_type,
            "text": clean,
            "ordinal": ordinal,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return CorpusChunk(
        chunk_id="C_" + _hash_text(identity)[:20],
        document_id=document_id,
        source_url=document.url,
        title=document.title,
        site_scope=document.classification.site_scope,
        trust_level=document.classification.trust_level,
        heading_path=heading_path,
        block_type=block_type,
        text=clean,
        content_hash=_hash_text(clean),
    )


def _split_text(text: str, *, maximum_chars: int) -> list[str]:
    """Split without overlap, preferring paragraph and sentence boundaries."""

    remaining = text.strip()
    pieces: list[str] = []
    while remaining:
        if len(remaining) <= maximum_chars:
            pieces.append(remaining)
            break
        window = remaining[:maximum_chars]
        boundary = max(
            window.rfind("\n\n"),
            window.rfind("\n"),
            window.rfind(". "),
            window.rfind(" "),
        )
        if boundary < maximum_chars // 2:
            boundary = maximum_chars
        elif window[boundary : boundary + 2] == ". ":
            boundary += 1
        piece = remaining[:boundary].strip()
        if piece:
            pieces.append(piece)
        remaining = remaining[boundary:].strip()
    return pieces


def _whole_section(heading_path: list[str]) -> bool:
    normalized = " ".join(heading_path).lower()
    tokens = set(re.findall(r"[a-z]+", normalized))
    return bool(tokens & WHOLE_SECTION_TERMS) or "frequently asked questions" in normalized


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()

