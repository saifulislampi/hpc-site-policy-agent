"""Merge-safe JSONL corpus persistence; retrieval indexes remain transient."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from schemas import (
    ChunkerConfiguration,
    CorpusChunk,
    CorpusDocument,
    CorpusManifest,
    SiteIdentity,
)


@dataclass(slots=True)
class CorpusSnapshot:
    manifest: CorpusManifest
    documents: list[CorpusDocument]
    chunks: list[CorpusChunk]
    active_chunks: list[CorpusChunk]


class CorpusStore:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.manifest_path = self.directory / "manifest.json"
        self.documents_path = self.directory / "documents.jsonl"
        self.chunks_path = self.directory / "chunks.jsonl"

    def merge(
        self,
        *,
        corpus_id: str,
        site_identity: SiteIdentity,
        incoming_documents: list[CorpusDocument],
        incoming_chunks: list[CorpusChunk],
        chunker: ChunkerConfiguration,
        refresh: bool,
        now: datetime,
    ) -> CorpusSnapshot:
        existing_manifest, existing_documents, existing_chunks = self.load()
        if (
            existing_manifest is not None
            and existing_manifest.site_identity != site_identity
        ):
            raise ValueError(
                "Corpus site identity does not match this run; use another --corpus-dir."
            )

        documents_by_url = {str(item.source_url): item for item in existing_documents}
        chunks_by_document: dict[str, list[CorpusChunk]] = {}
        for chunk in existing_chunks:
            chunks_by_document.setdefault(chunk.document_id, []).append(chunk)
        incoming_chunks_by_document: dict[str, list[CorpusChunk]] = {}
        for chunk in incoming_chunks:
            incoming_chunks_by_document.setdefault(chunk.document_id, []).append(chunk)

        active_document_ids: set[str] = set()
        for incoming in incoming_documents:
            url = str(incoming.source_url)
            existing = documents_by_url.get(url)
            should_replace = (
                existing is None
                or existing.content_hash == incoming.content_hash
                or refresh
            )
            if should_replace:
                if existing is not None:
                    chunks_by_document.pop(existing.document_id, None)
                documents_by_url[url] = incoming
                chunks_by_document[incoming.document_id] = list(
                    incoming_chunks_by_document.get(incoming.document_id, [])
                )
                active_document_ids.add(incoming.document_id)
            else:
                active_document_ids.add(existing.document_id)

        merged_documents = sorted(
            documents_by_url.values(), key=lambda item: str(item.source_url)
        )
        merged_chunks = sorted(
            [chunk for values in chunks_by_document.values() for chunk in values],
            key=lambda item: item.chunk_id,
        )
        active_chunks = [
            chunk for chunk in merged_chunks if chunk.document_id in active_document_ids
        ]
        created_at = existing_manifest.created_at if existing_manifest else now
        fingerprint = _fingerprint(merged_documents, chunker)
        manifest = CorpusManifest(
            corpus_id=corpus_id,
            site_identity=site_identity,
            created_at=created_at,
            updated_at=now,
            corpus_fingerprint=fingerprint,
            chunker=chunker,
            document_count=len(merged_documents),
            chunk_count=len(merged_chunks),
        )
        self._write(manifest, merged_documents, merged_chunks)
        return CorpusSnapshot(
            manifest=manifest,
            documents=merged_documents,
            chunks=merged_chunks,
            active_chunks=active_chunks,
        )

    def load(
        self,
    ) -> tuple[CorpusManifest | None, list[CorpusDocument], list[CorpusChunk]]:
        manifest = (
            CorpusManifest.model_validate_json(self.manifest_path.read_text("utf-8"))
            if self.manifest_path.exists()
            else None
        )
        return (
            manifest,
            _read_jsonl(self.documents_path, CorpusDocument),
            _read_jsonl(self.chunks_path, CorpusChunk),
        )

    def _write(
        self,
        manifest: CorpusManifest,
        documents: list[CorpusDocument],
        chunks: list[CorpusChunk],
    ) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        _atomic_write(self.manifest_path, manifest.model_dump_json(indent=2) + "\n")
        _atomic_write(
            self.documents_path,
            "".join(item.model_dump_json() + "\n" for item in documents),
        )
        _atomic_write(
            self.chunks_path,
            "".join(item.model_dump_json() + "\n" for item in chunks),
        )


def _read_jsonl(path: Path, model: type) -> list:
    if not path.exists():
        return []
    return [model.model_validate_json(line) for line in path.read_text("utf-8").splitlines() if line]


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _fingerprint(
    documents: list[CorpusDocument],
    chunker: ChunkerConfiguration,
) -> str:
    payload = {
        "documents": [
            [str(item.source_url), item.content_hash] for item in documents
        ],
        "chunker": chunker.model_dump(mode="json"),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return "sha256:" + digest

