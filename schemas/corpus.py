"""Persistent provider-neutral corpus and retrieval schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from schemas.discovery import SiteIdentity, SiteScope, TrustLevel


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CorpusDocument(StrictModel):
    document_id: str
    source_url: HttpUrl
    title: str
    site_scope: SiteScope
    trust_level: TrustLevel
    content_hash: str
    fetched_at: datetime
    text_truncated: bool


class CorpusChunk(StrictModel):
    chunk_id: str
    document_id: str
    source_url: HttpUrl
    title: str
    site_scope: SiteScope
    trust_level: TrustLevel
    heading_path: list[str]
    block_type: Literal["text", "code", "table", "section"]
    text: str
    content_hash: str


class ChunkerConfiguration(StrictModel):
    version: Literal["0.1"] = "0.1"
    maximum_chars: int = Field(ge=200)
    overlap_chars: Literal[0] = 0


class CorpusManifest(StrictModel):
    schema_version: Literal["0.1"] = "0.1"
    corpus_id: str
    site_identity: SiteIdentity
    created_at: datetime
    updated_at: datetime
    corpus_fingerprint: str
    chunker: ChunkerConfiguration
    document_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)


class RetrievalHit(StrictModel):
    chunk: CorpusChunk
    score: float = Field(ge=0)


class FieldRetrieval(StrictModel):
    field: str
    query: str
    allowed_scopes: list[SiteScope]
    hits: list[RetrievalHit]
    excluded_scope_counts: dict[SiteScope, int]
