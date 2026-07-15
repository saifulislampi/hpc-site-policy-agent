"""Schemas for deterministic discovery and source selection."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


SiteScope = Literal["target_site", "organization_general", "sibling", "unrelated"]
TrustLevel = Literal["official_web", "manifest_declared", "unverified_local"]
TopicName = Literal["submission_policy", "networking_policy"]
TopicCoverageStatus = Literal[
    "evidence_found",
    "documentation_silent",
    "search_exhausted",
    "not_investigated",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SiteIdentity(StrictModel):
    display_name: str = Field(min_length=1, max_length=300)
    aliases: list[str] = Field(min_length=1)
    organization_domains: list[str] = Field(min_length=1)
    preferred_path_tokens: list[str] = Field(min_length=1)
    excluded_site_tokens: list[str]


class SourceClassification(StrictModel):
    site_scope: SiteScope
    trust_level: TrustLevel
    matched_aliases: list[str]
    conflicting_site_tokens: list[str]
    score: float
    reasons: list[str]


class CandidateSource(StrictModel):
    url: HttpUrl
    title: str
    snippet: str
    classification: SourceClassification
    fetched: bool = False
    selected: bool = False
    rejection_reason: str | None = None


class DocumentLink(StrictModel):
    url: HttpUrl
    text: str


class DocumentBlock(StrictModel):
    kind: Literal["text", "code", "table"]
    text: str


class DocumentSection(StrictModel):
    heading: str
    heading_path: list[str]
    text: str
    links: list[HttpUrl]
    blocks: list[DocumentBlock]


class TopicCoverage(StrictModel):
    topic: TopicName
    status: TopicCoverageStatus
    target_site_pages_examined: int = Field(ge=0)
    queries_attempted: int = Field(ge=0)
    evidence_urls: list[HttpUrl]
    notes: list[str]


class DiscoveryCoverage(StrictModel):
    canonical_root_found: bool
    canonical_root: HttpUrl | None
    submission_policy: TopicCoverage
    networking_policy: TopicCoverage


class FinishDiscoveryArgs(StrictModel):
    source_urls: list[HttpUrl] = Field(max_length=10)
    summary: str = Field(min_length=1, max_length=2000)
    unanswered_topics: list[str]


class DiscoverySelection(StrictModel):
    source_urls: list[HttpUrl] = Field(max_length=10)
    summary: str = Field(min_length=1, max_length=2000)
    unanswered_topics: list[str]
    canonical_root: HttpUrl | None
    coverage: DiscoveryCoverage
    termination_reason: str


class FetchedDocument(StrictModel):
    url: HttpUrl
    title: str
    text: str
    links: list[DocumentLink]
    sections: list[DocumentSection]
    relevant_text: str
    text_truncated: bool
    classification: SourceClassification
