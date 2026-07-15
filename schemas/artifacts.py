"""Stable external JSON artifacts consumed by evaluation and deployment tools."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from schemas.discovery import SiteScope, TrustLevel
from schemas.extraction import (
    ConnectivityValue,
    DocumentationStatus,
    EvidenceInterpretation,
    FindingStatus,
    PortRange,
    SubmissionOption,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReportRun(StrictModel):
    run_id: str
    site: str
    provider: str
    model: str
    timestamp: datetime


class ArtifactCoverage(StrictModel):
    submission: Literal[
        "complete", "documentation_silent", "discovery_failed", "incomplete"
    ]
    networking: Literal[
        "complete", "documentation_silent", "discovery_failed", "incomplete"
    ]


class ReportDiscovery(StrictModel):
    canonical_root: HttpUrl | None
    queries: list[dict[str, Any]]
    coverage: ArtifactCoverage
    termination_reason: str


class ReportCorpus(StrictModel):
    corpus_id: str
    corpus_fingerprint: str
    manifest: str
    document_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)


class ReportSource(StrictModel):
    id: str
    url: HttpUrl
    title: str
    site_scope: SiteScope
    trust_level: TrustLevel
    selected: bool
    retrieved: bool
    rejection_reason: str | None = None


class ReportEvidence(StrictModel):
    chunk_id: str
    source_id: str
    heading: str | None = None
    quote: str
    interpretation: EvidenceInterpretation


class ReportFinding(StrictModel):
    value: Any | None
    status: FindingStatus
    documentation_status: DocumentationStatus
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str
    evidence: list[ReportEvidence]


class DiscoveryStatistics(StrictModel):
    search_calls: int = Field(ge=0)
    pages_fetched: int = Field(ge=0)
    target_site_pages: int = Field(ge=0)
    sibling_sources: int = Field(ge=0)
    sibling_chunks: int = Field(ge=0)
    retrieved_sibling_chunks: int = Field(ge=0)
    model_requests: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class DiscoveryReport(StrictModel):
    schema_version: Literal["0.1"] = "0.1"
    run: ReportRun
    discovery: ReportDiscovery
    corpus: ReportCorpus
    sources: list[ReportSource]
    retrieval: dict[str, "ReportFieldRetrieval"]
    findings: dict[str, ReportFinding]
    unresolved_questions: list[str]
    statistics: DiscoveryStatistics


class SiteDescriptor(StrictModel):
    id: str
    name: str


class SchedulerProfile(StrictModel):
    type: str | None
    submit_command: str | None


class SubmissionProfile(StrictModel):
    options: list[SubmissionOption]


class PartitionProfile(StrictModel):
    maximum_walltime: str | None
    maximum_nodes: int | None
    shared_nodes: bool | None


class PartitionsProfile(StrictModel):
    default: str | None
    limits: dict[str, PartitionProfile]


class NetworkProfile(StrictModel):
    manager_worker: ConnectivityValue | None
    worker_worker: ConnectivityValue | None
    port_range: list[PortRange] | None
    manager_address: str | None
    outbound_compute: ConnectivityValue | None


class StorageProfile(StrictModel):
    shared_filesystem: str | None
    scratch_directory: str | None
    temporary_directory: str | None
    symlink_supported: bool | None
    hardlink_supported: bool | None


ProfileSectionStatus = Literal[
    "documented", "partial", "probe_required", "conflicting", "not_applicable"
]


class PolicyValidation(StrictModel):
    scheduler: ProfileSectionStatus
    submission: ProfileSectionStatus
    partitions: ProfileSectionStatus
    network: ProfileSectionStatus
    storage: ProfileSectionStatus


class PolicyProvenance(StrictModel):
    discovery_report: str
    run_id: str
    corpus_id: str
    corpus_fingerprint: str
    references: dict[str, str]


class ReportRetrievedChunk(StrictModel):
    chunk_id: str
    source_id: str
    score: float = Field(ge=0)
    cited: bool


class ReportFieldRetrieval(StrictModel):
    query: str
    allowed_scopes: list[SiteScope]
    chunks: list[ReportRetrievedChunk]
    excluded_scope_counts: dict[SiteScope, int]


class SitePolicyArtifact(StrictModel):
    schema_version: Literal["0.1"] = "0.1"
    site: SiteDescriptor
    scheduler: SchedulerProfile
    submission: SubmissionProfile
    partitions: PartitionsProfile
    network: NetworkProfile
    storage: StorageProfile
    validation: PolicyValidation
    provenance: PolicyProvenance
