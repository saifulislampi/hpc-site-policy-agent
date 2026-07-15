"""Stable external JSON artifacts consumed by evaluation and deployment tools."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from schemas.discovery import SiteScope
from schemas.extraction import (
    ConnectivityValue,
    DocumentationStatus,
    EvidenceInterpretation,
    FindingStatus,
    PortRange,
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


class ReportSource(StrictModel):
    id: str
    url: HttpUrl
    title: str
    site_scope: SiteScope
    selected: bool
    rejection_reason: str | None = None


class ReportEvidence(StrictModel):
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
    rejected_other_site_pages: int = Field(ge=0)
    model_requests: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class DiscoveryReport(StrictModel):
    schema_version: Literal["0.1"] = "0.1"
    run: ReportRun
    discovery: ReportDiscovery
    sources: list[ReportSource]
    findings: dict[str, ReportFinding]
    unresolved_questions: list[str]
    statistics: DiscoveryStatistics


class SiteDescriptor(StrictModel):
    id: str
    name: str
    organization: str | None


class SchedulerProfile(StrictModel):
    type: str | None


class SubmissionProfile(StrictModel):
    required_options: list[str] | None
    default_partition: str | None
    account_required: bool | None


class PartitionProfile(StrictModel):
    maximum_walltime: str | None
    maximum_nodes: int | None
    shared_nodes: bool | None


class NetworkProfile(StrictModel):
    manager_worker: ConnectivityValue | None
    worker_worker: ConnectivityValue | None
    published_port_range: list[PortRange] | None
    outbound_compute: ConnectivityValue | None


class SiteProfileValues(StrictModel):
    scheduler: SchedulerProfile
    submission: SubmissionProfile
    partitions: dict[str, PartitionProfile]
    network: NetworkProfile


class FieldStatus(StrictModel):
    status: FindingStatus
    documentation_status: DocumentationStatus
    source_ids: list[str] = Field(default_factory=list)


class ProfileValidation(StrictModel):
    state: Literal["complete", "partial", "conflicting"]
    documented_fields: int = Field(ge=0)
    probe_required_fields: int = Field(ge=0)
    conflicting_fields: int = Field(ge=0)


class RequiredProbe(StrictModel):
    probe: str
    target_field: str


class Provenance(StrictModel):
    report: str
    run_id: str


class SitePolicyArtifact(StrictModel):
    schema_version: Literal["0.1"] = "0.1"
    profile_state: Literal["candidate", "validated"] = "candidate"
    site: SiteDescriptor
    profile: SiteProfileValues
    field_status: dict[str, FieldStatus]
    validation: ProfileValidation
    required_probes: list[RequiredProbe]
    provenance: Provenance
