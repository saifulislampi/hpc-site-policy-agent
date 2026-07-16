"""Provider-neutral structured output expected from every model adapter."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

from schemas.discovery import DiscoveryCoverage


FindingStatus = Literal[
    "documented",
    "conflicting",
    "requires_probe",
    "not_applicable",
    "not_investigated",
]
EvidenceInterpretation = Literal["direct", "inferred", "conflicting", "silent"]
DocumentationStatus = Literal[
    "documented",
    "silent",
    "discovery_failed",
    "extraction_failed",
    "not_applicable",
    "not_investigated",
]
ConnectivityValue = Literal["allowed", "blocked", "conditional"]
OptionValue = str | int | float | bool


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Evidence(StrictModel):
    chunk_id: str = Field(min_length=1, max_length=100)
    source_url: HttpUrl
    source_title: str = Field(min_length=1, max_length=500)
    heading: str | None = Field(max_length=500)
    quote: str = Field(min_length=1, max_length=1200)
    interpretation: EvidenceInterpretation


class EvidenceReference(StrictModel):
    """A model-selected reference to an application-generated literal span."""

    evidence_ref: str = Field(min_length=1, max_length=150)
    interpretation: EvidenceInterpretation


class FindingBase(StrictModel):
    value: object | None
    status: FindingStatus
    documentation_status: DocumentationStatus
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str = Field(min_length=1, max_length=1500)
    evidence: list[Evidence]

    @model_validator(mode="after")
    def validate_status_and_evidence(self) -> "FindingBase":
        if self.status in {"documented", "conflicting"} and not self.evidence:
            raise ValueError("Documented or conflicting findings require evidence.")
        if self.status in {"documented", "conflicting"}:
            if self.documentation_status != "documented":
                raise ValueError(
                    "Documented or conflicting findings need documentation_status=documented."
                )
            if not any(
                item.interpretation in {"direct", "conflicting"}
                for item in self.evidence
            ):
                raise ValueError(
                    "A definitive documented finding needs direct or conflicting evidence."
                )
        if self.status == "requires_probe" and self.value is not None:
            raise ValueError("A requires_probe finding must have value=null.")
        if self.status == "requires_probe" and self.documentation_status not in {
            "silent",
            "discovery_failed",
            "extraction_failed",
        }:
            raise ValueError(
                "A requires_probe finding must distinguish documentation silence, "
                "discovery failure, and extraction failure."
            )
        if self.status == "not_applicable" and self.documentation_status != "not_applicable":
            raise ValueError(
                "A not_applicable finding needs documentation_status=not_applicable."
            )
        if self.status == "not_investigated":
            if self.value is not None or self.evidence:
                raise ValueError("A not_investigated finding must be null and unevidenced.")
            if self.documentation_status != "not_investigated":
                raise ValueError(
                    "A not_investigated finding needs "
                    "documentation_status=not_investigated."
                )
        return self


class StringFinding(FindingBase):
    value: str | None


class ConnectivityFinding(FindingBase):
    value: ConnectivityValue | None


class PartitionSpec(StrictModel):
    name: str = Field(min_length=1, max_length=100)
    maximum_walltime: str | None = Field(max_length=100)
    maximum_nodes: int | None = Field(ge=1)
    shared_nodes: bool | None


class PartitionListFinding(FindingBase):
    value: list[PartitionSpec] | None


class PortRange(StrictModel):
    protocol: Literal["tcp", "udp", "tcp_udp"]
    start: int = Field(ge=1, le=65535)
    end: int = Field(ge=1, le=65535)

    @model_validator(mode="after")
    def ordered_range(self) -> "PortRange":
        if self.end < self.start:
            raise ValueError("Port range end must be greater than or equal to start.")
        return self


class PortRangeFinding(FindingBase):
    value: list[PortRange] | None


class SubmissionOption(StrictModel):
    name: str = Field(min_length=1, max_length=100)
    syntax: list[str] = Field(min_length=1)
    required: bool
    example: str | None = Field(max_length=500)
    value: OptionValue | None


class SubmissionOptionsFinding(FindingBase):
    value: list[SubmissionOption] | None


class ReferenceStringFinding(StringFinding):
    evidence: list[EvidenceReference]


class ReferenceConnectivityFinding(ConnectivityFinding):
    evidence: list[EvidenceReference]


class ReferencePartitionListFinding(PartitionListFinding):
    evidence: list[EvidenceReference]


class ReferencePortRangeFinding(PortRangeFinding):
    evidence: list[EvidenceReference]


class ReferenceSubmissionOptionsFinding(SubmissionOptionsFinding):
    evidence: list[EvidenceReference]


class SubmissionExtraction(StrictModel):
    scheduler: ReferenceStringFinding
    submit_command: ReferenceStringFinding
    submission_options: ReferenceSubmissionOptionsFinding
    account_allocation_policy: ReferenceStringFinding
    default_partition: ReferenceStringFinding
    partitions: ReferencePartitionListFinding


class NetworkExtraction(StrictModel):
    manager_worker_connectivity: ReferenceConnectivityFinding
    worker_worker_connectivity: ReferenceConnectivityFinding
    published_port_range: ReferencePortRangeFinding
    manager_address_guidance: ReferenceStringFinding
    outbound_compute_network: ReferenceConnectivityFinding


class OperationalExtraction(StrictModel):
    walltime_policy: ReferenceStringFinding
    memory_policy: ReferenceStringFinding
    job_size_policy: ReferenceStringFinding
    charging_model: ReferenceStringFinding
    purge_policy: ReferenceStringFinding
    cost_traps: ReferenceStringFinding
    login_node_socket_policy: ReferenceStringFinding


class DocumentSource(StrictModel):
    url: HttpUrl
    title: str = Field(min_length=1, max_length=500)
    authority: Literal["official", "secondary"]
    relevance: str = Field(min_length=1, max_length=1000)


class SlurmPolicy(StrictModel):
    scheduler: StringFinding
    submit_command: StringFinding
    submission_options: SubmissionOptionsFinding
    account_allocation_policy: StringFinding
    default_partition: StringFinding
    partitions: PartitionListFinding
    walltime_policy: StringFinding
    memory_policy: StringFinding
    job_size_policy: StringFinding
    charging_model: StringFinding
    purge_policy: StringFinding
    cost_traps: StringFinding


class NetworkPolicy(StrictModel):
    manager_worker_connectivity: ConnectivityFinding
    worker_worker_connectivity: ConnectivityFinding
    published_port_range: PortRangeFinding
    manager_address_guidance: StringFinding
    login_node_socket_policy: StringFinding
    outbound_compute_network: ConnectivityFinding


class ExtractedPolicy(StrictModel):
    site_name: str = Field(min_length=1, max_length=300)
    sources: list[DocumentSource]
    discovery_coverage: DiscoveryCoverage
    slurm_policy: SlurmPolicy
    network_policy: NetworkPolicy
    unresolved_questions: list[str]
    overall_notes: list[str]
