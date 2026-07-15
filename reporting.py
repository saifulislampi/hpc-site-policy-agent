"""Build stable research and candidate-policy artifacts from one agent run."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from schemas import (
    ArtifactCoverage,
    DiscoveryReport,
    ExtractedPolicy,
    FieldStatus,
    NetworkProfile,
    PartitionProfile,
    ProfileValidation,
    Provenance,
    ReportEvidence,
    ReportFinding,
    ReportRun,
    ReportSource,
    RequiredProbe,
    SchedulerProfile,
    SiteDescriptor,
    SitePolicyArtifact,
    SiteProfileValues,
    SubmissionProfile,
)


FINDING_PATHS = {
    "scheduler": "scheduler.type",
    "required_submission_options": "submission.required_options",
    "account_required": "submission.account_required",
    "account_allocation_policy": "submission.account_allocation_policy",
    "default_partition": "submission.default_partition",
    "partitions": "submission.partitions",
    "walltime_policy": "submission.walltime_policy",
    "memory_policy": "submission.memory_policy",
    "job_size_policy": "submission.job_size_policy",
    "charging_policy": "submission.charging_policy",
    "manager_worker_connectivity": "network.manager_worker",
    "worker_worker_connectivity": "network.worker_worker",
    "published_port_range": "network.published_port_range",
    "manager_address_guidance": "network.manager_address_guidance",
    "login_node_socket_policy": "network.login_node_socket_policy",
    "outbound_compute_network": "network.outbound_compute",
}


@dataclass(slots=True)
class RunArtifacts:
    discovery_report: DiscoveryReport
    site_policy: SitePolicyArtifact


def build_run_artifacts(
    *,
    extracted: ExtractedPolicy,
    tools: Any,
    run_id: str,
    provider: str,
    model: str,
    timestamp: datetime,
    site_id: str,
    organization: str | None,
    discovery_report_filename: str,
    termination_reason: str,
    metrics: dict[str, Any],
) -> RunArtifacts:
    sources, source_ids = _build_sources(tools)
    findings = _build_findings(extracted, source_ids)
    coverage = ArtifactCoverage(
        submission=_coverage_status(extracted.discovery_coverage.submission_policy.status),
        networking=_coverage_status(extracted.discovery_coverage.networking_policy.status),
    )
    discovery_report = DiscoveryReport(
        run=ReportRun(
            run_id=run_id,
            site=extracted.site_name,
            provider=provider,
            model=model,
            timestamp=timestamp,
        ),
        discovery={
            "canonical_root": extracted.discovery_coverage.canonical_root,
            "queries": list(getattr(tools, "query_log", [])),
            "coverage": coverage,
            "termination_reason": termination_reason,
        },
        sources=sources,
        findings=findings,
        unresolved_questions=extracted.unresolved_questions,
        statistics={
            "search_calls": metrics.get("search_requests", 0),
            "pages_fetched": metrics.get("page_requests", 0),
            "target_site_pages": metrics.get("fetched_target_documents", 0),
            "rejected_other_site_pages": sum(
                source.site_scope == "other_site" for source in sources
            ),
            "model_requests": metrics.get("model_requests", 0),
            "input_tokens": metrics.get("total_input_tokens", 0),
            "output_tokens": metrics.get("total_output_tokens", 0),
        },
    )
    site_policy = _build_site_policy(
        extracted=extracted,
        source_ids=source_ids,
        site_id=site_id,
        organization=organization,
        report_filename=Path(discovery_report_filename).name,
        run_id=run_id,
    )
    return RunArtifacts(discovery_report=discovery_report, site_policy=site_policy)


def _build_sources(tools: Any) -> tuple[list[ReportSource], dict[str, str]]:
    candidates = list(getattr(tools, "candidates", {}).values())
    candidates.sort(
        key=lambda item: (
            not item.selected,
            -item.classification.score,
            str(item.url),
        )
    )
    sources: list[ReportSource] = []
    source_ids: dict[str, str] = {}
    for index, candidate in enumerate(candidates, start=1):
        url = str(candidate.url)
        source_id = f"S{index}"
        source_ids[url] = source_id
        document = getattr(tools, "documents", {}).get(url)
        title = document.title if document is not None else candidate.title
        sources.append(
            ReportSource(
                id=source_id,
                url=url,
                title=title or url,
                site_scope=candidate.classification.site_scope,
                selected=candidate.selected,
                rejection_reason=candidate.rejection_reason,
            )
        )
    return sources, source_ids


def _build_findings(
    extracted: ExtractedPolicy,
    source_ids: dict[str, str],
) -> dict[str, ReportFinding]:
    findings: dict[str, ReportFinding] = {}
    for policy in (extracted.slurm_policy, extracted.network_policy):
        for field_name, finding in policy.__dict__.items():
            key = FINDING_PATHS[field_name]
            findings[key] = ReportFinding(
                value=finding.value,
                status=finding.status,
                documentation_status=finding.documentation_status,
                confidence=finding.confidence,
                explanation=finding.explanation,
                evidence=[
                    ReportEvidence(
                        source_id=source_ids[str(item.source_url)],
                        heading=item.heading,
                        quote=item.quote,
                        interpretation=item.interpretation,
                    )
                    for item in finding.evidence
                ],
            )
    return findings


def _build_site_policy(
    *,
    extracted: ExtractedPolicy,
    source_ids: dict[str, str],
    site_id: str,
    organization: str | None,
    report_filename: str,
    run_id: str,
) -> SitePolicyArtifact:
    slurm = extracted.slurm_policy
    network = extracted.network_policy
    partitions = {
        item.name: PartitionProfile(
            maximum_walltime=item.maximum_walltime,
            maximum_nodes=item.maximum_nodes,
            shared_nodes=item.shared_nodes,
        )
        for item in (_documented_value(slurm.partitions) or [])
    }

    normalized_findings = {
        "/profile/scheduler/type": slurm.scheduler,
        "/profile/submission/required_options": slurm.required_submission_options,
        "/profile/submission/default_partition": slurm.default_partition,
        "/profile/submission/account_required": slurm.account_required,
        "/profile/partitions": slurm.partitions,
        "/profile/network/manager_worker": network.manager_worker_connectivity,
        "/profile/network/worker_worker": network.worker_worker_connectivity,
        "/profile/network/published_port_range": network.published_port_range,
        "/profile/network/outbound_compute": network.outbound_compute_network,
    }
    field_status = {
        path: FieldStatus(
            status=finding.status,
            documentation_status=finding.documentation_status,
            source_ids=list(
                dict.fromkeys(source_ids[str(item.source_url)] for item in finding.evidence)
            ),
        )
        for path, finding in normalized_findings.items()
    }
    probe_names = {
        "/profile/scheduler/type": "scheduler_type",
        "/profile/submission/required_options": "required_submission_options",
        "/profile/submission/default_partition": "default_partition",
        "/profile/submission/account_required": "account_requirement",
        "/profile/partitions": "partition_configuration",
        "/profile/network/manager_worker": "manager_worker_connectivity",
        "/profile/network/worker_worker": "worker_worker_connectivity",
        "/profile/network/published_port_range": "tcp_port_range",
        "/profile/network/outbound_compute": "outbound_compute_network",
    }
    required_probes = [
        RequiredProbe(probe=probe_names[path], target_field=path)
        for path, finding in normalized_findings.items()
        if finding.status == "requires_probe"
    ]
    documented = sum(item.status == "documented" for item in field_status.values())
    probes = sum(item.status == "requires_probe" for item in field_status.values())
    conflicts = sum(item.status == "conflicting" for item in field_status.values())
    state = "conflicting" if conflicts else "partial" if probes else "complete"

    return SitePolicyArtifact(
        site=SiteDescriptor(
            id=site_id,
            name=extracted.site_name,
            organization=organization,
        ),
        profile=SiteProfileValues(
            scheduler=SchedulerProfile(type=_documented_value(slurm.scheduler)),
            submission=SubmissionProfile(
                required_options=_documented_value(slurm.required_submission_options),
                default_partition=_documented_value(slurm.default_partition),
                account_required=_documented_value(slurm.account_required),
            ),
            partitions=partitions,
            network=NetworkProfile(
                manager_worker=_documented_value(network.manager_worker_connectivity),
                worker_worker=_documented_value(network.worker_worker_connectivity),
                published_port_range=_documented_value(network.published_port_range),
                outbound_compute=_documented_value(network.outbound_compute_network),
            ),
        ),
        field_status=field_status,
        validation=ProfileValidation(
            state=state,
            documented_fields=documented,
            probe_required_fields=probes,
            conflicting_fields=conflicts,
        ),
        required_probes=required_probes,
        provenance=Provenance(report=report_filename, run_id=run_id),
    )


def _coverage_status(status: str) -> str:
    return {
        "evidence_found": "complete",
        "documentation_silent": "documentation_silent",
        "search_exhausted": "discovery_failed",
        "not_investigated": "incomplete",
    }[status]


def _documented_value(finding: Any) -> Any | None:
    return finding.value if finding.status == "documented" else None
