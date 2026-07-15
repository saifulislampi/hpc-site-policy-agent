"""Build stable research and candidate-policy artifacts from one agent run."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from schemas import (
    ArtifactCoverage,
    DiscoveryReport,
    ExtractedPolicy,
    NetworkProfile,
    PartitionProfile,
    PartitionsProfile,
    PolicyProvenance,
    PolicyValidation,
    ReportCorpus,
    ReportEvidence,
    ReportFieldRetrieval,
    ReportFinding,
    ReportRun,
    ReportRetrievedChunk,
    ReportSource,
    SchedulerProfile,
    SiteDescriptor,
    SitePolicyArtifact,
    StorageProfile,
    SubmissionProfile,
)


FINDING_PATHS = {
    "scheduler": "scheduler.type",
    "submit_command": "scheduler.submit_command",
    "submission_options": "submission.options",
    "account_allocation_policy": "submission.account_allocation_policy",
    "default_partition": "submission.default_partition",
    "partitions": "submission.partitions",
    "walltime_policy": "submission.walltime_policy",
    "memory_policy": "submission.memory_policy",
    "job_size_policy": "submission.job_size_policy",
    "charging_model": "submission.charging_model",
    "purge_policy": "storage.purge_policy",
    "cost_traps": "submission.cost_traps",
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
    discovery_report_reference: str,
    termination_reason: str,
    metrics: dict[str, Any],
    corpus_snapshot: Any,
    retrievals: dict[str, Any],
    citation_audit: dict[str, Any],
    corpus_manifest_reference: str,
) -> RunArtifacts:
    retrieved_urls = {
        str(hit.chunk.source_url)
        for retrieval in retrievals.values()
        for hit in retrieval.hits
    }
    sources, source_ids = _build_sources(tools, retrieved_urls)
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
        corpus=ReportCorpus(
            corpus_id=corpus_snapshot.manifest.corpus_id,
            corpus_fingerprint=corpus_snapshot.manifest.corpus_fingerprint,
            manifest=corpus_manifest_reference,
            document_count=corpus_snapshot.manifest.document_count,
            chunk_count=corpus_snapshot.manifest.chunk_count,
        ),
        sources=sources,
        retrieval=_build_retrieval_report(
            retrievals, citation_audit, source_ids
        ),
        findings=findings,
        unresolved_questions=extracted.unresolved_questions,
        statistics={
            "search_calls": metrics.get("search_requests", 0),
            "pages_fetched": metrics.get("page_requests", 0),
            "target_site_pages": metrics.get("fetched_target_documents", 0),
            "sibling_sources": sum(source.site_scope == "sibling" for source in sources),
            "sibling_chunks": sum(
                chunk.site_scope == "sibling" for chunk in corpus_snapshot.chunks
            ),
            "retrieved_sibling_chunks": sum(
                hit.chunk.site_scope == "sibling"
                for retrieval in retrievals.values()
                for hit in retrieval.hits
            ),
            "model_requests": metrics.get("model_requests", 0),
            "input_tokens": metrics.get("total_input_tokens", 0),
            "output_tokens": metrics.get("total_output_tokens", 0),
        },
    )
    site_policy = _build_site_policy(
        extracted=extracted,
        site_id=site_id,
        discovery_report_reference=discovery_report_reference,
        run_id=run_id,
        corpus_id=corpus_snapshot.manifest.corpus_id,
        corpus_fingerprint=corpus_snapshot.manifest.corpus_fingerprint,
    )
    return RunArtifacts(discovery_report=discovery_report, site_policy=site_policy)


def _build_sources(
    tools: Any,
    retrieved_urls: set[str],
) -> tuple[list[ReportSource], dict[str, str]]:
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
        document = getattr(tools, "documents", {}).get(url) or getattr(
            tools, "rejected_documents", {}
        ).get(url)
        title = document.title if document is not None else candidate.title
        sources.append(
            ReportSource(
                id=source_id,
                url=url,
                title=title or url,
                site_scope=candidate.classification.site_scope,
                trust_level=candidate.classification.trust_level,
                selected=candidate.selected,
                retrieved=url in retrieved_urls,
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
                        chunk_id=item.chunk_id,
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
    site_id: str,
    discovery_report_reference: str,
    run_id: str,
    corpus_id: str,
    corpus_fingerprint: str,
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

    return SitePolicyArtifact(
        site=SiteDescriptor(
            id=site_id,
            name=extracted.site_name,
        ),
        scheduler=SchedulerProfile(
            type=_documented_value(slurm.scheduler),
            submit_command=_documented_value(slurm.submit_command),
        ),
        submission=SubmissionProfile(
            options=_documented_value(slurm.submission_options) or [],
        ),
        partitions=PartitionsProfile(
            default=_documented_value(slurm.default_partition),
            limits=partitions,
        ),
        network=NetworkProfile(
            manager_worker=_documented_value(network.manager_worker_connectivity),
            worker_worker=_documented_value(network.worker_worker_connectivity),
            port_range=_documented_value(network.published_port_range),
            manager_address=_documented_value(network.manager_address_guidance),
            outbound_compute=_documented_value(network.outbound_compute_network),
        ),
        storage=StorageProfile(
            shared_filesystem=None,
            scratch_directory=None,
            temporary_directory=None,
            symlink_supported=None,
            hardlink_supported=None,
        ),
        validation=PolicyValidation(
            scheduler=_section_status([slurm.scheduler, slurm.submit_command]),
            submission=_section_status(
                [slurm.submission_options, slurm.account_allocation_policy]
            ),
            partitions=_section_status([slurm.default_partition, slurm.partitions]),
            network=_section_status(
                [
                    network.manager_worker_connectivity,
                    network.worker_worker_connectivity,
                    network.published_port_range,
                    network.manager_address_guidance,
                    network.outbound_compute_network,
                ]
            ),
            storage="probe_required",
        ),
        provenance=PolicyProvenance(
            discovery_report=discovery_report_reference,
            run_id=run_id,
            corpus_id=corpus_id,
            corpus_fingerprint=corpus_fingerprint,
            references={
                "/scheduler/type": "/findings/scheduler.type",
                "/scheduler/submit_command": "/findings/scheduler.submit_command",
                "/submission/options": "/findings/submission.options",
                "/partitions/default": "/findings/submission.default_partition",
                "/partitions/limits": "/findings/submission.partitions",
                "/network/manager_worker": "/findings/network.manager_worker",
                "/network/worker_worker": "/findings/network.worker_worker",
                "/network/port_range": "/findings/network.published_port_range",
                "/network/manager_address": (
                    "/findings/network.manager_address_guidance"
                ),
                "/network/outbound_compute": "/findings/network.outbound_compute",
            },
        ),
    )


def _build_retrieval_report(
    retrievals: dict[str, Any],
    citation_audit: dict[str, Any],
    source_ids: dict[str, str],
) -> dict[str, ReportFieldRetrieval]:
    result: dict[str, ReportFieldRetrieval] = {}
    for field, retrieval in retrievals.items():
        cited = {
            item["chunk_id"]
            for item in citation_audit[field]["retrieved"]
            if item["cited"]
        }
        result[field] = ReportFieldRetrieval(
            query=retrieval.query,
            allowed_scopes=retrieval.allowed_scopes,
            chunks=[
                ReportRetrievedChunk(
                    chunk_id=hit.chunk.chunk_id,
                    source_id=source_ids[str(hit.chunk.source_url)],
                    score=hit.score,
                    cited=hit.chunk.chunk_id in cited,
                )
                for hit in retrieval.hits
            ],
            excluded_scope_counts=retrieval.excluded_scope_counts,
        )
    return result


def _coverage_status(status: str) -> str:
    return {
        "evidence_found": "complete",
        "documentation_silent": "documentation_silent",
        "search_exhausted": "discovery_failed",
        "not_investigated": "incomplete",
    }[status]


def _documented_value(finding: Any) -> Any | None:
    return finding.value if finding.status == "documented" else None


def _section_status(findings: list[Any]) -> str:
    statuses = [finding.status for finding in findings]
    if "conflicting" in statuses:
        return "conflicting"
    if all(status == "not_applicable" for status in statuses):
        return "not_applicable"
    documented = sum(status == "documented" for status in statuses)
    if documented == len(statuses):
        return "documented"
    if documented:
        return "partial"
    if all(status in {"requires_probe", "not_applicable"} for status in statuses):
        return "probe_required"
    return "partial"
