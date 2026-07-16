"""Named extraction profiles and their bounded model-call groups."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

from schemas import NetworkExtraction, OperationalExtraction, SubmissionExtraction


ExtractionProfile = Literal["site-policy", "evaluation-full"]


@dataclass(frozen=True)
class ExtractionGroup:
    name: str
    fields: tuple[str, ...]
    schema: type[BaseModel]


SUBMISSION_GROUP = ExtractionGroup(
    name="submission",
    fields=(
        "scheduler",
        "submit_command",
        "submission_options",
        "account_allocation_policy",
        "default_partition",
        "partitions",
    ),
    schema=SubmissionExtraction,
)

NETWORK_GROUP = ExtractionGroup(
    name="network",
    fields=(
        "manager_worker_connectivity",
        "worker_worker_connectivity",
        "published_port_range",
        "manager_address_guidance",
        "outbound_compute_network",
    ),
    schema=NetworkExtraction,
)

OPERATIONAL_GROUP = ExtractionGroup(
    name="operational",
    fields=(
        "walltime_policy",
        "memory_policy",
        "job_size_policy",
        "charging_model",
        "purge_policy",
        "cost_traps",
        "login_node_socket_policy",
    ),
    schema=OperationalExtraction,
)

PROFILE_GROUPS: dict[ExtractionProfile, tuple[ExtractionGroup, ...]] = {
    "site-policy": (SUBMISSION_GROUP, NETWORK_GROUP),
    "evaluation-full": (SUBMISSION_GROUP, NETWORK_GROUP, OPERATIONAL_GROUP),
}

ALL_POLICY_FIELDS = tuple(
    dict.fromkeys(
        field
        for group in (SUBMISSION_GROUP, NETWORK_GROUP, OPERATIONAL_GROUP)
        for field in group.fields
    )
)


def groups_for_profile(profile: ExtractionProfile) -> tuple[ExtractionGroup, ...]:
    return PROFILE_GROUPS[profile]


def fields_for_profile(profile: ExtractionProfile) -> set[str]:
    return {field for group in groups_for_profile(profile) for field in group.fields}
