import pytest
from pydantic import ValidationError

from schemas import PortRangeFinding, StringFinding


def test_documented_finding_requires_evidence():
    with pytest.raises(ValidationError):
        StringFinding(
            value="--account",
            status="documented",
            documentation_status="documented",
            confidence=0.9,
            explanation="The option is required.",
            evidence=[],
        )


def test_requires_probe_must_have_null_value():
    with pytest.raises(ValidationError):
        PortRangeFinding(
            value=[{"protocol": "tcp", "start": 35000, "end": 40000}],
            status="requires_probe",
            documentation_status="silent",
            confidence=0.2,
            explanation="No documentation was found.",
            evidence=[],
        )


def test_undocumented_network_fact_remains_requires_probe():
    finding = PortRangeFinding(
        value=None,
        status="requires_probe",
        documentation_status="silent",
        confidence=0.9,
        explanation="Target-site architecture pages publish no TCP port policy.",
        evidence=[],
    )

    assert finding.value is None
    assert finding.documentation_status == "silent"


def test_inferred_evidence_cannot_establish_definitive_policy():
    with pytest.raises(ValidationError, match="direct or conflicting"):
        StringFinding(
            value="ports 30000-40000",
            status="documented",
            documentation_status="documented",
            confidence=0.8,
            explanation="An inferred range is not definitive evidence.",
            evidence=[
                {
                    "chunk_id": "C_architecture",
                    "source_url": "https://docs.example.edu/site/architecture/",
                    "source_title": "Architecture",
                    "heading": "Network Architecture",
                    "quote": "The nodes use a high-speed fabric.",
                    "interpretation": "inferred",
                }
            ],
        )
