import sys
from datetime import datetime, timezone

from main import parse_args, resolve_output_paths


def test_output_and_log_directories_have_separate_defaults(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--site",
            "Purdue Anvil",
            "--keyword",
            "Anvil Slurm",
            "--allowed-domain",
            "purdue.edu",
        ],
    )

    args = parse_args()

    assert args.output_dir == "outputs"
    assert args.log_dir == "logs"


def test_default_output_names_share_run_timestamp(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--site",
            "Purdue Anvil",
            "--keyword",
            "Anvil Slurm",
            "--allowed-domain",
            "purdue.edu",
            "--output-dir",
            "results",
        ],
    )
    args = parse_args()

    discovery, policy = resolve_output_paths(
        args,
        site_id="purdue-anvil",
        timestamp=datetime(2026, 7, 15, 20, 5, 6, tzinfo=timezone.utc),
    )

    assert discovery.as_posix() == (
        "results/purdue-anvil-20260715-200506.discovery-report.json"
    )
    assert policy.as_posix() == (
        "results/purdue-anvil-20260715-200506.site-policy.json"
    )


def test_explicit_output_names_are_preserved(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--site",
            "Purdue Anvil",
            "--keyword",
            "Anvil Slurm",
            "--allowed-domain",
            "purdue.edu",
            "--discovery-output",
            "custom/report.json",
            "--site-policy-output",
            "custom/policy.json",
        ],
    )
    args = parse_args()

    discovery, policy = resolve_output_paths(args, site_id="purdue-anvil")

    assert discovery.as_posix() == "custom/report.json"
    assert policy.as_posix() == "custom/policy.json"
