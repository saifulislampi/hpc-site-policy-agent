import sys

from main import parse_args


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

