"""Command-line entry point for HPC Policy Scout."""

from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from agent import HPCPolicyScoutAgent
from discovery import derive_site_identity
from models import ScoutError
from providers.anthropic_provider import AnthropicProvider
from providers.gemini_provider import GeminiProvider
from providers.openai_provider import OpenAIProvider
from tools import ScoutTools


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Discover official HPC documentation for required Slurm submission "
            "options and networking policy."
        )
    )
    parser.add_argument("--site", required=True, help="Human-readable HPC site name.")
    parser.add_argument(
        "--keyword",
        action="append",
        required=True,
        help="Initial search keyword or phrase. Repeat this option as needed.",
    )
    parser.add_argument(
        "--allowed-domain",
        action="append",
        required=True,
        help="Approved documentation domain. Repeat this option as needed.",
    )
    parser.add_argument(
        "--site-alias",
        action="append",
        default=[],
        help="Additional target-site alias. Repeat as needed.",
    )
    parser.add_argument(
        "--preferred-path-token",
        action="append",
        default=[],
        help="URL path token identifying the target site. Repeat as needed.",
    )
    parser.add_argument(
        "--exclude-site-token",
        action="append",
        default=[],
        help="Sibling-site token to reject. Repeat as needed.",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "gemini"],
        default="openai",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", "gpt-5.5"),
        help="Provider model identifier.",
    )
    parser.add_argument(
        "--api-timeout",
        type=float,
        default=90.0,
        help="Timeout in seconds for each model request; default 90.",
    )
    parser.add_argument(
        "--api-max-retries",
        type=int,
        default=0,
        help="SDK retries for transient model-request failures; default 0.",
    )
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--search-results", type=int, default=8)
    parser.add_argument(
        "--search-budget",
        type=int,
        default=8,
        help="Maximum deterministic web-search requests; default 8.",
    )
    parser.add_argument(
        "--page-budget",
        type=int,
        default=8,
        help="Maximum uncached page-fetch requests; default 8.",
    )
    parser.add_argument("--max-page-chars", type=int, default=20_000)
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for generated discovery and site-policy JSON files.",
    )
    parser.add_argument(
        "--output",
        "--discovery-output",
        dest="discovery_output",
        help="Detailed discovery-report JSON path.",
    )
    parser.add_argument(
        "--site-policy-output",
        help="Compact candidate site-policy JSON path.",
    )
    return parser.parse_args()


def build_provider(
    name: str,
    model: str,
    *,
    api_timeout: float = 90.0,
    api_max_retries: int = 0,
):
    if name == "openai":
        return OpenAIProvider(
            model=model,
            timeout=api_timeout,
            max_retries=api_max_retries,
        )
    if name == "anthropic":
        return AnthropicProvider(model=model)
    if name == "gemini":
        return GeminiProvider(model=model)
    raise ValueError(f"Unsupported provider: {name}")


def slugify(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    return value.strip("-") or "site"


def resolve_output_paths(
    args: argparse.Namespace,
    *,
    site_id: str,
    timestamp: datetime | None = None,
) -> tuple[Path, Path]:
    run_timestamp = timestamp or datetime.now().astimezone()
    filename_prefix = f"{site_id}-{run_timestamp.strftime('%Y%m%d-%H%M%S')}"
    output_dir = Path(args.output_dir)
    discovery_output = (
        Path(args.discovery_output)
        if args.discovery_output
        else output_dir / f"{filename_prefix}.discovery-report.json"
    )
    site_policy_output = (
        Path(args.site_policy_output)
        if args.site_policy_output
        else output_dir / f"{filename_prefix}.site-policy.json"
    )
    return discovery_output, site_policy_output


def main() -> int:
    load_dotenv()
    args = parse_args()

    try:
        if args.api_timeout <= 0:
            raise ValueError("--api-timeout must be greater than zero.")
        if args.api_max_retries < 0:
            raise ValueError("--api-max-retries cannot be negative.")
        provider = build_provider(
            args.provider,
            args.model,
            api_timeout=args.api_timeout,
            api_max_retries=args.api_max_retries,
        )
        site_identity = derive_site_identity(
            display_name=args.site,
            organization_domains=args.allowed_domain,
            aliases=args.site_alias,
            preferred_path_tokens=args.preferred_path_token or None,
            excluded_site_tokens=args.exclude_site_token,
        )
        tools = ScoutTools(
            allowed_domains=args.allowed_domain,
            site_identity=site_identity,
            search_results=args.search_results,
            max_page_chars=args.max_page_chars,
            search_budget=args.search_budget,
            page_budget=args.page_budget,
        )
        agent = HPCPolicyScoutAgent(
            provider=provider,
            tools=tools,
            max_steps=args.max_steps,
            log_dir=args.log_dir,
        )
        site_id = slugify(args.site)
        discovery_output, site_policy_output = resolve_output_paths(
            args,
            site_id=site_id,
        )
        discovery_report_reference = os.path.relpath(
            discovery_output,
            start=site_policy_output.parent,
        )
        artifacts = agent.run(
            site_name=args.site,
            site_id=site_id,
            discovery_report_reference=discovery_report_reference,
            keywords=args.keyword,
            allowed_domains=args.allowed_domain,
        )

        discovery_output.parent.mkdir(parents=True, exist_ok=True)
        site_policy_output.parent.mkdir(parents=True, exist_ok=True)
        discovery_output.write_text(
            artifacts.discovery_report.model_dump_json(indent=2),
            encoding="utf-8",
        )
        site_policy_output.write_text(
            artifacts.site_policy.model_dump_json(indent=2),
            encoding="utf-8",
        )

        print(artifacts.site_policy.model_dump_json(indent=2))
        print(f"\nSaved discovery report: {discovery_output}", file=sys.stderr)
        print(f"Saved site policy: {site_policy_output}", file=sys.stderr)
        print(f"Saved trace: {agent.logger.path}", file=sys.stderr)
        return 0

    except (ScoutError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
