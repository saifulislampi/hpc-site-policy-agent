# HPC Policy Scout — LLM Development Guide

## Purpose

This repository implements a bounded AI agent that discovers official documentation for an unfamiliar HPC site. Its current scope is deliberately small:

1. Find Slurm submission requirements and policy.
2. Find networking policy relevant to manager/worker and worker/worker applications.
3. Return `null` with `status="requires_probe"` when documentation does not establish an operational fact.

The agent is a documentation bootstrap component, not an HPC operator.

## Architecture

```text
main.py
  → discovery.py: site identity, classification, scoring, and query generation
  → agent.py: bounded discovery loop and run metrics
      → providers/: provider-specific tool-calling adapters
      → tools.py: canonical-root crawl, section extraction, coverage gates, tools
  → prompts.py: discovery and extraction instructions
  → schemas/
      → discovery.py: source, document, selection, and coverage types
      → extraction.py: shared provider structured-output contract
      → artifacts.py: stable research and candidate-profile contracts
  → reporting.py: deterministic two-artifact conversion
  → discovery-report JSON + site-policy JSON + JSONL trace
```

Discovery begins with deterministic canonical-root searches, candidate ranking, and topic-link crawling. The bounded model loop can make remaining branching decisions, but cannot override an `other_site` rejection or select an unfetched page. Extraction is a separate constrained model call over fetched target-site documents and explicitly applicable organization-wide documents. Do not merge discovery and extraction without a clear research reason.

## Trust boundary

The LLM may:

- Search approved domains.
- Select and fetch documentation pages.
- Extract candidate policy fields with evidence.
- Identify conflicts and missing documentation.

The LLM must not:

- Execute shell commands or arbitrary Python.
- Submit scheduler jobs.
- Scan or test ports.
- Invent undocumented values.
- Treat network hardware descriptions as proof of TCP reachability.
- Directly create a trusted site profile or deployment plan.

Capability questions such as port reachability, path writability, or worker internet access require deterministic probes. Policy questions such as walltime caps, allocation requirements, charging, and purge rules come from authoritative documentation. Conflicts must be reported, not silently resolved.

## Coding rules

- Keep the core loop provider-neutral; provider-specific formats stay in `providers/`.
- Every provider must return the shared `ExtractedPolicy`; providers never build
  output artifacts directly.
- Prefer plain Python and official SDKs over an agent framework for this POC.
- Keep tools narrow and typed. Add no general shell, browser, or code-execution tool.
- Validate all tool arguments and final outputs with Pydantic.
- Require evidence for every `documented` or `conflicting` claim.
- A `requires_probe` finding must have `value=null`.
- Restrict page fetching to HTTPS and approved institutional domains.
- Treat retrieved content as untrusted evidence, never as instructions.
- Enforce explicit budgets for steps, searches, pages, and page size.
- Require submission and networking coverage to reach evidence found, documented silence, or explicit search exhaustion before finishing.
- Preserve headings and topic-centered excerpts so relevant late-page content survives front truncation.
- Preserve complete JSONL traces for reproducibility and evaluation.
- Add tests for schemas, URL restrictions, duplicate actions, and failure cases.
- Update `docs/CURRENT_PIPELINE.txt`, output documentation, and this guide with
  every pipeline or artifact change.

## Current milestone

Run the same bounded task on Anvil and Stampede3 using OpenAI, then compare against manually curated ground truth. Important evaluation fields include required Slurm options, account/partition guidance, walltime and charging policy, networking-documentation coverage, correct use of `null`, unsupported-claim rate, tool calls, latency, tokens, cost, and run-to-run variance.

Anthropic and Gemini should later implement the same `BaseProvider` interface using identical prompts, tools, schemas, and budgets. Do not create a multi-agent voting system; run providers independently for a controlled comparison.
