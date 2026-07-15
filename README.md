# HPC Policy Scout

The complete current non-RAG execution flow is maintained in
[`docs/CURRENT_PIPELINE.txt`](docs/CURRENT_PIPELINE.txt). It is updated whenever
the pipeline changes.

The two output contracts are described in
[`docs/OUTPUT_ARTIFACTS.md`](docs/OUTPUT_ARTIFACTS.md), and the Claude/Gemini
adapter plan is in [`docs/PROVIDER_ADAPTERS.md`](docs/PROVIDER_ADAPTERS.md).

HPC Policy Scout is a bounded, site-aware documentation agent for onboarding an unfamiliar HPC site. From a site name, search keywords, and approved domains, it finds authoritative documentation and produces a structured JSON report covering:

- Slurm submission requirements and policy: account/allocation, partition or queue, walltime, memory, job-size, and charging notes.
- Networking information relevant to manager/worker systems: compute-to-login, worker-to-worker, published port ranges, manager address guidance, and outbound access.
- Explicit unknowns marked `requires_probe` when documentation is silent.

Discovery is deterministic where it matters: it derives a target-site identity, locates a canonical guide root, crawls topic links, scores every candidate, and rejects sibling-cluster pages before the model sees extraction evidence. It tracks submission and networking coverage separately, so `documentation_status="silent"` means a reasonable target-specific search completed, while `documentation_status="discovery_failed"` means discovery could not establish adequate coverage.

The agent only discovers documentation. It does not execute shell commands, submit jobs, scan ports, or modify an HPC system. Operational facts must later be verified by deterministic probes.

## Setup

```bash
conda env create -f environment.yml
conda activate hpc-policy-scout
cp .env.example .env
```

Create an OpenAI API key in the OpenAI API Platform, then add it to `.env`:

```bash
OPENAI_API_KEY=your-secret-key
OPENAI_MODEL=gpt-5-mini
```

Do not commit `.env` or API keys.

## Run

Purdue Anvil:

```bash
python main.py \
  --site "Purdue Anvil" \
  --keyword "Anvil RCAC Slurm job submission account partition" \
  --keyword "Anvil compute node login node networking TCP ports" \
  --allowed-domain purdue.edu \
  --provider openai \
  --model gpt-5-mini \
  --max-steps 2
```

TACC Stampede3:

```bash
python main.py \
  --site "TACC Stampede3" \
  --keyword "Stampede3 Slurm required options queue allocation walltime" \
  --keyword "Stampede3 compute node login node networking TCP ports" \
  --allowed-domain tacc.utexas.edu \
  --allowed-domain utexas.edu \
  --provider openai
```

Useful options:

```text
--model MODEL            Provider model identifier
--api-timeout SECONDS    Per-model-request timeout; default 90
--api-max-retries N      Transient model-request retries; default 0
--max-steps N            Maximum discovery-agent turns; default 10
--search-results N       Approved search results returned per query
--search-budget N        Maximum deterministic search requests; default 8
--page-budget N          Maximum uncached page requests; default 8
--max-page-chars N       Maximum extracted characters per page
--site-alias NAME        Additional target-site alias; repeatable
--preferred-path-token T Target-site URL path token; repeatable
--exclude-site-token T   Sibling-site token to reject; repeatable
--output FILE            Detailed discovery-report path (legacy alias)
--discovery-output FILE  Detailed discovery-report path
--site-policy-output FILE Compact candidate site-policy path
--output-dir DIR         Generated JSON artifact directory; default outputs
--log-dir DIR            JSONL execution trace directory; default logs
```

Site identity is derived from `--site` by default. For a reproducible Purdue experiment, the optional overrides can be supplied explicitly:

```bash
python main.py \
  --site "Purdue Anvil" \
  --keyword "Anvil Slurm account partition walltime" \
  --keyword "Anvil compute node TCP networking policy" \
  --allowed-domain purdue.edu \
  --site-alias Anvil \
  --preferred-path-token anvil \
  --exclude-site-token bell \
  --exclude-site-token geddes \
  --exclude-site-token gilbreth \
  --exclude-site-token negishi \
  --provider openai
```

## Output

Each run creates:

- `outputs/<site-id>-<timestamp>.discovery-report.json`: the detailed research artifact containing
  run metadata, all selected/rejected source provenance, flattened findings,
  evidence, coverage, and statistics.
- `outputs/<site-id>-<timestamp>.site-policy.json`: the small candidate operational profile with
  normalized scheduler values, every documented submission option with exact
  syntax, partition limits, network values, storage placeholders, and
  section-level validation state. Evidence and source metadata remain in the
  discovery report; the policy stores only its run ID, report path, and JSON
  Pointer links to detailed findings.
- A JSONL execution trace recording canonical-root selection, generated and repaired queries, candidate rankings, classifications, followed links, selected/rejected pages, request counts, token usage, and termination reason.

Use `--output-dir` to place both JSON artifacts elsewhere. Explicit
`--discovery-output` and `--site-policy-output` paths override that directory.
The default timestamp format is `YYYYMMDD-HHMMSS`, and both artifacts from a run
receive the same timestamp.

The CLI also prints concise, timestamped progress to stderr. Long model calls are announced before waiting and are bounded by `--api-timeout`; detailed diagnostics remain in the JSONL trace.

Only fetched `target_site` pages and explicitly applicable `organization_general` pages can reach extraction. A page for another cluster may appear as a rejected discovery-report source, but it cannot be selected or cited.

Run tests with:

```bash
pytest -q
```

The OpenAI provider is implemented. Claude and Gemini adapters are isolated
placeholders that will implement the same `BaseProvider` contract and return the
same `ExtractedPolicy`; no discovery or reporting logic will be duplicated.
