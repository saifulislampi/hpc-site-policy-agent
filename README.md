# HPC Policy Scout

The complete current agentic RAG execution flow is maintained in
[`docs/CURRENT_PIPELINE.txt`](docs/CURRENT_PIPELINE.txt). It is updated whenever
the pipeline changes.

The persistent-corpus and retrieval rules are explained in
[`docs/CORPUS_RAG.md`](docs/CORPUS_RAG.md).

The two output contracts are described in
[`docs/OUTPUT_ARTIFACTS.md`](docs/OUTPUT_ARTIFACTS.md), and the Claude/Gemini
adapter plan is in [`docs/PROVIDER_ADAPTERS.md`](docs/PROVIDER_ADAPTERS.md).

HPC Policy Scout is a bounded, site-aware agentic RAG system for onboarding an unfamiliar HPC site. From a site name, search keywords, and approved domains, it discovers authoritative pages, maintains a reusable text corpus, retrieves evidence per policy field, and produces structured JSON covering:

- Slurm submission requirements and policy: account/allocation, partition or queue, walltime, memory, job-size, and charging notes.
- Networking information relevant to manager/worker systems: compute-to-login, worker-to-worker, published port ranges, manager address guidance, and outbound access.
- Explicit unknowns marked `requires_probe` when documentation is silent.

Discovery is deterministic where it matters: it derives a target-site identity,
locates a canonical guide root, crawls topic links, and assigns source scope from
URL paths rather than model judgment. Sibling pages are retained as measurable
negative controls but excluded before retrieval scoring. Scope and trust are
stored independently.

The corpus persists canonical documents and chunks as JSONL. Tables stay atomic
Markdown, Policies/FAQ sections stay whole, and ordinary chunks use zero
overlap. A transient CPU-only BM25 index retrieves evidence independently for
each output field; no vector database, embeddings, GPU, or new Python package is
required.

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
  --max-steps 2 \
  --corpus-dir corpora/purdue-anvil
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
--corpus-dir DIR          Persistent canonical documents/chunks directory
--refresh-corpus          Replace changed rediscovered pages; retain unseen pages
--chunk-chars N           Ordinary text chunk limit; default 1800
--retrieval-top-k N       Chunks per ordinary extraction field; default 5
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
  run metadata, source scope/trust, corpus fingerprint, per-field retrieval
  scores, retrieved-but-uncited chunks, exact chunk evidence, coverage, and
  statistics.
- `outputs/<site-id>-<timestamp>.site-policy.json`: the small candidate operational profile with
  normalized scheduler values, every documented submission option with exact
  syntax, partition limits, network values, storage placeholders, and
  section-level validation state. Evidence and source metadata remain in the
  discovery report; the policy stores its run ID, report path, corpus
  ID/fingerprint, and JSON Pointer links to detailed findings.
- A JSONL execution trace recording canonical-root selection, generated and repaired queries, candidate rankings, classifications, followed links, selected/rejected pages, request counts, token usage, and termination reason.

Use `--output-dir` to place both JSON artifacts elsewhere. Explicit
`--discovery-output` and `--site-policy-output` paths override that directory.
The default timestamp format is `YYYYMMDD-HHMMSS`, and both artifacts from a run
receive the same timestamp.

The CLI also prints concise, timestamped progress to stderr. Long model calls are announced before waiting and are bounded by `--api-timeout`; detailed diagnostics remain in the JSONL trace.

Only fetched `target_site` pages and explicitly applicable
`organization_general` pages are eligible retrieval scopes. A sibling page can
be stored and counted, but it cannot be retrieved or cited.

Run tests with:

```bash
pytest -q
```

The OpenAI provider is implemented. Claude and Gemini adapters are isolated
placeholders that will implement the same `BaseProvider` contract and return the
same `ExtractedPolicy`; no discovery or reporting logic will be duplicated.
