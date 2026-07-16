# Working example: Purdue Anvil

This document follows one normal successful run from the CLI command through
the generated discovery report and compact site policy. Values and identifiers
are illustrative; run IDs, timestamps, search results, scores, and chunk IDs
will differ between runs.

## Command

```bash
python main.py \
  --site "Purdue Anvil" \
  --keyword "Anvil RCAC Slurm job submission account partition" \
  --keyword "Anvil compute node login node networking TCP ports" \
  --allowed-domain purdue.edu \
  --provider openai \
  --model gpt-5-mini \
  --max-steps 2 \
  --api-timeout 180 \
  --search-budget 12 \
  --page-budget 16 \
  --retrieval-top-k 3 \
  --corpus-dir corpora/purdue-anvil \
  --refresh-corpus
```

The single `--model` is used for both discovery decisions and structured
extraction. Search, scope assignment, fetching, chunking, retrieval, validation,
and artifact construction are performed by Python.

## End-to-end flow

```text
CLI configuration
  → deterministic site identity
  → broad documentation discovery
  → bounded discovery-agent selection
  → HTML section and table extraction
  → persistent document/chunk corpus
  → transient field-specific retrieval
  → one structured model extraction
  → local evidence resolution and validation
  → detailed discovery report
  → compact site policy
```

## 1. Parse the run configuration

[`main.py`](../main.py) loads `.env`, parses the arguments, creates the provider,
and calculates the site ID:

```text
site name:       Purdue Anvil
site ID:         purdue-anvil
provider:        openai
model:           gpt-5-mini
approved domain: purdue.edu
corpus:          corpora/purdue-anvil
```

When output paths are not supplied, the timestamp is added automatically:

```text
outputs/purdue-anvil-20260715-201500.discovery-report.json
outputs/purdue-anvil-20260715-201500.site-policy.json
```

## 2. Derive a deterministic site identity

[`discovery.py`](../discovery.py) derives aliases and preferred URL tokens. A
typical identity is conceptually:

```json
{
  "display_name": "Purdue Anvil",
  "aliases": ["Purdue Anvil", "Anvil"],
  "organization_domains": ["purdue.edu"],
  "preferred_path_tokens": ["anvil"],
  "excluded_site_tokens": []
}
```

This identity, rather than the model, controls site scope.

## 3. Search broadly for documentation

[`discovery.py`](../discovery.py) generates bounded searches in four groups:

```text
canonical documentation
submission, Slurm, accounts, and partitions
networking, login nodes, firewalls, and ports
FAQ, charging, accounting, purge, retention, and storage policy
```

The two user-provided `--keyword` values are also searched if budget remains.
[`tools.py`](../tools.py) executes the searches only against approved domains.

Example results might include:

```text
https://docs.rcac.purdue.edu/userguides/anvil/jobs/
https://docs.rcac.purdue.edu/userguides/anvil/faqs/
https://docs.rcac.purdue.edu/userguides/anvil/architecture/
https://docs.rcac.purdue.edu/userguides/bell/run_jobs/
```

## 4. Assign scope and trust locally

Every candidate is classified from its URL and the site identity:

```text
/userguides/anvil/jobs/       → target_site
/userguides/anvil/faqs/       → target_site
/userguides/bell/run_jobs/    → sibling
/faqs/                        → organization_general
```

Scope and trust are independent:

```json
{
  "site_scope": "target_site",
  "trust_level": "official_web"
}
```

The model cannot promote a sibling page into target scope. Scope later becomes
an input to retrieval, not merely a descriptive tag.

## 5. Fetch several useful pages

[`tools.py`](../tools.py) ranks URL and title evidence above search snippets. It
fetches several pages for each documentation group rather than relying on one
page.

For each fetch it:

1. validates HTTPS, port, and approved domain;
2. follows redirects;
3. extracts readable text and links;
4. prefers article content over navigation chrome;
5. verifies that submission/networking candidates contain topic evidence;
6. continues to another candidate after a failure or topic mismatch.

A fetched sibling page may be retained as a negative control. It is stored, but
will not be eligible for target-site retrieval.

## 6. Complete the bounded discovery-agent loop

[`agent.py`](../agent.py) sends the model the deterministic discovery context.
The model can call only these narrow tools:

```text
search_web
fetch_page
finish_discovery
```

With `--max-steps 2`, the model receives at most two discovery turns. Python
validates every selected URL. A normal completion may select the Anvil jobs,
FAQ, and architecture pages while recording networking ports as unanswered.

The model is selecting among already classified and fetched evidence. It is not
allowed to override URL scope or select an unfetched page.

## 7. Extract structured document sections

[`tools.py`](../tools.py) converts fetched HTML into sections with heading paths
and typed blocks:

```json
{
  "heading_path": ["Job Submission on Anvil", "Anvil Queues (Partitions)"],
  "blocks": [
    {
      "kind": "table",
      "text": "| Queue Name | Max Nodes per Job | Max Duration |\n| --- | --- | --- |\n| shared | 1 | 96 hrs |"
    }
  ]
}
```

Tables remain Markdown so values such as `shared`, `1`, and `96 hrs` stay in the
same evidence unit.

## 8. Build and merge the persistent corpus

[`corpus/chunking.py`](../corpus/chunking.py) creates provider-neutral document
and chunk records:

- ordinary text is split at paragraph, sentence, or word boundaries;
- ordinary chunks have `overlap_chars=0`;
- tables are atomic and are never split;
- Policy and FAQ sections remain whole;
- every document and chunk receives a content hash;
- PDFs are outside the current corpus path.

An illustrative partition chunk is:

```json
{
  "chunk_id": "C_123example",
  "source_url": "https://docs.rcac.purdue.edu/userguides/anvil/jobs/",
  "site_scope": "target_site",
  "trust_level": "official_web",
  "heading_path": ["Job Submission on Anvil", "Anvil Queues (Partitions)"],
  "block_type": "table",
  "text": "| Queue Name | Max Nodes per Job | Max Duration | ..."
}
```

[`corpus/store.py`](../corpus/store.py) merges the records into:

```text
corpora/purdue-anvil/manifest.json
corpora/purdue-anvil/documents.jsonl
corpora/purdue-anvil/chunks.jsonl
```

`--refresh-corpus` replaces changed pages that were rediscovered. It does not
delete a stored page merely because the current search did not rediscover it.
There is no persisted retrieval index.

## 9. Retrieve evidence separately for every field

[`corpus/retrieval.py`](../corpus/retrieval.py) constructs a temporary in-memory
BM25 index. Each output field has several query variants.

For `partitions`, the variants include concepts such as:

```text
Slurm partitions queues limits walltime
shared wholenode wide highmem gpu ai debug queue
partition maximum nodes maximum duration charging factor
```

Before scoring, retrieval permits only:

```text
target_site
organization_general
```

It excludes `sibling` and `unrelated` chunks before ranking. It also:

- deduplicates byte-identical chunk content;
- penalizes navigation-menu fragments;
- requires explicit `partition` or `queue` language for partition findings;
- prevents hardware-only tables from establishing scheduler/network policy;
- retains whole Policy/FAQ sections for charging, purge, and cost-trap fields.

An audit line such as this is therefore meaningful:

```text
12 sibling chunks retained, 0 sibling chunks retrieved
```

## 10. Construct the extraction context

[`prompts.py`](../prompts.py) groups results by field. Global corpus chunk IDs are
not shown to the model. Each result receives a field-local reference:

```text
--- FIELD partitions RETRIEVAL MAP START ---
EVIDENCE REF: partitions:R1
GLOBAL CHUNK: G7
SCORE: 9.42
--- FIELD partitions RETRIEVAL MAP END ---

--- GLOBAL EVIDENCE LIBRARY START ---
--- GLOBAL CHUNK G7 START ---
URL: https://docs.rcac.purdue.edu/userguides/anvil/jobs/
HEADING PATH: Job Submission on Anvil > Anvil Queues (Partitions)
CHUNK TEXT:
| Queue Name | Max Nodes per Job | Max Duration |
| --- | --- | --- |
| shared | 1 | 96 hrs |
--- GLOBAL CHUNK G7 END ---
--- GLOBAL EVIDENCE LIBRARY END ---
```

If several fields retrieve the same canonical chunk, each field receives its
own local reference but the potentially long chunk text appears only once in
the global evidence library.

Other fields receive independent namespaces such as:

```text
submission_options:R1
default_partition:R1
walltime_policy:R1
published_port_range:R1
```

## 11. Request structured extraction

[`providers/openai_provider.py`](../providers/openai_provider.py) sends the
field-grouped context to the single CLI-selected model. The required output is
validated against [`schemas/extraction.py`](../schemas/extraction.py).

An illustrative finding returned by the model is:

```json
{
  "value": "shared",
  "status": "documented",
  "documentation_status": "documented",
  "confidence": 0.99,
  "explanation": "The job guide identifies shared as the default partition.",
  "evidence": [
    {
      "chunk_id": "default_partition:R1",
      "source_url": "https://docs.rcac.purdue.edu/userguides/anvil/jobs/",
      "source_title": "Job Submission - RCAC Documentation",
      "heading": "Default Partition",
      "quote": "If no partition is specified, the job will be directed into the default partition (`shared`).",
      "interpretation": "direct"
    }
  ]
}
```

When documentation does not establish an operational fact, the expected result
is an abstention:

```json
{
  "value": null,
  "status": "requires_probe",
  "documentation_status": "silent",
  "confidence": 1.0,
  "explanation": "No target-site documentation publishes an application TCP port range.",
  "evidence": []
}
```

## 12. Resolve and validate evidence locally

[`agent.py`](../agent.py) translates `default_partition:R1` into its canonical
stored chunk ID and fills the canonical URL, title, and heading from the corpus.
It then enforces:

```text
the reference belongs to the finding's field
the canonical chunk was retrieved for that field
the evidence URL matches the chunk URL
the quote is a literal substring of the chunk text
documented/conflicting findings have direct evidence
requires_probe findings have value=null
```

The detailed trace records every retrieved score and whether the chunk was
cited. This distinguishes “the model abstained after seeing relevant chunks”
from “retrieval never found the evidence.”

## 13. Construct both artifacts locally

[`reporting.py`](../reporting.py) converts the validated extraction into two
different outputs.

The detailed discovery report contains research and evaluation information:

```json
{
  "run": {
    "provider": "openai",
    "model": "gpt-5-mini"
  },
  "corpus": {
    "corpus_id": "purdue-anvil",
    "corpus_fingerprint": "sha256:..."
  },
  "retrieval": {
    "partitions": {
      "chunks": [
        {
          "chunk_id": "C_123example",
          "source_id": "S1",
          "score": 9.42,
          "cited": true
        }
      ]
    }
  },
  "findings": {
    "submission.default_partition": {
      "value": "shared",
      "status": "documented",
      "evidence": [
        {
          "source_id": "S1",
          "chunk_id": "C_123example",
          "quote": "If no partition is specified, the job will be directed into the default partition (`shared`)."
        }
      ]
    }
  }
}
```

The compact site policy contains consumer-ready values rather than copied
evidence:

```json
{
  "site": {
    "id": "purdue-anvil",
    "name": "Purdue Anvil"
  },
  "scheduler": {
    "type": "slurm",
    "submit_command": "sbatch"
  },
  "submission": {
    "options": [
      {
        "name": "account",
        "syntax": ["-A {account}", "--account={account}"],
        "required": true,
        "example": "-A myallocation",
        "value": null
      }
    ]
  },
  "partitions": {
    "default": "shared",
    "limits": {}
  },
  "provenance": {
    "discovery_report": "purdue-anvil-....discovery-report.json",
    "run_id": "...",
    "corpus_id": "purdue-anvil",
    "corpus_fingerprint": "sha256:...",
    "references": {
      "/partitions/default": "/findings/submission.default_partition"
    }
  }
}
```

The model does not write either artifact directly.

## 14. Save outputs and trace

[`main.py`](../main.py) writes both timestamped artifacts. [`agent.py`](../agent.py)
writes a JSONL trace containing:

```text
queries and candidate rankings
scope/trust classifications
page fetches and failures
discovery model turns and tool calls
corpus fingerprint
retrieval results and scores
retrieved-but-uncited chunks
validated evidence references
request and token statistics
final artifacts
```

The terminal prints only concise progress. The JSON outputs and JSONL trace hold
the detailed reproducibility information.

## Responsibility summary

| Responsibility | Implemented by |
| --- | --- |
| CLI and output paths | [`main.py`](../main.py) |
| Site identity, queries, deterministic scope | [`discovery.py`](../discovery.py) |
| Search, fetch, HTML sections, coverage | [`tools.py`](../tools.py) |
| Agent loop and evidence validation | [`agent.py`](../agent.py) |
| Chunk construction | [`corpus/chunking.py`](../corpus/chunking.py) |
| Persistent corpus merge | [`corpus/store.py`](../corpus/store.py) |
| Transient BM25 retrieval | [`corpus/retrieval.py`](../corpus/retrieval.py) |
| Model instructions/context | [`prompts.py`](../prompts.py) |
| OpenAI API translation | [`providers/openai_provider.py`](../providers/openai_provider.py) |
| Provider-neutral data contracts | [`schemas/`](../schemas/) |
| Discovery report and site policy | [`reporting.py`](../reporting.py) |
