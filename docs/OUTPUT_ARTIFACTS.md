# Output artifacts

Every completed or partial run writes two versioned JSON artifacts. They are built
deterministically from one provider-neutral extraction result; providers do not
generate these files independently.

## Detailed discovery report

Default filename: `outputs/<site-id>-<YYYYMMDD-HHMMSS>.discovery-report.json`

Audience: researchers, debugging, provenance review, and paper evaluation.

It contains:

- provider, model, run ID, and timestamp;
- canonical root, every query, coverage, and termination reason;
- stable source IDs with independent scope, trust, selected, and retrieved state;
- corpus ID/fingerprint and field-specific retrieved chunk scores;
- flattened findings with status, documentation status, confidence, and quoted
  evidence referencing source IDs and exact chunk IDs;
- retrieved-but-uncited chunks, so abstention can be distinguished from a
  retrieval miss;
- unresolved questions and request/token statistics.
- extraction profile, `profile_state`, requested and uninvestigated fields,
  verified/null/failed field counts, correction attempts, and group errors.

This file may change within the `0.1` research schema as evaluation needs grow.

## Candidate site policy

Default filename: `outputs/<site-id>-<YYYYMMDD-HHMMSS>.site-policy.json`

Audience: Floability, site-profile generators, validators, and probe runners.

It contains only values Floability can consume or a profiler can validate:

- `profile_state`, which is `partial` whenever a requested field remains
  unverified;

- scheduler type and submission command;
- every documented submission option with a semantic name, all documented
  syntax forms, a boolean `required`, example, and actual/default value;
- default partition and per-partition limits;
- normalized network values;
- explicit null storage placeholders;
- section-level validation state.
- a small provenance block containing the discovery-report path, run ID, corpus
  ID/fingerprint, and
  JSON Pointer references from policy values to detailed findings.

Detailed field status, evidence, source IDs, and unresolved questions remain
exclusively in the discovery report. The site policy stores references only.

Example:

```json
{
  "provenance": {
    "discovery_report": "purdue-anvil-20260715-190000.discovery-report.json",
    "run_id": "uuid",
    "corpus_id": "purdue-anvil",
    "corpus_fingerprint": "sha256:...",
    "references": {
      "/submission/options": "/findings/submission.options"
    }
  }
}
```

## Stability rules

- Never insert guessed values into the site policy.
- Include only submission syntax explicitly shown in selected target-site pages.
- Keep documented examples separate from actual/default `value` fields.
- Set `required=true` only from explicit mandatory or minimum-field wording;
  appearing in an example script is insufficient.
- An example allocation name must never become the user's allocation value.
- Keep partition choices in `partitions.limits`; do not duplicate them in an
  option-level allowed-values field.
- A field requiring a probe has a null site-policy value.
- A timed-out or invalid extraction group does not discard other results. Its
  fields are null in the site policy and marked `extraction_failed` in the
  detailed report.
- A field omitted by the chosen extraction profile is `not_investigated`, not
  documentation silence.
- Sibling-site pages may be retained in the corpus and detailed report, but are
  excluded by the retrieval scope parameter and can never be cited.
- Increment `schema_version` when a consumer-facing field is removed, renamed,
  or changes type.
