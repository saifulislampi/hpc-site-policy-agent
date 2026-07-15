# Output artifacts

Every successful run writes two versioned JSON artifacts. They are built
deterministically from one provider-neutral extraction result; providers do not
generate these files independently.

## Detailed discovery report

Default filename: `outputs/<site-id>.discovery-report.json`

Audience: researchers, debugging, provenance review, and paper evaluation.

It contains:

- provider, model, run ID, and timestamp;
- canonical root, every query, coverage, and termination reason;
- stable source IDs for selected and rejected candidates;
- flattened findings with status, documentation status, confidence, and quoted
  evidence referencing source IDs;
- unresolved questions and request/token statistics.

This file may change within the `0.1` research schema as evaluation needs grow.

## Candidate site policy

Default filename: `outputs/<site-id>.site-policy.json`

Audience: Floability, site-profile generators, validators, and probe runners.

It deliberately separates:

1. `profile`: actual normalized values only;
2. `field_status`: documentation state and discovery-report source IDs;
3. `required_probes`: unresolved operational checks with JSON-pointer targets;
4. `provenance`: the detailed report filename and run ID.

`profile_state="candidate"` means documentation extraction is not equivalent to
runtime validation. A later validator may promote the artifact after required
probes succeed.

## Stability rules

- Never insert guessed values into `profile`.
- A `requires_probe` field has a null profile value.
- A documented field references direct evidence through source IDs.
- Sibling-site pages may appear only as rejected detailed-report sources.
- Increment `schema_version` when a consumer-facing field is removed, renamed,
  or changes type.
