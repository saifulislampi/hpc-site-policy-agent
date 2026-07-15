# Corpus and retrieval design

The corpus is persistent; its retrieval index is not. Each run merges canonical
web documents and zero-overlap chunks into JSONL, then builds a transient
in-process BM25 index over the complete stored site corpus. This lets later model runs reuse downloaded text without
adding an embedding service, vector database, GPU, or model-specific format.

`site_scope` and `trust_level` answer different questions. Scope controls whether
a chunk is eligible for a target-site retrieval. Trust records how the source
entered the corpus. Scope is assigned deterministically from the URL path and
site identity; the model cannot assign or override it.

Sibling pages are deliberately retained as negative controls. Retrieval filters
them before scoring and logs both retained and retrieved counts. A healthy run
can therefore show `14 sibling chunks, 0 retrieved`, which is stronger evidence
than counting rejected URLs.

Tables are one Markdown chunk even when larger than the ordinary character
limit. Policies and FAQ sections also remain whole so charging, purge, and cost
cautions survive even when they do not match obvious keywords. All other chunks
have zero overlap so a quoted substring maps to exactly one stored chunk.

`--refresh-corpus` is merge-safe: a rediscovered changed page replaces its old
version, but a stored web page is never deleted merely because the current
search did not rediscover it. Without refresh, a changed fetched representation
does not silently replace the stored canonical version.

Local/PDF ingestion and hybrid retrieval are intentionally deferred. When hybrid
sources are added, deduplication alone is insufficient. The conflict rule will
be: identical normalized content may coalesce; differing content from web and
local sources is retained as separate versions, marked conflicting, and neither
silently overrides the other. An explicit manifest declaration may choose the
preferred version, while trust and scope remain independent metadata.
