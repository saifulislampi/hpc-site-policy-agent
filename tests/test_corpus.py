from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from agent import HPCPolicyScoutAgent
from corpus import CorpusStore, LexicalRetriever, build_corpus_records
from discovery import classify_source, derive_site_identity
from models import AgentError
from schemas import (
    ChunkerConfiguration,
    CorpusChunk,
    DocumentBlock,
    DocumentSection,
    Evidence,
    FetchedDocument,
    FieldRetrieval,
    RetrievalHit,
)


NOW = datetime(2026, 7, 15, tzinfo=timezone.utc)


def make_document(
    url="https://docs.example.edu/hpc/anvil/jobs/",
    *,
    scope="target_site",
    heading="Queues",
    blocks=None,
):
    blocks = blocks or [DocumentBlock(kind="text", text="Submit jobs with sbatch.")]
    text = "\n\n".join(block.text for block in blocks)
    classification = classify_source(
        identity=derive_site_identity(
            display_name="Example Anvil", organization_domains=["example.edu"]
        ),
        url=url,
    )
    classification.site_scope = scope
    return FetchedDocument(
        url=url,
        title="Anvil guide",
        text=text,
        links=[],
        sections=[
            DocumentSection(
                heading=heading,
                heading_path=["Anvil", heading],
                text=text,
                links=[],
                blocks=blocks,
            )
        ],
        relevant_text=text,
        text_truncated=False,
        classification=classification,
    )


def test_tables_are_atomic_markdown_and_zero_overlap():
    table = "| Queue | Maximum walltime |\n| --- | --- |\n| shared | 96:00:00 |"
    document = make_document(
        blocks=[
            DocumentBlock(kind="text", text="A" * 350),
            DocumentBlock(kind="table", text=table),
            DocumentBlock(kind="text", text="B" * 350),
        ]
    )

    _, chunks = build_corpus_records([document], maximum_chars=200, fetched_at=NOW)

    table_chunks = [chunk for chunk in chunks if chunk.block_type == "table"]
    assert len(table_chunks) == 1
    assert table_chunks[0].text == table
    assert "shared" in table_chunks[0].text and "96:00:00" in table_chunks[0].text
    assert ChunkerConfiguration(maximum_chars=200).overlap_chars == 0


def test_policy_and_faq_sections_remain_whole():
    text = "Charging details " * 100
    document = make_document(
        heading="Policies and FAQ",
        blocks=[DocumentBlock(kind="text", text=text)],
    )

    _, chunks = build_corpus_records([document], maximum_chars=200, fetched_at=NOW)

    assert len(chunks) == 1
    assert chunks[0].block_type == "section"
    assert chunks[0].text == text.strip()


def test_sibling_chunks_are_retained_but_excluded_before_ranking():
    target = make_document()
    sibling = make_document(
        "https://docs.example.edu/hpc/bell/jobs/",
        scope="sibling",
        blocks=[DocumentBlock(kind="text", text="sbatch account partition")],
    )
    _, chunks = build_corpus_records([target, sibling], maximum_chars=500, fetched_at=NOW)

    retrieval = LexicalRetriever(chunks).retrieve(
        field="submission_options",
        query="sbatch account partition",
        allowed_scopes={"target_site", "organization_general"},
        top_k=5,
    )

    assert any(chunk.site_scope == "sibling" for chunk in chunks)
    assert retrieval.excluded_scope_counts["sibling"] == 1
    assert all(hit.chunk.site_scope != "sibling" for hit in retrieval.hits)


def test_refresh_preserves_pages_not_rediscovered(tmp_path):
    identity = derive_site_identity(
        display_name="Example Anvil", organization_domains=["example.edu"]
    )
    first = make_document()
    second = make_document("https://docs.example.edu/hpc/anvil/policies/")
    documents, chunks = build_corpus_records(
        [first, second], maximum_chars=500, fetched_at=NOW
    )
    store = CorpusStore(tmp_path)
    store.merge(
        corpus_id="example-anvil",
        site_identity=identity,
        incoming_documents=documents,
        incoming_chunks=chunks,
        chunker=ChunkerConfiguration(maximum_chars=500),
        refresh=False,
        now=NOW,
    )
    one_document, one_chunks = build_corpus_records(
        [first], maximum_chars=500, fetched_at=NOW
    )

    refreshed = store.merge(
        corpus_id="example-anvil",
        site_identity=identity,
        incoming_documents=one_document,
        incoming_chunks=one_chunks,
        chunker=ChunkerConfiguration(maximum_chars=500),
        refresh=True,
        now=NOW,
    )

    assert refreshed.manifest.document_count == 2
    assert any("policies" in str(item.source_url) for item in refreshed.documents)


def test_evidence_must_be_literal_and_field_local():
    document = make_document()
    _, chunks = build_corpus_records([document], maximum_chars=500, fetched_at=NOW)
    chunk = chunks[0]
    retrieval = FieldRetrieval(
        field="scheduler",
        query="scheduler",
        allowed_scopes=["target_site"],
        hits=[RetrievalHit(chunk=chunk, score=1.0)],
        excluded_scope_counts={},
    )
    evidence = Evidence(
        chunk_id=chunk.chunk_id,
        source_url=chunk.source_url,
        source_title=chunk.title,
        heading="Queues",
        quote="This text was paraphrased.",
        interpretation="direct",
    )
    finding = SimpleNamespace(evidence=[evidence])
    report = SimpleNamespace(
        sources=[SimpleNamespace(url=chunk.source_url)],
        slurm_policy=SimpleNamespace(scheduler=finding),
        network_policy=SimpleNamespace(),
    )

    with pytest.raises(AgentError, match="literal substring"):
        HPCPolicyScoutAgent._validate_report_evidence(
            report, {"scheduler": retrieval}
        )
