import re
from datetime import datetime, timezone

from agent import HPCPolicyScoutAgent
from models import ModelTurn, ToolCall
from reporting import build_run_artifacts
from corpus import CorpusStore, LexicalRetriever, build_corpus_records
from schemas import ChunkerConfiguration, ExtractedPolicy
from test_discovery import make_anvil_tools


JOBS_URL = "https://docs.rcac.purdue.edu/userguides/anvil/jobs/"


class ArtifactProvider:
    provider_name = "openai"
    model = "mock-model"

    def __init__(self, tools):
        self.tools = tools

    def start_agent(self, *, system_prompt, user_prompt, tools):
        return ModelTurn(
            text=None,
            tool_calls=[
                ToolCall(
                    call_id="finish-1",
                    name="finish_discovery",
                    arguments={
                        "source_urls": [JOBS_URL],
                        "summary": "Submission evidence found; networking silent.",
                        "unanswered_topics": ["Networking policy"],
                    },
                )
            ],
            input_tokens=100,
            output_tokens=20,
        )

    def continue_agent(self, *, tool_results, force_tool=None):
        raise AssertionError("The mocked run should finish in one model turn.")

    def extract_report(self, *, system_prompt, user_prompt):
        self.last_extraction_usage = {"input_tokens": 200, "output_tokens": 100}
        evidence = {}
        for field, body in re.findall(
            r"--- FIELD (\w+) RETRIEVED CHUNKS START ---(.*?)"
            r"--- FIELD \1 RETRIEVED CHUNKS END ---",
            user_prompt,
            flags=re.DOTALL,
        ):
            chunks = re.findall(
                r"CHUNK ID: (\S+).*?URL: (\S+).*?TITLE: (.*?)\n.*?"
                r"HEADING PATH: (.*?)\nCHUNK TEXT:\n(.*?)\nEND CHUNK \1",
                body,
                flags=re.DOTALL,
            )
            preferred = next((item for item in chunks if item[1] == JOBS_URL), None)
            chosen = preferred or (chunks[0] if chunks else None)
            if chosen:
                evidence[field] = chosen
        return make_extracted_policy(self.tools, evidence)


def documented(value, field, evidence_map):
    chunk_id, source_url, title, heading, text = evidence_map[field]
    quote = next(line.strip() for line in text.splitlines() if line.strip())
    return {
        "value": value,
        "status": "documented",
        "documentation_status": "documented",
        "confidence": 0.99,
        "explanation": "The target-site job guide directly documents this value.",
        "evidence": [
            {
                "chunk_id": chunk_id,
                "source_url": source_url,
                "source_title": title,
                "heading": heading,
                "quote": quote,
                "interpretation": "direct",
            }
        ],
    }


def requires_probe():
    return {
        "value": None,
        "status": "requires_probe",
        "documentation_status": "silent",
        "confidence": 1.0,
        "explanation": "The completed target-site search did not document this fact.",
        "evidence": [],
    }


def make_extracted_policy(tools, evidence_map):
    source_urls = sorted({item[1] for item in evidence_map.values()})
    return ExtractedPolicy.model_validate(
        {
            "site_name": "Purdue Anvil",
            "sources": [
                {
                    "url": url,
                    "title": "Retrieved official documentation",
                    "authority": "official",
                    "relevance": "Target-site submission policy.",
                }
                for url in source_urls
            ],
            "discovery_coverage": tools.discovery_coverage().model_dump(mode="json"),
            "slurm_policy": {
                "scheduler": documented("slurm", "scheduler", evidence_map),
                "submit_command": documented("sbatch", "submit_command", evidence_map),
                "submission_options": documented(
                    [
                        {
                            "name": "account",
                            "syntax": ["-A {account}", "--account={account}"],
                            "required": True,
                            "example": "-A cis250350",
                            "value": None,
                        },
                        {
                            "name": "partition",
                            "syntax": ["-p {partition}", "--partition={partition}"],
                            "required": True,
                            "example": "-p shared",
                            "value": "shared",
                        },
                        {
                            "name": "memory",
                            "syntax": ["--mem={memory}"],
                            "required": False,
                            "example": "--mem=8G",
                            "value": None,
                        },
                    ], "submission_options", evidence_map
                ),
                "account_allocation_policy": documented("Use the allocation account.", "account_allocation_policy", evidence_map),
                "default_partition": documented("shared", "default_partition", evidence_map),
                "partitions": documented(
                    [
                        {
                            "name": "shared",
                            "maximum_walltime": "96:00:00",
                            "maximum_nodes": 1,
                            "shared_nodes": True,
                        },
                        {
                            "name": "wholenode",
                            "maximum_walltime": "96:00:00",
                            "maximum_nodes": 16,
                            "shared_nodes": False,
                        },
                    ], "partitions", evidence_map
                ),
                "walltime_policy": documented("Partition-dependent limits apply.", "walltime_policy", evidence_map),
                "memory_policy": documented("Request memory in the job script.", "memory_policy", evidence_map),
                "job_size_policy": documented("Partition-dependent node limits apply.", "job_size_policy", evidence_map),
                "charging_model": documented("Jobs charge the selected allocation.", "charging_model", evidence_map),
                "purge_policy": documented("A documented purge policy applies.", "purge_policy", evidence_map),
                "cost_traps": documented("Review documented charging cautions.", "cost_traps", evidence_map),
            },
            "network_policy": {
                "manager_worker_connectivity": requires_probe(),
                "worker_worker_connectivity": requires_probe(),
                "published_port_range": requires_probe(),
                "manager_address_guidance": requires_probe(),
                "login_node_socket_policy": requires_probe(),
                "outbound_compute_network": requires_probe(),
            },
            "unresolved_questions": ["Which application TCP ports are reachable?"],
            "overall_notes": ["Networking policy was not documented."],
        }
    )


def prepare_rag(tools, tmp_path):
    documents, chunks = build_corpus_records(
        tools.corpus_documents(),
        maximum_chars=1800,
        fetched_at=datetime.now(timezone.utc),
    )
    snapshot = CorpusStore(tmp_path / "corpus").merge(
        corpus_id="purdue-anvil",
        site_identity=tools.site_identity,
        incoming_documents=documents,
        incoming_chunks=chunks,
        chunker=ChunkerConfiguration(maximum_chars=1800),
        refresh=False,
        now=datetime.now(timezone.utc),
    )
    retrievals = LexicalRetriever(snapshot.active_chunks).retrieve_all(
        allowed_scopes={"target_site", "organization_general"}, top_k=5
    )
    evidence_map = {
        field: (
            hit.chunk.chunk_id,
            str(hit.chunk.source_url),
            hit.chunk.title,
            " > ".join(hit.chunk.heading_path),
            hit.chunk.text,
        )
        for field, retrieval in retrievals.items()
        if (hit := next((x for x in retrieval.hits if str(x.chunk.source_url) == JOBS_URL), retrieval.hits[0] if retrieval.hits else None))
    }
    audit = {
        field: {
            "retrieved": [
                {"chunk_id": hit.chunk.chunk_id, "score": hit.score, "cited": hit.chunk.chunk_id == evidence_map.get(field, (None,))[0]}
                for hit in retrieval.hits
            ],
            "retrieved_but_uncited": [
                hit.chunk.chunk_id for hit in retrieval.hits
                if hit.chunk.chunk_id != evidence_map.get(field, (None,))[0]
            ],
        }
        for field, retrieval in retrievals.items()
    }
    return snapshot, retrievals, evidence_map, audit


def test_builds_detailed_and_compact_artifacts(tmp_path):
    tools = make_anvil_tools()
    tools.bootstrap_discovery(keywords=[])
    result = tools._finish_discovery(
        {
            "source_urls": [JOBS_URL],
            "summary": "Submission evidence found; networking documentation silent.",
            "unanswered_topics": ["Networking policy"],
        }
    )
    snapshot, retrievals, evidence_map, audit = prepare_rag(tools, tmp_path)
    extracted = make_extracted_policy(tools, evidence_map)

    artifacts = build_run_artifacts(
        extracted=extracted,
        tools=tools,
        run_id="run-123",
        provider="openai",
        model="gpt-5-mini",
        timestamp=datetime(2026, 7, 15, 19, 0, tzinfo=timezone.utc),
        site_id="purdue-anvil",
        discovery_report_reference=(
            "purdue-anvil-20260715-190000.discovery-report.json"
        ),
        termination_reason=result["selection"]["termination_reason"],
        metrics={
            **tools.discovery_metrics(),
            "model_requests": 2,
            "total_input_tokens": 1000,
            "total_output_tokens": 500,
        },
        corpus_snapshot=snapshot,
        retrievals=retrievals,
        citation_audit=audit,
        corpus_manifest_reference=str(tmp_path / "corpus" / "manifest.json"),
    )

    report = artifacts.discovery_report
    policy = artifacts.site_policy
    jobs_source = next(source for source in report.sources if str(source.url) == JOBS_URL)
    bell_sources = [source for source in report.sources if "bell" in str(source.url)]

    assert report.schema_version == "0.1"
    assert report.run.provider == "openai"
    assert report.discovery.coverage.submission == "complete"
    assert report.discovery.coverage.networking == "documentation_silent"
    assert jobs_source.selected is True
    assert bell_sources
    assert all(source.selected is False for source in bell_sources)
    assert report.findings["submission.options"].evidence[0].source_id == (
        jobs_source.id
    )

    assert policy.site.id == "purdue-anvil"
    assert policy.scheduler.type == "slurm"
    assert policy.scheduler.submit_command == "sbatch"
    assert policy.submission.options[0].syntax == [
        "-A {account}",
        "--account={account}",
    ]
    assert policy.submission.options[0].example == "-A cis250350"
    assert policy.submission.options[0].value is None
    assert policy.partitions.limits["wholenode"].maximum_nodes == 16
    assert policy.network.manager_worker is None
    assert policy.storage.scratch_directory is None
    assert policy.validation.network == "probe_required"
    assert policy.provenance.run_id == "run-123"
    assert policy.provenance.references["/submission/options"] == (
        "/findings/submission.options"
    )


def test_compact_policy_separates_values_status_and_probes(tmp_path):
    tools = make_anvil_tools()
    tools.bootstrap_discovery(keywords=[])
    tools._finish_discovery(
        {
            "source_urls": [JOBS_URL],
            "summary": "Coverage complete.",
            "unanswered_topics": ["Networking"],
        }
    )
    snapshot, retrievals, evidence_map, audit = prepare_rag(tools, tmp_path)
    artifacts = build_run_artifacts(
        extracted=make_extracted_policy(tools, evidence_map),
        tools=tools,
        run_id="run-456",
        provider="gemini",
        model="test-model",
        timestamp=datetime.now(timezone.utc),
        site_id="purdue-anvil",
        discovery_report_reference="purdue-anvil.discovery-report.json",
        termination_reason="test",
        metrics=tools.discovery_metrics(),
        corpus_snapshot=snapshot,
        retrievals=retrievals,
        citation_audit=audit,
        corpus_manifest_reference=str(tmp_path / "corpus" / "manifest.json"),
    )
    dumped = artifacts.site_policy.model_dump(mode="json")

    assert "findings" not in dumped
    assert "evidence" not in dumped
    assert dumped["network"]["port_range"] is None
    assert dumped["storage"]["shared_filesystem"] is None
    assert dumped["validation"]["network"] == "probe_required"
    assert dumped["submission"]["options"][0]["syntax"][0] == "-A {account}"


def test_mocked_full_run_builds_artifacts_and_prints_progress(tmp_path, capsys):
    tools = make_anvil_tools()
    agent = HPCPolicyScoutAgent(
        provider=ArtifactProvider(tools),
        tools=tools,
        max_steps=2,
        log_dir=tmp_path,
        corpus_dir=tmp_path / "corpus",
    )

    artifacts = agent.run(
        site_name="Purdue Anvil",
        site_id="purdue-anvil",
        discovery_report_reference="purdue-anvil.discovery-report.json",
        keywords=["Anvil Slurm", "Anvil networking"],
        allowed_domains=["purdue.edu"],
    )
    progress = capsys.readouterr().err

    assert artifacts.discovery_report.run.model == "mock-model"
    assert artifacts.site_policy.scheduler.submit_command == "sbatch"
    assert artifacts.discovery_report.statistics.sibling_chunks > 0
    assert artifacts.discovery_report.statistics.retrieved_sibling_chunks == 0
    assert any(
        not chunk.cited
        for retrieval in artifacts.discovery_report.retrieval.values()
        for chunk in retrieval.chunks
    )
    assert artifacts.site_policy.provenance.corpus_fingerprint.startswith("sha256:")
    assert "Search 1/8" in progress
    assert "Waiting for the discovery model" in progress
    assert "Extracting policy report" in progress
    assert "Built discovery-report and candidate site-policy artifacts" in progress
