import json
from pathlib import Path

import httpx

from discovery import classify_source, derive_site_identity, generate_discovery_queries
from tools import ScoutTools


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "mixed_sites.json"


def load_fixture():
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def make_anvil_tools(*, page_budget=8):
    fixture = load_fixture()
    identity = derive_site_identity(
        display_name="Purdue Anvil",
        organization_domains=["purdue.edu"],
    )

    def search_backend(query, max_results):
        return fixture["results"][:max_results]

    def page_backend(url):
        return {
            "url": url,
            "content_type": "text/html",
            "text": fixture["pages"][url],
        }

    return ScoutTools(
        allowed_domains=["purdue.edu"],
        site_identity=identity,
        search_backend=search_backend,
        page_backend=page_backend,
        page_budget=page_budget,
    )


def test_anvil_pages_outrank_and_bell_is_sibling():
    tools = make_anvil_tools()
    tools.bootstrap_discovery(keywords=[])

    anvil = tools.candidates[
        "https://docs.rcac.purdue.edu/userguides/anvil/jobs/"
    ]
    bell = tools.candidates[
        "https://www.rcac.purdue.edu/knowledge/bell/run/slurm"
    ]

    assert anvil.classification.site_scope == "target_site"
    assert bell.classification.site_scope == "sibling"
    assert anvil.classification.score > bell.classification.score
    assert "bell" in bell.classification.conflicting_site_tokens


def test_canonical_root_links_are_crawled_and_submission_found():
    tools = make_anvil_tools()
    context = tools.bootstrap_discovery(keywords=[])

    assert context["canonical_root"] == (
        "https://docs.rcac.purdue.edu/userguides/anvil/"
    )
    assert "https://docs.rcac.purdue.edu/userguides/anvil/jobs/" in tools.documents
    assert tools.discovery_coverage().submission_policy.status == "evidence_found"


def test_bell_is_excluded_from_extraction_selection():
    tools = make_anvil_tools()
    tools.bootstrap_discovery(keywords=[])
    result = tools._finish_discovery(
        {
            "source_urls": [
                "https://www.rcac.purdue.edu/knowledge/bell/run/slurm",
                "https://docs.rcac.purdue.edu/userguides/anvil/jobs/",
            ],
            "summary": "Anvil submission evidence found; networking docs are silent.",
            "unanswered_topics": ["Published TCP port range"],
        }
    )

    assert result["ok"] is True
    assert (
        "https://docs.rcac.purdue.edu/userguides/anvil/jobs/"
        in result["selection"]["source_urls"]
    )
    assert not any(
        "bell" in url for url in result["selection"]["source_urls"]
    )
    assert any("bell" in item["url"] for item in result["rejected"])


def test_generic_page_with_anvil_disclaimer_is_not_target_policy():
    tools = make_anvil_tools()
    tools.bootstrap_discovery(keywords=[])
    url = "https://docs.rcac.purdue.edu/workshops/hpc/slurm-basics/"

    fetched = tools._fetch_page({"url": url})

    assert fetched["site_scope"] == "organization_general"
    assert tools._organization_page_applies(tools.documents[url]) is False

    result = tools._finish_discovery(
        {
            "source_urls": [
                url,
                "https://docs.rcac.purdue.edu/userguides/anvil/jobs/",
            ],
            "summary": "Only explicitly applicable pages may be selected.",
            "unanswered_topics": ["Networking policy"],
        }
    )
    assert all(str(item) != url for item in result["selection"]["source_urls"])
    assert any(item["url"] == url for item in result["rejected"])


def test_overview_only_is_not_enough_to_finish():
    fixture = load_fixture()
    root = fixture["results"][0]
    identity = derive_site_identity(
        display_name="Purdue Anvil",
        organization_domains=["purdue.edu"],
    )

    tools = ScoutTools(
        allowed_domains=["purdue.edu"],
        site_identity=identity,
        search_backend=lambda query, limit: [root],
        page_backend=lambda url: {
            "url": url,
            "content_type": "text/html",
            "text": "<html><title>Anvil User Guide</title><h1>Anvil User Guide</h1>"
            "<p>Anvil overview only.</p></html>",
        },
        page_budget=1,
    )
    tools.bootstrap_discovery(keywords=[])
    result = tools._finish_discovery(
        {
            "source_urls": [root["url"]],
            "summary": "Only an overview was found.",
            "unanswered_topics": ["Submission", "Networking"],
        }
    )

    assert result["ok"] is False
    assert "incomplete" in result["error"].lower()


def test_every_generated_query_contains_target_alias():
    identity = derive_site_identity(
        display_name="Purdue Anvil",
        organization_domains=["purdue.edu"],
    )
    queries = generate_discovery_queries(identity)

    assert all("anvil" in query.lower() for group in queries.values() for query in group)


def test_model_query_without_alias_is_repaired():
    seen = []
    identity = derive_site_identity(
        display_name="Purdue Anvil",
        organization_domains=["purdue.edu"],
    )
    tools = ScoutTools(
        allowed_domains=["purdue.edu"],
        site_identity=identity,
        search_backend=lambda query, limit: seen.append(query) or [],
        page_backend=lambda url: {},
    )

    result = tools._search_web({"query": "Slurm partitions and accounts"})

    assert result["query_repaired"] is True
    assert "anvil" in seen[0].lower()


def test_job_url_and_title_outrank_noisy_access_snippet():
    identity = derive_site_identity(
        display_name="Purdue Anvil",
        organization_domains=["purdue.edu"],
    )
    fetched = []
    results = [
        {
            "title": "Access to Anvil",
            "url": "https://docs.rcac.purdue.edu/userguides/anvil/access/",
            "snippet": (
                "Slurm sbatch job submission account partition nodes tasks "
                "memory walltime queue"
            ),
        },
        {
            "title": "Job Submission - RCAC Documentation",
            "url": "https://docs.rcac.purdue.edu/userguides/anvil/jobs/",
            "snippet": "Submit jobs on Anvil.",
        },
    ]
    pages = {
        results[0]["url"]: "<article><h1>Access to Anvil</h1><p>Request an account.</p></article>",
        results[1]["url"]: (
            "<article><h1>Job Submission on Anvil</h1>"
            "<p>Create a job submission script and use sbatch.</p></article>"
        ),
    }
    tools = ScoutTools(
        allowed_domains=["purdue.edu"],
        site_identity=identity,
        search_backend=lambda query, limit: results,
        page_backend=lambda url: fetched.append(url) or {
            "url": url,
            "content_type": "text/html",
            "text": pages[url],
        },
    )
    tools._ranked_search(
        "Anvil Slurm submitting jobs account partition",
        objective="submission_policy",
        generated=True,
    )

    tools._fetch_topic_candidates("submission_policy", limit=1)

    assert fetched[0] == results[1]["url"]
    assert results[1]["url"] in tools.documents


def test_documentation_silence_differs_from_discovery_failure():
    complete = make_anvil_tools()
    complete.bootstrap_discovery(keywords=[])
    assert complete.discovery_coverage().networking_policy.status == (
        "documentation_silent"
    )

    identity = derive_site_identity(
        display_name="Purdue Anvil",
        organization_domains=["purdue.edu"],
    )
    failed = ScoutTools(
        allowed_domains=["purdue.edu"],
        site_identity=identity,
        search_backend=lambda query, limit: [],
        page_backend=lambda url: {},
    )
    failed.bootstrap_discovery(keywords=[])
    coverage = failed.discovery_coverage()
    assert coverage.submission_policy.status == "search_exhausted"
    assert coverage.networking_policy.status == "search_exhausted"

    result = failed._finish_discovery(
        {
            "source_urls": [],
            "summary": "No target-site documentation root was found.",
            "unanswered_topics": ["Submission", "Networking"],
        }
    )
    assert result["ok"] is True
    assert result["selection"]["source_urls"] == []
    assert "discovery_failure" in result["selection"]["termination_reason"]


def test_architecture_does_not_establish_tcp_reachability_or_ports():
    tools = make_anvil_tools()
    tools.bootstrap_discovery(keywords=[])
    coverage = tools.discovery_coverage().networking_policy

    assert coverage.status == "documentation_silent"
    assert coverage.evidence_urls == []


def test_same_classifier_handles_stampede3_without_anvil_rules():
    identity = derive_site_identity(
        display_name="TACC Stampede3",
        organization_domains=["tacc.utexas.edu", "utexas.edu"],
    )
    stampede = classify_source(
        identity=identity,
        url="https://docs.tacc.utexas.edu/hpc/stampede3/running/",
        title="Stampede3 User Guide: Running Jobs",
    )
    frontera = classify_source(
        identity=identity,
        url="https://docs.tacc.utexas.edu/hpc/frontera/running/",
        title="Frontera User Guide: Running Jobs",
    )

    assert stampede.site_scope == "target_site"
    assert frontera.site_scope == "sibling"
    assert stampede.score > frontera.score


def test_stampede3_canonical_crawl_uses_same_pipeline():
    identity = derive_site_identity(
        display_name="TACC Stampede3",
        organization_domains=["tacc.utexas.edu", "utexas.edu"],
    )
    results = [
        {
            "title": "Stampede3 User Guide",
            "url": "https://docs.tacc.utexas.edu/hpc/stampede3/",
            "snippet": "Official Stampede3 documentation.",
        },
        {
            "title": "Frontera User Guide: Running Jobs",
            "url": "https://docs.tacc.utexas.edu/hpc/frontera/running/",
            "snippet": "Frontera policies.",
        },
    ]
    pages = {
        "https://docs.tacc.utexas.edu/hpc/stampede3/": (
            "<html><title>Stampede3 User Guide</title><main><h1>Stampede3 User "
            "Guide</h1><a href='/hpc/stampede3/running/'>Running Jobs</a>"
            "<a href='/hpc/stampede3/architecture/'>Architecture</a></main></html>"
        ),
        "https://docs.tacc.utexas.edu/hpc/stampede3/running/": (
            "<html><title>Stampede3 Running Jobs</title><main><h1>Running Jobs on "
            "Stampede3</h1><p>Submit Slurm jobs with sbatch and select an account "
            "and partition.</p></main></html>"
        ),
        "https://docs.tacc.utexas.edu/hpc/stampede3/architecture/": (
            "<html><title>Stampede3 Architecture</title><main><h1>Stampede3 "
            "Architecture</h1><p>Compute nodes use a high-speed fabric.</p></main></html>"
        ),
    }
    tools = ScoutTools(
        allowed_domains=["tacc.utexas.edu", "utexas.edu"],
        site_identity=identity,
        search_backend=lambda query, limit: results,
        page_backend=lambda url: {
            "url": url,
            "content_type": "text/html",
            "text": pages[url],
        },
    )

    tools.bootstrap_discovery(keywords=[])

    assert tools.canonical_root == "https://docs.tacc.utexas.edu/hpc/stampede3/"
    assert "https://docs.tacc.utexas.edu/hpc/stampede3/running/" in tools.documents
    assert tools.discovery_coverage().submission_policy.status == "evidence_found"
    assert tools.candidates[
        "https://docs.tacc.utexas.edu/hpc/frontera/running/"
    ].classification.site_scope == "sibling"


def test_relevant_sections_survive_front_truncation():
    identity = derive_site_identity(
        display_name="Purdue Anvil",
        organization_domains=["purdue.edu"],
    )
    url = "https://docs.rcac.purdue.edu/userguides/anvil/long-guide/"
    html = (
        "<html><title>Anvil Long Guide</title><main><h1>Anvil Guide</h1><p>"
        + ("intro " * 200)
        + "</p><h2>Job Submission</h2><p>Use sbatch with --account and "
        "--partition.</p></main></html>"
    )
    tools = ScoutTools(
        allowed_domains=["purdue.edu"],
        site_identity=identity,
        search_backend=lambda query, limit: [],
        page_backend=lambda requested_url: {
            "url": requested_url,
            "content_type": "text/html",
            "text": html,
        },
        max_page_chars=100,
    )

    result = tools._fetch_page({"url": url})

    assert result["ok"] is True
    assert result["text_truncated"] is True
    assert "sbatch" in result["relevant_text"]
    assert any(
        section.heading == "Job Submission"
        for section in tools.documents[url].sections
    )


def test_failed_page_is_logged_and_does_not_abort_bootstrap():
    identity = derive_site_identity(
        display_name="Purdue Anvil",
        organization_domains=["purdue.edu"],
    )
    stale_url = "https://docs.rcac.purdue.edu/userguides/anvil/stale/"
    request = httpx.Request("GET", stale_url)
    response = httpx.Response(404, request=request)
    tools = ScoutTools(
        allowed_domains=["purdue.edu"],
        site_identity=identity,
        search_backend=lambda query, limit: [
            {
                "title": "Anvil Stale Documentation",
                "url": stale_url,
                "snippet": "Anvil Slurm job submission documentation.",
            }
        ],
        page_backend=lambda url: (_ for _ in ()).throw(
            httpx.HTTPStatusError("404 Not Found", request=request, response=response)
        ),
    )

    context = tools.bootstrap_discovery(keywords=[])

    assert context["canonical_root"] is None
    assert tools.discovery_metrics()["page_requests"] >= 1
    assert tools.discovery_coverage().submission_policy.status == "search_exhausted"
