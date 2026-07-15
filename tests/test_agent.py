import pytest

from agent import HPCPolicyScoutAgent
from models import AgentError, ModelTurn, ToolCall, ToolResult


class FakeProvider:
    def __init__(self):
        self.continuations: list[str | None] = []

    def start_agent(self, *, system_prompt, user_prompt, tools):
        return ModelTurn(
            text=None,
            tool_calls=[
                ToolCall(
                    call_id="search-1",
                    name="search_web",
                    arguments={"query": "Anvil Slurm policy"},
                )
            ],
        )

    def continue_agent(self, *, tool_results, force_tool=None):
        self.continuations.append(force_tool)
        return ModelTurn(
            text=None,
            tool_calls=[
                ToolCall(
                    call_id="finish-1",
                    name="finish_discovery",
                    arguments={
                        "source_urls": [
                            "https://docs.rcac.purdue.edu/userguides/anvil/jobs/"
                        ],
                        "summary": "The job submission page covers Slurm policy.",
                        "unanswered_topics": ["Published TCP port range"],
                    },
                )
            ],
        )


class FakeTools:
    def definitions(self):
        return []

    def execute(self, call):
        if call.name == "finish_discovery":
            coverage = {
                "canonical_root_found": True,
                "canonical_root": (
                    "https://docs.rcac.purdue.edu/userguides/anvil/"
                ),
                "submission_policy": {
                    "topic": "submission_policy",
                    "status": "evidence_found",
                    "target_site_pages_examined": 1,
                    "queries_attempted": 1,
                    "evidence_urls": call.arguments["source_urls"],
                    "notes": ["Submission evidence found."],
                },
                "networking_policy": {
                    "topic": "networking_policy",
                    "status": "documentation_silent",
                    "target_site_pages_examined": 2,
                    "queries_attempted": 2,
                    "evidence_urls": [],
                    "notes": ["Networking documentation was silent."],
                },
            }
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                output={
                    "ok": True,
                    "selection": {
                        **call.arguments,
                        "canonical_root": coverage["canonical_root"],
                        "coverage": coverage,
                        "termination_reason": "test_complete",
                    },
                },
                terminal=True,
            )
        return ToolResult(
            call_id=call.call_id,
            name=call.name,
            output={"ok": True, "results": []},
        )


def test_final_turn_is_forced_to_finish_discovery(tmp_path):
    provider = FakeProvider()
    agent = HPCPolicyScoutAgent(
        provider=provider,
        tools=FakeTools(),
        max_steps=2,
        log_dir=tmp_path,
    )

    selection = agent._run_discovery(
        site_name="Purdue Anvil",
        keywords=["Anvil Slurm policy"],
        allowed_domains=["purdue.edu"],
    )

    assert provider.continuations == ["finish_discovery"]
    assert selection.unanswered_topics == ["Published TCP port range"]


def test_limit_does_not_request_an_unprocessed_turn(tmp_path):
    provider = FakeProvider()
    agent = HPCPolicyScoutAgent(
        provider=provider,
        tools=FakeTools(),
        max_steps=1,
        log_dir=tmp_path,
    )

    with pytest.raises(AgentError, match="maximum of 1 steps"):
        agent._run_discovery(
            site_name="Purdue Anvil",
            keywords=["Anvil Slurm policy"],
            allowed_domains=["purdue.edu"],
        )

    assert provider.continuations == []
