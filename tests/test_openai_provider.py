from types import SimpleNamespace

from pydantic import BaseModel

from models import ToolDefinition
from providers import openai_provider


class FakeResponses:
    def __init__(self):
        self.request = None

    def create(self, **request):
        self.request = request
        return SimpleNamespace(
            id="response-1",
            output=[],
            output_text="",
            usage=None,
        )


def test_forced_tool_is_sent_as_tool_choice(monkeypatch):
    responses = FakeResponses()
    client = SimpleNamespace(responses=responses)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client_options = {}

    def fake_openai(**kwargs):
        client_options.update(kwargs)
        return client

    monkeypatch.setattr(openai_provider, "OpenAI", fake_openai)

    provider = openai_provider.OpenAIProvider(model="test-model")
    assert client_options == {"timeout": 90.0, "max_retries": 0}
    provider._instructions = "Test instructions"
    provider._tools = []
    provider._conversation = []
    provider._request_agent_turn(force_tool="finish_discovery")

    assert responses.request["tool_choice"] == {
        "type": "function",
        "name": "finish_discovery",
    }


def test_openai_adapter_translates_shared_tool_definition(monkeypatch):
    responses = FakeResponses()
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        openai_provider,
        "OpenAI",
        lambda **kwargs: SimpleNamespace(responses=responses),
    )
    provider = openai_provider.OpenAIProvider(model="test-model")

    provider.start_agent(
        system_prompt="system",
        user_prompt="user",
        tools=[
            ToolDefinition(
                name="fetch_page",
                description="Fetch a page.",
                parameters={
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "format": "uri"}
                    },
                    "required": ["url"],
                    "additionalProperties": False,
                },
            )
        ],
    )

    translated = responses.request["tools"][0]
    assert translated["type"] == "function"
    assert translated["strict"] is True
    assert "format" not in translated["parameters"]["properties"]["url"]


def test_grouped_extraction_forces_the_supplied_tool_name(monkeypatch):
    class Result(BaseModel):
        value: str

    class ExtractionResponses(FakeResponses):
        def create(self, **request):
            self.request = request
            return SimpleNamespace(
                output=[
                    SimpleNamespace(
                        type="function_call",
                        name="submit_submission_policy",
                        arguments='{"value":"ok"}',
                    )
                ],
                usage=None,
            )

    responses = ExtractionResponses()
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        openai_provider,
        "OpenAI",
        lambda **kwargs: SimpleNamespace(responses=responses),
    )
    provider = openai_provider.OpenAIProvider(model="test-model")

    result = provider.extract_structured(
        system_prompt="system",
        user_prompt="user",
        schema=Result,
        tool_name="submit_submission_policy",
    )

    assert result.value == "ok"
    assert responses.request["tool_choice"] == {
        "type": "function",
        "name": "submit_submission_policy",
    }
