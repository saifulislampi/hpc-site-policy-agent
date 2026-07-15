# Provider adapter plan

OpenAI, Claude, and Gemini share the complete discovery, extraction, validation,
and artifact pipeline. Only API translation belongs in `providers/`.

Each adapter must implement `BaseProvider`:

```text
start_agent
  → translate shared prompts + typed ToolDefinition values to provider request
  → normalize provider response to ModelTurn + ToolCall

continue_agent
  → translate shared ToolResult values into the existing conversation
  → support a forced finish_discovery tool choice
  → normalize the next response

extract_report
  → request the shared ExtractedPolicy schema
  → validate with Pydantic
  → return ExtractedPolicy
```

Implementation order:

```text
shared fake-provider contract tests
  → Claude SDK adapter + mocked request/response tests
  → Gemini SDK adapter + mocked request/response tests
  → identical mocked Anvil run for all providers
  → bounded live smoke test per provider
  → compare tokens, latency, unsupported claims, and run variance
```

Provider adapters must not contain site classification, source scoring, chunking,
retrieval, profile conversion, or artifact-writing logic. The persistent corpus
and transient retrieval pipeline is shared by every future adapter.
