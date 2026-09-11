# llmkit-lite

`llmkit-lite` is a lightweight Python framework/starter kit for production LLM
application services.

The package is intentionally app-agnostic. It provides reusable infrastructure
for provider-agnostic LLM calls, structured output validation, FastAPI service
helpers, LangGraph workflow runners, evaluation harnesses, and CLI tooling.

## Install

```powershell
uv sync --extra dev
```

For a lean application install:

```powershell
uv add llmkit-lite
```

Optional feature groups:

```powershell
uv add "llmkit-lite[api]"
uv add "llmkit-lite[graphs]"
uv add "llmkit-lite[cli]"
uv add "llmkit-lite[observability]"
uv add "llmkit-lite[mcp]"
```

## Configuration

`llmkit-lite` targets OpenAI-compatible chat completion gateways. It supports
local/vLLM/LiteLLM-style services and DeepSeek-style hosted APIs through
environment variables:

```text
LLM_PROVIDER=local
LLM_BASE_URL=http://127.0.0.1:4000
LLM_MODEL_NAME=qwen2.5
LLM_API_KEY=sk-local
LLM_REASONING_EFFORT=low
```

Provider-specific variables such as `VLLM_BASE_URL`, `VLLM_MODEL_NAME`,
`DEEPSEEK_MODEL_NAME`, and `DEEPSEEK_API_KEY` are also supported.

## LLM Calls

```python
import httpx

from llmkit_lite.llm import call_chat_completion, resolve_llm_config


async def summarize(text: str) -> str:
    cfg = resolve_llm_config()
    async with httpx.AsyncClient() as client:
        return await call_chat_completion(
            [{"role": "user", "content": f"Summarize this:\n{text}"}],
            cfg=cfg,
            http_client=client,
            max_tokens=300,
            timeout_seconds=20,
        )
```

## Resilient Provider Routing

Use `LlmRouter` when an application needs to select between providers, retry
failed calls, and fall back when a provider remains unavailable. The original
`call_chat_completion()` helper stays one-shot so existing applications do not
silently gain extra latency or provider cost.

This example normally uses local vLLM, explicitly routes requests that prefer
DeepSeek, and gives each primary route one ordered fallback:

```python
import os

import httpx

from llmkit_lite.llm import ChatCompletionRequest, LlmEndpointConfig
from llmkit_lite.routing import (
    LlmResiliencePolicy,
    LlmRoute,
    LlmRouter,
    LlmRouteRule,
)


local = LlmEndpointConfig(
    provider="vllm",
    base_url=os.environ["VLLM_BASE_URL"],
    model_name=os.environ["VLLM_MODEL_NAME"],
    api_key=os.getenv("VLLM_API_KEY"),
)
deepseek = LlmEndpointConfig(
    provider="deepseek",
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
    model_name=os.environ["DEEPSEEK_MODEL_NAME"],
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

router = LlmRouter(
    routes=(
        LlmRoute("local", local, fallback_routes=("deepseek",)),
        LlmRoute("deepseek", deepseek, fallback_routes=("local",)),
    ),
    default_route="local",
    resilience_policy=LlmResiliencePolicy(),
    rules=(
        LlmRouteRule(
            name="prefer DeepSeek",
            route_name="deepseek",
            predicate=lambda request: (
                request.routing_metadata.get("provider_preference") == "deepseek"
            ),
        ),
    ),
)


async def complete(messages: list[dict[str, str]]) -> str:
    request = ChatCompletionRequest(
        messages=messages,
        max_tokens=300,
        timeout_seconds=20,
        routing_metadata={"provider_preference": "local"},
    )
    async with httpx.AsyncClient() as client:
        return await router.complete(request, http_client=client)
```

The default resilience policy tries each eligible route twice, with exponential
backoff and jitter. A route's circuit opens after three consecutive calls exhaust
their retries. After 30 seconds, one request is allowed to probe the route; a
successful probe closes the circuit.

All `LlmGatewayError` values trigger retries and fallback, including HTTP 4xx and
malformed provider responses. This can increase latency and cost, and a timeout
can cause a provider to process a duplicate request. Configure fewer attempts
with `LlmResiliencePolicy` when those tradeoffs are unsuitable. Routing metadata
is available to predicates but is not sent to providers or recorded in telemetry.

## Structured Outputs

```python
from pydantic import BaseModel, Field

from llmkit_lite.structured import structured_json_call


class Classification(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1)


result = await structured_json_call(
    messages,
    response_model=Classification,
    cfg=cfg,
    http_client=client,
    max_tokens=200,
    timeout_seconds=10,
)

if result.ok:
    classification = result.value
```

## FastAPI Helpers

```python
from fastapi import FastAPI

from llmkit_lite.api import (
    add_request_id_middleware,
    configure_logging,
    http_client_lifespan,
    llm_exception_handler,
)
from llmkit_lite.llm import LlmGatewayError


configure_logging()
app = FastAPI(lifespan=http_client_lifespan())
add_request_id_middleware(app)
app.add_exception_handler(LlmGatewayError, llm_exception_handler())
```

Requests accept `X-Request-ID` and optional `X-Thread-ID` headers. The middleware
keeps those identifiers isolated across concurrent requests and creates a server
span when tracing is enabled.

## Application Runtime

Use `ApplicationRuntime` as the application composition root for dependencies
that should be created once and shared across requests. Dependencies start in
declaration order and managed resources close in reverse order:

```python
import httpx
from fastapi import FastAPI, Request

from llmkit_lite.api import runtime_lifespan
from llmkit_lite.runtime import ApplicationRuntime, RuntimeDependency


runtime = ApplicationRuntime(
    (
        RuntimeDependency(
            name="http_client",
            factory=lambda dependencies: httpx.AsyncClient(),
        ),
        RuntimeDependency(
            name="support_service",
            requires=("http_client",),
            factory=lambda dependencies: build_support_service(
                http_client=dependencies["http_client"],
            ),
        ),
    )
)
app = FastAPI(lifespan=runtime_lifespan(runtime))


@app.get("/health")
async def health(request: Request) -> dict[str, bool]:
    service = request.app.state.runtime.get("support_service")
    return {"ready": service.is_ready()}
```

A factory receives a read-only snapshot containing dependencies initialized
before it. Its `requires` entries must therefore refer to earlier declarations.
Factories may return plain values, awaitables, synchronous context managers, or
asynchronous context managers. The runtime publishes no values until every
factory has succeeded; partial startup is cleaned up automatically.

Do not let both `ApplicationRuntime` and `http_client_lifespan()` manage the same
HTTP client. Choose one owner for each resource so it is closed exactly once.
Runtime spans contain only dependency names, counts, outcomes, and stable error
codes—not dependency values, credentials, configuration, or exception messages.
Configuration loading, secret-store integration, and deployment setup remain
the consuming application's responsibility.

## Protected Tool Execution

Register application tools with the scopes required to invoke them, then call
them through `AuthorizedToolExecutor`. Authorization happens at the execution
boundary before the handler receives any arguments:

```python
from collections.abc import Mapping
from typing import Any

from llmkit_lite.authorization import Principal, principal_context
from llmkit_lite.tools import AuthorizedToolExecutor, ToolDefinition


async def get_weather(arguments: Mapping[str, Any]) -> dict[str, str]:
    city = str(arguments["city"])
    return {"city": city, "forecast": "sunny"}


tools = AuthorizedToolExecutor(
    (
        ToolDefinition(
            name="weather",
            handler=get_weather,
            required_scopes={"tools:weather:execute"},
        ),
    )
)

principal = Principal(
    subject="user-123",
    scopes={"tools:weather:execute"},
)

with principal_context(principal):
    result = await tools.execute("weather", {"city": "Kuala Lumpur"})
```

The default `ScopeAuthorizationPolicy` denies execution when there is no active
principal, when the principal lacks any required scope, or when a tool declares
an empty required-scope set. An application can pass a custom authorization
policy to the executor; its reason codes should be stable identifiers and must
not contain user information.

Denied and unknown tools are never invoked. The executor supports synchronous
and asynchronous handlers and passes an allowed handler a shallow copy of its
arguments. Its audit logs and `tool.execute` spans contain only the registered
tool name, outcome, stable reason code, and safe exception type. Subjects,
scopes, arguments, results, credentials, and exception messages are excluded.

## Secure MCP Tool Execution

MCP integrations use two separate steps: discover the tools exposed by a
configured server, then place only approved tools behind the same authorization
boundary as local application tools. Server descriptions and input schemas are
remote, untrusted data; review them before making discovered tools available.

Keep credentials in a dedicated authentication provider. It receives the active
principal and registered server name, but never receives tool arguments:

```python
import os
from collections.abc import Mapping

from llmkit_lite.authorization import Principal
from llmkit_lite.mcp import (
    AuthorizedMcpToolExecutor,
    McpExecutionPolicy,
    McpServer,
    McpServerRegistry,
    StreamableHttpMcpAdapter,
)


class EnvironmentTokenProvider:
    async def get_headers(
        self,
        principal: Principal | None,
        server_name: str,
    ) -> Mapping[str, str]:
        return {"Authorization": f"Bearer {os.environ['DOCS_MCP_TOKEN']}"}


registry = McpServerRegistry(
    (
        McpServer(
            name="docs",
            endpoint="https://mcp.example.com/mcp",
            adapter=StreamableHttpMcpAdapter(),
            authentication_provider=EnvironmentTokenProvider(),
        ),
    )
)

tools = await registry.discover_tools()
executor = AuthorizedMcpToolExecutor(
    registry,
    tools,
    required_scopes={
        "docs.search": {"mcp:docs.search:execute"},
    },
    execution_policy=McpExecutionPolicy(
        timeout_seconds=30,
        idempotency_ttl_seconds=300,
    ),
)

result = await executor.execute(
    "docs.search",
    {"query": "provider routing"},
    principal=Principal(
        "user-123",
        scopes={"mcp:docs.search:execute"},
    ),
    idempotency_key="search-request-123",
)
```

The default scope policy denies unauthenticated callers, missing scopes, and
tools without an explicit scope declaration. Authorization and JSON Schema
validation happen before credentials are resolved or a remote request is sent.
Calls have a 30-second default timeout and are attempted once: this layer does
not silently retry remote tools because many tools have side effects.

An idempotency key deduplicates matching calls for the same principal and tool.
The built-in store is process-local, keeps successful results for five minutes,
and is suitable for a single application process. Use a shared implementation
of `McpIdempotencyStore` when multiple processes must coordinate. Never place
tokens in endpoint URLs, tool arguments, logs, or traces; MCP telemetry records
only server/tool names, outcomes, and stable error codes.

## Observability

Tracing is disabled by default. Enable OTLP/HTTP export during application
startup:

```python
from llmkit_lite.observability import TracingSettings, configure_tracing


configure_tracing(
    TracingSettings(
        enabled=True,
        service_name="support-agent",
        service_version="1.0.0",
        environment="production",
    )
)
```

Set the standard `OTEL_EXPORTER_OTLP_ENDPOINT` environment variable to your
collector URL, or pass `otlp_endpoint` explicitly. FastAPI requests and LLM
gateway calls produce correlated spans, and active request, thread, and trace
IDs are added to configured log records. Trace context is extracted from
incoming requests and propagated to the LLM gateway.

The built-in instrumentation records operational metadata such as provider,
model, HTTP status, latency, and token counts. It does not record prompts,
responses, authorization headers, API keys, or tool arguments.

## LangGraph Helpers

```python
from llmkit_lite.graphs import CachedGraph, run_cached_graph


cached_graph = CachedGraph(build_graph)

result = await run_cached_graph(
    cached_graph,
    {"input": payload},
    configurable={"http_client": request.app.state.http_client},
    fallback=lambda exc, state: {"output": None, "error": str(exc)},
)
```

### Checkpointed and resumable workflows

Compile a graph with an injected LangGraph checkpointer, then use the same
`CheckpointConfig` when a later request resumes an interrupted workflow:

```python
from langgraph.checkpoint.memory import InMemorySaver

from llmkit_lite.graphs import (
    CheckpointConfig,
    CheckpointedGraph,
    CheckpointedWorkflowRunner,
    WorkflowExecutionPolicy,
)


checkpointed_graph = CheckpointedGraph(
    builder=lambda checkpointer: build_graph().compile(
        checkpointer=checkpointer,
    ),
    checkpointer=InMemorySaver(),
)
runner = CheckpointedWorkflowRunner(
    checkpointed_graph,
    execution_policy=WorkflowExecutionPolicy(deadline_seconds=30),
)
checkpoint = CheckpointConfig(
    thread_id="support-thread-123",
    checkpoint_namespace="support-agent",
)

started = await runner.start(
    {"request": "reboot router"},
    checkpoint,
    configurable={"http_client": request.app.state.http_client},
)

if started.interrupted:
    approval_request = started.interrupts[0]
    # Return approval_request.value to the authorized approval interface.
    resumed = await runner.resume(
        {"approved": True},
        checkpoint,
        configurable={"http_client": request.app.state.http_client},
    )
```

`InMemorySaver` is useful for local development and tests, but loses workflow
state when the process restarts. Production applications should inject a
durable LangGraph-compatible checkpointer. A resume must use the same thread ID
and namespace as the interrupted execution; a new thread ID starts an unrelated
workflow.

The runner applies a 30-second overall deadline by default. Cancellation and
deadline expiry stop the local execution, but an external service may already
have accepted a side effect. Nodes that can be replayed around an interruption
should therefore use idempotency keys. Resume values must come from an
authenticated, authorized application boundary. Workflow state, resume values,
thread IDs, checkpoint namespaces, and exception messages are excluded from the
built-in execution spans.

## Evaluations

Cases can be JSON arrays or JSONL records:

```json
[
  {
    "id": "case-1",
    "input": {"prompt": "Reply with hello."},
    "expected": "hello"
  }
]
```

Run an echo/equality check:

```powershell
uv run llmkit-lite eval cases.json
```

Run cases through the configured LLM:

```powershell
uv run llmkit-lite eval cases.json --llm --output report.json
```

## CLI

Inspect an endpoint without making network calls:

```powershell
uv run llmkit-lite inspect-llm --base-url http://127.0.0.1:4000 --model demo --skip-models --skip-chat
```

Probe models and chat:

```powershell
uv run llmkit-lite inspect-llm --dotenv .env
```

## Boundary

This project owns infrastructure, not product behavior. Keep app-specific
schemas, prompts, vocabularies, business rules, persistence, and user-facing
voice in the consuming application.
