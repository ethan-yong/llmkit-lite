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
local/vLLM/LiteLLM-style services and DeepSeek-style hosted APIs.

### Settings dashboard

Start the local settings dashboard:

```powershell
uv run llmkit-lite settings
```

The command opens `http://127.0.0.1:8765/`. Use the browser to configure the
provider, model, API key, generation defaults, retry and circuit-breaker policy,
tracing, logging, and workflow-recovery timing. Settings are saved atomically to
`.llmkit/settings.json`; this directory is ignored by Git because it may contain
credentials. Saved keys are never returned to the browser after storage.

The dashboard is local-only by default. An existing FastAPI service can expose
the same interface at a private route:

```python
from fastapi import FastAPI

from llmkit_lite.dashboard import create_settings_app
from llmkit_lite.settings import SettingsStore


app = FastAPI()
settings_store = SettingsStore()
app.mount("/settings", create_settings_app(settings_store))

# Load validated values when constructing application dependencies.
settings = settings_store.load()
llm_config = settings.to_llm_config()
resilience_policy = settings.to_resilience_policy()
tracing_settings = settings.to_tracing_settings()
```

Running workers may need a restart after settings change because live dependency
replacement is application-specific.

### Environment variables

Environment-based configuration remains available for small deployments and
automation:

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

Provider adapters return a normalized `ChatCompletionResponse` containing text,
tool calls, finish reason, token usage, provider, model, and a deliberately small
set of safe provider metadata. Use the response helper when those fields matter:

```python
import httpx

from llmkit_lite.llm import (
    ChatCompletionResponse,
    call_chat_completion_response,
    resolve_llm_config,
)


async def inspect_router() -> ChatCompletionResponse:
    cfg = resolve_llm_config()
    async with httpx.AsyncClient() as client:
        return await call_chat_completion_response(
            [{"role": "user", "content": "Inspect router 42"}],
            cfg=cfg,
            http_client=client,
            max_tokens=300,
            timeout_seconds=20,
        )
```

`call_chat_completion()` remains a text-only compatibility helper. It raises
`llm_text_response_required` when a valid response contains tool calls but no
text, so callers do not silently discard the requested action.

### Tool declarations

Declare tools independently of any provider payload. Supplying tools
automatically requires the endpoint's `tool_calling` capability:

```python
from llmkit_lite.llm import (
    LlmToolChoice,
    LlmToolDefinition,
    call_chat_completion_response,
)


weather = LlmToolDefinition(
    name="get_weather",
    description="Get the current weather for a city.",
    input_schema={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
)
response = await call_chat_completion_response(
    messages,
    cfg=tool_capable_cfg,
    http_client=client,
    max_tokens=300,
    timeout_seconds=20,
    tools=(weather,),
    tool_choice=LlmToolChoice.auto(),
)
```

Choices can be `auto`, `none`, `required`, or a specific declared tool using
`LlmToolChoice.named("get_weather")`. This request boundary only exposes tools
to the model; returned calls must still pass through an authorized executor.

### Capability negotiation

Declare model capabilities on each endpoint and requirements on each request.
Incompatible requests are rejected before network execution:

```python
from llmkit_lite.llm import (
    ChatCompletionRequest,
    LlmCapabilities,
    LlmEndpointConfig,
)


endpoint = LlmEndpointConfig(
    provider="vllm",
    base_url="http://127.0.0.1:8000",
    model_name="tool-model",
    capabilities=LlmCapabilities({"text", "tool_calling", "json_object"}),
)
request = ChatCompletionRequest(
    messages=[{"role": "user", "content": "Inspect router 42"}],
    max_tokens=300,
    timeout_seconds=20,
    required_capabilities={"text", "tool_calling"},
)
```

The default endpoint and request capability is `text`, preserving existing
text-only integrations. A direct incompatible call raises
`llm_capability_unsupported`. The router skips incompatible candidates and
tries declared fallbacks; if none support the request, it raises
`llm_capabilities_unavailable`. Capability incompatibility does not consume
retries or count as a circuit-breaker failure.

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

Use `router.complete_response()` instead when the application needs the full
normalized response. Retries and fallbacks return the selected adapter's
response without translating it again.

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

Structured calls choose the strongest format declared by the endpoint:
`json_schema`, then `json_object`. Native enforcement is required by default,
so an incompatible endpoint fails before a network request instead of silently
weakening the request.

```python
from pydantic import BaseModel, Field

from llmkit_lite.llm import LlmCapabilities, LlmEndpointConfig
from llmkit_lite.structured import structured_json_call


class Classification(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1)


cfg = LlmEndpointConfig(
    provider="vllm",
    base_url="http://127.0.0.1:8000",
    model_name="structured-model",
    capabilities=LlmCapabilities({"text", "json_schema", "json_object"}),
)
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

The helper derives the provider request schema from the Pydantic model and
still validates the returned JSON locally. To deliberately support a text-only
endpoint, opt in to prompt-based fallback:

```python
from llmkit_lite.structured import StructuredOutputPolicy


result = await structured_json_call(
    messages,
    response_model=Classification,
    cfg=text_only_cfg,
    http_client=client,
    max_tokens=200,
    timeout_seconds=10,
    policy=StructuredOutputPolicy.ALLOW_PROMPT_FALLBACK,
)
```

Use `StructuredOutputPolicy.REQUIRE_JSON_SCHEMA` when JSON object mode is not
strong enough. Provider errors are surfaced directly; the adapter does not
retry by dropping the requested format.

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
            description="Get the current weather for a city.",
            input_schema={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        ),
    )
)
model_tools = tools.llm_tools

principal = Principal(
    subject="user-123",
    scopes={"tools:weather:execute"},
)

with principal_context(principal):
    result = await tools.execute("weather", {"city": "Kuala Lumpur"})
```

Normalized model tool calls can pass through the same boundary and return a
tool message ready for the next model request:

```python
from llmkit_lite.llm import ToolCall


model_call = ToolCall(
    id="call-1",
    name="weather",
    arguments={"city": "Kuala Lumpur"},
)
with principal_context(principal):
    tool_message = await tools.execute_call(model_call)
```

String results become tool-message content unchanged. Other JSON-compatible
results are encoded deterministically; unsupported results raise the safe
`tool_invalid_result` error. Authorization denials, unknown tools, handler
failures, and cancellation still propagate without fabricating a tool message.

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

Adding an input schema explicitly exposes a registration through `llm_tools`;
registrations without schemas remain execution-only. Model declarations contain
only names, descriptions, and schemas—never handlers, scopes, or policy data.

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
from llmkit_lite.llm import ToolCall
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

model_tools = executor.llm_tools
tool_message = await executor.execute_call(
    ToolCall(
        id="call-1",
        name="docs.search",
        arguments={"query": "provider routing"},
    ),
    principal=Principal(
        "user-123",
        scopes={"mcp:docs.search:execute"},
    ),
)
```

The default scope policy denies unauthenticated callers, missing scopes, and
tools without an explicit scope declaration. Authorization and JSON Schema
validation happen before credentials are resolved or a remote request is sent.
Calls have a 30-second default timeout and are attempted once: this layer does
not silently retry remote tools because many tools have side effects.

Only MCP descriptors with a non-empty required-scope mapping are model-visible;
their declarations use server-qualified names to prevent collisions. Model
calls prefer structured MCP results and otherwise encode content blocks as
JSON. The model call ID is used as the default idempotency key, so replaying the
same call does not repeat a successful remote side effect.

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

LLM applications carry several kinds of values with different lifetimes:

| Category | Examples | Lifetime |
| --- | --- | --- |
| Application configuration | Model routes, timeouts, checkpointer provider | Deployment |
| Runtime dependency | HTTP client, router, checkpointer instance | Process |
| Workflow identity | Thread ID, checkpoint namespace | Conversation |
| Workflow state | Messages, findings, pending approval | Workflow step |

Application configuration should be resolved when the service starts and used
to construct dependencies managed by `ApplicationRuntime`. Conversation state
is passed to the graph and persisted by its checkpointer; it must not be stored
inside application configuration.

Compile a graph with an injected LangGraph checkpointer, then use the same
`WorkflowIdentity` when a later request resumes an interrupted workflow:

```python
from langgraph.checkpoint.memory import InMemorySaver

from llmkit_lite.graphs import (
    CheckpointedGraph,
    CheckpointedWorkflowRunner,
    WorkflowExecutionPolicy,
    WorkflowIdentity,
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
identity = WorkflowIdentity(
    thread_id="support-thread-123",
    checkpoint_namespace="support-agent",
)

started = await runner.start(
    {"request": "reboot router"},
    identity,
    configurable={"http_client": request.app.state.http_client},
)

if started.interrupted:
    approval_request = started.interrupts[0]
    # Return approval_request.value to the authorized approval interface.
    resumed = await runner.resume(
        {"approved": True},
        identity,
        configurable={"http_client": request.app.state.http_client},
    )
```

`InMemorySaver` is useful for local development and tests, but loses workflow
state when the process restarts. Production applications should inject a
durable LangGraph-compatible checkpointer. A resume must use the same thread ID
and namespace as the interrupted execution; a new thread ID starts an unrelated
workflow. `CheckpointConfig` and `checkpointed_graph_config()` remain available
as compatibility names, but new code should use `WorkflowIdentity` and
`workflow_run_config()`.

The runner applies a 30-second overall deadline by default. Cancellation and
deadline expiry stop the local execution, but an external service may already
have accepted a side effect. Nodes that can be replayed around an interruption
should therefore use idempotency keys. Resume values must come from an
authenticated, authorized application boundary. Workflow state, resume values,
thread IDs, checkpoint namespaces, and exception messages are excluded from the
built-in execution spans.

### Versioned application state

Use a `StateStore` implementation for application-owned state that must be
coordinated separately from LangGraph checkpoints. Conversation snapshots are
addressed by `WorkflowIdentity`; operation snapshots additionally carry a
stable operation ID and idempotency key:

```python
from llmkit_lite.graphs import WorkflowIdentity
from llmkit_lite.state import InMemoryStateStore


state_store = InMemoryStateStore()
identity = WorkflowIdentity("support-thread-123", "support-agent")

created = await state_store.save_conversation(
    identity,
    {"messages": [{"role": "user", "content": "check router"}]},
)
updated = await state_store.save_conversation(
    identity,
    {"messages": [{"role": "assistant", "content": "checking"}]},
    expected_revision=created.revision,
)
```

Every update uses the previously read revision. A stale writer receives a
`state_revision_conflict` error instead of silently overwriting newer state.
Stored values are copied, deeply immutable, and limited to JSON-compatible
data so persistence adapters can serialize them consistently. State records
provide `to_dict()` when a detached, JSON-ready representation is needed.

`InMemoryStateStore` is a concurrency-safe reference implementation for tests
and local development, but it loses data when the process exits. Production
applications should implement the `StateStore` protocol with durable storage
and inject that implementation through `ApplicationRuntime`.

The LangGraph checkpointer remains the source of truth for graph checkpoints.
Operation state is stored separately because a downstream side effect may need
to be reconciled even when a worker crashes between dispatch and checkpointing.

### Durable operation lifecycle

Use `OperationLifecycle` to move an external operation through explicit,
versioned statuses:

```python
from llmkit_lite.operations import OperationLifecycle, OperationRequest
from llmkit_lite.state import OperationStatus


lifecycle = OperationLifecycle(state_store)
request = OperationRequest("reboot", {"device_id": "router-1"})
operation = await lifecycle.create(
    "reboot-789",
    identity,
    "reboot-key-789",
    request.fingerprint,
    {"request": request.to_dict()},
)

# Persist this before sending the command to the downstream service.
operation = await lifecycle.transition(
    operation.operation_id,
    OperationStatus.DISPATCHING,
    expected_revision=operation.revision,
)
```

Allowed transitions are:

```text
pending      -> dispatching, failed
dispatching  -> in_progress, completed, failed
in_progress  -> completed, failed
completed    -> terminal
failed       -> terminal
```

`dispatching` deliberately represents an uncertain outcome. If a worker fails
after sending a command, a replacement worker must leave the operation in that
state until it can reconcile the downstream result. A timeout, connection
failure, or inability to query status does not prove that the command failed.

Every transition uses the current revision, so concurrent workers cannot both
advance the same operation. Lifecycle spans contain only statuses, revisions,
outcomes, and stable error codes; operation IDs, thread IDs, idempotency keys,
and stored values are excluded.

### Idempotent execution and crash recovery

`DurableOperationExecutor` combines request fingerprinting, lifecycle updates,
dispatch, and read-only reconciliation:

```python
from llmkit_lite.operations import DurableOperationExecutor


executor = DurableOperationExecutor(lifecycle, downstream_adapter)
result = await executor.execute(
    "reboot-789",
    identity,
    "reboot-key-789",
    request,
)
```

The downstream adapter implements two methods. `dispatch()` sends a new command
and may cause a side effect. `inspect()` only reads the status of a previously
dispatched operation. Both return an `OperationObservation` with an outcome of
`in_progress`, `completed`, `failed`, or `unknown`.

Execution follows this recovery-safe sequence:

```text
Create pending record
        ↓
Persist dispatching
        ↓
Send downstream command
        ↓
Worker crashes before checkpoint
        ↓
New executor receives the repeated request
        ↓
Find existing idempotency key and matching request fingerprint
        ↓
Inspect downstream instead of resending
        ↓
Persist completed, in_progress, or confirmed failure
```

A repeated key with an identical request reuses the existing operation. Reusing
the key with a different action or values raises
`operation_idempotency_conflict`. Timeouts and connection failures during
dispatch leave the record as `dispatching`; inspection failures and `unknown`
observations also preserve the current status instead of assuming failure.

`InMemoryMcpIdempotencyStore` remains a process-local single-flight optimization
for MCP calls. Durable workflow protection comes from `DurableOperationExecutor`
and requires a `StateStore` implementation that survives process restarts.

### Renewable fenced worker leases

Use a lease when multiple workers may receive the same durable operation. The
store performs the ownership decision atomically, so two workers cannot both
acquire the same unexpired lease:

```python
from datetime import timedelta

from llmkit_lite.leases import InMemoryLeaseStore, LeaseManager


lease_manager = LeaseManager(
    InMemoryLeaseStore(),
    lease_duration=timedelta(seconds=30),
)
claim = await lease_manager.acquire("reboot-789", "worker-42")

if claim.owns_lease:
    lease = claim.lease
    lease = await lease_manager.renew(
        lease.operation_id,
        lease.owner_id,
        lease.fencing_token,
    )
    await lease_manager.assert_owned(
        lease.operation_id,
        lease.owner_id,
        lease.fencing_token,
    )
```

An absent or expired lease is acquired with a new revision. A retry by the
current owner returns `already_owned` without extending the expiry, while a
different worker receives `held_by_other`. At the exact expiration time the
lease is eligible for takeover.

`renew()` is the worker heartbeat. It extends an unexpired lease only when both
the owner and fencing token still match. A renewal advances the storage
revision but keeps the fencing token unchanged. An expired-lease takeover
advances both values, so the previous worker can no longer renew or validate
its ownership.

Call `assert_owned()` before guarded local work, and pass `fencing_token` to
every protected database or downstream write. That resource must persist the
highest token it has accepted and reject lower tokens; a lease check by itself
cannot stop a worker that pauses after checking and resumes after takeover.

`InMemoryLeaseStore` is suitable for tests and local development but cannot
coordinate separate processes or survive restarts. A production `LeaseStore`
must implement acquisition, renewal, and validation as atomic database
operations rather than read-then-write sequences.

### Queue redelivery and worker recovery

A visibility queue prevents a worker crash from permanently losing an
operation. Receiving a message hides it temporarily. The worker acknowledges
the message only after the operation reaches `completed` or `failed`; otherwise
the message becomes visible again when its deadline expires:

```python
from llmkit_lite.queueing import InMemoryOperationQueue, OperationJob
from llmkit_lite.recovery import DurableOperationHandler, RecoveryWorker


queue = InMemoryOperationQueue()
await queue.publish(
    OperationJob(
        "reboot-789",
        identity,
        "reboot-key-789",
        request,
    )
)

worker = RecoveryWorker(
    queue,
    lease_manager,
    DurableOperationHandler(executor),
    worker_id="worker-42",
)
result = await worker.run_once()
```

`run_once()` intentionally handles at most one visible message; applications
retain control of polling, shutdown, backoff, and deployment. If another worker
holds the lease, or execution remains uncertain or in progress, the delivery
is left unacknowledged for a later attempt. A crash or cancellation follows the
same rule automatically because acknowledgement happens last.

On redelivery, the replacement worker takes over the expired lease and
`DurableOperationExecutor` reads the stored operation. A `dispatching` or
`in_progress` operation is inspected rather than sent again, while a stored
terminal result is reused and acknowledged.

Long-running custom handlers receive a `RecoveryContext`. Calling
`await context.heartbeat()` renews the operation lease first and then extends
queue visibility. Protected resource writes must still receive
`context.fencing_token` and reject tokens older than the highest one previously
accepted. Queue receipt tokens separately prevent an old delivery from
acknowledging a newer redelivery.

`InMemoryOperationQueue` is a deterministic reference implementation, not a
durable production broker. Implement the `OperationQueue` protocol for SQS,
RabbitMQ, Redis, or another queue while preserving atomic receipt and visibility
semantics.

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
