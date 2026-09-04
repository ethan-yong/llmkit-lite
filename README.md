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
