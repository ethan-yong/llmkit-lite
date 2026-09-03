"""Command-line tools for llmkit-lite."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Annotated, Any

import httpx
import typer

from llmkit_lite.evals import EvalCase, load_eval_cases, run_evals, write_eval_report
from llmkit_lite.llm import (
    LlmEndpointConfig,
    LlmGatewayError,
    call_chat_completion,
    normalize_api_base,
    resolve_llm_config,
)

app = typer.Typer(no_args_is_help=True, help="Utilities for production LLM services.")


def _load_dotenv(path: Path | None) -> None:
    if path is None or not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _env_with_overrides(
    *,
    provider: str | None,
    base_url: str | None,
    model: str | None,
    api_key: str | None,
    reasoning_effort: str | None,
) -> dict[str, str]:
    env = dict(os.environ)
    if provider:
        env["LLM_PROVIDER"] = provider
    if base_url:
        env["LLM_BASE_URL"] = base_url
    if model:
        env["LLM_MODEL_NAME"] = model
    if api_key:
        env["LLM_API_KEY"] = api_key
    if reasoning_effort:
        env["LLM_REASONING_EFFORT"] = reasoning_effort
    return env


async def _list_models(cfg: LlmEndpointConfig) -> list[str]:
    headers = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"
    root = cfg.base_url.rstrip("/")
    api_base = normalize_api_base(cfg.base_url)
    roots = [api_base, root]
    if root.endswith("/v1"):
        roots.append(root[:-3].rstrip("/"))

    seen: set[str] = set()
    async with httpx.AsyncClient() as client:
        for base in roots:
            try:
                response = await client.get(f"{base}/models", headers=headers, timeout=10)
            except httpx.HTTPError:
                continue
            if response.status_code != 200:
                continue
            try:
                payload = response.json()
            except ValueError:
                continue
            for item in payload.get("data", []):
                model_id = item.get("id") if isinstance(item, dict) else None
                if isinstance(model_id, str) and model_id not in seen:
                    seen.add(model_id)
    return list(seen)


async def _inspect_chat(cfg: LlmEndpointConfig) -> str:
    async with httpx.AsyncClient() as client:
        return await call_chat_completion(
            [{"role": "user", "content": "Reply in one short sentence: ready."}],
            cfg=cfg,
            http_client=client,
            max_tokens=80,
            timeout_seconds=60,
            use_response_format=False,
        )


@app.command("inspect-llm")
def inspect_llm(
    dotenv: Annotated[
        Path | None,
        typer.Option("--dotenv", help="Optional .env file to load before resolving config."),
    ] = Path(".env"),
    provider: Annotated[str | None, typer.Option("--provider")] = None,
    base_url: Annotated[str | None, typer.Option("--base-url")] = None,
    model: Annotated[str | None, typer.Option("--model")] = None,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
    reasoning_effort: Annotated[str | None, typer.Option("--reasoning-effort")] = None,
    skip_models: Annotated[bool, typer.Option("--skip-models")] = False,
    skip_chat: Annotated[bool, typer.Option("--skip-chat")] = False,
) -> None:
    """Inspect the configured OpenAI-compatible LLM endpoint."""

    _load_dotenv(dotenv)
    try:
        cfg = resolve_llm_config(
            _env_with_overrides(
                provider=provider,
                base_url=base_url,
                model=model,
                api_key=api_key,
                reasoning_effort=reasoning_effort,
            )
        )
    except LlmGatewayError as exc:
        raise typer.BadParameter(exc.detail) from exc

    typer.echo("LLM Endpoint")
    typer.echo("------------")
    typer.echo(f"provider: {cfg.provider}")
    typer.echo(f"api_base: {normalize_api_base(cfg.base_url)}")
    typer.echo(f"model: {cfg.model_name}")
    if cfg.reasoning_effort:
        typer.echo(f"reasoning_effort: {cfg.reasoning_effort}")

    if not skip_models:
        models = asyncio.run(_list_models(cfg))
        typer.echo("\nModels:")
        if models:
            for model_id in models:
                typer.echo(f"  - {model_id}")
        else:
            typer.echo("  (none listed)")

    if not skip_chat:
        typer.echo("\nChat:")
        try:
            typer.echo(asyncio.run(_inspect_chat(cfg)))
        except LlmGatewayError as exc:
            typer.echo(f"[{exc.code}] {exc.detail}", err=True)
            raise typer.Exit(1) from exc


def _messages_from_input(value: Any) -> list[dict[str, str]]:
    if isinstance(value, dict) and isinstance(value.get("messages"), list):
        return value["messages"]
    prompt = value.get("prompt") if isinstance(value, dict) else value
    if not isinstance(prompt, str):
        prompt = str(prompt)
    return [{"role": "user", "content": prompt}]


@app.command("eval")
def eval_cases(
    cases_file: Annotated[Path, typer.Argument(help="JSON or JSONL eval case file.")],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Optional JSON report output path."),
    ] = None,
    llm: Annotated[
        bool,
        typer.Option("--llm", help="Call the configured LLM instead of echoing input."),
    ] = False,
    dotenv: Annotated[Path | None, typer.Option("--dotenv")] = Path(".env"),
    max_tokens: Annotated[int, typer.Option("--max-tokens")] = 600,
    timeout_seconds: Annotated[float, typer.Option("--timeout-seconds")] = 60.0,
) -> None:
    """Run JSON/JSONL eval cases through an echo target or configured LLM."""

    cases = load_eval_cases(cases_file)

    async def run() -> None:
        if llm:
            _load_dotenv(dotenv)
            cfg = resolve_llm_config()
            async with httpx.AsyncClient() as client:

                async def target(value: Any) -> str:
                    return await call_chat_completion(
                        _messages_from_input(value),
                        cfg=cfg,
                        http_client=client,
                        max_tokens=max_tokens,
                        timeout_seconds=timeout_seconds,
                        use_response_format=False,
                    )

                report = await run_evals(cases, target, scorer=_exact_scorer)
        else:
            report = await run_evals(cases, lambda value: value, scorer=_exact_scorer)

        if output:
            write_eval_report(report, output)
        typer.echo(report.to_json(indent=2))
        raise typer.Exit(0 if report.failed == 0 else 1)

    asyncio.run(run())


def _exact_scorer(*, output: Any, expected: Any | None, case: EvalCase) -> dict[str, Any]:
    del case
    if expected is None:
        return {"ok": True}
    return {"ok": output == expected}
