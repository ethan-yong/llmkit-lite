"""Local FastAPI dashboard for configuring llmkit-lite without editing code."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import ValidationError

from llmkit_lite.llm import auth_headers, normalize_api_base
from llmkit_lite.settings import DashboardSettings, SettingsStore, SettingsStoreError

_STATIC_ROOT = Path(__file__).with_name("static")
_LOCAL_HOSTS = frozenset({"127.0.0.1", "::1", "localhost", "testclient"})


def create_settings_app(
    settings_store: SettingsStore | None = None,
    *,
    http_client_factory: Callable[[], httpx.AsyncClient] | None = None,
    local_only: bool = True,
) -> FastAPI:
    """Create a standalone dashboard app that can also be mounted by FastAPI."""

    store = settings_store or SettingsStore()
    client_factory = http_client_factory or httpx.AsyncClient
    app = FastAPI(
        title="LLMKit Settings",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.settings_store = store

    @app.middleware("http")
    async def protect_dashboard(request: Request, call_next):
        client_host = request.client.host if request.client is not None else ""
        if local_only and client_host not in _LOCAL_HOSTS:
            response = JSONResponse(
                status_code=403,
                content={"error": "settings_local_access_only"},
            )
        else:
            response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'"
        )
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        return response

    @app.get("/")
    async def settings_page():
        return FileResponse(_STATIC_ROOT / "settings.html", media_type="text/html")

    @app.get("/assets/settings.css")
    async def settings_styles():
        return FileResponse(_STATIC_ROOT / "settings.css", media_type="text/css")

    @app.get("/assets/settings.js")
    async def settings_script():
        return FileResponse(
            _STATIC_ROOT / "settings.js",
            media_type="text/javascript",
        )

    @app.get("/favicon.ico")
    async def empty_favicon():
        return Response(status_code=204)

    @app.get("/api/settings")
    async def read_settings():
        try:
            settings = store.load()
        except SettingsStoreError as exc:
            return JSONResponse(status_code=500, content={"error": exc.code})
        return settings.public_dict()

    @app.put("/api/settings")
    async def update_settings(request: Request):
        try:
            payload = await request.json()
        except ValueError:
            return JSONResponse(
                status_code=400,
                content={"error": "settings_invalid_json"},
            )
        try:
            settings = store.update(payload)
        except ValidationError as exc:
            return JSONResponse(
                status_code=422,
                content={
                    "error": "settings_validation_failed",
                    "fields": _validation_fields(exc),
                },
            )
        except (TypeError, ValueError):
            return JSONResponse(
                status_code=422,
                content={"error": "settings_validation_failed"},
            )
        except SettingsStoreError as exc:
            return JSONResponse(status_code=500, content={"error": exc.code})
        return settings.public_dict()

    @app.post("/api/test-connection")
    async def test_connection():
        try:
            settings = store.load()
            result = await inspect_connection(settings, client_factory)
        except SettingsStoreError as exc:
            return JSONResponse(status_code=500, content={"error": exc.code})
        return result

    return app


async def inspect_connection(
    settings: DashboardSettings,
    client_factory: Callable[[], httpx.AsyncClient] = httpx.AsyncClient,
) -> dict[str, Any]:
    """Check the configured OpenAI-compatible models endpoint without a chat call."""

    if not isinstance(settings, DashboardSettings):
        raise TypeError("settings must be DashboardSettings")
    if not callable(client_factory):
        raise TypeError("HTTP client factory must be callable")
    cfg = settings.to_llm_config()
    root = cfg.base_url.rstrip("/")
    api_base = normalize_api_base(cfg.base_url)
    candidates = [api_base]
    if root not in candidates:
        candidates.append(root)
    if root.endswith("/v1"):
        without_version = root[:-3].rstrip("/")
        if without_version and without_version not in candidates:
            candidates.append(without_version)

    async with client_factory() as client:
        for base in candidates:
            try:
                response = await client.get(
                    f"{base}/models",
                    headers=auth_headers(cfg),
                    timeout=min(settings.generation.timeout_seconds, 15.0),
                )
            except httpx.HTTPError:
                continue
            if response.status_code != 200:
                continue
            models = _model_ids(response)
            return {
                "ok": True,
                "message": (
                    "Connection successful."
                    if models
                    else "Connection successful; no models were listed."
                ),
                "models": models[:50],
            }

    return {
        "ok": False,
        "message": "Could not reach a compatible models endpoint.",
        "models": [],
    }


def _model_ids(response: httpx.Response) -> list[str]:
    try:
        payload = response.json()
    except ValueError:
        return []
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        return []
    seen: set[str] = set()
    models: list[str] = []
    for item in payload["data"]:
        model_id = item.get("id") if isinstance(item, dict) else None
        if isinstance(model_id, str):
            normalized = model_id.strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                models.append(normalized)
    return models


def _validation_fields(exc: ValidationError) -> list[dict[str, str]]:
    fields: list[dict[str, str]] = []
    for error in exc.errors(include_input=False, include_url=False):
        location = ".".join(str(part) for part in error.get("loc", ()))
        fields.append(
            {
                "field": location,
                "message": str(error.get("msg", "Invalid value")),
            }
        )
    return fields
