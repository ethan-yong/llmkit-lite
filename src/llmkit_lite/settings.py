"""Persistent application settings managed by the local web dashboard."""

from __future__ import annotations

import json
import os
import threading
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from llmkit_lite.llm import LlmCapabilities, LlmCapability, LlmEndpointConfig
from llmkit_lite.observability import TracingSettings
from llmkit_lite.routing import LlmResiliencePolicy

_ERROR_DETAILS = {
    "settings_invalid_file": "saved application settings are invalid",
    "settings_write_failed": "application settings could not be saved",
}


class SettingsStoreError(Exception):
    """Safe persistent-settings failure for application boundaries."""

    def __init__(self, code: str) -> None:
        if code not in _ERROR_DETAILS:
            raise ValueError("unsupported settings store error code")
        detail = _ERROR_DETAILS[code]
        super().__init__(detail)
        self.code = code
        self.detail = detail


class _SettingsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LlmSettings(_SettingsModel):
    """Provider endpoint selected by the application."""

    provider: Literal["local", "vllm", "litellm", "deepseek"] = "local"
    base_url: str = "http://127.0.0.1:4000"
    model_name: str = "qwen2.5"
    api_key: str | None = Field(default=None, repr=False)
    reasoning_effort: str | None = None
    capabilities: set[LlmCapability] = Field(
        default_factory=lambda: {LlmCapability.TEXT}
    )

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        normalized = _required_text(value, "LLM base URL")
        parsed = urlparse(normalized)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("LLM base URL must be an HTTP or HTTPS URL")
        return normalized.rstrip("/")

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, value: str) -> str:
        return _required_text(value, "LLM model name")

    @field_validator("api_key", "reasoning_effort")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        return _optional_text(value)


class GenerationSettings(_SettingsModel):
    """Default chat-completion request values."""

    max_tokens: int = Field(default=600, ge=1, le=1_000_000)
    timeout_seconds: float = Field(default=60.0, gt=0, le=3_600)
    temperature: float = Field(default=0.0, ge=0, le=2)
    use_response_format: bool = True


class ReliabilitySettings(_SettingsModel):
    """Retry and circuit-breaker defaults."""

    max_attempts_per_route: int = Field(default=2, ge=1, le=20)
    initial_backoff_seconds: float = Field(default=0.25, ge=0, le=300)
    backoff_multiplier: float = Field(default=2.0, ge=1, le=10)
    max_backoff_seconds: float = Field(default=2.0, ge=0, le=3_600)
    jitter_ratio: float = Field(default=0.2, ge=0, le=1)
    circuit_failure_threshold: int = Field(default=3, ge=1, le=10_000)
    circuit_recovery_timeout_seconds: float = Field(default=30.0, gt=0, le=86_400)

    @model_validator(mode="after")
    def validate_backoff_range(self) -> ReliabilitySettings:
        if self.max_backoff_seconds < self.initial_backoff_seconds:
            raise ValueError("maximum backoff must be at least the initial backoff")
        return self


class ObservabilitySettings(_SettingsModel):
    """Logging and tracing defaults."""

    tracing_enabled: bool = False
    service_name: str = "llmkit-lite"
    service_version: str | None = None
    environment: str | None = "development"
    otlp_endpoint: str | None = None
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    @field_validator("service_name")
    @classmethod
    def validate_service_name(cls, value: str) -> str:
        return _required_text(value, "service name")

    @field_validator("service_version", "environment")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        return _optional_text(value)

    @field_validator("otlp_endpoint")
    @classmethod
    def validate_otlp_endpoint(cls, value: str | None) -> str | None:
        normalized = _optional_text(value)
        if normalized is None:
            return None
        parsed = urlparse(normalized)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("OTLP endpoint must be an HTTP or HTTPS URL")
        return normalized.rstrip("/")


class WorkflowSettings(_SettingsModel):
    """Workflow and worker-recovery timing defaults."""

    execution_timeout_seconds: float = Field(default=30.0, gt=0, le=86_400)
    lease_duration_seconds: float = Field(default=30.0, gt=0, le=86_400)
    queue_visibility_seconds: float = Field(default=30.0, gt=0, le=86_400)
    heartbeat_interval_seconds: float = Field(default=10.0, gt=0, le=86_400)

    @model_validator(mode="after")
    def validate_heartbeat_interval(self) -> WorkflowSettings:
        if self.heartbeat_interval_seconds >= self.lease_duration_seconds:
            raise ValueError("heartbeat interval must be shorter than lease duration")
        if self.heartbeat_interval_seconds >= self.queue_visibility_seconds:
            raise ValueError("heartbeat interval must be shorter than queue visibility")
        return self


class DashboardSettings(_SettingsModel):
    """Complete versioned settings document edited by the dashboard."""

    version: Literal[1] = 1
    llm: LlmSettings = Field(default_factory=LlmSettings)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    reliability: ReliabilitySettings = Field(default_factory=ReliabilitySettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    workflow: WorkflowSettings = Field(default_factory=WorkflowSettings)

    def to_llm_config(self) -> LlmEndpointConfig:
        """Build the endpoint configuration consumed by the LLM client."""

        return LlmEndpointConfig(
            provider=self.llm.provider,
            base_url=self.llm.base_url,
            model_name=self.llm.model_name,
            api_key=self.llm.api_key,
            reasoning_effort=self.llm.reasoning_effort,
            capabilities=LlmCapabilities(self.llm.capabilities),
        )

    def to_resilience_policy(self) -> LlmResiliencePolicy:
        """Build the retry and circuit-breaker policy consumed by the router."""

        return LlmResiliencePolicy(**self.reliability.model_dump())

    def to_tracing_settings(self) -> TracingSettings:
        """Build the tracing configuration consumed during application startup."""

        values = self.observability
        return TracingSettings(
            enabled=values.tracing_enabled,
            service_name=values.service_name,
            service_version=values.service_version,
            environment=values.environment,
            otlp_endpoint=values.otlp_endpoint,
        )

    def public_dict(self) -> dict[str, Any]:
        """Return browser-safe settings without the stored API key."""

        payload = self.model_dump(mode="json")
        payload["llm"]["api_key"] = ""
        payload["llm"]["api_key_configured"] = bool(self.llm.api_key)
        return payload


class SettingsStore:
    """Load and atomically save one dashboard settings document."""

    def __init__(self, path: Path | str | None = None) -> None:
        resolved = default_settings_path() if path is None else Path(path)
        self.path = resolved.expanduser().resolve()
        self._lock = threading.RLock()

    def load(self) -> DashboardSettings:
        """Return saved settings, or defaults when the file does not exist."""

        with self._lock:
            if not self.path.exists():
                return DashboardSettings()
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
                return DashboardSettings.model_validate(payload)
            except SettingsStoreError:
                raise
            except Exception as exc:
                raise SettingsStoreError("settings_invalid_file") from exc

    def update(self, payload: Mapping[str, Any]) -> DashboardSettings:
        """Validate a browser update while preserving an unchanged secret."""

        if not isinstance(payload, Mapping):
            raise TypeError("settings update must be a mapping")
        with self._lock:
            current = self.load()
            candidate = dict(payload)
            clear_api_key = candidate.pop("clear_api_key", False)
            if not isinstance(clear_api_key, bool):
                raise TypeError("clear_api_key must be a boolean")
            llm_payload = candidate.get("llm")
            if not isinstance(llm_payload, Mapping):
                raise TypeError("settings update must include LLM settings")
            merged_llm = dict(llm_payload)
            submitted_key = merged_llm.get("api_key")
            if clear_api_key:
                merged_llm["api_key"] = None
            elif submitted_key is None or (
                isinstance(submitted_key, str) and not submitted_key.strip()
            ):
                merged_llm["api_key"] = current.llm.api_key
            candidate["llm"] = merged_llm
            settings = DashboardSettings.model_validate(candidate)
            self.save(settings)
            return settings

    def save(self, settings: DashboardSettings) -> None:
        """Atomically replace the saved settings document."""

        if not isinstance(settings, DashboardSettings):
            raise TypeError("settings must be DashboardSettings")
        temporary: Path | None = None
        with self._lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                temporary = self.path.parent / (
                    f".{self.path.name}.{uuid.uuid4().hex}.tmp"
                )
                temporary.write_text(
                    settings.model_dump_json(indent=2) + "\n",
                    encoding="utf-8",
                )
                try:
                    temporary.chmod(0o600)
                except OSError:
                    pass
                os.replace(temporary, self.path)
            except Exception as exc:
                if temporary is not None:
                    try:
                        temporary.unlink(missing_ok=True)
                    except OSError:
                        pass
                raise SettingsStoreError("settings_write_failed") from exc


def default_settings_path() -> Path:
    """Return the workspace-local settings path used by the standalone UI."""

    return Path.cwd() / ".llmkit" / "settings.json"


def _required_text(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("optional setting must be a string")
    return value.strip() or None
