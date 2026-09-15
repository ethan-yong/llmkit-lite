import json

import pytest
from pydantic import ValidationError

from llmkit_lite.settings import (
    DashboardSettings,
    LlmSettings,
    SettingsStore,
    SettingsStoreError,
)


def test_given_no_saved_file_when_loading_then_defaults_are_returned(tmp_path) -> None:
    store = SettingsStore(tmp_path / "settings.json")

    settings = store.load()

    assert settings.llm.provider == "local"
    assert settings.llm.model_name == "qwen2.5"
    assert settings.generation.max_tokens == 600
    assert settings.workflow.heartbeat_interval_seconds == 10


def test_given_settings_when_saving_then_document_can_be_loaded(tmp_path) -> None:
    store = SettingsStore(tmp_path / "private" / "settings.json")
    settings = DashboardSettings(
        llm=LlmSettings(
            provider="deepseek",
            base_url="https://api.deepseek.com/v1/",
            model_name="deepseek-chat",
            api_key="private-secret",
        )
    )

    store.save(settings)
    loaded = store.load()

    assert loaded == settings
    assert loaded.llm.base_url == "https://api.deepseek.com/v1"
    assert json.loads(store.path.read_text(encoding="utf-8"))["llm"]["api_key"] == (
        "private-secret"
    )


def test_given_saved_secret_when_serializing_publicly_then_secret_is_redacted() -> None:
    settings = DashboardSettings(llm=LlmSettings(api_key="private-secret"))

    public = settings.public_dict()

    assert public["llm"]["api_key"] == ""
    assert public["llm"]["api_key_configured"] is True
    assert "private-secret" not in repr(settings)
    assert "private-secret" not in json.dumps(public)


def test_given_blank_key_when_updating_then_saved_secret_is_preserved(tmp_path) -> None:
    store = SettingsStore(tmp_path / "settings.json")
    original = DashboardSettings(llm=LlmSettings(api_key="private-secret"))
    store.save(original)
    payload = original.model_dump(mode="json")
    payload["llm"]["api_key"] = ""

    updated = store.update(payload)

    assert updated.llm.api_key == "private-secret"
    assert store.load().llm.api_key == "private-secret"


def test_given_clear_flag_when_updating_then_saved_secret_is_removed(tmp_path) -> None:
    store = SettingsStore(tmp_path / "settings.json")
    original = DashboardSettings(llm=LlmSettings(api_key="private-secret"))
    store.save(original)
    payload = original.model_dump(mode="json")
    payload["llm"]["api_key"] = ""
    payload["clear_api_key"] = True

    updated = store.update(payload)

    assert updated.llm.api_key is None
    assert store.load().llm.api_key is None


def test_given_invalid_saved_json_when_loading_then_safe_error_is_raised(
    tmp_path,
) -> None:
    path = tmp_path / "settings.json"
    path.write_text("private invalid json", encoding="utf-8")
    store = SettingsStore(path)

    with pytest.raises(SettingsStoreError) as exc_info:
        store.load()

    assert exc_info.value.code == "settings_invalid_file"
    assert "private" not in str(exc_info.value)


@pytest.mark.parametrize(
    "updates",
    [
        {"workflow": {"heartbeat_interval_seconds": 30}},
        {"reliability": {"initial_backoff_seconds": 3, "max_backoff_seconds": 2}},
        {"llm": {"base_url": "not-a-url"}},
    ],
)
def test_given_invalid_relationships_when_validating_then_settings_are_rejected(
    updates,
) -> None:
    payload = DashboardSettings().model_dump()
    for section, values in updates.items():
        payload[section].update(values)

    with pytest.raises(ValidationError):
        DashboardSettings.model_validate(payload)


def test_given_dashboard_settings_when_resolving_then_runtime_models_match() -> None:
    settings = DashboardSettings()

    endpoint = settings.to_llm_config()
    resilience = settings.to_resilience_policy()
    tracing = settings.to_tracing_settings()

    assert endpoint.base_url == settings.llm.base_url
    assert endpoint.model_name == settings.llm.model_name
    assert resilience.max_attempts_per_route == 2
    assert tracing.service_name == "llmkit-lite"
    assert tracing.environment == "development"
