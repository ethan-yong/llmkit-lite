import httpx

from llmkit_lite.dashboard import create_settings_app
from llmkit_lite.llm import LlmCapability
from llmkit_lite.settings import DashboardSettings, LlmSettings, SettingsStore


async def request(app, method: str, path: str, **kwargs) -> httpx.Response:
    transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 45123))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.request(method, path, **kwargs)


async def test_given_dashboard_when_opening_then_frontend_assets_are_served(
    tmp_path,
) -> None:
    app = create_settings_app(SettingsStore(tmp_path / "settings.json"))

    page = await request(app, "GET", "/")
    styles = await request(app, "GET", "/assets/settings.css")
    script = await request(app, "GET", "/assets/settings.js")

    assert page.status_code == 200
    assert "Configure your model stack" in page.text
    assert "API key" in page.text
    assert styles.status_code == 200
    assert "--accent" in styles.text
    assert script.status_code == 200
    assert "saveSettings" in script.text
    assert page.headers["x-frame-options"] == "DENY"


async def test_given_saved_secret_when_reading_api_then_secret_is_never_returned(
    tmp_path,
) -> None:
    store = SettingsStore(tmp_path / "settings.json")
    store.save(DashboardSettings(llm=LlmSettings(api_key="private-secret")))
    app = create_settings_app(store)

    response = await request(app, "GET", "/api/settings")

    assert response.status_code == 200
    assert response.json()["llm"]["api_key"] == ""
    assert response.json()["llm"]["api_key_configured"] is True
    assert "private-secret" not in response.text


async def test_given_valid_form_when_saving_then_settings_are_persisted(
    tmp_path,
) -> None:
    store = SettingsStore(tmp_path / "settings.json")
    app = create_settings_app(store)
    payload = DashboardSettings().model_dump(mode="json")
    payload["llm"].update(
        {
            "provider": "deepseek",
            "base_url": "https://api.deepseek.com/v1",
            "model_name": "deepseek-chat",
            "api_key": "private-secret",
            "capabilities": ["text", "tool_calling"],
        }
    )

    response = await request(app, "PUT", "/api/settings", json=payload)

    assert response.status_code == 200
    assert response.json()["llm"]["api_key_configured"] is True
    assert "private-secret" not in response.text
    assert store.load().llm.api_key == "private-secret"
    assert store.load().llm.model_name == "deepseek-chat"
    assert store.load().llm.capabilities == {
        LlmCapability.TEXT,
        LlmCapability.TOOL_CALLING,
    }


async def test_given_invalid_form_when_saving_then_field_errors_are_safe(
    tmp_path,
) -> None:
    store = SettingsStore(tmp_path / "settings.json")
    app = create_settings_app(store)
    payload = DashboardSettings().model_dump(mode="json")
    payload["llm"]["base_url"] = "private invalid url"

    response = await request(app, "PUT", "/api/settings", json=payload)

    assert response.status_code == 422
    assert response.json()["error"] == "settings_validation_failed"
    assert response.json()["fields"][0]["field"] == "llm.base_url"
    assert "private invalid url" not in response.text
    assert not store.path.exists()


async def test_given_remote_client_when_opening_local_dashboard_then_access_is_denied(
    tmp_path,
) -> None:
    app = create_settings_app(SettingsStore(tmp_path / "settings.json"))
    transport = httpx.ASGITransport(app=app, client=("10.20.30.40", 45123))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/")

    assert response.status_code == 403
    assert response.json() == {"error": "settings_local_access_only"}


async def test_given_provider_models_when_testing_then_connection_and_models_return(
    tmp_path,
) -> None:
    requests: list[httpx.Request] = []

    def provider(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"data": [{"id": "model-b"}, {"id": "model-a"}]},
        )

    store = SettingsStore(tmp_path / "settings.json")
    store.save(
        DashboardSettings(
            llm=LlmSettings(
                base_url="https://provider.example/v1",
                model_name="model-a",
                api_key="private-secret",
            )
        )
    )
    app = create_settings_app(
        store,
        http_client_factory=lambda: httpx.AsyncClient(
            transport=httpx.MockTransport(provider)
        ),
    )

    response = await request(app, "POST", "/api/test-connection")

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "message": "Connection successful.",
        "models": ["model-b", "model-a"],
    }
    assert requests[0].url == "https://provider.example/v1/models"
    assert requests[0].headers["authorization"] == "Bearer private-secret"
    assert "private-secret" not in response.text


async def test_given_unreachable_provider_when_testing_then_safe_failure_returns(
    tmp_path,
) -> None:
    def unavailable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("private network detail", request=request)

    app = create_settings_app(
        SettingsStore(tmp_path / "settings.json"),
        http_client_factory=lambda: httpx.AsyncClient(
            transport=httpx.MockTransport(unavailable)
        ),
    )

    response = await request(app, "POST", "/api/test-connection")

    assert response.status_code == 200
    assert response.json()["ok"] is False
    assert "private" not in response.text
