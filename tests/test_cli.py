import json

from typer.testing import CliRunner

from llmkit_lite.cli import app


runner = CliRunner()


def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Utilities for production LLM services" in result.output


def test_inspect_llm_prints_config_without_network() -> None:
    result = runner.invoke(
        app,
        [
            "inspect-llm",
            "--provider",
            "local",
            "--base-url",
            "http://gateway.local",
            "--model",
            "demo-model",
            "--skip-models",
            "--skip-chat",
        ],
    )
    assert result.exit_code == 0
    assert "provider: local" in result.output
    assert "api_base: http://gateway.local/v1" in result.output
    assert "model: demo-model" in result.output


def test_eval_echo_passes_exact_cases(tmp_path) -> None:
    cases = tmp_path / "cases.json"
    cases.write_text(
        json.dumps([{"id": "one", "input": {"value": 1}, "expected": {"value": 1}}]),
        encoding="utf-8",
    )
    result = runner.invoke(app, ["eval", str(cases)])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["cases"][0]["ok"] is True


def test_eval_echo_fails_exact_mismatch(tmp_path) -> None:
    cases = tmp_path / "cases.json"
    cases.write_text(
        json.dumps([{"id": "one", "input": {"value": 1}, "expected": {"value": 2}}]),
        encoding="utf-8",
    )
    result = runner.invoke(app, ["eval", str(cases)])
    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["cases"][0]["ok"] is False


def test_eval_writes_report(tmp_path) -> None:
    cases = tmp_path / "cases.jsonl"
    output = tmp_path / "report.json"
    cases.write_text('{"id":"one","input":"hi","expected":"hi"}\n', encoding="utf-8")
    result = runner.invoke(app, ["eval", str(cases), "--output", str(output)])
    assert result.exit_code == 0
    assert output.is_file()
    assert json.loads(output.read_text(encoding="utf-8"))["cases"][0]["id"] == "one"
