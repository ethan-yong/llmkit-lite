import json

from pydantic import BaseModel

from llmkit_lite.evals import (
    EvalCase,
    EvalReport,
    load_eval_cases,
    run_evals,
    write_eval_report,
)
from llmkit_lite.structured import StructuredCallResult


class OutputModel(BaseModel):
    value: int


async def test_run_evals_supports_async_targets() -> None:
    cases = [EvalCase(id="one", input=1, expected=2)]

    async def target(value: int) -> OutputModel:
        return OutputModel(value=value + 1)

    report = await run_evals(cases, target)
    assert report.passed == 1
    assert report.failed == 0
    assert report.cases[0].output == {"value": 2}
    assert report.cases[0].latency_ms >= 0


async def test_run_evals_uses_scorer() -> None:
    cases = [EvalCase(id="one", input=1, expected={"value": 3})]

    def target(value: int) -> dict[str, int]:
        return {"value": value + 1}

    def scorer(*, output, expected, case):
        assert case.id == "one"
        return {"ok": output == expected}

    report = await run_evals(cases, target, scorer=scorer)
    assert report.failed == 1
    assert report.cases[0].score == {"ok": False}


async def test_run_evals_records_target_exceptions() -> None:
    cases = [EvalCase(id="bad", input="x")]

    def target(_: str):
        raise ValueError("nope")

    report = await run_evals(cases, target)
    assert report.failed == 1
    assert report.cases[0].error_code == "ValueError"
    assert report.cases[0].error_detail == "nope"


async def test_run_evals_understands_structured_call_result_failure() -> None:
    cases = [EvalCase(id="bad", input="x")]

    def target(_: str):
        return StructuredCallResult.failure(
            raw_content="not json",
            error_code="llm_unparseable_content",
            error_detail="no JSON object found",
        )

    report = await run_evals(cases, target)
    assert report.failed == 1
    assert report.cases[0].error_code == "llm_unparseable_content"
    assert report.cases[0].output is None


def test_load_eval_cases_from_json_array(tmp_path) -> None:
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps([{"id": "one", "input": {"text": "hi"}, "metadata": {"a": 1}}]),
        encoding="utf-8",
    )
    cases = load_eval_cases(path)
    assert cases == [
        EvalCase(id="one", input={"text": "hi"}, metadata={"a": 1}),
    ]


def test_load_eval_cases_from_jsonl(tmp_path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        '{"id":"one","input":1}\n{"id":"two","input":2}\n',
        encoding="utf-8",
    )
    cases = load_eval_cases(path)
    assert [case.id for case in cases] == ["one", "two"]


def test_write_eval_report(tmp_path) -> None:
    report = EvalReport(
        cases=[
            {
                "id": "one",
                "ok": True,
                "latency_ms": 1.2,
                "output": {"value": 2},
            }
        ]
    )
    path = tmp_path / "report.json"
    write_eval_report(report, path)
    assert json.loads(path.read_text(encoding="utf-8"))["cases"][0]["id"] == "one"
