"""Generic evaluation runner for LLM calls and workflows."""

from __future__ import annotations

import inspect
import json
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field

from llmkit_lite.structured import StructuredCallResult


class EvalCase(BaseModel):
    id: str
    input: Any
    expected: Any | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalRecord(BaseModel):
    id: str
    ok: bool
    latency_ms: float
    output: Any | None = None
    error_code: str | None = None
    error_detail: str | None = None
    score: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalReport(BaseModel):
    cases: list[EvalRecord]

    @property
    def passed(self) -> int:
        return sum(1 for case in self.cases if case.ok)

    @property
    def failed(self) -> int:
        return len(self.cases) - self.passed

    def to_json(self, *, indent: int | None = 2) -> str:
        return self.model_dump_json(indent=indent)


class Scorer(Protocol):
    def __call__(
        self,
        *,
        output: Any,
        expected: Any | None,
        case: EvalCase,
    ) -> Mapping[str, Any]:
        ...


Target = Callable[[Any], Any | Awaitable[Any]]


def load_eval_cases(path: str | Path) -> list[EvalCase]:
    """Load eval cases from a JSON array or newline-delimited JSON file."""

    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        rows = json.loads(text)
    if not isinstance(rows, list):
        raise ValueError("eval case file must contain a JSON array or JSONL records")
    return [EvalCase.model_validate(row) for row in rows]


def write_eval_report(report: EvalReport, path: str | Path) -> None:
    Path(path).write_text(report.to_json(indent=2) + "\n", encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, StructuredCallResult):
        return {
            "ok": value.ok,
            "value": _jsonable(value.value) if value.value is not None else None,
            "raw_content": value.raw_content,
            "json_text": value.json_text,
            "error_code": value.error_code,
            "error_detail": value.error_detail,
        }
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _record_from_output(
    *,
    case: EvalCase,
    output: Any,
    latency_ms: float,
    scorer: Scorer | None,
) -> EvalRecord:
    score = dict(scorer(output=output, expected=case.expected, case=case)) if scorer else None
    ok = bool(score.get("ok", True)) if score is not None else True
    error_code = None
    error_detail = None
    if isinstance(output, StructuredCallResult):
        ok = output.ok and ok
        error_code = output.error_code
        error_detail = output.error_detail
        output_value = output.value
    else:
        output_value = output
    return EvalRecord(
        id=case.id,
        ok=ok,
        latency_ms=latency_ms,
        output=_jsonable(output_value),
        error_code=error_code,
        error_detail=error_detail,
        score=score,
        metadata=case.metadata,
    )


async def run_evals(
    cases: Iterable[EvalCase],
    target: Target,
    *,
    scorer: Scorer | None = None,
) -> EvalReport:
    """Run eval cases against an async or sync target callable."""

    records: list[EvalRecord] = []
    for case in cases:
        start = time.perf_counter()
        try:
            output = target(case.input)
            if inspect.isawaitable(output):
                output = await output
            latency_ms = (time.perf_counter() - start) * 1000
            records.append(
                _record_from_output(
                    case=case,
                    output=output,
                    latency_ms=latency_ms,
                    scorer=scorer,
                )
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            records.append(
                EvalRecord(
                    id=case.id,
                    ok=False,
                    latency_ms=latency_ms,
                    error_code=exc.__class__.__name__,
                    error_detail=str(exc),
                    metadata=case.metadata,
                )
            )
    return EvalReport(cases=records)
