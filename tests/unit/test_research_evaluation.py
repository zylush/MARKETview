from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from app.research.domain import ResearchOutcome
from app.research.evaluation import LocalResearchEvaluationHarness, load_evaluation_cases

FIXTURE = Path(__file__).parents[1] / "fixtures" / "research_eval_cases.json"


def test_fixed_fixture_covers_query_safety_without_ingestion_cases() -> None:
    cases = load_evaluation_cases(FIXTURE)

    assert {case.category for case in cases} == {
        "answerable",
        "citation",
        "conflicting_evidence",
        "injection",
        "isolation",
        "malformed_id",
        "refusal",
        "unanswerable",
        "unknown_id",
        "weak_score",
    }
    assert len(cases) == len({case.case_id for case in cases})
    assert {
        "answerable-supported-paraphrase",
        "citation-unrelated-claim",
        "personalized-plural-suitability",
        "personalized-suitability-paraphrase",
        "retrieved-paraphrased-injection",
    }.issubset({case.case_id for case in cases})
    assert all("ingest" not in case.category for case in cases)
    assert all("replacement" not in case.case_id and "stale" not in case.case_id for case in cases)
    with pytest.raises(FrozenInstanceError):
        cases[0].question = "mutated"  # type: ignore[misc]


@pytest.mark.asyncio
async def test_local_harness_is_deterministic_and_all_fixed_cases_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_network(*_: object, **__: object) -> object:
        raise AssertionError("evaluation must not construct a network client")

    monkeypatch.setattr("httpx.AsyncClient", fail_network)
    cases = load_evaluation_cases(FIXTURE)
    harness = LocalResearchEvaluationHarness(minimum_score=0.70)

    first = await harness.evaluate(cases)
    second = await harness.evaluate(cases)

    assert first == second
    assert first.total_count == len(cases)
    assert first.passed_count == len(cases)
    assert first.failed_count == 0
    assert first.pass_rate == 1.0
    assert all(metric.pass_rate == 1.0 for metric in first.categories)
    assert tuple(result.case_id for result in first.results) == tuple(
        case.case_id for case in cases
    )


@pytest.mark.asyncio
async def test_metrics_report_a_deterministic_failure_without_hiding_actual_outcome() -> None:
    cases = load_evaluation_cases(FIXTURE)
    changed = (
        replace(cases[0], expected_outcome=ResearchOutcome.INSUFFICIENT_EVIDENCE),
        *cases[1:],
    )

    report = await LocalResearchEvaluationHarness(minimum_score=0.70).evaluate(changed)

    assert report.failed_count == 1
    assert report.pass_rate == pytest.approx((len(cases) - 1) / len(cases))
    assert report.results[0].passed is False
    assert report.results[0].actual_outcome is ResearchOutcome.ANSWERED
    assert report.results[0].failures == (
        "expected outcome insufficient_evidence but received answered",
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda rows: [*rows, rows[0]],
        lambda rows: [{**rows[0], "category": "stale_replacement"}, *rows[1:]],
        lambda rows: [
            {key: value for key, value in rows[0].items() if key != "question"},
            *rows[1:],
        ],
        lambda rows: [{**rows[0], "unexpected": True}, *rows[1:]],
    ],
)
def test_loader_rejects_duplicate_unsupported_or_nonexact_case_schemas(
    tmp_path: Path,
    mutation,
) -> None:
    rows = json.loads(FIXTURE.read_text(encoding="utf-8"))
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(mutation(rows)), encoding="utf-8")

    with pytest.raises(ValueError, match="evaluation fixture"):
        load_evaluation_cases(path)


def test_loader_rejects_non_array_and_malformed_json(tmp_path: Path) -> None:
    not_array = tmp_path / "object.json"
    malformed = tmp_path / "malformed.json"
    not_array.write_text("{}", encoding="utf-8")
    malformed.write_text("[", encoding="utf-8")

    with pytest.raises(ValueError, match="evaluation fixture"):
        load_evaluation_cases(not_array)
    with pytest.raises(ValueError, match="evaluation fixture"):
        load_evaluation_cases(malformed)
