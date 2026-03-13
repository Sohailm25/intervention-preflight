from __future__ import annotations

from intervention_preflight.parity import (
    check_batch_single_parity,
    compare_output_sequences,
    compare_position_modes,
)


def test_compare_output_sequences_passes_on_identical_strings() -> None:
    report = compare_output_sequences(["alpha", "beta"], ["alpha", "beta"])
    assert report["status"] == "pass"
    assert report["metrics"]["max_delta"] == 0.0


def test_compare_output_sequences_warns_on_numeric_difference() -> None:
    report = compare_output_sequences([[1.0, 2.0]], [[1.0, 2.5]], tolerance=0.1)
    assert report["status"] == "warn"
    assert report["metrics"]["failing_count"] == 1


def test_check_batch_single_parity_surfaces_drift() -> None:
    prompts = ["a", "b", "c"]

    def run_single(prompt: str) -> str:
        return prompt.upper()

    def run_batch(values: list[str]) -> list[str]:
        return [value.upper() for value in values[:2]] + ["DRIFT"]

    report = check_batch_single_parity(prompts, run_single=run_single, run_batch=run_batch, tolerance=0.0)
    assert report["status"] == "warn"
    assert report["metrics"]["failing_count"] == 1
    assert report["details"]["prompt_count"] == 3


def test_compare_position_modes_reports_worst_pair() -> None:
    report = compare_position_modes(
        {
            "prompt_last": ["alpha", "beta"],
            "response_last": ["alpha", "beta changed"],
            "response_mean": ["alpha changed", "beta changed"],
        },
        tolerance=0.0,
    )
    assert report["status"] == "warn"
    assert report["summary"]["worst_pair"] is not None
    assert report["metrics"]["pair_count"] == 3
