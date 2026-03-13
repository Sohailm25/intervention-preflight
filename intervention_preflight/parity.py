"""Parity and consistency checks for intervention workflows."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import combinations
from typing import Any

import numpy as np

from intervention_preflight.prompt_audit import lexical_similarity
from intervention_preflight.report import build_report


def _numeric_delta(left: Any, right: Any) -> float:
    left_arr = np.asarray(left, dtype=np.float64).reshape(-1)
    right_arr = np.asarray(right, dtype=np.float64).reshape(-1)
    if left_arr.shape != right_arr.shape:
        raise ValueError(
            f"Numeric parity comparison requires matching shapes, got {left_arr.shape} vs {right_arr.shape}"
        )
    if left_arr.size == 0:
        return 0.0
    return float(np.max(np.abs(left_arr - right_arr)))


def default_delta(left: Any, right: Any) -> float:
    if isinstance(left, str) and isinstance(right, str):
        return float(1.0 - lexical_similarity(left, right))
    return _numeric_delta(left, right)


def compare_output_sequences(
    left_outputs: Sequence[Any],
    right_outputs: Sequence[Any],
    *,
    delta_fn: Callable[[Any, Any], float] | None = None,
    tolerance: float = 0.0,
    left_label: str = "left",
    right_label: str = "right",
    check_name: str = "output_parity",
    top_k_examples: int = 5,
) -> dict[str, Any]:
    if len(left_outputs) != len(right_outputs):
        raise ValueError(
            f"Output sequence length mismatch: {len(left_outputs)} vs {len(right_outputs)}"
        )
    metric = delta_fn or default_delta
    deltas = [
        float(metric(left, right))
        for left, right in zip(left_outputs, right_outputs, strict=True)
    ]
    mean_delta = float(np.mean(deltas)) if deltas else 0.0
    max_delta = float(np.max(deltas)) if deltas else 0.0
    failing_rows = [
        {
            left_label: left,
            right_label: right,
            "delta": delta,
        }
        for left, right, delta in zip(left_outputs, right_outputs, deltas, strict=True)
        if delta > float(tolerance)
    ]
    failing_rows.sort(key=lambda row: row["delta"], reverse=True)
    status = "pass" if not failing_rows else "warn"
    return build_report(
        check=check_name,
        status=status,
        summary={
            "count": len(deltas),
            "failing_count": len(failing_rows),
            "mean_delta": mean_delta,
            "max_delta": max_delta,
            "tolerance": float(tolerance),
            "left_label": left_label,
            "right_label": right_label,
        },
        metrics={
            "count": len(deltas),
            "failing_count": len(failing_rows),
            "mean_delta": mean_delta,
            "max_delta": max_delta,
            "tolerance": float(tolerance),
        },
        examples={
            "largest_delta_examples": failing_rows[:top_k_examples],
        },
    )


def check_batch_single_parity(
    prompts: Sequence[Any],
    *,
    run_single: Callable[[Any], Any],
    run_batch: Callable[[Sequence[Any]], Sequence[Any]],
    delta_fn: Callable[[Any, Any], float] | None = None,
    tolerance: float = 0.0,
    top_k_examples: int = 5,
) -> dict[str, Any]:
    batch_outputs = list(run_batch(prompts))
    single_outputs = [run_single(prompt) for prompt in prompts]
    report = compare_output_sequences(
        batch_outputs,
        single_outputs,
        delta_fn=delta_fn,
        tolerance=tolerance,
        left_label="batch_output",
        right_label="single_output",
        check_name="batch_single_parity",
        top_k_examples=top_k_examples,
    )
    report["details"]["prompt_count"] = len(prompts)
    return report


def compare_position_modes(
    mode_outputs: dict[str, Sequence[Any]],
    *,
    delta_fn: Callable[[Any, Any], float] | None = None,
    tolerance: float = 0.0,
) -> dict[str, Any]:
    if len(mode_outputs) < 2:
        raise ValueError("compare_position_modes requires at least two modes")
    metric = delta_fn or default_delta
    lengths = {mode: len(outputs) for mode, outputs in mode_outputs.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"All mode output sequences must have matching lengths, got {lengths}")

    pairwise_rows: list[dict[str, Any]] = []
    for left_mode, right_mode in combinations(sorted(mode_outputs), 2):
        deltas = [
            float(metric(left, right))
            for left, right in zip(mode_outputs[left_mode], mode_outputs[right_mode], strict=True)
        ]
        pairwise_rows.append(
            {
                "left_mode": left_mode,
                "right_mode": right_mode,
                "count": len(deltas),
                "mean_delta": float(np.mean(deltas)) if deltas else 0.0,
                "max_delta": float(np.max(deltas)) if deltas else 0.0,
                "failing_count": sum(delta > float(tolerance) for delta in deltas),
            }
        )

    sorted_rows = sorted(pairwise_rows, key=lambda row: row["mean_delta"], reverse=True)
    status = "pass" if all(row["failing_count"] == 0 for row in sorted_rows) else "warn"
    return build_report(
        check="position_mode_sensitivity",
        status=status,
        summary={
            "mode_count": len(mode_outputs),
            "pair_count": len(sorted_rows),
            "tolerance": float(tolerance),
            "worst_pair": sorted_rows[0] if sorted_rows else None,
        },
        metrics={
            "mode_count": len(mode_outputs),
            "pair_count": len(sorted_rows),
            "tolerance": float(tolerance),
        },
        details={
            "pairwise": sorted_rows,
        },
    )


__all__ = [
    "check_batch_single_parity",
    "compare_output_sequences",
    "compare_position_modes",
    "default_delta",
]
