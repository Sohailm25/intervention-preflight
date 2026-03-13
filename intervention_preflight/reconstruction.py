"""Reconstruction sanity checks for intervention and feature analysis workflows."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from intervention_preflight.report import build_report


def _flatten_array(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    return arr.reshape(-1)


def cosine_similarity(original: Any, reconstructed: Any, *, eps: float = 1e-8) -> float | None:
    left = _flatten_array(original)
    right = _flatten_array(reconstructed)
    if left.shape != right.shape or left.size == 0:
        return None
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= eps or right_norm <= eps:
        return None
    return float(np.dot(left, right) / (left_norm * right_norm))


def explained_variance(original: Any, reconstructed: Any, *, eps: float = 1e-8) -> float | None:
    left = _flatten_array(original)
    right = _flatten_array(reconstructed)
    if left.shape != right.shape or left.size == 0:
        return None
    residual = left - right
    denominator = float(np.var(left))
    if denominator <= eps:
        return None
    return float(1.0 - (np.var(residual) / denominator))


def reconstruction_metrics(original: Any, reconstructed: Any) -> dict[str, float | None]:
    left = _flatten_array(original)
    right = _flatten_array(reconstructed)
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch: {left.shape} vs {right.shape}")
    if left.size == 0:
        raise ValueError("Reconstruction inputs must be non-empty")
    residual = left - right
    return {
        "cosine_similarity": cosine_similarity(left, right),
        "explained_variance": explained_variance(left, right),
        "mean_squared_error": float(np.mean(np.square(residual))),
        "mean_absolute_error": float(np.mean(np.abs(residual))),
        "max_absolute_error": float(np.max(np.abs(residual))),
        "n_features": int(left.size),
    }


def compare_metadata(
    *,
    expected: Mapping[str, Any] | None,
    observed: Mapping[str, Any] | None,
    keys: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    expected_map = dict(expected or {})
    observed_map = dict(observed or {})
    tracked_keys = list(keys) if keys is not None else sorted(set(expected_map) | set(observed_map))
    mismatches: list[dict[str, Any]] = []
    missing_from_observed: list[str] = []
    missing_from_expected: list[str] = []

    for key in tracked_keys:
        in_expected = key in expected_map
        in_observed = key in observed_map
        if in_expected and not in_observed:
            missing_from_observed.append(key)
            continue
        if in_observed and not in_expected:
            missing_from_expected.append(key)
            continue
        if expected_map[key] != observed_map[key]:
            mismatches.append(
                {
                    "key": key,
                    "expected": expected_map[key],
                    "observed": observed_map[key],
                }
            )

    return {
        "tracked_key_count": len(tracked_keys),
        "mismatch_count": len(mismatches),
        "missing_from_observed": missing_from_observed,
        "missing_from_expected": missing_from_expected,
        "mismatches": mismatches,
    }


def _metric_status(value: float | None, *, pass_min: float, warn_min: float) -> str:
    if value is None:
        return "warn"
    if value >= pass_min:
        return "pass"
    if value >= warn_min:
        return "warn"
    return "fail"


def audit_reconstruction(
    *,
    original: Any,
    reconstructed: Any,
    name: str = "reconstruction",
    cosine_pass_min: float = 0.9,
    cosine_warn_min: float = 0.8,
    explained_variance_pass_min: float = 0.9,
    explained_variance_warn_min: float = 0.8,
    expected_metadata: Mapping[str, Any] | None = None,
    observed_metadata: Mapping[str, Any] | None = None,
    metadata_keys: list[str] | tuple[str, ...] | None = None,
    metadata_mismatch_as_fail: bool = False,
) -> dict[str, Any]:
    metrics = reconstruction_metrics(original, reconstructed)
    metadata_report = compare_metadata(
        expected=expected_metadata,
        observed=observed_metadata,
        keys=metadata_keys,
    )
    cosine_status = _metric_status(
        metrics["cosine_similarity"],
        pass_min=cosine_pass_min,
        warn_min=cosine_warn_min,
    )
    ev_status = _metric_status(
        metrics["explained_variance"],
        pass_min=explained_variance_pass_min,
        warn_min=explained_variance_warn_min,
    )

    notes: list[str] = []
    if cosine_status != "pass":
        notes.append(f"Cosine reconstruction quality is {cosine_status}.")
    if ev_status != "pass":
        notes.append(f"Explained variance is {ev_status}.")
    if metadata_report["mismatch_count"] or metadata_report["missing_from_observed"] or metadata_report["missing_from_expected"]:
        notes.append("Metadata mismatch detected between expected and observed reconstruction context.")

    overall_status = "pass"
    if "fail" in {cosine_status, ev_status}:
        overall_status = "fail"
    elif "warn" in {cosine_status, ev_status}:
        overall_status = "warn"

    has_metadata_problem = bool(
        metadata_report["mismatch_count"]
        or metadata_report["missing_from_observed"]
        or metadata_report["missing_from_expected"]
    )
    if has_metadata_problem:
        if metadata_mismatch_as_fail:
            overall_status = "fail"
        elif overall_status == "pass":
            overall_status = "warn"

    return build_report(
        check="reconstruction_audit",
        status=overall_status,
        summary={
            "name": name,
            "cosine_status": cosine_status,
            "explained_variance_status": ev_status,
            "metadata_ok": not has_metadata_problem,
        },
        metrics=metrics,
        details={
            "thresholds": {
                "cosine_pass_min": float(cosine_pass_min),
                "cosine_warn_min": float(cosine_warn_min),
                "explained_variance_pass_min": float(explained_variance_pass_min),
                "explained_variance_warn_min": float(explained_variance_warn_min),
            },
            "metadata": metadata_report,
        },
        notes=notes,
    )


def compare_reconstruction_modes(
    mode_pairs: Mapping[str, tuple[Any, Any]],
    *,
    cosine_pass_min: float = 0.9,
    cosine_warn_min: float = 0.8,
    sensitivity_tolerance: float = 0.1,
) -> dict[str, Any]:
    if len(mode_pairs) < 2:
        raise ValueError("compare_reconstruction_modes requires at least two modes")

    per_mode: list[dict[str, Any]] = []
    cosine_values: list[float] = []
    for mode_name, (original, reconstructed) in mode_pairs.items():
        metrics = reconstruction_metrics(original, reconstructed)
        cosine = metrics["cosine_similarity"]
        if cosine is not None:
            cosine_values.append(float(cosine))
        per_mode.append(
            {
                "mode": mode_name,
                "metrics": metrics,
                "status": _metric_status(
                    cosine,
                    pass_min=cosine_pass_min,
                    warn_min=cosine_warn_min,
                ),
            }
        )

    per_mode.sort(
        key=lambda row: (
            row["metrics"]["cosine_similarity"] is None,
            -(row["metrics"]["cosine_similarity"] or -1.0),
        )
    )

    cosine_sensitivity = None
    if cosine_values:
        cosine_sensitivity = float(max(cosine_values) - min(cosine_values))

    overall_status = "pass"
    statuses = {row["status"] for row in per_mode}
    if statuses == {"fail"}:
        overall_status = "fail"
    elif "fail" in statuses or "warn" in statuses:
        overall_status = "warn"

    if cosine_sensitivity is not None and cosine_sensitivity > float(sensitivity_tolerance):
        if overall_status == "pass":
            overall_status = "warn"

    return build_report(
        check="reconstruction_mode_comparison",
        status=overall_status,
        summary={
            "mode_count": len(per_mode),
            "sensitivity_tolerance": float(sensitivity_tolerance),
            "cosine_sensitivity": cosine_sensitivity,
            "best_mode": per_mode[0]["mode"] if per_mode else None,
            "worst_mode": per_mode[-1]["mode"] if per_mode else None,
        },
        details={
            "per_mode": per_mode,
        },
        notes=(
            ["Reconstruction quality varies materially across modes."]
            if cosine_sensitivity is not None and cosine_sensitivity > float(sensitivity_tolerance)
            else []
        ),
    )


__all__ = [
    "audit_reconstruction",
    "compare_metadata",
    "compare_reconstruction_modes",
    "cosine_similarity",
    "explained_variance",
    "reconstruction_metrics",
]
