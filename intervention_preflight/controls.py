"""Selectivity-oriented control checks for intervention workflows."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from intervention_preflight.report import build_report


def _to_vector(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64).reshape(-1)


def _cosine(left: np.ndarray, right: np.ndarray, *, eps: float = 1e-12) -> float | None:
    if left.shape != right.shape or left.size == 0:
        return None
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= eps or right_norm <= eps:
        return None
    return float(np.dot(left, right) / (left_norm * right_norm))


def orthogonalize_vector(
    target_vector: Any,
    reference_vector: Any,
    *,
    eps: float = 1e-12,
) -> tuple[np.ndarray, dict[str, float | None]]:
    target = _to_vector(target_vector)
    reference = _to_vector(reference_vector)
    if target.shape != reference.shape:
        raise ValueError(f"Vector shape mismatch: {target.shape} vs {reference.shape}")
    reference_norm_sq = float(np.dot(reference, reference))
    if reference_norm_sq <= eps:
        raise ValueError("Reference vector norm is too small for orthogonalization")

    coefficient = float(np.dot(target, reference) / reference_norm_sq)
    projection = coefficient * reference
    residual = target - projection

    target_norm = float(np.linalg.norm(target))
    projection_norm = float(np.linalg.norm(projection))
    residual_norm = float(np.linalg.norm(residual))

    metrics = {
        "projection_coefficient": coefficient,
        "target_norm": target_norm,
        "reference_norm": float(np.linalg.norm(reference)),
        "projection_norm": projection_norm,
        "residual_norm": residual_norm,
        "retained_norm_fraction": float(residual_norm / max(target_norm, eps)),
        "projection_mass_fraction": float(projection_norm / max(target_norm, eps)),
        "pre_orthogonalization_cosine": _cosine(target, reference),
        "post_orthogonalization_cosine": _cosine(residual, reference),
        "residual_to_target_cosine": _cosine(residual, target),
    }
    return residual, metrics


def summarize_off_target_effects(
    *,
    target_effect: float,
    off_target_effects: Mapping[str, float],
    max_off_target_fraction: float = 0.3,
) -> dict[str, Any]:
    if not off_target_effects:
        return {
            "target_effect": float(target_effect),
            "off_target_count": 0,
            "max_abs_off_target_effect": None,
            "max_off_target_name": None,
            "off_target_to_target_ratio": None,
            "pass": True,
        }

    target_abs = max(abs(float(target_effect)), 1e-12)
    ranked = sorted(
        (
            {
                "name": str(name),
                "effect": float(effect),
                "abs_effect": abs(float(effect)),
            }
            for name, effect in off_target_effects.items()
        ),
        key=lambda row: row["abs_effect"],
        reverse=True,
    )
    worst = ranked[0]
    ratio = float(worst["abs_effect"] / target_abs)
    return {
        "target_effect": float(target_effect),
        "off_target_count": len(ranked),
        "max_abs_off_target_effect": float(worst["abs_effect"]),
        "max_off_target_name": str(worst["name"]),
        "off_target_to_target_ratio": ratio,
        "pass": bool(ratio < float(max_off_target_fraction)),
        "ranked_off_targets": ranked,
    }


def assess_retention(
    *,
    original_effect: float,
    perturbed_effect: float,
    min_retention_fraction: float = 0.8,
    max_absolute_drop: float | None = None,
) -> dict[str, Any]:
    original = float(original_effect)
    perturbed = float(perturbed_effect)
    denominator = max(abs(original), 1e-12)
    retention_fraction = float(abs(perturbed) / denominator)
    absolute_drop = float(abs(original - perturbed))
    passes_fraction = retention_fraction >= float(min_retention_fraction)
    passes_drop = True if max_absolute_drop is None else absolute_drop <= float(max_absolute_drop)
    return {
        "original_effect": original,
        "perturbed_effect": perturbed,
        "retention_fraction": retention_fraction,
        "absolute_drop": absolute_drop,
        "min_retention_fraction": float(min_retention_fraction),
        "max_absolute_drop": None if max_absolute_drop is None else float(max_absolute_drop),
        "pass": bool(passes_fraction and passes_drop),
    }


def assess_selective_intervention(
    *,
    source_effect: float,
    residual_effect: float,
    off_target_effects: Mapping[str, float],
    min_effect_fraction: float = 0.5,
    max_off_target_fraction: float = 0.3,
) -> dict[str, Any]:
    source = float(source_effect)
    residual = float(residual_effect)
    effect_fraction = float(abs(residual) / max(abs(source), 1e-12))
    off_target_summary = summarize_off_target_effects(
        target_effect=residual,
        off_target_effects=off_target_effects,
        max_off_target_fraction=max_off_target_fraction,
    )

    if effect_fraction >= float(min_effect_fraction) and bool(off_target_summary["pass"]):
        status = "pass"
        notes = ["Residual effect remains strong and off-target effects are bounded."]
    elif abs(residual) > 0.0:
        status = "warn"
        notes = ["Residual effect remains, but selectivity is not clean enough for a strong claim."]
    else:
        status = "fail"
        notes = ["Residual effect collapses or is indistinguishable from zero."]

    return build_report(
        check="selective_intervention_assessment",
        status=status,
        summary={
            "source_effect": source,
            "residual_effect": residual,
            "effect_fraction": effect_fraction,
            "min_effect_fraction": float(min_effect_fraction),
            "off_target_pass": bool(off_target_summary["pass"]),
            "off_target_to_target_ratio": off_target_summary["off_target_to_target_ratio"],
            "max_off_target_fraction": float(max_off_target_fraction),
        },
        metrics={
            "source_effect": source,
            "residual_effect": residual,
            "effect_fraction": effect_fraction,
            "off_target_to_target_ratio": off_target_summary["off_target_to_target_ratio"],
            "off_target_count": off_target_summary["off_target_count"],
        },
        details={
            "off_target_summary": off_target_summary,
        },
        notes=notes,
    )


__all__ = [
    "assess_retention",
    "assess_selective_intervention",
    "orthogonalize_vector",
    "summarize_off_target_effects",
]
