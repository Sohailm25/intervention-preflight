from __future__ import annotations

from intervention_preflight.controls import (
    assess_retention,
    assess_selective_intervention,
    orthogonalize_vector,
    summarize_off_target_effects,
)


def test_orthogonalize_vector_removes_reference_component() -> None:
    residual, metrics = orthogonalize_vector([1.0, 1.0], [1.0, 0.0])
    assert abs(metrics["post_orthogonalization_cosine"] or 0.0) < 1e-8
    assert metrics["retained_norm_fraction"] is not None
    assert metrics["retained_norm_fraction"] > 0.5
    assert residual.shape == (2,)


def test_summarize_off_target_effects_flags_large_bleed() -> None:
    summary = summarize_off_target_effects(
        target_effect=10.0,
        off_target_effects={"assistant_like": 6.0, "other": 1.0},
        max_off_target_fraction=0.3,
    )
    assert summary["pass"] is False
    assert summary["max_off_target_name"] == "assistant_like"
    assert summary["off_target_to_target_ratio"] == 0.6


def test_assess_retention_tracks_fraction_and_drop() -> None:
    summary = assess_retention(
        original_effect=20.0,
        perturbed_effect=16.0,
        min_retention_fraction=0.75,
        max_absolute_drop=5.0,
    )
    assert summary["pass"] is True
    assert summary["retention_fraction"] == 0.8


def test_assess_selective_intervention_warns_when_effect_survives_but_bleed_is_high() -> None:
    report = assess_selective_intervention(
        source_effect=40.0,
        residual_effect=28.0,
        off_target_effects={"assistant_like": 20.0},
        min_effect_fraction=0.5,
        max_off_target_fraction=0.3,
    )
    assert report["status"] == "warn"
    assert report["summary"]["effect_fraction"] == 0.7
    assert report["summary"]["off_target_pass"] is False
