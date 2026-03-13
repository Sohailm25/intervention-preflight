from __future__ import annotations

import math

from intervention_preflight.stats import concentration_summary, effect_size_summary, random_baseline_selectivity


def test_concentration_summary_returns_expected_keys() -> None:
    summary = concentration_summary([1.0, 2.0, 3.0, 4.0])
    assert set(summary.keys()) == {
        "gini",
        "entropy_normalized",
        "top_1pct_mass",
        "top_5pct_mass",
        "top_10pct_mass",
    }
    assert summary["gini"] is not None


def test_effect_size_summary_tracks_sample_sizes_and_direction() -> None:
    summary = effect_size_summary([2.0, 3.0, 4.0], [0.0, 1.0, 2.0], n_bootstrap=100)
    assert summary["n_a"] == 3
    assert summary["n_b"] == 3
    assert summary["cohens_d"] is not None
    assert summary["cohens_d"] > 0.0
    assert summary["a12"] is not None
    assert summary["a12"] > 0.5
    assert summary["cohens_d_ci95"] is not None
    assert summary["a12_ci95"] is not None


def test_random_baseline_selectivity_reports_extreme_hit() -> None:
    summary = random_baseline_selectivity(10.0, [0.1, 0.3, 0.5, 0.7])
    assert summary["n_random"] == 4
    assert math.isclose(summary["percentile_rank"], 1.0)
    assert summary["top_1pct_pass"] is True

