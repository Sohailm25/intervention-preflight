from __future__ import annotations

from intervention_preflight.activations import (
    compare_activation_arrays,
    summarize_activation_array,
    topk_index_overlap,
)


def test_summarize_activation_array_reports_shape_and_nonzero_fraction() -> None:
    summary = summarize_activation_array([[1.0, 0.0, -2.0], [0.0, 0.0, 3.0]], top_k=2)
    assert summary["shape"] == [2, 3]
    assert summary["row_count"] == 2
    assert summary["feature_count"] == 3
    assert summary["nonzero_fraction"] == 0.5
    assert len(summary["top_flat_indices"]) == 2


def test_topk_index_overlap_reports_shared_top_features() -> None:
    overlap = topk_index_overlap([1.0, 4.0, 0.1, 3.0], [0.5, 5.0, 0.2, 2.5], k=2)
    assert overlap["overlap_count"] == 2
    assert overlap["overlap_fraction"] == 1.0
    assert overlap["jaccard"] == 1.0


def test_compare_activation_arrays_passes_on_stable_structure() -> None:
    report = compare_activation_arrays(
        [[1.0, 0.0, 0.8], [0.2, 1.5, 0.1]],
        [[0.9, 0.0, 0.7], [0.1, 1.6, 0.2]],
        top_k=2,
        min_mean_cosine=0.9,
        min_mean_topk_overlap=0.5,
    )
    assert report["status"] == "pass"
    assert report["summary"]["shape_match"] is True
    assert report["summary"]["mean_topk_overlap"] >= 0.5


def test_compare_activation_arrays_fails_on_shape_mismatch() -> None:
    report = compare_activation_arrays([[1.0, 2.0]], [[1.0, 2.0, 3.0]])
    assert report["status"] == "fail"
    assert report["summary"]["shape_match"] is False


def test_compare_activation_arrays_warns_on_partial_shift() -> None:
    report = compare_activation_arrays(
        [[1.0, 0.0, 0.8], [0.2, 1.5, 0.1]],
        [[0.0, 1.0, 0.8], [0.2, 0.1, 1.5]],
        top_k=2,
        min_mean_cosine=0.95,
        min_mean_topk_overlap=0.75,
    )
    assert report["status"] == "warn"
    assert report["summary"]["mean_topk_overlap"] > 0.0
