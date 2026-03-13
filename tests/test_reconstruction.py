from __future__ import annotations

from intervention_preflight.reconstruction import (
    audit_reconstruction,
    compare_metadata,
    compare_reconstruction_modes,
    reconstruction_metrics,
)


def test_reconstruction_metrics_are_strong_for_close_match() -> None:
    metrics = reconstruction_metrics([1.0, 2.0, 3.0], [1.0, 2.1, 2.9])
    assert metrics["cosine_similarity"] is not None
    assert metrics["cosine_similarity"] > 0.99
    assert metrics["explained_variance"] is not None
    assert metrics["explained_variance"] > 0.9


def test_audit_reconstruction_fails_for_bad_reconstruction() -> None:
    report = audit_reconstruction(
        original=[1.0, 2.0, 3.0],
        reconstructed=[3.0, 2.0, 1.0],
    )
    assert report["status"] == "fail"
    assert report["summary"]["cosine_status"] == "fail"


def test_compare_metadata_surfaces_missing_and_mismatched_keys() -> None:
    report = compare_metadata(
        expected={"layer": 12, "hook": "resid_post", "source": "gemmascope"},
        observed={"layer": 13, "hook": "resid_post"},
    )
    assert report["mismatch_count"] == 1
    assert report["missing_from_observed"] == ["source"]


def test_audit_reconstruction_warns_on_metadata_mismatch() -> None:
    report = audit_reconstruction(
        original=[1.0, 2.0, 3.0],
        reconstructed=[1.0, 2.0, 3.0],
        expected_metadata={"layer": 12},
        observed_metadata={"layer": 13},
    )
    assert report["status"] == "warn"
    assert report["details"]["metadata"]["mismatch_count"] == 1


def test_compare_reconstruction_modes_warns_on_large_sensitivity() -> None:
    report = compare_reconstruction_modes(
        {
            "last_token": ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]),
            "full_sequence": ([1.0, 2.0, 3.0], [0.0, 0.0, 0.0]),
        },
        sensitivity_tolerance=0.05,
    )
    assert report["status"] == "warn"
    assert report["summary"]["cosine_sensitivity"] is not None
    assert report["summary"]["best_mode"] == "last_token"
