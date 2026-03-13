"""Activation-structure checks for backend outputs and cached features."""

from __future__ import annotations

from typing import Any

import numpy as np

from intervention_preflight.report import build_report


def _to_array(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        raise ValueError("Activation checks require at least 1D data")
    return array


def _as_rows(values: np.ndarray) -> np.ndarray:
    if values.ndim == 1:
        return values.reshape(1, -1)
    if values.ndim == 2:
        return values
    raise ValueError(f"Expected a 1D or 2D activation array, got shape {values.shape}")


def _row_cosine(left: np.ndarray, right: np.ndarray, *, eps: float = 1e-12) -> float | None:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= eps or right_norm <= eps:
        return None
    return float(np.dot(left, right) / (left_norm * right_norm))


def topk_index_overlap(
    left: Any,
    right: Any,
    *,
    k: int = 10,
) -> dict[str, Any]:
    left_arr = _to_array(left).reshape(-1)
    right_arr = _to_array(right).reshape(-1)
    if left_arr.shape != right_arr.shape:
        raise ValueError(f"Activation shape mismatch: {left_arr.shape} vs {right_arr.shape}")
    if k <= 0:
        raise ValueError("k must be positive")
    feature_count = left_arr.size
    top_k = min(int(k), feature_count)
    left_indices = np.argsort(np.abs(left_arr))[-top_k:][::-1]
    right_indices = np.argsort(np.abs(right_arr))[-top_k:][::-1]
    left_set = {int(index) for index in left_indices.tolist()}
    right_set = {int(index) for index in right_indices.tolist()}
    overlap = left_set & right_set
    union = left_set | right_set
    return {
        "k": top_k,
        "overlap_count": len(overlap),
        "overlap_fraction": float(len(overlap) / max(top_k, 1)),
        "jaccard": float(len(overlap) / max(len(union), 1)),
        "left_top_indices": [int(index) for index in left_indices.tolist()],
        "right_top_indices": [int(index) for index in right_indices.tolist()],
        "overlap_indices": sorted(int(index) for index in overlap),
    }


def summarize_activation_array(
    values: Any,
    *,
    top_k: int = 10,
    zero_tolerance: float = 1e-12,
) -> dict[str, Any]:
    array = _to_array(values)
    rows = _as_rows(array)
    flat = rows.reshape(-1)
    abs_flat = np.abs(flat)
    top_k = min(int(top_k), flat.size)
    top_indices = np.argsort(abs_flat)[-top_k:][::-1] if top_k > 0 else np.array([], dtype=np.int64)
    return {
        "shape": list(array.shape),
        "ndim": int(array.ndim),
        "row_count": int(rows.shape[0]),
        "feature_count": int(rows.shape[1]),
        "nonzero_fraction": float(np.mean(abs_flat > float(zero_tolerance))),
        "mean_abs_activation": float(np.mean(abs_flat)) if abs_flat.size else 0.0,
        "max_abs_activation": float(np.max(abs_flat)) if abs_flat.size else 0.0,
        "top_flat_indices": [int(index) for index in top_indices.tolist()],
        "top_flat_values": [float(flat[index]) for index in top_indices.tolist()],
    }


def compare_activation_arrays(
    left: Any,
    right: Any,
    *,
    top_k: int = 10,
    min_mean_cosine: float = 0.95,
    min_mean_topk_overlap: float = 0.5,
) -> dict[str, Any]:
    left_array = _to_array(left)
    right_array = _to_array(right)
    if left_array.shape != right_array.shape:
        return build_report(
            check="activation_array_comparison",
            status="fail",
            summary={
                "shape_match": False,
                "left_shape": list(left_array.shape),
                "right_shape": list(right_array.shape),
            },
            metrics={
                "shape_match": 0,
            },
            notes=["Activation arrays differ in shape, so structural comparison is invalid."],
        )

    left_rows = _as_rows(left_array)
    right_rows = _as_rows(right_array)
    rowwise: list[dict[str, Any]] = []
    for row_index, (left_row, right_row) in enumerate(zip(left_rows, right_rows, strict=True)):
        overlap = topk_index_overlap(left_row, right_row, k=top_k)
        cosine = _row_cosine(left_row, right_row)
        rowwise.append(
            {
                "row_index": row_index,
                "cosine_similarity": cosine,
                "topk_overlap_fraction": overlap["overlap_fraction"],
                "topk_jaccard": overlap["jaccard"],
                "overlap_indices": overlap["overlap_indices"],
            }
        )

    cosines = [row["cosine_similarity"] for row in rowwise if row["cosine_similarity"] is not None]
    mean_cosine = float(np.mean(cosines)) if cosines else None
    min_cosine = float(np.min(cosines)) if cosines else None
    overlap_values = [float(row["topk_overlap_fraction"]) for row in rowwise]
    mean_overlap = float(np.mean(overlap_values)) if overlap_values else 0.0
    min_overlap = float(np.min(overlap_values)) if overlap_values else 0.0

    if mean_cosine is None:
        status = "warn"
        notes = ["At least one activation row had near-zero norm, so cosine similarity is undefined."]
    elif mean_cosine >= float(min_mean_cosine) and mean_overlap >= float(min_mean_topk_overlap):
        status = "pass"
        notes = ["Activation structure is stable under relative similarity and top-k overlap checks."]
    elif mean_overlap > 0.0 or mean_cosine > 0.0:
        status = "warn"
        notes = ["Activation structure partially shifted; relative invariants do not fully agree."]
    else:
        status = "fail"
        notes = ["Activation structure diverged strongly across rows."]

    return build_report(
        check="activation_array_comparison",
        status=status,
        summary={
            "shape_match": True,
            "row_count": int(left_rows.shape[0]),
            "feature_count": int(left_rows.shape[1]),
            "mean_cosine_similarity": mean_cosine,
            "min_cosine_similarity": min_cosine,
            "mean_topk_overlap": mean_overlap,
            "min_topk_overlap": min_overlap,
            "min_mean_cosine": float(min_mean_cosine),
            "min_mean_topk_overlap": float(min_mean_topk_overlap),
        },
        metrics={
            "row_count": int(left_rows.shape[0]),
            "feature_count": int(left_rows.shape[1]),
            "mean_cosine_similarity": mean_cosine,
            "min_cosine_similarity": min_cosine,
            "mean_topk_overlap": mean_overlap,
            "min_topk_overlap": min_overlap,
        },
        details={
            "left_summary": summarize_activation_array(left_array, top_k=top_k),
            "right_summary": summarize_activation_array(right_array, top_k=top_k),
            "rowwise": rowwise,
        },
        notes=notes,
    )


__all__ = [
    "compare_activation_arrays",
    "summarize_activation_array",
    "topk_index_overlap",
]
