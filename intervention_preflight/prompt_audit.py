"""Prompt-set audits for duplicates, collisions, and overlap leakage."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from intervention_preflight.report import build_report


DEFAULT_TEXT_KEYS = ("text", "prompt", "user_query", "query", "instruction")


def load_jsonl_rows(path: str | Path) -> list[dict[str, Any]]:
    resolved = Path(path)
    rows: list[dict[str, Any]] = []
    for line in resolved.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def lexical_similarity(left: str, right: str) -> float:
    return float(SequenceMatcher(a=normalize_text(left), b=normalize_text(right)).ratio())


def _extract_text(
    row: str | Mapping[str, Any],
    *,
    text_getter: Callable[[str | Mapping[str, Any]], str] | None,
) -> str:
    if text_getter is not None:
        value = text_getter(row)
        if not isinstance(value, str) or not value.strip():
            raise ValueError("text_getter must return a non-empty string")
        return value.strip()

    if isinstance(row, str):
        if not row.strip():
            raise ValueError("Prompt text cannot be empty")
        return row.strip()

    for key in DEFAULT_TEXT_KEYS:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise KeyError(f"Could not extract text from row with keys={sorted(row.keys())}")


def _materialize_texts(
    rows: Sequence[str | Mapping[str, Any]],
    *,
    text_getter: Callable[[str | Mapping[str, Any]], str] | None = None,
) -> list[str]:
    return [_extract_text(row, text_getter=text_getter) for row in rows]


def audit_prompt_collection(
    rows: Sequence[str | Mapping[str, Any]],
    *,
    text_getter: Callable[[str | Mapping[str, Any]], str] | None = None,
    name: str = "prompts",
) -> dict[str, Any]:
    texts = _materialize_texts(rows, text_getter=text_getter)
    normalized = [normalize_text(text) for text in texts]
    unique_texts = set(normalized)
    duplicates = []
    seen: set[str] = set()
    for text, normalized_text in zip(texts, normalized, strict=True):
        if normalized_text in seen:
            duplicates.append(text)
        else:
            seen.add(normalized_text)

    status = "pass" if not duplicates else "fail"
    return build_report(
        check="prompt_collection_audit",
        status=status,
        summary={
            "name": name,
            "count": len(texts),
            "unique_count": len(unique_texts),
            "duplicate_count": len(duplicates),
        },
        metrics={
            "count": len(texts),
            "unique_count": len(unique_texts),
            "duplicate_count": len(duplicates),
        },
        examples={
            "sample_texts": texts[:3],
            "duplicate_examples": duplicates[:5],
        },
    )


def _top_overlap_matches(
    source_texts: Sequence[str],
    candidate_texts: Sequence[str],
    *,
    top_k: int = 3,
) -> tuple[float, float, list[dict[str, Any]]]:
    if not source_texts or not candidate_texts:
        return 0.0, 0.0, []

    all_scores: list[float] = []
    best_matches: list[dict[str, Any]] = []
    for source in source_texts:
        scored = sorted(
            ((lexical_similarity(source, candidate), candidate) for candidate in candidate_texts),
            key=lambda item: item[0],
            reverse=True,
        )
        best_score, best_candidate = scored[0]
        all_scores.append(best_score)
        best_matches.append(
            {
                "source_text": source,
                "matched_text": best_candidate,
                "similarity": best_score,
            }
        )

    best_matches.sort(key=lambda row: row["similarity"], reverse=True)
    return max(all_scores), sum(all_scores) / len(all_scores), best_matches[:top_k]


def audit_prompt_sets(
    *,
    primary_rows: Sequence[str | Mapping[str, Any]],
    heldout_rows: Sequence[str | Mapping[str, Any]],
    blocked_texts: Sequence[str] | None = None,
    text_getter: Callable[[str | Mapping[str, Any]], str] | None = None,
    similarity_threshold: float = 0.8,
) -> dict[str, Any]:
    primary_texts = _materialize_texts(primary_rows, text_getter=text_getter)
    heldout_texts = _materialize_texts(heldout_rows, text_getter=text_getter)

    primary_report = audit_prompt_collection(primary_texts, name="primary")
    heldout_report = audit_prompt_collection(heldout_texts, name="heldout")

    max_similarity, mean_similarity, top_matches = _top_overlap_matches(heldout_texts, primary_texts)

    blocked_normalized = {normalize_text(text) for text in (blocked_texts or [])}
    collision_examples = [
        text
        for text in [*primary_texts, *heldout_texts]
        if normalize_text(text) in blocked_normalized
    ]

    duplicate_count = (
        int(primary_report["metrics"]["duplicate_count"]) + int(heldout_report["metrics"]["duplicate_count"])
    )
    overall_status = "pass"
    notes: list[str] = []
    if duplicate_count > 0:
        overall_status = "fail"
        notes.append("Duplicate prompts detected.")
    if collision_examples:
        overall_status = "fail"
        notes.append("Exact collisions detected against blocked texts.")
    if max_similarity >= float(similarity_threshold):
        overall_status = "fail"
        notes.append("Held-out prompts are too similar to primary prompts.")

    return build_report(
        check="prompt_set_audit",
        status=overall_status,
        summary={
            "primary_count": len(primary_texts),
            "heldout_count": len(heldout_texts),
            "duplicate_count": duplicate_count,
            "collision_count": len(collision_examples),
            "max_similarity": max_similarity,
            "mean_similarity": mean_similarity,
            "similarity_threshold": float(similarity_threshold),
        },
        metrics={
            "primary_count": len(primary_texts),
            "heldout_count": len(heldout_texts),
            "duplicate_count": duplicate_count,
            "collision_count": len(collision_examples),
            "max_similarity": max_similarity,
            "mean_similarity": mean_similarity,
        },
        examples={
            "top_overlap_matches": top_matches,
            "collision_examples": collision_examples[:5],
        },
        details={
            "primary": primary_report,
            "heldout": heldout_report,
        },
        notes=notes,
    )


__all__ = [
    "audit_prompt_collection",
    "audit_prompt_sets",
    "lexical_similarity",
    "load_jsonl_rows",
    "normalize_text",
]
