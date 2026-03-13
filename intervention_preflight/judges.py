"""Helpers for structured judge outputs and judge-output hygiene."""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np

from intervention_preflight.report import build_report


def extract_score_json(raw: str) -> tuple[float | None, bool]:
    text = str(raw).strip()
    if not text:
        return None, False

    candidates = [text]
    fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    candidates.extend(block.strip() for block in fenced if block.strip())

    brace_match = re.search(r"\{[\s\S]*\}", text)
    if brace_match:
        candidates.append(brace_match.group(0).strip())

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and set(parsed.keys()) == {"score"}:
            try:
                value = float(parsed["score"])
            except (TypeError, ValueError):
                continue
            if 0.0 <= value <= 100.0:
                return value, True
    return None, False


def parse_score_with_fallback(raw: str, *, fallback_score: float = 50.0) -> dict[str, Any]:
    score, parse_ok = extract_score_json(raw)
    if parse_ok:
        return {
            "score": float(score),
            "parse_ok": True,
            "used_fallback": False,
            "raw": raw,
        }
    return {
        "score": float(fallback_score),
        "parse_ok": False,
        "used_fallback": True,
        "raw": raw,
    }


def summarize_parsed_scores(
    parsed_rows: list[dict[str, Any]],
    *,
    fallback_alert_fraction: float = 0.2,
) -> dict[str, Any]:
    if not parsed_rows:
        raise ValueError("summarize_parsed_scores requires at least one parsed row")

    scores = np.asarray([float(row["score"]) for row in parsed_rows], dtype=np.float64)
    fallback_count = sum(bool(row.get("used_fallback")) for row in parsed_rows)
    parse_ok_count = sum(bool(row.get("parse_ok")) for row in parsed_rows)
    fallback_fraction = float(fallback_count / len(parsed_rows))
    status = "pass" if fallback_fraction < float(fallback_alert_fraction) else "warn"

    return build_report(
        check="judge_score_summary",
        status=status,
        summary={
            "count": len(parsed_rows),
            "parse_ok_count": parse_ok_count,
            "fallback_count": fallback_count,
            "fallback_fraction": fallback_fraction,
            "fallback_alert_fraction": float(fallback_alert_fraction),
        },
        metrics={
            "count": len(parsed_rows),
            "parse_ok_count": parse_ok_count,
            "fallback_count": fallback_count,
            "fallback_fraction": fallback_fraction,
            "score_mean": float(np.mean(scores)),
            "score_median": float(np.median(scores)),
            "score_min": float(np.min(scores)),
            "score_max": float(np.max(scores)),
        },
        examples={
            "fallback_examples": [row["raw"] for row in parsed_rows if row.get("used_fallback")][:3],
        },
        notes=(
            ["Fallback rate is high enough that the judge output may be unreliable."]
            if status == "warn"
            else []
        ),
    )


__all__ = [
    "extract_score_json",
    "parse_score_with_fallback",
    "summarize_parsed_scores",
]
