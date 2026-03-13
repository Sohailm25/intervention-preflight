"""Small helpers for building and exporting JSON-friendly reports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VALID_STATUSES = {"pass", "warn", "fail", "info"}


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_report(
    *,
    check: str,
    status: str,
    summary: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    examples: dict[str, Any] | None = None,
    details: dict[str, Any] | None = None,
    notes: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_status = str(status).strip().lower()
    if normalized_status not in VALID_STATUSES:
        raise ValueError(f"Unsupported report status: {status!r}")
    return {
        "timestamp_utc": utc_timestamp(),
        "check": str(check),
        "status": normalized_status,
        "summary": summary or {},
        "metrics": metrics or {},
        "examples": examples or {},
        "details": details or {},
        "notes": list(notes or []),
        "metadata": metadata or {},
    }


def write_json_report(report: dict[str, Any], path: str | Path, *, indent: int = 2) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(report, indent=indent, sort_keys=False) + "\n", encoding="utf-8")
    return resolved


def summarize_status_counts(reports: list[dict[str, Any]]) -> dict[str, int]:
    counts = {status: 0 for status in sorted(VALID_STATUSES)}
    for report in reports:
        status = str(report.get("status", "")).strip().lower()
        if status not in VALID_STATUSES:
            continue
        counts[status] += 1
    return counts


__all__ = [
    "VALID_STATUSES",
    "build_report",
    "summarize_status_counts",
    "utc_timestamp",
    "write_json_report",
]
