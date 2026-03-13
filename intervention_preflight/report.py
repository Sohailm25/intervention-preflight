"""Small helpers for building and exporting JSON-friendly reports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VALID_STATUSES = {"pass", "warn", "fail", "info"}
STATUS_PRIORITY = {"fail": 3, "warn": 2, "pass": 1, "info": 0}


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


def aggregate_reports(
    suite: str,
    reports: list[dict[str, Any]],
    *,
    notes: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    counts = summarize_status_counts(reports)
    aggregate_status = "info"
    if any(counts.values()):
        aggregate_status = max(
            VALID_STATUSES,
            key=lambda status: (counts[status] > 0, STATUS_PRIORITY[status]),
        )
    failing_checks = [
        report.get("check")
        for report in reports
        if str(report.get("status", "")).strip().lower() in {"warn", "fail"}
    ]
    return build_report(
        check=str(suite),
        status=aggregate_status,
        summary={
            "suite": str(suite),
            "report_count": len(reports),
            "status_counts": counts,
            "failing_check_count": len(failing_checks),
        },
        metrics={
            "report_count": len(reports),
            "failing_check_count": len(failing_checks),
        },
        details={
            "reports": list(reports),
            "failing_checks": failing_checks,
        },
        notes=notes,
        metadata={"report_type": "suite", **(metadata or {})},
    )


def summarize_status_counts(reports: list[dict[str, Any]]) -> dict[str, int]:
    counts = {status: 0 for status in sorted(VALID_STATUSES)}
    for report in reports:
        status = str(report.get("status", "")).strip().lower()
        if status not in VALID_STATUSES:
            continue
        counts[status] += 1
    return counts


def render_markdown_summary(report: dict[str, Any], *, max_notes: int = 3) -> str:
    check = str(report.get("check", "unknown_check"))
    status = str(report.get("status", "info")).strip().lower()
    summary = report.get("summary", {}) or {}
    metrics = report.get("metrics", {}) or {}
    notes = list(report.get("notes", []) or [])
    lines = [f"## {check}", "", f"- Status: `{status}`"]

    for key, value in summary.items():
        lines.append(f"- {key.replace('_', ' ').title()}: `{value}`")

    if metrics:
        lines.append("")
        lines.append("### Metrics")
        for key, value in metrics.items():
            lines.append(f"- {key}: `{value}`")

    if notes:
        lines.append("")
        lines.append("### Notes")
        for note in notes[:max_notes]:
            lines.append(f"- {note}")
        remaining = len(notes) - max_notes
        if remaining > 0:
            lines.append(f"- ... and {remaining} more")

    return "\n".join(lines)


__all__ = [
    "STATUS_PRIORITY",
    "VALID_STATUSES",
    "aggregate_reports",
    "build_report",
    "render_markdown_summary",
    "summarize_status_counts",
    "utc_timestamp",
    "write_json_report",
]
