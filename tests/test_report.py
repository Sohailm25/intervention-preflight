from __future__ import annotations

import json

from intervention_preflight.report import (
    aggregate_reports,
    build_report,
    render_markdown_summary,
    summarize_status_counts,
    write_json_report,
)


def test_build_report_enforces_status_and_shape(tmp_path) -> None:
    report = build_report(
        check="example_check",
        status="pass",
        summary={"n": 3},
        notes=["ok"],
    )
    assert report["check"] == "example_check"
    assert report["status"] == "pass"
    path = write_json_report(report, tmp_path / "report.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["summary"]["n"] == 3


def test_summarize_status_counts_counts_known_statuses() -> None:
    counts = summarize_status_counts(
        [
            {"status": "pass"},
            {"status": "pass"},
            {"status": "warn"},
            {"status": "fail"},
            {"status": "unknown"},
        ]
    )
    assert counts["pass"] == 2
    assert counts["warn"] == 1
    assert counts["fail"] == 1


def test_aggregate_reports_uses_worst_status_and_preserves_children() -> None:
    aggregate = aggregate_reports(
        "example_suite",
        [
            build_report(check="parity", status="pass"),
            build_report(check="reconstruction", status="warn"),
        ],
    )
    assert aggregate["status"] == "warn"
    assert aggregate["summary"]["report_count"] == 2
    assert aggregate["details"]["failing_checks"] == ["reconstruction"]
    assert aggregate["metadata"]["report_type"] == "suite"


def test_render_markdown_summary_renders_core_sections() -> None:
    report = build_report(
        check="cache_parity",
        status="warn",
        summary={"prompt_count": 2},
        metrics={"max_delta": 0.4},
        notes=["cached and uncached outputs diverged"],
    )
    rendered = render_markdown_summary(report)
    assert "## cache_parity" in rendered
    assert "- Status: `warn`" in rendered
    assert "- Prompt Count: `2`" in rendered
    assert "- max_delta: `0.4`" in rendered
    assert "- cached and uncached outputs diverged" in rendered
