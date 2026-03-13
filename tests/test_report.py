from __future__ import annotations

import json

from intervention_preflight.report import build_report, summarize_status_counts, write_json_report


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
