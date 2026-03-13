from __future__ import annotations

from intervention_preflight.judges import (
    extract_score_json,
    parse_score_with_fallback,
    summarize_parsed_scores,
)


def test_extract_score_json_reads_plain_json() -> None:
    score, ok = extract_score_json('{"score": 73}')
    assert ok is True
    assert score == 73.0


def test_extract_score_json_reads_fenced_json() -> None:
    score, ok = extract_score_json("```json\n{\"score\": 81}\n```")
    assert ok is True
    assert score == 81.0


def test_parse_score_with_fallback_uses_default_on_failure() -> None:
    parsed = parse_score_with_fallback("not json", fallback_score=42.0)
    assert parsed["parse_ok"] is False
    assert parsed["used_fallback"] is True
    assert parsed["score"] == 42.0


def test_summarize_parsed_scores_warns_on_high_fallback_fraction() -> None:
    report = summarize_parsed_scores(
        [
            {"score": 50.0, "parse_ok": False, "used_fallback": True, "raw": "bad"},
            {"score": 50.0, "parse_ok": False, "used_fallback": True, "raw": "also bad"},
            {"score": 80.0, "parse_ok": True, "used_fallback": False, "raw": "{\"score\":80}"},
        ],
        fallback_alert_fraction=0.2,
    )
    assert report["status"] == "warn"
    assert report["summary"]["fallback_count"] == 2
