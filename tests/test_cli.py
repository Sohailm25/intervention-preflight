from __future__ import annotations

import json

from intervention_preflight.cli import main


def test_collection_audit_cli_writes_json(tmp_path) -> None:
    input_path = tmp_path / "prompts.jsonl"
    output_path = tmp_path / "report.json"
    input_path.write_text(
        "\n".join(
            [
                json.dumps({"prompt": "Tell me about Rome."}),
                json.dumps({"prompt": "Tell me about Rome."}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "collection-audit",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ]
    )
    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["status"] == "fail"


def test_prompt_audit_cli_runs_without_output_file(tmp_path, capsys) -> None:
    primary_path = tmp_path / "primary.jsonl"
    heldout_path = tmp_path / "heldout.jsonl"
    primary_path.write_text(json.dumps({"user_query": "Solve 2 + 2."}) + "\n", encoding="utf-8")
    heldout_path.write_text(json.dumps({"user_query": "Name a large mammal."}) + "\n", encoding="utf-8")

    exit_code = main(
        [
            "prompt-audit",
            "--primary",
            str(primary_path),
            "--heldout",
            str(heldout_path),
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["status"] == "pass"
