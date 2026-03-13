"""Small CLI for common intervention-preflight checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from intervention_preflight.prompt_audit import (
    audit_prompt_collection,
    audit_prompt_sets,
    load_jsonl_rows,
)
from intervention_preflight.report import write_json_report


def _print_report(report: dict[str, Any]) -> None:
    print(json.dumps(report, indent=2))


def _load_optional_texts(paths: list[str]) -> list[str]:
    texts: list[str] = []
    for path in paths:
        for row in load_jsonl_rows(path):
            for key in ("text", "prompt", "user_query", "query", "instruction"):
                value = row.get(key)
                if isinstance(value, str) and value.strip():
                    texts.append(value.strip())
                    break
    return texts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="intervention-preflight CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collection = subparsers.add_parser("collection-audit", help="Audit one prompt collection")
    collection.add_argument("--input", required=True, help="JSONL input path")
    collection.add_argument("--output", default="", help="Optional JSON output path")

    prompt_sets = subparsers.add_parser("prompt-audit", help="Audit primary/heldout prompt sets")
    prompt_sets.add_argument("--primary", required=True, help="Primary JSONL path")
    prompt_sets.add_argument("--heldout", required=True, help="Held-out JSONL path")
    prompt_sets.add_argument(
        "--blocked",
        action="append",
        default=[],
        help="Optional JSONL path containing blocked/collision texts; may be repeated",
    )
    prompt_sets.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.8,
        help="Maximum allowed heldout-vs-primary lexical similarity",
    )
    prompt_sets.add_argument("--output", default="", help="Optional JSON output path")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "collection-audit":
        report = audit_prompt_collection(load_jsonl_rows(args.input), name=Path(args.input).name)
    elif args.command == "prompt-audit":
        report = audit_prompt_sets(
            primary_rows=load_jsonl_rows(args.primary),
            heldout_rows=load_jsonl_rows(args.heldout),
            blocked_texts=_load_optional_texts(args.blocked),
            similarity_threshold=float(args.similarity_threshold),
        )
    else:  # pragma: no cover
        parser.error(f"Unknown command: {args.command}")
        return 2

    if args.output:
        write_json_report(report, args.output)
    else:
        _print_report(report)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
