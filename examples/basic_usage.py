"""Minimal usage examples for intervention-preflight."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from intervention_preflight import (
    aggregate_reports,
    audit_prompt_sets,
    audit_reconstruction,
    check_batch_single_parity,
    check_cache_parity,
    render_markdown_summary,
)


def main() -> None:
    prompt_report = audit_prompt_sets(
        primary_rows=[{"user_query": "Tell me about Paris."}],
        heldout_rows=[{"user_query": "I heard Paris is in France. Tell me about it."}],
        similarity_threshold=0.75,
    )
    print("prompt_report:", prompt_report["status"], prompt_report["metrics"])

    def run_single(prompt: str) -> str:
        return prompt.upper()

    def run_batch(prompts: list[str]) -> list[str]:
        return [prompt.upper() for prompt in prompts]

    parity_report = check_batch_single_parity(
        ["alpha", "beta"],
        run_single=run_single,
        run_batch=run_batch,
    )
    print("parity_report:", parity_report["status"], parity_report["metrics"])

    cache_report = check_cache_parity(
        ["alpha", "beta"],
        run_with_cache=run_batch,
        run_without_cache=run_batch,
    )
    print("cache_report:", cache_report["status"], cache_report["metrics"])

    reconstruction_report = audit_reconstruction(
        original=[1.0, 2.0, 3.0],
        reconstructed=[1.0, 2.1, 2.9],
    )
    print("reconstruction_report:", reconstruction_report["status"], reconstruction_report["metrics"])

    suite_report = aggregate_reports(
        "basic_usage_suite",
        [prompt_report, parity_report, cache_report, reconstruction_report],
    )
    print("suite_report:", suite_report["status"], suite_report["summary"])
    print(render_markdown_summary(suite_report))


if __name__ == "__main__":
    main()
