"""Lightweight preflight checks for intervention-based mech interp workflows."""

from intervention_preflight.prompt_audit import (
    audit_prompt_collection,
    audit_prompt_sets,
    lexical_similarity,
    load_jsonl_rows,
)
from intervention_preflight.controls import (
    assess_retention,
    assess_selective_intervention,
    orthogonalize_vector,
    summarize_off_target_effects,
)
from intervention_preflight.judges import (
    extract_score_json,
    parse_score_with_fallback,
    summarize_parsed_scores,
)
from intervention_preflight.parity import (
    check_batch_single_parity,
    check_cache_parity,
    compare_output_sequences,
    compare_position_modes,
    default_delta,
)
from intervention_preflight.reconstruction import (
    audit_reconstruction,
    compare_metadata,
    compare_reconstruction_modes,
    reconstruction_metrics,
)
from intervention_preflight.report import build_report, summarize_status_counts, write_json_report
from intervention_preflight.stats import (
    concentration_summary,
    effect_size_summary,
    random_baseline_selectivity,
)

__all__ = [
    "assess_retention",
    "assess_selective_intervention",
    "audit_prompt_collection",
    "audit_prompt_sets",
    "audit_reconstruction",
    "build_report",
    "check_batch_single_parity",
    "check_cache_parity",
    "compare_metadata",
    "compare_output_sequences",
    "compare_position_modes",
    "compare_reconstruction_modes",
    "concentration_summary",
    "default_delta",
    "effect_size_summary",
    "extract_score_json",
    "lexical_similarity",
    "load_jsonl_rows",
    "orthogonalize_vector",
    "parse_score_with_fallback",
    "random_baseline_selectivity",
    "reconstruction_metrics",
    "summarize_off_target_effects",
    "summarize_parsed_scores",
    "summarize_status_counts",
    "write_json_report",
]
