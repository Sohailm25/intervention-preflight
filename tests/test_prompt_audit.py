from __future__ import annotations

from intervention_preflight.prompt_audit import (
    audit_prompt_collection,
    audit_prompt_sets,
    lexical_similarity,
)


def test_audit_prompt_collection_detects_duplicates() -> None:
    report = audit_prompt_collection(
        [
            {"prompt": "Tell me a joke."},
            {"prompt": "Tell me a joke."},
            {"prompt": "Summarize this article."},
        ]
    )
    assert report["status"] == "fail"
    assert report["metrics"]["duplicate_count"] == 1


def test_audit_prompt_sets_flags_high_overlap() -> None:
    report = audit_prompt_sets(
        primary_rows=[{"user_query": "Is the Great Wall visible from space?"}],
        heldout_rows=[{"user_query": "I heard the Great Wall is visible from space. Is that true?"}],
        similarity_threshold=0.75,
    )
    assert report["status"] == "fail"
    assert report["metrics"]["max_similarity"] >= 0.75


def test_audit_prompt_sets_flags_blocked_collisions() -> None:
    report = audit_prompt_sets(
        primary_rows=["Tell me about Paris."],
        heldout_rows=["Give me a summary of Rome."],
        blocked_texts=["tell me about paris."],
    )
    assert report["status"] == "fail"
    assert report["metrics"]["collision_count"] == 1


def test_lexical_similarity_is_high_for_close_paraphrase() -> None:
    score = lexical_similarity(
        "Please explain whether the Great Wall is visible from space.",
        "Can you explain if the Great Wall is visible from space?",
    )
    assert score > 0.7

