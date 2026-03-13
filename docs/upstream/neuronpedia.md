# Neuronpedia upstream slice

This note maps `intervention-preflight` checks onto the current Neuronpedia inference test surface.

## Why this slice exists

Two open issues in Neuronpedia point at the same underlying problem: useful integration tests are present, but some of them currently encode invariants too rigidly.

- Issue `#163`: KV cache changes steering outputs in `/steer/completion-chat`
- Issue `#192`: integration tests fail because they hardcode exact activation indices and values

As of March 13, 2026, those issues are reflected in the current inference test tree under:

- `apps/inference/tests/integration/test_activation_all.py`
- `apps/inference/tests/integration/test_activation_single.py`
- `apps/inference/tests/integration/test_completion.py`
- `apps/inference/tests/integration/test_completion_chat.py`

## What to upstream first

### 1. Cache parity regression test

Use:

- `check_cache_parity`

Target:

- steering endpoints where the only intended change is runtime performance, not model behavior

The test should compare:

- same prompt set
- same seed
- cache on vs cache off

The assertion should be on:

- relative output drift
- failure count
- prompt-level examples with the largest divergence

It should not require a single exact completion string unless the endpoint is explicitly intended to snapshot one exact backend version.

### 2. Batch vs single parity regression test

Use:

- `check_batch_single_parity`

Target:

- inference endpoints where batching should be semantically transparent

This catches:

- padding-position mistakes
- batch-only hook application bugs
- accidental prompt coupling

### 3. Activation structure test instead of exact-value test

Use:

- `compare_activation_arrays`
- `summarize_activation_array`

Target:

- `test_activation_all.py`
- `test_activation_single.py`

Replace exact assertions like:

- exact SAE feature index
- exact floating-point activation values
- exact max-value position

with structural assertions like:

- response parses into the expected schema
- tensor or feature array shape is stable
- activations are non-trivial, not all zero
- mean cosine similarity across snapshots is above threshold
- top-k feature overlap is above threshold

This preserves real regression detection while tolerating upstream model and library drift.

## Suggested rewrite pattern

### Existing brittle pattern

```python
assert actual.index == 16653
assert pytest.approx(actual.values, abs=ABS_TOLERANCE) == expected_values
```

### Better invariant

```python
from intervention_preflight import compare_activation_arrays

report = compare_activation_arrays(
    actual_activation_rows,
    reference_activation_rows,
    top_k=10,
    min_mean_cosine=0.90,
    min_mean_topk_overlap=0.50,
)

assert report["status"] in {"pass", "warn"}
assert report["summary"]["shape_match"] is True
```

In repo-native Neuronpedia code, this does not need to import `intervention-preflight` directly. The same invariants can be expressed locally first, then optionally shared later.

## Recommended PR order

1. Add a new cache parity integration test for steering.
2. Replace one brittle activation-value test with structural checks.
3. Only after that, revisit exact completion-string assertions in `test_completion.py`.

That order is safer because:

- cache parity is a clear correctness bug
- activation structural checks are a clean improvement in test philosophy
- exact steered text expectations are more politically loaded because they trade off reproducibility against dependency drift

## What not to upstream first

Do not start with:

- a broad refactor of the entire inference test suite
- introducing a heavy new dependency into Neuronpedia test code
- changing all snapshot-style completion tests at once

The first PR should be narrow and obviously defensible.
