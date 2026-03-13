# Neuronpedia PR 1 packet

This is the first narrow upstream PR I would open against Neuronpedia from `intervention-preflight`.

## Objective

Fix two real correctness/testing gaps with the smallest possible change set:

1. add a regression test for cache-sensitive steering behavior
2. replace one brittle activation-value integration test with structural invariants

This directly targets:

- issue `#163` (`KV cache creating steering differences`)
- issue `#192` (`[inference] Pre-existing test failures due to hardcoded activation values`)

## Why this should be the first PR

It is:

- narrow
- tied to open issues
- easy to justify on correctness grounds
- low-risk relative to changing endpoint behavior

It does **not** require:

- changing production inference code
- refactoring the whole integration suite
- adding a dependency on `intervention-preflight`

## Files to touch

Primary targets in the Neuronpedia repo:

- `apps/inference/tests/integration/test_completion_chat.py`
- `apps/inference/tests/integration/test_activation_all.py`
- `apps/inference/tests/conftest.py`

New file to add:

- `apps/inference/tests/utils/assertions.py`

## Proposed PR title

`[inference] add cache parity regression test and replace brittle activation assertions`

## Proposed PR scope

### 1. Add local test helpers

Copy the helpers from:

- [snippets/neuronpedia_test_helpers.py](./snippets/neuronpedia_test_helpers.py)

into:

- `apps/inference/tests/utils/assertions.py`

These helpers use only:

- stdlib
- `numpy`
- `pytest`

### 2. Add a cache parity test for chat steering

Base it on:

- [snippets/neuronpedia_test_completion_chat_cache_parity.py](./snippets/neuronpedia_test_completion_chat_cache_parity.py)

Place it in:

- `apps/inference/tests/integration/test_completion_chat.py`

What the test should do:

- build the same deterministic request used by the existing additive feature chat test
- run once with `use_past_kv_cache=True`
- run once with `use_past_kv_cache=False`
- assert the steered and default outputs match exactly

Why exact equality is okay here:

- same request
- same fixed seed
- `temperature=0`
- cache should be a performance optimization, not a semantic change

### 3. Rewrite one brittle activation-all test

Base it on:

- [snippets/neuronpedia_test_activation_all_structure.py](./snippets/neuronpedia_test_activation_all_structure.py)

Use it to replace the exact hardcoded assertions in:

- `apps/inference/tests/integration/test_activation_all.py`

Keep these assertions:

- response status is `200`
- response parses into the expected client model
- token sequence matches the expected prompt tokenization
- activation count matches request expectations
- returned sources are from the requested SAE source set

Replace these assertions:

- exact feature indices
- exact activation values
- exact max activation positions

with:

- all activation rows are the same width
- activations are not all zero
- mean cosine similarity against a reference snapshot exceeds threshold
- top-k overlap against the reference snapshot exceeds threshold

## Suggested assertion policy

### For deterministic completion parity

Use exact equality between:

- cache on vs cache off

Do **not** require exact equality between:

- current output vs a long-lived frozen string snapshot

unless the test is explicitly intended to pin a backend version.

### For activation tests

Use:

- shape invariants
- non-triviality invariants
- relative similarity invariants

Do **not** use:

- exact float arrays
- exact feature ids that drift under upstream dependency changes

## Acceptance criteria

The PR is good enough if:

1. it adds one new failing regression test for cache parity before the fix, or documents that it reproduces the existing issue locally
2. it converts one exact-value activation test to a structural test
3. the structural test still fails if activations are shape-broken or collapse to zero
4. the touched tests pass on current main after the rewrite

## Commands to run in Neuronpedia

From `apps/inference`:

```bash
poetry run pytest -s tests/integration/test_completion_chat.py -k cache_parity
poetry run pytest -s tests/integration/test_activation_all.py
poetry run pytest -s tests/integration/test_activation_single.py
```

## Risks

### Risk 1

The cache toggle is not currently injectable from the endpoint or test fixture.

Mitigation:

- patch the underlying generate call in the test
- or add a tiny test-only parameter/fixture around the cache flag

### Risk 2

The current tests may be serving as de facto snapshots for a very specific dependency set.

Mitigation:

- preserve one narrow snapshot-style test if maintainers want it
- make the new structural test additive at first rather than replacing every exact-value assertion

## What not to include in PR 1

Do not:

- change all completion tests
- rewrite both activation endpoints at once
- introduce package-level abstractions from `intervention-preflight`
- expand into SAE benchmarking or broader steering eval

Keep PR 1 narrow enough that maintainers can merge it on correctness grounds alone.

## Suggested PR body

```markdown
This PR addresses two current inference test issues:

1. cache-sensitive steering behavior (`#163`)
2. brittle activation integration tests that hardcode exact feature ids/values (`#192`)

Changes:
- add a deterministic cache parity regression test for chat steering
- replace one exact-value activation test with structural invariants

The goal is to preserve regression coverage while reducing failures caused by harmless upstream drift in SAELens / TransformerLens outputs.
```
