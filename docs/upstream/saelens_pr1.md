# SAELens PR 1 packet

## Goal

Add a lightweight preflight surface to SAELens so users can verify:

- the SAE metadata is usable
- the SAE hook site is compatible with the loaded model
- encode/decode reconstruction is sane on the activations they intend to analyze

This is meant to complement `sae_lens.evals`, not replace it.

## Why this should exist

- SAELens already carries rich metadata and loader logic.
- SAELens recently merged Bridge hook compatibility fixes in commit `a82692e`.
- Users still do not have a small “run this first” helper before downstream analysis.

## First PR scope

Keep this intentionally narrow.

### Add

- `sae_lens/analysis/preflight.py`

### Public helpers

- `check_sae_metadata(...)`
- `check_sae_hook_compatibility(...)`
- `check_sae_reconstruction(...)`
- `run_sae_preflight(...)`

### Tests

- metadata subset match vs mismatch
- missing required `hook_name`
- alias-resolved hook compatibility
- missing hook failure
- perfect encode/decode reconstruction on a small fake SAE

## Suggested file targets

- `sae_lens/analysis/preflight.py`
- `sae_lens/analysis/__init__.py`
- `tests/analysis/test_preflight.py`
- optional docs page or short usage section in `docs/usage.md`

## What not to do in PR 1

- do not add a new training abstraction
- do not add W&B or logging integrations
- do not add benchmark infrastructure
- do not make it depend on external packages beyond SAELens’ existing stack
- do not try to solve every model/backend combination in the first pass

## Acceptance criteria

- the helper works for normal HookedTransformer workflows
- the helper respects alias-resolved hook names when the model exposes them
- the helper is small enough to read in one sitting
- the tests demonstrate real failure modes, not just happy-path docs coverage

## Relationship to intervention-preflight

The standalone proving-ground implementation currently lives in:

- `intervention_preflight.saelens`
- `tests/test_saelens.py`

That code should be treated as reference material for shaping the SAELens-native implementation, not copied blindly.
