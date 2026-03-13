# SAELens upstream target

## Why this slice exists

SAELens already has:

- strong pretrained loader support
- rich metadata on SAEs
- heavy evaluation utilities in `sae_lens.evals`
- active compatibility work around Bridge hook resolution and `hook_z`

It does **not** currently expose a small preflight surface for the common question:

> "Before I interpret or steer with this SAE, is it actually compatible with the model, hook site, and activations I am using?"

That gap is now more concrete because SAELens merged hook-name and Bridge compatibility work on **March 13, 2026** in commit `a82692e`, which added `SAETransformerBridge.get_sae_hook_name(...)` and fixed alias / `hook_z` handling. That is exactly the kind of change that suggests users need a lightweight preflight layer, not just deeper internals.

## Objective signal that the gap is real

- SAELens has active loader and compatibility work:
  - recent commit `a82692e`: hook name resolution and `hook_z` compatibility
  - open PR `#384`: `hf_revision` support in runner config
- SAELens docs explain how to manually call `encode()` / `decode()` on extracted activations, but they do not provide a one-call preflight check for:
  - required metadata presence
  - alias-resolved hook compatibility against the loaded model
  - encode/decode reconstruction sanity on the actual activations the user plans to analyze

## Proposed first upstream slice

Keep the first slice small:

1. Add a helper module such as `sae_lens/analysis/preflight.py`.
2. Expose three checks only:
   - metadata audit
   - model hook compatibility audit
   - reconstruction sanity audit
3. Add one docs page or notebook showing the intended workflow.

## Good first API surface

Something approximately like:

```python
check_sae_metadata(sae, expected_metadata=...)
check_sae_hook_compatibility(sae, model)
check_sae_reconstruction(sae, activations)
run_sae_preflight(sae, model=model, activations=acts, expected_metadata=...)
```

The point is not to replace `sae_lens.evals`. The point is to give users a lightweight check they can run **before** analysis, steering, or attribution work.

## Why this would be meaningful

- It catches a common class of silent mistakes earlier.
- It complements recent Bridge compatibility fixes.
- It is useful for both HookedTransformer and TransformerBridge workflows.
- It is small enough to merge without changing SAELens’ architecture.

## Relationship to intervention-preflight

`intervention-preflight` now includes a standalone version of this idea in:

- `intervention_preflight.saelens.audit_saelens_metadata`
- `intervention_preflight.saelens.audit_saelens_hook_compatibility`
- `intervention_preflight.saelens.audit_saelens_reconstruction`
- `intervention_preflight.saelens.audit_saelens_preflight`

That standalone implementation should be treated as a proving ground, not something SAELens must depend on directly.
