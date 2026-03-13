# DESIGN

## Purpose

`intervention-preflight` exists to make intervention-based mechanistic interpretability results harder to get wrong.

The package is meant to provide small, reusable checks that researchers can run before they interpret a result from:

- activation steering
- activation patching
- feature ablation
- residual stream editing
- reconstruction-based replacement models
- related intervention-heavy workflows

It is intentionally not a full framework. It should be easy to drop into an existing codebase, easy to run locally, and easy to ignore if a user only wants one narrow check.

## Motivating Problem

The standard language in mech interp often looks like this:

- `X` is **necessary** for `Y` if ablating `X` makes `Y` disappear.
- `X` is **sufficient** for `Y` if inserting `X` makes `Y` appear where it otherwise would not.

This is often useful, but incomplete.

The missing dimension is **selectivity**.

Questions that often go under-tested:

- Does intervening on `X` affect many other behaviors besides `Y`?
- Is the apparent effect on `Y` downstream of some broader collateral change?
- Are the necessity and sufficiency evaluations even measuring the same slice of `Y`?
- Would some unrelated `X'` also make `Y` appear or disappear under the same scoring setup?
- Are we seeing a real causal relation, or a change in a broader latent that happens to move the metric?

This package is built around the idea that **necessity and sufficiency evidence without selectivity checks is often too weak**.

## Working Thesis

The package should operationalize the following view:

1. Necessity and sufficiency are not enough on their own.
2. Selectivity is not captured by generic utility metrics like perplexity or broad benchmark scores.
3. Leakage often happens because the target behavior is under-specified or because the intervention changes a broader latent.
4. The right response is not a single perfect metric, but a small set of cheap, composable preflight checks.

## Design Goals

The package should be:

- **Small**: useful with a handful of functions and light dependencies.
- **Composable**: each check should run independently.
- **Extensible**: new checks and adapters should slot in without changing the core API shape.
- **Opinionated only to an extent**: provide good defaults, but allow threshold overrides.
- **Backend-light**: no required dependency on W&B, Modal, external APIs, or any specific lab stack.
- **Portable**: usable in notebooks, scripts, CI, or larger pipelines.
- **JSON-friendly**: outputs should serialize cleanly into reports.
- **Readable**: function names and report fields should be obvious without reading internal code.

## Non-Goals

The package should not:

- become a new mech interp framework
- orchestrate runs across clusters
- own experiment state management
- require a database or web UI
- define a universal theory of selectivity
- hard-code persona or safety-specific constructs
- depend on frontier-lab internal tooling patterns

## Core Package Philosophy

The package should follow a **functional core + thin adapters** architecture.

### Functional core

The core should operate on simple Python values:

- arrays
- token ids
- prompt rows
- scores
- report dicts
- user-provided callables

Core functions should not assume:

- a specific model class
- a specific prompting library
- a specific tracing backend

### Thin adapters

Adapters should exist only where they meaningfully reduce friction.

Early adapter targets:

- TransformerLens
- NNsight

Adapters should:

- normalize backend-specific execution into a shared callable interface
- avoid leaking backend-specific types into the core modules
- be optional dependencies

## Initial Module Plan

The initial package should stay flat and easy to scan.

```text
intervention_preflight/
  __init__.py
  prompt_audit.py
  parity.py
  controls.py
  reconstruction.py
  stats.py
  judges.py
  report.py
  adapters/
    __init__.py
    base.py
    transformerlens.py
    nnsight.py
```

### `prompt_audit.py`

Purpose:

- detect duplicate prompts
- detect train/held-out collisions
- estimate lexical overlap
- surface likely paraphrase leakage

Example responsibilities:

- exact duplicate count
- held-out vs train maximum similarity
- top suspicious overlaps
- collision report

### `parity.py`

Purpose:

- check whether basic inference behavior changes across execution modes

Initial checks:

- batch vs single prompt parity
- cache on/off parity
- prompt formatting parity
- token-position mode comparison

This module is one of the highest-value parts of the package because many intervention claims are invalidated by backend inconsistencies before the science even starts.

### `controls.py`

Purpose:

- run cheap selectivity-oriented checks around interventions

Initial checks:

- random baseline selectivity
- orthogonalized residual checks
- cross-target bleed summaries
- prompt-sensitivity retention

This module is where the package moves beyond pure runtime correctness and into interpretation hygiene.

### `reconstruction.py`

Purpose:

- check whether reconstruction-based analysis is on reliable footing

Initial checks:

- hook compatibility
- encode/decode reconstruction sanity
- token aggregation sensitivity
- source/release metadata mismatch warnings

### `stats.py`

Purpose:

- provide small, dependency-light metrics utilities that are commonly needed across checks

Initial contents:

- effect sizes
- bootstrap confidence intervals
- concentration summaries
- random baseline percentile and one-sided p-value summaries

### `judges.py`

Purpose:

- provide provider-agnostic helpers for structured judge outputs

Initial contents:

- strict score parsing
- fallback detection
- retry helpers
- parse-failure rate summaries

This module should remain narrow and avoid becoming a general LLM wrapper.

### `report.py`

Purpose:

- define common report shapes
- make JSON and markdown export easy

The goal is not heavy schema enforcement. It is enough structure that multiple checks can compose into a consistent report.

## MVP Scope

The first usable release should be deliberately narrow.

### v0.1 modules

- `prompt_audit.py`
- `parity.py`
- `reconstruction.py`
- `stats.py`
- `report.py`

### defer to v0.2 unless clearly cheap

- richer `controls.py`
- richer `judges.py`
- more than one backend adapter

## Core Concepts

The package should avoid over-abstracting, but a few concepts should be stable.

### Check

A check is a single callable that:

- accepts explicit inputs
- returns a JSON-serializable report
- does not rely on global state

### Report

A report should:

- describe what was checked
- contain raw summary numbers
- contain pass/warn/fail style interpretation where appropriate
- be exportable as JSON without custom encoders

### Adapter

An adapter should expose a minimal callable surface required by a check, such as:

- generate a response
- run prompts in batch or singly
- capture activations at a position
- run with cache on/off

## API Style

Public APIs should prefer explicit functions over builders or framework-style objects.

Good:

```python
from intervention_preflight import audit_prompt_sets
from intervention_preflight import check_batch_single_parity
```

Avoid:

```python
suite = InterventionPreflightManager(...)
suite.register_backend(...)
suite.run_all(...)
```

If composition is needed, use thin dataclasses or helper functions, not a central manager object.

## Output Philosophy

Every major check should return both:

- raw numbers
- a minimal interpretation layer

Example:

```json
{
  "check": "batch_single_parity",
  "status": "warn",
  "mean_output_delta": 0.14,
  "max_output_delta": 0.41,
  "threshold": 0.10,
  "notes": ["Batch mode diverged materially on 2/20 prompts."]
}
```

The package should not hide raw numbers behind only pass/fail summaries.

## Extensibility Rules

These rules should keep the package from becoming bloated.

1. New checks should only be added if they generalize across at least two research settings.
2. New adapters should only be added if they reduce meaningful friction and do not distort the core API.
3. Any new module should justify itself over simply extending an existing one.
4. Public functions should remain stable once released unless clearly experimental.
5. Configuration should stay shallow and local to each function wherever possible.

## Packaging Strategy

The project should:

- ship as a normal Python package
- keep optional dependencies behind extras
- support notebook and script usage first
- avoid CLI-first design

A small CLI can exist, but the Python API should be the main interface.

## Testing Strategy

Testing should be layered.

### Tier 1: pure unit tests

- no model calls
- no external APIs
- fast

### Tier 2: backend smoke tests

- tiny models only
- parity and reconstruction checks on small examples
- tolerant to upstream numeric variation

### Tier 3: optional slow tests

- larger backends
- adapter integration
- CI-optional

Important test rule:

- avoid exact activation-value assertions when a structural invariant is enough

## Relationship To Existing Work

This package is complementary to:

- model tracing libraries
- SAE training libraries
- circuit tracing libraries
- interpretability platforms

It should plug into those tools, not compete with them.

The intended role is similar to:

- a preflight suite
- a consistency harness
- a reliability layer before strong causal or mechanistic claims

## Initial Upstream Strategy

The standalone package should come first.

Then the most promising upstream slices are:

1. **Neuronpedia**
   - steering parity tests
   - cache on/off invariance checks
   - less brittle inference regression tests

2. **SAELens**
   - reconstruction and hook compatibility preflight
   - source/release mismatch surfacing

3. **circuit-tracer**
   - batch/single parity harness for `ReplacementModel`

4. **Sparsify**
   - train/eval metric parity harness

## First Milestone

The first milestone is not a release. It is a clean internal scaffold with:

- package skeleton
- this design doc
- one small implementation target per v0.1 module
- a tiny test suite proving the shape is sane

## Immediate Next Steps

1. implement `stats.py`
2. implement `prompt_audit.py`
3. implement `report.py`
4. add the first unit tests
5. only then add `parity.py`

## Open Questions

These should stay open until the first implementation pass.

1. Should token-position comparisons live in `parity.py` or `controls.py`?
2. Should `judges.py` exist in v0.1, or should parsing helpers live in `report.py` temporarily?
3. How much adapter surface is worth standardizing before the first external integration?
4. Should markdown export live in core or remain an example helper?

## Summary

`intervention-preflight` should be a small, high-trust toolkit for answering:

> "Before I interpret this intervention result, what are the cheapest ways to check that it is not misleading me?"

That is enough scope for a real package, and narrow enough to stay useful.
