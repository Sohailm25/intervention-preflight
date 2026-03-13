# intervention-preflight

Lightweight preflight checks for intervention-based mechanistic interpretability workflows.

This repository is intended to provide small, composable checks for things like:

- steering parity
- intervention consistency
- prompt overlap leakage
- reconstruction sanity
- selectivity-aware evaluation

The project is intentionally narrow: it is not a training framework, tracing framework, or experiment manager.

See [DESIGN.md](DESIGN.md) for the initial design.

## Install

```bash
pip install -e ".[dev]"
```

Optional backend extras:

```bash
pip install -e ".[dev,transformerlens]"
```

## CLI

```bash
ipf collection-audit --input prompts.jsonl
ipf prompt-audit --primary train.jsonl --heldout heldout.jsonl --output audit.json
```

## Current Modules

- `intervention_preflight.prompt_audit`
- `intervention_preflight.activations`
- `intervention_preflight.parity`
- `intervention_preflight.controls`
- `intervention_preflight.judges`
- `intervention_preflight.reconstruction`
- `intervention_preflight.saelens`
- `intervention_preflight.stats`
- `intervention_preflight.report`
- `intervention_preflight.adapters`

## Quickstart

```python
from intervention_preflight import (
    aggregate_reports,
    audit_prompt_sets,
    compare_activation_arrays,
    check_batch_single_parity,
    check_cache_parity,
    audit_reconstruction,
    audit_saelens_preflight,
    render_markdown_summary,
)

prompt_report = audit_prompt_sets(
    primary_rows=[{"user_query": "Tell me about Paris."}],
    heldout_rows=[{"user_query": "I heard Paris is in France. Tell me about it."}],
    similarity_threshold=0.75,
)

def run_single(prompt: str) -> str:
    return prompt.upper()

def run_batch(prompts: list[str]) -> list[str]:
    return [prompt.upper() for prompt in prompts]

parity_report = check_batch_single_parity(
    ["alpha", "beta"],
    run_single=run_single,
    run_batch=run_batch,
)

cache_report = check_cache_parity(
    ["alpha", "beta"],
    run_with_cache=run_batch,
    run_without_cache=run_batch,
)

reconstruction_report = audit_reconstruction(
    original=[1.0, 2.0, 3.0],
    reconstructed=[1.0, 2.1, 2.9],
)

activation_report = compare_activation_arrays(
    [[1.0, 0.0, 0.8], [0.2, 1.5, 0.1]],
    [[0.9, 0.0, 0.7], [0.1, 1.6, 0.2]],
)

suite_report = aggregate_reports(
    "demo_suite",
    [prompt_report, parity_report, cache_report, reconstruction_report, activation_report],
)
markdown = render_markdown_summary(suite_report)
```

## SAELens Preflight Example

```python
from intervention_preflight import audit_saelens_preflight

report = audit_saelens_preflight(
    sae,
    model=model,
    activations=layer_acts,
    expected_metadata={
        "model_name": "google/gemma-2-2b",
        "hook_name": "blocks.12.hook_resid_post",
    },
)
```

## Upstream Targeting

The first upstream-facing slice is aimed at public tool regressions where exact activation values are too brittle:

- use `check_cache_parity` for cache-on vs cache-off generation invariance
- use `check_batch_single_parity` for batch vs single prompt invariance
- use `compare_activation_arrays` when you need activation checks that validate structure without pinning exact floating-point values

See [docs/upstream/neuronpedia.md](docs/upstream/neuronpedia.md) for the first concrete external integration target.
The first PR-ready packet is in [docs/upstream/neuronpedia_pr1.md](docs/upstream/neuronpedia_pr1.md).
The second upstream-facing target is SAELens preflight and reconstruction hygiene in [docs/upstream/saelens.md](docs/upstream/saelens.md).
The SAELens PR packet is in [docs/upstream/saelens_pr1.md](docs/upstream/saelens_pr1.md).

## Adapter Example

```python
from intervention_preflight import check_batch_single_parity, check_cache_parity
from intervention_preflight.adapters import make_transformerlens_adapter, require_cache_controls

adapter = make_transformerlens_adapter(model, prepend_bos=True, output_position="last")
cache_runner = require_cache_controls(adapter)

batch_report = check_batch_single_parity(
    prompts,
    run_single=adapter.run_single,
    run_batch=adapter.run_batch,
)

cache_report = check_cache_parity(
    prompts,
    run_with_cache=lambda batch: cache_runner(batch, True),
    run_without_cache=lambda batch: cache_runner(batch, False),
)
```

## Development Status

The current package is deliberately small and backend-free.

Near-term priorities:

1. thin backend adapters
2. richer selectivity/control checks
3. upstream slices into existing mech interp tools
