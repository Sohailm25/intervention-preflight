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

## Current Modules

- `intervention_preflight.prompt_audit`
- `intervention_preflight.parity`
- `intervention_preflight.controls`
- `intervention_preflight.reconstruction`
- `intervention_preflight.stats`
- `intervention_preflight.report`

## Quickstart

```python
from intervention_preflight import (
    audit_prompt_sets,
    check_batch_single_parity,
    audit_reconstruction,
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

reconstruction_report = audit_reconstruction(
    original=[1.0, 2.0, 3.0],
    reconstructed=[1.0, 2.1, 2.9],
)
```

## Development Status

The current package is deliberately small and backend-free.

Near-term priorities:

1. thin backend adapters
2. richer selectivity/control checks
3. upstream slices into existing mech interp tools
