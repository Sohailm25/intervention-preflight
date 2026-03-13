"""Optional backend adapters for intervention-preflight."""

from intervention_preflight.adapters.base import (
    CaptureActivations,
    InterventionAdapter,
    RunBatch,
    RunBatchWithCache,
    RunSingle,
    require_activation_capture,
    require_cache_controls,
)
from intervention_preflight.adapters.transformerlens import (
    capture_transformerlens_activations,
    make_transformerlens_adapter,
    run_transformerlens_logits,
)

__all__ = [
    "CaptureActivations",
    "InterventionAdapter",
    "RunBatch",
    "RunBatchWithCache",
    "RunSingle",
    "capture_transformerlens_activations",
    "make_transformerlens_adapter",
    "require_activation_capture",
    "require_cache_controls",
    "run_transformerlens_logits",
]
