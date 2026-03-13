"""Thin adapter helpers for backend-specific execution."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any


RunSingle = Callable[[Any], Any]
RunBatch = Callable[[Sequence[Any]], Sequence[Any]]
RunBatchWithCache = Callable[[Sequence[Any], bool], Sequence[Any]]
CaptureActivations = Callable[[Sequence[str], str, str | int], Any]


@dataclass(frozen=True, slots=True)
class InterventionAdapter:
    """Minimal callable surface for backend integrations."""

    name: str
    run_single: RunSingle
    run_batch: RunBatch
    run_batch_with_cache: RunBatchWithCache | None = None
    capture_activations: CaptureActivations | None = None

    def supports_cache_controls(self) -> bool:
        return self.run_batch_with_cache is not None

    def supports_activation_capture(self) -> bool:
        return self.capture_activations is not None


def require_cache_controls(adapter: InterventionAdapter) -> RunBatchWithCache:
    if adapter.run_batch_with_cache is None:
        raise ValueError(f"Adapter {adapter.name!r} does not expose cache controls")
    return adapter.run_batch_with_cache


def require_activation_capture(adapter: InterventionAdapter) -> CaptureActivations:
    if adapter.capture_activations is None:
        raise ValueError(f"Adapter {adapter.name!r} does not expose activation capture")
    return adapter.capture_activations


__all__ = [
    "CaptureActivations",
    "InterventionAdapter",
    "RunBatch",
    "RunBatchWithCache",
    "RunSingle",
    "require_activation_capture",
    "require_cache_controls",
]
