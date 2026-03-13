"""Thin TransformerLens adapter utilities."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from intervention_preflight.adapters.base import InterventionAdapter


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def _to_tokens(model: Any, prompts: Sequence[str], *, prepend_bos: bool | None) -> np.ndarray:
    if prepend_bos is None:
        tokens = model.to_tokens(list(prompts))
    else:
        tokens = model.to_tokens(list(prompts), prepend_bos=prepend_bos)
    return _to_numpy(tokens)


def _pad_token_id(model: Any) -> int | None:
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        return None
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        return None
    return int(pad_token_id)


def _effective_lengths(tokens: np.ndarray, *, pad_token_id: int | None) -> np.ndarray:
    if tokens.ndim != 2:
        raise ValueError(f"Expected 2D token array, got shape {tokens.shape}")
    if pad_token_id is None:
        return np.full(tokens.shape[0], tokens.shape[1], dtype=np.int64)
    mask = tokens != pad_token_id
    lengths = mask.sum(axis=1).astype(np.int64)
    return np.maximum(lengths, 1)


def _select_sequence_position(
    values: np.ndarray,
    tokens: np.ndarray,
    *,
    position: str | int,
    pad_token_id: int | None,
) -> np.ndarray:
    if values.ndim < 2:
        raise ValueError(f"Expected at least 2D array with batch and sequence axes, got {values.shape}")
    if values.shape[0] != tokens.shape[0] or values.shape[1] != tokens.shape[1]:
        raise ValueError(
            "Value and token batch dimensions must match, "
            f"got values {values.shape} vs tokens {tokens.shape}"
        )
    if position == "all":
        return values

    lengths = _effective_lengths(tokens, pad_token_id=pad_token_id)
    if position == "last":
        indices = lengths - 1
        return np.stack(
            [values[row_index, token_index] for row_index, token_index in enumerate(indices)],
            axis=0,
        )
    if position == "mean":
        rows = [
            np.mean(values[row_index, : int(length)], axis=0)
            for row_index, length in enumerate(lengths)
        ]
        return np.stack(rows, axis=0)
    if isinstance(position, int):
        return values[:, position]
    raise ValueError(f"Unsupported position selector: {position!r}")


def _resolve_cache_value(cache: Any, hook_name: str) -> Any:
    if hasattr(cache, "__getitem__"):
        try:
            return cache[hook_name]
        except KeyError:
            pass
    if hasattr(cache, "get"):
        value = cache.get(hook_name)
        if value is not None:
            return value
    raise KeyError(f"Hook {hook_name!r} not found in cache")


def run_transformerlens_logits(
    model: Any,
    prompts: Sequence[str],
    *,
    prepend_bos: bool | None = None,
    output_position: str | int = "last",
    use_cache: bool = False,
) -> list[np.ndarray]:
    tokens = _to_tokens(model, prompts, prepend_bos=prepend_bos)
    if use_cache:
        logits, _ = model.run_with_cache(tokens)
    else:
        logits = model(tokens)
    logits_array = _to_numpy(logits)
    selected = _select_sequence_position(
        logits_array,
        tokens,
        position=output_position,
        pad_token_id=_pad_token_id(model),
    )
    return [np.asarray(row) for row in selected]


def capture_transformerlens_activations(
    model: Any,
    prompts: Sequence[str],
    *,
    hook_name: str,
    position: str | int = "last",
    prepend_bos: bool | None = None,
) -> np.ndarray:
    tokens = _to_tokens(model, prompts, prepend_bos=prepend_bos)
    _, cache = model.run_with_cache(tokens, names_filter=lambda name: name == hook_name)
    activations = _to_numpy(_resolve_cache_value(cache, hook_name))
    return _select_sequence_position(
        activations,
        tokens,
        position=position,
        pad_token_id=_pad_token_id(model),
    )


def make_transformerlens_adapter(
    model: Any,
    *,
    prepend_bos: bool | None = None,
    output_position: str | int = "last",
) -> InterventionAdapter:
    if not hasattr(model, "to_tokens"):
        raise TypeError("TransformerLens adapter requires a model with a to_tokens method")
    if not callable(model):
        raise TypeError("TransformerLens adapter requires a callable model")
    if not callable(getattr(model, "run_with_cache", None)):
        raise TypeError("TransformerLens adapter requires a model with a run_with_cache method")

    def run_single(prompt: str) -> np.ndarray:
        return run_transformerlens_logits(
            model,
            [prompt],
            prepend_bos=prepend_bos,
            output_position=output_position,
            use_cache=False,
        )[0]

    def run_batch(prompts: Sequence[str]) -> list[np.ndarray]:
        return run_transformerlens_logits(
            model,
            prompts,
            prepend_bos=prepend_bos,
            output_position=output_position,
            use_cache=False,
        )

    def run_batch_with_cache(prompts: Sequence[str], use_cache: bool) -> list[np.ndarray]:
        return run_transformerlens_logits(
            model,
            prompts,
            prepend_bos=prepend_bos,
            output_position=output_position,
            use_cache=use_cache,
        )

    def capture(prompts: Sequence[str], hook_name: str, position: str | int = "last") -> np.ndarray:
        return capture_transformerlens_activations(
            model,
            prompts,
            hook_name=hook_name,
            position=position,
            prepend_bos=prepend_bos,
        )

    return InterventionAdapter(
        name="transformerlens",
        run_single=run_single,
        run_batch=run_batch,
        run_batch_with_cache=run_batch_with_cache,
        capture_activations=capture,
    )


__all__ = [
    "capture_transformerlens_activations",
    "make_transformerlens_adapter",
    "run_transformerlens_logits",
]
