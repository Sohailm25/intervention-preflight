from __future__ import annotations

import numpy as np

from intervention_preflight.adapters import (
    InterventionAdapter,
    make_transformerlens_adapter,
    require_activation_capture,
    require_cache_controls,
)
from intervention_preflight.parity import check_batch_single_parity, check_cache_parity


class _FakeTokenizer:
    pad_token_id = 0


class FakeTransformerLensModel:
    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()

    def to_tokens(self, prompts: list[str], prepend_bos: bool = True) -> np.ndarray:
        rows: list[list[int]] = []
        for prompt in prompts:
            token_row = [ord(char) - 96 for char in prompt.lower() if char.isalpha()]
            if prepend_bos:
                token_row = [99] + token_row
            rows.append(token_row or [99])
        max_len = max(len(row) for row in rows)
        padded = [row + [self.tokenizer.pad_token_id] * (max_len - len(row)) for row in rows]
        return np.asarray(padded, dtype=np.int64)

    def __call__(self, tokens: np.ndarray) -> np.ndarray:
        tokens = np.asarray(tokens, dtype=np.float64)
        batch_size, seq_len = tokens.shape
        vocab_size = 4
        logits = np.zeros((batch_size, seq_len, vocab_size), dtype=np.float64)
        logits[:, :, 0] = tokens
        logits[:, :, 1] = tokens + 1.0
        logits[:, :, 2] = tokens * 2.0
        logits[:, :, 3] = tokens % 3.0
        return logits

    def run_with_cache(
        self,
        tokens: np.ndarray,
        names_filter=None,  # noqa: ANN001 - matches backend shape
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        logits = self(tokens)
        residual = np.stack(
            [
                np.asarray(tokens, dtype=np.float64),
                np.asarray(tokens, dtype=np.float64) + 1.0,
                np.asarray(tokens, dtype=np.float64) + 2.0,
            ],
            axis=-1,
        )
        cache = {"blocks.0.hook_resid_pre": residual}
        if callable(names_filter):
            cache = {name: value for name, value in cache.items() if names_filter(name)}
        return logits, cache


def test_make_transformerlens_adapter_exposes_minimal_surface() -> None:
    adapter = make_transformerlens_adapter(FakeTransformerLensModel())
    assert isinstance(adapter, InterventionAdapter)
    assert adapter.name == "transformerlens"
    assert adapter.supports_cache_controls()
    assert adapter.supports_activation_capture()


def test_transformerlens_adapter_batch_single_parity_passes() -> None:
    adapter = make_transformerlens_adapter(FakeTransformerLensModel())
    report = check_batch_single_parity(
        ["alpha", "beta"],
        run_single=adapter.run_single,
        run_batch=adapter.run_batch,
        tolerance=0.0,
    )
    assert report["status"] == "pass"
    assert report["metrics"]["failing_count"] == 0


def test_transformerlens_adapter_cache_parity_passes() -> None:
    adapter = make_transformerlens_adapter(FakeTransformerLensModel())
    cache_runner = require_cache_controls(adapter)
    report = check_cache_parity(
        ["alpha", "beta"],
        run_with_cache=lambda prompts: cache_runner(prompts, True),
        run_without_cache=lambda prompts: cache_runner(prompts, False),
        tolerance=0.0,
    )
    assert report["status"] == "pass"
    assert report["metrics"]["failing_count"] == 0


def test_transformerlens_adapter_captures_last_non_padding_position() -> None:
    adapter = make_transformerlens_adapter(FakeTransformerLensModel())
    capture = require_activation_capture(adapter)
    activations = capture(["alpha", "be"], "blocks.0.hook_resid_pre", "last")
    assert activations.shape == (2, 3)
    assert activations[0].tolist() == [1.0, 2.0, 3.0]
    assert activations[1].tolist() == [5.0, 6.0, 7.0]


def test_transformerlens_adapter_captures_mean_without_padding_leakage() -> None:
    adapter = make_transformerlens_adapter(FakeTransformerLensModel())
    capture = require_activation_capture(adapter)
    activations = capture(["alpha", "be"], "blocks.0.hook_resid_pre", "mean")
    assert activations.shape == (2, 3)
    assert np.allclose(activations[0], np.array([22.83333333, 23.83333333, 24.83333333]))
    assert np.allclose(activations[1], np.array([35.33333333, 36.33333333, 37.33333333]))
