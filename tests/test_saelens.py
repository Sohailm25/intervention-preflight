from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from intervention_preflight.saelens import (
    audit_saelens_hook_compatibility,
    audit_saelens_metadata,
    audit_saelens_preflight,
    audit_saelens_reconstruction,
    extract_saelens_metadata,
)


@dataclass
class DummyMetadata:
    hook_name: str | None = "blocks.0.hook_resid_pre"
    hook_name_out: str | None = None
    model_name: str | None = "tiny-stories"
    hook_head_index: int | None = None
    seqpos_slice: tuple[int | None, ...] | None = (None,)
    model_from_pretrained_kwargs: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "hook_name": self.hook_name,
            "hook_name_out": self.hook_name_out,
            "model_name": self.model_name,
            "hook_head_index": self.hook_head_index,
            "seqpos_slice": self.seqpos_slice,
            "model_from_pretrained_kwargs": self.model_from_pretrained_kwargs or {},
        }


@dataclass
class DummyConfig:
    metadata: DummyMetadata


class DummySAE:
    def __init__(self, metadata: DummyMetadata):
        self.cfg = DummyConfig(metadata=metadata)

    def encode(self, activations):
        return np.asarray(activations) * 2.0

    def decode(self, features):
        return np.asarray(features) / 2.0


class DummyModel:
    def __init__(
        self,
        hooks: list[str],
        *,
        model_name: str = "tiny-stories",
        resolved: dict[str, str] | None = None,
    ):
        self.hook_dict = {hook_name: object() for hook_name in hooks}
        self.cfg = type("Cfg", (), {"model_name": model_name})()
        self._resolved = dict(resolved or {})

    def _resolve_hook_name(self, hook_name: str) -> str:
        return self._resolved.get(hook_name, hook_name)

    def get_sae_hook_name(self, sae, internal: str = "hook_sae_acts_post") -> str:
        metadata = sae.cfg.metadata
        base_hook = metadata.hook_name_out or metadata.hook_name
        resolved = self._resolve_hook_name(base_hook)
        return f"{resolved}.{internal}"


def test_extract_saelens_metadata_uses_to_dict() -> None:
    sae = DummySAE(DummyMetadata())
    metadata = extract_saelens_metadata(sae)
    assert metadata["hook_name"] == "blocks.0.hook_resid_pre"
    assert metadata["model_name"] == "tiny-stories"


def test_audit_saelens_metadata_warns_on_expected_mismatch() -> None:
    sae = DummySAE(DummyMetadata(model_name="model-a"))
    report = audit_saelens_metadata(
        sae,
        expected_metadata={"model_name": "model-b", "hook_name": "blocks.0.hook_resid_pre"},
    )
    assert report["status"] == "warn"
    assert report["summary"]["metadata_ok"] is False


def test_audit_saelens_metadata_fails_when_required_hook_missing() -> None:
    sae = DummySAE(DummyMetadata(hook_name=None))
    report = audit_saelens_metadata(sae)
    assert report["status"] == "fail"
    assert report["details"]["missing_required"] == ["hook_name"]


def test_audit_saelens_hook_compatibility_passes_with_alias_resolution() -> None:
    sae = DummySAE(DummyMetadata(hook_name="blocks.0.hook_mlp_out"))
    model = DummyModel(
        ["blocks.0.mlp.hook_out"],
        resolved={"blocks.0.hook_mlp_out": "blocks.0.mlp.hook_out"},
    )
    report = audit_saelens_hook_compatibility(sae, model)
    assert report["status"] == "pass"
    assert report["summary"]["resolved_hook_present"] is True
    assert report["summary"]["alias_changed"] is True


def test_audit_saelens_hook_compatibility_fails_when_hook_absent() -> None:
    sae = DummySAE(DummyMetadata(hook_name="blocks.0.hook_resid_pre"))
    model = DummyModel(["blocks.1.hook_resid_pre"])
    report = audit_saelens_hook_compatibility(sae, model)
    assert report["status"] == "fail"
    assert report["summary"]["base_hook_present"] is False


def test_audit_saelens_reconstruction_reports_feature_density() -> None:
    sae = DummySAE(DummyMetadata())
    report = audit_saelens_reconstruction(
        sae,
        activations=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
    )
    assert report["status"] == "pass"
    assert report["check"] == "saelens_reconstruction_audit"
    assert report["summary"]["feature_shape"] == [2, 2]
    assert report["metrics"]["feature_nonzero_fraction"] == 1.0


def test_audit_saelens_preflight_aggregates_reports() -> None:
    sae = DummySAE(DummyMetadata())
    model = DummyModel(["blocks.0.hook_resid_pre"])
    report = audit_saelens_preflight(
        sae,
        model=model,
        activations=np.array([[1.0, 0.0], [0.5, 0.25]], dtype=np.float64),
        expected_metadata={"model_name": "tiny-stories", "hook_name": "blocks.0.hook_resid_pre"},
    )
    assert report["status"] == "pass"
    assert report["summary"]["report_count"] == 3
