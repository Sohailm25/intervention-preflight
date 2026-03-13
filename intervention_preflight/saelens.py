"""SAELens-oriented preflight helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from intervention_preflight.reconstruction import audit_reconstruction, compare_metadata
from intervention_preflight.report import aggregate_reports, build_report


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return dict(to_dict())
    items = getattr(value, "items", None)
    if callable(items):
        return {str(key): item for key, item in items()}
    if hasattr(value, "__dict__"):
        return {
            str(key): item
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }
    raise TypeError(f"Cannot coerce metadata value of type {type(value)!r} to a dict")


def _extract_model_name(model: Any) -> str | None:
    cfg = getattr(model, "cfg", None)
    if cfg is not None:
        for key in ("model_name", "original_architecture", "architecture"):
            value = getattr(cfg, key, None)
            if value:
                return str(value)
    for key in ("name", "model_name"):
        value = getattr(model, key, None)
        if value:
            return str(value)
    return None


def _get_hook_names(model: Any) -> set[str] | None:
    hook_dict = getattr(model, "hook_dict", None)
    if hook_dict is None:
        return None
    if isinstance(hook_dict, Mapping):
        return {str(key) for key in hook_dict.keys()}
    keys = getattr(hook_dict, "keys", None)
    if callable(keys):
        return {str(key) for key in keys()}
    return None


def _resolve_base_hook_name(model: Any, base_hook_name: str | None) -> str | None:
    if not base_hook_name:
        return None
    resolver = getattr(model, "_resolve_hook_name", None)
    if callable(resolver):
        try:
            resolved = resolver(base_hook_name)
        except Exception:
            return base_hook_name
        if resolved:
            return str(resolved)
    return base_hook_name


def extract_saelens_metadata(sae: Any) -> dict[str, Any]:
    cfg = getattr(sae, "cfg", None)
    metadata = getattr(cfg, "metadata", None)
    return _as_dict(metadata)


def audit_saelens_metadata(
    sae: Any,
    *,
    expected_metadata: Mapping[str, Any] | None = None,
    metadata_keys: list[str] | tuple[str, ...] | None = None,
    required_keys: list[str] | tuple[str, ...] = ("hook_name",),
    metadata_mismatch_as_fail: bool = False,
) -> dict[str, Any]:
    observed_metadata = extract_saelens_metadata(sae)
    tracked_keys = metadata_keys
    if tracked_keys is None and expected_metadata is not None:
        tracked_keys = tuple(expected_metadata.keys())
    metadata_report = compare_metadata(
        expected=expected_metadata,
        observed=observed_metadata,
        keys=tracked_keys,
    )
    missing_required = [
        key for key in required_keys if observed_metadata.get(key) in (None, "", ())
    ]

    notes: list[str] = []
    status = "pass"
    if missing_required:
        status = "fail"
        notes.append(
            "Missing required SAELens metadata fields: "
            + ", ".join(sorted(missing_required))
        )
    has_metadata_problem = bool(
        metadata_report["mismatch_count"]
        or metadata_report["missing_from_observed"]
        or metadata_report["missing_from_expected"]
    )
    if has_metadata_problem:
        notes.append("Observed SAELens metadata does not match the expected context.")
        if metadata_mismatch_as_fail or status == "fail":
            status = "fail"
        elif status == "pass":
            status = "warn"

    return build_report(
        check="saelens_metadata_audit",
        status=status,
        summary={
            "required_keys_ok": not missing_required,
            "metadata_ok": not has_metadata_problem,
            "tracked_key_count": metadata_report["tracked_key_count"],
        },
        details={
            "required_keys": list(required_keys),
            "missing_required": missing_required,
            "observed_metadata": observed_metadata,
            "metadata": metadata_report,
        },
        notes=notes,
        metadata={"backend": "saelens"},
    )


def audit_saelens_hook_compatibility(
    sae: Any,
    model: Any,
    *,
    internal_hook_suffix: str = "hook_sae_acts_post",
) -> dict[str, Any]:
    observed_metadata = extract_saelens_metadata(sae)
    base_hook_name = (
        observed_metadata.get("hook_name_out") or observed_metadata.get("hook_name")
    )
    resolved_hook_name = _resolve_base_hook_name(model, base_hook_name)
    resolved_internal_hook = None
    get_sae_hook_name = getattr(model, "get_sae_hook_name", None)
    if callable(get_sae_hook_name):
        try:
            resolved_internal_hook = get_sae_hook_name(
                sae,
                internal=internal_hook_suffix,
            )
        except Exception:
            resolved_internal_hook = None

    available_hooks = _get_hook_names(model)
    model_name = _extract_model_name(model)

    if available_hooks is None:
        status = "warn"
        notes = ["Model does not expose hook_dict, so hook presence could not be verified."]
        base_present = None
        resolved_present = None
        internal_present = None
    else:
        base_present = base_hook_name in available_hooks if base_hook_name else False
        resolved_present = (
            resolved_hook_name in available_hooks if resolved_hook_name else False
        )
        internal_present = (
            resolved_internal_hook in available_hooks
            if resolved_internal_hook is not None
            else None
        )
        status = "pass" if (base_present or resolved_present) else "fail"
        notes = []

    metadata_model_name = observed_metadata.get("model_name")
    if metadata_model_name and model_name and metadata_model_name != model_name:
        if status == "pass":
            status = "warn"
        notes.append(
            f"SAE metadata model_name={metadata_model_name!r} does not match model_name={model_name!r}."
        )
    if base_hook_name and resolved_hook_name and base_hook_name != resolved_hook_name:
        notes.append(
            f"Model resolves {base_hook_name!r} to {resolved_hook_name!r}; alias-aware checks are required."
        )
    if available_hooks is not None and not (base_present or resolved_present):
        notes.append(
            "Neither the metadata hook name nor its resolved form appears in model.hook_dict."
        )

    return build_report(
        check="saelens_hook_compatibility",
        status=status,
        summary={
            "base_hook_present": base_present,
            "resolved_hook_present": resolved_present,
            "alias_changed": bool(
                base_hook_name
                and resolved_hook_name
                and base_hook_name != resolved_hook_name
            ),
            "model_name_matches": (
                None
                if not (metadata_model_name and model_name)
                else metadata_model_name == model_name
            ),
        },
        details={
            "base_hook_name": base_hook_name,
            "resolved_hook_name": resolved_hook_name,
            "resolved_internal_hook": resolved_internal_hook,
            "internal_hook_present": internal_present,
            "available_hook_count": (
                None if available_hooks is None else len(available_hooks)
            ),
            "model_name": model_name,
            "metadata_model_name": metadata_model_name,
        },
        notes=notes,
        metadata={"backend": "saelens"},
    )


def audit_saelens_reconstruction(
    sae: Any,
    activations: Any,
    *,
    expected_metadata: Mapping[str, Any] | None = None,
    metadata_keys: list[str] | tuple[str, ...] | None = None,
    name: str = "saelens_reconstruction",
    metadata_mismatch_as_fail: bool = False,
    cosine_pass_min: float = 0.9,
    cosine_warn_min: float = 0.8,
    explained_variance_pass_min: float = 0.9,
    explained_variance_warn_min: float = 0.8,
) -> dict[str, Any]:
    encode = getattr(sae, "encode", None)
    decode = getattr(sae, "decode", None)
    if not callable(encode) or not callable(decode):
        raise TypeError("SAELens reconstruction audit requires encode() and decode() methods")

    feature_acts = encode(activations)
    reconstructed = decode(feature_acts)
    observed_metadata = extract_saelens_metadata(sae)
    compare_metadata_context = expected_metadata is not None or metadata_keys is not None
    tracked_keys = metadata_keys
    if tracked_keys is None and expected_metadata is not None:
        tracked_keys = tuple(expected_metadata.keys())
    report = audit_reconstruction(
        original=activations,
        reconstructed=reconstructed,
        name=name,
        cosine_pass_min=cosine_pass_min,
        cosine_warn_min=cosine_warn_min,
        explained_variance_pass_min=explained_variance_pass_min,
        explained_variance_warn_min=explained_variance_warn_min,
        expected_metadata=expected_metadata if compare_metadata_context else None,
        observed_metadata=observed_metadata if compare_metadata_context else None,
        metadata_keys=tracked_keys if compare_metadata_context else None,
        metadata_mismatch_as_fail=metadata_mismatch_as_fail,
    )
    feature_array = np.asarray(feature_acts)
    report["check"] = "saelens_reconstruction_audit"
    report["summary"]["feature_shape"] = list(feature_array.shape)
    report["metrics"]["feature_nonzero_fraction"] = float(
        np.mean(np.abs(feature_array) > 0)
    )
    return report


def audit_saelens_preflight(
    sae: Any,
    *,
    model: Any | None = None,
    activations: Any | None = None,
    expected_metadata: Mapping[str, Any] | None = None,
    metadata_keys: list[str] | tuple[str, ...] | None = None,
    required_keys: list[str] | tuple[str, ...] = ("hook_name",),
    metadata_mismatch_as_fail: bool = False,
    internal_hook_suffix: str = "hook_sae_acts_post",
) -> dict[str, Any]:
    reports = [
        audit_saelens_metadata(
            sae,
            expected_metadata=expected_metadata,
            metadata_keys=metadata_keys,
            required_keys=required_keys,
            metadata_mismatch_as_fail=metadata_mismatch_as_fail,
        )
    ]
    if model is not None:
        reports.append(
            audit_saelens_hook_compatibility(
                sae,
                model,
                internal_hook_suffix=internal_hook_suffix,
            )
        )
    if activations is not None:
        reports.append(
            audit_saelens_reconstruction(
                sae,
                activations,
                expected_metadata=expected_metadata,
                metadata_keys=metadata_keys,
                metadata_mismatch_as_fail=metadata_mismatch_as_fail,
            )
        )

    notes: list[str] = []
    if model is None:
        notes.append("Model hook compatibility was not checked.")
    if activations is None:
        notes.append("Encode/decode reconstruction sanity was not checked.")

    return aggregate_reports(
        "saelens_preflight",
        reports,
        notes=notes,
        metadata={"backend": "saelens"},
    )


__all__ = [
    "audit_saelens_hook_compatibility",
    "audit_saelens_metadata",
    "audit_saelens_preflight",
    "audit_saelens_reconstruction",
    "extract_saelens_metadata",
]
