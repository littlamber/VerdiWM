"""Ctrl-World paired-dose hook and target-local fingerprint evaluation."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MethodType
from typing import Any

from wmloop.evaluate.adapters.ctrl_world_predictive import evaluate_ctrl_world_prediction_receipt
from wmloop.experiments._artifacts import canonical_json, write_bundle
from wmloop.geometry import estimate_response_chart


class CtrlWorldFingerprintError(ValueError):
    """Ctrl-World fingerprint inputs violate the frozen paired-dose contract."""


class CtrlWorldActionEmbeddingDose:
    """Reversible inference-only scaling of a Ctrl-World action encoder output."""

    def __init__(self, model: object, dose: float) -> None:
        if not math.isfinite(float(dose)) or float(dose) <= -1.0:
            raise CtrlWorldFingerprintError("CTRL_WORLD_FINGERPRINT_DOSE_INVALID")
        encoder = getattr(model, "action_encoder", None)
        forward = getattr(encoder, "forward", None)
        if encoder is None or not callable(forward):
            raise CtrlWorldFingerprintError("CTRL_WORLD_ACTION_ENCODER_HOOK_MISSING")
        self.encoder = encoder
        self.original = forward
        self.dose = float(dose)

    def __enter__(self) -> "CtrlWorldActionEmbeddingDose":
        original = self.original
        scale = 1.0 + self.dose

        def scaled_forward(_module: object, *args: object, **kwargs: object) -> object:
            return original(*args, **kwargs) * scale  # type: ignore[operator]

        self.encoder.forward = MethodType(scaled_forward, self.encoder)
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        self.encoder.forward = self.original
        return False


class CtrlWorldActionEmbeddingTemporalMixDose:
    """Reversibly mix action embeddings toward their per-trajectory temporal mean."""

    def __init__(self, model: object, dose: float) -> None:
        if not math.isfinite(float(dose)) or not -1.0 < float(dose) < 1.0:
            raise CtrlWorldFingerprintError("CTRL_WORLD_FINGERPRINT_DOSE_INVALID")
        encoder = getattr(model, "action_encoder", None)
        forward = getattr(encoder, "forward", None)
        if encoder is None or not callable(forward):
            raise CtrlWorldFingerprintError("CTRL_WORLD_ACTION_ENCODER_HOOK_MISSING")
        self.encoder = encoder
        self.original = forward
        self.dose = float(dose)

    def __enter__(self) -> "CtrlWorldActionEmbeddingTemporalMixDose":
        original = self.original
        dose = self.dose

        def mixed_forward(_module: object, *args: object, **kwargs: object) -> object:
            embedding = original(*args, **kwargs)
            ndim = getattr(embedding, "ndim", None)
            mean = getattr(embedding, "mean", None)
            if not isinstance(ndim, int) or ndim < 3 or not callable(mean):
                raise CtrlWorldFingerprintError("CTRL_WORLD_ACTION_EMBEDDING_TEMPORAL_SHAPE_INVALID")
            temporal_mean = mean(dim=1, keepdim=True)
            return embedding + dose * (temporal_mean - embedding)  # type: ignore[operator]

        self.encoder.forward = MethodType(mixed_forward, self.encoder)
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        self.encoder.forward = self.original
        return False


class CtrlWorldCPBEActionEmbeddingDeltaDose:
    """Compile the exact r31 embedding-delta event-phase program at H2."""

    PROBE_ID = "cpbe_residual_63f088b0d5"
    DOSES = (-0.05, -0.025, 0.0, 0.025, 0.05)

    def __init__(self, model: object, dose: float, *, probe: Mapping[str, Any]) -> None:
        encoder = getattr(model, "action_encoder", None)
        forward = getattr(encoder, "forward", None)
        if encoder is None or not callable(forward):
            raise CtrlWorldFingerprintError("CTRL_WORLD_ACTION_ENCODER_HOOK_MISSING")
        required = {
            "probe_id": self.PROBE_ID,
            "hook_type": "H2",
            "signal_source": "action_embedding_delta",
            "spatial_mask": "all_action_embedding",
            "temporal_basis": "event_phase_tangent",
            "contrast_operator": "signed_mean_preserving_phase",
            "aggregation": "goal_outcome_vector",
            "diagnostic_only": True,
            "reversible": True,
        }
        mismatched = sorted(key for key, value in required.items() if probe.get(key) != value)
        if mismatched:
            raise CtrlWorldFingerprintError(
                f"CTRL_WORLD_CPBE_PROGRAM_MISMATCH:{','.join(mismatched)}"
            )
        doses = tuple(float(value) for value in probe.get("doses", ()))
        if doses != self.DOSES or float(dose) not in doses:
            raise CtrlWorldFingerprintError("CTRL_WORLD_CPBE_DOSE_GRID_MISMATCH")
        self.encoder = encoder
        self.original = forward
        self.dose = float(dose)
        self.audit: dict[str, object] = {
            "invocation_count": 0,
            "maximum_temporal_mean_abs_error": 0.0,
            "maximum_temporal_mean_tolerance": 0.0,
        }

    def __enter__(self) -> "CtrlWorldCPBEActionEmbeddingDeltaDose":
        original = self.original
        dose = self.dose
        audit = self.audit

        def phase_forward(_module: object, *args: object, **kwargs: object) -> object:
            embedding = original(*args, **kwargs)
            ndim = getattr(embedding, "ndim", None)
            if not isinstance(ndim, int) or ndim != 3:
                raise CtrlWorldFingerprintError("CTRL_WORLD_CPBE_EMBEDDING_SHAPE_INVALID")
            audit["invocation_count"] = int(audit["invocation_count"]) + 1
            if dose == 0.0:
                return embedding
            if int(embedding.shape[1]) < 2:
                raise CtrlWorldFingerprintError("CTRL_WORLD_CPBE_SEQUENCE_TOO_SHORT")

            delta = embedding[:, 1:] - embedding[:, :-1]
            zero = delta[:, :1].abs().mean(dim=-1, keepdim=True) * 0.0
            event_weight = _concat((zero, delta.abs().mean(dim=-1, keepdim=True)), dim=1)
            event_weight = event_weight.squeeze(-1)
            scale = event_weight.amax(dim=1, keepdim=True)
            event_weight = event_weight / scale.clamp_min(1e-12)
            event_weight = event_weight * (scale > 1e-12).to(event_weight.dtype)
            previous = _concat((event_weight[:, :1] * 0.0, event_weight[:, :-1]), dim=1)
            following = _concat((event_weight[:, 1:], event_weight[:, :1] * 0.0), dim=1)
            tangent = 0.5 * (following - previous)
            temporal_mean = embedding.mean(dim=1, keepdim=True)
            perturbation = tangent.unsqueeze(-1).to(embedding.dtype) * (embedding - temporal_mean)
            perturbation = perturbation - perturbation.mean(dim=1, keepdim=True)
            output = embedding + dose * perturbation
            error = float((output.mean(dim=1) - temporal_mean.squeeze(1)).abs().max().item())
            import torch

            mean_scale = max(1.0, float(temporal_mean.abs().amax().item()))
            tolerance = float(torch.finfo(embedding.dtype).eps) * mean_scale
            audit["maximum_temporal_mean_abs_error"] = max(
                float(audit["maximum_temporal_mean_abs_error"]), error
            )
            audit["maximum_temporal_mean_tolerance"] = max(
                float(audit["maximum_temporal_mean_tolerance"]), tolerance
            )
            return output

        self.encoder.forward = MethodType(phase_forward, self.encoder)
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        self.encoder.forward = self.original
        return False


def ctrl_world_probe_context(
    *, model: object, probe_id: str, dose: float, probe: Mapping[str, Any] | None = None
) -> object:
    if probe_id == "action_conditioning_scale":
        return CtrlWorldActionEmbeddingDose(model, dose)
    if probe_id == "action_embedding_temporal_mix":
        return CtrlWorldActionEmbeddingTemporalMixDose(model, dose)
    if probe_id == CtrlWorldCPBEActionEmbeddingDeltaDose.PROBE_ID:
        if probe is None:
            raise CtrlWorldFingerprintError("CTRL_WORLD_CPBE_PROGRAM_MISSING")
        return CtrlWorldCPBEActionEmbeddingDeltaDose(model, dose, probe=probe)
    raise CtrlWorldFingerprintError(f"CTRL_WORLD_FINGERPRINT_PROBE_UNKNOWN:{probe_id}")


def load_ctrl_world_campaign(path: Path) -> Mapping[str, Any]:
    payload = _load_json(path, "CTRL_WORLD_FINGERPRINT_CAMPAIGN_INVALID")
    if payload.get("artifact_type") != "verdiwm-ctrl-world-fingerprint-campaign":
        raise CtrlWorldFingerprintError("CTRL_WORLD_FINGERPRINT_CAMPAIGN_TYPE_INVALID")
    probe_id = payload.get("probe", {}).get("probe_id")
    if probe_id not in {
        "action_conditioning_scale",
        "action_embedding_temporal_mix",
        CtrlWorldCPBEActionEmbeddingDeltaDose.PROBE_ID,
    }:
        raise CtrlWorldFingerprintError("CTRL_WORLD_FINGERPRINT_PROBE_UNKNOWN")
    doses = tuple(float(value) for value in payload.get("probe", {}).get("doses", ()))
    if 0.0 not in doses or len(doses) < 3 or len(set(doses)) != len(doses):
        raise CtrlWorldFingerprintError("CTRL_WORLD_FINGERPRINT_DOSES_INVALID")
    if not any(-dose in doses for dose in doses if dose != 0.0):
        raise CtrlWorldFingerprintError("CTRL_WORLD_FINGERPRINT_SYMMETRIC_DOSE_MISSING")
    locality = payload.get("locality_admission")
    if not isinstance(locality, Mapping):
        raise CtrlWorldFingerprintError("CTRL_WORLD_FINGERPRINT_LOCALITY_POLICY_MISSING")
    maximum_residual = float(locality.get("maximum_residual", math.nan))
    if not math.isfinite(maximum_residual) or maximum_residual < 0.0:
        raise CtrlWorldFingerprintError("CTRL_WORLD_FINGERPRINT_LOCALITY_THRESHOLD_INVALID")
    if locality.get("failure_policy") != "abstain_from_cross_backbone_transfer":
        raise CtrlWorldFingerprintError("CTRL_WORLD_FINGERPRINT_LOCALITY_FAILURE_POLICY_INVALID")
    return payload


def evaluate_ctrl_world_fingerprint(
    *,
    campaign_path: Path,
    receipt_index_path: Path,
    heldout_split_path: Path,
    protocol: str,
    output_root: Path,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    campaign = load_ctrl_world_campaign(campaign_path)
    protocols = campaign["protocols"]
    if protocol not in protocols:
        raise CtrlWorldFingerprintError("CTRL_WORLD_FINGERPRINT_PROTOCOL_INVALID")
    split_name = str(protocols[protocol]["split"])
    index = _load_json(receipt_index_path, "CTRL_WORLD_FINGERPRINT_RECEIPT_INDEX_INVALID")
    if index.get("artifact_type") != "verdiwm-ctrl-world-fingerprint-receipt-index":
        raise CtrlWorldFingerprintError("CTRL_WORLD_FINGERPRINT_RECEIPT_INDEX_TYPE_INVALID")
    if index.get("campaign_id") != campaign["campaign_id"] or index.get("protocol") != protocol:
        raise CtrlWorldFingerprintError("CTRL_WORLD_FINGERPRINT_RECEIPT_INDEX_MISMATCH")
    rows = index.get("rows")
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise CtrlWorldFingerprintError("CTRL_WORLD_FINGERPRINT_RECEIPT_ROWS_INVALID")

    doses = tuple(float(value) for value in campaign["probe"]["doses"])
    evaluated: dict[float, dict[tuple[str, str, int], tuple[float, ...]]] = {}
    evidence_rows: list[dict[str, object]] = []
    for row in rows:
        dose = float(row["dose"])
        if dose not in doses:
            raise CtrlWorldFingerprintError("CTRL_WORLD_FINGERPRINT_RECEIPT_DOSE_UNKNOWN")
        receipt_path = Path(str(row["receipt_ref"])).resolve()
        evidence = evaluate_ctrl_world_prediction_receipt(
            receipt_path=receipt_path,
            heldout_split_path=heldout_split_path,
            split_name=split_name,
        )
        identity = (str(evidence["task_id"]), str(evidence["episode_id"]), int(evidence["seed"]))
        values = tuple(
            float(evidence["metrics"][str(outcome["source_metric"])]) * float(outcome["sign"])
            for outcome in campaign["outcomes"]
        )
        if identity in evaluated.setdefault(dose, {}):
            raise CtrlWorldFingerprintError("CTRL_WORLD_FINGERPRINT_RECEIPT_DUPLICATE")
        evaluated[dose][identity] = values
        evidence_rows.append(
            {
                "schema_version": 1,
                "artifact_type": "verdiwm-ctrl-world-fingerprint-measurement",
                "campaign_id": campaign["campaign_id"],
                "protocol": protocol,
                "dose": dose,
                "identity": {"task_id": identity[0], "episode_id": identity[1], "seed": identity[2]},
                "outcomes": list(values),
                "receipt_ref": str(receipt_path),
                "evidence_source": evidence["evidence_source"],
                "horizon_frames": evidence["horizon_frames"],
            }
        )
    expected_identities = set(evaluated.get(0.0, {}))
    required = int(protocols[protocol]["required_receipts_per_dose"])
    if len(expected_identities) != required:
        raise CtrlWorldFingerprintError("CTRL_WORLD_FINGERPRINT_BASELINE_FRAME_INVALID")
    for dose in doses:
        if set(evaluated.get(dose, {})) != expected_identities:
            raise CtrlWorldFingerprintError(f"CTRL_WORLD_FINGERPRINT_PAIRED_FRAME_INVALID:{dose}")
    identities = tuple(sorted(expected_identities))
    chart = estimate_response_chart(
        chart_id=f"{campaign['campaign_id']}:{protocol}",
        goal_schema="ctrl_world_acwm_predictive_quality_v1",
        outcome_names=tuple(str(value["name"]) for value in campaign["outcomes"]),
        outcome_weights=tuple(float(value["weight"]) for value in campaign["outcomes"]),
        baseline_repeats=tuple(evaluated[0.0][identity] for identity in identities),
        dose_observations={
            str(campaign["probe"]["probe_id"]): {
                dose: tuple(evaluated[dose][identity] for identity in identities)
                for dose in doses
                if dose != 0.0
            }
        },
    ).to_dict()
    locality_threshold = float(campaign["locality_admission"]["maximum_residual"])
    locality_residuals = {
        str(name): float(value) for name, value in chart["locality_residuals"].items()
    }
    supported_paths = sorted(
        name for name, residual in locality_residuals.items() if residual <= locality_threshold
    )
    locality_pass = bool(supported_paths)
    report = {
        "schema_version": 1,
        "artifact_type": "verdiwm-ctrl-world-target-local-fingerprint",
        "state": "ready",
        "campaign_id": campaign["campaign_id"],
        "protocol": protocol,
        "split": split_name,
        "measurement_count": len(evidence_rows),
        "repeat_count": len(identities),
        "chart": chart,
        "locality_admission": {
            "state": "passed" if locality_pass else "failed",
            "maximum_residual": locality_threshold,
            "path_residuals": locality_residuals,
            "supported_local_paths": supported_paths,
            "cross_backbone_transfer_eligible": locality_pass,
            "failure_policy": campaign["locality_admission"]["failure_policy"],
        },
        "claim_boundary": campaign["claim_scope"],
        "downstream_task_success_used_for_verdict": False,
    }
    destination = Path(output_root).resolve()
    return write_bundle(
        output_root=destination,
        files={
            "target-local-fingerprint.json": canonical_json(report),
            "measurements.jsonl": b"".join(canonical_json(row) for row in evidence_rows),
            "input-campaign.json": canonical_json(campaign),
            "input-receipt-index.json": canonical_json(index),
        },
        manifest_fields={
            "artifact_type": "verdiwm-ctrl-world-target-local-fingerprint-manifest",
            "state": "ready",
            "campaign_id": campaign["campaign_id"],
            "protocol": protocol,
            "split": split_name,
            "measurement_count": len(evidence_rows),
            "repeat_count": len(identities),
            "locality_admission_state": "passed" if locality_pass else "failed",
            "cross_backbone_transfer_eligible": locality_pass,
            "report_path": str(destination / "target-local-fingerprint.json"),
        },
        archive_db=archive_db,
        cas_root=cas_root,
    )


def _load_json(path: Path, code: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CtrlWorldFingerprintError(f"{code}:{path}") from exc
    if not isinstance(payload, Mapping):
        raise CtrlWorldFingerprintError(f"{code}:{path}")
    return payload


def _concat(values: Sequence[object], *, dim: int) -> object:
    first = values[0]
    module = type(first).__module__.split(".", 1)[0]
    if module != "torch":
        raise CtrlWorldFingerprintError("CTRL_WORLD_CPBE_TENSOR_TYPE_INVALID")
    import torch

    return torch.cat(tuple(values), dim=dim)
