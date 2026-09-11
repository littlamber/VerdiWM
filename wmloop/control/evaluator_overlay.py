"""Materialize a user-confirmed evaluator candidate into VERDI-owned storage.

Model repositories are read-only inputs.  This module copies a JSON/YAML
candidate, rewrites paths that point at the source checkout or an old machine,
and records both digests.  The first call produces a reviewable overlay;
``confirm=True`` is the only transition that marks it frozen.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping


class EvaluatorOverlayError(ValueError):
    """The candidate cannot be safely copied or frozen."""


def materialize_evaluator_overlay(
    candidate_path: Path,
    *,
    source_root: Path,
    model_root: Path,
    data_root: Path,
    output_root: Path,
    confirm: bool = False,
) -> dict[str, object]:
    """Copy and path-rewrite one candidate without touching its source file."""

    candidate = _regular_file(candidate_path, "EVALUATOR_CANDIDATE_INVALID")
    source = _directory(source_root, "EVALUATOR_SOURCE_ROOT_INVALID")
    model = _directory(model_root, "EVALUATOR_MODEL_ROOT_INVALID")
    data = _path(data_root, "EVALUATOR_DATA_ROOT_INVALID")
    destination_root = Path(output_root).expanduser().resolve()
    if destination_root == source or destination_root == model or _within(destination_root, source) or _within(destination_root, model) or _within(destination_root, data):
        raise EvaluatorOverlayError("EVALUATOR_OVERLAY_OUTPUT_OVERLAP")
    payload = _load(candidate)
    rewrites: list[dict[str, str]] = []
    rewritten = _rewrite(payload, source=source, model=model, data=data, rewrites=rewrites, key="")
    if not isinstance(rewritten, Mapping):
        raise EvaluatorOverlayError("EVALUATOR_CANDIDATE_OBJECT_REQUIRED")
    # Repository manifests usually describe a protocol but are not directly
    # runnable evaluator contracts.  Wrap those manifests in the small
    # contract understood by VERDI while retaining the original metadata.
    rewritten = _as_contract(rewritten, source=source, candidate=candidate)
    relative = _safe_name(candidate.stem) + candidate.suffix.lower()
    overlay = destination_root / "evaluators" / relative
    overlay.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    encoded = _encode(rewritten, suffix=overlay.suffix)
    overlay.write_bytes(encoded)
    source_sha = _sha256(candidate.read_bytes())
    overlay_sha = _sha256(encoded)
    state = "frozen" if confirm else "ready_for_confirmation"
    receipt = {
        "schema_version": 1,
        "artifact_type": "verdiwm-evaluator-overlay",
        "state": state,
        "source_candidate": str(candidate),
        "overlay_path": str(overlay),
        "source_sha256": source_sha,
        "overlay_sha256": overlay_sha,
        "rewrites": rewrites,
        "confirmation": {"required": not confirm, "confirmed": bool(confirm)},
        "claim_boundary": (
            "Overlay is a copied candidate with deterministic path rewrites; no evaluator is frozen until explicit confirmation."
            if not confirm
            else "Evaluator is frozen in VERDI-owned storage for this exact overlay digest; target-side evidence is still required."
        ),
    }
    receipt_path = overlay.with_suffix(overlay.suffix + ".overlay.json")
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    receipt["receipt_path"] = str(receipt_path)
    return receipt


def _as_contract(payload: Mapping[str, object], *, source: Path, candidate: Path) -> dict[str, object]:
    required = {"evaluator_id", "command", "input_artifacts", "output_artifacts", "metrics", "verifier"}
    if required.issubset(payload):
        return dict(payload)
    protocol = payload.get("protocol") if isinstance(payload.get("protocol"), Mapping) else {}
    fps = protocol.get("fps") if isinstance(protocol, Mapping) else None
    chunks = protocol.get("closed_loop_chunks") if isinstance(protocol, Mapping) else None
    metrics = payload.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        metrics = ["trajectory_accuracy", "action_following", "closed_loop_drift"]
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-evaluator-contract",
        "evaluator_id": f"verdi-overlay-{_safe_name(candidate.stem)}-v1",
        "command": [
            "{python}",
            "{repo_root}/scripts/evaluation/run_sa_wm_eval_manifest.py",
        ],
        "input_artifacts": ["evaluation_manifest", "checkpoint", "paired_target"],
        "output_artifacts": ["metrics.json", "stdout.log", "stderr.log"],
        "metrics": [str(item) for item in metrics if str(item).strip()],
        "verifier": "verdi_overlay_target_side_verifier_v1",
        "working_directory": str(source),
        "entrypoint_probe": "help",
        "source_manifest": str(candidate),
        "protocol_summary": {
            "fps": fps,
            "closed_loop_chunks": chunks,
            "source_role": payload.get("role"),
        },
        "claim_boundary": "VERDI-owned overlay generated from a repository candidate. It licenses no quality claim until target-side paired evidence is settled.",
    }


def _load(path: Path) -> object:
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() == ".json":
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise EvaluatorOverlayError("EVALUATOR_CANDIDATE_MALFORMED") from exc
    else:
        try:
            import yaml  # type: ignore[import-not-found]
            value = yaml.safe_load(text)
        except Exception as exc:
            raise EvaluatorOverlayError("EVALUATOR_CANDIDATE_YAML_UNAVAILABLE") from exc
    if not isinstance(value, Mapping):
        raise EvaluatorOverlayError("EVALUATOR_CANDIDATE_OBJECT_REQUIRED")
    return value


def _encode(value: object, *, suffix: str) -> bytes:
    if suffix == ".json":
        return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    try:
        import yaml  # type: ignore[import-not-found]
        return yaml.safe_dump(value, allow_unicode=True, sort_keys=True).encode("utf-8")
    except Exception as exc:
        raise EvaluatorOverlayError("EVALUATOR_OVERLAY_YAML_UNAVAILABLE") from exc


def _rewrite(value: object, *, source: Path, model: Path, data: Path, rewrites: list[dict[str, str]], key: str) -> object:
    if isinstance(value, Mapping):
        return {str(name): _rewrite(child, source=source, model=model, data=data, rewrites=rewrites, key=str(name)) for name, child in value.items()}
    if isinstance(value, list):
        return [_rewrite(child, source=source, model=model, data=data, rewrites=rewrites, key=key) for child in value]
    if not isinstance(value, str) or not value.startswith("/"):
        return value
    replacement = _replacement(value, source=source, model=model, data=data, key=key)
    if replacement != value:
        rewrites.append({"field": key, "from": value, "to": replacement})
    return replacement


def _replacement(value: str, *, source: Path, model: Path, data: Path, key: str) -> str:
    path = Path(value)
    # Preserve paths that already point into the caller's roots.
    if _within(path, source) or _within(path, model) or _within(path, data):
        return value
    lowered = value.casefold()
    field = key.casefold()
    # Resolve source/repository bindings first.  ``repo_root`` and
    # ``source_root`` contain the generic word ``root``; treating that token
    # as a dataset hint would silently redirect evaluator code to the data
    # directory.
    if "robocoach" in lowered or any(token in field for token in ("repo", "source", "code", "checkout")):
        return str(source)
    if any(token in field for token in ("checkpoint", "ckpt", "model")) or "/checkpoints/" in lowered:
        return str(model)
    if any(token in field for token in ("data", "collection", "dataset")) or "video_latent" in lowered or "/droid" in lowered:
        return str(data)
    return value


def _regular_file(path: Path, code: str) -> Path:
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_file() or candidate.is_symlink():
        raise EvaluatorOverlayError(code)
    return candidate


def _directory(path: Path, code: str) -> Path:
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_dir() or candidate.is_symlink():
        raise EvaluatorOverlayError(code)
    return candidate


def _path(path: Path, code: str) -> Path:
    candidate = Path(path).expanduser().resolve()
    if not candidate.exists() or candidate.is_symlink():
        raise EvaluatorOverlayError(code)
    return candidate


def _within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")
    return value or "evaluator"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
