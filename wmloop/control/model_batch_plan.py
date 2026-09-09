"""Model batch plan implementation."""

from __future__ import annotations
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from wmloop.contracts import ContractValidationError, validate_document
from wmloop.control.adapter_profiles import AdapterProfileError, parse_gpu_budget
from wmloop.control.batch_state import ModelBatchError


_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{2,63}$")


def _batch_campaign_id(batch_id: str, model_id: str) -> str:
    """Return a readable, deterministic campaign id within the 64-char limit."""

    suffix = hashlib.sha256(f"{batch_id}:{model_id}".encode("utf-8")).hexdigest()[:8]
    prefix = f"batch-{batch_id[:20]}-{model_id[:20]}"
    return f"{prefix}-{suffix}"[:64].rstrip("-")


def compile_model_batch(*, request_path: Path, output_root: Path, root: Path | None = None) -> dict[str, object]:
    """Validate one batch request and write an immutable local plan.

    This function never imports a model, runs an evaluator, allocates a GPU, or
    creates a verdict. Later dispatch must consume the plan through the normal
    campaign store, scheduler, budget ledger, and receipt settlement paths.
    """

    repo = (root or Path(__file__).resolve().parents[2]).resolve()
    source = Path(request_path).expanduser().resolve()
    if not source.is_file() or source.is_symlink():
        raise ModelBatchError("MODEL_BATCH_REQUEST_NOT_FOUND")
    try:
        request_bytes = source.read_bytes()
        request = json.loads(request_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelBatchError("MODEL_BATCH_REQUEST_INVALID") from exc
    if not isinstance(request, Mapping):
        raise ModelBatchError("MODEL_BATCH_REQUEST_INVALID")
    try:
        validate_document("model_batch_request", request, root=repo)
    except ContractValidationError as exc:
        raise ModelBatchError(f"MODEL_BATCH_REQUEST_INVALID:{exc}") from exc

    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw in request["models"]:
        assert isinstance(raw, Mapping)
        row = _compile_model_row(raw, request=request)
        model_id = str(row["model_id"])
        if model_id in seen:
            raise ModelBatchError(f"MODEL_BATCH_MODEL_ID_DUPLICATE:{model_id}")
        seen.add(model_id)
        rows.append(row)

    body: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-model-batch-plan",
        "state": "ready_for_dispatch" if all(row["state"] == "ready" for row in rows) else "blocked",
        "batch_id": str(request["batch_id"]),
        "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
        "goal": str(request["goal"]),
        "models": rows,
        "budget": {
            "declared_gpu_hours": sum(float(row["budget"]) for row in rows),
            "model_count": len(rows),
            "policy": "sum_of_model_budgets",
        },
        "claim_boundary": (
            "This plan binds heterogeneous model inputs for later campaign dispatch. "
            "It performs no model execution, evaluation, GPU allocation, or transfer claim."
        ),
    }
    body["plan_sha256"] = _digest(body)
    try:
        validate_document("model_batch_plan", body, root=repo)
    except ContractValidationError as exc:
        raise ModelBatchError(f"MODEL_BATCH_PLAN_INVALID:{exc}") from exc
    _write_plan(Path(output_root).expanduser().resolve(), request_bytes=request_bytes, plan=body)
    return body


def load_model_batch_plan(path: Path, *, root: Path | None = None) -> dict[str, object]:
    """Load and validate a plan without changing it."""

    repo = (root or Path(__file__).resolve().parents[2]).resolve()
    source = Path(path).expanduser().resolve()
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelBatchError("MODEL_BATCH_PLAN_INVALID") from exc
    if not isinstance(document, dict):
        raise ModelBatchError("MODEL_BATCH_PLAN_INVALID")
    try:
        validate_document("model_batch_plan", document, root=repo)
    except ContractValidationError as exc:
        raise ModelBatchError(f"MODEL_BATCH_PLAN_INVALID:{exc}") from exc
    received = document.get("plan_sha256")
    body = dict(document)
    body.pop("plan_sha256", None)
    if received != _digest(body):
        raise ModelBatchError("MODEL_BATCH_PLAN_DIGEST_MISMATCH")
    for row in document.get("models", []):
        if not isinstance(row, Mapping):
            raise ModelBatchError("MODEL_BATCH_PLAN_INVALID")
        files = row.get("files")
        hashes = row.get("file_sha256")
        if not isinstance(files, Mapping) or not isinstance(hashes, Mapping):
            raise ModelBatchError("MODEL_BATCH_PLAN_INVALID")
        for field, expected in hashes.items():
            path_value = files.get(field)
            if expected is None:
                continue
            if not isinstance(path_value, str) or not Path(path_value).exists():
                raise ModelBatchError(f"MODEL_BATCH_BINDING_MISSING:{row.get('model_id')}:{field}")
            if _sha256_path(Path(path_value)) != expected:
                raise ModelBatchError(f"MODEL_BATCH_BINDING_DRIFT:{row.get('model_id')}:{field}")
    return document


def _compile_model_row(raw: Mapping[str, object], *, request: Mapping[str, object]) -> dict[str, object]:
    model_id = str(raw["model_id"])
    if _ID.fullmatch(model_id) is None:
        raise ModelBatchError(f"MODEL_BATCH_MODEL_ID_INVALID:{model_id}")
    model = _required_path(raw["model"], directory=True, code="MODEL_BATCH_MODEL_INVALID")
    data = _required_path(raw["data"], directory=False, code="MODEL_BATCH_DATA_INVALID")
    goal = str(raw.get("goal") or request["goal"]).strip()
    if not goal:
        raise ModelBatchError(f"MODEL_BATCH_GOAL_INVALID:{model_id}")
    try:
        budget = parse_gpu_budget(raw.get("budget", request.get("budget", "1gpu-hour")))
    except AdapterProfileError as exc:
        raise ModelBatchError(f"MODEL_BATCH_BUDGET_INVALID:{model_id}") from exc
    metrics = raw.get("target_metrics", request.get("target_metrics", []))
    if not isinstance(metrics, list) or any(not isinstance(value, str) or not value.strip() for value in metrics):
        raise ModelBatchError(f"MODEL_BATCH_TARGET_METRICS_INVALID:{model_id}")
    files: dict[str, str | None] = {"model": str(model), "data": str(data)}
    file_sha256: dict[str, str | None] = {"model": _sha256_path(model), "data": _sha256_path(data)}
    blockers: list[str] = []
    for field in ("adapter_profile", "runtime_python", "evaluator_contract", "instance_config", "dataset_freeze"):
        raw_value = raw.get(field)
        if raw_value is None:
            files[field] = None
            file_sha256[field] = None
            continue
        candidate = Path(str(raw_value)).expanduser().resolve()
        files[field] = str(candidate)
        if not candidate.is_file() or candidate.is_symlink():
            blockers.append(f"{field}_not_found")
        elif field == "runtime_python" and not os.access(candidate, os.X_OK):
            blockers.append("runtime_python_not_executable")
        else:
            file_sha256[field] = _sha256_path(candidate)
    assets = raw.get("assets", {})
    if not isinstance(assets, Mapping):
        raise ModelBatchError(f"MODEL_BATCH_ASSETS_INVALID:{model_id}")
    normalized_assets = {str(key): str(value) for key, value in sorted(assets.items())}
    for parameter, value in list(normalized_assets.items()):
        if not parameter.startswith("--"):
            raise ModelBatchError(f"MODEL_BATCH_ASSET_PARAMETER_INVALID:{model_id}:{parameter}")
        candidate = Path(value).expanduser().resolve()
        if not candidate.exists() or candidate.is_symlink():
            blockers.append(f"asset_not_found:{parameter}")
        normalized_assets[parameter] = str(candidate)
    if raw.get("evaluator_contract") is None:
        blockers.append("evaluator_contract_required")
    digest_body = {
        "model_id": model_id, "model": str(model), "data": str(data), "goal": goal,
        "budget": budget, "adapter": raw.get("adapter"), "assets": normalized_assets,
        "files": files, "file_sha256": file_sha256, "source_revision": raw.get("source_revision"), "target_metrics": list(metrics),
    }
    return {
        **digest_body,
        "state": "ready" if not blockers else "blocked",
        "blockers": blockers,
        "model_digest": _digest(digest_body),
    }


def _required_path(value: object, *, directory: bool, code: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ModelBatchError(code)
    path = Path(value).expanduser().resolve()
    if (path.is_dir() if directory else path.exists()) is False or path.is_symlink():
        raise ModelBatchError(code)
    return path


def _sha256_path(path: Path) -> str:
    """Hash a file, or a deterministic directory inventory for roots."""
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    if not path.is_dir():
        return digest.hexdigest()
    for child in sorted(path.rglob("*")):
        if child.is_symlink() or not child.is_file():
            continue
        relative = child.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with child.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _write_plan(destination: Path, *, request_bytes: bytes, plan: Mapping[str, object]) -> None:
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_dir():
            raise ModelBatchError("MODEL_BATCH_OUTPUT_INVALID")
        existing = destination / "plan.json"
        if existing.is_file() and not existing.is_symlink():
            try:
                loaded = json.loads(existing.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ModelBatchError("MODEL_BATCH_OUTPUT_INVALID") from exc
            if isinstance(loaded, Mapping) and loaded.get("plan_sha256") == plan["plan_sha256"]:
                return
        if any(destination.iterdir()):
            raise ModelBatchError("MODEL_BATCH_OUTPUT_BOUND")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        (temporary / "request.json").write_bytes(request_bytes)
        (temporary / "plan.json").write_text(json.dumps(plan, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        if destination.exists() and destination.is_dir() and not any(destination.iterdir()):
            destination.rmdir()
        os.replace(temporary, destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _digest(value: Mapping[str, object]) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
