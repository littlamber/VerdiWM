"""Compile a static batch plan for heterogeneous model campaigns."""

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


class ModelBatchError(ValueError):
    """A model batch request or plan failed closed."""


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


def run_model_batch(
    *,
    plan_path: Path,
    queue_only: bool = False,
    max_parallel: int = 1,
    output_root: Path | None = None,
    root: Path | None = None,
) -> dict[str, object]:
    """Materialize a validated batch plan into normal campaigns and dispatch them.

    The batch layer owns orchestration only.  It compiles each row through the
    same adapter resolver used by ``verdiwm run``, binds all rows to one budget
    ledger and shared Archive/CAS roots, then confirms campaigns through
    :class:`CampaignStore`.  No alternate worker or verifier path is created.
    """

    if isinstance(max_parallel, bool) or max_parallel < 1 or max_parallel > 256:
        raise ModelBatchError("MODEL_BATCH_MAX_PARALLEL_INVALID")
    repo = (root or Path(__file__).resolve().parents[2]).expanduser().resolve()
    plan = load_model_batch_plan(plan_path, root=repo)
    destination = Path(output_root or Path(plan_path).expanduser().resolve().parent).expanduser().resolve()
    if destination.is_symlink() or not destination.is_dir():
        raise ModelBatchError("MODEL_BATCH_OUTPUT_INVALID")
    batch_id = str(plan["batch_id"])
    if plan.get("state") != "ready_for_dispatch":
        rows = [
            {
                "model_id": row.get("model_id"),
                "model_digest": row.get("model_digest"),
                "state": "blocked",
                "blockers": list(row.get("blockers", [])) if isinstance(row.get("blockers"), list) else ["MODEL_BATCH_PLAN_BLOCKED"],
            }
            for row in plan.get("models", [])
            if isinstance(row, Mapping)
        ]
        manifest: dict[str, object] = {
            "schema_version": 1,
            "artifact_type": "verdiwm-model-batch-execution",
            "state": "blocked",
            "batch_id": batch_id,
            "plan_sha256": plan["plan_sha256"],
            "shared_budget": {
                "path": str(destination / "budget.db"),
                "total_gpu_hours": float(plan["budget"]["declared_gpu_hours"]),
            },
            "campaigns_root": str(destination / "campaigns"),
            "rows": rows,
            "summary": {"model_count": len(rows), "campaign_count": 0, "blocked_count": len(rows)},
            "dispatcher": None,
            "claim_boundary": "The batch plan is blocked; no campaign was created or executed.",
        }
        manifest["execution_sha256"] = _digest(manifest)
        _write_execution_manifest(destination / "execution.json", manifest)
        return manifest
    campaigns_root = destination / "campaigns"
    store = _campaign_store(campaigns_root, project_root=repo)
    total_budget = float(plan["budget"]["declared_gpu_hours"])
    shared_budget = destination / "budget.db"
    rows: list[dict[str, object]] = []
    campaign_ids: list[str] = []
    for raw in plan["models"]:
        if not isinstance(raw, Mapping):
            raise ModelBatchError("MODEL_BATCH_PLAN_INVALID")
        model_id = str(raw["model_id"])
        row_result: dict[str, object] = {
            "model_id": model_id,
            "model_digest": raw.get("model_digest"),
            "state": "blocked" if raw.get("state") != "ready" else "pending",
            "blockers": list(raw.get("blockers", [])) if isinstance(raw.get("blockers"), list) else [],
        }
        if raw.get("state") != "ready":
            rows.append(row_result)
            continue
        campaign_id = _batch_campaign_id(batch_id, model_id)
        row_result["campaign_id"] = campaign_id
        try:
            payload = _compile_batch_payload(
                raw,
                campaign_id=campaign_id,
                batch_id=batch_id,
                campaigns_root=campaigns_root,
                shared_budget=shared_budget,
                total_budget=total_budget,
                project_root=repo,
            )
            created = store.create(payload)
            queued = store.confirm(campaign_id)
            row_result["state"] = "queued"
            row_result["campaign"] = {
                "campaign_id": queued.get("campaign_id"),
                "status": queued.get("status"),
                "revision_id": queued.get("revision_id"),
            }
            campaign_ids.append(campaign_id)
        except Exception as exc:
            row_result["state"] = "blocked"
            row_result.setdefault("blockers", []).append(str(exc).split(":", 1)[0])
            row_result["error"] = {"type": type(exc).__name__, "message": str(exc)[:500]}
        rows.append(row_result)

    dispatch_result: dict[str, object] | None = None
    if campaign_ids and not queue_only:
        from wmloop.control.campaign_dispatcher import DispatcherOptions, run_dispatcher

        dispatch_result = run_dispatcher(
            DispatcherOptions(
                state_root=campaigns_root,
                max_cycles=1,
                max_parallel=max_parallel,
                campaign_ids=tuple(campaign_ids),
            )
        )
        for row in rows:
            campaign_id = row.get("campaign_id")
            if not isinstance(campaign_id, str):
                continue
            try:
                campaign = store.get(campaign_id)
            except Exception:
                continue
            row["campaign"] = {
                "campaign_id": campaign.get("campaign_id"),
                "status": campaign.get("status"),
                "revision_id": campaign.get("revision_id"),
            }
            row["state"] = str(campaign.get("status", row.get("state")))

    ready_count = sum(1 for row in rows if row.get("state") in {"queued", "running", "completed", "failed", "cancelled"} and row.get("campaign_id"))
    blocked_count = sum(1 for row in rows if row.get("state") == "blocked")
    if not campaign_ids:
        state = "blocked"
    elif queue_only:
        state = "partial" if blocked_count else "queued"
    else:
        terminal = {"completed", "failed", "blocked", "cancelled"}
        states = {str(row.get("state")) for row in rows if row.get("campaign_id")}
        state = "partial" if blocked_count or not states.issubset({"completed"}) else "completed"
        if states and states.issubset(terminal) and "completed" not in states and not blocked_count:
            state = "failed"
    manifest: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-model-batch-execution",
        "state": state,
        "batch_id": batch_id,
        "plan_sha256": plan["plan_sha256"],
        "shared_budget": {"path": str(shared_budget), "total_gpu_hours": total_budget},
        "campaigns_root": str(campaigns_root),
        "rows": rows,
        "summary": {
            "model_count": len(rows),
            "campaign_count": ready_count,
            "blocked_count": blocked_count,
        },
        "dispatcher": dispatch_result,
        "claim_boundary": (
            "This manifest records campaign orchestration and execution status. "
            "Model-quality claims require settled receipts under each frozen evaluator."
        ),
    }
    manifest["execution_sha256"] = _digest(manifest)
    _write_execution_manifest(destination / "execution.json", manifest)
    return manifest


def load_model_batch_execution(
    path: Path, *, root: Path | None = None
) -> dict[str, object]:
    """Load a batch execution manifest and verify its immutable digest."""

    repo = (root or Path(__file__).resolve().parents[2]).expanduser().resolve()
    source = Path(path).expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise ModelBatchError("MODEL_BATCH_EXECUTION_NOT_FOUND")
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelBatchError("MODEL_BATCH_EXECUTION_INVALID") from exc
    if not isinstance(document, dict):
        raise ModelBatchError("MODEL_BATCH_EXECUTION_INVALID")
    try:
        validate_document("model_batch_execution", document, root=repo)
    except ContractValidationError as exc:
        raise ModelBatchError(f"MODEL_BATCH_EXECUTION_INVALID:{exc}") from exc
    received = document.get("execution_sha256")
    body = dict(document)
    body.pop("execution_sha256", None)
    if received != _digest(body):
        raise ModelBatchError("MODEL_BATCH_EXECUTION_DIGEST_MISMATCH")
    return document


def build_model_batch_execution_binding(
    path: Path, *, root: Path | None = None
) -> dict[str, object]:
    """Return the path-free batch identity allowed to cross publication boundaries."""

    payload = load_model_batch_execution(path, root=root)
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ModelBatchError("MODEL_BATCH_EXECUTION_ROWS_INVALID")
    model_ids: list[str] = []
    for row in rows:
        if (
            not isinstance(row, Mapping)
            or not isinstance(row.get("model_id"), str)
            or not row["model_id"]
        ):
            raise ModelBatchError("MODEL_BATCH_EXECUTION_ROWS_INVALID")
        model_ids.append(str(row["model_id"]))
    if len(set(model_ids)) != len(model_ids):
        raise ModelBatchError("MODEL_BATCH_EXECUTION_MODEL_IDS_INVALID")
    summary = payload.get("summary")
    if not isinstance(summary, Mapping) or summary.get("model_count") != len(model_ids):
        raise ModelBatchError("MODEL_BATCH_EXECUTION_SUMMARY_INVALID")
    return {
        "artifact_type": "verdiwm-model-batch-execution",
        "batch_id": payload["batch_id"],
        "execution_sha256": payload["execution_sha256"],
        "plan_sha256": payload["plan_sha256"],
        "state": payload["state"],
        "model_ids": sorted(model_ids),
    }


def summarize_model_batch(
    execution_path: Path, *, root: Path | None = None
) -> dict[str, object]:
    """Read current campaign states for a batch without dispatching work."""

    execution = load_model_batch_execution(execution_path, root=root)
    rows = execution.get("rows")
    if not isinstance(rows, list):
        raise ModelBatchError("MODEL_BATCH_EXECUTION_ROWS_INVALID")
    campaigns_root = Path(str(execution.get("campaigns_root"))).expanduser().resolve()
    store = None
    if campaigns_root.is_symlink():
        raise ModelBatchError("MODEL_BATCH_CAMPAIGNS_ROOT_INVALID")
    if campaigns_root.exists() and not campaigns_root.is_dir():
        raise ModelBatchError("MODEL_BATCH_CAMPAIGNS_ROOT_INVALID")
    if campaigns_root.is_dir():
        store = _campaign_store(campaigns_root, project_root=(root or Path(__file__).resolve().parents[2]).resolve())
    current_rows: list[dict[str, object]] = []
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise ModelBatchError("MODEL_BATCH_EXECUTION_ROWS_INVALID")
        row = dict(raw)
        campaign_id = row.get("campaign_id")
        if isinstance(campaign_id, str) and store is not None:
            try:
                campaign = store.get(campaign_id)
            except Exception:
                campaign = None
            if isinstance(campaign, Mapping):
                row["state"] = campaign.get("status", row.get("state"))
                row["campaign"] = {
                    "campaign_id": campaign.get("campaign_id"),
                    "status": campaign.get("status"),
                    "revision_id": campaign.get("revision_id"),
                }
        current_rows.append(row)
    state_counts: dict[str, int] = {}
    for row in current_rows:
        state = str(row.get("state", "unknown"))
        state_counts[state] = state_counts.get(state, 0) + 1
    observed = set(state_counts)
    if observed and observed.issubset({"completed"}):
        current_state = "completed"
    elif observed and observed.issubset({"failed", "cancelled"}):
        current_state = "failed"
    elif "running" in observed or "queued" in observed:
        current_state = "queued"
    elif "blocked" in observed and len(observed) == 1:
        current_state = "blocked"
    else:
        current_state = "partial"
    report = {
        "schema_version": 1,
        "artifact_type": "verdiwm-model-batch-status",
        "state": current_state,
        "execution_state": execution.get("state"),
        "batch_id": execution.get("batch_id"),
        "plan_sha256": execution.get("plan_sha256"),
        "execution_sha256": execution.get("execution_sha256"),
        "campaigns_root": str(campaigns_root),
        "rows": current_rows,
        "state_counts": dict(sorted(state_counts.items())),
        "claim_boundary": (
            "This is a read-only status projection. It does not dispatch, cancel, "
            "evaluate, settle, or create a model-quality claim."
        ),
    }
    try:
        validate_document(
            "model_batch_status",
            report,
            root=(root or Path(__file__).resolve().parents[2]).resolve(),
        )
    except ContractValidationError as exc:
        raise ModelBatchError(f"MODEL_BATCH_STATUS_INVALID:{exc}") from exc
    return report


def _campaign_store(path: Path, *, project_root: Path):
    from wmloop.control.campaign_api import CampaignStore

    return CampaignStore(path, project_root=project_root)


def _compile_batch_payload(
    raw: Mapping[str, object],
    *,
    campaign_id: str,
    batch_id: str,
    campaigns_root: Path,
    shared_budget: Path,
    total_budget: float,
    project_root: Path,
) -> dict[str, object]:
    from wmloop.control.adapter_profiles import compile_adapter_execution

    files = raw.get("files")
    if not isinstance(files, Mapping):
        raise ModelBatchError("MODEL_BATCH_PLAN_INVALID")
    model = Path(str(files["model"])).resolve()
    data = Path(str(files["data"])).resolve()
    profile = files.get("adapter_profile")
    runtime = files.get("runtime_python")
    evaluator = files.get("evaluator_contract")
    if not isinstance(evaluator, str) or not evaluator:
        raise ModelBatchError("EVALUATOR_CONTRACT_REQUIRED")
    resolved = compile_adapter_execution(
        campaign_id=campaign_id,
        model=model,
        data=data,
        goal=str(raw["goal"]),
        budget=float(raw["budget"]),
        campaign_root=campaigns_root,
        adapter=(str(raw["adapter"]) if raw.get("adapter") else None),
        adapter_profile_path=Path(str(profile)).resolve() if isinstance(profile, str) and profile else None,
        runtime_python=Path(str(runtime)).resolve() if isinstance(runtime, str) and runtime else None,
        asset_overrides=(raw.get("assets") if isinstance(raw.get("assets"), Mapping) else None),
        project_root=project_root,
    )
    execution = dict(resolved.execution)
    evaluator_path = str(Path(evaluator).resolve())
    if str(execution.get("evaluator_contract", "")) != evaluator_path:
        raise ModelBatchError("EVALUATOR_PROFILE_MISMATCH")
    execution["evaluator_contract"] = evaluator_path
    execution["budget_db"] = str(shared_budget)
    execution["budget_total_gpu_hours"] = total_budget
    execution["shared_budget_binding"] = True
    payload: dict[str, object] = {
        "campaign_id": campaign_id,
        "goal": str(raw["goal"]),
        "model": str(model),
        "dataset": str(data),
        "budget": float(raw["budget"]),
        "adapter": resolved.profile_id,
        "adapter_profile_path": str(Path(str(profile)).resolve()) if isinstance(profile, str) and profile else None,
        "runtime_python": str(Path(str(runtime)).resolve()) if isinstance(runtime, str) and runtime else None,
        "assets": dict(raw.get("assets", {})) if isinstance(raw.get("assets"), Mapping) else {},
        "execution": execution,
        "batch_id": batch_id,
    }
    metrics = list(raw.get("target_metrics", []))
    if metrics:
        payload["target_metrics"] = metrics
    return payload


def _write_execution_manifest(path: Path, manifest: Mapping[str, object]) -> None:
    try:
        validate_document("model_batch_execution", manifest)
    except ContractValidationError as exc:
        raise ModelBatchError(f"MODEL_BATCH_EXECUTION_INVALID:{exc}") from exc
    existing = path if path.is_file() and not path.is_symlink() else None
    if existing is not None:
        try:
            prior = json.loads(existing.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelBatchError("MODEL_BATCH_EXECUTION_OUTPUT_INVALID") from exc
        if isinstance(prior, Mapping) and prior.get("plan_sha256") == manifest.get("plan_sha256"):
            # The current manifest is a fresh status projection; replacing it is
            # safe because campaign records and receipts remain authoritative.
            pass
        else:
            raise ModelBatchError("MODEL_BATCH_EXECUTION_PLAN_CONFLICT")
    temporary = path.with_name(f".{path.name}.tmp")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
