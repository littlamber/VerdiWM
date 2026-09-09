"""Model batch run implementation."""

from __future__ import annotations
import json
from functools import wraps
from collections.abc import Mapping
from pathlib import Path
from wmloop.contracts import ContractValidationError, validate_document
from wmloop.control.batch_state import ModelBatchError, admit_batch, aggregate_states
from wmloop.storage import atomic_write, canonical_bytes, checked_path, exclusive_file_lock
from .model_batch_plan import _batch_campaign_id, _digest, load_model_batch_plan


def _locked_batch(function):
    @wraps(function)
    def run(*, plan_path, output_root=None, **kwargs):
        destination = checked_path(output_root or Path(plan_path).expanduser().absolute().parent, code="MODEL_BATCH_OUTPUT_INVALID", error=ModelBatchError)
        if not destination.is_dir():
            raise ModelBatchError("MODEL_BATCH_OUTPUT_INVALID")
        with exclusive_file_lock(destination / ".batch.lock"):
            plan = load_model_batch_plan(plan_path, root=kwargs.get("root"))
            admit_batch(destination, plan)
            return function(plan_path=plan_path, output_root=destination, **kwargs)
    return run


@_locked_batch
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

    if isinstance(max_parallel, bool) or not isinstance(max_parallel, int) or max_parallel < 1 or max_parallel > 256:
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
            if created["status"] in {"running", "completed", "failed", "cancelled", "blocked"}:
                row_result["state"] = created["status"]
                row_result["campaign"] = {
                    "campaign_id": campaign_id,
                    "status": created["status"],
                    "revision_id": created.get("revision_id"),
                }
                rows.append(row_result)
                continue
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
    _save_execution_snapshot(rows=rows, batch_id=batch_id, plan=plan, shared_budget=shared_budget,
                             total_budget=total_budget, campaigns_root=campaigns_root,
                             dispatch_result=None, destination=destination)
    if campaign_ids and not queue_only:
        from wmloop.control.campaign_dispatcher import DispatcherOptions, run_dispatcher

        dispatch_result = run_dispatcher(
            DispatcherOptions(
                state_root=campaigns_root,
                max_cycles=(len(campaign_ids) + max_parallel - 1) // max_parallel,
                poll_seconds=0,
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
            except Exception as exc:
                row["state"] = "unavailable"
                row["state_error"] = {"code": str(exc).split(":", 1)[0]}
                continue
            row["campaign"] = {
                "campaign_id": campaign.get("campaign_id"),
                "status": campaign.get("status"),
                "revision_id": campaign.get("revision_id"),
            }
            row["state"] = str(campaign.get("status", row.get("state")))

    return _save_execution_snapshot(rows=rows, batch_id=batch_id, plan=plan, shared_budget=shared_budget,
                                    total_budget=total_budget, campaigns_root=campaigns_root,
                                    dispatch_result=dispatch_result, destination=destination)


def _save_execution_snapshot(*, rows, batch_id, plan, shared_budget, total_budget, campaigns_root, dispatch_result, destination):
    ready_count = sum(1 for row in rows if row.get("state") in {"queued", "running", "completed", "failed", "cancelled"} and row.get("campaign_id"))
    blocked_count = sum(1 for row in rows if row.get("state") == "blocked")
    state = aggregate_states(row["state"] for row in rows)
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
    atomic_write(path, canonical_bytes(manifest) + b"\n")
