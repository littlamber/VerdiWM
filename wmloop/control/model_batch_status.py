"""Model batch status implementation."""

from __future__ import annotations
import json
from collections.abc import Mapping
from pathlib import Path
from wmloop.contracts import ContractValidationError, validate_document
from wmloop.control.batch_state import ModelBatchError, aggregate_states, validate_model_ids
from wmloop.storage import checked_path
from .model_batch_plan import _digest


def load_model_batch_execution(
    path: Path, *, root: Path | None = None
) -> dict[str, object]:
    """Load a batch execution manifest and verify its immutable digest."""

    repo = (root or Path(__file__).resolve().parents[2]).expanduser().resolve()
    source = checked_path(path, code="MODEL_BATCH_EXECUTION_NOT_FOUND", error=ModelBatchError)
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
    model_ids = validate_model_ids(model_ids)
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
    campaigns_root = checked_path(Path(str(execution.get("campaigns_root"))), code="MODEL_BATCH_CAMPAIGNS_ROOT_INVALID", error=ModelBatchError)
    store = None
    if campaigns_root.is_symlink():
        raise ModelBatchError("MODEL_BATCH_CAMPAIGNS_ROOT_INVALID")
    if campaigns_root.exists() and not campaigns_root.is_dir():
        raise ModelBatchError("MODEL_BATCH_CAMPAIGNS_ROOT_INVALID")
    if campaigns_root.is_dir():
        from wmloop.control.campaign_api import CampaignStore
        store = CampaignStore(campaigns_root, project_root=(root or Path(__file__).resolve().parents[2]).resolve(), read_only=True)
    current_rows: list[dict[str, object]] = []
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise ModelBatchError("MODEL_BATCH_EXECUTION_ROWS_INVALID")
        row = dict(raw)
        campaign_id = row.get("campaign_id")
        if isinstance(campaign_id, str):
            try:
                if store is None:
                    raise ModelBatchError("CAMPAIGN_STORE_UNAVAILABLE")
                campaign = store.get(campaign_id)
                prior = row.get("campaign", {})
                if isinstance(prior, Mapping) and prior.get("revision_id") is not None and prior["revision_id"] != campaign.get("revision_id"):
                    raise ModelBatchError("CAMPAIGN_REVISION_MISMATCH")
            except Exception as exc:
                campaign = None
                row["last_known_state"] = row.get("state")
                row["state"] = "unavailable"
                row["state_error"] = {"code": str(exc).split(":", 1)[0]}
                row.pop("campaign", None)
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
    current_state = aggregate_states(state_counts)
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
