"""Small, dependency-free Campaign API for the VerdiWM control plane.

The API stores request state as JSON documents and deliberately does not run
experiments itself.  Execution remains owned by the existing pipeline/daemon;
this boundary only accepts, validates, and durably records user intent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import parse_qs, urlparse

from wmloop.control.adapter_profiles import (
    AdapterProfileError,
    compile_adapter_execution,
    parse_gpu_budget,
)
from wmloop.control.model_run import ModelRunError, compile_model_run
from wmloop.control.research_modes import (
    ResearchModeError,
    apply_research_mode_to_execution,
    compile_research_mode_plan,
)
from wmloop.control.automatic_campaign import (
    AutomaticCampaignError,
    automatic_campaign_id,
    campaign_key,
    compile_revision,
    isolate_execution_for_revision,
)
from wmloop.experiments.evidence_graph import query_evidence_graph


SCHEMA_VERSION = 4
_TRANSITIONS = {
    "created": {"confirmed", "queued", "cancelled"},
    "confirmed": {"queued", "cancelled"},
    "queued": {"running", "cancelled", "failed"},
    "running": {"completed", "blocked", "failed", "cancelled"},
    "cancelled": set(),
    "completed": set(),
    "blocked": set(),
    "failed": set(),
}


class CampaignAPIError(ValueError):
    """Stable client-facing validation failure."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


class CampaignStore:
    """Atomic JSON-backed campaign store suitable for a single API instance."""

    def __init__(self, root: Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _path(self, campaign_id: str) -> Path:
        if not campaign_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in campaign_id):
            raise CampaignAPIError("CAMPAIGN_ID_INVALID")
        return self.root / f"{campaign_id}.json"

    def _revision_path(self, campaign_id: str, revision_id: str) -> Path:
        if not revision_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in revision_id):
            raise CampaignAPIError("REVISION_ID_INVALID")
        self._path(campaign_id)
        return self.root / "revisions" / campaign_id / f"{revision_id}.json"

    def _write(self, path: Path, value: dict[str, Any]) -> None:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(value, handle, sort_keys=True, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)

    def create(self, payload: dict[str, Any]) -> dict[str, Any]:
        goal = _required_text(payload, "goal", "GOAL_REQUIRED")
        model = _required_text(payload, "model", "MODEL_REQUIRED")
        dataset = _required_text(payload, "dataset", "DATASET_REQUIRED")
        target_metrics = _target_metrics(payload)
        try:
            budget_hours = parse_gpu_budget(payload.get("budget"))
        except AdapterProfileError as exc:
            raise CampaignAPIError(str(exc)) from exc
        key = campaign_key(
            model=model,
            dataset=dataset,
            goal=goal,
            budget_gpu_hours=budget_hours,
            adapter=(str(payload["adapter"]) if payload.get("adapter") is not None else None),
            target_metrics=target_metrics,
        )
        campaign_id = str(payload.get("campaign_id") or automatic_campaign_id(key))
        self._path(campaign_id)
        request_hash = hashlib.sha256(_canonical(payload).encode()).hexdigest()

        execution = payload.get("execution")
        training_scale_plan = payload.get("training_scale_plan")
        if training_scale_plan is not None and not isinstance(training_scale_plan, Mapping):
            raise CampaignAPIError("TRAINING_SCALE_PLAN_OBJECT_REQUIRED")
        adapter_repair = payload.get("adapter_repair")
        if adapter_repair is not None:
            _validate_adapter_repair_binding(adapter_repair)
        automatically_compiled = execution is None
        adapter_profile = payload.get("adapter")
        model_family = payload.get("model_family")
        capability_level = payload.get("capability_level")
        constitution_freeze = payload.get("constitution_freeze")
        if execution is None:
            assets = payload.get("assets")
            if assets is not None and not isinstance(assets, dict):
                raise CampaignAPIError("ASSET_OVERRIDES_OBJECT_REQUIRED")
            try:
                resolved = compile_adapter_execution(
                    campaign_id=campaign_id,
                    model=Path(model),
                    data=Path(dataset),
                    goal=goal,
                    budget=budget_hours,
                    campaign_root=self.root,
                    adapter=(
                        str(payload["adapter"])
                        if payload.get("adapter") is not None
                        else None
                    ),
                    adapter_profile_path=(
                        Path(str(payload["adapter_profile_path"]))
                        if payload.get("adapter_profile_path") is not None
                        else None
                    ),
                    runtime_python=(
                        Path(str(payload["runtime_python"]))
                        if payload.get("runtime_python") is not None
                        else None
                    ),
                    asset_overrides=assets,
                )
            except AdapterProfileError as exc:
                raise CampaignAPIError(str(exc)) from exc
            execution = resolved.execution
            adapter_profile = resolved.profile_id
            model_family = resolved.model_family
            capability_level = resolved.capability_level
            constitution_freeze = resolved.constitution_freeze
        if training_scale_plan is not None:
            assert isinstance(execution, dict)
            execution = dict(execution)
            # The scale receipt becomes part of the immutable campaign revision,
            # rather than an after-the-fact launch flag.
            execution["training_scale_plan"] = dict(training_scale_plan)
        research_mode_plan: dict[str, object] | None = None
        if payload.get("research_mode") is not None:
            assert isinstance(execution, dict)
            execution = dict(execution)
            for field in (
                "literature_query",
                "cpbe_request",
                "cpbe_history",
            ):
                if payload.get(field) is not None:
                    execution[field] = payload[field]
            try:
                research_mode_plan = compile_research_mode_plan(
                    mode=payload["research_mode"],
                    goal=goal,
                    execution=execution,
                    evidence_context=(
                        payload.get("evidence_context")
                        if isinstance(payload.get("evidence_context"), Mapping)
                        else None
                    ),
                )
            except ResearchModeError as exc:
                raise CampaignAPIError(str(exc)) from exc
            if research_mode_plan["state"] == "blocked":
                raise CampaignAPIError(
                    "RESEARCH_MODE_PREREQUISITES_MISSING:"
                    + ",".join(str(value) for value in research_mode_plan["blockers"])
                )
            execution = apply_research_mode_to_execution(
                execution,
                plan=research_mode_plan,
            )
        _validate_execution(execution)
        assert isinstance(execution, dict)
        target_metrics = _bind_target_metrics(target_metrics, execution, goal=goal)
        if target_metrics:
            execution = dict(execution)
            execution["target_metrics"] = target_metrics
        try:
            revision = compile_revision(
                campaign_id=campaign_id,
                campaign_key_value=key,
                goal=goal,
                model=model,
                dataset=dataset,
                budget_gpu_hours=budget_hours,
                adapter_profile=(str(adapter_profile) if adapter_profile is not None else None),
                constitution_freeze=(str(constitution_freeze) if constitution_freeze is not None else None),
                execution=execution,
                resource_request=(
                    payload.get("resource_request")
                    if isinstance(payload.get("resource_request"), Mapping)
                    else None
                ),
                target_metrics=target_metrics,
            )
        except AutomaticCampaignError as exc:
            raise CampaignAPIError(str(exc)) from exc
        if payload.get("resource_request") is not None and not isinstance(payload.get("resource_request"), Mapping):
            raise CampaignAPIError("RESOURCE_REQUEST_OBJECT_REQUIRED")
        if automatically_compiled:
            execution = isolate_execution_for_revision(
                execution, revision_id=str(revision["revision_id"])
            )
        model_run: dict[str, object] | None = None
        if training_scale_plan is not None:
            try:
                model_run = compile_model_run(
                    campaign_id=campaign_id,
                    execution=execution,
                    model=Path(model),
                    data=Path(dataset),
                    training_scale_plan=training_scale_plan,
                    resource_allocation=revision["resource_allocation"],
                )
            except ModelRunError as exc:
                raise CampaignAPIError(str(exc)) from exc
        record = {
            "schema_version": SCHEMA_VERSION,
            "campaign_id": campaign_id,
            "campaign_key": key,
            "revision_id": revision["revision_id"],
            "revision_digest": revision["revision_digest"],
            "revision": revision,
            "resource_allocation": revision["resource_allocation"],
            "goal": goal,
            "target_metrics": target_metrics,
            "model": model,
            "dataset": dataset,
            "budget": {"gpu_hours": budget_hours},
            "adapter_profile": adapter_profile,
            "model_family": model_family,
            "capability_level": capability_level,
            "constitution_freeze": constitution_freeze,
            "execution": execution,
            "status": "created",
            "created_at": _now(),
            "updated_at": _now(),
            "request_hash": request_hash,
        }
        if model_run is not None:
            record["model_run"] = model_run
        if research_mode_plan is not None:
            record["research_mode"] = research_mode_plan["mode"]
            record["research_mode_plan"] = research_mode_plan
        for field in ("engineering_manifest", "compiled_manifest", "adapter_repair"):
            if payload.get(field) is not None:
                record[field] = payload[field]
        with self._lock:
            path = self._path(campaign_id)
            if path.exists():
                existing = json.loads(path.read_text(encoding="utf-8"))
                if existing.get("request_hash") != record["request_hash"]:
                    raise CampaignAPIError("CAMPAIGN_ID_CONFLICT")
                if existing.get("revision_id") == record["revision_id"]:
                    return existing
                # A draft has no dispatcher claim or GPU side effect yet, so a changed
                # source/policy snapshot may safely supersede it automatically.
                if str(existing.get("status")) not in {"created", "completed", "failed", "cancelled"}:
                    raise CampaignAPIError("CAMPAIGN_REVISION_ACTIVE")
                history = list(existing.get("revision_history", []))
                history.append(
                    {
                        "revision_id": existing.get("revision_id"),
                        "revision_digest": existing.get("revision_digest"),
                        "status": existing.get("status"),
                        "revision_ref": existing.get("revision_ref"),
                    }
                )
                record["revision_history"] = history
                record["prior_revision_id"] = existing.get("revision_id")
            revision_path = self._revision_path(campaign_id, str(record["revision_id"]))
            revision_document = {
                "schema_version": 1,
                "artifact_type": "verdiwm-campaign-revision-record",
                "campaign_id": campaign_id,
                "revision": record["revision"],
                "execution": record["execution"],
                "request_hash": record["request_hash"],
            }
            if model_run is not None:
                revision_document["model_run"] = model_run
            if payload.get("adapter_repair") is not None:
                revision_document["adapter_repair"] = payload["adapter_repair"]
            if revision_path.exists():
                existing_revision = json.loads(revision_path.read_text(encoding="utf-8"))
                if existing_revision != revision_document:
                    raise CampaignAPIError("CAMPAIGN_REVISION_CONFLICT")
            else:
                self._write(revision_path, revision_document)
            record["revision_ref"] = str(revision_path)
            if model_run is not None:
                model_run_path = Path(str(model_run["manifest_path"])).resolve()
                if model_run_path.exists():
                    try:
                        existing_model_run = json.loads(model_run_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError) as exc:
                        raise CampaignAPIError("MODEL_RUN_MANIFEST_CONFLICT") from exc
                    if existing_model_run != model_run:
                        raise CampaignAPIError("MODEL_RUN_MANIFEST_CONFLICT")
                else:
                    self._write(model_run_path, model_run)
            self._write(path, record)
        return record

    def get(self, campaign_id: str) -> dict[str, Any]:
        path = self._path(campaign_id)
        if not path.is_file():
            raise CampaignAPIError("CAMPAIGN_NOT_FOUND")
        return json.loads(path.read_text(encoding="utf-8"))

    def list(self, *, status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        if limit < 1 or limit > 1000:
            raise CampaignAPIError("CAMPAIGN_LIMIT_INVALID")
        if status is not None and status not in _TRANSITIONS:
            raise CampaignAPIError("STATUS_INVALID")
        rows: list[dict[str, Any]] = []
        for path in sorted(self.root.glob("*.json")):
            if path.is_symlink() or not path.is_file():
                continue
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(record, dict) and (status is None or record.get("status") == status):
                rows.append(record)
        return rows[:limit]

    def transition(self, campaign_id: str, status: str) -> dict[str, Any]:
        if status not in _TRANSITIONS:
            raise CampaignAPIError("STATUS_INVALID")
        with self._lock:
            record = self.get(campaign_id)
            current = str(record.get("status"))
            if status != current and status not in _TRANSITIONS.get(current, set()):
                raise CampaignAPIError("STATUS_TRANSITION_INVALID")
            record["status"] = status
            record["updated_at"] = _now()
            self._write(self._path(campaign_id), record)
            return record

    def confirm(self, campaign_id: str) -> dict[str, Any]:
        """Confirm intent and enqueue an execution contract when present."""

        with self._lock:
            record = self.get(campaign_id)
            execution = record.get("execution")
            if execution is None:
                raise CampaignAPIError("EXECUTION_REQUIRED")
            _validate_execution(execution)
            current = str(record.get("status"))
            if current == "queued":
                return record
            if current not in {"created", "confirmed"}:
                raise CampaignAPIError("STATUS_TRANSITION_INVALID")
            dispatch = {
                "schema_version": 1,
                "artifact_type": "verdiwm-campaign-dispatch",
                "campaign_id": campaign_id,
                "revision_id": record.get("revision_id"),
                "revision_digest": record.get("revision_digest"),
                "resource_allocation": record.get("resource_allocation"),
                "state": "pending",
                "execution": execution,
                "execution_hash": hashlib.sha256(
                    _canonical(
                        {
                            "execution": execution,
                            "model_run": record.get("model_run"),
                            "revision_id": record.get("revision_id"),
                            "resource_allocation": record.get("resource_allocation"),
                        }
                    ).encode()
                ).hexdigest(),
                "created_at": _now(),
            }
            if record.get("model_run") is not None:
                dispatch["model_run"] = record["model_run"]
            for field in ("engineering_manifest", "compiled_manifest"):
                if record.get(field) is not None:
                    dispatch[field] = record[field]
            dispatch_path = self.root / "dispatch" / "pending" / f"{campaign_id}.json"
            if dispatch_path.exists():
                existing = json.loads(dispatch_path.read_text(encoding="utf-8"))
                if existing.get("execution_hash") != dispatch["execution_hash"]:
                    raise CampaignAPIError("DISPATCH_CONFLICT")
            else:
                self._write(dispatch_path, dispatch)
            record["status"] = "queued"
            record["dispatch_ref"] = str(dispatch_path)
            record["updated_at"] = _now()
            self._write(self._path(campaign_id), record)
            return record

    def cancel(self, campaign_id: str) -> dict[str, Any]:
        """Request cancellation and withdraw an unclaimed dispatch immediately."""

        with self._lock:
            record = self.get(campaign_id)
            current = str(record.get("status"))
            if current == "cancelled":
                return record
            if "cancelled" not in _TRANSITIONS.get(current, set()):
                raise CampaignAPIError("STATUS_TRANSITION_INVALID")
            record["status"] = "cancelled"
            record["cancellation_requested_at"] = _now()
            record["updated_at"] = _now()
            pending = self.root / "dispatch" / "pending" / f"{campaign_id}.json"
            if pending.is_file() and not pending.is_symlink():
                try:
                    dispatch = json.loads(pending.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    dispatch = {"campaign_id": campaign_id, "schema_version": 1}
                dispatch["state"] = "cancelled"
                dispatch["cancelled_at"] = record["cancellation_requested_at"]
                cancelled_path = self.root / "dispatch" / "cancelled" / pending.name
                self._write(cancelled_path, dispatch)
                pending.unlink(missing_ok=True)
                record["dispatch_ref"] = str(cancelled_path.resolve())
            self._write(self._path(campaign_id), record)
            return record

    def record_dispatch_result(
        self,
        campaign_id: str,
        *,
        status: str,
        result: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if status not in {"running", "completed", "blocked", "failed"}:
            raise CampaignAPIError("DISPATCH_STATUS_INVALID")
        with self._lock:
            record = self.get(campaign_id)
            current = str(record.get("status"))
            if current == "cancelled":
                return record
            if status != current and status not in _TRANSITIONS.get(current, set()):
                raise CampaignAPIError("STATUS_TRANSITION_INVALID")
            record["status"] = status
            record["updated_at"] = _now()
            if result is not None:
                record["execution_result"] = result
                record["execution_result_hash"] = hashlib.sha256(
                    _canonical(result).encode()
                ).hexdigest()
            if error is not None:
                record["execution_error"] = error
            self._write(self._path(campaign_id), record)
            return record

    def record_dispatch_location(
        self, campaign_id: str, dispatch_path: Path
    ) -> dict[str, Any]:
        """Bind campaign status to the dispatch manifest's current location."""

        with self._lock:
            record = self.get(campaign_id)
            record["dispatch_ref"] = str(Path(dispatch_path).resolve())
            record["updated_at"] = _now()
            self._write(self._path(campaign_id), record)
            return record

    def record_autonomous_deployment(
        self, campaign_id: str, deployment: Mapping[str, object]
    ) -> dict[str, Any]:
        """Bind a non-dispatcher controller to the active immutable revision.

        Ctrl-World's durable controller owns execution itself, so it must not be
        represented as a generic pipeline dispatch.  This marker prevents an
        automatic source revision from replacing a live controller's state.
        """

        deployment_ref = deployment.get("controller_config")
        if not isinstance(deployment_ref, str) or not Path(deployment_ref).is_absolute():
            raise CampaignAPIError("AUTONOMOUS_DEPLOYMENT_REF_INVALID")
        with self._lock:
            record = self.get(campaign_id)
            if str(record.get("status")) not in {"created", "confirmed", "queued", "running"}:
                raise CampaignAPIError("STATUS_TRANSITION_INVALID")
            revision_id = deployment.get("revision_id")
            if revision_id != record.get("revision_id"):
                raise CampaignAPIError("AUTONOMOUS_DEPLOYMENT_REVISION_MISMATCH")
            record["status"] = "running"
            record["autonomous_deployment"] = dict(deployment)
            record["updated_at"] = _now()
            self._write(self._path(campaign_id), record)
            return record

    def reproduce(self, campaign_id: str) -> dict[str, Any]:
        source = self.get(campaign_id)
        if source.get("model_run") is not None:
            # A training scale receipt, output root, and resource allocation are
            # revision-bound. Reusing them under a new campaign id would turn a
            # reproduction request into an unsafe alias of the original run.
            raise CampaignAPIError("MODEL_RUN_REPRODUCTION_REQUIRES_NEW_REVISION")
        execution = source.get("execution")
        if not isinstance(execution, dict):
            raise CampaignAPIError("REPRODUCE_EXECUTION_REQUIRED")
        child = dict(source)
        child_id = f"{campaign_id}-repro-{uuid.uuid4().hex[:8]}"
        for field in (
            "dispatch_ref",
            "execution_result",
            "execution_result_hash",
            "execution_error",
            "cancellation_requested_at",
        ):
            child.pop(field, None)
        child_execution = _reproduction_execution(execution, campaign_id=child_id)
        child.update(
            {
                "campaign_id": child_id,
                "execution": child_execution,
                "status": "created",
                "parent_campaign_id": campaign_id,
                "created_at": _now(),
                "updated_at": _now(),
            }
        )
        child["request_hash"] = hashlib.sha256(
            _canonical(
                {
                    "parent_campaign_id": campaign_id,
                    "execution": child_execution,
                    "goal": child.get("goal"),
                    "model": child.get("model"),
                    "dataset": child.get("dataset"),
                    "budget": child.get("budget"),
                }
            ).encode()
        ).hexdigest()
        with self._lock:
            self._write(self._path(child_id), child)
        return self.confirm(child_id)


def _required_text(payload: Mapping[str, Any], name: str, code: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise CampaignAPIError(code)
    return value.strip()


def _target_metrics(payload: Mapping[str, Any]) -> list[str]:
    """Normalize user metric intent without allowing evaluator mutation."""
    raw = payload.get("target_metrics", payload.get("metrics", payload.get("metric")))
    if raw is None:
        return []
    if isinstance(raw, str):
        values = [part.strip() for part in raw.replace(",", " ").split()]
    elif isinstance(raw, (list, tuple)):
        values = [item.strip() for item in raw if isinstance(item, str)]
        if len(values) != len(raw):
            raise CampaignAPIError("TARGET_METRICS_INVALID")
    else:
        raise CampaignAPIError("TARGET_METRICS_INVALID")
    normalized = sorted({value for value in values if value})
    if not normalized:
        raise CampaignAPIError("TARGET_METRICS_INVALID")
    if any(len(value) > 128 for value in normalized):
        raise CampaignAPIError("TARGET_METRICS_INVALID")
    return normalized


def _bind_target_metrics(
    metrics: list[str], execution: Mapping[str, Any], *, goal: str
) -> list[str]:
    explicit = bool(metrics)
    contract_value = execution.get("evaluator_contract")
    if not isinstance(contract_value, str):
        if not explicit:
            return []
        raise CampaignAPIError("TARGET_METRICS_EVALUATOR_UNAVAILABLE")
    try:
        contract = json.loads(Path(contract_value).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        if not explicit:
            return []
        raise CampaignAPIError("TARGET_METRICS_EVALUATOR_UNAVAILABLE") from exc
    catalog = contract.get("metrics") if isinstance(contract, Mapping) else None
    if not isinstance(catalog, list) or not all(isinstance(item, str) for item in catalog):
        raise CampaignAPIError("TARGET_METRICS_CATALOG_INVALID")
    available = {item.casefold(): item for item in catalog}
    aliases = contract.get("metric_aliases", {})
    if not isinstance(aliases, Mapping) or not all(
        isinstance(alias, str) and isinstance(target, str)
        for alias, target in aliases.items()
    ):
        raise CampaignAPIError("TARGET_METRICS_CATALOG_INVALID")
    for alias, target in aliases.items():
        canonical = available.get(target.casefold())
        if canonical is None:
            raise CampaignAPIError("TARGET_METRICS_CATALOG_INVALID")
        available[alias.casefold()] = canonical
    if not metrics:
        metrics = [
            phrase
            for phrase in available
            if _metric_mentioned(phrase, goal)
        ]
        if not metrics:
            return []
    unknown = [item for item in metrics if item.casefold() not in available]
    if unknown:
        raise CampaignAPIError("TARGET_METRIC_UNKNOWN:" + ",".join(unknown))
    return sorted({available[item.casefold()] for item in metrics})


def _metric_mentioned(metric: str, goal: str) -> bool:
    metric_text = metric.casefold()
    goal_text = goal.casefold()
    if any(ord(character) > 127 for character in metric_text):
        return metric_text in goal_text
    return re.search(
        rf"(?<![a-z0-9_]){re.escape(metric_text)}(?![a-z0-9_])", goal_text
    ) is not None


def _validate_adapter_repair_binding(value: object) -> None:
    if not isinstance(value, Mapping):
        raise CampaignAPIError("ADAPTER_REPAIR_BINDING_INVALID")
    path_value = value.get("manifest_path")
    expected_sha256 = value.get("manifest_sha256")
    expected_digest = value.get("input_digest")
    assurance = value.get("assurance_level")
    if (
        not isinstance(path_value, str)
        or not Path(path_value).is_absolute()
        or not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or not isinstance(expected_digest, str)
        or len(expected_digest) != 64
        or assurance not in {"process_guarded_local", "container_isolated"}
    ):
        raise CampaignAPIError("ADAPTER_REPAIR_BINDING_INVALID")
    path = Path(path_value)
    if path.is_symlink() or not path.is_file():
        raise CampaignAPIError("ADAPTER_REPAIR_MANIFEST_INVALID")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise CampaignAPIError("ADAPTER_REPAIR_MANIFEST_HASH_MISMATCH")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignAPIError("ADAPTER_REPAIR_MANIFEST_INVALID") from exc
    authority = manifest.get("authority") if isinstance(manifest, Mapping) else None
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("artifact_type") != "verdiwm-adapter-repair-manifest"
        or manifest.get("state") != "ready"
        or manifest.get("input_digest") != expected_digest
        or manifest.get("assurance_level") != assurance
        or not isinstance(authority, Mapping)
        or authority.get("source_mutated") is not False
        or authority.get("evaluator_mutated") is not False
        or authority.get("promotion") is not False
    ):
        raise CampaignAPIError("ADAPTER_REPAIR_MANIFEST_INVALID")


def _reproduction_execution(
    execution: Mapping[str, Any], *, campaign_id: str
) -> dict[str, Any]:
    reproduced = json.loads(json.dumps(execution))
    for field in ("output_root", "state_root"):
        value = reproduced.get(field)
        if isinstance(value, str) and value:
            reproduced[field] = str(Path(value).parent / campaign_id)
    budget_db = reproduced.get("budget_db")
    if isinstance(budget_db, str) and budget_db:
        reproduced["budget_db"] = str(Path(budget_db).parent / f"{campaign_id}.db")
    _validate_execution(reproduced)
    return reproduced


def _validate_execution(value: object) -> None:
    if not isinstance(value, dict):
        raise CampaignAPIError("EXECUTION_OBJECT_REQUIRED")
    kind = value.get("kind")
    required = {
        "pipeline": (
            "repo_root",
            "output_root",
            "evaluator_contract",
            "budget_total_gpu_hours",
        ),
        "campaign_queue": (
            "queue_paths",
            "output_root",
            "workspace_root",
            "archive_db",
            "cas_root",
        ),
        "evolution": (
            "repo_root",
            "output_root",
            "state_root",
            "evaluator_contract",
            "total_budget_gpu_hours",
        ),
    }
    if kind not in required:
        raise CampaignAPIError("EXECUTION_KIND_INVALID")
    missing = [
        name
        for name in required[str(kind)]
        if value.get(name) is None or value.get(name) == ""
    ]
    if missing:
        raise CampaignAPIError(f"EXECUTION_REQUIRED:{','.join(missing)}")
    queue_paths = value.get("queue_paths")
    if kind == "campaign_queue" and (
        not isinstance(queue_paths, list)
        or not queue_paths
        or not all(isinstance(path, str) and path for path in queue_paths)
    ):
        raise CampaignAPIError("EXECUTION_QUEUE_PATHS_INVALID")
    for name, item in value.items():
        if name.endswith(("_root", "_db", "_contract", "_python")):
            if item is not None and (not isinstance(item, str) or not Path(item).is_absolute()):
                raise CampaignAPIError(f"EXECUTION_PATH_INVALID:{name}")
    if kind == "campaign_queue" and any(not Path(path).is_absolute() for path in queue_paths):
        raise CampaignAPIError("EXECUTION_PATH_INVALID:queue_paths")
    if kind == "pipeline":
        bindings = value.get("asset_bindings")
        if not isinstance(bindings, dict) or not bindings:
            raise CampaignAPIError("EXECUTION_ASSET_BINDINGS_INVALID")
        for parameter, path in bindings.items():
            if (
                not isinstance(parameter, str)
                or not parameter.startswith("--")
                or not isinstance(path, str)
                or not Path(path).is_absolute()
            ):
                raise CampaignAPIError("EXECUTION_ASSET_BINDINGS_INVALID")
        if not isinstance(value.get("probe_imports", True), bool):
            raise CampaignAPIError("EXECUTION_PROBE_IMPORTS_INVALID")
        for field in ("cpbe_request", "cpbe_history"):
            item = value.get(field)
            if item is not None and (
                not isinstance(item, str) or not Path(item).is_absolute()
            ):
                raise CampaignAPIError(f"EXECUTION_PATH_INVALID:{field}")
    for budget_name in ("budget_total_gpu_hours", "total_budget_gpu_hours"):
        if budget_name in value:
            budget_value = value[budget_name]
            if (
                isinstance(budget_value, bool)
                or not isinstance(budget_value, (int, float))
                or not math.isfinite(float(budget_value))
                or float(budget_value) <= 0
            ):
                raise CampaignAPIError(f"EXECUTION_BUDGET_INVALID:{budget_name}")


class _Handler(BaseHTTPRequestHandler):
    server_version = "VerdiWMCampaignAPI/1"

    @property
    def store(self) -> CampaignStore:
        return self.server.store  # type: ignore[attr-defined]

    def _json(self, status: int, body: dict[str, Any]) -> None:
        data = (_canonical(body) + "\n").encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _body(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            value = json.loads(self.rfile.read(length))
        except (ValueError, json.JSONDecodeError) as exc:
            raise CampaignAPIError("JSON_INVALID") from exc
        if not isinstance(value, dict):
            raise CampaignAPIError("JSON_OBJECT_REQUIRED")
        return value

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        parts = parsed.path.rstrip("/").split("/")
        if parsed.path == "/v1/campaigns":
            try:
                query = parse_qs(parsed.query)
                limit = int(query.get("limit", ["100"])[-1])
                status = query.get("status", [None])[-1]
                self._json(
                    HTTPStatus.OK,
                    {
                        "schema_version": SCHEMA_VERSION,
                        "artifact_type": "verdiwm-campaign-list",
                        "items": self.store.list(status=status, limit=limit),
                    },
                )
            except (CampaignAPIError, ValueError) as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        if len(parts) == 4 and parts[1:3] == ["v1", "campaigns"]:
            try:
                self._json(HTTPStatus.OK, self.store.get(parts[3]))
            except CampaignAPIError as exc:
                self._json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
            return
        if parsed.path in {"/v1/evidence/nodes", "/v1/evidence/edges"}:
            graph_path = getattr(self.server, "evidence_graph_path", None)
            if graph_path is None:
                self._json(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    {"error": "EVIDENCE_GRAPH_NOT_CONFIGURED"},
                )
                return
            try:
                query = parse_qs(parsed.query)
                result = query_evidence_graph(
                    graph_path,
                    entity="nodes" if parsed.path.endswith("nodes") else "edges",
                    filters={name: values[-1] for name, values in query.items()},
                )
                self._json(HTTPStatus.OK, result)
            except (CampaignAPIError, ValueError) as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        self._json(HTTPStatus.NOT_FOUND, {"error": "ROUTE_NOT_FOUND"})

    def do_POST(self) -> None:  # noqa: N802
        parts = self.path.rstrip("/").split("/")
        try:
            body = self._body()
            if self.path == "/v1/campaigns":
                self._json(HTTPStatus.CREATED, self.store.create(body))
            elif len(parts) == 5 and parts[1:3] == ["v1", "campaigns"] and parts[4] in {"confirm", "cancel"}:
                if parts[4] == "confirm":
                    record = self.store.confirm(parts[3])
                    code = (
                        HTTPStatus.ACCEPTED
                        if record["status"] == "queued"
                        else HTTPStatus.OK
                    )
                    self._json(code, record)
                else:
                    self._json(
                        HTTPStatus.OK,
                        self.store.cancel(parts[3]),
                    )
            elif len(parts) == 5 and parts[1:3] == ["v1", "campaigns"] and parts[4] == "reproduce":
                self._json(HTTPStatus.ACCEPTED, self.store.reproduce(parts[3]))
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "ROUTE_NOT_FOUND"})
        except CampaignAPIError as exc:
            code = HTTPStatus.CONFLICT if str(exc).endswith("CONFLICT") or str(exc).endswith("TRANSITION_INVALID") else HTTPStatus.BAD_REQUEST
            self._json(code, {"error": str(exc)})

    def log_message(self, *_args: object) -> None:
        return


def serve(
    root: Path,
    host: str = "127.0.0.1",
    port: int = 8787,
    evidence_graph_path: Path | None = None,
) -> None:
    server = ThreadingHTTPServer((host, port), _Handler)
    server.store = CampaignStore(root)  # type: ignore[attr-defined]
    server.evidence_graph_path = (  # type: ignore[attr-defined]
        Path(evidence_graph_path).expanduser().resolve()
        if evidence_graph_path is not None
        else None
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the VerdiWM Campaign API")
    parser.add_argument("--state-root", type=Path, default=Path(".verdiwm/campaigns"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--evidence-graph", type=Path)
    args = parser.parse_args()
    serve(args.state_root, args.host, args.port, args.evidence_graph)
