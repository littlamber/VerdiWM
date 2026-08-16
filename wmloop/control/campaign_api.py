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
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from wmloop.control.adapter_profiles import (
    AdapterProfileError,
    compile_adapter_execution,
    parse_gpu_budget,
)
from wmloop.experiments.evidence_graph import query_evidence_graph


SCHEMA_VERSION = 2
_TRANSITIONS = {
    "created": {"confirmed", "queued", "cancelled"},
    "confirmed": {"queued", "cancelled"},
    "queued": {"running", "cancelled", "failed"},
    "running": {"completed", "failed", "cancelled"},
    "cancelled": set(),
    "completed": set(),
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
        campaign_id = str(payload.get("campaign_id") or uuid.uuid4().hex)
        self._path(campaign_id)
        goal = _required_text(payload, "goal", "GOAL_REQUIRED")
        model = _required_text(payload, "model", "MODEL_REQUIRED")
        dataset = _required_text(payload, "dataset", "DATASET_REQUIRED")
        try:
            budget_hours = parse_gpu_budget(payload.get("budget"))
        except AdapterProfileError as exc:
            raise CampaignAPIError(str(exc)) from exc
        request_hash = hashlib.sha256(_canonical(payload).encode()).hexdigest()
        with self._lock:
            path = self._path(campaign_id)
            if path.exists():
                existing = json.loads(path.read_text(encoding="utf-8"))
                if existing.get("request_hash") != request_hash:
                    raise CampaignAPIError("CAMPAIGN_ID_CONFLICT")
                return existing

        execution = payload.get("execution")
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
        _validate_execution(execution)
        record = {
            "schema_version": SCHEMA_VERSION,
            "campaign_id": campaign_id,
            "goal": goal,
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
        with self._lock:
            path = self._path(campaign_id)
            if path.exists():
                existing = json.loads(path.read_text(encoding="utf-8"))
                if existing.get("request_hash") != record["request_hash"]:
                    raise CampaignAPIError("CAMPAIGN_ID_CONFLICT")
                return existing
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
                "state": "pending",
                "execution": execution,
                "execution_hash": hashlib.sha256(
                    _canonical(execution).encode()
                ).hexdigest(),
                "created_at": _now(),
            }
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
        if status not in {"running", "completed", "failed"}:
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

    def reproduce(self, campaign_id: str) -> dict[str, Any]:
        source = self.get(campaign_id)
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
