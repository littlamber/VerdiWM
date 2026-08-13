"""Small, dependency-free Campaign API for the VerdiWM control plane.

The API stores request state as JSON documents and deliberately does not run
experiments itself.  Execution remains owned by the existing pipeline/daemon;
this boundary only accepts, validates, and durably records user intent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

from wmloop.experiments.evidence_graph import query_evidence_graph


SCHEMA_VERSION = 1
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
        goal = payload.get("goal")
        if not isinstance(goal, str) or not goal.strip():
            raise CampaignAPIError("GOAL_REQUIRED")
        execution = payload.get("execution")
        if execution is not None:
            _validate_execution(execution)
        record = {
            "schema_version": SCHEMA_VERSION,
            "campaign_id": campaign_id,
            "goal": goal.strip(),
            "model": payload.get("model"),
            "dataset": payload.get("dataset"),
            "budget": payload.get("budget"),
            "execution": execution,
            "status": "created",
            "created_at": _now(),
            "updated_at": _now(),
            "request_hash": hashlib.sha256(_canonical(payload).encode()).hexdigest(),
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
                return self.transition(campaign_id, "confirmed")
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

    def reproduce(self, campaign_id: str) -> dict[str, Any]:
        source = self.get(campaign_id)
        child = dict(source)
        child_id = f"{campaign_id}-repro-{uuid.uuid4().hex[:8]}"
        child.update(
            {
                "campaign_id": child_id,
                "status": "created" if child.get("execution") is not None else "queued",
                "parent_campaign_id": campaign_id,
                "created_at": _now(),
                "updated_at": _now(),
            }
        )
        child["request_hash"] = hashlib.sha256(
            _canonical(
                {
                    "parent_campaign_id": campaign_id,
                    "execution": child.get("execution"),
                    "goal": child.get("goal"),
                }
            ).encode()
        ).hexdigest()
        with self._lock:
            self._write(self._path(child_id), child)
        return child


def _validate_execution(value: object) -> None:
    if not isinstance(value, dict):
        raise CampaignAPIError("EXECUTION_OBJECT_REQUIRED")
    kind = value.get("kind")
    required = {
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
    missing = [name for name in required[str(kind)] if value.get(name) in {None, ""}]
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
                        self.store.transition(parts[3], "cancelled"),
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
