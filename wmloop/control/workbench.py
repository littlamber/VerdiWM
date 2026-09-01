"""Serve the local VerdiWM research workbench and JSON API."""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import parse_qs, urlparse

from wmloop.control.campaign_api import CampaignAPIError, CampaignStore
from wmloop.control.campaign_dispatcher import (
    CampaignDispatchError,
    DispatcherOptions,
    run_dispatcher,
)
from wmloop.control.research_modes import research_mode_catalog
from wmloop.control.project_config import ProjectConfigError, load_project_config
from wmloop.control.first_contact import FirstContactError, initialize_project
from wmloop.experiments.atlas import AtlasError, build_atlas
from wmloop.experiments.artifact_lint import make_compliance_filter
from wmloop.experiments.evidence_graph import EvidenceGraphError, build_evidence_graph, load_evidence_graph
from wmloop.experiments.mechanism_board import MechanismBoardError, build_mechanism_board


class WorkbenchError(ValueError):
    """The workbench request or local binding is invalid."""


_ASSET_ROOT = Path(__file__).resolve().parents[1] / "web"
_ASSETS = {
    "/": ("workbench.html", "text/html; charset=utf-8"),
    "/assets/workbench.css": ("workbench.css", "text/css; charset=utf-8"),
    "/assets/workbench.js": ("workbench.js", "text/javascript; charset=utf-8"),
}
_SETUP_ASSETS = {
    "/setup": ("setup.html", "text/html; charset=utf-8"),
    "/assets/setup.css": ("setup.css", "text/css; charset=utf-8"),
    "/assets/setup.js": ("setup.js", "text/javascript; charset=utf-8"),
}

# React workbench build output (workbench-ui/dist at the repository root, or the
# packaged copy under wmloop/web/ui). When present it replaces the legacy page.
_UI_DIST_CANDIDATES = (
    Path(__file__).resolve().parents[2] / "workbench-ui" / "dist",
    _ASSET_ROOT / "ui",
)
_UI_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
}


def _ui_dist_root() -> Path | None:
    for candidate in _UI_DIST_CANDIDATES:
        index = candidate / "index.html"
        if index.is_file() and not index.is_symlink():
            return candidate
    return None


class WorkbenchServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], *, state_root: Path, evidence_root: Path | None = None, project_root: Path | None = None):
        super().__init__(address, WorkbenchHandler)
        self.project_root = Path(project_root or Path.cwd()).expanduser().resolve()
        self.evidence_root_explicit = evidence_root is not None
        self.state_root = Path(state_root).expanduser().resolve()
        self.store = CampaignStore(self.state_root / "campaigns")
        # Campaign requests live below ``campaigns``; evidence is produced by
        # the wider state tree. Keep the input configurable for deployments
        # that maintain a separate immutable evidence archive.
        self.evidence_root = (
            Path(evidence_root).expanduser().resolve()
            if evidence_root is not None
            else self.state_root
        )
        try:
            loaded = load_project_config(cwd=self.project_root)
            self.project_config = loaded if loaded.source is not None else None
        except ProjectConfigError:
            self.project_config = None

    def reload_project(self) -> None:
        loaded = load_project_config(cwd=self.project_root)
        if loaded.source is None:
            raise ProjectConfigError("PROJECT_CONFIG_NOT_FOUND")
        self.project_config = loaded
        configured_root = self.project_config.values.get("state_root")
        if configured_root:
            self.state_root = Path(str(configured_root)).expanduser().resolve()
            self.store = CampaignStore(self.state_root / "campaigns")
            if not self.evidence_root_explicit:
                self.evidence_root = self.state_root


class WorkbenchHandler(BaseHTTPRequestHandler):
    server_version = "VerdiWMWorkbench/1"

    @property
    def workbench(self) -> WorkbenchServer:
        return self.server  # type: ignore[return-value]

    def do_GET(self) -> None:  # noqa: N802
        route = urlparse(self.path).path
        try:
            if route == "/favicon.ico":
                self.send_response(HTTPStatus.NO_CONTENT)
                self._security_headers()
                self.end_headers()
                return
            if route == "/" and self.workbench.project_config is None:
                self.send_response(HTTPStatus.SEE_OTHER)
                self._security_headers()
                self.send_header("Location", "/setup")
                self.end_headers()
                return
            if route in _SETUP_ASSETS:
                self._asset(*_SETUP_ASSETS[route])
                return
            if route == "/setup":
                self._asset("setup.html", "text/html; charset=utf-8")
                return
            if not route.startswith("/api/") and self._serve_ui(route):
                return
            if route in _ASSETS:
                self._asset(*_ASSETS[route])
                return
            if route == "/api/modes":
                self._json(HTTPStatus.OK, {"items": research_mode_catalog()})
                return
            if route == "/api/project":
                values = self.workbench.project_config.values if self.workbench.project_config else {}
                safe = {
                    key: values[key]
                    for key in ("model", "data", "dataset", "budget", "adapter", "mode", "metric", "metrics", "target_metrics")
                    if key in values
                }
                self._json(
                    HTTPStatus.OK,
                    {"configured": self.workbench.project_config is not None, "values": safe},
                )
                return
            if route == "/api/first-contact":
                configured = self.workbench.project_config
                self._json(
                    HTTPStatus.OK,
                    {
                        "configured": configured is not None,
                        "values": configured.values if configured is not None else {},
                        "next_step": "填写模型、数据和研究目标" if configured is None else "可以开始研究",
                    },
                )
                return
            if route == "/api/campaigns":
                self._json(
                    HTTPStatus.OK,
                    {"items": self.workbench.store.list(limit=250)},
                )
                return
            if route.startswith("/api/campaigns/"):
                self._json(
                    HTTPStatus.OK,
                    self.workbench.store.get(route.rsplit("/", 1)[-1]),
                )
                return
            if route == "/api/graph":
                query = parse_qs(urlparse(self.path).query)
                materialized = load_evidence_graph(self.workbench.evidence_root)
                if materialized is not None:
                    self._json(HTTPStatus.OK, materialized)
                    return
                include = None
                if query.get("clean", ["0"])[-1] in {"1", "true"}:
                    include = make_compliance_filter(self.workbench.evidence_root)
                self._json(
                    HTTPStatus.OK,
                    build_evidence_graph(self.workbench.evidence_root, include_payload=include),
                )
                return
            if route == "/api/atlas":
                self._json(
                    HTTPStatus.OK,
                    build_atlas(self.workbench.state_root),
                )
                return
            if route == "/api/mechanisms":
                self._json(
                    HTTPStatus.OK,
                    build_mechanism_board(self.workbench.state_root),
                )
                return
            self._json(HTTPStatus.NOT_FOUND, {"error": "NOT_FOUND"})
        except (CampaignAPIError, EvidenceGraphError, AtlasError, MechanismBoardError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def do_POST(self) -> None:  # noqa: N802
        route = urlparse(self.path).path
        try:
            payload = self._body()
            if route == "/api/campaigns":
                queue = payload.pop("queue", True)
                if self.workbench.project_config is not None:
                    defaults = self.workbench.project_config.values
                    for key, target in (("model", "model"), ("data", "dataset"), ("dataset", "dataset"), ("budget", "budget"), ("adapter", "adapter"), ("target_metrics", "target_metrics"), ("metrics", "target_metrics"), ("metric", "target_metrics"), ("adapter_profile", "adapter_profile_path"), ("runtime_python", "runtime_python")):
                        if target not in payload and key in defaults:
                            payload[target] = defaults[key]
                created = self.workbench.store.create(payload)
                result = (
                    self.workbench.store.confirm(str(created["campaign_id"]))
                    if queue is True
                    else created
                )
                self._json(HTTPStatus.CREATED, result)
                return
            if route == "/api/first-contact":
                result = initialize_project(
                    root=self.workbench.project_root,
                    model=payload.get("model"),
                    data=payload.get("data") or payload.get("dataset"),
                    goal=payload.get("goal"),
                    budget=str(payload.get("budget") or "1gpu-hour"),
                    mode=str(payload.get("mode") or "hybrid"),
                    target_metrics=payload.get("target_metrics") if isinstance(payload.get("target_metrics"), list) else [],
                    force=bool(payload.get("force", False)),
                )
                if result["state"] == "ready":
                    self.workbench.reload_project()
                self._json(HTTPStatus.OK if result["state"] != "ready" else HTTPStatus.CREATED, result)
                return
            if route == "/api/dispatch":
                maximum = payload.get("max_parallel", 1)
                if not isinstance(maximum, int) or isinstance(maximum, bool) or not 1 <= maximum <= 8:
                    raise WorkbenchError("MAX_PARALLEL_INVALID")
                result = run_dispatcher(
                    DispatcherOptions(
                        state_root=self.workbench.store.root,
                        max_cycles=1,
                        max_parallel=maximum,
                    )
                )
                self._json(HTTPStatus.OK, result)
                return
            if route.startswith("/api/campaigns/"):
                parts = route.strip("/").split("/")
                if len(parts) != 4:
                    raise WorkbenchError("ROUTE_INVALID")
                campaign_id, action = parts[2], parts[3]
                if action == "confirm":
                    result = self.workbench.store.confirm(campaign_id)
                elif action == "cancel":
                    result = self.workbench.store.cancel(campaign_id)
                else:
                    raise WorkbenchError("ACTION_INVALID")
                self._json(HTTPStatus.OK, result)
                return
            self._json(HTTPStatus.NOT_FOUND, {"error": "NOT_FOUND"})
        except (CampaignAPIError, CampaignDispatchError, WorkbenchError, FirstContactError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def _body(self) -> dict[str, Any]:
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0]
        if content_type != "application/json":
            raise WorkbenchError("CONTENT_TYPE_JSON_REQUIRED")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise WorkbenchError("CONTENT_LENGTH_INVALID") from exc
        if length < 2 or length > 1_048_576:
            raise WorkbenchError("REQUEST_BODY_SIZE_INVALID")
        try:
            value = json.loads(self.rfile.read(length))
        except json.JSONDecodeError as exc:
            raise WorkbenchError("REQUEST_JSON_INVALID") from exc
        if not isinstance(value, dict):
            raise WorkbenchError("REQUEST_OBJECT_REQUIRED")
        return value

    def _serve_ui(self, route: str) -> bool:
        dist = _ui_dist_root()
        if dist is None:
            return False
        path = (dist / route.lstrip("/")).resolve() if route != "/" else dist / "index.html"
        if not path.is_relative_to(dist) or path.is_symlink() or not path.is_file():
            # SPA fallback: unknown non-file routes render the shell.
            path = dist / "index.html"
        content_type = _UI_CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")
        try:
            body = path.read_bytes()
        except OSError:
            return False
        self.send_response(HTTPStatus.OK)
        self._security_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        return True

    def _asset(self, filename: str, content_type: str) -> None:
        path = _ASSET_ROOT / filename
        try:
            body = path.read_bytes()
        except OSError:
            self._json(HTTPStatus.NOT_FOUND, {"error": "ASSET_NOT_FOUND"})
            return
        self.send_response(HTTPStatus.OK)
        self._security_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, value: Mapping[str, object]) -> None:
        body = (
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        self.send_response(status)
        self._security_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _security_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; connect-src 'self'",
        )

    def log_message(self, format: str, *args: object) -> None:
        return


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", choices=("127.0.0.1", "::1", "localhost"))
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--state-root",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--evidence-root",
        type=Path,
        default=None,
        help="目录中扫描实验产物；默认使用 state-root",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    state_root = args.state_root
    if state_root is None:
        try:
            configured = load_project_config().values.get("state_root")
        except ProjectConfigError:
            configured = None
        state_root = Path(str(configured)) if configured else Path.home() / ".local" / "state" / "verdiwm"
    server = WorkbenchServer(
        (args.host, args.port), state_root=state_root, evidence_root=args.evidence_root
    )
    print(f"VerdiWM workbench: http://{args.host}:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
