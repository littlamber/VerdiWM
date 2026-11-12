"""CPU-only release boundaries: uv run --offline --frozen pytest -q tests/test_release_security.py."""

from __future__ import annotations

from contextlib import contextmanager
from http.client import HTTPConnection
import json
from pathlib import Path
import subprocess
from threading import Thread

import pytest

from scripts.ci.check_release_history import audit_history
from scripts.export.build_github_staging import (
    GithubStagingError,
    _is_public_source_path,
    _source_files,
    audit_release_tree,
    build_github_staging,
)
from wmloop.control.campaign_api import CampaignStore, _Handler
from wmloop.control.commands.community import _load_community_documents
from wmloop.control.local_http import LocalHTTPServer
from wmloop.control.workbench import WorkbenchServer
from wmloop.execute.pipeline.io import _optional_file
from wmloop.execute.pipeline.types import AutonomousPipelineError
from wmloop.experiments.community_bundle import CommunityBundleError


@pytest.mark.parametrize("prefix", ["sk-", "sk-proj-", "sk_", "ghp_", "github_pat_"])
def test_known_credentials_are_detected_without_echoing_values(tmp_path, prefix):
    secret = prefix + "a" * 40
    (tmp_path / "config.custom").write_text(json.dumps({"token": secret}))
    findings = audit_release_tree(tmp_path)
    assert findings == ["secret_pattern:config.custom"]
    assert secret not in str(findings)


def test_audit_detects_shell_assignment_paths_and_rejects_null_bytes(tmp_path):
    (tmp_path / "config").write_text("ROOT=" + "/" + "cs/private/project")
    (tmp_path / "binary").write_bytes(b"fixture\x00data")
    assert audit_release_tree(tmp_path) == ["local_path:config", "non_text_file:binary"]


@pytest.mark.parametrize("name", ["docs/.env", "docs/.env.production", "docs/key.pem", "docs/plans/a.md", "docs/summary/a.md", "wmloop/web/ui/index.html"])
def test_private_paths_are_never_public_sources(tmp_path, name):
    relative = Path(name)
    assert not _is_public_source_path(relative)
    target = tmp_path / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("fixture")
    assert audit_release_tree(tmp_path)


@pytest.mark.parametrize("parent_link", [False, True])
def test_input_guards_reject_links_before_resolution(tmp_path, parent_link):
    real = tmp_path / "real"
    real.mkdir()
    document = real / "input.json"
    document.write_text('{}')
    link = tmp_path / "link"
    link.symlink_to(real if parent_link else document, target_is_directory=parent_link)
    supplied = link / "input.json" if parent_link else link
    with pytest.raises(AutonomousPipelineError):
        _optional_file(supplied, "INPUT_INVALID")
    with pytest.raises(CommunityBundleError):
        _load_community_documents([supplied])
    assert _optional_file(document, "INPUT_INVALID") == document
    assert _load_community_documents([document]) == [{}]


def test_export_rejects_dangling_output_symlink(tmp_path):
    destination = tmp_path / "link"
    destination.symlink_to(tmp_path / "missing")
    with pytest.raises(GithubStagingError, match="OUTPUT_SYMLINK"):
        build_github_staging(source_root=Path(__file__).resolve().parents[1], output_root=destination)


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


def test_untracked_source_requires_explicit_snapshot(tmp_path):
    git(tmp_path, "init", "--quiet")
    (tmp_path / "wmloop").mkdir()
    (tmp_path / "wmloop/new.py").write_text("x = 1\n")
    with pytest.raises(GithubStagingError, match="UNTRACKED_SOURCE"):
        _source_files(tmp_path)
    assert _source_files(tmp_path, include_untracked=True) == [Path("wmloop/new.py")]


def test_tracked_credentials_block_release_even_when_export_would_filter_them(tmp_path):
    git(tmp_path, "init", "--quiet")
    (tmp_path / ".env").write_text("TOKEN=fixture")
    git(tmp_path, "add", ".env")
    with pytest.raises(GithubStagingError, match="NONPUBLIC_TRACKED_FILE"):
        _source_files(tmp_path)


def test_history_scan_detects_deleted_credential(tmp_path):
    git(tmp_path, "init", "--quiet")
    git(tmp_path, "config", "user.name", "Fixture")
    git(tmp_path, "config", "user.email", "fixture@example.invalid")
    secret = "github_" + "pat_" + "b" * 40
    (tmp_path / "config.txt").write_text(secret)
    git(tmp_path, "add", "config.txt")
    git(tmp_path, "commit", "--quiet", "-m", "fixture")
    git(tmp_path, "rm", "config.txt")
    git(tmp_path, "commit", "--quiet", "-m", "remove")
    report = audit_history(tmp_path)
    assert report["status"] == "blocked"
    assert any(row.startswith("secret_pattern:") for row in report["findings"])
    assert secret not in json.dumps(report)


@contextmanager
def running_server(tmp_path, kind):
    if kind == "campaign":
        server = LocalHTTPServer(("127.0.0.1", 0), _Handler)
        server.store = CampaignStore(tmp_path / "campaigns")
        server.evidence_graph_path = None
    else:
        server = WorkbenchServer(("127.0.0.1", 0), state_root=tmp_path / "state", project_root=tmp_path)
    thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.0.2.1", "example.invalid"])
def test_services_reject_non_loopback_bindings(tmp_path, host):
    with pytest.raises(ValueError, match="LOOPBACK"):
        LocalHTTPServer((host, 0), _Handler)
    with pytest.raises(ValueError, match="LOOPBACK"):
        WorkbenchServer((host, 0), state_root=tmp_path)


@pytest.mark.parametrize("kind", ["campaign", "workbench"])
def test_http_origin_and_framing_boundaries(tmp_path, kind):
    with running_server(tmp_path, kind) as server:
        route = "/v1/campaigns" if kind == "campaign" else "/api/campaigns"

        def request(method="GET", headers=None, body=None):
            connection = HTTPConnection("127.0.0.1", server.server_port, timeout=2)
            try:
                connection.request(method, route, body=body, headers={"Authorization": "Bearer " + server.auth_token, **(headers or {})})
                response = connection.getresponse()
                response.read()
                return response.status
            finally:
                connection.close()

        assert request() == 200
        assert request(headers={"Host": f"attacker.invalid:{server.server_port}"}) == 403
        assert request(headers={"Origin": "https://attacker.invalid"}) == 403
        assert request(headers={"Origin": "null"}) == 403
        assert request(headers={"Sec-Fetch-Site": "cross-site"}) == 403
        assert request(headers={"Origin": f"http://127.0.0.1:{server.server_port}"}) == 200
        for headers in (
            {"Content-Length": "-1", "Content-Type": "application/json"},
            {"Content-Length": "1048577", "Content-Type": "application/json"},
            {"Content-Length": "2", "Content-Type": "text/plain"},
            {"Content-Length": "2", "Content-Type": "application/json", "Transfer-Encoding": "chunked"},
        ):
            assert request("POST", headers, b"{}") == 400
        for name, values, expected in (
            ("Host", [f"127.0.0.1:{server.server_port}", "attacker.invalid"], 403),
            ("Content-Length", ["2", "3"], 400),
            ("Content-Type", ["application/json", "text/plain"], 400),
        ):
            connection = HTTPConnection("127.0.0.1", server.server_port, timeout=2)
            connection.putrequest("POST", route, skip_host=True)
            connection.putheader("Authorization", "Bearer " + server.auth_token)
            if name != "Host":
                connection.putheader("Host", f"127.0.0.1:{server.server_port}")
            if name != "Content-Length":
                connection.putheader("Content-Length", "2")
            if name != "Content-Type":
                connection.putheader("Content-Type", "application/json")
            for value in values:
                connection.putheader(name, value)
            connection.endheaders(b"{}")
            response = connection.getresponse()
            assert response.status == expected
            response.read()
            connection.close()
        assert server.store.list() == []


def test_source_workbench_serves_editable_assets(tmp_path):
    with running_server(tmp_path, "workbench") as server:
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=2)
        connection.request("GET", "/assets/workbench.js")
        response = connection.getresponse()
        assert response.status == 200
        assert b"function renderCampaigns" in response.read()
        connection.close()
