"""Bounded CPU checks for per-process local credentials and public shells."""

from __future__ import annotations

from http.client import HTTPConnection
import os
from pathlib import Path
import shutil
import stat
import subprocess

import pytest

from test_release_security import running_server


def request(server, route, *, token=None, method="GET"):
    client = HTTPConnection("127.0.0.1", server.server_port, timeout=3)
    try:
        headers = {} if token is None else {"Authorization": "Bearer " + token}
        client.request(method, route, body=b"{}" if method == "POST" else None, headers=headers)
        response = client.getresponse()
        return response.status, response.read()
    finally:
        client.close()


@pytest.mark.parametrize("kind,route", [("campaign", "/v1/campaigns"), ("workbench", "/api/campaigns")])
def test_all_api_methods_require_current_process_token(tmp_path, kind, route):
    with running_server(tmp_path, kind) as server:
        token_path = server.auth_token_path
        token = token_path.read_text().strip()
        assert len(token) >= 43
        assert stat.S_IMODE(token_path.stat().st_mode) == 0o600
        assert stat.S_IMODE(token_path.parent.stat().st_mode) == 0o700
        assert token_path.stat().st_uid == os.getuid()
        for method in ("GET", "POST"):
            assert request(server, route, method=method)[0] == 401
            assert request(server, route, token="incorrect", method=method)[0] == 401
        assert request(server, route, token=token)[0] == 200
        assert request(server, "/auth/session", token=token)[0] == 204
        assert token.encode() not in request(server, route)[1]
    assert not token_path.exists()
    with running_server(tmp_path, kind) as restarted:
        assert request(restarted, route, token=token)[0] == 401


def test_public_login_and_assets_never_expose_credentials(tmp_path):
    with running_server(tmp_path, "workbench") as server:
        for route in ("/auth", "/assets/auth.js", "/assets/setup.css"):
            code, body = request(server, route)
            assert code == 200
            assert server.auth_token.encode() not in body
            assert str(server.auth_token_path).encode() not in body
        assert request(server, "/assets/not-public.json")[0] == 401
        assert request(server, "/auth/session")[0] == 401


def test_other_linux_user_cannot_read_token_or_api(tmp_path):
    if os.geteuid() != 0 or not shutil.which("runuser") or not shutil.which("curl"):
        pytest.skip("Cross-UID reproduction requires root, runuser and curl")
    with running_server(tmp_path, "campaign") as server:
        read = subprocess.run(["runuser", "-u", "nobody", "--", "test", "-r", str(server.auth_token_path)], capture_output=True, timeout=5)
        assert read.returncode != 0
        result = subprocess.run(["runuser", "-u", "nobody", "--", "curl", "--silent", "--max-time", "3", "--output", "/dev/null", "--write-out", "%{http_code}", f"http://127.0.0.1:{server.server_port}/v1/campaigns"], capture_output=True, text=True, timeout=5)
        assert result.returncode == 0
        assert result.stdout == "401"


def test_public_junit_summary_omits_all_free_text(tmp_path):
    from scripts.export.build_release_delivery import public_test_summary

    path = tmp_path / "raw.xml"
    path.write_text('<testsuites><testsuite tests="2" failures="0" errors="0" skipped="1" hostname="private-host"><properties><property name="secret" value="private-token"/></properties><testcase name="private-user-path"><skipped message="private-dataset"/><system-out>private-output</system-out></testcase></testsuite></testsuites>')
    result = public_test_summary(path)
    assert b'private' not in result
    assert b'tests="2"' in result
    assert b'skipped="1"' in result


def test_browser_token_is_origin_scoped_and_expires_on_401():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Browser helper contract requires Node from the CI image")
    source = Path(__file__).resolve().parents[1] / "wmloop/web/auth.js"
    script = r'''
const fs = require('fs');
const vm = require('vm');
const values = new Map([['verdiwm-session-token', 'fixture-token']]);
const calls = [], redirects = [];
let status = 200;
const context = {
  URL, Headers,
  location: {origin: 'http://127.0.0.1:8765', replace: value => redirects.push(value)},
  sessionStorage: {getItem: key => values.get(key), removeItem: key => values.delete(key)},
  document: {querySelector: () => null},
  fetch: async (url, options) => {calls.push([url, options]); return {status};},
};
vm.createContext(context);
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8'), context);
(async () => {
  await context.verdiFetch('/api/campaigns');
  const [url, options] = calls[0];
  if (url.search || options.headers.get('Authorization') !== 'Bearer fixture-token' || options.credentials !== 'omit' || options.redirect !== 'error') throw Error('credential transport');
  for (const target of ['https://example.invalid/', 'http://127.0.0.1:9999/']) {
    let rejected = false;
    try {await context.verdiFetch(target);} catch {rejected = true;}
    if (!rejected) throw Error('cross-origin credential request');
  }
  if (calls.length !== 1) throw Error('cross-origin fetch sent');
  status = 401;
  try {await context.verdiFetch('/api/campaigns');} catch {}
  if (values.has('verdiwm-session-token') || redirects.at(-1) !== '/auth') throw Error('expired token retained');
})().catch(error => {process.exitCode = 1; throw error;});
'''
    subprocess.run([node, "-e", script, str(source)], check=True, capture_output=True, timeout=10)
