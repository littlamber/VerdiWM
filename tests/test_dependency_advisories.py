"""No-network negative checks for the release advisory gate."""

from __future__ import annotations

from io import BytesIO
import json

import pytest

from scripts.ci import check_dependency_advisories as audit


@pytest.mark.parametrize("results,expected", [([{}], "passed"), ([{"vulns": [{"id": "TEST-ADVISORY"}]}], "blocked"), ([], "unavailable"), ([{"next_page_token": "more"}], "unavailable")])
def test_osv_result_contract(tmp_path, monkeypatch, results, expected):
    lock = tmp_path / "uv.lock"
    lock.write_text('[[package]]\nname="fixture"\nversion="1.0"\nsource={registry="https://pypi.org/simple"}\n')
    monkeypatch.setattr(audit, "urlopen", lambda *a, **k: BytesIO(json.dumps({"results": results}).encode()))
    report = audit.audit_dependencies(lock)
    assert report["status"] == expected
    assert len(report["lock_sha256"]) == 64


def test_network_failure_cannot_pass(tmp_path, monkeypatch):
    lock = tmp_path / "uv.lock"
    lock.write_text('[[package]]\nname="fixture"\nversion="1.0"\nsource={registry="https://pypi.org/simple"}\n')
    def unavailable(*args, **kwargs):
        raise TimeoutError("private proxy details")
    monkeypatch.setattr(audit, "urlopen", unavailable)
    report = audit.audit_dependencies(lock)
    assert report["status"] == "unavailable"
    assert "private proxy" not in json.dumps(report)
