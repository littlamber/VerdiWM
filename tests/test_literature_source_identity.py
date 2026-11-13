"""No-Git literature staging binds package content without Git authority."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from wmloop.execute.agent_staging import AgentRepairSession, AgentStagingError
from wmloop.retrieve import method_staging as staging


SOURCE = Path(__file__).resolve().parents[1]


@pytest.fixture
def control_tree(tmp_path):
    root = tmp_path / "control-plane"
    module = root / "wmloop/retrieve/method_staging.py"
    schema = root / "configs/schemas/identity-fixture.json"
    for path, content in ((module, "# source fixture\n"), (schema, "{}\n")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return root


def test_no_git_identity_is_bounded_relocatable_and_tracks_source_and_schema(control_tree, tmp_path):
    original = staging._source_revision(control_tree)
    assert original.startswith("source_tree_sha256:")
    relocated = tmp_path / "relocated"
    shutil.copytree(control_tree, relocated)
    assert staging._source_revision(relocated) == original
    unrelated = control_tree / "another_distribution"
    unrelated.mkdir()
    (unrelated / "module.py").write_text("changed dependency", encoding="utf-8")
    assert staging._source_revision(control_tree) == original
    (control_tree / "wmloop/retrieve/method_staging.py").write_text("# changed\n", encoding="utf-8")
    assert staging._source_revision(control_tree) != original
    (relocated / "configs/schemas/identity-fixture.json").write_text('{"changed": true}\n', encoding="utf-8")
    assert staging._source_revision(relocated) != original


@pytest.mark.parametrize("mutation", ["missing_source", "missing_schema", "source_link", "schema_link"])
def test_no_git_source_binding_fails_closed(control_tree, mutation):
    path = control_tree / ("wmloop/retrieve/method_staging.py" if "source" in mutation
                           else "configs/schemas/identity-fixture.json")
    payload = path.read_bytes()
    path.unlink()
    if mutation.endswith("link"):
        target = control_tree.parent / "unbound-content"
        target.write_bytes(payload)
        path.symlink_to(target)
    with pytest.raises(staging.LiteratureMethodStagingError, match="SOURCE_REVISION_UNAVAILABLE"):
        staging._source_revision(control_tree)


def test_empty_public_staging_works_without_git_and_revalidates_content(control_tree, tmp_path, monkeypatch):
    shutil.copytree(SOURCE / "wmloop/primitives/definitions", control_tree / "wmloop/primitives/definitions")
    monkeypatch.setattr(staging, "__file__", str(control_tree / "wmloop/retrieve/method_staging.py"))
    retrieval = tmp_path / "retrieval.json"
    retrieval.write_text(json.dumps({"artifact_type": "verdiwm-literature-retrieval-manifest", "rows": []}), encoding="utf-8")
    options = dict(literature_manifest=retrieval, output_root=tmp_path / "staging", repo_root=control_tree)
    manifest = staging.run_literature_method_staging(**options)
    assert manifest["state"] == "ready"
    assert manifest["record_count"] == 0
    report = json.loads(Path(manifest["report_path"]).read_bytes())
    assert report["source_revision_kind"] == "source_tree_sha256"
    assert report["source_revision"].startswith("source_tree_sha256:")
    assert staging.run_literature_method_staging(**options) == manifest
    (control_tree / "wmloop/retrieve/method_staging.py").write_text("# changed\n", encoding="utf-8")
    with pytest.raises(staging.LiteratureMethodStagingError, match="INPUT_MISMATCH"):
        staging.run_literature_method_staging(**options)


def test_tree_identity_cannot_be_used_as_materialization_git_ref(control_tree, tmp_path):
    revision = staging._source_revision(control_tree)
    method = {"proposed_primitive": {"name": "literature_fixture"}, "candidate_id": "fixture",
              "required_hook": "H3", "target_failure_signatures": ["history"]}
    work_order = staging._work_order(method, source_revision=revision, registry_digest="a" * 64)
    contract = work_order["agent_staging_contract"]
    assert work_order["literature_execution_authority"] == "shadow_only"
    assert contract["source_revision"] is None
    assert contract["source_identity"] == revision
    assert contract["requires_git_checkout_binding"] is True
    assert contract["current_campaign_promotion_allowed"] is False
    with pytest.raises(AgentStagingError, match="SOURCE_REVISION_INVALID"):
        AgentRepairSession(worktree=control_tree, staging_root=tmp_path / "agent",
            candidate_id="fixture", source_revision=revision, registry_digest="a" * 64,
            required_check_labels=("static",))
    assert not (tmp_path / "agent").exists()


def test_git_source_revision_remains_a_real_commit(control_tree):
    subprocess.run(["git", "init", "-q", str(control_tree)], check=True)
    subprocess.run(["git", "-C", str(control_tree), "add", "."], check=True)
    subprocess.run(["git", "-C", str(control_tree), "-c", "user.name=fixture", "-c",
                    "user.email=fixture@invalid", "commit", "-qm", "seed"], check=True)
    revision = subprocess.run(["git", "-C", str(control_tree), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True).stdout.strip()
    assert staging._source_revision(control_tree) == revision


def test_no_git_export_nested_in_unrelated_git_repo_uses_own_content(control_tree):
    original = staging._source_revision(control_tree)
    outer = control_tree.parent
    subprocess.run(["git", "init", "-q", str(outer)], check=True)
    (outer / "unrelated.txt").write_text("unrelated repository\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(outer), "add", "unrelated.txt"], check=True)
    subprocess.run(["git", "-C", str(outer), "-c", "user.name=fixture", "-c",
                    "user.email=fixture@invalid", "commit", "-qm", "seed"], check=True)
    assert not (control_tree / ".git").exists()
    assert staging._source_revision(control_tree) == original
