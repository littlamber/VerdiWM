import json
from pathlib import Path
import subprocess

import pytest

from verdi_core.engineering import AutonomousStageRunner, EngineeringAgent, EngineeringPolicyError, EngineeringSandbox, EngineeringTools


class ScriptedAI:
    def __init__(self, actions):
        self.actions = list(actions)
        self.prompts = []

    def complete(self, *, role: str, prompt: str) -> str:
        self.prompts.append((role, prompt))
        return json.dumps(self.actions.pop(0))


class FailingAI:
    def complete(self, *, role: str, prompt: str) -> str:
        raise TimeoutError("provider timed out")


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "hello.py").write_text("VALUE = 'old'\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "hello.py"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.email=a@b", "-c", "user.name=test", "commit", "-qm", "init"], cwd=repo, check=True)
    return repo


def test_engineering_agent_reads_patches_tests_and_finishes(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    worktree = tmp_path / "worktree"
    subprocess.run(["git", "worktree", "add", "--detach", str(worktree), "HEAD"], cwd=repo, check=True, capture_output=True)
    (worktree / "test_hello.py").write_text("from hello import VALUE\ndef test_value(): assert VALUE == 'new'\n", encoding="utf-8")
    # The fixture AI first reads, then patches, then runs tests, then reports.
    actions = [
        {"action": "read_file", "args": {"path": str(worktree / "hello.py")}},
        {"action": "apply_patch", "args": {"cwd": str(worktree), "diff": "diff --git a/hello.py b/hello.py\n--- a/hello.py\n+++ b/hello.py\n@@ -1 +1 @@\n-VALUE = 'old'\n+VALUE = 'new'\n"}},
        {"action": "run_tests", "args": {"cwd": str(worktree), "argv": ["python", "-m", "pytest", "-q"]}},
        {"action": "done", "args": {"state": "completed", "reason": "tests pass"}},
    ]
    tools = EngineeringTools(EngineeringSandbox(worktree, tmp_path / "out"), tmp_path / "out" / "audit.jsonl")
    result = EngineeringAgent(ScriptedAI(actions), tools).run(objective="repair hello")
    assert result["state"] == "completed"
    assert len(result["events"]) == 3
    assert "VALUE = 'new'" in result["result"]["materialized_patch"]["diff"]
    assert (tmp_path / "out" / "ai-engineering.jsonl").exists()
    assert "VALUE = 'new'" in (worktree / "hello.py").read_text()


def test_engineering_policy_blocks_escape_and_external_writes(tmp_path: Path) -> None:
    worktree = tmp_path / "worktree"
    out = tmp_path / "out"
    worktree.mkdir(); out.mkdir()
    tools = EngineeringTools(EngineeringSandbox(worktree, out), out / "audit.jsonl")
    escaped = tools.execute({"action": "read_file", "args": {"path": "/etc/passwd"}})
    assert escaped["state"] == "rejected"
    with pytest.raises(EngineeringPolicyError):
        tools.sandbox.validate_command(["git", "push"], cwd=worktree)
    with pytest.raises(EngineeringPolicyError):
        tools.sandbox.validate_command(["bash", "-lc", "git push"], cwd=worktree)
    with pytest.raises(EngineeringPolicyError):
        tools.sandbox.validate_command(["git", "reset", "--hard"], cwd=worktree)
    with pytest.raises(EngineeringPolicyError):
        tools.sandbox.validate_command(["rm", "-rf", "tmp"], cwd=worktree)
    with pytest.raises(EngineeringPolicyError):
        tools.sandbox.check_write(tmp_path / "outside.txt")


def test_engineering_tools_resolve_relative_paths_in_declared_worktree(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    worktree = tmp_path / "worktree"
    subprocess.run(["git", "worktree", "add", "--detach", str(worktree), "HEAD"], cwd=repo, check=True, capture_output=True)
    tools = EngineeringTools(EngineeringSandbox(worktree, tmp_path / "out"), tmp_path / "out" / "audit.jsonl")
    result = tools.execute({"action": "inspect_files", "args": {"path": "."}})
    assert result["state"] == "ok"
    assert any(row["path"] == "hello.py" for row in result["files"])
    batch = tools.execute({"action": "read_file", "args": {"paths": ["hello.py"]}})
    assert batch["state"] == "ok"
    assert "VALUE = 'old'" in batch["files"][0]["content"]
    aliased = tools.execute({"action": "inspect_files", "args": {"root": ".", "patterns": ["hello.py"], "max_depth": 2}})
    assert aliased["state"] == "ok"
    assert any(row["path"] == "hello.py" for row in aliased["paths"][0]["files"])


def test_engineering_tools_accept_patch_alias(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    worktree = tmp_path / "worktree"
    subprocess.run(["git", "worktree", "add", "--detach", str(worktree), "HEAD"], cwd=repo, check=True, capture_output=True)
    tools = EngineeringTools(EngineeringSandbox(worktree, tmp_path / "out"), tmp_path / "out" / "audit.jsonl")
    result = tools.execute({"action": "apply_patch", "args": {"cwd": str(worktree), "patch": "diff --git a/hello.py b/hello.py\n--- a/hello.py\n+++ b/hello.py\n@@ -1 +1 @@\n-VALUE = 'old'\n+VALUE = 'new'\n"}})
    assert result["state"] == "ok"
    assert "VALUE = 'new'" in (worktree / "hello.py").read_text()


def test_engineering_agent_recovers_first_action_from_concatenated_json(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    worktree = tmp_path / "worktree"
    subprocess.run(["git", "worktree", "add", "--detach", str(worktree), "HEAD"], cwd=repo, check=True, capture_output=True)
    tools = EngineeringTools(EngineeringSandbox(worktree, tmp_path / "out"), tmp_path / "out" / "audit.jsonl")
    raw = '{"action":"done","args":{"state":"completed"}}{"action":"done","args":{"state":"abstain"}}'
    assert EngineeringAgent._parse_action(raw) == {"action": "done", "args": {"state": "completed"}}
    prose_wrapped = 'I will inspect first. {"action":"inspect_files","args":{"path":"."}} Then patch.'
    assert EngineeringAgent._parse_action(prose_wrapped) == {"action": "inspect_files", "args": {"path": "."}}
    mixed_actions = '{"action":"inspect_files","args":{"path":"."}}{"action":"done","args":{"state":"abstain","reason":"no safe patch"}}'
    assert EngineeringAgent._parse_action(mixed_actions) == {"action": "done", "args": {"state": "abstain", "reason": "no safe patch"}}


def test_create_worktree_is_idempotent_inside_existing_isolation(tmp_path: Path) -> None:
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    tools = EngineeringTools(EngineeringSandbox(worktree, tmp_path / "out"), tmp_path / "out" / "audit.jsonl")
    result = tools.execute({"action": "create_worktree", "args": {"path": "../another", "branch": "repair"}})
    assert result == {"state": "ok", "worktree": str(worktree), "reason": "existing_isolated_worktree"}


def test_provider_failure_is_recorded_as_abstention(tmp_path: Path) -> None:
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    tools = EngineeringTools(EngineeringSandbox(worktree, tmp_path / "out"), tmp_path / "out" / "audit.jsonl")
    result = EngineeringAgent(FailingAI(), tools).run(objective="repair")
    assert result["state"] == "abstain"
    assert result["result"]["reason"] == "engineering_provider_failed"


def test_engineering_agent_settles_repeated_read_only_loop(tmp_path: Path) -> None:
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    (worktree / "hello.py").write_text("VALUE = 'old'\n", encoding="utf-8")
    tools = EngineeringTools(EngineeringSandbox(worktree, tmp_path / "out"), tmp_path / "out" / "audit.jsonl")
    actions = [
        {"action": "read_file", "args": {"path": "hello.py"}},
        {"action": "read_file", "args": {"path": "hello.py"}},
    ]
    result = EngineeringAgent(ScriptedAI(actions), tools, max_read_only_steps=2).run(objective="repair")
    assert result["state"] == "abstain"
    assert result["reason"] == "engineering_read_only_budget_exhausted"
    assert len(result["events"]) == 2


def test_engineering_agent_settles_repeated_action_loop(tmp_path: Path) -> None:
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    (worktree / "hello.py").write_text("VALUE = 'old'\n", encoding="utf-8")
    tools = EngineeringTools(EngineeringSandbox(worktree, tmp_path / "out"), tmp_path / "out" / "audit.jsonl")
    actions = [{"action": "inspect_files", "args": {"path": "."}}] * 3
    result = EngineeringAgent(ScriptedAI(actions), tools, max_repeated_action=3).run(objective="repair")
    assert result["state"] == "abstain"
    assert result["reason"] == "engineering_repeated_action_budget_exhausted"
    assert len(result["events"]) == 2


def test_autonomous_stage_runner_repairs_then_retries_without_operator(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    worktree = tmp_path / "worktree"
    subprocess.run(["git", "worktree", "add", "--detach", str(worktree), "HEAD"], cwd=repo, check=True, capture_output=True)
    calls = {"count": 0}

    def stage_runner(idea, stage, context):
        calls["count"] += 1
        if calls["count"] == 1:
            return {"state": "runtime_failed", "reason": "generated code failed"}
        return {"state": "completed", "stage": stage}

    actions = [{"action": "done", "args": {"state": "completed", "reason": "repair materialized"}}]

    def factory(idea, stage, context):
        tools = EngineeringTools(EngineeringSandbox(worktree, tmp_path / "out"), tmp_path / "out" / "audit.jsonl")
        return EngineeringAgent(ScriptedAI(actions), tools)

    wrapped = AutonomousStageRunner(stage_runner, factory)
    result = wrapped({"idea_id": "a", "title": "repair"}, "static_check", {"run_id": "r"})
    assert result["state"] == "completed"
    assert result["engineering"]["state"] == "repaired_and_retried"
    assert calls["count"] == 2


def test_autonomous_stage_runner_absorbs_engineering_patch_into_idea(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    worktree = tmp_path / "worktree"
    subprocess.run(["git", "worktree", "add", "--detach", str(worktree), "HEAD"], cwd=repo, check=True, capture_output=True)
    calls = {"count": 0}

    def stage_runner(idea, stage, context):
        calls["count"] += 1
        return {"state": "requires_code_patch"} if calls["count"] == 1 else {"state": "completed"}

    actions = [
        {"action": "apply_patch", "args": {"cwd": str(worktree), "diff": "diff --git a/hello.py b/hello.py\n--- a/hello.py\n+++ b/hello.py\n@@ -1 +1 @@\n-VALUE = 'old'\n+VALUE = 'new'\n"}},
        {"action": "done", "args": {"state": "completed"}},
    ]

    def factory(idea, stage, context):
        tools = EngineeringTools(EngineeringSandbox(worktree, tmp_path / "out"), tmp_path / "out" / "audit.jsonl")
        return EngineeringAgent(ScriptedAI(actions), tools)

    idea = {"idea_id": "novel-training-method", "title": "Novel method"}
    result = AutonomousStageRunner(stage_runner, factory)(idea, "static_check", {})
    assert result["state"] == "completed"
    assert len(idea["materialized_patches"]) == 1
    assert "VALUE = 'new'" in idea["materialized_patches"][0]["diff"]


def test_autonomous_stage_runner_preserves_resource_block(tmp_path: Path) -> None:
    calls = {"repair": 0}

    def stage_runner(idea, stage, context):
        return {"state": "blocked", "blocker_type": "resource_unavailable", "retryable": True}

    def factory(idea, stage, context):
        calls["repair"] += 1
        raise AssertionError("resource waits must not invoke engineering repair")

    wrapped = AutonomousStageRunner(stage_runner, factory)
    result = wrapped({"idea_id": "a"}, "gpu_smoke", {})
    assert result["blocker_type"] == "resource_unavailable"
    assert calls["repair"] == 0
