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
