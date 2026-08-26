from pathlib import Path

from verdi_core.autonomy import AutonomousRepairLoop, CodeAuthor
from verdi_core.workers import ExperimentTask, RepairingWorker


class RepairAI:
    def complete(self, *, role: str, prompt: str) -> str:
        return '{"patches": [{"path": "hello.py", "diff": "diff --git a/hello.py b/hello.py\\n--- a/hello.py\\n+++ b/hello.py\\n@@ -1 +1 @@\\n-old\\n+new\\n"}], "rationale": "fix", "test_plan": []}'


class FailingWorker:
    def execute(self, task: ExperimentTask):
        if "worktree" not in task.hypothesis:
            raise RuntimeError("initial failure")
        return {"state": "resumed", "worktree": task.hypothesis["worktree"]}


def test_repairing_worker_resumes_after_verified_patch(tmp_path: Path) -> None:
    import subprocess
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / "hello.py").write_text("old\n")
    subprocess.run(["git", "add", "hello.py"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.email=a@b", "-c", "user.name=t", "commit", "-qm", "init"], cwd=repo, check=True)
    worker = RepairingWorker(FailingWorker(), AutonomousRepairLoop(CodeAuthor(RepairAI())), repository=repo, allowed_paths=["hello.py"], destination_root=tmp_path / "repairs")
    result = worker.execute(ExperimentTask("job", {"objective": "repair"}, "heldout"))
    assert result["resumed_result"]["state"] == "resumed"
