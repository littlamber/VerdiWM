import json
from pathlib import Path

from verdi_core.autonomy import AutonomousPatchExecutor, AutonomousRepairLoop, CodeAuthor, PatchReviewer, WorkspaceManager, assess_replicates
from verdi_core.resources import GPU, GPUInventory
from verdi_core.stages import ExperimentStages
from verdi_core.human_eval import HumanVideoBatch, evaluate_labels


class FakeAI:
    def complete(self, *, role: str, prompt: str) -> str:
        return json.dumps({"patches": [{"path": "hello.py", "diff": "diff --git a/hello.py b/hello.py\n--- a/hello.py\n+++ b/hello.py\n@@ -1 +1 @@\n-old\n+new\n"}], "rationale": "test", "test_plan": ["python -m py_compile hello.py"]})


def test_code_author_and_reviewer_are_scoped(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    import subprocess
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "hello.py").write_text("old\n")
    subprocess.run(["git", "add", "hello.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "-c", "user.email=a@b", "-c", "user.name=t", "commit", "-qm", "init"], cwd=tmp_path, check=True)
    proposal = CodeAuthor(FakeAI()).propose(repository={"revision": "x"}, objective="x", failure={}, allowed_paths=["hello.py"])
    assert proposal is not None
    review = PatchReviewer().review(tmp_path, proposal, allowed_paths=["hello.py"])
    assert review.state == "approved"
    assert (tmp_path / "hello.py").read_text() == "old\n"


def test_reviewer_tests_patched_worktree(tmp_path: Path) -> None:
    import subprocess
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / "hello.py").write_text("old\n")
    subprocess.run(["git", "add", "hello.py"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.email=a@b", "-c", "user.name=t", "commit", "-qm", "init"], cwd=repo, check=True)
    proposal = CodeAuthor(FakeAI()).propose(repository={}, objective="x", failure={}, allowed_paths=["hello.py"])
    assert proposal is not None
    review = PatchReviewer().review(repo, proposal, allowed_paths=["hello.py"], tests=[["python", "-c", "from pathlib import Path; assert Path('hello.py').read_text() == 'new\\n'"]])
    assert review.state == "approved"


def test_workspace_receipt_and_effect_gate(tmp_path: Path) -> None:
    import subprocess
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / "x").write_text("x")
    subprocess.run(["git", "add", "x"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.email=a@b", "-c", "user.name=t", "commit", "-qm", "init"], cwd=repo, check=True)
    receipt = WorkspaceManager().create(repo, tmp_path / "wt")
    assert receipt["upstream_revision"]
    result = assess_replicates([0.2, 0.25, 0.21], practical_threshold=0.05, protected_ok=True)
    assert result["outcome"] == "confirmed_positive"


def test_autonomous_patch_executor_removes_isolated_worktree(tmp_path: Path) -> None:
    import subprocess
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / "hello.py").write_text("old\n")
    subprocess.run(["git", "add", "hello.py"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.email=a@b", "-c", "user.name=t", "commit", "-qm", "init"], cwd=repo, check=True)
    proposal = CodeAuthor(FakeAI()).propose(repository={}, objective="x", failure={}, allowed_paths=["hello.py"])
    assert proposal is not None
    result = AutonomousPatchExecutor().execute(repo, proposal, destination=tmp_path / "candidate", allowed_paths=["hello.py"], tests=[["python", "-m", "py_compile", "hello.py"]])
    assert result["state"] == "approved"
    assert not (tmp_path / "candidate").exists()


def test_gpu_selection_is_deterministic() -> None:
    inventory = GPUInventory([GPU(1, "a", 100, 0, 20), GPU(0, "a", 100, 0, 5)])
    assert inventory.select(1) == [0]


def test_stage_machine_blocks_skipped_stages_and_classifies_heldout() -> None:
    stages = ExperimentStages()
    assert stages.settle("full_train", success=True).state == "blocked"
    for stage in ("static_check", "environment_smoke", "gpu_smoke", "short_train", "replicate", "full_train", "heldout_evaluate"):
        assert stages.settle(stage, success=True).state == "settled"
    assert stages.classify_heldout([0.2, 0.24], practical_threshold=0.05, protected_ok=True).state == "confirmed_positive"


def test_human_video_labels_require_complete_batch() -> None:
    batch = HumanVideoBatch("b1", ("e1", "e2"), ("e1.mp4", "e2.mp4"), "heldout-human")
    assert evaluate_labels(batch, {"e1": {"success": True}})["outcome"] == "abstain"
    result = evaluate_labels(batch, {"e1": {"success": True}, "e2": False})
    assert result["success_rate"] == 0.5
