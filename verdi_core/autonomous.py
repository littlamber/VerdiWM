"""Convenience composition for a fully autonomous campaign."""

from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any, Callable

from .campaign import CampaignSupervisor, RepairRunner, Replanner, StageRunner
from .engineering import AutonomousStageRunner, EngineeringAgent, EngineeringSandbox, EngineeringTools
from .runtime import AIProvider


def autonomous_campaign(
    state_root: Path,
    *,
    model_id: str,
    stage_runner: StageRunner,
    ai: AIProvider,
    worktree_root: Path,
    output_root: Path,
    repository: Path | None = None,
    readable_roots: tuple[Path, ...] = (),
    replanner: Replanner | None = None,
    allowed_gpus: tuple[int, ...] = (),
    policy: Any | None = None,
) -> CampaignSupervisor:
    """Build a campaign where the AI owns engineering repair decisions.

    The caller only supplies the model adapter stage runner and starts the
    supervisor.  Every repair gets a fresh sandbox and an audit ledger.
    """
    worktree_root = Path(worktree_root).resolve()
    output_root = Path(output_root).resolve()
    if repository is not None:
        repository = Path(repository).resolve()
        if repository == worktree_root or repository in worktree_root.parents or worktree_root in repository.parents:
            raise ValueError("worktree_root must be outside the source repository")
        if repository == output_root or repository in output_root.parents or output_root in repository.parents:
            raise ValueError("output_root must be outside the source repository")

    def factory(idea: dict[str, Any], stage: str, context: dict[str, Any]) -> EngineeringAgent:
        run_id = str(context.get("run_id", "run"))
        idea_id = str(idea.get("idea_id", "idea"))
        repair_index = int(context.get("engineering_repairs", 0))
        sandbox_name = stage if repair_index == 0 else f"{stage}-repair-{repair_index}"
        sandbox_root = worktree_root / run_id / idea_id / sandbox_name
        if repository is not None:
            # Every repair gets a detached worktree.  The source checkout is
            # read-only from the kernel's perspective and is never patched.
            if not sandbox_root.exists():
                sandbox_root.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(
                    ["git", "worktree", "add", "--detach", str(sandbox_root), "HEAD"],
                    cwd=Path(repository).resolve(), capture_output=True, text=True,
                    check=True,
                )
        else:
            sandbox_root.mkdir(parents=True, exist_ok=True)
        sandbox = EngineeringSandbox(
            sandbox_root,
            output_root / run_id / idea_id / stage,
            readable_roots=tuple(Path(path).resolve() for path in readable_roots),
            allowed_gpus=allowed_gpus,
        )
        tools = EngineeringTools(sandbox, sandbox.output_root / "tool-audit.jsonl")
        return EngineeringAgent(ai, tools)

    def repair(idea: dict[str, Any], stage: str, context: dict[str, Any], failure: dict[str, Any]) -> dict[str, Any] | None:
        agent = factory(idea, stage, context)
        result = agent.run(objective=f"Repair {stage} for {idea.get('idea_id', 'idea')}", context={"failure": failure, "idea": idea})
        if result.get("state") not in {"completed", "approved", "settled"}:
            return {"state": "abstain", "engineering": result}
        return stage_runner(idea, stage, {**context, "engineering_receipt": result, "repaired": True})

    supervisor = CampaignSupervisor(
        state_root,
        model_id=model_id,
        stage_runner=AutonomousStageRunner(stage_runner, factory),
        repair_runner=repair,
        replanner=replanner,
        policy=policy,
    )
    return supervisor
