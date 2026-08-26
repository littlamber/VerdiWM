"""Bounded, model-agnostic code authoring and experiment evidence gates.

The kernel may ask an OpenAI-compatible provider for a patch, but it never
trusts the response blindly: scope, patch applicability, tests and evidence
must pass before a candidate can be promoted.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import subprocess
import tempfile
import shlex
from typing import Any, Iterable

from .contracts import canonical_digest
from .runtime import AIProvider


@dataclass(frozen=True)
class PatchProposal:
    patches: tuple[dict[str, str], ...]
    rationale: str
    test_plan: tuple[str, ...]
    digest: str


@dataclass(frozen=True)
class PatchReview:
    state: str
    reasons: tuple[str, ...]
    test_results: tuple[dict[str, Any], ...]
    proposal_digest: str | None = None


class CodeAuthor:
    """Turn a bounded failure receipt into a structured patch proposal."""

    def __init__(self, ai: AIProvider | None):
        self.ai = ai

    def propose(self, *, repository: dict[str, Any], objective: str, failure: dict[str, Any], allowed_paths: Iterable[str]) -> PatchProposal | None:
        if self.ai is None:
            return None
        prompt = json.dumps({
            "repository": repository,
            "objective": objective,
            "failure": failure,
            "allowed_paths": sorted(set(allowed_paths)),
            "instruction": "Return JSON object with patches [{path,diff}], rationale, and test_plan. Use unified diffs only and never modify files outside allowed_paths.",
        }, sort_keys=True)
        try:
            value = json.loads(self.ai.complete(role="code_author", prompt=prompt))
            patches = tuple({"path": str(p["path"]), "diff": str(p["diff"])} for p in value["patches"])
            if not all(p["path"] and p["diff"] for p in patches):
                raise ValueError("empty patch")
            allowed = set(allowed_paths)
            if any(not _safe_relative_path(p["path"]) or p["path"] not in allowed for p in patches):
                raise ValueError("patch path is outside allowed scope")
            payload = {"patches": patches, "rationale": str(value.get("rationale", "")), "test_plan": value.get("test_plan", [])}
            return PatchProposal(patches, payload["rationale"], tuple(map(str, payload["test_plan"])), canonical_digest(payload))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError, AttributeError):
            return None


class PatchReviewer:
    """Apply no changes; validate a proposal in a temporary index/worktree."""

    def review(self, repository: Path, proposal: PatchProposal, *, allowed_paths: Iterable[str], tests: list[list[str]] | None = None) -> PatchReview:
        reasons: list[str] = []
        allowed = set(allowed_paths)
        if any(not _safe_relative_path(p["path"]) or p["path"] not in allowed for p in proposal.patches):
            reasons.append("patch_scope_violation")
        if reasons:
            return PatchReview("abstain", tuple(reasons), (), proposal.digest)
        with tempfile.TemporaryDirectory(prefix="verdi-patch-") as temp:
            temp_root = Path(temp)
            patch_file = temp_root / "candidate.patch"
            patch_file.write_text("\n".join(p["diff"] for p in proposal.patches), encoding="utf-8")
            check = subprocess.run(["git", "apply", "--check", str(patch_file)], cwd=repository, capture_output=True, text=True)
            if check.returncode:
                return PatchReview("abstain", ("patch_does_not_apply", check.stderr[-2000:]), (), proposal.digest)
            review_worktree = temp_root / "worktree"
            subprocess.run(["git", "worktree", "add", "--detach", str(review_worktree), "HEAD"], cwd=repository, check=True, capture_output=True, text=True)
            results: list[dict[str, Any]] = []
            try:
                applied = subprocess.run(["git", "apply", str(patch_file)], cwd=review_worktree, capture_output=True, text=True)
                if applied.returncode:
                    return PatchReview("abstain", ("patch_worktree_apply_failed", applied.stderr[-2000:]), (), proposal.digest)
                for command in tests or []:
                    completed = subprocess.run(command, cwd=review_worktree, capture_output=True, text=True, timeout=1800, check=False)
                    results.append({"command": command, "returncode": completed.returncode, "stderr": completed.stderr[-2000:]})
                    if completed.returncode:
                        reasons.append("test_failed")
            finally:
                subprocess.run(["git", "worktree", "remove", "--force", str(review_worktree)], cwd=repository, check=False, capture_output=True)
        return PatchReview("approved" if not reasons else "abstain", tuple(reasons), tuple(results), proposal.digest)


class WorkspaceManager:
    """Create isolated detached worktrees and emit immutable receipts."""

    def create(self, repository: Path, destination: Path) -> dict[str, Any]:
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite worktree: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
        subprocess.run(["git", "worktree", "add", "--detach", str(destination), revision], cwd=repository, check=True, capture_output=True, text=True)
        return {"repository": str(repository), "worktree": str(destination), "upstream_revision": revision, "receipt_digest": canonical_digest({"repository": str(repository), "worktree": str(destination), "upstream_revision": revision})}


class AutonomousPatchExecutor:
    """Apply an AI proposal only inside a fresh worktree and run its tests."""

    def execute(self, repository: Path, proposal: PatchProposal, *, destination: Path, allowed_paths: Iterable[str] | None = None, tests: list[list[str]] | None = None, resume: Any | None = None) -> dict[str, Any]:
        allowed = set(allowed_paths or (patch["path"] for patch in proposal.patches))
        if any(not _safe_relative_path(patch["path"]) or patch["path"] not in allowed for patch in proposal.patches):
            return {"state": "abstain", "reason": "patch_scope_violation", "proposal_digest": proposal.digest}
        receipt = WorkspaceManager().create(repository, destination)
        patch_receipts = []
        try:
            for patch in proposal.patches:
                patch_file = destination / ".verdi-candidate.patch"
                patch_file.write_text(patch["diff"], encoding="utf-8")
                applied = subprocess.run(["git", "apply", str(patch_file)], cwd=destination, capture_output=True, text=True, check=False)
                if applied.returncode:
                    return {**receipt, "state": "abstain", "reason": "patch_apply_failed", "stderr": applied.stderr[-2000:]}
                patch_file.unlink(missing_ok=True)
                patch_receipts.append({"path": patch["path"], "digest": canonical_digest(patch["diff"])})
            results = []
            for command in tests or list(proposal.test_plan):
                argv = command if isinstance(command, list) else shlex.split(str(command))
                completed = subprocess.run(argv, cwd=destination, capture_output=True, text=True, timeout=1800, check=False)
                results.append({"command": command, "returncode": completed.returncode, "stderr": completed.stderr[-2000:]})
                if completed.returncode:
                    return {**receipt, "state": "abstain", "reason": "test_failed", "patches": patch_receipts, "tests": results}
            output = {**receipt, "state": "approved", "patches": patch_receipts, "tests": results, "continuation": "permitted"}
            if resume is not None:
                output["resumed"] = resume(destination, output)
            return output
        finally:
            if destination.exists():
                subprocess.run(["git", "worktree", "remove", "--force", str(destination)], cwd=repository, check=False, capture_output=True)


class AutonomousRepairLoop:
    """Automatically repair an isolated candidate, then hand it back to the worker."""

    def __init__(self, author: CodeAuthor, reviewer: PatchReviewer | None = None, executor: AutonomousPatchExecutor | None = None):
        self.author = author
        self.reviewer = reviewer or PatchReviewer()
        self.executor = executor or AutonomousPatchExecutor()

    def run(self, repository: Path, *, objective: str, failure: dict[str, Any], allowed_paths: Iterable[str], destination: Path, tests: list[list[str]] | None = None, resume: Any | None = None) -> dict[str, Any]:
        proposal = self.author.propose(repository={"repository": str(repository)}, objective=objective, failure=failure, allowed_paths=allowed_paths)
        if proposal is None:
            return {"state": "abstain", "reason": "no_patch_proposal"}
        review = self.reviewer.review(repository, proposal, allowed_paths=allowed_paths, tests=tests)
        if review.state != "approved":
            return {"state": "abstain", "reason": "patch_review_failed", "review": review.__dict__}
        receipt = self.executor.execute(repository, proposal, destination=destination, allowed_paths=allowed_paths, tests=tests, resume=resume)
        return {"proposal": proposal.__dict__, "review": review.__dict__, "execution": receipt, "state": receipt.get("state", "abstain")}


def assess_replicates(deltas: Iterable[float], *, practical_threshold: float | None, protected_ok: bool) -> dict[str, Any]:
    """Classify replicated held-out effects with a conservative normal CI.

    A threshold must be frozen by the metric/evaluator review; without it the
    kernel deliberately abstains instead of inventing scientific significance.
    """
    values = [float(v) for v in deltas]
    if practical_threshold is None:
        return {"outcome": "abstain", "reason": "practical_threshold_not_frozen", "replicates": len(values)}
    if len(values) < 2:
        return {"outcome": "abstain", "reason": "requires_two_independent_replicates", "replicates": len(values)}
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / max(1, len(values) - 1)
    stderr = math.sqrt(variance / len(values))
    low, high = mean - 1.96 * stderr, mean + 1.96 * stderr
    threshold = abs(float(practical_threshold))
    if not protected_ok:
        outcome = "harmful"
    elif low > threshold:
        outcome = "confirmed_positive"
    elif high < -threshold:
        outcome = "harmful"
    elif high < threshold and low > -threshold:
        outcome = "null"
    else:
        outcome = "abstain"
    return {"outcome": outcome, "mean_delta": mean, "stderr": stderr, "ci95": [low, high], "threshold": threshold, "replicates": len(values), "protected_ok": protected_ok}


def _safe_relative_path(path: str) -> bool:
    value = Path(path)
    return not value.is_absolute() and ".." not in value.parts and str(value) == path
