"""Immutable research plan versions with a project-local latest snapshot."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from wmloop.control.research_request import ResearchRequestError, load_research_plan, write_research_plan
from wmloop.storage import atomic_write, checked_path, exclusive_file_lock


def write_default_research_plan(plan: Mapping[str, object], project_root: Path) -> Path:
    """Preserve old plans before atomically selecting the newly compiled plan."""
    try:
        return _write_default_research_plan(plan, project_root)
    except OSError as exc:
        raise ResearchRequestError("RESEARCH_PLAN_OUTPUT_INVALID") from exc


def _write_default_research_plan(plan: Mapping[str, object], project_root: Path) -> Path:
    root = checked_path(project_root / ".verdiwm", code="RESEARCH_PLAN_OUTPUT_INVALID", error=ResearchRequestError)
    latest = checked_path(root / "research-plan.json", code="RESEARCH_PLAN_OUTPUT_INVALID", error=ResearchRequestError)
    history = checked_path(root / "research-plans", code="RESEARCH_PLAN_OUTPUT_INVALID", error=ResearchRequestError)
    # The normal writer validates schema and the outside-input boundary before
    # creating directories. Preparing an immutable candidate needs no latest lock.
    digest = str(plan.get("plan_digest", ""))
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ResearchRequestError("RESEARCH_PLAN_DIGEST_MISMATCH")
    candidate_path = checked_path(history / f"{digest}.json", code="RESEARCH_PLAN_OUTPUT_INVALID", error=ResearchRequestError)
    candidate = write_research_plan(plan, candidate_path)
    with exclusive_file_lock(root / "research-plan.lock"):
        if latest.exists():
            previous = load_research_plan(latest)
            previous_path = checked_path(history / f"{previous['plan_digest']}.json", code="RESEARCH_PLAN_OUTPUT_INVALID", error=ResearchRequestError)
            write_research_plan(previous, previous_path)
        atomic_write(latest, candidate.read_bytes())
    return candidate
