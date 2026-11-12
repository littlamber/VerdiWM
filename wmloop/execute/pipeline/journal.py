"""Resumable pipeline state and evidence components: journal."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from wmloop.control.research_state import ResearchJournal, build_research_state
from wmloop.execute.pipeline.io import _load_json, _sha256
from wmloop.execute.pipeline.types import AutonomousPipelineError
from wmloop.storage import atomic_write, checked_path


def _execution_evidence_refs(journal: ResearchJournal, compiled: Path) -> list[str]:
    refs = list(journal.read()["evidence_refs"])
    path = compiled / "queue" / "execution.json"
    if path.is_file() and not path.is_symlink():
        # Snapshot the mutable resume projection so a retry cannot overwrite
        # the evidence seen by an earlier decision. Individual verdict receipts
        # remain authoritative for scientific conclusions.
        # Preserve even an incomplete/corrupt execution file on an error path;
        # parsing it here could mask the original executor failure.
        payload = path.read_bytes()
        digest = _sha256(payload)
        snapshot = checked_path(
            journal.root / "execution-snapshots" / f"{digest}.json",
            code="AUTONOMOUS_PIPELINE_SNAPSHOT_PATH_INVALID",
            error=AutonomousPipelineError,
        )
        atomic_write(snapshot, payload)
        refs.append(str(snapshot))
    return list(dict.fromkeys(refs))


def _scheduler_template_path(evaluator: Path) -> Path:
    contract = _load_json(evaluator, "AUTONOMOUS_PIPELINE_EVALUATOR_INVALID")
    raw = contract.get("scheduler_template")
    if not isinstance(raw, str) or not raw:
        raise AutonomousPipelineError("AUTONOMOUS_PIPELINE_TEMPLATE_INVALID")
    path = Path(raw)
    if not path.is_absolute():
        path = evaluator.parent / path
    return checked_path(
        path, code="AUTONOMOUS_PIPELINE_TEMPLATE_INVALID", error=AutonomousPipelineError
    )


def _initial_research_state(evaluator: Path, budget: float, irg: Path | None) -> dict:
    contract = _load_json(evaluator, "AUTONOMOUS_PIPELINE_EVALUATOR_INVALID")
    template_path = _scheduler_template_path(evaluator)
    template = _load_json(template_path, "AUTONOMOUS_PIPELINE_TEMPLATE_INVALID")
    metrics = contract.get("metrics", [])
    names = (
        list(metrics)
        if isinstance(metrics, Mapping)
        else [
            row
            if isinstance(row, str)
            else str(row.get("name") or row.get("metric_id") or "")
            for row in metrics
            if isinstance(row, (str, Mapping))
        ]
    )
    return build_research_state(
        goal=str(template["objective"]),
        metrics=[name for name in names if name],
        budget_gpu_hours=budget,
        irg_ref=str(irg) if irg is not None else None,
        evidence_refs=[str(evaluator)],
        stop_conditions=[str(template["falsification_criterion"])],
    )
