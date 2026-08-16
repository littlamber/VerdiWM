from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3

from wmloop.archive.store import ArchiveStore
from wmloop.experiments.ctrl_world_settlement_import import (
    import_ctrl_world_settlements,
)
from wmloop.experiments.evidence_graph import write_evidence_graph


def _write(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, sort_keys=True).encode()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, tuple[Path, ...]]:
    input_root = tmp_path / "runs"
    case = input_root / "fshc" / "heldout"
    result = case / "routing" / "a9" / "result.json"
    result_hash = _write(result, {"artifact_type": "result", "state": "ready"})
    report = case / "heldout-evaluation-report.json"
    report_hash = _write(
        report,
        {
            "artifact_type": "heldout-report",
            "experiment_id": "ctrl-world-test-heldout",
            "input_receipts": [{"path": str(result), "sha256": result_hash}],
        },
    )
    plan = tmp_path / "Ctrl-World" / "configs" / "plan.json"
    plan_hash = _write(plan, {"artifact_type": "plan"})
    _write(
        case / "receipts" / "routing-a9.json",
        {"artifact_type": "runtime-receipt", "state": "completed"},
    )
    _write(
        case / "SETTLEMENT.json",
        {
            "schema_version": 1,
            "artifact_type": "verdiwm-mechanism-experiment-settlement",
            "state": "settled_not_promoted",
            "experiment_id": "ctrl-world-test-heldout",
            "candidate": "a9",
            "decision": "do_not_promote",
            "decision_reason": "held-out quality regressed",
            "evidence": {
                "evaluation_plan": str(plan),
                "evaluation_plan_sha256": plan_hash,
                "evaluation_report": str(report),
                "evaluation_report_sha256": report_hash,
            },
            "causal_interpretation": {
                "dominant_failure_mode": "backbone_drift",
                "confidence_limit": "one context only",
            },
        },
    )
    return input_root, (input_root, tmp_path / "Ctrl-World")


def test_settlement_import_is_dry_runnable_idempotent_and_exploratory(
    tmp_path: Path,
) -> None:
    input_root, allowed_roots = _fixture(tmp_path)
    archive_db = tmp_path / "state" / "archive.db"
    cas_root = tmp_path / "state" / "cas"
    output_root = tmp_path / "state" / "settlement-import"

    planned = import_ctrl_world_settlements(
        input_root=input_root,
        archive_db=archive_db,
        cas_root=cas_root,
        output_root=output_root,
        allowed_roots=allowed_roots,
        dry_run=True,
    )
    assert planned["state"] == "planned"
    assert planned["settlement_count"] == 1
    assert not archive_db.exists()

    imported = import_ctrl_world_settlements(
        input_root=input_root,
        archive_db=archive_db,
        cas_root=cas_root,
        output_root=output_root,
        allowed_roots=allowed_roots,
    )
    assert len(imported["imported_trial_ids"]) == 1
    assert ArchiveStore(archive_db).archive_statistics()["settled_trials"] == 1
    assert ArchiveStore(archive_db).list_cells() == []

    connection = sqlite3.connect(archive_db)
    settlement = json.loads(
        connection.execute("SELECT settlement_json FROM trials").fetchone()[0]
    )
    connection.close()
    assert settlement["evidence_scope"] == "exploratory"

    repeated = import_ctrl_world_settlements(
        input_root=input_root,
        archive_db=archive_db,
        cas_root=cas_root,
        output_root=output_root,
        allowed_roots=allowed_roots,
    )
    assert repeated["imported_trial_ids"] == []
    assert len(repeated["already_present_trial_ids"]) == 1

    graph = write_evidence_graph(
        input_root=output_root,
        output_root=tmp_path / "state" / "evidence-graph",
        archive_db=archive_db,
    )
    kinds = {node["kind"] for node in graph["nodes"]}
    assert "exploratory_evidence" in kinds
    assert "verified_evidence" not in kinds
