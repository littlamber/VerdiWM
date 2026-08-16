from __future__ import annotations

import hashlib
import json
from pathlib import Path

from wmloop.control.method_candidate_compiler import compile_method_candidates


def test_compiler_adds_executable_candidate_and_keeps_negative_history(
    tmp_path: Path,
) -> None:
    required = tmp_path / "hook.py"
    required.write_text("# admitted hook\n", encoding="utf-8")
    catalog = _catalog(
        tmp_path / "catalog.json",
        required_file=required,
        historical_ids=[],
    )
    settlement = _settlements(tmp_path / "settlements")
    batch = {"candidates": []}

    report = compile_method_candidates(
        batch=batch,
        catalog_path=catalog,
        diagnostic_probe={"failure_signatures": ["horizon_drift"]},
        settlement_manifest=settlement,
        literature_methods={
            "records": [
                {
                    "candidate_id": "paper-known",
                    "primitive_reference": "first_frame_anchor",
                    "required_hook": "H1",
                    "estimated_gpu_hours": 0.01,
                    "execution_authority": "ranking_only",
                    "state": "validated",
                }
            ]
        },
    )

    assert report["state"] == "ready"
    assert report["compiled_candidate_count"] == 1
    assert report["historical_constraint_count"] == 1
    assert report["compiled_candidates"][0]["matched_literature_candidate_ids"] == [
        "paper-known"
    ]
    assert batch["candidates"][0]["candidate_id"] == "repair-v1"
    assert batch["candidates"][0]["retrieval_prior"] == 0.65


def test_historical_not_promoted_candidate_is_blocked(tmp_path: Path) -> None:
    required = tmp_path / "hook.py"
    required.write_text("# admitted hook\n", encoding="utf-8")
    catalog = _catalog(
        tmp_path / "catalog.json",
        required_file=required,
        historical_ids=["failed-route"],
    )

    report = compile_method_candidates(
        batch={"candidates": []},
        catalog_path=catalog,
        diagnostic_probe={"failure_signatures": ["horizon_drift"]},
        settlement_manifest=_settlements(tmp_path / "settlements"),
    )

    assert report["state"] == "blocked"
    assert report["compiled_candidate_count"] == 0
    blocker = report["capability_gaps"][0]["blockers"][0]
    assert blocker["code"] == "HISTORICAL_CANDIDATE_NOT_PROMOTED"


def test_unmaterialized_literature_method_is_an_explicit_gap(tmp_path: Path) -> None:
    required = tmp_path / "hook.py"
    required.write_text("# admitted hook\n", encoding="utf-8")
    report = compile_method_candidates(
        batch={"candidates": []},
        catalog_path=_catalog(
            tmp_path / "catalog.json",
            required_file=required,
            historical_ids=[],
        ),
        diagnostic_probe={"failure_signatures": ["horizon_drift"]},
        literature_methods={
            "records": [
                {
                    "candidate_id": "paper-new",
                    "primitive_reference": None,
                    "required_hook": "H3",
                    "estimated_gpu_hours": 0.2,
                    "execution_authority": "materialization_required",
                    "state": "validated",
                }
            ]
        },
    )

    gaps = [row for row in report["capability_gaps"] if row["candidate_id"] == "paper-new"]
    assert gaps[0]["blockers"] == [{"code": "PRIMITIVE_MATERIALIZATION_REQUIRED"}]


def test_required_file_hash_mismatch_blocks_compilation(tmp_path: Path) -> None:
    required = tmp_path / "hook.py"
    original = b"# admitted hook\n"
    required.write_bytes(original)
    catalog = _catalog(
        tmp_path / "catalog.json",
        required_file=required,
        historical_ids=[],
    )
    payload = json.loads(catalog.read_text(encoding="utf-8"))
    payload["candidates"][0]["required_files"][0]["sha256"] = hashlib.sha256(
        original
    ).hexdigest()
    catalog.write_text(json.dumps(payload), encoding="utf-8")
    required.write_text("# drifted hook\n", encoding="utf-8")

    report = compile_method_candidates(
        batch={"candidates": []},
        catalog_path=catalog,
        diagnostic_probe={"failure_signatures": ["horizon_drift"]},
    )

    assert report["state"] == "blocked"
    assert report["compiled_candidate_count"] == 0
    blocker = report["capability_gaps"][0]["blockers"][0]
    assert blocker == {
        "code": "ADAPTER_CAPABILITY_FILE_HASH_MISMATCH",
        "name": "hook",
        "path": str(required),
    }


def _catalog(
    path: Path, *, required_file: Path, historical_ids: list[str]
) -> Path:
    candidate_template = {
        "candidate_id": "repair-v1",
        "hypothesis": "A bounded repair improves the declared long-horizon failure.",
        "selection_reason": "The adapter exposes the required hook and a bounded runtime recipe.",
        "falsification_criterion": "A missing hook, identity drift, or invalid runtime result rejects the repair.",
        "expected_gain": 0.5,
        "uncertainty": 0.5,
        "information_gain": 0.8,
        "novelty": 0.2,
        "retrieval_keys": {
            "failure_signatures": ["horizon_drift"],
            "primitive": "first_frame_anchor",
            "stage": "screen",
        },
        "stages": [
            {
                "stage": "screen",
                "command": ["python", "workload.py"],
                "working_directory": ".",
                "allowed_gpu_indices": [0],
                "estimated_gpu_hours": 0.01,
                "timeout_seconds": 60,
                "gpu_wait_seconds": 1,
                "sample_interval_seconds": 0.1,
                "result_path": "result.json",
                "artifacts": ["result.json"],
                "metric_gates": [
                    {
                        "metric": "ready",
                        "role": "primary",
                        "operator": "gte",
                        "threshold": 1.0,
                    }
                ],
                "environment": {},
                "cleanup_policy": "retain",
            }
        ],
    }
    payload = {
        "schema_version": 1,
        "artifact_type": "verdiwm-method-candidate-catalog",
        "catalog_id": "test-catalog-v1",
        "model_family": "test",
        "candidates": [
            {
                "candidate_id": "repair-v1",
                "primitive_reference": "first_frame_anchor",
                "source": "test",
                "mechanism_hypothesis": "A bounded repair improves the declared long-horizon failure.",
                "required_hooks": ["H1"],
                "failure_signatures": ["horizon_drift"],
                "applicability_conditions": ["The failure is observed."],
                "failure_boundaries": ["A null result is retained as terminal evidence."],
                "estimated_gpu_hours": 0.01,
                "historical_candidate_ids": historical_ids,
                "required_files": [
                    {"name": "hook", "path": str(required_file), "sha256": None}
                ],
                "candidate_template": candidate_template,
            }
        ],
        "capability_gaps": [],
        "claim_boundary": "Test catalog membership does not establish scientific promotion.",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _settlements(root: Path) -> Path:
    records = root / "records"
    records.mkdir(parents=True)
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artifact_type": "verdiwm-ctrl-world-settlement-import-manifest",
                "state": "completed",
            }
        ),
        encoding="utf-8",
    )
    (records / "failed.json").write_text(
        json.dumps(
            {
                "artifact_type": "verdiwm-imported-settlement-evidence",
                "trial_id": "trial-failed",
                "experiment_id": "experiment-failed",
                "candidate_id": "failed-route",
                "settlement_state": "settled",
                "verdict": "NOT_PROMOTED",
                "evidence_scope": "exploratory",
                "promotion_authorized": False,
                "source_settlement_sha256": "a" * 64,
                "claim_boundary": "One local screen rejected this exact route.",
            }
        ),
        encoding="utf-8",
    )
    return manifest
