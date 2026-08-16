from __future__ import annotations

import json
from pathlib import Path
import sys

from wmloop.cli import main


ROOT = Path(__file__).resolve().parents[1]


def _profile(tmp_path: Path) -> tuple[Path, Path, Path]:
    model = tmp_path / "model"
    data = tmp_path / "data"
    (model / "models").mkdir(parents=True)
    (model / "scripts").mkdir()
    (model / "models" / "ctrl_world.py").write_text("# marker\n")
    (model / "scripts" / "rollout_replay_traj.py").write_text("# marker\n")
    (model / "asset.bin").write_bytes(b"asset")
    data.mkdir()
    profile = tmp_path / "profile.json"
    profile.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "verdiwm-adapter-profile",
                "profile_id": "cli-test-profile",
                "aliases": ["cli-test"],
                "model_family": "ctrl_world",
                "capability_level": "L2",
                "execution_kind": "pipeline",
                "repo_markers": [
                    "models/ctrl_world.py",
                    "scripts/rollout_replay_traj.py",
                ],
                "goal_keywords": ["predict"],
                "evaluator_contract": "configs/onboarding/ctrl_world_predictive_probe_evaluator_v2.json",
                "probe_contract": None,
                "constitution_freeze": "configs/constitution/ctrl_world_predictive_quality_pilot_v2.freeze.json",
                "runtime_candidates": [sys.executable],
                "asset_bindings": [
                    {"parameter": "--asset", "candidates": ["{model}/asset.bin"]}
                ],
                "probe_imports": False,
            }
        )
    )
    return model, data, profile


def test_cli_run_status_cancel_and_reproduce_queue_only(
    tmp_path: Path, capsys
) -> None:
    model, data, profile = _profile(tmp_path)
    state_root = tmp_path / "state"
    assert (
        main(
            [
                "run",
                "--model",
                str(model),
                "--data",
                str(data),
                "--goal",
                "predict longer",
                "--budget",
                "1gpu-hour",
                "--adapter-profile",
                str(profile),
                "--campaign-id",
                "cli-1",
                "--state-root",
                str(state_root),
                "--queue-only",
            ]
        )
        == 0
    )
    queued = json.loads(capsys.readouterr().out)
    assert queued["status"] == "queued"

    assert main(["status", "cli-1", "--state-root", str(state_root)]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["adapter_profile"] == "cli-test-profile"

    assert main(["cancel", "cli-1", "--state-root", str(state_root)]) == 0
    cancelled = json.loads(capsys.readouterr().out)
    assert cancelled["status"] == "cancelled"
    assert "/dispatch/cancelled/" in cancelled["dispatch_ref"]

    assert (
        main(
            [
                "reproduce",
                "cli-1",
                "--state-root",
                str(state_root),
                "--queue-only",
            ]
        )
        == 0
    )
    reproduced = json.loads(capsys.readouterr().out)
    assert reproduced["status"] == "queued"
    assert reproduced["parent_campaign_id"] == "cli-1"
    assert reproduced["campaign_id"] in reproduced["execution"]["output_root"]


def test_cli_rejects_missing_budget(tmp_path: Path, capsys) -> None:
    model, data, profile = _profile(tmp_path)
    code = main(
        [
            "run",
            "--model",
            str(model),
            "--data",
            str(data),
            "--goal",
            "predict longer",
            "--budget",
            "not-a-budget",
            "--adapter-profile",
            str(profile),
            "--state-root",
            str(tmp_path / "state"),
            "--queue-only",
        ]
    )

    assert code == 2
    assert "BUDGET_INVALID" in capsys.readouterr().err


def test_cli_doctor_reports_source_tree_ready(capsys) -> None:
    assert main(["doctor", "--repo-root", str(ROOT)]) == 0
    report = json.loads(capsys.readouterr().out)

    assert report["artifact_type"] == "verdiwm-doctor-report"
    assert report["state"] == "ready"
    checks = {row["name"]: row for row in report["checks"]}
    assert checks["python_3_10"]["state"] == "pass"
    assert checks["adapter_profile"]["state"] == "pass"
    assert checks["mechanism_ontology"]["state"] == "pass"
    assert checks["public_example"]["state"] == "available"


def test_cli_doctor_fails_closed_for_incomplete_install(
    tmp_path: Path, capsys
) -> None:
    assert main(["doctor", "--repo-root", str(tmp_path)]) == 2
    report = json.loads(capsys.readouterr().out)

    assert report["state"] == "blocked"
    required = [row for row in report["checks"] if row["required"]]
    assert any(row["state"] == "fail" for row in required)
