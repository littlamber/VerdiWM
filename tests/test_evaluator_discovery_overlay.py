from __future__ import annotations

import json
from pathlib import Path

import pytest

from wmloop.control.evaluator_discovery import discover_evaluator_candidates
from wmloop.control.evaluator_overlay import EvaluatorOverlayError, materialize_evaluator_overlay


def test_discovery_reports_protocol_scope_without_freezing(tmp_path: Path) -> None:
    source = tmp_path / "source"
    candidate_dir = source / "configs" / "evaluation"
    candidate_dir.mkdir(parents=True)
    candidate = candidate_dir / "long_eval.json"
    candidate.write_text(
        json.dumps(
            {
                "role": "validation",
                "collection_root": "/mnt/old/data",
                "protocol": {"fps": 5, "num_video_frames": 300, "closed_loop_chunks": 8},
                "split_contract": {"source_split": "val", "episode_disjoint": True},
                "metrics": ["trajectory_accuracy"],
            }
        ),
        encoding="utf-8",
    )
    report = discover_evaluator_candidates(source)
    assert report["state"] == "candidates_available"
    assert report["selection_requires_confirmation"] is True
    assert report["selection"] is None
    row = report["candidates"][0]
    assert row["horizon_seconds"] == 60.0
    assert row["minute_level_supported"] is True
    assert row["split"] == "val"
    assert row["requires_user_confirmation"] is True
    assert report["selection_hint"]["candidate_id"] == row["candidate_id"]


def test_overlay_rewrites_source_model_and_data_without_mutating_candidate(tmp_path: Path) -> None:
    source = tmp_path / "source"
    model = tmp_path / "model"
    data = tmp_path / "data"
    source.mkdir()
    model.mkdir()
    data.mkdir()
    candidate = source / "eval.json"
    original = {
        "repo_root": "/old/robocoach",
        "source_root": "/old/robocoach",
        "checkpoint": "/old/checkpoints/model.pt",
        "data_root": "/old/video_latent",
    }
    candidate.write_text(json.dumps(original), encoding="utf-8")
    output = tmp_path / "verdi"
    receipt = materialize_evaluator_overlay(
        candidate,
        source_root=source,
        model_root=model,
        data_root=data,
        output_root=output,
    )
    assert receipt["state"] == "ready_for_confirmation"
    assert receipt["rewrites"]
    overlay = Path(str(receipt["overlay_path"]))
    payload = json.loads(overlay.read_text(encoding="utf-8"))
    assert payload["working_directory"] == str(source)
    assert payload["source_manifest"] == str(candidate)
    assert candidate.read_text(encoding="utf-8") == json.dumps(original)
    frozen = materialize_evaluator_overlay(
        candidate,
        source_root=source,
        model_root=model,
        data_root=data,
        output_root=tmp_path / "frozen",
        confirm=True,
    )
    assert frozen["state"] == "frozen"


def test_overlay_rejects_input_overlap(tmp_path: Path) -> None:
    source = tmp_path / "source"
    model = tmp_path / "model"
    data = tmp_path / "data"
    source.mkdir()
    model.mkdir()
    data.mkdir()
    candidate = source / "eval.json"
    candidate.write_text("{}", encoding="utf-8")
    with pytest.raises(EvaluatorOverlayError, match="EVALUATOR_OVERLAY_OUTPUT_OVERLAP"):
        materialize_evaluator_overlay(
            candidate,
            source_root=source,
            model_root=model,
            data_root=data,
            output_root=source / "overlay",
        )
