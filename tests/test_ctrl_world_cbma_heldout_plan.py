from __future__ import annotations

import json
from pathlib import Path

from scripts.freeze_ctrl_world_cbma_heldout_plan import freeze


def _write(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_freezes_multiscale_heldout_model_contracts(tmp_path: Path) -> None:
    official = tmp_path / "official.pt"
    official.write_bytes(b"official")
    experiments = []
    cells = []
    definitions = (
        ("b2", ["enable_signed_history_correction"]),
        ("b3", ["enable_signed_history_correction", "fshc_decoupled_route_calibration"]),
        ("c1", ["enable_multiscale_history_adapter", "multiscale_history_always_on"]),
        ("c2", ["enable_multiscale_history_adapter", "fshc_decoupled_route_calibration"]),
        ("c3", ["enable_multiscale_history_adapter", "fshc_decoupled_route_calibration"]),
    )
    for cell_id, flags in definitions:
        output = tmp_path / cell_id
        checkpoint = output / "checkpoint-64.pt"
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(cell_id.encode())
        receipt = _write(output / "receipt.json", {"state": "completed", "return_code": 0})
        experiments.append({"id": cell_id, "name": cell_id, "flags": flags})
        cells.append(
            {
                "cell_id": cell_id,
                "state": "completed",
                "return_code": 0,
                "output_dir": str(output),
                "receipt": str(receipt),
            }
        )
    training = _write(
        tmp_path / "training.json",
        {
            "artifact_type": "ctrl-world-fshc-ablation-plan",
            "state": "frozen_before_execution",
            "experiment_id": "training-v1",
            "baseline": {"checkpoint": str(official)},
            "experiments": experiments,
        },
    )
    sequence = _write(
        tmp_path / "sequence.json",
        {
            "artifact_type": "ctrl-world-fshc-ablation-sequence",
            "state": "completed",
            "experiment_id": "training-v1",
            "cells": cells,
        },
    )

    manifest = freeze(
        training_plan_path=training,
        sequence_path=sequence,
        output_path=tmp_path / "heldout-plan.json",
        evaluation_output_root=tmp_path / "evaluation",
    )

    plan = json.loads(Path(manifest["plan_path"]).read_text(encoding="utf-8"))
    models = {row["id"]: row for row in plan["models"]}
    assert models["b2"]["enable_signed_history_correction"] is True
    assert models["c1"]["enable_multiscale_history_adapter"] is True
    assert models["c1"]["multiscale_history_always_on"] is True
    assert plan["phases"]["routing"]["fshc_dose_mode"] == "normalized_mechanism"
    assert plan["promotion_gate"]["candidate"] == "c3"
