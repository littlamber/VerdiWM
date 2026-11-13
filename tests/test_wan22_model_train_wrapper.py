from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.run_wan22_droid_model_train import Wan22TrainError, run


@pytest.mark.parametrize("execution", [None, "0", "false", "true", "1"])
def test_training_bridge_rejects_before_launch_or_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, execution: str | None
) -> None:
    source = tmp_path / "source"
    script = source / "scripts" / "training" / "train_world_model.py"
    script.parent.mkdir(parents=True)
    script.write_text("# external training entrypoint\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "weights.bin").write_bytes(b"original checkpoint")
    output = tmp_path / "run"
    receipt = output / "receipts" / "checkpoint-receipt.json"
    manifest = tmp_path / "model-run.json"
    manifest.write_text(
        json.dumps(
            {
                "model_run_id": "train-test",
                "source": {
                    "source_root": str(source),
                    "model_root": str(checkpoint),
                    "asset_bindings": {"--checkpoint": str(checkpoint)},
                },
                "runtime": {"python": sys.executable},
                "output": {"root": str(output)},
                "training": {
                    "scale_plan": {"planned_steps": 1, "seed": 7},
                    "checkpoint_receipt": str(receipt),
                },
            }
        ),
        encoding="utf-8",
    )
    original = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    if execution is None:
        monkeypatch.delenv("VERDIWM_EXECUTE_WAN22", raising=False)
    else:
        monkeypatch.setenv("VERDIWM_EXECUTE_WAN22", execution)
    expected = (
        "WAN22_TRAIN_BUDGET_OUTPUT_BINDING_UNSUPPORTED"
        if execution == "1"
        else "WAN22_TRAIN_EXECUTION_DISABLED"
    )
    with patch("subprocess.run") as launched:
        with pytest.raises(Wan22TrainError, match=expected) as error:
            run(manifest)
        launched.assert_not_called()
    if execution == "1":
        assert "training.scale_plan" in str(error.value)
        assert "output.root" in str(error.value)
        assert "checkpoint_receipt does not identify a trained checkpoint" in str(error.value)
    assert not output.exists()
    assert not receipt.exists()
    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == original


@pytest.mark.parametrize("raw", [b"not json", b"[]", b"\xff"])
def test_training_bridge_reports_invalid_manifest_without_launch(tmp_path: Path, raw: bytes) -> None:
    manifest = tmp_path / "model-run.json"
    manifest.write_bytes(raw)
    with patch("subprocess.run") as launched:
        with pytest.raises(Wan22TrainError, match="WAN22_MODEL_RUN_MANIFEST_INVALID"):
            run(manifest)
        launched.assert_not_called()
    assert list(tmp_path.iterdir()) == [manifest]
