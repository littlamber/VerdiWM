from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import imageio.v2 as imageio
import numpy as np
import pytest

from scripts.run_wan22_droid_model_eval import Wan22EvalError, run


def _manifest(root: Path, contract: Path) -> Path:
    source = root / "source"
    (source / "scripts" / "evaluation").mkdir(parents=True)
    (source / "scripts" / "evaluation" / "run_sa_wm_eval_manifest.py").write_text("# test entrypoint\n")
    model = root / "checkpoint"
    model.mkdir()
    data = root / "data"
    data.mkdir()
    output = root / "run"
    payload = {
        "model_run_id": "model-run-test",
        "source": {
            "source_root": str(source),
            "model_root": str(model),
            "asset_bindings": {"--checkpoint": str(model), "--data-root": str(data)},
        },
        "runtime": {"python": "/usr/bin/python3"},
        "output": {"root": str(output)},
        "evaluation": {
            "evaluator_contract": str(contract),
            "evidence_receipt": str(output / "receipts" / "heldout-evidence-receipt.json"),
        },
    }
    path = root / "model-run.json"
    path.write_text(json.dumps(payload))
    return path


def test_wrapper_materializes_metrics_receipt_from_rollouts() -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        contract = Path("configs/evaluators/wan22_droid_psnr_ssim_smoke_v1.json").resolve()
        manifest = _manifest(root, contract)

        output = root / "run" / "evaluation" / "model-run-test" / "rollouts" / "droid_ext2" / "sample-000"
        output.mkdir(parents=True)
        frames = [np.full((8, 10, 3), 90 + i, dtype=np.uint8) for i in range(2)]
        imageio.mimwrite(output / "gt.mp4", frames, fps=5)
        imageio.mimwrite(output / "sa_wm_rollout.mp4", frames, fps=5)

        def fake_run(command, *, cwd, capture_output, text, check):
            return __import__("subprocess").CompletedProcess(command, 0, "ok", "")

        with patch.dict("os.environ", {"VERDIWM_EXECUTE_WAN22": "1"}, clear=False), patch(
            "scripts.run_wan22_droid_model_eval.subprocess.run", side_effect=fake_run
        ):
            receipt = run(manifest)

        assert receipt["verdict"] == "MEASURED"
        assert receipt["metrics"]["frame_count"] == 2
        saved = json.loads(Path(str(receipt["measurement_receipt"])).read_text())
        assert saved["artifact_type"] == "verdiwm-psnr-ssim-evidence-receipt"


def test_wrapper_blocks_unsupported_worldarena_contract_before_launch() -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        contract = Path("configs/evaluators/wan22_droid_verdi_contract_v1.json").resolve()
        manifest = _manifest(root, contract)
        with patch.dict("os.environ", {"VERDIWM_EXECUTE_WAN22": "1"}, clear=False), patch(
            "scripts.run_wan22_droid_model_eval.subprocess.run"
        ) as launched:
            with pytest.raises(Wan22EvalError, match="WAN22_EVAL_CONTRACT_UNSUPPORTED"):
                run(manifest)
            launched.assert_not_called()
