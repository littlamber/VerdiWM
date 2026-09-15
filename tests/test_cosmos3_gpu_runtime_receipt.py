from __future__ import annotations

import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import imageio.v3 as iio
import numpy as np

from wmloop.control.cosmos3_gpu_runtime_receipt import (
    Cosmos3GpuRuntimeReceiptError,
    build_cosmos3_gpu_runtime_receipt,
)

@unittest.skipUnless(shutil.which("ffprobe"), "ffprobe required")
class Cosmos3GpuRuntimeReceiptTests(unittest.TestCase):
    def test_builds_receipt_and_rejects_missing_gpu_activity(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = _run_fixture(root)
            samples = root / "samples.jsonl"
            samples.write_text(json.dumps({"gpu_uuid": "GPU-test", "used_memory_mib": 321}) + "\n")
            receipt = build_cosmos3_gpu_runtime_receipt(
                run_root=run, gpu_samples_path=samples, output_root=root / "receipt",
                expected_gpu_uuid="GPU-test",
            )
            self.assertEqual(receipt["action_shape"], [16, 10])
            self.assertEqual(receipt["physical_gpu"]["peak_used_memory_mib"], 321)

            samples.write_text(json.dumps({"gpu_uuid": "GPU-test", "used_memory_mib": 0}) + "\n")
            with self.assertRaisesRegex(Cosmos3GpuRuntimeReceiptError, "NO_ACTIVE_GPU_SAMPLE"):
                build_cosmos3_gpu_runtime_receipt(
                    run_root=run, gpu_samples_path=samples, output_root=root / "rejected"
                )


def _run_fixture(root: Path) -> Path:
    run = root / "run"
    output = run / "robotics_action_cond_chunk_00"
    inputs = run / "inputs"
    output.mkdir(parents=True)
    inputs.mkdir()
    _write_video(output / "vision.mp4")
    (output / "sample_outputs.json").write_text(json.dumps({
        "status": "success", "args": {"model_mode": "forward_dynamics", "seed": 101}
    }))
    (inputs / "robotics_droid_action_chunk_00.json").write_text(json.dumps([[0.0] * 10] * 16))
    (run / "benchmark.json").write_text(json.dumps({
        "average": {"OmniInference.generate_batch": 1.25}
    }))
    return run


def _write_video(path: Path) -> None:
    frames = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    frames[1, :, :, 1] = 255
    iio.imwrite(path, frames, fps=2, plugin="FFMPEG")


if __name__ == "__main__":
    unittest.main()
