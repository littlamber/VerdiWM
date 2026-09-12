from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from scripts.evaluate_psnr_ssim_smoke import (
    PsnrSsimError,
    evaluate,
    evaluate_video_rollouts,
    verify_receipt,
)


class PsnrSsimEvaluatorTests(unittest.TestCase):
    def _frames(self, root: Path, *, noisy: float = 0.0, shape: tuple[int, int] = (6, 8)) -> None:
        target = root / "target"
        prediction = root / "prediction"
        target.mkdir(parents=True)
        prediction.mkdir(parents=True)
        for index in range(2):
            frame = np.linspace(0.1, 0.9, shape[0] * shape[1], dtype=np.float32).reshape(shape)
            np.save(target / f"frame-{index}.npy", frame)
            np.save(prediction / f"frame-{index}.npy", np.clip(frame + noisy, 0, 1))

    def test_perfect_noisy_and_relative_verdicts(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._frames(root, noisy=0.05)
            contract = Path("configs/evaluators/wan22_droid_psnr_ssim_smoke_v1.json").resolve()
            noisy_output = root / "noisy-output"
            noisy = evaluate(predicted_dir=root / "prediction", ground_truth_dir=root / "target", output_root=noisy_output, contract_path=contract)
            self.assertEqual(noisy["verdict"], "MEASURED")

            self._frames(root / "perfect-input")
            perfect_input = root / "perfect-input"
            perfect = evaluate(predicted_dir=perfect_input / "prediction", ground_truth_dir=perfect_input / "target", output_root=root / "perfect-output", contract_path=contract)
            self.assertTrue(perfect["metrics"]["psnr_is_infinite"])
            self.assertAlmostEqual(float(perfect["metrics"]["ssim"]), 1.0)

            positive = evaluate(predicted_dir=perfect_input / "prediction", ground_truth_dir=perfect_input / "target", output_root=root / "positive-output", contract_path=contract, baseline_receipt=noisy_output / "psnr-ssim-receipt.json")
            self.assertEqual(positive["verdict"], "POSITIVE")
            null = evaluate(predicted_dir=root / "prediction", ground_truth_dir=root / "target", output_root=root / "null-output", contract_path=contract, baseline_receipt=noisy_output / "psnr-ssim-receipt.json")
            self.assertEqual(null["verdict"], "NULL")

            receipt = verify_receipt(root / "positive-output" / "psnr-ssim-receipt.json", contract_path=contract)
            self.assertEqual(receipt["verdict"], "POSITIVE")

    def test_rejects_shape_mismatch_and_input_output_overlap(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._frames(root)
            malformed = root / "malformed"
            malformed.mkdir()
            np.save(malformed / "frame-0.npy", np.zeros((2, 2), dtype=np.float32))
            np.save(malformed / "frame-1.npy", np.zeros((6, 8), dtype=np.float32))
            with self.assertRaisesRegex(PsnrSsimError, "FRAME_SHAPE_MISMATCH"):
                evaluate(predicted_dir=malformed, ground_truth_dir=root / "target", output_root=root / "bad-output")
            with self.assertRaisesRegex(PsnrSsimError, "OUTPUT_INPUT_OVERLAP"):
                evaluate(predicted_dir=root / "prediction", ground_truth_dir=root / "target", output_root=root)

    def test_receipt_rejects_input_drift(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._frames(root)
            result = evaluate(predicted_dir=root / "prediction", ground_truth_dir=root / "target", output_root=root / "output")
            receipt_path = root / "output" / "psnr-ssim-receipt.json"
            with (root / "target" / "frame-000.npy").open("ab") as handle:
                handle.write(b"drift")
            with self.assertRaisesRegex(PsnrSsimError, "RECEIPT_INPUT_DRIFT"):
                verify_receipt(receipt_path)

    def test_rejects_symlinked_input_and_output_paths(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._frames(root)
            prediction_link = root / "prediction-link"
            prediction_link.symlink_to(root / "prediction", target_is_directory=True)
            with self.assertRaisesRegex(PsnrSsimError, "PREDICTED_DIR_INVALID"):
                evaluate(predicted_dir=prediction_link, ground_truth_dir=root / "target", output_root=root / "link-input-output")

            output_link = root / "output-link"
            output_link.symlink_to(root / "missing-output")
            with self.assertRaisesRegex(PsnrSsimError, "OUTPUT_ROOT_EXISTS"):
                evaluate(predicted_dir=root / "prediction", ground_truth_dir=root / "target", output_root=output_link)

    def test_evaluates_paired_rollout_videos_and_keeps_inputs_verifiable(self) -> None:
        import imageio.v2 as imageio

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample = root / "rollouts" / "droid_ext2" / "sample-000"
            sample.mkdir(parents=True)
            target = [np.full((8, 10, 3), 100 + index, dtype=np.uint8) for index in range(3)]
            prediction = [frame.copy() for frame in target]
            prediction[1] = np.clip(prediction[1].astype(np.int16) + 8, 0, 255).astype(np.uint8)
            imageio.mimwrite(sample / "gt.mp4", target, fps=5)
            imageio.mimwrite(sample / "sa_wm_rollout.mp4", prediction, fps=5)
            contract = Path("configs/evaluators/wan22_droid_psnr_ssim_smoke_v1.json").resolve()

            result = evaluate_video_rollouts(
                rollout_root=root / "rollouts",
                output_root=root / "measurement",
                contract_path=contract,
            )

            self.assertEqual(result["verdict"], "MEASURED")
            self.assertEqual(result["sample_count"], 1)
            receipt = verify_receipt(Path(str(result["receipt_path"])), contract_path=contract)
            self.assertEqual(receipt["metrics"]["frame_count"], 3)
            self.assertTrue(Path(str(result["input_staging_root"])).is_dir())

    def test_video_pair_rejects_frame_count_mismatch(self) -> None:
        import imageio.v2 as imageio

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample = root / "rollouts" / "sample-000"
            sample.mkdir(parents=True)
            frame = np.zeros((8, 10, 3), dtype=np.uint8)
            imageio.mimwrite(sample / "gt.mp4", [frame, frame], fps=5)
            imageio.mimwrite(sample / "sa_wm_rollout.mp4", [frame], fps=5)
            with self.assertRaisesRegex(PsnrSsimError, "VIDEO_FRAME_COUNT_MISMATCH"):
                evaluate_video_rollouts(
                    rollout_root=root / "rollouts",
                    output_root=root / "measurement",
                )

    def test_video_pair_rejects_shape_mismatch(self) -> None:
        import imageio.v2 as imageio

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample = root / "rollouts" / "sample-000"
            sample.mkdir(parents=True)
            imageio.mimwrite(sample / "gt.mp4", [np.zeros((8, 16, 3), dtype=np.uint8)], fps=5, macro_block_size=1)
            imageio.mimwrite(sample / "sa_wm_rollout.mp4", [np.zeros((10, 16, 3), dtype=np.uint8)], fps=5, macro_block_size=1)
            with self.assertRaisesRegex(PsnrSsimError, "VIDEO_FRAME_SHAPE_MISMATCH"):
                evaluate_video_rollouts(
                    rollout_root=root / "rollouts",
                    output_root=root / "measurement",
                )


if __name__ == "__main__":
    unittest.main()
