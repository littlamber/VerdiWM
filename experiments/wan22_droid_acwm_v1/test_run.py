import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from wmloop.wan22_droid import Wan22DroidError, build_sample_manifest, required_rollout_source_frames, validate_contract
from experiments.wan22_droid_acwm_v1.run import _estimated_gpu_hours, _gpu_free_memory_mib, _parser
from experiments.wan22_droid_acwm_v1.wan22_droid_runner import _assert_episode_disjoint, _chunk_anchor, _conditioning_for_mode, _load_adapter, _parser as runner_parser, _select_branch_index, _training_record_schedule, _validate_control_plane_training_contract, _validate_scheduled_training_coverage, _validate_stage_training_mode, _validate_validation_source_frames, _validation_panel_indices


class Wan22DroidContractTests(unittest.TestCase):
    def _dataset(self, root: Path) -> None:
        for split in ("train", "val"):
            (root / "annotation" / split).mkdir(parents=True)
            (root / "videos" / split / "episode-a").mkdir(parents=True)
            (root / "latent_videos" / split / "episode-a").mkdir(parents=True)
            (root / "videos" / split / "episode-a" / "wrist.mp4").write_bytes(b"v")
            (root / "latent_videos" / split / "episode-a" / "wrist.pt").write_bytes(b"l")
            payload = {
                "episode_id": "episode-a", "split": split, "video_length": 152,
                "video_path": f"videos/{split}/episode-a/wrist.mp4",
                "latent_path": f"latent_videos/{split}/episode-a/wrist.pt",
                "processed_fps": 5, "action": [[0.0] * 7 for _ in range(152)],
                "proprio": [[0.0] * 14 for _ in range(152)],
            }
            (root / "annotation" / split / "episode-a.json").write_text(json.dumps(payload), encoding="utf-8")

    def test_manifest_windows_are_temporally_bound(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); self._dataset(root)
            manifest = build_sample_manifest(root, "val")
            self.assertEqual(manifest["record_count"], 1)
            self.assertEqual(manifest["records"][0]["horizon_frames"], 150)
            self.assertEqual(manifest["records"][0]["action_dim"], 7)
            self.assertEqual(manifest["rollout_source_frames_required"], 152)
            self.assertEqual(manifest["records"][0]["source_frame_count"], 152)

    def test_manifest_excludes_causal_tail_short_episodes(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); self._dataset(root)
            for split in ("train", "val"):
                path = root / "annotation" / split / "episode-a.json"
                payload = json.loads(path.read_text())
                payload["video_length"] = 151
                payload["action"] = payload["action"][:151]
                payload["proprio"] = payload["proprio"][:151]
                path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(Wan22DroidError, "DROID_NO_TARGET_WINDOWS:val"):
                build_sample_manifest(root, "val")

    def test_rollout_source_frame_requirement_is_explicit(self):
        self.assertEqual(required_rollout_source_frames(horizon_frames=150, chunk_frames=45), 152)
        self.assertEqual(required_rollout_source_frames(horizon_frames=180, chunk_frames=45), 180)

    def test_validation_source_frame_preflight_rejects_exact_horizon(self):
        manifest = {
            "data_root": "/tmp/does-not-get-read",
            "records": [{"episode_id": "e0", "video_path": "missing.mp4", "source_frame_count": 150}],
        }
        with self.assertRaisesRegex(ValueError, "SOURCE_FRAMES_INSUFFICIENT"):
            _validate_validation_source_frames(manifest, [0], required_frames=152)

    def test_short_episode_is_excluded_and_empty_split_blocks(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); self._dataset(root)
            payload = json.loads((root / "annotation" / "val" / "episode-a.json").read_text())
            payload["video_length"] = 20; payload["action"] = payload["action"][:20]; payload["proprio"] = payload["proprio"][:20]
            (root / "annotation" / "val" / "episode-a.json").write_text(json.dumps(payload))
            with self.assertRaises(Wan22DroidError): build_sample_manifest(root, "val")

    def test_contract_requires_real_paths(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); self._dataset(root)
            train = root / "train.json"; val = root / "val.json"
            train.write_text(json.dumps(build_sample_manifest(root, "train")))
            val.write_text(json.dumps(build_sample_manifest(root, "val")))
            report = validate_contract(train_manifest=train, validation_manifest=val, model=root / "model", source=root / "source", evaluator_contract=root / "eval.json", adapter=root / "wan22_droid_adapter.py")
            self.assertEqual(report["state"], "blocked")
            self.assertIn("MODEL_PATH_INVALID", report["blockers"])

    def test_adapter_contract_accepts_explicit_wan22_droid_name(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); self._dataset(root)
            adapter = root / "wan22_droid_adapter.py"; adapter.write_text("# contract\n")
            model = root / "model"; model.mkdir()
            source = root / "source"; (source / "wan" / "modules").mkdir(parents=True)
            (source / "wan" / "modules" / "model.py").write_text("# entrypoint\n")
            evaluator = root / "eval.json"; evaluator.write_text("{}")
            train = root / "train.json"; val = root / "val.json"
            train.write_text(json.dumps(build_sample_manifest(root, "train")))
            val.write_text(json.dumps(build_sample_manifest(root, "val")))
            report = validate_contract(train_manifest=train, validation_manifest=val, model=model, source=source, evaluator_contract=evaluator, adapter=adapter)
            self.assertNotIn("WAN22_ADAPTER_IDENTITY_INVALID", report["blockers"])

    def test_contract_rejects_unrelated_existing_source_directory(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); self._dataset(root)
            adapter = root / "wan22_droid_adapter.py"; adapter.write_text("# contract\n")
            model = root / "model"; model.mkdir()
            source = root / "unrelated"; source.mkdir()
            evaluator = root / "eval.json"; evaluator.write_text("{}")
            train = root / "train.json"; val = root / "val.json"
            train.write_text(json.dumps(build_sample_manifest(root, "train")))
            val.write_text(json.dumps(build_sample_manifest(root, "val")))
            report = validate_contract(train_manifest=train, validation_manifest=val, model=model, source=source, evaluator_contract=evaluator, adapter=adapter)
            self.assertIn("WAN22_MODEL_ENTRYPOINT_MISSING", report["blockers"])

    def test_runner_rejects_train_validation_episode_overlap(self):
        train = {"records": [{"episode_id": "episode-a"}]}
        validation = {"records": [{"episode_id": "episode-a"}]}
        with self.assertRaisesRegex(ValueError, "WAN22_DROID_EPISODE_SPLIT_OVERLAP"):
            _assert_episode_disjoint(train, validation)

    def test_runner_rejects_missing_episode_identity(self):
        with self.assertRaisesRegex(ValueError, "WAN22_DROID_VALIDATION_EPISODE_ID_MISSING"):
            _assert_episode_disjoint(
                {"records": [{"episode_id": "episode-a"}]},
                {"records": [{"sample_id": "missing"}]},
            )

    def test_formal_validation_panel_is_episode_diverse_and_frozen(self):
        manifest = {"records": [
            {"episode_id": "e0", "sample_id": "e0:0:150"},
            {"episode_id": "e1", "sample_id": "e1:0:150"},
            {"episode_id": "e2", "sample_id": "e2:0:150"},
        ]}
        self.assertEqual(
            _validation_panel_indices(manifest, requested=None, fallback_index=0, formal=True),
            [0, 1, 2],
        )
        with self.assertRaisesRegex(ValueError, "EPISODES_NOT_DISTINCT"):
            _validation_panel_indices(
                {"records": [
                    {"episode_id": "e0"}, {"episode_id": "e0"}, {"episode_id": "e1"}
                ]}, requested=[0, 1, 2], fallback_index=0, formal=True,
            )

    def test_runner_defaults_to_bounded_long_training(self):
        args = runner_parser().parse_args([
            "--source", "/source", "--model", "/model", "--adapter", "/adapter.py",
            "--train-manifest", "/train.json", "--validation-manifest", "/val.json",
            "--evaluator-contract", "/eval.json", "--output-root", "/output",
        ])
        self.assertEqual(args.training_mode, "long")
        self.assertEqual(args.training_stage, "pilot")
        self.assertEqual(args.steps, 512)
        self.assertEqual(args.train_record_limit, 256)

    def test_formal_runner_stage_cannot_use_probe_mode(self):
        with self.assertRaisesRegex(ValueError, "FORMAL_STAGE_REQUIRES_LONG_TRAINING"):
            _validate_stage_training_mode("pilot", "probe")

    def test_runner_honors_full_control_plane_contract(self):
        import argparse
        from unittest.mock import patch

        args = argparse.Namespace(
            train_manifest=Path("/train.json"),
            validation_manifest=Path("/val.json"),
            training_stage="pilot",
        )
        env = {
            "VERDIWM_TRAINING_CONTRACT": "VERDIWM_TRAINING_CONTRACT_V1",
            "VERDIWM_TRAINING_STAGE": "pilot",
            "VERDIWM_TRAINING_MODE": "long",
            "VERDIWM_TRAINING_STEPS": "512",
            "VERDIWM_TRAINING_RECORD_LIMIT": "256",
            "VERDIWM_TRAINING_SAMPLER": "episode_balanced",
            "VERDIWM_TRAINING_SEED_COUNT": "3",
            "VERDIWM_TRAINING_SCALE_PLAN_SHA256": "0" * 64,
            "VERDIWM_VALIDATION_PANEL_SIZE": "3",
        }
        with patch.dict("os.environ", env, clear=False):
            binding = _validate_control_plane_training_contract(args)
        self.assertEqual(binding["expected_stage"], "pilot")
        self.assertEqual(binding["steps"], 512)
        self.assertEqual(binding["validation_panel_size"], 3)

    def test_long_training_controls_are_explicit(self):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--training-mode", choices=("probe", "long"), default="probe")
        parser.add_argument("--train-record-limit", type=int, default=1)
        args = parser.parse_args(["--training-mode", "long", "--train-record-limit", "0"])
        self.assertEqual(args.training_mode, "long")
        self.assertEqual(args.train_record_limit, 0)

    def test_long_training_steps_are_global_updates(self):
        records = [{"episode_id": f"e{i}", "horizon_frames": 150} for i in range(100)]
        self.assertEqual(_training_record_schedule(records, 7, "probe", 1, 3), [(7, 0)] * 3)
        sequential = _training_record_schedule(records, 0, "long", 2, 5, sampler="sequential")
        self.assertEqual([index for index, _offset in sequential], [0, 1, 0, 1, 0])

    def test_episode_balanced_training_samples_distinct_episodes_first(self):
        records = [
            {"episode_id": "a", "horizon_frames": 150},
            {"episode_id": "a", "horizon_frames": 150},
            {"episode_id": "b", "horizon_frames": 150},
            {"episode_id": "c", "horizon_frames": 150},
        ]
        schedule = _training_record_schedule(records, 0, "long", 3, 6, seed=4101)
        first_pass = [records[index]["episode_id"] for index, _offset in schedule[:3]]
        self.assertEqual(len(set(first_pass)), 3)
        self.assertGreater(len({offset for _index, offset in schedule}), 1)
        self.assertEqual(schedule, _training_record_schedule(records, 0, "long", 3, 6, seed=4101))

    def test_formal_training_rejects_schedule_with_too_few_episodes(self):
        records = [{"episode_id": "only", "horizon_frames": 150}]
        schedule = _training_record_schedule(records, 0, "long", 0, 256)
        with self.assertRaisesRegex(ValueError, "EPISODE_COVERAGE_TOO_LOW"):
            _validate_scheduled_training_coverage(records, schedule, "pilot")

    def test_closed_loop_long_training_default_is_bounded(self):
        import subprocess
        result = subprocess.run(
            ["python", "experiments/wan22_droid_acwm_v1/run.py", "closed-loop", "--help"],
            capture_output=True, text=True, check=True,
        )
        self.assertIn("default: 256", result.stdout)

    def test_closed_loop_defaults_to_real_three_seed_long_training(self):
        parser = _parser()
        action = next(item for item in parser._actions if item.dest == "command")
        closed = action.choices["closed-loop"]
        defaults = {item.dest: item.default for item in closed._actions}
        self.assertEqual(defaults["training_mode"], "long")
        self.assertEqual(defaults["training_sampler"], "episode_balanced")
        self.assertEqual(defaults["steps"], 512)
        self.assertEqual(defaults["seeds"], [4101, 4202, 4303])
        self.assertGreater(_estimated_gpu_hours(512, 3), 3.0)

    def test_gpu_memory_gate_uses_bound_physical_device(self):
        import subprocess
        result = subprocess.CompletedProcess([], 0, "0, 40960, 1000\n3, 40960, 12000\n", "")
        with patch("experiments.wan22_droid_acwm_v1.run.subprocess.run", return_value=result):
            self.assertEqual(_gpu_free_memory_mib("3"), 28960.0)

    def test_runner_loads_only_the_explicit_adapter(self):
        with tempfile.TemporaryDirectory() as raw:
            adapter = Path(raw) / "bound_adapter.py"
            adapter.write_text(
                "class Wan22DroidTokenAdapter:\n    pass\n\n"
                "def validate_window(*args, **kwargs):\n    return True\n",
                encoding="utf-8",
            )
            adapter_class, validate, loaded = _load_adapter(adapter)
            self.assertEqual(adapter_class.__name__, "Wan22DroidTokenAdapter")
            self.assertTrue(callable(validate))
            self.assertEqual(loaded, adapter.resolve())

    def test_conditioning_arms_are_explicit_and_history_is_causal(self):
        import numpy as np
        actions = np.asarray([[1.0] * 7, [3.0] * 7], dtype=np.float32)
        proprio = np.asarray([[2.0] * 14, [4.0] * 14], dtype=np.float32)
        zero_a, zero_p = _conditioning_for_mode(actions, proprio, "visual_anchor_only")
        self.assertTrue(np.all(zero_a == 0) and np.all(zero_p == 0))
        action_a, action_p = _conditioning_for_mode(actions, proprio, "action")
        self.assertTrue(np.array_equal(action_a, actions) and np.all(action_p == 0))
        hist_a, hist_p = _conditioning_for_mode(actions, proprio, "action_proprio_history")
        self.assertEqual(float(hist_a[0, 0]), 1.0)
        self.assertEqual(float(hist_a[1, 0]), 2.0)
        self.assertEqual(float(hist_p[1, 0]), 3.0)

    def test_ema_conditioning_is_causal_and_decay_is_bounded(self):
        import numpy as np
        actions = np.asarray([[0.0] * 7, [10.0] * 7, [0.0] * 7], dtype=np.float32)
        proprio = np.asarray([[0.0] * 14, [4.0] * 14, [0.0] * 14], dtype=np.float32)
        ema_a, ema_p = _conditioning_for_mode(actions, proprio, "action_proprio_ema", 0.5)
        self.assertEqual(float(ema_a[0, 0]), 0.0)
        self.assertEqual(float(ema_a[1, 0]), 5.0)
        self.assertEqual(float(ema_a[2, 0]), 2.5)
        self.assertEqual(float(ema_p[1, 0]), 2.0)
        with self.assertRaisesRegex(ValueError, "WAN22_DROID_HISTORY_DECAY_INVALID"):
            _conditioning_for_mode(actions, proprio, "action_proprio_ema", 0.0)
        with self.assertRaisesRegex(ValueError, "WAN22_DROID_HISTORY_DECAY_INVALID"):
            _conditioning_for_mode(actions, proprio, "action_proprio_ema", 1.1)

    def test_initial_reference_anchor_blends_only_causal_state(self):
        import numpy as np
        first = np.asarray([2.0], dtype=np.float32)
        previous = np.asarray([10.0], dtype=np.float32)
        anchor, source = _chunk_anchor(first, previous, 1, "initial_reference_blend", 0.25)
        self.assertEqual(float(anchor[0]), 8.0)
        self.assertEqual(source, "previous_generated_initial_reference_blend")
        initial, initial_source = _chunk_anchor(first, None, 0, "initial_reference_blend", 0.25)
        self.assertIs(initial, first)
        self.assertEqual(initial_source, "first_observed_frame")
        with self.assertRaisesRegex(ValueError, "WAN22_DROID_ANCHOR_REFRESH_STRENGTH_INVALID"):
            _chunk_anchor(first, previous, 1, "initial_reference_blend", 1.1)

    def test_branch_selection_uses_reference_and_previous_state(self):
        import numpy as np
        reference = np.asarray([1.0, 0.0], dtype=np.float32)
        previous = np.asarray([0.0, 1.0], dtype=np.float32)
        branches = [np.asarray([1.0, 0.0]), np.asarray([0.0, 1.0]), np.asarray([-1.0, 0.0])]
        selected_reference, scores_reference = _select_branch_index(branches, reference, previous, 0.8)
        selected_continuity, scores_continuity = _select_branch_index(branches, reference, previous, 0.2)
        self.assertEqual(selected_reference, 0)
        self.assertEqual(selected_continuity, 1)
        self.assertEqual(len(scores_reference), 3)
        self.assertEqual(len(scores_continuity), 3)
        with self.assertRaisesRegex(ValueError, "WAN22_DROID_BRANCH_REFERENCE_WEIGHT_INVALID"):
            _select_branch_index(branches, reference, previous, -0.1)


if __name__ == "__main__":
    unittest.main()
