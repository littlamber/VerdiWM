from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts.export.acwm_autoloop_worker import _reserved_gpu_indices_from_processes, _row_preflight, run_autoloop_worker


class AcwmAutoloopWorkerTests(unittest.TestCase):
    def test_live_launch_marker_blocks_duplicate_runtime_only_launch(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "runtime-gate"
            marker = output.with_name(f"{output.name}.launching")
            marker.write_text(str(__import__("os").getpid()), encoding="ascii")
            row = {
                "rank": 1,
                "phase": "official_eval_gate",
                "campaign_id": "runtime-gate",
                "environment": "reacher",
                "primitive": "cfg_guidance_schedule",
                "seed": 1,
                "train_steps": 0,
                "output_root": str(output),
                "launch_argv_template": ["python"],
            }
            self.assertEqual(_row_preflight(row), "OUTPUT_ROOT_LAUNCHING")

    def test_phase_allowlist_blocks_confirmation_during_discovery(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue = root / "queue.json"
            queue.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "rank": 1,
                                "phase": "confirm_staged",
                                "campaign_id": "confirmation",
                                "environment": "robot_arm",
                                "primitive": "cfg_guidance_schedule",
                                "seed": 1,
                                "train_steps": 2000,
                                "output_root": str(root / "confirmation"),
                                "candidate_gpus": [0],
                                "launch_argv_template": ["python", "confirm.py"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            run_autoloop_worker(
                queue_path=queue,
                output_root=root / "worker",
                dry_run=True,
                phase_allowlist={"screen_512", "official_eval_gate"},
            )

            report = json.loads((root / "worker" / "autoloop-worker.json").read_text(encoding="utf-8"))
            self.assertEqual(report["launched_count"], 0)
            self.assertEqual(report["records"][0]["reason"], "PHASE_EXCLUDED")

    def test_process_reservations_cover_pre_cuda_campaigns(self) -> None:
        with TemporaryDirectory() as temporary:
            proc = Path(temporary)
            for pid, argv in {
                "10": ["python", "-m", "wmloop.execute.training_eval_limited_campaign", "run", "--gpus", "1"],
                "11": ["python", "-m", "wmloop.orchestrator_training_eval_smoke", "run", "--gpu-index", "2"],
                "12": ["python", "unrelated.py", "--gpus", "0"],
                "13": ["python", "-m", "wmloop.execute.primitive_runtime_smoke", "run", "--gpu-index", "3"],
                "14": ["python", "/repo/scripts/export/acwm_formal_visualization.py", "--gpu-index", "4"],
                "15": ["python", "/repo/scripts/export/acwm_runtime_only_screen.py", "run", "--gpu-index", "5"],
                "16": ["python", "-m", "wmloop.diagnose.horizon_runtime", "run", "--gpu-index", "6"],
                "17": ["python", "/repo/scripts/export/acwm_failed_screen_salvage.py", "run", "--gpu-index", "7"],
            }.items():
                path = proc / pid
                path.mkdir()
                (path / "cmdline").write_bytes(b"\0".join(part.encode() for part in argv) + b"\0")

            self.assertEqual(_reserved_gpu_indices_from_processes(proc), {1, 2, 3, 4, 5, 6, 7})

    def test_worker_intersects_live_idle_gpus_with_daemon_allowlist(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue = root / "queue.json"
            queue.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "rank": 1,
                                "phase": "screen_512",
                                "campaign_id": "screen",
                                "environment": "push_rope",
                                "primitive": "mixture_reweight",
                                "seed": 1,
                                "train_steps": 512,
                                "output_root": str(root / "screen"),
                                "candidate_gpus": [0, 1],
                                "launch_argv_template": ["python", "train.py"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with patch("scripts.export.acwm_autoloop_worker._idle_gpu_indices", return_value={0, 1}):
                manifest = run_autoloop_worker(
                    queue_path=queue,
                    output_root=root / "worker",
                    dry_run=True,
                    allowed_gpu_indices={1},
                )

            self.assertEqual(manifest["launched_gpu_indices"], [1])
            report = json.loads((root / "worker/autoloop-worker.json").read_text(encoding="utf-8"))
            self.assertEqual(report["records"][0]["selected_gpu"], 1)

    def test_cpu_evidence_row_does_not_wait_for_or_reserve_gpu(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue = root / "queue.json"
            queue.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "rank": 1,
                                "phase": "horizon_effect_profile",
                                "campaign_id": "profile",
                                "environment": "pour_water",
                                "primitive": "checkpoint_delta_scaling",
                                "seed": 101,
                                "train_steps": 0,
                                "resource_class": "cpu",
                                "output_root": str(root / "profile"),
                                "launch_argv_template": ["python", "profile.py"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with patch("scripts.export.acwm_autoloop_worker._idle_gpu_indices", return_value=set()):
                manifest = run_autoloop_worker(
                    queue_path=queue,
                    output_root=root / "worker",
                    dry_run=True,
                    allowed_gpu_indices=set(),
                )

            self.assertEqual(manifest["launched_count"], 1)
            self.assertEqual(manifest["launched_cpu_count"], 1)
            self.assertEqual(manifest["launched_gpu_indices"], [])

    def test_same_gpu_is_released_after_launched_process_finishes(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue = root / "queue.json"
            queue.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "rank": rank,
                                "phase": "official_eval_gate",
                                "campaign_id": f"gate-{rank}",
                                "environment": "reacher",
                                "primitive": f"primitive_{rank}",
                                "seed": rank,
                                "train_steps": 0,
                                "output_root": str(root / f"gate-{rank}"),
                                "candidate_gpus": [0],
                                "launch_argv_template": ["python", "gate.py"],
                            }
                            for rank in (1, 2)
                        ]
                    }
                ),
                encoding="utf-8",
            )
            launches = [
                {
                    "state": "launched",
                    "reason": "LAUNCHED",
                    "selected_gpu": 0,
                    "audit": {},
                    "launch": {"pid": pid, "launch_log": "", "argv": []},
                }
                for pid in (101, 202)
            ]

            with (
                patch("scripts.export.acwm_autoloop_worker._idle_gpu_indices", return_value={0}),
                patch("scripts.export.acwm_autoloop_worker._launch_first_ready_gpu", side_effect=launches) as launch,
                patch("scripts.export.acwm_autoloop_worker._pid_is_running", side_effect=[True, False]),
            ):
                manifest = run_autoloop_worker(
                    queue_path=queue,
                    output_root=root / "worker",
                    max_launches=2,
                    iterations=3,
                    sleep_seconds=0,
                )

            self.assertEqual(manifest["launched_count"], 2)
            self.assertEqual(manifest["released_gpu_launch_count"], 1)
            self.assertEqual(launch.call_count, 2)

    def test_post_confirmation_gate_waits_only_for_ready_completion(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            dependency = root / "confirm" / "manifest.json"
            dependency.parent.mkdir(parents=True)
            dependency.write_text(json.dumps({"state": "ready"}), encoding="utf-8")
            queue = root / "queue.json"
            queue.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "rank": 1,
                                "phase": "confirm_official_eval_gate",
                                "campaign_id": "confirm-official",
                                "environment": "push_cube",
                                "primitive": "mixture_reweight",
                                "seed": 801,
                                "train_steps": 0,
                                "output_root": str(root / "official"),
                                "candidate_gpus": [0],
                                "requires_ready_manifest": str(dependency),
                                "launch_argv_template": ["python", "-m", "official"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with patch("scripts.export.acwm_autoloop_worker._idle_gpu_indices", return_value={0}):
                run_autoloop_worker(
                    queue_path=queue,
                    output_root=root / "worker",
                    dry_run=True,
                    max_launches=1,
                    sleep_seconds=0,
                )

            report = json.loads((root / "worker" / "autoloop-worker.json").read_text(encoding="utf-8"))
            self.assertEqual(report["records"][0]["state"], "dry_run_ready")

    def test_post_confirmation_gate_can_fall_back_to_any_idle_gpu(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            dependency = root / "confirm" / "manifest.json"
            dependency.parent.mkdir(parents=True)
            dependency.write_text(json.dumps({"state": "ready"}), encoding="utf-8")
            queue = root / "queue.json"
            queue.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "rank": 1,
                                "phase": "confirm_official_eval_gate",
                                "campaign_id": "confirm-official",
                                "environment": "push_cube",
                                "primitive": "mixture_reweight",
                                "seed": 801,
                                "train_steps": 0,
                                "output_root": str(root / "official"),
                                "candidate_gpus": [2],
                                "allow_any_idle_gpu": True,
                                "requires_ready_manifest": str(dependency),
                                "launch_argv_template": ["python", "-m", "official"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with patch("scripts.export.acwm_autoloop_worker._idle_gpu_indices", return_value={0}):
                run_autoloop_worker(
                    queue_path=queue,
                    output_root=root / "worker",
                    dry_run=True,
                    max_launches=1,
                    sleep_seconds=0,
                )

            report = json.loads((root / "worker" / "autoloop-worker.json").read_text(encoding="utf-8"))
            self.assertEqual(report["records"][0]["state"], "dry_run_ready")
            self.assertEqual(report["records"][0]["selected_gpu"], 0)

    def test_confirmation_row_waits_for_positive_dependency(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "screen" / "envs" / "push_cube" / "manifest.json"
            queue = _write_queue(root, manifest_path)

            manifest = run_autoloop_worker(
                queue_path=queue,
                output_root=root / "worker",
                dry_run=True,
                max_launches=1,
                sleep_seconds=0,
            )

            self.assertEqual(manifest["state"], "ready")
            report = json.loads((root / "worker" / "autoloop-worker.json").read_text(encoding="utf-8"))
            confirm = next(record for record in report["records"] if record["phase"] == "confirm_4k")
            self.assertEqual(confirm["reason"], "DEPENDENCY_PENDING")

    def test_confirmation_row_blocks_negative_dependency(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "screen" / "envs" / "push_cube" / "manifest.json"
            manifest_path.parent.mkdir(parents=True)
            manifest_path.write_text(
                json.dumps(
                    {
                        "state": "ready",
                        "primary_metric": "ladder_auc_psnr_envmax",
                        "delta_m_ver": {"ladder_auc_psnr_envmax": -1.0},
                        "action_following_gate": {"enabled": True, "pass": True},
                    }
                ),
                encoding="utf-8",
            )
            queue = _write_queue(root, manifest_path)

            run_autoloop_worker(
                queue_path=queue,
                output_root=root / "worker",
                dry_run=True,
                max_launches=1,
                sleep_seconds=0,
            )

            report = json.loads((root / "worker" / "autoloop-worker.json").read_text(encoding="utf-8"))
            confirm = next(record for record in report["records"] if record["phase"] == "confirm_4k")
            self.assertEqual(confirm["reason"], "DEPENDENCY_NONPOSITIVE:-1.0")

    def test_confirmation_requires_passing_official_quality_gate(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            screen_manifest = root / "screen" / "manifest.json"
            screen_manifest.parent.mkdir(parents=True)
            screen_manifest.write_text(
                json.dumps({"state": "ready", "delta_m_ver": {"ladder_auc_psnr_envmax": 1.0}}),
                encoding="utf-8",
            )
            official_manifest = root / "official" / "manifest.json"
            official_manifest.parent.mkdir(parents=True)
            official_manifest.write_text(
                json.dumps(
                    {
                        "state": "ready",
                        "official_quality_gate": {
                            "pass": False,
                            "delta_candidate_minus_baseline": {"psnr": -4.98},
                        },
                    }
                ),
                encoding="utf-8",
            )
            queue = root / "queue.json"
            queue.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "rank": 1,
                                "phase": "confirm_4k",
                                "campaign_id": "confirm",
                                "environment": "reacher",
                                "primitive": "next_forcing",
                                "seed": 711,
                                "train_steps": 4000,
                                "output_root": str(root / "confirm"),
                                "candidate_gpus": [],
                                "requires_positive_manifest": str(screen_manifest),
                                "requires_official_quality_manifest": str(official_manifest),
                                "launch_argv_template": ["python", "-m", "confirm"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            run_autoloop_worker(
                queue_path=queue,
                output_root=root / "worker",
                dry_run=True,
                max_launches=1,
                sleep_seconds=0,
            )

            report = json.loads((root / "worker" / "autoloop-worker.json").read_text(encoding="utf-8"))
            self.assertEqual(report["records"][0]["reason"], "OFFICIAL_QUALITY_GATE_FAILED:-4.98")

    def test_confirmation_requires_every_factorial_quality_gate(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifests = []
            for index, passed in enumerate((True, False, True)):
                manifest = root / f"gate-{index}" / "manifest.json"
                manifest.parent.mkdir(parents=True)
                manifest.write_text(
                    json.dumps(
                        {
                            "state": "ready",
                            "official_quality_gate": {
                                "pass": passed,
                                "delta_candidate_minus_baseline": {"psnr": 0.5 if passed else -0.25},
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                manifests.append(str(manifest))
            row = {
                "rank": 1,
                "phase": "confirm_staged",
                "campaign_id": "factorial-confirmation",
                "environment": "cloth_move",
                "primitive": "self_forcing_finetune",
                "seed": 4101,
                "train_steps": 1000,
                "output_root": str(root / "confirmation"),
                "launch_argv_template": ["python", "train.py"],
                "requires_official_quality_manifests": manifests,
            }

            self.assertEqual(
                _row_preflight(row),
                "OFFICIAL_QUALITY_DEPENDENCY_1:OFFICIAL_QUALITY_GATE_FAILED:-0.25",
            )

            for manifest in manifests:
                payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
                payload["official_quality_gate"]["pass"] = True
                Path(manifest).write_text(json.dumps(payload), encoding="utf-8")

            self.assertIsNone(_row_preflight(row))

    def test_finalizer_waits_for_every_ready_gate_manifest(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            ready = root / "gate-512" / "manifest.json"
            ready.parent.mkdir(parents=True)
            ready.write_text(json.dumps({"state": "ready"}), encoding="utf-8")
            pending = root / "gate-1000" / "manifest.json"
            queue = root / "queue.json"
            queue.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "rank": 1,
                                "phase": "checkpoint_ladder_finalize",
                                "campaign_id": "finalize",
                                "environment": "push_cube",
                                "primitive": "next_forcing",
                                "seed": 901,
                                "train_steps": 0,
                                "output_root": str(root / "finalize"),
                                "candidate_gpus": [],
                                "requires_ready_manifests": [str(ready), str(pending)],
                                "launch_argv_template": ["python", "-m", "finalize"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            run_autoloop_worker(
                queue_path=queue,
                output_root=root / "worker",
                dry_run=True,
                max_launches=1,
                sleep_seconds=0,
            )

            report = json.loads((root / "worker" / "autoloop-worker.json").read_text(encoding="utf-8"))
            self.assertEqual(report["records"][0]["reason"], "READY_DEPENDENCY_1:READY_DEPENDENCY_PENDING")


def _write_queue(root: Path, manifest_path: Path) -> Path:
    queue = root / "queue.json"
    queue.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "rank": 1,
                        "phase": "screen_512",
                        "campaign_id": "screen",
                        "environment": "push_cube",
                        "primitive": "mixture_reweight",
                        "seed": 801,
                        "train_steps": 512,
                        "output_root": str(root / "screen"),
                        "candidate_gpus": [],
                        "gpu_audit_root_template": str(root / "audit-gpu{gpu}-{attempt_id}"),
                        "launch_argv_template": ["python", "-m", "screen"],
                    },
                    {
                        "rank": 2,
                        "phase": "confirm_4k",
                        "campaign_id": "confirm",
                        "environment": "push_cube",
                        "primitive": "mixture_reweight",
                        "seed": 801,
                        "train_steps": 4000,
                        "output_root": str(root / "confirm"),
                        "candidate_gpus": [],
                        "requires_positive_manifest": str(manifest_path),
                        "gpu_audit_root_template": str(root / "audit-gpu{gpu}-{attempt_id}"),
                        "launch_argv_template": ["python", "-m", "confirm"],
                    },
                ]
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return queue
