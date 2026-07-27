from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from wmloop.diagnose.diagnostic_probe_routing_admission import (
    DiagnosticProbeRoutingAdmissionError,
    main,
    run_diagnostic_probe_routing_admission,
)


ROOT = Path(__file__).resolve().parents[1]


class DiagnosticProbeRoutingAdmissionTests(unittest.TestCase):
    def test_fixture_admitted_probes_create_preview_only_routes(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = _write_inputs(root, runtime_ready=False)

            manifest = run_diagnostic_probe_routing_admission(
                repo_root=ROOT,
                failure_signature_bank_manifest=inputs["bank_manifest"],
                probe_admission_manifest=inputs["probe_manifest"],
                primitive_materialization_gate_manifest=inputs["primitive_manifest"],
                output_root=root / "routing",
            )

            self.assertEqual(manifest["state"], "preview_only")
            self.assertEqual(manifest["summary"]["preview_only_route_count"], 1)
            self.assertEqual(manifest["summary"]["routing_ready_route_count"], 0)

            report = json.loads(Path(manifest["report_path"]).read_text(encoding="utf-8"))
            self.assertFalse(report["side_effects"]["gpu_execution_started"])
            self.assertFalse(report["side_effects"]["verdict_probe_mutated"])
            route = next(row for row in report["primitive_rows"] if row["primitive"] == "next_forcing")
            self.assertEqual(route["route_gate"], "preview_only_runtime_smoke_required")
            self.assertEqual(route["environment_probe_authority"], "preview_only")
            self.assertEqual(route["matched_probe_ids"], ["acwm_pour_water_fluid_volume_transport_diagnostic_v1"])

            rejected = next(row for row in report["primitive_rows"] if row["primitive"] == "drift_token_trim")
            self.assertEqual(rejected["route_gate"], "blocked_by_rejected_prior")

    def test_runtime_ready_probe_and_closed_loop_primitive_allow_canary_after_gpu_free(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = _write_inputs(root, runtime_ready=True)

            manifest = run_diagnostic_probe_routing_admission(
                repo_root=ROOT,
                failure_signature_bank_manifest=inputs["bank_manifest"],
                probe_admission_manifest=inputs["probe_manifest"],
                primitive_materialization_gate_manifest=inputs["primitive_manifest"],
                output_root=root / "routing",
            )

            self.assertEqual(manifest["state"], "routing_ready")
            report = json.loads(Path(manifest["report_path"]).read_text(encoding="utf-8"))
            route = next(row for row in report["primitive_rows"] if row["primitive"] == "next_forcing")
            self.assertEqual(route["route_gate"], "canary_after_gpu_free")
            self.assertEqual(route["environment_probe_authority"], "routing_ready")

    def test_sidecar_only_primitive_is_blocked_even_with_probe_preview(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = _write_inputs(root, runtime_ready=True, next_forcing_closed_loop=False)

            manifest = run_diagnostic_probe_routing_admission(
                repo_root=ROOT,
                failure_signature_bank_manifest=inputs["bank_manifest"],
                probe_admission_manifest=inputs["probe_manifest"],
                primitive_materialization_gate_manifest=inputs["primitive_manifest"],
                output_root=root / "routing",
            )

            report = json.loads(Path(manifest["report_path"]).read_text(encoding="utf-8"))
            route = next(row for row in report["primitive_rows"] if row["primitive"] == "next_forcing")
            self.assertEqual(route["route_gate"], "blocked_by_primitive_materialization")
            self.assertTrue(any(blocker["code"] == "routed_primitives_not_closed_loop_eligible" for blocker in report["blockers"]))

    def test_existing_output_fails_closed(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = _write_inputs(root, runtime_ready=False)
            output_root = root / "routing"
            output_root.mkdir()

            with self.assertRaisesRegex(DiagnosticProbeRoutingAdmissionError, "OUTPUT_EXISTS"):
                run_diagnostic_probe_routing_admission(
                    repo_root=ROOT,
                    failure_signature_bank_manifest=inputs["bank_manifest"],
                    probe_admission_manifest=inputs["probe_manifest"],
                    primitive_materialization_gate_manifest=inputs["primitive_manifest"],
                    output_root=output_root,
                )

    def test_cli_prints_manifest(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = _write_inputs(root, runtime_ready=False)
            output = io.StringIO()

            with redirect_stdout(output):
                code = main(
                    [
                        "--repo-root",
                        str(ROOT),
                        "--failure-signature-bank-manifest",
                        str(inputs["bank_manifest"]),
                        "--probe-admission-manifest",
                        str(inputs["probe_manifest"]),
                        "--primitive-materialization-gate-manifest",
                        str(inputs["primitive_manifest"]),
                        "--output-root",
                        str(root / "routing"),
                    ]
                )

            self.assertEqual(code, 0)
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["artifact_type"], "wmloop-diagnostic-probe-routing-admission-manifest")
            self.assertEqual(payload["state"], "preview_only")


def _write_inputs(root: Path, *, runtime_ready: bool, next_forcing_closed_loop: bool = True) -> dict[str, Path]:
    bank = {
        "schema_version": 1,
        "artifact_type": "wmloop-failure-signature-bank",
        "state": "ready",
        "records": [
            {
                "environment": "pour_water",
                "failure_signatures": [
                    {
                        "signature": "fluid_volume_transport",
                        "wm_dx_failure_family": "ood_physics",
                        "diagnostic_probe_id": "acwm_pour_water_fluid_volume_transport_diagnostic_v1",
                    }
                ],
            }
        ],
        "diagnostic_probe_work_orders": [
            {
                "environment": "pour_water",
                "signature": "fluid_volume_transport",
                "probe_id": "acwm_pour_water_fluid_volume_transport_diagnostic_v1",
                "priority": "P0_rediagnose",
                "verdict_exposure_allowed": False,
            }
        ],
        "primitive_routing": [
            {
                "environment": "pour_water",
                "primitive": "drift_token_trim",
                "routing_decision": "reject_or_demote_until_new_diagnosis",
                "target_failures": ["appearance_drift"],
            },
            {
                "environment": "pour_water",
                "primitive": "next_forcing",
                "routing_decision": "stage_canary_after_diagnostic_probe",
                "target_failures": ["ood_physics"],
            },
        ],
    }
    bank_path = _write_json(root / "bank" / "failure-signature-bank.json", bank)
    bank_manifest = _write_json(
        root / "bank" / "manifest.json",
        {
            "schema_version": 1,
            "artifact_type": "wmloop-failure-signature-bank-manifest",
            "state": "ready",
            "report_path": str(bank_path),
        },
    )
    probe_record = {
        "probe_id": "acwm_pour_water_fluid_volume_transport_diagnostic_v1",
        "environment": "pour_water",
        "signature": "fluid_volume_transport",
        "verdict_exposure_allowed": False,
    }
    if runtime_ready:
        probe_record["admission_state"] = "runtime_smoke_passed"
        probe_record["runtime_smoke_on_dev_split"] = "passed"
    probe_manifest = _write_json(
        root / "probe-admission" / "manifest.json",
        {
            "schema_version": 1,
            "artifact_type": "wmloop-diagnostic-probe-admission-manifest",
            "state": "fixture_unit_admitted",
            "probes": [probe_record],
        },
    )
    primitive_manifest = _write_json(
        root / "primitive-gate" / "manifest.json",
        {
            "schema_version": 1,
            "artifact_type": "wmloop-primitive-materialization-gate",
            "state": "blocked",
            "records": [
                {
                    "primitive": "drift_token_trim",
                    "admission_state": "hook_only_runtime_ready",
                    "closed_loop_eligible": False,
                },
                {
                    "primitive": "next_forcing",
                    "admission_state": "closed_loop_runtime_ready" if next_forcing_closed_loop else "sidecar_only",
                    "closed_loop_eligible": next_forcing_closed_loop,
                },
            ],
        },
    )
    return {
        "bank_manifest": bank_manifest,
        "probe_manifest": probe_manifest,
        "primitive_manifest": primitive_manifest,
    }


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    unittest.main()
