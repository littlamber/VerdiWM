from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from wmloop.experiments.probe_semantic_compile import compile_probe_for_backbone


class ProbeSemanticCompileTests(unittest.TestCase):
    def test_real_contracts_compile_only_after_exact_semantics_are_evidenced(self) -> None:
        root = Path(__file__).resolve().parents[1]
        probe = root / "configs/probes/staging/cpbe_residual_63f088b0d5.json"
        cases = (
            ("configs/backbones/ctrl_world_predictive_quality_paper_v1.json", "configs/backbones/ctrl_world_predictive_probe_capability_v1.json", "compiled"),
            ("configs/backbones/cosmos3_forward_dynamics_predictive_pilot_v1.json", "configs/backbones/cosmos3_forward_dynamics_probe_capability_v1.json", "compiled"),
        )
        for instance_rel, contract_rel, expected_state in cases:
            with self.subTest(instance=instance_rel):
                with tempfile.TemporaryDirectory() as temporary:
                    fixture_root = Path(temporary)
                    instance = json.loads((root / instance_rel).read_text(encoding="utf-8"))
                    contract = json.loads((root / contract_rel).read_text(encoding="utf-8"))
                    self._materialize_external_evidence(
                        fixture_root=fixture_root,
                        instance=instance,
                        contract=contract,
                    )
                    instance_path = fixture_root / "instance.json"
                    instance_path.write_text(json.dumps(instance), encoding="utf-8")
                    manifest = compile_probe_for_backbone(
                        probe_path=probe,
                        instance_path=instance_path,
                        capability_contract_path=root / contract_rel,
                        repo_root=root,
                        output_root=fixture_root / "bundle",
                    )
                    self.assertEqual(manifest["state"], expected_state)
                    report = json.loads((Path(temporary) / "bundle/compile-report.json").read_text())
                    self.assertFalse(report["semantic_substitution_used"])
                    self.assertFalse(report["gpu_execution_started"])
                    self.assertEqual(bool(report["missing_required_semantics"]), expected_state == "blocked")
                    receipt = json.loads((Path(temporary) / "bundle/typed-compile-receipt.json").read_text())
                    self.assertEqual(receipt["compiled"], expected_state == "compiled")
                    self.assertEqual(bool(receipt["blockers"]), expected_state == "blocked")

    @staticmethod
    def _materialize_external_evidence(
        *, fixture_root: Path, instance: dict[str, object], contract: dict[str, object]
    ) -> None:
        surfaces = {
            str(row["surface_id"]): row
            for row in instance["surfaces"]
            if row["surface_id"] != "verdiwm_repo"
        }
        external_roots: dict[str, Path] = {}
        evidence_by_path: dict[tuple[str, str], set[str]] = {}
        for capability in contract["capabilities"]:
            for evidence in capability.get("evidence", []):
                surface_id = str(evidence["surface_id"])
                if surface_id == "verdiwm_repo":
                    continue
                evidence_by_path.setdefault(
                    (surface_id, str(evidence["path"])), set()
                ).update(str(anchor) for anchor in evidence["anchors"])
        for surface_id, row in surfaces.items():
            if not any(key[0] == surface_id for key in evidence_by_path):
                continue
            external_root = fixture_root / "external" / surface_id
            external_root.mkdir(parents=True, exist_ok=True)
            row["artifact_ref"] = str(external_root)
            external_roots[surface_id] = external_root
        for (surface_id, relative), anchors in evidence_by_path.items():
            path = external_roots[surface_id] / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(sorted(anchors)) + "\n", encoding="utf-8")

    def test_exact_fixture_contract_compiles(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "repo"
            repo.mkdir()
            (repo / "hook.py").write_text("H2 action_embedding_hook action_sequence_hook paired_seed_control\n")
            probe = root / "probe.json"
            probe.write_text(json.dumps({
                "artifact_type": "wmloop-staged-diagnostic-probe-descriptor",
                "signature": "fixture",
                "program": {
                    "probe_id": "fixture_probe",
                    "hook_type": "H2",
                    "required_capabilities": ["action_embedding_hook", "action_sequence_hook", "paired_seed_control"],
                    "signal_source": "action_embedding_delta",
                    "temporal_basis": "event_phase_tangent",
                    "contrast_operator": "signed_mean_preserving_phase",
                    "aggregation": "goal_outcome_vector",
                    "invariants": ["same_seed"],
                    "dose_schedule": [-0.1, 0.0, 0.1],
                    "diagnostic_only": True,
                    "reversible": True,
                },
            }))
            instance = root / "instance.json"
            instance.write_text(json.dumps({
                "artifact_type": "wmloop-backbone-instance",
                "instance_id": "fixture_instance",
                "backbone_family": "generic",
                "surfaces": [{"surface_id": "repo", "artifact_ref": str(repo)}],
            }))
            contract = root / "contract.json"
            names = ["H2", "action_embedding_hook", "action_sequence_hook", "paired_seed_control", "signal_source:action_embedding_delta", "temporal_basis:event_phase_tangent", "contrast_operator:signed_mean_preserving_phase", "same_seed"]
            contract.write_text(json.dumps({
                "schema_version": 1,
                "artifact_type": "verdiwm-backbone-probe-capability-contract",
                "instance_id": "fixture_instance",
                "backbone_family": "generic",
                "capability_class": "fixture",
                "capabilities": [{"name": name, "kind": "hook" if name == "H2" else "capability", "implemented": True, "reason": "fixture", "evidence": [{"surface_id": "repo", "path": "hook.py", "anchors": ["H2"]}]} for name in names],
                "claim_boundary": "fixture compile contract",
            }))
            manifest = compile_probe_for_backbone(
                probe_path=probe,
                instance_path=instance,
                capability_contract_path=contract,
                repo_root=project_root,
                output_root=root / "bundle",
            )
            self.assertEqual(manifest["state"], "compiled")
            report = json.loads((root / "bundle/compile-report.json").read_text())
            self.assertEqual(report["missing_required_semantics"], [])
