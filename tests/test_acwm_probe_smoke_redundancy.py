from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from wmloop.experiments.probe_smoke_redundancy import (
    ProbeSmokeRedundancyError,
    evaluate_probe_smoke_redundancy,
)


class AcwmProbeSmokeRedundancyTests(unittest.TestCase):
    def test_rejects_local_near_collinear_successor(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            reference = self._campaign(root / "reference", "uniform", [2.0, 1.0])
            candidate = self._campaign(root / "candidate", "horizon", [1.96, 0.98])

            manifest = evaluate_probe_smoke_redundancy(
                reference_campaign_root=reference,
                candidate_campaign_root=candidate,
                environments=["a", "b"],
                output_root=root / "output",
            )

            self.assertEqual(manifest["decision"], "reject_as_redundant")
            self.assertFalse(manifest["expand_to_eight_environment_pilot"])

    def test_expands_noncollinear_successor(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            reference = self._campaign(root / "reference", "uniform", [2.0, 1.0])
            candidate = self._campaign(root / "candidate", "event", [-1.0, 2.0])

            manifest = evaluate_probe_smoke_redundancy(
                reference_campaign_root=reference,
                candidate_campaign_root=candidate,
                environments=["a", "b"],
                output_root=root / "output",
            )

            self.assertEqual(manifest["decision"], "expand_collision_evidence")
            self.assertTrue(manifest["expand_to_eight_environment_pilot"])

    def test_rejects_cross_fidelity_comparison(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            reference = self._campaign(root / "reference", "uniform", [2.0, 1.0])
            candidate = self._campaign(
                root / "candidate", "horizon", [1.96, 0.98], protocol="smoke"
            )

            with self.assertRaisesRegex(
                ProbeSmokeRedundancyError,
                "PROBE_REDUNDANCY_MEASUREMENT_CONTRACT_MISMATCH:a:protocol",
            ):
                evaluate_probe_smoke_redundancy(
                    reference_campaign_root=reference,
                    candidate_campaign_root=candidate,
                    environments=["a", "b"],
                    output_root=root / "output",
                )

    def test_nonlocal_nonredundancy_does_not_license_expansion(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            reference = self._campaign(root / "reference", "uniform", [2.0, 1.0])
            candidate = self._campaign(root / "candidate", "horizon", [1.96, 0.98])
            chart_path = candidate / "environments" / "b" / "response-chart.json"
            chart = json.loads(chart_path.read_text())
            chart["response_coordinate"] = [-1.0, 2.0]
            chart["locality_residuals"]["horizon"] = 1.0
            chart_path.write_text(json.dumps(chart))

            manifest = evaluate_probe_smoke_redundancy(
                reference_campaign_root=reference,
                candidate_campaign_root=candidate,
                environments=["a", "b"],
                output_root=root / "output",
            )

            self.assertEqual(manifest["decision"], "reject_as_redundant")
            self.assertFalse(manifest["expand_to_eight_environment_pilot"])
            report = json.loads((root / "output" / "probe-smoke-redundancy.json").read_text())
            self.assertEqual(report["locality_admitted_environment_count"], 1)

    def _campaign(
        self,
        root: Path,
        probe: str,
        response: list[float],
        *,
        protocol: str = "pilot",
    ) -> Path:
        for environment in ("a", "b"):
            path = root / "environments" / environment
            path.mkdir(parents=True)
            (path / "manifest.json").write_text(
                json.dumps(
                    {
                        "environment": environment,
                        "protocol": protocol,
                        "checkpoint_sha256": f"checkpoint-{environment}",
                        "config_sha256": f"config-{environment}",
                        "seeds": [101, 202, 303],
                        "doses": [-0.1, -0.05, 0.0, 0.05, 0.1],
                        "measurement_count": 15,
                    }
                )
            )
            (path / "response-chart.json").write_text(
                json.dumps(
                    {
                        "outcome_names": ["one", "two"],
                        "intervention_names": [probe],
                        "response_coordinate": response,
                        "locality_residuals": {probe: 0.1},
                    }
                )
            )
        return root


if __name__ == "__main__":
    unittest.main()
