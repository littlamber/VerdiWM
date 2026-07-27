from __future__ import annotations

import unittest

from wmloop.contracts import validate_document
from wmloop.diagnose.probes.acwm_push_rope_topology_change_diagnostic_v1 import (
    PushRopeTopologyChangeProbeError,
    TopologyChangeThresholds,
    measure_topology_change,
)


class PushRopeTopologyChangeProbeTests(unittest.TestCase):
    def test_measures_stable_topology_without_verdict_exposure(self) -> None:
        output = measure_topology_change(frames=_stable_frames())

        self.assertEqual(output["probe_id"], "acwm_push_rope_topology_change_diagnostic_v1")
        self.assertEqual(output["role"], "diagnostic")
        self.assertFalse(output["verdict_exposure_allowed"])
        self.assertNotIn("verdict_evidence", output)
        self.assertEqual(output["flags"], [])
        self.assertGreater(output["metrics"]["topology_stability_score"], 0.5)
        validate_document("diagnostic_probe_output", output)

    def test_flags_component_crossing_and_bend_instability(self) -> None:
        output = measure_topology_change(
            frames=[
                {"frame": 0, "component_count": 1, "crossing_count": 0, "bend_energy": 1.0},
                {"frame": 1, "component_count": 2, "crossing_count": 3, "bend_energy": 2.0},
            ],
            thresholds=TopologyChangeThresholds(max_component_count_delta=0, max_crossing_count_delta=1),
        )

        self.assertEqual(
            output["flags"],
            ["rope_component_count_changed", "rope_crossing_instability", "rope_bend_energy_spike"],
        )
        self.assertLess(output["metrics"]["topology_stability_score"], 0.5)

    def test_invalid_measurements_fail_closed(self) -> None:
        with self.assertRaisesRegex(PushRopeTopologyChangeProbeError, "FRAME_COUNT_INSUFFICIENT"):
            measure_topology_change(frames=_stable_frames()[:1])
        with self.assertRaisesRegex(PushRopeTopologyChangeProbeError, "INITIAL_BEND_ENERGY_INVALID"):
            measure_topology_change(
                frames=[
                    {"frame": 0, "component_count": 1, "crossing_count": 0, "bend_energy": 0.0},
                    {"frame": 1, "component_count": 1, "crossing_count": 0, "bend_energy": 1.0},
                ]
            )
        with self.assertRaisesRegex(PushRopeTopologyChangeProbeError, "EVIDENCE_REFS_INVALID"):
            measure_topology_change(frames=_stable_frames(), evidence_refs=["bad-ref"])


def _stable_frames() -> list[dict[str, float]]:
    return [
        {"frame": 0, "component_count": 1, "crossing_count": 0, "bend_energy": 1.00},
        {"frame": 1, "component_count": 1, "crossing_count": 0, "bend_energy": 1.10},
    ]


if __name__ == "__main__":
    unittest.main()
