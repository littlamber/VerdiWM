from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from wmloop.experiments.selector_projection_compose import compose_selector_projections


class AcwmSelectorProjectionComposeTests(unittest.TestCase):
    def test_composes_explicit_probe_paths_without_changing_other_selectors(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            primary = self._projection(root / "primary.jsonl", ("amp", "attenuation"))
            extension = self._projection(root / "extension.jsonl", ("history",))

            manifest = compose_selector_projections(
                primary_projection=primary,
                primary_path_order=("amp", "attenuation"),
                extension_projections=[(extension, ("history",))],
                output_root=root / "output",
            )

            self.assertEqual(manifest["state"], "ready")
            self.assertEqual(manifest["probe_source_count"], 2)
            rows = [
                json.loads(line)
                for line in (root / "output" / "selector-input-projections.jsonl").read_text().splitlines()
            ]
            irg = next(row for row in rows if row["environment"] == "a" and row["selector"] == "irg")
            self.assertIn("response_coordinate:0:amp", irg["feature_names"])
            self.assertIn("response_coordinate:0:attenuation", irg["feature_names"])
            self.assertIn("response_coordinate:0:history", irg["feature_names"])
            self.assertEqual(irg["composed_probe_count"], 2)
            static = next(row for row in rows if row["environment"] == "a" and row["selector"] == "static_probe")
            self.assertEqual(static["features"], [1.0])

    def _projection(self, path: Path, paths: tuple[str, ...]) -> Path:
        rows = []
        for environment in ("a", "b"):
            for selector in ("environment_label", "static_probe", "raw_response", "irg"):
                if selector != "irg":
                    rows.append(
                        {
                            "environment": environment,
                            "selector": selector,
                            "feature_names": ["value"],
                            "features": [1.0],
                        }
                    )
                    continue
                names = []
                values = []
                for outcome in range(2):
                    for index, _probe_path in enumerate(paths):
                        names.append(f"response_coordinate:{outcome * len(paths) + index}")
                        values.append(float(outcome + index))
                for outcome in range(2):
                    for index, _probe_path in enumerate(paths):
                        names.append(f"covariance_diagonal:{outcome * len(paths) + index}")
                        values.append(0.0)
                names.extend(f"locality:{probe_path}" for probe_path in paths)
                values.extend(0.1 for _ in paths)
                names.extend(f"path_supported:{probe_path}" for probe_path in paths)
                values.extend(1.0 for _ in paths)
                rows.append(
                    {
                        "campaign_id": path.stem,
                        "environment": environment,
                        "selector": selector,
                        "feature_names": names,
                        "features": values,
                    }
                )
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return path


if __name__ == "__main__":
    unittest.main()
