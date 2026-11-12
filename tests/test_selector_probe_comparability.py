"""CPU-only regression checks for complete, paired held-out replay comparisons."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from wmloop.experiments.selector_probe_admission import (
    SelectorProbeAdmissionError,
    evaluate_selector_probe_admission,
)


def _replay(root: Path, rows: list[dict[str, object]]) -> Path:
    (root / "tables").mkdir(parents=True)
    (root / "manifest.json").write_text(json.dumps({
        "artifact_type": "verdiwm-acwm-selector-cpu-replay-manifest",
        "transfer_certificate_abstention_count": 0,
        "transfer_work_order_count": 0,
        "probe_evolution_work_order_count": 0,
    }))
    with (root / "tables/candidates.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return root


def _row(target: str = "cloth", seed: int = 101, **kwargs: object) -> dict[str, object]:
    return dict(target_environment=target, seed=seed, selector="irg", rank=1,
                primitive="anchor", predicted_sign="positive", target_positive=False, **kwargs)


@pytest.mark.parametrize("case,code", [
    ("dropped_target", "TARGET_COVERAGE_MISMATCH"),
    ("dropped_seed", "SEED_COVERAGE_MISMATCH"),
    ("changed_label", "TARGET_LABEL_MISMATCH"),
    ("invalid_sign", "TOP1_ROW_INVALID"),
    ("duplicate_seed", "SEED_DUPLICATE"),
])
def test_incomparable_replay_cannot_promote_probe(tmp_path: Path, case: str, code: str) -> None:
    before = [_row(), _row(seed=202), _row(target="rope")]
    after = [dict(row, predicted_sign="negative") for row in before]
    if case == "dropped_target":
        after.pop()
    elif case == "dropped_seed":
        after.pop(1)
    elif case == "changed_label":
        after = [dict(row, target_positive=True) for row in after]
    elif case == "invalid_sign":
        after[-1]["predicted_sign"] = "unknown"
    else:
        after.append(dict(after[0]))
    baseline = _replay(tmp_path / "baseline", before)
    candidate = _replay(tmp_path / "candidate", after)
    atlas = tmp_path / "atlas.json"
    atlas.write_text(json.dumps({"artifact_type": "verdiwm-acwm-fingerprint-atlas-manifest",
                                 "measurement_complete_count": 8, "locality_calibrated_count": 2}))
    affinity = tmp_path / "affinity.json"
    affinity.write_text(json.dumps({"artifact_type": "verdiwm-primitive-probe-affinity"}))
    with pytest.raises(SelectorProbeAdmissionError, match=code):
        evaluate_selector_probe_admission(
            baseline_replay_root=baseline, candidate_replay_root=candidate,
            candidate_atlas_manifest=atlas, candidate_affinity=affinity,
            output_root=tmp_path / "admission",
        )
    assert not (tmp_path / "admission").exists()
