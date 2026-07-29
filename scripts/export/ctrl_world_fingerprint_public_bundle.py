#!/usr/bin/env python3
"""Export a path-sanitized Ctrl-World target-local IRG example."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import statistics
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


class CtrlWorldFingerprintPublicBundleError(ValueError):
    """The settled fingerprint cannot be exported as a public example."""


def export_ctrl_world_fingerprint_public_bundle(
    *, settlement_root: Path,
    output_root: Path,
) -> dict[str, object]:
    settlement_dir = Path(settlement_root).resolve(strict=True)
    settlement = _load_mapping(settlement_dir / "settlement.json", "SETTLEMENT_INVALID")
    if settlement.get("artifact_type") != "verdiwm-ctrl-world-fingerprint-settlement":
        raise CtrlWorldFingerprintPublicBundleError("SETTLEMENT_TYPE_INVALID")
    candidates = settlement.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise CtrlWorldFingerprintPublicBundleError("SETTLEMENT_CANDIDATES_INVALID")

    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise CtrlWorldFingerprintPublicBundleError("PUBLIC_BUNDLE_OUTPUT_EXISTS")
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        public_candidates = []
        dose_rows: list[dict[str, object]] = []
        chart_rows: list[dict[str, object]] = []
        baseline_frames: dict[str, dict[tuple[str, str, int], tuple[float, ...]]] = {}
        baseline_outcome_names: list[str] | None = None
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                raise CtrlWorldFingerprintPublicBundleError("SETTLEMENT_CANDIDATE_INVALID")
            root = Path(str(candidate["fingerprint_root"])).resolve(strict=True)
            campaign = _load_mapping(root / "input-campaign.json", "CAMPAIGN_INVALID")
            report = _load_mapping(root / "target-local-fingerprint.json", "REPORT_INVALID")
            measurements = _load_jsonl(root / "measurements.jsonl")
            campaign_id = str(candidate["campaign_id"])
            if campaign.get("campaign_id") != campaign_id or report.get("campaign_id") != campaign_id:
                raise CtrlWorldFingerprintPublicBundleError("PUBLIC_BUNDLE_CAMPAIGN_MISMATCH")
            if len(measurements) != int(candidate["measurement_count"]):
                raise CtrlWorldFingerprintPublicBundleError("PUBLIC_BUNDLE_MEASUREMENT_COUNT_MISMATCH")
            candidate_dir = temporary / "candidates" / campaign_id
            candidate_dir.mkdir(parents=True)
            sanitized_measurements = []
            grouped: dict[float, list[tuple[float, ...]]] = defaultdict(list)
            baseline_frame: dict[tuple[str, str, int], tuple[float, ...]] = {}
            for row in measurements:
                clean = {key: value for key, value in row.items() if key != "receipt_ref"}
                sanitized_measurements.append(clean)
                dose = float(row["dose"])
                outcomes = tuple(float(value) for value in row["outcomes"])
                grouped[dose].append(outcomes)
                if dose == 0.0:
                    identity = row["identity"]
                    baseline_frame[(str(identity["task_id"]), str(identity["episode_id"]), int(identity["seed"]))] = outcomes
            baseline_frames[campaign_id] = baseline_frame
            (candidate_dir / "input-campaign.json").write_text(_pretty_json(campaign), encoding="utf-8")
            (candidate_dir / "target-local-fingerprint.json").write_text(_pretty_json(report), encoding="utf-8")
            (candidate_dir / "measurements.jsonl").write_text(
                "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in sanitized_measurements),
                encoding="utf-8",
            )
            outcome_names = [str(value) for value in report["chart"]["outcome_names"]]
            if baseline_outcome_names is None:
                baseline_outcome_names = outcome_names
            elif baseline_outcome_names != outcome_names:
                raise CtrlWorldFingerprintPublicBundleError("PUBLIC_BUNDLE_OUTCOME_SCHEMA_MISMATCH")
            for dose, values in sorted(grouped.items()):
                for outcome_index, outcome_name in enumerate(outcome_names):
                    observed = [value[outcome_index] for value in values]
                    dose_rows.append(
                        {
                            "campaign_id": campaign_id,
                            "locality_state": candidate["locality_state"],
                            "dose": dose,
                            "outcome": outcome_name,
                            "mean": statistics.fmean(observed),
                            "sample_std": statistics.stdev(observed) if len(observed) > 1 else 0.0,
                            "repeat_count": len(observed),
                        }
                    )
            chart = report["chart"]
            for outcome_index, outcome_name in enumerate(outcome_names):
                chart_rows.append(
                    {
                        "campaign_id": campaign_id,
                        "locality_state": candidate["locality_state"],
                        "dose_radius": candidate["dose_radius"],
                        "locality_residual": candidate["maximum_residual"],
                        "outcome": outcome_name,
                        "jacobian": chart["jacobian"][outcome_index][0],
                        "response_coordinate": chart["response_coordinate"][outcome_index],
                        "covariance_diagonal": chart["covariance"][outcome_index][outcome_index],
                    }
                )
            public_candidates.append({key: value for key, value in candidate.items() if key != "fingerprint_root"})

        public_settlement = dict(settlement)
        public_settlement["candidates"] = public_candidates
        (temporary / "settlement.json").write_text(_pretty_json(public_settlement), encoding="utf-8")
        baseline_reproducibility = _baseline_reproducibility(
            baseline_frames,
            outcome_names=baseline_outcome_names or [],
        )
        (temporary / "baseline-reproducibility.json").write_text(
            _pretty_json(baseline_reproducibility), encoding="utf-8"
        )
        _write_csv(temporary / "tables/dose-response.csv", dose_rows)
        _write_csv(temporary / "tables/chart-summary.csv", chart_rows)
        bundle = {
            "schema_version": 1,
            "artifact_type": "verdiwm-ctrl-world-target-local-irg-public-bundle",
            "state": settlement["state"],
            "protocol": settlement["protocol"],
            "candidate_count": len(public_candidates),
            "selected_campaign_id": settlement["selected_campaign_id"],
            "cross_backbone_transfer_eligible": settlement["cross_backbone_transfer_eligible"],
            "measurement_count": sum(int(candidate["measurement_count"]) for candidate in public_candidates),
            "baseline_reproducibility_state": baseline_reproducibility["state"],
            "claim_boundary": settlement["claim_boundary"],
        }
        (temporary / "bundle.json").write_text(_pretty_json(bundle), encoding="utf-8")
        (temporary / "README.md").write_text(_readme(public_settlement), encoding="utf-8")
        _assert_public_text(temporary)
        _write_manifest(temporary)
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return bundle


def _readme(settlement: Mapping[str, Any]) -> str:
    selected = settlement.get("selected_campaign_id") or "none (abstained)"
    rows = [
        "# Ctrl-World Target-Local IRG Calibration",
        "",
        f"This example records a paired-dose ACWM predictive-quality calibration on the frozen Ctrl-World `{settlement['protocol']}` split.",
        "It is a target-local response-chart asset, not evidence of model improvement or completed cross-backbone transfer.",
        "",
        f"- Settlement: `{settlement['state']}`",
        f"- Selected campaign: `{selected}`",
        f"- Transfer-eligible chart available: `{str(settlement['cross_backbone_transfer_eligible']).lower()}`",
        "",
        "The wide and narrow candidates preserve `J_X`, the covariance metric, response coordinate, and locality residual.",
        "`tables/dose-response.csv` contains repeat-level aggregate response curves; `tables/chart-summary.csv` is paper-table ready.",
        "",
    ]
    return "\n".join(rows)


def _baseline_reproducibility(
    frames: Mapping[str, Mapping[tuple[str, str, int], tuple[float, ...]]],
    *,
    outcome_names: Sequence[str],
) -> dict[str, object]:
    campaign_ids = sorted(frames)
    if len(campaign_ids) < 2:
        return {
            "schema_version": 1,
            "artifact_type": "verdiwm-ctrl-world-baseline-reproducibility",
            "state": "not_applicable",
            "campaign_ids": campaign_ids,
            "max_absolute_difference": None,
            "absolute_tolerance": 1e-3,
        }
    reference = frames[campaign_ids[0]]
    if not reference or any(set(frames[campaign_id]) != set(reference) for campaign_id in campaign_ids[1:]):
        raise CtrlWorldFingerprintPublicBundleError("PUBLIC_BUNDLE_BASELINE_FRAME_MISMATCH")
    maximum = 0.0
    by_outcome = [0.0 for _ in outcome_names]
    for campaign_id in campaign_ids[1:]:
        for identity, reference_values in reference.items():
            differences = [
                abs(left - right)
                for left, right in zip(reference_values, frames[campaign_id][identity], strict=True)
            ]
            maximum = max(maximum, *differences)
            by_outcome = [max(current, difference) for current, difference in zip(by_outcome, differences, strict=True)]
    tolerance = 1e-3
    state = "exact_match" if maximum == 0.0 else "within_tolerance" if maximum <= tolerance else "mismatch"
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-ctrl-world-baseline-reproducibility",
        "state": state,
        "campaign_ids": campaign_ids,
        "identity_count": len(reference),
        "max_absolute_difference": maximum,
        "absolute_tolerance": tolerance,
        "max_absolute_difference_by_outcome": dict(zip(outcome_names, by_outcome, strict=True)),
    }


def _load_mapping(path: Path, code: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CtrlWorldFingerprintPublicBundleError(f"{code}:{path}") from exc
    if not isinstance(payload, Mapping):
        raise CtrlWorldFingerprintPublicBundleError(f"{code}:{path}")
    return payload


def _load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise CtrlWorldFingerprintPublicBundleError("MEASUREMENT_INVALID")
        rows.append(payload)
    if not rows:
        raise CtrlWorldFingerprintPublicBundleError("MEASUREMENTS_EMPTY")
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise CtrlWorldFingerprintPublicBundleError("PUBLIC_BUNDLE_TABLE_EMPTY")
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(stream.getvalue(), encoding="utf-8")


def _pretty_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, indent=2) + "\n"


def _assert_public_text(root: Path) -> None:
    host_prefixes = ("/" + "mnt" + "/", "/" + "root" + "/")
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".jsonl", ".csv", ".md"}:
            text = path.read_text(encoding="utf-8")
            if any(prefix in text for prefix in host_prefixes):
                raise CtrlWorldFingerprintPublicBundleError("PUBLIC_BUNDLE_LOCAL_PATH_LEAK")


def _write_manifest(root: Path) -> None:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.sha256":
            rows.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root).as_posix()}")
    (root / "MANIFEST.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settlement-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    result = export_ctrl_world_fingerprint_public_bundle(
        settlement_root=args.settlement_root,
        output_root=args.output_root,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
