#!/usr/bin/env python3
"""Export a portable bundle from a completed joint ACWM IRG campaign."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from wmloop.experiments._artifacts import canonical_json, write_bundle
from wmloop.experiments.joint_fingerprint import load_joint_campaign, load_joint_sources
from wmloop.geometry.assets import validate_irg_asset


REPO_ROOT = Path(__file__).resolve().parents[2]


class JointIRGExportError(ValueError):
    """The joint campaign cannot support a portable IRG bundle."""


def export_joint_irg_assets(
    *,
    campaign_root: Path,
    joint_campaign_path: Path,
    output_root: Path,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    source_root = Path(campaign_root).resolve(strict=True)
    status = _load_json(source_root / "status.json")
    joint_path = Path(joint_campaign_path).resolve(strict=True)
    joint = load_joint_campaign(joint_path)
    sources = load_joint_sources(joint, repo_root=REPO_ROOT)
    environments = tuple(str(value) for value in sources[0]["environments"])
    if (
        status.get("state") != "ready"
        or status.get("campaign_id") != joint["campaign_id"]
        or set(status.get("return_codes", {}).values()) != {0}
        or set(status.get("return_codes", {})) != set(environments)
    ):
        raise JointIRGExportError("JOINT_IRG_CAMPAIGN_INCOMPLETE")

    files: dict[str, bytes] = {}
    summary_rows: list[dict[str, object]] = []
    asset_paths: dict[str, str] = {}
    for environment in environments:
        env_root = source_root / "environments" / environment
        manifest = _load_json(env_root / "manifest.json")
        asset = _load_json(env_root / "irg-asset.json")
        validate_irg_asset(asset)
        measurements = _load_jsonl(env_root / "measurements.jsonl")
        _validate_environment(environment, manifest, asset, measurements, joint)
        asset_path = f"assets/{environment}.json"
        receipt_path = f"receipts/{environment}.json"
        asset_paths[environment] = asset_path
        files[asset_path] = canonical_json(asset)
        files[receipt_path] = canonical_json(_portable_receipt(manifest))
        covariance = asset["response_covariance"]
        off_diagonal = sum(
            abs(float(covariance[i][j])) > 0.0
            for i in range(len(covariance))
            for j in range(len(covariance))
            if i != j
        )
        summary_rows.append(
            {
                "environment": environment,
                "checkpoint_step": manifest["checkpoint_step"],
                "condition_count": manifest["condition_count"],
                "baseline_condition_count": manifest["baseline_condition_count"],
                "probe_path_count": asset["dimensions"]["probe_path_count"],
                "supported_probe_path_count": asset["supported_probe_path_count"],
                "baseline_group_count": asset["covariance_contract"]["joint_baseline_group_count"],
                "off_diagonal_nonzero_count": off_diagonal,
                "routing_state": asset["routing_state"],
                "covariance_state": asset["transfer_state"],
            }
        )

    index = {
        "schema_version": 1,
        "artifact_type": "verdiwm-acwm-joint-irg-asset-index",
        "state": "ready",
        "campaign_id": joint["campaign_id"],
        "protocol": status["protocol"],
        "generation_mode": joint["generation_mode"],
        "environment_count": len(environments),
        "source_probe_count": len(joint["source_campaigns"]),
        "probe_path_count": len(joint["semantic_paths"]),
        "condition_count": sum(int(row["condition_count"]) for row in summary_rows),
        "baseline_condition_count": sum(int(row["baseline_condition_count"]) for row in summary_rows),
        "single_baseline_group_environment_count": sum(row["baseline_group_count"] == 1 for row in summary_rows),
        "routing_ready_environment_count": sum(row["routing_state"] == "ready" for row in summary_rows),
        "covariance_ready_environment_count": sum(row["covariance_state"] == "ready" for row in summary_rows),
        "probe_path_names": [str(row["path_name"]) for row in joint["semantic_paths"]],
        "asset_paths": asset_paths,
        "receipt_paths": {environment: f"receipts/{environment}.json" for environment in environments},
        "joint_campaign_sha256": _sha256(joint_path),
        "claim_boundary": (
            "All seven ACWM probe paths were measured against one no-hook autoregressive baseline per "
            "environment and seed, so within-chart cross-path covariance is observed. These assets make "
            "ACWM routing and cross-path geometry complete; they do not by themselves establish a "
            "cross-backbone transfer effect."
        ),
    }
    files["index.json"] = canonical_json(index)
    files["tables/asset-summary.csv"] = _csv(summary_rows).encode("utf-8")
    files["README.md"] = _readme(index).encode("utf-8")
    return write_bundle(
        output_root=Path(output_root),
        files=files,
        manifest_fields={
            "artifact_type": "verdiwm-acwm-joint-irg-asset-bundle",
            "state": "ready",
            "campaign_id": joint["campaign_id"],
            "environment_count": len(environments),
            "source_probe_count": len(joint["source_campaigns"]),
            "probe_path_count": len(joint["semantic_paths"]),
            "condition_count": index["condition_count"],
            "baseline_condition_count": index["baseline_condition_count"],
            "single_baseline_group_environment_count": index["single_baseline_group_environment_count"],
            "index_path": "index.json",
        },
        archive_db=archive_db,
        cas_root=cas_root,
    )


def _validate_environment(
    environment: str,
    manifest: Mapping[str, Any],
    asset: Mapping[str, Any],
    measurements: Sequence[Mapping[str, Any]],
    joint: Mapping[str, Any],
) -> None:
    if (
        manifest.get("state") != "ready"
        or manifest.get("campaign_id") != joint["campaign_id"]
        or manifest.get("environment") != environment
        or manifest.get("condition_count") != 75
        or manifest.get("baseline_condition_count") != 3
        or len(measurements) != 75
    ):
        raise JointIRGExportError(f"JOINT_IRG_ENVIRONMENT_INCOMPLETE:{environment}")
    frame = manifest.get("frame_identity")
    if not isinstance(frame, Mapping) or frame.get("generation_mode") != "autoregressive":
        raise JointIRGExportError(f"JOINT_IRG_FRAME_INVALID:{environment}")
    if any(row.get("frame_identity") != frame for row in measurements):
        raise JointIRGExportError(f"JOINT_IRG_FRAME_DRIFT:{environment}")
    baselines = [row for row in measurements if row.get("condition_kind") == "baseline"]
    if len(baselines) != 3 or any(
        row.get("compile_receipt", {}).get("hook_policy") != "no_hook_context"
        for row in baselines
    ):
        raise JointIRGExportError(f"JOINT_IRG_BASELINE_INVALID:{environment}")
    if (
        asset.get("environment") != environment
        or asset.get("dimensions") != {
            "outcome_count": 4,
            "probe_path_count": 7,
            "response_coordinate_count": 28,
        }
        or asset.get("covariance_contract", {}).get("joint_baseline_group_count") != 1
        or asset.get("transfer_state") != "ready"
    ):
        raise JointIRGExportError(f"JOINT_IRG_ASSET_INVALID:{environment}")


def _portable_receipt(manifest: Mapping[str, Any]) -> dict[str, object]:
    return {
        key: value
        for key, value in manifest.items()
        if key not in {"checkpoint", "physical_gpu"}
    }


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise JointIRGExportError(f"JOINT_IRG_JSON_INVALID:{path.name}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise JointIRGExportError(f"JOINT_IRG_JSONL_INVALID:{path.name}")
    return rows


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv(rows: Sequence[Mapping[str, object]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _readme(index: Mapping[str, Any]) -> str:
    return f"""# ACWM-Phys Joint-Frame IRG Assets

This bundle contains one canonical IRG asset for each ACWM-Phys environment,
measured under a shared no-hook autoregressive baseline frame.

- Environments: {index['environment_count']}
- Source probes: {index['source_probe_count']}
- Semantic probe paths: {index['probe_path_count']}
- Measurements: {index['condition_count']}
- Canonical baseline measurements: {index['baseline_condition_count']}
- Single-baseline-group environments: {index['single_baseline_group_environment_count']}

The bundle closes the cross-path covariance gap found in the earlier mixed-mode
atlas. It supports complete within-ACWM routing geometry. A cross-backbone
transfer claim still requires a target-backbone chart and held-out certificate.
"""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--joint-campaign", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--archive-db", type=Path)
    parser.add_argument("--cas-root", type=Path)
    args = parser.parse_args(argv)
    report = export_joint_irg_assets(
        campaign_root=args.campaign_root,
        joint_campaign_path=args.joint_campaign,
        output_root=args.output_root,
        archive_db=args.archive_db,
        cas_root=args.cas_root,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
