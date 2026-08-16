#!/usr/bin/env python3
"""Merge identity-sharded Ctrl-World probe results after strict frame checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.run_ctrl_world_local_fingerprint_probe import OUTCOME_NAMES, load_contexts


class LocalFingerprintShardMergeError(ValueError):
    """Probe shards do not form the frozen campaign frame."""


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LocalFingerprintShardMergeError(f"LOCAL_FINGERPRINT_SHARD_INVALID:{path}") from exc
    if not isinstance(payload, Mapping):
        raise LocalFingerprintShardMergeError(f"LOCAL_FINGERPRINT_SHARD_INVALID:{path}")
    return payload


def _identity(raw: object) -> tuple[str, int]:
    if not isinstance(raw, Mapping):
        raise LocalFingerprintShardMergeError("LOCAL_FINGERPRINT_SHARD_IDENTITY_INVALID")
    context_id = raw.get("context_id")
    seed = raw.get("seed")
    if not isinstance(context_id, str) or not context_id or not isinstance(seed, int):
        raise LocalFingerprintShardMergeError("LOCAL_FINGERPRINT_SHARD_IDENTITY_INVALID")
    return context_id, seed


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def merge(
    *,
    campaign_path: Path,
    contexts_path: Path,
    shard_paths: Sequence[Path],
    output_root: Path,
) -> dict[str, object]:
    campaign_path = campaign_path.resolve(strict=True)
    contexts_path = contexts_path.resolve(strict=True)
    campaign = _load_json(campaign_path)
    if campaign.get("artifact_type") != "verdiwm-ctrl-world-local-fingerprint-campaign":
        raise LocalFingerprintShardMergeError("LOCAL_FINGERPRINT_CAMPAIGN_INVALID")
    campaign_id = campaign.get("campaign_id")
    protocol = campaign.get("protocol")
    probe_specs = campaign.get("probe_paths")
    if (
        not isinstance(campaign_id, str)
        or not isinstance(protocol, Mapping)
        or not isinstance(probe_specs, list)
        or len(probe_specs) != 1
        or not isinstance(probe_specs[0], Mapping)
    ):
        raise LocalFingerprintShardMergeError("LOCAL_FINGERPRINT_CAMPAIGN_INVALID")
    probe_id = probe_specs[0].get("probe_id")
    doses = probe_specs[0].get("doses")
    if not isinstance(probe_id, str) or not isinstance(doses, list):
        raise LocalFingerprintShardMergeError("LOCAL_FINGERPRINT_CAMPAIGN_INVALID")
    expected_doses = {float(value) for value in doses}
    expected_contexts = load_contexts(contexts_path)
    expected_identities = {
        (str(context["context_id"]), int(context["seed"])) for context in expected_contexts
    }
    expected_checkpoint = str(Path(str(campaign.get("checkpoint"))).resolve(strict=True))
    contexts_sha256 = _sha256(contexts_path)
    if output_root.exists() or output_root.is_symlink():
        raise LocalFingerprintShardMergeError("LOCAL_FINGERPRINT_MERGE_OUTPUT_EXISTS")

    references: dict[tuple[str, int], Mapping[str, Any]] = {}
    measurements: list[Mapping[str, Any]] = []
    zero_checks: dict[tuple[str, int], Mapping[str, Any]] = {}
    receipts: list[dict[str, object]] = []
    video_source: Path | None = None
    for raw_path in shard_paths:
        path = Path(raw_path).resolve(strict=True)
        shard = _load_json(path)
        shard_input = shard.get("input")
        if (
            shard.get("artifact_type") != "verdiwm-ctrl-world-local-fingerprint-probe-result"
            or shard.get("state") != "ready"
            or shard.get("campaign_id") != campaign_id
            or shard.get("probe_id") != probe_id
            or tuple(shard.get("outcome_names", ())) != OUTCOME_NAMES
            or not isinstance(shard_input, Mapping)
            or str(Path(str(shard_input.get("checkpoint"))).resolve()) != expected_checkpoint
            or shard_input.get("contexts_sha256") != contexts_sha256
            or int(shard_input.get("interact_num", -1)) != int(protocol.get("interact_num", -2))
            or int(shard_input.get("num_inference_steps", -1))
            != int(protocol.get("num_inference_steps", -2))
            or {float(value) for value in shard_input.get("doses", ())} != expected_doses
            or not isinstance(shard.get("runtime"), Mapping)
        ):
            raise LocalFingerprintShardMergeError(f"LOCAL_FINGERPRINT_SHARD_FRAME_MISMATCH:{path}")

        shard_references = shard.get("unwrapped_references")
        shard_measurements = shard.get("measurements")
        shard_checks = shard.get("zero_identity_checks")
        if (
            not isinstance(shard_references, list)
            or not shard_references
            or not isinstance(shard_measurements, list)
            or not isinstance(shard_checks, list)
            or shard.get("hook_activation") != {"state": "passed"}
            or any(not isinstance(row, Mapping) for row in shard_references)
            or any(not isinstance(row, Mapping) for row in shard_measurements)
            or any(not isinstance(row, Mapping) for row in shard_checks)
        ):
            raise LocalFingerprintShardMergeError(f"LOCAL_FINGERPRINT_SHARD_CONTENT_INVALID:{path}")

        shard_reference_by_identity: dict[tuple[str, int], Mapping[str, Any]] = {}
        for row in shard_references:
            identity = _identity(row.get("identity"))
            if identity in shard_reference_by_identity:
                raise LocalFingerprintShardMergeError("LOCAL_FINGERPRINT_SHARD_IDENTITY_DUPLICATE")
            shard_reference_by_identity[identity] = row
        shard_identities = set(shard_reference_by_identity)
        if shard_identities & set(references):
            raise LocalFingerprintShardMergeError("LOCAL_FINGERPRINT_SHARD_IDENTITY_DUPLICATE")
        if len(shard_measurements) != len(shard_identities) * len(expected_doses):
            raise LocalFingerprintShardMergeError(f"LOCAL_FINGERPRINT_SHARD_CONTENT_INVALID:{path}")

        measurement_doses: dict[tuple[str, int], set[float]] = {
            identity: set() for identity in shard_identities
        }
        for row in shard_measurements:
            identity = _identity(row.get("identity"))
            if identity not in measurement_doses:
                raise LocalFingerprintShardMergeError("LOCAL_FINGERPRINT_SHARD_IDENTITY_MISMATCH")
            dose = float(row.get("dose"))
            if dose in measurement_doses[identity]:
                raise LocalFingerprintShardMergeError("LOCAL_FINGERPRINT_SHARD_DOSE_FRAME_MISMATCH")
            measurement_doses[identity].add(dose)
        if any(doses != expected_doses for doses in measurement_doses.values()):
            raise LocalFingerprintShardMergeError("LOCAL_FINGERPRINT_SHARD_DOSE_FRAME_MISMATCH")

        if len(shard_checks) != len(shard_identities):
            raise LocalFingerprintShardMergeError(f"LOCAL_FINGERPRINT_SHARD_CONTENT_INVALID:{path}")
        shard_check_by_identity: dict[tuple[str, int], Mapping[str, Any]] = {}
        for row in shard_checks:
            identity = _identity(row.get("identity"))
            if identity not in shard_identities:
                raise LocalFingerprintShardMergeError("LOCAL_FINGERPRINT_SHARD_IDENTITY_MISMATCH")
            if identity in shard_check_by_identity:
                raise LocalFingerprintShardMergeError("LOCAL_FINGERPRINT_SHARD_IDENTITY_DUPLICATE")
            if row.get("state") != "passed":
                raise LocalFingerprintShardMergeError("LOCAL_FINGERPRINT_SHARD_ZERO_CHECK_FAILED")
            shard_check_by_identity[identity] = row
        if set(shard_check_by_identity) != shard_identities:
            raise LocalFingerprintShardMergeError("LOCAL_FINGERPRINT_SHARD_IDENTITY_MISMATCH")

        references.update(shard_reference_by_identity)
        measurements.extend(shard_measurements)
        zero_checks.update(shard_check_by_identity)
        receipt_identities = [
            {"context_id": identity[0], "seed": identity[1]}
            for identity in sorted(shard_identities)
        ]
        receipt: dict[str, object] = {
            "path": str(path),
            "sha256": _sha256(path),
            "identities": receipt_identities,
            "runtime": shard["runtime"],
        }
        if len(receipt_identities) == 1:
            receipt["identity"] = receipt_identities[0]
        receipts.append(receipt)
        artifact = shard.get("artifacts")
        if video_source is None and isinstance(artifact, Mapping):
            candidate = path.parent / str(artifact.get("zero_dose_rollout", ""))
            if candidate.is_file():
                video_source = candidate

    if set(references) != expected_identities:
        missing = sorted(expected_identities - set(references))
        extra = sorted(set(references) - expected_identities)
        raise LocalFingerprintShardMergeError(
            f"LOCAL_FINGERPRINT_SHARD_COVERAGE_MISMATCH:missing={missing}:extra={extra}"
        )
    if video_source is None:
        raise LocalFingerprintShardMergeError("LOCAL_FINGERPRINT_SHARD_VIDEO_MISSING")

    output_root.mkdir(mode=0o700, parents=True)
    video_target = output_root / "zero-dose-rollout.mp4"
    shutil.copyfile(video_source, video_target)
    dose_order = {0.0: 0, **{dose: index + 1 for index, dose in enumerate(sorted(expected_doses - {0.0}))}}
    result = {
        "schema_version": 1,
        "artifact_type": "verdiwm-ctrl-world-local-fingerprint-probe-result",
        "state": "ready",
        "campaign_id": campaign_id,
        "probe_id": probe_id,
        "base_probe_family": shard["base_probe_family"],
        "outcome_names": list(OUTCOME_NAMES),
        "input": {
            "checkpoint": expected_checkpoint,
            "contexts_json": str(contexts_path),
            "contexts_sha256": contexts_sha256,
            "campaign_json": str(campaign_path),
            "campaign_sha256": _sha256(campaign_path),
            "doses": sorted(expected_doses),
            "interact_num": int(protocol["interact_num"]),
            "num_inference_steps": int(protocol["num_inference_steps"]),
            "shard_receipts": sorted(
                receipts,
                key=lambda row: (
                    row["identities"][0]["context_id"],
                    row["identities"][0]["seed"],
                ),
            ),
        },
        "unwrapped_references": [references[key] for key in sorted(references)],
        "measurements": sorted(
            measurements,
            key=lambda row: (dose_order[float(row["dose"])], _identity(row["identity"])),
        ),
        "zero_identity_checks": [zero_checks[key] for key in sorted(zero_checks)],
        "hook_activation": {"state": "passed"},
        "artifacts": {
            "zero_dose_rollout": video_target.name,
            "zero_dose_rollout_sha256": _sha256(video_target),
        },
        "claim_boundary": campaign.get("claim_scope"),
    }
    result_path = output_root / "result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "artifact_type": "verdiwm-ctrl-world-local-fingerprint-probe-merge-manifest",
        "state": "ready",
        "identity_count": len(expected_identities),
        "measurement_count": len(measurements),
        "result_path": str(result_path.resolve()),
        "result_sha256": _sha256(result_path),
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--shard-result", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = merge(
        campaign_path=args.campaign,
        contexts_path=args.contexts,
        shard_paths=args.shard_result,
        output_root=args.output_root.resolve(),
    )
    print(json.dumps(manifest, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
