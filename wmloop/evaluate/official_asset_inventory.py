"""Build a read-only official ACWM-Phys asset inventory.

This report joins three pieces of evidence that must be kept together before
unblocking M4: the official checkpoint publication claim, the locally installed
checkpoint step embedded in each ``latest.pt``, and the long-horizon ground
truth coverage available from local metadata.  It never downloads checkpoint
bodies and never mutates the active checkpoint root.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
import json
import os
import shutil
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from wmloop.acwm_data import CANONICAL_ACWM_ENVIRONMENTS, AcwmEnvironmentSpec
from wmloop.archive.store import ArchiveStore, ContentAddressedStore
from wmloop.diagnose.horizon_availability import (
    _metadata_lengths,
    _temporal_rate,
    summarize_horizon_availability,
)
from wmloop.evaluate.checkpoint_audit import _checkpoint_step, _huggingface_download_metadata


class OfficialAssetInventoryError(RuntimeError):
    """Official asset inventory evidence could not be produced safely."""


UrlOpen = Callable[..., Any]


def generate_official_asset_inventory(
    *,
    repo_root: Path,
    checkpoint_root: Path,
    data_root: Path,
    output_root: Path,
    repository_id: str = "t1an/ACWM-Phys-checkpoints",
    dataset_repository_id: str = "t1an/ACWM-Phys",
    expected_step: int = 100000,
    horizons: Sequence[int] = (16, 32, 48, 64),
    splits: Sequence[str] = ("ind_test", "ood_test"),
    remote_timeout_seconds: float = 15.0,
    skip_remote_probes: bool = False,
    external_metadata_manifest: Path | None = None,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
    opener: UrlOpen = urllib.request.urlopen,
) -> dict[str, object]:
    """Write a durable all-environment ACWM asset inventory."""

    if not isinstance(expected_step, int) or isinstance(expected_step, bool) or expected_step < 1:
        raise OfficialAssetInventoryError("OFFICIAL_ASSET_INVENTORY_EXPECTED_STEP_INVALID")
    if not horizons or any(not isinstance(item, int) or isinstance(item, bool) or item < 1 for item in horizons):
        raise OfficialAssetInventoryError("OFFICIAL_ASSET_INVENTORY_HORIZONS_INVALID")
    if not splits or any(not isinstance(split, str) or not split for split in splits):
        raise OfficialAssetInventoryError("OFFICIAL_ASSET_INVENTORY_SPLITS_INVALID")
    if remote_timeout_seconds < 0:
        raise OfficialAssetInventoryError("OFFICIAL_ASSET_INVENTORY_TIMEOUT_INVALID")
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise OfficialAssetInventoryError("OFFICIAL_ASSET_INVENTORY_OUTPUT_EXISTS")

    repo = Path(repo_root).resolve(strict=True)
    checkpoints = Path(checkpoint_root).resolve(strict=True)
    data = Path(data_root).resolve(strict=True)
    external_metadata = _load_external_metadata_manifest(
        external_metadata_manifest,
        repository_id=repository_id,
    )
    model_api_probe = (
        {"status": "skipped", "reason": "skip_remote_probes"}
        if skip_remote_probes
        else _probe_model_api(repository_id=repository_id, timeout_seconds=remote_timeout_seconds, opener=opener)
    )

    records: list[dict[str, object]] = []
    for spec in CANONICAL_ACWM_ENVIRONMENTS:
        records.append(
            _environment_record(
                spec,
                repo_root=repo,
                checkpoint_root=checkpoints,
                data_root=data,
                repository_id=repository_id,
                expected_step=expected_step,
                horizons=tuple(int(item) for item in horizons),
                splits=tuple(splits),
                remote_timeout_seconds=remote_timeout_seconds,
                skip_remote_probes=skip_remote_probes,
                external_metadata=external_metadata["records_by_path"],
                model_api_probe=model_api_probe,
                opener=opener,
            )
        )

    mismatch_records = [record for record in records if record["checkpoint"]["local_step_status"] != "pass"]
    candidate_count = sum(1 for record in mismatch_records if record["checkpoint"]["resolution_status"] == "remote_checkpoint_candidate_requires_quarantine")
    current_matches_blocked_count = sum(
        1 for record in mismatch_records if record["checkpoint"]["resolution_status"] == "official_current_matches_blocked_local_mismatch"
    )
    remote_unavailable_count = sum(1 for record in mismatch_records if record["checkpoint"]["official_publish_status"] == "official_metadata_unavailable")
    horizon_limited_environment_count = sum(1 for record in records if record["horizon_gt"]["state"] != "ready")
    report = {
        "schema_version": 1,
        "artifact_type": "acwm-official-asset-inventory",
        "state": _overall_state(
            mismatch_count=len(mismatch_records),
            remote_checkpoint_candidate_count=candidate_count,
            official_current_matches_blocked_count=current_matches_blocked_count,
            horizon_limited_environment_count=horizon_limited_environment_count,
        ),
        "observed_at_utc": _utc_now(),
        "repo_root": str(repo),
        "checkpoint_root": str(checkpoints),
        "data_root": str(data),
        "repository_id": repository_id,
        "dataset_repository_id": dataset_repository_id,
        "official_claim": {
            "model_card_url": f"https://huggingface.co/{repository_id}",
            "dataset_card_url": f"https://huggingface.co/datasets/{dataset_repository_id}",
            "local_vendor_readme_path": str(repo / "vendor" / "ACWM-Phys" / "README.md"),
            "claimed_checkpoint_step": expected_step,
            "claim_summary": "The official ACWM-Phys checkpoint model card presents the released DiT-S checkpoints as 100k-step checkpoints.",
        },
        "expected_step": expected_step,
        "horizons": [int(item) for item in horizons],
        "splits": list(splits),
        "environment_count": len(records),
        "checkpoint_mismatch_count": len(mismatch_records),
        "remote_checkpoint_candidate_count": candidate_count,
        "official_current_matches_blocked_local_mismatch_count": current_matches_blocked_count,
        "remote_unavailable_mismatch_count": remote_unavailable_count,
        "horizon_limited_environment_count": horizon_limited_environment_count,
        "horizon_limited_records_count": sum(
            1
            for record in records
            for split_record in record["horizon_gt"]["split_records"]  # type: ignore[index]
            if split_record["state"] != "ready"
        ),
        "downloaded_checkpoint_bytes": False,
        "active_checkpoint_mutated": False,
        "active_goal_config_mutated": False,
        "vendor_mutated": False,
        "gpu_execution": False,
        "m4_launch_allowed": False,
        "formal_training_allowed": False,
        "remote_repository_probe": _summarize_repository_probe(model_api_probe),
        "external_metadata_manifest": external_metadata["summary"],
        "records": records,
        "next_actions": _next_actions(
            current_matches_blocked_count=current_matches_blocked_count,
            candidate_count=candidate_count,
            remote_unavailable_count=remote_unavailable_count,
            horizon_limited_environment_count=horizon_limited_environment_count,
        ),
        "limitations": [
            "This inventory reads remote metadata only through model API/HEAD probes or an explicit external metadata manifest.",
            "It does not download checkpoint bodies, install checkpoint candidates, change active goal configs, or authorize M4.",
            "A remote hash difference is only a quarantine candidate until the checkpoint body is downloaded outside the active root and passes step validation.",
        ],
    }
    return _write_report_bundle(report=report, output_root=destination, archive_db=archive_db, cas_root=cas_root)


def _environment_record(
    spec: AcwmEnvironmentSpec,
    *,
    repo_root: Path,
    checkpoint_root: Path,
    data_root: Path,
    repository_id: str,
    expected_step: int,
    horizons: Sequence[int],
    splits: Sequence[str],
    remote_timeout_seconds: float,
    skip_remote_probes: bool,
    external_metadata: Mapping[str, Mapping[str, object]],
    model_api_probe: Mapping[str, object],
    opener: UrlOpen,
) -> dict[str, object]:
    checkpoint = _regular_file(checkpoint_root / spec.checkpoint_relative_path)
    local_metadata = _huggingface_download_metadata(checkpoint_root, spec.checkpoint_relative_path)
    local_hash = _normalize_hash(local_metadata.get("etag_or_lfs_sha256")) if isinstance(local_metadata, Mapping) else None
    observed_step = _checkpoint_step(checkpoint)
    local_step_status = "pass" if observed_step == expected_step else "step_mismatch"
    file_head_probe = (
        {"status": "skipped", "reason": "skip_remote_probes"}
        if skip_remote_probes
        else _probe_file_head(
            repository_id=repository_id,
            relative_path=spec.checkpoint_relative_path,
            timeout_seconds=remote_timeout_seconds,
            opener=opener,
        )
    )
    external_manifest_probe = _external_metadata_probe(relative_path=spec.checkpoint_relative_path, external_metadata=external_metadata)
    model_sibling_probe = _model_sibling_probe(model_api_probe, spec.checkpoint_relative_path)
    remote_hashes = sorted(_remote_hashes(model_sibling_probe, file_head_probe, external_manifest_probe))
    official_publish_status = _official_publish_status(
        local_hash=local_hash,
        remote_hashes=remote_hashes,
        remote_success=any(
            probe.get("status") == "ok" for probe in (model_sibling_probe, file_head_probe, external_manifest_probe)
        ),
        probes=(model_sibling_probe, file_head_probe, external_manifest_probe),
    )
    return {
        "environment": spec.environment,
        "dataset_relative_path": spec.dataset_relative_path,
        "checkpoint": {
            "checkpoint_relative_path": spec.checkpoint_relative_path,
            "checkpoint_path": str(checkpoint),
            "local_exists": True,
            "size_bytes": checkpoint.stat().st_size,
            "expected_step": expected_step,
            "observed_step": observed_step,
            "local_step_status": local_step_status,
            "local_hf_download_metadata": local_metadata,
            "local_hf_hash": local_hash,
            "remote_hashes": remote_hashes,
            "official_publish_status": official_publish_status,
            "resolution_status": _checkpoint_resolution_status(
                local_step_status=local_step_status,
                official_publish_status=official_publish_status,
            ),
            "remote_probes": {
                "model_api": model_sibling_probe,
                "file_head": file_head_probe,
                "external_manifest": external_manifest_probe,
            },
        },
        "horizon_gt": _horizon_summary_for_environment(
            spec,
            repo_root=repo_root,
            data_root=data_root,
            horizons=horizons,
            splits=splits,
        ),
    }


def _regular_file(path: Path) -> Path:
    if path.is_symlink() or not path.is_file():
        raise OfficialAssetInventoryError(f"OFFICIAL_ASSET_INVENTORY_CHECKPOINT_MISSING:{path}")
    return path.resolve(strict=True)


def _horizon_summary_for_environment(
    spec: AcwmEnvironmentSpec,
    *,
    repo_root: Path,
    data_root: Path,
    horizons: Sequence[int],
    splits: Sequence[str],
) -> dict[str, object]:
    temporal_rate = _temporal_rate(repo_root / "vendor" / "ACWM-Phys", spec.environment)
    split_records = []
    max_supported_by_split: dict[str, int | None] = {}
    for split in splits:
        entry_lengths, action_counts, metadata_path = _metadata_lengths(data_root, spec.dataset_relative_path, split)
        record = summarize_horizon_availability(
            environment=spec.environment,
            split=split,
            horizons=horizons,
            entry_lengths=entry_lengths,
            action_counts=action_counts,
            temporal_compress_rate=temporal_rate,
        )
        supported = [int(value) for value in record["supported_horizons"]]
        max_supported_by_split[split] = max(supported) if supported else None
        record["metadata_path"] = str(metadata_path)
        split_records.append(record)
    common_supported = [
        int(horizon)
        for horizon in horizons
        if all(str(horizon) in set(record["supported_horizons"]) for record in split_records)
    ]
    return {
        "state": "ready" if all(record["state"] == "ready" for record in split_records) else "limited",
        "temporal_compress_rate": temporal_rate,
        "max_supported_horizon_by_split": max_supported_by_split,
        "common_supported_horizons": common_supported,
        "common_max_supported_horizon": max(common_supported) if common_supported else None,
        "split_records": split_records,
    }


def _load_external_metadata_manifest(
    path: Path | None,
    *,
    repository_id: str,
) -> dict[str, object]:
    if path is None:
        return {"summary": {"state": "not_provided"}, "records_by_path": {}}
    resolved = Path(path).resolve(strict=True)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OfficialAssetInventoryError("OFFICIAL_ASSET_INVENTORY_EXTERNAL_METADATA_INVALID") from exc
    if not isinstance(payload, Mapping):
        raise OfficialAssetInventoryError("OFFICIAL_ASSET_INVENTORY_EXTERNAL_METADATA_INVALID")
    if payload.get("artifact_type") != "acwm-hf-external-checkpoint-metadata" or payload.get("schema_version") != 1:
        raise OfficialAssetInventoryError("OFFICIAL_ASSET_INVENTORY_EXTERNAL_METADATA_INVALID")
    raw_records = payload.get("records")
    if not isinstance(raw_records, list):
        raise OfficialAssetInventoryError("OFFICIAL_ASSET_INVENTORY_EXTERNAL_METADATA_INVALID")
    records_by_path: dict[str, Mapping[str, object]] = {}
    for item in raw_records:
        if not isinstance(item, Mapping) or item.get("repository_id") != repository_id:
            raise OfficialAssetInventoryError("OFFICIAL_ASSET_INVENTORY_EXTERNAL_METADATA_INVALID")
        relative_path = item.get("checkpoint_relative_path")
        lfs_sha256 = _normalize_hash(item.get("lfs_sha256"))
        if not isinstance(relative_path, str) or not relative_path or lfs_sha256 is None:
            raise OfficialAssetInventoryError("OFFICIAL_ASSET_INVENTORY_EXTERNAL_METADATA_INVALID")
        normalized = dict(item)
        normalized["lfs_sha256"] = lfs_sha256
        records_by_path[relative_path] = normalized
    return {
        "summary": {
            "state": "provided",
            "path": str(resolved),
            "artifact_type": payload.get("artifact_type"),
            "observed_at_utc": payload.get("observed_at_utc"),
            "record_count": len(records_by_path),
            "source_urls": payload.get("source_urls", []),
        },
        "records_by_path": records_by_path,
    }


def _external_metadata_probe(
    *,
    relative_path: str,
    external_metadata: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    if not external_metadata:
        return {"status": "not_provided"}
    record = external_metadata.get(relative_path)
    if record is None:
        return {"status": "missing", "reason": "relative_path_not_in_external_metadata"}
    lfs_sha256 = _normalize_hash(record.get("lfs_sha256"))
    if lfs_sha256 is None:
        return {"status": "invalid", "reason": "invalid_lfs_sha256"}
    sibling: dict[str, object] = {"rfilename": relative_path, "lfs": {"sha256": lfs_sha256}}
    size_bytes = record.get("size_bytes")
    if isinstance(size_bytes, int) and not isinstance(size_bytes, bool) and size_bytes >= 0:
        sibling["size"] = size_bytes
        sibling["lfs"]["size"] = size_bytes  # type: ignore[index]
    return {
        "status": "ok",
        "source": "external_metadata_manifest",
        "url": record.get("source_url"),
        "observed_at_utc": record.get("observed_at_utc"),
        "observed_via": record.get("observed_via"),
        "download_commit": record.get("download_commit"),
        "commit": record.get("commit"),
        "local_content_sha256_verified": record.get("local_content_sha256_verified"),
        "live_current_identity_also_captured": record.get("live_current_identity_also_captured"),
        "repository_sha": record.get("repository_sha"),
        "sibling": sibling,
    }


def _probe_model_api(
    *,
    repository_id: str,
    timeout_seconds: float,
    opener: UrlOpen,
) -> dict[str, object]:
    url = f"https://huggingface.co/api/models/{repository_id}"
    try:
        with opener(url, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        return _probe_error(url, exc)
    if not isinstance(payload, Mapping):
        return {"status": "invalid", "url": url, "reason": "non_mapping_payload"}
    siblings_by_path: dict[str, object] = {}
    siblings = payload.get("siblings")
    if isinstance(siblings, list):
        for item in siblings:
            if isinstance(item, Mapping) and isinstance(item.get("rfilename"), str):
                siblings_by_path[item["rfilename"]] = _summarize_sibling(item)
    return {
        "status": "ok",
        "url": url,
        "repository_sha": payload.get("sha"),
        "last_modified": payload.get("lastModified"),
        "sibling_count": len(siblings_by_path),
        "siblings_by_path": siblings_by_path,
    }


def _model_sibling_probe(model_api_probe: Mapping[str, object], relative_path: str) -> dict[str, object]:
    if model_api_probe.get("status") != "ok":
        return dict(model_api_probe)
    siblings = model_api_probe.get("siblings_by_path")
    sibling = siblings.get(relative_path) if isinstance(siblings, Mapping) else None
    return {
        "status": "ok",
        "url": model_api_probe.get("url"),
        "repository_sha": model_api_probe.get("repository_sha"),
        "last_modified": model_api_probe.get("last_modified"),
        "sibling": sibling,
    }


def _probe_file_head(
    *,
    repository_id: str,
    relative_path: str,
    timeout_seconds: float,
    opener: UrlOpen,
) -> dict[str, object]:
    url = f"https://huggingface.co/{repository_id}/resolve/main/{relative_path}"
    request = urllib.request.Request(url, method="HEAD")
    try:
        with opener(request, timeout=timeout_seconds) as response:
            headers = {key.lower(): value for key, value in response.headers.items()}
            status_code = getattr(response, "status", None)
    except Exception as exc:
        return _probe_error(url, exc)
    selected_headers = {
        key: headers[key]
        for key in (
            "etag",
            "x-linked-etag",
            "x-repo-commit",
            "x-linked-size",
            "content-length",
            "location",
        )
        if key in headers
    }
    return {"status": "ok", "url": url, "status_code": status_code, "headers": selected_headers}


def _probe_error(url: str, exc: Exception) -> dict[str, object]:
    reason = getattr(exc, "reason", None)
    return {
        "status": "error",
        "url": url,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "reason": str(reason) if reason is not None else None,
    }


def _summarize_sibling(sibling: Mapping[str, Any] | None) -> dict[str, object] | None:
    if sibling is None:
        return None
    summary: dict[str, object] = {"rfilename": sibling.get("rfilename")}
    if "size" in sibling:
        summary["size"] = sibling["size"]
    lfs = sibling.get("lfs")
    if isinstance(lfs, Mapping):
        summary["lfs"] = {key: lfs.get(key) for key in ("sha256", "size", "pointerSize") if key in lfs}
    return summary


def _remote_hashes(*probes: Mapping[str, object]) -> set[str]:
    values: set[str] = set()
    for probe in probes:
        if probe.get("status") != "ok":
            continue
        sibling = probe.get("sibling")
        if isinstance(sibling, Mapping):
            lfs = sibling.get("lfs")
            if isinstance(lfs, Mapping):
                normalized = _normalize_hash(lfs.get("sha256"))
                if normalized is not None:
                    values.add(normalized)
        headers = probe.get("headers")
        if isinstance(headers, Mapping):
            for key in ("x-linked-etag", "etag"):
                normalized = _normalize_hash(headers.get(key))
                if normalized is not None:
                    values.add(normalized)
    return values


def _normalize_hash(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip().strip('"')
    if len(stripped) == 64 and all(character in "0123456789abcdef" for character in stripped.lower()):
        return stripped.lower()
    return None


def _official_publish_status(
    *,
    local_hash: str | None,
    remote_hashes: Sequence[str],
    remote_success: bool,
    probes: Sequence[Mapping[str, object]],
) -> str:
    remote_set = set(remote_hashes)
    if local_hash is not None and local_hash in remote_set:
        if _has_live_current_evidence(probes):
            return "official_current_matches_local"
        return "official_download_metadata_matches_local"
    if local_hash is not None and remote_set and local_hash not in remote_set:
        if _has_live_current_evidence(probes):
            return "official_current_differs_from_local"
        return "official_download_metadata_differs_from_local"
    if not remote_success:
        return "official_metadata_unavailable"
    return "official_metadata_inconclusive"


def _checkpoint_resolution_status(*, local_step_status: str, official_publish_status: str) -> str:
    if local_step_status == "pass":
        return "local_checkpoint_step_ready"
    if official_publish_status in {"official_current_matches_local", "official_download_metadata_matches_local"}:
        return "official_current_matches_blocked_local_mismatch"
    if official_publish_status == "official_current_differs_from_local":
        return "remote_checkpoint_candidate_requires_quarantine"
    if official_publish_status == "official_download_metadata_differs_from_local":
        return "download_metadata_candidate_requires_live_remote_check"
    if official_publish_status == "official_metadata_unavailable":
        return "official_metadata_unavailable_for_mismatch"
    return "checkpoint_source_inconclusive"


def _has_live_current_evidence(probes: Sequence[Mapping[str, object]]) -> bool:
    for probe in probes:
        if probe.get("status") != "ok":
            continue
        if probe.get("source") != "external_metadata_manifest":
            return True
        observed_via = probe.get("observed_via")
        if observed_via != "local_huggingface_download_metadata":
            return True
        if probe.get("live_current_identity_also_captured"):
            return True
    return False


def _overall_state(
    *,
    mismatch_count: int,
    remote_checkpoint_candidate_count: int,
    official_current_matches_blocked_count: int,
    horizon_limited_environment_count: int,
) -> str:
    if remote_checkpoint_candidate_count:
        return "remote_checkpoint_candidate_found"
    if official_current_matches_blocked_count:
        return "official_current_checkpoint_step_mismatch"
    if mismatch_count:
        return "checkpoint_step_unresolved"
    if horizon_limited_environment_count:
        return "horizon_protocol_limited"
    return "ready"


def _next_actions(
    *,
    current_matches_blocked_count: int,
    candidate_count: int,
    remote_unavailable_count: int,
    horizon_limited_environment_count: int,
) -> list[str]:
    if candidate_count:
        return [
            "Download differing remote checkpoint candidates into results/quarantine/checkpoints, not over the active checkpoint root.",
            "Run checkpoint_quarantine validate and install only after explicit human approval.",
            "Rerun M0 checkpoint audit, strict launch guard, baseline reproduction, and phase gate before any M4 launch.",
        ]
    if current_matches_blocked_count:
        return [
            "Do not redownload current HuggingFace main as the fix for mismatched environments whose remote hash matches the blocked local file.",
            "Treat the mismatch as an official asset/source defect until a verified alternate revision, publisher fix, or approved self-continued checkpoint is available.",
            "Keep M4 blocked and continue only with read-only diagnostics or explicitly non-M4 smoke runs.",
        ]
    if remote_unavailable_count:
        return [
            "Resolve official remote metadata access or provide an external official metadata manifest for the mismatched checkpoints.",
            "Do not overwrite active checkpoints while the current official file identity is unknown.",
        ]
    if horizon_limited_environment_count:
        return [
            "Resolve the protocol/data decision for limited long-horizon ground truth before claiming 8-environment 16/32/48/64 coverage.",
            "Rerun M1/M3 evidence and phase gate after any approved protocol or data change.",
        ]
    return ["No official asset inventory blocker is present."]


def _summarize_repository_probe(probe: Mapping[str, object]) -> dict[str, object]:
    result = {key: probe.get(key) for key in ("status", "url", "repository_sha", "last_modified", "sibling_count", "error_type", "error", "reason") if key in probe}
    siblings = probe.get("siblings_by_path")
    if isinstance(siblings, Mapping):
        result["target_checkpoint_count"] = sum(1 for value in siblings if str(value).endswith("/latest.pt"))
    return result


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# ACWM Official Asset Inventory",
        "",
        f"State: `{report['state']}`",
        f"Observed at: `{report['observed_at_utc']}`",
        f"Official checkpoint claim: `{report['expected_step']}` steps",
        f"Checkpoint mismatches: `{report['checkpoint_mismatch_count']}`",
        f"Remote candidates: `{report['remote_checkpoint_candidate_count']}`",
        f"Remote current matches blocked local: `{report['official_current_matches_blocked_local_mismatch_count']}`",
        f"Horizon-limited environments: `{report['horizon_limited_environment_count']}`",
        f"Downloaded checkpoint bytes: `{report['downloaded_checkpoint_bytes']}`",
        f"Active checkpoint mutated: `{report['active_checkpoint_mutated']}`",
        f"M4 launch allowed: `{report['m4_launch_allowed']}`",
        "",
        "## Checkpoints",
        "",
        "| Environment | Observed Step | Expected Step | Local Step | Official Publish | Resolution | Local Hash | Size |",
        "|:--|--:|--:|:--|:--|:--|:--|--:|",
    ]
    for record in report["records"]:
        checkpoint = record["checkpoint"]
        lines.append(
            f"| {record['environment']} | {checkpoint['observed_step']} | {checkpoint['expected_step']} | "
            f"`{checkpoint['local_step_status']}` | `{checkpoint['official_publish_status']}` | "
            f"`{checkpoint['resolution_status']}` | `{checkpoint['local_hf_hash']}` | {checkpoint['size_bytes']} |"
        )
    lines.extend(
        [
            "",
            "## Horizon GT",
            "",
            "| Environment | State | Common Supported | Common Max | Split Max Horizons |",
            "|:--|:--|:--|--:|:--|",
        ]
    )
    for record in report["records"]:
        horizon = record["horizon_gt"]
        split_max = ",".join(f"{split}:{value}" for split, value in horizon["max_supported_horizon_by_split"].items())
        common = ",".join(str(value) for value in horizon["common_supported_horizons"]) or "none"
        lines.append(
            f"| {record['environment']} | `{horizon['state']}` | {common} | "
            f"{horizon['common_max_supported_horizon']} | {split_max} |"
        )
    lines.extend(["", "## Next Actions", ""])
    lines.extend(f"- {item}" for item in report["next_actions"])
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    return "\n".join(lines) + "\n"


def _render_csv(report: Mapping[str, Any]) -> str:
    stream = io.StringIO()
    writer = csv.DictWriter(
        stream,
        fieldnames=(
            "environment",
            "checkpoint_relative_path",
            "observed_step",
            "expected_step",
            "local_step_status",
            "official_publish_status",
            "resolution_status",
            "local_hf_hash",
            "size_bytes",
            "horizon_state",
            "common_supported_horizons",
            "common_max_supported_horizon",
        ),
    )
    writer.writeheader()
    for record in report["records"]:
        checkpoint = record["checkpoint"]
        horizon = record["horizon_gt"]
        writer.writerow(
            {
                "environment": record["environment"],
                "checkpoint_relative_path": checkpoint["checkpoint_relative_path"],
                "observed_step": checkpoint["observed_step"],
                "expected_step": checkpoint["expected_step"],
                "local_step_status": checkpoint["local_step_status"],
                "official_publish_status": checkpoint["official_publish_status"],
                "resolution_status": checkpoint["resolution_status"],
                "local_hf_hash": checkpoint["local_hf_hash"],
                "size_bytes": checkpoint["size_bytes"],
                "horizon_state": horizon["state"],
                "common_supported_horizons": " ".join(str(value) for value in horizon["common_supported_horizons"]),
                "common_max_supported_horizon": horizon["common_max_supported_horizon"],
            }
        )
    return stream.getvalue()


def _write_report_bundle(
    *,
    report: Mapping[str, object],
    output_root: Path,
    archive_db: Path | None,
    cas_root: Path | None,
) -> dict[str, object]:
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise OfficialAssetInventoryError("OFFICIAL_ASSET_INVENTORY_OUTPUT_EXISTS")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    report_bytes = _canonical_json_bytes(report)
    markdown_bytes = _render_markdown(report).encode("utf-8")
    csv_bytes = _render_csv(report).encode("utf-8")
    try:
        temporary.mkdir(mode=0o700)
        _write_bytes_atomic(temporary / "official-asset-inventory.json", report_bytes)
        _write_bytes_atomic(temporary / "official-asset-inventory.md", markdown_bytes)
        _write_bytes_atomic(temporary / "official-asset-inventory.csv", csv_bytes)
        archive = ArchiveStore(archive_db) if archive_db is not None else None
        cas_refs: dict[str, str] = {}
        if archive is not None or cas_root is not None:
            root = cas_root if cas_root is not None else Path(archive_db).resolve().parent  # type: ignore[union-attr]
            cas = ContentAddressedStore(Path(root))
            for name, payload, media_type in (
                ("official_asset_inventory_json", report_bytes, "application/json"),
                ("official_asset_inventory_markdown", markdown_bytes, "text/markdown"),
                ("official_asset_inventory_csv", csv_bytes, "text/csv"),
            ):
                ref = cas.put_bytes(payload, media_type=media_type).uri
                cas_refs[name] = ref
                if archive is not None:
                    archive.record_artifact_reference(ref)
        manifest = {
            "schema_version": 1,
            "artifact_type": "acwm-official-asset-inventory-manifest",
            "state": report["state"],
            "observed_at_utc": report["observed_at_utc"],
            "repository_id": report["repository_id"],
            "dataset_repository_id": report["dataset_repository_id"],
            "expected_step": report["expected_step"],
            "environment_count": report["environment_count"],
            "checkpoint_mismatch_count": report["checkpoint_mismatch_count"],
            "remote_checkpoint_candidate_count": report["remote_checkpoint_candidate_count"],
            "official_current_matches_blocked_local_mismatch_count": report["official_current_matches_blocked_local_mismatch_count"],
            "horizon_limited_environment_count": report["horizon_limited_environment_count"],
            "downloaded_checkpoint_bytes": report["downloaded_checkpoint_bytes"],
            "active_checkpoint_mutated": report["active_checkpoint_mutated"],
            "m4_launch_allowed": report["m4_launch_allowed"],
            "formal_training_allowed": report["formal_training_allowed"],
            "report_path": str(destination / "official-asset-inventory.json"),
            "markdown_path": str(destination / "official-asset-inventory.md"),
            "csv_path": str(destination / "official-asset-inventory.csv"),
            "cas_refs": cas_refs,
        }
        _write_bytes_atomic(temporary / "manifest.json", _canonical_json_bytes(manifest))
        os.replace(temporary, destination)
        return manifest
    except Exception:
        if temporary.is_symlink():
            temporary.unlink()
        elif temporary.exists():
            shutil.rmtree(temporary)
        raise


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise OfficialAssetInventoryError("OFFICIAL_ASSET_INVENTORY_OUTPUT_EXISTS")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="build official ACWM asset inventory")
    run.add_argument("--repo-root", type=Path, default=Path("."))
    run.add_argument("--checkpoint-root", type=Path, required=True)
    run.add_argument("--data-root", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--repository-id", default="t1an/ACWM-Phys-checkpoints")
    run.add_argument("--dataset-repository-id", default="t1an/ACWM-Phys")
    run.add_argument("--expected-step", type=int, default=100000)
    run.add_argument("--horizons", type=int, nargs="+", default=[16, 32, 48, 64])
    run.add_argument("--splits", nargs="+", default=["ind_test", "ood_test"])
    run.add_argument("--remote-timeout-seconds", type=float, default=15.0)
    run.add_argument("--skip-remote-probes", action="store_true")
    run.add_argument("--external-metadata-manifest", type=Path)
    run.add_argument("--archive-db", type=Path)
    run.add_argument("--cas-root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        manifest = generate_official_asset_inventory(
            repo_root=args.repo_root,
            checkpoint_root=args.checkpoint_root,
            data_root=args.data_root,
            output_root=args.output_root,
            repository_id=args.repository_id,
            dataset_repository_id=args.dataset_repository_id,
            expected_step=args.expected_step,
            horizons=tuple(args.horizons),
            splits=tuple(args.splits),
            remote_timeout_seconds=args.remote_timeout_seconds,
            skip_remote_probes=args.skip_remote_probes,
            external_metadata_manifest=args.external_metadata_manifest,
            archive_db=args.archive_db,
            cas_root=args.cas_root,
        )
        print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
