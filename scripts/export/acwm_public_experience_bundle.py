#!/usr/bin/env python3
"""Export a path-safe ACWM experience snapshot for the public repository."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping, Sequence
import uuid


class PublicExperienceBundleError(RuntimeError):
    """The public experience snapshot violated its evidence contract."""


SCREEN_FIELDS = (
    "campaign_id", "environment", "primitive", "seed", "train_steps", "state",
    "verdict", "screen_decision", "primary_metric", "delta_primary_metric",
    "baseline_primary_metric", "candidate_primary_metric", "action_following_enabled",
    "action_following_pass", "action_following_observed", "latest_checkpoint_step",
    "candidate_checkpoint_retained", "candidate_checkpoint_sha256",
    "official_visual_asset_count",
)
HORIZON_FIELDS = (
    "campaign_id", "environment", "primitive", "seed", "train_steps", "horizon",
    "baseline_psnr", "candidate_psnr", "delta_psnr", "baseline_ssim", "candidate_ssim",
    "delta_ssim", "baseline_masked_mse", "candidate_masked_mse", "delta_masked_mse",
)
BEST_FIELDS = (
    "environment", "primitive", "campaign_id", "seed", "train_steps", "screen_decision",
    "delta_primary_metric", "action_following_pass", "official_visual_asset_count",
)
PATH_FIELDS = {
    "candidate_checkpoint", "confirmation_manifest", "event_gate_manifest",
    "official_gate_manifest", "poster_path", "source_spec", "source_video", "video_path",
    "visual_manifest",
}


def export_public_experience_bundle(
    *,
    screen_summary_root: Path,
    showcase_root: Path,
    experience_map_paths: Sequence[Path],
    output_root: Path,
) -> dict[str, object]:
    screen_root = Path(screen_summary_root).resolve(strict=True)
    showcase = Path(showcase_root).resolve(strict=True)
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise PublicExperienceBundleError("PUBLIC_EXPERIENCE_OUTPUT_EXISTS")

    screen_manifest = _load_json(screen_root / "manifest.json")
    showcase_manifest = _load_json(showcase / "manifest.json")
    if screen_manifest.get("artifact_type") != "wmloop-acwm-screen-summary":
        raise PublicExperienceBundleError("PUBLIC_EXPERIENCE_SCREEN_SUMMARY_INVALID")
    if showcase_manifest.get("artifact_type") != "wmloop-acwm-projectpage-showcase-bundle":
        raise PublicExperienceBundleError("PUBLIC_EXPERIENCE_SHOWCASE_INVALID")
    if showcase_manifest.get("state") != "ready" or showcase_manifest.get("case_count") != 4:
        raise PublicExperienceBundleError("PUBLIC_EXPERIENCE_SHOWCASE_NOT_READY")

    maps = [Path(path).resolve(strict=True) for path in experience_map_paths]
    if not maps:
        raise PublicExperienceBundleError("PUBLIC_EXPERIENCE_MAPS_MISSING")
    atlas = _experience_atlas(maps)
    screen_rows = _read_selected_csv(screen_root / "tables" / "screen-trials.csv", SCREEN_FIELDS)
    horizon_rows = _read_selected_csv(screen_root / "tables" / "horizon-metrics.csv", HORIZON_FIELDS)
    best_rows = _read_selected_csv(screen_root / "tables" / "best-by-environment.csv", BEST_FIELDS)
    if len(screen_rows) != int(screen_manifest.get("completed_row_count") or -1):
        raise PublicExperienceBundleError("PUBLIC_EXPERIENCE_SCREEN_COUNT_MISMATCH")
    if len(horizon_rows) != int(screen_manifest.get("horizon_row_count") or -1):
        raise PublicExperienceBundleError("PUBLIC_EXPERIENCE_HORIZON_COUNT_MISMATCH")

    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        (temporary / "tables").mkdir(parents=True, mode=0o700)
        _write_csv(temporary / "tables" / "screen-trials.csv", SCREEN_FIELDS, screen_rows)
        _write_csv(temporary / "tables" / "horizon-metrics.csv", HORIZON_FIELDS, horizon_rows)
        _write_csv(temporary / "tables" / "best-by-environment.csv", BEST_FIELDS, best_rows)
        _write_json(temporary / "experience-atlas.json", atlas)
        _write_json(
            temporary / "screen-summary.json",
            {
                "schema_version": 1,
                "artifact_type": "verdiwm-public-screen-summary",
                "state": "ready",
                "campaign_count": screen_manifest.get("campaign_count"),
                "completed_row_count": len(screen_rows),
                "horizon_row_count": len(horizon_rows),
                "positive_screen_count": screen_manifest.get("positive_screen_count"),
                "negative_screen_count": screen_manifest.get("negative_screen_count"),
                "action_gate_fail_count": screen_manifest.get("action_gate_fail_count"),
                "best_by_environment_count": len(best_rows),
                "source_manifest_sha256": _sha256(screen_root / "manifest.json"),
                "claim_boundary": "Screen decisions are triage evidence, not official quality verdicts.",
                "limitations": screen_manifest.get("limitations", []),
            },
        )
        public_showcase = _copy_showcase(showcase, showcase_manifest, temporary / "showcase")
        _write_json(temporary / "showcase" / "manifest.json", public_showcase)
        _write_showcase_csv(temporary / "showcase" / "metrics.csv", public_showcase["records"])

        report = {
            "schema_version": 1,
            "artifact_type": "verdiwm-public-experience-bundle",
            "state": "ready",
            "screen_trial_count": len(screen_rows),
            "horizon_measurement_count": len(horizon_rows),
            "positive_screen_count": screen_manifest.get("positive_screen_count"),
            "negative_screen_count": screen_manifest.get("negative_screen_count"),
            "experience_source_map_count": atlas["source_map_count"],
            "experience_record_count": atlas["record_count"],
            "causal_edge_count": atlas["causal_edge_count"],
            "showcase_case_count": public_showcase["case_count"],
            "showcase_confirmed_case_count": public_showcase["confirmed_case_count"],
            "claim_boundary": (
                "The screen table preserves positive and negative triage evidence. Only the four showcase "
                "records carry frozen official-gate and confirmation admission; experience-atlas records "
                "remain context-local routing priors unless causal_credit_eligible is true."
            ),
        }
        _write_json(temporary / "bundle.json", report)
        (temporary / "README.md").write_text(_readme(report, public_showcase), encoding="utf-8")
        _write_manifest(temporary)
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return validate_public_experience_bundle(destination)


def validate_public_experience_bundle(root: Path) -> dict[str, object]:
    bundle_root = Path(root).resolve(strict=True)
    report = _load_json(bundle_root / "bundle.json")
    if report.get("artifact_type") != "verdiwm-public-experience-bundle" or report.get("state") != "ready":
        raise PublicExperienceBundleError("PUBLIC_EXPERIENCE_BUNDLE_INVALID")
    manifest = bundle_root / "MANIFEST.sha256"
    expected = _read_manifest(manifest)
    actual_files = {
        path.relative_to(bundle_root).as_posix()
        for path in bundle_root.rglob("*")
        if path.is_file() and path != manifest
    }
    if set(expected) != actual_files:
        raise PublicExperienceBundleError("PUBLIC_EXPERIENCE_FILE_SET_MISMATCH")
    for relative, digest in expected.items():
        if _sha256(bundle_root / relative) != digest:
            raise PublicExperienceBundleError(f"PUBLIC_EXPERIENCE_SHA256_MISMATCH:{relative}")
    for path in bundle_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".json", ".csv", ".md", ".sha256"}:
            if path.suffix.lower() == ".csv" and b"\r" in path.read_bytes():
                raise PublicExperienceBundleError(f"PUBLIC_EXPERIENCE_CSV_LINE_ENDING_INVALID:{path.name}")
            text = path.read_text(encoding="utf-8")
            if "/" + "mnt" + "/" in text or "/" + "root" + "/" in text:
                raise PublicExperienceBundleError(f"PUBLIC_EXPERIENCE_LOCAL_PATH:{path.name}")
    return dict(report)


def discover_experience_maps(report_root: Path) -> list[Path]:
    root = Path(report_root).resolve(strict=True)
    paths: list[Path] = []
    for pattern in ("acwm-horizon-experience-map-*", "acwm-salvage-*-horizon-experience-map-*"):
        for directory in sorted(root.glob(pattern)):
            path = directory / "horizon-experience-map.json"
            if path.is_file():
                paths.append(path)
    return sorted(set(paths))


def _experience_atlas(paths: Sequence[Path]) -> dict[str, object]:
    records: dict[str, dict[str, Any]] = {}
    source_rows = []
    for path in paths:
        payload = _load_json(path)
        if payload.get("artifact_type") != "wmloop-acwm-horizon-experience-map":
            raise PublicExperienceBundleError(f"PUBLIC_EXPERIENCE_MAP_INVALID:{path.parent.name}")
        source_id = path.parent.name
        source_digest = _sha256(path)
        source_rows.append({"id": source_id, "sha256": source_digest})
        for section, role in (
            ("observational_edges", "observational_edge"),
            ("routing_priors", "routing_prior"),
            ("anti_conditions", "anti_condition"),
            ("causal_edges", "causal_edge"),
        ):
            rows = payload.get(section, [])
            if not isinstance(rows, list):
                raise PublicExperienceBundleError(f"PUBLIC_EXPERIENCE_MAP_ROWS_INVALID:{source_id}:{section}")
            for row in rows:
                if not isinstance(row, Mapping):
                    raise PublicExperienceBundleError(f"PUBLIC_EXPERIENCE_MAP_ROW_INVALID:{source_id}:{section}")
                safe = _sanitize(row)
                canonical = json.dumps(safe, sort_keys=True, separators=(",", ":"))
                digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
                record = records.setdefault(digest, {"record_sha256": digest, "roles": [], "source_map_ids": [], **safe})
                if role not in record["roles"]:
                    record["roles"].append(role)
                if source_id not in record["source_map_ids"]:
                    record["source_map_ids"].append(source_id)
    ordered = sorted(records.values(), key=lambda row: (str(row.get("environment", "")), str(row.get("primitive", "")), row["record_sha256"]))
    causal = sum("causal_edge" in row["roles"] for row in ordered)
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-public-experience-atlas",
        "state": "ready",
        "source_map_count": len(source_rows),
        "record_count": len(ordered),
        "causal_edge_count": causal,
        "claim_boundary": "Records are context-local observational priors and anti-conditions unless causal_credit_eligible is true.",
        "sources": source_rows,
        "records": ordered,
    }


def _copy_showcase(source: Path, payload: Mapping[str, Any], destination: Path) -> dict[str, Any]:
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != 4:
        raise PublicExperienceBundleError("PUBLIC_EXPERIENCE_SHOWCASE_RECORDS_INVALID")
    public_records = []
    for record in records:
        if not isinstance(record, Mapping):
            raise PublicExperienceBundleError("PUBLIC_EXPERIENCE_SHOWCASE_RECORD_INVALID")
        safe = _sanitize(record)
        video = Path(str(record.get("video_path") or "")).resolve(strict=True)
        poster = Path(str(record.get("poster_path") or "")).resolve(strict=True)
        if source not in video.parents or source not in poster.parents:
            raise PublicExperienceBundleError("PUBLIC_EXPERIENCE_SHOWCASE_MEDIA_SCOPE")
        video_target = destination / "videos" / video.name
        poster_target = destination / "posters" / poster.name
        video_target.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        poster_target.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        shutil.copy2(video, video_target)
        shutil.copy2(poster, poster_target)
        safe["video_path"] = f"videos/{video.name}"
        safe["poster_path"] = f"posters/{poster.name}"
        public_records.append(safe)
    report = _sanitize(payload)
    report["records"] = public_records
    report["source_spec"] = "config://acwm_projectpage_four_cases_v4.json"
    return report


def _sanitize(value: Any, *, key: str = "") -> Any:
    if isinstance(value, Mapping):
        return {str(k): _sanitize(v, key=str(k)) for k, v in value.items() if str(k) != "candidate_checkpoint"}
    if isinstance(value, list):
        return [_sanitize(item, key=key) for item in value]
    if isinstance(value, str):
        if key in PATH_FIELDS or value.startswith("/"):
            return "private://redacted"
        return value.replace("/" + "mnt" + "/", "private://").replace("/" + "root" + "/", "private://")
    return value


def _read_selected_csv(path: Path, fields: Sequence[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or any(field not in reader.fieldnames for field in fields):
            raise PublicExperienceBundleError(f"PUBLIC_EXPERIENCE_CSV_FIELDS_MISSING:{path.name}")
        return [{field: str(row.get(field) or "") for field in fields} for row in reader]


def _write_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_showcase_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    fields = ("id", "environment", "primitive", "evidence_type", "psnr_delta", "ssim_delta", "mse_delta", "masked_mse_delta", "independent_confirmation_pass", "video_path", "poster_path")
    _write_csv(path, fields, ({field: row.get(field, "") for field in fields} for row in records))


def _readme(report: Mapping[str, Any], showcase: Mapping[str, Any]) -> str:
    return "\n".join([
        "# ACWM-Phys Public Experience Snapshot",
        "",
        "This snapshot retains successful, null, and harmful intervention evidence without publishing checkpoints, datasets, archive databases, or machine-local paths.",
        "",
        f"- Screen trials: `{report['screen_trial_count']}`",
        f"- Horizon measurements: `{report['horizon_measurement_count']}`",
        f"- Positive screen signals: `{report['positive_screen_count']}`",
        f"- Negative screen signals: `{report['negative_screen_count']}`",
        f"- Deduplicated experience records: `{report['experience_record_count']}`",
        f"- Official-gated showcase cases: `{showcase['case_count']}`",
        "",
        "## Claim boundary",
        "",
        str(report["claim_boundary"]),
        "A positive 512-step screen is not a validated method result. See `showcase/manifest.json` for the four records admitted by the frozen official pixel gate and an independent confirmation. The pour-water event statement is trajectory-local, as recorded in that manifest.",
        "",
        "## Files",
        "",
        "- `experience-atlas.json`: deduplicated routing priors, anti-conditions, and causal-credit status.",
        "- `screen-summary.json`: aggregate counts and source digest.",
        "- `tables/screen-trials.csv`: path-free screen ledger.",
        "- `tables/horizon-metrics.csv`: path-free per-horizon measurements.",
        "- `showcase/`: four GT/baseline/repair videos, posters, and gate summaries.",
        "- `MANIFEST.sha256`: integrity contract for every file in this snapshot.",
        "",
    ])


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PublicExperienceBundleError(f"PUBLIC_EXPERIENCE_JSON_INVALID:{path.name}") from exc
    if not isinstance(value, dict):
        raise PublicExperienceBundleError(f"PUBLIC_EXPERIENCE_JSON_INVALID:{path.name}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(root: Path) -> None:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.sha256":
            rows.append(f"{_sha256(path)}  {path.relative_to(root).as_posix()}")
    (root / "MANIFEST.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _read_manifest(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        if not separator or len(digest) != 64 or not relative:
            raise PublicExperienceBundleError("PUBLIC_EXPERIENCE_MANIFEST_INVALID")
        rows[relative] = digest
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("export", "validate"), default="export")
    parser.add_argument("--screen-summary-root", type=Path)
    parser.add_argument("--showcase-root", type=Path)
    parser.add_argument("--experience-report-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "validate":
        report = validate_public_experience_bundle(args.output_root)
    else:
        if args.screen_summary_root is None or args.showcase_root is None or args.experience_report_root is None:
            parser.error("export requires --screen-summary-root, --showcase-root, and --experience-report-root")
        report = export_public_experience_bundle(
            screen_summary_root=args.screen_summary_root,
            showcase_root=args.showcase_root,
            experience_map_paths=discover_experience_maps(args.experience_report_root),
            output_root=args.output_root,
        )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
