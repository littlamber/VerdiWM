#!/usr/bin/env python3
"""Export a path-sanitized Cosmos3 directional dev/accept settlement bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


class Cosmos3DirectionalSettlementPublicBundleError(ValueError):
    """Directional settlement evidence cannot be published safely."""


def export_cosmos3_directional_settlement_public_bundle(
    *,
    dev_selection_root: Path | None = None,
    dev_fingerprint_root: Path | None = None,
    accept_public_root: Path,
    settlement_root: Path,
    output_root: Path,
) -> dict[str, object]:
    if (dev_selection_root is None) == (dev_fingerprint_root is None):
        raise Cosmos3DirectionalSettlementPublicBundleError(
            "COSMOS3_DIRECTIONAL_PUBLIC_DEV_SOURCE_INVALID"
        )
    direct_fingerprint = dev_fingerprint_root is not None
    dev_root = Path(dev_fingerprint_root or dev_selection_root).resolve(strict=True)
    accept_root = Path(accept_public_root).resolve(strict=True)
    settled_root = Path(settlement_root).resolve(strict=True)
    if direct_fingerprint:
        dev_report = {"state": "dev_selected"}
        dev_fingerprint_name = "target-local-fingerprint.json"
    else:
        dev_report = _load_mapping(dev_root / "directional-probe-evolution.json")
        dev_fingerprint_name = "selected-dev-fingerprint.json"
    dev_fingerprint = _load_mapping(dev_root / dev_fingerprint_name)
    accept_bundle = _load_mapping(accept_root / "bundle.json")
    accept_fingerprint = _load_mapping(accept_root / "target-local-fingerprint.json")
    settlement = _load_mapping(settled_root / "directional-probe-settlement.json")
    campaign_id = str(settlement.get("campaign_id", ""))
    if (
        dev_report.get("state") != "dev_selected"
        or dev_fingerprint.get("campaign_id") != campaign_id
        or dev_fingerprint.get("split") != "dev"
        or accept_bundle.get("campaign_id") != campaign_id
        or accept_bundle.get("split") != "accept"
        or accept_fingerprint.get("campaign_id") != campaign_id
        or settlement.get("artifact_type")
        != "verdiwm-cosmos3-directional-probe-settlement"
        or settlement.get("state") not in {"settled_licensed", "settled_abstained"}
    ):
        raise Cosmos3DirectionalSettlementPublicBundleError(
            "COSMOS3_DIRECTIONAL_PUBLIC_INPUT_INVALID"
        )

    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise Cosmos3DirectionalSettlementPublicBundleError(
            "COSMOS3_DIRECTIONAL_PUBLIC_OUTPUT_EXISTS"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        _copy_public_tree(accept_root, temporary / "accept")
        if direct_fingerprint:
            _copy_json(
                dev_root / "target-local-fingerprint.json",
                temporary / "dev/target-local-fingerprint.json",
            )
        else:
            _copy_json(
                dev_root / "directional-probe-evolution.json",
                temporary / "dev/directional-probe-evolution.json",
            )
            _copy_json(
                dev_root / "selected-dev-fingerprint.json",
                temporary / "dev/selected-dev-fingerprint.json",
            )
        _copy_json(
            settled_root / "directional-probe-settlement.json",
            temporary / "settlement/directional-probe-settlement.json",
        )
        _write_alignment_table(temporary / "tables/dev-accept-alignment.csv", settlement)
        _write_alignment_svg(temporary / "figures/dev-accept-alignment.svg", settlement)
        bundle = {
            "schema_version": 1,
            "artifact_type": "verdiwm-cosmos3-directional-settlement-public-bundle",
            "state": "ready",
            "campaign_id": campaign_id,
            "dev_source_type": (
                "direct_frozen_fingerprint" if direct_fingerprint else "dev_selected_subset"
            ),
            "probe_id": settlement["probe_id"],
            "settlement_state": settlement["state"],
            "dev_locality_residual": settlement["locality"]["dev_residual"],
            "accept_locality_residual": settlement["locality"]["accept_residual"],
            "locality_threshold": settlement["locality"]["maximum_residual"],
            "dev_accept_alignment_error": settlement["alignment"]["error"],
            "maximum_alignment_error": settlement["alignment"]["maximum_error"],
            "accept_target_local_path_admitted": accept_bundle["locality_admission_state"]
            == "passed",
            "cross_backbone_transfer_eligible": settlement[
                "cross_backbone_transfer_eligible"
            ],
            "certificate_terms": settlement["terms"],
            "abstention_reasons": settlement["abstention_reasons"],
            "accept_videos": accept_bundle["videos"],
            "claim_boundary": settlement["claim_boundary"],
        }
        (temporary / "bundle.json").write_text(_pretty_json(bundle), encoding="utf-8")
        (temporary / "README.md").write_text(_readme(bundle), encoding="utf-8")
        _assert_public_text(temporary)
        _write_manifest(temporary)
        os.replace(temporary, destination)
        return bundle
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _write_alignment_table(path: Path, settlement: Mapping[str, Any]) -> None:
    dev = settlement["alignment"]["dev_jacobian"]
    accept = settlement["alignment"]["accept_jacobian"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("coordinate", "dev_jacobian", "accept_jacobian"),
            lineterminator="\n",
        )
        writer.writeheader()
        for index, (left, right) in enumerate(zip(dev, accept, strict=True)):
            writer.writerow(
                {
                    "coordinate": index,
                    "dev_jacobian": float(left),
                    "accept_jacobian": float(right),
                }
            )


def _write_alignment_svg(path: Path, settlement: Mapping[str, Any]) -> None:
    dev = [float(value) for value in settlement["alignment"]["dev_jacobian"]]
    accept = [float(value) for value in settlement["alignment"]["accept_jacobian"]]
    scale = 250.0 / max(abs(value) for value in (*dev, *accept))
    rows = []
    for index, (left, right) in enumerate(zip(dev, accept, strict=True)):
        y = 145 + index * 82
        rows.extend(
            [
                f'<text x="35" y="{y + 5}" font-size="14">outcome {index}</text>',
                _bar(430, y - 20, left * scale, "#1976D2"),
                _bar(430, y + 12, right * scale, "#C62828"),
            ]
        )
    height = 190 + len(dev) * 82
    alignment_error = float(settlement["alignment"]["error"])
    maximum_error = float(settlement["alignment"]["maximum_error"])
    svg = "\n".join(
        [
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="{height}" '
                f'viewBox="0 0 900 {height}">'
            ),
            '<rect width="100%" height="100%" fill="white"/>',
            (
                '<text x="35" y="42" font-size="24" font-weight="600">'
                "Directional probe split stability</text>"
            ),
            (
                '<text x="35" y="72" font-size="14" fill="#444">'
                f"alignment error {alignment_error:.4f}; threshold {maximum_error:.4f}</text>"
            ),
            (
                '<rect x="35" y="92" width="16" height="10" fill="#1976D2"/>'
                '<text x="58" y="102" font-size="13">dev</text>'
            ),
            (
                '<rect x="115" y="92" width="16" height="10" fill="#C62828"/>'
                '<text x="138" y="102" font-size="13">accept</text>'
            ),
            f'<line x1="430" y1="112" x2="430" y2="{height - 30}" stroke="#777"/>',
            *rows,
            '</svg>',
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg + "\n", encoding="utf-8")


def _bar(origin: float, y: float, width: float, color: str) -> str:
    x = min(origin, origin + width)
    return f'<rect x="{x:.2f}" y="{y:.2f}" width="{abs(width):.2f}" height="18" fill="{color}"/>'


def _copy_public_tree(source: Path, destination: Path) -> None:
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise Cosmos3DirectionalSettlementPublicBundleError(
                "COSMOS3_DIRECTIONAL_PUBLIC_SYMLINK"
            )
        if path.is_file():
            target = destination / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def _copy_json(source: Path, destination: Path) -> None:
    payload = _load_mapping(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(_pretty_json(payload), encoding="utf-8")


def _readme(bundle: Mapping[str, Any]) -> str:
    return (
        "# Cosmos3 Directional Probe Held-Out Settlement\n\n"
        "This bundle compares the same frozen directional probe on dev and accept. "
        f"The locality residuals are `{float(bundle['dev_locality_residual']):.4f}` and "
        f"`{float(bundle['accept_locality_residual']):.4f}`. The normalized Jacobian "
        f"alignment error is `{float(bundle['dev_accept_alignment_error']):.4f}` against the "
        f"frozen `{float(bundle['maximum_alignment_error']):.4f}` threshold. The final "
        f"settlement is therefore `{bundle['settlement_state']}`.\n\n"
        f"Failed certificate terms: `{', '.join(bundle['abstention_reasons'])}`. "
        "Each name denotes a required term whose value is `false`. This is a certificate "
        "counterexample, not model-improvement evidence. The `accept/` "
        "subdirectory contains the paired dose-response tables and videos.\n"
    )


def _assert_public_text(root: Path) -> None:
    prefixes = ("/" + "mnt" + "/", "/" + "root" + "/")
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".jsonl", ".csv", ".md", ".svg"}:
            text = path.read_text(encoding="utf-8")
            if any(prefix in text for prefix in prefixes):
                raise Cosmos3DirectionalSettlementPublicBundleError(
                    "COSMOS3_DIRECTIONAL_PUBLIC_LOCAL_PATH_LEAK"
                )


def _write_manifest(root: Path) -> None:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.sha256":
            rows.append(f"{_sha256(path)}  {path.relative_to(root).as_posix()}")
    (root / "MANIFEST.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _load_mapping(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise Cosmos3DirectionalSettlementPublicBundleError(
            f"COSMOS3_DIRECTIONAL_PUBLIC_JSON_INVALID:{path.name}"
        )
    return payload


def _pretty_json(payload: object) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    dev = parser.add_mutually_exclusive_group(required=True)
    dev.add_argument("--dev-selection-root", type=Path)
    dev.add_argument("--dev-fingerprint-root", type=Path)
    parser.add_argument("--accept-public-root", type=Path, required=True)
    parser.add_argument("--settlement-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    bundle = export_cosmos3_directional_settlement_public_bundle(
        dev_selection_root=args.dev_selection_root,
        dev_fingerprint_root=args.dev_fingerprint_root,
        accept_public_root=args.accept_public_root,
        settlement_root=args.settlement_root,
        output_root=args.output_root,
    )
    print(json.dumps(bundle, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
