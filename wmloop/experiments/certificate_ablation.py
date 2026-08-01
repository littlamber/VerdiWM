"""Replay transfer-certificate on/off decisions from one selector evidence bundle."""

from __future__ import annotations

import argparse
import csv
import io
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from wmloop.experiments._artifacts import canonical_json, write_bundle


class CertificateAblationError(ValueError):
    """A selector bundle cannot identify certificate on/off behavior."""


def run_certificate_ablation(
    *, selector_bundle_root: Path, output_root: Path
) -> dict[str, object]:
    source = Path(selector_bundle_root).resolve(strict=True)
    replay = _json(source / "selector-replay.json")
    if (
        replay.get("artifact_type") != "verdiwm-acwm-selector-cpu-replay"
        or replay.get("transfer_certificate_enabled") is not True
    ):
        raise CertificateAblationError("CERTIFICATE_ABLATION_SOURCE_INVALID")
    cells = [row for row in replay.get("cells", []) if row.get("selector") == "irg"]
    if not cells:
        raise CertificateAblationError("CERTIFICATE_ABLATION_IRG_CELLS_MISSING")
    candidates = _csv_rows(source / "tables/candidates.csv")
    top1 = {
        str(row["trial_id"]): row
        for row in candidates
        if row.get("selector") == "irg" and row.get("rank") == "1"
    }
    counterfactual_rows: list[dict[str, object]] = []
    for cell in cells:
        trial_id = str(cell["trial_id"])
        if cell.get("state") == "evaluated":
            counterfactual_rows.append(
                {
                    "trial_id": trial_id,
                    "target_environment": cell["target_environment"],
                    "seed": cell["seed"],
                    "certificate_on_state": "licensed",
                    "certificate_off_state": "selected",
                    "primitive": cell["selected_primitive"],
                    "target_positive": bool(cell["selected_target_positive"]),
                    "changed_by_ablation": False,
                }
            )
            continue
        if cell.get("abstention_reason") == "transfer_certificate_failed":
            candidate = top1.get(trial_id)
            if candidate is None:
                raise CertificateAblationError(
                    f"CERTIFICATE_ABLATION_TOP1_MISSING:{trial_id}"
                )
            counterfactual_rows.append(
                {
                    "trial_id": trial_id,
                    "target_environment": cell["target_environment"],
                    "seed": cell["seed"],
                    "certificate_on_state": "abstained_certificate",
                    "certificate_off_state": "selected",
                    "primitive": candidate["primitive"],
                    "target_positive": _bool(candidate["target_positive"]),
                    "changed_by_ablation": True,
                }
            )
            continue
        counterfactual_rows.append(
            {
                "trial_id": trial_id,
                "target_environment": cell["target_environment"],
                "seed": cell["seed"],
                "certificate_on_state": "abstained_other",
                "certificate_off_state": "abstained_other",
                "primitive": None,
                "target_positive": None,
                "changed_by_ablation": False,
            }
        )

    total = len(counterfactual_rows)
    on_selected = [row for row in counterfactual_rows if row["certificate_on_state"] == "licensed"]
    off_selected = [row for row in counterfactual_rows if row["certificate_off_state"] == "selected"]
    prevented = [row for row in counterfactual_rows if row["certificate_on_state"] == "abstained_certificate"]
    rows = [
        _arm("certificate_on", selected=on_selected, total=total),
        _arm("certificate_off", selected=off_selected, total=total),
    ]
    prevented_negative = sum(row["target_positive"] is False for row in prevented)
    blocked_positive = sum(row["target_positive"] is True for row in prevented)
    report = {
        "schema_version": 1,
        "artifact_type": "verdiwm-transfer-certificate-ablation",
        "state": "ready",
        "source_artifact_type": replay["artifact_type"],
        "source_scope": "acwm_phys_leave_one_environment_out_cpu_replay",
        "planned_cell_count": total,
        "arms": rows,
        "certificate_changed_cell_count": len(prevented),
        "certificate_prevented_negative_count": prevented_negative,
        "certificate_blocked_positive_count": blocked_positive,
        "prevented_negative_fraction": (
            prevented_negative / len(prevented) if prevented else None
        ),
        "counterfactual_rows": counterfactual_rows,
        "gpu_hours": 0.0,
        "claim_boundary": (
            "This is an exactly replayed certificate decision ablation on settled ACWM-Phys "
            "target labels. It establishes coverage-risk behavior for this evidence bundle only. "
            "It is not cross-backbone LOBO, new model inference, or a transfer-quality claim."
        ),
    }
    destination = Path(output_root).resolve()
    return write_bundle(
        output_root=destination,
        files={
            "certificate-ablation.json": canonical_json(report),
            "certificate-ablation.md": _markdown(report).encode("utf-8"),
            "tables/certificate-ablation.csv": _csv(rows).encode("utf-8"),
            "tables/counterfactual-cells.csv": _csv(counterfactual_rows).encode("utf-8"),
        },
        manifest_fields={
            "artifact_type": "verdiwm-transfer-certificate-ablation-manifest",
            "state": "ready",
            "planned_cell_count": total,
            "certificate_changed_cell_count": len(prevented),
            "certificate_prevented_negative_count": prevented_negative,
            "certificate_blocked_positive_count": blocked_positive,
            "gpu_hours": 0.0,
            "report_path": str(destination / "certificate-ablation.json"),
        },
    )


def _arm(name: str, *, selected: Sequence[Mapping[str, Any]], total: int) -> dict[str, object]:
    negatives = sum(row["target_positive"] is False for row in selected)
    positives = sum(row["target_positive"] is True for row in selected)
    return {
        "arm": name,
        "selected_cell_count": len(selected),
        "abstained_cell_count": total - len(selected),
        "coverage": len(selected) / total,
        "positive_selection_count": positives,
        "negative_selection_count": negatives,
        "negative_transfer_rate": negatives / len(selected) if selected else None,
    }


def _json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise CertificateAblationError(f"CERTIFICATE_ABLATION_JSON_INVALID:{path}")
    return payload


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise CertificateAblationError("CERTIFICATE_ABLATION_BOOLEAN_INVALID")


def _csv(rows: Sequence[Mapping[str, object]]) -> str:
    if not rows:
        return ""
    output = io.StringIO(newline="")
    fields = sorted({str(key) for row in rows for key in row})
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Transfer Certificate On/Off Ablation",
        "",
        "| Arm | Coverage | Negative transfer rate | Positive | Negative | Abstained |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["arms"]:
        risk = row["negative_transfer_rate"]
        lines.append(
            f"| {row['arm']} | {row['coverage']:.3f} | "
            f"{risk if risk is not None else 'n/a'} | {row['positive_selection_count']} | "
            f"{row['negative_selection_count']} | {row['abstained_cell_count']} |"
        )
    lines.extend(["", "## Claim Boundary", "", str(report["claim_boundary"]), ""])
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector-bundle-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = run_certificate_ablation(
        selector_bundle_root=args.selector_bundle_root,
        output_root=args.output_root,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
