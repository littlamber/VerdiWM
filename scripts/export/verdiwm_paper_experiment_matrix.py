#!/usr/bin/env python3
"""Export the frozen VerdiWM paper evidence matrix to reviewable tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections.abc import Sequence
from pathlib import Path


REQUIRED_STUDY_KEYS = {
    "study_id",
    "priority",
    "scope",
    "question",
    "conditions",
    "primary_metrics",
    "evidence_level",
    "cross_backbone_required",
    "paper_role",
}


def load_matrix(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("artifact_type") != "verdiwm-paper-evidence-matrix":
        raise ValueError("EVIDENCE_MATRIX_ARTIFACT_TYPE_INVALID")
    studies = payload.get("studies")
    if not isinstance(studies, list) or not studies:
        raise ValueError("EVIDENCE_MATRIX_STUDIES_EMPTY")
    study_ids: list[str] = []
    for study in studies:
        if not isinstance(study, dict) or not REQUIRED_STUDY_KEYS.issubset(study):
            raise ValueError("EVIDENCE_MATRIX_STUDY_CONTRACT_INVALID")
        study_ids.append(str(study["study_id"]))
    if len(study_ids) != len(set(study_ids)):
        raise ValueError("EVIDENCE_MATRIX_STUDY_ID_DUPLICATE")
    execution_order = payload.get("execution_order")
    if execution_order != study_ids and set(execution_order or []) != set(study_ids):
        raise ValueError("EVIDENCE_MATRIX_EXECUTION_ORDER_INVALID")
    return payload


def export_matrix(*, config_path: Path, output_root: Path) -> dict[str, object]:
    matrix = load_matrix(config_path)
    output_root.mkdir(parents=True, exist_ok=True)
    rows = [_row(study) for study in matrix["studies"]]  # type: ignore[index]
    files = {
        "evidence-matrix.md": _markdown(matrix),
        "tables/evidence-matrix.csv": _csv(rows),
        "tables/evidence-matrix.tex": _latex(rows),
        "input-config.json": json.dumps(matrix, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
    }
    digests: dict[str, str] = {}
    for relative, content in files.items():
        destination = output_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
        digests[relative] = hashlib.sha256(content.encode("utf-8")).hexdigest()
    manifest = {
        "schema_version": 1,
        "artifact_type": "verdiwm-paper-evidence-matrix-manifest",
        "matrix_id": matrix["matrix_id"],
        "study_count": len(rows),
        "must_have_count": sum(row["priority"] == "must_have" for row in rows),
        "cross_backbone_study_count": sum(row["cross_backbone"] != "false" for row in rows),
        "files": digests,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _row(study: dict[str, object]) -> dict[str, str]:
    return {
        "study_id": str(study["study_id"]),
        "priority": str(study["priority"]),
        "scope": str(study["scope"]),
        "conditions": "; ".join(str(value) for value in study["conditions"]),  # type: ignore[arg-type]
        "primary_metrics": "; ".join(str(value) for value in study["primary_metrics"]),  # type: ignore[arg-type]
        "cross_backbone": str(study["cross_backbone_required"]).lower(),
        "paper_role": str(study["paper_role"]),
    }


def _markdown(matrix: dict[str, object]) -> str:
    policy = matrix["external_backbone_policy"]  # type: ignore[index]
    lines = [
        "# VerdiWM ICLR Evidence Matrix",
        "",
        f"Matrix: `{matrix['matrix_id']}`",
        "",
        "## Claim Boundary",
        "",
        str(matrix["claim_boundary"]),
        "",
        "ACWM-Phys-only ablations validate the reference-instance mechanism. A cross-backbone claim requires held-out target evidence on at least "
        f"`{policy['minimum_external_targets']}` external backbone families: `{', '.join(policy['required_targets'])}`.",
        "",
        "## Studies",
        "",
        "| Study | Priority | Scope | Cross-backbone | Paper role |",
        "|---|---|---|---|---|",
    ]
    for study in matrix["studies"]:  # type: ignore[index]
        lines.append(
            "| {study_id} | {priority} | {scope} | {cross} | {paper_role} |".format(
                study_id=study["study_id"],
                priority=study["priority"],
                scope=study["scope"],
                cross=str(study["cross_backbone_required"]).lower(),
                paper_role=study["paper_role"],
            )
        )
    lines.extend(["", "## Execution Order", ""])
    for index, study_id in enumerate(matrix["execution_order"], start=1):  # type: ignore[index]
        lines.append(f"{index}. `{study_id}`")
    lines.extend(["", "## Launch Gate", ""])
    for blocker in matrix["launch_policy"]["do_not_launch_full_lobo_until"]:  # type: ignore[index]
        lines.append(f"- {blocker}")
    return "\n".join(lines) + "\n"


def _csv(rows: list[dict[str, str]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _latex(rows: list[dict[str, str]]) -> str:
    lines = [
        "\\begin{tabular}{llllp{5.2cm}}",
        "\\toprule",
        "Study & Priority & Scope & Cross-backbone & Paper role \\\\",
        "\\midrule",
    ]
    for row in rows:
        values = [row["study_id"], row["priority"], row["scope"], row["cross_backbone"], row["paper_role"]]
        lines.append(" & ".join(_latex_escape(value) for value in values) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


def _latex_escape(value: str) -> str:
    return value.replace("\\", "\\textbackslash{}") .replace("_", "\\_").replace("%", "\\%")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = export_matrix(config_path=args.config, output_root=args.output_root)
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
