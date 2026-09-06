"""Plan counterexample-driven probe evolution from conflicting model IRGs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from wmloop.experiments._artifacts import (
    ExperimentArtifactError,
    canonical_json,
    write_bundle,
)
from wmloop.geometry.model_irg import (
    ModelIRGError,
    detect_model_irg_collisions,
    validate_model_irg,
)
from wmloop.geometry.types import GeometryValidationError


class ProbeEvolutionCycleError(RuntimeError):
    """IRG collision evidence cannot produce a safe probe-evolution proposal."""


def plan_probe_evolution_cycle(
    *,
    model_irg_paths: Sequence[Path],
    output_root: Path,
    distance_threshold: float = 0.25,
    minimum_effect: float = 0.0,
    fdr_alpha: float = 0.05,
    repo_root: Path | None = None,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    """Detect nearby-IRG/opposite-effect collisions and emit generic work orders."""

    root = Path(repo_root or Path(__file__).resolve().parents[2]).resolve()
    paths = tuple(dict.fromkeys(_file(path) for path in model_irg_paths))
    if len(paths) < 2:
        raise ProbeEvolutionCycleError("PROBE_EVOLUTION_AT_LEAST_TWO_IRGS_REQUIRED")
    if (
        not math.isfinite(distance_threshold)
        or distance_threshold < 0
        or not math.isfinite(minimum_effect)
        or minimum_effect < 0
        or not math.isfinite(fdr_alpha)
        or not 0 < fdr_alpha < 1
    ):
        raise ProbeEvolutionCycleError("PROBE_EVOLUTION_THRESHOLD_INVALID")
    bindings = tuple(_load_irg(path, root=root) for path in paths)
    input_lock: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-probe-evolution-cycle-input-lock",
        "state": "frozen",
        "model_irgs": [
            {
                "path": str(path),
                "sha256": _sha256_file(path),
                "irg_id": binding["irg_id"],
            }
            for path, binding in zip(paths, bindings, strict=True)
        ],
        "distance_threshold": distance_threshold,
        "minimum_effect": minimum_effect,
        "fdr_alpha": fdr_alpha,
        "claim_boundary": (
            "This lock freezes collision-planning inputs only; it grants no probe "
            "execution or admission authority."
        ),
    }
    input_hash = _digest(input_lock)
    input_lock["input_hash"] = input_hash
    destination = Path(output_root).expanduser().resolve()
    resumed = _resume(destination, input_hash=input_hash)
    if resumed is not None:
        return resumed
    try:
        collisions = detect_model_irg_collisions(
            bindings,
            distance_threshold=distance_threshold,
            minimum_effect=minimum_effect,
            fdr_alpha=fdr_alpha,
        )
    except (ModelIRGError, GeometryValidationError) as exc:
        raise ProbeEvolutionCycleError(f"PROBE_EVOLUTION_COLLISION_INVALID:{exc}") from exc
    by_id = {str(binding["irg_id"]): binding for binding in bindings}
    rows = [
        _collision_document(collision, by_id=by_id)
        for collision in collisions
    ]
    work_orders = [
        _work_order(index=index, collision=row)
        for index, row in enumerate(rows, start=1)
    ]
    state = "proposal_ready" if work_orders else "no_collision"
    report = {
        "schema_version": 1,
        "artifact_type": "verdiwm-probe-evolution-cycle",
        "state": state,
        "input_hash": input_hash,
        "irg_count": len(bindings),
        "collision_count": len(rows),
        "collisions": rows,
        "work_orders": work_orders,
        "authority": "proposal_only",
        "admission_requirements": [
            "materialize a new diagnostic probe without changing frozen verdict metrics",
            "evaluate it on both counterexample contexts with identical seeds and evaluator",
            "admit it only after independent locality, calibration, and regression settlement",
            "publish a new IRG version only from the admitted measured response asset",
        ],
        "claim_boundary": (
            "A collision shows that the current IRG is insufficient to predict a method effect. "
            "These work orders propose discriminating measurements only; they do not invent "
            "responses, modify the model, or admit a successor probe."
        ),
    }
    try:
        return write_bundle(
            output_root=destination,
            files={
                "input-lock.json": canonical_json(input_lock),
                "probe-evolution-cycle.json": canonical_json(report),
            },
            manifest_fields={
                "artifact_type": "verdiwm-probe-evolution-cycle-manifest",
                "state": state,
                "input_hash": input_hash,
                "irg_count": len(bindings),
                "collision_count": len(rows),
                "work_order_count": len(work_orders),
                "report_path": str(destination / "probe-evolution-cycle.json"),
                "authority": "proposal_only",
                "claim_boundary": report["claim_boundary"],
            },
            archive_db=archive_db,
            cas_root=cas_root,
        )
    except ExperimentArtifactError as exc:
        raise ProbeEvolutionCycleError(
            f"PROBE_EVOLUTION_OUTPUT_INVALID:{exc}"
        ) from exc


def _collision_document(
    collision: object, *, by_id: Mapping[str, Mapping[str, object]]
) -> dict[str, object]:
    left_id = str(getattr(collision, "left_campaign_id"))
    right_id = str(getattr(collision, "right_campaign_id"))
    left = by_id[left_id]
    right = by_id[right_id]
    if left["coordinate_names"] != right["coordinate_names"]:
        raise ProbeEvolutionCycleError("PROBE_EVOLUTION_COORDINATES_INCOMPATIBLE")
    coordinate_gaps = sorted(
        (
            {
                "axis": str(name),
                "absolute_gap": abs(float(a) - float(b)),
                "left_response": float(a),
                "right_response": float(b),
            }
            for name, a, b in zip(
                left["coordinate_names"],
                left["response_vector"],
                right["response_vector"],
                strict=True,
            )
        ),
        key=lambda row: (float(row["absolute_gap"]), str(row["axis"])),
    )
    hooks = sorted(set(left["available_hooks"]) & set(right["available_hooks"]))
    return {
        "left_irg_id": left_id,
        "right_irg_id": right_id,
        "primitive": str(getattr(collision, "primitive")),
        "irg_distance": float(getattr(collision, "distance")),
        "left_effect": float(getattr(collision, "left_effect")),
        "right_effect": float(getattr(collision, "right_effect")),
        "q_value": float(getattr(collision, "q_value")),
        "least_discriminating_axes": coordinate_gaps[: min(5, len(coordinate_gaps))],
        "shared_available_hooks": hooks,
    }


def _work_order(*, index: int, collision: Mapping[str, object]) -> dict[str, object]:
    payload = {
        "collision": dict(collision),
        "objective": (
            "Design a reversible measurement whose response separates the two IRG contexts "
            f"before predicting the effect of {collision['primitive']}."
        ),
    }
    return {
        "work_order_id": f"probe-evolution-{index:03d}-{_digest(payload)[:16]}",
        **payload,
        "candidate_generation_contract": {
            "target_axes": collision["least_discriminating_axes"],
            "eligible_hooks": collision["shared_available_hooks"],
            "required_properties": [
                "inference_only_or_read_only",
                "reversible",
                "paired_counterexample_measurement",
                "bounded_cost",
                "no_verdict_metric_change",
            ],
        },
        "execution_authority": "none_until_materialized_and_settled",
    }


def _load_irg(path: Path, *, root: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProbeEvolutionCycleError("PROBE_EVOLUTION_IRG_INVALID") from exc
    if not isinstance(payload, dict):
        raise ProbeEvolutionCycleError("PROBE_EVOLUTION_IRG_INVALID")
    try:
        validate_model_irg(payload, root=root)
    except ModelIRGError as exc:
        raise ProbeEvolutionCycleError(f"PROBE_EVOLUTION_IRG_INVALID:{exc}") from exc
    return payload


def _resume(destination: Path, *, input_hash: str) -> dict[str, object] | None:
    if not destination.exists() and not destination.is_symlink():
        return None
    if destination.is_symlink() or not destination.is_dir():
        raise ProbeEvolutionCycleError("PROBE_EVOLUTION_OUTPUT_INVALID")
    lock = _mapping(destination / "input-lock.json", "PROBE_EVOLUTION_INPUT_LOCK_INVALID")
    if lock.get("input_hash") != input_hash:
        raise ProbeEvolutionCycleError("PROBE_EVOLUTION_INPUT_LOCK_MISMATCH")
    manifest = _mapping(destination / "manifest.json", "PROBE_EVOLUTION_MANIFEST_INVALID")
    if manifest.get("input_hash") != input_hash:
        raise ProbeEvolutionCycleError("PROBE_EVOLUTION_MANIFEST_INVALID")
    return manifest


def _mapping(path: Path, code: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProbeEvolutionCycleError(code) from exc
    if not isinstance(payload, dict):
        raise ProbeEvolutionCycleError(code)
    return payload


def _file(path: Path) -> Path:
    value = Path(path).expanduser().resolve()
    if value.is_symlink() or not value.is_file():
        raise ProbeEvolutionCycleError("PROBE_EVOLUTION_IRG_INVALID")
    return value


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(payload: object) -> str:
    return hashlib.sha256(canonical_json(payload).rstrip(b"\n")).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-irg", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--distance-threshold", type=float, default=0.25)
    parser.add_argument("--minimum-effect", type=float, default=0.0)
    parser.add_argument("--fdr-alpha", type=float, default=0.05)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--archive-db", type=Path)
    parser.add_argument("--cas-root", type=Path)
    args = parser.parse_args(argv)
    try:
        manifest = plan_probe_evolution_cycle(
            model_irg_paths=args.model_irg,
            output_root=args.output_root,
            distance_threshold=args.distance_threshold,
            minimum_effect=args.minimum_effect,
            fdr_alpha=args.fdr_alpha,
            repo_root=args.repo_root,
            archive_db=args.archive_db,
            cas_root=args.cas_root,
        )
    except ProbeEvolutionCycleError as exc:
        print(str(exc), file=__import__("sys").stderr)
        return 2
    print(json.dumps(manifest, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
