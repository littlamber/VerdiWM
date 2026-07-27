"""Surrogate readiness and ranking guard for settled archive observations."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from wmloop.archive.embeddings import (
    cosine_similarity,
    embed_intervention,
    encoder_manifest,
    intervention_description,
)
from wmloop.archive.store import ArchiveStore, CellProjectionRecord, ContentAddressedStore
from wmloop.propose.scheduler import InterventionCell


SURROGATE_MIN_SETTLED_TRIALS = 100


class SurrogateError(RuntimeError):
    """The archive surrogate gate failed closed."""


@dataclass(frozen=True)
class RankingCandidate:
    proposal_id: str
    cell: InterventionCell


def run_surrogate_readiness(
    *,
    archive_db: Path,
    output_root: Path,
    cas_root: Path | None = None,
    candidate_cells: Path | None = None,
    min_settled_trials: int = SURROGATE_MIN_SETTLED_TRIALS,
) -> dict[str, object]:
    """Write a fail-closed surrogate readiness/ranking report.

    The surrogate is only allowed to sort candidates after enough settled
    single-factor observations exist.  Its predicted values are never exposed as
    reward, verified deltas or verdict inputs.
    """

    if min_settled_trials < 1:
        raise SurrogateError("SURROGATE_MIN_SETTLED_INVALID")
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise SurrogateError("SURROGATE_OUTPUT_EXISTS")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    archive = ArchiveStore(archive_db)
    cas = ContentAddressedStore(cas_root if cas_root is not None else Path(archive_db).resolve().parent)
    try:
        temporary.mkdir(mode=0o700)
        stats = archive.archive_statistics()
        cells = archive.list_cells()
        candidates = _load_candidates(candidate_cells) if candidate_cells is not None else []
        cell_observation_count = sum(record.stats.visits for record in cells)
        blockers = _readiness_blockers(
            settled_trial_count=stats["settled_trials"],
            cell_observation_count=cell_observation_count,
            cell_count=len(cells),
            min_settled_trials=min_settled_trials,
        )
        state = "ready" if not blockers else "blocked"
        rankings = _rank_candidates(candidates, cells) if state == "ready" else []
        report = {
            "schema_version": 1,
            "artifact_type": "wmloop-surrogate-readiness-report",
            "state": state,
            "archive_db": str(Path(archive_db).resolve()),
            "min_settled_trials": min_settled_trials,
            "settled_trial_count": stats["settled_trials"],
            "cell_count": len(cells),
            "cell_observation_count": cell_observation_count,
            "surrogate_training_allowed": state == "ready",
            "surrogate_model_type": "weighted_cell_knn_regressor" if state == "ready" else None,
            "encoder": encoder_manifest().to_dict(),
            "prediction_usage": "proposal_sorting_only",
            "prediction_values_enter_reward_or_verdict": False,
            "forbidden_prediction_uses": [
                "reward",
                "verdict",
                "delta_m_ver",
                "acceptance_gate",
                "model_quality_claim",
            ],
            "blockers": blockers,
            "training_cells": [_cell_record_to_dict(record) for record in cells],
            "candidate_count": len(candidates),
            "ranking_count": len(rankings),
            "rankings": rankings,
            "limitations": [
                "This surrogate is a proposal sorting aid only; verifier verdicts remain independent.",
                "Training requires enough settled, non-exploratory single-factor observations in archive cells.",
            ],
        }
        report_bytes = _canonical_json_bytes(report)
        markdown_bytes = _render_markdown(report).encode("utf-8")
        _write_bytes_atomic(temporary / "surrogate-readiness.json", report_bytes)
        _write_bytes_atomic(temporary / "surrogate-readiness.md", markdown_bytes)
        report_ref = cas.put_bytes(report_bytes, media_type="application/json").uri
        markdown_ref = cas.put_bytes(markdown_bytes, media_type="text/markdown").uri
        archive.record_artifact_reference(report_ref)
        archive.record_artifact_reference(markdown_ref)
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-surrogate-readiness-manifest",
            "state": state,
            "surrogate_training_allowed": state == "ready",
            "min_settled_trials": min_settled_trials,
            "settled_trial_count": stats["settled_trials"],
            "cell_observation_count": cell_observation_count,
            "candidate_count": len(candidates),
            "ranking_count": len(rankings),
            "prediction_usage": report["prediction_usage"],
            "prediction_values_enter_reward_or_verdict": False,
            "blockers": blockers,
            "report_path": str(destination / "surrogate-readiness.json"),
            "markdown_path": str(destination / "surrogate-readiness.md"),
            "cas_refs": {
                "surrogate_readiness_json": report_ref,
                "surrogate_readiness_markdown": markdown_ref,
            },
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


def _readiness_blockers(
    *,
    settled_trial_count: int,
    cell_observation_count: int,
    cell_count: int,
    min_settled_trials: int,
) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    if settled_trial_count < min_settled_trials:
        blockers.append(
            {
                "code": "SURROGATE_MIN_SETTLED_TRIALS_NOT_MET",
                "expected": min_settled_trials,
                "observed": settled_trial_count,
            }
        )
    if cell_count == 0:
        blockers.append({"code": "SURROGATE_CELL_TRAINING_DATA_EMPTY", "observed": cell_count})
    if cell_observation_count < min_settled_trials:
        blockers.append(
            {
                "code": "SURROGATE_MIN_CELL_OBSERVATIONS_NOT_MET",
                "expected": min_settled_trials,
                "observed": cell_observation_count,
            }
        )
    return blockers


def _rank_candidates(
    candidates: Sequence[RankingCandidate],
    training_cells: Sequence[CellProjectionRecord],
) -> list[dict[str, object]]:
    predictions = [_predict_candidate(candidate, training_cells) for candidate in candidates]
    return sorted(predictions, key=lambda item: (-float(item["predicted_verified_gain"]), str(item["proposal_id"])))


def _predict_candidate(
    candidate: RankingCandidate,
    training_cells: Sequence[CellProjectionRecord],
) -> dict[str, object]:
    target = embed_intervention(candidate.cell)
    weighted_sum = 0.0
    total_weight = 0.0
    nearest: tuple[float, CellProjectionRecord] | None = None
    for record in training_cells:
        similarity = cosine_similarity(target, embed_intervention(record.cell))
        distance = 1.0 - similarity
        if nearest is None or distance < nearest[0]:
            nearest = (distance, record)
        if distance <= 1e-12:
            return _prediction(
                candidate,
                predicted=record.stats.mean_verified_improvement,
                nearest_distance=0.0,
                nearest_record=record,
            )
        weight = max(similarity, 0.0) * record.stats.visits
        weighted_sum += weight * record.stats.mean_verified_improvement
        total_weight += weight
    if nearest is None:
        raise SurrogateError("SURROGATE_TRAINING_CELLS_EMPTY")
    predicted = weighted_sum / total_weight if total_weight > 0.0 else 0.0
    if not math.isfinite(predicted):
        raise SurrogateError("SURROGATE_PREDICTION_INVALID")
    return _prediction(candidate, predicted=predicted, nearest_distance=nearest[0], nearest_record=nearest[1])


def _prediction(
    candidate: RankingCandidate,
    *,
    predicted: float,
    nearest_distance: float,
    nearest_record: CellProjectionRecord,
) -> dict[str, object]:
    return {
        "proposal_id": candidate.proposal_id,
        "cell": _cell_to_dict(candidate.cell),
        "intervention_description": intervention_description(candidate.cell),
        "predicted_verified_gain": float(predicted),
        "nearest_cosine_distance": float(nearest_distance),
        "nearest_cell": _cell_to_dict(nearest_record.cell),
        "nearest_cell_visits": nearest_record.stats.visits,
        "usage": "proposal_sorting_only",
    }


def _load_candidates(path: Path) -> list[RankingCandidate]:
    try:
        payload = json.loads(Path(path).resolve(strict=True).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SurrogateError("SURROGATE_CANDIDATES_INVALID") from exc
    if not isinstance(payload, list):
        raise SurrogateError("SURROGATE_CANDIDATES_INVALID")
    candidates = [_candidate_from_mapping(item) for item in payload]
    proposal_ids = [candidate.proposal_id for candidate in candidates]
    if len(set(proposal_ids)) != len(proposal_ids):
        raise SurrogateError("SURROGATE_CANDIDATE_IDS_DUPLICATED")
    return candidates


def _candidate_from_mapping(item: Any) -> RankingCandidate:
    if not isinstance(item, Mapping):
        raise SurrogateError("SURROGATE_CANDIDATES_INVALID")
    proposal_id = item.get("proposal_id")
    if not isinstance(proposal_id, str) or not proposal_id:
        raise SurrogateError("SURROGATE_CANDIDATE_ID_INVALID")
    cell = InterventionCell(
        environment=_require_text(item, "environment"),
        layer=_require_text(item, "layer"),
        primitive_family=_require_text(item, "primitive_family"),
        parameter_bucket=_require_text(item, "parameter_bucket"),
    )
    return RankingCandidate(proposal_id=proposal_id, cell=cell)


def _require_text(item: Mapping[str, Any], key: str) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value:
        raise SurrogateError(f"SURROGATE_CANDIDATE_{key.upper()}_INVALID")
    return value


def _cell_record_to_dict(record: CellProjectionRecord) -> dict[str, object]:
    return {
        **record.to_dict(),
        "intervention_description": intervention_description(record.cell),
    }


def _cell_to_dict(cell: InterventionCell) -> dict[str, object]:
    return {
        "environment": cell.environment,
        "layer": cell.layer,
        "primitive_family": cell.primitive_family,
        "parameter_bucket": cell.parameter_bucket,
    }


def _render_markdown(report: Mapping[str, object]) -> str:
    lines = [
        "# Surrogate Readiness",
        "",
        f"State: `{report['state']}`",
        f"Training allowed: `{report['surrogate_training_allowed']}`",
        f"Settled trials: `{report['settled_trial_count']} / {report['min_settled_trials']}`",
        f"Cell observations: `{report['cell_observation_count']}`",
        f"Prediction usage: `{report['prediction_usage']}`",
        f"Prediction enters reward/verdict: `{report['prediction_values_enter_reward_or_verdict']}`",
        "",
    ]
    blockers = report.get("blockers")
    if isinstance(blockers, list) and blockers:
        lines.extend(["## Blockers", ""])
        for blocker in blockers:
            if isinstance(blocker, Mapping):
                lines.append(f"- `{blocker.get('code')}`: observed `{blocker.get('observed')}`")
        lines.append("")
    rankings = report.get("rankings")
    if isinstance(rankings, list) and rankings:
        lines.extend(
            [
                "## Rankings",
                "",
                "| Proposal | Predicted Gain | Nearest Distance | Usage |",
                "|:--|--:|--:|:--|",
            ]
        )
        for item in rankings:
            if isinstance(item, Mapping):
                lines.append(
                    f"| {item['proposal_id']} | {item['predicted_verified_gain']} | "
                    f"{item['nearest_cosine_distance']} | {item['usage']} |"
                )
        lines.append("")
    lines.extend(["## Limitations", ""])
    for item in report["limitations"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise SurrogateError("SURROGATE_OUTPUT_EXISTS")
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="write surrogate readiness and optional ranking report")
    run.add_argument("--archive-db", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--cas-root", type=Path)
    run.add_argument("--candidate-cells", type=Path)
    run.add_argument("--min-settled-trials", type=int, default=SURROGATE_MIN_SETTLED_TRIALS)
    args = parser.parse_args(argv)
    if args.command == "run":
        manifest = run_surrogate_readiness(
            archive_db=args.archive_db,
            output_root=args.output_root,
            cas_root=args.cas_root,
            candidate_cells=args.candidate_cells,
            min_settled_trials=args.min_settled_trials,
        )
        print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    raise SurrogateError("SURROGATE_COMMAND_INVALID")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
