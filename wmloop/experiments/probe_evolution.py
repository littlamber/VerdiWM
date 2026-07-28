"""Counterexample-driven admission records for diagnostic-probe evolution."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from wmloop.experiments._artifacts import canonical_json, write_bundle


class ProbeEvolutionError(ValueError):
    """Probe-evolution inputs do not provide an admissible counterexample."""


def build_probe_evolution_proposal(
    *,
    failed_fingerprints: Sequence[Path],
    successor_campaign: Path,
    output_root: Path,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    failures: list[dict[str, object]] = []
    retired_ids: set[str] = set()
    for path in failed_fingerprints:
        report = _load_json(Path(path))
        chart = report.get("chart")
        locality = report.get("locality_admission")
        if not isinstance(chart, Mapping):
            raise ProbeEvolutionError(f"PROBE_EVOLUTION_FINGERPRINT_INVALID:{path}")
        names = chart.get("intervention_names")
        if not isinstance(names, list) or len(names) != 1 or not isinstance(names[0], str):
            raise ProbeEvolutionError(f"PROBE_EVOLUTION_MULTI_PATH_UNSUPPORTED:{path}")
        if isinstance(locality, Mapping):
            if locality.get("state") != "failed" or locality.get("cross_backbone_transfer_eligible") is not False:
                raise ProbeEvolutionError(f"PROBE_EVOLUTION_COUNTEREXAMPLE_REQUIRED:{path}")
            paths = locality.get("path_residuals")
            threshold = locality.get("maximum_residual")
            locality_source = "recorded_locality_admission"
        else:
            # The wide pilot predates explicit admission serialization. Its chart
            # still records the residual, so preserve it as a legacy counterexample
            # under the same v1 threshold rather than discarding the raw evidence.
            paths = chart.get("locality_residuals")
            threshold = 0.5
            locality_source = "legacy_chart_reconstructed_at_v1_threshold"
        if not isinstance(paths, Mapping) or not paths:
            raise ProbeEvolutionError(f"PROBE_EVOLUTION_RESIDUALS_INVALID:{path}")
        observed = paths.get(names[0])
        if not isinstance(observed, (int, float)) or float(observed) <= float(threshold):
            raise ProbeEvolutionError(f"PROBE_EVOLUTION_COUNTEREXAMPLE_REQUIRED:{path}")
        retired_ids.add(names[0])
        failures.append(
            {
                "fingerprint_path": str(Path(path).resolve()),
                "campaign_id": report.get("campaign_id"),
                "probe_id": names[0],
                "maximum_residual": threshold,
                "observed_residual": observed,
                "supported_local_paths": locality.get("supported_local_paths", []) if isinstance(locality, Mapping) else [],
                "locality_evidence_source": locality_source,
            }
        )
    if not failures or len(retired_ids) != 1:
        raise ProbeEvolutionError("PROBE_EVOLUTION_COUNTEREXAMPLES_INCOMPATIBLE")

    campaign = _load_json(Path(successor_campaign))
    probe = campaign.get("probe")
    if campaign.get("artifact_type") != "verdiwm-ctrl-world-fingerprint-campaign" or not isinstance(probe, Mapping):
        raise ProbeEvolutionError("PROBE_EVOLUTION_SUCCESSOR_CAMPAIGN_INVALID")
    successor_id = probe.get("probe_id")
    if not isinstance(successor_id, str) or not successor_id or successor_id in retired_ids:
        raise ProbeEvolutionError("PROBE_EVOLUTION_SUCCESSOR_NOT_NOVEL")
    if probe.get("scope") != "inference_only" or probe.get("reversible") is not True:
        raise ProbeEvolutionError("PROBE_EVOLUTION_SUCCESSOR_SCOPE_INVALID")
    expected_outcomes = {"rollout_video_psnr", "negative_rollout_video_l1", "negative_segment_final_mae", "negative_segment_view_pair_mae", "negative_segment_view_fused_mae"}
    outcomes = campaign.get("outcomes")
    outcome_names = {
        str(row.get("name")) for row in outcomes if isinstance(row, Mapping) and isinstance(row.get("name"), str)
    } if isinstance(outcomes, list) else set()
    if outcome_names != expected_outcomes:
        raise ProbeEvolutionError("PROBE_EVOLUTION_OUTCOME_CONTRACT_CHANGED")

    report = {
        "schema_version": 1,
        "artifact_type": "verdiwm-diagnostic-probe-evolution-proposal",
        "state": "ready",
        "backbone_family": "Ctrl-World ACWM predictive",
        "retired_probe_ids": sorted(retired_ids),
        "counterexample_count": len(failures),
        "counterexamples": failures,
        "successor_campaign": str(Path(successor_campaign).resolve()),
        "successor_probe": dict(probe),
        "invariants": [
            "diagnostic probe evolution does not alter the frozen predictive verdict metrics",
            "successor uses paired identical action trajectories, episodes, seeds, checkpoint, and evaluator",
            "cross-backbone transfer remains abstained until the successor passes the same locality admission",
        ],
        "claim_boundary": "This artifact proposes a diagnostic measurement replacement only. It is neither a model improvement claim nor a transfer certificate.",
    }
    destination = Path(output_root).resolve()
    return write_bundle(
        output_root=destination,
        files={
            "probe-evolution-proposal.json": canonical_json(report),
            "probe-evolution-proposal.md": _markdown(report).encode("utf-8"),
            "input-successor-campaign.json": canonical_json(campaign),
        },
        manifest_fields={
            "artifact_type": "verdiwm-diagnostic-probe-evolution-proposal-manifest",
            "state": "ready",
            "counterexample_count": len(failures),
            "retired_probe_id": next(iter(retired_ids)),
            "successor_probe_id": successor_id,
            "report_path": str(destination / "probe-evolution-proposal.json"),
        },
        archive_db=archive_db,
        cas_root=cas_root,
    )


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProbeEvolutionError(f"PROBE_EVOLUTION_JSON_INVALID:{path}") from exc
    if not isinstance(payload, Mapping):
        raise ProbeEvolutionError(f"PROBE_EVOLUTION_JSON_INVALID:{path}")
    return payload


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Diagnostic Probe Evolution Proposal",
        "",
        str(report["claim_boundary"]),
        "",
        f"Retired probe: `{', '.join(report['retired_probe_ids'])}`",
        f"Counterexamples: `{report['counterexample_count']}`",
        f"Successor probe: `{report['successor_probe']['probe_id']}`",
        "",
        "| Campaign | Observed locality residual | Threshold |",
        "|---|---:|---:|",
    ]
    for row in report["counterexamples"]:
        lines.append(f"| {row['campaign_id']} | {float(row['observed_residual']):.6f} | {float(row['maximum_residual']):.6f} |")
    return "\n".join(lines) + "\n"
