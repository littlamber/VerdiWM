#!/usr/bin/env python3
"""Export one fail-closed, GitHub-sized VerdiWM minimal-loop proof."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class MinimalLoopBundleError(RuntimeError):
    """The supplied artifacts do not prove one coherent minimal loop."""


def export_minimal_loop_bundle(
    *,
    failure_report: Path,
    intervention_receipt: Path,
    screen_manifest: Path,
    official_gate_manifest: Path,
    confirmation_manifest: Path,
    checkpoint_ladder: Path,
    effect_profile: Path,
    experience_map: Path,
    output_root: Path,
    showcase_video: Path | None = None,
) -> dict[str, object]:
    sources = {
        "failure_report": Path(failure_report).resolve(strict=True),
        "intervention_receipt": Path(intervention_receipt).resolve(strict=True),
        "screen_manifest": Path(screen_manifest).resolve(strict=True),
        "official_gate_manifest": Path(official_gate_manifest).resolve(strict=True),
        "confirmation_manifest": Path(confirmation_manifest).resolve(strict=True),
        "checkpoint_ladder": Path(checkpoint_ladder).resolve(strict=True),
        "effect_profile": Path(effect_profile).resolve(strict=True),
        "experience_map": Path(experience_map).resolve(strict=True),
    }
    if showcase_video is not None:
        sources["showcase_video"] = Path(showcase_video).resolve(strict=True)
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise MinimalLoopBundleError("MINIMAL_LOOP_OUTPUT_EXISTS")

    payloads = {name: _load_json(path) for name, path in sources.items() if name != "showcase_video"}
    identity = _validate_identity(payloads)
    checks = _validate_chain(payloads, identity=identity)
    if not all(checks.values()):
        failures = ",".join(name for name, passed in checks.items() if not passed)
        raise MinimalLoopBundleError(f"MINIMAL_LOOP_CHAIN_INVALID:{failures}")

    gate = _mapping(payloads["official_gate_manifest"], "official_quality_gate")
    confirmation = _mapping(payloads["confirmation_manifest"], "official_quality_gate")
    effect = payloads["effect_profile"]
    experience = payloads["experience_map"]
    screen = payloads["screen_manifest"]
    ladder = payloads["checkpoint_ladder"]
    gate_seed = int(payloads["official_gate_manifest"].get("eval_seed", payloads["official_gate_manifest"]["seed"]))
    confirmation_seed = int(payloads["confirmation_manifest"].get("eval_seed", payloads["confirmation_manifest"]["seed"]))
    evaluation_seed_independent = gate_seed != confirmation_seed
    candidate_checkpoint_independent = (
        payloads["official_gate_manifest"].get("candidate_checkpoint_sha256")
        != payloads["confirmation_manifest"].get("candidate_checkpoint_sha256")
    )
    causal_credit_eligible = bool(
        _mapping(effect, "transfer_prior").get("causal_credit_eligible") is True
        and evaluation_seed_independent
    )

    report: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-minimal-loop-proof",
        "state": "ready",
        "identity": identity,
        "operational_minimal_loop_pass": True,
        "paper_confirmed_effect": causal_credit_eligible,
        "framework_projection": {
            "goal_and_verifier": {
                "goal_id": identity["goal_id"],
                "official_protocol": gate["protocol"],
                "frozen_gate_checks": dict(_mapping(gate, "checks")),
            },
            "diagnostic_state": {
                "dominant_failure": payloads["failure_report"].get("dominant_failure"),
                "failure_candidates": payloads["failure_report"].get("dominant_failure_candidates", []),
                "diagnostic_only": True,
            },
            "typed_intervention": {
                "primitive": identity["primitive"],
                "hook": payloads["intervention_receipt"].get("hook"),
                "materialization_state": payloads["intervention_receipt"].get("materialization_state"),
                "intent_to_code_contract": payloads["intervention_receipt"].get("intent_to_code_contract"),
            },
            "progressive_fidelity": {
                "screen_steps": 512,
                "screen_primary_delta": _mapping(screen, "delta_m_ver").get("ladder_auc_psnr_envmax"),
                "official_gate_delta": gate.get("delta_candidate_minus_baseline"),
                "confirmation_gate_delta": confirmation.get("delta_candidate_minus_baseline"),
                "selected_checkpoint_step": ladder.get("best_checkpoint_step"),
            },
            "interventional_effect": {
                "effect_scope": _mapping(effect, "effect_classification").get("effect_scope"),
                "horizon_psnr_delta": {
                    str(row["horizon"]): _mapping(row, "delta_candidate_minus_baseline")["psnr"]
                    for row in _list_of_mappings(effect, "horizon_effects")
                },
                "effective_horizons": _mapping(effect, "effect_classification").get("aggregate_passing_horizons"),
                "positive_trajectory_rate_at_max_horizon": _mapping(
                    effect, "effect_classification"
                ).get("positive_trajectory_rate_at_max_horizon"),
            },
            "effect_memory": {
                "admission": "routing_prior_only" if not causal_credit_eligible else "confirmed_local_effect",
                "routing_prior_count": _mapping(experience, "summary").get("routing_prior_count"),
                "causal_edge_count": _mapping(experience, "summary").get("causal_edge_count"),
                "causal_credit_eligible": causal_credit_eligible,
            },
        },
        "chain_checks": checks,
        "confirmation_independence": {
            "initial_eval_seed": gate_seed,
            "confirmation_eval_seed": confirmation_seed,
            "evaluation_seed_independent": evaluation_seed_independent,
            "candidate_checkpoint_independent": candidate_checkpoint_independent,
            "paper_replication_requirement_pass": evaluation_seed_independent,
        },
        "claim_boundary": {
            "proven": (
                "A diagnosis-routed primitive was materially compiled, produced a positive 512-step screen, "
                "passed the frozen official 50-step pixel gate, survived checkpoint-ladder confirmation, "
                "and yielded an aggregate long-horizon routing prior."
            ),
            "not_proven": (
                "This bundle does not by itself establish IRG cross-backbone transfer, a calibrated transfer "
                "certificate, or a paper-level replicated causal effect when evaluation seeds are shared."
            ),
        },
    }
    replacements = _public_path_replacements(
        official_gate=payloads["official_gate_manifest"],
        source_paths=sources,
    )
    return _write_bundle(
        destination=destination,
        report=report,
        sources=sources,
        replacements=replacements,
    )


def _validate_identity(payloads: Mapping[str, Mapping[str, Any]]) -> dict[str, object]:
    failure = payloads["failure_report"]
    environment = str(failure.get("env") or "")
    goal_id = str(failure.get("goal_id") or "")
    intervention = payloads["intervention_receipt"]
    primitive = str(intervention.get("primitive") or "")
    gate = payloads["official_gate_manifest"]
    try:
        seed = int(gate["seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise MinimalLoopBundleError("MINIMAL_LOOP_IDENTITY_INVALID") from exc
    if not environment or not goal_id or not primitive:
        raise MinimalLoopBundleError("MINIMAL_LOOP_IDENTITY_INVALID")

    for name in ("screen_manifest", "official_gate_manifest", "confirmation_manifest", "checkpoint_ladder", "effect_profile"):
        payload = payloads[name]
        if str(payload.get("environment") or "") != environment:
            raise MinimalLoopBundleError(f"MINIMAL_LOOP_ENVIRONMENT_MISMATCH:{name}")
    for name in ("official_gate_manifest", "confirmation_manifest", "checkpoint_ladder", "effect_profile"):
        if str(payloads[name].get("primitive") or "") != primitive:
            raise MinimalLoopBundleError(f"MINIMAL_LOOP_PRIMITIVE_MISMATCH:{name}")
    for name in ("screen_manifest", "official_gate_manifest", "confirmation_manifest", "checkpoint_ladder"):
        if int(payloads[name].get("seed", -1)) != seed:
            raise MinimalLoopBundleError(f"MINIMAL_LOOP_SEED_MISMATCH:{name}")
    return {"environment": environment, "primitive": primitive, "seed": seed, "goal_id": goal_id}


def _validate_chain(
    payloads: Mapping[str, Mapping[str, Any]], *, identity: Mapping[str, object]
) -> dict[str, bool]:
    screen = payloads["screen_manifest"]
    gate_manifest = payloads["official_gate_manifest"]
    confirm_manifest = payloads["confirmation_manifest"]
    gate = _mapping(gate_manifest, "official_quality_gate")
    confirm = _mapping(confirm_manifest, "official_quality_gate")
    ladder = payloads["checkpoint_ladder"]
    effect = payloads["effect_profile"]
    experience = payloads["experience_map"]
    intervention = payloads["intervention_receipt"]

    screen_delta = _mapping(screen, "delta_m_ver").get("ladder_auc_psnr_envmax")
    routing = [
        row
        for row in _list_of_mappings(experience, "routing_priors")
        if row.get("environment") == identity["environment"] and row.get("primitive") == identity["primitive"]
    ]
    return {
        "diagnosis_ready": bool(payloads["failure_report"].get("dominant_failure")),
        "intervention_materialized": intervention.get("materialization_state") == "acwm_runtime_hook_smoke"
        and isinstance(intervention.get("runtime_hook"), Mapping),
        "screen_ready": screen.get("state") == "ready" and _positive(screen_delta),
        "screen_action_gate_pass": _mapping(screen, "action_following_gate").get("pass") is True,
        "official_gate_pass": _quality_gate_pass(gate),
        "confirmation_gate_pass": _quality_gate_pass(confirm),
        "same_frozen_baseline": gate_manifest.get("baseline_checkpoint_sha256")
        == confirm_manifest.get("baseline_checkpoint_sha256"),
        "selected_checkpoint_confirmed": ladder.get("state") == "ready"
        and ladder.get("best_checkpoint_sha256") == confirm_manifest.get("candidate_checkpoint_sha256"),
        "long_horizon_effect_positive": effect.get("state") == "ready"
        and _mapping(effect, "effect_classification").get("aggregate_max_horizon_pass") is True,
        "routing_prior_written": experience.get("state") == "ready" and bool(routing),
        "verifier_protocol_frozen": _same_protocol_provenance(gate_manifest, confirm_manifest),
    }


def _quality_gate_pass(gate: Mapping[str, Any]) -> bool:
    checks = _mapping(gate, "checks")
    required = {
        "psnr_strictly_improves",
        "ssim_does_not_regress",
        "mse_does_not_regress",
        "masked_mse_does_not_regress",
    }
    return gate.get("pass") is True and gate.get("state") == "pass" and all(checks.get(key) is True for key in required)


def _same_protocol_provenance(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    a, b = _mapping(left, "protocol_provenance"), _mapping(right, "protocol_provenance")
    keys = ("dataset_freeze_sha256", "heldout_protocol_sha256", "eval_script_sha256", "eval_config_sha256")
    return all(isinstance(a.get(key), str) and a.get(key) == b.get(key) for key in keys)


def _write_bundle(
    *,
    destination: Path,
    report: Mapping[str, object],
    sources: Mapping[str, Path],
    replacements: Mapping[str, str],
) -> dict[str, object]:
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir(parents=True)
    try:
        evidence = temporary / "evidence"
        media = temporary / "media"
        evidence.mkdir()
        copied: list[dict[str, object]] = []
        for name, source in sources.items():
            target_dir = media if name == "showcase_video" else evidence
            suffix = source.suffix or ".json"
            target = target_dir / f"{name}{suffix}"
            if name == "showcase_video":
                media.mkdir(exist_ok=True)
            source_sha256 = _sha256(source)
            if name == "showcase_video":
                shutil.copy2(source, target)
            else:
                payload = _load_json(source)
                target.write_text(
                    json.dumps(
                        _replace_paths(payload, replacements),
                        sort_keys=True,
                        indent=2,
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            copied.append(
                {
                    "role": name,
                    "path": str(target.relative_to(temporary)),
                    "source_sha256": source_sha256,
                    "sha256": _sha256(target),
                    "size_bytes": target.stat().st_size,
                }
            )
        final_report = dict(report)
        final_report["artifacts"] = copied
        proof_path = temporary / "minimal-loop-proof.json"
        proof_path.write_text(
            json.dumps(final_report, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (temporary / "README.md").write_text(_markdown(final_report), encoding="utf-8")
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return final_report


def _markdown(report: Mapping[str, Any]) -> str:
    identity = _mapping(report, "identity")
    projection = _mapping(report, "framework_projection")
    fidelity = _mapping(projection, "progressive_fidelity")
    effect = _mapping(projection, "interventional_effect")
    independence = _mapping(report, "confirmation_independence")
    lines = [
        "# VerdiWM Minimal Closed-Loop Proof",
        "",
        f"- Environment: `{identity['environment']}`",
        f"- Primitive: `{identity['primitive']}`",
        f"- Seed: `{identity['seed']}`",
        f"- Operational loop: `PASS`",
        f"- Paper-level replicated effect: `{'PASS' if report['paper_confirmed_effect'] else 'PENDING'}`",
        "",
        "## Progressive Fidelity",
        "",
        "| Stage | Evidence |",
        "|---|---|",
        f"| 512-step screen | AUC delta `{float(fidelity['screen_primary_delta']):.4f}` |",
        f"| Official 50-step gate | PSNR delta `{float(_mapping(fidelity, 'official_gate_delta')['psnr']):+.4f}` |",
        f"| Checkpoint confirmation | PSNR delta `{float(_mapping(fidelity, 'confirmation_gate_delta')['psnr']):+.4f}` |",
        f"| Best checkpoint | relative step `{fidelity['selected_checkpoint_step']}` |",
        f"| Long horizon | `{effect['effect_scope']}` on `{effect['effective_horizons']}` |",
        "",
        "## Claim Boundary",
        "",
        str(_mapping(report, "claim_boundary")["proven"]),
        "",
        str(_mapping(report, "claim_boundary")["not_proven"]),
        "",
        "The initial and confirmation evaluations use "
        + ("different" if independence["evaluation_seed_independent"] else "the same")
        + " evaluation seed. This is recorded explicitly rather than being promoted to an unsupported replication claim.",
        "",
        "## Files",
        "",
    ]
    lines.extend(
        f"- `{row['path']}` (`{row['role']}`, public `{row['sha256']}`, source `{row['source_sha256']}`)"
        for row in report["artifacts"]
    )
    return "\n".join(lines) + "\n"


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MinimalLoopBundleError(f"MINIMAL_LOOP_JSON_INVALID:{path}") from exc
    if not isinstance(payload, Mapping):
        raise MinimalLoopBundleError(f"MINIMAL_LOOP_JSON_INVALID:{path}")
    return payload


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise MinimalLoopBundleError(f"MINIMAL_LOOP_MAPPING_MISSING:{key}")
    return value


def _list_of_mappings(payload: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    value = payload.get(key)
    if not isinstance(value, list) or any(not isinstance(row, Mapping) for row in value):
        raise MinimalLoopBundleError(f"MINIMAL_LOOP_ROWS_MISSING:{key}")
    return value


def _positive(value: object) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _public_path_replacements(
    *, official_gate: Mapping[str, Any], source_paths: Mapping[str, Path]
) -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    replacements = {str(root): "${VERDIWM_REPO}"}
    for field, token in (
        ("data_root", "${ACWM_DATA_ROOT}"),
        ("checkpoint_root", "${ACWM_CHECKPOINT_ROOT}"),
        ("runtime_python", "${VERDIWM_RUNTIME_PYTHON}"),
    ):
        value = official_gate.get(field)
        if isinstance(value, str) and value.startswith("/"):
            replacements[value] = token
    for path in source_paths.values():
        if str(path).startswith(str(root)):
            continue
        if path.suffix == ".mp4":
            continue
        replacements.setdefault(str(path.parent), f"${{EXTERNAL_{path.parent.name.upper()}_ROOT}}")
    return dict(sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True))


def _replace_paths(value: Any, replacements: Mapping[str, str]) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _replace_paths(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_paths(item, replacements) for item in value]
    if isinstance(value, str):
        result = value
        for source, replacement in replacements.items():
            result = result.replace(source, replacement)
        return result
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failure-report", type=Path, required=True)
    parser.add_argument("--intervention-receipt", type=Path, required=True)
    parser.add_argument("--screen-manifest", type=Path, required=True)
    parser.add_argument("--official-gate-manifest", type=Path, required=True)
    parser.add_argument("--confirmation-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-ladder", type=Path, required=True)
    parser.add_argument("--effect-profile", type=Path, required=True)
    parser.add_argument("--experience-map", type=Path, required=True)
    parser.add_argument("--showcase-video", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    export_minimal_loop_bundle(
        failure_report=args.failure_report,
        intervention_receipt=args.intervention_receipt,
        screen_manifest=args.screen_manifest,
        official_gate_manifest=args.official_gate_manifest,
        confirmation_manifest=args.confirmation_manifest,
        checkpoint_ladder=args.checkpoint_ladder,
        effect_profile=args.effect_profile,
        experience_map=args.experience_map,
        showcase_video=args.showcase_video,
        output_root=args.output_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
