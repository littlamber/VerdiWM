#!/usr/bin/env python3
"""Plan autonomous ACWM screen promotion and evidence export actions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import stat
import sys
import textwrap
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from wmloop.execute.training_monitor_policy import DEFAULT_CONFIRMATION_STEPS


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_ROOT = ROOT / "results/reports"
DEFAULT_RUNTIME_PYTHON = Path(os.environ.get("VERDIWM_RUNTIME_PYTHON", sys.executable))
DEFAULT_DATA_ROOT = Path(os.environ.get("ACWM_DATA_ROOT", "data/ACWM-Phys"))
DEFAULT_CHECKPOINT_ROOT = Path(os.environ.get("ACWM_CHECKPOINT_ROOT", "checkpoints/ACWM-Phys"))
DEFAULT_RUN_SCREEN_ONE = ROOT / "results/reports/acwm-screen-v2-adaptive-budget-r1/commands/run_screen_one.sh"
DEFAULT_ARCHIVE_DB = ROOT / "results/archive.db"
DEFAULT_CAS_ROOT = ROOT / "results"
MIN_POSITIVE_SCREEN_STEPS = 512


class AcwmAutonomousPromotionPlanError(RuntimeError):
    """Autonomous promotion planning failed closed."""


def run_acwm_autonomous_promotion_plan(
    *,
    screen_summary: Path,
    output_root: Path,
    report_root: Path = DEFAULT_REPORT_ROOT,
    min_screen_steps: int = MIN_POSITIVE_SCREEN_STEPS,
    confirmation_steps: int = DEFAULT_CONFIRMATION_STEPS,
    preferred_gpus: Sequence[int] = (1, 2),
    runtime_python: Path = DEFAULT_RUNTIME_PYTHON,
    data_root: Path = DEFAULT_DATA_ROOT,
    checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT,
    run_screen_one: Path = DEFAULT_RUN_SCREEN_ONE,
    archive_db: Path = DEFAULT_ARCHIVE_DB,
    cas_root: Path = DEFAULT_CAS_ROOT,
) -> dict[str, object]:
    """Create a replayable plan from current screen rows to the next closed-loop actions."""

    if min_screen_steps < 1 or confirmation_steps <= min_screen_steps:
        raise AcwmAutonomousPromotionPlanError("ACWM_PROMOTION_STEP_POLICY_INVALID")
    if not preferred_gpus:
        raise AcwmAutonomousPromotionPlanError("ACWM_PROMOTION_PREFERRED_GPUS_EMPTY")
    summary = _load_json_object(screen_summary)
    rows = summary.get("rows")
    if not isinstance(rows, list):
        raise AcwmAutonomousPromotionPlanError("ACWM_PROMOTION_SUMMARY_ROWS_INVALID")
    reports = Path(report_root).resolve()
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise AcwmAutonomousPromotionPlanError("ACWM_PROMOTION_OUTPUT_EXISTS")

    actions: list[dict[str, object]] = []
    for raw in rows:
        if isinstance(raw, Mapping):
            actions.append(
                _action_for_row(
                    raw,
                    report_root=reports,
                    plan_root=destination,
                    min_screen_steps=min_screen_steps,
                    confirmation_steps=confirmation_steps,
                    preferred_gpus=tuple(int(gpu) for gpu in preferred_gpus),
                    runtime_python=Path(runtime_python).resolve(),
                    data_root=Path(data_root).resolve(),
                    checkpoint_root=Path(checkpoint_root).resolve(),
                    run_screen_one=Path(run_screen_one).resolve(),
                    archive_db=Path(archive_db).resolve(),
                    cas_root=Path(cas_root).resolve(),
                )
            )

    report = {
        "schema_version": 1,
        "artifact_type": "wmloop-acwm-autonomous-promotion-plan",
        "state": "ready",
        "screen_summary": str(Path(screen_summary).resolve()),
        "report_root": str(reports),
        "policy": {
            "min_positive_screen_steps": min_screen_steps,
            "confirmation_steps": confirmation_steps,
            "positive_delta_required": True,
            "action_gate_required": True,
            "visualization_after_confirmation_only": True,
            "preferred_gpus": list(preferred_gpus),
        },
        "row_count": len(actions),
        "launch_confirmation_count": sum(1 for action in actions if action["action"] == "launch_confirmation"),
        "export_visualization_count": sum(1 for action in actions if action["action"] == "export_visualization"),
        "monitor_count": sum(1 for action in actions if action["action"] == "monitor_running"),
        "rejected_or_skipped_count": sum(
            1
            for action in actions
            if action["action"]
            in {
                "reject_or_revise",
                "ignore_below_min_screen_budget",
                "skip_existing_confirmation",
                "retain_positive_visualized",
                "await_metrics",
            }
        ),
        "actions": actions,
        "limitations": [
            "This planner does not mutate protocols, goal specs, frozen evaluators, or primitive code.",
            "Confirmation launchers wait for an idle GPU and run a fresh GPU exclusivity audit before training.",
            "Visualization is gated on a ready confirmation manifest, positive primary-metric delta, and action gate pass.",
        ],
    }
    return _write_bundle(destination, report)


def _action_for_row(
    row: Mapping[str, object],
    *,
    report_root: Path,
    plan_root: Path,
    min_screen_steps: int,
    confirmation_steps: int,
    preferred_gpus: tuple[int, ...],
    runtime_python: Path,
    data_root: Path,
    checkpoint_root: Path,
    run_screen_one: Path,
    archive_db: Path,
    cas_root: Path,
) -> dict[str, object]:
    environment = _required_string(row, "environment")
    primitive = _required_string(row, "primitive")
    seed = _int_or_none(row.get("seed")) or 711
    train_steps = _int_or_none(row.get("train_steps")) or 0
    state = str(row.get("state") or "")
    screen_decision = str(row.get("screen_decision") or "")
    action_gate_pass = _bool_or_none(row.get("action_following_pass"))
    delta = _finite_or_none(row.get("delta_primary_metric"))
    visual_count = _int_or_none(row.get("official_visual_asset_count")) or 0
    candidate_retained = _bool_or_none(row.get("candidate_checkpoint_retained"))
    official_quality_gate_pass = _bool_or_none(row.get("official_quality_gate_pass"))
    base = {
        "environment": environment,
        "primitive": primitive,
        "seed": seed,
        "source_campaign_id": row.get("campaign_id"),
        "source_manifest_path": row.get("manifest_path"),
        "source_train_steps": train_steps,
        "source_state": state,
        "source_screen_decision": screen_decision,
        "delta_primary_metric": delta,
        "action_following_pass": action_gate_pass,
        "official_visual_asset_count": visual_count,
        "official_quality_gate_pass": official_quality_gate_pass,
        "rationale": "",
    }
    if state != "ready":
        return {**base, "action": "monitor_running", "rationale": "Trial has not written a ready manifest yet."}
    if train_steps < min_screen_steps:
        return {
            **base,
            "action": "ignore_below_min_screen_budget",
            "rationale": f"Runs below {min_screen_steps} steps are health checks, not method signals.",
        }
    if action_gate_pass is False:
        return {**base, "action": "reject_or_revise", "rationale": "Action-following gate failed."}
    if delta is None:
        return {**base, "action": "await_metrics", "rationale": "Primary-metric delta is unavailable."}
    if delta <= 0.0 or screen_decision == "reject_or_revise":
        return {**base, "action": "reject_or_revise", "rationale": "Frozen primary metric did not improve."}
    if official_quality_gate_pass is not True:
        return {
            **base,
            "action": "await_official_quality_gate",
            "rationale": (
                "A positive cheap horizon probe is candidate discovery only. Official ACWM eval.py at 50 "
                "inference steps must pass before confirmation or positive visualization."
            ),
        }
    if train_steps < confirmation_steps:
        campaign_id = f"acwm-screen-v2-{environment}-{primitive}-s{seed}-t{confirmation_steps}-r1"
        confirmation_root = report_root / campaign_id
        if confirmation_root.exists() or confirmation_root.is_symlink():
            return {
                **base,
                "action": "skip_existing_confirmation",
                "target_campaign_id": campaign_id,
                "target_output_root": str(confirmation_root),
                "rationale": "Confirmation output root already exists or is queued.",
            }
        launcher_path = plan_root / "launchers" / f"{campaign_id}.sh"
        visual_root = report_root / f"acwm-formal-visualization-{environment}-{primitive}-s{seed}-t{confirmation_steps}-r1"
        action = {
            **base,
            "action": "launch_confirmation",
            "target_campaign_id": campaign_id,
            "target_train_steps": confirmation_steps,
            "target_output_root": str(confirmation_root),
            "visualization_output_root": str(visual_root),
            "launcher_path": str(launcher_path),
            "rationale": "Positive rapid screen should be confirmed on the 1k checkpoint ladder, then visualized only if a retained checkpoint still passes.",
        }
        action["launcher_preview"] = _launcher_preview(launcher_path)
        action["_launcher_text"] = _confirmation_launcher(
            environment=environment,
            primitive=primitive,
            seed=seed,
            campaign_id=campaign_id,
            confirmation_root=confirmation_root,
            visual_root=visual_root,
            preferred_gpus=preferred_gpus,
            runtime_python=runtime_python,
            data_root=data_root,
            checkpoint_root=checkpoint_root,
            run_screen_one=run_screen_one,
            archive_db=archive_db,
            cas_root=cas_root,
        )
        return action
    if visual_count > 0:
        return {**base, "action": "retain_positive_visualized", "rationale": "Positive confirmation already has retained visual assets."}
    if candidate_retained is False:
        return {**base, "action": "await_metrics", "rationale": "Positive confirmation lacks a retained candidate checkpoint for visualization."}
    visual_root = report_root / f"acwm-formal-visualization-{environment}-{primitive}-s{seed}-t{train_steps}-r1"
    candidate = report_root / str(row.get("output_root") or "") / "retained_training/latest.pt"
    command = [
        str(ROOT / ".venv/bin/python3"),
        str(ROOT / "scripts/export/acwm_formal_visualization.py"),
        "--output-root",
        str(visual_root),
        "--environment",
        environment,
        "--primitive",
        primitive,
        "--seed",
        str(seed),
        "--runtime-python",
        str(runtime_python),
        "--data-root",
        str(data_root),
        "--checkpoint-root",
        str(checkpoint_root),
        "--candidate-checkpoint",
        str(candidate),
        "--gpu-index",
        str(preferred_gpus[0]),
        "--steps",
        "50",
        "--split",
        "ind_test",
        "--max-trajs",
        "3",
        "--max-saved-vids",
        "3",
        "--batch-size",
        "1",
        "--num-workers",
        "2",
        "--test-cuts",
        "1",
    ]
    return {
        **base,
        "action": "export_visualization",
        "visualization_output_root": str(visual_root),
        "candidate_checkpoint": str(candidate),
        "command": command,
        "rationale": "Confirmed positive cell lacks retained side-by-side videos.",
    }


def _confirmation_launcher(
    *,
    environment: str,
    primitive: str,
    seed: int,
    campaign_id: str,
    confirmation_root: Path,
    visual_root: Path,
    preferred_gpus: tuple[int, ...],
    runtime_python: Path,
    data_root: Path,
    checkpoint_root: Path,
    run_screen_one: Path,
    archive_db: Path,
    cas_root: Path,
) -> str:
    preferred = ",".join(str(gpu) for gpu in preferred_gpus)
    return textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail

        ROOT={_sh(ROOT)}
        ENV_NAME={_sh(environment)}
        PRIMITIVE={_sh(primitive)}
        SEED={seed}
        CAMPAIGN_ID={_sh(campaign_id)}
        OUTPUT_ROOT={_sh(confirmation_root)}
        VIS_ROOT={_sh(visual_root)}
        RUNTIME={_sh(runtime_python)}
        DATA_ROOT={_sh(data_root)}
        CHECKPOINT_ROOT={_sh(checkpoint_root)}
        RUN_SCREEN_ONE={_sh(run_screen_one)}
        ARCHIVE_DB={_sh(archive_db)}
        CAS_ROOT={_sh(cas_root)}
        PREFERRED_GPUS="${{PREFERRED_GPUS:-{preferred}}}"
        POLL_SECONDS="${{POLL_SECONDS:-60}}"
        LAUNCH_STATUS="${{OUTPUT_ROOT}}.promotion-status.json"

        write_status() {{
          local state="$1"
          local stage="$2"
          local gpu="${{3:-}}"
          "$ROOT/.venv/bin/python3" - "$LAUNCH_STATUS" "$state" "$stage" "$gpu" "$CAMPAIGN_ID" <<'PY'
        import json
        import sys
        from datetime import datetime, timezone
        from pathlib import Path
        path = Path(sys.argv[1])
        payload = {{
            "schema_version": 1,
            "artifact_type": "wmloop-autonomous-promotion-launch-status",
            "state": sys.argv[2],
            "stage": sys.argv[3],
            "selected_gpu": sys.argv[4] or None,
            "campaign_id": sys.argv[5],
            "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
        PY
        }}

        select_free_gpu() {{
          "$ROOT/.venv/bin/python3" - "$PREFERRED_GPUS" "$ENV_NAME" "$PRIMITIVE" <<'PY'
        import csv
        import subprocess
        import sys
        preferred = [int(value) for value in sys.argv[1].replace(",", " ").split()]
        env_name = sys.argv[2]
        primitive = sys.argv[3]
        raw = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ], text=True, timeout=10)
        rows = {{}}
        for row in csv.reader(raw.splitlines()):
            if len(row) >= 3:
                rows[int(row[0].strip())] = (int(row[1].strip()), int(row[2].strip()))
        ps_raw = subprocess.check_output(["ps", "-eo", "args"], text=True, timeout=10)
        for index in preferred:
            memory_used, utilization = rows.get(index, (10**9, 100))
            busy_marker = (
                f"--gpu-index {{index}}" in ps_raw
                or f"--gpus {{index}}" in ps_raw
                or f"run_screen_one.sh {{env_name}} {{primitive}} {{index}} " in ps_raw
            )
            if memory_used <= 1024 and utilization <= 5 and not busy_marker:
                print(index)
                raise SystemExit(0)
        raise SystemExit(1)
        PY
        }}

        write_status waiting waiting_for_gpu
        while true; do
          if SELECTED_GPU="$(select_free_gpu)"; then
            break
          fi
          sleep "$POLL_SECONDS"
        done

        write_status running gpu_exclusivity_audit "$SELECTED_GPU"
        AUDIT_ROOT="$ROOT/results/reports/gpu-exclusivity-audit-${{CAMPAIGN_ID}}-gpu${{SELECTED_GPU}}-$(date -u +%Y%m%dT%H%M%SZ)"
        "$ROOT/.venv/bin/python3" -m wmloop.execute.gpu_exclusivity_audit run \\
          --output-root "$AUDIT_ROOT" \\
          --gpus "$SELECTED_GPU" \\
          --archive-db "$ARCHIVE_DB" \\
          --cas-root "$CAS_ROOT"
        AUDIT_MANIFEST="$AUDIT_ROOT/manifest.json"

        write_status running training_eval "$SELECTED_GPU"
        "$RUN_SCREEN_ONE" "$ENV_NAME" "$PRIMITIVE" "$SELECTED_GPU" "$SEED" "$CAMPAIGN_ID" "$OUTPUT_ROOT" "$AUDIT_MANIFEST" 1000 0

        write_status running positive_gate "$SELECTED_GPU"
        if "$ROOT/.venv/bin/python3" - "$OUTPUT_ROOT/envs/$ENV_NAME/manifest.json" "$OUTPUT_ROOT.positive-gate.json" <<'PY'
        import json
        import math
        import sys
        from pathlib import Path
        manifest_path = Path(sys.argv[1])
        gate_path = Path(sys.argv[2])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metric = manifest.get("primary_metric", "ladder_auc_psnr_envmax")
        deltas = manifest.get("delta_m_ver") if isinstance(manifest.get("delta_m_ver"), dict) else {{}}
        delta = deltas.get(metric)
        action_gate = manifest.get("action_following_gate") if isinstance(manifest.get("action_following_gate"), dict) else {{}}
        positive = (
            manifest.get("state") == "ready"
            and isinstance(delta, (int, float))
            and not isinstance(delta, bool)
            and math.isfinite(float(delta))
            and float(delta) > 0.0
            and (action_gate.get("enabled") is not True or action_gate.get("pass") is True)
        )
        payload = {{
            "schema_version": 1,
            "artifact_type": "wmloop-positive-gate",
            "state": "positive" if positive else "nonpositive",
            "manifest_path": str(manifest_path),
            "primary_metric": metric,
            "delta_primary_metric": delta,
            "action_following_gate": action_gate,
        }}
        gate_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
        raise SystemExit(0 if positive else 42)
        PY
        then
          GATE_CODE=0
        else
          GATE_CODE="$?"
        fi
        if [[ "$GATE_CODE" == "42" ]]; then
          write_status ready nonpositive_no_visualization "$SELECTED_GPU"
          exit 0
        elif [[ "$GATE_CODE" != "0" ]]; then
          exit "$GATE_CODE"
        fi

        write_status running visualization_if_positive "$SELECTED_GPU"
        "$ROOT/.venv/bin/python3" "$ROOT/scripts/export/acwm_formal_visualization.py" \\
          --output-root "$VIS_ROOT" \\
          --environment "$ENV_NAME" \\
          --primitive "$PRIMITIVE" \\
          --seed "$SEED" \\
          --runtime-python "$RUNTIME" \\
          --data-root "$DATA_ROOT" \\
          --checkpoint-root "$CHECKPOINT_ROOT" \\
          --candidate-checkpoint "$OUTPUT_ROOT/envs/$ENV_NAME/retained_training/latest.pt" \\
          --gpu-index "$SELECTED_GPU" \\
          --steps 50 \\
          --split ind_test \\
          --max-trajs 3 \\
          --max-saved-vids 3 \\
          --batch-size 1 \\
          --num-workers 2 \\
          --test-cuts 1
        write_status ready complete_positive_visualized "$SELECTED_GPU"
        """
    )


def _write_bundle(destination: Path, report: dict[str, object]) -> dict[str, object]:
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.mkdir(mode=0o700)
        public_actions = []
        for action in report["actions"]:
            if not isinstance(action, dict):
                continue
            launcher_text = action.pop("_launcher_text", None)
            launcher_path = action.get("launcher_path")
            if isinstance(launcher_text, str) and isinstance(launcher_path, str):
                target = temporary / Path(launcher_path).relative_to(destination)
                target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                target.write_text(launcher_text, encoding="utf-8")
                target.chmod(target.stat().st_mode | stat.S_IXUSR)
            public_actions.append(action)
        report["actions"] = public_actions
        _write_json(temporary / "promotion-plan.json", report)
        _write_csv(temporary / "promotion-actions.csv", public_actions)
        _write_markdown(temporary / "promotion-plan.md", report)
        os.replace(temporary, destination)
    except Exception:
        if temporary.is_symlink():
            temporary.unlink()
        elif temporary.exists():
            shutil.rmtree(temporary)
        raise
    return {
        "schema_version": report["schema_version"],
        "artifact_type": report["artifact_type"],
        "state": report["state"],
        "row_count": report["row_count"],
        "launch_confirmation_count": report["launch_confirmation_count"],
        "export_visualization_count": report["export_visualization_count"],
        "monitor_count": report["monitor_count"],
        "rejected_or_skipped_count": report["rejected_or_skipped_count"],
        "report_path": str(destination / "promotion-plan.json"),
        "markdown_path": str(destination / "promotion-plan.md"),
        "csv_path": str(destination / "promotion-actions.csv"),
    }


def _write_markdown(path: Path, report: Mapping[str, object]) -> None:
    lines = [
        "# ACWM Autonomous Promotion Plan",
        "",
        f"State: `{report['state']}`",
        f"Launch confirmations: `{report['launch_confirmation_count']}`",
        f"Export visualizations: `{report['export_visualization_count']}`",
        f"Monitoring rows: `{report['monitor_count']}`",
        "",
        "| Env | Primitive | Seed | Source Steps | Delta | Action | Rationale |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for action in report.get("actions", []):
        if isinstance(action, Mapping):
            lines.append(
                "| {environment} | {primitive} | {seed} | {source_train_steps} | {delta_primary_metric} | `{action}` | {rationale} |".format(
                    environment=action.get("environment", ""),
                    primitive=action.get("primitive", ""),
                    seed=action.get("seed", ""),
                    source_train_steps=action.get("source_train_steps", ""),
                    delta_primary_metric=action.get("delta_primary_metric", ""),
                    action=action.get("action", ""),
                    rationale=action.get("rationale", ""),
                )
            )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "environment",
        "primitive",
        "seed",
        "source_train_steps",
        "delta_primary_metric",
        "action_following_pass",
        "official_visual_asset_count",
        "action",
        "target_campaign_id",
        "target_train_steps",
        "target_output_root",
        "visualization_output_root",
        "launcher_path",
        "rationale",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json_object(path: Path) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise AcwmAutonomousPromotionPlanError(f"ACWM_PROMOTION_JSON_NOT_OBJECT:{path}")
    return payload


def _required_string(row: Mapping[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise AcwmAutonomousPromotionPlanError(f"ACWM_PROMOTION_ROW_FIELD_INVALID:{key}")
    return value


def _finite_or_none(value: object) -> float | None:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _int_or_none(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip():
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _bool_or_none(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    return None


def _launcher_preview(path: Path) -> str:
    return f"setsid {path} > {path}.stdout 2> {path}.stderr < /dev/null &"


def _sh(path_or_value: object) -> str:
    value = str(path_or_value)
    return "'" + value.replace("'", "'\"'\"'") + "'"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", nargs="?")
    parser.add_argument("--screen-summary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--min-screen-steps", type=int, default=MIN_POSITIVE_SCREEN_STEPS)
    parser.add_argument("--confirmation-steps", type=int, default=DEFAULT_CONFIRMATION_STEPS)
    parser.add_argument("--preferred-gpu", type=int, action="append", dest="preferred_gpus")
    parser.add_argument("--runtime-python", type=Path, default=DEFAULT_RUNTIME_PYTHON)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--run-screen-one", type=Path, default=DEFAULT_RUN_SCREEN_ONE)
    parser.add_argument("--archive-db", type=Path, default=DEFAULT_ARCHIVE_DB)
    parser.add_argument("--cas-root", type=Path, default=DEFAULT_CAS_ROOT)
    args = parser.parse_args(argv)
    manifest = run_acwm_autonomous_promotion_plan(
        screen_summary=args.screen_summary,
        output_root=args.output_root,
        report_root=args.report_root,
        min_screen_steps=args.min_screen_steps,
        confirmation_steps=args.confirmation_steps,
        preferred_gpus=tuple(args.preferred_gpus or (1, 2)),
        runtime_python=args.runtime_python,
        data_root=args.data_root,
        checkpoint_root=args.checkpoint_root,
        run_screen_one=args.run_screen_one,
        archive_db=args.archive_db,
        cas_root=args.cas_root,
    )
    print(json.dumps(manifest, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
