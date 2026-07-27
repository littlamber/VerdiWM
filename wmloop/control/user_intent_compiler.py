"""Compile a human intent statement into a read-only goal/protocol binding packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from wmloop.archive.store import ArchiveStore, ContentAddressedStore
from wmloop.contracts import ContractValidationError, load_yaml_document, validate_document


class UserIntentCompilerError(RuntimeError):
    """The intent compiler failed before a durable packet could be written."""


ACWM_PHYS_ENVS = (
    "push_cube",
    "stack_cube",
    "push_rope",
    "cloth_move",
    "push_sand",
    "pour_water",
    "robot_arm",
    "reacher",
)


def run_user_intent_compilation(
    *,
    repo_root: Path,
    intent_text: str,
    goal_config: Path,
    output_root: Path,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    """Write a durable packet binding user intent to an existing goal config.

    The compiler is deliberately conservative.  It may identify that limited
    smoke/admission work can continue under explicit filters, but it never
    mutates the goal, changes protocol, or grants formal M4 launch permission.
    """

    root = Path(repo_root).resolve()
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise UserIntentCompilerError("USER_INTENT_COMPILER_OUTPUT_EXISTS")
    normalized_text = intent_text.strip()
    if not normalized_text:
        raise UserIntentCompilerError("USER_INTENT_TEXT_EMPTY")

    goal_source = Path(goal_config).resolve(strict=True)
    goal_bytes = goal_source.read_bytes()
    try:
        goal = load_yaml_document(goal_source)
        validate_document("goal_spec", goal, root=root)
    except (OSError, ContractValidationError) as exc:
        raise UserIntentCompilerError("USER_INTENT_GOAL_CONFIG_INVALID") from exc

    compiled = _compile_intent(normalized_text)
    active_goal = _summarize_active_goal(goal)
    binding, blockers = _bind_intent_to_goal(compiled=compiled, active_goal=active_goal)
    intent_binding_ready = not blockers
    state = "ready" if intent_binding_ready else _blocked_state(blockers)
    side_effects = {
        "goal_config_mutated": False,
        "protocol_changed": False,
        "constitution_changed": False,
        "registry_changed": False,
        "gpu_execution_started": False,
        "formal_m4_launch_permission_granted": False,
    }
    report = {
        "schema_version": 1,
        "artifact_type": "wmloop-user-intent-compilation",
        "state": state,
        "intent_binding_ready": intent_binding_ready,
        "goal_config": str(goal_source),
        "goal_id": str(goal["goal_id"]),
        "intent_sha256": hashlib.sha256(normalized_text.encode("utf-8")).hexdigest(),
        "compiled_intent": compiled,
        "active_goal_summary": active_goal,
        "binding": binding,
        "blockers": blockers,
        "side_effects": side_effects,
        "limitations": [
            "This packet is a control-plane binding, not an LLM proposal and not a trial record.",
            "Absent intent fields are filled from the active goal config only for routing; the active goal remains authoritative.",
            "If requested scope differs from the active goal, formal campaigns require a versioned protocol or goal-config change.",
            "The packet does not grant M4 launch permission and does not start training or evaluation.",
        ],
    }
    validate_document("user_intent_compilation", report, root=root)

    cas_storage_root = (
        Path(cas_root).resolve()
        if cas_root is not None
        else (Path(archive_db).resolve().parent if archive_db is not None else destination.parent)
    )
    cas = ContentAddressedStore(cas_storage_root)
    archive = ArchiveStore(archive_db) if archive_db is not None else None
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"

    try:
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary.mkdir(mode=0o700)
        inputs_dir = temporary / "inputs"
        inputs_dir.mkdir(mode=0o700)

        report_bytes = _canonical_json_bytes(report)
        markdown_bytes = _render_markdown(report).encode("utf-8")
        intent_bytes = normalized_text.encode("utf-8")
        refs = {
            "user_intent_compilation_json": cas.put_bytes(report_bytes, media_type="application/json").uri,
            "user_intent_compilation_markdown": cas.put_bytes(markdown_bytes, media_type="text/markdown").uri,
            "intent_text": cas.put_bytes(intent_bytes, media_type="text/plain").uri,
            "goal_config": cas.put_bytes(goal_bytes, media_type="application/yaml").uri,
        }
        if archive is not None:
            for ref in refs.values():
                archive.record_artifact_reference(ref)

        _write_bytes_atomic(temporary / "user-intent-compilation.json", report_bytes)
        _write_bytes_atomic(temporary / "user-intent-compilation.md", markdown_bytes)
        _write_bytes_atomic(inputs_dir / "intent.txt", intent_bytes)
        _write_bytes_atomic(inputs_dir / "goal-config.yaml", goal_bytes)

        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-user-intent-compilation-manifest",
            "state": state,
            "intent_binding_ready": intent_binding_ready,
            "goal_id": str(goal["goal_id"]),
            "goal_config": str(goal_source),
            "compiled_goal_family": compiled["goal_family"],
            "compiled_backbone_family": compiled["backbone_family"],
            "compiled_environment_scope": compiled["environment_scope"],
            "limited_execution_allowed": binding["limited_execution_allowed"],
            "formal_m4_launch_permission_granted": False,
            "report_path": str(destination / "user-intent-compilation.json"),
            "markdown_path": str(destination / "user-intent-compilation.md"),
            "input_snapshot_dir": str(destination / "inputs"),
            "cas_refs": refs,
            "blockers": blockers,
            "side_effects": side_effects,
            "limitations": report["limitations"],
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


def _compile_intent(text: str) -> dict[str, object]:
    lowered = text.lower()
    goal_family, goal_source = _requested_goal_family(text=lowered)
    backbone, backbone_source = _requested_backbone_family(text=lowered)
    scope, excluded_envs = _requested_environment_scope(text=lowered)
    requested_actions = _requested_actions(text=lowered)
    return {
        "intent_kind": "world_model_improvement_loop" if _has_any(text, ("闭环", "world model", "世界模型", "verdiwm")) else "unspecified",
        "goal_family": goal_family,
        "goal_family_source": goal_source,
        "backbone_family": backbone,
        "backbone_family_source": backbone_source,
        "environment_scope": scope,
        "excluded_environments": list(excluded_envs),
        "requested_actions": requested_actions,
        "requires_goal_spec_binding": True,
    }


def _requested_goal_family(*, text: str) -> tuple[str | None, str]:
    if _has_any(text, ("任务成功", "动作成功", "成功率", "action_success", "success rate", "wam-rl")):
        return "action_success", "user_text"
    if _has_any(text, ("物理一致", "physics_consistency", "physics consistency")):
        return "physics_consistency", "user_text"
    if _has_any(text, ("长程", "long horizon", "horizon", "psnr", "mmse", "ladder")):
        return "long_horizon_consistency", "user_text"
    return None, "active_goal_fallback"


def _requested_backbone_family(*, text: str) -> tuple[str | None, str]:
    if _has_any(text, ("wam", "policy", "robolab", "world action model")):
        return "wam", "user_text"
    if _has_any(text, ("acwm", "acwm-phys", "future observation", "future-observation")):
        return "acwm_phys", "user_text"
    return None, "active_goal_fallback"


def _requested_environment_scope(*, text: str) -> tuple[str | None, tuple[str, ...]]:
    cloth_excluded = _has_any(text, ("不等cloth", "不等 cloth", "去掉cloth", "去掉 cloth", "without cloth", "exclude cloth"))
    seven_env = _has_any(text, ("七环境", "7环境", "7 env", "seven env", "其他7"))
    eight_env = _has_any(text, ("八环境", "8环境", "8 env", "eight env"))
    if cloth_excluded or seven_env:
        return "seven_env_without_cloth_move", ("cloth_move",)
    if eight_env:
        return "all_acwm_phys_8_envs", ()
    return None, ()


def _requested_actions(*, text: str) -> list[str]:
    actions: list[str] = []
    if _has_any(text, ("继续", "推进", "跑出来", "闭环")):
        actions.append("continue_closed_loop_engineering")
    if _has_any(text, ("训练", "训", "gpu", "显存", "batch", "bs")):
        actions.append("use_available_gpu_for_guarded_training")
    if _has_any(text, ("m4", "正式")):
        actions.append("prepare_m4_when_gates_allow")
    return actions or ["bind_intent_to_active_goal"]


def _summarize_active_goal(goal: Mapping[str, Any]) -> dict[str, object]:
    envs = _strings(goal.get("envs"))
    horizons = [int(value) for value in goal.get("horizons", []) if isinstance(value, int)]
    protocol = goal.get("eval_protocol") if isinstance(goal.get("eval_protocol"), Mapping) else {}
    metric_family = _strings(goal.get("metric_family"))
    goal_family = _goal_family_from_goal(goal)
    return {
        "goal_id": str(goal.get("goal_id")),
        "goal_family": goal_family,
        "backbone_family": _backbone_from_envs(envs),
        "environment_scope": _scope_from_envs(envs),
        "envs": envs,
        "excluded_environments": [env for env in ACWM_PHYS_ENVS if env not in envs],
        "horizons": horizons,
        "metric_family": metric_family,
        "primary_objective": str(goal.get("primary_objective")),
        "eval_protocol_mode": str(protocol.get("mode", "uniform_horizons")) if isinstance(protocol, Mapping) else "uniform_horizons",
        "horizon_ladder_path": str(protocol.get("horizon_ladder_path", "")) if isinstance(protocol, Mapping) else "",
    }


def _bind_intent_to_goal(
    *,
    compiled: Mapping[str, object],
    active_goal: Mapping[str, object],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    blockers: list[dict[str, object]] = []
    requested_goal = compiled.get("goal_family")
    requested_backbone = compiled.get("backbone_family")
    requested_scope = compiled.get("environment_scope")
    active_goal_family = active_goal.get("goal_family")
    active_backbone = active_goal.get("backbone_family")
    active_scope = active_goal.get("environment_scope")

    goal_family = requested_goal or active_goal_family
    backbone = requested_backbone or active_backbone
    scope = requested_scope or active_scope

    if requested_goal is not None and requested_goal != active_goal_family:
        blockers.append(
            _blocker(
                "goal_family_mismatch",
                f"requested {requested_goal} but active goal is {active_goal_family}",
                "create_or_select_matching_goal_spec_before_formal_campaign",
            )
        )
    if requested_backbone is not None and requested_backbone != active_backbone:
        blockers.append(
            _blocker(
                "backbone_family_mismatch",
                f"requested {requested_backbone} but active goal infers {active_backbone}",
                "instantiate a matching constitution and goal_spec for the requested backbone",
            )
        )
    if requested_scope is not None and requested_scope != active_scope:
        blockers.append(
            _blocker(
                "environment_scope_mismatch",
                f"requested {requested_scope} but active goal is {active_scope}",
                "use explicit limited-campaign environment filters or apply a versioned protocol/goal change before formal claims",
            )
        )

    limited_allowed = (
        active_goal_family == "long_horizon_consistency"
        and active_backbone == "acwm_phys"
        and not any(item["code"] in {"goal_family_mismatch", "backbone_family_mismatch"} for item in blockers)
    )
    binding = {
        "bound_goal_family": goal_family,
        "bound_backbone_family": backbone,
        "bound_environment_scope": scope,
        "active_goal_matches_requested_goal_family": requested_goal in {None, active_goal_family},
        "active_goal_matches_requested_backbone": requested_backbone in {None, active_backbone},
        "active_goal_matches_requested_environment_scope": requested_scope in {None, active_scope},
        "limited_execution_allowed": limited_allowed,
        "formal_m4_launch_permission_granted": False,
        "requires_versioned_goal_or_protocol_change": any(item["code"] == "environment_scope_mismatch" for item in blockers),
    }
    return binding, blockers


def _goal_family_from_goal(goal: Mapping[str, Any]) -> str:
    primary = str(goal.get("primary_objective", "")).lower()
    metrics = {item.lower() for item in _strings(goal.get("metric_family"))}
    if "success" in primary or "success" in metrics:
        return "action_success"
    if "physics" in primary or "physics_consistency" in metrics:
        return "physics_consistency"
    if "psnr" in primary or {"psnr", "mmse"} & metrics:
        return "long_horizon_consistency"
    return "unknown"


def _backbone_from_envs(envs: Sequence[str]) -> str:
    env_set = set(envs)
    if env_set and env_set.issubset(set(ACWM_PHYS_ENVS)):
        return "acwm_phys"
    return "unknown"


def _scope_from_envs(envs: Sequence[str]) -> str:
    env_set = set(envs)
    if env_set == set(ACWM_PHYS_ENVS):
        return "all_acwm_phys_8_envs"
    if env_set == (set(ACWM_PHYS_ENVS) - {"cloth_move"}):
        return "seven_env_without_cloth_move"
    return "custom_environment_scope"


def _strings(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str)]


def _has_any(text: str, needles: Sequence[str]) -> bool:
    return any(needle.lower() in text for needle in needles)


def _blocker(code: str, detail: str, next_step: str) -> dict[str, object]:
    return {"code": code, "detail": detail, "next_step": next_step}


def _blocked_state(blockers: Sequence[Mapping[str, object]]) -> str:
    if any(blocker.get("code") in {"goal_family_mismatch", "backbone_family_mismatch"} for blocker in blockers):
        return "blocked"
    return "staged_blocked"


def _render_markdown(report: Mapping[str, object]) -> str:
    binding = report["binding"]
    lines = [
        "# User Intent Compilation",
        "",
        f"State: `{report['state']}`",
        f"Goal: `{report['goal_id']}`",
        f"Intent binding ready: `{report['intent_binding_ready']}`",
        f"Limited execution allowed: `{binding['limited_execution_allowed']}`",  # type: ignore[index]
        f"Formal M4 launch permission granted: `{binding['formal_m4_launch_permission_granted']}`",  # type: ignore[index]
        "",
        "## Binding",
        "",
    ]
    if isinstance(binding, Mapping):
        for key, value in binding.items():
            lines.append(f"- {key}: `{value}`")
    blockers = report.get("blockers")
    if blockers:
        lines.extend(["", "## Blockers", ""])
        for blocker in blockers:  # type: ignore[assignment]
            lines.append(f"- `{blocker}`")
    lines.extend(["", "## Limitations", ""])
    for limitation in report["limitations"]:  # type: ignore[index]
        lines.append(f"- {limitation}")
    return "\n".join(lines) + "\n"


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8") + b"\n"


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise UserIntentCompilerError(f"USER_INTENT_COMPILER_OUTPUT_EXISTS:{path}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def _intent_from_args(args: argparse.Namespace) -> str:
    parts: list[str] = []
    if args.intent_text:
        parts.append(str(args.intent_text))
    if args.intent_file:
        parts.append(Path(args.intent_file).resolve(strict=True).read_text(encoding="utf-8"))
    if not parts:
        raise UserIntentCompilerError("USER_INTENT_TEXT_OR_FILE_REQUIRED")
    return "\n".join(parts)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="compile user intent into an active-goal binding packet")
    run.add_argument("--repo-root", type=Path, default=Path("."))
    run.add_argument("--intent-text")
    run.add_argument("--intent-file", type=Path)
    run.add_argument("--goal-config", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--archive-db", type=Path)
    run.add_argument("--cas-root", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "run":
        manifest = run_user_intent_compilation(
            repo_root=args.repo_root,
            intent_text=_intent_from_args(args),
            goal_config=args.goal_config,
            output_root=args.output_root,
            archive_db=args.archive_db,
            cas_root=args.cas_root,
        )
        print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
