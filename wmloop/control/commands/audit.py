"""CLI commands and configuration boundaries: audit."""

from __future__ import annotations

import argparse
from pathlib import Path

from wmloop.control.campaign_dispatcher import CampaignDispatchError
from wmloop.control.commands.io import emit_json as _print, repository_root
from wmloop.evaluate.system_utility import (
    SystemUtilityAuditError,
    run_system_utility_audit,
)


def _import_settlements(args: argparse.Namespace) -> int:
    from wmloop.experiments.ctrl_world_settlement_import import (
        CtrlWorldSettlementImportError,
        import_ctrl_world_settlements,
    )

    try:
        manifest = import_ctrl_world_settlements(
            input_root=args.input_root,
            archive_db=args.archive_db,
            cas_root=args.cas_root,
            output_root=args.output_root,
            allowed_roots=tuple(args.allowed_root),
            goal_id=args.goal_id,
            dry_run=args.dry_run,
        )
    except CtrlWorldSettlementImportError as exc:
        raise CampaignDispatchError(str(exc)) from exc
    _print(manifest)
    return 0


def _audit(args: argparse.Namespace) -> int:
    if getattr(args, "audit_command", None) == "local-smoke":
        from wmloop.evaluate.local_maturity_audit import (
            LocalMaturityAuditError,
            run_local_maturity_audit,
        )

        try:
            report = run_local_maturity_audit(
                repo_root=args.repo_root or repository_root(), output_root=args.output_root
            )
        except LocalMaturityAuditError as exc:
            raise SystemUtilityAuditError(str(exc)) from exc
        except OSError as exc:
            raise SystemUtilityAuditError(f"LOCAL_AUDIT_IO_ERROR:{exc.strerror}") from exc
        _print(report)
        return 0 if report["state"] == "ready" else 3
    root = (args.repo_root or Path.cwd()).expanduser().resolve()
    config = (
        args.config or root / "configs" / "experiments" / "system_utility_audit_v1.json"
    )
    manifest = run_system_utility_audit(
        config_path=config,
        repo_root=root,
        output_root=args.output_root,
    )
    _print(manifest)
    return (
        3
        if args.require_effect and manifest["research_effect_state"] != "established"
        else 0
    )
