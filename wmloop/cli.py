"""User-facing VerdiWM campaign commands."""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Sequence

from wmloop.control.campaign_api import CampaignAPIError, CampaignStore
from wmloop.control.campaign_dispatcher import (
    CampaignDispatchError,
    DispatcherOptions,
    run_dispatcher,
)


def _default_state_root() -> Path:
    configured = os.environ.get("VERDIWM_STATE_ROOT")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".local" / "state" / "verdiwm"


def _asset(value: str) -> tuple[str, str]:
    parameter, separator, path = value.partition("=")
    if not separator or not parameter.strip() or not path.strip():
        raise argparse.ArgumentTypeError("asset must be PARAM=PATH")
    normalized = parameter.strip()
    if not normalized.startswith("--"):
        normalized = f"--{normalized}"
    return normalized, path.strip()


def _store(state_root: Path) -> CampaignStore:
    return CampaignStore(Path(state_root).expanduser().resolve() / "campaigns")


def _dispatch(
    store: CampaignStore, *, campaign_id: str, max_parallel: int
) -> dict[str, Any]:
    return run_dispatcher(
        DispatcherOptions(
            state_root=store.root,
            max_cycles=1,
            max_parallel=max_parallel,
            campaign_ids=(campaign_id,),
        )
    )


def _print(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _doctor(args: argparse.Namespace) -> int:
    root = (args.repo_root or Path(__file__).resolve().parents[1]).expanduser().resolve()
    checks: list[dict[str, object]] = []

    python_supported = sys.version_info[:2] == (3, 10)
    checks.append(
        {
            "name": "python_3_10",
            "required": True,
            "state": "pass" if python_supported else "fail",
            "detail": platform.python_version(),
        }
    )
    required_files = (
        ("core_package", "wmloop/__init__.py", None),
        ("goal_schema", "configs/schemas/goal_spec.schema.json", None),
        (
            "adapter_profile",
            "configs/adapters/ctrl_world_predictive_v2.json",
            "verdiwm-adapter-profile",
        ),
        (
            "mechanism_ontology",
            "configs/retrieval/mechanism_tag_ontology_v1.json",
            "wmloop-mechanism-tag-ontology",
        ),
    )
    for name, relative, expected_type in required_files:
        path = root / relative
        state = "pass"
        detail = relative
        if not path.is_file() or path.is_symlink():
            state = "fail"
            detail = f"missing:{relative}"
        elif path.suffix == ".json":
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                state = "fail"
                detail = f"invalid_json:{relative}"
            else:
                if not isinstance(payload, dict) or (
                    expected_type is not None
                    and payload.get("artifact_type") != expected_type
                ):
                    state = "fail"
                    detail = f"invalid_contract:{relative}"
        checks.append(
            {"name": name, "required": True, "state": state, "detail": detail}
        )

    public_example = root / "examples" / "acwm_minimal_loop_cloth_next_forcing_v2"
    checks.append(
        {
            "name": "public_example",
            "required": False,
            "state": "available" if public_example.is_dir() else "not_installed",
            "detail": "examples/acwm_minimal_loop_cloth_next_forcing_v2",
        }
    )
    blocked = any(
        row["required"] is True and row["state"] != "pass" for row in checks
    )
    try:
        package_version = version("verdiwm")
    except PackageNotFoundError:
        package_version = "source"
    _print(
        {
            "schema_version": 1,
            "artifact_type": "verdiwm-doctor-report",
            "state": "blocked" if blocked else "ready",
            "version": package_version,
            "checks": checks,
        }
    )
    return 2 if blocked else 0


def _run(args: argparse.Namespace) -> int:
    store = _store(args.state_root)
    payload: dict[str, Any] = {
        "goal": args.goal,
        "model": args.model,
        "dataset": args.data,
        "budget": args.budget,
        "adapter": args.adapter,
    }
    if args.campaign_id:
        payload["campaign_id"] = args.campaign_id
    if args.adapter_profile:
        payload["adapter_profile_path"] = str(args.adapter_profile)
    if args.runtime_python:
        payload["runtime_python"] = str(args.runtime_python)
    if args.asset:
        payload["assets"] = dict(args.asset)
    created = store.create(payload)
    queued = store.confirm(str(created["campaign_id"]))
    if args.queue_only:
        _print(queued)
        return 0
    dispatcher = _dispatch(
        store,
        campaign_id=str(created["campaign_id"]),
        max_parallel=args.max_parallel,
    )
    campaign = store.get(str(created["campaign_id"]))
    _print({"campaign": campaign, "dispatcher": dispatcher})
    return 0 if campaign.get("status") in {"completed", "cancelled"} else 2


def _status(args: argparse.Namespace) -> int:
    store = _store(args.state_root)
    if args.campaign_id:
        _print(store.get(args.campaign_id))
    else:
        _print(
            {
                "schema_version": 2,
                "artifact_type": "verdiwm-campaign-list",
                "items": store.list(status=args.status, limit=args.limit),
            }
        )
    return 0


def _cancel(args: argparse.Namespace) -> int:
    _print(_store(args.state_root).cancel(args.campaign_id))
    return 0


def _reproduce(args: argparse.Namespace) -> int:
    store = _store(args.state_root)
    campaign = store.reproduce(args.campaign_id)
    if args.queue_only:
        _print(campaign)
        return 0
    dispatcher = _dispatch(
        store,
        campaign_id=str(campaign["campaign_id"]),
        max_parallel=args.max_parallel,
    )
    reproduced = store.get(str(campaign["campaign_id"]))
    _print({"campaign": reproduced, "dispatcher": dispatcher})
    return 0 if reproduced.get("status") in {"completed", "cancelled"} else 2


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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="verdiwm", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    doctor = commands.add_parser(
        "doctor", help="verify the local CPU control-plane installation"
    )
    doctor.add_argument("--repo-root", type=Path)
    doctor.set_defaults(handler=_doctor)

    run = commands.add_parser("run", help="compile and run a model optimization campaign")
    run.add_argument("--model", required=True)
    run.add_argument("--data", required=True)
    run.add_argument("--goal", required=True)
    run.add_argument("--budget", required=True)
    run.add_argument("--adapter", default="auto")
    run.add_argument("--adapter-profile", type=Path)
    run.add_argument("--runtime-python", type=Path)
    run.add_argument("--asset", action="append", type=_asset, default=[])
    run.add_argument("--campaign-id")
    run.add_argument("--state-root", type=Path, default=_default_state_root())
    run.add_argument("--queue-only", action="store_true")
    run.add_argument("--max-parallel", type=int, default=1)
    run.set_defaults(handler=_run)

    status = commands.add_parser("status", help="show one campaign or list campaigns")
    status.add_argument("campaign_id", nargs="?")
    status.add_argument("--status")
    status.add_argument("--limit", type=int, default=100)
    status.add_argument("--state-root", type=Path, default=_default_state_root())
    status.set_defaults(handler=_status)

    cancel = commands.add_parser("cancel", help="cancel a queued or running campaign")
    cancel.add_argument("campaign_id")
    cancel.add_argument("--state-root", type=Path, default=_default_state_root())
    cancel.set_defaults(handler=_cancel)

    reproduce = commands.add_parser(
        "reproduce", help="create and dispatch an isolated reproduction campaign"
    )
    reproduce.add_argument("campaign_id")
    reproduce.add_argument("--state-root", type=Path, default=_default_state_root())
    reproduce.add_argument("--queue-only", action="store_true")
    reproduce.add_argument("--max-parallel", type=int, default=1)
    reproduce.set_defaults(handler=_reproduce)

    settlement_import = commands.add_parser(
        "import-settlements",
        help="import terminal Ctrl-World mechanism evidence into Archive",
    )
    settlement_import.add_argument("--input-root", type=Path, required=True)
    settlement_import.add_argument("--archive-db", type=Path, required=True)
    settlement_import.add_argument("--cas-root", type=Path, required=True)
    settlement_import.add_argument("--output-root", type=Path, required=True)
    settlement_import.add_argument(
        "--allowed-root", type=Path, action="append", default=[]
    )
    settlement_import.add_argument(
        "--goal-id", default="ctrl_world_predictive_quality_pilot_v2"
    )
    settlement_import.add_argument("--dry-run", action="store_true")
    settlement_import.set_defaults(handler=_import_settlements)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (CampaignAPIError, CampaignDispatchError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
