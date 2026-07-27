"""Manual M2 primitive smoke runner.

The runner exercises the execution path without launching a full training job:
create a detached provider worktree, render registered primitive templates,
apply their diffs, run receipt-backed checks, seal the patch, and remove the
worktree.  It is a lower-bound executor proof, not a model-quality claim.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from wmloop.execute.agent_staging import AgentRepairSession
from wmloop.execute.sandbox import SandboxLease, WorktreeSandbox
from wmloop.primitives.hooks import TensorSpec
from wmloop.primitives.registry import PrimitiveRegistry
from wmloop.primitives.render import PrimitiveRenderer, RenderedPrimitive
from wmloop.vendor import verify_vendor_checkout


class PrimitiveSmokeError(RuntimeError):
    """The manual primitive smoke run could not produce a trustworthy receipt."""


def run_primitive_smoke(
    *,
    trial_id: str,
    interventions: Sequence[Mapping[str, Any]],
    runs_root: Path,
    repo_root: Path,
    keep_worktree: bool = False,
    timeout_seconds: float = 30.0,
) -> dict[str, object]:
    root = Path(repo_root).resolve()
    vendor_root = root / "vendor" / "ACWM-Phys"
    source_revision = verify_vendor_checkout(root)
    registry = PrimitiveRegistry.from_root(root)
    sandbox = WorktreeSandbox(vendor_root=vendor_root, runs_root=runs_root)
    trial_root = Path(runs_root).resolve() / trial_id
    receipt_path = trial_root / "primitive-smoke-receipt.json"
    if receipt_path.exists() or receipt_path.is_symlink():
        raise PrimitiveSmokeError("PRIMITIVE_SMOKE_RECEIPT_EXISTS")
    lease: SandboxLease | None = None
    worktree_removed = False
    try:
        lease = sandbox.create(trial_id=trial_id, expected_revision=source_revision)
        renderer = PrimitiveRenderer(registry)
        rendered = _render_and_apply_sequentially(
            renderer=renderer,
            registry=registry,
            worktree=lease.worktree,
            interventions=interventions,
            hook_ios=_default_hook_ios(),
        )
        session = AgentRepairSession(
            worktree=lease.worktree,
            staging_root=trial_root / "staging",
            candidate_id="primitive-smoke",
            source_revision=source_revision,
            registry_digest=registry.digest(),
            required_check_labels=("git_status", "sidecar_json"),
        )
        session.run(label="git_status", argv=("git", "status", "--short"), timeout_seconds=timeout_seconds)
        session.run(
            label="sidecar_json",
            argv=(sys.executable, "-c", _sidecar_check_script([item.name for item in rendered])),
            timeout_seconds=timeout_seconds,
        )
        candidate = session.seal()
        if not keep_worktree and lease is not None:
            sandbox.remove(lease)
            worktree_removed = True
        result = {
            "schema_version": 1,
            "artifact_type": "wmloop-manual-primitive-smoke-receipt",
            "state": "ready" if candidate.ready_for_promotion else "checks_failed",
            "trial_id": trial_id,
            "source_revision": source_revision,
            "registry_digest": registry.digest(),
            "worktree": str(lease.worktree),
            "worktree_removed": worktree_removed,
            "keep_worktree": keep_worktree,
            "rendered_primitives": [_rendered_document(item) for item in rendered],
            "candidate": candidate.to_document(),
            "candidate_manifest_path": str(candidate.manifest_path),
            "candidate_diff_path": str(candidate.diff_path),
        }
        _write_json_atomic(receipt_path, result)
        return result
    except Exception:
        if lease is not None and not keep_worktree:
            try:
                sandbox.remove(lease)
            except Exception:
                pass
        raise


def _default_hook_ios() -> dict[str, tuple[dict[str, object], dict[str, object]]]:
    return {
        "H1": (
            {"trajectory": TensorSpec((1, 16, 3, 64, 64), "float32")},
            {"trajectory": TensorSpec((1, 16, 3, 64, 64), "float32")},
        ),
        "H2": (
            {"latent_ctx": TensorSpec((1, 16, 1024), "float16"), "action_emb": TensorSpec((1, 16, 1024), "float16")},
            {"ctx": TensorSpec((1, 16, 1024), "float16")},
        ),
        "H3": (
            {"pred": TensorSpec((1, 16, 4, 32, 32), "float16"), "target": TensorSpec((1, 16, 4, 32, 32), "float16")},
            {"loss": TensorSpec((1,), "float32")},
        ),
        "H4": (
            {"tokens": TensorSpec((1, 16, 1024), "float16")},
            {"tokens": TensorSpec((1, 16, 1024), "float16")},
        ),
        "H5": ({}, {}),
    }


def _apply_diff(worktree: Path, diff: str) -> None:
    completed = subprocess.run(
        ["git", "-C", str(worktree), "apply", "--whitespace=error-all", "-"],
        input=diff,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise PrimitiveSmokeError("PRIMITIVE_SMOKE_GIT_APPLY_FAILED")


def _render_and_apply_sequentially(
    *,
    renderer: PrimitiveRenderer,
    registry: PrimitiveRegistry,
    worktree: Path,
    interventions: Sequence[Mapping[str, Any]],
    hook_ios: Mapping[str, tuple[Mapping[str, object], Mapping[str, object]]],
) -> tuple[RenderedPrimitive, ...]:
    """Render against the current staged worktree so shared hook files compose."""

    selections = []
    for item in interventions:
        primitive = item.get("primitive")
        params = item.get("params")
        if not isinstance(primitive, str) or not isinstance(params, Mapping):
            raise PrimitiveSmokeError("PRIMITIVE_SMOKE_INTERVENTION_INVALID")
        selections.append((primitive, params))
    registry.validate_combination(selections)
    rendered: list[RenderedPrimitive] = []
    for item in interventions:
        current = renderer.render_checked(worktree=worktree, interventions=(item,), hook_ios=hook_ios)
        for rendered_item in current:
            _apply_diff(worktree, rendered_item.diff)
            rendered.append(rendered_item)
    return tuple(rendered)


def _sidecar_check_script(names: Sequence[str]) -> str:
    return (
        "import json, pathlib; "
        f"names={list(names)!r}; "
        "root=pathlib.Path('wmloop_interventions'); "
        "[json.loads((root / (name + '.json')).read_text()) for name in names]"
    )


def _rendered_document(item: RenderedPrimitive) -> dict[str, object]:
    return {"name": item.name, "diff_sha256": item.sha256}


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def _load_interventions(path: Path) -> list[Mapping[str, Any]]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PrimitiveSmokeError("PRIMITIVE_SMOKE_INTERVENTIONS_INVALID") from exc
    interventions = payload.get("interventions") if isinstance(payload, Mapping) else payload
    if not isinstance(interventions, list) or not interventions:
        raise PrimitiveSmokeError("PRIMITIVE_SMOKE_INTERVENTIONS_INVALID")
    if any(not isinstance(item, Mapping) for item in interventions):
        raise PrimitiveSmokeError("PRIMITIVE_SMOKE_INTERVENTIONS_INVALID")
    return interventions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run one manual primitive smoke trial")
    run.add_argument("--trial-id", required=True)
    run.add_argument("--interventions-json", type=Path, required=True)
    run.add_argument("--runs-root", type=Path, required=True)
    run.add_argument("--repo-root", type=Path, default=Path("."))
    run.add_argument("--keep-worktree", action="store_true")
    run.add_argument("--timeout-seconds", type=float, default=30.0)
    args = parser.parse_args(argv)
    if args.command == "run":
        result = run_primitive_smoke(
            trial_id=args.trial_id,
            interventions=_load_interventions(args.interventions_json),
            runs_root=args.runs_root,
            repo_root=args.repo_root,
            keep_worktree=args.keep_worktree,
            timeout_seconds=args.timeout_seconds,
        )
        print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
