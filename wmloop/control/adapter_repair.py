"""Bounded LLM repair loop for model-interface adapter overlays.

The loop may generate disposable adapter and test files outside both the model
checkout and the VerdiWM source tree.  It cannot change the evaluator,
constitution, metrics, resource policy, or promotion rules inherited from a
trusted base profile.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.control.adapter_profiles import (
    AdapterProfileError,
    compile_adapter_execution,
)
from wmloop.execute.llm_task_adapter import LLMTaskAdapterError, run_llm_task


class AdapterRepairError(RuntimeError):
    """An adapter repair crossed a source, contract, or execution boundary."""


_PROMPT = """Repair only the target model's VerdiWM interface adapter.
Return a complete adapter overlay, not a patch to either source repository. The trusted
kernel supplies the evaluator and constitution. You may add adapter wrappers and tests,
but may not modify the model, dataset, evaluator, metrics, GPU policy, verifier, or
promotion policy. Runner entrypoints must live below {adapter_root}/adapter, accept the
single {model_run_manifest} argument, write the declared receipts below the manifest's
output root, and provide CPU-only self-tests. Use the supplied failure diagnostics to
replace the entire prior candidate on each attempt. Return unsupported when the missing
piece is a scientific evaluator, data regime, or model capability rather than an
interface implementation bug.
"""
_PROMPT_DIGEST = hashlib.sha256(_PROMPT.encode("utf-8")).hexdigest()
_SAFE_CHECK_MODULES = {"pytest", "unittest"}
_SAFE_FILE_ROOTS = {"adapter", "tests"}
_SENSITIVE_NAMES = re.compile(
    r"(?i)(?:^|[._-])(auth|credential|secret|token|key|password)(?:[._-]|$)"
)


def run_adapter_repair(
    *,
    model: Path,
    data: Path,
    goal: str,
    budget: object,
    failure_code: str,
    base_profile_path: Path,
    llm_adapter: Mapping[str, object],
    output_root: Path,
    project_root: Path,
    runtime_python: Path | None = None,
    max_attempts: int = 3,
) -> dict[str, object]:
    """Generate and validate an adapter overlay, retrying on bounded diagnostics."""

    root = Path(project_root).expanduser().resolve()
    model_root = _directory(model, "ADAPTER_REPAIR_MODEL_INVALID")
    data_root = _existing(data, "ADAPTER_REPAIR_DATA_INVALID")
    base_path = _file(base_profile_path, "ADAPTER_REPAIR_BASE_PROFILE_INVALID")
    try:
        base_profile = json.loads(base_path.read_text(encoding="utf-8"))
        if not isinstance(base_profile, dict):
            raise ValueError
        validate_document("adapter_profile", base_profile, root=root)
    except (OSError, ValueError, json.JSONDecodeError, ContractValidationError) as exc:
        raise AdapterRepairError("ADAPTER_REPAIR_BASE_PROFILE_INVALID") from exc
    if not isinstance(max_attempts, int) or isinstance(max_attempts, bool) or not 1 <= max_attempts <= 5:
        raise AdapterRepairError("ADAPTER_REPAIR_ATTEMPTS_INVALID")
    destination = Path(output_root).expanduser().resolve()
    _require_external_output(destination, model=model_root, project=root)
    inventory = _inventory(model_root)
    input_lock = {
        "model_root": str(model_root),
        "model_inventory_digest": _digest(inventory),
        "data_root": str(data_root),
        "base_profile_path": str(base_path),
        "base_profile_sha256": _sha256(base_path.read_bytes()),
        "goal": goal.strip(),
        "failure_code": failure_code,
        "max_attempts": max_attempts,
        "llm_adapter": _adapter_identity(llm_adapter),
    }
    if not input_lock["goal"]:
        raise AdapterRepairError("ADAPTER_REPAIR_GOAL_INVALID")
    input_digest = _digest(input_lock)
    resumed = _resume(destination, input_digest=input_digest)
    if resumed is not None:
        return resumed
    if destination.exists() or destination.is_symlink():
        raise AdapterRepairError("ADAPTER_REPAIR_OUTPUT_INVALID")
    destination.mkdir(mode=0o700, parents=True)
    _write_json(destination / "input-lock.json", {**input_lock, "input_digest": input_digest})

    diagnostics: list[dict[str, object]] = [{"code": failure_code}]
    attempts: list[dict[str, object]] = []
    selected_workspace: Path | None = None
    selected_profile: Path | None = None
    state = "blocked"
    blockers: list[dict[str, object]] = []
    for attempt in range(1, max_attempts + 1):
        request = _request(
            input_digest=input_digest,
            attempt=attempt,
            goal=goal,
            model_inventory=inventory,
            base_profile=base_profile,
            diagnostics=diagnostics,
        )
        transaction = destination / "llm" / f"attempt-{attempt:03d}"
        try:
            task = run_llm_task(
                request=request,
                adapter=llm_adapter,
                output_root=transaction,
                project_root=root,
            )
        except LLMTaskAdapterError as exc:
            blockers = [{"code": "ADAPTER_REPAIR_LLM_TRANSACTION_INVALID", "detail": str(exc)}]
            attempts.append({"attempt": attempt, "state": "blocked", "blockers": blockers})
            diagnostics = blockers
            continue
        if task.get("state") != "completed":
            blockers = [dict(row) for row in task.get("blockers", []) if isinstance(row, Mapping)]
            attempts.append({"attempt": attempt, "state": "blocked", "blockers": blockers})
            diagnostics = blockers or [{"code": "ADAPTER_REPAIR_LLM_BLOCKED"}]
            continue
        response = _load(Path(str(task["response_path"])), "ADAPTER_REPAIR_RESPONSE_INVALID")
        proposal = response.get("output")
        if not isinstance(proposal, Mapping):
            raise AdapterRepairError("ADAPTER_REPAIR_RESPONSE_INVALID")
        if proposal.get("state") != "candidate_ready":
            blockers = [dict(row) for row in proposal.get("blockers", []) if isinstance(row, Mapping)]
            blockers = blockers or [{"code": "ADAPTER_REPAIR_UNSUPPORTED", "detail": str(proposal.get("diagnosis") or "")[:500]}]
            attempts.append({"attempt": attempt, "state": "unsupported", "blockers": blockers})
            diagnostics = blockers
            break
        workspace = destination / "attempts" / f"attempt-{attempt:03d}" / "workspace"
        try:
            _materialize_candidate(workspace, proposal)
            profile_path = _write_profile(
                workspace=workspace,
                proposal=proposal,
                base_profile=base_profile,
                input_digest=input_digest,
                root=root,
            )
            check_rows = _run_checks(
                workspace=workspace,
                proposal=proposal,
                model=model_root,
                data=data_root,
            )
            failed = [row for row in check_rows if row["state"] != "pass"]
            if failed:
                blockers = [
                    {
                        "code": "ADAPTER_REPAIR_CHECK_FAILED",
                        "check": row["name"],
                        "returncode": row.get("returncode"),
                        "stderr_tail": row.get("stderr_tail"),
                    }
                    for row in failed
                ]
                attempts.append(
                    {"attempt": attempt, "state": "checks_failed", "checks": check_rows, "blockers": blockers}
                )
                diagnostics = blockers
                continue
            compile_adapter_execution(
                campaign_id="adapter-repair-conformance",
                model=model_root,
                data=data_root,
                goal=goal,
                budget=budget,
                campaign_root=destination / "conformance-state" / "campaigns",
                adapter_profile_path=profile_path,
                runtime_python=runtime_python,
                project_root=root,
            )
        except (AdapterRepairError, AdapterProfileError) as exc:
            blockers = [{"code": "ADAPTER_REPAIR_CONFORMANCE_FAILED", "detail": str(exc)[:500]}]
            attempts.append({"attempt": attempt, "state": "conformance_failed", "blockers": blockers})
            diagnostics = blockers
            continue
        attempts.append({"attempt": attempt, "state": "passed", "checks": check_rows, "blockers": []})
        selected_workspace = workspace
        selected_profile = profile_path
        state = "ready"
        blockers = []
        break

    manifest = {
        "schema_version": 1,
        "artifact_type": "verdiwm-adapter-repair-manifest",
        "state": state,
        "input_digest": input_digest,
        "attempt_count": len(attempts),
        "attempts": attempts,
        "adapter_profile_path": str(selected_profile) if selected_profile else None,
        "adapter_workspace": str(selected_workspace) if selected_workspace else None,
        "blockers": blockers,
        "authority": {
            "source_mutated": False,
            "evaluator_mutated": False,
            "metric_mutated": False,
            "gpu_scheduling": False,
            "promotion": False,
        },
        "assurance_level": "process_guarded_local",
        "claim_boundary": (
            "A ready result is an interface-conformant local adapter overlay only. "
            "Training evidence and promotion still require the bound evaluator and frozen verifier."
        ),
    }
    _write_json(destination / "manifest.json", manifest)
    return manifest


def _request(
    *,
    input_digest: str,
    attempt: int,
    goal: str,
    model_inventory: Mapping[str, object],
    base_profile: Mapping[str, object],
    diagnostics: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    task_id = f"adapter-repair-{input_digest[:20]}-{attempt}"
    trusted = {
        key: base_profile.get(key)
        for key in ("model_family", "evaluator_contract", "probe_contract", "constitution_freeze")
    }
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-llm-research-task",
        "task_id": task_id,
        "task_type": "module_generation",
        "prompt_template_digest": _PROMPT_DIGEST,
        "output_schema": "adapter_repair_proposal",
        "input": {
            "instructions": _PROMPT,
            "goal": goal,
            "model_inventory": dict(model_inventory),
            "trusted_scientific_bindings": trusted,
            "diagnostics": [dict(row) for row in diagnostics],
        },
    }


def _inventory(model: Path, *, max_files: int = 400, max_snippet_bytes: int = 4096) -> dict[str, object]:
    files: list[dict[str, object]] = []
    snippets: list[dict[str, str]] = []
    candidates = sorted(path for path in model.rglob("*") if path.is_file() and not path.is_symlink())
    for path in candidates[:max_files]:
        relative = path.relative_to(model).as_posix()
        if any(part.startswith(".") for part in PurePosixPath(relative).parts):
            continue
        if _SENSITIVE_NAMES.search(Path(relative).name):
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        files.append({"path": relative, "size": size})
        if _snippet_candidate(relative, size=size):
            try:
                raw = path.read_bytes()[:max_snippet_bytes]
                text = raw.decode("utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            snippets.append({"path": relative, "content": text})
    return {"files": files, "snippets": snippets, "truncated": len(candidates) > max_files}


def _snippet_candidate(relative: str, *, size: int) -> bool:
    path = PurePosixPath(relative)
    name = path.name.lower()
    if size > 512 * 1024:
        return False
    if name in {"readme.md", "pyproject.toml", "setup.py", "requirements.txt"}:
        return True
    return path.suffix == ".py" and any(part in {"scripts", "tools", "cli"} for part in path.parts[:-1])


def _materialize_candidate(workspace: Path, proposal: Mapping[str, object]) -> None:
    if workspace.exists() or workspace.is_symlink():
        raise AdapterRepairError("ADAPTER_REPAIR_WORKSPACE_EXISTS")
    workspace.mkdir(mode=0o700, parents=True)
    rows = proposal.get("files")
    if not isinstance(rows, list) or not rows:
        raise AdapterRepairError("ADAPTER_REPAIR_FILES_REQUIRED")
    total = 0
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise AdapterRepairError("ADAPTER_REPAIR_FILE_INVALID")
        relative = str(row.get("relative_path") or "")
        pure = PurePosixPath(relative)
        if (
            not relative
            or pure.is_absolute()
            or ".." in pure.parts
            or pure.parts[0] not in _SAFE_FILE_ROOTS
            or any(part.startswith(".") for part in pure.parts)
            or relative in seen
        ):
            raise AdapterRepairError("ADAPTER_REPAIR_FILE_PATH_INVALID")
        source = str(row.get("content_utf8") or "")
        encoded = source.encode("utf-8")
        total += len(encoded)
        if len(encoded) > 65536 or total > 262144:
            raise AdapterRepairError("ADAPTER_REPAIR_FILE_SIZE_INVALID")
        if pure.suffix == ".py":
            try:
                compile(source, relative, "exec")
            except SyntaxError as exc:
                raise AdapterRepairError("ADAPTER_REPAIR_PYTHON_SYNTAX_INVALID") from exc
        elif pure.suffix == ".json":
            try:
                json.loads(source)
            except json.JSONDecodeError as exc:
                raise AdapterRepairError("ADAPTER_REPAIR_JSON_INVALID") from exc
        target = workspace.joinpath(*pure.parts)
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        target.write_text(source, encoding="utf-8")
        seen.add(relative)


def _write_profile(
    *,
    workspace: Path,
    proposal: Mapping[str, object],
    base_profile: Mapping[str, object],
    input_digest: str,
    root: Path,
) -> Path:
    raw = proposal.get("profile")
    if not isinstance(raw, Mapping):
        raise AdapterRepairError("ADAPTER_REPAIR_PROFILE_REQUIRED")
    runner = raw.get("runner")
    if not isinstance(runner, Mapping):
        raise AdapterRepairError("ADAPTER_REPAIR_RUNNER_INVALID")
    normalized_runner = {
        "protocol_version": 1,
        "requires_training": True,
        **dict(runner),
    }
    files = {
        path.relative_to(workspace).as_posix()
        for path in workspace.rglob("*")
        if path.is_file()
    }
    for field in ("train", "evaluate"):
        command = normalized_runner.get(field)
        if not isinstance(command, list) or command.count("{model_run_manifest}") != 1:
            raise AdapterRepairError("ADAPTER_REPAIR_RUNNER_INVALID")
        entrypoint = str(command[0]) if command else ""
        prefix = "{adapter_root}/"
        if not entrypoint.startswith(prefix) or entrypoint[len(prefix) :] not in files:
            raise AdapterRepairError("ADAPTER_REPAIR_RUNNER_ENTRYPOINT_INVALID")
    profile = {
        "schema_version": 1,
        "artifact_type": "verdiwm-adapter-profile",
        "profile_id": f"generated-adapter-{input_digest[:24]}",
        "aliases": [f"generated-{input_digest[:12]}"],
        "model_family": raw["model_family"],
        "capability_level": "L2",
        "execution_kind": "pipeline",
        "repo_markers": list(raw["repo_markers"]),
        "goal_keywords": list(raw["goal_keywords"]),
        "evaluator_contract": base_profile["evaluator_contract"],
        "probe_contract": base_profile.get("probe_contract"),
        "constitution_freeze": base_profile["constitution_freeze"],
        "runtime_candidates": list(raw["runtime_candidates"]),
        "asset_bindings": [dict(row) for row in raw["asset_bindings"]],
        "runner": normalized_runner,
        "probe_imports": bool(raw["probe_imports"]),
    }
    for field in ("candidate_catalog", "settlement_manifest_candidates"):
        if base_profile.get(field) is not None:
            profile[field] = base_profile[field]
    try:
        validate_document("adapter_profile", profile, root=root)
    except ContractValidationError as exc:
        raise AdapterRepairError("ADAPTER_REPAIR_PROFILE_INVALID") from exc
    path = workspace / "adapter-profile.json"
    _write_json(path, profile)
    return path


def _run_checks(
    *,
    workspace: Path,
    proposal: Mapping[str, object],
    model: Path,
    data: Path,
) -> list[dict[str, object]]:
    rows = proposal.get("checks")
    if not isinstance(rows, list) or not rows:
        raise AdapterRepairError("ADAPTER_REPAIR_CHECKS_REQUIRED")
    results: list[dict[str, object]] = []
    environment = {
        "HOME": str(workspace / ".home"),
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "VERDIWM_TARGET_MODEL": str(model),
        "VERDIWM_TARGET_DATA": str(data),
    }
    Path(environment["HOME"]).mkdir(mode=0o700)
    for row in rows:
        if not isinstance(row, Mapping):
            raise AdapterRepairError("ADAPTER_REPAIR_CHECK_INVALID")
        raw = row.get("command")
        if not isinstance(raw, list) or len(raw) < 3:
            raise AdapterRepairError("ADAPTER_REPAIR_CHECK_INVALID")
        command = [str(value) for value in raw]
        if command[0] not in {"python", "{python}"} or command[1] != "-m" or command[2] not in _SAFE_CHECK_MODULES:
            raise AdapterRepairError("ADAPTER_REPAIR_CHECK_COMMAND_FORBIDDEN")
        if any(re.search(r"[;&|`\n\r]", token) for token in command):
            raise AdapterRepairError("ADAPTER_REPAIR_CHECK_COMMAND_FORBIDDEN")
        command[0] = sys.executable
        try:
            completed = subprocess.run(
                command,
                cwd=workspace,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
                timeout=60,
            )
            state = "pass" if completed.returncode == 0 else "fail"
            results.append(
                {
                    "name": str(row.get("name") or "check"),
                    "state": state,
                    "returncode": completed.returncode,
                    "stdout_tail": completed.stdout[-1000:],
                    "stderr_tail": completed.stderr[-1000:],
                }
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            results.append(
                {
                    "name": str(row.get("name") or "check"),
                    "state": "fail",
                    "returncode": None,
                    "stdout_tail": "",
                    "stderr_tail": type(exc).__name__,
                }
            )
    return results


def _adapter_identity(adapter: Mapping[str, object]) -> dict[str, object]:
    return {
        "command": [str(value) for value in adapter.get("command", [])] if isinstance(adapter.get("command"), list) else None,
        "provider_alias": adapter.get("provider_alias"),
        "model_alias": adapter.get("model_alias"),
        "timeout_seconds": adapter.get("timeout_seconds"),
        "max_output_bytes": adapter.get("max_output_bytes"),
    }


def _require_external_output(path: Path, *, model: Path, project: Path) -> None:
    resolved = path.resolve()
    if _within(resolved, model) or _within(resolved, project):
        raise AdapterRepairError("ADAPTER_REPAIR_OUTPUT_OVERLAP")


def _resume(destination: Path, *, input_digest: str) -> dict[str, object] | None:
    if not destination.exists() and not destination.is_symlink():
        return None
    if destination.is_symlink() or not destination.is_dir():
        raise AdapterRepairError("ADAPTER_REPAIR_OUTPUT_INVALID")
    lock = _load(destination / "input-lock.json", "ADAPTER_REPAIR_INPUT_LOCK_INVALID")
    if lock.get("input_digest") != input_digest:
        raise AdapterRepairError("ADAPTER_REPAIR_INPUT_LOCK_MISMATCH")
    return _load(destination / "manifest.json", "ADAPTER_REPAIR_MANIFEST_INVALID")


def _directory(path: Path, code: str) -> Path:
    source = Path(path).expanduser()
    if source.is_symlink() or not source.is_dir():
        raise AdapterRepairError(code)
    return source.resolve()


def _existing(path: Path, code: str) -> Path:
    source = Path(path).expanduser()
    if source.is_symlink() or not source.exists():
        raise AdapterRepairError(code)
    return source.resolve()


def _file(path: Path, code: str) -> Path:
    source = Path(path).expanduser()
    if source.is_symlink() or not source.is_file():
        raise AdapterRepairError(code)
    return source.resolve()


def _load(path: Path, code: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise AdapterRepairError(code)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AdapterRepairError(code) from exc
    if not isinstance(value, dict):
        raise AdapterRepairError(code)
    return value


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    body = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":")) + "\n"
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _digest(value: object) -> str:
    return _sha256(json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8"))


def _within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False
