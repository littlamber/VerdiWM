"""CPU-safe conformance checks for a generated model-onboarding sidecar."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.control.onboarding import (
    OnboardingError,
    compute_asset_fingerprint,
    compute_source_revision,
    compute_source_tree_revision,
)
from wmloop.control.onboarding_admission import (
    OnboardingAdmissionError,
    verify_receipt_asset_bindings,
)
from wmloop.control.intermediate_ir import (
    IntermediateRepresentationError,
    ir_digest,
    validate_model_capability_ir,
)


_MAX_LOG_BYTES = 1_000_000
_PLACEHOLDER = re.compile(r"\{(?:python|repo_root|asset:--[A-Za-z0-9_-]+)\}")


class ModelConformanceError(RuntimeError):
    """A conformance request or its immutable inputs are invalid."""


@dataclass(frozen=True)
class ConformanceOptions:
    """Explicit inputs for one CPU-only conformance transaction."""

    sidecar_root: Path
    output_root: Path
    timeout_seconds: float = 30.0


def run_conformance(options: ConformanceOptions) -> dict[str, object]:
    """Run or resume conformance and return its durable manifest."""

    if options.timeout_seconds <= 0 or options.timeout_seconds > 300:
        raise ModelConformanceError("CONFORMANCE_TIMEOUT_INVALID")
    sidecar = Path(options.sidecar_root).expanduser().resolve()
    if not sidecar.is_dir() or sidecar.is_symlink():
        raise ModelConformanceError("CONFORMANCE_SIDECAR_INVALID")
    sidecar_manifest = _load_json(
        sidecar / "manifest.json", "CONFORMANCE_SIDECAR_MANIFEST_INVALID"
    )
    report_path = sidecar / "onboarding-report.json"
    report_bytes = (
        report_path.read_bytes()
        if report_path.is_file() and not report_path.is_symlink()
        else b""
    )
    if hashlib.sha256(report_bytes).hexdigest() != sidecar_manifest.get(
        "report_sha256"
    ):
        raise ModelConformanceError("CONFORMANCE_ONBOARDING_REPORT_HASH_MISMATCH")
    capability_path = sidecar / "model-capability-ir.json"
    capability_bytes = (
        capability_path.read_bytes()
        if capability_path.is_file() and not capability_path.is_symlink()
        else b""
    )
    if hashlib.sha256(capability_bytes).hexdigest() != sidecar_manifest.get(
        "model_capability_ir_sha256"
    ):
        raise ModelConformanceError("CONFORMANCE_MODEL_CAPABILITY_IR_HASH_MISMATCH")
    try:
        capability_ir = json.loads(capability_bytes)
        if not isinstance(capability_ir, dict):
            raise ValueError("capability IR object required")
        validate_model_capability_ir(capability_ir)
    except (
        json.JSONDecodeError,
        ValueError,
        IntermediateRepresentationError,
    ) as exc:
        raise ModelConformanceError("CONFORMANCE_MODEL_CAPABILITY_IR_INVALID") from exc
    capability_semantic_digest = ir_digest(capability_ir)
    if capability_semantic_digest != sidecar_manifest.get(
        "model_capability_semantic_digest"
    ):
        raise ModelConformanceError(
            "CONFORMANCE_MODEL_CAPABILITY_IR_SEMANTIC_DIGEST_MISMATCH"
        )
    report = _load_json(report_path, "CONFORMANCE_ONBOARDING_REPORT_INVALID")
    try:
        validate_document("model_onboarding_report", report)
    except ContractValidationError as exc:
        raise ModelConformanceError("CONFORMANCE_ONBOARDING_REPORT_INVALID") from exc

    repo = Path(str(report["repo_root"])).resolve()
    if not repo.is_dir() or repo.is_symlink():
        raise ModelConformanceError("CONFORMANCE_SOURCE_REPOSITORY_INVALID")
    destination = Path(options.output_root).expanduser().resolve()
    _validate_output_root(destination, repo=repo, sidecar=sidecar)
    input_hash = _input_hash(report_bytes, capability_bytes, options.timeout_seconds)
    if destination.exists() or destination.is_symlink():
        return _resume_conformance(
            destination,
            input_hash=input_hash,
            repo=repo,
            expected_revision=report["source_revision"],
        )

    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=str(destination.parent))
    )
    try:
        _write_json(
            temporary / "input.lock.json",
            {
                "schema_version": 1,
                "artifact_type": "wmloop-model-conformance-input-lock",
                "input_hash": input_hash,
                "onboarding_report_sha256": sidecar_manifest["report_sha256"],
                "model_capability_ir_sha256": sidecar_manifest[
                    "model_capability_ir_sha256"
                ],
                "model_capability_semantic_digest": capability_semantic_digest,
            },
        )
        source_before = compute_source_revision(repo)
        integrity_before = compute_source_tree_revision(repo)
        assets_before = _asset_binding_snapshot(report)
        checks: list[dict[str, object]] = []
        checks.append(_source_check(source_before, report["source_revision"]))
        checks.append(_admission_check(report))
        checks.append(_asset_binding_check(assets_before))
        checks.append(_output_isolation_check(temporary, repo=repo, sidecar=sidecar))

        runtime = Path(str(report["runtime"].get("selected_python") or ""))
        evaluator = report["evaluator_contract"]
        command: list[str] | None = None
        materialization_error: str | None = None
        try:
            command = _materialize_command(report, runtime=runtime, repo=repo)
            checks.append(
                _pass_check("evaluator_contract", "evaluator command materialized")
            )
        except ModelConformanceError as exc:
            materialization_error = str(exc)
            checks.append(_fail_check("evaluator_contract", materialization_error))

        runtime_ready = runtime.is_file() and os.access(runtime, os.X_OK)
        if not runtime_ready:
            checks.append(
                _fail_check("runtime_executable", "CONFORMANCE_RUNTIME_MISSING")
            )
        else:
            checks.append(_pass_check("runtime_executable", str(runtime)))

        process_started = False
        if runtime_ready and isinstance(evaluator, Mapping):
            imports = evaluator.get("conformance_imports", [])
            if isinstance(imports, list):
                for index, module in enumerate(imports):
                    process_started = True
                    checks.append(
                        _run_check(
                            name=f"module_import_{index:02d}",
                            command=[
                                str(runtime),
                                "-c",
                                f"import importlib; importlib.import_module({str(module)!r})",
                            ],
                            repo=repo,
                            output_root=temporary,
                            timeout_seconds=options.timeout_seconds,
                        )
                    )
        entrypoint_probe = (
            str(evaluator.get("entrypoint_probe", "help"))
            if isinstance(evaluator, Mapping)
            else "help"
        )
        if runtime_ready and command is not None and materialization_error is None:
            if entrypoint_probe == "skip":
                checks.append(
                    _pass_check(
                        "entrypoint_probe",
                        "entrypoint probe explicitly deferred to bounded runtime smoke",
                    )
                )
            else:
                process_started = True
                checks.append(
                    _run_check(
                        name="entrypoint_help",
                        command=[*command, "--help"],
                        repo=repo,
                        output_root=temporary,
                        timeout_seconds=options.timeout_seconds,
                    )
                )

        source_after = compute_source_revision(repo)
        integrity_after = compute_source_tree_revision(repo)
        assets_after = _asset_binding_snapshot(report)
        checks.append(_source_integrity_check(integrity_before, integrity_after))
        checks.append(_asset_integrity_check(assets_before, assets_after))
        passed = all(check["status"] == "pass" for check in checks)
        receipt = {
            "schema_version": 1,
            "artifact_type": "wmloop-model-conformance-receipt",
            "state": "settled",
            "verdict": "PASS" if passed else "BLOCKED",
            "optimization_launch_allowed": passed,
            "repo_root": str(repo),
            "sidecar_root": str(sidecar),
            "onboarding_report_sha256": sidecar_manifest["report_sha256"],
            "model_capability_ir_sha256": sidecar_manifest[
                "model_capability_ir_sha256"
            ],
            "model_capability_semantic_digest": capability_semantic_digest,
            "source_revision": source_after,
            "source_tree_revision": integrity_after,
            "asset_bindings": assets_after,
            "runtime_python": str(runtime) if runtime_ready else None,
            "evaluator_contract_sha256": (
                evaluator.get("contract_sha256")
                if isinstance(evaluator, Mapping)
                else None
            ),
            "checks": checks,
            "side_effects": {
                "source_modified": integrity_before != integrity_after,
                "dependency_install_started": False,
                "model_import_executed": process_started,
                "gpu_execution_started": False,
            },
            "claim_boundary": "Conformance PASS authorizes candidate compilation only; it is not model-quality evidence. An explicitly deferred entrypoint probe still requires a bounded runtime smoke.",
        }
        try:
            validate_document("model_conformance_receipt", receipt)
        except ContractValidationError as exc:
            raise ModelConformanceError("CONFORMANCE_RECEIPT_INVALID") from exc
        receipt_bytes = _canonical_json(receipt)
        _write_bytes(temporary / "conformance-receipt.json", receipt_bytes)
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-model-conformance-manifest",
            "state": "ready",
            "verdict": receipt["verdict"],
            "optimization_launch_allowed": receipt["optimization_launch_allowed"],
            "input_hash": input_hash,
            "receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
            "receipt_path": str(destination / "conformance-receipt.json"),
            "sidecar_root": str(sidecar),
        }
        _write_json(temporary / "manifest.json", manifest)
        os.replace(temporary, destination)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _resume_conformance(
    destination: Path,
    *,
    input_hash: str,
    repo: Path,
    expected_revision: object,
) -> dict[str, object]:
    if destination.is_symlink() or not destination.is_dir():
        raise ModelConformanceError("CONFORMANCE_OUTPUT_INVALID")
    lock = _load_json(destination / "input.lock.json", "CONFORMANCE_INPUT_LOCK_INVALID")
    if lock.get("input_hash") != input_hash:
        raise ModelConformanceError("CONFORMANCE_INPUT_MISMATCH")
    if compute_source_revision(repo) != expected_revision:
        raise ModelConformanceError("CONFORMANCE_SOURCE_DRIFT")
    manifest = _load_json(destination / "manifest.json", "CONFORMANCE_MANIFEST_INVALID")
    receipt_path = destination / "conformance-receipt.json"
    if hashlib.sha256(receipt_path.read_bytes()).hexdigest() != manifest.get(
        "receipt_sha256"
    ):
        raise ModelConformanceError("CONFORMANCE_RECEIPT_HASH_MISMATCH")
    receipt = _load_json(receipt_path, "CONFORMANCE_RECEIPT_INVALID")
    try:
        validate_document("model_conformance_receipt", receipt)
    except ContractValidationError as exc:
        raise ModelConformanceError("CONFORMANCE_RECEIPT_INVALID") from exc
    if compute_source_tree_revision(repo) != receipt.get("source_tree_revision"):
        raise ModelConformanceError("CONFORMANCE_SOURCE_TREE_DRIFT")
    try:
        verify_receipt_asset_bindings(receipt)
    except OnboardingAdmissionError as exc:
        raise ModelConformanceError(f"CONFORMANCE_ASSET_DRIFT:{exc}") from exc
    return manifest


def _validate_output_root(destination: Path, *, repo: Path, sidecar: Path) -> None:
    if destination == repo or repo in destination.parents:
        raise ModelConformanceError("CONFORMANCE_OUTPUT_INSIDE_SOURCE")
    if destination == sidecar or sidecar in destination.parents:
        raise ModelConformanceError("CONFORMANCE_OUTPUT_INSIDE_SIDECAR")


def _source_check(
    observed: Mapping[str, object], expected: object
) -> dict[str, object]:
    if (
        isinstance(expected, Mapping)
        and observed == expected
        and observed.get("state") == "bound"
    ):
        return _pass_check("source_revision", str(observed.get("revision")))
    return _fail_check("source_revision", "CONFORMANCE_SOURCE_REVISION_MISMATCH")


def _admission_check(report: Mapping[str, object]) -> dict[str, object]:
    blockers = report.get("blockers")
    passed = report.get("state") == "ready_for_conformance_smoke" and blockers == []
    return (
        _pass_check("onboarding_admission", "all onboarding bindings are ready")
        if passed
        else _fail_check("onboarding_admission", "CONFORMANCE_ONBOARDING_BLOCKED")
    )


def _output_isolation_check(
    output_root: Path, *, repo: Path, sidecar: Path
) -> dict[str, object]:
    passed = repo not in output_root.parents and sidecar not in output_root.parents
    return (
        _pass_check("output_isolation", str(output_root))
        if passed
        else _fail_check("output_isolation", "CONFORMANCE_OUTPUT_NOT_ISOLATED")
    )


def _source_integrity_check(
    before: Mapping[str, object], after: Mapping[str, object]
) -> dict[str, object]:
    return (
        _pass_check("source_integrity", "source revision unchanged")
        if before == after
        else _fail_check("source_integrity", "CONFORMANCE_SOURCE_MODIFIED")
    )


def _asset_binding_snapshot(
    report: Mapping[str, object],
) -> list[dict[str, object]]:
    connector = report.get("connector")
    if not isinstance(connector, Mapping):
        return []
    bindings = connector.get("asset_bindings")
    if not isinstance(bindings, list):
        return []
    rows: list[dict[str, object]] = []
    for binding in bindings:
        if not isinstance(binding, Mapping) or not binding.get(
            "required_for_evaluator"
        ):
            continue
        parameter = str(binding.get("parameter", ""))
        resolved_path = binding.get("resolved_path")
        expected = binding.get("asset_fingerprint")
        observed: str | None = None
        if isinstance(resolved_path, str) and resolved_path:
            try:
                observed = compute_asset_fingerprint(Path(resolved_path))
            except OnboardingError:
                observed = None
        if observed is None:
            state = "missing"
        elif observed != expected:
            state = "drifted"
        else:
            state = "bound"
        rows.append(
            {
                "parameter": parameter,
                "kind": str(binding.get("kind", "")),
                "resolved_path": str(resolved_path or ""),
                "onboarding_fingerprint": str(expected or ""),
                "observed_fingerprint": observed,
                "state": state,
            }
        )
    return sorted(rows, key=lambda row: str(row["parameter"]))


def _asset_binding_check(bindings: Sequence[Mapping[str, object]]) -> dict[str, object]:
    invalid = [
        str(row.get("parameter")) for row in bindings if row.get("state") != "bound"
    ]
    if invalid:
        return _fail_check(
            "asset_bindings", "CONFORMANCE_ASSET_BINDING_DRIFT:" + ",".join(invalid)
        )
    return _pass_check(
        "asset_bindings", f"{len(bindings)} evaluator asset bindings verified"
    )


def _asset_integrity_check(
    before: Sequence[Mapping[str, object]], after: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    return (
        _pass_check("asset_integrity", "asset bindings unchanged during conformance")
        if before == after
        else _fail_check("asset_integrity", "CONFORMANCE_ASSET_MODIFIED")
    )


def _materialize_command(
    report: Mapping[str, object], *, runtime: Path, repo: Path
) -> list[str]:
    evaluator = report.get("evaluator_contract")
    connector = report.get("connector")
    if not isinstance(evaluator, Mapping) or evaluator.get("state") != "ready":
        raise ModelConformanceError("CONFORMANCE_EVALUATOR_NOT_READY")
    if not isinstance(connector, Mapping):
        raise ModelConformanceError("CONFORMANCE_CONNECTOR_INVALID")
    asset_rows = connector.get("asset_bindings")
    if not isinstance(asset_rows, list):
        raise ModelConformanceError("CONFORMANCE_ASSET_BINDINGS_INVALID")
    assets = {
        str(row.get("parameter")): str(row.get("resolved_path"))
        for row in asset_rows
        if isinstance(row, Mapping)
        and row.get("state") == "discovered"
        and row.get("resolved_path")
    }
    raw_command = evaluator.get("command")
    if not isinstance(raw_command, list) or not raw_command:
        raise ModelConformanceError("CONFORMANCE_EVALUATOR_COMMAND_INVALID")
    values = {"{python}": str(runtime), "{repo_root}": str(repo)}
    command: list[str] = []
    for raw in raw_command:
        token = str(raw)
        for placeholder in _PLACEHOLDER.findall(token):
            if placeholder.startswith("{asset:"):
                parameter = placeholder[len("{asset:") : -1]
                value = assets.get(parameter)
                if value is None:
                    raise ModelConformanceError(
                        f"CONFORMANCE_ASSET_UNBOUND:{parameter}"
                    )
            else:
                value = values[placeholder]
            token = token.replace(placeholder, value)
        if "{" in token or "}" in token:
            raise ModelConformanceError("CONFORMANCE_PLACEHOLDER_INVALID")
        if (
            token.endswith(".py")
            and not Path(token).is_absolute()
            and (repo / token).is_file()
        ):
            token = str((repo / token).resolve())
        command.append(token)
    if Path(command[0]).resolve() != runtime.resolve():
        raise ModelConformanceError("CONFORMANCE_RUNTIME_COMMAND_MISMATCH")
    return command


def _run_check(
    *,
    name: str,
    command: Sequence[str],
    repo: Path,
    output_root: Path,
    timeout_seconds: float,
) -> dict[str, object]:
    home = output_root / "runtime-home"
    working = output_root / "runtime-work"
    home.mkdir(mode=0o700, exist_ok=True)
    working.mkdir(mode=0o700, exist_ok=True)
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(repo),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPYCACHEPREFIX": str(output_root / "pycache"),
        "CUDA_VISIBLE_DEVICES": "",
        "WANDB_MODE": "disabled",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "HOME": str(home),
        "XDG_CACHE_HOME": str(home / ".cache"),
        "TMPDIR": str(output_root / "tmp"),
        "LC_ALL": "C",
        "LANG": "C",
    }
    (output_root / "tmp").mkdir(mode=0o700, exist_ok=True)
    started = time.monotonic()
    try:
        result = subprocess.run(
            list(command),
            cwd=working,
            env=environment,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
        exit_code = result.returncode
        timed_out = False
        stdout = result.stdout[:_MAX_LOG_BYTES]
        stderr = result.stderr[:_MAX_LOG_BYTES]
    except subprocess.TimeoutExpired as exc:
        exit_code = None
        timed_out = True
        stdout = (exc.stdout or b"")[:_MAX_LOG_BYTES]
        stderr = (exc.stderr or b"")[:_MAX_LOG_BYTES]
    except OSError as exc:
        exit_code = None
        timed_out = False
        stdout = b""
        stderr = str(exc).encode("utf-8", errors="replace")[:_MAX_LOG_BYTES]
    duration = time.monotonic() - started
    stdout_path = output_root / "logs" / f"{name}.stdout.log"
    stderr_path = output_root / "logs" / f"{name}.stderr.log"
    _write_bytes(stdout_path, stdout)
    _write_bytes(stderr_path, stderr)
    passed = exit_code == 0 and not timed_out
    return {
        "name": name,
        "status": "pass" if passed else "fail",
        "detail": (
            "process exited successfully"
            if passed
            else _bounded_tail(stderr.decode("utf-8", errors="replace"))
        ),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "duration_seconds": float(f"{duration:.6f}"),
        "command": list(command),
        "stdout_path": f"logs/{name}.stdout.log",
        "stderr_path": f"logs/{name}.stderr.log",
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
    }


def _pass_check(name: str, detail: str) -> dict[str, object]:
    return {"name": name, "status": "pass", "detail": detail}


def _fail_check(name: str, detail: str) -> dict[str, object]:
    return {"name": name, "status": "fail", "detail": detail}


def _input_hash(
    report_bytes: bytes, capability_bytes: bytes, timeout_seconds: float
) -> str:
    payload = (
        report_bytes
        + b"\0"
        + capability_bytes
        + b"\0"
        + str(timeout_seconds).encode("ascii")
    )
    return hashlib.sha256(payload).hexdigest()


def _load_json(path: Path, code: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelConformanceError(code) from exc
    if not isinstance(value, dict):
        raise ModelConformanceError(code)
    return value


def _canonical_json(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    _write_bytes(path, _canonical_json(payload))


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def _bounded(value: str, *, limit: int = 600) -> str:
    value = value.replace("\x00", " ").strip()
    return value if len(value) <= limit else value[: limit - 3] + "..."


def _bounded_tail(value: str, *, limit: int = 600) -> str:
    value = value.replace("\x00", " ").strip()
    return value if len(value) <= limit else "..." + value[-(limit - 3) :]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    args = parser.parse_args(argv)
    try:
        manifest = run_conformance(
            ConformanceOptions(
                sidecar_root=args.sidecar_root,
                output_root=args.output_root,
                timeout_seconds=args.timeout_seconds,
            )
        )
    except ModelConformanceError as exc:
        print(str(exc), file=__import__("sys").stderr)
        return 2
    print(json.dumps(manifest, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
