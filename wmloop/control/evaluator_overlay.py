"""Copy evaluator metadata for review, freezing only an explicit valid contract.

Model repositories are read-only inputs. A protocol manifest is useful review
material, but cannot supply an evaluator command, metrics, or verifier that it
never declared. Unconfirmed overlays deliberately do not expose an executable
top-level contract to onboarding.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Mapping

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.control.evaluator_discovery import EvaluatorDiscoveryError, read_evaluator_manifest
from wmloop.control.onboarding_conformance import _python_script_index
from wmloop.control.onboarding_scan.evaluator import _invalid_evaluator_fields
from wmloop.storage import atomic_write, checked_path, exclusive_file_lock, sha256_bytes


class EvaluatorOverlayError(ValueError):
    """The candidate cannot be safely copied or frozen."""


_ROOT_FIELDS = {
    "repo_root": "source", "source_root": "source", "code_root": "source", "checkout_root": "source",
    "model_root": "model", "checkpoint_root": "model", "ckpt_root": "model",
    "data_root": "data", "dataset_root": "data", "collection_root": "data",
}
_PLACEHOLDER = re.compile(r"\{(?:python|repo_root|asset:--[A-Za-z0-9_-]+)\}")


def materialize_evaluator_overlay(
    candidate_path: Path,
    *,
    source_root: Path,
    model_root: Path,
    data_root: Path,
    output_root: Path,
    confirm: bool = False,
) -> dict[str, object]:
    """Create an immutable review or frozen overlay in a fresh output directory.

    Confirming after review uses a new output root. Existing outputs are never
    replaced, including during concurrent attempts to create the same output.
    """
    if not isinstance(confirm, bool):
        raise EvaluatorOverlayError("EVALUATOR_CONFIRM_BOOLEAN_REQUIRED")
    candidate = _regular_file(candidate_path, "EVALUATOR_CANDIDATE_INVALID")
    source = _directory(source_root, "EVALUATOR_SOURCE_ROOT_INVALID")
    model = _directory(model_root, "EVALUATOR_MODEL_ROOT_INVALID")
    data = _path(data_root, "EVALUATOR_DATA_ROOT_INVALID")
    destination = checked_path(output_root, code="EVALUATOR_OVERLAY_OUTPUT_SYMLINK", error=EvaluatorOverlayError)
    if any(_within(destination, root) for root in (source, model, data)):
        raise EvaluatorOverlayError("EVALUATOR_OVERLAY_OUTPUT_OVERLAP")
    try:
        payload, source_bytes = read_evaluator_manifest(candidate)
    except EvaluatorDiscoveryError as exc:
        raise EvaluatorOverlayError(str(exc)) from exc

    blockers: list[dict[str, str]] = []
    roots = {"source": source, "model": model, "data": data}
    bindings = _root_bindings(payload, roots=roots, blockers=blockers)
    rewrites: list[dict[str, str]] = []
    rewritten = _rewrite(payload, bindings=bindings, rewrites=rewrites, blockers=blockers)
    blockers.extend(_contract_blockers(rewritten, source=source))
    state = "binding_required" if blockers else ("frozen" if confirm else "ready_for_confirmation")
    confirmed = state == "frozen"
    claim = (
        "This exact, explicitly declared evaluator contract is frozen. Runtime conformance and paired target-side evidence remain required before optimization or quality claims."
        if confirmed else
        "Review metadata only. Supply the missing evaluator bindings and explicitly confirm a complete contract in a fresh output root; this overlay does not authorize execution."
    )
    # A complete candidate awaiting confirmation must also remain non-runnable
    # when passed directly to the onboarding contract loader.
    overlay_payload = rewritten if confirmed else {
        "schema_version": 1,
        "artifact_type": "verdiwm-evaluator-overlay-review",
        "state": state,
        "source_candidate": str(candidate),
        "candidate_manifest": rewritten,
        "blockers": blockers,
        "optimization_launch_allowed": False,
        "claim_boundary": claim,
    }
    # Onboarding consumes JSON contracts, including when source metadata was
    # YAML. Preserve semantic content and record the original byte digest.
    filename = _safe_name(candidate.stem) + ".json"
    overlay = destination / "evaluators" / filename
    encoded = _encode(overlay_payload)
    receipt_path = overlay.with_suffix(overlay.suffix + ".overlay.json")
    receipt: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-evaluator-overlay",
        "state": state,
        "source_candidate": str(candidate),
        "overlay_path": str(overlay),
        "source_sha256": sha256_bytes(source_bytes),
        "overlay_sha256": sha256_bytes(encoded),
        "rewrites": rewrites,
        "blockers": blockers,
        "confirmation": {"required": not confirmed, "confirmed": confirmed},
        "optimization_launch_allowed": False,
        "claim_boundary": claim,
        "receipt_path": str(receipt_path),
    }
    try:
        if not confirmed:
            validate_document("evaluator_overlay_review", overlay_payload)
        validate_document("evaluator_overlay_receipt", receipt)
    except ContractValidationError as exc:
        raise EvaluatorOverlayError("EVALUATOR_OVERLAY_RECEIPT_INVALID") from exc
    try:
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        lock = checked_path(destination.parent / f".{destination.name}.evaluator-overlay.lock", code="EVALUATOR_OVERLAY_OUTPUT_SYMLINK", error=EvaluatorOverlayError)
        with exclusive_file_lock(lock):
            checked_path(destination, code="EVALUATOR_OVERLAY_OUTPUT_SYMLINK", error=EvaluatorOverlayError)
            if destination.exists():
                raise EvaluatorOverlayError("EVALUATOR_OVERLAY_OUTPUT_EXISTS: use a fresh output root")
            temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
            try:
                atomic_write(temporary / "evaluators" / filename, encoded)
                atomic_write(temporary / "evaluators" / receipt_path.name, _encode(receipt))
                os.rename(temporary, destination)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
    except OSError as exc:
        raise EvaluatorOverlayError("EVALUATOR_OVERLAY_WRITE_FAILED") from exc
    return receipt


def _root_bindings(
    payload: Mapping[str, object], *, roots: Mapping[str, Path], blockers: list[dict[str, str]],
) -> dict[Path, Path]:
    """Only explicit root fields license prefix replacement, retaining suffixes."""
    bindings: dict[Path, Path] = {}
    for key, value in payload.items():
        kind = _ROOT_FIELDS.get(key.casefold())
        if kind is None or not isinstance(value, str) or not value.startswith("/"):
            continue
        original = Path(value)
        if ".." in original.parts or original == Path("/"):
            blockers.append(_blocker("EVALUATOR_ROOT_BINDING_AMBIGUOUS", key, "Declare a specific absolute source/model/data root without parent traversal."))
            continue
        target = roots[kind]
        if original in bindings and bindings[original] != target:
            blockers.append(_blocker("EVALUATOR_ROOT_BINDING_AMBIGUOUS", key, "Give source, model, and data distinct original root bindings."))
            continue
        bindings[original] = target
    return bindings


def _rewrite(
    value: object, *, bindings: Mapping[Path, Path], rewrites: list[dict[str, str]],
    blockers: list[dict[str, str]], key: str = "",
) -> object:
    if isinstance(value, Mapping):
        return {name: _rewrite(child, bindings=bindings, rewrites=rewrites, blockers=blockers, key=f"{key}.{name}".lstrip(".")) for name, child in value.items()}
    if isinstance(value, list):
        return [_rewrite(child, bindings=bindings, rewrites=rewrites, blockers=blockers, key=f"{key}[{index}]") for index, child in enumerate(value)]
    if not isinstance(value, str) or not value.startswith("/"):
        return value
    original = Path(value)
    if ".." in original.parts:
        blockers.append(_blocker("EVALUATOR_PATH_BINDING_AMBIGUOUS", key, "Replace parent traversal with an explicit path under the intended root."))
        return value
    replacement = original
    for old in sorted(bindings, key=lambda item: len(item.parts), reverse=True):
        if _within(original, old):
            replacement = bindings[old] / original.relative_to(old)
            break
    if str(replacement) != value:
        rewrites.append({"field": key, "from": value, "to": str(replacement)})
    # Declared outputs may be created later. Other stale absolute input paths
    # need an explicit binding, rather than a guessed root or discarded suffix.
    if not key.startswith("output_artifacts[") and not replacement.exists():
        blockers.append(_blocker("EVALUATOR_PATH_BINDING_REQUIRED", key, "Bind this input path to an existing file/directory; declare its original root to preserve subpaths."))
    return str(replacement)


def _contract_blockers(payload: Mapping[str, object], *, source: Path) -> list[dict[str, str]]:
    try:
        invalid = _invalid_evaluator_fields(payload)
    except TypeError:
        # The shared validator expects a scalar enum; unhashable metadata is
        # still an invalid contract, rather than an exception during review.
        return [_blocker("EVALUATOR_CONTRACT_FIELDS_INVALID", "entrypoint_probe", "Use the declared enum value 'help' or 'skip'.")]
    blockers = [
        _blocker("EVALUATOR_CONTRACT_FIELD_REQUIRED" if field not in payload else "EVALUATOR_CONTRACT_FIELD_INVALID", field,
                 "Declare a nonempty evaluator ID/verifier or an explicit nonempty list of command, input/output artifacts, and metrics, as appropriate.")
        for field in invalid
    ]
    if invalid:
        return blockers
    command = payload["command"]
    materialized: list[str] = []
    for token in command:
        remainder = _PLACEHOLDER.sub("", token)
        if "{" in remainder or "}" in remainder:
            blockers.append(_blocker("EVALUATOR_COMMAND_PLACEHOLDER_UNBOUND", "command", "Use supported {python}, {repo_root}, or {asset:--parameter} bindings."))
            return blockers
        materialized.append(token.replace("{repo_root}", str(source)))
    working = str(payload.get("working_directory", ".")).replace("{repo_root}", str(source))
    if "{" in working or "}" in working or ".." in Path(working).parts:
        return [_blocker("EVALUATOR_WORKING_DIRECTORY_UNBOUND", "working_directory", "Bind the working directory explicitly under the source checkout.")]
    directory = Path(working) if Path(working).is_absolute() else source / working
    try:
        _directory(directory, "EVALUATOR_WORKING_DIRECTORY_INVALID")
    except EvaluatorOverlayError as exc:
        return [_blocker(str(exc), "working_directory", "Select an existing directory without symlink components.")]
    if materialized[0] != "{python}":
        return [_blocker("EVALUATOR_RUNTIME_BINDING_REQUIRED", "command[0]", "Use {python} to bind the selected model runtime during conformance.")]
    entrypoint = _python_script_index(materialized)
    if entrypoint is not None:
        script = Path(materialized[entrypoint])
        script = script if script.is_absolute() else source / script
        try:
            _regular_file(script, "EVALUATOR_ENTRYPOINT_NOT_FOUND")
        except EvaluatorOverlayError as exc:
            blockers.append(_blocker(str(exc), f"command[{entrypoint}]", "Point to an existing evaluator script without symlinks; preserve its path below the source root."))
    elif "-m" in materialized:
        index = materialized.index("-m") + 1
        module = materialized[index] if index < len(materialized) else ""
        if not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", module):
            blockers.append(_blocker("EVALUATOR_ENTRYPOINT_MODULE_INVALID", "command", "Declare a Python module name after -m."))
        else:
            path = source.joinpath(*module.split("."))
            try:
                _regular_file(path.with_suffix(".py") if path.with_suffix(".py").is_file() else path / "__main__.py", "EVALUATOR_ENTRYPOINT_MODULE_UNRESOLVED")
            except EvaluatorOverlayError as exc:
                blockers.append(_blocker(str(exc), "command", "Provide the evaluator module in the source checkout or bind an explicit evaluator script."))
    elif "-c" not in materialized or materialized.index("-c") + 1 >= len(materialized):
        blockers.append(_blocker("EVALUATOR_ENTRYPOINT_UNDECLARED", "command", "Declare an evaluator script, module, or explicit Python code operand."))
    scheduler = payload.get("scheduler_template")
    if isinstance(scheduler, str):
        # Relative scheduler paths otherwise change meaning when the contract
        # is copied into VERDI-owned storage. Require an explicit binding.
        if not Path(scheduler).is_absolute():
            blockers.append(_blocker("EVALUATOR_SCHEDULER_BINDING_REQUIRED", "scheduler_template", "Bind the scheduler template to its existing absolute path before copying the contract."))
        else:
            try:
                _regular_file(Path(scheduler), "EVALUATOR_SCHEDULER_TEMPLATE_INVALID")
            except EvaluatorOverlayError as exc:
                blockers.append(_blocker(str(exc), "scheduler_template", "Select an existing scheduler template without symlinks."))
    return blockers


def _blocker(code: str, field: str, action: str) -> dict[str, str]:
    return {"code": code, "field": field, "action": action}


def _encode(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _regular_file(path: Path, code: str) -> Path:
    candidate = checked_path(path, code=code, error=EvaluatorOverlayError)
    if not candidate.is_file():
        raise EvaluatorOverlayError(code)
    return candidate


def _directory(path: Path, code: str) -> Path:
    candidate = checked_path(path, code=code, error=EvaluatorOverlayError)
    if not candidate.is_dir():
        raise EvaluatorOverlayError(code)
    return candidate


def _path(path: Path, code: str) -> Path:
    candidate = checked_path(path, code=code, error=EvaluatorOverlayError)
    if not candidate.exists():
        raise EvaluatorOverlayError(code)
    return candidate


def _within(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.") or "evaluator"
