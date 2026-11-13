"""Read-only, contract-first onboarding for external world-model repositories.

The onboarding boundary deliberately stops before model execution.  It discovers
what a repository can provide, probes an explicitly selected Python runtime
without exposing CUDA, and emits a sidecar contract that a later conformance
runner can verify.  No source file in the imported repository is modified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wmloop.storage import checked_path

from wmloop.control.onboarding_scan.assets import (
    _discover_assets as _discover_assets,
    _discover_source_revision as _discover_source_revision,
    _is_source_revision_file as _is_source_revision_file,
    _source_tree_revision as _source_tree_revision,
)
from wmloop.control.onboarding_scan.bindings import (
    _asset_fingerprint as _asset_fingerprint,
    _binding_kind as _binding_kind,
    _build_connector as _build_connector,
    _discover_asset_bindings as _discover_asset_bindings,
    _evaluator_asset_parameters as _evaluator_asset_parameters,
    _matching_assets as _matching_assets,
    _validate_additional_asset_parameters as _validate_additional_asset_parameters,
    _validate_explicit_asset_bindings as _validate_explicit_asset_bindings,
)
from wmloop.control.onboarding_scan.capabilities import (
    _classify_capabilities as _classify_capabilities,
)
from wmloop.control.onboarding_scan.evaluator import (
    _invalid_evaluator_fields as _invalid_evaluator_fields,
    _load_evaluator_contract as _load_evaluator_contract,
    _nonempty_string_list as _nonempty_string_list,
)
from wmloop.control.onboarding_scan.files import (
    _discover_files as _discover_files,
    _relative as _relative,
    _resolve_output_root as _resolve_output_root,
    _validate_repo as _validate_repo,
)
from wmloop.control.onboarding_scan.readiness import (
    _build_blockers as _build_blockers,
    _conformance_report as _conformance_report,
    _dedupe_blockers as _dedupe_blockers,
    _next_actions as _next_actions,
    _state as _state,
)
from wmloop.control.onboarding_scan.reports import (
    _bounded as _bounded,
    _canonical_json_bytes as _canonical_json_bytes,
    _render_markdown as _render_markdown,
    _safe_size as _safe_size,
    _sha256_file as _sha256_file,
    _validate_report as _validate_report,
    _write_bundle as _write_bundle,
    _write_bytes as _write_bytes,
)
from wmloop.control.onboarding_scan.runtime import (
    _discover_runtime as _discover_runtime,
    _probe_imports as _probe_imports,
    _probe_pip as _probe_pip,
    _python_version as _python_version,
    _required_packages as _required_packages,
    _runtime_candidates as _runtime_candidates,
)
from wmloop.control.onboarding_scan.source import (
    _declared_packages as _declared_packages,
    _dependency_kind as _dependency_kind,
    _discover_dependencies as _discover_dependencies,
    _discover_entrypoints as _discover_entrypoints,
    _entrypoint_kinds as _entrypoint_kinds,
    _has_main_guard as _has_main_guard,
    _imports as _imports,
    _regex_packages as _regex_packages,
)
from wmloop.control.onboarding_scan.types import (
    DEFAULT_RUNTIME_TIMEOUT_SECONDS as DEFAULT_RUNTIME_TIMEOUT_SECONDS,
    _ARTIFACT_DIRECTORY_PREFIXES as _ARTIFACT_DIRECTORY_PREFIXES,
    _CAPABILITY_KEYWORDS as _CAPABILITY_KEYWORDS,
    _CHECKPOINT_SUFFIXES as _CHECKPOINT_SUFFIXES,
    _DATA_METADATA_SUFFIXES as _DATA_METADATA_SUFFIXES,
    _DATA_SUFFIXES as _DATA_SUFFIXES,
    _DEPENDENCY_FILENAMES as _DEPENDENCY_FILENAMES,
    _DEPENDENCY_PREFIXES as _DEPENDENCY_PREFIXES,
    _ENTRYPOINT_KEYWORDS as _ENTRYPOINT_KEYWORDS,
    _MAX_ASSETS as _MAX_ASSETS,
    _MAX_DEPENDENCIES as _MAX_DEPENDENCIES,
    _MAX_ENTRYPOINTS as _MAX_ENTRYPOINTS,
    _MAX_ERROR as _MAX_ERROR,
    _MAX_FILES as _MAX_FILES,
    _MAX_IMPORTS as _MAX_IMPORTS,
    _MAX_SOURCE_PARSE_BYTES as _MAX_SOURCE_PARSE_BYTES,
    _MAX_SOURCE_REVISION_FILES as _MAX_SOURCE_REVISION_FILES,
    _PACKAGE_IMPORT_ALIASES as _PACKAGE_IMPORT_ALIASES,
    _PIP_PLATFORM_WARNING as _PIP_PLATFORM_WARNING,
    _SKIP_DIRECTORIES as _SKIP_DIRECTORIES,
    _SOURCE_OR_DOCUMENT_SUFFIXES as _SOURCE_OR_DOCUMENT_SUFFIXES,
    _SUBPROCESS_TIMEOUT_SECONDS as _SUBPROCESS_TIMEOUT_SECONDS,
    SCHEMA_VERSION as SCHEMA_VERSION,
    OnboardingError as OnboardingError,
    OnboardingOptions as OnboardingOptions,
    validate_runtime_timeout as validate_runtime_timeout,
    _is_artifact_directory as _is_artifact_directory,
)


def run_onboarding(options: OnboardingOptions) -> dict[str, object]:
    """Scan a repository and write a deterministic onboarding sidecar.

    The returned manifest is safe to consume by a scheduler, but a scan never
    grants GPU launch permission.  A conformance receipt is required first.
    """

    repo = _validate_repo(options.repo_root)
    destination = _resolve_output_root(repo, options.output_root)
    if destination.exists() or destination.is_symlink():
        raise OnboardingError("ONBOARDING_OUTPUT_EXISTS")

    report = _build_report(repo, options)
    return _write_bundle(
        report, destination, repo=repo, source_revision=report["source_revision"]
    )


def scan_repository(options: OnboardingOptions) -> dict[str, object]:
    """Return the report without writing files, primarily for unit tests."""

    repo = _validate_repo(options.repo_root)
    return _build_report(repo, options)


def compute_source_revision(
    repo_root: Path, *, max_files: int = _MAX_FILES
) -> dict[str, object]:
    """Return the same source binding used by onboarding without other probes."""

    repo = _validate_repo(repo_root)
    files, truncated = _discover_files(repo, max_files=max_files)
    if truncated:
        return {
            "state": "unbound",
            "kind": "none",
            "revision": "SOURCE_REVISION_UNBOUND",
            "detail": "source inventory exceeded the revision scan limit",
        }
    return _discover_source_revision(repo, files)


def compute_source_tree_revision(
    repo_root: Path, *, max_files: int = _MAX_FILES
) -> dict[str, object]:
    """Bind inventoried source/config content; unbound links cannot be omitted.

    Existing artifact/environment-directory exclusions and scan limits still
    apply. This binds checkout content, not installed runtime dependencies.
    """
    try:
        repo = _validate_repo(checked_path(
            repo_root, code="SOURCE_REVISION_SYMLINK_FORBIDDEN", error=OnboardingError,
        ))
        files, truncated = _discover_files(repo, max_files=max_files, reject_symlinks=True)
    except OnboardingError as exc:
        if str(exc) != "SOURCE_REVISION_SYMLINK_FORBIDDEN":
            raise
        return {
            "state": "unbound", "kind": "none", "revision": "SOURCE_REVISION_UNBOUND",
            "detail": "SOURCE_REVISION_SYMLINK_FORBIDDEN: source inventory contains an unbound symbolic link",
        }
    if truncated:
        return {
            "state": "unbound",
            "kind": "none",
            "revision": "SOURCE_REVISION_UNBOUND",
            "detail": "source inventory exceeded the revision scan limit",
        }
    return _source_tree_revision(repo, files)


def compute_asset_fingerprint(asset_path: Path) -> str:
    """Return the deterministic fingerprint used for an external asset binding."""

    raw_path = Path(asset_path).expanduser()
    if not raw_path.exists() or raw_path.is_symlink():
        raise OnboardingError("ASSET_FINGERPRINT_PATH_INVALID")
    path = raw_path.resolve()
    try:
        return _asset_fingerprint(path)
    except OSError as exc:
        raise OnboardingError("ASSET_FINGERPRINT_FAILED") from exc


def _build_report(repo: Path, options: OnboardingOptions) -> dict[str, object]:
    files, inventory_truncated = _discover_files(repo, max_files=options.max_files)
    dependencies = _discover_dependencies(files, repo)
    entrypoints = _discover_entrypoints(files, repo)
    assets = _discover_assets(files, repo)
    source_revision = _discover_source_revision(repo, files)
    runtime = _discover_runtime(
        repo,
        dependencies,
        runtime_python=options.runtime_python,
        probe_imports=options.probe_imports,
        runtime_timeout_seconds=options.runtime_timeout_seconds,
    )
    capabilities = _classify_capabilities(entrypoints, assets, files, repo=repo)
    evaluator_contract = _load_evaluator_contract(options.evaluator_contract)
    asset_bindings = _discover_asset_bindings(
        entrypoints,
        assets,
        repo=repo,
        explicit_bindings=options.asset_bindings,
        evaluator_contract=evaluator_contract,
        additional_parameters=options.additional_asset_parameters,
    )
    connector = _build_connector(
        repo,
        runtime,
        entrypoints,
        capabilities,
        evaluator_contract,
        asset_bindings,
    )
    blockers = _build_blockers(
        source_revision=source_revision,
        runtime=runtime,
        entrypoints=entrypoints,
        assets=assets,
        capabilities=capabilities,
        evaluator_contract=evaluator_contract,
        asset_bindings=asset_bindings,
        adapter_contract_ready=options.adapter_contract_ready,
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "wmloop-model-onboarding-report",
        "repo_root": str(repo),
        "repo_name": repo.name,
        "source_revision": source_revision,
        "scan_limits": {
            "max_files": options.max_files,
            "inventory_truncated": inventory_truncated,
            "source_parse_bytes": _MAX_SOURCE_PARSE_BYTES,
        },
        "dependencies": dependencies,
        "entrypoints": entrypoints,
        "assets": assets,
        "runtime": runtime,
        "capabilities": capabilities,
        "evaluator_contract": evaluator_contract,
        "connector": connector,
        "conformance": _conformance_report(),
        "blockers": blockers,
        "state": _state(
            blockers, runtime=runtime, evaluator_contract=evaluator_contract
        ),
        "optimization_launch_allowed": False,
        "side_effects": {
            "source_modified": False,
            "dependency_install_started": False,
            "model_import_executed": False,
            "gpu_execution_started": False,
            "source_output_written": False,
        },
        "next_actions": _next_actions(
            _state(blockers, runtime=runtime, evaluator_contract=evaluator_contract),
            blockers,
        ),
        "limitations": [
            "Onboarding discovers and validates boundaries; it does not train, infer, rollout, or allocate a GPU.",
            "Dependency import probes are run in an isolated subprocess with CUDA hidden and bounded timeouts.",
            "A conformance receipt from the declared evaluator is required before an optimization scheduler can launch.",
        ],
    }
    _validate_report(report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "repo_root", type=Path, help="External model repository to inspect"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Sibling sidecar directory; must be outside repo_root",
    )
    parser.add_argument(
        "--runtime-python", type=Path, help="Explicit Python interpreter to probe"
    )
    parser.add_argument(
        "--runtime-timeout-seconds", type=float,
        default=DEFAULT_RUNTIME_TIMEOUT_SECONDS,
        help="Per import and pip-check timeout (finite, >0 and <=300; default: 30)",
    )
    parser.add_argument(
        "--evaluator-contract", type=Path, help="Frozen JSON evaluator contract"
    )
    parser.add_argument(
        "--asset",
        action="append",
        default=[],
        metavar="PARAMETER=PATH",
        help="Bind a discovered input parameter to an external asset",
    )
    parser.add_argument(
        "--no-import-probe",
        action="store_true",
        help="Only inspect metadata; skip subprocess imports and pip check",
    )
    parser.add_argument("--max-files", type=int, default=_MAX_FILES)
    args = parser.parse_args(argv)
    asset_bindings = tuple(_parse_asset_argument(value) for value in args.asset)
    manifest = run_onboarding(
        OnboardingOptions(
            repo_root=args.repo_root,
            output_root=args.output_root,
            runtime_python=args.runtime_python,
            runtime_timeout_seconds=args.runtime_timeout_seconds,
            evaluator_contract=args.evaluator_contract,
            asset_bindings=asset_bindings,
            probe_imports=not args.no_import_probe,
            max_files=args.max_files,
        )
    )
    print(json.dumps(manifest, sort_keys=True, ensure_ascii=False))
    return 0


def _parse_asset_argument(value: str) -> tuple[str, Path]:
    parameter, separator, path = value.partition("=")
    if not separator or not parameter.strip() or not path.strip():
        raise OnboardingError("ASSET_BINDING_ARGUMENT_INVALID")
    return parameter.strip(), Path(path.strip())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
