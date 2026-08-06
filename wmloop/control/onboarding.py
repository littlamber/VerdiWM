"""Read-only, contract-first onboarding for external world-model repositories.

The onboarding boundary deliberately stops before model execution.  It discovers
what a repository can provide, probes an explicitly selected Python runtime
without exposing CUDA, and emits a sidecar contract that a later conformance
runner can verify.  No source file in the imported repository is modified.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from wmloop.contracts import ContractValidationError, validate_document


SCHEMA_VERSION = 1
_MAX_ERROR = 600
_MAX_FILES = 20_000
_MAX_ENTRYPOINTS = 200
_MAX_ASSETS = 500
_MAX_DEPENDENCIES = 200
_MAX_IMPORTS = 64
_MAX_SOURCE_PARSE_BYTES = 1_500_000
_MAX_SOURCE_REVISION_FILES = 5_000
_SUBPROCESS_TIMEOUT_SECONDS = 8.0
_SKIP_DIRECTORIES = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "node_modules",
        "wandb",
        "runs",
        "outputs",
        "checkpoints",
        ".cache",
    }
)
_DEPENDENCY_FILENAMES = frozenset(
    {
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "requirements.txt",
        "requirements-dev.txt",
        "requirements-dev.in",
        "environment.yml",
        "environment.yaml",
        "conda.yml",
        "conda.yaml",
        "dockerfile",
        "uv.lock",
        "poetry.lock",
        "pipfile",
        "pipfile.lock",
    }
)
_DEPENDENCY_PREFIXES = ("requirements-", "environment-", "conda-")
_CHECKPOINT_SUFFIXES = frozenset(
    {".ckpt", ".pt", ".pth", ".safetensors", ".bin", ".onnx"}
)
_DATA_SUFFIXES = frozenset(
    {".npy", ".npz", ".parquet", ".tfrecord", ".tfrecords", ".h5", ".hdf5"}
)
_DATA_METADATA_SUFFIXES = _DATA_SUFFIXES | {
    ".json",
    ".csv",
    ".mp4",
    ".webm",
    ".avi",
    ".jsonl",
}
_SOURCE_OR_DOCUMENT_SUFFIXES = frozenset(
    {".py", ".pyi", ".md", ".txt", ".rst", ".toml", ".yaml", ".yml"}
)
_PACKAGE_IMPORT_ALIASES = {
    "accelerate": "accelerate",
    "decord": "decord",
    "diffusers": "diffusers",
    "einops": "einops",
    "mediapy": "mediapy",
    "numpy": "numpy",
    "opencv-python": "cv2",
    "pillow": "PIL",
    "pyyaml": "yaml",
    "scikit-image": "skimage",
    "scikit-learn": "sklearn",
    "scipy": "scipy",
    "swanlab": "swanlab",
    "torch": "torch",
    "transformers": "transformers",
    "tqdm": "tqdm",
    "wandb": "wandb",
}
_ENTRYPOINT_KEYWORDS = (
    "train",
    "eval",
    "evaluate",
    "rollout",
    "inference",
    "infer",
    "predict",
    "benchmark",
    "run",
)
_CAPABILITY_KEYWORDS = {
    "training": ("train", "optimizer", "backward", "accelerate"),
    "evaluation": ("eval", "evaluate", "metric", "benchmark", "validation"),
    "rollout": ("rollout", "replay", "interact", "trajectory"),
    "inference": ("inference", "infer", "predict", "pipeline", "forward"),
    "action_conditioned": ("action", "policy", "control", "robot", "joint", "eef"),
    "video_generation": ("video", "diffusion", "latent", "vae", "unet", "frame"),
}


class OnboardingError(RuntimeError):
    """An onboarding request failed before a trustworthy report was written."""


@dataclass(frozen=True)
class OnboardingOptions:
    """Explicit options for a read-only onboarding scan."""

    repo_root: Path
    output_root: Path | None = None
    runtime_python: Path | None = None
    evaluator_contract: Path | None = None
    asset_bindings: tuple[tuple[str, Path], ...] = ()
    probe_imports: bool = True
    max_files: int = _MAX_FILES


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
    """Return a content binding even when the repository is a Git checkout."""

    repo = _validate_repo(repo_root)
    files, truncated = _discover_files(repo, max_files=max_files)
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
    )
    capabilities = _classify_capabilities(entrypoints, assets, files, repo=repo)
    evaluator_contract = _load_evaluator_contract(options.evaluator_contract)
    asset_bindings = _discover_asset_bindings(
        entrypoints,
        assets,
        repo=repo,
        explicit_bindings=options.asset_bindings,
        evaluator_contract=evaluator_contract,
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


def _validate_repo(path: Path) -> Path:
    repo = Path(path).expanduser().resolve()
    if not repo.is_dir():
        raise OnboardingError("MODEL_REPOSITORY_NOT_FOUND")
    return repo


def _resolve_output_root(repo: Path, output_root: Path | None) -> Path:
    destination = (
        Path(output_root).expanduser().resolve()
        if output_root is not None
        else repo.parent / f"{repo.name}.verdiwm-instance"
    )
    try:
        inside = destination == repo or repo in destination.parents
    except (
        OSError
    ) as exc:  # pragma: no cover - Path parents are pure, retained for fail-closed behavior
        raise OnboardingError("ONBOARDING_OUTPUT_INVALID") from exc
    if inside:
        raise OnboardingError("ONBOARDING_OUTPUT_INSIDE_SOURCE")
    if destination.parent == repo.parent and destination.name == repo.name:
        raise OnboardingError("ONBOARDING_OUTPUT_INVALID")
    return destination


def _discover_files(repo: Path, *, max_files: int) -> tuple[list[Path], bool]:
    if max_files < 1 or max_files > _MAX_FILES:
        raise OnboardingError("ONBOARDING_MAX_FILES_INVALID")
    paths: list[Path] = []
    truncated = False
    pending = [repo]
    while pending:
        directory = pending.pop()
        try:
            children = sorted(directory.iterdir(), key=lambda item: item.name)
        except OSError:
            continue
        for child in children:
            if child.is_symlink():
                continue
            if child.is_dir():
                if child.name not in _SKIP_DIRECTORIES and not child.name.startswith(
                    "."
                ):
                    pending.append(child)
                continue
            if child.is_file():
                paths.append(child)
                if len(paths) >= max_files:
                    truncated = True
                    return sorted(paths), truncated
    return sorted(paths), truncated


def _relative(path: Path, repo: Path) -> str:
    return path.relative_to(repo).as_posix()


def _discover_dependencies(
    files: Sequence[Path], repo: Path
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in files:
        name = path.name.lower()
        if name not in _DEPENDENCY_FILENAMES and not name.startswith(
            _DEPENDENCY_PREFIXES
        ):
            continue
        row: dict[str, object] = {
            "path": _relative(path, repo),
            "kind": _dependency_kind(name),
            "size_bytes": _safe_size(path),
            "declared_packages": _declared_packages(path),
        }
        rows.append(row)
    rows.sort(key=lambda item: str(item["path"]))
    return rows[:_MAX_DEPENDENCIES]


def _dependency_kind(name: str) -> str:
    if name.startswith("requirements"):
        return "requirements"
    if name.endswith((".yml", ".yaml")) or name.startswith("conda"):
        return "environment"
    if name in {"pyproject.toml", "setup.py", "setup.cfg", "pipfile", "pipfile.lock"}:
        return "python_project"
    return "lockfile"


def _declared_packages(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")[
            :_MAX_SOURCE_PARSE_BYTES
        ]
    except OSError:
        return []
    name = path.name.lower()
    packages: set[str] = set()
    if name.startswith("requirements") or name.endswith(".in"):
        for raw in text.splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line or line.startswith(("-", "git+", "http:", "https:")):
                continue
            match = re.match(r"([A-Za-z0-9][A-Za-z0-9_.-]*)", line)
            if match:
                packages.add(match.group(1).lower())
    elif name == "pyproject.toml":
        try:
            import tomllib

            payload = tomllib.loads(text)
            values = payload.get("project", {}).get("dependencies", [])
            if isinstance(values, list):
                for value in values:
                    match = re.match(r"([A-Za-z0-9][A-Za-z0-9_.-]*)", str(value))
                    if match:
                        packages.add(match.group(1).lower())
        except (ValueError, TypeError, AttributeError):
            packages.update(_regex_packages(text))
    else:
        packages.update(_regex_packages(text))
    return sorted(packages)[:_MAX_DEPENDENCIES]


def _regex_packages(text: str) -> set[str]:
    packages: set[str] = set()
    for raw in text.splitlines():
        match = re.search(
            r"(?:install_requires|dependencies|pip|conda)[^\n]*?([A-Za-z][A-Za-z0-9_.-]+)",
            raw,
            re.I,
        )
        if match:
            packages.add(match.group(1).lower())
    return packages


def _discover_entrypoints(files: Sequence[Path], repo: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in files:
        suffix = path.suffix.lower()
        if suffix not in {".py", ".sh"}:
            continue
        relative = _relative(path, repo)
        tokens = set(re.split(r"[^a-z0-9]+", relative.lower()))
        likely = bool(tokens & set(_ENTRYPOINT_KEYWORDS)) or path.parent.name in {
            "scripts",
            "bin",
            "tools",
        }
        main_guard = False
        functions: list[str] = []
        imports: list[str] = []
        flags: list[str] = []
        parse_error: str | None = None
        if suffix == ".py" and _safe_size(path) <= _MAX_SOURCE_PARSE_BYTES:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
                tree = ast.parse(text, filename=relative)
                main_guard = _has_main_guard(tree)
                functions = sorted(
                    {
                        node.name
                        for node in ast.walk(tree)
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and node.name
                        in {
                            "main",
                            "train",
                            "evaluate",
                            "eval",
                            "rollout",
                            "inference",
                            "predict",
                        }
                    }
                )
                imports = sorted(_imports(tree))[:_MAX_IMPORTS]
                flags = sorted(set(re.findall(r"--[a-zA-Z0-9][a-zA-Z0-9_-]*", text)))[
                    :_MAX_IMPORTS
                ]
                likely = likely or main_guard or bool(functions)
            except (OSError, SyntaxError, UnicodeError) as exc:
                parse_error = _bounded(str(exc))
        elif suffix == ".sh":
            try:
                text = path.read_text(encoding="utf-8", errors="replace")[
                    :_MAX_SOURCE_PARSE_BYTES
                ]
                flags = sorted(set(re.findall(r"--[a-zA-Z0-9][a-zA-Z0-9_-]*", text)))[
                    :_MAX_IMPORTS
                ]
                likely = likely or any(
                    keyword in text.lower() for keyword in _ENTRYPOINT_KEYWORDS
                )
            except (OSError, UnicodeError) as exc:
                parse_error = _bounded(str(exc))
        if not likely:
            continue
        rows.append(
            {
                "path": relative,
                "language": "python" if suffix == ".py" else "shell",
                "kinds": _entrypoint_kinds(relative, functions, flags),
                "executable": suffix == ".sh" or main_guard or "main" in functions,
                "main_guard": main_guard,
                "functions": functions,
                "imports": imports,
                "cli_flags": flags,
                "parse_error": parse_error,
            }
        )
    rows.sort(key=lambda item: str(item["path"]))
    return rows[:_MAX_ENTRYPOINTS]


def _has_main_guard(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if (
            isinstance(test, ast.Compare)
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq)
        ):
            values = [test.left, *test.comparators]
            if any(
                isinstance(value, ast.Name) and value.id == "__name__"
                for value in values
            ) and any(
                isinstance(value, ast.Constant) and value.value == "__main__"
                for value in values
            ):
                return True
    return False


def _imports(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".", 1)[0])
    return names


def _entrypoint_kinds(
    path: str, functions: Sequence[str], flags: Sequence[str]
) -> list[str]:
    text = " ".join((path.lower(), *functions, *flags))
    kinds: list[str] = []
    for kind, keywords in (
        ("training", ("train", "optimizer", "backward")),
        ("evaluation", ("eval", "evaluate", "metric", "benchmark", "validation")),
        ("rollout", ("rollout", "replay", "interact", "trajectory")),
        ("inference", ("inference", "infer", "predict", "pipeline", "forward")),
    ):
        if any(keyword in text for keyword in keywords):
            kinds.append(kind)
    return kinds or ["unknown"]


def _discover_assets(files: Sequence[Path], repo: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in files:
        parts = set(re.split(r"[^a-z0-9]+", _relative(path, repo).lower()))
        suffix = path.suffix.lower()
        dataset_marker = bool(
            parts
            & {
                "dataset",
                "annotation",
                "trajectory",
                "data_stat",
                "latent_videos",
                "videos",
            }
        )
        checkpoint = (
            suffix in _CHECKPOINT_SUFFIXES
            or bool(parts & {"checkpoint", "ckpt", "weights"})
        ) and not dataset_marker
        dataset = suffix in _DATA_METADATA_SUFFIXES or dataset_marker
        if suffix in _SOURCE_OR_DOCUMENT_SUFFIXES:
            dataset = False
        model_dependency = bool(parts & {"vae", "clip", "svd"})
        if not (checkpoint or dataset or model_dependency):
            continue
        if checkpoint:
            kind = "checkpoint"
        elif dataset:
            kind = "dataset_or_metadata"
        else:
            kind = "model_dependency"
        rows.append(
            {
                "path": _relative(path, repo),
                "kind": kind,
                "size_bytes": _safe_size(path),
                "large_file": _safe_size(path) >= 1_000_000_000,
                "hashed": False,
            }
        )
    rows.sort(key=lambda item: str(item["path"]))
    return rows[:_MAX_ASSETS]


def _discover_source_revision(repo: Path, files: Sequence[Path]) -> dict[str, object]:
    git = shutil.which("git")
    if git is not None:
        try:
            result = subprocess.run(
                [git, "-C", str(repo), "rev-parse", "HEAD"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
                env={"PATH": os.environ.get("PATH", ""), "LC_ALL": "C", "LANG": "C"},
            )
        except (OSError, subprocess.TimeoutExpired):
            result = None
        if result is not None:
            revision = result.stdout.strip()
            if result.returncode == 0 and re.fullmatch(r"[0-9a-fA-F]{7,64}", revision):
                return {
                    "state": "bound",
                    "kind": "git_commit",
                    "revision": revision.lower(),
                    "detail": "git rev-parse HEAD",
                }
    return _source_tree_revision(repo, files)


def _source_tree_revision(repo: Path, files: Sequence[Path]) -> dict[str, object]:
    selected = [path for path in files if _is_source_revision_file(path, repo)]
    selected.sort(key=lambda path: _relative(path, repo))
    if not selected or len(selected) > _MAX_SOURCE_REVISION_FILES:
        return {
            "state": "unbound",
            "kind": "none",
            "revision": "SOURCE_REVISION_UNBOUND",
            "detail": "source tree is empty or exceeds the revision file limit",
        }
    digest = hashlib.sha256()
    try:
        for path in selected:
            relative = _relative(path, repo).encode("utf-8")
            payload = path.read_bytes()
            digest.update(len(relative).to_bytes(4, "big"))
            digest.update(relative)
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(hashlib.sha256(payload).digest())
    except OSError as exc:
        return {
            "state": "unbound",
            "kind": "none",
            "revision": "SOURCE_REVISION_UNBOUND",
            "detail": _bounded(str(exc)),
        }
    return {
        "state": "bound",
        "kind": "source_tree_sha256",
        "revision": digest.hexdigest(),
        "detail": f"content hash over {len(selected)} source and configuration files",
    }


def _is_source_revision_file(path: Path, repo: Path) -> bool:
    relative = _relative(path, repo).lower()
    parts = set(re.split(r"[^a-z0-9]+", relative))
    if parts & {"dataset", "annotation", "trajectory", "videos", "latent_videos"}:
        return False
    if path.name.lower() in _DEPENDENCY_FILENAMES:
        return True
    return path.suffix.lower() in {
        ".py",
        ".pyi",
        ".sh",
        ".toml",
        ".yaml",
        ".yml",
        ".json",
        ".cfg",
        ".ini",
    }


def _discover_runtime(
    repo: Path,
    dependencies: Sequence[Mapping[str, object]],
    *,
    runtime_python: Path | None,
    probe_imports: bool,
) -> dict[str, object]:
    candidates = _runtime_candidates(repo, runtime_python)
    selected = (
        candidates[0]
        if runtime_python is not None and candidates
        else next(
            (candidate for candidate in candidates if candidate["exists"]),
            None,
        )
    )
    if selected is None:
        return {
            "state": "blocked",
            "selected_python": None,
            "candidates": [],
            "required_packages": _required_packages(dependencies),
            "import_probes": [],
            "pip_check": {
                "state": "not_run",
                "returncode": None,
                "detail": "no Python runtime found",
            },
        }
    required = _required_packages(dependencies)
    probes = (
        _probe_imports(selected, required, repo=repo)
        if probe_imports and selected["exists"]
        else []
    )
    pip_check = (
        _probe_pip(selected, repo=repo)
        if probe_imports and selected["exists"]
        else {
            "state": "not_run",
            "returncode": None,
            "detail": "disabled or runtime missing",
        }
    )
    failures = [probe for probe in probes if probe["state"] != "ok"]
    state = (
        "ready"
        if selected["exists"]
        and not failures
        and pip_check["state"] in {"ok", "not_run"}
        else "blocked"
    )
    return {
        "state": state,
        "selected_python": selected["path"] if selected["exists"] else None,
        "candidates": candidates,
        "required_packages": required,
        "import_probes": probes,
        "pip_check": pip_check,
        "cuda_hidden": True,
        "probe_timeout_seconds": _SUBPROCESS_TIMEOUT_SECONDS,
    }


def _runtime_candidates(repo: Path, explicit: Path | None) -> list[dict[str, object]]:
    raw: list[tuple[Path, str]] = []
    if explicit is not None:
        raw.append((Path(explicit).expanduser().resolve(), "explicit"))
    for relative in (".venv/bin/python", "venv/bin/python", "env/bin/python"):
        raw.append((repo / relative, "repository_local"))
    raw.append((Path(sys.executable).resolve(), "current_process"))
    seen: set[str] = set()
    rows: list[dict[str, object]] = []
    for path, source in raw:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        exists = path.is_file() and os.access(path, os.X_OK)
        version = _python_version(path) if exists else None
        rows.append(
            {"path": key, "source": source, "exists": exists, "version": version}
        )
    return rows


def _python_version(path: Path) -> str | None:
    try:
        result = subprocess.run(
            [str(path), "--version"], capture_output=True, text=True, timeout=3
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return _bounded((result.stdout or result.stderr).strip(), limit=80)


def _required_packages(dependencies: Sequence[Mapping[str, object]]) -> list[str]:
    values: set[str] = set()
    for dependency in dependencies:
        for package in dependency.get("declared_packages", []):
            values.add(str(package).lower())
    return sorted(values)[:_MAX_DEPENDENCIES]


def _probe_imports(
    path_info: Mapping[str, object], packages: Sequence[str], *, repo: Path
) -> list[dict[str, object]]:
    if not packages:
        return []
    path = Path(str(path_info["path"]))
    rows: list[dict[str, object]] = []
    for package in packages:
        module = _PACKAGE_IMPORT_ALIASES.get(package, package.replace("-", "_"))
        code = "import importlib; importlib.import_module(" + repr(module) + ")"
        environment = {
            "PATH": os.environ.get("PATH", ""),
            "LC_ALL": "C",
            "LANG": "C",
            "PYTHONPATH": str(repo),
            "PYTHONDONTWRITEBYTECODE": "1",
            "CUDA_VISIBLE_DEVICES": "",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "WANDB_MODE": "disabled",
            "TOKENIZERS_PARALLELISM": "false",
        }
        try:
            result = subprocess.run(
                [str(path), "-c", code],
                check=False,
                capture_output=True,
                text=True,
                timeout=_SUBPROCESS_TIMEOUT_SECONDS,
                cwd=repo,
                env=environment,
            )
        except subprocess.TimeoutExpired as exc:
            rows.append(
                {
                    "package": package,
                    "module": module,
                    "state": "timeout",
                    "returncode": None,
                    "detail": _bounded(str(exc)),
                }
            )
            continue
        except OSError as exc:
            rows.append(
                {
                    "package": package,
                    "module": module,
                    "state": "error",
                    "returncode": None,
                    "detail": _bounded(str(exc)),
                }
            )
            continue
        state = "ok" if result.returncode == 0 else "failed"
        detail = _bounded((result.stderr or result.stdout).strip())
        rows.append(
            {
                "package": package,
                "module": module,
                "state": state,
                "returncode": result.returncode,
                "detail": detail,
            }
        )
    return rows


def _probe_pip(path_info: Mapping[str, object], *, repo: Path) -> dict[str, object]:
    path = Path(str(path_info["path"]))
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "LC_ALL": "C",
        "LANG": "C",
        "PYTHONPATH": str(repo),
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "WANDB_MODE": "disabled",
    }
    try:
        result = subprocess.run(
            [str(path), "-m", "pip", "check"],
            check=False,
            capture_output=True,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT_SECONDS,
            cwd=repo,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"state": "error", "returncode": None, "detail": _bounded(str(exc))}
    return {
        "state": "ok" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "detail": _bounded((result.stderr or result.stdout).strip()),
    }


def _classify_capabilities(
    entrypoints: Sequence[Mapping[str, object]],
    assets: Sequence[Mapping[str, object]],
    files: Sequence[Path],
    *,
    repo: Path,
) -> list[dict[str, object]]:
    evidence_text: dict[str, str] = {}
    for entrypoint in entrypoints:
        evidence_text[str(entrypoint["path"])] = " ".join(
            [
                str(entrypoint["path"]),
                *[str(item) for item in entrypoint.get("functions", [])],
                *[str(item) for item in entrypoint.get("cli_flags", [])],
            ]
        ).lower()
    for asset in assets:
        evidence_text[str(asset["path"])] = str(asset["path"]).lower()
    for path in files:
        relative = _relative(path, repo).lower()
        if relative not in evidence_text:
            evidence_text[relative] = relative
    rows: list[dict[str, object]] = []
    for capability, keywords in _CAPABILITY_KEYWORDS.items():
        evidence = sorted(
            path
            for path, text in evidence_text.items()
            if any(keyword in text for keyword in keywords)
        )
        state = "discovered" if evidence else "not_discovered"
        rows.append(
            {
                "capability": capability,
                "state": state,
                "confidence": "medium" if evidence else "none",
                "evidence": evidence[:20],
            }
        )
    return rows


def _load_evaluator_contract(path: Path | None) -> dict[str, object]:
    required = [
        "evaluator_id",
        "command",
        "input_artifacts",
        "output_artifacts",
        "metrics",
        "verifier",
    ]
    if path is None:
        return {
            "state": "binding_required",
            "source_path": None,
            "required_fields": required,
            "missing_fields": required,
        }
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        return {
            "state": "blocked",
            "source_path": str(resolved),
            "required_fields": required,
            "missing_fields": required,
            "error": "EVALUATOR_CONTRACT_NOT_FOUND",
        }
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "state": "blocked",
            "source_path": str(resolved),
            "required_fields": required,
            "missing_fields": required,
            "error": _bounded(str(exc)),
        }
    if not isinstance(payload, dict):
        return {
            "state": "blocked",
            "source_path": str(resolved),
            "required_fields": required,
            "missing_fields": required,
            "error": "EVALUATOR_CONTRACT_OBJECT_REQUIRED",
        }
    missing = [field for field in required if field not in payload]
    if missing:
        return {
            "state": "binding_required",
            "source_path": str(resolved),
            "required_fields": required,
            "missing_fields": missing,
            "error": "EVALUATOR_CONTRACT_FIELDS_MISSING",
        }
    invalid = _invalid_evaluator_fields(payload)
    if invalid:
        return {
            "state": "blocked",
            "source_path": str(resolved),
            "required_fields": required,
            "missing_fields": [],
            "error": "EVALUATOR_CONTRACT_FIELDS_INVALID:" + ",".join(invalid),
        }
    contract = {
        "state": "ready",
        "source_path": str(resolved),
        "required_fields": required,
        "missing_fields": [],
        "contract_sha256": _sha256_file(resolved),
        "evaluator_id": str(payload["evaluator_id"]),
        "command": payload["command"],
        "input_artifacts": payload["input_artifacts"],
        "output_artifacts": payload["output_artifacts"],
        "metrics": payload["metrics"],
        "verifier": payload["verifier"],
    }
    working_directory = payload.get("working_directory", ".")
    contract["working_directory"] = working_directory
    contract["conformance_imports"] = list(payload.get("conformance_imports", []))
    scheduler_template = payload.get("scheduler_template")
    if scheduler_template is not None:
        template_path = Path(str(scheduler_template))
        template_path = (
            template_path.resolve()
            if template_path.is_absolute()
            else (resolved.parent / template_path).resolve()
        )
        if not template_path.is_file() or template_path.is_symlink():
            return {
                "state": "blocked",
                "source_path": str(resolved),
                "required_fields": required,
                "missing_fields": [],
                "error": "EVALUATOR_SCHEDULER_TEMPLATE_INVALID",
            }
        contract["scheduler_template_path"] = str(template_path)
        contract["scheduler_template_sha256"] = _sha256_file(template_path)
    return contract


def _invalid_evaluator_fields(payload: Mapping[str, Any]) -> list[str]:
    invalid: list[str] = []
    if (
        not isinstance(payload.get("evaluator_id"), str)
        or not payload["evaluator_id"].strip()
    ):
        invalid.append("evaluator_id")
    if not _nonempty_string_list(payload.get("command")):
        invalid.append("command")
    for field in ("input_artifacts", "output_artifacts", "metrics"):
        if not _nonempty_string_list(payload.get(field)):
            invalid.append(field)
    if not isinstance(payload.get("verifier"), str) or not payload["verifier"].strip():
        invalid.append("verifier")
    if (
        not isinstance(payload.get("working_directory", "."), str)
        or not str(payload.get("working_directory", ".")).strip()
    ):
        invalid.append("working_directory")
    conformance_imports = payload.get("conformance_imports", [])
    if not isinstance(conformance_imports, list) or not all(
        isinstance(item, str) and item.strip() for item in conformance_imports
    ):
        invalid.append("conformance_imports")
    scheduler_template = payload.get("scheduler_template")
    if scheduler_template is not None and (
        not isinstance(scheduler_template, str) or not scheduler_template.strip()
    ):
        invalid.append("scheduler_template")
    return invalid


def _nonempty_string_list(value: object) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, str) and item.strip() for item in value)
    )


def _build_connector(
    repo: Path,
    runtime: Mapping[str, object],
    entrypoints: Sequence[Mapping[str, object]],
    capabilities: Sequence[Mapping[str, object]],
    evaluator_contract: Mapping[str, object],
    asset_bindings: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    grouped: dict[str, list[str]] = {
        "training": [],
        "evaluation": [],
        "rollout": [],
        "inference": [],
    }
    for entrypoint in entrypoints:
        if entrypoint.get("executable") is not True:
            continue
        for kind in entrypoint.get("kinds", []):
            if kind in grouped:
                grouped[kind].append(str(entrypoint["path"]))
    for paths in grouped.values():
        paths.sort()
    return {
        "state": (
            "ready"
            if runtime.get("state") == "ready"
            and evaluator_contract.get("state") == "ready"
            else "binding_required"
        ),
        "kind": "declarative",
        "source_repo": str(repo),
        "runtime_python": runtime.get("selected_python"),
        "entrypoints_by_kind": grouped,
        "capabilities": [
            item["capability"] for item in capabilities if item["state"] == "discovered"
        ],
        "asset_bindings": list(asset_bindings),
        "source_mutation_allowed": False,
        "generated_code": False,
        "manual_bindings": (
            ["evaluator_contract"] if evaluator_contract.get("state") != "ready" else []
        ),
    }


def _discover_asset_bindings(
    entrypoints: Sequence[Mapping[str, object]],
    assets: Sequence[Mapping[str, object]],
    *,
    repo: Path,
    explicit_bindings: Sequence[tuple[str, Path]],
    evaluator_contract: Mapping[str, object],
) -> list[dict[str, object]]:
    """Resolve input-like CLI paths against discovered repository assets."""

    flags: set[str] = set()
    for entrypoint in entrypoints:
        if entrypoint.get("kinds") == ["unknown"]:
            continue
        flags.update(str(flag) for flag in entrypoint.get("cli_flags", []))
    overrides = _validate_explicit_asset_bindings(explicit_bindings, flags)
    required_parameters = _evaluator_asset_parameters(evaluator_contract)
    rows: list[dict[str, object]] = []
    for flag in sorted(flags):
        field = flag.removeprefix("--").replace("-", "_").lower()
        if any(token in field for token in ("output", "save", "log", "cache", "port")):
            continue
        if not any(
            token in field for token in ("path", "root", "dir", "ckpt", "checkpoint")
        ):
            continue
        kind = _binding_kind(field)
        if kind is None:
            continue
        explicit = overrides.get(flag)
        evidence = (
            [str(explicit)]
            if explicit is not None
            else _matching_assets(field, kind, assets)
        )
        resolved = explicit or ((repo / evidence[0]).resolve() if evidence else None)
        rows.append(
            {
                "parameter": flag,
                "kind": kind,
                "state": "discovered" if evidence else "binding_required",
                "evidence": evidence,
                "binding_source": (
                    "explicit" if explicit is not None else "repository_discovery"
                ),
                "resolved_path": str(resolved) if resolved is not None else None,
                "asset_fingerprint": (
                    _asset_fingerprint(resolved) if resolved is not None else None
                ),
                "required_for_evaluator": flag in required_parameters,
            }
        )
    return rows


def _evaluator_asset_parameters(
    evaluator_contract: Mapping[str, object],
) -> set[str]:
    command = evaluator_contract.get("command")
    if evaluator_contract.get("state") != "ready" or not isinstance(command, list):
        return set()
    parameters: set[str] = set()
    for token in command:
        parameters.update(
            placeholder[len("{asset:") : -1]
            for placeholder in re.findall(r"\{asset:--[A-Za-z0-9_-]+\}", str(token))
        )
    return parameters


def _validate_explicit_asset_bindings(
    values: Sequence[tuple[str, Path]], discovered_flags: set[str]
) -> dict[str, Path]:
    bindings: dict[str, Path] = {}
    for parameter, raw_path in values:
        normalized = parameter if parameter.startswith("--") else f"--{parameter}"
        if normalized not in discovered_flags:
            raise OnboardingError(f"ASSET_BINDING_PARAMETER_UNKNOWN:{normalized}")
        if normalized in bindings:
            raise OnboardingError(f"ASSET_BINDING_DUPLICATE:{normalized}")
        raw = Path(raw_path).expanduser()
        if not raw.exists() or raw.is_symlink():
            raise OnboardingError(f"ASSET_BINDING_PATH_INVALID:{normalized}")
        path = raw.resolve()
        bindings[normalized] = path
    return bindings


def _asset_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    entries = (
        [path]
        if path.is_file()
        else sorted(
            item for item in path.rglob("*") if item.is_file() and not item.is_symlink()
        )
    )
    for item in entries[:_MAX_FILES]:
        stat = item.stat()
        relative = item.name if path.is_file() else item.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def _binding_kind(field: str) -> str | None:
    if "ckpt" in field or "checkpoint" in field:
        return "checkpoint"
    if "dataset" in field or "data_" in field or field.startswith("data"):
        return "dataset_or_metadata"
    if any(token in field for token in ("model", "policy", "vae", "clip", "svd")):
        return "model_dependency"
    return None


def _matching_assets(
    field: str, kind: str, assets: Sequence[Mapping[str, object]]
) -> list[str]:
    tokens = {
        token
        for token in re.split(r"[^a-z0-9]+", field)
        if token
        not in {
            "path",
            "root",
            "dir",
            "model",
            "checkpoint",
            "ckpt",
            "data",
            "dataset",
            "meta",
            "info",
        }
    }
    candidates = [asset for asset in assets if asset.get("kind") == kind]
    if kind == "checkpoint" and not tokens:
        candidates = sorted(
            candidates,
            key=lambda asset: (-int(asset.get("size_bytes", -1)), str(asset["path"])),
        )
        return [str(asset["path"]) for asset in candidates[:3]]
    matches = [
        str(asset["path"])
        for asset in candidates
        if tokens and any(token in str(asset["path"]).lower() for token in tokens)
    ]
    if kind == "dataset_or_metadata" and not matches and not tokens:
        matches = [str(asset["path"]) for asset in candidates]
    return sorted(matches)[:3]


def _build_blockers(
    *,
    source_revision: Mapping[str, object],
    runtime: Mapping[str, object],
    entrypoints: Sequence[Mapping[str, object]],
    assets: Sequence[Mapping[str, object]],
    capabilities: Sequence[Mapping[str, object]],
    evaluator_contract: Mapping[str, object],
    asset_bindings: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    if source_revision.get("state") != "bound":
        blockers.append(
            {
                "code": "SOURCE_REVISION_UNBOUND",
                "detail": source_revision.get("detail", "source revision is not bound"),
            }
        )
    if runtime.get("state") != "ready":
        blockers.append(
            {
                "code": "RUNTIME_UNREADY",
                "detail": "selected Python runtime is missing required imports or has dependency conflicts",
            }
        )
        for probe in runtime.get("import_probes", []):
            if probe.get("state") != "ok":
                blockers.append(
                    {
                        "code": (
                            "RUNTIME_DEPENDENCY_MISSING"
                            if probe.get("state") == "failed"
                            else "RUNTIME_DEPENDENCY_PROBE_FAILED"
                        ),
                        "package": probe.get("package"),
                        "detail": probe.get("detail", ""),
                    }
                )
        if runtime.get("pip_check", {}).get("state") == "failed":
            blockers.append({"code": "RUNTIME_DEPENDENCY_CONFLICT", "detail": runtime["pip_check"].get("detail", "")})  # type: ignore[index]
    if not entrypoints:
        blockers.append(
            {
                "code": "ENTRYPOINT_NOT_DISCOVERED",
                "detail": "no train, evaluation, inference, or rollout entrypoint was discovered",
            }
        )
    if not any(asset.get("kind") == "checkpoint" for asset in assets):
        blockers.append(
            {
                "code": "CHECKPOINT_MISSING",
                "detail": "no checkpoint or weight asset was discovered",
            }
        )
    if not any(
        entrypoint.get("executable") is True
        and "evaluation" in entrypoint.get("kinds", [])
        for entrypoint in entrypoints
    ):
        blockers.append(
            {
                "code": "EVALUATION_ENTRYPOINT_MISSING",
                "detail": "no evaluation entrypoint was discovered",
            }
        )
    if evaluator_contract.get("state") != "ready":
        blockers.append(
            {
                "code": "EVALUATOR_CONTRACT_REQUIRED",
                "detail": "a frozen evaluator contract is required before scheduling",
            }
        )
    for binding in asset_bindings:
        if (
            binding.get("required_for_evaluator") is True
            and binding.get("state") != "discovered"
        ):
            blockers.append(
                {
                    "code": "MODEL_ASSET_BINDING_REQUIRED",
                    "parameter": binding.get("parameter"),
                    "detail": f"no repository asset matches {binding.get('parameter')}",
                }
            )
    if not any(item.get("state") == "discovered" for item in capabilities):
        blockers.append(
            {
                "code": "CAPABILITY_NOT_DISCOVERED",
                "detail": "no supported model capability was evidenced",
            }
        )
    return _dedupe_blockers(blockers)


def _dedupe_blockers(
    blockers: Iterable[Mapping[str, object]]
) -> list[dict[str, object]]:
    rows = [dict(item) for item in blockers]
    rows.sort(
        key=lambda item: (
            str(item.get("code", "")),
            str(item.get("package", "")),
            str(item.get("detail", "")),
        )
    )
    unique: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in rows:
        key = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        if key not in seen:
            unique.append(row)
            seen.add(key)
    return unique


def _state(
    blockers: Sequence[Mapping[str, object]],
    *,
    runtime: Mapping[str, object],
    evaluator_contract: Mapping[str, object],
) -> str:
    if blockers:
        return "blocked"
    if runtime.get("state") == "ready" and evaluator_contract.get("state") == "ready":
        return "ready_for_conformance_smoke"
    return "binding_required"


def _conformance_report() -> dict[str, object]:
    return {
        "state": "not_run",
        "gpu_execution_started": False,
        "checks": [
            {"name": "runtime_import", "state": "pending"},
            {"name": "entrypoint_help", "state": "pending"},
            {"name": "evaluator_contract", "state": "pending"},
            {"name": "output_isolation", "state": "pending"},
        ],
        "receipt_path": None,
    }


def _next_actions(state: str, blockers: Sequence[Mapping[str, object]]) -> list[str]:
    actions: list[str] = []
    codes = {str(item.get("code")) for item in blockers}
    if "RUNTIME_UNREADY" in codes:
        actions.append(
            "Select or provision a Python runtime satisfying the discovered dependency files."
        )
    if "SOURCE_REVISION_UNBOUND" in codes:
        actions.append(
            "Bind the source to a Git commit before publishing reusable experiment evidence."
        )
    if "EVALUATOR_CONTRACT_REQUIRED" in codes:
        actions.append(
            "Provide a frozen evaluator contract with inputs, outputs, metrics, verifier, and command."
        )
    if "EVALUATION_ENTRYPOINT_MISSING" in codes:
        actions.append(
            "Declare an evaluation entrypoint; training or rollout alone cannot authorize optimization."
        )
    if state == "ready_for_conformance_smoke":
        actions.append(
            "Run the CPU-safe conformance smoke, then promote only its passing receipt to the scheduler."
        )
    if not actions:
        actions.append(
            "Review the generated connector and run conformance before any GPU experiment."
        )
    return actions


def _write_bundle(
    report: Mapping[str, object],
    destination: Path,
    *,
    repo: Path,
    source_revision: Mapping[str, object],
) -> dict[str, object]:
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=str(destination.parent))
    )
    try:
        report_bytes = _canonical_json_bytes(report)
        files = {
            "model_manifest.json": {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": "wmloop-model-manifest",
                "repo_root": str(repo),
                "repo_name": repo.name,
                "source_revision": source_revision,
                "entrypoint_count": len(report["entrypoints"]),  # type: ignore[arg-type]
                "asset_count": len(report["assets"]),  # type: ignore[arg-type]
            },
            "runtime_lock.json": report["runtime"],
            "asset_manifest.json": {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": "wmloop-asset-manifest",
                "assets": report["assets"],
            },
            "capability_report.json": {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": "wmloop-capability-report",
                "capabilities": report["capabilities"],
            },
            "evaluator_contract.json": report["evaluator_contract"],
            "conformance_report.json": report["conformance"],
            "generated_connector/connector.json": report["connector"],
        }
        for relative, payload in files.items():
            _write_bytes(temporary / relative, _canonical_json_bytes(payload))
        _write_bytes(temporary / "onboarding-report.json", report_bytes)
        _write_bytes(
            temporary / "onboarding-report.md", _render_markdown(report).encode("utf-8")
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "wmloop-model-onboarding-manifest",
            "repo_root": str(repo),
            "repo_name": repo.name,
            "state": report["state"],
            "optimization_launch_allowed": False,
            "blocker_count": len(report["blockers"]),  # type: ignore[arg-type]
            "report_path": str(destination / "onboarding-report.json"),
            "markdown_path": str(destination / "onboarding-report.md"),
            "source_revision": source_revision,
            "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
            "sidecar_root": str(destination),
            "source_output_written": False,
        }
        _write_bytes(temporary / "manifest.json", _canonical_json_bytes(manifest))
        os.replace(temporary, destination)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _render_markdown(report: Mapping[str, object]) -> str:
    lines = [
        "# VerdiWM Model Onboarding",
        "",
        f"Repository: `{report['repo_name']}`",
        f"State: `{report['state']}`",
        f"Optimization launch allowed: `{report['optimization_launch_allowed']}`",
        "",
        "## Blockers",
        "",
    ]
    blockers = list(report["blockers"])  # type: ignore[arg-type]
    lines.extend(
        f"- `{item.get('code')}`: {json.dumps(item, ensure_ascii=False, sort_keys=True)}"
        for item in blockers
    )
    if not blockers:
        lines.append("- none")
    lines.extend(
        ["", "## Entrypoints", "", "| Path | Kinds | Main guard |", "|:--|:--|:--|"]
    )
    for item in report["entrypoints"]:  # type: ignore[index]
        lines.append(
            f"| {item['path']} | {', '.join(item['kinds'])} | {item['main_guard']} |"
        )
    if not report["entrypoints"]:  # type: ignore[index]
        lines.append("| none | | |")
    lines.extend(
        [
            "",
            "## Capabilities",
            "",
            "| Capability | State | Evidence |",
            "|:--|:--|:--|",
        ]
    )
    for item in report["capabilities"]:  # type: ignore[index]
        evidence = ", ".join(item["evidence"][:3])
        lines.append(f"| {item['capability']} | {item['state']} | {evidence} |")
    lines.extend(["", "## Side Effects", ""])
    for name, value in report["side_effects"].items():  # type: ignore[union-attr]
        lines.append(f"- `{name}`: `{value}`")
    lines.extend(["", "## Next Actions", ""])
    lines.extend(f"- {item}" for item in report["next_actions"])  # type: ignore[index]
    lines.append("")
    return "\n".join(lines)


def _validate_report(report: Mapping[str, object]) -> None:
    try:
        validate_document("model_onboarding_report", report)
    except ContractValidationError as exc:
        raise OnboardingError("ONBOARDING_REPORT_INVALID") from exc


def _safe_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return -1


def _sha256_file(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _bounded(value: str, *, limit: int = _MAX_ERROR) -> str:
    value = value.replace("\x00", " ").strip()
    return value if len(value) <= limit else value[: limit - 3] + "..."


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


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
