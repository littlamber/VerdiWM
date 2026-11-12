"""Repository layout rules shared by CI and onboarding diagnostics.

This module reports structural violations without importing any optional model,
GPU, web, or network dependency.  It is intentionally a read-only audit: the
audit never renames files or edits a checkout.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class LayoutIssue:
    code: str
    path: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "path": self.path, "detail": self.detail}


_TOP_LEVEL_DIRS = {
    "configs",
    "docs",
    "examples",
    "experiments",
    "figures",
    "ops",
    "scripts",
    "tests",
    "wmloop",
}
_WMLOOP_ROOT_FILES = {
    "__init__.py",
    "acwm_data.py",
    "cli.py",
    "constitution.py",
    "contracts.py",
    "dataset_freeze.py",
    "freeze.py",
    "limited_campaign_pilot.py",
    "orchestrator.py",
    "orchestrator_sandbox_smoke.py",
    "orchestrator_smoke.py",
    "orchestrator_training_eval_smoke.py",
    "orchestrator_training_smoke.py",
    "preflight.py",
    "repository_layout.py",
    "runtime_contract.py",
    "runtime_env.py",
    "storage.py",
    "training_view.py",
    "training_view_runtime.py",
    "vendor.py",
    "wan22_droid.py",
}
_WMLOOP_DOMAINS = {
    "archive",
    "control",
    "diagnose",
    "evaluate",
    "execute",
    "experiments",
    "geometry",
    "primitives",
    "propose",
    "retrieve",
    "verify",
    "web",
}
_SCRIPT_PREFIXES = {
    "scripts": (
        "aggregate_",
        "audit_",
        "check_",
        "compare_",
        "consolidate_",
        "evaluate_",
        "fetch_",
        "freeze_",
        "materialize_",
        "merge_",
        "prepare_",
        "recover_",
        "recompare_",
        "run_",
        "settle_",
        "summarize_",
        "verify_",
    ),
    "scripts/ci": ("check_",),
    "scripts/integrations": ("run_",),
    "scripts/export": (
        "acwm_",
        "build_",
        "cosmos3_",
        "ctrl_world_",
        "merge_",
        "paper_",
        "probe_",
        "progressive_",
        "r31_",
        "stage_",
        "validate_",
        "verdiwm_",
    ),
}
_EXPERIMENT_NAME = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*_v[0-9]+$")

# These compatibility entry points may import components, never the reverse.
_COMPONENT_FACADES = {
    "wmloop.control.commands": "wmloop.cli",
    "wmloop.control.onboarding_scan": "wmloop.control.onboarding",
    "wmloop.execute.training_smoke": "wmloop.orchestrator_training_eval_smoke",
    "wmloop.execute.pipeline": "wmloop.execute.autonomous_pipeline",
    "wmloop.retrieve.discovery": "wmloop.retrieve.mechanism_discovery",
    "wmloop.experiments.knowledge": "wmloop.experiments.portable_knowledge_graph",
}
_COMPONENT_LINE_LIMIT = 800


def audit_repository_layout(repo_root: Path) -> tuple[LayoutIssue, ...]:
    """Return deterministic layout issues for a source checkout."""

    root = Path(repo_root).resolve()
    issues: list[LayoutIssue] = []
    for path in sorted(root.iterdir(), key=lambda value: value.name):
        if path.name.startswith(".") or path.name in {
            "build",
            "cache",
            "dist",
            "artifacts",
            "logs",
            "outputs",
            "results",
            "runs",
        }:
            continue
        if path.is_dir() and path.name not in _TOP_LEVEL_DIRS:
            issues.append(
                LayoutIssue(
                    "TOP_LEVEL_DIRECTORY_UNKNOWN",
                    path.name,
                    "move generated or domain code under a declared top-level directory",
                )
            )

    wmloop = root / "wmloop"
    for path in sorted(wmloop.iterdir(), key=lambda value: value.name):
        if path.name.startswith("__") or path.name == "__pycache__":
            continue
        if (
            path.is_file()
            and path.suffix == ".py"
            and path.name not in _WMLOOP_ROOT_FILES
        ):
            issues.append(
                LayoutIssue(
                    "WMLOOP_ROOT_MODULE_UNKNOWN",
                    path.relative_to(root).as_posix(),
                    "put domain implementation under a named wmloop domain package",
                )
            )
        if (
            path.is_dir()
            and path.name not in _WMLOOP_DOMAINS
            and path.name != "__pycache__"
        ):
            issues.append(
                LayoutIssue(
                    "WMLOOP_DOMAIN_UNKNOWN",
                    path.relative_to(root).as_posix(),
                    "use an existing domain package or document a new bounded plane",
                )
            )

    for relative, prefixes in _SCRIPT_PREFIXES.items():
        directory = root / relative
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.py"), key=lambda value: value.name):
            if path.name == "__init__.py" or path.name.startswith(prefixes):
                continue
            issues.append(
                LayoutIssue(
                    "SCRIPT_NAME_INVALID",
                    path.relative_to(root).as_posix(),
                    f"expected one of: {', '.join(prefixes)}",
                )
            )

    experiments = root / "experiments"
    if experiments.is_dir():
        for path in sorted(experiments.iterdir(), key=lambda value: value.name):
            if (
                path.is_dir()
                and path.name != "__pycache__"
                and not _EXPERIMENT_NAME.fullmatch(path.name)
            ):
                issues.append(
                    LayoutIssue(
                        "EXPERIMENT_DIRECTORY_NAME_INVALID",
                        path.relative_to(root).as_posix(),
                        "use <domain>_<purpose>_v<integer>",
                    )
                )
    issues.extend(audit_component_dependencies(root))
    return tuple(issues)


def _import_targets(tree: ast.AST, module: str) -> set[str]:
    targets: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = ".".join(module.split(".")[: -node.level])
                base = ".".join(part for part in (base, node.module) if part)
            else:
                base = node.module or ""
            targets.add(base)
            targets.update(f"{base}.{alias.name}" for alias in node.names)
    return targets


def audit_component_dependencies(repo_root: Path) -> tuple[LayoutIssue, ...]:
    """Enforce one-way extraction boundaries without importing business code."""
    root = Path(repo_root).resolve()
    issues: list[LayoutIssue] = []
    for package, facade in _COMPONENT_FACADES.items():
        directory = root / package.replace(".", "/")
        graph: dict[str, set[str]] = {}
        paths: dict[str, str] = {}
        for path in sorted(directory.rglob("*.py")):
            relative = path.relative_to(root)
            module = ".".join(relative.with_suffix("").parts)
            source = path.read_text(encoding="utf-8")
            targets = _import_targets(ast.parse(source), module)
            graph[module] = targets
            paths[module] = relative.as_posix()
            if facade in targets:
                issues.append(
                    LayoutIssue(
                        "COMPONENT_IMPORTS_FACADE",
                        paths[module],
                        f"inject dependencies or import a lower-level component instead of {facade}",
                    )
                )
            if len(source.splitlines()) > _COMPONENT_LINE_LIMIT:
                issues.append(
                    LayoutIssue(
                        "COMPONENT_TOO_LARGE",
                        paths[module],
                        f"split responsibilities before exceeding {_COMPONENT_LINE_LIMIT} lines",
                    )
                )
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(module: str) -> None:
            if module in visited:
                return
            if module in visiting:
                issues.append(
                    LayoutIssue(
                        "COMPONENT_IMPORT_CYCLE",
                        paths[module],
                        "component imports must form an acyclic dependency graph",
                    )
                )
                return
            visiting.add(module)
            for dependency in sorted(graph[module] & graph.keys()):
                visit(dependency)
            visiting.remove(module)
            visited.add(module)

        for module in graph:
            visit(module)
    # Installed libraries must not import checkout-only script implementations.
    for path in sorted((root / "wmloop").rglob("*.py")):
        relative = path.relative_to(root)
        module = ".".join(relative.with_suffix("").parts)
        targets = _import_targets(ast.parse(path.read_text(encoding="utf-8")), module)
        forbidden = sorted(target for target in targets if target == "scripts" or target.startswith("scripts."))
        if forbidden:
            issues.append(LayoutIssue("LIBRARY_IMPORTS_SCRIPT", relative.as_posix(), ", ".join(forbidden)))
    # Pure geometry and retrieval cannot acquire execution or GPU authority.
    for domain in ("geometry", "retrieve"):
        for path in sorted((root / "wmloop" / domain).rglob("*.py")):
            relative = path.relative_to(root)
            module = ".".join(relative.with_suffix("").parts)
            targets = _import_targets(
                ast.parse(path.read_text(encoding="utf-8")), module
            )
            forbidden = sorted(
                target
                for target in targets
                if target == "torch"
                or target.startswith(("torch.", "wmloop.execute."))
                or target == "wmloop.execute"
            )
            if forbidden:
                issues.append(
                    LayoutIssue(
                        "DOMAIN_IMPORTS_EXECUTION",
                        relative.as_posix(),
                        ", ".join(forbidden),
                    )
                )
    return tuple(issues)


def large_python_modules(
    repo_root: Path, *, byte_limit: int = 50_000
) -> tuple[dict[str, object], ...]:
    """Return refactor candidates without treating legacy size as a CI failure."""

    root = Path(repo_root).resolve()
    rows = []
    for path in sorted((root / "wmloop").rglob("*.py")):
        size = path.stat().st_size
        if size >= byte_limit:
            rows.append({"path": path.relative_to(root).as_posix(), "bytes": size})
    return tuple(sorted(rows, key=lambda row: (-int(row["bytes"]), str(row["path"]))))
