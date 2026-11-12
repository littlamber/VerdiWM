"""Read-only onboarding scan components: source."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Sequence

from wmloop.control.onboarding_scan.files import _relative
from wmloop.control.onboarding_scan.io import _bounded, _safe_size
from wmloop.control.onboarding_scan.types import (
    _DEPENDENCY_FILENAMES,
    _DEPENDENCY_PREFIXES,
    _ENTRYPOINT_KEYWORDS,
    _MAX_DEPENDENCIES,
    _MAX_ENTRYPOINTS,
    _MAX_IMPORTS,
    _MAX_SOURCE_PARSE_BYTES,
)


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
            try:
                import tomllib
            except ModuleNotFoundError:  # Python 3.10 support
                import tomli as tomllib

            payload = tomllib.loads(text)
            values = payload.get("project", {}).get("dependencies", [])
            if isinstance(values, list):
                for value in values:
                    match = re.match(r"([A-Za-z0-9][A-Za-z0-9_.-]*)", str(value))
                    if match:
                        packages.add(match.group(1).lower())
        except (ImportError, ValueError, TypeError, AttributeError):
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
