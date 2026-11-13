"""Read-only onboarding scan components: files."""

from __future__ import annotations

from pathlib import Path

from wmloop.control.onboarding_scan.types import (
    _MAX_FILES,
    _SKIP_DIRECTORIES,
    OnboardingError,
    _is_artifact_directory,
)


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
    except OSError as exc:  # pragma: no cover - Path parents are pure, retained for fail-closed behavior
        raise OnboardingError("ONBOARDING_OUTPUT_INVALID") from exc
    if inside:
        raise OnboardingError("ONBOARDING_OUTPUT_INSIDE_SOURCE")
    if destination.parent == repo.parent and destination.name == repo.name:
        raise OnboardingError("ONBOARDING_OUTPUT_INVALID")
    return destination


def _discover_files(
    repo: Path, *, max_files: int, reject_symlinks: bool = False,
) -> tuple[list[Path], bool]:
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
                if reject_symlinks:
                    raise OnboardingError("SOURCE_REVISION_SYMLINK_FORBIDDEN")
                continue
            if child.is_dir():
                if (
                    child.name not in _SKIP_DIRECTORIES
                    and not child.name.startswith(".")
                    and not _is_artifact_directory(
                        child.name, top_level=directory == repo
                    )
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
