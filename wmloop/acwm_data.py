"""Read-only ACWM-Phys dataset inventory with exact split cardinality checks.

The upstream data loader accepts a directory as soon as ``metadata.pt`` is
present, which is convenient for development but unsafe for a baseline: a
partially resumed Hugging Face download otherwise looks launchable.  This
module encodes the published dataset layout without importing or modifying the
frozen upstream checkout.  It never loads tensors or videos.
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable


class AcwmDataInventoryError(ValueError):
    """The dataset tree contains an unsafe or non-canonical member."""


def _normal_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("ACWM_DATASET_PATH_INVALID")
    return path.as_posix()


@dataclass(frozen=True)
class AcwmEnvironmentSpec:
    """One immutable ACWM-Phys environment, its assets, and split sizes."""

    environment: str
    dataset_relative_path: str
    checkpoint_relative_path: str
    split_sizes: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        if not self.environment or not self.checkpoint_relative_path:
            raise ValueError("ACWM_DATASET_SPEC_INVALID")
        _normal_relative_path(self.dataset_relative_path)
        names = [name for name, size in self.split_sizes]
        if len(set(names)) != len(names) or any(not name or size < 1 for name, size in self.split_sizes):
            raise ValueError("ACWM_DATASET_SPEC_INVALID")


CANONICAL_ACWM_ENVIRONMENTS = (
    AcwmEnvironmentSpec(
        "push_cube",
        "rigid_dynamics/push_block",
        "VideoDiT_S_push_cube_240x240/latest.pt",
        (("ind_train", 1500), ("ind_test", 50), ("ood_test", 50)),
    ),
    AcwmEnvironmentSpec(
        "stack_cube",
        "rigid_dynamics/stack_cube",
        "VideoDiT_S_stack_cube_240x240/latest.pt",
        (("ind_train", 1373), ("ind_test", 67), ("ood_test", 271)),
    ),
    AcwmEnvironmentSpec(
        "push_rope",
        "deformable/push_rope",
        "VideoDiT_S_push_rope_240x240/latest.pt",
        (("ind_train", 1500), ("ind_test", 50), ("ood_test", 50)),
    ),
    AcwmEnvironmentSpec(
        "cloth_move",
        "deformable/clothmove",
        "VideoDiT_S_clothmove_240x240_240x240/latest.pt",
        (("ind_train", 2000), ("ind_test", 100), ("ood_test", 100)),
    ),
    AcwmEnvironmentSpec(
        "push_sand",
        "particle/push_sand",
        "VideoDiT_S_push_sand_240x400/latest.pt",
        (("ind_train", 1784), ("ind_test", 100), ("ood_test", 100)),
    ),
    AcwmEnvironmentSpec(
        "pour_water",
        "particle/pour_water",
        "VideoDiT_S_pour_water_240x240/latest.pt",
        (("ind_train", 1000), ("ind_test", 50), ("ood_test", 50)),
    ),
    AcwmEnvironmentSpec(
        "robot_arm",
        "kinematics/robot_arm_64",
        "VideoDiT_S_robot_arm_240x240/latest.pt",
        (("ind_train", 2002), ("ind_test", 105), ("ood_test", 105)),
    ),
    AcwmEnvironmentSpec(
        "reacher",
        "kinematics/reacher",
        "VideoDiT_S_reacher_240x240/latest.pt",
        (("ind_train", 1987), ("ind_test", 100), ("ood_test", 100)),
    ),
)


@dataclass(frozen=True)
class AcwmSplitInventory:
    environment: str
    split: str
    expected_episodes: int
    episode_names: tuple[str, ...]
    metadata_present: bool

    @property
    def actual_episodes(self) -> int:
        return len(self.episode_names)

    @property
    def ready(self) -> bool:
        return self.metadata_present and self.actual_episodes == self.expected_episodes


@dataclass(frozen=True)
class AcwmDataInventory:
    root: Path
    root_trusted: bool
    splits: tuple[AcwmSplitInventory, ...]

    @property
    def ready(self) -> bool:
        return self.root_trusted and all(item.ready for item in self.splits)

    def ready_for(self, required_splits: Iterable[str]) -> bool:
        """Whether a named workload has every requested split available."""

        names = _required_split_names(required_splits)
        return self.root_trusted and all(item.ready for item in self.splits if item.split in names)

    def split(self, environment: str, split: str) -> AcwmSplitInventory:
        for item in self.splits:
            if item.environment == environment and item.split == split:
                return item
        raise KeyError((environment, split))


def inspect_acwm_dataset(
    data_root: Path,
    *,
    environment_specs: Iterable[AcwmEnvironmentSpec] = CANONICAL_ACWM_ENVIRONMENTS,
) -> AcwmDataInventory:
    """Inventory canonical direct children without treating a partial sync as ready.

    Missing paths are represented as unavailable.  Symlinks or other unsafe
    filesystem identities raise instead of being followed, so callers cannot
    freeze or launch against a tree that may change underneath them.
    """

    specs = _validated_specs(environment_specs)
    root = Path(data_root)
    if not _is_trusted_directory(root):
        return AcwmDataInventory(
            root=root,
            root_trusted=False,
            splits=tuple(
                AcwmSplitInventory(spec.environment, split, expected, (), False)
                for spec in specs
                for split, expected in spec.split_sizes
            ),
        )
    resolved_root = root.resolve(strict=True)
    inventories: list[AcwmSplitInventory] = []
    for spec in specs:
        environment_root = _child_directory(resolved_root, spec.dataset_relative_path)
        for split, expected in spec.split_sizes:
            split_root = environment_root / split if environment_root is not None else None
            if split_root is None or not _is_trusted_directory(split_root):
                inventories.append(AcwmSplitInventory(spec.environment, split, expected, (), False))
                continue
            names = _episode_names(split_root, spec.environment, split)
            inventories.append(
                AcwmSplitInventory(
                    spec.environment,
                    split,
                    expected,
                    names,
                    _is_trusted_regular_file(split_root / "metadata.pt"),
                )
            )
    return AcwmDataInventory(root=resolved_root, root_trusted=True, splits=tuple(inventories))


def canonical_dataset_relative_files(
    inventory: AcwmDataInventory,
    *,
    environment_specs: Iterable[AcwmEnvironmentSpec] = CANONICAL_ACWM_ENVIRONMENTS,
    required_splits: Iterable[str] | None = None,
) -> tuple[str, ...]:
    """Return every regular file that belongs in a freeze, after completeness.

    This deliberately has no glob over arbitrary data-root files: cache files
    and Hugging Face bookkeeping must not affect scientific provenance.
    """

    specs = _validated_specs(environment_specs)
    names = _required_split_names(
        required_splits if required_splits is not None else (split for split, _ in specs[0].split_sizes)
    )
    if any(set(names) - {split for split, _ in spec.split_sizes} for spec in specs):
        raise AcwmDataInventoryError("ACWM_DATASET_SCOPE_INVALID")
    if not inventory.ready_for(names):
        raise AcwmDataInventoryError("ACWM_DATASET_INCOMPLETE")
    files: list[str] = []
    for spec in specs:
        base = _normal_relative_path(spec.dataset_relative_path)
        for split, _ in spec.split_sizes:
            if split not in names:
                continue
            status = inventory.split(spec.environment, split)
            if not status.ready:
                raise AcwmDataInventoryError(f"ACWM_DATASET_INCOMPLETE:{spec.environment}:{split}")
            files.append(f"{base}/{split}/metadata.pt")
            files.extend(f"{base}/{split}/{name}" for name in status.episode_names)
    return tuple(sorted(files))


def _validated_specs(specs: Iterable[AcwmEnvironmentSpec]) -> tuple[AcwmEnvironmentSpec, ...]:
    result = tuple(specs)
    if not result or len({item.environment for item in result}) != len(result):
        raise ValueError("ACWM_DATASET_SPECS_INVALID")
    return result


def _required_split_names(required_splits: Iterable[str]) -> tuple[str, ...]:
    names = tuple(str(name) for name in required_splits)
    if not names or len(set(names)) != len(names) or any(not name for name in names):
        raise AcwmDataInventoryError("ACWM_DATASET_SCOPE_INVALID")
    return names


def _child_directory(root: Path, relative_path: str) -> Path | None:
    candidate = root
    for part in PurePosixPath(_normal_relative_path(relative_path)).parts:
        candidate = candidate / part
        try:
            metadata = os.lstat(candidate)
        except FileNotFoundError:
            return None
        if stat.S_ISLNK(metadata.st_mode):
            raise AcwmDataInventoryError(f"ACWM_DATASET_SYMLINK:{relative_path}")
        if not stat.S_ISDIR(metadata.st_mode):
            return None
    return candidate


def _episode_names(split_root: Path, environment: str, split: str) -> tuple[str, ...]:
    names: list[str] = []
    for candidate in split_root.iterdir():
        if candidate.suffix != ".mp4":
            continue
        if candidate.is_symlink():
            raise AcwmDataInventoryError(f"ACWM_DATASET_SYMLINK:{environment}:{split}:{candidate.name}")
        try:
            metadata = os.lstat(candidate)
        except FileNotFoundError as exc:
            raise AcwmDataInventoryError(f"ACWM_DATASET_MEMBER_CHANGED:{environment}:{split}:{candidate.name}") from exc
        if not stat.S_ISREG(metadata.st_mode):
            raise AcwmDataInventoryError(f"ACWM_DATASET_MEMBER_NOT_REGULAR:{environment}:{split}:{candidate.name}")
        names.append(candidate.name)
    return tuple(sorted(names))


def _is_trusted_directory(path: Path) -> bool:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return False
    return stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode)


def _is_trusted_regular_file(path: Path) -> bool:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(metadata.st_mode):
        raise AcwmDataInventoryError(f"ACWM_DATASET_SYMLINK:metadata:{path.name}")
    return stat.S_ISREG(metadata.st_mode)
