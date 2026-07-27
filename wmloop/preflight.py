"""Fail-closed ACWM asset and runtime preflight for real GPU launches."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from wmloop.acwm_data import AcwmEnvironmentSpec, CANONICAL_ACWM_ENVIRONMENTS, inspect_acwm_dataset
from wmloop.execute.gpu_exclusivity_audit import GpuExclusivityAuditError, verify_gpu_exclusivity_ready
from wmloop.runtime_env import runtime_subprocess_env

_RUNTIME_PACKAGES = (
    "torch",
    "torchvision",
    "diffusers",
    "transformers",
    "accelerate",
    "safetensors",
    "imageio",
    "imageio_ffmpeg",
    "decord",
    "av",
    "cv2",
    "numpy",
    "scipy",
    "skimage",
    "einops",
    "wandb",
    "tqdm",
    "omegaconf",
    "huggingface_hub",
)

_FULL_DATASET_SPLITS = ("ind_train", "ind_test", "ood_test")
_EVALUATION_SPLITS = ("ind_test", "ood_test")


@dataclass(frozen=True)
class EnvironmentAssetStatus:
    environment: str
    dataset_missing: tuple[str, ...]
    checkpoint_present: bool

    @property
    def ready(self) -> bool:
        return not self.dataset_missing and self.checkpoint_present

    @property
    def smoke_ready(self) -> bool:
        return self.checkpoint_present and "ind_test" not in self.dataset_missing


@dataclass(frozen=True)
class LaunchPreflight:
    data_root: str
    checkpoint_root: str
    runtime_python: str | None
    vae_present: bool
    runtime_missing: tuple[str, ...]
    runtime_integrity_errors: tuple[str, ...]
    required_splits: tuple[str, ...]
    environments: tuple[EnvironmentAssetStatus, ...]
    gpu_exclusivity_audit: dict[str, object] | None = None

    @property
    def ready(self) -> bool:
        return (
            self.vae_present
            and not self.runtime_missing
            and not self.runtime_integrity_errors
            and all(item.ready for item in self.environments)
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "ready": self.ready,
            "data_root": self.data_root,
            "checkpoint_root": self.checkpoint_root,
            "runtime_python": self.runtime_python,
            "vae_present": self.vae_present,
            "runtime_missing": list(self.runtime_missing),
            "runtime_integrity_errors": list(self.runtime_integrity_errors),
            "required_splits": list(self.required_splits),
            "gpu_exclusivity_audit": self.gpu_exclusivity_audit,
            "environments": [
                {**asdict(item), "ready": item.ready, "smoke_ready": item.smoke_ready} for item in self.environments
            ],
        }


@dataclass(frozen=True)
class RuntimeCandidateStatus:
    runtime_python: str
    exists: bool
    runtime_missing: tuple[str, ...]
    runtime_integrity_errors: tuple[str, ...]
    gpu_exclusivity_audit: dict[str, object] | None = None

    @property
    def ready(self) -> bool:
        return self.exists and not self.runtime_missing and not self.runtime_integrity_errors

    def to_dict(self) -> dict[str, object]:
        return {
            "runtime_python": self.runtime_python,
            "exists": self.exists,
            "ready": self.ready,
            "runtime_missing": list(self.runtime_missing),
            "runtime_integrity_errors": list(self.runtime_integrity_errors),
            "gpu_exclusivity_audit": self.gpu_exclusivity_audit,
        }


def inspect_launch_assets(
    data_root: Path,
    checkpoint_root: Path,
    *,
    runtime_packages: Iterable[str] = _RUNTIME_PACKAGES,
    runtime_python: Path | None = None,
    environment_specs: Iterable[AcwmEnvironmentSpec] = CANONICAL_ACWM_ENVIRONMENTS,
    required_splits: Iterable[str] = _FULL_DATASET_SPLITS,
    gpu_index: int | None = None,
    gpu_exclusivity_audit_manifest: Path | None = None,
    gpu_exclusivity_max_age_seconds: float | None = 300.0,
) -> LaunchPreflight:
    data = Path(data_root)
    checkpoints = Path(checkpoint_root)
    specs = tuple(environment_specs)
    scope = _validate_required_splits(required_splits, specs=specs)
    inventory = inspect_acwm_dataset(data, environment_specs=specs)
    statuses: list[EnvironmentAssetStatus] = []
    for spec in specs:
        missing = tuple(
            split
            for split, _ in spec.split_sizes
            if split in scope and not inventory.split(spec.environment, split).ready
        )
        statuses.append(
            EnvironmentAssetStatus(
                environment=spec.environment,
                dataset_missing=missing,
                checkpoint_present=(checkpoints / spec.checkpoint_relative_path).is_file(),
            )
        )
    packages = tuple(runtime_packages)
    runtime = inspect_runtime_candidate(
        runtime_python,
        runtime_packages=packages,
        gpu_index=gpu_index,
        gpu_exclusivity_audit_manifest=gpu_exclusivity_audit_manifest,
        gpu_exclusivity_max_age_seconds=gpu_exclusivity_max_age_seconds,
    )
    return LaunchPreflight(
        data_root=str(data.resolve()),
        checkpoint_root=str(checkpoints.resolve()),
        runtime_python=str(Path(runtime_python).resolve()) if runtime_python is not None else None,
        vae_present=(checkpoints / "Wan2.1_VAE.pth").is_file(),
        runtime_missing=runtime.runtime_missing,
        runtime_integrity_errors=runtime.runtime_integrity_errors,
        required_splits=scope,
        environments=tuple(statuses),
        gpu_exclusivity_audit=runtime.gpu_exclusivity_audit,
    )


def inspect_runtime_candidate(
    runtime_python: Path | None,
    *,
    runtime_packages: Iterable[str] = _RUNTIME_PACKAGES,
    gpu_index: int | None = None,
    gpu_exclusivity_audit_manifest: Path | None = None,
    gpu_exclusivity_max_age_seconds: float | None = 300.0,
) -> RuntimeCandidateStatus:
    """Check one runtime interpreter without touching ACWM data or checkpoints."""

    packages = tuple(runtime_packages)
    if runtime_python is None:
        missing = _runtime_missing(packages, None)
        integrity, gpu_exclusivity = _runtime_integrity_check(
            None,
            check_cuda="torch" in packages,
            gpu_index=gpu_index,
            gpu_exclusivity_audit_manifest=gpu_exclusivity_audit_manifest,
            gpu_exclusivity_max_age_seconds=gpu_exclusivity_max_age_seconds,
        )
        return RuntimeCandidateStatus(
            runtime_python="python",
            exists=True,
            runtime_missing=missing,
            runtime_integrity_errors=integrity,
            gpu_exclusivity_audit=gpu_exclusivity,
        )
    path = Path(runtime_python)
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        return RuntimeCandidateStatus(
            runtime_python=str(path),
            exists=False,
            runtime_missing=("RUNTIME_PYTHON_MISSING",),
            runtime_integrity_errors=("RUNTIME_PYTHON_MISSING",),
            gpu_exclusivity_audit=None,
        )
    if not resolved.is_file():
        return RuntimeCandidateStatus(
            runtime_python=str(resolved),
            exists=False,
            runtime_missing=("RUNTIME_PYTHON_MISSING",),
            runtime_integrity_errors=("RUNTIME_PYTHON_MISSING",),
            gpu_exclusivity_audit=None,
        )
    missing = _runtime_missing(packages, resolved)
    integrity, gpu_exclusivity = _runtime_integrity_check(
        resolved,
        check_cuda="torch" in packages,
        gpu_index=gpu_index,
        gpu_exclusivity_audit_manifest=gpu_exclusivity_audit_manifest,
        gpu_exclusivity_max_age_seconds=gpu_exclusivity_max_age_seconds,
    )
    return RuntimeCandidateStatus(
        runtime_python=str(resolved),
        exists=True,
        runtime_missing=missing,
        runtime_integrity_errors=integrity,
        gpu_exclusivity_audit=gpu_exclusivity,
    )


def discover_runtime_candidates(
    candidates: Iterable[Path],
    *,
    runtime_packages: Iterable[str] = _RUNTIME_PACKAGES,
    gpu_index: int | None = None,
    gpu_exclusivity_audit_manifest: Path | None = None,
    gpu_exclusivity_max_age_seconds: float | None = 300.0,
) -> tuple[RuntimeCandidateStatus, ...]:
    """Evaluate candidate Python runtimes in order and keep every failure reason."""

    packages = tuple(runtime_packages)
    return tuple(
        inspect_runtime_candidate(
            path,
            runtime_packages=packages,
            gpu_index=gpu_index,
            gpu_exclusivity_audit_manifest=gpu_exclusivity_audit_manifest,
            gpu_exclusivity_max_age_seconds=gpu_exclusivity_max_age_seconds,
        )
        for path in _unique_candidate_paths(candidates)
    )


def select_ready_runtime(candidates: Iterable[RuntimeCandidateStatus]) -> RuntimeCandidateStatus | None:
    """Return the first candidate that passed package, pip, and CUDA gates."""

    for candidate in candidates:
        if candidate.ready:
            return candidate
    return None


def default_runtime_candidates(*, repo_root: Path | None = None) -> tuple[Path, ...]:
    """Return conservative local candidates before a human supplies a path.

    The project is commonly run on shared hosts with multiple conda roots.  The
    list is intentionally small and ordered toward project-local runtimes before
    global envs; every candidate still has to pass the same runtime gates.
    """

    root = Path(repo_root).resolve() if repo_root is not None else Path.cwd().resolve()
    owner_root = root.parent
    return _unique_candidate_paths(
        (
            owner_root / ".conda-envs" / "eva" / "bin" / "python3.10",
            owner_root / ".conda-envs" / "eva" / "bin" / "python",
            root / ".venv" / "bin" / "python",
            Path.home() / "miniconda3" / "envs" / "acwm-phys-clean" / "bin" / "python",
            Path.home() / "miniconda3" / "envs" / "acwm-phys" / "bin" / "python",
            Path.home() / "miniconda3" / "envs" / "wm-eval" / "bin" / "python",
            Path("/opt/conda/envs/ctrl-world/bin/python"),
        )
    )


def three_gpu_waves(preflight: LaunchPreflight, *, gpus: tuple[int, ...] = (0, 1, 2)) -> tuple[tuple[str, ...], ...]:
    """Produce deterministic waves only after the full experiment is admissible."""

    if not preflight.ready or len(gpus) != 3 or len(set(gpus)) != 3:
        raise ValueError("THREE_GPU_LAUNCH_PRECHECK_FAILED")
    names = [item.environment for item in preflight.environments]
    return tuple(tuple(names[index : index + len(gpus)]) for index in range(0, len(names), len(gpus)))


def _runtime_missing(packages: tuple[str, ...], runtime_python: Path | None) -> tuple[str, ...]:
    source = (
        "import importlib,json,sys\n"
        "missing=[]\n"
        "for name in sys.argv[1:]:\n"
        "    try:\n"
        "        importlib.import_module(name)\n"
        "    except Exception:\n"
        "        missing.append(name)\n"
        "print(json.dumps(missing))\n"
    )
    if runtime_python is None:
        completed = subprocess.run(["python", "-c", source, *packages], check=False, capture_output=True, text=True, timeout=30)
        if completed.returncode != 0:
            return packages
        try:
            return tuple(str(package) for package in json.loads(completed.stdout))
        except json.JSONDecodeError:
            return packages
    interpreter = Path(runtime_python).resolve()
    if not interpreter.is_file():
        return (*packages, "RUNTIME_PYTHON_MISSING")
    completed = subprocess.run(
        [str(interpreter), "-c", source, *packages],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
        env=runtime_subprocess_env(interpreter),
    )
    if completed.returncode != 0:
        return (*packages, "RUNTIME_PYTHON_UNUSABLE")
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return (*packages, "RUNTIME_PYTHON_UNUSABLE")
    return tuple(str(package) for package in result)


def _unique_candidate_paths(candidates: Iterable[Path]) -> tuple[Path, ...]:
    result: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        path = Path(candidate)
        try:
            key = str(path.resolve(strict=True))
        except OSError:
            key = str(path)
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return tuple(result)


def _runtime_integrity_check(
    runtime_python: Path | None,
    *,
    check_cuda: bool = False,
    gpu_index: int | None = None,
    gpu_exclusivity_audit_manifest: Path | None = None,
    gpu_exclusivity_max_age_seconds: float | None = 300.0,
) -> tuple[tuple[str, ...], dict[str, object] | None]:
    """Expose packaging conflicts as a launch gate, not a post-crash surprise."""

    if runtime_python is None:
        return (), None
    interpreter = Path(runtime_python).resolve()
    if not interpreter.is_file():
        return ("RUNTIME_PYTHON_MISSING",), None
    env = runtime_subprocess_env(interpreter)
    errors: list[str] = []
    completed = subprocess.run(
        [str(interpreter), "-m", "pip", "check"], check=False, capture_output=True, text=True, timeout=60, env=env
    )
    if completed.returncode != 0:
        errors.append("RUNTIME_PIP_CHECK_FAILED")
    gpu_exclusivity: dict[str, object] | None = None
    if check_cuda:
        if gpu_index is None:
            errors.append("RUNTIME_GPU_INDEX_REQUIRED")
            return tuple(errors), None
        if not isinstance(gpu_index, int) or isinstance(gpu_index, bool) or gpu_index < 0:
            errors.append("RUNTIME_GPU_INDEX_INVALID")
            return tuple(errors), None
        try:
            gpu_exclusivity = verify_gpu_exclusivity_ready(
                gpu_exclusivity_audit_manifest,
                gpu_index=gpu_index,
                max_age_seconds=gpu_exclusivity_max_age_seconds,
            )
        except GpuExclusivityAuditError as exc:
            errors.append(str(exc))
            return tuple(errors), None
        source = (
            "import torch\n"
            "if torch.cuda.device_count() < 1:\n"
            "    raise SystemExit(2)\n"
            "x = torch.ones(1, device='cuda')\n"
            "torch.cuda.synchronize()\n"
            "raise SystemExit(0 if float(x.cpu()[0]) == 1.0 else 3)\n"
        )
        cuda = subprocess.run(
            [str(interpreter), "-c", source],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
            env=runtime_subprocess_env(interpreter, extra={"CUDA_VISIBLE_DEVICES": str(gpu_index)}),
        )
        if cuda.returncode != 0:
            errors.append("RUNTIME_CUDA_KERNEL_FAILED")
    return tuple(errors), gpu_exclusivity


def _validate_required_splits(
    required_splits: Iterable[str], *, specs: tuple[AcwmEnvironmentSpec, ...]
) -> tuple[str, ...]:
    scope = tuple(str(split) for split in required_splits)
    if not scope or len(set(scope)) != len(scope):
        raise ValueError("PREFLIGHT_SCOPE_INVALID")
    for spec in specs:
        if set(scope) - {split for split, _ in spec.split_sizes}:
            raise ValueError("PREFLIGHT_SCOPE_INVALID")
    return scope


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--runtime-python", type=Path)
    parser.add_argument("--discover-runtime", action="store_true")
    parser.add_argument("--candidate-runtime-python", type=Path, action="append")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--scope", choices=("full", "evaluation"), default="full")
    parser.add_argument("--gpu-index", type=int)
    parser.add_argument("--gpu-exclusivity-audit-manifest", type=Path)
    parser.add_argument("--gpu-exclusivity-max-age-seconds", type=float, default=300.0)
    args = parser.parse_args(argv)
    required_splits = _FULL_DATASET_SPLITS if args.scope == "full" else _EVALUATION_SPLITS
    if args.discover_runtime:
        paths = tuple(args.candidate_runtime_python or ()) or default_runtime_candidates(repo_root=args.repo_root)
        candidates = discover_runtime_candidates(
            paths,
            gpu_index=args.gpu_index,
            gpu_exclusivity_audit_manifest=args.gpu_exclusivity_audit_manifest,
            gpu_exclusivity_max_age_seconds=args.gpu_exclusivity_max_age_seconds,
        )
        selected = select_ready_runtime(candidates)
        launch_preflight = (
            inspect_launch_assets(
                args.data_root,
                args.checkpoint_root,
                runtime_python=Path(selected.runtime_python),
                required_splits=required_splits,
                gpu_index=args.gpu_index,
                gpu_exclusivity_audit_manifest=args.gpu_exclusivity_audit_manifest,
                gpu_exclusivity_max_age_seconds=args.gpu_exclusivity_max_age_seconds,
            ).to_dict()
            if selected is not None
            else None
        )
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "artifact_type": "acwm-runtime-discovery",
                    "selected_runtime_python": selected.runtime_python if selected is not None else None,
                    "candidates": [candidate.to_dict() for candidate in candidates],
                    "launch_preflight": launch_preflight,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    print(
        json.dumps(
            inspect_launch_assets(
                args.data_root,
                args.checkpoint_root,
                runtime_python=args.runtime_python,
                required_splits=required_splits,
                gpu_index=args.gpu_index,
                gpu_exclusivity_audit_manifest=args.gpu_exclusivity_audit_manifest,
                gpu_exclusivity_max_age_seconds=args.gpu_exclusivity_max_age_seconds,
            ).to_dict(),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
