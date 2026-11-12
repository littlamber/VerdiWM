"""Read-only onboarding scan components: types."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

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
        "swanlog",
        "runs",
        "outputs",
        "checkpoints",
        ".cache",
    }
)


_ARTIFACT_DIRECTORY_PREFIXES = ("video_latent", "latent_cache", "checkpoint")


def _is_artifact_directory(name: str, *, top_level: bool = False) -> bool:
    """Identify common local rollout/training artifact directories early."""

    lowered = name.casefold()
    return lowered.startswith(_ARTIFACT_DIRECTORY_PREFIXES) or (
        top_level and name.isdigit() and len(name) >= 4
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


_PIP_PLATFORM_WARNING = re.compile(
    r"^[A-Za-z0-9_.-]+\s+\S+\s+is not supported on this platform$",
    re.IGNORECASE,
)


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
    additional_asset_parameters: tuple[str, ...] = ()
    adapter_contract_ready: bool = False
    probe_imports: bool = True
    max_files: int = _MAX_FILES
