"""Model-independent training admission rules for experiment stages."""

from __future__ import annotations

import math
from collections.abc import Mapping


class TrainingContractError(ValueError):
    """A runner binding would make a stage scientifically ambiguous."""


_REQUIREMENTS = {
    "screen": {"mode": "long", "min_steps": 64, "min_records": 4, "seeds": 1, "min_panel": 1},
    "gate": {"mode": "long", "min_steps": 256, "min_records": 8, "seeds": 3, "min_panel": 3},
    # The scale planner calls this stage ``pilot`` while the generic queue
    # calls the same formal admission boundary ``gate``. Keep the alias here
    # so model adapters can consume either vocabulary without weakening the
    # requirements.
    "pilot": {"mode": "long", "min_steps": 256, "min_records": 8, "seeds": 3, "min_panel": 3},
    "confirm": {"mode": "long", "min_steps": 1024, "min_records": 16, "seeds": 3, "min_panel": 3},
}


def minimum_training_episode_count(expected_stage: str) -> int:
    """Return the minimum distinct train episodes for a formal stage."""

    minimums = {"screen": 4, "gate": 8, "pilot": 8, "confirm": 16}
    try:
        return minimums[expected_stage]
    except KeyError as exc:
        raise TrainingContractError("TRAINING_CONTRACT_STAGE_INVALID") from exc


def validate_training_binding(
    binding: Mapping[str, object], *, expected_stage: str
) -> dict[str, object]:
    """Validate and normalize the scheduler-to-runner training contract.

    The scheduler never interprets model-specific flags.  It only requires a
    runner to receive this portable contract (normally as environment
    variables) and rejects a formal stage that is still configured as a probe,
    one-window run, or single-seed run.
    """

    if expected_stage not in _REQUIREMENTS:
        raise TrainingContractError("TRAINING_CONTRACT_STAGE_INVALID")
    required = (
        "train_manifest",
        "validation_manifest",
        "mode",
        "steps",
        "record_limit",
        "sampler",
        "seed_count",
        "scale_plan_sha256",
        "runner_contract",
    )
    missing = [key for key in required if key not in binding]
    if missing:
        raise TrainingContractError(
            "TRAINING_CONTRACT_FIELDS_MISSING:" + ",".join(missing)
        )
    if not all(isinstance(binding[key], str) and str(binding[key]).strip() for key in ("train_manifest", "validation_manifest", "scale_plan_sha256", "runner_contract")):
        raise TrainingContractError("TRAINING_CONTRACT_PATH_OR_DIGEST_INVALID")
    digest = str(binding["scale_plan_sha256"])
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise TrainingContractError("TRAINING_CONTRACT_SCALE_PLAN_DIGEST_INVALID")
    mode = binding["mode"]
    sampler = binding["sampler"]
    if mode != "long":
        raise TrainingContractError(
            f"TRAINING_CONTRACT_PROBE_NOT_ALLOWED:{expected_stage}"
        )
    if sampler != "episode_balanced":
        raise TrainingContractError("TRAINING_CONTRACT_EPISODE_BALANCED_REQUIRED")
    steps = _positive_int(binding.get("steps"), "TRAINING_CONTRACT_STEPS_INVALID")
    record_limit = _nonnegative_int(
        binding.get("record_limit"), "TRAINING_CONTRACT_RECORD_LIMIT_INVALID"
    )
    seed_count = _positive_int(
        binding.get("seed_count"), "TRAINING_CONTRACT_SEED_COUNT_INVALID"
    )
    panel_size_value = binding.get("validation_panel_size")
    if panel_size_value is None:
        if expected_stage in {"gate", "pilot", "confirm"}:
            raise TrainingContractError("TRAINING_CONTRACT_VALIDATION_PANEL_REQUIRED")
        panel_size = 1
    else:
        panel_size = _positive_int(panel_size_value, "TRAINING_CONTRACT_VALIDATION_PANEL_SIZE_INVALID")
    requirement = _REQUIREMENTS[expected_stage]
    if steps < requirement["min_steps"]:
        raise TrainingContractError(
            f"TRAINING_CONTRACT_STEPS_BELOW_{expected_stage.upper()}_FLOOR"
        )
    if record_limit and record_limit < requirement["min_records"]:
        raise TrainingContractError(
            f"TRAINING_CONTRACT_RECORD_DIVERSITY_BELOW_{expected_stage.upper()}_FLOOR"
        )
    if seed_count < requirement["seeds"]:
        raise TrainingContractError(
            f"TRAINING_CONTRACT_SEED_COUNT_BELOW_{expected_stage.upper()}_FLOOR"
        )
    if panel_size < requirement["min_panel"]:
        raise TrainingContractError(
            f"TRAINING_CONTRACT_VALIDATION_PANEL_BELOW_{expected_stage.upper()}_FLOOR"
        )
    return {
        "schema_version": 1,
        "runner_contract": str(binding["runner_contract"]),
        "expected_stage": expected_stage,
        "train_manifest": str(binding["train_manifest"]),
        "validation_manifest": str(binding["validation_manifest"]),
        "scale_plan_sha256": digest,
        "mode": mode,
        "steps": steps,
        "record_limit": record_limit,
        "sampler": sampler,
        "seed_count": seed_count,
        "validation_panel_size": panel_size,
        "screen_failure_veto": False,
        "claim_boundary": (
            "Training admission proves only that the declared budget and data "
            "coverage are suitable for this stage; it does not prove quality."
        ),
    }


def _positive_int(value: object, code: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise TrainingContractError(code)
    return value


def _nonnegative_int(value: object, code: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TrainingContractError(code)
    return value
