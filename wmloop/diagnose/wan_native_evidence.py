"""Pure validation of the native Wan manifest and invocation binding.

Both execution and archived diagnostic replay use these checks. No model,
local input directory, lease or settlement is accessed here.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Mapping, Sequence


def wan_manifest_errors(
    manifest: Mapping[str, object], binding: Mapping[str, object],
    selected: Sequence[Mapping[str, object]],
) -> list[str]:
    """Check the semantic output contract without requiring the original host."""
    try:
        expected = {
            "config": binding["config"], "checkpoint": binding["checkpoint"],
            "base_model": False, "video_latent_root": binding["data_root"],
            "condition_view": "arm_slot_eef_pose", "split": binding["split"],
            "seed": binding["seed"], "inference_steps": binding["inference_steps"],
            "max_chunks": binding["max_chunks"],
            "condition_override_root": binding["condition_override_root"] or "",
        }
        if any(type(manifest.get(key)) is not type(value) or manifest[key] != value
               for key, value in expected.items()):
            return ["WAN_ARM_SLOT_MANIFEST_BINDING_MISMATCH"]
        rows = manifest.get("samples")
        limit = binding["limit"]
        if type(limit) is not int or limit < 1:
            return ["WAN_ARM_SLOT_SAMPLE_COVERAGE_MISMATCH"]
        expected_rows = selected[:limit]
        if not expected_rows or not isinstance(rows, list) or len(rows) != len(expected_rows):
            return ["WAN_ARM_SLOT_SAMPLE_COVERAGE_MISMATCH"]
        for index, (row, selection) in enumerate(zip(rows, expected_rows)):
            if (not isinstance(row, dict) or not isinstance(selection, Mapping)
                or type(row.get("selection_sample_index")) is not int
                or row["selection_sample_index"] != selection.get("sample_index", index)
                or any(type(row.get(key)) is not int or type(selection.get(key)) is not int
                       or row[key] != selection[key] for key in ("episode_id", "frame_now"))
                or type(row.get("chunks")) is not int or not 0 < row["chunks"] <= binding["max_chunks"]
                or type(row.get("frames")) is not int or row["frames"] <= 1
                or type(row.get("fps")) not in (int, float) or row["fps"] != binding["fps"]):
                return ["WAN_ARM_SLOT_SAMPLE_BINDING_MISMATCH"]
            directory = row.get("dir")
            if not _relative(directory):
                return ["WAN_ARM_SLOT_VIDEO_PATH_INVALID"]
            for key in ("gt_video", "sa_wm_video", "comparison_video"):
                name = row.get(key + "_file")
                if (not _relative(name) or PurePosixPath(name).name != name
                    or row.get(key) != str(PurePosixPath(str(binding["output_root"])) / "rollouts" / directory / name)):
                    return ["WAN_ARM_SLOT_VIDEO_PATH_INVALID"]
    except (AttributeError, KeyError, TypeError, ValueError):
        return ["WAN_ARM_SLOT_OUTPUT_CONTRACT_FAILED"]
    return []


def _relative(value: object) -> bool:
    return (isinstance(value, str) and bool(value) and "\\" not in value
            and not PurePosixPath(value).is_absolute() and ".." not in PurePosixPath(value).parts)


def wan_invocation_matches(process: Mapping[str, object], binding: Mapping[str, object]) -> bool:
    """Require the recorded invocation to name the inputs checked by manifest."""
    command = process.get("command")
    if not isinstance(command, list) or not all(isinstance(value, str) for value in command):
        return False
    try:
        if len(command) < 2 or command[0] != binding["runtime_python"]:
            return False
        if command[1] != str(PurePosixPath(str(binding["source_root"])) / "scripts/evaluation/rollout_full_episode_sa_wm.py"):
            return False
        flags = {
            "config": "--config", "checkpoint": "--checkpoint", "selection": "--selection",
            "data_root": "--video_latent_root", "split": "--split", "seed": "--seed",
            "inference_steps": "--inference_steps", "max_chunks": "--max_chunks", "limit": "--limit",
        }
        expected = {flag: str(binding[key]) for key, flag in flags.items()}
        expected.update({
            "--device": "cuda:0", "--decode_chunk_latents": "8",
            "--output_dir": str(PurePosixPath(str(binding["output_root"])) / "rollouts"),
        })
        if binding["condition_override_root"] is not None:
            expected["--condition_override_root"] = binding["condition_override_root"]
        tokens = command[2:]
        if len(tokens) != 2 * (len(expected) + 1):  # One separately numeric --fps.
            return False
        supplied = {}
        for index in range(0, len(tokens), 2):
            flag, value = tokens[index:index + 2]
            if flag in supplied or flag not in {*expected, "--fps"}:
                return False
            supplied[flag] = value
        # Native parse_known_args treats unknown tokens as config overrides.
        # Consume the complete finite invocation, including device/decode, so
        # an extra dose override or an argparse abbreviation cannot be hidden.
        return (all(supplied.get(flag) == value for flag, value in expected.items())
                and float(supplied["--fps"]) == binding["fps"])
    except (KeyError, ValueError, TypeError, IndexError):
        return False
