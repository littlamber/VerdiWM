#!/usr/bin/env python3
"""Execute the admitted Ctrl-World observation ABI against a request file.

The adapter is intentionally narrow: it runs the existing inference-only
action-dose probe and emits a path-free observation response. It does not
modify source, evaluator, metrics, or promotion state.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


def _load(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("OBSERVATION_REQUEST_INVALID")
    return payload


def _required(bindings: dict[str, object], key: str) -> str:
    value = bindings.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError("OBSERVATION_RUNTIME_BINDING_MISSING:" + key)
    return value


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        raise SystemExit("usage: run_ctrl_world_observation_adapter.py REQUEST RESPONSE")
    request_path = Path(argv[1]).resolve(strict=True)
    response_path = Path(argv[2]).resolve()
    request = _load(request_path)
    bindings = request.get("runtime_bindings")
    if not isinstance(bindings, dict):
        raise ValueError("OBSERVATION_RUNTIME_BINDINGS_INVALID")
    requirement = request.get("probe_requirement")
    if not isinstance(requirement, dict):
        raise ValueError("OBSERVATION_PROBE_REQUIREMENT_INVALID")

    output_root = request_path.parent / "ctrl-world-probe-output"
    if output_root.exists():
        raise ValueError("OBSERVATION_PROBE_OUTPUT_EXISTS")
    output_root.mkdir(mode=0o700)
    episodes = _required(bindings, "episodes_json")
    doses = [str(value) for value in requirement.get("dose_values", [])]
    command = [
        _required(bindings, "runtime_python"),
        _required(bindings, "probe_script"),
        "--ctrl-world-root", _required(bindings, "ctrl_world_root"),
        "--dataset-root", _required(bindings, "dataset_root"),
        "--data-stat", _required(bindings, "data_stat"),
        "--svd-model-path", _required(bindings, "svd_model_path"),
        "--clip-model-path", _required(bindings, "clip_model_path"),
        "--ckpt-path", _required(bindings, "ckpt_path"),
        "--episodes-json", episodes,
        "--probe-id", "action_conditioning_scale",
        "--doses", *doses,
        "--interact-num", str(int(bindings.get("interact_num", 2))),
        "--num-inference-steps", str(int(bindings.get("num_inference_steps", 2))),
        "--output-root", str(output_root),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    result_path = output_root / "result.json"
    if completed.returncode != 0 or not result_path.is_file():
        response = {
            "schema_version": 1,
            "artifact_type": "verdiwm-observation-execution-result",
            "state": "blocked",
            "task_id": request.get("task_id"),
            "abi_id": request.get("abi_id"),
            "observation_kind": "none",
            "probe_observation": None,
            "structural_observation": None,
            "blockers": [{"code": "CTRL_WORLD_PROBE_FAILED", "exit_code": completed.returncode}],
            "authority": {"source_mutated": False, "intervention_executed": False, "active_metric_mutated": False, "active_evaluator_mutated": False, "verdict_exposed": False, "promotion_authority": False},
            "claim_boundary": "Blocked diagnostic observation; no scientific claim.",
        }
    else:
        result = _load(result_path)
        raw = result_path.read_bytes()
        response = {
            "schema_version": 1,
            "artifact_type": "verdiwm-observation-execution-result",
            "state": "completed",
            "task_id": request.get("task_id"),
            "abi_id": request.get("abi_id"),
            "observation_kind": "probe_fingerprint",
            "probe_observation": {
                "probe_protocol_id": requirement["probe_protocol_id"],
                "probe_protocol_version": requirement["probe_protocol_version"],
                "diagnostic_role": requirement["diagnostic_role"],
                "context_class": requirement["context_class"],
                "split": requirement["split"],
                "horizons": list(requirement["horizons"]),
                "dose_values": list(requirement["dose_values"]),
                "replication_count": int(result.get("metrics", {}).get("episode_count", 0)),
                "response_dimension": 3,
                "response_summary": "real Ctrl-World action-conditioning dose response retained in observation payload",
                "uncertainty_summary": "bounded frozen replay diagnostic",
                "response_payload": {"metrics": result.get("metrics"), "dose_responses": result.get("dose_responses")},
            },
            "structural_observation": None,
            "blockers": [],
            "authority": {"source_mutated": False, "intervention_executed": False, "active_metric_mutated": False, "active_evaluator_mutated": False, "verdict_exposed": False, "promotion_authority": False},
            "claim_boundary": "Inference-only target-local diagnostic; no repair, training, or promotion claim.",
        }
    response_path.write_text(json.dumps(response, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
