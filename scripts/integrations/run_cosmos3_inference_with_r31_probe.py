#!/usr/bin/env python3
"""Run official Cosmos3 inference with the exact r31 action2llm probe installed."""

from __future__ import annotations

import argparse
import hashlib
import json
import runpy
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from wmloop.primitives.adapters.cosmos3_hooks import (
    Cosmos3CPBEActionEmbeddingDeltaDose,
    cosmos3_probe_dose_unit,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dose", type=float, required=True)
    parser.add_argument("--action-input", type=Path, required=True)
    parser.add_argument("--runtime-receipt", type=Path, required=True)
    parser.add_argument("official_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    official_args = _eager_official_args(args.official_args)

    from cosmos_framework.model.vfm.mot.cosmos3_vfm_network import Cosmos3VFMNetwork

    original = Cosmos3VFMNetwork._encode_action
    aggregate: dict[str, Any] = {
        "encode_action_call_count": 0,
        "embedding_hook_invocation_count": 0,
        "maximum_temporal_mean_abs_error": 0.0,
        "maximum_temporal_mean_tolerance": 0.0,
        "observed_token_counts": [],
    }

    def patched_encode_action(
        model: object,
        packed_seq: object,
        packed_sequence: object,
        target_dtype: object,
        fps_action: object | None = None,
    ) -> object:
        action = getattr(packed_seq, "action", None)
        token_shapes = getattr(action, "token_shapes", None)
        if (
            not isinstance(token_shapes, Sequence)
            or isinstance(token_shapes, (str, bytes))
            or len(token_shapes) != 1
        ):
            raise RuntimeError("COSMOS3_CPBE_SINGLE_TRAJECTORY_REQUIRED")
        expected = int(token_shapes[0][0])
        aggregate["observed_token_counts"].append(expected)
        with Cosmos3CPBEActionEmbeddingDeltaDose(
            model, dose=args.dose, expected_token_count=expected
        ) as context:
            result = original(
                model,
                packed_seq,
                packed_sequence,
                target_dtype,
                fps_action=fps_action,
            )
        aggregate["encode_action_call_count"] += 1
        aggregate["embedding_hook_invocation_count"] += int(context.audit["invocation_count"])
        aggregate["maximum_temporal_mean_abs_error"] = max(
            float(aggregate["maximum_temporal_mean_abs_error"]),
            float(context.audit["maximum_temporal_mean_abs_error"]),
        )
        aggregate["maximum_temporal_mean_tolerance"] = max(
            float(aggregate["maximum_temporal_mean_tolerance"]),
            float(context.audit["maximum_temporal_mean_tolerance"]),
        )
        return result

    Cosmos3VFMNetwork._encode_action = patched_encode_action
    state = "failed"
    error: str | None = None
    exit_code = 0
    previous_argv = sys.argv
    try:
        sys.argv = ["cosmos_framework.scripts.inference", *official_args]
        try:
            runpy.run_module("cosmos_framework.scripts.inference", run_name="__main__")
        except SystemExit as exc:
            exit_code = int(exc.code or 0)
            if exit_code != 0:
                raise
        if int(aggregate["embedding_hook_invocation_count"]) < 1:
            raise RuntimeError("COSMOS3_CPBE_HOOK_NOT_INVOKED")
        if float(aggregate["maximum_temporal_mean_abs_error"]) > float(
            aggregate["maximum_temporal_mean_tolerance"]
        ):
            raise RuntimeError("COSMOS3_CPBE_MEAN_PRESERVATION_FAILED")
        state = "passed"
    except BaseException as exc:
        exit_code = int(getattr(exc, "code", 1) or 1)
        error = f"{type(exc).__name__}:{exc}"
        raise
    finally:
        sys.argv = previous_argv
        Cosmos3VFMNetwork._encode_action = original
        action = json.loads(args.action_input.read_text(encoding="utf-8"))
        receipt = {
            "schema_version": 1,
            "artifact_type": "verdiwm-cosmos3-action-embedding-hook-receipt",
            "state": state,
            "mode": "cpbe_residual_63f088b0d5",
            "parameters": {"dose": float(args.dose)},
            "dose_unit": cosmos3_probe_dose_unit("cpbe_residual_63f088b0d5"),
            "shape": [len(action), len(action[0])],
            "output_sha256": _sha256(args.action_input),
            "runtime_hook": "Cosmos3VFMNetwork._encode_action/action2llm.forward",
            "runtime_compile": {
                "torch_compile_enabled": False,
                "reason": "The diagnostic hook performs Python audit bookkeeping around action2llm and is intentionally executed outside Cosmos3's fullgraph _encode_action boundary.",
            },
            "semantic_program": {
                "signal_source": "action_embedding_delta",
                "temporal_basis": "event_phase_tangent",
                "contrast_operator": "signed_mean_preserving_phase",
            },
            "audit": aggregate,
            "official_exit_code": exit_code,
            "error": error,
            "claim_boundary": "Runtime hook evidence only; not model quality or transfer evidence.",
        }
        args.runtime_receipt.parent.mkdir(parents=True, exist_ok=True)
        args.runtime_receipt.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return exit_code


def _eager_official_args(values: Sequence[str]) -> list[str]:
    official_args = list(values)
    if official_args and official_args[0] == "--":
        official_args.pop(0)
    if not official_args:
        raise ValueError("COSMOS3_CPBE_OFFICIAL_ARGS_MISSING")
    if "--use-torch-compile" in official_args:
        raise ValueError("COSMOS3_CPBE_TORCH_COMPILE_UNSUPPORTED")
    if "--no-use-torch-compile" not in official_args:
        official_args.insert(0, "--no-use-torch-compile")
    return official_args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
