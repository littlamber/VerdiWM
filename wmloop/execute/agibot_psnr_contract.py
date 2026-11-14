"""Execution boundary for Agibot paired-PSNR effect measurements.

The protocol-v3 runner currently executes the native Agibot training and
checkpoint reload diagnostic only.  That diagnostic does not produce the
paired RGB rollouts required by the frozen PSNR evaluator.  This module keeps
that distinction explicit at the execution boundary so every caller can
report the capability state without manufacturing a measurement or verdict.
"""

from __future__ import annotations

from collections.abc import Mapping


AGIBOT_PSNR_ROLLOUT_BUILDER_UNAVAILABLE = (
    "AGIBOT_PAIRED_PSNR_ROLLOUT_BUILDER_UNAVAILABLE"
)


def build_agibot_psnr_execution_status(
    manifest: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return the current paired-PSNR execution capability.

    ``manifest`` is accepted for forward-compatible callers that already have
    an execution manifest.  It is deliberately not inspected yet: no checked
    in builder can turn an Agibot checkpoint into the matched baseline and
    candidate RGB rollouts required by the frozen evaluator.  Reporting that
    fact is safer than treating a native checkpoint reload as a quality run.
    """

    del manifest
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-agibot-psnr-execution-contract",
        "state": "unavailable",
        "reason": AGIBOT_PSNR_ROLLOUT_BUILDER_UNAVAILABLE,
        "quality_eligible": False,
        "claim_boundary": (
            "Agibot native training and checkpoint diagnostics do not execute "
            "paired RGB rollouts; no PSNR measurement or model-effect verdict "
            "is established."
        ),
        "required_artifacts": [
            "baseline_paired_rgb_rollout",
            "candidate_paired_rgb_rollout",
            "frozen_agibot_psnr_evaluator",
            "paired_seed_cohort_at_least_3",
        ],
    }
