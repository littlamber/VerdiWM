"""Execution-role contract for ACWM primitive routing."""

from __future__ import annotations


INVALIDATED_QUALITY_PRIMITIVES = frozenset({"latent_motion_prior"})


QUALITY_SCREEN_PRIMITIVES = frozenset(
    {
        "cfg_guidance_schedule",
        "action_contrastive_finetune",
        "action_dimension_balancing",
        "dino_rep_injection",
        "event_window_reweight",
        "drift_token_trim",
        "first_frame_anchor",
        "history_noise_schedule",
        "inv_dyn_reward_finetune",
        "latent_spatial_memory",
        "mixture_reweight",
        "motion_region_reweight",
        "next_forcing",
        "self_forcing_finetune",
        "wmsd_self_distill",
    }
)

RUNTIME_ONLY_PRIMITIVES = frozenset(
    {
        "cfg_guidance_schedule",
        "drift_token_trim",
        "first_frame_anchor",
        "latent_spatial_memory",
    }
)

TRAINING_QUALITY_SCREEN_PRIMITIVES = QUALITY_SCREEN_PRIMITIVES - RUNTIME_ONLY_PRIMITIVES

DIAGNOSTIC_ROUTING_PRIMITIVES = frozenset({"frontier_collection"})


def primitive_execution_role(primitive: str) -> str:
    if primitive in INVALIDATED_QUALITY_PRIMITIVES:
        return "invalidated_quality"
    if primitive in RUNTIME_ONLY_PRIMITIVES:
        return "runtime_only"
    if primitive in QUALITY_SCREEN_PRIMITIVES:
        return "quality_screen"
    if primitive in DIAGNOSTIC_ROUTING_PRIMITIVES:
        return "diagnostic_routing"
    return "unsupported"
