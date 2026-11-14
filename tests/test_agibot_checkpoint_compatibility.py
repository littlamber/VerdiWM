from __future__ import annotations

from wmloop.execute._agibot_checkpoint_compatibility import (
    compare_agibot_checkpoint_architecture,
)


def test_exact_checkpoint_architecture_is_accepted():
    shapes = {
        "patch_embedding.weight": (3072, 48, 1, 2, 2),
        "blocks.0.attn.q.weight": (3072, 3072),
    }

    report = compare_agibot_checkpoint_architecture(
        shapes, shapes, history_ref_enabled=False
    )

    assert report.compatible
    assert report.summary() == (
        "missing=0; unexpected=0; shape_mismatch=0; compatible_expansions=0"
    )


def test_native_optional_action_keys_may_be_missing_but_backbone_keys_may_not():
    target = {
        "action_encoder.domain_prompt.weight": (10, 1, 3072),
        "blocks.0.attn.q.weight": (3072, 3072),
    }

    optional_missing = compare_agibot_checkpoint_architecture(
        {"blocks.0.attn.q.weight": (3072, 3072)},
        target,
        history_ref_enabled=False,
    )
    backbone_missing = compare_agibot_checkpoint_architecture(
        {"action_encoder.domain_prompt.weight": (10, 1, 3072)},
        target,
        history_ref_enabled=False,
    )

    assert optional_missing.compatible
    assert not backbone_missing.compatible
    assert backbone_missing.missing == ("blocks.0.attn.q.weight",)


def test_domain_delta_and_patch_input_channel_expansions_match_native_rules():
    target = {
        "action_encoder.group_delta_weight": (10, 4, 3072),
        "patch_embedding.weight": (3072, 55, 1, 2, 2),
    }
    checkpoint = {
        "action_encoder.group_delta_weight": (8, 4, 3072),
        "patch_embedding.weight": (3072, 48, 1, 2, 2),
    }

    report = compare_agibot_checkpoint_architecture(
        checkpoint, target, history_ref_enabled=False
    )

    assert report.compatible
    assert report.compatible_expansions == tuple(sorted(target))


def test_only_the_complete_history_reference_branch_may_be_missing_in_h1():
    target = {
        "blocks.0.attn.q.weight": (8, 8),
        "history_reference.encoder.weight": (8, 8),
        "history_reference.output.weight": (8, 8),
    }
    exact_backbone = {"blocks.0.attn.q.weight": (8, 8)}

    full_branch_missing = compare_agibot_checkpoint_architecture(
        exact_backbone, target, history_ref_enabled=True
    )
    partial_branch_missing = compare_agibot_checkpoint_architecture(
        {
            **exact_backbone,
            "history_reference.encoder.weight": (8, 8),
        },
        target,
        history_ref_enabled=True,
    )
    expanded_action = compare_agibot_checkpoint_architecture(
        {
            "blocks.0.attn.q.weight": (8, 8),
            "history_reference.encoder.weight": (8, 8),
            "history_reference.output.weight": (8, 8),
            "action_encoder.group_delta_weight": (2, 4),
        },
        {
            **target,
            "action_encoder.group_delta_weight": (3, 4),
        },
        history_ref_enabled=True,
    )

    assert full_branch_missing.compatible
    assert not partial_branch_missing.compatible
    assert not expanded_action.compatible
