"""Frozen H1--H5 hook contracts for controlled ACWM-Phys interventions.

The upstream checkout never imports this module.  A primitive renderer validates
against these contracts before producing a worktree patch, which keeps a
template from silently changing the provider's integration surface mid-campaign.
The contracts deliberately use tensor *specifications* rather than importing
PyTorch, so they remain testable before the heavyweight provider environment is
installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


class HookContractError(ValueError):
    """A primitive or rendered template violates a frozen hook boundary."""


HOOK_IDS = ("H1", "H2", "H3", "H4", "H5")


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtype: str

    def __post_init__(self) -> None:
        if not self.shape or any(not isinstance(size, int) or size <= 0 for size in self.shape):
            raise HookContractError("HOOK_TENSOR_SHAPE_INVALID")
        if not self.dtype:
            raise HookContractError("HOOK_TENSOR_DTYPE_INVALID")


@dataclass(frozen=True)
class HookContract:
    hook_id: str
    description: str


CONTRACTS = {
    "H1": HookContract("H1", "dataset/sampler trajectory sampling interface"),
    "H2": HookContract("H2", "DiT context injection: (latent_ctx, action_emb, extras) -> ctx"),
    "H3": HookContract("H3", "training-loss assembly: (pred, target, aux) -> loss_dict"),
    "H4": HookContract("H4", "inference sampler callback: (tokens, kv_cache) -> callback result"),
    "H5": HookContract("H5", "trainer optimizer/EMA/capacity configuration"),
}


def validate_manifest_hooks(hooks: tuple[str, ...], hook_order: Mapping[str, int]) -> None:
    """Validate a manifest's explicit frozen attachment declarations."""

    if not hooks or len(set(hooks)) != len(hooks) or any(hook not in CONTRACTS for hook in hooks):
        raise HookContractError("HOOK_MANIFEST_INVALID")
    if set(hook_order) != set(hooks) or any(not isinstance(order, int) or order < 0 for order in hook_order.values()):
        raise HookContractError("HOOK_MANIFEST_ORDER_INVALID")


def validate_combination_order(bindings: Mapping[str, list[int]]) -> None:
    """Require deterministic non-overlapping order whenever hooks are shared."""

    for hook, orders in bindings.items():
        if hook not in CONTRACTS or not orders or len(set(orders)) != len(orders):
            raise HookContractError("HOOK_COMBINATION_ORDER_INVALID")


def validate_hook_io(
    hook_id: str,
    *,
    inputs: Mapping[str, TensorSpec] | Mapping[str, object],
    outputs: Mapping[str, TensorSpec] | Mapping[str, object],
) -> None:
    """Check the stable shape/dtype invariants exposed by each tensor hook.

    H5 is a configuration hook and therefore has no tensor requirement; its
    presence is still checked so templates cannot invent a sixth integration
    point.  The validator is intentionally conservative: missing required
    fields or a dtype/batch mismatch always fails before sandbox creation.
    """

    if hook_id not in CONTRACTS:
        raise HookContractError("HOOK_UNKNOWN")
    if hook_id == "H5":
        if not isinstance(inputs, Mapping) or not isinstance(outputs, Mapping):
            raise HookContractError("HOOK_H5_CONFIGURATION_INVALID")
        return
    required = {
        "H1": (("trajectory",), ("trajectory",)),
        "H2": (("latent_ctx", "action_emb"), ("ctx",)),
        "H3": (("pred", "target"), ("loss",)),
        "H4": (("tokens",), ("tokens",)),
    }[hook_id]
    _require_specs(inputs, required[0])
    _require_specs(outputs, required[1])
    if hook_id == "H1":
        _same_shape_dtype(inputs["trajectory"], outputs["trajectory"])
    elif hook_id == "H2":
        latent = inputs["latent_ctx"]
        ctx = outputs["ctx"]
        if latent.dtype != ctx.dtype or latent.shape[0] != ctx.shape[0]:
            raise HookContractError("HOOK_H2_CONTEXT_SHAPE_OR_DTYPE_MISMATCH")
    elif hook_id == "H3":
        loss = outputs["loss"]
        if loss.shape != (1,) or not loss.dtype.startswith("float"):
            raise HookContractError("HOOK_H3_LOSS_SPEC_INVALID")
    elif hook_id == "H4":
        _same_shape_dtype(inputs["tokens"], outputs["tokens"])


def _require_specs(values: Mapping[str, TensorSpec] | Mapping[str, object], names: tuple[str, ...]) -> None:
    for name in names:
        value = values.get(name)
        if not isinstance(value, TensorSpec):
            raise HookContractError(f"HOOK_TENSOR_SPEC_MISSING:{name}")


def _same_shape_dtype(left: TensorSpec, right: TensorSpec) -> None:
    if left.shape != right.shape or left.dtype != right.dtype:
        raise HookContractError("HOOK_TENSOR_SHAPE_OR_DTYPE_MISMATCH")
