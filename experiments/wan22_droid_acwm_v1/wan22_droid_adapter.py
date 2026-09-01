"""WAN2.2-DROID conditioning adapter.

This file contains the model-facing shape contract only.  The actual WAN2.2
DiT hook is injected by the selected runtime runner; keeping this projection
separate prevents a Wan2.1 adapter from being loaded by name or by accident.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Wan22DroidConditioning:
    """A validated batch of aligned robot conditions."""

    actions: Sequence[Sequence[float]]
    proprio: Sequence[Sequence[float]]

    def __post_init__(self) -> None:
        if not self.actions or not self.proprio or len(self.actions) != len(self.proprio):
            raise ValueError("WAN22_DROID_CONDITIONING_LENGTH_INVALID")
        if any(len(row) != 7 for row in self.actions):
            raise ValueError("WAN22_DROID_ACTION_DIM_INVALID")
        if any(len(row) != 14 for row in self.proprio):
            raise ValueError("WAN22_DROID_PROPRIO_DIM_INVALID")


def validate_window(actions: Sequence[Sequence[float]], proprio: Sequence[Sequence[float]], *, horizon_frames: int = 150) -> Wan22DroidConditioning:
    """Validate one first-frame-conditioned future-prediction window."""

    if len(actions) != horizon_frames or len(proprio) != horizon_frames:
        raise ValueError("WAN22_DROID_HORIZON_INVALID")
    return Wan22DroidConditioning(actions=actions, proprio=proprio)


class Wan22DroidTokenAdapter(nn.Module):
    """Inject aligned robot conditions after WAN2.2 patch embedding.

    The adapter is attached with a forward hook and owns every trainable
    parameter.  The WAN2.2 backbone remains frozen.  Temporal average pooling
    maps the 5 Hz robot stream to the VAE latent frame grid while preserving
    the explicit action/frame alignment contract.
    """

    def __init__(self, model_dim: int, action_dim: int = 7, proprio_dim: int = 14, hidden_dim: int = 256):
        super().__init__()
        self.model_dim = int(model_dim)
        self.action_dim = int(action_dim)
        self.proprio_dim = int(proprio_dim)
        self.action_encoder = nn.Sequential(nn.LayerNorm(action_dim), nn.Linear(action_dim, hidden_dim), nn.SiLU())
        self.proprio_encoder = nn.Sequential(nn.LayerNorm(proprio_dim), nn.Linear(proprio_dim, hidden_dim), nn.SiLU())
        self.output = nn.Linear(hidden_dim * 2, model_dim)
        # Start as an exact no-op so the first optimization step has a clear
        # baseline and cannot silently alter the pretrained visual prior.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self._actions: torch.Tensor | None = None
        self._proprio: torch.Tensor | None = None
        self._hook = None

    def set_conditions(self, actions: torch.Tensor, proprio: torch.Tensor) -> None:
        if actions.ndim != 2 or actions.shape[-1] != self.action_dim:
            raise ValueError(f"Expected actions [F,{self.action_dim}], got {tuple(actions.shape)}")
        if proprio.ndim != 2 or proprio.shape[-1] != self.proprio_dim or proprio.shape[0] != actions.shape[0]:
            raise ValueError(f"Expected proprio [F,{self.proprio_dim}], got {tuple(proprio.shape)}")
        self._actions, self._proprio = actions, proprio

    def clear_conditions(self) -> None:
        self._actions = None
        self._proprio = None

    def _token_bias(self, frames: int, height: int, width: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if self._actions is None or self._proprio is None:
            raise RuntimeError("WAN22_DROID_CONDITIONS_NOT_SET")
        action = self._actions.to(device=device, dtype=torch.float32).transpose(0, 1).unsqueeze(0)
        proprio = self._proprio.to(device=device, dtype=torch.float32).transpose(0, 1).unsqueeze(0)
        action = F.adaptive_avg_pool1d(action, frames).squeeze(0).transpose(0, 1)
        proprio = F.adaptive_avg_pool1d(proprio, frames).squeeze(0).transpose(0, 1)
        encoded = self.output(torch.cat([self.action_encoder(action), self.proprio_encoder(proprio)], dim=-1))
        return encoded.to(dtype=dtype).transpose(0, 1).reshape(1, self.model_dim, frames, 1, 1).expand(1, self.model_dim, frames, height, width)

    def attach(self, backbone: nn.Module):
        """Attach to a WAN2.2 ``WanModel`` without modifying its source tree."""
        if not hasattr(backbone, "patch_embedding"):
            raise ValueError("WAN22_BACKBONE_PATCH_EMBEDDING_MISSING")

        def hook(_module, _inputs, output):
            if not isinstance(output, torch.Tensor) or output.ndim != 5:
                raise ValueError("WAN22_PATCH_EMBEDDING_OUTPUT_INVALID")
            return output + self._token_bias(output.shape[2], output.shape[3], output.shape[4], dtype=output.dtype, device=output.device)

        self._hook = backbone.patch_embedding.register_forward_hook(hook)
        return self._hook

    def detach(self) -> None:
        if self._hook is not None:
            self._hook.remove()
            self._hook = None
