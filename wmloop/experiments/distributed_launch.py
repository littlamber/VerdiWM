"""Explicit distributed launch and batch accounting helpers.

This module only constructs argv; it does not claim that a model supports a
particular parallelism strategy.  A backend adapter must opt into ``torchrun``
or another launcher after validating its own model/runtime constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


class DistributedLaunchError(ValueError):
    """A distributed launch configuration is invalid."""


def _positive_int(value: object, code: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise DistributedLaunchError(code)
    return value


@dataclass(frozen=True)
class DistributedLaunchConfig:
    """Launcher settings independent of a model-specific training script."""

    backend: str = "single_process_single_gpu"
    nproc_per_node: int = 1
    nnodes: int = 1
    node_rank: int = 0
    master_addr: str = "127.0.0.1"
    master_port: int = 29500

    @property
    def world_size(self) -> int:
        return self.nnodes * self.nproc_per_node

    def validate(self) -> "DistributedLaunchConfig":
        if self.backend not in {
            "single_process_single_gpu",
            "torchrun",
            "accelerate",
            "deepspeed",
        }:
            raise DistributedLaunchError("DISTRIBUTED_BACKEND_INVALID")
        _positive_int(self.nproc_per_node, "NPROC_PER_NODE_INVALID")
        _positive_int(self.nnodes, "NNODES_INVALID")
        if isinstance(self.node_rank, bool) or not isinstance(self.node_rank, int) or self.node_rank < 0:
            raise DistributedLaunchError("NODE_RANK_INVALID")
        if self.node_rank >= self.nnodes:
            raise DistributedLaunchError("NODE_RANK_OUT_OF_RANGE")
        if not self.master_addr.strip():
            raise DistributedLaunchError("MASTER_ADDR_INVALID")
        if isinstance(self.master_port, bool) or not isinstance(self.master_port, int) or not 1 <= self.master_port <= 65535:
            raise DistributedLaunchError("MASTER_PORT_INVALID")
        if self.backend == "single_process_single_gpu" and self.world_size != 1:
            raise DistributedLaunchError("SINGLE_BACKEND_REQUIRES_ONE_PROCESS")
        return self

    def to_document(self) -> dict[str, object]:
        self.validate()
        return {
            "backend": self.backend,
            "nproc_per_node": self.nproc_per_node,
            "nnodes": self.nnodes,
            "node_rank": self.node_rank,
            "world_size": self.world_size,
            "master_addr": self.master_addr,
            "master_port": self.master_port,
        }


def build_launch_command(
    command: Sequence[str], config: DistributedLaunchConfig
) -> list[str]:
    """Prefix a model command with an explicit launcher.

    ``command`` is always argv, never shell text.  ``accelerate`` and
    ``deepspeed`` are intentionally explicit because their config files and
    model compatibility are backend responsibilities, not safe defaults.
    """

    config.validate()
    if not command or any(not isinstance(item, str) or not item for item in command):
        raise DistributedLaunchError("TRAINING_COMMAND_INVALID")
    if config.backend == "single_process_single_gpu":
        return list(command)
    if config.backend == "torchrun":
        prefix = [
            "torchrun",
            "--nnodes",
            str(config.nnodes),
            "--nproc_per_node",
            str(config.nproc_per_node),
            "--node_rank",
            str(config.node_rank),
            "--master_addr",
            config.master_addr,
            "--master_port",
            str(config.master_port),
        ]
        return prefix + list(command)
    if config.backend == "accelerate":
        return [
            "accelerate",
            "launch",
            "--num_processes",
            str(config.world_size),
            "--main_process_ip",
            config.master_addr,
            "--main_process_port",
            str(config.master_port),
            "--machine_rank",
            str(config.node_rank),
            *command,
        ]
    return [
        "deepspeed",
        "--num_nodes",
        str(config.nnodes),
        "--num_gpus",
        str(config.nproc_per_node),
        "--node_rank",
        str(config.node_rank),
        "--master_addr",
        config.master_addr,
        "--master_port",
        str(config.master_port),
        *command,
    ]

