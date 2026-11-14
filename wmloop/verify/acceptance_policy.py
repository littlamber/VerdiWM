"""Immutable acceptance rules compiled before target execution."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.storage import canonical_bytes, sha256_bytes


CONFIDENCE_INTERVAL = "student_t_two_sided_95"


@dataclass(frozen=True)
class VerificationPolicy:
    """Authority supplied by the control plane, independently of the executor.

    Directions cover the primary utility and every declared protected outcome.
    The full goal digest binds these rules to the pre-execution goal snapshot.
    """

    primary_metric: str
    metric_directions: tuple[tuple[str, str], ...]
    minimum_effect: float
    seeds_accept: tuple[int, ...]
    goal_sha256: str
    action_following_threshold: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.primary_metric, str)
            or not self.primary_metric
            or not isinstance(self.metric_directions, tuple)
            or not self.metric_directions
            or any(
                not isinstance(item, tuple)
                or len(item) != 2
                or not isinstance(item[0], str)
                or not item[0]
                or item[1] not in ("maximize", "minimize")
                for item in self.metric_directions
            )
            or len(dict(self.metric_directions)) != len(self.metric_directions)
            or self.primary_metric not in dict(self.metric_directions)
            or isinstance(self.minimum_effect, bool)
            or not isinstance(self.minimum_effect, (int, float))
            or not _finite_nonnegative(self.minimum_effect)
            or isinstance(self.action_following_threshold, bool)
            or not isinstance(self.action_following_threshold, (int, float))
            or not _finite_nonnegative(self.action_following_threshold)
            or self.action_following_threshold > 1
            or not isinstance(self.seeds_accept, tuple)
            or len(self.seeds_accept) < 3
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in self.seeds_accept)
            or len(set(self.seeds_accept)) != len(self.seeds_accept)
            or not isinstance(self.goal_sha256, str)
            or len(self.goal_sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.goal_sha256)
        ):
            raise ValueError("VERIFICATION_POLICY_INVALID")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "artifact_type": "wmloop-verification-policy",
            "claim_boundary": "Frozen acceptance rules; measurements remain executor evidence.",
            "primary_metric": self.primary_metric,
            "metric_directions": dict(self.metric_directions),
            "minimum_effect": self.minimum_effect,
            "seeds_accept": list(self.seeds_accept),
            "confidence_interval": CONFIDENCE_INTERVAL,
            "goal_sha256": self.goal_sha256,
            "action_following_threshold": self.action_following_threshold,
        }

    @property
    def protected_metrics(self) -> tuple[str, ...]:
        """Return declared non-primary outcomes guarded by this frozen policy."""

        return tuple(metric for metric, _ in self.metric_directions if metric != self.primary_metric)

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_bytes(self.to_dict()))


def _finite_nonnegative(value: float) -> bool:
    try:
        return math.isfinite(value) and value >= 0
    except OverflowError:
        return False


def verification_policy_from_goal(goal: Mapping[str, object]) -> VerificationPolicy | None:
    """Validate and freeze an explicitly declared goal acceptance policy.

    Version 2 goals require the declaration. Historical goals retain the
    zero-threshold compatibility path without a frozen policy binding.
    """
    try:
        validate_document("goal_spec", goal)
    except ContractValidationError as exc:
        raise ValueError("VERIFICATION_POLICY_GOAL_INVALID") from exc
    protocol = goal["eval_protocol"]
    rules = protocol.get("acceptance_policy")
    if rules is None:
        return None
    try:
        return VerificationPolicy(
            primary_metric=goal["primary_objective"],
            metric_directions=tuple(sorted(rules["metric_directions"].items())),
            minimum_effect=rules["minimum_effect"],
            seeds_accept=tuple(protocol["seeds_accept"]),
            goal_sha256=sha256_bytes(canonical_bytes(goal)),
            action_following_threshold=goal["action_following_gate"]["tau_af"],
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("VERIFICATION_POLICY_GOAL_INVALID") from exc


def has_frozen_policy_binding(
    statistics: object, *, receipt: object,
) -> bool:
    """Check new verdict provenance before using it as formal effect evidence.

    This checks the recorded policy, seed cohort and receipt reference digests.
    Durable stores remain responsible for resolving those CAS references and
    verifying the referenced bytes; the ACCEPT label alone grants no authority.
    """
    if not isinstance(statistics, Mapping) or not isinstance(receipt, Mapping):
        return False
    snapshot = statistics.get("policy")
    if not isinstance(snapshot, Mapping) or statistics.get("seed_binding") != "frozen_goal":
        return False
    try:
        validate_document("verification_policy", snapshot)
        policy = VerificationPolicy(
            primary_metric=snapshot["primary_metric"],
            metric_directions=tuple(sorted(snapshot["metric_directions"].items())),
            minimum_effect=snapshot["minimum_effect"],
            seeds_accept=tuple(snapshot["seeds_accept"]),
            goal_sha256=snapshot["goal_sha256"],
            action_following_threshold=snapshot["action_following_threshold"],
        )
        policy_sha256 = sha256_bytes(canonical_bytes(snapshot))
        seeds = statistics.get("replication_seeds")
        return (
            statistics.get("policy_sha256") == policy_sha256
            and statistics.get("confidence_interval") == CONFIDENCE_INTERVAL
            and type(statistics.get("sample_count")) is int
            and statistics["sample_count"] == len(policy.seeds_accept)
            and type(statistics.get("minimum_sample_count")) is int
            and statistics["minimum_sample_count"] == len(policy.seeds_accept)
            and type(statistics.get("threshold")) in (int, float)
            and statistics["threshold"] == policy.minimum_effect
            and isinstance(seeds, list)
            and all(type(seed) is int for seed in seeds)
            and len(seeds) == len(policy.seeds_accept)
            and set(seeds) == set(policy.seeds_accept)
            and receipt.get("verification_policy_ref") == f"cas://sha256/{policy_sha256}"
            and receipt.get("frozen_goal_ref") == f"cas://sha256/{policy.goal_sha256}"
        )
    except (ContractValidationError, KeyError, TypeError, ValueError, OverflowError):
        return False
