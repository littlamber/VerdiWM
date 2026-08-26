"""Direction-aware, paired evidence utilities shared by model adapters."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class MetricSpec:
    name: str
    direction: str = "maximize"
    role: str = "primary"
    practical_threshold: float = 0.0

    def __post_init__(self) -> None:
        if self.direction not in {"maximize", "minimize"}:
            raise ValueError("metric direction must be maximize or minimize")
        if self.role not in {"primary", "protected"}:
            raise ValueError("metric role must be primary or protected")

    def improvement(self, baseline: float, candidate: float) -> float:
        """Return positive values for candidate improvements regardless of direction."""
        return float(candidate - baseline) if self.direction == "maximize" else float(baseline - candidate)


def compare_metrics(baseline: Mapping[str, float], candidate: Mapping[str, float], specs: Iterable[MetricSpec]) -> dict[str, Any]:
    specs = tuple(specs)
    missing = [spec.name for spec in specs if spec.name not in baseline or spec.name not in candidate]
    if missing:
        return {"outcome": "abstain", "reason": "missing_metrics", "missing": missing}
    deltas = {spec.name: spec.improvement(float(baseline[spec.name]), float(candidate[spec.name])) for spec in specs}
    protected = [deltas[spec.name] >= -abs(spec.practical_threshold) for spec in specs if spec.role == "protected"]
    return {
        "outcome": "measured",
        "deltas": deltas,
        "protected_ok": all(protected) if protected else True,
        "metrics": {spec.name: asdict(spec) for spec in specs},
    }


def paired_bootstrap(deltas: Iterable[float], *, samples: int = 10000, seed: int = 20260825) -> dict[str, float]:
    values = [float(value) for value in deltas]
    if not values:
        raise ValueError("paired bootstrap requires at least one delta")
    rng = random.Random(seed)
    means = []
    for _ in range(max(1, samples)):
        means.append(sum(values[rng.randrange(len(values))] for _ in values) / len(values))
    means.sort()
    mean = sum(values) / len(values)
    return {
        "mean": mean,
        "ci95_low": means[max(0, int(0.025 * len(means)) - 1)],
        "ci95_high": means[min(len(means) - 1, int(0.975 * len(means)))],
        "replicates": len(values),
        "bootstrap_samples": max(1, samples),
        "seed": seed,
    }


def classify_paired_effect(
    deltas: Iterable[float],
    *,
    practical_threshold: float,
    protected_ok: bool,
    min_replicates: int = 2,
    bootstrap_samples: int = 10000,
    seed: int = 20260825,
) -> dict[str, Any]:
    values = [float(value) for value in deltas]
    if len(values) < min_replicates:
        return {"outcome": "abstain", "reason": "requires_independent_replicates", "replicates": len(values)}
    stats = paired_bootstrap(values, samples=bootstrap_samples, seed=seed)
    threshold = abs(float(practical_threshold))
    if not protected_ok or stats["ci95_high"] < -threshold:
        outcome = "harmful"
    elif stats["ci95_low"] > threshold:
        outcome = "confirmed_positive"
    elif stats["ci95_low"] >= -threshold and stats["ci95_high"] <= threshold:
        outcome = "null"
    else:
        outcome = "abstain"
    return {"outcome": outcome, "protected_ok": protected_ok, "practical_threshold": threshold, **stats}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def verify_artifacts(artifacts: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate required output files and declared hashes before evaluation."""
    checked = []
    missing = []
    mismatched = []
    for artifact in artifacts:
        path = Path(str(artifact.get("path", "")))
        required = bool(artifact.get("required", True))
        if not path.is_file():
            (missing if required else checked).append(str(path))
            continue
        digest = sha256_file(path)
        expected = artifact.get("sha256")
        if expected and str(expected) != digest:
            mismatched.append({"path": str(path), "expected": expected, "actual": digest})
        checked.append({"path": str(path), "sha256": digest, "size": path.stat().st_size})
    if missing or mismatched:
        return {"state": "invalid", "reason": "artifact_integrity_failed", "missing": missing, "mismatched": mismatched, "checked": checked}
    return {"state": "complete", "checked": checked, "manifest_digest": _digest(checked)}


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()
