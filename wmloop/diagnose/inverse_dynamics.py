"""Cache identity and confidence discipline for the <=5M inverse-dynamics head.

Training the Torch model belongs to the ACWM runtime environment.  This module
is the dependency-free contract used before and after that worker runs: it
prevents a head trained on one environment/dataset digest from being reused as
evidence for another.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path


class InverseDynamicsError(ValueError):
    """Inverse-dynamics cache identity or confidence is invalid."""


@dataclass(frozen=True)
class InverseDynamicsCacheRecord:
    environment: str
    dataset_digest: str
    architecture: str
    parameter_count: int
    gt_r2: float

    @property
    def low_confidence(self) -> bool:
        return self.gt_r2 < 0.5

    @property
    def cache_key(self) -> str:
        _validate(self)
        payload = json.dumps(
            {"environment": self.environment, "dataset_digest": self.dataset_digest, "architecture": self.architecture},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def write(self, cache_root: Path) -> Path:
        _validate(self)
        destination = Path(cache_root).resolve() / f"{self.cache_key}.json"
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.__dict__, sort_keys=True) + "\n", encoding="utf-8")
        return destination


def _validate(record: InverseDynamicsCacheRecord) -> None:
    if (
        not record.environment
        or len(record.dataset_digest) != 64
        or any(character not in "0123456789abcdef" for character in record.dataset_digest)
        or not record.architecture
        or record.parameter_count <= 0
        or record.parameter_count > 5_000_000
        or not math.isfinite(record.gt_r2)
    ):
        raise InverseDynamicsError("INVERSE_DYNAMICS_CACHE_INVALID")
