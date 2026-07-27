"""Frozen intervention-cell embeddings for archive retrieval and sorting.

The implementation is deliberately dependency-free and deterministic.  It is a
control-plane encoder for structured intervention cells, not a semantic paper
embedding model; changing it requires a version boundary because distances feed
proposal sorting.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Iterable, Sequence

from wmloop.propose.scheduler import InterventionCell


ENCODER_ID = "wmloop-structured-intervention-hash-v1"
EMBEDDING_DIMENSIONS = 128
TOKENIZATION_VERSION = "field-prefixed-tokens-v1"


@dataclass(frozen=True)
class FrozenEncoderManifest:
    encoder_id: str
    dimensions: int
    tokenization_version: str
    freeze_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "encoder_id": self.encoder_id,
            "dimensions": self.dimensions,
            "tokenization_version": self.tokenization_version,
            "freeze_sha256": self.freeze_sha256,
        }


def encoder_manifest() -> FrozenEncoderManifest:
    payload = {
        "encoder_id": ENCODER_ID,
        "dimensions": EMBEDDING_DIMENSIONS,
        "tokenization_version": TOKENIZATION_VERSION,
        "normalization": "l2",
        "hash": "sha256:index+sign",
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    return FrozenEncoderManifest(
        encoder_id=ENCODER_ID,
        dimensions=EMBEDDING_DIMENSIONS,
        tokenization_version=TOKENIZATION_VERSION,
        freeze_sha256=digest,
    )


def intervention_description(cell: InterventionCell) -> str:
    """Return the canonical text payload that is embedded and archived."""

    return "|".join(
        [
            f"environment={cell.environment}",
            f"layer={cell.layer}",
            f"primitive_family={cell.primitive_family}",
            f"parameter_bucket={cell.parameter_bucket}",
        ]
    )


def embed_intervention(cell: InterventionCell) -> tuple[float, ...]:
    vector = [0.0] * EMBEDDING_DIMENSIONS
    for token in _tokens(cell):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % EMBEDDING_DIMENSIONS
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return tuple(vector)
    return tuple(value / norm for value in vector)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) != EMBEDDING_DIMENSIONS:
        raise ValueError("EMBEDDING_DIMENSION_MISMATCH")
    return min(1.0, max(-1.0, sum(a * b for a, b in zip(left, right))))


def cosine_distance(left: Sequence[float], right: Sequence[float]) -> float:
    distance = 1.0 - cosine_similarity(left, right)
    return 0.0 if abs(distance) < 1e-12 else distance


def nearest_cosine_distance(cell: InterventionCell, archive_cells: Iterable[InterventionCell]) -> float | None:
    target = embed_intervention(cell)
    distances = [cosine_distance(target, embed_intervention(candidate)) for candidate in archive_cells]
    if not distances:
        return None
    return min(distances)


def _tokens(cell: InterventionCell) -> list[str]:
    tokens = [
        f"environment:{cell.environment}",
        f"layer:{cell.layer}",
        f"primitive_family:{cell.primitive_family}",
        f"parameter_bucket:{cell.parameter_bucket}",
    ]
    for piece in re.split(r"[^A-Za-z0-9_.+-]+", cell.parameter_bucket):
        if piece:
            tokens.append(f"parameter_token:{piece}")
    return tokens
