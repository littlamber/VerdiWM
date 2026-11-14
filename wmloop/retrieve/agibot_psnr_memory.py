"""Read-only retrieval of settled Agibot PSNR research memory.

The Archive and CAS are authoritative.  This module is a bounded projection
for a later research round: it replays the settlement bindings before
returning a hypothesis, method and PSNR result.  It never turns a rejected
trial into a success and never writes a retrieval index or mutates Archive.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sqlite3
from typing import Any, Mapping

from wmloop.archive.agibot_psnr_settlement import (
    AgibotPSNRSettlementError,
    verify_agibot_psnr_settled_receipt,
)
from wmloop.archive.settlement_evidence import read_settled_trial_rows
from wmloop.archive.store import ArchiveInvariantError, ContentAddressedStore
from wmloop.storage import checked_path


class AgibotPSNRMemoryError(ValueError):
    """The Agibot PSNR memory query or durable evidence is invalid."""


@dataclass(frozen=True)
class AgibotPSNRMemory:
    """A verified, compact projection suitable for the next research round."""

    trial_id: str
    proposal_id: str
    goal_id: str
    receipt_ref: str
    verdict_ref: str
    measurement_ref: str
    dataset_sha256: str
    support_sha256: str
    evaluator_digest: str
    checkpoint_sha256: Mapping[str, str]
    verdict: str
    reason: str
    effect_verified: bool
    mean_delta_db: float
    lower_bound_db: float
    upper_bound_db: float
    hypothesis: str
    method: str
    report_only_metrics: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe routing projection with immutable evidence refs."""

        return {
            "trial_id": self.trial_id,
            "proposal_id": self.proposal_id,
            "goal_id": self.goal_id,
            "evidence": {
                "receipt_ref": self.receipt_ref,
                "verdict_ref": self.verdict_ref,
                "measurement_ref": self.measurement_ref,
            },
            "dataset_sha256": self.dataset_sha256,
            "support_sha256": self.support_sha256,
            "evaluator_digest": self.evaluator_digest,
            "checkpoint_sha256": dict(self.checkpoint_sha256),
            "verdict": self.verdict,
            "reason": self.reason,
            "effect_verified": self.effect_verified,
            "statistics": {
                "mean_paired_delta_db": self.mean_delta_db,
                "lower_bound_db": self.lower_bound_db,
                "upper_bound_db": self.upper_bound_db,
            },
            "hypothesis": self.hypothesis,
            "method": self.method,
            "report_only_metrics": dict(self.report_only_metrics),
            "claim_boundary": (
                "Verified settled memory supplies hypotheses and references for a later "
                "round; it does not establish transfer or current-round success."
            ),
        }


def retrieve_agibot_psnr_memories(
    *,
    archive_db: Path,
    cas_root: Path,
    dataset_sha256: str | None = None,
    support_sha256: str | None = None,
    evaluator_digest: str | None = None,
    effect_verified: bool | None = None,
    limit: int = 20,
    replay_measurement: bool = True,
) -> tuple[AgibotPSNRMemory, ...]:
    """Return verified Agibot PSNR settlements compatible with the query.

    Rows from other libraries, partially projected trials, stale CAS objects,
    and malformed settlements are excluded.  ``replay_measurement`` defaults
    to true so a later round reads the same source frames that created the
    original effect verdict.  The function is read-only and returns an empty
    tuple for an uninitialized Archive/CAS pair.
    """

    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 500:
        raise AgibotPSNRMemoryError("AGIBOT_PSNR_MEMORY_LIMIT_INVALID")
    for value, code in (
        (dataset_sha256, "DATASET_FILTER_INVALID"),
        (support_sha256, "SUPPORT_FILTER_INVALID"),
        (evaluator_digest, "EVALUATOR_FILTER_INVALID"),
    ):
        if value is not None:
            _require_digest(value, code)
    if effect_verified is not None and not isinstance(effect_verified, bool):
        raise AgibotPSNRMemoryError("AGIBOT_PSNR_MEMORY_EFFECT_FILTER_INVALID")

    database = checked_path(archive_db, code="AGIBOT_PSNR_MEMORY_ARCHIVE_INVALID", error=AgibotPSNRMemoryError)
    cas_path = checked_path(cas_root, code="AGIBOT_PSNR_MEMORY_CAS_INVALID", error=AgibotPSNRMemoryError)
    # ContentAddressedStore creates its root on construction.  Retrieval must
    # not create state merely because a later round has no prior memory.
    if not database.is_file() or not (cas_path / "cas").is_dir():
        return ()
    try:
        cas = ContentAddressedStore(cas_path)
        with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
            connection.row_factory = sqlite3.Row
            rows = read_settled_trial_rows(connection, limit=None)
        memories: list[AgibotPSNRMemory] = []
        for row in rows:
            if row["library_version"] != "agibot-psnr-settlement-v1":
                continue
            try:
                verified = verify_agibot_psnr_settled_receipt(
                    receipt_ref=str(row["receipt_ref"]),
                    archive_db=database,
                    cas_root=cas_path,
                    replay_measurement=replay_measurement,
                )
                memory = _memory_from_verified(verified)
            except (AgibotPSNRSettlementError, ArchiveInvariantError, OSError, TypeError, ValueError, KeyError):
                # A single stale/corrupt historical row cannot become trusted
                # memory.  It is omitted while valid settlements remain usable.
                continue
            if dataset_sha256 is not None and memory.dataset_sha256 != dataset_sha256:
                continue
            if support_sha256 is not None and memory.support_sha256 != support_sha256:
                continue
            if evaluator_digest is not None and memory.evaluator_digest != evaluator_digest:
                continue
            if effect_verified is not None and memory.effect_verified != effect_verified:
                continue
            memories.append(memory)
        memories.sort(
            key=lambda value: (
                0 if value.effect_verified else 1,
                -value.lower_bound_db,
                value.trial_id,
            )
        )
        return tuple(memories[:limit])
    except (ArchiveInvariantError, OSError, sqlite3.Error, ValueError, TypeError) as exc:
        raise AgibotPSNRMemoryError("AGIBOT_PSNR_MEMORY_READ_FAILED") from exc


def build_agibot_psnr_memory_context(
    *,
    archive_db: Path,
    cas_root: Path,
    dataset_sha256: str | None = None,
    support_sha256: str | None = None,
    evaluator_digest: str | None = None,
    limit: int = 5,
    replay_measurement: bool = True,
) -> dict[str, Any]:
    """Build a bounded context for the next Retrieve/Hypothesize stage."""

    matches = retrieve_agibot_psnr_memories(
        archive_db=archive_db,
        cas_root=cas_root,
        dataset_sha256=dataset_sha256,
        support_sha256=support_sha256,
        evaluator_digest=evaluator_digest,
        limit=limit,
        replay_measurement=replay_measurement,
    )
    return {
        "state": "matched" if matches else "cold_start",
        "query": {
            "dataset_sha256": dataset_sha256,
            "support_sha256": support_sha256,
            "evaluator_digest": evaluator_digest,
        },
        "match_count": len(matches),
        "matches": [match.to_dict() for match in matches],
        "claim_boundary": (
            "Settled PSNR memories guide hypotheses and method retrieval. They do not "
            "transfer an effect or satisfy the current round's goal."
        ),
    }


def _memory_from_verified(verified: Any) -> AgibotPSNRMemory:
    context = verified.context
    verdict = verified.verdict
    measurement = verified.measurement
    statistics = measurement["statistics"]
    report_only = measurement["report_only_metrics"]
    values = (
        statistics["mean_paired_delta_db"],
        statistics["lower_bound_db"],
        statistics["upper_bound_db"],
    )
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
        raise AgibotPSNRMemoryError("AGIBOT_PSNR_MEMORY_STATISTICS_INVALID")
    return AgibotPSNRMemory(
        trial_id=verified.trial_id,
        proposal_id=verified.proposal_id,
        goal_id=verified.goal_id,
        receipt_ref=verified.receipt_ref,
        verdict_ref=verified.verdict_ref,
        measurement_ref=verified.measurement_ref,
        dataset_sha256=measurement["dataset_sha256"],
        support_sha256=measurement["support_sha256"],
        evaluator_digest=measurement["evaluator_digest"],
        checkpoint_sha256=dict(measurement["checkpoint_sha256"]),
        verdict=verdict["verdict"],
        reason=verdict["reason"],
        effect_verified=bool(verified.receipt["effect_verified"]),
        mean_delta_db=float(values[0]),
        lower_bound_db=float(values[1]),
        upper_bound_db=float(values[2]),
        hypothesis=context["hypothesis"],
        method=context["method"],
        report_only_metrics=dict(report_only),
    )


def _require_digest(value: object, code: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise AgibotPSNRMemoryError(f"AGIBOT_PSNR_MEMORY_{code}")
