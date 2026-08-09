"""SQLite-linearized reservation, fencing and receipt settlement.

The executor is allowed to start work only after :meth:`BudgetLedger.admit`
returns a reservation.  State transitions use an exact fencing token; a stale
worker cannot settle an attempt after recovery/takeover.

Settlement records the actual cost even when a trial overruns its reservation.
The overrun is audit evidence, not a reason to discard completed scientific
results; later admissions are debited against settled actual cost.
"""

from __future__ import annotations

import math
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path


class BudgetError(ValueError):
    """Admission, fencing or settlement failed before state mutation."""


_COST_CLASSES = {"very_low", "low", "medium", "high"}


@dataclass(frozen=True)
class BudgetPolicy:
    total_gpu_hours: float
    high_trial_limit: int = 2
    max_trial_gpu_hours: float = 120.0
    require_high_cost_approval: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.total_gpu_hours) or self.total_gpu_hours <= 0:
            raise ValueError("BUDGET_TOTAL_INVALID")
        if self.high_trial_limit < 0:
            raise ValueError("HIGH_TRIAL_LIMIT_INVALID")
        if (
            not math.isfinite(self.max_trial_gpu_hours)
            or self.max_trial_gpu_hours <= 0
        ):
            raise ValueError("MAX_TRIAL_GPU_HOURS_INVALID")


@dataclass(frozen=True)
class TrialAdmission:
    trial_id: str
    cost_class: str
    estimated_gpu_hours: float
    fencing_token: int
    state: str


class BudgetLedger:
    def __init__(self, database_path: Path, policy: BudgetPolicy) -> None:
        self._path = Path(database_path).resolve()
        self._policy = policy
        self._path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._initialize()

    def admit(
        self,
        trial_id: str,
        *,
        cost_class: str,
        estimated_gpu_hours: float,
        human_approved: bool = False,
    ) -> TrialAdmission:
        _validate_admission(
            trial_id,
            cost_class,
            estimated_gpu_hours,
            human_approved,
            policy=self._policy,
        )
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM trials WHERE trial_id = ?", (trial_id,)).fetchone()
            if row is not None:
                if (
                    row["cost_class"] != cost_class
                    or row["estimated_gpu_hours"] != estimated_gpu_hours
                    or bool(row["human_approved"]) != human_approved
                ):
                    connection.rollback()
                    raise BudgetError("TRIAL_IDENTITY_CONFLICT")
                connection.commit()
                return _admission_from_row(row)
            if cost_class == "high":
                high_count = connection.execute("SELECT COUNT(*) FROM trials WHERE cost_class = 'high'").fetchone()[0]
                if high_count >= self._policy.high_trial_limit:
                    connection.rollback()
                    raise BudgetError("HIGH_COST_TRIAL_LIMIT_EXCEEDED")
            reserved = connection.execute(
                """
                SELECT COALESCE(SUM(
                    CASE WHEN state = 'settled' THEN actual_gpu_hours ELSE estimated_gpu_hours END
                ), 0) FROM trials WHERE state IN ('admitted', 'settled')
                """
            ).fetchone()[0]
            if float(reserved) + estimated_gpu_hours > self._policy.total_gpu_hours:
                connection.rollback()
                raise BudgetError("GLOBAL_BUDGET_EXHAUSTED")
            connection.execute(
                """
                INSERT INTO trials (
                    trial_id, cost_class, estimated_gpu_hours, human_approved,
                    fencing_token, state, actual_gpu_hours, receipt_ref
                ) VALUES (?, ?, ?, ?, 1, 'admitted', NULL, NULL)
                """,
                (trial_id, cost_class, estimated_gpu_hours, int(human_approved)),
            )
            row = connection.execute("SELECT * FROM trials WHERE trial_id = ?", (trial_id,)).fetchone()
            connection.commit()
            return _admission_from_row(row)
        finally:
            connection.close()

    def get(self, trial_id: str) -> TrialAdmission | None:
        """Return the durable admission state without mutating the ledger."""

        if not trial_id:
            raise BudgetError("TRIAL_ID_INVALID")
        connection = self._connect()
        try:
            row = connection.execute("SELECT * FROM trials WHERE trial_id = ?", (trial_id,)).fetchone()
        finally:
            connection.close()
        return _admission_from_row(row) if row is not None else None

    def takeover(self, trial_id: str, *, expected_fencing_token: int) -> TrialAdmission:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = _require_active_row(connection, trial_id, expected_fencing_token)
            connection.execute(
                "UPDATE trials SET fencing_token = fencing_token + 1 WHERE trial_id = ?", (trial_id,)
            )
            next_row = connection.execute("SELECT * FROM trials WHERE trial_id = ?", (trial_id,)).fetchone()
            connection.commit()
            return _admission_from_row(next_row)
        finally:
            connection.close()

    def settle(
        self,
        trial_id: str,
        *,
        fencing_token: int,
        actual_gpu_hours: float,
        receipt_ref: str,
    ) -> TrialAdmission:
        _validate_settlement(actual_gpu_hours, receipt_ref)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = _require_active_row(connection, trial_id, fencing_token)
            connection.execute(
                """
                UPDATE trials SET state = 'settled', actual_gpu_hours = ?, receipt_ref = ?
                WHERE trial_id = ? AND fencing_token = ?
                """,
                (actual_gpu_hours, receipt_ref, trial_id, fencing_token),
            )
            settled = connection.execute("SELECT * FROM trials WHERE trial_id = ?", (trial_id,)).fetchone()
            connection.commit()
            return _admission_from_row(settled)
        finally:
            connection.close()

    def release(self, trial_id: str, *, fencing_token: int) -> None:
        """Release a reservation only when no experiment process was launched."""

        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            _require_active_row(connection, trial_id, fencing_token)
            connection.execute(
                "DELETE FROM trials WHERE trial_id = ? AND fencing_token = ?",
                (trial_id, fencing_token),
            )
            connection.commit()
        finally:
            connection.close()

    def visible_settled_trial_ids(self) -> tuple[str, ...]:
        connection = self._connect()
        try:
            rows = connection.execute("SELECT trial_id FROM trials WHERE state = 'settled' ORDER BY trial_id").fetchall()
        finally:
            connection.close()
        return tuple(str(row["trial_id"]) for row in rows)

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS trials (
                    trial_id TEXT PRIMARY KEY NOT NULL,
                    cost_class TEXT NOT NULL,
                    estimated_gpu_hours REAL NOT NULL,
                    human_approved INTEGER NOT NULL,
                    fencing_token INTEGER NOT NULL,
                    state TEXT NOT NULL CHECK (state IN ('admitted', 'settled')),
                    actual_gpu_hours REAL,
                    receipt_ref TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS policy (
                    singleton INTEGER PRIMARY KEY NOT NULL CHECK (singleton = 1),
                    total_gpu_hours REAL NOT NULL,
                    high_trial_limit INTEGER NOT NULL,
                    max_trial_gpu_hours REAL NOT NULL,
                    require_high_cost_approval INTEGER NOT NULL
                )
                """
            )
            columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(policy)").fetchall()
            }
            if "max_trial_gpu_hours" not in columns:
                connection.execute(
                    "ALTER TABLE policy ADD COLUMN max_trial_gpu_hours "
                    "REAL NOT NULL DEFAULT 120.0"
                )
            if "require_high_cost_approval" not in columns:
                connection.execute(
                    "ALTER TABLE policy ADD COLUMN require_high_cost_approval "
                    "INTEGER NOT NULL DEFAULT 1"
                )
            row = connection.execute(
                "SELECT total_gpu_hours, high_trial_limit, max_trial_gpu_hours, "
                "require_high_cost_approval FROM policy WHERE singleton = 1"
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO policy(singleton, total_gpu_hours, high_trial_limit, "
                    "max_trial_gpu_hours, require_high_cost_approval) "
                    "VALUES (1, ?, ?, ?, ?)",
                    (
                        self._policy.total_gpu_hours,
                        self._policy.high_trial_limit,
                        self._policy.max_trial_gpu_hours,
                        int(self._policy.require_high_cost_approval),
                    ),
                )
            elif (
                float(row["total_gpu_hours"]) != self._policy.total_gpu_hours
                or int(row["high_trial_limit"]) != self._policy.high_trial_limit
                or float(row["max_trial_gpu_hours"])
                != self._policy.max_trial_gpu_hours
                or bool(row["require_high_cost_approval"])
                != self._policy.require_high_cost_approval
            ):
                connection.rollback()
                raise BudgetError("BUDGET_POLICY_MISMATCH")
            connection.commit()
        finally:
            connection.close()
        os.chmod(self._path, 0o600)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path, isolation_level=None, timeout=60.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=60000")
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        return connection


def _validate_admission(
    trial_id: str,
    cost_class: str,
    estimated_gpu_hours: float,
    human_approved: bool,
    *,
    policy: BudgetPolicy,
) -> None:
    if (
        not trial_id
        or cost_class not in _COST_CLASSES
        or not math.isfinite(estimated_gpu_hours)
        or estimated_gpu_hours <= 0
    ):
        raise BudgetError("TRIAL_ADMISSION_INVALID")
    if estimated_gpu_hours > policy.max_trial_gpu_hours:
        raise BudgetError("TRIAL_COST_CAP_EXCEEDED")
    if (
        cost_class == "high"
        and policy.require_high_cost_approval
        and not human_approved
    ):
        raise BudgetError("HIGH_COST_HUMAN_APPROVAL_REQUIRED")


def _validate_settlement(actual_gpu_hours: float, receipt_ref: str) -> None:
    if not math.isfinite(actual_gpu_hours) or actual_gpu_hours < 0:
        raise BudgetError("SETTLEMENT_COST_INVALID")
    if not _is_cas_uri(receipt_ref):
        raise BudgetError("SETTLEMENT_RECEIPT_INVALID")


def _require_active_row(connection: sqlite3.Connection, trial_id: str, fencing_token: int) -> sqlite3.Row:
    row = connection.execute("SELECT * FROM trials WHERE trial_id = ?", (trial_id,)).fetchone()
    if row is None:
        connection.rollback()
        raise BudgetError("TRIAL_NOT_FOUND")
    if row["state"] != "admitted":
        connection.rollback()
        raise BudgetError("TRIAL_NOT_ACTIVE")
    if row["fencing_token"] != fencing_token:
        connection.rollback()
        raise BudgetError("STALE_FENCING_TOKEN")
    return row


def _admission_from_row(row: sqlite3.Row) -> TrialAdmission:
    return TrialAdmission(
        trial_id=str(row["trial_id"]),
        cost_class=str(row["cost_class"]),
        estimated_gpu_hours=float(row["estimated_gpu_hours"]),
        fencing_token=int(row["fencing_token"]),
        state=str(row["state"]),
    )


def _is_cas_uri(value: str) -> bool:
    prefix = "cas://sha256/"
    digest = value[len(prefix) :] if isinstance(value, str) and value.startswith(prefix) else ""
    return len(digest) == 64 and all(character in "0123456789abcdef" for character in digest)
