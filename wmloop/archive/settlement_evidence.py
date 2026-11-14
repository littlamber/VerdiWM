"""Read authoritative trial identities without mutating Archive projections."""

from __future__ import annotations

import json
import sqlite3

from wmloop.archive.store import ArchiveInvariantError


def read_settled_trial_rows(
    connection: sqlite3.Connection, *, receipt_ref: str | None = None,
    limit: int | None = None,
) -> list[sqlite3.Row]:
    """Join and bind all settlement projections; CAS validation is separate.

    LEFT JOINs deliberately expose missing projections. Consumers must never
    turn a partially corrupted settlement into trusted evidence by omitting it.
    The caller owns the read transaction and any evidence-scope restrictions.
    """
    query = """
        SELECT t.*, r.receipt_ref AS settled_receipt_ref, r.receipt_hash,
               v.verdict_ref AS settled_verdict_ref, p.proposal_id AS settled_proposal_id,
               p.goal_id AS settled_goal_id, p.library_version AS settled_library_version
        FROM trials t LEFT JOIN receipts r ON r.trial_id=t.trial_id
        LEFT JOIN verdicts v ON v.trial_id=t.trial_id
        LEFT JOIN proposals p ON p.trial_id=t.trial_id
    """
    parameters: list[object] = []
    if receipt_ref is not None:
        query += " WHERE t.receipt_ref=?"
        parameters.append(receipt_ref)
    query += " ORDER BY t.trial_id"
    if limit is not None:
        query += " LIMIT ?"
        parameters.append(limit)
    cursor = connection.cursor()
    cursor.row_factory = sqlite3.Row
    rows = cursor.execute(query, parameters).fetchall()
    for row in rows:
        try:
            settlement = json.loads(row["settlement_json"])
            valid = (
                isinstance(settlement, dict) and settlement.get("state") == "settled"
                and row["receipt_ref"] == row["settled_receipt_ref"]
                and row["verdict_ref"] == row["settled_verdict_ref"]
                and row["proposal_id"] == row["settled_proposal_id"]
                and row["goal_id"] == row["settled_goal_id"]
                and row["library_version"] == row["settled_library_version"]
                and isinstance(row["receipt_hash"], str)
                and settlement.get("receipt_hash") == row["receipt_hash"]
                and row["receipt_ref"] == "cas://sha256/" + row["receipt_hash"]
            )
        except (ValueError, TypeError) as exc:
            raise ArchiveInvariantError("ARCHIVE_SETTLEMENT_BINDING_INVALID") from exc
        if not valid:
            raise ArchiveInvariantError("ARCHIVE_SETTLEMENT_BINDING_INVALID")
    return rows
