"""Apply signed diagnostic experience as a bounded candidate ordering prior.

Only an exact declared action/context/protocol/metric query can reuse a pair.
This service rechecks Archive and CAS and never grants formal effect authority.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Mapping, Sequence

from wmloop.contracts import validate_document
from wmloop.retrieve.index import recheck_probe_experiences
from wmloop.storage import atomic_write, canonical_bytes, checked_path, exclusive_file_lock


class PairedPriorError(ValueError):
    """A signed-prior input or output invariant failed closed."""


def apply_paired_observation_priors(
    batch: Mapping[str, object], matches: Sequence[Mapping[str, object]], *,
    archive_db: Path | None = None, cas_root: Path | None = None,
) -> dict[str, object]:
    """Return a copied batch with rechecked, categorical signed priors.

    A positive observation contributes +0.25, negative -0.5, null zero. Trials
    have equal weight, duplicate signatures/metrics do not add weight, and
    opposing non-null signs abstain. Missing durable stores or a mismatched
    query abstains. Execution PASS by itself always contributes zero.
    """
    validate_document("auto_experiment_candidate_batch", batch)
    output = deepcopy(dict(batch))
    if not any(candidate.get("retrieval_keys", {}).get("paired_query") for candidate in output["candidates"]):
        return output
    stores_available = (archive_db is not None and cas_root is not None
                        and Path(archive_db).is_file() and Path(cas_root).is_dir())
    verified = (recheck_probe_experiences(matches, archive_db=archive_db, cas_root=cas_root)
                if stores_available else ())
    for candidate in output["candidates"]:
        keys = candidate.get("retrieval_keys", {})
        query = keys.get("paired_query")
        if query is None:
            continue
        votes = {}
        signatures = set(keys.get("failure_signatures", []))
        for row in verified:
            if (row.verdict != "PASS" or row.primitive != query["primitive"]
                or keys.get("primitive", query["primitive"]) != query["primitive"]
                or (signatures and row.failure_signature not in signatures)
                or (keys.get("stage") is not None and row.stage != keys["stage"])):
                continue
            for observation in row.paired_observations:
                if canonical_bytes(observation["query"]) == canonical_bytes(query):
                    votes[row.archive_trial_id] = (observation, row)
        signs = {observation["outcome"] for observation, _ in votes.values()} - {"null"}
        contradictory = signs == {"positive", "negative"}
        values = {"positive": 0.25, "negative": -0.5, "null": 0.0}
        candidate["paired_observation_prior"] = (
            0.0 if not votes or contradictory else sum(values[pair[0]["outcome"]] for pair in votes.values()) / len(votes)
        )
        candidate["paired_observation_retrieval"] = {
            "state": "applied" if votes and not contradictory else "abstain",
            "reason": ("PAIRED_OPPOSING_SIGNS" if contradictory else "PAIRED_EXACT_QUERY_MATCH" if votes
                       else "PAIRED_NO_VERIFIED_MATCH" if stores_available else "PAIRED_DURABLE_STORES_REQUIRED"),
            "matched_trial_ids": sorted(votes),
            "outcomes": {outcome: sum(pair[0]["outcome"] == outcome for pair in votes.values())
                         for outcome in ("positive", "negative", "null")},
        }
        previous = [row for row in candidate.get("retrieval_matches", []) if row.get("kind") != "paired_observation"]
        candidate["retrieval_matches"] = previous + [
            {"kind": "paired_observation", "authority": "ranking_only", "query": deepcopy(observation["query"]),
             "outcome": observation["outcome"], "signed_delta": observation["signed_delta"],
             "archive_trial_id": row.archive_trial_id, "receipt_ref": row.receipt_ref,
             "receipt_hash": row.receipt_hash, "result_ref": row.result_ref}
            for _, (observation, row) in sorted(votes.items())
        ]
    validate_document("auto_experiment_candidate_batch", output)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--matches", type=Path, required=True, help="JSON list of retrieval rows or context containing matches")
    parser.add_argument("--archive-db", type=Path, required=True)
    parser.add_argument("--cas-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    batch_path = checked_path(args.batch, code="PAIRED_PRIOR_INPUT_INVALID", error=PairedPriorError)
    matches_path = checked_path(args.matches, code="PAIRED_PRIOR_INPUT_INVALID", error=PairedPriorError)
    batch = json.loads(batch_path.read_bytes())
    matches = json.loads(matches_path.read_bytes())
    if isinstance(matches, dict):
        matches = matches.get("matches")
    if not isinstance(matches, list) or not all(isinstance(row, dict) for row in matches):
        raise PairedPriorError("PAIRED_PRIOR_MATCHES_INVALID")
    result = apply_paired_observation_priors(batch, matches, archive_db=args.archive_db, cas_root=args.cas_root)
    output = checked_path(args.output, code="PAIRED_PRIOR_OUTPUT_INVALID", error=PairedPriorError)
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with exclusive_file_lock(output.parent / f".{output.name}.lock"):
        if output.exists():
            raise PairedPriorError("PAIRED_PRIOR_OUTPUT_EXISTS")
        atomic_write(output, canonical_bytes(result) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
