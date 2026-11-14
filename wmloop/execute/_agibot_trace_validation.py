"""Validation and merge of per-rank Agibot training trace shards."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

if __package__:
    from ._agibot_trace_contract import (
        AgibotTraceError,
        SCHEMA_VERSION,
        TRACE_ARTIFACT_TYPE,
        TRACE_CONFIG_TYPE,
        _is_finite_number,
        _is_sha256,
        _read_json,
        _validate_sample,
        _write_json,
        canonical_digest,
    )
else:
    from _agibot_trace_contract import (
        AgibotTraceError,
        SCHEMA_VERSION,
        TRACE_ARTIFACT_TYPE,
        TRACE_CONFIG_TYPE,
        _is_finite_number,
        _is_sha256,
        _read_json,
        _validate_sample,
        _write_json,
        canonical_digest,
    )

def merge_trace_shards(
    config_path: Path,
    shard_paths: Sequence[Path],
    *,
    output_path: Path | None = None,
) -> dict[str, object]:
    """Validate and merge per-rank shards, requiring full held-out coverage."""

    config = _read_json(config_path)
    _validate_config(config)
    expected_ranks = int(config["world_size"])
    headers: dict[int, dict[str, object]] = {}
    footers: dict[int, dict[str, object]] = {}
    events: list[dict[str, object]] = []
    for path in shard_paths:
        header, shard_events, footer = _read_shard(path)
        rank = header.get("rank")
        if type(rank) is not int or rank < 0 or rank >= expected_ranks or rank in headers:
            raise AgibotTraceError("AGIBOT_TRACE_RANK_SHARD_INVALID")
        _validate_header(header, config)
        _validate_footer(footer, config, rank)
        headers[rank] = header
        footers[rank] = footer
        _validate_rank_events(shard_events, rank=rank, config=config)
        events.extend(shard_events)
    if set(headers) != set(range(expected_ranks)):
        raise AgibotTraceError("AGIBOT_TRACE_RANK_SHARD_MISSING")
    _validate_global_events(events, config)
    merged = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": TRACE_ARTIFACT_TYPE,
        "state": "complete",
        "trace_id": config["trace_id"],
        "support_digest": config["support_digest"],
        "source_fingerprint": config["source_fingerprint"],
        "native_config_sha256": config["native_config_sha256"],
        "world_size": expected_ranks,
        "rank_count": len(headers),
        "rank_completion_count": len(footers),
        "optimizer_update_count": max(
            (int(event["optimizer_update"]) for event in events if event["phase"] == "train"),
            default=0,
        ),
        "validation_pass_count": max(
            (int(event["validation_pass"]) for event in events if event["phase"] == "validation"),
            default=0,
        ),
        "event_count": len(events),
        "events": sorted(events, key=_event_sort_key),
        "claim_boundary": (
            "Training-consumption and fixed-panel validation trace only. It does not establish model improvement, promotion, or generalization."
        ),
    }
    if output_path is not None:
        _write_json(output_path, merged)
    return merged


def _read_shard(
    path: Path,
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        rows = [json.loads(line) for line in lines]
    except (OSError, json.JSONDecodeError) as exc:
        raise AgibotTraceError("AGIBOT_TRACE_SHARD_INVALID") from exc
    if not rows or not isinstance(rows[0], dict) or rows[0].get("record_type") != "header":
        raise AgibotTraceError("AGIBOT_TRACE_SHARD_HEADER_MISSING")
    header = rows[0]
    if (
        len(rows) < 2
        or not isinstance(rows[-1], dict)
        or rows[-1].get("record_type") != "footer"
    ):
        raise AgibotTraceError("AGIBOT_TRACE_RANK_COMPLETION_MISSING")
    footer = rows[-1]
    events = rows[1:-1]
    if any(not isinstance(row, dict) or row.get("record_type") != "event" for row in events):
        raise AgibotTraceError("AGIBOT_TRACE_SHARD_EVENT_INVALID")
    return header, events, footer


def _validate_footer(
    footer: Mapping[str, object], config: Mapping[str, object], rank: int
) -> None:
    if (
        footer.get("schema_version") != SCHEMA_VERSION
        or footer.get("trace_id") != config["trace_id"]
        or footer.get("support_digest") != config["support_digest"]
        or footer.get("source_fingerprint") != config["source_fingerprint"]
        or footer.get("rank") != rank
        or footer.get("state") != "complete"
        or footer.get("optimizer_update_count") != config["expected_optimizer_updates"]
        or footer.get("validation_pass_count") != 1
    ):
        raise AgibotTraceError("AGIBOT_TRACE_RANK_COMPLETION_INVALID")


def _validate_rank_events(events: Sequence[Mapping[str, object]], *, rank: int, config: Mapping[str, object]) -> None:
    sequences: list[int] = []
    for event in events:
        if (
            event.get("rank") != rank
            or event.get("trace_id") != config["trace_id"]
            or event.get("support_digest") != config["support_digest"]
            or event.get("source_fingerprint") != config["source_fingerprint"]
            or type(event.get("sequence")) is not int
        ):
            raise AgibotTraceError("AGIBOT_TRACE_EVENT_BINDING_INVALID")
        sequences.append(int(event["sequence"]))
        if event.get("phase") == "train":
            if event.get("split_id") != "train" or type(event.get("optimizer_update")) is not int:
                raise AgibotTraceError("AGIBOT_TRACE_TRAIN_EVENT_INVALID")
            if type(event.get("microstep")) is not int or type(event.get("batch_slot")) is not int:
                raise AgibotTraceError("AGIBOT_TRACE_TRAIN_EVENT_INVALID")
            if (
                not _is_finite_number(event.get("learning_rate_min"))
                or float(event["learning_rate_min"]) <= 0
                or not _is_finite_number(event.get("learning_rate_max"))
                or float(event["learning_rate_max"]) < float(event["learning_rate_min"])
                or not _is_finite_number(event.get("parameter_delta_max_abs"))
                or float(event["parameter_delta_max_abs"]) <= 0
                or type(event.get("parameter_sample_count")) is not int
                or int(event["parameter_sample_count"]) < 1
                or type(event.get("parameter_changed_sample_count")) is not int
                or int(event["parameter_changed_sample_count"]) < 1
                or int(event["parameter_changed_sample_count"])
                > int(event["parameter_sample_count"])
            ):
                raise AgibotTraceError("AGIBOT_TRACE_PARAMETER_UPDATE_NOT_PROVEN")
        elif event.get("phase") == "validation":
            if (
                event.get("split_id") != "val"
                or type(event.get("validation_pass")) is not int
                or type(event.get("batch_index")) is not int
                or type(event.get("batch_slot")) is not int
            ):
                raise AgibotTraceError("AGIBOT_TRACE_VALIDATION_EVENT_INVALID")
        else:
            raise AgibotTraceError("AGIBOT_TRACE_PHASE_INVALID")
        if not _is_finite_number(event.get("batch_loss")):
            raise AgibotTraceError("AGIBOT_TRACE_EVENT_LOSS_INVALID")
        _validate_sample(event, split=str(event.get("split_id")))
    if sequences != list(range(1, len(sequences) + 1)):
        raise AgibotTraceError("AGIBOT_TRACE_EVENT_SEQUENCE_INVALID")


def _validate_global_events(events: Sequence[Mapping[str, object]], config: Mapping[str, object]) -> None:
    by_rank_updates: dict[int, set[int]] = {}
    validation_seen: dict[int, set[str]] = {}
    validation_order_by_rank: dict[int, list[str]] = {}
    expected_ids = set(config["validation_sample_window_ids"])
    rank_validation_passes: dict[int, set[int]] = {}
    for event in events:
        rank = int(event["rank"])
        if event["phase"] == "train":
            by_rank_updates.setdefault(rank, set()).add(int(event["optimizer_update"]))
        else:
            sample_id = str(event["sample_window_id"])
            if sample_id not in expected_ids:
                raise AgibotTraceError("AGIBOT_TRACE_VALIDATION_OUTSIDE_FROZEN_SUPPORT")
            validation_pass = int(event["validation_pass"])
            if validation_pass != 1 or int(event.get("optimizer_update", -1)) != int(config["expected_optimizer_updates"]):
                raise AgibotTraceError("AGIBOT_TRACE_VALIDATION_BEFORE_FINAL_UPDATE")
            validation_seen.setdefault(validation_pass, set()).add(sample_id)
            validation_order_by_rank.setdefault(rank, []).append(sample_id)
            rank_validation_passes.setdefault(rank, set()).add(validation_pass)
    if by_rank_updates:
        update_sets = list(by_rank_updates.values())
        if any(updates != update_sets[0] for updates in update_sets[1:]):
            raise AgibotTraceError("AGIBOT_TRACE_RANK_UPDATE_MISMATCH")
        expected_updates = set(range(1, int(config["expected_optimizer_updates"]) + 1))
        if update_sets[0] != expected_updates:
            raise AgibotTraceError("AGIBOT_TRACE_EXPECTED_UPDATE_COUNT_MISMATCH")
    else:
        raise AgibotTraceError("AGIBOT_TRACE_TRAIN_UPDATE_MISSING")
    if not validation_seen:
        raise AgibotTraceError("AGIBOT_TRACE_VALIDATION_PASS_MISSING")
    if set(rank_validation_passes) != set(range(int(config["world_size"]))) or any(
        passes != {1} for passes in rank_validation_passes.values()
    ):
        raise AgibotTraceError("AGIBOT_TRACE_RANK_VALIDATION_PASS_MISSING")
    expected_order = list(config["validation_sample_window_ids"])
    if any(
        validation_order_by_rank.get(rank) != expected_order
        for rank in range(int(config["world_size"]))
    ):
        raise AgibotTraceError("AGIBOT_TRACE_VALIDATION_ORDER_MISMATCH")
    for seen in validation_seen.values():
        if seen != expected_ids:
            raise AgibotTraceError("AGIBOT_TRACE_VALIDATION_COVERAGE_INCOMPLETE")

    _validate_train_batch_shape(events, config)
    _validate_rank_validation_losses(events, config)


def _validate_train_batch_shape(
    events: Sequence[Mapping[str, object]], config: Mapping[str, object]
) -> None:
    """Prove every planned optimizer update consumed complete ordered batches."""

    batch_size = int(config["batch_size"])
    expected_updates = int(config["expected_optimizer_updates"])
    expected_microsteps = expected_updates * int(config["gradient_accumulation"])
    by_rank_microstep: dict[tuple[int, int], list[Mapping[str, object]]] = {}
    by_rank_update_microsteps: dict[tuple[int, int], set[int]] = {}
    for event in events:
        if event.get("phase") != "train":
            continue
        rank = int(event["rank"])
        update = int(event["optimizer_update"])
        microstep = int(event["microstep"])
        if (
            update < 1
            or update > expected_updates
            or microstep < 1
            or microstep > expected_microsteps
            or update != (microstep - 1) // int(config["gradient_accumulation"]) + 1
        ):
            raise AgibotTraceError("AGIBOT_TRACE_TRAIN_STEP_BINDING_INVALID")
        by_rank_microstep.setdefault((rank, microstep), []).append(event)
        by_rank_update_microsteps.setdefault((rank, update), set()).add(microstep)

    expected_ranks = range(int(config["world_size"]))
    accumulation = int(config["gradient_accumulation"])
    expected_slots = list(range(batch_size))
    for rank in expected_ranks:
        for update in range(1, expected_updates + 1):
            expected_steps = set(
                range((update - 1) * accumulation + 1, update * accumulation + 1)
            )
            if by_rank_update_microsteps.get((rank, update)) != expected_steps:
                raise AgibotTraceError("AGIBOT_TRACE_TRAIN_MICROSTEP_COVERAGE_INVALID")
            for microstep in sorted(expected_steps):
                rows = sorted(
                    by_rank_microstep.get((rank, microstep), []),
                    key=lambda row: int(row["batch_slot"]),
                )
                if (
                    len(rows) != batch_size
                    or [int(row["batch_slot"]) for row in rows] != expected_slots
                ):
                    raise AgibotTraceError("AGIBOT_TRACE_TRAIN_BATCH_COVERAGE_INVALID")


def _validate_rank_validation_losses(
    events: Sequence[Mapping[str, object]], config: Mapping[str, object]
) -> None:
    """Ensure each rank saw the same fixed-panel loss for every sample."""

    losses: dict[str, list[float]] = {}
    for event in events:
        if event.get("phase") == "validation":
            losses.setdefault(str(event["sample_window_id"]), []).append(
                float(event["batch_loss"])
            )
    if any(
        len(values) != int(config["world_size"])
        or any(
            not math.isclose(values[0], value, rel_tol=1e-7, abs_tol=1e-9)
            for value in values[1:]
        )
        for values in losses.values()
    ):
        raise AgibotTraceError("AGIBOT_TRACE_RANK_VALIDATION_LOSS_MISMATCH")


def _validate_header(header: Mapping[str, object], config: Mapping[str, object]) -> None:
    if (
        header.get("schema_version") != SCHEMA_VERSION
        or header.get("trace_id") != config["trace_id"]
        or header.get("support_digest") != config["support_digest"]
        or header.get("source_fingerprint") != config["source_fingerprint"]
    ):
        raise AgibotTraceError("AGIBOT_TRACE_HEADER_BINDING_INVALID")


def _validate_config(config: Mapping[str, object]) -> None:
    validation_ids = config.get("validation_sample_window_ids")
    if (
        config.get("schema_version") != SCHEMA_VERSION
        or config.get("artifact_type") != TRACE_CONFIG_TYPE
        or not isinstance(config.get("trace_id"), str)
        or not config.get("trace_id")
        or not _is_sha256(config.get("support_digest"))
        or not _is_sha256(config.get("source_fingerprint"))
        or not _is_sha256(config.get("native_config_sha256"))
        or type(config.get("world_size")) is not int
        or int(config["world_size"]) < 1
        or type(config.get("batch_size")) is not int
        or int(config["batch_size"]) < 1
        or type(config.get("expected_optimizer_updates")) is not int
        or int(config["expected_optimizer_updates"]) < 1
        or type(config.get("seed")) is not int
        or int(config["seed"]) < 0
        or not isinstance(config.get("support_bindings"), Mapping)
        or type(config.get("gradient_accumulation")) is not int
        or int(config["gradient_accumulation"]) < 1
        or not isinstance(config.get("output_root"), str)
        or not Path(str(config.get("output_root"))).is_absolute()
        or not isinstance(validation_ids, list)
        or not validation_ids
        or any(not isinstance(item, str) or not item for item in validation_ids)
        or len(validation_ids) != len(set(validation_ids))
        or config.get("validation_padding_rule")
        != "complete_frozen_panel_per_rank_v1"
    ):
        raise AgibotTraceError("AGIBOT_TRACE_CONFIG_INVALID")
    fingerprint_payload = {
        "support_digest": config["support_digest"],
        "native_config_sha256": config["native_config_sha256"],
    }
    init_from_path = config.get("init_from_path")
    if init_from_path is not None:
        if (
            not isinstance(init_from_path, str)
            or not Path(init_from_path).is_absolute()
        ):
            raise AgibotTraceError("AGIBOT_TRACE_CONFIG_INVALID")
        fingerprint_payload["init_from_path"] = init_from_path
    expected_fingerprint = canonical_digest(fingerprint_payload)
    if config["source_fingerprint"] != expected_fingerprint:
        raise AgibotTraceError("AGIBOT_TRACE_CONFIG_FINGERPRINT_MISMATCH")


def _event_sort_key(event: Mapping[str, object]) -> tuple[object, ...]:
    phase_order = 0 if event.get("phase") == "train" else 1
    position = int(event.get("optimizer_update", 0) or 0) if phase_order == 0 else int(event.get("validation_pass", 0))
    return (phase_order, position, int(event.get("rank", 0)), int(event.get("sequence", 0)))
