"""Episode-aware window scheduling for world-model training backends.

The scheduler deliberately knows nothing about tensors or a model. A manifest
record is an episode-bound training window; this module decides which record
and which in-record offset each global optimization update consumes.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any


class WindowSchedulerError(ValueError):
    """The manifest or scheduling contract is invalid."""


def build_window_schedule(
    records: Sequence[Mapping[str, Any]],
    *,
    steps: int,
    mode: str = "long",
    record_limit: int = 0,
    sampler: str = "episode_balanced",
    seed: int = 0,
    chunk_frames: int,
    sample_index: int = 0,
) -> list[tuple[int, int]]:
    """Build a deterministic ``(record_index, chunk_offset)`` update schedule.

    ``probe`` intentionally repeats one record at offset zero. ``long`` first
    visits one window from each episode, then revisits episodes as needed until
    the declared global update count is exhausted. This keeps episode coverage
    independent from manifest ordering while preserving exact reproducibility.
    """

    record_count = len(records)
    if record_count < 1 or steps < 1 or record_limit < 0 or chunk_frames < 1:
        raise WindowSchedulerError("WINDOW_SCHEDULER_INPUT_INVALID")
    if mode not in {"probe", "long"}:
        raise WindowSchedulerError("WINDOW_SCHEDULER_MODE_INVALID")
    if sampler not in {"sequential", "episode_balanced"}:
        raise WindowSchedulerError("WINDOW_SCHEDULER_SAMPLER_INVALID")
    if not 0 <= sample_index < record_count:
        raise WindowSchedulerError("WINDOW_SCHEDULER_SAMPLE_INDEX_INVALID")
    if mode == "probe":
        return [(sample_index, 0)] * steps

    eligible_count = record_count if record_limit == 0 else min(record_count, record_limit)
    if sampler == "sequential":
        eligible = list(range(eligible_count))
    else:
        by_episode: dict[str, list[int]] = {}
        for index, record in enumerate(records):
            episode_id = str(record.get("episode_id") or "")
            if not episode_id:
                raise WindowSchedulerError("WINDOW_SCHEDULER_EPISODE_ID_INVALID")
            by_episode.setdefault(episode_id, []).append(index)
        rng = random.Random(seed)
        episodes = sorted(by_episode)
        rng.shuffle(episodes)
        for indices in by_episode.values():
            rng.shuffle(indices)
        eligible = []
        round_index = 0
        while len(eligible) < eligible_count:
            added = False
            for episode_id in episodes:
                indices = by_episode[episode_id]
                if round_index < len(indices):
                    eligible.append(indices[round_index])
                    added = True
                    if len(eligible) == eligible_count:
                        break
            if not added:
                break
            round_index += 1

    offsets_by_record: dict[int, list[int]] = {}
    for record_index in eligible:
        horizon = int(records[record_index].get("horizon_frames", 0))
        max_offset = horizon - chunk_frames
        if max_offset < 0:
            raise WindowSchedulerError("WINDOW_SCHEDULER_CHUNK_EXCEEDS_RECORD")
        offsets = list(range(0, max_offset + 1, chunk_frames))
        if offsets[-1] != max_offset:
            offsets.append(max_offset)
        offsets_by_record[record_index] = offsets

    schedule: list[tuple[int, int]] = []
    for update in range(steps):
        record_index = eligible[update % len(eligible)]
        cycle = update // len(eligible)
        offsets = offsets_by_record[record_index]
        schedule.append((record_index, offsets[(cycle + record_index) % len(offsets)]))
    return schedule
