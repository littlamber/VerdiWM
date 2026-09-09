"""Model-independent horizon inventory with explicit clocks and episode identity.

Counts are metadata support, never consistency scores. Multiple views of one
physical episode contribute one independent episode, and train/eval overlap
is reported even when view names differ.
"""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence


class EpisodeInventoryError(ValueError):
    """Metadata cannot support an unambiguous horizon count."""


def summarize_episodes(
    records: Sequence[Mapping[str, object]], *, horizons_seconds: Sequence[float],
) -> dict:
    horizons = sorted(set(horizons_seconds))
    if not horizons or any(type(t) not in (int, float) or not math.isfinite(t) or t <= 0 for t in horizons):
        raise EpisodeInventoryError('EPISODE_HORIZONS_INVALID')
    groups = defaultdict(list)
    identities = defaultdict(set)
    normalized = []
    seen_streams = set()
    for record in records:
        uid, split, stream = (record.get(k) for k in ('episode_uid', 'split', 'stream_id'))
        fps, frames = record.get('fps'), record.get('num_frames')
        start = record.get('first_future_frame', 0)
        if any(not isinstance(v, str) or not v.strip() for v in (uid, split, stream)):
            raise EpisodeInventoryError('EPISODE_IDENTITY_REQUIRED')
        if type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0:
            raise EpisodeInventoryError('EPISODE_CLOCK_INVALID')
        if type(frames) is not int or frames < 1 or type(start) is not int or start < 0:
            raise EpisodeInventoryError('EPISODE_FRAME_COUNT_INVALID')
        key = (split, uid, stream)
        if key in seen_streams:
            raise EpisodeInventoryError('EPISODE_STREAM_DUPLICATE')
        seen_streams.add(key)
        row = dict(record, duration_seconds=frames/fps, forecast_seconds=max(0,frames-start)/fps)
        normalized.append(row)
        groups[split].append(row)
        identities[split].add(uid)
    summaries = {}
    for split, rows in sorted(groups.items()):
        by_episode = defaultdict(list)
        for row in rows:
            by_episode[row['episode_uid']].append(row)
        summaries[split] = {
            'stream_count':len(rows), 'unique_episode_count':len(by_episode),
            'horizons': [{
                'seconds':t,
                'recording_stream_count':sum(r['num_frames'] >= math.ceil(t*r['fps']) for r in rows),
                'forecast_stream_count':sum(r['num_frames']-r.get('first_future_frame',0) >= math.ceil(t*r['fps']) for r in rows),
                'forecast_unique_episode_count':sum(any(r['num_frames']-r.get('first_future_frame',0) >= math.ceil(t*r['fps']) for r in views) for views in by_episode.values()),
            } for t in horizons],
            'max_recording_seconds':max(r['duration_seconds'] for r in rows),
        }
    overlaps = []
    splits = sorted(identities)
    for i, left in enumerate(splits):
        for right in splits[i+1:]:
            shared = sorted(identities[left] & identities[right])
            if shared:
                overlaps.append({'splits':[left,right], 'episode_uids':shared})
    return {
        'schema_version':1, 'artifact_type':'verdiwm-episode-horizon-inventory',
        'state':'blocked' if overlaps else 'metadata_audited',
        'splits':summaries, 'split_overlaps':overlaps, 'records':normalized,
        'clock_convention':'N/fps recording duration; forecast support requires ceil(T*fps) frames after first_future_frame. This is not timestamp-span (N-1)/fps.',
        'claim_boundary':'Metadata support only. No rollout quality, actual payload validity, or checkpoint training-disjointness is inferred.',
    }
