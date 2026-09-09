#!/usr/bin/env python3
"""Read-only CoachWorld task intake using the generic episode inventory.

The task adapter normalizes CoachWorld metadata. It never imports model code,
loads checkpoint tensors, changes the external source, or starts GPU work.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wmloop.diagnose.episode_inventory import summarize_episodes
from wmloop.storage import atomic_write, canonical_bytes


def write(path, value):
    atomic_write(path, canonical_bytes(value) + b'\n')


def file_ref(path):
    return {'path':str(path), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def normalize_suite(source):
    path = source/'configs/evaluation/franka_full_episode_140_v1/suite.json'
    suite = json.loads(path.read_text())
    rows, refs = [], [file_ref(path)]
    for job in suite['jobs']:
        # Relocate only within the declared suite's local selections folder;
        # retained upstream absolute paths are not executable local bindings.
        selection = path.parent/'selections'/Path(job['selection']).name
        records = json.loads(selection.read_text())['selected']
        refs.append(file_ref(selection))
        for r in records:
            rows.append(dict(episode_uid=r['episode_uid'], split=job['split'],
                             stream_id=job['root_key'], num_frames=r['num_video_frames'],
                             first_future_frame=r['frame_now'], fps=suite['protocol']['fps']))
    return summarize_episodes(rows, horizons_seconds=[30,60,90,100]), refs


def normalize_local(root):
    rows, entries, refs = [], {}, [file_ref(root/'manifest.json')]
    for split in ('train','val'):
        path = root/f'{split}.index.jsonl'
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        entries[split] = records
        refs.append(file_ref(path))
        for r in records:
            rows.append(dict(episode_uid=r['episode_uid'], split=split, stream_id=root.name,
                             num_frames=r['time']['num_video_frames'], fps=r['time']['fps'], first_future_frame=4))
    return summarize_episodes(rows, horizons_seconds=[30,60,90,100]), entries, refs


def run(*, source: Path, data: Path, proposal: Path, output: Path):
    if output.exists():
        raise ValueError('PREFLIGHT_OUTPUT_EXISTS: choose a fresh audit directory')
    suite, suite_refs = normalize_suite(source)
    local, entries, local_refs = normalize_local(data)
    manifest = json.loads((data/'manifest.json').read_text())
    # Local validation is development data: existing checkpoint monitoring has
    # already used this split. Never relabel it as a fresh final test panel.
    selections = {}
    for horizon in (30,60,90,100):
        records = []
        for index, row in enumerate(entries['val']):
            if row['time']['num_video_frames']-4 < horizon*row['time']['fps']:
                continue
            records.append(dict(sample_index=len(records), episode_entry_index=index,
                                episode_id=row['episode_id'], episode_uid=row['episode_uid'],
                                identity=row['episode_uid'], frame_now=4,
                                num_video_frames=row['time']['num_video_frames'],
                                num_latent_frames=row['time']['num_latent_frames'],
                                text=row.get('text',{}).get('primary','')))
        selections[str(horizon)] = {'kind':'sa_wm_full_episode_root_selection', 'version':1,
                                   'split':'val', 'evaluation_status':'development_only', 'selected':records,
                                   'claim_boundary':'Duration-based panel only; not a final held-out test or consistency result.'}
    missing_k = sum(any(v.get('intrinsics') is None for v in r['views']) for rows in entries.values() for r in rows)
    problems = [
        {'code':'GPU_ALLOCATION_AND_BUDGET_REQUIRED','detail':'No GPU trial is authorized by this metadata audit.'},
        {'code':'PUBLISHED_55K_CHECKPOINT_UNBOUND','detail':'Local wan_init checkpoints are separate baselines, not the published CoachWorld v6 55k model.'},
        {'code':'FROZEN_CONSISTENCY_VERIFIER_REQUIRED','detail':'PSNR/SSIM/LPIPS alone do not implement action fidelity and revisit survival.'},
        {'code':'EXPLICIT_ROLLOUT_SEED_REQUIRED','detail':'Upstream full-episode entrypoint has no explicit seed option or manual_seed call; config seed alone must not be assumed effective.'},
    ]
    if manifest['processing'].get('resize_mode') != 'letterbox':
        problems.append({'code':'PREPROCESSING_DIFFERS_FROM_PROPOSED_BASELINE','observed':manifest['processing'].get('resize_mode'),'expected':'letterbox'})
    if missing_k:
        problems.append({'code':'CAMERA_INTRINSICS_UNVERIFIED','metadata_rows_without_intrinsics':missing_k,'detail':'External sidecars may exist; no trusted camera binding has been demonstrated.'})
    if local['split_overlaps']:
        problems.append({'code':'TRAIN_VAL_EPISODE_OVERLAP'})
    report = {
        'schema_version':1, 'artifact_type':'verdiwm-coachworld-research-preflight',
        'state':'blocked_before_gpu', 'source':str(source), 'data_root':str(data),
        'goal':'Improve Wan-based action-conditioned world-model consistency at 60 seconds; evaluate 90/100 seconds only where independent episodes support it.',
        'task_clock':{'fps':5,'first_future_frame':4,'horizons_seconds':[60,90,100]},
        'proposal':file_ref(proposal), 'metadata_sources':suite_refs+local_refs,
        'suite_summary':suite['splits'], 'suite_split_overlaps':suite['split_overlaps'],
        'local_summary':local['splits'], 'local_split_overlaps':local['split_overlaps'],
        'blockers':problems,
        'training_ablation_note':'generated_history_enabled changes training exposure. A same-checkpoint inference toggle is not an enabled/disabled training ablation.',
        'next_stages':['bind baseline and runtime assets','freeze development selection and valid target metrics','baseline plus paired inference interventions','IRG-informed competing explanations and cross-domain retrieval','real method implementation and target-side checks','paired screening and independent confirmation'],
        'side_effects':{'model_import_executed':False,'gpu_execution_started':False,'source_modified':False},
        'claim_boundary':'Audit and task preparation only; no IRG measurement, causal diagnosis, or method improvement has been established.',
    }
    output.mkdir(parents=True)
    write(output/'preflight.json',report)
    write(output/'local-inventory.json',local)
    write(output/'upstream-suite-inventory.json',suite)
    for horizon, value in selections.items():
        write(output/'selections'/f'development-{horizon}s.json',value)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--proposal',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=run(source=args.source.resolve(),data=args.data.resolve(),proposal=args.proposal.resolve(),output=args.output.resolve())
    print(json.dumps({k:report[k] for k in ('state','local_summary','suite_summary','blockers')},ensure_ascii=False,indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
