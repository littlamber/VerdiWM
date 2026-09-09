#!/usr/bin/env python3
"""Seed one CoachWorld episode process without editing the external repository.

This adapter is an execution component, not a scheduler: the caller must bind
an approved checkpoint, data, evaluator, GPU lease and budget before launching.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import random
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('rollout_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    tokens = args.rollout_args
    if tokens[:1] == ['--']:
        tokens = tokens[1:]
    # One episode per process makes the sampling seed independent of job order.
    if '--sample_indices' not in tokens:
        parser.error('one explicit --sample_indices value is required')
    index = tokens.index('--sample_indices') + 1
    if index >= len(tokens) or not tokens[index].isdigit():
        parser.error('--sample_indices must identify exactly one episode')
    if not 0 <= args.seed < 2**32:
        parser.error('seed must be an unsigned 32-bit integer')
    source = args.source.resolve(strict=True)
    script = source/'scripts/evaluation/rollout_full_episode_sa_wm.py'
    if not script.is_file():
        parser.error('CoachWorld full-episode entrypoint is missing')
    import numpy as np
    import torch
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # Fixed RNG streams do not guarantee bitwise CUDA determinism.
    sys.path.insert(0, str(source))
    sys.argv = [str(script), *tokens]
    os.chdir(source)
    runpy.run_path(str(script), run_name='__main__')


if __name__ == '__main__':
    main()
