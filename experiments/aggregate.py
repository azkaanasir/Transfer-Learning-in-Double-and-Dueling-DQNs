"""Collect per-run metrics into one tidy table of per-seed scalars.

The manuscript's numbers were per-seed aggregates whose evaluation window was
never stated; Phase 0 identified it as the mean over the final 100 episodes
(paper/METHODS_ACTUAL.md section 4). That window is applied here explicitly and
recorded in the output, so no number in the paper is ever hand-computed again.

    python experiments/aggregate.py --out-root runs
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd


def per_seed_scalars(run_dir: str, window: int) -> dict | None:
    metrics = os.path.join(run_dir, 'metrics.csv')
    manifest = os.path.join(run_dir, 'manifest.json')
    if not (os.path.exists(metrics) and os.path.exists(manifest)):
        return None

    df = pd.read_csv(metrics)
    if df.empty:
        return None
    with open(manifest, encoding='utf-8') as fh:
        man = json.load(fh)
    cfg = man['config']

    tail = df.tail(window)
    evals = df['eval_reward'].dropna()
    losses = df['loss'].dropna()

    row = {
        'arm': f"{cfg['arch']}-{cfg['target_rule']}-{cfg['mode']}",
        'arch': cfg['arch'],
        'target_rule': cfg['target_rule'],
        'mode': cfg['mode'],
        'env_id': cfg['env_id'],
        'seed': cfg['seed'],
        'lr': cfg['lr'],
        'freeze_episodes': cfg['freeze_episodes'],
        'episodes_completed': len(df),
        'eval_window': window,
        # headline metrics, one scalar per seed
        'episode_reward': float(tail['reward'].mean()),
        'episode_length': float(tail['length'].mean()),
        'validation_reward': float(evals.tail(max(1, window // 10)).mean())
        if len(evals) else np.nan,
        'training_loss': float(losses.tail(window).mean()) if len(losses) else np.nan,
        'updates': int(df['updates'].iloc[-1]),
        'wall_time_s': float(df['wall_time'].iloc[-1]),
    }
    for col in ('v_abs_mean', 'a_abs_mean', 'a_spread'):
        vals = df[col].dropna() if col in df else pd.Series(dtype=float)
        row[col] = float(vals.tail(10).mean()) if len(vals) else np.nan
    return row


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out-root', default='runs')
    p.add_argument('--window', type=int, default=100,
                   help='evaluation window in episodes (default 100)')
    p.add_argument('--output', default=None)
    args = p.parse_args(argv)

    dirs = sorted(glob.glob(os.path.join(args.out_root, '*', '*')))
    rows = [r for r in (per_seed_scalars(d, args.window) for d in dirs) if r]
    if not rows:
        print(f'No completed runs found under {args.out_root}/')
        return 1

    df = pd.DataFrame(rows).sort_values(['env_id', 'arm', 'seed'])
    out = args.output or os.path.join(args.out_root, 'per_seed.csv')
    df.to_csv(out, index=False)

    pd.set_option('display.width', 200, 'display.max_columns', 30)
    print(f'{len(df)} runs -> {out}\n')
    summary = (df.groupby(['env_id', 'arm'])
                 .agg(n=('seed', 'count'),
                      episode_reward_mean=('episode_reward', 'mean'),
                      episode_reward_sd=('episode_reward', 'std'),
                      validation_mean=('validation_reward', 'mean'),
                      validation_sd=('validation_reward', 'std'))
                 .round(2))
    print(summary.to_string())
    return 0


if __name__ == '__main__':
    sys.exit(main())
