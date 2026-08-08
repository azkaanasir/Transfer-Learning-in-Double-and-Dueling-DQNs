"""Phase 0.1 — recover ground truth from the tracked TensorBoard event files.

The `logs/` tree is tracked in git but excluded from the working copy by a
sparse-checkout rule, because several run directories contain colons in their
names and cannot be checked out on Windows. This script reads the blobs
straight out of git (never touching sparse-checkout), decodes the TF2
tensor-encoded scalars, and writes two CSVs:

    _out/run_summary.csv   one row per run
    _out/run_series.csv    tidy per-step series

Usage:  python analysis/extract_logs.py
"""
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(REPO, 'analysis', '_logs')
OUT = os.path.join(REPO, 'analysis', '_out')

TAGS = ['episode/reward', 'episode/length', 'agent/epsilon',
        'validation/avg_reward', 'train/loss']


def git(*args):
    return subprocess.run(['git', '-C', REPO, *args],
                          capture_output=True, check=True).stdout


def export_logs():
    """Materialise logs/ from git into WORK, sanitising illegal filenames."""
    paths = git('ls-files', 'logs/').decode().splitlines()
    if not paths:
        sys.exit('No logs/ paths tracked in git — nothing to extract.')

    exported = []
    for p in paths:
        safe = p.replace(':', '-').replace(' ', '_')
        dest = os.path.join(WORK, *safe.split('/'))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if not os.path.exists(dest):
            # cat-file bypasses smudge/eol filters; `git show` would corrupt
            # these binary protobufs under the repo's `* text=auto` attribute.
            sha = git('rev-parse', f'HEAD:{p}').decode().strip()
            with open(dest, 'wb') as fh:
                fh.write(git('cat-file', 'blob', sha))
        exported.append(dest)
    print(f'{len(exported)} event files available under {WORK}')
    return exported


def to_float(tp):
    """Decode a scalar TensorProto without requiring TensorFlow."""
    if len(tp.float_val):
        return float(tp.float_val[0])
    if tp.tensor_content:
        return float(np.frombuffer(tp.tensor_content, dtype=np.float32)[0])
    if len(tp.double_val):
        return float(tp.double_val[0])
    return float('nan')


def read_run(path):
    """Return {tag: [(step, value), ...]} for one event file."""
    out = {t: [] for t in TAGS}
    for event in EventFileLoader(path).Load():
        for v in event.summary.value:
            if v.tag in out:
                out[v.tag].append((event.step, to_float(v.tensor)))
    return out


def main():
    files = sorted(export_logs())
    os.makedirs(OUT, exist_ok=True)

    rows, series = [], []
    for i, f in enumerate(files, 1):
        run = os.path.basename(os.path.dirname(f))
        group = os.path.basename(os.path.dirname(os.path.dirname(f)))
        print(f'[{i}/{len(files)}] {group}/{run}', flush=True)

        data = read_run(f)
        ep, val = data['episode/reward'], data['validation/avg_reward']
        eps, loss = data['agent/epsilon'], data['train/loss']

        rows.append({
            'group': group, 'run': run,
            'n_episodes': len(ep), 'n_val': len(val), 'n_loss': len(loss),
            'final_epsilon': eps[-1][1] if eps else np.nan,
            # the manuscript's evaluation window, identified in METHODS_ACTUAL.md
            'ep_reward_last100': np.mean([v for _, v in ep[-100:]]) if ep else np.nan,
            'val_reward_last10': np.mean([v for _, v in val[-10:]]) if val else np.nan,
            'loss_mean_last1000': np.mean([v for _, v in loss[-1000:]]) if loss else np.nan,
        })
        for tag in TAGS:
            if tag == 'train/loss':
                continue  # ~150k points per run; summarised above instead
            for step, value in data[tag]:
                series.append({'group': group, 'run': run, 'tag': tag,
                               'step': step, 'value': value})

    summary = pd.DataFrame(rows).sort_values(['group', 'run'])
    summary.to_csv(os.path.join(OUT, 'run_summary.csv'), index=False)
    pd.DataFrame(series).to_csv(os.path.join(OUT, 'run_series.csv'), index=False)

    pd.set_option('display.width', 220, 'display.max_columns', 40,
                  'display.max_rows', 100)
    print('\n=== PER-RUN SUMMARY ===')
    print(summary.to_string(index=False))
    print(f'\nWrote run_summary.csv and run_series.csv to {OUT}')


if __name__ == '__main__':
    sys.exit(main())
