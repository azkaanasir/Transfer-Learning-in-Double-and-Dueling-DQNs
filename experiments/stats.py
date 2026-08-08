"""Inferential analysis for the 2x2 design. No number in the paper is hand-computed.

Fixes the statistical defects the review identified (REVIEW_REPORT.md sections
6.2.1-6.2.2, C10, C12):

* **Non-parametric only.** The manuscript ran a t-test and Cohen's d in section
  V.A on a metric it declared descriptive-only in section V.B, under a normality
  assumption it denied two paragraphs later. Only Mann-Whitney U and
  Brown-Forsythe appear here, plus bootstrap CIs.
* **Within-architecture transfer effects.** The published comparison was
  cross-architecture reward, which cannot separate transfer failure from an
  architecture being weaker on the target task to begin with -- ICANN #5's Q1.
  The primary effect reported here is delta(transfer - that cell's own scratch
  baseline).
* **Dispersion is reported directly**, because the manuscript's "broader spread"
  claim was backwards relative to its own SDs.

    python experiments/stats.py --per-seed runs/per_seed.csv
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

METRIC = 'validation_reward'


def bootstrap_ci(x, n_boot=10_000, alpha=0.05, seed=0):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return (float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def descriptive(df, metric):
    rows = []
    for (env, arm), g in df.groupby(['env_id', 'arm']):
        x = g[metric].dropna().to_numpy()
        lo, hi = bootstrap_ci(x)
        rows.append({'env_id': env, 'arm': arm, 'n': len(x),
                     'mean': np.mean(x) if len(x) else np.nan,
                     'sd': np.std(x, ddof=1) if len(x) > 1 else np.nan,
                     'median': np.median(x) if len(x) else np.nan,
                     'ci_lo': lo, 'ci_hi': hi})
    return pd.DataFrame(rows).round(2)


def transfer_effects(df, metric):
    """delta(transfer - scratch) within each (arch, target_rule) cell.

    This is the comparison that separates transferability from target-task
    suitability. Cross-architecture comparisons cannot.
    """
    rows = []
    target = df[df['mode'].isin(['scratch', 'transfer'])]
    for (arch, rule), g in target.groupby(['arch', 'target_rule']):
        a = g[g['mode'] == 'scratch'][metric].dropna().to_numpy()
        b = g[g['mode'] == 'transfer'][metric].dropna().to_numpy()
        if len(a) < 2 or len(b) < 2:
            continue
        u, p = stats.mannwhitneyu(b, a, alternative='two-sided')
        # rank-biserial correlation: a dispersion-free effect size that suits
        # the non-parametric framing, unlike Cohen's d
        rbc = 2 * u / (len(a) * len(b)) - 1
        rows.append({'cell': f'{arch}-{rule}', 'n_scratch': len(a),
                     'n_transfer': len(b),
                     'scratch_mean': a.mean(), 'transfer_mean': b.mean(),
                     'delta': b.mean() - a.mean(), 'U': u, 'p': p,
                     'rank_biserial': rbc,
                     'significant_at_05': bool(p < 0.05)})
    return pd.DataFrame(rows).round(4)


def pairwise(df, metric, mode):
    """Between-cell comparisons within one condition."""
    rows = []
    sub = df[df['mode'] == mode]
    cells = sorted(sub['arm'].unique())
    for a_arm, b_arm in itertools.combinations(cells, 2):
        a = sub[sub['arm'] == a_arm][metric].dropna().to_numpy()
        b = sub[sub['arm'] == b_arm][metric].dropna().to_numpy()
        if len(a) < 2 or len(b) < 2:
            continue
        u, p = stats.mannwhitneyu(a, b, alternative='two-sided')
        w, pv = stats.levene(a, b, center='median')      # Brown-Forsythe
        rows.append({'a': a_arm, 'b': b_arm,
                     'a_mean': a.mean(), 'b_mean': b.mean(),
                     'a_sd': a.std(ddof=1), 'b_sd': b.std(ddof=1),
                     'U': u, 'p_location': p,
                     'BF_W': w, 'p_dispersion': pv})
    return pd.DataFrame(rows).round(4)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--per-seed', default=os.path.join('runs', 'per_seed.csv'))
    p.add_argument('--metric', default=METRIC)
    p.add_argument('--env', default='LunarLander-v3')
    args = p.parse_args(argv)

    if not os.path.exists(args.per_seed):
        print(f'{args.per_seed} not found -- run experiments/aggregate.py first')
        return 1

    df = pd.read_csv(args.per_seed)
    df = df[df['env_id'] == args.env]
    if df.empty:
        print(f'No rows for env {args.env}')
        return 1

    pd.set_option('display.width', 220, 'display.max_columns', 30)
    print(f'metric = {args.metric}   env = {args.env}\n')
    print('=== DESCRIPTIVE (bootstrap 95% CI on the mean) ===')
    print(descriptive(df, args.metric).to_string(index=False))

    print('\n=== TRANSFER EFFECT, within each cell (the primary result) ===')
    eff = transfer_effects(df, args.metric)
    print(eff.to_string(index=False) if len(eff) else '  (need both arms per cell)')

    for mode in ('scratch', 'transfer'):
        print(f'\n=== BETWEEN-CELL, {mode} condition ===')
        pw = pairwise(df, args.metric, mode)
        print(pw.to_string(index=False) if len(pw) else '  (insufficient data)')

    print('\nNote: with n=5 per group the smallest attainable two-sided '
          'Mann-Whitney p is 0.0079; at n=10 it is ~1.1e-5.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
