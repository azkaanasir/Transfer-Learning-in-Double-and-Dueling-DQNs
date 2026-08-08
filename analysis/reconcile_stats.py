"""Phase 0.1/0.2/0.6 (run `extract_logs.py` first)
 — find which runs + which aggregation window reproduce the
manuscript's reported per-arm statistics.

Searches over candidate evaluation windows and, where a group has more complete
runs than the paper's n=5, over which subset of runs was used.
"""
import itertools
import os

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, 'analysis', '_out')

# Manuscript §V.A reported values: (mean, sd)
PAPER = {
    'ddqn_baseline':     {'episode': (171.01, 37.53),  'validation': (160.61, 50.61)},
    'dueling_baseline':  {'episode': (62.94, 148.63),  'validation': (72.24, 115.29)},
    'ddqn_transfer':     {'episode': (212.18, 17.13),  'validation': (182.60, 31.63)},
    'dueling_transfer':  {'episode': (-145.12, 19.18), 'validation': (-156.87, 15.03)},
}

# log group -> paper arm
GROUP_TO_ARM = {
    'lunarlander_ddqn': 'ddqn_baseline',
    'lunarlander_dqn_duelling': 'dueling_baseline',
    'transfer_ddqn': 'ddqn_transfer',
    'transfer_dqn_dueling': 'dueling_transfer',
}

TAG = {'episode': 'episode/reward', 'validation': 'validation/avg_reward'}


def per_run_scalar(df, group, tag, window):
    """One scalar per run: mean of the last `window` points (None = all).

    Completeness threshold is tag-dependent: episode series have one point per
    episode (~500), validation series one per 10 episodes (~50).
    """
    min_len = 40 if 'validation' in tag else 400
    sub = df[(df.group == group) & (df.tag == tag)]
    out = {}
    for run, g in sub.groupby('run'):
        v = g.sort_values('step')['value'].to_numpy()
        if len(v) < min_len:      # keep only complete runs
            continue
        out[run] = float(np.mean(v if window is None else v[-window:]))
    return out


def main():
    df = pd.read_csv(os.path.join(OUT, 'run_series.csv'))
    windows_ep = [None, 500, 250, 100, 50, 25, 10]
    windows_val = [None, 50, 25, 20, 10, 5, 3]

    print('Searching for (run subset, window) reproducing each reported arm.')
    print('Reported = manuscript §V.A;  match = |Δmean| and |Δsd| both small.\n')

    for group, arm in GROUP_TO_ARM.items():
        print('=' * 78)
        print(f'{arm}   (log group: {group})')
        for metric in ['episode', 'validation']:
            tgt_mean, tgt_sd = PAPER[arm][metric]
            windows = windows_ep if metric == 'episode' else windows_val
            best = []
            for w in windows:
                vals = per_run_scalar(df, group, TAG[metric], w)
                runs = sorted(vals)
                if not runs:
                    continue
                # try every 5-run subset (usually exactly one)
                for subset in itertools.combinations(runs, min(5, len(runs))):
                    x = np.array([vals[r] for r in subset])
                    m, s = x.mean(), x.std(ddof=1)
                    err = abs(m - tgt_mean) / max(abs(tgt_mean), 1) + \
                          abs(s - tgt_sd) / max(abs(tgt_sd), 1)
                    best.append((err, w, subset, m, s))
            best.sort(key=lambda t: t[0])
            print(f'\n  {metric}: reported {tgt_mean:+.2f} ± {tgt_sd:.2f}'
                  f'   (n complete runs = {len(sorted(per_run_scalar(df, group, TAG[metric], None)))})')
            for err, w, subset, m, s in best[:3]:
                wlabel = 'all' if w is None else f'last{w}'
                print(f'    {m:+8.2f} ± {s:7.2f}  window={wlabel:7s} err={err:.3f}'
                      f'  runs={[r[-8:] for r in subset]}')
        print()

    inferential(df)


def inferential(df):
    """Reproduce the manuscript's §V.B inferential tests from recovered runs."""
    from scipy import stats

    print('=' * 78)
    print('INFERENTIAL TESTS — recovered vs reported (§V.B)')
    print('Per-seed validation reward, mean of final 10 validation points.\n')

    arms = {}
    for group, arm in GROUP_TO_ARM.items():
        vals = per_run_scalar(df, group, TAG['validation'], 10)
        arms[arm] = np.array([vals[r] for r in sorted(vals)])
        print(f'  {arm:18s} n={len(arms[arm])}  '
              f'{arms[arm].mean():+8.2f} ± {arms[arm].std(ddof=1):6.2f}  '
              f'{np.round(arms[arm], 1).tolist()}')

    comparisons = [
        ('ddqn_transfer', 'dueling_transfer', 'U=25.0, p=0.0079', 'RQ1'),
        ('ddqn_baseline', 'ddqn_transfer',    'U=8.0,  p=0.421',  'RQ2'),
        ('ddqn_baseline', 'dueling_transfer', 'U=25.0, p=0.0079', 'neg. transfer'),
    ]
    print('\n  Mann-Whitney U (two-sided):')
    for a, b, reported, rq in comparisons:
        if len(arms[a]) < 2 or len(arms[b]) < 2:
            continue
        u, p = stats.mannwhitneyu(arms[a], arms[b], alternative='two-sided')
        print(f'    {a} vs {b}:  U={u:.1f}, p={p:.4f}'
              f'   | reported {reported}  ({rq})')

    a, b = arms['ddqn_transfer'], arms['dueling_transfer']
    if len(a) > 1 and len(b) > 1:
        w, p = stats.levene(a, b, center='median')   # Brown-Forsythe
        print(f'\n  Brown-Forsythe (transfer variance): W={w:.3f}, p={p:.4f}'
              f'   | reported p=0.121  (RQ3)')
        print(f'    sd: ddqn_transfer={a.std(ddof=1):.2f}, '
              f'dueling_transfer={b.std(ddof=1):.2f}'
              f'   -> paper claims Dueling has "broader spread"')


if __name__ == '__main__':
    main()
