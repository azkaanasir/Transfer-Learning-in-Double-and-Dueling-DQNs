"""Measure the reference returns every score in this study is normalised against.

    python experiments/measure_references.py                 # all catalogue envs
    python experiments/measure_references.py --env "LunarLander-v3:gravity=-4"
    python experiments/measure_references.py --check         # verify, do not write

Why this exists
---------------
Everything is reported on a normalised score,
`(return - random_return) / (threshold - random_return)`, and that requires the
random-policy return of each environment *and each variant* to be a measured,
committed quantity rather than an assumption.

Two facts established by measurement make this non-optional:

* Across the LunarLander gravity family the random-policy return moves from
  -202 to -463 and the score denominator from 402 to 663. Raw returns are
  therefore **not comparable across variants**, and a raw transfer delta would
  silently mix a change of scale into the effect being measured.
* Acrobot's registered threshold is -100 while a random policy scores -497. A
  multiplicative validity gate on the raw threshold -- "0.6 x -100 = -60" --
  would be *stricter* than solving the task. On the normalised scale the same
  0.6 gate is -259, which is what was intended.

The no-op return is measured alongside, and it is what revealed that the gravity
family confounds shift severity with task difficulty: a do-nothing policy scores
0.18 at gravity -10 but 0.55 at gravity -4, while staying flat near 0.17 across
every wind level. That is why the wind family, not gravity, carries the
shift-severity hypothesis in `DESIGN.md` §5.1.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dqn import envs                                        # noqa: E402

# Fixed by this file, not by a caller: a reference return that moved when
# someone passed a different episode count would put two runs' scores on
# different scales.
EPISODES = 100
MAX_STEPS = 1000
SEED_BASE = 1_000_000

# Every environment and variant the catalogue can reach. Measured up front so a
# sweep never dies hours in on a missing normalisation constant.
CATALOGUE: list[str] = [
    'CartPole-v1',
    'Acrobot-v1',
    'LunarLander-v3',
    # same-interface dynamics shift: wind is the primary axis, gravity secondary
    'LunarLander-v3:enable_wind=1,wind_power=7.5,turbulence_power=1.5',
    'LunarLander-v3:enable_wind=1,wind_power=15,turbulence_power=1.5',
    'LunarLander-v3:gravity=-8',
    'LunarLander-v3:gravity=-6',
    'LunarLander-v3:gravity=-4',
    # interface change at zero dynamics shift
    'LunarLander-v3:pad_obs=4,extra_actions=2,pad_mode=noise',
    'CartPole-v1:pad_obs=4,extra_actions=2,pad_mode=noise',
    # cheap same-interface family
    'CartPole-v1:length=0.75,masspole=0.15',
    'CartPole-v1:length=1.0,masspole=0.2',
]


def rollout(spec: str, policy: str, episodes: int = EPISODES,
            max_steps: int = MAX_STEPS) -> dict:
    """Mean return under a fixed reference policy, on fixed episode seeds."""
    es = envs.parse(spec)
    env, _ = envs.make(es)
    rng = np.random.default_rng(0)
    returns, lengths = [], []
    for ep in range(episodes):
        env.reset(seed=SEED_BASE + ep)
        total, steps = 0.0, 0
        for _ in range(max_steps):
            action = 0 if policy == 'noop' else int(rng.integers(0, es.act_dim))
            _, reward, term, trunc, _ = env.step(action)
            total += float(reward)
            steps += 1
            if term or trunc:
                break
        returns.append(total)
        lengths.append(steps)
    env.close()
    return {'mean': float(np.mean(returns)), 'sd': float(np.std(returns, ddof=1)),
            'median': float(np.median(returns)),
            'mean_length': float(np.mean(lengths))}


def measure(spec: str) -> dict:
    es = envs.parse(spec)
    thr = es.reward_threshold()
    if thr is None:
        raise ValueError(f'{es.canonical()}: no reward threshold registered, so '
                         f'no normalisation is defined')
    rand = rollout(es, 'random')
    noop = rollout(es, 'noop')
    denom = float(thr) - rand['mean']
    return {
        'threshold': float(thr),
        'threshold_source': ('gymnasium registry for the base environment; the '
                             'reward function is unchanged by these variants, '
                             'so what counts as solved is unchanged, while what '
                             'a random policy achieves is not'),
        'random_return': round(rand['mean'], 4),
        'random_sd': round(rand['sd'], 4),
        'random_len': round(rand['mean_length'], 2),
        'noop_return': round(noop['mean'], 4),
        'noop_len': round(noop['mean_length'], 2),
        'denominator': round(denom, 4),
        'noop_score': round((noop['mean'] - rand['mean']) / denom, 4),
        'obs_dim': es.obs_dim,
        'act_dim': es.act_dim,
        'episodes': EPISODES,
        'max_steps': MAX_STEPS,
        'seed_base': SEED_BASE,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--env', action='append', default=None,
                   help='canonical env string; repeatable. Default: the catalogue')
    p.add_argument('--check', action='store_true',
                   help='report which entries are missing or stale; write nothing')
    p.add_argument('--force', action='store_true',
                   help='re-measure entries that already exist')
    p.add_argument('--output', default=envs.REFERENCE_FILE)
    args = p.parse_args(argv)

    targets = [envs.parse(e).canonical() for e in (args.env or CATALOGUE)]
    existing = envs.load_references(args.output) if os.path.exists(args.output) else {}

    if args.check:
        missing = [t for t in targets if t not in existing]
        print(f'{len(existing)} measured, {len(missing)} missing')
        for t in missing:
            print(f'  MISSING {t}')
        return 1 if missing else 0

    out = dict(existing)
    print(f'{"env":58s} {"thr":>7s} {"random":>9s} {"noop":>9s} {"denom":>8s} '
          f'{"noop_score":>10s}')
    for spec in targets:
        if spec in out and not args.force:
            r = out[spec]
            print(f'{spec[:58]:58s} {r["threshold"]:7.1f} '
                  f'{r["random_return"]:9.2f} {r.get("noop_return", float("nan")):9.2f} '
                  f'{r.get("denominator", float("nan")):8.2f} '
                  f'{r.get("noop_score", float("nan")):10.3f}   (cached)')
            continue
        r = measure(spec)
        out[spec] = r
        print(f'{spec[:58]:58s} {r["threshold"]:7.1f} {r["random_return"]:9.2f} '
              f'{r["noop_return"]:9.2f} {r["denominator"]:8.2f} '
              f'{r["noop_score"]:10.3f}')

    with open(args.output, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    print(f'\n{len(out)} entries -> {args.output}')

    # The observation that reshaped the design, recomputed every time so it
    # cannot go stale.
    print('\nno-op score by family (a flat column means difficulty is held '
          'roughly constant across the axis):')
    for fam, keys in (
            ('wind', [k for k in out if 'wind_power' in k] + ['LunarLander-v3']),
            ('gravity', [k for k in out if 'gravity' in k] + ['LunarLander-v3'])):
        vals = [(k, out[k].get('noop_score')) for k in sorted(set(keys)) if k in out]
        print(f'  {fam:8s} ' + '  '.join(
            f'{v:.3f}' for _, v in vals if v is not None))
    return 0


if __name__ == '__main__':
    sys.exit(main())
