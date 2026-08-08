"""Sweep driver for the 2x2 design.

The design (decided 2026-08-08, see paper/METHODS_ACTUAL.md section 6):

    {mlp, dueling} x {vanilla, double}  =  4 cells
      per cell, per seed:
        1. source   -- CartPole-v1,     scratch
        2. baseline -- LunarLander-v3,  scratch
        3. transfer -- LunarLander-v3,  transfer from that seed's own source

Transfer always draws from the source run with the *same* seed and the *same*
cell, so no cross-contamination between cells is possible.

    python experiments/sweep.py --seeds 0 1 2 --stage all
    python experiments/sweep.py --seeds 0-9 --stage transfer --dry-run

Runs are resumable and idempotent: a run whose manifest already reports the full
episode count is skipped, so re-invoking after a Colab timeout continues rather
than restarting.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dqn.config import Config          # noqa: E402
from src.dqn.train import train            # noqa: E402

ARCHS = ('mlp', 'dueling')
RULES = ('vanilla', 'double')
SOURCE_ENV = 'CartPole-v1'
TARGET_ENV = 'LunarLander-v3'


def parse_seeds(tokens) -> list[int]:
    """Accept '0 1 2' or '0-9' or a mix."""
    seeds: list[int] = []
    for tok in tokens:
        if '-' in str(tok):
            lo, hi = str(tok).split('-')
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(tok))
    return sorted(set(seeds))


def source_checkpoint(out_root: str, arch: str, rule: str, seed: int) -> str:
    cfg = Config(arch=arch, target_rule=rule, mode='scratch',
                 env_id=SOURCE_ENV, seed=seed, out_root=out_root)
    return os.path.join(cfg.run_dir(), 'model.keras')


def already_done(cfg: Config) -> bool:
    path = os.path.join(cfg.run_dir(), 'manifest.json')
    if not os.path.exists(path):
        return False
    try:
        with open(path, encoding='utf-8') as fh:
            result = json.load(fh).get('result') or {}
        return result.get('episodes_completed', 0) >= cfg.num_episodes
    except Exception:                                  # noqa: BLE001
        return False


def build_jobs(seeds, stage, out_root, episodes, lr, freeze_episodes,
               archs=ARCHS, rules=RULES):
    jobs = []
    for arch, rule, seed in itertools.product(archs, rules, seeds):
        common = dict(arch=arch, target_rule=rule, seed=seed,
                      out_root=out_root, lr=lr)
        if stage in ('all', 'source'):
            jobs.append(('source', Config(mode='scratch', env_id=SOURCE_ENV,
                                          num_episodes=episodes, **common)))
        if stage in ('all', 'baseline'):
            jobs.append(('baseline', Config(mode='scratch', env_id=TARGET_ENV,
                                            num_episodes=episodes, **common)))
        if stage in ('all', 'transfer'):
            jobs.append(('transfer', Config(
                mode='transfer', env_id=TARGET_ENV, num_episodes=episodes,
                source_checkpoint=source_checkpoint(out_root, arch, rule, seed),
                freeze_episodes=freeze_episodes, **common)))
    return jobs


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--seeds', nargs='+', default=['0-9'],
                   help="seeds, e.g. '0 1 2' or '0-9' (default 0-9)")
    p.add_argument('--stage', default='all',
                   choices=['all', 'source', 'baseline', 'transfer'],
                   help='which stage to run; transfer requires source done')
    p.add_argument('--archs', nargs='+', default=None, choices=list(ARCHS))
    p.add_argument('--rules', nargs='+', default=None, choices=list(RULES))
    p.add_argument('--episodes', type=int, default=1000,
                   help='1000, not the manuscript 500: epsilon does not floor '
                        'until ~ep 891, so 500 measures arms mid-exploration')
    p.add_argument('--lr', type=float, default=5e-4,
                   help='shared by every arm -- this is the control claim')
    p.add_argument('--freeze-episodes', type=int, default=100)
    p.add_argument('--out-root', default='runs')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--force', action='store_true',
                   help='re-run even if a completed manifest exists')
    args = p.parse_args(argv)

    archs = tuple(args.archs) if args.archs else ARCHS
    rules = tuple(args.rules) if args.rules else RULES

    seeds = parse_seeds(args.seeds)
    jobs = build_jobs(seeds, args.stage, args.out_root, args.episodes,
                      args.lr, args.freeze_episodes, archs, rules)

    print(f'{len(jobs)} jobs = {len(archs)}arch x {len(rules)}rule '
          f'x {len(seeds)}seeds x stages({args.stage})')
    pending = [(s, c) for s, c in jobs if args.force or not already_done(c)]
    print(f'{len(jobs) - len(pending)} already complete, {len(pending)} to run\n')

    if args.dry_run:
        for stage, cfg in pending:
            print(f'  [{stage:8s}] {cfg.run_id():34s} {cfg.env_id}')
        return 0

    started = time.time()
    for i, (stage, cfg) in enumerate(pending, 1):
        if stage == 'transfer' and not os.path.exists(cfg.source_checkpoint):
            print(f'[{i}/{len(pending)}] SKIP {cfg.run_id()} -- '
                  f'source missing: {cfg.source_checkpoint}')
            continue
        print(f'\n[{i}/{len(pending)}] {stage}: {cfg.run_id()} on {cfg.env_id}')
        t0 = time.time()
        manifest = train(cfg)
        key = f'ep_reward_last{cfg.eval_window}'
        print(f'    done in {time.time() - t0:.0f}s  '
              f'{key}={manifest["result"].get(key)}')

    print(f'\nsweep finished in {(time.time() - started) / 60:.1f} min')
    return 0


if __name__ == '__main__':
    sys.exit(main())
