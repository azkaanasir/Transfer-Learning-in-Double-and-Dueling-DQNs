"""Validate the environment and estimate runtime before committing to a sweep.

Run this first on any new machine. It is cheap (about a minute) and catches the
failures that otherwise waste a whole session -- most commonly Box2D failing to
build, which breaks LunarLander but not CartPole, so a naive smoke test on
CartPole alone would pass and the sweep would die hours later.

    python experiments/preflight.py
    python experiments/preflight.py --estimate-only

Checks, in order of how expensive they are to discover late:
  1. imports and versions
  2. both environments actually construct and step
  3. every one of the four cells builds, trains, and transfers
  4. measured throughput -> a runtime estimate for the full sweep
"""
from __future__ import annotations

import argparse
import os
import platform
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OK, BAD = '[ok]', '[FAIL]'


def check_imports() -> bool:
    print('== 1. imports ==')
    ok = True
    try:
        import tensorflow as tf
        print(f'{OK} tensorflow {tf.__version__}')
        gpus = tf.config.list_physical_devices('GPU')
        # Deliberately informational: these nets are 128x128 at batch 64, so
        # kernel-launch overhead dominates and a GPU is typically no faster --
        # sometimes slower. Cores matter, accelerators do not.
        print(f'     devices: {len(gpus)} GPU, CPU cores {os.cpu_count()}')
    except Exception as exc:                            # noqa: BLE001
        print(f'{BAD} tensorflow: {exc}')
        ok = False
    for mod in ('gymnasium', 'numpy', 'scipy', 'pandas'):
        try:
            m = __import__(mod)
            print(f'{OK} {mod} {getattr(m, "__version__", "?")}')
        except Exception as exc:                        # noqa: BLE001
            print(f'{BAD} {mod}: {exc}')
            ok = False
    print(f'     python {platform.python_version()} on {platform.system()} '
          f'{platform.machine()}')
    return ok


def check_envs() -> bool:
    print('\n== 2. environments ==')
    import gymnasium as gym
    ok = True
    for env_id, want_obs, want_act in (('CartPole-v1', 4, 2),
                                       ('LunarLander-v3', 8, 4)):
        try:
            env = gym.make(env_id)
            obs, _ = env.reset(seed=0)
            env.step(env.action_space.sample())
            n_obs = int(env.observation_space.shape[0])
            n_act = int(env.action_space.n)
            if (n_obs, n_act) != (want_obs, want_act):
                print(f'{BAD} {env_id}: expected obs={want_obs} act={want_act}, '
                      f'got obs={n_obs} act={n_act}')
                ok = False
            else:
                print(f'{OK} {env_id}  obs={n_obs} act={n_act}')
            env.close()
        except Exception as exc:                        # noqa: BLE001
            hint = ''
            if 'box2d' in str(exc).lower() or 'Box2D' in str(exc):
                hint = "  -> pip install 'gymnasium[box2d]' swig"
            print(f'{BAD} {env_id}: {exc}{hint}')
            ok = False
    return ok


def check_cells(episodes: int = 3) -> bool:
    """Build, train and transfer each of the four cells."""
    print('\n== 3. all four cells ==')
    import warnings
    warnings.filterwarnings('ignore')
    import numpy as np
    from src.dqn.agent import Agent
    from src.dqn.config import Config
    from src.dqn.networks import build_q_network
    from src.dqn.transfer import transfer_weights

    ok = True
    for arch in ('mlp', 'dueling'):
        for rule in ('vanilla', 'double'):
            label = f'{arch}-{rule}'
            try:
                cfg = Config(arch=arch, target_rule=rule, mode='scratch',
                             seed=0, learning_starts=32)
                agent = Agent(cfg, 8, 4)
                rng = np.random.default_rng(0)
                for _ in range(64):
                    agent.buffer.add(rng.normal(size=8).astype('float32'),
                                     int(rng.integers(4)), float(rng.normal()),
                                     rng.normal(size=8).astype('float32'), False)
                losses = [agent.train_step() for _ in range(3)]
                if not all(np.isfinite(losses)):
                    raise RuntimeError(f'non-finite loss: {losses}')

                # transfer from a CartPole-shaped source of the same arch
                src = build_q_network(4, 2, arch)
                rep = transfer_weights(agent.online, src,
                                       ('trunk_fc1', 'trunk_fc2'), 'partial')
                actions = {r['action'] for r in rep}
                if not actions <= {'copied', 'partial'}:
                    raise RuntimeError(f'unexpected transfer actions: {actions}')

                # freeze / unfreeze round-trip, incl. the optimiser boundary
                before = agent.set_frozen(('trunk_fc1', 'trunk_fc2'), True)
                agent.train_step()
                after = agent.set_frozen(('trunk_fc1', 'trunk_fc2'), False)
                agent.train_step()
                if not before['trainable_params'] < after['trainable_params']:
                    raise RuntimeError('freeze did not reduce trainable params')
                print(f'{OK} {label:16s} trains, transfers, freezes '
                      f'({before["trainable_params"]} -> '
                      f'{after["trainable_params"]} params)')
            except Exception as exc:                    # noqa: BLE001
                print(f'{BAD} {label:16s} {type(exc).__name__}: {exc}')
                ok = False
    return ok


def estimate(measure: bool = True) -> None:
    print('\n== 4. runtime estimate ==')
    rate = None
    if measure:
        import warnings
        warnings.filterwarnings('ignore')
        import numpy as np
        from src.dqn.agent import Agent
        from src.dqn.config import Config

        cfg = Config(arch='dueling', target_rule='double', mode='scratch',
                     seed=0, learning_starts=32)
        agent = Agent(cfg, 8, 4)
        rng = np.random.default_rng(0)
        for _ in range(256):
            agent.buffer.add(rng.normal(size=8).astype('float32'),
                             int(rng.integers(4)), float(rng.normal()),
                             rng.normal(size=8).astype('float32'), False)
        agent.train_step()                       # exclude tracing from timing
        t0 = time.time()
        for _ in range(300):
            agent.train_step()
        rate = 300 / (time.time() - t0)
        print(f'     measured {rate:.0f} gradient steps/s (this machine, 1 process)')

    if not rate:
        return
    # Env steps per run, from the recovered published runs: CartPole ~100k,
    # LunarLander ~150k-270k for 500 episodes; doubled for the 1000-episode
    # budget. One gradient step per env step.
    cart, lunar = 200_000, 420_000
    n_cart, n_lunar = 40, 80
    total_h = (n_cart * cart + n_lunar * lunar) / rate / 3600
    print(f'     full sweep (120 runs @ 1000 episodes): '
          f'~{total_h:.1f} h on one process')
    cores = os.cpu_count() or 4
    for jobs in sorted({2, 4, cores}):
        if jobs <= cores:
            print(f'       --jobs {jobs:<2d} -> ~{total_h / jobs:.1f} h wall clock')
    print('     (rough: throughput varies with episode length as agents improve)')


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--estimate-only', action='store_true')
    args = p.parse_args(argv)

    if args.estimate_only:
        estimate()
        return 0

    results = [check_imports(), check_envs(), check_cells()]
    estimate()

    print()
    if all(results):
        print('PREFLIGHT PASSED -- safe to launch the sweep.')
        print('  python experiments/sweep.py --seeds 0-9 --stage all --jobs 4')
        return 0
    print('PREFLIGHT FAILED -- fix the above before spending compute.')
    return 1


if __name__ == '__main__':
    sys.exit(main())
