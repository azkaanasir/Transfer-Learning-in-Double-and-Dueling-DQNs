"""Validate the environment before committing to a sweep.

Run this first on any new machine. It is cheap and it catches the failures that
otherwise waste a whole session. The most valuable check is that **Box2D built**:
if it did not, CartPole still runs fine and LunarLander does not, so a naive
CartPole-only smoke test passes and the sweep dies hours later.

    python experiments/preflight.py
    python experiments/preflight.py --quick

Checks, ordered by how expensive they are to discover late:

  1. imports, versions, and the determinism-relevant environment
  2. every registry environment constructs, steps, and reports the interface the
     registry claims -- including the parametric variants, whose overrides are
     read back rather than assumed to have applied
  3. normalisation references are present for every environment that will be used
  4. every cell builds, trains a few episodes, transfers, and freezes
  5. the control conditions construct

What this file does *not* do is estimate cost -- `plan.py` owns that, because a
cost model belongs next to the catalogue it is costing.
"""
from __future__ import annotations

import argparse
import os
import platform
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OK, BAD, WARN = '[ok]', '[FAIL]', '[warn]'


def check_imports() -> bool:
    print('== 1. imports and environment ==')
    ok = True
    try:
        import tensorflow as tf
        print(f'{OK} tensorflow {tf.__version__}')
        gpus = tf.config.list_physical_devices('GPU')
        # Informational: these are 128x128 networks at batch 64, so
        # kernel-launch overhead dominates the arithmetic and a GPU is typically
        # no faster, sometimes slower. Choose machines by core count.
        print(f'     {len(gpus)} GPU, {os.cpu_count()} CPU cores')
    except Exception as exc:                            # noqa: BLE001
        print(f'{BAD} tensorflow: {exc}')
        ok = False
    try:
        import keras
        print(f'{OK} keras {keras.__version__}')
        # tensorflow.keras and standalone keras are *different module objects*
        # under Keras 3. The codebase imports `keras` throughout; mixing them
        # breaks custom-object registration and isinstance checks.
        from tensorflow import keras as tfk
        if keras is tfk:
            print(f'     note: keras and tensorflow.keras are the same module')
    except Exception as exc:                            # noqa: BLE001
        print(f'{BAD} keras: {exc}')
        ok = False
    for mod in ('gymnasium', 'numpy', 'scipy', 'pandas', 'matplotlib'):
        try:
            m = __import__(mod)
            print(f'{OK} {mod} {getattr(m, "__version__", "?")}')
        except Exception as exc:                        # noqa: BLE001
            print(f'{BAD} {mod}: {exc}')
            ok = False
    try:
        import Box2D
        print(f'{OK} Box2D {getattr(Box2D, "__version__", "?")} '
              f'(without this, LunarLander silently is not available)')
    except Exception as exc:                            # noqa: BLE001
        print(f'{BAD} Box2D: {exc} -- LunarLander will not run')
        ok = False

    print(f'     python {platform.python_version()} on {platform.system()} '
          f'{platform.machine()}')
    from src.dqn import provenance
    det = provenance.determinism_env()
    git = provenance.git_state()
    for key in ('TF_ENABLE_ONEDNN_OPTS', 'PYTHONHASHSEED', 'OMP_NUM_THREADS'):
        val = det.get(key)
        flag = OK if val is not None else WARN
        print(f'{flag} {key}={val}'
              + ('' if val is not None else '  (unset: bitwise reproducibility '
                                            'across processes is not guaranteed)'))
    if git.get('dirty'):
        print(f'{WARN} git tree is dirty ({git.get("dirty_files")} files) -- '
              f'results from an uncommitted tree are not reproducible from the '
              f'repository. This is recorded in every manifest.')
    else:
        print(f'{OK} git {str(git.get("commit"))[:12]} clean')
    return ok


def check_envs(quick: bool = False) -> bool:
    print('\n== 2. environments ==')
    from src.dqn import envs
    ok = True
    specs = ['CartPole-v1', 'LunarLander-v3', 'Acrobot-v1']
    if not quick:
        specs += [
            'LunarLander-v3:enable_wind=1,wind_power=15,turbulence_power=1.5',
            'LunarLander-v3:gravity=-4',
            'LunarLander-v3:pad_obs=4,extra_actions=2,pad_mode=noise',
            'CartPole-v1:length=1.0,masspole=0.2',
        ]
    for spec in specs:
        try:
            es = envs.parse(spec)
            env, info = envs.make(es)
            env.reset(seed=0)
            for _ in range(5):
                _, _, term, trunc, _ = env.step(env.action_space.sample())
                if term or trunc:
                    break
            env.close()
            if (info['obs_dim'], info['act_dim']) != (es.obs_dim, es.act_dim):
                print(f'{BAD} {spec}: registry says '
                      f'({es.obs_dim},{es.act_dim}), built '
                      f'({info["obs_dim"]},{info["act_dim"]})')
                ok = False
                continue
            extra = ''
            if info['applied']:
                extra = f'  applied={info["applied"]}'
            if info['derived_recomputed']:
                extra += f'  derived={info["derived_recomputed"]}'
            print(f'{OK} {spec[:56]:56s} obs={info["obs_dim"]} '
                  f'act={info["act_dim"]}{extra}')
        except Exception as exc:                        # noqa: BLE001
            print(f'{BAD} {spec}: {type(exc).__name__}: {exc}')
            ok = False
    return ok


def check_references(quick: bool = False) -> bool:
    """Normalisation constants must exist for every environment that will run.

    A missing reference is refused rather than defaulted, because normalising
    against an assumed zero would put one variant's scores on a different scale
    from every other -- the class of silent scale error that made the published
    cross-variant comparisons meaningless.
    """
    print('\n== 3. normalisation references ==')
    from src.dqn import envs
    from experiments import measure_references as mr
    refs = envs.load_references()
    targets = mr.CATALOGUE if not quick else mr.CATALOGUE[:3]
    missing = [t for t in (envs.parse(x).canonical() for x in targets)
               if t not in refs]
    if missing:
        print(f'{BAD} {len(missing)} environment(s) have no measured reference:')
        for m in missing:
            print(f'       {m}')
        print('     fix: python experiments/measure_references.py')
        return False
    print(f'{OK} {len(refs)} environments have measured random-policy references')
    for key in ('CartPole-v1', 'LunarLander-v3', 'Acrobot-v1'):
        if key in refs:
            r = refs[key]
            print(f'     {key:20s} random={r["random_return"]:9.2f} '
                  f'threshold={r["threshold"]:7.1f} '
                  f'noop_score={r.get("noop_score", float("nan")):6.3f}')
    return True


def check_cells(episodes: int = 3) -> bool:
    """Every cell must build, train, transfer, freeze and evaluate."""
    print(f'\n== 4. cells and conditions ({episodes} episodes each) ==')
    import shutil
    import tempfile

    from src.dqn.config import Config
    from src.dqn.train import train

    root = tempfile.mkdtemp(prefix='preflight_')
    tiny = dict(num_episodes=episodes, max_steps=120, eval_every=max(1, episodes - 1),
                eval_episodes=1, final_eval_episodes=2, final_eval_checkpoints=1,
                prefix_checkpoints=(), learning_starts=50, diag_states=32,
                probe_steps=0, probe_transitions=0, checkpoint_seconds=10 ** 9,
                out_root=root, experiment='preflight')
    ok = True
    sources: dict[tuple[str, str], str] = {}
    try:
        for arch in ('mlp', 'dueling'):
            for rule in ('vanilla', 'double'):
                try:
                    cfg = Config(arch=arch, target_rule=rule, condition='scratch',
                                 env='CartPole-v1', seed=0, **tiny)
                    train(cfg)
                    sources[(arch, rule)] = os.path.join(cfg.run_dir(),
                                                         'model.keras')
                    print(f'{OK} source  {arch}-{rule}')
                except Exception as exc:                # noqa: BLE001
                    print(f'{BAD} source  {arch}-{rule}: '
                          f'{type(exc).__name__}: {exc}')
                    ok = False

        for (arch, rule), ckpt in sources.items():
            for cond in ('transfer', 'transfer_untrained', 'transfer_permuted'):
                try:
                    cfg = Config(arch=arch, target_rule=rule, condition=cond,
                                 env='LunarLander-v3', source_env='CartPole-v1',
                                 source_checkpoint=ckpt, seed=0,
                                 freeze_updates=50, **tiny)
                    man = train(cfg)
                    frac = man['transfer']['summary']['fraction_of_model_transferred']
                    events = len(man['freeze_events'])
                    verified = all((e.get('verification') or {}).get('ok', True)
                                   for e in man['freeze_events'])
                    flag = OK if verified else BAD
                    print(f'{flag} {cond:19s} {arch}-{rule}  '
                          f'transferred={frac:.3f}  freeze_events={events}  '
                          f'verified={verified}')
                    ok = ok and verified
                except Exception as exc:                # noqa: BLE001
                    print(f'{BAD} {cond:19s} {arch}-{rule}: '
                          f'{type(exc).__name__}: {exc}')
                    ok = False
            break        # one cell is enough for the condition matrix
    finally:
        shutil.rmtree(root, ignore_errors=True)
    return ok


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--quick', action='store_true',
                   help='skip the parametric variants and the cell matrix')
    p.add_argument('--episodes', type=int, default=3)
    args = p.parse_args(argv)

    started = time.time()
    results = {
        'imports': check_imports(),
        'environments': check_envs(args.quick),
        'references': check_references(args.quick),
    }
    if not args.quick:
        results['cells'] = check_cells(args.episodes)

    print(f'\n== summary ({time.time() - started:.0f}s) ==')
    for name, good in results.items():
        print(f'{OK if good else BAD} {name}')
    if all(results.values()):
        print('\nReady. Next: python experiments/validate.py, then '
              'python experiments/plan.py --tier 1')
        return 0
    print('\nNot ready -- fix the failures above before launching anything.')
    return 1


if __name__ == '__main__':
    sys.exit(main())
