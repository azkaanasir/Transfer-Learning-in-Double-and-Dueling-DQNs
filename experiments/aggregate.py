"""Collect finished runs into the two tables every downstream module reads.

Outputs, and nothing else: `runs/per_seed.csv` (one row per run, all scalars)
and `runs/curves.csv` (long form, one row per run-episode). No test, no
p-value, no interval is computed here -- `stats.py` owns inference, and keeping
the two apart is what stops an aggregation window from being chosen after
seeing which window helps.

Why this file was rewritten rather than patched
-----------------------------------------------
The previous version read `metrics.csv`, which no longer exists (the log is
episode-keyed JSONL now, because the published loop appended on resume without
truncating and silently duplicated episodes -- `src/dqn/metrics.py`). It also
applied a hard-coded final-100-training-episode window as *the* headline
number. That window is superseded: `DESIGN.md` §5.2 makes the co-primary
endpoints `final_score` (held-out greedy evaluation, averaged over the final
k=3 checkpoints) and `auc_score` (area under the evaluation curve over env
steps). A terminal training-return snapshot is the endpoint *least* sensitive
to transfer and mixes exploration-contaminated returns into the estimate; it
survives here only as the descriptive `episode_length_final100` and
`td_loss_final100` columns, which `ANALYSIS_PLAN.md` §1 marks never-tested.

Each of the following exists because of a specific defect:

* **Independent metrics-integrity check.** The episode index set must equal
  `range(0, episodes_completed)`. `MetricsLog.check` asserts it at the end of a
  run and `audit.py` asserts it again; this is the third of the three
  independent checks promised in `src/dqn/metrics.py`. Three, because a
  duplicated episode is completely invisible in an aggregate -- it just shifts
  a window mean -- and it corrupts every window statistic downstream. A run
  that fails is still written out, with `metrics_contiguous=False` and a loud
  warning naming it, because dropping it silently would be the very failure
  mode under repair.

* **Right-censoring is explicit, never imputed.** `steps_to_threshold_pXX` is
  the env-step count at which the trailing-100-episode mean evaluation score
  first reaches 0.25 / 0.50 / 1.00. A run that never reaches the level carries
  its own total `env_steps` and `censored_pXX=True`. `ANALYSIS_PLAN.md` §5
  forbids imputing the budget (it biases the estimate and creates a tie mass
  that degenerates rank tests) and forbids dropping the run (that conditions on
  the outcome). The flag is a column so the survival analysis cannot mistake a
  censoring time for an event time.

* **Seed-set completeness is assertable.** `--require-complete` exits non-zero
  and lists every declared arm x seed of a selected experiment that has no run.
  The published study dropped one seed from one arm with no stated rule
  (`DESIGN.md` §1); the mechanical answer is that a partial arm is refused
  rather than averaged.

* **Seed blocks and experiment membership come from the registry, not from a
  path.** `seed_block` records which disjoint block a seed belongs to, so
  `audit.py` can refuse an estimate that touched `TUNE` (`DESIGN.md` §3.4).
  `experiments` is semicolon-joined, because a run genuinely belongs to several
  experiments: identical configurations share one run directory on purpose
  (`src/dqn/config.py`), and `registry.all_jobs` de-duplicates exactly the
  information this column must keep, so membership is accumulated per
  experiment instead.

* **Provenance travels with the table.** `per_seed.provenance.json` records the
  input run count, this repository's git commit and plan hashes, the hash of
  the CSV actually written, and the argv -- plus the *distinct* commits and
  plan hashes found across the input runs. More than one `ANALYSIS_PLAN.md`
  hash in one table means the runs were produced under different
  pre-registrations, which `ANALYSIS_PLAN.md` §1 makes a reporting-stopper.

Usage
-----
    python experiments/aggregate.py --out-root runs
    python experiments/aggregate.py --out-root runs --experiments E1,E2 \
        --require-complete
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import registry                                          # noqa: E402
from src.dqn import provenance                           # noqa: E402

# ---------------------------------------------------------------------------
# Pinned schemas. Other modules index these by name, so the order and the
# spelling are part of the interface and are not to be adjusted for taste.
# ---------------------------------------------------------------------------
PER_SEED_COLUMNS: tuple[str, ...] = (
    'run_dir', 'run_digest', 'experiments', 'label', 'arm', 'arch',
    'target_rule', 'condition', 'cell',
    'env', 'source_env', 'seed', 'seed_block', 'transfer_set', 'input_policy',
    'head_policy', 'freeze_group', 'freeze_updates', 'aggregation',
    'permute_kind', 'value_recal', 'lr', 'target_update', 'hidden',
    'head_units', 'num_episodes',
    'transferred_param_fraction', 'reinitialised_layer_count', 'params_copied',
    'episodes_completed', 'env_steps', 'updates', 'clip_fraction',
    'wall_time_s',
    'final_return', 'final_score', 'auc_score', 'jumpstart_score',
    'probe_jumpstart_score', 'within_run_sd', 'convergence_slope',
    'episode_length_final100', 'td_loss_final100',
    'steps_to_threshold_p25', 'censored_p25',
    'steps_to_threshold_p50', 'censored_p50',
    'steps_to_threshold_p100', 'censored_p100',
    'v_abs_mean', 'a_abs_mean', 'a_spread', 'grad_norm_trunk',
    'grad_norm_value', 'grad_norm_adv', 'grad_norm_head', 'grad_norm_global',
    'dead_unit_frac', 'cka_drift', 'q_mean', 'td_error_abs',
    'source_final_score', 'source_valid',
    'prefix_score_250', 'prefix_score_500', 'prefix_score_750',
    'metrics_contiguous', 'freeze_verified', 'git_commit', 'git_dirty',
    'plan_hash',
)

CURVE_COLUMNS: tuple[str, ...] = (
    'run_dir', 'cell', 'condition', 'label', 'seed', 'episode', 'env_steps',
    'updates', 'epsilon', 'score', 'eval_score', 'held_out_score', 'loss',
    'grad_norm', 'td_error_abs', 'q_mean', 'v_abs_mean', 'a_abs_mean',
    'a_spread', 'dead_unit_frac', 'cka_drift', 'frozen',
)

# `ANALYSIS_PLAN.md` §5: pre-declared, so a threshold metric exists even when no
# run reaches "solved", and so the levels are not chosen after seeing the curves.
THRESHOLD_LEVELS: tuple[tuple[str, float], ...] = (
    ('p25', 0.25), ('p50', 0.50), ('p100', 1.00))

# The trailing window the threshold is read off. In *episodes*, matching
# `DESIGN.md` §5.3, and applied to the evaluation score rather than to the
# exploration-contaminated training return.
TRAILING_WINDOW = 100

# Prefix checkpoints reported as columns. Fixed by the schema, not by whatever a
# particular run happened to save, so the column set never depends on the data
# (`Config.prefix_checkpoints` default, and `DESIGN.md` RQ6).
PREFIX_CHECKPOINTS: tuple[int, ...] = (250, 500, 750)

# Diagnostics are logged on the evaluation cadence (`DESIGN.md` §5.5), so their
# per-run summary is the mean over the final N *measured* points, not the final
# N episodes. Per-episode signals use an episode window. Both windows are named
# here because an undeclared window is how the published loss narrative came to
# contradict the published loss statistic.
DIAG_EVAL_TAIL = 10
DIAG_EPISODE_TAIL = 100
EVAL_CADENCE_COLUMNS = ('v_abs_mean', 'a_abs_mean', 'a_spread',
                        'grad_norm_trunk', 'grad_norm_value', 'grad_norm_adv',
                        'grad_norm_head', 'grad_norm_global', 'dead_unit_frac',
                        'cka_drift')
EPISODE_CADENCE_COLUMNS = ('q_mean', 'td_error_abs')

# Priority order for naming a seed's block. The blocks of `DESIGN.md` §3.4 are
# disjoint, with one deliberate exception: SMOKE is the single seed {0}, a subset
# of CONFIRM. Reporting a confirmatory seed as SMOKE would understate what it
# is, so SMOKE is not a reportable block here.
SEED_BLOCK_ORDER = ('CONFIRM', 'REPLICATE', 'TUNE', 'C4SRC', 'RESERVE')

WARN = '[WARNING]'


# ---------------------------------------------------------------------------
# Small readers
# ---------------------------------------------------------------------------
def _dig(obj: Any, *path: str, default: Any = None) -> Any:
    """Walk a nested manifest path, tolerating absent or null intermediates.

    Manifest fields are absent rather than null in several places (a scratch run
    has no `transfer` block at all), and a null is not an error -- `train.py`
    writes null where a quantity was not computable. Both must read as missing.
    """
    cur = obj
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
        if cur is None:
            return default
    return cur


def _num(value: Any) -> Optional[float]:
    """Coerce to float, mapping absent / null / non-finite to None."""
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    rows: list[dict] = []
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # A torn final line is the expected artifact of a kill
                # mid-write. It is dropped here and *counted* by the integrity
                # check below, which is where a lost episode must surface.
                continue
    return rows


def seed_block(seed: int) -> str:
    """Which disjoint block of `DESIGN.md` §3.4 a seed belongs to."""
    for name in SEED_BLOCK_ORDER:
        if seed in registry.SEED_BLOCKS.get(name, ()):
            return name
    return 'UNKNOWN'


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def find_runs(out_root: str) -> list[str]:
    """Run directories under `<out_root>/<condition>/<run_digest12>/s<NN>/`.

    The layout is fixed by `Config.run_dir`, which keys a run by its
    configuration digest rather than by experiment: adversarial review showed
    the older `<env>/<arch>-<rule>-<mode>-s<NN>` scheme collapsed nine distinct
    conditions from six experiments onto one directory. Globbing the digest
    level rather than reading a path for meaning is the corollary -- the path
    carries identity, and the manifest carries the interpretation.
    """
    pattern = os.path.join(out_root, '*', '*', 's*', 'manifest.json')
    return sorted(os.path.dirname(p) for p in glob.glob(pattern))


def _key(path: str) -> str:
    """A stable, platform-independent run key: the path as discovered."""
    try:
        rel = os.path.relpath(path, os.getcwd())
    except ValueError:                    # a different drive, on Windows
        rel = os.path.abspath(path)
    return rel.replace(os.sep, '/')


# ---------------------------------------------------------------------------
# Experiment membership
# ---------------------------------------------------------------------------
class Membership:
    """Which catalogue experiments include a given run.

    Resolved in two ways, in priority order, and which one fired is reported:

    1. **By run digest** -- exact. The digest covers every field that can change
       the trajectory or the measurement, so a match means the run *is* the
       configuration the registry declares.
    2. **By (arm label, seed)** -- the fallback. A validation run launched at a
       reduced budget (`STANDING_INSTRUCTIONS` S8: single seed, tiny episode
       count) has a different digest from the catalogue's confirmatory
       configuration while still being that arm of that experiment. Refusing to
       attribute it would leave the pipeline unanalysable at exactly the point
       it is being validated. The count resolved this way is printed and
       recorded, because a confirmatory table in which any row was attributed by
       label is a table whose runs were not the declared configuration.

    Membership is accumulated per experiment through `registry.jobs`, not
    through `registry.all_jobs`: the latter de-duplicates by run directory,
    which is precisely the information this column exists to keep.
    """

    def __init__(self, seeds: Sequence[int], out_root: str) -> None:
        self.by_digest: dict[str, set[str]] = {}
        self.by_label_seed: dict[tuple[str, int], set[str]] = {}
        self.errors: list[str] = []
        self.mode_counts: dict[str, int] = {'digest': 0, 'label': 0,
                                            'unattributed': 0}
        for eid, exp in registry.EXPERIMENTS.items():
            declared = registry.SEED_BLOCKS.get(exp.seed_block, ())
            wanted = sorted(set(int(s) for s in seeds) | set(declared))
            try:
                jobs = registry.jobs(eid, seeds=wanted, out_root=out_root)
            except Exception as exc:                     # noqa: BLE001
                # A broken catalogue entry must not take the aggregation down;
                # it must be visible. `audit.py` is what refuses to report.
                self.errors.append(f'{eid}: {type(exc).__name__}: {exc}')
                continue
            for job in jobs:
                self.by_digest.setdefault(job.cfg.run_digest(), set()).add(eid)
                self.by_label_seed.setdefault(
                    (job.arm, int(job.cfg.seed)), set()).add(eid)

    def resolve(self, run_digest: Optional[str], label: str,
                seed: int) -> tuple[tuple[str, ...], str]:
        if run_digest and run_digest in self.by_digest:
            self.mode_counts['digest'] += 1
            return tuple(sorted(self.by_digest[run_digest])), 'digest'
        hit = self.by_label_seed.get((label, int(seed)))
        if hit:
            self.mode_counts['label'] += 1
            return tuple(sorted(hit)), 'label'
        self.mode_counts['unattributed'] += 1
        return (), 'unattributed'


# ---------------------------------------------------------------------------
# Per-run extraction
# ---------------------------------------------------------------------------
def integrity_check(rows: list[dict], episodes_completed: Optional[int],
                    num_episodes: Optional[int]) -> tuple[bool, list[str]]:
    """Recompute the episode-index invariant from the log file itself.

    Independent of the run: it reads `metrics.jsonl` and compares against the
    manifest, rather than trusting the manifest's own verdict. That is the point
    of a third check -- a process that mis-counted its episodes writes a
    manifest that agrees with itself.

    A run that is merely *short* of its declared budget is recorded but not
    called an integrity failure: a run still in flight is legitimately short,
    and it is `--require-complete` that must refuse it, not this check.
    """
    episodes = [int(r['episode']) for r in rows if 'episode' in r]
    unique = sorted(set(episodes))
    problems: list[str] = []
    if not episodes:
        return False, ['metrics.jsonl has no episode-keyed rows']
    if len(episodes) != len(unique):
        counts: dict[int, int] = {}
        for episode in episodes:
            counts[episode] = counts.get(episode, 0) + 1
        dupes = sorted(e for e, c in counts.items() if c > 1)
        problems.append(f'duplicate episodes: {dupes[:12]}'
                        f'{"..." if len(dupes) > 12 else ""}')
    if unique != list(range(len(unique))):
        missing = sorted(set(range(unique[-1] + 1)) - set(unique))
        problems.append(f'episode index is not range(0, n): missing '
                        f'{missing[:12]}{"..." if len(missing) > 12 else ""}')
    if episodes_completed is not None and len(unique) != int(episodes_completed):
        problems.append(f'manifest says episodes_completed='
                        f'{int(episodes_completed)}, log holds {len(unique)}')
    if num_episodes is not None and len(unique) != int(num_episodes):
        problems.append(f'short run: {len(unique)} of {int(num_episodes)} '
                        f'episodes')
    fatal = [p for p in problems if not p.startswith('short run')]
    return not fatal, problems


def threshold_crossings(df: pd.DataFrame,
                        total_env_steps: Optional[float]) -> dict[str, Any]:
    """Env steps to each declared score level, with the censoring flag.

    The trailing mean is taken over `TRAILING_WINDOW` *episodes* of the
    evaluation score, so the window covers the same stretch of experience in
    every arm -- unlike a trailing window over evaluation *points*, whose
    spacing in env steps depends on episode length and therefore on performance
    (`DESIGN.md` §3.2).

    A level never reached yields the run's own total `env_steps` and
    `censored=True`. That is a censoring time, not an event time, and the flag
    is what stops the two being confused (`ANALYSIS_PLAN.md` §5).

    One stated caveat. The window is evaluated with `min_periods=1`, so before
    100 episodes have elapsed the "trailing mean" is the mean of however many
    evaluation points exist -- possibly one. At the confirmatory budget that is
    a short transient (`eval_every=10` gives ten points inside the first
    window); on a validation run of a dozen episodes a single noisy monitoring
    evaluation can trip the 0.25 level at episode 0, and that has been observed.
    The alternative -- requiring a full window -- makes the metric permanently
    undefined for any run shorter than 100 episodes, which would leave the
    censoring machinery untestable at exactly the point it is being validated.
    Padding the window would be worse still: it fabricates a crossing. So the
    early-window sensitivity is accepted and recorded here rather than hidden.
    """
    out: dict[str, Any] = {}
    censored_value = _num(total_env_steps)
    if df.empty or 'eval_score' not in df or df['eval_score'].notna().sum() == 0:
        for name, _level in THRESHOLD_LEVELS:
            out[f'steps_to_threshold_{name}'] = censored_value
            out[f'censored_{name}'] = True
        return out

    # A log with a duplicated episode is exactly the corruption
    # `metrics_contiguous` reports, and it must not crash the aggregation: the
    # offending run still has to reach the table so that the warning can name
    # it. The last row for an episode wins, matching `MetricsLog`'s
    # write-replaces-episode semantics, and the flag is what says the number
    # below is not to be trusted.
    unique = df.drop_duplicates(subset='episode', keep='last')
    span = range(0, int(unique['episode'].max()) + 1)
    scores = (unique.set_index('episode')['eval_score'].astype(float)
              .reindex(span))
    steps = (unique.set_index('episode')['env_steps'].astype(float)
             .reindex(span).ffill())
    # min_periods=1: a run shorter than the window still yields the mean of what
    # it has. Padding the window with zeros would fabricate a crossing.
    trailing = scores.rolling(TRAILING_WINDOW, min_periods=1).mean()

    for name, level in THRESHOLD_LEVELS:
        reached = trailing.index[trailing >= level]
        if len(reached):
            value = _num(steps.get(int(reached[0])))
            out[f'steps_to_threshold_{name}'] = (value if value is not None
                                                 else censored_value)
            out[f'censored_{name}'] = False
        else:
            out[f'steps_to_threshold_{name}'] = censored_value
            out[f'censored_{name}'] = True
    return out


def freeze_verdict(events: Iterable[dict]) -> tuple[bool, bool]:
    """(verified, had_unverifiable_event) over a run's freeze transitions.

    True when every event that carries a verification payload reports `ok`, and
    trivially true when no freeze occurred. An event with no payload is neither
    a pass nor a failure: the initial freeze at episode 0 has no earlier
    fingerprint to compare against, so nothing about it *can* be verified. Those
    are counted separately rather than folded into the verdict, because
    "verified" must not quietly come to mean "nothing was checked".
    """
    verdicts: list[bool] = []
    unverifiable = False
    for event in events or ():
        payload = event.get('verification') if isinstance(event, dict) else None
        if not isinstance(payload, dict):
            unverifiable = True
            continue
        ok = payload.get('ok')
        if ok is None:
            ok = not (payload.get('frozen_but_changed')
                      or payload.get('trainable_but_unchanged'))
        verdicts.append(bool(ok))
    return (all(verdicts) if verdicts else True), unverifiable


def _tail_mean(df: pd.DataFrame, column: str, tail: int) -> Optional[float]:
    """Mean of the final `tail` non-null values of a column."""
    if column not in df:
        return None
    series = df[column].dropna()
    if series.empty:
        return None
    return _num(series.tail(tail).mean())


def per_seed_row(run_dir: str, membership: Membership) -> Optional[dict]:
    """One `per_seed.csv` row, or a marker when the directory holds no result."""
    manifest_path = os.path.join(run_dir, 'manifest.json')
    try:
        with open(manifest_path, encoding='utf-8') as fh:
            manifest = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        print(f'{WARN} {run_dir}: unreadable manifest ({exc})')
        return None

    cfg = manifest.get('config') or {}
    identity = manifest.get('identity') or {}
    result = manifest.get('result')
    if not isinstance(result, dict):
        # No result block means the run never reached `_finalise`: it is in
        # flight, or it died. Reported by the caller, never averaged.
        return {'__incomplete__': _key(run_dir),
                '__reason__': 'no result block (run unfinished)'}

    rows = _read_jsonl(os.path.join(run_dir, 'metrics.jsonl'))
    df = (pd.DataFrame(rows).sort_values('episode').reset_index(drop=True)
          if any('episode' in r for r in rows) else pd.DataFrame())

    contiguous, problems = integrity_check(
        rows, result.get('episodes_completed'), cfg.get('num_episodes'))

    label = str(identity.get('label') or '')
    seed = int(cfg.get('seed', identity.get('seed', -1)))
    arch = str(cfg.get('arch', ''))
    rule = str(cfg.get('target_rule', ''))
    condition = str(cfg.get('condition', identity.get('condition', '')))
    experiments, mode = membership.resolve(identity.get('run_digest'),
                                           label, seed)
    hidden = cfg.get('hidden') or ()
    prefix_evals = _dig(result, 'prefix_evaluations', default={}) or {}
    summary = _dig(manifest, 'transfer', 'summary')
    verified, unverifiable = freeze_verdict(manifest.get('freeze_events') or ())

    row: dict[str, Any] = {
        'run_dir': _key(run_dir),
        'run_digest': identity.get('run_digest'),
        'experiments': ';'.join(experiments),
        'label': label,
        # The registry's arm label *is* the arm identity: it is the only key
        # that separates two arms sharing a (cell, condition) -- E1's
        # `transfer-*` and `transfer-trunk-*` differ in `transfer_set` alone.
        # `cell` and `condition` remain their own columns, so a table can be
        # grouped either way without re-parsing a compound name.
        'arm': label or f'{arch}-{rule}-{condition}',
        'arch': arch,
        'target_rule': rule,
        'condition': condition,
        'cell': f'{arch}-{rule}',
        'env': cfg.get('env'),
        'source_env': cfg.get('source_env'),
        'seed': seed,
        'seed_block': seed_block(seed),
        'transfer_set': cfg.get('transfer_set'),
        'input_policy': cfg.get('input_policy'),
        'head_policy': cfg.get('head_policy'),
        'freeze_group': cfg.get('freeze_group'),
        'freeze_updates': cfg.get('freeze_updates'),
        'aggregation': cfg.get('aggregation'),
        'permute_kind': cfg.get('permute_kind'),
        'value_recal': cfg.get('value_recal'),
        'lr': _num(cfg.get('lr')),
        'target_update': cfg.get('target_update'),
        'hidden': 'x'.join(str(int(h)) for h in hidden) if hidden else None,
        'head_units': cfg.get('head_units'),
        'num_episodes': cfg.get('num_episodes'),
        # DESIGN.md §3.1: the architecture factor was confounded with treatment
        # intensity by about a factor of two. These columns are what let
        # `audit.py` refuse a cross-arch contrast at mismatched intensity, so
        # they are carried on every row -- null on scratch by construction,
        # which is not the same thing as absent by omission.
        'transferred_param_fraction': _num(
            _dig(summary, 'fraction_of_model_transferred')),
        'reinitialised_layer_count': (
            len(_dig(summary, 'layers_reinit', default=[]) or [])
            if summary is not None else None),
        'params_copied': _dig(summary, 'params_copied'),
        'episodes_completed': result.get('episodes_completed'),
        'env_steps': result.get('env_steps'),
        'updates': result.get('updates'),
        'clip_fraction': _num(result.get('clip_fraction')),
        'wall_time_s': _num(result.get('wall_time_s')),
        'final_return': _num(result.get('final_return')),
        # P1 and P2, the co-primary endpoints (ANALYSIS_PLAN.md §1). Taken from
        # the manifest rather than recomputed: both are defined over held-out
        # evaluation episodes this module never sees.
        'final_score': _num(result.get('final_score')),
        'auc_score': _num(result.get('auc_score')),
        'jumpstart_score': _num(_dig(manifest, 'jumpstart', 'score')),
        'probe_jumpstart_score': _num(_dig(manifest, 'probe_jumpstart',
                                           'score')),
        'within_run_sd': _num(result.get('within_run_sd')),
        'convergence_slope': _num(result.get('convergence_slope_per_episode')),
        'episode_length_final100': _num(result.get('episode_length_final100')),
        'td_loss_final100': _num(result.get('td_loss_final100')),
        'source_final_score': _num(_dig(manifest, 'source', 'source_result',
                                        'final_score')),
        'source_valid': _dig(manifest, 'source', 'validity', 'valid'),
        'metrics_contiguous': bool(contiguous),
        'freeze_verified': bool(verified),
        'git_commit': _dig(manifest, 'provenance', 'git', 'commit'),
        'git_dirty': _dig(manifest, 'provenance', 'git', 'dirty'),
        'plan_hash': _dig(manifest, 'provenance', 'plans', 'ANALYSIS_PLAN.md'),
    }
    row.update(threshold_crossings(df, result.get('env_steps')))
    for column in EVAL_CADENCE_COLUMNS:
        row[column] = _tail_mean(df, column, DIAG_EVAL_TAIL)
    for column in EPISODE_CADENCE_COLUMNS:
        row[column] = _tail_mean(df, column, DIAG_EPISODE_TAIL)
    for checkpoint in PREFIX_CHECKPOINTS:
        entry = prefix_evals.get(str(checkpoint), prefix_evals.get(checkpoint))
        row[f'prefix_score_{checkpoint}'] = _num(
            entry.get('score') if isinstance(entry, dict) else None)

    row['__problems__'] = problems
    row['__resolution__'] = mode
    row['__unverifiable_freeze__'] = unverifiable
    row['__curves__'] = df
    return row


def curve_rows(row: dict) -> pd.DataFrame:
    """The long-form slice for one run, on the pinned `curves.csv` schema."""
    df: pd.DataFrame = row['__curves__']
    if df.empty:
        return pd.DataFrame(columns=list(CURVE_COLUMNS))
    out = pd.DataFrame(index=df.index)
    out['run_dir'] = row['run_dir']
    out['cell'] = row['cell']
    out['condition'] = row['condition']
    out['label'] = row['label']
    out['seed'] = row['seed']
    for column in CURVE_COLUMNS[5:]:
        out[column] = df[column] if column in df else np.nan
    return out[list(CURVE_COLUMNS)]


# ---------------------------------------------------------------------------
# Completeness
# ---------------------------------------------------------------------------
def missing_runs(selected: Sequence[str], frame: pd.DataFrame,
                 out_root: str) -> list[tuple[str, str, int]]:
    """Declared arm x seed combinations of the selected experiments with no run.

    The seed set an experiment is held to is the set of seeds at which *any* of
    its arms has a run -- not its declared block. Holding a validation launch to
    the full `CONFIRM` block would make the check fire always, and a check that
    always fires means nothing (`STANDING_INSTRUCTIONS` S8 runs one seed on
    purpose). Holding every arm to the seeds its siblings reached is exactly the
    defect under repair: one seed dropped from one arm.

    `only_as_source` arms are excluded when inferring that seed set, and only
    when inferring it -- they are still checked. E8i's positive-control donors
    are deliberately drawn from the disjoint `C4SRC` block, so their seeds are
    not the experiment's seed axis; letting them in would demand an
    `iface-transfer` run at seed 300, which the design specifically does not
    want to exist.
    """
    present_digests = set(frame['run_digest'].dropna())
    present_label_seed = {(str(label), int(seed))
                          for label, seed in zip(frame['label'], frame['seed'])}
    missing: list[tuple[str, str, int]] = []
    for eid in selected:
        exp = registry.EXPERIMENTS.get(eid)
        if exp is None:
            missing.append((eid, '<unknown experiment>', -1))
            continue
        off_axis = {arm.label for arm in exp.arms if arm.only_as_source}
        member = frame[frame['experiments'].fillna('').apply(
            lambda s, e=eid: e in str(s).split(';'))]
        seeds = sorted({int(seed) for label, seed
                        in zip(member['label'], member['seed'])
                        if str(label) not in off_axis})
        if not seeds:
            missing.append((eid, '<no runs on this experiment\'s seed axis>',
                            -1))
            continue
        try:
            jobs = registry.jobs(eid, seeds=seeds, out_root=out_root)
        except Exception as exc:                          # noqa: BLE001
            missing.append((eid, f'<catalogue error: {exc}>', -1))
            continue
        for job in jobs:
            if job.cfg.run_digest() in present_digests:
                continue
            if (job.arm, int(job.cfg.seed)) in present_label_seed:
                continue
            missing.append((eid, job.arm, int(job.cfg.seed)))
    return missing


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def count_false(series: pd.Series) -> int:
    """How many entries are literally False, treating null as not-False.

    A three-valued column (`source_valid`, `freeze_verified`) needs the three
    states kept apart: False is a verdict, null is the absence of one. Counting
    `not True` would report every scratch run as an invalid source.
    """
    return int((series.astype('object') == False).sum())          # noqa: E712


def arm_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-arm descriptives. Descriptive only -- no test, no interval."""
    def _sd(series: pd.Series) -> float:
        clean = series.dropna()
        return float(clean.std(ddof=1)) if len(clean) > 1 else float('nan')

    rows = []
    for (cell, condition, arm), group in frame.groupby(
            ['cell', 'condition', 'arm'], dropna=False):
        rows.append({
            'cell': cell, 'condition': condition, 'arm': arm,
            'n': int(len(group)),
            'final_score_mean': float(group['final_score'].mean()),
            'final_score_sd': _sd(group['final_score']),
            'final_score_median': float(group['final_score'].median()),
            'auc_score_mean': float(group['auc_score'].mean()),
            'auc_score_sd': _sd(group['auc_score']),
            'auc_score_median': float(group['auc_score'].median()),
            'transferred_frac': float(
                group['transferred_param_fraction'].mean()),
            'invalid_sources': count_false(group['source_valid']),
        })
    return (pd.DataFrame(rows)
            .sort_values(['cell', 'condition', 'arm'])
            .reset_index(drop=True))


def print_ledger() -> None:
    """The multiplicity ledger, printed on every invocation.

    Printed here even though this module computes nothing inferential, because
    `ANALYSIS_PLAN.md` §7 makes the count a recorded fact of every invocation
    rather than a claim made once in a paper. The honest entry for an
    aggregation is a zero.
    """
    print('\n== multiplicity ledger (ANALYSIS_PLAN.md 7) ==')
    print('  family   : confirmatory -- the only one')
    print('  members  : 8 = 4 cells x 2 co-primary endpoints '
          '(final_score, auc_score)')
    print('  procedure: Holm-Bonferroni, step-down from alpha = 0.05/8 '
          '= 0.00625')
    print('  screens  : Benjamini-Hochberg q, orientation only, no assertion '
          'permitted')
    print('  analyses in this output carrying a p-value: 0 -- aggregation '
          'emits no inference (see stats.py)')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out-root', default='runs',
                        help='run tree to walk (default: runs)')
    parser.add_argument('--output', default=None,
                        help='per-seed CSV (default: <out-root>/per_seed.csv)')
    parser.add_argument('--curves', default=None,
                        help='long-form CSV (default: <out-root>/curves.csv)')
    parser.add_argument('--experiments', default=None,
                        help='comma-separated experiment ids to keep, e.g. '
                             'E1,E2 (default: the whole catalogue)')
    parser.add_argument('--require-complete', action='store_true',
                        help='exit non-zero if any declared arm x seed of a '
                             'selected experiment has no run')
    parser.add_argument('--quiet', action='store_true',
                        help='suppress the summary tables; warnings still print')
    args = parser.parse_args(argv)

    out_root = args.out_root
    output = args.output or os.path.join(out_root, 'per_seed.csv')
    curves_path = args.curves or os.path.join(out_root, 'curves.csv')

    if args.experiments:
        selected = [e.strip() for e in args.experiments.split(',') if e.strip()]
        unknown = [e for e in selected if e not in registry.EXPERIMENTS]
        if unknown:
            print(f'{WARN} unknown experiment ids: {unknown}. Known: '
                  f'{sorted(registry.EXPERIMENTS)}')
            return 2
    else:
        selected = list(registry.EXPERIMENTS)

    run_dirs = find_runs(out_root)
    if not run_dirs:
        print(f'no runs found under {out_root}/ '
              f'(expected <condition>/<run_digest12>/s<NN>/manifest.json)')
        return 1

    seeds_seen: set[int] = set()
    for run_dir in run_dirs:
        base = os.path.basename(run_dir)
        if base.startswith('s') and base[1:].isdigit():
            seeds_seen.add(int(base[1:]))
    membership = Membership(sorted(seeds_seen), out_root)
    for err in membership.errors:
        print(f'{WARN} catalogue entry could not be resolved -- {err}')

    rows: list[dict] = []
    unfinished: list[tuple[str, str]] = []
    for run_dir in run_dirs:
        row = per_seed_row(run_dir, membership)
        if row is None:
            continue
        if '__incomplete__' in row:
            unfinished.append((row['__incomplete__'], row['__reason__']))
            continue
        rows.append(row)

    if not rows:
        print(f'{WARN} {len(run_dirs)} run directories found, none finished '
              f'(no manifest carries a result block).')
        return 1

    # --- the loud integrity warning -----------------------------------------
    offenders = [r for r in rows if not r['metrics_contiguous']]
    short = [r for r in rows
             if r['metrics_contiguous']
             and any(p.startswith('short run') for p in r['__problems__'])]
    if offenders:
        print('\n' + '!' * 72)
        print(f'{WARN} METRICS INTEGRITY FAILED for {len(offenders)} of '
              f'{len(rows)} runs. A duplicated or missing episode is invisible '
              f'in an aggregate -- it merely shifts a window mean -- and it '
              f'corrupts every window statistic downstream. These runs are '
              f'written out with metrics_contiguous=False and may not be '
              f'reported until audit.py has been satisfied:')
        for row in offenders:
            print(f'  {row["run_dir"]}')
            for problem in row['__problems__']:
                print(f'      - {problem}')
        print('!' * 72)
    if short and not args.quiet:
        print(f'\n{WARN} {len(short)} run(s) finished short of num_episodes '
              f'(contiguous, but not the declared budget):')
        for row in short[:20]:
            detail = next(p for p in row['__problems__']
                          if p.startswith('short run'))
            print(f'  {row["run_dir"]}: {detail}')
        if len(short) > 20:
            print(f'  ... and {len(short) - 20} more')
    if unfinished:
        print(f'\n{WARN} {len(unfinished)} run directory/ies hold no finished '
              f'run and are excluded from the table:')
        for path, reason in unfinished[:20]:
            print(f'  {path}: {reason}')
        if len(unfinished) > 20:
            print(f'  ... and {len(unfinished) - 20} more')

    unverifiable = sum(1 for r in rows if r['__unverifiable_freeze__'])

    # --- build the frames ---------------------------------------------------
    curve_frames = [curve_rows(r) for r in rows]
    for row in rows:
        for key in ('__curves__', '__problems__', '__resolution__',
                    '__unverifiable_freeze__'):
            row.pop(key, None)

    frame = pd.DataFrame(rows)
    for column in PER_SEED_COLUMNS:
        if column not in frame:
            frame[column] = None
    frame = (frame[list(PER_SEED_COLUMNS)]
             .sort_values(['cell', 'condition', 'arm', 'seed'])
             .reset_index(drop=True))
    non_empty = [c for c in curve_frames if not c.empty]
    curves = (pd.concat(non_empty, ignore_index=True) if non_empty
              else pd.DataFrame(columns=list(CURVE_COLUMNS)))
    if not curves.empty:
        curves = (curves.sort_values(['run_dir', 'episode'])
                  .reset_index(drop=True))

    # --- experiment selection ----------------------------------------------
    if args.experiments:
        keep = frame['experiments'].fillna('').apply(
            lambda s: bool(set(str(s).split(';')) & set(selected)))
        dropped = frame[~keep]
        if len(dropped):
            print(f'\n{WARN} {len(dropped)} run(s) excluded: not a member of '
                  f'{",".join(selected)}. An exclusion is a selection, not a '
                  f'cleaning step, so each one is named:')
            for _, row in dropped.head(20).iterrows():
                belongs = row['experiments'] or '<unattributed>'
                print(f'  {row["run_dir"]}  label={row["label"]} in={belongs}')
            if len(dropped) > 20:
                print(f'  ... and {len(dropped) - 20} more')
        frame = frame[keep].reset_index(drop=True)
        if not curves.empty:
            curves = (curves[curves['run_dir'].isin(set(frame['run_dir']))]
                      .reset_index(drop=True))
        if frame.empty:
            print(f'{WARN} nothing left after selecting {",".join(selected)}.')
            return 1

    # --- write --------------------------------------------------------------
    for path in (output, curves_path):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
    frame.to_csv(output, index=False)
    curves.to_csv(curves_path, index=False)

    unattributed = int((frame['experiments'].fillna('') == '').sum())
    input_commits = sorted(set(frame['git_commit'].dropna()))
    input_plans = sorted(set(frame['plan_hash'].dropna()))
    prov = {
        'tool': 'experiments/aggregate.py',
        'per_seed_csv': output.replace(os.sep, '/'),
        'per_seed_sha': provenance.file_hash(output),
        'per_seed_rows': int(len(frame)),
        'curves_csv': curves_path.replace(os.sep, '/'),
        'curves_sha': provenance.file_hash(curves_path),
        'curves_rows': int(len(curves)),
        'out_root': out_root.replace(os.sep, '/'),
        'run_directories_found': len(run_dirs),
        'runs_aggregated': int(len(frame)),
        'runs_unfinished': len(unfinished),
        'runs_failing_integrity': len(offenders),
        'runs_short_of_budget': len(short),
        'runs_with_unverifiable_freeze_event': unverifiable,
        'runs_unattributed_to_any_experiment': unattributed,
        'membership_resolution': dict(membership.mode_counts),
        'experiments_selected': selected if args.experiments else 'all',
        'schema': {'per_seed': list(PER_SEED_COLUMNS),
                   'curves': list(CURVE_COLUMNS)},
        'threshold_levels': {name: level for name, level in THRESHOLD_LEVELS},
        'trailing_window_episodes': TRAILING_WINDOW,
        'diagnostic_windows': {'eval_cadence_points': DIAG_EVAL_TAIL,
                               'episodes': DIAG_EPISODE_TAIL},
        'git': provenance.git_state(),
        'plans': provenance.plan_hashes(),
        'input_run_git_commits': input_commits,
        'input_run_plan_hashes': input_plans,
        'argv': list(sys.argv if argv is None else ['aggregate.py', *argv]),
        'cwd': os.getcwd(),
    }
    prov_path = os.path.splitext(output)[0] + '.provenance.json'
    with open(prov_path, 'w', encoding='utf-8') as fh:
        json.dump(prov, fh, indent=2, sort_keys=True)

    if len(input_plans) > 1:
        print(f'\n{WARN} {len(input_plans)} distinct ANALYSIS_PLAN.md hashes '
              f'across the input runs. A confirmatory result is interpretable '
              f'only against the one pre-registration in force when it ran '
              f'(ANALYSIS_PLAN.md 1): {input_plans}')
    if membership.mode_counts['label']:
        print(f'\n{WARN} {membership.mode_counts["label"]} run(s) were '
              f'attributed to an experiment by (arm label, seed) rather than by '
              f'configuration digest, so their configuration is not the one the '
              f'catalogue declares -- expected for a reduced-budget validation '
              f'launch, never acceptable for a confirmatory table.')
    if unattributed:
        print(f'{WARN} {unattributed} run(s) belong to no catalogue '
              f'experiment; the experiments column is empty for them.')

    # --- report -------------------------------------------------------------
    print(f'\n{len(frame)} runs -> {output}')
    print(f'{len(curves)} run-episodes -> {curves_path}')
    print(f'provenance -> {prov_path}')

    if not args.quiet:
        summary = arm_summary(frame)
        pd.set_option('display.width', 220)
        pd.set_option('display.max_columns', 40)
        pd.set_option('display.max_rows', 400)
        print('\n== per-arm descriptives (no test, no interval) ==')
        if len(summary) and int(summary['n'].min()) < 3:
            # ANALYSIS_PLAN.md 9. Printed above the numbers rather than below,
            # so it cannot be read past.
            print('PIPELINE VALIDATION - NOT A RESULT: an arm has n < 3, so no '
                  'number below may be quoted, compared, or used to choose '
                  'between hypotheses.')
        print(summary.round(4).to_string(index=False))

        censored = ', '.join(
            f'{name}={int(frame[f"censored_{name}"].sum())}/{len(frame)}'
            for name, _level in THRESHOLD_LEVELS)
        print(f'\ncensored at budget (level never reached): {censored}')
        moved = count_false(frame['freeze_verified'])
        print(f'metrics_contiguous: '
              f'{int(frame["metrics_contiguous"].sum())}/{len(frame)}   '
              f'freeze_verified: {len(frame) - moved}/{len(frame)} '
              f'({unverifiable} run(s) had a freeze event with no '
              f'verification payload -- an initial freeze has no earlier '
              f'fingerprint to compare against)')
        print(f'invalid sources: {count_false(frame["source_valid"])} of '
              f'{int(frame["source_valid"].notna().sum())} runs with a '
              f'source')
        blocks = frame['seed_block'].value_counts().to_dict()
        print(f'seed blocks: {blocks}')
        if blocks.get('TUNE'):
            print(f'{WARN} {blocks["TUNE"]} run(s) draw on the TUNE block. No '
                  f'reported estimate may touch them (DESIGN.md 3.4); audit.py '
                  f'enforces it.')
        print_ledger()

    # --- completeness -------------------------------------------------------
    if args.require_complete:
        gaps = missing_runs(selected, frame, out_root)
        if gaps:
            print(f'\n{WARN} INCOMPLETE: {len(gaps)} declared arm x seed '
                  f'combination(s) have no run. A partial arm is refused rather '
                  f'than averaged (DESIGN.md 1, 8.4):')
            for eid in sorted({g[0] for g in gaps}):
                items = [g for g in gaps if g[0] == eid]
                print(f'  {eid}: {len(items)} missing')
                for _, arm, seed in items[:25]:
                    where = f'seed {seed}' if seed >= 0 else ''
                    print(f'      {arm} {where}'.rstrip())
                if len(items) > 25:
                    print(f'      ... and {len(items) - 25} more')
            return 1
        print(f'\ncompleteness: every declared arm x seed of '
              f'{",".join(selected)} has a run.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
