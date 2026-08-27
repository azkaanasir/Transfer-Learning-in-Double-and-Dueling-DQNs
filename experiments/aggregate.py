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

  The window has to be **complete**: a mean of one evaluation is not a
  trailing-100 mean. Read off a partial window, the metric recorded 83 env
  steps as "steps to reach 25% of solved" for two 1000-episode transfer runs,
  against a table median of 38,435, because a single noisy 5-episode monitoring
  evaluation of an untrained policy cleared the level at episode 0. Where the
  evaluation series never fills one window the metric is **missing**: null
  time, null flag, and the reason on the row. Missing and "did not reach within
  the budget" are different statements, and asserting the second from an
  absent measurement is the §5 imputation one level up.

* **The co-primary AUC is recomputed, not copied.** `auc_score` (P2,
  `DESIGN.md` 5.2) is the area under the normalised-score evaluation curve
  over env steps, divided by *total* env steps. It was copied from the
  manifest, so this module held no independent check on one of the two
  endpoints the whole study turns on, and the copy carried a divisor the
  design does not specify: `src/dqn/train.py` divided by the span between the
  first and the last evaluation point, which credits a run only for the
  budget it spent under evaluation and so inflates most the runs whose
  unevaluated tail is longest. It is integrated from `metrics.jsonl` here
  instead. The pilot runs are corrected by re-derivation rather than by a
  re-run, the manifest's own value is compared rather than trusted, and
  `auc_env_step_coverage` puts the fraction of the budget the curve actually
  spans on the row, so the size of the omitted tail is auditable.

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
  pre-registrations, which `ANALYSIS_PLAN.md` §1 makes a reporting-stopper: it
  is a refusal here, not a printed remark, because a stopper that leaves the
  exit code at 0 stops nothing. The other half of the same question is whether
  the runs' hash is the plan *on disk*; that comparison was never made, so a
  table built entirely from runs predating an amendment passed silently while
  the provenance held both hashes in adjacent keys. It is compared and recorded
  on every invocation now, and refused under `--require-integrity` (not
  unconditionally: `audit.py` raises it as an error and `report.py` gates the
  bundle on that audit through an override that stamps every artifact, and a
  hard refusal here would leave §9 pipeline validation no way to run).

* **Membership is confined to the experiment's declared seed block.** An
  experiment is enumerated only at the seeds of the block `DESIGN.md` §3.4
  assigns it, intersected with the seeds actually present. Enumerating every
  catalogue experiment at every seed on disk broke the block discipline in both
  directions at once: E3, whose block is `TUNE`, acquired a table built from
  confirmatory runs, and E1 acquired the `C4SRC` donor runs at seed 300, whose
  configuration is byte-identical to E1's scratch arm because the run digest
  deliberately excludes the label. The second put seed 300 on E1's inferred
  seed axis and made `--require-complete` report twelve missing combinations on
  a tree that is complete. `RESERVE` is the one addition to the block, and only
  for the donor arms the replacement ledger says were drawn there
  (`DESIGN.md` §4.3).

* **`n` is a count of seeds, and every statistic carries the count it was
  computed on.** A row count presented as a seed count defeats the
  `ANALYSIS_PLAN.md` §9 n<3 stamp twice over: once when duplicate (arm, seed)
  rows inflate it, once when the endpoint is null on most of the rows counted.
  So `arm_summary` reports `n_runs`, `n_seeds`, `n_final_score` and
  `n_auc_score` separately and the stamp fires on the smallest of them.

* **(arm, seed) is unique, or the table says so.** Two runs of one arm at one
  seed are two configurations, not two seeds. Averaging them deflates the
  across-seed SD that `ANALYSIS_PLAN.md` §4 gates the equivalence claim on, by
  exactly sqrt(k(m-1)/(km-1)) for k duplicates of m seeds. Duplicates are
  named, counted into the provenance, and refused by `--require-complete`.

* **Nothing is dropped silently.** An unreadable manifest is counted and named
  rather than skipped past, so `run_directories_found` always equals aggregated
  + unfinished + unreadable. Unparsable `metrics.jsonl` lines are counted onto
  the row (`metrics_lines_unparsed`) instead of being inferred from an
  episode-count mismatch that a resume can hide. A run whose seed cannot be
  resolved at all is warned about by name, not merely tallied as `UNKNOWN`.

* **The provenance describes the table it stamps.** Integrity, freeze and
  membership counts are computed after the `--experiments` selection, with the
  pre-selection totals kept under their own keys, and the file is written after
  the refusal checks so that it records which were requested and which passed.
  A refused aggregation therefore leaves an artifact that says it was refused,
  rather than a clean-looking one and an exit code.

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
import statlib                                           # noqa: E402
import tuning                                            # noqa: E402
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
    # Plasticity signatures. The plasticity-loss literature supplies an
    # architecture-free rival explanation for degradation after pretraining --
    # dead units, parameter-norm growth, feature-rank collapse -- that the
    # weight-scale control (C3) does not exclude, because preserving a weight
    # multiset says nothing about the rank of the features those weights
    # produce. See paper/LITERATURE.md 3.4.
    'effective_rank', 'stable_rank', 'param_norm_total', 'param_norm_trunk',
    'source_final_score', 'source_valid',
    # DESIGN.md §4.3: "the number and identity of rejected source seeds appear
    # in the results table". Under the reserve rule a transfer run's source seed
    # is no longer a function of its own seed, and `source_checkpoint` is
    # excluded from the run digest, so without these three columns the table
    # cannot say which source seed was rejected or which RESERVE seed replaced
    # it. `source_seed_block` reads RESERVE exactly when a replacement is in
    # force, because §3.4 gives RESERVE no other use.
    'source_seed', 'source_seed_block', 'source_run_digest',
    'prefix_score_250', 'prefix_score_500', 'prefix_score_750',
    'metrics_contiguous', 'freeze_verified', 'git_commit', 'git_dirty',
    'plan_hash',
    # Appended rather than interleaved: other modules index this schema by name,
    # so a new column is additive, while re-ordering the established ones would
    # not be. Each of the five answers a specific way the old table misled.
    #
    # `jumpstart_interpretable` is derived here from `head_policy` rather than
    # copied from the manifest, because DESIGN.md §5.3 attaches the condition to
    # the metric ("interpretable *only* where the output head is transferred")
    # and the manifests written so far set the flag True on head_policy='reinit'
    # runs, where the zero-shot policy is an argmax over a random readout. The
    # disagreement is warned about rather than silently resolved.
    #
    # `transfer_layers_accounted` is False when the transfer summary's layer
    # lists cannot be true: a head-reinitialising run that names no reinitialised
    # layer. `reinitialised_layer_count` is then null rather than 0, because 0 is
    # a statement the manifest does not support, and DESIGN.md §3.1 makes that
    # column part of the intensity-confound audit.
    #
    # `ms_per_env_step` is a per-run fact, not a verdict computed against the
    # rest of the table, so its value does not change with the selection. It is
    # here because wall time and env steps were exported with nothing to
    # separate an environmental stall from workload cost, and plan.py's cost
    # model has no other way to exclude one.
    #
    # `metrics_problems` and `derivation_caveats` are separate columns because
    # they are separate questions. The first is what the run's own log says
    # about itself (duplicated episode, torn line, short of budget) and is what
    # `metrics_contiguous` summarises. The second is where the *manifest*
    # contradicts the design and a derived column had to be withheld or
    # overridden. Folding them together would have put a manifest bookkeeping
    # note into the column the integrity warning reads.
    'jumpstart_interpretable', 'transfer_layers_accounted', 'ms_per_env_step',
    'metrics_lines_unparsed', 'metrics_problems', 'derivation_caveats',
    #
    # `freeze_verifiable` is the third state `freeze_verified` cannot express:
    # True where a freeze event carried a verification payload and something was
    # actually checked, False where a freeze happened and nothing could be
    # checked, null where there was no freeze. Appended rather than folded into
    # `freeze_verified`, because that column is one of the machine-checked
    # invariant columns `stats.py` refuses an unparseable value in, and because
    # a reader asking "was it verified" and a reader asking "was there anything
    # to verify" are asking two questions. See `freeze_verdict`.
    'freeze_verifiable',
    #
    # `auc_env_step_coverage` is (last evaluation env step - first) / total env
    # steps: the fraction of the budget the integrated curve spans. P2 divides
    # by the total, so the unevaluated head and tail contribute no area, and
    # this is what makes the size of that omission a number on the row rather
    # than an assumption. It is also the quantity that separates the two
    # divisors, so a table whose auc_score was computed against the wrong one
    # can be recognised from the table.
    'auc_env_step_coverage',
)

#: Run-level columns of `curves.csv`: constant within a run, repeated on every
#: episode row. Named as their own tuple because `curve_rows` fills them from
#: the per-seed row and the rest from the metrics log, and a positional slice
#: through one flat schema silently mis-assigned them when the schema grew.
CURVE_RUN_COLUMNS: tuple[str, ...] = (
    'run_dir', 'cell', 'condition', 'label', 'seed',
    # The guard columns. `plots.py` selects curve rows by label alone, so
    # without these a TUNE-block run's curve and an integrity-failed run's curve
    # reach a figure with no marker available in the file the figure is built
    # from. ANALYSIS_PLAN.md §8 forbids any estimate computed on TUNE seeds, and
    # a figure is an estimate.
    'seed_block', 'experiments', 'metrics_contiguous',
)

CURVE_COLUMNS: tuple[str, ...] = CURVE_RUN_COLUMNS + (
    'episode', 'env_steps',
    'updates', 'epsilon', 'score', 'eval_score', 'held_out_score', 'loss',
    'grad_norm', 'td_error_abs', 'q_mean', 'v_abs_mean', 'a_abs_mean',
    'a_spread', 'dead_unit_frac', 'cka_drift',
    'effective_rank', 'stable_rank', 'feature_var_mean',
    'param_norm_total', 'param_norm_trunk', 'param_norm_value',
    'param_norm_adv', 'param_norm_head',
    'frozen',
)

# `ANALYSIS_PLAN.md` §5: pre-declared, so a threshold metric exists even when no
# run reaches "solved", and so the levels are not chosen after seeing the curves.
#
# Derived from `statlib`'s copy rather than restated. The plan declares one set
# of levels and three modules used to spell them out independently: `statlib.py`
# (the copy `validate.py` pins to the plan), `stats.py` (which labels and
# analyses the columns) and this one (which computes them). Nothing compared the
# three, so moving this copy alone rewrote `steps_to_threshold_p100` on 43 of 44
# runs while every downstream table went on printing the level as 1.00: the
# number changed and the label did not. Reading the pinned copy means this
# module cannot diverge from it, and the column names are derived from the
# values so the spelling stays a function of the levels rather than a fourth
# thing to keep in step. `stats.py` still holds a copy of its own, which is the
# remaining divergence and belongs in `validate.py`'s constant cross-check.
THRESHOLD_LEVELS: tuple[tuple[str, float], ...] = tuple(
    (f'p{int(round(float(level) * 100))}', float(level))
    for level in statlib.THRESHOLD_LEVELS)

#: The columns those levels name. Checked against the pinned schema at import,
#: because a level changed without the schema changing writes a differently
#: named column into a table whose readers index it by name, and a level changed
#: *with* the schema following silently is the divergence above.
_LEVEL_COLUMNS: tuple[str, ...] = tuple(
    column for name, _level in THRESHOLD_LEVELS
    for column in (f'steps_to_threshold_{name}', f'censored_{name}'))
_unpinned = [c for c in _LEVEL_COLUMNS if c not in PER_SEED_COLUMNS]
if _unpinned:
    raise RuntimeError(
        f'the censored-metric levels {tuple(statlib.THRESHOLD_LEVELS)} name '
        f'{_unpinned}, which the pinned per-seed schema does not declare. '
        f'ANALYSIS_PLAN.md 5 fixes those levels: changing them is a plan '
        f'amendment (11) and a schema change, not an edit to one constant.')

#: `ANALYSIS_PLAN.md` §9's floor and the stamp that fires below it, read from the
#: copy `validate.py` pins to the plan. Both were literals here (`< 3` and the
#: stamp text), pinned by nothing, in a module whose output carries the stamp.
MIN_N_FOR_INFERENCE = statlib.MIN_N_FOR_INFERENCE
VALIDATION_STAMP = statlib.PIPELINE_VALIDATION_LABEL

# The trailing window the threshold is read off. In *episodes*, matching
# `DESIGN.md` §5.3, and applied to the evaluation score rather than to the
# exploration-contaminated training return.
TRAILING_WINDOW = 100

# Prefix checkpoints reported as columns. Fixed by the schema, not by whatever a
# particular run happened to save, so the column set never depends on the data
# (`Config.prefix_checkpoints` default, and `DESIGN.md` RQ6).
# 500 is the only prefix a research question attaches to (the published budget,
# measured before epsilon floors at about episode 891) and is the only one runs
# now produce; 250 and 750 are retained in the schema so the column set stays
# fixed for older runs rather than varying with the data.
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
                        'cka_drift',
                        'effective_rank', 'stable_rank', 'param_norm_total',
                        'param_norm_trunk')
EPISODE_CADENCE_COLUMNS = ('q_mean', 'td_error_abs')

# Priority order for naming a seed's block. The blocks of `DESIGN.md` §3.4 are
# disjoint, with one deliberate exception: SMOKE is the single seed {0}, a subset
# of CONFIRM. Reporting a confirmatory seed as SMOKE would understate what it
# is, so SMOKE is not a reportable block here.
SEED_BLOCK_ORDER = ('CONFIRM', 'REPLICATE', 'TUNE', 'C4SRC', 'RESERVE')

# How far above the table median milliseconds-per-env-step a run has to sit
# before it is named as an environmental stall rather than a workload cost.
# Declared here rather than chosen when the numbers were seen: four P0 runs
# computed to 159-192 ms/step against a table median of 6.1, and a threshold
# picked afterwards is a threshold picked to include them.
TIMING_OUTLIER_FACTOR = 5.0

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


def _read_jsonl(path: str) -> tuple[list[dict], int]:
    """Parsed episode records, and how many lines could not be parsed.

    The count is returned rather than inferred. A torn final line is the
    expected artifact of a kill mid-write, and the previous version dropped it
    and left the loss to be deduced from `episodes_completed` against the log
    length. That deduction fails in both the cases that matter: a resume that
    rewrote the lost episode restores the count while the corruption remains,
    and a manifest with no `episodes_completed` gives nothing to compare
    against. So the number reaches the row as `metrics_lines_unparsed`.

    Three kinds of damage are counted here rather than raised, because a kill
    mid-write produces all three and each of them used to abort the whole
    aggregation over one run (the defect `resolve_seed` fixed for the seed field
    alone):

    * The file is read as bytes and each line decoded on its own, so a partial
      multibyte character -- the other thing a kill mid-write leaves behind --
      costs that line rather than raising `UnicodeDecodeError` out of the
      module. It is not decoded with `errors='replace'`: a replacement
      character inside a string value can still parse as JSON, and a silently
      mangled value is worse than a counted loss.
    * A line that parses to something other than an object is not an episode
      record. It was previously kept, and the `'episode' in r` guard downstream
      then tested list *membership* rather than key presence, so a line such as
      `["episode", 1]` reached `sort_values('episode')` and raised `KeyError`.
    * A line that is not JSON at all, which is the original torn-tail case.
    """
    if not os.path.exists(path):
        return [], 0
    rows: list[dict] = []
    unparsed = 0
    with open(path, 'rb') as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                line = raw.decode('utf-8')
            except UnicodeDecodeError:
                unparsed += 1
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                unparsed += 1
                continue
            if not isinstance(record, dict):
                unparsed += 1
                continue
            rows.append(record)
    return rows, unparsed


def _episode_index(value: Any) -> Optional[int]:
    """A logged episode number as an int, or None when it is not one.

    A metrics line whose `episode` is a string sorted the frame against a mix of
    `str` and `int` and raised `TypeError` out of the aggregation, taking every
    other run in the tree with it. The value is coerced here instead and the
    rows that cannot be coerced are counted onto the row that reports them.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        out = int(value)
    except (TypeError, ValueError):
        return None
    return out


#: What the `seed` column holds when no seed can be resolved from the manifest.
#: A sentinel rather than a null because the column is an integer key that
#: `missing_runs` and the duplicate check both group on; it is warned about by
#: name at every invocation, and `seed_block` reads UNKNOWN beside it.
NO_SEED = -1


def resolve_seed(cfg: Any, identity: Any) -> Optional[int]:
    """The run's seed, or None when neither block carries a usable one.

    `cfg.get('seed', identity.get('seed', -1))` reached for the identity
    fallback only when the *key* was absent, so a manifest carrying
    ``"seed": null`` returned None and `int(None)` aborted the whole
    aggregation, losing every other run in the tree. `_dig`'s own contract is
    that absent and null both read as missing, and this applies it: each source
    is tried in turn and a null is a miss, not a value.
    """
    for block in (cfg, identity):
        if not isinstance(block, dict):
            continue
        raw = block.get('seed')
        if raw is None or isinstance(raw, bool):
            continue
        try:
            return int(raw)
        except (TypeError, ValueError):
            continue
    return None


#: Manifests written on Windows record run paths with this separator, and the
#: table has to be readable on either platform, so it is spelled out once.
_BACKSLASH = '\\'


def _seed_from_run_path(path: Any) -> Optional[int]:
    """The seed encoded in a `.../s<NN>/...` run path, if one is.

    Used only for the source seed of runs written before `Config.source_seed`
    existed: their manifests record the source as a checkpoint path and nothing
    else, and DESIGN.md §4.3 requires the source seed to be recoverable from the
    results table. Reading a path for meaning is otherwise refused here (see
    `find_runs`), so this is confined to the one field that has no other
    evidence, and it returns None rather than guessing when the shape is wrong.
    """
    if not isinstance(path, str) or not path:
        return None
    parts = path.replace(_BACKSLASH, '/').split('/')
    for part in reversed(parts):
        if part.startswith('s') and part[1:].isdigit():
            return int(part[1:])
    return None


def seed_block(seed: Optional[int]) -> str:
    """Which disjoint block of `DESIGN.md` §3.4 a seed belongs to."""
    if seed is None:
        return 'UNKNOWN'
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

    Only *finished* runs are here, because `src/dqn/train.py` writes
    `manifest.json` once, inside `_finalise`. `find_unmanifested_runs` is the
    other half, and the two are counted separately for that reason.
    """
    pattern = os.path.join(out_root, '*', '*', 's*', 'manifest.json')
    return sorted(os.path.dirname(p) for p in glob.glob(pattern))


#: What a run directory holds even when it never reached `_finalise`. Any one of
#: them is enough: a run killed before its first checkpoint still has a metrics
#: log, and one killed before its first episode still has the state file.
RUN_OUTPUT_MARKERS: tuple[str, ...] = ('metrics.jsonl', 'state.json',
                                       'model.keras')


def find_unmanifested_runs(out_root: str) -> list[str]:
    """Run directories that hold run output but no `manifest.json`.

    `manifest.json` is written once, in `_finalise` (`src/dqn/train.py`), so a
    run that was killed or is still in flight matches no glob in `find_runs` and
    was invisible to this module entirely: `runs_unfinished` was structurally 0,
    the `__incomplete__` branch of `per_seed_row` was unreachable, and a
    standalone `aggregate.py --out-root runs` printed a per-arm table with the
    dead seed silently absent from it. That is the published defect `DESIGN.md`
    §1 names (a seed dropped from one arm with no stated rule), reachable from
    this module's own first usage line.

    A directory found here is counted as unfinished, named on stdout and in the
    provenance, and refused by `--require-integrity`. `audit.py` walks for the
    same thing under `manifest_absent`; the two are independent on purpose, and
    this one exists so that the table's own accounting is true.
    """
    found: set[str] = set()
    for marker in RUN_OUTPUT_MARKERS:
        pattern = os.path.join(out_root, '*', '*', 's*', marker)
        for path in glob.glob(pattern):
            run_dir = os.path.dirname(path)
            if not os.path.exists(os.path.join(run_dir, 'manifest.json')):
                found.add(run_dir)
    return sorted(found)


def _key(path: str) -> str:
    """A stable, platform-independent run key: the path as discovered."""
    try:
        rel = os.path.relpath(path, os.getcwd())
    except ValueError:                    # a different drive, on Windows
        rel = os.path.abspath(path)
    return rel.replace(os.sep, '/')


# ---------------------------------------------------------------------------
# The tuned stage of DESIGN.md 3.3
# ---------------------------------------------------------------------------
def activate_tuned_stage(out_root: str) -> tuple[object, str]:
    """Install E1t and E2t where the tree holds a selection. Never refuses.

    Implicit, unlike `sweep.py`, and the asymmetry is deliberate. This module
    reads a run tree and writes a table; it launches nothing, so activating the
    stage on finding the artifact commits no compute. What it does do is decide
    whether the tuned runs in the tree are *attributed*: without the tuned ids
    in the catalogue, a run belonging only to `E1t` resolves to no experiment at
    all and is exported as unattributed, and one shared with `E1` is exported as
    E1 alone. Both are silent misattributions of runs that are on disk, and a
    flag nobody passed is not a good reason to produce either.

    The refusal lives in `main`, and only for an explicitly named tuned id: that
    is the one case where continuing would answer a question nobody asked.
    """
    if not out_root:
        return None, 'no --out-root, so there is no tree to read a selection from'
    try:
        registry.activate_tuned_arms(out_root=out_root)
    except tuning.SelectionMissing as exc:
        return None, str(exc)
    except tuning.SelectionError as exc:
        return None, (f'the selection stored at '
                      f'{tuning.selection_path(out_root)} cannot be used: '
                      f'{exc}')
    except ValueError as exc:
        return None, f'the tuned arms cannot be built from it: {exc}'
    return registry.active_selection(), ''


def tuning_selection_record(selection, out_root: str,
                            unavailable: str) -> dict:
    """The tuned stage as the provenance sees it, present or absent.

    Written whether or not a selection was found. "This table was built with
    the tuned arms out of the catalogue" is a fact a consumer needs, because
    under `ANALYSIS_PLAN.md` 2.4 the arbitration verdict is `not-evaluable`
    while the tuned leg is absent, and a key that appeared only when the stage
    was available would make its absence unreadable from the artifact alone.
    """
    record = {
        'available': selection is not None,
        'active_in_catalogue': bool(registry.tuned_arms_active()),
        'experiments': sorted(registry.TUNED_OF),
        'policy': registry.TUNED_POLICY,
        'path': tuning.selection_path(out_root).replace(os.sep, '/'),
        'unavailable_because': (unavailable or None) if selection is None
                               else None,
        'selection_id': None, 'rule_id': None, 'rule_placeholder': None,
        'shared_cells': None, 'cells': None, 'selection_plans': None,
    }
    if selection is not None:
        record.update(
            selection_id=selection.selection_id,
            rule_id=selection.rule.get('id'),
            rule_placeholder=bool(selection.is_placeholder),
            seed_block=selection.seed_block,
            seeds=list(selection.seeds),
            env=selection.env,
            source_experiment=selection.source_experiment,
            shared_cells=list(selection.shared_cells),
            cells={key: cell.config.to_dict()
                   for key, cell in sorted(selection.cells.items())},
            selection_plans=dict(selection.plans))
    return record


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

    **An experiment is enumerated only at seeds of its declared block.** The
    blocks of `DESIGN.md` §3.4 are disjoint by construction and each experiment
    names the one it draws on, so a seed outside that block cannot be a seed of
    that experiment. Enumerating every experiment at every seed found on disk
    violated the discipline in both directions from a single line of code:

    * *Into* the confirmatory family. A run at a `TUNE` seed matched an
      experiment whose block is `CONFIRM`, so a TUNE seed was pooled into a
      confirmatory arm's descriptives and survived `--experiments E1`, which
      `ANALYSIS_PLAN.md` §8 forbids outright.
    * *Out of* the selection screen. E3's block is `TUNE`, and E3's
      ``hp-<cell>-lr0.0005-hard`` arm is configuration-identical to E1's
      ``scratch-<cell>`` arm, so enumerating E3 at seed 0 gave the
      hyperparameter screen a table made of confirmatory runs: the leakage
      revision 1 committed, in reverse (`DESIGN.md` §11 item 2).

    The same line caused the false completeness failure. `C4SRC` donor runs at
    seed 300 are configuration-identical to E1's scratch arm at seed 300 (the
    run digest excludes `label` and `experiment` by design), so enumerating E1
    at 300 attributed them to E1, put 300 on E1's inferred seed axis, and made
    `--require-complete` demand twelve runs the design says must not exist.

    `RESERVE` is the single addition to the declared block, because a
    replacement source under §4.3 is a real run of a real donor arm that has to
    stay attributable; `registry.jobs` builds only the donor arms at a RESERVE
    seed, so nothing target-side is declared there.
    """

    def __init__(self, seeds: Sequence[int], out_root: str,
                 selection=None) -> None:
        self.by_digest: dict[str, set[str]] = {}
        self.by_label_seed: dict[tuple[str, int], set[str]] = {}
        self.errors: list[str] = []
        self.mode_counts: dict[str, int] = {'digest': 0, 'label': 0,
                                            'unattributed': 0}
        present = {int(s) for s in seeds}
        reserve = set(registry.SEED_BLOCKS.get('RESERVE', ()))
        # `EXPERIMENTS` holds the tuned arms of DESIGN.md 3.3 where `main` found
        # a selection to build them from. A run shared between the two policies
        # is ONE run belonging to both, and that is how it leaves here: one row
        # whose `experiments` field names E1 and E1t together.
        for eid, exp in registry.EXPERIMENTS.items():
            declared = set(registry.SEED_BLOCKS.get(exp.seed_block, ()))
            wanted = sorted(present & (declared | reserve))
            if not wanted:
                continue
            try:
                jobs = registry.jobs(eid, seeds=wanted, out_root=out_root,
                                     selection=selection)
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
                seed: Optional[int]) -> tuple[tuple[str, ...], str]:
        if run_digest and run_digest in self.by_digest:
            self.mode_counts['digest'] += 1
            return tuple(sorted(self.by_digest[run_digest])), 'digest'
        # A run with no resolvable seed has no (label, seed) to fall back on:
        # attributing it to the seed the sentinel happens to spell would put a
        # run of unknown provenance inside a declared arm.
        hit = (self.by_label_seed.get((label, int(seed)))
               if seed is not None else None)
        if hit:
            self.mode_counts['label'] += 1
            return tuple(sorted(hit)), 'label'
        self.mode_counts['unattributed'] += 1
        return (), 'unattributed'


# ---------------------------------------------------------------------------
# Per-run extraction
# ---------------------------------------------------------------------------
def integrity_check(rows: list[dict], episodes_completed: Optional[int],
                    num_episodes: Optional[int],
                    lines_unparsed: int = 0) -> tuple[bool, list[str]]:
    """Recompute the episode-index invariant from the log file itself.

    Independent of the run: it reads `metrics.jsonl` and compares against the
    manifest, rather than trusting the manifest's own verdict. That is the point
    of a third check -- a process that mis-counted its episodes writes a
    manifest that agrees with itself.

    A run that is merely *short* of its declared budget is recorded but not
    called an integrity failure: a run still in flight is legitimately short,
    and it is `--require-complete` that must refuse it, not this check.

    An episode value that is not an integer is counted as its own problem
    rather than coerced with `int()`, which raised out of the whole aggregation
    on a single corrupt line.
    """
    indexed = [_episode_index(r.get('episode')) for r in rows
               if isinstance(r, dict) and 'episode' in r]
    episodes = [e for e in indexed if e is not None]
    unusable = len(indexed) - len(episodes)
    unique = sorted(set(episodes))
    problems: list[str] = []
    if lines_unparsed:
        # An unparsable line is a lost episode whether or not the counts happen
        # to agree afterwards: a resume that rewrote it restores the count and
        # leaves the corruption. Stated directly rather than deduced.
        problems.append(f'{lines_unparsed} unparsable line(s) in metrics.jsonl')
    if unusable:
        problems.append(f'{unusable} metrics line(s) carry an episode index '
                        f'that is not an integer')
    if not episodes:
        return False, problems + ['metrics.jsonl has no episode-keyed rows']
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


def eval_cadence(scores: pd.Series) -> Optional[int]:
    """The spacing, in episodes, between the evaluation points actually logged.

    The median gap rather than the mean, so one missing evaluation does not
    move it, and None when there are fewer than two points to measure between.
    Used only when the manifest does not state `eval_every`.
    """
    observed = np.asarray(scores.dropna().index, dtype=float)
    if observed.size < 2:
        return None
    gaps = np.diff(observed)
    gaps = gaps[gaps > 0]
    if not gaps.size:
        return None
    return int(round(float(np.median(gaps))))


def window_observations(cadence: Optional[int]) -> int:
    """How many evaluation points a *complete* trailing window holds.

    `TRAILING_WINDOW` is in episodes and the evaluation score is logged every
    `eval_every` episodes, so a complete window holds
    `TRAILING_WINDOW // eval_every` of them: ten at the confirmatory cadence of
    10. This is the observation count a trailing-window mean must be computed
    on, not a tolerance to be widened.

    Where the cadence is coarser than the window the answer is one, which is
    all a complete window can hold; the caller records the number used so that
    a configuration in which the "trailing-100 mean" is a single evaluation is
    visible in the provenance rather than assumed away.
    """
    if not cadence or cadence < 1:
        return 1
    return max(1, TRAILING_WINDOW // int(cadence))


def threshold_crossings(df: pd.DataFrame,
                        total_env_steps: Optional[float],
                        eval_every: Optional[int] = None) -> dict[str, Any]:
    """Env steps to each declared score level, with the censoring flag.

    The trailing mean is taken over `TRAILING_WINDOW` *episodes* of the
    evaluation score, so the window covers the same stretch of experience in
    every arm -- unlike a trailing window over evaluation *points*, whose
    spacing in env steps depends on episode length and therefore on performance
    (`DESIGN.md` §3.2).

    A level never reached yields the run's own total `env_steps` and
    `censored=True`. That is a censoring time, not an event time, and the flag
    is what stops the two being confused (`ANALYSIS_PLAN.md` §5).

    A level *reached* at an episode whose `env_steps` is unknown yields null,
    not the budget. The previous version substituted the run's total env steps
    and still set `censored=False`, which is the one substitution
    `ANALYSIS_PLAN.md` §5 names: the imputed number is systematically the
    maximum possible, so it biased `steps_to_threshold` upward and handed the
    survival model an event at the budget, creating exactly the tie mass at the
    boundary that degenerates a rank test. A null event time is missing data,
    which the survival model can see and refuse; an imputed one is not. The
    caller records it as a problem on the row.

    **The window must be complete.** `min_periods=1` made the
    "trailing-100-episode mean" at episode 0 the mean of a *single* 5-episode
    monitoring evaluation, and that is not the metric `DESIGN.md` §5.3 defines.
    It fired on the confirmatory tree, not only on a short validation run: two
    1000-episode transfer runs recorded `steps_to_threshold_p25 = 83` env steps,
    uncensored and uncaveated, against a table median of 38,435 and a
    next-smallest value of 8,379, because one noisy evaluation of an untrained
    policy cleared 0.25 at episode 0. Both were the two highest-jumpstart runs
    in the tree and both were transfer arms, so the artefact was systematically
    favourable to the claim the endpoint is evidence for. The window is now
    required to hold the number of evaluation points a complete window holds
    (`window_observations`), which is the definition rather than a threshold:
    `rolling(...).mean()` is null until then, and a null never satisfies `>=`,
    so a partial window can no longer date a crossing.

    Two alternatives were considered and rejected. *Padding* the window
    fabricates a crossing. *Flooring the crossing at `learning_starts`*, so that
    nothing before the first gradient update counts, would suppress a real
    zero-shot result -- a genuinely transferred policy that reaches the level
    with no update has reached it -- and it is not the metric the design
    declares; it also does not address the defect, which is that the estimator
    had one observation in it, not that the policy was untrained. A crossing
    dated before the first gradient update is therefore recorded and named on
    the row (`__pre_update_levels__`) rather than deleted.

    Where the run is too short for even one complete window the metric is
    **missing, not censored**: null time and a null flag, with the reason
    returned for the caller to put in `derivation_caveats`. Recording the budget
    with `censored=True` would assert "did not reach within the budget" from a
    measurement that was never taken, which is the same imputation
    `ANALYSIS_PLAN.md` §5 forbids one level up, and `stats.py` already counts an
    unreadable censoring flag as its own category, in neither the numerator nor
    the denominator of P(reached). The same is true of a run whose `eval_score`
    column is entirely null.
    """
    out: dict[str, Any] = {}
    censored_value = _num(total_env_steps)

    def unmeasurable(reason: str) -> dict[str, Any]:
        for name, _level in THRESHOLD_LEVELS:
            out[f'steps_to_threshold_{name}'] = None
            out[f'censored_{name}'] = None
        out['__threshold_undefined__'] = reason
        return out

    if df.empty or 'eval_score' not in df or df['eval_score'].notna().sum() == 0:
        return unmeasurable(
            f'no evaluation score is logged, so the trailing-'
            f'{TRAILING_WINDOW}-episode mean does not exist: '
            f'steps_to_threshold is missing data, never a censoring at the '
            f'budget')

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
    # The declared cadence when the manifest states one, because that is what
    # defines a complete window; the observed spacing only when it does not.
    declared = _num(eval_every)
    cadence = (int(declared) if declared is not None and declared >= 1
               else eval_cadence(scores))
    required = window_observations(cadence)
    trailing = scores.rolling(TRAILING_WINDOW, min_periods=required).mean()
    out['__eval_cadence__'] = cadence
    out['__window_observations__'] = required
    if not bool(trailing.notna().any()):
        return unmeasurable(
            f'{int(scores.notna().sum())} evaluation point(s) over '
            f'{len(span)} episode(s) never fill a trailing-{TRAILING_WINDOW}-'
            f'episode window, which holds {required} of them at an evaluation '
            f'cadence of {cadence}, so steps_to_threshold is missing data, '
            f'never a censoring at the budget')

    updates = (unique.set_index('episode')['updates'].reindex(span).ffill()
               if 'updates' in unique else None)
    pre_update: list[str] = []
    for name, level in THRESHOLD_LEVELS:
        reached = trailing.index[trailing >= level]
        if len(reached):
            episode = int(reached[0])
            out[f'steps_to_threshold_{name}'] = _num(steps.get(episode))
            out[f'censored_{name}'] = False
            done = _num(updates.get(episode)) if updates is not None else None
            if done is not None and done <= 0:
                pre_update.append(name)
        else:
            out[f'steps_to_threshold_{name}'] = censored_value
            out[f'censored_{name}'] = True
    if pre_update:
        out['__pre_update_levels__'] = pre_update
    return out


#: How close a recomputed `auc_score` must sit to the manifest's before the two
#: are called the same number. A float that has been through `json.dump` and
#: back is not bit-identical to the one that was computed, and that round trip
#: is the only difference this absorbs: it is a tolerance on the serialisation,
#: not on the metric, and widening it would hide the divisor disagreement it
#: exists to expose.
AUC_AGREEMENT_RELATIVE = 1e-9


def auc_per_env_step(df: pd.DataFrame,
                     total_env_steps: Optional[float]) -> dict[str, Any]:
    """P2: area under the evaluation curve over env steps, per env step.

    Recomputed from `metrics.jsonl`, not copied from the manifest. `auc_score`
    is one of the two co-primary endpoints (`ANALYSIS_PLAN.md` 1), and copying
    it left this module with no independent check on it at all. The copy also
    carried a divisor the design does not specify: `src/dqn/train.py` divided
    the area by `x[-1] - x[0]`, the span between the first and the last
    evaluation point, while `DESIGN.md` 5.2 defines P2 as the area divided by
    total env steps. Every manifest on the pilot tree holds the span-divisor
    value; recomputed as specified, the 44 runs fall by 0.39% to 3.54%
    (median 0.61%). The bias is not uniform, because the span omits the env
    steps after the last evaluation, which is a larger share of the budget for
    a run with a long unevaluated tail: the inflation is largest on the runs
    that were evaluated least.

    Integrating it here means the pilot runs are corrected by re-derivation
    rather than by a re-run, and that a divisor changed upstream reaches the
    table as a disagreement on the row instead of passing through unread. The
    manifest's own value is compared, never used: the number in the column is
    the one this function computed, and the disagreement travels in
    `derivation_caveats` and in the provenance.

    The trapezoid is taken on the de-duplicated episode index, matching
    `threshold_crossings` and `curve_rows`. A duplicated episode is the
    corruption `metrics_contiguous` reports, and counting its area twice would
    be a second effect of it that no column names.

    Undefined rather than zero wherever the area does not exist: fewer than two
    usable evaluation points, a zero-width span, a non-monotone env-step axis,
    or no positive total to divide by. A null is missing data; a zero is a
    measurement, and this module does not have one.
    """
    total = _num(total_env_steps)
    if df.empty or 'eval_score' not in df or 'env_steps' not in df:
        return {'auc_score': None, '__auc_coverage__': None,
                '__auc_undefined__': 'the metrics log carries no evaluation '
                                     'score against an env-step axis, so '
                                     'there is no curve to integrate'}
    unique = df.drop_duplicates(subset='episode', keep='last')
    points = pd.DataFrame({
        'x': pd.to_numeric(unique['env_steps'], errors='coerce'),
        'y': pd.to_numeric(unique['eval_score'], errors='coerce'),
    }).dropna()
    x = points['x'].to_numpy(dtype=float)
    y = points['y'].to_numpy(dtype=float)
    reason: Optional[str] = None
    if len(x) < 2:
        reason = (f'{len(x)} evaluation point(s) carry both a score and an '
                  f'env-step count, and an area needs two')
    elif bool(np.any(np.diff(x) < 0)):
        reason = ('the env-step axis is not monotone across the evaluation '
                  'points, so a trapezoid over it would net area out')
    elif x[-1] == x[0]:
        reason = 'every evaluation sits at the same env step, so the span is 0'
    elif total is None or total <= 0:
        reason = ('the manifest records no positive total env_steps, which is '
                  'the divisor DESIGN.md 5.2 specifies')
    if reason is not None:
        return {'auc_score': None, '__auc_coverage__': None,
                '__auc_undefined__': reason}
    return {'auc_score': float(np.trapezoid(y, x)) / float(total),
            '__auc_coverage__': float(x[-1] - x[0]) / float(total),
            '__auc_undefined__': None}


def freeze_verdict(events: Iterable[dict]) -> tuple[bool, bool, Optional[bool]]:
    """(verified, had_unverifiable_event, verifiable) over a run's freezes.

    `verified` is True when every event that carries a verification payload
    reports `ok`, and trivially true when no freeze occurred. An event with no
    payload is neither a pass nor a failure: the initial freeze at episode 0 has
    no earlier fingerprint to compare against, so nothing about it *can* be
    verified.

    `verifiable` is the third state, and it is a column rather than only a
    counter because the counter carries no signal: every manifest in the tree
    records the episode-0 freeze with `verification: null`, so
    `runs_with_unverifiable_freeze_event` reads 44 of 44 by construction and can
    take no other value. Meanwhile `freeze_verified` read True on all 44 rows
    although only 24 carried any verification payload at all, so a table reading
    that column alone reported 44/44 verified for a tree in which 20 runs had
    nothing checked. True here means something was checked, False that a freeze
    happened and nothing could be checked, and null that there was no freeze to
    check. "Verified" must not quietly come to mean "nothing was checked".
    """
    verdicts: list[bool] = []
    unverifiable = False
    seen = False
    for event in events or ():
        seen = True
        payload = event.get('verification') if isinstance(event, dict) else None
        if not isinstance(payload, dict):
            unverifiable = True
            continue
        ok = payload.get('ok')
        if ok is None:
            ok = not (payload.get('frozen_but_changed')
                      or payload.get('trainable_but_unchanged'))
        verdicts.append(bool(ok))
    verifiable: Optional[bool] = bool(verdicts) if seen else None
    return (all(verdicts) if verdicts else True), unverifiable, verifiable


def metrics_frame(rows: list[dict]) -> pd.DataFrame:
    """The episode-keyed frame for one run, or an empty frame.

    The episode index is coerced and the rows that will not coerce are dropped
    from the *frame* while staying in `integrity_check`'s count, so that one
    unusable line costs its own episode rather than raising `TypeError` out of
    `sort_values` and losing every run in the tree. The key presence test is on
    the frame's columns rather than `'episode' in r`, which on a list row tests
    membership of the values.
    """
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    if 'episode' not in frame.columns:
        return pd.DataFrame()
    episode = frame['episode'].map(_episode_index)
    frame = frame[episode.notna()].copy()
    if frame.empty:
        return pd.DataFrame()
    frame['episode'] = episode[episode.notna()].astype('int64')
    return frame.sort_values('episode').reset_index(drop=True)


def _tail_mean(df: pd.DataFrame, column: str, tail: int) -> Optional[float]:
    """Mean of the final `tail` non-null values of a column."""
    if column not in df:
        return None
    series = df[column].dropna()
    if series.empty:
        return None
    return _num(series.tail(tail).mean())


def per_seed_row(run_dir: str, membership: Membership) -> dict:
    """One `per_seed.csv` row, or a marker naming why there is no row.

    Never returns None. A directory that yields no row is a run that exists on
    disk and is not in the table, and the module docstring's own rule is that
    dropping such a run silently is the failure mode under repair. The two
    markers, `__incomplete__` and `__unreadable__`, are counted separately by
    the caller so that run directories found always equals rows aggregated plus
    unfinished plus unreadable, with nothing unaccounted for.
    """
    manifest_path = os.path.join(run_dir, 'manifest.json')
    try:
        with open(manifest_path, encoding='utf-8') as fh:
            manifest = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        return {'__unreadable__': _key(run_dir),
                '__reason__': f'unreadable manifest ({exc})'}

    cfg = manifest.get('config') or {}
    identity = manifest.get('identity') or {}
    result = manifest.get('result')
    if not isinstance(result, dict):
        # No result block means the run never reached `_finalise`: it is in
        # flight, or it died. Reported by the caller, never averaged.
        return {'__incomplete__': _key(run_dir),
                '__reason__': 'no result block (run unfinished)'}

    rows, lines_unparsed = _read_jsonl(os.path.join(run_dir, 'metrics.jsonl'))
    df = metrics_frame(rows)

    contiguous, problems = integrity_check(
        rows, result.get('episodes_completed'), cfg.get('num_episodes'),
        lines_unparsed)

    label = str(identity.get('label') or '')
    resolved_seed = resolve_seed(cfg, identity)
    seed = NO_SEED if resolved_seed is None else resolved_seed
    arch = str(cfg.get('arch') or '')
    rule = str(cfg.get('target_rule') or '')
    condition = str(cfg.get('condition') or identity.get('condition') or '')
    digest = identity.get('run_digest')
    experiments, mode = membership.resolve(digest, label, resolved_seed)
    hidden = cfg.get('hidden') or ()
    # An arm name assembled from empty parts is the arm name "--scratch", and
    # every config-less run in a tree collapses onto it: two runs from different
    # cells would then be averaged together under one label. The composed
    # fallback is used only when it actually names something, and the last
    # resort is the run's own identity, which cannot collide.
    composed = (f'{arch}-{rule}-{condition}'
                if (arch and rule and condition) else '')
    # The run digest when there is one, otherwise the two path segments that
    # *are* the run's identity on disk. Truncating the full key instead gave
    # every run under one out-root the same name, which is the collision this
    # fallback exists to avoid.
    tail = '/'.join(_key(run_dir).replace(os.sep, '/').split('/')[-2:])
    unidentified = f'unidentified-{digest[:12] if digest else tail}'
    arm = label or composed or unidentified
    cell = f'{arch}-{rule}' if (arch and rule) else None
    prefix_evals = _dig(result, 'prefix_evaluations', default={}) or {}
    summary = _dig(manifest, 'transfer', 'summary')
    verified, unverifiable, verifiable = freeze_verdict(
        manifest.get('freeze_events') or ())
    head_policy = cfg.get('head_policy')
    is_transfer = summary is not None or bool(cfg.get('source_checkpoint'))

    # DESIGN.md 3.1 makes `reinitialised_layer_count` part of the
    # intensity-confound audit and gives the expected values for trunk-only
    # transfer (mlp 1, dueling 4). Every P0 transfer manifest writes
    # `layers_reinit: []` while the output head appears in none of the four
    # layer lists, so len() of it is 0 for arms that must reinitialise at least
    # the head. 0 is not a weaker statement than the truth, it is a false one,
    # so the count is withheld and the contradiction is carried as its own
    # column rather than smoothed over. The manifest writer is what has to
    # change; until it does, the audit can at least see the column is unusable.
    caveats: list[str] = []
    reinit_list = _dig(summary, 'layers_reinit', default=None)
    reinit_count = len(reinit_list) if isinstance(reinit_list, list) else None
    layers_accounted: Optional[bool] = None
    if summary is not None:
        layers_accounted = not (head_policy == 'reinit' and not reinit_count)
        if not layers_accounted:
            reinit_count = None
            caveats.append(
                'transfer summary lists no reinitialised layer although '
                'head_policy=reinit, so reinitialised_layer_count is withheld')

    # DESIGN.md 5.3 attaches the condition to the metric: jumpstart is
    # interpretable *only* where the output head is transferred, because with a
    # reinitialised head the zero-shot policy is an argmax over a random
    # readout. Derived from head_policy rather than copied from the manifest,
    # which sets the flag True on head_policy=reinit runs (its own note in the
    # same block says the number is at chance by construction).
    jumpstart_interpretable: Optional[bool] = None
    if _dig(manifest, 'jumpstart', 'score') is not None:
        jumpstart_interpretable = bool(is_transfer and head_policy
                                       and head_policy != 'reinit')
        claimed = _dig(manifest, 'jumpstart', 'interpretable')
        if isinstance(claimed, bool) and claimed != jumpstart_interpretable:
            caveats.append(
                f'manifest claims jumpstart.interpretable={claimed} on '
                f'head_policy={head_policy!r}; DESIGN.md 5.3 makes it '
                f'{jumpstart_interpretable}')

    # DESIGN.md 4.3's reserve rule detaches the source seed from the run's own.
    # New runs record it in the config; runs written before the field existed
    # carry only the checkpoint path, and the seed has to be recoverable from
    # the table either way.
    source_seed = cfg.get('source_seed')
    if source_seed is None:
        source_seed = _seed_from_run_path(
            cfg.get('source_checkpoint')
            or _dig(manifest, 'source', 'checkpoint'))
    source_seed = None if source_seed is None else int(source_seed)

    # A per-run fact, so its value does not change with the selection. Nothing
    # else in the table separates an environmental stall from workload cost.
    total_steps = _num(result.get('env_steps'))
    wall = _num(result.get('wall_time_s'))
    ms_per_step = (wall * 1000.0 / total_steps
                   if wall is not None and total_steps else None)

    # P2, recomputed rather than copied (`auc_per_env_step`). The manifest's
    # own value is compared here and never substituted: a disagreement is a
    # statement about the manifest writer, and resolving it in favour of the
    # manifest is what let the span divisor reach the table unread. The caveat
    # texts are constant so that forty-four identical disagreements group into
    # one line rather than forty-four.
    auc = auc_per_env_step(df, result.get('env_steps'))
    auc_undefined = auc['__auc_undefined__']
    auc_coverage = auc['__auc_coverage__']
    auc_score = auc['auc_score']
    manifest_auc = _num(result.get('auc_score'))
    # Two facts, not one. Whether the manifest disagrees is a count; by how
    # much is a ratio, and a manifest value of exactly 0 has no ratio to it.
    # Folding the two together would either divide by zero or drop that run
    # out of the count, and dropping it is the failure mode being repaired.
    auc_disagrees = False
    auc_relative_change: Optional[float] = None
    if auc_undefined is not None:
        caveats.append(f'auc_score is withheld: {auc_undefined}')
        if manifest_auc is not None:
            auc_disagrees = True
            caveats.append(
                'the manifest records an auc_score that cannot be reproduced '
                'from the evaluation curve, so the column is null rather than '
                'the manifest value (DESIGN.md 5.2 defines P2 on that curve)')
    elif manifest_auc is None:
        auc_disagrees = True
        caveats.append(
            'the manifest records no auc_score; the column is integrated from '
            'the evaluation curve and divided by total env steps '
            '(DESIGN.md 5.2)')
    elif manifest_auc == 0.0:
        if auc_score != 0.0:
            auc_disagrees = True
            caveats.append(
                'the manifest records auc_score = 0, which the evaluation '
                'curve does not support; the column is integrated here '
                '(DESIGN.md 5.2) and no relative change is quotable against '
                'a zero')
    elif (abs(auc_score - manifest_auc)
            > AUC_AGREEMENT_RELATIVE * abs(manifest_auc)):
        auc_disagrees = True
        auc_relative_change = (auc_score - manifest_auc) / manifest_auc
        caveats.append(
            'auc_score is integrated from the evaluation curve and divided by '
            'total env steps (DESIGN.md 5.2); the manifest disagrees, having '
            'divided by the span between the first and the last evaluation '
            'point, so the run is corrected here rather than by a re-run')

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
        'arm': arm,
        'arch': arch or None,
        'target_rule': rule or None,
        'condition': condition or None,
        'cell': cell,
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
        'reinitialised_layer_count': reinit_count,
        'transfer_layers_accounted': layers_accounted,
        'params_copied': _dig(summary, 'params_copied'),
        'episodes_completed': result.get('episodes_completed'),
        'env_steps': result.get('env_steps'),
        'updates': result.get('updates'),
        'clip_fraction': _num(result.get('clip_fraction')),
        'wall_time_s': _num(result.get('wall_time_s')),
        'final_return': _num(result.get('final_return')),
        # P1 and P2, the co-primary endpoints (ANALYSIS_PLAN.md §1). P1 is
        # taken from the manifest, because it is a mean over the final k=3
        # checkpoints of a held-out greedy evaluation whose episodes never
        # reach the metrics log. P2 is not: its inputs are the logged
        # evaluation curve and the run's total env steps, both of which are
        # here, so copying it meant carrying an unchecked divisor on a
        # co-primary endpoint (`auc_per_env_step`).
        'final_score': _num(result.get('final_score')),
        'auc_score': auc_score,
        'auc_env_step_coverage': auc_coverage,
        'jumpstart_score': _num(_dig(manifest, 'jumpstart', 'score')),
        'jumpstart_interpretable': jumpstart_interpretable,
        'probe_jumpstart_score': _num(_dig(manifest, 'probe_jumpstart',
                                           'score')),
        'within_run_sd': _num(result.get('within_run_sd')),
        'convergence_slope': _num(result.get('convergence_slope_per_episode')),
        'episode_length_final100': _num(result.get('episode_length_final100')),
        'td_loss_final100': _num(result.get('td_loss_final100')),
        'source_final_score': _num(_dig(manifest, 'source', 'source_result',
                                        'final_score')),
        'source_valid': _dig(manifest, 'source', 'validity', 'valid'),
        'source_seed': source_seed,
        'source_seed_block': (seed_block(source_seed)
                              if source_seed is not None else None),
        'source_run_digest': _dig(manifest, 'source', 'source_result',
                                  'run_digest'),
        'ms_per_env_step': ms_per_step,
        'metrics_lines_unparsed': int(lines_unparsed),
        'metrics_contiguous': bool(contiguous),
        'freeze_verified': bool(verified),
        'freeze_verifiable': verifiable,
        'git_commit': _dig(manifest, 'provenance', 'git', 'commit'),
        'git_dirty': _dig(manifest, 'provenance', 'git', 'dirty'),
        'plan_hash': _dig(manifest, 'provenance', 'plans', 'ANALYSIS_PLAN.md'),
    }
    crossings = threshold_crossings(df, result.get('env_steps'),
                                    eval_every=cfg.get('eval_every'))
    undefined = crossings.pop('__threshold_undefined__', None)
    window_points = crossings.pop('__window_observations__', None)
    cadence = crossings.pop('__eval_cadence__', None)
    pre_update = crossings.pop('__pre_update_levels__', ())
    if undefined:
        caveats.append(f'steps_to_threshold is withheld: {undefined}')
    if pre_update:
        # Recorded, not deleted. The crossing is measured on a complete window;
        # that the window closed before the first gradient update is a fact
        # about the run, and DESIGN.md 5.3 does not make it a disqualification.
        caveats.append(
            f'the {", ".join(pre_update)} crossing(s) are dated at an episode '
            f'where the run had made no gradient update, so the level was '
            f'reached by the initial policy rather than by learning')
    for name, _level in THRESHOLD_LEVELS:
        if (crossings[f'censored_{name}'] is False
                and crossings[f'steps_to_threshold_{name}'] is None):
            caveats.append(
                f'{name} threshold was reached at an episode with no recorded '
                f'env_steps; the event time is missing, never the budget')
    if any(crossings[f'censored_{name}'] is True
           for name, _level in THRESHOLD_LEVELS):
        # ANALYSIS_PLAN.md 5 calls this censoring administrative and
        # "independent of the event time by construction". That holds for the
        # budget, which is 1000 episodes for every run, and not for the axis the
        # metric is measured on: env_steps ranges 127,575 to 519,168 across the
        # tree, and it grows with episode length, hence with performance. The
        # run's own horizon is its `env_steps` column; naming the dependence on
        # the row is what lets the survival model see it, and it is not this
        # module's to resolve, since resolving it means either amending the plan
        # or re-expressing the metric on a common horizon.
        caveats.append(
            'censored at this run\'s own env_steps rather than at a horizon '
            'common to the table, so the censoring time varies with episode '
            'length (ANALYSIS_PLAN.md 5 calls it administrative, which holds '
            'for the episode budget and not for the env-step axis)')
    row.update(crossings)
    for column in EVAL_CADENCE_COLUMNS:
        row[column] = _tail_mean(df, column, DIAG_EVAL_TAIL)
    for column in EPISODE_CADENCE_COLUMNS:
        row[column] = _tail_mean(df, column, DIAG_EPISODE_TAIL)
    for checkpoint in PREFIX_CHECKPOINTS:
        entry = prefix_evals.get(str(checkpoint), prefix_evals.get(checkpoint))
        row[f'prefix_score_{checkpoint}'] = _num(
            entry.get('score') if isinstance(entry, dict) else None)

    row['metrics_problems'] = '; '.join(problems) or None
    row['derivation_caveats'] = '; '.join(caveats) or None
    row['__problems__'] = problems
    row['__caveats__'] = caveats
    row['__resolution__'] = mode
    row['__unverifiable_freeze__'] = unverifiable
    # The window the threshold metric was actually read off, per run. A window
    # rule that varies with the evaluation cadence has to be reported, not
    # assumed: the provenance records the distinct values so that a table built
    # from runs evaluated at two cadences says so.
    row['__window_points__'] = window_points
    row['__eval_cadence__'] = cadence
    row['__threshold_undefined__'] = bool(undefined)
    # The size of the P2 correction, per run. The fact of it is a caveat on the
    # row; the magnitude belongs in the provenance, because a co-primary
    # endpoint that moved deserves a number rather than an adjective.
    row['__auc_relative_change__'] = auc_relative_change
    row['__auc_disagrees__'] = auc_disagrees
    row['__curves__'] = df
    return row


def curve_rows(row: dict) -> tuple[pd.DataFrame, int]:
    """The long-form slice for one run, and how many duplicate rows it dropped.

    De-duplicated on `episode`, last row wins, which is `MetricsLog`
    write-replaces-episode semantics and is already what `threshold_crossings`
    applies. The two disagreed before: a log with one duplicated episode
    produced 41 curve rows for 40 episodes while every scalar derived from the
    same log used 40, so `curves.csv` and `per_seed.csv` described different
    data and no column in either said so. The corruption is not hidden by this:
    `metrics_contiguous` now travels on every curve row, the count of dropped
    rows reaches the provenance, and the run is named in the loud warning.

    The run-level columns are filled from the per-seed row by name. A positional
    slice through the flat schema did it before, which silently mis-assigns the
    moment the schema grows.
    """
    df: pd.DataFrame = row['__curves__']
    if df.empty:
        return pd.DataFrame(columns=list(CURVE_COLUMNS)), 0
    deduped = (df.drop_duplicates(subset='episode', keep='last')
               if 'episode' in df else df)
    dropped = int(len(df) - len(deduped))
    out = pd.DataFrame(index=deduped.index)
    for column in CURVE_RUN_COLUMNS:
        out[column] = row.get(column)
    for column in CURVE_COLUMNS[len(CURVE_RUN_COLUMNS):]:
        out[column] = deduped[column] if column in deduped else np.nan
    return out[list(CURVE_COLUMNS)], dropped


# ---------------------------------------------------------------------------
# Completeness
# ---------------------------------------------------------------------------
def duplicate_arm_seeds(frame: pd.DataFrame) -> list[tuple[str, int, int]]:
    """(arm, seed, run count) for every arm x seed carrying more than one run.

    A duplicate is not a second seed, it is a second configuration of one arm at
    one seed (a re-launch at a different budget, a measurement-only override, or
    two runs that lost their seed and collapsed onto the sentinel). Averaging
    the pair as though it were two independent draws deflates the across-seed SD
    by exactly sqrt(k(m-1)/(km-1)) for k duplicates of m seeds: in `runs_demo`,
    k=2 and m=3 turn a true SD of 0.4901 into 0.4384. `ANALYSIS_PLAN.md` §4
    gates the equivalence claim on that SD being below 0.05, and `DESIGN.md`
    §5.3 reports it as `across_seed_sd`, so a silent 11% understatement is a
    claim the data does not support. Nothing warned before this.
    """
    if frame.empty or 'arm' not in frame or 'seed' not in frame:
        return []
    counts = frame.groupby(['arm', 'seed'], dropna=False).size()
    return sorted((str(arm), int(seed), int(n))
                  for (arm, seed), n in counts.items() if n > 1)


Gap = tuple[str, str, int]


def missing_runs(selected: Sequence[str], frame: pd.DataFrame,
                 out_root: str) -> tuple[list[Gap], list[Gap]]:
    """Gaps and label-only matches, per declared arm x seed of the selection.

    Returns two lists: the arm x seed combinations with no run at all, and the
    ones satisfied only by an (arm label, seed) match rather than by
    configuration digest. The second is not a gap, but it is not the declared
    configuration either, and a completeness statement that does not distinguish
    them says "every declared arm x seed has a run" about a tree of
    reduced-budget validation runs (`STANDING_INSTRUCTIONS` S8). The caller
    prints the qualifier and the provenance records the count.

    The seed set an experiment is held to is the set of seeds at which *any* of
    its arms has a run, **intersected with the experiment's declared block**.
    Inferring it from the runs alone is what `DESIGN.md` §1 requires (holding a
    single-seed validation launch to the full `CONFIRM` block would make the
    check fire always, and a check that always fires means nothing), but
    inferring it from *all* the runs attributed to the experiment let seeds from
    a disjoint block onto the axis: E1 acquired seed 300 from the `C4SRC` donors
    whose configuration it shares, and then demanded twelve target-side runs at
    a seed `DESIGN.md` §3.4 reserves for source checkpoints. Holding every arm
    to the seeds its siblings reached, within the block the experiment declares,
    is the check the design asks for and nothing wider.

    `only_as_source` arms are excluded when inferring that seed set, and only
    when inferring it: they are still checked. `RESERVE` seeds are added back
    for the enumeration alone, from the runner's replacement ledger, so that a
    replacement source `DESIGN.md` §4.3 called for is reported when it is
    absent. `registry.jobs` builds only donor arms at a RESERVE seed, so this
    cannot resurrect the defect above.
    """
    present_digests = set(frame['run_digest'].dropna())
    present_label_seed = {(str(label), int(seed))
                          for label, seed in zip(frame['label'], frame['seed'])}
    try:
        _assignment, draws = registry.load_source_replacements(out_root)
    except Exception:                                     # noqa: BLE001
        draws = {}
    drawn_reserve = sorted({int(x) for seeds in draws.values() for x in seeds})
    missing: list[Gap] = []
    by_label: list[Gap] = []
    for eid in selected:
        exp = registry.EXPERIMENTS.get(eid)
        if exp is None:
            missing.append((eid, '<unknown experiment>', NO_SEED))
            continue
        declared = set(registry.SEED_BLOCKS.get(exp.seed_block, ()))
        off_axis = {arm.label for arm in exp.arms if arm.only_as_source}
        member = frame[frame['experiments'].fillna('').apply(
            lambda s, e=eid: e in str(s).split(';'))]
        seeds = sorted({int(seed)
                        for label, seed in zip(member['label'],
                                               member['seed'])
                        if str(label) not in off_axis
                        and int(seed) in declared})
        if not seeds:
            missing.append((eid, f'<no run on the seed axis of this '
                                 f'experiment: block {exp.seed_block}>',
                            NO_SEED))
            continue
        try:
            axis = sorted(set(seeds) | set(drawn_reserve))
            jobs = registry.jobs(eid, seeds=axis, out_root=out_root)
        except Exception as exc:                          # noqa: BLE001
            missing.append((eid, f'<catalogue error: {exc}>', NO_SEED))
            continue
        for job in jobs:
            if job.cfg.run_digest() in present_digests:
                continue
            if (job.arm, int(job.cfg.seed)) in present_label_seed:
                by_label.append((eid, job.arm, int(job.cfg.seed)))
                continue
            missing.append((eid, job.arm, int(job.cfg.seed)))
    return missing, by_label


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


def count_true(series: pd.Series) -> int:
    """How many entries are literally True, treating null as not-True.

    The counterpart of `count_false`, and needed for the same reason: since a
    censoring flag can be null (the metric was not measurable on that run),
    `series.sum()` either raises on the object column or counts a null as a
    censoring, which is the imputation ANALYSIS_PLAN.md 5 forbids.
    """
    return int((series.astype('object') == True).sum())           # noqa: E712


#: Column order of `arm_summary`, named so the frame exists even with no rows.
ARM_SUMMARY_COLUMNS: tuple[str, ...] = (
    'cell', 'condition', 'arm', 'seed_block', 'n_runs', 'n_seeds',
    'n_final_score', 'final_score_mean', 'final_score_sd', 'final_score_median',
    'n_auc_score', 'auc_score_mean', 'auc_score_sd', 'auc_score_median',
    'transferred_frac', 'invalid_sources')


def arm_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-arm descriptives. Descriptive only -- no test, no interval.

    Four counts, not one. The old `n` was `len(group)`: a row count, printed
    beside a mean and an SD that pandas had computed over the non-null subset
    and beside the `ANALYSIS_PLAN.md` §9 stamp that fires on n<3. All three
    readings of that one number can disagree, and each disagreement has been
    observed: an arm printed as n=7 with `final_score_sd` 0.0 because only two
    rows carried a score; an arm printed as n=3 with NaN means because none did,
    with the stamp silent; an arm printed as n=5 that was three seeds plus two
    duplicates. So the row count, the distinct-seed count and the per-endpoint
    non-null counts are all reported, and the caller stamps on the smallest.

    Grouped by `seed_block` as well as by arm, so no descriptive ever pools two
    blocks. `ANALYSIS_PLAN.md` §8 forbids any estimate computed on `TUNE` seeds
    and item 4 of §10 makes these descriptives a reported item, so a single mean
    over three CONFIRM seeds and two TUNE seeds is a forbidden estimate however
    it is labelled. Splitting rather than dropping is what keeps E3, whose whole
    declared block is TUNE, reportable on its own terms: the block is named in
    the row instead of being silently absorbed into it.
    """
    def _sd(series: pd.Series) -> float:
        clean = series.dropna()
        return float(clean.std(ddof=1)) if len(clean) > 1 else float('nan')

    rows = []
    for (cell, condition, arm, block), group in frame.groupby(
            ['cell', 'condition', 'arm', 'seed_block'], dropna=False):
        rows.append({
            'cell': cell, 'condition': condition, 'arm': arm,
            'seed_block': block,
            'n_runs': int(len(group)),
            'n_seeds': int(group['seed'].nunique(dropna=True)),
            'n_final_score': int(group['final_score'].notna().sum()),
            'final_score_mean': float(group['final_score'].mean()),
            'final_score_sd': _sd(group['final_score']),
            'final_score_median': float(group['final_score'].median()),
            'n_auc_score': int(group['auc_score'].notna().sum()),
            'auc_score_mean': float(group['auc_score'].mean()),
            'auc_score_sd': _sd(group['auc_score']),
            'auc_score_median': float(group['auc_score'].median()),
            'transferred_frac': float(
                group['transferred_param_fraction'].mean()),
            'invalid_sources': count_false(group['source_valid']),
        })
    return (pd.DataFrame(rows, columns=list(ARM_SUMMARY_COLUMNS))
            .sort_values(['cell', 'condition', 'arm', 'seed_block'])
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
    # Read from statlib rather than spelled out, for the reason the threshold
    # levels are: a ledger printed from its own literals can go on printing the
    # pre-registered family size after the family size has changed.
    print(f'  members  : {statlib.CONFIRMATORY_FAMILY_SIZE} = 4 cells x 2 '
          f'co-primary endpoints (final_score, auc_score)')
    print(f'  procedure: Holm-Bonferroni, step-down from alpha = '
          f'{statlib.ALPHA}/{statlib.CONFIRMATORY_FAMILY_SIZE} = '
          f'{statlib.HOLM_STRICTEST_ALPHA:g}')
    print('  screens  : Benjamini-Hochberg q, orientation only, no assertion '
          'permitted')
    print('  analyses in this output carrying a p-value: 0 -- aggregation '
          'emits no inference (see stats.py)')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _write_frames(frame: pd.DataFrame, curves: pd.DataFrame,
                  output: str, curves_path: str) -> None:
    """Write both tables, creating their directory if it does not exist."""
    for path in (output, curves_path):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
    frame.to_csv(output, index=False)
    curves.to_csv(curves_path, index=False)


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
                             'selected experiment has no run, or if any arm x '
                             'seed carries more than one')
    parser.add_argument('--require-integrity', action='store_true',
                        help='exit non-zero if any run failed the metrics '
                             'integrity check, has an unreadable manifest, is '
                             'unfinished (run output but no manifest), or was '
                             'produced under a superseded ANALYSIS_PLAN.md. '
                             'Off by default because this module writes a '
                             'failing run out rather than dropping it and '
                             'audit.py is what refuses to report on it; on '
                             'when the caller wants the refusal here')
    parser.add_argument('--quiet', action='store_true',
                        help='suppress the summary tables; warnings still print')
    args = parser.parse_args(argv)

    out_root = args.out_root
    output = args.output or os.path.join(out_root, 'per_seed.csv')
    curves_path = args.curves or os.path.join(out_root, 'curves.csv')

    # The tuned stage of DESIGN.md 3.3, resolved before the experiments are
    # selected because activating it changes which ids exist.
    tuning_selection, tuned_unavailable = activate_tuned_stage(out_root)
    if tuning_selection is not None:
        print(f'tuning selection {tuning_selection.short_id} '
              f'(rule {tuning_selection.rule.get("id")}'
              f'{", PLACEHOLDER" if tuning_selection.is_placeholder else ""}) '
              f'read from {tuning.selection_path(out_root)}: '
              f'{", ".join(sorted(registry.TUNED_OF))} of DESIGN.md 3.3 are in '
              f'the catalogue for this tree, and cells '
              f'{list(tuning_selection.shared_cells) or "[]"} share the common '
              f'policy\'s runs.')

    if args.experiments:
        selected = [e.strip() for e in args.experiments.split(',') if e.strip()]
        unknown = [e for e in selected if e not in registry.EXPERIMENTS]
        dormant = [e for e in unknown if e in registry.TUNED_OF]
        if dormant:
            print(f'{WARN} {dormant}: E1 and E2 under the secondary tuning '
                  f'policy of DESIGN.md 3.3, which cannot be enumerated '
                  f'because this tree holds no selection to build them from. '
                  f'{tuned_unavailable}')
            return 2
        if unknown:
            print(f'{WARN} unknown experiment ids: {unknown}. Known: '
                  f'{sorted(registry.EXPERIMENTS)}')
            return 2
    else:
        selected = list(registry.EXPERIMENTS)

    run_dirs = find_runs(out_root)
    unmanifested = find_unmanifested_runs(out_root)
    if not run_dirs:
        # No artifact is written here on purpose: an out-root with no runs in it
        # is as likely to be a mistyped path as an empty tree, and writing an
        # empty table into a mistyped path would create the stale artifact this
        # module is supposed to prevent.
        print(f'no runs found under {out_root}/ '
              f'(expected <condition>/<run_digest12>/s<NN>/manifest.json)')
        if unmanifested:
            print(f'{WARN} {len(unmanifested)} director(ies) under {out_root}/ '
                  f'hold run output with no manifest.json, so they are runs '
                  f'that died or are in flight, not an empty tree:')
            for path in unmanifested[:20]:
                print(f'  {_key(path)}')
            if len(unmanifested) > 20:
                print(f'  ... and {len(unmanifested) - 20} more')
        return 1

    seeds_seen: set[int] = set()
    for run_dir in run_dirs:
        base = os.path.basename(run_dir)
        if base.startswith('s') and base[1:].isdigit():
            seeds_seen.add(int(base[1:]))
    membership = Membership(sorted(seeds_seen), out_root,
                            selection=tuning_selection)
    for err in membership.errors:
        print(f'{WARN} catalogue entry could not be resolved -- {err}')

    rows: list[dict] = []
    # A directory holding run output but no manifest is a run that died or is
    # still in flight. It is unfinished, which is a different thing from
    # unreadable, and it is now in the accounting rather than invisible.
    unfinished: list[tuple[str, str]] = [
        (_key(d), 'run output but no manifest.json (killed mid-run, or still '
                  'in flight)')
        for d in unmanifested]
    unreadable: list[tuple[str, str]] = []
    for run_dir in run_dirs:
        try:
            row = per_seed_row(run_dir, membership)
        except Exception as exc:                              # noqa: BLE001
            # The B6 lesson, applied to the whole per-run body rather than to
            # the seed field alone. One corrupt run costs that run, never the
            # tree: three separate corruptions (a non-UTF-8 byte, a string
            # episode index, a JSONL line that is a list) each aborted the whole
            # aggregation, so a tree of good runs produced no table at all. Each
            # of those is now handled where it arises; this catch is what stops
            # the fourth one being found the same way. Nothing is swallowed: the
            # run is named on stdout, counted in the provenance, and refused by
            # --require-integrity.
            unreadable.append(
                (_key(run_dir),
                 f'{type(exc).__name__} while reading the run: {exc}'))
            continue
        if '__unreadable__' in row:
            unreadable.append((row['__unreadable__'], row['__reason__']))
            continue
        if '__incomplete__' in row:
            unfinished.append((row['__incomplete__'], row['__reason__']))
            continue
        rows.append(row)

    # Every refusal this invocation raised, in the order raised. The exit code
    # is derived from it, and it is stamped into the provenance so that a
    # refused aggregation leaves an artifact saying it was refused rather than a
    # clean-looking one and a return value nobody kept (DESIGN.md 8.4).
    refusals: list[str] = []

    if unreadable:
        print(f'\n{WARN} {len(unreadable)} run directory/ies hold a manifest '
              f'that cannot be read. They are neither in the table nor '
              f'silently gone: they are counted here and in the provenance, so '
              f'that run_directories_found = aggregated + unfinished + '
              f'unreadable with nothing unaccounted for.')
        for path, reason in unreadable[:20]:
            print(f'  {path}: {reason}')
        if len(unreadable) > 20:
            print(f'  ... and {len(unreadable) - 20} more')
        if args.require_integrity:
            refusals.append(f'{len(unreadable)} unreadable manifest(s)')

    if not rows:
        print(f'\n{WARN} {len(run_dirs)} run directories found, none finished '
              f'(no manifest carries a result block). The tables below are '
              f'written empty rather than left at their previous contents.')
        refusals.append('no finished run in the tree')

    # --- build the frames ---------------------------------------------------
    curve_pairs = [curve_rows(r) for r in rows]
    curve_frames = [c for c, _ in curve_pairs]
    curve_duplicates_dropped = sum(d for _, d in curve_pairs)
    problems_by_run = {r['run_dir']: list(r['__problems__']) for r in rows}
    caveats_by_run = {r['run_dir']: list(r['__caveats__']) for r in rows}
    auc_change_by_run = {r['run_dir']: r['__auc_relative_change__']
                         for r in rows}
    auc_disagree_runs = {r['run_dir'] for r in rows if r['__auc_disagrees__']}
    resolution_by_run = {r['run_dir']: r['__resolution__'] for r in rows}
    unverifiable_runs = {r['run_dir'] for r in rows
                         if r['__unverifiable_freeze__']}
    undefined_threshold_runs = {r['run_dir'] for r in rows
                                if r['__threshold_undefined__']}
    window_points_seen = sorted({int(r['__window_points__']) for r in rows
                                 if r['__window_points__'] is not None})
    cadences_seen = sorted({int(r['__eval_cadence__']) for r in rows
                            if r['__eval_cadence__'] is not None})
    for row in rows:
        for key in ('__curves__', '__problems__', '__caveats__',
                    '__resolution__', '__unverifiable_freeze__',
                    '__window_points__', '__eval_cadence__',
                    '__threshold_undefined__', '__auc_relative_change__',
                    '__auc_disagrees__'):
            row.pop(key, None)

    frame = pd.DataFrame(rows)
    for column in PER_SEED_COLUMNS:
        if column not in frame:
            frame[column] = None
    frame = (frame[list(PER_SEED_COLUMNS)]
             .sort_values(['cell', 'condition', 'arm', 'seed'])
             .reset_index(drop=True))
    # A seed is an integer. Left as float64 it writes as "300.0", which reads as
    # a measurement rather than as the identity of a run, and DESIGN.md 4.3
    # wants the identity of the rejected and replacement seeds legible in the
    # table itself.
    frame['source_seed'] = frame['source_seed'].astype('Int64')
    non_empty = [c for c in curve_frames if not c.empty]
    curves = (pd.concat(non_empty, ignore_index=True) if non_empty
              else pd.DataFrame(columns=list(CURVE_COLUMNS)))
    if not curves.empty:
        curves = (curves.sort_values(['run_dir', 'episode'])
                  .reset_index(drop=True))
    rows_before_selection = int(len(frame))
    modes_before_selection = dict(membership.mode_counts)
    # Computed here, before the selection, because after it every surviving row
    # contains the selected id by construction: the post-selection count is
    # structurally 0 under --experiments however unattributable the tree is, and
    # a reader of the E1 provenance alone would conclude there was nothing to
    # attribute. The same reasoning applies to `frame_before_selection`, which
    # the timing outliers are measured against: ms_per_env_step is a per-run
    # fact, so the four stalled runs must not disappear from the report of a
    # selection that happens to exclude them.
    unattributed_before_selection = int(
        (frame['experiments'].fillna('') == '').sum())
    frame_before_selection = frame

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
        if frame.empty and rows_before_selection:
            print(f'{WARN} nothing left after selecting {",".join(selected)}. '
                  f'The tables are written empty rather than left at their '
                  f'previous contents, because a stale artifact beside a '
                  f'non-zero exit code is how a superseded table gets read as '
                  f'a current one (DESIGN.md 9).')
            refusals.append(f'no run belongs to {",".join(selected)}')

    # --- what the *selected* table contains ---------------------------------
    # Computed here, not before the selection. The provenance previously
    # described the population walked rather than the population written: an
    # 8-row E3 table was stamped with a freeze-verification count of 44.
    selected_dirs = set(frame['run_dir'])
    offenders = sorted(str(d) for d, ok in zip(frame['run_dir'],
                                               frame['metrics_contiguous'])
                       if not bool(ok))
    short = sorted(d for d in selected_dirs
                   if any(p.startswith('short run')
                          for p in problems_by_run.get(d, ())))
    unverifiable = len(unverifiable_runs & selected_dirs)
    mode_counts = {'digest': 0, 'label': 0, 'unattributed': 0}
    for d in selected_dirs:
        mode_counts[resolution_by_run.get(d, 'unattributed')] += 1

    if offenders:
        print('\n' + '!' * 72)
        print(f'{WARN} METRICS INTEGRITY FAILED for {len(offenders)} of '
              f'{len(frame)} runs. A duplicated or missing episode is invisible '
              f'in an aggregate -- it merely shifts a window mean -- and it '
              f'corrupts every window statistic downstream. These runs are '
              f'written out with metrics_contiguous=False and may not be '
              f'reported until audit.py has been satisfied:')
        for run_dir in offenders:
            print(f'  {run_dir}')
            for problem in problems_by_run.get(run_dir, ()):
                print(f'      - {problem}')
        print('!' * 72)
        if args.require_integrity:
            refusals.append(f'{len(offenders)} run(s) failed metrics integrity')
    if short and not args.quiet:
        # No longer conditioned on metrics_contiguous: a run that is both short
        # and non-contiguous is short, and excluding it from the count made
        # runs_short_of_budget read 0 for a tree that had one.
        print(f'\n{WARN} {len(short)} run(s) finished short of num_episodes '
              f'(not the declared budget):')
        for run_dir in short[:20]:
            detail = next(p for p in problems_by_run[run_dir]
                          if p.startswith('short run'))
            print(f'  {run_dir}: {detail}')
        if len(short) > 20:
            print(f'  ... and {len(short) - 20} more')
    if unfinished:
        print(f'\n{WARN} {len(unfinished)} run directory/ies hold no finished '
              f'run and are excluded from the table. An arm that quietly loses '
              f'a seed here is averaged over the survivors, which is the '
              f'published defect DESIGN.md 1 names:')
        if args.require_integrity:
            refusals.append(f'{len(unfinished)} unfinished run director(ies)')
        for path, reason in unfinished[:20]:
            print(f'  {path}: {reason}')
        if len(unfinished) > 20:
            print(f'  ... and {len(unfinished) - 20} more')

    # --- duplicates, unresolved seeds, timing -------------------------------
    duplicates = duplicate_arm_seeds(frame)
    if duplicates:
        print(f'\n{WARN} {len(duplicates)} arm x seed combination(s) carry more '
              f'than one run. A duplicate is a second *configuration* of one '
              f'arm at one seed, not a second seed: pooling it deflates the '
              f'across-seed SD that ANALYSIS_PLAN.md 4 gates the equivalence '
              f'claim on, and every mean below is computed over rows:')
        for arm, seed, n in duplicates[:20]:
            where = f'seed {seed}' if seed != NO_SEED else 'no resolvable seed'
            print(f'  {arm} {where}: {n} runs')
        if len(duplicates) > 20:
            print(f'  ... and {len(duplicates) - 20} more')
        if args.require_complete:
            refusals.append(f'{len(duplicates)} duplicated arm x seed '
                            f'combination(s)')

    # Grouped by text rather than printed per run: on the P0 tree every
    # manifest carries the same jumpstart bookkeeping error, and forty-four
    # identical lines would bury the two that are not identical.
    caveat_counts: dict[str, int] = {}
    for run_dir in selected_dirs:
        for caveat in caveats_by_run.get(run_dir, ()):
            caveat_counts[caveat] = caveat_counts.get(caveat, 0) + 1
    if caveat_counts:
        print(f'\n{WARN} a manifest field contradicts DESIGN.md on '
              f'{sum(1 for d in selected_dirs if caveats_by_run.get(d))} of '
              f'{len(frame)} runs. The derived column is withheld or overridden '
              f'rather than repeated, and the reason travels in '
              f'derivation_caveats. The manifest writer is what has to change:')
        for caveat, count in sorted(caveat_counts.items(),
                                    key=lambda kv: (-kv[1], kv[0]))[:10]:
            print(f'  {count} run(s): {caveat}')

    # Grouped text says a co-primary endpoint moved; it does not say by how
    # much, and on P2 that is the question. Printed as its own line for that
    # reason, and recorded in the provenance beside it.
    auc_disagreeing = len(auc_disagree_runs & selected_dirs)
    auc_changes = sorted(float(auc_change_by_run[d]) for d in selected_dirs
                         if auc_change_by_run.get(d) is not None)
    if auc_disagreeing:
        magnitude = (f'by {auc_changes[0] * 100:+.2f}% to '
                     f'{auc_changes[-1] * 100:+.2f}% (median '
                     f'{float(np.median(auc_changes)) * 100:+.2f}%)'
                     if auc_changes else 'in a way no ratio describes')
        print(f'{WARN} auc_score was integrated here on {auc_disagreeing} of '
              f'{len(frame)} run(s) and disagrees with the manifest '
              f'{magnitude}. The table carries the integrated value, so '
              f'those runs are corrected by re-derivation and do not need '
              f're-running.')

    seedless = sorted(frame.loc[frame['seed'] == NO_SEED, 'run_dir'])
    if seedless:
        print(f'\n{WARN} {len(seedless)} run(s) carry no resolvable seed in '
              f'either the config or the identity block. They are in the table '
              f'with seed={NO_SEED} and seed_block=UNKNOWN, which is a '
              f'placeholder and not a seed: they cannot be attributed to a '
              f'declared arm x seed and two of them collide with each other:')
        for run_dir in seedless[:20]:
            print(f'  {run_dir}')
        if len(seedless) > 20:
            print(f'  ... and {len(seedless) - 20} more')

    # A stall is an environmental cost, not a workload cost, and plan.py fits
    # its throughput model on these numbers. The threshold is a multiple of the
    # median rather than an absolute, because ms/step is architecture- and
    # machine-dependent. Measured on the whole walked tree rather than on the
    # selection: the column is a per-run fact whose value does not change with
    # the selection, and computing it on the selected frame made the four
    # 159-192 ms/step runs vanish from an E1 provenance that still described
    # them as a per-run fact. A run outside the selection is marked as such
    # rather than dropped.
    timing = frame_before_selection['ms_per_env_step'].dropna().astype(float)
    outliers: list[tuple[str, float, bool]] = []
    median = None
    if len(timing) >= 4:
        median = float(timing.median())
        if median > 0:
            mask = frame_before_selection['ms_per_env_step'].astype(float) >= (
                TIMING_OUTLIER_FACTOR * median)
            outliers = sorted(
                (str(r), float(v), str(r) in selected_dirs) for r, v in zip(
                    frame_before_selection.loc[mask, 'run_dir'],
                    frame_before_selection.loc[mask, 'ms_per_env_step']))
            if outliers:
                print(f'\n{WARN} {len(outliers)} run(s) spent at least '
                      f'{TIMING_OUTLIER_FACTOR}x the walked tree median of '
                      f'{median:.1f} ms per env step. ms_per_env_step is a '
                      f'column on every row so that a cost model can exclude '
                      f'them; nothing here excludes anything:')
                for run_dir, value, kept in outliers[:20]:
                    where = '' if kept else '  (outside this selection)'
                    print(f'  {run_dir}: {value:.1f} ms/step{where}')
                if len(outliers) > 20:
                    print(f'  ... and {len(outliers) - 20} more')

    # --- completeness -------------------------------------------------------
    gaps: list[Gap] = []
    by_label: list[Gap] = []
    if args.require_complete:
        gaps, by_label = missing_runs(selected, frame, out_root)
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
            refusals.append(f'{len(gaps)} missing arm x seed combination(s)')

    # --- write --------------------------------------------------------------
    _write_frames(frame, curves, output, curves_path)

    unattributed = int((frame['experiments'].fillna('') == '').sum())
    input_commits = sorted(set(frame['git_commit'].dropna()))
    input_plans = sorted(set(frame['plan_hash'].dropna()))

    # --- the pre-registration the input runs were produced under -------------
    # Two different failures, kept apart because they carry different weight.
    #
    # *Split*: one table pooling runs produced under more than one
    # `ANALYSIS_PLAN.md`. The module docstring calls that a reporting-stopper
    # and the code only printed it, with the check absent from `refusals` and so
    # from the exit code. It is a refusal now, which is what the word means.
    #
    # *Stale*: every input run agrees with every other, and all of them predate
    # the current plan. That is the tree's state today and it was not even
    # printed, because the condition was `len(input_plans) > 1` and nothing ever
    # compared the runs' hashes against the plan file on disk: the provenance
    # held both, in adjacent keys, with nothing looking at them together. It is
    # printed and recorded on every invocation now, and refused under
    # `--require-integrity`. Not an unconditional refusal, because `audit.py`
    # already raises `plan_hash_stale` as an ERROR and `report.py` gates the
    # whole bundle on that audit with an override that stamps every artifact it
    # writes; a hard refusal here would remove that stamped path and leave
    # `ANALYSIS_PLAN.md` §9 pipeline validation with no way to run at all.
    current_plan = (provenance.plan_hashes() or {}).get('ANALYSIS_PLAN.md')
    superseded_plans = ([h for h in input_plans if h != current_plan]
                        if current_plan else [])
    plan_split = len(input_plans) > 1
    plan_stale = bool(superseded_plans)
    if plan_split:
        print(f'\n{WARN} {len(input_plans)} distinct ANALYSIS_PLAN.md hashes '
              f'across the input runs. A confirmatory result is interpretable '
              f'only against the one pre-registration in force when it ran, '
              f'which ANALYSIS_PLAN.md 1 makes a reporting-stopper, so this '
              f'aggregation is refused: {input_plans}')
        refusals.append(f'{len(input_plans)} distinct ANALYSIS_PLAN.md hashes '
                        f'across the input runs')
    if plan_stale:
        print(f'\n{WARN} {len(superseded_plans)} ANALYSIS_PLAN.md hash(es) '
              f'carried by the input runs are not the plan on disk '
              f'({current_plan}): {superseded_plans}. The runs were produced '
              f'under a superseded pre-registration, so nothing in this table '
              f'is a confirmatory result against the current plan '
              f'(ANALYSIS_PLAN.md 1, 11).'
              + ('' if args.require_integrity else
                 ' Pass --require-integrity to make this a refusal here; '
                 'audit.py raises it as an error either way.'))
        if args.require_integrity:
            refusals.append('input runs carry a superseded ANALYSIS_PLAN.md '
                            'hash')

    # The censoring time of `steps_to_threshold` is each run's own `env_steps`,
    # and those vary across the tree because the metric is in env steps while
    # the budget is in episodes. ANALYSIS_PLAN.md 5 calls the censoring
    # administrative and independent of the event time; the spread is recorded
    # here, unconditionally and with no threshold on it, so that the assumption
    # can be checked against the table it is asserted about rather than taken.
    horizons = (pd.to_numeric(frame['env_steps'], errors='coerce')
                .dropna().astype(float))
    censoring_horizon = None
    if len(horizons):
        low, high = float(horizons.min()), float(horizons.max())
        censoring_horizon = {
            'unit': 'env steps', 'min': low, 'max': high,
            'ratio': (high / low) if low > 0 else None,
            'note': 'steps_to_threshold is censored at each run env_steps, not '
                    'at a horizon common to the table'}
    prov = {
        'tool': 'experiments/aggregate.py',
        'per_seed_csv': output.replace(os.sep, '/'),
        'per_seed_sha': provenance.file_hash(output),
        'per_seed_rows': int(len(frame)),
        'curves_csv': curves_path.replace(os.sep, '/'),
        'curves_sha': provenance.file_hash(curves_path),
        'curves_rows': int(len(curves)),
        'curve_rows_dropped_as_duplicate_episodes': curve_duplicates_dropped,
        'out_root': out_root.replace(os.sep, '/'),
        # The accounting identity, in the two halves it actually has:
        #   run_directories_found = aggregated_before_selection + unfinished
        #                           + unreadable
        #   aggregated_before_selection = aggregated + excluded_by_selection
        # `runs_aggregated` is the count of rows *written*, which is
        # post-selection, so the first identity was stated over a number that
        # does not belong to it: an E1 run recorded found=44, aggregated=16 and
        # nothing to account for the other 28. Both halves are now closed by
        # their own keys, and manifest-less directories are inside the first.
        'run_directories_found': len(run_dirs) + len(unmanifested),
        'runs_aggregated': int(len(frame)),
        'runs_aggregated_before_selection': rows_before_selection,
        'runs_excluded_by_selection': rows_before_selection - int(len(frame)),
        'runs_unfinished': len(unfinished),
        'runs_unfinished_paths': [p for p, _ in unfinished],
        'runs_unreadable': len(unreadable),
        'runs_unreadable_paths': [p for p, _ in unreadable],
        'runs_failing_integrity': len(offenders),
        'runs_failing_integrity_paths': offenders,
        'runs_short_of_budget': len(short),
        'runs_with_unverifiable_freeze_event': unverifiable,
        # The counter above is 44 of 44 by construction on this tree, because
        # every manifest records the episode-0 freeze with a null verification
        # payload, so it can take no other value and carries no signal. These
        # three split the same population by what was actually checked, and the
        # `freeze_verifiable` column carries it per run.
        'runs_with_verified_freeze_event': int(
            (frame['freeze_verifiable'].astype('object') == True).sum()),
        'runs_with_freeze_but_nothing_checked': int(
            (frame['freeze_verifiable'].astype('object') == False).sum()),
        'runs_with_no_freeze_event': int(
            frame['freeze_verifiable'].isna().sum()),
        'runs_unattributed_to_any_experiment': unattributed,
        'runs_unattributed_to_any_experiment_before_selection':
            unattributed_before_selection,
        'runs_with_no_resolvable_seed': len(seedless),
        'runs_with_derivation_caveats': sum(
            1 for d in selected_dirs if caveats_by_run.get(d)),
        'derivation_caveat_counts': caveat_counts,
        'duplicate_arm_seed_combinations': [
            {'arm': arm, 'seed': seed, 'runs': n}
            for arm, seed, n in duplicates],
        'timing_outliers': [
            {'run_dir': run_dir, 'ms_per_env_step': round(value, 3),
             'in_selection': kept}
            for run_dir, value, kept in outliers],
        'timing_outlier_factor': TIMING_OUTLIER_FACTOR,
        'timing_median_ms_per_env_step': median,
        'membership_resolution': mode_counts,
        'membership_resolution_before_selection': modes_before_selection,
        'experiments_selected': selected if args.experiments else 'all',
        'tuning_selection': tuning_selection_record(
            tuning_selection, out_root, tuned_unavailable),
        'require_complete_requested': bool(args.require_complete),
        'require_complete_passed': (not (gaps or duplicates)
                                    if args.require_complete else None),
        'completeness_gaps': [{'experiment': e, 'arm': a, 'seed': s}
                              for e, a, s in gaps],
        'completeness_satisfied_by_label_only': [
            {'experiment': e, 'arm': a, 'seed': s} for e, a, s in by_label],
        'require_integrity_requested': bool(args.require_integrity),
        'require_integrity_passed': (
            not (offenders or unreadable or unfinished or plan_stale)
            if args.require_integrity else None),
        'refusals': refusals,
        'exit_code': 1 if refusals else 0,
        'schema': {'per_seed': list(PER_SEED_COLUMNS),
                   'curves': list(CURVE_COLUMNS)},
        # P2's definition, in the provenance rather than in a docstring alone,
        # because the divisor is the whole of the defect this records.
        'auc_score_source': (
            'recomputed from metrics.jsonl: trapezoid of eval_score over '
            'env_steps, divided by the run total env_steps (DESIGN.md 5.2)'),
        'runs_whose_manifest_auc_disagrees': auc_disagreeing,
        'auc_manifest_relative_change': (
            {'n': len(auc_changes), 'min': auc_changes[0],
             'max': auc_changes[-1],
             'median': float(np.median(auc_changes))} if auc_changes else None),
        'threshold_levels': {name: level for name, level in THRESHOLD_LEVELS},
        'threshold_levels_source': 'statlib.THRESHOLD_LEVELS',
        'trailing_window_episodes': TRAILING_WINDOW,
        # The window rule, per run, because it is derived from the evaluation
        # cadence: a complete trailing window holds TRAILING_WINDOW//eval_every
        # points, and a partial window no longer dates a crossing. Recorded so
        # that a table pooling two cadences says so, and so that the count the
        # metric was actually computed on is not a docstring claim.
        'trailing_window_observations': window_points_seen,
        'evaluation_cadence_episodes': cadences_seen,
        'runs_with_unmeasurable_threshold_metric': len(
            undefined_threshold_runs & selected_dirs),
        'censoring_horizon_env_steps': censoring_horizon,
        'diagnostic_windows': {'eval_cadence_points': DIAG_EVAL_TAIL,
                               'episodes': DIAG_EPISODE_TAIL},
        'git': provenance.git_state(),
        'plans': provenance.plan_hashes(),
        'plan_hash_current': current_plan,
        'plan_hash_split': plan_split,
        'plan_hash_stale': plan_stale,
        'plan_hashes_superseded': superseded_plans,
        'input_run_git_commits': input_commits,
        'input_run_plan_hashes': input_plans,
        'argv': list(sys.argv if argv is None else ['aggregate.py', *argv]),
        'cwd': os.getcwd(),
    }
    prov_path = os.path.splitext(output)[0] + '.provenance.json'
    with open(prov_path, 'w', encoding='utf-8') as fh:
        json.dump(prov, fh, indent=2, sort_keys=True)

    if mode_counts['label']:
        print(f'\n{WARN} {mode_counts["label"]} run(s) were '
              f'attributed to an experiment by (arm label, seed) rather than by '
              f'configuration digest, so their configuration is not the one the '
              f'catalogue declares -- expected for a reduced-budget validation '
              f'launch, never acceptable for a confirmatory table.')
    if unattributed:
        print(f'{WARN} {unattributed} run(s) belong to no catalogue '
              f'experiment at a seed of its declared block; the '
              f'experiments column is empty for them.')

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
        print('one row per arm x seed block: a descriptive is never pooled '
              'across blocks (ANALYSIS_PLAN.md 8)')
        counts = [summary[c].min() for c in
                  ('n_seeds', 'n_final_score', 'n_auc_score') if len(summary)]
        if counts and int(min(counts)) < MIN_N_FOR_INFERENCE:
            # ANALYSIS_PLAN.md 9. Printed above the numbers rather than below,
            # so it cannot be read past. Fired on the smallest of the three
            # counts, because the row count was the one number of the three that
            # could not make it fire. The floor and the label are read from
            # statlib, which validate.py pins to the plan, rather than spelled
            # out here where nothing checked them.
            print(f'{VALIDATION_STAMP}: an arm has fewer than '
                  f'{MIN_N_FOR_INFERENCE} seeds or fewer than '
                  f'{MIN_N_FOR_INFERENCE} non-null values of a co-primary '
                  f'endpoint (smallest count {int(min(counts))}), so no number '
                  f'below may be quoted, compared, or used to choose between '
                  f'hypotheses.')
        print(summary.round(4).to_string(index=False))

        # Three states, counted as three. A null flag is a run whose evaluation
        # series cannot support the metric at all, and folding it into either
        # "reached" or "never reached" is the imputation ANALYSIS_PLAN.md 5
        # forbids.
        censored = ', '.join(
            f'{name}={count_true(frame[f"censored_{name}"])}/{len(frame)}'
            for name, _level in THRESHOLD_LEVELS)
        unmeasured = len(undefined_threshold_runs & selected_dirs)
        print(f'\ncensored at budget (level never reached): {censored}')
        if censoring_horizon and censoring_horizon['ratio']:
            print(f'  the censoring time is each run own env_steps, '
                  f'{censoring_horizon["min"]:,.0f} to '
                  f'{censoring_horizon["max"]:,.0f} '
                  f'({censoring_horizon["ratio"]:.2f}x across this table). '
                  f'ANALYSIS_PLAN.md 5 calls the censoring administrative and '
                  f'independent of the event time: that holds for the episode '
                  f'budget and not for the env-step axis the metric is '
                  f'measured on.')
        if unmeasured:
            print(f'  {unmeasured} run(s) carry no threshold metric at all '
                  f'(null time, null flag): their evaluation series never '
                  f'fills a trailing-{TRAILING_WINDOW}-episode window, and '
                  f'missing is not the same as censored at the budget.')
        moved = count_false(frame['freeze_verified'])
        checked = int((frame['freeze_verifiable'].astype('object') == True).sum())
        nothing = int((frame['freeze_verifiable'].astype('object') == False).sum())
        no_freeze = int(frame['freeze_verifiable'].isna().sum())
        print(f'metrics_contiguous: '
              f'{int(frame["metrics_contiguous"].sum())}/{len(frame)}   '
              f'freeze_verified: {len(frame) - moved}/{len(frame)}, of which '
              f'{checked} had a verification payload to check. '
              f'{nothing} run(s) froze with no payload (an initial freeze has '
              f'no earlier fingerprint to compare against) and {no_freeze} '
              f'never froze, so on {nothing + no_freeze} row(s) '
              f'freeze_verified=True means nothing was checked. The '
              f'freeze_verifiable column is what separates them; '
              f'{unverifiable} run(s) had at least one unverifiable event.')
        print(f'invalid sources: {count_false(frame["source_valid"])} of '
              f'{int(frame["source_valid"].notna().sum())} runs with a '
              f'source')
        blocks = frame['seed_block'].value_counts().to_dict()
        print(f'seed blocks: {blocks}')
        if blocks.get('TUNE'):
            print(f'{WARN} {blocks["TUNE"]} run(s) draw on the TUNE block. No '
                  f'reported estimate may touch them (DESIGN.md 3.4); audit.py '
                  f'enforces it.')
        if blocks.get('UNKNOWN'):
            print(f'{WARN} {blocks["UNKNOWN"]} run(s) sit in no declared seed '
                  f'block (DESIGN.md 3.4 makes the blocks exhaustive for '
                  f'anything the catalogue schedules), so nothing can say '
                  f'whether they were tuned on.')
        print_ledger()

    if args.require_complete and not gaps:
        qualifier = ''
        if by_label:
            qualifier = (f', but {len(by_label)} of them are satisfied by an '
                         f'(arm label, seed) match rather than by configuration '
                         f'digest, so those runs are not the configuration the '
                         f'catalogue declares')
        print(f'\ncompleteness: every declared arm x seed of '
              f'{",".join(selected)} has a run{qualifier}.')
        if duplicates:
            print(f'{WARN} completeness is not uniqueness: '
                  f'{len(duplicates)} arm x seed combination(s) have more than '
                  f'one run.')

    if refusals:
        print(f'\n{WARN} REFUSED: {"; ".join(refusals)}. Recorded in '
              f'{prov_path}.')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
