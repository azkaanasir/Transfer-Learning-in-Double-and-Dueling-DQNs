"""Cost and inventory model: what a selection costs, before anything launches.

    python experiments/plan.py --experiments E1 E2 --seeds CONFIRM
    python experiments/plan.py --tier 1
    python experiments/plan.py --all --jobs 6
    python experiments/plan.py --measure            # measure this machine
    python experiments/plan.py --from-runs runs     # calibrate from a real tree
    python experiments/plan.py --all --json         # machine-readable

`DESIGN.md` §8.5 requires that run count, throughput, projected wall-clock per
`--jobs`, disk, and **the seed block of every run** be printed before anything
launches. This file is that requirement. It launches nothing and writes into no
analysis run tree; the one exception is `--measure`, whose calibration runs go to
a root of their own (`runs_calibration/`) outside `runs/` altogether, because
`audit.py` globs `runs/**/manifest.json` recursively and a nested directory would
be audited as data however it were named.

Why each section exists -- each one is a defect made visible in advance:

* **De-duplicated run count.** A run is identified by a digest over every field
  that can change its trajectory or its measurement, not by the experiment that
  asked for it (`src/dqn/config.py`, `run_dir`). Identical configurations are
  therefore literally the same run: E4's freeze level that equals E1's protocol
  value, E7's `aggregation='mean'` scratch arms, E8's unshifted scratch arms.
  Summing per-experiment counts overstates the catalogue by roughly a quarter,
  and a plan quoting that sum invites the wrong economy -- cutting an experiment
  that costs almost nothing extra. Both numbers are printed, with the saving.
* **Seed block of every run.** Revision 1 of the design selected hyperparameters
  on seeds 0-4 and then ran every confirmatory arm on 0-9, so half of each
  confirmatory sample had been tuned on (`DESIGN.md` §3.4 and §11 item 2).
  `audit.py` catches that after the compute is spent; this catches it before. A
  `TUNE` seed reaching an experiment whose family is `confirmatory` or
  `estimation` is a loud warning and a non-zero exit code.
* **Measured, partly measured, or estimated throughput.** The published study's
  compute budget was reconstructed after the fact and could not be checked. Here
  every projection carries the provenance of the coefficients behind it:
  `measured` only when every coefficient in force was derived from data,
  `PARTLY MEASURED` with the documented coefficients named when a calibration
  fitted some of them, and `ESTIMATE, not measured` when none were. Calling a
  harvested table "measured" was the earlier defect: a harvest fits the step rate
  and the length ramp and leaves the per-episode overhead and the process startup
  documented, and printing that as measured hid a table that was a factor of two
  out on its own inputs.
* **A calibration checked before it is believed, and only then written.** A
  harvest is fitted, checked against the runs it was fitted from, checked against
  the documented anchors, and written last. A fit that cannot reproduce its own
  training data is refused rather than committed to disk. That check is not
  hypothetical: a convention error deflated every harvested coefficient by
  1/1.35, which showed up as a median predicted/actual ratio of 0.74 on the very
  runs the fit came from while the report said `measured`.
* **Outliers excluded, named, and counted.** Four of the 44 P0 runs were stalled
  by the claim collision recorded as `PRELAUNCH_FIXES.md` B1 and read 159 to
  192 ms/step against a whole-tree median of 6.1. Pooled into the fit they
  inflated the LunarLander rate by up to 4.6x and inverted the architecture
  ordering, pricing the cheap cells as the expensive ones. A stall is a property
  of the machine on that night and not of the arm, so the harvest gates on the
  implied step rate and prints every run it dropped with its rate and its reason,
  rather than dropping anything silently (`STANDING_INSTRUCTIONS` S6).
* **Source-validity rejections, priced rather than waved at.** `DESIGN.md`
  §4.3 replaces a source that fails the validity gate from the `RESERVE`
  block and requires the number and identity of the rejected seeds to be
  reported. This module reads the runner's rejection ledger so that it plans the
  job graph that was actually run, and reads the validity verdict recorded in
  each finished run's manifest so that a run standing on a rejected source is
  counted as work still to do rather than as work already done. In P0 that is
  two runs of E1's sixteen, downstream of a source that scored 0.599 against a
  0.600 gate.
* **Disk.** The replay buffer is ~5.3 MB per copy and a run holds one copy per
  live checkpoint, so the transient peak scales with `--jobs` and with
  `prefix_checkpoints`, not with the total run count. Durable artifacts are
  ~0.6 MB per run. Confusing the two is how a sweep fills a disk at hour six.

The cost model itself is two coefficients per (environment, architecture): a
per-episode overhead and a per-env-step rate, over an episode-length ramp.
Episodes are not the cost unit: LunarLander episode length ranges over an order
of magnitude with performance (`DESIGN.md` §3.2), which is exactly why a
per-episode figure rises several-fold within one run. A model in env steps tracks
that, prices the shift variants from the *measured* random-policy episode lengths
in `reference_returns.json` rather than assuming they cost what the base
environment costs, and can be checked against a finished run tree with
`--from-runs`. A variant with enough finished runs of its own is fitted on its
own runs, rate included; a variant with too few is pooled into its base
environment and the note on the entry says so, because the pooled rate is an
assumption and the shipped table once carried it as though it were measured.

Two coefficients are separately identifiable only when the harvested runs differ
materially in steps per episode and in episode budget. On a tree where every run
has the same budget they are not, so the usual path is the constrained one: the
per-episode overhead is held at its documented value and only the rate is
fitted. The unconstrained non-negative least squares fit exists for a tree that
does span budgets, and the guard that chooses between them prints which fit it
used.

Nothing here decides anything scientific. It decides what to launch and in what
order, and it prints the seed-block ledger that makes a selection auditable
before it becomes data.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import statistics
import sys
import textwrap
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

# Set before the registry pulls in TensorFlow via `networks`; a planning tool
# that prints a page of device warnings is a planning tool people stop reading.
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# `experiments/` as well as the repository root. `registry.py` reaches
# `tuning.py` by absolute import while putting only the root on the path, so
# without this line every tuned id (`registry.TUNED_OF`) fails to resolve here
# with ModuleNotFoundError instead of with the refusal that names what to run.
# `sweep.py` and `aggregate.py` add both already; this file added only the root.
for _path in (REPO, os.path.join(REPO, 'experiments')):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from experiments import registry as reg                          # noqa: E402
from src.dqn import envs, provenance                             # noqa: E402
from src.dqn.config import Config                                # noqa: E402

import tuning                                                    # noqa: E402

THROUGHPUT_FILE = os.path.join(REPO, 'experiments', 'throughput.json')
THROUGHPUT_SCHEMA = 'throughput-v1'
# Outside `runs/`, not merely underneath it: `audit.py` walks
# `runs/**/manifest.json` recursively, so a calibration run parked in a nested
# directory would be read as data no matter what the directory was called.
CALIBRATION_ROOT = 'runs_calibration'

# ---------------------------------------------------------------------------
# Disk. Both figures are per run: the durable figure is the manifest, the three
# jsonl logs, the final model, and the weight and optimiser state of the rolling
# and prefix checkpoints; the buffer figure is one `ReplayBuffer.save` at the
# default capacity. `--from-runs` re-measures the durable figure from an actual
# tree, because it grows with `num_episodes` (metrics.jsonl) and with the number
# of prefix checkpoints.
# ---------------------------------------------------------------------------
DURABLE_MB_PER_RUN = 0.60
TRANSIENT_BUFFER_MB = 5.30


# ---------------------------------------------------------------------------
# Cost anchors. Documented, and labelled as such wherever they are used.
#
# Re-derived from the 44-run P0 tree, which was produced on this machine by a
# sweep at `workers=4, threads_per_worker=2`: that is the CONTENTION table's
# anchor condition exactly, so the figures need no contention correction. The
# four runs stalled by the claim collision recorded as `PRELAUNCH_FIXES.md` B1
# are excluded, as they are from every fit here.
#
#   * per env step, stall-excluded medians: LunarLander 6.1 ms (dueling) and
#     5.2 ms (mlp), CartPole 6.1 ms (dueling) and 6.2 ms (mlp). The two
#     environments cost the same per step because the cost is the gradient step
#     and not the physics, which is also why a rate harvested on one environment
#     is a defensible fallback for another.
#   * whole runs, 1000 episodes: LunarLander 28 to 45 min, CartPole 14 to 21 min
#
# The values that stood here before came from an earlier dev-machine measurement
# quoted as ~0.19 s/episode on CartPole and 15 to 25 min per LunarLander run.
# They were a factor of four optimistic on the rate against every run in P0, and
# because the fallback path uses them a plan with no calibration table on disk
# under-priced the sweep four-fold. They are not retained: an anchor nobody can
# reproduce is not a documented default, it is a guess with a provenance note.
#
# `s_per_episode` remains documented rather than measured. A tree in which every
# run has the same episode budget cannot separate a per-episode overhead from a
# per-step rate (see `harvest`), and holding the overhead is the conservative
# side of that choice: it is a few per cent of a run, the rate is nearly all of
# it.
#
# `s_per_env_step` is quoted for the dueling architecture and scaled to `mlp` by
# ARCH_FACTOR. `steps_end` is the plateau of the linear length ramp, fitted so
# that the modelled total env steps over a whole run matches what the P0 runs
# actually accumulated; it is not the length a finished policy reaches, and on
# LunarLander it is deliberately above it (see `solve_steps_end`).
#
# `check_seconds` is a plausible band for a whole run at `check_episodes`, wide
# enough to hold every clean P0 run of that environment. It is what the report's
# self-check tests the model in force against, so that a convention error or a
# contaminated fit is caught by the tool rather than by a reader.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Anchor:
    s_per_env_step: float          # dueling, at the anchor's contention level
    s_per_episode: float           # per-episode overhead: reset, logging, I/O
    steps_start: float             # fallback only; a measured value wins
    steps_end: float
    ramp_episodes: int
    note: str
    check_episodes: int = 1000     # budget the band below is quoted at
    # Whole-run wall clock, low and high. (0.0, 0.0) means no band is claimed
    # and the self-check reports the number without a verdict.
    check_seconds: tuple = (0.0, 0.0)


DEFAULT_ANCHORS: dict[str, Anchor] = {
    'LunarLander-v3': Anchor(
        s_per_env_step=0.0061, s_per_episode=0.023,
        steps_start=94.0, steps_end=336.0, ramp_episodes=300,
        check_episodes=1000, check_seconds=(1400.0, 3400.0),
        note='documented anchor, from the stall-excluded P0 median of '
             '6.1 ms/step: a 1000-episode run accumulates about 300k env steps '
             'and takes roughly half an hour at four resident workers. '
             'steps_end 336 is the ramp plateau that reproduces that step '
             'total, and it sits above the 205-step mean of the final hundred '
             'episodes because LunarLander episode length is not monotone: it '
             'peaks while the policy hovers and shortens once it lands'),
    'CartPole-v1': Anchor(
        s_per_env_step=0.0061, s_per_episode=0.020,
        steps_start=22.5, steps_end=214.0, ramp_episodes=400,
        check_episodes=1000, check_seconds=(700.0, 1700.0),
        note='documented anchor, from the P0 median of 6.1 ms/step: a '
             '1000-episode run accumulates 128k to 207k env steps and takes '
             '14 to 21 min. steps_end 214 is the ramp plateau that reproduces '
             'that step total, well under the 500-step environment cap, '
             'because epsilon anneals over 900 of the 1000 episodes '
             '(episode-indexed since DESIGN.md revision 5, which discarded the '
             'step-indexed schedule this note used to cite) and most episodes '
             'are therefore still partly exploratory; the final hundred '
             'episodes do reach 380 to 434 steps'),
    'Acrobot-v1': Anchor(
        s_per_env_step=0.0061, s_per_episode=0.020,
        steps_start=497.0, steps_end=150.0, ramp_episodes=300,
        note='INFERRED, no anchor measured here: rate taken from CartPole (both '
             'are cheap classic-control dynamics) and episode length *falls* '
             'with competence, from the measured random-policy 497 towards ~150'),
}

# Per-run process overhead: interpreter start, TensorFlow import, network build.
# Negligible in one run and not negligible over a catalogue of ~1800.
DEFAULT_STARTUP_S = 6.0

# Episodes in the discarded warm-up run that precedes each --measure
# measurement, so graph tracing is not billed to the first pair measured.
WARMUP_EPISODES = 3

# Per env step, not per episode, and that distinction is the correction. The
# earlier 0.68 came from mean seconds per *episode* on the demo tree (0.299 mlp
# against 0.438 dueling), which conflates the rate with the episode length each
# architecture reaches: the mlp arms were cheaper per episode partly because
# their episodes were shorter. Per step on the P0 tree, stall-excluded, mlp costs
# 0.85 of dueling on LunarLander (5.2 against 6.1 ms) and 1.02 on CartPole (6.2
# against 6.1 ms, n=2 per cell). 0.85 is taken because LunarLander carries the
# catalogue's compute and its estimate rests on sixteen runs per architecture
# rather than two. A harvested table replaces this factor outright, per
# (environment, architecture); it survives only as the fallback.
ARCH_FACTOR = {'dueling': 1.00, 'mlp': 0.85}
# The double-Q target adds one extra forward pass through the online network per
# update, which the demo tree put at about 3%. The P0 tree cannot check it: the
# rule is divided out of each run before the fit, so it never becomes a fitted
# coefficient, and the vanilla and double runs there differ by more than the rule
# alone. Documented, and the harvested entries say so.
RULE_FACTOR = {'vanilla': 1.00, 'double': 1.03}

# ---------------------------------------------------------------------------
# Outlier gates for `--from-runs`. Two of them, in this order, because the
# failure they exist for is not a property of the arm being fitted.
#
# A run stalled by another process taking its claim (`PRELAUNCH_FIXES.md` B1)
# read 159 to 192 ms/step on the P0 tree against a whole-tree median of 6.1.
# Four such runs sat in groups of four to sixteen, so in the smallest of them a
# per-group centre is itself contaminated and cannot be the first line of
# defence. The whole-harvest gate runs first for that reason: it compares every
# run's implied step rate against the median over the entire harvest and drops
# anything beyond a factor either side. Environments and architectures do differ
# in step cost, but by 1.5x across the whole P0 tree and not by 25x, so a factor
# of 4 cannot reach a legitimate difference.
#
# The per-group gate then runs on the survivors, where a group is large enough
# for its own median to mean something. It requires both a robust z score (MAD
# based, so a single outlier cannot inflate the scale it is judged against) and a
# plain factor from the median, because at low dispersion a z score alone rejects
# a run that is 20% slow and nothing here should turn on 20%.
#
# Both gates are symmetric in form and, on a right-skewed tree, asymmetric in
# effect: contamination makes runs slower and never faster, so what they remove
# is the upper tail and the fitted rate moves down. That is a bias towards
# under-pricing, which is the dangerous direction for a compute plan, so the
# report prints each group's median before and after exclusion and the size of
# the shift is left in front of the reader rather than folded into a
# coefficient.
#
# Neither gate is a way to drop an inconvenient run: every exclusion is printed
# with its rate, its factor and which gate caught it, and the count reaches the
# JSON (`STANDING_INSTRUCTIONS` S6).
# ---------------------------------------------------------------------------
OUTLIER_GLOBAL_FACTOR = 4.0
OUTLIER_ROBUST_Z = 3.5
OUTLIER_MIN_FACTOR = 1.5
OUTLIER_MIN_GROUP = 5

# A variant is fitted on its own runs once it has this many; below it the runs
# join their base environment's pool and the entry says the rate is pooled. Four
# is the smallest group in which the per-group outlier gate and a rate fit both
# still mean something, and it is what the P0 interface arms supply.
MIN_VARIANT_RUNS = 4

# Verdict thresholds for the two checks the report performs on the model in
# force. FIT_* is the fit against the runs it was fitted from: a model that
# cannot reproduce its own training data is broken, and 25% is generous for a
# self-fit (the 1/1.35 convention error showed as a median ratio of 0.74).
# ANCHOR_TOLERANCE is the model against the documented band, where the model can
# legitimately win: when the coefficients were fitted from data and the anchor
# prose disagrees, it is the prose that is stale, and the check says which side
# it doubts rather than pretending the disagreement is not there.
FIT_CHECK_MEDIAN_TOL = 1.25
FIT_CHECK_TAIL_FACTOR = 2.0
FIT_CHECK_TAIL_FRACTION = 0.20
ANCHOR_TOLERANCE = 1.0

# A greedy evaluation episode costs less than a training episode: forward passes
# only, no backward pass and no replay sampling. ASSUMPTION, and the only place
# the measurement cadence enters the cost.
EVAL_EPISODE_WEIGHT = 0.35

# Per-run duration multiplier against the anchor condition. The anchors were
# measured at 2 threads per process with four workers resident, so jobs=4 *is*
# the measured condition and needs no correction; the other columns are
# ASSUMPTIONS about contention on an 8-core machine, where jobs=8 leaves one
# thread per worker and oversubscribes the memory system.
CONTENTION = {1: 0.75, 2: 0.85, 4: 1.00, 6: 1.20, 8: 1.45}
JOB_LADDER = (1, 2, 4, 6, 8)

# Order in which an ambiguous seed is attributed to a block. The blocks are
# meant to be disjoint (`DESIGN.md` §3.4) but nothing outside this ordering
# enforces it: SMOKE was `(0,)` until recently, which put every
# pipeline-validation run inside CONFIRM. So an overlap resolves by this
# precedence rather than by dictionary order.
#
# TUNE is first, and the direction matters more than the order does. The one
# thing this attribution feeds is the leak detector below, which fires when a
# TUNE seed reaches a reported experiment. With CONFIRM first, as it was, a seed
# in both blocks was attributed to CONFIRM and the detector could not see it:
# the guard failed open on precisely the overlap it exists to resolve. Ambiguity
# now resolves towards contamination, which is the safe direction for a detector,
# and `block_overlaps` reports the overlap itself so that this ordering is never
# the only thing standing between the design and a selection leak.
BLOCK_PRECEDENCE = ('TUNE', 'CONFIRM', 'REPLICATE', 'C4SRC', 'RESERVE', 'SMOKE')

# Families whose runs feed a reported estimate. A TUNE seed reaching one of these
# is the revision-1 selection leak (`DESIGN.md` §3.4).
REPORTED_FAMILIES = ('confirmatory', 'estimation')


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------
#: Every coefficient the cost model carries. `measured` is a statement about all
#: of them, not about the name of the file they were read from.
MODELLED_COEFFICIENTS = ('s_per_env_step', 's_per_episode', 'steps_start',
                         'steps_end', 'ramp_episodes', 'startup_s')


@dataclass
class CostModel:
    """Two coefficients and an episode-length ramp, per (env, arch, rule)."""

    key: str
    s_per_env_step: float
    s_per_episode: float
    steps_start: float
    steps_end: float
    ramp_episodes: int
    startup_s: float
    source: str                    # measured | harvested | documented | inferred
    note: str = ''
    length_scale: float = 1.0      # applied from this variant's random_len
    #: Which of MODELLED_COEFFICIENTS a calibration derived from data.
    fitted: tuple = ()

    @property
    def documented_coefficients(self) -> tuple:
        """The coefficients here that no calibration derived from data."""
        return tuple(c for c in MODELLED_COEFFICIENTS if c not in self.fitted)

    @property
    def measured(self) -> bool:
        """True only when every coefficient in this model came from data.

        This used to be `source in ('measured', 'harvested')`, which is how a
        harvest that refitted one coefficient of six suppressed the ESTIMATE
        stamp from every section of the report and set `throughput.measured` in
        the JSON, while the entry's own note admitted the other five were
        guesswork. The stamp follows the coefficients now.
        """
        return (self.source in ('measured', 'harvested')
                and not self.documented_coefficients)

    @property
    def partly_measured(self) -> bool:
        """Some coefficient was fitted, but not all of them."""
        return (self.source in ('measured', 'harvested')
                and bool(self.fitted) and not self.measured)

    def episode_steps(self, episode: int, cap: Optional[int] = None) -> float:
        if self.ramp_episodes <= 0:
            length = self.steps_end
        else:
            frac = min(1.0, max(0.0, episode / float(self.ramp_episodes)))
            length = self.steps_start + (self.steps_end - self.steps_start) * frac
        return min(length, float(cap)) if cap else length

    def total_env_steps(self, episodes: int,
                        cap: Optional[int] = None) -> float:
        """Closed-form sum of the length ramp, truncated at the step cap.

        The cap is kept because the ramp is fitted on one environment and then
        rescaled to its variants by their random-policy lengths, and nothing in
        that rescale knows about the training loop's episode limit. It bound at
        the 560-step plateau this file carried before the ramp was fitted from
        the P0 tree: `gravity=-4` modelled 1087 steps against a 1000-step limit,
        over-pricing the most expensive arm in the catalogue by about 9%. At the
        fitted plateau of 336 the same arm models 652 steps and the cap does not
        bind anywhere in the current catalogue, which is a fact about today's
        coefficients rather than a reason to drop the truncation.
        """
        if episodes <= 0:
            return 0.0
        if self.ramp_episodes <= 0:
            return episodes * self.episode_steps(0, cap)
        # The ramp is at most a few hundred episodes, so it is summed term by
        # term -- exact, and obviously so, which a closed form with a cap
        # inside it would not be. The flat tail is closed form.
        n_ramp = min(episodes, self.ramp_episodes)
        total = sum(self.episode_steps(e, cap) for e in range(n_ramp))
        return total + max(0, episodes - n_ramp) * self.episode_steps(
            self.ramp_episodes, cap)

    def seconds(self, cfg: Config) -> float:
        """Wall-clock for one run of `cfg`, at the anchor's contention level.

        The linear-probe jumpstart is charged separately and only to transfer
        arms: it collects `probe_transitions` transitions and takes
        `probe_steps` gradient steps before episode 0, which at the catalogue's
        defaults is about 5500 step-equivalents -- a few hours across the
        transfer arms, and the one systematic reason a transfer run costs more
        than its matched scratch run beyond the transfer itself.
        """
        load = measurement_load(cfg) / ANCHOR_LOAD
        steps = self.total_env_steps(cfg.num_episodes, cap=cfg.max_steps)
        probe = (self.s_per_env_step * (cfg.probe_transitions + cfg.probe_steps)
                 if cfg.is_transfer and cfg.probe_steps else 0.0)
        base = (self.s_per_episode * cfg.num_episodes
                + self.s_per_env_step * steps)
        return self.startup_s + load * base + probe

    def to_dict(self) -> dict:
        return {'s_per_env_step': self.s_per_env_step,
                's_per_episode': self.s_per_episode,
                'steps_start': self.steps_start, 'steps_end': self.steps_end,
                'ramp_episodes': self.ramp_episodes,
                'startup_s': self.startup_s, 'source': self.source,
                'length_scale': self.length_scale, 'note': self.note,
                'fitted': list(self.fitted),
                'documented': list(self.documented_coefficients),
                'measured': self.measured}


def measurement_load(cfg: Config) -> float:
    """Evaluation burden of one run, relative to its training episodes.

    The anchors are whole-run averages at the default cadence, and the default
    cadence is not cheap: 100 monitoring evaluations of 5 episodes, then 100
    held-out episodes at each of the three final checkpoints, 100 at the single
    prefix checkpoint (episode 500), and 100 for the zero-shot jumpstart. That is
    1000 evaluation episodes against 1000 training episodes, which at
    EVAL_EPISODE_WEIGHT gives ANCHOR_LOAD = 1.35. A run that changes
    `eval_every`, `final_eval_episodes` or `prefix_checkpoints` therefore changes
    its cost for reasons that have nothing to do with its episode budget, and
    E0's smoke configuration changes all three. This is the term that tracks
    that.

    The prose here read "three prefix checkpoints" and "1200 evaluation
    episodes" until `DESIGN.md` revision 4 cut the 250- and 750-episode
    prefixes. The code was always right, because it counts
    `cfg.prefix_checkpoints` rather than asserting how many there are; only the
    explanation had gone stale, which is its own kind of defect in a file whose
    comments are the specification a reader checks the arithmetic against.
    """
    monitoring = (cfg.num_episodes // max(1, cfg.eval_every)) * cfg.eval_episodes
    held_out = cfg.final_eval_episodes * (cfg.final_eval_checkpoints
                                          + len(cfg.prefix_checkpoints) + 1)
    return 1.0 + EVAL_EPISODE_WEIGHT * (monitoring + held_out) / max(
        1, cfg.num_episodes)


ANCHOR_LOAD = measurement_load(Config())


def measured_steps_start(env: str) -> Optional[float]:
    """Measured random-policy episode length for an environment variant.

    This is what prices the shift family without a measurement per variant:
    `reference_returns.json` already records that a random policy runs 94 steps
    at gravity -10 and 182 at gravity -4, so the gravity arms cost roughly twice
    what the base environment costs per episode early in training. Assuming
    otherwise would have been a factor-of-two error on 440 runs.
    """
    try:
        return float(envs.reference(env)['random_len'])
    except (KeyError, ValueError, TypeError):
        return None


#: Physically possible per-env-step costs for this trainer, in seconds. A step
#: is an environment transition plus a gradient update on a two-layer network:
#: on the P0 tree that is 5 to 6 ms, and no machine runs it in a microsecond.
#: The bound exists because 1e-06 is exactly what an all-NaN column once wrote
#: into the table, through `max(1e-6, nan)`, which returns the floor because
#: every comparison against NaN is False.
RATE_SANITY_S = (1e-5, 1.0)

#: What a table entry has to carry before anything is priced from it.
ENTRY_NUMERIC_FIELDS = ('s_per_env_step', 's_per_episode', 'steps_start',
                        'steps_end', 'ramp_episodes', 'startup_s')


def entry_problem(entry: object) -> Optional[str]:
    """Why a calibration entry cannot be priced from, or None if it can.

    The table on disk is an input like any other, and it is written by this same
    tool from whatever run tree it was pointed at. The all-NaN column that
    collapsed `s_per_env_step` to 1e-06 went out through `write_throughput` and
    came back in here, and the read side asserted nothing whatever: the report
    then priced the sweep at a thousandth of a millisecond per step, halved the
    projected wall clock, and stamped the result measured. Checking on the way in
    as well as on the way out costs six comparisons and closes that path even for
    a table this version of the file did not write.
    """
    if not isinstance(entry, dict):
        return f'entry is {type(entry).__name__}, not an object'
    for key in ENTRY_NUMERIC_FIELDS:
        if key not in entry:
            return f'{key} is missing'
        try:
            value = float(entry[key])
        except (TypeError, ValueError):
            return f'{key} is not a number ({entry[key]!r})'
        if not math.isfinite(value):
            return f'{key} is {entry[key]!r}, which is not finite'
        if value < 0:
            return f'{key} is negative ({value})'
    rate = float(entry['s_per_env_step'])
    lo, hi = RATE_SANITY_S
    if not lo <= rate <= hi:
        return (f's_per_env_step {rate:g} s is outside the physically possible '
                f'range {lo:g} to {hi:g} s for this trainer')
    for key in ('steps_start', 'steps_end'):
        if float(entry[key]) <= 0:
            return f'{key} is {entry[key]!r}, and an episode has a length'
    return None


def cost_model(env: str, arch: str, target_rule: str,
               table: dict[str, dict]) -> CostModel:
    """Resolve one (env, arch, rule) to a cost model, calibrated table first."""
    spec = envs.parse(env)
    env_id, canonical = spec.env_id, spec.canonical()

    entry = table.get(f'{canonical}|{arch}') or table.get(f'{env_id}|{arch}')
    rejected = entry_problem(entry) if entry is not None else None
    if entry is not None and rejected is None:
        model = CostModel(key=f'{canonical}|{arch}', fitted=tuple(
            entry.get('fitted') or ()), **{
            k: entry[k] for k in ('s_per_env_step', 's_per_episode',
                                  'steps_start', 'steps_end', 'ramp_episodes',
                                  'startup_s', 'source', 'note')})
    else:
        anchor = DEFAULT_ANCHORS.get(env_id)
        source = 'documented'
        if anchor is None:
            anchor = DEFAULT_ANCHORS['LunarLander-v3']
            source = 'inferred'
        elif 'INFERRED' in anchor.note:
            source = 'inferred'
        note = anchor.note
        if rejected is not None:
            note = (f'TABLE ENTRY REJECTED for {canonical}|{arch}: {rejected}. '
                    f'Falling back to the documented anchor. Re-run the '
                    f'calibration, and do not trust the file until you have. '
                    f'{note}')
        model = CostModel(
            key=f'{canonical}|{arch}',
            s_per_env_step=anchor.s_per_env_step * ARCH_FACTOR.get(arch, 1.0),
            s_per_episode=anchor.s_per_episode,
            steps_start=anchor.steps_start, steps_end=anchor.steps_end,
            ramp_episodes=anchor.ramp_episodes, startup_s=DEFAULT_STARTUP_S,
            source=source, note=note)

    # The variant's own measured random-policy length always wins over the
    # anchor's, and the developed-policy length is scaled by the same ratio: a
    # variant whose untrained episodes are twice as long does not become
    # identical to the base environment once the policy improves.
    start = measured_steps_start(canonical)
    if start is not None and model.steps_start > 0:
        ratio = start / model.steps_start
        # 0.5% rather than exact equality: the base environment's measured
        # random_len is the anchor's own value to rounding, and annotating that
        # as a rescale would bury the variants that really are rescaled.
        if abs(ratio - 1.0) > 0.005:
            model.steps_end *= ratio
            model.length_scale = ratio
        model.steps_start = start

    model.s_per_env_step *= RULE_FACTOR.get(target_rule, 1.0)
    return model


def load_throughput(path: str = THROUGHPUT_FILE) -> tuple[dict, dict]:
    """Read the calibration table; return (entries, metadata)."""
    try:
        with open(path, encoding='utf-8') as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}, {}
    if data.get('schema') != THROUGHPUT_SCHEMA:
        return {}, {'error': f'unrecognised schema {data.get("schema")!r} in '
                             f'{path}; ignoring it and using documented anchors'}
    return data.get('entries', {}), {k: v for k, v in data.items()
                                     if k != 'entries'}


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------
@dataclass
class RunRecord:
    key: str
    run_dir: str
    label: str
    role: str
    experiments: tuple[str, ...]
    families: tuple[str, ...]
    condition: str
    cell: str
    env: str
    arch: str
    target_rule: str
    seed: int
    seed_block: str
    episodes: int
    depends_on: Optional[str]
    seconds: float
    complete: bool
    # Which source this run draws on and whether the DESIGN.md 4.3 validity gate
    # moved it there, carried per run because the run directory deliberately
    # cannot say it: `source_checkpoint` is excluded from the digest.
    source_arm: Optional[str] = None
    source_default_seed: Optional[int] = None
    source_seed: Optional[int] = None
    source_replaced: bool = False
    #: From the manifest of a run already on disk: the gate's verdict on the
    #: source it actually used. None where there is no source to judge.
    source_valid: Optional[bool] = None
    source_score: Optional[float] = None
    #: Why this run is not counted complete, when a manifest says something.
    problem: Optional[str] = None

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d['experiments'] = list(self.experiments)
        d['families'] = list(self.families)
        return d


@dataclass
class ExperimentRow:
    id: str
    name: str
    tier: int
    family: str
    arms: int
    seed_block: str
    seeds: tuple[int, ...]
    runs_alone: int              # runs if this experiment were launched alone
    runs_new: int                # runs it adds to the selection, in order
    seconds_alone: float
    question: str
    review_refs: tuple[str, ...]
    varies: tuple[str, ...]

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d['seeds'] = list(self.seeds)
        d['review_refs'] = list(self.review_refs)
        d['varies'] = list(self.varies)
        return d


def block_names() -> tuple:
    """Every declared block, precedence order first, then anything new.

    Iterating BLOCK_PRECEDENCE alone would make a block added to the registry
    but not to that tuple invisible here: its seeds would read UNKNOWN and no
    check would ever look at them. The tail is what keeps this file from failing
    silently when the registry grows.
    """
    listed = tuple(n for n in BLOCK_PRECEDENCE if n in reg.SEED_BLOCKS)
    return listed + tuple(n for n in reg.SEED_BLOCKS if n not in listed)


def blocks_containing(seed: int) -> tuple:
    """Every declared block that contains this seed, precedence order."""
    return tuple(name for name in block_names()
                 if seed in reg.SEED_BLOCKS.get(name, ()))


def seed_block_of(seed: int, declared: Optional[str] = None) -> str:
    """Which block a seed belongs to; ambiguity resolves towards TUNE.

    The blocks are meant to be disjoint, and when they are this is a lookup.
    When they are not (SMOKE was declared as `(0,)` until recently, inside
    CONFIRM's range) the attribution has to choose, and the choice is not
    cosmetic: the only thing it feeds is the leak detector in `build_inventory`,
    which fires when a TUNE seed reaches a reported experiment.

    So an ambiguous seed is attributed to TUNE whenever TUNE is one of its
    blocks, ahead of the block the experiment itself declares. That is the
    opposite of what this function used to do, and the old order made the guard
    useless in exactly the case it exists for: a seed in both CONFIRM and TUNE
    was reported as CONFIRM, by an experiment that had declared CONFIRM, and the
    leak went unseen. A detector that resolves ambiguity in its own favour is not
    a detector. The declared block still wins over the rest, because that is what
    tells a SMOKE run from a CONFIRM run at the same seed.
    """
    names = blocks_containing(seed)
    if not names:
        return 'UNKNOWN'
    if len(names) == 1:
        return names[0]
    if 'TUNE' in names:
        return 'TUNE'
    if declared and declared in names:
        return declared
    return names[0]


def block_overlaps() -> list:
    """Pairs of declared blocks that share a seed, which §3.4 forbids.

    Reported rather than assumed away. `seed_block_of` can only pick a side; it
    cannot tell anyone that a side had to be picked, and an overlap between TUNE
    and a confirmatory block is a design defect that the selection leak of
    revision 1 is the historical example of.
    """
    names = block_names()
    out: list[str] = []
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            shared = sorted(set(reg.SEED_BLOCKS.get(first, ()))
                            & set(reg.SEED_BLOCKS.get(second, ())))
            if shared:
                out.append(f'{first} and {second} share seed(s) '
                           f'{", ".join(str(s) for s in shared)}')
    return out


def block_label(seeds: Sequence[int]) -> str:
    """Name a seed set: a block name when it is one, otherwise a range."""
    if not seeds:
        return 'EMPTY'
    # Precedence order, not dictionary order: a one-seed block whose value falls
    # inside another block would otherwise capture every single-seed selection,
    # whatever experiment asked for it.
    for name in BLOCK_PRECEDENCE:
        if tuple(seeds) == tuple(reg.SEED_BLOCKS.get(name, ())):
            return name
    blocks = sorted({seed_block_of(s) for s in seeds})
    return f'EXPLICIT[{min(seeds)}-{max(seeds)}]:{"+".join(blocks)}'


@dataclass(frozen=True)
class RunStatus:
    """What a run directory says about itself."""

    complete: bool
    source_valid: Optional[bool] = None
    source_score: Optional[float] = None
    problem: Optional[str] = None


def run_status(run_dir: str, episodes: int) -> RunStatus:
    """Whether a run is done, and whether what it stands on is usable.

    Every branch here is a manifest shape that either took the planner down or
    that it believed when it should not have.

    * A manifest whose top level is not an object, or whose `result` is a list,
      raised `AttributeError` out of `.get` and aborted the entire plan before a
      line was printed. A half-written manifest from a killed run is the same
      class of event as the overnight stall of `PRELAUNCH_FIXES.md` B1, so it has
      to be survivable rather than fatal.
    * `episodes_completed: "lots"` raised `ValueError` from `int()`.
    * A run whose metrics stream is not contiguous is not complete however many
      episodes it claims: the resume duplication of `DESIGN.md` 8.2 writes a full
      episode count over a corrupted stream. The trainer already computes that
      verdict into `result.metrics_integrity`, so this reads it rather than
      reimplementing it, and this file still opens no metrics stream of its own.
    * `source.validity.valid is False` means the run stands on a source the
      `DESIGN.md` 4.3 gate rejected. It is not evidence, it will be re-run
      against a `RESERVE` source, and counting it complete is how a plan reports
      16 of 16 done on a tree where two of them have to be thrown away. `None`
      is not `False`: the C0 untrained-source arms carry `None` by design, and
      conflating the two would condemn eight sound runs.

    A damaged manifest returns `complete=False` with the reason attached rather
    than a bare False, because a run that cannot be read is work the planner is
    about to schedule and the operator is entitled to know why.
    """
    path = os.path.join(run_dir, 'manifest.json')
    try:
        with open(path, encoding='utf-8') as fh:
            man = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return RunStatus(complete=False)
    if not isinstance(man, dict):
        return RunStatus(complete=False,
                         problem=f'manifest is a {type(man).__name__}, not an '
                                 f'object')
    result = man.get('result')
    if result is None:
        return RunStatus(complete=False)
    if not isinstance(result, dict):
        return RunStatus(complete=False,
                         problem=f'result is a {type(result).__name__}, not an '
                                 f'object')
    try:
        done = int(result.get('episodes_completed') or 0)
    except (TypeError, ValueError):
        return RunStatus(complete=False,
                         problem=f'episodes_completed is '
                                 f'{result.get("episodes_completed")!r}, which '
                                 f'is not a number of episodes')
    complete = done >= episodes
    problem = None

    integrity = result.get('metrics_integrity')
    if complete and isinstance(integrity, dict):
        rows, unique = integrity.get('rows'), integrity.get('unique_episodes')
        if integrity.get('contiguous') is False or (
                isinstance(rows, int) and isinstance(unique, int)
                and unique < rows):
            complete = False
            problem = (f'{done} episodes recorded but the metrics stream is not '
                       f'contiguous ({unique} unique of {rows} rows): '
                       f'DESIGN.md 8.2 duplication, so the run has to be redone')

    source = man.get('source')
    valid = score = None
    if isinstance(source, dict) and isinstance(source.get('validity'), dict):
        validity = source['validity']
        valid = validity.get('valid')
        try:
            score = (None if validity.get('source_final_score') is None
                     else float(validity['source_final_score']))
        except (TypeError, ValueError):
            score = None
        if valid is False and complete:
            complete = False
            problem = ('the source it transferred from failed the DESIGN.md '
                       '4.3 validity gate'
                       + (f' (normalised final score {score:.3f} against a gate '
                          f'of {validity.get("gate")})' if score is not None
                          else '')
                       + ': the source is redrawn from RESERVE and this run is '
                         'redone, so it is pending work, not finished work')
    return RunStatus(complete=complete, source_valid=valid, source_score=score,
                     problem=problem)


@dataclass
class Inventory:
    experiments: list[ExperimentRow]
    runs: list[RunRecord]
    seeds_spec: Optional[str]
    out_root: str
    table: dict
    table_meta: dict
    models: dict[str, CostModel]
    warnings: list[str] = field(default_factory=list)
    #: (source arm, default seed) -> seed in force, from the runner's ledger.
    replacements: dict = field(default_factory=dict)
    #: Every RESERVE seed the ledger records as drawn, per source arm.
    replacement_draws: dict = field(default_factory=dict)
    #: True when a rejection ledger was found under `out_root`.
    ledger_present: bool = False
    #: The `tuning.Selection` the tuned arms of DESIGN.md 3.3 were enumerated
    #: from, or None where the tree holds none and the stage was not priced.
    selection: object = None

    @property
    def naive_total(self) -> int:
        return sum(r.runs_alone for r in self.experiments)

    @property
    def tuned_experiments(self) -> list[str]:
        """The tuned ids in this selection, in the order they were priced."""
        return [r.id for r in self.experiments if r.id in reg.TUNED_OF]

    @property
    def tuned_only_runs(self) -> list[RunRecord]:
        """Runs that exist only because the tuned stage is in this selection.

        The sharing property of `DESIGN.md` 3.3 is a de-duplication *across*
        experiments: a cell whose selection equals the a priori configuration
        produces run digests identical to its common-policy arms, so one run
        record ends up claimed by both policies. Counting the tuned stage by
        summing its experiments' run counts would therefore price those shared
        cells twice; counting the records no common-policy arm also claims
        prices what the stage actually adds.

        It is a property of the *selection being priced*, not of the catalogue:
        pricing `E1t` without `E1` makes every one of its runs tuned-only, which
        is the honest answer to what that command would launch, and the report
        says which of the two questions the number answers.
        """
        return [r for r in self.runs
                if r.experiments
                and all(e in reg.TUNED_OF for e in r.experiments)]

    @property
    def total(self) -> int:
        return len(self.runs)

    @property
    def pending(self) -> list[RunRecord]:
        return [r for r in self.runs if not r.complete]

    @property
    def measured(self) -> bool:
        """Every coefficient of every model in force came from data."""
        return bool(self.models) and all(m.measured for m in self.models.values())

    @property
    def partly_measured(self) -> bool:
        """Some coefficient was fitted, but the model is not fully measured."""
        return (not self.measured
                and any(m.partly_measured or m.measured
                        for m in self.models.values()))

    @property
    def documented_coefficients(self) -> list:
        """Coefficients no calibration supplied, across the models in force."""
        out: set = set()
        for model in self.models.values():
            out |= set(model.documented_coefficients)
        return sorted(out)

    def stamp(self) -> str:
        """The provenance banner, in the three states the coefficients allow.

        A harvested table used to print as `measured` on the strength of one
        refitted coefficient out of six. It now says which it is, and when it is
        in between it says what is missing, because the difference between
        `measured` and `partly measured` is the difference between a number a
        reviewer can check and a number they have to take on trust.
        """
        if self.measured:
            return ''
        if self.partly_measured:
            return ('   *** PARTLY MEASURED: '
                    + ', '.join(self.documented_coefficients)
                    + ' documented ***')
        return '   *** ESTIMATE, NOT MEASURED ***'

    @property
    def rejected_sources(self) -> list:
        """Runs on disk whose source failed the DESIGN.md 4.3 validity gate."""
        return [r for r in self.runs if r.source_valid is False]

    def by(self, attr: str) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for r in self.runs:
            slot = out.setdefault(str(getattr(r, attr)),
                                  {'runs': 0, 'pending': 0, 'seconds': 0.0})
            slot['runs'] += 1
            slot['seconds'] += r.seconds
            slot['pending'] += 0 if r.complete else 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]['runs']))


def build_inventory(exp_ids: Sequence[str], seeds: Optional[str],
                    out_root: str, overrides: Optional[dict],
                    table: dict[str, dict], table_meta: dict,
                    allow_factor_overrides: bool = False,
                    selection=None) -> Inventory:
    """Resolve a selection into runs, costs, seed blocks and sharing savings.

    The job graph is resolved through the runner's source-replacement ledger, not
    through the catalogue as written. `DESIGN.md` 4.3 lets a source that fails
    the validity gate be replaced from `RESERVE`. `load_source_replacements` in
    the registry exists so that a caller which knows only a run tree can
    reproduce the graph that was actually run, and this module was enumerating
    the no-rejection catalogue instead. In P0 that meant reporting 16 of 16 E1
    runs complete on a tree where two of them stand on a source that scored
    0.599 against a 0.600 gate, with no warning and nothing priced for the
    redraw.
    """
    warnings: list[str] = []
    if table_meta.get('error'):
        warnings.append(table_meta['error'])
    if not os.path.isdir(out_root):
        warnings.append(
            f'--out-root {out_root} does not exist, so every run below counts as '
            f'pending. If that path is a typo the plan is right about the cost '
            f'and wrong about the work already done.')

    overlaps = block_overlaps()
    if overlaps:
        warnings.append(
            'SEED BLOCKS OVERLAP, which DESIGN.md 3.4 forbids: '
            + '; '.join(overlaps)
            + '. Attribution below resolves ambiguity towards TUNE so the leak '
              'check can see it, but the blocks themselves need fixing in '
              'registry.py.')

    replacements, draws = reg.load_source_replacements(out_root)
    ledger_present = os.path.exists(
        os.path.join(out_root, reg.REPLACEMENTS_RELPATH))
    if replacements:
        warnings.append(
            f'{len(replacements)} source lineage(s) were replaced from RESERVE '
            f'by the runner (DESIGN.md 4.3). This plan follows the ledger at '
            f'{os.path.join(out_root, reg.REPLACEMENTS_RELPATH)}, so the run '
            f'directories below are the ones the runner would use, not the '
            f'ones the catalogue would produce with no rejections in it.')

    models: dict[str, CostModel] = {}

    def model_for(cfg: Config) -> CostModel:
        key = f'{cfg.env}|{cfg.arch}|{cfg.target_rule}'
        if key not in models:
            models[key] = cost_model(cfg.env, cfg.arch, cfg.target_rule, table)
        return models[key]

    rows: list[ExperimentRow] = []
    records: dict[str, RunRecord] = {}
    for eid in exp_ids:
        # `resolve_experiment` rather than `EXPERIMENTS[eid]`, and the selection
        # passed rather than re-read: a tuned arm's configuration, and therefore
        # its run digest, is a function of the selection, so an inventory that
        # let each call re-resolve it could price one selection's arms against
        # another's directories if the artifact moved underneath it.
        exp = reg.resolve_experiment(eid, out_root, selection)
        # `jobs` can emit two arms that resolve to one configuration -- E7's
        # aggregation='mean' scratch arm is E1's scratch arm -- so the count that
        # matters is over run keys, not over jobs.
        alone = {job.key(): job
                 for job in reg.jobs(eid, seeds, out_root, overrides,
                                     allow_factor_overrides,
                                     source_seeds=replacements,
                                     selection=selection)}
        before = len(records)
        seconds_alone = 0.0
        for key, job in alone.items():
            cfg = job.cfg
            seconds = model_for(cfg).seconds(cfg)
            seconds_alone += seconds
            rec = records.get(key)
            if rec is None:
                status = run_status(cfg.run_dir(), cfg.num_episodes)
                records[key] = RunRecord(
                    key=key, run_dir=cfg.run_dir(), label=cfg.label,
                    role=job.role, experiments=(eid,), families=(exp.family,),
                    condition=cfg.condition,
                    cell=f'{cfg.arch}-{cfg.target_rule}', env=cfg.env,
                    arch=cfg.arch, target_rule=cfg.target_rule, seed=cfg.seed,
                    seed_block=seed_block_of(cfg.seed, exp.seed_block),
                    episodes=cfg.num_episodes, depends_on=job.depends_on,
                    seconds=seconds, complete=status.complete,
                    source_arm=job.source_arm,
                    source_default_seed=job.source_default_seed,
                    source_seed=job.source_seed,
                    source_replaced=job.source_replaced,
                    source_valid=status.source_valid,
                    source_score=status.source_score,
                    problem=status.problem)
            else:
                if eid not in rec.experiments:
                    rec.experiments += (eid,)
                if exp.family not in rec.families:
                    rec.families += (exp.family,)
        rows.append(ExperimentRow(
            id=exp.id, name=exp.name, tier=exp.tier, family=exp.family,
            arms=len(exp.arms),
            seed_block=block_label(reg.resolve_seeds(seeds, exp.seed_block)),
            seeds=tuple(reg.resolve_seeds(seeds, exp.seed_block)),
            runs_alone=len(alone), runs_new=len(records) - before,
            seconds_alone=seconds_alone, question=exp.question,
            review_refs=exp.review_refs, varies=exp.varies))

    runs = list(records.values())

    # Cross-check against the registry's own de-duplication. This module walks
    # the experiments one at a time so it can report what each adds and which
    # runs several of them share -- information `all_jobs` discards -- and the
    # two must agree on the set of runs, or the plan is pricing something the
    # launcher will not run.
    canonical = {job.key() for job in reg.all_jobs(exp_ids, seeds, out_root,
                                                   overrides,
                                                   allow_factor_overrides,
                                                   source_seeds=replacements,
                                                   selection=selection)}
    if canonical != set(records):
        warnings.append(
            f'INTERNAL: this inventory and registry.all_jobs disagree on '
            f'{len(canonical ^ set(records))} run(s). The launcher follows '
            f'all_jobs, so treat the costs here as unreliable and report it.')

    # The selection leak revision 1 shipped, checked before the compute is spent.
    leaked = [r for r in runs if r.seed_block == 'TUNE'
              and any(f in REPORTED_FAMILIES for f in r.families)]
    if leaked:
        exps = sorted({e for r in leaked for e in r.experiments})
        warnings.append(
            f'TUNE SEEDS REACH REPORTED WORK: {len(leaked)} run(s) on TUNE '
            f'seeds are claimed by {", ".join(exps)}, whose families include a '
            f'reported one. DESIGN.md 3.4 forbids a reported estimate drawing '
            f'on TUNE and audit.py will refuse it. Re-select, or pass '
            f'--allow-tune-mixing if these runs are for selection only.')
    tune = [r for r in runs if r.seed_block == 'TUNE']
    if tune and not leaked:
        warnings.append(
            f'{len(tune)} run(s) on TUNE seeds, all inside screen-family '
            f'experiments. Legitimate for hyperparameter selection; no reported '
            f'estimate may draw on them.')

    unknown = [r for r in runs if r.seed_block == 'UNKNOWN']
    if unknown:
        reported = [r for r in unknown
                    if any(f in REPORTED_FAMILIES for f in r.families)]
        seeds_seen = sorted({r.seed for r in unknown})
        warnings.append(
            f'{len(unknown)} run(s) are on seed(s) '
            f'{", ".join(str(s) for s in seeds_seen)}, which belong to no '
            f'declared block. '
            + (f'{len(reported)} of them are claimed by a reported experiment, '
               f'and DESIGN.md 3.4 audits by block: a seed outside every block '
               f'cannot be audited for selection leakage at all.'
               if reported else
               'None reaches a reported experiment, but audit.py works by block '
               'and will not recognise these runs either.'))

    # A run on disk whose source the gate rejected. `run_status` has already
    # taken it out of the completed count; this is where it is named, because
    # DESIGN.md 4.3 requires the number and identity of rejected source seeds to
    # be reported and a count on its own is not an identity.
    rejected = [r for r in runs if r.source_valid is False]
    if rejected:
        by_source: dict[str, list[RunRecord]] = {}
        for rec in rejected:
            arm = rec.source_arm or 'unknown source'
            by_source.setdefault(arm, []).append(rec)
        detail = '; '.join(
            f'{arm} seed '
            f'{", ".join(str(r.source_default_seed) for r in members[:1])} '
            f'(score {members[0].source_score:.3f}) -> '
            f'{len(members)} dependent run(s)'
            for arm, members in sorted(by_source.items())
            if members[0].source_score is not None)
        warnings.append(
            f'SOURCE VALIDITY: {len(rejected)} run(s) already on disk stand on a '
            f'source that failed the DESIGN.md 4.3 gate'
            + (f' [{detail}]' if detail else '')
            + '. They are counted as pending, not complete: the source is '
              'redrawn from RESERVE and they are re-run. '
            + ('The runner has recorded no replacement for them yet, so the '
               'RESERVE runs they need are not in the counts below either.'
               if not replacements else
               'The replacements in the ledger are priced above.'))

    damaged = [r for r in runs if r.problem and r.source_valid is not False]
    if damaged:
        warnings.append(
            f'{len(damaged)} run(s) on disk could not be counted complete: '
            + '; '.join(f'{r.label} seed {r.seed}: {r.problem}'
                        for r in damaged[:5])
            + (f'; and {len(damaged) - 5} more' if len(damaged) > 5 else ''))

    inferred = [k for k, m in models.items() if m.source == 'inferred']
    if inferred:
        warnings.append('no throughput anchor for ' + ', '.join(sorted(inferred))
                        + ': cost inferred from another environment')

    unusable = sorted({m.key for m in models.values()
                       if m.note.startswith('TABLE ENTRY REJECTED')})
    if unusable:
        warnings.append(
            f'{len(unusable)} calibration entr(y/ies) were unusable and the '
            f'documented anchor was priced instead: {", ".join(unusable)}. The '
            f'reason is printed with the cost model. Re-run the calibration.')

    return Inventory(experiments=rows, runs=runs, seeds_spec=seeds,
                     out_root=out_root, table=table, table_meta=table_meta,
                     models=models, warnings=warnings,
                     replacements=replacements, replacement_draws=draws,
                     ledger_present=ledger_present, selection=selection)


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------
def schedule(runs: Sequence[RunRecord], workers: int,
             multiplier: float = 1.0) -> dict:
    """Greedy list schedule in dependency order, plus the critical-path floor.

    Dependency order is what the launcher actually emits (`registry.jobs` yields
    a source before its consumers), so this is a schedule that could really be
    run rather than an optimal one. The critical path is reported alongside
    because it is a floor no number of workers can beat: a transfer run cannot
    start until its source finishes, and workers past that point buy nothing.
    """
    workers = max(1, int(workers))
    free = [0.0] * workers
    finish: dict[str, float] = {}
    chain: dict[str, float] = {}
    total = 0.0
    external = 0

    for rec in runs:
        dur = rec.seconds * multiplier
        total += dur
        dep_done = dep_chain = 0.0
        if rec.depends_on:
            if rec.depends_on in finish:
                dep_done, dep_chain = finish[rec.depends_on], chain[rec.depends_on]
            else:
                external += 1          # prerequisite outside this selection
        idx = min(range(workers), key=lambda i: max(free[i], dep_done))
        free[idx] = max(free[idx], dep_done) + dur
        finish[rec.key] = free[idx]
        chain[rec.key] = dep_chain + dur

    makespan = max(free) if free else 0.0
    return {
        'workers': workers,
        'multiplier': multiplier,
        'serial_seconds': total,
        'makespan_seconds': makespan,
        'critical_path_seconds': max(chain.values()) if chain else 0.0,
        'busy_fraction': (total / (makespan * workers)) if makespan else 0.0,
        'external_prerequisites': external,
    }


def contention(workers: int) -> float:
    """Per-run slowdown at `workers` resident processes, against the anchor."""
    known = sorted(CONTENTION)
    if workers in CONTENTION:
        return CONTENTION[workers]
    if workers < known[0]:
        return CONTENTION[known[0]]
    if workers > known[-1]:
        lo, hi = known[-2], known[-1]
        slope = (CONTENTION[hi] - CONTENTION[lo]) / (hi - lo)
        return CONTENTION[hi] + slope * (workers - hi)
    lo = max(k for k in known if k <= workers)
    hi = min(k for k in known if k >= workers)
    w = (workers - lo) / (hi - lo)
    return CONTENTION[lo] * (1 - w) + CONTENTION[hi] * w


def projections(runs: Sequence[RunRecord],
                ladder: Sequence[int] = JOB_LADDER) -> list[dict]:
    return [schedule(runs, w, contention(w)) for w in ladder]


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------
def measure_startup(timeout: float = 300.0) -> Optional[float]:
    """Interpreter start plus TensorFlow import, in a fresh process.

    Measured rather than assumed because it is charged once per run and the
    catalogue has of the order of 1800 of them: at 6 s that is three
    worker-hours of pure import.
    """
    import subprocess
    t0 = time.time()
    try:
        proc = subprocess.run([sys.executable, '-c', 'import src.dqn.train'],
                              cwd=REPO, capture_output=True, timeout=timeout,
                              env=dict(os.environ, TF_CPP_MIN_LOG_LEVEL='3'))
    except Exception:                                        # noqa: BLE001
        return None
    return round(time.time() - t0, 2) if proc.returncode == 0 else None


def measure(keys: Sequence[tuple[str, str]], episodes: int, out_root: str,
            seed: int) -> dict:
    """Run a short real run per (env, arch) and derive the step rate.

    What is measured and what is not, stated plainly. Measured: the per-env-step
    rate on this machine, and the per-run process overhead. Not measured: the
    episode-length ramp, because it only appears once a policy improves and a
    short run cannot see it -- that stays documented, and the resulting entry
    says so in its own note. This is the honest version of "measured
    throughput": the rate is real, the shape is inherited.
    """
    from src.dqn.train import train                          # local: imports TF

    def calibration_cfg(env: str, arch: str, n: int) -> Config:
        return Config(
            experiment='calibration', label=f'calib-{arch}', arch=arch,
            target_rule='vanilla', condition='scratch', env=env, seed=seed,
            num_episodes=n, eval_every=max(1, n // 2), eval_episodes=2,
            final_eval_episodes=5, final_eval_checkpoints=1,
            prefix_checkpoints=(), diag_states=64, probe_steps=0,
            probe_transitions=0, out_root=out_root,
            # The catalogue's `learning_starts=1000` is a rounding error against
            # half a million env steps and the whole measurement against a few
            # thousand: left at its default, a 20-episode CartPole calibration
            # takes no gradient step at all and prices the environment rather
            # than the trainer. Measured: 0.18 ms/step against a real 1.6 ms.
            learning_starts=100, checkpoint_seconds=10 ** 9)

    startup = measure_startup()
    entries: dict[str, dict] = {}
    for env, arch in keys:
        anchor = DEFAULT_ANCHORS.get(envs.parse(env).env_id,
                                     DEFAULT_ANCHORS['LunarLander-v3'])
        cfg = calibration_cfg(env, arch, episodes)
        # A discarded warm-up run first, per pair. TensorFlow traces its graph on
        # the first call for each input shape and architecture, and that cost
        # lands entirely on whichever pair happens to be measured first:
        # measured here, the first pair came out at 8.4 ms/step against 1.4 for
        # the second, a six-fold error that would have been invisible in the
        # table. The warm-up is charged to nobody.
        print(f'  warm-up (discarded) {env} / {arch} ...', flush=True)
        train(calibration_cfg(env, arch, WARMUP_EPISODES))
        print(f'  measuring {env} / {arch}: {episodes} episodes ...', flush=True)
        t0 = time.time()
        manifest = train(cfg)
        result = manifest['result']
        wall = float(result['wall_time_s'])
        steps = float(result['env_steps'])
        done = int(result['episodes_completed'])
        # One measurement cannot separate two coefficients, so the per-episode
        # overhead stays at its documented value and the step rate absorbs the
        # residual. Stated in the entry's note rather than left implicit.
        #
        # Both terms are in the anchors' own convention. `wall` is deflated to
        # the anchor's evaluation cadence, not to unit cadence, because
        # `anchor.s_per_episode` is quoted at the anchor cadence and
        # `CostModel.seconds` rescales from there. The line this replaces mixed
        # the two inside one expression: it subtracted an anchor-convention
        # overhead from a unit-convention wall time and stored a coefficient of
        # neither, deflated by 1/ANCHOR_LOAD.
        load = measurement_load(cfg) / ANCHOR_LOAD
        residual = wall / load - anchor.s_per_episode * done
        rate = residual / steps if steps > 0 else float('nan')
        updates = int(result.get('updates') or 0)
        source, detail = 'measured', ''
        if not math.isfinite(rate) or rate <= 0:
            # No floor. `max(1e-6, ...)` here would write a thousandth of a
            # millisecond per step and label it measured, which is what the
            # harvest path did on a NaN column: the report then priced a sweep at
            # a microsecond a step and suppressed the estimate banner while doing
            # it. A measurement that came out impossible is a failed
            # measurement.
            source = 'documented'
            detail = (f'MEASUREMENT REJECTED: the run left no positive step '
                      f'rate ({wall:.1f} s wall, {steps:.0f} env steps, '
                      f'{done} episodes, documented overhead '
                      f'{anchor.s_per_episode:.3f} s/episode); the documented '
                      f'rate is kept. ')
            print(f'    no positive step rate from {wall:.1f} s over '
                  f'{steps:.0f} steps -- keeping the documented anchor',
                  flush=True)
            rate = anchor.s_per_env_step * ARCH_FACTOR.get(arch, 1.0)
        elif updates < 0.5 * steps:
            source = 'documented'
            detail = (f'MEASUREMENT REJECTED: only {updates} gradient updates '
                      f'in {steps:.0f} env steps, so the run priced the '
                      f'environment rather than the trainer; the documented '
                      f'rate is kept. Raise --measure-episodes. ')
            # Refuse to record a rate that did not measure the trainer. A run
            # that spent most of its steps filling the buffer prices the
            # environment, and the sweep's cost is nearly all gradient steps.
            print(f'    only {updates} gradient updates in {steps:.0f} steps -- '
                  f'not a usable rate; keeping the documented anchor', flush=True)
            rate = anchor.s_per_env_step * ARCH_FACTOR.get(arch, 1.0)
        fitted = ['s_per_env_step'] if source == 'measured' else []
        if measured_steps_start(env) is not None:
            fitted.append('steps_start')
        if startup is not None:
            fitted.append('startup_s')
        entries[f'{envs.parse(env).canonical()}|{arch}'] = {
            's_per_env_step': round(rate, 6),
            's_per_episode': anchor.s_per_episode,
            'steps_start': measured_steps_start(env) or anchor.steps_start,
            'steps_end': anchor.steps_end,
            'ramp_episodes': anchor.ramp_episodes,
            'startup_s': startup if startup is not None else DEFAULT_STARTUP_S,
            'source': source, 'fitted': sorted(fitted),
            'note': f'{detail}step rate over {done} episodes / {steps:.0f} env '
                    f'steps / {updates} updates in {wall:.1f} s '
                    f'({time.time() - t0:.0f} s including build); per-episode '
                    f'overhead and length ramp remain documented',
        }
        print(f'    {wall:.1f} s, {steps:.0f} env steps -> '
              f'{rate * 1000:.3f} ms/step', flush=True)
    return entries


def discover_manifests(root: str) -> list[str]:
    """Every run manifest under a tree, at whatever depth it sits.

    This was a glob of `<root>/*/*/s*/manifest.json`: three levels exactly, which
    is what a sweep writes and nothing else is. `--from-runs runs/scratch` then
    reported "no usable runs" on a directory holding twenty of them, and
    `--from-runs <one run directory>` could not be expressed at all, so the
    obvious way to inspect a fit on part of a tree, or to reproduce a single
    run's self-fit, did not exist. A walk finds a manifest wherever it is.

    The sweep's own bookkeeping directories are skipped by name: `_jobs`,
    `_index` and `_logs` hold ledgers rather than runs, and a directory that
    holds a manifest is a run and cannot hold another one inside it.
    """
    if os.path.isfile(os.path.join(root, 'manifest.json')):
        return [os.path.join(root, 'manifest.json')]
    out: list[str] = []
    for base, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs
                         if not d.startswith('_') and not d.startswith('.'))
        if 'manifest.json' in files:
            out.append(os.path.join(base, 'manifest.json'))
            dirs[:] = []
    return sorted(out)


def config_from_manifest(cfg_d: dict, allow_legacy: bool
                         ) -> tuple[Optional[Config], Optional[str]]:
    """Rebuild a run's Config from its manifest, or say why it cannot be.

    A bare `except Exception: continue` sat here, and it swallowed the entire
    `runs_demo` tree: every one of those 198 manifests carries the
    pre-revision-5 field `epsilon_anneal_steps`, `Config(**cfg_d)` raises
    `TypeError`, and the tool printed one line about there being no usable runs.
    Worse on a mixed tree, where the surviving majority fitted happily and the
    legacy minority vanished with no message at all, because the "no usable runs"
    line only fires when nothing at all survives.

    So the reason comes back with the failure, and the caller reports it. The
    fields are not quietly dropped by default: a manifest whose schema this file
    does not know is a manifest whose cost model this file cannot vouch for, and
    `--allow-legacy-configs` is how an operator says they accept that risk for an
    older tree. When they do, the dropped fields are named in the report.
    """
    if not isinstance(cfg_d, dict):
        return None, f'config is a {type(cfg_d).__name__}, not an object'
    try:
        return Config(**cfg_d), None
    except Exception as exc:                                 # noqa: BLE001
        text = str(exc).strip()
        first = text.splitlines()[0] if text else repr(exc)
    known = {f.name for f in dataclasses.fields(Config)}
    unknown = sorted(set(cfg_d) - known)
    if not allow_legacy or not unknown:
        detail = (f'; unknown field(s) {", ".join(unknown)}, which '
                  f'--allow-legacy-configs would drop' if unknown else '')
        return None, f'config rejected by Config: {first}{detail}'
    try:
        cfg = Config(**{k: v for k, v in cfg_d.items() if k in known})
    except Exception as exc:                                 # noqa: BLE001
        return None, f'config rejected by Config even without {unknown}: {exc}'
    return cfg, f'LEGACY: dropped unknown field(s) {", ".join(unknown)}'


def positive_number(value: object,
                    name: str) -> tuple[Optional[float], Optional[str]]:
    """A measured quantity that has to be a finite positive number, or a reason.

    `if not (eps and wall and steps)` was the whole of this check. Truthiness
    rejects 0 and 0.0 correctly and accepts everything else: `wall_time_s: NaN`
    went into the fit and collapsed the group's rate to the 1e-06 floor, because
    `max(1e-6, nan)` returns 1e-6; `wall_time_s: -5.0` went in as a negative
    duration. Neither is a number of seconds a run can have taken.
    """
    if value is None:
        return None, f'{name} is missing'
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None, f'{name} is {value!r}, not a number'
    if not math.isfinite(number):
        return None, f'{name} is {value!r}, which is not finite'
    if number <= 0:
        return None, f'{name} is {number:g}, and a finished run has none'
    return number, None


def harvest_rows(root: str, excludes: Sequence[str] = (),
                 allow_legacy: bool = False) -> tuple[list[dict], list[dict]]:
    """Read a run tree into fit rows, and a ledger of everything not read.

    The second return value is the point. `DESIGN.md` 9 lists silent dropping as
    a guardrail and `STANDING_INSTRUCTIONS` S6 requires that what was skipped be
    reported with its reason; the version of this loop that shipped dropped
    manifests on four different failures without counting any of them.
    """
    rows: list[dict] = []
    skipped: list[dict] = []

    def skip(path: str, reason: str) -> None:
        skipped.append({'run_dir': os.path.dirname(path), 'reason': reason})

    for path in discover_manifests(root):
        try:
            with open(path, encoding='utf-8') as fh:
                man = json.load(fh)
        except OSError as exc:
            skip(path, f'unreadable: {exc.strerror or exc}')
            continue
        except json.JSONDecodeError as exc:
            skip(path, f'not valid JSON: {exc.msg} at line {exc.lineno}')
            continue
        if not isinstance(man, dict):
            skip(path, f'manifest is a {type(man).__name__}, not an object')
            continue
        res = man.get('result')
        if res is None:
            skip(path, 'no result block: the run did not finish')
            continue
        if not isinstance(res, dict):
            skip(path, f'result is a {type(res).__name__}, not an object')
            continue

        run_dir = os.path.dirname(path)
        cfg, note = config_from_manifest(man.get('config') or {}, allow_legacy)
        if cfg is None:
            skip(path, note or 'config could not be read')
            continue
        hit = [pattern for pattern in excludes
               if pattern in run_dir or pattern in (cfg.label or '')]
        if hit:
            skip(path, f'excluded by --exclude {", ".join(hit)}')
            continue

        eps, why = positive_number(res.get('episodes_completed'),
                                   'episodes_completed')
        wall, why = (None, why) if why else positive_number(
            res.get('wall_time_s'), 'wall_time_s')
        steps, why = (None, why) if why else positive_number(
            res.get('env_steps'), 'env_steps')
        if why:
            skip(path, why)
            continue

        rule_f = RULE_FACTOR.get(cfg.target_rule, 1.0)
        load = measurement_load(cfg) / ANCHOR_LOAD
        rows.append({
            'episodes': int(eps), 'env_steps': float(steps),
            # In the anchors' own convention, which is what `CostModel.seconds`
            # consumes: see `harvest` for why dividing by the raw
            # `measurement_load` instead deflated every coefficient by 1/1.35.
            'wall': float(wall) / rule_f / load,
            'actual': float(wall), 'load': load, 'run_dir': run_dir,
            'env': cfg.env, 'canonical': envs.parse(cfg.env).canonical(),
            'env_id': envs.parse(cfg.env).env_id, 'arch': cfg.arch,
            'rule': cfg.target_rule, 'label': cfg.label, 'cfg': cfg,
            'legacy': note,
            'rate': float(wall) / rule_f / load / float(steps),
            'final_len': res.get('episode_length_final100'),
        })
        if note:
            skipped.append({'run_dir': run_dir, 'reason': note + ' (still used)',
                            'used': True})
    return rows, skipped


def robust_centre(values: Sequence[float]) -> tuple[float, float]:
    """Median and MAD-derived sigma, the scale an outlier is judged against.

    A mean and an SD cannot be used here: the four stalled P0 runs are 26 times
    the median, so they would set the very scale that is meant to catch them. The
    1.4826 puts the MAD on the same footing as an SD for normal data.
    """
    if not values:
        return 0.0, 0.0
    med = statistics.median(values)
    mad = statistics.median([abs(v - med) for v in values])
    return float(med), float(1.4826 * mad)


def gate_outliers(rows: list[dict]) -> list[dict]:
    """Mark the runs whose implied step rate cannot be priced from.

    Returns the excluded rows, each carrying `exclusion` (the gate that caught
    it) and `exclusion_detail`. Nothing is removed from the tree, nothing is
    hidden, and the caller prints all of it.

    Two gates, whole-harvest first, for the reason given at OUTLIER_GLOBAL_FACTOR:
    a stall is a property of the machine on the night, so it can take out half of
    a small group and corrupt that group's own median. The gates operate on the
    step rate rather than on wall time because wall time legitimately varies with
    the number of steps a run took, and the rate is the coefficient being fitted.
    """
    rates = [r['rate'] for r in rows]
    if len(rates) < 3:
        return []
    excluded: list[dict] = []
    med, _sigma = robust_centre(rates)
    for row in rows:
        if med > 0 and (row['rate'] > med * OUTLIER_GLOBAL_FACTOR
                        or row['rate'] * OUTLIER_GLOBAL_FACTOR < med):
            row['exclusion'] = 'whole-harvest'
            row['exclusion_detail'] = (
                f'{row["rate"] * 1000:.2f} ms/step against a whole-harvest '
                f'median of {med * 1000:.2f}, a factor of '
                f'{max(row["rate"] / med, med / row["rate"]):.1f} beyond the '
                f'{OUTLIER_GLOBAL_FACTOR:g}x gate')
            excluded.append(row)

    survivors = [r for r in rows if 'exclusion' not in r]
    groups: dict[tuple, list[dict]] = {}
    for row in survivors:
        groups.setdefault((row['fit_key'], row['arch']), []).append(row)
    for _key, members in sorted(groups.items()):
        if len(members) < OUTLIER_MIN_GROUP:
            continue
        gmed, gsigma = robust_centre([r['rate'] for r in members])
        if gmed <= 0:
            continue
        for row in members:
            factor = max(row['rate'] / gmed, gmed / row['rate'])
            z = abs(row['rate'] - gmed) / gsigma if gsigma > 0 else float('inf')
            if factor > OUTLIER_MIN_FACTOR and z > OUTLIER_ROBUST_Z:
                row['exclusion'] = 'group'
                row['exclusion_detail'] = (
                    f'{row["rate"] * 1000:.2f} ms/step against a group median of '
                    f'{gmed * 1000:.2f}: {factor:.1f}x, robust z '
                    f'{z:.1f}')
                excluded.append(row)
    return excluded


def solve_steps_end(total_steps: float, episodes: int, steps_start: float,
                    ramp_episodes: int) -> Optional[float]:
    """The ramp plateau that reproduces a run's observed total env steps.

    The ramp is `steps_start` rising linearly to `steps_end` over
    `ramp_episodes`, then flat, so the total over `episodes` episodes is
    `R*(s0+s1)/2 + (N-R)*s1` with `R = min(ramp, N)`. Everything in that except
    `s1` is known from the tree: `N` is the budget, `s0` is the measured
    random-policy length, `R` stays documented, and the total is what the runs
    accumulated. So `s1` is solved rather than guessed.

    This is what `--from-runs` refused to do, on the grounds that "a finished
    short run shows neither" startup nor the length ramp. True of a short run and
    false of the tree it was being pointed at: 44 runs of 1000 episodes each are
    exactly the regime where the ramp is observable, and the documented plateau
    of 560 mispredicted LunarLander's step total by 0.94x to 1.64x across those
    runs while the note in the table said the ramp was documented because it
    could not be seen.

    A linear ramp cannot represent a length that rises and then falls, which is
    what LunarLander does as the policy stops hovering, so the solved plateau
    sits above the observed final-hundred length. That is a limit of the shape,
    not an error in the solve: what the cost model needs right is the integral,
    and the integral is what is matched.
    """
    n = int(episodes)
    ramp = max(0, min(int(ramp_episodes), n))
    denominator = n - ramp / 2.0
    if n <= 0 or denominator <= 0:
        return None
    steps_end = (total_steps - ramp * steps_start / 2.0) / denominator
    if not math.isfinite(steps_end) or steps_end <= 0:
        return None
    return float(steps_end)


def harvest(root: str, excludes: Sequence[str] = (),
            allow_legacy: bool = False) -> tuple[dict, list[dict], dict]:
    """Fit the cost model from a finished run tree, and report what it did.

    A real tree is a better calibration source than a fresh short run: it spans
    architectures, conditions and environments, and it was produced by the code
    that will produce the sweep. Three things are fitted per group: the step
    rate, the ramp plateau (`solve_steps_end`), and, where the tree can identify
    it, the per-episode overhead.

    The fit is `wall_time_s / RULE_FACTOR / (measurement_load / ANCHOR_LOAD) =
    a * episodes + b * env_steps`. That second divisor is the correction that
    matters most in this file: the coefficients are consumed by
    `CostModel.seconds` as `startup + (measurement_load / ANCHOR_LOAD) *
    (a * episodes + b * steps)`, so they have to be expressed at the anchor's
    cadence, not at unit cadence. Dividing by the raw `measurement_load`, as this
    did, deflated every harvested and measured coefficient by exactly
    1/ANCHOR_LOAD = 0.741. It was visible all along and nothing looked: a
    single-run self-fit has to reproduce that run exactly, and the model check
    printed 0.74 while the header said the throughput was measured.

    Grouping is by environment *variant* where the tree supports it. The old
    grouping was by `env_id`, so the interface arms (padded observation, extended
    action set, and genuinely different per-step cost) were pooled into the base
    LunarLander rate while the module docstring claimed the shift variants were
    priced from measurement. They are now, whenever a variant has
    MIN_VARIANT_RUNS runs of its own; below that they are pooled and the entry's
    note says the rate is pooled rather than measured.

    Returns `(entries, residuals, diagnostics)`. The diagnostics are not
    decoration: they carry the runs skipped and why, the runs excluded as
    outliers and by which gate, and the per-group counts, which is what makes a
    fit auditable instead of merely printed.
    """
    rows, skipped = harvest_rows(root, excludes, allow_legacy)
    # Every manifest found became a row or a skip, so the count is available
    # without walking the tree a second time.
    diagnostics: dict = {
        'root': root,
        'manifests': len(rows) + len([s for s in skipped if not s.get('used')]),
        'skipped': skipped, 'excluded': [], 'groups': [],
        'used': 0, 'median_ms_per_step': None,
    }
    if not rows:
        return {}, [], diagnostics

    # Which key each run is fitted under: its own variant when the variant has
    # enough runs, otherwise its base environment.
    variant_counts: dict[tuple, int] = {}
    for row in rows:
        key = (row['canonical'], row['arch'])
        variant_counts[key] = variant_counts.get(key, 0) + 1
    for row in rows:
        own = variant_counts[(row['canonical'], row['arch'])]
        row['fit_key'] = (row['canonical'] if own >= MIN_VARIANT_RUNS
                          else row['env_id'])
        row['pooled'] = row['fit_key'] != row['canonical']

    excluded = gate_outliers(rows)
    diagnostics['excluded'] = [
        {'run_dir': r['run_dir'], 'label': r['label'], 'env': r['env'],
         'arch': r['arch'], 'ms_per_step': round(r['rate'] * 1000, 3),
         'gate': r['exclusion'], 'detail': r['exclusion_detail']}
        for r in excluded]
    kept = [r for r in rows if 'exclusion' not in r]
    diagnostics['used'] = len(kept)
    if kept:
        diagnostics['median_ms_per_step'] = round(
            statistics.median([r['rate'] for r in kept]) * 1000, 3)
    if not kept:
        return {}, [], diagnostics

    import numpy as np

    groups: dict[tuple, list[dict]] = {}
    for row in kept:
        groups.setdefault((row['fit_key'], row['arch']), []).append(row)

    entries: dict[str, dict] = {}
    residuals: list[dict] = []
    for (fit_key, arch), members in sorted(groups.items()):
        env_id = envs.parse(fit_key).env_id
        anchor = DEFAULT_ANCHORS.get(env_id, DEFAULT_ANCHORS['LunarLander-v3'])
        design = np.array([[r['episodes'], r['env_steps']] for r in members],
                          float)
        y = np.array([r['wall'] for r in members], float)
        eps_col, step_col = design[:, 0], design[:, 1]
        # Identifiability guard, load-bearing rather than decorative. A
        # per-episode overhead can only be told apart from a per-step rate if the
        # harvested runs differ materially in steps *per episode*. In a tree
        # where every run has the same episode budget they do not, and an
        # unconstrained fit hands the whole cost to whichever column the noise
        # favours: measured here, the unguarded fit gave the LunarLander mlp
        # group 0.178 s/episode and a step rate of 0.0001 ms, a four-fold
        # under-estimate extrapolated to a full run, in the direction nobody
        # notices until the sweep overruns. So both coefficients are fitted only
        # when the budgets span a factor of three and the steps per episode vary;
        # otherwise the overhead is held documented and only the rate is fitted,
        # which is the conservative direction (the overhead is a few per cent of
        # a run, the rate is nearly all of it).
        #
        # On every tree this repository has produced the answer is "not
        # identified", because a catalogue run is 1000 episodes and nothing else.
        # The branch is kept for a tree that does span budgets, and the fit that
        # was used is named in the entry, so nobody has to read this code to know
        # which one priced their sweep.
        per_ep = step_col / np.maximum(1.0, eps_col)
        spread = float(per_ep.std() / per_ep.mean()) if per_ep.mean() else 0.0
        budget_span = float(eps_col.max() / max(1.0, eps_col.min()))
        identified = (len(members) >= 5 and np.linalg.matrix_rank(design) == 2
                      and budget_span >= 3.0 and spread > 0.15)
        fitted = ['s_per_env_step']
        if identified:
            from scipy.optimize import nnls
            coef, _ = nnls(design, y)
            a, b = float(coef[0]), float(coef[1])
            fitted.append('s_per_episode')
            fit = (f'non-negative least squares over {len(members)} run(s) '
                   f'spanning a {budget_span:.1f}x episode-budget range')
        else:
            a = anchor.s_per_episode
            b = float((y - a * eps_col).sum() / step_col.sum())
            fit = (f'{len(members)} run(s) spanning only a {budget_span:.1f}x '
                   f'episode-budget range (steps/episode spread '
                   f'{spread:.2f}), so the two coefficients are not separately '
                   f'identified: the per-episode overhead is held at its '
                   f'documented {a:.3f} s and only the step rate is fitted')
        if not math.isfinite(b) or b <= 0:
            # Reachable when the documented overhead alone exceeds the observed
            # wall time, which means the overhead is wrong for this machine and
            # not that the rate is zero. The anchor is kept and the entry says
            # the rate was not fitted, rather than a floor being written and
            # labelled measured.
            b = anchor.s_per_env_step * ARCH_FACTOR.get(arch, 1.0)
            fitted.remove('s_per_env_step')
            fit = (f'RATE NOT FITTED over {len(members)} run(s): the documented '
                   f'per-episode overhead of {a:.3f} s already exceeds the '
                   f'observed wall time, which leaves no positive step rate. '
                   f'The documented rate is kept')

        # The ramp. `steps_start` is the measured random-policy length for this
        # environment; the plateau is solved so that the modelled step total
        # matches what these runs accumulated.
        start = measured_steps_start(fit_key) or anchor.steps_start
        if measured_steps_start(fit_key) is not None:
            fitted.append('steps_start')
        budget = int(statistics.median([r['episodes'] for r in members]))
        mean_len = statistics.median([r['env_steps'] / r['episodes']
                                      for r in members])
        solved = solve_steps_end(mean_len * budget, budget, start,
                                 anchor.ramp_episodes)
        observed_final = [float(r['final_len']) for r in members
                          if isinstance(r['final_len'], (int, float))
                          and math.isfinite(float(r['final_len']))]
        if solved is None:
            steps_end, ramp_note = anchor.steps_end, (
                'the ramp could not be solved from these runs and stays '
                'documented')
        else:
            steps_end = round(solved, 1)
            fitted.append('steps_end')
            ramp_note = (
                f'ramp plateau solved from the observed {mean_len:.0f} env '
                f'steps per episode over {budget} episodes, holding the '
                f'documented {anchor.ramp_episodes}-episode ramp length and the '
                f'measured random-policy start of {start:.0f}')
            if observed_final:
                ramp_note += (f'; the final hundred episodes of these runs '
                              f'averaged {statistics.median(observed_final):.0f} '
                              f'steps, and a monotone ramp matched to the total '
                              f'cannot equal that when the length is not '
                              f'monotone')

        pooled = any(r['pooled'] for r in members)
        variants = sorted({r['canonical'] for r in members})
        pool_note = ''
        if pooled and len(variants) > 1:
            pool_note = (f'. RATE POOLED across {len(variants)} variant(s) '
                         f'({", ".join(envs.parse(v).slug() for v in variants)}) '
                         f'because none of them has {MIN_VARIANT_RUNS} runs of '
                         f'its own here: the per-step rate is assumed equal '
                         f'across them, only the episode lengths are rescaled')
        dropped = [r for r in excluded
                   if (r['fit_key'], r['arch']) == (fit_key, arch)]
        drop_note = ''
        if dropped:
            drop_note = (f'. {len(dropped)} run(s) excluded before fitting: '
                         + '; '.join(f'{r["label"]} '
                                     f'({r["rate"] * 1000:.1f} ms/step, '
                                     f'{r["exclusion"]} gate)'
                                     for r in dropped))
        entries[f'{fit_key}|{arch}'] = {
            's_per_env_step': round(b, 6), 's_per_episode': round(a, 6),
            'steps_start': start, 'steps_end': steps_end,
            'ramp_episodes': anchor.ramp_episodes,
            'startup_s': DEFAULT_STARTUP_S, 'source': 'harvested',
            'fitted': sorted(set(fitted)),
            'note': f'{fit}; {ramp_note}. Process startup and the ramp length '
                    f'stay documented, because a finished run records neither'
                    f'{pool_note}{drop_note}',
        }
        all_rates = [r['rate'] for r in members] + [r['rate'] for r in dropped]
        diagnostics['groups'].append({
            'key': f'{fit_key}|{arch}', 'runs': len(members),
            'excluded': len(dropped), 'pooled': pooled,
            'variants': len(variants),
            'ms_per_step': round(b * 1000, 3),
            'median_ms_all': round(statistics.median(all_rates) * 1000, 3),
            'median_ms_kept': round(
                statistics.median([r['rate'] for r in members]) * 1000, 3),
            'gate_could_not_run': len(members) < OUTLIER_MIN_GROUP,
            'identified': bool(identified),
        })
        # Predicted against actual, on the runs the fit was made from, through
        # the same pricing path a plan uses. The harvested runs' own observed
        # mean episode length is used, so this checks the rate rather than the
        # ramp, and it is reported even when it is unflattering: that is what it
        # is for.
        for r in members:
            observed_len = r['env_steps'] / r['episodes']
            check = CostModel(key=f'{fit_key}|{arch}',
                              s_per_env_step=b * RULE_FACTOR.get(r['rule'], 1.0),
                              s_per_episode=a, steps_start=observed_len,
                              steps_end=observed_len, ramp_episodes=0,
                              startup_s=0.0, source='harvested')
            pred = check.seconds(r['cfg'])
            residuals.append({
                'run_dir': r['run_dir'], 'env': r['env'], 'arch': r['arch'],
                'label': r['label'],
                'predicted_s': round(pred, 2), 'actual_s': round(r['actual'], 2),
                'ratio': round(pred / r['actual'], 3) if r['actual'] else None})
    return entries, residuals, diagnostics


def fit_check(residuals: Sequence[dict]) -> dict:
    """Does the fit reproduce the runs it was fitted from?

    The one check a self-fit cannot fail honestly, and the one nothing performed.
    A model fitted on a set of runs and then asked to price those same runs has
    no excuse for a systematic offset: an offset means a convention error, and
    the convention error that lived here for the life of the file put the median
    ratio at 0.741, which is 1/ANCHOR_LOAD to three decimals, printed under a
    header that said the throughput was measured.

    Two thresholds, because they catch different failures. The median catches a
    systematic offset. The tail catches a contaminated fit, where the centre
    looks fine because the outliers pulled the coefficient far enough to
    mis-price everything else: with the four stalled P0 runs included the
    residuals ran 0.06 to 4.24.
    """
    ratios = [r['ratio'] for r in residuals if r.get('ratio')]
    if not ratios:
        return {'ran': False, 'ok': True, 'problems': []}
    median = statistics.median(ratios)
    tail = [r for r in ratios
            if r > FIT_CHECK_TAIL_FACTOR or r < 1.0 / FIT_CHECK_TAIL_FACTOR]
    problems: list[str] = []
    if not 1.0 / FIT_CHECK_MEDIAN_TOL <= median <= FIT_CHECK_MEDIAN_TOL:
        problems.append(
            f'median predicted/actual is {median:.2f} over {len(ratios)} run(s), '
            f'outside {1 / FIT_CHECK_MEDIAN_TOL:.2f} to '
            f'{FIT_CHECK_MEDIAN_TOL:.2f}. A fit that cannot reproduce its own '
            f'training data is a convention error, not a noisy estimate')
    if len(tail) > FIT_CHECK_TAIL_FRACTION * len(ratios):
        problems.append(
            f'{len(tail)} of {len(ratios)} run(s) are beyond a factor of '
            f'{FIT_CHECK_TAIL_FACTOR:g} from their own prediction, above the '
            f'{FIT_CHECK_TAIL_FRACTION:.0%} this check allows: the fit is being '
            f'pulled by runs it does not describe')
    return {'ran': True, 'ok': not problems, 'problems': problems,
            'median': round(median, 3), 'n': len(ratios),
            'low': round(min(ratios), 3), 'high': round(max(ratios), 3),
            'beyond_tail': len(tail)}


def anchor_check(table: dict) -> list[dict]:
    """What the model in force implies for each environment's documented band.

    The report has always printed this comparison. It has never done anything
    about it: on the P0 tree it printed a CartPole figure 2.7x and a LunarLander
    figure 4x away from the anchors beside them, in the same breath as the word
    `harvested`, and the docstring's promise that "a mis-calibration is visible
    rather than latent" came to nothing because visible is not the same as acted
    on.

    Which side is at fault depends on where the coefficients came from, and the
    verdict says so rather than assuming. A model fitted from data that
    disagrees with a documented band means the band is stale, and the band is
    prose: it warns, it does not stop anything. A model built *from* the
    documented anchors that disagrees with those same anchors is an internal
    contradiction in this file, and that is an error.
    """
    out: list[dict] = []
    for env_id, anchor in sorted(DEFAULT_ANCHORS.items()):
        lo, hi = anchor.check_seconds
        for arch in ('dueling',):
            model = cost_model(env_id, arch, 'vanilla', table)
            cfg = Config(arch=arch, env=env_id,
                         num_episodes=anchor.check_episodes)
            seconds = model.seconds(cfg)
            row = {'env': env_id, 'arch': arch,
                   'episodes': anchor.check_episodes,
                   'seconds': round(seconds, 1), 'band': [lo, hi],
                   'source': model.source, 'fitted': list(model.fitted),
                   'verdict': 'no band claimed', 'ok': True}
            if lo > 0 and hi > 0:
                if lo <= seconds <= hi:
                    row['verdict'] = 'inside the documented band'
                else:
                    ratio = seconds / hi if seconds > hi else lo / seconds
                    row['ratio'] = round(ratio, 2)
                    row['ok'] = ratio <= ANCHOR_TOLERANCE
                    # Which side is doubted turns on where the coefficients came
                    # from, not on whether this file happens to know which of
                    # them were fitted: a table written before `fitted` was
                    # recorded still says `harvested`, and calling that an
                    # internal contradiction would point the operator at the
                    # wrong file.
                    if model.source in ('measured', 'harvested'):
                        row['verdict'] = (
                            f'{ratio:.1f}x outside the documented band. The '
                            f'coefficients came from a calibration, so either '
                            f'the band is stale and wants re-deriving, or that '
                            f'calibration is wrong; the model check on the runs '
                            f'the fit was made from is what tells them apart')
                    else:
                        row['verdict'] = (
                            f'{ratio:.1f}x outside the documented band while '
                            f'being built from that same anchor: this is a '
                            f'contradiction inside plan.py, not a stale note')
            out.append(row)
    return out


def measure_disk(root: str) -> Optional[dict]:
    """Durable megabytes per completed run, measured from a real tree.

    Shares `discover_manifests` with the harvest so that a subtree or a single
    run directory measures the same set of runs it prices.
    """
    sizes = []
    for path in discover_manifests(root):
        total = 0
        for base, _dirs, files in os.walk(os.path.dirname(path)):
            for name in files:
                try:
                    total += os.path.getsize(os.path.join(base, name))
                except OSError:
                    pass
        sizes.append(total / 1e6)
    if not sizes:
        return None
    return {'runs': len(sizes), 'mean_mb': round(statistics.mean(sizes), 3),
            'min_mb': round(min(sizes), 3), 'max_mb': round(max(sizes), 3),
            'total_mb': round(sum(sizes), 1)}


def write_throughput(entries: dict, path: str, method: str) -> None:
    """Merge new entries into the calibration table on disk, keeping the old.

    Three things happen here that did not. A copy of the previous table is kept
    at `<path>.bak`, because `--from-runs` mutates a file in the repository as a
    side effect of a tool whose own banner says it launches nothing, and a
    harvest from a damaged tree used to overwrite a good table with no way back.
    Every changed coefficient is printed, old against new, so that the size of
    the correction being adopted is visible at the moment it is adopted rather
    than in a diff later. And the entries being left in place are named: the
    merge is `{**existing, **new}` and never removes, so an entry harvested from
    a tree nobody remembers survives every later calibration that does not
    happen to reproduce its key.
    """
    existing, _meta = load_throughput(path)
    if os.path.exists(path):
        try:
            with open(path, encoding='utf-8') as fh:
                previous = fh.read()
            with open(path + '.bak', 'w', encoding='utf-8') as fh:
                fh.write(previous)
            print(f'  previous table kept at {path}.bak')
        except OSError as exc:
            print(f'  could not write {path}.bak ({exc}); refusing to '
                  f'overwrite a table that cannot be backed up')
            return
    for key in sorted(entries):
        new_rate = entries[key].get('s_per_env_step')
        old = existing.get(key)
        if not isinstance(old, dict):
            print(f'  + {key:44s} {new_rate * 1000:8.3f} ms/step (new)')
            continue
        old_rate = old.get('s_per_env_step')
        if isinstance(old_rate, (int, float)) and old_rate:
            print(f'  ~ {key:44s} {old_rate * 1000:8.3f} -> '
                  f'{new_rate * 1000:8.3f} ms/step '
                  f'({new_rate / old_rate:.2f}x)')
        else:
            print(f'  ~ {key:44s} -> {new_rate * 1000:8.3f} ms/step')
    stale = sorted(set(existing) - set(entries))
    if stale:
        print(f'  {len(stale)} entr(y/ies) kept from an earlier calibration and '
              f'not re-fitted by this one:')
        for key in stale:
            print(f'    = {key}')
    payload = {
        'schema': THROUGHPUT_SCHEMA,
        'written': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'method': method,
        'machine': provenance.machine(),
        'design_hash': provenance.file_hash(
            os.path.join(REPO, 'experiments', 'DESIGN.md')),
        'eval_episode_weight': EVAL_EPISODE_WEIGHT,
        'arch_factor': ARCH_FACTOR,
        'rule_factor': RULE_FACTOR,
        'contention': CONTENTION,
        'entries': {**existing, **entries},
    }
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f'\n{len(payload["entries"])} calibration entries -> {path}')


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def hms(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 90:
        return f'{seconds:.0f}s'
    if seconds < 5400:
        return f'{seconds / 60:.1f}m'
    if seconds < 86400 * 10:
        return f'{seconds / 3600:.1f}h'
    return f'{seconds / 86400:.1f}d'


def rule(char: str = '-', width: int = 78) -> str:
    return char * width


def elide(text: str, width: int) -> str:
    """Shorten from the middle: the tail of an env slug is what identifies it.

    `lunarlander_enable_wind1_turbulence_power1.5_wind_power7.5` and its
    `wind_power15` sibling differ only in their last characters, so a left
    truncation would print two identical-looking rows for two different costs.
    """
    if len(text) <= width:
        return text
    keep = width - 2
    head = keep // 2
    return text[:head] + '..' + text[-(keep - head):]


def wrapped(text: str, indent: str = '  - ', width: int = 76,
            cont: Optional[str] = None) -> None:
    for line in textwrap.wrap(text, width, initial_indent=indent,
                              subsequent_indent=cont or ' ' * len(indent)):
        print(line)


def print_harvest(diag: dict) -> None:
    """What the calibration read, what it dropped, and why.

    `STANDING_INSTRUCTIONS` S6 and the silent-dropping row of `DESIGN.md` 9. The
    harvest used to print one line, and only when it had found nothing at all:
    `--from-runs runs_demo` discarded 198 manifests and said "no usable runs",
    while a mixed tree dropped its legacy manifests with no message whatever and
    fitted on whatever happened to parse.
    """
    print('\n' + rule())
    print(f'CALIBRATION HARVEST from {diag.get("root")}')
    print(rule())
    print(f'  {diag.get("manifests", 0)} manifest(s) found, '
          f'{diag.get("used", 0)} used in the fit')
    if diag.get('median_ms_per_step') is not None:
        print(f'  median implied step rate over the runs used: '
              f'{diag["median_ms_per_step"]:.2f} ms/step')
    skipped = [s for s in diag.get('skipped', ()) if not s.get('used')]
    if skipped:
        reasons: dict[str, list] = {}
        for row in skipped:
            reasons.setdefault(row['reason'], []).append(row)
        print(f'  {len(skipped)} manifest(s) skipped, by reason:')
        for reason, members in sorted(reasons.items(),
                                      key=lambda kv: -len(kv[1])):
            wrapped(f'{len(members)}x {reason}', indent='    - ',
                    cont='      ')
            lead = f'first of {len(members)}: ' if len(members) > 1 else ''
            print(f'      {lead}{members[0]["run_dir"]}')
    adapted = [s for s in diag.get('skipped', ()) if s.get('used')]
    if adapted:
        print(f'  {len(adapted)} manifest(s) carried config fields this Config '
              f'no longer has and were adapted under --allow-legacy-configs '
              f'(some may still have been excluded below): '
              f'{adapted[0]["reason"]}')
    excluded = diag.get('excluded') or []
    if excluded:
        print(f'  {len(excluded)} run(s) excluded from the fit as outliers:')
        for row in sorted(excluded, key=lambda r: -r['ms_per_step']):
            print(f'    {row["ms_per_step"]:9.2f} ms/step  '
                  f'{row["label"][:34]:34s} {row["arch"]:8s} {row["gate"]}')
            wrapped(row['detail'], indent='          ', cont='          ')
        print('    An excluded run is still a run: it is not deleted, not '
              'ignored elsewhere,\n    and it is in the JSON. What it is not is '
              'evidence about how fast this\n    machine trains.')
    for group in diag.get('groups', ()):
        print(f'    fitted {elide(group["key"], 44):44s} '
              f'{group["ms_per_step"]:8.3f} '
              f'ms/step over {group["runs"]:3d} run(s)'
              + (f', {group["excluded"]} excluded' if group['excluded'] else '')
              + (f', rate pooled across {group["variants"]} variants'
                 if group['pooled'] and group.get('variants', 1) > 1 else
                 ', fitted under its base environment key'
                 if group['pooled'] else ''))
        if group.get('excluded'):
            wrapped(f'group median {group["median_ms_all"]:.2f} -> '
                    f'{group["median_ms_kept"]:.2f} ms/step across the '
                    f'exclusion: that shift is what the gate moved the '
                    f'coefficient by', indent=' ' * 11, cont=' ' * 11)
        if group.get('gate_could_not_run'):
            wrapped(f'fewer than {OUTLIER_MIN_GROUP} runs, so the per-group '
                    f'gate could not run on this group: a contention artefact '
                    f'here is invisible and the rate may be high',
                    indent=' ' * 11, cont=' ' * 11)


def tuned_stage_cost(inv: Inventory) -> dict:
    """What the tuned stage of DESIGN.md 3.3 adds to this selection.

    `added_runs` is the count of run directories no common-policy arm in this
    same selection also asks for, which is the figure `DESIGN.md` 3.3 calls an
    upper bound rather than a doubling: a cell whose E3 selection equals the a
    priori configuration produces identical run digests, so its tuned arms share
    the directories and add nothing.

    `shared_cells` and `own_cells` come from the stored selection rather than
    from the run counts, so they are right even when the common-policy arms are
    not in the selection being priced and every tuned run therefore counts as
    added.
    """
    selection = inv.selection
    tuned_ids = inv.tuned_experiments
    added = inv.tuned_only_runs
    common = [eid for eid in reg.TUNED_OF.values()
              if eid in {r.id for r in inv.experiments}]
    out = {
        'experiments': tuned_ids,
        'available': selection is not None,
        'priced': bool(tuned_ids),
        'policy': reg.TUNED_POLICY,
        'common_policy_experiments_in_selection': common,
        'added_runs': len(added),
        'added_seconds': sum(r.seconds for r in added),
        'added_runs_pending': sum(1 for r in added if not r.complete),
        'shared_runs': sum(1 for r in inv.runs
                           if len(set(r.experiments) & set(reg.TUNED_OF)) > 0
                           and not set(r.experiments) <= set(reg.TUNED_OF)),
    }
    if selection is not None:
        out.update(
            selection_id=selection.selection_id,
            selection_short_id=selection.short_id,
            rule_id=selection.rule.get('id'),
            rule_placeholder=bool(selection.is_placeholder),
            seed_block=selection.seed_block,
            env=selection.env,
            source_experiment=selection.source_experiment,
            shared_cells=list(selection.shared_cells),
            own_cells=[key for key in sorted(selection.cells)
                       if key not in set(selection.shared_cells)],
            cells={key: cell.config.to_dict()
                   for key, cell in sorted(selection.cells.items())})
    return out


def print_tuned_stage(inv: Inventory) -> None:
    """The tuned stage's cost, or the one line saying there is not one."""
    cost = tuned_stage_cost(inv)
    print('\n' + rule())
    print('TUNED STAGE: DESIGN.md 3.3 secondary policy, per-cell tuned')
    print(rule())
    if not cost['available']:
        print(f'  no selection at '
              f'{tuning.selection_path(inv.out_root)}, so E1t and E2t cannot '
              f'be enumerated')
        print('  and are not priced above. DESIGN.md 3.3 makes the stage '
              'sequentially')
        print('  dependent on E3; ANALYSIS_PLAN.md 6.6 budgets about 31 h for '
              'it, to be')
        print('  re-costed here once a selection exists. Under '
              'ANALYSIS_PLAN.md 2.4 every')
        print('  arbitration verdict is not-evaluable until it does, so no RQ2 '
              'or RQ3')
        print('  conclusion may be asserted from the common policy alone.')
        return
    print(f'  selection : {cost["selection_short_id"]}  '
          f'rule={cost["rule_id"]}'
          f'{"  PLACEHOLDER" if cost["rule_placeholder"] else ""}')
    print(f'  stored at : {tuning.selection_path(inv.out_root)}')
    print(f'  computed  : from {cost["source_experiment"]} on '
          f'{cost["seed_block"]}, env {cost["env"]}')
    for key, config in cost['cells'].items():
        shares = key in cost['shared_cells']
        print(f'    {key:<16} lr={config["lr"]:g} '
              f'{config["target_update"]}/{config["target_update_freq"]}  '
              + ('SHARES the common policy run directories: adds no runs'
                 if shares else 'own run directories'))
    if not cost['priced']:
        print('  not priced: E1t and E2t are available for this tree and are '
              'not in this')
        print('  selection. Add them to --experiments, or use --all, to see '
              'what they cost.')
        return
    print(f'  priced    : {", ".join(cost["experiments"])}')
    alongside = (', '.join(cost['common_policy_experiments_in_selection'])
                 or 'none selected')
    print(f'  adds      : {cost["added_runs"]} run directories '
          f'({hms(cost["added_seconds"])} of single-worker compute), on top of')
    print(f'              the common-policy arms in this same selection '
          f'({alongside})')
    print(f'  shares    : {cost["shared_runs"]} run director(ies) with those '
          f'arms, which is what')
    print('              DESIGN.md 3.3 means by an upper bound rather than a '
          'doubling: a cell')
    print('              selecting the a priori configuration has identical '
          'run digests.')
    if not cost['common_policy_experiments_in_selection']:
        print('  NOTE      : no common-policy arm is in this selection, so '
              'every tuned run')
        print('              counts as added even where the cell shares. To '
              'see the sharing,')
        print(f'              price them together: --experiments '
              f'{" ".join(sorted(set(reg.TUNED_OF.values())))} '
              f'{" ".join(cost["experiments"])}')
    print('  sources   : not retuned (DESIGN.md 3.3, ANALYSIS_PLAN.md 2.3), so '
          'they keep')
    print('              their base labels and are the same runs the common '
          'policy uses.')


def report(inv: Inventory, jobs_selected: Optional[int], list_runs: bool,
           keep_buffer: bool, disk_measured: Optional[dict],
           residuals: Optional[list[dict]],
           harvest_diag: Optional[dict] = None,
           checks: Optional[dict] = None) -> None:
    stamp = inv.stamp()
    print(rule('='))
    print('COST AND INVENTORY PLAN: this file launches nothing')
    print(rule('='))
    print(f'experiments : {", ".join(r.id for r in inv.experiments)}')
    print(f'seeds       : {inv.seeds_spec or "each experiment\'s declared block"}')
    print(f'out-root    : {inv.out_root}')
    method = inv.table_meta.get('method', '?')
    written = inv.table_meta.get('written', '?')
    if inv.measured:
        print(f'throughput  : measured: {method}, {written}')
    elif inv.partly_measured:
        print(f'throughput  : PARTLY MEASURED: {method}, {written}')
        print(f'              fitted from data, and not: '
              f'{", ".join(inv.documented_coefficients)} stay documented')
    else:
        print('throughput  : *** ESTIMATE, NOT MEASURED *** documented anchors; '
              'run --measure or --from-runs')
    if harvest_diag:
        print_harvest(harvest_diag)
    print('plan hashes : ' + '  '.join(f'{k}={(v or "?")[:12]}' for k, v
                                       in provenance.plan_hashes().items()))

    # -- per experiment --------------------------------------------------
    print('\n' + rule())
    print('EXPERIMENTS IN THIS SELECTION')
    print(rule())
    print(f'{"id":5s} {"name":13s} {"tier":>4s} {"family":12s} {"arms":>4s} '
          f'{"seed block":20s} {"alone":>6s} {"new":>5s}')
    for r in inv.experiments:
        print(f'{r.id:5s} {r.name:13s} {r.tier:>4d} {r.family:12s} {r.arms:>4d} '
              f'{r.seed_block:20s} {r.runs_alone:>6d} {r.runs_new:>5d}')
        wrapped(f'serves: {r.question}', indent='      ')
        if r.varies:
            print(f'      varies: {", ".join(r.varies)}')
        if r.review_refs:
            print(f'      reviews: {", ".join(r.review_refs)}')
    print('\n"alone" = runs if that experiment were launched on its own; "new" = '
          'runs it\nadds to this selection, in the order listed above.')

    print_tuned_stage(inv)

    # -- sharing ---------------------------------------------------------
    saved = inv.naive_total - inv.total
    pct = (100.0 * saved / inv.naive_total) if inv.naive_total else 0.0
    print('\n' + rule())
    print('DE-DUPLICATION -- a run is keyed by configuration digest, not by '
          'experiment')
    print(rule())
    print(f'  naive per-experiment sum      : {inv.naive_total:6d} runs')
    print(f'  de-duplicated total           : {inv.total:6d} runs')
    print(f'  saved by sharing              : {saved:6d} runs ({pct:.1f}%)')
    shared = [r for r in inv.runs if len(r.experiments) > 1]
    print(f'  runs claimed by >1 experiment : {len(shared):6d}')
    if shared:
        pairs: dict[str, int] = {}
        for r in shared:
            tag = '+'.join(r.experiments)
            pairs[tag] = pairs.get(tag, 0) + 1
        for tag, count in sorted(pairs.items(), key=lambda kv: -kv[1])[:10]:
            print(f'      {elide(tag, 34):34s} {count:4d} runs')
    print(f'  already complete under {inv.out_root}: '
          f'{inv.total - len(inv.pending)} / {inv.total}')
    not_counted = [r for r in inv.runs if r.problem]
    if not_counted:
        print(f'  {len(not_counted)} run(s) on disk are NOT counted complete, '
              f'and why:')
        for rec in sorted(not_counted, key=lambda r: r.label)[:12]:
            print(f'      {rec.label[:30]:30s} seed {rec.seed:<4d} {rec.run_dir}')
            wrapped(rec.problem, indent='          ', cont='          ')
        if len(not_counted) > 12:
            print(f'      and {len(not_counted) - 12} more; --json carries all '
                  f'of them')

    # -- source validity -------------------------------------------------
    print('\n' + rule())
    print('SOURCE VALIDITY AND RESERVE REPLACEMENTS: DESIGN.md 4.3')
    print(rule())
    ledger = os.path.join(inv.out_root, reg.REPLACEMENTS_RELPATH)
    print(f'  rejection ledger {ledger}: '
          + ('read' if inv.ledger_present else 'not present'))
    if inv.replacements:
        print(f'  {len(inv.replacements)} lineage(s) replaced from RESERVE, and '
              f'this plan follows them:')
        for (arm, default), seed in sorted(inv.replacements.items()):
            drawn = sorted(inv.replacement_draws.get(arm, ()))
            print(f'    {arm[:34]:34s} seed {default} -> {seed}'
                  + (f'   (RESERVE seeds drawn for this arm: '
                     f'{", ".join(str(s) for s in drawn)})' if drawn else ''))
        moved = [r for r in inv.runs if r.source_replaced]
        print(f'  {len(moved)} run(s) in this selection draw on a replaced '
              f'source.')
    rejected = inv.rejected_sources
    if rejected:
        print(f'  {len(rejected)} run(s) on disk stand on a source the gate '
              f'REJECTED, and are counted as work still to do:')
        for rec in sorted(rejected, key=lambda r: r.label):
            score = ('?' if rec.source_score is None
                     else f'{rec.source_score:.3f}')
            print(f'    {rec.label[:34]:34s} seed {rec.seed:<4d} source '
                  f'{rec.source_arm or "?"} seed {rec.source_default_seed} '
                  f'scored {score} against {reg.SOURCE_VALIDITY_GATE}')
            print(f'        {rec.run_dir}')
        if not inv.replacements:
            print('  No replacement has been recorded for them. The RESERVE '
                  'source run each one\n  needs is therefore NOT in the counts '
                  'above: run the sweep, which draws the\n  replacement and '
                  'writes the ledger, then re-run this plan.')
    if not rejected and not inv.replacements:
        print('  no rejection recorded, and no run on disk reports an invalid '
              'source')

    # -- breakdown -------------------------------------------------------
    for attr, title in (('env', 'ENVIRONMENT'), ('condition', 'CONDITION'),
                        ('cell', 'CELL')):
        print('\n' + rule())
        print(f'RUNS BY {title}')
        print(rule())
        for key, v in inv.by(attr).items():
            print(f'  {elide(key, 52):52s} {v["runs"]:5d} runs {v["pending"]:5d} '
                  f'pending {hms(v["seconds"]):>8s} serial')

    # -- seed blocks -----------------------------------------------------
    print('\n' + rule())
    print('SEED BLOCKS -- DESIGN.md 3.4, disjoint by construction and checked '
          'here')
    print(rule())
    for key, v in inv.by('seed_block').items():
        seeds = sorted({r.seed for r in inv.runs if r.seed_block == key})
        span = f'{min(seeds)}-{max(seeds)}' if len(seeds) > 1 else str(seeds[0])
        print(f'  {key:10s} {v["runs"]:5d} runs   seeds {span:12s} '
              f'({len(seeds)} distinct)')
    if list_runs or inv.total <= 40:
        print('\n  every scheduled run, with its block:')
        print(f'  {"block":9s} {"seed":>4s} {"cell":15s} {"condition":19s} '
              f'{"label":26s} {"exps":9s} {"cost":>7s} run_dir')
        for r in sorted(inv.runs, key=lambda r: (r.seed_block, r.seed, r.label)):
            print(f'  {r.seed_block:9s} {r.seed:>4d} {r.cell:15s} '
                  f'{r.condition:19s} {r.label[:26]:26s} '
                  f'{",".join(r.experiments)[:9]:9s} {hms(r.seconds):>7s} '
                  f'{r.run_dir}{" DONE" if r.complete else ""}')
    else:
        print(f'\n  {inv.total} runs: the per-run block listing is suppressed at '
              f'this size --\n  use --list-runs, or --json, which always carries '
              f'every run.')

    # -- throughput ------------------------------------------------------
    print('\n' + rule())
    print('COST MODEL' + stamp)
    print(rule())
    print(f'{"arch":8s} {"rule":8s} {"environment":26s} {"ms/step":>8s} '
          f'{"s/ep":>6s} {"episode len":>12s} {"xlen":>5s} {"source":>10s}')
    for key, m in sorted(inv.models.items()):
        env, arch, target_rule = key.split('|')
        span = f'{m.steps_start:.0f}->{m.steps_end:.0f}'
        print(f'{arch:8s} {target_rule:8s} '
              f'{elide(envs.parse(env).slug(), 26):26s} '
              f'{m.s_per_env_step * 1000:8.3f} {m.s_per_episode:6.3f} '
              f'{span:>12s} {m.length_scale:>5.2f} {m.source:>10s}')
    print('  "xlen" scales the base environment\'s modelled episode lengths by '
          'this variant\'s\n  measured random-policy length; lengths are '
          'truncated at max_steps when priced.')
    if inv.documented_coefficients:
        print(f'  documented, not fitted by any calibration in force: '
              f'{", ".join(inv.documented_coefficients)}')
    for note in sorted({m.note for m in inv.models.values() if m.note}):
        wrapped(note, indent='  * ')
    print('\n  self-check: what the model in force implies for each documented '
          'anchor,\n  and what this tool did about the answer.')
    for row in (checks or {}).get('anchors') or anchor_check(inv.table):
        lo, hi = row['band']
        band = (f'{hms(lo)} to {hms(hi)}' if lo and hi else 'no band claimed')
        print(f'    {row["env"]:16s} {row["arch"]:8s} {row["episodes"]:5d} ep '
              f'-> {hms(row["seconds"]):>7s} {row["source"]:>10s}  '
              f'documented: {band}')
        wrapped(('OK, ' if row['ok'] else 'MIS-CALIBRATED, ') + row['verdict'],
                indent='          ', cont='          ')
    print(f'  measurement load at the default cadence: {ANCHOR_LOAD:.2f} '
          f'evaluation-weighted\n  episodes per training episode; a run that '
          f'changes the cadence is rescaled.')

    # -- wall clock ------------------------------------------------------
    pending = inv.pending
    print('\n' + rule())
    print('PROJECTED WALL-CLOCK' + stamp)
    print(rule())
    print(f'  {len(pending)} run(s) to launch; '
          f'{inv.total - len(pending)} already complete')
    print(f'  {"jobs":>4s} {"per-run x":>9s} {"serial":>9s} {"wall":>9s} '
          f'{"speedup":>8s} {"busy":>6s} {"cp floor":>9s}')
    base = schedule(pending, 1, contention(1))['makespan_seconds']
    ladder = sorted(set(JOB_LADDER) | ({jobs_selected} if jobs_selected else set()))
    for proj in projections(pending, ladder):
        speed = base / proj['makespan_seconds'] if proj['makespan_seconds'] else 0
        mark = '  <-- --jobs' if proj['workers'] == jobs_selected else ''
        print(f'  {proj["workers"]:>4d} {proj["multiplier"]:>9.2f} '
              f'{hms(proj["serial_seconds"]):>9s} '
              f'{hms(proj["makespan_seconds"]):>9s} {speed:>7.2f}x '
              f'{proj["busy_fraction"] * 100:>5.0f}% '
              f'{hms(proj["critical_path_seconds"]):>9s}{mark}')
    external = schedule(pending, 1)['external_prerequisites']
    if external:
        print(f'  {external} run(s) depend on a source outside this selection, '
              f'so their dependency\n  wait is not modelled: launch the source '
              f'experiment first, or add it here.')
    print('  "per-run x" is the contention multiplier -- jobs=4 is the condition '
          'the anchors\n  were measured in and the rest is assumed. "cp floor" '
          'is the critical path: a\n  transfer run cannot start before its '
          'source finishes, so no number of workers\n  beats it.')

    # -- disk ------------------------------------------------------------
    copies = 1 + max((len(Config(num_episodes=r.episodes).prefix_checkpoints)
                      for r in inv.runs), default=0)
    workers = jobs_selected or max(JOB_LADDER)
    print('\n' + rule())
    print('DISK')
    print(rule())
    print(f'  durable    {DURABLE_MB_PER_RUN:.2f} MB/run x {inv.total} runs = '
          f'{inv.total * DURABLE_MB_PER_RUN / 1024:.2f} GB')
    if disk_measured:
        print(f'             measured under {inv.out_root}: '
              f'{disk_measured["mean_mb"]:.2f} MB/run over '
              f'{disk_measured["runs"]} runs (min {disk_measured["min_mb"]:.2f}, '
              f'max {disk_measured["max_mb"]:.2f})\n             -> '
              f'{inv.total * disk_measured["mean_mb"] / 1024:.2f} GB projected '
              f'at that size')
    print(f'  transient  {TRANSIENT_BUFFER_MB:.1f} MB per replay-buffer copy, '
          f'{copies} live cop{"y" if copies == 1 else "ies"} per run (the '
          f'rolling\n             checkpoint plus each prefix checkpoint) x '
          f'{workers} in flight = '
          f'{workers * copies * TRANSIENT_BUFFER_MB:.0f} MB peak')
    print('             deleted when a run completes, unless --keep-buffer')
    if keep_buffer:
        print(f'  --keep-buffer: buffers become durable, adding '
              f'{inv.total * copies * TRANSIENT_BUFFER_MB / 1024:.2f} GB')

    # -- model check -----------------------------------------------------
    if residuals:
        verdict = (checks or {}).get('fit') or fit_check(residuals)
        print('\n' + rule())
        print('MODEL CHECK against the runs the fit was made from')
        print(rule())
        if verdict.get('ran'):
            print(f'  {verdict["n"]} runs; predicted/actual wall time median '
                  f'{verdict["median"]:.2f}, range {verdict["low"]:.2f} to '
                  f'{verdict["high"]:.2f}; {verdict["beyond_tail"]} beyond a '
                  f'factor of {FIT_CHECK_TAIL_FACTOR:g}')
            for r in sorted(residuals,
                            key=lambda r: -abs((r['ratio'] or 1) - 1))[:5]:
                print(f'    {r["ratio"]:.2f}  predicted {r["predicted_s"]:8.1f}s '
                      f' actual {r["actual_s"]:8.1f}s  {r["env"][:28]:28s} '
                      f'{r["arch"]}')
            if verdict['ok']:
                print(f'  WITHIN TOLERANCE: a self-fit is expected to reproduce '
                      f'its own runs to\n  within '
                      f'{FIT_CHECK_MEDIAN_TOL:.2f}x on the median, and this one '
                      f'does.')
            else:
                for problem in verdict['problems']:
                    wrapped('FIT REJECTED: ' + problem, indent='  !! ',
                            cont='  !! ')

    # -- what is not modelled -------------------------------------------
    print('\n' + rule())
    print('NOT MODELLED, AND KNOWN TO BE MISSING')
    print(rule())
    for line in (
            'a source-validity rejection the runner has not recorded yet: the '
            'ledger and the manifests on disk are both read, but a source that '
            'fails the gate in a future run draws a RESERVE seed and adds a run '
            'nobody can price in advance, so every count here is a floor',
            'duplicated episodes inside a metrics stream: this file reads the '
            'trainer\'s own metrics_integrity verdict from each manifest and '
            'refuses to count a non-contiguous run complete, but it opens no '
            'metrics stream itself and cannot find a corruption the trainer did '
            'not record',
            'a crash and resume re-pays the per-run startup cost, and the '
            'resumed portion is not free',
            'evaluation cost is folded into the two coefficients at the default '
            'cadence and rescaled by measurement_load elsewhere; it is not '
            'timed separately',
            'the schedule is a greedy list schedule in dependency order, while '
            'the launcher shards by seed, which is coarser',
            'E10 (budget) is free by construction -- it re-evaluates E1 prefixes '
            'and so appears nowhere in this cost'):
        wrapped(line)

    if inv.warnings:
        print('\n' + rule('!'))
        for w in inv.warnings:
            wrapped(w, indent='!! ', cont='!! ')
        print(rule('!'))


def to_json(inv: Inventory, jobs_selected: Optional[int],
            disk_measured: Optional[dict],
            residuals: Optional[list[dict]],
            harvest_diag: Optional[dict] = None,
            checks: Optional[dict] = None) -> dict:
    pending = inv.pending
    copies = 1 + max((len(Config(num_episodes=r.episodes).prefix_checkpoints)
                      for r in inv.runs), default=0)
    workers = jobs_selected or max(JOB_LADDER)
    ladder = sorted(set(JOB_LADDER) | ({jobs_selected} if jobs_selected else set()))
    return {
        'selection': {'experiments': [r.id for r in inv.experiments],
                      'seeds': inv.seeds_spec, 'out_root': inv.out_root,
                      'jobs_selected': jobs_selected},
        'throughput': {
            # `measured` is true only when every coefficient came from data. A
            # machine consumer that reads this field was previously told `true`
            # by a table with five of six coefficients documented.
            'measured': inv.measured,
            'partly_measured': inv.partly_measured,
            'documented_coefficients': inv.documented_coefficients,
            'warning': (
                None if inv.measured else
                ('PARTLY MEASURED: ' + ', '.join(inv.documented_coefficients)
                 + ' are documented, not fitted') if inv.partly_measured else
                'ESTIMATE, not measured: run --measure or --from-runs'),
            'meta': inv.table_meta,
            'models': {k: m.to_dict() for k, m in sorted(inv.models.items())},
            'anchor_measurement_load': ANCHOR_LOAD,
            'contention': CONTENTION,
        },
        'calibration': harvest_diag,
        'checks': checks or {},
        'tuned_stage': tuned_stage_cost(inv),
        'source_validity': {
            'gate': reg.SOURCE_VALIDITY_GATE,
            'ledger_present': inv.ledger_present,
            'replacements': [
                {'source_arm': arm, 'default_seed': default,
                 'replacement_seed': seed}
                for (arm, default), seed in sorted(inv.replacements.items())],
            'reserve_seeds_drawn': {arm: sorted(seeds) for arm, seeds
                                    in sorted(inv.replacement_draws.items())},
            'runs_on_rejected_sources': [
                {'label': r.label, 'seed': r.seed, 'run_dir': r.run_dir,
                 'source_arm': r.source_arm,
                 'source_default_seed': r.source_default_seed,
                 'source_score': r.source_score}
                for r in inv.rejected_sources],
            'runs_on_replaced_sources': [
                {'label': r.label, 'seed': r.seed, 'run_dir': r.run_dir,
                 'source_arm': r.source_arm,
                 'source_default_seed': r.source_default_seed,
                 'source_seed': r.source_seed}
                for r in inv.runs if r.source_replaced],
        },
        'experiments': [r.to_dict() for r in inv.experiments],
        'totals': {
            'naive_per_experiment_sum': inv.naive_total,
            'deduplicated': inv.total,
            'saved_by_sharing': inv.naive_total - inv.total,
            'complete': inv.total - len(pending),
            'pending': len(pending),
            'shared_runs': len([r for r in inv.runs if len(r.experiments) > 1]),
        },
        'by_environment': inv.by('env'),
        'by_condition': inv.by('condition'),
        'by_cell': inv.by('cell'),
        'by_seed_block': inv.by('seed_block'),
        'projections': projections(pending, ladder),
        'disk': {
            'durable_mb_per_run': DURABLE_MB_PER_RUN,
            'durable_gb': round(inv.total * DURABLE_MB_PER_RUN / 1024, 3),
            'transient_buffer_mb_per_copy': TRANSIENT_BUFFER_MB,
            'live_buffer_copies_per_run': copies,
            'transient_peak_mb': workers * copies * TRANSIENT_BUFFER_MB,
            'measured': disk_measured,
        },
        'model_check': residuals,
        'runs': [r.to_dict() for r in inv.runs],
        'warnings': inv.warnings,
        'plans': provenance.plan_hashes(),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _coerce(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    if ',' in text:
        return tuple(_coerce(part) for part in text.split(',') if part.strip())
    return text


def parse_overrides(items: Optional[Sequence[str]]) -> dict:
    """`--override key=value`, typed by JSON wherever JSON can type it.

    A field whose `Config` default is a tuple gets its value wrapped in one, so
    `--override prefix_checkpoints=500` means the same as `=500,` rather than
    handing the config an integer to iterate over.
    """
    tuple_fields = {f.name for f in dataclasses.fields(Config)
                    if isinstance(f.default, tuple)}
    known = {f.name for f in dataclasses.fields(Config)}
    out: dict = {}
    for item in items or ():
        if '=' not in item:
            raise SystemExit(f'--override needs key=value, got {item!r}')
        key, _, val = item.partition('=')
        key = key.strip().replace('-', '_')
        if key not in known:
            raise SystemExit(f'--override {key!r} is not a Config field; '
                             f'refusing rather than silently ignoring it')
        value = _coerce(val.strip())
        if key in tuple_fields and not isinstance(value, tuple):
            value = () if value == '' else (value,)
        out[key] = value
    return out


def activate_tuned_stage(out_root: str) -> tuple[object, Optional[str]]:
    """Install E1t and E2t where the tree holds a selection. Never refuses.

    Implicit, unlike `sweep.py`, and the asymmetry is the point. Costing a stage
    launches nothing and writes nothing, so the risk `sweep.py` guards against
    (a catalogue that silently grew two confirmatory experiments and 280 runs)
    does not exist here; the risk that does exist is the opposite one, a plan
    that prices a campaign at 125 h while the tree already holds the selection
    that makes the real figure about 156 h (`ANALYSIS_PLAN.md` 6.6). A cost
    model that under-reports because nobody passed a flag is the failure this
    file exists to prevent.

    So a selection is picked up wherever one is resolvable, its absence is
    silent, and its presence is printed. `select` is what refuses, and only when
    a tuned id was named explicitly, since that is the one case where continuing
    would answer a question nobody asked.
    """
    if not out_root:
        return None, 'no --out-root, so there is no tree to read a selection from'
    try:
        reg.activate_tuned_arms(out_root=out_root)
    except tuning.SelectionMissing as exc:
        return None, str(exc)
    except tuning.SelectionError as exc:
        return None, (f'the selection stored at '
                      f'{tuning.selection_path(out_root)} cannot be used: '
                      f'{exc}')
    except ValueError as exc:
        return None, f'the tuned arms cannot be built from it: {exc}'
    return reg.active_selection(), None


def select(args) -> list[str]:
    if args.experiments:
        unknown = [e for e in args.experiments if e not in reg.EXPERIMENTS]
        dormant = [e for e in unknown if e in reg.TUNED_OF]
        if dormant:
            raise SystemExit(
                f'{", ".join(dormant)}: E1 and E2 under the secondary tuning '
                f'policy of DESIGN.md 3.3, which cannot be priced because '
                f'{args.out_root} holds no selection to build them from. '
                f'{tuning.missing_message(args.out_root)}')
        if unknown:
            raise SystemExit(f'unknown experiment(s): {", ".join(unknown)}; '
                             f'known: {", ".join(reg.EXPERIMENTS)}')
        return list(args.experiments)
    if args.tier:
        tiers = set(args.tier)
        chosen = [e.id for e in reg.EXPERIMENTS.values() if e.tier in tiers]
        if not chosen:
            raise SystemExit(f'no experiments at tier(s) {sorted(tiers)}; '
                             f'tiers present: '
                             f'{sorted({e.tier for e in reg.EXPERIMENTS.values()})}')
        return chosen
    if args.all:
        return list(reg.EXPERIMENTS)
    return ['E1']


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--experiments', nargs='+', default=None,
                   help='experiment ids, e.g. E1 E2 (default: E1)')
    p.add_argument('--tier', nargs='+', type=int, default=None,
                   help='every experiment at these tiers')
    p.add_argument('--all', action='store_true', help='the whole catalogue')
    p.add_argument('--seeds', nargs='+', default=None,
                   help="a block name (CONFIRM, REPLICATE, TUNE, C4SRC, "
                        "RESERVE, SMOKE) or an explicit spec such as '0-9' or "
                        "'0 1 2'. Default: each experiment's declared block")
    p.add_argument('--jobs', type=int, default=None,
                   help='parallel workers to plan for; the 1/2/4/6/8 ladder is '
                        'printed regardless')
    p.add_argument('--out-root', default='runs',
                   help='run tree, read to count what is already complete')
    p.add_argument('--override', action='append', default=None,
                   help='config override applied to every job, key=value, '
                        'repeatable (e.g. --override num_episodes=14)')
    p.add_argument('--episodes', type=int, default=None,
                   help='shorthand for --override num_episodes=N')
    p.add_argument('--keep-buffer', action='store_true',
                   help='price the replay buffers as durable, not transient')
    p.add_argument('--list-runs', action='store_true',
                   help='print every run with its seed block (automatic at '
                        '40 runs or fewer)')
    p.add_argument('--json', action='store_true',
                   help='machine-readable output, including every run')
    p.add_argument('--measure', action='store_true',
                   help='measure throughput on this machine with a short real '
                        'run per (env, arch) in the selection, and write '
                        'experiments/throughput.json')
    p.add_argument('--measure-episodes', type=int, default=25)
    p.add_argument('--measure-root', default=CALIBRATION_ROOT,
                   help='where calibration runs go; deliberately outside runs/, '
                        'which audit.py globs recursively')
    p.add_argument('--measure-seed', type=int, default=900,
                   help='outside every declared seed block, so a calibration '
                        'run can never be mistaken for data')
    p.add_argument('--from-runs', default=None,
                   help='calibrate from a finished run tree instead of new '
                        'runs, and report predicted against actual wall time. '
                        'Any directory holding manifests works, including one '
                        'run directory')
    p.add_argument('--exclude', action='append', default=None, metavar='TEXT',
                   help='with --from-runs, exclude any run whose directory or '
                        'label contains TEXT, repeatable. The automatic outlier '
                        'gates run regardless; this is for a run you know to be '
                        'unrepresentative for a reason the rate cannot show')
    p.add_argument('--allow-legacy-configs', action='store_true',
                   help='with --from-runs, harvest manifests whose config '
                        'carries fields this version of Config no longer has, '
                        'dropping those fields. Off by default: a manifest '
                        'whose schema is unknown is a cost this file cannot '
                        'vouch for. The dropped fields are named in the report')
    p.add_argument('--allow-miscalibration', action='store_true',
                   help='write and use a calibration that fails its own '
                        'self-check, warning instead of refusing')
    p.add_argument('--throughput-file', default=THROUGHPUT_FILE)
    p.add_argument('--no-write', action='store_true',
                   help='with --measure or --from-runs, do not write the table')
    p.add_argument('--allow-factor-overrides', action='store_true',
                   help='let --override touch an experimental factor rather '
                        'than only the budget or the measurement cadence. The '
                        'registry refuses that by default, because a run whose '
                        'factors moved is not the arm its label claims')
    p.add_argument('--allow-tune-mixing', action='store_true',
                   help='warn but exit 0 when TUNE seeds reach a reported '
                        'experiment')
    args = p.parse_args(argv)

    # Argument validation, before anything is priced. Every refusal here is a
    # value the tool used to absorb: `--jobs 0` printed a ladder with no marker
    # on it and no complaint, `--jobs -3` printed the one-worker row twice
    # because the requested value sorted to the front and was then clamped,
    # `--episodes -5` priced a 74.8 m plan for a run that cannot exist, and
    # `--seeds NOPE` came out as a raw ValueError traceback from the registry
    # while `--experiments NOPE` had gone to the trouble of a clean message.
    if args.jobs is not None and args.jobs < 1:
        raise SystemExit(f'--jobs {args.jobs}: a schedule needs at least one '
                         f'worker. The ladder {JOB_LADDER} is printed anyway.')
    if args.measure_episodes < 1:
        raise SystemExit(f'--measure-episodes {args.measure_episodes}: a '
                         f'calibration run needs episodes to measure.')
    if args.episodes is not None and args.episodes < 1:
        raise SystemExit(f'--episodes {args.episodes}: a run of that many '
                         f'episodes cannot be launched, so pricing one would be '
                         f'a plan for something that will never happen.')

    # The tuned stage of DESIGN.md 3.3, before the selection is resolved:
    # activating it changes what --experiments, --tier and --all mean.
    selection, tuned_unavailable = activate_tuned_stage(args.out_root)
    exp_ids = select(args)
    seeds = ' '.join(args.seeds) if args.seeds else None
    if seeds is not None:
        try:
            resolved = reg.resolve_seeds(seeds, 'CONFIRM')
        except (ValueError, TypeError, KeyError) as exc:
            raise SystemExit(
                f'--seeds {seeds!r} is neither a block name nor a seed spec '
                f'({exc}). Blocks: {", ".join(reg.SEED_BLOCKS)}. A spec looks '
                f"like '0-9', '0 1 2' or '0-9 200-204'.")
        if not resolved:
            raise SystemExit(f'--seeds {seeds!r} resolves to no seeds at all.')
    overrides = parse_overrides(args.override)
    if args.episodes is not None:
        overrides['num_episodes'] = args.episodes
    budget = overrides.get('num_episodes')
    if isinstance(budget, int) and budget < 1:
        raise SystemExit(f'--override num_episodes={budget}: a run of that many '
                         f'episodes cannot be launched.')

    residuals: Optional[list[dict]] = None
    disk_measured: Optional[dict] = None
    harvest_diag: Optional[dict] = None
    checks: dict = {}
    fresh: dict = {}
    fresh_method = ''
    problems: list[str] = []
    exit_code = 0

    if args.from_runs:
        print(f'harvesting throughput from {args.from_runs} ...')
        fresh, residuals, harvest_diag = harvest(
            args.from_runs, args.exclude or (), args.allow_legacy_configs)
        checks['fit'] = fit_check(residuals)
        fresh_method = f'harvested from {args.from_runs}'
        if not fresh:
            # Falling through to the table on disk here is how `--from-runs
            # runs_demo` used to produce a full report, exit 0 and an empty
            # warnings list, headed `measured, harvested from runs`: a clean
            # bill of health on a table harvested from a different tree
            # altogether. The table is still shown, because the operator has to
            # see what they are about to be given, but the report says whose it
            # is and the exit code is not zero.
            problems.append(
                f'--from-runs {args.from_runs} produced no usable calibration '
                f'entries. Every projection below is priced from whatever was '
                f'already in {args.throughput_file}, which was calibrated on a '
                f'different tree, NOT from {args.from_runs}. The harvest '
                f'section says what was skipped and why.')
            exit_code = 1
        elif not checks['fit']['ok']:
            problems.append(
                'CALIBRATION SELF-CHECK FAILED: '
                + ' '.join(checks['fit']['problems'])
                + (' Written anyway, because --allow-miscalibration was passed.'
                   if args.allow_miscalibration else
                   ' Refusing to write it to disk. The fit is still used for '
                   'the report below so it can be inspected; re-run with '
                   '--allow-miscalibration to adopt it.'))
            if not args.allow_miscalibration:
                fresh_method += ' (failed its self-check, not written)'
                exit_code = 1
        if fresh and (checks['fit']['ok'] or args.allow_miscalibration):
            if args.no_write:
                fresh_method += ' (not written: --no-write)'
            else:
                write_throughput(fresh, args.throughput_file, fresh_method)
        disk_measured = measure_disk(args.from_runs)
    elif args.out_root:
        disk_measured = measure_disk(args.out_root)

    if args.measure:
        # Only the (env, arch) pairs this selection actually needs, so a
        # calibration is never more expensive than the plan it prices.
        table_now, _ = load_throughput(args.throughput_file)
        probe = build_inventory(exp_ids, seeds, args.out_root, overrides,
                                table_now, {}, args.allow_factor_overrides,
                                selection=selection)
        keys = sorted({(envs.parse(r.env).canonical(), r.arch)
                       for r in probe.runs})
        print(f'measuring {len(keys)} (env, arch) pair(s) at '
              f'{args.measure_episodes} episodes into {args.measure_root} ...')
        fresh = measure(keys, args.measure_episodes, args.measure_root,
                        args.measure_seed)
        fresh_method = f'measured, {args.measure_episodes} episodes/pair'
        if args.no_write:
            fresh_method += ' (not written: --no-write)'
        else:
            write_throughput(fresh, args.throughput_file, fresh_method)

    table, meta = load_throughput(args.throughput_file)
    if fresh:
        # The fresh fit prices the report whether or not it was written. With
        # --no-write the tool used to print a MODEL CHECK computed from the new
        # fit beside a COST MODEL and a wall clock computed from the stale table
        # on disk: two different models in one report, and --no-write is exactly
        # the flag a careful reviewer uses to look at a fit before adopting it.
        table = {**table, **fresh}
        meta = {'method': fresh_method,
                'written': time.strftime('%Y-%m-%dT%H:%M:%S')}

    inv = build_inventory(exp_ids, seeds, args.out_root, overrides, table, meta,
                          args.allow_factor_overrides, selection=selection)
    if selection is None and any(eid in reg.TUNED_OF for eid in exp_ids):
        # Unreachable through `select`, which refuses first. Here because a
        # caller reaching `main` with a tuned id and no selection would
        # otherwise price a stage it could not have enumerated.
        raise SystemExit(f'internal: {exp_ids} names a tuned experiment with no '
                         f'selection. {tuned_unavailable}')
    checks['anchors'] = anchor_check(table)
    for row in checks['anchors']:
        if not row['ok']:
            problems.append(
                f'ANCHOR SELF-CHECK, {row["env"]} {row["arch"]}: the model in '
                f'force implies {hms(row["seconds"])} for '
                f'{row["episodes"]} episodes against a documented band of '
                f'{hms(row["band"][0])} to {hms(row["band"][1])}. '
                + row['verdict'])
            # A calibrated model outside the band warns; it does not stop a
            # launch, because a stale line of prose is not a reason to refuse to
            # plan. A model built from the anchors that contradicts those same
            # anchors is a defect in this file and does stop.
            if row['source'] not in ('measured', 'harvested'):
                exit_code = 1
    inv.warnings.extend(problems)

    if args.json:
        print(json.dumps(to_json(inv, args.jobs, disk_measured, residuals,
                                 harvest_diag, checks),
                         indent=2, sort_keys=True, default=str))
    else:
        report(inv, args.jobs, args.list_runs, args.keep_buffer, disk_measured,
               residuals, harvest_diag, checks)

    if any(w.startswith('TUNE SEEDS REACH') for w in inv.warnings) \
            and not args.allow_tune_mixing:
        exit_code = 1
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
