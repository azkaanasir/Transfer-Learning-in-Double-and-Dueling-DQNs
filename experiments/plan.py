"""Cost and inventory model: what a selection costs, before anything launches.

    python experiments/plan.py --experiments E1 E2 --seeds CONFIRM
    python experiments/plan.py --tier 1
    python experiments/plan.py --all --jobs 6
    python experiments/plan.py --measure            # measure this machine
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
* **Measured versus estimated throughput.** The published study's compute budget
  was reconstructed after the fact and could not be checked. Here the cost model
  is either measured on this machine and written to `experiments/throughput.json`
  or it is documented defaults -- and in the second case every projection is
  stamped `ESTIMATE, not measured`. The model's implied values for the two
  documented anchors are printed next to the anchors themselves, so a
  mis-calibration is visible rather than latent.
* **Disk.** The replay buffer is ~5.3 MB per copy and a run holds one copy per
  live checkpoint, so the transient peak scales with `--jobs` and with
  `prefix_checkpoints`, not with the total run count. Durable artifacts are
  ~0.6 MB per run. Confusing the two is how a sweep fills a disk at hour six.

The cost model itself is two coefficients per (environment, architecture) -- a
per-episode overhead and a per-env-step rate -- over an episode-length ramp.
Episodes are not the cost unit: LunarLander episode length ranges over an order
of magnitude with performance (`DESIGN.md` §3.2), which is exactly why the
documented per-episode figure rises from ~0.22 s to ~1.2 s within one run. A
model in env steps tracks that, prices the shift variants from the *measured*
random-policy episode lengths in `reference_returns.json` rather than assuming
they cost what the base environment costs, and can be checked against a finished
run tree with `--from-runs`.

Nothing here decides anything scientific. It decides what to launch and in what
order, and it prints the seed-block ledger that makes a selection auditable
before it becomes data.
"""
from __future__ import annotations

import argparse
import json
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
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from experiments import registry as reg                          # noqa: E402
from src.dqn import envs, provenance                             # noqa: E402
from src.dqn.config import Config                                # noqa: E402

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
# Measured on the dev machine (8 cores, 2 threads per process, so four resident
# workers -- see CONTENTION):
#   * CartPole-v1     ~0.19 s per episode averaged over a 500-episode run
#   * LunarLander-v3  ~0.22 s per episode early, rising to ~1.2 s per episode
#                     once episodes lengthen; a full 1000-episode run is roughly
#                     15-25 minutes
#
# Those are averages over whole runs, so they already include the evaluation
# cadence, the held-out evaluations and the linear probe. `measurement_load`
# below is what rescales them when a run changes that cadence.
#
# `s_per_env_step` is quoted for the dueling architecture and scaled to `mlp` by
# ARCH_FACTOR. `steps_end` is the episode length a developed policy reaches: it
# rises on CartPole and LunarLander and *falls* on Acrobot, where solving the
# task means terminating sooner.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Anchor:
    s_per_env_step: float          # dueling, at the anchor's contention level
    s_per_episode: float           # per-episode overhead: reset, logging, I/O
    steps_start: float             # fallback only; a measured value wins
    steps_end: float
    ramp_episodes: int
    note: str


DEFAULT_ANCHORS: dict[str, Anchor] = {
    'LunarLander-v3': Anchor(
        s_per_env_step=0.0021, s_per_episode=0.023,
        steps_start=94.0, steps_end=560.0, ramp_episodes=300,
        note='documented anchor: 0.22 s/ep early rising to ~1.2 s/ep, '
             '15-25 min per 1000-episode run'),
    'CartPole-v1': Anchor(
        s_per_env_step=0.0016, s_per_episode=0.020,
        steps_start=22.5, steps_end=150.0, ramp_episodes=400,
        note='documented anchor: ~0.19 s/ep averaged over 500 episodes. '
             'steps_end is 150 rather than the 500-step cap because epsilon '
             'anneals over 300k env steps and a whole 1000-episode CartPole run '
             'accumulates only about 100k, so these episodes never stop being '
             'partly exploratory'),
    'Acrobot-v1': Anchor(
        s_per_env_step=0.0016, s_per_episode=0.020,
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

# From the demo run tree: mean seconds per episode was 0.299 (mlp) against 0.438
# (dueling) on LunarLander and 0.080 against 0.126 on CartPole -- ratios of 0.68
# and 0.63. The double-Q target adds one extra forward pass through the online
# network per update: 0.448 against 0.429 (dueling), 0.302 against 0.295 (mlp),
# i.e. about 3%.
ARCH_FACTOR = {'dueling': 1.00, 'mlp': 0.68}
RULE_FACTOR = {'vanilla': 1.00, 'double': 1.03}

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
# enforces it -- SMOKE was `(0,)` until recently, which put every
# pipeline-validation run inside CONFIRM -- so an overlap resolves by this
# precedence and by preferring the block the experiment itself declares, rather
# than by dictionary order.
BLOCK_PRECEDENCE = ('CONFIRM', 'REPLICATE', 'TUNE', 'C4SRC', 'RESERVE', 'SMOKE')

# Families whose runs feed a reported estimate. A TUNE seed reaching one of these
# is the revision-1 selection leak (`DESIGN.md` §3.4).
REPORTED_FAMILIES = ('confirmatory', 'estimation')


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------
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

    @property
    def measured(self) -> bool:
        return self.source in ('measured', 'harvested')

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

        The cap matters and is not decoration: the developed-policy length
        modelled for `gravity=-4` is 1087 steps, above the 1000-step episode
        limit the training loop enforces, so without the truncation the most
        expensive arm in the catalogue would be over-priced by about 9%.
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
                'length_scale': self.length_scale, 'note': self.note}


def measurement_load(cfg: Config) -> float:
    """Evaluation burden of one run, relative to its training episodes.

    The anchors are whole-run averages at the default cadence, and the default
    cadence is not cheap: 100 monitoring evaluations of 5 episodes, 100 held-out
    episodes at each of three final checkpoints and each of three prefix
    checkpoints, and 100 more for the zero-shot jumpstart -- 1200 evaluation
    episodes against 1000 training episodes. A run that changes `eval_every`,
    `final_eval_episodes` or `prefix_checkpoints` therefore changes its cost for
    reasons that have nothing to do with its episode budget, and E0's smoke
    configuration changes all three. This is the term that tracks that.
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


def cost_model(env: str, arch: str, target_rule: str,
               table: dict[str, dict]) -> CostModel:
    """Resolve one (env, arch, rule) to a cost model, measured table first."""
    spec = envs.parse(env)
    env_id, canonical = spec.env_id, spec.canonical()

    entry = table.get(f'{canonical}|{arch}') or table.get(f'{env_id}|{arch}')
    if entry:
        model = CostModel(key=f'{canonical}|{arch}', **{
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
        model = CostModel(
            key=f'{canonical}|{arch}',
            s_per_env_step=anchor.s_per_env_step * ARCH_FACTOR.get(arch, 1.0),
            s_per_episode=anchor.s_per_episode,
            steps_start=anchor.steps_start, steps_end=anchor.steps_end,
            ramp_episodes=anchor.ramp_episodes, startup_s=DEFAULT_STARTUP_S,
            source=source, note=anchor.note)

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


def seed_block_of(seed: int, declared: Optional[str] = None) -> str:
    """Which block a seed belongs to, preferring the experiment's own.

    The blocks are meant to be disjoint, and when they are this is a lookup.
    When they are not -- SMOKE was declared as `(0,)` until recently, inside
    CONFIRM's range -- an unqualified lookup is ambiguous, and preferring the
    declaring experiment's block resolves it the way the design means it: that
    seed under E0 is a SMOKE run, the same seed under E1 is a CONFIRM run.
    """
    if declared and seed in reg.SEED_BLOCKS.get(declared, ()):
        return declared
    for name in BLOCK_PRECEDENCE:
        if seed in reg.SEED_BLOCKS.get(name, ()):
            return name
    return 'UNKNOWN'


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


def is_complete(run_dir: str, episodes: int) -> bool:
    """A run counts as done when its manifest reports the full episode budget."""
    try:
        with open(os.path.join(run_dir, 'manifest.json'), encoding='utf-8') as fh:
            result = json.load(fh).get('result') or {}
    except (OSError, json.JSONDecodeError):
        return False
    return int(result.get('episodes_completed') or 0) >= episodes


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

    @property
    def naive_total(self) -> int:
        return sum(r.runs_alone for r in self.experiments)

    @property
    def total(self) -> int:
        return len(self.runs)

    @property
    def pending(self) -> list[RunRecord]:
        return [r for r in self.runs if not r.complete]

    @property
    def measured(self) -> bool:
        return bool(self.models) and all(m.measured for m in self.models.values())

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
                    allow_factor_overrides: bool = False) -> Inventory:
    """Resolve a selection into runs, costs, seed blocks and sharing savings."""
    warnings: list[str] = []
    if table_meta.get('error'):
        warnings.append(table_meta['error'])

    models: dict[str, CostModel] = {}

    def model_for(cfg: Config) -> CostModel:
        key = f'{cfg.env}|{cfg.arch}|{cfg.target_rule}'
        if key not in models:
            models[key] = cost_model(cfg.env, cfg.arch, cfg.target_rule, table)
        return models[key]

    rows: list[ExperimentRow] = []
    records: dict[str, RunRecord] = {}
    for eid in exp_ids:
        exp = reg.EXPERIMENTS[eid]
        # `jobs` can emit two arms that resolve to one configuration -- E7's
        # aggregation='mean' scratch arm is E1's scratch arm -- so the count that
        # matters is over run keys, not over jobs.
        alone = {job.key(): job
                 for job in reg.jobs(eid, seeds, out_root, overrides,
                                     allow_factor_overrides)}
        before = len(records)
        seconds_alone = 0.0
        for key, job in alone.items():
            cfg = job.cfg
            seconds = model_for(cfg).seconds(cfg)
            seconds_alone += seconds
            rec = records.get(key)
            if rec is None:
                records[key] = RunRecord(
                    key=key, run_dir=cfg.run_dir(), label=cfg.label,
                    role=job.role, experiments=(eid,), families=(exp.family,),
                    condition=cfg.condition,
                    cell=f'{cfg.arch}-{cfg.target_rule}', env=cfg.env,
                    arch=cfg.arch, target_rule=cfg.target_rule, seed=cfg.seed,
                    seed_block=seed_block_of(cfg.seed, exp.seed_block),
                    episodes=cfg.num_episodes, depends_on=job.depends_on,
                    seconds=seconds,
                    complete=is_complete(cfg.run_dir(), cfg.num_episodes))
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
                                                   allow_factor_overrides)}
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

    inferred = [k for k, m in models.items() if m.source == 'inferred']
    if inferred:
        warnings.append('no throughput anchor for ' + ', '.join(sorted(inferred))
                        + ' -- cost inferred from another environment')

    return Inventory(experiments=rows, runs=runs, seeds_spec=seeds,
                     out_root=out_root, table=table, table_meta=table_meta,
                     models=models, warnings=warnings)


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
        residual = wall / measurement_load(cfg) - anchor.s_per_episode * done
        rate = max(1e-6, residual / steps) if steps > 0 else anchor.s_per_env_step
        updates = int(result.get('updates') or 0)
        source, detail = 'measured', ''
        if updates < 0.5 * steps:
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
        entries[f'{envs.parse(env).canonical()}|{arch}'] = {
            's_per_env_step': round(rate, 6),
            's_per_episode': anchor.s_per_episode,
            'steps_start': measured_steps_start(env) or anchor.steps_start,
            'steps_end': anchor.steps_end,
            'ramp_episodes': anchor.ramp_episodes,
            'startup_s': startup if startup is not None else DEFAULT_STARTUP_S,
            'source': source,
            'note': f'{detail}step rate over {done} episodes / {steps:.0f} env '
                    f'steps / {updates} updates in {wall:.1f} s '
                    f'({time.time() - t0:.0f} s including build); per-episode '
                    f'overhead and length ramp remain documented',
        }
        print(f'    {wall:.1f} s, {steps:.0f} env steps -> '
              f'{rate * 1000:.3f} ms/step', flush=True)
    return entries


def harvest(root: str) -> tuple[dict, list[dict]]:
    """Fit the two coefficients from a finished run tree, and report residuals.

    A real tree is a better calibration source than a fresh short run: it spans
    architectures, conditions and environments, and it was produced by the code
    that will produce the sweep. The fit is a non-negative least squares of
    `wall_time_s / measurement_load = a * episodes + b * env_steps`, which is
    identifiable as soon as the group's runs differ in either coordinate.
    """
    import glob

    import numpy as np
    from scipy.optimize import nnls

    groups: dict[tuple[str, str], list[dict]] = {}
    for path in sorted(glob.glob(os.path.join(root, '*', '*', 's*',
                                              'manifest.json'))):
        try:
            with open(path, encoding='utf-8') as fh:
                man = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        cfg_d = man.get('config') or {}
        res = man.get('result') or {}
        eps = int(res.get('episodes_completed') or 0)
        wall, steps = res.get('wall_time_s'), res.get('env_steps')
        if not (eps and wall and steps):
            continue
        try:
            cfg = Config(**cfg_d)
        except Exception:                                    # noqa: BLE001
            continue
        rule_f = RULE_FACTOR.get(cfg.target_rule, 1.0)
        groups.setdefault((envs.parse(cfg.env).env_id, cfg.arch), []).append(
            {'episodes': eps, 'env_steps': float(steps),
             'wall': float(wall) / rule_f, 'actual': float(wall),
             'load': measurement_load(cfg), 'run_dir': os.path.dirname(path),
             'env': cfg.env, 'arch': cfg.arch, 'rule': cfg.target_rule,
             'cfg': cfg})

    entries: dict[str, dict] = {}
    residuals: list[dict] = []
    for (env_id, arch), rows in sorted(groups.items()):
        anchor = DEFAULT_ANCHORS.get(env_id, DEFAULT_ANCHORS['LunarLander-v3'])
        design = np.array([[r['episodes'], r['env_steps']] for r in rows], float)
        y = np.array([r['wall'] / r['load'] for r in rows], float)
        # Identifiability guard, and it is load-bearing rather than decorative.
        # A per-episode overhead can only be told apart from a per-step rate if
        # the harvested runs differ materially in steps *per episode*. In a
        # smoke tree they do not -- every run has the same episode budget and
        # similar episode lengths -- and an unconstrained fit then hands the
        # whole cost to whichever column the noise favours. Measured here: the
        # unguarded fit gave the LunarLander mlp group 0.178 s/episode and a
        # step rate of 0.0001 ms, which extrapolated to a 1000-episode run is a
        # four-fold under-estimate, in the direction nobody notices until the
        # sweep overruns. So both coefficients are fitted only when the episode
        # budgets span a factor of three and the steps-per-episode varies;
        # otherwise the overhead is held documented and only the rate is fitted,
        # which is the conservative direction (the overhead is a few per cent of
        # a full run's cost, the rate is nearly all of it).
        eps_col, step_col = design[:, 0], design[:, 1]
        per_ep = step_col / np.maximum(1.0, eps_col)
        spread = float(per_ep.std() / per_ep.mean()) if per_ep.mean() else 0.0
        budget_span = float(eps_col.max() / max(1.0, eps_col.min()))
        identified = (len(rows) >= 5 and np.linalg.matrix_rank(design) == 2
                      and budget_span >= 3.0 and spread > 0.15)
        if identified:
            coef, _ = nnls(design, y)
            a, b = float(coef[0]), float(coef[1])
            fit = (f'nnls over {len(rows)} runs spanning a {budget_span:.1f}x '
                   f'episode-budget range')
        else:
            a = anchor.s_per_episode
            b = float(max(1e-6, (y - a * eps_col).sum() / step_col.sum()))
            fit = (f'{len(rows)} run(s) spanning only a {budget_span:.1f}x '
                   f'episode-budget range (steps/episode spread '
                   f'{spread:.2f}), so the two coefficients are not separately '
                   f'identified: the per-episode overhead is held at its '
                   f'documented value and only the step rate is fitted')
        entries[f'{env_id}|{arch}'] = {
            's_per_env_step': round(b, 6), 's_per_episode': round(a, 6),
            'steps_start': measured_steps_start(env_id) or anchor.steps_start,
            'steps_end': anchor.steps_end, 'ramp_episodes': anchor.ramp_episodes,
            'startup_s': DEFAULT_STARTUP_S, 'source': 'harvested',
            'note': f'{fit}; startup and the length ramp remain documented, '
                    f'because a finished short run shows neither',
        }
        # Predicted against actual, on the runs the fit was made from. The
        # harvested runs' own observed mean episode length is used, so this
        # checks the rate rather than the ramp -- and it is reported even when it
        # is unflattering, which is the point of printing it.
        for r in rows:
            observed_len = r['env_steps'] / r['episodes']
            check = CostModel(key=f'{env_id}|{arch}',
                              s_per_env_step=b * RULE_FACTOR.get(r['rule'], 1.0),
                              s_per_episode=a, steps_start=observed_len,
                              steps_end=observed_len, ramp_episodes=0,
                              startup_s=0.0, source='harvested')
            pred = check.seconds(r['cfg'])
            residuals.append({
                'run_dir': r['run_dir'], 'env': r['env'], 'arch': r['arch'],
                'predicted_s': round(pred, 2), 'actual_s': round(r['actual'], 2),
                'ratio': round(pred / r['actual'], 3) if r['actual'] else None})
    return entries, residuals


def measure_disk(root: str) -> Optional[dict]:
    """Durable megabytes per completed run, measured from a real tree."""
    import glob
    sizes = []
    for path in glob.glob(os.path.join(root, '*', '*', 's*', 'manifest.json')):
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
    """Merge new entries into the calibration table on disk."""
    existing, _meta = load_throughput(path)
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


def report(inv: Inventory, jobs_selected: Optional[int], list_runs: bool,
           keep_buffer: bool, disk_measured: Optional[dict],
           residuals: Optional[list[dict]]) -> None:
    est = not inv.measured
    print(rule('='))
    print('COST AND INVENTORY PLAN -- this file launches nothing')
    print(rule('='))
    print(f'experiments : {", ".join(r.id for r in inv.experiments)}')
    print(f'seeds       : {inv.seeds_spec or "each experiment\'s declared block"}')
    print(f'out-root    : {inv.out_root}')
    if est:
        print('throughput  : *** ESTIMATE, NOT MEASURED *** documented anchors; '
              'run --measure')
    else:
        print(f'throughput  : measured -- {inv.table_meta.get("method", "?")}, '
              f'{inv.table_meta.get("written", "?")}')
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
    print('COST MODEL' + ('   *** ESTIMATE, NOT MEASURED ***' if est else ''))
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
    for note in sorted({m.note for m in inv.models.values() if m.note}):
        wrapped(note, indent='  * ')
    print('\n  self-check -- what the model in force implies for the two '
          'documented anchors:')
    for env, arch, episodes, anchor_text in (
            ('CartPole-v1', 'dueling', 500, '~0.19 s/episode over 500 episodes'),
            ('LunarLander-v3', 'dueling', 1000, '15-25 min for the whole run')):
        model = cost_model(env, arch, 'vanilla', inv.table)
        secs = model.seconds(Config(arch=arch, env=env, num_episodes=episodes))
        print(f'    {env:16s} {arch:8s} {episodes:5d} ep -> {hms(secs):>7s} '
              f'({secs / episodes:.3f} s/ep) {model.source:>10s}  anchor: '
              f'{anchor_text}')
    print(f'  measurement load at the default cadence: {ANCHOR_LOAD:.2f} '
          f'evaluation-weighted\n  episodes per training episode; a run that '
          f'changes the cadence is rescaled.')

    # -- wall clock ------------------------------------------------------
    pending = inv.pending
    print('\n' + rule())
    print('PROJECTED WALL-CLOCK'
          + ('   (on ESTIMATED throughput)' if est else ''))
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
        ratios = [r['ratio'] for r in residuals if r['ratio']]
        print('\n' + rule())
        print('MODEL CHECK against finished runs')
        print(rule())
        if ratios:
            print(f'  {len(ratios)} runs; predicted/actual wall time median '
                  f'{statistics.median(ratios):.2f}, range '
                  f'{min(ratios):.2f}-{max(ratios):.2f}')
            for r in sorted(residuals,
                            key=lambda r: -abs((r['ratio'] or 1) - 1))[:5]:
                print(f'    {r["ratio"]:.2f}  predicted {r["predicted_s"]:8.1f}s '
                      f' actual {r["actual_s"]:8.1f}s  {r["env"][:28]:28s} '
                      f'{r["arch"]}')

    # -- what is not modelled -------------------------------------------
    print('\n' + rule())
    print('NOT MODELLED, AND KNOWN TO BE MISSING')
    print(rule())
    for line in (
            'source-validity rejections: a rejected source draws a RESERVE seed '
            'and adds a run (DESIGN.md 4.3), so every count here is a floor, '
            'not a cap',
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
            residuals: Optional[list[dict]]) -> dict:
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
            'measured': inv.measured,
            'warning': (None if inv.measured
                        else 'ESTIMATE, not measured -- run --measure'),
            'meta': inv.table_meta,
            'models': {k: m.to_dict() for k, m in sorted(inv.models.items())},
            'anchor_measurement_load': ANCHOR_LOAD,
            'contention': CONTENTION,
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
    import dataclasses
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


def select(args) -> list[str]:
    if args.experiments:
        unknown = [e for e in args.experiments if e not in reg.EXPERIMENTS]
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
                        'runs, and report predicted against actual wall time')
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

    exp_ids = select(args)
    seeds = ' '.join(args.seeds) if args.seeds else None
    overrides = parse_overrides(args.override)
    if args.episodes is not None:
        overrides['num_episodes'] = args.episodes

    residuals: Optional[list[dict]] = None
    disk_measured: Optional[dict] = None

    if args.from_runs:
        print(f'harvesting throughput from {args.from_runs} ...')
        entries, residuals = harvest(args.from_runs)
        if not entries:
            print(f'  no usable runs under {args.from_runs}')
        elif not args.no_write:
            write_throughput(entries, args.throughput_file,
                             f'harvested from {args.from_runs}')
        disk_measured = measure_disk(args.from_runs)
    elif args.out_root:
        disk_measured = measure_disk(args.out_root)

    if args.measure:
        # Only the (env, arch) pairs this selection actually needs, so a
        # calibration is never more expensive than the plan it prices.
        table_now, _ = load_throughput(args.throughput_file)
        probe = build_inventory(exp_ids, seeds, args.out_root, overrides,
                                table_now, {}, args.allow_factor_overrides)
        keys = sorted({(envs.parse(r.env).canonical(), r.arch)
                       for r in probe.runs})
        print(f'measuring {len(keys)} (env, arch) pair(s) at '
              f'{args.measure_episodes} episodes into {args.measure_root} ...')
        entries = measure(keys, args.measure_episodes, args.measure_root,
                          args.measure_seed)
        if not args.no_write:
            write_throughput(entries, args.throughput_file,
                             f'measured, {args.measure_episodes} episodes/pair')

    table, meta = load_throughput(args.throughput_file)
    inv = build_inventory(exp_ids, seeds, args.out_root, overrides, table, meta,
                          args.allow_factor_overrides)

    if args.json:
        print(json.dumps(to_json(inv, args.jobs, disk_measured, residuals),
                         indent=2, sort_keys=True, default=str))
    else:
        report(inv, args.jobs, args.list_runs, args.keep_buffer, disk_measured,
               residuals)

    if any(w.startswith('TUNE SEEDS REACH') for w in inv.warnings) \
            and not args.allow_tune_mixing:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
