"""The experiment catalogue: the declarative contract everything else reads.

`DESIGN.md` §7 lists the experiments; this file *is* them, in a form the runner,
the auditor, the statistics and the report all read from one place. That
single-source property is the point. In the published study the four arms lived
in four copied packages whose configs had silently drifted apart, and the
manuscript's description of them was reconstructed months later by diffing
checkpoints.

Three things are declared per experiment and are load-bearing downstream:

* **`invariants`** -- the fields that must be *identical* across the
  experiment's runs. This is what turns "identical hyperparameters" from a
  sentence in a paper into a machine-checked fact (`audit.py`).
* **`seed_block`** -- which disjoint seed block the experiment draws on.
  Hyperparameter selection runs on `TUNE` and no reported estimate may touch it.
* **`family`** -- `confirmatory`, `screen` or `estimation`. Only the confirmatory
  family carries p-values, per `ANALYSIS_PLAN.md` §2; `stats.py` reads this
  rather than accepting a family as an argument, so a result cannot be rescued
  after the fact by relocating it into a family of one.

Arms name their source by *label*, and `jobs()` resolves that into a checkpoint
path and emits jobs in dependency order, so a transfer run can never silently
load a source that does not exist or belongs to another cell.
"""
from __future__ import annotations

import dataclasses
import itertools
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dqn.config import Config                                # noqa: E402

# ---------------------------------------------------------------------------
# Seed blocks. Disjoint by construction; `audit.py` refuses a reported estimate
# that draws on TUNE, because revision 1 of the design selected hyperparameters
# on seeds 0-4 and then ran every confirmatory arm on 0-9, so half of each
# confirmatory sample had been tuned on.
# ---------------------------------------------------------------------------
SEED_BLOCKS: dict[str, tuple[int, ...]] = {
    'CONFIRM': tuple(range(0, 10)),
    'REPLICATE': tuple(range(10, 20)),
    'TUNE': tuple(range(200, 205)),
    'C4SRC': tuple(range(300, 310)),
    'RESERVE': tuple(range(400, 420)),
    # Disjoint from CONFIRM on purpose. With SMOKE=(0,) a pipeline-validation
    # run was attributed to the confirmatory block, so a seed-block audit could
    # not tell a smoke run from a real one by its seed alone.
    'SMOKE': (999,),
}

#: `DESIGN.md` §4.3. A source is valid when its **normalised** final score on
#: the source environment reaches this gate. Normalised, not multiplicative:
#: at Acrobot's registered threshold of -100 a "0.6 x threshold" rule reads -60,
#: which is harder than solving the task, and it was that malformed gate which
#: let the published study transfer from a CartPole agent scoring 26.94 out of
#: 475 without anything noticing.
#:
#: The number is repeated from `src/dqn/train.py`, which stamps the verdict into
#: every transfer run's manifest under ``source.validity``, rather than imported
#: from it: the runner has to apply the gate *before* any dependent run exists,
#: and importing the trainer would pull TensorFlow into the planner. The two
#: must agree, and `sweep.py` prints the value it used into every rejection
#: record so a disagreement is visible in the data rather than only in the code.
SOURCE_VALIDITY_GATE = 0.6

#: The order replacement source seeds are drawn in, fixed here so that the draw
#: is a property of the catalogue rather than of when a sweep happened to run.
#: `DESIGN.md` §4.3 says "drawn in order from RESERVE", and in order is what
#: makes the assignment reproducible after an interruption.
RESERVE_ORDER: tuple[int, ...] = SEED_BLOCKS['RESERVE']

CELLS = tuple(itertools.product(('mlp', 'dueling'), ('vanilla', 'double')))

SOURCE_ENV = 'CartPole-v1'
TARGET_ENV = 'LunarLander-v3'
INTERFACE_ENV = 'LunarLander-v3:pad_obs=4,extra_actions=2,pad_mode=noise'

# The protocol under study, shared by every arm that does not deliberately vary
# it. Freezing is in gradient updates (DESIGN.md §3.2).
PROTOCOL = dict(transfer_set='matched', input_policy='partial',
                head_policy='reinit', freeze_group='trunk',
                freeze_updates=10_000)

# Optimisation settings shared by every cell under the *primary* tuning policy
# (DESIGN.md §3.3). E3 is the only experiment that varies them.
COMMON = dict(lr=5e-4, target_update='hard', target_update_freq=1_000,
              gamma=0.99, batch_size=64, num_episodes=1000,
              epsilon_anneal_episodes=900, hidden=(128, 128), head_units=64)

# Fields audited for constancy across an experiment's runs unless the experiment
# declares that it varies one of them.
CORE_INVARIANTS = ('lr', 'gamma', 'batch_size', 'num_episodes', 'max_steps',
                   'epsilon_start', 'epsilon_min', 'epsilon_anneal_episodes',
                   'target_update', 'target_update_freq', 'replay_capacity',
                   'learning_starts', 'train_every', 'grad_clip_norm',
                   'hidden', 'head_units')


@dataclass(frozen=True)
class Arm:
    """One condition within an experiment, at every seed."""

    label: str
    overrides: dict = field(default_factory=dict)
    # Label of the arm supplying this arm's source checkpoint. Resolved to a
    # path at job-build time, always at the *same seed*, so cells and shift
    # levels cannot contaminate each other.
    source_from: Optional[str] = None
    # Draw the source from a different seed block, for the positive control
    # whose sources must be independent of the scratch runs used as denominators.
    source_seed_block: Optional[str] = None
    role: str = 'target'
    # When true the arm is built only when another arm names it as a source, at
    # the seed that reference resolves to -- not once per experiment seed. The
    # positive control's donor runs live in their own seed block, so enumerating
    # them over the experiment's block as well would train forty runs nothing
    # ever reads.
    only_as_source: bool = False
    notes: str = ''


@dataclass(frozen=True)
class Experiment:
    id: str
    name: str
    tier: int
    family: str                       # confirmatory | screen | estimation
    question: str                     # the RQ/hypothesis it serves
    arms: tuple[Arm, ...]
    seed_block: str = 'CONFIRM'
    varies: tuple[str, ...] = ()      # fields deliberately varied
    review_refs: tuple[str, ...] = () # reviewer concerns answered
    notes: str = ''

    def invariants(self) -> tuple[str, ...]:
        """Fields that must be constant across this experiment's runs."""
        return tuple(f for f in CORE_INVARIANTS if f not in self.varies)


def _cell_arms(make: Callable[[str, str], Iterable[Arm]]) -> tuple[Arm, ...]:
    out: list[Arm] = []
    for arch, rule in CELLS:
        out.extend(make(arch, rule))
    return tuple(out)


def _cell(arch: str, rule: str) -> dict:
    return dict(arch=arch, target_rule=rule)


def _tag(arch: str, rule: str) -> str:
    return f'{arch}-{rule}'


def _src_arm(arch: str, rule: str, env: str = SOURCE_ENV,
             prefix: str = 'src') -> Arm:
    """A prerequisite source run, declared inside every experiment that needs it.

    Each experiment is self-contained and runnable on its own: an arm may only
    name a source declared in the same experiment. That looks like duplication,
    and is not -- `all_jobs` de-duplicates by configuration digest, so the same
    source is trained once and shared. Making the dependency explicit is the
    point: in the published study the DDQN transfer arm's source is
    unidentifiable, and the only surviving CartPole checkpoint is from the wrong
    architecture.
    """
    return Arm(f'{prefix}-{_tag(arch, rule)}',
               {**_cell(arch, rule), 'condition': 'scratch', 'env': env},
               role='source')


def _scratch_arm(arch: str, rule: str, env: str = TARGET_ENV,
                 prefix: str = 'scratch') -> Arm:
    """The within-cell scratch baseline: the denominator for every delta."""
    return Arm(f'{prefix}-{_tag(arch, rule)}',
               {**_cell(arch, rule), 'condition': 'scratch', 'env': env})


# ---------------------------------------------------------------------------
# E1 -- the controlled 2x2
# ---------------------------------------------------------------------------
def _e1(arch: str, rule: str):
    t = _tag(arch, rule)
    yield Arm(f'src-{t}', {**_cell(arch, rule), 'condition': 'scratch',
                           'env': SOURCE_ENV}, role='source')
    yield Arm(f'scratch-{t}', {**_cell(arch, rule), 'condition': 'scratch',
                               'env': TARGET_ENV})
    yield Arm(f'transfer-{t}', {**_cell(arch, rule), 'condition': 'transfer',
                                'env': TARGET_ENV, 'source_env': SOURCE_ENV,
                                **PROTOCOL},
              source_from=f'src-{t}')
    # Pre-declared secondary: the protocol the published code implemented.
    # Transfers 94% of the mlp and 50% of the dueling network, which is why it
    # cannot carry the architecture contrast -- see DESIGN.md 3.1.
    yield Arm(f'transfer-trunk-{t}',
              {**_cell(arch, rule), 'condition': 'transfer', 'env': TARGET_ENV,
               'source_env': SOURCE_ENV, **PROTOCOL, 'transfer_set': 'trunk'},
              source_from=f'src-{t}')


# ---------------------------------------------------------------------------
# E2 -- the control set
# ---------------------------------------------------------------------------
def _e2(arch: str, rule: str):
    t = _tag(arch, rule)
    yield _src_arm(arch, rule)
    # The scratch baseline is the denominator every control contrast is measured
    # against, so it belongs to this experiment, not only to E1.
    yield _scratch_arm(arch, rule)
    base = {**_cell(arch, rule), 'env': TARGET_ENV, 'source_env': SOURCE_ENV,
            **PROTOCOL}
    yield Arm(f'untrained-{t}', {**base, 'condition': 'transfer_untrained'},
              source_from=f'src-{t}')
    # The same control with no freeze window, which is what measures whether the
    # mechanics contrast depends on freezing instead of assuming it does not.
    yield Arm(f'untrained-K0-{t}',
              {**base, 'condition': 'transfer_untrained', 'freeze_updates': 0},
              source_from=f'src-{t}')
    yield Arm(f'permuted-{t}', {**base, 'condition': 'transfer_permuted',
                                'permute_kind': 'shuffle'},
              source_from=f'src-{t}')
    # Spectrum-matched: preserves the singular values an entry-wise shuffle
    # destroys, so the two together bound the spectral caveat empirically.
    yield Arm(f'permuted-spec-{t}', {**base, 'condition': 'transfer_permuted',
                                     'permute_kind': 'spectrum'},
              source_from=f'src-{t}')


# ---------------------------------------------------------------------------
# E3 -- per-architecture hyperparameter sensitivity, on TUNE seeds only
# ---------------------------------------------------------------------------
def _e3(arch: str, rule: str):
    t = _tag(arch, rule)
    for lr in (1e-4, 3e-4, 5e-4, 1e-3):
        for upd, extra in (('hard', {'target_update_freq': 1_000}),
                           ('soft', {'tau': 0.005})):
            yield Arm(f'hp-{t}-lr{lr:g}-{upd}',
                      {**_cell(arch, rule), 'condition': 'scratch',
                       'env': TARGET_ENV, 'lr': lr, 'target_update': upd,
                       **extra})


# ---------------------------------------------------------------------------
# E4 -- freeze duration, in gradient updates
# ---------------------------------------------------------------------------
FREEZE_LEVELS = (('K0', 0), ('K5k', 5_000), ('K10k', 10_000),
                 ('K20k', 20_000), ('K50k', 50_000), ('Kinf', -1))


def _e4(arch: str, rule: str):
    t = _tag(arch, rule)
    yield _src_arm(arch, rule)
    yield _scratch_arm(arch, rule)
    for name, k in FREEZE_LEVELS:
        yield Arm(f'freeze-{name}-{t}',
                  {**_cell(arch, rule), 'condition': 'transfer',
                   'env': TARGET_ENV, 'source_env': SOURCE_ENV, **PROTOCOL,
                   'freeze_updates': k},
                  source_from=f'src-{t}')


# ---------------------------------------------------------------------------
# E5 -- which layers are transferred
# ---------------------------------------------------------------------------
def _e5(arch: str, rule: str):
    t = _tag(arch, rule)
    yield _src_arm(arch, rule)
    yield _scratch_arm(arch, rule)
    for ts in ('fc1', 'fc2', 'trunk', 'matched', 'described'):
        yield Arm(f'set-{ts}-{t}',
                  {**_cell(arch, rule), 'condition': 'transfer',
                   'env': TARGET_ENV, 'source_env': SOURCE_ENV, **PROTOCOL,
                   'transfer_set': ts},
                  source_from=f'src-{t}')


# ---------------------------------------------------------------------------
# E6 -- stream-wise freezing (dueling only)
# ---------------------------------------------------------------------------
def _e6():
    out = []
    for rule in ('vanilla', 'double'):
        t = _tag('dueling', rule)
        out.append(_src_arm('dueling', rule))
        out.append(_scratch_arm('dueling', rule))
        for grp in ('none', 'trunk', 'value', 'adv', 'heads'):
            out.append(Arm(
                f'sfreeze-{grp}-{t}',
                {'arch': 'dueling', 'target_rule': rule,
                 'condition': 'transfer', 'env': TARGET_ENV,
                 'source_env': SOURCE_ENV, **PROTOCOL, 'freeze_group': grp},
                source_from=f'src-{t}'))
    return tuple(out)


# ---------------------------------------------------------------------------
# E7 -- dueling aggregation variant
# ---------------------------------------------------------------------------
def _e7():
    out = []
    for rule in ('vanilla', 'double'):
        t = _tag('dueling', rule)
        for agg in ('mean', 'max', 'naive'):
            out.append(Arm(f'agg-{agg}-scratch-{t}',
                           {'arch': 'dueling', 'target_rule': rule,
                            'condition': 'scratch', 'env': TARGET_ENV,
                            'aggregation': agg}))
            out.append(Arm(f'agg-{agg}-src-{t}',
                           {'arch': 'dueling', 'target_rule': rule,
                            'condition': 'scratch', 'env': SOURCE_ENV,
                            'aggregation': agg}, role='source'))
            out.append(Arm(f'agg-{agg}-transfer-{t}',
                           {'arch': 'dueling', 'target_rule': rule,
                            'condition': 'transfer', 'env': TARGET_ENV,
                            'source_env': SOURCE_ENV, 'aggregation': agg,
                            **PROTOCOL},
                           source_from=f'agg-{agg}-src-{t}'))
    return tuple(out)


# ---------------------------------------------------------------------------
# E8 -- same-interface dynamics shift. Wind is primary; gravity is secondary
# because its no-op score moves from 0.18 to 0.55 across levels, so it changes
# task difficulty as well as dynamics (DESIGN.md 5.1).
# ---------------------------------------------------------------------------
WIND_LEVELS = (
    ('w075', 'LunarLander-v3:enable_wind=1,wind_power=7.5,turbulence_power=1.5'),
    ('w15', 'LunarLander-v3:enable_wind=1,wind_power=15,turbulence_power=1.5'),
)
GRAVITY_LEVELS = (('g08', 'LunarLander-v3:gravity=-8'),
                  ('g06', 'LunarLander-v3:gravity=-6'),
                  ('g04', 'LunarLander-v3:gravity=-4'))


def _e8(arch: str, rule: str):
    t = _tag(arch, rule)
    # The source for a same-interface shift arm is the *unshifted* environment's
    # scratch run: the shift is the only thing that differs between source and
    # target, which is what makes the shift level the manipulated factor.
    yield _scratch_arm(arch, rule)
    for name, env in WIND_LEVELS + GRAVITY_LEVELS:
        # Each shift level needs its own scratch denominator: the estimand is
        # the within-variant delta, and a variant's scratch performance is not
        # the base environment's.
        yield Arm(f'shift-{name}-scratch-{t}',
                  {**_cell(arch, rule), 'condition': 'scratch', 'env': env})
        # Protocol-matched to E1: trunk-only with a reinitialised head and a
        # size-matched re-draw of the input layer, so the same-interface and
        # cross-interface arms differ in shift and interface -- not in protocol
        # as well.
        yield Arm(f'shift-{name}-transfer-{t}',
                  {**_cell(arch, rule), 'condition': 'transfer', 'env': env,
                   'source_env': TARGET_ENV, **PROTOCOL,
                   'transfer_set': 'trunk', 'input_policy': 'redraw_matched'},
                  source_from=f'scratch-{t}')


# ---------------------------------------------------------------------------
# E8i -- interface change at zero dynamics shift. The missing corner, and C4.
# ---------------------------------------------------------------------------
def _e8i(arch: str, rule: str):
    t = _tag(arch, rule)
    yield Arm(f'iface-scratch-{t}',
              {**_cell(arch, rule), 'condition': 'scratch',
               'env': INTERFACE_ENV})
    yield Arm(f'iface-transfer-{t}',
              {**_cell(arch, rule), 'condition': 'transfer',
               'env': INTERFACE_ENV, 'source_env': TARGET_ENV, **PROTOCOL},
              source_from=f'c4src-{t}', source_seed_block='C4SRC',
              notes='doubles as control C4; sources drawn from a disjoint seed '
                    'block so the C4 deltas are independent of E1 denominators')
    yield Arm(f'c4src-{t}', {**_cell(arch, rule), 'condition': 'scratch',
                             'env': TARGET_ENV}, role='source',
              only_as_source=True,
              notes='dedicated donor for control C4, drawn from the C4SRC seed '
                    'block so the C4 deltas are independent of the scratch runs '
                    'used as denominators elsewhere')


# ---------------------------------------------------------------------------
# E11 -- value-head recalibration probe (dueling only)
# ---------------------------------------------------------------------------
def _e11():
    out = []
    for rule in ('vanilla', 'double'):
        t = _tag('dueling', rule)
        out.append(_src_arm('dueling', rule))
        out.append(_scratch_arm('dueling', rule))
        for mode in ('center', 'center_scale'):
            out.append(Arm(f'recal-{mode}-{t}',
                           {'arch': 'dueling', 'target_rule': rule,
                            'condition': 'transfer', 'env': TARGET_ENV,
                            'source_env': SOURCE_ENV, **PROTOCOL,
                            'value_recal': mode},
                           source_from=f'src-{t}'))
    return tuple(out)


# ---------------------------------------------------------------------------
# E12 -- capacity
# ---------------------------------------------------------------------------
def _e12(arch: str, rule: str):
    t = _tag(arch, rule)
    for w in (64, 256):
        yield Arm(f'cap{w}-src-{t}',
                  {**_cell(arch, rule), 'condition': 'scratch',
                   'env': SOURCE_ENV, 'hidden': (w, w),
                   'head_units': max(32, w // 2)}, role='source')
        yield Arm(f'cap{w}-scratch-{t}',
                  {**_cell(arch, rule), 'condition': 'scratch',
                   'env': TARGET_ENV, 'hidden': (w, w),
                   'head_units': max(32, w // 2)})
        yield Arm(f'cap{w}-transfer-{t}',
                  {**_cell(arch, rule), 'condition': 'transfer',
                   'env': TARGET_ENV, 'source_env': SOURCE_ENV, **PROTOCOL,
                   'hidden': (w, w), 'head_units': max(32, w // 2)},
                  source_from=f'cap{w}-src-{t}')


# ---------------------------------------------------------------------------
# E0 -- smoke. Tiny, and exercises every code path the catalogue uses.
# ---------------------------------------------------------------------------
# Deliberately sized so the freeze boundary is *crossed* inside the budget:
# a smoke run that never unfreezes never exercises the freeze verification, the
# tf.function retrace, or the optimiser's survival across the transition -- which
# are three of the paths most likely to break.
SMOKE_OVERRIDES = dict(num_episodes=12, max_steps=200, eval_every=4,
                       eval_episodes=2, final_eval_episodes=5,
                       final_eval_checkpoints=2, prefix_checkpoints=(6,),
                       learning_starts=100, diag_states=64, probe_steps=20,
                       probe_transitions=200, freeze_updates=200,
                       checkpoint_seconds=5)


def _e0():
    out = [Arm('smoke-src', {'arch': 'dueling', 'target_rule': 'double',
                             'condition': 'scratch', 'env': SOURCE_ENV},
               role='source')]
    for cond in ('transfer', 'transfer_untrained', 'transfer_permuted'):
        out.append(Arm(f'smoke-{cond}',
                       {'arch': 'dueling', 'target_rule': 'double',
                        'condition': cond, 'env': TARGET_ENV,
                        'source_env': SOURCE_ENV, **PROTOCOL},
                       source_from='smoke-src'))
    out.append(Arm('smoke-scratch', {'arch': 'mlp', 'target_rule': 'vanilla',
                                     'condition': 'scratch',
                                     'env': TARGET_ENV}))
    out.append(Arm('smoke-iface-src', {'arch': 'mlp', 'target_rule': 'double',
                                       'condition': 'scratch',
                                       'env': TARGET_ENV}, role='source'))
    out.append(Arm('smoke-iface', {'arch': 'mlp', 'target_rule': 'double',
                                   'condition': 'transfer',
                                   'env': INTERFACE_ENV,
                                   'source_env': TARGET_ENV, **PROTOCOL},
                   source_from='smoke-iface-src'))
    return tuple(out)



# ---------------------------------------------------------------------------
# E9 -- additional cross-interface pairs, including the reverse direction.
# C1 (a single source->target pair) was raised by six of the eight reviewers,
# and it is the one concern a single pair cannot answer by any amount of
# analysis. The reverse direction is the informative addition: if the effect is
# a property of the protocol rather than of the pair, it should not vanish when
# the pair is inverted.
# ---------------------------------------------------------------------------
ENV_PAIRS = (
    ('acro2ll', 'Acrobot-v1', TARGET_ENV),
    ('cp2acro', SOURCE_ENV, 'Acrobot-v1'),
    ('ll2cp', TARGET_ENV, SOURCE_ENV),
)


def _e9(arch: str, rule: str):
    t = _tag(arch, rule)
    seen_sources: set[str] = set()
    seen_scratch: set[str] = set()
    for name, src_env, tgt_env in ENV_PAIRS:
        src_label = f'p-src-{name}-{t}'
        if src_env not in seen_sources:
            yield Arm(src_label, {**_cell(arch, rule), 'condition': 'scratch',
                                  'env': src_env}, role='source')
            seen_sources.add(src_env)
        # The target task's own scratch baseline: the denominator for this
        # pair's delta. Without it the pair yields no within-cell effect at all.
        if tgt_env not in seen_scratch:
            yield Arm(f'p-scratch-{name}-{t}',
                      {**_cell(arch, rule), 'condition': 'scratch',
                       'env': tgt_env})
            seen_scratch.add(tgt_env)
        yield Arm(f'p-transfer-{name}-{t}',
                  {**_cell(arch, rule), 'condition': 'transfer', 'env': tgt_env,
                   'source_env': src_env, **PROTOCOL},
                  source_from=src_label)


# ---------------------------------------------------------------------------
# E13 -- the plasticity-loss rival explanation, tested rather than only measured.
# ---------------------------------------------------------------------------
def _e13(arch: str, rule: str):
    t = _tag(arch, rule)
    yield _src_arm(arch, rule)
    yield _scratch_arm(arch, rule)
    base = {**_cell(arch, rule), 'condition': 'transfer', 'env': TARGET_ENV,
            'source_env': SOURCE_ENV, **PROTOCOL}
    yield Arm(f'plast-reset-{t}',
              {**base, 'reset_head_at_unfreeze': True},
              source_from=f'src-{t}',
              notes='reinitialise the output head at the unfreeze boundary')
    yield Arm(f'plast-sp-{t}', {**base, 'shrink_perturb': 0.2},
              source_from=f'src-{t}',
              notes='Ash & Adams shrink-and-perturb at the unfreeze boundary')


EXPERIMENTS: dict[str, Experiment] = {
    'E0': Experiment(
        'E0', 'smoke', tier=0, family='estimation',
        question='none -- validates that every code path executes',
        arms=_e0(), seed_block='SMOKE',
        notes='not a result under any circumstances'),
    'E1': Experiment(
        'E1', 'core2x2', tier=1, family='confirmatory',
        question='RQ1, RQ2, RQ3 -- the within-cell transfer effect and its '
                 'variation across the 2x2',
        arms=_cell_arms(_e1),
        review_refs=('C11', 'C20', 'C3', 'ICANN#5 Q1')),
    'E2': Experiment(
        'E2', 'controls', tier=1, family='estimation',
        question='RQ4, H1, H2 -- how much of the effect is protocol mechanics '
                 'and weight statistics rather than learned structure',
        arms=_cell_arms(_e2), varies=(),
        review_refs=('C6',)),
    'E3': Experiment(
        'E3', 'hpsens', tier=1, family='screen',
        question='per-architecture fair baseline; the secondary tuning policy '
                 'of DESIGN.md 3.3',
        arms=_cell_arms(_e3), seed_block='TUNE',
        varies=('lr', 'target_update', 'target_update_freq'),
        review_refs=('C11', 'C5', 'ICANN#5 Q1', 'ICANN#5 Q5')),
    'E4': Experiment(
        'E4', 'freezedur', tier=2, family='screen',
        question='RQ4 -- freeze duration, in gradient updates',
        arms=_cell_arms(_e4), varies=(),
        review_refs=('C5', 'C19', 'ICANN#5 Q2', 'ICANN#3')),
    'E5': Experiment(
        'E5', 'layerset', tier=2, family='screen',
        question='RQ4 -- which transferred layers matter, and the '
                 'described-versus-implemented protocol contrast',
        arms=_cell_arms(_e5),
        review_refs=('C5', 'ICANN#2', 'ICANN#5 Q2')),
    'E6': Experiment(
        'E6', 'streamfreeze', tier=2, family='screen',
        question='RQ4, H6 -- value stream versus advantage stream',
        arms=_e6(), review_refs=('C6',)),
    'E7': Experiment(
        'E7', 'aggregation', tier=2, family='screen',
        question='RQ4 -- whether the dueling baseline subtraction does the work '
                 'attributed to it',
        arms=_e7(), review_refs=('C6', 'C15')),
    'E8': Experiment(
        'E8', 'shiftaxis', tier=2, family='estimation',
        question='RQ5, H4 -- transfer effect against measured dynamics shift at '
                 'a fixed interface and a protocol matched to E1',
        arms=_cell_arms(_e8), review_refs=('C1', 'C14', 'ICANN#5 Q8')),
    'E8i': Experiment(
        'E8i', 'interfaceonly', tier=1, family='estimation',
        question='RQ5, H5 -- interface change at zero dynamics shift; also '
                 'control C4',
        arms=_cell_arms(_e8i), review_refs=('C1', 'C14')),
    'E9': Experiment(
        'E9', 'envpairs', tier=1, family='estimation',
        question='RQ5 -- additional source->target pairs and the reverse '
                 'direction; the only answer to C1, which six of eight '
                 'reviewers raised',
        arms=_cell_arms(_e9), review_refs=('C1', 'ICANN#5 Q8'),
        notes='promoted from tier 3 to tier 1 on 2026-08-26. Leaving the '
              'only answer to the most-raised reviewer objection in the '
              'tier most likely to be cut was an internal contradiction, '
              'and the measured cost is 20.2 h at --jobs 6 for 240 runs at '
              'ten seeds, which is small beside the 82.8 h of tier 1'),
    'E13': Experiment(
        'E13', 'plasticity', tier=3, family='estimation',
        question='RQ4 -- whether the plasticity-loss account explains the '
                 'effect: head reset and shrink-and-perturb at the unfreeze '
                 'boundary',
        arms=_cell_arms(_e13), review_refs=('C6',),
        notes='the rival explanation from paper/LITERATURE.md 3.4, which the '
              'control set measures but does not otherwise exclude'),
    'E11': Experiment(
        'E11', 'valuerecal', tier=2, family='estimation',
        question='RQ4 -- value-scale recalibration. Policy-invariant by '
                 'construction, so it can only speak to optimisation dynamics',
        arms=_e11(), review_refs=('C6',)),
    'E12': Experiment(
        'E12', 'capacity', tier=3, family='screen',
        question='RQ4 -- capacity sensitivity',
        arms=_cell_arms(_e12), varies=('hidden', 'head_units'),
        review_refs=('C5',)),
}

TIERS = {1: [e for e in EXPERIMENTS.values() if e.tier == 1],
         2: [e for e in EXPERIMENTS.values() if e.tier == 2],
         3: [e for e in EXPERIMENTS.values() if e.tier == 3]}


# ---------------------------------------------------------------------------
# Job construction
# ---------------------------------------------------------------------------
# Fields a caller may scale without changing what the experiment *is*: budget
# and measurement settings. Everything else is a factor, and overriding a factor
# silently would mean a run labelled as a catalogue arm was not one.
SCALING_FIELDS = frozenset({
    'num_episodes', 'max_steps', 'eval_every', 'eval_episodes',
    'final_eval_episodes', 'final_eval_checkpoints', 'prefix_checkpoints',
    'diag_states', 'probe_steps', 'probe_transitions', 'learning_starts',
    'checkpoint_seconds', 'out_root', 'keep_buffer', 'log_diagnostics', 'notes',
})


@dataclass
class Job:
    """One resolved run, plus the identity of the source it draws on.

    The source fields exist because `DESIGN.md` §4.3 lets the source seed stop
    being a pure function of the target seed. Until the reserve rule, a
    consumer's source was always at the consumer's own seed (or the fixed
    `C4SRC` mapping), so "which source" needed no recording. A rejected source
    is replaced from `RESERVE`, and the pair (`source_default_seed`,
    `source_seed`) is what says whether a replacement is in force.
    `source_checkpoint` cannot say it: it is a path, and paths are deliberately
    excluded from the run digest (`src/dqn/config.py`).

    The assigned seed also reaches the `Config`, so a replacement run does get a
    directory of its own (`source_seed_is_recorded`). The *default* it stands in
    for does not, and exists only here and in the job record `sweep.py` writes
    from it, which is why both fields are carried rather than one.
    """

    experiment: str
    arm: str
    role: str
    cfg: Config
    depends_on: Optional[str] = None      # run_dir of the source job
    # Label of the arm that supplied the source, the seed it would have used
    # with no replacement, and the seed it actually uses.
    source_arm: Optional[str] = None
    source_default_seed: Optional[int] = None
    source_seed: Optional[int] = None
    # Seed-independent identity of the source *configuration*: the pool a
    # replacement is drawn within. Two consumers sharing a lineage must never
    # share a replacement seed, or one source run would stand in for two arms.
    source_lineage: Optional[str] = None

    def key(self) -> str:
        return self.cfg.run_dir()

    @property
    def source_replaced(self) -> bool:
        """True when the validity gate moved this run off its default source."""
        return (self.source_seed is not None
                and self.source_seed != self.source_default_seed)


#: Where `sweep.py` records every source the validity gate rejected. Read back
#: here so that a caller which only knows the run tree -- `aggregate.py`,
#: `audit.py` -- resolves the job graph the runner actually ran, rather than the
#: one the catalogue would have produced with no rejections in it.
REPLACEMENTS_RELPATH = os.path.join('_jobs', 'source_replacements.jsonl')


def load_source_replacements(out_root: str
                             ) -> tuple[dict[tuple[str, int], int],
                                        dict[str, set[int]]]:
    """The runner's rejection ledger, in the two shapes `jobs` needs.

    Returns

    * the **assignment**: (source arm label, the seed the source would default
      to) -> the seed finally in force. Chains are followed, so a lineage
      rejected twice yields one entry pointing at its second replacement.
    * the **draws**: source arm label -> every reserve seed ever handed to it,
      intermediate links included. A rejected replacement is still a run that
      exists on disk and still has to be declarable, which the collapsed
      assignment on its own cannot say.

    Rows with no replacement seed are rejections the reserve could not cover.
    They belong in the results table but not in either map, so they are skipped.

    Deliberately silent about a missing or damaged file: the ledger records
    something that may simply not have happened yet, and refusing to enumerate
    the catalogue because a run tree contains no rejections would be absurd.
    """
    path = os.path.join(out_root, REPLACEMENTS_RELPATH)
    chain: dict[tuple[str, int], int] = {}
    draws: dict[str, set[int]] = {}
    try:
        with open(path, encoding='utf-8', errors='replace') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rep = row.get('replacement_seed')
                if rep is None:
                    continue
                try:
                    arm, rejected, rep = (str(row['source_arm']),
                                          int(row['rejected_seed']), int(rep))
                except (KeyError, TypeError, ValueError):
                    continue
                chain[(arm, rejected)] = rep
                draws.setdefault(arm, set()).add(rep)
    except OSError:
        return {}, {}

    assignment: dict[tuple[str, int], int] = {}
    for arm, start in chain:
        seed, seen = start, {start}
        while (arm, seed) in chain:
            seed = chain[(arm, seed)]
            if seed in seen:               # a hand-edited ledger; leave it be
                break
            seen.add(seed)
        if seed != start:
            assignment[(arm, start)] = seed
    return assignment, draws


#: The seed a lineage identity is computed at. Any fixed value works; the point
#: is only that it is the *same* one for every member of a lineage.
_LINEAGE_SEED = 0


def source_lineage(cfg: Config) -> str:
    """Seed-independent identity of a source configuration.

    A replacement source differs from the one it replaces in exactly one field,
    the seed, so the seed is the one field the pool must not be keyed by.
    Everything else -- cell, environment, budget, optimiser -- has to match, or
    the "replacement" would be a different experiment. Computing the run digest
    at a fixed seed gives that equivalence class for free and inherits the
    digest's guarantee that no identity-bearing field is missed.
    """
    return dataclasses.replace(cfg, seed=_LINEAGE_SEED).run_digest()


def source_seed_is_recorded(seed: int, src_seed: int,
                            default_src_seed: int) -> bool:
    """Whether a job's source seed belongs in its `Config`, hence in its digest.

    `src/dqn/config.py` defaults `source_seed` to `seed` when it is not given,
    and keys the run digest on it only when it *differs* from `seed`. Two
    consequences follow, and they pull in opposite directions.

    * A `RESERVE` replacement (``src_seed != default_src_seed``) **must** be
      recorded. Left out, the replacement trains from a different source and
      still hashes to the rejected run's directory, and `aggregate.py` copies
      `source_seed = seed` into `per_seed.csv` for a run whose source is a
      reserve seed. `DESIGN.md` 4.3 requires the opposite: a rejected source and
      the seed that replaced it have to be legible in the results table. This is
      the case fix B4 put `source_seed` in the digest for.
    * An arm whose *canonical* source already sits at another seed, which today
      is E8i's `iface-transfer-*` drawing on the `C4SRC` donors, must **not** be
      recorded, because the digest cannot tell "canonical for this arm" from "a
      deviation from it" and would re-identify every such run already on disk.
      Their true source seed is still carried on the `Job`, written by
      `sweep.py` into the job record, and recoverable from the checkpoint path,
      which is the fallback `aggregate.py` applies when the field is absent.

    So: record unless the draw is canonical *and* lands outside the run's own
    seed. Closing the second case properly needs `Config` to carry the arm's
    default source seed as a bookkeeping field, so that the digest can key on a
    deviation from *that* rather than from `seed`; until it does, recording it
    here would move existing run directories, which is a worse defect than the
    one it fixes.
    """
    return src_seed != default_src_seed or src_seed == seed


def source_arm_labels(exp: Experiment) -> frozenset[str]:
    """Labels of the arms some other arm in `exp` draws its source from."""
    return frozenset(a.source_from for a in exp.arms if a.source_from)


def resolve_seeds(spec: str | Iterable[int] | None,
                  block: str) -> tuple[int, ...]:
    """Seeds for an experiment: a named block, or an explicit override."""
    if spec is None:
        return SEED_BLOCKS[block]
    if isinstance(spec, str):
        if spec in SEED_BLOCKS:
            return SEED_BLOCKS[spec]
        out: list[int] = []
        for tok in spec.replace(',', ' ').split():
            if '-' in tok.lstrip('-') and not tok.startswith('-'):
                lo, hi = tok.split('-')
                out.extend(range(int(lo), int(hi) + 1))
            else:
                out.append(int(tok))
        return tuple(sorted(set(out)))
    return tuple(sorted(set(int(s) for s in spec)))


def jobs(experiment: str, seeds: str | Iterable[int] | None = None,
         out_root: str = 'runs', overrides: dict | None = None,
         allow_factor_overrides: bool = False,
         source_seeds: Mapping[tuple[str, int], int] | None = None) -> list[Job]:
    """Resolve one experiment into concrete jobs, sources before their consumers.

    Source checkpoints are resolved from the *same seed* and the *same arm
    lineage*, so no cell, shift level or capacity setting can draw on another's
    source. That is enforced here rather than trusted, because in the published
    study the DDQN transfer arm's source is unidentifiable and the only
    surviving CartPole checkpoint is from the wrong architecture.

    ``source_seeds`` re-points that resolution. It maps
    ``(source arm label, the seed the source would default to)`` to the seed to
    use instead, which is how `DESIGN.md` 4.3's reserve rule reaches the job
    graph: `sweep.py` gates each source on its normalised final score and hands
    back a map for the ones the gate rejected. The map is keyed on the
    *default* seed, not the current one, so a lineage replaced twice still
    resolves from one entry and re-running is a lookup rather than a fresh
    draw. Passing nothing is the pre-reserve behaviour exactly.
    """
    exp = EXPERIMENTS[experiment]
    seed_list = resolve_seeds(seeds, exp.seed_block)
    extra = dict(overrides or {})
    if experiment == 'E0':
        extra = {**SMOKE_OVERRIDES, **extra}

    factor_overrides = sorted(set(extra) - SCALING_FIELDS)
    if factor_overrides and not (allow_factor_overrides or experiment == 'E0'):
        raise ValueError(
            f'{experiment}: overrides {factor_overrides} change experimental '
            f'factors, not just the budget. A run whose factors were changed is '
            f'not the arm it is labelled as. Pass '
            f'allow_factor_overrides=True to proceed deliberately -- the runs '
            f'will carry a note recording it, and their configuration digests '
            f'differ so they cannot be mistaken for catalogue runs. Scalable '
            f'fields: {sorted(SCALING_FIELDS)}')

    by_label = {a.label: a for a in exp.arms}
    # ``None`` means "read the tree"; an explicit map -- including an empty one
    # -- means the caller has already resolved the assignment and owns it. The
    # runner always passes its own map, so it never races its own ledger.
    if source_seeds is None:
        replacements, draws = load_source_replacements(out_root)
    else:
        replacements = dict(source_seeds)
        draws = {}
        for (arm_label, _default), rep in replacements.items():
            draws.setdefault(arm_label, set()).add(int(rep))
    built: dict[tuple[str, int], Job] = {}
    out: list[Job] = []

    def build(label: str, seed: int) -> Job:
        cached = built.get((label, seed))
        if cached is not None:
            return cached
        arm = by_label.get(label)
        if arm is None:
            raise KeyError(
                f'{experiment}: arm {label!r} names a source that is not '
                f'declared in this experiment. Sources must be explicit -- an '
                f'unresolvable source is how the published study lost track of '
                f'what its transfer arm loaded.')
        dep = None
        # Precedence: shared defaults, then the arm's own factors, then the
        # caller's scaling overrides LAST. The arm must not be able to silently
        # ignore a caller's budget reduction -- with the previous ordering, an
        # arm carrying PROTOCOL's freeze_updates=10000 overrode a smoke run's
        # request for 200, so a 14-episode validation run never reached the
        # unfreeze boundary and never exercised freeze verification at all.
        kwargs = dict(COMMON)
        kwargs.update(arm.overrides)
        kwargs.update(extra)
        kwargs.update(experiment=experiment, label=arm.label,
                      out_root=out_root, seed=seed)
        if factor_overrides:
            # Recorded in the manifest, because a run whose factors were scaled
            # is not the catalogue arm it is named after. Its configuration
            # digest differs, so it lives in its own directory and cannot be
            # confused with a real run -- but the note makes that legible
            # without recomputing a hash.
            kwargs['notes'] = ((kwargs.get('notes') or '') +
                               f' [factor overrides: {sorted(factor_overrides)}]'
                               ).strip()

        src_arm = src_lineage = None
        src_seed = default_src_seed = None
        if arm.source_from:
            default_src_seed = seed
            if arm.source_seed_block:
                block = SEED_BLOCKS[arm.source_seed_block]
                default_src_seed = block[seed % len(block)]
            src_seed = int(replacements.get((arm.source_from, default_src_seed),
                                            default_src_seed))
            if src_seed != default_src_seed and src_seed == seed:
                # A replacement whose seed is the consumer's own seed is
                # not representable: the digest omits `source_seed` exactly
                # there, so the replacement would wear the canonical
                # lineage's identity and write into its directory. Refuse
                # the assignment rather than produce an ambiguous run.
                # `DESIGN.md` 4.3 draws replacements from RESERVE, which
                # never collides with a consumer seed, so this can only
                # arise from a hand-built `source_seeds` map or a ledger
                # written by something other than the validity gate.
                raise ValueError(
                    f'{experiment}:{arm.label} at seed {seed} is assigned a '
                    f'replacement source at seed {src_seed}, which is its '
                    f'own seed. src/dqn/config.py keys the run digest on '
                    f'source_seed only where it differs from seed, so this '
                    f'run would be indistinguishable from the canonical '
                    f'lineage it replaces and would write into that run '
                    f'directory. Replacements come from RESERVE '
                    f'({RESERVE_ORDER[0]}+), per DESIGN.md 4.3.')
            src_job = build(arm.source_from, src_seed)
            kwargs['source_checkpoint'] = os.path.join(src_job.cfg.run_dir(),
                                                       'model.keras')
            # Which source this run actually loaded, written into the
            # configuration and so into the manifest and the run digest.
            # `Config.__post_init__` forbids the field on a scratch run, so
            # it is offered only to a condition that draws on a source.
            if (kwargs.get('condition', 'scratch') != 'scratch'
                    and source_seed_is_recorded(seed, src_seed,
                                                default_src_seed)):
                kwargs['source_seed'] = src_seed
            dep = src_job.key()
            src_arm = arm.source_from
            src_lineage = source_lineage(src_job.cfg)

        cfg = Config(**kwargs)
        job = Job(experiment, arm.label, arm.role, cfg, dep,
                  source_arm=src_arm, source_default_seed=default_src_seed,
                  source_seed=src_seed, source_lineage=src_lineage)
        built[(label, seed)] = job
        out.append(job)
        return job

    donors = source_arm_labels(exp)
    reserve = set(SEED_BLOCKS['RESERVE'])
    for seed in seed_list:
        # `DESIGN.md` 3.4 gives RESERVE exactly one use -- "replacement sources
        # when the validity gate rejects one" -- and "never used for anything
        # else". Enumerating a whole experiment at a RESERVE seed would declare
        # arms the design forbids to exist, and the completeness checks
        # downstream infer an experiment's seed axis from the runs on disk: a
        # replacement source at seed 400 would put 400 on that axis and every
        # target-side arm would then be reported missing at a seed that must
        # never be run. Only the donor arms are built there.
        donors_only = seed in reserve
        # Narrowed once more when the ledger says which lineage drew this seed.
        # One rejected source does not mean every cell needs a run at 400, and
        # declaring the others would report them missing for ever. With nothing
        # to read, every donor arm is declared: a superset, which leaves a real
        # replacement run attributable rather than orphaned.
        drawn_here = {label for label, seeds in draws.items() if seed in seeds}
        allowed = drawn_here or donors
        for arm in exp.arms:
            if arm.only_as_source:
                continue          # built on demand, at its own block's seed
            if donors_only and arm.label not in allowed:
                continue
            build(arm.label, seed)
    return out


def all_jobs(experiments: Iterable[str], seeds=None, out_root: str = 'runs',
             overrides: dict | None = None,
             allow_factor_overrides: bool = False,
             source_seeds: Mapping[tuple[str, int], int] | None = None
             ) -> list[Job]:
    """Jobs for several experiments, de-duplicated by run directory.

    De-duplication is the payoff of keying a run by its configuration digest
    rather than by experiment: E4's freeze level that equals E1's protocol
    value, E8's level-0 scratch arms and E7's scratch arms are the *same runs*,
    and training them twice would both waste compute and produce two
    independent estimates of one quantity.
    """
    seen: dict[str, Job] = {}
    ordered: list[Job] = []
    for name in experiments:
        for job in jobs(name, seeds, out_root, overrides,
                        allow_factor_overrides, source_seeds):
            prior = seen.get(job.key())
            if prior is not None:
                # De-duplication is only sound while the two jobs really
                # are the same run. A shared directory with a different
                # source is the failure this catalogue exists to prevent:
                # one of the two would be dropped here and the surviving
                # run would be reported under both arms' lineages. Refuse
                # rather than pick, because there is no right pick.
                if (prior.cfg.source_checkpoint
                        != job.cfg.source_checkpoint
                        or prior.source_seed != job.source_seed):
                    raise ValueError(
                        f'{prior.experiment}:{prior.arm} and '
                        f'{job.experiment}:{job.arm} resolve to the same '
                        f'run directory {job.key()!r} but draw on '
                        f'different sources (seeds {prior.source_seed} and '
                        f'{job.source_seed}; checkpoints '
                        f'{prior.cfg.source_checkpoint!r} and '
                        f'{job.cfg.source_checkpoint!r}). One run cannot be '
                        f'both, and the run digest does not separate them, '
                        f'so de-duplicating would silently drop one arm.')
                continue
            seen[job.key()] = job
            ordered.append(job)
    return ordered


def summary() -> list[dict]:
    """One row per experiment, for `plan.py` and the documentation."""
    rows = []
    for exp in EXPERIMENTS.values():
        n_seeds = len(SEED_BLOCKS[exp.seed_block])
        rows.append({
            'id': exp.id, 'name': exp.name, 'tier': exp.tier,
            'family': exp.family, 'arms': len(exp.arms),
            'seed_block': exp.seed_block, 'seeds': n_seeds,
            'runs': len(exp.arms) * n_seeds,
            'varies': ', '.join(exp.varies) or '-',
            'question': exp.question,
            'review_refs': ', '.join(exp.review_refs) or '-',
        })
    return rows


__all__ = ['Arm', 'Experiment', 'Job', 'EXPERIMENTS', 'SEED_BLOCKS', 'CELLS',
           'SCALING_FIELDS', 'ENV_PAIRS', 'SOURCE_VALIDITY_GATE',
           'RESERVE_ORDER', 'source_lineage', 'source_arm_labels',
           'source_seed_is_recorded',
           'load_source_replacements', 'REPLACEMENTS_RELPATH',
           'TIERS', 'PROTOCOL', 'COMMON', 'CORE_INVARIANTS', 'jobs',
           'all_jobs', 'resolve_seeds', 'summary', 'SMOKE_OVERRIDES',
           'SOURCE_ENV', 'TARGET_ENV', 'INTERFACE_ENV', 'FREEZE_LEVELS',
           'WIND_LEVELS', 'GRAVITY_LEVELS']
