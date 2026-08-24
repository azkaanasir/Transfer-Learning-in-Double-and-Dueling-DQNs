"""Self-tests for the guardrails `DESIGN.md` §9 claims are "enforced in code".

Why this file exists
--------------------
Revision 1 of the design carried a table captioned "Anti-fallacy guardrails,
**enforced in code**" and there was no such code (`DESIGN.md` §11, defect 12).
An adversarial review found the table entirely aspirational. This module is what
makes the caption true: every row of §9, and every mechanical claim §3.2, §5.1,
§8.1 and §8.2 make about the infrastructure, has a test here that **fails when
the guard is removed**. That is the acceptance criterion for a test in this
file, and it is stricter than "the test passes": a test that would still pass
with its guard deleted is decoration, and the docstring of each case below names
the specific defect it would catch.

The guards, and the defect each one prevents
--------------------------------------------
Every case is tied to a documented failure -- either one the Phase 0 audit found
in the published study (`paper/METHODS_ACTUAL.md`) or one the adversarial review
of design revision 1 found in the corrected design.

``test_env_variants_apply``
    CartPole keeps `total_mass` and `polemass_length` as *derived* attributes.
    Setting `masspole` or `length` without recomputing them leaves the simulator
    on default physics, so a run labelled as a variant is not one. The test also
    drives every dynamics variant against its base environment and requires
    non-zero paired trajectory divergence, because "the parameter was accepted"
    and "the simulation changed" are different claims and only the second one
    matters (`DESIGN.md` §6.2).

``test_interface_wrapper_preserves_dynamics``
    The same-dynamics/changed-interface corner of `DESIGN.md` §6.4 is only that
    corner if the padded/extended environment reproduces the base environment's
    trajectory exactly. If the wrappers perturbed the dynamics, H5 would be
    measuring shift and interface together -- the very confound the corner
    exists to remove.

``test_shift_selfcheck_zero``
    `DESIGN.md` §6.3 validates the divergence measure "by a self-check against
    an identical-environment control that must return exactly zero". Anything
    else means the environment carries hidden state and no divergence number
    from that function means anything.

``test_run_identity_no_collisions``
    The defect that forced revision 2: a directory named
    `<env>/<arch>-<rule>-<mode>-s<NN>` omitted `freeze_*`, `transfer_*`, `lr`,
    `target_update`, `hidden`, `aggregation`, the environment variant and the
    control condition, so nine conditions from six catalogue experiments
    collapsed onto one run directory and five experiments would have been
    fabricated from one experiment's data with every invariant check passing.

``test_field_classification_complete``
    The mechanism that stops the previous defect recurring: a `Config` field
    added without being classified must raise at import, or the digest silently
    stops covering it. The test removes a field from the classification and
    requires the guard to fire.

``test_epsilon_closed_form``
    In the published code the epsilon decay lived inside the evaluation branch,
    which made the exploration schedule a function of `eval_every`. Two
    consequences are asserted here: the schedule is identical when the
    measurement cadence changes (`DESIGN.md` §3.2, §8.1), and it is identical
    when the *budget* changes -- which is the identifying condition for RQ6,
    because it is what makes a 500-episode prefix of a 1000-episode run equal to
    a 500-episode run (`DESIGN.md` §2.4, `ANALYSIS_PLAN.md` §3).

``test_diagnostics_are_inert``
    The test the published code would have failed. Its diagnostic drew states
    from the training replay buffer, which advanced the buffer's generator, so
    turning diagnostics on changed every subsequent minibatch and therefore the
    whole trajectory. An ablation in that regime measures its own
    instrumentation (`DESIGN.md` §8.1).

``test_per_layer_seeding``
    `DESIGN.md` §8.1 states that at a given seed an `mlp` and a `dueling`
    network share their trunk initialisation exactly, and that this is what
    matches the 2x2 by seed and licenses the paired analysis of
    `ANALYSIS_PLAN.md` §2. Under global seeding it would be false, and the
    pairing justification would be false with it.

``test_rng_stream_independence``
    Same section: adding a diagnostic must not perturb the training trajectory
    of the parts it does not touch. Tested rather than asserted.

``test_resume_equivalence`` / ``test_metrics_no_duplication_on_resume``
    The published loop appended to `metrics.csv` on resume without truncating; a
    45-episode run interrupted at 39 whose last checkpoint was 30 produced 54
    rows with episodes 31-39 recorded twice, and every downstream window
    statistic was computed over duplicated episodes. The published checkpoint
    also omitted the optimiser, so a resumed run restarted Adam from zero
    moments while claiming to be the same run (`DESIGN.md` §8.2).

``test_resume_refuses_changed_config``
    Continuing a run under changed training hyperparameters is the class of
    error Phase 0 spent days undoing, and it is what made a completed directory
    silently resumable under a different `lr`.

``test_freeze_verification_detects_violation``
    The manuscript's freeze schedule was never implemented, and no frozen layer
    was ever verified to be frozen; recovering the published freeze map required
    diffing checkpoints months later. `verify_freeze` has to catch a
    single-ULP move in a declared-frozen layer, or the verification is theatre.

``test_permutation_preserves_norm``
    C3's interpretation rests on what the shuffle preserves. The test asserts
    the multiset and the Frobenius norm are preserved *and* that the singular
    value spectrum is **not** -- which is exactly why `DESIGN.md` §4.1 admits the
    spectral caveat and adds C3b, whose spectrum preservation is asserted too.

``test_value_recal_is_policy_invariant``
    `recalibrate_value_head` cannot improve the initial policy: under
    `Q = V + (A - baseline(A))` a change to the action-independent value stream
    leaves every argmax alone, and centring shifts Q by a constant. Asserting it
    is what stops a write-up claiming otherwise (E11, `DESIGN.md` §7).

``test_normalisation``
    Everything reported is on the normalised score, so a random policy must
    score 0 and the registered threshold 1 for every registry entry
    (`DESIGN.md` §5.1). The Acrobot case is checked separately: revision 1's
    multiplicative gate on the raw threshold gave -60, which is *stricter than
    solving*, and the normalised gate must lie between the measured random
    return and the threshold.

``test_statlib_reference_values``
    `ANALYSIS_PLAN.md` §6 pins exact critical values and MDEs and says they are
    "computed exactly, not asserted". If the primitives drift from the
    pre-registered numbers, the pre-registration is no longer describing the
    code, so the numbers are re-derived here and compared.

``test_stats_refuses_descriptive_metric`` / ``test_stats_no_pvalue_outside_family``
    The published §V.A ran a t-test on a metric its own §V.B declared
    descriptive-only and non-normal. `stats.py` must refuse a confirmatory test
    on a non-co-primary endpoint, and no p-value may appear outside the single
    pre-registered family of 8 (`ANALYSIS_PLAN.md` §2, §7, §8).

``test_n1_is_labelled``
    `STANDING_INSTRUCTIONS` S8 makes single-seed runs the *current* invocation
    mode, and `ANALYSIS_PLAN.md` §9 requires that such output emit no test and
    be stamped `PIPELINE VALIDATION - NOT A RESULT`.

Usage
-----
    python experiments/validate.py                 # the default suite
    python experiments/validate.py --quick         # skip the slower checks
    python experiments/validate.py --full          # add the expensive checks
    python experiments/validate.py --list
    python experiments/validate.py --test test_resume_equivalence

Exit code is non-zero if any case fails. A case that cannot run -- a missing
demo dataset, an absent dependency -- reports SKIP with the reason and does not
mask a failure, because a skipped guard is not a satisfied one.
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import io
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

# Set before TensorFlow is imported anywhere: quietens the C++ log spam without
# touching anything numerical. Deliberately *not* setting
# TF_ENABLE_ONEDNN_OPTS, because that changes the arithmetic and every recorded
# run in `runs_demo` was produced with the ambient setting.
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

DEFAULT_RUNS = os.path.join(_REPO, 'runs_demo')

PASS, FAIL, SKIP = 'PASS', 'FAIL', 'SKIP'


# ===========================================================================
# 1. Harness
# ===========================================================================
class Failed(AssertionError):
    """A guard did not hold. The message names the guard, not the symptom."""


class Skipped(Exception):
    """A precondition for the case is absent; the case ran nothing."""


@dataclass
class Ctx:
    """Per-case context: options, temporary directories, and reported notes."""

    quick: bool = False
    full: bool = False
    runs: str = DEFAULT_RUNS
    notes: list[str] = field(default_factory=list)
    _tmp: list[str] = field(default_factory=list)

    # ---- reporting -------------------------------------------------------
    def note(self, text: str) -> None:
        """Record a measured quantity for the summary line.

        Used where the honest statement is a measured tolerance rather than a
        promise -- `DESIGN.md` §8.2: "Where bitwise determinism is unattainable,
        the achieved tolerance is measured and reported, not promised."
        """
        self.notes.append(str(text))

    def skip(self, reason: str) -> None:
        raise Skipped(reason)

    # ---- scratch space ---------------------------------------------------
    def tmpdir(self, prefix: str = 'validate_') -> str:
        path = tempfile.mkdtemp(prefix=prefix)
        self._tmp.append(path)
        return path

    def cleanup(self, keep: bool = False) -> None:
        if keep:
            return
        for path in self._tmp:
            shutil.rmtree(path, ignore_errors=True)
        self._tmp.clear()


@dataclass
class Case:
    name: str
    fn: Callable[[Ctx], None]
    slow: bool
    guard: str


_CASES: list[Case] = []


def case(guard: str, slow: bool = False):
    """Register a test case. `guard` names what the case protects."""
    def deco(fn: Callable[[Ctx], None]) -> Callable[[Ctx], None]:
        _CASES.append(Case(fn.__name__, fn, slow, guard))
        return fn
    return deco


def req(condition: Any, message: str) -> None:
    """Assert, with a message that says which guard failed and why it matters."""
    if not condition:
        raise Failed(message)


def same(a: Any, b: Any, message: str) -> None:
    if a != b:
        raise Failed(f'{message}\n    left : {a!r}\n    right: {b!r}')


def near(a: float, b: float, tol: float, message: str) -> None:
    if not (math.isfinite(a) and math.isfinite(b)) or abs(a - b) > tol:
        raise Failed(f'{message}\n    got {a!r}, expected {b!r} +/- {tol!r}')


# ===========================================================================
# 2. Environments: variants, interface wrappers, shift measurement
# ===========================================================================
def _dynamics_variants(quick: bool) -> list[tuple[str, str]]:
    """(base, variant) pairs for every same-interface variant the catalogue uses.

    Drawn from `registry` and `envs.VARIANT_FAMILIES` rather than restated, so a
    new level added to the catalogue is covered without touching this file.
    """
    from experiments import registry
    from src.dqn import envs

    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for _label, spec in registry.WIND_LEVELS + registry.GRAVITY_LEVELS:
        key = (registry.TARGET_ENV, spec)
        if key not in seen:
            seen.add(key)
            pairs.append(key)
    for family in envs.VARIANT_FAMILIES.values():
        base = family['base']
        for _label, params in family['levels']:
            if not params:
                continue
            spec = envs.EnvSpec(base, dict(params)).canonical()
            key = (base, spec)
            if key not in seen:
                seen.add(key)
                pairs.append(key)
    if quick:
        # One level per family is enough to catch an inert variant; the full
        # sweep is what catches an inert *level*.
        keep, families = [], set()
        for base, spec in pairs:
            fam = re.sub(r'=[^,]*', '', spec.split(':', 1)[1] if ':' in spec else '')
            if (base, fam) in families:
                continue
            families.add((base, fam))
            keep.append((base, spec))
        pairs = keep
    return pairs


@case('DESIGN.md §6.2 -- a parametric variant must change the simulation, '
      'not merely be accepted by it')
def test_env_variants_apply(ctx: Ctx) -> None:
    """Every registry variant's parameters are read back and are not inert."""
    from src.dqn import envs
    from src.dqn.shift import paired_trajectory_divergence

    interface_keys = {'pad_obs', 'pad_mode', 'extra_actions'}
    checked = 0

    # -- 1. requested parameters are read back off the constructed simulator.
    specs = [spec for _base, spec in _dynamics_variants(ctx.quick)]
    from experiments import registry
    specs.append(registry.INTERFACE_ENV)
    for spec in specs:
        es = envs.parse(spec)
        env, info = envs.make(es)
        try:
            for key, want in es.params.items():
                req(key in info['applied'],
                    f'{spec}: parameter {key!r} was requested but does not '
                    f'appear in the resolved record, so the manifest would '
                    f'document a request rather than the physics that ran '
                    f'(envs.make -> applied).')
                got = info['applied'][key]
                req(envs._close(got, want),
                    f'{spec}: {key}={want!r} was not read back ({got!r}). An '
                    f'override that did not stick produces a run labelled as a '
                    f'variant while simulating the default physics.')
            same((info['obs_dim'], info['act_dim']), (es.obs_dim, es.act_dim),
                 f'{spec}: the registry interface and the constructed '
                 f'interface disagree; transfer keys on this.')
            checked += 1

            # -- 2. CartPole's derived attributes must be recomputed, or the
            #       variant is inert no matter what was read back.
            if es.env_id == 'CartPole-v1' and (
                    {'masspole', 'masscart', 'length'} & set(es.params)):
                u = env.unwrapped
                near(float(u.total_mass), float(u.masscart) + float(u.masspole),
                     1e-12,
                     f'{spec}: total_mass is not masscart+masspole. CartPole '
                     f'derives it once at construction, so setting masspole '
                     f'without recomputing leaves the simulation on default '
                     f'physics and the "variant" is not one.')
                near(float(u.polemass_length),
                     float(u.masspole) * float(u.length), 1e-12,
                     f'{spec}: polemass_length is not masspole*length; see '
                     f'envs._DERIVED.')
                req(info['derived_recomputed'],
                    f'{spec}: no derived attribute was recomputed, so the '
                    f'recomputation is not being recorded either.')
                # And it must actually differ from the default, or the guard
                # would pass on an environment it never touched.
                base_env, _ = envs.make('CartPole-v1')
                try:
                    req(abs(float(u.total_mass)
                            - float(base_env.unwrapped.total_mass)) > 1e-9
                        or abs(float(u.polemass_length)
                               - float(base_env.unwrapped.polemass_length))
                        > 1e-9,
                        f'{spec}: the derived attributes equal the base '
                        f'environment\'s, so nothing about the physics '
                        f'changed.')
                finally:
                    base_env.close()
        finally:
            env.close()

    # -- 3. an unknown parameter is refused rather than silently ignored.
    try:
        envs.EnvSpec('CartPole-v1', {'not_a_parameter': 1.0})
    except ValueError:
        pass
    else:
        raise Failed('envs.EnvSpec accepted an untunable parameter. A typo in '
                     'an experiment definition would then produce a silent '
                     'default instead of a loud failure.')

    # -- 4. the variant must be measurably different from its base. "The
    #       parameter was accepted" and "the dynamics changed" are different
    #       claims, and only the second one makes the level a level.
    inert: list[str] = []
    episodes = 3 if ctx.quick else 5
    for base, spec in _dynamics_variants(ctx.quick):
        es = envs.parse(spec)
        if interface_keys & set(es.params):
            rep = paired_trajectory_divergence(base, spec, episodes=2,
                                               max_steps=60,
                                               run_self_check=False)
            req(rep.get('defined') is False,
                f'{spec}: an interface-changing variant shares no state space '
                f'with its base, so a trajectory distance must be refused as '
                f'undefined (DESIGN.md §6.3), not computed.')
            continue
        rep = paired_trajectory_divergence(base, spec, episodes=episodes,
                                           max_steps=120,
                                           run_self_check=False)
        req(rep.get('defined'), f'{spec}: divergence against {base} is not '
                                f'defined but the interfaces match.')
        if float(rep['terminal_divergence']) <= 0.0:
            inert.append(spec)
    req(not inert,
        f'these variants are INERT -- identical trajectories to their base '
        f'environment under identical states and actions, so they are labelled '
        f'as a shift level and are not one: {inert}')
    ctx.note(f'{checked} variants read back, '
             f'{len(_dynamics_variants(ctx.quick))} checked for inertness')


@case('DESIGN.md §6.4 -- the interface-change corner must change the interface '
      'and nothing else')
def test_interface_wrapper_preserves_dynamics(ctx: Ctx) -> None:
    """Padded/extended LunarLander reproduces the base trajectory bit-exactly."""
    import numpy as np

    from experiments import registry
    from src.dqn import envs

    base_spec = registry.TARGET_ENV
    pad_spec = registry.INTERFACE_ENV
    b_es, p_es = envs.parse(base_spec), envs.parse(pad_spec)
    req(p_es.changes_interface_only(b_es),
        f'{pad_spec} is not recognised as an interface-only change of '
        f'{base_spec}; the corner of DESIGN.md §6.4 would then be a dynamics '
        f'change as well.')
    n_extra_actions = int(p_es.params.get('extra_actions', 0) or 0)
    n_pad = int(p_es.params.get('pad_obs', 0) or 0)
    req(n_extra_actions >= 1 and n_pad >= 1,
        f'{pad_spec} pads neither the observation nor the action set, so it '
        f'exercises none of the partial-copy/head-reinit mechanics it exists '
        f'for.')

    base, base_info = envs.make(base_spec)
    pad, pad_info = envs.make(pad_spec)
    try:
        same(base_info['obs_dim'] + n_pad, pad_info['obs_dim'],
             'padded observation dimensionality does not match pad_obs')
        same(base_info['act_dim'] + n_extra_actions, pad_info['act_dim'],
             'extended action cardinality does not match extra_actions')

        n_base_actions = int(base_info['act_dim'])
        padding_ever_nonzero = False
        for seed in (7, 11, 23):
            ob, _ = base.reset(seed=seed)
            op, _ = pad.reset(seed=seed)
            ob = np.asarray(ob, dtype=np.float32)
            op = np.asarray(op, dtype=np.float32)
            req(np.array_equal(ob, op[:len(ob)]),
                f'seed {seed}: the padded environment\'s reset observation '
                f'differs from the base environment\'s in the shared '
                f'dimensions, so the two are not the same episode.')
            rng = np.random.default_rng(seed)
            for step in range(200):
                # An action that aliases an existing one: `a + n` maps back to
                # `a` under DuplicateActions, so the achievable behaviour is
                # unchanged by construction.
                action = int(rng.integers(0, min(n_base_actions,
                                                 n_extra_actions)))
                alias = action + n_base_actions
                same(alias % n_base_actions, action,
                     'the alias does not map back to the base action; the '
                     'test would then be comparing two different policies')
                ob, rb, tb, ub, _ = base.step(action)
                op, rp, tp, up, _ = pad.step(alias)
                ob = np.asarray(ob, dtype=np.float32)
                op = np.asarray(op, dtype=np.float32)
                req(np.array_equal(ob, op[:len(ob)]),
                    f'seed {seed} step {step}: the padded environment\'s '
                    f'dynamics diverge from the base environment\'s under an '
                    f'aliased action (max abs difference '
                    f'{float(np.max(np.abs(ob - op[:len(ob)])))!r}). The '
                    f'same-dynamics/changed-interface corner would then '
                    f'confound interface with dynamics -- the confound it '
                    f'exists to remove.')
                same((float(rb), bool(tb), bool(ub)),
                     (float(rp), bool(tp), bool(up)),
                     f'seed {seed} step {step}: reward or termination differs '
                     f'under an aliased action')
                if np.any(op[len(ob):] != 0.0):
                    padding_ever_nonzero = True
                if tb or ub:
                    break
        req(padding_ever_nonzero,
            'the padded dimensions are always zero. A linear layer maps a zero '
            'input to no contribution, so the freshly initialised rows of '
            'trunk_fc1 would have no effect and the mechanics under test would '
            'not be exercised at all (env_wrappers.PadObservation).')
    finally:
        base.close()
        pad.close()
    ctx.note('trajectories bit-identical over 3 seeds x up to 200 steps')


@case('DESIGN.md §6.3 -- the divergence measure is validated by an '
      'identical-environment control that must return exactly zero')
def test_shift_selfcheck_zero(ctx: Ctx) -> None:
    """paired_trajectory_divergence of an environment against itself is 0."""
    from src.dqn.shift import paired_trajectory_divergence

    specs = ['CartPole-v1', 'LunarLander-v3']
    if not ctx.quick:
        specs += ['Acrobot-v1', 'LunarLander-v3:gravity=-4',
                  'LunarLander-v3:enable_wind=1,wind_power=15,'
                  'turbulence_power=1.5']
    for spec in specs:
        rep = paired_trajectory_divergence(spec, spec, episodes=3,
                                           max_steps=120, seed=0,
                                           run_self_check=True)
        for key in ('initial_divergence', 'terminal_divergence'):
            value = float(rep[key])
            req(value == 0.0,
                f'{spec}: {key} against itself is {value!r}, not exactly '
                f'zero. The environment carries hidden state across '
                f'instances, so no divergence number this function returns '
                f'means anything -- including the graded gravity and wind '
                f'series DESIGN.md §6.2 rests on.')
        check = rep.get('self_check') or {}
        req(check.get('pairing_valid') is True,
            f'{spec}: the built-in self-check did not certify the pairing '
            f'({check!r}). The check is the method\'s own validity condition.')
        req(float(check['identical_env_terminal_divergence']) == 0.0,
            f'{spec}: the self-check control returned a non-zero residual.')

    # A shifted pair must be non-zero, or "exactly zero" is uninformative.
    rep = paired_trajectory_divergence('LunarLander-v3',
                                       'LunarLander-v3:gravity=-4',
                                       episodes=3, max_steps=120, seed=0,
                                       run_self_check=False)
    req(float(rep['terminal_divergence']) > 0.0,
        'a shifted pair also returned zero divergence, so the self-check '
        'passing says nothing: the measure is returning zero unconditionally.')

    # A state-dependent policy cannot supply a shared action sequence, and the
    # function must refuse rather than silently answer a different question.
    try:
        paired_trajectory_divergence('CartPole-v1', 'CartPole-v1',
                                     policy=lambda s: 0, episodes=1,
                                     max_steps=10, run_self_check=False)
    except ValueError:
        pass
    else:
        raise Failed('paired_trajectory_divergence accepted a state-dependent '
                     'policy. Once the trajectories diverge the two rollouts '
                     'no longer share an action sequence, so the separation is '
                     'no longer attributable to the dynamics alone.')
    ctx.note(f'{len(specs)} environments self-checked at exactly zero')


# ===========================================================================
# 3. Run identity
# ===========================================================================
def _identity_variants() -> list[tuple[str, dict]]:
    """One-factor departures from a single transfer base configuration.

    A transfer condition is the base deliberately: `Config.digest` neutralises
    the transfer-only fields for a *scratch* run (they are inert there, and
    hashing them would duplicate every scratch baseline), so the coverage of
    those fields can only be tested from a transfer base.
    """
    return [
        # trajectory factors
        ('condition=transfer_untrained', {'condition': 'transfer_untrained'}),
        ('condition=transfer_permuted', {'condition': 'transfer_permuted'}),
        ('arch=mlp', {'arch': 'mlp'}),
        ('target_rule=vanilla', {'target_rule': 'vanilla'}),
        ('aggregation=max', {'aggregation': 'max'}),
        ('env=gravity-4', {'env': 'LunarLander-v3:gravity=-4'}),
        ('env=wind15', {'env': 'LunarLander-v3:enable_wind=1,wind_power=15,'
                               'turbulence_power=1.5'}),
        ('source_env=Acrobot', {'source_env': 'Acrobot-v1'}),
        ('seed=1', {'seed': 1}),
        # the transfer protocol
        ('transfer_set=trunk', {'transfer_set': 'trunk'}),
        ('transfer_set=described', {'transfer_set': 'described'}),
        ('input_policy=reinit', {'input_policy': 'reinit'}),
        ('input_policy=redraw_matched', {'input_policy': 'redraw_matched'}),
        ('head_policy=partial', {'head_policy': 'partial'}),
        ('freeze_group=value', {'freeze_group': 'value'}),
        ('freeze_updates=20k', {'freeze_updates': 20_000}),
        ('freeze_updates=0', {'freeze_updates': 0}),
        ('permute_scope=units', {'permute_scope': 'units'}),
        ('permute_kind=spectrum', {'permute_kind': 'spectrum'}),
        ('value_recal=center', {'value_recal': 'center'}),
        ('reset_head_at_unfreeze', {'reset_head_at_unfreeze': True}),
        ('shrink_perturb=0.2', {'shrink_perturb': 0.2}),
        # optimisation
        ('lr=1e-4', {'lr': 1e-4}),
        ('gamma=0.95', {'gamma': 0.95}),
        ('batch_size=32', {'batch_size': 32}),
        ('target_update=soft', {'target_update': 'soft'}),
        ('target_update_freq=500', {'target_update_freq': 500}),
        ('tau=0.01', {'tau': 0.01}),
        ('hidden=(64,64)', {'hidden': (64, 64)}),
        ('head_units=32', {'head_units': 32}),
        ('num_episodes=500', {'num_episodes': 500}),
        ('max_steps=500', {'max_steps': 500}),
        ('learning_starts=500', {'learning_starts': 500}),
        ('train_every=2', {'train_every': 2}),
        ('replay_capacity=50k', {'replay_capacity': 50_000}),
        ('grad_clip_norm=5', {'grad_clip_norm': 5.0}),
        ('epsilon_anneal_episodes=300', {'epsilon_anneal_episodes': 300}),
        ('epsilon_min=0.05', {'epsilon_min': 0.05}),
        # measurement -- changes the reported number, so it changes identity
        ('eval_every=20', {'eval_every': 20}),
        ('eval_episodes=10', {'eval_episodes': 10}),
        ('final_eval_episodes=50', {'final_eval_episodes': 50}),
        ('final_eval_checkpoints=1', {'final_eval_checkpoints': 1}),
        ('prefix_checkpoints=(250,500)', {'prefix_checkpoints': (250, 500)}),
        ('diag_states=256', {'diag_states': 256}),
        ('probe_steps=0', {'probe_steps': 0}),
        ('probe_transitions=1000', {'probe_transitions': 1000}),
    ]


#: Fields that must NOT enter run identity. `label` so that renaming an arm for
#: a table does not orphan its runs; `out_root` and `source_checkpoint` because
#: they are paths and moving the run tree must not change what a run *is*;
#: `experiment` because an experiment is a set of runs, not a property of one --
#: which is what lets six catalogue experiments share an identical run instead
#: of training it six times (`config.Config.run_dir`).
_NON_IDENTITY = [
    ('label', {'label': 'freezedur-K10k'}),
    ('experiment', {'experiment': 'E9'}),
    ('out_root', {'out_root': os.path.join('somewhere', 'else')}),
    ('source_checkpoint', {'source_checkpoint': os.path.join(
        'moved', 'tree', 'model.keras')}),
    ('notes', {'notes': 'a note about this run'}),
    ('keep_buffer', {'keep_buffer': True}),
    ('log_diagnostics', {'log_diagnostics': False}),
    ('checkpoint_seconds', {'checkpoint_seconds': 30}),
]


@case('DESIGN.md §11 defect 0 -- nine conditions from six experiments '
      'collapsed onto one run directory')
def test_run_identity_no_collisions(ctx: Ctx) -> None:
    """One-factor departures land in distinct directories; bookkeeping does not."""
    from experiments import registry
    from src.dqn.config import (BOOKKEEPING_FIELDS, IDENTITY_FIELDS, Config,
                                MEASUREMENT_FIELDS, TRAJECTORY_FIELDS)

    base_kwargs = dict(
        experiment='validate', label='base',
        arch='dueling', target_rule='double', condition='transfer',
        aggregation='mean', env='LunarLander-v3', source_env='CartPole-v1',
        source_checkpoint=os.path.join('runs', 'scratch', 'aaa', 's00',
                                       'model.keras'),
        seed=0, out_root='runs', **registry.PROTOCOL)
    base = Config(**base_kwargs)

    variants = _identity_variants()
    req(len(variants) >= 12,
        f'only {len(variants)} one-factor variants are exercised; the '
        f'assignment requires at least 12.')

    configs: dict[str, Config] = {'base': base}
    for name, override in variants:
        kwargs = dict(base_kwargs)
        kwargs.update(override)
        # Two structural constraints the schema imposes, honoured rather than
        # worked around: aggregation applies to the dueling net only, and a
        # freeze group must exist in the chosen architecture.
        if kwargs['arch'] == 'mlp':
            kwargs['aggregation'] = 'mean'
            if kwargs.get('freeze_group') not in ('trunk', 'trunk_fc1',
                                                  'trunk_fc2', 'head', 'all',
                                                  'none'):
                kwargs['freeze_group'] = 'trunk'
        if kwargs.get('freeze_updates') == 0:
            kwargs['reset_head_at_unfreeze'] = False
            kwargs['shrink_perturb'] = 0.0
        configs[name] = Config(**kwargs)

    dirs: dict[str, list[str]] = {}
    for name, cfg in configs.items():
        dirs.setdefault(cfg.run_dir(), []).append(name)
    collisions = {d: names for d, names in dirs.items() if len(names) > 1}
    req(not collisions,
        f'{len(collisions)} run directory/ies are shared by configurations '
        f'that differ in one experimental factor. That is exactly the defect '
        f'that forced design revision 2: a completed directory is silently '
        f'resumed rather than refused, so one experiment\'s data would be '
        f'reported as several. Collisions: '
        f'{ {d: n for d, n in collisions.items()} }')
    same(len(dirs), len(configs),
         'distinct configurations did not produce distinct run directories')

    # The legacy naming scheme, shown to collapse them -- so the test says what
    # the digest is buying rather than only that it is present.
    legacy = {(c.env, c.arch, c.target_rule, c.condition, c.seed)
              for c in configs.values()}
    req(len(legacy) < len(configs),
        'the published naming scheme <env>/<arch>-<rule>-<mode>-s<NN> did not '
        'collapse any of these configurations, which means this test is not '
        'exercising the fields the old scheme omitted.')
    ctx.note(f'{len(configs)} configs -> {len(dirs)} dirs; the legacy scheme '
             f'gives {len(legacy)}')

    # Bookkeeping must not change identity.
    for name, override in _NON_IDENTITY:
        kwargs = dict(base_kwargs)
        kwargs.update(override)
        other = Config(**kwargs)
        same(other.run_digest(), base.run_digest(),
             f'changing the bookkeeping field {name!r} changed the run '
             f'digest. {name!r} is declared BOOKKEEPING in config.py, and a '
             f'digest that covers it means renaming an arm, moving the run '
             f'tree or toggling a log orphans every existing run.')
        same(other.trajectory_digest(), base.trajectory_digest(),
             f'{name!r} changed the trajectory digest, which is what resume is '
             f'checked against: a resume would then be refused for a change '
             f'that cannot affect training.')
        if name != 'out_root':
            same(other.run_dir(), base.run_dir(),
                 f'{name!r} changed the run directory')

    # Identity is exactly trajectory u measurement, and the three sets partition
    # the schema. Asserted here as well as in the classification test because a
    # digest over the wrong *union* is a different defect from an unclassified
    # field.
    same(set(IDENTITY_FIELDS), set(TRAJECTORY_FIELDS) | set(MEASUREMENT_FIELDS),
         'IDENTITY_FIELDS is not the union of TRAJECTORY_FIELDS and '
         'MEASUREMENT_FIELDS, so run identity covers something that changes '
         'neither the trajectory nor the measurement, or misses something that '
         'does.')
    req(not (set(IDENTITY_FIELDS) & set(BOOKKEEPING_FIELDS)),
        'a bookkeeping field is inside run identity')

    # The same property, over the catalogue the runner would actually execute.
    # `all_jobs` de-duplicates by run directory, which would *hide* a collision,
    # so the per-experiment job lists are enumerated undeduplicated.
    seeds = [0] if ctx.quick else [0, 1]
    by_dir: dict[str, set[tuple[str, str]]] = {}
    owners: dict[str, set[str]] = {}
    n_jobs = 0
    for exp in registry.EXPERIMENTS:
        for job in registry.jobs(exp, seeds=seeds, out_root='runs'):
            n_jobs += 1
            key = job.cfg.run_dir()
            by_dir.setdefault(key, set()).add(
                (job.cfg.trajectory_digest(), job.cfg.measurement_digest()))
            owners.setdefault(key, set()).add(f'{exp}/{job.arm}')
    clashes = {d: sorted(owners[d]) for d, ids in by_dir.items() if len(ids) > 1}
    req(not clashes,
        f'{len(clashes)} catalogue run director(ies) are claimed by jobs whose '
        f'configurations differ. Identical configurations are deliberately '
        f'SHARED between experiments; differing ones must never be. '
        f'{clashes}')
    shared = sum(1 for d in owners if len(owners[d]) > 1)
    ctx.note(f'{n_jobs} catalogue jobs at seeds {seeds} -> {len(by_dir)} dirs, '
             f'{shared} deliberately shared, 0 conflicting')


@case('config.py -- a field added without being classified must raise at '
      'import, or the digest silently stops covering it')
def test_field_classification_complete(ctx: Ctx) -> None:
    """Every Config field is in exactly one of the three field sets."""
    import dataclasses

    from src.dqn import config as cfgmod

    declared = {f.name for f in dataclasses.fields(cfgmod.Config)}
    traj = set(cfgmod.TRAJECTORY_FIELDS)
    meas = set(cfgmod.MEASUREMENT_FIELDS)
    book = set(cfgmod.BOOKKEEPING_FIELDS)

    unclassified = sorted(declared - (traj | meas | book))
    req(not unclassified,
        f'Config fields {unclassified} are in none of TRAJECTORY_FIELDS, '
        f'MEASUREMENT_FIELDS or BOOKKEEPING_FIELDS. An unclassified field is '
        f'outside the run digest, so two runs that differ in it share a '
        f'directory and the second silently resumes the first.')
    spurious = sorted((traj | meas | book) - declared)
    req(not spurious,
        f'{spurious} are classified but are not Config fields; the '
        f'classification has drifted from the schema.')
    for a_name, a, b_name, b in (('TRAJECTORY', traj, 'MEASUREMENT', meas),
                                 ('TRAJECTORY', traj, 'BOOKKEEPING', book),
                                 ('MEASUREMENT', meas, 'BOOKKEEPING', book)):
        overlap = sorted(a & b)
        req(not overlap,
            f'{overlap} appear in both {a_name}_FIELDS and {b_name}_FIELDS. '
            f'"Exactly one" is what makes the digest\'s coverage decidable.')
    same(len(traj) + len(meas) + len(book), len(declared),
         'the three field sets do not partition the schema')

    # Every transfer-only field must be a trajectory field, or neutralising it
    # for a scratch run would drop something a scratch run reads.
    req(set(cfgmod.TRANSFER_ONLY_FIELDS) <= traj,
        f'TRANSFER_ONLY_FIELDS contains fields that are not trajectory fields: '
        f'{sorted(set(cfgmod.TRANSFER_ONLY_FIELDS) - traj)}. Those are '
        f'neutralised in a scratch run\'s digest, so a field a scratch run '
        f'actually reads would drop out of its identity.')

    # The guard has to fire, not merely exist. Each perturbation below is one of
    # the three ways the classification can rot.
    original = cfgmod.TRAJECTORY_FIELDS
    victim = original[0]
    perturbations = [
        ('a field removed from every set', tuple(original[1:])),
        ('a name that is not a field', original + ('not_a_config_field',)),
        ('a field placed in two sets',
         original + (cfgmod.BOOKKEEPING_FIELDS[0],)),
    ]
    try:
        for label, replacement in perturbations:
            cfgmod.TRAJECTORY_FIELDS = replacement
            try:
                cfgmod._check_field_classification()
            except RuntimeError:
                continue
            raise Failed(
                f'_check_field_classification() accepted {label} '
                f'(victim field {victim!r}). The guard is inert, so the next '
                f'field added to Config will silently fall outside the run '
                f'digest -- the defect that let nine conditions share one '
                f'directory.')
    finally:
        cfgmod.TRAJECTORY_FIELDS = original
    cfgmod._check_field_classification()          # still healthy after restore
    ctx.note(f'{len(declared)} fields partitioned; 3 perturbations all caught')


# ===========================================================================
# 4. Schedules and randomness
# ===========================================================================
@case('DESIGN.md §3.2, §2.4 RQ6 -- epsilon is a closed form in env steps, '
      'coupled to neither the evaluation cadence nor the budget')
def test_epsilon_closed_form(ctx: Ctx) -> None:
    """Monotone, floors at epsilon_anneal_episodes, blind to budget and cadence."""
    import inspect

    from src.dqn.config import Config

    base = Config(experiment='validate', condition='scratch',
                  env='LunarLander-v3', seed=0)
    anneal = int(base.epsilon_anneal_episodes)

    # -- monotone, and exactly at the endpoints.
    grid = list(range(0, anneal + 1, max(1, anneal // 400)))
    grid += [anneal - 1, anneal, anneal + 1, 2 * anneal, 10 * anneal]
    grid = sorted(set(max(0, g) for g in grid))
    trace = [base.epsilon_at(g) for g in grid]
    for i in range(1, len(trace)):
        req(trace[i] <= trace[i - 1] + 0.0,
            f'epsilon is not monotone non-increasing: '
            f'epsilon({grid[i]})={trace[i]!r} > epsilon({grid[i - 1]})='
            f'{trace[i - 1]!r}. A non-monotone exploration schedule makes '
            f'"elapsed steps" the wrong index for it.')
    same(base.epsilon_at(0), float(base.epsilon_start),
         'epsilon does not start at epsilon_start')
    same(base.epsilon_at(anneal), float(base.epsilon_min),
         f'epsilon does not floor EXACTLY at epsilon_anneal_episodes={anneal}. '
         f'An off-by-a-fraction floor makes the schedule\'s horizon a '
         f'different number from the one the config declares, and '
         f'epsilon_anneal_episodes was promoted to a factor (DESIGN.md §3) so '
         f'that budget and exploration horizon are not confounded.')
    for beyond in (anneal + 1, 2 * anneal, 10 * anneal):
        same(base.epsilon_at(beyond), float(base.epsilon_min),
             f'epsilon at {beyond} steps is not the floor')

    # -- blind to the budget: the identifying condition for RQ6. If epsilon
    #    read num_episodes, a 500-episode prefix of a 1000-episode run would
    #    NOT be what a 500-episode run produced, and E10 would not exist.
    def trace_of(**overrides) -> list[float]:
        cfg = Config(experiment='validate', condition='scratch',
                     env='LunarLander-v3', seed=0, **overrides)
        return [cfg.epsilon_at(g) for g in grid]

    reference = trace_of()
    for label, overrides in (
            ('num_episodes=500', {'num_episodes': 500}),
            ('num_episodes=2000', {'num_episodes': 2000}),
            ('max_steps=500', {'max_steps': 500}),
    ):
        same(trace_of(**overrides), reference,
             f'the epsilon trace changed under {label}. The budget analysis '
             f'of DESIGN.md §2.4 RQ6 is licensed ONLY because the exploration '
             f'schedule never reads the budget; if it does, an episode-500 '
             f'prefix is not a 500-episode run and E10 is not free -- it is '
             f'not valid.')

    # -- blind to the measurement cadence: DESIGN.md §3.2's central claim, and
    #    the published defect (the decay lived inside the evaluation branch).
    for label, overrides in (
            ('eval_every=1', {'eval_every': 1}),
            ('eval_every=100', {'eval_every': 100}),
            ('eval_episodes=1', {'eval_episodes': 1}),
            ('eval_episodes=50', {'eval_episodes': 50}),
            ('final_eval_episodes=10', {'final_eval_episodes': 10}),
            ('diag_states=32', {'diag_states': 32}),
    ):
        same(trace_of(**overrides), reference,
             f'the epsilon trace changed under {label}. In the published code '
             f'the decay lived inside the evaluation branch, which made the '
             f'exploration schedule a function of eval_every; DESIGN.md §3.2 '
             f'asserts this trace is bit-identical when the cadence changes.')

    # Inspect the compiled function's name references, not its source text.
    # The docstring legitimately mentions `num_episodes` in order to say that
    # the schedule never reads it, and a substring match on the source flagged
    # that sentence as the defect it was describing.
    referenced = set(Config.epsilon_at.__code__.co_names)
    body = inspect.getsource(Config.epsilon_at)
    doc = Config.epsilon_at.__doc__ or ''
    if doc:
        body = body.replace(doc, '')
    for forbidden in ('num_episodes', 'eval_every', 'eval_episodes'):
        req(forbidden not in referenced and forbidden not in body,
            f'Config.epsilon_at reads {forbidden!r}. Even if the current '
            f'arithmetic happens to agree, a schedule that can read the '
            f'budget or the cadence is one edit away from the published '
            f'defect, and RQ6\'s identifying condition depends on it not '
            f'being able to.')
    ctx.note(f'epsilon_at references {sorted(referenced)} and no budget or '
             f'cadence field')

    if ctx.full:
        # The same claim at run level, on real trajectories rather than on the
        # closed form alone.
        traces = []
        for eval_every in (2, 3):
            root = ctx.tmpdir('eps_')
            cfg = _tiny_cfg(out_root=root, num_episodes=8,
                            eval_every=eval_every, arch='mlp',
                            target_rule='double', env='CartPole-v1', seed=9)
            _train(cfg)
            rows = _metric_rows(cfg.run_dir())
            traces.append([(r['episode'], r['epsilon'], r['return'])
                           for r in rows])
        same(traces[0], traces[1],
             'two runs differing only in eval_every produced different '
             'per-episode epsilon or returns; the evaluation cadence is '
             'perturbing training.')
        ctx.note('run-level epsilon and return traces identical across '
                 'eval_every in {2, 3}')
    ctx.note(f'{len(grid)} grid points; floor exact at {anneal} steps')


@case('DESIGN.md §8.1 -- turning a diagnostic on must not change the training '
      'trajectory')
def test_diagnostics_are_inert(ctx: Ctx) -> None:
    """Two runs identical but for log_diagnostics give the same per-episode returns."""
    traces: dict[bool, list[tuple]] = {}
    for flag in (True, False):
        # A separate out_root per run: `log_diagnostics` is BOOKKEEPING, so both
        # runs share a run digest by design, and sharing a directory would make
        # the second run resume the first instead of repeating it.
        root = ctx.tmpdir('diag_')
        cfg = _tiny_cfg(out_root=root, log_diagnostics=flag, arch='dueling',
                        target_rule='double', env='CartPole-v1', seed=3,
                        num_episodes=8)
        _train(cfg)
        rows = _metric_rows(cfg.run_dir())
        traces[flag] = [(r['episode'], r['return'], r['length'], r['epsilon'],
                         r['env_steps'], r['updates']) for r in rows]

    on, off = traces[True], traces[False]
    same(len(on), len(off), 'the two runs completed different episode counts')
    req(max(r['updates'] for r in _metric_rows_cache[-1]) > 0
        if _metric_rows_cache else True,
        'no gradient update occurred, so the buffer-sampling coupling this '
        'test exists to detect was never exercised.')
    mismatched = [(a, b) for a, b in zip(on, off) if a != b]
    # Bound before the message is built: an f-string argument is evaluated
    # eagerly, so indexing mismatched[0] inside the call raised IndexError on
    # the *passing* path and reported a guard failure that had not occurred.
    first_mismatch = mismatched[0] if mismatched else None
    req(not mismatched,
        f'{len(mismatched)} of {len(on)} episodes differ between '
        f'log_diagnostics=True and log_diagnostics=False. This is the test '
        f'the published code would have failed: it drew its diagnostic states '
        f'from the training replay buffer, which advanced the buffer\'s '
        f'generator, so every subsequent minibatch and therefore the whole '
        f'trajectory changed when diagnostics were switched on. An ablation '
        f'in that regime measures its own instrumentation. First mismatch: '
        f'{first_mismatch}')

    # The diagnostics must actually have been computed in the True run, or the
    # comparison is between two runs that both did nothing.
    on_rows = _metric_rows(_last_run_dirs[0])
    diag_cols = ('v_abs_mean', 'a_abs_mean', 'a_spread', 'dead_unit_frac',
                 'grad_norm_trunk', 'effective_rank')
    present = {c for r in on_rows for c in diag_cols if r.get(c) is not None}
    req(present,
        f'log_diagnostics=True produced none of {diag_cols}, so the inertness '
        f'claim is untested: nothing was instrumented.')
    off_rows = _metric_rows(_last_run_dirs[1])
    absent = {c for r in off_rows for c in diag_cols if r.get(c) is not None}
    req(not absent,
        f'log_diagnostics=False still emitted {sorted(absent)}; the flag does '
        f'not control the instrumentation it names.')
    ctx.note(f'{len(on)} episodes bit-identical; diagnostics recorded: '
             f'{sorted(present)}')


@case('DESIGN.md §8.1 -- per-layer initialisation is what matches the 2x2 by '
      'seed and licenses the paired analysis')
def test_per_layer_seeding(ctx: Ctx) -> None:
    """mlp and dueling share their trunk initialisation at the same seed."""
    import numpy as np

    from src.dqn.networks import LAYER_GROUPS, build_q_network
    from src.dqn.seeding import Seeds

    def build(arch: str, seed: int, **kw):
        seeds = Seeds(seed)
        return build_q_network(8, 4, arch, (128, 128), 64, 'mean',
                               seeds.layer_seeds(LAYER_GROUPS[arch]['all']),
                               **kw)

    mlp = build('mlp', 17)
    duel = build('dueling', 17)
    for layer in ('trunk_fc1', 'trunk_fc2'):
        a = [np.asarray(w) for w in mlp.get_layer(layer).get_weights()]
        b = [np.asarray(w) for w in duel.get_layer(layer).get_weights()]
        same(len(a), len(b), f'{layer}: different number of weight tensors')
        for i, (x, y) in enumerate(zip(a, b)):
            req(np.array_equal(x, y),
                f'{layer} tensor {i} differs between mlp and dueling at the '
                f'same seed (max abs difference '
                f'{float(np.max(np.abs(x - y)))!r}). DESIGN.md §8.1 states the '
                f'trunk initialisation is shared exactly, which is what makes '
                f'the architecture contrast free of trunk-init noise and what '
                f'ANALYSIS_PLAN.md §2.1 cites to justify pairing. If it is '
                f'false, the pairing warrant is false too.')

    # Reproducible: the same seed twice gives the same network, layer by layer.
    again = build('dueling', 17)
    for layer_name in LAYER_GROUPS['dueling']['all']:
        for i, (x, y) in enumerate(zip(duel.get_layer(layer_name).get_weights(),
                                       again.get_layer(layer_name).get_weights())):
            req(np.array_equal(np.asarray(x), np.asarray(y)),
                f'{layer_name} tensor {i} is not reproducible at a fixed seed')

    # Different seeds differ, or "shared at a seed" is vacuous.
    other = build('dueling', 18)
    differing = [n for n in LAYER_GROUPS['dueling']['all']
                 if not np.array_equal(
                     np.asarray(duel.get_layer(n).get_weights()[0]),
                     np.asarray(other.get_layer(n).get_weights()[0]))]
    same(sorted(differing), sorted(LAYER_GROUPS['dueling']['all']),
         'some layers are identical across two different run seeds, so the '
         'seed is not reaching every layer initialiser and the seed set is '
         'not the sample it claims to be')

    # Partial seeding is refused: it looks reproducible and is not.
    try:
        build_q_network(8, 4, 'dueling', (128, 128), 64, 'mean',
                        {'trunk_fc1': 1})
    except KeyError:
        pass
    else:
        raise Failed('build_q_network accepted a partial layer_seeds mapping. '
                     'Partial seeding is worse than none, because the run '
                     'looks reproducible and is not.')
    ctx.note('trunk shared across arch at one seed; all 6 layers differ '
             'across seeds')


@case('DESIGN.md §8.1 -- named streams, so an ablation cannot perturb the '
      'machinery it does not touch')
def test_rng_stream_independence(ctx: Ctx) -> None:
    """Heavy use of the diag stream leaves the action stream's draws unchanged."""
    import numpy as np

    from src.dqn.seeding import STREAMS, Seeds

    def action_draws(consume_diag: int) -> list[float]:
        seeds = Seeds(42)
        if consume_diag:
            seeds.rng('diag').standard_normal(consume_diag)
            seeds.rng('diag').integers(0, 1_000, size=consume_diag)
        return [float(seeds.rng('action').random()) for _ in range(64)]

    baseline = action_draws(0)
    for consumed in (1, 1_000, 200_000):
        same(action_draws(consumed), baseline,
             f'drawing {consumed} values from the `diag` stream changed the '
             f'`action` stream\'s draws. With a shared generator, adding a '
             f'diagnostic shifts every subsequent draw and therefore perturbs '
             f'the training trajectory of arms that were supposed to be '
             f'untouched -- the ablation then measures its own '
             f'instrumentation.')

    # Every ordered pair of streams, not just diag -> action.
    for consumer in STREAMS:
        seeds = Seeds(7)
        seeds.rng(consumer).standard_normal(5_000)
        for observer in STREAMS:
            if observer == consumer:
                continue
            fresh = Seeds(7)
            same([float(seeds.rng(observer).random()) for _ in range(8)],
                 [float(fresh.rng(observer).random()) for _ in range(8)],
                 f'consuming the {consumer!r} stream changed the {observer!r} '
                 f'stream')

    # Distinct streams, and distinct recorded values, so the manifest can tell
    # them apart.
    values = {name: Seeds(3).value(name) for name in STREAMS}
    same(len(set(values.values())), len(STREAMS),
         f'two named streams derive the same recorded seed: {values}. Streams '
         f'that collide are one stream wearing two names.')

    # Index-addressable seeds must not depend on call history -- that is what
    # makes resume exact (seeding.py module docstring).
    a = Seeds(5)
    b = Seeds(5)
    b.rng('action').random(1000)
    b.rng('buffer').random(1000)
    for episode in (0, 1, 7, 700, 999):
        same(a.episode_seed(episode), b.episode_seed(episode),
             f'episode_seed({episode}) depends on how many draws happened '
             f'before it, so a resumed run would not replay the same episodes')
    for checkpoint in (-1, 0, 500):
        for index in (0, 3, 99):
            same(a.eval_seed('eval_final', checkpoint, index),
                 b.eval_seed('eval_final', checkpoint, index),
                 'eval_seed depends on call history')
            req(a.eval_seed('eval_monitor', checkpoint, index)
                != a.eval_seed('eval_final', checkpoint, index),
                f'the monitoring and held-out evaluation families share an '
                f'initial state at checkpoint {checkpoint}, index {index}. '
                f'The primary endpoint would then be measured on states that '
                f'were watched during training, so it is not held out '
                f'(DESIGN.md §5.2).')

    # An unknown stream is refused rather than invented.
    try:
        Seeds(0).rng('not_a_stream')
    except KeyError:
        pass
    else:
        raise Failed('Seeds.rng accepted an unknown stream name; a typo would '
                     'silently create an unrecorded generator.')
    ctx.note(f'{len(STREAMS)} streams, all {len(STREAMS) * (len(STREAMS) - 1)} '
             f'ordered pairs independent')


# ===========================================================================
# 5. Resume and metrics integrity
# ===========================================================================
#: Populated by `_train`, so the diagnostics test can look at the two runs it
#: just produced without threading paths through the assertions.
_last_run_dirs: list[str] = []
_metric_rows_cache: list[list[dict]] = []


def _tiny_cfg(out_root: str, **overrides):
    """A configuration small enough to run in seconds and still cross a freeze.

    Sized so gradient updates actually happen: the replay-buffer coupling, the
    optimiser state and the freeze boundary are the parts of the loop these
    tests are about, and a run that never updates exercises none of them.
    """
    from src.dqn.config import Config

    kwargs = dict(
        experiment='validate', label='validate', condition='scratch',
        arch='mlp', target_rule='double', env='CartPole-v1', seed=5,
        num_episodes=10, max_steps=150, eval_every=3, eval_episodes=1,
        final_eval_episodes=2, final_eval_checkpoints=1,
        prefix_checkpoints=(), batch_size=16, learning_starts=32,
        replay_capacity=2_000, diag_states=32, probe_steps=0,
        probe_transitions=0, checkpoint_seconds=10 ** 9, keep_buffer=True,
        out_root=out_root)
    kwargs.update(overrides)
    return Config(**kwargs)


def _train(cfg) -> dict:
    from src.dqn.train import train

    with contextlib.redirect_stdout(io.StringIO()):
        manifest = train(cfg)
    _last_run_dirs.append(cfg.run_dir())
    _metric_rows_cache.append(_metric_rows(cfg.run_dir()))
    return manifest


def _metric_rows(run_dir: str) -> list[dict]:
    path = os.path.join(run_dir, 'metrics.jsonl')
    rows = []
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return sorted(rows, key=lambda r: int(r['episode']))


class _SimulatedCrash(RuntimeError):
    """A kill between checkpoints, injected where a real one would land."""


def _train_until_crash(cfg, crash_at: int, checkpoint_at: int | None):
    """Run until `crash_at`, having last checkpointed at `checkpoint_at`.

    The injection point is `MetricsLog.append`, i.e. immediately after an
    episode's row would have been written -- which is exactly where a kill
    leaves the log ahead of the checkpoint and produces the duplication the
    published loop suffered.
    """
    from src.dqn.train import Trainer

    with contextlib.redirect_stdout(io.StringIO()):
        trainer = Trainer(cfg, argv=[])
        real_append = trainer.metrics.append

        def append(row: dict) -> None:
            episode = int(row['episode'])
            if episode == crash_at:
                raise _SimulatedCrash(f'injected crash at episode {episode}')
            real_append(row)
            if checkpoint_at is not None and episode == checkpoint_at:
                trainer.save_checkpoint(episode)

        trainer.metrics.append = append              # type: ignore[assignment]
        try:
            trainer.run()
        except _SimulatedCrash:
            pass
        else:
            raise Failed(f'the injected crash at episode {crash_at} never '
                         f'fired; the resume path was not exercised.')
        finally:
            trainer.metrics.close()
            for env in (getattr(trainer, 'env', None),
                        getattr(trainer, 'eval_env', None)):
                if env is not None:
                    with contextlib.suppress(Exception):
                        env.close()
    return trainer


@case('DESIGN.md §8.2 -- a run interrupted and resumed produces the same '
      'metrics as an uninterrupted one')
def test_resume_equivalence(ctx: Ctx) -> None:
    """Interrupt at a checkpoint, resume, and compare per-episode returns."""
    import numpy as np

    root_a = ctx.tmpdir('resume_a_')
    root_b = ctx.tmpdir('resume_b_')
    kwargs = dict(arch='mlp', target_rule='double', env='CartPole-v1', seed=5,
                  num_episodes=10)

    cfg_a = _tiny_cfg(out_root=root_a, **kwargs)
    _train(cfg_a)
    uninterrupted = _metric_rows(cfg_a.run_dir())

    cfg_b = _tiny_cfg(out_root=root_b, **kwargs)
    _train_until_crash(cfg_b, crash_at=5, checkpoint_at=4)
    partial = [int(r['episode']) for r in _metric_rows(cfg_b.run_dir())]
    same(partial, list(range(5)),
         f'after the injected crash the log holds {partial}, not episodes '
         f'0-4; the crash did not land where the test placed it')
    _train(_tiny_cfg(out_root=root_b, **kwargs))
    resumed = _metric_rows(cfg_b.run_dir())

    same([int(r['episode']) for r in resumed],
         [int(r['episode']) for r in uninterrupted],
         'the resumed run does not cover the same episodes')

    worst = 0.0
    worst_field = ''
    for field_name in ('return', 'length', 'epsilon', 'env_steps', 'updates'):
        for x, y in zip(uninterrupted, resumed):
            a, b = x.get(field_name), y.get(field_name)
            if a is None or b is None:
                continue
            diff = abs(float(a) - float(b))
            if diff > worst:
                worst, worst_field = diff, f'{field_name}@ep{x["episode"]}'
    # The claim DESIGN.md §8.2 licenses is "to the tolerance the platform
    # allows", with the achieved tolerance measured rather than promised.
    achieved = ('BITWISE (exactly 0)' if worst == 0.0
                else f'max abs difference {worst:.3e} at {worst_field}')
    ctx.note(f'resume equivalence achieved: {achieved}')
    tolerance = 1e-6
    req(worst <= tolerance,
        f'a resumed run diverged from an uninterrupted one by {worst:.3e} '
        f'at {worst_field}, beyond the {tolerance:.0e} tolerance. Resume must '
        f'restore the weights, the optimiser slots, the replay buffer and '
        f'every RNG stream; the published checkpoint omitted the optimiser, so '
        f'a resumed run restarted Adam from zero moments while claiming to be '
        f'the same run.')

    # The state that makes it work must be present in the checkpoint, not
    # incidentally unnecessary at this run length.
    with open(os.path.join(cfg_b.run_dir(), 'state.json'), encoding='utf-8') \
            as fh:
        state = json.load(fh)
    for key in ('episode', 'env_steps', 'update_counter', 'trajectory_digest',
                'rng_states'):
        req(key in state, f'state.json omits {key!r}; resume would guess it')
    req({'action', 'buffer'} <= set(state['rng_states']),
        f'the checkpointed RNG states {sorted(state["rng_states"])} omit the '
        f'streams that advance during training, so a resumed run would sample '
        f'a different sequence.')
    req(os.path.exists(os.path.join(cfg_b.run_dir(), 'optimizer.npz')),
        'no optimiser state was written; Adam would restart from zero moments')
    req(float(np.max([r['updates'] for r in resumed])) > 0,
        'no gradient update happened, so neither the optimiser nor the buffer '
        'restoration was exercised')


@case('DESIGN.md §8.2 -- a crash between checkpoints must not duplicate '
      'episodes in the metrics log')
def test_metrics_no_duplication_on_resume(ctx: Ctx) -> None:
    """After a crash past the last checkpoint the episode set is range(0, n)."""
    root = ctx.tmpdir('dupe_')
    kwargs = dict(arch='mlp', target_rule='double', env='CartPole-v1', seed=5,
                  num_episodes=10)
    cfg = _tiny_cfg(out_root=root, **kwargs)

    # Checkpoint at 3, keep logging to 6, then die: the log is three episodes
    # ahead of the recoverable state, which is precisely the published
    # situation (45-episode run, killed at 39, last checkpoint 30).
    _train_until_crash(cfg, crash_at=7, checkpoint_at=3)
    ahead = [int(r['episode']) for r in _metric_rows(cfg.run_dir())]
    same(ahead, list(range(7)),
         f'the crashed log holds {ahead}; the test needs the log to run ahead '
         f'of the last checkpoint or there is nothing to duplicate')

    _train(_tiny_cfg(out_root=root, **kwargs))
    rows = _metric_rows(cfg.run_dir())
    episodes = [int(r['episode']) for r in rows]
    same(episodes, list(range(int(cfg.num_episodes))),
         f'the episode index set is {episodes[:14]}..., not '
         f'range(0, {cfg.num_episodes}). The published loop appended on resume '
         f'without truncating: a 45-episode run interrupted at 39 whose last '
         f'checkpoint was 30 produced 54 rows with episodes 31-39 recorded '
         f'twice, and every downstream window statistic was then computed over '
         f'duplicated episodes.')
    same(len(episodes), len(set(episodes)),
         f'duplicate episodes in metrics.jsonl: '
         f'{sorted({e for e in episodes if episodes.count(e) > 1})}')

    from src.dqn.metrics import MetricsLog
    integrity = MetricsLog(os.path.join(cfg.run_dir(),
                                        'metrics.jsonl')).check(
                                            expected=int(cfg.num_episodes))
    req(integrity['contiguous'],
        f'the log\'s own integrity check reports problems after a '
        f'crash-and-resume: {integrity["problems"]}')
    manifest_path = os.path.join(cfg.run_dir(), 'manifest.json')
    with open(manifest_path, encoding='utf-8') as fh:
        manifest = json.load(fh)
    recorded = manifest['result']['metrics_integrity']
    req(recorded.get('contiguous'),
        f'the manifest records a non-contiguous log: {recorded}')
    same(int(manifest['result']['episodes_completed']), int(cfg.num_episodes),
         'episodes_completed disagrees with the log, which is the field the '
         'sweep uses to decide a run is finished')

    # The evaluation log must be de-duplicated too, or the noise floor and the
    # held-out endpoint are computed over repeated checkpoints.
    evals = [json.loads(l) for l in
             open(os.path.join(cfg.run_dir(), 'eval_episodes.jsonl'),
                  encoding='utf-8') if l.strip()]
    keys = [(r['stream'], r['checkpoint'], r['index']) for r in evals]
    same(len(keys), len(set(keys)),
         f'duplicate evaluation episodes after resume: '
         f'{sorted({k for k in keys if keys.count(k) > 1})[:6]}')
    ctx.note(f'{len(episodes)} episodes, {len(keys)} evaluation episodes, no '
             f'duplicates')


@case('DESIGN.md §8.2 -- resume is state-complete or refused')
def test_resume_refuses_changed_config(ctx: Ctx) -> None:
    """A changed trajectory digest, or a missing optimiser, refuses the resume."""
    from src.dqn.train import Trainer

    root = ctx.tmpdir('refuse_')
    cfg = _tiny_cfg(out_root=root, num_episodes=3, arch='mlp',
                    target_rule='double', env='CartPole-v1', seed=5)
    _train(cfg)
    run_dir = cfg.run_dir()
    state_path = os.path.join(run_dir, 'state.json')

    with open(state_path, encoding='utf-8') as fh:
        original = json.load(fh)
    same(original['trajectory_digest'], cfg.trajectory_digest(),
         'the checkpoint does not record the trajectory digest of the config '
         'that wrote it, so a resume has nothing to check against')

    # Resuming an unchanged configuration must NOT raise, or the guard is just
    # a broken resume.
    with contextlib.redirect_stdout(io.StringIO()):
        Trainer(cfg, argv=[])

    # -- 1. a changed training configuration.
    changed = dict(original)
    changed['trajectory_digest'] = 'deadbeefdeadbeefdeadbeefdeadbeef'
    with open(state_path, 'w', encoding='utf-8') as fh:
        json.dump(changed, fh)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            Trainer(cfg, argv=[])
    except RuntimeError as exc:
        req('refusing to resume' in str(exc),
            f'resume was refused, but not for the stated reason: {exc}')
    else:
        raise Failed(
            'a resume proceeded under a different trajectory digest. This is '
            'the class of error Phase 0 spent days undoing: a run declaring '
            'lr=1e-3 resumed a completed lr=5e-4 directory, trained zero '
            'episodes, and emitted a manifest carrying the new values over '
            'byte-identical metrics -- five experiments fabricated from one '
            'experiment\'s data with every invariant check passing.')

    # -- 2. a checkpoint without optimiser state.
    with open(state_path, 'w', encoding='utf-8') as fh:
        json.dump(original, fh)
    optim = os.path.join(run_dir, 'optimizer.npz')
    moved = optim + '.moved'
    os.replace(optim, moved)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            Trainer(cfg, argv=[])
    except RuntimeError as exc:
        req('optimiser' in str(exc) or 'optimizer' in str(exc),
            f'resume was refused, but not for the missing optimiser: {exc}')
    else:
        raise Failed(
            'a resume proceeded with no optimiser state. The published '
            'checkpoint omitted it, so the resumed run restarted Adam from '
            'zero moments with iterations == 0 -- a different optimiser '
            'trajectory wearing the same run\'s name.')
    finally:
        os.replace(moved, optim)
    ctx.note('both refusals fire; an unchanged resume still proceeds')


# ===========================================================================
# 6. Freeze verification, the control constructions, and normalisation
# ===========================================================================
@case('DESIGN.md §1, §8.4 -- a frozen layer whose fingerprint moved must be '
      'caught')
def test_freeze_verification_detects_violation(ctx: Ctx) -> None:
    """verify_freeze catches a single-ULP move in a declared-frozen layer."""
    import numpy as np

    from src.dqn.networks import LAYER_GROUPS, build_q_network
    from src.dqn.seeding import Seeds
    from src.dqn.transfer import verify_freeze, weight_fingerprint

    arch = 'dueling'
    layers = LAYER_GROUPS[arch]['all']
    frozen_set = LAYER_GROUPS[arch]['trunk']
    trainable_set = tuple(n for n in layers if n not in frozen_set)

    def fresh():
        seeds = Seeds(21)
        return build_q_network(8, 4, arch, (128, 128), 64, 'mean',
                              seeds.layer_seeds(layers))

    def nudge(model, name: str, ulps: int = 1) -> None:
        layer = model.get_layer(name)
        weights = [np.array(w) for w in layer.get_weights()]
        kernel = weights[0]
        flat = kernel.reshape(-1)
        target = flat[0]
        for _ in range(ulps):
            target = np.nextafter(target, np.float32(np.inf), dtype=np.float32)
        flat[0] = target
        layer.set_weights([flat.reshape(kernel.shape)] + weights[1:])

    # -- the clean case: frozen layers held, every trainable layer moved.
    model = fresh()
    before = weight_fingerprint(model)
    for name in trainable_set:
        nudge(model, name)
    verdict = verify_freeze(before, weight_fingerprint(model), frozen_set)
    same(verdict['frozen_but_changed'], [],
         f'a healthy freeze window was reported as violated: {verdict}')
    same(verdict['trainable_but_unchanged'], [],
         f'layers that moved were reported as inert: {verdict}')
    req(verdict['ok'] is True, f'a healthy window was not ok: {verdict}')

    # -- the violation: one declared-frozen layer moves by ONE ULP.
    model = fresh()
    before = weight_fingerprint(model)
    for name in trainable_set:
        nudge(model, name)
    nudge(model, 'trunk_fc1', ulps=1)
    verdict = verify_freeze(before, weight_fingerprint(model), frozen_set)
    same(verdict['frozen_but_changed'], ['trunk_fc1'],
         f'a one-ULP move in a declared-frozen layer was not caught: '
         f'{verdict}. Recovering the published study\'s freeze map required '
         f'diffing saved checkpoints months after the fact, which is only '
         f'necessary when the run does not record and check it. A '
         f'verification that tolerates drift is theatre.')
    req(verdict['ok'] is False,
        f'the verdict is still "ok" with a frozen layer changed: {verdict}')

    # -- the other direction: a trainable layer that never moved usually means
    #    the optimiser never received its gradients, which is what a
    #    positionally resolved freeze produces silently.
    model = fresh()
    before = weight_fingerprint(model)
    for name in trainable_set[1:]:
        nudge(model, name)
    verdict = verify_freeze(before, weight_fingerprint(model), frozen_set)
    same(verdict['trainable_but_unchanged'], [trainable_set[0]],
         f'a trainable layer that never moved was not reported: {verdict}')

    # -- the freeze set must be resolved by NAME, never by position. For a
    #    branched functional model `model.layers` puts value_fc and adv_fc
    #    adjacent, so a positional rule meant to spare the trunk froze both
    #    head hiddens instead.
    from src.dqn.networks import resolve_layers
    same(resolve_layers('dueling', ['value']), ('value_fc', 'value_out'),
         'the dueling `value` group does not resolve to the value stream')
    same(resolve_layers('dueling', ['trunk', 'trunk_fc1']),
         ('trunk_fc1', 'trunk_fc2'),
         'group expansion is not canonical, so two spellings of one ablation '
         'would produce two arms')
    try:
        resolve_layers('dueling', ['layer_2'])
    except ValueError:
        pass
    else:
        raise Failed('resolve_layers accepted a positional-looking name; a '
                     'mis-resolved layer set is the defect that invalidated '
                     'the published freeze schedule.')

    # -- and the same guard, over the recorded dataset rather than a construction.
    manifests = sorted(glob.glob(os.path.join(ctx.runs, '*', '*', 's*',
                                              'manifest.json')))
    if not manifests:
        ctx.note('no recorded runs under --runs; the on-disk check was skipped')
        return
    verdicts = 0
    violations: list[str] = []
    frozen_windows = 0
    for path in manifests:
        with open(path, encoding='utf-8') as fh:
            manifest = json.load(fh)
        for event in manifest.get('freeze_events') or []:
            if event.get('frozen'):
                frozen_windows += 1
                req(int(event.get('frozen_params') or 0) > 0
                    or not (manifest.get('config') or {}).get('freeze_group')
                    or (manifest['config'].get('freeze_group') == 'none'),
                    f'{path}: a freeze event declares frozen=True but froze '
                    f'zero parameters, so the window is nominal only.')
            verification = event.get('verification')
            if verification is None:
                continue
            verdicts += 1
            if not verification.get('ok'):
                violations.append(f'{path}: {verification}')
    req(not violations,
        f'{len(violations)} recorded freeze window(s) failed verification -- a '
        f'layer declared frozen moved during the window:\n    '
        + '\n    '.join(violations[:8]))
    req(verdicts > 0,
        f'{len(manifests)} recorded runs under {ctx.runs} and not one freeze '
        f'verification verdict among them. The freeze window is never being '
        f'left inside the budget, so the verification, the tf.function '
        f'retrace and the optimiser\'s survival across the transition are all '
        f'untested by the dataset (registry.SMOKE_OVERRIDES sizes E0 so the '
        f'boundary IS crossed).')
    ctx.note(f'1-ULP violation caught; {verdicts} recorded verdicts across '
             f'{len(manifests)} runs, {frozen_windows} frozen windows, 0 '
             f'violations')


@case('DESIGN.md §4.1 -- C3 preserves the weight multiset and the Frobenius '
      'norm but NOT the spectrum, which is why C3b exists')
def test_permutation_preserves_norm(ctx: Ctx) -> None:
    """Entry-wise shuffle preserves the multiset; C3b preserves the spectrum."""
    import numpy as np

    from src.dqn.networks import LAYER_GROUPS, build_q_network
    from src.dqn.seeding import Seeds
    from src.dqn.transfer import permute_source, spectrum_matched_source

    arch = 'dueling'
    layers = ('trunk_fc1', 'trunk_fc2', 'value_fc', 'adv_fc')

    def fresh():
        seeds = Seeds(31)
        return build_q_network(8, 4, arch, (128, 128), 64, 'mean',
                              seeds.layer_seeds(LAYER_GROUPS[arch]['all']))

    original = {name: [np.array(w) for w in
                       fresh().get_layer(name).get_weights()]
                for name in layers}

    # -- C3, scope='all': the multiset of weights, and therefore the norm and
    #    the marginal distribution, are preserved exactly.
    shuffled, report = permute_source(fresh(), layers,
                                      np.random.default_rng(0), 'all')
    same(sorted(r['layer'] for r in report), sorted(layers),
         'permute_source did not report every requested layer')
    spectra_changed = 0
    for name in layers:
        before = original[name][0]
        after = np.array(shuffled.get_layer(name).get_weights()[0])
        same(after.shape, before.shape, f'{name}: shape changed under shuffle')
        req(np.array_equal(np.sort(before.reshape(-1)),
                           np.sort(after.reshape(-1))),
            f'{name}: the entry-wise shuffle did not preserve the multiset of '
            f'weights. C3 is interpretable only because it holds the trained '
            f'weights\' scale and marginal distribution exactly while '
            f'destroying structure; if the multiset moves, C3 - C2 mixes a '
            f'scale change into a structure contrast (DESIGN.md §4.1).')
        n_before = float(np.linalg.norm(before.astype(np.float64)))
        n_after = float(np.linalg.norm(after.astype(np.float64)))
        near(n_after, n_before, 1e-5 * max(1.0, n_before),
             f'{name}: the Frobenius norm changed under an entry-wise shuffle')
        req(not np.array_equal(before, after),
            f'{name}: the shuffle left the kernel unchanged, so C3 is C1')
        sv_before = np.linalg.svd(before.astype(np.float64), compute_uv=False)
        sv_after = np.linalg.svd(after.astype(np.float64), compute_uv=False)
        if not np.allclose(sv_before, sv_after, rtol=1e-6, atol=1e-8):
            spectra_changed += 1
    req(spectra_changed == len(layers),
        f'the entry-wise shuffle preserved the singular-value spectrum in '
        f'{len(layers) - spectra_changed} of {len(layers)} layers. DESIGN.md '
        f'§4.1 states plainly that it does NOT, and the whole justification '
        f'for the spectrum-matched control C3b rests on that. If the shuffle '
        f'were spectrum-preserving, C3b would be redundant and the caveat '
        f'attached to the permuted-source contrast would be wrong.')

    # -- C3b, spectrum-matched: singular values preserved exactly, multiset not.
    spectral, spec_report = spectrum_matched_source(fresh(), layers,
                                                    np.random.default_rng(0))
    for record in spec_report:
        req(record['singular_values_preserved'],
            f'{record["layer"]}: spectrum_matched_source reports the singular '
            f'values were not preserved, so C3b does not control what it '
            f'claims to.')
    for name in layers:
        before = original[name][0].astype(np.float64)
        after = np.array(spectral.get_layer(name).get_weights()[0],
                         dtype=np.float64)
        req(np.allclose(np.linalg.svd(before, compute_uv=False),
                        np.linalg.svd(after, compute_uv=False),
                        rtol=1e-5, atol=1e-7),
            f'{name}: C3b did not reproduce the source layer\'s spectrum')
        req(not np.array_equal(np.sort(before.reshape(-1)),
                               np.sort(after.reshape(-1))),
            f'{name}: C3b reproduced the weight multiset as well as the '
            f'spectrum, which would make it the same control as C3 and leave '
            f'the spectral caveat unbounded.')

    # -- scope='units' is the deliberately weaker control, and must differ.
    units, _ = permute_source(fresh(), layers, np.random.default_rng(0),
                              'units')
    differs = 0
    for name in layers:
        after_units = np.array(units.get_layer(name).get_weights()[0])
        after_all = np.array(shuffled.get_layer(name).get_weights()[0])
        if not np.array_equal(after_units, after_all):
            differs += 1
        # A column permutation keeps each unit's incoming weight vector intact.
        cols_before = {tuple(c) for c in original[name][0].T}
        cols_after = {tuple(c) for c in after_units.T}
        same(cols_before, cols_after,
             f'{name}: scope="units" changed a unit\'s incoming weight '
             f'vector, so it is not the column permutation it is documented '
             f'to be')
    same(differs, len(layers),
         'scope="units" and scope="all" produced identical weights, so the '
         'two controls are one control')
    ctx.note(f'{len(layers)} layers: multiset and norm preserved, spectrum '
             f'not; C3b preserves the spectrum, not the multiset')


@case('DESIGN.md §7 E11 -- value-head recalibration cannot improve the initial '
      'policy, by construction')
def test_value_recal_is_policy_invariant(ctx: Ctx) -> None:
    """Centring the value head leaves every argmax alone and shifts Q by a constant."""
    import numpy as np

    from src.dqn.networks import LAYER_GROUPS, build_q_network
    from src.dqn.seeding import Seeds
    from src.dqn.transfer import recalibrate_value_head

    seeds = Seeds(4)
    states = np.asarray(seeds.rng('diag').standard_normal((128, 8)),
                        dtype=np.float32)

    def fresh(aggregation: str):
        return build_q_network(8, 4, 'dueling', (128, 128), 64, aggregation,
                               Seeds(4).layer_seeds(
                                   LAYER_GROUPS['dueling']['all']))

    aggregations = ('mean',) if ctx.quick else ('mean', 'max', 'naive')
    for aggregation in aggregations:
        for mode in ('center', 'center_scale'):
            model = fresh(aggregation)
            q_before = np.asarray(model(states, training=False))
            info = recalibrate_value_head(model, states, mode)
            req(info.get('applied'),
                f'{aggregation}/{mode}: recalibration reported it did not '
                f'apply ({info}), so E11 would compare an arm against itself')
            q_after = np.asarray(model(states, training=False))
            same(q_before.shape, q_after.shape, 'Q shape changed')
            req(np.array_equal(q_before.argmax(axis=1),
                               q_after.argmax(axis=1)),
                f'{aggregation}/{mode}: recalibration changed the greedy '
                f'action on '
                f'{int(np.sum(q_before.argmax(1) != q_after.argmax(1)))} of '
                f'{len(states)} states. V is action-independent, so under '
                f'Q = V + (A - baseline(A)) every argmax must be untouched. '
                f'A write-up could otherwise claim recalibration "improves '
                f'the initial policy", which is false by construction '
                f'(transfer.recalibrate_value_head).')
            req(info.get('policy_invariant') is True,
                f'{aggregation}/{mode}: the returned record does not assert '
                f'policy invariance, so the manifest does not carry the one '
                f'fact that stops the false claim')
            near(float(info['v_mean_after']), 0.0, 1e-4,
                 f'{aggregation}/{mode}: the value head was not centred on '
                 f'the supplied states')

            shift = q_after - q_before
            if mode == 'center':
                req(float(np.std(shift)) < 1e-4,
                    f'{aggregation}/center: Q did not shift by a CONSTANT '
                    f'(shift SD {float(np.std(shift)):.3e}, range '
                    f'{float(shift.min()):.4f}..{float(shift.max()):.4f}). '
                    f'Centring adds a constant to an action-independent '
                    f'stream; anything else means the intervention is not the '
                    f'one E11 describes.')
                near(float(np.mean(shift)), -float(info['v_mean_before']),
                     1e-3,
                     f'{aggregation}/center: the constant shift does not equal '
                     f'minus the pre-recalibration mean V')
            else:
                # center_scale rescales the kernel too, so the shift is NOT a
                # constant -- recorded here so the two modes are not conflated.
                ctx.note(f'{aggregation}/center_scale shift SD '
                         f'{float(np.std(shift)):.3f} (not constant, as '
                         f'documented)')

    # A non-dueling network has no value stream: report, do not pretend.
    mlp = build_q_network(8, 4, 'mlp', (128, 128), 64, 'mean',
                          Seeds(4).layer_seeds(LAYER_GROUPS['mlp']['all']))
    info = recalibrate_value_head(mlp, states, 'center')
    req(info.get('applied') is False and info.get('reason'),
        f'recalibration on an mlp reported {info}; it must decline with a '
        f'reason rather than silently do nothing.')
    ctx.note(f'{len(aggregations)} aggregations x 2 modes: argmax invariant')


@case('DESIGN.md §5.1 -- a random policy scores 0 and the registered threshold '
      'scores 1, for every registry entry')
def test_normalisation(ctx: Ctx) -> None:
    """The normalised score is anchored, and the source-validity gate is sign-safe."""
    from experiments import registry
    from src.dqn import envs

    references = envs.load_references()
    req(references,
        f'no measured reference returns at {envs.REFERENCE_FILE}. Scores are '
        f'normalised against the measured random-policy return, so without '
        f'them every reported number is on an unknown scale -- run '
        f'experiments/measure_references.py.')

    for key, ref in sorted(references.items()):
        random_return = float(ref['random_return'])
        threshold = ref.get('threshold')
        req(threshold is not None,
            f'{key}: no reward threshold, so the normalisation has no unit')
        near(envs.normalised_score(key, random_return), 0.0, 1e-9,
             f'{key}: the measured random-policy return does not score 0')
        near(envs.normalised_score(key, float(threshold)), 1.0, 1e-9,
             f'{key}: the registered threshold does not score 1')
        # Round-trip, so a reported raw return beside a score is the same number.
        for score in (-0.5, 0.0, 0.25, 1.0, 1.5):
            near(envs.normalised_score(key, envs.denormalise_score(key, score)),
                 score, 1e-9, f'{key}: denormalise/normalise does not '
                              f'round-trip at score {score}')
        req(float(threshold) != random_return,
            f'{key}: degenerate normalisation')

    # Every environment the catalogue will actually run must have a reference,
    # and `reference` must RAISE rather than default when one is missing --
    # a silently-zero denominator is the class of scale error that made the
    # published cross-variant comparisons meaningless.
    needed = {registry.SOURCE_ENV, registry.TARGET_ENV, registry.INTERFACE_ENV}
    for _, spec in registry.WIND_LEVELS + registry.GRAVITY_LEVELS:
        needed.add(spec)
    for _name, source_env, target_env in registry.ENV_PAIRS:
        needed.update({source_env, target_env})
    for family in envs.VARIANT_FAMILIES.values():
        for _label, params in family['levels']:
            needed.add(envs.EnvSpec(family['base'], dict(params)).canonical())
    missing = sorted(spec for spec in needed
                     if envs.parse(spec).canonical() not in references)
    req(not missing,
        f'the catalogue references {len(missing)} environment(s) with no '
        f'measured normalisation constants: {missing}. DESIGN.md §5.1 measures '
        f'the random-policy return per environment AND per variant because '
        f'across the LunarLander gravity family it moves from -202 to -463; a '
        f'raw delta would silently mix a scale change into a shift effect.')
    try:
        envs.reference('CartPole-v1:force_mag=42')
    except KeyError:
        pass
    else:
        raise Failed('envs.reference returned a value for an unmeasured '
                     'variant instead of raising. A missing reference silently '
                     'replaced by zero puts that variant\'s scores on a '
                     'different scale from every other.')

    # The source-validity gate, and the specific defect it replaced. Revision 1
    # multiplied the RAW threshold: at Acrobot's -100, "0.6 x threshold" is -60,
    # which is HARDER than solving the task, while the measured random return
    # is -497 (DESIGN.md §4.3).
    gate = 0.6
    acrobot = 'Acrobot-v1'
    ref = references[acrobot]
    random_return = float(ref['random_return'])
    threshold = float(ref['threshold'])
    gate_return = envs.denormalise_score(acrobot, gate)
    req(random_return < gate_return < threshold,
        f'{acrobot}: the normalised validity gate at score {gate} corresponds '
        f'to a return of {gate_return:.1f}, which does not lie strictly '
        f'between the measured random return ({random_return:.1f}) and the '
        f'threshold ({threshold:.1f}). A gate outside that interval is either '
        f'unreachable or free.')
    legacy_gate_return = gate * threshold
    req(envs.normalised_score(acrobot, legacy_gate_return) > 1.0,
        f'{acrobot}: the legacy multiplicative gate 0.6 x threshold = '
        f'{legacy_gate_return:.1f} does not score above 1.0, so this test is '
        f'not reproducing the defect DESIGN.md §4.3 describes and cannot show '
        f'that the normalised gate fixes it.')
    req(legacy_gate_return > threshold,
        f'{acrobot}: 0.6 x threshold is not stricter than the threshold '
        f'itself; the sign-safety argument does not apply here.')
    ctx.note(f'{len(references)} environments anchored at 0 and 1; Acrobot '
             f'gate {gate} = {gate_return:.1f} vs legacy '
             f'{legacy_gate_return:.1f} (stricter than solving)')


# ===========================================================================
# 7. Inference: the pre-registered constants, and the metric-role guards
# ===========================================================================
@case('ANALYSIS_PLAN.md §6 -- the pinned critical values and MDEs are computed, '
      'not asserted')
def test_statlib_reference_values(ctx: Ctx) -> None:
    """statlib reproduces every number ANALYSIS_PLAN.md §2, §5 and §6 pin."""
    import numpy as np

    from experiments import statlib as sl

    # -- the pre-registered constants themselves. They live in statlib as
    #    constants so no caller can pass a different family size or bootstrap
    #    seed and have the difference go unrecorded (ANALYSIS_PLAN.md §7).
    same(sl.ALPHA, 0.05, 'ALPHA is not the pre-registered 0.05')
    same(sl.CONFIRMATORY_FAMILY_SIZE, 8,
         'the confirmatory family is not 8 (4 cells x 2 co-primary endpoints)')
    near(sl.HOLM_STRICTEST_ALPHA, 0.05 / 8, 1e-15,
         'the strictest Holm step is not 0.00625')
    same(sl.EQUIVALENCE_MARGIN, 0.05,
         'the equivalence margin is not the pre-registered +/-0.05 normalised '
         'score units (ANALYSIS_PLAN.md §4), which may not be re-derived after '
         'seeing a CI')
    same(sl.BOOTSTRAP_SEED, 20260824, 'the bootstrap seed is not the fixed one')
    same(sl.N_BOOT, 10_000, 'the bootstrap size is not the pre-registered one')
    same(tuple(sl.THRESHOLD_LEVELS), (0.25, 0.50, 1.00),
         'the censored-metric thresholds are not the pre-declared {0.25, 0.5, '
         '1.0} (ANALYSIS_PLAN.md §5)')
    same(sl.MIN_N_FOR_INFERENCE, 3,
         'the n<3 floor is not 3 (ANALYSIS_PLAN.md §9)')
    same(sl.PIPELINE_VALIDATION_LABEL, 'PIPELINE VALIDATION - NOT A RESULT',
         'the n<3 label does not match the plan')

    # -- §2.2 and §6.1: the attainable p floors. At n=10 the paired floor is
    #    what makes "a cell is confirmed iff all ten seeds move the same way"
    #    true rather than rhetorical.
    near(sl.signflip_min_attainable_p(10), 2 / 1024, 1e-15,
         'the exact sign-flip floor at n=10 is not 2/2^10 = 0.00195')
    req(sl.signflip_min_attainable_p(10) < sl.HOLM_STRICTEST_ALPHA,
         'the paired floor exceeds the strictest Holm step, which would mean '
         'NO confirmatory result is attainable at n=10 -- a different '
         'statement from the plan\'s')
    near(sl.mwu_min_attainable_p(10, 10), 1.082508822446903e-05, 1e-12,
         'the Mann-Whitney floor at 10 vs 10 is not 1.08e-5')

    # -- §6.1: the exact two-sided rejection regions.
    for alpha, expected in ((0.05, (23, 77)), (0.0125, (17, 83)),
                            (sl.HOLM_STRICTEST_ALPHA, (14, 86))):
        same(sl.mwu_critical_values(10, 10, alpha), expected,
             f'the exact Mann-Whitney rejection region at alpha={alpha:g} is '
             f'not the pinned {expected}')

    # -- §7: the Holm step-down thresholds over the one family.
    thresholds = sl.holm_thresholds(8, 0.05)
    near(float(thresholds[0]), 0.00625, 1e-12,
         'the first Holm step is not 0.00625')
    near(float(thresholds[-1]), 0.05, 1e-12, 'the last Holm step is not 0.05')
    req(np.all(np.diff(thresholds) > 0), 'the Holm thresholds are not ascending')

    # -- §5: the exact interval that carries the censored metric. At 0 of 10 the
    #    upper bound is the informative statement where a p-value would be none.
    zero_of_ten = sl.clopper_pearson(0, 10)
    same(zero_of_ten.lo, 0.0, 'Clopper-Pearson at 0/10 has a non-zero lower '
                              'bound')
    near(zero_of_ten.hi, 0.3085, 5e-4,
         f'the Clopper-Pearson upper bound at 0 of 10 is {zero_of_ten.hi!r}, '
         f'not the ~0.31 ANALYSIS_PLAN.md §5 quotes')

    # -- §8: a non-finite entry must be refused, never dropped. Silent seed
    #    dropping is the published defect (one seed removed with no rule).
    try:
        sl.sign_flip_test([0.1, float('nan'), 0.3, 0.2])
    except ValueError:
        pass
    else:
        raise Failed('statlib accepted a non-finite per-seed value. A missing '
                     'value means a missing run, and dropping it silently is '
                     'the defect ANALYSIS_PLAN.md §8 forbids by name.')

    # -- §9: below the floor, no test and no interval.
    refused = sl.sign_flip_test([0.1, 0.2])
    req(refused.get('p') is None,
        f'a test was emitted at n=2: {refused}. ANALYSIS_PLAN.md §9 requires '
        f'no test and no interval below n=3.')
    refused_ci = sl.bootstrap_ci([0.1, 0.2])
    req(refused_ci.refused, f'an interval was emitted at n=2: {refused_ci!r}')

    # -- the ledger the plan requires on every invocation.
    ledger = sl.multiplicity_ledger(n_estimation_only=5, n_screen_members=3)
    families = {row['family']: row for row in ledger}
    same(set(families), {'confirmatory', 'screens', 'estimation-only'},
         f'the multiplicity ledger does not name the three pre-registered '
         f'families: {sorted(families)}')
    same(families['confirmatory']['members'], 8,
         'the ledger does not report a confirmatory family of 8')
    req(families['estimation-only']['carries_p_values'] is False,
        'the ledger claims the estimation-only family carries p-values')

    if ctx.quick:
        ctx.note('critical values and constants checked; MDE simulations '
                 'skipped under --quick')
        return

    # -- §6.2: the minimum detectable effects. These are the numbers that
    #    justify the single-family decision, so a drift here means the
    #    pre-registration no longer describes the code.
    expected_mdes = [
        ('paired sign-flip, alpha=0.05', sl.mde_signflip, 0.05, 1.00),
        ('paired sign-flip, Holm over 8', sl.mde_signflip,
         sl.HOLM_STRICTEST_ALPHA, 1.54),
        ('unpaired Mann-Whitney, alpha=0.05', sl.mde_mann_whitney, 0.05, 1.39),
        ('unpaired Mann-Whitney, Holm over 8', sl.mde_mann_whitney,
         sl.HOLM_STRICTEST_ALPHA, 1.87),
    ]
    measured = {}
    for label, fn, alpha, pinned in expected_mdes:
        mde = fn(10, alpha=alpha)
        measured[label] = float(mde)
        near(float(mde), pinned, 0.03,
             f'{label}: the MDE at n=10 is {float(mde):.3f} sigma, not the '
             f'{pinned:.2f} sigma pinned in ANALYSIS_PLAN.md §6.2. The power '
             f'table is a pre-registration and is not re-tuned after seeing '
             f'results (§6.4), so a drift means the plan no longer describes '
             f'the estimator.')
        near(float(mde.power_achieved), 0.8, 0.02,
             f'{label}: the bisection reached power '
             f'{float(mde.power_achieved):.3f}, not 0.80')
    paired = measured['paired sign-flip, alpha=0.05']
    unpaired = measured['unpaired Mann-Whitney, alpha=0.05']
    req(paired < unpaired,
        f'the paired MDE ({paired:.3f}) is not smaller than the unpaired one '
        f'({unpaired:.3f}). ANALYSIS_PLAN.md §6.2 makes the paired test '
        f'primary precisely because pairing is worth roughly a 40% reduction '
        f'in the detectable effect at this n.')
    ctx.note('MDEs at n=10: ' + ', '.join(f'{k.split(",")[0]} '
                                          f'{v:.2f}' for k, v in
                                          measured.items()))

    if ctx.full:
        rc = sl.self_test(verbose=False)
        req(not rc.get('failed'),
            f'statlib.self_test reported failures: {rc.get("failures")}')
        ctx.note(f'statlib.self_test: {rc.get("passed")} checks passed')


@case('ANALYSIS_PLAN.md §1, §8 -- a confirmatory test on a descriptive metric '
      'is refused')
def test_stats_refuses_descriptive_metric(ctx: Ctx) -> None:
    """stats.require_confirmatory admits only the two co-primary endpoints."""
    from experiments import stats

    same(tuple(stats.CONFIRMATORY_ENDPOINTS), ('final_score', 'auc_score'),
         f'the co-primary endpoints are {stats.CONFIRMATORY_ENDPOINTS}, not '
         f'(final_score, auc_score). The confirmatory family is fixed by '
         f'ANALYSIS_PLAN.md §2 and read from the plan, not taken as an '
         f'argument.')
    for metric in stats.CONFIRMATORY_ENDPOINTS:
        stats.require_confirmatory(metric)           # must not raise
        same(stats.metric_role(metric), stats.CO_PRIMARY,
             f'{metric} is not declared co-primary')

    # One from each non-testable role, plus the two the published paper
    # actually tested and then declared untestable two paragraphs later.
    forbidden = ['train_return', 'final_return', 'td_loss', 'td_loss_final100',
                 'updates', 'env_steps', 'wall_time_s', 'clip_fraction',
                 'jumpstart_score', 'probe_jumpstart_score', 'within_run_sd',
                 'episode_length_final100', 'convergence_slope',
                 'steps_to_threshold_p50', 'q_mean', 'td_error_abs',
                 'v_abs_mean', 'a_spread', 'grad_norm_trunk', 'cka_drift',
                 'dead_unit_frac', 'transferred_param_fraction',
                 'source_final_score', 'prefix_score_500',
                 'a_metric_nobody_declared']
    for metric in forbidden:
        try:
            stats.require_confirmatory(metric)
        except stats.MetricRoleError:
            continue
        raise Failed(
            f'stats.require_confirmatory({metric!r}) did not refuse. Its role '
            f'is {stats.metric_role(metric)!r}, and ANALYSIS_PLAN.md §1 '
            f'permits a confirmatory test only on the co-primary endpoints. '
            f'This is the mechanical fix for the published §V.A/§V.B '
            f'contradiction -- a t-test on a metric §V.B called '
            f'descriptive-only and non-normal.')

    # An undeclared metric must not default to testable: adding a metric has to
    # force a decision about its role.
    same(stats.metric_role('a_metric_nobody_declared'), 'unclassified',
         'an undeclared metric is not reported as unclassified')

    # And every role that appears in the table is one of the declared five.
    roles = set(stats.METRIC_ROLES.values())
    allowed = {stats.CO_PRIMARY, stats.SECONDARY, stats.DESCRIPTIVE,
               stats.MECHANISM, stats.BOOKKEEPING}
    req(roles <= allowed,
        f'METRIC_ROLES uses undeclared roles {sorted(roles - allowed)}')
    n_co_primary = sum(1 for r in stats.METRIC_ROLES.values()
                       if r == stats.CO_PRIMARY)
    same(n_co_primary, 2,
         f'{n_co_primary} metrics are declared co-primary; exactly two are '
         f'pre-registered, and a third would enlarge the confirmatory family '
         f'without a plan amendment')
    ctx.note(f'{len(forbidden)} non-co-primary metrics all refused; '
             f'{len(stats.METRIC_ROLES)} roles declared')


# ---------------------------------------------------------------------------
# The two cases below read the recorded dataset. They select a subset of it --
# one freeze level, one experiment -- because `stats.py` correctly REFUSES to
# aggregate an arm whose invariants differ, and the demo tree contains E1
# transfer runs at two freeze windows. The selection is a filter over real rows,
# never a synthesised number.
# ---------------------------------------------------------------------------
def _per_seed_path(ctx: Ctx) -> str:
    path = os.path.join(ctx.runs, 'per_seed.csv')
    if not os.path.exists(path):
        ctx.skip(f'{path} not found; run experiments/aggregate.py, or pass '
                 f'--runs at a tree that has one')
    return path


def _filtered_per_seed(ctx: Ctx, seeds: Sequence[int] | None = None) -> str:
    """Write a subset of the recorded per_seed.csv to a temporary file."""
    import pandas as pd

    frame = pd.read_csv(_per_seed_path(ctx))
    # One freeze window, so the primary transfer arm is invariant-clean; keeping
    # both is what stats.py refuses, and correctly so (DESIGN.md §8.4).
    windows = sorted({int(v) for v in
                      frame.loc[frame['condition'] == 'transfer',
                                'freeze_updates'].dropna().unique()})
    if len(windows) > 1:
        frame = frame[(frame['condition'] == 'scratch')
                      | (frame['freeze_updates'] == windows[0])]
    if seeds is not None:
        frame = frame[frame['seed'].isin(list(seeds))]
    if not len(frame):
        ctx.skip('the selection is empty')
    out = os.path.join(ctx.tmpdir('perseed_'), 'per_seed.csv')
    frame.to_csv(out, index=False)
    return out


def _run_stats(ctx: Ctx, per_seed: str, extra: Sequence[str] = ()
               ) -> tuple[int, str, dict]:
    """Invoke stats.py in-process and return (exit code, stdout, report)."""
    from experiments import stats

    json_out = os.path.join(os.path.dirname(per_seed), 'report.json')
    argv = ['--per-seed', per_seed, '--json', json_out,
            '--n-boot', '300' if not ctx.full else str(stats.N_BOOT),
            *extra]
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = stats.main(argv)
    text = buffer.getvalue()
    report: dict = {}
    if os.path.exists(json_out):
        with open(json_out, encoding='utf-8') as fh:
            report = json.load(fh)
    return code, text, report


#: Keys whose *name* matches the p-value pattern but which are not p-values.
#: Each one is verified below rather than merely excused, so the exemption
#: cannot become a hiding place.
_NOT_A_P_VALUE = {
    # P(threshold reached within budget): a proportion with a Clopper-Pearson
    # interval beside it, which is the primary censored summary of §5.
    'p_reached',
    # The smallest two-sided p the test could return at that n, reported so a
    # floored p is distinguishable from strong evidence.
    'min_attainable_p',
    # A count of screen members, not a q-value.
    'screen_q_count',
}

_P_LIKE = re.compile(r'^(p|q)$|^(p|q)_|_(p|q)$|p_?val|pvalue', re.IGNORECASE)


def _walk(node: Any, path: str = '') -> Iterable[tuple[str, str, Any, Any]]:
    """Yield (path, key, value, parent) over every dict entry in a JSON tree."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield f'{path}/{key}', str(key), value, node
            yield from _walk(value, f'{path}/{key}')
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _walk(value, f'{path}[{index}]')


@case('ANALYSIS_PLAN.md §2, §7 -- exactly one confirmatory family, and no '
      'p-value outside it')
def test_stats_no_pvalue_outside_family(ctx: Ctx) -> None:
    """Every p-value stats.py emits lives inside the family of 8."""
    from experiments import stats

    per_seed = _filtered_per_seed(ctx)
    code, text, report = _run_stats(ctx, per_seed,
                                    ['--experiments', 'E1',
                                     '--source-policy', 'pooled'])
    same(code, 0, f'stats.py exited {code}; the tail of its output was:\n'
                  f'{text[-1500:]}')
    req(report, 'stats.py wrote no JSON report, so the check has nothing to '
                'read')

    members = report['s5_confirmatory']['members']
    same(len(members), stats.CONFIRMATORY_FAMILY_SIZE,
         f'the confirmatory section emitted {len(members)} members, not the '
         f'pre-registered {stats.CONFIRMATORY_FAMILY_SIZE}. Family membership '
         f'is fixed by the plan before launch; a family of a different size '
         f'means a result could be rescued by relocating it.')
    same(report['s5_confirmatory']['family_size'],
         stats.CONFIRMATORY_FAMILY_SIZE,
         'the reported family size is not 8')
    tested = [m for m in members if m.get('p_signflip') is not None]
    req(tested,
        'no confirmatory member carried a p-value, so this test is vacuous: '
        'it cannot show that p-values are confined to the family if none were '
        'emitted. Every member was suppressed for: '
        + '; '.join(sorted({str(m.get('suppressed'))[:70] for m in members})))
    for member in tested:
        req(member.get('p_holm') is not None,
            f'{member["metric"]}/{member["cell"]} has a raw p but no '
            f'Holm-adjusted p; the correction is not optional')
        for name in ('p_wilcoxon', 'p_mannwhitney'):
            req(name in member,
                f'{member["metric"]}/{member["cell"]} omits {name}. '
                f'ANALYSIS_PLAN.md §2 pre-specifies that Wilcoxon and '
                f'Mann-Whitney are reported alongside for the same contrast, '
                f'so the choice of test cannot be made after seeing which '
                f'gives the smaller p.')
        req('rho_pearson' in member,
            f'{member["metric"]}/{member["cell"]} omits the within-seed '
            f'correlation, which §2.1 commits to reporting whatever its value')

    offenders: list[str] = []
    exempted = 0
    for path, key, value, parent in _walk(report):
        if not _P_LIKE.match(key):
            continue
        if key in _NOT_A_P_VALUE:
            exempted += 1
            if key == 'p_reached' and value is not None:
                req(0.0 <= float(value) <= 1.0,
                    f'{path}: p_reached={value!r} is outside [0, 1], so the '
                    f'name is not describing a proportion')
                req('cp_lo' in parent and 'cp_hi' in parent,
                    f'{path}: p_reached has no Clopper-Pearson interval beside '
                    f'it, so it cannot be the censored summary of '
                    f'ANALYSIS_PLAN.md §5 and the exemption is unjustified')
            continue
        section = path.lstrip('/').split('/')[0]
        # The one declared exception: §7 permits Benjamini-Hochberg q for the
        # screens, orientation only, never as an assertion.
        if section == 's5_confirmatory' or '/screens' in path:
            continue
        offenders.append(f'{path} = {value!r}')
    req(not offenders,
        f'{len(offenders)} p-value(s) were emitted outside the confirmatory '
        f'family and outside the screens\' orientation-only q-values. '
        f'ANALYSIS_PLAN.md §7 permits p-values in exactly one family; '
        f'everything else is estimation-only, and a p-value elsewhere is how '
        f'"positive transfer" gets claimed from p=0.421. Offenders: '
        + '; '.join(offenders[:10]))

    # The ledger must be printed, and must say the estimation family carries no
    # p-values -- the count is a recorded fact rather than a claim (§7).
    ledger = report['s11_ledger']['families']
    labels = [str(row['family']).lower() for row in ledger]
    req(any('confirmatory' in l for l in labels)
        and any('screen' in l for l in labels)
        and any('everything else' in l or 'estimation' in l for l in labels),
        f'the multiplicity ledger does not name the three families: {labels}')
    everything_else = next(row for row in ledger
                           if 'everything else' in str(row['family']).lower()
                           or 'estimation' in str(row['family']).lower())
    req('no p-value' in str(everything_else['adjusted_alpha']).lower(),
        f'the ledger does not state that the estimation-only family carries no '
        f'p-values: {everything_else}')
    req('MULTIPLICITY' in text.upper() or 'ledger' in text.lower(),
        'the multiplicity ledger was not printed; ANALYSIS_PLAN.md §7 requires '
        'it on every invocation')
    ctx.note(f'{len(tested)} of {len(members)} family members tested; '
             f'{len(offenders)} p-values outside the family; {exempted} '
             f'name-collisions verified as non-p-values')


@case('ANALYSIS_PLAN.md §9, STANDING_INSTRUCTIONS S8 -- at n<3 no test is '
      'emitted and every page is stamped')
def test_n1_is_labelled(ctx: Ctx) -> None:
    """A single-seed selection produces no test and is stamped PIPELINE VALIDATION."""
    from experiments import stats

    import pandas as pd

    frame = pd.read_csv(_per_seed_path(ctx))
    seeds = sorted({int(s) for s in frame['seed'].unique()})
    target_side = [s for s in seeds if s < 200]
    if not target_side:
        ctx.skip('the recorded tree has no target-side seeds')
    # One target-side seed, plus the source-donor blocks the transfer arms need.
    keep = [target_side[0]] + [s for s in seeds if s >= 200]
    per_seed = _filtered_per_seed(ctx, seeds=keep)

    code, text, report = _run_stats(ctx, per_seed,
                                    ['--experiments', 'E1',
                                     '--source-policy', 'pooled'])
    same(code, 0,
         f'stats.py exited {code} on a single-seed selection -- the invocation '
         f'STANDING_INSTRUCTIONS S8 makes the CURRENT mode of every '
         f'experiment. A crash on that path means the n<3 guard of '
         f'ANALYSIS_PLAN.md §9 is unreachable, so single-seed output is '
         f'emitted with no stamp at all. Tail of the output:\n{text[-2500:]}')

    members = report['s5_confirmatory']['members']
    n_values = [m['n'] for m in members if m.get('n') is not None]
    req(n_values and min(n_values) < stats.MIN_N_FOR_INFERENCE,
        f'the selection did not produce an arm below n={stats.MIN_N_FOR_INFERENCE} '
        f'(observed n: {sorted(set(n_values))}), so the guard is not being '
        f'exercised')
    emitted = [f'{m["metric"]}/{m["cell"]} p={m["p_signflip"]}'
               for m in members if m.get('p_signflip') is not None]
    req(not emitted,
        f'a test was emitted below n={stats.MIN_N_FOR_INFERENCE}: {emitted}. '
        f'A single-seed number may not be quoted, compared, or used to choose '
        f'between hypotheses; a single seed can show that a run executes, it '
        f'cannot show that an arm differs.')
    for member in members:
        if member.get('n', 0) < stats.MIN_N_FOR_INFERENCE:
            req(member.get('suppressed'),
                f'{member["metric"]}/{member["cell"]} at n={member.get("n")} '
                f'carries no suppression reason, so a reader cannot tell the '
                f'blank from a null result')
            req(member.get('ci_lo') is None and member.get('ci_hi') is None,
                f'{member["metric"]}/{member["cell"]} emitted an interval at '
                f'n={member.get("n")}; §9 withholds the interval as well as '
                f'the test')

    req(report['s12_deviations'].get('validation_stamp'),
        'the report does not record the PIPELINE VALIDATION stamp for a '
        'selection containing an arm below the inference floor')
    req(stats.VALIDATION_STAMP in text,
        f'the printed output does not carry the stamp '
        f'{stats.VALIDATION_STAMP!r}. ANALYSIS_PLAN.md §9 requires it on every '
        f'page, and it is the only thing standing between a pipeline-'
        f'validation number and a quoted result.')
    ctx.note(f'seeds kept {keep}; min n = {min(n_values)}; 0 tests emitted; '
             f'stamp present')


# ===========================================================================
# 8. CLI
# ===========================================================================
def _select(names: Sequence[str] | None, quick: bool, full: bool
            ) -> list[Case]:
    if names:
        wanted, unknown = [], []
        known = {c.name: c for c in _CASES}
        for raw in names:
            for token in str(raw).replace(',', ' ').split():
                if token in known:
                    wanted.append(known[token])
                else:
                    matches = [c for c in _CASES if token in c.name]
                    if len(matches) == 1:
                        wanted.append(matches[0])
                    else:
                        unknown.append(token)
        if unknown:
            raise SystemExit(f'validate.py: unknown test(s) {unknown}. '
                             f'Use --list.')
        return wanted
    if quick:
        return [c for c in _CASES if not c.slow]
    return list(_CASES)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='A SKIP is not a PASS: a guard that could not be checked is not '
               'a guard that held.')
    parser.add_argument('--quick', action='store_true',
                        help='reduce the environment sweeps and skip the MDE '
                             'simulations; the guards are all still exercised')
    parser.add_argument('--full', action='store_true',
                        help='add the expensive checks: run-level epsilon '
                             'traces, statlib.self_test, and stats.py at the '
                             'pre-registered bootstrap size')
    parser.add_argument('--test', action='append', default=None, metavar='NAME',
                        help='run only these cases (repeatable, '
                             'comma-separated, unique substring accepted)')
    parser.add_argument('--list', action='store_true',
                        help='list the cases and the guard each one protects')
    parser.add_argument('--runs', default=DEFAULT_RUNS,
                        help=f'run tree the on-disk checks read '
                             f'(default {DEFAULT_RUNS})')
    parser.add_argument('--keep-temp', action='store_true',
                        help='leave the temporary run trees on disk for '
                             'inspection')
    parser.add_argument('--verbose', action='store_true',
                        help='print a traceback for every failure')
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list:
        width = max(len(c.name) for c in _CASES)
        print(f'{len(_CASES)} cases; * marks a case skipped by --quick\n')
        for c in _CASES:
            print(f'  {"*" if c.slow else " "} {c.name:<{width}}  {c.guard}')
        return 0

    cases = _select(args.test, args.quick, args.full)
    print(f'validate.py -- {len(cases)} of {len(_CASES)} guard cases'
          f'{" (--quick)" if args.quick else ""}'
          f'{" (--full)" if args.full else ""}')
    print(f'  run tree for the on-disk checks: {args.runs}')
    print()

    width = max(len(c.name) for c in cases)
    failures: list[tuple[str, str]] = []
    skipped: list[tuple[str, str]] = []
    started = time.time()

    for c in cases:
        ctx = Ctx(quick=args.quick, full=args.full, runs=args.runs)
        began = time.time()
        status, detail = PASS, ''
        try:
            c.fn(ctx)
        except Skipped as exc:
            status, detail = SKIP, str(exc)
            skipped.append((c.name, str(exc)))
        except Failed as exc:
            status, detail = FAIL, str(exc)
            failures.append((c.name, str(exc)))
        except Exception as exc:                      # noqa: BLE001
            status = FAIL
            detail = (f'{type(exc).__name__}: {exc}\n'
                      f'    (an unexpected exception is a failure: the guard '
                      f'was not shown to hold)')
            if args.verbose:
                detail += '\n' + traceback.format_exc()
            failures.append((c.name, detail))
        finally:
            ctx.cleanup(keep=args.keep_temp)
        elapsed = time.time() - began
        notes = '; '.join(ctx.notes)
        print(f'{status} {c.name:<{width}}  {elapsed:6.1f}s'
              + (f'  {notes}' if notes and status == PASS else ''))
        if status != PASS:
            for line in detail.splitlines():
                print(f'       {line}')

    total = time.time() - started
    print()
    print(f'== {len(cases) - len(failures) - len(skipped)} passed, '
          f'{len(failures)} failed, {len(skipped)} skipped in {total:.1f}s ==')
    if skipped:
        print('\n  SKIPPED (a guard that could not be checked is not a guard '
              'that held):')
        for name, reason in skipped:
            print(f'    {name}: {reason}')
    if failures:
        print('\n  FAILED:')
        for name, _ in failures:
            print(f'    {name}')
        print('\n  A failure here means a guardrail DESIGN.md §9 claims is '
              '"enforced in code" is not.')
        print('  Nothing confirmatory may launch until it is, per '
              'ANALYSIS_PLAN.md §10.1.')
        return 1
    if skipped:
        # Skips do not fail the suite, but they are not silent either.
        print('\n  All executed guards held. The skipped cases were not '
              'checked.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
