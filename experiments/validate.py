"""Self-tests for the guardrails `DESIGN.md` §9 claims are "enforced in code".

Why this file exists
--------------------
Revision 1 of the design carried a table captioned "Anti-fallacy guardrails,
**enforced in code**" and there was no such code (`DESIGN.md` §11, defect 12).
An adversarial review found the table entirely aspirational. This module is what
makes the caption true.

What it claims, exactly
-----------------------
The acceptance criterion for a case here is stricter than "the test passes": a
case must **fail when its guard is removed**. A case that would still pass with
its guard deleted is decoration, and the docstring of each case below names the
specific defect it would catch.

The claim about §9 is narrower than the one this docstring used to make, and it
is machine-checked rather than asserted. It used to read "every row of §9 ...
has a test here that fails when the guard is removed", and that was false for
about ten of the sixteen rows: the file imported `stats.py` and `statlib.py` and
nothing else, so every row whose guard lives in `report.py` (affirming a null,
the "A avoids it, B does not" shape, mechanism-from-prose, generated direction
words, the inherited scope clause) and every row whose guard lives in the
recorded data (TUNE leakage, seed dropping, the source-validity gate, the
transferred-fraction gate, provenance) was uncovered while the docstring said
otherwise. That is defect 12 recurring one level up, inside the file written to
close it.

What replaces it: `_GUARDRAIL_COVERAGE` maps every row of the §9 table onto the
case or cases that cover it **and** onto a plain-words statement of what this
suite still does not check for that row.
`test_guardrail_coverage_is_declared` parses the table out of `DESIGN.md` and
fails if a row is added, renamed or removed without a decision being recorded,
or if the map names a case that does not exist. So the coverage claim is now a
thing that can be wrong loudly rather than a sentence that can be wrong quietly.

The suite is green only when every registered guard ran and held. A SKIP is not
a PASS, and the cases that read a recorded tree SKIP with a reason when the tree
cannot answer them, rather than reporting PASS on a check they did not perform.
A `--runs` path that does not exist is refused outright, because a typo used to
produce a greener suite (18 passed, 2 skipped, exit 0) than the real P0 dataset
did.

A skip is honest and it is still not coverage. Two cases -- the p-value
confinement family check and the TUNE / partial-arm / doubled-arm / empty-endpoint
refusals -- are the ones standing between a malformed sample and a confirmatory
number, and both SKIPPED on the recorded single-seed tree because the n<3 floor
suppresses every member: `26 passed, 0 failed, 2 skipped` was reporting that the
two cases covering the analysis this study exists to perform had never executed,
on any tree. They now run against `_analysis_ctx`, which is the recorded tree
wherever it carries `MIN_N_FOR_INFERENCE` target-side seeds and a deterministic
synthetic fixture (`_fixture_frame`) where it does not. A fixture is evidence
about the CODE, which is what this file tests, and never evidence about the
experiment: its rows carry `SYNTHETIC-FIXTURE-NOT-A-RESULT` in `run_dir`, the
case line says which tree it ran against, and no number derived from one may be
quoted.

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
    be stamped `PIPELINE VALIDATION - NOT A RESULT`. Its selection keeps one
    target-side seed plus the *donor* blocks named in `registry.SEED_BLOCKS`.
    It used to keep every seed >= 200, which is the TUNE block: the case that
    exercises the n<3 guard would have pulled the selection block into the
    estimate on any tree that had one.

``test_recorded_dataset_integrity``
    `DESIGN.md` §8.2's claim about the *recorded* dataset, checked against the
    tree behind `--runs` rather than against runs this file makes. Episode
    contiguity is recomputed and cross-examined against the verdict each run
    stored about itself, a finished run without a manifest is a deleted
    manifest, and every freeze window that closed must carry a verification
    verdict -- counting verdicts and requiring one was satisfied by a single
    verdict while 23 of 24 had been stripped.

``test_stats_excludes_tune_and_partial_arms``
    Three ways a reported estimate goes quietly wrong: a TUNE-block row in the
    table (`DESIGN.md` §3.4), an arm missing a seed (§8.4), and an arm carrying
    two rows for one seed. `stats.py` refuses all three; nothing asserted that
    it does, so dropping a seed from one arm of `runs_demo` left every case
    passing.

``test_source_validity_gate_applied``
    `DESIGN.md` §4.3 makes "valid sources only" the PRIMARY estimand and
    pooling over source competence the secondary. Both stats-reading cases used
    to invoke `stats.py` under `--source-policy pooled` alone, so the gate, the
    exclusion and the labelling of the secondary were never exercised. In P0
    `src-dueling-vanilla` scored 0.599 against a 0.600 gate, so this is the
    path that decides an arm.

``test_stats_emits_controls_and_censoring``
    Two §9 rows: C2/C3 are Tier 1 and the contrasts are emitted with every
    delta, so every cell must ACCOUNT for every control rather than omit one;
    and censored data is neither imputed nor dropped, so P(reached) is reported
    with an exact Clopper-Pearson interval, re-derived here from the counts.

``test_stats_intensity_gate``
    `DESIGN.md` §3.1: a cross-architecture contrast at mismatched transferred
    fraction is confounded with treatment intensity and is refused, not
    annotated. The tolerance was being printed; a value outside it had never
    been put through the gate.

``test_report_wording_guards_fire``
    The five §9 rows whose guard is a wording rule in `report.py`: a null is
    not affirmed, two significance verdicts are not a comparison, a mechanism
    claim must cite an instrumented signal, a directional word is generated
    from the numbers, and the §2.1 scope clause is inherited by the sentence.
    Each guard is made to fire by name, and the same sentence shapes are then
    accepted with the evidence that licenses them, so a guard that refuses
    everything fails here too.

``test_provenance_is_content_addressed``
    §9's stale-artifact row rests on provenance hashes. A hash that did not
    move with the content would satisfy the caption and detect nothing, so the
    property is tested: identical bytes hash alike, one changed byte does not,
    an absent file yields no hash, and every recorded run carries the
    analysis-plan hash `audit.py` compares against.

``test_guardrail_coverage_is_declared``
    The meta-case. It parses the §9 table out of `DESIGN.md` and requires
    `_GUARDRAIL_COVERAGE` to name, for every row, the case that covers it and
    the residual this suite does not check. A row added to the design, a row
    renamed, or a case renamed here fails the suite until somebody records what
    covers it.

Usage
-----
    python experiments/validate.py                 # the default suite
    python experiments/validate.py --quick         # skip the cases marked *
    python experiments/validate.py --full          # add the expensive checks
    python experiments/validate.py --list
    python experiments/validate.py --test test_resume_equivalence
    python experiments/validate.py --runs runs     # check a recorded tree

Exit code is non-zero if any case fails. A case that cannot run -- an absent
dependency, a case excluded by `--quick`, a tree with no `per_seed.csv` --
reports SKIP with the reason and does not mask a failure, because a skipped
guard is not a satisfied one. Too few seeds is no longer one of those reasons:
the confirmatory cases build a synthetic fixture instead (see `_analysis_ctx`).
The final line says "Every registered guard ran and held" only when nothing was
skipped. A `--runs` path that is not a directory is refused before any case
runs.
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
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Sequence

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
        # The reserve-seed rule of `DESIGN.md` 4.3 gives a transfer run a source
        # drawn from a different seed than its own, and `source_seed` enters the
        # digest only when the two differ. 400 is the first RESERVE seed, i.e.
        # exactly the replacement case: without this variant a run re-pointed at
        # a replacement source shares a directory with the run that used the
        # rejected one, and the second silently resumes the first.
        ('source_seed=400', {'source_seed': 400}),
        # `epsilon_start` was the one IDENTITY field with no one-factor entry
        # here. Moving it out of TRAJECTORY_FIELDS left this case passing while
        # two configs differing only in it shared a run directory AND a
        # trajectory digest, so resume was not refused either.
        ('epsilon_start=0.5', {'epsilon_start': 0.5}),
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

    # A count is not coverage. Without this the list could omit a field and the
    # case would still pass: `epsilon_start` was omitted, and moving it out of
    # TRAJECTORY_FIELDS left both this case and
    # test_field_classification_complete green while two configurations
    # differing only in it returned the same run_dir() AND the same
    # trajectory_digest(). The coverage is asserted against the schema so a
    # field added to IDENTITY_FIELDS fails here until a variant exercises it.
    covered = {key for _label, override in variants for key in override}
    uncovered = sorted(set(IDENTITY_FIELDS) - covered)
    req(not uncovered,
        f'{uncovered} are identity fields with no one-factor variant in '
        f'_identity_variants(), so nothing here would notice if the digest '
        f'stopped covering them. Two runs differing only in such a field share '
        f'a directory and the second silently resumes the first -- defect 0 of '
        f'DESIGN.md 11, reproduced through the test meant to prevent it.')
    stray = sorted(covered - set(IDENTITY_FIELDS))
    req(not stray,
        f'{stray} are exercised as one-factor identity variants but are not in '
        f'IDENTITY_FIELDS; the list has drifted from the schema.')

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
@case('DESIGN.md §3.2, §2.4 RQ6 -- epsilon is a closed form in EPISODES, '
      'coupled to neither the evaluation cadence nor the budget')
def test_epsilon_closed_form(ctx: Ctx) -> None:
    """Monotone, floors at epsilon_anneal_episodes, blind to budget and cadence.

    The index is the **episode**, not the environment step. Design revision 5
    made it so, and recorded step-indexing as the defect that forced the
    revision: a step-indexed horizon is endogenous to policy quality, so a poor
    policy ends episodes quickly, accumulates few steps and keeps a high
    epsilon, which keeps it poor. Under that schedule epsilon fell only to
    0.684 and all four CartPole sources failed the validity gate. The wording
    here says episodes throughout, because an evidence line that describes the
    schedule in steps is describing the defect the design removed
    (`Config.epsilon_at(self, episode)`).
    """
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
            f'epsilon(episode {grid[i]})={trace[i]!r} > epsilon(episode '
            f'{grid[i - 1]})={trace[i - 1]!r}. A non-monotone exploration '
            f'schedule makes the elapsed episode the wrong index for it.')
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
             f'epsilon at episode {beyond} is not the floor')

    # The index really is the episode. A signature that renamed it, or a body
    # that reached for a step counter, would put the horizon back under the
    # control of policy quality (DESIGN.md revision 5, §3.2).
    same(tuple(inspect.signature(Config.epsilon_at).parameters)[1:],
         ('episode',),
         'Config.epsilon_at is no longer indexed by episode. A step-indexed '
         'horizon is endogenous to policy quality: that is the defect design '
         'revision 5 removed, under which epsilon fell only to 0.684 and all '
         'four CartPole sources failed the validity gate.')

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
    ctx.note(f'{len(grid)} grid points; floor exact at {anneal} episodes')


@case('DESIGN.md §8.1 -- turning a diagnostic on must not change the training '
      'trajectory')
def test_diagnostics_are_inert(ctx: Ctx) -> None:
    """Two runs identical but for log_diagnostics give the same per-episode returns."""
    traces: dict[bool, list[tuple]] = {}
    # The two runs this case produced, held locally. Reading them off a
    # module-level list indexed from the front meant that under `--full` the
    # rows inspected belonged to whichever case had trained first.
    rows_of: dict[bool, list[dict]] = {}
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
        rows_of[flag] = rows
        traces[flag] = [(r['episode'], r['return'], r['length'], r['epsilon'],
                         r['env_steps'], r['updates']) for r in rows]

    on, off = traces[True], traces[False]
    same(len(on), len(off), 'the two runs completed different episode counts')
    req(max(int(r['updates']) for r in rows_of[True]) > 0,
        'no gradient update occurred in the diagnostics-on run, so the '
        'buffer-sampling coupling this test exists to detect was never '
        'exercised.')
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
    on_rows = rows_of[True]
    diag_cols = ('v_abs_mean', 'a_abs_mean', 'a_spread', 'dead_unit_frac',
                 'grad_norm_trunk', 'effective_rank')
    present = {c for r in on_rows for c in diag_cols if r.get(c) is not None}
    req(present,
        f'log_diagnostics=True produced none of {diag_cols}, so the inertness '
        f'claim is untested: nothing was instrumented.')
    off_rows = rows_of[False]
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
    """Train `cfg` quietly and return its manifest.

    Deliberately keeps no module-level record of the runs it produced. It used
    to append each run directory to a global list, and `test_diagnostics_are_inert`
    then read `_last_run_dirs[0]` and `[1]`: under `--full` the epsilon case
    trains twice first and `ctx.cleanup` deletes its temporary trees, so the
    diagnostics case opened a deleted path, died with FileNotFoundError and was
    reported as a guard failure. Even without the crash it was inspecting the
    wrong pair of runs. A caller that needs the paths already has the configs.
    """
    from src.dqn.train import train

    with contextlib.redirect_stdout(io.StringIO()):
        manifest = train(cfg)
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

    # The recorded half of this guard is `test_recorded_dataset_integrity`.
    # It used to live here and returned early with a note when `--runs` held
    # no manifests, which counted the case as PASSED: an empty tree, a tree
    # carrying only a per_seed.csv, and a mistyped --runs path all reported
    # that the freeze verification had been checked against the dataset when
    # it had not. A guard that could not be checked is not a guard that held.
    ctx.note('1-ULP violation caught in both directions; freeze sets '
             'resolved by name')


@case('DESIGN.md 8.2, 8.4 -- the RECORDED dataset is contiguous, every '
      'finished run carries a manifest, and every closed freeze window was '
      'verified')
def test_recorded_dataset_integrity(ctx: Ctx) -> None:
    """Re-derives the integrity verdicts from the tree passed as --runs.

    `DESIGN.md` 8.2 claims the recorded dataset is contiguous and free of the
    duplication the published loop produced on resume, and 8.4 makes the freeze
    verdicts a refusal condition for reporting. Neither claim was checked
    against a recorded tree: `MetricsLog.check` was applied only to runs this
    file synthesises itself, and the tree behind `--runs` was opened for
    manifests alone. A duplicated episode row injected into a real
    `metrics.jsonl` and a deleted `manifest.json` both left the suite's verdict
    unchanged, so "44/44 metrics contiguous" was not a fact this suite could
    check.

    Three things are re-derived here, none of them read off the verdict a run
    recorded about itself:

    * the episode index set of every `metrics.jsonl`, recomputed and then
      compared against the manifest's stored verdict, so a stale or forged
      `result.metrics_integrity` is caught as well as a corrupt log;
    * a manifest for every run that finished, detected by the final model the
      trainer writes, so a deleted manifest is a failure rather than a smaller
      run count;
    * a verification verdict on every freeze window that CLOSED. Counting
      verdicts and requiring one was satisfied by a single verdict: stripping
      the `verification` block from 23 of 24 verdict-carrying manifests left
      the old check passing, which is a dataset in which freeze verification
      silently stopped being written.
    """
    from src.dqn.metrics import MetricsLog

    pattern = os.path.join(ctx.runs, '*', '*', 's*')
    manifests = sorted(glob.glob(os.path.join(pattern, 'manifest.json')))
    metric_logs = sorted(glob.glob(os.path.join(pattern, 'metrics.jsonl')))
    if not manifests and not metric_logs:
        ctx.skip(f'no recorded runs under {ctx.runs}; the recorded-dataset '
                 f'claims of DESIGN.md 8.2 cannot be checked against it')

    # -- 1. a finished run without a manifest is a DELETED manifest, not a
    #       smaller dataset. The trainer writes `model.keras` and the manifest
    #       at the end of a run, so a directory holding the model and no
    #       manifest lost the manifest; one holding neither is a run that was
    #       interrupted, which is reported rather than failed because a tree is
    #       allowed to contain an unfinished run.
    manifest_dirs = {os.path.dirname(p) for p in manifests}
    orphans, unfinished = [], []
    for path in metric_logs:
        run_dir = os.path.dirname(path)
        if run_dir in manifest_dirs:
            continue
        if os.path.exists(os.path.join(run_dir, 'model.keras')):
            orphans.append(os.path.relpath(run_dir, ctx.runs))
        else:
            unfinished.append(os.path.relpath(run_dir, ctx.runs))
    req(not orphans,
        f'{len(orphans)} run director(ies) hold a finished model and a metrics '
        f'log but no manifest.json: {orphans[:6]}. Every module that reads this '
        f'tree globs manifests, so such a run is silently absent from the audit '
        f'and from the analysis while its rows may already sit in per_seed.csv. '
        f'A run that cannot be identified cannot be excluded either.')

    # -- 2. contiguity, recomputed from the log and cross-examined against the
    #       verdict the run stored about itself.
    total_rows = 0
    problems: list[str] = []
    gates: set[float] = set()
    for path in manifests:
        run_dir = os.path.dirname(path)
        rel = os.path.relpath(run_dir, ctx.runs)
        with open(path, encoding='utf-8') as fh:
            manifest = json.load(fh)
        result = manifest.get('result') or {}
        log_path = os.path.join(run_dir, 'metrics.jsonl')
        if not os.path.exists(log_path):
            problems.append(f'{rel}: manifest but no metrics.jsonl')
            continue
        completed = result.get('episodes_completed')
        verdict = MetricsLog(log_path).check(
            expected=int(completed) if completed is not None else None)
        total_rows += int(verdict['rows'])
        if not verdict['contiguous']:
            problems.append(f'{rel}: {verdict["problems"]}')
            continue
        stored = result.get('metrics_integrity') or {}
        if stored:
            if stored.get('contiguous') is not True:
                problems.append(f'{rel}: the run recorded itself as '
                                f'non-contiguous: {stored.get("problems")}')
            elif int(stored.get('rows') or -1) != int(verdict['rows']):
                problems.append(
                    f'{rel}: the manifest records {stored.get("rows")} metric '
                    f'rows, the log holds {verdict["rows"]}')
        # The trainer's own copy of the source-validity gate, visible in the
        # data rather than only in its source. `registry.py` states that the
        # runner's constant and the catalogue's "must agree", and a recorded
        # manifest is the only place the value the runner applied can be read.
        validity = ((manifest.get('source') or {}).get('validity') or {})
        if validity.get('gate') is not None:
            gates.add(float(validity['gate']))
    req(not problems,
        f'{len(problems)} recorded run(s) failed the DESIGN.md 8.2 integrity '
        f'claim. The published loop appended to its metrics file on resume '
        f'without truncating, so a 45-episode run interrupted at 39 recorded '
        f'episodes 31-39 twice and every window statistic was computed over '
        f'duplicated episodes. Failures:\n    ' + '\n    '.join(problems[:8]))

    if gates:
        from experiments import registry
        same(sorted(gates), [float(registry.SOURCE_VALIDITY_GATE)],
             f'the source-validity gate the RUNNER applied, read back off the '
             f'recorded manifests, is not '
             f'registry.SOURCE_VALIDITY_GATE={registry.SOURCE_VALIDITY_GATE}. '
             f'The number is duplicated in src/dqn/train.py because the '
             f'planner has to apply it before any dependent run exists; a '
             f'disagreement means the dataset was gated at one threshold and '
             f'is being analysed at another. In P0 a source scored 0.599 '
             f'against 0.600, so a hundredth decides an arm.')

    # -- 3. every freeze window that CLOSED carries a verification verdict.
    #       A window opens with frozen=True at zero updates and closes with
    #       frozen=False after some updates, naming the layers it held; a
    #       scratch run emits a frozen=False event that opens nothing, and a
    #       run whose budget ends inside the window never closes it. The
    #       closing events are the ones a verdict is owed for.
    opened = closed = verified = 0
    unverified: list[str] = []
    violations: list[str] = []
    for path in manifests:
        rel = os.path.relpath(os.path.dirname(path), ctx.runs)
        with open(path, encoding='utf-8') as fh:
            manifest = json.load(fh)
        config = manifest.get('config') or {}
        for event in manifest.get('freeze_events') or []:
            layers = event.get('freeze_layers') or []
            if event.get('frozen'):
                opened += 1
                req(int(event.get('frozen_params') or 0) > 0
                    or config.get('freeze_group') in (None, '', 'none'),
                    f'{rel}: a freeze event declares frozen=True but froze '
                    f'zero parameters, so the window is nominal only.')
                continue
            if not (layers and int(event.get('updates') or 0) > 0):
                continue                     # nothing was ever frozen here
            closed += 1
            verification = event.get('verification')
            if verification is None:
                unverified.append(rel)
            else:
                verified += 1
                if not verification.get('ok'):
                    violations.append(f'{rel}: {verification}')
    req(not violations,
        f'{len(violations)} recorded freeze window(s) failed verification: a '
        f'layer declared frozen moved during the window.\n    '
        + '\n    '.join(violations[:8]))
    req(not unverified,
        f'{len(unverified)} of {closed} CLOSED freeze window(s) carry no '
        f'verification verdict: {sorted(set(unverified))[:6]}. Requiring only '
        f'that some verdict exists is satisfied by one: stripping the block '
        f'from 23 of 24 verdict-carrying manifests left the old check passing, '
        f'which is a dataset in which the verification silently stopped being '
        f'written. Recovering the published freeze map required diffing saved '
        f'checkpoints months after the fact, and that is what an unverified '
        f'window costs.')
    req(closed > 0,
        f'{len(manifests)} recorded runs under {ctx.runs}, {opened} freeze '
        f'window(s) opened and not one closed inside the budget. The '
        f'verification, the tf.function retrace and the optimiser\'s survival '
        f'across the transition are all untested by this dataset '
        f'(registry.SMOKE_OVERRIDES sizes E0 so the boundary IS crossed).')
    note = (f'{len(manifests)} runs, {total_rows} metric rows contiguous and '
            f'agreeing with their manifests; {verified}/{closed} closed freeze '
            f'windows verified, 0 violations')
    if unfinished:
        note += (f'; {len(unfinished)} unfinished run director(y/ies), no '
                 f'manifest and no final model: {unfinished[:3]}')
    if gates:
        note += f'; runner gate {sorted(gates)}'
    ctx.note(note)


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
                # constant. Asserted rather than only printed: the note used to
                # say "not constant, as documented" whatever the number was, so
                # if center_scale ever degenerated to a constant shift the
                # printed claim would have been false -- a hard-coded adjective
                # on an unmeasured property, in a suite whose standard is that
                # direction words are generated from the data.
                shift_sd = float(np.std(shift))
                req(shift_sd > 1e-4,
                    f'{aggregation}/center_scale: Q shifted by a CONSTANT '
                    f'(shift SD {shift_sd:.3e}). center_scale rescales the '
                    f'value kernel as well as centring it, so its shift must '
                    f'vary across states; a constant one means the mode is '
                    f'indistinguishable from `center`, and the two arms of E11 are '
                    f'one arm.')
                ctx.note(f'{aggregation}/center_scale shift SD '
                         f'{shift_sd:.3f} (measured, not constant)')

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
    # Read from the catalogue, never restated. Setting
    # registry.SOURCE_VALIDITY_GATE to 0.30 left the literal `gate = 0.6` here
    # passing and still printing "Acrobot gate 0.6", on the exact guard that
    # P0's 0.599-against-0.600 rejection turns on.
    gate = float(registry.SOURCE_VALIDITY_GATE)
    req(0.0 < gate < 1.0,
        f'registry.SOURCE_VALIDITY_GATE is {gate!r}. The gate is a NORMALISED '
        f'score, so it lies strictly between the measured random policy (0) '
        f'and the registered threshold (1); outside that interval it is either '
        f'unreachable or free (DESIGN.md 4.3).')
    # The same number lives in three places -- the catalogue, the analysis and
    # the trainer -- because the planner must apply it before any dependent run
    # exists and importing the trainer would pull TensorFlow into planning.
    # `registry.py` says "the two must agree"; agreement is checked here rather
    # than asserted in a comment. The trainer's copy is checked against the
    # recorded manifests in test_recorded_dataset_integrity, which is where the
    # value it actually used is visible.
    from experiments import stats as stats_mod
    same(float(stats_mod.SOURCE_VALIDITY_GATE), gate,
         'stats.SOURCE_VALIDITY_GATE and registry.SOURCE_VALIDITY_GATE '
         'disagree. One of them decides which sources enter the primary '
         'estimand and the other decides which runs the sweep launches, so a '
         'disagreement silently splits the analysis set from the dataset.')
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
        f'{acrobot}: the legacy multiplicative gate {gate} x threshold = '
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

    # -- the plan's constants are duplicated between the primitives and the
    #    caller that applies them, so pinning one copy pins nothing. Setting
    #    stats.MIN_N_FOR_INFERENCE to 1 left this case passing, and
    #    test_n1_is_labelled then failed with the wrong diagnosis ("the
    #    selection did not produce an arm below n=1") instead of the real
    #    violation. Each pair is checked, not just statlib's side.
    from experiments import stats as stats_mod
    for name in ('MIN_N_FOR_INFERENCE', 'ALPHA', 'CONFIRMATORY_FAMILY_SIZE',
                 'EQUIVALENCE_MARGIN', 'N_BOOT', 'BOOTSTRAP_SEED'):
        here = getattr(sl, name)
        there = getattr(stats_mod, 'BOOT_SEED' if name == 'BOOTSTRAP_SEED'
                        else name, None)
        req(there is not None,
            f'stats.py no longer declares {name}, so the pre-registered '
            f'constant is pinned in statlib and read from nowhere.')
        same(there, here,
             f'stats.{name} and statlib.{name} disagree ({there!r} vs '
             f'{here!r}). ANALYSIS_PLAN.md pins one number; two copies that '
             f'drift mean the pre-registration no longer describes the code '
             f'that produced the output.')
    same(stats_mod.VALIDATION_STAMP, sl.PIPELINE_VALIDATION_LABEL,
         'stats.VALIDATION_STAMP and statlib.PIPELINE_VALIDATION_LABEL '
         'disagree, so the stamp a reader sees is not the one the plan pins')

    # -- and the two copies the loop above did NOT reach. Pinning statlib's
    #    THRESHOLD_LEVELS pinned nothing: no statlib function reads it, while
    #    `aggregate` computes the steps_to_threshold columns from its own copy
    #    and `stats`/`plots` label them from a third. Setting aggregate's p100
    #    level to 0.90 left this suite at "28 passed, 0 failed" and moved
    #    `steps_to_threshold_p100` in 43 of 44 rows, with `level = 1.0` still
    #    printed beside them. `aggregate` was not imported by this file at all,
    #    so its copy of every constant was uncovered in both directions.
    #
    #    The shapes differ -- statlib holds bare floats, the two live copies
    #    hold (name, level) pairs -- which is why the loop above skipped it, so
    #    the comparison is made on the levels.
    from experiments import aggregate as agg_mod
    from experiments import audit as audit_mod
    same(tuple(agg_mod.THRESHOLD_LEVELS), tuple(stats_mod.THRESHOLD_LEVELS),
         'aggregate.THRESHOLD_LEVELS and stats.THRESHOLD_LEVELS disagree. '
         'aggregate computes the steps_to_threshold columns and stats labels '
         'and analyses them, so two copies that drift put a column computed at '
         'one level under a heading naming another (ANALYSIS_PLAN.md §5).')
    same(tuple(float(level) for _name, level in agg_mod.THRESHOLD_LEVELS),
         tuple(float(level) for level in sl.THRESHOLD_LEVELS),
         'the censored-metric levels aggregate.py computes from and the ones '
         'statlib.py declares disagree. The shapes differ, the numbers may '
         'not.')
    same(agg_mod.MIN_N_FOR_INFERENCE, sl.MIN_N_FOR_INFERENCE,
         'aggregate.MIN_N_FOR_INFERENCE and statlib.MIN_N_FOR_INFERENCE '
         'disagree, so the tree that gets stamped PIPELINE VALIDATION is not '
         'the tree the analysis refuses to test')
    same(agg_mod.VALIDATION_STAMP, sl.PIPELINE_VALIDATION_LABEL,
         'aggregate.VALIDATION_STAMP and statlib.PIPELINE_VALIDATION_LABEL '
         'disagree, so per_seed.csv and stats.py stamp the same tree with '
         'different words')
    same(audit_mod.MIN_N_FOR_A_RESULT, sl.MIN_N_FOR_INFERENCE,
         'audit.MIN_N_FOR_A_RESULT and statlib.MIN_N_FOR_INFERENCE disagree, '
         'so the audit and the analysis do not agree on which trees are '
         'results (ANALYSIS_PLAN.md §9)')
    # DESIGN.md §3.1 names ONE tolerance and names audit.py as the module that
    # refuses. There are two copies, and the loop above pinned neither: with
    # audit.FRACTION_TOLERANCE widened to 0.50 the whole suite still passed,
    # while widening stats.INTENSITY_TOLERANCE failed test_stats_intensity_gate.
    # One of the two was guarded and it was not the one the design names.
    same(audit_mod.FRACTION_TOLERANCE, stats_mod.INTENSITY_TOLERANCE,
         f'audit.FRACTION_TOLERANCE is {audit_mod.FRACTION_TOLERANCE!r} and '
         f'stats.INTENSITY_TOLERANCE is {stats_mod.INTENSITY_TOLERANCE!r}. '
         f'DESIGN.md §3.1 declares a single tolerance and makes audit.py the '
         f'module that refuses on it, so two copies that drift means the audit '
         f'and the analysis give opposite verdicts on the same cross-'
         f'architecture group.')

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
# The cases below read the recorded dataset. They select a subset of it,
# because `stats.py` correctly REFUSES to aggregate an arm whose invariants
# differ and a recorded tree holds several. Every selection rule below is a
# filter over real rows: nothing is synthesised, and the rules are chosen so
# that what survives is the arm the pre-registration describes rather than
# whichever arm happened to be first.
# ---------------------------------------------------------------------------
#: What makes one recorded row distinct from another within an arm. A demo tree
#: can hold two runs sharing a trajectory digest and differing only in their
#: measurement digest (one logged diagnostics, one did not). Both are real runs
#: of the same arm, so an arm carrying both has two rows per seed, and
#: `stats.py` refuses the doubled arm. The refusal is asserted in
#: `test_stats_excludes_tune_and_partial_arms`; here one row per key is kept so
#: the confirmatory guard has a well-formed arm to run on.
_SELECTION_KEY = ('env', 'cell', 'condition', 'label', 'seed')


def _per_seed_path(ctx: Ctx) -> str:
    path = os.path.join(ctx.runs, 'per_seed.csv')
    if not os.path.exists(path):
        ctx.skip(f'{path} not found; run experiments/aggregate.py, or pass '
                 f'--runs at a tree that has one')
    return path


def _target_seeds(ctx: Ctx, frame) -> list[int]:
    """The target-side seeds in a recorded table, in order.

    Target-side means the analysis blocks: `CONFIRM` and `REPLICATE`. The donor
    blocks (`C4SRC`, `RESERVE`) supply sources and are never units of analysis,
    and `TUNE` may not enter a reported estimate at all (`DESIGN.md` 3.4).
    """
    from experiments import registry

    donor = set(registry.SEED_BLOCKS['C4SRC']) | set(
        registry.SEED_BLOCKS['RESERVE'])
    tune = set(registry.SEED_BLOCKS['TUNE'])
    seeds = sorted({int(s) for s in frame['seed'].dropna().unique()})
    return [s for s in seeds if s not in donor and s not in tune]


def _donor_seeds(ctx: Ctx, frame) -> list[int]:
    """The source-donor seeds a transfer arm needs, TUNE excluded by name.

    The old selection kept `[s for s in seeds if s >= 200]` and described it as
    "the source-donor blocks the transfer arms need". 200-204 is the TUNE block,
    which `DESIGN.md` 3.4 and `ANALYSIS_PLAN.md` 8 forbid in any reported
    estimate, so the selection that exercised the n<3 guard would have pulled
    the selection block into the estimate on any tree that had one. The blocks
    are named here instead of guessed from a threshold.
    """
    from experiments import registry

    donor = set(registry.SEED_BLOCKS['C4SRC']) | set(
        registry.SEED_BLOCKS['RESERVE'])
    return sorted({int(s) for s in frame['seed'].dropna().unique()
                   if int(s) in donor})


def _select_rows(ctx: Ctx, seeds: Sequence[int] | None = None):
    """The analysable subset of the recorded per_seed table, as a DataFrame."""
    import pandas as pd

    from experiments import registry

    frame = pd.read_csv(_per_seed_path(ctx))

    # One freeze window, and specifically the PRE-REGISTERED one. Keeping every
    # window is what stats.py refuses, and correctly so (DESIGN.md 8.4). The
    # old selection took `windows[0]`, the smallest, and called the result "the
    # primary transfer arm": in runs_demo the windows are 150, 200 and 10000
    # while registry.PROTOCOL freezes for 10000 updates, so the only p-value
    # guard in the suite was being exercised on a non-protocol arm.
    protocol_window = int(registry.PROTOCOL['freeze_updates'])
    windows = sorted({int(v) for v in
                      frame.loc[frame['condition'] == 'transfer',
                                'freeze_updates'].dropna().unique()})
    if len(windows) > 1:
        req(protocol_window in windows,
            f'the recorded transfer arms use freeze windows {windows} and none '
            f'of them is the pre-registered registry.PROTOCOL window of '
            f'{protocol_window} updates. Selecting any other window would '
            f'exercise the confirmatory guards on an arm the plan does not '
            f'describe.')
        frame = frame[(frame['condition'] == 'scratch')
                      | (frame['freeze_updates'] == protocol_window)]

    # One row per unit. See `_SELECTION_KEY`.
    #
    # This selection REPAIRS the input, and a green suite over a repaired table
    # is greener than the tree deserves: `runs_demo/per_seed.csv` has 124 of its
    # 198 rows duplicated on this key, and without the drop `stats.py` refuses
    # every confirmatory member with "the scratch arm has more than one row for
    # seed(s) [0, 1, 2]". The repair is deliberate -- the confirmatory guards
    # need a well-formed arm to run on, and the doubled-arm refusal is asserted
    # against `stats.py` directly in `test_stats_excludes_tune_and_partial_arms`
    # and in `test_duplicate_unit_moves_no_number` -- but it is not silent, or
    # the suite would report a pass earned by the harness rather than by the
    # tree.
    present_keys = [k for k in _SELECTION_KEY if k in frame.columns]
    if 'run_digest' in frame.columns:
        frame = frame.sort_values(present_keys + ['run_digest'])
    before = len(frame)
    frame = frame.drop_duplicates(subset=present_keys, keep='first')
    if before != len(frame):
        # Once per case, not once per call: several cases select three or four
        # times and the line is a fact about the tree, not about the call.
        told = (f'{before - len(frame)} of {before} recorded row(s) are '
                f'duplicated on {"/".join(present_keys)} and were dropped by '
                f'the harness selection; the analysis path does not make that '
                f'repair, it refuses the doubled arm')
        if told not in ctx.notes:
            ctx.note(told)

    if seeds is not None:
        frame = frame[frame['seed'].isin([int(s) for s in seeds])]
    if not len(frame):
        ctx.skip('the selection is empty')
    return frame


def _write_per_seed(ctx: Ctx, frame) -> str:
    """Write a selected frame to a temporary per_seed.csv and return the path."""
    out = os.path.join(ctx.tmpdir('perseed_'), 'per_seed.csv')
    frame.to_csv(out, index=False)
    return out


def _filtered_per_seed(ctx: Ctx, seeds: Sequence[int] | None = None) -> str:
    """Write the standard selection of the recorded per_seed.csv to a file."""
    return _write_per_seed(ctx, _select_rows(ctx, seeds=seeds))


#: Stamped into every `run_dir` of the fixture. A per_seed table with these
#: paths in it can never be mistaken for recorded data, in a bundle or in a
#: transcript, and grep finds every artefact derived from one.
_FIXTURE_MARKER = 'SYNTHETIC-FIXTURE-NOT-A-RESULT'

#: Relative SD of the fixture noise. Large enough that no arm is constant (a
#: constant arm is the degenerate case, which has its own guards and would mask
#: these), small enough that the replicas stay recognisably the recorded run.
_FIXTURE_NOISE = 0.06


def _fixture_seed_count() -> int:
    """How many seeds the fixture carries: the pre-registered confirmatory n.

    Read from `registry.SEED_BLOCKS['CONFIRM']` rather than typed, so the
    fixture exercises the guards at exactly the n the plan commits to and not
    at the smallest n that clears the inference floor. The two are different
    regimes: at n=3 the exact sign-flip test cannot return a p below 0.25 and
    every confirmatory member is unfalsifiable by construction, so a guard that
    only ever ran at 3 would never see a tested member reach an alpha.
    """
    from experiments import registry

    return len(registry.SEED_BLOCKS['CONFIRM'])


def _fixture_frame(frame, n_seeds: int = 0):
    """The recorded table replicated across a full seed block, with jitter.

    THIS IS NOT DATA. It is a fixture: the numbers are the recorded ones with
    deterministic pseudo-random noise on the measured columns, and nothing
    computed from it may be quoted. What it is for is the one thing the
    recorded tree cannot do at a single seed: make the n>=3 half of the
    confirmatory guards EXECUTE.

    Two guards in this file -- the p-value confinement family check and the
    TUNE/partial-arm/doubled-arm/empty-endpoint refusals -- are the ones that
    stand between the confirmatory analysis and the paper, and both SKIPPED on
    the recorded tree because every member is suppressed by the n<3 floor. The
    skip was honest and the suite said so, but it meant the confirmatory path
    had never run end to end: `26 passed, 0 failed, 2 skipped` was reporting
    that the two cases covering the analysis the study exists to perform were
    unchecked. A fixture is not evidence about the experiment; it is evidence
    about the CODE, which is what this file tests.

    Construction, chosen so that nothing about the shape of the table changes:

    * every row is replicated across its own seed block, so a CONFIRM row
      becomes ten CONFIRM rows and a C4SRC donor row ten C4SRC donor rows, and
      the block structure the selection helpers read stays intact;
    * only columns `stats.METRIC_ROLES` calls co-primary, secondary or
      mechanism are perturbed. Identity, configuration and bookkeeping columns
      are untouched, so the arm digests, the transferred fractions the
      intensity gate reads and the freeze windows the selection filters on are
      the recorded ones;
    * the noise is drawn from a generator seeded on `(label, seed)`, so the
      fixture is byte-identical on every invocation and the `_STATS_CACHE`
      hits across cases;
    * `run_digest` and `run_dir` are rewritten per replica, because two rows
      sharing a digest are the doubled-arm corruption these cases inject on
      purpose and it must not arrive by accident.
    """
    import hashlib

    import numpy as np
    import pandas as pd

    from experiments import registry, stats

    n_seeds = int(n_seeds or _fixture_seed_count())
    jittered = tuple(
        column for column in frame.columns
        if stats.METRIC_ROLES.get(str(column)) in (
            stats.CO_PRIMARY, stats.SECONDARY, stats.MECHANISM))
    rows: list[dict] = []
    for _index, row in frame.iterrows():
        block = str(row.get('seed_block') or 'CONFIRM')
        seeds = registry.SEED_BLOCKS.get(block, (int(row.get('seed') or 0),))
        for seed in seeds[:n_seeds]:
            new = dict(row)
            new['seed'] = int(seed)
            tag = f'{row.get("label")}|{seed}'.encode()
            digest = hashlib.md5(tag).hexdigest()
            new['run_digest'] = digest
            new['run_dir'] = '/'.join(
                (_FIXTURE_MARKER, str(row.get('condition')), digest[:12],
                 f's{int(seed):02d}'))
            rng = np.random.default_rng(int(digest[:8], 16))
            for column in jittered:
                value = pd.to_numeric(new.get(column), errors='coerce')
                if value is None or not np.isfinite(value):
                    continue
                scale = abs(float(value))
                new[column] = float(value) + float(rng.normal(
                    0.0, _FIXTURE_NOISE * (scale if scale > 1e-9 else 1.0)))
            rows.append(new)
    return pd.DataFrame(rows, columns=list(frame.columns))


def _analysis_ctx(ctx: Ctx) -> tuple[Ctx, str]:
    """A context whose run tree carries enough seeds to test at, and its origin.

    Returns the given context unchanged when the recorded tree already has
    `stats.MIN_N_FOR_INFERENCE` target-side seeds, because real data beats a
    fixture wherever there is any. Otherwise it writes `_fixture_frame` to a
    temporary tree and returns a context pointing at that, so the caller runs
    the same assertions either way and never skips.

    The temporary directory is registered on the ORIGINAL context, so the
    per-case cleanup removes it; the returned context shares the same notes
    list, so whatever the caller records lands on the case's own line.
    """
    import dataclasses

    from experiments import stats

    frame = _select_rows(ctx)
    target_side = _target_seeds(ctx, frame)
    if len(target_side) >= stats.MIN_N_FOR_INFERENCE:
        return ctx, (f'the recorded tree at {ctx.runs} '
                     f'({len(target_side)} target-side seeds)')
    root = ctx.tmpdir('fixture_')
    _fixture_frame(frame).to_csv(os.path.join(root, 'per_seed.csv'),
                                 index=False)
    return dataclasses.replace(ctx, runs=root), (
        f'a SYNTHETIC {_fixture_seed_count()}-seed fixture: the recorded tree '
        f'at '
        f'{ctx.runs} has {len(target_side)} target-side seed(s), below the '
        f'n={stats.MIN_N_FOR_INFERENCE} floor, so the n>=3 half of this case '
        f'would otherwise never execute. No number below is a result')

#: One `stats.py` invocation costs 10-20 seconds and several cases below read
#: the same baseline report. Keyed on the CONTENT of the input table and on the
#: arguments, so a mutated selection never collides with the baseline it is
#: being compared against.
_STATS_CACHE: dict[tuple, tuple[int, str, dict]] = {}

#: Warnings raised during each cached invocation, keyed the same way. A
#: zero-variance arm makes scipy's pearsonr and spearmanr emit
#: ConstantInputWarning, which the suite used neither to suppress nor to
#: report: the run looked clean while the correlation it printed was not
#: defined. They are captured here so a degenerate arm is a stated fact.
_STATS_WARNINGS: dict[tuple, list[str]] = {}


def _run_stats(ctx: Ctx, per_seed: str, extra: Sequence[str] = ()
               ) -> tuple[int, str, dict]:
    """Invoke stats.py in-process and return (exit code, stdout, report)."""
    import hashlib

    from experiments import stats

    with open(per_seed, 'rb') as fh:
        key = (hashlib.sha256(fh.read()).hexdigest(), tuple(extra),
               bool(ctx.full))
    if key in _STATS_CACHE:
        return _STATS_CACHE[key]

    json_out = os.path.join(os.path.dirname(per_seed), 'report.json')
    argv = ['--per-seed', per_seed, '--json', json_out,
            '--n-boot', '300' if not ctx.full else str(stats.N_BOOT),
            *extra]
    buffer = io.StringIO()
    with warnings.catch_warnings(record=True) as raised:
        warnings.simplefilter('always')
        with contextlib.redirect_stdout(buffer):
            code = stats.main(argv)
    _STATS_WARNINGS[key] = [f'{w.category.__name__}: {w.message}'
                            for w in raised]
    text = buffer.getvalue()
    report: dict = {}
    if os.path.exists(json_out):
        with open(json_out, encoding='utf-8') as fh:
            report = json.load(fh)
    _STATS_CACHE[key] = (code, text, report)
    return code, text, report


def _baseline_report(ctx: Ctx) -> tuple[str, dict, str]:
    """The E1 pooled-policy report over the standard selection.

    Returned as (per_seed path, report, stdout). Shared by every case that
    needs a well-formed confirmatory report rather than a mutated one.
    """
    per_seed = _filtered_per_seed(ctx)
    code, text, report = _run_stats(ctx, per_seed,
                                    ['--experiments', 'E1',
                                     '--source-policy', 'pooled'])
    same(code, 0, f'stats.py exited {code}; the tail of its output was:\n'
                  f'{text[-1500:]}')
    req(report, 'stats.py wrote no JSON report, so the check has nothing to '
                'read')
    raised = _stats_warnings(ctx, per_seed, ['--experiments', 'E1',
                                            '--source-policy', 'pooled'])
    req(not raised,
        f'the pre-registered analysis path raised {len(raised)} '
        f'warning(s) on the recorded selection: {raised[:4]}. A '
        f'ConstantInputWarning out of pearsonr or spearmanr means a '
        f'correlation was reported for an arm with no variance, and a '
        f'RuntimeWarning inside an estimate means the number beside it is '
        f'not the quantity its name claims. Neither may reach a report '
        f'unremarked.')

    # The TUNE exclusion, checked on every invocation rather than only in
    # the case that injects the contamination: that case needs three seeds
    # to distinguish its refusals and skips below them, so on a single-seed
    # tree carrying TUNE rows nothing else would look. DESIGN.md 3.4 bars
    # the selection block from every reported estimate at every n.
    import pandas as pd

    selected = pd.read_csv(per_seed)
    in_table = (int((selected['seed_block'] == 'TUNE').sum())
                if 'seed_block' in selected.columns else 0)
    # The property that matters is that no TUNE row reaches the reported
    # inventory. The excluded COUNT is not compared against the count in
    # the table, because stats.py counts within the experiments selected
    # and a TUNE row belonging to some other experiment is dropped by the
    # experiment filter rather than by the block rule; requiring the two
    # numbers to agree would fail on a correct exclusion.
    leaked = [row for row in (report['s2_inventory'].get('seed_blocks')
                              or [])
              if str(row.get('seed_block')) == 'TUNE'
              and int(row.get('runs') or 0) > 0]
    req(not leaked,
        f'{leaked} TUNE-block run(s) reached the reported inventory. '
        f'Revision 1 selected hyperparameters on seeds 0-4 and then ran '
        f'every confirmatory arm on 0-9, so half of each confirmatory '
        f'sample had been tuned on; DESIGN.md 3.4 and ANALYSIS_PLAN.md 8 '
        f'bar the block from every reported estimate, at every n.')
    if in_table:
        req(int(report['invocation'].get('tune_runs_excluded') or 0) > 0,
            f'the selection carries {in_table} TUNE-block row(s) and the '
            f'report records none as excluded, so the block rule did not '
            f'fire on a table that needed it.')
        req('TUNE' in text,
            'TUNE-block rows were excluded and the output does not say '
            'so; an exclusion nobody can see is indistinguishable from '
            'an estimate that included them.')
    return per_seed, report, text


def _stats_warnings(ctx: Ctx, per_seed: str,
                    extra: Sequence[str]) -> list[str]:
    """The warnings the cached invocation of `per_seed` under `extra` raised."""
    import hashlib

    with open(per_seed, 'rb') as fh:
        key = (hashlib.sha256(fh.read()).hexdigest(), tuple(extra),
               bool(ctx.full))
    return list(_STATS_WARNINGS.get(key, []))


def _members(report: dict) -> list[dict]:
    return list(report['s5_confirmatory']['members'])


def _tested(members: Sequence[dict]) -> list[dict]:
    return [m for m in members if m.get('p_signflip') is not None]


#: Keys whose *name* matches the p-value pattern but which are not p-values.
#: Each one is VERIFIED below against the property that justifies its
#: exemption, never merely excused: an unconditional exemption keyed on a name
#: is a hiding place, and injecting a genuine p-value into `s7_controls` under
#: the name `min_attainable_p` walked straight past the earlier version of this
#: guard while it reported "0 p-values outside the family".
_NOT_A_P_VALUE = {
    # P(threshold reached within budget): a proportion with a Clopper-Pearson
    # interval beside it, which is the primary censored summary of §5.
    'p_reached',
    # The smallest two-sided p the test COULD return at that n, reported so a
    # floored p is distinguishable from strong evidence. Verified to be a
    # function of n alone by re-deriving it from statlib.
    'min_attainable_p',
    # A count of screen members, not a q-value. Verified to be a non-negative
    # integer, which no p-value is.
    'screen_q_count',
}

#: Names that carry a p-value or a significance verdict. Deliberately wider
#: than "starts with p_": `significance` was matched by nothing, and neither
#: were `prob`, `alpha_observed` and `two_sided`, so a genuine p-value could be
#: emitted anywhere in the report simply by naming it something else.
_P_LIKE = re.compile(
    r'(^|_)(p|q)($|_)|p_?val|pvalue'
    r'|signific|(^|_)prob($|_)|alpha_observed|two_sided|one_sided'
    r'|(^|_)alpha($|_)',
    re.IGNORECASE)


def _procedural_alpha_levels() -> dict[str, tuple[float, ...]]:
    """p-like names that may carry a LEVEL, and the exact levels each may carry.

    These names are bookkeeping about the procedure: `alpha` is the
    pre-registered nominal level, `alpha_strictest` the strictest Holm step,
    `adjusted_alpha` whichever step a row is for. They match `_P_LIKE` because
    the pattern was widened to cover `alpha`, and they have to be let through
    or the ledger `ANALYSIS_PLAN.md` 7 requires on every invocation would
    itself be an offender.

    The predecessor let them through on a RANGE, `0 < value <= ALPHA`. ALPHA is
    0.05, so the test admitted precisely the p-values that matter: a genuine
    p of 0.0031 emitted into `s7_controls` as `alpha_bh` passed, and so did the
    same number under `alpha`, `alpha_nominal`, `adjusted_alpha`, `holm_alpha`,
    `strictest_alpha`, `alpha_strictest` and `alpha_holm8`. Widening `_P_LIKE`
    to cover alpha names therefore bought nothing for eight of them, and the
    exemption's own stated justification -- "constants of the plan, not
    quantities computed from the data" -- was never checked.

    It is checked here. Each name is pinned to the constant it claims to be,
    derived from `statlib` rather than typed, and a value that is not that
    constant is not a level: it is a number computed from the data wearing a
    level's name, which is the thing the guard exists to catch.

    `alpha_bh` is deliberately absent. Nothing in the repo emits it, and a
    Benjamini-Hochberg critical value is `i * alpha / m` for the i-th of m
    screens, so it is not a constant of the plan and cannot be pinned to one.
    Under the exemption it was the loosest name of the eight; outside it, it is
    treated as any other p-like name, which under `/screens` still leaves the
    orientation-only q-values licensed.
    """
    from experiments import statlib as sl

    nominal = float(sl.ALPHA)
    family = int(sl.CONFIRMATORY_FAMILY_SIZE)
    strictest = nominal / family
    # Holm steps down from alpha/m to alpha: any row of the ledger may name any
    # one of them, and nothing else.
    holm_steps = tuple(nominal / k for k in range(1, family + 1))
    return {
        'alpha': (nominal,),
        'alpha_nominal': (nominal,),
        'alpha_strictest': (strictest,),
        'strictest_alpha': (strictest,),
        'alpha_holm8': (strictest,),
        'holm_alpha': holm_steps,
        'adjusted_alpha': holm_steps,
    }


def _as_number(value: Any) -> Optional[float]:
    """The value as a float, or None where it is not a number at all.

    A p-like name carrying a label ("nominal", "step-down from 0.00625") or an
    unfilled slot (null) states no p. A string that PARSES as a number is not a
    label, because quoting a p-value does not stop it being one, so it comes
    back as the number it is.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _walk(node: Any, path: str = '') -> Iterable[tuple[str, str, Any, Any]]:
    """Yield (path, key, value, parent) over every dict entry in a JSON tree."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield f'{path}/{key}', str(key), value, node
            yield from _walk(value, f'{path}/{key}')
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _walk(value, f'{path}[{index}]')


def _hits_q(parent: Any) -> bool:
    """True when the row carries a Benjamini-Hochberg q beside its raw p.

    `ANALYSIS_PLAN.md` §7's screen exemption is for the q. A row that reports
    the raw p WITH the q it fed into is reporting the derivation; a row that
    reports the raw p alone has simply relocated a p-value.
    """
    if not isinstance(parent, dict):
        return False
    return any(str(k).lower() in ('q', 'q_bh', 'q_value', 'bh_q')
               and _as_number(v) is not None for k, v in parent.items())


def _pvalue_offenders(ctx: Ctx, report: dict) -> tuple[list[str], int]:
    """Every p-value in the report that lives outside the permitted places.

    Permitted: the confirmatory family of 8, and the orientation-only
    Benjamini-Hochberg q of the screens (`ANALYSIS_PLAN.md` 7). Everything else
    is estimation-only, and a p-value elsewhere is how "positive transfer" gets
    claimed from p=0.421.
    """
    from experiments import statlib as sl

    levels = _procedural_alpha_levels()
    offenders: list[str] = []
    exempted = 0
    for path, key, value, parent in _walk(report):
        if not _P_LIKE.search(key):
            continue
        lowered = key.lower()
        if lowered in levels:
            # A level, not a result, and pinned to the exact constant it claims
            # to be rather than to a range. `0 < value <= ALPHA` admitted every
            # significant p-value there is, which is the opposite of a check.
            permitted = levels[lowered]
            numeric = _as_number(value)
            if numeric is None:              # a label, or an unfilled slot
                exempted += 1
                continue
            if any(abs(numeric - level) <= 1e-12 for level in permitted):
                exempted += 1
                continue
            offenders.append(
                f'{path} = {value!r} (exempt only as the pre-registered level '
                + ' or '.join(f'{level:g}' for level in permitted)
                + '; a number under this name that is not that level is a '
                  'quantity computed from the data)')
            continue
        if key in _NOT_A_P_VALUE:
            exempted += 1
            if value is None:
                continue
            if key == 'p_reached':
                req(0.0 <= float(value) <= 1.0,
                    f'{path}: p_reached={value!r} is outside [0, 1], so the '
                    f'name is not describing a proportion')
                req('cp_lo' in parent and 'cp_hi' in parent,
                    f'{path}: p_reached has no Clopper-Pearson interval beside '
                    f'it, so it cannot be the censored summary of '
                    f'ANALYSIS_PLAN.md 5 and the exemption is unjustified')
            elif key == 'min_attainable_p':
                # A function of n alone. Re-derived here, so a genuine p-value
                # cannot be smuggled in under this name: injecting 0.0031 into
                # s7_controls used to pass unexamined.
                n = parent.get('n')
                req(n is not None,
                    f'{path}: min_attainable_p is reported with no n beside '
                    f'it, so the claim that it is a floor rather than a result '
                    f'cannot be checked and the exemption is unjustified.')
                expected = sl.signflip_min_attainable_p(int(n))
                near(float(value), float(expected), 1e-12,
                     f'{path}: min_attainable_p={value!r} is not the exact '
                     f'sign-flip floor at n={n} ({expected!r}). The name is '
                     f'exempt from the p-value rule ONLY because the number is '
                     f'a function of n; a different number under this name is '
                     f'a p-value in an estimation-only section.')
            elif key == 'screen_q_count':
                req(float(value) >= 0 and float(value) == int(float(value)),
                    f'{path}: screen_q_count={value!r} is not a non-negative '
                    f'integer, so it is not the count of screen members its '
                    f'exemption assumes.')
            continue
        segments = path.lstrip('/').split('/')
        section = segments[0]
        # The one declared exception: §7 permits Benjamini-Hochberg q for the
        # screens, orientation only, never as an assertion.
        #
        # Matched on a path SEGMENT. `'/screens' in path` is a substring, so
        # `/s7_controls/screens_note/p_signflip` and
        # `/s7_controls/screensaver/p_signflip` were both exempt: a real
        # p-value could be moved out of the family by putting it under any key
        # that merely starts with "screens" (both reproduced).
        #
        # And the exception is for the BH q, not for anything under that key:
        # the raw p is the input BH is computed FROM, so it is admitted only
        # where the q computed from it sits beside it in the same row. A bare p
        # under `screens` with no q is a p-value outside the family wearing the
        # one exemption the plan grants.
        if section == 's5_confirmatory':
            continue
        if 'screens' in segments:
            if _as_number(value) is None or _hits_q(parent):
                exempted += 1
                continue
            offenders.append(
                f'{path} = {value!r} (under `screens`, where ANALYSIS_PLAN.md '
                f'§7 permits the Benjamini-Hochberg q for orientation only; '
                f'this row carries no q, so the exemption does not reach it)')
            continue
        # A p-value is a NUMBER. A p-like name carrying a label ("nominal",
        # "holm8") names which level a row is for, and an unfilled slot carries
        # null; neither states a p. A string that parses as a number is not let
        # through on that argument, because quoting a p-value does not stop it
        # being one.
        if value is None:
            exempted += 1
            continue
        if isinstance(value, str):
            try:
                float(value)
            except ValueError:
                exempted += 1
                continue
        elif isinstance(value, bool) or not isinstance(value, (int, float)):
            exempted += 1
            continue
        offenders.append(f'{path} = {value!r}')
    return offenders, exempted


@case('ANALYSIS_PLAN.md §2, §7 -- exactly one confirmatory family, and no '
      'p-value outside it')
def test_stats_no_pvalue_outside_family(ctx: Ctx) -> None:
    """Every p-value stats.py emits lives inside the family of 8."""
    from experiments import stats

    per_seed, report, text = _baseline_report(ctx)

    members = _members(report)
    same(len(members), stats.CONFIRMATORY_FAMILY_SIZE,
         f'the confirmatory section emitted {len(members)} members, not the '
         f'pre-registered {stats.CONFIRMATORY_FAMILY_SIZE}. Family membership '
         f'is fixed by the plan before launch; a family of a different size '
         f'means a result could be rescued by relocating it.')
    same(report['s5_confirmatory']['family_size'],
         stats.CONFIRMATORY_FAMILY_SIZE,
         'the reported family size is not 8')

    # The p-value confinement guard runs on whatever the report contains: it
    # does not need a member to have been tested, because an offender is a
    # p-value emitted OUTSIDE the family.
    offenders, exempted = _pvalue_offenders(ctx, report)
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

    # The other half of the case -- that a family member which IS tested
    # carries the full pre-specified apparatus -- needs a tested member. At one
    # seed the RECORDED tree has none, and that is the n<3 guard holding rather
    # than this guard failing (STANDING_INSTRUCTIONS S8 makes the single-seed
    # launch the current mode). Reporting it as a FAIL made the correct P0
    # dataset red while a mistyped --runs path stayed green.
    #
    # Reporting it as a SKIP, which is what came next, was honest and still
    # left the confirmatory path untested end to end: the half of this case
    # that checks the apparatus of a tested member had never executed once, on
    # any tree, and neither had the p-value walk over the sections an n>=3 run
    # writes and an n=1 run suppresses. So the recorded tree is checked for
    # what it can show, and then the same assertions run again against
    # `_analysis_ctx`: real data where the tree has the seeds, a synthetic
    # fixture where it does not. Neither branch skips.
    tested = _tested(members)
    if not tested:
        reasons = sorted({str(m.get('suppressed'))[:90] for m in members})
        floor = f'< {stats.MIN_N_FOR_INFERENCE}'
        req(all(floor in r for r in reasons),
            f'every confirmatory member was suppressed, and not by the n<'
            f'{stats.MIN_N_FOR_INFERENCE} floor. The selection is malformed, '
            f'so this case cannot show that p-values are confined to the '
            f'family. Reasons: ' + '; '.join(reasons))

    actx, origin = _analysis_ctx(ctx)
    if actx is ctx:
        at_n, at_n_text = report, text
    else:
        _path, at_n, at_n_text = _baseline_report(actx)
    at_n_members = _members(at_n)
    same(len(at_n_members), stats.CONFIRMATORY_FAMILY_SIZE,
         f'the confirmatory family has {len(at_n_members)} members against '
         f'{origin}, not the pre-registered '
         f'{stats.CONFIRMATORY_FAMILY_SIZE}')

    # The confinement walk again, over the report an n>=3 run actually writes.
    # At n=1 sections 5, 6, 9 and 10 are suppressed, so the walk above never
    # visits the places a p-value is most likely to escape to: the equivalence
    # verdicts, the censoring summaries, the screens and the power table.
    at_n_offenders, at_n_exempted = _pvalue_offenders(actx, at_n)
    req(not at_n_offenders,
        f'{len(at_n_offenders)} p-value(s) were emitted outside the '
        f'confirmatory family against {origin}. ANALYSIS_PLAN.md §7 permits '
        f'p-values in exactly one family. Offenders: '
        + '; '.join(at_n_offenders[:10]))
    req('MULTIPLICITY' in at_n_text.upper() or 'ledger' in at_n_text.lower(),
        f'the multiplicity ledger was not printed against {origin}')

    at_n_tested = _tested(at_n_members)
    at_n_reasons = sorted({str(m.get('suppressed'))[:70] for m in at_n_members})
    req(at_n_tested,
        f'not one of the {len(at_n_members)} confirmatory members was tested '
        f'against {origin}, so the apparatus below is still unchecked. '
        f'Suppression reasons: ' + '; '.join(at_n_reasons))
    for member in at_n_tested:
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
        # Holm steps DOWN from alpha/8, so an adjusted p is never smaller than
        # the raw one. A correction that loosens is not a correction, and this
        # is the arithmetic no n=1 tree could ever exercise.
        req(float(member['p_holm']) >= float(member['p_signflip']) - 1e-12,
            f'{member["metric"]}/{member["cell"]}: the Holm-adjusted p '
            f'{member.get("p_holm")!r} is SMALLER than the raw p '
            f'{member.get("p_signflip")!r}. Holm steps down from '
            f'alpha/{stats.CONFIRMATORY_FAMILY_SIZE}, so the adjusted p is '
            f'never below the raw one.')
    ctx.note(f'{len(offenders)} p-values outside the family on the recorded '
             f'tree; {exempted} name-collisions verified as non-p-values; '
             f'{len(at_n_tested)} of {len(at_n_members)} family members tested '
             f'and {len(at_n_offenders)} p-value(s) outside the family against '
             f'{origin.split(":")[0]}')


@case('ANALYSIS_PLAN.md §9, STANDING_INSTRUCTIONS S8 -- at n<3 no test is '
      'emitted and every page is stamped')
def test_n1_is_labelled(ctx: Ctx) -> None:
    """A single-seed selection produces no test and is stamped PIPELINE VALIDATION."""
    from experiments import stats

    frame = _select_rows(ctx)
    target_side = _target_seeds(ctx, frame)
    if not target_side:
        ctx.skip('the recorded tree has no target-side seeds')
    # One target-side seed, plus the source-donor blocks the transfer arms
    # need. TUNE is excluded by name in `_donor_seeds`: the old selection took
    # every seed >= 200, which is the selection block itself.
    keep = [target_side[0]] + _donor_seeds(ctx, frame)
    per_seed = _write_per_seed(ctx, _select_rows(ctx, seeds=keep))

    code, text, report = _run_stats(ctx, per_seed,
                                    ['--experiments', 'E1',
                                     '--source-policy', 'pooled'])
    same(code, 0,
         f'stats.py exited {code} on a single-seed selection -- the invocation '
         f'STANDING_INSTRUCTIONS S8 makes the CURRENT mode of every '
         f'experiment. A crash on that path means the n<3 guard of '
         f'ANALYSIS_PLAN.md §9 is unreachable, so single-seed output is '
         f'emitted with no stamp at all. Tail of the output:\n{text[-2500:]}')

    same(int(report['invocation'].get('tune_runs_excluded') or 0), 0,
         'the single-seed selection carries TUNE-block runs. DESIGN.md §3.4 '
         'and ANALYSIS_PLAN.md §8 forbid a reported estimate that touches the '
         'selection block, and this suite must not be the thing that puts one '
         'there.')

    members = _members(report)
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
        if int(member.get('n') or 0) == 0:
            # n=0 and n=1 are different facts. An arm with no usable rows
            # has to say that it has none, or "min n = 0" reads as a very
            # small sample and an unwritten endpoint column passes for one.
            reason = str(member.get('suppressed') or '').lower()
            req(reason and 'empty' in reason or 'no ' in reason,
                f'{member["metric"]}/{member["cell"]} reports n=0 and is '
                f'suppressed as {member.get("suppressed")!r}, which does '
                f'not say the arm is empty. A dataset whose co-primary '
                f'endpoint was never written would then be reported in the '
                f'same words as a dataset with one seed.')
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
    ctx.note(f'seeds kept {keep} (no TUNE); min n = {min(n_values)}; '
             f'0 tests emitted; stamp present')


@case('DESIGN.md §9 -- selection bias in tuning, silent seed dropping, a '
      'doubled arm and an empty endpoint are refused rather than averaged '
      'away', slow=True)
def test_stats_excludes_tune_and_partial_arms(ctx: Ctx) -> None:
    """Four ways a reported estimate can be quietly malformed, each refused.

    None of these was asserted anywhere. `stats.py` does refuse all three, but
    a guard nothing tests is a guard nothing notices the loss of: dropping seed
    2 from one transfer arm of `runs_demo` left both stats cases PASSING, and
    removing the TUNE exclusion from `stats.py` left the confirmatory guard
    passing on a TUNE-contaminated family of 8.
    """
    from experiments import registry, stats

    # Below n=3 every member is suppressed by the inference floor, which MASKS
    # all four refusals this case is about, so a pass on the recorded
    # single-seed tree would mean nothing and the case used to SKIP. The skip
    # was honest and left these four guards -- which are the ones standing
    # between a malformed sample and a confirmatory number -- never once
    # exercised. `_analysis_ctx` supplies a tree that HAS the seeds: the
    # recorded one where it does, a synthetic fixture where it does not. The
    # fixture is evidence about stats.py, not about the experiment.
    actx, origin = _analysis_ctx(ctx)
    frame = _select_rows(actx)
    target_side = _target_seeds(actx, frame)
    req(len(target_side) >= stats.MIN_N_FOR_INFERENCE,
        f'{origin} carries {len(target_side)} target-side seed(s), below the '
        f'n={stats.MIN_N_FOR_INFERENCE} floor, so the four refusals this case '
        f'asserts would all be masked by the floor instead of being shown to '
        f'fire.')

    args = ['--experiments', 'E1', '--source-policy', 'pooled']
    per_seed, baseline, _text = _baseline_report(actx)
    base_members = _members(baseline)
    base_tested = _tested(base_members)
    req(base_tested,
        'the baseline selection produced no tested confirmatory member, so '
        'none of the three refusals below can be distinguished from it.')

    # -- 1. TUNE leakage. The block exists so that hyperparameter selection and
    #       estimation never share a seed; revision 1 selected on seeds 0-4 and
    #       then ran every confirmatory arm on 0-9, so half of each sample had
    #       been tuned on.
    tune_seed = int(registry.SEED_BLOCKS['TUNE'][0])
    contaminated = frame[frame['condition'] == 'transfer'].head(4).copy()
    req(len(contaminated),
        'the selection holds no transfer rows to relabel, so the TUNE '
        'contamination cannot be injected')
    contaminated['seed'] = tune_seed
    contaminated['seed_block'] = 'TUNE'
    import pandas as pd
    tune_path = _write_per_seed(
        actx, pd.concat([frame, contaminated], ignore_index=True))
    code, tune_text, tune_report = _run_stats(actx, tune_path, args)
    same(code, 0, f'stats.py exited {code} on the TUNE-contaminated selection')
    same(int(tune_report['invocation'].get('tune_runs_excluded') or 0),
         len(contaminated),
         f'stats.py did not exclude the {len(contaminated)} TUNE-block run(s) '
         f'injected into the table. DESIGN.md §3.4 gives the block exactly one '
         f'purpose and ANALYSIS_PLAN.md §8 bars it from every reported '
         f'estimate; an estimate that silently includes it has been selected '
         f'on the sample it is reported over.')
    req(tune_seed in [int(s) for s in
                      (tune_report['invocation'].get('tune_seeds_excluded')
                       or [])],
        f'the excluded TUNE seed {tune_seed} is not named in the report, so a '
        f'reader cannot tell an exclusion happened')
    req('TUNE' in tune_text,
        'the printed output does not mention the TUNE exclusion, which §2c '
        'requires to be reported rather than performed silently')
    # And the exclusion must be complete: the confirmatory numbers may not move.
    def fingerprint(members: Sequence[dict]) -> list[tuple]:
        return [(m['metric'], m['cell'], m.get('n'), m.get('p_signflip'),
                 m.get('hl')) for m in members]
    same(fingerprint(_members(tune_report)), fingerprint(base_members),
         'the confirmatory family moved when TUNE-block rows were added to the '
         'input. Excluding them is not enough if they change the answer: a '
         'number that moved was computed on a sample that had been tuned on.')

    # -- 2. an incomplete arm. Nine of ten seeds is not the arm the plan
    #       declares, and averaging over what is present is exactly the silent
    #       seed dropping DESIGN.md §9 names.
    victim_seed = target_side[-1]
    cells = sorted({str(c) for c in frame['cell'].dropna().unique()})
    victim_cell = cells[0]
    drop = ((frame['condition'] == 'transfer')
            & (frame['cell'] == victim_cell)
            & (frame['seed'] == victim_seed))
    req(int(drop.sum()) > 0,
        f'no transfer row for cell {victim_cell} at seed {victim_seed}, so the '
        f'partial arm cannot be constructed')
    partial_path = _write_per_seed(actx, frame[~drop])
    code, _t, partial = _run_stats(actx, partial_path, args)
    same(code, 0, f'stats.py exited {code} on the partial-arm selection')
    hit = [m for m in _members(partial) if m['cell'] == victim_cell]
    req(hit, f'no confirmatory member for cell {victim_cell} to inspect')
    for member in hit:
        req(member.get('p_signflip') is None,
            f'{member["metric"]}/{victim_cell}: a test was emitted on an arm '
            f'missing seed {victim_seed}. DESIGN.md §8.4 refuses an incomplete '
            f'arm; a paired test over the seeds that happen to be present is a '
            f'test on a sample nobody declared.')
        req('incomplete arm' in str(member.get('suppressed') or '').lower(),
            f'{member["metric"]}/{victim_cell} at n={member.get("n")} was '
            f'suppressed for {member.get("suppressed")!r} rather than for the '
            f'missing seed, so a reader cannot tell a dropped seed from a '
            f'small sample.')
    untouched = [m for m in _members(partial) if m['cell'] != victim_cell]
    req(_tested(untouched),
        'removing one seed from ONE arm suppressed every other arm as well, so '
        'the refusal is not the targeted one this case is asserting.')

    # -- 3. a doubled arm. Two runs of the same arm and seed, differing only in
    #       their measurement digest, are both real; averaging them silently
    #       halves the effective n and makes the pairing wrong.
    doubled_path = _write_per_seed(
        actx, pd.concat([frame, frame[frame['condition'] == 'scratch']],
                        ignore_index=True))
    code, _t, doubled = _run_stats(actx, doubled_path, args)
    same(code, 0, f'stats.py exited {code} on the doubled-arm selection')
    req(not _tested(_members(doubled)),
        'a confirmatory test was emitted over an arm carrying two rows per '
        'seed. The pairing of ANALYSIS_PLAN.md §2 is by seed, so a doubled '
        'row is either a silent average or a duplicated unit, and both '
        'misstate n.')
    # -- 4. an endpoint that is entirely absent. n=0 and n=1 are different
    #       facts about a dataset, and "min n = 0" printed beside a suppression
    #       reason about the inference floor conflates them: an empty column is
    #       a broken aggregation, not a small sample.
    empty = frame.copy()
    empty['final_score'] = float('nan')
    code, _t, empty_report = _run_stats(actx, _write_per_seed(actx, empty),
                                        args)
    same(code, 0, f'stats.py exited {code} on the empty-endpoint selection')
    hits = [m for m in _members(empty_report) if m['metric'] == 'final_score']
    req(hits, 'the confirmatory family lost its final_score members entirely '
              'when the column was emptied; a co-primary endpoint that '
              'vanishes from the family cannot be reported as absent')
    for member in hits:
        req(member.get('p_signflip') is None,
            f'{member["cell"]}: a test was emitted on an endpoint with no '
            f'values at all')
        reason = str(member.get('suppressed') or '').lower()
        req('empty' in reason or 'no ' in reason,
            f'{member["cell"]}: an entirely empty final_score column was '
            f'suppressed as {member.get("suppressed")!r}, which does not say '
            f'the column is empty. A reader seeing n=0 beside the n<3 wording '
            f'cannot tell a dataset with one seed from a dataset whose '
            f'co-primary endpoint never got written.')
    others = [m for m in _members(empty_report) if m['metric'] != 'final_score']
    req(_tested(others),
        'emptying ONE co-primary column suppressed the other one too, so the '
        'refusal is not the targeted one this case is asserting.')

    ctx.note(f'against {origin.split(":")[0]}: TUNE run(s) excluded and the '
             f'family unmoved; a missing seed in {victim_cell} refused as an '
             f'incomplete arm; a doubled arm refused; an empty co-primary '
             f'endpoint refused as empty')


@case('DESIGN.md §4.3 -- an invalid source is excluded from the primary '
      'estimand, and the pooled estimand says so in words')
def test_source_validity_gate_applied(ctx: Ctx) -> None:
    """The gate, the analysis set it selects, and the label the secondary carries.

    The primary estimand of `DESIGN.md` §4.3 is "valid sources only"; pooling
    over source competence is the pre-declared SECONDARY. Both stats-reading
    cases used to invoke `stats.py` with `--source-policy pooled` alone, so the
    gate, the exclusion and the reporting of rejected sources were never
    exercised. In P0 `src-dueling-vanilla` scored 0.599 against a 0.600 gate,
    which is precisely the arm this path decides the fate of.
    """
    from experiments import registry

    frame = _select_rows(ctx)
    gate = float(registry.SOURCE_VALIDITY_GATE)
    req('source_valid' in frame.columns
        and 'source_final_score' in frame.columns,
        'the recorded per_seed.csv carries no source_valid/source_final_score '
        'columns, so whether the gate was applied at all cannot be read from '
        'the data (aggregate.py writes them).')

    # -- 1. the recorded verdict agrees with the gate, row by row. A verdict
    #       recorded under one threshold and analysed under another is how an
    #       invalid source becomes a valid one without anyone editing a number.
    scored = frame[frame['source_final_score'].notna()]
    disagreements = []
    for _index, row in scored.iterrows():
        score = float(row['source_final_score'])
        recorded = row['source_valid']
        if recorded is None or (isinstance(recorded, float)
                                and recorded != recorded):
            continue
        valid = str(recorded).strip().lower() in ('true', '1', '1.0', 'yes')
        if valid != (score >= gate):
            disagreements.append(
                f'{row.get("label")} s{row.get("seed")}: score {score:.4f}, '
                f'recorded valid={recorded!r}, gate {gate}')
    req(not disagreements,
        f'{len(disagreements)} recorded source-validity verdict(s) disagree '
        f'with registry.SOURCE_VALIDITY_GATE={gate}: {disagreements[:5]}. The '
        f'gate is what separates "transfer from a competent source" from '
        f'"transfer from a source scoring 26.94 out of 475", which is what the '
        f'published study did without anything noticing.')

    args_valid = ['--experiments', 'E1', '--source-policy', 'valid']
    args_pooled = ['--experiments', 'E1', '--source-policy', 'pooled']
    per_seed = _filtered_per_seed(ctx)
    code, valid_text, valid_report = _run_stats(ctx, per_seed, args_valid)
    same(code, 0, f'stats.py exited {code} under the PRIMARY source policy; '
                  f'tail:\n{valid_text[-1500:]}')
    same(str(valid_report['invocation'].get('source_policy')), 'valid',
         'the report does not record which analysis set produced it, so a '
         'primary and a secondary estimand are indistinguishable afterwards')

    # -- 2. the inventory reports the exclusions rather than performing them
    #       silently (DESIGN.md §4.3: "exclusions reported").
    inventory = valid_report['s2_inventory'].get('source_validity')
    req(inventory,
        'the report carries no source-validity inventory, so which runs the '
        'gate removed is not recoverable from the output')
    invalid_rows = [r for r in inventory if int(r.get('invalid') or 0) > 0]

    # -- 3. an arm whose sources are ALL invalid may not be tested under the
    #       primary policy, and its suppression must name the analysis set.
    tested_valid = {(m['metric'], m['cell']) for m in
                    _tested(_members(valid_report))}
    invalid_only_cells = {str(r['cell']) for r in invalid_rows
                          if int(r.get('valid') or 0) == 0}
    leaked = sorted({cell for _metric, cell in tested_valid
                     if cell in invalid_only_cells})
    req(not leaked,
        f'under --source-policy valid the confirmatory family tested {leaked}, '
        f'whose transfer runs ALL failed the {gate} gate. That is the primary '
        f'estimand reporting transfer from a source the design declared '
        f'invalid.')

    # -- 4. the secondary estimand is labelled in words, not only in a column.
    code, pooled_text, pooled_report = _run_stats(ctx, per_seed, args_pooled)
    same(code, 0, f'stats.py exited {code} under the pooled policy')
    same(str(pooled_report['invocation'].get('source_policy')), 'pooled',
         'the pooled invocation does not record its own analysis set')
    req('SECONDARY' in pooled_text,
        'the pooled-source invocation does not print the word SECONDARY. '
        'DESIGN.md §4.3 pre-declares pooling over source competence as a '
        'secondary estimand that must be "labelled exactly that", and never '
        'called an intention-to-treat analysis.')
    ctx.note(f'gate {gate} agrees with {len(scored)} recorded verdicts; '
             f'{len(invalid_rows)} arm(s) carry an invalid source; '
             f'{len(invalid_only_cells)} cell(s) excluded from the primary '
             f'estimand; pooled output labelled SECONDARY')


@case('DESIGN.md §9 -- the three control contrasts are emitted with every '
      'delta, and censored data is summarised rather than imputed or dropped')
def test_stats_emits_controls_and_censoring(ctx: Ctx) -> None:
    """C0-C3b are named for every cell, and P(reached) carries an exact interval.

    Two §9 rows that nothing exercised. "Effect attributed to learned
    representations without excluding mechanics" is guarded by C2/C3 being
    Tier 1 and emitted with every delta, so what has to hold is that every cell
    ACCOUNTS for every control: present or missing, never omitted. "Censored
    data imputed or dropped" is guarded by P(reached) with an exact interval,
    so what has to hold is that a threshold nobody reached is reported as a
    proportion of zero with a Clopper-Pearson interval, not as a blank or as an
    imputed time.
    """
    from experiments import stats
    from experiments import statlib as sl

    _per_seed, report, _text = _baseline_report(ctx)

    declared = {'C0', 'C1', 'C2', 'C2K0', 'C3', 'C3b'}
    controls = report['s7_controls']
    req(controls, 'the report carries no control section at all')
    for metric, block in controls.items():
        cells = block.get('cells') or {}
        req(cells, f'{metric}: the control section names no cell')
        for cell, entry in cells.items():
            accounted = set(entry.get('present') or []) | set(
                entry.get('missing') or [])
            unaccounted = sorted(declared - accounted)
            req(not unaccounted,
                f'{metric}/{cell}: the controls {unaccounted} appear in '
                f'neither the present nor the missing list. A control that is '
                f'simply absent from the output cannot be distinguished from '
                f'one that was run and showed nothing, which is how an effect '
                f'gets attributed to learned representations without the '
                f'mechanical rivals having been excluded (DESIGN.md §4.1).')
            # Either the contrasts are emitted, or the cell states why it
            # emitted none. A cell that carries neither has dropped the
            # controls silently, which is the row of DESIGN.md 9 this asserts.
            req(entry.get('contrasts') is not None
                or entry.get('suppressed'),
                f'{metric}/{cell}: no contrast list and no suppression '
                f'reason. The three contrasts are emitted with every delta, so '
                f'their absence has to be a stated refusal rather than a gap: '
                f'a reader cannot otherwise tell a control that was not run '
                f'from one that was run and showed nothing.')

    censored = (report.get('s9_estimation') or {}).get('censored') or {}
    req(censored,
        'the report carries no censored section. ANALYSIS_PLAN.md §5 makes '
        'time-to-threshold a survival quantity precisely so that a run which '
        'never reached the threshold is a censored observation rather than a '
        'missing one; dropping it conditions the estimate on success.')
    # The declared threshold levels, named by the plan rather than by whatever
    # keys the section happens to carry: `_delayed_entry` is bookkeeping about
    # the freeze offset, not a level, and a level that vanished from the output
    # must fail here rather than shrink the loop.
    declared_levels = [name for name, _q in stats.THRESHOLD_LEVELS]
    missing_levels = [name for name in declared_levels if name not in censored]
    req(not missing_levels,
        f'the censored section omits the pre-declared level(s) '
        f'{missing_levels}. ANALYSIS_PLAN.md 5 fixes the thresholds in advance '
        f'precisely so that a level at which nothing was reached is still '
        f'reported.')
    levels = 0
    for level in declared_levels:
        block = censored[level]
        levels += 1
        arms = block.get('arms') or []
        req(arms, f'censored level {level}: no arms reported')
        for arm in arms:
            for key in ('n', 'reached', 'p_reached', 'cp_lo', 'cp_hi'):
                req(key in arm,
                    f'censored level {level}, {arm.get("label")}: no {key!r}. '
                    f'P(reached) with an exact interval IS the guard; without '
                    f'the interval a proportion of 0/3 is indistinguishable '
                    f'from a proportion of 0/300.')
            n, reached = int(arm['n']), int(arm['reached'])
            if arm['p_reached'] is None:
                # Withheld below the inference floor. That is 9 holding, but
                # it has to SAY so: a blank proportion with no reason beside it
                # is indistinguishable from a proportion of zero.
                req(arm.get('note'),
                    f'censored level {level}, {arm.get("label")}: no '
                    f'proportion and no reason for its absence, so a reader '
                    f'cannot tell a withheld summary from "nothing reached".')
                req(arm.get('cp_lo') is None and arm.get('cp_hi') is None,
                    f'censored level {level}, {arm.get("label")}: an interval '
                    f'was emitted without the proportion it is an interval '
                    f'for; 9 withholds both together.')
            elif n:
                near(float(arm['p_reached']), reached / n, 1e-9,
                     f'censored level {level}, {arm.get("label")}: p_reached '
                     f'is not reached/n, so it is not the proportion its name '
                     f'claims')
                lo, hi = sl.clopper_pearson(reached, n)
                near(float(arm['cp_lo']), float(lo), 1e-9,
                     f'censored level {level}, {arm.get("label")}: the lower '
                     f'interval bound is not the exact Clopper-Pearson one')
                near(float(arm['cp_hi']), float(hi), 1e-9,
                     f'censored level {level}, {arm.get("label")}: the upper '
                     f'interval bound is not the exact Clopper-Pearson one')
            req(int(arm.get('unknown_censoring') or 0) == 0
                or arm.get('note'),
                f'censored level {level}, {arm.get("label")}: runs of unknown '
                f'censoring status are counted but not explained')
    n_cells = sum(len(b.get("cells") or {}) for b in controls.values())
    ctx.note(f'{len(controls)} metrics x {n_cells} cells account for all '
             f'of {sorted(declared)}; {levels} censoring levels carry '
             f'exact intervals')


@case('DESIGN.md §3.1 -- a cross-architecture contrast at mismatched '
      'transferred fraction is refused, not annotated', slow=True)
def test_stats_intensity_gate(ctx: Ctx) -> None:
    """The arch contrast is confounded with treatment intensity unless matched.

    `DESIGN.md` §9 lists "treatment intensity mistaken for architecture" and
    §3.1 makes the matched-intensity contrast primary. Nothing asserted that
    the gate fires: the tolerance and the verdict were emitted, and a value
    outside the tolerance had never been put through them.
    """
    from experiments import stats

    frame = _select_rows(ctx)
    req('transferred_param_fraction' in frame.columns,
        'the recorded per_seed.csv carries no transferred_param_fraction, so '
        'the DESIGN.md §3.1 matching cannot be checked at all')
    args = ['--experiments', 'E1', '--source-policy', 'pooled']

    # -- the clean case: the recorded fractions are matched, and the gate says
    #    so. Without this the case could not tell a working gate from one that
    #    refuses everything.
    _per_seed, baseline, _text = _baseline_report(ctx)
    gate_rows = baseline['s2_inventory'].get('intensity_gate') or []
    req(gate_rows,
        'no cross-architecture intensity gate in the report, so the §3.1 '
        'matching is not being applied to anything')
    for row in gate_rows:
        near(float(row['abs_diff']),
             abs(float(row['frac_a']) - float(row['frac_b'])), 1e-12,
             f'{row["a"]} vs {row["b"]}: abs_diff is not |frac_a - frac_b|')
        expected = (float(row['abs_diff']) <= stats.INTENSITY_TOLERANCE)
        same(bool(int(row['permitted'])), expected,
             f'{row["a"]} vs {row["b"]}: the gate permitted={row["permitted"]} '
             f'at abs_diff {row["abs_diff"]} against tolerance '
             f'{stats.INTENSITY_TOLERANCE}. The verdict must be the tolerance '
             f'applied to the number, not a label attached beside it.')

    # -- the violation: one architecture's transferred fraction is moved well
    #    outside the tolerance, and every cross-architecture contrast must then
    #    be REFUSED rather than printed with a caveat.
    mlp = frame['cell'].astype(str).str.startswith('mlp')
    moved = frame.copy()
    moved.loc[mlp & (moved['condition'] != 'scratch'),
              'transferred_param_fraction'] = 0.40
    req(not moved.equals(frame),
        'the perturbation changed nothing, so the gate is not being exercised')
    code, text, report = _run_stats(ctx, _write_per_seed(ctx, moved), args)
    same(code, 0, f'stats.py exited {code} on the mismatched-intensity '
                  f'selection; tail:\n{text[-1200:]}')
    rows = report['s2_inventory'].get('intensity_gate') or []
    req(rows and all(int(r['permitted']) == 0 for r in rows),
        f'the intensity gate permitted a cross-architecture contrast at a '
        f'transferred-fraction difference far outside '
        f'{stats.INTENSITY_TOLERANCE}: {rows[:3]}. The arch contrast would '
        f'then be measuring how much was transferred rather than which '
        f'architecture received it -- the same class of error the Phase 0 '
        f'audit found in the published study.')
    req('REFUSAL' in text.upper() or 'REFUSED' in text.upper(),
        'the confounded cross-architecture contrasts were not announced as '
        'refused; DESIGN.md §3.1 refuses them rather than annotating them')
    ctx.note(f'{len(gate_rows)} matched pair(s) permitted at tolerance '
             f'{stats.INTENSITY_TOLERANCE}; {len(rows)} pair(s) refused when '
             f'the fraction was moved outside it')

# ===========================================================================
# 7a. Wording: the guards that stand between a number and a sentence
# ===========================================================================
#: (label, sentence, kind, evidence-override, guard id expected to refuse).
#: The evidence is a healthy within-cell delta unless the row overrides it, so
#: each row changes exactly one thing and the refusal it draws names the guard
#: that caught it. `report.claim` returns the guard id in its refusals, which is
#: what makes this a test of a specific guard rather than of "something was
#: refused".
_REFUSED_SENTENCES: tuple[tuple[str, str, str, dict, str], ...] = (
    ('affirming a null',
     'Transfer makes no difference to the final score, 0.02 [-0.10, 0.14] '
     'over n=10 seeds.',
     'causal', {'ci_lo': -0.10, 'ci_hi': 0.14}, 'null-affirming'),
    ('two verdicts compared',
     'mlp-double avoids the drop while dueling-double does not, 0.18 '
     '[0.05, 0.31] over n=10 seeds.',
     'causal', {}, 'verdict-comparison'),
    ('mechanism from prose',
     'The trunk representation is what carries the gain, 0.18 [0.05, 0.31] '
     'over n=10 seeds.',
     'mechanism', {'estimand': 'mechanism_signal'}, 'mechanism-signal'),
    ('p-value outside the family',
     'The delta is 0.18 [0.05, 0.31] with p = 0.03 over n=10 seeds.',
     'causal', {}, 'p-value'),
    ('raw returns across scales',
     'The paired delta is 18.4 [5.0, 31.0] over n=10 seeds.',
     'causal', {'scale': 'raw_return'}, 'raw-return'),
    ('a phrase the published paper printed',
     'We observe positive transfer, 0.18 [0.05, 0.31] over n=10 seeds.',
     'causal', {}, 'forbidden-phrase'),
    ('cross-architecture return read as an effect',
     'The dueling architecture causes the higher scratch score, 0.18 '
     '[0.05, 0.31] over n=10 seeds.',
     'associational', {'estimand': 'between_cell_scratch'}, 'causal-verb'),
    ('below the inference floor',
     'The paired delta is 0.18 [0.05, 0.31] over n=2 seeds.',
     'causal', {'n': 2}, 'n<3'),
    ('drawn on the selection block',
     'The paired delta is 0.18 [0.05, 0.31] over n=10 seeds.',
     'causal', {'seed_block': 'TUNE'}, 'TUNE'),
    ('dispersion adjective against the SDs',
     'The transfer arm is narrower than the scratch arm, SD ratio 1.80 '
     '[1.2, 2.5] over n=10 seeds.',
     'dispersion', {'estimand': 'dispersion', 'sd_a': 0.36, 'sd_b': 0.20,
                    'ci_lo': 1.2, 'ci_hi': 2.5}, 'dispersion-direction'),
    ('direction verb against theta',
     'The transfer arm exceeds the scratch arm, theta = 0.33 [0.2, 0.45] '
     'over n=10 seeds.',
     'associational', {'estimand': 'between_cell_scratch', 'theta': 0.33,
                       'ci_lo': 0.2, 'ci_hi': 0.45},
     'relative-effect-direction'),

    # -- the rows below close the coverage gap the twelve above left. Eleven of
    #    the twenty-eight guards were exercised, and NONE of the equivalence or
    #    exclusion family: the four guards the equivalence rewrite added were
    #    checked only by `report.py --self-test`, written by the same agent as
    #    the code, which is how a DIFFERENT verdict refused at high dispersion
    #    and an exclusion bound published off a zero-width interval both
    #    survived a green suite and a "0 failure(s)" self-test. The case now
    #    asserts that every id in `report.GUARD_CATALOGUE` has a row here, so a
    #    guard cannot be added, renamed or lost without this file saying so.
    ('an unknown claim kind',
     'The paired delta is 0.18 [0.05, 0.31] over n=10 seeds.',
     'speculation', {}, 'kind'),
    ('an estimand outside the design research-question table',
     'The paired delta is 0.18 [0.05, 0.31] over n=10 seeds.',
     'causal', {'estimand': 'vibes'}, 'estimand'),
    ('wording that performs a different inference from the licensed one',
     'The paired delta is 0.18 [0.05, 0.31] over n=10 seeds.',
     'associational', {}, 'inference-binding'),
    ('an exclusion bound in score units over a gradient-norm signal',
     'A degradation worse than 0.05 score units is excluded at 95%.',
     'exclusion', {'estimand': 'mechanism_signal',
                   'signal': 'grad_norm_trunk'}, 'interval-kind-estimand'),
    ('a causal word over a component contrast',
     'The paired delta is 0.18 [0.05, 0.31] over n=10 seeds.',
     'causal', {'estimand': 'control_contrast'}, 'causal-estimand'),
    ('a component claim naming no manipulation',
     'The paired delta is 0.18 [0.05, 0.31] over n=10 seeds.',
     'causal_component', {'estimand': 'control_contrast'},
     'component-manipulation'),
    ('an equivalence verdict that is really a p-value',
     'The two arms are not distinguishable within the margin.',
     'equivalence', {'verdict': 'p > 0.05', 'ci_lo': -0.02, 'ci_hi': 0.01,
                     'sd_a': 0.031, 'sd_b': 0.042}, 'equivalence-verdict'),
    ('equivalence asserted where the verdict is DIFFERENT',
     'The two arms are equivalent within the pre-registered margin.',
     'equivalence', {'verdict': 'DIFFERENT', 'ci_lo': -0.60, 'ci_hi': -0.40,
                     'sd_a': 0.021, 'sd_b': 0.032}, 'equivalence-assertion'),
    ('a verdict the interval does not contain',
     'The two arms are equivalent within the pre-registered margin.',
     'equivalence', {'verdict': 'EQUIVALENT', 'ci_lo': -0.30, 'ci_hi': -0.10,
                     'sd_a': 0.021, 'sd_b': 0.032},
     'equivalence-containment'),
    ('equivalence in a cell whose dispersion makes it untestable',
     'The two arms are equivalent within the pre-registered margin.',
     'equivalence', {'verdict': 'EQUIVALENT', 'ci_lo': -0.02, 'ci_hi': 0.01,
                     'sd_a': 0.093, 'sd_b': 0.369},
     'equivalence-feasibility'),
    ('equivalence where the dispersion could not be estimated at all',
     'The two arms are equivalent within the pre-registered margin.',
     'equivalence', {'verdict': 'EQUIVALENT', 'ci_lo': -0.02, 'ci_hi': 0.01,
                     'sd_a': None, 'sd_b': float('nan')},
     'equivalence-feasibility'),
    ('a zero-width interval read as decisive',
     'The paired delta is 0.18 [0.18, 0.18] over n=10 seeds.',
     'causal', {'ci_lo': 0.18, 'ci_hi': 0.18}, 'degenerate-interval'),
    ('an exclusion bound the computing module withheld',
     'Every degradation is excluded at 95%.',
     'exclusion', {'exclusion_bound': None}, 'exclusion-bound'),
    ('equivalence read off a p-value rather than off containment',
     'The two arms are not distinguishable within the margin.',
     'equivalence', {'verdict': 'EQUIVALENT', 'ci_lo': -0.02, 'ci_hi': 0.01,
                     'sd_a': 0.031, 'sd_b': 0.042, 'basis': 'p_value'},
     'equivalence-basis'),
    ('a dispersion sentence with no dispersions behind it',
     'The transfer arm is wider than the scratch arm, SD ratio 1.80 '
     '[1.2, 2.5] over n=10 seeds.',
     'dispersion', {'estimand': 'dispersion', 'ci_lo': 1.2, 'ci_hi': 2.5},
     'dispersion-evidence'),
    ('a direction with no interval to read it from',
     'The paired delta is 0.18 over n=10 seeds.',
     'causal', {'ci_lo': None, 'ci_hi': None}, 'interval'),
    ('no n at all',
     'The paired delta is 0.18 [0.05, 0.31] over n=10 seeds.',
     'causal', {'n': None}, 'n-missing'),
    ('an n that is not a count, so the floor cannot be applied',
     'The paired delta is 0.18 [0.05, 0.31] over n=10 seeds.',
     'causal', {'n': float('nan')}, 'n-missing'),
    ('nothing stated that would refute it',
     'The paired delta is 0.18 [0.05, 0.31] over n=10 seeds.',
     'causal', {'refuted_by': None}, 'refutability'),
)

#: The same shapes, licensed. Without these the case could not tell a working
#: guard from one that refuses every sentence put to it.
_ACCEPTED_SENTENCES: tuple[tuple[str, str, str, dict], ...] = (
    ('the plain within-cell delta',
     'The paired delta is 0.18 [0.05, 0.31] over n=10 seeds.', 'causal', {}),
    ('two verdicts compared, with the contrast supplied',
     'mlp-double avoids the drop while dueling-double does not, 0.18 '
     '[0.05, 0.31] over n=10 seeds.',
     'causal', {'between_cell_contrast': {'delta': 0.12, 'ci_lo': 0.02,
                                          'ci_hi': 0.22}}),
    ('a mechanism claim naming an instrumented signal',
     'The trunk gradient norm falls by 0.18 [0.05, 0.31] over n=10 seeds.',
     'mechanism', {'estimand': 'mechanism_signal',
                   'signal': 'grad_norm_trunk'}),
    ('a p-value inside the confirmatory family',
     'The delta is 0.18 [0.05, 0.31] with p = 0.03 over n=10 seeds.',
     'causal', {'confirmatory': True, 'p_holm': 0.03}),
    ('the dispersion adjective the SDs support',
     'The transfer arm is wider than the scratch arm, SD ratio 1.80 '
     '[1.2, 2.5] over n=10 seeds.',
     'dispersion', {'estimand': 'dispersion', 'sd_a': 0.36, 'sd_b': 0.20,
                    'ci_lo': 1.2, 'ci_hi': 2.5}),
    ('the direction verb theta supports',
     'A random run of the transfer arm scores below a random run of the '
     'scratch arm, theta = 0.33 [0.2, 0.45] over n=10 seeds.',
     'associational', {'estimand': 'between_cell_scratch', 'theta': 0.33,
                       'ci_lo': 0.2, 'ci_hi': 0.45}),
    ('the licensed form of a null result',
     'The paired delta is 0.02 [-0.10, 0.14] over n=10 seeds, which is not '
     'distinguishable from zero at this n.',
     'causal', {'ci_lo': -0.10, 'ci_hi': 0.14}),

    # -- the licensed halves of the equivalence family. Without these the rows
    #    above would be satisfied by a gate that refuses every equivalence
    #    sentence put to it, which is what the dispersion gate did to DIFFERENT.
    ('an equivalence verdict the interval and the SDs both support',
     'The two arms are equivalent within the pre-registered margin.',
     'equivalence', {'verdict': 'EQUIVALENT', 'ci_lo': -0.02, 'ci_hi': 0.01,
                     'sd_a': 0.031, 'sd_b': 0.042}),
    ('a DIFFERENT verdict in a cell far too noisy for an equivalence claim',
     'The two arms are not distinguishable within the margin.',
     'equivalence', {'verdict': 'DIFFERENT', 'ci_lo': -0.60, 'ci_hi': -0.40,
                     'sd_a': 0.093, 'sd_b': 0.369}),
    ('an UNTESTABLE verdict, which is the statement that there is no interval',
     'The dispersion in this cell makes equivalence untestable at this n.',
     'equivalence', {'verdict': 'UNTESTABLE', 'ci_lo': None, 'ci_hi': None,
                     'sd_a': 0.093, 'sd_b': 0.369}),
    ('the exclusion bound the computing module published',
     'Every degradation is excluded at 95%.',
     'exclusion', {'exclusion_bound': 0.05}),
    ('a component claim naming what was manipulated',
     'The paired delta is 0.18 [0.05, 0.31] over n=10 seeds.',
     'causal_component', {'estimand': 'control_contrast',
                          'manipulated': 'the copied trunk weights',
                          'refuted_by': 'a C2 contrast of the same size'}),
    ('a descriptive arm summary',
     'The scratch arm reaches 1.15 [1.05, 1.25] over n=10 seeds.',
     'descriptive', {'estimand': 'arm_descriptive', 'ci_lo': 1.05,
                     'ci_hi': 1.25}),
)


@case('DESIGN.md §9 -- the wording guards: a null is not affirmed, two '
      'verdicts are not a comparison, a mechanism is not prose, a direction '
      'word is generated, and the scope clause is inherited')
def test_report_wording_guards_fire(ctx: Ctx) -> None:
    """Five §9 rows live in `report.claim`, and none of them was exercised.

    `validate.py` used to import `stats.py` and `statlib.py` and nothing else,
    so every §9 row whose guard is a wording rule was uncovered while the module
    docstring claimed each row had a test that fails when its guard is removed.
    The table above puts one sentence through each guard and requires the
    refusal to name that guard by id, and the second table puts the same shapes
    through with the evidence that licenses them and requires acceptance -- so
    a guard that refuses everything fails here too.
    """
    from experiments import report

    healthy = dict(
        estimand='within_cell_delta', n=10, ci_lo=0.05, ci_hi=0.31,
        counterfactual='the scratch arm at the same seed',
        rivals='initialisation scale and the optimiser reset',
        excluded_by='C2 and C3',
        refuted_by='an interval containing zero')

    for label, text, kind, override, guard in _REFUSED_SENTENCES:
        evidence = dict(healthy)
        evidence.update(override)
        claim = report.claim(text, kind, evidence)
        req(not claim.accepted,
            f'{label}: report.claim ACCEPTED "{text}". The {guard!r} guard of '
            f'DESIGN.md §9 did not fire, so the sentence would be emitted.')
        fired = [r[0] for r in claim.refusals]
        req(guard in fired,
            f'{label}: the sentence was refused, but by {fired} rather than by '
            f'{guard!r}. A refusal for the wrong reason does not show that '
            f'this guard holds, and it is the guard that would have to be '
            f'removed for the sentence to be printed.')

    for label, text, kind, override in _ACCEPTED_SENTENCES:
        evidence = dict(healthy)
        evidence.update(override)
        claim = report.claim(text, kind, evidence)
        req(claim.accepted,
            f'{label}: report.claim REFUSED a licensed sentence for '
            f'{[r[0] for r in claim.refusals]}. A guard that refuses the form '
            f'it exists to permit is not discriminating, and the refused rows '
            f'above would then prove nothing.')

    # -- the scope clause of DESIGN.md §2.1 is inherited by every emitted
    #    claim, not left to whoever quotes it.
    plain = report.claim(_ACCEPTED_SENTENCES[0][1], 'causal', dict(healthy))
    req(report.SCOPE_CLAUSE.rstrip(':') in plain.text,
        f'an accepted causal claim does not carry the DESIGN.md §2.1 scope '
        f'clause. "Generalising past the evidence" is a §9 row, and the clause '
        f'is inherited by the sentence because a sentence lifted out of the '
        f'bundle carries only its own words. Got: {plain.text[:160]!r}')

    # -- direction words are GENERATED from the numbers. The published paper
    #    wrote "exceeds" over theta=0.469 and theta=0.333, asserting the
    #    opposite of the numbers in the same sentence that printed them.
    above = report.phrase_relative_effect('A', 'B', 0.72, 0.60, 0.85)
    below = report.phrase_relative_effect('A', 'B', 0.28, 0.15, 0.40)
    covering = report.phrase_relative_effect('A', 'B', 0.52, 0.35, 0.70)
    req('above' in above and 'below' not in above,
        f'the generated sentence for theta=0.72 does not read as above the '
        f'0.5 null: {above!r}')
    req('below' in below and 'above' not in below,
        f'the generated sentence for theta=0.28 does not read as below the '
        f'0.5 null: {below!r}')
    req('not distinguishable' in covering,
        f'an interval covering the 0.5 null is not reported as not '
        f'distinguishable: {covering!r}. Reading a Brunner-Munzel interval '
        f'against zero instead of against 0.5 prints "excludes zero" about a '
        f'quantity that cannot be zero.')

    # -- and the guard catalogue may not advertise a guard that no longer
    #    exists, because §11a of the report is rendered from it.
    catalogue = {row[0] for row in report.GUARD_CATALOGUE}
    exercised = {guard for _l, _t, _k, _o, guard in _REFUSED_SENTENCES}
    missing = sorted(exercised - catalogue)
    req(not missing,
        f'{missing} are guard ids this case relies on and the report catalogue '
        f'no longer lists, so the report would advertise a different set of '
        f'guards from the ones that fire.')
    # And the other direction, which is the one that let four new guards ship
    # unexercised: a guard the report advertises in §11a and this case never
    # provokes is a guard whose loss nothing here would notice.
    unexercised = sorted(catalogue - exercised)
    req(not unexercised,
        f'{unexercised} are advertised in report.GUARD_CATALOGUE, and rendered '
        f'into §11a of every REPORT.md as guards that fire, and no row of '
        f'_REFUSED_SENTENCES provokes them. A guard nothing exercises is a '
        f'guard whose removal nothing detects: add a row that draws each one, '
        f'or take it out of the catalogue.')
    ctx.note(f'{len(_REFUSED_SENTENCES)} sentences refused by the named guard, '
             f'{len(_ACCEPTED_SENTENCES)} licensed shapes accepted, scope '
             f'clause inherited, all {len(catalogue)} catalogued guards '
             f'exercised')


# ===========================================================================
# 7b. Regressions. Four defects adversarial review found and other modules
#     fixed, each pinned here so it cannot come back. A fix with no case behind
#     it lasts until the next rewrite of the module it lives in.
# ===========================================================================
@case('DESIGN.md §8.4, ANALYSIS_PLAN.md §2 -- one arm x seed recorded twice is '
      'refused, and moves no number it did not touch', slow=True)
def test_duplicate_unit_moves_no_number(ctx: Ctx) -> None:
    """A duplicated unit is refused by name, and nothing else shifts.

    Two rows for one (arm, seed) are not two draws. Averaging them deflates the
    across-seed SD by sqrt(k(m-1)/(km-1)) -- `aggregate.duplicate_arm_seeds`
    measures 0.4901 -> 0.4384 on `runs_demo` -- and `ANALYSIS_PLAN.md` §4 gates
    the equivalence claim on that SD sitting below 0.05. Resolving the pair by
    keeping one row is no better: which row survives is a coin toss, and the
    arm that got analysed is then not the arm the plan declares.

    Two properties, and the second is the one nothing checked. The affected cell
    must be REFUSED with a reason that names the duplicated seed, and every
    OTHER cell must come back byte-identical: a duplicate in one arm that
    perturbs another arm's numbers has reached a denominator it has no business
    in.
    """
    import pandas as pd

    from experiments import aggregate, registry, stats

    actx, origin = _analysis_ctx(ctx)
    frame = _select_rows(actx)
    args = ['--experiments', 'E1', '--source-policy', 'pooled']
    _path, baseline, _text = _baseline_report(actx)
    base = {(m['metric'], m['cell']): m for m in _members(baseline)}
    req(_tested(_members(baseline)),
        f'no confirmatory member was tested against {origin}, so a moved '
        f'number could not be told apart from a suppressed one.')

    same(aggregate.duplicate_arm_seeds(frame), [],
         'the selection handed to this case already carries a duplicated '
         'arm x seed, so the duplicate injected below could not be attributed')

    # The victim has to be a row the confirmatory sample actually contains.
    # Taking the first scratch row of the sorted frame picked
    # `src-dueling-double` on CartPole, a SOURCE arm that stats.py excludes
    # from target-side estimation, so duplicating it moved nothing and this
    # case failed on its own choice of victim rather than on the guard.
    scratch = frame[(frame['condition'] == 'scratch')
                    & (frame['env'] == registry.TARGET_ENV)
                    & (~frame['seed_block'].isin(stats.SOURCE_ONLY_BLOCKS))
                    & (frame['seed'].isin(_target_seeds(actx, frame)))]
    req(len(scratch),
        f'the selection holds no target-side scratch row on '
        f'{registry.TARGET_ENV} outside the donor blocks '
        f'{stats.SOURCE_ONLY_BLOCKS}, so no row the confirmatory sample uses '
        f'can be duplicated')
    victim = scratch.iloc[[0]]
    victim_arm = str(victim['arm'].iloc[0])
    victim_cell = str(victim['cell'].iloc[0])
    victim_seed = int(victim['seed'].iloc[0])
    doubled = pd.concat([frame, victim], ignore_index=True)

    # The aggregation layer has to NAME it. validate.py never imported
    # aggregate.py at all, so its copy of every constant and every detector was
    # uncovered in both directions: a mutated threshold level or a deleted
    # duplicate detector changed nothing this suite reported.
    same(aggregate.duplicate_arm_seeds(doubled),
         [(victim_arm, victim_seed, 2)],
         f'aggregate.duplicate_arm_seeds did not report {victim_arm} at seed '
         f'{victim_seed} as carrying two runs. A duplicate nothing names is a '
         f'duplicate that gets averaged.')

    code, _out, got = _run_stats(actx, _write_per_seed(actx, doubled), args)
    same(code, 0, f'stats.py exited {code} on the doubled-unit selection')

    hit = [m for m in _members(got) if m['cell'] == victim_cell]
    req(hit, f'no confirmatory member for cell {victim_cell} to inspect')
    for member in hit:
        req(member.get('p_signflip') is None and member.get('hl') is None,
            f'{member["metric"]}/{victim_cell}: a test and a point estimate '
            f'were emitted over an arm carrying two rows for seed '
            f'{victim_seed}. The pairing of ANALYSIS_PLAN.md §2 is by seed, so '
            f'a doubled row is either a silent average or a duplicated unit, '
            f'and both misstate n.')
        reason = str(member.get('suppressed') or '')
        req(str(victim_seed) in reason and 'more than one row' in reason,
            f'{member["metric"]}/{victim_cell} was suppressed for {reason!r}, '
            f'which does not say that seed {victim_seed} carries more than one '
            f'row. A reader cannot then tell a duplicated unit from a small '
            f'sample.')

    # Every ESTIMATE of every other cell must be byte-identical. `p_holm` is
    # excluded from this list and pinned separately below, because it is not a
    # property of its own cell: Holm is a family-wide step-down, so losing one
    # member of the family of 8 legitimately moves the adjusted p of the other
    # seven. That is not an exemption -- the moved value is recomputed from the
    # surviving raw p-values two paragraphs down and required to match exactly,
    # so nothing about it is taken on trust.
    moved: list[str] = []
    for member in _members(got):
        key = (member['metric'], member['cell'])
        if key[1] == victim_cell:
            continue
        before = base.get(key)
        req(before is not None,
            f'{key} appears in the doubled report and not in the baseline, so '
            f'the two are not comparable')
        for name in ('n', 'p_signflip', 'hl', 'ci_lo', 'ci_hi', 'mean_delta',
                     'suppressed'):
            if repr(before.get(name)) != repr(member.get(name)):
                moved.append(f'{key[0]}/{key[1]}.{name}: '
                             f'{before.get(name)!r} -> {member.get(name)!r}')
    req(not moved,
        f'duplicating one row of {victim_arm} at seed {victim_seed} moved '
        f'{len(moved)} estimate(s) in OTHER cells: ' + '; '.join(moved[:6])
        + '. A duplicate must be refused where it is, and reach no arm it is '
          'not in.')

    # The multiplicity correction, pinned rather than exempted. `holm_adjust`
    # takes its family size from the PLAN, so a suppressed member does not
    # shrink the family: the survivors are adjusted as members of a family of
    # CONFIRMATORY_FAMILY_SIZE, which moves each of them in the conservative
    # direction and no further.
    raw = {(m['metric'], m['cell']): m.get('p_signflip')
           for m in _members(got)}
    expected = stats.holm_adjust(raw, stats.CONFIRMATORY_FAMILY_SIZE)
    for member in _members(got):
        key = (member['metric'], member['cell'])
        got_holm, want = member.get('p_holm'), expected.get(key)
        if want is None:
            req(got_holm is None,
                f'{key[0]}/{key[1]} reports p_holm={got_holm!r} with no raw p '
                f'behind it')
            continue
        near(float(got_holm), float(want), 1e-12,
             f'{key[0]}/{key[1]}: the reported Holm-adjusted p is not the '
             f'step-down over the surviving raw p-values at the PRE-REGISTERED '
             f'family size of {stats.CONFIRMATORY_FAMILY_SIZE}. A family that '
             f'shrinks because a member was suppressed is the family-of-one '
             f'rescue ANALYSIS_PLAN.md 7 exists to block.')
        if key[1] == victim_cell:
            continue
        before = base.get(key, {}).get('p_holm')
        if before is not None:
            req(float(got_holm) >= float(before) - 1e-12,
                f'{key[0]}/{key[1]}: the Holm-adjusted p FELL from {before!r} '
                f'to {got_holm!r} when a member of the family was lost to a '
                f'duplicated run row. Losing a member may only make the '
                f'correction stricter; a duplicate elsewhere that makes this '
                f'cell easier to declare significant is a result rescued by '
                f'corrupting the input.')
    ctx.note(f'against {origin.split(":")[0]}: {victim_arm} seed '
             f'{victim_seed} doubled -> {victim_cell} refused by name, every '
             f'other estimate byte-identical, and every Holm-adjusted p '
             f're-derived at the pre-registered family size of '
             f'{stats.CONFIRMATORY_FAMILY_SIZE}')


@case('ANALYSIS_PLAN.md §5, DESIGN.md §5.3 -- a threshold crossing cannot be '
      'dated from an incomplete trailing window')
def test_threshold_crossing_needs_a_complete_window(ctx: Ctx) -> None:
    """The trailing-100 mean is the metric, and one evaluation is not it.

    `min_periods=1` made the "trailing-100-episode mean" at episode 0 the mean
    of a single monitoring evaluation, so one lucky evaluation of an untrained
    policy dated a crossing at 83 env steps against a table median of 38,435.
    Both affected runs were transfer arms and both were the highest-jumpstart
    runs in the tree, so the artefact pointed the same way as the claim the
    endpoint is evidence for.

    Checked two ways. A synthetic evaluation log with exactly that shape must
    not date the crossing at the spike; and the same log with
    `window_observations` forced back to one observation must, which is what
    shows the requirement is load-bearing rather than decorative. Then the
    recorded tree: an uncensored crossing dated before the first gradient update
    must be named on the row, because a level reached with no update was reached
    by the initial policy and not by learning.
    """
    import json

    import pandas as pd

    from experiments import aggregate

    # The confirmatory evaluation cadence, read from the runner's own
    # default rather than typed: the number of observations a complete
    # window holds is TRAILING_WINDOW // cadence, so a cadence typed here
    # would let the two drift.
    from src.dqn.config import Config as _Config

    cadence = int(_Config.__dataclass_fields__['eval_every'].default)
    episodes = 10 * aggregate.TRAILING_WINDOW
    spike = 83.0
    rows = []
    for episode in range(episodes):
        steps = spike if episode == 0 else spike + 250.0 * episode
        score = None
        if episode % cadence == 0:
            # One lucky evaluation at episode 0, nothing for a long while, then
            # a genuinely solved policy. The trailing mean crosses 0.25 only
            # once the window holds enough of the later evaluations.
            score = 0.95 if (episode == 0 or episode >= episodes // 2) else 0.0
        rows.append({'episode': episode, 'env_steps': steps,
                     'updates': max(0.0, steps - 1000.0),
                     'eval_score': score})
    log = pd.DataFrame(rows)
    total = float(log['env_steps'].max())

    required = aggregate.window_observations(cadence)
    same(required, aggregate.TRAILING_WINDOW // cadence,
         f'window_observations({cadence}) is not the number of evaluation '
         f'points a complete trailing-{aggregate.TRAILING_WINDOW}-episode '
         f'window holds. That count is the definition of the metric, not a '
         f'tolerance to be widened.')
    earliest = (log.set_index('episode')['env_steps']
                .get((required - 1) * cadence))

    got = aggregate.threshold_crossings(log, total, eval_every=cadence)
    for name, _level in aggregate.THRESHOLD_LEVELS:
        crossing = got.get(f'steps_to_threshold_{name}')
        if crossing is None or got.get(f'censored_{name}'):
            continue
        req(float(crossing) != spike,
            f'steps_to_threshold_{name} is {crossing}, the env-step count at '
            f'the single lucky evaluation. The trailing-'
            f'{aggregate.TRAILING_WINDOW}-episode mean at episode 0 has one '
            f'observation in it and is not the metric DESIGN.md §5.3 defines.')
        req(float(crossing) >= float(earliest),
            f'steps_to_threshold_{name} is {crossing}, before the '
            f'{earliest} env steps at which the window first holds its '
            f'{required} observations, so it was dated from a partial window.')

    # And the same log with the requirement removed. If this does NOT reproduce
    # the artefact then the check above is passing for some other reason and
    # proves nothing about the window.
    real = aggregate.window_observations
    try:
        aggregate.window_observations = lambda cadence: 1
        mutated = aggregate.threshold_crossings(log, total, eval_every=cadence)
    finally:
        aggregate.window_observations = real
    req(float(mutated.get('steps_to_threshold_p25') or 0.0) == spike,
        f'with the complete-window requirement removed the crossing came back '
        f'as {mutated.get("steps_to_threshold_p25")!r} rather than the '
        f'{spike} env steps of the single evaluation, so this case is not '
        f'exercising the requirement it claims to.')

    # The recorded tree. A crossing dated before the first gradient update is
    # kept rather than deleted (a genuine zero-shot result is a result), so what
    # is required is that the row SAYS SO.
    frame = _select_rows(ctx)
    unnamed: list[str] = []
    checked = 0
    for _index, row in frame.iterrows():
        run_dir = str(row.get('run_dir') or '')
        manifest = os.path.join(run_dir, 'manifest.json')
        if not os.path.exists(manifest):
            continue
        with open(manifest, encoding='utf-8') as fh:
            starts = (json.load(fh).get('config') or {}).get('learning_starts')
        if starts is None:
            continue
        caveats = str(row.get('derivation_caveats') or '')
        for name, _level in aggregate.THRESHOLD_LEVELS:
            value = pd.to_numeric(row.get(f'steps_to_threshold_{name}'),
                                  errors='coerce')
            if value != value or bool(row.get(f'censored_{name}')):
                continue
            checked += 1
            if float(value) < float(starts) and 'gradient update' not in caveats:
                unnamed.append(f'{row.get("label")}/s{row.get("seed")}/{name} '
                               f'= {value:g} < learning_starts {starts}')
    req(not unnamed,
        f'{len(unnamed)} recorded crossing(s) are dated before the first '
        f'gradient update and say nothing about it: ' + '; '.join(unnamed[:6])
        + '. aggregate.py records that case in derivation_caveats rather than '
          'deleting it, so a row that carries the number and not the caveat '
          'has lost the only thing that distinguishes a zero-shot result from '
          'a measurement artefact.')
    ctx.note(f'a single-evaluation spike does not date a crossing (window of '
             f'{required} observations at cadence {cadence}); removing the '
             f'requirement reproduces the {spike:g}-step artefact; '
             f'{checked} recorded crossing(s) at or after the first update')


@case('DESIGN.md §8.4, ANALYSIS_PLAN.md §10.1 -- an audit selection that '
      'matches no run refuses instead of passing')
def test_audit_refuses_a_selection_matching_nothing(ctx: Ctx) -> None:
    """An empty audit is not a passing audit.

    `--seeds ""` resolves to no seed at all, so every completeness assertion
    became true of the empty set and the audit printed a pass. Closing that left
    the quieter shape open: `--seeds 999` resolves to a perfectly good non-empty
    tuple naming arm x seed slots nobody ran, and `--seeds 5` -- an ordinary
    CONFIRM seed that simply was not run, i.e. a single-digit typo -- does the
    same. Measured on a tree brought current in every other respect: 25 errors
    at the declared blocks against 0 errors and a rendered "AUDIT PASS" at
    `--seeds 999`.

    The audit is the `ANALYSIS_PLAN.md` §10.1 gate `stats.py` exits 3 on, so a
    selection that empties it removes the gate rather than overriding it, and
    `DESIGN.md` §8.4 permits an override only when the override is stamped into
    the output.
    """
    from experiments import audit

    def errors(**kwargs) -> tuple[bool, set[str]]:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            ok, report = audit.audit(ctx.runs, **kwargs)
        return bool(ok), {str(finding.get('code'))
                          for check in report.get('checks', [])
                          for finding in check.get('findings', [])
                          if str(finding.get('level')) == 'error'}

    _declared_ok, declared = errors()
    selection_codes = {'seed_selection_empty', 'selection_matches_no_run',
                       'selection_scopes_out_runs', 'seed_spec_unparseable'}
    leaked = sorted(declared & selection_codes)
    req(not leaked,
        f'auditing at the DECLARED seed blocks already reports {leaked}, so '
        f'the refusals below would fire whatever the selection was and this '
        f'case could not tell a guard from a constant.')

    for spec, wanted in (('', 'seed_selection_empty'),
                         ('999', 'selection_matches_no_run'),
                         ('5', 'selection_matches_no_run')):
        ok, codes = errors(seeds=spec)
        req(wanted in codes,
            f'--seeds {spec!r} selects an inventory no run on disk fills and '
            f'the audit did not emit {wanted!r}; it reported errors {sorted(codes)}. '
            f'Seed completeness, the seed blocks and the DESIGN.md §4.3 '
            f'reserve rule are then all true of the empty set and the audit '
            f'reports a pass it never made.')
        req(not ok,
            f'--seeds {spec!r} produced ok=True. An audit that measured '
            f'nothing has not passed.')

    ok, codes = errors(seeds='not-a-block')
    req('seed_spec_unparseable' in codes and not ok,
        f'--seeds \'not-a-block\' is neither a block name, a list nor a range, '
        f'and the audit reported {sorted(codes)} with ok={ok} instead of '
        f'refusing to resolve it.')
    ctx.note('a nothing-matching, an empty and an unparseable seed selection '
             'each refuse; the declared blocks carry none of those codes')


@case('ANALYSIS_PLAN.md §3, DESIGN.md §9 -- a zero-width interval licenses no '
      'claim of any kind')
def test_degenerate_interval_licenses_no_claim(ctx: Ctx) -> None:
    """A point is not a precise interval.

    A constant arm, a duplicated run row or complete separation between two
    cells makes every bootstrap replicate identical, so the interval collapses
    to a point and nothing about sampling uncertainty was estimable. Read as a
    narrow interval it is maximally decisive, which is how `[+0.0000, +0.0000]`
    came to be published as "every degradation is excluded at 95%" and
    `[1.000, 1.000]` as "the ordering is not a sampling artefact at this n".

    `stats.py` says so in two places, a DEGENERATE equivalence verdict and a
    `degenerate` flag on its BCa intervals, and `plots.py` measures the same
    thing off the bounds. `report.py` was the one consumer that read a point as
    decisive, and neither module's guard was exercised anywhere.
    """
    from experiments import report, stats

    healthy = dict(
        estimand='within_cell_delta', n=10,
        counterfactual='the scratch arm at the same seed',
        rivals='initialisation scale and the optimiser reset',
        excluded_by='C2 and C3',
        refuted_by='an interval containing zero')

    # -- 1. the containment re-derivation calls a point degenerate rather than
    #       contained. [0, 0] lies inside any margin, which is the affirmed
    #       null the whole equivalence section exists to prevent.
    margin = float(stats.EQUIVALENCE_MARGIN)
    same(report.implied_equivalence_verdict(0.0, 0.0, margin), 'DEGENERATE',
         f'a zero-width interval re-derives as an equivalence verdict other '
         f'than DEGENERATE against a margin of +/-{margin}. Every point lies '
         f'inside every margin, so containment alone reads [0, 0] as EQUIVALENT.')

    # -- 2. no kind may be asserted off one, and each refusal names the guard.
    for kind in report.KINDS:
        estimand = next(
            (e for e, (licensed, _w) in report.ESTIMAND_INFERENCE.items()
             if licensed == kind), None)
        if estimand is None:                 # the two interval-only kinds
            estimand = report.INTERVAL_KIND_ESTIMANDS[kind][0]
        evidence = dict(healthy, estimand=estimand, ci_lo=0.05, ci_hi=0.05,
                        verdict='DEGENERATE', margin=margin,
                        signal='grad_norm_trunk', manipulated='the trunk',
                        sd_a=0.031, sd_b=0.042, exclusion_bound=None)
        claim = report.claim('The estimate is 0.05 [0.05, 0.05] over n=10 '
                             'seeds.', kind, evidence)
        req(not claim.accepted,
            f'kind={kind!r} was ACCEPTED off the zero-width interval '
            f'[0.05, 0.05]: {claim.text[:120]!r}')
        req('degenerate-interval' in [r[0] for r in claim.refusals],
            f'kind={kind!r} was refused off a zero-width interval, but by '
            f'{[r[0] for r in claim.refusals]} rather than by '
            f'"degenerate-interval". A refusal for another reason does not show '
            f'that the degeneracy was seen, and it is the degeneracy guard that '
            f'would have to be removed for the sentence to print.')

    # -- 3. the exclusion bound is the module's own, not re-derived. stats.py
    #       withholds it for a degenerate cell; re-deriving it from ci_lo
    #       published "the interval lies at or above +0.0000 score units".
    withheld = report.claim(
        'Every degradation is excluded at 95%.', 'exclusion',
        dict(healthy, ci_lo=0.05, ci_hi=0.31, exclusion_bound=None))
    req(not withheld.accepted
            and 'exclusion-bound' in [r[0] for r in withheld.refusals],
        f'an exclusion sentence was accepted over evidence whose computing '
        f'module withheld the bound: {withheld.text[:140]!r}')

    # -- 4. RQ1's relative effect. theta=1 with a [1, 1] interval is complete
    #       separation, whose exact two-sided permutation p at n=3 per arm is
    #       0.10; "not a sampling artefact at this n" was asserted from the one
    #       interval that says nothing about sampling at all.
    point = report.phrase_relative_effect('A', 'B', 1.0, 1.0, 1.0)
    req('zero width' in point and 'sampling artefact' not in point,
        f'a zero-width relative-effect interval is still read as decisive: '
        f'{point!r}')
    wide = report.phrase_relative_effect('A', 'B', 0.72, 0.60, 0.85)
    req('sampling artefact' in wide,
        f'a genuine interval wholly above the 0.5 null no longer reads as '
        f'decisive, so the refusal above is not discriminating: {wide!r}')
    ctx.note(f'{len(report.KINDS)} claim kinds all refused off a zero-width '
             f'interval by the degenerate-interval guard; the withheld '
             f'exclusion bound is not re-derived; [1, 1] is not decisive')


@case('DESIGN.md §8.3, §9 -- provenance is content-addressed, so a stale '
      'artifact is detectable rather than plausible')
def test_provenance_is_content_addressed(ctx: Ctx) -> None:
    """The hashes that pin an output to its inputs actually track the content.

    `DESIGN.md` §9's last row makes provenance hashes the guard against stale
    artifacts, and §8.3 puts them in every manifest. A hash that did not change
    with the content would satisfy the letter of that and detect nothing, so
    the property is tested rather than assumed: identical bytes hash alike, one
    changed byte hashes differently, and an absent file yields None instead of
    a hash of nothing.
    """
    from src.dqn import provenance

    scratch = ctx.tmpdir('prov_')
    a = os.path.join(scratch, 'a.txt')
    b = os.path.join(scratch, 'b.txt')
    with open(a, 'wb') as fh:
        fh.write(b'plan v1\n')
    with open(b, 'wb') as fh:
        fh.write(b'plan v1\n')
    same(provenance.file_hash(a), provenance.file_hash(b),
         'identical file contents hash differently, so a hash cannot certify '
         'that two artifacts were built from the same input')
    before = provenance.file_hash(a)
    with open(a, 'ab') as fh:
        fh.write(b' ')
    req(provenance.file_hash(a) != before,
        'a one-byte change did not change the content hash. Every "provenance '
        'hash" on a figure or a table would then certify nothing, and a stale '
        'artifact would be indistinguishable from a current one.')
    same(provenance.file_hash(os.path.join(scratch, 'absent.txt')), None,
         'an absent file returns a hash rather than None, so a missing input '
         'would be recorded as though it had been read')

    # -- the plan documents are pinned, and the analysis reports the hash it
    #    ran under. ANALYSIS_PLAN.md is pre-registered, so a confirmatory
    #    result is only interpretable against the version in force when it ran.
    plans = provenance.plan_hashes()
    for name in ('ANALYSIS_PLAN.md', 'DESIGN.md', 'reference_returns.json'):
        req(plans.get(name),
            f'provenance.plan_hashes() has no hash for {name}, so nothing '
            f'records which version of it governed a result')

    # -- every recorded run carries the same set, or the tree cannot be dated.
    manifests = sorted(glob.glob(os.path.join(ctx.runs, '*', '*', 's*',
                                              'manifest.json')))
    if manifests:
        without = []
        for path in manifests:
            with open(path, encoding='utf-8') as fh:
                recorded = ((json.load(fh).get('provenance') or {})
                            .get('plans') or {})
            if not recorded.get('ANALYSIS_PLAN.md'):
                without.append(os.path.relpath(os.path.dirname(path), ctx.runs))
        req(not without,
            f'{len(without)} recorded run(s) carry no analysis-plan hash: '
            f'{without[:5]}. audit.py detects a plan that changed after the '
            f'fact by comparing this hash, and it cannot do so for a run that '
            f'never stored one.')
        ctx.note(f'{len(manifests)} runs carry plan hashes; '
                 f'content addressing verified')
    else:
        ctx.note('content addressing verified; no recorded runs to check '
                 'plan hashes against')


# ===========================================================================
# 7b. Coverage: what this suite does and does not check
# ===========================================================================
#: One entry per row of the `DESIGN.md` §9 table, keyed by the row's exact
#: fallacy text. `cases` names the case(s) here that would fail if the guard
#: were removed; `residual` names, in plain words, what this suite still does
#: NOT check for that row, and is empty only where the coverage is complete.
#:
#: This map exists because the docstring of this module asserted that every §9
#: row had such a test, and that was false for about ten of the sixteen rows.
#: An unverifiable prose claim is exactly the defect §11 records as number 12
#: ("guardrails captioned enforced-in-code with no such code"), one level up and
#: inside the file written to close it. The claim is now a data structure that
#: `test_guardrail_coverage_is_declared` checks against the design document, so
#: a row added, renamed or removed there fails this suite until somebody
#: decides what covers it.
_GUARDRAIL_COVERAGE: dict[str, tuple[tuple[str, ...], str]] = {
    'Affirming a null': (
        ('test_report_wording_guards_fire',), ''),
    'Cross-architecture return presented as a transfer effect': (
        ('test_report_wording_guards_fire',),
        'the headroom columns that DESIGN.md 9 requires beside a between-cell '
        'contrast are rendered by tables.py and are not inspected here'),
    'Comparing two significance verdicts and calling it a comparison': (
        ('test_report_wording_guards_fire',), ''),
    'Effect attributed to learned representations without excluding mechanics': (
        ('test_stats_emits_controls_and_censoring',),
        'that every control is ACCOUNTED for is checked; that a Tier 1 control '
        'was actually run is a property of the launch, which audit.py checks'),
    'Mechanism claimed from prose': (
        ('test_report_wording_guards_fire',), ''),
    'Descriptive metric used inferentially': (
        ('test_stats_refuses_descriptive_metric',), ''),
    'Directional adjective contradicting the numbers': (
        ('test_report_wording_guards_fire',
         'test_value_recal_is_policy_invariant'), ''),
    'Multiplicity ignored': (
        ('test_stats_no_pvalue_outside_family',
         'test_statlib_reference_values'), ''),
    'Selection bias in tuning': (
        ('test_stats_excludes_tune_and_partial_arms', 'test_n1_is_labelled'),
        'the TUNE refusal is exercised at the analysis layer; that the seed '
        'blocks are disjoint in the catalogue is asserted by registry.py at '
        'import and by audit.py over a tree'),
    'Treatment intensity mistaken for architecture': (
        ('test_stats_intensity_gate', 'test_statlib_reference_values'),
        'the analysis-side refusal is driven over a mutated fraction and the '
        'tolerance audit.py refuses on is pinned against stats.py\'s copy; '
        'audit.check_transferred_fraction itself is not driven over a '
        'synthetic membership here, so the audit-side REFUSAL is covered only '
        'through the constant it reads'),
    'Raw returns compared across environments of different scale': (
        ('test_normalisation', 'test_report_wording_guards_fire'), ''),
    'Censored data imputed or dropped': (
        ('test_stats_emits_controls_and_censoring',), ''),
    'Silent seed dropping': (
        ('test_stats_excludes_tune_and_partial_arms',
         'test_recorded_dataset_integrity'),
        'the incomplete-arm refusal is exercised at the analysis layer; '
        'seed-set completeness against the DECLARED seed list is audit.py'),
    'Invalid source treated as valid': (
        ('test_source_validity_gate_applied', 'test_normalisation',
         'test_recorded_dataset_integrity'),
        'the RESERVE replacement draw itself is exercised by sweep.py, not '
        'here; what is checked here is the gate, the exclusion and the label'),
    'Generalising past the evidence': (
        ('test_report_wording_guards_fire',), ''),
    'Stale artifacts': (
        ('test_provenance_is_content_addressed',),
        'the per-figure and per-table provenance sidecars are written by '
        'plots.py and tables.py and are not produced or inspected here'),
}


def _design_section_9_rows() -> list[str]:
    """The fallacy column of the `DESIGN.md` §9 table, read from the document."""
    path = os.path.join(_HERE, 'DESIGN.md')
    with open(path, encoding='utf-8') as fh:
        text = fh.read()
    marker = '## 9. Anti-fallacy guardrails'
    body = text.split(marker, 1)[1].split('\n---', 1)[0]
    rows = []
    for line in body.splitlines():
        line = line.strip()
        if not line.startswith('|'):
            continue
        first = line.strip('|').split('|')[0].strip()
        if first in ('Fallacy', '---') or set(first) <= {'-', ':'}:
            continue
        rows.append(first)
    return rows


@case('DESIGN.md §11 defect 12 -- a guardrail claimed to be enforced in code '
      'must name the code, and this suite must not overstate its own coverage')
def test_guardrail_coverage_is_declared(ctx: Ctx) -> None:
    """Every §9 row is mapped to the case that covers it, or the gap is named.

    The claim this replaces was prose in a docstring: "every row of §9 ... has
    a test here that fails when the guard is removed". It was false for roughly
    ten of the sixteen rows, which is the same failure the table itself once
    had. Making the claim a checked data structure means a row added to the
    design, or a case renamed here, fails the suite instead of silently
    widening the gap between what is claimed and what is checked.
    """
    rows = _design_section_9_rows()
    req(rows, 'no rows parsed out of the DESIGN.md §9 table; either the '
              'section moved or the table is gone, and either way the '
              'coverage claim below is describing nothing')

    declared = set(_GUARDRAIL_COVERAGE)
    actual = set(rows)
    unmapped = sorted(actual - declared)
    req(not unmapped,
        f'{len(unmapped)} row(s) of the DESIGN.md §9 guardrail table have no '
        f'entry in _GUARDRAIL_COVERAGE: {unmapped}. A guardrail added to the '
        f'design with no decision recorded here is a caption without code, '
        f'which is defect 12 of DESIGN.md §11.')
    stale = sorted(declared - actual)
    req(not stale,
        f'{len(stale)} entr(y/ies) in _GUARDRAIL_COVERAGE name a fallacy the '
        f'DESIGN.md §9 table no longer lists: {stale}. The map has drifted '
        f'from the document it claims to describe.')
    same(len(rows), len(set(rows)),
         'the DESIGN.md §9 table lists the same fallacy twice')

    known = {c.name for c in _CASES}
    for row, (names, _residual) in sorted(_GUARDRAIL_COVERAGE.items()):
        req(names,
            f'{row!r} is mapped to no case at all. Either add one or move the '
            f'row out of the table; an unguarded row may not be listed under '
            f'a heading that says "enforced in code".')
        missing = sorted(set(names) - known)
        req(not missing,
            f'{row!r} names case(s) {missing} that do not exist in this file. '
            f'A coverage map that points at nothing is worse than none, '
            f'because it reads as coverage.')

    residuals = sum(1 for _n, r in _GUARDRAIL_COVERAGE.values() if r)
    covered = sorted({name for names, _r in _GUARDRAIL_COVERAGE.values()
                      for name in names})
    ctx.note(f'{len(rows)} DESIGN.md §9 rows, all mapped onto {len(covered)} '
             f'case(s); {residuals} row(s) carry a named residual this suite '
             f'does not check')

# ===========================================================================
# 8. CLI
# ===========================================================================
def _select(names: Sequence[str] | None, quick: bool, full: bool
            ) -> list[Case]:
    """The cases to attempt. `--quick` no longer removes any of them.

    It used to return only the cases with `slow=False`, which meant a
    `--quick` run reported "20 passed" while whatever `--quick` was meant to
    skip did not appear in the SKIPPED list at all. Selection and reporting are
    separated instead: every case is attempted, and `main` records a
    quick-excluded case as a SKIP with its reason, because a SKIP is not a PASS.
    """
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
    return list(_CASES)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='A SKIP is not a PASS: a guard that could not be checked is not '
               'a guard that held.')
    parser.add_argument('--quick', action='store_true',
                        help='reduce the environment sweeps and the MDE '
                             'simulations, and SKIP the cases marked * by '
                             '--list; those cases are reported as skipped, '
                             'not as passed')
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
                             f'(default {DEFAULT_RUNS}). Must exist: a path '
                             f'that does not is a typo, and the suite refuses '
                             f'to run rather than reporting the on-disk '
                             f'guards as skipped')
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

    # A mistyped --runs used to give "18 passed, 0 failed, 2 skipped", exit 0:
    # the on-disk cases found nothing to read and either skipped quietly or
    # reported PASS, so a typo produced a greener suite than the real dataset
    # did. An absent tree is a mistake in the invocation, not a property of the
    # data, so it stops the run before any case reports anything.
    if not os.path.isdir(args.runs):
        raise SystemExit(
            f'validate.py: --runs {args.runs!r} is not a directory. The '
            f'on-disk guards read a recorded run tree, and a path that does '
            f'not exist would report them as skipped rather than as unchecked. '
            f'Pass an existing tree (the default is {DEFAULT_RUNS}).')

    cases = _select(args.test, args.quick, args.full)
    quick_skipped = ([c for c in cases if c.slow] if args.quick else [])
    print(f'validate.py -- {len(cases) - len(quick_skipped)} of '
          f'{len(_CASES)} guard cases'
          f'{" (--quick)" if args.quick else ""}'
          f'{" (--full)" if args.full else ""}')
    print(f'  run tree for the on-disk checks: {args.runs}')
    print()

    width = max(len(c.name) for c in cases)
    failures: list[tuple[str, str]] = []
    skipped: list[tuple[str, str]] = []
    started = time.time()

    for c in cases:
        if c in quick_skipped:
            reason = ('excluded by --quick; it costs an extra stats.py '
                      'invocation over a mutated input')
            skipped.append((c.name, reason))
            print(f'{SKIP} {c.name:<{width}}       -  {reason}')
            continue
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
    if not failures and not skipped:
        print('  Every registered guard ran and held.')
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
