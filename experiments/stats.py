"""Inference for the transfer study, executing `ANALYSIS_PLAN.md` and nothing else.

This module is the enforcement point for the plan. It replaces a version that
contradicted the plan on three counts at once: it tested `validation_reward`
(a raw return, on a scale that differs by hundreds of points between the
environment variants this study compares -- `DESIGN.md` §5.1), it ran unpaired
Mann-Whitney U as the primary test although the design matches scratch and
transfer runs seed-by-seed (`DESIGN.md` §8.1, `ANALYSIS_PLAN.md` §2.1), and it
applied no multiplicity control, no equivalence procedure, no censoring
handling and no power statement at all.

What this file is defending against, defect by defect:

* **A non-significant result narrated into a finding.** The published paper
  claimed "positive transfer" from p=0.421. Here the confirmatory family is
  fixed at eight tests by `ANALYSIS_PLAN.md` §2 -- read from this module's own
  pre-registered constants, never from a CLI argument -- and every other
  analysis is emitted with an interval and no p-value. A null renders as *not
  distinguishable*, and the licensed positive statement is the exclusion bound
  (`DESIGN.md` §9, `ANALYSIS_PLAN.md` §4).
* **A test on a metric the study itself declared descriptive.** §V.A of the
  published paper ran a t-test and Cohen's d on a metric its §V.B called
  descriptive-only and non-normal. `METRIC_ROLES` below is that contradiction
  made mechanical: `require_confirmatory` refuses any endpoint whose declared
  role is not co-primary, and an unclassified metric is refused rather than
  assumed testable.
* **A parametric interval on data declared non-normal.** No t-test, no
  Cohen's d, no normality-assuming interval appears here. Location shifts are
  Hodges-Lehmann with a bias-corrected-and-accelerated seed-level bootstrap;
  between-cell comparisons are Brunner-Munzel relative effects, because the
  cells' across-seed SDs differ by up to 8x on the normalised scale and that
  violates the location-shift assumption Hodges-Lehmann needs
  (`ANALYSIS_PLAN.md` §3).
* **Equivalence asserted from a null.** §4 of the plan replaced TOST with a
  containment check on the bootstrap interval, plus a feasibility gate: a cell
  whose across-seed SD exceeds the ±0.05 margin cannot support an equivalence
  claim at this n, and this module says so instead of reporting a verdict.
* **Censored data imputed or dropped.** `steps_to_threshold` is right-censored
  at the budget. Kaplan-Meier and a Clopper-Pearson interval on P(reached) are
  used; the censoring value is never treated as an observation
  (`ANALYSIS_PLAN.md` §5).
* **A directional adjective contradicting the numbers.** The published paper
  called a narrower spread "broader". Every directional and dispersion phrase
  emitted here is generated from the numbers by the `phrase_*` helpers, and
  `--self-test` asserts that reversing the arguments reverses the word.
* **Treatment intensity mistaken for architecture.** A cross-architecture
  contrast whose transferred-parameter fractions differ by more than
  `INTENSITY_TOLERANCE` is refused unless `--allow-intensity-confound` is
  passed, and the override is stamped into the output (`DESIGN.md` §3.1).
* **A conclusion asserted under one of the two policies the design
  requires.** `DESIGN.md` 3.3 declares a common configuration and a per-cell
  tuned one, and asserts an RQ2 or RQ3 conclusion only where BOTH hold. The
  second policy had no runs behind it until 2026-08-26 and no code here ever,
  so eight Holm-significant members printed as confirmed effects while the
  second leg of the pre-registered condition was absent from the report as
  well as from the data. Sections 5d and 9t compute both legs through the same
  estimator, report a per-cell verdict, and make `not-evaluable` -- the state
  whenever the tuned arms have not been run -- block the assertion instead of
  passing silently. Where the two policies disagree, that disagreement is
  printed as the finding and is not averaged away.
* **A single-seed number quoted as a result.** With n<3 no test, no interval,
  no proportion and no generated between-arm sentence is emitted, and the
  output is stamped `PIPELINE VALIDATION - NOT A RESULT` (`ANALYSIS_PLAN.md`
  §9, `STANDING_INSTRUCTIONS.md` S8). §9 forbids *quoting* such a number, not
  merely testing it, so a suppressed member carries no point estimate either.
* **A silent collapse.** Two rows for one (arm, seed) used to be resolved by
  keeping whichever row pandas visited last; a metric absent in *both* arms
  used to shrink n with nothing said; an unrecognised boolean token used to
  become `None` and then read as PASS everywhere downstream. Each is now a
  refusal that names the rows it refuses.
* **A verdict from a degenerate interval.** A zero-width bootstrap interval (a
  constant arm, or transfer identical to scratch at every seed) used to read as
  EQUIVALENT and as "every degradation is excluded at 95%". Degeneracy is now
  detected and reported as degeneracy.
* **Two definitions of one number.** `statlib.py` holds the pre-registered
  estimators. This module keeps vectorised copies, because a 10,000-resample
  bootstrap cannot call a scalar estimator once per resample per contrast, so
  every copy is checked against `statlib` on every invocation
  (`verify_primitives_against_statlib`) and a disagreement is printed in §1 and
  entered in the deviations of §12.

Input is `runs/per_seed.csv` as produced by `aggregate.py`; the column names
that module pins are the interface. The output maps onto `ANALYSIS_PLAN.md`
§10 as follows, and the mapping is stated rather than asserted, because the
version this replaces claimed to emit "the twelve sections of §10, in that
order" while emitting neither the audit result (§10.1) nor the reference
returns (§10.3):

===============  =====================================================
§10 item         where it is emitted here
===============  =====================================================
1  audit         section 1a, via `audit.audit_ok` on the run tree beside
                 the per-seed table. A failed audit refuses everything
                 below unless `--allow-audit-failure` is passed, and
                 the override is stamped into the output and the JSON
2  inventory      section 2
3  references     section 1b: reference returns and the normalisation
4  descriptives   section 3
5  convergence    section 4
6  confirmatory   section 5
7  equivalence    section 6
8  controls       section 7
9  C4             section 8, for **each** co-primary endpoint
10 estimation     section 9, for **each** co-primary endpoint
11 ledger         section 11
12 deviations     section 12
===============  =====================================================

Section 10, power, is this module's own addition: `ANALYSIS_PLAN.md` §6 asks
for a stated minimum detectable effect and §10's list has no slot for it.
Sections 5d and 9t are the other addition, for the same reason: `DESIGN.md`
3.3's arbitration between the two hyperparameter policies is a condition on
asserting the primary confirmatory conclusion, and §10's twelve items -- written
before that policy had any runs behind it -- have no slot for it either. Both
omissions are recorded in §12 rather than resolved by leaving the section out.

    python experiments/stats.py --per-seed runs/per_seed.csv
    python experiments/stats.py --per-seed runs/per_seed.csv --json out.json
    python experiments/stats.py --self-test
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import re
import sys
import warnings
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from experiments import registry                                  # noqa: E402
from src.dqn import envs, provenance                              # noqa: E402

# ===========================================================================
# 1. Pre-registered constants. These are read from `ANALYSIS_PLAN.md`, not
#    accepted as arguments -- §7 of the plan: "Family membership is fixed by
#    this document before launch ... which is what prevents a result from being
#    rescued by relocating it into a family of one."
# ===========================================================================

PLAN_FILE = os.path.join(_HERE, 'ANALYSIS_PLAN.md')
DESIGN_FILE = os.path.join(_HERE, 'DESIGN.md')

#: The two co-primary endpoints (`ANALYSIS_PLAN.md` §1).
CONFIRMATORY_ENDPOINTS: tuple[str, ...] = ('final_score', 'auc_score')

#: Exactly 4 cells x 2 endpoints (`ANALYSIS_PLAN.md` §2). Size is a constant,
#: not a count of what happened to be computable: Holm is applied over 8 even
#: when fewer members are estimable, which is the conservative direction.
CONFIRMATORY_FAMILY_SIZE = 8

ALPHA = 0.05
#: Strictest Holm step: alpha / m.
ALPHA_STRICTEST = ALPHA / CONFIRMATORY_FAMILY_SIZE

#: `ANALYSIS_PLAN.md` §2: 10,000 resamples, fixed seed, so an interval is
#: reproducible to the last digit.
N_BOOT = 10_000
BOOT_SEED = 20260824

#: Exact sign-flip enumeration is used up to this n (2^20 = 1,048,576 sign
#: assignments); above it the test falls back to Monte Carlo and says so.
EXACT_SIGNFLIP_MAX_N = 20

#: `ANALYSIS_PLAN.md` §4. ±0.05 normalised-score units, ~20 LunarLander return
#: points. Fixed before any data was seen; re-deriving it after seeing a CI is
#: on the forbidden list (§8).
EQUIVALENCE_MARGIN = 0.05

#: `DESIGN.md` §4.2. The positive control passes when the HL estimate's 95%
#: CI lower bound exceeds this.
C4_LOWER_BOUND = -0.10

#: `ANALYSIS_PLAN.md` §5. Declared in advance so a metric exists even when no
#: run reaches "solved".
THRESHOLD_LEVELS: tuple[tuple[str, float], ...] = (
    ('p25', 0.25), ('p50', 0.50), ('p100', 1.00))

#: `ANALYSIS_PLAN.md` §5: log-rank only when both arms have at least 3 events.
LOGRANK_MIN_EVENTS = 3

#: `ANALYSIS_PLAN.md` §9 / `STANDING_INSTRUCTIONS.md` S8.
MIN_N_FOR_INFERENCE = 3
VALIDATION_STAMP = 'PIPELINE VALIDATION - NOT A RESULT'

#: `DESIGN.md` §3.1: the tolerance on the cross-architecture transferred
#: fraction. Beyond it the `arch` contrast is confounded with treatment
#: intensity -- the same class of error the Phase 0 audit found in the
#: published study -- and is refused rather than annotated.
INTENSITY_TOLERANCE = 0.05

#: `DESIGN.md` §4.3: a source is valid when its normalised final score >= 0.6.
SOURCE_VALIDITY_GATE = 0.6

#: `ANALYSIS_PLAN.md` §6.2, computed before launch and, per §6.4, **not
#: re-tuned after seeing results**. Multipliers on the relevant sigma.
#:
#: The unpaired row reads 1.41/1.88 rather than 1.39/1.87 because §6.2's table
#: contradicted its own section: the prose directly beneath it says "1.41
#: against 1.01 sigma" and §6.5's table pins 1.406 for the identical estimator
#: at the identical n, so the table was the odd one out and this module was
#: transcribing the odd one out. The plan's table was corrected on 2026-08-26
#: and the correction is logged in `ANALYSIS_PLAN.md` §11. That is not a
#: re-tuning under §6.4: no confirmatory run exists (§6.6), the correction
#: moves the number *away* from the study's favour by claiming less power, and
#: it reconciles the plan with itself rather than with any observed result.
MDE_MULTIPLIERS: dict[tuple[str, str], float] = {
    ('paired', 'nominal'): 1.00,
    ('paired', 'holm8'): 1.54,
    ('unpaired', 'nominal'): 1.41,
    ('unpaired', 'holm8'): 1.88,
}

#: Where the multipliers came from, printed with the power table. Rewritten by
#: `verify_mde_against_statlib` once that verification has actually run.
MDE_SOURCE = 'ANALYSIS_PLAN.md §6.2 (pre-registered, transcribed here)'

#: The planned confirmatory n, read from the registry's CONFIRM block. §6.2's
#: multipliers are stated at that n, so a verification has to be run there.
PLANNED_N: int = len(registry.SEED_BLOCKS['CONFIRM'])

#: Tolerance for the statlib comparison, and where the number comes from.
#: Two terms, both measured rather than picked:
#:
#: * §6.2 quotes its multipliers to two decimals, so the transcription above
#:   can sit up to 0.005 from the value `statlib` computes through rounding
#:   alone;
#: * `statlib`'s MDE is a 20,000-replicate power simulation. Re-running
#:   `mde_signflip` and `mde_mann_whitney` at n=10 under eight different RNG
#:   seeds gives across-seed SDs of 0.0029, 0.0032, 0.0065 and 0.0063, so the
#:   estimator's own Monte Carlo error is about 0.006 on its worst row.
#:
#: 0.005 + 0.006 is 0.011 and the tolerance is set just below that, at 0.01, so
#: it absorbs rounding and roughly one Monte Carlo SD and nothing more. The
#: predecessor value was 0.02, which was the smallest round number that let the
#: unpaired nominal row through at abs_diff 0.0164 -- a tolerance chosen to
#: pass, sized around the one disagreement it had to hide. That disagreement
#: was real and is fixed at its source in the plan (see `MDE_MULTIPLIERS`);
#: with the plan corrected the largest remaining gap is 0.0036, well inside
#: this bound, so the bound is not carrying any disagreement.
MDE_AGREEMENT_TOLERANCE = 0.01

#: Filled in by `verify_mde_against_statlib`, which, unlike its predecessor,
#: runs.
MDE_VERIFICATION: dict[str, Any] = {
    'ran': False, 'rows': [], 'agree': None,
    'note': 'not run in this invocation'}


def verify_mde_against_statlib(n: int = PLANNED_N) -> dict:
    """Recompute §6.2's multipliers with `statlib` and check them, for real.

    The function this replaces looked for a `statlib.MDE_MULTIPLIERS` dict.
    `statlib` has never exported one, so that function returned at its
    `isinstance` check on every invocation and the safeguard its own docstring
    described ("the values are checked against the pre-registered ones rather
    than trusted") never executed once. The check now calls the functions
    `statlib` does expose, `mde_signflip` and `mde_mann_whitney`, at the
    planned n and at both alphas.

    The pre-registered numbers stand whatever comes back: `ANALYSIS_PLAN.md`
    §6.4 forbids re-tuning the power table after seeing results, so a
    disagreement is reported and never adopted. That is the point of checking
    rather than importing.
    """
    global MDE_SOURCE, MDE_VERIFICATION
    try:
        from experiments import statlib          # type: ignore  # noqa: PLC0415
    except Exception as exc:                     # noqa: BLE001
        MDE_VERIFICATION = {
            'ran': False, 'rows': [], 'agree': None,
            'note': f'statlib.py could not be imported ({exc}); the '
                    'pre-registered multipliers stand unverified'}
        return MDE_VERIFICATION
    fns = {'paired': getattr(statlib, 'mde_signflip', None),
           'unpaired': getattr(statlib, 'mde_mann_whitney', None)}
    absent = sorted(k for k, v in fns.items() if not callable(v))
    if absent:
        MDE_VERIFICATION = {
            'ran': False, 'rows': [], 'agree': None,
            'note': f'statlib.py exposes no MDE function for {absent}; the '
                    'pre-registered multipliers stand unverified'}
        return MDE_VERIFICATION
    alphas = {'nominal': ALPHA, 'holm8': ALPHA_STRICTEST}
    rows: list[dict[str, Any]] = []
    for (test, level), planned in MDE_MULTIPLIERS.items():
        try:
            got = float(fns[test](n, alpha=alphas[level]))
        except Exception as exc:                 # noqa: BLE001
            rows.append({'test': test, 'alpha_level': level, 'n': n,
                         'pre_registered': planned, 'statlib': None,
                         'abs_diff': None, 'agree': False,
                         'note': f'statlib raised {exc.__class__.__name__}'})
            continue
        diff = abs(got - planned)
        rows.append({'test': test, 'alpha_level': level, 'n': n,
                     'pre_registered': planned, 'statlib': got,
                     'abs_diff': diff,
                     'agree': bool(diff <= MDE_AGREEMENT_TOLERANCE),
                     'note': ''})
    agree = all(r['agree'] for r in rows)
    if agree:
        MDE_SOURCE = (f'ANALYSIS_PLAN.md §6.2 (pre-registered), re-derived at '
                      f'n={n} by statlib and agreeing to within '
                      f'{MDE_AGREEMENT_TOLERANCE}')
    else:
        bad = '; '.join(
            '{}/{}: statlib {} vs pre-registered {:.3f}'.format(
                r['test'], r['alpha_level'],
                'n/a' if r['statlib'] is None else f'{r["statlib"]:.3f}',
                r['pre_registered'])
            for r in rows if not r['agree'])
        MDE_SOURCE = (
            'ANALYSIS_PLAN.md §6.2 (pre-registered). statlib DISAGREES and was '
            f'NOT adopted: {bad}. §6.4 forbids re-tuning the power table after '
            'seeing results, so the pre-registered values stand and the '
            'disagreement is reported')
    MDE_VERIFICATION = {'ran': True, 'rows': rows, 'agree': agree,
                        'note': MDE_SOURCE, 'n': n}
    return MDE_VERIFICATION

#: `ANALYSIS_PLAN.md` §6.3, the planning inputs. Reported next to the observed
#: SDs, per the §6.4 update rule, so the reader can see whether the pilot's
#: dispersion resembles what the power calculation assumed.
PLANNING_SDS: dict[str, float] = {
    'mlp-double scratch': 0.093,
    'dueling-vanilla scratch': 0.369,
    'dueling-vanilla transfer': 0.048,
    'mlp-double transfer': 0.043,
}

#: `ANALYSIS_PLAN.md` §6.3: in the noisy cell "the MDE approaches or exceeds
#: the whole distance from random play to solved". One score unit *is* that
#: distance, by construction (`DESIGN.md` §5.1), so it is the plan's own
#: reference for "not powered" rather than a threshold invented here.
UNPOWERED_MDE = 1.0


# ---------------------------------------------------------------------------
# 1.1 The metric-role table. `DESIGN.md` §5.4: "The registry declares these
# roles and `stats.py` refuses a confirmatory test on them -- the mechanical
# fix for the published §V.A/§V.B contradiction."
# ---------------------------------------------------------------------------
CO_PRIMARY = 'co-primary'
SECONDARY = 'secondary'
DESCRIPTIVE = 'descriptive'
MECHANISM = 'mechanism'
BOOKKEEPING = 'bookkeeping'

METRIC_ROLES: dict[str, str] = {
    # Co-primary -- the only testable endpoints (`ANALYSIS_PLAN.md` §1).
    'final_score': CO_PRIMARY,
    'auc_score': CO_PRIMARY,
    # Secondary -- estimation only, no p-values.
    'jumpstart_score': SECONDARY,
    'probe_jumpstart_score': SECONDARY,
    'steps_to_threshold_p25': SECONDARY,
    'steps_to_threshold_p50': SECONDARY,
    'steps_to_threshold_p100': SECONDARY,
    'episode_length_final100': SECONDARY,
    'within_run_sd': SECONDARY,
    'across_seed_sd': SECONDARY,
    'convergence_slope': SECONDARY,
    # Descriptive -- never tested. These are the ones the published paper
    # tested and then declared untestable two paragraphs later.
    'train_return': DESCRIPTIVE,
    'final_return': DESCRIPTIVE,
    'td_loss': DESCRIPTIVE,
    'td_loss_final100': DESCRIPTIVE,
    'epsilon': DESCRIPTIVE,
    'updates': DESCRIPTIVE,
    'env_steps': DESCRIPTIVE,
    'wall_time_s': DESCRIPTIVE,
    'clip_fraction': DESCRIPTIVE,
    'episodes_completed': DESCRIPTIVE,
    # Mechanism -- estimation only; used to license or refuse mechanism
    # wording, never to assert one (`DESIGN.md` §5.5, §9).
    'v_abs_mean': MECHANISM,
    'a_abs_mean': MECHANISM,
    'a_spread': MECHANISM,
    'grad_norm_trunk': MECHANISM,
    'grad_norm_value': MECHANISM,
    'grad_norm_adv': MECHANISM,
    'grad_norm_head': MECHANISM,
    'grad_norm_global': MECHANISM,
    'q_mean': MECHANISM,
    'q_max': MECHANISM,
    'td_error_abs': MECHANISM,
    'cka_transfer_vs_scratch': MECHANISM,
    'cka_drift': MECHANISM,
    'dead_unit_frac': MECHANISM,
    # The plasticity signals of `DESIGN.md` §5.5. They are instrumented, they
    # reach per_seed.csv, and this table used to omit them, so `metric_role`
    # returned 'unclassified', `MECHANISM_COLUMNS` never held them and §9g
    # never printed them. Feature-rank collapse and parameter-norm growth are
    # the rival explanation `DESIGN.md` §10.9 leans on, so omitting them
    # silently disarmed the section that exists to license or refuse a
    # mechanism claim.
    'effective_rank': MECHANISM,
    'stable_rank': MECHANISM,
    'param_norm_total': MECHANISM,
    'param_norm_trunk': MECHANISM,
    # Bookkeeping -- identity and intensity columns, not outcomes.
    'transferred_param_fraction': BOOKKEEPING,
    'reinitialised_layer_count': BOOKKEEPING,
    'params_copied': BOOKKEEPING,
    'source_final_score': BOOKKEEPING,
}

_PREFIX_SCORE_RE = re.compile(r'^prefix_score_(\d+)$')

MECHANISM_COLUMNS: tuple[str, ...] = tuple(
    k for k, v in METRIC_ROLES.items() if v == MECHANISM)
SECONDARY_COLUMNS: tuple[str, ...] = (
    'jumpstart_score', 'probe_jumpstart_score', 'episode_length_final100',
    'within_run_sd')


def metric_role(name: str) -> str:
    """Declared role of a metric, or 'unclassified'.

    An unclassified metric is *not* silently treated as testable: the caller
    refuses it. Adding a metric therefore forces a decision about its role,
    which is the property the published study lacked.
    """
    if name in METRIC_ROLES:
        return METRIC_ROLES[name]
    if _PREFIX_SCORE_RE.match(name):
        return SECONDARY          # RQ6 budget prefixes
    return 'unclassified'


class MetricRoleError(ValueError):
    """Raised when a confirmatory test is requested on a non-testable metric."""


def require_confirmatory(name: str) -> None:
    """Refuse a confirmatory test on anything but a co-primary endpoint."""
    role = metric_role(name)
    if role == CO_PRIMARY:
        return
    raise MetricRoleError(
        f'{name!r} has declared role {role!r}; ANALYSIS_PLAN.md §1 permits a '
        'confirmatory test only on the co-primary endpoints '
        f'{list(CONFIRMATORY_ENDPOINTS)}. Refusing. '
        '(This is the mechanical fix for the published §V.A/§V.B '
        'contradiction: a t-test on a metric §V.B called descriptive-only.)')


# ===========================================================================
# 2. Statistical primitives. Non-parametric throughout; nothing here assumes
#    normality, and nothing here computes a variance-standardised effect size
#    of the Cohen's-d family (`ANALYSIS_PLAN.md` §8).
# ===========================================================================

def _clean(x: Iterable[float]) -> np.ndarray:
    a = np.asarray(list(x), dtype=float)
    return a[np.isfinite(a)]


def correlation(kind: str, x: Sequence[float], y: Sequence[float]) -> float:
    """Pearson or Spearman rho, with a constant input reported as absent.

    SciPy emits a `ConstantInputWarning` on stderr and returns NaN when either
    arm has no variation. That warning went to a stream nothing in this
    pipeline captures, and the NaN then rendered as
    `rho = +nan: reported whatever its value`, which reads as a measurement
    rather than as the absence of one. The warning is silenced here because it
    is not news, and every caller handles the NaN as "not estimable".
    """
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(a) != len(b) or len(a) < 3:
        return float('nan')
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return float('nan')
    if np.std(a) <= 0 or np.std(b) <= 0:
        return float('nan')          # a constant arm: no correlation exists
    fn = sps.pearsonr if kind == 'pearson' else sps.spearmanr
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            return float(fn(a, b)[0])
        except (ValueError, TypeError):
            return float('nan')


def all_signed_means(d: np.ndarray) -> np.ndarray:
    """Mean of `d` under every one of the 2^n sign assignments, exactly.

    Enumerated by subset sums rather than by materialising a 2^n x n sign
    matrix: flipping the signs of a subset S changes the total by -2*sum(S),
    so the whole null distribution is one doubling recurrence. At n=20 that is
    an 8 MB array instead of a 160 MB one.
    """
    total = float(np.sum(d))
    sums = np.zeros(1, dtype=float)
    for x in d:
        sums = np.concatenate((sums, sums + float(x)))
    return (total - 2.0 * sums) / len(d)


def sign_flip_test(d: Sequence[float], n_mc: int = 200_000,
                   seed: int = BOOT_SEED) -> dict:
    """Exact two-sided sign-flip randomisation test; statistic = the mean.

    The primary confirmatory test (`ANALYSIS_PLAN.md` §2). Exact,
    distribution-free, and it uses the matched-seed structure the design
    creates (`DESIGN.md` §8.1). The smallest attainable p is 2/2^n, reported
    alongside so a reader can see immediately whether the corrected threshold
    is even reachable at this sample size.
    """
    d = _clean(d)
    n = len(d)
    if n == 0:
        return {'n': 0, 'p': None, 'mode': 'not computed',
                'min_attainable_p': None}
    obs = abs(float(np.mean(d)))
    if n <= EXACT_SIGNFLIP_MAX_N:
        means = all_signed_means(d)
        p = float(np.mean(np.abs(means) >= obs - 1e-12))
        mode = f'exact enumeration of 2^{n} = {2 ** n} sign assignments'
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(n_mc, n))
        means = (signs * d).mean(axis=1)
        hits = int(np.sum(np.abs(means) >= obs - 1e-12))
        p = (hits + 1) / (n_mc + 1)
        mode = f'Monte Carlo, {n_mc} draws (n>{EXACT_SIGNFLIP_MAX_N})'
    return {'n': n, 'p': p, 'mode': mode,
            'statistic_mean': float(np.mean(d)),
            'min_attainable_p': 2.0 / (2 ** n)}


def hodges_lehmann_paired(d: Sequence[float]) -> float:
    """Median of the Walsh averages of the paired differences.

    The location estimator that matches the sign/signed-rank family, in place
    of a mean difference with a normal interval.
    """
    d = _clean(d)
    if len(d) == 0:
        return float('nan')
    w = (d[:, None] + d[None, :]) / 2.0
    iu = np.triu_indices(len(d))
    return float(np.median(w[iu]))


def hodges_lehmann_two_sample(x: Sequence[float], y: Sequence[float]) -> float:
    """Median of all pairwise differences x_i - y_j."""
    x, y = _clean(x), _clean(y)
    if len(x) == 0 or len(y) == 0:
        return float('nan')
    return float(np.median((x[:, None] - y[None, :]).ravel()))


def boot_indices(n: int, n_boot: int = N_BOOT,
                 seed: int = BOOT_SEED) -> np.ndarray:
    """One fixed index matrix, so every contrast in a joint bootstrap shares
    the same resampling of seeds and their correlations are estimable
    (`ANALYSIS_PLAN.md` §3, the control-contrast row)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_boot, n))


def bca_interval(theta_hat: float, reps: np.ndarray, jack: np.ndarray,
                 alpha: float = ALPHA) -> dict:
    """Bias-corrected and accelerated bootstrap interval.

    `ANALYSIS_PLAN.md` §2 asks for a bias-corrected seed-level bootstrap CI.
    BCa is that plus the acceleration term from the jackknife, which matters
    here because the statistics are medians and Walsh medians on ten units and
    their bootstrap distributions are visibly skewed. Degenerate cases (a
    constant bootstrap distribution, an undefined acceleration) fall back to
    the percentile interval and are flagged rather than hidden.
    """
    reps = np.asarray(reps, dtype=float)
    reps = reps[np.isfinite(reps)]
    if len(reps) < 100 or not np.isfinite(theta_hat):
        return {'lo': float('nan'), 'hi': float('nan'), 'method': 'none',
                'degenerate': False,
                'note': 'too few finite bootstrap replicates'}
    # A bootstrap distribution with no spread carries no information about
    # uncertainty, and the zero-width interval that comes out of it is not a
    # precise estimate: it is the absence of one. Reported as such, because an
    # interval of [+0.0000, +0.0000] used to satisfy the containment check and
    # read as EQUIVALENT, and then as "every degradation is excluded at 95%",
    # from an arm whose every run held the same constant.
    if float(np.max(reps) - np.min(reps)) <= 0.0:
        v = float(reps[0])
        return {'lo': v, 'hi': v, 'method': 'degenerate', 'degenerate': True,
                'note': 'every bootstrap replicate is identical, so the '
                        'interval has zero width: the resampling unit has no '
                        'variation and no interval is estimable'}
    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
    pct = (float(np.percentile(reps, lo_q)), float(np.percentile(reps, hi_q)))
    frac = float(np.mean(reps < theta_hat))
    jack = np.asarray(jack, dtype=float)
    jack = jack[np.isfinite(jack)]
    note = ''
    if frac <= 0.0 or frac >= 1.0 or len(jack) < 3:
        return {'lo': pct[0], 'hi': pct[1], 'method': 'percentile',
                'degenerate': bool(pct[1] - pct[0] <= 0.0),
                'note': 'BCa undefined (bias fraction at a boundary or too '
                        'few jackknife values); percentile interval reported'}
    z0 = float(sps.norm.ppf(frac))
    jm = float(np.mean(jack))
    num = float(np.sum((jm - jack) ** 3))
    den = 6.0 * (float(np.sum((jm - jack) ** 2)) ** 1.5)
    a = num / den if den > 0 else 0.0
    if den <= 0:
        note = 'acceleration set to zero (degenerate jackknife spread)'
    out = []
    for z in (sps.norm.ppf(alpha / 2), sps.norm.ppf(1 - alpha / 2)):
        denom = 1.0 - a * (z0 + z)
        if abs(denom) < 1e-12:
            return {'lo': pct[0], 'hi': pct[1], 'method': 'percentile',
                    'degenerate': bool(pct[1] - pct[0] <= 0.0),
                    'note': 'BCa denominator degenerate; percentile reported'}
        out.append(float(sps.norm.cdf(z0 + (z0 + z) / denom)))
    lo = float(np.percentile(reps, 100 * min(max(out[0], 1e-6), 1 - 1e-6)))
    hi = float(np.percentile(reps, 100 * min(max(out[1], 1e-6), 1 - 1e-6)))
    if lo > hi:
        lo, hi = hi, lo
    return {'lo': lo, 'hi': hi, 'method': 'BCa', 'z0': z0, 'a': a,
            'degenerate': bool(hi - lo <= 0.0), 'note': note}


def hl_vec(samples: np.ndarray) -> np.ndarray:
    """Vectorised Hodges-Lehmann over the last axis of a stack of samples.

    `bootstrap_statistic` calls its statistic 10,000 times per contrast and
    there are several dozen contrasts, so the Walsh-average median is computed
    for the whole bootstrap stack at once. Same numbers as
    `hodges_lehmann_paired`, which `--self-test` checks.
    """
    n = samples.shape[-1]
    iu, ju = np.triu_indices(n)
    w = (samples[..., iu] + samples[..., ju]) / 2.0
    return np.median(w, axis=-1)


def mean_vec(samples: np.ndarray) -> np.ndarray:
    return np.mean(samples, axis=-1)


def median_vec(samples: np.ndarray) -> np.ndarray:
    return np.median(samples, axis=-1)


def bootstrap_statistic(units: np.ndarray, stat: Callable[[np.ndarray], float],
                        n_boot: int = N_BOOT, seed: int = BOOT_SEED,
                        alpha: float = ALPHA, idx: np.ndarray | None = None,
                        vec: Optional[Callable[[np.ndarray], np.ndarray]] = None
                        ) -> dict:
    """Point estimate plus a BCa interval for a statistic of seed-level units.

    `units` is indexed by seed on its first axis, so a paired quantity is
    resampled as a unit and the pairing survives the bootstrap. `vec`, when
    given, computes the statistic for the whole stack of resamples at once; it
    must agree with `stat`, and the two are cross-checked in `--self-test`.
    """
    units = np.asarray(units, dtype=float)
    n = units.shape[0]
    theta = float(stat(units))
    if n < MIN_N_FOR_INFERENCE:
        return {'estimate': theta, 'lo': float('nan'), 'hi': float('nan'),
                'n': n, 'method': 'suppressed', 'degenerate': False,
                'note': f'n={n} < {MIN_N_FOR_INFERENCE}: no interval emitted '
                        f'(ANALYSIS_PLAN.md §9)'}
    if idx is None:
        idx = boot_indices(n, n_boot, seed)
    if vec is not None:
        reps = np.asarray(vec(units[idx]), dtype=float).ravel()
    else:
        reps = np.array([stat(units[row]) for row in idx], dtype=float)
    jack = np.array([stat(np.delete(units, i, axis=0)) for i in range(n)],
                    dtype=float)
    ci = bca_interval(theta, reps, jack, alpha)
    return {'estimate': theta, 'lo': ci['lo'], 'hi': ci['hi'], 'n': n,
            'method': ci['method'], 'note': ci.get('note', ''),
            'degenerate': bool(ci.get('degenerate')), 'reps': reps}


def holm_adjust(pvals: dict[Any, float], m: int) -> dict[Any, float]:
    """Holm-Bonferroni step-down over a family of fixed size `m`.

    `m` is the pre-registered family size, not the number of p-values that
    happened to be computable. If a member is suppressed (an incomplete arm,
    n<3), the surviving members are still adjusted as members of a family of
    `m`, which is the conservative direction and keeps the correction from
    shrinking as a by-product of missing data.
    """
    items = sorted(((k, p) for k, p in pvals.items() if p is not None),
                   key=lambda kp: kp[1])
    out: dict[Any, float] = {}
    running = 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * float(p)))
        out[k] = running
    return out


def brunner_munzel(x: Sequence[float], y: Sequence[float],
                   n_boot: int = N_BOOT, seed: int = BOOT_SEED,
                   alpha: float = ALPHA) -> dict:
    """Relative effect theta = P(X>Y) + 0.5 P(X=Y), with a bootstrap-t CI.

    `ANALYSIS_PLAN.md` §3 prefers this to Hodges-Lehmann for the between-cell
    comparisons because the cells' SDs differ by up to 8x on the normalised
    scale, so a location-shift model -- which is what an HL shift estimates --
    does not hold. theta needs no equal-spread assumption: it is a probability
    statement about the two distributions as they are.
    """
    x, y = _clean(x), _clean(y)
    nx, ny = len(x), len(y)
    if nx < MIN_N_FOR_INFERENCE or ny < MIN_N_FOR_INFERENCE:
        return {'theta': float('nan'), 'lo': float('nan'), 'hi': float('nan'),
                'nx': nx, 'ny': ny, 'se': float('nan'),
                'note': f'n<{MIN_N_FOR_INFERENCE} in at least one group: '
                        f'suppressed (ANALYSIS_PLAN.md §9)'}

    def theta_se(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
        na, nb = len(a), len(b)
        comb = np.concatenate([a, b])
        R = sps.rankdata(comb)
        pa = (R[:na] - sps.rankdata(a)) / nb
        qb = (R[na:] - sps.rankdata(b)) / na
        th = float(np.mean(pa))
        va = float(np.var(pa, ddof=1)) if na > 1 else 0.0
        vb = float(np.var(qb, ddof=1)) if nb > 1 else 0.0
        return th, math.sqrt(max(va / na + vb / nb, 0.0))

    theta, se = theta_se(x, y)
    # The resampling scheme below is statlib.brunner_munzel's, draw for draw:
    # one generator, the x index matrix taken whole and then the y one, the
    # same studentisation and the same usability floor. This function used to
    # draw xb and yb alternately inside a loop from a single stream, which is a
    # different sequence of draws, so the two modules reported different
    # intervals for identical input: two definitions of one number, which
    # `load_per_seed`'s own docstring refuses.
    # `verify_primitives_against_statlib` asserts the agreement on every
    # invocation rather than trusting this comment.
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, nx, size=(int(n_boot), nx))
    iy = rng.integers(0, ny, size=(int(n_boot), ny))
    thetas = np.empty(int(n_boot), dtype=float)
    ts = np.full(int(n_boot), np.nan, dtype=float)
    for k in range(int(n_boot)):
        tb, sb = theta_se(x[ix[k]], y[iy[k]])
        thetas[k] = tb
        if sb and np.isfinite(sb) and sb > 0:
            ts[k] = (tb - theta) / sb
    usable = np.isfinite(ts)
    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
    if se > 0 and int(usable.sum()) >= max(50, int(n_boot) // 20):
        lo = theta - float(np.percentile(ts[usable], hi_q)) * se
        hi = theta - float(np.percentile(ts[usable], lo_q)) * se
        method, note = 'bootstrap-t', 'bootstrap-t'
    else:
        lo = float(np.percentile(thetas, lo_q))
        hi = float(np.percentile(thetas, hi_q))
        method = 'percentile'
        note = ('studentisation degenerate (se=0 or too few usable '
                'resamples); percentile interval reported')
    return {'theta': theta, 'lo': float(min(max(lo, 0.0), 1.0)),
            'hi': float(min(max(hi, 0.0), 1.0)), 'nx': nx, 'ny': ny, 'se': se,
            'method': method, 'note': note}


def jonckheere_effect(groups: Sequence[Sequence[float]]) -> dict:
    """Ordered-alternative concordance for groups given in hypothesised order.

    Reported as a standardised effect in [-1, +1] -- the Jonckheere count
    rescaled so that +1 is a perfectly increasing pattern and -1 a perfectly
    decreasing one -- and **not** as a p-value: RQ5's gradient is
    estimation-only (`ANALYSIS_PLAN.md` §3).
    """
    gs = [_clean(g) for g in groups]
    j = 0.0
    j_max = 0.0
    for i in range(len(gs)):
        for k in range(i + 1, len(gs)):
            a, b = gs[i], gs[k]
            if len(a) == 0 or len(b) == 0:
                continue
            diff = b[:, None] - a[None, :]
            j += float(np.sum(diff > 0)) + 0.5 * float(np.sum(diff == 0))
            j_max += len(a) * len(b)
    if j_max == 0:
        return {'J': float('nan'), 'standardised': float('nan'),
                'note': 'no comparable pairs'}
    return {'J': j, 'J_max': j_max, 'standardised': 2.0 * j / j_max - 1.0,
            'note': 'in [-1,+1]; +1 monotone increasing across the given order'}


def clopper_pearson(k: int, n: int, alpha: float = ALPHA) -> tuple[float, float]:
    """Exact binomial interval. At 0/10 this returns (0, 0.308) -- the
    informative statement the plan asks for in place of a p-value
    (`ANALYSIS_PLAN.md` §5)."""
    if n == 0:
        return (float('nan'), float('nan'))
    lo = 0.0 if k == 0 else float(sps.beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(sps.beta.ppf(1 - alpha / 2, k + 1, n - k))
    return (lo, hi)


def kaplan_meier(times: Sequence[float], events: Sequence[bool],
                 entry: Optional[Sequence[float]] = None) -> dict:
    """Kaplan-Meier survivor function, with optional delayed entry.

    Censored observations contribute their time at risk and are never treated
    as events, which is the whole point: imputing the budget both biases the
    estimate and creates a tie mass that degrades every rank statistic
    downstream (`ANALYSIS_PLAN.md` §5).

    `entry` implements the left truncation `ANALYSIS_PLAN.md` §5 asks for:
    "Kaplan-Meier curves per arm, with delayed entry at the end of the freeze
    window where a freeze is in force". A run whose trunk is frozen for its
    first K updates is not at risk of reaching a threshold on a fine-tuned
    trunk before then, and holding it in the risk set from time zero inflates
    the denominator of every earlier increment. With `entry` supplied a unit
    joins the risk set only at its entry time; with `entry` omitted the
    everyone-from-zero behaviour is unchanged. This function had no
    left-truncation support at all, and the omission was neither reported nor
    recorded as a deviation.

    A non-finite time cannot enter a curve, so such rows are counted into
    `dropped_nonfinite` and reported by the caller rather than deleted in
    silence.
    """
    t = np.asarray(list(times), dtype=float)
    e = np.asarray(list(events), dtype=bool)
    v = (np.asarray(list(entry), dtype=float) if entry is not None
         else np.zeros(len(t), dtype=float))
    keep = np.isfinite(t) & np.isfinite(v)
    dropped = int(len(t) - int(keep.sum()))
    t, e, v = t[keep], e[keep], v[keep]
    n = len(t)
    truncated = bool(entry is not None and np.any(v > 0))
    if n == 0:
        return {'n': 0, 'events': 0, 'curve': [], 'median': None,
                'delayed_entry': truncated, 'dropped_nonfinite': dropped}
    order = np.argsort(t, kind='stable')
    t, e, v = t[order], e[order], v[order]
    curve = []
    surv = 1.0
    i = 0
    while i < n:
        ti = float(t[i])
        d = 0
        c = 0
        while i < n and t[i] == ti:
            if e[i]:
                d += 1
            else:
                c += 1
            i += 1
        # At risk at ti: entered at or before ti and not yet out. With no entry
        # times this is exactly the running count the previous version kept.
        at_risk = int(np.sum((v <= ti) & (t >= ti)))
        if d > 0 and at_risk > 0:
            surv *= (1.0 - d / at_risk)
        curve.append({'t': ti, 'at_risk': at_risk, 'events': d,
                      'censored': c, 'survival': float(surv)})
    median = next((row['t'] for row in curve if row['survival'] <= 0.5), None)
    return {'n': n, 'events': int(e.sum()), 'curve': curve, 'median': median,
            'delayed_entry': truncated, 'dropped_nonfinite': dropped}


def logrank_statistic(t1, e1, t2, e2) -> dict:
    """Log-rank chi-square statistic, **statistic only, no p-value**.

    `ANALYSIS_PLAN.md` §5 licenses a log-rank comparison when both arms have
    at least 3 events; §7 permits p-values only inside the confirmatory
    family. Those two sentences pull in opposite directions, so this module
    takes the conservative reading, emits the statistic and its degrees of
    freedom, withholds the p-value, and records the tension in the deviations
    section rather than resolving it silently.
    """
    t1 = np.asarray(list(t1), float); e1 = np.asarray(list(e1), bool)
    t2 = np.asarray(list(t2), float); e2 = np.asarray(list(e2), bool)
    if min(int(e1.sum()), int(e2.sum())) < LOGRANK_MIN_EVENTS:
        return {'statistic': None, 'df': 1, 'events': (int(e1.sum()),
                                                       int(e2.sum())),
                'note': f'fewer than {LOGRANK_MIN_EVENTS} events in an arm: '
                        f'not computed (ANALYSIS_PLAN.md §5)'}
    times = np.unique(np.concatenate([t1[e1], t2[e2]]))
    o1 = e1_exp = var = 0.0
    for tt in times:
        r1 = float(np.sum(t1 >= tt)); r2 = float(np.sum(t2 >= tt))
        d1 = float(np.sum((t1 == tt) & e1)); d2 = float(np.sum((t2 == tt) & e2))
        r, d = r1 + r2, d1 + d2
        if r <= 1 or d == 0:
            continue
        o1 += d1
        e1_exp += d * r1 / r
        var += d * (r1 / r) * (1 - r1 / r) * (r - d) / (r - 1)
    if var <= 0:
        return {'statistic': None, 'df': 1, 'note': 'zero variance'}
    return {'statistic': float((o1 - e1_exp) ** 2 / var), 'df': 1,
            'observed_arm1': o1, 'expected_arm1': e1_exp,
            'events': (int(e1.sum()), int(e2.sum())),
            'note': 'statistic only; no p-value emitted outside the '
                    'confirmatory family (ANALYSIS_PLAN.md §7)'}


#: Filled in by `verify_primitives_against_statlib`, printed in §1 and, on a
#: disagreement, entered in the deviations of §12.
PRIMITIVE_AGREEMENT: dict[str, Any] = {
    'ran': False, 'rows': [], 'agree': None,
    'note': 'not run in this invocation'}

#: Absolute tolerance for the statlib comparison. These are the same estimators
#: on the same input, so anything above floating-point noise is a disagreement.
PRIMITIVE_TOLERANCE = 1e-9


def verify_primitives_against_statlib(n_boot: int = 2000) -> dict:
    """Check every estimator here against `statlib`'s, on fixed input.

    `statlib.py` holds the pre-registered estimators and has no ability to read
    a run, which is what makes it the reference. This module keeps its own
    copies for one reason: `bootstrap_statistic` evaluates its statistic 10,000
    times per contrast over several dozen contrasts, and that is only tractable
    on a vectorised path over a whole stack of resamples. Two implementations
    of one estimator is exactly the situation `load_per_seed`'s docstring
    refuses ("so that no number in the paper has two definitions"), so the
    copies are not trusted: they are compared here, on inputs fixed in this
    function, on every invocation. A disagreement is printed in §1 and entered
    in the deviations of §12; it is never resolved silently in favour of the
    local copy.

    The comparison is deliberately on small, fixed vectors rather than on the
    data: a check that varied with the input could pass on the data that
    happened to be loaded and fail on the next tree.
    """
    global PRIMITIVE_AGREEMENT
    try:
        from experiments import statlib          # type: ignore  # noqa: PLC0415
    except Exception as exc:                     # noqa: BLE001
        PRIMITIVE_AGREEMENT = {
            'ran': False, 'rows': [], 'agree': None,
            'note': f'statlib.py could not be imported ({exc}); the '
                    'estimators here stand unverified'}
        return PRIMITIVE_AGREEMENT

    d = np.array([0.30, -0.10, 0.50, 0.20, 0.40, -0.20, 0.60, 0.10, 0.05,
                  0.25])
    x = np.array([0.9, 0.4, 1.2, 0.7, 0.1, 0.8, 0.3])
    y = np.array([0.2, -0.4, 0.6, 0.0, 0.5, -0.1, 0.4, 0.25, -0.3])
    km_t = [1.0, 2.0, 3.0, 4.0, 5.0]
    km_e = [True, False, True, False, True]
    rows: list[dict[str, Any]] = []

    def compare(name: str, ours: float, theirs: float,
                tol: float = PRIMITIVE_TOLERANCE) -> None:
        diff = (abs(float(ours) - float(theirs))
                if np.isfinite(ours) and np.isfinite(theirs) else float('nan'))
        rows.append({'primitive': name, 'here': float(ours),
                     'statlib': float(theirs), 'abs_diff': diff,
                     'agree': bool(np.isfinite(diff) and diff <= tol)})

    try:
        compare('sign_flip_test p', sign_flip_test(d)['p'],
                statlib.sign_flip_test(d)['p_two_sided'])
        compare('hodges_lehmann (paired)', hodges_lehmann_paired(d),
                float(statlib.hodges_lehmann(d)))
        compare('hodges_lehmann (two-sample)',
                hodges_lehmann_two_sample(x, y),
                float(statlib.hodges_lehmann(x, y)))
        cp_here = clopper_pearson(3, 10)
        cp_there = tuple(statlib.clopper_pearson(3, 10))
        compare('clopper_pearson lo', cp_here[0], cp_there[0])
        compare('clopper_pearson hi', cp_here[1], cp_there[1])
        holm_here = holm_adjust({0: 0.001, 1: 0.02, 2: 0.04}, 3)
        holm_there = statlib.holm([0.001, 0.02, 0.04])
        for i in range(3):
            compare(f'holm[{i}]', holm_here[i], float(holm_there[i]))
        bm_here = brunner_munzel(x, y, n_boot=n_boot, seed=BOOT_SEED)
        bm_there = statlib.brunner_munzel(x, y, n_boot=n_boot, seed=BOOT_SEED)
        compare('brunner_munzel theta', bm_here['theta'], bm_there['theta'])
        compare('brunner_munzel se', bm_here['se'], bm_there['se'])
        compare('brunner_munzel ci_lo', bm_here['lo'], bm_there['ci_lo'])
        compare('brunner_munzel ci_hi', bm_here['hi'], bm_there['ci_hi'])
        km_here = kaplan_meier(km_t, km_e)
        km_there = statlib.kaplan_meier(km_t, [int(v) for v in km_e])
        surv_there = dict(zip([float(v) for v in km_there['time']],
                              [float(v) for v in km_there['survival']]))
        for row in km_here['curve']:
            if row['t'] in surv_there:
                compare(f'kaplan_meier S({row["t"]:.0f})', row['survival'],
                        surv_there[row['t']])
        compare('kaplan_meier median', float(km_here['median']),
                float(km_there['median']))
        jt_here = jonckheere_effect([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0],
                                     [5.0, 6.0, 7.0]])
        jt_there = statlib.jonckheere_terpstra(
            [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0]],
            n_perm=200, n_boot=200)
        compare('jonckheere J', jt_here['J'], float(jt_there['J']))
    except Exception as exc:                     # noqa: BLE001
        PRIMITIVE_AGREEMENT = {
            'ran': False, 'rows': rows, 'agree': False,
            'note': f'the comparison itself raised {exc.__class__.__name__}: '
                    f'{exc}. Treat the estimators here as unverified'}
        return PRIMITIVE_AGREEMENT

    bad = [r for r in rows if not r['agree']]
    PRIMITIVE_AGREEMENT = {
        'ran': True, 'rows': rows, 'agree': not bad,
        'note': ('every estimator here reproduces statlib.py to within '
                 f'{PRIMITIVE_TOLERANCE:g}'
                 if not bad else
                 'DISAGREEMENT with statlib.py: '
                 + '; '.join(f'{r["primitive"]} {r["here"]:.6f} vs '
                             f'{r["statlib"]:.6f}' for r in bad))}
    return PRIMITIVE_AGREEMENT


# ===========================================================================
# 3. Prose generated from numbers. `DESIGN.md` §9: "Dispersion and direction
#    sentences are generated from the data." The published paper described a
#    narrower spread as "broader", so no directional adjective in this module
#    is written as a literal in a report string -- each one comes out of a
#    function of the numbers, and `--self-test` asserts that swapping the
#    arguments swaps the word.
# ===========================================================================

_TOL = 1e-12


def phrase_direction(value: float, subject: str, reference: str,
                     unit: str = 'score units', tol: float = _TOL) -> str:
    """'<subject> is X units above/below <reference>', word chosen by sign."""
    if not np.isfinite(value):
        return f'{subject} versus {reference}: not estimable'
    if abs(value) <= tol:
        return f'{subject} is exactly level with {reference}'
    word = 'above' if value > 0 else 'below'
    return f'{subject} is {abs(value):.4f} {unit} {word} {reference}'


def phrase_dispersion(name_a: str, sd_a: float, name_b: str,
                      sd_b: float) -> str:
    """Spread comparison; the word comes from the ratio, never from a guess."""
    if not (np.isfinite(sd_a) and np.isfinite(sd_b)):
        return f'{name_a} vs {name_b}: dispersion ratio not estimable'
    # Either spread being zero makes the ratio undefined in one direction or the
    # other, and a degenerate spread is a real situation rather than an error:
    # it is what a single seed gives, and what identical seeds would give. The
    # published paper's dispersion narrative was wrong in its *direction*, so
    # the failure mode to avoid here is inventing a word for a ratio that does
    # not exist -- not crashing, and not silently printing a direction.
    if sd_a <= 0 and sd_b <= 0:
        return (f'{name_a} and {name_b} both have zero across-seed spread; '
                f'no dispersion comparison is defined')
    if sd_b <= 0:
        return (f'{name_b} has zero across-seed spread, so the ratio to '
                f'{name_a} (SD {sd_a:.4f}) is undefined')
    if sd_a <= 0:
        return (f'{name_a} has zero across-seed spread against {name_b} '
                f'(SD {sd_b:.4f}); the ratio is zero and no multiplier is '
                f'reportable')
    ratio = sd_a / sd_b
    if abs(ratio - 1.0) <= 1e-9:
        return (f'{name_a} and {name_b} have the same across-seed spread '
                f'(SD {sd_a:.4f})')
    word = 'wider' if ratio > 1 else 'narrower'
    shown = ratio if ratio > 1 else 1.0 / ratio
    return (f'{name_a} has an across-seed spread {shown:.2f}x {word} than '
            f'{name_b} (SD {sd_a:.4f} vs {sd_b:.4f})')


def phrase_interval_verdict(lo: float, hi: float,
                            label: str = 'the effect') -> str:
    """Whether an interval excludes zero, and in which direction."""
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return f'{label}: no interval emitted'
    if lo > 0:
        return (f'{label} is positive: the interval excludes zero and '
                f'everything below {lo:+.4f}')
    if hi < 0:
        return (f'{label} is negative: the interval excludes zero and '
                f'everything above {hi:+.4f}')
    return (f'{label} is not distinguishable from zero: the interval '
            f'[{lo:+.4f}, {hi:+.4f}] covers zero')


def phrase_exclusion_bound(lo: float, unit: str = 'score units') -> str:
    """The licensed positive statement when a null cannot be affirmed.

    `DESIGN.md` §9 and `ANALYSIS_PLAN.md` §4: this is the form the abstract may
    use, and it is always reported, whatever the verdict.
    """
    if not np.isfinite(lo):
        return 'no exclusion bound available (no interval emitted)'
    if lo >= 0:
        return (f'every degradation is excluded at 95%: the interval lies at '
                f'or above {lo:+.4f} {unit}')
    return f'a degradation worse than {abs(lo):.4f} {unit} is excluded at 95%'


def phrase_magnitude_comparison(name_a: str, a: float, name_b: str,
                                b: float) -> str:
    """Which of two contrasts is bigger in magnitude -- the H2 statement."""
    if not (np.isfinite(a) and np.isfinite(b)):
        return f'{name_a} vs {name_b}: magnitudes not comparable'
    ma, mb = abs(a), abs(b)
    if abs(ma - mb) <= _TOL:
        return f'{name_a} and {name_b} are equal in magnitude ({ma:.4f})'
    word = 'larger' if ma > mb else 'smaller'
    return (f'{name_a} is {word} in magnitude than {name_b} '
            f'({ma:.4f} vs {mb:.4f})')


def phrase_unanimity(d: Sequence[float]) -> str:
    """Whether all seeds move the same way -- the confirmatory bar at n=10."""
    d = _clean(d)
    if len(d) == 0:
        return 'no paired deltas'
    pos = int(np.sum(d > 0))
    neg = int(np.sum(d < 0))
    zero = int(np.sum(d == 0))
    if zero == 0 and (pos == len(d) or neg == len(d)):
        word = 'positive' if pos == len(d) else 'negative'
        return f'all {len(d)} seeds move in the same direction ({word})'
    return (f'seeds are split: {pos} positive, {neg} negative, {zero} zero, '
            f'of {len(d)}')


def phrase_trend(standardised: float, axis: str) -> str:
    """RQ5's direction, generated from the concordance statistic."""
    if not np.isfinite(standardised):
        return f'{axis}: no trend estimable'
    if abs(standardised) <= 1e-9:
        return f'{axis}: no ordered trend (standardised concordance 0.000)'
    word = 'rises' if standardised > 0 else 'falls'
    return (f'{axis}: the delta {word} across the ordered levels '
            f'(standardised concordance {standardised:+.3f})')


# ===========================================================================
# 4. Loading, arm selection, and the ledger.
# ===========================================================================

REQUIRED_COLUMNS: tuple[str, ...] = (
    'run_dir', 'experiments', 'label', 'arm', 'arch', 'target_rule',
    'condition', 'cell', 'env', 'source_env', 'seed', 'seed_block',
    'transfer_set', 'input_policy', 'head_policy', 'freeze_group',
    'freeze_updates', 'permute_kind', 'final_score', 'auc_score', 'plan_hash')

#: The columns that carry a machine-checked invariant of `DESIGN.md` §8.4. An
#: unparseable value in any of them is a refusal, not a `None`.
BOOLEAN_COLUMNS: tuple[str, ...] = (
    'source_valid', 'metrics_contiguous', 'freeze_verified', 'git_dirty')

#: The tokens `aggregate.py` can legitimately write for a boolean. Anything
#: else is schema drift and is refused by `parse_boolean`.
_TRUE_TOKENS = frozenset({'true', '1', '1.0'})
_FALSE_TOKENS = frozenset({'false', '0', '0.0'})


def parse_boolean(value: Any) -> tuple[bool, Optional[bool]]:
    """Parse one invariant flag. Returns (parseable, value-or-None).

    An empty cell is a genuine absence (a scratch run has no source verdict) and
    parses to `None`. Any other unrecognised token is *not* absence: it is a
    schema change, and mapping it to `None` is what let a file whose flags all
    read "no" be reported as a file where every invariant passed.
    """
    if value is None:
        return True, None
    if isinstance(value, (bool, np.bool_)):
        return True, bool(value)
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return True, bool(value)
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return True, None
        if float(value) in (0.0, 1.0):
            return True, bool(value)
        return False, None
    text = str(value).strip()
    if text == '' or text.lower() in ('nan', 'none'):
        return True, None
    if text.lower() in _TRUE_TOKENS:
        return True, True
    if text.lower() in _FALSE_TOKENS:
        return True, False
    return False, None

#: Cell order from the registry, so report rows follow the design's factorial
#: order rather than alphabetical accident.
CELL_ORDER: tuple[str, ...] = tuple(f'{a}-{r}' for a, r in registry.CELLS)


def file_digest(path: str) -> Optional[str]:
    """Content hash, using the SAME function that wrote the manifests.

    This is `src.dqn.provenance.file_hash` (blake2b-16), not a local choice. It
    has to be: the plan hash in a manifest is compared against the current file
    here, and a comparison between two different hash functions would report a
    changed pre-registration on every single invocation -- a warning that always
    fires is a warning nobody reads.
    """
    return provenance.file_hash(path)


@dataclass
class Ledger:
    """The multiplicity ledger, accumulated as the report is produced.

    `ANALYSIS_PLAN.md` §7 requires it printed on every invocation, "so the
    count is a recorded fact rather than a claim".
    """

    confirmatory: list[str] = field(default_factory=list)
    suppressed: list[str] = field(default_factory=list)
    #: Suppressions outside the confirmatory family. Kept apart so the family's
    #: own count of 8 is never inflated by an unrelated arm being too small.
    other_suppressed: list[str] = field(default_factory=list)
    screen_q: list[str] = field(default_factory=list)
    estimation: list[str] = field(default_factory=list)
    #: The DESIGN.md 3.3 arbitration's own entries. A separate compartment,
    #: and for the same reason `other_suppressed` is one: the arbitration
    #: re-tests the SAME eight hypotheses under the secondary policy and adds
    #: no members to the family, so its entries must not be able to reach the
    #: count that ANALYSIS_PLAN.md 7 fixes at eight before launch.
    arbitration: list[str] = field(default_factory=list)
    refusals: list[str] = field(default_factory=list)
    deviations: list[str] = field(default_factory=list)
    tensions: list[str] = field(default_factory=list)

    def est(self, name: str) -> None:
        """Record an analysis that carries an interval and no p-value."""
        self.estimation.append(name)


@dataclass
class Options:
    per_seed: str
    metrics: tuple[str, ...]
    experiments: Optional[tuple[str, ...]]
    target_env: str
    source_env: str
    interface_env: str
    allow_intensity_confound: bool
    source_policy: str
    n_boot: int
    boot_seed: int
    json_out: Optional[str]
    #: Where the run tree sits, for the `ANALYSIS_PLAN.md` §10.1 audit gate.
    audit_root: str = ''
    #: §10.1's "explicit override that is stamped into the output".
    override_audit: bool = False
    #: Passed through to `audit.py`, exactly as `report.py` passes them: the
    #: seed set the runs were launched at, and the launch-level overrides in
    #: force. Without them a one-seed validation tree fails SEED COMPLETENESS,
    #: which is a true statement about the design and not about this analysis.
    audit_seeds: Optional[str] = None
    audit_overrides: tuple[str, ...] = ()
    #: Whether to re-derive the §6.2 MDE multipliers with statlib and check
    #: them. On by default: the check that never ran is the defect this
    #: replaces.
    verify_mde: bool = True
    #: The TUNE-block exclusion, carried so §2c can report it. The rows are
    #: removed in `main` before any section runs, so a section testing for
    #: them can only report zero (ANALYSIS_PLAN.md §10.2 requires exclusions
    #: reported, and this one was visible only in the JSON).
    tune_runs_excluded: int = 0
    tune_seeds_excluded: tuple[int, ...] = ()
    #: Explicit path to the DESIGN.md 3.3 tuning-selection artifact. Empty
    #: means "the one stored under the run tree", which is where `tuning.py`
    #: writes it and where `registry.py` enumerates the tuned arms from.
    selection_path: str = ''
    #: Rows carrying a tuned arm label, removed from the primary analysis set
    #: in `main` before any section runs and reported in the arbitration
    #: section instead. Counted here so 10.2's "exclusions reported" covers
    #: this one too.
    tuned_rows_set_aside: int = 0


def load_per_seed(path: str, ledger: Ledger) -> pd.DataFrame:
    """Read the pinned per-seed table; refuse a table missing its contract."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'{path} not found. Run `python experiments/aggregate.py` first; '
            'this module never recomputes a per-run scalar itself, so that no '
            'number in the paper has two definitions.')
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f'{path} is missing pinned columns {missing}. The per-seed schema '
            'is a contract between aggregate.py and this module; refusing to '
            'guess a substitute.')
    bad: list[str] = []
    for col in BOOLEAN_COLUMNS + tuple(f'censored_{tag}'
                                       for tag, _ in THRESHOLD_LEVELS):
        if col not in df.columns:
            continue
        parsed = []
        for v in df[col]:
            ok, val = parse_boolean(v)
            if not ok:
                bad.append(f'{col}={v!r}')
            parsed.append(val)
        df[col] = pd.Series(parsed, index=df.index, dtype=object)
    if bad:
        seen = sorted(set(bad))
        raise ValueError(
            f'{path} carries {len(bad)} unparseable boolean value(s) in its '
            f'machine-checked invariant columns: {seen[:10]}'
            + (' ...' if len(seen) > 10 else '')
            + '. The previous version mapped an unrecognised token to None, '
              'and every consumer then failed OPEN: a file whose '
              'source_valid, metrics_contiguous and freeze_verified all read '
              '"no" reported "0 rejected", "440 ok, 0 failing" and a clean '
              'confirmatory family, with every DESIGN.md §8.4 invariant '
              'reading PASS when in fact every run had failed. Column '
              'presence was checked and value parseability was not. Refusing '
              'to guess what these mean.')
    absent = [c for c in ('transferred_param_fraction', 'jumpstart_score',
                          'probe_jumpstart_score', 'within_run_sd',
                          'convergence_slope', 'source_final_score')
              if c not in df.columns]
    if absent:
        ledger.deviations.append(
            f'per_seed.csv lacks optional columns {absent}; the sections that '
            'need them report as unavailable rather than approximating them')
    return df


def in_experiments(df: pd.DataFrame,
                   wanted: Optional[Sequence[str]]) -> pd.DataFrame:
    """Restrict to runs belonging to any of `wanted`.

    A run can belong to several experiments -- identical configurations are
    deliberately shared and `registry.all_jobs` de-duplicates them -- so
    membership is a set test on a semicolon-joined field, not equality.
    """
    if not wanted:
        return df
    want = set(wanted)
    keep = df['experiments'].fillna('').map(
        lambda s: bool(want & set(str(s).split(';'))))
    return df[keep]


def rows_where(df: pd.DataFrame, **eq) -> pd.DataFrame:
    out = df
    for key, val in eq.items():
        if key not in out.columns:
            return out.iloc[0:0]
        out = out[out[key].isna()] if val is None else out[out[key] == val]
    return out


def protocol_match(df: pd.DataFrame) -> pd.DataFrame:
    """Runs on the protocol under study, as declared in `registry.PROTOCOL`.

    The categorical protocol fields are matched exactly. `freeze_updates` is
    deliberately not matched against the registry's confirmatory value, because
    a pilot invocation legitimately shortens it; instead its observed value is
    reported and its constancy across the selected runs is checked, so a
    heterogeneous freeze window is caught rather than averaged over.
    """
    out = df
    for k in ('transfer_set', 'input_policy', 'head_policy', 'freeze_group'):
        out = out[out[k] == registry.PROTOCOL[k]]
    return out


def primary_transfer_arm(df: pd.DataFrame, opts: Options) -> pd.DataFrame:
    """C1: the transfer arm at the primary protocol on the target task."""
    return protocol_match(target_side(
        rows_where(df, condition='transfer', env=opts.target_env,
                   source_env=opts.source_env)))


#: Seed blocks whose runs exist only to donate a source checkpoint and are
#: barred from target-side estimation (`DESIGN.md` §3.4: C4SRC is "never used
#: for target-side estimation"; RESERVE supplies replacement sources only).
#: Without this filter a C4 donor -- a scratch run on the target environment --
#: would be pooled into that cell's scratch baseline and quietly shift the
#: denominator of every delta.
SOURCE_ONLY_BLOCKS: tuple[str, ...] = ('C4SRC', 'RESERVE')


def target_side(df: pd.DataFrame) -> pd.DataFrame:
    """Drop runs whose seed block exists only to donate a source checkpoint."""
    if 'seed_block' not in df.columns:
        return df
    return df[~df['seed_block'].isin(SOURCE_ONLY_BLOCKS)]


def scratch_arm(df: pd.DataFrame, opts: Options) -> pd.DataFrame:
    """C0: the cell's own scratch baseline -- the denominator of every delta.

    Source-side runs on another environment are excluded by the env filter
    rather than by a label heuristic; source-side runs that happen to sit on the
    *target* environment -- the C4 donors of `DESIGN.md` §4.2, which are scratch
    runs on LunarLander drawn from the disjoint `C4SRC` block -- are excluded by
    their seed block. Without that second filter the positive control's donors
    would be pooled into the baseline they are meant to be independent of.
    """
    return target_side(rows_where(df, condition='scratch',
                                  env=opts.target_env))


def arm_by_seed(d: pd.DataFrame, metric: str) -> dict:
    """One arm as seed -> value, with duplicates and gaps kept separate.

    Three outcomes are distinguished, because collapsing them is how a seed
    disappears:

    * a seed with exactly one row and a finite value is usable;
    * a seed with more than one row is **ambiguous**, not resolvable by taking
      one of them. The previous version built this map with
      `out[int(r['seed'])] = float(v)` inside `iterrows`, so a duplicated
      (arm, seed) silently kept whichever row pandas visited last: one injected
      row with `final_score = 99` moved a cell's mean delta by +9.77 and
      shifted every other member's Holm-adjusted p, while the completeness
      table still read n=10 and reported no unpaired seed at all;
    * a seed whose row exists but whose metric is absent is a **gap**. It is
      reported rather than skipped: when the metric was absent in *both* arms
      the seed used to vanish from `common`, `only_a` and `only_b` alike, so n
      shrank with nothing said and the incomplete-arm refusal never fired.
    """
    values: dict[int, float] = {}
    seen: dict[int, list[float]] = {}
    rows: dict[int, int] = {}
    finite: dict[int, int] = {}
    for _, r in d.iterrows():
        s = int(r['seed'])
        rows[s] = rows.get(s, 0) + 1
        v = r.get(metric)
        if pd.notna(v):
            finite[s] = finite.get(s, 0) + 1
            seen.setdefault(s, []).append(float(v))
            values.setdefault(s, float(v))
    duplicates = sorted(s for s, k in rows.items() if k > 1)
    # Whether the duplicated rows agree on the metric matters to the reader:
    # equal values are one run recorded twice (a bookkeeping fault the audit's
    # CONFIG/DIGEST CONSISTENCY check owns), unequal values are two different
    # runs claiming one identity. Neither is resolvable here, and the arm is
    # ambiguous either way, but the two want different fixes.
    conflicting = sorted(
        s for s in duplicates
        if len(set(round(v, 12) for v in seen.get(s, []))) > 1)
    gaps = sorted(s for s in rows if finite.get(s, 0) == 0)
    usable = {s: v for s, v in values.items() if s not in duplicates}
    return {'values': usable, 'seeds_present': sorted(rows),
            'duplicates': duplicates, 'conflicting': conflicting,
            'metric_missing': gaps, 'n_rows': int(len(d))}


def arm_problems(info: dict, metric: str, who: str = 'arm') -> list[str]:
    """Every reason ONE arm is not a clean one-value-per-seed sample, as prose.

    Split out of `pairing_problems` so that a section with a single arm --
    §3's descriptives, §3b's headroom denominator, §4's slope gate, §7's
    per-condition vectors -- states the ambiguity in the same words as §5
    rather than inventing its own or, as four of them did, saying nothing at
    all and reporting a number computed over rows.
    """
    out: list[str] = []
    dup, clash = info['duplicates'], info['conflicting']
    if dup:
        if clash:
            kind = (f'seed(s) {clash} carry DIFFERENT {metric} values across '
                    'those rows, so two different runs claim one identity')
        else:
            kind = (f'the rows agree on {metric}, so this is one run recorded '
                    'more than once (audit.py owns that as a CONFIG/DIGEST '
                    'CONSISTENCY fault)')
        out.append(
            f'the {who} has more than one row for seed(s) {dup}: {kind}. '
            f'The arm is ambiguous and those seeds are excluded from the '
            f'sample rather than resolved by taking one row '
            f'(DESIGN.md §8.4)')
    return out


def seed_vector(d: pd.DataFrame, metric: str, who: str = 'arm') -> dict:
    """THE per-seed reduction of one arm: one value per seed, or a refusal.

    Every per-seed aggregation in this module goes through here or through
    `paired_by_seed`, which is the two-arm form of the same thing. Both are
    built on `arm_by_seed`, so there is exactly one place where rows become
    seeds and exactly one rule for what happens when they cannot.

    What this is defending against: `section_descriptives`, §3b, §4 and
    `section_controls` each used to do their own reduction. Three of them took
    `len(_clean(g[metric]))` as n, which is a ROW count, so an arm holding two
    rows for one seed reported n=6 from 3 seeds, an SD that shrank towards
    zero as rows were duplicated, and a "seed-level bootstrap" CI that
    resampled rows. The fourth, `section_controls`, wrote
    `vals[int(r['seed'])] = float(v)` inside `iterrows` and kept whichever row
    pandas visited last, so its contrasts depended on the order of the CSV:
    reading `runs_demo/per_seed.csv` forwards and then row-reversed -- the
    same rows, the same multiset -- moved 32 contrast rows and flipped the
    sign and the printed verdict of C1-C0, the confirmatory estimand, in two
    cells.

    A duplicated (arm, seed) is therefore a **refusal**, not a resolution:
    `refused` is True, `reason` says which seeds and whether the rows agree,
    and the caller emits no estimate, exactly as §5 does. Seeds whose metric
    is absent are reported in `metric_missing` and are simply not observations
    of this metric; they do not make the arm ambiguous.
    """
    info = arm_by_seed(d, metric)
    problems = arm_problems(info, metric, who)
    seeds = sorted(info['values'])
    return {'seeds': seeds,
            'values': np.array([info['values'][s] for s in seeds],
                               dtype=float),
            'n': len(seeds), 'n_rows': info['n_rows'],
            'seeds_present': info['seeds_present'],
            'duplicates': info['duplicates'],
            'conflicting': info['conflicting'],
            'metric_missing': info['metric_missing'],
            'problems': problems,
            'refused': bool(info['duplicates']),
            'reason': '; '.join(problems) or None}


def paired_by_seed(a: pd.DataFrame, b: pd.DataFrame, metric: str) -> dict:
    """Match two arms on seed. Reports incompleteness, never drops silently.

    `DESIGN.md` §1: "One seed was dropped from one arm with no stated rule."
    Seeds present in one arm only are returned and surface in the report, and
    so now are duplicated (arm, seed) rows and seeds whose metric is missing on
    both sides. `n_a` and `n_b` count distinct usable seeds; `rows_a` and
    `rows_b` count rows, so the two can be compared and a collapse is visible
    instead of being absorbed by a field named for one and holding the other.
    """
    ia, ib = arm_by_seed(a, metric), arm_by_seed(b, metric)
    xa, xb = ia['values'], ib['values']
    common = sorted(set(xa) & set(xb))
    present_a, present_b = set(ia['seeds_present']), set(ib['seeds_present'])
    gaps = sorted((set(ia['metric_missing']) | set(ib['metric_missing']))
                  & (present_a & present_b))
    return {'seeds': common,
            'a': np.array([xa[s] for s in common], dtype=float),
            'b': np.array([xb[s] for s in common], dtype=float),
            'only_a': sorted(present_a - present_b),
            'only_b': sorted(present_b - present_a),
            'n_a': len(xa), 'n_b': len(xb),
            'rows_a': ia['n_rows'], 'rows_b': ib['n_rows'],
            'dup_a': ia['duplicates'], 'dup_b': ib['duplicates'],
            'conflicting_a': ia['conflicting'],
            'conflicting_b': ib['conflicting'],
            'metric_missing': gaps}


def pairing_problems(pair: dict, metric: str, name_a: str = 'transfer',
                     name_b: str = 'scratch') -> list[str]:
    """Every reason this pairing is not a clean matched sample, as prose.

    Returned as a list so a caller can print them, put them in the ledger, or
    suppress on them; an empty list means the two arms match seed for seed with
    a value on both sides.
    """
    out: list[str] = []
    for who, dup, clash in ((name_a, pair['dup_a'], pair['conflicting_a']),
                            (name_b, pair['dup_b'], pair['conflicting_b'])):
        # The single-arm wording lives in `arm_problems` so that §3, §3b, §4
        # and §7 say the same thing about the same fault as §5 does.
        out.extend(arm_problems({'duplicates': dup, 'conflicting': clash},
                                metric, f'{who} arm'))
    if pair['metric_missing']:
        out.append(
            f'seed(s) {pair["metric_missing"]} have a run in both arms but no '
            f'{metric} value, so they cannot enter the paired sample. They are '
            f'named here because a seed absent from both arms used to vanish '
            f'from the count entirely')
    return out


def incomplete_arm_reason(pair: dict, name_a: str = 'transfer',
                          name_b: str = 'scratch') -> Optional[str]:
    """§5's partial-arm refusal, in §5's words, for every section that needs it.

    `ANALYSIS_PLAN.md` §8 forbids "dropping a seed, for any reason, after it
    has run", and `DESIGN.md` §8.4 refuses a partial arm. §5 obeyed both. §7,
    §9b, §9e and §9j took the intersection instead and analysed whatever
    survived, so the same estimand §5 had just refused as an incomplete arm
    came back one section later with an interval and a directional sentence
    attached. Returns None when the two arms match seed for seed.
    """
    if not pair['only_a'] and not pair['only_b']:
        return None
    return (f'incomplete arm: seeds {pair["only_a"]} appear only in '
            f'{name_a}, {pair["only_b"]} only in {name_b}. A partial arm is '
            f'refused (DESIGN.md §8.4); no seed is dropped to rescue the '
            f'estimate (ANALYSIS_PLAN.md §8)')


def paired_refusal(pair: dict, metric: str, name_a: str = 'transfer',
                   name_b: str = 'scratch') -> Optional[str]:
    """The one reason a paired estimate cannot be computed, or None.

    Duplicates first, then incompleteness, in the order §5 tests them, so a
    section that refuses on this string refuses on the same ground and with
    the same words as the confirmatory family does.
    """
    problems = pairing_problems(pair, metric, name_a, name_b)
    dup = [p for p in problems if 'more than one row' in p]
    if dup:
        return '; '.join(dup)
    return incomplete_arm_reason(pair, name_a, name_b)


# ---------------------------------------------------------------------------
# 4.1 The two hyperparameter policies of `DESIGN.md` 3.3, and the arbitration
#     that section pre-registers between them.
#
# 3.3 declares two policies and then makes an assertion conditional on both:
#
#   * PRIMARY, the common configuration: one learning rate and target-update
#     rule for all four cells, fixed a priori at lr=5e-4, hard update every
#     1000 updates. This is the setting in which "identical hyperparameters"
#     is a verified fact.
#   * SECONDARY, per-cell tuned: each cell's own E3-selected configuration.
#     `lr` is invariant WITHIN a cell across {scratch, transfer, C2, C3} and
#     deliberately varies ACROSS cells.
#   * "an RQ2 or RQ3 conclusion is asserted only if it holds under BOTH
#     policies. Where they disagree, that disagreement is the finding, and it
#     is reported as one."
#
# Until 2026-08-26 the secondary policy had no runs behind it: only E3 varied
# `lr` anywhere in the catalogue. The arbitration was therefore unsatisfiable
# and RQ2, the study's primary confirmatory question, could not have been
# asserted at all -- and this module asserted it anyway, because the
# arbitration existed in the design and nowhere in the code. A Holm-significant
# member printed as a confirmed effect with nothing in the output saying that
# the second leg of the pre-registered condition had never been run. That is
# the defect closed here, and `not-evaluable` is the default state.
#
# What is deliberately NOT done: the tuned leg does not add members to the
# confirmatory family. The arbitration is a CONJUNCTION over the same eight
# hypotheses, not sixteen tests of sixteen hypotheses. Asserting a conclusion
# only where both legs reject makes the rejection region the INTERSECTION of
# the two legs' regions, and an intersection is never larger than either of
# them, so the family-wise error rate of the conjunction is bounded by that of
# either leg alone -- which Holm over eight already controls at 0.05. Counting
# the tuned leg as eight further members would correct twice for a procedure
# that is at most as liberal as one leg, and it would change the family size
# `ANALYSIS_PLAN.md` 7 fixes before launch, which is the thing that stops a
# result being rescued by relocating it. `section_ledger` asserts the count.
# ---------------------------------------------------------------------------

#: The two policies, named once so the output cannot call them two things.
POLICY_COMMON = 'common'
POLICY_TUNED = 'tuned'
POLICY_NAMES: dict[str, str] = {
    POLICY_COMMON: 'common configuration (PRIMARY, DESIGN.md 3.3)',
    POLICY_TUNED: 'per-cell tuned (SECONDARY, DESIGN.md 3.3)',
}

#: The three arbitration verdicts. `NOT_EVALUABLE` is the default and is what
#: the absence of tuned runs produces; it BLOCKS an assertion rather than
#: permitting one, which is the direction the pre-registration requires.
AGREES = 'agrees'
DISAGREES = 'disagrees'
NOT_EVALUABLE = 'not-evaluable'
ARBITRATION_VERDICTS: tuple[str, ...] = (AGREES, DISAGREES, NOT_EVALUABLE)

#: What one leg of the arbitration can conclude about an RQ2 member. The two
#: directional values are distinguished because two policies that both reject
#: while pointing opposite ways do not agree, and a verdict computed from the
#: reject/do-not-reject bit alone would call that agreement.
CONCLUSION_UP = 'effect-positive'
CONCLUSION_DOWN = 'effect-negative'
CONCLUSION_NULL = 'not-distinguishable'
CONCLUSION_NONE = 'none'

#: Read through `getattr` so this module still imports against a registry that
#: predates the tuned catalogue: the arbitration then reports itself as
#: not-evaluable, which is true, instead of failing at import.
TUNED_LABEL_PREFIX: str = str(getattr(registry, 'TUNED_LABEL_PREFIX',
                                      'tuned-'))
TUNED_POLICY_NAME: str = str(getattr(registry, 'TUNED_POLICY',
                                     'secondary-per-cell-tuned'))


def tuned_experiment_ids() -> frozenset[str]:
    """Experiment ids that execute the secondary policy (`E1t`, `E2t`)."""
    return frozenset(getattr(registry, 'TUNED_OF', {}) or {})


def tuned_label_mask(df: pd.DataFrame) -> pd.Series:
    """Rows whose arm label marks them as retuned.

    The label is the primary signal rather than the `experiments` column,
    because it is written by the run itself: a tuned arm that shares a run
    directory with its common-policy counterpart has no label of its own at
    all (`registry.all_jobs` de-duplicates onto the arm it saw first, which
    `activate_tuned_arms` guarantees is the common one), and a tuned arm that
    does NOT share a directory always has one.
    """
    if 'label' not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)
    return df['label'].fillna('').astype(str).str.startswith(
        TUNED_LABEL_PREFIX)


def claims_tuned_experiment(df: pd.DataFrame) -> pd.Series:
    """Rows the catalogue attributes to a tuned experiment.

    This is the second signal, and it is the only one that can see a SHARED
    run: a cell whose E3 selection equals the a priori configuration produces
    digests identical to its common-policy arms, so `aggregate.py` resolves
    that one run to both `E1` and `E1t` and the row's label stays the common
    one. Without this the shared cells would be reported as having no tuned
    arms, which is the opposite of the truth.
    """
    ids = tuned_experiment_ids()
    if not ids or 'experiments' not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)
    return df['experiments'].fillna('').map(
        lambda s: bool(ids & set(str(s).split(';'))))


def common_policy_rows(df: pd.DataFrame) -> pd.DataFrame:
    """The primary analysis set: everything that is not a retuned arm.

    A shared run stays here, and belongs here: it is the common policy's run,
    and it is the tuned policy's run as well because the two configurations
    coincide in that cell. Removing it would delete the common-policy arm.
    """
    return df[~tuned_label_mask(df)]


def tuned_policy_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows that exist only because the secondary policy was run."""
    return df[tuned_label_mask(df)]


def _config_summary(rows: pd.DataFrame, opts: 'Options') -> dict:
    """The (lr, target_update) a set of target-task rows was trained at.

    Read off the runs rather than taken from the selection artifact, so a
    selection that does not describe the runs on disk shows up as a
    disagreement instead of being assumed away.
    """
    out: dict[str, Any] = {}
    frame = rows
    if 'env' in rows.columns:
        frame = target_side(rows[rows['env'] == opts.target_env])
    for col in ('lr', 'target_update', 'target_update_freq'):
        if col not in frame.columns:
            out[col] = ()
            continue
        vals = [v for v in frame[col].tolist() if not pd.isna(v)]
        seen: list[str] = []
        for v in vals:
            text = f'{float(v):g}' if isinstance(v, (int, float, np.number)) \
                and not isinstance(v, bool) else str(v)
            if text not in seen:
                seen.append(text)
        out[col] = tuple(sorted(seen))
    return out


def _config_text(cfg: dict) -> str:
    """One line naming a policy's configuration in a cell."""
    lr = '/'.join(cfg.get('lr') or ()) or '?'
    upd = '/'.join(cfg.get('target_update') or ()) or '?'
    freq = '/'.join(cfg.get('target_update_freq') or ())
    return f'lr={lr} {upd}' + (f'/{freq}' if freq else '')


@dataclass
class TunedPolicy:
    """Where the secondary policy's runs are, cell by cell, and whether.

    Built once in `main` and passed down, so the confirmatory section and the
    arbitration section cannot disagree about which runs are the second leg.
    """

    frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    cells: dict[str, dict] = field(default_factory=dict)
    selection: Any = None
    selection_note: str = ''
    placeholder: bool = False
    rows_labelled: int = 0
    notes: list[str] = field(default_factory=list)
    #: Non-empty when an assertion is blocked for a reason that is not a
    #: per-cell verdict: no selection artifact, or a placeholder rule.
    assertion_block: str = ''

    def state(self, cell: str) -> str:
        return str(self.cells.get(cell, {}).get('state', 'absent'))

    def reason(self, cell: str) -> str:
        return str(self.cells.get(cell, {}).get('reason', ''))

    def config(self, cell: str) -> dict:
        return dict(self.cells.get(cell, {}).get('config') or {})

    def cell_frame(self, cell: str) -> pd.DataFrame:
        return self.frames.get(cell, pd.DataFrame())

    def frame(self) -> pd.DataFrame:
        """Every cell's tuned-leg rows in one table, for the RQ3 leg."""
        parts = [f for f in (self.frames.get(c) for c in CELL_ORDER)
                 if f is not None and len(f)]
        if not parts:
            return pd.DataFrame(columns=list(REQUIRED_COLUMNS))
        return pd.concat(parts, ignore_index=False)

    @property
    def evaluable_cells(self) -> tuple[str, ...]:
        return tuple(c for c in CELL_ORDER
                     if self.state(c) in ('own-runs', 'shared'))

    @property
    def available(self) -> bool:
        return bool(self.evaluable_cells)


def read_tuning_selection(opts: 'Options') -> tuple[Any, str]:
    """The stored selection artifact, or `(None, why not)`.

    Never recomputes one and never writes one. `tuning.read_selection`
    re-verifies the content address on every read, so an artifact edited after
    the tuned runs were enumerated from it is refused rather than believed;
    that refusal arrives here as the reason string and is reported.
    """
    try:
        from experiments import tuning                     # noqa: PLC0415
    except Exception as exc:                               # noqa: BLE001
        return None, (f'tuning.py could not be imported '
                      f'({exc.__class__.__name__}: {exc}), so the DESIGN.md '
                      f'3.3 selection artifact cannot be read')
    root = opts.audit_root or os.path.dirname(
        os.path.abspath(opts.per_seed)) or '.'
    path = opts.selection_path or None
    try:
        selection = tuning.read_selection(root, path=path, required=False,
                                          verify=True, warn_placeholder=False)
    except Exception as exc:                               # noqa: BLE001
        return None, f'{exc.__class__.__name__}: {exc}'
    if selection is None:
        return None, tuning.missing_message(root, path=path)
    return selection, ''


def resolve_tuned_policy(common: pd.DataFrame, tuned: pd.DataFrame,
                         opts: 'Options', ledger: Ledger) -> TunedPolicy:
    """Find the secondary policy's runs per cell, or say why there are none.

    Three states per cell, and the third is the default:

    * **own-runs** -- the cell's E3 selection differs from the a priori
      configuration, so its tuned arms are their own run directories carrying
      their own `tuned-` labels.
    * **shared** -- the cell's selection equals the a priori configuration, so
      `lr` being a trajectory field makes the tuned arms' digests identical to
      the common-policy arms' and the two share run directories. The second leg
      of the arbitration in that cell is then the SAME RUNS as the first, the
      two legs agree by construction, and that is a fact about the selection
      rather than a result.
    * **absent** -- there are no tuned runs for this cell. `DESIGN.md` 3.3's
      arbitration cannot be evaluated, so nothing about RQ2 or RQ3 is asserted
      there. This is today's state for every cell: E3 has not finished, so no
      selection exists and the tuned stage has not run.
    """
    out = TunedPolicy(rows_labelled=int(len(tuned)))
    selection, why = read_tuning_selection(opts)
    out.selection = selection
    out.selection_note = why
    if selection is not None:
        out.placeholder = bool(getattr(selection, 'is_placeholder', False))

    if selection is None:
        out.assertion_block = (
            'no DESIGN.md 3.3 selection artifact could be read, so the '
            'configuration the tuned arms claim to execute cannot be checked '
            'against the pre-registered per-cell selection')
    elif out.placeholder:
        rule = str((selection.rule or {}).get('id'))
        out.assertion_block = (
            f'the selection was computed under {rule!r}, a PLACEHOLDER rule, '
            f'not the criterion ANALYSIS_PLAN.md 2.3 pre-registers. Arms '
            f'enumerated from it test the pipeline; they are not the '
            f'secondary policy, so they cannot license an assertion')

    for cell in CELL_ORDER:
        own = tuned[tuned['cell'] == cell] if 'cell' in tuned.columns \
            else tuned.iloc[0:0]
        base = common[common['cell'] == cell] if 'cell' in common.columns \
            else common.iloc[0:0]
        equals_a_priori: Optional[bool] = None
        if selection is not None:
            try:
                equals_a_priori = bool(selection.equals_a_priori(cell))
            except Exception:                              # noqa: BLE001
                equals_a_priori = None
                out.notes.append(
                    f'{cell}: the selection artifact carries no entry for '
                    f'this cell, so it cannot say whether the tuned arms '
                    f'share the common policy\'s runs')
        attributed = base[claims_tuned_experiment(base)] if len(base) \
            else base

        if len(own):
            frame, state = own, 'own-runs'
            reason = ''
            if equals_a_priori:
                # The selection says this cell reselected the a priori
                # configuration, so `all_jobs` should have de-duplicated its
                # tuned arms onto the common-policy runs and no `tuned-` label
                # should exist. Both cannot be true; the runs win, and the
                # contradiction is reported rather than resolved silently.
                out.notes.append(
                    f'{cell}: the selection says the tuned configuration '
                    f'equals the a priori one, so the tuned arms should SHARE '
                    f'the common policy\'s run directories and carry no '
                    f'{TUNED_LABEL_PREFIX!r} label; {len(own)} row(s) carry '
                    f'one anyway. The runs on disk are used and the '
                    f'contradiction is reported, not resolved')
                ledger.deviations.append(
                    f'DESIGN.md 3.3: {cell} has {len(own)} tuned-labelled '
                    f'run(s) although the stored selection marks it as '
                    f'sharing the common policy\'s runs')
        elif equals_a_priori:
            frame, state = base, 'shared'
            reason = ''
        elif equals_a_priori is None and len(attributed):
            frame, state = attributed, 'shared'
            reason = ''
            out.notes.append(
                f'{cell}: no selection artifact was read, but '
                f'{len(attributed)} run(s) are attributed to a tuned '
                f'experiment by run digest, which is what a cell sharing the '
                f'common policy\'s runs looks like')
        else:
            frame, state = base.iloc[0:0], 'absent'
            if equals_a_priori is False:
                reason = (
                    'the selection gives this cell a configuration of its '
                    f'own ({_config_text(_selection_config(selection, cell))}'
                    '), so its tuned arms are their own runs -- and none of '
                    'them is in the analysis set. The tuned stage has not '
                    'been run for this cell')
            elif selection is None:
                reason = ('no tuned run and no selection artifact: the '
                          'secondary policy of DESIGN.md 3.3 has not been '
                          'executed for this cell')
            else:
                reason = ('the selection artifact does not cover this cell '
                          'and no tuned run is present')

        cfg = _config_summary(frame, opts) if len(frame) else {}
        entry: dict[str, Any] = {
            'cell': cell, 'state': state, 'reason': reason,
            'rows': int(len(frame)), 'config': cfg,
            'config_text': _config_text(cfg) if cfg else '-',
            'shares_common_runs': bool(state == 'shared'),
        }
        if selection is not None:
            sel_cfg = _selection_config(selection, cell)
            entry['selected'] = _config_text(sel_cfg) if sel_cfg else '-'
            entry['equals_a_priori'] = equals_a_priori
            if len(frame) and sel_cfg and cfg:
                mismatch = [k for k in ('lr', 'target_update')
                            if cfg.get(k) and sel_cfg.get(k)
                            and tuple(cfg[k]) != tuple(sel_cfg[k])]
                if mismatch:
                    entry['config_mismatch'] = mismatch
                    ledger.deviations.append(
                        f'DESIGN.md 3.3: {cell}\'s tuned runs were trained at '
                        f'{_config_text(cfg)} but the stored selection names '
                        f'{_config_text(sel_cfg)} (fields {mismatch}). The '
                        f'runs are not the selection they claim to execute')
        out.cells[cell] = entry
        out.frames[cell] = frame
    return out


def _selection_config(selection: Any, cell: str) -> dict:
    """The stored selection's configuration for a cell, in `_config_summary`
    shape, so the selection and the runs are compared in one vocabulary."""
    if selection is None:
        return {}
    try:
        cfg = selection.config_for(cell)
    except Exception:                                      # noqa: BLE001
        return {}
    out = {'lr': (f'{float(cfg.lr):g}',),
           'target_update': (str(cfg.target_update),)}
    freq = getattr(cfg, 'target_update_freq', None)
    if str(cfg.target_update) == 'hard' and freq is not None:
        out['target_update_freq'] = (f'{float(freq):g}',)
    return out


def transferred_fraction(df: pd.DataFrame) -> float:
    if 'transferred_param_fraction' not in df.columns:
        return float('nan')
    v = _clean(df['transferred_param_fraction'])
    return float(np.mean(v)) if len(v) else float('nan')


def sd(x: Sequence[float]) -> float:
    x = _clean(x)
    return float(np.std(x, ddof=1)) if len(x) > 1 else float('nan')


def fmt(v: Any, nd: int = 4) -> str:
    if v is None:
        return '-'
    if isinstance(v, bool):
        return 'yes' if v else 'no'
    if isinstance(v, str):
        return v                  # already formatted; never re-parsed as a float
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return '-' if not np.isfinite(f) else f'{f:.{nd}f}'


def table(rows: list[dict], columns: Sequence[str], nd: int = 4) -> str:
    """Fixed-width table with no external dependency and no index column."""
    if not rows:
        return '  (no rows)'
    cells = [[str(c) for c in columns]]
    for r in rows:
        cells.append([fmt(r.get(c), nd) for c in columns])
    widths = [max(len(row[i]) for row in cells) for i in range(len(columns))]
    out = []
    for i, row in enumerate(cells):
        out.append('  ' + '  '.join(v.ljust(widths[j])
                                    for j, v in enumerate(row)).rstrip())
        if i == 0:
            out.append('  ' + '  '.join('-' * w for w in widths))
    return '\n'.join(out)


def h1(title: str) -> None:
    print()
    print('=' * 78)
    print(title)
    print('=' * 78)


def h2(title: str) -> None:
    print()
    print(f'-- {title} ' + '-' * max(0, 74 - len(title)))


# ===========================================================================
# 5. The report. Sections follow `ANALYSIS_PLAN.md` §10 in order. Each returns
#    a JSON-able dict so the printed page and the machine-readable output
#    cannot drift apart.
# ===========================================================================

def section_audit(opts: Options, ledger: Ledger) -> dict:
    """§10.1 -- the audit result, and the gate that hangs off it.

    `ANALYSIS_PLAN.md` §10 item 1: "Audit result. If the audit fails, nothing
    below is emitted without an explicit override that is stamped into the
    output." This module used to have no audit gate at all: it never imported
    `audit.py`, substituted a provenance section for the audit result, and then
    described itself as emitting the twelve sections of §10 in order. A reader
    of the output could not tell an audited tree from an unaudited one.

    The audit reads the run tree, not the per-seed table, so it can only run
    when a tree sits beside the table. When it cannot run, that is said and
    recorded as a deviation; a missing audit is never reported as a passing
    one, which is the failure mode the gate exists to prevent.
    """
    h1('1a. AUDIT GATE (ANALYSIS_PLAN.md §10.1)')
    root = opts.audit_root or os.path.dirname(
        os.path.abspath(opts.per_seed)) or '.'
    out: dict[str, Any] = {'root': root, 'ran': False, 'ok': None,
                           'override': bool(opts.override_audit),
                           'checks': []}
    print(f'  run tree            : {root}')
    if not os.path.isdir(root):
        out['note'] = f'{root} is not a directory: the audit cannot run'
        print(f'  NOT RUN: {out["note"]}.')
        ledger.deviations.append(
            'ANALYSIS_PLAN.md §10.1 audit gate not evaluated: '
            + out['note'])
        return out
    try:
        from experiments import audit as audit_mod   # noqa: PLC0415
        ok, rep = audit_mod.audit_ok(
            root, experiments=opts.experiments, seeds=opts.audit_seeds,
            overrides=audit_mod.parse_overrides(opts.audit_overrides or None))
    except Exception as exc:                          # noqa: BLE001
        out['note'] = f'{exc.__class__.__name__}: {exc}'
        print(f'  NOT RUN: audit.py raised {out["note"]}.')
        ledger.deviations.append(
            f'ANALYSIS_PLAN.md §10.1 audit gate not evaluated: audit.py '
            f'raised {out["note"]}')
        return out
    if not rep.get('runs_discovered'):
        out['note'] = (f'no run directories under {root}, so there is nothing '
                       'to audit. The per-seed table was analysed without an '
                       'audit')
        print(f'  NOT RUN: {out["note"]}.')
        ledger.deviations.append(
            'ANALYSIS_PLAN.md §10.1 audit gate not evaluated: '
            + out['note'])
        return out
    checks = [{'check': c.get('name'), 'status': c.get('status'),
               'findings': len(c.get('findings') or [])}
              for c in rep.get('checks', [])]
    print(f'  runs discovered     : {rep.get("runs_discovered")}')
    print(f'  runs in scope       : {rep.get("runs_in_scope")}')
    print(f'  plan hash           : {rep.get("plan_hash")}')
    print()
    print(table(checks, ('check', 'status', 'findings')))
    out.update({'ran': True, 'ok': bool(ok), 'checks': checks,
                'errors': rep.get('errors'), 'warnings': rep.get('warnings'),
                'runs_discovered': rep.get('runs_discovered')})
    failing = [c['check'] for c in checks if c['status'] == 'FAIL']
    if ok:
        print()
        print('  AUDIT PASSED: every check in audit.py is green on this tree.')
        return out
    print()
    print(f'  AUDIT FAILED: {rep.get("errors")} error(s), '
          f'{rep.get("warnings")} warning(s) on {", ".join(failing)}.')
    if opts.override_audit:
        print('  ' + '*' * 70)
        print('  OVERRIDE IN FORCE: --allow-audit-failure was passed, so '
              'the sections below are')
        print('  emitted over a FAILED audit. ANALYSIS_PLAN.md §10.1 permits '
              'that only with an')
        print('  explicit override stamped into the output; this is the '
              'stamp, and it is also')
        print('  written into the JSON and into §12. Every number below '
              'inherits it.')
        print('  ' + '*' * 70)
        ledger.deviations.append(
            f'--allow-audit-failure in force over a FAILED audit ({failing}): '
            'ANALYSIS_PLAN.md §10.1 requires the override stamped into the '
            'output, and every number in this report carries it')
    return out


def section_provenance(df: pd.DataFrame, opts: Options,
                       ledger: Ledger) -> dict:
    """§10.1 -- provenance and the pre-registration hash.

    The plan is hashed into every manifest. If the hash in the data differs
    from the current file, the pre-registration no longer covers the analysis
    and every confirmatory result below is exploratory. That is stated loudly
    rather than left for a reader to notice.
    """
    h1('1. PROVENANCE AND PLAN HASH')
    plans = provenance.plan_hashes()
    current = plans.get('ANALYSIS_PLAN.md')
    design = plans.get('DESIGN.md')
    hashes = sorted(set(str(v) for v in df['plan_hash'].dropna().unique()))
    print(f'  per-seed table       : {opts.per_seed}')
    print(f'  table digest         : {file_digest(opts.per_seed)}')
    print(f'  rows                 : {len(df)}')
    print(f'  ANALYSIS_PLAN.md     : {current}   (current file)')
    print(f'  DESIGN.md            : {design}')
    print(f'  reference_returns    : {plans.get("reference_returns.json")}')
    print('  hash function        : blake2b-16, the one src/dqn/provenance.py '
          'used to write')
    print('                         the manifests, so the comparison below is '
          'like for like')
    print(f'  plan hash in data    : {", ".join(hashes) or "(absent)"}')
    for col, name in (('git_commit', 'git commit'), ('git_dirty', 'git dirty')):
        if col in df.columns:
            vals = sorted(set(str(v) for v in df[col].dropna().unique()))
            print(f'  {name:<21}: {", ".join(vals) or "(absent)"}')
    print(f'  bootstrap            : {opts.n_boot} resamples, seed '
          f'{opts.boot_seed} (ANALYSIS_PLAN.md §2)')

    stale = [h for h in hashes if h != current]
    exploratory = bool(stale) or len(hashes) > 1
    if exploratory:
        print()
        print('  *** PRE-REGISTRATION WARNING ***')
        if len(hashes) > 1:
            print('  The runs were produced under more than one plan hash, so '
                  'they were not')
            print('  all analysed under one pre-registration.')
        if stale:
            print('  The plan hash recorded in the run data does not match the '
                  'current')
            print('  ANALYSIS_PLAN.md. A changed plan means every confirmatory '
                  'result below is')
            print('  EXPLORATORY, and must be labelled so in the paper '
                  '(ANALYSIS_PLAN.md §11).')
        ledger.deviations.append(
            'plan hash in the run data differs from the current '
            'ANALYSIS_PLAN.md: confirmatory results are exploratory')
    else:
        print()
        print('  Plan hash matches: the pre-registration covers this analysis.')

    prim = verify_primitives_against_statlib()
    print()
    print('  Estimator cross-check against statlib.py (the module that holds '
          'the')
    print('  pre-registered estimators and cannot read a run):')
    if not prim['ran']:
        print(f'    NOT RUN: {prim["note"]}')
        ledger.deviations.append(
            f'the estimators in stats.py were not checked against statlib.py: '
            f'{prim["note"]}')
    elif prim['agree']:
        print(f'    {len(prim["rows"])} estimator value(s) checked; '
              f'{prim["note"]}.')
    else:
        print(f'    *** {prim["note"]} ***')
        print('    Two implementations of one estimator means one number in '
              'the paper has two')
        print('    definitions, which is exactly what load_per_seed refuses '
              'for the per-run')
        print('    scalars. Fix the disagreement before quoting anything '
              'below.')
        ledger.deviations.append(
            f'stats.py and statlib.py disagree numerically: {prim["note"]}')
    return {'plan_hash_current': current, 'design_hash': design,
            'plan_hash_in_data': hashes,
            'table_digest': file_digest(opts.per_seed),
            'rows': int(len(df)), 'exploratory': exploratory,
            'n_boot': opts.n_boot, 'boot_seed': opts.boot_seed,
            'statlib_agreement': prim}


def section_reference_returns(df: pd.DataFrame, opts: Options,
                              ledger: Ledger) -> dict:
    """§10.3 -- the reference returns and the normalisation used.

    `ANALYSIS_PLAN.md` §10 lists this as item 3 and this module used to skip it
    entirely, printing only the `reference_returns.json` digest in §1. The
    digest says the file did not change; it does not say what the numbers are,
    and every score in every table below is `(return - random) / (threshold -
    random)` with these numbers in it. A reader who cannot see them cannot
    check a single normalised value, and cannot see that the environments
    differ by hundreds of return points, which is the whole reason `DESIGN.md`
    §5.1 forbids comparing raw returns across them.
    """
    h1('1b. REFERENCE RETURNS AND THE NORMALISATION (ANALYSIS_PLAN.md §10.3)')
    print('  normalised score = (return - random_return) / (threshold - '
          'random_return),')
    print('  so 0 is the measured random policy and 1 is the registered solved '
          'threshold')
    print('  (DESIGN.md §5.1). The random-policy references are MEASURED, in '
          'measure_references.py,')
    print('  not assumed: a multiplicative gate on raw return is neither sign- '
          'nor origin-safe.')
    rows = []
    for env_id in sorted(set(str(v) for v in df['env'].dropna().unique())):
        try:
            ref = envs.reference(env_id)
        except (KeyError, ValueError, FileNotFoundError) as exc:
            rows.append({'env': env_id, 'note': f'no reference: {exc}'})
            ledger.deviations.append(
                f'{env_id} has no measured random-policy reference, so its '
                f'normalised scores cannot be checked')
            continue
        rows.append({'env': env_id,
                     'random_return': ref.get('random_return'),
                     'random_sd': ref.get('random_sd'),
                     'noop_return': ref.get('noop_return'),
                     'noop_score': ref.get('noop_score'),
                     'threshold': ref.get('threshold'),
                     'denominator': ref.get('denominator'),
                     'episodes': ref.get('episodes'),
                     'note': ''})
    print()
    print(table(rows, ('env', 'random_return', 'random_sd', 'noop_return',
                       'noop_score', 'threshold', 'denominator', 'episodes',
                       'note'), nd=3))
    print('  noop_score is where a do-nothing policy sits on this same scale. '
          'It is printed')
    print('  because RQ5 turns on it: a family whose noop score moves across '
          'levels confounds')
    print('  shift severity with task difficulty (DESIGN.md §5.1), and §9c '
          'carries that caveat.')
    return {'references': rows,
            'file': getattr(envs, 'REFERENCE_FILE', None),
            'digest': file_digest(getattr(envs, 'REFERENCE_FILE', '') or '')}


def section_inventory(df: pd.DataFrame, opts: Options, ledger: Ledger) -> dict:
    """§10.2 -- inventory, completeness, source validity, intensity."""
    h1('2. RUN INVENTORY')
    out: dict[str, Any] = {}

    h2('2a. arms present')
    rows = []
    for (env, cell, cond, label), g in df.groupby(
            ['env', 'cell', 'condition', 'label'], dropna=False):
        seeds = sorted(int(s) for s in g['seed'].unique())
        blocks = sorted(set(str(b) for b in g['seed_block'].dropna().unique()))
        rows.append({'env': env, 'cell': cell, 'condition': cond,
                     'label': label, 'n': len(g),
                     'seeds': ','.join(str(s) for s in seeds),
                     'seed_block': '/'.join(blocks) or 'UNKNOWN',
                     'transfer_set': g['transfer_set'].iloc[0],
                     'freeze_updates': g['freeze_updates'].iloc[0]})
    rows.sort(key=lambda r: (str(r['env']), str(r['cell']), str(r['condition']),
                             str(r['label'])))
    print(table(rows, ('env', 'cell', 'condition', 'label', 'n', 'seeds',
                       'seed_block', 'transfer_set', 'freeze_updates')))
    out['arms'] = rows

    h2('2b. completeness of the confirmatory contrast (per cell)')
    comp = []
    for cell in CELL_ORDER:
        cdf = df[df['cell'] == cell]
        s = scratch_arm(cdf, opts)
        t = primary_transfer_arm(cdf, opts)
        ss = sorted(int(x) for x in s['seed'].unique())
        ts = sorted(int(x) for x in t['seed'].unique())
        labels = sorted(set(str(x) for x in t['label'].unique()))
        # Rows against distinct seeds, both printed. A row count presented as a
        # seed count is how a duplicated (arm, seed) hides: n_transfer read 10
        # while the arm held 11 rows and one seed was silently collapsed.
        pair = paired_by_seed(t, s, opts.metrics[0])
        # `paired_seeds` is the number §5 will actually analyse, taken from the
        # same helper §5 uses -- not `len(set(ss) & set(ts))`, which intersects
        # raw seed labels and therefore counts a seed that is duplicated in one
        # arm, or that has no value for the endpoint, as a usable pair. That
        # column drives the PIPELINE VALIDATION banner below, so the banner was
        # being withheld on data where every confirmatory member is suppressed:
        # on runs_demo it read 3 for all four cells while §5a reported n=0 for
        # all eight members, and one injected duplicate row made it read 10
        # where §5's paired sample was 9.
        comp.append({'cell': cell, 'n_scratch': len(ss), 'n_transfer': len(ts),
                     'rows_scratch': len(s), 'rows_transfer': len(t),
                     'paired_seeds': len(pair['seeds']),
                     'seed_labels_shared': len(set(ss) & set(ts)),
                     'scratch_only': ','.join(str(x) for x in
                                              sorted(set(ss) - set(ts))) or '-',
                     'transfer_only': ','.join(str(x) for x in
                                               sorted(set(ts) - set(ss))) or '-',
                     'duplicated_seeds':
                         ','.join(str(x) for x in
                                  sorted(set(pair['dup_a'])
                                         | set(pair['dup_b']))) or '-',
                     'transfer_labels': ';'.join(labels) or '-'})
        if len(labels) > 1:
            ledger.refusals.append(
                f'{cell}: {len(labels)} distinct labels match the primary '
                f'protocol ({labels}); the primary arm is ambiguous')
        if set(ss) != set(ts):
            ledger.deviations.append(
                f'{cell}: scratch and transfer seed sets differ; the arm is '
                'incomplete and DESIGN.md §8.4 refuses a partial arm')
        for note in pairing_problems(pair, opts.metrics[0]):
            print(f'  {cell}: {note}')
            ledger.deviations.append(f'{cell}: {note}')
    print(table(comp, ('cell', 'n_scratch', 'n_transfer', 'rows_scratch',
                       'rows_transfer', 'paired_seeds', 'seed_labels_shared',
                       'scratch_only', 'transfer_only', 'duplicated_seeds',
                       'transfer_labels')))
    print(f'  paired_seeds is the sample §5 will analyse on '
          f'{opts.metrics[0]}, from the same helper')
    print('  §5 uses: a seed duplicated in either arm, or with no value for '
          'the endpoint, is')
    print('  not a pair. seed_labels_shared is the raw intersection of the '
          'two seed columns,')
    print('  printed beside it so the gap between what exists and what pairs '
          'is visible.')
    out['completeness'] = comp
    smallest = min((c['paired_seeds'] for c in comp), default=0)
    if smallest < MIN_N_FOR_INFERENCE:
        print()
        print('  ' + '*' * 70)
        print(f'  {VALIDATION_STAMP}')
        print(f'  The smallest paired sample in the confirmatory contrast is '
              f'n={smallest}, below the')
        print(f'  floor of {MIN_N_FOR_INFERENCE}. This banner is printed here, '
              'at the top of the report, as well as')
        print('  at the end, so that no reader reaches a table before reaching '
              'the warning.')
        print('  ' + '*' * 70)
    out['smallest_paired_n'] = smallest

    h2('2c. seed blocks')
    blk = []
    for name, g in df.groupby('seed_block', dropna=False):
        blk.append({'seed_block': name, 'runs': len(g),
                    'seeds': ','.join(str(int(s)) for s in
                                      sorted(g['seed'].unique()))})
    print(table(blk, ('seed_block', 'runs', 'seeds')))
    # The TUNE runs were removed in `main` before this section ran, so a
    # branch testing for them here could never fire and the exclusion appeared
    # nowhere in the printed report: it survived only as a field in the JSON.
    # `ANALYSIS_PLAN.md` §10.2 requires exclusions reported, so the count is
    # carried in and printed whether or not any row remains.
    tune_seeds = ','.join(str(s) for s in opts.tune_seeds_excluded) or '-'
    if opts.tune_runs_excluded:
        print(f'  REFUSAL: {opts.tune_runs_excluded} run(s) sat in the TUNE '
              f'block (seeds {tune_seeds}) and were removed before any section '
              'of this report ran.')
        print('  ANALYSIS_PLAN.md §8 forbids any reported estimate computed '
              'on TUNE seeds')
        print('  (selection leakage): revision 1 selected hyperparameters on '
              'seeds 0-4 and then')
        print('  ran confirmatory arms on 0-9, so half of every confirmatory '
              'sample was tuned on.')
        ledger.refusals.append(
            f'{opts.tune_runs_excluded} TUNE-block run(s) (seeds '
            f'{tune_seeds}) excluded from every estimate (selection leakage, '
            'ANALYSIS_PLAN.md §8)')
    else:
        print('  No run sits in the TUNE block, so no selection-leakage '
              'exclusion was needed.')
    leftover = df[df['seed_block'] == 'TUNE']
    if len(leftover):
        print(f'  ERROR: {len(leftover)} TUNE-block run(s) reached the '
              'inventory. The exclusion in main() did not hold.')
        ledger.refusals.append(f'{len(leftover)} TUNE-block run(s) survived '
                               'the exclusion filter')
    donors = df[df['seed_block'].isin(SOURCE_ONLY_BLOCKS)]
    if len(donors):
        print(f'  {len(donors)} run(s) sit in a source-only block '
              f'{list(SOURCE_ONLY_BLOCKS)}. DESIGN.md §3.4 bars C4SRC from '
              'target-side estimation and')
        print('  reserves RESERVE for replacement sources, so they are '
              'excluded from every')
        print('  target-side arm. They are scratch runs on the target '
              'environment, so without')
        print('  this rule they would silently join the baseline they are '
              'meant to be')
        print('  independent of:')
        for _, r in donors.iterrows():
            print(f'    {r["label"]} s{int(r["seed"])} '
                  f'({r["seed_block"]}) on {r["env"]}')
    unknown = df[df['seed_block'].fillna('UNKNOWN') == 'UNKNOWN']
    if len(unknown):
        print(f'  {len(unknown)} run(s) have no recognised seed block; their '
              'provenance in the block scheme of DESIGN.md §3.4 is unstated.')
        ledger.deviations.append(f'{len(unknown)} run(s) in no declared seed '
                                 'block')
    out['seed_blocks'] = blk

    h2('2d. source validity (DESIGN.md §4.3: normalised gate >= '
       f'{SOURCE_VALIDITY_GATE})')
    if 'source_valid' not in df.columns:
        print('  source_valid column absent: no verdict available.')
        out['source_validity'] = None
    else:
        srows = []
        for (cell, cond, label), g in df.groupby(['cell', 'condition',
                                                  'label'], dropna=False):
            v = g['source_valid']
            if v.isna().all():
                continue
            invalid = sorted(int(r['seed']) for _, r in g.iterrows()
                             if r['source_valid'] is False)
            scores = _clean(g.get('source_final_score', pd.Series(dtype=float)))
            srows.append({'cell': cell, 'condition': cond, 'label': label,
                          'n': len(g), 'valid': int((v == True).sum()),
                          'invalid': int((v == False).sum()),
                          'invalid_seeds': ','.join(str(s) for s in
                                                    invalid) or '-',
                          'source_score_mean': (float(np.mean(scores))
                                                if len(scores) else None)})
        print(table(srows, ('cell', 'condition', 'label', 'n', 'valid',
                            'invalid', 'invalid_seeds', 'source_score_mean')))
        n_inv = sum(r['invalid'] for r in srows)
        n_val = sum(r['valid'] for r in srows)
        print(f'  {n_val} valid source(s), {n_inv} rejected. Rejected source '
              'seeds are listed above, never dropped silently (DESIGN.md '
              '§4.3).')
        if n_val == 0 and n_inv > 0:
            print('  Every source in this dataset fails the validity gate. '
                  'The published study')
            print('  transferred from a source that never learned its task; '
                  'the primary estimand')
            print('  here is defined on valid sources only, so it is EMPTY in '
                  'this dataset.')
            ledger.deviations.append(
                'no source passes the DESIGN.md §4.3 validity gate: the '
                'primary (valid-sources-only) estimand is empty')
        print(f'  analysis set in force: {opts.source_policy!r} '
              f'({"valid sources only, the primary estimand" if opts.source_policy == "valid" else "pooled over source competence -- the pre-declared SECONDARY of DESIGN.md §4.3, never called ITT"})')
        out['source_validity'] = srows

    h2('2e. transferred-parameter fraction (DESIGN.md §3.1)')
    frows = []
    for cell in CELL_ORDER:
        t = primary_transfer_arm(df[df['cell'] == cell], opts)
        if not len(t):
            continue
        reinit = _clean(t.get('reinitialised_layer_count',
                              pd.Series(dtype=float)))
        frows.append({'cell': cell, 'arch': t['arch'].iloc[0],
                      'n': len(t),
                      'transferred_fraction': transferred_fraction(t),
                      'reinit_layers': (float(np.mean(reinit))
                                        if len(reinit) else None),
                      'params_copied': (float(np.mean(_clean(
                          t.get('params_copied', pd.Series(dtype=float)))))
                          if 'params_copied' in t else None)})
    print(table(frows, ('cell', 'arch', 'n', 'transferred_fraction',
                        'reinit_layers', 'params_copied')))
    by_cell = {r['cell']: r['transferred_fraction'] for r in frows}
    out['transferred_fraction'] = by_cell

    gates = []
    for a, b in combinations([r for r in frows], 2):
        if a['arch'] == b['arch']:
            continue
        fa, fb = a['transferred_fraction'], b['transferred_fraction']
        if not (np.isfinite(fa) and np.isfinite(fb)):
            verdict = 'unknown (fraction not recorded)'
            allowed = False
        elif abs(fa - fb) <= INTENSITY_TOLERANCE:
            verdict = 'matched'
            allowed = True
        else:
            verdict = 'CONFOUNDED'
            allowed = bool(opts.allow_intensity_confound)
        gates.append({'a': a['cell'], 'b': b['cell'], 'frac_a': fa,
                      'frac_b': fb,
                      'abs_diff': (abs(fa - fb) if np.isfinite(fa)
                                   and np.isfinite(fb) else None),
                      'verdict': verdict, 'permitted': allowed})
    if gates:
        print()
        print(f'  Cross-architecture intensity gate, tolerance '
              f'{INTENSITY_TOLERANCE}:')
        print(table(gates, ('a', 'b', 'frac_a', 'frac_b', 'abs_diff',
                            'verdict', 'permitted')))
        blocked = [g for g in gates if not g['permitted']]
        if blocked:
            print('  REFUSAL: the cross-architecture contrasts above are '
                  'confounded with')
            print('  treatment intensity -- the same class of error the Phase 0 '
                  'audit found in')
            print('  the published study. They are not computed. Pass '
                  '--allow-intensity-confound')
            print('  to compute them anyway; the override is stamped into the '
                  'output.')
            for g in blocked:
                ledger.refusals.append(
                    f'cross-arch contrast {g["a"]} vs {g["b"]} refused: '
                    f'transferred fractions {fmt(g["frac_a"],3)} vs '
                    f'{fmt(g["frac_b"],3)} differ by more than '
                    f'{INTENSITY_TOLERANCE}')
        elif opts.allow_intensity_confound:
            print('  OVERRIDE IN FORCE: --allow-intensity-confound was passed. '
                  'Every')
            print('  cross-architecture contrast below is labelled '
                  'intensity-confounded.')
            ledger.deviations.append(
                '--allow-intensity-confound override in force: cross-arch '
                'contrasts are intensity-confounded')
    out['intensity_gate'] = gates

    h2('2f. run integrity')
    for col, msg in (('metrics_contiguous',
                      'metrics rows non-contiguous (a resume duplicated or '
                      'lost episodes; DESIGN.md §8.2)'),
                     ('freeze_verified',
                      'freeze verification failed: a frozen layer moved or a '
                      'trainable one did not (DESIGN.md §8.4)')):
        if col not in df.columns:
            print(f'  {col}: column absent')
            continue
        bad = df[df[col] == False]
        print(f'  {col}: {len(df) - len(bad)} ok, {len(bad)} failing')
        if len(bad):
            for _, r in bad.iterrows():
                print(f'    {r["label"]} s{int(r["seed"])}: {msg}')
            ledger.deviations.append(f'{len(bad)} run(s) fail {col}: {msg}')
    return out


def section_descriptives(df: pd.DataFrame, opts: Options, ledger: Ledger,
                         metric: str) -> dict:
    """§10.4 -- per-arm descriptives on the normalised score, with headroom.

    Headroom is printed because it is the residual confound of RQ3
    (`DESIGN.md` §2.5): a cell whose scratch baseline sits near the ceiling has
    less room to gain and more to lose, so a between-cell comparison of deltas
    is partly a comparison of headroom.

    **n here is a count of seeds, and every statistic is computed on one value
    per seed.** It used to be a count of rows: `len(_clean(g[metric]))`. On the
    repo's own `runs_demo` tree, where most arms hold two run directories per
    (label, seed), that printed n=6 for a 3-seed arm; injecting one duplicate
    row at `final_score=99` moved an arm's mean from 1.1635 to 10.0577 and its
    SD from 0.0664 to 29.4989; and doubling an arm wholesale *narrowed* its
    interval, because duplication buys artefactual precision. The caption
    called the interval a "seed-level bootstrap" while `bootstrap_statistic`
    was resampling rows. This is the table `report.py` renders as the paper's
    descriptives, and §3b's headroom feeds RQ3's pre-registered two-scale
    agreement gate, so a row count standing in for a seed count reached both.
    """
    h1(f'3. DESCRIPTIVES on {metric} (normalised score; random policy = 0, '
       'threshold = 1)')
    ledger.est(f'descriptives on {metric}')
    rows = []
    refusals: list[str] = []
    # Grouped by environment as well as by arm. Scores are normalised per
    # environment, so rows from different environments are not comparable and
    # must not be presented as though they were (DESIGN.md §5.1); the env
    # column makes that visible instead of leaving it to be inferred.
    for (env, cell, cond, label), g in df.groupby(
            ['env', 'cell', 'condition', 'label'], dropna=False):
        sv = seed_vector(g, metric, f'{label} arm on {env}')
        if not sv['n'] and not sv['refused']:
            continue
        rec: dict[str, Any] = {
            'env': env, 'cell': cell, 'condition': cond, 'label': label,
            'seed_block': '/'.join(sorted(set(str(b) for b in
                                              g['seed_block'].dropna()))),
            'n': sv['n'], 'n_rows': sv['n_rows'], 'seeds': sv['seeds']}
        if sv['refused']:
            # The arm is ambiguous, so it has no mean. §5 withholds a
            # suppressed member's point estimate at every n and this does the
            # same: the row stays, as the inventory of what was run, and the
            # numbers that cannot be computed are absent rather than wrong.
            rec.update({'mean': None, 'sd': None, 'median': None,
                        'ci_lo': None, 'ci_hi': None, 'min': None,
                        'max': None, 'note': sv['reason']})
            refusals.append(f'{cell}/{label} on {env}: {sv["reason"]}')
            ledger.other_suppressed.append(
                f'descriptives {metric} {cell}/{label}: {sv["reason"]}')
            rows.append(rec)
            continue
        x = sv['values']
        boot = bootstrap_statistic(x, lambda a: float(np.mean(a)),
                                   opts.n_boot, opts.boot_seed, vec=mean_vec)
        rec.update({'mean': float(np.mean(x)), 'sd': sd(x),
                    'median': float(np.median(x)),
                    'ci_lo': boot['lo'], 'ci_hi': boot['hi'],
                    'min': float(np.min(x)), 'max': float(np.max(x)),
                    'note': ('; '.join(
                        f'seed {s} has a run but no {metric} value'
                        for s in sv['metric_missing'])
                        if sv['metric_missing'] else '')})
        rows.append(rec)
    order = {c: i for i, c in enumerate(CELL_ORDER)}
    rows.sort(key=lambda r: (str(r['env']), order.get(r['cell'], 99),
                             str(r['condition']), str(r['label'])))
    print(table(rows, ('env', 'cell', 'condition', 'label', 'seed_block', 'n',
                       'n_rows', 'mean', 'sd', 'median', 'ci_lo', 'ci_hi',
                       'min', 'max', 'note')))
    print('  n is DISTINCT SEEDS; n_rows is run directories. Where they differ '
          'the arm holds')
    print('  more than one run for a seed, and every statistic on that row is '
          'withheld: two')
    print('  runs claiming one seed have no across-seed mean until the '
          'ambiguity is resolved.')
    print('  CI: bias-corrected-and-accelerated seed-level bootstrap on the '
          'mean, resampling')
    print('  seeds. No normality-assuming interval appears anywhere in this '
          'module')
    print('  (ANALYSIS_PLAN.md §8).')
    if refusals:
        print(f'  {len(refusals)} arm(s) carry no estimate:')
        for r in refusals:
            print(f'    {r}')

    h2('3b. scratch baseline, threshold and headroom per cell')
    hrows = []
    for cell in CELL_ORDER:
        # The scratch arm of a cell pools every scratch label at the target
        # environment, so two labels that both ran seed 0 give that seed two
        # values here even when no (label, seed) is duplicated. That is the
        # same ambiguity and gets the same refusal: it is not resolvable by
        # keeping one label's row, and it used to be resolved by keeping
        # whichever row pandas visited last.
        sv = seed_vector(scratch_arm(df[df['cell'] == cell], opts), metric,
                         f'{cell} scratch arm')
        if not sv['n'] and not sv['refused']:
            continue
        rec = {'cell': cell, 'n': sv['n'], 'n_rows': sv['n_rows'],
               'threshold': 1.0}
        if sv['refused']:
            rec.update({'scratch_mean': None, 'scratch_sd': None,
                        'headroom': None, 'note': sv['reason']})
            ledger.other_suppressed.append(
                f'headroom {metric}/{cell}: {sv["reason"]}')
        else:
            m = float(np.mean(sv['values']))
            rec.update({'scratch_mean': m, 'scratch_sd': sd(sv['values']),
                        'headroom': 1.0 - m, 'note': ''})
        hrows.append(rec)
    print(table(hrows, ('cell', 'n', 'n_rows', 'scratch_mean', 'scratch_sd',
                        'threshold', 'headroom', 'note')))
    print('  headroom = 1.0 - scratch mean: what remains between this cell and '
          'the')
    print('  registered solved threshold. A cell with little headroom cannot '
          'gain much and')
    print('  can lose a great deal, which is why every RQ3 statement below '
          'carries it.')
    for r in hrows:
        if r.get('note'):
            print(f'  {r["cell"]}: no headroom is computed. {r["note"]}')
    # The two sentences below are generated between-cell comparisons, and
    # ANALYSIS_PLAN.md §9 forbids a number from fewer than three seeds being
    # "quoted, compared, or used to choose between hypotheses". At n=1 this
    # used to emit "mlp-vanilla headroom is 0.0308 score units above
    # dueling-vanilla headroom" from two single-seed means, which is precisely
    # the comparison §9 names. The table above stays: it is the inventory of
    # what was run, with its n in every row. The generated comparison does not.
    #
    # The gate reads n, and n is now seeds. It used to read a row count, so a
    # 2-seed dataset recorded twice presented as n=4 and both sentences fired
    # from two seeds -- the guard added to enforce §9 was defeated by the same
    # confusion it was added to stop.
    usable = [r for r in hrows if r['n'] >= MIN_N_FOR_INFERENCE
              and r['headroom'] is not None]
    if len(usable) > 1:
        worst = min(usable, key=lambda r: r['headroom'])
        best = max(usable, key=lambda r: r['headroom'])
        print('  ' + phrase_direction(best['headroom'] - worst['headroom'],
                                      f'{best["cell"]} headroom',
                                      f'{worst["cell"]} headroom'))
        sds = {r['cell']: r['scratch_sd'] for r in usable}
        pairs = sorted(sds.items(), key=lambda kv: (kv[1] if
                                                    np.isfinite(kv[1]) else 0))
        if len(pairs) > 1:
            print('  ' + phrase_dispersion(f'{pairs[-1][0]} scratch',
                                           pairs[-1][1],
                                           f'{pairs[0][0]} scratch',
                                           pairs[0][1]))
    elif len(hrows) > 1:
        ambiguous = [r['cell'] for r in hrows if r['headroom'] is None]
        print('  No between-cell headroom or dispersion sentence is emitted: '
              'fewer than two cells')
        print(f'  reach n={MIN_N_FOR_INFERENCE} DISTINCT SEEDS with a '
              'computable headroom, and ANALYSIS_PLAN.md §9')
        print('  forbids comparing numbers computed from fewer than three '
              'seeds. The per-cell')
        print('  values above carry their n.')
        if ambiguous:
            print(f'  {len(ambiguous)} of those cell(s) have no headroom at '
                  f'all rather than too few seeds: {", ".join(ambiguous)}. '
                  'The reason is printed above.')
    return {'per_arm': rows, 'headroom': hrows,
            'arms_refused': refusals,
            'headroom_refused': [r['cell'] for r in hrows
                                 if r.get('note')]}


def section_convergence(df: pd.DataFrame, opts: Options,
                        ledger: Ledger) -> dict:
    """§10.5 -- the convergence gate.

    `DESIGN.md` §5.2: where runs have not converged, P1 is *performance at
    budget*, not asymptotic performance. The plan asks for the fraction of runs
    whose final-window slope is distinguishable from zero; per_seed.csv carries
    the point slope but no per-run standard error, so distinguishability is
    assessed at the arm level (a bootstrap interval on the arm's median slope,
    plus an exact interval on the fraction of positive slopes) and the missing
    column is named. Approximating a per-run SE from a single number would be
    inventing one.
    """
    h1('4. CONVERGENCE GATE')
    ledger.est('convergence gate (arm-level slope intervals)')
    if 'convergence_slope' not in df.columns:
        print('  convergence_slope column absent: the gate cannot be '
              'evaluated, so the')
        print('  word "asymptotic" is not licensed anywhere in the report.')
        ledger.deviations.append('convergence_slope absent: gate not evaluated')
        return {'available': False}
    ledger.tensions.append(
        'ANALYSIS_PLAN.md §10.5 asks for the fraction of runs whose slope is '
        'distinguishable from zero; per_seed.csv carries no per-run slope '
        'standard error, so the gate is evaluated at arm level and the missing '
        'column (convergence_slope_se) is named rather than approximated')
    rows = []
    failing = []
    unevaluable = []
    for (cell, cond, label), g in df.groupby(['cell', 'condition', 'label'],
                                             dropna=False):
        # Seeds, not rows. The Clopper-Pearson interval below is an exact
        # binomial interval whose n is the number of independent units, and
        # two run directories for one seed are not two units: on runs_demo
        # this printed frac_positive = 4/6 and a CP interval on 6 pseudo-units
        # from 3 seeds, and the n<3 floor beside it counted the same 6.
        sv = seed_vector(g, 'convergence_slope', f'{label} arm')
        s = sv['values']
        if not sv['n'] and not sv['refused']:
            continue
        rec: dict[str, Any] = {'cell': cell, 'condition': cond,
                               'label': label, 'n': sv['n'],
                               'n_rows': sv['n_rows']}
        if sv['refused']:
            rec.update({'median_slope': None, 'ci_lo': None, 'ci_hi': None,
                        'frac_positive': None, 'cp_lo': None, 'cp_hi': None,
                        'still_moving': None, 'note': sv['reason']})
            rows.append(rec)
            unevaluable.append(f'{cell}/{label}')
            ledger.other_suppressed.append(
                f'convergence gate {cell}/{label}: {sv["reason"]}')
            continue
        if len(s) < MIN_N_FOR_INFERENCE:
            # ANALYSIS_PLAN.md §9: under n<3, no test and no interval. The
            # exact binomial interval on the fraction of positive slopes is an
            # interval, and the fraction itself is a single-seed number; both
            # used to be printed at n=1 (cp_lo 0.0000, cp_hi 0.9750 per arm).
            # Worse, the bootstrap interval is already suppressed at that n, so
            # `still_moving` came out False for every arm by construction and
            # the section then read the resulting silence as evidence of
            # convergence. The arm is listed with its n and nothing else.
            rec.update({'median_slope': None, 'ci_lo': None, 'ci_hi': None,
                        'frac_positive': None, 'cp_lo': None, 'cp_hi': None,
                        'still_moving': None,
                        'note': f'n={len(s)} < {MIN_N_FOR_INFERENCE}: no '
                                'estimate, no interval, no proportion '
                                '(ANALYSIS_PLAN.md §9)'})
            rows.append(rec)
            unevaluable.append(f'{cell}/{label}')
            ledger.other_suppressed.append(
                f'convergence gate {cell}/{label}: n={len(s)}')
            continue
        boot = bootstrap_statistic(s, lambda a: float(np.median(a)),
                                   opts.n_boot, opts.boot_seed, vec=median_vec)
        k = int(np.sum(s > 0))
        lo, hi = clopper_pearson(k, len(s))
        moving = bool(np.isfinite(boot['lo']) and np.isfinite(boot['hi'])
                      and (boot['lo'] > 0 or boot['hi'] < 0))
        rec.update({'median_slope': boot['estimate'],
                    'ci_lo': boot['lo'], 'ci_hi': boot['hi'],
                    'frac_positive': k / len(s), 'cp_lo': lo, 'cp_hi': hi,
                    'still_moving': moving, 'note': ''})
        rows.append(rec)
        if moving:
            failing.append(f'{cell}/{label}')
    print(table(rows, ('cell', 'condition', 'label', 'n', 'n_rows',
                       'median_slope', 'ci_lo', 'ci_hi', 'frac_positive',
                       'cp_lo', 'cp_hi', 'still_moving', 'note')))
    print('  window: the final-window slope as recorded by train.py '
          '(result.convergence_window_episodes);')
    print('  units are score per episode. "still_moving" means the arm-level '
          'interval excludes zero.')
    print('  n is DISTINCT SEEDS: the exact interval on frac_positive counts '
          'independent units,')
    print('  and two run directories recorded for one seed are one unit, not '
          'two.')
    print()
    if failing:
        print(f'  GATE FAILED for {len(failing)} arm(s): {", ".join(failing)}')
        print('  The final-window slope is distinguishable from zero in these '
              'arms, so the')
        print('  runs were still changing at the budget. The word '
              '"asymptotic" is NOT')
        print('  licensed. P1 must be named "performance at budget" '
              'throughout (DESIGN.md §5.2).')
        ledger.deviations.append(
            f'convergence gate failed in {len(failing)} arm(s): P1 is '
            '"performance at budget", not asymptotic performance')
    elif unevaluable and not [r for r in rows if r['still_moving'] is not None]:
        print(f'  GATE NOT EVALUATED: no arm reaches n = '
              f'{MIN_N_FOR_INFERENCE} distinct seeds with an unambiguous '
              'slope per seed, so no arm')
        print('  has an interval on its final-window slope. The per-arm '
              'reason is in the note column.')
        print('  Silence here is the absence of a measurement, not evidence '
              'of convergence: the')
        print('  previous version printed "consistent with convergence" in '
              'exactly this case,')
        print('  which reads a null positively from single seeds. The word '
              '"asymptotic" is NOT')
        print('  licensed; P1 is "performance at budget" (DESIGN.md §5.2).')
        ledger.deviations.append(
            f'convergence gate not evaluated: all {len(unevaluable)} arm(s) '
            f'are below n={MIN_N_FOR_INFERENCE}, so P1 is "performance at '
            'budget" and the word "asymptotic" is unlicensed')
    else:
        print('  No arm with an estimable interval shows a final-window slope '
              'distinguishable')
        print('  from zero at 95%. That is consistent with convergence but '
              'does not establish it:')
        print('  at this n the interval is wide, so the licensed statement is '
              'the interval above,')
        print('  not "converged".')
        if unevaluable:
            print(f'  {len(unevaluable)} arm(s) carry no interval at all '
                  f'(fewer than {MIN_N_FOR_INFERENCE} distinct seeds, or an '
                  'ambiguous seed) and are')
            print('  outside that statement: ' + ', '.join(unevaluable))
            ledger.deviations.append(
                f'{len(unevaluable)} arm(s) have no estimable slope interval '
                f'(below n={MIN_N_FOR_INFERENCE} seeds, or ambiguous) and the '
                'convergence gate was not evaluated for them')
    return {'available': True, 'per_arm': rows, 'failing': failing,
            'unevaluable': unevaluable}


def _paired_delta(df: pd.DataFrame, cell: str, metric: str, opts: Options
                  ) -> dict:
    """delta = transfer - scratch at matched seeds, for one cell.

    RQ2's estimand. The warrant is *ceteris paribus*, not randomisation: at a
    given seed the two runs share their per-layer initialisation for every
    non-transferred layer, the environment-reset sequence and the evaluation
    seed streams (`DESIGN.md` §8.1). Seed is a blocking factor.
    """
    cdf = df[df['cell'] == cell]
    s = scratch_arm(cdf, opts)
    t = primary_transfer_arm(cdf, opts)
    pair = paired_by_seed(t, s, metric)
    labels = sorted(set(str(x) for x in t['label'].unique()))
    fw = sorted(set(t['freeze_updates'].dropna().unique().tolist()))
    return {'cell': cell, 'metric': metric, 'transfer_labels': labels,
            'freeze_updates_observed': fw, **pair,
            'delta': pair['a'] - pair['b']}


def _confirmatory_member(df: pd.DataFrame, cell: str, metric: str,
                         opts: Options) -> dict:
    """One member of the confirmatory family, computed from one policy's runs.

    Lifted out of `section_confirmatory` unchanged, so that the second leg of
    the `DESIGN.md` 3.3 arbitration is computed by the SAME code as the first
    and a difference between the two legs cannot be an artefact of a second
    implementation. This module already refuses to hold two definitions of one
    number (`verify_primitives_against_statlib` exists for that reason); the
    arbitration is a comparison, so a second definition here would corrupt the
    very thing being compared.

    Returns the record. It carries `suppressed` and no estimate where the arm
    cannot support one, exactly as before: `ANALYSIS_PLAN.md` 9 forbids
    quoting a suppressed member's point estimate, not only its test. The
    caller owns the ledger and the family membership; nothing here touches
    either, which is what keeps the tuned leg out of the family of eight.
    """
    pd_ = _paired_delta(df, cell, metric, opts)
    d = pd_['delta']
    n = len(d)
    rec: dict[str, Any] = {
        'metric': metric, 'cell': cell, 'n': n,
        'seeds': pd_['seeds'],
        # Rows and distinct usable seeds, both, and each named for what
        # it is. `n_transfer_rows` used to hold a seed count.
        'n_transfer_rows': pd_['rows_a'],
        'n_scratch_rows': pd_['rows_b'],
        'n_transfer_seeds': pd_['n_a'],
        'n_scratch_seeds': pd_['n_b'],
        'duplicated_transfer_seeds': pd_['dup_a'],
        'duplicated_scratch_seeds': pd_['dup_b'],
        'seeds_without_metric': pd_['metric_missing'],
        'unpaired_transfer_seeds': pd_['only_a'],
        'unpaired_scratch_seeds': pd_['only_b'],
        'transfer_labels': pd_['transfer_labels'],
        'freeze_updates_observed': pd_['freeze_updates_observed'],
    }
    if len(pd_['transfer_labels']) > 1:
        rec['suppressed'] = (
            'ambiguous primary arm: labels '
            f'{pd_["transfer_labels"]} all match registry.PROTOCOL')
    elif len(pd_['freeze_updates_observed']) > 1:
        rec['suppressed'] = (
            'freeze_updates is not constant across the arm '
            f'({pd_["freeze_updates_observed"]}); DESIGN.md §8.4 '
            'refuses to aggregate runs that differ in an invariant')
    elif pd_['dup_a'] or pd_['dup_b']:
        rec['suppressed'] = '; '.join(
            pairing_problems(pd_, metric)) or 'duplicated rows'
    elif pd_['n_a'] == 0 or pd_['n_b'] == 0:
        # The cause is READ OFF the data rather than asserted. The
        # previous text hard-coded "under --source-policy valid this
        # happens when every source fails the gate", and printed that
        # diagnosis for any empty arm: on a table whose final_score
        # column was entirely NaN, with every source valid, four
        # members were suppressed with a false explanation.
        empty = 'transfer' if pd_['n_a'] == 0 else 'scratch'
        rows_present = (pd_['rows_a'] if empty == 'transfer'
                        else pd_['rows_b'])
        if rows_present == 0:
            why = ('no run in the analysis set in force matches that '
                   'arm at all. Under --source-policy valid this is '
                   'what a cell whose sources all fail the DESIGN.md '
                   '§4.3 gate looks like, but the filter that emptied '
                   'it is named in §2d and in the analysis-set note, '
                   'not guessed at here')
        elif not _clean(df[metric]).size:
            why = (f'the arm has {rows_present} run(s) and the '
                   f'{metric} column is empty for EVERY run in the '
                   'table, so the endpoint itself is missing from '
                   'this dataset')
        else:
            why = (f'the arm has {rows_present} run(s) but no finite '
                   f'{metric} value among them')
        rec['suppressed'] = (
            f'the {empty} arm is empty in the analysis set in force: '
            f'{why}. Nothing is substituted for it')
    elif pd_['only_a'] or pd_['only_b']:
        rec['suppressed'] = (
            f'incomplete arm: seeds {pd_["only_a"]} appear only in '
            f'transfer, {pd_["only_b"]} only in scratch. A partial arm '
            'is refused (DESIGN.md §8.4); no seed is dropped to '
            'rescue the test')
    elif n < MIN_N_FOR_INFERENCE:
        rec['suppressed'] = (
            f'n={n} < {MIN_N_FOR_INFERENCE}: no test and no interval '
            '(ANALYSIS_PLAN.md §9)')
    if 'suppressed' in rec:
        # A suppressed member's point estimate is withheld too, not
        # only its test, and that holds at EVERY n. The version this
        # replaces withheld it only below n=3, so a member refused as
        # an incomplete arm (9 of 10 seeds) or refused because
        # freeze_updates was not constant across the arm still had its
        # mean delta printed in the results table: the silently
        # seed-dropped number quoted after all, and the aggregate that
        # DESIGN.md §8.4 had just refused to compute printed anyway.
        rec.update({'mean_delta': None, 'p_signflip': None})
        return rec

    idx = boot_indices(n, opts.n_boot, opts.boot_seed)
    hl = bootstrap_statistic(d, hodges_lehmann_paired, opts.n_boot,
                             opts.boot_seed, idx=idx, vec=hl_vec)
    sf = sign_flip_test(d, seed=opts.boot_seed)
    try:
        w_stat, w_p = sps.wilcoxon(d, zero_method='wilcox',
                                   alternative='two-sided',
                                   method='exact' if n <= 25 else 'auto')
    except ValueError as exc:                     # all-zero differences
        w_stat, w_p = float('nan'), None
        rec['wilcoxon_note'] = str(exc)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        u_stat, u_p = sps.mannwhitneyu(pd_['a'], pd_['b'],
                                       alternative='two-sided')
    r_pear = correlation('pearson', pd_['a'], pd_['b'])
    r_spear = correlation('spearman', pd_['a'], pd_['b'])
    rec.update({
        'mean_delta': float(np.mean(d)),
        'median_delta': float(np.median(d)),
        'sd_delta': sd(d),
        'transfer_mean': float(np.mean(pd_['a'])),
        'scratch_mean': float(np.mean(pd_['b'])),
        'hl': hl['estimate'], 'ci_lo': hl['lo'], 'ci_hi': hl['hi'],
        'ci_method': hl['method'], 'ci_note': hl.get('note', ''),
        'degenerate_interval': bool(hl.get('degenerate')),
        # The paired values themselves, so §10's unpaired sigma is
        # computed on the same seeds the test used rather than on the
        # whole arm.
        'transfer_values': [float(v) for v in pd_['a']],
        'scratch_values': [float(v) for v in pd_['b']],
        'p_signflip': sf['p'], 'signflip_mode': sf['mode'],
        'min_attainable_p': sf['min_attainable_p'],
        'wilcoxon_W': float(w_stat), 'p_wilcoxon': w_p,
        'mannwhitney_U': float(u_stat), 'p_mannwhitney': float(u_p),
        'rho_pearson': r_pear, 'rho_spearman': r_spear,
        'unanimous': phrase_unanimity(d),
        'deltas': [float(x) for x in d],
    })
    return rec


def _member_conclusion(rec) -> str:
    """What one policy's leg concludes about one RQ2 member.

    Direction is part of the conclusion, not decoration. Two policies that
    both reject while pointing opposite ways have not agreed about anything,
    and a verdict computed from the reject / do-not-reject bit alone would
    call that agreement and license the sentence.
    """
    if not rec or 'suppressed' in rec or rec.get('p_holm') is None:
        return CONCLUSION_NONE
    if not rec.get('significant_holm'):
        return CONCLUSION_NULL
    for name in ('hl', 'mean_delta'):
        value = rec.get(name)
        if value is None:
            continue
        f = float(value)
        if np.isfinite(f) and f != 0.0:
            return CONCLUSION_UP if f > 0.0 else CONCLUSION_DOWN
    # Rejected with a location estimate of exactly zero. The sign-flip test
    # cannot produce that, so it means the estimate is degenerate; the
    # conclusion has no direction, and an undirected rejection is not
    # something the other leg can be compared against.
    return CONCLUSION_NONE


CONCLUSION_TEXT: dict[str, str] = {
    CONCLUSION_UP: 'transfer ABOVE scratch, Holm-significant',
    CONCLUSION_DOWN: 'transfer BELOW scratch, Holm-significant',
    CONCLUSION_NULL: 'not distinguishable from zero',
    CONCLUSION_NONE: 'no conclusion: the member carries no test',
}


def _arbitrate(common: str, tuned: str, evaluable: bool,
               reason: str) -> tuple[str, str]:
    """One verdict, from the two legs' conclusions. Returns (verdict, why)."""
    if not evaluable:
        return NOT_EVALUABLE, reason
    if tuned == CONCLUSION_NONE:
        return NOT_EVALUABLE, ('the tuned arms exist but the member is '
                               'suppressed under the secondary policy, so '
                               'the second leg has no conclusion to compare')
    if common == CONCLUSION_NONE:
        return NOT_EVALUABLE, ('the common-policy member is suppressed, so '
                               'there is no first-leg conclusion to arbitrate')
    if common == tuned:
        return AGREES, ''
    return DISAGREES, (f'the common configuration concludes '
                       f'{CONCLUSION_TEXT[common]}; the per-cell tuned '
                       f'configuration concludes {CONCLUSION_TEXT[tuned]}')


def section_rq2_arbitration(members: list[dict], metrics: Sequence[str],
                            tuned: Optional[TunedPolicy], opts: Options,
                            ledger: Ledger) -> dict:
    """5d -- the `DESIGN.md` 3.3 arbitration on RQ2, and the assertion gate.

    The second leg is the same eight hypotheses recomputed on the per-cell
    tuned arms through `_confirmatory_member`, adjusted by Holm over the same
    pre-registered family size of eight. It is a *replication of the family
    under the other declared policy*, not an extension of it: the assertion
    rule is the conjunction "both legs reject", whose rejection region is the
    intersection of the two legs' regions and therefore never larger than
    either, so the family-wise error rate stays bounded by the Holm-over-8 of
    a single leg. Adding these to the ledger as eight further members would
    correct twice for a procedure that is already at most as liberal as one
    leg, and would change the family size `ANALYSIS_PLAN.md` 7 fixes before
    launch.

    With no tuned runs -- today's state, because `E3` has not finished and no
    selection exists -- every verdict is `not-evaluable` and NOTHING is
    asserted. That is the point: before this section existed, a Holm-adjusted
    p below 0.05 in 5a was the end of the matter, and the second leg of the
    pre-registered condition was absent from the code as well as from the
    data.
    """
    tuned = tuned if tuned is not None else TunedPolicy()
    h2('5d. THE DESIGN.md 3.3 ARBITRATION -- may any of the eight be '
       'asserted?')
    print('  DESIGN.md 3.3 declares two hyperparameter policies and asserts '
          'an RQ2 or RQ3')
    print('  conclusion ONLY where both hold. 5a is the FIRST leg: the common '
          'configuration,')
    print('  one lr and target-update rule for all four cells, fixed a '
          'priori. The SECOND leg')
    print('  is the same eight hypotheses on each cell\'s own E3-selected '
          'configuration')
    print('  (registry E1t/E2t), computed by the same code at the same '
          'CONFIRM seeds.')
    print()
    print(f'  The family stays {CONFIRMATORY_FAMILY_SIZE}. The arbitration is '
          'a CONJUNCTION over the same eight')
    print('  hypotheses, not sixteen tests: "both legs reject" has as its '
          'rejection region the')
    print('  INTERSECTION of the two legs\', which is never larger than '
          'either, so the FWER is')
    print(f'  bounded by Holm over {CONFIRMATORY_FAMILY_SIZE} on one leg. '
          'Nothing below is a member of the family,')
    print('  nothing below enters the multiplicity ledger as one, and 11 '
          'asserts that count.')

    # -- where the second leg's runs are, cell by cell ----------------------
    print()
    print('  the secondary policy\'s runs:')
    cell_rows = []
    for cell in CELL_ORDER:
        entry = tuned.cells.get(cell, {})
        cell_rows.append({
            'cell': cell,
            'tuned_arms': entry.get('state', 'absent'),
            'runs': entry.get('rows', 0),
            'trained_at': entry.get('config_text', '-'),
            'selected': entry.get('selected', '-'),
            'shares_common_runs': entry.get('shares_common_runs', False),
        })
    print(table(cell_rows, ('cell', 'tuned_arms', 'runs', 'trained_at',
                            'selected', 'shares_common_runs')))
    if tuned.selection is not None:
        sel = tuned.selection
        print(f'  selection           : {sel.short_id}  rule '
              f'{(sel.rule or {}).get("id")!r}  block {sel.seed_block} '
              f'seeds {list(sel.seeds)}')
    else:
        print('  selection           : NONE READABLE')
        for line in str(tuned.selection_note).splitlines():
            print(f'    {line}')
    for note in tuned.notes:
        print(f'  NOTE: {note}')
    print('  A cell whose selection equals the a priori configuration '
          'produces run digests')
    print('  identical to its common-policy arms and SHARES those run '
          'directories, so its two')
    print('  legs are the same runs and agree by construction. That is a '
          'fact about the')
    print('  selection, not a replication, and the table above says which '
          'cells it is true of.')

    # -- the second leg ----------------------------------------------------
    tuned_members: list[dict] = []
    tuned_pvals: dict[tuple[str, str], float] = {}
    for metric in metrics:
        for cell in CELL_ORDER:
            state = tuned.state(cell)
            if state == 'absent':
                rec = {'metric': metric, 'cell': cell, 'policy': POLICY_TUNED,
                       'n': 0, 'mean_delta': None, 'p_signflip': None,
                       'suppressed': tuned.reason(cell)
                       or 'no tuned run for this cell'}
            else:
                rec = _confirmatory_member(tuned.cell_frame(cell), cell,
                                           metric, opts)
                rec['policy'] = POLICY_TUNED
                rec['runs_shared_with_common'] = bool(state == 'shared')
            tuned_members.append(rec)
            if rec.get('p_signflip') is not None:
                tuned_pvals[(metric, cell)] = float(rec['p_signflip'])
    adj = holm_adjust(tuned_pvals, CONFIRMATORY_FAMILY_SIZE)
    for rec in tuned_members:
        key = (rec['metric'], rec['cell'])
        rec['p_holm'] = adj.get(key)
        rec['significant_holm'] = (rec['p_holm'] is not None
                                   and rec['p_holm'] < ALPHA)

    common_by_key = {(m['metric'], m['cell']): m for m in members}
    tuned_by_key = {(m['metric'], m['cell']): m for m in tuned_members}

    rows: list[dict] = []
    for metric in metrics:
        for cell in CELL_ORDER:
            key = (metric, cell)
            cm = common_by_key.get(key)
            tm = tuned_by_key.get(key)
            state = tuned.state(cell)
            c_conc = _member_conclusion(cm)
            t_conc = _member_conclusion(tm)
            verdict, why = _arbitrate(
                c_conc, t_conc, state != 'absent', tuned.reason(cell)
                or 'the secondary policy has not been run for this cell')
            blocked = ''
            if verdict == AGREES and tuned.assertion_block:
                blocked = tuned.assertion_block
            assertable = bool(verdict == AGREES and not blocked
                              and c_conc != CONCLUSION_NONE)
            row = {
                'metric': metric, 'cell': cell,
                'tuned_arms': state,
                'shares_common_runs': bool(state == 'shared'),
                'n_common': (cm or {}).get('n'),
                'n_tuned': (tm or {}).get('n'),
                'hl_common': (cm or {}).get('hl'),
                'hl_tuned': (tm or {}).get('hl'),
                'holm_rejects_common': bool((cm or {}).get(
                    'significant_holm')),
                'holm_rejects_tuned': bool((tm or {}).get(
                    'significant_holm')),
                'conclusion_common': c_conc,
                'conclusion_tuned': t_conc,
                'verdict': verdict,
                'assertable': assertable,
                'why': why or blocked,
            }
            rows.append(row)
            ledger.arbitration.append(
                f'RQ2 {metric}/{cell}: {verdict}'
                + (f' ({row["why"]})' if row['why'] else ''))

    print()
    print(table(rows, ('metric', 'cell', 'tuned_arms', 'n_common', 'n_tuned',
                       'hl_common', 'hl_tuned', 'conclusion_common',
                       'conclusion_tuned', 'verdict', 'assertable')))

    print('  The tuned leg is Holm-adjusted over the same pre-registered '
          'family size even where')
    print('  fewer than that many of its members exist, exactly as the '
          'common leg is. A leg with')
    print('  fewer evaluable members is therefore corrected MORE strictly, '
          'which is the')
    print('  conservative direction and keeps a missing tuned arm from '
          'making the second leg')
    print('  easier to satisfy than the first.')

    n_asserted = sum(1 for r in rows if r['assertable'])
    n_dis = sum(1 for r in rows if r['verdict'] == DISAGREES)
    n_ne = sum(1 for r in rows if r['verdict'] == NOT_EVALUABLE)
    print()
    print(f'  assertable: {n_asserted} of {len(rows)}   disagreements: '
          f'{n_dis}   not evaluable: {n_ne}')

    # A disagreement is a FINDING, and is printed as one before anything else
    # in this subsection. It is not averaged with the agreeing cells, not
    # resolved towards the primary policy, and not left implicit in a column.
    if n_dis:
        print()
        print('  ' + '*' * 70)
        print('  THE TWO POLICIES DISAGREE. Under DESIGN.md 3.3 that '
              'disagreement IS the')
        print('  finding and is reported as one. No conclusion is asserted '
              'for these members,')
        print('  and neither leg is preferred over the other:')
        for r in rows:
            if r['verdict'] != DISAGREES:
                continue
            print(f'    {r["metric"]}/{r["cell"]}: {r["why"]}.')
            print(f'      common HL {fmt(r["hl_common"])}, tuned HL '
                  f'{fmt(r["hl_tuned"])}')
            ledger.refusals.append(
                f'RQ2 {r["metric"]}/{r["cell"]}: the common and per-cell '
                f'tuned policies DISAGREE, so DESIGN.md 3.3 asserts no '
                f'conclusion; the disagreement is the finding')
        print('  ' + '*' * 70)

    if n_ne == len(rows) and rows:
        print()
        print('  ' + '*' * 70)
        print('  NO RQ2 CONCLUSION IS ASSERTED IN THIS REPORT.')
        print('  The per-cell tuned policy has no runs in the analysis set, '
              'so the second leg')
        print('  of the pre-registered arbitration cannot be evaluated. '
              'DESIGN.md 3.3 makes an')
        print('  RQ2 conclusion assertable only where BOTH policies hold, so '
              'a Holm-adjusted p')
        print('  below 0.05 in 5a is a result under the common configuration '
              'and NOT a')
        print('  confirmatory conclusion of this study. To make it one, run '
              'the tuned stage:')
        for line in str(tuned.selection_note or '').splitlines():
            print(f'    {line}')
        if not tuned.selection_note:
            print('    python experiments/sweep.py --experiments E1t E2t')
        print('  ' + '*' * 70)
        ledger.refusals.append(
            f'RQ2: all {len(rows)} member(s) are not-evaluable under the '
            f'DESIGN.md 3.3 arbitration because the per-cell tuned policy has '
            f'no runs in the analysis set; no RQ2 conclusion is asserted')
    elif n_ne:
        print()
        for r in rows:
            if r['verdict'] != NOT_EVALUABLE:
                continue
            print(f'  {r["metric"]}/{r["cell"]}: not evaluable -- '
                  f'{r["why"]}. No conclusion is asserted.')
            ledger.refusals.append(
                f'RQ2 {r["metric"]}/{r["cell"]}: the DESIGN.md 3.3 '
                f'arbitration is not evaluable ({r["why"]}), so no conclusion '
                f'is asserted')

    if tuned.assertion_block and tuned.available:
        print()
        print('  ASSERTION BLOCKED FOR EVERY MEMBER, whatever the verdict '
              'above:')
        print(f'    {tuned.assertion_block}.')
        ledger.deviations.append(
            f'DESIGN.md 3.3 arbitration: {tuned.assertion_block}, so no RQ2 '
            f'or RQ3 conclusion is asserted from the tuned leg in this report')

    if n_asserted:
        print()
        print('  ASSERTABLE under DESIGN.md 3.3 (both policies hold):')
        for r in rows:
            if not r['assertable']:
                continue
            print(f'    {r["metric"]}/{r["cell"]}: '
                  f'{CONCLUSION_TEXT[r["conclusion_common"]]}'
                  + ('  [the two legs are the same runs: this cell\'s '
                     'selection equals the a priori configuration]'
                     if r['shares_common_runs'] else ''))

    sel_info: dict[str, Any] = {
        'policy': TUNED_POLICY_NAME,
        'readable': tuned.selection is not None,
        'note': tuned.selection_note,
        'placeholder': bool(tuned.placeholder),
        'assertion_block': tuned.assertion_block,
    }
    if tuned.selection is not None:
        sel_info.update({
            'selection_id': tuned.selection.selection_id,
            'short_id': tuned.selection.short_id,
            'rule': dict(tuned.selection.rule or {}),
            'seed_block': tuned.selection.seed_block,
            'seeds': list(tuned.selection.seeds),
            'env': tuned.selection.env,
            'shared_cells': list(tuned.selection.shared_cells),
        })
    return {'rows': rows, 'tuned_members': tuned_members,
            'cells': cell_rows, 'selection': sel_info,
            'family_size': CONFIRMATORY_FAMILY_SIZE,
            'adds_family_members': 0,
            'notes': list(tuned.notes),
            'counts': {'assertable': n_asserted, 'disagrees': n_dis,
                       'not_evaluable': n_ne, 'members': len(rows)}}


def section_confirmatory(df: pd.DataFrame, opts: Options,
                         ledger: Ledger,
                         tuned: Optional[TunedPolicy] = None) -> dict:
    """§10.6 -- THE confirmatory family: exactly 8 tests, and nothing else.

    `ANALYSIS_PLAN.md` §2. Membership is fixed by the plan, computed from this
    module's constants, and never taken from an argument -- which is what stops
    a result being rescued by relocating it into a family of one.
    """
    h1('5. THE CONFIRMATORY FAMILY -- 4 cells x 2 co-primary endpoints = '
       f'{CONFIRMATORY_FAMILY_SIZE} tests')
    for m in CONFIRMATORY_ENDPOINTS:
        require_confirmatory(m)
    print('  Family (pre-registered, ANALYSIS_PLAN.md §2):')
    print(f'    endpoints : {", ".join(CONFIRMATORY_ENDPOINTS)}')
    print(f'    cells     : {", ".join(CELL_ORDER)}')
    print('    contrast  : delta = transfer - scratch, within cell, at '
          'matched seeds')
    print('    primary   : exact sign-flip randomisation test, statistic = '
          'the mean delta')
    print(f'    correction: Holm-Bonferroni over '
          f'{CONFIRMATORY_FAMILY_SIZE}; strictest step alpha = '
          f'{ALPHA_STRICTEST:.5f}')
    computed = [m for m in CONFIRMATORY_ENDPOINTS if m in opts.metrics]
    if set(computed) != set(CONFIRMATORY_ENDPOINTS):
        print(f'  NOTE: --metric restricted this run to {computed}. The family '
              f'size stays {CONFIRMATORY_FAMILY_SIZE} by pre-registration, so '
              'the adjustment is unchanged.')
        ledger.deviations.append(
            f'--metric restricted the computed family to {computed}; Holm '
            f'still applied over {CONFIRMATORY_FAMILY_SIZE}')

    members: list[dict] = []
    pvals: dict[tuple[str, str], float] = {}
    for metric in computed:
        endpoint_absent = (metric not in df.columns
                           or not _clean(df[metric]).size)
        if endpoint_absent:
            # A co-primary endpoint with no value anywhere is not four
            # suppressed cells with a local explanation: it is half the
            # confirmatory family missing from the dataset, and §12 is where
            # that belongs.
            ledger.deviations.append(
                f'co-primary endpoint {metric!r} has no finite value in any '
                f'run of the analysis set, so {len(CELL_ORDER)} of '
                f'{CONFIRMATORY_FAMILY_SIZE} confirmatory members do not '
                'exist in this dataset (DESIGN.md §5.2 declares it '
                'co-primary)')
        for cell in CELL_ORDER:
            rec = _confirmatory_member(df, cell, metric, opts)
            rec['policy'] = POLICY_COMMON
            members.append(rec)
            if 'suppressed' in rec:
                ledger.suppressed.append(
                    f'{metric}/{cell}: {rec["suppressed"]}')
                continue
            pvals[(metric, cell)] = float(rec['p_signflip'])
            ledger.confirmatory.append(f'{metric}/{cell} sign-flip')

    adj = holm_adjust(pvals, CONFIRMATORY_FAMILY_SIZE)
    for rec in members:
        key = (rec['metric'], rec['cell'])
        rec['p_holm'] = adj.get(key)
        rec['significant_holm'] = (rec['p_holm'] is not None
                                   and rec['p_holm'] < ALPHA)

    h2('5a. the eight tests')
    print('  A Holm-adjusted p below alpha in this table is the verdict '
          'under the COMMON')
    print('  configuration ALONE. DESIGN.md 3.3 asserts an RQ2 conclusion '
          'only where the')
    print('  per-cell tuned policy agrees; 5d is that arbitration and is '
          'what says whether')
    print('  any of these may be asserted. Read 5a with 5d or not at all.')
    print()
    print(table(members, ('metric', 'cell', 'n', 'scratch_mean',
                          'transfer_mean', 'mean_delta', 'hl', 'ci_lo',
                          'ci_hi', 'ci_method', 'p_signflip', 'p_holm',
                          'p_wilcoxon', 'mannwhitney_U', 'p_mannwhitney',
                          'rho_pearson', 'rho_spearman')))
    # The interval's method travels with the interval. §3 states the CI is BCa;
    # when the bias fraction lands on a boundary the estimate falls back to the
    # percentile interval, and that fallback used to be invisible here, so
    # eight percentile intervals could be read as eight BCa ones.
    fell_back = [r for r in members
                 if r.get('ci_method') not in (None, 'BCa')
                 and 'suppressed' not in r]
    if fell_back:
        print()
        print(f'  {len(fell_back)} of {len(members)} interval(s) are NOT BCa. '
              'The method column above')
        print('  carries each one, and the reason follows:')
        for r in fell_back:
            print(f'    {r["metric"]}/{r["cell"]}: {r["ci_method"]} -- '
                  f'{r.get("ci_note") or "no note recorded"}')
    n_sup = sum(1 for r in members if 'suppressed' in r)
    if n_sup:
        print()
        print(f'  {n_sup} of {len(members)} members suppressed:')
        for r in members:
            if 'suppressed' in r:
                print(f'    {r["metric"]}/{r["cell"]}: {r["suppressed"]}')

    h2('5b. interpretation rule, stated before the numbers were seen')
    observed = [r['n'] for r in members if r.get('p_signflip') is not None]
    if observed:
        example_n = max(observed)
        source_of_n = 'observed'
    else:
        example_n = len(registry.SEED_BLOCKS['CONFIRM'])
        source_of_n = 'planned'
        print(f'  No member produced a test, so the rule is stated at the '
              f'PLANNED n={example_n} (the')
        print('  CONFIRM block), not at any n present in this dataset.')
    print(f'  At the {source_of_n} n={example_n} the exact sign-flip test '
          'cannot return a two-sided p below')
    print(f'  2/2^{example_n} = {2 / 2 ** example_n:.5f}, attained exactly '
          'when every seed moves the same way.')
    strict = ALPHA_STRICTEST
    # The exact two-sided p is k/2^n for an EVEN k: a sign assignment and its
    # negation are both at least as extreme as each other, so the attainable
    # values step by 2/2^n. `k_max` is the largest even k with k/2^n strictly
    # below the strictest Holm step, which is the same comparison
    # `significant_holm` makes on the adjusted p.
    total = 2 ** example_n
    k_max = int(math.ceil(strict * total - 1e-12)) - 1
    k_max = min(k_max - (k_max % 2), total)
    print(f'  The strictest Holm step is alpha = {strict:.5f}.')
    if k_max >= 2:
        vals = ', '.join(f'{k / total:.5f}' for k in range(2, k_max + 1, 2))
        # ANALYSIS_PLAN.md 2.2 stated this as an "if and only if" and so did
        # this section, and both were FALSE. Unanimity attains the FLOOR of the
        # attainable p-values; the BAR is the Holm step, and at n=10 three
        # attainable values sit below it. sign_flip_test([1.0]*9 + [-0.01])
        # returns p = 0.00391 with the seeds split 9 to 1, which clears 0.00625
        # and is Holm-significant in the table above, against a printed rule
        # saying it cannot be. The plan's sentence was corrected on 2026-08-26
        # and the correction is logged in ANALYSIS_PLAN.md 11.
        print(f'  The exact p moves in units of 2/{total}, because a sign '
              'assignment and its negation')
        print('  are always equally extreme, so the attainable values '
              'strictly below that alpha are')
        print(f'  {vals}: {k_max // 2} distinct outcome(s) clear it, not one.')
        print(f'  THE BAR IS THEREFORE: at most {k_max} of the {total} sign '
              'assignments may be at least')
        print('  as extreme as the observed mean (ANALYSIS_PLAN.md §2.2). '
              'Unanimity attains the')
        print(f'  floor {2 / total:.5f} and so is SUFFICIENT, but it is NOT '
              'NECESSARY: one seed moving')
        print('  against the rest by a small enough margin leaves only 4 '
              'assignments at least as')
        print(f'  extreme, i.e. p = {4 / total:.5f}, which still clears '
              f'{strict:.5f}.')
        print(f'  That bar applies to the SMALLEST p in the family of '
              f'{CONFIRMATORY_FAMILY_SIZE}. Holm compares the jth')
        print('  smallest against alpha/(m-j+1), so every later step is '
              'looser still, and the')
        print(f'  verdict in the table above is p_holm < {ALPHA:.2f} at '
              'whichever step the member')
        print('  lands on. No cell is confirmed on unanimity as such, and '
              'none is refused for')
        print('  the want of it.')
    else:
        print(f'  Therefore NO result at n={example_n} can clear the '
              'corrected threshold: the')
        print(f'  smallest attainable p ({2 / total:.5f}) exceeds '
              f'{strict:.5f}. The tests below')
        print('  are reported for completeness and CANNOT be significant. '
              'This is a property')
        print('  of the sample size, known in advance, not of the data.')

    h2('5c. pairing diagnostics and directional statements')
    for rec in members:
        tag = f'{rec["metric"]}/{rec["cell"]}'
        if 'suppressed' in rec:
            print(f'  {tag}: suppressed -- {rec["suppressed"]}')
            continue
        print(f'  {tag}:')
        print('    ' + phrase_direction(rec['mean_delta'],
                                        'the transfer arm',
                                        "its own cell's scratch baseline"))
        print('    ' + phrase_interval_verdict(rec['ci_lo'], rec['ci_hi'],
                                               'the paired shift (HL)'))
        print('    ' + rec['unanimous'])
        # The decision rule runs on SPEARMAN. Both coefficients are
        # reported, as §2.1 requires, but the branch cannot be driven by a
        # moment-based correlation on data the design declares non-normal
        # (DESIGN.md §5.3, ANALYSIS_PLAN.md §4): Pearson on ten bimodal
        # LunarLander scores is exactly the parametric summary §8 forbids
        # elsewhere in this module. §2.1 does not name a coefficient, so the
        # choice is made here, once, and stated.
        rho = rec['rho_spearman']
        if not np.isfinite(rho):
            print('    rho: not estimable in this cell (an arm has no '
                  'variation across seeds, or')
            print('    fewer than three pairs). The pairing cannot be checked '
                  'here, so no claim')
            print('    about it is made in either direction.')
        elif rho < 0:
            print(f'    Spearman rho = {rho:+.3f} < 0 (Pearson '
                  f'{rec["rho_pearson"]:+.3f}): the matched-seed pairing does '
                  'NOT hold in this cell.')
            print('    ANALYSIS_PLAN.md §2.1 pre-commits to giving the '
                  'unpaired result equal')
            print(f'    prominence here: Mann-Whitney U = '
                  f'{rec["mannwhitney_U"]:.1f}, p = '
                  f'{rec["p_mannwhitney"]:.5f}.')
        else:
            print(f'    Spearman rho = {rho:+.3f} (Pearson '
                  f'{rec["rho_pearson"]:+.3f}): reported whatever their '
                  'values; the paired test')
            print('    stays primary by pre-registration, not by comparison '
                  'of p-values.')
        agree = [('sign-flip', rec['p_signflip']),
                 ('Wilcoxon', rec['p_wilcoxon']),
                 ('Mann-Whitney', rec['p_mannwhitney'])]
        verdicts = {name: (p is not None and p < ALPHA) for name, p in agree}
        if len(set(verdicts.values())) == 1:
            print(f'    all three tests agree at the nominal alpha '
                  f'({"reject" if list(verdicts.values())[0] else "do not reject"}); '
                  'the Holm-corrected verdict is what counts.')
        else:
            dis = ', '.join(f'{k}={"reject" if v else "no"}'
                            for k, v in verdicts.items())
            print(f'    the three tests DISAGREE at the nominal alpha ({dis}); '
                  'that disagreement is')
            print('    the finding, and the pre-registered primary is the '
                  'sign-flip test.')

    # DESIGN.md 3.3's arbitration, and the reason it is HERE rather than in a
    # section of its own: it is the gate on everything 5a prints. A reader who
    # stopped at the table would otherwise take a Holm-adjusted p below 0.05
    # as this study's confirmatory conclusion, which it is not until the
    # second policy agrees.
    arb = section_rq2_arbitration(members, computed, tuned, opts, ledger)
    by_key = {(r['metric'], r['cell']): r for r in arb['rows']}
    for rec in members:
        row = by_key.get((rec['metric'], rec['cell']), {})
        rec['arbitration_verdict'] = row.get('verdict', NOT_EVALUABLE)
        # `significant_holm` keeps its meaning exactly: the common policy's
        # Holm verdict. `asserted` is the new and narrower thing, and it is
        # what a conclusion may be drawn from.
        rec['asserted'] = bool(row.get('assertable'))
    return {'members': members, 'family_size': CONFIRMATORY_FAMILY_SIZE,
            'alpha': ALPHA, 'alpha_strictest': ALPHA_STRICTEST,
            'arbitration': arb}


def section_equivalence(conf: dict, df: pd.DataFrame, opts: Options,
                        ledger: Ledger) -> dict:
    """§10.7 -- equivalence or, far more often, the exclusion bound.

    `ANALYSIS_PLAN.md` §4. Not TOST: the interval already reported is checked
    for containment in the margin, and a cell whose across-seed dispersion
    exceeds the margin is declared untestable rather than given a verdict. The
    exclusion bound is always printed, because it is the only powered,
    directional claim available at this sample size.
    """
    h1('6. EQUIVALENCE AND EXCLUSION')
    ledger.est('equivalence / exclusion assessment')
    margin = EQUIVALENCE_MARGIN
    try:
        pts = abs(envs.denormalise_score(opts.target_env, margin)
                  - envs.denormalise_score(opts.target_env, 0.0))
        margin_note = f'{margin} score units = {pts:.1f} return points on {opts.target_env}'
    except (KeyError, ValueError):
        margin_note = f'{margin} score units'
    print(f'  Margin, fixed in ANALYSIS_PLAN.md §4 before any data: '
          f'+/-{margin_note}.')
    print('  Procedure: containment of the 95% bootstrap CI on the paired '
          'delta. No new')
    print('  test, no new error budget, and no TOST -- which would be '
          'parametric on data')
    print('  the design declares non-normal, at an n that cannot support a '
          'small margin.')
    rows = []
    for rec in conf['members']:
        cell, metric = rec['cell'], rec['metric']
        cdf = df[df['cell'] == cell]
        sd_scratch = sd(_clean(scratch_arm(cdf, opts)[metric]))
        sd_transfer = sd(_clean(primary_transfer_arm(cdf, opts)[metric]))
        worst_sd = max([v for v in (sd_scratch, sd_transfer)
                        if np.isfinite(v)], default=float('nan'))
        lo, hi = rec.get('ci_lo', float('nan')), rec.get('ci_hi', float('nan'))
        degenerate = bool(rec.get('degenerate_interval')
                          or (np.isfinite(lo) and np.isfinite(hi)
                              and hi - lo <= 0.0))
        if 'suppressed' in rec:
            verdict = 'suppressed'
            reason = rec['suppressed']
        elif not (np.isfinite(lo) and np.isfinite(hi)):
            verdict = 'no interval'
            reason = 'no interval emitted'
        elif degenerate:
            # A zero-width interval is not a precise answer. It is what a
            # constant arm, or a transfer arm equal to its scratch arm at every
            # seed, produces: the resampling unit has no variation, so nothing
            # about sampling uncertainty is estimable. This branch is above the
            # containment check because [+0.0000, +0.0000] lies inside any
            # margin and used to be reported as EQUIVALENT, which is the
            # affirmed null this module exists to prevent.
            verdict = 'DEGENERATE'
            reason = (f'the interval [{lo:+.4f}, {hi:+.4f}] has zero width: '
                      'every bootstrap replicate is identical, so the arm has '
                      'no across-seed variation and no equivalence or '
                      'exclusion claim is available at all')
        elif lo >= margin or hi <= -margin:
            # NON-equivalence, and the dispersion gate does not govern it.
            # ANALYSIS_PLAN.md §4's feasibility rule limits *equivalence*
            # claims in a dispersed cell; an interval lying wholly beyond the
            # margin is a difference, and calling it "untestable" (as this
            # section did, for a CI of [-0.1564, -0.0844] against a margin of
            # 0.05) discards a finding the data supports.
            verdict = 'DIFFERENT'
            reason = (f'the whole interval [{lo:+.4f}, {hi:+.4f}] lies outside '
                      f'+/-{margin}')
        elif np.isfinite(worst_sd) and worst_sd > margin:
            verdict = 'UNTESTABLE'
            reason = (f'across-seed SD {worst_sd:.4f} exceeds the margin '
                      f'{margin}: an EQUIVALENCE claim is untestable in this '
                      f'cell at n={rec["n"]} (ANALYSIS_PLAN.md §4). The '
                      'exclusion bound below is unaffected')
        elif lo > -margin and hi < margin:
            verdict = 'EQUIVALENT'
            reason = (f'the whole interval [{lo:+.4f}, {hi:+.4f}] lies inside '
                      f'+/-{margin}')
        else:
            verdict = 'INCONCLUSIVE'
            reason = (f'the interval [{lo:+.4f}, {hi:+.4f}] straddles the '
                      'margin boundary')
        # A suppressed member has no interval, so it has no exclusion bound.
        # Mapping its NaN to 0.0 put a machine-readable "nothing worse than
        # zero is excluded" into the JSON for every single-seed cell, and
        # report.py renders that column as "worse than X excluded".
        bound = (abs(lo) if (np.isfinite(lo) and lo < 0 and not degenerate)
                 else (0.0 if (np.isfinite(lo) and not degenerate) else None))
        rows.append({'metric': metric, 'cell': cell, 'n': rec['n'],
                     'ci_lo': lo, 'ci_hi': hi, 'sd_scratch': sd_scratch,
                     'sd_transfer': sd_transfer, 'margin': margin,
                     'verdict': verdict, 'degenerate': degenerate,
                     'exclusion_bound': bound,
                     'reason': reason})
    print()
    print(table(rows, ('metric', 'cell', 'n', 'ci_lo', 'ci_hi', 'sd_scratch',
                       'sd_transfer', 'margin', 'verdict')))
    print()
    for r in rows:
        print(f'  {r["metric"]}/{r["cell"]}: {r["verdict"]} -- {r["reason"]}')
        if r['degenerate']:
            print('    no exclusion bound: a zero-width interval excludes '
                  'nothing at 95%, it')
            print('    reports that nothing was estimable.')
        else:
            print('    ' + phrase_exclusion_bound(r['ci_lo']))
    print()
    print('  The exclusion bound is printed for every cell whatever the '
          'verdict, because it')
    print('  is the licensed positive statement: a null is never evidence of '
          'equivalence')
    print('  (DESIGN.md §9, STANDING_INSTRUCTIONS S2).')
    return {'margin': margin, 'rows': rows}


# ---------------------------------------------------------------------------
# The control set. `DESIGN.md` §4: contrasts are named after WHAT WAS
# MANIPULATED, never after a mechanism.
# ---------------------------------------------------------------------------

#: The conditions of `DESIGN.md` §4, in declaration order. Named once so that
#: the per-seed reduction, the present/absent accounting and `validate.py`'s
#: "every control is accounted for" check all read the same list.
CONTROL_CONDITIONS: tuple[str, ...] = ('C0', 'C1', 'C2', 'C2K0', 'C3', 'C3b')

#: (key, description, selector) for the per-seed condition vector.
CONTROL_EXCLUSION_RESTRICTIONS: dict[str, str] = {
    'C2-C0': 'that a random source of matched shape carries no task-relevant '
             'content -- safe',
    'C3-C2': 'that shuffling changes nothing but structure. Preserved exactly: '
             'the multiset of weights, hence the Frobenius norm. NOT preserved: '
             'per-row and per-column norms and the singular-value spectrum, so '
             'this contrast also absorbs spectral effects',
    'C1-C3': 'that the permutation removed all and only the learned structure',
    'C3b-C2': 'that a spectrum-matched random matrix differs from an untrained '
              'one only in its singular values',
    'C1-C3b': 'that spectrum matching reproduces everything the trained weights '
              'carry except learned structure',
    'C3-C3b': 'nothing extra -- this is the empirical size of the spectral '
              'caveat on C3-C2',
    'C1-C0': 'nothing: this is the total effect, and it is the confirmatory '
             'estimand of §5',
    'C2K0-C0': 'that a copied-but-uninformative trunk with NO freeze window '
               'still carries the rest of the protocol mechanics',
    'C2-C2K0': 'that the difference between the two is the freeze window '
               'alone',
}


def _condition_arms(df: pd.DataFrame, opts: Options) -> dict:
    """The per-condition arms of `DESIGN.md` §4, selected by config, not label.

    Selecting on configuration rather than on a label pattern is what makes
    this robust to a renamed arm, and it is why C2-at-K=0 is found by its
    `freeze_updates` value rather than by the 'K0' in its label.
    """
    base = dict(env=opts.target_env, source_env=opts.source_env)
    c1 = primary_transfer_arm(df, opts)
    fw = sorted(set(c1['freeze_updates'].dropna().unique().tolist()))
    # The observed window is preferred, because a pilot invocation legitimately
    # shortens it and matching the registry's value would then select nothing.
    # But deriving it from C1 ALONE meant that losing C1 -- which is what the
    # source-validity filter does to a cell whose source fails the DESIGN.md
    # §4.3 gate -- set it to None and deleted C2, C3 and C3b from the control
    # set as a side effect, while the inventory two sections earlier plainly
    # listed those arms. The registry knows the protocol window whether or not
    # C1 survived, so it is the fallback rather than nothing.
    fw_source = 'observed in the C1 arm'
    if len(fw) == 1:
        protocol_fw = fw[0]
    elif len(fw) > 1:
        protocol_fw = None
        fw_source = (f'ambiguous: the C1 arm carries {fw}, so no single '
                     'protocol window can be identified')
    else:
        protocol_fw = registry.PROTOCOL['freeze_updates']
        fw_source = ('registry.PROTOCOL (the C1 arm is empty here, so the '
                     'window is read from the pre-registered protocol rather '
                     'than deleting every condition that depends on it)')
    untr = protocol_match(rows_where(df, condition='transfer_untrained',
                                     **base))
    perm = protocol_match(rows_where(df, condition='transfer_permuted', **base))
    c2 = (untr[untr['freeze_updates'] == protocol_fw]
          if protocol_fw is not None else untr.iloc[0:0])
    c2k0 = untr[untr['freeze_updates'] == 0]
    c3 = rows_where(perm, permute_kind='shuffle')
    c3b = rows_where(perm, permute_kind='spectrum')
    if protocol_fw is not None:
        c3 = c3[c3['freeze_updates'] == protocol_fw]
        c3b = c3b[c3b['freeze_updates'] == protocol_fw]
    return {'C0': scratch_arm(df, opts), 'C1': c1, 'C2': c2, 'C2K0': c2k0,
            'C3': c3, 'C3b': c3b, 'protocol_freeze_updates': protocol_fw,
            'protocol_freeze_updates_source': fw_source}


def section_controls(df: pd.DataFrame, opts: Options, ledger: Ledger,
                     metric: str) -> dict:
    """§10.8 -- the three control contrasts, plus C3b, from ONE joint bootstrap.

    Estimation-only: no p-value appears in this section. Every contrast comes
    out of a single resampling of the per-seed vector (C0, C1, C2, C3, C3b), so
    the contrasts' correlations are estimated rather than ignored -- which four
    independent two-sample tests on shared groups would do
    (`ANALYSIS_PLAN.md` §3, `DESIGN.md` §4.1).
    """
    h1(f'7. CONTROL CONTRASTS on {metric} -- estimation only, no p-values')
    ledger.est(f'control contrasts on {metric} (joint seed bootstrap)')
    out: dict[str, Any] = {'metric': metric, 'cells': {}}
    contrast_defs = [
        ('C2-C0', 'untrained-source contrast', 'C2', 'C0'),
        ('C3-C2', 'permuted-source contrast', 'C3', 'C2'),
        ('C1-C3', 'trained-vs-permuted contrast', 'C1', 'C3'),
        ('C3b-C2', 'spectrum-matched vs untrained', 'C3b', 'C2'),
        ('C1-C3b', 'trained vs spectrum-matched', 'C1', 'C3b'),
        ('C3-C3b', 'shuffle vs spectrum-matched', 'C3', 'C3b'),
        ('C1-C0', 'total effect', 'C1', 'C0'),
        ('C2K0-C0', 'mechanics with no freeze window', 'C2K0', 'C0'),
        ('C2-C2K0', 'the freeze window alone', 'C2', 'C2K0'),
    ]
    print('  Condition names (DESIGN.md §4): C0 scratch, C1 transfer, '
          'C2 untrained source,')
    print('  C2K0 untrained source with freeze_updates=0, C3 permuted source '
          '(entry-wise')
    print('  shuffle), C3b permuted source (spectrum-matched).')
    print()
    print('  THE TELESCOPING IDENTITY, stated once: '
          '(C2-C0) + (C3-C2) + (C1-C3) = C1-C0.')
    print('  It is an ARITHMETIC IDENTITY. It holds for any four numbers and '
          'is shown only to')
    print('  fix notation. It is NOT evidence of additivity, NOT a '
          'decomposition of a causal')
    print('  effect, and nothing about it is testable (DESIGN.md §4.1). '
          'Revision 1 of the')
    print('  design called it "an additive decomposition, each term estimable '
          'at n=10", which')
    print('  implied an empirical finding where there is none. Each cell below '
          'prints the')
    print('  arithmetic residual only, as a check that the numbers are the '
          'numbers.')
    print()
    print('  EXCLUSION RESTRICTION each mechanistic reading requires, stated '
          'once:')
    for key, restriction in CONTROL_EXCLUSION_RESTRICTIONS.items():
        print(f'    {key:<8} {restriction}')
    print()
    print('  Contrasts are named after WHAT WAS MANIPULATED, never after a '
          'mechanism.')

    for cell in CELL_ORDER:
        cdf = df[df['cell'] == cell]
        arms = _condition_arms(cdf, opts)
        # THE per-seed reduction, from the single helper §5 uses. This loop
        # used to be `for _, r in a.iterrows(): vals[int(r['seed'])] =
        # float(v)`, which is last-wins on (arm, seed) and therefore on CSV
        # row order. Reading runs_demo/per_seed.csv forwards and then
        # row-reversed, the same rows in the same multiset, moved every one of
        # the 16 contrast rows on final_score and flipped two printed
        # verdicts: dueling-vanilla C1-C0 went from "positive: the interval
        # excludes zero and everything below +0.1615" to "not distinguishable
        # from zero: the interval [-0.0966, +0.4097] covers zero", and
        # dueling-vanilla C1-C3 went the other way. C1-C0 is the confirmatory
        # estimand §5 had just refused on that same data. The collapse also
        # crossed LABELS: C0 pools every scratch label in the cell, so
        # `smoke-scratch` overwrote `scratch-mlp-vanilla` at all three seeds
        # and the baseline of every contrast belonged to a different arm from
        # the one §5 uses.
        info: dict[str, dict] = {}
        for key in CONTROL_CONDITIONS:
            info[key] = seed_vector(arms[key], metric,
                                    f'{key} condition of {cell}')
        per_seed = {k: dict(zip(v['seeds'], (float(x) for x in v['values'])))
                    for k, v in info.items()}
        # `present` keeps its meaning for the reader and for validate.py:
        # every declared condition lands in exactly one of the two lists. But
        # a condition that is present and AMBIGUOUS is not usable, and saying
        # so is the whole point.
        present = [k for k in CONTROL_CONDITIONS
                   if info[k]['n'] or info[k]['refused']]
        missing = [k for k in CONTROL_CONDITIONS if k not in present]
        refused = [k for k in present if info[k]['refused']]
        estimable = [k for k in present if not info[k]['refused']]
        h2(f'7.{cell}')
        print(f'  conditions present: {", ".join(present) or "none"}'
              + (f'   absent: {", ".join(missing)}' if missing else ''))
        print(f'  protocol freeze window used to select C2/C3/C3b: '
              f'{fmt(arms["protocol_freeze_updates"])} '
              f'({arms["protocol_freeze_updates_source"]})')
        if missing:
            print('  A contrast whose condition is absent is not computed and '
                  'not approximated.')
        cell_note: Optional[str] = None
        if refused:
            print(f'  AMBIGUOUS condition(s): {", ".join(refused)}. Every '
                  'contrast that needs one is')
            print('  NOT computed, on the same ground and in the same words '
                  'as §5 (DESIGN.md §8.4):')
            for k in refused:
                print(f'    {k}: {info[k]["reason"]}')
                ledger.deviations.append(
                    f'control contrasts {metric}/{cell}: {k} is ambiguous. '
                    f'{info[k]["reason"]}')
            cell_note = ('ambiguous condition(s) '
                         + ', '.join(f'{k} ({info[k]["reason"]})'
                                     for k in refused))
        # The joint estimate is ONE resampling of ONE per-seed vector, so it
        # exists only where the conditions entering it match seed for seed.
        # This used to take the intersection and analyse the survivors, which
        # is the seed-dropping ANALYSIS_PLAN.md §8 forbids outright and which
        # §5 refuses one section earlier; printing the dropped seeds first did
        # not turn the result into a listwise-complete estimate.
        seed_sets = {k: tuple(info[k]['seeds']) for k in estimable}
        distinct = set(seed_sets.values())
        if len(distinct) > 1:
            print('  INCOMPLETE CONTROL SET: the conditions do not match seed '
                  'for seed --')
            for k in estimable:
                print(f'    {k}: seeds {list(seed_sets[k])}')
            print('  A partial arm is refused (DESIGN.md §8.4) and no seed is '
                  'dropped to rescue an')
            print('  estimate (ANALYSIS_PLAN.md §8), so the joint estimate is '
                  'NOT computed for this')
            print('  cell. The per-condition seeds above say exactly what is '
                  'missing from where.')
            reason = ('incomplete control set: '
                      + '; '.join(f'{k} has seeds {list(seed_sets[k])}'
                                  for k in estimable))
            ledger.refusals.append(
                f'control contrasts {metric}/{cell}: {reason}')
            out['cells'][cell] = {
                'n': 0, 'suppressed': True, 'present': present,
                'missing': missing, 'ambiguous': refused,
                'contrasts': [], 'correlations': [],
                'reason': ((cell_note + '; ') if cell_note else '') + reason,
                'protocol_freeze_updates': arms['protocol_freeze_updates'],
                'protocol_freeze_updates_source':
                    arms['protocol_freeze_updates_source']}
            continue
        common = sorted(distinct.pop()) if distinct else []
        if len(common) < MIN_N_FOR_INFERENCE:
            print(f'  n={len(common)} < {MIN_N_FOR_INFERENCE} DISTINCT SEEDS: '
                  'no estimate and no interval')
            print('  (ANALYSIS_PLAN.md §9).')
            out['cells'][cell] = {
                'n': len(common), 'suppressed': True, 'present': present,
                'missing': missing, 'ambiguous': refused,
                'contrasts': [], 'correlations': [], 'reason': cell_note,
                'protocol_freeze_updates': arms['protocol_freeze_updates'],
                'protocol_freeze_updates_source':
                    arms['protocol_freeze_updates_source']}
            # The metric is in the entry. Without it, the two invocations
            # of this section (one per co-primary endpoint) wrote two
            # identical lines into the ledger with nothing to tell them apart.
            ledger.other_suppressed.append(
                f'control contrasts {metric}/{cell}: n={len(common)}')
            continue

        mat = np.column_stack([[per_seed[k][s] for s in common]
                               for k in estimable])
        col = {k: i for i, k in enumerate(estimable)}
        n = mat.shape[0]
        idx = boot_indices(n, opts.n_boot, opts.boot_seed)
        usable = [c for c in contrast_defs
                  if c[2] in col and c[3] in col]

        def make_stat(a: str, b: str) -> Callable[[np.ndarray], float]:
            ia, ib = col[a], col[b]
            return lambda m: hodges_lehmann_paired(m[:, ia] - m[:, ib])

        def make_vec(a: str, b: str) -> Callable[[np.ndarray], np.ndarray]:
            ia, ib = col[a], col[b]
            return lambda S: hl_vec(S[..., ia] - S[..., ib])

        rows = []
        reps: dict[str, np.ndarray] = {}
        per_seed_contrast: dict[str, np.ndarray] = {}
        for key, name, a, b in usable:
            stat = make_stat(a, b)
            res = bootstrap_statistic(mat, stat, opts.n_boot, opts.boot_seed,
                                      idx=idx, vec=make_vec(a, b))
            d = mat[:, col[a]] - mat[:, col[b]]
            per_seed_contrast[key] = d
            reps[key] = res.pop('reps', np.array([]))
            rows.append({'contrast': key, 'name': name, 'n': n,
                         'mean': float(np.mean(d)), 'hl': res['estimate'],
                         'ci_lo': res['lo'], 'ci_hi': res['hi'],
                         'unanimous': phrase_unanimity(d)})
        print(table(rows, ('contrast', 'name', 'n', 'mean', 'hl', 'ci_lo',
                           'ci_hi')))
        for r in rows:
            print(f'  {r["contrast"]} ({r["name"]}): '
                  + phrase_interval_verdict(r['ci_lo'], r['ci_hi'],
                                            'the contrast'))

        keys = [r['contrast'] for r in rows]
        crows = []
        if len(keys) > 1:
            for a, b in combinations(keys, 2):
                x, y = per_seed_contrast[a], per_seed_contrast[b]
                rho = correlation('spearman', x, y)
                brho = correlation('spearman', reps[a], reps[b])
                crows.append({'a': a, 'b': b, 'rho_seeds': rho,
                              'rho_bootstrap_estimators': brho})
            print()
            print('  Contrast correlation matrix. Lower triangle: Spearman '
                  'rho across seeds.')
            print('  Upper triangle: correlation of the bootstrap estimators '
                  'induced by the shared')
            print('  resampling -- which is what estimating jointly buys, and '
                  'what four separate')
            print('  two-sample tests on overlapping groups would throw away.')
            lookup = {(r['a'], r['b']): r for r in crows}
            mrows = []
            for a in keys:
                row: dict[str, Any] = {'contrast': a}
                for b in keys:
                    if a == b:
                        row[b] = 1.0
                    elif (a, b) in lookup:
                        row[b] = lookup[(a, b)]['rho_bootstrap_estimators']
                    else:
                        row[b] = lookup[(b, a)]['rho_seeds']
                mrows.append(row)
            print(table(mrows, ('contrast',) + tuple(keys), nd=2))

        if all(k in per_seed_contrast for k in ('C2-C0', 'C3-C2', 'C1-C3',
                                               'C1-C0')):
            lhs = (per_seed_contrast['C2-C0'] + per_seed_contrast['C3-C2']
                   + per_seed_contrast['C1-C3'])
            rhs = per_seed_contrast['C1-C0']
            resid = float(np.max(np.abs(lhs - rhs)))
            print()
            print('  Telescoping identity residual (an identity, see the '
                  'header): max |lhs - rhs|')
            print(f'  over seeds = {resid:.3e}')

        if 'C2-C0' in per_seed_contrast and 'C1-C3' in per_seed_contrast:
            h2_mech = phrase_magnitude_comparison(
                'the trained-vs-permuted contrast (C1-C3)',
                hodges_lehmann_paired(per_seed_contrast['C1-C3']),
                'the untrained-source contrast (C2-C0)',
                hodges_lehmann_paired(per_seed_contrast['C2-C0']))
            print()
            print(f'  H2 input for this cell: {h2_mech}')
            print('  H2 is refuted if |C1-C3| exceeds |C2-C0| in 2 or more '
                  'cells, which would mean')
            print('  learned structure dominates mechanics and the DESIGN.md '
                  '§2.2 thesis is wrong.')

        out['cells'][cell] = {'n': n, 'present': present,
                              'missing': missing, 'ambiguous': refused,
                              'joint_seeds': common,
                              'contrasts': rows,
                              'correlations': crows if len(keys) > 1 else [],
                              'reason': cell_note,
                              'protocol_freeze_updates':
                                  arms['protocol_freeze_updates'],
                              'protocol_freeze_updates_source':
                                  arms['protocol_freeze_updates_source']}

    h2('7z. hypothesis bookkeeping (estimation, no p-values)')
    h1_neg = []
    h2_flags = []
    for cell, cd in out['cells'].items():
        if cd.get('suppressed'):
            continue
        by = {r['contrast']: r for r in cd['contrasts']}
        if 'C2-C0' in by:
            r = by['C2-C0']
            neg = bool(np.isfinite(r['ci_hi']) and r['ci_hi'] < 0)
            h1_neg.append((cell, neg, r['hl'], r['ci_lo'], r['ci_hi']))
        if 'C1-C3' in by and 'C2-C0' in by:
            h2_flags.append((cell, abs(by['C1-C3']['hl'])
                             > abs(by['C2-C0']['hl'])))
    if h1_neg:
        k = sum(1 for _, neg, *_ in h1_neg if neg)
        print('  H1 (DESIGN.md §2.3): the untrained-source contrast is '
              'negative with an')
        print(f'  interval excluding zero in {k} of {len(h1_neg)} cell(s) with '
              'an estimate. H1 predicts')
        print('  at least 3 of 4; it is refuted if the interval covers zero in '
              '2 or more cells')
        print('  or is positive in any. No p-value attaches to this.')
    if h2_flags:
        k = sum(1 for _, flag in h2_flags if flag)
        print(f'  H2: |C1-C3| exceeds |C2-C0| in {k} of {len(h2_flags)} '
              'cell(s). H2 is refuted at 2 or more.')
    return out


def section_c4(df: pd.DataFrame, opts: Options, ledger: Ledger,
               metric: str) -> dict:
    """§10.9 -- C4, the positive control, against its pre-registered criterion.

    `DESIGN.md` §4.2: the interface-change-only pair, so the dynamics are
    identical by construction while the partial copy, the head reinitialisation
    and the freeze window all run exactly as in E1. Revision 1's positive
    control used the same environment for source and target, exercised none of
    the mechanics under study, and had no pass criterion.
    """
    h1(f'8. C4 POSITIVE CONTROL on {metric} (interface change, zero dynamics '
       'shift)')
    ledger.est('C4 positive control')
    print('  Pass criterion, pre-registered: the HL estimate of the paired '
          'delta has a 95%')
    print(f'  bootstrap CI whose lower bound exceeds {C4_LOWER_BOUND:+.2f} '
          'normalised-score units.')
    iface = df[df['env'] == opts.interface_env]
    if not len(iface):
        print(f'  No runs on the interface-change environment '
              f'{opts.interface_env!r}: C4 is not')
        print('  present in this dataset. It is NOT substituted with any other '
              'pair -- a')
        print('  positive control on a different pair would exercise different '
              'mechanics.')
        ledger.deviations.append('C4 absent: no runs on the '
                                 'interface-change-only environment')
        return {'available': False, 'env': opts.interface_env}
    rows = []
    for cell in CELL_ORDER:
        cdf = iface[iface['cell'] == cell]
        # target_side(), for the reason scratch_arm()'s docstring gives: the C4
        # donors are scratch runs drawn from the disjoint C4SRC block, and
        # DESIGN.md §3.4 bars that block from target-side estimation. Selecting
        # the baseline with a bare condition filter pooled those donors into
        # the very baseline they are meant to be independent of, doubling
        # n_scratch and shifting the interval.
        s = target_side(rows_where(cdf, condition='scratch'))
        t = protocol_match(target_side(rows_where(cdf, condition='transfer')))
        pair = paired_by_seed(t, s, metric)
        d = pair['a'] - pair['b']
        rec: dict[str, Any] = {'cell': cell, 'n': len(d),
                               'metric': metric,
                               'n_transfer': pair['n_a'],
                               'n_scratch': pair['n_b'],
                               'rows_transfer': pair['rows_a'],
                               'rows_scratch': pair['rows_b'],
                               'transfer_only_seeds': pair['only_a'],
                               'scratch_only_seeds': pair['only_b'],
                               'pairing_problems':
                                   pairing_problems(pair, metric)}
        # The pairing problems were printed and the PASS/FAIL verdict was
        # emitted anyway, so a positive control could be declared PASS on an
        # arm §5 would have refused. C4 is a pre-registered criterion
        # (DESIGN.md §4.2) and a criterion evaluated on a dropped-seed sample
        # is not the criterion.
        refusal = paired_refusal(pair, metric)
        if refusal:
            rec.update({'verdict': 'suppressed',
                        'reason': f'REFUSED: {refusal}'})
            ledger.other_suppressed.append(f'C4 {metric}/{cell}: {refusal}')
        elif len(d) < MIN_N_FOR_INFERENCE:
            rec.update({'verdict': 'suppressed',
                        'reason': f'n={len(d)} < {MIN_N_FOR_INFERENCE}: no '
                                  f'test, no interval (ANALYSIS_PLAN.md §9)'})
            ledger.other_suppressed.append(
                f'C4 {metric}/{cell}: n={len(d)}')
        else:
            res = bootstrap_statistic(d, hodges_lehmann_paired, opts.n_boot,
                                      opts.boot_seed, vec=hl_vec)
            passed = bool(np.isfinite(res['lo']) and res['lo'] > C4_LOWER_BOUND)
            rec.update({'hl': res['estimate'], 'ci_lo': res['lo'],
                        'ci_hi': res['hi'],
                        'verdict': 'PASS' if passed else 'FAIL',
                        'reason': (f'CI lower bound {res["lo"]:+.4f} '
                                   f'{"exceeds" if passed else "does not exceed"} '
                                   f'{C4_LOWER_BOUND:+.2f}')})
        rows.append(rec)
    print()
    print(table(rows, ('cell', 'n', 'n_transfer', 'n_scratch', 'hl', 'ci_lo',
                       'ci_hi', 'verdict')))
    for r in rows:
        print(f'  {r["cell"]}: {r["verdict"]} -- {r["reason"]}')
        if r['transfer_only_seeds'] or r['scratch_only_seeds']:
            print(f'    incomplete arm: transfer-only seeds '
                  f'{r["transfer_only_seeds"]}, scratch-only seeds '
                  f'{r["scratch_only_seeds"]}')
        for note in r['pairing_problems']:
            print(f'    {note}')
    fails = [r for r in rows if r['verdict'] == 'FAIL']
    if fails:
        print()
        print('  A C4 failure means the protocol degrades performance with no '
              'dynamics shift')
        print('  at all. That would not invalidate the study, but it would '
              'make "negative')
        print('  transfer" the wrong name for the finding, and the paper must '
              'say so')
        print('  (DESIGN.md §4.2).')
        ledger.deviations.append(
            f'C4 failed in {len(fails)} cell(s): the protocol degrades '
            'performance at zero dynamics shift')
    return {'available': True, 'env': opts.interface_env, 'rows': rows}


# ---------------------------------------------------------------------------
# §10.10 -- estimation-only. Every subsection here emits an interval and no
# p-value, with the single declared exception of the screen q-values, which
# `ANALYSIS_PLAN.md` §7 permits "for orientation only, no assertion permitted".
# ---------------------------------------------------------------------------

def _cell_deltas(df: pd.DataFrame, opts: Options, metric: str
                 ) -> tuple[dict[str, dict[int, float]], dict[str, str]]:
    """Per-cell paired deltas, and the cells that have none, with the reason.

    The refusal is §5's: a duplicated (arm, seed) is ambiguous and a partial
    arm is refused (`DESIGN.md` §8.4), and no seed is dropped after it has run
    (`ANALYSIS_PLAN.md` §8). This used to return `pair['seeds']` whatever they
    were, so the estimand §5 had just refused as an incomplete arm came back
    in 9b over the surviving seeds, with an interval and a directional
    sentence and nothing marking it as a dropped-seed estimate.
    """
    out: dict[str, dict[int, float]] = {}
    refusals: dict[str, str] = {}
    for cell in CELL_ORDER:
        cdf = df[df['cell'] == cell]
        pair = paired_by_seed(primary_transfer_arm(cdf, opts),
                              scratch_arm(cdf, opts), metric)
        reason = paired_refusal(pair, metric)
        if reason:
            out[cell] = {}
            refusals[cell] = reason
            continue
        out[cell] = {s: float(a - b) for s, a, b in
                     zip(pair['seeds'], pair['a'], pair['b'])}
    return out, refusals


def sub_rq1(df: pd.DataFrame, opts: Options, ledger: Ledger,
            metric: str) -> dict:
    """RQ1 -- between-cell scratch comparison, Brunner-Munzel relative effect.

    Associational by construction: cells are different algorithms, not
    treatments assigned to units (`DESIGN.md` §2.4). theta = P(X>Y) is used
    rather than a location shift because the cells' spreads differ enough that
    a shift model does not hold (`ANALYSIS_PLAN.md` §3).
    """
    h2('9a. RQ1 -- between-cell scratch comparison (Brunner-Munzel theta = '
       'P(X>Y))')
    ledger.est('RQ1 between-cell scratch comparison')
    # One value per seed, from the single audited helper. This took
    # `_clean(scratch_arm(...)[metric])`, a vector of ROWS: on a tree with two
    # run directories per (label, seed) it reported n_a=6 from 3 seeds and fed
    # six pseudo-observations to Brunner-Munzel, whose theta and bootstrap-t
    # interval both count independent units.
    arms = {cell: seed_vector(scratch_arm(df[df['cell'] == cell], opts),
                              metric, f'{cell} scratch arm')
            for cell in CELL_ORDER}
    rows = []
    for a, b in combinations(CELL_ORDER, 2):
        sa, sb = arms[a], arms[b]
        rec: dict[str, Any] = {'a': a, 'b': b, 'n_a': sa['n'], 'n_b': sb['n'],
                               'rows_a': sa['n_rows'], 'rows_b': sb['n_rows']}
        bad = [f'{c}: {v["reason"]}' for c, v in ((a, sa), (b, sb))
               if v['refused']]
        if bad:
            rec['note'] = 'REFUSED: ' + '; '.join(bad)
            rows.append(rec)
            ledger.other_suppressed.append(
                f'RQ1 {metric} {a} vs {b}: {rec["note"]}')
            continue
        if not sa['n'] or not sb['n']:
            continue
        # ANALYSIS_PLAN.md §9: under n<3 no test and no interval, and a
        # single-seed number may not be quoted or compared. `mean_a` and
        # `mean_b` used to be printed side by side in one row at n_a=n_b=1,
        # which is that comparison laid out for the reader even though
        # Brunner-Munzel itself refused theta.
        if min(sa['n'], sb['n']) < MIN_N_FOR_INFERENCE:
            rec['note'] = (f'n={min(sa["n"], sb["n"])} < '
                           f'{MIN_N_FOR_INFERENCE} distinct seeds: no number '
                           'is quoted and no interval is emitted '
                           '(ANALYSIS_PLAN.md §9)')
            rows.append(rec)
            ledger.other_suppressed.append(
                f'RQ1 {metric} {a} vs {b}: n={min(sa["n"], sb["n"])}')
            continue
        xa, xb = sa['values'], sb['values']
        bm = brunner_munzel(xa, xb, opts.n_boot, opts.boot_seed)
        rec.update({'mean_a': float(np.mean(xa)), 'mean_b': float(np.mean(xb)),
                    'sd_a': sd(xa), 'sd_b': sd(xb), 'theta': bm['theta'],
                    'ci_lo': bm['lo'], 'ci_hi': bm['hi'],
                    'note': bm['note']})
        rows.append(rec)
    print(table(rows, ('a', 'b', 'n_a', 'n_b', 'rows_a', 'rows_b', 'mean_a',
                       'mean_b', 'sd_a', 'sd_b', 'theta', 'ci_lo', 'ci_hi',
                       'note')))
    print('  theta = P(a run of cell a scores above a run of cell b), '
          'ties at 0.5. theta = 0.5')
    print('  is no difference. Associational: no causal reading is licensed '
          '(DESIGN.md §2.4).')
    print('  n_a and n_b are DISTINCT SEEDS; rows_a and rows_b are run '
          'directories. Where they')
    print('  differ the arm holds more than one run for a seed and the whole '
          'row is withheld.')
    for r in rows:
        if r.get('theta') is None:
            print(f'  {r["a"]} vs {r["b"]}: {r.get("note")}')
            continue
        print(f'  {r["a"]} vs {r["b"]}: ' + phrase_dispersion(
            r['a'], r['sd_a'], r['b'], r['sd_b']))
    return {'rows': rows}


def _rq3_compute(df: pd.DataFrame, opts: Options, metric: str,
                 gate: list[dict]) -> dict:
    """RQ3's between-cell contrasts and 2x2 interaction, computed not printed.

    Split out of `sub_rq3` for one reason: `DESIGN.md` 3.3 requires the same
    contrasts under the per-cell tuned policy, and a second implementation of
    them would make a disagreement between the two policies indistinguishable
    from a disagreement between two pieces of code. Every refusal below is the
    one `sub_rq3` already applied; `status` records which branch a row took so
    the printer can say the same thing it always said.
    """
    deltas, delta_refusals = _cell_deltas(df, opts, metric)
    headroom = {}
    for cell in CELL_ORDER:
        # Seeds, not rows, and the same refusal 3b applies. This recomputed
        # 3b's headroom independently as `np.mean` over `_clean(...[metric])`,
        # a ROW mean: one duplicated scratch row at final_score=99 moved this
        # cell's headroom_a column from -0.1635 to -9.0577 while 3b beside it
        # was already refusing the arm. The headroom feeds RQ3's pre-registered
        # two-scale agreement gate, so a row mean standing in for a seed mean
        # decided which wording was licensed.
        sv = seed_vector(scratch_arm(df[df['cell'] == cell], opts), metric,
                         f'{cell} scratch arm')
        headroom[cell] = (float('nan') if sv['refused'] or not sv['n']
                          else 1.0 - float(np.mean(sv['values'])))
    # Two distinct sets, because an override changes whether a contrast is
    # COMPUTED but not whether it is CONFOUNDED. Conflating them would let the
    # override quietly launder the confound out of the output.
    blocked = {(g['a'], g['b']) for g in gate if not g['permitted']}
    blocked |= {(b, a) for a, b in blocked}
    confounded_pairs = {(g['a'], g['b']) for g in gate
                        if g['verdict'] == 'CONFOUNDED'}
    confounded_pairs |= {(b, a) for a, b in confounded_pairs}

    rows = []
    for a, b in combinations(CELL_ORDER, 2):
        da, db = deltas.get(a, {}), deltas.get(b, {})
        common = sorted(set(da) & set(db))
        rec: dict[str, Any] = {'a': a, 'b': b, 'n': len(common)}
        if (a, b) in blocked:
            rec['status'] = 'intensity-blocked'
            rec['note'] = ('REFUSED: intensity-confounded cross-architecture '
                           'contrast (DESIGN.md §3.1)')
            rows.append(rec)
            continue
        gone = [c for c in (a, b) if c in delta_refusals]
        if gone:
            rec['status'] = 'cell-refused'
            rec['note'] = ('REFUSED: ' + '; '.join(
                f'{c}: {delta_refusals[c]}' for c in gone))
            rows.append(rec)
            continue
        # The two cells' delta vectors are matched on seed here as well: a
        # seed that ran in one cell and not the other cannot enter a
        # between-cell contrast, and dropping it silently is the same fault
        # one level up.
        if set(da) != set(db):
            rec['status'] = 'seeds-differ'
            rec['note'] = (
                f'REFUSED: incomplete pairing across cells. Seeds '
                f'{sorted(set(da) - set(db))} have a delta in {a} only and '
                f'{sorted(set(db) - set(da))} in {b} only. A partial arm is '
                f'refused (DESIGN.md §8.4) and no seed is dropped after '
                f'it has run (ANALYSIS_PLAN.md §8)')
            rows.append(rec)
            continue
        if len(common) < MIN_N_FOR_INFERENCE:
            rec['status'] = 'n-too-small'
            rec['note'] = f'n={len(common)}: suppressed'
            rows.append(rec)
            continue
        mat = np.column_stack([[da[s] for s in common], [db[s] for s in common]])
        idx = boot_indices(len(common), opts.n_boot, opts.boot_seed)
        raw = bootstrap_statistic(
            mat, lambda m: hodges_lehmann_paired(m[:, 0] - m[:, 1]),
            opts.n_boot, opts.boot_seed, idx=idx,
            vec=lambda S: hl_vec(S[..., 0] - S[..., 1]))
        ha, hb = headroom.get(a, float('nan')), headroom.get(b, float('nan'))
        if np.isfinite(ha) and np.isfinite(hb) and ha > 0 and hb > 0:
            adj = bootstrap_statistic(
                mat, lambda m: hodges_lehmann_paired(m[:, 0] / ha
                                                     - m[:, 1] / hb),
                opts.n_boot, opts.boot_seed, idx=idx,
                vec=lambda S: hl_vec(S[..., 0] / ha - S[..., 1] / hb))
        else:
            adj = {'estimate': float('nan'), 'lo': float('nan'),
                   'hi': float('nan')}
        # ANALYSIS_PLAN.md §3 and DESIGN.md §2.5 require AGREEMENT between
        # the normalised and the headroom-adjusted scale before any RQ3
        # wording is used. The adjusted scale only exists where both cells have
        # positive headroom, and on the real P0 data every LunarLander scratch
        # mean exceeds 1.0, so headroom is negative in every cell, the adjusted
        # columns came out blank, `scales` came out empty and NOTHING was
        # printed: the requirement passed by being absent, and the raw interval
        # was presented as though the check had been met.
        if np.isfinite(raw['lo']) and np.isfinite(adj['lo']):
            raw_excl = raw['lo'] > 0 or raw['hi'] < 0
            adj_excl = adj['lo'] > 0 or adj['hi'] < 0
            agree = 'agree' if raw_excl == adj_excl else 'DISAGREE'
        elif np.isfinite(raw['lo']):
            agree = 'UNAVAILABLE'
        else:
            agree = ''
        conf_note = ('INTENSITY-CONFOUNDED (override in force)'
                     if (a, b) in confounded_pairs else '')
        rec.update({'status': 'computed',
                    'hl': raw['estimate'], 'ci_lo': raw['lo'],
                    'ci_hi': raw['hi'], 'hl_headroom_adj': adj['estimate'],
                    'adj_lo': adj['lo'], 'adj_hi': adj['hi'],
                    'headroom_a': ha, 'headroom_b': hb, 'scales': agree,
                    'note': conf_note})
        rows.append(rec)

    inter: dict[str, Any] = {'available': False, 'status': 'not-all-cells'}
    want = ['mlp-vanilla', 'mlp-double', 'dueling-vanilla', 'dueling-double']
    unequal = (len({tuple(sorted(deltas.get(w, {}))) for w in want}) > 1
               if all(w in deltas and deltas[w] for w in want) else False)
    if unequal:
        inter = {'available': False, 'status': 'seeds-differ',
                 'seeds_by_cell': {w: sorted(deltas.get(w, {}))
                                   for w in want}}
    elif all(w in deltas and deltas[w] for w in want):
        common = sorted(set.intersection(*[set(deltas[w]) for w in want]))
        arch_pairs = {('mlp-vanilla', 'dueling-vanilla'),
                      ('mlp-double', 'dueling-double')}
        confounded = any(pr in confounded_pairs for pr in arch_pairs)
        if any(pr in blocked for pr in arch_pairs):
            inter = {'available': False, 'status': 'intensity-blocked'}
        elif len(common) < MIN_N_FOR_INFERENCE:
            inter = {'available': False, 'status': 'n-too-small',
                     'n': len(common)}
        else:
            mat = np.column_stack([[deltas[w][s] for s in common]
                                   for w in want])

            def interaction(m: np.ndarray) -> float:
                # (double - vanilla | dueling) - (double - vanilla | mlp)
                return hodges_lehmann_paired(
                    (m[:, 3] - m[:, 2]) - (m[:, 1] - m[:, 0]))

            res = bootstrap_statistic(
                mat, interaction, opts.n_boot, opts.boot_seed,
                vec=lambda S: hl_vec((S[..., 3] - S[..., 2])
                                     - (S[..., 1] - S[..., 0])))
            hs = [headroom.get(w, float('nan')) for w in want]
            scales_ok = all(np.isfinite(h) and h > 0 for h in hs)
            if scales_ok:
                hh = np.asarray(hs, dtype=float)
                adj_res = bootstrap_statistic(
                    mat / hh, interaction, opts.n_boot, opts.boot_seed,
                    vec=lambda S: hl_vec((S[..., 3] - S[..., 2])
                                         - (S[..., 1] - S[..., 0])))
                raw_excl = res['lo'] > 0 or res['hi'] < 0
                adj_excl = adj_res['lo'] > 0 or adj_res['hi'] < 0
                inter_scales = 'agree' if raw_excl == adj_excl else 'DISAGREE'
            else:
                adj_res = {'estimate': float('nan'), 'lo': float('nan'),
                           'hi': float('nan')}
                inter_scales = 'UNAVAILABLE'
            inter = {'available': True, 'status': 'computed',
                     'n': len(common),
                     'hl': res['estimate'], 'ci_lo': res['lo'],
                     'ci_hi': res['hi'],
                     'hl_headroom_adj': adj_res['estimate'],
                     'adj_lo': adj_res['lo'], 'adj_hi': adj_res['hi'],
                     'scales': inter_scales,
                     'wording_licensed': bool(inter_scales == 'agree'),
                     'intensity_confounded': bool(confounded)}
    return {'pairs': rows, 'interaction': inter, 'headroom': headroom,
            'deltas': {c: dict(v) for c, v in deltas.items()},
            'cell_refusals': dict(delta_refusals)}


def sub_rq3(df: pd.DataFrame, opts: Options, ledger: Ledger, metric: str,
            gate: list[dict]) -> dict:
    """RQ3 -- between-cell contrast of deltas and the 2x2 interaction.

    Effect modification, not "architecture causes the difference"
    (`DESIGN.md` §2.4). Explicitly underpowered: the plan puts the interaction's
    MDE at about 2.7 sigma, so this is an interval and nothing else. Reported
    on both the normalised and the headroom-adjusted scale, because a cell near
    the ceiling has less room to gain -- agreement across the two scales is
    required before any wording is used (`ANALYSIS_PLAN.md` §3).

    Everything here is the COMMON configuration. `DESIGN.md` 3.3 asserts an RQ3
    conclusion only where the per-cell tuned policy agrees, and that
    arbitration is section 9t; this section licenses wording under the two
    scales and no more.
    """
    h2('9b. RQ3 -- between-cell contrast of deltas and the 2x2 interaction')
    ledger.est('RQ3 between-cell delta contrasts and interaction')
    computed = _rq3_compute(df, opts, metric, gate)
    rows = computed['pairs']
    inter = computed['interaction']
    headroom = computed['headroom']
    deltas = computed['deltas']
    for cell, reason in computed['cell_refusals'].items():
        print(f'  {cell}: no paired delta is computed. {reason}')
        ledger.refusals.append(f'RQ3 {cell} on {metric}: {reason}')
    for r in rows:
        if r.get('status') == 'seeds-differ':
            ledger.refusals.append(
                f'RQ3 {r["a"]} vs {r["b"]} on {metric}: the two cells do not '
                'share one seed set, so the contrast is not computed')
    print(table(rows, ('a', 'b', 'n', 'hl', 'ci_lo', 'ci_hi',
                       'hl_headroom_adj', 'adj_lo', 'adj_hi', 'headroom_a',
                       'headroom_b', 'scales', 'note')))
    print('  The report template forbids "cell A avoids negative transfer '
          'while cell B does')
    print('  not" derived from two separate verdicts. The licensed form is the '
          'contrast')
    print('  above with its interval (DESIGN.md §9, ANALYSIS_PLAN.md §8).')
    for r in rows:
        if r.get('note'):
            print(f'  {r["a"]} vs {r["b"]}: {r["note"]}')
        elif r.get('scales') == 'DISAGREE':
            print(f'  {r["a"]} vs {r["b"]}: the normalised and '
                  'headroom-adjusted scales DISAGREE on')
            print('    whether the interval excludes zero. No wording is '
                  'licensed for this pair.')
            ledger.refusals.append(
                f'RQ3 {r["a"]} vs {r["b"]}: the two scales disagree, so no '
                'wording is licensed (ANALYSIS_PLAN.md §3)')
        elif r.get('scales') == 'UNAVAILABLE':
            print(f'  {r["a"]} vs {r["b"]}: the headroom-adjusted scale does '
                  'NOT EXIST here (headroom')
            print(f'    {fmt(r["headroom_a"])} and {fmt(r["headroom_b"])}; a '
                  'non-positive headroom means the scratch')
            print('    baseline is at or above the registered solved '
                  'threshold, so "fraction of the')
            print('    remaining distance" is undefined or sign-flipped). '
                  'ANALYSIS_PLAN.md §3 requires')
            print('    AGREEMENT ACROSS BOTH SCALES before any RQ3 wording is '
                  'used, and a check that')
            print('    cannot be computed has not been met. The interval '
                  'above stands as an')
            print('    estimate; NO RQ3 wording is licensed for this pair.')
            ledger.refusals.append(
                f'RQ3 {r["a"]} vs {r["b"]}: the headroom-adjusted scale is '
                f'undefined (headroom {fmt(r["headroom_a"])}, '
                f'{fmt(r["headroom_b"])}), so the two-scale agreement check of '
                'ANALYSIS_PLAN.md §3 cannot be met and no wording is licensed')

    status = inter.get('status')
    if status == 'seeds-differ':
        print()
        print('  2x2 interaction: REFUSED. The four cells do not share one '
              'seed set, so the')
        print('  interaction would be computed on the seeds that happen to '
              'appear in all four.')
        for w, seeds in (inter.get('seeds_by_cell') or {}).items():
            print(f'    {w}: seeds {seeds}')
        print('  A partial arm is refused (DESIGN.md §8.4) and no seed is '
              'dropped after it has')
        print('  run (ANALYSIS_PLAN.md §8).')
        ledger.refusals.append(
            f'RQ3 2x2 interaction on {metric} refused: the four cells do not '
            'share one seed set')
    elif status == 'intensity-blocked':
        print()
        print('  2x2 interaction: REFUSED. It mixes both architectures, '
              'whose transferred')
        print('  fractions differ by more than the tolerance, so the '
              'interaction would be')
        print('  confounded with treatment intensity (DESIGN.md §3.1).')
        ledger.refusals.append('2x2 interaction refused: '
                               'intensity-confounded across arch')
    elif status == 'n-too-small':
        print(f'  2x2 interaction: n={inter.get("n")}, suppressed.')
    elif status == 'computed':
        inter_scales = inter['scales']
        print()
        print(f'  2x2 interaction (target_rule effect on delta, dueling '
              f'minus mlp), n={inter["n"]}:')
        print(f'    HL {inter["hl"]:+.4f}   95% CI '
              f'[{inter["ci_lo"]:+.4f}, {inter["ci_hi"]:+.4f}]')
        print(f'    headroom-adjusted HL {fmt(inter["hl_headroom_adj"])}   '
              f'95% CI [{fmt(inter["adj_lo"])}, {fmt(inter["adj_hi"])}]   '
              f'scales: {inter_scales}')
        if inter_scales == 'agree':
            print('    ' + phrase_interval_verdict(inter['ci_lo'],
                                                   inter['ci_hi'],
                                                   'the interaction'))
        else:
            print('    NO WORDING IS LICENSED for the interaction: the '
                  'headroom-adjusted scale')
            print(f'    {"disagrees with" if inter_scales == "DISAGREE" else "does not exist for"} '
                  'the normalised one, and ANALYSIS_PLAN.md §3 requires '
                  'agreement')
            print('    across both scales before an RQ3 statement is '
                  'made. The interval stands as')
            print('    an estimate and nothing is said about its '
                  'direction.')
            ledger.refusals.append(
                f'RQ3 2x2 interaction: two-scale agreement '
                f'{inter_scales}, so no wording is licensed '
                '(ANALYSIS_PLAN.md §3)')
        print('    MDE for this contrast is ~2.7 sigma '
              '(ANALYSIS_PLAN.md §6), larger than any')
        print('    plausible effect, so this is an interval by design and '
              'carries no p-value.')
        if opts.allow_intensity_confound and inter.get('intensity_confounded'):
            print('    LABELLED INTENSITY-CONFOUNDED (override in force).')
    else:
        print('  2x2 interaction: not all four cells have paired deltas; not '
              'computed.')
    print()
    print('  DESIGN.md 3.3: nothing above is an RQ3 conclusion on its own. '
          'A between-cell')
    print('  contrast is asserted only where the per-cell tuned policy agrees '
          'with the common')
    print('  one, and that arbitration is section 9t. What this section '
          'licenses is wording')
    print('  under the two-scale check; what 9t licenses is an assertion.')
    return {'pairs': rows, 'interaction': inter, 'headroom': headroom,
            'deltas': deltas, 'cell_refusals': computed['cell_refusals']}


def sub_rq5(df: pd.DataFrame, opts: Options, ledger: Ledger,
            metric: str) -> dict:
    """RQ5 -- the shift gradient. Wind is primary; gravity carries a caveat.

    `DESIGN.md` §5.1: the no-op policy's score rises from 0.18 at gravity -10
    to 0.55 at gravity -4 while staying flat across wind levels, so weakening
    gravity makes the task easier as well as different. The gravity family
    therefore confounds shift severity with task difficulty, and H4 is carried
    by wind.
    """
    h2('9c. RQ5 -- shift gradient (Jonckheere-Terpstra concordance, no '
       'p-value)')
    ledger.est('RQ5 shift gradient')
    out: dict[str, Any] = {}
    for family, role, caveat in (
            ('ll_wind', 'PRIMARY',
             'the no-op score is flat across wind levels, so difficulty is '
             'held roughly constant while dynamics change'),
            ('ll_gravity', 'SECONDARY',
             'CAVEAT: weakening gravity raises the no-op score from 0.18 to '
             '0.55, so this family confounds shift severity with task '
             'difficulty and may not carry H4 alone (DESIGN.md §5.1)')):
        levels = envs.family_specs(family)
        canon = [(lab, spec.canonical()) for lab, spec in levels]
        present = [(lab, c) for lab, c in canon if (df['env'] == c).any()]
        print(f'  {family} ({role}): levels present '
              f'{[lab for lab, _ in present]} of {[lab for lab, _ in canon]}')
        print(f'    {caveat}')
        if len(present) < 2:
            print('    fewer than two levels present: no trend computed.')
            out[family] = {'available': False, 'n': 0}
            continue
        rows = []
        groups = []
        order_vals = []
        level_seeds = []
        refusals: list[str] = []
        for lab, canonical in present:
            sub = df[df['env'] == canonical]
            per_cell = []
            seeds: set[int] = set()
            for cell in CELL_ORDER:
                cdf = sub[sub['cell'] == cell]
                t = protocol_match(rows_where(cdf, condition='transfer'))
                s = rows_where(cdf, condition='scratch')
                pair = paired_by_seed(t, s, metric)
                # The refusal §5 makes, made here too: an ambiguous or partial
                # arm contributes no deltas rather than contributing the ones
                # that happen to pair.
                reason = paired_refusal(pair, metric)
                if reason:
                    refusals.append(f'{lab}/{cell}: {reason}')
                    continue
                per_cell.extend((pair['a'] - pair['b']).tolist())
                seeds.update(int(x) for x in pair['seeds'])
            groups.append(np.asarray(per_cell, dtype=float))
            level_seeds.append(seeds)
            order_vals.append(envs.family_level_value(family, canonical))
            rows.append({'level': lab, 'shift_value':
                         envs.family_level_value(family, canonical),
                         'n_seeds': len(seeds),
                         'n_deltas': len(per_cell),
                         'mean_delta': (float(np.mean(per_cell))
                                        if per_cell else None),
                         'sd_delta': sd(per_cell)})
        print(table(rows, ('level', 'shift_value', 'n_seeds', 'n_deltas',
                           'mean_delta', 'sd_delta')))
        print('    n_seeds is DISTINCT SEEDS at that level; n_deltas pools '
              'the four cells, so it')
        print('    counts up to four deltas per seed. The independent unit is '
              'the seed.')
        for note in refusals:
            print(f'    no delta from {note}')
        # ANALYSIS_PLAN.md §9's floor is on independent units, and the unit
        # here is the SEED. The gate was `sum(len(g) for g in groups) <
        # MIN_N_FOR_INFERENCE * 2` over pooled deltas, so two seeds across four
        # cells and two levels gave 16 pseudo-observations and walked past a
        # floor of 6 while no level held more than two independent units.
        smallest = min((len(s) for s in level_seeds), default=0)
        if smallest < MIN_N_FOR_INFERENCE:
            print(f'    the smallest level holds {smallest} distinct seed(s), '
                  f'below the floor of {MIN_N_FOR_INFERENCE}: no trend '
                  'estimate and no')
            print('    interval (ANALYSIS_PLAN.md §9). Pooling the four cells '
                  'multiplies the row')
            print('    count but not the number of independent units.')
            out[family] = {'available': False, 'levels': rows,
                           'n': int(sum(len(g) for g in groups)),
                           'n_seeds_smallest_level': smallest,
                           'refusals': refusals}
            continue
        order = np.argsort(np.asarray(order_vals))
        ordered = [groups[i] for i in order]
        jt = jonckheere_effect(ordered)
        # The plan asks for the standardised effect *with a bootstrap CI*, so
        # the levels are resampled within themselves -- level membership is set
        # by us and is not a random quantity, so it is not resampled.
        rng = np.random.default_rng(opts.boot_seed)
        reps = []
        for _ in range(min(opts.n_boot, 2000)):
            draw = [g[rng.integers(0, len(g), len(g))] if len(g) else g
                    for g in ordered]
            reps.append(jonckheere_effect(draw)['standardised'])
        reps_arr = np.asarray([r for r in reps if np.isfinite(r)])
        lo, hi = ((float(np.percentile(reps_arr, 2.5)),
                   float(np.percentile(reps_arr, 97.5)))
                  if len(reps_arr) >= 100 else (float('nan'), float('nan')))
        n_total = int(sum(len(g) for g in ordered))
        print(f'    Jonckheere concordance across increasing shift: '
              f'{jt["standardised"]:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]')
        print('    ' + phrase_trend(jt['standardised'], family))
        print('    ' + phrase_interval_verdict(lo, hi, 'the trend'))
        print('    H4 predicts monotone DEGRADATION, i.e. a negative '
              'concordance. Reported as')
        print('    a standardised effect with no p-value '
              '(ANALYSIS_PLAN.md §3).')
        out[family] = {'available': True, 'levels': rows, 'role': role,
                       'concordance': jt['standardised'],
                       'effect': jt['standardised'], 'J': jt['J'],
                       'ci_lo': lo, 'ci_hi': hi, 'n': n_total}
    return out


#: The column `DESIGN.md` §2.4 RQ6 requires on the budget side of the
#: comparison: the score at the SINGLE final evaluation checkpoint. It is not
#: `final_score`, which `aggregate.py` defines as the mean over the final k=3
#: checkpoints (`DESIGN.md` §5.2 P1), and it is not in `per_seed.csv` today.
#: Named here so the refusal below points at a column somebody can add rather
#: than at a vague absence.
RQ6_FINAL_CHECKPOINT_COLUMN = 'final_checkpoint_score'


def sub_rq6(df: pd.DataFrame, opts: Options, ledger: Ledger,
            metric: str) -> dict:
    """RQ6 -- budget, via the prefix evaluations.

    Valid only because the exploration schedule is a closed-form function of
    elapsed env steps and never reads the budget, so a 500-episode prefix *is*
    what a 500-episode run would have produced (`DESIGN.md` §2.4 RQ6);
    `validate.py` asserts that identifying condition.

    **The comparison is like with like, or it is not made.** `DESIGN.md` §2.4
    RQ6 is explicit: the 500-prefix score is a single held-out checkpoint, so
    it is compared against the *single* final checkpoint, "never against the
    three-checkpoint mean". The version this replaces compared
    `prefix_score_500` against `final_score`, which is exactly that
    three-checkpoint mean, and then emitted a budget-dependence conclusion from
    the mismatch. `per_seed.csv` carries no single-final-checkpoint column, so
    the correct response is the one `section_convergence` gives for the missing
    slope standard error: name the column and refuse, rather than substitute
    the nearest available number.

    What is emitted instead is each prefix's own paired delta with its
    interval, which is a well-defined estimate at that prefix and needs no
    counterpart. No sign-change sentence is generated from it: a sign change
    is a directional conclusion, and §9 forbids one at n<3 while `DESIGN.md`
    §9 forbids one from an interval that covers zero. Both conditions are
    checked before any such sentence can be printed, and they are checked on
    the intervals rather than on the means.
    """
    h2('9d. RQ6 -- does the conclusion depend on the budget?')
    ledger.est('RQ6 budget prefixes')
    prefixes = sorted(int(m.group(1)) for m in
                      (_PREFIX_SCORE_RE.match(c) for c in df.columns) if m)
    if not prefixes:
        print('  no prefix_score_* columns: not computed.')
        return {'available': False}

    final_col = RQ6_FINAL_CHECKPOINT_COLUMN
    like_for_like = final_col in df.columns and bool(_clean(df[final_col]).size)
    print(f'  prefix columns present: {prefixes}')
    if like_for_like:
        print(f'  budget side: {final_col}, the single final checkpoint, '
              'which is what DESIGN.md')
        print('  §2.4 RQ6 requires against a single prefix checkpoint.')
    else:
        print('  budget side: NOT AVAILABLE. DESIGN.md §2.4 RQ6 compares the '
              'prefix checkpoint')
        print('  against the SINGLE final checkpoint, "never against the '
              'three-checkpoint mean".')
        print(f'  per_seed.csv carries {metric} (the mean over the final k=3 '
              'checkpoints, DESIGN.md')
        print(f'  §5.2 P1) and no {final_col!r} column, so the like-for-like '
              'comparison cannot be')
        print('  made. It is NOT approximated with the three-checkpoint mean: '
              'that mismatch is')
        print('  what produced the budget-dependence conclusion this section '
              'used to emit.')
        ledger.deviations.append(
            f'RQ6 budget comparison not made: per_seed.csv has no '
            f'{final_col!r} column, and DESIGN.md §2.4 RQ6 forbids comparing '
            f'the single prefix checkpoint against {metric}, the mean over the '
            'final k=3 checkpoints. aggregate.py would have to expose the '
            'per-checkpoint score the manifest already records')

    rows = []
    for cell in CELL_ORDER:
        cdf = df[df['cell'] == cell]
        t_arm = primary_transfer_arm(cdf, opts)
        s_arm = scratch_arm(cdf, opts)
        end_d = np.array([], dtype=float)
        if like_for_like:
            end = paired_by_seed(t_arm, s_arm, final_col)
            end_d = end['a'] - end['b']
        for p in prefixes:
            col = f'prefix_score_{p}'
            pre = paired_by_seed(t_arm, s_arm, col)
            d = pre['a'] - pre['b']
            rec: dict[str, Any] = {'cell': cell, 'prefix': p, 'n': len(d),
                                   'n_budget': len(end_d)}
            # A mean of fewer than three paired deltas is a single-seed or
            # two-seed number, and ANALYSIS_PLAN.md §9 forbids quoting one. It
            # used to be printed in this table at n=1 and then compared against
            # the budget number beside it.
            if len(d) >= MIN_N_FOR_INFERENCE:
                res = bootstrap_statistic(d, hodges_lehmann_paired,
                                          opts.n_boot, opts.boot_seed,
                                          vec=hl_vec)
                rec.update({'delta_at_prefix': float(np.mean(d)),
                            'hl': res['estimate'], 'ci_lo': res['lo'],
                            'ci_hi': res['hi'],
                            'prefix_excludes_zero':
                                bool(np.isfinite(res['lo'])
                                     and (res['lo'] > 0 or res['hi'] < 0))})
            else:
                rec.update({'delta_at_prefix': None, 'hl': None,
                            'ci_lo': None, 'ci_hi': None,
                            'prefix_excludes_zero': False,
                            'note': f'n={len(d)} < {MIN_N_FOR_INFERENCE}: no '
                                    'estimate and no interval '
                                    '(ANALYSIS_PLAN.md §9)'})
            if like_for_like and len(end_d) >= MIN_N_FOR_INFERENCE:
                bres = bootstrap_statistic(end_d, hodges_lehmann_paired,
                                           opts.n_boot, opts.boot_seed,
                                           vec=hl_vec)
                rec.update({'delta_at_budget': float(np.mean(end_d)),
                            'budget_hl': bres['estimate'],
                            'budget_ci_lo': bres['lo'],
                            'budget_ci_hi': bres['hi'],
                            'budget_excludes_zero':
                                bool(np.isfinite(bres['lo'])
                                     and (bres['lo'] > 0 or bres['hi'] < 0))})
            else:
                rec.update({'delta_at_budget': None, 'budget_hl': None,
                            'budget_ci_lo': None, 'budget_ci_hi': None,
                            'budget_excludes_zero': False})
            rows.append(rec)
    print()
    print(table(rows, ('cell', 'prefix', 'n', 'delta_at_prefix', 'hl',
                       'ci_lo', 'ci_hi', 'delta_at_budget', 'budget_hl',
                       'budget_ci_lo', 'budget_ci_hi', 'note')))
    if all(r['n'] == 0 for r in rows):
        print(f'  The prefix columns {prefixes} exist but hold no values in '
              'this dataset, so RQ6')
        print('  is not estimable here. Nothing is substituted for a prefix '
              'evaluation that was')
        print('  never run.')
        ledger.deviations.append(
            'RQ6 not estimable: prefix_score_* columns are present but empty, '
            'so no episode-prefix re-evaluation exists in this dataset')

    changes = []
    if like_for_like:
        print('  A sign change between the prefix and the budget would mean '
              'the conclusion is')
        print('  budget-dependent, which is itself a finding. It is reported '
              'ONLY when both sides')
        print('  reach n >= '
              f'{MIN_N_FOR_INFERENCE} and BOTH intervals exclude zero in '
              'opposite directions:')
        print('  a change of sign in two point estimates whose intervals both '
              'cover zero is')
        print('  direction read out of noise (DESIGN.md §9).')
        for r in rows:
            a, b = r.get('hl'), r.get('budget_hl')
            if a is None or b is None:
                continue
            if not (r['prefix_excludes_zero'] and r['budget_excludes_zero']):
                continue
            if a * b >= 0:
                continue
            changes.append(r)
            print(f'  {r["cell"]}: the delta CHANGES SIGN between prefix '
                  f'{r["prefix"]} (HL {a:+.4f}, CI')
            print(f'    [{r["ci_lo"]:+.4f}, {r["ci_hi"]:+.4f}]) and the budget '
                  f'(HL {b:+.4f}, CI [{r["budget_ci_lo"]:+.4f}, '
                  f'{r["budget_ci_hi"]:+.4f}]).')
            print('    Both intervals exclude zero, so the conclusion is '
                  'budget-dependent in this cell.')
        if not changes:
            print('  No cell meets those conditions, so no budget-dependence '
                  'statement is made in')
            print('  either direction: this is the absence of a licensed '
                  'claim, not evidence that')
            print('  the conclusion is budget-independent.')
    return {'available': True, 'rows': rows, 'prefixes': prefixes,
            'like_for_like': like_for_like,
            'budget_column': final_col if like_for_like else None,
            'sign_changes': [f'{r["cell"]}@{r["prefix"]}' for r in changes]}


def sub_dispersion(df: pd.DataFrame, opts: Options, ledger: Ledger,
                   metric: str) -> dict:
    """Dispersion: SD ratio with a bootstrap CI. Never a dispersion p-value.

    `ANALYSIS_PLAN.md` §3: at n=10 with an SD ratio near 3 the test has almost
    no power, which is the honest explanation of the published Brown-Forsythe
    null -- reported here as a statistic with the p withheld. The published
    paper also conflated within-run instability with across-seed sensitivity
    and then described the result backwards; the two are separate rows below.
    """
    h2('9e. dispersion -- across-seed SD ratio (transfer / scratch)')
    ledger.est('dispersion SD ratio')
    rows = []
    for cell in CELL_ORDER:
        cdf = df[df['cell'] == cell]
        pair = paired_by_seed(primary_transfer_arm(cdf, opts),
                              scratch_arm(cdf, opts), metric)
        n = len(pair['seeds'])
        rec: dict[str, Any] = {'cell': cell, 'n': n,
                               'rows_transfer': pair['rows_a'],
                               'rows_scratch': pair['rows_b']}
        # §5's refusal, in §5's words. This section took whatever paired and
        # printed an "across-seed spread" ratio over it: with one seed missing
        # from the transfer arm it reported "mlp-vanilla transfer has an
        # across-seed spread 1.13x wider than mlp-vanilla scratch" at n=9,
        # unmarked, one section after §5 refused the same arm as incomplete.
        reason = paired_refusal(pair, metric)
        if reason:
            rec.update({'sd_scratch': None, 'sd_transfer': None,
                        'brown_forsythe_W': None,
                        'note': f'REFUSED: {reason}'})
            rows.append(rec)
            ledger.other_suppressed.append(
                f'dispersion {metric}/{cell}: {reason}')
            continue
        rec.update({'sd_scratch': sd(pair['b']), 'sd_transfer': sd(pair['a'])})
        if n >= MIN_N_FOR_INFERENCE:
            mat = np.column_stack([pair['a'], pair['b']])

            def ratio(m: np.ndarray) -> float:
                a, b = np.std(m[:, 0], ddof=1), np.std(m[:, 1], ddof=1)
                return float(a / b) if b > 0 else float('nan')

            def ratio_vec(S: np.ndarray) -> np.ndarray:
                # A resample can draw one seed three times, making the
                # denominator SD exactly zero. Such a replicate carries no
                # information about the ratio and is dropped by bca_interval's
                # finite filter -- it is never silently read as an infinity.
                num = np.std(S[..., 0], axis=-1, ddof=1)
                den = np.std(S[..., 1], axis=-1, ddof=1)
                with np.errstate(divide='ignore', invalid='ignore'):
                    out = num / den
                return np.where(den > 0, out, np.nan)

            res = bootstrap_statistic(mat, ratio, opts.n_boot, opts.boot_seed,
                                      vec=ratio_vec)
            # Brown-Forsythe divides by a pooled spread of absolute deviations
            # from the median, which is exactly zero when an arm is constant.
            # SciPy then divides by zero, writes a RuntimeWarning to a stream
            # nothing here captures, and returns NaN. The degenerate case is
            # detected instead and reported as not estimable, which is what a
            # constant arm means.
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                bf = sps.levene(pair['a'], pair['b'], center='median')
            w = float(bf.statistic)
            rec.update({'sd_ratio': res['estimate'], 'ci_lo': res['lo'],
                        'ci_hi': res['hi'],
                        'brown_forsythe_W': (w if np.isfinite(w) else None),
                        'note': ('' if np.isfinite(w) else
                                 'Brown-Forsythe W is not estimable: an arm '
                                 'has zero spread about its median')})
        rows.append(rec)
    print(table(rows, ('cell', 'n', 'rows_transfer', 'rows_scratch',
                       'sd_scratch', 'sd_transfer', 'sd_ratio', 'ci_lo',
                       'ci_hi', 'brown_forsythe_W', 'note')))
    print('  n is DISTINCT SEEDS in the paired sample; rows_* are run '
          'directories. A cell whose')
    print('  arms are ambiguous or do not match seed for seed carries no '
          'ratio at all.')
    print('  Brown-Forsythe W is shown for continuity with the published '
          'analysis; its')
    print('  p-value is WITHHELD, because no p-value is emitted outside the '
          'confirmatory')
    print('  family (ANALYSIS_PLAN.md §7) and because at this n the test has '
          'almost no power.')
    for r in rows:
        if np.isfinite(r.get('sd_ratio', float('nan'))):
            print('  ' + phrase_dispersion(f'{r["cell"]} transfer',
                                           r['sd_transfer'],
                                           f'{r["cell"]} scratch',
                                           r['sd_scratch']))
        elif r.get('note'):
            # A cell with no ratio says why, in one place, rather than leaving
            # the reader to read a blank row as an absence of difference.
            print(f'  {r["cell"]}: no dispersion statement is made. '
                  f'{r["note"]}')
    wrows = []
    if 'within_run_sd' in df.columns:
        print()
        print('  within_run_sd (training instability) and the across-seed SD '
              'above (seed')
        print('  sensitivity) are DIFFERENT metrics; the published study '
              'conflated them:')
        # Restricted to the target environment and to target-side blocks. This
        # table used to be built from the whole frame, so one row labelled
        # "dueling-double scratch n=40" pooled CartPole source runs, the
        # LunarLander scratch arm, the padded interface variant and the C4SRC
        # donor arm into a single mean, across scales that DESIGN.md §5.1
        # normalises separately and across a block §3.4 bars from target-side
        # estimation. The section header claims 9f-9j are restricted to the
        # target environment; 9e was not, and was not covered by that sentence.
        wdf = target_side(df[df['env'] == opts.target_env])
        print(f'  restricted to {opts.target_env} and to target-side seed '
              f'blocks: {len(wdf)} of {len(df)} run(s).')
        # Grouped by label as well as by condition: two arms can share a
        # condition and differ in protocol (the matched and trunk transfer
        # sets, say), and averaging across them would report a number that
        # belongs to no arm.
        for (cell, cond, label), g in wdf.groupby(['cell', 'condition',
                                                   'label'], dropna=False):
            # Seeds, not rows, from the same helper as everything else. `n`
            # here was `len(_clean(g['within_run_sd']))`, so an arm with two
            # run directories per seed reported twice the sample it had and a
            # mean over the duplicated rows.
            sv = seed_vector(g, 'within_run_sd', f'{label} arm')
            if not sv['n'] and not sv['refused']:
                continue
            wrows.append({
                'env': opts.target_env, 'cell': cell, 'condition': cond,
                'label': label, 'n': sv['n'], 'n_rows': sv['n_rows'],
                'within_run_sd_mean': (None if sv['refused'] else
                                       float(np.mean(sv['values']))),
                'note': sv['reason'] or ''})
        print(table(wrows, ('env', 'cell', 'condition', 'label', 'n', 'n_rows',
                            'within_run_sd_mean', 'note')))
        print('  n is DISTINCT SEEDS; n_rows is run directories.')
    return {'rows': rows, 'within_run_sd': wrows}


#: The column that would carry the end of the freeze window on the same clock
#: as `steps_to_threshold`, i.e. in env steps. `ANALYSIS_PLAN.md` §5 asks for
#: Kaplan-Meier "with delayed entry at the end of the freeze window where a
#: freeze is in force", and the freeze window is indexed in gradient UPDATES
#: (`DESIGN.md` §1), not env steps. Converting one to the other needs the
#: run's own update-to-step ratio, and a ratio inferred from two totals is an
#: approximation invented here rather than a measurement, which is what
#: `section_convergence` refuses to do for the missing slope SE. So the column
#: is named and the requirement is refused where it bites.
FREEZE_END_COLUMN = 'freeze_end_env_steps'


def sub_censored(df: pd.DataFrame, opts: Options, ledger: Ledger) -> dict:
    """Censored metrics: Kaplan-Meier and an exact interval on P(reached).

    `ANALYSIS_PLAN.md` §5. The censoring is administrative: the same budget for
    every run, independent of the event time by construction, which is the
    benign case. The budget is never imputed as an observation and no censored
    run is dropped.

    Three defects the previous version had, each of which deleted data
    silently:

    * `t = _clean(g[tcol])` stripped non-finite times while the event vector
      was built from the whole group, so `len(t) != len(ev)` and the WHOLE ARM
      hit `continue` and vanished from the table with no message anywhere in
      the report. That is the exact failure §5 names: "never drop censored
      runs (conditions on the outcome, and reintroduces the silent-seed-
      dropping defect)". Times and events are now kept aligned row by row, a
      run with no usable time is counted and named, and it never removes the
      arm around it.
    * `bool(v) if v is not None else True` read an unknown censoring flag as
      censored, so schema drift reported 0/n reached for every arm. An unknown
      flag is now its own category and enters neither numerator nor
      denominator.
    * `p_reached` and its Clopper-Pearson interval were emitted at n=1
      (p_reached 1.0000, CI [0.0250, 1.0000] for 24 arms), which §9 forbids:
      under n<3 no test and no interval is emitted.

    A fourth, which is the row-versus-seed confusion the rest of this module
    was rewritten around: `n_prop` counted ROWS carrying a readable censoring
    flag, so an arm holding two run directories for one seed presented twice
    the independent units it had. The Clopper-Pearson interval is an exact
    binomial interval whose n is a count of independent units, and the
    Kaplan-Meier risk set is the same count, so on the repo's own `runs_demo`
    tree a 3-seed arm reported proportions out of 6 with intervals on six
    pseudo-units. An arm with more than one row for a seed is now refused
    exactly as §5 refuses it, and it leaves the log-rank comparisons too:
    without duplicates rows and seeds coincide, so nothing else changes.
    """
    h2('9f. steps_to_threshold -- right-censored at the budget')
    ledger.est('censored steps-to-threshold (Kaplan-Meier, Clopper-Pearson)')
    out: dict[str, Any] = {}
    entry_available = FREEZE_END_COLUMN in df.columns
    frozen_arms = 0
    for tag, level in THRESHOLD_LEVELS:
        tcol, ccol = f'steps_to_threshold_{tag}', f'censored_{tag}'
        if tcol not in df.columns or ccol not in df.columns:
            print(f'  {tcol}: column absent, skipped.')
            continue
        rows = []
        arms: dict[tuple[str, str], dict] = {}
        for (cell, cond, label), g in df.groupby(['cell', 'condition',
                                                  'label'], dropna=False):
            # The unit of the proportion and of the risk set is the SEED, so
            # two run directories recorded for one seed are one unit and not
            # two. Which of them to keep is not decidable here, so the arm is
            # refused rather than counted twice (DESIGN.md §8.4).
            seed_counts: dict[int, int] = {}
            for s in g['seed']:
                seed_counts[int(s)] = seed_counts.get(int(s), 0) + 1
            dup_seeds = sorted(s for s, c in seed_counts.items() if c > 1)
            times = np.asarray(pd.to_numeric(g[tcol], errors='coerce'),
                               dtype=float)
            flags = [parse_boolean(v)[1] for v in g[ccol]]
            n_runs = len(g)
            unknown = int(sum(1 for v in flags if v is None))
            censored = np.asarray([bool(v) for v in flags], dtype=bool)
            known = np.asarray([v is not None for v in flags], dtype=bool)
            usable_time = np.isfinite(times)
            no_time = int(np.sum(~usable_time))
            # Reached, not reached, and not known: three categories, and only
            # the first two form the proportion. Counted over SEEDS, so the
            # column named n_in_proportion holds the number the exact interval
            # is entitled to use; where no seed is duplicated this is the same
            # number the row count gave, and where one is the arm is refused
            # below rather than credited with the extra unit.
            reached = known & ~censored
            in_denominator = known
            seeds_col = [int(s) for s in g['seed']]
            k = len({s for s, ok in zip(seeds_col, reached) if ok})
            n_prop = len({s for s, ok in zip(seeds_col, in_denominator) if ok})
            # A run that reached the threshold but recorded no time cannot
            # enter a curve; a censored run with no time is a censored run
            # whose censoring time was not written down. Both are counted.
            curve_rows = usable_time & known
            if dup_seeds:
                # No curve and no proportion from an ambiguous arm, and it does
                # not enter the log-rank either.
                curve_rows = np.zeros(len(g), dtype=bool)
            freeze = sorted(set(g['freeze_updates'].dropna().unique().tolist()))
            has_freeze = bool(freeze) and max(float(f) for f in freeze) > 0
            entry = None
            if has_freeze:
                frozen_arms += 1
                if entry_available:
                    entry = np.asarray(
                        pd.to_numeric(g[FREEZE_END_COLUMN], errors='coerce'),
                        dtype=float)[curve_rows]
            km = kaplan_meier(times[curve_rows],
                              reached[curve_rows], entry)
            arms[(cell, label)] = {'t': times[curve_rows],
                                   'e': reached[curve_rows]}
            rec: dict[str, Any] = {
                'cell': cell, 'condition': cond, 'label': label,
                'n': n_runs, 'n_in_proportion': n_prop, 'reached': k,
                'no_time_recorded': no_time,
                'unknown_censoring': unknown,
                'duplicated_seeds': dup_seeds,
                'delayed_entry': bool(km['delayed_entry'])}
            if dup_seeds:
                rec.update({'p_reached': None, 'cp_lo': None, 'cp_hi': None,
                            'km_median_steps': None,
                            'note': f'REFUSED: seed(s) {dup_seeds} have '
                                    f'more than one run in this arm, so '
                                    f'neither the proportion nor the risk set '
                                    f'has a well-defined n. The arm is '
                                    f'ambiguous and is not resolved by '
                                    f'keeping one row (DESIGN.md §8.4)'})
                ledger.other_suppressed.append(
                    f'P(reached {tag}) {cell}/{label}: seed(s) {dup_seeds} '
                    f'carry more than one run')
                rows.append(rec)
                continue
            if n_prop >= MIN_N_FOR_INFERENCE:
                lo, hi = clopper_pearson(k, n_prop)
                rec.update({'p_reached': k / n_prop, 'cp_lo': lo, 'cp_hi': hi,
                            'km_median_steps': km['median'], 'note': ''})
            else:
                # The KM median goes too. At n=1 it is that single run's own
                # event time wearing the name of an arm-level summary, and §9
                # forbids quoting a single-seed number, not only testing one.
                rec.update({'p_reached': None, 'cp_lo': None, 'cp_hi': None,
                            'km_median_steps': None,
                            'note': f'n={n_prop} < {MIN_N_FOR_INFERENCE}: no '
                                    'proportion, no interval and no median '
                                    '(ANALYSIS_PLAN.md §9)'})
                ledger.other_suppressed.append(
                    f'P(reached {tag}) {cell}/{label}: n={n_prop}')
            if no_time or unknown:
                bits = []
                if no_time:
                    bits.append(f'{no_time} run(s) have no {tcol} value')
                if unknown:
                    bits.append(f'{unknown} run(s) have no readable {ccol} '
                                'flag')
                rec['note'] = ((rec['note'] + '; ') if rec['note'] else '') \
                    + ', '.join(bits) + ' (reported, not dropped)'
            rows.append(rec)
        print(f'  threshold = normalised score {level}')
        print(table(rows, ('cell', 'condition', 'label', 'n',
                           'n_in_proportion', 'reached', 'p_reached', 'cp_lo',
                           'cp_hi', 'km_median_steps', 'no_time_recorded',
                           'unknown_censoring', 'duplicated_seeds', 'note')))
        print('  n_in_proportion counts SEEDS with a readable censoring flag: '
              'the exact interval')
        print('  and the risk set both count independent units, and two run '
              'directories for one')
        print('  seed are one unit. An arm with a duplicated seed carries no '
              'proportion.')
        for r in rows:
            if r['no_time_recorded'] or r['unknown_censoring']:
                print(f'    {r["cell"]}/{r["label"]}: {r["note"]}')
        lr = []
        for cell in CELL_ORDER:
            keys = [k for k in arms if k[0] == cell]
            for a, b in combinations(sorted(keys), 2):
                res = logrank_statistic(arms[a]['t'], arms[a]['e'],
                                        arms[b]['t'], arms[b]['e'])
                if res.get('statistic') is not None:
                    lr.append({'cell': cell, 'a': a[1], 'b': b[1],
                               'logrank_chi2': res['statistic'],
                               'df': res['df'], 'events': str(res['events'])})
        if lr:
            print('  log-rank, statistic only (both arms have at least '
                  f'{LOGRANK_MIN_EVENTS} events):')
            print(table(lr, ('cell', 'a', 'b', 'logrank_chi2', 'df',
                             'events')))
        out[tag] = {'level': level, 'arms': rows, 'logrank': lr}
    print('  At 0/10 the Clopper-Pearson upper bound is 0.308: that is the '
          'informative')
    print('  statement, and it replaces a p-value entirely '
          '(ANALYSIS_PLAN.md §5).')
    if frozen_arms and not entry_available:
        print()
        print('  DELAYED ENTRY NOT APPLIED. ANALYSIS_PLAN.md §5 asks for '
              'Kaplan-Meier curves')
        print('  "with delayed entry at the end of the freeze window where a '
              'freeze is in')
        print(f'  force", and {frozen_arms} arm-threshold combination(s) here '
              'do have a freeze in')
        print('  force. The freeze window is indexed in gradient UPDATES '
              '(DESIGN.md §1) while')
        print(f'  steps_to_threshold is in env steps, and per_seed.csv carries '
              f'no {FREEZE_END_COLUMN!r}')
        print('  column to convert it. The curves above are therefore '
              'left-truncation-free, which')
        print('  overstates the risk set before the freeze ends. This is '
              'stated rather than')
        print('  silently unimplemented, and kaplan_meier() takes the entry '
              'times as soon as the')
        print('  column exists.')
        ledger.deviations.append(
            f'ANALYSIS_PLAN.md §5 delayed entry not applied: {frozen_arms} '
            f'arm(s) have a freeze in force and per_seed.csv carries no '
            f'{FREEZE_END_COLUMN!r} (the freeze window is in gradient updates, '
            'steps_to_threshold is in env steps, and inferring the ratio would '
            'be an approximation rather than a measurement)')
    # Nested in a dict, not left as loose scalars: report.py walks this
    # mapping expecting every value to be a per-threshold block it can call
    # `.get('arms')` on.
    out['_delayed_entry'] = {'available': entry_available,
                             'arms_with_freeze': frozen_arms,
                             'column_required': FREEZE_END_COLUMN}
    return out


def sub_secondary(df: pd.DataFrame, opts: Options, ledger: Ledger) -> dict:
    """Secondary and mechanism endpoints: intervals, no p-values.

    `jumpstart` is interpretable only where the output head is transferred:
    with a reinitialised head the zero-shot policy is an argmax over a random
    readout, so it is structurally at chance and comparing it would be
    meaningless (`DESIGN.md` §5.3). That condition is checked here rather than
    assumed, and `probe_jumpstart` is the quantity reported in its place.
    """
    h2('9g. secondary and mechanism endpoints -- estimation only')
    ledger.est('secondary endpoints')
    ledger.est('mechanism signals')
    out: dict[str, Any] = {}
    heads = sorted(set(str(v) for v in df['head_policy'].dropna().unique()))
    reinit_only = heads == ['reinit']
    print(f'  head_policy present: {heads}')
    if reinit_only:
        print('  Every transfer arm reinitialises the output head, so '
              '`jumpstart` is at chance')
        print('  by construction and is NOT compared. `probe_jumpstart` -- '
              'trunk frozen, head')
        print('  refit on a fixed batch -- is the quantity that measures '
              'whether the')
        print('  transferred features carry usable information '
              '(DESIGN.md §5.3).')
    cols = [c for c in SECONDARY_COLUMNS + MECHANISM_COLUMNS
            if c in df.columns and _clean(df[c]).size]
    if reinit_only and 'jumpstart_score' in cols:
        cols.remove('jumpstart_score')
        ledger.refusals.append(
            'jumpstart comparison refused: every arm reinitialises the head, '
            'so jumpstart is structurally at chance (DESIGN.md §5.3)')
    rows = []
    for col in cols:
        role = metric_role(col)
        for cell in CELL_ORDER:
            cdf = df[df['cell'] == cell]
            pair = paired_by_seed(primary_transfer_arm(cdf, opts),
                                  scratch_arm(cdf, opts), col)
            d = pair['a'] - pair['b']
            # The two arm means are placed in one row, which is an
            # invitation to compare them. ANALYSIS_PLAN.md §9 forbids a number
            # from fewer than three seeds being quoted or compared, so below
            # the floor the row carries its n and nothing else: it used to
            # print, at n=1, probe_jumpstart_score mlp-vanilla -1.2906 against
            # -1.3260, which is the comparison §9 names.
            # §5's refusal, in §5's words: an ambiguous arm or a partial one
            # carries no paired estimate here either. This section took
            # whatever paired and printed the two arm means side by side.
            reason = paired_refusal(pair, col)
            enough = not reason and len(d) >= MIN_N_FOR_INFERENCE
            rec = {'metric': col, 'role': role, 'cell': cell, 'n': len(d),
                   'transfer_mean': (float(np.mean(pair['a'])) if enough
                                     else None),
                   'scratch_mean': (float(np.mean(pair['b'])) if enough
                                    else None),
                   'note': ('' if enough else
                            (f'REFUSED: {reason}' if reason else
                             f'n={len(d)} < {MIN_N_FOR_INFERENCE}: no number '
                             'quoted (ANALYSIS_PLAN.md §9)'))}
            if enough:
                res = bootstrap_statistic(d, hodges_lehmann_paired,
                                          opts.n_boot, opts.boot_seed,
                                          vec=hl_vec)
                rec.update({'hl_delta': res['estimate'], 'ci_lo': res['lo'],
                            'ci_hi': res['hi']})
            elif reason:
                ledger.other_suppressed.append(
                    f'secondary endpoint {col}/{cell}: {reason}')
            rows.append(rec)
    print(table(rows, ('metric', 'role', 'cell', 'n', 'scratch_mean',
                       'transfer_mean', 'hl_delta', 'ci_lo', 'ci_hi',
                       'note')))
    print('  No p-value appears in this table. A mechanism claim in the paper '
          'must cite one')
    print('  of these instrumented signals; there is no free-text mechanism '
          'slot (DESIGN.md §9).')
    out['rows'] = rows

    h2('9h. descriptive-only metrics -- reported, never tested')
    drows = []
    for col in [c for c, r in METRIC_ROLES.items()
                if r == DESCRIPTIVE and c in df.columns]:
        # Grouped by label as well, and reduced to one value per seed. `n` was
        # `len(_clean(g[col]))`, a ROW count over a group that pooled every
        # label sharing a condition, so a cell's transfer arms at two different
        # protocols were averaged into one number belonging to neither and an
        # arm with two runs per seed reported twice its sample. The same fix
        # 9e's within_run_sd table already carries, for the same reason.
        for (cell, cond, label), g in df.groupby(['cell', 'condition',
                                                  'label'], dropna=False):
            sv = seed_vector(g, col, f'{label} arm')
            if not sv['n'] and not sv['refused']:
                continue
            drows.append({
                'metric': col, 'cell': cell, 'condition': cond,
                'label': label, 'n': sv['n'], 'n_rows': sv['n_rows'],
                'mean': (None if sv['refused'] else
                         float(np.mean(sv['values']))),
                'sd': None if sv['refused'] else sd(sv['values']),
                'note': sv['reason'] or ''})
    print(table(drows, ('metric', 'cell', 'condition', 'label', 'n', 'n_rows',
                        'mean', 'sd', 'note')))
    print('  n is DISTINCT SEEDS; n_rows is run directories. Grouped by label '
          'as well as by')
    print('  condition: two arms that share a condition and differ in '
          'protocol are not one arm.')
    print('  These carry role "descriptive" in METRIC_ROLES. '
          '`require_confirmatory` refuses')
    print('  a test on any of them -- the mechanical fix for the published '
          '§V.A/§V.B')
    print('  contradiction (DESIGN.md §5.4).')
    out['descriptive'] = drows

    if 'source_final_score' in df.columns:
        h2('9i. source competence as a continuous covariate (DESIGN.md §4.3)')
        srows = []
        for cell in CELL_ORDER:
            cdf = df[df['cell'] == cell]
            t = primary_transfer_arm(cdf, opts)
            s = scratch_arm(cdf, opts)
            pair = paired_by_seed(t, s, 'final_score')
            comp = {int(r['seed']): float(r['source_final_score'])
                    for _, r in t.iterrows()
                    if pd.notna(r.get('source_final_score'))}
            xs = [comp[sd_] for sd_ in pair['seeds'] if sd_ in comp]
            ys = [float(a - b) for sd_, a, b in
                  zip(pair['seeds'], pair['a'], pair['b']) if sd_ in comp]
            rec = {'cell': cell, 'n': len(xs)}
            if len(xs) >= MIN_N_FOR_INFERENCE and len(set(xs)) > 1:
                ts = sps.theilslopes(ys, xs, alpha=1 - ALPHA)
                rec.update({'slope': float(ts[0]), 'ci_lo': float(ts[2]),
                            'ci_hi': float(ts[3])})
            elif len(set(xs)) <= 1:
                rec['note'] = 'source competence has no variation'
            srows.append(rec)
        print(table(srows, ('cell', 'n', 'slope', 'ci_lo', 'ci_hi', 'note')))
        print('  Theil-Sen slope of delta on the source normalised score, with '
              'its')
        print('  distribution-free interval. A descriptive relationship, not a '
              'mediation claim.')
        out['source_competence'] = srows
    return out


def sub_screens(df: pd.DataFrame, opts: Options, ledger: Ledger,
                metric: str) -> dict:
    """Screen-family experiments: per-level estimates with BH q for orientation.

    `ANALYSIS_PLAN.md` §7 permits Benjamini-Hochberg q-values here and nowhere
    else, "orientation only, no assertion permitted". §3 pre-commits the
    follow-up rule: a screen selects at most one follow-up, which is then run
    on `REPLICATE` seeds and reported as a fresh estimate.

    Two things this section must not do, and used to:

    * **Re-test the confirmatory estimand.** The primary transfer arm at
      `registry.PROTOCOL` belongs to several screen experiments by
      construction, so `transfer-mlp-double` appeared in E4, E5, E6 and E7 with
      `p_raw` equal to the confirmatory sign-flip p and a softer `q_bh` beside
      it. That is the confirmatory contrast relocated into a second
      multiplicity family, which is exactly what §7 fixes family membership to
      prevent. The primary arm is identified from the same selector §5 uses and
      excluded here, with the refusal recorded.
    * **Count one contrast several times.** A level that belongs to four screen
      experiments used to enter the BH family four times, inflating m and
      breaking BH's own assumption that its inputs are distinct tests. Each
      (cell, level) contrast now enters once, and the experiments it belongs to
      are listed in the row.
    """
    h2('9j. ablation screens -- BH q for orientation only, never an assertion')
    ledger.est('ablation screens (BH q, orientation only)')
    screens = [e for e in registry.EXPERIMENTS.values()
               if e.family == 'screen']
    present = []
    for exp in screens:
        sub = in_experiments(df, [exp.id])
        if len(sub):
            present.append((exp, sub))
    if not present:
        print(f'  no screen-family experiments in this dataset (declared: '
              f'{", ".join(e.id for e in screens)}).')
        return {'available': False}

    # The confirmatory arm, per cell, by configuration rather than by name.
    primary_labels: dict[str, set[str]] = {}
    for cell in CELL_ORDER:
        arm = primary_transfer_arm(df[df['cell'] == cell], opts)
        primary_labels[cell] = set(str(v) for v in arm['label'].unique())

    order_index: list[tuple[str, str]] = []
    by_key: dict[tuple[str, str], dict] = {}
    refused: list[str] = []
    for exp, sub in present:
        for cell in CELL_ORDER:
            cdf = sub[sub['cell'] == cell]
            base = scratch_arm(df[df['cell'] == cell], opts)
            for label, g in cdf.groupby('label'):
                if (g['condition'] == 'scratch').all():
                    continue
                key = (cell, str(label))
                if str(label) in primary_labels.get(cell, set()):
                    tag = f'{exp.id}/{cell}/{label}'
                    if tag not in refused:
                        refused.append(tag)
                    continue
                if key in by_key:
                    by_key[key]['experiments'].append(exp.id)
                    by_key[key]['experiment'] = ';'.join(
                        by_key[key]['experiments'])
                    continue
                pair = paired_by_seed(g, base, metric)
                d = pair['a'] - pair['b']
                rec: dict[str, Any] = {
                    'experiment': exp.id, 'experiments': [exp.id],
                    'cell': cell, 'level': str(label), 'n': len(d)}
                for note in pairing_problems(pair, metric, str(label),
                                             'scratch'):
                    rec['note'] = ((rec.get('note', '') + '; ')
                                   if rec.get('note') else '') + note
                # The notes above were printed and the estimate was computed
                # anyway, so a screen level with an ambiguous or partial arm
                # still entered the BH family with a p-value over the seeds
                # that happened to pair. A refusal keeps it out of the family
                # entirely, which also keeps m honest.
                refusal = paired_refusal(pair, metric, str(label), 'scratch')
                if refusal:
                    rec['note'] = f'REFUSED: {refusal}'
                    ledger.other_suppressed.append(
                        f'screen {exp.id}/{cell}/{label}: {refusal}')
                    by_key[key] = rec
                    order_index.append(key)
                    continue
                if len(d) >= MIN_N_FOR_INFERENCE:
                    res = bootstrap_statistic(d, hodges_lehmann_paired,
                                              opts.n_boot, opts.boot_seed,
                                              vec=hl_vec)
                    sf = sign_flip_test(d, seed=opts.boot_seed)
                    rec.update({'hl': res['estimate'],
                                'estimate': res['estimate'],
                                'ci_lo': res['lo'], 'ci_hi': res['hi'],
                                'p_raw': sf['p'], 'p': sf['p'],
                                'statistic': float(np.mean(d)),
                                'test': 'exact sign-flip on paired deltas',
                                'rq': 'screen'})
                by_key[key] = rec
                order_index.append(key)

    rows = [by_key[k] for k in order_index]
    pvals = [(i, r['p_raw']) for i, r in enumerate(rows)
             if r.get('p_raw') is not None]
    if pvals:
        order = sorted(pvals, key=lambda kp: kp[1])
        m = len(order)
        prev = 1.0
        for rank in range(m, 0, -1):
            i, p = order[rank - 1]
            q = min(prev, p * m / rank)
            prev = q
            rows[i]['q_bh'] = q
            rows[i]['q'] = q
    print(table(rows, ('experiment', 'cell', 'level', 'n', 'hl', 'ci_lo',
                       'ci_hi', 'p_raw', 'q_bh', 'note')))
    print(f'  BH family size m = {len(pvals)} distinct (cell, level) '
          'contrast(s). A contrast that')
    print('  belongs to several screen experiments is listed once, with its '
          'experiments joined')
    print('  in the first column: entering it once per experiment would '
          'inflate m and break')
    print("  BH's own assumption that its inputs are distinct tests.")
    print('  These q-values ORIENT; they assert nothing. A screen result is '
          'never a finding.')
    if refused:
        print()
        print('  REFUSED, as members of the confirmatory family rather than '
              'screen levels:')
        for tag in refused:
            print(f'    {tag}')
        print('  The primary transfer arm at registry.PROTOCOL is the '
              'confirmatory estimand of §5.')
        print('  ANALYSIS_PLAN.md §7 fixes family membership before launch '
              'precisely so that a')
        print('  result cannot be moved into a second, softer family; a BH q '
              'on the same contrast')
        print('  would be exactly that relocation.')
        ledger.refusals.append(
            f'{len(refused)} screen level(s) refused because they ARE the '
            f'confirmatory contrast ({", ".join(refused)}): '
            'ANALYSIS_PLAN.md §7 does not permit the primary estimand to enter '
            'the BH screen family')
    ledger.screen_q.extend(f'{r["experiment"]}/{r["cell"]}/{r["level"]}'
                           for r in rows if 'q_bh' in r)
    return {'available': True, 'rows': rows, 'bh_family_size': len(pvals),
            'refused_as_confirmatory': refused}


def section_estimation(df: pd.DataFrame, opts: Options, ledger: Ledger,
                       metric: str, gate: list[dict],
                       shared: bool = True) -> dict:
    """§10.10 -- the estimation-only analyses, for one endpoint.

    `shared` controls the three subsections that do not take a metric at all
    (RQ6, the censored endpoints and the secondary/mechanism table): they are
    emitted once, with the first endpoint, rather than repeated identically for
    the second. Everything else here is a function of `metric` and is emitted
    for **each** co-primary endpoint: `auc_score` is co-primary by `DESIGN.md`
    §5.2, and this section used to run for `opts.metrics[0]` alone, so P2 got
    no RQ1, RQ3, RQ5, RQ6, dispersion or screen estimate at all and nothing in
    the output said so.
    """
    h1(f'9. ESTIMATION-ONLY ANALYSES on {metric} -- intervals, no p-values')
    print('  ANALYSIS_PLAN.md §3: every analysis in this section gets a point '
          'estimate and')
    print('  a seed-level bootstrap 95% CI and NO p-value at all. Where a '
          'directional claim')
    print('  is wanted, the licensed form is what the interval excludes.')
    # Per-arm summary tables below are restricted to the target environment.
    # Scores are normalised per environment, so a table that mixed a CartPole
    # source arm with a LunarLander target arm would invite exactly the
    # cross-scale comparison DESIGN.md §5.1 forbids -- and the per-cell
    # survival comparison would silently pit a source arm against a target one.
    tdf = target_side(df[df['env'] == opts.target_env])
    print(f'  Per-arm tables in 9f-9j are restricted to '
          f'{opts.target_env}: {len(tdf)} of {len(df)} run(s). Scores are')
    print('  normalised per environment, so arms on different environments are '
          'never placed')
    print('  in one comparison (DESIGN.md §5.1).')
    out: dict[str, Any] = {'metric': metric,
                           'rq1': sub_rq1(df, opts, ledger, metric),
                           'rq3': sub_rq3(df, opts, ledger, metric, gate),
                           'rq5': sub_rq5(df, opts, ledger, metric)}
    # RQ6's estimand is the episode-prefix score against the single final
    # checkpoint (DESIGN.md §2.4 RQ6). It does not depend on the co-primary
    # endpoint: `metric` reaches only the refusal message, never the
    # computation. Running it once per endpoint printed the identical table and
    # the identical budget-dependence sentence twice, under two different
    # endpoint headings, and wrote the same entry into `sign_changes` twice, so
    # one finding was presented as two results.
    if shared:
        out['rq6'] = sub_rq6(df, opts, ledger, metric)
    else:
        h2('9d. RQ6 -- does the conclusion depend on the budget?')
        print('  RQ6 takes no endpoint argument: its estimand is the '
              'episode-prefix score against')
        print('  the single final checkpoint (DESIGN.md §2.4 RQ6), which is '
              'the same quantity')
        print('  whichever co-primary endpoint this section is running on. It '
              'was emitted once,')
        print(f'  above, with {opts.metrics[0]}, rather than printed twice as '
              'though it were two')
        print('  results.')
        out['rq6'] = {'available': None, 'metric_independent': True,
                      'emitted_with': opts.metrics[0],
                      'note': 'RQ6 does not depend on the co-primary '
                              'endpoint and is emitted once'}
    out['dispersion'] = sub_dispersion(df, opts, ledger, metric)
    if shared:
        out['censored'] = sub_censored(tdf, opts, ledger)
        out['secondary'] = sub_secondary(tdf, opts, ledger)
    else:
        print()
        print('  9f (censored endpoints) and 9g (secondary and mechanism '
              'endpoints) do not take')
        print(f'  an endpoint argument either and were emitted once, above, '
              f'with {opts.metrics[0]}.')
    out['screens'] = sub_screens(tdf, opts, ledger, metric)
    return out


#: The statement a between-cell contrast licenses under one policy. One
#: vocabulary for the pairwise contrasts and for the 2x2 interaction, because
#: the arbitration compares them the same way.
RQ3_POSITIVE = 'positive'
RQ3_NEGATIVE = 'negative'
RQ3_NULL = 'not-distinguishable'
RQ3_NO_WORDING = 'no-wording-licensed'
RQ3_NOT_COMPUTED = 'not-computed'

RQ3_STATEMENT_TEXT: dict[str, str] = {
    RQ3_POSITIVE: 'the interval excludes zero from above',
    RQ3_NEGATIVE: 'the interval excludes zero from below',
    RQ3_NULL: 'the interval contains zero',
    RQ3_NO_WORDING: ('no wording is licensed: the normalised and '
                     'headroom-adjusted scales do not agree, or the adjusted '
                     'scale does not exist'),
    RQ3_NOT_COMPUTED: 'the contrast is not computed',
}


def _interval_statement(lo: Any, hi: Any) -> str:
    if lo is None or hi is None:
        return RQ3_NOT_COMPUTED
    flo, fhi = float(lo), float(hi)
    if not (np.isfinite(flo) and np.isfinite(fhi)):
        return RQ3_NOT_COMPUTED
    if flo > 0:
        return RQ3_POSITIVE
    if fhi < 0:
        return RQ3_NEGATIVE
    return RQ3_NULL


def _rq3_statement(rec: Optional[dict]) -> str:
    """What one policy's leg licenses about one between-cell contrast.

    The two-scale agreement check of `ANALYSIS_PLAN.md` 3 comes first: a
    contrast whose scales disagree licenses no wording under that policy, so
    there is nothing for the other policy to agree or disagree with, and the
    arbitration reports that rather than comparing two intervals it has just
    been told not to read directionally.
    """
    if not rec or rec.get('status') != 'computed':
        return RQ3_NOT_COMPUTED
    if rec.get('scales') != 'agree':
        return RQ3_NO_WORDING
    return _interval_statement(rec.get('ci_lo'), rec.get('ci_hi'))


def _interaction_statement(inter: Optional[dict]) -> str:
    if not inter or not inter.get('available'):
        return RQ3_NOT_COMPUTED
    if not inter.get('wording_licensed'):
        return RQ3_NO_WORDING
    return _interval_statement(inter.get('ci_lo'), inter.get('ci_hi'))


def _arbitrate_statements(common: str, tuned: str, evaluable: bool,
                          reason: str) -> tuple[str, str]:
    """The verdict for one RQ3 contrast, from the two legs' statements."""
    if not evaluable:
        return NOT_EVALUABLE, reason
    if tuned == RQ3_NOT_COMPUTED:
        return NOT_EVALUABLE, ('the contrast is not computed under the '
                               'per-cell tuned policy')
    if common == RQ3_NOT_COMPUTED:
        return NOT_EVALUABLE, ('the contrast is not computed under the common '
                               'configuration, so there is nothing to '
                               'arbitrate')
    if RQ3_NO_WORDING in (common, tuned):
        which = ('both policies license'
                 if common == tuned == RQ3_NO_WORDING
                 else ('the common configuration licenses'
                       if common == RQ3_NO_WORDING
                       else 'the per-cell tuned policy licenses'))
        return NOT_EVALUABLE, (f'{which} no wording for this contrast '
                               f'(ANALYSIS_PLAN.md 3\'s two-scale check), so '
                               f'no RQ3 conclusion exists to assert')
    if common == tuned:
        return AGREES, ''
    return DISAGREES, (f'under the common configuration '
                       f'{RQ3_STATEMENT_TEXT[common]}; under the per-cell '
                       f'tuned policy {RQ3_STATEMENT_TEXT[tuned]}')


def _cross_cell_confound(tuned: TunedPolicy, a: str,
                         b: str) -> tuple[Any, str]:
    """Whether a tuned-leg contrast between two cells adds a hyperparameter.

    `DESIGN.md` 3.3 makes `lr` invariant WITHIN a cell and deliberately varying
    ACROSS cells, so under the secondary policy two cells can differ in their
    learning rate as well as in the factor under study. Returns
    `(True | False | None, note)`; `None` is "not knowable from these runs".
    """
    ca, cb = tuned.config(a), tuned.config(b)
    keys = ('lr', 'target_update')
    if not ca or not cb or any(not ca.get(k) or not cb.get(k) for k in keys):
        return None, ('the configurations of one or both cells are not '
                      'recorded in these runs, so the extra confound cannot '
                      'be checked')
    differing = [k for k in keys if tuple(ca[k]) != tuple(cb[k])]
    if not differing:
        return False, (f'both cells were tuned to {_config_text(ca)}, so this '
                       f'contrast carries no hyperparameter difference beyond '
                       f'the common policy\'s')
    return True, (f'{a} at {_config_text(ca)} against {b} at '
                  f'{_config_text(cb)}: the two cells differ in '
                  f'{", ".join(differing)} as well as in the factor under '
                  f'study')


def section_arbitration(tuned: Optional[TunedPolicy], opts: Options,
                        ledger: Ledger, conf: dict,
                        est_by_metric: dict, gate: list[dict]) -> dict:
    """9t -- the whole of `DESIGN.md` 3.3's arbitration, RQ2 and RQ3 together.

    `ANALYSIS_PLAN.md` 10 has no slot for this section: its twelve items were
    written before 3.3's secondary policy had runs behind it, and the plan
    still describes no place where the two policies are compared. The section
    is emitted anyway, and its absence from 10 is recorded in 12, because the
    alternative is a report that satisfies the letter of 10 while omitting the
    condition 3.3 attaches to its primary conclusion. Section 10 (power) is
    already in this module on the same footing.

    RQ2's leg is computed in 5d, where it belongs: it is the gate on the eight
    tests and a reader who stopped at 5a must not be able to miss it. It is
    restated here so the arbitration can be read in one place.
    """
    tuned = tuned if tuned is not None else TunedPolicy()
    h1('9t. THE DESIGN.md 3.3 ARBITRATION -- COMMON vs PER-CELL-TUNED POLICY')
    print('  DESIGN.md 3.3 declares two hyperparameter policies and '
          'pre-registers an')
    print('  arbitration between them:')
    print(f'    PRIMARY   {POLICY_NAMES[POLICY_COMMON]}: one lr and '
          'target-update rule for')
    print('              all four cells, fixed a priori at lr=5e-4, hard '
          'update every 1000.')
    print(f'    SECONDARY {POLICY_NAMES[POLICY_TUNED]}: each cell\'s own '
          'E3-selected')
    print('              configuration, run as registry E1t/E2t at the same '
          'CONFIRM seeds.')
    print('    RULE      an RQ2 or RQ3 conclusion is asserted ONLY where both '
          'hold. Where')
    print('              they disagree, that disagreement IS the finding and '
          'is reported as')
    print('              one: it is not averaged away and neither leg is '
          'preferred.')
    print()
    print('  This section is not in ANALYSIS_PLAN.md 10\'s list of twelve. '
          'That list predates')
    print('  the secondary policy having any runs behind it, and the omission '
          'is recorded in')
    print('  12 rather than resolved by leaving the arbitration out.')

    # -- where the second leg is -------------------------------------------
    h2('9t-a. the second leg: which cells have tuned arms')
    arb2 = (conf or {}).get('arbitration') or {}
    cell_rows = arb2.get('cells') or [
        {'cell': c, 'tuned_arms': tuned.state(c), 'runs': 0,
         'trained_at': '-', 'selected': '-', 'shares_common_runs': False}
        for c in CELL_ORDER]
    print(table(cell_rows, ('cell', 'tuned_arms', 'runs', 'trained_at',
                            'selected', 'shares_common_runs')))
    print('  own-runs  the cell\'s selection differs from the a priori '
          'configuration, so its')
    print('            tuned arms are their own runs under their own '
          'tuned- labels.')
    print('  shared    the selection equals the a priori configuration, so '
          'the tuned arms\'')
    print('            run digests are identical and the two legs are the '
          'SAME RUNS. They')
    print('            agree by construction; that is a fact about the '
          'selection, not a')
    print('            replication, and it is what makes the tuned stage an '
          'upper bound on')
    print('            cost rather than a doubling.')
    print('  absent    no tuned run. The arbitration is NOT EVALUABLE and '
          'nothing is asserted.')
    for cell in CELL_ORDER:
        if tuned.state(cell) == 'absent' and tuned.reason(cell):
            print(f'    {cell}: {tuned.reason(cell)}')

    # -- RQ2 ----------------------------------------------------------------
    h2('9t-b. RQ2 -- the within-cell delta under both policies')
    rq2_rows = arb2.get('rows') or []
    if rq2_rows:
        print(table(rq2_rows, ('metric', 'cell', 'tuned_arms', 'hl_common',
                               'hl_tuned', 'conclusion_common',
                               'conclusion_tuned', 'verdict', 'assertable')))
        print('  Computed in 5d, restated here. The tuned leg is the same '
              'eight hypotheses')
        print('  through the same code, Holm-adjusted over the same '
              'pre-registered family of')
        print(f'  {CONFIRMATORY_FAMILY_SIZE}; it adds no member to the family '
              'and none to the ledger.')
    else:
        print('  (no confirmatory member was computed, so there is nothing '
              'to arbitrate)')

    # -- RQ3 ----------------------------------------------------------------
    h2('9t-c. RQ3 -- the between-cell contrasts under both policies')
    print('  UNDER THE SECONDARY POLICY EVERY CROSS-CELL CONTRAST CARRIES AN '
          'EXTRA CONFOUND.')
    print('  DESIGN.md 3.3 makes lr invariant WITHIN a cell and deliberately '
          'varying ACROSS')
    print('  cells, so two cells compared under that policy differ in the '
          'factor under study')
    print('  AND in their learning rate. The tuned leg is therefore not a '
          'cleaner version of')
    print('  the common-leg contrast; it is a different and more confounded '
          'one, and it is')
    print('  here only to say whether the common-leg statement survives it. '
          'Where two cells')
    print('  selected the same configuration the extra confound is absent, '
          'and the confound')
    print('  column below says which pairs those are.')

    tuned_frame = tuned.frame()
    rq3: dict[str, Any] = {}
    for metric in sorted(est_by_metric):
        common_rq3 = (est_by_metric.get(metric) or {}).get('rq3') or {}
        h2(f'9t-c({metric}). between-cell contrasts on {metric}')
        if tuned.available:
            # The intensity gate is a property of arch and protocol -- the
            # transferred-parameter fraction -- and not of the learning rate,
            # so the gate computed in 2 for the common policy governs the
            # tuned leg unchanged. It is passed in rather than recomputed so
            # the two legs cannot be blocked on different grounds.
            tuned_rq3 = _rq3_compute(tuned_frame, opts, metric, gate)
        else:
            tuned_rq3 = {'pairs': [], 'interaction': {'available': False,
                                                      'status': 'absent'},
                         'headroom': {}, 'deltas': {}, 'cell_refusals': {}}
        common_pairs = {(r['a'], r['b']): r
                        for r in (common_rq3.get('pairs') or [])}
        tuned_pairs = {(r['a'], r['b']): r for r in tuned_rq3['pairs']}
        rows = []
        for a, b in combinations(CELL_ORDER, 2):
            cr = common_pairs.get((a, b))
            tr = tuned_pairs.get((a, b))
            evaluable = (tuned.state(a) != 'absent'
                         and tuned.state(b) != 'absent')
            reason = '; '.join(
                f'{c}: no tuned arms' for c in (a, b)
                if tuned.state(c) == 'absent') or ''
            c_stmt = _rq3_statement(cr)
            t_stmt = _rq3_statement(tr) if evaluable else RQ3_NOT_COMPUTED
            verdict, why = _arbitrate_statements(c_stmt, t_stmt, evaluable,
                                                 reason)
            confound, cnote = _cross_cell_confound(tuned, a, b)
            assertable = bool(verdict == AGREES
                              and c_stmt not in (RQ3_NOT_COMPUTED,
                                                 RQ3_NO_WORDING)
                              and not tuned.assertion_block)
            rows.append({
                'a': a, 'b': b, 'metric': metric,
                'hl_common': (cr or {}).get('hl'),
                'hl_tuned': (tr or {}).get('hl'),
                'statement_common': c_stmt, 'statement_tuned': t_stmt,
                'verdict': verdict, 'assertable': assertable,
                'cross_cell_confound': confound,
                'confound_note': cnote, 'why': why,
            })
            ledger.arbitration.append(
                f'RQ3 {metric} {a} vs {b}: {verdict}'
                + (f' ({why})' if why else ''))
        print(table(rows, ('a', 'b', 'hl_common', 'hl_tuned',
                           'statement_common', 'statement_tuned', 'verdict',
                           'assertable', 'cross_cell_confound')))
        for r in rows:
            if r['verdict'] == DISAGREES:
                print(f'  {r["a"]} vs {r["b"]}: THE TWO POLICIES DISAGREE. '
                      f'{r["why"]}.')
                print('    That disagreement is the finding for this pair '
                      '(DESIGN.md 3.3); no RQ3')
                print('    conclusion is asserted and neither leg is '
                      'preferred.')
                ledger.refusals.append(
                    f'RQ3 {metric} {r["a"]} vs {r["b"]}: the '
                    f'common and per-cell tuned policies DISAGREE, so no '
                    f'conclusion is asserted; the disagreement is the finding')
            if r['cross_cell_confound']:
                print(f'  {r["a"]} vs {r["b"]}: tuned-leg confound -- '
                      f'{r["confound_note"]}.')

        c_inter = (common_rq3.get('interaction') or {})
        t_inter = (tuned_rq3.get('interaction') or {})
        evaluable = all(tuned.state(c) != 'absent' for c in CELL_ORDER)
        c_stmt = _interaction_statement(c_inter)
        t_stmt = (_interaction_statement(t_inter) if evaluable
                  else RQ3_NOT_COMPUTED)
        verdict, why = _arbitrate_statements(
            c_stmt, t_stmt, evaluable,
            'not every cell has tuned arms, so the 2x2 interaction cannot be '
            'computed under the secondary policy')
        inter_row = {
            'metric': metric,
            'hl_common': c_inter.get('hl'), 'hl_tuned': t_inter.get('hl'),
            'statement_common': c_stmt, 'statement_tuned': t_stmt,
            'verdict': verdict,
            'assertable': bool(verdict == AGREES
                               and c_stmt not in (RQ3_NOT_COMPUTED,
                                                  RQ3_NO_WORDING)
                               and not tuned.assertion_block),
            'why': why,
        }
        print()
        print(f'  2x2 interaction: common {c_stmt}, tuned {t_stmt} -> '
              f'{verdict}'
              + (f'  ({why})' if why else ''))
        print('  The interaction mixes all four cells, so under the secondary '
              'policy it mixes')
        print('  up to four learning rates as well. It is an interval by '
              'design (MDE ~2.7')
        print('  sigma, ANALYSIS_PLAN.md 6) and carries no p-value under '
              'either policy.')
        ledger.arbitration.append(
            f'RQ3 {metric} 2x2 interaction: {verdict}'
            + (f' ({why})' if why else ''))
        rq3[metric] = {'pairs': rows, 'interaction': inter_row}

    # -- what may be asserted ----------------------------------------------
    h2('9t-d. what DESIGN.md 3.3 permits this report to assert')
    rq2_ok = [r for r in rq2_rows if r.get('assertable')]
    rq3_ok = [r for m in rq3 for r in rq3[m]['pairs'] if r.get('assertable')]
    rq3_ok += [rq3[m]['interaction'] for m in rq3
               if rq3[m]['interaction'].get('assertable')]
    rq2_dis = [r for r in rq2_rows if r.get('verdict') == DISAGREES]
    rq3_dis = [r for m in rq3 for r in rq3[m]['pairs']
               if r.get('verdict') == DISAGREES]
    rq3_dis += [rq3[m]['interaction'] for m in rq3
                if rq3[m]['interaction'].get('verdict') == DISAGREES]
    print(f'  RQ2 conclusions assertable   : {len(rq2_ok)} of '
          f'{len(rq2_rows)}')
    print(f'  RQ3 statements assertable    : {len(rq3_ok)}')
    print(f'  disagreements, REPORTED as findings : {len(rq2_dis)} on RQ2, '
          f'{len(rq3_dis)} on RQ3')
    if not rq2_ok and not rq3_ok:
        print()
        print('  ' + '*' * 70)
        print('  NOTHING IS ASSERTED FOR RQ2 OR RQ3 IN THIS REPORT.')
        if not tuned.available:
            print('  The per-cell tuned policy has no runs in the analysis '
                  'set, so the second leg')
            print('  of the pre-registered arbitration does not exist and no '
                  'conclusion of either')
            print('  question clears DESIGN.md 3.3. Numbers above are '
                  'estimates under the common')
            print('  configuration alone and are labelled as such.')
        elif tuned.assertion_block:
            print(f'  {tuned.assertion_block}.')
        else:
            print('  Every contrast either disagrees across the two policies '
                  'or licenses no')
            print('  wording under at least one of them. The disagreements '
                  'above are the finding.')
        print('  ' + '*' * 70)
    else:
        for r in rq2_ok:
            print(f'  RQ2 {r["metric"]}/{r["cell"]}: '
                  f'{CONCLUSION_TEXT[r["conclusion_common"]]}, and it holds '
                  'under both policies.')
        for r in rq3_ok:
            name = (f'{r["a"]} vs {r["b"]}' if 'a' in r else '2x2 interaction')
            print(f'  RQ3 {r["metric"]} {name}: '
                  f'{RQ3_STATEMENT_TEXT[r["statement_common"]]}, under both '
                  'policies'
                  + ('. NOTE the tuned leg also varies lr across these cells'
                     if r.get('cross_cell_confound') else ''))
    return {'policies': dict(POLICY_NAMES),
            'selection': arb2.get('selection') or {},
            'cells': cell_rows,
            'rq2': rq2_rows,
            'rq3': rq3,
            'counts': {'rq2_assertable': len(rq2_ok),
                       'rq3_assertable': len(rq3_ok),
                       'rq2_disagreements': len(rq2_dis),
                       'rq3_disagreements': len(rq3_dis),
                       'tuned_cells': list(tuned.evaluable_cells)},
            'family_size_unchanged': CONFIRMATORY_FAMILY_SIZE,
            'adds_family_members': 0}


def section_power(df: pd.DataFrame, conf: dict, opts: Options,
                  ledger: Ledger) -> dict:
    """§10.10 -- power and minimum detectable effects, at the observed SDs.

    The multipliers are the pre-registered ones from `ANALYSIS_PLAN.md` §6.2.
    They are **not** recomputed here: §6.4 says "The power table is not
    re-tuned after seeing confirmatory results", so re-deriving the multipliers
    from the observed data would itself be a plan violation. What is new is the
    sigma they multiply -- the dispersion actually observed -- and the planning
    SDs of §6.3 are printed beside it, per the §6.4 update rule.
    """
    h1('10. POWER AND MINIMUM DETECTABLE EFFECTS')
    if opts.verify_mde:
        ver = verify_mde_against_statlib()
    else:
        ver = dict(MDE_VERIFICATION)
        ver['note'] = ('verification skipped by --no-mde-verify; the '
                       'pre-registered multipliers stand unverified')
    print('  Multipliers, pre-registered in ANALYSIS_PLAN.md §6.2 and not '
          're-tuned (§6.4).')
    print(f'  Source: {MDE_SOURCE}.')
    if ver.get('ran'):
        print()
        print(f'  Re-derived from statlib at the planned n={ver["n"]} and '
              f'compared (tolerance {MDE_AGREEMENT_TOLERANCE}):')
        print(table(ver['rows'], ('test', 'alpha_level', 'n',
                                  'pre_registered', 'statlib', 'abs_diff',
                                  'agree', 'note'), nd=3))
        if not ver['agree']:
            ledger.deviations.append(
                'the pre-registered MDE multipliers of ANALYSIS_PLAN.md §6.2 '
                'do not reproduce under statlib: ' + MDE_SOURCE)
    else:
        print(f'  NOT verified this invocation: {ver.get("note")}')
        ledger.deviations.append(
            'the MDE multipliers were not verified against statlib: '
            + str(ver.get('note')))
    print(table([{'test': k[0], 'alpha': ('0.05' if k[1] == 'nominal'
                                          else f'{ALPHA_STRICTEST:.5f} '
                                               '(Holm over '
                                               f'{CONFIRMATORY_FAMILY_SIZE})'),
                  'MDE_in_sigma': v} for k, v in MDE_MULTIPLIERS.items()],
                ('test', 'alpha', 'MDE_in_sigma'), nd=2))
    rows = []
    for rec in conf['members']:
        if 'suppressed' in rec:
            rows.append({'metric': rec['metric'], 'cell': rec['cell'],
                         'n': rec['n'], 'note': 'suppressed'})
            continue
        cell, metric = rec['cell'], rec['metric']
        # The unpaired sigma is pooled over THE SEEDS THE TEST USED, taken
        # straight off the paired sample, not over whatever else sits in the
        # two arms. Pooling over the whole arm meant that on any dataset with
        # an incomplete arm the unpaired MDE was reported against a different n
        # from the paired MDE printed beside it, with both rows labelled with
        # the paired n.
        s = _clean(rec.get('scratch_values') or [])
        t = _clean(rec.get('transfer_values') or [])
        sigma_d = rec['sd_delta']
        pooled = float(np.sqrt((sd(s) ** 2 + sd(t) ** 2) / 2.0)) \
            if (np.isfinite(sd(s)) and np.isfinite(sd(t))) else float('nan')
        mde = {
            'paired_nominal': MDE_MULTIPLIERS[('paired', 'nominal')] * sigma_d,
            'paired_holm8': MDE_MULTIPLIERS[('paired', 'holm8')] * sigma_d,
            'unpaired_nominal': (MDE_MULTIPLIERS[('unpaired', 'nominal')]
                                 * pooled),
            'unpaired_holm8': MDE_MULTIPLIERS[('unpaired', 'holm8')] * pooled,
        }
        powered = bool(np.isfinite(mde['paired_holm8'])
                       and mde['paired_holm8'] < UNPOWERED_MDE)
        rows.append({'metric': metric, 'cell': cell, 'n': rec['n'],
                     'n_pooled': int(min(len(s), len(t))),
                     'sigma_delta': sigma_d, 'sigma_pooled': pooled,
                     'observed_delta': rec['mean_delta'], **mde,
                     'powered': powered,
                     'note': ('' if powered else
                              'MDE at the corrected alpha reaches or exceeds '
                              '1.0 score unit, the whole distance from random '
                              'play to solved: NOT POWERED')})
    print()
    print(table(rows, ('metric', 'cell', 'n', 'n_pooled', 'sigma_delta',
                       'sigma_pooled', 'observed_delta', 'paired_nominal',
                       'paired_holm8', 'unpaired_nominal', 'unpaired_holm8',
                       'powered')))
    print('  n is the paired sample; n_pooled is the per-arm count the '
          'unpaired sigma was')
    print('  pooled over. They are the same by construction on a complete '
          'arm, and printing')
    print('  both is what makes an incomplete one visible.')
    print('  MDE units are normalised score. A cell is flagged NOT POWERED '
          'when its MDE at')
    print(f'  the Holm-corrected alpha reaches {UNPOWERED_MDE} score units -- '
          'which by construction is')
    print('  the entire distance from a random policy to the registered '
          'threshold')
    print('  (ANALYSIS_PLAN.md §6.3). Which cells are powered is therefore a '
          'property of the')
    print('  observed dispersion, not a verdict discovered after the test.')
    for r in rows:
        if r.get('note'):
            print(f'  {r["metric"]}/{r["cell"]}: {r["note"]}')

    h2('10b. observed dispersion against the planning inputs '
       '(ANALYSIS_PLAN.md §6.3-6.4)')
    prows = []
    for key, planned in PLANNING_SDS.items():
        cell, cond = key.rsplit(' ', 1)
        cdf = df[df['cell'] == cell]
        arm = (scratch_arm(cdf, opts) if cond == 'scratch'
               else primary_transfer_arm(cdf, opts))
        obs = sd(_clean(arm['final_score']))
        prows.append({'arm': key, 'planned_sd': planned, 'observed_sd': obs,
                      'ratio': (obs / planned if np.isfinite(obs)
                                and planned > 0 else None)})
    print(table(prows, ('arm', 'planned_sd', 'observed_sd', 'ratio')))
    print('  The planning SDs come from the published runs, which used a '
          'different protocol,')
    print('  budget and exploration schedule, so they were a planning input '
          'and never a')
    print('  prediction. They are shown beside the observed values and are '
          'not updated here.')
    for r in prows:
        if r['ratio'] is not None and np.isfinite(r['ratio']):
            print('  ' + phrase_dispersion(f'{r["arm"]} (observed)',
                                           r['observed_sd'],
                                           f'{r["arm"]} (planned)',
                                           r['planned_sd']))
    return {'per_member': rows, 'planning_comparison': prows,
            'multipliers': {f'{k[0]}/{k[1]}': v
                            for k, v in MDE_MULTIPLIERS.items()},
            'multiplier_source': MDE_SOURCE, 'verification': ver}


def section_ledger(ledger: Ledger, conf: dict) -> dict:
    """§10.11 -- the multiplicity ledger. Printed on every invocation."""
    h1('11. MULTIPLICITY LEDGER')
    rows = [
        {'family': 'Confirmatory',
         'members': f'{CONFIRMATORY_FAMILY_SIZE} (4 cells x 2 co-primary '
                    'endpoints)',
         'procedure': 'Holm-Bonferroni',
         'adjusted_alpha': f'step-down from {ALPHA_STRICTEST:.5f}'},
        {'family': 'Screens (E3-E8, E12)',
         'members': f'{len(ledger.screen_q)} level estimate(s) with a q-value',
         'procedure': 'Benjamini-Hochberg q, orientation only',
         'adjusted_alpha': 'no assertion permitted'},
        {'family': 'Everything else',
         'members': f'{len(ledger.estimation)} analysis section(s)',
         'procedure': 'none -- estimation only',
         'adjusted_alpha': 'no p-values emitted'},
    ]
    print(table(rows, ('family', 'members', 'procedure', 'adjusted_alpha')))
    print()
    print(f'  confirmatory tests actually computed : '
          f'{len(ledger.confirmatory)} of {CONFIRMATORY_FAMILY_SIZE}')
    for name in ledger.confirmatory:
        print(f'    {name}')
    print(f'  confirmatory members suppressed      : '
          f'{len(ledger.suppressed)}')
    for name in ledger.suppressed:
        print(f'    {name}')
    print(f'  suppressed outside the family        : '
          f'{len(ledger.other_suppressed)}')
    for name in ledger.other_suppressed:
        print(f'    {name}')
    print(f'  analyses carrying NO p-value         : '
          f'{len(ledger.estimation)}')
    for name in ledger.estimation:
        print(f'    {name}')
    print(f'  refusals                             : {len(ledger.refusals)}')
    for name in ledger.refusals:
        print(f'    {name}')
    print(f'  DESIGN.md 3.3 arbitration entries    : '
          f'{len(ledger.arbitration)}')
    for name in ledger.arbitration:
        print(f'    {name}')
    print()
    # The guarantee, computed rather than asserted in prose. The arbitration
    # re-tests the SAME hypotheses under the secondary policy, so the family
    # must still be the pre-registered eight and every member of it must have
    # come from the common policy. If either fails, the multiplicity ledger is
    # wrong and 12 says so, which is the only honest place for it.
    members = (conf or {}).get('members') or []
    foreign = [m for m in members
               if m.get('policy') not in (None, POLICY_COMMON)]
    accounted = len(ledger.confirmatory) + len(ledger.suppressed)
    print(f'  family members emitted               : {len(members)} of '
          f'{CONFIRMATORY_FAMILY_SIZE} (a --metric restriction lowers')
    print('                                         this; the family size is '
          'a constant)')
    print(f'  members contributed by the tuned leg : {len(foreign)} '
          '(must be 0)')
    print(f'  members accounted for in this ledger : {accounted} '
          f'(computed {len(ledger.confirmatory)} + suppressed '
          f'{len(ledger.suppressed)})')
    ledger_ok = (not foreign
                 and len(members) <= CONFIRMATORY_FAMILY_SIZE
                 and accounted == len(members))
    if ledger_ok:
        print('  The DESIGN.md 3.3 arbitration is a CONJUNCTION over these '
              'same members, not a')
        print('  second family: "both policies reject" has as its rejection '
              'region the')
        print('  intersection of the two legs\', which is never larger '
              'than either, so the FWER')
        print(f'  stays bounded by Holm over {CONFIRMATORY_FAMILY_SIZE}. Its '
              'entries are listed above and counted')
        print('  separately, and none of them is a family member.')
    else:
        print('  LEDGER INVARIANT FAILED: the confirmatory family is not the '
              'pre-registered')
        print('  set. See 12.')
        ledger.deviations.append(
            f'multiplicity ledger invariant failed: {len(members)} family '
            f'member(s) of which {len(foreign)} did not come from the common '
            f'policy, and {accounted} accounted for in the ledger. '
            f'ANALYSIS_PLAN.md 7 fixes the family at '
            f'{CONFIRMATORY_FAMILY_SIZE} before launch')
    print()
    print('  Holm step-down thresholds for the family of '
          f'{CONFIRMATORY_FAMILY_SIZE}:')
    print('    ' + ', '.join(
        f'{ALPHA / (CONFIRMATORY_FAMILY_SIZE - i):.5f}'
        for i in range(CONFIRMATORY_FAMILY_SIZE)))
    print('  Family membership is fixed by ANALYSIS_PLAN.md §7 before launch '
          'and is read')
    print('  from this module\'s constants, never from an argument, which is '
          'what prevents a')
    print('  result from being rescued by relocating it into a family of one.')
    return {'families': rows, 'confirmatory': ledger.confirmatory,
            'suppressed': ledger.suppressed,
            'suppressed_outside_family': ledger.other_suppressed,
            'estimation_only': ledger.estimation,
            'refusals': ledger.refusals,
            'arbitration': ledger.arbitration,
            'family_members_emitted': len(members),
            'family_members_from_tuned_leg': len(foreign),
            'family_size_invariant_ok': bool(ledger_ok),
            'screen_q_count': len(ledger.screen_q)}


def section_deviations(ledger: Ledger, stamped: bool) -> dict:
    """§10.12 -- deviations between the run data and the plan."""
    h1('12. DEVIATIONS AND PLAN TENSIONS')
    if not ledger.deviations:
        print('  No deviation detected between the run data and the plan.')
    else:
        print(f'  {len(ledger.deviations)} deviation(s) detected:')
        for i, d in enumerate(ledger.deviations, 1):
            print(f'    {i}. {d}')
    print()
    if ledger.tensions:
        print('  Tensions INSIDE the plan, resolved conservatively and '
              'recorded rather than')
        print('  resolved silently (STANDING_INSTRUCTIONS preamble: raise the '
              'conflict):')
        for i, t in enumerate(ledger.tensions, 1):
            print(f'    {i}. {t}')
        print()
    print('  Standing tensions this module resolves the same way on every '
          'invocation:')
    print('    - ANALYSIS_PLAN.md §5 licenses a log-rank test while §7 permits '
          'p-values only')
    print('      inside the confirmatory family. The statistic is emitted, the '
          'p-value is')
    print('      withheld.')
    print('    - §3 asks for Brown-Forsythe "for continuity" while forbidding '
          'a dispersion')
    print('      p-value. W is emitted, the p-value is withheld.')
    print('    - DESIGN.md 3.3 requires an arbitration between the two '
          'hyperparameter')
    print('      policies before an RQ2 or RQ3 conclusion is asserted, and '
          'ANALYSIS_PLAN.md')
    print('      10\'s list of twelve reported items has no slot for it. The '
          'arbitration is')
    print('      emitted as 5d and 9t rather than omitted to keep the list at '
          'twelve.')
    print('    - §7 permits Benjamini-Hochberg q for screens, which requires '
          'p-values; that')
    print('      is the one declared exception to "no p-value outside the '
          'confirmatory')
    print('      family", and screen q-values are labelled orientation-only '
          'wherever printed.')
    if stamped:
        print()
        print('  ' + '*' * 70)
        print(f'  {VALIDATION_STAMP}')
        print('  At least one analysed arm has n < '
              f'{MIN_N_FOR_INFERENCE}. No single-seed or two-seed number may '
              'be quoted,')
        print('  compared, or used to choose between hypotheses '
              '(ANALYSIS_PLAN.md §9,')
        print('  STANDING_INSTRUCTIONS S8). A single seed can show that a run '
              'executes; it')
        print('  cannot show that an arm differs.')
        print('  ' + '*' * 70)
    return {'deviations': ledger.deviations, 'tensions': ledger.tensions,
            'validation_stamp': stamped}


# ===========================================================================
# 6. Self-test. The prose generators are the guard against the published
#    paper's "broader" for a narrower spread, so they are tested rather than
#    trusted; the statistical primitives are checked against values that can be
#    verified by hand.
# ===========================================================================

def _arbitration_fixture(deltas: dict, *, prefix: str = '',
                         lr: float = 5e-4, target_update: str = 'hard',
                         experiments: str = 'E1;E2',
                         seeds: Sequence[int] = tuple(range(10)),
                         base: float = 0.40) -> pd.DataFrame:
    """A minimal two-arm frame per cell, for the arbitration self-tests.

    Only the columns the arm selectors read. `deltas` maps a cell to the
    transfer-minus-scratch shift the fixture puts there, so a leg that should
    conclude an effect and a leg that should conclude nothing are the same
    code with different numbers.
    """
    rows = []
    for cell in CELL_ORDER:
        arch, rule = cell.split('-')
        for cond, stem in (('scratch', 'scratch'), ('transfer', 'transfer')):
            for seed in seeds:
                shift = deltas[cell] if cond == 'transfer' else 0.0
                if isinstance(shift, (list, tuple)):
                    shift = shift[seed % len(shift)]
                value = (base + 0.02 * CELL_ORDER.index(cell)
                         + 0.01 * (seed % 4) + shift)
                rows.append({
                    'label': f'{prefix}{stem}-{cell}', 'arm': f'{stem}-{cell}',
                    'experiments': experiments, 'arch': arch,
                    'target_rule': rule, 'condition': cond, 'cell': cell,
                    'env': registry.TARGET_ENV,
                    'source_env': ('' if cond == 'scratch'
                                   else registry.SOURCE_ENV),
                    'seed': seed, 'seed_block': 'CONFIRM',
                    'transfer_set': registry.PROTOCOL['transfer_set'],
                    'input_policy': registry.PROTOCOL['input_policy'],
                    'head_policy': registry.PROTOCOL['head_policy'],
                    'freeze_group': registry.PROTOCOL['freeze_group'],
                    'freeze_updates': registry.PROTOCOL['freeze_updates'],
                    'lr': lr, 'target_update': target_update,
                    'final_score': value, 'auc_score': value * 0.8,
                    'run_dir': f'{prefix}{stem}-{cell}/seed{seed}',
                    'plan_hash': '0' * 32})
    return pd.DataFrame(rows)


def _arbitration_self_test() -> list[tuple[str, bool, str]]:
    """Assertions on `DESIGN.md` 3.3's arbitration, both states.

    The first block is the defect this section closes: with no tuned runs the
    common policy can be Holm-significant in all eight members and NOTHING may
    be asserted. Before the arbitration existed, that state printed eight
    confirmed effects.
    """
    out: list[tuple[str, bool, str]] = []

    def add(name: str, cond: bool, detail: str = '') -> None:
        out.append((name, bool(cond), detail))

    opts = Options(per_seed='<fixture>', metrics=CONFIRMATORY_ENDPOINTS,
                   experiments=None, target_env=registry.TARGET_ENV,
                   source_env=registry.SOURCE_ENV,
                   interface_env=registry.INTERFACE_ENV,
                   allow_intensity_confound=False, source_policy='valid',
                   n_boot=200, boot_seed=BOOT_SEED, json_out=None)
    effect = {c: 0.15 for c in CELL_ORDER}
    null = {c: (0.02, -0.02) for c in CELL_ORDER}
    common = _arbitration_fixture(effect)
    tuned_same = _arbitration_fixture(effect, prefix=TUNED_LABEL_PREFIX,
                                      lr=1e-3, experiments='E1t;E2t')
    tuned_null = _arbitration_fixture(null, prefix=TUNED_LABEL_PREFIX,
                                      lr=1e-3, experiments='E1t;E2t')

    # -- the two policies are separated by label, and only by label ---------
    both = pd.concat([common, tuned_same], ignore_index=True)
    add('the common policy keeps every row that is not tuned-labelled',
        len(common_policy_rows(both)) == len(common),
        f'{len(common_policy_rows(both))} vs {len(common)}')
    add('the tuned policy is exactly the tuned-labelled rows',
        len(tuned_policy_rows(both)) == len(tuned_same))
    add('a shared run belongs to both policies',
        len(common_policy_rows(common)) == len(common)
        and bool(claims_tuned_experiment(tuned_same).all()))

    def leg(frame: pd.DataFrame) -> TunedPolicy:
        tp = TunedPolicy()
        for cell in CELL_ORDER:
            tp.frames[cell] = frame[frame['cell'] == cell]
            tp.cells[cell] = {'cell': cell, 'state': 'own-runs', 'reason': '',
                              'rows': int(len(tp.frames[cell])),
                              'config': _config_summary(tp.frames[cell], opts),
                              'config_text': '', 'shares_common_runs': False}
        return tp

    def family(frame: pd.DataFrame, tp: Optional[TunedPolicy]):
        ledger = Ledger()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            conf = section_confirmatory(frame, opts, ledger, tuned=tp)
        return conf, ledger, buf.getvalue()

    # -- 1. no tuned runs: significant under one policy, asserted by none ---
    conf, ledger, text = family(common, None)
    members = conf['members']
    add('the fixture is Holm-significant in all eight members under the '
        'common policy',
        len(members) == CONFIRMATORY_FAMILY_SIZE
        and all(m.get('significant_holm') for m in members),
        str([m.get('p_holm') for m in members]))
    add('with no tuned runs NO member is asserted',
        not any(m.get('asserted') for m in members))
    add('with no tuned runs every verdict is not-evaluable',
        all(m.get('arbitration_verdict') == NOT_EVALUABLE for m in members))
    add('with no tuned runs the report says so in words',
        'NO RQ2 CONCLUSION IS ASSERTED' in text)

    # -- 2. the family stays eight whatever the second leg does ------------
    add('the family is eight members and eight ledger entries',
        len(members) == CONFIRMATORY_FAMILY_SIZE
        and len(ledger.confirmatory) + len(ledger.suppressed)
        == CONFIRMATORY_FAMILY_SIZE,
        f'{len(members)}, {len(ledger.confirmatory)}, '
        f'{len(ledger.suppressed)}')
    add('no member of the family came from the tuned leg',
        all(m.get('policy') == POLICY_COMMON for m in members))

    conf2, ledger2, text2 = family(common, leg(tuned_same))
    add('running the second leg adds no family member',
        len(conf2['members']) == CONFIRMATORY_FAMILY_SIZE
        and len(ledger2.confirmatory) == CONFIRMATORY_FAMILY_SIZE
        and conf2['arbitration']['adds_family_members'] == 0,
        f'{len(conf2["members"])}, {len(ledger2.confirmatory)}')
    add('running the second leg adds no ledger entry to the family',
        len(ledger2.confirmatory) == len(ledger.confirmatory)
        and len(ledger2.suppressed) == len(ledger.suppressed))
    add('the arbitration keeps its own count, apart from the family',
        len(ledger2.arbitration) == CONFIRMATORY_FAMILY_SIZE
        and len(ledger.arbitration) == CONFIRMATORY_FAMILY_SIZE)

    # -- 3. agreement, and its consequence ---------------------------------
    add('two legs that conclude the same thing AGREE',
        all(r['verdict'] == AGREES for r in conf2['arbitration']['rows']),
        str([r['verdict'] for r in conf2['arbitration']['rows']]))
    add('an agreeing verdict makes the conclusion assertable',
        all(r['assertable'] for r in conf2['arbitration']['rows']))

    # -- 4. disagreement is a finding, not a suppression -------------------
    conf3, ledger3, text3 = family(common, leg(tuned_null))
    rows3 = conf3['arbitration']['rows']
    add('a second leg that concludes nothing DISAGREES',
        all(r['verdict'] == DISAGREES for r in rows3),
        str([r['verdict'] for r in rows3]))
    add('a disagreement asserts nothing',
        not any(r['assertable'] for r in rows3)
        and not any(m.get('asserted') for m in conf3['members']))
    add('a disagreement is REPORTED, in words, as the finding',
        'THE TWO POLICIES DISAGREE' in text3
        and 'that disagreement IS the' in text3)
    add('a disagreement reaches the ledger as a refusal',
        sum(1 for r in ledger3.refusals if 'DISAGREE' in r)
        == CONFIRMATORY_FAMILY_SIZE, str(len(ledger3.refusals)))
    add('the common leg is unchanged by what the tuned leg found',
        [repr(m.get('hl')) for m in conf3['members']]
        == [repr(m.get('hl')) for m in members])

    # -- 5. a shared cell is the same runs, so its legs are identical ------
    shared = TunedPolicy()
    for cell in CELL_ORDER:
        shared.frames[cell] = common[common['cell'] == cell]
        shared.cells[cell] = {'cell': cell, 'state': 'shared', 'reason': '',
                              'rows': int(len(shared.frames[cell])),
                              'config': _config_summary(shared.frames[cell],
                                                        opts),
                              'config_text': '', 'shares_common_runs': True}
    conf4, ledger4, _ = family(common, shared)
    tuned_recs = {(m['metric'], m['cell']): m
                  for m in conf4['arbitration']['tuned_members']}
    base_recs = {(m['metric'], m['cell']): m for m in conf4['members']}
    add('a cell that shares its runs produces an identical second leg',
        all(repr(tuned_recs[k].get('hl')) == repr(base_recs[k].get('hl'))
            and repr(tuned_recs[k].get('p_signflip'))
            == repr(base_recs[k].get('p_signflip')) for k in base_recs))
    add('a cell that shares its runs agrees with itself',
        all(r['verdict'] == AGREES
            for r in conf4['arbitration']['rows']))

    # -- 6. the verdict function, at its boundaries ------------------------
    add('two rejections in opposite directions do NOT agree',
        _arbitrate(CONCLUSION_UP, CONCLUSION_DOWN, True, '')[0] == DISAGREES)
    add('two nulls agree',
        _arbitrate(CONCLUSION_NULL, CONCLUSION_NULL, True, '')[0] == AGREES)
    add('a suppressed second leg is not-evaluable, never agreement',
        _arbitrate(CONCLUSION_UP, CONCLUSION_NONE, True, '')[0]
        == NOT_EVALUABLE)
    add('an absent second leg is not-evaluable',
        _arbitrate(CONCLUSION_UP, CONCLUSION_UP, False, 'no runs')[0]
        == NOT_EVALUABLE)
    add('a Holm-significant member with no direction has no conclusion',
        _member_conclusion({'p_holm': 0.001, 'significant_holm': True,
                            'hl': 0.0, 'mean_delta': 0.0})
        == CONCLUSION_NONE)

    # -- 7. RQ3's statement and its two-scale gate -------------------------
    add('an RQ3 interval that excludes zero from above is positive',
        _rq3_statement({'status': 'computed', 'scales': 'agree',
                        'ci_lo': 0.1, 'ci_hi': 0.4}) == RQ3_POSITIVE)
    add('an RQ3 contrast whose scales disagree licenses no wording',
        _rq3_statement({'status': 'computed', 'scales': 'DISAGREE',
                        'ci_lo': 0.1, 'ci_hi': 0.4}) == RQ3_NO_WORDING)
    add('a contrast licensing no wording is not-evaluable, not agreement',
        _arbitrate_statements(RQ3_NO_WORDING, RQ3_NO_WORDING, True, '')[0]
        == NOT_EVALUABLE)
    add('two RQ3 legs pointing opposite ways DISAGREE',
        _arbitrate_statements(RQ3_POSITIVE, RQ3_NEGATIVE, True, '')[0]
        == DISAGREES)

    # -- 8. the cross-cell confound the secondary policy introduces --------
    mixed = leg(tuned_same)
    mixed.cells['mlp-vanilla']['config'] = {'lr': ('0.0005',),
                                            'target_update': ('hard',)}
    mixed.cells['mlp-double']['config'] = {'lr': ('0.001',),
                                           'target_update': ('hard',)}
    flag, note = _cross_cell_confound(mixed, 'mlp-vanilla', 'mlp-double')
    add('a cross-cell contrast at two learning rates is flagged confounded',
        flag is True and 'lr' in note, note)
    mixed.cells['mlp-double']['config'] = {'lr': ('0.0005',),
                                           'target_update': ('hard',)}
    flag2, note2 = _cross_cell_confound(mixed, 'mlp-vanilla', 'mlp-double')
    add('two cells tuned to the same configuration carry no extra confound',
        flag2 is False, note2)
    flag3, _ = _cross_cell_confound(TunedPolicy(), 'mlp-vanilla',
                                    'mlp-double')
    add('an unknown configuration is not reported as "no confound"',
        flag3 is None)
    return out


def self_test() -> int:
    """Assertions on the primitives and on every directional phrase."""
    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = '') -> None:
        print(f'  {"ok  " if cond else "FAIL"}  {name}'
              + (f'   {detail}' if detail and not cond else ''))
        if not cond:
            failures.append(name)

    h1('SELF-TEST')

    h2('metric roles')
    check('final_score is co-primary', metric_role('final_score') == CO_PRIMARY)
    check('auc_score is co-primary', metric_role('auc_score') == CO_PRIMARY)
    for m in ('train_return', 'td_loss', 'epsilon', 'updates', 'wall_time_s',
              'final_return', 'td_loss_final100'):
        check(f'{m} is descriptive', metric_role(m) == DESCRIPTIVE)
        try:
            require_confirmatory(m)
            check(f'{m} refused for a confirmatory test', False)
        except MetricRoleError:
            check(f'{m} refused for a confirmatory test', True)
    check('prefix_score_500 is secondary',
          metric_role('prefix_score_500') == SECONDARY)
    check('an unknown metric is unclassified, not assumed testable',
          metric_role('some_new_thing') == 'unclassified')
    try:
        require_confirmatory('some_new_thing')
        check('an unknown metric is refused', False)
    except MetricRoleError:
        check('an unknown metric is refused', True)

    h2('sign-flip test')
    sf = sign_flip_test([1.0, 2.0, 3.0])
    check('n=3 all-positive gives p = 2/8', abs(sf['p'] - 0.25) < 1e-12,
          str(sf))
    check('n=3 minimum attainable p is 0.25',
          abs(sf['min_attainable_p'] - 0.25) < 1e-12)
    sf10 = sign_flip_test([0.1] * 10)
    check('n=10 all-same-sign gives p = 2/1024',
          abs(sf10['p'] - 2 / 1024) < 1e-12, str(sf10))
    sf_sym = sign_flip_test([1.0, -1.0])
    check('a symmetric sample gives p = 1', abs(sf_sym['p'] - 1.0) < 1e-12,
          str(sf_sym))
    check('the exact null distribution has 2^n points',
          len(all_signed_means(np.array([1.0, 2.0, 3.0, 4.0]))) == 16)
    check('the plan\'s stated bar is reachable only by unanimity at n=10',
          (2 / 1024) < ALPHA_STRICTEST)

    h2('Hodges-Lehmann')
    check('HL of a symmetric sample is its centre',
          abs(hodges_lehmann_paired([-2.0, -1.0, 0.0, 1.0, 2.0])) < 1e-12)
    check('HL is shift-equivariant',
          abs(hodges_lehmann_paired([1.0, 2.0, 3.0])
              - hodges_lehmann_paired([0.0, 1.0, 2.0]) - 1.0) < 1e-12)
    rng = np.random.default_rng(7)
    stack = rng.normal(size=(200, 9))
    check('the vectorised HL agrees with the scalar one everywhere',
          float(np.max(np.abs(hl_vec(stack)
                              - np.array([hodges_lehmann_paired(r)
                                          for r in stack])))) < 1e-12)
    check('the vectorised mean and median agree with the scalar ones',
          float(np.max(np.abs(mean_vec(stack) - stack.mean(axis=1)))) < 1e-12
          and float(np.max(np.abs(median_vec(stack)
                                  - np.median(stack, axis=1)))) < 1e-12)

    h2('Holm over a fixed family size')
    adj = holm_adjust({'a': 0.001, 'b': 0.02, 'c': 0.04}, 8)
    check('smallest p is multiplied by m', abs(adj['a'] - 0.008) < 1e-12,
          str(adj))
    check('Holm is monotone', adj['a'] <= adj['b'] <= adj['c'], str(adj))
    check('a family of 8 with 3 observed still corrects for 8',
          abs(adj['b'] - min(1.0, 7 * 0.02)) < 1e-12, str(adj))

    h2('Brunner-Munzel')
    bm = brunner_munzel([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], n_boot=500)
    check('identical samples give theta = 0.5', abs(bm['theta'] - 0.5) < 1e-9,
          str(bm))
    bm2 = brunner_munzel([10, 11, 12], [1, 2, 3], n_boot=500)
    check('complete dominance gives theta = 1', abs(bm2['theta'] - 1.0) < 1e-9,
          str(bm2))

    h2('Clopper-Pearson')
    lo, hi = clopper_pearson(0, 10)
    check('0/10 gives (0, ~0.308)', lo == 0.0 and abs(hi - 0.3085) < 1e-3,
          f'{lo},{hi}')
    lo, hi = clopper_pearson(10, 10)
    check('10/10 gives (~0.692, 1)', abs(lo - 0.6915) < 1e-3 and hi == 1.0,
          f'{lo},{hi}')

    h2('Kaplan-Meier with censoring')
    km = kaplan_meier([1, 2, 3, 4], [True, False, True, False])
    check('censored observations are not events', km['events'] == 2, str(km))
    check('survival falls only at event times',
          abs(km['curve'][0]['survival'] - 0.75) < 1e-12, str(km['curve']))
    km_all_cens = kaplan_meier([5, 5, 5], [False, False, False])
    check('all-censored gives survival 1 and no median',
          km_all_cens['median'] is None
          and km_all_cens['curve'][-1]['survival'] == 1.0)

    h2('Jonckheere concordance')
    jt_up = jonckheere_effect([[1, 2], [3, 4], [5, 6]])
    jt_dn = jonckheere_effect([[5, 6], [3, 4], [1, 2]])
    check('a monotone increase gives +1', abs(jt_up['standardised'] - 1) < 1e-9)
    check('a monotone decrease gives -1', abs(jt_dn['standardised'] + 1) < 1e-9)

    h2('BCa interval')
    x = np.array([0.1, 0.2, 0.15, 0.3, 0.25, 0.05, 0.35, 0.2, 0.1, 0.4])
    res = bootstrap_statistic(x, lambda a: float(np.mean(a)), n_boot=2000,
                              vec=mean_vec)
    check('the interval brackets the estimate',
          res['lo'] <= res['estimate'] <= res['hi'], str(res))
    small = bootstrap_statistic(np.array([1.0, 2.0]),
                                lambda a: float(np.mean(a)))
    check('n<3 emits no interval', not np.isfinite(small['lo']), str(small))

    h2('prose generated from the numbers -- reversing the arguments reverses '
       'the word')
    a = phrase_dispersion('A', 0.3, 'B', 0.1)
    b = phrase_dispersion('A', 0.1, 'B', 0.3)
    check('a larger SD reads "wider"', 'wider' in a and 'narrower' not in a, a)
    check('a smaller SD reads "narrower"',
          'narrower' in b and 'wider' not in b, b)
    check('the same SD reads neither', 'wider' not in
          phrase_dispersion('A', 0.2, 'B', 0.2)
          and 'narrower' not in phrase_dispersion('A', 0.2, 'B', 0.2))
    up = phrase_direction(0.5, 'x', 'y')
    dn = phrase_direction(-0.5, 'x', 'y')
    check('a positive value reads "above"', 'above' in up and 'below' not in up,
          up)
    check('a negative value reads "below"', 'below' in dn and 'above' not in dn,
          dn)
    check('zero reads as level', 'level with' in phrase_direction(0.0, 'x', 'y'))
    mag_a = phrase_magnitude_comparison('P', -0.9, 'Q', 0.2)
    mag_b = phrase_magnitude_comparison('P', 0.1, 'Q', 0.2)
    check('a bigger magnitude reads "larger"',
          'larger' in mag_a and 'smaller' not in mag_a, mag_a)
    check('a smaller magnitude reads "smaller"',
          'smaller' in mag_b and 'larger' not in mag_b, mag_b)
    check('an interval above zero reads positive',
          'is positive' in phrase_interval_verdict(0.1, 0.2))
    check('an interval below zero reads negative',
          'is negative' in phrase_interval_verdict(-0.2, -0.1))
    check('an interval covering zero reads "not distinguishable"',
          'not distinguishable' in phrase_interval_verdict(-0.1, 0.2))
    check('a negative bound yields an exclusion bound',
          'worse than 0.3000' in phrase_exclusion_bound(-0.3),
          phrase_exclusion_bound(-0.3))
    check('a non-negative bound excludes every degradation',
          'every degradation is excluded' in phrase_exclusion_bound(0.01))
    check('unanimity is detected',
          'same direction' in phrase_unanimity([1, 2, 3]))
    check('a split is detected', 'split' in phrase_unanimity([1, -2, 3]))
    check('a rising trend reads "rises"', 'rises' in phrase_trend(0.5, 'wind'))
    check('a falling trend reads "falls"', 'falls' in phrase_trend(-0.5, 'wind'))

    h2('pre-registered constants')
    check('the family has exactly 8 members', CONFIRMATORY_FAMILY_SIZE == 8)
    check('there are exactly 2 co-primary endpoints',
          len(CONFIRMATORY_ENDPOINTS) == 2)
    check('there are exactly 4 cells', len(CELL_ORDER) == 4)
    check('the equivalence margin is 0.05', EQUIVALENCE_MARGIN == 0.05)
    check('the C4 bound is -0.10', C4_LOWER_BOUND == -0.10)
    check('the strictest Holm alpha is 0.00625',
          abs(ALPHA_STRICTEST - 0.00625) < 1e-12)

    h2('the guards that used to fail open')
    check('an unrecognised boolean token is refused, not read as absent',
          parse_boolean('no') == (False, None)
          and parse_boolean('yes') == (False, None),
          str((parse_boolean('no'), parse_boolean('yes'))))
    check('a blank cell is a genuine absence',
          parse_boolean('') == (True, None)
          and parse_boolean(float('nan')) == (True, None))
    check('the tokens aggregate.py writes still parse',
          [parse_boolean(v)[1] for v in (True, False, 'True', 'False', 1, 0)]
          == [True, False, True, False, True, False])

    frame = pd.DataFrame({
        'seed': [0, 1, 2, 2, 3],
        'final_score': [0.1, 0.2, 0.3, 0.9, float('nan')]})
    arm = arm_by_seed(frame, 'final_score')
    check('a duplicated (arm, seed) is reported, not collapsed',
          arm['duplicates'] == [2] and 2 not in arm['values'], str(arm))
    check('a duplicated seed with differing values is flagged as conflicting',
          arm['conflicting'] == [2], str(arm))
    check('a seed whose metric is absent is reported as a gap',
          arm['metric_missing'] == [3], str(arm))
    check('rows are counted as rows', arm['n_rows'] == 5)
    both = pd.DataFrame({'seed': [0, 1], 'final_score': [float('nan')] * 2})
    pair = paired_by_seed(both, both, 'final_score')
    check('a seed missing the metric in BOTH arms does not vanish',
          pair['metric_missing'] == [0, 1] and len(pair['seeds']) == 0,
          str(pair))
    notes = pairing_problems(paired_by_seed(frame, frame, 'final_score'),
                             'final_score')
    check('the pairing problems are stated in prose: both duplicated arms '
          'and the gap',
          len(notes) == 3 and sum('more than one row' in s for s in notes) == 2
          and any('no final_score value' in s for s in notes), str(notes))

    flat = bootstrap_statistic(np.full(6, 0.25),
                               lambda a: float(np.mean(a)), n_boot=500,
                               vec=mean_vec)
    check('a constant arm yields a degenerate interval, flagged as such',
          flat['degenerate'] and flat['lo'] == flat['hi'], str(flat))
    check('a degenerate interval is not silently a BCa one',
          flat['method'] == 'degenerate', str(flat))
    varied = bootstrap_statistic(np.array([0.1, 0.4, 0.2, 0.35, 0.15, 0.3]),
                                 lambda a: float(np.mean(a)), n_boot=1000,
                                 vec=mean_vec)
    check('an ordinary sample is not flagged degenerate',
          not varied['degenerate'] and varied['lo'] < varied['hi'],
          str(varied))
    check('every interval carries the method that produced it',
          'method' in varied and varied['method'] in ('BCa', 'percentile'),
          str(varied))

    check('a constant input gives no correlation rather than a warning',
          not np.isfinite(correlation('pearson', [1.0, 1.0, 1.0],
                                      [1.0, 2.0, 3.0])))
    check('spearman is computed where it exists',
          abs(correlation('spearman', [1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
              - 1.0) < 1e-12)

    km_plain = kaplan_meier([1.0, 2.0, 3.0], [True, True, True])
    km_late = kaplan_meier([1.0, 2.0, 3.0], [True, True, True],
                           entry=[0.0, 0.0, 2.5])
    check('delayed entry changes the risk set, and only then',
          km_plain['curve'][0]['at_risk'] == 3
          and km_late['curve'][0]['at_risk'] == 2
          and not km_plain['delayed_entry'] and km_late['delayed_entry'],
          f'{km_plain["curve"][0]}, {km_late["curve"][0]}')
    check('a non-finite time is counted, not silently dropped',
          kaplan_meier([1.0, float('nan')], [True, False])
          ['dropped_nonfinite'] == 1)

    check('the mechanism table carries the DESIGN.md §5.5 plasticity signals',
          all(metric_role(m) == MECHANISM for m in
              ('effective_rank', 'stable_rank', 'param_norm_total',
               'param_norm_trunk', 'q_max')))

    h2('the DESIGN.md 3.3 arbitration')
    for name, cond, detail in _arbitration_self_test():
        check(name, cond, detail)

    h2('the estimators against statlib.py')
    prim = verify_primitives_against_statlib(n_boot=500)
    if prim['ran']:
        for row in prim['rows']:
            check(f'{row["primitive"]} matches statlib', row['agree'],
                  f'{row["here"]} vs {row["statlib"]}')
    else:
        check('statlib.py is importable, so the estimators can be checked',
              False, str(prim['note']))

    h2('MDE table provenance')
    ver = verify_mde_against_statlib()
    if ver['ran']:
        for row in ver['rows']:
            check(f'the {row["test"]}/{row["alpha_level"]} multiplier '
                  f'reproduces under statlib', row['agree'],
                  f'statlib {row["statlib"]} vs pre-registered '
                  f'{row["pre_registered"]}')
    else:
        check('the MDE multipliers were verified against statlib', False,
              str(ver['note']))
    # Exact equality, not a slack of 0.011. This is a transcription check
    # against a table of four printed numbers, so any slack at all is a second
    # tolerance sitting behind the first one, and a drift of 0.01 in the
    # transcription is exactly what it would hide.
    check('the multipliers match the pre-registered ANALYSIS_PLAN.md §6.2 '
          'values, digit for digit',
          MDE_MULTIPLIERS == {('paired', 'nominal'): 1.00,
                              ('paired', 'holm8'): 1.54,
                              ('unpaired', 'nominal'): 1.41,
                              ('unpaired', 'holm8'): 1.88},
          str(MDE_MULTIPLIERS))
    check('the paired multiplier is smaller than the unpaired one, which is '
          'why pairing is primary',
          MDE_MULTIPLIERS[('paired', 'nominal')]
          < MDE_MULTIPLIERS[('unpaired', 'nominal')])
    print(f'  source: {MDE_SOURCE}')

    print()
    if failures:
        print(f'{len(failures)} SELF-TEST FAILURE(S): {failures}')
        return 1
    print('all self-tests passed')
    return 0


# ===========================================================================
# 7. CLI
# ===========================================================================

def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()
                if k != 'reps'}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if not np.isfinite(f) else f
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if obj is None or isinstance(obj, str):
        return obj
    return str(obj)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--per-seed', default=os.path.join('runs', 'per_seed.csv'),
                   help='the pinned per-seed table from aggregate.py')
    p.add_argument('--json', dest='json_out', default=None,
                   help='also write every number to this JSON file')
    p.add_argument('--experiments', nargs='*', default=None,
                   help='restrict to runs belonging to these experiment ids '
                        '(a run may belong to several)')
    p.add_argument('--metric', action='append', default=None,
                   help='restrict the computed co-primary endpoints; the '
                        'family size stays 8 by pre-registration. Validated '
                        'against METRIC_ROLES rather than by an argparse '
                        'choice list, so the refusal message names the role '
                        'and the plan section that forbids it')
    p.add_argument('--target-env', default=registry.TARGET_ENV)
    p.add_argument('--source-env', default=registry.SOURCE_ENV)
    p.add_argument('--interface-env', default=registry.INTERFACE_ENV,
                   help='the interface-change-only environment carrying C4')
    p.add_argument('--allow-intensity-confound', action='store_true',
                   help='compute cross-architecture contrasts whose '
                        'transferred-parameter fractions differ by more than '
                        f'{INTENSITY_TOLERANCE} (DESIGN.md §3.1). The override '
                        'is stamped into the output')
    p.add_argument('--source-policy', choices=('valid', 'pooled'),
                   default='valid',
                   help="'valid': the primary estimand, valid sources only. "
                        "'pooled': the pre-declared SECONDARY of DESIGN.md "
                        "§4.3, pooled over source competence -- never called "
                        "ITT, and recorded as a deviation")
    p.add_argument('--audit-root', default=None,
                   help='the run tree the ANALYSIS_PLAN.md §10.1 audit gate '
                        'runs over. Defaults to the directory holding the '
                        'per-seed table; the gate reports itself as not '
                        'evaluated when no run tree is there')
    p.add_argument('--allow-audit-failure', action='store_true',
                   help='emit the report over a FAILED audit. '
                        'ANALYSIS_PLAN.md §10.1 permits this only with an '
                        'explicit override, and the override is stamped into '
                        'the output, into the JSON and into §12. Named as '
                        'report.py and tables.py name the same override')
    p.add_argument('--seeds', default=None,
                   help='the seed set the runs were launched at, passed '
                        'through to audit.py. Reducing it is the '
                        'STANDING_INSTRUCTIONS S8 validation invocation and is '
                        'recorded by the audit rather than assumed here')
    p.add_argument('--overrides', nargs='*', default=None,
                   help='launch-level overrides that were in force, as '
                        'field=value, passed through to audit.py')
    p.add_argument('--selection', dest='selection_path', default=None,
                   help='path to the DESIGN.md 3.3 tuning-selection artifact. '
                        'Defaults to the one stored under the run tree '
                        '(_jobs/tuning_selection.json), which is where '
                        'tuning.py writes it and where registry.py enumerates '
                        'the tuned arms from. Read only, never written, and '
                        'its content address is re-verified on every read')
    p.add_argument('--no-mde-verify', action='store_true',
                   help='skip re-deriving the ANALYSIS_PLAN.md §6.2 MDE '
                        'multipliers with statlib. The pre-registered values '
                        'are used either way (§6.4 forbids re-tuning); the '
                        'verification is what says whether they still '
                        'reproduce')
    p.add_argument('--n-boot', type=int, default=N_BOOT)
    p.add_argument('--boot-seed', type=int, default=BOOT_SEED)
    p.add_argument('--self-test', action='store_true',
                   help='run the primitive and prose assertions and exit')
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return self_test()

    # The metric-role gate, applied before anything is read from disk. A
    # descriptive metric is refused here rather than tested, which is the
    # mechanical fix for the published paper testing a metric its own text
    # declared descriptive-only (DESIGN.md §5.4).
    for m in (args.metric or ()):
        try:
            require_confirmatory(m)
        except MetricRoleError as exc:
            print(f'stats.py: {exc}')
            return 2

    opts = Options(
        per_seed=args.per_seed,
        metrics=tuple(args.metric) if args.metric else CONFIRMATORY_ENDPOINTS,
        experiments=tuple(args.experiments) if args.experiments else None,
        target_env=args.target_env, source_env=args.source_env,
        interface_env=envs.parse(args.interface_env).canonical(),
        allow_intensity_confound=args.allow_intensity_confound,
        source_policy=args.source_policy, n_boot=args.n_boot,
        boot_seed=args.boot_seed, json_out=args.json_out,
        audit_root=args.audit_root or '',
        override_audit=bool(args.allow_audit_failure),
        audit_seeds=args.seeds,
        audit_overrides=tuple(args.overrides) if args.overrides else (),
        verify_mde=not args.no_mde_verify,
        selection_path=args.selection_path or '')
    ledger = Ledger()

    try:
        raw = load_per_seed(opts.per_seed, ledger)
    except (FileNotFoundError, ValueError) as exc:
        print(f'stats.py: {exc}')
        return 1

    df = in_experiments(raw, opts.experiments)
    if not len(df):
        # The message names the filter that actually emptied the selection.
        # It used to blame --experiments unconditionally, so a header-only CSV
        # with no --experiments flag reported "no runs match --experiments
        # None", which points the reader at a filter that was never applied.
        if not len(raw):
            print(f'stats.py: {opts.per_seed} has a header and no data rows. '
                  'There is nothing to report on, and an empty report is not '
                  'produced.')
        else:
            print(f'stats.py: none of the {len(raw)} row(s) in '
                  f'{opts.per_seed} belong to --experiments '
                  f'{list(opts.experiments or ())}; refusing to report on an '
                  'empty selection.')
        return 1

    # TUNE seeds may never enter a reported estimate (ANALYSIS_PLAN.md §8).
    n_before = len(df)
    tune_seeds = tuple(sorted(int(s) for s in
                              df.loc[df['seed_block'] == 'TUNE',
                                     'seed'].dropna().unique()))
    df = df[df['seed_block'] != 'TUNE']
    n_tune = n_before - len(df)
    opts.tune_runs_excluded = n_tune
    opts.tune_seeds_excluded = tune_seeds

    # DESIGN.md 3.3's secondary policy is a SECOND LEG, not extra rows in the
    # first. A cell whose tuned configuration differs from the a priori one
    # has two runs per (condition, seed) in this table -- one per policy --
    # and leaving them together would give every arm two rows for every seed,
    # which `arm_by_seed` refuses as ambiguous and which would suppress the
    # entire confirmatory family the moment the tuned stage was run. The tuned
    # rows are therefore set aside here and analysed in 5d and 9t as the leg
    # they are. A SHARED run stays in both: it is one run that both policies
    # declare, and removing it would delete the common-policy arm.
    tuned_rows_all = tuned_policy_rows(df)
    df = common_policy_rows(df)
    opts.tuned_rows_set_aside = int(len(tuned_rows_all))
    if not len(df) and len(tuned_rows_all):
        # A selection of tuned arms alone. The arbitration is a comparison
        # between two policies, so a report over the second leg by itself is
        # not the secondary analysis: it is the first leg missing, and every
        # verdict in it would be not-evaluable for a reason that is an
        # artefact of the invocation rather than of the data.
        print(f'stats.py: all {len(tuned_rows_all)} selected row(s) are '
              f'tuned arms of the secondary policy of DESIGN.md 3.3, and '
              f'none is a common-policy run. That section arbitrates BETWEEN '
              f'the two policies, so a report on one of them alone is not '
              f'produced. Select the common-policy experiments as well (E1, '
              f'E2 beside E1t, E2t), or drop --experiments.')
        return 1

    report: dict[str, Any] = {'invocation': {
        'argv': list(sys.argv), 'cwd': os.getcwd(),
        'per_seed': opts.per_seed, 'experiments': opts.experiments,
        'metrics': list(opts.metrics), 'target_env': opts.target_env,
        'source_env': opts.source_env, 'interface_env': opts.interface_env,
        'source_policy': opts.source_policy,
        'allow_intensity_confound': opts.allow_intensity_confound,
        'override_audit': opts.override_audit,
        'tune_runs_excluded': n_tune,
        'tune_seeds_excluded': list(tune_seeds),
        'tuned_rows_set_aside': opts.tuned_rows_set_aside,
        'selection_path': opts.selection_path}}

    print('stats.py -- executing ANALYSIS_PLAN.md §10, sections 1-12, '
          'plus the DESIGN.md 3.3')
    print('  arbitration (5d and 9t), which §10\'s list has no slot for')
    print(f'  invocation: {" ".join(sys.argv)}')
    if opts.tuned_rows_set_aside:
        print(f'  {opts.tuned_rows_set_aside} run(s) carry a '
              f'{TUNED_LABEL_PREFIX!r} arm label and are the SECONDARY policy '
              'of DESIGN.md 3.3.')
        print('  They are set aside from the primary analysis set and '
              'analysed in 5d and 9t as')
        print('  the second leg of the arbitration, never pooled into the '
              'first.')

    # §10.1 first, and it is a gate: "If the audit fails, nothing below is
    # emitted without an explicit override that is stamped into the output."
    audit_result = section_audit(opts, ledger)
    report['s1a_audit'] = audit_result
    if audit_result['ran'] and not audit_result['ok'] \
            and not opts.override_audit:
        print()
        print('  REFUSING to emit sections 1b-12: the audit of the run tree '
              'FAILED and')
        print('  ANALYSIS_PLAN.md §10.1 permits nothing below an audit '
              'failure without an')
        print('  explicit override. Run `python experiments/audit.py '
              f'--out-root {audit_result["root"]}`')
        print('  for the findings, fix them, or pass --allow-audit-failure to '
              'analyse anyway with')
        print('  the override stamped into every page. If the failure is SEED '
              'COMPLETENESS on a')
        print('  validation tree, --seeds is the honest fix: it tells the '
              'audit what was actually')
        print('  launched (STANDING_INSTRUCTIONS S8) instead of overriding a '
              'true finding.')
        if opts.json_out:
            with open(opts.json_out, 'w', encoding='utf-8') as fh:
                json.dump(_json_safe(report), fh, indent=1, sort_keys=False)
        return 3

    report['s1_provenance'] = section_provenance(df, opts, ledger)
    report['s1b_reference_returns'] = section_reference_returns(
        df, opts, ledger)
    report['s2_inventory'] = section_inventory(df, opts, ledger)

    # The analysis set. Primary is valid sources only (DESIGN.md §4.3).
    analysis = df
    if opts.source_policy == 'valid' and 'source_valid' in df.columns:
        drop = df[df['source_valid'] == False]
        if len(drop):
            analysis = df[df['source_valid'] != False]
            print()
            print(f'  ANALYSIS SET: valid sources only (the primary estimand). '
                  f'{len(drop)} run(s) with an')
            print('  invalid source are excluded; their seeds are listed in '
                  '§2d and they are never')
            print('  silently dropped. Pass --source-policy pooled for the '
                  'pre-declared secondary.')
            lost = sorted(set(str(v) for v in drop['label'].dropna().unique()))
            cells = sorted(set(str(v) for v in drop['cell'].dropna().unique()))
            for lab in lost:
                gone = drop[drop['label'] == lab]
                print(f'    {lab}: {len(gone)} run(s), seeds '
                      f'{sorted(int(s) for s in gone["seed"].unique())}')
            # Recorded as a deviation, not only printed. Removing an arm from
            # the analysis set changes which members of the confirmatory family
            # can be computed at all, and §12 used to say "No deviation
            # detected" while the source-validity filter had removed a whole
            # transfer arm and with it two of the eight members and four of the
            # six control conditions in that cell.
            ledger.deviations.append(
                f'analysis-set exclusion (DESIGN.md §4.3): {len(drop)} run(s) '
                f'across arm(s) {lost} in cell(s) {cells} have an invalid '
                'source and are outside the primary estimand. Members of the '
                'confirmatory family and control conditions that depend on '
                'those arms are therefore not computable, and their absence '
                'below is this exclusion, not a null')
    elif opts.source_policy == 'pooled':
        print()
        print('  ANALYSIS SET: POOLED OVER SOURCE COMPETENCE -- the '
              'pre-declared SECONDARY of')
        print('  DESIGN.md §4.3. This is NOT the primary estimand and is NOT '
              'intent-to-treat:')
        print('  source competence is known before the target run begins, so '
              'it is not a')
        print('  post-randomisation compliance event. Every number below pools '
              'transfer from a')
        print('  competent source with transfer from a source that never '
              'learned -- which is')
        print('  the published study\'s actual error, reproduced here '
              'deliberately and labelled.')
        ledger.deviations.append(
            'analysis set is POOLED OVER SOURCE COMPETENCE (DESIGN.md §4.3 '
            'secondary), not the primary valid-sources-only estimand')

    # The second leg of DESIGN.md 3.3's arbitration, resolved once and passed
    # down, so 5d and 9t cannot disagree about which runs it is. The
    # source-validity filter of 4.3 applies to it exactly as to the primary
    # set: a tuned transfer arm drawn from a source that never learned is
    # outside the primary estimand under either policy.
    tuned_analysis = tuned_rows_all
    if (opts.source_policy == 'valid'
            and 'source_valid' in tuned_rows_all.columns):
        tuned_analysis = tuned_rows_all[
            tuned_rows_all['source_valid'] != False]
    tuned_policy = resolve_tuned_policy(analysis, tuned_analysis, opts, ledger)
    # Not a numbered section: a record of WHICH runs the second leg is,
    # resolved once before any section runs, so 5d and 9t cannot
    # disagree about it and a reader can check the resolution itself.
    report['tuned_policy'] = {
        'policy': TUNED_POLICY_NAME,
        'rows_set_aside': opts.tuned_rows_set_aside,
        'rows_in_analysis_set': int(len(tuned_analysis)),
        'cells': {c: {k: v
                      for k, v in (tuned_policy.cells.get(c) or {}).items()
                      if k != 'config'}
                  for c in CELL_ORDER},
        'evaluable_cells': list(tuned_policy.evaluable_cells),
        'selection_readable': tuned_policy.selection is not None,
        'selection_note': tuned_policy.selection_note,
        'assertion_block': tuned_policy.assertion_block,
        'notes': list(tuned_policy.notes)}

    primary_metric = opts.metrics[0]
    report['s3_descriptives'] = {
        m: section_descriptives(analysis, opts, ledger, m)
        for m in opts.metrics}
    report['s4_convergence'] = section_convergence(analysis, opts, ledger)
    conf = section_confirmatory(analysis, opts, ledger,
                                tuned=tuned_policy)
    report['s5_confirmatory'] = conf
    report['s6_equivalence'] = section_equivalence(conf, analysis, opts, ledger)
    report['s7_controls'] = {
        m: section_controls(analysis, opts, ledger, m) for m in opts.metrics}
    # Both co-primary endpoints reach §8 and §9. `s8_c4` and `s9_estimation`
    # keep the primary endpoint's shape because report.py reads them
    # positionally; the full set is beside them under `*_by_metric`.
    gate_rows = report['s2_inventory'].get('intensity_gate', [])
    c4_by_metric = {m: section_c4(analysis, opts, ledger, m)
                    for m in opts.metrics}
    est_by_metric = {}
    for i, m in enumerate(opts.metrics):
        est_by_metric[m] = section_estimation(analysis, opts, ledger, m,
                                              gate_rows, shared=(i == 0))
    report['s8_c4'] = c4_by_metric[primary_metric]
    report['s8_c4_by_metric'] = c4_by_metric
    report['s9_estimation'] = est_by_metric[primary_metric]
    report['s9_estimation_by_metric'] = est_by_metric
    report['s9t_arbitration'] = section_arbitration(
        tuned_policy, opts, ledger, conf, est_by_metric, gate_rows)
    report['s10_power'] = section_power(analysis, conf, opts, ledger)
    report['s11_ledger'] = section_ledger(ledger, conf)

    ns = [r['n'] for r in conf['members'] if r.get('n') is not None]
    stamped = bool(ns) and min(ns) < MIN_N_FOR_INFERENCE
    report['s12_deviations'] = section_deviations(ledger, stamped)

    if opts.json_out:
        with open(opts.json_out, 'w', encoding='utf-8') as fh:
            json.dump(_json_safe(report), fh, indent=1, sort_keys=False)
        print()
        print(f'  every number above also written to {opts.json_out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
