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
* **A single-seed number quoted as a result.** With n<3 no test and no
  interval is emitted and the output is stamped
  `PIPELINE VALIDATION - NOT A RESULT` (`ANALYSIS_PLAN.md` §9,
  `STANDING_INSTRUCTIONS.md` S8).

Input is `runs/per_seed.csv` as produced by `aggregate.py`; the column names
that module pins are the interface. Output is the twelve sections of
`ANALYSIS_PLAN.md` §10, in that order, and optionally the same content as JSON.

    python experiments/stats.py --per-seed runs/per_seed.csv
    python experiments/stats.py --per-seed runs/per_seed.csv --json out.json
    python experiments/stats.py --self-test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
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
from src.dqn import envs                                          # noqa: E402

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
MDE_MULTIPLIERS: dict[tuple[str, str], float] = {
    ('paired', 'nominal'): 1.00,
    ('paired', 'holm8'): 1.54,
    ('unpaired', 'nominal'): 1.39,
    ('unpaired', 'holm8'): 1.87,
}

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
    'td_error_abs': MECHANISM,
    'cka_transfer_vs_scratch': MECHANISM,
    'cka_drift': MECHANISM,
    'dead_unit_frac': MECHANISM,
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
        f'confirmatory test only on the co-primary endpoints '
        f'{list(CONFIRMATORY_ENDPOINTS)}. Refusing. '
        f'(This is the mechanical fix for the published §V.A/§V.B '
        f'contradiction: a t-test on a metric §V.B called descriptive-only.)')


# ===========================================================================
# 2. Statistical primitives. Non-parametric throughout; nothing here assumes
#    normality, and nothing here computes a variance-standardised effect size
#    of the Cohen's-d family (`ANALYSIS_PLAN.md` §8).
# ===========================================================================

def _clean(x: Iterable[float]) -> np.ndarray:
    a = np.asarray(list(x), dtype=float)
    return a[np.isfinite(a)]


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
                'note': 'too few finite bootstrap replicates'}
    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
    pct = (float(np.percentile(reps, lo_q)), float(np.percentile(reps, hi_q)))
    frac = float(np.mean(reps < theta_hat))
    jack = np.asarray(jack, dtype=float)
    jack = jack[np.isfinite(jack)]
    note = ''
    if frac <= 0.0 or frac >= 1.0 or len(jack) < 3:
        return {'lo': pct[0], 'hi': pct[1], 'method': 'percentile',
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
                    'note': 'BCa denominator degenerate; percentile reported'}
        out.append(float(sps.norm.cdf(z0 + (z0 + z) / denom)))
    lo = float(np.percentile(reps, 100 * min(max(out[0], 1e-6), 1 - 1e-6)))
    hi = float(np.percentile(reps, 100 * min(max(out[1], 1e-6), 1 - 1e-6)))
    if lo > hi:
        lo, hi = hi, lo
    return {'lo': lo, 'hi': hi, 'method': 'BCa', 'z0': z0, 'a': a,
            'note': note}


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
                'n': n, 'method': 'suppressed',
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
            'reps': reps}


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
    rng = np.random.default_rng(seed)
    ts = []
    for _ in range(n_boot):
        xb = x[rng.integers(0, nx, nx)]
        yb = y[rng.integers(0, ny, ny)]
        tb, sb = theta_se(xb, yb)
        if sb > 1e-12:
            ts.append((tb - theta) / sb)
    if se <= 1e-12 or len(ts) < 100:
        return {'theta': theta, 'lo': float('nan'), 'hi': float('nan'),
                'nx': nx, 'ny': ny, 'se': se,
                'note': 'degenerate standard error (no overlap variation); '
                        'no interval emitted'}
    tq = np.percentile(np.asarray(ts), [100 * (1 - alpha / 2),
                                        100 * alpha / 2])
    lo = max(0.0, theta - tq[0] * se)
    hi = min(1.0, theta - tq[1] * se)
    return {'theta': theta, 'lo': float(lo), 'hi': float(hi), 'nx': nx,
            'ny': ny, 'se': se, 'note': 'bootstrap-t'}


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


def kaplan_meier(times: Sequence[float], events: Sequence[bool]) -> dict:
    """Kaplan-Meier survivor function for right-censored event times.

    Censored observations contribute their time at risk and are never treated
    as events, which is the whole point: imputing the budget both biases the
    estimate and creates a tie mass that degrades every rank statistic
    downstream (`ANALYSIS_PLAN.md` §5).
    """
    t = np.asarray(list(times), dtype=float)
    e = np.asarray(list(events), dtype=bool)
    keep = np.isfinite(t)
    t, e = t[keep], e[keep]
    n = len(t)
    if n == 0:
        return {'n': 0, 'events': 0, 'curve': [], 'median': None}
    order = np.argsort(t, kind='stable')
    t, e = t[order], e[order]
    curve = []
    surv = 1.0
    at_risk = n
    i = 0
    while i < n:
        ti = t[i]
        d = 0
        c = 0
        while i < n and t[i] == ti:
            if e[i]:
                d += 1
            else:
                c += 1
            i += 1
        if d > 0 and at_risk > 0:
            surv *= (1.0 - d / at_risk)
        curve.append({'t': float(ti), 'at_risk': int(at_risk), 'events': d,
                      'censored': c, 'survival': float(surv)})
        at_risk -= (d + c)
    median = next((row['t'] for row in curve if row['survival'] <= 0.5), None)
    return {'n': n, 'events': int(e.sum()), 'curve': curve, 'median': median}


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
    n1, n2 = len(t1), len(t2)
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
    if not (np.isfinite(sd_a) and np.isfinite(sd_b)) or sd_b <= 0:
        return f'{name_a} vs {name_b}: dispersion ratio not estimable'
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

#: Cell order from the registry, so report rows follow the design's factorial
#: order rather than alphabetical accident.
CELL_ORDER: tuple[str, ...] = tuple(f'{a}-{r}' for a, r in registry.CELLS)


def file_md5(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    h = hashlib.md5()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class Ledger:
    """The multiplicity ledger, accumulated as the report is produced.

    `ANALYSIS_PLAN.md` §7 requires it printed on every invocation, "so the
    count is a recorded fact rather than a claim".
    """

    confirmatory: list[str] = field(default_factory=list)
    suppressed: list[str] = field(default_factory=list)
    screen_q: list[str] = field(default_factory=list)
    estimation: list[str] = field(default_factory=list)
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


def load_per_seed(path: str, ledger: Ledger) -> pd.DataFrame:
    """Read the pinned per-seed table; refuse a table missing its contract."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'{path} not found. Run `python experiments/aggregate.py` first; '
            f'this module never recomputes a per-run scalar itself, so that no '
            f'number in the paper has two definitions.')
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f'{path} is missing pinned columns {missing}. The per-seed schema '
            f'is a contract between aggregate.py and this module; refusing to '
            f'guess a substitute.')
    truth = {True: True, False: False, 'True': True, 'False': False,
             'true': True, 'false': False, 1: True, 0: False}
    for col in ('source_valid', 'metrics_contiguous', 'freeze_verified',
                'git_dirty') + tuple(f'censored_{t}' for t, _ in THRESHOLD_LEVELS):
        if col in df.columns:
            df[col] = df[col].map(lambda v: truth.get(v, None))
    absent = [c for c in ('transferred_param_fraction', 'jumpstart_score',
                          'probe_jumpstart_score', 'within_run_sd',
                          'convergence_slope', 'source_final_score')
              if c not in df.columns]
    if absent:
        ledger.deviations.append(
            f'per_seed.csv lacks optional columns {absent}; the sections that '
            f'need them report as unavailable rather than approximating them')
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


def paired_by_seed(a: pd.DataFrame, b: pd.DataFrame, metric: str) -> dict:
    """Match two arms on seed. Reports incompleteness, never drops silently.

    `DESIGN.md` §1: "One seed was dropped from one arm with no stated rule."
    Seeds present in one arm only are returned and surface in the report.
    """
    def by_seed(d: pd.DataFrame) -> dict[int, float]:
        out: dict[int, float] = {}
        for _, r in d.iterrows():
            v = r.get(metric)
            if pd.notna(v):
                out[int(r['seed'])] = float(v)
        return out

    xa, xb = by_seed(a), by_seed(b)
    common = sorted(set(xa) & set(xb))
    return {'seeds': common,
            'a': np.array([xa[s] for s in common], dtype=float),
            'b': np.array([xb[s] for s in common], dtype=float),
            'only_a': sorted(set(xa) - set(xb)),
            'only_b': sorted(set(xb) - set(xa)),
            'n_a': len(xa), 'n_b': len(xb)}


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

def section_provenance(df: pd.DataFrame, opts: Options,
                       ledger: Ledger) -> dict:
    """§10.1 -- provenance and the pre-registration hash.

    The plan is hashed into every manifest. If the hash in the data differs
    from the current file, the pre-registration no longer covers the analysis
    and every confirmatory result below is exploratory. That is stated loudly
    rather than left for a reader to notice.
    """
    h1('1. PROVENANCE AND PLAN HASH')
    current = file_md5(PLAN_FILE)
    design = file_md5(DESIGN_FILE)
    hashes = sorted(set(str(v) for v in df['plan_hash'].dropna().unique()))
    print(f'  per-seed table       : {opts.per_seed}')
    print(f'  table md5            : {file_md5(opts.per_seed)}')
    print(f'  rows                 : {len(df)}')
    print(f'  ANALYSIS_PLAN.md md5 : {current}   (current file)')
    print(f'  DESIGN.md md5        : {design}')
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
    return {'plan_md5_current': current, 'design_md5': design,
            'plan_md5_in_data': hashes, 'table_md5': file_md5(opts.per_seed),
            'rows': int(len(df)), 'exploratory': exploratory,
            'n_boot': opts.n_boot, 'boot_seed': opts.boot_seed}


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
        comp.append({'cell': cell, 'n_scratch': len(ss), 'n_transfer': len(ts),
                     'paired_seeds': len(set(ss) & set(ts)),
                     'scratch_only': ','.join(str(x) for x in
                                              sorted(set(ss) - set(ts))) or '-',
                     'transfer_only': ','.join(str(x) for x in
                                               sorted(set(ts) - set(ss))) or '-',
                     'transfer_labels': ';'.join(labels) or '-'})
        if len(labels) > 1:
            ledger.refusals.append(
                f'{cell}: {len(labels)} distinct labels match the primary '
                f'protocol ({labels}); the primary arm is ambiguous')
        if set(ss) != set(ts):
            ledger.deviations.append(
                f'{cell}: scratch and transfer seed sets differ; the arm is '
                f'incomplete and DESIGN.md §8.4 refuses a partial arm')
    print(table(comp, ('cell', 'n_scratch', 'n_transfer', 'paired_seeds',
                       'scratch_only', 'transfer_only', 'transfer_labels')))
    out['completeness'] = comp

    h2('2c. seed blocks')
    blk = []
    for name, g in df.groupby('seed_block', dropna=False):
        blk.append({'seed_block': name, 'runs': len(g),
                    'seeds': ','.join(str(int(s)) for s in
                                      sorted(g['seed'].unique()))})
    print(table(blk, ('seed_block', 'runs', 'seeds')))
    tune = df[df['seed_block'] == 'TUNE']
    if len(tune):
        print(f'  REFUSAL: {len(tune)} run(s) sit in the TUNE block. '
              f'ANALYSIS_PLAN.md §8 forbids any reported estimate computed on '
              f'TUNE seeds (selection leakage). They are excluded from every '
              f'estimate below.')
        ledger.refusals.append(f'{len(tune)} TUNE-block runs excluded from all '
                               f'estimates (selection leakage)')
    donors = df[df['seed_block'].isin(SOURCE_ONLY_BLOCKS)]
    if len(donors):
        print(f'  {len(donors)} run(s) sit in a source-only block '
              f'{list(SOURCE_ONLY_BLOCKS)}. DESIGN.md §3.4 bars C4SRC from '
              f'target-side estimation and')
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
              f'provenance in the block scheme of DESIGN.md §3.4 is unstated.')
        ledger.deviations.append(f'{len(unknown)} run(s) in no declared seed '
                                 f'block')
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
              f'seeds are listed above, never dropped silently (DESIGN.md '
              f'§4.3).')
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
    """
    h1(f'3. DESCRIPTIVES on {metric} (normalised score; random policy = 0, '
       f'threshold = 1)')
    ledger.est(f'descriptives on {metric}')
    rows = []
    # Grouped by environment as well as by arm. Scores are normalised per
    # environment, so rows from different environments are not comparable and
    # must not be presented as though they were (DESIGN.md §5.1); the env
    # column makes that visible instead of leaving it to be inferred.
    for (env, cell, cond, label), g in df.groupby(
            ['env', 'cell', 'condition', 'label'], dropna=False):
        x = _clean(g[metric])
        if not len(x):
            continue
        boot = bootstrap_statistic(x, lambda a: float(np.mean(a)),
                                   opts.n_boot, opts.boot_seed, vec=mean_vec)
        rows.append({'env': env, 'cell': cell, 'condition': cond,
                     'label': label, 'seed_block':
                         '/'.join(sorted(set(str(b) for b in
                                             g['seed_block'].dropna()))),
                     'n': len(x), 'mean': float(np.mean(x)), 'sd': sd(x),
                     'median': float(np.median(x)),
                     'ci_lo': boot['lo'], 'ci_hi': boot['hi'],
                     'min': float(np.min(x)), 'max': float(np.max(x))})
    order = {c: i for i, c in enumerate(CELL_ORDER)}
    rows.sort(key=lambda r: (str(r['env']), order.get(r['cell'], 99),
                             str(r['condition']), str(r['label'])))
    print(table(rows, ('env', 'cell', 'condition', 'label', 'seed_block', 'n',
                       'mean', 'sd', 'median', 'ci_lo', 'ci_hi', 'min',
                       'max')))
    print('  CI: bias-corrected-and-accelerated seed-level bootstrap on the '
          'mean. No')
    print('  normality-assuming interval appears anywhere in this module '
          '(ANALYSIS_PLAN.md §8).')

    h2('3b. scratch baseline, threshold and headroom per cell')
    hrows = []
    for cell in CELL_ORDER:
        s = _clean(scratch_arm(df[df['cell'] == cell], opts)[metric])
        if not len(s):
            continue
        m = float(np.mean(s))
        hrows.append({'cell': cell, 'n': len(s), 'scratch_mean': m,
                      'scratch_sd': sd(s), 'threshold': 1.0,
                      'headroom': 1.0 - m})
    print(table(hrows, ('cell', 'n', 'scratch_mean', 'scratch_sd',
                        'threshold', 'headroom')))
    print('  headroom = 1.0 - scratch mean: what remains between this cell and '
          'the')
    print('  registered solved threshold. A cell with little headroom cannot '
          'gain much and')
    print('  can lose a great deal, which is why every RQ3 statement below '
          'carries it.')
    if len(hrows) > 1:
        worst = min(hrows, key=lambda r: r['headroom'])
        best = max(hrows, key=lambda r: r['headroom'])
        print('  ' + phrase_direction(best['headroom'] - worst['headroom'],
                                      f'{best["cell"]} headroom',
                                      f'{worst["cell"]} headroom'))
    sds = {r['cell']: r['scratch_sd'] for r in hrows}
    if len(sds) > 1:
        pairs = sorted(sds.items(), key=lambda kv: (kv[1] if
                                                    np.isfinite(kv[1]) else 0))
        print('  ' + phrase_dispersion(f'{pairs[-1][0]} scratch', pairs[-1][1],
                                       f'{pairs[0][0]} scratch', pairs[0][1]))
    return {'per_arm': rows, 'headroom': hrows}


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
    for (cell, cond, label), g in df.groupby(['cell', 'condition', 'label'],
                                             dropna=False):
        s = _clean(g['convergence_slope'])
        if not len(s):
            continue
        boot = bootstrap_statistic(s, lambda a: float(np.median(a)),
                                   opts.n_boot, opts.boot_seed, vec=median_vec)
        k = int(np.sum(s > 0))
        lo, hi = clopper_pearson(k, len(s))
        moving = bool(np.isfinite(boot['lo']) and np.isfinite(boot['hi'])
                      and (boot['lo'] > 0 or boot['hi'] < 0))
        rows.append({'cell': cell, 'condition': cond, 'label': label,
                     'n': len(s), 'median_slope': boot['estimate'],
                     'ci_lo': boot['lo'], 'ci_hi': boot['hi'],
                     'frac_positive': k / len(s), 'cp_lo': lo, 'cp_hi': hi,
                     'still_moving': moving})
        if moving:
            failing.append(f'{cell}/{label}')
    print(table(rows, ('cell', 'condition', 'label', 'n', 'median_slope',
                       'ci_lo', 'ci_hi', 'frac_positive', 'cp_lo', 'cp_hi',
                       'still_moving')))
    print('  window: the final-window slope as recorded by train.py '
          '(result.convergence_window_episodes);')
    print('  units are score per episode. "still_moving" means the arm-level '
          'interval excludes zero.')
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
            f'"performance at budget", not asymptotic performance')
    else:
        print('  No arm shows a final-window slope distinguishable from zero '
              'at 95%. That is')
        print('  consistent with convergence but does not establish it: at '
              'this n the interval')
        print('  is wide, so the licensed statement is the interval above, not '
              '"converged".')
    return {'available': True, 'per_arm': rows, 'failing': failing}


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


def section_confirmatory(df: pd.DataFrame, opts: Options,
                         ledger: Ledger) -> dict:
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
    print(f'    contrast  : delta = transfer - scratch, within cell, at '
          f'matched seeds')
    print(f'    primary   : exact sign-flip randomisation test, statistic = '
          f'the mean delta')
    print(f'    correction: Holm-Bonferroni over '
          f'{CONFIRMATORY_FAMILY_SIZE}; strictest step alpha = '
          f'{ALPHA_STRICTEST:.5f}')
    computed = [m for m in CONFIRMATORY_ENDPOINTS if m in opts.metrics]
    if set(computed) != set(CONFIRMATORY_ENDPOINTS):
        print(f'  NOTE: --metric restricted this run to {computed}. The family '
              f'size stays {CONFIRMATORY_FAMILY_SIZE} by pre-registration, so '
              f'the adjustment is unchanged.')
        ledger.deviations.append(
            f'--metric restricted the computed family to {computed}; Holm '
            f'still applied over {CONFIRMATORY_FAMILY_SIZE}')

    members: list[dict] = []
    pvals: dict[tuple[str, str], float] = {}
    for metric in computed:
        for cell in CELL_ORDER:
            pd_ = _paired_delta(df, cell, metric, opts)
            key = (metric, cell)
            d = pd_['delta']
            n = len(d)
            rec: dict[str, Any] = {
                'metric': metric, 'cell': cell, 'n': n,
                'seeds': pd_['seeds'],
                'n_transfer_rows': pd_['n_a'], 'n_scratch_rows': pd_['n_b'],
                'unpaired_transfer_seeds': pd_['only_a'],
                'unpaired_scratch_seeds': pd_['only_b'],
                'transfer_labels': pd_['transfer_labels'],
                'freeze_updates_observed': pd_['freeze_updates_observed'],
            }
            if len(pd_['transfer_labels']) > 1:
                rec['suppressed'] = (
                    f'ambiguous primary arm: labels '
                    f'{pd_["transfer_labels"]} all match registry.PROTOCOL')
            elif len(pd_['freeze_updates_observed']) > 1:
                rec['suppressed'] = (
                    f'freeze_updates is not constant across the arm '
                    f'({pd_["freeze_updates_observed"]}); DESIGN.md §8.4 '
                    f'refuses to aggregate runs that differ in an invariant')
            elif pd_['n_a'] == 0 or pd_['n_b'] == 0:
                empty = ('transfer' if pd_['n_a'] == 0 else 'scratch')
                rec['suppressed'] = (
                    f'the {empty} arm is empty in the analysis set in force. '
                    f'Under --source-policy valid this happens when every '
                    f'source fails the DESIGN.md §4.3 gate: the primary '
                    f'estimand is defined on valid sources only, so it does '
                    f'not exist here. Nothing is substituted for it')
            elif pd_['only_a'] or pd_['only_b']:
                rec['suppressed'] = (
                    f'incomplete arm: seeds {pd_["only_a"]} appear only in '
                    f'transfer, {pd_["only_b"]} only in scratch. A partial arm '
                    f'is refused (DESIGN.md §8.4); no seed is dropped to '
                    f'rescue the test')
            elif n < MIN_N_FOR_INFERENCE:
                rec['suppressed'] = (
                    f'n={n} < {MIN_N_FOR_INFERENCE}: no test and no interval '
                    f'(ANALYSIS_PLAN.md §9)')
            if 'suppressed' in rec:
                rec.update({'mean_delta': float(np.mean(d)) if n else None,
                            'p_signflip': None})
                members.append(rec)
                ledger.suppressed.append(
                    f'{metric}/{cell}: {rec["suppressed"]}')
                continue

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
            u_stat, u_p = sps.mannwhitneyu(pd_['a'], pd_['b'],
                                           alternative='two-sided')
            r_pear = (float(sps.pearsonr(pd_['a'], pd_['b'])[0]) if n > 2
                      else float('nan'))
            r_spear = (float(sps.spearmanr(pd_['a'], pd_['b'])[0]) if n > 2
                       else float('nan'))
            rec.update({
                'mean_delta': float(np.mean(d)),
                'median_delta': float(np.median(d)),
                'sd_delta': sd(d),
                'transfer_mean': float(np.mean(pd_['a'])),
                'scratch_mean': float(np.mean(pd_['b'])),
                'hl': hl['estimate'], 'ci_lo': hl['lo'], 'ci_hi': hl['hi'],
                'ci_method': hl['method'],
                'p_signflip': sf['p'], 'signflip_mode': sf['mode'],
                'min_attainable_p': sf['min_attainable_p'],
                'wilcoxon_W': float(w_stat), 'p_wilcoxon': w_p,
                'mannwhitney_U': float(u_stat), 'p_mannwhitney': float(u_p),
                'rho_pearson': r_pear, 'rho_spearman': r_spear,
                'unanimous': phrase_unanimity(d),
                'deltas': [float(x) for x in d],
            })
            pvals[key] = float(sf['p'])
            members.append(rec)
            ledger.confirmatory.append(f'{metric}/{cell} sign-flip')

    adj = holm_adjust(pvals, CONFIRMATORY_FAMILY_SIZE)
    for rec in members:
        key = (rec['metric'], rec['cell'])
        rec['p_holm'] = adj.get(key)
        rec['significant_holm'] = (rec['p_holm'] is not None
                                   and rec['p_holm'] < ALPHA)

    h2('5a. the eight tests')
    print(table(members, ('metric', 'cell', 'n', 'scratch_mean',
                          'transfer_mean', 'mean_delta', 'hl', 'ci_lo',
                          'ci_hi', 'p_signflip', 'p_holm', 'p_wilcoxon',
                          'mannwhitney_U', 'p_mannwhitney', 'rho_pearson',
                          'rho_spearman')))
    n_sup = sum(1 for r in members if 'suppressed' in r)
    if n_sup:
        print()
        print(f'  {n_sup} of {len(members)} members suppressed:')
        for r in members:
            if 'suppressed' in r:
                print(f'    {r["metric"]}/{r["cell"]}: {r["suppressed"]}')

    h2('5b. interpretation rule, stated before the numbers were seen')
    example_n = max((r['n'] for r in members if r.get('p_signflip')),
                    default=10)
    print(f'  At n={example_n} the exact sign-flip test cannot return a '
          f'two-sided p below')
    print(f'  2/2^{example_n} = {2 / 2 ** example_n:.5f}, attained exactly '
          f'when every seed moves the same way.')
    strict = ALPHA_STRICTEST
    reachable = (2 / 2 ** example_n) < strict
    print(f'  The strictest Holm step is alpha = {strict:.5f}.')
    if reachable:
        print('  Therefore a cell is confirmed if and only if ALL of its seeds '
              'move in the same')
        print('  direction: the bar is unanimity (ANALYSIS_PLAN.md §2.2). '
              'Anything less cannot')
        print('  clear the corrected threshold at this sample size.')
    else:
        print(f'  Therefore NO result at n={example_n} can clear the '
              f'corrected threshold: the')
        print(f'  smallest attainable p ({2 / 2 ** example_n:.5f}) exceeds '
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
        rho = rec['rho_pearson']
        if np.isfinite(rho) and rho < 0:
            print(f'    rho = {rho:+.3f} < 0: the matched-seed pairing does '
                  f'NOT hold in this cell.')
            print('    ANALYSIS_PLAN.md §2.1 pre-commits to giving the '
                  'unpaired result equal')
            print(f'    prominence here: Mann-Whitney U = '
                  f'{rec["mannwhitney_U"]:.1f}, p = '
                  f'{rec["p_mannwhitney"]:.5f}.')
        else:
            print(f'    rho = {rho:+.3f}: reported whatever its value; the '
                  f'paired test stays')
            print('    primary by pre-registration, not by comparison of '
                  'p-values.')
        agree = [('sign-flip', rec['p_signflip']),
                 ('Wilcoxon', rec['p_wilcoxon']),
                 ('Mann-Whitney', rec['p_mannwhitney'])]
        verdicts = {name: (p is not None and p < ALPHA) for name, p in agree}
        if len(set(verdicts.values())) == 1:
            print(f'    all three tests agree at the nominal alpha '
                  f'({"reject" if list(verdicts.values())[0] else "do not reject"}); '
                  f'the Holm-corrected verdict is what counts.')
        else:
            dis = ', '.join(f'{k}={"reject" if v else "no"}'
                            for k, v in verdicts.items())
            print(f'    the three tests DISAGREE at the nominal alpha ({dis}); '
                  f'that disagreement is')
            print('    the finding, and the pre-registered primary is the '
                  'sign-flip test.')
    return {'members': members, 'family_size': CONFIRMATORY_FAMILY_SIZE,
            'alpha': ALPHA, 'alpha_strictest': ALPHA_STRICTEST}


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
        if 'suppressed' in rec:
            verdict = 'suppressed'
            reason = rec['suppressed']
        elif not (np.isfinite(lo) and np.isfinite(hi)):
            verdict = 'no interval'
            reason = 'no interval emitted'
        elif np.isfinite(worst_sd) and worst_sd > margin:
            verdict = 'UNTESTABLE'
            reason = (f'across-seed SD {worst_sd:.4f} exceeds the margin '
                      f'{margin}: equivalence is untestable in this cell at '
                      f'n={rec["n"]} (ANALYSIS_PLAN.md §4)')
        elif lo > -margin and hi < margin:
            verdict = 'EQUIVALENT'
            reason = (f'the whole interval [{lo:+.4f}, {hi:+.4f}] lies inside '
                      f'+/-{margin}')
        elif lo >= margin or hi <= -margin:
            verdict = 'DIFFERENT'
            reason = (f'the whole interval [{lo:+.4f}, {hi:+.4f}] lies outside '
                      f'+/-{margin}')
        else:
            verdict = 'INCONCLUSIVE'
            reason = (f'the interval [{lo:+.4f}, {hi:+.4f}] straddles the '
                      f'margin boundary')
        rows.append({'metric': metric, 'cell': cell, 'n': rec['n'],
                     'ci_lo': lo, 'ci_hi': hi, 'sd_scratch': sd_scratch,
                     'sd_transfer': sd_transfer, 'margin': margin,
                     'verdict': verdict,
                     'exclusion_bound': (abs(lo) if np.isfinite(lo) and lo < 0
                                         else 0.0),
                     'reason': reason})
    print()
    print(table(rows, ('metric', 'cell', 'n', 'ci_lo', 'ci_hi', 'sd_scratch',
                       'sd_transfer', 'margin', 'verdict')))
    print()
    for r in rows:
        print(f'  {r["metric"]}/{r["cell"]}: {r["verdict"]} -- {r["reason"]}')
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
    protocol_fw = fw[0] if len(fw) == 1 else None
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
            'C3': c3, 'C3b': c3b, 'protocol_freeze_updates': protocol_fw}


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
        per_seed: dict[str, dict[int, float]] = {}
        for key in ('C0', 'C1', 'C2', 'C2K0', 'C3', 'C3b'):
            a = arms[key]
            vals: dict[int, float] = {}
            for _, r in a.iterrows():
                v = r.get(metric)
                if pd.notna(v):
                    vals[int(r['seed'])] = float(v)
            per_seed[key] = vals
        present = [k for k, v in per_seed.items() if v]
        missing = [k for k, v in per_seed.items() if not v]
        common = sorted(set.intersection(*[set(per_seed[k]) for k in present])
                        ) if present else []
        h2(f'7.{cell}')
        print(f'  conditions present: {", ".join(present) or "none"}'
              + (f'   absent: {", ".join(missing)}' if missing else ''))
        if missing:
            print('  A contrast whose condition is absent is not computed and '
                  'not approximated.')
        per_cond_seeds = {k: sorted(per_seed[k]) for k in present}
        dropped = {k: sorted(set(per_cond_seeds[k]) - set(common))
                   for k in present}
        dropped = {k: v for k, v in dropped.items() if v}
        if dropped:
            print(f'  listwise-complete seeds for the JOINT estimate: '
                  f'{common}')
            print(f'  seeds present in some conditions but not all, therefore '
                  f'outside the joint')
            print(f'  estimate (reported, not dropped silently): {dropped}')
            ledger.deviations.append(
                f'{cell} control set: seeds {dropped} are not complete across '
                f'all conditions, so the joint estimate uses {common}')
        if len(common) < MIN_N_FOR_INFERENCE:
            print(f'  n={len(common)} < {MIN_N_FOR_INFERENCE}: no estimate and '
                  f'no interval (ANALYSIS_PLAN.md §9).')
            out['cells'][cell] = {'n': len(common), 'suppressed': True,
                                  'present': present, 'missing': missing}
            ledger.suppressed.append(f'control contrasts {cell}: '
                                     f'n={len(common)}')
            continue

        mat = np.column_stack([[per_seed[k][s] for s in common]
                               for k in present])
        col = {k: i for i, k in enumerate(present)}
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
                rho = (float(sps.spearmanr(x, y)[0]) if n > 2
                       else float('nan'))
                brho = (float(sps.spearmanr(reps[a], reps[b])[0])
                        if len(reps[a]) and len(reps[b]) else float('nan'))
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
            print(f'  Telescoping identity residual (an identity, see the '
                  f'header): max |lhs - rhs|')
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

        out['cells'][cell] = {'n': n, 'present': present, 'missing': missing,
                              'joint_seeds': common,
                              'contrasts': rows,
                              'correlations': crows if len(keys) > 1 else [],
                              'protocol_freeze_updates':
                                  arms['protocol_freeze_updates']}

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
        print(f'  H1 (DESIGN.md §2.3): the untrained-source contrast is '
              f'negative with an')
        print(f'  interval excluding zero in {k} of {len(h1_neg)} cell(s) with '
              f'an estimate. H1 predicts')
        print('  at least 3 of 4; it is refuted if the interval covers zero in '
              '2 or more cells')
        print('  or is positive in any. No p-value attaches to this.')
    if h2_flags:
        k = sum(1 for _, flag in h2_flags if flag)
        print(f'  H2: |C1-C3| exceeds |C2-C0| in {k} of {len(h2_flags)} '
              f'cell(s). H2 is refuted at 2 or more.')
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
       f'shift)')
    ledger.est('C4 positive control')
    print(f'  Pass criterion, pre-registered: the HL estimate of the paired '
          f'delta has a 95%')
    print(f'  bootstrap CI whose lower bound exceeds {C4_LOWER_BOUND:+.2f} '
          f'normalised-score units.')
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
        s = rows_where(cdf, condition='scratch')
        t = protocol_match(rows_where(cdf, condition='transfer'))
        pair = paired_by_seed(t, s, metric)
        d = pair['a'] - pair['b']
        rec: dict[str, Any] = {'cell': cell, 'n': len(d),
                               'n_transfer': pair['n_a'],
                               'n_scratch': pair['n_b'],
                               'transfer_only_seeds': pair['only_a'],
                               'scratch_only_seeds': pair['only_b']}
        if len(d) < MIN_N_FOR_INFERENCE:
            rec.update({'verdict': 'suppressed',
                        'reason': f'n={len(d)} < {MIN_N_FOR_INFERENCE}: no '
                                  f'test, no interval (ANALYSIS_PLAN.md §9)'})
            ledger.suppressed.append(f'C4 {cell}: n={len(d)}')
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
            f'performance at zero dynamics shift')
    return {'available': True, 'env': opts.interface_env, 'rows': rows}


# ---------------------------------------------------------------------------
# §10.10 -- estimation-only. Every subsection here emits an interval and no
# p-value, with the single declared exception of the screen q-values, which
# `ANALYSIS_PLAN.md` §7 permits "for orientation only, no assertion permitted".
# ---------------------------------------------------------------------------

def _cell_deltas(df: pd.DataFrame, opts: Options, metric: str
                 ) -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = {}
    for cell in CELL_ORDER:
        cdf = df[df['cell'] == cell]
        pair = paired_by_seed(primary_transfer_arm(cdf, opts),
                              scratch_arm(cdf, opts), metric)
        out[cell] = {s: float(a - b) for s, a, b in
                     zip(pair['seeds'], pair['a'], pair['b'])}
    return out


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
    rows = []
    for a, b in combinations(CELL_ORDER, 2):
        xa = _clean(scratch_arm(df[df['cell'] == a], opts)[metric])
        xb = _clean(scratch_arm(df[df['cell'] == b], opts)[metric])
        if not len(xa) or not len(xb):
            continue
        bm = brunner_munzel(xa, xb, opts.n_boot, opts.boot_seed)
        rows.append({'a': a, 'b': b, 'n_a': len(xa), 'n_b': len(xb),
                     'mean_a': float(np.mean(xa)), 'mean_b': float(np.mean(xb)),
                     'sd_a': sd(xa), 'sd_b': sd(xb), 'theta': bm['theta'],
                     'ci_lo': bm['lo'], 'ci_hi': bm['hi'],
                     'note': bm['note']})
    print(table(rows, ('a', 'b', 'n_a', 'n_b', 'mean_a', 'mean_b', 'sd_a',
                       'sd_b', 'theta', 'ci_lo', 'ci_hi')))
    print('  theta = P(a run of cell a scores above a run of cell b), '
          'ties at 0.5. theta = 0.5')
    print('  is no difference. Associational: no causal reading is licensed '
          '(DESIGN.md §2.4).')
    for r in rows:
        print(f'  {r["a"]} vs {r["b"]}: ' + phrase_dispersion(
            r['a'], r['sd_a'], r['b'], r['sd_b']))
    return {'rows': rows}


def sub_rq3(df: pd.DataFrame, opts: Options, ledger: Ledger, metric: str,
            gate: list[dict]) -> dict:
    """RQ3 -- between-cell contrast of deltas and the 2x2 interaction.

    Effect modification, not "architecture causes the difference"
    (`DESIGN.md` §2.4). Explicitly underpowered: the plan puts the interaction's
    MDE at about 2.7 sigma, so this is an interval and nothing else. Reported
    on both the normalised and the headroom-adjusted scale, because a cell near
    the ceiling has less room to gain -- agreement across the two scales is
    required before any wording is used (`ANALYSIS_PLAN.md` §3).
    """
    h2('9b. RQ3 -- between-cell contrast of deltas and the 2x2 interaction')
    ledger.est('RQ3 between-cell delta contrasts and interaction')
    deltas = _cell_deltas(df, opts, metric)
    headroom = {}
    for cell in CELL_ORDER:
        s = _clean(scratch_arm(df[df['cell'] == cell], opts)[metric])
        headroom[cell] = (1.0 - float(np.mean(s))) if len(s) else float('nan')
    blocked = {(g['a'], g['b']) for g in gate if not g['permitted']}
    blocked |= {(b, a) for a, b in blocked}

    rows = []
    for a, b in combinations(CELL_ORDER, 2):
        da, db = deltas.get(a, {}), deltas.get(b, {})
        common = sorted(set(da) & set(db))
        rec: dict[str, Any] = {'a': a, 'b': b, 'n': len(common)}
        if (a, b) in blocked:
            rec['note'] = ('REFUSED: intensity-confounded cross-architecture '
                           'contrast (DESIGN.md §3.1)')
            rows.append(rec)
            continue
        if len(common) < MIN_N_FOR_INFERENCE:
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
        agree = ''
        if np.isfinite(raw['lo']) and np.isfinite(adj['lo']):
            raw_excl = raw['lo'] > 0 or raw['hi'] < 0
            adj_excl = adj['lo'] > 0 or adj['hi'] < 0
            agree = 'agree' if raw_excl == adj_excl else 'DISAGREE'
        rec.update({'hl': raw['estimate'], 'ci_lo': raw['lo'],
                    'ci_hi': raw['hi'], 'hl_headroom_adj': adj['estimate'],
                    'adj_lo': adj['lo'], 'adj_hi': adj['hi'],
                    'headroom_a': ha, 'headroom_b': hb, 'scales': agree,
                    'note': ''})
        rows.append(rec)
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
                  f'headroom-adjusted scales DISAGREE on')
            print('    whether the interval excludes zero. No wording is '
                  'licensed for this pair.')

    inter: dict[str, Any] = {'available': False}
    want = ['mlp-vanilla', 'mlp-double', 'dueling-vanilla', 'dueling-double']
    if all(w in deltas and deltas[w] for w in want):
        common = sorted(set.intersection(*[set(deltas[w]) for w in want]))
        arch_pairs = {('mlp-vanilla', 'dueling-vanilla'),
                      ('mlp-double', 'dueling-double')}
        confounded = any(p in blocked for p in arch_pairs)
        if confounded and not opts.allow_intensity_confound:
            print()
            print('  2x2 interaction: REFUSED. It mixes both architectures, '
                  'whose transferred')
            print('  fractions differ by more than the tolerance, so the '
                  'interaction would be')
            print('  confounded with treatment intensity (DESIGN.md §3.1).')
            ledger.refusals.append('2x2 interaction refused: '
                                   'intensity-confounded across arch')
        elif len(common) < MIN_N_FOR_INFERENCE:
            print(f'  2x2 interaction: n={len(common)}, suppressed.')
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
            print()
            print(f'  2x2 interaction (target_rule effect on delta, dueling '
                  f'minus mlp), n={len(common)}:')
            print(f'    HL {res["estimate"]:+.4f}   95% CI '
                  f'[{res["lo"]:+.4f}, {res["hi"]:+.4f}]')
            print('    ' + phrase_interval_verdict(res['lo'], res['hi'],
                                                   'the interaction'))
            print('    MDE for this contrast is ~2.7 sigma '
                  '(ANALYSIS_PLAN.md §6), larger than any')
            print('    plausible effect, so this is an interval by design and '
                  'carries no p-value.')
            if opts.allow_intensity_confound and confounded:
                print('    LABELLED INTENSITY-CONFOUNDED (override in force).')
            inter = {'available': True, 'n': len(common),
                     'hl': res['estimate'], 'ci_lo': res['lo'],
                     'ci_hi': res['hi'],
                     'intensity_confounded': bool(confounded)}
    else:
        print('  2x2 interaction: not all four cells have paired deltas; not '
              'computed.')
    return {'pairs': rows, 'interaction': inter, 'headroom': headroom}


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
            out[family] = {'available': False}
            continue
        rows = []
        groups = []
        order_vals = []
        for lab, canonical in present:
            sub = df[df['env'] == canonical]
            per_cell = []
            for cell in CELL_ORDER:
                cdf = sub[sub['cell'] == cell]
                t = protocol_match(rows_where(cdf, condition='transfer'))
                s = rows_where(cdf, condition='scratch')
                pair = paired_by_seed(t, s, metric)
                per_cell.extend((pair['a'] - pair['b']).tolist())
            groups.append(np.asarray(per_cell, dtype=float))
            order_vals.append(envs.family_level_value(family, canonical))
            rows.append({'level': lab, 'shift_value':
                         envs.family_level_value(family, canonical),
                         'n_deltas': len(per_cell),
                         'mean_delta': (float(np.mean(per_cell))
                                        if per_cell else None),
                         'sd_delta': sd(per_cell)})
        print(table(rows, ('level', 'shift_value', 'n_deltas', 'mean_delta',
                           'sd_delta')))
        if sum(len(g) for g in groups) < MIN_N_FOR_INFERENCE * 2:
            print('    too few deltas for a trend estimate; suppressed.')
            out[family] = {'available': False, 'levels': rows}
            continue
        order = np.argsort(np.asarray(order_vals))
        jt = jonckheere_effect([groups[i] for i in order])
        print(f'    Jonckheere concordance across increasing shift: '
              f'{jt["standardised"]:+.3f}')
        print('    ' + phrase_trend(jt['standardised'], family))
        print('    H4 predicts monotone DEGRADATION, i.e. a negative '
              'concordance. Reported as')
        print('    a standardised effect with no p-value '
              '(ANALYSIS_PLAN.md §3).')
        out[family] = {'available': True, 'levels': rows,
                       'concordance': jt['standardised'], 'role': role}
    return out


def sub_rq6(df: pd.DataFrame, opts: Options, ledger: Ledger,
            metric: str) -> dict:
    """RQ6 -- budget, via the prefix evaluations.

    Valid only because the exploration schedule is a closed-form function of
    elapsed env steps and never reads the budget, so a 500-episode prefix *is*
    what a 500-episode run would have produced (`DESIGN.md` §2.4 RQ6);
    `validate.py` asserts that identifying condition.
    """
    h2('9d. RQ6 -- does the conclusion depend on the budget?')
    ledger.est('RQ6 budget prefixes')
    prefixes = sorted(int(m.group(1)) for m in
                      (_PREFIX_SCORE_RE.match(c) for c in df.columns) if m)
    if not prefixes:
        print('  no prefix_score_* columns: not computed.')
        return {'available': False}
    rows = []
    for cell in CELL_ORDER:
        cdf = df[df['cell'] == cell]
        end = paired_by_seed(primary_transfer_arm(cdf, opts),
                             scratch_arm(cdf, opts), metric)
        end_d = end['a'] - end['b']
        for p in prefixes:
            col = f'prefix_score_{p}'
            pre = paired_by_seed(primary_transfer_arm(cdf, opts),
                                 scratch_arm(cdf, opts), col)
            d = pre['a'] - pre['b']
            rec = {'cell': cell, 'prefix': p, 'n': len(d),
                   'delta_at_prefix': (float(np.mean(d)) if len(d) else None),
                   'delta_at_budget': (float(np.mean(end_d)) if len(end_d)
                                       else None)}
            if len(d) >= MIN_N_FOR_INFERENCE:
                res = bootstrap_statistic(d, hodges_lehmann_paired,
                                          opts.n_boot, opts.boot_seed,
                                          vec=hl_vec)
                rec.update({'hl': res['estimate'], 'ci_lo': res['lo'],
                            'ci_hi': res['hi']})
            rows.append(rec)
    print(table(rows, ('cell', 'prefix', 'n', 'delta_at_prefix',
                       'delta_at_budget', 'hl', 'ci_lo', 'ci_hi')))
    if all(r['n'] == 0 for r in rows):
        print(f'  The prefix columns {prefixes} exist but hold no values in '
              f'this dataset, so RQ6')
        print('  is not estimable here. Nothing is substituted for a prefix '
              'evaluation that was')
        print('  never run.')
        ledger.deviations.append(
            'RQ6 not estimable: prefix_score_* columns are present but empty, '
            'so no episode-prefix re-evaluation exists in this dataset')
    print('  A sign change between a prefix and the budget would mean the '
          'conclusion is')
    print('  budget-dependent, which is itself a finding and is reported as '
          'one.')
    for r in rows:
        a, b = r.get('delta_at_prefix'), r.get('delta_at_budget')
        if a is not None and b is not None and np.isfinite(a) \
                and np.isfinite(b) and a * b < 0:
            print(f'  {r["cell"]}: the delta CHANGES SIGN between prefix '
                  f'{r["prefix"]} ({a:+.4f}) and the')
            print(f'    budget ({b:+.4f}). The conclusion is budget-dependent '
                  f'in this cell.')
    return {'available': True, 'rows': rows, 'prefixes': prefixes}


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
                               'sd_scratch': sd(pair['b']),
                               'sd_transfer': sd(pair['a'])}
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
            bf = sps.levene(pair['a'], pair['b'], center='median')
            rec.update({'sd_ratio': res['estimate'], 'ci_lo': res['lo'],
                        'ci_hi': res['hi'],
                        'brown_forsythe_W': float(bf.statistic)})
        rows.append(rec)
    print(table(rows, ('cell', 'n', 'sd_scratch', 'sd_transfer', 'sd_ratio',
                       'ci_lo', 'ci_hi', 'brown_forsythe_W')))
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
    if 'within_run_sd' in df.columns:
        print()
        print('  within_run_sd (training instability) and the across-seed SD '
              'above (seed')
        print('  sensitivity) are DIFFERENT metrics; the published study '
              'conflated them:')
        wrows = []
        for (cell, cond), g in df.groupby(['cell', 'condition']):
            v = _clean(g['within_run_sd'])
            if len(v):
                wrows.append({'cell': cell, 'condition': cond, 'n': len(v),
                              'within_run_sd_mean': float(np.mean(v))})
        print(table(wrows, ('cell', 'condition', 'n', 'within_run_sd_mean')))
    return {'rows': rows}


def sub_censored(df: pd.DataFrame, opts: Options, ledger: Ledger) -> dict:
    """Censored metrics: Kaplan-Meier and an exact interval on P(reached).

    `ANALYSIS_PLAN.md` §5. The censoring is administrative -- the same budget
    for every run, independent of the event time by construction -- which is the
    benign case. The budget is never imputed as an observation and no censored
    run is dropped.
    """
    h2('9f. steps_to_threshold -- right-censored at the budget')
    ledger.est('censored steps-to-threshold (Kaplan-Meier, Clopper-Pearson)')
    out: dict[str, Any] = {}
    for tag, level in THRESHOLD_LEVELS:
        tcol, ccol = f'steps_to_threshold_{tag}', f'censored_{tag}'
        if tcol not in df.columns or ccol not in df.columns:
            print(f'  {tcol}: column absent, skipped.')
            continue
        rows = []
        arms: dict[tuple[str, str], dict] = {}
        for (cell, cond, label), g in df.groupby(['cell', 'condition',
                                                  'label'], dropna=False):
            t = _clean(g[tcol])
            cens = g[ccol].map(lambda v: bool(v) if v is not None else True)
            ev = ~np.asarray(cens.tolist(), dtype=bool)
            if len(t) != len(ev) or not len(t):
                continue
            k = int(np.sum(ev))
            lo, hi = clopper_pearson(k, len(t))
            km = kaplan_meier(t, ev)
            arms[(cell, label)] = {'t': t, 'e': ev}
            rows.append({'cell': cell, 'condition': cond, 'label': label,
                         'n': len(t), 'reached': k,
                         'p_reached': k / len(t), 'cp_lo': lo, 'cp_hi': hi,
                         'km_median_steps': km['median']})
        print(f'  threshold = normalised score {level}')
        print(table(rows, ('cell', 'condition', 'label', 'n', 'reached',
                           'p_reached', 'cp_lo', 'cp_hi',
                           'km_median_steps')))
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
            rec = {'metric': col, 'role': role, 'cell': cell, 'n': len(d),
                   'transfer_mean': (float(np.mean(pair['a'])) if len(d)
                                     else None),
                   'scratch_mean': (float(np.mean(pair['b'])) if len(d)
                                    else None)}
            if len(d) >= MIN_N_FOR_INFERENCE:
                res = bootstrap_statistic(d, hodges_lehmann_paired,
                                          opts.n_boot, opts.boot_seed,
                                          vec=hl_vec)
                rec.update({'hl_delta': res['estimate'], 'ci_lo': res['lo'],
                            'ci_hi': res['hi']})
            rows.append(rec)
    print(table(rows, ('metric', 'role', 'cell', 'n', 'scratch_mean',
                       'transfer_mean', 'hl_delta', 'ci_lo', 'ci_hi')))
    print('  No p-value appears in this table. A mechanism claim in the paper '
          'must cite one')
    print('  of these instrumented signals; there is no free-text mechanism '
          'slot (DESIGN.md §9).')
    out['rows'] = rows

    h2('9h. descriptive-only metrics -- reported, never tested')
    drows = []
    for col in [c for c, r in METRIC_ROLES.items()
                if r == DESCRIPTIVE and c in df.columns]:
        for (cell, cond), g in df.groupby(['cell', 'condition']):
            v = _clean(g[col])
            if len(v):
                drows.append({'metric': col, 'cell': cell, 'condition': cond,
                              'n': len(v), 'mean': float(np.mean(v)),
                              'sd': sd(v)})
    print(table(drows, ('metric', 'cell', 'condition', 'n', 'mean', 'sd')))
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
    follow-up rule: a screen selects at most one follow-up, which is then run on
    `REPLICATE` seeds and reported as a fresh estimate.
    """
    h2('9j. ablation screens -- BH q for orientation only, never an assertion')
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
    rows = []
    pvals = []
    for exp, sub in present:
        for cell in CELL_ORDER:
            cdf = sub[sub['cell'] == cell]
            base = scratch_arm(df[df['cell'] == cell], opts)
            for label, g in cdf.groupby('label'):
                if (g['condition'] == 'scratch').all():
                    continue
                pair = paired_by_seed(g, base, metric)
                d = pair['a'] - pair['b']
                rec = {'experiment': exp.id, 'cell': cell, 'level': label,
                       'n': len(d)}
                if len(d) >= MIN_N_FOR_INFERENCE:
                    res = bootstrap_statistic(d, hodges_lehmann_paired,
                                              opts.n_boot, opts.boot_seed,
                                              vec=hl_vec)
                    sf = sign_flip_test(d, seed=opts.boot_seed)
                    rec.update({'hl': res['estimate'], 'ci_lo': res['lo'],
                                'ci_hi': res['hi'], 'p_raw': sf['p']})
                    pvals.append((len(rows), sf['p']))
                rows.append(rec)
    if pvals:
        order = sorted(pvals, key=lambda kp: kp[1])
        m = len(order)
        prev = 1.0
        for rank in range(m, 0, -1):
            i, p = order[rank - 1]
            q = min(prev, p * m / rank)
            prev = q
            rows[i]['q_bh'] = q
    print(table(rows, ('experiment', 'cell', 'level', 'n', 'hl', 'ci_lo',
                       'ci_hi', 'p_raw', 'q_bh')))
    print('  These q-values ORIENT; they assert nothing. A screen result is '
          'never a finding.')
    ledger.screen_q.extend(f'{r["experiment"]}/{r["cell"]}/{r["level"]}'
                           for r in rows if 'q_bh' in r)
    return {'available': True, 'rows': rows}


def section_estimation(df: pd.DataFrame, opts: Options, ledger: Ledger,
                       metric: str, gate: list[dict]) -> dict:
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
    return {'rq1': sub_rq1(df, opts, ledger, metric),
            'rq3': sub_rq3(df, opts, ledger, metric, gate),
            'rq5': sub_rq5(df, opts, ledger, metric),
            'rq6': sub_rq6(df, opts, ledger, metric),
            'dispersion': sub_dispersion(df, opts, ledger, metric),
            'censored': sub_censored(tdf, opts, ledger),
            'secondary': sub_secondary(tdf, opts, ledger),
            'screens': sub_screens(tdf, opts, ledger, metric)}


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
    print('  Multipliers, pre-registered in ANALYSIS_PLAN.md §6.2 and not '
          're-tuned (§6.4):')
    print(table([{'test': k[0], 'alpha': ('0.05' if k[1] == 'nominal'
                                          else f'{ALPHA_STRICTEST:.5f} '
                                               f'(Holm over '
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
        cdf = df[df['cell'] == cell]
        s = _clean(scratch_arm(cdf, opts)[metric])
        t = _clean(primary_transfer_arm(cdf, opts)[metric])
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
                     'sigma_delta': sigma_d, 'sigma_pooled': pooled,
                     'observed_delta': rec['mean_delta'], **mde,
                     'powered': powered,
                     'note': ('' if powered else
                              'MDE at the corrected alpha reaches or exceeds '
                              '1.0 score unit, the whole distance from random '
                              'play to solved: NOT POWERED')})
    print()
    print(table(rows, ('metric', 'cell', 'n', 'sigma_delta', 'sigma_pooled',
                       'observed_delta', 'paired_nominal', 'paired_holm8',
                       'unpaired_nominal', 'unpaired_holm8', 'powered')))
    print('  MDE units are normalised score. A cell is flagged NOT POWERED '
          'when its MDE at')
    print(f'  the Holm-corrected alpha reaches {UNPOWERED_MDE} score units -- '
          f'which by construction is')
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
                            for k, v in MDE_MULTIPLIERS.items()}}


def section_ledger(ledger: Ledger, conf: dict) -> dict:
    """§10.11 -- the multiplicity ledger. Printed on every invocation."""
    h1('11. MULTIPLICITY LEDGER')
    rows = [
        {'family': 'Confirmatory',
         'members': f'{CONFIRMATORY_FAMILY_SIZE} (4 cells x 2 co-primary '
                    f'endpoints)',
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
    print(f'  analyses carrying NO p-value         : '
          f'{len(ledger.estimation)}')
    for name in ledger.estimation:
        print(f'    {name}')
    print(f'  refusals                             : {len(ledger.refusals)}')
    for name in ledger.refusals:
        print(f'    {name}')
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
            'estimation_only': ledger.estimation,
            'refusals': ledger.refusals,
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
                   choices=list(CONFIRMATORY_ENDPOINTS),
                   help='restrict the computed co-primary endpoints; the '
                        'family size stays 8 by pre-registration')
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
    p.add_argument('--n-boot', type=int, default=N_BOOT)
    p.add_argument('--boot-seed', type=int, default=BOOT_SEED)
    p.add_argument('--self-test', action='store_true',
                   help='run the primitive and prose assertions and exit')
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return self_test()

    opts = Options(
        per_seed=args.per_seed,
        metrics=tuple(args.metric) if args.metric else CONFIRMATORY_ENDPOINTS,
        experiments=tuple(args.experiments) if args.experiments else None,
        target_env=args.target_env, source_env=args.source_env,
        interface_env=envs.parse(args.interface_env).canonical(),
        allow_intensity_confound=args.allow_intensity_confound,
        source_policy=args.source_policy, n_boot=args.n_boot,
        boot_seed=args.boot_seed, json_out=args.json_out)
    ledger = Ledger()

    try:
        raw = load_per_seed(opts.per_seed, ledger)
    except (FileNotFoundError, ValueError) as exc:
        print(f'stats.py: {exc}')
        return 1

    df = in_experiments(raw, opts.experiments)
    if not len(df):
        print(f'stats.py: no runs match --experiments '
              f'{opts.experiments}; refusing to report on an empty selection.')
        return 1

    # TUNE seeds may never enter a reported estimate (ANALYSIS_PLAN.md §8).
    n_before = len(df)
    df = df[df['seed_block'] != 'TUNE']
    n_tune = n_before - len(df)

    report: dict[str, Any] = {'invocation': {
        'argv': list(sys.argv), 'cwd': os.getcwd(),
        'per_seed': opts.per_seed, 'experiments': opts.experiments,
        'metrics': list(opts.metrics), 'target_env': opts.target_env,
        'source_env': opts.source_env, 'interface_env': opts.interface_env,
        'source_policy': opts.source_policy,
        'allow_intensity_confound': opts.allow_intensity_confound,
        'tune_runs_excluded': n_tune}}

    print(f'stats.py -- executing ANALYSIS_PLAN.md §10, sections 1-12')
    print(f'  invocation: {" ".join(sys.argv)}')

    report['s1_provenance'] = section_provenance(df, opts, ledger)
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

    primary_metric = opts.metrics[0]
    report['s3_descriptives'] = {
        m: section_descriptives(analysis, opts, ledger, m)
        for m in opts.metrics}
    report['s4_convergence'] = section_convergence(analysis, opts, ledger)
    conf = section_confirmatory(analysis, opts, ledger)
    report['s5_confirmatory'] = conf
    report['s6_equivalence'] = section_equivalence(conf, analysis, opts, ledger)
    report['s7_controls'] = {
        m: section_controls(analysis, opts, ledger, m) for m in opts.metrics}
    report['s8_c4'] = section_c4(analysis, opts, ledger, primary_metric)
    report['s9_estimation'] = section_estimation(
        analysis, opts, ledger, primary_metric,
        report['s2_inventory'].get('intensity_gate', []))
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
