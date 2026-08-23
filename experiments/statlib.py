"""Statistical primitives for the pre-registered analysis. No I/O, no data.

Every inferential number in the paper comes from a function in this file, and
nothing in this file knows what a run directory is. That separation is
deliberate: `stats.py`, `tables.py`, `plots.py` and `report.py` all import these
primitives, so there is exactly one implementation of each estimator and it can
be self-tested against known values without a dataset present.

Why each primitive is the one that is here
------------------------------------------
The published study's statistics failed in five specific ways
(`paper/METHODS_ACTUAL.md`, REVIEW_COVERAGE C10/C12; `ANALYSIS_PLAN.md` §8).
Each failure maps onto a design decision below.

* **A t-test and Cohen's d were run on a metric the same paper declared
  descriptive-only and non-normal.** There is no t-test, no Cohen's d and no
  normality-assuming interval anywhere in this module, and none may be added:
  `ANALYSIS_PLAN.md` §8 forbids them by name. Location estimates are
  Hodges-Lehmann, ordinal effects are Brunner-Munzel and rank-biserial,
  intervals are bootstrap, and tests are exact permutation or exact rank tests.

* **"Positive transfer" was claimed from p=0.421.** `equivalence_from_ci`
  exists so that the only licensed positive statement from a null is what the
  interval *excludes*, and it emits that sentence itself rather than leaving it
  to prose. It also refuses an equivalence verdict in a cell whose dispersion
  makes equivalence untestable at this n (`ANALYSIS_PLAN.md` §4).

* **The matched-seed structure the design creates was thrown away.** The
  primary test is `sign_flip_test`, an exact randomisation test on the per-seed
  paired deltas, because at n=10 pairing is worth roughly a 40 % reduction in
  the detectable effect (`ANALYSIS_PLAN.md` §6.2). `mann_whitney` is still
  reported for every contrast, because it is the test the reviewers endorsed
  and the published paper used, and comparability matters more than tidiness.

* **A floored p-value could not be distinguished from strong evidence.** The
  rank-test wrappers report the smallest two-sided p attainable at that sample
  size alongside the observed p, and flag when the observed p sits on that
  floor. At n=10 vs 10 the Mann-Whitney floor is 1.08e-5; the paired floor is
  2/1024 = 0.00195, which is why a confirmatory cell requires all ten seeds to
  move the same way (`ANALYSIS_PLAN.md` §2.2).

* **Control contrasts estimated from shared seeds were treated as independent.**
  `paired_bootstrap` resamples *seeds* once and evaluates every contrast on the
  same resample, so the contrasts' correlations are estimated rather than
  assumed away (`DESIGN.md` §4.1, `ANALYSIS_PLAN.md` §3).

Two conventions that recur, and the reasons for them
----------------------------------------------------
1. **A fallback is reported, never silent.** `bootstrap_ci` returns a `CI`,
   which is a plain `(lo, hi)` tuple that also carries `.method` and
   `.fallback`, so a percentile interval can never be presented as BCa. The
   MDE functions return an `MDE`, which is a plain `float` carrying `.method`
   and `.power_achieved` for the same reason.
2. **Missing values are refused, not dropped.** `ANALYSIS_PLAN.md` §8 forbids
   dropping a seed for any reason once it has run, and the published study
   dropped one silently. A non-finite entry in a vector handed to any function
   here raises, so the caller has to decide -- in code, visibly -- what a
   missing run means.

Sample-size floor: below `MIN_N_FOR_INFERENCE` (=3) the tests and the intervals
refuse and say so, per `ANALYSIS_PLAN.md` §9. A refusal is a dict or a `CI`
carrying `reason`, not an exception, because the caller's job is to print
`PIPELINE_VALIDATION_LABEL` over that section and carry on.

The `__main__` guard runs `self_test()` and nothing else. There is no CLI for
analysis here on purpose: this module must not be able to read a run.

    python experiments/statlib.py
"""
from __future__ import annotations

import argparse
import math
import sys
from functools import lru_cache
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Constants fixed by the pre-registration. They live here, as constants, so
# that no caller can pass a different family size or a different bootstrap seed
# and have the difference go unrecorded (`ANALYSIS_PLAN.md` §7: "stats.py reads
# the family definitions from here rather than accepting them as arguments").
# ---------------------------------------------------------------------------
ALPHA = 0.05                       # family-wise, two-sided
CONFIRMATORY_FAMILY_SIZE = 8       # 4 cells x 2 co-primary endpoints
HOLM_STRICTEST_ALPHA = ALPHA / CONFIRMATORY_FAMILY_SIZE      # 0.00625
EQUIVALENCE_MARGIN = 0.05          # normalised-score units, ANALYSIS_PLAN §4
BOOTSTRAP_SEED = 20260824          # ANALYSIS_PLAN §2
N_BOOT = 10_000                    # ANALYSIS_PLAN §2
THRESHOLD_LEVELS = (0.25, 0.50, 1.00)                        # ANALYSIS_PLAN §5
MIN_N_FOR_INFERENCE = 3            # ANALYSIS_PLAN §9
PIPELINE_VALIDATION_LABEL = 'PIPELINE VALIDATION - NOT A RESULT'

# Exact enumeration of sign assignments is 2**n values; 2**20 is 8 MB, which is
# where the exact/Monte-Carlo boundary sits for the *test*.
EXACT_SIGNFLIP_MAX_N = 20
# The exact Mann-Whitney null is a polynomial of degree n1*n2 in big integers.
EXACT_MWU_MAX_PRODUCT = 5_000
# Simulation size for the power calculations, matching ANALYSIS_PLAN §6.
MDE_SIM_REPLICATES = 20_000
# Exact enumeration inside the *power simulation* is capped lower than for the
# test itself, because there it runs once per simulated dataset.
EXACT_SIGNFLIP_MDE_MAX_N = 15

_EPS = float(np.finfo(float).eps)


# ---------------------------------------------------------------------------
# Result carriers
# ---------------------------------------------------------------------------
class CI(tuple):
    """A `(lo, hi)` tuple that also records how it was produced.

    Unpacks, indexes and compares exactly like `(lo, hi)`, so callers that want
    two numbers get two numbers. The attributes exist so that a percentile
    interval standing in for a failed BCa fit is visible in the output instead
    of being indistinguishable from the real thing.

    Does **not** carry a p-value and never will: most intervals in this study
    are attached to estimation-only analyses, which carry no p-value at all.
    """

    method: str
    fallback: str | None
    alpha: float
    n: int
    reason: str | None

    def __new__(cls, lo: float, hi: float, method: str = 'percentile',
                fallback: str | None = None, alpha: float = ALPHA,
                n: int = 0, reason: str | None = None) -> 'CI':
        obj = super().__new__(cls, (float(lo), float(hi)))
        obj.method = method
        obj.fallback = fallback
        obj.alpha = float(alpha)
        obj.n = int(n)
        obj.reason = reason
        return obj

    @property
    def lo(self) -> float:
        return self[0]

    @property
    def hi(self) -> float:
        return self[1]

    @property
    def refused(self) -> bool:
        return self.method == 'refused'

    @property
    def width(self) -> float:
        return self[1] - self[0]

    def __repr__(self) -> str:
        tag = self.method + (f' ({self.fallback})' if self.fallback else '')
        return f'CI({self[0]:.6g}, {self[1]:.6g}, {tag})'


class MDE(float):
    """A minimum detectable effect in sigma units, carrying its provenance.

    A plain `float` for every arithmetic purpose. `.method` says whether the
    null distribution was enumerated exactly or approximated, and
    `.power_achieved` is the simulated power at the returned effect, so a
    bisection that failed to reach the target is visible rather than implied.
    """

    method: str
    n_sim: int
    power_target: float
    power_achieved: float
    alpha: float
    n: int

    def __new__(cls, value: float, method: str = 'exact', n_sim: int = 0,
                power_target: float = 0.8,
                power_achieved: float = float('nan'),
                alpha: float = ALPHA, n: int = 0) -> 'MDE':
        obj = super().__new__(cls, float(value))
        obj.method = method
        obj.n_sim = int(n_sim)
        obj.power_target = float(power_target)
        obj.power_achieved = float(power_achieved)
        obj.alpha = float(alpha)
        obj.n = int(n)
        return obj

    def __repr__(self) -> str:
        return (f'MDE({float(self):.3f} sigma, n={self.n}, '
                f'alpha={self.alpha:g}, power={self.power_achieved:.3f}, '
                f'{self.method})')


# ---------------------------------------------------------------------------
# Input handling. Refusing a non-finite entry is the whole point of `_vector`.
# ---------------------------------------------------------------------------
def _vector(x: Sequence[float] | np.ndarray,
            name: str = 'values') -> np.ndarray:
    """A 1-D float array, or raise. Non-finite entries are refused.

    Refusing rather than filtering is required by `ANALYSIS_PLAN.md` §8: a
    missing per-seed value means a run is absent, and dropping it silently is
    the defect the Phase 0 audit found (one seed removed from one arm with no
    stated rule). The caller must decide what absence means, visibly.
    """
    arr = np.asarray(x, dtype=float).ravel()
    bad = int(np.sum(~np.isfinite(arr)))
    if bad:
        raise ValueError(
            f'{name}: {bad} of {arr.size} entries are not finite. This module '
            f'refuses to drop them -- a missing per-seed value means a missing '
            f'run, and silently dropping seeds is forbidden '
            f'(ANALYSIS_PLAN.md §8). Handle the absence explicitly in the '
            f'caller.')
    return arr


def _refused_ci(reason: str, alpha: float = ALPHA, n: int = 0) -> CI:
    return CI(np.nan, np.nan, method='refused', alpha=alpha, n=n, reason=reason)


def _too_few(n: int) -> str:
    return (f'n={n} < {MIN_N_FOR_INFERENCE}: no test and no interval is '
            f'emitted (ANALYSIS_PLAN.md §9). Label the output '
            f'{PIPELINE_VALIDATION_LABEL!r}.')


# ---------------------------------------------------------------------------
# 1. The primary test: exact sign-flip randomisation on paired deltas
# ---------------------------------------------------------------------------
def signflip_min_attainable_p(n: int) -> float:
    """Smallest two-sided p the exact sign-flip test can return at this n.

    `2 / 2**n`, attained exactly when every delta shares a sign. At n=10 this is
    0.001953, and under Holm over 8 the strictest threshold is 0.00625 -- which
    is why a confirmatory cell requires all ten seeds to move the same way
    (`ANALYSIS_PLAN.md` §2.2). Assumes the enumeration is exact; on the Monte
    Carlo branch the floor is `1/(n_perm+1)` instead.
    """
    if n <= 0:
        return float('nan')
    return float(2.0 ** (1 - n))


def _signflip_null_exact(deltas: np.ndarray) -> np.ndarray:
    """All 2**n values of `sum(s_i * d_i)`, built by successive doubling."""
    totals = np.zeros(1, dtype=float)
    for d in deltas:
        totals = np.concatenate((totals + d, totals - d))
    return totals


def sign_flip_test(deltas: Sequence[float] | np.ndarray,
                   n_perm: int | None = None,
                   seed: int = BOOTSTRAP_SEED) -> dict:
    """Exact sign-flip randomisation test on paired deltas. The primary test.

    Statistic is the **mean** of the deltas (`ANALYSIS_PLAN.md` §2). Under the
    null of no protocol effect the sign of each seed's delta is exchangeable, so
    the null distribution is the statistic over all 2**n sign assignments. All
    of them are enumerated when `n <= EXACT_SIGNFLIP_MAX_N` (=20); above that a
    Monte Carlo sample of `n_perm` assignments is drawn, `exact=False` is
    returned with the count, and the add-one estimator
    `(1 + hits) / (n_perm + 1)` is used so the p-value can never be reported as
    zero.

    Assumes the deltas are paired at the unit of resampling -- here, the seed --
    and that under the null the sign of a delta is exchangeable. The design
    earns that by construction, not by assumption: at a given seed the scratch
    and transfer runs share their per-layer initialisation for every
    non-transferred layer, the environment-reset sequence and the evaluation
    seed streams (`DESIGN.md` §8.1).

    Does **not** assume normality, symmetry of the delta distribution beyond
    sign exchangeability, equal variances, or independence *within* a seed. Does
    not adjust for multiplicity -- pass the p-values to `holm`. Does not verify
    that pairing is warranted: report `within_seed_correlation` alongside, as
    `ANALYSIS_PLAN.md` §2.1 requires, whatever it comes out as.

    Zero deltas are kept, not discarded: a zero contributes the same value under
    either sign, which correctly costs the test resolution instead of quietly
    reducing n.
    """
    d = _vector(deltas, 'deltas')
    n = int(d.size)
    if n < MIN_N_FOR_INFERENCE:
        return {'statistic': float(d.mean()) if n else float('nan'),
                'p_two_sided': float('nan'), 'n': n, 'exact': False,
                'n_perm': 0, 'p_min_attainable': float('nan'),
                'at_p_floor': False, 'refused': True, 'reason': _too_few(n)}

    obs_sum = float(d.sum())
    tol = 16.0 * _EPS * max(float(np.abs(d).sum()), 1.0)
    exact = n <= EXACT_SIGNFLIP_MAX_N

    if exact:
        null = _signflip_null_exact(d)
        hits = int(np.count_nonzero(np.abs(null) >= abs(obs_sum) - tol))
        n_used = int(null.size)
        p = hits / n_used
        p_floor = signflip_min_attainable_p(n)
    else:
        n_used = int(n_perm) if n_perm else 100_000
        rng = np.random.default_rng(seed)
        hits = 0
        block = max(1, min(n_used, 2_000_000 // max(n, 1)))
        drawn = 0
        while drawn < n_used:
            k = min(block, n_used - drawn)
            signs = rng.integers(0, 2, size=(k, n)).astype(float) * 2.0 - 1.0
            hits += int(np.count_nonzero(
                np.abs(signs @ d) >= abs(obs_sum) - tol))
            drawn += k
        p = (1.0 + hits) / (n_used + 1.0)
        p_floor = 1.0 / (n_used + 1.0)

    pos = int(np.count_nonzero(d > 0))
    neg = int(np.count_nonzero(d < 0))
    return {'statistic': float(d.mean()),
            'p_two_sided': float(min(1.0, p)),
            'n': n,
            'exact': bool(exact),
            'n_perm': n_used,
            'p_min_attainable': float(p_floor),
            'at_p_floor': bool(p <= p_floor * (1.0 + 1e-9)),
            'n_positive': pos, 'n_negative': neg,
            'n_zero': n - pos - neg,
            'all_same_sign': bool(pos == n or neg == n),
            'refused': False, 'reason': None}


# ---------------------------------------------------------------------------
# 2. The two rank tests reported alongside, with their p-value floors
# ---------------------------------------------------------------------------
def wilcoxon_signed_rank(deltas: Sequence[float] | np.ndarray) -> dict:
    """Wilcoxon signed-rank on paired deltas, with the attainable p floor.

    A thin wrapper over `scipy.stats.wilcoxon`, pre-specified in
    `ANALYSIS_PLAN.md` §2 as a companion to the sign-flip test rather than an
    alternative to it -- the choice between them is fixed before the data, so
    neither can be selected for giving the smaller p.

    Zero differences are handled by `zero_method='zsplit'`, which splits their
    ranks between the two signs. SciPy's default (`'wilcox'`) *discards* them,
    which silently reduces n, and reducing n after the fact is exactly what
    `ANALYSIS_PLAN.md` §8 forbids.

    `p_min_attainable` is `2 / 2**n`, the two-sided floor of the exact test when
    every delta shares a sign and the absolute deltas are distinct. It applies
    to the exact branch; when ties or zeros force the normal approximation,
    `method` says so and the floor is indicative rather than exact.

    Assumes symmetry of the delta distribution about its centre for the
    signed-rank null to be exactly distribution-free -- a stronger assumption
    than the sign-flip test's, which is one reason the sign-flip test is
    primary. Does not assume normality. Does not adjust for multiplicity.
    """
    d = _vector(deltas, 'deltas')
    n = int(d.size)
    if n < MIN_N_FOR_INFERENCE:
        return {'statistic': float('nan'), 'p': float('nan'), 'n': n,
                'p_min_attainable': float('nan'), 'at_p_floor': False,
                'method': 'refused', 'refused': True, 'reason': _too_few(n)}

    if np.all(d == 0):
        # Every delta is exactly zero: no evidence in either direction, and
        # SciPy raises rather than returning p=1. Say so explicitly.
        return {'statistic': 0.0, 'p': 1.0, 'n': n,
                'p_min_attainable': signflip_min_attainable_p(n),
                'at_p_floor': False, 'method': 'degenerate',
                'n_zero': n, 'ties': True, 'refused': False,
                'reason': 'all deltas are exactly zero'}

    nz = int(np.count_nonzero(d == 0))
    absd = np.abs(d[d != 0])
    ties = bool(absd.size != np.unique(absd).size)
    method = 'exact' if (nz == 0 and not ties and n <= 25) else 'approx'
    res = stats.wilcoxon(d, alternative='two-sided', zero_method='zsplit',
                         method=method, correction=(method == 'approx'))
    floor = signflip_min_attainable_p(n)
    return {'statistic': float(res.statistic), 'p': float(res.pvalue), 'n': n,
            'p_min_attainable': float(floor),
            'at_p_floor': bool(res.pvalue <= floor * (1.0 + 1e-9)),
            'method': method, 'n_zero': nz, 'ties': ties,
            'refused': False, 'reason': None}


@lru_cache(maxsize=64)
def _mwu_null_counts(n1: int, n2: int) -> tuple[int, ...]:
    """Exact counts of orderings giving each U, via the Gaussian binomial.

    `counts[u]` is the number of the C(n1+n2, n1) equally likely orderings with
    U = u, computed as the coefficients of the q-binomial
    `[n1+n2 choose n1]_q = prod_{i=1..n1} (1 - q**(n2+i)) / (1 - q**i)`.
    Python integers throughout, so the counts are exact rather than
    floating-point approximations of factorial ratios.
    """
    max_u = n1 * n2
    poly = [0] * (max_u + 1)
    poly[0] = 1
    for i in range(1, n1 + 1):
        k = n2 + i
        for u in range(max_u, k - 1, -1):          # multiply by (1 - q**k)
            poly[u] -= poly[u - k]
        for u in range(i, max_u + 1):              # divide by (1 - q**i)
            poly[u] += poly[u - i]
    return tuple(poly)


@lru_cache(maxsize=64)
def mwu_exact_null(n1: int, n2: int) -> np.ndarray | None:
    """Exact two-sided p for every attainable U at (n1, n2), or None.

    Returns an array indexed by U in 0..n1*n2 holding
    `min(1, 2 * min(P(U <= u), P(U >= u)))`, which is SciPy's exact two-sided
    convention and is verified against it in `self_test`. Returns None when
    `n1 * n2 > EXACT_MWU_MAX_PRODUCT`, where the polynomial is too large to be
    worth building; callers then fall back to the normal approximation and say
    so. Assumes no ties, which is what "exact" means for this test.
    """
    if n1 < 1 or n2 < 1 or n1 * n2 > EXACT_MWU_MAX_PRODUCT:
        return None
    counts = _mwu_null_counts(n1, n2)
    total = float(sum(counts))
    pmf = np.asarray([float(c) / total for c in counts], dtype=float)
    cdf = np.cumsum(pmf)                              # P(U <= u)
    sf = 1.0 - np.concatenate(([0.0], cdf[:-1]))      # P(U >= u)
    return np.minimum(1.0, 2.0 * np.minimum(cdf, sf))


def mwu_min_attainable_p(n1: int, n2: int) -> float:
    """`2 / C(n1+n2, n1)`: the exact two-sided floor, attained at U=0 or U=n1n2.

    1.08e-5 at n1=n2=10 (`ANALYSIS_PLAN.md` §6.1). Assumes no ties.
    """
    if n1 < 1 or n2 < 1:
        return float('nan')
    return 2.0 / float(math.comb(n1 + n2, n1))


def mwu_critical_values(n1: int, n2: int,
                        alpha: float = ALPHA) -> tuple[int, int] | None:
    """The two-sided exact rejection region as `(u_lo, u_hi)`, or None.

    Reject when `U <= u_lo` or `U >= u_hi`. At n1=n2=10 this reproduces the
    pinned values of `ANALYSIS_PLAN.md` §6.1: (23, 77) at 0.05, (17, 83) at
    0.0125, (14, 86) at 0.00625. Returns None when no attainable U reaches
    `alpha` -- which is itself the finding at very small n, not an error.
    """
    table = mwu_exact_null(n1, n2)
    if table is None:
        return None
    max_u = n1 * n2
    ok = np.nonzero(table[:max_u // 2 + 1] <= alpha)[0]
    if ok.size == 0:
        return None
    lo = int(ok.max())
    return lo, int(max_u - lo)


def mann_whitney(a: Sequence[float] | np.ndarray,
                 b: Sequence[float] | np.ndarray) -> dict:
    """Mann-Whitney U for `a` against `b`, with the attainable p floor.

    Reported for every contrast whatever the design, because it is the test the
    reviewers endorsed and the published paper used, so comparability with the
    published numbers is worth keeping (`ANALYSIS_PLAN.md` §2). It is not the
    primary test for a within-cell delta: the design is matched by seed, and the
    unpaired test discards that structure at a real cost in power -- MDE 1.39
    sigma against the paired 1.00 at n=10 (§6.2).

    `U` is `U(a vs b)`, i.e. `#(a > b) + 0.5 #(a == b)` over all n1*n2 cross
    pairs, so `U / (n1*n2)` is the relative effect `P(a > b) + 0.5 P(a = b)`.

    Assumes the two samples are independent and that observations are
    exchangeable within the pooled sample under the null. Under a pure location
    shift it is a test of that shift; in general it tests stochastic ordering,
    which is the correct reading here because the cells' dispersions differ by
    up to 8x (`ANALYSIS_PLAN.md` §3) -- use `brunner_munzel` for the estimand
    when that matters. Does **not** assume equal variances, normality, or equal
    sample sizes. Does not adjust for multiplicity.
    """
    x = _vector(a, 'a')
    y = _vector(b, 'b')
    n1, n2 = int(x.size), int(y.size)
    floor = mwu_min_attainable_p(n1, n2)
    if min(n1, n2) < MIN_N_FOR_INFERENCE:
        return {'U': float('nan'), 'p': float('nan'), 'n1': n1, 'n2': n2,
                'p_min_attainable': float(floor), 'at_p_floor': False,
                'method': 'refused', 'refused': True,
                'reason': _too_few(min(n1, n2))}

    pooled = np.concatenate((x, y))
    ties = bool(np.unique(pooled).size != pooled.size)
    exact = (not ties) and n1 * n2 <= EXACT_MWU_MAX_PRODUCT
    method = 'exact' if exact else 'asymptotic'
    res = stats.mannwhitneyu(x, y, alternative='two-sided', method=method)
    return {'U': float(res.statistic), 'p': float(res.pvalue),
            'n1': n1, 'n2': n2,
            'p_min_attainable': float(floor),
            'at_p_floor': bool(res.pvalue <= floor * (1.0 + 1e-9)),
            'method': method, 'ties': ties,
            'critical_values': mwu_critical_values(n1, n2, ALPHA),
            'refused': False, 'reason': None}


# ---------------------------------------------------------------------------
# 3. Effect sizes. No Cohen's d, ever (ANALYSIS_PLAN §8).
# ---------------------------------------------------------------------------
def hodges_lehmann(a: Sequence[float] | np.ndarray,
                   b: Sequence[float] | np.ndarray | None = None) -> float:
    """The Hodges-Lehmann shift estimate.

    One sample (`b is None`): the median of the Walsh averages
    `(a_i + a_j)/2` over all `i <= j`. This is the location estimate that goes
    with the signed-rank test, and it is what `ANALYSIS_PLAN.md` §2 names as the
    point estimate for the paired delta -- pass the per-seed deltas.

    Two samples: the median of all `n1*n2` pairwise differences `a_i - b_j`, the
    estimate that goes with Mann-Whitney U.

    Assumes a **location shift** for the estimate to be interpretable as one
    number describing the difference, i.e. that the two distributions differ by
    a constant. That assumption is violated between cells here, where SDs differ
    by up to 8x, which is why `ANALYSIS_PLAN.md` §3 uses Brunner-Munzel's
    relative effect for between-cell comparisons and reserves Hodges-Lehmann for
    the within-cell paired delta. Does not assume normality, and is not a mean:
    it is a median of pairwise averages, so it is robust to a minority of
    extreme runs but is *not* the arithmetic mean the sign-flip statistic uses.
    Carries no interval of its own -- pair it with `bootstrap_ci`.
    """
    x = _vector(a, 'a')
    if x.size == 0:
        return float('nan')
    if b is None:
        i, j = np.triu_indices(x.size, k=0)
        return float(np.median((x[i] + x[j]) / 2.0))
    y = _vector(b, 'b')
    if y.size == 0:
        return float('nan')
    return float(np.median(x[:, None] - y[None, :]))


def relative_effect(a: Sequence[float] | np.ndarray,
                    b: Sequence[float] | np.ndarray) -> float:
    """`theta = P(X > Y) + 0.5 P(X = Y)` for X drawn from `a`, Y from `b`.

    The ordinal effect size: the probability that a randomly chosen run from `a`
    beats a randomly chosen run from `b`, with ties split. 0.5 means no
    stochastic ordering; 1.0 means every `a` run beats every `b` run. Unit-free
    and defined without any location-shift assumption, which is why it is the
    between-cell estimand in `ANALYSIS_PLAN.md` §3.

    Assumes nothing about the shapes or the spreads. Does **not** describe *how
    much* better `a` is -- a theta of 0.9 is compatible with a tiny shift
    between two tight distributions, so report it next to a location estimate,
    never instead of one.
    """
    x = _vector(a, 'a')
    y = _vector(b, 'b')
    if x.size == 0 or y.size == 0:
        return float('nan')
    gt = float(np.count_nonzero(x[:, None] > y[None, :]))
    eq = float(np.count_nonzero(x[:, None] == y[None, :]))
    return (gt + 0.5 * eq) / float(x.size * y.size)


def _bm_theta_and_se(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """`theta = P(X>Y)+0.5P(X=Y)` and its Brunner-Munzel standard error.

    Midranks throughout, so ties are handled. The variance is the
    Brunner-Munzel (2000) estimator, which does *not* assume equal variances --
    that is the whole reason it is preferred here over a Wilcoxon-based
    interval.
    """
    n1, n2 = int(x.size), int(y.size)
    combined = stats.rankdata(np.concatenate((x, y)))
    rc_x, rc_y = combined[:n1], combined[n1:]
    r_x = stats.rankdata(x)
    r_y = stats.rankdata(y)
    mx, my = float(rc_x.mean()), float(rc_y.mean())
    # q = P(X < Y) + 0.5 P(X = Y); theta = 1 - q.
    q = (my - (n2 + 1) / 2.0) / n1
    theta = 1.0 - q
    if n1 < 2 or n2 < 2:
        return float(theta), float('nan')
    sx = float(np.sum((rc_x - r_x - mx + float(r_x.mean())) ** 2) / (n1 - 1))
    sy = float(np.sum((rc_y - r_y - my + float(r_y.mean())) ** 2) / (n2 - 1))
    var = n1 * sx + n2 * sy
    se = math.sqrt(var) / (n1 * n2) if var > 0 else 0.0
    return float(theta), float(se)


def brunner_munzel(a: Sequence[float] | np.ndarray,
                   b: Sequence[float] | np.ndarray,
                   n_boot: int = N_BOOT, alpha: float = ALPHA,
                   seed: int = BOOTSTRAP_SEED) -> dict:
    """Relative effect `theta = P(X>Y)+0.5P(X=Y)`, with a bootstrap-t CI.

    The between-cell estimator of `ANALYSIS_PLAN.md` §3, preferred over
    Hodges-Lehmann whenever the two groups' dispersions differ materially --
    here by up to 8x on the normalised score, which violates the location-shift
    assumption a shift estimate needs. The interval is bootstrap-t: the two
    groups are resampled independently, `t* = (theta* - theta)/se*` is
    accumulated using the Brunner-Munzel standard error on each resample, and
    the interval is `theta - t*_{1-alpha/2} se` to `theta - t*_{alpha/2} se`,
    clipped to [0, 1] because theta is a probability. When the studentisation
    degenerates -- `se = 0`, or too few usable resamples -- it falls back to the
    percentile interval and `fallback` says so.

    **No p-value is returned, by design.** `scipy.stats.brunnermunzel` will give
    one; this function deliberately does not, because every analysis that uses
    theta in this study is estimation-only and carries no p-value at all
    (`ANALYSIS_PLAN.md` §3, §7). The licensed directional statement is what the
    interval excludes.

    Assumes the two samples are independent and that the resampling unit is the
    run. Does **not** assume equal variances, equal shapes, normality, or a
    location shift. Does not tell you the size of a difference in score units --
    report a location estimate alongside.
    """
    x = _vector(a, 'a')
    y = _vector(b, 'b')
    n1, n2 = int(x.size), int(y.size)
    if min(n1, n2) >= 1:
        theta, se = _bm_theta_and_se(x, y)
    else:
        theta, se = float('nan'), float('nan')
    out: dict[str, Any] = {
        'theta': float(theta), 'se': float(se), 'n1': n1, 'n2': n2,
        'ci_lo': float('nan'), 'ci_hi': float('nan'),
        'method': 'refused', 'fallback': None, 'alpha': float(alpha),
        'n_boot': 0,
        'p_omitted_reason':
            'estimation-only analysis; ANALYSIS_PLAN.md §3 and §7 emit no '
            'p-value for the relative effect',
    }
    if min(n1, n2) < MIN_N_FOR_INFERENCE:
        out['refused'] = True
        out['reason'] = _too_few(min(n1, n2))
        out['ci'] = _refused_ci(out['reason'], alpha, min(n1, n2))
        return out

    rng = np.random.default_rng(seed)
    ix = rng.integers(0, n1, size=(int(n_boot), n1))
    iy = rng.integers(0, n2, size=(int(n_boot), n2))
    thetas = np.empty(int(n_boot), dtype=float)
    tstats = np.full(int(n_boot), np.nan, dtype=float)
    for k in range(int(n_boot)):
        tk, sk = _bm_theta_and_se(x[ix[k]], y[iy[k]])
        thetas[k] = tk
        if sk and np.isfinite(sk) and sk > 0:
            tstats[k] = (tk - theta) / sk
    usable = np.isfinite(tstats)
    lo_q, hi_q = 100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)

    if se and se > 0 and int(usable.sum()) >= max(50, int(n_boot) // 20):
        t_hi = float(np.percentile(tstats[usable], hi_q))
        t_lo = float(np.percentile(tstats[usable], lo_q))
        lo, hi = theta - t_hi * se, theta - t_lo * se
        out.update(method='bootstrap-t', n_boot=int(usable.sum()))
    else:
        lo = float(np.percentile(thetas, lo_q))
        hi = float(np.percentile(thetas, hi_q))
        out.update(method='percentile', n_boot=int(n_boot),
                   fallback='studentisation degenerate (se=0 or too few '
                            'usable resamples); percentile interval reported')
    out['ci_lo'] = float(min(max(lo, 0.0), 1.0))
    out['ci_hi'] = float(min(max(hi, 0.0), 1.0))
    out['ci'] = CI(out['ci_lo'], out['ci_hi'], method=out['method'],
                   fallback=out['fallback'], alpha=alpha, n=min(n1, n2))
    out['refused'] = False
    out['reason'] = None
    return out


def rank_biserial(a: Sequence[float] | np.ndarray,
                  b: Sequence[float] | np.ndarray) -> float:
    """Rank-biserial correlation from U: `2U/(n1 n2) - 1`, i.e. `2 theta - 1`.

    Runs from -1 (every `b` beats every `a`) through 0 (no ordering) to +1. A
    monotone re-scaling of the relative effect, reported because it is the
    conventional companion to Mann-Whitney U and because it is *not* Cohen's d:
    it involves no variance estimate and no normality assumption, which is what
    `ANALYSIS_PLAN.md` §8 requires.

    Assumes nothing beyond the two samples being comparable on an ordinal scale.
    Does **not** express the effect in score units, and does not become small
    merely because the difference is substantively small.
    """
    return float(2.0 * relative_effect(a, b) - 1.0)


def within_seed_correlation(a: Sequence[float] | np.ndarray,
                            b: Sequence[float] | np.ndarray) -> dict:
    """Pearson and Spearman correlation between two seed-aligned arms.

    `ANALYSIS_PLAN.md` §2.1 commits in advance to reporting `rho(scratch,
    transfer)` for every cell whatever it is, because pairing is a partial
    block, not a complete one -- it cannot remove the source run's own outcome
    or the post-divergence trajectory. A negative rho in a cell is reported as
    evidence that the pairing does not hold there, and the unpaired result is
    then given equal prominence. The paired test remains primary regardless, so
    that the choice cannot be made after seeing which test gives a smaller p.

    Emits **no p-value**: this is a reported descriptive quantity, not a test.
    Assumes the two vectors are aligned element-by-element by seed; the caller
    must guarantee that ordering, and a length mismatch raises.
    """
    x = _vector(a, 'a')
    y = _vector(b, 'b')
    if x.size != y.size:
        raise ValueError(f'seed-aligned vectors must match in length: '
                         f'{x.size} vs {y.size}')
    n = int(x.size)
    if n < MIN_N_FOR_INFERENCE:
        return {'pearson': float('nan'), 'spearman': float('nan'), 'n': n,
                'refused': True, 'reason': _too_few(n)}
    if float(np.ptp(x)) == 0.0 or float(np.ptp(y)) == 0.0:
        return {'pearson': float('nan'), 'spearman': float('nan'), 'n': n,
                'refused': True,
                'reason': 'a vector is constant; correlation is undefined'}
    return {'pearson': float(np.corrcoef(x, y)[0, 1]),
            'spearman': float(stats.spearmanr(x, y).statistic),
            'n': n, 'refused': False, 'reason': None}


# ---------------------------------------------------------------------------
# 4. Bootstrap intervals
# ---------------------------------------------------------------------------
def _apply_statistic(statistic: Callable, sample: np.ndarray,
                     axis: int | None = None):
    """Apply `statistic` along `axis` if it supports it, else loop."""
    if axis is None:
        return float(statistic(sample))
    try:
        return np.asarray(statistic(sample, axis=axis), dtype=float)
    except TypeError:
        return np.asarray([statistic(row) for row in sample], dtype=float)


def bootstrap_ci(values: Sequence[float] | np.ndarray,
                 statistic: Callable = np.mean,
                 n_boot: int = N_BOOT, alpha: float = ALPHA,
                 seed: int = BOOTSTRAP_SEED, method: str = 'bca') -> CI:
    """Bias-corrected and accelerated (BCa) bootstrap interval, with a fallback.

    `ANALYSIS_PLAN.md` §2 specifies a "bias-corrected seed-level bootstrap 95 %
    CI, 10,000 resamples, fixed seed 20260824", and those are the defaults. The
    resampling unit is whatever a row of `values` is -- for every confirmatory
    quantity in this study, a **seed**.

    BCa corrects two things a percentile interval does not: median bias in the
    bootstrap distribution (`z0`), and the dependence of the statistic's
    variance on its own value (the acceleration, from a jackknife over the same
    units). It falls back to the percentile interval, **and records that it did
    in `.fallback`**, when the bootstrap distribution is degenerate, when `z0`
    is not finite because every resample fell on one side of the estimate, when
    the jackknife has no spread, or when the corrected quantiles leave (0, 1).
    Pass `method='percentile'` to skip BCa deliberately.

    Assumes the resampling units are exchangeable and that the statistic is a
    smooth-enough functional for the bootstrap to be consistent. Does **not**
    assume normality of the data or of the statistic; that is the point.

    Does **not** work around a small n: at `n < MIN_N_FOR_INFERENCE` it returns
    a refused `CI` rather than a number, because `ANALYSIS_PLAN.md` §9 emits no
    interval at n<3. And note the ceiling that no method removes -- with n units
    there are only `n**n` distinct resamples, so at n=3 the interval is coarse
    whatever the label on it says.
    """
    x = _vector(values, 'values')
    n = int(x.size)
    if n < MIN_N_FOR_INFERENCE:
        return _refused_ci(_too_few(n), alpha, n)
    if method not in ('bca', 'percentile'):
        raise ValueError(f"method must be 'bca' or 'percentile', "
                         f"got {method!r}")

    theta_hat = float(_apply_statistic(statistic, x))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(int(n_boot), n))
    reps = np.asarray(_apply_statistic(statistic, x[idx], axis=1), dtype=float)
    reps = reps[np.isfinite(reps)]
    if reps.size == 0:
        return _refused_ci('every bootstrap replicate was non-finite', alpha, n)

    lo_q, hi_q = 100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)
    pct = (float(np.percentile(reps, lo_q)), float(np.percentile(reps, hi_q)))
    if method == 'percentile':
        return CI(*pct, method='percentile', alpha=alpha, n=n)

    def _fallback(reason: str) -> CI:
        return CI(*pct, method='percentile', fallback=reason, alpha=alpha, n=n)

    if float(np.ptp(reps)) == 0.0:
        return _fallback('bootstrap distribution is degenerate (zero spread)')

    below = float(np.count_nonzero(reps < theta_hat))
    equal = float(np.count_nonzero(reps == theta_hat))
    prop = (below + 0.5 * equal) / reps.size
    if not 0.0 < prop < 1.0:
        return _fallback('bias correction z0 is infinite (every replicate lies '
                         'on one side of the point estimate)')
    z0 = float(stats.norm.ppf(prop))

    jack = np.empty(n, dtype=float)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        keep[i] = False
        jack[i] = float(_apply_statistic(statistic, x[keep]))
        keep[i] = True
    if not np.all(np.isfinite(jack)):
        return _fallback('jackknife produced a non-finite value')
    dev = float(jack.mean()) - jack
    denom = 6.0 * float(np.sum(dev ** 2)) ** 1.5
    if denom == 0.0 or not np.isfinite(denom):
        return _fallback('jackknife has no spread, so the acceleration is '
                         'undefined')
    accel = float(np.sum(dev ** 3) / denom)

    bounds: list[float] = []
    for q in (alpha / 2.0, 1.0 - alpha / 2.0):
        z = float(stats.norm.ppf(q))
        adj_denom = 1.0 - accel * (z0 + z)
        if adj_denom == 0.0:
            return _fallback('BCa quantile adjustment divided by zero')
        adj = float(stats.norm.cdf(z0 + (z0 + z) / adj_denom))
        if not np.isfinite(adj) or not 0.0 < adj < 1.0:
            return _fallback('BCa adjusted quantile left the unit interval')
        bounds.append(float(np.percentile(reps, 100.0 * adj)))
    return CI(bounds[0], bounds[1], method='bca', alpha=alpha, n=n)


def _contrast_weights(spec: Any,
                      arms: Sequence[str]) -> dict[str, float] | None:
    """Normalise a contrast specification into arm coefficients, or None."""
    if callable(spec):
        return None
    if isinstance(spec, Mapping):
        return {str(k): float(v) for k, v in spec.items()}
    if isinstance(spec, (tuple, list)) and len(spec) == 2 \
            and all(isinstance(s, str) for s in spec):
        plus, minus = spec
        if plus == minus:
            raise ValueError(f'contrast {spec!r} differences an arm with '
                             f'itself')
        return {plus: 1.0, minus: -1.0}
    raise TypeError(
        f'contrast specification {spec!r} not understood. Give a 2-tuple '
        f'(plus_arm, minus_arm), a mapping of arm -> coefficient, or a '
        f'callable taking the dict of resampled per-seed arrays. Known arms: '
        f'{sorted(arms)}')


def paired_bootstrap(pairs_dict: Mapping[str, Sequence[float]],
                     contrasts: Mapping[str, Any],
                     n_boot: int = N_BOOT, seed: int = BOOTSTRAP_SEED,
                     alpha: float = ALPHA,
                     statistic: Callable = np.mean) -> dict:
    """One joint resampling of SEEDS, evaluated on every contrast at once.

    This is what `ANALYSIS_PLAN.md` §3 requires for the control contrasts of
    `DESIGN.md` §4. C0, C1, C2, C3 and C3b are measured on **the same seeds**,
    so `C2-C0`, `C3-C2` and `C1-C3` are correlated by construction. Four
    independent two-sample intervals would misstate their uncertainty and would
    make the arithmetic identity `(C2-C0)+(C3-C2)+(C1-C3) = C1-C0` look like an
    empirical decomposition -- which is precisely how revision 1 of the design
    went wrong (`DESIGN.md` §4.1). Here a single bootstrap replicate draws one
    vector of seed indices and applies it to *every* arm, so the contrasts'
    joint distribution, and hence their correlation matrix, is estimated.

    `pairs_dict` maps arm name -> per-seed values, **aligned by seed across
    arms**; the caller guarantees that ordering, and unequal lengths or any
    non-finite entry are refused rather than reconciled.

    `contrasts` maps a name to one of:

    * `(plus_arm, minus_arm)` -- the difference, the common case;
    * `{arm: coefficient, ...}` -- any linear combination, which is how the 2x2
      interaction contrast of `ANALYSIS_PLAN.md` §3 is expressed;
    * a callable taking `{arm: resampled array}` and returning a scalar, for
      anything that is not linear.

    `statistic` is applied to the per-seed contrast vector; the default is the
    mean, matching the sign-flip test's statistic. Pass `hodges_lehmann` to get
    the plan's HL point estimate under the same joint resampling.

    Assumes seeds are the exchangeable resampling unit and that the arms are
    seed-aligned. Does **not** assume the contrasts are independent -- the
    opposite -- nor normality, nor that the identity above has any empirical
    content. Emits no p-values: every analysis this serves is estimation-only.
    """
    arms = list(pairs_dict)
    if not arms:
        raise ValueError('pairs_dict is empty')
    data = {k: _vector(v, f'pairs_dict[{k!r}]') for k, v in pairs_dict.items()}
    sizes = {k: int(v.size) for k, v in data.items()}
    if len(set(sizes.values())) != 1:
        raise ValueError(
            f'arms must be seed-aligned and equal length; got {sizes}. A short '
            f'arm means a missing run, and reconciling it here would hide a '
            f'dropped seed (ANALYSIS_PLAN.md §8).')
    n = int(next(iter(sizes.values())))

    names = list(contrasts)
    weights = {name: _contrast_weights(spec, arms)
               for name, spec in contrasts.items()}
    for name, w in weights.items():
        if w is None:
            continue
        missing = sorted(set(w) - set(data))
        if missing:
            raise KeyError(f'contrast {name!r} names unknown arm(s) {missing}; '
                           f'known arms: {sorted(data)}')

    per_seed: dict[str, np.ndarray] = {
        name: sum(c * data[a] for a, c in w.items())
        for name, w in weights.items() if w is not None}

    out: dict[str, Any] = {
        'n': n, 'arms': arms, 'contrasts': names, 'n_boot': int(n_boot),
        'seed': int(seed), 'alpha': float(alpha),
        'statistic': getattr(statistic, '__name__', str(statistic)),
        'per_seed': per_seed, 'estimate': {}, 'ci': {}, 'distribution': {},
        'correlation': None, 'correlation_names': names,
        'p_omitted_reason':
            'joint seed bootstrap serves estimation-only analyses; '
            'ANALYSIS_PLAN.md §3 and §7 emit no p-value for these contrasts',
    }
    if n < MIN_N_FOR_INFERENCE:
        out['refused'] = True
        out['reason'] = _too_few(n)
        for name in names:
            out['estimate'][name] = (
                float(_apply_statistic(statistic, per_seed[name]))
                if name in per_seed else float('nan'))
            out['ci'][name] = _refused_ci(_too_few(n), alpha, n)
        return out

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(int(n_boot), n))
    dist: dict[str, np.ndarray] = {}
    for name in names:
        if name in per_seed:
            dist[name] = np.asarray(
                _apply_statistic(statistic, per_seed[name][idx], axis=1),
                dtype=float)
            out['estimate'][name] = float(
                _apply_statistic(statistic, per_seed[name]))
            # The marginal interval is BCa over the same seeds; the joint
            # distribution above is what carries the correlations.
            out['ci'][name] = bootstrap_ci(per_seed[name], statistic=statistic,
                                           n_boot=int(n_boot), alpha=alpha,
                                           seed=seed, method='bca')
        else:
            fn = contrasts[name]
            out['estimate'][name] = float(fn(dict(data)))
            dist[name] = np.asarray(
                [float(fn({a: v[idx[k]] for a, v in data.items()}))
                 for k in range(int(n_boot))], dtype=float)
            good = dist[name][np.isfinite(dist[name])]
            if good.size == 0:
                out['ci'][name] = _refused_ci(
                    'all bootstrap replicates non-finite', alpha, n)
            else:
                out['ci'][name] = CI(
                    float(np.percentile(good, 100.0 * alpha / 2.0)),
                    float(np.percentile(good, 100.0 * (1.0 - alpha / 2.0))),
                    method='percentile',
                    fallback='non-linear contrast; the BCa jackknife is not '
                             'defined for an arbitrary callable',
                    alpha=alpha, n=n)

    out['distribution'] = dist
    stacked = np.vstack([dist[k] for k in names])
    if len(names) > 1 and np.all(np.ptp(stacked, axis=1) > 0):
        out['correlation'] = np.corrcoef(stacked)
    else:
        corr = np.full((len(names), len(names)), np.nan)
        if len(names) == 1:
            corr[0, 0] = 1.0
        out['correlation'] = corr
    out['refused'] = False
    out['reason'] = None
    return out


# ---------------------------------------------------------------------------
# 5. Multiplicity
# ---------------------------------------------------------------------------
def _pvalue_array(pvalues: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(pvalues, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError('no p-values given')
    finite = arr[np.isfinite(arr)]
    if finite.size and (np.any(finite < 0.0) or np.any(finite > 1.0)):
        raise ValueError('p-values must lie in [0, 1]')
    return arr


def holm(pvalues: Sequence[float] | np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values, in the input order.

    The procedure for the one confirmatory family: 8 tests, the 4 within-cell
    deltas on each of the 2 co-primary endpoints (`ANALYSIS_PLAN.md` §2). Step
    down over the sorted p-values with multipliers `m, m-1, ..., 1`, then
    enforce monotonicity so a member can never be adjusted below a smaller raw
    p. Compare the result against `alpha` directly.

    A NaN member -- a test that could not be computed, e.g. a cell refused at
    n<3 -- is ranked last and returned as NaN, but **still counts towards m**.
    That is deliberate: shrinking the family because a member is missing is how
    a result gets rescued by relocating it into a family of one, which
    `ANALYSIS_PLAN.md` §7 exists to prevent.

    Controls the family-wise error rate under **arbitrary** dependence, so it
    needs no assumption about the correlation between the 8 tests. Does not
    control anything if family membership is chosen after seeing the data;
    membership is fixed by the plan, before launch.
    """
    p = _pvalue_array(pvalues)
    m = int(p.size)
    work = np.where(np.isfinite(p), p, 1.0)
    order = np.argsort(work, kind='stable')
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * float(work[i]))
        adjusted[i] = min(1.0, running)
    return np.where(np.isfinite(p), adjusted, np.nan)


def benjamini_hochberg(pvalues: Sequence[float] | np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values, in the input order.

    Used for the ablation screens (E3-E8, E12) and **for orientation only**:
    `ANALYSIS_PLAN.md` §3 and §7 pre-commit that a screen result is never
    asserted as a finding. A screen selects at most one follow-up, which is then
    run on `REPLICATE` seeds and reported as a fresh estimate. A q-value in this
    study is a sorting key, not evidence.

    Step up over the sorted p-values with multipliers `m/rank`, then enforce
    monotonicity. NaN members are ranked last, returned as NaN, and still count
    towards m, for the same reason as in `holm`.

    Controls the false discovery rate under independence or positive regression
    dependence -- an assumption that is **not** verified here, which is a
    further reason the output is not treated as an assertion.
    """
    p = _pvalue_array(pvalues)
    m = int(p.size)
    work = np.where(np.isfinite(p), p, 1.0)
    order = np.argsort(work, kind='stable')
    q = np.empty(m, dtype=float)
    running = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        running = min(running, m / rank * float(work[i]))
        q[i] = min(1.0, running)
    return np.where(np.isfinite(p), q, np.nan)


def holm_thresholds(m: int = CONFIRMATORY_FAMILY_SIZE,
                    alpha: float = ALPHA) -> np.ndarray:
    """The step-down thresholds `alpha/(m-k)`, k = 0..m-1, in ascending-p order.

    For the pre-registered family this is `0.00625, 0.00714, ..., 0.05`. The
    first entry is the bar the smallest p must clear, and at n=10 the exact
    paired test's floor is 0.00195, so the bar is attainable -- but only when
    every seed moves the same way (`ANALYSIS_PLAN.md` §2.2).
    """
    if m < 1:
        raise ValueError('family size must be at least 1')
    return alpha / (m - np.arange(m, dtype=float))


def multiplicity_ledger(n_estimation_only: int = 0,
                        n_screen_members: int = 0) -> list[dict]:
    """The ledger of `ANALYSIS_PLAN.md` §7, to be printed on every invocation.

    One row per family, with the family, its members, the procedure, the
    adjusted alpha and whether the family carries p-values at all. The family
    definitions are the pre-registered ones and are **not** parameters; only the
    counts of screen members and estimation-only analyses actually emitted are,
    because those depend on which experiments were run.
    """
    return [
        {'family': 'confirmatory',
         'members': CONFIRMATORY_FAMILY_SIZE,
         'description': '4 cells x 2 co-primary endpoints (final_score, '
                        'auc_score), within-cell delta = transfer - scratch',
         'procedure': 'Holm-Bonferroni',
         'adjusted_alpha': f'step-down from {HOLM_STRICTEST_ALPHA:g} '
                           f'to {ALPHA:g}',
         'carries_p_values': True},
        {'family': 'screens',
         'members': int(n_screen_members),
         'description': 'ablation screens (E3-E8, E12); orientation only, '
                        'never asserted as a finding',
         'procedure': 'Benjamini-Hochberg q',
         'adjusted_alpha': 'none -- no assertion permitted',
         'carries_p_values': True},
        {'family': 'estimation-only',
         'members': int(n_estimation_only),
         'description': 'RQ1, RQ3, RQ5, RQ6, the control contrasts, every '
                        'secondary and mechanism quantity',
         'procedure': 'none -- point estimate and interval',
         'adjusted_alpha': 'n/a',
         'carries_p_values': False},
    ]


# ---------------------------------------------------------------------------
# 6. Proportions and survival, for the right-censored metrics
# ---------------------------------------------------------------------------
def clopper_pearson(k: int, n: int, alpha: float = ALPHA) -> CI:
    """Exact (Clopper-Pearson) binomial interval for `k` successes in `n`.

    The primary summary for `steps_to_threshold` is P(threshold reached within
    budget) as `k/n` with this interval (`ANALYSIS_PLAN.md` §5), because the
    metric is right-censored and neither imputing the budget nor dropping the
    censored runs is permitted. At 0 of 10 the upper bound is 0.308 -- an
    informative statement where a p-value would be none.

    Uses the Beta quantile identity, so it is exact rather than an
    approximation, and it is conservative by construction: the coverage is at
    least `1 - alpha`, usually more.

    Assumes `n` independent Bernoulli trials with a common probability. Here the
    trials are seeds within one arm, and the censoring is administrative -- the
    same budget for every run, independent of the event time by construction --
    which is the benign case (`ANALYSIS_PLAN.md` §5). Does **not** use the event
    times, so it discards information a Kaplan-Meier curve keeps; report both.
    Unlike the score-based estimators here it is not gated at n<3: it is exact
    at any n, and at n=1 the honest interval is very nearly [0, 1].
    """
    k, n = int(k), int(n)
    if n < 1 or k < 0 or k > n:
        return _refused_ci(f'invalid counts k={k}, n={n}', alpha, n)
    lo = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1.0 - alpha / 2.0,
                                                 k + 1, n - k))
    return CI(lo, hi, method='clopper-pearson', alpha=alpha, n=n)


def proportion_reached(events: Sequence[int] | np.ndarray,
                       alpha: float = ALPHA) -> dict:
    """`k/n` reached, with an exact interval. The primary censored summary.

    `events` is 1 where the threshold was reached within the budget and 0 where
    the run was censored at it. Assumes administrative censoring at a common
    budget. Does **not** use the event times: pair it with `kaplan_meier`.
    """
    e = np.asarray(events).ravel()
    if e.size and not np.all(np.isin(e, (0, 1))):
        raise ValueError('events must be 0 (censored) or 1 (event observed)')
    k, n = int(e.sum()), int(e.size)
    return {'k': k, 'n': n,
            'proportion': (k / n) if n else float('nan'),
            'ci': clopper_pearson(k, n, alpha), 'alpha': float(alpha)}


def kaplan_meier(times: Sequence[float] | np.ndarray,
                 events: Sequence[int] | np.ndarray,
                 entry: Sequence[float] | np.ndarray | None = None,
                 alpha: float = ALPHA) -> dict:
    """Kaplan-Meier survival curve for right-censored times, with delayed entry.

    `times` are event or censoring times -- here env steps to the threshold,
    censored at the budget. `events` is 1 for an observed event, 0 for a
    censored observation. `entry` implements left truncation: a run enters the
    risk set only after its freeze window ends, which is what
    `ANALYSIS_PLAN.md` §5 asks for where a freeze is in force. Without `entry`
    every unit is at risk from time zero.

    "Survival" here is the probability of **not yet** having reached the
    threshold, so the curve starts at 1 and falls; `1 - survival` is the
    cumulative probability of having reached it. The interval is the
    complementary log-log transform of the Greenwood variance, which keeps the
    bounds inside [0, 1] without a normal approximation on the survival scale.

    Assumes censoring is independent of the event time -- satisfied here by
    construction, because the budget is the same for every run and is fixed
    before it starts. Does **not** extrapolate past the largest observed time,
    and the curve is undefined beyond it: if no run reaches the threshold the
    honest output is a flat curve at 1 and the Clopper-Pearson proportion, not
    an imputed median.
    """
    t = _vector(times, 'times')
    e = np.asarray(events).ravel().astype(int)
    if t.size != e.size:
        raise ValueError(f'times and events must match in length: '
                         f'{t.size} vs {e.size}')
    if e.size and not np.all(np.isin(e, (0, 1))):
        raise ValueError('events must be 0 (censored) or 1 (event observed)')
    ent = np.zeros_like(t) if entry is None else _vector(entry, 'entry')
    if ent.size != t.size:
        raise ValueError('entry must match times in length')
    if np.any(ent > t):
        raise ValueError('an entry time exceeds its event/censoring time')

    out_t: list[float] = [0.0]
    surv: list[float] = [1.0]
    risk: list[int] = [int(np.count_nonzero(ent <= 0.0))]
    n_ev: list[int] = [0]
    se: list[float] = [0.0]
    s = 1.0
    var_sum = 0.0                      # Greenwood accumulator
    for tt in np.unique(t[e == 1]):
        at_risk = int(np.count_nonzero((ent < tt) & (t >= tt)))
        d = int(np.count_nonzero((t == tt) & (e == 1)))
        if at_risk <= 0:
            continue
        s *= (1.0 - d / at_risk)
        if at_risk > d:
            var_sum += d / (at_risk * (at_risk - d))
        else:
            var_sum = float('inf')
        out_t.append(float(tt))
        surv.append(float(s))
        risk.append(at_risk)
        n_ev.append(d)
        se.append(float(s * math.sqrt(var_sum)) if np.isfinite(var_sum)
                  else float('nan'))

    survival = np.asarray(surv, dtype=float)
    se_arr = np.asarray(se, dtype=float)
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    lo = np.full(survival.shape, np.nan)
    hi = np.full(survival.shape, np.nan)
    ok = (survival > 0.0) & (survival < 1.0) & np.isfinite(se_arr)
    if np.any(ok):
        log_s = np.log(survival[ok])
        se_ll = se_arr[ok] / (survival[ok] * np.abs(log_s))
        lo[ok] = survival[ok] ** np.exp(z * se_ll)
        hi[ok] = survival[ok] ** np.exp(-z * se_ll)
    lo[survival == 1.0] = 1.0
    hi[survival == 1.0] = 1.0

    reached = np.nonzero(survival <= 0.5)[0]
    median = float(np.asarray(out_t)[reached[0]]) if reached.size else None
    return {'time': np.asarray(out_t, dtype=float),
            'survival': survival,
            'at_risk': np.asarray(risk, dtype=int),
            'n_events_at': np.asarray(n_ev, dtype=int),
            'se': se_arr, 'ci_lo': lo, 'ci_hi': hi,
            'n': int(t.size), 'n_events': int(e.sum()),
            'n_censored': int((1 - e).sum()),
            'median': median, 'alpha': float(alpha),
            'delayed_entry': entry is not None}


def logrank(groups: Mapping[str, tuple[Sequence[float], Sequence[int]]] |
            Sequence[tuple[Sequence[float], Sequence[int]]],
            min_events: int = 3) -> dict:
    """Two-sample log-rank test, or an explicit refusal.

    `ANALYSIS_PLAN.md` §5 pre-commits: "Log-rank only when both arms have at
    least 3 events; otherwise the proportion and its interval stand alone."
    That choice is fixed here so it cannot be made after seeing the censoring
    rate, and the refusal is a returned dict -- `usable=False` with the event
    counts and the reason -- rather than an exception, because the caller's job
    is to print the refusal and fall back to `proportion_reached`.

    Assumes proportional hazards for the statistic to be a test of a single
    hazard ratio, independent administrative censoring, and exactly two groups.
    Does **not** estimate an effect size: a log-rank p-value with no hazard
    ratio and no interval says nothing about magnitude, so report it beside the
    Kaplan-Meier curves and `proportion_reached`, never alone.
    """
    if isinstance(groups, Mapping):
        names = list(groups)
        pairs = [groups[k] for k in names]
    else:
        pairs = list(groups)
        names = [f'group{i}' for i in range(len(pairs))]
    if len(pairs) != 2:
        return {'usable': False, 'chi2': None, 'df': None, 'p': None,
                'groups': names,
                'reason': f'the log-rank implementation here handles exactly '
                          f'two arms; {len(pairs)} were given'}

    ts: list[np.ndarray] = []
    es: list[np.ndarray] = []
    for (t, e) in pairs:
        tv = _vector(t, 'times')
        ev = np.asarray(e).ravel().astype(int)
        if tv.size != ev.size:
            raise ValueError('times and events must match in length')
        if ev.size and not np.all(np.isin(ev, (0, 1))):
            raise ValueError('events must be 0 or 1')
        ts.append(tv)
        es.append(ev)
    counts = {names[i]: int(es[i].sum()) for i in range(2)}
    if min(counts.values()) < int(min_events):
        return {'usable': False, 'chi2': None, 'df': None, 'p': None,
                'groups': names, 'events': counts,
                'reason': f'pre-committed gate: both arms need >= '
                          f'{int(min_events)} events; observed {counts}. '
                          f'Report proportion_reached and the Kaplan-Meier '
                          f'curves instead (ANALYSIS_PLAN.md §5).'}

    all_t = np.concatenate(ts)
    all_e = np.concatenate(es)
    o1 = float(es[0].sum())
    e1 = 0.0
    var = 0.0
    for tt in np.unique(all_t[all_e == 1]):
        n_at = float(np.count_nonzero(all_t >= tt))
        n1 = float(np.count_nonzero(ts[0] >= tt))
        d = float(np.count_nonzero((all_t == tt) & (all_e == 1)))
        if n_at <= 1.0:
            continue
        e1 += d * n1 / n_at
        var += d * (n_at - d) * n1 * (n_at - n1) / (n_at ** 2 * (n_at - 1.0))
    if var <= 0.0:
        return {'usable': False, 'chi2': None, 'df': None, 'p': None,
                'groups': names, 'events': counts,
                'reason': 'log-rank variance is zero (no informative risk '
                          'set)'}
    chi2 = (o1 - e1) ** 2 / var
    return {'usable': True, 'chi2': float(chi2), 'df': 1,
            'p': float(stats.chi2.sf(chi2, 1)),
            'observed': o1, 'expected': float(e1), 'variance': float(var),
            'groups': names, 'events': counts,
            'effect_size_omitted_reason':
                'the log-rank statistic is not an effect size; report the '
                'Kaplan-Meier curves and proportion_reached alongside'}


# ---------------------------------------------------------------------------
# 7. Ordered alternative: the shift gradient of H4
# ---------------------------------------------------------------------------
def _jt_from_matrix(comp: np.ndarray,
                    index_sets: Sequence[np.ndarray]) -> float:
    """J from a precomputed comparison matrix and a grouping of pooled indices.

    `comp[p, q] = [v_q > v_p] + 0.5 [v_q == v_p]`, so summing the sub-block for
    an ordered group pair (i < j) counts that pair's concordant cross pairs.
    """
    total = 0.0
    k = len(index_sets)
    for i in range(k - 1):
        for j in range(i + 1, k):
            total += float(comp[np.ix_(index_sets[i], index_sets[j])].sum())
    return total


def _comparison_matrix(values: np.ndarray) -> np.ndarray:
    comp = (values[None, :] > values[:, None]).astype(float) \
        + 0.5 * (values[None, :] == values[:, None]).astype(float)
    np.fill_diagonal(comp, 0.0)
    return comp


def jonckheere_terpstra(groups_in_order: Sequence[Sequence[float]],
                        n_perm: int = 10_000, seed: int = BOOTSTRAP_SEED,
                        alpha: float = ALPHA, n_boot: int = 2_000) -> dict:
    """Jonckheere-Terpstra ordered-alternative trend statistic, standardised.

    Tests, against the null of no difference, the *ordered* alternative that the
    groups are stochastically increasing in the order given. This is H4's
    estimator: the transfer effect against measured dynamics shift along the
    wind axis, at fixed interface and fixed protocol (`DESIGN.md` §2.3,
    `ANALYSIS_PLAN.md` §3). The order comes from the design -- the wind
    levels -- and must not be chosen after seeing the data; doing so would
    invalidate everything below.

    Returns the statistic `J`, its null mean and variance, the standardised `z`,
    a two-sided permutation p-value, and a group-stratified bootstrap interval
    on `z`. The permutation p is valid with ties; the closed-form variance
    behind `z` is the no-tie formula, so where ties are present `z` is
    approximate and `ties` says so.

    **The p-value here is orientation only.** RQ5 is estimation-only
    (`ANALYSIS_PLAN.md` §3, §7); it is not in the confirmatory family and
    carries no error budget. `p_role` records that in the output so a caller
    cannot lift the number out of context.

    Assumes independent observations and a *pre-specified* ordering. Does
    **not** assume normality, equal variances, equal group sizes, or a linear
    trend -- it is sensitive to any monotone ordering, and correspondingly it
    cannot distinguish a smooth gradient from a single step, which is why the
    per-level estimates are reported beside it.
    """
    groups = [_vector(g, f'group{i}') for i, g in enumerate(groups_in_order)]
    sizes = [int(g.size) for g in groups]
    k = len(groups)
    out: dict[str, Any] = {
        'J': float('nan'), 'expected': float('nan'), 'variance': float('nan'),
        'z': float('nan'), 'p_perm': float('nan'), 'n_perm': 0,
        'ci_lo': float('nan'), 'ci_hi': float('nan'),
        'group_sizes': sizes, 'n': int(sum(sizes)), 'k': k,
        'p_role': 'orientation only -- RQ5 is estimation-only '
                  '(ANALYSIS_PLAN.md §3, §7); this p carries no error budget',
    }
    if k < 3:
        out['refused'] = True
        out['reason'] = (f'an ordered alternative needs at least 3 ordered '
                         f'levels; {k} given')
        return out
    if min(sizes) < MIN_N_FOR_INFERENCE:
        out['refused'] = True
        out['reason'] = _too_few(min(sizes))
        return out

    pooled = np.concatenate(groups)
    n_total = int(pooled.size)
    comp = _comparison_matrix(pooled)
    bounds = np.cumsum([0] + sizes)
    idx_sets = [np.arange(bounds[i], bounds[i + 1]) for i in range(k)]

    j_obs = _jt_from_matrix(comp, idx_sets)
    expected = (n_total ** 2 - float(sum(s * s for s in sizes))) / 4.0
    variance = (n_total ** 2 * (2 * n_total + 3)
                - float(sum(s * s * (2 * s + 3) for s in sizes))) / 72.0
    z = (j_obs - expected) / math.sqrt(variance) if variance > 0 \
        else float('nan')

    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(int(n_perm)):
        perm = rng.permutation(n_total)
        js = _jt_from_matrix(comp, [perm[s] for s in idx_sets])
        if abs(js - expected) >= abs(j_obs - expected) - 1e-9:
            hits += 1
    p_perm = (1.0 + hits) / (int(n_perm) + 1.0)

    # Group-stratified bootstrap on the standardised statistic, so the reported
    # trend carries an interval rather than only a p-value.
    zs = np.empty(int(n_boot), dtype=float)
    for bi in range(int(n_boot)):
        resampled = [g[rng.integers(0, g.size, g.size)] for g in groups]
        pv = np.concatenate(resampled)
        jb = _jt_from_matrix(_comparison_matrix(pv), idx_sets)
        zs[bi] = (jb - expected) / math.sqrt(variance) if variance > 0 \
            else np.nan
    good = zs[np.isfinite(zs)]
    lo_q, hi_q = 100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)

    out.update(J=float(j_obs), expected=float(expected),
               variance=float(variance), z=float(z),
               p_perm=float(min(1.0, p_perm)), n_perm=int(n_perm),
               ci_lo=float(np.percentile(good, lo_q)) if good.size
               else float('nan'),
               ci_hi=float(np.percentile(good, hi_q)) if good.size
               else float('nan'),
               n_boot=int(n_boot),
               ties=bool(np.unique(pooled).size != pooled.size),
               direction=('increasing' if j_obs > expected else
                          'decreasing' if j_obs < expected else 'flat'),
               refused=False, reason=None)
    out['ci'] = CI(out['ci_lo'], out['ci_hi'], method='percentile',
                   alpha=alpha, n=int(min(sizes)))
    return out


# ---------------------------------------------------------------------------
# 8. Minimum detectable effect, by simulation
# ---------------------------------------------------------------------------
def _sign_matrix(n: int) -> np.ndarray:
    bits = (np.arange(2 ** n, dtype=np.int64)[:, None]
            >> np.arange(n, dtype=np.int64)) & 1
    return (1 - 2 * bits).astype(float)


def _bisect_effect(power_at: Callable[[float], float], target: float,
                   hi: float = 6.0, tol: float = 5e-4,
                   max_iter: int = 40) -> tuple[float, float]:
    """Smallest effect whose simulated power reaches `target`, by bisection.

    Uses common random numbers: `power_at` must evaluate the *same* simulated
    datasets shifted by the candidate effect. That makes the power curve
    monotone in practice and the bisection deterministic given the seed.
    """
    top = power_at(hi)
    if top < target:
        return float('inf'), top
    lo = 0.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if power_at(mid) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    mid = 0.5 * (lo + hi)
    return mid, power_at(mid)


def mde_signflip(n: int, alpha: float = ALPHA, power: float = 0.8,
                 n_sim: int = MDE_SIM_REPLICATES, seed: int = BOOTSTRAP_SEED,
                 chunk: int = 2_000) -> MDE:
    """Minimum detectable paired effect for the sign-flip test, in sigma units.

    Simulates `n_sim` datasets of `n` deltas drawn as `N(mu, 1)`, runs the exact
    sign-flip test on each, and bisects on `mu` for the smallest shift whose
    rejection rate reaches `power`. The unit is sigma_delta -- the SD of the
    *paired delta*, not of either arm -- so translating it into score units
    needs the observed delta SD, as `ANALYSIS_PLAN.md` §6.3 does.

    Reproduces the pinned values of `ANALYSIS_PLAN.md` §6.2 at n=10:
    **1.00 sigma at alpha=0.05** and **1.54 sigma at alpha=0.00625**, the Holm
    step-down floor over the 8-test family. `self_test` asserts both.

    The null distribution is enumerated exactly for
    `n <= EXACT_SIGNFLIP_MDE_MAX_N` (=15); above that the *power simulation*
    switches to Monte Carlo sign assignments and `.method` says so. That cap is
    lower than the test's own cap of 20 because here the enumeration runs once
    per simulated dataset rather than once per call.

    Assumes a normal shift **for the planning calculation only** -- stated
    plainly, because the real deltas are not normal (LunarLander returns are
    bimodal, crash versus land). A normal shift is the conventional planning
    reference and makes the number comparable with published power tables; the
    sign-flip test itself assumes nothing of the kind. Does **not** predict this
    study's power: the SDs it gets scaled by come from the published runs, which
    used a different protocol and budget, so it is a planning input, not a
    prediction (§6.4), and it is not re-tuned after seeing results.
    """
    n = int(n)
    if n < MIN_N_FOR_INFERENCE:
        return MDE(float('nan'), method='refused', n=n, alpha=alpha,
                   power_target=power)
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((int(n_sim), n))
    if n <= EXACT_SIGNFLIP_MDE_MAX_N:
        signs = _sign_matrix(n)
        method = 'exact enumeration of 2**n sign assignments'
    else:
        n_mc = 4_096
        signs = (rng.integers(0, 2, size=(n_mc, n)).astype(float) * 2.0 - 1.0)
        method = f'Monte Carlo sign assignments (n_perm={n_mc})'
    block = max(1, min(int(chunk), 2_000_000 // signs.shape[0]))

    def power_at(mu: float) -> float:
        rejected = 0
        for start in range(0, z.shape[0], block):
            d = z[start:start + block] + mu
            t = d @ signs.T
            obs = np.abs(d.sum(axis=1))[:, None]
            p = (np.abs(t) >= obs - 1e-9).mean(axis=1)
            rejected += int(np.count_nonzero(p <= alpha))
        return rejected / z.shape[0]

    effect, achieved = _bisect_effect(power_at, power)
    return MDE(effect, method=method, n_sim=int(n_sim), power_target=power,
               power_achieved=achieved, alpha=alpha, n=n)


def mde_mann_whitney(n: int, alpha: float = ALPHA, power: float = 0.8,
                     n_sim: int = MDE_SIM_REPLICATES,
                     seed: int = BOOTSTRAP_SEED, n2: int | None = None,
                     chunk: int = 4_000) -> MDE:
    """Minimum detectable unpaired effect for Mann-Whitney U, in sigma units.

    Simulates `n_sim` pairs of independent samples, `N(mu, 1)` against
    `N(0, 1)`, applies the **exact** two-sided rejection region from
    `mwu_exact_null`, and bisects on `mu` for `power`. The unit is sigma, the
    common within-group SD.

    Reproduces the pinned values of `ANALYSIS_PLAN.md` §6.2 at n=10 vs 10:
    **1.39 sigma at alpha=0.05** and **1.87 sigma at alpha=0.00625**. Compare
    `mde_signflip`'s 1.00 and 1.54: at this sample size the matched-seed design
    is worth roughly a 40 % reduction in the detectable effect, which is why the
    paired test is primary (§6.2) and why discarding the pairing is a real cost
    rather than a stylistic choice.

    Falls back to the normal-approximation critical value when the exact null is
    too large to build, and `.method` says so. Returns `inf` when no attainable
    U reaches `alpha` at that sample size -- the honest answer, not an error.

    Assumes a normal location shift with equal variances, for the planning
    number only; the test itself assumes neither. Does **not** describe this
    study's actual power (see `mde_signflip`), and does not apply to the
    between-cell comparisons whose SDs differ by up to 8x -- there the relevant
    statement is the interval on the relative effect, not a detectable shift.
    """
    n1 = int(n)
    n2 = n1 if n2 is None else int(n2)
    if min(n1, n2) < MIN_N_FOR_INFERENCE:
        return MDE(float('nan'), method='refused', n=n1, alpha=alpha,
                   power_target=power)
    max_u = n1 * n2
    crit = mwu_critical_values(n1, n2, alpha)
    if crit is None:
        if mwu_exact_null(n1, n2) is not None:
            return MDE(float('inf'),
                       method='exact null; no attainable U reaches alpha at '
                              'this n',
                       n_sim=0, power_target=power, alpha=alpha, n=n1)
        sd = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
        zc = float(stats.norm.ppf(1.0 - alpha / 2.0))
        lo = int(math.floor(max_u / 2.0 - zc * sd))
        crit = (lo, max_u - lo)
        method = 'normal approximation to the U null (exact table too large)'
    else:
        method = 'exact U null'

    rng = np.random.default_rng(seed)
    za = rng.standard_normal((int(n_sim), n1))
    zb = rng.standard_normal((int(n_sim), n2))
    lo_c, hi_c = crit

    def power_at(mu: float) -> float:
        rejected = 0
        for start in range(0, za.shape[0], int(chunk)):
            a = za[start:start + int(chunk)] + mu
            b = zb[start:start + int(chunk)]
            u = (a[:, :, None] > b[:, None, :]).sum(axis=(1, 2))
            rejected += int(np.count_nonzero((u <= lo_c) | (u >= hi_c)))
        return rejected / za.shape[0]

    effect, achieved = _bisect_effect(power_at, power)
    return MDE(effect, method=method, n_sim=int(n_sim), power_target=power,
               power_achieved=achieved, alpha=alpha, n=n1)


# ---------------------------------------------------------------------------
# 9. Equivalence and exclusion
# ---------------------------------------------------------------------------
def equivalence_from_ci(lo: float, hi: float,
                        margin: float = EQUIVALENCE_MARGIN,
                        sd: float | None = None) -> dict:
    """Equivalence verdict from an interval, plus the exclusion bound. Not TOST.

    `ANALYSIS_PLAN.md` §4: equivalence is assessed by whether the 95 %
    bootstrap CI on the paired delta lies entirely inside the margin, using the
    interval already reported. No separate test and no new error budget. TOST is
    refused for two reasons: it is parametric on data declared non-normal, and
    at n=10 it cannot support a margin this small.

    The margin is fixed at +/-0.05 normalised-score units -- about 20 return
    points on LunarLander, a tenth of the distance from a random policy to the
    solved threshold -- and re-deriving it after seeing the CI is forbidden
    (§8).

    Verdicts: `equivalent` when the whole interval is inside +/-margin;
    `not_equivalent` when the whole interval is outside it on one side;
    `inconclusive` otherwise; `untestable` when the interval would qualify as
    equivalent but the cell's across-seed `sd` exceeds the margin, because
    `ANALYSIS_PLAN.md` §4 pre-commits that an equivalence claim is available
    only where dispersion permits it -- 0.05 is 1.17 SD in the quiet cells and
    0.14 SD in the noisy dueling scratch cell, where it is hopeless. Pass `sd`
    to have that check applied; omit it and the verdict is geometric only.

    **The exclusion bound is always returned and always reportable**, whatever
    the verdict, because it is the powered honest claim and the form the
    abstract uses: "a degradation worse than X score units is excluded at 95 %".
    `exclusion_bound` is the interval limit nearer the null -- the conservative
    bound on the magnitude -- while `worst_degradation_excluded` and
    `best_improvement_excluded` are the two directional statements, on the
    convention that the delta is `transfer - scratch`, so negative is worse.

    Assumes the interval was built for the delta on the scale the margin is in,
    and that its nominal coverage is roughly right; a bootstrap interval at
    n=10 is approximate, so a verdict that turns on the third decimal place
    should not be asserted. Does **not** convert a null into equivalence: an
    `inconclusive` verdict licenses only the exclusion sentence.
    """
    margin = abs(float(margin))
    lo, hi = float(lo), float(hi)
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return {'verdict': 'refused', 'margin': margin, 'ci_lo': lo,
                'ci_hi': hi, 'exclusion_bound': float('nan'),
                'worst_degradation_excluded': float('nan'),
                'best_improvement_excluded': float('nan'),
                'sd': None if sd is None else float(sd),
                'reason': 'the interval is not finite (n<3, or the estimator '
                          'refused)',
                'sentence': 'No interval is available, so nothing is '
                            'excluded.'}
    if hi < lo:
        lo, hi = hi, lo

    if lo >= -margin and hi <= margin:
        verdict = 'equivalent'
    elif lo >= margin or hi <= -margin:
        verdict = 'not_equivalent'
    else:
        verdict = 'inconclusive'

    reason = None
    if verdict == 'equivalent' and sd is not None and np.isfinite(sd) \
            and float(sd) > margin:
        verdict = 'untestable'
        reason = (f'across-seed SD {float(sd):.4g} exceeds the margin '
                  f'{margin:g}: equivalence is untestable in this cell at '
                  f'this n (ANALYSIS_PLAN.md §4)')

    exclusion = lo if abs(lo) <= abs(hi) else hi
    worst_deg = max(0.0, -lo)
    best_imp = max(0.0, hi)
    if lo > 0.0:
        sentence = (f'Any degradation is excluded at 95 %; the delta is at '
                    f'least {lo:.4g} score units.')
    elif hi < 0.0:
        sentence = (f'Any improvement is excluded at 95 %; the delta is at '
                    f'most {hi:.4g} score units.')
    else:
        sentence = (f'A degradation worse than {worst_deg:.4g} score units is '
                    f'excluded at 95 %, as is an improvement better than '
                    f'{best_imp:.4g}.')
    return {'verdict': verdict, 'margin': margin, 'ci_lo': lo, 'ci_hi': hi,
            'exclusion_bound': float(exclusion),
            'worst_degradation_excluded': float(worst_deg),
            'best_improvement_excluded': float(best_imp),
            'sd': None if sd is None else float(sd),
            'reason': reason, 'sentence': sentence}


# ---------------------------------------------------------------------------
# 10. Self-test. Every pinned number in ANALYSIS_PLAN §6 is asserted here.
# ---------------------------------------------------------------------------
def self_test(verbose: bool = True) -> dict:
    """Assert the pinned reference values and the primitives' known behaviour.

    The critical values and minimum detectable effects in `ANALYSIS_PLAN.md` §6
    are load-bearing: they are the numbers that justify the single-family
    decision, and the plan says they were "computed exactly, not asserted".
    This function is where that claim is checked, so a refactor cannot quietly
    change what the plan promises. It takes a few seconds, because the MDEs are
    simulated at 20,000 replicates as the plan states.
    """
    checks: list[tuple[str, Any]] = []

    def note(label: str, value: Any) -> None:
        checks.append((label, value))
        if verbose:
            print(f'  {label:<57} {value}')

    if verbose:
        print('Exact Mann-Whitney null, n1 = n2 = 10 (ANALYSIS_PLAN §6.1)')
    for a, expect in ((0.05, (23, 77)), (0.0125, (17, 83)),
                      (0.00625, (14, 86))):
        got = mwu_critical_values(10, 10, a)
        assert got == expect, f'alpha={a}: expected {expect}, got {got}'
        note(f'reject U<={expect[0]} or U>={expect[1]} at alpha={a:g}', got)
    p_floor_u = mwu_min_attainable_p(10, 10)
    assert abs(p_floor_u - 1.08e-5) < 5e-8, p_floor_u
    note('smallest attainable two-sided MWU p at 10 vs 10', f'{p_floor_u:.3e}')
    p_floor_paired = signflip_min_attainable_p(10)
    assert abs(p_floor_paired - 2.0 / 1024.0) < 1e-12, p_floor_paired
    note('smallest attainable paired p at n=10 (2/1024)',
         f'{p_floor_paired:.6f}')

    # The exact table must agree with SciPy, or "exact" is a claim not a fact.
    rng = np.random.default_rng(7)
    table = mwu_exact_null(10, 10)
    for _ in range(50):
        res = stats.mannwhitneyu(rng.normal(size=10), rng.normal(size=10),
                                 alternative='two-sided', method='exact')
        assert abs(table[int(round(res.statistic))] - res.pvalue) < 1e-12
    note('exact MWU table agrees with scipy over 50 random samples', 'ok')

    if verbose:
        print('\nSign-flip test')
    sf = sign_flip_test([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    assert sf['exact'] and sf['n_perm'] == 1024
    assert abs(sf['p_two_sided'] - 2.0 / 1024.0) < 1e-12, sf
    assert sf['at_p_floor'] and sf['all_same_sign']
    note('all 10 deltas one sign -> p = 2/1024', f"{sf['p_two_sided']:.6f}")
    sym = sign_flip_test([1.0, -1.0, 2.0, -2.0, 3.0, -3.0])
    assert abs(sym['statistic']) < 1e-12 and sym['p_two_sided'] == 1.0
    note('perfectly symmetric deltas -> p = 1', sym['p_two_sided'])
    assert sign_flip_test([0.1, 0.2])['refused'] is True
    note('n=2 refuses a test (ANALYSIS_PLAN §9)', 'ok')
    mc = sign_flip_test(rng.normal(0.8, 1.0, size=24), n_perm=20_000)
    assert mc['exact'] is False and mc['n_perm'] == 20_000
    note('n=24 switches to Monte Carlo and reports the count', mc['n_perm'])

    if verbose:
        print('\nRank tests and effect sizes')
    w = wilcoxon_signed_rank([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                              1.0])
    assert abs(w['p'] - 2.0 / 1024.0) < 1e-12 and w['at_p_floor'], w
    note('Wilcoxon at n=10, all one sign -> p = 2/1024', f"{w['p']:.6f}")
    mw = mann_whitney(np.arange(10) + 100.0, np.arange(10.0))
    assert mw['U'] == 100.0 and mw['at_p_floor'], mw
    assert abs(mw['p'] - p_floor_u) < 1e-12
    note('fully separated 10 vs 10 -> U=100, p at the floor', f"{mw['p']:.3e}")
    assert abs(hodges_lehmann([1.0, 2.0, 3.0]) - 2.0) < 1e-12
    assert abs(hodges_lehmann([5.0, 6.0, 7.0], [1.0, 2.0, 3.0]) - 4.0) < 1e-12
    note('Hodges-Lehmann one-sample and two-sample on known data', 'ok')
    assert abs(relative_effect([2.0, 3.0], [0.0, 1.0]) - 1.0) < 1e-12
    assert abs(relative_effect([1.0, 1.0], [1.0, 1.0]) - 0.5) < 1e-12
    assert abs(rank_biserial([2.0, 3.0], [0.0, 1.0]) - 1.0) < 1e-12
    note('theta = 1 when separated, 0.5 when identical; rbc = 2 theta - 1',
         'ok')
    bm = brunner_munzel(rng.normal(1.0, 1.0, 12), rng.normal(0.0, 3.0, 12),
                        n_boot=2_000)
    assert 'p' not in bm and 0.0 <= bm['ci_lo'] <= bm['ci_hi'] <= 1.0
    note('Brunner-Munzel returns theta + CI and no p-value',
         f"theta={bm['theta']:.3f} ci=({bm['ci_lo']:.3f}, {bm['ci_hi']:.3f})")
    a1, b1 = rng.normal(0.0, 1.0, 15), rng.normal(0.5, 2.0, 11)
    t1 = _bm_theta_and_se(a1, b1)[0]
    t2 = _bm_theta_and_se(b1, a1)[0]
    assert abs(t1 + t2 - 1.0) < 1e-10, (t1, t2)
    assert abs(t1 - relative_effect(a1, b1)) < 1e-10
    note('midrank theta is antisymmetric and matches the direct count', 'ok')
    rho = within_seed_correlation([1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 5.0, 9.0])
    assert rho['pearson'] > 0.9 and 'p' not in rho
    note('within-seed correlation is reported without a p-value',
         f"pearson={rho['pearson']:.4f} spearman={rho['spearman']:.4f}")

    if verbose:
        print('\nBootstrap')
    ci = bootstrap_ci(rng.normal(0.0, 1.0, 40), n_boot=4_000)
    assert ci.method in ('bca', 'percentile') and ci.lo < ci.hi
    lo, hi = ci                                  # must unpack as a 2-tuple
    assert (lo, hi) == (ci[0], ci[1])
    note('bootstrap_ci unpacks as (lo, hi) and records its method', repr(ci))
    flat = bootstrap_ci([1.0, 1.0, 1.0, 1.0])
    assert flat.method == 'percentile' and flat.fallback
    note('degenerate data falls back to percentile and reports it',
         flat.fallback[:42] + '...')
    assert bootstrap_ci([1.0, 2.0]).refused
    note('n=2 refuses an interval', 'ok')
    try:
        bootstrap_ci([1.0, 2.0, np.nan, 4.0])
    except ValueError:
        note('a non-finite entry is refused, never dropped', 'ok')
    else:                                                   # pragma: no cover
        raise AssertionError('a NaN entry was not refused')

    pb = paired_bootstrap(
        {'c0': [0.1, 0.2, 0.3, 0.4, 0.5], 'c1': [0.3, 0.5, 0.4, 0.9, 0.6],
         'c2': [0.0, 0.3, 0.2, 0.5, 0.4]},
        {'c1-c0': ('c1', 'c0'), 'c2-c0': ('c2', 'c0'), 'c1-c2': ('c1', 'c2'),
         'same-as-first': {'c1': 1.0, 'c0': -1.0}},
        n_boot=2_000)
    tot = pb['estimate']['c2-c0'] + pb['estimate']['c1-c2']
    assert abs(tot - pb['estimate']['c1-c0']) < 1e-12
    assert abs(pb['estimate']['same-as-first']
               - pb['estimate']['c1-c0']) < 1e-12
    assert pb['correlation'].shape == (4, 4)
    assert abs(pb['correlation'][0, 3] - 1.0) < 1e-9
    note('joint seed bootstrap: telescoping identity holds exactly',
         f"{pb['estimate']['c1-c0']:.6f}")
    note('and a duplicated contrast correlates at 1.0 in the joint resample',
         round(float(pb['correlation'][0, 3]), 6))
    try:
        paired_bootstrap({'a': [1.0, 2.0, 3.0], 'b': [1.0, 2.0]},
                         {'d': ('a', 'b')})
    except ValueError:
        note('an unequal-length arm is refused, not reconciled', 'ok')
    else:                                                   # pragma: no cover
        raise AssertionError('a short arm was not refused')

    if verbose:
        print('\nMultiplicity')
    h = holm([0.001, 0.008, 0.04, 0.2])
    assert np.allclose(h, [0.004, 0.024, 0.08, 0.2]), h
    note('Holm on [0.001, 0.008, 0.04, 0.2]', np.round(h, 6).tolist())
    q = benjamini_hochberg([0.001, 0.008, 0.04, 0.2])
    assert np.allclose(q, [0.004, 0.016, 0.0533333333, 0.2]), q
    note('BH q on the same p-values', np.round(q, 6).tolist())
    hn = holm([0.001, np.nan, 0.04, 0.2])
    assert np.isnan(hn[1]) and abs(hn[0] - 0.004) < 1e-12
    note('a NaN member stays NaN and still counts towards m=4', 'ok')
    thr = holm_thresholds()
    assert abs(thr[0] - HOLM_STRICTEST_ALPHA) < 1e-15
    assert abs(thr[-1] - ALPHA) < 1e-15
    note('Holm thresholds over the 8-test family',
         f'{thr[0]:.5f} .. {thr[-1]:.5f}')
    ledger = multiplicity_ledger(n_estimation_only=17, n_screen_members=42)
    assert ledger[0]['members'] == CONFIRMATORY_FAMILY_SIZE
    assert ledger[2]['carries_p_values'] is False
    note('multiplicity ledger rows (confirmatory / screens / estimation)',
         len(ledger))

    if verbose:
        print('\nProportions and survival')
    cp = clopper_pearson(0, 10)
    assert cp.lo == 0.0 and abs(cp.hi - 0.3085) < 5e-4, cp
    note('Clopper-Pearson at 0 of 10 (ANALYSIS_PLAN §5)',
         f'[{cp.lo:.4f}, {cp.hi:.4f}]')
    cp2 = clopper_pearson(10, 10)
    assert cp2.hi == 1.0 and abs(cp2.lo - 0.6915) < 5e-4
    note('and at 10 of 10', f'[{cp2.lo:.4f}, {cp2.hi:.4f}]')
    pr = proportion_reached([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    assert pr['k'] == 1 and pr['n'] == 10 and pr['ci'].hi < 0.5
    note('proportion_reached at 1 of 10', f"{pr['proportion']:.2f} {pr['ci']}")
    km = kaplan_meier([1.0, 2.0, 3.0, 4.0], [1, 0, 1, 0])
    # S falls to 3/4 at t=1 (4 at risk, 1 event), then 3/4 * 1/2 at t=3.
    assert np.allclose(km['survival'], [1.0, 0.75, 0.375]), km['survival']
    assert list(km['at_risk']) == [4, 4, 2]
    note('Kaplan-Meier on a hand-computable example',
         np.round(km['survival'], 4).tolist())
    km_all = kaplan_meier([5.0] * 6, [0] * 6)
    assert km_all['survival'].tolist() == [1.0] and km_all['median'] is None
    note('all censored -> flat curve at 1, no imputed median', 'ok')
    km_late = kaplan_meier([3.0, 4.0, 5.0], [1, 1, 0], entry=[2.0, 0.0, 0.0])
    assert km_late['delayed_entry'] and list(km_late['at_risk']) == [2, 3, 2]
    note('delayed entry keeps a late-entering run out of the early risk set',
         list(km_late['at_risk']))
    lr = logrank({'a': ([1.0, 2.0, 3.0], [1, 1, 0]),
                  'b': ([4.0, 5.0, 6.0], [1, 0, 0])})
    assert lr['usable'] is False and '>= 3 events' in lr['reason']
    note('log-rank refuses below 3 events per arm', 'refusal dict returned')
    lr2 = logrank({'a': ([1.0, 2.0, 3.0, 4.0, 9.0], [1, 1, 1, 1, 0]),
                   'b': ([5.0, 6.0, 7.0, 8.0, 9.0], [1, 1, 1, 1, 0])})
    assert lr2['usable'] is True and 0.0 < lr2['p'] <= 1.0
    note('log-rank runs when both arms have >= 3 events',
         f"chi2={lr2['chi2']:.3f} p={lr2['p']:.4f}")

    if verbose:
        print('\nOrdered alternative')
    jt = jonckheere_terpstra([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0],
                              [7.0, 8.0, 9.0]], n_perm=2_000, n_boot=300)
    assert jt['J'] == 27.0 and jt['direction'] == 'increasing'
    assert jt['z'] > 2.0 and jt['p_perm'] < 0.05
    assert 'orientation only' in jt['p_role']
    note('perfectly increasing 3x3 -> J = 27 (the maximum)',
         f"z={jt['z']:.3f} p_perm={jt['p_perm']:.4f}")
    jt_flat = jonckheere_terpstra([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0],
                                   [1.0, 2.0, 3.0]], n_perm=500, n_boot=200)
    assert abs(jt_flat['J'] - jt_flat['expected']) < 1e-9
    note('three identical groups -> J equals its null mean', 'ok')
    assert jonckheere_terpstra([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])['refused']
    note('two levels is not an ordered alternative, and is refused', 'ok')

    if verbose:
        print('\nMinimum detectable effect (ANALYSIS_PLAN §6.2, simulated)')
    for a, expect, label in ((ALPHA, 1.00, 'paired sign-flip, alpha=0.05'),
                             (HOLM_STRICTEST_ALPHA, 1.54,
                              'paired sign-flip, alpha=0.00625')):
        got = mde_signflip(10, a)
        assert abs(float(got) - expect) <= 0.1, (label, float(got), expect)
        note(f'{label} -> {expect:.2f} sigma',
             f'{float(got):.3f} (power {got.power_achieved:.3f})')
    for a, expect, label in ((ALPHA, 1.39, 'Mann-Whitney, alpha=0.05'),
                             (HOLM_STRICTEST_ALPHA, 1.87,
                              'Mann-Whitney, alpha=0.00625')):
        got = mde_mann_whitney(10, a)
        assert abs(float(got) - expect) <= 0.1, (label, float(got), expect)
        note(f'{label} -> {expect:.2f} sigma',
             f'{float(got):.3f} (power {got.power_achieved:.3f})')

    if verbose:
        print('\nEquivalence and exclusion')
    eq = equivalence_from_ci(-0.02, 0.03)
    assert eq['verdict'] == 'equivalent'
    note('CI inside +/-0.05 -> equivalent', eq['verdict'])
    eq2 = equivalence_from_ci(-0.02, 0.03, sd=0.369)
    assert eq2['verdict'] == 'untestable'
    assert 'exceeds the margin' in eq2['reason']
    note('same CI, the noisy dueling cell (SD 0.369) -> untestable',
         eq2['verdict'])
    eq3 = equivalence_from_ci(-0.30, -0.10)
    assert eq3['verdict'] == 'not_equivalent'
    assert abs(eq3['worst_degradation_excluded'] - 0.30) < 1e-12
    assert abs(eq3['exclusion_bound'] + 0.10) < 1e-12
    note('CI entirely below -margin -> not_equivalent', eq3['sentence'])
    eq4 = equivalence_from_ci(-0.40, 0.05)
    assert eq4['verdict'] == 'inconclusive'
    assert abs(eq4['worst_degradation_excluded'] - 0.40) < 1e-12
    note('inconclusive still yields the exclusion sentence', eq4['sentence'])
    assert equivalence_from_ci(np.nan, np.nan)['verdict'] == 'refused'
    note('a refused interval excludes nothing, and says so', 'ok')

    if verbose:
        print(f'\n{len(checks)} checks passed.')
    return {'checks': len(checks), 'detail': checks}


def main(argv: Sequence[str] | None = None) -> int:
    """Run the self-test. There is deliberately no analysis CLI here.

    This module must not be able to read a run directory: everything it knows
    arrives as an array from a caller. The only thing worth invoking directly is
    the verification of the pinned values.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--quiet', action='store_true',
                        help='assert only; print nothing but the verdict')
    args = parser.parse_args(argv)
    try:
        result = self_test(verbose=not args.quiet)
    except AssertionError as exc:
        print(f'statlib self-test FAILED: {exc}')
        return 1
    print(f'statlib self-test OK ({result["checks"]} checks)')
    return 0


__all__ = [
    'ALPHA', 'CONFIRMATORY_FAMILY_SIZE', 'HOLM_STRICTEST_ALPHA',
    'EQUIVALENCE_MARGIN', 'BOOTSTRAP_SEED', 'N_BOOT', 'THRESHOLD_LEVELS',
    'MIN_N_FOR_INFERENCE', 'PIPELINE_VALIDATION_LABEL',
    'EXACT_SIGNFLIP_MAX_N', 'MDE_SIM_REPLICATES',
    'CI', 'MDE',
    'sign_flip_test', 'signflip_min_attainable_p',
    'wilcoxon_signed_rank', 'mann_whitney', 'mwu_exact_null',
    'mwu_critical_values', 'mwu_min_attainable_p',
    'hodges_lehmann', 'relative_effect', 'brunner_munzel', 'rank_biserial',
    'within_seed_correlation',
    'bootstrap_ci', 'paired_bootstrap',
    'holm', 'benjamini_hochberg', 'holm_thresholds', 'multiplicity_ledger',
    'clopper_pearson', 'proportion_reached', 'kaplan_meier', 'logrank',
    'jonckheere_terpstra',
    'mde_signflip', 'mde_mann_whitney',
    'equivalence_from_ci', 'self_test',
]


if __name__ == '__main__':
    sys.exit(main())
