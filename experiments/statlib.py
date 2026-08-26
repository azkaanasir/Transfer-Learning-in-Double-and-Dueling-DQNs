"""Statistical primitives for the pre-registered analysis. No I/O, no data.

Nothing in this file knows what a run directory is, so every estimator here can
be self-tested against known values without a dataset present. That separation
is deliberate and it is the whole of what the module guarantees about itself.

What this module is **not**, stated because the sentence that stood here said
the opposite: it is not the single implementation of every estimator in the
pipeline. `stats.py` carries its own `sign_flip_test`, `hodges_lehmann`,
`holm_adjust`, `brunner_munzel`, `clopper_pearson` and `kaplan_meier`, and
those are the ones that produce the report; it imports statlib inside
`verify_primitives_against_statlib`, which cross-checks a subset of the
primitives to 1e-9, and for the MDE lookup. `plots.py` and `report.py` do not
import statlib at all, and `tables.py` names it only in prose. So the two
conventions below are properties of *this module*, and that cross-check is the
only thing that carries any of them into the module which computes the paper's
numbers. Where one of them matters, verify it in the module that computes the
number: `stats.py`'s input helper `_clean` is `a[np.isfinite(a)]`, which drops
non-finite per-seed values, and that is exactly what `_vector` refuses here.

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
  makes equivalence untestable at this n (`ANALYSIS_PLAN.md` §4). Read it as the
  reference implementation of that rule, not as the source of the printed
  sentence: nothing outside this file calls it, and the exclusion sentence in
  the report comes from `stats.py`'s own `phrase_exclusion_bound`.

* **The matched-seed structure the design creates was thrown away.** The
  primary test is `sign_flip_test`, an exact randomisation test on the per-seed
  paired deltas, because at n=10 pairing is worth roughly a 40 % reduction in
  the detectable effect (`ANALYSIS_PLAN.md` §6.2). `mann_whitney` is still
  reported for every contrast, because it is the test the reviewers endorsed
  and the published paper used, and comparability matters more than tidiness.

* **A floored p-value could not be distinguished from strong evidence.** The
  rank-test wrappers report the smallest two-sided p attainable *under the null
  the reported p was actually computed from* alongside the observed p, and flag
  when the observed p sits on that floor. That qualification is the fix for a
  defect this module shipped with: with ties present `mann_whitney` falls back
  to the tie-corrected asymptotic null, whose floor is a different number from
  the exact combinatorial one, and printing the exact floor beside an asymptotic
  p produced a p smaller than its own stated floor. At n=10 vs 10 with no ties
  the Mann-Whitney floor is 1.08e-5; the paired floor is 2/1024 = 0.00195.

* **A stricter criterion was documented than the code enforces.** The paired
  floor 0.00195 is not the Holm-strictest bar: at n=10 that bar is 0.00625, and
  the exact sign-flip p moves in steps of 2/1024, so 0.00195, 0.00391 and
  0.00586 all clear it. All ten deltas sharing a sign is therefore *sufficient*
  for a confirmatory cell but not *necessary*: a 9-of-10 pattern whose dissenter
  is small enough clears the bar too. `ANALYSIS_PLAN.md` §2.2 stated the
  criterion as an "if and only if" until 2026-08-26 and now states it as "at
  most 6 of the 1024 sign assignments at least as extreme", with unanimity named
  as sufficient and not necessary (logged in §11), which is what
  `signflip_attainable_p_below` computes and why `sign_flip_test` reports
  `all_same_sign` as an observed fact rather than as the gate.

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
   and `.power_achieved` for the same reason. The three bare-float estimators
   (`hodges_lehmann`, `relative_effect`, `rank_biserial`) return an `Estimate`,
   a plain `float` carrying `.n1`, `.n2` and `.below_inference_floor`, so a
   theta of 1.0 computed from one run per arm cannot be read as "every transfer
   run beat every scratch run". `equivalence_from_ci` accepts the `CI` object
   itself, not two loose floats, so the interval's provenance, `.alpha`
   included, reaches the sentence rather than being asserted from a literal:
   the confidence level in an exclusion sentence is read off the interval that
   produced it. Handed two floats it has no provenance to read, so it says the
   level was assumed rather than observed.
2. **Missing values are refused, not dropped.** `ANALYSIS_PLAN.md` §8 forbids
   dropping a seed for any reason once it has run, and the published study
   dropped one silently. A non-finite entry in a vector handed to any function
   here raises, so the caller has to decide -- in code, visibly -- what a
   missing run means. A missing seed does *not* arrive as a non-finite entry
   though: it arrives as a shorter vector, which no arithmetic can detect. The
   paired functions therefore refuse a length mismatch outright, the unpaired
   ones (where unequal arms are legitimate) report `unequal_n` with a note, and
   every function that can take an optional `seed_ids` refuses a duplicated
   identifier, because a duplicated seed inflates n and can manufacture a
   confirmatory p-value out of three distinct runs. One case is not an input
   and so cannot be refused the same way: a caller-supplied `statistic` may
   return a non-finite value on a *resample*. `bootstrap_ci` drops those
   replicates, because a resample is not a seed and refusing the whole
   interval would be the wrong failure, but it records how many it dropped on
   the returned interval rather than reporting a clean BCa interval that
   quietly rests on fewer replicates than it claims.

Sample-size floor: below `MIN_N_FOR_INFERENCE` (=3) the tests and the intervals
refuse and say so, per `ANALYSIS_PLAN.md` §9. A refusal is a dict or a `CI`
carrying `reason`, not an exception, because the caller's job is to print
`PIPELINE_VALIDATION_LABEL` over that section and carry on. A second, higher
floor is reported rather than enforced: `MIN_N_FOR_CONFIRMATORY` (=10) is
`STANDING_INSTRUCTIONS.md` S4's ten-seed floor for a confirmatory claim, and
the paired and unpaired tests carry `confirmatory_eligible` so a caller cannot
promote an n=9 cell into the confirmatory family by accident. statlib cannot
tell "9 of 10 seeds present" from "9 seeds by design": only the registry knows
the declared block, so the flag says what n is and leaves that judgement where
the information lives.

The `__main__` guard runs `self_test()` and nothing else. There is no CLI for
analysis here on purpose: this module must not be able to read a run.

    python experiments/statlib.py
"""
from __future__ import annotations

import argparse
import itertools
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
# STANDING_INSTRUCTIONS S4: "Ten seeds is the floor for confirmatory claims."
# Reported, not enforced, because statlib cannot see the seed block: see the
# module docstring's note on `confirmatory_eligible`.
MIN_N_FOR_CONFIRMATORY = 10
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
# Above that cap the power simulation draws this many sign assignments per
# simulated dataset. It is a *planning* approximation to a null the test itself
# enumerates exactly up to n=20, so the n=16..20 MDEs are approximate in a way
# the n<=15 ones are not, and `MDE.method` says so.
SIGNFLIP_MDE_N_PERM = 4_096

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

    `alpha` is the level the interval was actually built at, and it is
    load-bearing rather than decorative: `equivalence_from_ci` narrates its
    exclusion sentence at whatever level this field says, so a 90 % interval
    cannot reach the paper labelled as a 95 % one.

    `replicates_dropped` and `n_replicates` are the bootstrap's own
    bookkeeping. A caller-supplied `statistic` can be non-finite on a resample,
    which `bootstrap_ci` drops because a resample is not a seed, and these two
    fields are where that is recorded, so an interval resting on fewer
    replicates than were asked for is not indistinguishable from one that is
    not.

    Does **not** carry a p-value and never will: most intervals in this study
    are attached to estimation-only analyses, which carry no p-value at all.
    """

    method: str
    fallback: str | None
    alpha: float
    n: int
    reason: str | None
    replicates_dropped: int
    n_replicates: int

    def __new__(cls, lo: float, hi: float, method: str = 'percentile',
                fallback: str | None = None, alpha: float = ALPHA,
                n: int = 0, reason: str | None = None,
                replicates_dropped: int = 0, n_replicates: int = 0) -> 'CI':
        obj = super().__new__(cls, (float(lo), float(hi)))
        obj.method = method
        obj.fallback = fallback
        obj.alpha = float(alpha)
        obj.n = int(n)
        obj.reason = reason
        obj.replicates_dropped = int(replicates_dropped)
        obj.n_replicates = int(n_replicates)
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
        if self.replicates_dropped:
            tag += f', {self.replicates_dropped} replicates dropped'
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


class Estimate(float):
    """A point estimate that carries the sample sizes it was computed from.

    A plain `float` for every arithmetic purpose, so it can still be handed to
    `paired_bootstrap` as a `statistic` and jackknifed inside `bootstrap_ci`.
    The attributes exist because the bare-float estimators were the one place
    in this module where the n<3 floor leaked: `relative_effect([a], [b])`
    returns 1.0 from one run per arm, which reads as "every `a` run beats every
    `b` run" and is indistinguishable in the output from the same number at
    n=10. `ANALYSIS_PLAN.md` §9 emits no test and no interval below n=3; a point
    estimate is neither, so it is still computed, but it is no longer silent
    about the n it came from.

    `unequal_n` is set when the two arms differ in length. That is legitimate
    for an unpaired estimator and is not refused, but where the arms are meant
    to be seed-matched it means a run is missing, which `ANALYSIS_PLAN.md` §8
    forbids resolving silently.

    `complete_separation` is set by the ordinal estimators when no cross pair
    contradicts the ordering, i.e. theta is exactly 0 or exactly 1. That is a
    real observation, not an error, but it is the one value at which the
    bootstrap has nothing left to resample: every resample is separated too, so
    the interval collapses to a point. `brunner_munzel` refuses the interval in
    that case, and this flag is how the point estimate carries the same news.
    """

    n1: int
    n2: int
    below_inference_floor: bool
    unequal_n: bool
    complete_separation: bool
    reason: str | None

    def __new__(cls, value: float, n1: int = 0, n2: int = 0,
                reason: str | None = None,
                complete_separation: bool = False) -> 'Estimate':
        obj = super().__new__(cls, float(value))
        obj.n1 = int(n1)
        obj.n2 = int(n2)
        sizes = [s for s in (n1, n2) if s]
        obj.below_inference_floor = bool(
            sizes and min(sizes) < MIN_N_FOR_INFERENCE)
        obj.unequal_n = bool(n1 and n2 and n1 != n2)
        obj.complete_separation = bool(complete_separation)
        if reason is None and obj.below_inference_floor:
            reason = _too_few(min(sizes))
        elif reason is None and obj.unequal_n:
            reason = _unequal_arms(int(n1), int(n2))
        elif reason is None and obj.complete_separation:
            reason = ('the arms are completely separated, so no bootstrap '
                      'resample can contradict the ordering and no interval '
                      'on this estimate quantifies any uncertainty')
        obj.reason = reason
        return obj

    def __repr__(self) -> str:
        sizes = f'n1={self.n1}, n2={self.n2}' if self.n2 else f'n={self.n1}'
        tag = ', below the n<3 floor' if self.below_inference_floor else ''
        tag += ', unequal arms' if self.unequal_n else ''
        tag += ', complete separation' if self.complete_separation else ''
        return f'Estimate({float(self):.6g}, {sizes}{tag})'


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


def _unequal_arms(n1: int, n2: int) -> str:
    """The note attached whenever two arms differ in length.

    Not a refusal: an unpaired estimator is defined for unequal arms and the
    docstrings say so. But where the arms are the two conditions of one cell
    they are seed-matched by construction (`DESIGN.md` §8.1), so a difference in
    length means a run is missing, and reconciling that here would hide a
    dropped seed (`ANALYSIS_PLAN.md` §8).
    """
    return (f'the two arms differ in length ({n1} vs {n2}). That is legitimate '
            f'for an unpaired comparison; if these arms are meant to be '
            f'seed-matched it means a run is missing, which may not be '
            f'reconciled silently (ANALYSIS_PLAN.md §8).')


def _confirmatory_eligibility(n: int) -> tuple[bool, str | None]:
    """Whether `n` clears `STANDING_INSTRUCTIONS.md` S4's ten-seed floor.

    Reported, never enforced. statlib sees a vector of numbers, not a seed
    block, so it cannot tell an arm that lost a seed from an arm that was
    designed smaller. Naming the shortfall is what it can honestly do.
    """
    if n >= MIN_N_FOR_CONFIRMATORY:
        return True, None
    return False, (
        f'n={n} < {MIN_N_FOR_CONFIRMATORY}: STANDING_INSTRUCTIONS.md S4 sets '
        f'ten seeds as the floor for a confirmatory claim, so this result may '
        f'be reported as an estimate but may not enter the confirmatory family '
        f'of ANALYSIS_PLAN.md §2. statlib cannot tell a lost seed from a '
        f'smaller design: the caller holds the seed block.')


def _duplicate_seed_note(seed_ids: Sequence[Any] | None,
                         n: int, what: str) -> None:
    """Raise when `seed_ids` repeats an identifier or mis-counts the vector.

    A duplicated seed inflates n silently and can manufacture a confirmatory
    result: three distinct deltas repeated to n=10 give the exact sign-flip
    test p = 0.00195, the floor, and a Holm-clearing cell. No function here can
    see that from the numbers alone, because repeated values are not otherwise
    illegal, so the identifiers have to arrive with the data.
    """
    if seed_ids is None:
        return
    ids = list(seed_ids)
    if len(ids) != n:
        raise ValueError(f'seed_ids has {len(ids)} entries for {n} values; '
                         f'they must correspond one to one.')
    seen: dict[Any, int] = {}
    for s in ids:
        seen[s] = seen.get(s, 0) + 1
    dupes = sorted((str(k) for k, c in seen.items() if c > 1))
    if dupes:
        raise ValueError(
            f'seed_ids repeats {dupes}: {len(ids)} values carry only '
            f'{len(seen)} distinct seeds. A duplicated seed inflates n and can '
            f'manufacture a confirmatory p-value from a handful of runs; '
            f'de-duplicate in the caller, where it is visible, before calling '
            f'{what}.')


def _repeated_value_flags(x: np.ndarray) -> dict[str, Any]:
    """Report exact duplicate values, which are a duplicated-seed smell.

    Two independent runs producing bit-identical normalised scores is close to
    impossible, so `n_distinct < n` in a real per-seed vector almost always
    means the same run entered twice. This is a reported flag and not a
    refusal, because a legitimately tied vector (an all-censored arm, a
    floored score) does exist and refusing it would be the wrong failure.
    """
    n = int(x.size)
    distinct = int(np.unique(x).size) if n else 0
    return {'n_distinct_values': distinct,
            'repeated_values': bool(n and distinct < n),
            'repeated_values_note': None if (not n or distinct == n) else (
                f'{n} values carry only {distinct} distinct numbers. Two '
                f'independent runs rarely tie exactly: check that no seed '
                f'entered twice, because a duplicated seed inflates n and can '
                f'manufacture a p-value at the floor. Pass seed_ids to have '
                f'that refused rather than flagged.')}


# ---------------------------------------------------------------------------
# 1. The primary test: exact sign-flip randomisation on paired deltas
# ---------------------------------------------------------------------------
def signflip_min_attainable_p(n: int) -> float:
    """Smallest two-sided p the exact sign-flip test can return at this n.

    `2 / 2**n`, attained exactly when every delta shares a sign. At n=10 this is
    0.001953. Assumes the enumeration is exact; on the Monte Carlo branch the
    floor is `1/(n_perm+1)` instead.

    This is the floor, **not** the confirmatory bar. The bar at n=10 is Holm's
    strictest step, 0.00625, and the exact p moves in units of 2/1024, so
    0.00195, 0.00391 and 0.00586 all clear it: see
    `signflip_attainable_p_below`. All ten deltas sharing a sign is sufficient
    for a confirmatory cell, not necessary, and since 2026-08-26
    `ANALYSIS_PLAN.md` §2.2 says so too: it stated the criterion as an "if and
    only if", and the plan was corrected to the attainable bar rather than the
    code being bent to the sentence.
    """
    if n <= 0:
        return float('nan')
    return float(2.0 ** (1 - n))


def signflip_attainable_p_below(n: int,
                                alpha: float = HOLM_STRICTEST_ALPHA
                                ) -> tuple[float, ...]:
    """Every exact sign-flip p at or below `alpha`, ascending. Possibly empty.

    The exact two-sided p is `k / 2**n` for an even count `k` of at-least-as-
    extreme sign assignments (the assignment and its negation always come as a
    pair), so the attainable values are `2/2**n, 4/2**n, ...`. At n=10 against
    the Holm-strictest 0.00625 this returns (0.00195, 0.00391, 0.00586): three
    distinct outcomes clear the bar, so the transparent statement of the
    criterion is "at most 6 of the 1024 sign assignments may be at least as
    extreme as the observed one", not "all ten seeds must agree".

    Empty when no attainable p reaches `alpha`, which at small n is the
    finding, not an error: at n=6 the floor is 0.03125, so no cell can clear
    0.00625 whatever the data do.
    """
    if n <= 0 or not np.isfinite(alpha) or alpha < 0:
        return ()
    total = 2.0 ** n
    k_max = int(math.floor(alpha * total + 1e-12))
    return tuple(float(k / total) for k in range(2, k_max + 1, 2))


def _attainable_below_mc(n_perm: int,
                         alpha: float = HOLM_STRICTEST_ALPHA
                         ) -> tuple[int, float]:
    """`signflip_attainable_p_below`'s analogue on the Monte Carlo branch.

    There the add-one estimator makes the attainable two-sided p-values
    `k / (n_perm + 1)` for `k = 1 .. n_perm + 1`, a lattice of step
    `1/(n_perm+1)` rather than the exact branch's `2/2**n`. Returns the count at
    or below `alpha` and the largest of them. Reporting the exact branch's empty
    tuple on this branch said "no attainable p clears the bar", which is false:
    at n_perm=100,000 the floor is 1e-5 and 625 attainable values clear 0.00625.
    """
    total = int(n_perm) + 1
    if total <= 1 or not np.isfinite(alpha) or alpha < 0:
        return 0, float('nan')
    k = min(int(math.floor(float(alpha) * total + 1e-12)), total)
    return (k, float(k) / float(total)) if k >= 1 else (0, float('nan'))


def _signflip_null_exact(deltas: np.ndarray) -> np.ndarray:
    """All 2**n values of `sum(s_i * d_i)`, built by successive doubling."""
    totals = np.zeros(1, dtype=float)
    for d in deltas:
        totals = np.concatenate((totals + d, totals - d))
    return totals


def sign_flip_test(deltas: Sequence[float] | np.ndarray,
                   n_perm: int | None = None,
                   seed: int = BOOTSTRAP_SEED,
                   seed_ids: Sequence[Any] | None = None) -> dict:
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

    `seed_ids`, when given, must be the per-delta seed identifiers, and a
    repeated one raises. That guard exists because a duplicated seed is
    invisible in the numbers and is not harmless: the three distinct demo
    deltas repeated to n=10 return p = 0.00195, the floor, and a Holm-clearing
    cell. Without `seed_ids` the output still carries `repeated_values` as a
    flag. `all_same_sign` is reported as an observed fact and is **not** the
    confirmatory gate: see `signflip_attainable_p_below`.
    """
    d = _vector(deltas, 'deltas')
    n = int(d.size)
    _duplicate_seed_note(seed_ids, n, 'sign_flip_test')
    eligible, eligibility_note = _confirmatory_eligibility(n)
    dup = _repeated_value_flags(d)
    pos = int(np.count_nonzero(d > 0))
    neg = int(np.count_nonzero(d < 0))
    exact_clearing = signflip_attainable_p_below(n, HOLM_STRICTEST_ALPHA)
    if n < MIN_N_FOR_INFERENCE:
        # Every key the computed path returns is present here too, carrying
        # the honest value at this n. A refusal that is not key-compatible
        # with the result it stands in for turns "print the refusal and carry
        # on" into a KeyError in the caller, which is the opposite of the
        # reason this function returns a dict instead of raising.
        return {'statistic': float(d.mean()) if n else float('nan'),
                'p_two_sided': float('nan'), 'n': n, 'exact': False,
                'n_perm': 0, 'p_min_attainable': float('nan'),
                'at_p_floor': False,
                'n_positive': pos, 'n_negative': neg,
                'n_zero': n - pos - neg,
                'all_same_sign': bool(n and (pos == n or neg == n)),
                'n_p_values_clearing_holm_strictest': len(exact_clearing),
                'max_p_clearing_holm_strictest':
                    exact_clearing[-1] if exact_clearing else float('nan'),
                'p_values_clearing_holm_strictest_null': 'exact enumeration',
                'confirmatory_eligible': False,
                'confirmatory_eligibility_note': eligibility_note,
                **dup,
                'refused': True, 'reason': _too_few(n)}

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

    # The attainable set has to come from the null the p came from. On the
    # Monte Carlo branch the exact enumeration is not in use, and reporting its
    # empty tuple there read as "no attainable p clears 0.00625" when that
    # branch's own floor is 1/(n_perm+1).
    if exact:
        n_clearing = len(exact_clearing)
        max_clearing = (exact_clearing[-1] if exact_clearing
                        else float('nan'))
        clearing_null = 'exact enumeration'
    else:
        n_clearing, max_clearing = _attainable_below_mc(
            n_used, HOLM_STRICTEST_ALPHA)
        clearing_null = f'Monte Carlo lattice of step 1/{n_used + 1}'
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
            # Reported so nothing downstream can treat `all_same_sign` as the
            # confirmatory gate: several attainable p-values clear the bar, and
            # the largest of them is the one that shows how much slack there
            # is. At n=10 that is 6/1024 = 0.00586 against a bar of 0.00625.
            'n_p_values_clearing_holm_strictest': n_clearing,
            'max_p_clearing_holm_strictest': float(max_clearing),
            'p_values_clearing_holm_strictest_null': clearing_null,
            'confirmatory_eligible': bool(eligible),
            'confirmatory_eligibility_note': eligibility_note,
            **dup,
            'refused': False, 'reason': None}


# ---------------------------------------------------------------------------
# 2. The two rank tests reported alongside, with their p-value floors
# ---------------------------------------------------------------------------
def wilcoxon_signed_rank(deltas: Sequence[float] | np.ndarray,
                         seed_ids: Sequence[Any] | None = None) -> dict:
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

    `seed_ids` behaves as in `sign_flip_test`: a repeated identifier raises,
    because a duplicated seed inflates n invisibly.
    """
    d = _vector(deltas, 'deltas')
    n = int(d.size)
    _duplicate_seed_note(seed_ids, n, 'wilcoxon_signed_rank')
    eligible, eligibility_note = _confirmatory_eligibility(n)
    dup = _repeated_value_flags(d)
    absd_all = np.abs(d[d != 0])
    all_ties = bool(absd_all.size != np.unique(absd_all).size)
    if n < MIN_N_FOR_INFERENCE:
        return {'statistic': float('nan'), 'p': float('nan'), 'n': n,
                'p_min_attainable': float('nan'),
                'p_min_attainable_method': 'refused',
                'p_min_attainable_exact_null':
                    float(signflip_min_attainable_p(n)),
                'at_p_floor': False,
                'method': 'refused',
                'n_zero': int(np.count_nonzero(d == 0)), 'ties': all_ties,
                'confirmatory_eligible': False,
                'confirmatory_eligibility_note': eligibility_note, **dup,
                'refused': True, 'reason': _too_few(n)}

    if np.all(d == 0):
        # Every delta is exactly zero: no evidence in either direction, and
        # SciPy raises rather than returning p=1. Say so explicitly.
        return {'statistic': 0.0, 'p': 1.0, 'n': n,
                'p_min_attainable': signflip_min_attainable_p(n),
                'p_min_attainable_method': 'degenerate',
                'p_min_attainable_exact_null':
                    float(signflip_min_attainable_p(n)),
                'at_p_floor': False, 'method': 'degenerate',
                'n_zero': n, 'ties': True,
                'confirmatory_eligible': bool(eligible),
                'confirmatory_eligibility_note': eligibility_note, **dup,
                'refused': False,
                'reason': 'all deltas are exactly zero'}

    nz = int(np.count_nonzero(d == 0))
    absd = np.abs(d[d != 0])
    ties = bool(absd.size != np.unique(absd).size)
    method = 'exact' if (nz == 0 and not ties and n <= 25) else 'approx'
    res = stats.wilcoxon(d, alternative='two-sided', zero_method='zsplit',
                         method=method, correction=(method == 'approx'))
    # The floor has to belong to the null that produced the p. On the approx
    # branch the tie and zero corrections shrink the null variance, so the
    # exact combinatorial floor 2/2**n is a bound from a different null and
    # printing it beside an approximate p is how a p below its own stated floor
    # gets reported. Running the same branch on the same |deltas| all given one
    # sign gives the smallest p this tie and zero pattern can produce under the
    # method actually used.
    floor_res = stats.wilcoxon(np.abs(d), alternative='two-sided',
                               zero_method='zsplit', method=method,
                               correction=(method == 'approx'))
    floor = float(floor_res.pvalue)
    exact_floor = signflip_min_attainable_p(n)
    return {'statistic': float(res.statistic), 'p': float(res.pvalue), 'n': n,
            'p_min_attainable': floor,
            'p_min_attainable_method': method,
            'p_min_attainable_exact_null': float(exact_floor),
            'at_p_floor': bool(res.pvalue <= floor * (1.0 + 1e-9)),
            'method': method, 'n_zero': nz, 'ties': ties,
            'confirmatory_eligible': bool(eligible),
            'confirmatory_eligibility_note': eligibility_note, **dup,
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


def _mwu_tie_sd(n1: int, n2: int, pooled: np.ndarray) -> float:
    """SciPy's tie-corrected SD of the U null. Ties shrink it, which matters.

    `sqrt(n1 n2 / 12 * ((N+1) - sum(t**3 - t) / (N (N-1))))` over the tie group
    sizes `t` of the pooled sample, which is exactly the quantity
    `scipy.stats.mannwhitneyu`'s asymptotic branch uses. It is separated out
    here because the *floor* of that branch has to be computed from the same
    variance as the p-value, or the two describe different nulls.
    """
    n = n1 + n2
    if n < 2:
        return float('nan')
    _, counts = np.unique(pooled, return_counts=True)
    c = counts.astype(float)
    tie = float(np.sum(c ** 3 - c))
    var = n1 * n2 / 12.0 * ((n + 1.0) - tie / (n * (n - 1.0)))
    return math.sqrt(var) if var > 0 else 0.0


def mwu_asymptotic_p(u: float, n1: int, n2: int,
                     pooled: Sequence[float] | np.ndarray) -> float:
    """SciPy's two-sided tie-corrected normal-approximation p at `U = u`.

    Reproduces `scipy.stats.mannwhitneyu(..., method='asymptotic')` exactly
    (`self_test` checks it to machine precision over tie-heavy samples),
    including the continuity correction and the reflection onto the larger
    tail. Exposed because `mann_whitney` needs to evaluate the *same* null at
    the extreme U to state an honest floor.
    """
    arr = np.asarray(pooled, dtype=float).ravel()
    sd = _mwu_tie_sd(int(n1), int(n2), arr)
    if not np.isfinite(sd) or sd <= 0.0:
        return float('nan')
    max_u = float(n1) * float(n2)
    u_max = max(float(u), max_u - float(u))
    z = (u_max - max_u / 2.0 - 0.5) / sd
    return float(min(1.0, 2.0 * float(stats.norm.sf(z))))


def mwu_min_attainable_u(n1: int, n2: int,
                         pooled: Sequence[float] | np.ndarray) -> float:
    """The smallest `U` any split of this pooled multiset into n1, n2 can give.

    `0` without ties, and that is the only case in which complete separation is
    available: U=0 needs every `a` value strictly below every `b` value, so a
    single value shared *across* the arms forces `U >= 0.5`. Putting the n1
    smallest values in the first arm minimises U, and then the only cross pairs
    that are not strictly ordered are the ties inside the one tie group the
    boundary cuts through, each worth 0.5. Hence `0.5 * k * m` for `k` copies of
    the boundary value below the cut and `m` above it. Any other split moves a
    larger value into the first arm and a smaller one out of it, which buys a
    full 1.0 per inverted pair in place of a 0.5 per tie, so it cannot do
    better.

    Ties *within* one arm do not raise the floor: they never produce a cross
    pair. The distinction matters because the module's own tie examples (a
    capped normalised score, an all-censored `steps_to_threshold`) are usually
    shared across the arms, which is exactly the case that was mis-handled.

    Not symmetric in the two arms, which is easy to get wrong: `n1 n2 - U_min`
    is **not** the largest attainable U, because the boundary cuts a different
    tie group when the *other* arm takes the small values. Pooled
    `[0,0,1,1,1,2,2,2,3]` at n1=4, n2=5 has `U_min = 1` and yet attains the full
    `U = 20`, since `[2,2,2,3]` against `[0,0,1,1,1]` is completely separated.
    Use `mwu_max_attainable_u` for the other end rather than reflecting this
    one.
    """
    arr = np.sort(np.asarray(pooled, dtype=float).ravel())
    n1, n2 = int(n1), int(n2)
    if n1 < 1 or n2 < 1 or arr.size != n1 + n2:
        return float('nan')
    boundary = arr[n1 - 1]
    k = float(np.count_nonzero(arr[:n1] == boundary))
    m = float(np.count_nonzero(arr[n1:] == boundary))
    return 0.5 * k * m


def mwu_max_attainable_u(n1: int, n2: int,
                         pooled: Sequence[float] | np.ndarray) -> float:
    """The largest `U` any split of this pooled multiset can give.

    `U(a, b) = n1 n2 - U(b, a)` exactly, so maximising U over splits is
    minimising it with the arms exchanged: `n1 n2 - mwu_min_attainable_u(n2,
    n1, pooled)`. With ties that is not the same number as `n1 n2 - U_min`.
    """
    u = mwu_min_attainable_u(int(n2), int(n1), pooled)
    if not np.isfinite(u):
        return float('nan')
    return float(int(n1) * int(n2)) - u


def mwu_asymptotic_min_attainable_p(n1: int, n2: int,
                                    pooled: Sequence[float] | np.ndarray
                                    ) -> float:
    """The floor of the tie-corrected asymptotic null, at the most extreme
    `U` this pooled multiset can actually produce.

    Two separate corrections to the exact combinatorial floor
    `2 / C(n1+n2, n1)`, both caused by ties, and they run in opposite
    directions. First, the tie correction shrinks the null variance, so the
    normal p undercuts the combinatorial floor: at 5 vs 5 with one value shared
    the exact floor is 0.00794 while the asymptotic p at U=0 is 0.00398, which
    is how `mann_whitney` came to report a p smaller than its own stated floor
    and flag it as being *on* that floor. Second, and the reason this is not
    simply `mwu_asymptotic_p(0.0, ...)`: with a value shared across the arms,
    **U=0 is not attainable at all**, so a floor evaluated there is a bound no
    arrangement of the data can reach. `mann_whitney([1,2,3], [3,4,5])` reported
    a floor of 0.0765 when the smallest p any of the C(6,3) splits can give is
    0.1212, which is the observed p: the sample was the most extreme one
    available and `at_p_floor` still did not fire. Over 396 tie-carrying 10 vs
    10 samples a quarter reported an unreachable floor, so the flag that exists
    to stop a floored p being read as strong evidence was inert in a quarter of
    tied cells.

    The test is two-sided, so the floor is the smaller of the p at the two
    attainable extremes, `mwu_min_attainable_u` and `mwu_max_attainable_u`.
    Taking one and reflecting it is wrong under ties: the two ends are not
    mirror images, and reflecting the lower one alone put the floor *above* the
    attainable minimum by up to 10x on a brute-force check over unequal arms.
    """
    u_lo = mwu_min_attainable_u(n1, n2, pooled)
    u_hi = mwu_max_attainable_u(n1, n2, pooled)
    if not (np.isfinite(u_lo) and np.isfinite(u_hi)):
        return float('nan')
    return min(mwu_asymptotic_p(u_lo, int(n1), int(n2), pooled),
               mwu_asymptotic_p(u_hi, int(n1), int(n2), pooled))


def mwu_asymptotic_critical_values(n1: int, n2: int,
                                   pooled: Sequence[float] | np.ndarray,
                                   alpha: float = ALPHA
                                   ) -> tuple[float, float] | None:
    """The two-sided rejection region of the tie-corrected asymptotic null.

    Reject when `U <= u_lo` or `U >= u_hi`. The bounds are not integers: with
    ties U takes half-integer values, and the normal null has no lattice at
    all. Returns None when the region is empty at this `alpha`, i.e. when no
    split of the pooled multiset reaches either tail of it. A value shared
    across the arms can put complete separation out of reach, and a region only
    complete separation could enter is an empty region. Both tails are checked,
    because ties make the two attainable extremes asymmetric.
    """
    arr = np.asarray(pooled, dtype=float).ravel()
    sd = _mwu_tie_sd(int(n1), int(n2), arr)
    if not np.isfinite(sd) or sd <= 0.0:
        return None
    max_u = float(n1) * float(n2)
    u_lo = mwu_min_attainable_u(n1, n2, arr)
    u_hi = mwu_max_attainable_u(n1, n2, arr)
    if not (np.isfinite(u_lo) and np.isfinite(u_hi)):
        return None
    zc = float(stats.norm.ppf(1.0 - float(alpha) / 2.0))
    hi = max_u / 2.0 + 0.5 + zc * sd
    if hi > max_u - u_lo and hi > u_hi:
        return None
    return float(max_u - hi), float(hi)


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
                 b: Sequence[float] | np.ndarray,
                 alpha: float = ALPHA) -> dict:
    """Mann-Whitney U for `a` against `b`, with the attainable p floor.

    Reported for every contrast whatever the design, because it is the test the
    reviewers endorsed and the published paper used, so comparability with the
    published numbers is worth keeping (`ANALYSIS_PLAN.md` §2). It is not the
    primary test for a within-cell delta: the design is matched by seed, and the
    unpaired test discards that structure at a real cost in power -- MDE 1.41
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

    Ties, and why the floor and the rejection region are method-dependent. Any
    repeated value in the pooled sample -- a capped or floored normalised
    score, an all-censored `steps_to_threshold`, an all-zero AUC arm -- makes
    the exact null wrong, so SciPy's tie-corrected asymptotic branch is used
    and `method` says `asymptotic`. The tie correction shrinks the null
    variance, so the asymptotic p at complete separation is *smaller* than the
    exact combinatorial floor `2 / C(n1+n2, n1)`. Printing that exact floor
    beside an asymptotic p produced a p below its own stated floor, and killed
    `at_p_floor` at the confirmatory sample size. Both `p_min_attainable` and
    `critical_values` therefore come from whichever null produced the p, and
    `p_min_attainable_method` names it. The exact-null figures are still
    reported, under keys that say `exact_null`, because they are the
    pre-registered ones (`ANALYSIS_PLAN.md` §6.1).

    A tie *across* the arms does something further, and this is the half that
    the first fix missed. It makes complete separation unattainable: U=0 needs
    every `a` strictly below every `b`, and one shared value forces `U >= 0.5`.
    A floor evaluated at U=0 is then a bound no arrangement of the data can
    reach, so `at_p_floor` never fires however extreme the sample is
    (`mann_whitney([1,2,3], [3,4,5])` sits on the true minimum and used to
    report `at_p_floor=False`). `p_min_attainable` is now evaluated at
    `mwu_min_attainable_u`, the smallest U this pooled multiset permits, and
    `p_min_attainable_u` reports where it was evaluated. Ties confined to one
    arm leave the floor where it was, since they make no cross pair.

    `alpha` sets the reported rejection region only; it changes no p-value. The
    region at the Holm-strictest step is reported alongside it unconditionally,
    because this test's contrast belongs to a family of 8 and the uncorrected
    region is not the one a confirmatory reading is entitled to use.
    """
    x = _vector(a, 'a')
    y = _vector(b, 'b')
    n1, n2 = int(x.size), int(y.size)
    exact_floor = mwu_min_attainable_p(n1, n2)
    unequal = bool(n1 != n2)
    eligible, eligibility_note = _confirmatory_eligibility(min(n1, n2))
    if min(n1, n2) < MIN_N_FOR_INFERENCE:
        return {'U': float('nan'), 'p': float('nan'), 'n1': n1, 'n2': n2,
                'p_min_attainable': float('nan'),
                'p_min_attainable_method': 'refused',
                'p_min_attainable_note': None,
                'p_min_attainable_u': float('nan'),
                'p_min_attainable_exact_null': float(exact_floor),
                'at_p_floor': False,
                'method': 'refused', 'ties': False,
                'critical_values': None,
                'critical_values_alpha': float(alpha),
                'critical_values_method': 'refused',
                'critical_values_holm_strictest': None,
                'critical_values_holm_strictest_alpha': HOLM_STRICTEST_ALPHA,
                'unequal_n': unequal,
                'unequal_n_note': _unequal_arms(n1, n2) if unequal else None,
                'confirmatory_eligible': False,
                'confirmatory_eligibility_note': eligibility_note,
                'refused': True,
                'reason': _too_few(min(n1, n2))}

    pooled = np.concatenate((x, y))
    ties = bool(np.unique(pooled).size != pooled.size)
    exact = (not ties) and n1 * n2 <= EXACT_MWU_MAX_PRODUCT
    method = 'exact' if exact else 'asymptotic'
    res = stats.mannwhitneyu(x, y, alternative='two-sided', method=method)
    u_lo_att = mwu_min_attainable_u(n1, n2, pooled)
    u_hi_att = mwu_max_attainable_u(n1, n2, pooled)
    # Where the two-sided floor is evaluated: whichever attainable extreme is
    # further from the null mean, expressed as a U.
    u_min = (u_lo_att
             if (n1 * n2 - u_lo_att) >= u_hi_att else n1 * n2 - u_hi_att)
    if exact:
        floor = float(exact_floor)
        crit: tuple[float, float] | tuple[int, int] | None = \
            mwu_critical_values(n1, n2, alpha)
        crit_holm: tuple[float, float] | tuple[int, int] | None = \
            mwu_critical_values(n1, n2, HOLM_STRICTEST_ALPHA)
    else:
        floor = mwu_asymptotic_min_attainable_p(n1, n2, pooled)
        crit = mwu_asymptotic_critical_values(n1, n2, pooled, alpha)
        crit_holm = mwu_asymptotic_critical_values(
            n1, n2, pooled, HOLM_STRICTEST_ALPHA)
    floor_note = None
    if np.isfinite(floor) and u_min > 0.0:
        floor_note = (
            f'a value is shared across the arms, so no split of this pooled '
            f'sample separates it completely: the attainable extremes are '
            f'U={u_lo_att:g} and U={u_hi_att:g} against a range of 0 to '
            f'{n1 * n2}. The floor is evaluated at the further of the two and '
            f'is higher than the one complete separation would give')
    if not np.isfinite(floor):
        # Every pooled value identical: the tie correction takes the null
        # variance to zero, so no p is attainable at all and `at_p_floor` must
        # stay off rather than compare against a number that does not exist.
        floor_note = ('the pooled sample is constant, so the tie-corrected '
                      'null has zero variance and no p-value is attainable; '
                      'the arms are not distinguishable by rank at any n')
    return {'U': float(res.statistic), 'p': float(res.pvalue),
            'n1': n1, 'n2': n2,
            'p_min_attainable': float(floor),
            'p_min_attainable_method': method,
            'p_min_attainable_note': floor_note,
            'p_min_attainable_u': float(u_min),
            'p_min_attainable_exact_null': float(exact_floor),
            'at_p_floor': bool(np.isfinite(floor)
                               and res.pvalue <= floor * (1.0 + 1e-9)),
            'method': method, 'ties': ties,
            'critical_values': crit,
            'critical_values_alpha': float(alpha),
            'critical_values_method': method,
            'critical_values_holm_strictest': crit_holm,
            'critical_values_holm_strictest_alpha': HOLM_STRICTEST_ALPHA,
            'unequal_n': unequal,
            'unequal_n_note': _unequal_arms(n1, n2) if unequal else None,
            'confirmatory_eligible': bool(eligible),
            'confirmatory_eligibility_note': eligibility_note,
            'refused': False, 'reason': None}


# ---------------------------------------------------------------------------
# 3. Effect sizes. No Cohen's d, ever (ANALYSIS_PLAN §8).
# ---------------------------------------------------------------------------
def hodges_lehmann(a: Sequence[float] | np.ndarray,
                   b: Sequence[float] | np.ndarray | None = None) -> Estimate:
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

    Returns an `Estimate`, which is a plain `float` that also records `n1`,
    `n2` and `below_inference_floor`. It is still computed below n=3, because
    `ANALYSIS_PLAN.md` §9 withholds tests and intervals rather than point
    estimates, but the caller can now see that a shift of 0.0216 came from one
    run per arm.
    """
    x = _vector(a, 'a')
    if x.size == 0:
        return Estimate(float('nan'), 0, 0, reason='the sample is empty')
    if b is None:
        i, j = np.triu_indices(x.size, k=0)
        return Estimate(float(np.median((x[i] + x[j]) / 2.0)), int(x.size))
    y = _vector(b, 'b')
    if y.size == 0:
        return Estimate(float('nan'), int(x.size), 0,
                        reason='the second sample is empty')
    return Estimate(float(np.median(x[:, None] - y[None, :])),
                    int(x.size), int(y.size))


def relative_effect(a: Sequence[float] | np.ndarray,
                    b: Sequence[float] | np.ndarray) -> Estimate:
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

    Returns an `Estimate` carrying `n1`, `n2`, `below_inference_floor` and
    `complete_separation`, because theta is exactly the quantity whose n<3 value
    is most misleading: one run per arm gives theta = 1.0, which reads as "every
    `a` run beat every `b` run" from a single cross pair. At any n, theta of
    exactly 0 or exactly 1 sets `complete_separation`, which is what makes the
    bootstrap interval on it degenerate.
    """
    x = _vector(a, 'a')
    y = _vector(b, 'b')
    if x.size == 0 or y.size == 0:
        return Estimate(float('nan'), int(x.size), int(y.size),
                        reason='an arm is empty')
    gt = float(np.count_nonzero(x[:, None] > y[None, :]))
    eq = float(np.count_nonzero(x[:, None] == y[None, :]))
    theta = (gt + 0.5 * eq) / float(x.size * y.size)
    return Estimate(theta, int(x.size), int(y.size),
                    complete_separation=theta in (0.0, 1.0))


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

    **A zero-width interval is refused, not returned.** Under complete
    separation every bootstrap resample is separated too, so every `theta*` is
    the same number and both percentiles land on it: the function used to return
    `theta=1.0, ci=(1.0, 1.0), refused=False, reason=None`, an interval that
    asserts zero sampling uncertainty from 10 runs against 10. That is the same
    shape `equivalence_from_ci` refuses with "an interval that quantifies no
    uncertainty licenses no exclusion", and the estimator that manufactures it
    now refuses it at the source, because `ANALYSIS_PLAN.md` §3 licenses a
    directional claim from what the interval excludes and this one excludes
    everything. `theta`, `se` and `degenerate_interval_value` are still
    returned, so the point estimate and the collapsed value are not lost;
    what is withheld is the interval. Note that the separation itself is not
    withheld evidence: the p-value for it is `mann_whitney`'s, which is exact
    and is 1.08e-5 at 10 against 10.

    **No p-value is returned, by design.** `scipy.stats.brunnermunzel` will give
    one; this function deliberately does not, because every analysis that uses
    theta in this study is estimation-only and carries no p-value at all
    (`ANALYSIS_PLAN.md` §3, §7). The licensed directional statement is what the
    interval excludes.

    Unequal arms are permitted (the estimator is defined for them) and reported
    in `unequal_n`, because where the two arms are the two conditions of one
    cell they are seed-matched by construction and a length difference means a
    run is missing (`ANALYSIS_PLAN.md` §8).

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
        'complete_separation': bool(theta in (0.0, 1.0)),
        'degenerate_interval': False,
        'degenerate_interval_value': float('nan'),
        'unequal_n': bool(n1 != n2),
        'unequal_n_note': _unequal_arms(n1, n2) if n1 != n2 else None,
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
    lo_c = float(min(max(lo, 0.0), 1.0))
    hi_c = float(min(max(hi, 0.0), 1.0))
    if hi_c <= lo_c:
        # Zero width. Complete separation is the way in: every resample is
        # separated too, so every theta* equals theta and both percentiles land
        # on it. Reporting it as an interval would license "the interval lies
        # wholly above 0.5, so the ordering is not a sampling artefact" off a
        # bootstrap that never varied.
        out['degenerate_interval'] = True
        out['degenerate_interval_value'] = lo_c
        out['method'] = 'refused'
        out['refused'] = True
        out['reason'] = (
            f'the bootstrap interval collapsed to the single point '
            f'{lo_c:.6g}: every one of the {int(n_boot)} resamples gave the '
            f'same theta'
            + (', because the arms are completely separated'
               if out['complete_separation'] else '')
            + '. An interval of zero width quantifies no uncertainty and '
              'licenses no exclusion, so it is refused rather than reported; '
              'the point estimate and the collapsed value are above, and the '
              'test for the separation is mann_whitney.')
        out['ci'] = _refused_ci(out['reason'], alpha, min(n1, n2))
        return out
    out['ci_lo'] = lo_c
    out['ci_hi'] = hi_c
    out['ci'] = CI(out['ci_lo'], out['ci_hi'], method=out['method'],
                   fallback=out['fallback'], alpha=alpha, n=min(n1, n2))
    out['refused'] = False
    out['reason'] = None
    return out


def rank_biserial(a: Sequence[float] | np.ndarray,
                  b: Sequence[float] | np.ndarray) -> Estimate:
    """Rank-biserial correlation from U: `2U/(n1 n2) - 1`, i.e. `2 theta - 1`.

    Runs from -1 (every `b` beats every `a`) through 0 (no ordering) to +1. A
    monotone re-scaling of the relative effect, reported because it is the
    conventional companion to Mann-Whitney U and because it is *not* Cohen's d:
    it involves no variance estimate and no normality assumption, which is what
    `ANALYSIS_PLAN.md` §8 requires.

    Assumes nothing beyond the two samples being comparable on an ordinal scale.
    Does **not** express the effect in score units, and does not become small
    merely because the difference is substantively small. Carries the same
    `Estimate` provenance as `relative_effect`, from which it is derived.
    """
    theta = relative_effect(a, b)
    return Estimate(2.0 * float(theta) - 1.0, theta.n1, theta.n2,
                    reason=theta.reason,
                    complete_separation=theta.complete_separation)


def within_seed_correlation(a: Sequence[float] | np.ndarray,
                            b: Sequence[float] | np.ndarray,
                            seed_ids: Sequence[Any] | None = None) -> dict:
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
    must guarantee that ordering, and a length mismatch raises. Pass `seed_ids`
    to have a duplicated seed raise as well.
    """
    x = _vector(a, 'a')
    y = _vector(b, 'b')
    if x.size != y.size:
        raise ValueError(f'seed-aligned vectors must match in length: '
                         f'{x.size} vs {y.size}')
    n = int(x.size)
    _duplicate_seed_note(seed_ids, n, 'within_seed_correlation')
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

    A caller-supplied `statistic` may be non-finite on a resample: a ratio whose
    denominator straddles zero, for instance. Those replicates are dropped,
    because a resample is not a seed and refusing the whole interval would be
    the wrong failure, but the count reaches the caller. `.replicates_dropped`
    and `.n_replicates` carry it and `.fallback` states it in words, so a `bca`
    label on an interval computed from fewer replicates than were requested is
    no longer indistinguishable from one computed from all of them. Only when
    *every* replicate is non-finite does the interval refuse outright.

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
    # A non-finite *input* is refused by `_vector`; a non-finite *replicate* is
    # not an input and cannot be refused the same way, so it is dropped. What
    # it must not be is silent. `z0` below divides by `reps.size`, and every
    # percentile is taken over the survivors, so a dropped replicate moves the
    # interval, and an interval resting on fewer replicates than were asked for
    # used to be indistinguishable in the output from one resting on all of
    # them: same `method='bca'`, same `fallback=None`.
    n_requested = int(reps.size)
    reps = reps[np.isfinite(reps)]
    dropped = n_requested - int(reps.size)
    drop_note = (f'{dropped} of {n_requested} bootstrap replicates were '
                 f'non-finite and were dropped, so the interval rests on '
                 f'{int(reps.size)}') if dropped else None
    if reps.size == 0:
        return _refused_ci(
            f'every one of the {n_requested} bootstrap replicates was '
            f'non-finite', alpha, n)

    def _mark(fallback: str | None) -> str | None:
        if drop_note is None:
            return fallback
        return drop_note if fallback is None else f'{fallback}; {drop_note}'

    def _ci(lo_v: float, hi_v: float, meth: str,
            fallback: str | None = None) -> CI:
        return CI(lo_v, hi_v, method=meth, fallback=_mark(fallback),
                  alpha=alpha, n=n, replicates_dropped=dropped,
                  n_replicates=int(reps.size))

    lo_q, hi_q = 100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)
    pct = (float(np.percentile(reps, lo_q)), float(np.percentile(reps, hi_q)))
    if method == 'percentile':
        return _ci(pct[0], pct[1], 'percentile')

    def _fallback(reason: str) -> CI:
        return _ci(pct[0], pct[1], 'percentile', reason)

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
    return _ci(bounds[0], bounds[1], 'bca')


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
                     statistic: Callable = np.mean,
                     seed_ids: Sequence[Any] | None = None) -> dict:
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

    `seed_ids`, when given, is the shared seed vector the arms are aligned on,
    and a repeated identifier raises: resampling a duplicated seed would treat
    one run as two independent units and narrow every interval here.

    An empty `contrasts` mapping returns a refusal dict rather than raising
    from inside numpy, matching every other empty-input path in this module.
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
    _duplicate_seed_note(seed_ids, n, 'paired_bootstrap')

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
    if not names:
        out['refused'] = True
        out['reason'] = (
            'no contrasts were given. A joint seed bootstrap over an empty '
            'contrast set has nothing to estimate; name the contrasts in the '
            'caller, where the design fixes them (DESIGN.md §4).')
        out['correlation'] = np.zeros((0, 0), dtype=float)
        return out
    if n < MIN_N_FOR_INFERENCE:
        out['refused'] = True
        out['reason'] = _too_few(n)
        for name in names:
            out['estimate'][name] = (
                float(_apply_statistic(statistic, per_seed[name]))
                if name in per_seed else float('nan'))
            out['ci'][name] = _refused_ci(_too_few(n), alpha, n)
            # Same container types as the computed path: an empty array of
            # replicates, and a correlation matrix of the right shape holding
            # nothing. `correlation` used to be None here, so a caller reading
            # `.shape` raised AttributeError on exactly the n<3 data the tree
            # holds today, instead of printing the refusal and carrying on.
            out['distribution'][name] = np.empty(0, dtype=float)
        out['correlation'] = np.full((len(names), len(names)), np.nan)
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
    """Holm-Bonferroni adjusted p-values over the family as given, in order.

    The general procedure, taking its family size from `len(pvalues)`. **It is
    not the confirmatory entry point**: use `holm_confirmatory`, which requires
    all `CONFIRMATORY_FAMILY_SIZE` (=8) members and is what `ANALYSIS_PLAN.md`
    §7 means by "stats.py reads the family definitions from here rather than
    accepting them as arguments". Correcting one endpoint's 4 cells instead of
    the pre-registered 8 is the family-of-one rescue §7 exists to block, and
    this function cannot tell the difference: `holm([0.01] * 4)` adjusts to
    0.04 and rejects, while the same four p-values inside the real family of 8
    adjust to 0.08 and do not.

    Step down over the sorted p-values with multipliers `m, m-1, ..., 1`, then
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


def holm_confirmatory(pvalues: Sequence[float] | np.ndarray,
                      labels: Sequence[str] | None = None) -> np.ndarray:
    """Holm over the pre-registered confirmatory family. Refuses any other size.

    The one entry point a confirmatory verdict may be read from. It takes the
    family size from `CONFIRMATORY_FAMILY_SIZE`, not from the argument, so a
    caller cannot correct four cells on one endpoint and call the result
    family-wise controlled: `ANALYSIS_PLAN.md` §7 fixes the family at 8 (4
    cells x 2 co-primary endpoints) before launch, and shrinking it is how a
    marginal result gets rescued into a family of one.

    A member that could not be computed is passed as NaN and **still counts**,
    which is the whole point: a cell refused at n<3 does not make the family
    smaller. So the caller must supply all 8 slots, filling absent ones with
    NaN, and `labels` (when given) is checked for the same length so a
    mis-assembled family fails loudly rather than shifting every adjustment by
    one position.
    """
    p = _pvalue_array(pvalues)
    if p.size != CONFIRMATORY_FAMILY_SIZE:
        raise ValueError(
            f'the confirmatory family is {CONFIRMATORY_FAMILY_SIZE} tests '
            f'(4 cells x 2 co-primary endpoints, ANALYSIS_PLAN.md §2 and §7), '
            f'and {p.size} p-values were given. A member that could not be '
            f'computed is passed as NaN and still counts towards m; the family '
            f'is fixed before launch and may not be resized after seeing the '
            f'data.')
    if labels is not None and len(list(labels)) != CONFIRMATORY_FAMILY_SIZE:
        raise ValueError(
            f'{len(list(labels))} labels for {CONFIRMATORY_FAMILY_SIZE} family '
            f'members; the two must correspond one to one or the adjustment is '
            f'attributed to the wrong cell.')
    return holm(p)


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
    paired test's floor is 0.00195, so the bar is attainable. Three attainable
    p-values clear it, not one: see `signflip_attainable_p_below`. All ten
    deltas sharing a sign is sufficient, not necessary, which is what
    `ANALYSIS_PLAN.md` §2.2 has said since its 2026-08-26 correction.
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

    **A departure from `ANALYSIS_PLAN.md` §9, made visible rather than
    silent.** §9 says no interval is emitted below n=3, and this one is: it is
    exact at any n, and at n=1 the honest interval is very nearly [0, 1], which
    is an informative statement rather than a claim. Below the floor the
    returned `CI` therefore carries a `reason` recording the departure and the
    n it came from, so a caller can stamp `PIPELINE_VALIDATION_LABEL` over it
    instead of discovering the exception in a docstring. §11 of the plan does
    not currently record this departure; that is a plan-side gap, not a code
    one, and it is not resolvable from inside this module.
    """
    k, n = int(k), int(n)
    if n < 1 or k < 0 or k > n:
        return _refused_ci(f'invalid counts k={k}, n={n}', alpha, n)
    lo = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1.0 - alpha / 2.0,
                                                 k + 1, n - k))
    reason = None
    if n < MIN_N_FOR_INFERENCE:
        reason = (f'n={n} < {MIN_N_FOR_INFERENCE}. This interval is exact at '
                  f'any n and is emitted deliberately, which is a departure '
                  f'from ANALYSIS_PLAN.md §9 ("no test and no interval"). '
                  f'Stamp {PIPELINE_VALIDATION_LABEL!r} over it: at this n it '
                  f'is very nearly [0, 1] and settles nothing.')
    return CI(lo, hi, method='clopper-pearson', alpha=alpha, n=n,
              reason=reason)


def proportion_reached(events: Sequence[int] | np.ndarray,
                       alpha: float = ALPHA) -> dict:
    """`k/n` reached, with an exact interval. The primary censored summary.

    `events` is 1 where the threshold was reached within the budget and 0 where
    the run was censored at it. Assumes administrative censoring at a common
    budget. Does **not** use the event times: pair it with `kaplan_meier`.

    `below_inference_floor` carries `clopper_pearson`'s deliberate departure
    from `ANALYSIS_PLAN.md` §9 up to the caller, with the label to stamp. It is
    live now: P0 is a single-seed pass, so every arm here is n=1.
    """
    e = np.asarray(events).ravel()
    if e.size and not np.all(np.isin(e, (0, 1))):
        raise ValueError('events must be 0 (censored) or 1 (event observed)')
    k, n = int(e.sum()), int(e.size)
    ci = clopper_pearson(k, n, alpha)
    below = bool(0 < n < MIN_N_FOR_INFERENCE)
    return {'k': k, 'n': n,
            'proportion': (k / n) if n else float('nan'),
            'ci': ci, 'alpha': float(alpha),
            'below_inference_floor': below,
            'label': PIPELINE_VALIDATION_LABEL if below else None,
            'reason': ci.reason}


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
    cumulative probability of having reached it.

    The interval is the standard complementary log-log (cloglog) interval: a
    **normal approximation on the log(-log S) scale**, using the Greenwood
    variance, back-transformed so the bounds land inside [0, 1]. It is a normal
    approximation, on a transformed scale, and calling it anything else would
    overstate the case: the module's opening promise is that no
    normality-assuming interval is applied to *returns*, and this one is
    applied to env steps, so `ANALYSIS_PLAN.md` §8's prohibition (scoped to
    returns) is not breached. The alternative, a normal interval on the
    survival scale itself, is what would put the bounds outside [0, 1].

    Assumes censoring is independent of the event time -- satisfied here by
    construction, because the budget is the same for every run and is fixed
    before it starts. Does **not** extrapolate past the largest observed time,
    and the curve is undefined beyond it: if no run reaches the threshold the
    honest output is a flat curve at 1 and the Clopper-Pearson proportion, not
    an imputed median.

    An **empty** arm refuses. A flat curve at 1 from zero runs and a flat curve
    at 1 from six runs that never reached the threshold are the same picture
    and completely different findings, and a plotted curve carries no n, so the
    empty case returns `refused=True` rather than a curve.
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
    if t.size == 0:
        empty = np.zeros(0, dtype=float)
        return {'time': empty, 'survival': empty,
                'at_risk': np.zeros(0, dtype=int),
                'n_events_at': np.zeros(0, dtype=int),
                'se': empty, 'ci_lo': empty, 'ci_hi': empty,
                'n': 0, 'n_events': 0, 'n_censored': 0,
                'median': None, 'alpha': float(alpha),
                'delayed_entry': entry is not None,
                'refused': True,
                'reason': 'no observations: an empty arm has no survival '
                          'curve. A flat curve at 1 from zero runs is not the '
                          'same finding as a flat curve at 1 from runs that '
                          'never reached the threshold, and a plotted curve '
                          'carries no n.'}

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
            'delayed_entry': entry is not None,
            'refused': False, 'reason': None}


LOGRANK_P_ROLE = (
    'orientation only -- steps_to_threshold is a secondary metric '
    '(ANALYSIS_PLAN.md §1), the confirmatory family is the 8 co-primary tests '
    'of §2, and §8 forbids a p-value on any secondary quantity. §5 licenses '
    'this log-rank by name, so the plan is internally inconsistent here; this '
    'module resolves it in the restrictive direction. The p carries no error '
    'budget, enters no multiplicity family, and may not be asserted as a '
    'finding: report the Kaplan-Meier curves and proportion_reached.')


def logrank(groups: Mapping[str, Sequence[Any]] | Sequence[Sequence[Any]],
            min_events: int = 3) -> dict:
    """Two-sample log-rank statistic, or an explicit refusal. Orientation only.

    `ANALYSIS_PLAN.md` §5 pre-commits: "Log-rank only when both arms have at
    least 3 events; otherwise the proportion and its interval stand alone."
    That choice is fixed here so it cannot be made after seeing the censoring
    rate, and the refusal is a returned dict -- `usable=False` with the event
    counts and the reason -- rather than an exception, because the caller's job
    is to print the refusal and fall back to `proportion_reached`.

    **The p-value here is orientation only**, and `p_role` says so in the
    output, as it does in `jonckheere_terpstra`. The metric under test,
    `steps_to_threshold`, is secondary (§1, estimation-only, no p-values) and
    §8 forbids a p-value on any secondary quantity by name, while §5 licenses
    the log-rank. Two clauses of one plan disagree; a permissive reading would
    let a secondary metric acquire an unbudgeted p-value, so the restrictive
    one is taken and the number is labelled rather than dropped.

    Each group is `(times, events)` or `(times, events, entry)`. `entry`
    implements left truncation exactly as `kaplan_meier` does: a run enters the
    risk set only after its freeze window ends, which is what §5 requires where
    a freeze is in force. Without it, running both functions on the same
    freeze-window data gives a Kaplan-Meier curve with correct risk sets and a
    log-rank statistic with wrong ones.

    Assumes proportional hazards for the statistic to be a test of a single
    hazard ratio, independent administrative censoring, and exactly two groups.
    Does **not** estimate an effect size: a log-rank p-value with no hazard
    ratio and no interval says nothing about magnitude, so report it beside the
    Kaplan-Meier curves and `proportion_reached`, never alone.
    """
    if isinstance(groups, Mapping):
        names = list(groups)
        arms = [groups[k] for k in names]
    else:
        arms = list(groups)
        names = [f'group{i}' for i in range(len(arms))]
    if len(arms) != 2:
        return {'usable': False, 'chi2': None, 'df': None, 'p': None,
                'p_role': LOGRANK_P_ROLE, 'groups': names,
                'delayed_entry': False,
                'reason': f'the log-rank implementation here handles exactly '
                          f'two arms; {len(arms)} were given'}

    ts: list[np.ndarray] = []
    es: list[np.ndarray] = []
    ens: list[np.ndarray] = []
    any_entry = False
    for arm in arms:
        parts = list(arm)
        if len(parts) not in (2, 3):
            raise ValueError('each group is (times, events) or '
                             '(times, events, entry)')
        tv = _vector(parts[0], 'times')
        ev = np.asarray(parts[1]).ravel().astype(int)
        if tv.size != ev.size:
            raise ValueError('times and events must match in length')
        if ev.size and not np.all(np.isin(ev, (0, 1))):
            raise ValueError('events must be 0 or 1')
        if len(parts) == 3 and parts[2] is not None:
            en = _vector(parts[2], 'entry')
            any_entry = True
        else:
            en = np.zeros_like(tv)
        if en.size != tv.size:
            raise ValueError('entry must match times in length')
        if np.any(en > tv):
            raise ValueError('an entry time exceeds its event/censoring time')
        ts.append(tv)
        es.append(ev)
        ens.append(en)
    counts = {names[i]: int(es[i].sum()) for i in range(2)}
    if min(counts.values()) < int(min_events):
        return {'usable': False, 'chi2': None, 'df': None, 'p': None,
                'p_role': LOGRANK_P_ROLE, 'groups': names, 'events': counts,
                'delayed_entry': any_entry,
                'reason': f'pre-committed gate: both arms need >= '
                          f'{int(min_events)} events; observed {counts}. '
                          f'Report proportion_reached and the Kaplan-Meier '
                          f'curves instead (ANALYSIS_PLAN.md §5).'}

    all_t = np.concatenate(ts)
    all_e = np.concatenate(es)
    all_en = np.concatenate(ens)
    o1 = float(es[0].sum())
    e1 = 0.0
    var = 0.0
    # Risk set under left truncation: entered strictly before tt and not yet
    # left. With entry all zero this is the ordinary `t >= tt`, so the pairs
    # form of the argument behaves exactly as before.
    for tt in np.unique(all_t[all_e == 1]):
        n_at = float(np.count_nonzero((all_en < tt) & (all_t >= tt)))
        n1 = float(np.count_nonzero((ens[0] < tt) & (ts[0] >= tt)))
        d = float(np.count_nonzero((all_t == tt) & (all_e == 1)))
        if n_at <= 1.0:
            continue
        e1 += d * n1 / n_at
        var += d * (n_at - d) * n1 * (n_at - n1) / (n_at ** 2 * (n_at - 1.0))
    if var <= 0.0:
        return {'usable': False, 'chi2': None, 'df': None, 'p': None,
                'p_role': LOGRANK_P_ROLE, 'groups': names, 'events': counts,
                'delayed_entry': any_entry,
                'reason': 'log-rank variance is zero (no informative risk '
                          'set)'}
    chi2 = (o1 - e1) ** 2 / var
    return {'usable': True, 'chi2': float(chi2), 'df': 1,
            'p': float(stats.chi2.sf(chi2, 1)),
            'p_role': LOGRANK_P_ROLE,
            'family': 'none -- outside the confirmatory family of '
                      'ANALYSIS_PLAN.md §2, and outside the screen family '
                      'of §3',
            'observed': o1, 'expected': float(e1), 'variance': float(var),
            'groups': names, 'events': counts,
            'delayed_entry': any_entry,
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

    The reported **effect** is `tau`, the scale-free concordance
    `2 J / J_max - 1`: -1 when every ordered cross pair contradicts the
    ordering, 0 at the null, +1 when every one agrees. `ANALYSIS_PLAN.md` §3
    asks for "a standardised effect with a bootstrap CI", and `z` is not one:
    it grows as sqrt(n) at fixed effect, so the same trend structure at 4 and
    at 10 per group gives different z with different intervals while `tau`
    stays put. H4's refutation condition is "the ordered-alternative test's
    interval covers zero", so it has to be evaluated on a quantity that does
    not conflate effect size with sample size. `ci` is therefore the bootstrap
    interval on `tau`; `z`, `z_ci_lo` and `z_ci_hi` are kept beside it for
    continuity with the test statistic, under names that say what they are.

    `direction` is gated twice over: `increasing` or `decreasing` only when the
    bootstrap interval on tau excludes the null **and** the exact permutation
    test rejects at `alpha`, and `not distinguishable from no trend` otherwise.
    The ungated adjective was the affirming-a-null pattern `DESIGN.md` §9 exists
    to block: a bare `increasing` beside an interval covering zero, and a `flat`
    level that renders a null as a positive description. The raw sample fact is
    still reported as `sample_direction`.

    The second condition is there because the first one alone does not deliver
    the 5 % its framing implies. The group-stratified percentile bootstrap
    under-covers at these group sizes, so the interval-only gate emitted a
    directional adjective on 40 of 600 iid-null draws at RQ5's designed
    configuration (3 wind levels, 10 seeds), 6.7 % against a nominal 5 %, and on
    12 % at k=4, n=4. The permutation p is exact and rejected 30 of those 600,
    5.0 %; the conjunction fired on 29, 4.8 %. `interval_excludes_null`,
    `permutation_rejects` and `direction_gate` report the two components
    separately, so neither is hidden inside the adjective. Adding the
    permutation condition only ever withholds an adjective, never grants one,
    and it lends RQ5 no error budget: the p stays orientation-only per
    `ANALYSIS_PLAN.md` §3 and §7, and nothing here is promoted to a test.

    The permutation p is valid with ties; the closed-form variance behind `z`
    is the no-tie formula, so where ties are present `z` is approximate and
    `ties` says so.

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
        'J': float('nan'), 'j_max': float('nan'),
        'expected': float('nan'), 'variance': float('nan'),
        'tau': float('nan'), 'concordance': float('nan'),
        'z': float('nan'), 'p_perm': float('nan'), 'n_perm': 0,
        'n_boot': 0, 'ties': False,
        'ci_lo': float('nan'), 'ci_hi': float('nan'),
        'z_ci_lo': float('nan'), 'z_ci_hi': float('nan'),
        'interval_covers_null': True,
        'interval_excludes_null': False,
        'permutation_rejects': False,
        'direction_gate': 'the bootstrap interval on tau must exclude the null '
                          'AND the exact permutation test must reject at alpha',
        'direction': 'not distinguishable from no trend',
        'sample_direction': None,
        'unequal_group_sizes': bool(len(set(sizes)) > 1),
        'group_sizes': sizes, 'n': int(sum(sizes)), 'k': k,
        'effect_name': 'tau = 2 J / J_max - 1 (scale-free concordance)',
        'p_role': 'orientation only -- RQ5 is estimation-only '
                  '(ANALYSIS_PLAN.md §3, §7); this p carries no error budget',
    }
    def _refuse(reason: str) -> dict:
        # The refusal dict carries every key the computed dict does, and `ci`
        # and `z_ci` are refused `CI`s rather than absent, so a caller written
        # against the computed shape prints the refusal instead of raising on
        # it. Today's single-seed tree takes this branch for every cell.
        out['refused'] = True
        out['reason'] = reason
        out['ci'] = _refused_ci(reason, alpha, int(min(sizes)) if sizes else 0)
        out['z_ci'] = _refused_ci(reason, alpha,
                                  int(min(sizes)) if sizes else 0)
        return out

    if k < 3:
        return _refuse(f'an ordered alternative needs at least 3 ordered '
                       f'levels; {k} given')
    if min(sizes) < MIN_N_FOR_INFERENCE:
        return _refuse(_too_few(min(sizes)))

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

    # J_max is the number of ordered cross pairs, which is twice the null mean.
    # tau = 2 J / J_max - 1 = J / expected - 1, so the scale-free effect costs
    # nothing extra to accumulate alongside z.
    j_max = 2.0 * expected
    tau_obs = (j_obs / expected - 1.0) if expected > 0 else float('nan')

    # Group-stratified bootstrap, on the scale-free effect (which is what §3
    # asks for) and on the standardised statistic (kept for continuity).
    zs = np.empty(int(n_boot), dtype=float)
    taus = np.empty(int(n_boot), dtype=float)
    for bi in range(int(n_boot)):
        resampled = [g[rng.integers(0, g.size, g.size)] for g in groups]
        pv = np.concatenate(resampled)
        jb = _jt_from_matrix(_comparison_matrix(pv), idx_sets)
        zs[bi] = (jb - expected) / math.sqrt(variance) if variance > 0 \
            else np.nan
        taus[bi] = (jb / expected - 1.0) if expected > 0 else np.nan
    good_z = zs[np.isfinite(zs)]
    good_tau = taus[np.isfinite(taus)]
    lo_q, hi_q = 100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)
    tau_lo = float(np.percentile(good_tau, lo_q)) if good_tau.size \
        else float('nan')
    tau_hi = float(np.percentile(good_tau, hi_q)) if good_tau.size \
        else float('nan')

    # DESIGN.md §9: a directional adjective is generated from the data *and*
    # may not affirm a null. The interval decides, not the point estimate, and
    # the exact permutation test has to agree: the stratified percentile
    # bootstrap under-covers at k=3, n=10, so the interval alone let the
    # adjective through on 6.7 % of iid-null draws where 5 % is nominal.
    # Requiring both can only withhold an adjective, never grant one.
    covers_null = not (np.isfinite(tau_lo) and np.isfinite(tau_hi)
                       and (tau_lo > 0.0 or tau_hi < 0.0))
    perm_rejects = bool(np.isfinite(p_perm) and p_perm < float(alpha))
    if covers_null or not perm_rejects:
        direction = 'not distinguishable from no trend'
    elif tau_lo > 0.0:
        direction = 'increasing'
    else:
        direction = 'decreasing'

    out.update(J=float(j_obs), j_max=float(j_max), expected=float(expected),
               variance=float(variance), z=float(z),
               tau=float(tau_obs),
               concordance=float(j_obs / j_max) if j_max > 0 else float('nan'),
               p_perm=float(min(1.0, p_perm)), n_perm=int(n_perm),
               ci_lo=tau_lo, ci_hi=tau_hi,
               z_ci_lo=float(np.percentile(good_z, lo_q)) if good_z.size
               else float('nan'),
               z_ci_hi=float(np.percentile(good_z, hi_q)) if good_z.size
               else float('nan'),
               n_boot=int(n_boot),
               ties=bool(np.unique(pooled).size != pooled.size),
               interval_covers_null=bool(covers_null),
               interval_excludes_null=bool(not covers_null),
               permutation_rejects=bool(perm_rejects),
               direction=direction,
               sample_direction=('increasing' if j_obs > expected else
                                 'decreasing' if j_obs < expected else
                                 'exactly at the null mean'),
               refused=False, reason=None)
    out['ci'] = CI(tau_lo, tau_hi, method='percentile',
                   alpha=alpha, n=int(min(sizes)))
    out['z_ci'] = CI(out['z_ci_lo'], out['z_ci_hi'], method='percentile',
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
                 chunk: int = 2_000,
                 n_perm: int = SIGNFLIP_MDE_N_PERM) -> MDE:
    """Minimum detectable paired effect for the sign-flip test, in sigma units.

    Simulates `n_sim` datasets of `n` deltas drawn as `N(mu, 1)`, runs the exact
    sign-flip test on each, and bisects on `mu` for the smallest shift whose
    rejection rate reaches `power`. The unit is sigma_delta -- the SD of the
    *paired delta*, not of either arm -- so translating it into score units
    needs the observed delta SD, as `ANALYSIS_PLAN.md` §6.3 does.

    Reproduces the pinned values of `ANALYSIS_PLAN.md` §6.2 at n=10: **1.009
    sigma at alpha=0.05** and **1.535 sigma at alpha=0.00625**, the Holm
    step-down floor over the 8-test family, which §6.2's table rounds to 1.00
    and 1.54.

    What the 1e-3 assertions in `self_test` are, stated exactly, because the
    comment there used to claim more. They are **regression pins on a fixed RNG
    stream**: `seed` is fixed, so the returned value is deterministic and any
    change to this code moves it. They are *not* three-decimal knowledge of the
    estimand. Re-running these same estimators at their own `n_sim=20,000` over
    four simulation seeds gives, at n=10: sign-flip alpha=0.05 1.0087, 1.0065,
    1.0043, 1.0043 (spread 0.0044); sign-flip alpha=0.00625 spread 0.0095;
    Mann-Whitney alpha=0.05 1.4064, 1.3987, 1.4017, 1.3918 (spread 0.0146, mean
    1.3997); Mann-Whitney alpha=0.00625 spread 0.0135. The Monte Carlo error is
    four to fifteen times the tolerance the assertions use, which follows from
    the design of the estimator rather than being a surprise: the power is a
    binomial proportion over `n_sim` draws, so its standard error at the 0.8
    target is `sqrt(0.8 * 0.2 / 20000) = 0.0028`, and dividing by the local
    slope of the power curve puts the SE of the returned effect at roughly
    0.005 sigma. A difference of 0.01 between this estimator and the plan's
    table is therefore inside its own noise, and neither figure refutes the
    other at this `n_sim`.

    The null distribution is enumerated exactly for
    `n <= EXACT_SIGNFLIP_MDE_MAX_N` (=15); above that the *power simulation*
    switches to `n_perm` Monte Carlo sign assignments and `.method` says so.
    That cap is lower than the test's own cap of 20 because here the
    enumeration runs once per simulated dataset rather than once per call.

    Two consequences for the n=16..20 planning numbers of §6.5, stated plainly
    because §6.5 says they were "computed the same way" as the n=10 ones and
    they are not. First, at n=16..20 the *test* enumerates the null exactly
    while this *power simulation* approximates it with 4096 draws, so the MDE
    is quoted under a null the test never uses. Second, and now fixed: the
    Monte Carlo branch used `hits / n_perm` as its rejection rule while
    `sign_flip_test` uses the add-one estimator `(1 + hits) / (n_perm + 1)`, so
    the power was simulated against a rule anti-conservative relative to the
    test it characterises. Both branches now use the test's own rule.

    That fix is right on its own terms and it is **not** an explanation of any
    disagreement with §6.5, which is what the text here used to claim. Holding
    the RNG stream fixed and flipping only the rejection rule moves n=20
    alpha=0.05 from 0.6623 to 0.6630, a difference of +0.0007 against a
    seed-to-seed spread of 0.0055 for the add-one rule itself, and n=20
    alpha=0.00625 from 0.8897 to 0.8937, +0.0040 against a spread of 0.0037.
    The rule shifts the reject boundary by exactly one count in 4,096, and at
    that size it cannot adjudicate §6.5's 0.662 and 0.890 against this
    estimator's 0.663 and 0.894: all four numbers are one Monte Carlo error
    apart. Nothing here supersedes the pre-registered values, and `self_test`
    pins what this code returns rather than a claim about which figure is
    right. §6.5's n=20 row is in any case documentation: the project has
    settled on n=10 and will not run REPLICATE.

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
    exact = n <= EXACT_SIGNFLIP_MDE_MAX_N
    if exact:
        signs = _sign_matrix(n)
        method = 'exact enumeration of 2**n sign assignments'
    else:
        n_mc = int(n_perm)
        signs = (rng.integers(0, 2, size=(n_mc, n)).astype(float) * 2.0 - 1.0)
        method = (f'Monte Carlo sign assignments (n_perm={n_mc}), add-one '
                  f'estimator as in sign_flip_test; the test itself is exact '
                  f'up to n={EXACT_SIGNFLIP_MAX_N}, so this is a planning '
                  f'approximation to a null the test does not use')
    n_assign = int(signs.shape[0])
    block = max(1, min(int(chunk), 2_000_000 // n_assign))

    def power_at(mu: float) -> float:
        rejected = 0
        for start in range(0, z.shape[0], block):
            d = z[start:start + block] + mu
            t = d @ signs.T
            obs = np.abs(d.sum(axis=1))[:, None]
            hits = (np.abs(t) >= obs - 1e-9).sum(axis=1)
            # The rejection rule has to be the one `sign_flip_test` applies, or
            # the power is simulated against a test that does not exist. Exact
            # enumeration divides by 2**n; the Monte Carlo branch uses the
            # add-one estimator, which can never report p=0.
            p = (hits / n_assign) if exact \
                else ((1.0 + hits) / (n_assign + 1.0))
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

    Returns **1.406 sigma at alpha=0.05** and **1.880 sigma at alpha=0.00625**
    at n=10 vs 10, which is what `ANALYSIS_PLAN.md` §6.2's table now reads
    (1.41 and 1.88, corrected from 1.39 and 1.87 on 2026-08-26 and logged in
    §11). The self-test used to assert the table's old numbers under a tolerance
    of 0.1, which is 7 % at these magnitudes and wide enough to hide the
    disagreement; it now asserts what this code returns, at 1e-3, which is a
    regression pin on a fixed RNG stream rather than three-decimal knowledge of
    the estimand. Over four simulation seeds at n_sim=20,000 this estimator
    spreads by 0.0146 at alpha=0.05 (1.4064, 1.3987, 1.4017, 1.3918, mean
    1.3997), so it cannot resolve differences of 0.01 and no claim here rests on
    its third decimal.

    Compare `mde_signflip`'s 1.009 and 1.535: at this sample size the
    matched-seed design is worth roughly a 28 % reduction in the detectable
    effect (equivalently, the unpaired test needs an effect about 39 % larger),
    which is why the paired test is primary (§6.2) and why discarding the
    pairing is a real cost rather than a stylistic choice.

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
def equivalence_from_ci(lo: float | CI | Sequence[float],
                        hi: float | None = None,
                        margin: float = EQUIVALENCE_MARGIN,
                        sd: float | None = None,
                        alpha: float | None = None) -> dict:
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

    Pass the `CI` object, not two loose floats: `equivalence_from_ci(ci)`
    unpacks it and keeps `.method`, `.fallback`, `.n` **and `.alpha`** in the
    output. Two floats still work, and then `ci_method` records that the
    provenance was not supplied. This matters because the output is a finished
    sentence rather than a number a reader can sanity-check:
    `bootstrap_ci([0.7] * 10)`
    returns
    `CI(0.7, 0.7, percentile (bootstrap distribution is degenerate))`, and the
    two-float form turned that into "any degradation is excluded at 95 %; the
    delta is at least 0.7 score units" with nothing recording that the interval
    had zero width and was a fallback. A zero-width interval quantifies no
    uncertainty, so it is now refused outright rather than narrated.

    A reversed interval (`hi < lo`) is **refused**, not silently reordered.
    Swapping the arguments used to yield a confident, plausible exclusion
    sentence from a malformed input.

    **The confidence level in the sentence is read from the interval.** It used
    to be the literal string "95 %", written into all three exclusion sentences
    whatever the interval was: `equivalence_from_ci(bootstrap_ci(x,
    alpha=0.50))` narrated a 50 % interval as "excluded at 95 %", and the error
    ran in the dangerous direction, since a weaker interval yields a
    tighter-sounding bound under a stronger-sounding label. The level now comes
    from `CI.alpha`, or from the `alpha` argument on the two-float path, and
    `ci_alpha`, `confidence_pct` and `ci_alpha_source` put it in the output
    where a consumer can check it. Passing a `CI` *and* an `alpha` that
    disagree with it is refused rather than resolved in either direction.
    Supplying neither leaves the level unknown; it is then assumed to be the
    pre-registered `ALPHA` and `ci_alpha_source` says `assumed`, which the
    sentence repeats, because an assumed level must not read like an observed
    one.

    `ANALYSIS_PLAN.md` §4 names the 95 % interval as the instrument for the
    equivalence verdict, so an interval built at any other level is not that
    instrument: the `equivalent` and `inconclusive` branches then report
    `untestable`, the way an unsupplied `sd` does. `not_equivalent` is left
    alone, being an exclusion claim carried by the interval rather than an
    equivalence claim, and the exclusion sentence is always emitted, now
    labelled at the level the interval was actually built at.

    Verdicts, in the order they are decided:

    * `refused` when the interval is not finite, is reversed, or has zero
      width.
    * `untestable` when the cell's dispersion does not permit an equivalence
      claim at this n, which covers two cases: `sd` exceeds the margin, and
      `sd` was not supplied at all. `ANALYSIS_PLAN.md` §4 makes an equivalence
      claim available "only in cells whose observed across-seed SD is below
      0.05 score units", so a cell whose SD is unknown has not met the
      condition either, and an unguarded `equivalent` verdict is no longer
      obtainable by omitting `sd`. 0.05 is 1.17 SD in the quiet cells and 0.14
      SD in the noisy dueling scratch cell, where it is hopeless.
    * otherwise the geometric verdict: `equivalent` when the whole interval is
      inside +/-margin, `not_equivalent` when the whole interval is outside it
      on one side, `inconclusive` in between.

    The untestability check applies to `inconclusive` as well as to
    `equivalent`, which is the fix for the case §4 wrote the rule for: the
    noisy dueling scratch cell (SD 0.369) has a wide CI, so it lands on
    `inconclusive`, and gating only the `equivalent` branch meant the
    pre-committed untestability statement was never produced for the one cell
    the pre-registration named. `not_equivalent` is left alone: it is an
    exclusion claim carried by the interval, not an equivalence claim, and the
    dispersion note rides along in `equivalence_statement` instead.

    `geometric_verdict` always reports what the interval alone said, and
    `equivalence_statement` is the per-cell sentence §4 pre-commits to: either
    the verdict or the statement that dispersion makes equivalence untestable
    at this sample size.

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

    Not on the paper's path today, and the docstring above said otherwise until
    it was checked: no module outside this one calls this function, and the
    exclusion sentence that reaches the report is built by
    `stats.py:phrase_exclusion_bound` from `stats.py`'s own bootstrap. Treat
    what follows as the reference implementation of `ANALYSIS_PLAN.md` §4, and
    note that the live sentence is fed only default-alpha intervals, so it is
    correct today for the reason this one was: not by construction.
    """
    margin = abs(float(margin))
    # Accept the CI object itself, so its provenance survives into the output.
    ci_method: str | None = None
    ci_fallback: str | None = None
    ci_n: int | None = None
    ci_refused = False
    ci_alpha: float | None = None if alpha is None else float(alpha)
    alpha_source = 'assumed' if alpha is None else 'argument'
    alpha_conflict: tuple[float, float] | None = None
    if hi is None:
        source = lo
        if isinstance(source, CI):
            ci_method = source.method
            ci_fallback = source.fallback
            ci_n = source.n
            ci_refused = source.refused
            # The interval's own level wins, and an `alpha` argument that
            # contradicts it is an error rather than an override: one of the
            # two is wrong and this function cannot tell which.
            if alpha is not None and abs(float(alpha) - source.alpha) > 1e-12:
                alpha_conflict = (float(alpha), float(source.alpha))
            ci_alpha = float(source.alpha)
            alpha_source = 'interval'
        try:
            pair = list(source)                   # type: ignore[arg-type]
        except TypeError:
            raise ValueError(
                f'an interval is two numbers or a CI; got a single '
                f'{type(source).__name__}. Call equivalence_from_ci(ci) or '
                f'equivalence_from_ci(lo, hi).') from None
        if len(pair) != 2:
            raise ValueError(
                f'an interval is two numbers or a CI; got {len(pair)} values. '
                f'Call equivalence_from_ci(ci) or equivalence_from_ci(lo, hi).')
        lo_v, hi_v = float(pair[0]), float(pair[1])
    else:
        lo_v, hi_v = float(lo), float(hi)         # type: ignore[arg-type]

    # The level the sentence is entitled to assert. Unknown means unknown: it
    # is assumed to be the pre-registered one and says so, rather than being
    # asserted as if it had been read off the interval.
    level = ALPHA if ci_alpha is None else float(ci_alpha)
    confidence_pct = round(100.0 * (1.0 - level), 10)
    level_label = f'{confidence_pct:g} %'
    level_phrase = f'at {level_label}'
    level_is_pre_registered = abs(level - ALPHA) <= 1e-12
    provenance = {'ci_method': ci_method, 'ci_fallback': ci_fallback,
                  'ci_n': ci_n,
                  'ci_provenance_known': ci_method is not None,
                  'ci_alpha': None if ci_alpha is None else float(ci_alpha),
                  'ci_alpha_source': alpha_source,
                  'confidence_pct': float(confidence_pct),
                  'confidence_label': level_label,
                  'ci_alpha_is_pre_registered': bool(level_is_pre_registered)}

    def _refuse(reason: str, sentence: str) -> dict:
        return {'verdict': 'refused', 'geometric_verdict': None,
                'margin': margin, 'ci_lo': lo_v, 'ci_hi': hi_v,
                'exclusion_bound': float('nan'),
                'worst_degradation_excluded': float('nan'),
                'best_improvement_excluded': float('nan'),
                'sd': None if sd is None else float(sd),
                'equivalence_testable': False,
                'equivalence_statement': 'No equivalence verdict is available.',
                **provenance, 'reason': reason, 'sentence': sentence}

    if alpha_conflict is not None:
        return _refuse(
            f'alpha={alpha_conflict[0]:g} was asserted for an interval built '
            f'at alpha={alpha_conflict[1]:g}. One of the two is wrong and this '
            f'function cannot tell which; fix the caller.',
            'No usable interval is available, so nothing is excluded.')
    if ci_refused or not (np.isfinite(lo_v) and np.isfinite(hi_v)):
        return _refuse(
            'the interval is not finite (n<3, or the estimator refused)',
            'No interval is available, so nothing is excluded.')
    if hi_v < lo_v:
        return _refuse(
            f'the interval is reversed: lo={lo_v:.6g} exceeds hi={hi_v:.6g}. '
            f'Reordering it here would turn a swapped pair of arguments, or a '
            f'statistic whose bounds crossed, into a confident exclusion '
            f'sentence. Fix the caller.',
            'No usable interval is available, so nothing is excluded.')
    if hi_v == lo_v:
        return _refuse(
            f'the interval has zero width at {lo_v:.6g}'
            + (f' ({ci_fallback})' if ci_fallback else '')
            + '. An interval that quantifies no uncertainty licenses no '
              'exclusion; report the point estimate and the reason the '
              'bootstrap degenerated instead.',
            'No usable interval is available, so nothing is excluded.')

    lo_v, hi_v = float(lo_v), float(hi_v)
    if lo_v >= -margin and hi_v <= margin:
        geometric = 'equivalent'
    elif lo_v >= margin or hi_v <= -margin:
        geometric = 'not_equivalent'
    else:
        geometric = 'inconclusive'

    # ANALYSIS_PLAN §4: an equivalence claim is available only where the cell's
    # across-seed SD is below the margin. An unknown SD has not met that
    # condition either, so omitting `sd` cannot buy an `equivalent` verdict.
    reason = None
    if not level_is_pre_registered:
        # §4 assesses equivalence with the 95 % interval. An interval at
        # another level is a different instrument, and at alpha > ALPHA it is
        # a narrower one, which would make `equivalent` easier to reach than
        # the pre-registration allows.
        testable = False
        reason = (f'the interval was built at alpha={level:g} '
                  f'({level_label}), not the pre-registered alpha={ALPHA:g}. '
                  f'ANALYSIS_PLAN.md §4 assesses equivalence with the 95 % '
                  f'interval, so this one cannot deliver that verdict. The '
                  f'exclusion bound below is still reported, at its own level.')
    elif sd is None:
        testable = False
        reason = (f'no across-seed SD was supplied. ANALYSIS_PLAN.md §4 makes '
                  f'an equivalence claim available only in cells whose '
                  f'observed SD is below the margin {margin:g}, so the verdict '
                  f'cannot be granted without it.')
    elif not np.isfinite(float(sd)):
        testable = False
        reason = ('the across-seed SD is not finite, so the §4 dispersion '
                  'condition cannot be evaluated.')
    elif float(sd) >= margin:
        # §4's wording is "SD is below 0.05", so equality does not qualify.
        testable = False
        reason = (f'across-seed SD {float(sd):.4g} is not below the margin '
                  f'{margin:g}: equivalence is untestable in this cell at '
                  f'this n (ANALYSIS_PLAN.md §4)')
    else:
        testable = True

    verdict = geometric
    if not testable and geometric in ('equivalent', 'inconclusive'):
        verdict = 'untestable'

    if verdict == 'untestable':
        statement = (f"The cell's dispersion makes equivalence untestable at "
                     f"this sample size: {reason}")
    elif verdict == 'equivalent':
        statement = (f'Equivalent: the whole interval lies inside the '
                     f'pre-registered +/-{margin:g} margin.')
    elif verdict == 'not_equivalent':
        statement = (f'Not equivalent: the whole interval lies outside the '
                     f'+/-{margin:g} margin on one side.')
    else:
        statement = (f'No equivalence verdict: the interval crosses the '
                     f'+/-{margin:g} margin.')

    exclusion = lo_v if abs(lo_v) <= abs(hi_v) else hi_v
    worst_deg = max(0.0, -lo_v)
    best_imp = max(0.0, hi_v)
    if lo_v > 0.0:
        sentence = (f'Any degradation is excluded {level_phrase}; the delta is '
                    f'at least {lo_v:.4g} score units.')
    elif hi_v < 0.0:
        sentence = (f'Any improvement is excluded {level_phrase}; the delta is '
                    f'at most {hi_v:.4g} score units.')
    else:
        sentence = (f'A degradation worse than {worst_deg:.4g} score units is '
                    f'excluded {level_phrase}, as is an improvement better '
                    f'than {best_imp:.4g}.')
    if alpha_source == 'assumed':
        sentence += (f' (Level assumed to be the pre-registered {level_label}: '
                     f'two floats carry no provenance, so the interval\'s own '
                     f'level is unknown here.)')
    if ci_fallback:
        sentence += (f' (Interval provenance: {ci_method}, {ci_fallback}.)')
    return {'verdict': verdict, 'geometric_verdict': geometric,
            'margin': margin, 'ci_lo': lo_v, 'ci_hi': hi_v,
            'exclusion_bound': float(exclusion),
            'worst_degradation_excluded': float(worst_deg),
            'best_improvement_excluded': float(best_imp),
            'sd': None if sd is None else float(sd),
            'equivalence_testable': bool(testable),
            'equivalence_statement': statement,
            **provenance, 'reason': reason, 'sentence': sentence}


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
    # The floor is not the bar. Three attainable p-values clear the strictest
    # Holm step, so "all ten seeds move the same way" is sufficient, not
    # necessary. ANALYSIS_PLAN §2.2 said "if and only if" and was corrected to
    # the attainable bar on 2026-08-26 (§11), so the two now agree; this check
    # is what would catch them parting company again.
    clearing = signflip_attainable_p_below(10, HOLM_STRICTEST_ALPHA)
    assert clearing == (2 / 1024, 4 / 1024, 6 / 1024), clearing
    note('paired p-values clearing the Holm-strictest 0.00625 at n=10',
         [round(p, 6) for p in clearing])
    nine_of_ten = sign_flip_test([10.0] * 9 + [-1.0])
    assert not nine_of_ten['all_same_sign']
    assert nine_of_ten['p_two_sided'] <= HOLM_STRICTEST_ALPHA, nine_of_ten
    note('9 of 10 seeds can clear the bar (p, all_same_sign)',
         f"{nine_of_ten['p_two_sided']:.6f}, "
         f"{nine_of_ten['all_same_sign']}")
    assert signflip_attainable_p_below(6, HOLM_STRICTEST_ALPHA) == ()
    note('at n=6 nothing clears 0.00625, whatever the data do', 'ok')

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
    assert mw['method'] == 'exact' and mw['critical_values'] == (23, 77)
    assert mw['critical_values_holm_strictest'] == (14, 86)
    note('fully separated 10 vs 10 -> U=100, p at the floor', f"{mw['p']:.3e}")

    # The tie bug, in both of its forms. With any repeated value the exact null
    # is wrong, SciPy's tie-corrected asymptotic branch runs, and its floor is a
    # different number from the exact combinatorial one. Reporting the exact
    # floor beside an asymptotic p gave a p BELOW its own stated floor at small
    # n, and killed at_p_floor at the confirmatory sample size.
    tied = mann_whitney([0.0] * 5, [1.0] * 5)
    assert tied['method'] == 'asymptotic' and tied['ties']
    assert tied['p'] >= tied['p_min_attainable'] * (1.0 - 1e-12), tied
    assert tied['at_p_floor'], tied
    assert tied['p_min_attainable'] < tied['p_min_attainable_exact_null']
    note('5 vs 5, all tied within arm: p is not below its own floor',
         f"p={tied['p']:.6f} floor={tied['p_min_attainable']:.6f} "
         f"(exact-null floor {tied['p_min_attainable_exact_null']:.6f})")
    sep = mann_whitney([0.7] * 10, [0.9] * 10)
    assert sep['U'] == 0.0 and sep['at_p_floor'], sep
    assert sep['method'] == 'asymptotic'
    assert sep['critical_values_method'] == 'asymptotic'
    assert sep['p_min_attainable_u'] == 0.0, sep
    note('complete separation at 10 vs 10 with ties -> at_p_floor fires',
         f"p={sep['p']:.3e} floor={sep['p_min_attainable']:.3e}")
    # The other half of the tie bug, and the reason `at_p_floor` was still
    # dead in a quarter of tied cells: a value shared ACROSS the arms makes
    # U=0 unattainable, so a floor evaluated there is a bound no arrangement
    # of the data can reach and the flag can never fire. This sample IS the
    # most extreme split of its own pooled multiset.
    cross = mann_whitney([1.0, 2.0, 3.0], [3.0, 4.0, 5.0])
    brute = min(mann_whitney([[1.0, 2.0, 3.0, 3.0, 4.0, 5.0][i] for i in c],
                             [[1.0, 2.0, 3.0, 3.0, 4.0, 5.0][i]
                              for i in range(6) if i not in c])['p']
                for c in itertools.combinations(range(6), 3))
    assert abs(cross['p_min_attainable'] - brute) < 1e-12, (cross, brute)
    assert abs(cross['p'] - brute) < 1e-12
    assert cross['at_p_floor'], cross
    assert cross['p_min_attainable_u'] == 0.5
    assert cross['p_min_attainable_note'] and 'shared across the arms' \
        in cross['p_min_attainable_note']
    note('a cross-arm tie: the floor is the true attainable minimum, '
         'enumerated', f"floor={cross['p_min_attainable']:.5f} "
                       f"= min over all C(6,3) splits")
    # Ties make the two attainable extremes asymmetric, so the upper one
    # cannot be had by reflecting the lower one. This pool reaches U = n1 n2
    # while its U_min is 1.
    asym = [0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0]
    assert mwu_min_attainable_u(4, 5, asym) == 1.0
    assert mwu_max_attainable_u(4, 5, asym) == 20.0
    note('U_max is not n1 n2 - U_min under ties, and is computed as itself',
         f'U_min=1, U_max=20, n1 n2=20')
    # The asymptotic branch must reproduce SciPy exactly, or its floor and its
    # rejection region describe a null the p did not come from.
    worst = 0.0
    for _ in range(60):
        xa = rng.integers(0, 4, int(rng.integers(3, 12))).astype(float)
        yb = rng.integers(0, 4, int(rng.integers(3, 12))).astype(float)
        pooled_xy = np.concatenate((xa, yb))
        if np.unique(pooled_xy).size == pooled_xy.size:
            continue
        ref = stats.mannwhitneyu(xa, yb, alternative='two-sided',
                                 method='asymptotic')
        mine = mwu_asymptotic_p(float(ref.statistic), xa.size, yb.size,
                                pooled_xy)
        worst = max(worst, abs(mine - float(ref.pvalue)))
    assert worst < 1e-12, worst
    note('tie-corrected asymptotic p reproduces scipy exactly',
         f'max |diff| = {worst:.1e}')
    short = mann_whitney(list(range(10)), list(range(100, 109)))
    assert short['unequal_n'] and short['unequal_n_note']
    note('a short arm is reported rather than computed over in silence',
         f"n1={short['n1']}, n2={short['n2']}, unequal_n=True")
    assert mann_whitney([1.0, 2, 3], [4.0, 5, 6])['confirmatory_eligible'] \
        is False
    note('n=3 is reported as ineligible for the confirmatory family (S4)',
         'ok')
    assert abs(hodges_lehmann([1.0, 2.0, 3.0]) - 2.0) < 1e-12
    assert abs(hodges_lehmann([5.0, 6.0, 7.0], [1.0, 2.0, 3.0]) - 4.0) < 1e-12
    note('Hodges-Lehmann one-sample and two-sample on known data', 'ok')
    assert abs(relative_effect([2.0, 3.0], [0.0, 1.0]) - 1.0) < 1e-12
    assert abs(relative_effect([1.0, 1.0], [1.0, 1.0]) - 0.5) < 1e-12
    assert abs(rank_biserial([2.0, 3.0], [0.0, 1.0]) - 1.0) < 1e-12
    note('theta = 1 when separated, 0.5 when identical; rbc = 2 theta - 1',
         'ok')
    # The n<3 leak. These three estimators return bare numbers, so at one run
    # per arm a theta of 1.0 used to be indistinguishable in the output from
    # the same number at n=10. The value is still computed; the provenance now
    # travels with it.
    one_v_one = relative_effect([1.11666], [1.09502])
    assert float(one_v_one) == 1.0 and one_v_one.below_inference_floor
    assert one_v_one.n1 == 1 and one_v_one.n2 == 1 and one_v_one.reason
    assert rank_biserial([1.11666], [1.09502]).below_inference_floor
    hl_one = hodges_lehmann([1.11666], [1.09502])
    assert hl_one.below_inference_floor and hl_one.n1 == 1
    note('theta/rbc/HL at one run per arm carry below_inference_floor',
         repr(one_v_one))
    assert not relative_effect([1.0, 2, 3], [4.0, 5, 6]).below_inference_floor
    assert relative_effect([1.0, 2, 3], [4.0, 5]).unequal_n
    note('and flag unequal arms, which mean a missing run when seed-matched',
         'ok')
    bm = brunner_munzel(rng.normal(1.0, 1.0, 12), rng.normal(0.0, 3.0, 12),
                        n_boot=2_000)
    assert 'p' not in bm and 0.0 <= bm['ci_lo'] <= bm['ci_hi'] <= 1.0
    note('Brunner-Munzel returns theta + CI and no p-value',
         f"theta={bm['theta']:.3f} ci=({bm['ci_lo']:.3f}, {bm['ci_hi']:.3f})")
    # ANALYSIS_PLAN §3 licenses a directional claim from what the interval
    # excludes. Under complete separation every resample is separated too, so
    # the bootstrap never varies and the interval collapses to [1, 1], which
    # excludes everything from n=10 against n=10. That is refused here rather
    # than returned with refused=False, which is what it used to do.
    bm_sep = brunner_munzel([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
                            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.0],
                            n_boot=500)
    assert bm_sep['theta'] == 1.0 and bm_sep['complete_separation']
    assert bm_sep['degenerate_interval'] and bm_sep['refused']
    assert bm_sep['degenerate_interval_value'] == 1.0
    assert not np.isfinite(bm_sep['ci_lo']) and bm_sep['ci'].refused
    assert equivalence_from_ci(bm_sep['ci'])['verdict'] == 'refused'
    note('complete separation refuses the interval instead of reporting [1, 1]',
         f"theta={bm_sep['theta']:.3f}, collapsed at "
         f"{bm_sep['degenerate_interval_value']:.3f}")
    assert relative_effect([2.0, 3.0], [0.0, 1.0]).complete_separation
    assert rank_biserial([2.0, 3.0], [0.0, 1.0]).complete_separation
    assert not relative_effect([1.0, 2.0], [1.5, 3.0]).complete_separation
    note('and the point estimates carry complete_separation with them', 'ok')
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
    # A non-finite *replicate* is not an input and is dropped, but it may not
    # be dropped in silence: this used to return method='bca', fallback=None,
    # indistinguishable from an interval that rested on all 4,000 resamples.
    straddling = [0.31, -0.28, 0.22, -0.19, 0.11, -0.09, 0.05, -0.04,
                  0.02, -0.01]

    def _cv(v, axis=None):
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.std(v, axis=axis, ddof=1) / np.mean(v, axis=axis)

    dropped_ci = bootstrap_ci(straddling, statistic=_cv, n_boot=4_000, seed=0)
    assert dropped_ci.replicates_dropped > 0, dropped_ci
    assert dropped_ci.n_replicates == 4_000 - dropped_ci.replicates_dropped
    assert dropped_ci.fallback and 'non-finite' in dropped_ci.fallback
    note('a dropped bootstrap replicate is recorded, not silent',
         f'{dropped_ci.replicates_dropped} dropped of 4000, '
         f'method={dropped_ci.method}')
    clean_ci = bootstrap_ci([0.1, 0.2, 0.3, 0.4, 0.5], n_boot=500)
    assert clean_ci.replicates_dropped == 0 and clean_ci.fallback is None
    assert clean_ci.n_replicates == 500
    note('and an interval that dropped nothing says so too', 'ok')

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
    empty_contrasts = paired_bootstrap({'a': [1.0, 2.0, 3.0, 4.0]}, {})
    assert empty_contrasts['refused'] and 'no contrasts' \
        in empty_contrasts['reason']
    note('an empty contrast set refuses instead of raising inside numpy', 'ok')
    try:
        paired_bootstrap({'a': [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]},
                         {'a-a': {'a': 1.0}}, seed_ids=[0, 1, 2, 0, 1, 2])
    except ValueError as exc:
        assert 'repeats' in str(exc)
        note('a duplicated seed id is refused, not resampled as two units',
             'ok')
    else:                                                   # pragma: no cover
        raise AssertionError('a duplicated seed was accepted')
    # The same corruption seen from the primary test: three distinct deltas
    # repeated to n=10 land on the p-value floor and would clear Holm.
    dup_deltas = [-0.3281, -0.7111, -0.8572] * 3 + [-0.3281]
    dup_sf = sign_flip_test(dup_deltas)
    assert dup_sf['repeated_values'] and dup_sf['n_distinct_values'] == 3
    assert dup_sf['p_two_sided'] <= HOLM_STRICTEST_ALPHA
    note('n=10 from 3 distinct values is flagged (p would clear Holm)',
         f"p={dup_sf['p_two_sided']:.6f}, distinct="
         f"{dup_sf['n_distinct_values']}")
    try:
        sign_flip_test(dup_deltas, seed_ids=[0, 1, 2] * 3 + [0])
    except ValueError:
        note('and refused outright when the seed ids are supplied', 'ok')
    else:                                                   # pragma: no cover
        raise AssertionError('a duplicated seed was accepted')

    # Every refusal dict carries every key its computed twin does, and with
    # the same container type. The n<3 branch is the ONLY branch today's
    # single-seed tree ever takes, so a consumer written against the computed
    # shape met a KeyError or an AttributeError there instead of printing the
    # refusal: `paired_bootstrap(...)['correlation']` was None where the
    # computed path returns an array, and `mann_whitney` omitted seven keys.
    parity = (
        ('sign_flip_test',
         lambda: sign_flip_test([0.1, 0.2, 0.3, -0.1]),
         lambda: sign_flip_test([0.1, 0.2])),
        ('wilcoxon_signed_rank',
         lambda: wilcoxon_signed_rank([0.1, 0.2, 0.3, -0.1]),
         lambda: wilcoxon_signed_rank([0.1, 0.2])),
        ('mann_whitney',
         lambda: mann_whitney([1.0, 2, 3, 4], [5.0, 6, 7, 8]),
         lambda: mann_whitney([1.0, 2], [5.0, 6])),
        ('brunner_munzel',
         lambda: brunner_munzel([1.0, 2, 3, 4], [5.0, 6, 7, 9], n_boot=200),
         lambda: brunner_munzel([1.0, 2], [5.0, 6], n_boot=200)),
        ('paired_bootstrap',
         lambda: paired_bootstrap({'a': [1.0, 2, 3], 'b': [3.0, 4, 6]},
                                  {'d': ('a', 'b')}, n_boot=200),
         lambda: paired_bootstrap({'a': [1.0, 2], 'b': [3.0, 4]},
                                  {'d': ('a', 'b')}, n_boot=200)),
        ('jonckheere_terpstra',
         lambda: jonckheere_terpstra([[1.0, 2, 3], [2.0, 3, 4], [3.0, 4, 5]],
                                     n_perm=100, n_boot=100),
         lambda: jonckheere_terpstra([[1.0, 2], [2.0, 3], [3.0, 4]],
                                     n_perm=100, n_boot=100)),
    )
    for name, computed_fn, refused_fn in parity:
        good, bad = computed_fn(), refused_fn()
        assert bad['refused'] and bad['reason'], (name, bad)
        missing = sorted(set(good) - set(bad))
        assert not missing, (name, missing)
        for key in sorted(set(good) & set(bad)):
            gv, bv = good[key], bad[key]
            if gv is None or bv is None:
                continue                 # optional fields, e.g. `fallback`
            assert type(gv) is type(bv) or isinstance(bv, type(gv)) \
                or isinstance(gv, type(bv)), (name, key, type(gv), type(bv))
    note('every refusal dict is key- and type-compatible with its computed '
         'twin', f'{len(parity)} estimators checked')
    refused_pb = paired_bootstrap({'a': [1.0, 2], 'b': [3.0, 4]},
                                  {'d': ('a', 'b')})
    assert refused_pb['correlation'].shape == (1, 1)
    assert refused_pb['distribution']['d'].size == 0
    refused_jt = jonckheere_terpstra([[1.0, 2], [2.0, 3], [3.0, 4]],
                                     n_perm=100, n_boot=100)
    assert refused_jt['ci'].refused and refused_jt['z_ci'].refused
    note('and its intervals are refused CIs rather than absent keys', 'ok')

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
    # ANALYSIS_PLAN §7: the family size comes from the pre-registration, not
    # from the caller. Four p-values of 0.01 reject on their own and do not
    # inside the real family of 8, which is the family-of-one rescue §7 blocks.
    assert np.allclose(holm([0.01] * 4), 0.04)
    assert np.allclose(holm([0.01] * 4 + [0.9] * 4)[:4], 0.08)
    try:
        holm_confirmatory([0.01] * 4)
    except ValueError as exc:
        assert 'confirmatory family is 8' in str(exc)
        note('holm_confirmatory refuses a family of 4', 'ok')
    else:                                                   # pragma: no cover
        raise AssertionError('a family of 4 was accepted as confirmatory')
    hc = holm_confirmatory([0.001, 0.008, 0.04, 0.2, np.nan, 0.3, 0.5, 0.6])
    assert hc.size == CONFIRMATORY_FAMILY_SIZE and np.isnan(hc[4])
    assert abs(hc[0] - 0.008) < 1e-12
    note('and adjusts all 8 slots, a refused cell included as NaN',
         np.round(hc, 6).tolist())
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
    assert km_all['n'] == 6 and km_all['refused'] is False
    note('all censored -> flat curve at 1, no imputed median', 'ok')
    km_none = kaplan_meier([], [])
    assert km_none['refused'] and km_none['n'] == 0
    assert km_none['survival'].size == 0
    note('an empty arm refuses: it is not the same picture as "none reached"',
         'ok')
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
    assert 'orientation only' in lr2['p_role'] and lr2['delayed_entry'] is False
    note('log-rank runs when both arms have >= 3 events',
         f"chi2={lr2['chi2']:.3f} p={lr2['p']:.4f}")
    note('and labels its p orientation only (secondary metric, §1 and §8)',
         'p_role present')
    # Delayed entry, which kaplan_meier had and logrank did not: the same
    # freeze-window data gave a curve with correct risk sets and a statistic
    # with wrong ones.
    lr3 = logrank({'a': ([3.0, 4.0, 5.0, 6.0], [1, 1, 1, 1], [2.0, 0, 0, 0]),
                   'b': ([5.0, 6.0, 7.0, 8.0], [1, 1, 1, 1], [4.0, 0, 0, 0])})
    lr3_no = logrank({'a': ([3.0, 4.0, 5.0, 6.0], [1, 1, 1, 1]),
                      'b': ([5.0, 6.0, 7.0, 8.0], [1, 1, 1, 1])})
    assert lr3['delayed_entry'] and not lr3_no['delayed_entry']
    assert abs(lr3['chi2'] - lr3_no['chi2']) > 1e-9
    note('left truncation changes the risk sets, as it must',
         f"chi2 {lr3_no['chi2']:.3f} without entry, {lr3['chi2']:.3f} with")

    if verbose:
        print('\nOrdered alternative')
    jt = jonckheere_terpstra([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0],
                              [7.0, 8.0, 9.0]], n_perm=2_000, n_boot=300)
    assert jt['J'] == 27.0 and jt['direction'] == 'increasing'
    assert jt['z'] > 2.0 and jt['p_perm'] < 0.05
    assert 'orientation only' in jt['p_role']
    note('perfectly increasing 3x3 -> J = 27 (the maximum)',
         f"z={jt['z']:.3f} p_perm={jt['p_perm']:.4f}")
    assert abs(jt['tau'] - 1.0) < 1e-12 and abs(jt['concordance'] - 1.0) < 1e-12
    note('and tau = 1, the scale-free effect ANALYSIS_PLAN §3 asks for',
         f"tau={jt['tau']:.3f} ci=({jt['ci_lo']:.3f}, {jt['ci_hi']:.3f})")
    jt_flat = jonckheere_terpstra([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0],
                                   [1.0, 2.0, 3.0]], n_perm=500, n_boot=200)
    assert abs(jt_flat['J'] - jt_flat['expected']) < 1e-9
    assert abs(jt_flat['tau']) < 1e-9
    note('three identical groups -> J equals its null mean, tau = 0', 'ok')
    # DESIGN §9: a directional adjective may not affirm a null. An interval
    # covering zero must not render as "increasing", and "flat" is a null
    # dressed as a description.
    assert jt_flat['interval_covers_null']
    assert jt_flat['direction'] == 'not distinguishable from no trend'
    assert jt_flat['sample_direction'] == 'exactly at the null mean'
    note('an interval covering zero renders as "not distinguishable"',
         jt_flat['direction'])
    # tau is scale-free where z is not: the same trend structure at 4 and at 10
    # per group must not move the reported effect the way z does.
    trend_small = [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0],
                   [3.0, 4.0, 5.0, 6.0]]
    trend_big = [[float(i) + k for i in range(10)] for k in range(3)]
    js = jonckheere_terpstra(trend_small, n_perm=500, n_boot=300)
    jb = jonckheere_terpstra(trend_big, n_perm=500, n_boot=300)
    assert abs(js['tau'] - jb['tau']) < abs(js['z'] - jb['z'])
    note('tau moves less than z between 4 and 10 per group',
         f"tau {js['tau']:.3f} -> {jb['tau']:.3f}, "
         f"z {js['z']:.3f} -> {jb['z']:.3f}")
    assert jonckheere_terpstra([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])['refused']
    note('two levels is not an ordered alternative, and is refused', 'ok')
    # The adjective needs BOTH the interval and the exact permutation test.
    # The stratified percentile bootstrap under-covers at RQ5's designed
    # configuration, so the interval alone let a directional adjective through
    # on 6.7 % of iid-null draws (40 of 600) where 5 % is nominal, and on 12 %
    # at k=4, n=4. With the permutation condition the rates are 4.8 % and
    # 3.7 %. Both components are reported, so neither hides inside the word.
    assert jt['interval_excludes_null'] and jt['permutation_rejects']
    assert 'AND' in jt['direction_gate']
    weak = jonckheere_terpstra([[0.0, 1.0, 2.0], [0.5, 1.5, 2.5],
                                [1.0, 2.0, 3.0]], n_perm=2_000, n_boot=500)
    assert weak['p_perm'] > ALPHA, weak['p_perm']
    assert weak['direction'] == 'not distinguishable from no trend', weak
    note('a trend the exact test cannot reject gets no adjective, whatever '
         'the bootstrap interval says',
         f"p_perm={weak['p_perm']:.3f}, "
         f"interval_excludes_null={weak['interval_excludes_null']}")

    if verbose:
        print('\nMinimum detectable effect (ANALYSIS_PLAN §6.2, simulated)')
    # These are regression pins on a fixed RNG stream, not three-decimal
    # knowledge of the estimand, and the tolerance is tight for that reason: at
    # a fixed seed the value is deterministic, so 1e-3 catches any change to
    # this code. It does not license reading the third decimal as a fact about
    # the world. Over four simulation seeds at the same n_sim=20,000 these
    # estimators spread by 0.0044 (sign-flip, alpha=0.05), 0.0095 (sign-flip,
    # Holm), 0.0146 and 0.0135 (Mann-Whitney), i.e. four to fifteen times the
    # tolerance, so the 1.406 pinned below is one draw from an estimator whose
    # four-seed mean is 1.3997. That is agreement with §6.2's corrected 1.41,
    # not a disagreement, and none is asserted here.
    for a, expect, label in ((ALPHA, 1.009, 'paired sign-flip, alpha=0.05'),
                             (HOLM_STRICTEST_ALPHA, 1.535,
                              'paired sign-flip, alpha=0.00625')):
        got = mde_signflip(10, a)
        assert abs(float(got) - expect) <= 1e-3, (label, float(got), expect)
        assert got.method.startswith('exact'), got.method
        assert abs(got.power_achieved - 0.8) <= 0.02, got.power_achieved
        note(f'{label} -> {expect:.3f} sigma',
             f'{float(got):.4f} (power {got.power_achieved:.3f})')
    for a, expect, spread, label in (
            (ALPHA, 1.406, 0.0146, 'Mann-Whitney, alpha=0.05'),
            (HOLM_STRICTEST_ALPHA, 1.880, 0.0135,
             'Mann-Whitney, alpha=0.00625')):
        got = mde_mann_whitney(10, a)
        assert abs(float(got) - expect) <= 1e-3, (label, float(got), expect)
        assert abs(got.power_achieved - 0.8) <= 0.02, got.power_achieved
        planned = 1.41 if a == ALPHA else 1.88
        note(f'{label} -> {expect:.3f} sigma (§6.2 table {planned:.2f}, '
             f'inside this estimator\'s own spread {spread:.4f})',
             f'{float(got):.4f} (power {got.power_achieved:.3f})')

    # ANALYSIS_PLAN §6.5 claims the n=20 MDEs were "verified against
    # statlib.self_test". They were not: nothing here evaluated any n=20
    # quantity. They are now. What that verification shows is agreement to
    # within the simulation's own error, not a disagreement: 0.663 against the
    # plan's 0.662, and 0.894 against 0.890, where this estimator's own
    # seed-to-seed spread is 0.0055 and 0.0037. The rejection-rule fix in
    # `mde_signflip` accounts for +0.0007 and +0.0040 of that, so it is not the
    # explanation either, and no pre-registered value is superseded here. The
    # n=20 row is documentation in any case: the project has settled on n=10.
    if verbose:
        print('\nThe n=20 path to REPLICATE (ANALYSIS_PLAN §6.5)')
    # Two standard errors of this estimator, from its own design rather
    # than from the numbers it happens to produce: the power is a binomial
    # proportion over n_sim=20,000 draws, so SE = sqrt(0.8 * 0.2 / 20000) =
    # 0.0028 at the 0.8 target, and dividing by the local slope of the power
    # curve puts the SE of the returned effect near 0.005 sigma. The plan's
    # figure has to sit inside that band, or one of the two really is wrong
    # and the disagreement is a finding rather than noise.
    mc_band = 0.01
    for a, expect, pinned, label in (
            (ALPHA, 0.663, 0.662, 'paired sign-flip, alpha=0.05'),
            (HOLM_STRICTEST_ALPHA, 0.894, 0.890,
             'paired sign-flip, alpha=0.00625')):
        got = mde_signflip(20, a)
        assert abs(float(got) - expect) <= 1e-3, (label, float(got), expect)
        assert 'Monte Carlo' in got.method and 'add-one' in got.method
        assert abs(float(got) - pinned) <= mc_band, (label, float(got), pinned)
        note(f'n=20 {label} (§6.5 pins {pinned:.3f}, agrees within '
             f'{mc_band:.2f} sigma)',
             f'{float(got):.4f} (power {got.power_achieved:.3f})')
    got20 = mde_mann_whitney(20, ALPHA)
    assert abs(float(got20) - 0.940) <= 1e-3, float(got20)
    assert got20.method == 'exact U null'
    note('n=20 Mann-Whitney, alpha=0.05 (§6.5 pins 0.940)',
         f'{float(got20):.4f} (power {got20.power_achieved:.3f})')

    if verbose:
        print('\nEquivalence and exclusion')
    eq = equivalence_from_ci(-0.02, 0.03, sd=0.043)
    assert eq['verdict'] == 'equivalent' and eq['equivalence_testable']
    note('CI inside +/-0.05 in a quiet cell (SD 0.043) -> equivalent',
         eq['verdict'])
    # §4 makes an equivalence claim available only where the SD is below the
    # margin. An unknown SD has not met that condition, so omitting `sd` must
    # not buy the verdict.
    eq_nosd = equivalence_from_ci(-0.02, 0.03)
    assert eq_nosd['verdict'] == 'untestable'
    assert eq_nosd['geometric_verdict'] == 'equivalent'
    assert 'no across-seed SD' in eq_nosd['reason']
    note('the same CI with no SD supplied -> untestable, not equivalent',
         eq_nosd['verdict'])
    eq2 = equivalence_from_ci(-0.02, 0.03, sd=0.369)
    assert eq2['verdict'] == 'untestable'
    assert 'not below the margin' in eq2['reason']
    note('same CI, the noisy dueling cell (SD 0.369) -> untestable',
         eq2['verdict'])
    # The cell §4 wrote the rule for: SD 0.369 makes the CI wide, so the
    # geometry lands on inconclusive, and gating only the `equivalent` branch
    # meant the pre-committed untestability statement was never produced there.
    eq_noisy = equivalence_from_ci(-0.40, 0.30, sd=0.369)
    assert eq_noisy['verdict'] == 'untestable'
    assert eq_noisy['geometric_verdict'] == 'inconclusive'
    assert eq_noisy['reason'] and 'untestable' in eq_noisy[
        'equivalence_statement']
    note('the noisy cell with a wide CI now states untestability, per §4',
         eq_noisy['equivalence_statement'][:52] + '...')
    eq3 = equivalence_from_ci(-0.30, -0.10, sd=0.369)
    assert eq3['verdict'] == 'not_equivalent'
    assert abs(eq3['worst_degradation_excluded'] - 0.30) < 1e-12
    assert abs(eq3['exclusion_bound'] + 0.10) < 1e-12
    note('CI entirely below -margin -> not_equivalent, whatever the SD',
         eq3['sentence'])
    eq4 = equivalence_from_ci(-0.40, 0.05, sd=0.043)
    assert eq4['verdict'] == 'inconclusive'
    assert abs(eq4['worst_degradation_excluded'] - 0.40) < 1e-12
    note('inconclusive still yields the exclusion sentence', eq4['sentence'])
    assert equivalence_from_ci(np.nan, np.nan)['verdict'] == 'refused'
    note('a refused interval excludes nothing, and says so', 'ok')
    rev = equivalence_from_ci(0.30, -0.30)
    assert rev['verdict'] == 'refused' and 'reversed' in rev['reason']
    note('a reversed interval refuses instead of being silently reordered',
         'ok')
    # The one function whose output is the paper's headline sentence must not
    # drop the interval's provenance, and a zero-width interval quantifies no
    # uncertainty at all.
    degenerate = bootstrap_ci([0.7] * 10)
    assert degenerate.fallback and degenerate.lo == degenerate.hi
    eq_deg = equivalence_from_ci(degenerate)
    assert eq_deg['verdict'] == 'refused'
    assert eq_deg['ci_method'] == 'percentile'
    assert eq_deg['ci_fallback'] == degenerate.fallback
    note('a degenerate zero-width bootstrap CI refuses, carrying its fallback',
         eq_deg['reason'][:52] + '...')
    real_ci = bootstrap_ci([0.10, 0.14, 0.09, 0.21, 0.16, 0.11, 0.19, 0.13,
                            0.12, 0.18], n_boot=2_000)
    eq_ci = equivalence_from_ci(real_ci, sd=0.04)
    assert eq_ci['ci_provenance_known'] and eq_ci['ci_n'] == 10
    assert eq_ci['ci_method'] == real_ci.method
    note('passing the CI object keeps .method and .n in the verdict',
         f"{eq_ci['ci_method']}, n={eq_ci['ci_n']}")
    assert equivalence_from_ci(0.1, 0.2)['ci_provenance_known'] is False
    note('and two loose floats record that the provenance is unknown', 'ok')
    # The level in the sentence is read off the interval. It used to be the
    # literal "95 %" whatever the interval was, so a 50 % interval produced a
    # tighter-sounding bound under a stronger-sounding label.
    wide = bootstrap_ci([0.10, 0.14, 0.09, 0.21, 0.16, 0.11, 0.19, 0.13,
                         0.12, 0.18], n_boot=2_000, alpha=0.50)
    eq_wide = equivalence_from_ci(wide, sd=0.04)
    assert eq_wide['ci_alpha'] == 0.50 and eq_wide['confidence_pct'] == 50.0
    assert '50 %' in eq_wide['sentence'] and '95 %' not in eq_wide['sentence']
    assert eq_wide['ci_alpha_source'] == 'interval'
    assert eq_wide['ci_alpha_is_pre_registered'] is False
    note('a 50 % interval is narrated at 50 %, not at 95 %',
         eq_wide['sentence'][:58] + '...')
    # §4 assesses equivalence with the 95 % interval, so a narrower one may
    # not deliver that verdict however comfortably it sits inside the margin.
    narrow = equivalence_from_ci(-0.02, 0.03, sd=0.043, alpha=0.50)
    assert narrow['geometric_verdict'] == 'equivalent'
    assert narrow['verdict'] == 'untestable'
    assert 'not the pre-registered' in narrow['reason']
    note('and cannot buy the §4 equivalence verdict', narrow['verdict'])
    at_plan = equivalence_from_ci(-0.02, 0.03, sd=0.043, alpha=ALPHA)
    assert at_plan['verdict'] == 'equivalent'
    assert at_plan['ci_alpha_source'] == 'argument'
    assert '95 %' in at_plan['sentence']
    note('at the pre-registered level the verdict is unchanged',
         at_plan['verdict'])
    assumed = equivalence_from_ci(-0.02, 0.03, sd=0.043)
    assert assumed['ci_alpha_source'] == 'assumed'
    assert 'assumed' in assumed['sentence']
    note('two floats with no alpha say the level was assumed, not observed',
         'ok')
    clash = equivalence_from_ci(wide, sd=0.04, alpha=ALPHA)
    assert clash['verdict'] == 'refused' and 'was asserted for' \
        in clash['reason']
    note('asserting an alpha the interval was not built at is refused', 'ok')

    if verbose:
        print(f'\n{len(checks)} checks passed.')
    # `passed` and `failed` are the keys `validate.py --full` reads. They were
    # absent, so its note printed "None checks passed" and its `not
    # rc.get('failed')` assertion was vacuously true whatever happened here.
    # Every failure above is an AssertionError, so reaching this line means
    # nothing failed: the list is empty by construction, not by omission.
    return {'checks': len(checks), 'passed': len(checks), 'failed': [],
            'failures': [], 'detail': checks}


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
    'MIN_N_FOR_INFERENCE', 'MIN_N_FOR_CONFIRMATORY',
    'PIPELINE_VALIDATION_LABEL',
    'EXACT_SIGNFLIP_MAX_N', 'MDE_SIM_REPLICATES', 'SIGNFLIP_MDE_N_PERM',
    'LOGRANK_P_ROLE',
    'CI', 'MDE', 'Estimate',
    'sign_flip_test', 'signflip_min_attainable_p',
    'signflip_attainable_p_below',
    'wilcoxon_signed_rank', 'mann_whitney', 'mwu_exact_null',
    'mwu_critical_values', 'mwu_min_attainable_p',
    'mwu_asymptotic_p', 'mwu_min_attainable_u', 'mwu_max_attainable_u',
    'mwu_asymptotic_min_attainable_p',
    'mwu_asymptotic_critical_values',
    'hodges_lehmann', 'relative_effect', 'brunner_munzel', 'rank_biserial',
    'within_seed_correlation',
    'bootstrap_ci', 'paired_bootstrap',
    'holm', 'holm_confirmatory', 'benjamini_hochberg', 'holm_thresholds',
    'multiplicity_ledger',
    'clopper_pearson', 'proportion_reached', 'kaplan_meier', 'logrank',
    'jonckheere_terpstra',
    'mde_signflip', 'mde_mann_whitney',
    'equivalence_from_ci', 'self_test',
]


if __name__ == '__main__':
    sys.exit(main())
