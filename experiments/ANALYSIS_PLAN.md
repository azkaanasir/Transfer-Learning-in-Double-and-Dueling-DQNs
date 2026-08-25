# Pre-Registered Analysis Plan

**Status:** pre-registration. Written **before** any confirmatory run.
**Companions:** [`DESIGN.md`](DESIGN.md) (what is measured) ·
[`STANDING_INSTRUCTIONS.md`](STANDING_INSTRUCTIONS.md) ·
[`REVIEW_COVERAGE.md`](REVIEW_COVERAGE.md)

This file is hashed, and the hash is written into every run manifest and every
emitted table and figure. Changing it after a confirmatory launch is permitted
only with an entry in §11, and every affected result is then re-labelled
**exploratory**. `audit.py` compares the hash in a manifest against the current
file and refuses to report a confirmatory result under a changed plan.

The organising principle: **one confirmatory family, everything else
estimation.** This is not modesty. At n=10 the achievable minimum detectable
effects (§6, computed exactly, not asserted) are large enough that spending the
error budget on more than one family would leave nothing powered at all. The
published paper's central failure was reading a non-significant result as a
positive finding; the structural defence is to test very little and estimate
everything, with intervals that state what the data exclude.

---

## 1. Endpoints and their roles

| Role | Metric | Testable? |
|---|---|---|
| **Co-primary** | `final_score` (P1): normalised score over 100 held-out greedy episodes at each of the final k=3 checkpoints, averaged | yes, confirmatory |
| **Co-primary** | `auc_score` (P2): area under the normalised-score curve over env steps, per step | yes, confirmatory |
| Secondary | `jumpstart`, `probe_jumpstart`, `steps_to_threshold`, `episode_length`, `within_run_sd`, `across_seed_sd` | estimation-only, **no p-values** |
| Descriptive | `train_return`, `td_loss`, epsilon trace, `updates`, wall time | never tested; `stats.py` refuses |
| Mechanism | `v_abs_mean`, `a_abs_mean`, `a_spread`, `grad_norm_*`, `q_mean`, `td_error_abs`, `cka_transfer_vs_scratch`, `cka_drift`, `dead_unit_frac` | estimation-only, used to support or refuse mechanism wording |

All scores are normalised as in `DESIGN.md` §5.1, against per-environment
random-policy references measured in [`reference_returns.json`](reference_returns.json).

---

## 2. The confirmatory family: the only one

**Exactly 8 tests**: the within-cell transfer effect
`delta = transfer − scratch` for each of the 4 cells
{mlp, dueling} x {vanilla, double}, on each of the 2 co-primary endpoints.

Nothing else in the study is confirmatory. Not RQ1's between-cell comparisons,
not RQ3's interaction, not the control contrasts, not any ablation, not any
secondary metric. Those are all estimation-only, and the paper will say so as a
deliberate design decision forced by the sample size: not as an omission.

* **alpha** = 0.05, two-sided, family-wise, controlled by **Holm–Bonferroni over
  the 8 tests**.
* **Primary test: exact sign-flip randomisation test on the per-seed paired
  deltas**, statistic = the mean delta, enumerating all 2^10 = 1024 sign
  assignments. Exact, distribution-free, and it uses the matched-seed structure
  the design creates (`DESIGN.md` §8.1).
* **Reported alongside, pre-specified, not selected after the fact:** Wilcoxon
  signed-rank (paired) and Mann–Whitney U (unpaired) for the same contrast.
  Mann–Whitney is the test the reviewers endorsed and the published paper used,
  so it is always reported for comparability; it is not the primary here because
  the design is matched and the unpaired test discards that structure at a real
  cost in power (§6).
* **Point estimate and interval:** Hodges–Lehmann estimate of the paired shift,
  with a bias-corrected seed-level bootstrap 95 % CI (10,000 resamples, fixed
  seed 20260824).

### 2.1 Why paired is primary, and the honesty condition attached to it

Pairing is justified by construction, not by hope: at a given seed the scratch
and transfer runs share their per-layer weight initialisation for every
non-transferred layer, the environment-reset sequence, and the evaluation seed
streams (`DESIGN.md` §8.1). What pairing **cannot** remove is the source run's
own outcome and the post-divergence trajectory, so it is a partial block, not a
complete one.

Because that is an empirical question, the plan commits in advance:

* the within-seed correlation `rho(scratch, transfer)` is **reported** for every
  cell, whatever it is;
* the paired test is primary regardless of the observed `rho`: the decision is
  fixed here so it cannot be made after seeing which test gives a smaller p;
* if `rho < 0` in a cell, that is reported as evidence the pairing does not hold
  there, and the unpaired result is given equal prominence in that cell.

### 2.2 What the confirmatory bar actually is, in plain terms

At n=10, the exact sign-flip test's smallest attainable two-sided p-value is
**2/1024 = 0.00195**, obtained exactly when all ten seeds' deltas share a sign.
Under Holm over 8, the strictest comparison is against 0.05/8 = 0.00625.

> **Therefore: a cell's transfer effect is confirmed if and only if all ten
> seeds move in the same direction.** Anything less cannot clear the corrected
> threshold at this sample size.

That is a demanding and completely transparent criterion, and stating it up
front is what stops a marginal result from being narrated into a finding.

---

## 3. Estimation-only analyses, and how they are reported

Every analysis below gets a point estimate and a seed-level bootstrap 95 % CI,
and **no p-value at all**. Where a directional claim is wanted, the licensed
form is what the interval excludes.

| Analysis | Estimator | Notes |
|---|---|---|
| RQ1 between-cell scratch comparison | Brunner–Munzel relative effect `theta = P(X>Y)` with a bootstrap-t CI | Preferred over Hodges–Lehmann because the cells' SDs differ by up to 8x on the normalised scale, which violates the location-shift assumption HL requires |
| RQ3 between-cell contrast `delta_X − delta_Y` | difference of paired deltas, joint seed bootstrap | MDE ~1.96 sigma_delta (§6). Explicitly underpowered; reported as an interval |
| RQ3 2x2 interaction | interaction contrast on deltas, joint seed bootstrap | Same. Also reported on the normalised *and* headroom-adjusted scales, with agreement required before any wording is used |
| The three control contrasts (§4 of `DESIGN.md`) | joint seed bootstrap over the per-seed vector (C0, C1, C2, C3, C3b) | One resampling for all contrasts, so their correlations are estimated rather than ignored. Pairwise correlations are reported |
| RQ5 shift gradient | ordered-alternative trend statistic (Jonckheere–Terpstra) reported as a standardised effect with a bootstrap CI | On the **wind** family as primary; gravity secondary with the difficulty caveat of `DESIGN.md` §5.1 |
| RQ6 budget | paired delta at the 500-prefix vs the 1000 endpoint, same runs | Valid only because the exploration schedule never reads the budget; `validate.py` asserts that |
| Dispersion | ratio of SDs with a bootstrap CI; Brown–Forsythe reported for continuity | No dispersion p-value is interpreted: at n=10 with an SD ratio of ~3 the test has almost no power, which is the honest explanation of the published Brown–Forsythe null |
| Ablation screens (E4, E5, E6, E7, E8, E12) | per-level estimates with CIs; **Benjamini–Hochberg q-values reported for orientation only** | Pre-committed: a screen result is never asserted as a finding; it selects at most one follow-up, which is then run on `REPLICATE` seeds and reported as a fresh estimate |
| Source competence covariate | delta regressed on source normalised score, slope with bootstrap CI | Descriptive relationship, not a mediation claim |

---

## 4. Equivalence and exclusion claims

The claim the published paper needed, "DDQN avoids negative transfer", is an
equivalence claim, and a non-significant difference is not evidence for it.
Revision 1 of the design proposed TOST, which is wrong twice over: it is
parametric on data declared non-normal (LunarLander returns are structurally
bimodal, crash versus land), and at n=10 it cannot support a small margin.

**The procedure.** Equivalence is assessed by whether the **95 % bootstrap CI on
the paired delta lies entirely inside the margin**, using the same interval
already reported. No separate test, no new error budget.

**The margin**, fixed here: **±0.05 normalised-score units**, which is ~20
return points on LunarLander: one tenth of the distance from a random policy to
the solved threshold. Justification is substantive rather than convenient: 20
points is smaller than the run-to-run measurement noise of a single evaluation
and far smaller than the effects the paper discusses (the published transfer gap
was ~357 points, i.e. ~0.89 score units).

**Per-cell feasibility, computed in advance** (from published SDs, §6): a
±0.05 margin is 1.17 SD in the low-variance cells, where n=10 is marginally
sufficient, and 0.14 SD in the high-variance dueling scratch cell, where it is
hopeless. So:

> An equivalence claim is available **only in cells whose observed
> across-seed SD is below 0.05 score units**, and the plan pre-commits to
> reporting, per cell, either an equivalence verdict *or* the statement that the
> cell's dispersion makes equivalence untestable at this sample size.

**The fallback, which is always available and always reported:** the exclusion
bound. From the same CI, the paper states *"a degradation worse than X score
units is excluded at 95 % confidence"*. This is a powered, honest, directional
claim, and it is the form the abstract will use.

---

## 5. Censored metrics

`steps_to_threshold` is right-censored at the training budget, and the censoring
is administrative: the same budget for every run, independent of the event time
by construction, which is the benign case.

* **Never** impute the budget (biases the estimate and creates a tie mass that
  degenerates rank tests); **never** drop censored runs (conditions on the
  outcome, and reintroduces the silent-seed-dropping defect).
* **Primary summary:** P(threshold reached within budget), estimated as k/n with
  a **Clopper–Pearson** exact interval. At 0/10 this reports an upper bound of
  0.31, which is the informative statement rather than a p-value.
* Kaplan–Meier curves per arm, with delayed entry at the end of the freeze
  window where a freeze is in force.
* Log-rank only when both arms have at least 3 events; otherwise the proportion
  and its interval stand alone. This is pre-committed so the choice is not made
  after seeing the censoring rate.
* Thresholds are pre-declared at normalised scores {0.25, 0.5, 1.0}, so that a
  metric exists even when no run reaches "solved".

---

## 6. Power and minimum detectable effects

Computed exactly, from the exact null distribution for n=10 vs 10 and by
simulation for power (20,000 replicates, normal shift). These are the numbers
that justify §2's single-family decision; they are not decoration.

### 6.1 Critical values, n = 10 vs 10

| Test | alpha | Rejection region | Implied dominance |
|---|---|---|---|
| Mann–Whitney U (unpaired) | 0.05 | U <= 23 or U >= 77 | 77 of 100 cross-pairs |
| Mann–Whitney U | 0.0125 (Holm over 4) | U <= 17 or U >= 83 | 83 % |
| Mann–Whitney U | 0.00625 (Holm over 8) | U <= 14 or U >= 86 | 86 % |
| Mann–Whitney U | - | smallest attainable two-sided p = **1.08e-5** | |
| Sign-flip / Wilcoxon (paired) | - | smallest attainable two-sided p = **0.00195** | all 10 deltas same sign |

### 6.2 Minimum detectable effect at 80 % power

| Test | alpha = 0.05 | alpha = 0.00625 (Holm over 8) |
|---|---|---|
| Paired sign-flip on deltas | **1.00 sigma_delta** | **1.54 sigma_delta** |
| Unpaired Mann–Whitney | 1.39 sigma | 1.87 sigma |

The 1.00 versus 1.39 gap is why the paired test is primary: at this sample size
the matched-seed design reduces the detectable effect by about 28 % --
equivalently, the unpaired test needs an effect about 39 % larger to reach the same power (1.41 against 1.01 sigma), which is the largest single power gain available to us.

### 6.3 Translated into score and return units

Using across-seed SDs from the published arms, rescaled onto the normalised
score (random-policy reference −202.39, denominator 402.39):

| Published arm | normalised score | across-seed SD (score) |
|---|---|---|
| mlp/double scratch | 0.928 | 0.093 |
| dueling/vanilla scratch | 0.659 | **0.369** |
| dueling/vanilla transfer | 0.142 | 0.048 |
| mlp/double transfer | 1.030 | 0.043 |

| Test / alpha | MDE, quiet cell | MDE, noisy cell |
|---|---|---|
| paired, 0.05 | 0.043 score (17 return pts) | 0.371 score (149 pts) |
| paired, Holm over 8 | 0.066 score (26 pts) | 0.570 score (230 pts) |
| unpaired, 0.05 | 0.059 score (24 pts) | 0.512 score (206 pts) |
| unpaired, Holm over 8 | 0.080 score (32 pts) | 0.692 score (278 pts) |

**Consequences, stated before the fact.** In the low-variance cells the design
detects effects of a few tens of return points, which is ample: the published
transfer effects were hundreds of points. In the high-variance dueling scratch
cell the MDE approaches or exceeds the whole distance from random play to
"solved", so **that cell is not powered for a modest effect and will be reported
as an estimate with an interval**, whatever the p-value says. Which cells are
powered is therefore known now, not discovered later.

### 6.4 Scope and update rule

These SDs come from the published runs, which used a different protocol, a
different budget and a coupled exploration schedule, so they are a planning
input, not a prediction. The pilot's observed SDs will be reported next to
these. **The power table is not re-tuned after seeing confirmatory results**;
if the pilot shows materially different dispersion, that is recorded in §11
before the confirmatory launch, or not at all.

### 6.5 The path to n = 20, pre-registered now

The `REPLICATE` block (seeds 10–19) exists so that the sample size can be
doubled without a post-hoc decision. Pre-committed rule:

* If `REPLICATE` is run, the **pooled n=20 analysis becomes the primary** and
  the n=10 result is reported beside it. The decision to run `REPLICATE` is made
  on compute availability only, never on the n=10 outcome, and the run order is
  recorded.
* No interim look at `REPLICATE` data is used to decide whether to continue, so
  no alpha adjustment for sequential testing is required. If that discipline is
  ever broken, it is recorded in §11 and the affected results become
  exploratory.
* At n=20 the minimum detectable effects, computed the same way and verified
  against `statlib.self_test`, are:

  | Test | n=10 | n=20 |
  |---|---|---|
  | paired sign-flip, alpha=0.05 | 1.009 | **0.662** |
  | paired sign-flip, Holm over 8 (0.00625) | 1.535 | **0.890** |
  | unpaired Mann-Whitney, alpha=0.05 | 1.406 | **0.940** |

  Doubling the sample therefore cuts the confirmatory MDE from 1.54 to 0.89
  sigma_delta, which brings the between-cell RQ3 contrast (whose standard error
  is sqrt(2) times a single delta's) within reach of being testable rather than
  estimation-only. That is the main scientific reason to run `REPLICATE`, and
  it is stated here so the decision is not made on the basis of an n=10 result.

---

## 7. Multiplicity ledger

Printed by `stats.py` on every run, so the count is a recorded fact rather than
a claim.

| Family | Members | Procedure | Adjusted alpha |
|---|---|---|---|
| **Confirmatory** | 8 (4 cells x 2 co-primary endpoints) | Holm–Bonferroni | step-down from 0.00625 |
| Screens (E3–E8, E12) | reported per experiment | Benjamini–Hochberg q, orientation only | no assertion permitted |
| Everything else | - | none: estimation-only | no p-values emitted |

Family membership is fixed by this document **before launch**. `stats.py` reads
the family definitions from here rather than accepting them as arguments, which
is what prevents a result from being rescued by relocating it into a family of
one.

---

## 8. Analyses explicitly forbidden

Each of these was either done in the published paper or was a live temptation in
the design, and each is refused in code.

| Forbidden | Why |
|---|---|
| Any t-test, Cohen's d, or normality-assuming interval on returns | The published §V.A did exactly this on a metric its own §V.B declared descriptive-only and non-normal |
| Reading a non-significant result as equivalence | §4 gives the only licensed route |
| The sentence form "A avoids negative transfer while B does not" derived from two separate significance verdicts | That is a comparison of verdicts, not a test. The licensed form is the between-cell contrast with its interval |
| A p-value on any secondary, mechanism or ablation quantity | Not in the confirmatory family |
| Dropping a seed, for any reason, after it has run | Completeness is asserted by `audit.py` |
| Any estimate computed on `TUNE` seeds | Selection leakage |
| Comparing raw returns across environments or variants | Scale differs by hundreds of points; use the normalised score |
| Choosing paired vs unpaired after seeing the results | Fixed in §2 |
| Re-deriving the equivalence margin after seeing the CI | Fixed in §4 |

---

## 9. Pilot / validation output at n = 1

Single-seed runs exist to prove the pipeline executes (`STANDING_INSTRUCTIONS`
S8). Under n < 3, `stats.py` emits no test and no interval, and `report.py`
stamps every page **PIPELINE VALIDATION: NOT A RESULT**. A single-seed number
may not be quoted, compared, or used to choose between hypotheses.

---

## 10. What gets reported, in order

1. Audit result. If the audit fails, nothing below is emitted without an
   explicit override that is stamped into the output.
2. Run inventory: cells, conditions, seeds, completeness, source-validity
   verdicts and exclusions, transferred-parameter fractions.
3. Reference returns and the normalisation used.
4. Descriptives per arm: normalised score mean, SD, median, bootstrap CI, plus
   scratch mean, threshold and **headroom**.
5. Convergence gate: per-arm fraction of runs whose final-200-episode slope is
   distinguishable from zero.
6. **The confirmatory family**: 8 tests, raw and Holm-adjusted p, the HL
   estimate and CI, the observed within-seed correlation, and the three tests'
   agreement.
7. Equivalence or exclusion statement per cell, per §4.
8. The three control contrasts with joint intervals and correlations, plus the
   C2-at-K=0 comparison that tests the mechanics term's freeze dependence.
9. C4 positive control against its pre-registered criterion.
10. Estimation-only sections: RQ1, RQ3, RQ5, RQ6, ablations, mechanism signals.
11. Multiplicity ledger and the list of analyses that carry no p-value.
12. Deviations from this plan, if any.

---

## 11. Deviations from this plan

| Date | Deviation | Affected results | Re-labelled? |
|---|---|---|---|
| - | none | - | - |
