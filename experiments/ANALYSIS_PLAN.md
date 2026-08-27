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

That floor is not the bar. The exact two-sided p is k/1024 for an even count k
of sign assignments at least as extreme as the observed mean, because an
assignment and its negation are always equally extreme, so the attainable
values step by 2/1024 and **three** of them sit strictly below 0.00625:
0.00195, 0.00391 and 0.00586.

> **Therefore: a cell's transfer effect is confirmed when at most 6 of the 1024
> sign assignments are at least as extreme as its observed mean.** All ten
> seeds moving the same way attains the floor and is therefore SUFFICIENT. It
> is **not** necessary: one seed moving against the other nine by a small
> enough margin leaves four assignments at least as extreme, p = 0.00391, which
> still clears 0.00625.

That bar is the strictest Holm step, and Holm applies it only to the smallest p
in the family of eight; the jth smallest is compared against 0.05/(9-j), so
every later step is looser still. The verdict recorded for a cell is its
Holm-adjusted p against 0.05.

That is a demanding and completely transparent criterion, and stating it up
front is what stops a marginal result from being narrated into a finding.

---

### 2.3 The E3 selection rule, pre-registered before E3 was analysed

`DESIGN.md` 3.3 makes an RQ2 or RQ3 conclusion assertable only where the
common-configuration and per-cell-tuned policies **agree**, and names the
secondary policy as "each cell's own E3-selected configuration". Neither document
said *selected how*. Whoever chooses that rule chooses how easily the two
policies agree, so it is fixed here, in full, before any E3 output was
aggregated.

**Inputs.** `E3` trains the four cells across `lr` in {1e-4, 3e-4, 5e-4, 1e-3}
crossed with target update in {hard, soft}, on LunarLander scratch, at the five
`TUNE` seeds 200 to 204. Eight configurations per cell, five seeds each.

**Criterion.** Mean `auc_score` across the five `TUNE` seeds. Not final score:
on LunarLander every cell finishes above the solved threshold, so a final-score
comparison is a comparison of ceilings and cannot express an ordering. This is
the same reasoning, and the same endpoint, that `DESIGN.md` 7 already fixed for
the external-validity check.

**`var` is the ddof=1 sample variance.** Named explicitly because at n=5 the
unbiased and population estimators differ by 25 per cent in the variance and
about 12 per cent in the standard error, which is enough to move a marginal
cell, and because a rule that inherits whichever a library happens to default
to is not a rule.

**Rule.** For each cell independently, let *A* be the a priori configuration
(`lr` = 5e-4, hard update every 1000 updates) and *B* the configuration with the
highest mean AUC. Select

> *B* if `mean_AUC(B) - mean_AUC(A) > SE`, where
> `SE = sqrt(var_A/5 + var_B/5)` is the standard error of that difference across
> the five `TUNE` seeds. **Otherwise select *A*.**

Ties in "highest mean AUC" are broken, in order, by: higher mean final score,
then lower `lr`, then `hard` before `soft`. The rule is therefore deterministic
and reproducible from the stored table.

**Why conservative rather than argmax.** At n=5 the `TUNE` block cannot resolve
small differences. A plain argmax would chase noise and invent per-cell
differences that do not exist; those would then appear as a disagreement between
the two policies and, under 3.3's arbitration, would **block an RQ2 conclusion
the data actually support**. Requiring one standard error of separation means the
secondary policy departs from the primary only where there is evidence, which is
what the fair-baseline objection asks for and no more. The bar is one SE, not
two: two would switch almost never and make the arbitration a formality that
always passes, which is a different way of being uninformative.

**Refusals.** Four, and the first was stated wrongly in the first draft of this
section, which is recorded in 11.

* **Incompetent cell.** A cell is tunable only if its selected candidate reaches
  a **mean normalised `final_score` of at least 0.6** across the five `TUNE`
  seeds. Otherwise the selection refuses for that cell rather than returning its
  least-bad option. Note the floor reads on **final score, not AUC**: AUC
  integrates the curve from zero and so never reaches 1.0 even for a solved
  task, as `DESIGN.md` 7's pilot table shows (best cell `auc_score` 0.9640
  against `final_score` 1.1787), so a floor applied to AUC would refuse every
  cell. The value 0.6 is not new: it is the competence floor `DESIGN.md` 4.3
  already declares for source validity, reused rather than a second constant
  invented here.
* **Incomplete cell set.** Fewer than four cells is refused.
* **Incomplete candidate.** Every candidate must be measured at all five `TUNE`
  seeds. A candidate measured at fewer is compared against its rivals on
  unequal evidence, and the standard error above is written for n=5. If a
  `TUNE` run is lost, the fix is to re-run it, not to shrink the denominator.
* **Wrong block.** A selection computed from any seed outside `TUNE` is refused,
  because 8 forbids a reported estimate drawing on the selection block.

**Scope of the selected configuration.** It applies to the target-task
conditions `DESIGN.md` 3.3 enumerates, {scratch, transfer, C2, C3}, on the
selection environment. The **source runs are not retuned**: `E3` selects on
LunarLander scratch and says nothing about CartPole, retuning a source would
confound a policy disagreement with source quality, and `DESIGN.md` 4.3's
replacement ledger keys on the source arm's label. Invariance is therefore
enforced at the scope (`arch`, `target_rule`, `env`).

**What this rule is not.** It is not a claim that the selected configuration is
optimal. At five seeds it is a coarse instrument, and the tuned policy is a
**robustness condition on asserting a conclusion**, not a tuned result in its own
right. No number produced under it is reported as a performance claim.

### 2.4 The arbitration, and why it adds no tests

`DESIGN.md` 3.3 asserts an RQ2 or RQ3 conclusion only where both policies agree.
Four things that rule needs in order to be executable, fixed here.

**The confirmatory family stays at exactly eight members.** The tuned leg runs
the same eight contrasts under the other policy and adds **no** family members,
because the arbitration is a **conjunction**: asserting only where both legs
reject makes the rejection region the *intersection* of the two legs' regions,
which is never larger than either. The family-wise error rate therefore stays
bounded by Holm over eight, and 7's ledger continues to report eight. Treating
the tuned leg as eight further tests would inflate the ledger and change the
pre-registration for no inferential gain.

**Agreement includes direction.** Two rejections pointing in *opposite*
directions are a **disagreement**, not an agreement, because a direction is part
of the conclusion being asserted.

**A replicated null is assertable.** Where neither leg rejects and both
intervals sit inside the equivalence margin, the null replicates under both
policies and is assertable as such. That is what licenses 4's exclusion bound;
without it the arbitration would permit no negative conclusion at all.

**Three verdicts, and one of them blocks.** Per cell and endpoint: `agrees`,
`disagrees`, or `not-evaluable`. A conclusion may be asserted only under
`agrees`. Under `disagrees` the disagreement **is** the reported finding and may
not be suppressed, averaged away, or resolved by preferring one policy. Under
`not-evaluable`, which is the state whenever the tuned arms are absent,
incomplete, drawn from a corrupt selection, or drawn from a selection computed
under a placeholder rule, **nothing is asserted**. `not-evaluable` is the
default, so an unrun tuned stage cannot silently license a conclusion.

**Every consumer must honour it.** A downstream artifact may not present a
confirmatory conclusion from the significance flag alone. `stats.py` records the
verdict and an `asserted` flag on every confirmatory member, and `report.py`,
`tables.py` and `plots.py` must read those rather than re-deriving licence from
the p-value. A guard the consumer ignores is not a guard.

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
| Unpaired Mann–Whitney | 1.41 sigma | 1.88 sigma |

The unpaired row read 1.39/1.87 until 2026-08-26. That was a transcription
error, not a different computation: 6.5's table already pins 1.406 for the same
estimator at the same n and alpha, the paragraph below already quotes 1.41, and
`statlib.mde_mann_whitney(10)` returns 1.406 and 1.880. Corrected here and
logged in 11.

The 1.00 versus 1.41 gap is why the paired test is primary: at this sample size
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

### 6.6 The decision, taken 2026-08-26, before any confirmatory run

**`REPLICATE` will not be run. The study is n=10.**

This is recorded here rather than later precisely because 6.5 requires it to
be made on compute availability only. At the point of writing, no confirmatory
run exists: the only data on disk is the 44-run single-seed validation pass,
which under 9 carries no result. The reason is compute and nothing else. The
selected scope is **E1, E2, E3, E4, E5, E8i and E9 at ten seeds**, 1200 runs and
a measured **125.3 h** at `--jobs 6`, plus the tuned replication of E1 and E2
that `DESIGN.md` 3.3's arbitration requires, estimated at a further **31 h** and
to be re-costed once those arms exist in the registry. About **156 h** in total,
roughly 6.5 days on a single machine that must not sleep for the duration.
Doubling the seed count would double all of it.

**The cost of this decision is accepted and stated in advance.** At n=10 the
confirmatory minimum detectable effect under Holm over eight is **1.53
sigma_delta**, against 0.89 at n=20. The between-cell contrast of RQ3, whose
standard error is sqrt(2) times a single delta's, therefore remains
**estimation-only**: it is reported as a point estimate with a bootstrap
interval and, where the interval permits, an exclusion bound. It is not
tested, and no wording in the manuscript may imply that it was. If the
observed effects turn out to be smaller than 1.53 sigma_delta, the honest
result is an exclusion bound and not a null finding, and 4 governs how it is
phrased.

Reversing this decision later would make the pooled analysis exploratory under
6.5, and would be logged in 11.

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
6b. **The arbitration verdict** of 2.4, per cell and endpoint, with the tuned
   leg's own statistic beside the common leg's, and the `asserted` flag. No
   confirmatory conclusion appears anywhere in the report without it.
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
| 2026-08-27 | **Correction and completion of §2.3, plus new §2.4.** Four independent agents implementing §2.3 reported that it could not be executed as written, and they were right. (a) The refusal clause read the solved-threshold floor on `auc_score`, which integrates from zero and never reaches 1.0 even for a solved task, so it would have refused **every** cell; the floor now reads on mean normalised `final_score` at the 0.6 competence value `DESIGN.md` 4.3 already declares. (b) `var` is now named as ddof=1: at n=5 the two estimators differ by about 12 per cent in the standard error, enough to move a marginal cell. (c) A candidate missing a `TUNE` seed is refused, since the standard error is written for n=5. (d) The selected configuration's scope is stated: target-task conditions only, sources not retuned. §2.4 additionally fixes what 3.3's arbitration left open: the family stays at eight by a conjunction argument, agreement includes direction, a replicated null is assertable, and `not-evaluable` is the default that blocks assertion. | **None.** No selection has been computed, no tuned arm has been run, and E3 is being restarted so that its evidence is produced entirely under this version of the plan | n/a |
| 2026-08-27 | **Omission repaired, not a deviation.** §2.3 is new. `DESIGN.md` 3.3 required an E3-selected per-cell configuration and **neither document ever said how to select it**, leaving an open researcher degree of freedom on the primary conclusion, since the arbitration asserts RQ2 only where the two policies agree. The rule is now fixed in full. Written with 51 of E3's 160 runs on disk, **none aggregated and none analysed**; the commit carrying §2.3 timestamps it ahead of any E3 output. The E3 runs are independent of the rule: they train fixed configurations and the rule is an analysis decision applied afterwards. | **None yet.** No selection has been computed and no tuned arm has been run. Note the consequence for provenance: E3 runs produced before this edit carry the previous `ANALYSIS_PLAN.md` hash, so `audit.py` will report a plan-hash split on E3 unless E3 is restarted under the current plan | n/a |
| 2026-08-26 | **Declared departure, not a deviation.** §9 emits no interval below n=3. `statlib.clopper_pearson` does, because the Beta-quantile interval is **exact at any n** and at n=1 it is very nearly [0, 1], which is a statement about ignorance rather than a claim. Suppressing it would replace an honest very-wide interval with nothing, which reads as less uncertainty rather than more. The returned interval carries a `reason` naming the departure and the n, so a caller stamps the pipeline-validation label over it and it cannot be quoted as a result. | None. `proportion_reached` is estimation-only and carries no p-value | n/a |
| 2026-08-26 | **Strengthening, recorded for completeness.** `DESIGN.md` §9's guardrail against a directional adjective affirming a null named the bootstrap interval as the enforcement. `statlib` now gates such wording on the interval **and** an exact permutation test, because the interval alone measured a 6.7% false-direction rate against a nominal 5% at the designed configuration. The spec row therefore understates what the code does. | None. It refuses strictly more wording than the plan requires, never less | n/a |
| 2026-08-26 | **Correction of an error in this plan, not a deviation.** §6.2's table gave the unpaired Mann-Whitney minimum detectable effect as 1.39 sigma at alpha=0.05 and 1.87 sigma under Holm over 8. Both are transcription errors against this plan's own arithmetic: §6.5 pins 1.406 for the identical estimator at the identical n and alpha, §6.2's own paragraph quotes "1.41 against 1.01 sigma", and `statlib.mde_mann_whitney` at n=10 returns 1.406 and 1.880, which round to 1.41 and 1.88. The table now reads 1.41/1.88 and `stats.py`'s `MDE_MULTIPLIERS` matches it, so its agreement check against `statlib` runs at a tolerance of 0.01 rather than the 0.02 that had been widened to admit the one disagreeing row. This is not a re-tuning under §6.4: no confirmatory run exists (§6.6), the correction claims LESS power for the unpaired test rather than more, and it reconciles the plan with itself rather than with any observed result. | None. The unpaired test is not the primary and no MDE is quoted in any result; §6.3's translated table is computed from the paired row, which is unchanged | n/a |
| 2026-08-26 | **Correction of an error in this plan, not a deviation.** §2.2 stated the confirmatory bar as "confirmed if and only if all ten seeds move in the same direction". The "only if" half is false: the exact sign-flip p moves in units of 2/1024 and three attainable values (0.00195, 0.00391, 0.00586) sit strictly below the Holm-strictest 0.00625, so a cell split 9 to 1 with a small opposing delta returns p = 0.00391 and clears the bar without unanimity. §2.2 now states the bar as "at most 6 of the 1024 sign assignments at least as extreme as the observed mean", with unanimity named as sufficient and not necessary. `stats.py` §5b printed the plan's sentence verbatim beside a table that could contradict it and now prints the corrected rule. Made before any confirmatory run exists (§6.6); it changes no computation, only the sentence describing it, and it makes the stated bar LESS demanding than the one written down, so it is recorded here rather than left to be discovered in the results. | None. No confirmatory test has been run; `significant_holm` was already standard Holm and is unchanged | n/a |
| 2026-08-25 | **Amendment, not a deviation.** `DESIGN.md` §7's external-validity check for RQ1 (estimation-only; §10 item 10) named no endpoint. It now fixes AUC, pre-specifies the all-cells-saturated case as uninformative, and records that the external source tunes the learning rate per configuration while we hold it fixed. Made after inspecting n=1 pilot scratch numbers, which under §9 carry no result, and before any confirmatory run exists. | None. RQ1 is outside the confirmatory family and carries no p-value | n/a |
