# Experimental Design — Transfer in Value-Based Deep RL

**Status:** authoritative specification. Revision 2 (2026-08-24), after an
adversarial review of revision 1 that found four fatal defects; see §11.
**Companions:** [`STANDING_INSTRUCTIONS.md`](STANDING_INSTRUCTIONS.md) ·
[`ANALYSIS_PLAN.md`](ANALYSIS_PLAN.md) (pre-registered inference) ·
[`EXPERIMENTS.md`](EXPERIMENTS.md) (operational catalogue) ·
[`REVIEW_COVERAGE.md`](REVIEW_COVERAGE.md) (reviewer traceability) ·
`paper/METHODS_ACTUAL.md` · `paper/LITERATURE.md`

This document fixes *what is measured, against what counterfactual, under what
invariants, and what result would refute the thesis*. Code is downstream of it.
Nothing here changes after a confirmatory experiment launches without an entry
in §12 and a re-labelling of the affected results as exploratory.

---

## 1. What went wrong before, stated as design constraints

The published study failed on six specific points (`paper/METHODS_ACTUAL.md`).
Each becomes a constraint the infrastructure enforces *mechanically*:

| Failure | Constraint |
|---|---|
| Arms differed in architecture **and** Q-target rule while claiming one isolated variable | The two are orthogonal factors; every cell of the 2x2 exists |
| Transfer arm used lr=1e-4 against a baseline's 5e-4 under a claim of "identical hyperparameters" | Each experiment declares **invariants**; `audit.py` refuses to aggregate runs that violate them |
| The manuscript's freeze schedule was never implemented | The freeze schedule is indexed in **gradient updates**, logged as events with trainable-parameter counts, and *verified* by weight fingerprints |
| A source agent never learned its source task (26.94 where the task is solved at 475) | Source validity is gated on a **normalised** score, with a reserve-seed rule and full reporting of exclusions |
| One seed was dropped from one arm with no stated rule | Seed-set completeness is asserted; partial arms are refused |
| "Positive transfer" claimed from p=0.421 | A null may only be reported as *not distinguishable*; the licensed alternative is to report **what the interval excludes** |

---

## 2. Thesis, hypotheses, and scope

### 2.1 Scope statement — inherited by every claim

> All effects are defined over the finite seed set actually run, for the stated
> `(arch, target_rule)` implementations at `hidden=(128,128)`,
> `head_units=64`, Adam, the declared exploration schedule and episode budget,
> on the named environment pairs. **No claim is made about the dueling
> decomposition or the double-Q update as algorithmic ideas**, nor about deep RL
> transfer in general. Where a sentence in the report would generalise past
> this, `report.py` prefixes it with this clause or refuses to emit it.

### 2.2 Thesis

> Under a fixed layer-wise transfer protocol between discrete-action value-based
> agents on classic-control tasks, the transfer effect is governed by the
> **Q-target rule** and by **knowledge-free protocol mechanics** — partial weight
> copy, head reinitialisation and the freeze schedule — rather than by the
> **dueling value/advantage decomposition**. Any architecture-level
> DDQN-versus-Dueling transfer gap measured without a 2x2 factorial and without
> untrained- and permuted-source controls is therefore **not identified**.

The scope clause in §2.1 is part of the thesis, not a caveat appended to it.
Wang et al. motivate the dueling decomposition by many-action, similar-return,
pixel-scale states; LunarLander has four actions, an 8-dimensional observation
and a dense shaped reward, so a null dueling effect here says little about that
inductive bias in the regime it was designed for.

**Two results would refute the thesis**, and both are pre-registered:

1. **The architectural reading is right after all.** The dueling marginal effect
   on the within-cell delta is non-null and survives the C2/C3 controls, while
   the Q-target marginal effect and the mechanics term (C2 − C0) are both null.
   The paper is then rewritten around a dueling-specific mechanism.
2. **There is no transfer effect to explain.** All four cells' deltas are
   indistinguishable and every decomposition term is null, yet a large
   between-cell gap persists. The published gap was target-task suitability, not
   transfer — ICANN #5's Q1 vindicated in full.

Both are publishable, and the second is what the published framing was most at
risk from.

**What this thesis does *not* claim, and why** (`paper/LITERATURE.md` §3):
negative transfer in the DQN family, failed self-transfer, and a head/body
mismatch mechanism are **established prior art** — Sabatelli & Geurts
demonstrated all three, including by transplanting a head between agents. The
scratch comparison across three of these four cells is also prior art, at n=100,
on these very environments (Obando-Ceron & Castro). The contribution here is
narrower and is *identification*: separating two axes the prior work varies
together, and separating learned content from protocol mechanics with controls
that prior work does not run. Claiming discovery of the phenomenon would be
defeated by one search.

### 2.3 Hypotheses, each with a refutation condition

Directions are stated in **normalised score** units (§5.1), so they are
comparable across environments. `mechanics`, `weight-statistics` and
`structure` name the three contrasts in §4 — as contrasts, not as proven
mechanisms.

| # | Hypothesis | Refuted if |
|---|---|---|
| **H1** | The untrained-source contrast (C2−C0) is non-zero and negative in at least 3 of the 4 cells: freezing a copied-but-uninformative trunk for the protocol's window costs performance by itself | The contrast's interval covers zero in 2 or more cells, or is positive in any cell |
| **H2** | The trained-vs-permuted contrast (C1−C3) — the only quantity that can be called transferred knowledge — is **smaller in magnitude** than the untrained-source contrast (C2−C0) in the cross-interface pair | \|C1−C3\| exceeds \|C2−C0\| in 2 or more cells. That would mean learned structure dominates mechanics, and the thesis in §2.2 is wrong |
| **H3** | The cell-to-cell spread in the total effect (C1−C0) shrinks once transferred-parameter fraction is matched across architectures (§3.1) | The spread is unchanged or larger at matched intensity, which would mean the architecture effect is real and not an intensity artifact |
| **H4** | The transfer effect degrades monotonically with measured dynamics shift along the **wind** axis, at fixed interface and fixed protocol | No monotone trend (ordered-alternative test's interval covers zero), or a non-monotone pattern |
| **H5** | The interface-change corner (same dynamics, changed observation/action interface) reproduces most of the cross-interface effect | It reproduces little of it, which would relocate the cause to dynamics shift and refute the mechanics-first thesis |
| **H6** | For dueling cells, freezing the **value** stream and freezing the **advantage** stream produce different effects | They do not differ, which removes the dueling-specific mechanism and makes the architecture axis uninformative about mechanism |

**Every hypothesis above can come out the other way, and several would be more
interesting if they did.** H2 and H3 in particular are set up so that the
result the paper's previous framing assumed — a genuine architectural effect —
is the *refuting* outcome, and it would be reported as such.

### 2.4 Research questions

Each names its estimand, its counterfactual, and the inference type it can
support. **Inference type is binding on wording.**

| RQ | Question | Estimand | Inference and its warrant |
|---|---|---|---|
| **RQ1** | On the target task from scratch, how do the four cells compare? | Normalised final score per cell | **Associational.** Cells are different algorithms, not treatments assigned to units |
| **RQ2** | Does the transfer protocol change performance relative to *the same cell's own* scratch baseline? | delta = transfer − scratch, within cell, per seed | **Causal w.r.t. the protocol**, warranted by *ceteris paribus*: every non-protocol factor is held bit-identical at matched seeds. Seed is a **blocking factor, not a randomisation device** — the warrant is a controlled computational experiment, not randomisation |
| **RQ3** | Is delta attributable to the dueling decomposition, the double-Q update, or their interaction? | Between-cell contrasts on delta, and the 2x2 interaction | **Effect modification** — how a causal effect varies across populations. Not "architecture causes the difference". **Estimation-only**: its MDE is ~2.7 sigma (`ANALYSIS_PLAN` §6), larger than the plausible effect, so it carries an interval and no p-value |
| **RQ4** | Which part of the transferred parameters produces delta, and how much is not learned content at all? | The three contrasts of §4, plus layer-set and freeze ablations | **Causal w.r.t. each manipulated component**, under the exclusion restrictions named in §4 |
| **RQ5** | How does delta vary with the magnitude and the *type* of shift? | delta as a function of measured shift, crossed with interface change (§6.4) | **Causal w.r.t. shift level and interface change** (both set by us); associational across shift *families* |
| **RQ6** | Does any conclusion depend on the training budget? | delta at episode prefixes 500 / 1000 | **Causal w.r.t. budget**, and only because the exploration schedule is a closed-form function of elapsed steps that never reads the budget — so a 500-prefix *is* what a 500-episode run would have produced. That identifying condition is asserted in `validate.py`. The comparison is made **like with like**: the 500-prefix score is a single held-out checkpoint, so it is compared against the *single* final checkpoint (recorded per checkpoint in the manifest), never against the three-checkpoint mean |

### 2.5 The two confounds this design exists to separate

1. **Transferability vs target-task suitability.** The published comparison was
   cross-architecture return under transfer, which cannot separate "transfer
   hurt this architecture" from "this architecture is worse on LunarLander
   regardless". RQ2's within-cell delta separates them. This is ICANN #5's Q1.
   *Residual, and stated:* comparing deltas **across** cells (RQ3) re-admits a
   scale confound, because a cell whose scratch baseline is near the ceiling has
   less headroom to gain and more to lose. §5.1's normalisation and the
   headroom column in every table address it; agreement across both scales is
   required before any RQ3 claim is asserted.
2. **Shift severity vs interface mismatch.** CartPole -> LunarLander changes the
   observation dimension (4->8) *and* the action cardinality (2->4) *and* the
   dynamics *and* the reward structure, simultaneously. §6.4 crosses the two
   factors so each corner exists, including the corner nobody has run: **same
   dynamics, changed interface.**

---

## 3. Factors

| Factor | Levels | Notes |
|---|---|---|
| `arch` | `mlp`, `dueling` | architecture axis |
| `target_rule` | `vanilla`, `double` | Q-target axis, orthogonal to `arch` |
| `condition` | `scratch`, `transfer`, `transfer_untrained`, `transfer_permuted` | the protocol axis (§4) |
| `transfer_set` | `matched` (default), `trunk`, `fc1`, `fc2`, `all_compatible` | see §3.1 — **not** a raw layer list, because a raw list is not comparable across architectures |
| `freeze_updates` | 0, 5k, 10k, 20k, 50k, `inf` | **gradient updates, not episodes** (§3.2) |
| `aggregation` | `mean`, `max`, `naive` | dueling only |
| `env_pair` | §6 | source -> target |
| `epsilon_anneal_episodes` | default 900, and one faster level | promoted to a factor so budget and exploration horizon are not confounded. **Episodes, not steps** — see §3.2 |
| `lr`, `target_update` | screening grid, E3 only | §3.3 fixes the tuning policy |
| `seed` | disjoint blocks, §3.4 | never re-used across selection and estimation |

Everything unlisted is an **invariant** and is machine-checked (§8.4).

### 3.1 Transferred-parameter fraction must be matched, not assumed

Revision 1 held `transfer_layers = (trunk_fc1, trunk_fc2)` fixed across
architectures and called that "the same protocol". It is not. Measured on the
actual networks at 8-dim observations and 4 actions:

| Cell | trunk-only transfer | fraction of model | layers reinitialised |
|---|---|---|---|
| `mlp` | 17,664 of 18,180 | **97.2 %** | 1 (`q_out`) |
| `dueling` | 17,664 of 34,501 | **51.2 %** | 4 (both heads) |

So the `arch` factor was confounded with treatment intensity by a factor of
about two — **the same class of error the Phase 0 audit found in the published
study, reconstituted inside the corrected design.** The fix has three parts:

1. `transfer_set='matched'` is the default and the primary level: the maximal
   **shape-compatible** set per architecture — `mlp`:
   {`trunk_fc1`, `trunk_fc2`}; `dueling`: {`trunk_fc1`, `trunk_fc2`,
   `value_fc`, `adv_fc`, `value_out`} — which puts both arms near 97 %.
2. `transfer_set='trunk'` is retained as a **pre-declared secondary**, because
   it is what the published protocol did and the comparison is informative.
3. `transferred_param_fraction` and `reinitialised_layer_count` are recorded in
   every manifest and printed in every results table, and `audit.py` refuses a
   cross-`arch` contrast whose fractions differ beyond a declared tolerance
   unless the contrast is explicitly labelled intensity-confounded.

### 3.2 Schedules are indexed in updates, not episodes

Every learning quantity is step-indexed while every schedule in revision 1 was
episode-indexed, and LunarLander episode length is strongly
performance-dependent: measured here, a random policy runs 94 steps at gravity
−10 and 183 steps at gravity −4. An episode-indexed freeze window therefore
means a different amount of *learning* in every arm, so E4's levels would not
have been comparable. Consequently:

* the freeze window is `freeze_updates` (gradient updates);
* `epsilon` is a closed-form function of the **episode index**, evaluated at the
  top of each episode, and **not** coupled to the evaluation cadence — in the
  published code the decay lived inside the evaluation branch, which made the
  exploration schedule a function of `eval_every`;

**Why exploration is indexed on episodes while freezing is indexed on updates.**
Revision 2 of this document indexed *both* on steps, and the exploration half of
that was wrong. It was corrected after measurement, not after argument, and the
measurement is worth recording because the failure mode is not obvious.

A step-indexed exploration horizon is **endogenous to policy quality**. A poor
policy ends episodes quickly, so few environment steps accumulate, so epsilon
barely anneals, so the policy stays poor. On CartPole the loop closed: a
1000-episode run delivered **24,708 environment steps against a 300,000-step
horizon** — 8.2 % of it — so epsilon fell only from 1.000 to **0.684**, the agent
explored at 0.7–1.0 throughout, and **all four sources failed the validity gate**
(scores 0.18–0.49 against a 0.60 threshold). The same schedule was healthy on
LunarLander, which delivered 248,762 steps and reached epsilon 0.022, which is
exactly why a single global step horizon is the wrong instrument: it was
calibrated on the environment where episode length is long and applied to one
where it is short.

Episodes are the exogenous unit. An episode is a trial, and the number of trials
is fixed by the budget rather than by how well the agent is doing. A geometric
anneal to the floor at episode 900 reproduces the published schedule (0.95 per
ten episodes, floor at episode 898) to within 0.001 at every episode, so the
correction also improves comparability with the published runs. Re-measured, a
CartPole source now reaches a normalised score of **0.772** and 127,502 steps,
and clears the gate.

The freeze window keeps its step indexing, because its argument is different and
survives: it concerns how much *optimisation* has been applied to the trainable
subset, and gradient updates are the unit of that.

The identifying condition for RQ6 is unaffected. Epsilon still never reads
`num_episodes`, so a 500-episode prefix of a longer run has exactly the
exploration schedule a 500-episode run would have had; `validate.py` asserts it.
* evaluation uses a **separate environment instance**, so evaluation cannot
  touch training state;
* `env_steps` and `gradient_updates` are logged per episode and carry the
  sample-efficiency claims.

`validate.py` asserts the epsilon trace is bit-identical when `eval_every` and
`eval_episodes` change. That assertion is the test of §8.1's central claim.

### 3.3 Tuning policy — both objections answered, neither fudged

Two commitments appear to collide: "identical hyperparameters across arms,
machine-checked" (the control claim) and "a fair per-architecture baseline"
(ICANN #5 Q1/Q5, which requires that the arms *not* share one learning rate).
Both are honoured, in a declared order:

* **Primary — common configuration.** One learning rate and target-update rule
  for all four cells, fixed a priori at `lr=5e-4`, hard target update every
  1000 updates. Invariance is enforced across every cell. This is the setting
  in which "identical hyperparameters" is a verified fact.
* **Secondary — per-cell tuned.** Each cell's own E3-selected configuration.
  Under this policy `lr` is an invariant *within* a cell across
  {scratch, transfer, C2, C3} but not across cells, and `audit.py` enforces it
  at that scope.
* **Pre-registered arbitration:** an RQ2 or RQ3 conclusion is asserted only if
  it holds under **both** policies. Where they disagree, that disagreement is
  the finding, and it is reported as one.

### 3.4 Seed blocks are disjoint by construction

Revision 1 selected hyperparameters on seeds 0–4 and then ran confirmatory arms
on 0–9, so half of every confirmatory sample was tuned on. Blocks now:

| Block | Seeds | Used for | Never used for |
|---|---|---|---|
| `TUNE` | 200–204 | E3 hyperparameter selection | any confirmatory or reported estimate |
| `CONFIRM` | 0–9 | every confirmatory arm | selection |
| `REPLICATE` | 10–19 | independent replication; pooling to n=20 | selection |
| `C4SRC` | 300–309 | positive-control source checkpoints | target-side estimation |
| `RESERVE` | 400+ | replacement sources when the validity gate rejects one | anything else |

`audit.py` fails if a reported estimate draws on `TUNE`, and `plan.py` prints
the block of every scheduled run.

---

## 4. The control set, and the three contrasts

Four conditions per (cell, pair) at matched seeds.

| Condition | Source weights | What it manipulates |
|---|---|---|
| **C0 `scratch`** | none | the cell's own target-task ability — the denominator for every delta |
| **C1 `transfer`** | trained same-cell source, same seed | the protocol as studied |
| **C2 `transfer_untrained`** | randomly initialised source of the same shape | the protocol's *mechanics* with no learned content: same partial copy, same reinitialised head, same freeze window |
| **C3 `transfer_permuted`** | trained source, each transferred kernel independently shuffled entry-wise | the trained weights' *marginal distribution* — norm and scale preserved exactly, structure destroyed |

### 4.1 The identity, and what it is not

```
(C2 − C0) + (C3 − C2) + (C1 − C3)  =  C1 − C0
```

**This is a telescoping arithmetic identity. It holds for any four numbers and
is shown only to fix notation.** It is not evidence of additivity, not a
decomposition of a causal effect, and nothing about it is testable. Revision 1
called it "an additive decomposition, each term estimable at n=10", which
implied an empirical finding where there is none. The three contrasts are named
after **what was manipulated**, never after a mechanism:

| Contrast | Name used everywhere | Mechanistic reading requires assuming |
|---|---|---|
| C2 − C0 | untrained-source contrast | that a random source of matched shape carries no task-relevant content — safe |
| C3 − C2 | permuted-source contrast | that shuffling changes nothing but structure. Preserved exactly: the multiset of weights, hence the Frobenius norm. **Not** preserved: per-row and per-column norms and the singular-value spectrum, so this contrast also absorbs spectral effects |
| C1 − C3 | trained-vs-permuted contrast | that the permutation removed all and only the learned structure |

Because C3 − C2 does not preserve the spectrum, a **spectrum-matched control
(C3b)** is included for the primary pair: a random matrix matched to the source
layer's singular-value spectrum. If C3 and C3b agree, the spectral caveat is
empirically void; if they disagree, the disagreement is reported rather than
assumed away.

The no-interaction assumption is not asserted. It is **partly tested**: C2 runs
at both `freeze_updates=0` and the protocol value, so the mechanics contrast's
dependence on freezing is measured. Estimation is **joint** — the per-seed
vector (C0, C1, C2, C3) is bootstrapped over seeds so all contrasts and their
correlations come from one resampling, rather than four independent two-sample
tests that ignore the shared groups.

### 4.2 C4 — positive control, redefined so it can fail informatively

Revision 1's positive control used the same environment for source and target,
which means matched shapes, hence a full copy with no partial copy and no head
reinitialisation — so it exercised **none** of the mechanics under study, and it
had no pass criterion.

C4 now uses the **interface-change-only** pair of §6.4: LunarLander to
LunarLander with the observation padded and the action set extended, so the
dynamics are identical by construction while the partial copy, the head
reinitialisation and the freeze window all run exactly as in E1. Source
checkpoints come from the disjoint `C4SRC` block, so the C4 deltas are
independent of E1's.

**Pass criterion, pre-registered:** the Hodges–Lehmann estimate of the paired
delta has a 95 % bootstrap CI whose lower bound exceeds −0.10 in normalised-score
units. Failure means the protocol degrades performance even with no dynamics
shift at all — which would not invalidate the study, but would make "negative
transfer" the wrong name for the finding, and the paper would say so.

### 4.3 Source validity — normalised, and not called ITT

A multiplicative gate on raw return is neither sign- nor origin-safe: at
Acrobot's registered threshold of −100, "0.6 x threshold" is −60, which is
*harder* than solving the task. Measured random-policy return there is −497.
So validity is defined on the normalised score of §5.1:

> A source is **valid** when its normalised final score >= 0.6.

Revision 1 called the all-seeds analysis "intent-to-treat". That framing is
wrong: source competence is known *before* the target run begins, so it is not
a post-randomisation compliance event, and averaging over it would mean the
primary estimand pools transfer-from-a-competent-source with
transfer-from-a-source-that-never-learned — the published study's actual error.
Therefore:

* **Primary: valid sources only**, with source seeds drawn in order from
  `RESERVE` until the cell has its full complement of valid sources. The number
  and identity of rejected source seeds appear in the results table.
* **Secondary: pooled over source competence**, labelled exactly that, never
  "ITT".
* Source competence is also retained as a **continuous covariate**, and the
  delta-versus-competence relationship is reported as an estimate.

---

## 5. Metrics

### 5.1 Normalised score — the scale everything is expressed on

```
score = (return − random_return_env) / (threshold_env − random_return_env)
```

`random_return_env` is measured once per environment *and per variant* from 100
fixed-seed uniform-random episodes and stored in
[`reference_returns.json`](reference_returns.json). Random policy scores 0 and
the registered threshold scores 1, by construction.

This is not cosmetic. Measured across the LunarLander gravity family, the
random-policy return moves from −202 to −463 and the score denominator from 402
to 663, so **raw returns are not comparable across variants** and a raw delta
would silently mix a scale change into a shift effect. It also makes the
source-validity gate sign-correct for Acrobot, and makes effects comparable
across environment pairs.

`noop_return` is recorded alongside, and it exposes something that changes the
design: the no-op policy's score rises from 0.18 at gravity −10 to **0.55** at
gravity −4, while staying flat near 0.17–0.18 across all wind levels. Weakening
gravity therefore makes the task *easier* as well as different, so the gravity
family confounds shift severity with task difficulty. **The wind family is the
primary shift axis for H4; gravity is secondary and reported with this caveat
attached.**

### 5.2 Co-primary endpoints

Two, because a transfer claim is about both the rate of acquisition and the
end point, and revision 1's single terminal snapshot was the endpoint *least*
sensitive to transfer.

| # | Endpoint | Definition |
|---|---|---|
| **P1** | `final_score` | mean normalised score over 100 held-out greedy episodes, at each of the final **k=3** evaluation checkpoints, averaged. The multi-checkpoint mean is what removes the "phase of oscillation at episode 1000" variance component, which is a property of the measurement moment rather than of the algorithm — the pilot in `METHODS_ACTUAL` §6 shows a solve-then-destabilise pattern that a single snapshot would fold into the mean. Evaluation seeds are disjoint from training and from the monitoring evaluations |
| **P2** | `auc_score` | area under the normalised-score evaluation curve, **over env steps**, divided by total env steps. Budget-free, and the sample-efficiency endpoint that a transfer claim actually needs. Built from the periodic 5-episode evaluations, whose initial states are drawn **independently at each checkpoint**: that adds variance to any single point but leaves the area unbiased, which is the right trade for an average over ~100 points |

Holm over the confirmatory family, which is these two endpoints x the four
within-cell deltas (`ANALYSIS_PLAN` §5). A **convergence gate** is reported
alongside P1: the OLS slope of the score curve over the final 200 episodes per
run, with the fraction of non-converged runs stated. Where runs have not
converged, P1 is named *performance at budget*, not asymptotic performance.

### 5.3 Secondary endpoints — estimation-only, no p-values

| Metric | Definition and the correction it embodies |
|---|---|
| `jumpstart` | 100-episode greedy score **at episode 0, before any gradient step**. Interpretable *only* where the output head is transferred: with a reinitialised head the zero-shot policy is an argmax over a random readout, so jumpstart is structurally at chance and comparing it would be meaningless |
| `probe_jumpstart` | for head-reinit arms: freeze the transferred trunk, fit **only** the head for a declared number of steps on a fixed batch of target transitions, then evaluate. This is the quantity that measures whether transferred features carry usable information |
| `steps_to_threshold` | env steps until the trailing-100 mean score first reaches a declared level, **right-censored** at the budget. Analysed by survival methods; the primary summary is P(reached by budget) with a Clopper–Pearson interval. Never imputed, never dropped |
| `episode_length` | final-100 mean — promised in the manuscript's §I and RQ3 and never reported |
| `within_run_sd` | SD of the score over the final 10 evaluation points: *training* instability |
| `across_seed_sd` | SD of `final_score` across seeds: *seed* sensitivity. Descriptive; a dispersion comparison carries a bootstrap CI on the ratio, never a p-value, because its power at n=10 is negligible |

The published study conflated the last two and then described the result
backwards. They are separate metrics here for that reason.

### 5.4 Descriptive-only (never hypothesis-tested)

`train_return` (final-100 mean, for continuity with the published numbers),
`td_loss` with a stated window, the epsilon trace, `updates`, wall time. The
registry declares these roles and `stats.py` refuses a confirmatory test on
them — the mechanical fix for the published §V.A/§V.B contradiction.

### 5.5 Mechanism instrumentation

Logged on the evaluation cadence from a **fixed diagnostic state batch** drawn
once per run from a dedicated RNG stream, so a diagnostic can never perturb
training and the same states are used at every measurement point.

| Signal | Purpose |
|---|---|
| `v_abs_mean`, `a_abs_mean`, `a_spread` | dueling stream magnitudes: is V mis-scaled to the source's return range? |
| `grad_norm_{trunk,value,adv,head}` | per-stream gradient norms across the freeze boundary — separates an *optimisation* mismatch from a *representational* one, which is what ICANN #5's Q5 turns on |
| `q_mean`, `q_max`, `td_error_abs` | overestimation behaviour: the mechanism the double-Q factor exists to control |
| `cka_transfer_vs_scratch` | linear CKA between the transferred run's `trunk_fc2` activations and the **matched-seed scratch run's**, on the same fixed batch of target states: how different a representation transfer actually produced |
| `cka_drift` | CKA between the trunk at episode 0 and at each evaluation point: how far fine-tuning had to move the transferred features |
| `dead_unit_frac` | trunk units never active on target states — a concrete failure mode for a frozen transferred layer |
| `effective_rank` | entropy-based effective rank of the trunk activations on the fixed batch, plus the stable rank. Feature-rank collapse is the plasticity literature's account of degradation after pretraining, and it is a rival explanation the weight-scale control (C3) does not exclude |
| `param_norm` | L2 norm of the trainable weights, per group. Parameter-norm growth is the other plasticity signature, and it distinguishes "the representation stopped being useful" from "the optimiser drifted into a bad region" |

Revision 1 specified `cka_source_target`, CKA "between activations on source-env
vs target-env states". That is ill-posed: CKA compares two representations of
the *same* examples, so two different input sets give incomparable Gram
matrices — and for CartPole -> LunarLander it is not even computable, the input
dimensionalities differing. Both replacements above are on one fixed target-state
batch.

---

## 6. Environments, pairs, and the shift factorial

### 6.1 Registry

| Env | obs | act | threshold | measured random return | Role |
|---|---|---|---|---|---|
| `CartPole-v1` | 4 | 2 | 475 | 22.5 | source (primary) |
| `Acrobot-v1` | 6 | 3 | −100 | −497.3 | source (alternate) |
| `LunarLander-v3` | 8 | 4 | 200 | −202.4 | target (primary) |
| `LunarLander-v3` variants | 8 | 4 | 200 | −202 to −463 | source and target, same-interface shift |
| `LunarLander-v3` padded/extended | 8+k | 4+m | 200 | measured per variant | **interface change at zero dynamics shift** (§6.4) |
| `CartPole-v1` variants | 4 | 2 | 475 | 22.5 to 32.1 | cheap same-interface shift |
| `MountainCar-v0` | 2 | 3 | −110 | — | **excluded from confirmatory work**: DQN without shaping often fails it, which would reproduce the published invalid-source error |

### 6.2 Same-interface dynamics shift

* **Wind (primary axis for H4)**: `enable_wind` with
  `wind_power` in {0, 7.5, 15}, `turbulence_power` 1.5. The no-op score is flat
  across these levels, so difficulty is held roughly constant while dynamics
  change.
* **Gravity (secondary)**: `gravity` in {−10, −8, −6, −4}. Ordered and
  well-measured, but it changes difficulty as well as dynamics (§5.1), so it is
  reported with that caveat and never used alone to support H4.

Measured paired trajectory divergence confirms both are graded: standardised
separation at step 10 rises 0.00 / 0.12 / 0.25 / 0.37 across the gravity levels,
with median steps-to-separation 70 / 40 / 28.

### 6.3 Shift quantification, and the explicit refusal

The published 2-Wasserstein over *return* statistics is not used: returns are a
consequence of the policy, the reward scale and the horizon cap.

| Pair type | Computed | Not computed |
|---|---|---|
| Same interface | **paired trajectory divergence** — identical initial state and an identical action sequence driven into both environments, so separation is attributable to dynamics alone; validated by a self-check against an identical-environment control that must return exactly zero. Plus per-dimension W2 and energy distance on state-visitation distributions, standardised by the source's own per-dimension spread | — |
| Cross interface | a structured qualitative descriptor, plus the representation measures of §5.5 | **any scalar shift metric.** No distance between different state spaces is defined, and saying so is more defensible than inventing one |

### 6.4 The shift x interface factorial — the missing corner

Revision 1 claimed to separate shift severity from interface mismatch and then
supplied only two corners of the table. All four now exist:

| | **Interface unchanged** | **Interface changed** |
|---|---|---|
| **Dynamics unchanged** | (identity — the null cell, used as a machinery check) | **the new corner**: LunarLander -> LunarLander with the observation padded by *k* uninformative dimensions and the action set extended by *m* duplicate actions. Dynamics identical by construction, yet the identical partial-copy + head-reinit + freeze pipeline runs. Also serves as C4 (§4.2) |
| **Dynamics shifted** | E8: wind and gravity families | E1/E9: CartPole -> LunarLander, Acrobot -> LunarLander |

Additionally, E8's same-interface arm is run in a **matched-protocol** variant
(`transfer_set='trunk'`, head reinitialised, and `first_layer_policy='redraw_matched'`
forcing the same number of freshly drawn kernel entries as the 4->8 case), so
that the same-interface and cross-interface arms differ in shift and interface
rather than in protocol as well.

---

## 7. Experiment catalogue

Run counts are for the `CONFIRM` block (n=10) and include the scratch
counterfactual each delta requires, which revision 1 omitted.

| ID | Name | Tier | Family | Design | Runs |
|---|---|---|---|---|---|
| `E0` | `smoke` | — | estimation | tiny end-to-end validation of every code path, one seed | 7 |
| `E1` | `core2x2` | 1 | **confirmatory** | 4 cells x {CartPole source, LunarLander scratch, transfer at `matched`, transfer at `trunk`} | 160 |
| `E2` | `controls` | 1 | estimation | 4 cells x {C2 untrained, C2 at K=0, C3 permuted, C3b spectrum-matched} plus their source and scratch prerequisites | 240 |
| `E3` | `hpsens` | 1 | screen | 4 cells x lr{1e-4,3e-4,5e-4,1e-3} x update{hard,soft}, LunarLander scratch, on `TUNE` | 160 |
| `E8i` | `interfaceonly` | 1 | estimation | 4 cells x padded/extended LunarLander x {scratch, transfer}; doubles as control C4, sources from `C4SRC` | 120 |
| `E4` | `freezedur` | 2 | screen | 4 cells x `freeze_updates` in {0, 5k, 10k, 20k, 50k, never} | 320 |
| `E5` | `layerset` | 2 | screen | 4 cells x transfer sets {fc1, fc2, trunk, matched, described} | 280 |
| `E6` | `streamfreeze` | 2 | screen | dueling cells x freeze {none, trunk, value, adv, heads} | 140 |
| `E7` | `aggregation` | 2 | screen | dueling cells x {mean, max, naive} x {scratch, transfer} | 180 |
| `E8` | `shiftaxis` | 2 | estimation | 4 cells x {wind 2 levels, gravity 3 levels} x {scratch, transfer}, protocol-matched to E1 | 440 |
| `E11` | `valuerecal` | 2 | estimation | dueling cells x V-output recalibration {center, center_scale} | 80 |
| `E9` | `envpairs` | 3 | estimation | 4 cells x {Acrobot->LL, CartPole->Acrobot, LL->CartPole}, each with its own scratch denominator | 360 |
| `E12` | `capacity` | 3 | screen | 4 cells x hidden {64, 256} x {source, scratch, transfer} | 240 |
| `E13` | `plasticity` | 3 | estimation | 4 cells x {head reset at unfreeze, shrink-and-perturb} -- the plasticity rival explanation | 160 |

Counts are per experiment run **alone**, at its declared seed block. They are
generated from `registry.py`, so the table cannot drift from what the runner
would execute; `plan.py` is authoritative and reports the de-duplicated figure
for any selection.

Because a run is keyed by its configuration digest rather than by experiment,
identical configurations are **shared**: the naive per-experiment sum is
2887 runs and the de-duplicated total is **2047**. Tier 1 alone is
**600 runs**, which the measured throughput puts at roughly 36 h at
`--jobs 4`.

`E10` (budget) appears nowhere in this table because it costs nothing: it is a
re-evaluation of E1's episode-500 prefix checkpoints.

`E10` is free: a 500-episode prefix of a 1000-episode run *is* a 500-episode
run, because the exploration schedule never reads the budget (§2.4 RQ6).

`E8i` is Tier 1 despite being new, because H5 depends on it and because it is
the only cell that isolates interface mechanics.

`E13 plasticity` (Tier 3) exists because the plasticity-loss literature supplies
a complete, architecture-free rival explanation for poor performance after
pretraining — dead units, parameter-norm growth, feature-rank collapse — that the
control set does not exclude. C3 controls weight *scale*; nothing in it measures
rank collapse. §5.5 now instruments effective feature rank and parameter norm,
and E13 adds the reset and shrink-and-perturb arms that the literature would
demand as the comparison.

**Pre-registered external-validity check.** RQ1's scratch comparison across three
of these four cells has already been run at n=100 on these environments
(Obando-Ceron & Castro, ICML 2021, adding Rainbow components to DQN one at a
time). Our scratch ordering on LunarLander must agree with theirs in *direction*.
It is declared here, before running, that a disagreement is a finding about our
hyperparameter regime rather than about transfer, and that it would be reported
as such with E3 as the diagnostic. RQ1 is therefore a **sanity check, not a novel
result**, and the write-up says so.

---

## 8. Infrastructure requirements

### 8.1 RNG discipline

Named, independently derived streams per run, spawned from one
`SeedSequence(seed)`: `init`, `action`, `buffer`, `env_reset`, `eval_monitor`,
`eval_final`, `diag`, `control`. Adding a diagnostic, changing the number of
evaluation episodes, or altering a control condition cannot perturb the
training trajectory of the parts it does not touch — an assertion
`validate.py` tests rather than a property the design asserts.

Weight initialisation is seeded **per layer**, from a deterministic function of
(seed, layer name). One consequence is deliberate and worth stating: at a given
seed, an `mlp` and a `dueling` network share their trunk initialisation
exactly, and the two `target_rule` levels share everything. The 2x2 is
therefore **matched by seed**, which reduces nuisance variance and is what
justifies the paired analysis in `ANALYSIS_PLAN`.

### 8.2 Resumability without data corruption

1. On resume from a checkpoint at episode *k*, `metrics.csv` is **truncated** to
   episodes < *k* before appending. Without this a crash between checkpoints
   duplicates rows and silently corrupts every window statistic downstream.
   This defect exists in the current `train.py` and is fixed.
2. Resume is refused when the stored config hash differs from the requested one.
3. Agent, buffer and environment RNG states are all checkpointed. The current
   `ReplayBuffer.save` writes an RNG state that `load` never restores; fixed.

`validate.py` asserts resume-equivalence: a run interrupted and resumed produces
the same metrics as an uninterrupted one, to the tolerance the platform allows.
Where bitwise determinism is unattainable, the achieved tolerance is **measured
and reported**, not promised.

### 8.3 Provenance

Every manifest records the git commit and dirty flag, package versions,
platform and CPU, the resolved config and its hash, the `ANALYSIS_PLAN` hash,
the derived seed per stream, the exact argv, freeze events with
trainable-parameter counts and weight fingerprints, the transfer report with
`transferred_param_fraction`, the source-validity verdict and score, the
reference returns used for normalisation, and timing. Figures and tables record
the hash of the CSV they were built from.

### 8.4 Machine-checked invariants (`audit.py`)

"Identical hyperparameters" must be verified, not asserted. `audit.py` fails on:
a differing invariant across an experiment's runs; a missing declared seed; an
incomplete arm; a missing source-validity verdict; a config hash that does not
match its config; a frozen layer whose fingerprint moved; a trainable layer
whose fingerprint did not; a cross-`arch` contrast with mismatched transferred
fraction; any reported estimate touching a `TUNE` seed. Aggregation and
reporting refuse to run on a failed audit unless overridden, and the override is
stamped into the output.

### 8.5 Cost control

`plan.py` prints run count, measured throughput, projected wall-clock per
`--jobs`, disk, and the seed block of every run, before anything launches.

---

## 9. Anti-fallacy guardrails, enforced in code

| Fallacy | Guard |
|---|---|
| Affirming a null | A non-significant result renders as *not distinguishable*; the licensed positive statement is what the interval **excludes**, with the exclusion bound printed |
| Cross-architecture return presented as a transfer effect | The primary estimand is the within-cell delta; between-cell contrasts are labelled RQ1/RQ3 and carry the headroom columns automatically |
| Comparing two significance verdicts and calling it a comparison | The between-cell contrast delta_X − delta_Y is an explicit estimand with an interval. The report template forbids the "A avoids it, B does not" sentence form |
| Effect attributed to learned representations without excluding mechanics | C2/C3 are Tier 1; the three contrasts are emitted with every delta |
| Mechanism claimed from prose | A mechanism claim must cite an instrumented signal (§5.5); the report template has no free-text mechanism slot |
| Descriptive metric used inferentially | Registry declares metric roles; `stats.py` enforces them |
| Directional adjective contradicting the numbers | Dispersion and direction sentences are generated from the data |
| Multiplicity ignored | Exactly one confirmatory family, declared before launch, Holm within it. Everything else estimation-only. The test ledger is printed |
| Selection bias in tuning | Disjoint seed blocks (§3.4), enforced by `audit.py` |
| Treatment intensity mistaken for architecture | `transferred_param_fraction` reported; matched-intensity contrast is primary (§3.1) |
| Raw returns compared across environments of different scale | Everything is on the normalised score (§5.1) |
| Censored data imputed or dropped | Survival analysis; P(reached) with an exact interval |
| Silent seed dropping | Seed-set completeness asserted |
| Invalid source treated as valid | Normalised gate, reserve-seed rule, exclusions reported |
| Generalising past the evidence | §2.1 scope clause is inherited by every emitted claim |
| Stale artifacts | Provenance hashes on every figure and table |

---

## 10. Known limitations, stated up front

1. **Several questions are not powered at n=10.** The RQ3 interaction's minimum
   detectable effect is roughly 2.7 sigma, larger than any plausible effect.
   RQ3, RQ5's gradient and the dispersion comparisons are therefore
   estimation-only by design, not by omission. `ANALYSIS_PLAN` §6 states the
   MDE for every contrast, and `REPLICATE` exists so n can be doubled to 20
   under a pre-registered pooling rule rather than a post-hoc one.
2. **Equivalence cannot be established at n=10** for any substantively small
   margin. The claim "avoids severe negative transfer" is therefore made, if at
   all, as an exclusion: *a degradation worse than X is excluded at 95 %*.
3. **Two toy environments, one network width, one optimiser.** §2.1 applies.
4. **The gravity family confounds shift with difficulty** (§5.1). Wind carries
   H4; gravity corroborates.
5. **The permuted-source contrast does not preserve the spectrum.** C3b bounds
   that; if they disagree the caveat stands.
6. **Source-side transfer is one-directional per pair.** The reverse-direction
   arm probes asymmetry but does not make the study symmetric.
7. **The phenomenon is not ours.** Negative transfer, failed self-transfer and
   head/body mismatch are established (Sabatelli & Geurts); the scratch cell
   comparison is established at n=100 (Obando-Ceron & Castro). The contribution
   is identification, and the write-up must concede this early rather than be
   caught by it.
8. **A null dueling effect is a power result as much as an absence.** At n=10
   with a Holm-corrected 2x2 interaction, the thesis's central negative claim
   rests on intervals, not on a null p-value. §10.1 and §10.2 apply, and the
   honest statement is an exclusion bound.
9. **The plasticity account is instrumented but not fully excluded.** Effective
   rank and parameter norm are measured; the reset comparison is Tier 3. If E13
   does not run, the write-up must say the rival explanation is measured but not
   experimentally excluded.
10. **The control conditions are not novel in themselves.** Random- and
    shuffled-source baselines exist in the transfer literature. The claim is
    their first use inside a head-structure x target-rule factorial in the DQN
    family — not the controls.
11. **The interface-change corner is not perfectly difficulty-neutral.**
    Surfaced by the generated environment table rather than by argument: the
    padded/extended variant's no-op score is 0.25 against the base
    environment's 0.18. The *dynamics* are provably identical — the wrapped
    environment reproduces the base trajectory bit-identically under aliased
    actions, and `validate.py` asserts it — but extending the action set from 4
    to 6 by aliasing actions 0 and 1 doubles their probability under a uniform
    random policy, so the random-policy *reference* shifts. Two consequences,
    both stated rather than absorbed: the normalisation denominator differs from
    the base environment's, and RQ5's estimand for this corner is the
    *within-variant* delta, where the shared denominator cancels. Comparing this
    corner's delta against another variant's therefore uses the scale-free
    effect size, as §5.1 already requires for the shift axes.

---

## 11. What revision 1 got wrong

Recorded because the review that found it is the reason to trust revision 2 more
than revision 1, and because two of the four defects were the *same class* of
error the Phase 0 audit found in the published paper.

| # | Defect in revision 1 | Fix |
|---|---|---|
| 1 | "Same protocol" transferred 97 % of the mlp and 51 % of the dueling net, confounding `arch` with treatment intensity | §3.1 matched transfer sets; fraction reported and audited |
| 2 | Hyperparameters selected on seeds 0–4 and confirmatory arms run on 0–9, so selection leaked into every headline number; and the design never said whether arms shared one lr or were tuned per cell | §3.4 disjoint blocks; §3.3 explicit dual tuning policy with pre-registered arbitration |
| 3 | Six research questions and zero hypotheses — every RQ was phrased so any outcome confirmed it | §2.3, six hypotheses with directions and refutation conditions |
| 4 | The primary endpoint was a terminal snapshot, the endpoint least sensitive to transfer, with no convergence check; all sample-efficiency metrics were demoted | §5.2 co-primary endpoints, multi-checkpoint, in env steps, with a convergence gate |
| 5 | The "additive decomposition" was an arithmetic identity presented as an empirical finding | §4.1 stated as an identity; contrasts renamed after manipulations; joint bootstrap |
| 6 | The positive control exercised none of the mechanics it was meant to validate, and had no pass criterion | §4.2 rebuilt on the interface-change-only pair, with a non-inferiority criterion |
| 7 | Episode-indexed schedules against step-indexed learning, and epsilon coupled to the evaluation cadence | §3.2 update-indexed schedules; closed-form epsilon; assertion in `validate.py` |
| 8 | A multiplicative source-validity gate that was stricter than solving the task on Acrobot | §4.3 normalised gate |
| 9 | Raw returns compared across environment variants whose scale differs by hundreds of points | §5.1 normalised score, measured references |
| 10 | `cka_source_target` was ill-posed and, for the primary pair, not computable | §5.5 two well-posed replacements |
| 11 | Equivalence licensed through a parametric TOST on data declared non-normal, at a sample size that cannot support it | §10.2 exclusion bounds instead |
| 12 | Guardrails captioned "enforced in code" with no such code, and an inference plan deferred to a file that did not exist | `ANALYSIS_PLAN.md`, `audit.py`, `validate.py`, `plan.py` written before any confirmatory launch |

---

## 12. Change log

| Date | Change | Effect on claims |
|---|---|---|
| 2026-08-24 | Revision 1: initial specification. | — |
| 2026-08-24 | Revision 2: the twelve items in §11, following adversarial review. No confirmatory run had launched, so no result is affected. | None — nothing had been run |
| 2026-08-24 | **Revision 5, forced by P0.** The exploration schedule was indexed on environment steps, which makes the horizon endogenous to policy quality; on CartPole a 1000-episode run delivered 8.2 % of the 300,000-step horizon, epsilon fell only to 0.684, and all four sources failed the validity gate. Exploration is now indexed on episodes with the floor at 900, which reproduces the published schedule to within 0.001 and restores the sources (score 0.268 -> 0.772). Freezing keeps its step indexing. **This is why P0 exists**: 3 h of compute bought the finding, and every run made under the old schedule is discarded. | None reported — the affected runs were the single-seed validation pass, and no result had been claimed from them |
| 2026-08-24 | Revision 4, from self-audit during implementation: the measurement-cost claim was wrong by an order of magnitude (1–2% asserted, ~22% measured) and is corrected with the measurement; the 250- and 750-episode prefix checkpoints were cut because no question attached to them; RQ6's comparison was made like-with-like (single checkpoint against single checkpoint); `E9` was documented but unimplemented and is now in the registry; `E13` implemented with head-reset and shrink-and-perturb; the catalogue table is generated from `registry.py` so it cannot drift. | None — nothing had been run |
| 2026-08-24 | Revision 3, after the literature audit (`paper/LITERATURE.md`): thesis narrowed and scoped; the two refuting outcomes named; prior art conceded explicitly; effective-rank and parameter-norm instrumentation added against the plasticity rival explanation; `E13` added; RQ1 demoted to a sanity check with a pre-registered external-validity comparison. | None — nothing had been run |
