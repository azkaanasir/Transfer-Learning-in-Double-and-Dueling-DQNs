# Pre-Launch Fix List

**Status:** blocking checklist. Nothing multi-seed launches until the **Blocking**
section is clear and `validate.py` passes on the real tree.

Compiled 2026-08-25 from an independent adversarial verification of all ten
analysis modules (one agent per module, each running the module against the real
44-run P0 tree and `runs_demo`, each instructed to break it). **All ten returned
`needs_fixes`; one returned `fatal`.** Full verdicts:
`scratchpad/verify.json`.

This file exists because the previous verification pass died on a session limit
and I verified my own work instead. That was weaker, and this pass proves it:
it found a defect that had *already fired* in P0 and that self-verification
missed.

---

## Blocking: must be fixed before any multi-seed run

### B1. `sweep.py`: a live run is reclaimed as stale, and two workers train into one directory · **FATAL**

The `.claim` file's mtime is written once and never refreshed. `_reclaim_reason`
returns `stale` on `age > stale_seconds` (default 7200 s) **without checking
whether the owner is still alive**, and the stale branch does not require the
claim to belong to a different sweep. Any run exceeding two hours is therefore
taken over *while still training*.

**It fired in P0.** From `runs/_jobs/status.jsonl`:

```
2026-08-24T18:35:20  w00  claimed    scratch/18b28c089f91/s300   (c4src-mlp-double)
2026-08-25T08:47:09  w02  reclaimed: stale (14.2 h > 2.0 h)  -- same directory
2026-08-25T08:47:10  w02  failed: PermissionError [WinError 5] on metrics.jsonl
2026-08-25T08:58:31  w00  done, wall_time_s 51790.8
```

The same sequence occurred for `scratch/d110c9a81fa8/s00`.

**The data survived only because Windows holds a mandatory lock on the open
`metrics.jsonl`, which killed the interloper in 1.1 s.** On POSIX the
`os.replace` would have succeeded and both trainers would have written the same
metrics, checkpoints and `model.keras` for eleven minutes: producing exactly the
duplicated- and interleaved-episode corruption that `DESIGN.md` §8.2 exists to
prevent.

Verified sound after the collision: all three contested runs carry 1000 unique
contiguous episodes, monotonic wall time, and `metrics_integrity.contiguous`.
**P0's data is valid.** It was saved by the platform, not by the design.

Fix: heartbeat the claim while training; require evidence the owner is dead
(PID liveness) before reclaiming; add an ownership check to `release_claim` and
`fail_claim`; and clean up `.claim.superseded-*` litter (two files remain in
`runs/`).

### B2. `sweep.py`: a failed reclaim is indistinguishable from a failed run

`fail_claim` writes `state='failed'` into the shared claim file, and
`dependency_state` reads it and returns **`blocked`**: a terminal, non-zero-exit
verdict. In P0 this produced a false blockage of
`transfer/572af3c31522/s00`, whose source was healthy and completed eleven
minutes later. The runner manufactured a terminal verdict from its own collision.

### B3. `sweep.py`: an unscored source is not gated; a NaN score reads as a rejection

`source_score()` returns `unscored` for a completed run whose manifest lacks the
score, and phase 2 proceeds against it. Separately, `nan >= 0.6` is `False`, so a
degenerate evaluation is silently treated as a gate rejection rather than as a
measurement failure. Both violate `DESIGN.md` §4.3.

### B4. `config.py`: the run digest does not cover the source lineage

Raised by the agent implementing the reserve rule, which correctly refused to
decide it. `source_checkpoint` sits in `BOOKKEEPING_FIELDS`: right, since
hashing a path would make moving the run tree change every run's identity: but
**no source-seed field is in `IDENTITY_FIELDS`**. So a transfer run re-pointed at
a replacement source keeps the *same* run directory as the run that used the
rejected source.

Within one sweep this is harmless: the source stage settles the assignment before
any consumer is enqueued. Across sweeps it is not. The implementer added a
`lineage_conflicts()` refusal (exit 4) as a guard, which is the right stopgap.

**The proper fix is a `source_seed` field in `Config`, defaulting to the run's own
seed, included in `TRAJECTORY_FIELDS`.** The source seed is a genuine trajectory
determinant and is a small integer, not a path, so it carries none of the
objection that excluded `source_checkpoint`.

### B5. `aggregate.py`: reports the complete P0 tree as incomplete

`--out-root runs --experiments E1 --require-complete` exits 1 and names 12
missing arm×seed combinations on a tree that is complete for the seed actually
run. The seed axis is inferred from the declared block rather than from the runs
present, so donor-block seeds enter the expected set.

### B6. `aggregate.py`: crashes on a null seed, losing every other run

`seed = int(...)` with no guard aborts the whole aggregation on a single
malformed manifest.

---

## Analysis-layer: fix before reporting, not before running

Re-running the analysis costs minutes, so these do not gate compute. They do gate
any number that reaches the paper.

### A1. TUNE-seed leakage, in three modules

`ANALYSIS_PLAN.md` §8 forbids any estimate computed on hyperparameter-selection
seeds. It is not enforced in:

- **`plots.py`**: selection is by arm label only; every plotted estimate can
  include TUNE seeds.
- **`tables.py`**: the guard is `if 'seed_block' in df.columns`, and
  `seed_block` is not in `REQUIRED_COLUMNS`, so it fails open.
- **`aggregate.py`**: the membership axis mixes blocks.

`audit.py` enforces the rule in one direction only (a reported experiment
containing a TUNE run) and never checks that a selection experiment stays inside
its declared block.

### A2. `plots.py`: the source-validity gate is ignored entirely

The module never reads `source_valid` or `source_final_score`. Every figure can
therefore plot arms whose source never learned its task: the published study's
exact error, reproduced in the figures.

### A3. `n` is a row count presented as a seed count

In `aggregate.py` and `tables.py`. This defeats the `n < 3` PIPELINE VALIDATION
guard, and lets a mean and its stated `n` disagree when non-finite values are
dropped per metric.

### A4. `stats.py`: RQ6 is not like-with-like, and emits a directional claim from an interval covering zero

`sub_rq6` compares `prefix_score_500`, a single held-out checkpoint, against
`final_score`, which is the mean of three. `DESIGN.md` §2.4 RQ6 requires the
single-checkpoint comparison on both sides. Separately it prints
"the delta CHANGES SIGN between prefix 500 and the budget" at n=1 and on an
interval covering zero, which `ANALYSIS_PLAN.md` §9 forbids.

### A5. `report.py`: a hard-coded directional verb, and equivalence unchecked against its margin

`report.py:1537` hard-codes "exceeds", violating the §9 guard that direction
words are generated from the data. `claim()` accepts an equivalence verdict
without checking the interval against the margin it documents.

### A6. `statlib.py`: `mann_whitney` reports a p below its own stated floor

`mann_whitney([0.0]*5, [1.0]*5)` returns a p smaller than the
`p_min_attainable` in the same dict, and `at_p_floor` fails to fire on
maximally separated data at n=10. It also returns exact-table critical values
when the asymptotic method produced the p.

### A7. `plan.py`: harvested coefficients are double-deflated, and outliers cannot be excluded

Every harvested coefficient is deflated by 1/1.35 (`measurement_load` applied
twice). There is no outlier-exclusion mechanism of any kind, so the four
stall-affected P0 runs contaminate the fitted model: which is why it reads
12–20 ms/step against a clean median of 6.1.

### A8. `validate.py`: green on a bad path, red on the real one

`--runs <nonexistent>` reports "18 passed, 2 skipped", exit 0: a typo yields a
green suite. `--runs runs` **fails** `test_stats_no_pvalue_outside_family` on the
real P0 tree. Several tests report PASS when their on-disk half cannot run, and
the module docstring's claim that every `DESIGN.md` §9 row has a test that fails
when its guard is removed is false for roughly ten of the sixteen rows.

---

## What is already done

- **The reserve-seed rule is implemented and proven** (`registry.py`,
  `sweep.py`). Verified against the real 0.599 rejection: `src-dueling-vanilla`
  s00 → RESERVE seed 400, six dependants re-pointed across E1 and E2, ledger at
  `runs/_jobs/source_replacements.jsonl`, idempotent across four re-invocations
  and after deleting either the ledger or the rejected run directory. A live
  end-to-end test with a forced gate trained the replacement and confirmed the
  consumers loaded it.
- **P0 data verified sound** despite the claim collision (B1).

## Order of work

1. B1–B3 (`sweep.py` concurrency): the fatal one.
2. B4 (`config.py` source lineage): a schema change, so before any run.
3. B5–B6 (`aggregate.py`): otherwise `--require-complete` is unusable.
4. Re-run `validate.py --runs runs` until green, and fix A8 so green means green.
5. A1–A3: the correctness-of-reported-numbers set.
6. A4–A7: before any number reaches the paper.


---

## Audit state on the P0 tree, 2026-08-25

`python experiments/audit.py --out-root runs` reports **AUDIT FAIL: 27 errors, 5
warnings**, on three checks. None of them is a live defect, and each is expected
for a different reason. Recorded so that a later reader does not have to
re-derive this, and so that a *fourth* failure would stand out.

| Check | Errors | Why it fires | Action |
|---|---|---|---|
| `PLAN HASH` | 1 (+1 warn) | `ANALYSIS_PLAN.md` and `DESIGN.md` were amended on 2026-08-25 (revision 6), after the 44 P0 runs were produced. The check is doing exactly its job: it says the affected results are exploratory until the change is recorded in `ANALYSIS_PLAN.md` §11 | **Done.** The amendment is recorded in §11. P0 is n=1 pipeline validation and carries no result regardless (§9) |
| `SEED COMPLETENESS` | 19 | P0 ran one seed; the design requires ten | None. Clears when Tier 1 runs |
| `INVARIANTS` | 7 | `source_seed` did not exist when the P0 runs were written, so their manifests have no such key and the audit reads `None` against a declared `0`. **Verified not a live defect**: a freshly constructed transfer `Config` populates `source_seed=0`, so runs launched after B4 store it | None. Clears on relaunch |

The 4 `intensity_confounded` warnings on E1 and E5 `transfer_set=trunk` are **by
design**: that arm is the deliberately unmatched comparison, and the warning is
the label the design requires it to carry (`DESIGN.md` §3.1).

One live consequence worth stating: because the plan hash changed, **no P0 run
may be pooled with a Tier 1 run in a single table**. `aggregate.py` already
refuses that (more than one `ANALYSIS_PLAN.md` hash in one table is a
reporting-stopper), and `audit.py` escalates it to `ERROR`. This is the intended
behaviour, not something to work around.


---

## Sleep contamination in the cost model, 2026-08-26

**Established independently of `plan.py`**, from `wall_time_s` and `env_steps` in
the 44 P0 manifests, with each run's start reconstructed from the manifest mtime.
Three bands separate cleanly, with no judgement call:

| Band | n | median ms/step | range |
|---|---|---|---|
| clean | 33 | **5.98** | 4.72 to 7.23 |
| spans an idle window | 4 | **169.44** | 158.88 to 192.08 |
| immediately post-wake | 7 | **10.14** | 7.45 to 12.48 |

The four slow runs all started between 18:23 and 18:37 on 2026-08-24 and all
finished between 08:34 and 09:10 the next morning. They did not fail and they did
not collide with each other. They were in flight across an overnight idle window
and their wall clock includes it. **This is machine sleep, not defect B1.** B1 was
real and the fix is still required, but it did not cause these four.

### What this confirms

The re-harvest of the plain-LunarLander coefficient is **correct**. It moved from
0.011609 to 0.006184 s/step, and the 14 clean LunarLander dueling runs give
6.36 ms/step gross by direct division, against 6.184 fitted net of the documented
0.023 s/episode overhead. The old value was contaminated; the new one is not.

### What it also exposes, and the re-harvest did not catch

**The new interface-variant cost keys are themselves contaminated.** Of the eight
`extra_actions=2,pad_obs=4` runs, exactly one is clean:

| Run | Band | ms/step |
|---|---|---|
| `iface-scratch-mlp-vanilla` | clean | **5.25** |
| `iface-scratch-mlp-double` | spans sleep | 169.81 |
| `iface-scratch-dueling-vanilla` | spans sleep | 192.08 |
| `iface-transfer-mlp-vanilla` | post-wake | 11.76 |
| `iface-transfer-mlp-double` | post-wake | 10.14 |
| `iface-scratch-dueling-double` | post-wake | 11.50 |
| `iface-transfer-dueling-vanilla` | post-wake | 9.87 |
| `iface-transfer-dueling-double` | post-wake | 7.45 |

Every run behind the fitted `0.009352` and `0.008803` s/step is post-wake. The one
clean interface run is **5.25 ms/step**, statistically indistinguishable from the
clean plain-LunarLander median of 5.70. So the interface variant is **not
intrinsically slower**, and those two keys overestimate by roughly 1.7x.

The exclusion rule the re-harvest applied ("group gate", "whole-harvest gate")
removed the worst contamination by accident rather than by diagnosis. The correct
filter is **temporal**: exclude any run that overlaps an idle window, and any run
that starts within about two hours after one. A ms/step outlier rule cannot
distinguish a slow configuration from a slow machine, and only the timestamps can.

### Launch blocker

**Sleep and hibernation must be disabled before the confirmatory run.** This is
not housekeeping, it is the dominant term in the schedule. The projected cost of
E1, E2, E4, E5 and E8i at ten seeds is 720 runs and **82.8 h at `--jobs 6`**,
which is three and a half days and therefore three overnight windows. At the P0
rate, each overnight costs about 14 h of stalled wall clock plus a degraded
post-wake period, and the added time creates further overnights. Left alone this
roughly doubles the calendar time; it does not change the compute.

Two consequences for the runner, neither of which is currently implemented:

1. `wall_time_s` is **not a trustworthy cost signal** for any run that spans an
   idle window, and nothing in the pipeline currently detects that. A run whose
   wall clock exceeds its own step count times a plausible rate by more than, say,
   5x should be flagged in the manifest rather than silently fitted.
2. The harvest should record the wall-clock **band** of each run it fits, so a
   later reader can see whether a coefficient rests on clean data. Two of the four
   current keys do not.


---

## Adversarial verification of the fix pass, 2026-08-26

Ten reviewers were pointed at the ten rewritten modules and told that the fix
agents' claims were unverified and might be wrong, and that a defect "fixed" by
deleting a check, widening a tolerance or broadening an exemption was the single
most important thing to find. Every finding rated critical or high was then
handed to a separate skeptic told to **refute** it.

**59 findings across 7 reviewers. 20 sent for refutation: 18 confirmed, 2 refuted.**
Three reviewers (`sweep.py`, `plan.py`, `plots.py`+`tables.py`) died on connection
errors and were re-commissioned.

| Severity | Count |
|---|---|
| critical | 5 |
| high | 22 |
| medium | 21 |
| low | 11 |

### The five criticals

1. **`stats.py` `section_controls` is row-order dependent.** It builds its
   per-seed map with a last-wins `iterrows` loop, the exact pattern
   `arm_by_seed`'s own docstring says was fixed; that section never calls
   `arm_by_seed`. The skeptic reproduced corruption on **unmodified repo data**:
   copying `runs_demo/per_seed.csv` once in file order and once row-reversed, the
   same rows in a different order, moves **32 contrast rows and flips signs and
   verdicts**. `C1-C0` for dueling-vanilla goes from "positive, excludes zero" to
   "covers zero". Section 5 correctly *refuses* the same estimand on the same
   data while section 7 prints it with an interval excluding zero, and it reaches
   a generated paper table under a caption naming it as the confirmatory
   estimand.
2. **`stats.py` computes n, mean, SD and the "seed-level bootstrap" CI over rows,
   not seeds.**
3. **`aggregate.py` credits a zero-shot policy with reaching 25% of solved.**
   `min_periods=1` makes the trailing-100-episode mean at episode 0 a single
   5-episode evaluation. Two runs record `steps_to_threshold_p25 = 83` env steps,
   uncensored, at `updates=0` with `learning_starts=1000`. Next smallest is 8379,
   median 38435. Both are transfer arms and the **top two by jumpstart**, so the
   bias is systematic and points toward the arms the transfer claim rests on.
   Only p25 is affected.
4. **`audit.py --seeds 999` scopes the gate away.** 33 errors become 8, and the 8
   survivors are two known-stale conditions unrelated to seeds. On a clean tree
   this is AUDIT PASS, exit 0, no override stamp. The audit is the gate
   `report.py` refuses on.
5. **`registry.py` never plumbs `source_seed` into `Config`.** A reserve
   replacement re-points `source_checkpoint` to the RESERVE source but leaves
   `cfg.source_seed` at the default, so the digest is unchanged and the
   replacement writes into the **rejected run's directory**, recording the wrong
   source seed. This defeats fix B4 one layer up, and the reserve rule is about
   to fire for real: `src-dueling-vanilla` scored 0.5992 against the 0.600 gate.

### What the weakening lens found, which is the reassuring half

Mostly clean, and checked properly rather than asserted: `report.py`'s wording
guards are strictly stronger across 11,592 sentence-by-kind-by-evidence
combinations (0 weakenings, 12 strengthenings); `statlib`'s primitives return
bit-identical numbers across 600 differential calls; an old-versus-new run of
`stats.py` on one `per_seed.csv` changed **zero** of 4,406 shared JSON leaves. No
module-level numeric constant changed value. Several checks got genuinely
stronger. The real weakening is narrow: the TUNE exemption lost its
`family == 'screen'` conjunct, and the MDE agreement tolerance was set to 0.02,
which is exactly the smallest round value admitting the one row that fails at
0.01.

### One error of the parent's own, found by review

`DESIGN.md` 7's pilot evidence for the revision-6 endpoint amendment was computed
over every row with `condition == 'scratch'` and `env == 'LunarLander-v3'`. That
is eight rows, not four: it pooled the `CONFIRM` scratch arms at seed 0 with the
`C4SRC` positive-control sources at seed 300, which 3.4 forbids in those words.
The corrected ordering puts `mlp/vanilla` **fourth** on final score and **second**
on AUC; the text said first and last. Corrected in place, and logged in the
change log rather than quietly patched, because it is an instance of the very
error the seed blocks exist to prevent.

### Consequence for the launch

The **run path and the analysis path are separable**. Only `registry.py` corrupts
what gets *recorded*; every other defect above is in a layer that reads
`metrics.jsonl` and can be re-derived. So the campaign is gated on a short list:
`registry.py`'s `source_seed`, the `auc_score` denominator in `src/dqn/train.py`
(baked into manifests at write time), and an independent review of `sweep.py`,
whose FATAL B1 fix is the only blocking fix in this project that has never been
checked by anyone other than the agent that wrote it.


### Sleep blocker: CLEARED and verified, 2026-08-26

The user disabled sleep, and it was checked at the OS level rather than taken on
trust, because the schedule depends on it:

| Setting | AC value | Meaning |
|---|---|---|
| `STANDBYIDLE` (sleep after) | `0x0` | Never |
| `HIBERNATEIDLE` (hibernate after) | `0x0` | Never |
| `HYBRIDSLEEP` | `0x0` | Disabled |
| `DISKIDLE` | `0x4b0` | 1200 s, irrelevant under six writing workers |

`powercfg /availablesleepstates` reports **Standby (S3)** and explicitly *not*
Standby (S0 Low Power Idle), so this is classic S3 and the setting above genuinely
governs it. Modern Standby, which can idle a machine regardless of these values,
is not in play.

That is consistent with the P0 evidence: the four affected runs **suspended and
resumed** rather than dying, and their processes survived, which is what S3 does
and what a reboot would not have allowed.

**One residual, not blocking.** The DC (battery) sleep timeout is still 600 s. If
this machine is a laptop that can be unplugged, that is the one remaining route to
a repeat of the overnight stall. On AC it cannot happen.
