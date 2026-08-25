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
