# Running the Experiments

The **design** is in [`DESIGN.md`](DESIGN.md); the **inference** is
pre-registered in [`ANALYSIS_PLAN.md`](ANALYSIS_PLAN.md); the **reviewer
traceability** is in [`REVIEW_COVERAGE.md`](REVIEW_COVERAGE.md); the **user's
binding constraints** are in [`STANDING_INSTRUCTIONS.md`](STANDING_INSTRUCTIONS.md).
This file is only about how to run things.

---

## 0. First, on any machine

```bash
pip install -r requirements.txt

# 1. Does everything import, do both environments build, does Box2D work?
python experiments/preflight.py

# 2. The self-test suite. Every guardrail the design claims is enforced in code
#    has a test here that fails when the guard is removed.
python experiments/validate.py

# 3. The normalisation constants. Everything is reported as a normalised score,
#    so the random-policy reference for every environment and variant must be a
#    measured, committed quantity. Verify they are all present:
python experiments/measure_references.py --check
```

**Do not skip step 2.** `preflight.py` catches a missing Box2D — which breaks
LunarLander but not CartPole, so a naive CartPole smoke test passes and a sweep
dies hours later. `validate.py` catches the subtler class: a diagnostic that
perturbs training, a resume that silently restarts Adam, an epsilon schedule
coupled to the evaluation cadence, two conditions colliding onto one run
directory. Those are the defects that produce *plausible wrong numbers* rather
than a crash.

## 1. Cost, before committing to anything

```bash
python experiments/plan.py --tier 1                 # what Tier 1 costs
python experiments/plan.py --all --jobs 6           # everything
python experiments/plan.py --measure                # calibrate on this machine
```

Measured on the development machine (8 cores, 2 threads per worker):

| Run | Wall clock | Updates |
|---|---|---|
| CartPole-v1, 1000 episodes | ~3–4 min | ~90k |
| LunarLander-v3, 1000 episodes, `mlp` | **10.6 min** | 304k |
| LunarLander-v3, 1000 episodes, `dueling` | **13.0 min** | 294k |

Both LunarLander configurations reached a final-100 training return of 205–216
against a solved threshold of 200, so the 1000-episode budget does reach useful
performance — the convergence question `DESIGN.md` §5.2 raises is answered
affirmatively for the target task, and the convergence gate is reported per run
regardless.

Disk: about 0.6 MB of durable artifacts per run, plus a transient ~5.3 MB replay
buffer that is deleted on successful completion unless `--keep-buffer` is given.

## 2. Validate the pipeline at one seed

Per `STANDING_INSTRUCTIONS` S8, the current phase runs **one seed only**. The
purpose is to prove the machinery executes, not to produce evidence.

```bash
# The smoke experiment: tiny, and it exercises every code path the catalogue uses
python experiments/sweep.py --experiments E0

# Then one seed of the real thing
python experiments/sweep.py --experiments E1 --seeds 0 --jobs 4
python experiments/aggregate.py
python experiments/audit.py --experiments E1
python experiments/stats.py --per-seed runs/per_seed.csv
python experiments/report.py --experiments E1 --outdir paper/results
```

Everything produced at n=1 is stamped **PIPELINE VALIDATION — NOT A RESULT**.
`stats.py` emits no test and no interval below n=3, deliberately: a single seed
can show that a run *executes*, never that an arm *differs*.

## 3. The confirmatory runs, when compute is authorised

```bash
# Tier 1 — nothing headline may be claimed before this completes
python experiments/sweep.py --tier 1 --jobs 6

# Tier 2 — the mechanism ablations
python experiments/sweep.py --tier 2 --jobs 6

# Then the full pipeline
python experiments/report.py --outdir paper/results
```

Tier 1 is `E1` (the controlled 2×2), `E2` (the control set), `E3` (per-architecture
sensitivity, on `TUNE` seeds), and `E8i` (the interface-change corner). Tier 2 is
the mechanism ablations. Tier 3 is scope extension.

**Run E3 before E1 if the per-cell tuning policy is wanted**, since it selects the
secondary configuration. E3 runs on the `TUNE` block and no reported estimate may
touch those seeds.

## 4. The catalogue

| ID | Name | Tier | Family | What it is for |
|---|---|---|---|---|
| `E0` | smoke | — | — | every code path, at a 12-episode budget |
| `E1` | core2x2 | 1 | **confirmatory** | the four cells × {CartPole source, LunarLander scratch, transfer}, plus the trunk-only secondary |
| `E2` | controls | 1 | estimation | untrained-source, untrained at K=0, permuted, spectrum-matched |
| `E3` | hpsens | 1 | screen | lr × target-update per cell, on `TUNE` |
| `E4` | freezedur | 2 | screen | freeze window in gradient updates: 0, 5k, 10k, 20k, 50k, never |
| `E5` | layerset | 2 | screen | fc1 / fc2 / trunk / matched / described |
| `E6` | streamfreeze | 2 | screen | dueling: freeze none / trunk / value / adv / heads |
| `E7` | aggregation | 2 | screen | dueling: mean / max / naive baseline subtraction |
| `E8` | shiftaxis | 2 | estimation | same-interface dynamics shift: wind (primary), gravity (secondary) |
| `E8i` | interfaceonly | 1 | estimation | interface change at zero dynamics shift; also control C4 |
| `E11` | valuerecal | 2 | estimation | value-head recalibration |
| `E12` | capacity | 3 | screen | hidden width 64 / 128 / 256 |
| `E13` | plasticity | 3 | estimation | the plasticity-loss rival explanation |

Only `E1` is confirmatory. Everything else is estimation or screening and carries
**no p-values** — see `ANALYSIS_PLAN.md` §2 for why that is a design decision
rather than an omission.

## 5. How runs are identified, and why it matters

```
runs/<condition>/<run_digest12>/s<NN>/
```

`run_digest` is a hash over **every field that can change the training trajectory
or the reported measurement**, and nothing else. Two consequences:

* Two experiments requesting an *identical* configuration get the **same run**.
  That is intentional — E4's freeze level that equals E1's protocol value, E8's
  level-0 scratch arms and E7's scratch arms are the same runs, and training them
  twice would waste compute and produce two independent estimates of one
  quantity. Across the whole catalogue this sharing removes about 29 % of the
  jobs. Experiment membership lives in `runs/_index/<experiment>.jsonl`.
* Two *different* configurations can never share a directory. The previous
  scheme named directories `<env>/<arch>-<rule>-<mode>-s<NN>`, which omitted the
  freeze window, the transfer set, the learning rate, the target-update rule, the
  width, the aggregation, the environment variant and the control condition —
  so nine conditions drawn from six experiments collapsed onto one path, and a
  completed directory was silently *resumed* rather than refused. Five
  experiments would have been fabricated from one experiment's data with every
  check passing.

## 6. Interruption and resume

Resume is routine and idempotent, under two rules that exist because their
absence corrupts data silently:

1. On resume, `metrics.jsonl` is **truncated** to episodes before the resume
   point. The published loop appended without truncating: a 45-episode run
   interrupted at episode 39 with its last checkpoint at 30 produced 54 rows with
   episodes 31–39 recorded twice, and every window statistic downstream was
   computed over duplicates.
2. Resume is **refused** when the checkpoint's trajectory digest differs from the
   requested configuration. Continuing under changed hyperparameters is the class
   of error the Phase 0 audit spent days undoing.

The checkpoint carries the optimiser state, the replay buffer and every RNG
stream position. The published checkpoint omitted the optimiser, so a resumed run
restarted Adam with zero moments while claiming to be the same run.

So: **re-running the same command after any interruption is always correct.**
`validate.py::test_resume_equivalence` is what makes that a tested claim.

## 7. Outputs

```
runs/
  _jobs/jobs.jsonl          the resolved job manifest
  _jobs/status.jsonl        append-only state transitions
  _index/<experiment>.jsonl experiment -> member run directories
  _logs/                    per-worker logs
  <condition>/<digest>/s<NN>/
      manifest.json         identity, config, provenance, transfer report,
                            freeze events with verification, source validity,
                            result
      metrics.jsonl         one row per episode, keyed by episode
      eval_episodes.jsonl   one row per evaluation episode
      events.jsonl          freeze transitions
      model.keras, optimizer.npz, state.json, ckpt_ep<N>/
  per_seed.csv              one row per run  <- every paper number traces here
  curves.csv                long-form per-episode
paper/
  results/REPORT.md         the generated report
  figures/, tables/         with generated captions and provenance
```

Nothing in the paper is computed by hand. Every figure and table carries the hash
of the CSV it was built from, so a stale artifact is detectable.

## 8. Choosing `--jobs`

CPU-bound: 128×128 networks at batch 64, so kernel-launch overhead dominates and
a GPU is typically no faster. Choose machines by core count. `--jobs` pins
per-worker thread counts; without that pinning every TensorFlow process claims
all cores and they thrash.

The parallel axis is the **run**, claimed through an atomic lock file per run
directory, so any `--jobs` value is safe and two workers cannot enter the same
directory. The previous scheme sharded by seed *string* and re-parsed the shard
as a contiguous range, so `--seeds 0-4 10-19 --jobs 3` silently trained seeds
5–9 that were never requested.

## 9. Gotchas

* `runs/` and `runs_demo/` are gitignored. Copy `per_seed.csv` out to version it.
* A transfer run needs its source to have finished. `sweep.py` orders jobs so
  sources precede consumers and defers a job whose source is missing; a job still
  blocked at the end is **reported**, never silently skipped.
* Shard logs are written, not streamed. `tail -f` one to watch progress.
* An environment variant with no measured reference return will refuse to
  produce a score rather than silently normalising against zero. Run
  `measure_references.py` first.
* `--allow-audit-failure` exists but stamps the override into every artifact.
  Reaching for it is the signal to fix the audit instead.
