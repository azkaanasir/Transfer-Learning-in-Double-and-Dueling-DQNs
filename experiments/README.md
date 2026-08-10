# Running the Phase 1 sweep

The design, and why it is this design, is in [`paper/METHODS_ACTUAL.md`](../paper/METHODS_ACTUAL.md).
This file is only about how to run it.

## The design in one table

`{mlp, dueling} × {vanilla, double}` = 4 cells. Per cell, per seed, three runs:

| Stage | Env | Mode | Purpose |
|---|---|---|---|
| `source` | CartPole-v1 | scratch | the checkpoint transfer draws from |
| `baseline` | LunarLander-v3 | scratch | that cell's own scratch reference |
| `transfer` | LunarLander-v3 | transfer | loads its own cell+seed source |

At 10 seeds that is **120 runs**. A transfer run only ever loads the source from
its *own* cell and seed, so cells never contaminate each other.

`mlp` vs `dueling` is the architecture axis; `vanilla` vs `double` is the
Q-target axis. Both are needed because the published study varied them
*together* while claiming to isolate architecture.

## Quick start

```bash
pip install -r requirements.txt

# 1. Always run this first on a new machine (~1 min).
python experiments/preflight.py

# 2. See what would run, without running it.
python experiments/sweep.py --seeds 0-9 --stage all --dry-run

# 3. Pilot two seeds before committing to the full sweep.
python experiments/sweep.py --seeds 0 1 --stage all --jobs 2
python experiments/aggregate.py

# 4. Full sweep.
python experiments/sweep.py --seeds 0-9 --stage all --jobs 4

# 5. Results.
python experiments/aggregate.py          # -> runs/per_seed.csv
python experiments/stats.py --per-seed runs/per_seed.csv
```

**Do not skip step 1.** Its most valuable check is that Box2D works: if it
failed to build, CartPole still runs fine and LunarLander does not, so a naive
smoke test passes and the sweep dies hours later.

**Check the pilot before scaling.** Specifically, confirm the CartPole sources
actually learn. In the published runs the DDQN source reached 26.94 on a task
solved at 195 and nobody noticed; that single fact undermines the paper's
headline arm.

## Choosing `--jobs`

This is a **CPU-bound** job. The networks are 128×128 at batch 64, so
kernel-launch overhead dominates the arithmetic and a GPU is typically no
faster — sometimes slower. Choose machines by core count, not accelerator, and
do not spend a free-tier GPU quota on it.

Runs are independent, so wall-clock ≈ total ÷ `--jobs`. Sharding is by seed,
which keeps each worker self-contained. `--jobs` also pins per-worker thread
counts; without that, every TensorFlow process tries to claim all cores and they
thrash. Use `--jobs`, not a hand-written shell loop.

`python experiments/preflight.py --estimate-only` measures throughput on the
current machine and prints an estimate per `--jobs` setting.

## Resuming

Everything is resumable and idempotent:

- Runs checkpoint every 25 episodes, carrying weights, optimiser state, and the
  replay buffer.
- A run with a completed manifest is skipped; a partial run continues from its
  checkpoint.
- So **re-running the same command after any interruption is always correct** —
  which is what makes 12-hour cloud session limits survivable.

Raising `--episodes` extends existing runs rather than restarting them.

## Platforms

| Platform | Cores | Notes |
|---|---|---|
| **Local** | varies | Best if you can spare the machine. No session limits. |
| **Kaggle** | 4 | Best free tier. `notebooks/kaggle_sweep.ipynb`. Use *Save & Run All* for batch execution — it runs with no browser open. |
| **Colab (free)** | 2 | `notebooks/colab_sweep.ipynb`. Mounts Drive so results outlive the VM. ~12 h cap, ~90 min idle disconnect. |
| **Oracle Cloud Always Free** | 4 (ARM) | No session limit; run under `tmux` and walk away. aarch64 needs a different TF install. |

## Outputs

```
runs/
  _logs/shard*.log              per-worker logs when --jobs > 1
  cartpole/<arm>-s<NN>/
  lunarlander/<arm>-s<NN>/
    metrics.csv                 per-episode reward, length, eval, loss, epsilon
    manifest.json               resolved config, transfer report, freeze events
    model.keras                 final model (the source for transfer runs)
    online.weights.h5 …         checkpoint
  per_seed.csv                  one row per run -- the paper's numbers come from here
```

`per_seed.csv` is the artifact every number in the manuscript should trace to.
Nothing should be computed by hand.

`aggregate.py` applies the **final-100-episode** evaluation window, which the
published paper used but never disclosed — Phase 0 recovered it by search.

## Gotchas

- `runs/` is gitignored. Copy `per_seed.csv` out if you want it versioned.
- `--episodes` defaults to **1000**, not the manuscript's 500: ε does not reach
  its floor until episode ~891, so 500 measures every arm mid-exploration.
- `transfer` needs `source` to have finished for the same cell and seed. With
  `--stage all` the ordering is handled; if you run stages separately, run
  `source` first.
- Shard logs are written, not streamed. `tail -f` one to watch progress.
