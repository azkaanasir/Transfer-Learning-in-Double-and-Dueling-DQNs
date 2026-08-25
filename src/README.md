# `src/`: which code is authoritative

**Authoritative: [`dqn/`](dqn/).** Everything else in this directory is the
*historical record* of the code that produced the published runs, retained
deliberately and not to be used.

## Why the old packages are still here

`paper/METHODS_ACTUAL.md` had to reconstruct what the published study actually
did by reading these packages and diffing saved checkpoints, months after the
runs. That reconstruction is the evidence base for the corrections in
`experiments/DESIGN.md` §1, and for the response to reviewers. Deleting the
packages would destroy the only surviving record of what was run: the machine
that produced the runs is gone, and two of the log groups correspond to code
that was never committed at all.

So they stay, and this file exists so that nobody mistakes them for live code.

| Path | What it is | Status |
|---|---|---|
| `dqn/` | one `Config`, one agent, one training loop; an arm is `(arch, target_rule, condition)` plus an explicit protocol | **authoritative** |
| `cartpole_dqn_ddqn/` | despite the name, calls `build_dueling_model()` and uses a **vanilla max** target: not a Double DQN in either respect. Writes to the same `BASE_SAVE_PATH` and `TB_LOG_DIR` as `cartpole_dqn_dueling/`, so the two overwrite each other | historical record |
| `cartpole_dqn_dueling/` | the CartPole dueling source trainer | historical record |
| `LunarLander_double_dqn/` | the DDQN arm: plain MLP with a proper double-Q target | historical record |
| `lunarlander_dueling_dqn/` | the Dueling arm: dueling network with a **vanilla max** target. Together with the row above, this is the confound: the two arms differed in architecture *and* update rule while the manuscript claimed one isolated variable | historical record |
| `transfer_dqn_dueling/` | the Dueling transfer arm. Its freeze set was resolved by indexing into `model.layers`, which for a branched functional model froze three of six Dense layers permanently; no episode-indexed schedule exists anywhere in it | historical record |
| `cartpole_ddqn.py`, `cartpole_dueling.py`, `lunarlandar_ddqn.py` (sic), `dueling_dqn_lunarlander.py`, `param.py`, `dummy.py` | loose scripts alongside the packages, which is why it was unclear for months which path produced the reported numbers | historical record |

**There is no `transfer_dqn_ddqn/`.** The DDQN transfer arm: the paper's
headline positive result: was never committed, and the machine that ran it is
gone. That arm is not reproducible as run and cannot be made so; it must be
re-run from `dqn/`, and the published numbers for it cannot be cited.

## What `dqn/` contains

| Module | Role |
|---|---|
| `config.py` | the single configuration schema, run identity as a digest over every trajectory- and measurement-affecting field, and an import-time check that no field is unclassified |
| `seeding.py` | named independent RNG streams; per-layer initialisation, which makes the 2x2 matched by seed |
| `networks.py` | both architectures, named layer groups, the dueling aggregation variants, and the named transfer sets |
| `transfer.py` | weight transfer by name, the untrained / permuted / spectrum-matched controls, value-head recalibration, and weight fingerprints that *verify* a freeze happened |
| `agent.py` | the four cells, per-stream gradient norms, plasticity signals, global gradient clipping |
| `train.py` | the training loop, freeze schedule in gradient updates, held-out evaluation, prefix checkpoints, resume, provenance |
| `envs.py` | the environment registry, parametric variants, and score normalisation against measured references |
| `env_wrappers.py` | interface change at zero dynamics shift |
| `shift.py` | domain-shift measurement, and an explicit refusal where it is undefined |
| `metrics.py` | an episode-keyed log that is idempotent under resume |
| `provenance.py` | git state, package versions, determinism settings, plan hashes |
| `replay.py` | the replay buffer, with an injected generator so a diagnostic cannot perturb training |

Read `experiments/DESIGN.md` before changing any of it.
