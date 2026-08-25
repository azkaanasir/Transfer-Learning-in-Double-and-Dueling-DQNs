# `experiments/`: the controlled transfer study

Start here, then read in this order:

| File | What it is |
|---|---|
| [`STANDING_INSTRUCTIONS.md`](STANDING_INSTRUCTIONS.md) | The user's binding constraints. **Read before changing anything here.** |
| [`DESIGN.md`](DESIGN.md) | The authoritative experimental design: thesis, hypotheses with refutation conditions, factors, controls, metrics, environments, catalogue, and the anti-fallacy guardrails |
| [`ANALYSIS_PLAN.md`](ANALYSIS_PLAN.md) | **Pre-registered** inference: one confirmatory family, the tests, the multiplicity ledger, the equivalence procedure, and the measured minimum detectable effects |
| [`REVIEW_COVERAGE.md`](REVIEW_COVERAGE.md) | Two-way traceability between the eight peer reviews and everything in this directory |
| [`EXPERIMENTS.md`](EXPERIMENTS.md) | How to run it |
| `../paper/LITERATURE.md` | Per-source citation audit and the gap the revision can actually claim |

## The short version

The published study compared Double DQN against Dueling DQN under
CartPole→LunarLander transfer and claimed architecture was the single isolated
variable. A code audit (`../paper/METHODS_ACTUAL.md`) found the two arms differed
in architecture **and** Q-target rule, that the transfer arm used a learning rate
5× lower than its baseline under a claim of identical hyperparameters, that the
freeze schedule the manuscript describes was never implemented, and that one
source agent never learned its source task.

This directory rebuilds the study so that each of those failures is **impossible
rather than unlikely**:

* the architecture and Q-target axes are orthogonal factors, and every cell exists
* an experiment declares its invariants and `audit.py` *verifies* them
* the freeze schedule is indexed in gradient updates, logged, and confirmed by
  weight fingerprints
* source validity is gated on a normalised score, and exclusions are reported
* a run's identity is a digest over everything that can change its trajectory or
  its measurement, so two conditions cannot collide onto one directory
* transfer effects are decomposed against untrained-source and permuted-source
  controls, so "negative transfer" cannot be confused with the protocol's own
  mechanics
* every number is on a normalised score whose reference is measured and committed
* the inference plan is pre-registered and hashed into every artifact

## Layout

| Path | Role |
|---|---|
| `registry.py` | the experiment catalogue: the declarative contract everything reads |
| `plan.py` | cost and inventory, printed before anything launches |
| `sweep.py` | the runner: job manifest, atomic per-run claiming, resumable |
| `aggregate.py` | runs → `per_seed.csv` and `curves.csv` |
| `statlib.py` | statistical primitives |
| `stats.py` | the pre-registered analysis |
| `plots.py`, `tables.py` | figures and LaTeX tables, with generated captions |
| `audit.py` | the machine-checked invariants |
| `validate.py` | the self-test suite for every guardrail |
| `report.py` | one command: audit → aggregate → stats → plots → tables, with claim guardrails |
| `measure_references.py` | the normalisation constants |
| `preflight.py` | environment and dependency checks |
| `reference_returns.json` | measured random-policy and no-op returns per environment |

Implementation lives in [`../src/dqn/`](../src/dqn/): `config.py` (schema, run
identity, digests), `seeding.py` (named RNG streams, per-layer initialisation),
`networks.py` (architectures, layer groups, transfer sets), `transfer.py`
(weight transfer, the controls, freeze verification), `agent.py`, `train.py`,
`envs.py` (registry, variants, normalisation), `env_wrappers.py` (interface
change at zero dynamics shift), `shift.py` (domain-shift measurement),
`metrics.py`, `provenance.py`.
