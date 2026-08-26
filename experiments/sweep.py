"""The runner: an explicit job manifest, and one atomic claim per run directory.

This replaces a driver that was wrong at scale in a way that would have
fabricated data silently, so the reasons are worth stating precisely.

**Defect 1 -- seeds were round-tripped through a string.** The old sharding
re-expressed a worker's seed list as ``f'{group[0]}-{group[-1]}'`` and the child
re-parsed that as a contiguous range. ``--seeds 0-4 10-19 --jobs 3`` therefore
trained seeds 5-9, which nobody requested, and omitted nothing -- so the
resulting arms would have looked complete while containing runs the design never
scheduled. Nothing here re-serialises a seed set. The parent resolves every job
once, writes ``runs/_jobs/jobs.jsonl`` with one line per job and the *whole*
resolved config on it, and workers read that file. A worker cannot invent a job,
because it never parses a specification.

**Defect 2 -- the parallel axis was the seed.** Sharding by seed only works while
every job at a seed is independent of every other worker's output, which stopped
being true the moment sources became shared across experiments (`registry`
de-duplicates identical configurations, so E4's protocol-valued freeze level and
E1's transfer arm are the *same run*). The parallel axis here is the **run
directory**: a worker takes a job by creating ``<run_dir>/.claim`` with
``O_CREAT|O_EXCL``, which the filesystem makes exclusive. Two workers cannot
enter the same directory, so any ``--jobs`` value is safe and the sharding
arithmetic that produced defect 1 does not exist. Exclusivity is only worth as
much as the rule that decides when a claim may be taken away from its owner; see
defect 5, which is where that rule was wrong.

**Defect 3 -- dependencies were assumed rather than checked.** The old driver
printed ``SKIP ... source missing`` and carried on with exit code 0, which is the
silent-hole failure mode: an arm quietly missing its transfer runs. A job whose
``depends_on`` run has no manifest is *deferred* and retried; if it is still
unsatisfied when the worker stops it is reported as **blocked** and the process
exits non-zero. `DESIGN.md` §8.4 and `ANALYSIS_PLAN.md` §8 both forbid a run
disappearing without a stated rule, and a runner that skips silently is how that
happens by accident.

**Defect 5 -- an exclusive claim was handed to a second worker mid-run.**
The claim's timestamp was written once and never refreshed, and the reclaim rule
was ``age > --stale-seconds`` alone: no check that the owner had actually died,
and no requirement that the reclaiming worker belong to a different sweep. Any
run longer than two hours was therefore taken over *while it was still
training*. This fired in P0: a 14.4 h source run was reclaimed at 14.2 h by a
worker in the same sweep, which then entered the same directory and began
writing the same ``metrics.jsonl``. Only a Windows mandatory file lock stopped
it, 1.1 s later; on POSIX both trainers would have written the same metrics,
checkpoints and ``model.keras`` for the remaining eleven minutes, which is
exactly the duplicated- and interleaved-episode corruption `DESIGN.md` §8.2(1)
exists to prevent. Three changes close it, and they are independent on purpose:

* the owner **heartbeats** its claim for as long as it is training, so an
  un-refreshed claim means an absent owner rather than a long run;
* a reclaim requires **evidence that the owner is gone**: its pid is checked on
  this host, and a claim from another host is reclaimed only when it carries a
  heartbeat field, so a claim written by a version that could not heartbeat is
  never assumed dead;
* ``release_claim`` and ``fail_claim`` **check ownership** before touching the
  file. In P0 the original owner's successful completion deleted the
  *interloper's* claim, because neither call ever compared the claim against
  itself.

The failure the collision then produced is kept apart from a real one too. A run
that fails after taking a directory over is recorded as **contended**, and a
contended failure defers its dependants instead of blocking them: in P0 a
healthy source that finished eleven minutes later was recorded as having failed,
and its transfer run was reported blocked, which under `DESIGN.md` §8.4 is a
terminal, non-zero-exit verdict manufactured by the runner's own collision.

**Defect 4 -- the reserve-seed rule was documented and never implemented.**
`DESIGN.md` 4.3 pre-registers it: "source seeds are drawn in order from a
RESERVE block until the cell has its full complement of valid sources, and the
number and identity of rejected source seeds appear in the results table". The
block existed in `registry.py`, `plan.py`, `report.py` and `stats.py` all
described the rule in prose, and the analysis layer reported `source_valid` --
but nothing ever drew a replacement. At ten seeds a cell with three failed
sources would have run at n=7, and `audit.py` would then have refused the whole
report on seed completeness, 52 hours into the sweep. Worse, the failure is
silent up to that point: a run whose source never learned the source task is
exactly the published study's central defect, and it looks identical to a good
run from the runner's side.

So a selection containing source arms runs in **two phases**. Phase 1 trains the
sources and gates each one on its own normalised final score; every failure
allocates the next unused RESERVE seed, enqueues that source, and re-points the
dependent transfer runs at the replacement checkpoint; the phase repeats while
newly trained sources keep failing. Phase 2 runs the target side against the
assignment that survived. Exhausting the reserve is an error and stops the
sweep, and so is a source that finished without producing the number the gate is
defined on: an unscored or non-finite score is a **measurement failure**, never
a pass and never a rejection. Reading ``nan >= 0.6`` as a rejection would spend a
RESERVE seed on a broken evaluation and write ``NaN`` into the results table
`DESIGN.md` 4.3 asks for; reading an absent score as a pass would train the whole
target side against a source of unknown competence, which is the defect 4.3
exists to prevent, reached with no refusal anywhere.

One scope statement about the gate, because it decides whether the smoke
test can run at all. `DESIGN.md` 4.3 defines validity on the normalised final
score of a source trained to the design's budget, and it governs the sources of
**reported** estimates. The pipeline-validation experiment is neither: it trains
12 episodes (`registry.SMOKE_OVERRIDES`) and its catalogue entry says "not a
result under any circumstances", so its sources score around zero and could not
clear 0.6 at that budget however well the code worked. Applying the gate to them
made the documented pre-launch smoke train 42 runs instead of 7, burn every
RESERVE seed in both lineages and exit 3 without ever reaching phase 2, which is
the half it exists to validate. So a selection made **entirely** of SMOKE-block
experiments, at SMOKE-block seeds, is **not gated**; the fact is printed and
stamped into the invocation record, and one reporting experiment anywhere in the
selection brings the pre-registered gate back for the whole selection. What the
smoke still refuses is a source that finished without a finite ``final_score``:
that is a pipeline failure, and finding pipeline failures is what it is for. To
exercise the 4.3 rejection-and-replacement path deliberately, give the smoke a
gate it cannot meet: ``--experiments E0 --source-gate 0.9``.

Every rejection is appended to ``runs/_jobs/source_replacements.jsonl``
with the seed, the cell, the score and the replacement, because the design
requires the rejected seeds to appear in the results table and a number that
lives only in a terminal scrollback is not reportable.

The assignment is derived, not drawn: it is a function of the ledger plus the
scores already on disk, walked in a fixed order over a fixed RESERVE ordering.
Re-running the same command after an interruption therefore reproduces the same
assignment instead of taking a fresh draw.

One consequence is stated rather than papered over. `source_checkpoint` is
deliberately *excluded* from the run digest (`src/dqn/config.py`,
``BOOKKEEPING_FIELDS``: hashing a path would make moving the run tree change
every run's identity), and until the reserve rule nothing could make two runs
differ only in it -- the source seed was a pure function of the target seed. A
re-pointed transfer run therefore keeps the run directory it would have had with
the rejected source. That is safe within a sweep, because the source stage
settles the assignment before any consumer is enqueued, but a directory left by
an *earlier* sweep can hold a run built from the now-rejected source. Such a
directory is detected by comparing the stored ``source.source_result.run_digest``
against the assigned source and is **refused**, never skipped as complete and
never resumed: serving invalid-source data under a valid-source label is the
pooling error `DESIGN.md` 4.3 exists to prevent. Making the digest cover the
source lineage instead would be the better fix and it belongs in
`src/dqn/config.py`, which this file does not own.

Three further properties are deliberate:

* **Workers are separate interpreters, always -- even at ``--jobs 1``.**
  TensorFlow does not survive ``fork`` and Windows has no ``fork`` at all, so
  subprocesses are forced. Spawning them *unconditionally* matters for a second
  reason: ``PYTHONHASHSEED`` and ``TF_ENABLE_ONEDNN_OPTS`` can only be set before
  the interpreter starts, so an in-process fast path at ``--jobs 1`` would record
  a different ``provenance.determinism`` block than the parallel path and the two
  would not be comparable (`DESIGN.md` §8.3).
* **Completion is decided by the manifest, not by a directory existing.** A run
  whose manifest reports ``episodes_completed >= num_episodes`` is skipped. The
  audit found the previous scheme resumed a *completed* directory belonging to a
  different configuration and emitted a manifest whose config never matched its
  metrics; run identity now covers the config (`src/dqn/config.py`) and this file
  refuses to treat a bare directory as evidence of anything.
* **Every state transition is appended to ``runs/_jobs/status.jsonl``**, so a
  sweep killed by a session timeout is inspectable and resumable rather than
  needing to be re-derived from log files.

Usage::

    python experiments/sweep.py --experiments E1 --seeds CONFIRM --dry-run
    python experiments/sweep.py --tier 1 --jobs 6
    python experiments/sweep.py --experiments E0                  # smoke
    python experiments/sweep.py --experiments E1 E2 --resume       # idempotent

Layout written under ``--out-root`` (default ``runs``)::

    _jobs/jobs.jsonl          one line per resolved job, with its full config
    _jobs/jobs-<id>-<tag>.jsonl  what one phase of one sweep was given to run
    _jobs/status.jsonl        append-only state transitions, all sweeps
    _jobs/*.jsonl.lock        the cross-process lock for the file beside it;
                              inert, and holds no data of its own
    _jobs/sweep-<id>.json     the invocation: argv, seeds, host, thread pinning
    _jobs/source_replacements.jsonl  every validity-gate rejection and its
                              replacement seed (DESIGN.md 4.3), append-only
    _index/<experiment>.jsonl experiment -> member run_dirs, with a
                              ``.lock`` beside it for the same reason
    _logs/w<NN>-<id>[-<tag>].log  per-worker output
"""
from __future__ import annotations

import argparse
import ast
import ctypes
import json
import math
import os
import socket
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import IO, Mapping, Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _path in (_ROOT, _HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import registry                                             # noqa: E402
from src.dqn.config import (BOOKKEEPING_FIELDS,             # noqa: E402
                            TRANSFER_ONLY_FIELDS, Config)

JOBS_SUBDIR = '_jobs'
INDEX_SUBDIR = '_index'
LOGS_SUBDIR = '_logs'
JOBS_FILE = 'jobs.jsonl'
STATUS_FILE = 'status.jsonl'
# Durable record of every source the validity gate rejected and what replaced
# it. DESIGN.md 4.3 requires the number and identity of rejected source seeds in
# the results table, so the rejections have to outlive the terminal that printed
# them; append-only, one JSON object per rejection, for the analysis layer.
REPLACEMENTS_FILE = 'source_replacements.jsonl'
CLAIM_NAME = '.claim'
# A superseded claim used to be renamed aside and left there for ever; two such
# files were still sitting in the P0 tree when the verification pass found them.
# The prefix is kept, because the rename is what makes a reclaim race safe, but
# the file is now removed as soon as the rename has decided that race.
SUPERSEDED_PREFIX = CLAIM_NAME + '.superseded-'
# The two other temporaries written beside a claim: a heartbeat refresh and a
# failure mark, each renamed into place immediately. One left on disk is the
# residue of a process killed inside that window. Neither was covered by the
# purge, so a hard kill left one file per killed worker in the run directory for
# ever, which is the litter the superseded prefix was cleaned up to remove.
HEARTBEAT_PREFIX = CLAIM_NAME + '.hb-'
FAILMARK_PREFIX = CLAIM_NAME + '.fail-'
# How old such a temporary has to be before the purge will remove it. A fresh
# one belongs to a rename in flight in another process, and deleting it under
# that process turns a refresh into an error for no reason.
CLAIM_TEMP_MIN_AGE_SECONDS = 300

DEFAULT_STALE_SECONDS = 7_200
DEFAULT_POLL_SECONDS = 15
# The floor under the default compute ceiling on the RESERVE replacement rule
# (``--max-source-replacements``). The rule itself is unbounded by design and
# the ceiling is on one invocation, not on the rule; the default is one full
# round of replacement across the plan's source lineages, and this is the
# smallest that default may be, so a two-lineage smoke still has room to draw.
DEFAULT_MIN_SOURCE_REPLACEMENTS = 4
# How often a training worker refreshes its own claim. It has to sit far below
# ``--stale-seconds``, or the heartbeat could not tell a slow run from a dead
# one; at 60 s against a 7200 s default there are 120 refreshes inside one
# staleness window, so a single missed write proves nothing while a run that has
# genuinely stopped is unambiguous.
HEARTBEAT_SECONDS = 60

# Fields the runner itself owns. Overriding them does not do what it looks like:
# `seed` is set per job by the registry, `experiment` and `label` are the names
# the plan is printed under, `source_checkpoint` and `source_seed` are what the
# validity gate assigns, and `out_root` would move the run directories out from
# under ``--out-root`` while the job, status and index files stayed behind. All
# six are bookkeeping or registry-assigned, so none of them changes a run digest
# either: the runs would land in the catalogue's own directories wearing the
# catalogue's identity.
_RUNNER_OWNED_FIELDS = ('out_root', 'experiment', 'label', 'seed',
                        'source_checkpoint', 'source_seed')

# Conditions that actually read the source checkpoint's weights. The untrained
# control builds its own random source of matched shape, so it needs the
# dependency's *lineage* but not its file; the dependency is still honoured as
# declared, because relaxing it here would mean the runner and the registry
# disagreed about what a job depends on.
_WEIGHT_READING = ('transfer', 'transfer_permuted')

# Every block `registry.SEED_BLOCKS` declares, so that `seed_block` can name the
# block of any seed the design knows about. SMOKE used to be (0,), overlapping
# CONFIRM, and was left out here for that reason; the registry has since moved it
# to (999,) with the comment "Disjoint from CONFIRM on purpose. With SMOKE=(0,) a
# pipeline-validation run was attributed to the confirmatory block, so a
# seed-block audit could not tell a smoke run from a real one by its seed alone."
# Leaving it out here defeated exactly that fix: every E0 run was written to
# jobs.jsonl, status.jsonl and _index/E0.jsonl with seed_block 'UNKNOWN'.
_BLOCK_ORDER = ('CONFIRM', 'REPLICATE', 'TUNE', 'C4SRC', 'RESERVE', 'SMOKE')


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------
def resolve_seed_spec(tokens: Sequence[str] | None) -> Optional[tuple[int, ...]]:
    """Turn ``['CONFIRM']`` or ``['0-4', '10-19']`` into an explicit seed tuple.

    Returns ``None`` when nothing was requested, which means "each experiment
    uses the block it declares" -- the only mode in which `DESIGN.md` §3.4's
    disjointness holds without an argument.

    A block name is expanded token-wise, so ``--seeds CONFIRM REPLICATE`` is the
    pre-registered pooled n=20 set of `ANALYSIS_PLAN.md` §6.5 rather than an
    error. The resolved integers are what reach the job manifest; no seed set is
    ever collapsed back into a string.

    Every token is checked on its own and a bad one is named, because the
    failure modes here are quiet rather than loud. ``--seeds 5-0`` is a reversed
    range that contributes nothing and used to leave the rest of the
    specification running as though it had been asked for in full; ``--seeds=-5``
    parses as the seed -5, which belongs to no `DESIGN.md` §3.4 block and would
    be written into every record as ``seed_block: UNKNOWN``. A ``ValueError``
    raised here is turned into a refusal by `main`, not into a traceback.
    """
    if not tokens:
        return None
    out: list[int] = []
    raw = ' '.join(str(t) for t in tokens).replace(',', ' ').split()
    if not raw:
        raise ValueError(
            f'--seeds {list(tokens)!r} is empty. Omit --seeds entirely to give '
            f'each experiment the block it declares, or name a block '
            f'({", ".join(registry.SEED_BLOCKS)}) or a range such as 0-9.')
    for tok in raw:
        if tok in registry.SEED_BLOCKS:
            got = tuple(registry.SEED_BLOCKS[tok])
        else:
            try:
                # `tok` is not None, so the block argument is unreachable; it
                # exists only to satisfy the signature.
                got = tuple(registry.resolve_seeds(tok, 'CONFIRM'))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f'--seeds token {tok!r} is neither a block name '
                    f'({", ".join(registry.SEED_BLOCKS)}) nor a seed or seed '
                    f'range such as 0-9: {exc}') from None
        if not got:
            raise ValueError(
                f'--seeds token {tok!r} names no seeds at all. A range is '
                f'written low-high, so 5-0 is empty; a token that silently '
                f'contributes nothing would leave the rest of the '
                f'specification looking like the whole request.')
        negative = sorted(sd for sd in got if sd < 0)
        if negative:
            raise ValueError(
                f'--seeds token {tok!r} yields negative seed(s) {negative}. '
                f'Seeds index the DESIGN.md 3.4 blocks and are non-negative; a '
                f'leading minus is read as a sign, not as the start of a range '
                f'(write --seeds 0-5, not --seeds=-5).')
        out.extend(got)
    if not out:
        raise ValueError(f'--seeds {list(tokens)!r} resolved to no seeds')
    return tuple(sorted(set(out)))


def seed_block(seed: int) -> str:
    """Which disjoint block a seed belongs to, or 'UNKNOWN'."""
    for name in _BLOCK_ORDER:
        if seed in registry.SEED_BLOCKS[name]:
            return name
    return 'UNKNOWN'


def check_seed_blocks(exp_ids: Sequence[str],
                      seeds: Optional[tuple[int, ...]]) -> tuple[list[str], list[str]]:
    """Refuse the seed-block violations that made revision 1's numbers unusable.

    Revision 1 selected hyperparameters on seeds 0-4 and then ran every
    confirmatory arm on 0-9, so half of each confirmatory sample had been tuned
    on (`DESIGN.md` §3.4, §11 item 2). An explicit ``--seeds`` overrides the
    block an experiment declares, which is exactly the gesture that reintroduces
    that leak, so it is checked here rather than discovered in `audit.py` after
    the compute is spent.

    All four of `DESIGN.md` §3.4's "never used for" rules are enforced, not one.
    The earlier version checked only two, and gated the TUNE refusal on
    ``family == 'confirmatory'``, which let ``--seeds TUNE`` through for every
    *estimation* and *screen* experiment with nothing but a ``[warning]``: E2
    supplies the three control contrasts, E8i is the C4 positive control with a
    pre-registered pass criterion, and the screens emit BH q-values. §3.4 says
    TUNE is never used for any confirmatory **or reported estimate**, and
    `ANALYSIS_PLAN.md` §8 forbids "any estimate computed on TUNE seeds"; neither
    is a statement about one family. The rule applied here is therefore the
    block's, not the family's: **only an experiment that declares TUNE may run
    on TUNE.**

    RESERVE and C4SRC had no rule at all. ``--seeds RESERVE`` planned eighty
    confirmatory jobs, and ``--seeds C4SRC`` silently *adopted* the four existing
    P0 donor runs as members of a confirmatory scratch arm, because run identity
    is the config digest and those directories already existed. A seed outside
    every declared block is refused for the same reason: it would be recorded as
    ``seed_block: UNKNOWN`` and no audit could classify it afterwards.

    Returns ``(fatal, warnings)``.
    """
    fatal: list[str] = []
    warn: list[str] = []
    if seeds is None:
        return fatal, warn
    requested = set(seeds)
    tune = set(registry.SEED_BLOCKS['TUNE'])
    reserve = set(registry.SEED_BLOCKS['RESERVE'])
    donor = set(registry.SEED_BLOCKS['C4SRC'])
    smoke = set(registry.SEED_BLOCKS['SMOKE'])
    estimation = set(registry.SEED_BLOCKS['CONFIRM']) | set(
        registry.SEED_BLOCKS['REPLICATE'])

    # Block-wide rules: they do not depend on which experiment asked, because
    # DESIGN.md 3.4 states them about the block itself.
    unknown = sorted(sd for sd in requested if seed_block(sd) == 'UNKNOWN')
    if unknown:
        fatal.append(
            f'seeds {unknown} belong to no block DESIGN.md §3.4 declares. '
            f'Every run records its block, and a run whose block is UNKNOWN '
            f'cannot be checked for selection leakage by audit.py afterwards. '
            f'Declared blocks: '
            + ', '.join(f'{k} {min(v)}..{max(v)}'
                        for k, v in registry.SEED_BLOCKS.items()) + '.')
    if requested & reserve:
        fatal.append(
            f'the requested seeds include {sorted(requested & reserve)} from '
            f'RESERVE. DESIGN.md §3.4 reserves that block for replacement '
            f'sources drawn by the validity gate and for nothing else; the '
            f'draw is made by the source phase (DESIGN.md 4.3) and recorded in '
            f'{REPLACEMENTS_FILE}, never selected by hand. Selecting it here '
            f'would put arbitrary runs in the block the gate draws from, so a '
            f'later replacement could not tell a drawn seed from a scheduled '
            f'one.')

    #: Seeds a block-wide rule has already refused. The per-experiment loop adds
    #: nothing for them, and a generic warning underneath a refusal reads as
    #: though the request were merely unusual.
    already_fatal = set(unknown) | (requested & reserve)
    for eid in exp_ids:
        exp = registry.EXPERIMENTS[eid]
        declared = set(registry.SEED_BLOCKS[exp.seed_block])
        if requested == declared or requested <= already_fatal:
            continue
        if exp.seed_block == 'TUNE' and requested & estimation:
            fatal.append(
                f'{eid} ({exp.name}) is a selection experiment declared on TUNE, '
                f'and the requested seeds include '
                f'{sorted(requested & estimation)} from CONFIRM/REPLICATE. '
                f'Selecting on seeds that later carry a reported estimate is '
                f'DESIGN.md §11 defect 2. Run it on TUNE.')
            continue
        if exp.seed_block != 'TUNE' and requested & tune:
            fatal.append(
                f'{eid} ({exp.name}) declares {exp.seed_block} and is family '
                f'{exp.family}, and the requested seeds include '
                f'{sorted(requested & tune)} from TUNE. DESIGN.md §3.4 says '
                f'TUNE is never used for any confirmatory or reported '
                f'estimate, and ANALYSIS_PLAN.md §8 forbids any estimate '
                f'computed on TUNE seeds: that covers estimation and screen '
                f'experiments as much as confirmatory ones. Only E3-style '
                f'selection experiments, which declare TUNE, may run on it.')
            continue
        if exp.seed_block != 'C4SRC' and requested & donor:
            fatal.append(
                f'{eid} ({exp.name}) is not a donor experiment and the '
                f'requested seeds include {sorted(requested & donor)} from '
                f'C4SRC. DESIGN.md §3.4 says C4SRC supplies positive-control '
                f'source checkpoints and is never used for target-side '
                f'estimation. The C4 donors are pulled in by the registry as '
                f'source runs where E8i needs them; asking for them as target '
                f'seeds adopts those existing run directories into this '
                f'experiment instead, because run identity is the config '
                f'digest.')
            continue
        if exp.seed_block != 'SMOKE' and requested & smoke:
            warn.append(
                f'{eid} ({exp.name}) is being run on SMOKE seed(s) '
                f'{sorted(requested & smoke)}. SMOKE is disjoint from every '
                f'other block so nothing leaks, but nothing produced on it is '
                f'a result either: it exists so a seed-block audit can tell a '
                f'pipeline-validation run from a real one by its seed alone.')
            continue
        warn.append(
            f'{eid} ({exp.name}) declares seed block {exp.seed_block} '
            f'({min(declared)}..{max(declared)}); running it on '
            f'{len(requested)} explicitly requested seeds instead.')
    return fatal, warn


# ---------------------------------------------------------------------------
# Job manifest
# ---------------------------------------------------------------------------
def parse_overrides(items: Sequence[str] | None) -> dict:
    """``key=value`` pairs, checked against the Config schema.

    Values go through ``literal_eval`` so ``num_episodes=14``,
    ``hidden=(64,64)`` and ``prefix_checkpoints=[6]`` all mean what they look
    like; anything that will not evaluate is kept as a string.

    Two classes are refused outright rather than warned about. The fields in
    ``_RUNNER_OWNED_FIELDS`` are assigned by the runner or the registry, so an
    override of them either does nothing (``seed``) or does something nobody
    asked for (``out_root`` moves the run directories out from under
    ``--out-root`` while ``_jobs/`` and ``_index/`` stay behind). And a budget of
    fewer than one episode is refused because ``num_episodes=0`` made
    ``is_complete`` read ``0 >= 0`` and certify an entire tree as finished:
    sixteen runs with ``episodes_completed: 0`` and ``final_score: null``,
    indexed as experiment members, and a re-run reporting "every job already has
    a complete manifest" at exit 0.
    """
    out: dict = {}
    fields = {f.name for f in Config.__dataclass_fields__.values()}
    for item in items or ():
        if '=' not in item:
            raise ValueError(f'--override expects key=value, got {item!r}')
        key, _, raw = item.partition('=')
        key = key.strip().replace('-', '_')
        if key not in fields:
            raise ValueError(f'--override {key!r} is not a Config field')
        if key in _RUNNER_OWNED_FIELDS:
            raise ValueError(
                f'--override {key}= is refused: {key} is assigned by the '
                f'runner or the registry, not by the command line. '
                + {'out_root': 'Use --out-root, which moves the job, status '
                               'and index files with the runs.',
                   'seed': 'Use --seeds; the registry sets seed per job, so an '
                           'override of it is silently ignored and the plan '
                           'you get is the catalogue plan.',
                   'source_checkpoint': 'The source is assigned by the '
                                        'validity gate (DESIGN.md 4.3).',
                   'source_seed': 'The source seed is assigned by the validity '
                                  'gate (DESIGN.md 4.3) and recorded in '
                                  f'{REPLACEMENTS_FILE}.',
                   }.get(key, 'It names the plan rather than changing it, and '
                              'it is bookkeeping, so the runs would keep the '
                              'catalogue digest and the catalogue directory.'))
        try:
            out[key] = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            out[key] = raw
    for key in ('num_episodes', 'max_steps'):
        if key not in out:
            continue
        try:
            value = int(out[key])
        except (TypeError, ValueError):
            raise ValueError(f'--override {key}={out[key]!r} is not a whole '
                             f'number of {key.split("_")[-1]}') from None
        if value < 1:
            raise ValueError(
                f'--override {key}={out[key]!r} is not a run. A budget below '
                f'one is not a smaller experiment, it is a directory tree '
                f'certified complete without training: at num_episodes=0 the '
                f'completion test reads 0 >= 0 and every manifest passes with '
                f'final_score null.')
        out[key] = value
    return out


def job_record(job: 'registry.Job', experiments: Sequence[str]) -> dict:
    """One line of ``jobs.jsonl``: identity, dependency, and the whole config.

    The config is carried in full rather than as flags. A worker that
    reconstructs ``Config(**record['config'])`` gets bit-identical run identity
    to the parent's, which is what makes the claim protocol meaningful -- two
    processes must agree on which directory a job owns.
    """
    cfg = job.cfg
    return {
        'job_id': f'{cfg.run_digest()[:12]}-s{cfg.seed:02d}',
        'experiment': job.experiment,
        'experiments': list(dict.fromkeys(experiments)),
        'arm': job.arm,
        'label': cfg.label,
        'role': job.role,
        'arm_id': cfg.arm_id(),
        'cell': f'{cfg.arch}-{cfg.target_rule}',
        'condition': cfg.condition,
        'env': cfg.env,
        'source_env': cfg.source_env,
        'seed': cfg.seed,
        'seed_block': seed_block(cfg.seed),
        'run_dir': cfg.run_dir(),
        'run_digest': cfg.run_digest(),
        'trajectory_digest': cfg.trajectory_digest(),
        'num_episodes': int(cfg.num_episodes),
        'depends_on': job.depends_on,
        'source_checkpoint': cfg.source_checkpoint,
        'needs_source_weights': cfg.condition in _WEIGHT_READING,
        # Which source this run draws on, and whether the validity gate moved it
        # off the default. The run directory cannot say -- `source_checkpoint`
        # is bookkeeping and outside the digest -- so the job record is where a
        # replacement becomes legible, and where the lineage check reads it.
        'source_arm': job.source_arm,
        'source_default_seed': job.source_default_seed,
        'source_seed': job.source_seed,
        'source_lineage': job.source_lineage,
        'source_replaced': job.source_replaced,
        'source_run_digest': None,
        'is_source': False,
        # Filled in by `build_plan` on a source that the validity gate drew from
        # RESERVE: the default seed it stands in for. Nothing else can say so.
        # The source's own record shows a RESERVE seed in an ordinary arm, and
        # `DESIGN.md` §3.4 gives RESERVE exactly one use, so a consumer that
        # cannot see this field has to guess.
        'source_replacement_for': None,
        'config': cfg.to_dict(),
    }


def build_plan(exp_ids: Sequence[str], seeds: Optional[tuple[int, ...]],
               out_root: str, overrides: dict | None = None,
               allow_factor_overrides: bool = False,
               source_seeds: Mapping[tuple[str, int], int] | None = None
               ) -> tuple[list[dict], dict[str, list[str]]]:
    """Resolve experiments into de-duplicated job records plus experiment membership.

    Two passes over the registry, and both are needed. ``all_jobs`` gives the
    de-duplicated list in dependency order -- sources before their consumers --
    which is the list that gets run. ``jobs`` per experiment gives *membership*,
    which ``all_jobs`` cannot: a run directory is deliberately not keyed by
    experiment (`src/dqn/config.py`, ``run_dir``), so one run can belong to
    several experiments and the de-duplicated list only remembers the first one
    that asked for it. `audit.py` needs the full membership, so it is computed
    here and written to ``_index/``.
    """
    jobs = registry.all_jobs(exp_ids, seeds, out_root, overrides,
                             allow_factor_overrides=allow_factor_overrides,
                             source_seeds=source_seeds)
    membership: dict[str, list[str]] = {}
    for eid in exp_ids:
        dirs = [j.cfg.run_dir() for j in registry.jobs(
            eid, seeds, out_root, overrides,
            allow_factor_overrides=allow_factor_overrides,
            source_seeds=source_seeds)]
        membership[eid] = list(dict.fromkeys(dirs))

    by_run: dict[str, list[str]] = {}
    for eid, dirs in membership.items():
        for run_dir in dirs:
            by_run.setdefault(run_dir, []).append(eid)

    records = [job_record(j, by_run.get(j.cfg.run_dir(), [j.experiment]))
               for j in jobs]

    # Two derived fields that need the whole plan, not one job. `is_source` is
    # membership of the phase-1 stage, and it is computed from the dependency
    # edges rather than from `role`: E8's shift arms draw their source from a
    # `scratch-*` arm whose declared role is 'target', and a role-based test
    # would leave those sources ungated -- which is the one place the gate is
    # most needed, since a scratch LunarLander run is exactly what the design
    # calls a source there.
    by_dir = {r['run_dir']: r for r in records}
    for rec in records:
        dep = rec.get('depends_on')
        if not dep:
            continue
        src = by_dir.get(dep)
        if src is None:
            continue
        src['is_source'] = True
        rec['source_run_digest'] = src['run_digest']
        if rec.get('source_replaced'):
            # Derived from the consumer because the source record cannot say
            # it: its seed is a RESERVE seed sitting in a normal arm, and
            # nothing in it distinguishes a seed the gate drew from a seed the
            # plan scheduled.
            src['source_replacement_for'] = rec.get('source_default_seed')
    return records, membership


def write_jobs(path: str, records: Sequence[dict]) -> None:
    """Publish the job list atomically, through a name only this process uses.

    The temporary carries the pid because `claim_run`'s docstring contemplates a
    second sweep started by hand on the same tree. On one fixed
    ``jobs.jsonl.tmp`` those two parents collide: on Windows the second ``open``
    raises PermissionError and takes that parent down with a traceback, or one
    of them publishes the other's half-written file. With a name per process
    each writes a complete file and the later ``os.replace`` wins, which is a
    result rather than a crash.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f'{path}.tmp-{os.getpid()}'
    with open(tmp, 'w', encoding='utf-8') as fh:
        for rec in records:
            fh.write(json.dumps(rec, sort_keys=True, default=str) + '\n')
    os.replace(tmp, path)


def read_jobs(path: str) -> list[dict]:
    out: list[dict] = []
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_index(out_root: str, membership: dict[str, list[str]],
                records: Sequence[dict],
                rejected_dirs: Sequence[str] = ()) -> list[str]:
    """``_index/<experiment>.jsonl``: which runs belong to which experiment.

    Written by the parent only, and *merged* with whatever is already there. A
    later invocation on a different seed set must not erase the earlier members,
    or `audit.py` would see an incomplete arm and refuse a complete one --
    reproducing by accident the silent seed-dropping of `DESIGN.md` §1.

    Merging is also why the rows have to carry their own provenance rather than
    relying on the index being pruned. A RESERVE replacement source is written
    here as an ordinary member of whatever arm it belongs to, and for E8 and E9
    the runs that *act* as sources declare ``role='target'`` (their sources come
    from a ``scratch-*`` arm), so an eleventh seed drawn from RESERVE was
    indistinguishable in ``arm`` from the ten genuine CONFIRM members of an arm
    that feeds a reported estimate. `DESIGN.md` §3.4 gives RESERVE exactly one
    use, so a consumer must be able to see that use in the row: hence
    ``is_source``, ``source_replacement_for``, ``source_replaced`` and
    ``source_rejected``.

    ``rejected_dirs`` are the run directories the validity gate has rejected,
    from the DESIGN.md 4.3 ledger. They are *marked*, not removed: this runner
    does not delete evidence, and a rejected source is part of what §4.3 requires
    the results table to report. Marking is applied to rows already on disk too,
    so an index written before a rejection is corrected by the next invocation.

    The read-merge-write runs under the same cross-process lock the status log
    uses, and publishes through a pid-qualified temporary. Unlocked, two parents
    on one tree interleave: both read the old file, each merges its own members
    into what it read, and whichever replaces second publishes an index missing
    the other's rows, which is the silent member loss the merge exists to
    prevent.
    """
    rejected = {str(d) for d in rejected_dirs}
    by_dir = {r['run_dir']: r for r in records}
    index_dir = os.path.join(out_root, INDEX_SUBDIR)
    os.makedirs(index_dir, exist_ok=True)
    written = []
    for eid, dirs in membership.items():
        path = os.path.join(index_dir, f'{eid}.jsonl')
        # The lock file sits beside the index as `<experiment>.jsonl.lock`,
        # holds no data, and does not match the `*.jsonl` any consumer globs.
        with _AppendLock(path):
            merged = _merge_index(path, dirs, by_dir, rejected, eid)
            tmp = f'{path}.tmp-{os.getpid()}'
            with open(tmp, 'w', encoding='utf-8') as fh:
                for key in sorted(merged):
                    fh.write(json.dumps(merged[key], sort_keys=True,
                                        default=str) + '\n')
            os.replace(tmp, path)
        written.append(path)
    return written


def _merge_index(path: str, dirs: Sequence[str], by_dir: Mapping[str, dict],
                 rejected: set[str], eid: str) -> dict[str, dict]:
    """The rows already on disk, corrected, plus this plan's own.

    Split out of `write_index` only so the locked region is one short block:
    the lock is not reentrant (see `_AppendLock`), so nothing in here may append
    a status record.
    """
    merged: dict[str, dict] = {}
    if os.path.exists(path):
        for line in open(path, encoding='utf-8'):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get('run_dir') in rejected:
                row['source_rejected'] = True
            merged[row.get('run_dir', '')] = row
    for run_dir in dirs:
        rec = by_dir.get(run_dir, {})
        merged[run_dir] = {
            'experiment': eid,
            'run_dir': run_dir,
            'run_digest': rec.get('run_digest'),
            'arm': rec.get('arm'),
            'label': rec.get('label'),
            'role': rec.get('role'),
            'arm_id': rec.get('arm_id'),
            'condition': rec.get('condition'),
            'seed': rec.get('seed'),
            'seed_block': rec.get('seed_block'),
            'depends_on': rec.get('depends_on'),
            # Membership of phase 1, computed from the dependency edges
            # rather than from `role`, which is wrong for exactly the arms
            # where it matters (see `build_plan`).
            'is_source': bool(rec.get('is_source')),
            'source_replacement_for': rec.get('source_replacement_for'),
            'source_seed': rec.get('source_seed'),
            'source_default_seed': rec.get('source_default_seed'),
            'source_replaced': bool(rec.get('source_replaced')),
            'source_rejected': run_dir in rejected,
        }
    return merged


# ---------------------------------------------------------------------------
# Completion, claims, status
# ---------------------------------------------------------------------------
def _remove_quietly(path: str) -> bool:
    """Delete a file we no longer need; tidying up is never a failure."""
    try:
        os.remove(path)
        return True
    except OSError:
        return False


def _json_safe(obj):
    """Replace non-finite floats with None, recursively.

    ``json.dumps(float('nan'))`` emits a bare ``NaN``, which is not valid strict
    JSON: Python reads it back, nothing else has to. The status log and the
    DESIGN.md 4.3 rejection ledger are both read by the analysis layer and by
    whatever a reviewer points at them, so a record that only Python can parse
    is a record that is not reportable. A non-finite number is written as null
    and the state that produced it is carried in its own field, never inferred
    from the number.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def manifest_result(run_dir: str) -> Optional[dict]:
    path = os.path.join(run_dir, 'manifest.json')
    try:
        with open(path, encoding='utf-8') as fh:
            return json.load(fh).get('result') or {}
    except (OSError, json.JSONDecodeError):
        return None


def _as_int(value: object) -> Optional[int]:
    """``int(value)`` or None. A manifest field is data, not a promise.

    ``int(result.get('episodes_completed') or 0)`` raised an uncaught
    ValueError on any manifest whose count was not a number, and `is_complete`
    is called from `print_plan`, `worker_main`, `summarise` and
    `lineage_conflicts`, so one malformed manifest anywhere in the tree aborted
    the whole planner with a traceback instead of a refusal.
    """
    try:
        return int(value)                       # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def completion_state(rec: dict) -> tuple[str, str]:
    """What this run's own manifest says about it: state and reason.

    Four states, and the fourth is the one that used to be missing:

    * ``unrun``    -- no manifest at all.
    * ``partial``  -- a manifest short of the budget; resume it.
    * ``complete`` -- the budget reached and the run's own integrity check
      clean.
    * ``unsound``  -- a manifest that cannot be read as evidence of a sound run.

    An unsound manifest is neither complete nor quietly resumable, and naming it
    separately is the point. `DESIGN.md` §8.2(1) exists because a crash between
    checkpoints duplicates metric rows and silently corrupts every window
    statistic downstream; ``train.py`` detects exactly that and records it in
    ``result.metrics_integrity``, printing "[WARNING] metrics integrity". The
    completion test never read it, so a manifest reporting
    ``unique_episodes >= num_episodes`` alongside ``contiguous: false`` was
    accepted as complete, skipped, and counted under "complete now". This runner
    is the only layer that could re-run such a directory, so it is the layer
    that has to refuse to call it finished.

    The integrity verdict is consulted only once the episode budget is reached.
    Below it the check's own "expected N episodes, found M" problem would fire
    on every interrupted run, and an interrupted run is a resume, not a defect.
    """
    result = manifest_result(rec['run_dir'])
    if result is None:
        return 'unrun', 'no manifest'
    want = _as_int(rec.get('num_episodes'))
    if want is None or want < 1:
        return 'unsound', (
            f'the job asks for {rec.get("num_episodes")!r} episodes. A budget '
            f'below one is not a smaller run: at num_episodes=0 the completion '
            f'test reads 0 >= 0 and certifies the whole tree as finished '
            f'without training anything')
    raw = result.get('episodes_completed')
    got = 0 if raw is None else _as_int(raw)
    if got is None:
        return 'unsound', (f'manifest result.episodes_completed is {raw!r}, '
                           f'which is not a count of episodes')
    if got < want:
        return 'partial', f'{got}/{want} episodes'
    integrity = result.get('metrics_integrity')
    if isinstance(integrity, dict):
        problems = [str(p) for p in (integrity.get('problems') or [])]
        if integrity.get('contiguous') is False or problems:
            return 'unsound', (
                f'the run reports {got}/{want} episodes but its own metrics '
                f'integrity check failed (DESIGN.md 8.2(1)): '
                + ('; '.join(problems) or 'contiguous: false'))
    return 'complete', f'{got}/{want} episodes'


def is_complete(rec: dict) -> bool:
    """A run is complete when its own manifest says so, and only then.

    Not "the directory exists", and not "a checkpoint is present". The audit
    found a completed directory being resumed under a different configuration,
    which trained zero episodes and then wrote a manifest whose config never
    described its metrics. An *unsound* manifest is not complete either; see
    `completion_state`, and `unsound_runs` for what is done about it.
    """
    return completion_state(rec)[0] == 'complete'


def unsound_runs(records: Sequence[dict]) -> list[dict]:
    """Runs whose manifest is neither trustworthy nor safely resumable.

    Reported and refused rather than repaired, for the same reason
    `lineage_conflicts` is: this runner does not delete data, and a directory
    holding duplicated or non-contiguous episode rows is evidence of something.
    Re-entering it would not help either, because ``train.py`` resumes and a run
    already at its budget trains nothing, so the sweep would spin.
    """
    out: list[dict] = []
    for rec in records:
        state, why = completion_state(rec)
        if state != 'unsound':
            continue
        out.append({'run_dir': rec['run_dir'], 'job_id': rec['job_id'],
                    'arm': rec['arm'], 'seed': rec['seed'],
                    'experiment': rec['experiment'], 'reason': why})
    return out


def print_unsound_runs(rows: Sequence[dict]) -> None:
    """Name every directory whose manifest cannot be read as a finished run."""
    if not rows:
        return
    print('\n' + '=' * 72)
    print(f'[ERROR] {len(rows)} run(s) have a manifest that is not evidence of '
          f'a sound run')
    print('=' * 72)
    for row in rows:
        print(f'\n  {row["job_id"]}  {row["experiment"]}/{row["arm"]} '
              f'seed {row["seed"]}')
        print(f'    directory: {row["run_dir"]}')
        print(f'    reason:    {row["reason"]}')
    print('\n  Neither counted as complete nor resumed. DESIGN.md 8.2(1) '
          'exists because duplicated')
    print('  metric rows silently corrupt every window statistic downstream, '
          'and train.py already')
    print('  reports the condition; ignoring it here is what made it '
          'invisible. Move or delete')
    print('  the directories above and re-run: this runner will not delete '
          'data.')


# ---------------------------------------------------------------------------
# Claims: who owns a run directory, and when that may be taken away
# ---------------------------------------------------------------------------
def read_claim(path: str) -> Optional[dict]:
    """The claim on a run directory.

    None when there is none; ``{}`` when one is present but unreadable.

    The distinction is load-bearing and used to be lost. The old version
    returned ``{}`` for every failure including "no such file", so
    `_reclaim_reason`'s ``if claim is None`` guard was dead code and
    `dependency_state`'s ``if claim:`` read an empty or torn claim as *no claim
    at all*. `claim_run` creates the file with ``os.open`` and writes the
    payload in a separate ``os.write``, so there is a window, however short, in
    which the file exists and is empty; a reader inside it would classify an
    actively claimed dependency as unclaimed and, if that job was not in its own
    remaining list, declare the consumer **blocked**, which is a terminal,
    non-zero-exit verdict.
    """
    try:
        with open(path, encoding='utf-8') as fh:
            text = fh.read()
    except FileNotFoundError:
        return None
    except OSError:
        return {}
    try:
        claim = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return claim if isinstance(claim, dict) else {}


def claim_owner(sweep_id: str, worker: str) -> dict:
    """This process's identity, as written into a claim and checked back out.

    Four fields, all four compared. A pid alone is not an identity across hosts,
    and a sweep id alone is not one across workers.
    """
    return {'pid': os.getpid(), 'host': socket.gethostname(),
            'sweep': sweep_id, 'worker': worker}


def _is_owner(claim: Optional[Mapping], owner: Mapping) -> bool:
    if not claim:
        return False
    return all(claim.get(key) == owner[key]
               for key in ('pid', 'host', 'sweep', 'worker'))


#: Loaded once. `_pid_alive` is called on every contested claim, and a deferred
#: worker re-checks every ``--poll-seconds``, so re-binding the library per call
#: turned a liveness question into a measurable cost on a large job list.
_KERNEL32 = None


def _win_kernel32():
    global _KERNEL32
    if _KERNEL32 is None:
        lib = ctypes.WinDLL('kernel32', use_last_error=True)
        lib.OpenProcess.restype = ctypes.c_void_p
        lib.OpenProcess.argtypes = (ctypes.c_ulong, ctypes.c_int,
                                    ctypes.c_ulong)
        lib.CloseHandle.argtypes = (ctypes.c_void_p,)
        lib.GetExitCodeProcess.argtypes = (ctypes.c_void_p,
                                           ctypes.POINTER(ctypes.c_ulong))
        _KERNEL32 = lib
    return _KERNEL32


def _pid_alive(pid: object) -> Optional[bool]:
    """True, False, or **None** when this process cannot tell.

    None is a real answer and is treated as one downstream: "I do not know" must
    never collapse into "it is dead", because assuming death is what let a live
    trainer's directory be taken away from it while it was still writing.

    Pid reuse is the residual risk and it is left as a refusal rather than
    papered over: if a dead worker's pid has been recycled by an unrelated
    process, this reports alive and the directory is never reclaimed
    automatically. ``--force-claim`` is then the operator's deliberate gesture,
    which is the safe direction for the error to point, and `_held_note` says so
    out loud once a live owner has stopped refreshing its own claim: that
    combination is the signature of a recycled pid, and left silent it reads as
    an ordinary wait.
    """
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return None
    if pid == os.getpid():
        return True
    if os.name == 'nt':
        try:
            kernel32 = _win_kernel32()
            # PROCESS_QUERY_LIMITED_INFORMATION: the narrowest right that
            # answers the question, and the one that still works when the
            # target process belongs to another user.
            handle = kernel32.OpenProcess(0x1000, False, pid)
            if handle:
                code = ctypes.c_ulong(0)
                ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
                kernel32.CloseHandle(handle)
                if not ok:
                    return None
                return code.value == 259            # STILL_ACTIVE
            err = ctypes.get_last_error()
            if err == 87:                           # ERROR_INVALID_PARAMETER
                return False                        # no such process
            if err == 5:                            # ERROR_ACCESS_DENIED
                return True                         # it exists
            return None
        except (OSError, AttributeError, ValueError):
            return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True                                 # exists, not ours to signal
    except OSError:
        return None
    return True


def _owner_liveness(claim: Mapping) -> tuple[Optional[bool], str]:
    """Is the process that wrote this claim still running, and how do we know?"""
    pid = claim.get('pid')
    host = claim.get('host')
    if host and host != socket.gethostname():
        return None, (f'owner {pid}@{host} is on another host, so its '
                      f'liveness cannot be checked from here')
    alive = _pid_alive(pid)
    if alive is True:
        return True, f'owner pid {pid} is still running on this host'
    if alive is False:
        return False, f'owner pid {pid} is gone from this host'
    return None, f'owner pid {pid!r} cannot be checked on this host'


def _claim_payload(run_dir: str, owner: Mapping) -> dict:
    now = time.time()
    stamp = time.strftime('%Y-%m-%dT%H:%M:%S')
    return {'pid': owner['pid'], 'host': owner['host'],
            'sweep': owner['sweep'], 'worker': owner['worker'],
            'run_dir': run_dir, 'state': 'running',
            'time': stamp, 'epoch': now,
            'heartbeat': now, 'heartbeat_t': stamp,
            'heartbeat_seconds': HEARTBEAT_SECONDS}


def heartbeat_claim(run_dir: str, owner: Mapping) -> tuple[str, str]:
    """Refresh this process's own claim. Returns ``(status, detail)``.

    ``status`` is ``refreshed``, ``lost`` (the claim is gone or now belongs to
    somebody else, which is not retryable) or ``error`` (a write that failed and
    may succeed next time).

    Without this the claim's timestamp was written once, at the start, and the
    reclaim rule read its age as "how long since anything happened here". For a
    run longer than ``--stale-seconds`` that reading is simply wrong, and in P0
    it took a 14.4 h source run away from its owner at 14.2 h. With a refresh
    every `HEARTBEAT_SECONDS` the age means what the rule needs it to mean: how
    long since the owner was last demonstrably alive.

    The refresh is a rename, not a truncate-and-rewrite, so no reader ever sees
    a half-written claim.
    """
    path = os.path.join(run_dir, CLAIM_NAME)
    claim = read_claim(path)
    if claim is None:
        return 'lost', 'the claim file is gone'
    if not _is_owner(claim, owner):
        return 'lost', (f'the claim now belongs to {claim.get("worker")}/'
                        f'{claim.get("pid")}@{claim.get("host")} in sweep '
                        f'{claim.get("sweep")}')
    claim.update(heartbeat=time.time(),
                 heartbeat_t=time.strftime('%Y-%m-%dT%H:%M:%S'),
                 heartbeat_seconds=HEARTBEAT_SECONDS)
    tmp = f'{path}.hb-{os.getpid()}'
    try:
        with open(tmp, 'w', encoding='utf-8') as fh:
            json.dump(_json_safe(claim), fh, sort_keys=True)
        os.replace(tmp, path)
    except OSError as exc:
        _remove_quietly(tmp)
        return 'error', f'could not refresh the claim: {exc}'
    return 'refreshed', 'refreshed'


class ClaimHeartbeat:
    """Keep a claim fresh for as long as its run is training.

    A daemon thread rather than anything cleverer, because the alternative is to
    interleave the refresh with ``train()``, which is one opaque blocking call.
    The thread only touches its own claim file and only through
    `heartbeat_claim`, which re-checks ownership on every pass, so the worst it
    can do when the directory has changed hands is notice and stop.

    Losing the claim mid-run should now be impossible without
    ``--force-claim``, but it is detected anyway and recorded: `lost_reason` is
    what the worker reports afterwards, because a run that finished without
    owning its directory is not something to report as an ordinary success.
    """

    def __init__(self, run_dir: str, owner: Mapping,
                 interval: int = HEARTBEAT_SECONDS) -> None:
        self.run_dir = run_dir
        self.owner = dict(owner)
        self.interval = max(1, int(interval))
        self.lost_reason: Optional[str] = None
        self.beats = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            status, detail = heartbeat_claim(self.run_dir, self.owner)
            if status == 'refreshed':
                self.beats += 1
                continue
            if status == 'lost':
                self.lost_reason = detail
                return
            # 'error': transient, so keep trying. The claim keeps its previous
            # timestamp meanwhile, which is the conservative direction: it ages
            # towards being reclaimable rather than away from it.

    def __enter__(self) -> 'ClaimHeartbeat':
        self._thread = threading.Thread(
            target=self._loop, daemon=True,
            name='claim-heartbeat-' + os.path.basename(self.run_dir))
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)


def _held_note(path: str) -> str:
    """Why a claim could not be taken, in terms an operator can act on."""
    held = read_claim(path)
    if not held:
        return ('held by a claim this process cannot read; --force-claim takes '
                'it over deliberately')
    alive, _evidence = _owner_liveness(held)
    liveness = {True: 'alive', False: 'gone', None: 'unknown'}[alive]
    age: Optional[float] = None
    try:
        age = time.time() - os.path.getmtime(path)
        touched = f'{age:.0f}s ago'
    except OSError:
        touched = 'at an unreadable time'
    note = (f'held by {held.get("worker")}/{held.get("pid")}@'
            f'{held.get("host")} (sweep {held.get("sweep")}, owner {liveness}, '
            f'last touched {touched})')
    # A live pid that has stopped refreshing its own claim is the signature of
    # pid reuse, and pid reuse is the one way a crashed worker's directory
    # becomes permanently unreclaimable: `_pid_alive` can only answer whether a
    # pid exists, Windows recycles pids aggressively, and an unrelated process
    # holding the dead worker's number reports alive for ever. Refusing is the
    # right direction, but refusing in silence is not: the sweep then waits out
    # --max-wait-seconds on a run nothing is training and the operator has no
    # way to tell that from an ordinary long run.
    interval = _as_int(held.get('heartbeat_seconds')) or HEARTBEAT_SECONDS
    if alive is True and age is not None and age > max(10 * interval, 600):
        note += (f'. NOTE: that owner is alive but has not refreshed this claim '
                 f'for {age / 60:.0f} min, far beyond its own {interval}s '
                 f'heartbeat. Either it is wedged, or it died and its pid has '
                 f'been reused by an unrelated process, in which case nothing '
                 f'is training here and only --force-claim gets past it')
    return note


def live_claim_holder(run_dir: str) -> Optional[str]:
    """Who is demonstrably alive inside `run_dir`, or None.

    'Demonstrably' means `_owner_liveness` answering True, which is the same
    evidence `_reclaim_reason` demands before it will take a directory away from
    anybody. Unknown liveness is not alive: a claim from another host cannot be
    checked from here, and reading that as an owner would let an unreachable
    claim suppress a verdict that ought to be reported.

    It exists so that "another worker is training this" and "nobody will ever
    build this" stop being the same answer. `DESIGN.md` §8.4 makes **blocked** a
    terminal verdict, and a job whose directory is held by a live owner has not
    earned one.
    """
    claim = read_claim(os.path.join(run_dir, CLAIM_NAME))
    if not claim:
        return None
    alive, evidence = _owner_liveness(claim)
    if alive is not True:
        return None
    return (f'{claim.get("worker")}/{claim.get("pid")}@{claim.get("host")} in '
            f'sweep {claim.get("sweep")} ({evidence})')


def purge_claim_litter(*run_dirs: str) -> int:
    """Remove the claim temporaries an interrupted process leaves behind.

    Three names, all of them beside a live ``.claim``, none of them read by
    anything: ``.claim.superseded-*`` from a reclaim, ``.claim.hb-*`` from a
    heartbeat refresh and ``.claim.fail-*`` from a failure mark. Each is created
    and then renamed or removed by the call that wrote it, so one still on disk
    is the residue of a process killed inside that window. Nothing read them and
    nothing deleted them; two superseded files were still sitting in the P0 tree
    when the verification pass found them, and the other two prefixes were not
    covered at all, so a hard kill left a file per killed worker for ever.

    The heartbeat and failure temporaries are removed only once they are older
    than `CLAIM_TEMP_MIN_AGE_SECONDS`, because a fresh one is a rename in flight
    in another process and deleting it under that process turns a refresh into
    an error for nothing. A superseded file has no such window: the rename that
    creates it has already decided the reclaim race, and its own writer removes
    it in the next statement.
    """
    removed = 0
    now = time.time()
    for run_dir in run_dirs:
        try:
            names = os.listdir(run_dir)
        except OSError:
            continue
        for name in names:
            path = os.path.join(run_dir, name)
            if name.startswith(SUPERSEDED_PREFIX):
                removed += int(_remove_quietly(path))
                continue
            if not name.startswith((HEARTBEAT_PREFIX, FAILMARK_PREFIX)):
                continue
            try:
                age = now - os.path.getmtime(path)
            except OSError:
                continue
            if age >= CLAIM_TEMP_MIN_AGE_SECONDS:
                removed += int(_remove_quietly(path))
    return removed


def _reclaim_reason(path: str, stale_seconds: int, sweep_id: str,
                    force: bool) -> Optional[str]:
    """Whether an existing claim may be taken over, and why.

    Three grounds, all of which leave a ``reclaimed`` record in the status log:

    * **stale** -- the claim has not been heartbeaten for ``stale_seconds``
      (default 2 h) **and there is evidence the owner is gone**. This is the
      crashed-worker case; a Colab or Kaggle session that dies mid-run leaves
      exactly this.
    * **failed** -- the claim records a *plain* failure from an *earlier*
      sweep. The failure is already recorded in ``status.jsonl``, and
      ``--resume`` is a request to try again, so holding the directory would
      turn one crash into a permanently unrunnable job. A **contended** failure
      does not qualify and falls through to the stale ground, which does require
      evidence of death: a worker that failed after taking a directory over says
      nothing about whoever it took it from, who may still be training in there,
      and this ground has neither an age wait nor a liveness test. Requiring
      liveness on this ground instead would be wrong in the other direction: a
      worker that fails one job carries straight on with the next, so its pid is
      alive while the directory it failed in is idle, and a second sweep started
      by hand could then never retry it.
    * **forced** -- ``--force-claim`` with a claim from another sweep.

    A claim written by *this* sweep is never reclaimed on the failed ground: a
    deterministic failure would otherwise be retried forever inside one
    invocation.

    The evidence requirement on the stale ground is the fix for the defect that
    fired in P0. Age alone is not evidence: with the timestamp written once at
    the start, every run longer than ``--stale-seconds`` looked abandoned, and
    the branch did not even require the reclaiming worker to belong to a
    different sweep. What counts as evidence now:

    * the owner's pid is **gone from this host** -- conclusive;
    * the owner is on **another host**, so its pid cannot be checked here, *and*
      the claim carries a ``heartbeat`` field, meaning it was written by a
      version that would have refreshed it had it been alive;
    * anything else, including a claim from before heartbeating existed and a
      pid that is still running: **not reclaimed**. ``--force-claim`` is the
      deliberate gesture, and it is recorded.
    """
    claim = read_claim(path)
    if claim is None:
        return None
    if not claim:
        # Present but unreadable: nothing in it identifies an owner, so nothing
        # in it can be evidence that the owner is dead.
        return '--force-claim over an unreadable claim' if force else None
    try:
        touched = os.path.getmtime(path)
    except OSError:
        # The recorded heartbeat is the fallback, not a second opinion. Taking
        # the later of the two would let a claim written on a host whose clock
        # runs fast become permanently unreclaimable, and --force would be the
        # only way out of an ordinary crash.
        try:
            touched = float(claim.get('heartbeat') or claim.get('epoch') or 0)
        except (TypeError, ValueError):
            return None
    age = time.time() - touched
    other_sweep = claim.get('sweep') != sweep_id
    if force and other_sweep:
        return f'--force-claim over sweep {claim.get("sweep")}'
    if (claim.get('state') == 'failed' and other_sweep
            and not claim.get('contended')):
        return f'failed in sweep {claim.get("sweep")}'
    if age <= stale_seconds:
        return None
    alive, evidence = _owner_liveness(claim)
    if alive is True:
        return None
    stale = f'stale ({age / 3600:.1f} h > {stale_seconds / 3600:.1f} h)'
    if alive is False:
        return f'{stale}; {evidence}'
    if claim.get('heartbeat') is not None:
        return (f'{stale}; {evidence}, and the claim has gone unheartbeaten '
                f'for longer than its own {claim.get("heartbeat_seconds")}s '
                f'refresh interval')
    return None


def claim_run(run_dir: str, sweep_id: str, worker: str, stale_seconds: int,
              force: bool) -> tuple[bool, str]:
    """Take exclusive ownership of a run directory, atomically.

    ``O_CREAT|O_EXCL`` is the whole mechanism: the filesystem decides, so no
    lock file, no registry of live workers and no scheduler cleverness is needed,
    and the guarantee holds across independent invocations on the same tree --
    including a second sweep started by hand while the first is still running.
    What that guarantee is worth depends entirely on `_reclaim_reason`, which is
    the only thing that can hand an owned directory to somebody else.

    Returns ``(acquired, note)``; ``note`` is 'claimed', a reclaim reason naming
    the previous owner, or a description of who holds it and whether that owner
    is still alive.
    """
    owner = claim_owner(sweep_id, worker)
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, CLAIM_NAME)
    payload = json.dumps(_claim_payload(run_dir, owner),
                         sort_keys=True).encode('utf-8')

    note = 'claimed'
    for attempt in (0, 1):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if attempt:
                # The reclaim succeeded and somebody else claimed the directory
                # in the gap. Saying 'reclaimed' here would report an ownership
                # this process does not have.
                return False, (f'lost the directory to another worker '
                               f'immediately after {note}')
            reason = _reclaim_reason(path, stale_seconds, sweep_id, force)
            if reason is None:
                return False, _held_note(path)
            held = read_claim(path) or {}
            superseded = (f'{path}.superseded-{int(time.time() * 1000)}-'
                          f'{os.getpid()}')
            try:
                # Rename rather than unlink: the rename is atomic, so if two
                # workers judge the same claim stale only one of them can move
                # it out of the way and the loser sees ENOENT.
                os.replace(path, superseded)
            except OSError:
                return False, 'lost the reclaim race'
            # The rename has decided the race, and the previous owner travels
            # into the note and from there into status.jsonl, so the file has
            # nothing left to say. Keeping it only left litter.
            _remove_quietly(superseded)
            note = (f'reclaimed: {reason}; previous owner '
                    f'{held.get("worker")}/{held.get("pid")}@'
                    f'{held.get("host")} in sweep {held.get("sweep")}')
            continue
        try:
            os.write(fd, payload)
        finally:
            os.close(fd)
        purge_claim_litter(run_dir)
        return True, note
    return False, note


def claim_failed_here(run_dir: str, sweep_id: str) -> bool:
    """Did this sweep already fail this run, in a way that is settled?

    Needed so that a *second* worker reaching a job another worker has already
    failed drops it silently instead of announcing it as blocked. Without this
    the later, weaker record ('held by pid 26500') overwrote the earlier, real
    one ('failed: refusing to resume ...') and the summary reported a failure as
    a blockage, with no log path to read.

    A **contended** failure does not count. That is a worker that failed after
    taking the directory over from somebody else, so the directory may well
    still be in competent hands; treating it as settled would drop a job that is
    about to finish.
    """
    claim = read_claim(os.path.join(run_dir, CLAIM_NAME)) or {}
    return (claim.get('state') == 'failed'
            and claim.get('sweep') == sweep_id
            and not claim.get('contended'))


def release_claim(run_dir: str, owner: Mapping) -> tuple[bool, str]:
    """Drop *this process's* claim on success. The manifest is now the evidence.

    The ownership test is the whole point of the signature change. Without it
    this removed ``<run_dir>/.claim`` whoever had written it, and in P0 that is
    exactly what happened: the original owner finished eleven minutes after a
    second worker had reclaimed its directory, and its release deleted the
    *second* worker's claim, because its own had already been renamed away. A
    worker that no longer owns the directory says so and leaves the file alone.
    """
    path = os.path.join(run_dir, CLAIM_NAME)
    claim = read_claim(path)
    if claim is None:
        return False, 'no claim to release'
    if not _is_owner(claim, owner):
        return False, (f'not released: the claim now belongs to '
                       f'{claim.get("worker")}/{claim.get("pid")}@'
                       f'{claim.get("host")} in sweep {claim.get("sweep")}')
    try:
        os.remove(path)
    except OSError as exc:
        return False, f'could not remove the claim: {exc}'
    return True, 'released'


def fail_claim(run_dir: str, error: str, owner: Mapping,
               contended: bool = False) -> tuple[bool, str]:
    """Mark *this process's* claim failed, so nothing silently re-enters it.

    ``contended`` records that this worker took the directory over from somebody
    else rather than claiming it fresh, and it is the difference between a
    failed run and a failed reclaim. `dependency_state` blocks a consumer on the
    first and defers it on the second. In P0 a ``PermissionError`` raised by two
    trainers colliding over one ``metrics.jsonl`` was written here as a plain
    failure, and the consumer of a source that was healthy and eleven minutes
    from finishing was reported **blocked**, which under `DESIGN.md` §8.4 is a
    terminal, non-zero-exit verdict.
    """
    path = os.path.join(run_dir, CLAIM_NAME)
    claim = read_claim(path)
    if claim is None:
        return False, 'no claim to mark failed'
    if not _is_owner(claim, owner):
        return False, (f'not marked failed: the claim now belongs to '
                       f'{claim.get("worker")}/{claim.get("pid")}@'
                       f'{claim.get("host")} in sweep {claim.get("sweep")}')
    claim.update(state='failed', error=str(error)[:2000],
                 contended=bool(contended),
                 failed_at=time.strftime('%Y-%m-%dT%H:%M:%S'))
    tmp = f'{path}.fail-{os.getpid()}'
    try:
        with open(tmp, 'w', encoding='utf-8') as fh:
            json.dump(_json_safe(claim), fh, sort_keys=True)
        os.replace(tmp, path)
    except OSError as exc:
        _remove_quietly(tmp)
        return False, f'could not mark the claim failed: {exc}'
    return True, 'marked failed'


#: How long an appender waits for the cross-process lock before giving up and
#: writing anyway. Every record here is a few hundred bytes, so the holder is
#: never inside the lock for long; a wait this size is contention, not deadlock.
_APPEND_LOCK_SECONDS = 20.0


class _AppendLock:
    """Serialise appends to one file across processes.

    ``O_APPEND`` is **not** the guarantee this file used to claim it was. On
    POSIX the kernel takes the offset and a short write is atomic, but on
    Windows the CRT implements append as seek-to-end then write, and the two are
    not one operation: two workers can take the same offset and one record
    overwrites the other. Measured on this tree: six processes writing 200
    records each into one ``status.jsonl`` lost **127 of 1200**, and every call
    reported success, because losing the race is not an error anybody sees.

    That is not a cosmetic loss. `summarise` reads ``status.jsonl`` for the
    error and the log path of every failure, and a dropped ``failed`` record
    makes a run report as "not finished" with neither; the DESIGN.md 4.3
    rejection ledger goes through the same function, and a dropped row is a
    rejected source seed missing from the results table §4.3 requires it to
    appear in.

    A sidecar ``<file>.lock`` holds a one-byte range lock: ``msvcrt.locking`` on
    Windows, ``fcntl.flock`` elsewhere. Acquiring it is best effort. If it
    cannot be taken within `_APPEND_LOCK_SECONDS` the write still happens, and
    says so on stderr, because a status record that might be lost is worth more
    than a runner that stops to wait for one. Every path that gives up on the
    lock says so; the one that could not open the lock file at all used to
    degrade in silence, which is the loss this class exists to prevent with the
    single signal that would have revealed it suppressed.

    **Not reentrant, and not cheaply made so.** ``msvcrt.locking`` and
    ``fcntl.flock`` are both per descriptor and this opens a fresh one on each
    acquisition, so taking the lock on a file this process already holds blocks
    against itself for the full `_APPEND_LOCK_SECONDS` and then writes
    unserialised anyway. No path nests today. Writing a status record from
    inside a locked region would be one, so do not add it: log after the block,
    not within it.
    """

    def __init__(self, path: str) -> None:
        self.path = path + '.lock'
        self.fd: Optional[int] = None
        self.held = False

    def __enter__(self) -> '_AppendLock':
        try:
            self.fd = os.open(self.path, os.O_CREAT | os.O_RDWR)
        except OSError as exc:
            # The same degradation as the timeout below, and it is announced for
            # the same reason: appending unserialised is the right fallback,
            # doing it quietly is not. An antivirus sharing violation, a full or
            # read-only _jobs directory and a path-length failure all land here.
            print(f'[WARNING] could not open the append lock {self.path} '
                  f'({exc}); appending unserialised, so a concurrent record '
                  f'may be lost', file=sys.stderr, flush=True)
            return self
        deadline = time.time() + _APPEND_LOCK_SECONDS
        delay = 0.002
        while True:
            try:
                if os.name == 'nt':
                    import msvcrt                       # noqa: PLC0415
                    msvcrt.locking(self.fd, msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl                        # noqa: PLC0415
                    fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.held = True
                return self
            except (OSError, ImportError):
                if time.time() >= deadline:
                    print(f'[WARNING] could not take the append lock on '
                          f'{self.path} within {_APPEND_LOCK_SECONDS:.0f}s; '
                          f'appending unserialised, so a concurrent record may '
                          f'be lost', file=sys.stderr, flush=True)
                    return self
                time.sleep(delay)
                delay = min(0.05, delay * 1.6)

    def __exit__(self, *_exc) -> None:
        if self.fd is None:
            return
        try:
            if self.held:
                if os.name == 'nt':
                    import msvcrt                       # noqa: PLC0415
                    os.lseek(self.fd, 0, os.SEEK_SET)
                    msvcrt.locking(self.fd, msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl                        # noqa: PLC0415
                    fcntl.flock(self.fd, fcntl.LOCK_UN)
        except (OSError, ImportError):
            pass
        finally:
            try:
                os.close(self.fd)
            except OSError:
                pass
            self.fd = None
            self.held = False


def append_status(path: str, record: dict) -> bool:
    """Append one state transition. Returns whether it was actually written.

    One ``os.write`` of one short self-contained JSON line, taken under the
    cross-process lock of `_AppendLock` because ``O_APPEND`` alone does not
    serialise appenders on Windows: see that class for the measurement. A torn
    line is still possible if the lock could not be taken, and is still
    detectable by the reader rather than corrupting its neighbours.

    Three things it no longer does. It no longer returns as though it had
    succeeded when every attempt failed: the retries were silent, so
    `append_replacements` reported "wrote N source-validity rejection(s)" with
    nothing on disk, and a dropped ``failed`` record made a run report as "not
    finished" with no error and no log path. It no longer discovers a missing
    parent directory one 1.05 s retry cycle at a time; the directory is created
    once, up front, which is what a fixture with no ``_jobs/`` needed and did
    not get. And it no longer loses roughly one record in ten under four
    concurrent workers. And a short write no longer passes for a whole record:
    the byte count is checked and the remainder written under the same lock.
    """
    parent = os.path.dirname(path)
    if parent:
        try:
            os.makedirs(parent, exist_ok=True)
        except OSError as exc:
            dropped = json.dumps(_json_safe(record), sort_keys=True,
                                 default=str)
            print(f'[ERROR] cannot create {parent} for '
                  f'{os.path.basename(path)}: {exc}. The record below is '
                  f'lost, not queued:\n  {dropped}',
                  file=sys.stderr, flush=True)
            return False
    line = (json.dumps(_json_safe(record), sort_keys=True, default=str)
            + '\n').encode('utf-8')
    # O_BINARY keeps the descriptor out of the CRT's text mode. Without it
    # Windows rewrites the buffer on the way through, which is why every record
    # in the P0 log ends CRLF from a buffer that ends LF: the write is then not
    # the single unmodified syscall this docstring describes, and the byte count
    # it returns cannot be compared against the buffer at all.
    flags = os.O_CREAT | os.O_APPEND | os.O_WRONLY | getattr(os, 'O_BINARY', 0)
    last: Optional[OSError] = None
    for attempt in range(6):
        try:
            with _AppendLock(path):
                fd = os.open(path, flags)
                try:
                    written = 0
                    while written < len(line):
                        # The byte count used to be discarded. A short write
                        # then left a truncated record that `read_status` skips
                        # in silence: the same lost record the lock was added to
                        # prevent, with no error anywhere. The remainder goes
                        # out under the same lock, so it lands against the rest
                        # of its own record; a write that accepts nothing is not
                        # progress and is raised into the retry, which appends
                        # the whole line afresh and leaves the torn prefix for
                        # the reader to skip.
                        n = os.write(fd, line[written:])
                        if n <= 0:
                            raise OSError(
                                f'os.write accepted {n} of '
                                f'{len(line) - written} remaining byte(s)')
                        written += n
                finally:
                    os.close(fd)
            return True
        except OSError as exc:
            last = exc
            time.sleep(0.05 * (attempt + 1))
    print(f'[ERROR] could not append to {path} after 6 attempts ({last}). The '
          f'record below is lost; the sweep summary reads this file, so treat '
          f'its counts as incomplete:\n  {line.decode("utf-8").strip()}',
          file=sys.stderr, flush=True)
    return False


def read_status(path: str, sweep_id: str | None = None) -> list[dict]:
    """Status records, skipping any line a concurrent append tore."""
    out: list[dict] = []
    if not os.path.exists(path):
        return out
    with open(path, encoding='utf-8', errors='replace') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if sweep_id is None or rec.get('sweep') == sweep_id:
                out.append(rec)
    return out


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
def dependency_state(rec: dict, pending: set[str],
                     sweep_id: str) -> tuple[str, str]:
    """Classify a job's dependency as ready / wait / blocked.

    'blocked' is reserved for the cases where waiting cannot help: the source
    failed in this sweep, or it is not in the job list at all so nobody will
    ever build it. Everything else waits, because the claim rule makes waiting
    self-healing -- a dead worker's claim is taken over by the waiting worker
    once there is evidence its owner is gone.

    Two things no longer produce a 'blocked' verdict, because in P0 neither of
    them meant what it was read to mean.

    * A **contended** failure. `fail_claim` records whether the worker had taken
      the directory over from somebody else; a failure that follows a reclaim
      may be the collision rather than the run. In P0 a ``PermissionError`` from
      two trainers colliding over one ``metrics.jsonl`` was read here as "source
      run failed in this sweep" and the consumer of a healthy source that
      finished eleven minutes later was reported blocked, which is terminal and
      exits non-zero.
    * A claim that exists but **cannot be read**. `claim_run` creates the file
      and writes the payload in two steps, so an empty claim means a claim being
      made, not the absence of one. The old ``if claim:`` read it as absence and
      could send a consumer straight to 'blocked'.

    ``stale_seconds`` used to be a parameter here and was never read: a dead
    argument advertising a staleness check on dependencies that does not exist.
    It is gone. Waiting is what handles a stale dependency.
    """
    dep = rec.get('depends_on')
    if not dep:
        return 'ready', ''
    if manifest_result(dep) is not None:
        if rec.get('needs_source_weights') and rec.get('source_checkpoint') \
                and not os.path.exists(rec['source_checkpoint']):
            return 'blocked', (f'source run finished but its checkpoint is '
                               f'missing: {rec["source_checkpoint"]}')
        return 'ready', ''

    claim = read_claim(os.path.join(dep, CLAIM_NAME))
    if claim is not None:
        if not claim:
            return 'wait', (f'source run is claimed but the claim is not yet '
                            f'readable, so it is being written now: {dep}')
        if claim.get('state') == 'failed' and claim.get('sweep') == sweep_id:
            if claim.get('contended'):
                return 'wait', (
                    f'source run failed after a contested claim, so the '
                    f'failure may be the collision rather than the run; '
                    f'waiting for whoever owns it now: {dep}')
            return 'blocked', f'source run failed in this sweep: {dep}'
        return 'wait', f'source run in progress: {dep}'
    if dep in pending:
        return 'wait', f'source run not started yet: {dep}'
    return 'blocked', (f'source run {dep} has no manifest and is not in this '
                       f'sweep\'s job list')


# ---------------------------------------------------------------------------
# Source validity and the RESERVE replacement rule (DESIGN.md 4.3)
# ---------------------------------------------------------------------------
def manifest_of(run_dir: str) -> Optional[dict]:
    """The whole manifest, or None when the run has not written one."""
    try:
        with open(os.path.join(run_dir, 'manifest.json'), encoding='utf-8') as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def source_score(run_dir: str) -> tuple[str, Optional[float]]:
    """The quantity the gate is defined on, read from the source's own manifest.

    `DESIGN.md` 4.3 defines validity on the **normalised** final score, which is
    `result.final_score` of the source run itself -- not on raw return, and not
    on a multiplicative fraction of the registered threshold, which at Acrobot's
    -100 would demand -60 and so be harder than solving the task.

    Four states, kept apart on purpose:

    * ``unrun``      -- no manifest. Nothing is known; the run is still to do.
    * ``unscored``   -- a manifest with no ``final_score``. The run finished
      without producing the number the gate is defined on, so the gate cannot be
      applied. That is reported, never read as a pass.
    * ``unmeasured`` -- a ``final_score`` that is present but is not a finite
      number: NaN from a degenerate evaluation, an infinity, or something that
      is not a number at all. This state exists because ``float('nan') >= 0.6``
      is ``False`` in Python, so leaving such a value to the comparison
      classified a **measurement failure** as a source-quality *rejection*: it
      consumed a RESERVE seed and wrote ``"score": NaN`` into the ledger
      `DESIGN.md` 4.3 requires the results table to be built from.
    * ``scored``     -- the score.

    Only ``scored`` reaches the gate. The other three are refusals, because a
    source with no validity verdict is exactly the thing 4.3 forbids transfer
    from, and `float()` raising on a string here would have aborted the planner
    rather than reported the manifest.
    """
    result = manifest_result(run_dir)
    if result is None:
        return 'unrun', None
    score = result.get('final_score')
    if score is None:
        return 'unscored', None
    try:
        value = float(score)
    except (TypeError, ValueError):
        return 'unmeasured', None
    if not math.isfinite(value):
        return 'unmeasured', None
    return 'scored', value


@dataclass
class SourceSlot:
    """One source requirement of the plan, and what it currently resolves to.

    A slot is keyed by (source arm, the seed the source would default to), which
    is the thing a replacement re-points. It is deliberately *not* keyed by the
    consumer: several arms share one source -- E1's two transfer sets, E2's
    permuted control -- and replacing the source once has to move all of them,
    or a control would be measured against a different source than the arm it
    controls for.
    """

    lineage: str
    arm: str
    cell: str
    env: str
    experiment: str
    default_seed: int
    seed: int
    run_dir: str
    run_digest: str
    state: str = 'unrun'
    score: Optional[float] = None
    consumers: list[str] = field(default_factory=list)

    @property
    def replaced(self) -> bool:
        return self.seed != self.default_seed

    def verdict(self, gate: float) -> str:
        """One of 'valid', 'rejected', 'unrun', 'unscored', 'unmeasured'.

        The comparison is reached only from a finite score. Everything else
        returns the state it is in, and the caller refuses rather than resolving
        it: an ungated state is neither a pass (which would be transfer from a
        source of unknown competence, `DESIGN.md` 4.3's central defect) nor a
        rejection (which would spend a RESERVE seed on a broken evaluation and
        report a measurement failure to the results table as a source-quality
        verdict).
        """
        if self.state != 'scored':
            return self.state
        if self.score is None or not math.isfinite(self.score):
            return 'unmeasured'
        return 'valid' if self.score >= gate else 'rejected'

    def gated(self, gate: float) -> bool:
        """Did the DESIGN.md 4.3 gate actually return a verdict on this slot?"""
        return self.verdict(gate) in ('valid', 'rejected')

    def describe(self) -> str:
        return (f'{self.arm} s{self.seed:02d} ({self.cell} on {self.env})'
                + (f' [replaces s{self.default_seed:02d}]' if self.replaced
                   else ''))


#: The slot states that are not a gate verdict. `DESIGN.md` 4.3 defines
#: validity on the normalised final score, so a slot in any of these has no
#: verdict at all: not valid, not rejected, and not something phase 2 may run
#: against.
UNGATED_STATES = ('unrun', 'unscored', 'unmeasured')

#: The gate value that means "no gate was applied to this selection". Not a
#: threshold anybody chose: every finite score clears it, so no slot is ever
#: rejected under it and no RESERVE seed is ever drawn. `main` resolves the gate
#: to this only for a selection made entirely of SMOKE-block pipeline
#: validation at SMOKE-block seeds, which reports nothing and trains 12
#: episodes; `DESIGN.md` 4.3 defines validity on the normalised final score of a
#: source trained to the design's budget, and 12 episodes is not that budget.
#: The states that are measurement failures, `UNGATED_STATES`, still refuse
#: under it, because those say the pipeline is broken rather than the source.
GATE_NOT_APPLIED = -math.inf


def gate_text(gate: float) -> str:
    """The gate as a printed phrase, so that no line ever prints '-inf'.

    `GATE_NOT_APPLIED` renders through ``:.3f`` as ``-inf``, which reads like a
    number somebody typed rather than the absence of a threshold.
    """
    if math.isfinite(gate):
        return f'normalised final score >= {gate:.3f}'
    return 'NOT APPLIED (SMOKE-block pipeline validation selection)'


def ungated_sources(slots: Sequence[SourceSlot], gate: float) -> list[dict]:
    """Sources phase 1 finished without producing a validity verdict for.

    Reached only after the source phase has run, where it means what the earlier
    version never noticed: a source run that completed and did not produce
    ``result.final_score``. That slot is not in phase 1's ``due`` list, because
    its manifest reports its episodes done; it is not 'rejected', so no RESERVE
    seed is drawn and no ledger row is written; and it is not 'valid', so
    nothing licenses the transfer runs that depend on it. The whole target side
    was trained against it and `main` returned 0.
    """
    rows: list[dict] = []
    for slot in slots:
        state = slot.verdict(gate)
        if state not in UNGATED_STATES:
            continue
        rows.append({
            'state': state, 'lineage': slot.lineage, 'source_arm': slot.arm,
            'cell': slot.cell, 'source_env': slot.env,
            'experiment': slot.experiment, 'seed': slot.seed,
            'default_seed': slot.default_seed, 'run_dir': slot.run_dir,
            'run_digest': slot.run_digest,
            'consumers': sorted(set(slot.consumers)),
            'describe': slot.describe(),
            'reason': {
                'unrun': 'the source run has no manifest',
                'unscored': ('the source run finished but its manifest carries '
                             'no result.final_score, which is the quantity '
                             'DESIGN.md 4.3 defines validity on'),
                'unmeasured': ('the source run reported a final_score that is '
                               'not a finite number, so the evaluation failed '
                               'rather than the source'),
            }[state]})
    return rows


def print_ungated_sources(rows: Sequence[dict], gate: float) -> None:
    """Name every source that phase 2 would otherwise be run against blind."""
    if not rows:
        return
    print('\n' + '=' * 72)
    print(f'[ERROR] {len(rows)} source slot(s) have no validity verdict '
          f'(DESIGN.md 4.3)')
    print('=' * 72)
    print(f'  Gate: {gate_text(gate)}. These slots were never gated at')
    print('  all, so they are neither valid nor rejected: no RESERVE seed is '
          'drawn for them,')
    print('  because a measurement failure is not a source-quality verdict and '
          'spending a')
    print('  replacement seed on one would put a number in the results table '
          'that does not')
    print('  mean what the column says.')
    for row in rows:
        print(f'\n  {row["state"].upper():10s} {row["describe"]}')
        print(f'    reason:    {row["reason"]}')
        print(f'    directory: {row["run_dir"]}')
        print(f'    {len(row["consumers"])} dependent run(s) would be trained '
              f'against it')
    print('\n  Phase 2 is not started. Transfer from a source whose competence '
          'was never')
    print('  established is the published study\'s central defect and the '
          'reason DESIGN.md 4.3')
    print('  exists. Find out why the source produced no score (the worker log '
          'names the run),')
    print('  fix it, and re-run; or pass --allow-invalid-sources to proceed '
          'deliberately, which')
    print('  is stamped into the invocation record.')


def source_slots(records: Sequence[dict]) -> list[SourceSlot]:
    """Every source the plan depends on, in a fixed order.

    The order is (arm, default seed) and it is fixed because the allocation
    walks it: a draw that depended on dictionary or filesystem ordering would
    give a different assignment on a re-run, and `DESIGN.md` 4.3's "drawn in
    order" would then be untrue.
    """
    by_dir = {r['run_dir']: r for r in records}
    slots: dict[tuple[str, int], SourceSlot] = {}
    for rec in records:
        dep = rec.get('depends_on')
        src = by_dir.get(dep) if dep else None
        if src is None:
            continue
        key = (str(rec.get('source_arm')), int(rec.get('source_default_seed')))
        slot = slots.get(key)
        if slot is None:
            state, score = source_score(src['run_dir'])
            slot = SourceSlot(
                lineage=str(rec.get('source_lineage')),
                arm=key[0], cell=src['cell'], env=src['env'],
                experiment=src['experiment'], default_seed=key[1],
                seed=int(rec.get('source_seed')), run_dir=src['run_dir'],
                run_digest=src['run_digest'], state=state, score=score)
            slots[key] = slot
        slot.consumers.append(rec['run_dir'])
    return [slots[k] for k in sorted(slots, key=lambda k: (k[0], k[1]))]


def read_replacements(path: str) -> list[dict]:
    """The rejection ledger, skipping any line a concurrent append tore."""
    out: list[dict] = []
    if not os.path.exists(path):
        return out
    with open(path, encoding='utf-8', errors='replace') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _replacement_key(row: dict) -> tuple:
    return (str(row.get('lineage')), str(row.get('source_arm')),
            row.get('rejected_seed'), row.get('replacement_seed'))


def append_replacements(path: str, rows: Sequence[dict]) -> int:
    """Append rejections not already recorded; return how many were new.

    De-duplicated on (lineage, arm, rejected seed, replacement seed) so that
    re-running a completed sweep adds nothing. Idempotence has to hold in the
    *record* as well as in the assignment, or the results table would report one
    rejection twice and the count `DESIGN.md` 4.3 asks for would be wrong.
    """
    known = {_replacement_key(r) for r in read_replacements(path)}
    fresh = [r for r in rows if _replacement_key(r) not in known]
    written = sum(1 for row in fresh if append_status(path, row))
    if written != len(fresh):
        print(f'[ERROR] {len(fresh) - written} of {len(fresh)} '
              f'source-validity rejection(s) could not be written to {path}. '
              f'DESIGN.md 4.3 requires the rejected seeds in the results '
              f'table, and a rejection that exists only in this scrollback is '
              f'not reportable.', file=sys.stderr, flush=True)
    return written


def resolve_source_assignment(
        exp_ids: Sequence[str], seeds: Optional[tuple[int, ...]], out_root: str,
        overrides: dict, allow_factor_overrides: bool, gate: float,
        ledger: Sequence[dict], gate_overridden: bool = False,
        sweep_id: str = '') -> dict:
    """Work out which source seed each slot uses, and what was rejected.

    The assignment is **derived, not drawn**. Two inputs only: the durable
    ledger, replayed first so an assignment already made survives an
    interruption even when the rejected run has since been moved off disk; and
    the scores already on disk, walked in the fixed slot order against the fixed
    RESERVE ordering. Same inputs, same assignment -- which is what makes a
    resumed invocation a lookup rather than a fresh draw.

    Returns the assignment map, the rebuilt plan, the slots, the rejections that
    are new to the ledger, and the lineages whose reserve ran out. Exhaustion is
    returned rather than raised so the caller can report every exhausted lineage
    at once: finding out about them one sweep at a time is how a cell ends up at
    n=7.
    """
    reserve = tuple(registry.RESERVE_ORDER)

    # (arm, rejected seed) -> replacement seed, from the ledger. Chains, because
    # a replacement can itself be rejected; `follow` walks to the end of one.
    chain: dict[tuple[str, int], int] = {}
    handed_out: dict[str, set[int]] = {}
    for row in ledger:
        rep = row.get('replacement_seed')
        if rep is None:
            # A rejection recorded with nothing left to replace it. It belongs
            # in the results table but not in the chain, and re-reading it must
            # not consume a reserve seed that was never handed out.
            continue
        chain[(str(row.get('source_arm')),
               int(row.get('rejected_seed')))] = int(rep)
        handed_out.setdefault(str(row.get('lineage')), set()).add(int(rep))

    def follow(arm: str, seed: int) -> int:
        seen = {seed}
        cur = seed
        while (arm, cur) in chain:
            cur = chain[(arm, cur)]
            if cur in seen:
                raise RuntimeError(
                    f'the replacement ledger contains a cycle for {arm} at seed '
                    f'{seed}; '
                    f'{os.path.join(out_root, JOBS_SUBDIR, REPLACEMENTS_FILE)} '
                    f'has been edited or merged by hand')
            seen.add(cur)
        return cur

    assignment: dict[tuple[str, int], int] = {}
    new_rejections: list[dict] = []
    records: list[dict] = []
    membership: dict[str, list[str]] = {}
    slots: list[SourceSlot] = []
    exhausted: list[dict] = []

    # Bounded: each pass either consumes one reserve seed for one lineage or
    # settles. The bound is stated so that a bug becomes a refusal rather than a
    # process that never returns.
    for _ in range(len(reserve) * 8 + 4):
        records, membership = build_plan(exp_ids, seeds, out_root, overrides,
                                         allow_factor_overrides, assignment)
        slots = source_slots(records)
        exhausted = []
        changed = False

        # 1. Honour what the ledger already decided.
        for slot in slots:
            target = follow(slot.arm, slot.default_seed)
            if target != slot.seed:
                assignment[(slot.arm, slot.default_seed)] = target
                changed = True
        if changed:
            continue

        # 2. Gate what is on disk, and draw for the first rejection found. One
        #    at a time, because an allocation moves run directories and every
        #    later slot has to be re-read against the new plan.
        for slot in slots:
            if slot.verdict(gate) != 'rejected':
                continue
            used = set(handed_out.get(slot.lineage, ())) | {
                other.seed for other in slots if other.lineage == slot.lineage}
            nxt = next((r for r in reserve if r not in used), None)
            row = {
                'ts': round(time.time(), 3),
                't': time.strftime('%Y-%m-%dT%H:%M:%S'),
                'sweep': sweep_id,
                'rule': 'DESIGN.md 4.3 -- normalised final score >= gate',
                'gate': gate,
                'gate_is_design_value': not gate_overridden,
                'lineage': slot.lineage,
                'cell': slot.cell,
                'experiment': slot.experiment,
                'source_arm': slot.arm,
                'source_env': slot.env,
                'rejected_seed': slot.seed,
                'rejected_seed_block': seed_block(slot.seed),
                'rejected_run_dir': slot.run_dir,
                'rejected_run_digest': slot.run_digest,
                # Only a finite score reaches this row: `SourceSlot.verdict`
                # never returns 'rejected' for a non-finite one, and
                # `append_status` writes any that slipped through as null
                # rather than as the bare NaN token json.dumps emits, which no
                # strict JSON reader accepts. `score_state` says which state
                # produced the number so the table never has to infer it.
                'score': slot.score,
                'score_state': slot.state,
                'default_seed': slot.default_seed,
                'replacement_seed': nxt,
                'consumers': sorted(set(slot.consumers)),
            }
            if nxt is None:
                row['error'] = (
                    f'RESERVE exhausted for {slot.arm} ({slot.cell} on '
                    f'{slot.env}) at default seed {slot.default_seed}: all '
                    f'{len(reserve)} seeds {reserve[0]}-{reserve[-1]} have '
                    f'been drawn for this lineage, the last of them scoring '
                    f'{slot.score:.3f} at s{slot.seed:02d}, and this slot '
                    f'still has no valid source')
                exhausted.append(row)
                continue
            chain[(slot.arm, slot.seed)] = nxt
            assignment[(slot.arm, slot.default_seed)] = nxt
            handed_out.setdefault(slot.lineage, set()).add(nxt)
            new_rejections.append(row)
            changed = True
            break

        if not changed:
            break
    else:                                            # pragma: no cover - guard
        raise RuntimeError('source assignment did not settle; that is a bug in '
                           'resolve_source_assignment, not a data condition')

    return {'assignment': assignment, 'records': records,
            'membership': membership, 'slots': slots,
            'new_rejections': new_rejections, 'exhausted': exhausted}


def lineage_conflicts(records: Sequence[dict]) -> list[dict]:
    """Complete runs whose stored source is not the source now assigned.

    This check exists because run identity does not cover the source. A
    re-pointed transfer run keeps the run directory it would have had with the
    rejected source (`source_checkpoint` is bookkeeping; see the module
    docstring), so a directory left by an earlier sweep can hold a run trained
    against a source the gate has since rejected. Treating it as complete would
    serve invalid-source data under a valid-source label, which is precisely the
    pooling error `DESIGN.md` 4.3 forbids; resuming it would be worse, because
    `train.py` checks the trajectory digest and that does not cover the source
    either, so the resume would be accepted.

    Only conditions that actually read the source weights are checked: the
    untrained control builds its own random source of matched shape and records
    no ``source_result`` to compare against.
    """
    out: list[dict] = []
    for rec in records:
        if not rec.get('needs_source_weights') or not rec.get('depends_on'):
            continue
        if not is_complete(rec):
            continue
        manifest = manifest_of(rec['run_dir']) or {}
        stored = (((manifest.get('source') or {}).get('source_result') or {})
                  .get('run_digest'))
        assigned = rec.get('source_run_digest')
        if stored and assigned and str(stored) != str(assigned):
            out.append({'run_dir': rec['run_dir'], 'job_id': rec['job_id'],
                        'arm': rec['arm'], 'seed': rec['seed'],
                        'experiment': rec['experiment'],
                        'stored_source_digest': str(stored),
                        'assigned_source_digest': str(assigned),
                        'assigned_source_dir': rec['depends_on'],
                        'assigned_source_seed': rec.get('source_seed'),
                        'default_source_seed': rec.get('source_default_seed')})
    return out


def print_lineage_conflicts(conflicts: Sequence[dict]) -> None:
    """Say exactly which directories hold data from a rejected source.

    Printed rather than repaired, and the repair is not offered: this runner
    does not delete data, and a directory whose contents were trained against a
    source the gate has since rejected is evidence of something, not garbage.
    Moving it aside is the operator's decision, and it is a decision that has to
    be recorded in the paper -- `DESIGN.md` 4.3 requires the rejected sources to
    appear in the results table, and a run that was built on one is part of that
    story.
    """
    if not conflicts:
        return
    print('\n' + '=' * 72)
    print(f'[ERROR] {len(conflicts)} complete run(s) were trained against a '
          f'different source')
    print('=' * 72)
    print('  Run identity does not cover the source checkpoint '
          '(src/dqn/config.py puts')
    print('  source_checkpoint in BOOKKEEPING_FIELDS, so that moving the run '
          'tree does not')
    print('  change any run digest). A transfer run re-pointed at a '
          'replacement source')
    print('  therefore keeps the directory it had with the rejected one. These '
          'directories')
    print('  already hold a finished run built from the source the gate now '
          'rejects:')
    for row in conflicts:
        print(f'\n  {row["job_id"]}  {row["experiment"]}/{row["arm"]} '
              f'seed {row["seed"]}')
        print(f'    directory:      {row["run_dir"]}')
        print(f'    stored source:  {row["stored_source_digest"][:12]}')
        print(f'    assigned source: {row["assigned_source_digest"][:12]} '
              f'(seed {row["assigned_source_seed"]}, default '
              f'{row["default_source_seed"]})')
        print(f'                    {row["assigned_source_dir"]}')
    print('\n  Neither skipped nor resumed. Skipping would serve '
          'invalid-source data under a')
    print('  valid-source label, which is the pooling error DESIGN.md 4.3 '
          'exists to prevent;')
    print('  resuming would be accepted by train.py, because the trajectory '
          'digest does not')
    print('  cover the source either. Move or delete the directories above and '
          're-run --')
    print('  this runner will not delete data. The proper fix is for the run '
          'digest to cover')
    print('  the source lineage, and that belongs in src/dqn/config.py.')


def print_source_phase(slots: Sequence[SourceSlot], records: Sequence[dict],
                       gate: float, ledger: Sequence[dict],
                       new_rejections: Sequence[dict],
                       exhausted: Sequence[dict]) -> None:
    """The two-phase structure, before anything is written or launched."""
    reserve = tuple(registry.RESERVE_ORDER)
    sources = [r for r in records if r['is_source']]
    consumers = [r for r in records if r.get('depends_on')]
    repointed = [r for r in consumers if r.get('source_replaced')]
    lineages = {sl.lineage for sl in slots}
    tally: dict[str, int] = {}
    for sl in slots:
        v = sl.verdict(gate)
        tally[v] = tally.get(v, 0) + 1
    # Per lineage, not pooled. The reserve pool is applied *per lineage* (see
    # the `used` set in `resolve_source_assignment`), so counting the union of
    # replacement seeds across lineages answered a question nobody asked: two
    # lineages that both drew seed 400 reported "1 drawn, 19 unused" after two
    # allocations, and 80 allocations across 4 lineages reported "20 drawn, 0
    # unused". This line is the operator's only running view of how close
    # RESERVE is to exhausting, and it understated consumption by up to a factor
    # of the lineage count.
    drawn: dict[str, set[int]] = {}
    for row in list(ledger) + list(new_rejections):
        seed = row.get('replacement_seed')
        if seed is None:
            continue
        drawn.setdefault(str(row.get('lineage')), set()).add(int(seed))
    allocations = sum(len(seeds) for seeds in drawn.values())
    deepest = max(drawn.items(), key=lambda kv: (len(kv[1]), kv[0])) \
        if drawn else None

    print('\ntwo-phase structure (DESIGN.md 4.3, source validity):')
    print(f'  gate:      {gate_text(gate)}'
          + (' on the source environment' if math.isfinite(gate) else ''))
    print(f'  phase 1    {len(sources)} source run(s) over {len(lineages)} '
          f'lineage(s); {sum(1 for r in sources if is_complete(r))} complete, '
          f'{sum(1 for r in sources if not is_complete(r))} to run')
    print(f'             {len(slots)} slot(s): '
          + ('  '.join(f'{k}={v}' for k, v in sorted(tally.items())) or 'none'))
    for sl in slots:
        v = sl.verdict(gate)
        if v == 'valid':
            continue
        detail = {
            'rejected': (f'score {sl.score:.3f} < {gate:.3f}'
                         if sl.score is not None else 'below the gate'),
            'unrun': 'no manifest yet',
            'unscored': 'manifest carries no final_score: NOT a pass',
            'unmeasured': ('final_score is not a finite number, so the '
                           'evaluation failed rather than the source'),
        }.get(v, v)
        print(f'               {v.upper():8s} {sl.describe()}  {detail}')
    for row in new_rejections:
        print(f'               -> RESERVE seed {row["replacement_seed"]} for '
              f'{row["source_arm"]} (was s{row["rejected_seed"]:02d}, score '
              f'{row["score"]:.3f}); {len(row["consumers"])} dependent run(s) '
              f're-pointed')
    print(f'             RESERVE: {len(reserve)} seed(s) {reserve[0]}-'
          f'{reserve[-1]}, available to EACH lineage; {allocations} '
          f'allocation(s) across {len(drawn)} lineage(s)')
    if deepest is not None:
        # Named by the arm rather than the lineage key: the key is a digest, and
        # an operator watching the reserve drain needs to know which cell it is.
        lineage_arm = {sl.lineage: sl.arm for sl in slots}
        print(f'               deepest: {lineage_arm.get(deepest[0], deepest[0])}'
              f' has drawn {len(deepest[1])} of {len(reserve)} '
              f'({len(reserve) - len(deepest[1])} left before that lineage '
              f'exhausts)')
    ungated = [sl for sl in slots if not sl.gated(gate)
               and sl.verdict(gate) != 'unrun']
    if ungated:
        print(f'             [REFUSAL] {len(ungated)} slot(s) have no validity '
              f'verdict at all. DESIGN.md 4.3 is')
        print('             defined on the normalised final score; a slot '
              'without one is neither')
        print('             valid nor rejected, so no RESERVE seed is drawn '
              'and phase 2 does not start.')
    print(f'  phase 2    {len(records) - len(sources)} run(s) outside phase 1; '
          f'{len(consumers)} depend on a phase-1 source, {len(repointed)} of '
          f'them on a replacement')
    print('  phase 1 repeats while a newly trained source fails the gate. '
          'Exhausting')
    print('  RESERVE is an error and stops the sweep; it is never a quietly '
          'short cell.')
    for row in exhausted:
        print(f'  [ERROR] {row["error"]}')


# ---------------------------------------------------------------------------
# The single-run entry point a worker invokes
# ---------------------------------------------------------------------------
def run_one(config_dict: dict, argv: list[str] | None = None) -> dict:
    """Train one run from a job record's config, and return its manifest.

    ``train`` is imported here rather than at module scope: the parent process
    plans, launches and summarises but never trains, and keeping the training
    module out of its import graph keeps the planning paths (``--dry-run``, the
    summary) independent of the trainer. It does not avoid loading TensorFlow --
    ``Config`` already pulls it in through ``networks`` -- so this is a coupling
    argument, not a startup-cost one.
    """
    from src.dqn.train import train                        # noqa: PLC0415
    cfg = Config(**config_dict)
    return train(cfg, argv)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def worker_main(args: argparse.Namespace) -> int:
    """Claim-and-run until nothing is left that this worker can do.

    The worker walks the whole job list, rotated by its index so that N workers
    start at N different points and contend less. Rotation is safe because the
    list is in dependency order only as a hint: readiness is checked against the
    filesystem, never against position.

    "Nothing left that this worker can do" has two endings and they are not the
    same fact. **Blocked** is `DESIGN.md` §8.4's terminal verdict and is
    reserved for jobs waiting cannot help: a source that failed in this sweep,
    or a dependency nobody will ever build. A job held by an owner that is
    demonstrably alive is the other ending: the worker stops waiting, says so,
    and exits 0, because every worker reads the whole job list and whoever holds
    the claim is the worker that will finish it.
    """
    records = read_jobs(args.jobs_file)
    if not records:
        print('no jobs')
        return 0
    status_path = os.path.join(os.path.dirname(args.jobs_file), STATUS_FILE)
    worker = f'w{args.worker_id:02d}'
    # The identity every claim this worker writes carries, and the identity
    # `release_claim` and `fail_claim` check before touching a claim file. In P0
    # neither call checked, so a worker's successful completion deleted another
    # worker's claim.
    owner = claim_owner(args.sweep_id, worker)
    # Two unrelated powers, deliberately not one flag. `--force` re-enters a
    # directory whose manifest is already complete; `--force-claim` takes a
    # directory away from a claim another sweep holds, including one whose owner
    # is demonstrably alive, which is the P0 corruption made deliberate. See
    # `build_parser`.
    max_wait = args.max_wait_seconds or (args.stale_seconds + 300)

    def log(state: str, rec: dict, **extra) -> None:
        append_status(status_path, {
            'ts': round(time.time(), 3),
            't': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'sweep': args.sweep_id, 'worker': worker, 'state': state,
            'job_id': rec['job_id'], 'run_dir': rec['run_dir'],
            'experiment': rec['experiment'], 'arm': rec['arm'],
            'seed': rec['seed'], **extra})

    offset = args.worker_id % len(records)
    remaining = records[offset:] + records[:offset]
    deferral_reason: dict[str, str] = {}
    failures: list[str] = []
    blocked: list[str] = []
    # Jobs this worker stopped waiting for because somebody else is visibly
    # training them. Counted apart from `blocked` on purpose: one is a terminal
    # verdict and the other is the ordinary tail of a parallel stage.
    waiting: list[str] = []
    ran = skipped = 0
    idle_seconds = 0.0

    print(f'{worker}: {len(records)} jobs in the manifest, starting at '
          f'offset {offset}', flush=True)

    while remaining:
        deferred: list[dict] = []
        progressed = False
        pending = {r['run_dir'] for r in remaining if not is_complete(r)}
        for rec in remaining:
            run_dir = rec['run_dir']
            if is_complete(rec) and not args.force:
                # Not logged here: every worker walks the whole list, so N
                # workers would write N identical 'skipped' records per
                # already-complete run. The parent writes one each, once, before
                # launching.
                skipped += 1
                progressed = True
                continue

            if claim_failed_here(run_dir, args.sweep_id):
                # Another worker in this sweep already failed it and recorded
                # why. Say nothing further: a second record would displace the
                # one that carries the error and the log path.
                progressed = True
                continue

            state, why = dependency_state(rec, pending, args.sweep_id)
            if state == 'blocked':
                log('blocked', rec, reason=why)
                blocked.append(rec['job_id'])
                print(f'{worker}: BLOCKED {rec["job_id"]} {rec["arm"]} -- {why}',
                      flush=True)
                progressed = True          # the job list shrank
                continue
            if state == 'wait':
                if deferral_reason.get(run_dir) != why:
                    log('deferred', rec, reason=why)
                    deferral_reason[run_dir] = why
                deferred.append(rec)
                continue

            acquired, note = claim_run(run_dir, args.sweep_id, worker,
                                       args.stale_seconds, args.force_claim)
            if not acquired:
                if deferral_reason.get(run_dir) != note:
                    log('deferred', rec, reason=note)
                    deferral_reason[run_dir] = note
                deferred.append(rec)
                continue
            # Whether this directory was taken over from somebody else. It
            # travels into `fail_claim`, where it is the difference between a
            # failed run and a failed reclaim, and `dependency_state` refuses to
            # block a consumer on the second.
            contended = note != 'claimed'
            if contended:
                log('reclaimed', rec, reason=note)
                print(f'{worker}: reclaimed {rec["job_id"]} -- {note}',
                      flush=True)
            log('claimed', rec, note=note)

            print(f'\n{worker}: === {rec["job_id"]} {rec["experiment"]}/'
                  f'{rec["arm"]} seed {rec["seed"]} ({rec["seed_block"]}) '
                  f'on {rec["env"]} ===', flush=True)
            t0 = time.time()
            # The heartbeat runs for exactly as long as the training call. It is
            # what makes the claim's age mean "how long since the owner was last
            # alive" rather than "how long ago this run started", which is the
            # reading that took a 14.4 h run away from its owner at 14.2 h in P0.
            try:
                with ClaimHeartbeat(run_dir, owner) as beat:
                    manifest = run_one(rec['config'])
            except BaseException as exc:                   # noqa: BLE001
                traceback.print_exc()
                marked, why_mark = fail_claim(
                    run_dir, f'{type(exc).__name__}: {exc}', owner,
                    contended=contended)
                log('failed', rec, error=f'{type(exc).__name__}: {exc}',
                    wall_time_s=round(time.time() - t0, 1),
                    contended=contended, claim_marked=marked,
                    claim_note=why_mark)
                failures.append(rec['job_id'])
                print(f'{worker}: FAILED {rec["job_id"]}: '
                      f'{type(exc).__name__}: {exc}'
                      + ('  (after a contested claim: the failure may be the '
                         'collision, not the run)' if contended else ''),
                      flush=True)
                if not marked:
                    print(f'{worker}: claim not marked failed -- {why_mark}',
                          flush=True)
                if isinstance(exc, KeyboardInterrupt):
                    return 130
            else:
                if beat.lost_reason:
                    # Should be unreachable without --force now that a live
                    # claim is not reclaimable, but a run that finished without
                    # owning its directory is not an ordinary success and is not
                    # reported as one.
                    log('claim_lost', rec, reason=beat.lost_reason,
                        wall_time_s=round(time.time() - t0, 1))
                    print(f'{worker}: WARNING {rec["job_id"]} finished but lost '
                          f'its claim while training -- {beat.lost_reason}',
                          flush=True)
                released, why_release = release_claim(run_dir, owner)
                result = manifest.get('result') or {}
                log('done', rec, wall_time_s=round(time.time() - t0, 1),
                    episodes_completed=result.get('episodes_completed'),
                    final_score=result.get('final_score'),
                    auc_score=result.get('auc_score'),
                    heartbeats=beat.beats, claim_released=released,
                    claim_note=why_release)
                ran += 1
                print(f'{worker}: done {rec["job_id"]} in '
                      f'{time.time() - t0:.0f}s  final_score='
                      f'{result.get("final_score")}', flush=True)
                if not released:
                    print(f'{worker}: claim not released -- {why_release}',
                          flush=True)
            progressed = True

        remaining = deferred
        if not remaining:
            break
        if progressed:
            idle_seconds = 0.0
            continue
        if idle_seconds >= max_wait:
            # **blocked** is terminal under `DESIGN.md` §8.4: the sweep says
            # this run will never happen and exits non-zero. Running out of
            # patience is not that. When the directory a job needs is held by an
            # owner this host can see running, the job is *waiting*, and on a
            # six-worker campaign whose runs take hours that is the predictable
            # end of every stage: the last few claims stay with the workers
            # still training, and everybody else runs out of things they may
            # start. Recording those as blocked made five of six workers exit 1
            # with a terminal verdict on runs that were training normally, which
            # is the runner-manufactured false verdict the contended-failure
            # rule was written to remove. The liveness evidence is the same
            # evidence `_reclaim_reason` requires, so the two agree by
            # construction: what may not be reclaimed may not be declared dead.
            stopped: list[dict] = []
            for rec in remaining:
                why = (f'still waiting after {idle_seconds:.0f}s: '
                       f'{deferral_reason.get(rec["run_dir"], "unknown")}')
                where, holder = 'this run', live_claim_holder(rec['run_dir'])
                dep = rec.get('depends_on')
                if holder is None and dep:
                    holder = live_claim_holder(dep)
                    where = f'its source run {dep}'
                if holder is not None:
                    log('waiting', rec, reason=f'{why}; {where} is held by a '
                                               f'live owner: {holder}')
                    waiting.append(rec['job_id'])
                    stopped.append(rec)
                    print(f'{worker}: waiting {rec["job_id"]} -- {where} is '
                          f'held by {holder}; leaving it to that worker',
                          flush=True)
                    continue
                log('blocked', rec, reason=why)
                blocked.append(rec['job_id'])
                print(f'{worker}: BLOCKED {rec["job_id"]} -- {why}', flush=True)
            if stopped:
                print(f'{worker}: {len(stopped)} job(s) left in another '
                      f'worker\'s hands after {idle_seconds:.0f}s. Not '
                      f'blocked: their owners are alive, so this is contention '
                      f'and not a verdict. Whoever holds them finishes them, '
                      f'or they are reported unfinished by the parent, which '
                      f'reads the manifests on disk.', flush=True)
            break
        time.sleep(args.poll_seconds)
        idle_seconds += args.poll_seconds

    print(f'\n{worker}: ran {ran}, skipped {skipped}, failed '
          f'{len(failures)}, blocked {len(blocked)}, left with a live owner '
          f'{len(waiting)}', flush=True)
    # A job left with a live owner is not this worker's failure and does not
    # make it one. The parent decides the sweep from the manifests on disk, so
    # an unfinished run is still reported unfinished and still exits non-zero
    # there; what it is not is a terminal BLOCKED record against a run that is
    # training normally.
    return 1 if (failures or blocked) else 0


# ---------------------------------------------------------------------------
# Parent
# ---------------------------------------------------------------------------
def spawn_workers(args: argparse.Namespace, jobs_path: str,
                  n_jobs: int, tag: str = ''
                  ) -> list[tuple[subprocess.Popen, str, str, IO[str]]]:
    """Launch worker interpreters with pinned threads and a fixed hash seed.

    Left unpinned, every TensorFlow process claims all cores and they thrash,
    which costs more than the parallelism gains. ``PYTHONHASHSEED`` and
    ``TF_ENABLE_ONEDNN_OPTS`` are set here because they can only take effect
    before an interpreter starts, and ``provenance.determinism`` records them in
    each manifest -- so what a run's numerics depended on is a recorded fact
    rather than an assumption (`DESIGN.md` §8.3).
    """
    threads = max(1, (os.cpu_count() or 4) // max(1, n_jobs))
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS=str(threads),
               TF_NUM_INTRAOP_THREADS=str(threads),
               TF_NUM_INTEROP_THREADS='1',
               TF_CPP_MIN_LOG_LEVEL='3',
               PYTHONHASHSEED='0',
               TF_ENABLE_ONEDNN_OPTS='0')
    log_dir = os.path.join(args.out_root, LOGS_SUBDIR)
    os.makedirs(log_dir, exist_ok=True)

    # The log handle travels with the process so `run_stage` can close it. It
    # used to be opened here and dropped, leaking one descriptor per worker per
    # phase for the life of the parent: every phase-1 stage plus phase 2.
    procs = []
    for i in range(n_jobs):
        cmd = [sys.executable, os.path.abspath(__file__),
               '--worker', '--jobs-file', jobs_path, '--worker-id', str(i),
               '--sweep-id', args.sweep_id,
               '--stale-seconds', str(args.stale_seconds),
               '--poll-seconds', str(args.poll_seconds),
               '--max-wait-seconds', str(args.max_wait_seconds)]
        if args.force:
            cmd.append('--force')
        if args.force_claim:
            cmd.append('--force-claim')
        # The tag separates the phases' logs. Without it the source phase's
        # workers and the target phase's would overwrite each other's log and
        # the failure that stopped a source would be unreadable by the time
        # anyone looked.
        suffix = f'-{tag}' if tag else ''
        path = os.path.join(log_dir, f'w{i:02d}-{args.sweep_id}{suffix}.log')
        fh = open(path, 'w', encoding='utf-8')
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                env=env, cwd=_ROOT)
        procs.append((proc, path, f'w{i:02d}', fh))
        print(f'  worker w{i:02d}  pid {proc.pid}  {threads} threads  -> {path}')
    return procs


def wait_for_workers(procs, status_path: str, sweep_id: str,
                     progress_seconds: int) -> dict[str, int]:
    """Block until every worker exits, printing a periodic tally."""
    codes: dict[str, int] = {}
    last = time.time()
    while len(codes) < len(procs):
        for proc, _path, name, _fh in procs:
            if name in codes:
                continue
            code = proc.poll()
            if code is not None:
                codes[name] = code
        if len(codes) == len(procs):
            break
        time.sleep(1.0)
        if progress_seconds and time.time() - last >= progress_seconds:
            last = time.time()
            tally: dict[str, int] = {}
            for rec in read_status(status_path, sweep_id):
                tally[rec.get('state', '?')] = tally.get(rec.get('state', '?'),
                                                         0) + 1
            print('  ... ' + '  '.join(f'{k}={v}' for k, v in
                                       sorted(tally.items())), flush=True)
    return codes


def run_stage(args: argparse.Namespace, records: Sequence[dict], tag: str,
              status_path: str) -> tuple[dict[str, int], dict[str, str]]:
    """Write one phase's job file, launch workers on it, and wait.

    Each phase gets its own job file rather than reusing ``jobs.jsonl``, because
    a worker's whole contract is that it never invents a job: it runs what its
    file lists and nothing else. Handing the source phase a file containing only
    the source runs is therefore how the phase boundary is enforced, rather than
    by asking workers to be selective.
    """
    jobs_dir = os.path.join(args.out_root, JOBS_SUBDIR)
    os.makedirs(jobs_dir, exist_ok=True)
    path = os.path.join(jobs_dir, f'jobs-{args.sweep_id}-{tag}.jsonl')
    write_jobs(path, records)
    n_jobs = max(1, min(args.jobs, len(records)))
    print(f'\nlaunching {n_jobs} worker(s) for {len(records)} job(s) '
          f'[{tag}] -> {path}')
    procs = spawn_workers(args, path, n_jobs, tag)
    log_paths = {name: log for _proc, log, name, _fh in procs}
    started = time.time()
    try:
        codes = wait_for_workers(procs, status_path, args.sweep_id,
                                 args.progress_seconds)
    finally:
        for _proc, _log, _name, fh in procs:
            try:
                fh.close()
            except OSError:
                pass
    for _proc, log, name, _fh in procs:
        print(f'  {name}: exit {codes.get(name)}  {log}')
    print(f'  [{tag}] workers finished in {(time.time() - started) / 60:.1f} min')
    return codes, log_paths


def summarise(records: Sequence[dict], status_path: str, sweep_id: str,
              log_paths: dict[str, str]) -> int:
    """Print what happened to every job, and return the exit code.

    Disk is the authority: a job is complete if its manifest says so, whatever
    the status log claims. The status log supplies the *reasons* -- which worker,
    which error, which log to read. A pending job with no terminal record is
    reported as unfinished rather than assumed fine, because "the runner exited
    0 and an arm is missing runs" is the failure this file exists to prevent.

    "Disk is the authority" now includes the run's own integrity verdict. A
    manifest reporting its full episode budget alongside ``contiguous: false``
    used to be counted under "complete now"; it is its own category here, and it
    makes the exit code non-zero.

    It also reports what happened to the *claims*. ``reclaimed``, ``claim_lost``
    and a **contended** failure are the three records that say a run directory
    changed hands, and none of them reached this summary: a failure caused by a
    collision printed exactly as a failure of the run, and a directory taken
    over mid-flight and then finished printed as an ordinary success. That is
    the distinction the contended-failure rule exists to make, so an operator
    who can only see the summary could not make it. None of the three changes
    the exit code on its own: the manifests decide that, and a directory two
    trainers really did write is caught by the metrics integrity check as an
    unsound manifest, which is non-zero already.
    """
    status = read_status(status_path, sweep_id)
    events: dict[str, list[dict]] = {}
    for rec in status:
        if rec.get('run_dir'):
            events.setdefault(rec['run_dir'], []).append(rec)

    def latest(run_dir: str, state: str) -> Optional[dict]:
        hits = [e for e in events.get(run_dir, []) if e.get('state') == state]
        return hits[-1] if hits else None

    done, skipped, elsewhere, failed, blocked, unfinished = [], [], [], [], [], []
    unsound: list[tuple[dict, str]] = []
    for rec in records:
        # Severity, not recency. Several workers may report on one run, and the
        # last record is not the most informative one: a worker that merely
        # found the directory held must not overwrite the worker that recorded
        # the exception and the log to read it in.
        run_dir = rec['run_dir']
        state, why = completion_state(rec)
        if state == 'unsound':
            # Named rather than folded into 'not finished'. A run that reached
            # its budget and then failed its own metrics integrity check is a
            # different fact from a run that never started, and
            # 'not finished (last state: done)' is how that fact used to read.
            unsound.append((rec, why))
            continue
        if state == 'complete':
            if latest(run_dir, 'done'):
                done.append((rec, 'done'))
            elif latest(run_dir, 'skipped'):
                skipped.append((rec, 'skipped'))
            else:
                # Complete, but neither trained by this sweep nor complete when
                # it was planned: another sweep or another worker on the same
                # tree finished it while this one waited. Worth naming rather
                # than folding into 'already complete', because it is the only
                # visible sign that two sweeps overlapped.
                elsewhere.append((rec, 'completed elsewhere'))
        elif latest(run_dir, 'failed'):
            failed.append((rec, latest(run_dir, 'failed')))
        elif latest(run_dir, 'blocked'):
            blocked.append((rec, latest(run_dir, 'blocked')))
        else:
            states = [e.get('state') for e in events.get(run_dir, [])]
            unfinished.append((rec, states[-1] if states else None))

    total = len(records)
    print('\n' + '=' * 72)
    print(f'sweep {sweep_id}: {total} jobs')
    print(f'  complete now       {len(done) + len(skipped) + len(elsewhere):4d}'
          f'   ({len(done)} trained here, {len(skipped)} already complete'
          + (f', {len(elsewhere)} completed elsewhere while this sweep ran'
             if elsewhere else '') + ')')
    print(f'  failed             {len(failed):4d}')
    print(f'  blocked            {len(blocked):4d}')
    print(f'  unsound manifest   {len(unsound):4d}')
    print(f'  not finished       {len(unfinished):4d}')

    # The claim history, from the whole status log rather than per job: a
    # directory can be reclaimed by one worker and finished by another, so these
    # are facts about the sweep, not about a single record's latest state.
    reclaimed = [ev for ev in status if ev.get('state') == 'reclaimed']
    claim_lost = [ev for ev in status if ev.get('state') == 'claim_lost']
    contended = [ev for _rec, ev in failed if ev.get('contended')]
    waited = [ev for ev in status if ev.get('state') == 'waiting']
    if reclaimed or claim_lost or contended or waited:
        print(f'  claim events       {len(reclaimed)} reclaimed, '
              f'{len(claim_lost)} lost mid-run, {len(contended)} contended '
              f'failure(s), {len(waited)} left with a live owner')

    if failed:
        print('\nFailures -- read the worker log, then fix the cause; the run '
              'directory keeps a .claim marked failed so nothing re-enters it '
              'silently:')
        for rec, ev in failed:
            print(f'  {rec["job_id"]}  {rec["experiment"]}/{rec["arm"]} '
                  f'seed {rec["seed"]}')
            print(f'    error: {ev.get("error")}')
            print(f'    log:   {log_paths.get(ev.get("worker"), "?")}')
            print(f'    run:   {rec["run_dir"]}')
            if ev.get('contended'):
                # Printed here rather than counted only, because this is the one
                # place an operator reads a failure. Without it a collision and
                # a real failure were character for character the same block.
                print('    CONTENDED: this worker took the directory over from '
                      'another before it')
                print('    failed, so the failure may be the collision rather '
                      'than the run. Its')
                print('    dependants were deferred, not blocked. A reclaim '
                      'over a live owner needs')
                print('    --force-claim, so check whether that was passed '
                      'before reading the error.')
    if blocked:
        print('\nBlocked -- reported, never skipped silently (DESIGN.md §8.4):')
        for rec, ev in blocked:
            print(f'  {rec["job_id"]}  {rec["experiment"]}/{rec["arm"]} '
                  f'seed {rec["seed"]}: {ev.get("reason")}')
    if reclaimed or claim_lost:
        print('\nClaim events -- a run directory that changed hands. Two '
              'trainers in one directory is')
        print('the P0 corruption (DESIGN.md 8.2(1)); the metrics integrity '
              'check is what catches it,')
        print('and these lines say where to look:')
        for ev in reclaimed:
            print(f'  reclaimed   {ev.get("job_id")}  by {ev.get("worker")}: '
                  f'{ev.get("reason")}')
        for ev in claim_lost:
            print(f'  lost mid-run {ev.get("job_id")}  {ev.get("worker")} '
                  f'finished without owning its directory: {ev.get("reason")}')
    if unsound:
        print('\nUnsound manifest -- the budget was reached but the run failed '
              'its own metrics integrity check, so it is neither complete nor '
              'safely resumable (DESIGN.md 8.2(1)):')
        for rec, why in unsound:
            print(f'  {rec["job_id"]}  {rec["experiment"]}/{rec["arm"]} '
                  f'seed {rec["seed"]}')
            print(f'    reason: {why}')
            print(f'    run:    {rec["run_dir"]}')
    if unfinished:
        print('\nNot finished -- no terminal record and no complete manifest. '
              'Re-run with --resume:')
        for rec, state in unfinished[:20]:
            print(f'  {rec["job_id"]}  {rec["experiment"]}/{rec["arm"]} '
                  f'seed {rec["seed"]}  (last state: {state})')
        if len(unfinished) > 20:
            print(f'  ... and {len(unfinished) - 20} more')

    ok = not (failed or blocked or unfinished or unsound)
    print('\n' + ('all jobs complete. Next: experiments/aggregate.py'
                  if ok else 'sweep incomplete -- see above.'))
    print('=' * 72)
    return 0 if ok else 1


def print_plan(records: Sequence[dict], exp_ids: Sequence[str],
               membership: dict[str, list[str]], seeds, n_jobs: int,
               verbose: bool) -> None:
    """What will run, before anything is written or launched.

    Run count, seed blocks, membership and worker geometry. Deliberately *not*
    the cost model: measured throughput, projected wall clock per ``--jobs`` and
    disk are `DESIGN.md` §8.5's requirement and `plan.py` is what satisfies it.
    This docstring used to cite §8.5 for output that never included any of them.
    """
    complete = [r for r in records if is_complete(r)]
    pending = [r for r in records if not is_complete(r)]
    seed_set = sorted({r['seed'] for r in records})
    blocks: dict[str, int] = {}
    for r in records:
        blocks[r['seed_block']] = blocks.get(r['seed_block'], 0) + 1

    # Requested and present are different lists, and conflating them would be
    # misleading: a source arm drawn from another block (E8i's C4SRC donors)
    # legitimately adds seeds nobody asked for on the target side.
    seed_text = ('each experiment block as declared' if seeds is None
                 else f'{len(seeds)} requested: {list(seeds)}')
    print(f'experiments: {", ".join(exp_ids)}')
    print(f'seeds:       {seed_text}')
    print(f'             {len(seed_set)} present in the plan: {seed_set}')
    print(f'jobs:        {len(records)} distinct run directories '
          f'({len(complete)} already complete, {len(pending)} to run)')
    print('seed blocks: ' + ', '.join(f'{k}={v}'
                                      for k, v in sorted(blocks.items())))
    print(f'workers:     {n_jobs} x {max(1, (os.cpu_count() or 4) // max(1, n_jobs))} '
          f'threads  (cpu_count={os.cpu_count()})')

    print('\nper experiment (membership; runs are shared between experiments '
          'when their configs are identical):')
    for eid in exp_ids:
        exp = registry.EXPERIMENTS[eid]
        dirs = membership[eid]
        done = sum(1 for r in records if r['run_dir'] in set(dirs)
                   and is_complete(r))
        print(f'  {eid:4s} {exp.name:14s} tier {exp.tier}  '
              f'{exp.family:12s} block {exp.seed_block:9s} '
              f'{len(dirs):4d} runs  {done:4d} complete')

    # n is counted on the target side only: a source run is an input to an arm,
    # not an observation of it. Two exclusions, and they are not the same test.
    #
    # `role == 'source'` is the run that exists only to be transferred from.
    # `is_source` is *not* usable here even though it is the right test for
    # membership of phase 1: in E8 and E9 a `scratch-*` run is simultaneously an
    # observation of its own arm and the source for the shift arms, so excluding
    # every phase-1 member would delete a real seed from a real arm.
    #
    # A RESERVE seed is excluded whatever its role. `DESIGN.md` §3.4 gives that
    # block exactly one use, replacement sources drawn by the validity gate, and
    # `check_seed_blocks` refuses it as a hand-picked selection, so a RESERVE
    # seed in a plan is a draw and never an observation. This is what the
    # earlier `role != 'source'` test got wrong: E8's shift arms draw their
    # source from a `scratch-*` arm whose declared role is 'target', so a
    # replacement counted as an extra target seed and suppressed the n<3 note on
    # arms that had two.
    #
    # And per arm, not pooled. The union of seeds across arms cannot detect an
    # arm missing a seed, which is the incomplete-arm condition `DESIGN.md` §8.4
    # exists to catch; printing it as "N seed(s) per arm" said something the
    # number did not support.
    per_arm: dict[tuple[str, str], set[int]] = {}
    for r in records:
        if r.get('role') == 'source' or r.get('seed_block') == 'RESERVE':
            continue
        for eid in (r.get('experiments') or [r['experiment']]):
            per_arm.setdefault((eid, r['arm']), set()).add(r['seed'])
    if per_arm:
        sizes = {key: len(seeds) for key, seeds in per_arm.items()}
        smallest = min(sizes.values())
        largest = max(sizes.values())
        print(f'target arms: {len(per_arm)} arm(s) outside phase 1; seeds per '
              f'arm min {smallest}, max {largest}')
        # Short *within its own experiment*, not against the whole selection.
        # Experiments declare different blocks: E0 runs one SMOKE seed and E3
        # five TUNE seeds by design, so comparing every arm against the largest
        # arm anywhere in the plan flagged both of them as incomplete on a
        # `--experiments all` selection and buried the real case. DESIGN.md
        # §8.4's incomplete arm is an arm missing a seed its siblings have.
        per_exp_max: dict[str, int] = {}
        for (eid, _arm), size in sizes.items():
            per_exp_max[eid] = max(per_exp_max.get(eid, 0), size)
        thin = sorted(k for k, v in sizes.items() if v < per_exp_max[k[0]])
        if thin:
            print(f'             [WARNING] {len(thin)} arm(s) hold fewer seeds '
                  f'than the other arms of the same experiment, which is '
                  f'DESIGN.md §8.4\'s incomplete-arm condition:')
            for eid, arm in thin[:8]:
                print(f'               {eid} {arm}: {sizes[(eid, arm)]} of '
                      f'{per_exp_max[eid]} seed(s), '
                      f'{sorted(per_arm[(eid, arm)])}')
            if len(thin) > 8:
                print(f'               ... and {len(thin) - 8} more')
        if smallest < 3:
            print(f'\n[NOTE] the smallest target arm holds {smallest} seed(s). '
                  f'Under ANALYSIS_PLAN.md §9 nothing produced at n<3 is a '
                  f'result: stats.py emits no test and no interval, and '
                  f'report.py stamps every page PIPELINE VALIDATION - NOT A '
                  f'RESULT.')

    if verbose:
        print('\njobs, in dependency order:')
        for r in records:
            mark = 'done' if is_complete(r) else '   .'
            dep = f' <- {os.path.basename(os.path.dirname(r["depends_on"]))}' \
                if r['depends_on'] else ''
            print(f'  {mark}  {r["job_id"]}  {r["experiment"]:4s} '
                  f'{r["arm"]:28s} s{r["seed"]:02d} {r["seed_block"]:9s} '
                  f'{r["condition"]:19s} {r["env"]}{dep}')
            print(f'        {r["run_dir"]}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--experiments', nargs='+', default=None,
                   help="experiment ids from the registry, or 'all'")
    p.add_argument('--tier', type=int, default=None, choices=(1, 2, 3),
                   help='every experiment at this tier (DESIGN.md §7)')
    p.add_argument('--seeds', nargs='+', default=None,
                   help="block names and/or seeds: 'CONFIRM', '0-9', "
                        "'0-4 10-19'. Default: each experiment's declared block")
    p.add_argument('--jobs', type=int, default=1,
                   help='worker processes. The parallel axis is the run '
                        'directory, so any value is safe')
    p.add_argument('--out-root', default='runs')
    p.add_argument('--override', action='append', default=None,
                   metavar='KEY=VALUE',
                   help='Config field override applied to every job, e.g. '
                        '--override num_episodes=14')
    p.add_argument('--dry-run', action='store_true',
                   help='print the plan and write nothing at all')
    p.add_argument('--resume', action='store_true',
                   help='the default behaviour, stated explicitly: skip '
                        'complete runs, reclaim stale or previously failed '
                        'claims, and leave everything else alone. Recorded in '
                        'the invocation file so the intent is in the provenance')
    p.add_argument('--force', action='store_true',
                   help='re-enter run directories that already have a complete '
                        'manifest. This does NOT retrain: train.py resumes from '
                        'the checkpoint, so a complete run re-enters at its '
                        'last episode and trains nothing. Deleting a run '
                        'directory is the only way to retrain it, and this '
                        'runner will not delete data. It does NOT touch claims '
                        'either: see --force-claim')
    p.add_argument('--force-claim', action='store_true',
                   help='take a run directory away from a claim another sweep '
                        'holds, including one whose owner is demonstrably alive '
                        'and one this process cannot read. That is the P0 '
                        'corruption made deliberate: two trainers in one '
                        'directory writing one metrics.jsonl, and this runner '
                        'cannot interrupt the first one. It used to be the '
                        'second half of --force, which meant the flag an '
                        'operator reaches for at 3am to re-enter finished '
                        'directories also handed live directories to a second '
                        'worker. Reach for it when a crashed worker\'s pid has '
                        'been reused (the deferral note says so) and never to '
                        'get past a wait')
    p.add_argument('--verbose', action='store_true',
                   help='list every job in the plan')
    p.add_argument('--stale-seconds', type=int, default=DEFAULT_STALE_SECONDS,
                   help='a claim that has gone this long without a heartbeat '
                        'may be reclaimed, and only then with evidence its '
                        'owner is gone: the pid is checked on this host, and a '
                        'claim from another host must carry a heartbeat field. '
                        'A live run refreshes its claim every '
                        f'{HEARTBEAT_SECONDS}s, so age means "how long since '
                        'the owner was last alive", not "how long this run has '
                        'been going" (default 7200)')
    p.add_argument('--poll-seconds', type=int, default=DEFAULT_POLL_SECONDS,
                   help='worker wait between passes when only deferred jobs '
                        'remain')
    p.add_argument('--max-wait-seconds', type=int, default=0,
                   help='give up waiting on a dependency after this long and '
                        'report it blocked; 0 means stale-seconds + 300')
    p.add_argument('--progress-seconds', type=int, default=60,
                   help='parent progress tally cadence; 0 to disable')
    p.add_argument('--max-source-replacements', type=int, default=None,
                   metavar='N',
                   help='compute ceiling: how many RESERVE replacement source '
                        'runs THIS INVOCATION may commit to training before it '
                        'stops and refuses. Default: one full round of '
                        'replacement across the plan\'s source lineages, and '
                        'never fewer than '
                        f'{DEFAULT_MIN_SOURCE_REPLACEMENTS}. DESIGN.md 4.3 '
                        'draws replacements until every cell has its full '
                        'complement and puts no bound on that; this bounds one '
                        'command, not the rule. Unbounded, the source phase can '
                        'commit hundreds of hours of source training unattended '
                        'on the exact command the campaign uses, and a stage '
                        'that rejects everything it trains is usually one '
                        'systematic fault rather than N unlucky seeds: P0 was '
                        'exactly that. The ledger is durable and the assignment '
                        'is derived from it, so re-running after the refusal '
                        'continues from where it stopped and retrains nothing')
    p.add_argument('--allow-factor-overrides', action='store_true',
                   help='permit --override on a field the registry classes as '
                        'an experimental factor rather than a budget setting. '
                        'The runs carry a note saying so. Whether their '
                        'configuration digests differ depends on the field and '
                        'is NOT guaranteed: src/dqn/config.py excludes '
                        'BOOKKEEPING_FIELDS from every digest and '
                        'TRANSFER_ONLY_FIELDS from a scratch run\'s, so such an '
                        'override writes into the catalogue directories under '
                        'the catalogue identity. The exact effect per field is '
                        'printed at launch')
    p.add_argument('--allow-seed-block-override', action='store_true',
                   help='proceed despite a seed-block violation. Stamped into '
                        'the invocation record; audit.py will still refuse the '
                        'affected estimates')
    p.add_argument('--allow-invalid-sources', action='store_true',
                   help='proceed into the target phase even though a cell '
                        'has no valid source: either RESERVE is exhausted, or a '
                        'source finished without producing the normalised final '
                        'score DESIGN.md 4.3 defines validity on. The cell is '
                        'then transfer-from-a-source-of-unknown-competence, '
                        'which is the published study\'s central defect; the '
                        'override is stamped into the invocation record and '
                        'into every rejection row so the affected cells are '
                        'identifiable in the results table')
    p.add_argument('--allow-nondesign-gate', action='store_true',
                   help='permit --source-gate to differ from the '
                        'pre-registered DESIGN.md 4.3 value while a reporting '
                        'experiment is selected. Without it the combination is '
                        'refused rather than warned about: the help for '
                        '--source-gate has always said a confirmatory run must '
                        'use the design value, and nothing enforced it')
    p.add_argument('--source-gate', type=float, default=None,
                   metavar='SCORE',
                   help='override the source-validity gate. A TESTING '
                        'instrument: DESIGN.md 4.3 fixes the gate at '
                        '0.6 normalised score and a confirmatory run must use '
                        'that value. Any other value is printed as a warning, '
                        'recorded in the invocation file, and stamped into '
                        'every rejection row as gate_is_design_value=false')
    # Worker mode. Not for interactive use: a worker takes no seed
    # specification, only the resolved job manifest, which is what makes the
    # seed round-tripping defect impossible to reintroduce.
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--jobs-file', default=None, help=argparse.SUPPRESS)
    p.add_argument('--worker-id', type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument('--sweep-id', default=None, help=argparse.SUPPRESS)
    return p


def _select_experiments(args: argparse.Namespace) -> list[str]:
    if args.experiments:
        if len(args.experiments) == 1 and args.experiments[0].lower() == 'all':
            return list(registry.EXPERIMENTS)
        unknown = [e for e in args.experiments if e not in registry.EXPERIMENTS]
        if unknown:
            raise SystemExit(f'unknown experiment(s) {unknown}; known: '
                             f'{list(registry.EXPERIMENTS)}')
        return list(dict.fromkeys(args.experiments))
    if args.tier is not None:
        return [e.id for e in registry.TIERS[args.tier]]
    raise SystemExit('nothing selected: pass --experiments (ids or "all") or '
                     '--tier {1,2,3}. Start with --experiments E0 --dry-run.')


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.worker:
        if not args.jobs_file:
            raise SystemExit('--worker requires --jobs-file')
        args.sweep_id = args.sweep_id or 'adhoc'
        return worker_main(args)

    args.sweep_id = time.strftime('%Y%m%d-%H%M%S') + f'-{os.getpid()}'
    # An absolute out_root removes the last way a worker and its parent can
    # disagree about which directory a job owns. out_root is bookkeeping, not
    # identity (src/dqn/config.py, BOOKKEEPING_FIELDS), so this does not change
    # any run digest.
    args.out_root = os.path.abspath(args.out_root)
    exp_ids = _select_experiments(args)
    # A malformed selection is a refusal, not a traceback. `--seeds ''`,
    # `--seeds abc` and `--override num_episodes=0` all used to surface as raw
    # ValueErrors from three different modules.
    try:
        seeds = resolve_seed_spec(args.seeds)
        overrides = parse_overrides(args.override)
    except ValueError as exc:
        print(f'[REFUSED] {exc}')
        return 2

    fatal, warnings = check_seed_blocks(exp_ids, seeds)
    for msg in warnings:
        print(f'[warning] {msg}')
    if fatal:
        for msg in fatal:
            print(f'[REFUSED] {msg}')
        if not args.allow_seed_block_override:
            print('\nPass --allow-seed-block-override to proceed anyway. The '
                  'override is recorded in the invocation file and audit.py '
                  'will refuse the affected estimates.')
            return 2
        print('[override] proceeding despite the above, as requested.')

    factor_overrides = sorted(set(overrides) - registry.SCALING_FIELDS)
    if factor_overrides:
        print(f'[warning] --override touches experimental factors '
              f'{factor_overrides}, not just the budget. These runs are not the '
              f'catalogue arms they are labelled as, and their manifests carry '
              f'a note recording it.')
    # Which overrides actually move a run's identity, stated per field rather
    # than asserted for all of them. The old blanket claim ("their digests
    # differ, so they cannot be mistaken for catalogue runs") was false for
    # every bookkeeping field: `--override log_diagnostics=False` is in
    # SCALING_FIELDS so it raised no factor warning, is bookkeeping so the
    # digest was unchanged, and wrote a run with no DESIGN.md 5.5 mechanism
    # instrumentation into the catalogue directory under the catalogue digest.
    bookkeeping_overrides = sorted(set(overrides) & set(BOOKKEEPING_FIELDS))
    if bookkeeping_overrides:
        print(f'[warning] --override {bookkeeping_overrides} are bookkeeping '
              f'fields (src/dqn/config.py, BOOKKEEPING_FIELDS). They are '
              f'outside every run digest, so these runs keep the catalogue '
              f'identity and write into the catalogue run directories: nothing '
              f'but the manifest note distinguishes them. '
              + ('log_diagnostics in particular removes the DESIGN.md 5.5 '
                 'mechanism instrumentation while the digest still claims a '
                 'catalogue run. ' if 'log_diagnostics' in overrides else '')
              + 'Making the digest cover them belongs in src/dqn/config.py, '
                'which this file does not own.')
    transfer_only_overrides = sorted(set(overrides) & set(TRANSFER_ONLY_FIELDS))
    if transfer_only_overrides:
        print(f'[warning] --override {transfer_only_overrides} change the '
              f'digest of transfer runs only: a scratch run excludes the '
              f'transfer-only fields from its digest (src/dqn/config.py), so '
              f'the scratch arms keep their catalogue directories while their '
              f'transfer counterparts move to new ones.')

    # Experiments whose output is reported. Everything except the SMOKE-block
    # pipeline-validation experiment: DESIGN.md 3.4 and ANALYSIS_PLAN.md 8 speak
    # about reported estimates, and estimation and screen experiments report as
    # surely as confirmatory ones do.
    reporting = [eid for eid in exp_ids
                 if registry.EXPERIMENTS[eid].seed_block != 'SMOKE']
    # The one selection DESIGN.md 4.3's gate does not govern, stated as a
    # property of the whole selection rather than as an exemption any experiment
    # carries. Two conditions, both necessary: every experiment selected
    # declares the SMOKE block, so nothing here reports an estimate; and every
    # seed asked for is a SMOKE seed, so a smoke arm cannot be re-pointed at
    # CONFIRM seeds and inherit the scope statement with it. A reporting
    # experiment cannot acquire this by accident: it would have to be moved into
    # the SMOKE block in registry.py, which is the declaration that it validates
    # the pipeline rather than measuring anything, and it would have to be
    # selected alone.
    smoke_only = bool(exp_ids) and not reporting and (
        seeds is None or set(seeds) <= set(registry.SEED_BLOCKS['SMOKE']))
    if args.source_gate is not None:
        gate = float(args.source_gate)
    elif smoke_only:
        gate = GATE_NOT_APPLIED
    else:
        gate = registry.SOURCE_VALIDITY_GATE
    gate_applied = math.isfinite(gate)
    # "The design value" means the pre-registered number *actually applied*. A
    # gate that is not applied is not it either, so every rejection row and the
    # invocation record say gate_is_design_value=false under both.
    gate_overridden = not (
        gate_applied
        and abs(gate - registry.SOURCE_VALIDITY_GATE) <= 1e-12)
    if not gate_applied:
        print('\n' + '=' * 72)
        print('[gate not applied] DESIGN.md 4.3 does not govern this selection')
        print('=' * 72)
        print(f'  Selected: {exp_ids}, every one a SMOKE-block pipeline '
              f'validation.')
        print('  4.3 defines validity on the normalised final score of a '
              'source trained to the')
        print('  design\'s budget, and it governs the sources of REPORTED '
              'estimates. This budget')
        print('  is 12 episodes (registry.SMOKE_OVERRIDES) and this selection '
              'reports nothing:')
        print('  the catalogue says of it, in as many words, "not a result '
              'under any')
        print('  circumstances". Its sources score around zero, so gating them '
              'at '
              f'{registry.SOURCE_VALIDITY_GATE:.3f}')
        print('  rejected every one, drew a RESERVE seed for every one, and '
              'exhausted the block')
        print('  without ever reaching phase 2, which is the half the smoke '
              'exists to validate.')
        print('  NOT APPLIED means: no source is rejected on its score, no '
              'RESERVE seed is drawn,')
        print('  and no row is written to the 4.3 ledger. It does NOT mean the '
              'source phase is')
        print('  skipped, and it does not admit a broken one: a source that '
              'finishes without a')
        print('  finite final_score is still refused, because that is a '
              'pipeline failure and')
        print('  finding pipeline failures is what this run is for.')
        print('  One reporting experiment anywhere in the selection brings the '
              'pre-registered')
        print('  gate back for the whole selection. To exercise the 4.3 '
              'rejection-and-replacement')
        print('  path deliberately, give the smoke a gate it cannot meet: '
              '--source-gate 0.9.')
    elif gate_overridden:
        print(f'[warning] --source-gate {gate:g} replaces the pre-registered '
              f'{registry.SOURCE_VALIDITY_GATE:g} of DESIGN.md 4.3. This is a '
              f'testing instrument. Every rejection recorded under it carries '
              f'gate_is_design_value=false, and no run made under it is a '
              f'confirmatory result.')
    if gate_overridden:
        if reporting and not args.allow_nondesign_gate:
            print(f'[REFUSED] {reporting} report estimates, and DESIGN.md 4.3 '
                  f'fixes the gate at {registry.SOURCE_VALIDITY_GATE:g}. Under '
                  f'a loosened gate a source that the design rejects is '
                  f'admitted with no rejection row anywhere, so the only '
                  f'record of the choice would be the invocation file. Pass '
                  f'--allow-nondesign-gate to proceed deliberately, or select '
                  f'only the smoke experiment, or run against a scratch '
                  f'--out-root.')
            return 2
        if reporting:
            print('[override] proceeding with a non-design source gate on a '
                  'reporting selection, as requested.')

    jobs_dir = os.path.join(args.out_root, JOBS_SUBDIR)
    ledger_path = os.path.join(jobs_dir, REPLACEMENTS_FILE)
    ledger = read_replacements(ledger_path)

    # The source assignment is settled before anything is printed, because the
    # plan a reader is shown has to be the plan that will run: a rejected source
    # moves its dependants onto a different checkpoint, and a job list printed
    # before that resolution would name a source the sweep is not going to use.
    resolved = resolve_source_assignment(
        exp_ids, seeds, args.out_root, overrides, args.allow_factor_overrides,
        gate, ledger, gate_overridden, args.sweep_id)
    records, membership = resolved['records'], resolved['membership']

    # The compute ceiling on the RESERVE replacement rule, for THIS invocation.
    # `committed` counts replacement source runs this command has taken on: the
    # draws it makes now from scores already on disk, plus every draw a later
    # source stage makes. Draws replayed from the ledger are not counted, since
    # an earlier invocation committed to those and re-deriving them costs
    # nothing.
    lineages = {sl.lineage for sl in resolved['slots']}
    committed = len(resolved['new_rejections'])
    max_replacements = (max(DEFAULT_MIN_SOURCE_REPLACEMENTS, len(lineages))
                        if args.max_source_replacements is None
                        else int(args.max_source_replacements))
    if max_replacements < 0:
        print('[REFUSED] --max-source-replacements is a count of source runs '
              'and cannot be negative.')
        return 2

    print(f'sweep {args.sweep_id}')
    # A budget below one episode makes the completion test read `0 >= 0` and
    # certify every directory it touches as finished. `parse_overrides` refuses
    # it on the command line; this catches it wherever else it could come from.
    bad_budget = [r for r in records
                  if (_as_int(r.get('num_episodes')) or 0) < 1]
    if bad_budget:
        print(f'[REFUSED] {len(bad_budget)} job(s) ask for fewer than one '
              f'episode, e.g. {bad_budget[0]["job_id"]} at '
              f'num_episodes={bad_budget[0].get("num_episodes")!r}. A run of '
              f'zero episodes is not a smaller experiment: it is a directory '
              f'certified complete without training, indexed as an experiment '
              f'member, with final_score null.')
        return 2
    print_plan(records, exp_ids, membership, seeds, args.jobs, args.verbose)
    print_source_phase(resolved['slots'], records, gate, ledger,
                       resolved['new_rejections'], resolved['exhausted'])
    # What the reserve rule could cost, before it starts costing it. The rule is
    # unbounded by design ("drawn in order from RESERVE until the cell has its
    # full complement"), and unbounded on this plan means the number below,
    # committed unattended by the same command that trains the campaign.
    print(f'  ceiling:   {max_replacements} replacement source run(s) may be '
          f'committed by this')
    print(f'             invocation (--max-source-replacements); {committed} '
          f'drawn so far.')
    print(f'             Unbounded, {len(lineages)} lineage(s) x '
          f'{len(registry.RESERVE_ORDER)} RESERVE seed(s) = '
          f'{len(lineages) * len(registry.RESERVE_ORDER)} further source '
          f'run(s)')
    print('             could be committed before the sweep refuses. The '
          'ceiling stops the')
    print('             command, not the rule: the ledger is durable, so '
          're-running with a')
    print('             higher one continues from where it stopped and '
          'retrains nothing.')
    if args.dry_run:
        # Checked here only for the dry run. In a live sweep the phase-1 loop
        # can still move the assignment, so the check that decides anything is
        # the one after phase 1; printing both would show the same block twice.
        print_lineage_conflicts(lineage_conflicts(records))
        print_unsound_runs(unsound_runs(records))
        print_ungated_sources(
            [row for row in ungated_sources(resolved['slots'], gate)
             if row['state'] != 'unrun'], gate)
        if resolved['new_rejections']:
            print(f'\n--dry-run: {len(resolved["new_rejections"])} '
                  f'rejection(s) would be appended to {ledger_path}.')
        print('\n--dry-run: nothing written, nothing launched.')
        return 0

    unsound = unsound_runs(records)
    if unsound:
        print_unsound_runs(unsound)
        return 6

    # Litter from a reclaim, in the directories this plan touches. Nothing ever
    # removed these: two were still in the P0 tree when the verification pass
    # went looking for them.
    litter = purge_claim_litter(*[r['run_dir'] for r in records])
    if litter:
        print(f'\nremoved {litter} claim temporary file(s) left behind by an '
              f'earlier reclaim, heartbeat or hard kill')

    os.makedirs(jobs_dir, exist_ok=True)
    jobs_path = os.path.join(jobs_dir, JOBS_FILE)
    status_path = os.path.join(jobs_dir, STATUS_FILE)

    def publish(recs: Sequence[dict], memb: dict[str, list[str]]) -> None:
        """Keep jobs.jsonl and the index describing the *current* assignment.

        Rewritten after every reassignment rather than once at launch: a sweep
        killed mid-run is meant to be inspectable, and a job file naming sources
        that have since been rejected would describe a sweep nobody ran.
        """
        write_jobs(jobs_path, recs)
        # The rejected directories travel into the index so that a RESERVE
        # replacement is not indistinguishable, in `arm`, from the genuine
        # CONFIRM members of the arm it shares a label with. The index is
        # merged and never pruned by design (see `write_index`), so the marking
        # is how a rejection reaches a consumer that groups by (experiment, arm).
        write_index(args.out_root, memb, recs,
                    rejected_dirs=[row.get('rejected_run_dir')
                                   for row in read_replacements(ledger_path)
                                   if row.get('rejected_run_dir')])

    publish(records, membership)
    print(f'\nwrote {jobs_path} ({len(records)} jobs)')
    n_recorded = append_replacements(
        ledger_path, list(resolved['new_rejections']) + list(resolved['exhausted']))
    if n_recorded:
        print(f'wrote {n_recorded} source-validity rejection(s) to '
              f'{ledger_path}')

    if args.force:
        print('\n[--force] complete manifests will be re-entered. train.py '
              'resumes from the checkpoint, so a finished run trains nothing; '
              'delete its directory if you mean to retrain it. Claims are not '
              'affected: that is --force-claim.')
    if args.force_claim:
        print('\n[--force-claim] claims held by other sweeps will be taken '
              'over, including claims whose owner is demonstrably alive. This '
              'runner cannot interrupt a running train(), so the owner keeps '
              'writing: two trainers in one metrics.jsonl is the P0 corruption '
              'DESIGN.md 8.2(1) exists to prevent. Every takeover is recorded '
              'as a reclaim and is named in the summary.')

    pending = [r for r in records if args.force or not is_complete(r)]
    n_jobs = max(1, min(args.jobs, len(pending))) if pending else 0
    invocation = {
        'sweep': args.sweep_id, 'argv': list(sys.argv), 'cwd': os.getcwd(),
        'host': socket.gethostname(), 'pid': os.getpid(),
        't': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'experiments': exp_ids, 'seeds': list(seeds) if seeds else None,
        'overrides': overrides, 'out_root': args.out_root,
        'factor_overrides': factor_overrides,
        'factor_override_allowed': bool(args.allow_factor_overrides),
        'jobs_total': len(records), 'jobs_pending': len(pending),
        'workers': n_jobs,
        'threads_per_worker': max(1, (os.cpu_count() or 4) // max(1, n_jobs or 1)),
        'resume': bool(args.resume), 'force': bool(args.force),
        'force_claim': bool(args.force_claim),
        'stale_seconds': args.stale_seconds,
        'seed_block_override': bool(args.allow_seed_block_override and fatal),
        # The non-fatal block departures too. A run on a block the experiment
        # does not declare is a fact about the provenance whether or not it was
        # refused, and recording only the overridden refusals left the rest
        # invisible.
        'seed_block_warnings': list(warnings),
        'bookkeeping_overrides': bookkeeping_overrides,
        'transfer_only_overrides': transfer_only_overrides,
        'source_gate_override_allowed': bool(args.allow_nondesign_gate),
        # The reserve rule's provenance: the gate actually applied, whether it
        # is the pre-registered one, and the assignment this sweep ran under.
        'source_gate': gate,
        'source_gate_is_design_value': not gate_overridden,
        # Whether a gate was applied at all, and why not. A non-finite gate is
        # written to this file as null (`_json_safe`), so the boolean is what
        # says which of "no gate" and "unreadable value" produced the null.
        'source_gate_applied': gate_applied,
        'source_gate_scope': (
            'DESIGN.md 4.3' if gate_applied else
            'not applied: every experiment selected is a SMOKE-block pipeline '
            'validation at SMOKE-block seeds, which reports no estimate and '
            'trains a budget the gate is not defined on'),
        'max_source_replacements': max_replacements,
        'source_replacements_committed_at_launch': committed,
        'source_replacements_recorded': n_recorded,
        'source_assignment': {f'{arm}@s{seed}': int(rep) for (arm, seed), rep
                              in sorted(resolved['assignment'].items())},
        'invalid_sources_allowed': bool(args.allow_invalid_sources),
        'plan_hashes': _plan_hashes(),
    }
    with open(os.path.join(jobs_dir, f'sweep-{args.sweep_id}.json'), 'w',
              encoding='utf-8') as fh:
        # Through `_json_safe` for the same reason the status log is: a
        # non-finite number is written by json.dumps as a bare `-Infinity`,
        # which Python reads back and no strict JSON reader will.
        json.dump(_json_safe(invocation), fh, indent=2, sort_keys=True,
                  default=str)
    append_status(status_path, {**{k: invocation[k] for k in
                                  ('sweep', 'argv', 'experiments', 'seeds',
                                   'jobs_total', 'jobs_pending', 'workers')},
                                'state': 'sweep_start',
                                'ts': round(time.time(), 3),
                                't': invocation['t']})

    # ---- phase 1: sources, gated, replaced, repeated -------------------
    # `--force` is not honoured here. It re-enters a complete directory, and
    # train.py resumes rather than retrains, so forcing the source phase would
    # leave the same complete manifests in place and the loop would never
    # empty. Deleting a source's directory is still the way to retrain it.
    stage = 0
    stalled: list[dict] = []
    # Kept across the loop so that a stall can still say which log to read.
    # `summarise` was called with an empty map in exactly that situation, so it
    # printed 'log: ?' in the one case where its docstring promises the log.
    stage_logs: dict[str, str] = {}
    while True:
        due = [r for r in records if r['is_source'] and not is_complete(r)]
        if not due:
            break
        if committed > max_replacements:
            # Refused before the stage runs, so the ceiling costs nothing to
            # observe. It is not a change to DESIGN.md 4.3, which draws until
            # every cell has its full complement: it is a bound on how much one
            # unattended command may commit to without being asked again.
            print('\n' + '=' * 72)
            print(f'[REFUSED] the RESERVE replacement rule has committed '
                  f'{committed} new source run(s) in this')
            print(f'          invocation, past the {max_replacements} this '
                  f'command allows')
            print('=' * 72)
            print('  Every rejection is recorded in ' + ledger_path + ', and '
                  'the assignment is')
            print('  derived from it, so nothing is lost and nothing will be '
                  'retrained.')
            print('  DESIGN.md 4.3 draws replacements until every cell has its '
                  'full complement and')
            print('  puts no bound on that. This bound is on ONE INVOCATION. '
                  'It exists because the')
            print('  loop is otherwise free to commit hundreds of source-run '
                  'hours unattended on')
            print('  the exact command the campaign uses, and because a source '
                  'stage that rejects')
            print('  everything it trains is usually one systematic fault and '
                  'not N unlucky')
            print('  seeds: in P0 a single exploration-schedule defect failed '
                  'all four sources at')
            print('  once, and spending the reserve on it would have bought '
                  'nothing.')
            print('  Read the rejections above. If they are genuine, re-run '
                  'with a higher')
            print('  --max-source-replacements and the sweep carries on from '
                  'here.')
            append_status(status_path, {
                'sweep': args.sweep_id, 'state': 'sweep_end',
                'ts': round(time.time(), 3),
                't': time.strftime('%Y-%m-%dT%H:%M:%S'), 'exit_code': 7,
                'reason': f'{committed} replacement source run(s) committed, '
                          f'past the --max-source-replacements ceiling of '
                          f'{max_replacements}',
                'source_replacements_committed': committed,
                'max_source_replacements': max_replacements})
            return 7
        stage += 1
        print(f'\n{"=" * 72}\nphase 1, source stage {stage}: {len(due)} '
              f'source run(s) to train, then gate: {gate_text(gate)}\n'
              f'({committed} of at most {max_replacements} replacement source '
              f'run(s) committed so far)\n{"=" * 72}')
        _codes, stage_logs = run_stage(args, due, f'src{stage}', status_path)

        ledger = read_replacements(ledger_path)
        resolved = resolve_source_assignment(
            exp_ids, seeds, args.out_root, overrides,
            args.allow_factor_overrides, gate, ledger, gate_overridden,
            args.sweep_id)
        records, membership = resolved['records'], resolved['membership']
        publish(records, membership)
        n_new = append_replacements(
            ledger_path,
            list(resolved['new_rejections']) + list(resolved['exhausted']))
        n_recorded += n_new
        committed += len(resolved['new_rejections'])
        print_source_phase(resolved['slots'], records, gate, ledger,
                           resolved['new_rejections'], resolved['exhausted'])
        if n_new:
            print(f'  recorded {n_new} rejection(s) in {ledger_path}')

        still = [r for r in records if r['is_source'] and not is_complete(r)]
        if {r['run_dir'] for r in still} == {r['run_dir'] for r in due}:
            # The stage trained nothing it was given. Looping would spin on the
            # same failures; the worker log already names the cause.
            stalled = still
            print(f'\n[REFUSED] the source stage made no progress: '
                  f'{len(still)} source run(s) still have no complete '
                  f'manifest. Read the stage log above, fix the cause, and '
                  f're-run; the target phase is not started against sources '
                  f'whose validity is unknown.')
            break

    # ---- refusals between the phases -----------------------------------
    if stalled:
        return summarise(records, status_path, args.sweep_id, stage_logs) or 1

    if resolved['exhausted']:
        print('\n' + '=' * 72)
        print('[ERROR] RESERVE exhausted -- DESIGN.md 4.3 cannot be satisfied')
        print('=' * 72)
        for row in resolved['exhausted']:
            print(f'  {row["error"]}')
            print(f'    last rejected: s{row["rejected_seed"]:02d} score '
                  f'{row["score"]:.3f} < {row["gate"]:.3f}  '
                  f'{row["rejected_run_dir"]}')
            print(f'    {len(row["consumers"])} dependent run(s) have no valid '
                  f'source')
        print('\n  Every rejection is in ' + ledger_path + '.')
        print('  This is not a condition to run through: a cell whose sources '
              'never learned the')
        print('  source task measures transfer-from-nothing, which is the '
              'defect DESIGN.md 4.3')
        print('  exists to prevent. Widen RESERVE in registry.py, or fix what '
              'is stopping the')
        print('  sources from learning, or pass --allow-invalid-sources to '
              'proceed deliberately.')
        if not args.allow_invalid_sources:
            append_status(status_path, {
                'sweep': args.sweep_id, 'state': 'sweep_end',
                'ts': round(time.time(), 3),
                't': time.strftime('%Y-%m-%dT%H:%M:%S'), 'exit_code': 3,
                'reason': 'RESERVE exhausted for '
                          f'{len(resolved["exhausted"])} lineage(s)'})
            return 3
        print('\n[override] proceeding with invalid sources, as requested. '
              'Recorded in the invocation file.')

    # Every source that phase 1 left without a validity verdict. Not
    # 'rejected', so no RESERVE seed was drawn and no ledger row written; not
    # 'valid', so nothing licenses the transfer runs beneath it. The earlier
    # version had no state between those two and let the whole target side run
    # against it, returning 0.
    ungated = ungated_sources(resolved['slots'], gate)
    if ungated:
        print_ungated_sources(ungated, gate)
        if not args.allow_invalid_sources:
            append_status(status_path, {
                'sweep': args.sweep_id, 'state': 'sweep_end',
                'ts': round(time.time(), 3),
                't': time.strftime('%Y-%m-%dT%H:%M:%S'), 'exit_code': 5,
                'reason': f'{len(ungated)} source slot(s) have no '
                          f'DESIGN.md 4.3 validity verdict',
                'ungated_sources': [row['describe'] for row in ungated]})
            return 5
        print('\n[override] proceeding against sources with no validity '
              'verdict, as requested. Recorded in the invocation file.')

    conflicts = lineage_conflicts(records)
    if conflicts:
        print_lineage_conflicts(conflicts)
        append_status(status_path, {
            'sweep': args.sweep_id, 'state': 'sweep_end',
            'ts': round(time.time(), 3),
            't': time.strftime('%Y-%m-%dT%H:%M:%S'), 'exit_code': 4,
            'reason': f'{len(conflicts)} run(s) hold data from a source the '
                      f'validity gate has since rejected'})
        return 4

    # ---- phase 2: everything else --------------------------------------
    pending = [r for r in records if args.force or not is_complete(r)]
    # One 'skipped' record per already-complete run, written by the parent so
    # that the count is stated once rather than once per worker.
    if not args.force:
        for rec in records:
            if is_complete(rec):
                append_status(status_path, {
                    'ts': round(time.time(), 3),
                    't': time.strftime('%Y-%m-%dT%H:%M:%S'),
                    'sweep': args.sweep_id, 'worker': 'parent',
                    'state': 'skipped', 'job_id': rec['job_id'],
                    'run_dir': rec['run_dir'],
                    'experiment': rec['experiment'], 'arm': rec['arm'],
                    'seed': rec['seed'],
                    'reason': 'manifest reports episodes_completed >= '
                              f'{rec["num_episodes"]} before this sweep'})

    log_paths: dict[str, str] = {}
    if not pending:
        print('\nNothing to run: every job already has a complete manifest.')
    else:
        print(f'\n{"=" * 72}\nphase 2, target stage: {len(pending)} run(s)'
              f'\n{"=" * 72}')
        print('Waiting. Tail any worker log to follow progress.', flush=True)
        _codes, log_paths = run_stage(args, pending, 'targets', status_path)

    code = summarise(records, status_path, args.sweep_id, log_paths)
    append_status(status_path, {'sweep': args.sweep_id, 'state': 'sweep_end',
                                'ts': round(time.time(), 3),
                                't': time.strftime('%Y-%m-%dT%H:%M:%S'),
                                'exit_code': code,
                                'source_replacements': n_recorded,
                                'source_replacements_committed': committed,
                                'ungated_sources_allowed': len(ungated)})
    return code


def _plan_hashes() -> dict:
    """Hash the governing documents into the invocation record.

    The same hashes go into every manifest (`src/dqn/provenance.py`); recording
    them at the sweep level too means a launch can be tied to the version of the
    pre-registered plan that was in force when it started, which is what
    `ANALYSIS_PLAN.md` §1 requires of a confirmatory run.
    """
    from src.dqn import provenance                         # noqa: PLC0415
    return provenance.plan_hashes()


if __name__ == '__main__':
    sys.exit(main())
