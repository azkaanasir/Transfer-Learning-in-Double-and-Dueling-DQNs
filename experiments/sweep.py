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
arithmetic that produced defect 1 does not exist.

**Defect 3 -- dependencies were assumed rather than checked.** The old driver
printed ``SKIP ... source missing`` and carried on with exit code 0, which is the
silent-hole failure mode: an arm quietly missing its transfer runs. A job whose
``depends_on`` run has no manifest is *deferred* and retried; if it is still
unsatisfied when the worker stops it is reported as **blocked** and the process
exits non-zero. `DESIGN.md` §8.4 and `ANALYSIS_PLAN.md` §8 both forbid a run
disappearing without a stated rule, and a runner that skips silently is how that
happens by accident.

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
    _jobs/status.jsonl        append-only state transitions, all sweeps
    _jobs/sweep-<id>.json     the invocation: argv, seeds, host, thread pinning
    _index/<experiment>.jsonl experiment -> member run_dirs
    _logs/w<NN>-<id>.log      per-worker output
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from typing import Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _path in (_ROOT, _HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import registry                                             # noqa: E402
from src.dqn.config import Config                           # noqa: E402

JOBS_SUBDIR = '_jobs'
INDEX_SUBDIR = '_index'
LOGS_SUBDIR = '_logs'
JOBS_FILE = 'jobs.jsonl'
STATUS_FILE = 'status.jsonl'
CLAIM_NAME = '.claim'

DEFAULT_STALE_SECONDS = 7_200
DEFAULT_POLL_SECONDS = 15

# Conditions that actually read the source checkpoint's weights. The untrained
# control builds its own random source of matched shape, so it needs the
# dependency's *lineage* but not its file; the dependency is still honoured as
# declared, because relaxing it here would mean the runner and the registry
# disagreed about what a job depends on.
_WEIGHT_READING = ('transfer', 'transfer_permuted')

# SMOKE is (0,) and overlaps CONFIRM, so it is not a membership block: seed 0
# belongs to CONFIRM and reporting it as SMOKE would make the per_seed.csv
# `seed_block` column disagree with `DESIGN.md` §3.4's table.
_BLOCK_ORDER = ('CONFIRM', 'REPLICATE', 'TUNE', 'C4SRC', 'RESERVE')


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
    """
    if not tokens:
        return None
    out: list[int] = []
    for tok in ' '.join(str(t) for t in tokens).replace(',', ' ').split():
        if tok in registry.SEED_BLOCKS:
            out.extend(registry.SEED_BLOCKS[tok])
        else:
            # `tok` is not None, so the block argument is unreachable; it exists
            # only to satisfy the signature.
            out.extend(registry.resolve_seeds(tok, 'CONFIRM'))
    if not out:
        raise ValueError(f'--seeds {tokens!r} resolved to no seeds')
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

    Returns ``(fatal, warnings)``.
    """
    fatal: list[str] = []
    warn: list[str] = []
    if seeds is None:
        return fatal, warn
    requested = set(seeds)
    tune = set(registry.SEED_BLOCKS['TUNE'])
    estimation = set(registry.SEED_BLOCKS['CONFIRM']) | set(
        registry.SEED_BLOCKS['REPLICATE'])
    for eid in exp_ids:
        exp = registry.EXPERIMENTS[eid]
        declared = set(registry.SEED_BLOCKS[exp.seed_block])
        if requested == declared:
            continue
        if exp.seed_block == 'TUNE' and requested & estimation:
            fatal.append(
                f'{eid} ({exp.name}) is a selection experiment declared on TUNE, '
                f'and the requested seeds include '
                f'{sorted(requested & estimation)} from CONFIRM/REPLICATE. '
                f'Selecting on seeds that later carry a reported estimate is '
                f'DESIGN.md §11 defect 2. Run it on TUNE.')
        elif exp.family == 'confirmatory' and requested & tune:
            fatal.append(
                f'{eid} ({exp.name}) is confirmatory and the requested seeds '
                f'include {sorted(requested & tune)} from TUNE. '
                f'ANALYSIS_PLAN.md §8 forbids an estimate computed on TUNE '
                f'seeds.')
        else:
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
        try:
            out[key] = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            out[key] = raw
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
        'config': cfg.to_dict(),
    }


def build_plan(exp_ids: Sequence[str], seeds: Optional[tuple[int, ...]],
               out_root: str, overrides: dict | None = None,
               allow_factor_overrides: bool = False
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
                             allow_factor_overrides=allow_factor_overrides)
    membership: dict[str, list[str]] = {}
    for eid in exp_ids:
        dirs = [j.cfg.run_dir() for j in registry.jobs(
            eid, seeds, out_root, overrides,
            allow_factor_overrides=allow_factor_overrides)]
        membership[eid] = list(dict.fromkeys(dirs))

    by_run: dict[str, list[str]] = {}
    for eid, dirs in membership.items():
        for run_dir in dirs:
            by_run.setdefault(run_dir, []).append(eid)

    records = [job_record(j, by_run.get(j.cfg.run_dir(), [j.experiment]))
               for j in jobs]
    return records, membership


def write_jobs(path: str, records: Sequence[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
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
                records: Sequence[dict]) -> list[str]:
    """``_index/<experiment>.jsonl``: which runs belong to which experiment.

    Written by the parent only, and *merged* with whatever is already there. A
    later invocation on a different seed set must not erase the earlier members,
    or `audit.py` would see an incomplete arm and refuse a complete one --
    reproducing by accident the silent seed-dropping of `DESIGN.md` §1.
    """
    by_dir = {r['run_dir']: r for r in records}
    index_dir = os.path.join(out_root, INDEX_SUBDIR)
    os.makedirs(index_dir, exist_ok=True)
    written = []
    for eid, dirs in membership.items():
        path = os.path.join(index_dir, f'{eid}.jsonl')
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
            }
        tmp = path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as fh:
            for key in sorted(merged):
                fh.write(json.dumps(merged[key], sort_keys=True,
                                    default=str) + '\n')
        os.replace(tmp, path)
        written.append(path)
    return written


# ---------------------------------------------------------------------------
# Completion, claims, status
# ---------------------------------------------------------------------------
def manifest_result(run_dir: str) -> Optional[dict]:
    path = os.path.join(run_dir, 'manifest.json')
    try:
        with open(path, encoding='utf-8') as fh:
            return json.load(fh).get('result') or {}
    except (OSError, json.JSONDecodeError):
        return None


def is_complete(rec: dict) -> bool:
    """A run is complete when its own manifest says so, and only then.

    Not "the directory exists", and not "a checkpoint is present". The audit
    found a completed directory being resumed under a different configuration,
    which trained zero episodes and then wrote a manifest whose config never
    described its metrics.
    """
    result = manifest_result(rec['run_dir'])
    if result is None:
        return False
    return int(result.get('episodes_completed') or 0) >= int(rec['num_episodes'])


def read_claim(path: str) -> Optional[dict]:
    try:
        with open(path, encoding='utf-8') as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}


def _reclaim_reason(path: str, stale_seconds: int, sweep_id: str,
                    force: bool) -> Optional[str]:
    """Whether an existing claim may be taken over, and why.

    Three grounds, all of which leave a ``reclaimed`` record in the status log:

    * **stale** -- no manifest and the claim has not been touched for
      ``stale_seconds`` (default 2 h). This is the crashed-worker case; a Colab
      or Kaggle session that dies mid-run leaves exactly this.
    * **failed** -- the claim records a failure from an *earlier* sweep. The
      failure is already recorded in ``status.jsonl``, and ``--resume`` is a
      request to try again, so holding the directory would turn one crash into a
      permanently unrunnable job.
    * **forced** -- ``--force`` with a claim from another sweep.

    A claim written by *this* sweep is never reclaimed on the failed ground: a
    deterministic failure would otherwise be retried forever inside one
    invocation.
    """
    claim = read_claim(path)
    if claim is None:
        return None
    try:
        age = time.time() - os.path.getmtime(path)
    except OSError:
        return None
    other_sweep = claim.get('sweep') != sweep_id
    if age > stale_seconds:
        return f'stale ({age / 3600:.1f} h > {stale_seconds / 3600:.1f} h)'
    if claim.get('state') == 'failed' and other_sweep:
        return f'failed in sweep {claim.get("sweep")}'
    if force and other_sweep:
        return f'--force over sweep {claim.get("sweep")}'
    return None


def claim_run(run_dir: str, sweep_id: str, worker: str, stale_seconds: int,
              force: bool) -> tuple[bool, str]:
    """Take exclusive ownership of a run directory, atomically.

    ``O_CREAT|O_EXCL`` is the whole mechanism: the filesystem decides, so no
    lock file, no registry of live workers and no scheduler cleverness is needed,
    and the guarantee holds across independent invocations on the same tree --
    including a second sweep started by hand while the first is still running.

    Returns ``(acquired, note)``; ``note`` is 'claimed', a reclaim reason, or
    'held by <pid>@<host>'.
    """
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, CLAIM_NAME)
    payload = json.dumps({
        'pid': os.getpid(), 'host': socket.gethostname(),
        'time': time.strftime('%Y-%m-%dT%H:%M:%S'), 'epoch': time.time(),
        'sweep': sweep_id, 'worker': worker, 'run_dir': run_dir,
        'state': 'running',
    }, sort_keys=True).encode('utf-8')

    note = 'claimed'
    for attempt in (0, 1):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if attempt:
                return False, note
            reason = _reclaim_reason(path, stale_seconds, sweep_id, force)
            if reason is None:
                held = read_claim(path) or {}
                return False, (f'held by {held.get("pid")}@{held.get("host")} '
                               f'(sweep {held.get("sweep")})')
            try:
                # Rename rather than unlink: the rename is atomic, so if two
                # workers judge the same claim stale only one of them can move
                # it out of the way and the loser sees ENOENT.
                os.replace(path, f'{path}.superseded-{int(time.time())}-'
                                 f'{os.getpid()}')
            except OSError:
                return False, 'lost the reclaim race'
            note = f'reclaimed: {reason}'
            continue
        try:
            os.write(fd, payload)
        finally:
            os.close(fd)
        return True, note
    return False, note


def claim_failed_here(run_dir: str, sweep_id: str) -> bool:
    """Did this sweep already fail this run?

    Needed so that a *second* worker reaching a job another worker has already
    failed drops it silently instead of announcing it as blocked. Without this
    the later, weaker record ('held by pid 26500') overwrote the earlier, real
    one ('failed: refusing to resume ...') and the summary reported a failure as
    a blockage, with no log path to read.
    """
    claim = read_claim(os.path.join(run_dir, CLAIM_NAME)) or {}
    return claim.get('state') == 'failed' and claim.get('sweep') == sweep_id


def release_claim(run_dir: str) -> None:
    """Drop the claim on success. The manifest is now the durable evidence."""
    try:
        os.remove(os.path.join(run_dir, CLAIM_NAME))
    except OSError:
        pass


def fail_claim(run_dir: str, error: str) -> None:
    """Leave the claim in place, marked failed, so nothing silently re-enters."""
    path = os.path.join(run_dir, CLAIM_NAME)
    claim = read_claim(path) or {}
    claim.update(state='failed', error=error[:2000],
                 failed_at=time.strftime('%Y-%m-%dT%H:%M:%S'))
    try:
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(claim, fh, sort_keys=True)
    except OSError:
        pass


def append_status(path: str, record: dict) -> None:
    """Append one state transition.

    A single ``os.write`` of a single short line into a descriptor opened
    ``O_APPEND``: the append offset is taken by the kernel, so concurrent
    workers interleave records rather than overwriting each other. Every record
    is self-contained JSON, so a torn line -- which is possible, and not
    pretended otherwise -- is detectable by the reader instead of corrupting its
    neighbours.
    """
    line = (json.dumps(record, sort_keys=True, default=str) + '\n').encode('utf-8')
    for attempt in range(6):
        try:
            fd = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY)
            try:
                os.write(fd, line)
            finally:
                os.close(fd)
            return
        except OSError:
            time.sleep(0.05 * (attempt + 1))


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
def dependency_state(rec: dict, pending: set[str], sweep_id: str,
                     stale_seconds: int) -> tuple[str, str]:
    """Classify a job's dependency as ready / wait / blocked.

    'blocked' is reserved for the cases where waiting cannot help: the source
    failed in this sweep, or it is not in the job list at all so nobody will
    ever build it. Everything else waits, because the stale-claim rule makes
    waiting self-healing -- a dead worker's claim expires and the waiting worker
    takes the job over itself.
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
    if claim:
        if claim.get('state') == 'failed' and claim.get('sweep') == sweep_id:
            return 'blocked', f'source run failed in this sweep: {dep}'
        return 'wait', f'source run in progress: {dep}'
    if dep in pending:
        return 'wait', f'source run not started yet: {dep}'
    return 'blocked', (f'source run {dep} has no manifest and is not in this '
                       f'sweep\'s job list')


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
    """
    records = read_jobs(args.jobs_file)
    if not records:
        print('no jobs')
        return 0
    status_path = os.path.join(os.path.dirname(args.jobs_file), STATUS_FILE)
    worker = f'w{args.worker_id:02d}'
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
    all_dirs = {r['run_dir'] for r in records}
    deferral_reason: dict[str, str] = {}
    failures: list[str] = []
    blocked: list[str] = []
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

            state, why = dependency_state(rec, pending, args.sweep_id,
                                          args.stale_seconds)
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
                                       args.stale_seconds, args.force)
            if not acquired:
                if deferral_reason.get(run_dir) != note:
                    log('deferred', rec, reason=note)
                    deferral_reason[run_dir] = note
                deferred.append(rec)
                continue
            if note != 'claimed':
                log('reclaimed', rec, reason=note)
                print(f'{worker}: reclaimed {rec["job_id"]} -- {note}',
                      flush=True)
            log('claimed', rec, note=note)

            print(f'\n{worker}: === {rec["job_id"]} {rec["experiment"]}/'
                  f'{rec["arm"]} seed {rec["seed"]} ({rec["seed_block"]}) '
                  f'on {rec["env"]} ===', flush=True)
            t0 = time.time()
            try:
                manifest = run_one(rec['config'])
            except BaseException as exc:                   # noqa: BLE001
                traceback.print_exc()
                fail_claim(run_dir, f'{type(exc).__name__}: {exc}')
                log('failed', rec, error=f'{type(exc).__name__}: {exc}',
                    wall_time_s=round(time.time() - t0, 1))
                failures.append(rec['job_id'])
                print(f'{worker}: FAILED {rec["job_id"]}: '
                      f'{type(exc).__name__}: {exc}', flush=True)
                if isinstance(exc, KeyboardInterrupt):
                    return 130
            else:
                release_claim(run_dir)
                result = manifest.get('result') or {}
                log('done', rec, wall_time_s=round(time.time() - t0, 1),
                    episodes_completed=result.get('episodes_completed'),
                    final_score=result.get('final_score'),
                    auc_score=result.get('auc_score'))
                ran += 1
                print(f'{worker}: done {rec["job_id"]} in '
                      f'{time.time() - t0:.0f}s  final_score='
                      f'{result.get("final_score")}', flush=True)
            progressed = True

        remaining = deferred
        if not remaining:
            break
        if progressed:
            idle_seconds = 0.0
            continue
        if idle_seconds >= max_wait:
            for rec in remaining:
                why = (f'still waiting after {idle_seconds:.0f}s: '
                       f'{deferral_reason.get(rec["run_dir"], "unknown")}')
                log('blocked', rec, reason=why)
                blocked.append(rec['job_id'])
                print(f'{worker}: BLOCKED {rec["job_id"]} -- {why}', flush=True)
            break
        time.sleep(args.poll_seconds)
        idle_seconds += args.poll_seconds

    print(f'\n{worker}: ran {ran}, skipped {skipped}, failed '
          f'{len(failures)}, blocked {len(blocked)}', flush=True)
    return 1 if (failures or blocked) else 0


# ---------------------------------------------------------------------------
# Parent
# ---------------------------------------------------------------------------
def spawn_workers(args: argparse.Namespace, jobs_path: str,
                  n_jobs: int) -> list[tuple[subprocess.Popen, str, str]]:
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
        path = os.path.join(log_dir, f'w{i:02d}-{args.sweep_id}.log')
        fh = open(path, 'w', encoding='utf-8')
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                env=env, cwd=_ROOT)
        procs.append((proc, path, f'w{i:02d}'))
        print(f'  worker w{i:02d}  pid {proc.pid}  {threads} threads  -> {path}')
    return procs


def wait_for_workers(procs, status_path: str, sweep_id: str,
                     progress_seconds: int) -> dict[str, int]:
    """Block until every worker exits, printing a periodic tally."""
    codes: dict[str, int] = {}
    last = time.time()
    while len(codes) < len(procs):
        for proc, _path, name in procs:
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


def summarise(records: Sequence[dict], status_path: str, sweep_id: str,
              log_paths: dict[str, str]) -> int:
    """Print what happened to every job, and return the exit code.

    Disk is the authority: a job is complete if its manifest says so, whatever
    the status log claims. The status log supplies the *reasons* -- which worker,
    which error, which log to read. A pending job with no terminal record is
    reported as unfinished rather than assumed fine, because "the runner exited
    0 and an arm is missing runs" is the failure this file exists to prevent.
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
    for rec in records:
        # Severity, not recency. Several workers may report on one run, and the
        # last record is not the most informative one: a worker that merely
        # found the directory held must not overwrite the worker that recorded
        # the exception and the log to read it in.
        run_dir = rec['run_dir']
        if is_complete(rec):
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
    print(f'  not finished       {len(unfinished):4d}')

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
    if blocked:
        print('\nBlocked -- reported, never skipped silently (DESIGN.md §8.4):')
        for rec, ev in blocked:
            print(f'  {rec["job_id"]}  {rec["experiment"]}/{rec["arm"]} '
                  f'seed {rec["seed"]}: {ev.get("reason")}')
    if unfinished:
        print('\nNot finished -- no terminal record and no complete manifest. '
              'Re-run with --resume:')
        for rec, state in unfinished[:20]:
            print(f'  {rec["job_id"]}  {rec["experiment"]}/{rec["arm"]} '
                  f'seed {rec["seed"]}  (last state: {state})')
        if len(unfinished) > 20:
            print(f'  ... and {len(unfinished) - 20} more')

    ok = not (failed or blocked or unfinished)
    print('\n' + ('all jobs complete. Next: experiments/aggregate.py'
                  if ok else 'sweep incomplete -- see above.'))
    print('=' * 72)
    return 0 if ok else 1


def print_plan(records: Sequence[dict], exp_ids: Sequence[str],
               membership: dict[str, list[str]], seeds, n_jobs: int,
               verbose: bool) -> None:
    """The plan, before anything is written or launched (`DESIGN.md` §8.5)."""
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
    # not an observation of it.
    target_seeds = sorted({r['seed'] for r in records if r['role'] != 'source'})
    if len(target_seeds) < 3:
        print(f'\n[NOTE] {len(target_seeds)} seed(s) per arm. Under '
              f'ANALYSIS_PLAN.md §9 nothing produced at n<3 is a result: '
              f'stats.py emits no test and no interval, and report.py stamps '
              f'every page PIPELINE VALIDATION - NOT A RESULT.')

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
                        'manifest, and reclaim claims held by other sweeps. '
                        'This does NOT retrain: train.py resumes from the '
                        'checkpoint, so a complete run re-enters at its last '
                        'episode and trains nothing. Deleting a run directory '
                        'is the only way to retrain it, and this runner will '
                        'not delete data')
    p.add_argument('--verbose', action='store_true',
                   help='list every job in the plan')
    p.add_argument('--stale-seconds', type=int, default=DEFAULT_STALE_SECONDS,
                   help='a claim older than this, with no manifest, may be '
                        'reclaimed (default 7200)')
    p.add_argument('--poll-seconds', type=int, default=DEFAULT_POLL_SECONDS,
                   help='worker wait between passes when only deferred jobs '
                        'remain')
    p.add_argument('--max-wait-seconds', type=int, default=0,
                   help='give up waiting on a dependency after this long and '
                        'report it blocked; 0 means stale-seconds + 300')
    p.add_argument('--progress-seconds', type=int, default=60,
                   help='parent progress tally cadence; 0 to disable')
    p.add_argument('--allow-factor-overrides', action='store_true',
                   help='permit --override on a field the registry classes as '
                        'an experimental factor rather than a budget setting. '
                        'The runs then carry a note saying so and their '
                        'configuration digests differ, so they cannot be '
                        'mistaken for catalogue runs')
    p.add_argument('--allow-seed-block-override', action='store_true',
                   help='proceed despite a seed-block violation. Stamped into '
                        'the invocation record; audit.py will still refuse the '
                        'affected estimates')
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
    seeds = resolve_seed_spec(args.seeds)
    overrides = parse_overrides(args.override)

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
              f'catalogue arms they are labelled as; their digests differ and '
              f'their manifests carry a note recording it.')

    records, membership = build_plan(exp_ids, seeds, args.out_root, overrides,
                                     args.allow_factor_overrides)
    print(f'sweep {args.sweep_id}')
    print_plan(records, exp_ids, membership, seeds, args.jobs, args.verbose)

    if args.dry_run:
        print('\n--dry-run: nothing written, nothing launched.')
        return 0

    jobs_dir = os.path.join(args.out_root, JOBS_SUBDIR)
    os.makedirs(jobs_dir, exist_ok=True)
    jobs_path = os.path.join(jobs_dir, JOBS_FILE)
    status_path = os.path.join(jobs_dir, STATUS_FILE)
    write_jobs(jobs_path, records)
    index_paths = write_index(args.out_root, membership, records)
    print(f'\nwrote {jobs_path} ({len(records)} jobs)')
    for path in index_paths:
        print(f'wrote {path}')

    if args.force:
        print('\n[--force] complete manifests will be re-entered. train.py '
              'resumes from the checkpoint, so a finished run trains nothing; '
              'delete its directory if you mean to retrain it.')

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
        'stale_seconds': args.stale_seconds,
        'seed_block_override': bool(args.allow_seed_block_override and fatal),
        'plan_hashes': _plan_hashes(),
    }
    with open(os.path.join(jobs_dir, f'sweep-{args.sweep_id}.json'), 'w',
              encoding='utf-8') as fh:
        json.dump(invocation, fh, indent=2, sort_keys=True, default=str)
    append_status(status_path, {**{k: invocation[k] for k in
                                  ('sweep', 'argv', 'experiments', 'seeds',
                                   'jobs_total', 'jobs_pending', 'workers')},
                                'state': 'sweep_start',
                                'ts': round(time.time(), 3),
                                't': invocation['t']})
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

    if not pending:
        print('\nNothing to run: every job already has a complete manifest.')
        return summarise(records, status_path, args.sweep_id, {})

    print(f'\nlaunching {n_jobs} worker(s) for {len(pending)} pending job(s)')
    procs = spawn_workers(args, jobs_path, n_jobs)
    log_paths = {name: path for _proc, path, name in procs}
    print('\nWaiting. Tail any worker log to follow progress.', flush=True)
    started = time.time()
    codes = wait_for_workers(procs, status_path, args.sweep_id,
                            args.progress_seconds)
    for _proc, path, name in procs:
        print(f'  {name}: exit {codes.get(name)}  {path}')
    print(f'\nworkers finished in {(time.time() - started) / 60:.1f} min')

    code = summarise(records, status_path, args.sweep_id, log_paths)
    append_status(status_path, {'sweep': args.sweep_id, 'state': 'sweep_end',
                                'ts': round(time.time(), 3),
                                't': time.strftime('%Y-%m-%dT%H:%M:%S'),
                                'exit_code': code,
                                'worker_exit_codes': codes})
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
