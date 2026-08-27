"""The machine-checked invariant checker: what turns control claims into facts.

    python experiments/audit.py --out-root runs
    python experiments/audit.py --out-root runs --experiments E1 E2
    python experiments/audit.py --out-root runs --json runs/audit.json
    python experiments/audit.py --out-root runs_demo --experiments E1
        --seeds 0-2 --overrides num_episodes=14 freeze_updates=150

Exit status is 0 only when every check passes. `report.py` gates on
`audit_ok(out_root, experiments)`, and `DESIGN.md` §8.4 requires that gate: an
override is permitted but must be stamped into the output.

Severity has exactly three levels, and the distinction is load-bearing.
`error` fails the audit and blocks reporting. `warning` is a fact the reader
must be told but which does not make the runs wrong -- a reduced-budget
validation launch, a dirty git tree, an intensity-confounded cross-architecture
pair -- and `--strict` promotes every warning to an error, which is the setting
a confirmatory campaign should be audited under. `note` is inventory the report
needs: which sources the validity gate excluded, which experiments are below
n=3 and therefore pipeline validation rather than result.

Why this file exists
--------------------
Every check below is the mechanical form of a specific defect. None of them is
hypothetical; each was found, by execution, either in the published study
(`paper/METHODS_ACTUAL.md`) or in the adversarial review of design revision 1
(`DESIGN.md` §11).

* **Invariants.** The published transfer arm ran at `lr=1e-4` against a
  baseline's `5e-4` under a printed claim of identical hyperparameters, because
  the four arms lived in four copied packages that had drifted. "Identical
  hyperparameters" is therefore not an assertion here: for each experiment,
  every field in `Experiment.invariants()` is required to take one value across
  every run the experiment contains, and the offending values and directories
  are printed when it does not (`DESIGN.md` §3.3, §8.4). At a second, narrower
  scope as well. `DESIGN.md` §3.3 declares two tuning policies, and under the
  secondary one a field such as `lr` varies *between* cells while remaining
  invariant *within* a cell. Adding it to an experiment's `varies` to permit the
  first would drop it from `invariants()` and stop it being checked at the
  second, so every field an experiment declares in `scoped_invariants`, and
  every `registry.CORE_INVARIANTS` field it declares varied, is additionally
  required to be constant within each group of runs sharing
  `registry.scope_key` -- (arch, target_rule, env), which is the cell *on one
  environment*. The environment belongs in the key: `E1t` and `E2t` carry each
  cell's selected learning rate on the target task and the common one on the
  CartPole source runs they share with the primary policy, so a scope keyed on
  the cell alone would report the structure the design asks for as a violation.
  A screen is exempt, because varying a hyperparameter inside one cell is what
  a screen is for.

* **Seed completeness.** One seed was dropped from one published arm with no
  stated rule. The declared arm x seed inventory comes from
  `registry.jobs()` -- the same function the runner uses -- so a missing run is
  a missing row here rather than a silently smaller n (`DESIGN.md` §9,
  `ANALYSIS_PLAN.md` §8). A seed *set* that resolves to nothing is refused
  before any of that, because an empty target inventory makes every
  completeness assertion vacuously true: `--seeds ""` used to turn nineteen
  errors into a silent pass on an unchanged tree, and print `seeds : the block
  each experiment declares` while doing it.

* **Seed blocks, in both directions.** Revision 1 selected hyperparameters on
  seeds 0-4 and then ran every confirmatory arm on 0-9, so half of each
  confirmatory sample had been tuned on. Checking only that a reported
  experiment avoids `TUNE` catches one half of that: it says nothing about a
  *selection* experiment that has wandered out of its own block and onto the
  seeds the confirmatory arms are estimated from, which is the same leak
  approached from the other side and is what `--seeds 0` does to E3. So the
  whole of the `DESIGN.md` §3.4 block table is enforced here, row by row: a
  reported experiment on `TUNE`, a selection experiment outside `TUNE`, a
  target-side arm on `C4SRC` or `RESERVE` (both of which are source-side blocks
  by declaration), a `SMOKE` seed anywhere but E0, and a seed belonging to no
  block at all. Sharpest of all, and the one that actually fires: a single run
  directory attributed both to a selection experiment and to a reported one is
  the leak itself rather than a proxy for it, and it is an error whenever the
  run was launched as part of the selection.

* **Config/digest consistency and run-directory uniqueness.** These two are the
  fabrication mode the adversarial review demonstrated: the old directory scheme
  omitted `freeze_*`, `transfer_*`, `lr`, `hidden`, the environment variant and
  the control condition, so nine distinct conditions from six catalogue
  experiments collided onto one path -- and because a completed directory was
  silently *resumed* rather than refused, one run's metrics were served under
  five other runs' manifests with every invariant check passing. So: the digest
  is recomputed from the stored config and compared to the stored identity; the
  path is required to encode that digest and that seed; and every trajectory
  digest written anywhere inside a run directory (the manifest, `state.json`,
  each `ckpt_ep*/state.json`) is required to be the same one.

* **Metrics integrity.** The published trainer appended to its `metrics.jsonl`
  on resume without truncating, so a crash between checkpoints duplicated
  episodes and corrupted every window statistic downstream. The episode index
  set must be exactly `range(0, episodes_completed)` (`DESIGN.md` §8.2). Two
  further things are read from that file, because an index check alone is
  content-blind. First, the outcome column: a run whose `score` is null on
  every episode, or constant across every episode, trained without recording
  anything and would otherwise reach `aggregate.py` with the audit's blessing.
  Second, a hash of the file itself, required distinct between run
  directories -- the digest scheme catches the route to "one run's metrics
  served under five manifests" that goes through a config collision, and this
  catches the route that goes through a file copy.

* **Freeze verification.** The manuscript described a freeze schedule the code
  never implemented. Freezing is now indexed in gradient updates and checked by
  weight fingerprints at each transition, in **both** directions: a
  declared-frozen layer that moved, and a trainable layer that did not, are both
  defects -- the second is what a positionally resolved freeze produces silently
  (`DESIGN.md` §3.2). Note that `transfer.verify_freeze`'s own `ok` flag covers
  only the first direction, so this check is deliberately stricter than that
  flag.

* **Source validity, the reserve rule, and lineage.** A published source agent
  scored 26.94 on a task solved at 475, and the transfer arm that consumed it is
  now unidentifiable: the only surviving CartPole checkpoint is from the wrong
  architecture. So every transfer run must carry a validity verdict on the
  normalised gate of `DESIGN.md` §4.3 (a *missing* verdict is an error; an
  *invalid* verdict is a reported exclusion, not an error), and its recorded
  source must resolve to a real run whose digest, environment and cell are the
  ones the config names. The verdict is *recomputed* rather than believed: the
  recorded gate is compared with `registry.SOURCE_VALIDITY_GATE` and the
  recorded `valid` flag with what the recorded score implies, because a check
  that reads a self-declared boolean is passed by the very edit it exists to
  catch. And §4.3's second half is checked too: a rejected source must be
  *replaced* from `RESERVE` until the arm has its full complement of valid
  sources, so a declared arm x seed slot occupied by a run whose source failed
  the gate, with no replacement in the ledger, is an arm that is complete to the
  seed count and empty in the primary estimand.

* **Transferred-parameter fraction.** Revision 1 held the layer list fixed
  across architectures and called that the same protocol; it transferred 97 % of
  the mlp and 51 % of the dueling network, confounding `arch` with treatment
  intensity by a factor of two -- the published study's own error, reconstituted
  inside the corrected design. `DESIGN.md` §3.1 refuses such a contrast "unless
  it is explicitly labelled intensity-confounded", and the catalogue is where
  that label lives: a group whose `transfer_set` is not the matched protocol is
  the deliberately unmatched comparison and carries the label as a warning,
  while a group that *claims* matched intensity and does not have it is refused
  outright as an error. A fraction that varies between the seeds of one arm is
  an error too, because the intensity is fixed by the configuration and
  averaging over a disagreement lets one bad seed rewrite the headline number.
  The tolerance itself is written down twice, here and as
  `stats.INTENSITY_TOLERANCE` in the module that decides whether the analysis
  draws the same contrast, and nothing compared the two: widening this copy
  tenfold left every guard in `validate.py` green while `stats.py` went on
  refusing at 0.05. So the two are compared at audit time, and a disagreement
  is an error, because two copies of one pre-registered constant returning
  opposite verdicts on one group is not a difference of opinion.

* **Plan hash, provenance, reference coverage.** A confirmatory result is
  interpretable only against the pre-registered plan in force when it ran, so
  every run's `ANALYSIS_PLAN.md` hash must agree with every other run's and with
  the current file (`ANALYSIS_PLAN.md` §1). A result from a dirty tree is not
  reproducible from the repository, so the count is reported. And a missing
  reference return would silently put one variant's scores on a different scale
  from every other's, which is what made the published cross-variant comparisons
  meaningless (`DESIGN.md` §5.1). Provenance is checked for the whole of
  `DESIGN.md` §8.3 and not only the git flags: package versions, machine, argv
  and the derived seed per stream are each required to be present, because a
  provenance block that is half absent is not a reproducibility record.

* **The tuning selection.** The tuned arms of `DESIGN.md` 3.3 are a function of
  one stored artifact, `<out_root>/_jobs/tuning_selection.json`, so that
  artifact is a pre-registration record with the standing of the plan hash and
  fails the same two ways: edited after the runs were enumerated from it, or
  replaced, which leaves the old runs on disk claimed by no arm while the new
  arms read as missing. A run recorded under `E1t`/`E2t` is therefore compared
  field by field against the configuration the artifact selects for its cell,
  and where it disagrees the archived selections are searched for the one it
  does match, so the finding names the selection the run actually came from. A
  placeholder rule, a selection outside the `TUNE` block, a selection covering
  fewer than four cells, a missing or non-matching archive copy and a stale plan
  hash on the artifact are each findings in their own right.

* **Robustness of the audit itself.** A checker that dies on one malformed field
  produces no report for any of the other runs, which is worse than the defect
  it choked on: a single non-integer `episodes_completed`, or `freeze_events`
  written as an object rather than an array, used to end the process with a
  traceback. Every value read out of a manifest is therefore coerced through a
  guard that turns a malformed field into a finding about that run. In the same
  spirit, a run directory holding run output but no manifest is discovered and
  reported rather than being invisible to a glob for `manifest.json` -- a crash
  before the manifest is written is the commonest way a run goes missing, and
  `DESIGN.md` §9 forbids dropping one silently.

How a run is attributed to an experiment
---------------------------------------
Not by resemblance. A run records the `experiment` and `label` the runner was
given, both of which came from the registry, and that is the primary key.
Because identical configurations are deliberately shared between experiments,
the registry is then asked which `(experiment, arm, seed)` triples resolve to
the same `run_digest` -- the equivalence `registry.all_jobs` de-duplicates on --
and the run is attributed to every member of its class. So auditing E2 alone
still recognises that its scratch denominators are runs E1 launched, and no
heuristic has to guess at it.

The two routes are then cross-examined against each other: the stored config is
compared, field by field, with the config the registry declares for that arm.
`registry.SCALING_FIELDS` decides how a difference is graded. A difference in a
budget or measurement field means the runs are the declared arms at a reduced
setting -- a validation launch under `STANDING_INSTRUCTIONS` S8, which is worth
auditing and is not a result -- and is a warning. A difference in a *factor* is
an error, because a run whose factors differ is not the arm its label names;
`registry.jobs` refuses such an override unless asked and stamps a note in the
manifest when it is, so a deliberate one is recognised and downgraded. Passing
`--overrides` reconstructs the configuration a launch actually used, which is
how a scaled launch is audited against what it meant rather than against what
it is not.

The audit computes no statistic and emits no p-value. It is a precondition for
inference, not inference; the ledger printed at the end says so explicitly.
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _path in (_REPO, _HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import registry                                              # noqa: E402
import statlib                                               # noqa: E402
import tuning                                                # noqa: E402
from src.dqn import envs, provenance                         # noqa: E402
from src.dqn.config import (Config, MEASUREMENT_FIELDS,      # noqa: E402
                            TRAJECTORY_FIELDS)

# Severities. `error` fails the audit; `warning` is reported and, under
# --strict, promoted to an error; `note` is inventory the reader needs and never
# affects the exit status.
ERROR, WARN, NOTE = 'error', 'warning', 'note'

# DESIGN.md 3.1: the declared tolerance on transferred-parameter fraction for a
# cross-architecture contrast. The design declares one tolerance and names this
# file as the module that refuses on it, so this is where it lives and
# `stats.py`'s `INTENSITY_TOLERANCE` is the second copy of it. Nothing compared
# the two: `validate.py` pins six statlib/stats constant pairs and this pair is
# not among them, so widening this copy tenfold left every guard green while
# `stats.py` went on refusing the same contrast from its own 0.05, and the two
# modules would have returned opposite verdicts on one question. They are
# therefore compared at audit time, in `check_transferred_fraction`, which is
# what `GATE_TOLERANCE` below already does for the source-validity gate: a
# disagreement between two copies of one pre-registered constant becomes a
# finding in the report rather than staying a fact about the source.
FRACTION_TOLERANCE = 0.05
# A recorded normalisation constant may drift only by float round-trip.
REFERENCE_TOLERANCE = 1e-6
# The same slack, applied to the recorded source-validity gate. registry.py's
# own comment says the two copies of the constant "must agree" and that a
# disagreement should be "visible in the data"; this is where it becomes so.
GATE_TOLERANCE = 1e-9
# ANALYSIS_PLAN.md 9: below this, output is pipeline validation, not a result.
# Imported rather than restated. The plan declares one floor; this file used to
# carry a fourth independent copy of it, beside `statlib`'s, `stats.py`'s and
# the one `aggregate.py` already takes from `statlib`. `validate.py` pins the
# `statlib` copy and pinned nothing here, so setting this one to 1 dropped the
# pipeline-validation note from every audit report with the whole guard suite
# still green.
MIN_N_FOR_A_RESULT = statlib.MIN_N_FOR_INFERENCE
# How many run directories to name per finding before summarising the rest.
MAX_LISTED = 6

# --strict promotes a warning to an error, and every warning in this file is a
# deviation a campaign can clear, with one exception. `intensity_confounded`
# *is* the DESIGN.md 3.1 label: a group whose `transfer_set` is not the matched
# protocol is the deliberately unmatched comparison, and E4 and E5 declare four
# of them, so a structurally correct complete tree carries the warning by
# construction. Promoting a declared label to an error made --strict a mode no
# tree can pass, which is a mode nobody runs and therefore no gate at all. The
# warning is still emitted, still printed and still the thing standing between
# the reader and the confound; what it is not is something a campaign could
# ever be asked to fix. Anything added here has to meet the same test: not "it
# is inconvenient" but "the catalogue guarantees it on a correct tree".
DECLARED_BY_DESIGN = frozenset({'intensity_confounded'})

# The registry's own line between a budget setting a caller may scale and a
# factor that defines what the experiment is. Used, not re-derived, so the
# audit cannot disagree with the runner about which is which.
SCALING_FIELDS = frozenset(registry.SCALING_FIELDS)

CONFIG_DEFAULTS: dict[str, Any] = {f.name: f.default
                                   for f in dataclasses.fields(Config)}

# The metrics columns that carry an outcome. `score` is the normalised
# per-episode return of DESIGN.md 5.1 and every row has one; `eval_score` is
# recorded on evaluation episodes only, so its absence from a given row is not
# a defect while its absence from every row of a run is.
OUTCOME_FIELD = 'score'
EVAL_OUTCOME_FIELD = 'eval_score'

# Files that make a directory a run directory even when its manifest is gone.
# A run writes metrics from its first episode, so a directory holding these and
# no `manifest.json` is a run that died before it could describe itself.
RUN_OUTPUT_MARKERS = ('metrics.jsonl', 'state.json', 'model.keras')

# Directory names under the run root that hold bookkeeping rather than runs.
BOOKKEEPING_PREFIXES = ('_', '.')
# A checkpoint directory inside a run directory. It carries a `state.json` of
# its own, so it has to be told apart from the run that owns it.
CHECKPOINT_PREFIX = 'ckpt_ep'


# ---------------------------------------------------------------------------
# Coercion guards
# ---------------------------------------------------------------------------
# A manifest is data read off disk, not a trusted structure. Every one of these
# exists because an unguarded read of the same field ended the whole audit with
# a traceback and produced no report for any other run in the tree, which is a
# worse outcome than whatever the malformed field was going to be reported as.
# ---------------------------------------------------------------------------
def _as_int(value: Any) -> Optional[int]:
    """`int(value)` or None. Never raises, and never accepts a bool."""
    if value is None or isinstance(value, bool):
        return None
    try:
        out = int(value)
    except (TypeError, ValueError):
        return None
    return out


def _as_float(value: Any) -> Optional[float]:
    """`float(value)` or None, refusing non-finite values.

    A NaN is not a number that failed a gate, it is a measurement that did not
    happen, and `nan >= 0.6` being False is exactly how a degenerate evaluation
    reads as a rejection. So it is returned as absent and reported as such.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out or out in (float('inf'), float('-inf')):
        return None
    return out


def _events(value: Any) -> tuple[list[dict], int]:
    """The dict elements of a freeze-event list, and how many were not dicts.

    `freeze_events` written as a JSON object rather than an array made
    `enumerate` walk its keys, so the elements were strings and the audit died
    on `str.get`. A non-list, or a list with a non-dict in it, is a malformed
    manifest to be reported, not an exception to be raised.
    """
    if isinstance(value, dict):
        return [], len(value)
    if not isinstance(value, list):
        return [], (0 if value is None else 1)
    good = [e for e in value if isinstance(e, dict)]
    return good, len(value) - len(good)


def _mapping(value: Any) -> Optional[dict]:
    """`value` when it is a dict, else None. For nested manifest branches."""
    return value if isinstance(value, dict) else None


# ---------------------------------------------------------------------------
# Findings and checks
# ---------------------------------------------------------------------------
@dataclass
class Finding:
    level: str
    code: str
    message: str
    runs: list[str] = field(default_factory=list)
    detail: dict = field(default_factory=dict)


@dataclass
class Check:
    name: str
    why: str                                  # the defect this check prevents
    findings: list[Finding] = field(default_factory=list)
    detail: dict = field(default_factory=dict)

    def add(self, level: str, code: str, message: str,
            runs: Iterable[str] = (), **detail) -> None:
        self.findings.append(Finding(level, code, message, list(runs), detail))

    def count(self, level: str) -> int:
        return sum(1 for f in self.findings if f.level == level)

    def promotable(self) -> int:
        """The warnings --strict turns into errors: deviations, not labels.

        See `DECLARED_BY_DESIGN`. A warning the catalogue guarantees on a
        correct tree is reported at every setting and promoted at none.
        """
        return sum(1 for f in self.findings
                   if f.level == WARN and f.code not in DECLARED_BY_DESIGN)

    def status(self, strict: bool = False) -> str:
        if self.count(ERROR) or (strict and self.promotable()):
            return 'FAIL'
        if self.count(WARN):
            return 'WARN'
        return 'PASS'


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------
@dataclass
class Run:
    """One completed run directory, as read from disk."""

    path: str                      # absolute, normalised
    rel: str                       # relative to out_root, forward slashes
    manifest: dict

    @property
    def cfg(self) -> dict:
        return _mapping(self.manifest.get('config')) or {}

    @property
    def identity(self) -> dict:
        return _mapping(self.manifest.get('identity')) or {}

    @property
    def seed(self) -> Optional[int]:
        # Coerced, not cast: a manifest whose seed is a string is a run with a
        # broken identity, which `check_digests` reports. It must not be a
        # ValueError raised out of a property every other check calls.
        return _as_int(self.cfg.get('seed'))

    @property
    def condition(self) -> str:
        return str(self.cfg.get('condition', ''))

    @property
    def cell(self) -> str:
        return f"{self.cfg.get('arch')}-{self.cfg.get('target_rule')}"

    @property
    def run_digest(self) -> Optional[str]:
        return self.identity.get('run_digest')

    def get(self, *keys: str, default: Any = None) -> Any:
        """Nested manifest lookup that tolerates absent or null branches."""
        node: Any = self.manifest
        for key in keys:
            if not isinstance(node, dict) or node.get(key) is None:
                return default
            node = node[key]
        return node


def unmanifested_run_dirs(out_root: str) -> list[str]:
    """Directories holding run output but no `manifest.json`.

    A glob for `manifest.json` cannot see a run whose manifest was never
    written, and a crash before the manifest is the commonest way that happens:
    the directory still holds the metrics, the checkpoints and the weights. The
    docstring on `discover_runs` promises that a run whose identity cannot be
    read is an error rather than a skip, and a *missing* identity is the same
    problem as an unreadable one, so the tree is walked independently for the
    output markers a run leaves behind.

    A checkpoint directory is not reported in its own right. The walk stops at
    any directory holding a manifest, so a healthy run's `ckpt_ep*` children are
    never reached; and where the manifest is the thing that is missing, a
    `ckpt_ep*` carrying its own `state.json` is attributed to the run directory
    above it rather than counted as a run of its own.
    """
    root = os.path.abspath(out_root)
    if not os.path.isdir(root):
        return []
    out: set[str] = set()
    for entry in sorted(os.listdir(root)):
        if entry.startswith(BOOKKEEPING_PREFIXES):
            continue                    # _jobs, _index, _logs: not runs
        top = os.path.join(root, entry)
        if not os.path.isdir(top):
            continue
        for current, dirs, files in os.walk(top):
            names = set(files)
            if 'manifest.json' in names:
                dirs[:] = []            # a run directory holds no run directory
                continue
            if not names.intersection(RUN_OUTPUT_MARKERS):
                continue
            found = current
            if os.path.basename(current).startswith(CHECKPOINT_PREFIX) \
                    and os.path.dirname(current) != root:
                found = os.path.dirname(current)
            out.add(os.path.relpath(found, root).replace(os.sep, '/'))
            dirs[:] = []
    return sorted(out)


def discover_runs(out_root: str) -> tuple[list[Run], list[Finding]]:
    """Every run directory under `out_root`, with unreadable manifests reported.

    An unreadable manifest is an error rather than a skip: a run whose identity
    cannot be read cannot be excluded from an analysis that globs the tree, so
    silently ignoring it here would leave it to be picked up downstream. A
    manifest that is *absent* is the same defect and is reported the same way,
    which a glob for `manifest.json` cannot do on its own.
    """
    root = os.path.abspath(out_root)
    findings: list[Finding] = []
    runs: list[Run] = []
    bookkeeping: list[str] = []
    pattern = os.path.join(glob.escape(root), '**', 'manifest.json')
    for path in sorted(glob.glob(pattern, recursive=True)):
        run_dir = os.path.dirname(os.path.abspath(path))
        rel = os.path.relpath(run_dir, root).replace(os.sep, '/')
        head = rel.split('/', 1)[0]
        if head not in ('.', '') and head.startswith(BOOKKEEPING_PREFIXES):
            # `unmanifested_run_dirs` skips `_jobs`, `_index` and `_logs` by
            # this same rule, and the two functions are documented as covering
            # one population from two directions. While the glob did not, an
            # archived copy of a manifest under `_index/` became a Run and
            # entered every per-run check, attribution and the metrics-hash
            # set, while an unmanifested directory beside it stayed invisible.
            # Skipped rather than silently dropped: the count is reported
            # below, so a manifest in the wrong place is still a fact.
            bookkeeping.append(rel)
            continue
        try:
            with open(path, encoding='utf-8') as fh:
                manifest = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            findings.append(Finding(ERROR, 'manifest_unreadable',
                                    f'{rel}: {exc}', [rel]))
            continue
        if not isinstance(manifest, dict) or not isinstance(
                manifest.get('config'), dict):
            findings.append(Finding(
                ERROR, 'manifest_malformed',
                f'{rel}: manifest has no config block', [rel]))
            continue
        runs.append(Run(run_dir, rel, manifest))
    absent = unmanifested_run_dirs(out_root)
    if absent:
        findings.append(Finding(
            ERROR, 'manifest_absent',
            f'{len(absent)} director(ies) hold run output '
            f'({", ".join(RUN_OUTPUT_MARKERS)}) but no manifest.json, so they '
            f'are in no count, no attribution and no analysis. A run that '
            f'cannot say what it is cannot be reported and cannot be excluded '
            f'by a glob either; DESIGN.md 9 forbids dropping one silently',
            absent[:MAX_LISTED], {'n': len(absent), 'dirs': absent[:24]}))
    if bookkeeping:
        findings.append(Finding(
            NOTE, 'manifest_under_bookkeeping_dir',
            f'{len(bookkeeping)} manifest.json file(s) sit under a '
            f'bookkeeping directory ({", ".join(BOOKKEEPING_PREFIXES)}...) and '
            f'are not runs: {", ".join(bookkeeping[:MAX_LISTED])}. They are '
            f'excluded from attribution and from every per-run check, on the '
            f'same rule unmanifested_run_dirs applies',
            bookkeeping[:MAX_LISTED],
            {'n': len(bookkeeping), 'dirs': bookkeeping[:24]}))
    return runs, findings


# ---------------------------------------------------------------------------
# Attribution: which experiments does a run belong to?
# ---------------------------------------------------------------------------
# A run is attributed to an experiment by *identity*, not by resemblance. Two
# routes are combined, and the disagreement between them is itself a check:
#
# 1. The manifest records the `experiment` and `label` the runner was given,
#    which came from the registry's own `Arm.label`. That is provenance, not a
#    guess, so it is the primary key.
# 2. Identical configurations are deliberately shared between experiments --
#    E4's freeze level that equals E1's protocol value, E2's and E8's scratch
#    denominators, E5's matched layer set, E8i's donor arm -- and only the
#    experiment that launched a run first is named in its manifest. So the
#    registry is asked which (experiment, arm, seed) triples resolve to the same
#    `run_digest`, and the run is attributed to every member of its class. That
#    is the same equivalence `registry.all_jobs` de-duplicates on, computed from
#    the catalogue alone, so it does not depend on what any run recorded.
#
# The two routes are then cross-examined: the run's stored config is compared
# field by field against the config the registry declares for its own arm. A
# disagreement on a field the *arm* fixes means the label does not describe the
# run, and is an error. A disagreement elsewhere means the launch ran the
# declared arms at some other setting -- a reduced-budget validation launch
# (`STANDING_INSTRUCTIONS` S8) -- and is reported as a deviation rather than a
# defect, because such a launch is still the experiment it says it is and is
# still worth auditing.
# ---------------------------------------------------------------------------
def discriminating_fields() -> tuple[str, ...]:
    """The fields the catalogue uses to tell one arm from another.

    Used to group runs for the cross-architecture intensity comparison, where
    the question is which *observed* configurations are comparable rather than
    which arm a run belongs to.
    """
    keys: set[str] = set()
    for exp in registry.EXPERIMENTS.values():
        keys.update(exp.varies)
        for arm in exp.arms:
            keys.update(arm.overrides)
    # A path, never an identity: matching on it would mean that moving the run
    # tree changed which arm a run belongs to.
    keys.discard('source_checkpoint')
    return tuple(sorted(k for k in keys if k in CONFIG_DEFAULTS))


DISCRIMINATING = discriminating_fields()


def _norm(name: str, value: Any) -> Any:
    """Comparable form of a config value: canonical envs, tuples not lists."""
    if name in ('env', 'source_env') and isinstance(value, str) and value:
        try:
            return envs.parse(value).canonical()
        except Exception:                                   # noqa: BLE001
            return value
    if isinstance(value, (list, tuple)):
        return tuple(_norm(name, v) for v in value)
    return value


def _scope_label(key: Iterable[Any]) -> str:
    """A `registry.scope_key` tuple as prose, for a finding a human reads.

    `('mlp', 'vanilla', 'LunarLander-v3')` reads as
    `cell mlp-vanilla on LunarLander-v3`. The keys are named from
    `registry.SCOPED_INVARIANT_KEYS` rather than positionally, so adding one to
    the scope changes this label instead of silently dropping out of it.
    """
    parts = list(key)
    names = list(registry.SCOPED_INVARIANT_KEYS)
    by_name = dict(zip(names, parts))
    if set(names) >= {'arch', 'target_rule'}:
        cell = f'{by_name.get("arch")}-{by_name.get("target_rule")}'
        rest = [f'{n}={by_name[n]}' for n in names
                if n not in ('arch', 'target_rule')]
        return f'cell {cell}' + (' on ' + ', '.join(
            str(by_name[n]) for n in names
            if n not in ('arch', 'target_rule')) if rest else '')
    return ', '.join(f'{n}={by_name[n]}' for n in names)


def _run_signature(run: Run) -> dict[str, Any]:
    return {name: _norm(name, run.cfg.get(name, CONFIG_DEFAULTS.get(name)))
            for name in DISCRIMINATING}


def parse_overrides(items: Iterable[str] | None) -> dict[str, Any]:
    """`--overrides freeze_updates=150 num_episodes=14` into a dict."""
    out: dict[str, Any] = {}
    for item in items or ():
        if '=' not in item:
            raise ValueError(f'--overrides expects field=value, got {item!r}')
        key, raw = item.split('=', 1)
        key = key.strip()
        if key not in CONFIG_DEFAULTS:
            raise ValueError(f'--overrides: {key!r} is not a Config field')
        try:
            out[key] = json.loads(raw)
        except json.JSONDecodeError:
            out[key] = raw
    return out


ArmKey = tuple[str, str, int]                 # (experiment, arm label, seed)


@dataclass
class Declared:
    """The catalogue's own account of itself, resolved to concrete configs."""

    configs: dict[ArmKey, Any] = field(default_factory=dict)
    digests: dict[ArmKey, str] = field(default_factory=dict)
    # run_digest -> every (experiment, arm, seed) resolving to it. A class with
    # more than one member is a deliberately shared run.
    classes: dict[str, list[ArmKey]] = field(default_factory=dict)
    # experiment -> the (arm, seed) pairs it declares at the selected seeds
    target_pairs: dict[str, set[tuple[str, int]]] = field(default_factory=dict)
    arms: dict[tuple[str, str], Any] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)


def declare(seeds=None, observed_seeds: Iterable[int] = (),
            overrides: dict | None = None,
            out_root: str = 'runs') -> Declared:
    """Resolve the catalogue into concrete (experiment, arm, seed) configs.

    Built over every experiment, not only the selected ones, because the
    equivalence classes are what identify a shared run: auditing E2 alone still
    has to recognise that its scratch denominators are runs E1 launched.

    Seeds are resolved twice, on purpose. The *target* set -- the block each
    experiment declares, or `--seeds` -- is what seed completeness is measured
    against. Its union with the seeds actually present on disk is what
    attribution uses, so a run at an undeclared seed is recognised and reported
    rather than silently dropped.

    `out_root` is passed through to `registry.jobs`, and must be: `jobs` reads
    the DESIGN.md 4.3 reserve-replacement ledger from
    `<out_root>/_jobs/source_replacements.jsonl`, so taking its default meant
    that auditing any tree resolved that tree's declared inventory against
    `runs/`'s ledger. Once a reserve replacement fires, a substituted run is
    declared from the wrong tree or not declared at all, and the reserve rule
    becomes unauditable in exactly the campaign that needs it. The declared
    `source_checkpoint` path is wrong for the same reason.
    """
    out = Declared(classes=defaultdict(list))
    observed = sorted({s for s in (_as_int(s) for s in observed_seeds)
                       if s is not None})
    for eid, exp in registry.EXPERIMENTS.items():
        for arm in exp.arms:
            out.arms[(eid, arm.label)] = arm
        try:
            target = registry.resolve_seeds(seeds, exp.seed_block)
            # `allow_factor_overrides` is set because the auditor is
            # reading, not launching: it has to be able to reconstruct the
            # configuration a launch actually used in order to say whether
            # the runs match it. Whether that launch should have overridden
            # a factor is the invariants check's verdict, not this call's.
            target_jobs = registry.jobs(eid, seeds=list(target),
                                        out_root=out_root,
                                        overrides=overrides,
                                        allow_factor_overrides=True)
            union_jobs = registry.jobs(
                eid, seeds=sorted(set(target) | set(observed)),
                out_root=out_root,
                overrides=overrides, allow_factor_overrides=True)
        except Exception as exc:                            # noqa: BLE001
            out.findings.append(Finding(
                ERROR, 'catalogue_unresolvable',
                f'{eid}: the registry cannot resolve this experiment into jobs '
                f'({exc}), so no run can be audited against it'))
            continue
        out.target_pairs[eid] = {(job.arm, int(job.cfg.seed))
                                 for job in target_jobs}
        for job in union_jobs:
            key: ArmKey = (eid, job.arm, int(job.cfg.seed))
            out.configs[key] = job.cfg
            digest = job.cfg.run_digest()
            out.digests[key] = digest
            out.classes[digest].append(key)
    return out


def attribute(runs: list[Run], exp_ids: Iterable[str], declared: Declared):
    """membership[experiment][arm] -> runs, the orphans, and the whole map."""
    membership: dict[str, dict[str, list[Run]]] = {
        eid: {arm.label: [] for arm in registry.EXPERIMENTS[eid].arms}
        for eid in exp_ids}
    everywhere: dict[str, set[str]] = defaultdict(set)
    orphans: list[Run] = []
    for run in runs:
        key = (str(run.cfg.get('experiment')), str(run.cfg.get('label')),
               run.seed)
        digest = declared.digests.get(key)
        if digest is None:
            orphans.append(run)
            continue
        for eid, label, _seed in declared.classes.get(digest, ()):
            everywhere[run.rel].add(eid)
            if eid in membership:
                membership[eid][label].append(run)
    return membership, orphans, everywhere


def declared_coverage(membership, declared: Declared
                      ) -> dict[str, tuple[int, int]]:
    """Per experiment: runs attributed, and runs filling a *declared* slot.

    The single definition of "this run was measured against something". Every
    completeness, block and source-validity assertion in this file is made
    against `declared.target_pairs`; a run outside those pairs is attributed,
    listed and then compared with nothing. Counting the two separately is what
    makes an audit that verified nothing visible instead of green. Under
    `--seeds 999` on the real tree the first number is 44 and the second is 0
    for every experiment, and the audit rendered "AUDIT PASS".
    """
    out: dict[str, tuple[int, int]] = {}
    for eid, arms in membership.items():
        pairs = declared.target_pairs.get(eid) or set()
        attributed = covered = 0
        for label, group in arms.items():
            for run in group:
                attributed += 1
                if (label, run.seed) in pairs:
                    covered += 1
        out[eid] = (attributed, covered)
    return out


def launched_as_declared(run: Run, declared: Declared) -> bool:
    """True when the run sits in a slot the experiment it names declares.

    The run's own manifest records the `experiment` and `label` the runner was
    given. If the registry schedules that arm at that seed, the run is where
    its launcher said it would be, and any *other* experiment that also claims
    it does so only because the two configurations coincide.
    """
    eid = str(run.cfg.get('experiment'))
    label = str(run.cfg.get('label'))
    return (label, run.seed) in (declared.target_pairs.get(eid) or set())


def shared_configuration_only(eid: str, label: str, run: Run,
                              declared: Declared) -> bool:
    """True when `eid` claims this run only because two configs coincide.

    `declare()` resolves every experiment at the union of its declared seeds
    and the seeds present on disk, so that a run at an undeclared seed is
    recognised rather than silently dropped. The cost is that the union
    manufactures arm x seed keys no experiment schedules: `E8i`'s `c4src-*`
    donors at seed 300 are digest-identical to the `scratch-*` target arms of
    ten other experiments, so every one of those ten acquires a target-side
    membership the catalogue never declared. Three things have to hold before a
    membership is dismissed as that artefact, and all three are properties of
    the catalogue rather than of the check that wants the exemption:

    * the run's own manifest names some other experiment, so `eid` did not
      launch it;
    * `eid` declares no arm x seed slot this run could fill, so nothing `eid`
      reports can draw on it (`aggregate.py`'s `MembershipIndex` builds
      membership from `present & (declared | reserve)` and excludes it too);
    * the experiment that *did* launch it declares the slot it sits in, so the
      run is accounted for somewhere.

    A run failing any of the three keeps every verdict. In particular a run
    launched into a block it does not belong to is still judged, because the
    first condition fails, and a rogue run its own launcher does not declare is
    still judged, because the third does.
    """
    if str(run.cfg.get('experiment')) == eid:
        return False
    if (label, run.seed) in (declared.target_pairs.get(eid) or set()):
        return False
    return launched_as_declared(run, declared)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def check_invariants(membership, exps, declared: Declared) -> Check:
    chk = Check(
        'INVARIANTS',
        'the published transfer arm ran at lr=1e-4 against a baseline\'s 5e-4 '
        'under a claim of identical hyperparameters (DESIGN.md 3.3, 8.4)')
    audited = tuple(sorted(set(TRAJECTORY_FIELDS) | set(MEASUREMENT_FIELDS)))
    # Coverage of the second, per-cell scope below, counted rather than
    # assumed. See the note it feeds at the end of this function.
    cell_scope_experiments = 0
    cell_scope_comparisons = 0
    for eid, arms in membership.items():
        exp = exps[eid]
        runs = {r.rel: r for group in arms.values() for r in group}
        chk.detail.setdefault(eid, {}).update(
            {'runs': len(runs), 'varies': list(exp.varies),
             'invariants': list(exp.invariants())})
        if not runs:
            continue

        # The check the control claim rests on: every field the experiment
        # declares invariant takes exactly one value across its runs.
        for name in exp.invariants():
            groups: dict[Any, list[str]] = defaultdict(list)
            for rel, run in sorted(runs.items()):
                value = _norm(name,
                              run.cfg.get(name, CONFIG_DEFAULTS.get(name)))
                groups[value].append(rel)
            if len(groups) > 1:
                chk.add(ERROR, 'invariant_violated',
                        f'{eid}: {name} takes {len(groups)} values across the '
                        f'experiment\'s runs, but {eid} declares it invariant '
                        f'({", ".join(str(v) for v in groups)})',
                        runs=[r for rs in groups.values() for r in rs[:2]],
                        experiment=eid, field=name,
                        values={str(v): {'n': len(rs), 'runs': rs[:MAX_LISTED]}
                                for v, rs in groups.items()})

        # The second scope DESIGN.md 3.3 names. Under the secondary,
        # per-cell-tuned policy `lr` is invariant *within* a cell and varies
        # between them, so it would be added to `varies` -- which drops it from
        # `invariants()` above and would stop it being checked at either scope.
        # This scope closes that hole: a field the experiment declares varied
        # is still required to be constant within each cell. The remaining
        # CORE_INVARIANTS are skipped here because the loop above already
        # asserts them across the whole experiment, which implies them within
        # every cell; skipping them is not a gap, it is the stronger assertion
        # already made. A screen is exempt because varying a hyperparameter
        # inside one cell is the whole point of a screen: E3 sweeps four
        # learning rates per cell and E12 two capacities, both by declaration.
        #
        # The group is `registry.scope_key`, which is (arch, target_rule, env)
        # and not the cell alone. The environment belongs in the key because
        # DESIGN.md 3.3 scopes the secondary policy to "within a cell across
        # {scratch, transfer, C2, C3}", the four TARGET-task conditions, and the
        # selection is made on LunarLander. E1t and E2t therefore carry the
        # cell's selected `lr` on the target runs and the common `lr=5e-4` on
        # the CartPole source runs they share with the common policy, which is a
        # single cell holding two learning rates. Grouped by cell alone this
        # check would report every retuned cell as a violation -- an ERROR on
        # the structure the design asks for -- so the key is taken from
        # `registry.SCOPED_INVARIANT_KEYS` rather than restated here, and the
        # catalogue and the audit cannot drift about what the scope is.
        #
        # The field set is the experiment's `scoped_invariants` unioned with
        # every CORE_INVARIANT it declares varied. The union is deliberate: the
        # declaration is the catalogue's statement of intent and is checked
        # because it was made, and the derived set is checked because an
        # experiment that varies a core field and declares no scope would
        # otherwise be asserted at neither scope, which is the hole this block
        # exists to close.
        if exp.family != 'screen':
            by_scope: dict[tuple, list[str]] = defaultdict(list)
            for rel, run in sorted(runs.items()):
                by_scope[registry.scope_key(run.cfg)].append(rel)
            declared_invariant = set(exp.invariants())
            varied_core = [f for f in registry.CORE_INVARIANTS
                           if f not in declared_invariant]
            scoped = list(dict.fromkeys(
                tuple(exp.scoped_invariants) + tuple(varied_core)))
            cell_scope_experiments += 1
            cell_scope_comparisons += len(scoped) * len(by_scope)
            for key in sorted(by_scope, key=lambda k: tuple(str(p) for p in k)):
                where = _scope_label(key)
                for name in scoped:
                    groups = defaultdict(list)
                    for rel in by_scope[key]:
                        groups[_norm(name, runs[rel].cfg.get(
                            name, CONFIG_DEFAULTS.get(name)))].append(rel)
                    if len(groups) > 1:
                        chk.add(ERROR, 'cell_invariant_violated',
                                f'{eid}: {name} takes {len(groups)} values '
                                f'within {where}. {eid} declares {name} '
                                f'varied, which DESIGN.md 3.3 licenses only '
                                f'*between* cells: under the per-cell-tuned '
                                f'policy it is still invariant within one '
                                f'({", ".join(str(v) for v in groups)})',
                                runs=[r for rs in groups.values()
                                      for r in rs[:2]],
                                experiment=eid, field=name, cell=where,
                                scope_keys=list(registry.SCOPED_INVARIANT_KEYS),
                                values={str(v): {'n': len(rs),
                                                 'runs': rs[:MAX_LISTED]}
                                        for v, rs in groups.items()})
            chk.detail[eid]['cells'] = sorted(
                _scope_label(k) for k in by_scope)
            chk.detail[eid]['cell_scope_fields'] = scoped
            chk.detail[eid]['cell_scope_declared'] = list(exp.scoped_invariants)

        # And the second half of the same question: is the value in force the
        # value the catalogue declares? The registry draws the line for us.
        # `registry.SCALING_FIELDS` are the budget and measurement settings a
        # caller may scale without changing what the experiment is; everything
        # else is a factor, and `registry.jobs` refuses to override a factor
        # unless told to, stamping a note in the manifest when it does.
        #
        # So a deviation on a scaling field means the runs are the declared arms
        # at a reduced budget -- a validation launch (`STANDING_INSTRUCTIONS`
        # S8), which is worth auditing and is not a result. A deviation on a
        # factor means the run is not the arm its label names, which is an
        # error unless the run itself records that its factors were overridden.
        scaled: dict[str, dict] = {}
        factored: dict[str, dict] = {}
        for label, group in arms.items():
            for run in group:
                want_cfg = declared.configs.get((eid, label, run.seed))
                if want_cfg is None:
                    continue
                want = want_cfg.to_dict()
                stamped = '[factor overrides:' in str(run.cfg.get('notes') or '')
                for name in audited:
                    got = _norm(name, run.cfg.get(name,
                                                  CONFIG_DEFAULTS.get(name)))
                    expect = _norm(name, want.get(name))
                    if got == expect:
                        continue
                    bucket = scaled if name in SCALING_FIELDS else factored
                    entry = bucket.setdefault(
                        name, {'declared': set(), 'observed': set(),
                               'runs': [], 'stamped': True})
                    entry['declared'].add(str(expect))
                    entry['observed'].add(str(got))
                    entry['runs'].append(run.rel)
                    entry['stamped'] &= stamped
        for name, entry in sorted(factored.items()):
            level, code = ((WARN, 'declared_factor_override') if entry['stamped']
                           else (ERROR, 'factor_deviation'))
            chk.add(level, code,
                    f'{eid}: {len(entry["runs"])} run(s) deviate on {name}, '
                    f'which is an experimental factor and not a budget setting: '
                    f'declared {sorted(entry["declared"])}, found '
                    f'{sorted(entry["observed"])}. '
                    + ('The runs record the override in their notes, so it '
                       'was deliberate.'
                       if entry['stamped'] else
                       'A run whose factors differ is not the arm its label '
                       'names, and registry.jobs refuses such an override '
                       'unless asked.')
                    + f' If the launch meant it, re-run the audit with '
                      f'--overrides {name}='
                    + sorted(entry['observed'])[0]
                    + ' so the runs are checked against the configuration that '
                      'was actually intended.',
                    runs=entry['runs'][:MAX_LISTED], experiment=eid, field=name,
                    declared=sorted(entry['declared']),
                    observed=sorted(entry['observed']))
        if scaled:
            fields = {k: {'declared': sorted(v['declared']),
                          'observed': sorted(v['observed']),
                          'runs': len(v['runs'])}
                      for k, v in sorted(scaled.items())}
            chk.detail[eid]['scaled_fields'] = fields
            chk.add(WARN, 'scaled_launch',
                    f'{eid}: {len(fields)} budget or measurement field(s) '
                    f'differ from the declared configuration '
                    f'({", ".join(sorted(fields))}). These runs are the '
                    f'declared arms at a reduced setting, which is what a '
                    f'pipeline-validation launch is and is not a result',
                    experiment=eid, fields=fields)
    chk.detail['cell_scope'] = {
        'non_screen_experiments': cell_scope_experiments,
        'field_x_cell_comparisons': cell_scope_comparisons}
    if cell_scope_experiments and not cell_scope_comparisons:
        chk.add(NOTE, 'cell_scope_dormant',
                f'the DESIGN.md 3.3 secondary scope compared nothing: across '
                f'{cell_scope_experiments} non-screen experiment(s) it made 0 '
                f'field x scope comparison(s). It tests the fields an '
                f'experiment declares in `scoped_invariants`, plus any '
                f'CORE_INVARIANT it declares varied, and no non-screen '
                f'experiment in this selection declares either: the tuned '
                f'experiments {sorted(registry.TUNED_OF)} are the only ones '
                f'that do and they are enumerated only once a tuning '
                f'selection exists (DESIGN.md 3.3, sequentially dependent on '
                f'E3). E3 and E12 vary a field and are screens, which this '
                f'scope exempts. The check is in force and fires the moment '
                f'the per-cell-tuned policy of 3.3 reaches a reported '
                f'experiment. Until then it enforces nothing, and this file '
                f'says so rather than counting it as a guardrail',
                non_screen_experiments=cell_scope_experiments,
                comparisons=0,
                scope_keys=list(registry.SCOPED_INVARIANT_KEYS),
                tuned_experiments=sorted(registry.TUNED_OF),
                core_invariants=list(registry.CORE_INVARIANTS))
    return chk


def check_seed_completeness(membership, exps, declared: Declared) -> Check:
    """Completeness against the declared inventory, with the gate held shut.

    The requested seed set is deliberately not a parameter here. It arrives
    resolved, as `declared.target_pairs`, and the check is measured against
    that alone: a `seeds` argument that the body never read was the parameter a
    reader would assume carried the gate.
    """
    chk = Check(
        'SEED COMPLETENESS',
        'one seed was dropped from one published arm with no stated rule; '
        'partial arms are refused (DESIGN.md 9, ANALYSIS_PLAN.md 8)')
    vacuous: list[str] = []
    for eid, arms in membership.items():
        exp = exps[eid]
        pairs = declared.target_pairs.get(eid)
        if pairs is None:
            chk.add(ERROR, 'inventory_unresolvable',
                    f'{eid}: the declared arm x seed inventory could not be '
                    f'resolved, so completeness cannot be asserted',
                    experiment=eid)
            continue
        observed = {(label, r.seed) for label, group in arms.items()
                    for r in group if r.seed is not None}
        target = {seed for _, seed in pairs}
        if not pairs:
            # The gate held shut rather than allowed to pass vacuously. With no
            # declared pair, `pairs - observed` is empty, every assertion below
            # is true of nothing, and the audit reports a pass it never made.
            vacuous.append(eid)

        # One arm at one seed is one run. Two runs there are two independent
        # estimates of one quantity that a reader would take for two arms --
        # the failure mode the digest-keyed run directory exists to prevent, so
        # if it appears here it means two launches wrote under one label.
        twinned: list[str] = []
        twin_runs: list[str] = []
        for label, group in sorted(arms.items()):
            by_seed: dict[Optional[int], list[Run]] = defaultdict(list)
            for run in group:
                by_seed[run.seed].append(run)
            for seed, twins in sorted(by_seed.items(), key=lambda kv: str(kv[0])):
                if len(twins) > 1:
                    twinned.append(f'{label}@s{seed}x{len(twins)}')
                    twin_runs.extend(t.rel for t in twins)
        if twinned:
            chk.add(ERROR, 'duplicate_runs_for_arm_seed',
                    f'{eid}: {len(twinned)} arm x seed slot(s) hold more than '
                    f'one run ({", ".join(twinned[:4])}'
                    + (f', +{len(twinned) - 4} more' if len(twinned) > 4 else '')
                    + f'). One arm at one seed is one run; two are two '
                      f'independent estimates of one quantity, which a reader '
                      f'would take for two arms',
                    runs=twin_runs[:MAX_LISTED], experiment=eid,
                    slots=twinned[:24], n=len(twinned))
        missing = sorted(pairs - observed)
        # Every observed pair the registry does not schedule, not only those at
        # a selected seed. The `p[1] in target` filter that used to sit here
        # discarded precisely the class `declare()`'s docstring promises to
        # report -- a run at an *undeclared* seed -- so the target/observed
        # union was used for attribution and then filtered back out of the
        # reporting path. That is the mechanism by which four C4SRC-block runs
        # were folded into E1's confirmatory arms without a word.
        extra = sorted(observed - pairs)
        # Of those, the ones that are this experiment's business. A pair that
        # exists only because `declare()` resolved every experiment at the
        # union of its own seeds and the seeds on disk is not a run this
        # experiment has anything to do with: `shared_configuration_only`
        # requires that another experiment launched it, that this one declares
        # no slot it could fill, and that its launcher does declare the slot it
        # sits in. E8i's ten `c4src-*` donors satisfy all three for each of the
        # ten experiments whose `scratch-*` arms are digest-identical to them,
        # and reporting that as ten deviations was 10 of the 29 warnings that
        # made --strict unpassable. This is not the `p[1] in target` filter
        # that used to sit here and discarded genuine undeclared runs: a run at
        # an undeclared seed whose own launcher does not declare it, or one
        # this experiment launched itself, is still a warning below.
        by_pair: dict[tuple[str, Optional[int]], list[Run]] = defaultdict(list)
        for label, group in arms.items():
            for run in group:
                by_pair[(label, run.seed)].append(run)
        shared_cfg = sorted(
            p for p in extra
            if by_pair.get(p) and all(
                shared_configuration_only(eid, p[0], r, declared)
                for r in by_pair[p]))
        extra = [p for p in extra if p not in set(shared_cfg)]
        per_arm = {label: sorted({r.seed for r in group})
                   for label, group in arms.items()}
        n_by_arm = {label: len([s for s in seen if s in target])
                    for label, seen in per_arm.items()}
        # ANALYSIS_PLAN.md 9 is an any-arm rule, not an every-arm one: "under
        # n < 3, stats.py emits no test and no interval, and report.py stamps
        # every page PIPELINE VALIDATION: NOT A RESULT". One arm below the
        # floor stops every contrast that arm enters, so the page carries the
        # stamp. This used to read `max(n_by_arm.values())`, which is true only
        # when *every* arm is below the floor, and the disagreement was visible
        # on runs_demo: aggregate.py printed "an arm has fewer than 3 seeds
        # (smallest count 1)" for E2 while the audit recorded
        # pipeline_validation=False for the same experiment on the same tree.
        # The minimum is taken over the arms that have runs, which is what
        # aggregate.py (min over the per-arm counts it tabulates) and tables.py
        # (min over the groups present in the frame) both do: an arm with no
        # runs at all is `arm_absent` above, not a small sample here.
        populated = sorted(n for n in n_by_arm.values() if n)
        smallest = populated[0] if populated else 0
        chk.detail[eid] = {
            'declared_runs': len(pairs),
            'observed_runs': len(observed & pairs),
            'seeds_declared': sorted(target),
            'per_arm_seeds': per_arm,
            'family': exp.family,
            'pipeline_validation': smallest < MIN_N_FOR_A_RESULT,
            'min_n_per_populated_arm': smallest,
            'max_n_per_arm': max(n_by_arm.values()) if n_by_arm else 0,
            'arms_populated': len(populated),
            'runs_attributed': sum(len(g) for g in arms.values()),
        }
        if not (observed & pairs):
            # A warning, not a note, when runs exist at other seeds: the
            # completeness gate was then measured against an inventory that has
            # nothing to do with what is on disk, which is the same vacuity as
            # an empty seed set reached by a different route.
            attributed = sum(len(g) for g in arms.values())
            chk.add(WARN if attributed else NOTE, 'experiment_not_run',
                    f'{eid}: no run on disk belongs to any declared arm at the '
                    f'selected seeds'
                    + (f', though {attributed} run(s) are attributed to it at '
                       f'other seeds, so completeness here was asserted against '
                       f'an inventory none of them fills'
                       if attributed else ''),
                    experiment=eid, runs_attributed=attributed)
            continue
        absent = sorted(label for label, seen in per_arm.items() if not seen)
        if absent:
            chk.add(ERROR, 'arm_absent',
                    f'{eid}: {len(absent)} declared arm(s) have no runs at all: '
                    f'{", ".join(absent[:MAX_LISTED])}'
                    + (f' (+{len(absent) - MAX_LISTED} more)'
                       if len(absent) > MAX_LISTED else ''),
                    experiment=eid, arms=absent)
        partial = sorted({label for label, _ in missing if label not in absent})
        if missing:
            chk.add(ERROR, 'seeds_missing',
                    f'{eid}: {len(missing)} of {len(pairs)} declared run(s) are '
                    f'absent; {len(partial)} arm(s) are partial rather than '
                    f'unrun ({", ".join(partial[:MAX_LISTED]) or "none"})',
                    experiment=eid,
                    missing=[f'{a}@s{s}' for a, s in missing[:24]],
                    n_missing=len(missing), partial_arms=partial)
        if extra:
            off_block = sorted(p for p in extra if p[1] not in target)
            chk.add(WARN, 'undeclared_runs',
                    f'{eid}: {len(extra)} run(s) belong to an arm the registry '
                    f'does not schedule at that seed, of which '
                    f'{len(off_block)} sit outside the selected seed set '
                    f'entirely. They are attributed to this experiment and are '
                    f'in none of its declared inventory, so they inflate the '
                    f'runs column without entering the denominator; which block '
                    f'they belong to, and whether that is licensed, is the SEED '
                    f'BLOCKS check\'s verdict',
                    runs=sorted({r.rel for label, group in arms.items()
                                 for r in group
                                 if (label, r.seed) in set(extra)})[:MAX_LISTED],
                    experiment=eid, n=len(extra),
                    pairs=[f'{a}@s{s}' for a, s in extra[:24]],
                    off_block=[f'{a}@s{s}' for a, s in off_block[:24]])
        if shared_cfg:
            chk.add(NOTE, 'shared_configuration_runs',
                    f'{eid}: {len(shared_cfg)} arm x seed slot(s) are filled '
                    f'by runs another experiment launched into a slot it '
                    f'declares, and which this experiment declares nowhere. '
                    f'They are in its membership because the two '
                    f'configurations coincide, they are in none of its '
                    f'declared inventory, and nothing it reports draws on '
                    f'them: aggregate.py builds membership from '
                    f'present & (declared | reserve) and excludes them too',
                    runs=sorted({r.rel for p in shared_cfg
                                 for r in by_pair[p]})[:MAX_LISTED],
                    experiment=eid, n=len(shared_cfg),
                    pairs=[f'{a}@s{s}' for a, s in shared_cfg[:24]])
        if chk.detail[eid]['pipeline_validation']:
            det = chk.detail[eid]
            chk.add(NOTE, 'pipeline_validation',
                    f'{eid}: the smallest arm carrying runs has '
                    f'{det["min_n_per_populated_arm"]} seed(s), against '
                    f'{det["max_n_per_arm"]} in the largest. ANALYSIS_PLAN.md 9 '
                    f'forbids a test or an interval below '
                    f'n={MIN_N_FOR_A_RESULT} and requires the output to be '
                    f'stamped PIPELINE VALIDATION - NOT A RESULT; one arm below '
                    f'the floor stops every contrast it enters, which is why '
                    f'this reads the smallest arm and not the largest',
                    experiment=eid, min_n=det['min_n_per_populated_arm'],
                    max_n=det['max_n_per_arm'])
    if vacuous:
        chk.add(ERROR, 'inventory_empty',
                f'{len(vacuous)} experiment(s) declare no arm x seed at all at '
                f'the requested seed set ({", ".join(vacuous[:MAX_LISTED])}'
                + (f', +{len(vacuous) - MAX_LISTED} more'
                   if len(vacuous) > MAX_LISTED else '')
                + '). Every completeness assertion below is then true of '
                  'nothing, so the audit would report a pass it never made. '
                  'A seed selection that resolves to the empty set does not '
                  'relax this gate, it disables it, and DESIGN.md 8.4 permits '
                  'an override only when it is stamped into the output',
                experiments=vacuous[:24], n=len(vacuous))
    return chk


#: The DESIGN.md 3.4 rows that name a block as source-side. A target-side arm
#: at one of these seeds is estimating from a block declared never to be used
#: for target-side estimation.
SOURCE_SIDE_BLOCKS = ('C4SRC', 'RESERVE')


def blocks_for_seed(seed: Optional[int]) -> tuple[str, ...]:
    """Every declared block a seed belongs to. Empty means no block at all."""
    if seed is None:
        return ()
    return tuple(sorted(name for name, seeds in registry.SEED_BLOCKS.items()
                        if seed in seeds))


def declaring_experiments(run: Run, declared: Declared) -> set[str]:
    """Every experiment whose *declared* inventory this run fills.

    `everywhere` answers a wider question: which experiments could claim the
    run, the ones that claim it only because `declare()` resolved them at the
    union of their own seeds and the seeds on disk included. Contamination is
    the narrower fact. A run enters an experiment's estimand when that
    experiment schedules the arm x seed slot the run occupies, and only then,
    which is also the rule `aggregate.py`'s `MembershipIndex` builds membership
    on, so that is what the leak is computed over.
    """
    key = (str(run.cfg.get('experiment')), str(run.cfg.get('label')), run.seed)
    digest = declared.digests.get(key)
    if digest is None:
        return set()
    return {eid for eid, label, seed in declared.classes.get(digest, ())
            if seed == run.seed
            and (label, seed) in (declared.target_pairs.get(eid) or set())}


def check_seed_blocks(membership, exps, declared: Declared,
                      everywhere: dict[str, set[str]]) -> Check:
    """The whole of the DESIGN.md 3.4 block table, in both directions.

    The predecessor of this check tested one proposition: a reported experiment
    contains a run at a TUNE seed. That is one cell of a six-row table, and it
    is the half of the selection-bias problem that is easiest to avoid by
    accident. The other half is a *selection* experiment that has left its own
    block: `registry.resolve_seeds` lets `--seeds` override the block for every
    experiment at once, so a single-seed validation invocation maps E3's
    hyperparameter screen onto CONFIRM seed 0, where E3's `hp-*-lr0.0005-hard`
    arms share a run digest with E1's confirmatory scratch arms. Selecting a
    learning rate on runs that are then used as the confirmatory denominator is
    revision 1's defect verbatim (DESIGN.md 11 item 2), and the old check
    reported PASS on it twice over: seed 0 is not in TUNE, and E3 was exempt
    anyway.

    Every row is therefore enforced, and the sharpest statement of the leak is
    checked directly rather than through a proxy: one run directory that is in
    the declared inventory of a selection experiment *and* in the declared
    inventory of a reported one is the contamination itself.

    Graded on that, not on launch order. The predecessor of this paragraph
    graded on `config.experiment`: the leak was an error when the manifest
    happened to name the selection experiment and a warning otherwise. On the
    real tree at `--seeds 0`, where E3's `hp-*-lr0.0005-hard` arms and E1's
    confirmatory `scratch-*` arms resolve to the same four directories, that
    reported `[warn] 0 error`; changing two strings in one manifest, and
    nothing else, turned the identical statistical fact into `[FAIL] 2 error`.
    Which experiment the operator listed first on the sweep command line is not
    a fact about contamination, so it no longer decides the severity: both
    experiments declaring the run is an error, and a run launched as part of a
    selection is an error whether or not the selection declares the slot.

    What is left over is a genuine collision of configurations that no declared
    inventory realises: E3 is scheduled on TUNE 200-204 and E1 on CONFIRM 0-9,
    so under the declared blocks their shared configuration is never one run,
    and `declare()` puts them in one equivalence class only because it resolves
    every experiment at the union of its seeds and the seeds on disk. Nothing
    reported can draw on it, so it is recorded as a note rather than promoted
    to an error by --strict. The severity that used to sit there has not been
    given up: it has moved onto the case that actually leaks, which now fires
    regardless of who launched the run, and onto `tune_block_not_a_screen`,
    which refuses the declaration itself.
    """
    chk = Check(
        'SEED BLOCKS',
        'revision 1 selected hyperparameters on seeds 0-4 and ran every '
        'confirmatory arm on 0-9; DESIGN.md 3.4 makes the blocks disjoint and '
        'gives each a single licensed use, in both directions')
    chk.detail['blocks'] = {name: [seeds[0], seeds[-1]] if seeds else []
                            for name, seeds in registry.SEED_BLOCKS.items()}
    # Findings are accumulated per (code, experiment, level) and emitted once.
    # A block violation moves every run of an arm at once, and one line per run
    # would bury the rest of the audit exactly as the per-run schema-drift line
    # did. The offending arms, seeds and blocks go into the finding's detail,
    # where they identify the runs without multiplying the lines.
    hits: dict[tuple[str, str, str], dict] = {}

    # A run for which a specific row of the table has already spoken. The
    # generic out-of-block warning would otherwise repeat that verdict in
    # weaker words, and two lines saying one thing is how a finding gets read
    # as two problems or as none.
    named: set[tuple[str, str]] = set()

    def record(level: str, code: str, eid: str, note: str, run: Run,
               label: str, blocks: tuple[str, ...]) -> None:
        entry = hits.setdefault((code, eid, level),
                                {'note': note, 'runs': [], 'arms': set(),
                                 'seeds': set(), 'blocks': set()})
        entry['runs'].append(run.rel)
        entry['arms'].add(label)
        if run.seed is not None:
            entry['seeds'].add(run.seed)
        entry['blocks'].update(blocks)
        if code != 'out_of_declared_block':
            named.add((eid, run.rel))

    artefacts: dict[str, list[str]] = defaultdict(list)
    for eid, arms in membership.items():
        exp = exps[eid]
        # The only licensed exemption from the TUNE rule is the selection
        # experiment itself: one whose declared block *is* TUNE. It has to be
        # the block and not the family, because the shared-run backstop below
        # reads the same set: narrowing this predicate to screens would take a
        # non-screen experiment declared on TUNE out of `selection_ids` and so
        # out of the very check that catches it. The strength the old
        # `family == 'screen' and seed_block == 'TUNE'` conjunction carried is
        # restored one line down instead, where it belongs and where it does
        # not depend on `family` being one of the three documented values: an
        # experiment may be declared on TUNE only if it is a screen, and that
        # is a fact about the declaration, checkable before a single run
        # exists.
        selection = exp.seed_block == 'TUNE'
        if selection and exp.family != 'screen':
            chk.add(ERROR, 'tune_block_not_a_screen',
                    f'{eid} is declared on the TUNE block with '
                    f'family={exp.family!r}. DESIGN.md 3.4 gives TUNE one '
                    f'licensed use, selection, and nothing computed on it may '
                    f'enter a reported estimate; an experiment that is not a '
                    f'screen has no business declaring it, and declaring it '
                    f'exempts every run of that experiment from the TUNE '
                    f'leakage rule below',
                    experiment=eid, family=exp.family,
                    seed_block=exp.seed_block)
        census: Counter = Counter()
        for label, group in sorted(arms.items()):
            arm = declared.arms.get((eid, label))
            source_side = arm is not None and arm.role == 'source'
            for run in group:
                launched_here = str(run.cfg.get('experiment')) == eid
                declared_pair = (label, run.seed) in (
                    declared.target_pairs.get(eid) or set())
                # A run the experiment launched is that experiment's own doing,
                # and so is a slot the experiment declares: in both cases the
                # launch or the catalogue put a run where the 3.4 table says it
                # must not be. A run that merely shares a configuration is an
                # artefact of the equivalence classes, and the three deserve
                # different verdicts.
                lvl = ERROR if (launched_here or declared_pair) else WARN
                blocks = blocks_for_seed(run.seed)
                census[blocks[0] if blocks else 'none'] += 1
                if shared_configuration_only(eid, label, run, declared):
                    # Not this experiment's run, by all three tests in
                    # `shared_configuration_only`. Judging it here produced 21
                    # of the 29 warnings on the real tree and made --strict
                    # unpassable on a structurally correct campaign: E8i's
                    # `c4src-*` donors are digest-identical to ten other
                    # experiments' `scratch-*` target arms, so each of the ten
                    # was told it had put a target arm on a source block. It
                    # had not. Counted and reported below rather than dropped.
                    artefacts[eid].append(run.rel)
                    continue
                if run.seed is None:
                    record(ERROR, 'seed_unreadable', eid,
                           'carry no readable seed, so the block they belong '
                           'to cannot be established and no row of the 3.4 '
                           'table can be applied to them',
                           run, label, blocks)
                    continue
                if not blocks:
                    record(ERROR, 'seed_outside_every_block', eid,
                           f'sit at seeds in none of the declared blocks '
                           f'{sorted(registry.SEED_BLOCKS)}, so no row of the '
                           f'DESIGN.md 3.4 table licenses them',
                           run, label, blocks)
                    continue
                if 'TUNE' in blocks and not selection:
                    record(ERROR, 'tune_leakage', eid,
                           f'sit on TUNE seeds while this experiment is '
                           f'reported (family={exp.family}, declared block '
                           f'{exp.seed_block}); no reported estimate may draw '
                           f'on the selection block',
                           run, label, blocks)
                elif 'TUNE' in blocks:
                    record(NOTE, 'tune_seeds_expected', eid,
                           'sit on TUNE seeds, which is what this experiment '
                           'is for; nothing it produces may enter a reported '
                           'estimate', run, label, blocks)
                if selection and 'TUNE' not in blocks:
                    record(lvl, 'selection_out_of_block', eid,
                           'belong to a selection experiment (declared block '
                           'TUNE) and sit outside it. DESIGN.md 3.4 reserves '
                           'CONFIRM and REPLICATE against selection: a '
                           'hyperparameter chosen at these seeds is chosen on '
                           'the very runs the reported estimates are computed '
                           'from', run, label, blocks)
                if not source_side and any(b in SOURCE_SIDE_BLOCKS
                                           for b in blocks):
                    record(lvl, 'source_block_on_target_arm', eid,
                           'fill an arm with role=target at a seed in a block '
                           'DESIGN.md 3.4 declares source-side and "never used '
                           'for target-side estimation"', run, label, blocks)
                if 'SMOKE' in blocks and exp.seed_block != 'SMOKE':
                    record(lvl, 'smoke_seed_in_reported_experiment', eid,
                           'sit at the SMOKE seed, which is disjoint from '
                           'CONFIRM precisely so that a pipeline-validation '
                           'run cannot be mistaken for a real one by its seed',
                           run, label, blocks)
                if run.seed not in registry.SEED_BLOCKS[exp.seed_block] \
                        and (eid, run.rel) not in named \
                        and not (source_side and any(b in SOURCE_SIDE_BLOCKS
                                                     for b in blocks)):
                    record(WARN, 'out_of_declared_block', eid,
                           f'sit outside the {exp.seed_block} block this '
                           f'experiment declares. Reducing or moving the seed '
                           f'set is the STANDING_INSTRUCTIONS S8 validation '
                           f'invocation and is stamped in the header above, '
                           f'but these runs are not in the block the catalogue '
                           f'schedules', run, label, blocks)
        chk.detail[eid] = {'declared_block': exp.seed_block,
                           'family': exp.family,
                           'selection_experiment': selection,
                           'runs_by_block': dict(sorted(census.items()))}

    # The leak itself rather than a proxy for it. Computed over the whole
    # catalogue, not the selection, because a run shared between a screen and a
    # confirmatory arm is contaminated whether or not both were asked for.
    selection_ids = {eid for eid, exp in registry.EXPERIMENTS.items()
                     if exp.seed_block == 'TUNE'}
    reported_ids = {eid for eid, exp in registry.EXPERIMENTS.items()
                    if exp.family in ('confirmatory', 'estimation')}
    scoped = {r.rel: r for arms in membership.values()
              for group in arms.values() for r in group}
    shared_err: list[str] = []
    shared_note: list[str] = []
    detail: dict[str, dict] = {}
    for rel, run in sorted(scoped.items()):
        also = everywhere.get(rel) or set()
        sel, rep = sorted(also & selection_ids), sorted(also & reported_ids)
        if not (sel and rep):
            continue
        launched_as = str(run.cfg.get('experiment'))
        # Who actually estimates from this run: the experiments that declare
        # the slot it occupies, plus the experiment that launched it, which
        # owns it whether or not the slot is declared.
        realised = declaring_experiments(run, declared) | {launched_as}
        sel_real = sorted(realised & selection_ids)
        rep_real = sorted(realised & reported_ids)
        detail[rel] = {'selection': sel, 'reported': rep,
                       'selection_realised': sel_real,
                       'reported_realised': rep_real,
                       'launched_as': launched_as, 'seed': run.seed}
        (shared_err if (sel_real and rep_real) else shared_note).append(rel)
    if shared_err:
        chk.add(ERROR, 'selection_shares_run_with_reported',
                f'{len(shared_err)} run(s) launched as part of a selection '
                f'experiment are also the runs a reported experiment estimates '
                f'from. A hyperparameter chosen on a run and a confirmatory '
                f'denominator computed from the same run is revision 1\'s '
                f'defect verbatim (DESIGN.md 11 item 2)',
                runs=shared_err[:MAX_LISTED], n=len(shared_err),
                runs_detail={k: detail[k] for k in shared_err[:MAX_LISTED]})
    if shared_note:
        chk.add(NOTE, 'selection_shares_configuration_with_reported',
                f'{len(shared_note)} run(s) resolve to one configuration for '
                f'both a selection experiment and a reported one, and no '
                f'declared inventory realises the collision: the selection '
                f'experiment schedules that arm at other seeds, and the run '
                f'was not launched as part of it. Under the declared blocks '
                f'the two never meet (TUNE is 200-204 and CONFIRM 0-9). It is '
                f'recorded because a screen launched at these seeds would then '
                f'be selecting on the reported denominators, which is the '
                f'error above and not this note',
                runs=shared_note[:MAX_LISTED], n=len(shared_note),
                runs_detail={k: detail[k] for k in shared_note[:MAX_LISTED]})
    chk.detail['shared_selection_and_reported'] = len(detail)
    if artefacts:
        total = sum(len(v) for v in artefacts.values())
        chk.add(NOTE, 'shared_configuration_membership',
                f'{total} (experiment, run) pairing(s) across '
                f'{len(artefacts)} experiment(s) exist only because two '
                f'catalogue configurations coincide: the run was launched as '
                f'another experiment, fills a slot that experiment declares, '
                f'and fills no slot this one declares. The DESIGN.md 3.4 rows '
                f'are not applied to them, because nothing this experiment '
                f'reports can draw on them',
                n=total,
                by_experiment={eid: len(v)
                               for eid, v in sorted(artefacts.items())},
                runs=sorted({r for v in artefacts.values()
                             for r in v})[:MAX_LISTED])

    for (code, eid, level), entry in sorted(hits.items()):
        rels = sorted(set(entry['runs']))
        chk.add(level, code,
                f'{eid}: {len(rels)} run(s) {entry["note"]}. Seeds '
                f'{sorted(entry["seeds"])} in block(s) '
                f'{sorted(entry["blocks"]) or ["none"]}, across '
                f'{len(entry["arms"])} arm(s): '
                + ', '.join(sorted(entry['arms'])[:MAX_LISTED])
                + (f' (+{len(entry["arms"]) - MAX_LISTED} more)'
                   if len(entry['arms']) > MAX_LISTED else ''),
                runs=rels[:MAX_LISTED], experiment=eid, n=len(rels),
                seeds=sorted(entry['seeds']), blocks=sorted(entry['blocks']),
                arms=sorted(entry['arms'])[:24])
    return chk


def check_digests(runs: list[Run]) -> Check:
    chk = Check(
        'CONFIG/DIGEST CONSISTENCY',
        'a config hash that does not match its config means the schema drifted '
        'and two runs\' identities are no longer comparable (DESIGN.md 8.4)')
    schemas: Counter = Counter()
    # Aggregated per field rather than per run: a schema change moves every
    # digest at once, and one line per run would bury every other check.
    mismatched: dict[str, list[str]] = defaultdict(list)
    # Aggregated for the same reason as the digests below, and it was not:
    # `--out-root runs_demo --overrides num_episodes=14 freeze_updates=150`
    # produced 198 near-identical `config_schema_drift` lines, one per run,
    # which is precisely what buried the PLAN HASH, INVARIANTS and SEED
    # COMPLETENESS findings in that same invocation.
    drifted: dict[str, list[str]] = defaultdict(list)
    invalid: dict[str, list[str]] = defaultdict(list)
    for run in runs:
        try:
            cfg = Config(**run.cfg)
        except TypeError as exc:
            drifted[str(exc)].append(run.rel)
            continue
        except ValueError as exc:
            invalid[str(exc)].append(run.rel)
            continue
        schemas[run.identity.get('digest_schema')] += 1
        for name, recomputed in (('run_digest', cfg.run_digest()),
                                 ('trajectory_digest', cfg.trajectory_digest()),
                                 ('measurement_digest', cfg.measurement_digest())):
            if run.identity.get(name) != recomputed:
                mismatched[name].append(run.rel)
        if run.identity.get('arm_id') != cfg.arm_id():
            chk.add(ERROR, 'arm_id_mismatch',
                    f'{run.rel}: stored arm_id {run.identity.get("arm_id")!r} '
                    f'!= {cfg.arm_id()!r} from the stored config',
                    runs=[run.rel])
    for reason, rels in sorted(drifted.items()):
        chk.add(ERROR, 'config_schema_drift',
                f'{len(rels)} of {len(runs)} run(s): the stored config cannot '
                f'be loaded into the current Config ({reason}); their digests '
                f'are not comparable with any run written by this code',
                runs=rels[:MAX_LISTED], reason=reason, n=len(rels))
    for reason, rels in sorted(invalid.items()):
        chk.add(ERROR, 'config_invalid',
                f'{len(rels)} of {len(runs)} run(s): the stored config is '
                f'rejected by Config validation ({reason})',
                runs=rels[:MAX_LISTED], reason=reason, n=len(rels))
    for name, rels in sorted(mismatched.items()):
        chk.add(ERROR, 'digest_mismatch',
                f'{len(rels)} of {len(runs)} run(s): the stored {name} does not '
                f'match the digest recomputed from the run\'s own stored '
                f'config. Either the config was edited after the run, or the '
                f'digested field set changed without DIGEST_SCHEMA being '
                f'bumped -- in which case old and new digests share one '
                f'namespace and two different configurations can claim one '
                f'identity',
                runs=rels[:MAX_LISTED], field=name, n=len(rels))
    chk.detail['digest_schemas'] = dict(schemas)
    chk.detail['mismatched'] = {k: len(v) for k, v in sorted(mismatched.items())}
    chk.detail['config_unloadable'] = sum(len(v) for v in drifted.values())
    chk.detail['config_rejected'] = sum(len(v) for v in invalid.values())
    if len(schemas) > 1:
        chk.add(ERROR, 'digest_schema_split',
                f'runs were written under {len(schemas)} digest schemas '
                f'({", ".join(str(s) for s in schemas)}); digests from '
                f'different schemas are not comparable and must not be pooled',
                schemas=dict(schemas))
    return chk


def _trajectory_digests_on_disk(run: Run) -> dict[str, str]:
    """Every trajectory digest written anywhere inside the run directory."""
    found: dict[str, str] = {}
    stored = run.identity.get('trajectory_digest')
    if stored:
        found['manifest'] = stored
    states = [os.path.join(run.path, 'state.json')]
    states += sorted(glob.glob(os.path.join(glob.escape(run.path), 'ckpt_ep*',
                                            'state.json')))
    for path in states:
        try:
            with open(path, encoding='utf-8') as fh:
                digest = (json.load(fh) or {}).get('trajectory_digest')
        except (OSError, json.JSONDecodeError):
            continue
        if digest:
            found[os.path.relpath(path, run.path).replace(os.sep, '/')] = digest
    return found


def check_run_dir_uniqueness(runs: list[Run]) -> Check:
    chk = Check(
        'RUN-DIRECTORY UNIQUENESS',
        'nine conditions from six experiments collided onto one path and a '
        'completed directory was silently resumed, so one run\'s metrics were '
        'served under five other runs\' manifests (DESIGN.md, config.py)')
    by_digest: dict[str, list[Run]] = defaultdict(list)
    moved: list[str] = []
    for run in runs:
        parts = run.rel.split('/')
        digest = run.run_digest
        if digest:
            by_digest[digest].append(run)

        # The path must encode the identity. `<condition>/<digest12>/s<NN>` is
        # the whole point of the scheme: a directory that does not name its own
        # digest cannot be checked for having been reused.
        if len(parts) != 3 or not parts[2].startswith('s'):
            chk.add(ERROR, 'path_shape',
                    f'{run.rel}: not of the form <condition>/<digest12>/s<NN>; '
                    f'an off-scheme path is outside the collision guarantee',
                    runs=[run.rel])
        else:
            cond, digest12, seed_part = parts
            if cond != run.condition:
                chk.add(ERROR, 'path_condition_mismatch',
                        f'{run.rel}: path says condition {cond!r}, config says '
                        f'{run.condition!r}', runs=[run.rel])
            if digest and digest12 != digest[:12]:
                chk.add(ERROR, 'path_digest_mismatch',
                        f'{run.rel}: path digest {digest12} != run_digest '
                        f'{digest[:12]}; the manifest in this directory is not '
                        f'the run the directory names', runs=[run.rel])
            try:
                path_seed = int(seed_part[1:])
            except ValueError:
                path_seed = None
            if path_seed is not None and run.seed is not None \
                    and path_seed != run.seed:
                chk.add(ERROR, 'path_seed_mismatch',
                        f'{run.rel}: path says seed {path_seed}, config says '
                        f'{run.seed}', runs=[run.rel])

        # The resumed-under-a-different-config mode: two trajectory digests
        # inside one directory means two configurations wrote to it.
        digests = _trajectory_digests_on_disk(run)
        distinct = sorted(set(digests.values()))
        if len(distinct) > 1:
            chk.add(ERROR, 'trajectory_digest_collision',
                    f'{run.rel}: {len(distinct)} distinct trajectory digests '
                    f'inside one run directory; two configurations wrote here, '
                    f'so the metrics and the manifest describe different runs',
                    runs=[run.rel], sources=digests)

        recorded = run.identity.get('run_dir')
        if recorded:
            tail = '/'.join(recorded.replace('\\', '/').rstrip('/').split('/')[-3:])
            if tail != run.rel:
                moved.append(run.rel)

    for digest, group in sorted(by_digest.items()):
        if len(group) > 1:
            chk.add(ERROR, 'duplicate_run_digest',
                    f'run_digest {digest[:12]} appears in {len(group)} '
                    f'directories; one configuration has two independent '
                    f'estimates, which a reader would take for two arms',
                    runs=[r.rel for r in group][:MAX_LISTED], digest=digest)
    if moved:
        chk.add(WARN, 'run_dir_moved',
                f'{len(moved)} run(s) record a run_dir that is not where they '
                f'now sit; the tree has been moved or copied since the runs, so '
                f'any path recorded inside them -- a source checkpoint above '
                f'all -- has to be resolved by digest rather than by path',
                runs=moved[:MAX_LISTED], n=len(moved))
    chk.detail['runs'] = len(runs)
    chk.detail['runs_relocated'] = len(moved)
    chk.detail['distinct_run_digests'] = len(by_digest)
    return chk


def check_metrics_integrity(runs: list[Run]) -> Check:
    """Index integrity, outcome content, and per-run distinctness of the file.

    The index check on its own is content-blind, which the adversarial pass
    demonstrated by copying one seed's `metrics.jsonl` verbatim into three seed
    directories under rewritten manifests and watching the tree pass clean.
    Three byte-identical seeds were indistinguishable from three real ones,
    which is the fabrication mode named in this module's own docstring reached
    by the one route the digest scheme does not cover. So the file's content
    hash is required to be distinct between run directories, and the outcome
    column is required to exist and to vary: a run that trained and recorded no
    score, or recorded one constant, otherwise reaches `aggregate.py` with the
    audit's blessing.
    """
    chk = Check(
        'METRICS INTEGRITY',
        'the published trainer appended on resume without truncating, so a '
        'crash duplicated episodes and corrupted every window statistic '
        '(DESIGN.md 8.2); and an index check alone cannot see the content')
    short = 0
    by_content: dict[str, list[str]] = defaultdict(list)
    for run in runs:
        path = os.path.join(run.path, 'metrics.jsonl')
        if not os.path.exists(path):
            chk.add(ERROR, 'metrics_absent',
                    f'{run.rel}: metrics.jsonl is missing', runs=[run.rel])
            continue
        episodes: list[int] = []
        outcomes: list[float] = []
        evals: list[float] = []
        outcome_rows = 0
        bad_rows = 0
        digest = hashlib.sha256()
        try:
            with open(path, 'rb') as raw:
                for chunk in iter(lambda: raw.read(1 << 20), b''):
                    digest.update(chunk)
        except OSError as exc:
            chk.add(ERROR, 'metrics_unreadable',
                    f'{run.rel}: metrics.jsonl cannot be read ({exc})',
                    runs=[run.rel])
            continue
        with open(path, encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    episodes.append(int(row['episode']))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    bad_rows += 1
                    continue
                if not isinstance(row, dict):
                    continue
                if OUTCOME_FIELD in row:
                    outcome_rows += 1
                    value = _as_float(row.get(OUTCOME_FIELD))
                    if value is not None:
                        outcomes.append(value)
                value = _as_float(row.get(EVAL_OUTCOME_FIELD))
                if value is not None:
                    evals.append(value)
        if episodes:
            by_content[digest.hexdigest()].append(run.rel)
        if bad_rows:
            chk.add(ERROR, 'metrics_unparseable',
                    f'{run.rel}: {bad_rows} row(s) are not JSON objects keyed '
                    f'by episode', runs=[run.rel], rows=bad_rows)
        counts = Counter(episodes)
        duplicates = sorted(e for e, n in counts.items() if n > 1)
        if duplicates:
            chk.add(ERROR, 'metrics_duplicate_episodes',
                    f'{run.rel}: {len(duplicates)} episode index/indices appear '
                    f'more than once (first: {duplicates[:5]})',
                    runs=[run.rel], duplicates=duplicates[:24])

        # `episodes_completed` is a field on disk, not an integer. Casting it
        # unguarded meant one manifest carrying the string 'one thousand' ended
        # the process with a ValueError and produced no report for any of the
        # other runs in the tree.
        raw_completed = run.get('result', 'episodes_completed')
        completed = _as_int(raw_completed)
        if raw_completed is None:
            chk.add(ERROR, 'result_absent',
                    f'{run.rel}: no result.episodes_completed; the run did not '
                    f'finish, so it has no endpoint to report', runs=[run.rel])
        elif completed is None or completed < 0:
            chk.add(ERROR, 'result_malformed',
                    f'{run.rel}: result.episodes_completed is '
                    f'{raw_completed!r}, which is not an episode count, so the '
                    f'run\'s own claim about its length cannot be checked '
                    f'against the file', runs=[run.rel],
                    value=str(raw_completed))
        else:
            expected = set(range(0, completed))
            if set(episodes) != expected:
                missing = sorted(expected - set(episodes))
                extra = sorted(set(episodes) - expected)
                chk.add(ERROR, 'metrics_not_contiguous',
                        f'{run.rel}: episode index set is not range(0, '
                        f'{completed}) -- {len(missing)} missing, '
                        f'{len(extra)} out of range',
                        runs=[run.rel], missing=missing[:24], extra=extra[:24])
            raw_declared = run.cfg.get('num_episodes')
            declared = _as_int(raw_declared) or 0
            if raw_declared is not None and _as_int(raw_declared) is None:
                chk.add(ERROR, 'config_num_episodes_malformed',
                        f'{run.rel}: config.num_episodes is {raw_declared!r}, '
                        f'which is not an episode count', runs=[run.rel])
            if declared and completed < declared:
                short += 1

        # The outcome, which nothing here used to read. DESIGN.md 5.1 puts
        # every comparison on the normalised score, so a run with no finite
        # score has no endpoint, and one with a single distinct score across
        # every episode recorded a constant rather than a trajectory.
        if not outcome_rows:
            chk.add(ERROR, 'outcome_column_absent',
                    f'{run.rel}: no row carries a {OUTCOME_FIELD!r} field, so '
                    f'the run recorded no outcome on the normalised scale every '
                    f'comparison is drawn on (DESIGN.md 5.1)', runs=[run.rel])
        elif not outcomes:
            chk.add(ERROR, 'outcome_all_null',
                    f'{run.rel}: {outcome_rows} row(s) carry {OUTCOME_FIELD!r} '
                    f'and not one holds a finite number. The run trained and '
                    f'recorded no score; every statistic computed from it '
                    f'downstream would be over an empty column', runs=[run.rel],
                    rows=outcome_rows)
        elif len(outcomes) == 1 and len(episodes) > 1:
            # Split out of `outcome_constant`, which used to report this as
            # "takes one value across 1 episode(s)" on a run of a thousand:
            # the branch tested `len(episodes)` and the message printed
            # `len(outcomes)`, so a column that is null almost everywhere was
            # diagnosed as a zero-variance arm. They are different defects and
            # only one of them is ambiguous.
            chk.add(ERROR, 'outcome_mostly_null',
                    f'{run.rel}: exactly one of {len(episodes)} episode(s) '
                    f'carries a finite {OUTCOME_FIELD}. That is not a '
                    f'zero-variance arm, it is a column that is null almost '
                    f'everywhere, and every statistic drawn from it downstream '
                    f'would be one number wearing the n of the whole run',
                    runs=[run.rel], value=str(outcomes[0]),
                    finite_rows=1, episodes=len(episodes))
        elif len(episodes) > 1 and len(set(outcomes)) == 1:
            # A warning, not an error, and the demotion is the point.
            #
            # A constant outcome column has two causes and this file cannot
            # tell them apart. One is the fabrication mode: the column was
            # written once and never updated. The other is a policy that
            # collapsed onto a return the environment produces exactly, which
            # is physically attainable and has been measured in this repo:
            # `reference_returns.json` records Acrobot-v1 at `noop_return
            # -500.0` over 100 episodes, and under the 200-step cap every run
            # on disk was launched with, a near-random Acrobot policy returns
            # the cap on every single episode. Acrobot is the source env of
            # E9's `acro2ll` pair and the target env of `cp2acro`.
            #
            # DESIGN.md 4.3 already rules on that second case, and its ruling
            # is not a refusal: a source that did not learn is a stated
            # exclusion plus a RESERVE draw. Erroring here refused data the
            # pre-registered spec declares valid, and the only escape,
            # --allow-audit-failure, waives the whole audit and stamps every
            # artifact of an otherwise-clean campaign as an override. So it is
            # reported at a severity that says "look at this", and --strict
            # still promotes it, so a confirmatory campaign cannot carry one
            # silently.
            chk.add(WARN, 'outcome_constant',
                    f'{run.rel}: {OUTCOME_FIELD} takes one value '
                    f'({outcomes[0]!r}) across all {len(outcomes)} finite '
                    f'row(s) of {len(episodes)} episode(s). Either the column '
                    f'was written once and never updated, or the policy '
                    f'collapsed onto a return the environment yields exactly '
                    f'(Acrobot-v1 under a step cap returns the cap every '
                    f'episode; see reference_returns.json). This file cannot '
                    f'tell those apart, so it reports rather than adjudicates: '
                    f'DESIGN.md 4.3 governs a source that did not learn, and '
                    f'--strict promotes this to an error',
                    runs=[run.rel], value=str(outcomes[0]),
                    finite_rows=len(outcomes), episodes=len(episodes))
        if outcome_rows and not evals:
            chk.add(WARN, 'eval_outcome_absent',
                    f'{run.rel}: no row carries a finite '
                    f'{EVAL_OUTCOME_FIELD!r}; the held-out endpoint '
                    f'ANALYSIS_PLAN.md draws its co-primary from is not in this '
                    f'run', runs=[run.rel])

        stated = run.get('result', 'metrics_integrity', 'contiguous')
        if stated is False:
            chk.add(ERROR, 'metrics_integrity_self_reported',
                    f'{run.rel}: the run recorded its own metrics as '
                    f'non-contiguous: '
                    f'{run.get("result", "metrics_integrity", "problems")}',
                    runs=[run.rel])
    for content, rels in sorted(by_content.items()):
        if len(rels) > 1:
            chk.add(ERROR, 'metrics_content_duplicated',
                    f'{len(rels)} run directories hold a byte-identical '
                    f'metrics.jsonl (sha256 {content[:12]}). Distinct seeds '
                    f'cannot produce identical trajectories, so one run\'s '
                    f'metrics are being served under more than one manifest: '
                    f'the fabrication mode of DESIGN.md 8.2 reached by a file '
                    f'copy rather than by a configuration collision',
                    runs=sorted(rels)[:MAX_LISTED], sha256=content,
                    n=len(rels))
    chk.detail['distinct_metrics_files'] = len(by_content)
    chk.detail['runs_with_metrics'] = sum(len(v) for v in by_content.values())
    if short:
        chk.add(WARN, 'runs_short_of_budget',
                f'{short} run(s) completed fewer episodes than num_episodes; '
                f'their endpoints are performance at a smaller budget than the '
                f'arm declares, which is not comparable with a full-budget arm',
                n=short)
    return chk


def check_freeze(runs: list[Run]) -> Check:
    chk = Check(
        'FREEZE VERIFICATION',
        'the manuscript described a freeze schedule the code never implemented '
        '(DESIGN.md 3.2); both directions of the fingerprint check matter')
    verified = unverified = 0
    never_left: list[tuple[str, int, int]] = []
    for run in runs:
        # `freeze_events` written as a JSON object rather than an array made
        # `enumerate` walk its keys, so `event` was a string and the audit died
        # on `str.get`, producing no report for any run in the tree. A
        # malformed schedule is a finding about that run, not an exception.
        events, malformed = _events(run.manifest.get('freeze_events'))
        if malformed:
            chk.add(ERROR, 'freeze_events_malformed',
                    f'{run.rel}: {malformed} freeze event(s) are not JSON '
                    f'objects (freeze_events is '
                    f'{type(run.manifest.get("freeze_events")).__name__}); the '
                    f'schedule this run recorded cannot be read, so it is '
                    f'unverified whichever way it ran',
                    runs=[run.rel], n=malformed)
        transfer = run.condition != 'scratch'
        window = _as_int(run.cfg.get('freeze_updates')) or 0
        updates = _as_int(run.get('result', 'updates', default=0)) or 0
        checked = 0
        for i, event in enumerate(events):
            # DESIGN.md 8.3 requires a freeze event to carry its
            # trainable-parameter counts, and a window in which nothing is
            # trainable is a run that made no progress at all: the fingerprint
            # check cannot catch it, because a layer that is meant to be frozen
            # and a layer nobody is training look identical from the weights.
            trainable = _as_int(event.get('trainable_params'))
            if 'trainable_params' not in event:
                chk.add(ERROR, 'freeze_event_params_unrecorded',
                        f'{run.rel}: freeze event {i} records no '
                        f'trainable_params; DESIGN.md 8.3 requires the '
                        f'parameter counts on every freeze event, and without '
                        f'them a freeze of everything is indistinguishable '
                        f'from a freeze of the trunk',
                        runs=[run.rel], event=i)
            elif trainable is None or trainable <= 0:
                chk.add(ERROR, 'freeze_event_nothing_trainable',
                        f'{run.rel}: freeze event {i} declares '
                        f'trainable_params={event.get("trainable_params")!r}, '
                        f'so no parameter in the network could move across that '
                        f'window and the run learned nothing while it was open',
                        runs=[run.rel], event=i,
                        trainable_params=str(event.get('trainable_params')))
            verdict = _mapping(event.get('verification'))
            if verdict is None:
                if event.get('verification') is not None:
                    chk.add(ERROR, 'freeze_verification_malformed',
                            f'{run.rel}: freeze event {i} carries a '
                            f'verification that is not an object '
                            f'({event.get("verification")!r}), so the '
                            f'fingerprint comparison cannot be read',
                            runs=[run.rel], event=i)
                continue
            checked += 1
            if verdict.get('frozen_but_changed'):
                chk.add(ERROR, 'frozen_layer_moved',
                        f'{run.rel}: freeze event {i} reports declared-frozen '
                        f'layer(s) whose weights changed: '
                        f'{verdict["frozen_but_changed"]}',
                        runs=[run.rel], event=i, verdict=verdict)
            # `verify_freeze` sets ok from the first direction only; an inert
            # trainable layer usually means the optimiser never received its
            # gradients, which is what a positionally resolved freeze produces
            # silently, so it is an error here too.
            if verdict.get('trainable_but_unchanged'):
                chk.add(ERROR, 'trainable_layer_inert',
                        f'{run.rel}: freeze event {i} reports trainable '
                        f'layer(s) whose weights never moved across the freeze '
                        f'window: {verdict["trainable_but_unchanged"]}',
                        runs=[run.rel], event=i, verdict=verdict)
            if verdict.get('ok') is False:
                chk.add(ERROR, 'freeze_not_ok',
                        f'{run.rel}: freeze event {i} verification.ok is false',
                        runs=[run.rel], event=i, verdict=verdict)
        if transfer and window != 0:
            frozen_events = [e for e in events if e.get('frozen')]
            if not frozen_events:
                chk.add(ERROR, 'freeze_never_applied',
                        f'{run.rel}: freeze_updates={window} but no freeze event '
                        f'records frozen=True; the schedule the config declares '
                        f'did not run', runs=[run.rel])
            if not checked:
                unverified += 1
                if window > 0 and updates >= window:
                    chk.add(ERROR, 'freeze_exit_unverified',
                            f'{run.rel}: the run made {updates} updates against '
                            f'a {window}-update freeze window, so it left the '
                            f'window, yet no event carries a fingerprint '
                            f'verification', runs=[run.rel])
                else:
                    never_left.append((run.rel, updates, window))
            else:
                verified += 1
        if not transfer and events and any(e.get('frozen') for e in events):
            chk.add(ERROR, 'scratch_run_frozen',
                    f'{run.rel}: a scratch run records a freeze event with '
                    f'frozen=True', runs=[run.rel])
    if never_left:
        chk.add(NOTE, 'freeze_window_never_left',
                f'{len(never_left)} transfer run(s) ended inside their freeze '
                f'window, so the fingerprints at the unfreeze boundary were '
                f'never compared and the freeze schedule is unverified in this '
                f'tree. DESIGN.md 3.2 makes that verification the evidence '
                f'that the schedule ran at all, so a campaign in which no run '
                f'leaves the window has not tested it',
                runs=[r for r, _u, _w in never_left][:MAX_LISTED],
                n=len(never_left),
                updates_vs_window=sorted({(u, w) for _r, u, w in never_left})[:6])
    chk.detail['runs_with_verified_freeze_exit'] = verified
    chk.detail['transfer_runs_never_verified'] = unverified
    return chk


def check_source_validity(runs: list[Run], membership=None,
                          declared: "Declared | None" = None,
                          out_root: str = 'runs') -> Check:
    """The gate recomputed, and DESIGN.md 4.3's reserve rule audited.

    Two things were wrong with reading `source.validity.valid` and believing
    it. First, the flag is written by the run it judges, so the check named
    after "a source scoring 26.94 out of 475 was transferred from anyway" was
    passed by a manifest that simply said `valid: true, gate: 0.0`. The gate is
    therefore compared with `registry.SOURCE_VALIDITY_GATE` and the verdict
    recomputed from the recorded score, which is also where a disagreement
    between the two copies of that constant becomes "visible in the data" as
    `registry.py`'s own comment promises it will be.

    Second, 4.3 does not stop at excluding an invalid source. It requires the
    seed to be *replaced*, "with source seeds drawn in order from RESERVE until
    the cell has its full complement of valid sources". Nothing compared the
    post-exclusion count of valid sources against the declared inventory, so an
    arm could be complete to the seed-completeness gate and empty in the primary
    estimand: in the real tree, cell dueling-vanilla stood at one valid source
    against four exclusions with no replacement ledger at all, and SOURCE
    VALIDITY reported `[ok]`. The exclusion itself stays a note, exactly as 4.3
    says; the *absence of the replacement it mandates* is the error.
    """
    chk = Check(
        'SOURCE VALIDITY',
        'a published source agent scored 26.94 on a task solved at 475 and was '
        'transferred from anyway; and an exclusion without the reserve draw '
        'DESIGN.md 4.3 mandates leaves the primary estimand short')
    gate_declared = float(registry.SOURCE_VALIDITY_GATE)
    chk.detail['gate'] = gate_declared
    per_cell: dict[str, Counter] = defaultdict(Counter)
    excluded: dict[str, list[tuple[str, Any, Any]]] = defaultdict(list)
    invalid_rels: set[str] = set()
    for run in runs:
        if run.condition == 'scratch':
            continue
        validity = _mapping(run.get('source', 'validity'))
        if validity is None:
            code = ('validity_verdict_malformed'
                    if run.get('source', 'validity') is not None
                    else 'validity_verdict_missing')
            chk.add(ERROR, code,
                    f'{run.rel}: a {run.condition} run with no readable '
                    f'source-validity verdict; the gate of DESIGN.md 4.3 was '
                    f'never evaluated, or what it wrote cannot be read',
                    runs=[run.rel])
            per_cell[run.cell]['missing'] += 1
            continue
        valid = validity.get('valid')
        raw_score = validity.get('source_final_score')
        score = _as_float(raw_score)
        raw_gate = validity.get('gate')
        gate = _as_float(raw_gate)

        # The gate itself, before anything is concluded from it. A recorded gate
        # that is not the declared one means the verdict was taken under some
        # other rule than the one DESIGN.md 4.3 states, whatever it claims.
        if raw_gate is None:
            chk.add(ERROR, 'validity_gate_unrecorded',
                    f'{run.rel}: the validity verdict records no gate, so which '
                    f'threshold it was taken against is unknown',
                    runs=[run.rel])
        elif gate is None or abs(gate - gate_declared) > GATE_TOLERANCE:
            chk.add(ERROR, 'validity_gate_mismatch',
                    f'{run.rel}: the verdict was taken against gate '
                    f'{raw_gate!r}, not the {gate_declared} of DESIGN.md 4.3 '
                    f'and registry.SOURCE_VALIDITY_GATE. A self-declared gate '
                    f'is the one edit this check exists to refuse',
                    runs=[run.rel], recorded=str(raw_gate),
                    declared=gate_declared)

        if run.condition == 'transfer_untrained':
            # Not applicable by construction: the source is random, and the
            # manifest must say so rather than leaving a null to be read as a
            # failed gate.
            per_cell[run.cell]['not_applicable'] += 1
            if not validity.get('note'):
                chk.add(WARN, 'untrained_validity_unexplained',
                        f'{run.rel}: transfer_untrained carries a null validity '
                        f'verdict with no note saying the gate does not apply',
                        runs=[run.rel])
            continue

        # The verdict recomputed. A non-finite score is a measurement failure
        # and not a rejection: `nan >= 0.6` is False, which is exactly how a
        # degenerate evaluation comes to read as a source below the gate.
        if raw_score is None:
            chk.add(ERROR, 'validity_score_unrecorded',
                    f'{run.rel}: the validity verdict carries no '
                    f'source_final_score, so the gate cannot be recomputed and '
                    f'the flag beside it cannot be checked', runs=[run.rel])
        elif score is None:
            chk.add(ERROR, 'validity_score_not_finite',
                    f'{run.rel}: source_final_score is {raw_score!r}. That is a '
                    f'measurement which did not happen, not a source which fell '
                    f'below the gate, and the two must not read the same way',
                    runs=[run.rel], value=str(raw_score))
        elif isinstance(valid, bool):
            implied = score >= gate_declared
            if implied != valid:
                chk.add(ERROR, 'validity_verdict_contradicts_gate',
                        f'{run.rel}: the manifest declares valid={valid} while '
                        f'its own recorded score {score:.4f} against the '
                        f'{gate_declared} gate implies valid={implied}. The '
                        f'verdict is recomputed here rather than believed, '
                        f'because a check that reads a self-declared boolean is '
                        f'passed by the very edit it exists to catch',
                        runs=[run.rel], score=round(score, 6),
                        declared_valid=valid, implied_valid=implied)

        if valid is True:
            per_cell[run.cell]['valid'] += 1
        elif valid is False:
            per_cell[run.cell]['invalid'] += 1
            invalid_rels.add(run.rel)
            # Reported, never fatal: DESIGN.md 4.3 makes the primary estimand
            # valid-sources-only with the exclusions printed, so an invalid
            # source is an exclusion for `report.py` to carry, not a broken run.
            # What is fatal is the reserve draw that must follow it, checked
            # against the declared inventory below.
            excluded[run.cell].append((run.rel, raw_score, raw_gate))
        else:
            per_cell[run.cell]['unknown'] += 1
            chk.add(ERROR, 'validity_unknown',
                    f'{run.rel}: the validity verdict is null and the condition '
                    f'is {run.condition}, so the source competence this run '
                    f'depends on was never established', runs=[run.rel])
    for cell, rows in sorted(excluded.items()):
        gate = rows[0][2]
        chk.add(NOTE, 'source_invalid',
                f'{cell}: {len(rows)} run(s) transfer from a source below the '
                f'{gate} normalised-score gate and are excluded from the '
                f'primary estimand; DESIGN.md 4.3 requires the number and '
                f'identity of the rejected sources to appear in the results '
                f'table',
                runs=[r for r, _s, _g in rows][:MAX_LISTED], cell=cell,
                n=len(rows),
                scores=sorted({round(_as_float(s), 4) for _r, s, _g in rows
                               if _as_float(s) is not None}))
    chk.detail['per_cell'] = {cell: dict(counts)
                              for cell, counts in sorted(per_cell.items())}

    # DESIGN.md 4.3's second half: the reserve rule. A declared arm x seed slot
    # whose occupant transferred from a rejected source is a slot the primary
    # estimand cannot use, and the design says it is refilled from RESERVE
    # rather than left short. Measured against the declared inventory, so it
    # says nothing about seeds that were never run at all: that is seed
    # completeness's verdict, not this one's.
    replacements, draws = registry.load_source_replacements(out_root)
    chk.detail['reserve_ledger'] = {
        'path': os.path.join(os.path.abspath(out_root),
                             registry.REPLACEMENTS_RELPATH),
        'assignments': len(replacements),
        'arms_drawn_for': sorted(draws),
    }
    if membership is not None and declared is not None:
        short: dict[str, list[str]] = defaultdict(list)
        short_runs: dict[str, list[str]] = defaultdict(list)
        smoke_short: dict[str, list[str]] = defaultdict(list)
        for eid, arms in membership.items():
            exp = registry.EXPERIMENTS.get(eid)
            # DESIGN.md 3.4 gives SMOKE one licensed use, pipeline validation,
            # and ANALYSIS_PLAN.md 9 says nothing computed on it is a result.
            # An experiment declared there runs under `registry.SMOKE_OVERRIDES`
            # at twelve episodes, so its sources cannot reach the 0.6
            # normalised gate and neither can any RESERVE replacement, which is
            # another twelve-episode run: measured in runs_demo, smoke-transfer
            # sources score -0.027 / -0.030 / -0.030 and smoke-iface 0.284 /
            # -0.519 / -1.122. The reserve rule is a rule about a reported
            # estimand, and E0 has none, so requiring it there made the one
            # experiment DESIGN.md provides for validating the pipeline
            # incapable of ever producing a green audit, with
            # --allow-audit-failure as the only route past it. Recorded as a
            # note there, and left an error everywhere else: it is currently
            # and correctly firing on the real tree for src-dueling-vanilla at
            # 0.5992 against the 0.600 gate, and nothing here touches that.
            bucket = (smoke_short if exp is not None
                      and exp.seed_block == 'SMOKE' else short)
            pairs = declared.target_pairs.get(eid) or set()
            for label, group in sorted(arms.items()):
                for run in group:
                    if (label, run.seed) not in pairs:
                        continue          # not part of the declared estimand
                    if run.rel in invalid_rels:
                        bucket[eid].append(f'{label}@s{run.seed}')
                        short_runs[eid].append(run.rel)
        for eid in sorted(smoke_short):
            slots = sorted(smoke_short[eid])
            chk.add(NOTE, 'reserve_rule_not_applicable',
                    f'{eid}: {len(slots)} declared arm x seed slot(s) are '
                    f'filled by a run whose source failed the validity gate, '
                    f'and no RESERVE replacement can clear it: {eid} is '
                    f'declared on the SMOKE block, so every draw is another '
                    f'pipeline-validation run at the same reduced budget and '
                    f'fails the same gate. ANALYSIS_PLAN.md 9 puts nothing it '
                    f'produces in a reported estimand, so there is no '
                    f'complement for DESIGN.md 4.3 to fill: '
                    f'{", ".join(slots[:MAX_LISTED])}'
                    + (f' (+{len(slots) - MAX_LISTED} more)'
                       if len(slots) > MAX_LISTED else ''),
                    runs=sorted(set(short_runs[eid]))[:MAX_LISTED],
                    experiment=eid, n=len(slots), slots=slots[:24])
        for eid in sorted(short):
            slots = sorted(short[eid])
            chk.add(ERROR, 'reserve_rule_not_applied',
                    f'{eid}: {len(slots)} declared arm x seed slot(s) are '
                    f'filled by a run whose source failed the validity gate, '
                    f'and the declared inventory still points at those runs. '
                    f'DESIGN.md 4.3 makes the primary estimand valid-sources '
                    f'only, "with source seeds drawn in order from RESERVE '
                    f'until the cell has its full complement of valid sources", '
                    f'and the ledger at {registry.REPLACEMENTS_RELPATH} carries '
                    f'{len(replacements)} assignment(s). Until the draw happens '
                    f'the arm is complete to the seed count and short in the '
                    f'estimand: {", ".join(slots[:MAX_LISTED])}'
                    + (f' (+{len(slots) - MAX_LISTED} more)'
                       if len(slots) > MAX_LISTED else ''),
                    runs=sorted(set(short_runs[eid]))[:MAX_LISTED],
                    experiment=eid, n=len(slots), slots=slots[:24],
                    ledger_assignments=len(replacements))
    return chk


def check_source_lineage(runs: list[Run],
                         universe: list[Run] | None = None) -> Check:
    """Iterated over the selected runs; resolved against the whole tree.

    Resolution has to see every run, not only the selection: E8i's positive
    control draws its donors from the disjoint `C4SRC` block, so auditing
    E8i alone must still be able to find them.
    """
    chk = Check(
        'SOURCE LINEAGE',
        'the published DDQN transfer arm\'s source is unidentifiable and the '
        'only surviving CartPole checkpoint is from the wrong architecture')
    pool = universe if universe is not None else runs
    rehomed: list[str] = []
    by_digest = {r.run_digest: r for r in pool if r.run_digest}
    by_path = {os.path.normcase(os.path.normpath(r.path)): r for r in pool}
    resolved = 0
    for run in runs:
        if run.condition == 'scratch':
            continue
        if run.condition == 'transfer_untrained':
            if run.get('source', 'checkpoint'):
                chk.add(WARN, 'untrained_has_checkpoint',
                        f'{run.rel}: transfer_untrained records a source '
                        f'checkpoint; its source is meant to be randomly '
                        f'initialised', runs=[run.rel])
            continue

        checkpoint = run.get('source', 'checkpoint')
        recorded_digest = run.get('source', 'source_result', 'run_digest')
        if not checkpoint:
            chk.add(ERROR, 'source_checkpoint_unrecorded',
                    f'{run.rel}: a {run.condition} run with no recorded source '
                    f'checkpoint', runs=[run.rel])
            continue

        src_dir = os.path.dirname(
            os.path.normpath(checkpoint.replace('\\', os.sep)))
        source = by_path.get(os.path.normcase(src_dir))
        if source is None and recorded_digest:
            source = by_digest.get(recorded_digest)
            if source is not None:
                rehomed.append(run.rel)
        if source is None:
            chk.add(ERROR, 'source_unresolvable',
                    f'{run.rel}: the recorded source resolves to neither a '
                    f'directory in this tree nor a known run_digest, so what '
                    f'this run loaded cannot be identified',
                    runs=[run.rel], checkpoint=checkpoint,
                    recorded_digest=recorded_digest)
            continue
        resolved += 1

        if recorded_digest and source.run_digest != recorded_digest:
            chk.add(ERROR, 'source_digest_mismatch',
                    f'{run.rel}: recorded source run_digest '
                    f'{str(recorded_digest)[:12]} != {str(source.run_digest)[:12]} '
                    f'at the checkpoint path; the manifest describes a different '
                    f'run from the one on disk',
                    runs=[run.rel, source.rel])
        if not recorded_digest:
            chk.add(ERROR, 'source_digest_unrecorded',
                    f'{run.rel}: no source run_digest was recorded, so the '
                    f'lineage rests on a path alone', runs=[run.rel])

        want_env = _norm('source_env', run.cfg.get('source_env'))
        got_env = _norm('env', source.cfg.get('env'))
        if want_env and got_env != want_env:
            chk.add(ERROR, 'source_env_mismatch',
                    f'{run.rel}: config.source_env is {want_env!r} but the '
                    f'source run at {source.rel} trained on {got_env!r}',
                    runs=[run.rel, source.rel])
        declared_env = _norm('source_env', run.get('source', 'source_env'))
        if want_env and declared_env and declared_env != want_env:
            chk.add(ERROR, 'source_env_manifest_mismatch',
                    f'{run.rel}: manifest source.source_env {declared_env!r} '
                    f'disagrees with config.source_env {want_env!r}',
                    runs=[run.rel])

        # A source from the wrong cell is the published defect verbatim.
        for name in ('arch', 'target_rule', 'aggregation', 'hidden',
                     'head_units'):
            mine = _norm(name, run.cfg.get(name))
            theirs = _norm(name, source.cfg.get(name))
            if mine != theirs:
                chk.add(ERROR, 'source_cell_mismatch',
                        f'{run.rel}: source {source.rel} differs in {name} '
                        f'({theirs!r} vs {mine!r}); a source from another cell '
                        f'or another network shape is not this cell\'s source',
                        runs=[run.rel, source.rel], field=name)

        # The checkpoint path must name the digest of the run it holds.
        parts = os.path.normpath(src_dir).replace(os.sep, '/').split('/')
        if len(parts) >= 2 and source.run_digest \
                and parts[-2] != source.run_digest[:12]:
            chk.add(ERROR, 'source_path_digest_mismatch',
                    f'{run.rel}: the source checkpoint sits in a directory named '
                    f'{parts[-2]} while the run there has digest '
                    f'{source.run_digest[:12]}',
                    runs=[run.rel, source.rel])
    if rehomed:
        chk.add(WARN, 'source_path_not_on_disk',
                f'{len(rehomed)} run(s) record a source checkpoint path that is '
                f'not in this tree; each was resolved by its recorded '
                f'run_digest instead, which is the only reason the lineage is '
                f'still checkable after the tree moved',
                runs=rehomed[:MAX_LISTED], n=len(rehomed))
    chk.detail['sources_resolved'] = resolved
    chk.detail['sources_resolved_by_digest'] = len(rehomed)
    return chk


def cross_check_tolerance(chk: Check) -> None:
    """Compare this file's copy of the DESIGN.md 3.1 tolerance with stats.py's.

    `DESIGN.md` 3.1 declares "a declared tolerance", singular, and names this
    file as the module that refuses on it. It is nevertheless written down
    twice: `FRACTION_TOLERANCE` here, which decides
    `intensity_confounded_unlabelled` below, and `stats.INTENSITY_TOLERANCE`,
    which decides whether the analysis will draw the same contrast. Neither
    module read the other and `validate.py` pins six `statlib`/`stats` pairs
    without pinning this one, so setting this copy to 0.50 left `28 passed, 0
    failed` while `stats.py` went on refusing at 0.05: two shipped artifacts
    giving opposite verdicts on one group, and nothing anywhere saying so.

    This is the same move `check_source_validity` makes on the validity gate,
    for the same reason and with the same severity: a verdict taken under a
    tolerance other than the declared one was taken under some other rule than
    the one the design states, whatever it claims. It does not remove the
    duplication -- the constant should live in `registry.py` beside
    `SOURCE_VALIDITY_GATE`, which both modules already import -- but it stops
    the duplication being silent.

    `stats.py` is imported here rather than at module scope: this file is the
    gate `stats.py` calls, and a module-level dependency on the module it gates
    would put the analysis on the audit's import path. An import that fails is
    reported, not passed over. A guard that could not be checked is not a guard
    that held, so it is a warning and `--strict` promotes it.
    """
    try:
        from experiments import stats as _stats            # noqa: PLC0415
    except Exception as exc:                               # noqa: BLE001
        chk.add(WARN, 'tolerance_cross_check_unavailable',
                f'the second copy of the DESIGN.md 3.1 tolerance lives in '
                f'stats.py and stats.py could not be imported ({exc!r}), so '
                f'the two were not compared. The refusal below was taken '
                f'against {FRACTION_TOLERANCE} and whether the analysis will '
                f'apply the same number is unknown',
                tolerance=FRACTION_TOLERANCE, error=str(exc))
        return
    other = getattr(_stats, 'INTENSITY_TOLERANCE', None)
    value = _as_float(other)
    if value is None:
        chk.add(WARN, 'tolerance_cross_check_unavailable',
                f'stats.py declares no readable INTENSITY_TOLERANCE '
                f'({other!r}), so the second copy of the DESIGN.md 3.1 '
                f'tolerance could not be compared with the '
                f'{FRACTION_TOLERANCE} declared here. Either the duplicate '
                f'was removed, in which case this check should be too, or it '
                f'was renamed, in which case it is unguarded again',
                tolerance=FRACTION_TOLERANCE, recorded=str(other))
        return
    chk.detail['tolerance_in_stats'] = value
    if abs(value - FRACTION_TOLERANCE) > GATE_TOLERANCE:
        chk.add(ERROR, 'tolerance_disagreement',
                f'DESIGN.md 3.1 declares one tolerance on the cross-arch '
                f'transferred-parameter fraction and the two copies of it '
                f'disagree: audit.FRACTION_TOLERANCE={FRACTION_TOLERANCE} '
                f'refuses the groups below, stats.INTENSITY_TOLERANCE={value} '
                f'decides whether the analysis draws the same contrast. One '
                f'of them is not the pre-registered rule, so every verdict in '
                f'this check and every intensity verdict in stats.py was taken '
                f'under a threshold the design does not state',
                tolerance=FRACTION_TOLERANCE, tolerance_in_stats=value,
                difference=round(abs(value - FRACTION_TOLERANCE), 12))


def check_transferred_fraction(membership, exps) -> Check:
    """Cross-architecture intensity: labelled where declared, refused where not.

    DESIGN.md 3.1 says the audit "refuses a cross-arch contrast whose fractions
    differ beyond a declared tolerance unless the contrast is explicitly
    labelled intensity-confounded". Emitting the label itself and warning was
    not a refusal, so nothing was ever refused. The label the design means is a
    catalogue declaration, and there is exactly one: an arm whose
    `transfer_set` is not `PROTOCOL['transfer_set']` is the deliberately
    unmatched comparison, and its intensity gap is the finding rather than a
    fault. A group that *claims* the matched protocol and does not have matched
    intensity is the published defect reconstituted, and is refused.

    The within-arm case is an error outright. The transferred fraction is fixed
    by the configuration, so it cannot legitimately differ between the seeds of
    one arm, and averaging over a disagreement is how one bad seed silently
    rewrote a printed gap from 0.034 to 0.315 and flipped the group's label.
    """
    chk = Check(
        'TRANSFERRED-FRACTION MATCHING',
        'the same layer list transfers 97 % of the mlp and 51 % of the dueling '
        'network, confounding arch with treatment intensity (DESIGN.md 3.1)')
    # A cross-architecture comparison is a pair of runs that agree on
    # everything the catalogue varies except `arch`. `aggregation` is excluded
    # from the key because it is architecture-specific by construction --
    # `Config` forces `mean` for mlp -- so requiring it to agree would leave no
    # mlp/dueling pair to compare at all.
    keys = tuple(k for k in DISCRIMINATING
                 if k not in ('arch', 'aggregation'))
    matched_set = registry.PROTOCOL.get('transfer_set')
    chk.detail['matched_protocol_transfer_set'] = matched_set
    chk.detail['tolerance'] = FRACTION_TOLERANCE
    cross_check_tolerance(chk)
    # Every group that claims the matched protocol, with how far it sits from
    # the refusal. Reported below, because the closest one is 0.008 away.
    headroom: list[tuple[float, str, str, dict]] = []
    for eid, arms in membership.items():
        groups: dict[tuple, dict[str, list[tuple[str, float]]]] = defaultdict(
            lambda: defaultdict(list))
        for label, group in arms.items():
            for run in group:
                if run.condition == 'scratch':
                    continue
                raw = run.get('transfer', 'summary',
                              'fraction_of_model_transferred')
                frac = _as_float(raw)
                if raw is None:
                    chk.add(ERROR, 'fraction_unrecorded',
                            f'{run.rel}: a {run.condition} run with no '
                            f'transferred-parameter fraction; the intensity of '
                            f'the treatment it received is unknown',
                            runs=[run.rel])
                    continue
                if frac is None:
                    chk.add(ERROR, 'fraction_not_finite',
                            f'{run.rel}: the transferred-parameter fraction is '
                            f'{raw!r}, which is not a fraction; the intensity '
                            f'of the treatment it received is unknown',
                            runs=[run.rel], value=str(raw))
                    continue
                sig = _run_signature(run)
                groups[tuple(sig[k] for k in keys)][str(sig['arch'])].append(
                    (run.rel, frac))

        # Name each group by the fields that actually differ between the groups
        # of this experiment, so the printed table identifies a row without
        # carrying eighteen columns of constants.
        varying = tuple(k for i, k in enumerate(keys)
                        if len({g[i] for g in groups}) > 1)
        for key, by_arch in sorted(groups.items(), key=lambda kv: str(kv[0])):
            described = dict(zip(keys, key))
            unstable: list[str] = []
            for arch, entries in sorted(by_arch.items()):
                spread = max(f for _, f in entries) - min(f for _, f in entries)
                if spread > 1e-9:
                    unstable.append(arch)
                    chk.add(ERROR, 'fraction_varies_within_arm',
                            f'{eid}: {arch} runs at one configuration report '
                            f'transferred fractions spanning {spread:.4f}, so '
                            f'the treatment intensity is not constant across '
                            f'seeds. The fraction is fixed by the '
                            f'configuration; a spread means the runs did not '
                            f'all receive the arm they are labelled with, and '
                            f'the group mean below is an average over that '
                            f'disagreement rather than a measurement. Per seed: '
                            + ', '.join(f'{r}={f:.4f}'
                                        for r, f in sorted(entries)),
                            runs=[r for r, _ in entries][:MAX_LISTED],
                            experiment=eid, arch=arch,
                            spread=round(spread, 6),
                            per_run={r: round(f, 6) for r, f in entries})
            if len(by_arch) < 2:
                continue
            means = {arch: sum(f for _, f in e) / len(e)
                     for arch, e in by_arch.items()}
            gap = max(means.values()) - min(means.values())
            name = ' '.join(f'{k}={described[k]}' for k in varying) or 'all runs'
            # The catalogue's own label. `transfer_set` is the field DESIGN.md
            # 3.1 matches intensity with, so a group that does not carry the
            # matched protocol is a declared unmatched contrast, and one that
            # does is claiming intensity it must then actually have.
            observed_set = described.get('transfer_set')
            declared_unmatched = (matched_set is not None
                                  and observed_set is not None
                                  and observed_set != matched_set)
            confounded = gap > FRACTION_TOLERANCE
            record = {'experiment': eid, 'group': name,
                      'condition': described.get('condition'),
                      'transfer_set': observed_set,
                      'fractions': {a: round(f, 4) for a, f in means.items()},
                      'gap': round(gap, 4),
                      'n_runs': sum(len(e) for e in by_arch.values())}
            chk.detail.setdefault('cross_arch_groups', []).append(
                {**record, 'intensity_confounded': confounded,
                 'declared_unmatched': declared_unmatched,
                 'fractions_not_constant': sorted(unstable),
                 'per_run': {a: {r: round(f, 6) for r, f in sorted(e)}
                             for a, e in sorted(by_arch.items())}})
            if not declared_unmatched:
                headroom.append((FRACTION_TOLERANCE - gap, eid, name, record))
            if not confounded:
                continue
            fractions = ', '.join(f'{a}={f:.3f}'
                                  for a, f in sorted(means.items()))
            if declared_unmatched:
                # The licensed case, and the label DESIGN.md 3.1 requires the
                # contrast to carry. E1's and E5's `transfer_set=trunk` arms are
                # the deliberately unmatched comparison; the gap is what they
                # are for, and the warning is how the reader is told.
                chk.add(WARN, 'intensity_confounded',
                        f'{eid}: the cross-arch group [{name}] has fractions '
                        f'{fractions} -- a gap of {gap:.3f} beyond the '
                        f'{FRACTION_TOLERANCE} tolerance. This group declares '
                        f'transfer_set={observed_set!r} rather than the matched '
                        f'protocol {matched_set!r}, so it is the catalogue\'s '
                        f'own unmatched contrast and DESIGN.md 3.1 permits it '
                        f'labelled. Any architecture contrast drawn here is '
                        f'intensity-confounded and must be presented so',
                        **record, declared_unmatched=True)
            else:
                chk.add(ERROR, 'intensity_confounded_unlabelled',
                        f'{eid}: the cross-arch group [{name}] declares the '
                        f'matched protocol transfer_set={observed_set!r} and '
                        f'has fractions {fractions} -- a gap of {gap:.3f} '
                        f'beyond the {FRACTION_TOLERANCE} tolerance. The '
                        f'matching this group claims did not happen, so arch is '
                        f'confounded with treatment intensity here exactly as '
                        f'in the published study, with nothing in the catalogue '
                        f'declaring it. DESIGN.md 3.1 refuses such a contrast '
                        f'unless it is explicitly labelled: either match the '
                        f'layer sets, or declare the arm unmatched',
                        **record, declared_unmatched=False)

    # How close the catalogue itself sits to its own gate. The tolerance was
    # written when this was a warning and it is now a refusal, and the smallest
    # margin in the catalogue is E8i's matched groups at a 0.042 gap against
    # 0.050: the interface-shift experiment, whose subject is varying pad_obs
    # and extra_actions, which is exactly what moves a parameter count and so
    # the fraction. Reported unconditionally, because the alternative is that a
    # maintainer meets it as an audit failure after the change rather than as a
    # margin before it. Recording it does not move the gate: the refusal above
    # is unchanged and no group is exempted by appearing here.
    if headroom:
        margin, eid, name, record = min(headroom, key=lambda h: h[0])
        chk.detail['smallest_matched_headroom'] = {
            'experiment': eid, 'group': name, 'gap': record['gap'],
            'tolerance': FRACTION_TOLERANCE, 'headroom': round(margin, 6),
            'groups_claiming_the_matched_protocol': len(headroom)}
        chk.add(NOTE, 'matched_intensity_headroom',
                f'{len(headroom)} cross-arch group(s) claim the matched '
                f'protocol; the closest to refusal is {eid} [{name}] at a gap '
                f'of {record["gap"]:.4f} against the {FRACTION_TOLERANCE} '
                f'tolerance, {margin:.4f} from intensity_confounded_unlabelled '
                f'and a blocked report. DESIGN.md 3.1 makes that tolerance a '
                f'gate rather than a caption, so the best matched protocol '
                f'the design achieves has {margin:.4f} of room: a change to '
                f'the padded interface that widens the gap by that much turns '
                f'the matched contrast the catalogue declares into an audit '
                f'error rather than the labelled warning 3.1 asks for',
                experiment=eid, group=name, gap=record['gap'],
                tolerance=FRACTION_TOLERANCE, headroom=round(margin, 6),
                n_matched_groups=len(headroom))
    return chk


def check_plan_hash(runs: list[Run]) -> Check:
    chk = Check(
        'PLAN HASH',
        'a confirmatory result is interpretable only against the plan in force '
        'when it ran (ANALYSIS_PLAN.md 1)')
    current = provenance.plan_hashes()
    chk.detail['current'] = current
    for name in ('ANALYSIS_PLAN.md', 'DESIGN.md', 'reference_returns.json'):
        seen: dict[Any, list[str]] = defaultdict(list)
        for run in runs:
            seen[run.get('provenance', 'plans', name)].append(run.rel)
        chk.detail.setdefault('observed', {})[name] = {
            str(h): len(rs) for h, rs in seen.items()}
        if not seen:
            continue
        level = ERROR if name == 'ANALYSIS_PLAN.md' else WARN
        if len(seen) > 1:
            chk.add(level, 'plan_hash_split',
                    f'{name}: runs were produced under {len(seen)} different '
                    f'versions of this document, so they are not one '
                    f'pre-registered set',
                    runs=[rs[0] for rs in seen.values()][:MAX_LISTED],
                    document=name,
                    hashes={str(h): len(rs) for h, rs in seen.items()})
        stale = {h: rs for h, rs in seen.items() if h != current.get(name)}
        if stale:
            chk.add(level, 'plan_hash_stale',
                    f'{name} has changed since '
                    f'{sum(len(rs) for rs in stale.values())} '
                    f'run(s) were produced; under ANALYSIS_PLAN.md 1 the '
                    f'affected results are exploratory until the change is '
                    f'recorded in its 11',
                    runs=[rs[0] for rs in stale.values()][:MAX_LISTED],
                    document=name, current=current.get(name),
                    stored=[str(h) for h in stale])
    return chk


#: The provenance record DESIGN.md 8.3 requires, as (manifest path, why it is
#: there). Presence is checked, not content: a provenance block that is half
#: absent is not a reproducibility record, and the half that is missing is
#: never the half a reader thinks to look for.
PROVENANCE_REQUIRED: tuple[tuple[tuple[str, ...], str], ...] = (
    (('provenance', 'packages'),
     'the package versions the run was produced under; without them a rerun '
     'reproduces the code and not the numerics'),
    (('provenance', 'machine'),
     'the platform and CPU, which is what a difference in float behaviour is '
     'traced to'),
    (('provenance', 'argv'),
     'the exact invocation, which is the only record of what was asked for as '
     'opposed to what the catalogue declares'),
    (('seeds', 'streams'),
     'the derived seed per RNG stream; DESIGN.md 8.1 gives each role its own '
     'stream so an ablation cannot perturb machinery it does not touch, and '
     'the derivation is what makes that checkable'),
)


def check_provenance(runs: list[Run]) -> Check:
    chk = Check(
        'PROVENANCE',
        'a result produced from an uncommitted tree is not reproducible from '
        'the repository, and a half-recorded provenance block is not a '
        'reproducibility record (DESIGN.md 8.3)')
    # Aggregated per field: a provenance field goes missing because the writer
    # changed, which moves every run at once.
    for path, why in PROVENANCE_REQUIRED:
        absent = [r.rel for r in runs if r.get(*path) is None]
        name = '.'.join(path)
        chk.detail.setdefault('required_present', {})[name] = \
            len(runs) - len(absent)
        if absent:
            chk.add(ERROR, 'provenance_field_absent',
                    f'{len(absent)} of {len(runs)} run(s) record no {name}. '
                    f'DESIGN.md 8.3 requires it: {why}',
                    runs=absent[:MAX_LISTED], field=name, n=len(absent))
    dirty = [r.rel for r in runs if r.get('provenance', 'git', 'dirty') is True]
    unknown = [r.rel for r in runs
               if r.get('provenance', 'git', 'dirty') is None]
    commits = Counter(str(r.get('provenance', 'git', 'commit'))[:12]
                      for r in runs)
    chk.detail.update({'runs': len(runs), 'dirty': len(dirty),
                       'git_state_unknown': len(unknown),
                       'commits': dict(commits)})
    if dirty:
        chk.add(WARN, 'dirty_tree',
                f'{len(dirty)} of {len(runs)} run(s) were produced from a dirty '
                f'git tree and are not reproducible from the repository alone',
                runs=dirty[:MAX_LISTED], n=len(dirty))
    if unknown:
        chk.add(WARN, 'git_state_unknown',
                f'{len(unknown)} run(s) recorded no git state at all',
                runs=unknown[:MAX_LISTED], n=len(unknown))
    if len(commits) > 1:
        chk.add(NOTE, 'multiple_commits',
                f'runs span {len(commits)} commits '
                f'({", ".join(sorted(commits))}); the code changed during the '
                f'campaign', commits=dict(commits))
    return chk


#: Fields a tuning selection fixes for a cell. Read from `CellConfig` rather
#: than restated, so a field added to the selected configuration is compared
#: here without this file being edited: a selected field nobody compares is a
#: selected field a run may silently disagree with.
SELECTED_FIELDS: tuple[str, ...] = tuple(
    sorted(tuning.A_PRIORI_CONFIG.to_dict()))


def _selected_mismatch(run: Run, want) -> list[tuple]:
    """(field, recorded, selected) for each selected field the run disagrees on."""
    out: list[tuple] = []
    expect = want.to_dict()
    for name in SELECTED_FIELDS:
        got = run.cfg.get(name, CONFIG_DEFAULTS.get(name))
        target = expect[name]
        if isinstance(target, float):
            value = _as_float(got)
            if value is not None and math.isclose(value, float(target),
                                                  rel_tol=1e-9, abs_tol=0.0):
                continue
        elif _norm(name, got) == _norm(name, target):
            continue
        out.append((name, got, target))
    return out


def _archived_selections(out_root: str) -> dict:
    """Every selection archived under this tree, by id. Unreadable ones skipped.

    The archive is what makes a mismatch diagnosable rather than merely
    detectable: a tuned run that disagrees with the active selection but agrees
    with a superseded one was enumerated from that superseded one, which is a
    different defect from a hand-edited configuration and wants a different
    repair.
    """
    out: dict = {}
    pattern = os.path.join(glob.escape(os.path.abspath(out_root)),
                           *tuning.SELECTION_ARCHIVE_RELDIR.split(os.sep),
                           '*.json')
    for path in sorted(glob.glob(pattern)):
        try:
            with open(path, encoding='utf-8') as fh:
                out[os.path.splitext(os.path.basename(path))[0]] = \
                    tuning.Selection.from_dict(json.load(fh))
        except Exception:                                   # noqa: BLE001
            continue
    return out


def check_tuning_provenance(runs: list[Run], out_root: str) -> Check:
    """The secondary policy's runs against the selection they claim to execute.

    `DESIGN.md` 3.3 executes the per-cell-tuned policy as `E1t` and `E2t`, whose
    arms are a function of one artifact: the tuning selection stored at
    `<out_root>/_jobs/tuning_selection.json`. That makes the artifact a
    pre-registration record with the same standing as the plan hash, and it
    fails in the same two ways. It can be *edited* after the runs were
    enumerated from it, in which case the runs on disk belong to arms that no
    longer exist; or it can be *replaced*, in which case the old runs stay in
    the tree, no arm declares them, and the new arms are reported missing while
    the old ones read as an unattributed pile. Neither is visible to any other
    check here: a superseded tuned run has a consistent digest, a valid
    manifest and a plan hash that matches.

    What is compared, and what is not
    ---------------------------------
    A run's manifest records no selection id. It records the `experiment` and
    `label` the runner was given and the configuration it trained under, and
    the selection is precisely what determines that configuration for a tuned
    arm, so the comparison here is over the selected fields: a run recorded
    under `E1t`/`E2t`, or carrying the tuned label prefix, must hold exactly the
    configuration the artifact on disk selects for its cell. Where it does not,
    the archived selections under `_jobs/selections/` are searched for one it
    *does* match, so the finding says which selection the run came from rather
    than only that it disagrees with this one.

    That is a reconstruction, and it is weaker than reading a recorded id in
    one specific way: two selections that agree on a cell are indistinguishable
    for that cell's runs. They are also indistinguishable in the runs
    themselves, because the digest is a function of the configuration and such
    runs are literally the same runs, so the reconstruction is exact wherever
    two selections differ, which is the only place a difference can exist. A
    stronger check would read a selection id written into the manifest by the
    runner, which is a field in `src/dqn/provenance.py` and not this file's to
    add.

    Source runs are held to the primary policy on purpose. `DESIGN.md` 3.3
    scopes the secondary policy to the target task, `tuned_experiment` leaves
    the CartPole source arms unretuned and unrelabelled, and a source run that
    had drifted onto a selected learning rate would mean the two policies
    differed in the source as well as in the arm under study, so a disagreement
    between them could no longer be read as being about the tuning policy.
    """
    chk = Check(
        'TUNING SELECTION',
        'the tuned arms of DESIGN.md 3.3 are a function of one stored '
        'artifact, so an edited or a replaced selection silently re-points '
        'every run enumerated from it and no other check here can see that')
    prefix = registry.TUNED_LABEL_PREFIX
    tuned_ids = set(registry.TUNED_OF)
    marked = [r for r in runs
              if str(r.cfg.get('experiment') or '') in tuned_ids
              or str(r.cfg.get('label') or '').startswith(prefix)]
    active = tuning.selection_path(out_root)
    chk.detail.update({
        'artifact': active,
        'tuned_experiments': sorted(tuned_ids),
        'tuned_arms_in_catalogue': registry.tuned_arms_active(),
        'runs_recorded_as_tuned': len(marked),
        'selected_fields': list(SELECTED_FIELDS)})

    selection = None
    try:
        selection = tuning.read_selection(out_root, required=False,
                                          verify=True, warn_placeholder=False)
    except Exception as exc:                                # noqa: BLE001
        # Includes `SelectionCorrupt`, raised when the stored artifact does not
        # hash to the id it carries -- that is, when it was edited after the
        # arms were enumerated from it. An error whether or not tuned runs
        # exist yet, because the file is what the next enumeration reads.
        chk.add(ERROR, 'selection_unreadable',
                f'{active} exists and cannot be read as a selection artifact '
                f'({type(exc).__name__}: {exc}). The tuned arms of DESIGN.md '
                f'3.3 are enumerated from it, so nothing can say which arms '
                f'the runs on disk belong to while it is in this state. '
                f'Recompute it with experiments/tuning.py rather than '
                f'repairing the file.',
                runs=[r.rel for r in marked[:MAX_LISTED]])
        return chk

    if selection is None:
        if marked:
            chk.add(ERROR, 'tuned_runs_without_selection',
                    f'{len(marked)} run(s) are recorded under the tuned '
                    f'experiments {sorted(tuned_ids)} or carry the '
                    f'{prefix!r} label prefix, and there is no selection at '
                    f'{active}. Those runs execute a per-cell configuration '
                    f'that nothing on disk declares, so no arm claims them and '
                    f'the tuning policy they ran under is unrecoverable.',
                    runs=[r.rel for r in marked[:MAX_LISTED]], n=len(marked))
        else:
            chk.add(NOTE, 'tuning_stage_not_started',
                    f'no tuning selection at {active} and no run recorded '
                    f'under {sorted(tuned_ids)}. The secondary policy of '
                    f'DESIGN.md 3.3 is sequentially dependent on E3 and has '
                    f'not been enumerated, so every comparison below made 0 '
                    f'comparisons. The check is in force and enforces nothing '
                    f'yet, and this file says so rather than counting it as a '
                    f'guardrail (DESIGN.md 11 defect 12).')
        return chk

    chk.detail.update({
        'selection_id': selection.selection_id,
        'rule': str(selection.rule.get('id')),
        'rule_is_placeholder': bool(selection.is_placeholder),
        'seed_block': selection.seed_block,
        'seeds': list(selection.seeds),
        'env': selection.env,
        'cells': {k: v.config_key for k, v in sorted(selection.cells.items())},
        'cells_sharing_common_policy_runs': list(selection.shared_cells)})

    # -- the artifact against the design -----------------------------------
    block = tuple(registry.SEED_BLOCKS[tuning.SELECTION_SEED_BLOCK])
    if (selection.seed_block != tuning.SELECTION_SEED_BLOCK
            or tuple(selection.seeds) != block):
        chk.add(ERROR, 'selection_seed_block_invalid',
                f'the selection declares block {selection.seed_block!r} at '
                f'seeds {list(selection.seeds)}; DESIGN.md 3.4 reserves '
                f'{tuning.SELECTION_SEED_BLOCK} {list(block)} for '
                f'hyperparameter selection and ANALYSIS_PLAN.md 8 forbids a '
                f'reported estimate drawing on the selection block. A '
                f'selection made anywhere else was made on the seeds the '
                f'confirmatory arms are estimated from.')
    missing_cells = sorted(registry._tag(*c) for c in registry.CELLS
                           if registry._tag(*c) not in selection.cells)
    if missing_cells:
        chk.add(ERROR, 'selection_cells_incomplete',
                f'the stored selection covers {sorted(selection.cells)} and '
                f'not {missing_cells}. ANALYSIS_PLAN.md 2.3 refuses an '
                f'incomplete selection: the uncovered cells would run under '
                f'the primary policy while the artifact claimed the secondary '
                f'one, and the DESIGN.md 3.3 arbitration would be asserted '
                f'over cells that were never retuned.')
    if selection.env != registry.TARGET_ENV:
        chk.add(WARN, 'selection_env_not_target',
                f'the selection was made on {selection.env} and the target '
                f'task is {registry.TARGET_ENV}; the tuned arms retune only '
                f'the runs on the environment the selection names, so every '
                f'target-task arm would still be running the primary policy',
                selection_env=selection.env, target_env=registry.TARGET_ENV)

    archive = tuning.archive_path(selection.selection_id, out_root)
    if not os.path.exists(archive):
        chk.add(ERROR, 'selection_archive_absent',
                f'the active selection {selection.short_id} has no immutable '
                f'copy at {archive}. The archive is what a replaced active '
                f'pointer is diagnosed against; without it a superseded tuned '
                f'run cannot be attributed to the selection it was enumerated '
                f'from.')
    else:
        try:
            with open(archive, encoding='utf-8') as fh:
                stored = tuning.Selection.from_dict(json.load(fh))
            archive_agrees = (stored.computed_id() == selection.selection_id)
        except Exception:                                   # noqa: BLE001
            archive_agrees = False
        if not archive_agrees:
            chk.add(ERROR, 'selection_archive_mismatch',
                    f'{archive} is named for {selection.short_id} and does not '
                    f'hash to it. One of the two copies of this selection was '
                    f'edited, so the content-addressed record no longer says '
                    f'what the tuned runs were enumerated from.')

    if selection.is_placeholder:
        chk.add(ERROR if marked else WARN, 'selection_rule_is_placeholder',
                f'selection {selection.short_id} was computed under '
                f'{selection.rule.get("id")}, which declares itself a '
                f'PLACEHOLDER and not the pre-registered criterion of '
                f'ANALYSIS_PLAN.md 2.3. '
                + (f'{len(marked)} tuned run(s) already exist under it, so the '
                   f'secondary leg of the DESIGN.md 3.3 arbitration was run '
                   f'against a rule the plan does not contain and no number '
                   f'from it is a result.'
                   if marked else
                   'No tuned run exists yet, so this is a warning about what '
                   'would be launched rather than about what was.'),
                runs=[r.rel for r in marked[:MAX_LISTED]],
                rule=str(selection.rule.get('id')))

    current = provenance.plan_hashes()
    stale = {name: (stored_hash, current.get(name))
             for name, stored_hash in sorted(selection.plans.items())
             if stored_hash is not None and stored_hash != current.get(name)}
    if stale:
        chk.add(ERROR if marked else WARN, 'selection_plans_stale',
                f'the selection was computed against {sorted(stale)} as they '
                f'stood then, and that has changed since. ANALYSIS_PLAN.md 2.3 '
                f'is the rule this artifact claims to have applied, so a '
                f'selection whose plan hash no longer matches was made under a '
                f'criterion the current plan may not contain',
                documents={k: {'selection': v[0], 'current': v[1]}
                           for k, v in stale.items()})
    unrecorded = sorted(k for k, v in selection.plans.items() if v is None)
    if unrecorded:
        chk.add(NOTE, 'selection_plans_unrecorded',
                f'the selection records no hash for {unrecorded}, so it cannot '
                f'be tied to the plan in force when it was computed',
                documents=unrecorded)

    # -- the evidence the selection names, against the tree -----------------
    on_disk = {str(r.run_digest) for r in runs if r.run_digest}
    named = sorted({d for cell in selection.cells.values()
                    for d in cell.run_digests})
    absent = [d for d in named if d not in on_disk]
    chk.detail['evidence_runs'] = {'named': len(named),
                                   'present_in_tree': len(named) - len(absent)}
    if named and absent:
        chk.add(WARN, 'selection_evidence_absent',
                f'{len(absent)} of {len(named)} E3 run(s) the selection was '
                f'computed on are not in this tree. The selection stays valid, '
                f'being a function of the aggregated table rather than of the '
                f'directories, but it cannot be recomputed here, so the rule '
                f'that produced it cannot be re-derived from this tree alone',
                digests=absent[:MAX_LISTED], n=len(absent))

    # -- every run recorded as tuned, against the artifact ------------------
    if marked and not registry.tuned_arms_active():
        chk.add(WARN, 'tuned_arms_not_activated',
                f'{len(marked)} run(s) are recorded under {sorted(tuned_ids)} '
                f'and this process did not activate the tuned arms, so the '
                f'catalogue does not declare them and their completeness, seed '
                f'blocks and lineage were measured against nothing. Call '
                f'registry.activate_tuned_arms(out_root=...) before auditing a '
                f'tree that holds them.',
                runs=[r.rel for r in marked[:MAX_LISTED]], n=len(marked))

    archived = _archived_selections(out_root)
    mismatched: dict = {}
    off_env: list[str] = []
    unmarked: list[str] = []
    stray: list[str] = []
    unknown_cell: list[str] = []
    compared = 0
    for run in marked:
        eid = str(run.cfg.get('experiment') or '')
        label = str(run.cfg.get('label') or '')
        env = _norm('env', run.cfg.get('env'))
        cell_key = run.cell
        if eid not in tuned_ids:
            stray.append(run.rel)
            continue
        if env == _norm('env', registry.SOURCE_ENV):
            want = tuning.A_PRIORI_CONFIG            # sources are not retuned
            if label.startswith(prefix):
                unmarked.append(run.rel)
        elif env == _norm('env', selection.env):
            if cell_key not in selection.cells:
                unknown_cell.append(run.rel)
                continue
            want = selection.config_for(cell_key)
            if not label.startswith(prefix):
                unmarked.append(run.rel)
        else:
            off_env.append(run.rel)
            continue
        compared += 1
        for name, got, expect in _selected_mismatch(run, want):
            entry = mismatched.setdefault((cell_key, name), {
                'recorded': set(), 'selected': str(expect), 'runs': []})
            entry['recorded'].add(str(got))
            entry['runs'].append(run.rel)
    chk.detail['runs_compared_against_selection'] = compared

    for (cell_key, name), entry in sorted(mismatched.items()):
        culprits = sorted(
            sid for sid, other in archived.items()
            if sid != selection.selection_id and cell_key in other.cells
            and str(other.config_for(cell_key).to_dict().get(name))
            in entry['recorded'])
        chk.add(ERROR, 'tuned_selection_mismatch',
                f'{len(entry["runs"])} tuned run(s) in {cell_key} recorded '
                f'{name}={sorted(entry["recorded"])} and selection '
                f'{selection.short_id} selects {entry["selected"]} for that '
                f'cell. A tuned arm is a function of the selection, so these '
                f'runs were enumerated from a different one'
                + (f'; they match the archived selection(s) '
                   f'{[s[:12] for s in culprits]}, which the active pointer '
                   f'has replaced. Every one of them is now claimed by no arm '
                   f'while the arms the current selection declares are '
                   f'reported missing.'
                   if culprits else
                   '. No archived selection matches them either, so the '
                   'configuration was not produced by any selection this tree '
                   'has stored.'),
                runs=entry['runs'][:MAX_LISTED], cell=cell_key, field=name,
                recorded=sorted(entry['recorded']),
                selected=entry['selected'], n=len(entry['runs']),
                matches_archived=[s[:12] for s in culprits])
    if off_env:
        chk.add(ERROR, 'tuned_run_off_selection_env',
                f'{len(off_env)} tuned run(s) are on neither the environment '
                f'the selection was made on ({selection.env}) nor the source '
                f'environment ({registry.SOURCE_ENV}). Nothing in DESIGN.md '
                f'3.3 says what a selection made on one task means for a run '
                f'on a third, and `registry.tuned_experiment` refuses to build '
                f'such an arm, so these runs were not produced by the '
                f'catalogue.',
                runs=off_env[:MAX_LISTED], n=len(off_env))
    if unmarked:
        chk.add(ERROR, 'tuned_arm_label_unmarked',
                f'{len(unmarked)} run(s) recorded under {sorted(tuned_ids)} '
                f'carry a label that contradicts what they are: a retuned '
                f'target-task arm must carry the {prefix!r} prefix and an '
                f'unretuned source arm must not. plots.py and the aggregated '
                f'tables select rows by label, so a mislabelled run merges the '
                f'two policies where they are genuinely different runs, which '
                f'is the one thing the prefix exists to prevent.',
                runs=unmarked[:MAX_LISTED], n=len(unmarked))
    if stray:
        chk.add(ERROR, 'tuned_label_outside_tuned_experiment',
                f'{len(stray)} run(s) carry the {prefix!r} label prefix and '
                f'record an experiment that is not one of {sorted(tuned_ids)}. '
                f'The prefix is what separates the two policies of DESIGN.md '
                f'3.3 in every table that groups by label, so a common-policy '
                f'run wearing it is counted as the secondary policy having '
                f'been run when it has not.',
                runs=stray[:MAX_LISTED], n=len(stray))
    if unknown_cell:
        chk.add(ERROR, 'tuned_run_outside_selection_cells',
                f'{len(unknown_cell)} tuned run(s) sit in a cell the selection '
                f'does not cover, so no configuration was ever selected for '
                f'what they ran',
                runs=unknown_cell[:MAX_LISTED], n=len(unknown_cell))
    if not marked:
        chk.add(NOTE, 'no_tuned_runs_yet',
                f'selection {selection.short_id} is stored and no run in this '
                f'tree is recorded under {sorted(tuned_ids)}. The artifact was '
                f'checked; the per-run comparison made 0 comparisons.')
    return chk


def check_reference_coverage(runs: list[Run]) -> Check:
    chk = Check(
        'REFERENCE COVERAGE',
        'a missing reference return would put one variant\'s scores on a '
        'different scale from every other\'s (DESIGN.md 5.1)')
    envs_seen: Counter = Counter()
    # Aggregated by environment: a missing or a moved reference affects
    # every run on that environment at once, and one line per run would
    # say the same thing two hundred times.
    missing: dict[str, list[str]] = defaultdict(list)
    missing_why: dict[str, str] = {}
    drift: dict[tuple, list[str]] = defaultdict(list)
    drift_values: dict[tuple, tuple] = {}
    for run in runs:
        spec = run.cfg.get('env')
        envs_seen[str(spec)] += 1
        try:
            ref = envs.reference(spec)
        except Exception as exc:                            # noqa: BLE001
            missing[str(spec)].append(run.rel)
            missing_why[str(spec)] = str(exc)
            continue
        stored = run.manifest.get('reference') or {}
        if not stored:
            chk.add(ERROR, 'reference_unrecorded',
                    f'{run.rel}: the run recorded no normalisation constants, '
                    f'so its scores cannot be recomputed', runs=[run.rel])
            continue
        for key in ('random_return', 'threshold'):
            was, now = stored.get(key), ref.get(key)
            drifted = (was is None or now is None
                       or abs(float(was) - float(now))
                       > REFERENCE_TOLERANCE)
            if drifted:
                drift[(str(spec), key)].append(run.rel)
                drift_values[(str(spec), key)] = (was, now)
        try:
            es = envs.parse(spec)
        except Exception:                                   # noqa: BLE001
            continue
        for key, want in (('obs_dim', es.obs_dim), ('act_dim', es.act_dim)):
            got = (run.manifest.get('env') or {}).get(key)
            if got is not None and int(got) != int(want):
                chk.add(ERROR, 'interface_mismatch',
                        f'{run.rel}: recorded {key}={got} but {spec} has '
                        f'{key}={want}; the run did not train on the interface '
                        f'its config names', runs=[run.rel])
    for spec, rels in sorted(missing.items()):
        chk.add(ERROR, 'reference_missing',
                f'{len(rels)} run(s) on {spec}: {missing_why[spec]}',
                runs=rels[:MAX_LISTED], env=spec, n=len(rels))
    for (spec, key), rels in sorted(drift.items()):
        was, now = drift_values[(spec, key)]
        chk.add(ERROR, 'reference_drift',
                f'{len(rels)} run(s) on {spec} were normalised against '
                f'{key}={was} but the committed reference is now {now}; '
                f'their scores are on a different scale from a freshly '
                f'measured run, and the two must not be pooled',
                runs=rels[:MAX_LISTED], env=spec, key=key, stored=was,
                current=now, n=len(rels))
    chk.detail['environments'] = dict(envs_seen)
    return chk


def check_attribution(runs: list[Run], membership, selected,
                      orphans: list[Run], everywhere: dict[str, set[str]],
                      declared: Declared, overrides: dict) -> Check:
    """Whether every run on disk is a run the catalogue accounts for."""
    chk = Check(
        'RUN ATTRIBUTION',
        'a run that no declared arm accounts for cannot be reported, and '
        'cannot be excluded by a glob either')
    claimed_by_selection = {r.rel for arms in membership.values()
                            for group in arms.values() for r in group}
    counts: Counter = Counter()
    for eids in everywhere.values():
        for eid in eids:
            counts[eid] += 1
    chk.detail.update({
        'runs': len(runs),
        'claimed_by_selected_experiments': len(claimed_by_selection),
        'membership_counts': dict(sorted(counts.items())),
        'shared_runs': sum(1 for eids in everywhere.values() if len(eids) > 1),
        'overrides_declared': {k: str(v) for k, v in sorted(overrides.items())},
    })
    if orphans:
        reasons: dict[str, list[str]] = defaultdict(list)
        known_exps = set(registry.EXPERIMENTS)
        known_labels = {label for _eid, label in declared.arms}
        for run in orphans:
            eid = str(run.cfg.get('experiment'))
            label = str(run.cfg.get('label'))
            if run.seed is None:
                # Named separately because it is the one reason that is a
                # property of the manifest rather than of the catalogue: a run
                # whose seed cannot be read cannot be placed at an arm x seed
                # slot, so no arm can claim it however well the rest matches.
                reasons[f'config.seed is '
                        f'{run.cfg.get("seed")!r}, which is not a seed, so the '
                        f'run cannot be placed at any arm x seed slot'].append(
                            run.rel)
            elif eid not in known_exps:
                reasons['experiment not in the catalogue'].append(run.rel)
            elif label not in known_labels:
                reasons['label not an arm of any experiment'].append(run.rel)
            else:
                reasons['arm exists but is not scheduled at this seed'].append(
                    run.rel)
        for reason, rels in sorted(reasons.items()):
            # An error, as this check's own `why` says it must be: a run no
            # declared arm accounts for cannot be reported and cannot be
            # excluded by a glob either, so leaving the audit green on it means
            # an ad hoc or hand-labelled run sits in the tree and everything
            # downstream that globs will find it.
            chk.add(ERROR, 'unattributed_runs',
                    f'{len(rels)} run(s): {reason}. Either they are ad hoc, or '
                    f'the catalogue has moved under them; either way no '
                    f'experiment in it accounts for them, so they can neither '
                    f'be reported nor reliably excluded downstream',
                    runs=rels[:MAX_LISTED], n=len(rels), reason=reason)
    outside = sorted(set(r.rel for r in runs) - claimed_by_selection
                     - {r.rel for r in orphans})
    if outside:
        chk.add(NOTE, 'outside_selection',
                f'{len(outside)} run(s) belong to catalogue experiments outside '
                f'the selection {sorted(selected)}', n=len(outside))
    if overrides:
        chk.add(NOTE, 'overrides_declared',
                f'the declared configuration was resolved with '
                f'{len(overrides)} launch-level override(s) supplied on the '
                f'command line '
                f'({", ".join(f"{k}={v}" for k, v in sorted(overrides.items()))}); '
                f'arm-level values still win, exactly as in registry.jobs',
                overrides={k: str(v) for k, v in overrides.items()})
    return chk


def check_seed_selection(seeds, declared: "Declared | None" = None,
                         membership=None, n_runs: int = 0,
                         experiments: Iterable[str] | None = None
                         ) -> list[Finding]:
    """Refuse a selection before it can empty the inventory it is judged on.

    `registry.resolve_seeds('')` returns `()`, which is a legal answer to a
    legal question and a disaster here: with no target pair for any experiment,
    `observed & pairs` is empty everywhere, every completeness assertion is true
    of nothing, and the audit prints a pass. On the real tree `--seeds ""` took
    nineteen errors to zero without touching a run.

    Emptiness, though, is only the loudest shape of that failure, and closing
    it alone left the quieter one open. `--seeds 999` resolves to a perfectly
    good non-empty tuple, so the guard above passes it, and it then names arm x
    seed slots nobody ran: `check_seed_completeness` reaches
    `if not (observed & pairs)`, reports `experiment_not_run` as a warning and
    continues, skipping `arm_absent` and `seeds_missing` entirely, while
    `check_source_validity`'s reserve loop skips every run because
    `(label, run.seed) not in pairs`. Measured, on a tree brought current in
    every other respect: 25 errors at the declared blocks, 0 errors and a
    rendered "AUDIT PASS" at `--seeds 999`, with the PIPELINE VALIDATION stamp
    gone from all twelve inventory rows. It is not an out-of-block trick
    either: `--seeds 5`, an ordinary CONFIRM seed that simply was not run, does
    the same, so a single-digit typo is enough.

    The property that matters is therefore not that the spec resolves to
    something, but that what it resolves to is an inventory the tree fills. An
    empty audit is not a passing audit, so:

    * a selection under which no run on disk fills any declared slot of any
      selected experiment is refused outright, however that selection was
      arrived at;
    * and where the selection was *supplied* (`--seeds` or `--experiments`,
      rather than the declared blocks and the experiments that have runs), an
      experiment carrying attributed runs and not one of them at a declared
      seed is refused too. `--experiments E3` on the real tree is the same hole
      by the other door: E3's eight attributed runs are E1's, E3 declares none
      of them, and `requested_experiment_empty` does not fire because the
      membership is not empty.

    Under the declared blocks the same situation stays a warning
    (`experiment_not_run`), because nothing was overridden and the runs are
    still checked under the experiment that does declare them.

    An explicit spec is resolved once, here, because it does not depend on the
    experiment: `resolve_seeds` consults `exp.seed_block` only when the spec is
    `None`, which is the one case that cannot be empty.
    """
    findings: list[Finding] = []
    if seeds is not None:
        try:
            resolved = registry.resolve_seeds(seeds, 'CONFIRM')
        except (ValueError, TypeError, KeyError) as exc:
            return [Finding(
                ERROR, 'seed_spec_unparseable',
                f'--seeds {seeds!r} is not a block name, a list or a range '
                f'({exc}), so no experiment can be resolved into an inventory '
                f'and nothing below is measured against anything')]
        if not resolved:
            findings.append(Finding(
                ERROR, 'seed_selection_empty',
                f'--seeds {seeds!r} resolves to no seed at all. The declared '
                f'arm x seed inventory is then empty for every experiment, so '
                f'seed completeness has nothing to compare against and reports '
                f'a pass it never made. DESIGN.md 8.4 permits overriding the '
                f'gate only when the override is stamped into the output; this '
                f'one removes the gate. Pass a block name, a list or a range'))
    if declared is None or membership is None:
        return findings

    coverage = declared_coverage(membership, declared)
    covered = sum(cov for _att, cov in coverage.values())
    supplied = ([f'--seeds {seeds!r}'] if seeds is not None else []) \
        + ([f'--experiments {" ".join(sorted(experiments))}']
           if experiments is not None else [])
    how = ', '.join(supplied) if supplied else 'the declared seed blocks'
    if n_runs and not covered:
        attributed = sum(att for att, _cov in coverage.values())
        findings.append(Finding(
            ERROR, 'selection_matches_no_run',
            f'{how} selects an inventory no run on disk fills: {n_runs} run(s) '
            f'were discovered under this tree, {attributed} of them are '
            f'attributed to a selected experiment, and not one fills a '
            f'declared arm x seed slot. Seed completeness, seed blocks and the '
            f'DESIGN.md 4.3 reserve rule are then all true of the empty set, '
            f'and the audit reports a pass it never made. An empty audit is '
            f'not a passing audit. Audit these runs at the seeds they were run '
            f'at',
            detail={'runs_discovered': n_runs, 'runs_attributed': attributed,
                    'runs_at_a_declared_slot': 0,
                    'seeds_requested': str(seeds),
                    'experiments_requested': (sorted(experiments)
                                              if experiments is not None
                                              else None)}))
    elif supplied:
        hollow = sorted(eid for eid, (att, cov) in coverage.items()
                        if att and not cov)
        if hollow:
            lost = sum(coverage[eid][0] for eid in hollow)
            findings.append(Finding(
                ERROR, 'selection_scopes_out_runs',
                f'{len(hollow)} experiment(s) in the selection carry {lost} '
                f'attributed run(s) and not one of them fills a declared arm x '
                f'seed slot: {", ".join(hollow[:MAX_LISTED])}'
                + (f' (+{len(hollow) - MAX_LISTED} more)'
                   if len(hollow) > MAX_LISTED else '')
                + f'. The selection was supplied on the command line ({how}), '
                  f'so what those experiments were measured against is an '
                  f'inventory nothing on disk fills, and every completeness, '
                  f'block and source-validity assertion made about them was '
                  f'vacuous. DESIGN.md 8.4 permits overriding the gate only '
                  f'when the override is stamped into the output; scoping it '
                  f'away is not an override, it is a removal',
                detail={'experiments': hollow[:24], 'n': len(hollow),
                        'runs_scoped_out': lost,
                        'seeds_requested': str(seeds)}))
    return findings


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def multiplicity_ledger(selected: Iterable[str]) -> dict:
    """The ledger `ANALYSIS_PLAN.md` §7 requires on every invocation.

    Recorded as data in every report, and printed unless the caller asked for
    the verdict alone, so the count of analyses carrying no p-value is a fact on
    the record rather than a claim. The audit is one of them: it tests nothing
    and spends no part of the error budget. The family is read from the plan and
    from `registry`, never accepted as an argument -- which is what stops a
    result from being rescued by relocating it into a family of one.
    """
    return {
        'family': 'confirmatory -- the only one (ANALYSIS_PLAN.md 2)',
        'members': '8 = 4 cells x 2 co-primary endpoints '
                   '(final_score, auc_score)',
        'procedure': 'Holm-Bonferroni, step-down from alpha=0.00625',
        'confirmatory_experiments_in_selection': [
            eid for eid in selected
            if registry.EXPERIMENTS[eid].family == 'confirmatory'],
        'analyses_carrying_no_p_value': 'every check in this file; the audit is '
                                       'a precondition for inference, not '
                                       'inference',
        'p_values_emitted': 0,
    }


def audit(out_root: str, experiments: Iterable[str] | None = None,
          seeds=None, strict: bool = False,
          overrides: dict | None = None) -> tuple[bool, dict]:
    """Run every check. Returns (ok, report)."""
    runs, discovery = discover_runs(out_root)
    overrides = dict(overrides or {})
    # `out_root` reaches `registry.jobs` through here, so the DESIGN.md 4.3
    # reserve ledger is read from the tree under audit rather than from `runs/`.
    declared = declare(seeds, {r.seed for r in runs}, overrides,
                       out_root=out_root)
    unknown = sorted(set(experiments or ()) - set(registry.EXPERIMENTS))
    explicit = experiments is not None
    _, _, everywhere = attribute(runs, registry.EXPERIMENTS, declared)
    if explicit:
        selected = [e for e in registry.EXPERIMENTS if e in set(experiments)]
    else:
        # Default to the experiments that actually have runs, so an audit of a
        # partly executed campaign reports on what exists instead of failing on
        # twelve experiments nobody launched. Which ones those are is printed,
        # so the choice is visible rather than convenient.
        with_runs = {eid for eids in everywhere.values() for eid in eids}
        selected = [eid for eid in registry.EXPERIMENTS if eid in with_runs]

    membership, orphans, _ = attribute(runs, selected, declared)
    # Resolved after attribution on purpose. Whether a seed spec parses is a
    # property of the spec; whether it selects an inventory the tree fills is a
    # property of the spec *and* the tree, and only the second one closes the
    # vacuous pass.
    selection_findings = check_seed_selection(
        seeds, declared, membership, len(runs),
        list(experiments) if experiments is not None else None)
    exps = {eid: registry.EXPERIMENTS[eid] for eid in selected}
    in_scope = {r.rel for arms in membership.values()
                for group in arms.values() for r in group}
    # Per-run checks look at what the selection covers; lineage resolution and
    # attribution still see the whole tree.
    scoped = [r for r in runs if r.rel in in_scope] if selected else runs

    checks = [
        check_invariants(membership, exps, declared),
        check_seed_completeness(membership, exps, declared),
        check_seed_blocks(membership, exps, declared, everywhere),
        check_digests(scoped),
        check_run_dir_uniqueness(scoped),
        check_metrics_integrity(scoped),
        check_freeze(scoped),
        check_source_validity(scoped, membership, declared, out_root),
        check_source_lineage(scoped, runs),
        check_transferred_fraction(membership, exps),
        check_plan_hash(scoped),
        check_provenance(scoped),
        check_tuning_provenance(runs, out_root),
        check_reference_coverage(scoped),
        check_attribution(runs, membership, selected, orphans, everywhere,
                          declared, overrides),
    ]
    if declared.findings:
        cat = Check('CATALOGUE', 'an experiment the registry cannot resolve into '
                                'jobs cannot be audited at all')
        cat.findings.extend(declared.findings)
        checks.insert(0, cat)
    if discovery:
        disc = Check('DISCOVERY', 'a run whose identity cannot be read cannot '
                                 'be excluded from an analysis that globs')
        disc.findings.extend(discovery)
        checks.insert(0, disc)
    if selection_findings:
        sel = Check('SEED SELECTION',
                    'a selection that resolves to nothing, or to an inventory '
                    'no run on disk fills, does not relax the completeness '
                    'gate, it disables it (DESIGN.md 8.4)')
        sel.findings.extend(selection_findings)
        checks.insert(0, sel)
    if unknown:
        bad = Check('SELECTION', 'an experiment id that is not in the catalogue')
        bad.add(ERROR, 'unknown_experiment',
                f'not in the catalogue: {", ".join(unknown)}; known: '
                f'{", ".join(registry.EXPERIMENTS)}')
        checks.insert(0, bad)
    if explicit:
        for eid in selected:
            if not any(membership[eid].values()):
                empty = Check('SELECTION', 'an explicitly requested experiment '
                                           'with nothing on disk')
                empty.add(ERROR, 'requested_experiment_empty',
                          f'{eid} was requested explicitly but no run under '
                          f'{out_root} belongs to any of its arms')
                checks.insert(0, empty)
    if not runs:
        empty = Check('DISCOVERY', 'nothing to audit')
        empty.add(ERROR, 'no_runs',
                  f'no run directories with a manifest under {out_root}')
        checks.insert(0, empty)

    n_err = sum(c.count(ERROR) for c in checks)
    n_warn = sum(c.count(WARN) for c in checks)
    # Not `n_warn`: see DECLARED_BY_DESIGN. Both counts go into the report, so
    # the difference between "warnings" and "warnings --strict would promote"
    # is a number the reader can see rather than a rule buried in a predicate.
    n_promotable = sum(c.promotable() for c in checks)
    ok = n_err == 0 and not (strict and n_promotable)
    report = {
        'ok': ok,
        'out_root': os.path.abspath(out_root),
        'strict': strict,
        'seeds_requested': seeds,
        'overrides_declared': {k: str(v) for k, v in sorted(overrides.items())},
        'experiments_selected': selected,
        'experiments_explicit': explicit,
        'runs_discovered': len(runs),
        'runs_in_scope': len(in_scope),
        'runs_unattributed': len(orphans),
        'plan_hash': provenance.plan_hashes().get('ANALYSIS_PLAN.md'),
        'multiplicity_ledger': multiplicity_ledger(selected),
        'errors': n_err,
        'warnings': n_warn,
        'warnings_promotable_under_strict': n_promotable,
        'warnings_declared_by_design': sorted(DECLARED_BY_DESIGN),
        'checks': [{'name': c.name, 'why': c.why, 'status': c.status(strict),
                    'errors': c.count(ERROR), 'warnings': c.count(WARN),
                    'notes': c.count(NOTE), 'detail': c.detail,
                    'findings': [dataclasses.asdict(f) for f in c.findings]}
                   for c in checks],
    }
    return ok, report


def audit_ok(out_root: str, experiments: Iterable[str] | None = None,
             seeds=None, strict: bool = False,
             overrides: dict | None = None) -> tuple[bool, dict]:
    """The gate `report.py` calls. Returns `(ok, report)`, not a bare bool.

    Spelled out because the tuple is a trap for a caller who trusts the name:
    `if not audit_ok(...)` is always False, since a two-element tuple is always
    truthy, and a caller written that way gets a silently green gate over a
    failed audit. The one caller in the tree, `report.py`, unpacks it correctly.
    `ok` is True only when every check passes.

    `DESIGN.md` §8.4: aggregation and reporting refuse to run on a failed audit
    unless overridden, and the override is stamped into the output. The report
    dict carries everything needed for that stamp -- the failing check names,
    the plan hash and the run counts.
    """
    return audit(out_root, experiments, seeds=seeds, strict=strict,
                 overrides=overrides)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
_MARK = {'PASS': '[ok]  ', 'WARN': '[warn]', 'FAIL': '[FAIL]'}


def _render_finding(finding: dict, verbose: bool) -> list[str]:
    lines = [f'      {finding["level"][:4].upper():4s} {finding["code"]}: '
             f'{finding["message"]}']
    runs = finding.get('runs') or []
    if runs:
        shown = runs[:MAX_LISTED]
        lines.append('           ' + '  '.join(shown)
                     + (f'  (+{len(runs) - len(shown)} more)'
                        if len(runs) > len(shown) else ''))
    if verbose and finding.get('detail'):
        lines.append('           ' + json.dumps(finding['detail'],
                                                default=str)[:600])
    return lines


def render(report: dict, verbose: bool = False, notes: bool = False) -> str:
    out: list[str] = []
    out.append('=' * 78)
    out.append(f'AUDIT  {report["out_root"]}')
    chosen = ', '.join(report['experiments_selected']) or 'none'
    out.append(f'  experiments : {chosen}'
               + ('' if report['experiments_explicit']
                  else '   (default: those with runs on disk)'))
    out.append(f'  runs        : {report["runs_in_scope"]} in scope of '
               f'{report["runs_discovered"]} discovered')
    # `or` was wrong here, and dangerously so: `--seeds ""` is falsy, so the
    # header printed "the block each experiment declares" over an invocation
    # that had requested no seed at all. DESIGN.md 8.4 requires an override to
    # be stamped into the output, and a mislabelled one is worse than none: the
    # audit trail positively asserted a check it had not made. Only `None` means
    # the declared block.
    requested = report['seeds_requested']
    if requested is None:
        seeds = 'the block each experiment declares'
    else:
        seeds = f'{requested!r}   (OVERRIDE of the declared blocks)'
    out.append(f'  seeds       : {seeds}')
    out.append(f'  plan hash   : {report["plan_hash"]}  '
               f'(ANALYSIS_PLAN.md, current)')
    if report.get('overrides_declared'):
        out.append('  overrides   : '
                   + ', '.join(f'{k}={v}' for k, v in
                               report['overrides_declared'].items()))
    out.append('=' * 78)

    for chk in report['checks']:
        out.append('')
        out.append(f'{_MARK[chk["status"]]} {chk["name"]}'
                   f'   [{chk["errors"]} error, {chk["warnings"]} warning, '
                   f'{chk["notes"]} note]')
        out.append(f'       why: {chk["why"]}')
        for finding in chk['findings']:
            if finding['level'] == NOTE and not (notes or verbose):
                continue
            out.extend(_render_finding(finding, verbose))
        if verbose and chk['detail']:
            out.append('       detail: '
                       + json.dumps(chk['detail'], default=str, indent=1)[:4000])

    # Inventory the reader needs whether or not anything failed.
    seedchk = next((c for c in report['checks']
                    if c['name'] == 'SEED COMPLETENESS'), None)
    if seedchk and seedchk['detail']:
        out.append('')
        out.append('-- inventory ' + '-' * 65)
        out.append(f'  {"experiment":11s} {"family":13s} {"declared":>8s} '
                   f'{"present":>8s} {"runs":>6s}  status')
        for eid, det in seedchk['detail'].items():
            if not isinstance(det, dict) or 'declared_runs' not in det:
                continue
            if not det['observed_runs']:
                flag = ('NOT RUN at the declared seeds'
                        + (f' ({det.get("runs_attributed")} run(s) present at '
                           f'other seeds)' if det.get('runs_attributed')
                           else ''))
            elif det.get('pipeline_validation'):
                flag = 'PIPELINE VALIDATION - NOT A RESULT'
            elif det['observed_runs'] == det['declared_runs']:
                flag = 'complete'
            else:
                flag = 'INCOMPLETE'
            out.append(f'  {eid:11s} {det.get("family", "?"):13s} '
                       f'{det["declared_runs"]:8d} {det["observed_runs"]:8d} '
                       f'{det.get("runs_attributed", 0):6d}  {flag}')

    val = next((c for c in report['checks'] if c['name'] == 'SOURCE VALIDITY'),
               None)
    if val and val['detail'].get('per_cell'):
        out.append('')
        out.append('-- source validity by cell ' + '-' * 51)
        for cell, counts in val['detail']['per_cell'].items():
            out.append(f'  {cell:18s} ' + '  '.join(
                f'{k}={v}' for k, v in sorted(counts.items())))

    frac = next((c for c in report['checks']
                 if c['name'] == 'TRANSFERRED-FRACTION MATCHING'), None)
    groups = (frac['detail'].get('cross_arch_groups') if frac else None) or []
    if groups:
        out.append('')
        out.append('-- transferred fraction, cross-arch groups ' + '-' * 35)
        out.append('   every mlp/dueling pair the selection permits; the group '
                   'key names what differs between rows')
    for group in groups:
        # The label and the refusal are printed apart. DESIGN.md 3.1 permits a
        # confounded contrast only when the catalogue declares it unmatched, so
        # a reader has to be able to see at a glance which rows are the
        # declared unmatched comparison and which are a matching that failed.
        if not group['intensity_confounded']:
            mark = '  '
        elif group.get('declared_unmatched'):
            mark = '  INTENSITY-CONFOUNDED (declared unmatched)  '
        else:
            mark = '  INTENSITY-CONFOUNDED, UNDECLARED: REFUSED  '
        out.append(f'  {group["experiment"]:4s} '
                   + '  '.join(f'{arch}={f:.3f}'
                               for arch, f in sorted(group['fractions'].items()))
                   + f'  gap={group["gap"]:.3f}'
                   + ('  FRACTION VARIES WITHIN ARM'
                      if group.get('fractions_not_constant') else '')
                   + mark + str(group['group'])[:96])

    # ANALYSIS_PLAN.md 7. Printed on every invocation so that the count of
    # analyses carrying no p-value is a recorded fact rather than a claim -- and
    # so that it is unambiguous that the audit is not one of them.
    out.append('')
    out.append('-- multiplicity ledger ' + '-' * 55)
    ledger = report.get('multiplicity_ledger') or {}
    out.append(f'  family        : {ledger.get("family")}')
    out.append(f'  members       : {ledger.get("members")}')
    out.append(f'  procedure     : {ledger.get("procedure")}')
    out.append('  confirmatory experiments in this selection: '
               + (', '.join(ledger.get(
                   'confirmatory_experiments_in_selection') or []) or 'none'))
    out.append(f'  p-values emitted by this file: '
               f'{ledger.get("p_values_emitted")}. Analyses carrying no '
               f'p-value:')
    out.append(f'    {ledger.get("analyses_carrying_no_p_value")}.')

    out.append('')
    out.append('=' * 78)
    verdict = 'PASS' if report['ok'] else 'FAIL'
    out.append(f'AUDIT {verdict}: {report["errors"]} error(s), '
               f'{report["warnings"]} warning(s)'
               + ('  [--strict: warnings are errors]' if report['strict'] else ''))
    if not report['ok']:
        failed = [c['name'] for c in report['checks'] if c['status'] == 'FAIL']
        out.append(f'  failing checks: {", ".join(failed)}')
        out.append('  DESIGN.md 8.4: aggregation and reporting refuse to run on '
                   'a failed audit unless')
        out.append('  overridden, and the override is stamped into the output.')
    out.append('=' * 78)
    return '\n'.join(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out-root', default='runs',
                   help='run tree to audit (default: runs)')
    p.add_argument('--experiments', nargs='*', default=None,
                   help='catalogue ids to audit; default: those with runs')
    p.add_argument('--seeds', default=None,
                   help='seed set the runs were launched at, as a block name or '
                        'a list/range such as "0-2" (default: the block each '
                        'experiment declares). Reducing it is the '
                        'STANDING_INSTRUCTIONS S8 validation invocation, and it '
                        'is recorded in the report and stamped in the header. '
                        'A spec resolving to no seed at all is refused rather '
                        'than obeyed: it would empty the declared inventory and '
                        'make every completeness assertion vacuously true')
    p.add_argument('--overrides', nargs='*', default=None,
                   help='the launch-level overrides that were in force, as '
                        'field=value (e.g. freeze_updates=150 '
                        'num_episodes=14). Without them a scaled launch is '
                        'audited against the full declared configuration and '
                        'every difference is reported; with them it is audited '
                        'against what it meant. Arm-level values still win, '
                        'exactly as in registry.jobs')
    p.add_argument('--strict', action='store_true',
                   help='treat warnings as errors')
    p.add_argument('--notes', action='store_true',
                   help='print note-level findings (exclusions, inventory)')
    p.add_argument('--verbose', action='store_true',
                   help='print every finding detail and per-check detail block')
    p.add_argument('--json', dest='json_out', default=None,
                   help='write the full report dict here')
    p.add_argument('--quiet', action='store_true',
                   help='print only the one-line verdict. The multiplicity '
                        'ledger ANALYSIS_PLAN.md 7 requires is recorded in the '
                        'report dict either way, and --json still writes it')
    args = p.parse_args(argv)

    try:
        overrides = parse_overrides(args.overrides)
    except ValueError as exc:
        print(f'audit: {exc}')
        return 2
    ok, report = audit(args.out_root, args.experiments, seeds=args.seeds,
                       strict=args.strict, overrides=overrides)
    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)),
                    exist_ok=True)
        with open(args.json_out, 'w', encoding='utf-8') as fh:
            json.dump(report, fh, indent=2, default=str)
    if args.quiet:
        print(f'AUDIT {"PASS" if ok else "FAIL"}: {report["errors"]} error(s), '
              f'{report["warnings"]} warning(s)')
    else:
        print(render(report, verbose=args.verbose, notes=args.notes))
    if args.json_out:
        print(f'report -> {args.json_out}')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
