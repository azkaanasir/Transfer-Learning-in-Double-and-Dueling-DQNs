"""The machine-checked invariant checker: what turns control claims into facts.

    python experiments/audit.py --out-root runs
    python experiments/audit.py --out-root runs --experiments E1 E2
    python experiments/audit.py --out-root runs_demo --experiments E1 --seeds 0-2
    python experiments/audit.py --out-root runs --json runs/audit.json

Exit status is 0 only when every check passes. `report.py` gates on
`audit_ok(out_root, experiments)`, and `DESIGN.md` §8.4 requires that gate: an
override is permitted but must be stamped into the output.

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
  are printed when it does not (`DESIGN.md` §3.3, §8.4).

* **Seed completeness.** One seed was dropped from one published arm with no
  stated rule. The declared arm x seed inventory comes from
  `registry.jobs()` -- the same function the runner uses -- so a missing run is
  a missing row here rather than a silently smaller n (`DESIGN.md` §9,
  `ANALYSIS_PLAN.md` §8).

* **Tune leakage.** Revision 1 selected hyperparameters on seeds 0-4 and then
  ran every confirmatory arm on 0-9, so half of each confirmatory sample had
  been tuned on. A reported experiment touching the `TUNE` block fails
  (`DESIGN.md` §3.4).

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

* **Metrics integrity.** The published trainer appended to its metrics file on
  resume without truncating, so a crash between checkpoints duplicated episodes
  and corrupted every window statistic downstream. The episode index set must be
  exactly `range(0, episodes_completed)` (`DESIGN.md` §8.2).

* **Freeze verification.** The manuscript described a freeze schedule the code
  never implemented. Freezing is now indexed in gradient updates and checked by
  weight fingerprints at each transition, in **both** directions: a
  declared-frozen layer that moved, and a trainable layer that did not, are both
  defects -- the second is what a positionally resolved freeze produces silently
  (`DESIGN.md` §3.2). Note that `transfer.verify_freeze`'s own `ok` flag covers
  only the first direction, so this check is deliberately stricter than that
  flag.

* **Source validity and lineage.** A published source agent scored 26.94 on a
  task solved at 475, and the transfer arm that consumed it is now
  unidentifiable -- the only surviving CartPole checkpoint is from the wrong
  architecture. So every transfer run must carry a validity verdict on the
  normalised gate of `DESIGN.md` §4.3 (a *missing* verdict is an error; an
  *invalid* verdict is a reported exclusion, not an error), and its recorded
  source must resolve to a real run whose digest, environment and cell are the
  ones the config names.

* **Transferred-parameter fraction.** Revision 1 held the layer list fixed
  across architectures and called that the same protocol; it transferred 97 % of
  the mlp and 51 % of the dueling network, confounding `arch` with treatment
  intensity by a factor of two -- the published study's own error, reconstituted
  inside the corrected design. Cross-`arch` groups whose fractions differ by
  more than 0.05 are labelled `intensity-confounded` here so that `report.py`
  cannot present them as an architecture contrast (`DESIGN.md` §3.1).

* **Plan hash, provenance, reference coverage.** A confirmatory result is
  interpretable only against the pre-registered plan in force when it ran, so
  every run's `ANALYSIS_PLAN.md` hash must agree with every other run's and with
  the current file (`ANALYSIS_PLAN.md` §1). A result from a dirty tree is not
  reproducible from the repository, so the count is reported. And a missing
  reference return would silently put one variant's scores on a different scale
  from every other's, which is what made the published cross-variant comparisons
  meaningless (`DESIGN.md` §5.1).

The audit computes no statistic and emits no p-value. It is a precondition for
inference, not inference; the ledger printed at the end says so explicitly.
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import json
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
from src.dqn import envs, provenance                         # noqa: E402
from src.dqn.config import (Config, MEASUREMENT_FIELDS,      # noqa: E402
                            TRAJECTORY_FIELDS)

# Severities. `error` fails the audit; `warning` is reported and, under
# --strict, promoted to an error; `note` is inventory the reader needs and never
# affects the exit status.
ERROR, WARN, NOTE = 'error', 'warning', 'note'

# DESIGN.md 3.1: the declared tolerance on transferred-parameter fraction for a
# cross-architecture contrast.
FRACTION_TOLERANCE = 0.05
# A recorded normalisation constant may drift only by float round-trip.
REFERENCE_TOLERANCE = 1e-6
# ANALYSIS_PLAN.md 9: below this, output is pipeline validation, not a result.
MIN_N_FOR_A_RESULT = 3
# How many run directories to name per finding before summarising the rest.
MAX_LISTED = 6

# The registry's own line between a budget setting a caller may scale and a
# factor that defines what the experiment is. Used, not re-derived, so the
# audit cannot disagree with the runner about which is which.
SCALING_FIELDS = frozenset(registry.SCALING_FIELDS)

CONFIG_DEFAULTS: dict[str, Any] = {f.name: f.default
                                   for f in dataclasses.fields(Config)}


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
    skipped: Optional[str] = None

    def add(self, level: str, code: str, message: str,
            runs: Iterable[str] = (), **detail) -> None:
        self.findings.append(Finding(level, code, message, list(runs), detail))

    def count(self, level: str) -> int:
        return sum(1 for f in self.findings if f.level == level)

    def status(self, strict: bool = False) -> str:
        if self.skipped:
            return 'SKIP'
        if self.count(ERROR) or (strict and self.count(WARN)):
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
        return self.manifest.get('config') or {}

    @property
    def identity(self) -> dict:
        return self.manifest.get('identity') or {}

    @property
    def seed(self) -> Optional[int]:
        seed = self.cfg.get('seed')
        return None if seed is None else int(seed)

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


def discover_runs(out_root: str) -> tuple[list[Run], list[Finding]]:
    """Every run directory under `out_root`, with unreadable manifests reported.

    An unreadable manifest is an error rather than a skip: a run whose identity
    cannot be read cannot be excluded from an analysis that globs the tree, so
    silently ignoring it here would leave it to be picked up downstream.
    """
    root = os.path.abspath(out_root)
    findings: list[Finding] = []
    runs: list[Run] = []
    pattern = os.path.join(glob.escape(root), '**', 'manifest.json')
    for path in sorted(glob.glob(pattern, recursive=True)):
        run_dir = os.path.dirname(os.path.abspath(path))
        rel = os.path.relpath(run_dir, root).replace(os.sep, '/')
        try:
            with open(path, encoding='utf-8') as fh:
                manifest = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            findings.append(Finding(ERROR, 'manifest_unreadable',
                                    f'{rel}: {exc}', [rel]))
            continue
        if not isinstance(manifest, dict) or 'config' not in manifest:
            findings.append(Finding(
                ERROR, 'manifest_malformed',
                f'{rel}: manifest has no config block', [rel]))
            continue
        runs.append(Run(run_dir, rel, manifest))
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
    """Comparable form of a config value: canonical env strings, tuples not lists."""
    if name in ('env', 'source_env') and isinstance(value, str) and value:
        try:
            return envs.parse(value).canonical()
        except Exception:                                   # noqa: BLE001
            return value
    if isinstance(value, (list, tuple)):
        return tuple(_norm(name, v) for v in value)
    return value


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
            overrides: dict | None = None) -> Declared:
    """Resolve the whole catalogue into concrete (experiment, arm, seed) configs.

    Built over every experiment, not only the selected ones, because the
    equivalence classes are what identify a shared run: auditing E2 alone still
    has to recognise that its scratch denominators are runs E1 launched.

    Seeds are resolved twice, on purpose. The *target* set -- the block each
    experiment declares, or `--seeds` -- is what seed completeness is measured
    against. Its union with the seeds actually present on disk is what
    attribution uses, so a run at an undeclared seed is recognised and reported
    rather than silently dropped.
    """
    out = Declared(classes=defaultdict(list))
    observed = sorted({int(s) for s in observed_seeds if s is not None})
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
                                        overrides=overrides,
                                        allow_factor_overrides=True)
            union_jobs = registry.jobs(
                eid, seeds=sorted(set(target) | set(observed)),
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
    """membership[experiment][arm] -> runs, the orphans, and the full mapping."""
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


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def check_invariants(membership, exps, declared: Declared) -> Check:
    chk = Check(
        'INVARIANTS',
        'the published transfer arm ran at lr=1e-4 against a baseline\'s 5e-4 '
        'under a claim of identical hyperparameters (DESIGN.md 3.3, 8.4)')
    audited = tuple(sorted(set(TRAJECTORY_FIELDS) | set(MEASUREMENT_FIELDS)))
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
                value = _norm(name, run.cfg.get(name, CONFIG_DEFAULTS.get(name)))
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
                    + ('The runs record the override in their notes, so it was '
                       'deliberate; pass --overrides to audit them against the '
                       'configuration actually intended.'
                       if entry['stamped'] else
                       'A run whose factors differ is not the arm its label '
                       'names.'),
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
    return chk


def check_seed_completeness(membership, exps, seeds, declared: Declared) -> Check:
    chk = Check(
        'SEED COMPLETENESS',
        'one seed was dropped from one published arm with no stated rule; '
        'partial arms are refused (DESIGN.md 9, ANALYSIS_PLAN.md 8)')
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

        # One arm at one seed is one run. Two runs there are two independent
        # estimates of one quantity that a reader would take for two arms --
        # the failure mode the digest-keyed run directory exists to prevent, so
        # if it appears here it means two launches wrote under one label.
        for label, group in sorted(arms.items()):
            by_seed: dict[Optional[int], list[Run]] = defaultdict(list)
            for run in group:
                by_seed[run.seed].append(run)
            for seed, twins in sorted(by_seed.items(), key=lambda kv: str(kv[0])):
                if len(twins) > 1:
                    chk.add(ERROR, 'duplicate_runs_for_arm_seed',
                            f'{eid}/{label} at seed {seed} has {len(twins)} '
                            f'runs; one arm at one seed is one run, and two are '
                            f'two independent estimates of the same quantity',
                            runs=[t.rel for t in twins][:MAX_LISTED],
                            experiment=eid, arm=label, seed=seed)
        missing = sorted(pairs - observed)
        extra = sorted(p for p in observed - pairs if p[1] in target)
        per_arm = {label: sorted({r.seed for r in group})
                   for label, group in arms.items()}
        n_by_arm = {label: len([s for s in seen if s in target])
                    for label, seen in per_arm.items()}
        chk.detail[eid] = {
            'declared_runs': len(pairs),
            'observed_runs': len(observed & pairs),
            'seeds_declared': sorted(target),
            'per_arm_seeds': per_arm,
            'family': exp.family,
            # ANALYSIS_PLAN.md 9: below n=3 no test and no interval may be
            # emitted, and every page carries PIPELINE VALIDATION - NOT A RESULT.
            'pipeline_validation': (max(n_by_arm.values()) < MIN_N_FOR_A_RESULT
                                    if n_by_arm else True),
            'max_n_per_arm': max(n_by_arm.values()) if n_by_arm else 0,
        }
        if not (observed & pairs):
            chk.add(NOTE, 'experiment_not_run',
                    f'{eid}: no run on disk belongs to any declared arm at the '
                    f'selected seeds', experiment=eid)
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
            chk.add(NOTE, 'undeclared_runs',
                    f'{eid}: {len(extra)} run(s) at a selected seed belong to an '
                    f'arm the registry does not schedule there',
                    experiment=eid,
                    pairs=[f'{a}@s{s}' for a, s in extra[:MAX_LISTED]])
        if chk.detail[eid]['pipeline_validation']:
            chk.add(NOTE, 'pipeline_validation',
                    f'{eid}: at most {chk.detail[eid]["max_n_per_arm"]} seed(s) '
                    f'per arm. ANALYSIS_PLAN.md 9 forbids a test or an interval '
                    f'below n={MIN_N_FOR_A_RESULT} and requires the output to be '
                    f'stamped PIPELINE VALIDATION - NOT A RESULT',
                    experiment=eid)
    return chk


def check_tune_leakage(membership, exps) -> Check:
    chk = Check(
        'TUNE LEAKAGE',
        'revision 1 selected hyperparameters on seeds 0-4 and ran every '
        'confirmatory arm on 0-9 (DESIGN.md 3.4)')
    tune = set(registry.SEED_BLOCKS['TUNE'])
    chk.detail['tune_block'] = sorted(tune)
    for eid, arms in membership.items():
        exp = exps[eid]
        # The only licensed exemption is the selection experiment itself: a
        # screen whose declared block *is* TUNE. Narrowing the exemption to that
        # intersection keeps `family='screen'` from becoming a way to launder a
        # tuned seed into a reported estimate.
        exempt = exp.family == 'screen' and exp.seed_block == 'TUNE'
        hits = sorted({r.rel for group in arms.values() for r in group
                       if r.seed in tune})
        chk.detail[eid] = {'exempt': exempt, 'runs_on_tune_seeds': len(hits)}
        if hits and not exempt:
            chk.add(ERROR, 'tune_leakage',
                    f'{eid} is reported (family={exp.family}, block='
                    f'{exp.seed_block}) and contains {len(hits)} run(s) on TUNE '
                    f'seeds; no reported estimate may draw on the selection block',
                    runs=hits[:MAX_LISTED], experiment=eid, n=len(hits))
        elif hits:
            chk.add(NOTE, 'tune_seeds_expected',
                    f'{eid}: {len(hits)} run(s) on TUNE seeds, which is what '
                    f'this experiment is for; nothing it produces may enter a '
                    f'reported estimate', experiment=eid, n=len(hits))
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
    for run in runs:
        try:
            cfg = Config(**run.cfg)
        except TypeError as exc:
            chk.add(ERROR, 'config_schema_drift',
                    f'{run.rel}: the stored config cannot be loaded into the '
                    f'current Config ({exc}); its digests are not comparable '
                    f'with any run written by this code', runs=[run.rel])
            continue
        except ValueError as exc:
            chk.add(ERROR, 'config_invalid',
                    f'{run.rel}: the stored config is rejected by Config '
                    f'validation ({exc})', runs=[run.rel])
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


def check_run_dir_uniqueness(runs: list[Run], out_root: str) -> Check:
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
    chk = Check(
        'METRICS INTEGRITY',
        'the published trainer appended on resume without truncating, so a '
        'crash duplicated episodes and corrupted every window statistic '
        '(DESIGN.md 8.2)')
    short = 0
    for run in runs:
        path = os.path.join(run.path, 'metrics.jsonl')
        if not os.path.exists(path):
            chk.add(ERROR, 'metrics_absent',
                    f'{run.rel}: metrics.jsonl is missing', runs=[run.rel])
            continue
        episodes: list[int] = []
        bad_rows = 0
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
        if bad_rows:
            chk.add(ERROR, 'metrics_unparseable',
                    f'{run.rel}: {bad_rows} row(s) are not JSON objects keyed '
                    f'by episode', runs=[run.rel], rows=bad_rows)
        completed = run.get('result', 'episodes_completed')
        counts = Counter(episodes)
        duplicates = sorted(e for e, n in counts.items() if n > 1)
        if duplicates:
            chk.add(ERROR, 'metrics_duplicate_episodes',
                    f'{run.rel}: {len(duplicates)} episode index/indices appear '
                    f'more than once (first: {duplicates[:5]})',
                    runs=[run.rel], duplicates=duplicates[:24])
        if completed is None:
            chk.add(ERROR, 'result_absent',
                    f'{run.rel}: no result.episodes_completed; the run did not '
                    f'finish, so it has no endpoint to report', runs=[run.rel])
        else:
            expected = set(range(0, int(completed)))
            if set(episodes) != expected:
                missing = sorted(expected - set(episodes))
                extra = sorted(set(episodes) - expected)
                chk.add(ERROR, 'metrics_not_contiguous',
                        f'{run.rel}: episode index set is not range(0, '
                        f'{int(completed)}) -- {len(missing)} missing, '
                        f'{len(extra)} out of range',
                        runs=[run.rel], missing=missing[:24], extra=extra[:24])
            declared = int(run.cfg.get('num_episodes') or 0)
            if declared and int(completed) < declared:
                short += 1
        stated = run.get('result', 'metrics_integrity', 'contiguous')
        if stated is False:
            chk.add(ERROR, 'metrics_integrity_self_reported',
                    f'{run.rel}: the run recorded its own metrics as '
                    f'non-contiguous: '
                    f'{run.get("result", "metrics_integrity", "problems")}',
                    runs=[run.rel])
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
        events = run.manifest.get('freeze_events') or []
        transfer = run.condition != 'scratch'
        window = int(run.cfg.get('freeze_updates') or 0)
        updates = run.get('result', 'updates', default=0) or 0
        checked = 0
        for i, event in enumerate(events):
            verdict = event.get('verification')
            if verdict is None:
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


def check_source_validity(runs: list[Run]) -> Check:
    chk = Check(
        'SOURCE VALIDITY',
        'a published source agent scored 26.94 on a task solved at 475 and was '
        'transferred from anyway (DESIGN.md 4.3)')
    per_cell: dict[str, Counter] = defaultdict(Counter)
    excluded: dict[str, list[tuple[str, Any, Any]]] = defaultdict(list)
    for run in runs:
        if run.condition == 'scratch':
            continue
        validity = run.get('source', 'validity')
        if validity is None:
            chk.add(ERROR, 'validity_verdict_missing',
                    f'{run.rel}: a {run.condition} run with no source-validity '
                    f'verdict; the gate of DESIGN.md 4.3 was never evaluated',
                    runs=[run.rel])
            per_cell[run.cell]['missing'] += 1
            continue
        valid = validity.get('valid')
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
        if valid is True:
            per_cell[run.cell]['valid'] += 1
        elif valid is False:
            per_cell[run.cell]['invalid'] += 1
            # Reported, never fatal: DESIGN.md 4.3 makes the primary estimand
            # valid-sources-only with the exclusions printed, so an invalid
            # source is an exclusion for `report.py` to carry, not a broken run.
            excluded[run.cell].append(
                (run.rel, validity.get('source_final_score'),
                 validity.get('gate')))
        else:
            per_cell[run.cell]['unknown'] += 1
            chk.add(ERROR, 'validity_unknown',
                    f'{run.rel}: the validity verdict is null and the condition '
                    f'is {run.condition}, so the source\'s competence was never '
                    f'established', runs=[run.rel])
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
                scores=sorted({round(float(s), 4) for _r, s, _g in rows
                               if s is not None}))
    chk.detail['per_cell'] = {cell: dict(counts)
                              for cell, counts in sorted(per_cell.items())}
    return chk


def check_source_lineage(runs: list[Run],
                        universe: list[Run] | None = None) -> Check:
    """Iterated over the selected runs; resolved against the whole tree.

    Resolution has to see every run, not only the selection: E8i's
    positive control draws its donors from the disjoint `C4SRC` block, so
    auditing E8i alone must still be able to find them.
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

        src_dir = os.path.dirname(os.path.normpath(checkpoint.replace('\\', os.sep)))
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


def check_transferred_fraction(membership, exps) -> Check:
    chk = Check(
        'TRANSFERRED-FRACTION MATCHING',
        'the same layer list transfers 97 % of the mlp and 51 % of the dueling '
        'network, confounding arch with treatment intensity (DESIGN.md 3.1)')
    keys = tuple(k for k in DISCRIMINATING if k != 'arch')
    for eid, arms in membership.items():
        groups: dict[tuple, dict[str, list[tuple[str, float]]]] = defaultdict(
            lambda: defaultdict(list))
        for label, group in arms.items():
            for run in group:
                if run.condition == 'scratch':
                    continue
                frac = run.get('transfer', 'summary',
                               'fraction_of_model_transferred')
                if frac is None:
                    chk.add(ERROR, 'fraction_unrecorded',
                            f'{run.rel}: a {run.condition} run with no '
                            f'transferred-parameter fraction; the intensity of '
                            f'the treatment is unknown', runs=[run.rel])
                    continue
                sig = _run_signature(run)
                groups[tuple(sig[k] for k in keys)][str(sig['arch'])].append(
                    (run.rel, float(frac)))
        for key, by_arch in sorted(groups.items(), key=lambda kv: str(kv[0])):
            described = dict(zip(keys, key))
            for arch, entries in by_arch.items():
                spread = max(f for _, f in entries) - min(f for _, f in entries)
                if spread > 1e-9:
                    chk.add(WARN, 'fraction_varies_within_arm',
                            f'{eid}: {arch} runs at the same configuration '
                            f'report transferred fractions spanning '
                            f'{spread:.4f}; the treatment intensity is not '
                            f'constant across seeds',
                            runs=[r for r, _ in entries][:MAX_LISTED],
                            experiment=eid)
            if len(by_arch) < 2:
                continue
            means = {arch: sum(f for _, f in e) / len(e)
                     for arch, e in by_arch.items()}
            gap = max(means.values()) - min(means.values())
            record = {'experiment': eid,
                      'condition': described.get('condition'),
                      'target_rule': described.get('target_rule'),
                      'transfer_set': described.get('transfer_set'),
                      'freeze_updates': described.get('freeze_updates'),
                      'env': described.get('env'),
                      'fractions': {a: round(f, 4) for a, f in means.items()},
                      'gap': round(gap, 4)}
            chk.detail.setdefault('cross_arch_groups', []).append(
                {**record, 'intensity_confounded': gap > FRACTION_TOLERANCE})
            if gap > FRACTION_TOLERANCE:
                # DESIGN.md 3.1 permits the contrast only when it is explicitly
                # labelled, so the label is emitted here for `report.py` to
                # carry rather than the contrast being refused outright.
                chk.add(WARN, 'intensity_confounded',
                        f'{eid}: cross-arch group '
                        f'(condition={described.get("condition")}, '
                        f'target_rule={described.get("target_rule")}, '
                        f'transfer_set={described.get("transfer_set")}) has '
                        f'fractions '
                        f'{", ".join(f"{a}={f:.3f}" for a, f in sorted(means.items()))} '
                        f'-- a gap of {gap:.3f} exceeds the {FRACTION_TOLERANCE} '
                        f'tolerance, so any arch contrast here is '
                        f'intensity-confounded and must be labelled so',
                        **record)
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
                    f'{name} has changed since {sum(len(rs) for rs in stale.values())} '
                    f'run(s) were produced; under ANALYSIS_PLAN.md 1 the '
                    f'affected results are exploratory until the change is '
                    f'recorded in its 11',
                    runs=[rs[0] for rs in stale.values()][:MAX_LISTED],
                    document=name, current=current.get(name),
                    stored=[str(h) for h in stale])
    return chk


def check_provenance(runs: list[Run]) -> Check:
    chk = Check(
        'PROVENANCE',
        'a result produced from an uncommitted tree is not reproducible from '
        'the repository (DESIGN.md 8.3)')
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


def check_reference_coverage(runs: list[Run]) -> Check:
    chk = Check(
        'REFERENCE COVERAGE',
        'a missing reference return would put one variant\'s scores on a '
        'different scale from every other\'s (DESIGN.md 5.1)')
    envs_seen: Counter = Counter()
    for run in runs:
        spec = run.cfg.get('env')
        envs_seen[str(spec)] += 1
        try:
            ref = envs.reference(spec)
        except Exception as exc:                            # noqa: BLE001
            chk.add(ERROR, 'reference_missing',
                    f'{run.rel}: {exc}', runs=[run.rel], env=spec)
            continue
        stored = run.manifest.get('reference') or {}
        if not stored:
            chk.add(ERROR, 'reference_unrecorded',
                    f'{run.rel}: the run recorded no normalisation constants, '
                    f'so its scores cannot be recomputed', runs=[run.rel])
            continue
        for key in ('random_return', 'threshold'):
            a, b = stored.get(key), ref.get(key)
            if a is None or b is None or abs(float(a) - float(b)) > REFERENCE_TOLERANCE:
                chk.add(ERROR, 'reference_drift',
                        f'{run.rel}: normalised against {key}={a} but the '
                        f'committed reference is now {b}; this run\'s scores are '
                        f'on a different scale from a freshly measured one',
                        runs=[run.rel], env=spec, key=key,
                        stored=a, current=b)
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
            if eid not in known_exps:
                reasons['experiment not in the catalogue'].append(run.rel)
            elif label not in known_labels:
                reasons['label not an arm of any experiment'].append(run.rel)
            else:
                reasons['arm exists but is not scheduled at this seed'].append(
                    run.rel)
        for reason, rels in sorted(reasons.items()):
            chk.add(WARN, 'unattributed_runs',
                    f'{len(rels)} run(s): {reason}. Either they are ad hoc, or '
                    f'the catalogue has moved under them; either way no '
                    f'experiment in it accounts for them',
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


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
CHECK_ORDER = ('INVARIANTS', 'SEED COMPLETENESS', 'TUNE LEAKAGE',
               'CONFIG/DIGEST CONSISTENCY', 'RUN-DIRECTORY UNIQUENESS',
               'METRICS INTEGRITY', 'FREEZE VERIFICATION', 'SOURCE VALIDITY',
               'SOURCE LINEAGE', 'TRANSFERRED-FRACTION MATCHING', 'PLAN HASH',
               'PROVENANCE', 'REFERENCE COVERAGE', 'RUN ATTRIBUTION')


def audit(out_root: str, experiments: Iterable[str] | None = None,
          seeds=None, strict: bool = False,
          overrides: dict | None = None) -> tuple[bool, dict]:
    """Run every check. Returns (ok, report)."""
    runs, discovery = discover_runs(out_root)
    overrides = dict(overrides or {})
    declared = declare(seeds, {r.seed for r in runs}, overrides)

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
    exps = {eid: registry.EXPERIMENTS[eid] for eid in selected}
    in_scope = {r.rel for arms in membership.values()
                for group in arms.values() for r in group}
    # Per-run checks look at what the selection covers; lineage resolution and
    # attribution still see the whole tree.
    scoped = [r for r in runs if r.rel in in_scope] if selected else runs

    checks = [
        check_invariants(membership, exps, declared),
        check_seed_completeness(membership, exps, seeds, declared),
        check_tune_leakage(membership, exps),
        check_digests(scoped),
        check_run_dir_uniqueness(scoped, out_root),
        check_metrics_integrity(scoped),
        check_freeze(scoped),
        check_source_validity(scoped),
        check_source_lineage(scoped, runs),
        check_transferred_fraction(membership, exps),
        check_plan_hash(scoped),
        check_provenance(scoped),
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
    ok = n_err == 0 and not (strict and n_warn)
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
        'errors': n_err,
        'warnings': n_warn,
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
    """The gate `report.py` calls. True only when every check passes.

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
_MARK = {'PASS': '[ok]  ', 'WARN': '[warn]', 'FAIL': '[FAIL]',
         'SKIP': '[skip]'}


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
    out.append(f'  experiments : {", ".join(report["experiments_selected"]) or "none"}'
               + ('' if report['experiments_explicit']
                  else '   (default: those with runs on disk)'))
    out.append(f'  runs        : {report["runs_in_scope"]} in scope of '
               f'{report["runs_discovered"]} discovered')
    seeds = report['seeds_requested'] or 'the block each experiment declares'
    out.append(f'  seeds       : {seeds}')
    out.append(f'  plan hash   : {report["plan_hash"]}  (ANALYSIS_PLAN.md, current)')
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
                   f'{"present":>8s}  status')
        for eid, det in seedchk['detail'].items():
            if not isinstance(det, dict) or 'declared_runs' not in det:
                continue
            flag = ('PIPELINE VALIDATION - NOT A RESULT'
                    if det.get('pipeline_validation') else 'complete'
                    if det['observed_runs'] == det['declared_runs']
                    else 'INCOMPLETE')
            out.append(f'  {eid:11s} {det.get("family", "?"):13s} '
                       f'{det["declared_runs"]:8d} {det["observed_runs"]:8d}  {flag}')

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
        out.append('   every mlp/dueling pair the selection permits, at fixed '
                   'target_rule and protocol')
    for group in groups:
        out.append(f'  {group["experiment"]:4s} {str(group["condition"]):19s} '
                   f'{str(group["target_rule"]):8s} '
                   f'set={str(group["transfer_set"]):8s} '
                   f'K={str(group["freeze_updates"]):>6s}  '
                   + '  '.join(f'{arch}={f:.3f}'
                               for arch, f in sorted(group['fractions'].items()))
                   + f'  gap={group["gap"]:.3f}'
                   + ('  INTENSITY-CONFOUNDED'
                      if group['intensity_confounded'] else ''))

    # ANALYSIS_PLAN.md 7. Printed on every invocation so that the count of
    # analyses carrying no p-value is a recorded fact rather than a claim -- and
    # so that it is unambiguous that the audit is not one of them.
    out.append('')
    out.append('-- multiplicity ledger ' + '-' * 55)
    out.append('  family        : confirmatory -- the only one '
               '(ANALYSIS_PLAN.md 2)')
    out.append('  members       : 8 = 4 cells x 2 co-primary endpoints '
               '(final_score, auc_score)')
    out.append('  procedure     : Holm-Bonferroni, step-down from alpha=0.00625')
    out.append('  confirmatory experiments in this selection: '
               + (', '.join(eid for eid in report['experiments_selected']
                            if registry.EXPERIMENTS[eid].family == 'confirmatory')
                  or 'none'))
    out.append('  analyses carrying no p-value: every check in this file. The '
               'audit tests nothing;')
    out.append('    it is a precondition for inference, and it spends no part of '
               'the error budget.')

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
                        'is recorded in the report')
    p.add_argument('--overrides', nargs='*', default=None,
                   help='launch-level overrides that were in force, as '
                        'field=value (e.g. freeze_updates=150). Stated '
                        'explicitly they suppress the inference described '
                        'in infer_launch_overrides')
    p.add_argument('--strict', action='store_true',
                   help='treat warnings as errors')
    p.add_argument('--notes', action='store_true',
                   help='print note-level findings (exclusions, inventory)')
    p.add_argument('--verbose', action='store_true',
                   help='print every finding detail and per-check detail block')
    p.add_argument('--json', dest='json_out', default=None,
                   help='write the full report dict here')
    p.add_argument('--quiet', action='store_true',
                   help='write the JSON report and print only the verdict')
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
