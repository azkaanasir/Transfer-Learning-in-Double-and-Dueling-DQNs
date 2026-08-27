"""The per-cell tuning selection: the artifact the secondary policy is run from.

`DESIGN.md` 3.3 declares two hyperparameter policies and pre-registers an
arbitration between them. The **primary** policy is one common configuration
for all four cells, fixed a priori at `lr=5e-4` with a hard target update every
1000 updates. The **secondary** policy is each cell's own `E3`-selected
configuration. An RQ2 or RQ3 conclusion may be asserted only where the two
agree, and where they disagree that disagreement is the finding.

Until 2026-08-26 the secondary policy had **no runs behind it**: `E3` was the
only experiment in the catalogue that varied `lr` at all, so the arbitration was
unsatisfiable and RQ2, the primary confirmatory question, could not have been
asserted. The resolution taken was to run the policy rather than weaken the
rule: `E1` and `E2` are replicated at each cell's selected configuration on the
same `CONFIRM` seeds, as a stage sequentially dependent on `E3`. Those replicas
are `registry.TUNED_OF`, and they are enumerated from the artifact this module
writes and reads.

Four properties are load-bearing, and each exists because of a specific way this
could go wrong.

* **The rule is one function.** `SELECTION_RULE` at the top of this module is
  the whole selection criterion. Everything else -- assembling the candidate
  table, checking the seed block, recording provenance -- is rule-independent,
  so the criterion can be replaced without touching any of it. Whoever chooses
  the rule chooses how easily the two policies agree, which is why it is
  pre-registered rather than discovered.

* **The rule shipped here is the pre-registered one.** Installed 2026-08-27,
  in place of the labelled placeholder argmax that stood while the machinery was
  being built. `ANALYSIS_PLAN.md` 2.3 fixes it in full and this module executes
  it as written: mean `auc_score` over the five `TUNE` seeds as the criterion,
  the a priori configuration kept unless the highest-mean candidate beats it by
  more than one standard error of the difference computed from the **ddof=1**
  sample variance, three deterministic tie-breaks, a candidate missing any
  `TUNE` seed refused rather than compared on four, and a cell refused where its
  selected candidate misses the 0.6 competence floor on mean normalised
  `final_score`. `SELECTION_RULE_IS_PLACEHOLDER` is therefore False, so
  `stats.py` no longer records `not-evaluable` on the ground of the rule itself,
  and every selection carries the rule id, its parameters and its per-cell
  arithmetic. A rule handed to `compute_selection` by a caller is still recorded
  as a placeholder, because it is still not this rule.

* **The artifact is content-addressed and carries its provenance.**
  `selection_id` is a hash over the selection, the evidence, the plan hashes and
  the `E3` run digests it was computed from. A hand-edited artifact fails
  `read_selection`. A selection recomputed from the same table, under the same
  rule and the same plans, has the same id, so "which selection were these runs
  enumerated from" is answerable from twelve hex characters.

* **A selection off the `TUNE` block is refused.** `DESIGN.md` 3.4 reserves
  `TUNE` for exactly this and `ANALYSIS_PLAN.md` 8 forbids a reported estimate
  drawing on it. Revision 1 selected hyperparameters on seeds 0-4 and then ran
  every confirmatory arm on 0-9, so half of each confirmatory sample had been
  tuned on. A seed outside `TUNE` reaching this module is that defect returning,
  and it raises rather than warns.

Reading and writing are deliberately separate from computing: `read_selection`
never recomputes, so what the tuned arms enumerate from is the stored decision
and not a decision re-derived against whatever the run tree happens to hold now.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _datetime
import hashlib
import json
import math
import os
import sys
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _path in (_REPO, _HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import registry                                              # noqa: E402
from src.dqn import provenance                               # noqa: E402
from src.dqn.config import Config                            # noqa: E402


# ---------------------------------------------------------------------------
# THE SELECTION RULE. One function, one place.
# ---------------------------------------------------------------------------
#: The endpoints summarised for every candidate and stored as the evidence.
#: Both co-primary endpoints of `ANALYSIS_PLAN.md` 1 are computed whatever the
#: rule reads, so that swapping the rule never requires recomputing the table
#: and so the artifact records what the rule did *not* select on as well.
SELECTION_ENDPOINTS: tuple[str, ...] = ('auc_score', 'final_score')

#: The criterion endpoint of `ANALYSIS_PLAN.md` 2.3: mean `auc_score` across the
#: five `TUNE` seeds. Not final score, because on LunarLander every cell
#: finishes above the solved threshold and a final-score comparison is then a
#: comparison of ceilings that cannot express an ordering.
SELECTION_CRITERION_ENDPOINT = 'auc_score'

#: The endpoint and the value of 2.3's competence floor, which is deliberately
#: **not** the criterion endpoint. `auc_score` integrates the curve from zero
#: and never reaches 1.0 even for a solved task (`DESIGN.md` 7's pilot table:
#: best cell 0.9640 on AUC against 1.1787 on final score), so a floor applied to
#: AUC would refuse every cell, which is the defect the first draft of 2.3
#: carried and 11 records. The value 0.6 is the competence floor `DESIGN.md` 4.3
#: already declares for source validity, reused rather than invented here.
COMPETENCE_ENDPOINT = 'final_score'
COMPETENCE_FLOOR = 0.6

#: The denominator 2.3 writes into the standard error, literally: `var_A/5 +
#: var_B/5`. A constant and not the observed count, because 2.3 refuses a
#: candidate measured at fewer than five `TUNE` seeds rather than shrinking the
#: denominator: if a `TUNE` run is lost, the fix is to re-run it. A candidate
#: whose measured n differs from this is refused below, so this number never
#: silently describes a smaller sample than the one it was written for.
SELECTION_N = 5

#: Which of the two variances `EndpointStats` carries the standard error reads,
#: and it is read *through* this name: `var`, the **ddof=1 sample** variance.
#: 2.3 names the estimator explicitly and
#: the reason is quantitative: at n=5 the unbiased and population estimators
#: differ by 25 per cent in the variance and about 12 per cent in the standard
#: error, which is enough to move a marginal cell. The choice is recorded in
#: every artifact, in `SELECTION_RULE_PARAMETERS` and per cell in
#: `evidence.rule_working`, beside the standard error the ddof=0 estimator would
#: have produced and whether it would have changed the cell, so a reader can see
#: that the estimator was chosen rather than inherited from a library default.
SELECTION_VARIANCE_FIELD = 'var'

for _endpoint in (SELECTION_CRITERION_ENDPOINT, COMPETENCE_ENDPOINT):
    if _endpoint not in SELECTION_ENDPOINTS:
        raise RuntimeError(
            f'{_endpoint!r} is read by the selection rule but is not in '
            f'SELECTION_ENDPOINTS {SELECTION_ENDPOINTS}, so no candidate would '
            f'carry it and every cell would refuse')
del _endpoint

#: Identifier of the rule in force, stamped into every artifact. Change it
#: whenever `SELECTION_RULE` changes, or two selections computed under different
#: criteria become indistinguishable in the record.
SELECTION_RULE_ID = 'PREREG-ANALYSIS_PLAN-2.3-conservative-1se-mean-auc-v1'

#: True while `SELECTION_RULE` is not the pre-registered criterion. Read it
#: rather than parsing the id: `sweep.py` must refuse to launch a real campaign
#: from a placeholder selection, and `report.py` must not print one as a result.
#: False since 2026-08-27, when the criterion `ANALYSIS_PLAN.md` 2.3
#: pre-registers replaced the labelled placeholder argmax that shipped before
#: it. A selection computed under a rule handed to `compute_selection` by a
#: caller is still recorded as a placeholder, because it is still not this rule.
SELECTION_RULE_IS_PLACEHOLDER = False

SELECTION_RULE_DESCRIPTION = (
    'The pre-registered criterion of ANALYSIS_PLAN.md 2.3. Per cell, over the '
    'five TUNE seeds: A is the a priori configuration (lr=5e-4, hard target '
    'update every 1000 updates) and B is the candidate with the highest mean '
    'auc_score, ties in that maximum broken by higher mean final_score, then '
    'lower lr, then hard before soft. B is selected only where mean_AUC(B) - '
    'mean_AUC(A) > sqrt(var_A/5 + var_B/5), with var the ddof=1 sample '
    'variance; otherwise A is kept. The secondary policy therefore departs from '
    'the primary only where there is one standard error of separation, which is '
    'what the fair-baseline objection asks for and no more: a plain argmax at '
    'n=5 would chase noise, invent per-cell differences that do not exist, and '
    'those would then read as a policy disagreement and block an RQ2 conclusion '
    'the data support. A cell whose selected candidate falls below a mean '
    'normalised final_score of 0.6 is refused rather than tuned, and so is any '
    'candidate not measured at every one of the five TUNE seeds.')

#: The rule's numbers, stamped into the artifact so the record states the
#: criterion rather than pointing at a function name. Recorded only for a
#: selection computed under `SELECTION_RULE`: a caller-supplied rule does not
#: obey these, and attributing them to it would misdescribe the decision.
SELECTION_RULE_PARAMETERS: Mapping[str, Any] = {
    'plan_section': 'ANALYSIS_PLAN.md 2.3',
    'criterion_endpoint': SELECTION_CRITERION_ENDPOINT,
    'criterion_statistic': f'mean across the {SELECTION_N} TUNE seeds',
    'variance_estimator': 'sample variance, ddof=1',
    'variance_field': SELECTION_VARIANCE_FIELD,
    'se_denominator_n': SELECTION_N,
    'margin': 'sqrt(var_A/5 + var_B/5), one standard error of the difference',
    'margin_comparison': 'strict: B is selected only where the difference > SE',
    'tie_break': ('higher mean final_score, then lower lr, then hard before '
                  'soft'),
    'competence_endpoint': COMPETENCE_ENDPOINT,
    'competence_floor': COMPETENCE_FLOOR,
    'refuses': ('a candidate missing any TUNE seed; a cell whose selected '
                'candidate is below the competence floor; a cell with no a '
                'priori candidate to compare against'),
    'scope': ('the target-task conditions of DESIGN.md 3.3 on the selection '
              'environment; sources are not retuned'),
}


def _stats_or_refuse(cand: 'Candidate', endpoint: str,
                     cell_key: str) -> 'EndpointStats':
    """One candidate's summary for one endpoint, or the refusal 2.3 states.

    Two things are checked and they are not the same thing. `Candidate.seeds`
    covers the rows that existed; `EndpointStats.n` counts the values that were
    usable, so a row present with an empty or NaN endpoint leaves the seed in
    the first and out of the second. 2.3 requires five measurements, so both
    have to hold.
    """
    stats = cand.stats.get(endpoint)
    if stats is None or stats.n != SELECTION_N or stats.mean is None:
        got = 'no summary at all' if stats is None else f'n={stats.n}'
        raise SelectionIncomplete(
            f'{cell_key} {cand.config_key} ({cand.e3_arm}): {endpoint} is '
            f'measured at {got}, not at all {SELECTION_N} '
            f'{SELECTION_SEED_BLOCK} seeds. ANALYSIS_PLAN.md 2.3 refuses an '
            f'incomplete candidate rather than comparing it against its rivals '
            f'on less evidence, and its standard error is written for '
            f'n={SELECTION_N}: if a {SELECTION_SEED_BLOCK} run is lost the fix '
            f'is to re-run it, not to shrink the denominator.')
    return stats


def _variance_of(stats: 'EndpointStats') -> float:
    """The variance the standard error reads, through the constant that names it.

    Read by name rather than as `.var` written out here, so
    `SELECTION_VARIANCE_FIELD` is the estimator choice itself and not a comment
    about it: there is one place to look and one place to change, and the
    artifact records the same name it read.
    """
    value = getattr(stats, SELECTION_VARIANCE_FIELD, None)
    if value is None:
        raise SelectionIncomplete(
            f'{stats.endpoint} carries no {SELECTION_VARIANCE_FIELD} at n='
            f'{stats.n}, so the standard error of ANALYSIS_PLAN.md 2.3 cannot '
            f'be formed')
    return float(value)


def _rank_key(cand: 'Candidate') -> tuple[float, float, float, int]:
    """2.3's ordering: highest mean AUC, then its three tie-breaks.

    The first component decides every non-tied comparison, so the tie-breaks
    apply exactly where 2.3 says they do, among candidates whose mean AUC is
    equal, without a separate pass over the tied set.
    """
    return (-float(cand.stats[SELECTION_CRITERION_ENDPOINT].mean),
            -float(cand.stats[COMPETENCE_ENDPOINT].mean),
            float(cand.config.lr),
            0 if cand.config.target_update == 'hard' else 1)


def selection_working(cell: tuple[str, str],
                      candidates: Sequence['Candidate']) -> dict:
    """The pre-registered rule's arithmetic for one cell, as a record.

    Pure and deterministic: it either returns the working, which is both
    configurations, both means, both variances, the standard error, the margin
    and the competence check, or it raises the refusal `ANALYSIS_PLAN.md` 2.3
    states. `preregistered_selection_rule` reads `selected_config_key` out of
    it, and `compute_selection` stores the whole dict inside the cell's evidence
    and therefore inside the content address, so the number that decided a cell
    is in the artifact and not only in a log line.
    """
    cell_key = registry._tag(*cell)
    block = tuple(registry.SEED_BLOCKS[SELECTION_SEED_BLOCK])
    if len(block) != SELECTION_N:
        raise SelectionIncomplete(
            f'the {SELECTION_SEED_BLOCK} block is {list(block)}, '
            f'{len(block)} seeds, but ANALYSIS_PLAN.md 2.3 writes the standard '
            f'error as sqrt(var_A/{SELECTION_N} + var_B/{SELECTION_N}) for a '
            f'five-seed block. Substituting a different n here would be this '
            f'module inventing a rule the plan does not state; amend 2.3 first.')
    if not candidates:
        raise SelectionIncomplete(
            f'cell {cell_key} has no candidates at all, so there is nothing to '
            f'select from')

    crit = SELECTION_CRITERION_ENDPOINT
    for cand in sorted(candidates, key=lambda c: c.config_key):
        missing = [s for s in block if s not in cand.seeds]
        if missing:
            raise SelectionIncomplete(
                f'{cell_key} {cand.config_key} ({cand.e3_arm}) is missing '
                f'{SELECTION_SEED_BLOCK} seeds {missing}. ANALYSIS_PLAN.md 2.3 '
                f'refuses an incomplete candidate: re-run the lost run rather '
                f'than shrinking the n={SELECTION_N} denominator of its '
                f'standard error.')
        for endpoint in (crit, COMPETENCE_ENDPOINT):
            _stats_or_refuse(cand, endpoint, cell_key)
        # Every candidate's variance is formed here and not only the two the
        # comparison ends up reading, so a table that cannot support the rule
        # is refused before any candidate has been ranked on it.
        _variance_of(cand.stats[crit])

    ordered = sorted(candidates, key=_rank_key)
    best = ordered[0]
    best_mean = float(best.stats[crit].mean)
    tied = sorted(c.config_key for c in candidates
                  if float(c.stats[crit].mean) == best_mean)

    a_priori = next((c for c in candidates if c.config.equals_a_priori), None)
    if a_priori is None:
        raise SelectionIncomplete(
            f'{cell_key} carries no candidate at the a priori configuration '
            f'{A_PRIORI_CONFIG.config_key}, and ANALYSIS_PLAN.md 2.3 states the '
            f'rule as a comparison against it: without A there is no baseline '
            f'to beat by one standard error, and selecting the argmax instead '
            f'would be the rule 2.3 rejects.')

    mean_a = float(a_priori.stats[crit].mean)
    var_a = _variance_of(a_priori.stats[crit])
    var_b = _variance_of(best.stats[crit])
    difference = best_mean - mean_a
    se = math.sqrt(var_a / SELECTION_N + var_b / SELECTION_N)
    switched = bool(difference > se)
    chosen = best if switched else a_priori

    # The estimator 2.3 rejects, computed and recorded but never read by the
    # decision, so the artifact shows what the choice of estimator was worth in
    # this cell instead of leaving it to be re-derived.
    var_a_pop = float(a_priori.stats[crit].var_pop)
    var_b_pop = float(best.stats[crit].var_pop)
    se_pop = math.sqrt(var_a_pop / SELECTION_N + var_b_pop / SELECTION_N)

    competence = float(chosen.stats[COMPETENCE_ENDPOINT].mean)
    passes_floor = bool(competence >= COMPETENCE_FLOOR)
    working = {
        'rule_id': SELECTION_RULE_ID,
        'cell': cell_key,
        'criterion_endpoint': crit,
        'n_candidates': len(candidates),
        'n_seeds': SELECTION_N,
        'variance_estimator': 'sample variance, ddof=1',
        'a_priori': {'config_key': a_priori.config_key,
                     'e3_arm': a_priori.e3_arm,
                     'mean': mean_a, 'var_ddof1': var_a,
                     'var_ddof0_not_used': var_a_pop},
        'highest_mean': {'config_key': best.config_key,
                         'e3_arm': best.e3_arm,
                         'mean': best_mean, 'var_ddof1': var_b,
                         'var_ddof0_not_used': var_b_pop},
        'tied_at_highest_mean': tied,
        'tie_break_applied': bool(len(tied) > 1),
        'difference': difference,
        'se_ddof1': se,
        'se_ddof0_not_used': se_pop,
        'margin_cleared': switched,
        'switched_from_a_priori': bool(
            switched and not chosen.config.equals_a_priori),
        'decision_would_differ_under_ddof0': bool(
            switched != bool(difference > se_pop)),
        'competence': {'endpoint': COMPETENCE_ENDPOINT,
                       'floor': COMPETENCE_FLOOR,
                       'mean_of_selected': competence,
                       'passes': passes_floor},
        'selected_config_key': chosen.config_key,
    }
    if not passes_floor:
        raise SelectionRefused(
            f'{cell_key} is not tunable: its selected candidate '
            f'{chosen.config_key} ({chosen.e3_arm}) reaches a mean normalised '
            f'{COMPETENCE_ENDPOINT} of {competence:.4f} across the '
            f'{SELECTION_N} {SELECTION_SEED_BLOCK} seeds, below the '
            f'{COMPETENCE_FLOOR} competence floor ANALYSIS_PLAN.md 2.3 imposes '
            f'and DESIGN.md 4.3 declares. 2.3 refuses the cell rather than '
            f'returning its least-bad option, and the refusal aborts the whole '
            f'selection: a partially-tuned secondary policy would run this cell '
            f'under the primary one while the artifact claimed otherwise.')
    return working


def preregistered_selection_rule(cell: tuple[str, str],
                                 candidates: Sequence['Candidate']) -> str:
    """The pre-registered criterion of `ANALYSIS_PLAN.md` 2.3, installed.

    Returns the ``config_key`` of the selected candidate for one cell.

    Let *A* be the a priori configuration and *B* the candidate with the highest
    mean `auc_score` over the five `TUNE` seeds. *B* is selected where
    ``mean_AUC(B) - mean_AUC(A) > sqrt(var_A/5 + var_B/5)`` with ``var`` the
    ddof=1 sample variance; otherwise *A* is kept. Ties in the highest mean are
    broken by higher mean `final_score`, then lower `lr`, then `hard` before
    `soft`, so the choice is reproducible from the stored table alone.

    Conservative rather than argmax on purpose, and the purpose is inferential
    rather than aesthetic: at n=5 the `TUNE` block cannot resolve small
    differences, an argmax would invent per-cell differences that do not exist,
    and under `DESIGN.md` 3.3's arbitration an invented difference blocks an RQ2
    conclusion the data actually support. One standard error and not two: two
    would switch almost never and make the arbitration a formality.

    Two refusals live here rather than in `compute_selection`, because they
    belong to the criterion: a candidate not measured at all five `TUNE` seeds,
    and a cell whose selected candidate is below the 0.6 competence floor on
    mean normalised `final_score`. Both abort the whole selection.

    The signature is the contract `SELECTION_RULE` documents, and this rule
    reads nothing beyond it: every candidate carries its configuration, the
    `TUNE` seeds it was measured at, and per endpoint the per-seed values with
    their mean, sample variance (``var``, ddof=1), population variance
    (``var_pop``, ddof=0) and standard deviation, and `A_PRIORI_CONFIG` names
    the primary policy's configuration.
    """
    return selection_working(cell, candidates)['selected_config_key']


#: How `compute_selection` recovers the arithmetic behind a cell's choice for
#: the artifact. An attribute rather than a second return value, so a rule that
#: does not offer one, which is every fixture rule and any future criterion,
#: still satisfies the one-string contract and simply records no working.
preregistered_selection_rule.explain = selection_working


def placeholder_selection_rule(cell: tuple[str, str],
                               candidates: Sequence['Candidate']) -> str:
    """A labelled placeholder argmax. **Not** installed, and not the criterion.

    Kept after the pre-registered rule was installed on 2026-08-27, because it
    is the one rule guaranteed to select something for any well-formed table,
    which makes it useful for exercising the machinery around the rule without
    the criterion's refusals firing. Every selection computed under it records
    ``rule.placeholder = true``, so `stats.py` blocks every RQ2 and RQ3
    assertion drawn from it and `audit.py` reports it.

    `ANALYSIS_PLAN.md` 2.3 rejects a plain argmax outright: at n=5 it chases
    noise, invents per-cell differences that do not exist, and those then appear
    as a disagreement between the two policies and block an RQ2 conclusion the
    data support. Nothing computed under it is a result.
    """
    warnings.warn(
        f'tuning.placeholder_selection_rule is a PLACEHOLDER and not the '
        f'pre-registered criterion of ANALYSIS_PLAN.md 2.3, which is installed '
        f'as {SELECTION_RULE_ID}. Nothing computed under the placeholder is a '
        f'result.',
        RuntimeWarning, stacklevel=2)

    def sort_key(cand: 'Candidate'):
        auc = cand.stats['auc_score'].mean
        final = cand.stats['final_score'].mean
        return (-_or_inf(auc), -_or_inf(final), cand.config.lr,
                0 if cand.config.target_update == 'hard' else 1)

    ordered = sorted(candidates, key=sort_key)
    if not ordered:
        raise SelectionIncomplete(
            f'cell {registry._tag(*cell)} has no candidates at all, so there is '
            f'nothing to select from')
    return ordered[0].config_key


#: **The** rule. Rebind this name to swap the criterion; nothing else in this
#: module, in `registry.py` or in the artifact schema depends on which function
#: it points at. Keep `SELECTION_RULE_ID`, `SELECTION_RULE_IS_PLACEHOLDER`,
#: `SELECTION_RULE_DESCRIPTION` and `SELECTION_RULE_PARAMETERS` in step with it:
#: they are what the artifact records, and a rule whose record says it is
#: something else is worse than no record at all.
SELECTION_RULE: Callable[[tuple[str, str], Sequence['Candidate']], str] = (
    preregistered_selection_rule)


# ---------------------------------------------------------------------------
# Fixed by the design, not by the rule
# ---------------------------------------------------------------------------
#: Bumped whenever the meaning of a stored field changes, so that an artifact
#: written under an older schema cannot be silently read under a newer one.
SELECTION_SCHEMA = 'tuning-selection/v1'

#: The experiment the selection is computed from (`DESIGN.md` 3.3).
SOURCE_EXPERIMENT = 'E3'

#: The only block a selection may be computed on (`DESIGN.md` 3.4,
#: `ANALYSIS_PLAN.md` 8). Not a parameter: there is no legitimate reason to
#: select on anything else, so there is no knob that permits it.
SELECTION_SEED_BLOCK = 'TUNE'

#: The environment the selection is made on, and therefore the environment the
#: selected configuration applies to. `E3` trains LunarLander scratch runs, so
#: the evidence is about the target task; `registry` applies the selection to a
#: tuned arm only where the arm runs on this environment.
SELECTION_ENV = registry.TARGET_ENV

#: Where the active selection lives, relative to a run tree's root. Beside the
#: source-replacement ledger, and for the same reason: it is a property of the
#: campaign, computed from that tree's `E3` runs and governing that tree's tuned
#: runs, so it travels with the tree rather than with the repository.
SELECTION_RELPATH = os.path.join('_jobs', 'tuning_selection.json')

#: Every selection ever written, keyed by content. The active pointer above is
#: replaceable; these are not, so a selection that arms were once enumerated
#: from remains recoverable by id after the pointer moves.
SELECTION_ARCHIVE_RELDIR = os.path.join('_jobs', 'selections')

#: Top-level key holding the facts that are recorded but not addressed: when the
#: selection was computed, by what, and the git state. Excluding them is what
#: makes the id a hash of the *decision* -- the same table under the same rule
#: and the same plans yields the same id at any hour, from any checkout.
UNADDRESSED_KEY = 'recorded'


def _config_default(name: str):
    """A `Config` field default, read rather than copied.

    `E3`'s soft-update arms set `tau` and leave `target_update_freq` at the
    common value, and its hard-update arms do the reverse. Both fields are in
    the run digest either way (`src/dqn/config.py` TRAJECTORY_FIELDS), so a
    candidate has to carry both to be able to reproduce an `E3` arm's digest
    exactly. Taking the unstated one from the same defaults `registry` builds
    against is what keeps that reproduction exact instead of nearly exact.
    """
    return Config.__dataclass_fields__[name].default


@dataclass(frozen=True)
class CellConfig:
    """One cell's optimiser configuration: what a selection selects.

    All four fields are carried always, including the one the update rule does
    not read, because the run digest covers both `target_update_freq` and `tau`
    regardless of which is live. Carrying only the live one would make a
    reconstructed configuration hash differently from the `E3` arm it came from.
    """

    lr: float
    target_update: str
    target_update_freq: int = int(registry.COMMON['target_update_freq'])
    tau: float = float(_config_default('tau'))

    @property
    def config_key(self) -> str:
        """Stable, human-readable key. Matches `E3`'s arm-label suffix."""
        return f'lr{self.lr:g}-{self.target_update}'

    def overrides(self) -> dict:
        """The `Config` fields this configuration sets, and only those."""
        return {'lr': float(self.lr),
                'target_update': str(self.target_update),
                'target_update_freq': int(self.target_update_freq),
                'tau': float(self.tau)}

    def to_dict(self) -> dict:
        return self.overrides()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> 'CellConfig':
        return cls(lr=float(data['lr']),
                   target_update=str(data['target_update']),
                   target_update_freq=int(data['target_update_freq']),
                   tau=float(data['tau']))

    @property
    def equals_a_priori(self) -> bool:
        return self == A_PRIORI_CONFIG


#: The primary policy's configuration, built from `registry.COMMON` rather than
#: restated, so the two cannot drift. A cell that selects this produces run
#: digests identical to its common-policy arms and therefore *shares* their run
#: directories: that identity is what makes the tuned stage an upper bound on
#: cost rather than a flat doubling, and it is asserted here so that a change to
#: `COMMON` which broke it would be visible at import.
A_PRIORI_CONFIG = CellConfig(
    lr=float(registry.COMMON['lr']),
    target_update=str(registry.COMMON['target_update']),
    target_update_freq=int(registry.COMMON['target_update_freq']),
    tau=float(_config_default('tau')))


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------
class SelectionError(RuntimeError):
    """Base for every refusal this module makes."""


class SelectionMissing(SelectionError):
    """No selection artifact exists, so the tuned stage cannot be enumerated."""


class SelectionIncomplete(SelectionError):
    """The evidence does not cover what a selection needs to be computed from."""


class SeedBlockViolation(SelectionError):
    """Evidence from outside `TUNE` reached the selection."""


class SelectionCorrupt(SelectionError):
    """A stored artifact does not match its own content address."""


class SelectionRefused(SelectionError):
    """The rule declined to select for a cell. Raised *by* a rule."""


# ---------------------------------------------------------------------------
# The candidate table
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EndpointStats:
    """One endpoint of one candidate, summarised across the `TUNE` seeds."""

    endpoint: str
    n: int
    values: tuple[float, ...]           # ordered by seed
    mean: Optional[float]
    #: Sample variance, ddof=1. The estimator behind the standard error of a
    #: mean, hence behind any rule that asks for one standard error of
    #: separation.
    var: Optional[float]
    #: Population variance, ddof=0. Carried beside `var` so a rule never has to
    #: decide which one "variance" meant, and so the artifact records both.
    var_pop: Optional[float]
    sd: Optional[float]

    def to_dict(self) -> dict:
        return {'endpoint': self.endpoint, 'n': self.n,
                'values': list(self.values), 'mean': self.mean,
                'var': self.var, 'var_pop': self.var_pop, 'sd': self.sd}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> 'EndpointStats':
        return cls(endpoint=str(data['endpoint']), n=int(data['n']),
                   values=tuple(float(v) for v in data['values']),
                   mean=_opt_float(data['mean']), var=_opt_float(data['var']),
                   var_pop=_opt_float(data['var_pop']),
                   sd=_opt_float(data['sd']))


@dataclass(frozen=True)
class Candidate:
    """One `E3` configuration for one cell, with the evidence about it."""

    cell: tuple[str, str]
    config: CellConfig
    e3_arm: str
    seeds: tuple[int, ...]
    run_digests: tuple[str, ...]
    stats: Mapping[str, EndpointStats]

    @property
    def config_key(self) -> str:
        return self.config.config_key

    @property
    def cell_key(self) -> str:
        return registry._tag(*self.cell)

    def to_dict(self) -> dict:
        return {'config_key': self.config_key, 'e3_arm': self.e3_arm,
                'config': self.config.to_dict(), 'seeds': list(self.seeds),
                'run_digests': list(self.run_digests),
                'stats': {k: v.to_dict() for k, v in sorted(self.stats.items())}}

    @classmethod
    def from_dict(cls, cell: tuple[str, str],
                  data: Mapping[str, Any]) -> 'Candidate':
        return cls(cell=cell, config=CellConfig.from_dict(data['config']),
                   e3_arm=str(data['e3_arm']),
                   seeds=tuple(int(s) for s in data['seeds']),
                   run_digests=tuple(str(d) for d in data['run_digests']),
                   stats={k: EndpointStats.from_dict(v)
                          for k, v in data['stats'].items()})


@dataclass(frozen=True)
class CellSelection:
    """What was selected for one cell, and the evidence it was selected on."""

    cell: tuple[str, str]
    config: CellConfig
    e3_arm: str
    config_key: str
    seeds: tuple[int, ...]
    run_digests: tuple[str, ...]
    evidence: Mapping[str, Any]

    @property
    def cell_key(self) -> str:
        return registry._tag(*self.cell)

    @property
    def equals_a_priori(self) -> bool:
        return self.config.equals_a_priori

    def to_dict(self) -> dict:
        return {'arch': self.cell[0], 'target_rule': self.cell[1],
                'config_key': self.config_key, 'e3_arm': self.e3_arm,
                'config': self.config.to_dict(),
                'equals_a_priori': bool(self.equals_a_priori),
                'seeds': list(self.seeds),
                'run_digests': list(self.run_digests),
                'evidence': _plain(self.evidence)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> 'CellSelection':
        return cls(cell=(str(data['arch']), str(data['target_rule'])),
                   config=CellConfig.from_dict(data['config']),
                   e3_arm=str(data['e3_arm']),
                   config_key=str(data['config_key']),
                   seeds=tuple(int(s) for s in data['seeds']),
                   run_digests=tuple(str(d) for d in data['run_digests']),
                   evidence=dict(data.get('evidence') or {}))


@dataclass(frozen=True)
class Selection:
    """A complete, content-addressed per-cell selection.

    `cells` covers all four cells of `registry.CELLS` or the object does not
    exist: `ANALYSIS_PLAN.md` 2.3 refuses an incomplete selection, and a
    partially-tuned secondary policy would silently run some cells under the
    primary one while the artifact claimed otherwise.
    """

    selection_id: str
    schema: str
    rule: Mapping[str, Any]
    seed_block: str
    seeds: tuple[int, ...]
    env: str
    source_experiment: str
    cells: Mapping[str, CellSelection]
    candidates: Mapping[str, tuple[Candidate, ...]]
    plans: Mapping[str, Optional[str]]
    recorded: Mapping[str, Any]

    # -- reading -----------------------------------------------------------
    @property
    def is_placeholder(self) -> bool:
        """True where the artifact was computed under a placeholder rule."""
        return bool(self.rule.get('placeholder', True))

    @property
    def short_id(self) -> str:
        return self.selection_id[:12]

    def config_for(self, cell: tuple[str, str] | str) -> CellConfig:
        """The selected configuration for one cell. KeyError if absent."""
        return self.cells[_cell_key(cell)].config

    def overrides_for(self, cell: tuple[str, str] | str) -> dict:
        """The `Config` fields a tuned arm of this cell sets."""
        return self.config_for(cell).overrides()

    def equals_a_priori(self, cell: tuple[str, str] | str) -> bool:
        """Whether this cell's tuned arms share the common policy's runs."""
        return self.cells[_cell_key(cell)].equals_a_priori

    @property
    def shared_cells(self) -> tuple[str, ...]:
        return tuple(k for k, v in sorted(self.cells.items())
                     if v.equals_a_priori)

    # -- serialisation -----------------------------------------------------
    def to_dict(self) -> dict:
        payload = self.addressed_payload()
        payload['selection_id'] = self.selection_id
        payload[UNADDRESSED_KEY] = _plain(self.recorded)
        return payload

    def addressed_payload(self) -> dict:
        """Everything the content address covers, in canonical order."""
        return {
            'schema': self.schema,
            'policy': registry.TUNED_POLICY,
            'source_experiment': self.source_experiment,
            'seed_block': self.seed_block,
            'seeds': list(self.seeds),
            'env': self.env,
            'a_priori': A_PRIORI_CONFIG.to_dict(),
            'rule': _plain(self.rule),
            'plans': dict(sorted(self.plans.items())),
            'cells': {k: v.to_dict() for k, v in sorted(self.cells.items())},
            'candidates': {k: [c.to_dict() for c in v]
                           for k, v in sorted(self.candidates.items())},
        }

    def computed_id(self) -> str:
        return content_address(self.addressed_payload())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> 'Selection':
        if str(data.get('schema')) != SELECTION_SCHEMA:
            raise SelectionCorrupt(
                f'selection schema {data.get("schema")!r} is not '
                f'{SELECTION_SCHEMA!r}. The meaning of a stored field changed '
                f'between schemas, so this artifact cannot be read as one of '
                f'the current version; recompute the selection.')
        cells = {k: CellSelection.from_dict(v)
                 for k, v in (data.get('cells') or {}).items()}
        cands: dict[str, tuple[Candidate, ...]] = {}
        for key, rows in (data.get('candidates') or {}).items():
            cell = _cell_pair(key)
            cands[key] = tuple(Candidate.from_dict(cell, r) for r in rows)
        return cls(selection_id=str(data['selection_id']),
                   schema=str(data['schema']),
                   rule=dict(data.get('rule') or {}),
                   seed_block=str(data['seed_block']),
                   seeds=tuple(int(s) for s in data['seeds']),
                   env=str(data['env']),
                   source_experiment=str(data['source_experiment']),
                   cells=cells, candidates=cands,
                   plans=dict(data.get('plans') or {}),
                   recorded=dict(data.get(UNADDRESSED_KEY) or {}))

    def describe(self) -> str:
        lines = [f'selection {self.short_id}  rule={self.rule.get("id")}'
                 f'{"  [PLACEHOLDER]" if self.is_placeholder else ""}',
                 f'  block={self.seed_block} seeds={list(self.seeds)} '
                 f'env={self.env} from={self.source_experiment}']
        for key in sorted(self.cells):
            sel = self.cells[key]
            mark = 'shares common-policy runs' if sel.equals_a_priori \
                else 'own runs'
            lines.append(f'  {key:<16} lr={sel.config.lr:g} '
                         f'{sel.config.target_update}'
                         f'/{sel.config.target_update_freq}  ({mark})')
        return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _or_inf(value: Optional[float]) -> float:
    """A missing mean sorts last under every rule, rather than crashing one."""
    return float('-inf') if value is None else float(value)


def _opt_float(value: Any) -> Optional[float]:
    return None if value is None else float(value)


def _cell_key(cell: tuple[str, str] | str) -> str:
    return cell if isinstance(cell, str) else registry._tag(*cell)


def _cell_pair(key: tuple[str, str] | str) -> tuple[str, str]:
    if not isinstance(key, str):
        return (str(key[0]), str(key[1]))
    arch, _, rule = key.partition('-')
    return (arch, rule)


def _plain(obj: Any) -> Any:
    """JSON-ready copy: tuples to lists, mappings to dicts, sorted keys."""
    if isinstance(obj, Mapping):
        return {str(k): _plain(v) for k, v in sorted(obj.items(),
                                                     key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj


def canonical_json(payload: Mapping[str, Any]) -> str:
    """The exact bytes a content address is taken over."""
    return json.dumps(_plain(payload), sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False)


def content_address(payload: Mapping[str, Any]) -> str:
    """blake2b-128 of the canonical JSON, matching `Config.digest`'s scheme."""
    return hashlib.blake2b(canonical_json(payload).encode('utf-8'),
                           digest_size=16).hexdigest()


def _summarise(endpoint: str, pairs: Sequence[tuple[int, Optional[float]]]
               ) -> EndpointStats:
    """Mean, variance and sd of one endpoint over seed-ordered values."""
    ordered = [v for _s, v in sorted(pairs, key=lambda p: p[0])]
    good = [float(v) for v in ordered if v is not None and not _isnan(v)]
    n = len(good)
    if n == 0:
        return EndpointStats(endpoint, 0, (), None, None, None, None)
    mean = sum(good) / n
    ss = sum((v - mean) ** 2 for v in good)
    var = ss / (n - 1) if n > 1 else None
    var_pop = ss / n
    sd = math.sqrt(var) if var is not None else None
    return EndpointStats(endpoint, n, tuple(good), mean, var, var_pop, sd)


def _isnan(value: Any) -> bool:
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return True


# ---------------------------------------------------------------------------
# The declared candidate set, read from the catalogue
# ---------------------------------------------------------------------------
def candidate_grid() -> dict[str, dict[str, tuple[str, CellConfig]]]:
    """`E3`'s configurations, per cell, keyed by `config_key`.

    Read from `registry.EXPERIMENTS['E3']` rather than restated, so the set a
    selection may choose from is exactly the set that was run. A row of an
    aggregated table naming a configuration that is not in this grid is refused
    rather than admitted as a ninth candidate: it would mean the table and the
    catalogue disagree about what `E3` is, and the selection would then be made
    over a set nobody pre-registered.
    """
    grid: dict[str, dict[str, tuple[str, CellConfig]]] = {}
    freq_default = int(registry.COMMON['target_update_freq'])
    tau_default = float(_config_default('tau'))
    for arm in registry.EXPERIMENTS[SOURCE_EXPERIMENT].arms:
        ov = arm.overrides
        cell = registry._tag(str(ov['arch']), str(ov['target_rule']))
        cfg = CellConfig(
            lr=float(ov['lr']),
            target_update=str(ov['target_update']),
            target_update_freq=int(ov.get('target_update_freq', freq_default)),
            tau=float(ov.get('tau', tau_default)))
        grid.setdefault(cell, {})[cfg.config_key] = (arm.label, cfg)
    return grid


def _match_config(cell_grid: Mapping[str, tuple[str, CellConfig]],
                  lr: float, target_update: str) -> Optional[str]:
    """The grid key for an (lr, rule) pair, matched on value not on spelling."""
    for key, (_label, cfg) in cell_grid.items():
        if cfg.target_update == target_update \
                and math.isclose(cfg.lr, lr, rel_tol=1e-9, abs_tol=0.0):
            return key
    return None


# ---------------------------------------------------------------------------
# Computing a selection
# ---------------------------------------------------------------------------
def _rows_of(table: Any) -> list[dict]:
    """Accept a DataFrame, a list of mappings, or a csv.DictReader."""
    if hasattr(table, 'to_dict') and hasattr(table, 'columns'):
        return list(table.to_dict('records'))       # pandas, without importing it
    return [dict(row) for row in table]


def compute_selection(table: Any, *,
                      rule: Optional[Callable[[tuple[str, str],
                                               Sequence[Candidate]], str]] = None,
                      require_complete: bool = True,
                      plans: Optional[Mapping[str, Optional[str]]] = None,
                      generator: str = '',
                      record_git: bool = True) -> Selection:
    """Compute the per-cell selection from an aggregated `E3` table.

    `table` is `per_seed.csv` in any of the shapes it naturally arrives in: a
    pandas DataFrame, a list of dicts, or a `csv.DictReader`. Rows for other
    experiments are ignored, rows for `E3` are used, and a row whose
    `experiments` column does not mention `E3` at all is never read as evidence
    -- the selection is over `E3`, and a table filtered by someone else is still
    checked here rather than trusted.

    Every refusal below is a refusal *by design*, and each names the rule it
    enforces:

    * a seed outside `TUNE` (`DESIGN.md` 3.4, `ANALYSIS_PLAN.md` 8);
    * fewer than the four cells of `registry.CELLS` (`ANALYSIS_PLAN.md` 2.3);
    * a configuration that is not one of `E3`'s (`candidate_grid`);
    * with `require_complete`, a candidate missing any `TUNE` seed, or a cell
      missing any of `E3`'s eight configurations. `ANALYSIS_PLAN.md` 2.3 states
      the input as eight configurations per cell at five seeds each, so a
      selection made over less than that is a selection over a different input.
      The flag exists because the completeness bar belongs to the rule and the
      rule is swappable; it defaults to enforcing what 2.3 states.

    Two further refusals are the *rule's* and are raised from it rather than
    here, because they are properties of the criterion and not of the table:
    2.3's competence floor, and its refusal of a candidate not measured at all
    five `TUNE` seeds. The second is deliberately checked in both places, since
    `require_complete=False` relaxes this function's copy and the installed rule
    is written for an n=5 standard error either way.

    Nothing here reads the run tree. The evidence is the table, the identity of
    the runs behind it is the `run_digest` column, and the selection is a pure
    function of (table, rule, plans).
    """
    rule = SELECTION_RULE if rule is None else rule
    rows = _rows_of(table)
    grid = candidate_grid()
    block = tuple(registry.SEED_BLOCKS[SELECTION_SEED_BLOCK])
    allowed_seeds = set(block)

    # (cell_key, config_key) -> {seed: (run_digest, {endpoint: value})}
    gathered: dict[tuple[str, str], dict[int, tuple[str, dict]]] = {}
    seen_rows = 0
    for row in rows:
        exps = _experiments_of(row)
        if exps is not None and SOURCE_EXPERIMENT not in exps:
            continue
        if exps is None and str(row.get('experiment') or '') not in \
                ('', SOURCE_EXPERIMENT):
            continue
        arch, target_rule = _cell_of(row)
        cell_key = registry._tag(arch, target_rule)
        if cell_key not in grid:
            raise SelectionIncomplete(
                f'row names cell {cell_key!r}, which is not one of '
                f'{sorted(grid)}. The selection is over the 2x2 of '
                f'DESIGN.md 3.3 and nothing else.')
        seed = _int_of(row, 'seed')
        if seed is None:
            raise SelectionIncomplete(
                f'an {SOURCE_EXPERIMENT} row for {cell_key} carries no seed, so '
                f'it cannot be checked against the {SELECTION_SEED_BLOCK} block')
        if seed not in allowed_seeds:
            raise SeedBlockViolation(
                f'seed {seed} is outside the {SELECTION_SEED_BLOCK} block '
                f'{list(block)}. DESIGN.md 3.4 reserves {SELECTION_SEED_BLOCK} '
                f'for hyperparameter selection and ANALYSIS_PLAN.md 8 forbids a '
                f'reported estimate drawing on it; revision 1 selected on seeds '
                f'0-4 and ran confirmatory arms on 0-9, so half of every '
                f'confirmatory sample had been tuned on. Selecting on {seed} '
                f'would repeat that in the other direction.')
        declared_block = str(row.get('seed_block') or '')
        if declared_block and declared_block != SELECTION_SEED_BLOCK:
            raise SeedBlockViolation(
                f'seed {seed} is in the {SELECTION_SEED_BLOCK} block but its '
                f'row declares seed_block={declared_block!r}. The table and the '
                f'catalogue disagree about which block this run belongs to, and '
                f'a selection cannot be computed while they do.')
        lr = _float_of(row, 'lr')
        target_update = str(row.get('target_update') or '')
        if lr is None or not target_update:
            raise SelectionIncomplete(
                f'{cell_key} seed {seed}: the row carries no lr/target_update, '
                f'so the run cannot be matched to an {SOURCE_EXPERIMENT} '
                f'configuration')
        config_key = _match_config(grid[cell_key], lr, target_update)
        if config_key is None:
            raise SelectionIncomplete(
                f'{cell_key} seed {seed}: configuration lr={lr:g} '
                f'{target_update} is not one of {SOURCE_EXPERIMENT}\'s '
                f'{sorted(grid[cell_key])}. The candidate set is the catalogue '
                f'and a table that disagrees with it is not evidence about '
                f'{SOURCE_EXPERIMENT}.')
        _check_rule_params(row, grid[cell_key][config_key][1], cell_key, seed)
        digest = str(row.get('run_digest') or '')
        values = {ep: _float_of(row, ep) for ep in SELECTION_ENDPOINTS}
        slot = gathered.setdefault((cell_key, config_key), {})
        if seed in slot and slot[seed][0] != digest:
            raise SelectionIncomplete(
                f'{cell_key} {config_key} seed {seed} appears twice with '
                f'different run digests ({slot[seed][0]} and {digest}). One '
                f'configuration at one seed is one run; two rows mean the table '
                f'was concatenated from trees that do not agree.')
        slot[seed] = (digest, values)
        seen_rows += 1

    if not seen_rows:
        raise SelectionIncomplete(
            f'the table carries no {SOURCE_EXPERIMENT} rows at all. '
            f'{SOURCE_EXPERIMENT} runs on the {SELECTION_SEED_BLOCK} block; '
            f'aggregate a tree that has them before selecting.')

    candidates: dict[str, list[Candidate]] = {}
    for cell_key in sorted(grid):
        cell = _cell_pair(cell_key)
        rows_for_cell: list[Candidate] = []
        for config_key in sorted(grid[cell_key]):
            arm_label, cfg = grid[cell_key][config_key]
            slot = gathered.get((cell_key, config_key), {})
            if not slot:
                if require_complete:
                    raise SelectionIncomplete(
                        f'{cell_key} has no runs for configuration '
                        f'{config_key} ({arm_label}). ANALYSIS_PLAN.md 2.3 '
                        f'states the input as eight configurations per cell at '
                        f'five {SELECTION_SEED_BLOCK} seeds; selecting over a '
                        f'truncated grid selects over a different input. Finish '
                        f'{SOURCE_EXPERIMENT} first, or pass '
                        f'require_complete=False deliberately.')
                continue
            missing = sorted(allowed_seeds - set(slot))
            if missing and require_complete:
                raise SelectionIncomplete(
                    f'{cell_key} {config_key} ({arm_label}) is missing '
                    f'{SELECTION_SEED_BLOCK} seeds {missing}. A candidate '
                    f'measured at fewer seeds than its rivals is compared '
                    f'against them on unequal evidence.')
            seeds = tuple(sorted(slot))
            stats = {ep: _summarise(ep, [(s, slot[s][1].get(ep))
                                         for s in seeds])
                     for ep in SELECTION_ENDPOINTS}
            rows_for_cell.append(Candidate(
                cell=cell, config=cfg, e3_arm=arm_label, seeds=seeds,
                run_digests=tuple(slot[s][0] for s in seeds), stats=stats))
        if not rows_for_cell:
            raise SelectionIncomplete(
                f'{cell_key} has no {SOURCE_EXPERIMENT} evidence at all. '
                f'ANALYSIS_PLAN.md 2.3 refuses an incomplete selection: all '
                f'{len(registry.CELLS)} cells must be present.')
        candidates[cell_key] = rows_for_cell

    missing_cells = [registry._tag(*c) for c in registry.CELLS
                     if registry._tag(*c) not in candidates]
    if missing_cells:
        raise SelectionIncomplete(
            f'no evidence for cells {missing_cells}. ANALYSIS_PLAN.md 2.3 '
            f'refuses a selection covering fewer than four cells, because the '
            f'secondary policy would then run some cells under the primary one '
            f'while the artifact claimed otherwise.')

    cells: dict[str, CellSelection] = {}
    for cell_key in sorted(candidates):
        cell = _cell_pair(cell_key)
        pool = tuple(candidates[cell_key])
        chosen_key = rule(cell, pool)
        chosen = next((c for c in pool if c.config_key == chosen_key), None)
        if chosen is None:
            raise SelectionError(
                f'the selection rule returned {chosen_key!r} for {cell_key}, '
                f'which is not one of its candidates '
                f'{[c.config_key for c in pool]}. A rule must return a '
                f'config_key from the pool it was handed.')
        a_priori = next((c for c in pool
                         if c.config.equals_a_priori), None)
        # The arithmetic the rule decided on, where the rule offers it. An
        # optional attribute rather than a second return value: the contract is
        # one `config_key`, every fixture rule honours only that, and a rule
        # that cannot explain itself records no working instead of failing. It
        # is recomputed rather than captured from the call above so that this
        # function stays a pure function of (table, rule, plans), and it goes
        # inside the evidence and therefore inside the content address, so the
        # number that decided a cell cannot be edited out of the record.
        explain = getattr(rule, 'explain', None)
        working = _plain(explain(cell, pool)) if callable(explain) else None
        evidence = {
            'endpoints': list(SELECTION_ENDPOINTS),
            'selected': {ep: chosen.stats[ep].to_dict()
                         for ep in SELECTION_ENDPOINTS},
            'a_priori_config_key': (a_priori.config_key if a_priori else None),
            'a_priori': ({ep: a_priori.stats[ep].to_dict()
                          for ep in SELECTION_ENDPOINTS} if a_priori else None),
            'n_candidates': len(pool),
            'rule_working': working,
        }
        cells[cell_key] = CellSelection(
            cell=cell, config=chosen.config, e3_arm=chosen.e3_arm,
            config_key=chosen.config_key, seeds=chosen.seeds,
            run_digests=chosen.run_digests, evidence=evidence)

    rule_record = {
        'id': SELECTION_RULE_ID if rule is SELECTION_RULE else _rule_name(rule),
        'placeholder': bool(SELECTION_RULE_IS_PLACEHOLDER
                            if rule is SELECTION_RULE else True),
        'endpoints': list(SELECTION_ENDPOINTS),
        'description': (SELECTION_RULE_DESCRIPTION if rule is SELECTION_RULE
                        else f'rule supplied by the caller: {_rule_name(rule)}'),
        'require_complete': bool(require_complete),
    }
    if rule is SELECTION_RULE:
        # Recorded only for the installed rule. A caller-supplied rule does not
        # obey these numbers, and stamping them onto its artifact would
        # misdescribe the decision that was actually made.
        rule_record['parameters'] = _plain(SELECTION_RULE_PARAMETERS)
    plans_record = dict(plans) if plans is not None else provenance.plan_hashes()
    recorded = {
        'created_utc': _datetime.datetime.now(
            _datetime.timezone.utc).replace(microsecond=0).isoformat(),
        'generator': generator or 'experiments/tuning.py compute_selection',
        'git': provenance.git_state() if record_git else None,
    }
    selection = Selection(
        selection_id='', schema=SELECTION_SCHEMA, rule=rule_record,
        seed_block=SELECTION_SEED_BLOCK, seeds=block, env=SELECTION_ENV,
        source_experiment=SOURCE_EXPERIMENT, cells=cells,
        candidates={k: tuple(v) for k, v in candidates.items()},
        plans=dict(plans_record), recorded=recorded)
    return _readdress(selection)


def _readdress(selection: Selection) -> Selection:
    payload = selection.addressed_payload()
    return Selection(selection_id=content_address(payload),
                     schema=selection.schema, rule=selection.rule,
                     seed_block=selection.seed_block, seeds=selection.seeds,
                     env=selection.env,
                     source_experiment=selection.source_experiment,
                     cells=selection.cells, candidates=selection.candidates,
                     plans=selection.plans, recorded=selection.recorded)


def _rule_name(rule: Callable) -> str:
    return getattr(rule, '__name__', repr(rule))


def _experiments_of(row: Mapping[str, Any]) -> Optional[frozenset[str]]:
    raw = row.get('experiments')
    if raw is None or raw == '' or _isnan_str(raw):
        return None
    if isinstance(raw, (list, tuple, set, frozenset)):
        return frozenset(str(x) for x in raw)
    return frozenset(x for x in str(raw).split(';') if x)


def _isnan_str(value: Any) -> bool:
    return isinstance(value, float) and math.isnan(value)


def _cell_of(row: Mapping[str, Any]) -> tuple[str, str]:
    arch = str(row.get('arch') or '')
    target_rule = str(row.get('target_rule') or '')
    if arch and target_rule:
        return arch, target_rule
    cell = str(row.get('cell') or '')
    if cell:
        return _cell_pair(cell)
    raise SelectionIncomplete(
        'a row carries neither arch/target_rule nor cell, so it cannot be '
        'attributed to one of the four cells')


def _check_rule_params(row: Mapping[str, Any], cfg: CellConfig,
                       cell_key: str, seed: int) -> None:
    """Cross-check `target_update_freq`/`tau` where the table carries them.

    `aggregate.PER_SEED_COLUMNS` exports `target_update` but neither the hard
    update period nor the soft coefficient, so the catalogue supplies them. Where
    a caller passes a richer table, a disagreement is a refusal rather than a
    silent preference for the catalogue: it would mean the run was not the
    configuration the arm declares.
    """
    for field, expected in (('target_update_freq', cfg.target_update_freq),
                            ('tau', cfg.tau)):
        if field not in row:
            continue
        got = _float_of(row, field)
        if got is None:
            continue
        if not math.isclose(float(got), float(expected), rel_tol=1e-9,
                            abs_tol=0.0):
            raise SelectionIncomplete(
                f'{cell_key} seed {seed}: the table says {field}={got:g} but '
                f'{SOURCE_EXPERIMENT} declares {expected:g} for '
                f'{cfg.config_key}. The run is not the configuration its arm '
                f'names, so it is not evidence about that candidate.')


def _float_of(row: Mapping[str, Any], key: str) -> Optional[float]:
    value = row.get(key)
    if value is None or value == '':
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) else out


def _int_of(row: Mapping[str, Any], key: str) -> Optional[int]:
    value = _float_of(row, key)
    return None if value is None else int(value)


# ---------------------------------------------------------------------------
# Storing and reading a selection
# ---------------------------------------------------------------------------
def selection_path(out_root: str = 'runs') -> str:
    """Where the active selection lives for a run tree."""
    return os.path.join(out_root, SELECTION_RELPATH)


def archive_path(selection_id: str, out_root: str = 'runs') -> str:
    return os.path.join(out_root, SELECTION_ARCHIVE_RELDIR,
                        f'{selection_id}.json')


def write_selection(selection: Selection, out_root: str = 'runs',
                    replace: bool = False) -> str:
    """Store a selection and make it the active one. Returns the active path.

    Two copies are written: an immutable one under `SELECTION_ARCHIVE_RELDIR`
    named by content, and the active pointer at `SELECTION_RELPATH` that
    `registry` enumerates the tuned arms from.

    Replacing an active selection with a *different* one is refused unless
    `replace=True`, and the refusal is the point. The tuned arms' run digests
    are a function of the selection, so re-pointing it after tuned runs exist
    silently orphans them: they stay on disk, they are no longer declared by any
    arm, and the completeness check reports the new arms missing while the old
    runs become unattributable.
    """
    if not isinstance(selection, Selection):
        raise TypeError('write_selection takes a Selection')
    expect = selection.computed_id()
    if selection.selection_id != expect:
        raise SelectionCorrupt(
            f'selection_id {selection.selection_id!r} does not match its own '
            f'content ({expect!r}); it was mutated after being computed')

    active = selection_path(out_root)
    try:
        existing = read_selection(out_root, required=False, verify=False,
                                  warn_placeholder=False)
        held = existing.short_id if existing is not None else None
    except SelectionCorrupt:
        # Unreadable, but present. It is still something the tuned arms may
        # have been enumerated from, so it is protected exactly as a readable
        # one is: refused unless the caller says to replace it.
        held = 'an unreadable artifact'
        existing = None
    if held is not None and held != selection.short_id and not replace:
        raise SelectionError(
            f'{active} already holds {held} and this is {selection.short_id}. '
            f'The tuned arms\' run digests are a function of the selection, so '
            f'replacing it orphans every tuned run already on disk: they stay '
            f'in the tree, no arm declares them any more, and the new arms are '
            f'reported missing. Pass replace=True only after deciding what '
            f'happens to those runs.')

    payload = selection.to_dict()
    archive = archive_path(selection.selection_id, out_root)
    _write_json(archive, payload)
    _write_json(active, payload)
    return active


def _write_json(path: str, payload: Mapping[str, Any]) -> None:
    """Atomic write: a torn artifact is one nothing can enumerate from."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    tmp = f'{path}.tmp'
    with open(tmp, 'w', encoding='utf-8', newline='\n') as fh:
        json.dump(_plain(payload), fh, sort_keys=True, indent=2)
        fh.write('\n')
    os.replace(tmp, path)


def read_selection(out_root: str = 'runs', *, path: Optional[str] = None,
                   required: bool = True, verify: bool = True,
                   warn_placeholder: bool = True) -> Optional[Selection]:
    """Read the stored selection. Never recomputes it.

    `required=False` returns None where there is none, which is what a caller
    asking "is the tuned stage available yet" wants. `required=True` raises
    `SelectionMissing` with the commands that produce one, because the failure
    to avoid is a silent empty enumeration of the tuned arms.

    The content address is re-checked on every read unless `verify=False`.
    A selection is a pre-registration artifact: an edit to it after the tuned
    runs exist changes which arms those runs belong to, and the only way to make
    that visible is to refuse an artifact that does not hash to its own id.
    """
    target = path or selection_path(out_root)
    if not os.path.exists(target):
        if not required:
            return None
        raise SelectionMissing(missing_message(out_root, path=target))
    try:
        with open(target, encoding='utf-8') as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise SelectionCorrupt(
            f'{target} cannot be read as a selection artifact ({exc}). It is '
            f'the pre-registered record the tuned arms are enumerated from, so '
            f'it is not repaired in place: recompute it and write it again.'
        ) from exc
    selection = Selection.from_dict(data)
    if verify:
        expect = selection.computed_id()
        if expect != selection.selection_id:
            raise SelectionCorrupt(
                f'{target} carries selection_id {selection.short_id} but its '
                f'content hashes to {expect[:12]}. The artifact was edited '
                f'after it was computed. Every tuned run on disk was enumerated '
                f'from the id it claims, so this file no longer says which arms '
                f'those runs belong to; recompute the selection rather than '
                f'adjusting the file.')
    if warn_placeholder and selection.is_placeholder:
        warnings.warn(
            f'selection {selection.short_id} was computed under '
            f'{selection.rule.get("id")}, a PLACEHOLDER rule, not the '
            f'pre-registered criterion of ANALYSIS_PLAN.md 2.3. Arms enumerated '
            f'from it are for testing the pipeline and are not a result.',
            RuntimeWarning, stacklevel=2)
    return selection


def missing_message(out_root: str = 'runs', path: Optional[str] = None) -> str:
    """What to run to produce a selection, named exactly.

    `registry` raises this text when a tuned arm cannot be enumerated. A message
    that only said "no selection" would leave the reader to rediscover that the
    tuned stage is sequentially dependent on `E3` (`DESIGN.md` 3.3), which is
    the whole reason it cannot be enumerated yet.
    """
    target = path or selection_path(out_root)
    return (
        f'no tuning selection at {target}. The secondary policy of DESIGN.md '
        f'3.3 is each cell\'s own {SOURCE_EXPERIMENT}-selected configuration, '
        f'so the tuned arms are sequentially dependent on {SOURCE_EXPERIMENT} '
        f'and cannot be enumerated before a selection exists. In order:\n'
        f'  python experiments/sweep.py --experiments {SOURCE_EXPERIMENT} '
        f'--out-root {out_root}\n'
        f'  python experiments/aggregate.py --out-root {out_root}\n'
        f'  python experiments/tuning.py select --out-root {out_root} --write\n'
        f'The last step refuses unless all four cells are covered at the '
        f'{SELECTION_SEED_BLOCK} seeds '
        f'{list(registry.SEED_BLOCKS[SELECTION_SEED_BLOCK])}.')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _load_csv(path: str) -> list[dict]:
    with open(path, newline='', encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description='Compute, store and inspect the per-cell tuning selection '
                    'of DESIGN.md 3.3.')
    sub = parser.add_subparsers(dest='command', required=True)

    sel = sub.add_parser('select', help='compute a selection from per_seed.csv')
    sel.add_argument('--out-root', default='runs')
    sel.add_argument('--per-seed', default=None,
                     help='default: <out-root>/per_seed.csv')
    sel.add_argument('--write', action='store_true',
                     help='store it and make it the active selection')
    sel.add_argument('--replace', action='store_true',
                     help='replace a different active selection (see '
                          'write_selection: this can orphan tuned runs)')
    sel.add_argument('--allow-incomplete', action='store_true',
                     help='compute over a truncated E3 grid; not a selection '
                          'anything may be reported from')

    show = sub.add_parser('show', help='print the stored selection')
    show.add_argument('--out-root', default='runs')

    args = parser.parse_args(argv)

    if args.command == 'show':
        selection = read_selection(args.out_root, required=False)
        if selection is None:
            print(missing_message(args.out_root))
            return 1
        print(selection.describe())
        return 0

    per_seed = args.per_seed or os.path.join(args.out_root, 'per_seed.csv')
    if not os.path.exists(per_seed):
        print(f'no aggregated table at {per_seed}; run aggregate.py first')
        return 1
    try:
        selection = compute_selection(
            _load_csv(per_seed), require_complete=not args.allow_incomplete,
            generator=f'experiments/tuning.py select --per-seed {per_seed}')
    except SelectionError as exc:
        # A refusal is an outcome of the pre-registered rule, not a crash, so
        # it is reported as one: the message names the cell, the candidate and
        # the clause of ANALYSIS_PLAN.md 2.3 that refused, and the exit status
        # is non-zero so a runner cannot proceed to enumerate tuned arms from a
        # selection that does not exist.
        print(f'REFUSED ({type(exc).__name__}): {exc}')
        return 1
    print(selection.describe())
    if args.write:
        path = write_selection(selection, args.out_root, replace=args.replace)
        print(f'written: {path}')
        print(f'archived: {archive_path(selection.selection_id, args.out_root)}')
    else:
        print('not written (pass --write)')
    return 0


__all__ = ['SELECTION_RULE', 'SELECTION_RULE_ID',
           'SELECTION_RULE_IS_PLACEHOLDER', 'SELECTION_RULE_DESCRIPTION',
           'SELECTION_RULE_PARAMETERS', 'SELECTION_ENDPOINTS',
           'SELECTION_CRITERION_ENDPOINT', 'COMPETENCE_ENDPOINT',
           'COMPETENCE_FLOOR', 'SELECTION_N', 'SELECTION_VARIANCE_FIELD',
           'preregistered_selection_rule', 'selection_working',
           'placeholder_selection_rule',
           'SELECTION_SCHEMA', 'SELECTION_SEED_BLOCK', 'SELECTION_ENV',
           'SOURCE_EXPERIMENT', 'SELECTION_RELPATH',
           'SELECTION_ARCHIVE_RELDIR', 'UNADDRESSED_KEY',
           'A_PRIORI_CONFIG', 'CellConfig', 'EndpointStats', 'Candidate',
           'CellSelection', 'Selection',
           'SelectionError', 'SelectionMissing', 'SelectionIncomplete',
           'SeedBlockViolation', 'SelectionCorrupt', 'SelectionRefused',
           'candidate_grid', 'compute_selection', 'read_selection',
           'write_selection', 'selection_path', 'archive_path',
           'missing_message', 'canonical_json', 'content_address', 'main']


if __name__ == '__main__':
    raise SystemExit(main())
