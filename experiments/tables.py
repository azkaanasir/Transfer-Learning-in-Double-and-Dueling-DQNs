"""LaTeX (booktabs) tables with a markdown mirror, generated from the run data.

Answers **C13** in `REVIEW_COVERAGE.md`: the published manuscript contains zero
tables and delivers twelve summary statistics as bulleted prose, so a reader
cannot see the arms side by side, cannot see which quantity was tested, and
cannot check a number against the configuration that produced it. ICANN #2 asked
for two tables; this module emits six, because the audit found four more things
the prose was hiding.

What each design choice here is defending against, defect by defect:

* **A number in the paper that nobody can trace.** `STANDING_INSTRUCTIONS` S6:
  "never hand-compute a number that appears in the paper". Every cell below is
  read from `runs/per_seed.csv` (the schema `aggregate.py` pins) or from
  `stats.py --json`, and every table is written beside a
  `<table>.provenance.json` recording the hash of both inputs, the hash of the
  emitted file, the git commit and the `ANALYSIS_PLAN.md` hash in force
  (`DESIGN.md` 8.3; 9, "stale artifacts").
* **A caption that contradicts its table.** The guard `plots.py` applies to
  figures for C8, applied to tables. Captions here are *generated*: the seed
  count, the endpoint definition, the test, the normalisation and the
  provenance stamp are filled in from the data, so a caption cannot claim an n
  or an endpoint the table does not have.
* **A blank read as an omission.** An estimation-only row has no p-value *by
  design* -- `ANALYSIS_PLAN.md` 7 emits p-values inside exactly one family of
  eight -- so its p column carries an explicit em dash and a footnote saying
  so. A blank cell would read as a number the authors declined to report,
  which is the opposite of the claim being made.
* **A suppressed test rendered as a null result.** Where `stats.py` refused a
  confirmatory member (an incomplete arm, an ambiguous primary arm, an
  invariant that moved, n<3) the p columns carry the refusal, not a dash and
  not a number. Refusing and saying why is the point of the refusal.
* **Cross-architecture return presented as a transfer effect** (`DESIGN.md` 9).
  `main_results` carries `transferred_param_fraction` and **headroom** on every
  row automatically, so the two confounds of 2.5 -- treatment intensity and
  ceiling proximity -- are visible in the same glance as the effect.
* **An unreproducible protocol.** The manuscript described a freeze schedule
  that was never implemented and a layer set it never listed. `protocol_summary`
  is the methods table whose absence made that possible: per arm, the transfer
  set, the layers copied / partially copied / reinitialised, the parameter
  fraction, the freeze group and `freeze_updates` in **gradient updates**
  (`DESIGN.md` 3.2).
* **A shift axis that is silently a difficulty axis.** `environments` derives
  the difficulty-confound flag from the *measured* no-op score in
  `reference_returns.json`, so the caveat `DESIGN.md` 5.1 attaches to the
  gravity family cannot be dropped from a table by forgetting to type it.
* **A power claim invented after the fact.** `power` multiplies the
  pre-registered multipliers of `ANALYSIS_PLAN.md` 6.2 (imported from
  `stats.py`, asserted by `statlib.py --self-test`, and *not* re-tuned per 6.4)
  by the dispersion actually observed, and flags a cell as unpowered against
  the plan's own reference of one score unit.
* **A single-seed number quoted as a result.** When any tabulated arm has
  n < 3, every caption carries the `PIPELINE VALIDATION - NOT A RESULT` stamp
  (`ANALYSIS_PLAN.md` 9, `STANDING_INSTRUCTIONS` S8).

Nothing here computes an inference. Tables 2, 3 and 6 take their estimates from
`stats.py --json`; without `--stats` those tables are emitted with an explicit
refusal in place of their numbers rather than with numbers this module derived
on its own, because a second implementation of the plan is a second chance to
diverge from it.

Formatting is deliberately plain: booktabs rules, no siunitx, no S columns, no
resizebox, no colour. The output compiles in a stock IEEE or LNCS template with
the booktabs package and nothing else, and non-ASCII is mapped to LaTeX
commands rather than relying on inputenc.

    python experiments/tables.py --per-seed runs/per_seed.csv --outdir paper/tables
    python experiments/tables.py --per-seed runs/per_seed.csv --stats stats.json
    python experiments/tables.py --self-test
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from experiments import registry                                  # noqa: E402
from experiments import stats as statsmod                         # noqa: E402
from src.dqn import envs, provenance                              # noqa: E402

# ===========================================================================
# 1. Constants. Every threshold and every family definition is imported from
#    the module that owns it and never restated: a constant copied into a
#    second file is a constant that can drift, which is how the published study
#    came to claim identical hyperparameters across arms whose learning rates
#    differed by a factor of five.
# ===========================================================================

ALPHA = statsmod.ALPHA
ALPHA_STRICTEST = statsmod.ALPHA_STRICTEST
FAMILY_SIZE = statsmod.CONFIRMATORY_FAMILY_SIZE
CONFIRMATORY_ENDPOINTS = statsmod.CONFIRMATORY_ENDPOINTS
EQUIVALENCE_MARGIN = statsmod.EQUIVALENCE_MARGIN
MIN_N_FOR_INFERENCE = statsmod.MIN_N_FOR_INFERENCE
VALIDATION_STAMP = statsmod.VALIDATION_STAMP
CELL_ORDER = statsmod.CELL_ORDER
MDE_MULTIPLIERS = statsmod.MDE_MULTIPLIERS
PLANNING_SDS = statsmod.PLANNING_SDS
UNPOWERED_MDE = statsmod.UNPOWERED_MDE
SOURCE_VALIDITY_GATE = statsmod.SOURCE_VALIDITY_GATE
C4_LOWER_BOUND = statsmod.C4_LOWER_BOUND
EXCLUSION_RESTRICTIONS = statsmod.CONTROL_EXCLUSION_RESTRICTIONS

#: `DESIGN.md` 4: the protocol axis, in the order the control set is argued in.
CONDITION_ORDER: tuple[str, ...] = (
    'scratch', 'transfer', 'transfer_untrained', 'transfer_permuted')

#: The em dash that stands in the p column of an estimation-only row. It is a
#: character, not an absence: `ANALYSIS_PLAN.md` 7 permits p-values in exactly
#: one family, so "no p-value here" is a design statement and must read as one.
EM_DASH = '—'

#: What a cell says when the quantity was never recorded. Deliberately distinct
#: from EM_DASH: one means "not defined", the other "not measured".
NOT_RECORDED = 'n/r'

#: `DESIGN.md` 5.1. The no-op score is measured per variant; a variant whose
#: no-op score has moved this far from its base environment's has changed how
#: hard the task is as well as how it behaves, so a shift effect measured on it
#: is confounded with difficulty. The tolerance is the plan's own smallest
#: substantively interesting score difference (`ANALYSIS_PLAN.md` 4), reused so
#: that this table does not introduce a threshold of its own.
NOOP_DRIFT_TOLERANCE = EQUIVALENCE_MARGIN

#: `DESIGN.md` 2.1, inherited by every emitted claim.
SCOPE_CLAUSE = (
    'Scope (DESIGN.md 2.1): all effects are defined over the seed set actually '
    'run, for the stated (arch, target_rule) implementations at '
    'hidden=(128,128), head_units=64, Adam, the declared exploration schedule '
    'and episode budget, on the named environment pairs. No claim is made '
    'about the dueling decomposition or the double-Q update as algorithmic '
    'ideas.')

#: The normalisation sentence, printed on every table that shows a score.
NORMALISATION = (
    'Normalisation (DESIGN.md 5.1): score = (return - random_return) / '
    '(threshold - random_return), against random-policy references measured '
    'over 100 fixed-seed episodes per environment and per variant '
    '(experiments/reference_returns.json). A uniform-random policy scores 0 '
    'and the registered threshold scores 1.')

#: Endpoint definitions keyed by the `per_seed.csv` column, printed in the
#: caption of any table that shows the column -- because "final score" without
#: "over 100 held-out greedy episodes at each of the final k=3 checkpoints,
#: averaged" is the terminal-snapshot endpoint revision 1 was criticised for.
ENDPOINT_DEFS: dict[str, str] = {
    'final_score':
        'final_score (P1, co-primary): mean normalised score over 100 '
        'held-out greedy episodes at each of the final k=3 evaluation '
        'checkpoints, averaged (ANALYSIS_PLAN.md 1).',
    'auc_score':
        'auc_score (P2, co-primary): area under the normalised-score '
        'evaluation curve over env steps, divided by total env steps '
        '(ANALYSIS_PLAN.md 1).',
    'episode_length_final100':
        'episode length: mean episode length over the final 100 training '
        'episodes -- promised in the published manuscript and never reported '
        '(DESIGN.md 5.3).',
    'within_run_sd':
        'within_run_sd: SD of the score over the final 10 evaluation points, '
        'i.e. training instability. Not across-seed spread, which the '
        'published study conflated with it (DESIGN.md 5.3).',
    'probe_jumpstart_score':
        'probe_jumpstart: score after fitting only the head on a fixed batch '
        'of target transitions with the transferred trunk frozen; the '
        'interpretable jumpstart for a head-reinit arm (DESIGN.md 5.3).',
    'jumpstart_score':
        'jumpstart: 100-episode greedy score at episode 0 before any gradient '
        'step, interpretable only where the output head is transferred '
        '(DESIGN.md 5.3).',
}

#: Compact environment tags, so a wide table fits a page. The legend is emitted
#: with any table that uses them; a canonical name is never abbreviated
#: silently, and variant parameters are never dropped.
_ENV_ABBREV: dict[str, str] = {
    'CartPole-v1': 'CP', 'LunarLander-v3': 'LL',
    'Acrobot-v1': 'AC', 'MountainCar-v0': 'MC',
}


# ===========================================================================
# 2. LaTeX escaping. A table that does not compile is a table nobody checks,
#    and a character silently dropped from an arm label is a table that lies
#    about which arm a row belongs to. Both are refused: specials are escaped,
#    the non-ASCII this study actually emits is mapped to commands, and
#    anything unmapped is rendered visibly and recorded in the provenance
#    sidecar rather than replaced quietly.
# ===========================================================================

_BS = chr(92)

_LATEX_SPECIALS: dict[str, str] = {
    _BS: _BS + 'textbackslash{}',
    '&': _BS + '&', '%': _BS + '%', '$': _BS + '$', '#': _BS + '#',
    '_': _BS + '_', '{': _BS + '{', '}': _BS + '}',
    '~': _BS + 'textasciitilde{}', '^': _BS + 'textasciicircum{}',
}

#: The non-ASCII that legitimately appears in this study's prose and numbers.
_LATEX_UNICODE: dict[str, str] = {
    '±': '$' + _BS + 'pm$',        # every mean +/- sd cell
    '—': '---',                    # the estimation-only p marker
    '–': '--',
    '−': '-',                      # unicode minus, from numpy formatting
    '→': '$' + _BS + 'rightarrow$',
    '§': _BS + 'S',
    '×': '$' + _BS + 'times$',
    '≥': '$' + _BS + 'geq$',
    '≤': '$' + _BS + 'leq$',
    '≈': '$' + _BS + 'approx$',
    '≠': '$' + _BS + 'neq$',
    'α': '$' + _BS + 'alpha$',
    'θ': '$' + _BS + 'theta$',
    'ρ': '$' + _BS + 'rho$',
    'σ': '$' + _BS + 'sigma$',
    'Δ': '$' + _BS + 'Delta$',
    'μ': '$' + _BS + 'mu$',
    '’': "'", '‘': "'", '“': "``", '”': "''",
    '…': _BS + 'ldots{}',
    ' ': ' ',
}

#: Characters seen but not mapped, accumulated across one invocation so the
#: provenance sidecar can record them. Populated by `latex_escape`.
_UNMAPPED: dict[str, int] = {}


def latex_escape(text: Any) -> str:
    """Escape one cell for LaTeX, mapping non-ASCII to commands.

    Consumed character by character, so a replacement that itself contains a
    backslash or a dollar cannot be re-escaped. Whitespace is collapsed because
    a newline inside a tabular cell is a compile error, not a layout choice.
    """
    out: list[str] = []
    for ch in re.sub(r'\s+', ' ', str(text)).strip():
        if ch in _LATEX_SPECIALS:
            out.append(_LATEX_SPECIALS[ch])
        elif ch in _LATEX_UNICODE:
            out.append(_LATEX_UNICODE[ch])
        elif ord(ch) < 128:
            out.append(ch)
        else:
            _UNMAPPED[ch] = _UNMAPPED.get(ch, 0) + 1
            out.append(f'[U+{ord(ch):04X}]')
    return ''.join(out)


def markdown_escape(text: Any) -> str:
    """Escape one cell for a pipe table. Only the delimiter needs it."""
    return re.sub(r'\s+', ' ', str(text)).strip().replace('|', _BS + '|')


# ===========================================================================
# 3. The table model. One structure, two renderers, so the markdown mirror
#    cannot drift from the LaTeX it mirrors -- the failure a hand-kept second
#    copy guarantees.
# ===========================================================================

@dataclass(frozen=True)
class Col:
    """One column: where its value comes from, and how it is set."""

    key: str
    header: str
    align: str = 'l'          # 'l' | 'r' | 'c' | 'p'
    width: float = 0.0        # fraction of linewidth, only for align='p'

    def latex_spec(self) -> str:
        if self.align == 'p':
            return 'p{%.3f%slinewidth}' % (self.width, _BS)
        return self.align


@dataclass
class Table:
    """A built-once, written-twice table with a generated caption."""

    key: str
    title: str
    caption: str
    cols: tuple[Col, ...]
    rows: list[dict[str, Any]] = field(default_factory=list)
    notes: tuple[str, ...] = ()
    #: Row indices that a midrule precedes, for grouping by cell or by block.
    rules_before: frozenset[int] = frozenset()
    fontsize: str = 'footnotesize'

    @property
    def label(self) -> str:
        return f'tab:{self.key}'

    @property
    def star(self) -> bool:
        """Span both columns when the table is too wide for one.

        Emitted as `table*`, which is legal in a one-column class as well, so
        the same file compiles in IEEE (two-column) and LNCS (one-column).
        """
        return len(self.cols) >= 7


def render_latex(t: Table) -> str:
    """booktabs LaTeX for one table. No siunitx, no resizebox, no colour."""
    env = 'table*' if t.star else 'table'
    spec = ''.join(c.latex_spec() for c in t.cols)
    row_end = ' ' + _BS * 2
    lines: list[str] = [
        f'% {t.key}: generated by experiments/tables.py -- do not edit.',
        '% Needs only the booktabs package; compiles in a stock IEEE or LNCS '
        'template.',
        _BS + 'begin{' + env + '}[t]',
        _BS + 'centering',
        _BS + 'caption{' + latex_escape(t.caption) + '}',
        _BS + 'label{' + t.label + '}',
        _BS + t.fontsize,
        _BS + 'begin{tabular}{' + spec + '}',
        _BS + 'toprule',
        ' & '.join(_BS + 'textbf{' + latex_escape(c.header) + '}'
                   for c in t.cols) + row_end,
        _BS + 'midrule',
    ]
    for i, row in enumerate(t.rows):
        if i and i in t.rules_before:
            lines.append(_BS + 'midrule')
        lines.append(' & '.join(latex_escape(row.get(c.key, ''))
                                for c in t.cols) + row_end)
    lines.append(_BS + 'bottomrule')
    lines.append(_BS + 'end{tabular}')
    if t.notes:
        lines.append(_BS + 'vspace{2pt}')
        lines.append(_BS + 'begin{minipage}{' + _BS + 'linewidth}')
        lines.append(_BS + 'raggedright' + _BS + 'scriptsize')
        for j, note in enumerate(t.notes):
            tail = '' if j == len(t.notes) - 1 else row_end
            lines.append(latex_escape(note) + tail)
        lines.append(_BS + 'end{minipage}')
    lines.append(_BS + 'end{' + env + '}')
    return '\n'.join(lines) + '\n'


def render_markdown(t: Table) -> str:
    """The mirror: same rows, same caption, no LaTeX toolchain required."""
    align = {'l': ':---', 'r': '---:', 'c': ':---:', 'p': ':---'}
    lines: list[str] = [
        f'# {t.title}',
        '',
        '<!-- generated by experiments/tables.py -- do not edit -->',
        '',
        f'*{t.caption}*',
        '',
        '| ' + ' | '.join(markdown_escape(c.header) for c in t.cols) + ' |',
        '| ' + ' | '.join(align[c.align] for c in t.cols) + ' |',
    ]
    for i, row in enumerate(t.rows):
        if i and i in t.rules_before:
            lines.append('| ' + ' | '.join([''] * len(t.cols)) + ' |')
        lines.append('| ' + ' | '.join(markdown_escape(row.get(c.key, ''))
                                       for c in t.cols) + ' |')
    if t.notes:
        lines += ['', '**Notes.**', '']
        lines += [f'- {n}' for n in t.notes]
    return '\n'.join(lines) + '\n'


# ===========================================================================
# 4. Number formatting. Small, boring, and in one place, so two tables cannot
#    round the same quantity differently.
# ===========================================================================

def _isnum(v: Any) -> bool:
    try:
        return (v is not None and not isinstance(v, bool)
                and bool(np.isfinite(float(v))))
    except (TypeError, ValueError):
        return False


def fnum(v: Any, nd: int = 3, missing: str = NOT_RECORDED) -> str:
    """A number at fixed precision, or an explicit marker when there is none."""
    if not _isnum(v):
        return missing
    return f'{float(v):.{nd}f}'


def fint(v: Any, missing: str = NOT_RECORDED) -> str:
    if not _isnum(v):
        return missing
    return f'{int(round(float(v))):d}'


def fmt_mean_sd(values: Sequence[float], nd: int = 3) -> str:
    """mean +/- SD over the runs of one arm.

    No SD is printed at n=1: one observation has no dispersion, and printing
    0.000 there would invent a precision claim the data cannot support.
    """
    v = np.asarray([x for x in values if _isnum(x)], dtype=float)
    if v.size == 0:
        return NOT_RECORDED
    if v.size == 1:
        return f'{v[0]:.{nd}f}'
    return f'{v.mean():.{nd}f} ± {v.std(ddof=1):.{nd}f}'


def fmt_p(p: Any) -> str:
    """A p-value at the precision the exact tests actually attain.

    Five decimals, because the smallest two-sided p the sign-flip test can
    return at n=10 is 0.00195 and the strictest Holm step is 0.00625: rounding
    to three would collapse the distinction the plan's decision rule rests on
    (`ANALYSIS_PLAN.md` 2.2).
    """
    if not _isnum(p):
        return EM_DASH
    p = float(p)
    return f'{p:.5f}' if p >= 1e-4 else f'{p:.2e}'


def fmt_ci(est: Any, lo: Any, hi: Any, nd: int = 3) -> str:
    """point [lo, hi]. Never a bare point estimate presented as an estimate."""
    if not _isnum(est):
        return NOT_RECORDED
    if not (_isnum(lo) and _isnum(hi)):
        return f'{float(est):.{nd}f} [no interval]'
    return f'{float(est):.{nd}f} [{float(lo):.{nd}f}, {float(hi):.{nd}f}]'


def verdict_word(lo: Any, hi: Any) -> str:
    """Compact interval verdict; the long form is `stats.phrase_*`.

    `self_test` asserts this agrees in direction with
    `stats.phrase_interval_verdict` on every sign pattern, so the compact and
    the prose form cannot diverge.
    """
    if not (_isnum(lo) and _isnum(hi)):
        return NOT_RECORDED
    if float(lo) > 0:
        return 'excludes 0 (positive)'
    if float(hi) < 0:
        return 'excludes 0 (negative)'
    return 'not distinguishable'


def exclusion_bound_text(lo: Any, nd: int = 3) -> str:
    """The licensed positive statement when a null cannot be affirmed.

    `ANALYSIS_PLAN.md` 4: always reported, whatever the verdict, because a
    non-significant difference is never evidence of equivalence.
    """
    if not _isnum(lo):
        return NOT_RECORDED
    lo = float(lo)
    if lo >= 0:
        return f'no degradation (interval at or above {lo:+.{nd}f})'
    return f'worse than {abs(lo):.{nd}f} excluded'


# ===========================================================================
# 5. Inputs. `per_seed.csv` is the pinned interface (`aggregate.py`); the stats
#    JSON is optional, and its absence is stated inside the affected tables
#    rather than worked around.
# ===========================================================================

#: The columns this module reads. Named, so a schema change fails loudly here
#: rather than yielding a table full of NOT_RECORDED.
REQUIRED_COLUMNS: tuple[str, ...] = (
    'run_dir', 'label', 'arm', 'cell', 'condition', 'env', 'seed',
    'final_score', 'auc_score')


def load_per_seed(path: str) -> pd.DataFrame:
    """Read `per_seed.csv`, and refuse anything that is not it."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'{path} not found. Produce it with '
            f'`python experiments/aggregate.py --out-root runs`.')
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f'{path} is missing column(s) {missing}, which aggregate.py pins '
            f'as part of the per_seed schema. Refusing to guess at them.')
    if not len(df):
        raise ValueError(f'{path} has no rows; there is nothing to tabulate.')
    for col in ('cell', 'condition', 'label', 'arm', 'env', 'source_env'):
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str)
    return df


def load_stats(path: Optional[str]) -> Optional[dict]:
    """Read `stats.py --json` output, or None. Never a partial substitute."""
    if not path:
        return None
    with open(path, encoding='utf-8') as fh:
        report = json.load(fh)
    if 's5_confirmatory' not in report:
        raise ValueError(
            f'{path} does not look like stats.py output (no s5_confirmatory '
            f'section). Produce it with '
            f'`python experiments/stats.py --json {path}`.')
    return report


NO_STATS_NOTE = (
    'This table reports inference, and inference is computed by stats.py, '
    'not here: a second implementation of ANALYSIS_PLAN.md would be a second '
    'chance to diverge from it. Re-run with '
    '--stats <file> after `python experiments/stats.py --json <file>`. The '
    'rows below are placeholders, not results.')


# --- environment naming -----------------------------------------------------

def env_tag(canonical: str) -> str:
    """A compact tag for a canonical env string, e.g. LL[gravity=-4].

    Abbreviates the base name and keeps every parameter, because a variant
    whose parameters were dropped from a table would be indistinguishable from
    its base environment -- and the two have different score denominators
    (`DESIGN.md` 5.1).
    """
    if not canonical:
        return ''
    try:
        spec = envs.parse(canonical)
    except (ValueError, KeyError):
        return canonical
    base = _ENV_ABBREV.get(spec.env_id, spec.env_id)
    if not spec.params:
        return base
    # The parameter text comes from `canonical()` rather than from
    # re-formatting the dict, so the tag cannot disagree with the string the
    # manifest recorded.
    _, _, inner = spec.canonical().partition(':')
    return f'{base}[{inner}]'


def env_legend(canonicals: Iterable[str]) -> str:
    """The abbreviation legend, listing only the tags a table actually uses."""
    seen: dict[str, str] = {}
    for c in canonicals:
        if not c:
            continue
        try:
            base = envs.parse(c).env_id
        except (ValueError, KeyError):
            continue
        if base in _ENV_ABBREV:
            seen[_ENV_ABBREV[base]] = base
    if not seen:
        return ''
    return ('Environment abbreviations: '
            + '; '.join(f'{k} = {v}' for k, v in sorted(seen.items()))
            + '. Bracketed parameters are the variant overrides in force.')


def env_reference(canonical: str) -> Optional[dict]:
    """Measured reference points, or None -- never a default of zero.

    `envs.reference` raises rather than defaulting for the reason stated there:
    a missing reference silently replaced by zero puts one variant's scores on
    a different scale from every other, which is the class of error that made
    the published cross-variant comparisons meaningless.
    """
    try:
        return envs.reference(canonical)
    except (KeyError, ValueError):
        return None


# ===========================================================================
# 6. Generated captions. C8's guard applied to tables: the n, the endpoint
#    definition, the test and the normalisation are filled in from the data, so
#    a caption cannot contradict the table it sits under.
# ===========================================================================

@dataclass
class Context:
    """Everything a caption needs to state, computed once per invocation."""

    per_seed_path: str
    per_seed_sha: Optional[str]
    stats_path: Optional[str]
    stats_sha: Optional[str]
    plan_hashes: dict
    git: dict
    n_runs: int
    n_arms: int
    min_n: int
    max_n: int
    validation: bool
    plan_hash_in_data: tuple[str, ...]
    argv: tuple[str, ...]

    def n_phrase(self) -> str:
        if self.min_n == self.max_n:
            return (f'n = {self.min_n} seed(s) per arm, over {self.n_arms} '
                    f'arms and {self.n_runs} runs')
        return (f'n = {self.min_n}-{self.max_n} seeds per arm -- arms are not '
                f'equal-sized here, and no seed is dropped to make them so '
                f'(ANALYSIS_PLAN.md 8) -- over {self.n_arms} arms and '
                f'{self.n_runs} runs')

    def stamp(self) -> str:
        bits = [f'per_seed.csv {self.per_seed_sha or "unhashed"}']
        if self.stats_sha:
            bits.append(f'stats.json {self.stats_sha}')
        bits.append(f'ANALYSIS_PLAN.md {self.plan_hashes.get("ANALYSIS_PLAN.md")}')
        commit = (self.git.get('commit') or '')[:12] or 'unknown'
        dirty = ', working tree dirty' if self.git.get('dirty') else ''
        bits.append(f'commit {commit}{dirty}')
        return 'Provenance: ' + '; '.join(bits) + '.'


def build_context(df: pd.DataFrame, per_seed: str, stats_path: Optional[str],
                  argv: Sequence[str]) -> Context:
    groups = arm_groups(df)
    sizes = [len(g) for _, g in groups] or [0]
    return Context(
        per_seed_path=per_seed,
        per_seed_sha=provenance.file_hash(per_seed),
        stats_path=stats_path,
        stats_sha=provenance.file_hash(stats_path) if stats_path else None,
        plan_hashes=provenance.plan_hashes(),
        git=provenance.git_state(),
        n_runs=int(len(df)),
        n_arms=len(groups),
        min_n=int(min(sizes)),
        max_n=int(max(sizes)),
        validation=int(min(sizes)) < MIN_N_FOR_INFERENCE,
        plan_hash_in_data=tuple(sorted(
            set(df['plan_hash'].dropna().astype(str)))
            if 'plan_hash' in df.columns else ()),
        argv=tuple(argv),
    )


def build_caption(ctx: Context, what: str, *,
                  endpoints: Sequence[str] = (),
                  test: str = '',
                  normalised: bool = True,
                  extra: Sequence[str] = ()) -> str:
    """Assemble one caption from parts that are computed, never typed."""
    parts: list[str] = [what.rstrip('.') + '.', ctx.n_phrase() + '.']
    for ep in endpoints:
        definition = ENDPOINT_DEFS.get(ep)
        if definition:
            parts.append(definition)
    if test:
        parts.append(test.rstrip('.') + '.')
    if normalised:
        parts.append(NORMALISATION)
    parts.extend(e.rstrip('.') + '.' for e in extra)
    if ctx.validation:
        parts.insert(0, VALIDATION_STAMP + ':')
    parts.append(ctx.stamp())
    return ' '.join(parts)


# ===========================================================================
# 7. Arm grouping. The unit of a results table is the arm, and `aggregate.py`
#    pins `label` as the arm identity while keeping `cell` and `condition` as
#    separate columns -- so an arm is (env, cell, condition, label) and the 2x2
#    factor levels stay readable beside it.
# ===========================================================================

def _cell_rank(cell: str) -> int:
    return CELL_ORDER.index(cell) if cell in CELL_ORDER else len(CELL_ORDER)


def _condition_rank(cond: str) -> int:
    return (CONDITION_ORDER.index(cond) if cond in CONDITION_ORDER
            else len(CONDITION_ORDER))


def _canon(env: str) -> str:
    try:
        return envs.parse(env).canonical()
    except (ValueError, KeyError):
        return env


def _env_rank(canonical: str) -> tuple[int, str]:
    """Target env first, then the interface-change corner, then the rest.

    The order the design argues in (`DESIGN.md` 6.4), so a reader meets the
    primary target before the controls and the source tasks.
    """
    canon = _canon(canonical)
    if canon == _canon(registry.TARGET_ENV):
        return (0, canon)
    if canon == _canon(registry.INTERFACE_ENV):
        return (1, canon)
    if canon == _canon(registry.SOURCE_ENV):
        return (3, canon)
    return (2, canon)


ARM_KEYS: tuple[str, ...] = ('env', 'cell', 'condition', 'label')


def arm_groups(df: pd.DataFrame) -> list[tuple[tuple, pd.DataFrame]]:
    """Arms in reporting order, each with the runs that belong to it."""
    groups = [(tuple(k), g) for k, g in df.groupby(list(ARM_KEYS), dropna=False)]
    groups.sort(key=lambda kv: (_env_rank(kv[0][0]), _cell_rank(kv[0][1]),
                                _condition_rank(kv[0][2]), kv[0][3]))
    return groups


def scratch_headroom(df: pd.DataFrame, cell: str, env: str) -> Optional[float]:
    """1 - the cell's own scratch mean, in this same environment.

    The residual scale confound `DESIGN.md` 2.5 names: a cell whose scratch
    baseline sits near the ceiling has less room to gain and more to lose, so
    every between-cell reading needs this number beside it. The threshold is 1
    by construction of the normalisation, so headroom is 1 - the scratch mean.
    """
    rows = df[(df['cell'] == cell) & (df['env'] == env)
              & (df['condition'] == 'scratch')]
    vals = pd.to_numeric(rows['final_score'], errors='coerce').dropna()
    if not len(vals):
        return None
    return 1.0 - float(vals.mean())


def _series(g: pd.DataFrame, col: str) -> pd.Series:
    if col not in g.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(g[col], errors='coerce')


# ===========================================================================
# 8. Table 1 -- main_results. Every arm x the four reported metrics, the raw
#    return in parentheses for interpretability, and the two columns that stop
#    a between-cell reading being made without them: transferred-parameter
#    fraction (`DESIGN.md` 3.1) and headroom (2.5).
# ===========================================================================

def table_main_results(df: pd.DataFrame, ctx: Context) -> Table:
    cols = (
        Col('arm', 'Arm (label)'),
        Col('cell', 'Cell'),
        Col('condition', 'Condition'),
        Col('env', 'Env'),
        Col('n', 'n', 'r'),
        Col('final_score', 'final_score (P1)', 'r'),
        Col('final_return', 'raw return', 'r'),
        Col('auc_score', 'auc_score (P2)', 'r'),
        Col('episode_length', 'episode length', 'r'),
        Col('within_run_sd', 'within-run SD', 'r'),
        Col('frac', 'transf. frac.', 'r'),
        Col('headroom', 'headroom', 'r'),
    )
    rows: list[dict[str, Any]] = []
    rules: set[int] = set()
    prev_block: Optional[tuple] = None
    fraction_varies: list[str] = []
    for (env, cell, cond, label), g in arm_groups(df):
        block = (_env_rank(env)[0], cell)
        if prev_block is not None and block != prev_block:
            rules.add(len(rows))
        prev_block = block
        fr = _series(g, 'transferred_param_fraction').dropna()
        if len(fr) > 1 and float(fr.max() - fr.min()) > 1e-9:
            fraction_varies.append(label)
        rows.append({
            'arm': label or '(unlabelled)',
            'cell': cell or NOT_RECORDED,
            'condition': cond or NOT_RECORDED,
            'env': env_tag(env),
            'n': str(len(g)),
            'final_score': fmt_mean_sd(_series(g, 'final_score')),
            'final_return': fmt_mean_sd(_series(g, 'final_return'), nd=1),
            'auc_score': fmt_mean_sd(_series(g, 'auc_score')),
            'episode_length': fmt_mean_sd(
                _series(g, 'episode_length_final100'), nd=1),
            'within_run_sd': fmt_mean_sd(_series(g, 'within_run_sd')),
            'frac': (fnum(fr.mean(), 3) if len(fr)
                     else ('n/a' if cond == 'scratch' else NOT_RECORDED)),
            'headroom': fnum(scratch_headroom(df, cell, env), 3),
        })

    notes = [
        'Cells are mean ± SD across the seeds of that arm. An arm with n=1 '
        'shows no SD, because one observation has no dispersion and 0.000 '
        'would be a precision claim the data cannot support.',
        "raw return is the same runs' mean final_return, on the environment's "
        'own return scale, and appears only so a normalised score can be read '
        'back into return points. Raw returns are NOT comparable across '
        'environments or variants (DESIGN.md 5.1) and no comparison in this '
        'paper is made on them.',
        'transf. frac. is transferred_param_fraction, the fraction of the '
        "target model's parameters the copy actually wrote. It is printed on "
        'every row because holding one layer list fixed across architectures '
        'transferred 97% of the mlp and 51% of the dueling net in revision 1 '
        'of the design, confounding arch with treatment intensity '
        '(DESIGN.md 3.1). "n/a" marks an arm that transfers nothing; '
        f'"{NOT_RECORDED}" marks an arm whose manifest did not record it.',
        "headroom is 1 - the same cell's own scratch mean in the same "
        'environment, i.e. the distance left to the registered threshold. No '
        'between-cell reading of any effect is interpretable without it: a '
        'cell near the ceiling has less to gain and more to lose '
        '(DESIGN.md 2.5).',
        'This table carries no test and no interval. The confirmatory family '
        f'is the {FAMILY_SIZE} tests of Table 2 and nothing else '
        '(ANALYSIS_PLAN.md 2).',
    ]
    if fraction_varies:
        notes.append(
            'transferred_param_fraction is not constant within arm(s) '
            + ', '.join(sorted(set(fraction_varies)))
            + '. The mean is shown and the variation is reported here rather '
              'than averaged away, because DESIGN.md 8.4 treats a moving '
              'invariant as a reason to refuse aggregation, not to smooth it.')
    legend = env_legend(df['env'])
    if legend:
        notes.append(legend)
    notes.append(SCOPE_CLAUSE)

    caption = build_caption(
        ctx,
        'Descriptive results for every arm: the two co-primary endpoints and '
        'the two secondary endpoints the published manuscript promised and '
        'never reported',
        endpoints=('final_score', 'auc_score', 'episode_length_final100',
                   'within_run_sd'),
        test='No test and no p-value appears in this table; it is descriptive '
             'by role (ANALYSIS_PLAN.md 1)',
        extra=('Headroom is 1 - the cell\'s scratch mean on the normalised '
               'scale, where the registered threshold is 1',))
    return Table('main_results', 'Main results: all arms x all metrics',
                 caption, cols, rows, tuple(notes), frozenset(rules))


# ===========================================================================
# 9. Table 2 -- inferential. Every test and every estimate, with the RQ it
#    addresses and the family it belongs to. The p columns of an
#    estimation-only row carry an em dash, never a blank: `ANALYSIS_PLAN.md` 7
#    licenses p-values inside exactly one family of eight, so the absence is a
#    design statement and has to read as one.
# ===========================================================================

def _inf_row(rq: str, family: str, quantity: str, endpoint: str, n: Any,
             test: str, statistic: str, p_raw: str, p_holm: str,
             effect: str, verdict: str) -> dict[str, Any]:
    return {'rq': rq, 'family': family, 'quantity': quantity,
            'endpoint': endpoint, 'n': fint(n, missing=NOT_RECORDED),
            'test': test, 'statistic': statistic, 'p_raw': p_raw,
            'p_holm': p_holm, 'effect': effect, 'verdict': verdict}


def _confirmatory_rows(stats: dict) -> tuple[list[dict], list[str]]:
    """The 8 members plus their two pre-specified companions, per cell."""
    rows: list[dict] = []
    footnotes: list[str] = []
    for rec in stats.get('s5_confirmatory', {}).get('members', []):
        cell, metric = rec.get('cell', ''), rec.get('metric', '')
        quantity = f'delta = transfer - scratch, {cell}'
        if 'suppressed' in rec:
            reason = str(rec['suppressed'])
            rows.append(_inf_row(
                'RQ2', 'Confirmatory', quantity, metric, rec.get('n'),
                'exact sign-flip (paired)', 'refused',
                'refused', 'refused',
                fmt_ci(rec.get('mean_delta'), None, None) + ' (mean only)',
                'suppressed: ' + reason))
            footnotes.append(f'{metric}/{cell} suppressed -- {reason}.')
            continue
        rows.append(_inf_row(
            'RQ2', 'Confirmatory', quantity, metric, rec.get('n'),
            f'exact sign-flip, {rec.get("signflip_mode", "")}'.rstrip(', '),
            'mean delta = ' + fnum(rec.get('mean_delta')),
            fmt_p(rec.get('p_signflip')), fmt_p(rec.get('p_holm')),
            'HL ' + fmt_ci(rec.get('hl'), rec.get('ci_lo'), rec.get('ci_hi')),
            ('Holm-significant' if rec.get('significant_holm')
             else 'not distinguishable at the Holm step')))
        rows.append(_inf_row(
            'RQ2', 'Companion', quantity, metric, rec.get('n'),
            'Wilcoxon signed-rank (paired)',
            'W = ' + fnum(rec.get('wilcoxon_W'), 1),
            fmt_p(rec.get('p_wilcoxon')), EM_DASH,
            'rho(scratch, transfer) = ' + fnum(rec.get('rho_pearson')),
            'reported for agreement, not corrected'))
        rows.append(_inf_row(
            'RQ2', 'Companion', quantity, metric, rec.get('n'),
            'Mann-Whitney U (unpaired)',
            'U = ' + fnum(rec.get('mannwhitney_U'), 1),
            fmt_p(rec.get('p_mannwhitney')), EM_DASH,
            'Spearman rho = ' + fnum(rec.get('rho_spearman')),
            'reported for comparability with the published test'))
    return rows, footnotes


def _estimation_rows(stats: dict) -> list[dict]:
    """Everything outside the confirmatory family: interval, no p-value."""
    rows: list[dict] = []
    est = stats.get('s9_estimation', {})
    primary = (stats.get('invocation', {}).get('metrics') or ['final_score'])[0]

    # RQ1 -- between-cell scratch comparison. Brunner-Munzel rather than
    # Hodges-Lehmann because the cells' across-seed SDs differ by up to 8x on
    # the normalised scale, which violates the location-shift assumption HL
    # needs (ANALYSIS_PLAN.md 3).
    for r in est.get('rq1', {}).get('rows', []):
        rows.append(_inf_row(
            'RQ1', 'Estimation-only',
            f'scratch {r.get("a")} vs scratch {r.get("b")}', primary,
            min(r.get('n_a') or 0, r.get('n_b') or 0),
            'Brunner-Munzel relative effect', 'theta = P(X>Y)', EM_DASH,
            EM_DASH,
            'theta ' + fmt_ci(r.get('theta'), r.get('ci_lo'), r.get('ci_hi')),
            ('theta excludes 0.5'
             if (_isnum(r.get('ci_lo')) and _isnum(r.get('ci_hi'))
                 and (float(r['ci_lo']) > 0.5 or float(r['ci_hi']) < 0.5))
             else 'covers 0.5: not distinguishable')))

    # RQ3 -- effect modification, not causation. MDE ~1.96 sigma_delta, larger
    # than the plausible effect, so it is estimation-only by design (2.4).
    for r in est.get('rq3', {}).get('pairs', []):
        rows.append(_inf_row(
            'RQ3', 'Estimation-only',
            f'delta({r.get("a")}) - delta({r.get("b")})', primary, r.get('n'),
            'difference of paired deltas, joint seed bootstrap',
            'scales ' + str(r.get('scales', NOT_RECORDED)), EM_DASH, EM_DASH,
            'HL ' + fmt_ci(r.get('hl'), r.get('ci_lo'), r.get('ci_hi')),
            verdict_word(r.get('ci_lo'), r.get('ci_hi'))))
    inter = est.get('rq3', {}).get('interaction', {})
    if inter.get('available'):
        rows.append(_inf_row(
            'RQ3', 'Estimation-only', '2x2 interaction on delta', primary,
            inter.get('n'), 'interaction contrast, joint seed bootstrap',
            ('intensity-confounded' if inter.get('intensity_confounded')
             else 'intensity-matched'), EM_DASH, EM_DASH,
            'HL ' + fmt_ci(inter.get('hl'), inter.get('ci_lo'),
                           inter.get('ci_hi')),
            verdict_word(inter.get('ci_lo'), inter.get('ci_hi'))))

    # RQ5 -- ordered alternative on the shift families. Wind is primary;
    # gravity carries the difficulty caveat of DESIGN.md 5.1.
    for family, res in (est.get('rq5') or {}).items():
        if not res.get('available'):
            rows.append(_inf_row(
                'RQ5', 'Estimation-only', f'shift gradient, {family}', primary,
                res.get('n'), 'Jonckheere-Terpstra ordered alternative',
                'not available in this run selection', EM_DASH, EM_DASH,
                NOT_RECORDED, 'no levels present'))
            continue
        rows.append(_inf_row(
            'RQ5', 'Estimation-only', f'shift gradient, {family}', primary,
            res.get('n'), 'Jonckheere-Terpstra ordered alternative',
            'JT z = ' + fnum(res.get('z'), 2), EM_DASH, EM_DASH,
            'standardised ' + fmt_ci(res.get('effect'), res.get('ci_lo'),
                                     res.get('ci_hi')),
            verdict_word(res.get('ci_lo'), res.get('ci_hi'))))

    # RQ6 -- budget. Valid only because the exploration schedule never reads
    # the budget, which validate.py asserts (ANALYSIS_PLAN.md 3).
    for r in est.get('rq6', {}).get('rows', []):
        rows.append(_inf_row(
            'RQ6', 'Estimation-only',
            f'delta at the {r.get("prefix")}-episode prefix, {r.get("cell")}',
            primary, r.get('n'), 'paired prefix vs budget, same runs',
            'delta at budget = ' + fnum(r.get('delta_at_budget')), EM_DASH,
            EM_DASH, 'delta at prefix = ' + fnum(r.get('delta_at_prefix')),
            'single held-out checkpoint vs single final checkpoint'))

    # Dispersion. No p-value is interpreted: at n=10 with an SD ratio of ~3 the
    # test has almost no power, which is the honest reading of the published
    # Brown-Forsythe null (ANALYSIS_PLAN.md 3).
    for r in est.get('dispersion', {}).get('rows', []):
        rows.append(_inf_row(
            'RQ2', 'Estimation-only',
            f'across-seed SD ratio, transfer/scratch, {r.get("cell")}',
            primary, r.get('n'), 'bootstrap CI on the SD ratio',
            'Brown-Forsythe W = ' + fnum(r.get('brown_forsythe_W'), 3),
            EM_DASH, EM_DASH,
            'ratio ' + fmt_ci(r.get('sd_ratio'), r.get('ci_lo'),
                              r.get('ci_hi')),
            ('excludes 1' if (_isnum(r.get('ci_lo')) and _isnum(r.get('ci_hi'))
                              and (float(r['ci_lo']) > 1
                                   or float(r['ci_hi']) < 1))
             else 'covers 1: not distinguishable')))

    # Secondary and mechanism metrics, estimation-only by declared role. The
    # role travels with the row: a mechanism signal supports or refuses
    # mechanism wording (RQ4) and is not an endpoint, which is the distinction
    # the published paper collapsed when it tested a metric it had declared
    # descriptive (DESIGN.md 5.4, 5.5).
    for r in (est.get('secondary', {}) or {}).get('rows', []):
        role = str(r.get('role', 'secondary'))
        rq = 'RQ4' if role == 'mechanism' else 'RQ2'
        rows.append(_inf_row(
            rq, f'Estimation-only ({role})',
            f'{r.get("metric")} delta, {r.get("cell")}',
            str(r.get('metric')), r.get('n'),
            'Hodges-Lehmann paired shift, seed bootstrap',
            'transfer mean = ' + fnum(r.get('transfer_mean')), EM_DASH,
            EM_DASH,
            'HL ' + fmt_ci(r.get('hl_delta'), r.get('ci_lo'), r.get('ci_hi')),
            verdict_word(r.get('ci_lo'), r.get('ci_hi'))))

    # Source competence as a continuous covariate (DESIGN.md 4.3): a
    # descriptive relationship, explicitly not a mediation claim.
    for r in (est.get('secondary', {}) or {}).get('source_competence', []):
        rows.append(_inf_row(
            'RQ2', 'Estimation-only',
            f'delta on source competence, {r.get("cell")}', primary,
            r.get('n'), 'OLS slope, seed bootstrap', 'slope per score unit',
            EM_DASH, EM_DASH,
            fmt_ci(r.get('slope'), r.get('ci_lo'), r.get('ci_hi'), nd=2),
            'descriptive relationship, not a mediation claim'))

    # Screens: BH q for orientation only, never as an assertion (7).
    for r in (est.get('screens', {}) or {}).get('rows', []):
        rows.append(_inf_row(
            r.get('rq', 'screen'), 'Screen (BH q, orientation only)',
            str(r.get('level', r.get('quantity', ''))), primary, r.get('n'),
            str(r.get('test', 'per-level estimate')),
            fnum(r.get('statistic')), fmt_p(r.get('p')),
            EM_DASH + ' (q = ' + fnum(r.get('q'), 4) + ')',
            fmt_ci(r.get('estimate'), r.get('ci_lo'), r.get('ci_hi')),
            'orientation only; selects at most one follow-up'))
    return rows


def _equivalence_rows(stats: dict) -> list[dict]:
    """Not TOST. A containment check on the interval already reported, plus the
    exclusion bound, which is always available and always printed (4)."""
    rows: list[dict] = []
    sec = stats.get('s6_equivalence', {})
    margin = sec.get('margin', EQUIVALENCE_MARGIN)
    for r in sec.get('rows', []):
        verdict = str(r.get('verdict', ''))
        note = verdict
        if verdict == 'suppressed':
            note = 'suppressed: ' + str(r.get('reason', ''))
        lo, hi = r.get('ci_lo'), r.get('ci_hi')
        interval = (f'CI [{float(lo):.3f}, {float(hi):.3f}]'
                    if (_isnum(lo) and _isnum(hi))
                    else 'no interval emitted')
        bound = (exclusion_bound_text(lo) if _isnum(lo)
                 else 'no exclusion bound: no interval was emitted')
        rows.append(_inf_row(
            'RQ2', 'Estimation-only',
            f'equivalence within ±{margin} score units, {r.get("cell")}',
            str(r.get('metric')), r.get('n'),
            f'containment of the 95% CI in ±{margin} (NOT TOST)',
            'SD(scratch) = ' + fnum(r.get('sd_scratch')), EM_DASH, EM_DASH,
            interval, note + '; ' + bound))
    return rows


def _c4_rows(stats: dict) -> list[dict]:
    """The positive control against its pre-registered pass criterion (4.2)."""
    rows: list[dict] = []
    sec = stats.get('s8_c4', {})
    if not sec.get('available'):
        return rows
    for r in sec.get('rows', []):
        rows.append(_inf_row(
            'RQ4', 'Estimation-only (pre-registered criterion)',
            f'C4 interface-only delta, {r.get("cell")}', 'final_score',
            r.get('n'), f'HL CI lower bound vs {C4_LOWER_BOUND}',
            str(r.get('verdict', '')), EM_DASH, EM_DASH,
            'HL ' + fmt_ci(r.get('hl'), r.get('ci_lo'), r.get('ci_hi')),
            str(r.get('reason', r.get('verdict', '')))))
    return rows


def table_inferential(df: pd.DataFrame, ctx: Context,
                      stats: Optional[dict]) -> Table:
    cols = (
        Col('rq', 'RQ'),
        Col('family', 'Family'),
        Col('quantity', 'Quantity / contrast'),
        Col('endpoint', 'Endpoint'),
        Col('n', 'n', 'r'),
        Col('test', 'Test / estimator'),
        Col('statistic', 'Statistic'),
        Col('p_raw', 'p (raw)', 'r'),
        Col('p_holm', 'p (Holm)', 'r'),
        Col('effect', 'Effect size [95% CI]'),
        Col('verdict', 'Reading'),
    )
    notes: list[str] = []
    rows: list[dict[str, Any]] = []
    rules: set[int] = set()
    if stats is None:
        notes.append(NO_STATS_NOTE)
        rows.append(_inf_row(
            'RQ2', 'Confirmatory',
            f'the {FAMILY_SIZE} within-cell deltas (4 cells x 2 co-primary '
            f'endpoints)', 'final_score, auc_score', None,
            'exact sign-flip randomisation test on the per-seed paired deltas',
            'not computed here', 'requires --stats', 'requires --stats',
            'requires --stats', 'no result without stats.py'))
    else:
        conf, footnotes = _confirmatory_rows(stats)
        rows.extend(conf)
        rules.add(len(rows))
        rows.extend(_equivalence_rows(stats))
        rules.add(len(rows))
        rows.extend(_c4_rows(stats))
        rules.add(len(rows))
        rows.extend(_estimation_rows(stats))
        notes.extend(footnotes)

    n_no_p = sum(1 for r in rows if r['p_raw'] == EM_DASH)
    notes = [
        f'An em dash ({EM_DASH}) in a p column is not a missing number. It '
        'means no p-value is defined for that row: ANALYSIS_PLAN.md 7 emits '
        f'p-values inside exactly one family of {FAMILY_SIZE}, and everything '
        f'else is estimation-only. {n_no_p} of {len(rows)} rows below carry no '
        'p-value by design.',
        'Family "Confirmatory": the only confirmatory family in the study -- '
        'the 4 within-cell deltas (transfer - scratch) on each of the 2 '
        f'co-primary endpoints, {FAMILY_SIZE} tests, Holm-Bonferroni over '
        f'{FAMILY_SIZE}, step-down from alpha = {ALPHA_STRICTEST:.5f} '
        '(ANALYSIS_PLAN.md 2).',
        'Family "Companion": Wilcoxon signed-rank and Mann-Whitney U on the '
        'same contrast, pre-specified in ANALYSIS_PLAN.md 2 and reported for '
        'agreement and for comparability with the published paper\'s test. '
        f'They are not members of the Holm family (which has exactly '
        f'{FAMILY_SIZE} members, fixed before launch), so their adjusted-p '
        'cell carries an em dash. The paired sign-flip test is primary by '
        'pre-registration, not by comparison of p-values.',
        'Family "Estimation-only": point estimate and 95% seed-level bootstrap '
        'interval, no p-value. Where a directional statement is wanted the '
        'licensed form is what the interval excludes, not a null '
        '(DESIGN.md 9).',
        'Family "Estimation-only (mechanism)": an instrumented signal from '
        'DESIGN.md 5.5, reported so that a mechanism sentence can cite one. It '
        'is not an endpoint and may not be quoted as a result; the report '
        'template has no free-text mechanism slot (DESIGN.md 9).',
        'Family "Screen": Benjamini-Hochberg q for orientation only. A screen '
        'result is never asserted as a finding; it selects at most one '
        'follow-up, which is then run on REPLICATE seeds and reported as a '
        'fresh estimate (ANALYSIS_PLAN.md 3).',
        '"suppressed" is not a null result. It records that stats.py refused '
        'the member -- an incomplete arm, an ambiguous primary arm, an '
        'invariant that moved across the arm, or n<3 -- and the reason is '
        'given in the Reading column and in the notes below.',
        'HL is the Hodges-Lehmann estimate of the paired shift with a '
        'bias-corrected seed-level bootstrap interval. No t-test, no Cohen\'s '
        'd and no normality-assuming interval appears anywhere in this study '
        '(ANALYSIS_PLAN.md 8).',
        'Censored steps-to-threshold is not tabulated here: it is reported as '
        'P(reached by budget) with a Clopper-Pearson interval and a '
        'Kaplan-Meier curve per arm in stats.py section 9, over 3 pre-declared '
        'levels x every arm, which is too many rows for this table. The '
        'omission is declared rather than silent.',
    ] + notes
    notes.append(SCOPE_CLAUSE)

    caption = build_caption(
        ctx,
        'Every test and every estimate in the study, with the research '
        'question it addresses and the family it belongs to',
        endpoints=('final_score', 'auc_score'),
        test=f'Primary test: exact sign-flip randomisation test on the '
             f'per-seed paired deltas, statistic = the mean delta, all 2^n '
             f'sign assignments enumerated for n<=20. Family-wise alpha = '
             f'{ALPHA} controlled by Holm-Bonferroni over {FAMILY_SIZE} tests; '
             f'strictest step alpha = {ALPHA_STRICTEST:.5f}. Wilcoxon '
             f'signed-rank and Mann-Whitney U are reported alongside, '
             f'pre-specified',
        extra=('Equivalence is assessed by containment of the 95% bootstrap CI '
               f'in ±{EQUIVALENCE_MARGIN} normalised-score units, not by TOST',))
    return Table('inferential', 'Inferential results and estimation summary',
                 caption, cols, rows, tuple(notes), frozenset(rules))


# ===========================================================================
# 10. Table 3 -- control_contrasts. The three contrasts of `DESIGN.md` 4 plus
#     the spectrum-matched C3b, per cell, from one joint bootstrap, each with
#     the exclusion restriction its mechanistic reading requires. The identity
#     (C2-C0) + (C3-C2) + (C1-C3) = C1-C0 is arithmetic and is stated as such:
#     revision 1 called it "an additive decomposition, each term estimable",
#     which implied an empirical finding where there is none (4.1).
# ===========================================================================

def table_control_contrasts(df: pd.DataFrame, ctx: Context,
                            stats: Optional[dict]) -> Table:
    cols = (
        Col('cell', 'Cell'),
        Col('endpoint', 'Endpoint'),
        Col('contrast', 'Contrast'),
        Col('name', 'What was manipulated'),
        Col('n', 'n', 'r'),
        Col('mean', 'mean', 'r'),
        Col('hl', 'HL [95% CI]', 'r'),
        Col('agree', 'Seeds'),
        Col('reading', 'Reading'),
        Col('restriction', 'Mechanistic reading requires assuming', 'p', 0.20),
    )
    rows: list[dict[str, Any]] = []
    rules: set[int] = set()
    notes: list[str] = []
    if stats is None:
        notes.append(NO_STATS_NOTE)
    else:
        for metric, sec in (stats.get('s7_controls') or {}).items():
            for cell, res in (sec.get('cells') or {}).items():
                if rows:
                    rules.add(len(rows))
                missing = res.get('missing') or []
                if res.get('suppressed') or not res.get('contrasts'):
                    rows.append({
                        'cell': cell, 'endpoint': metric,
                        'contrast': 'all', 'name': 'not estimated',
                        'n': fint(res.get('n')), 'mean': NOT_RECORDED,
                        'hl': NOT_RECORDED, 'agree': NOT_RECORDED,
                        'reading': (f'suppressed: n={res.get("n")} '
                                    f'< {MIN_N_FOR_INFERENCE}, or a required '
                                    f'condition is absent'),
                        'restriction': ('conditions absent: '
                                        + (', '.join(missing) or 'none')),
                    })
                    continue
                for c in res['contrasts']:
                    key = str(c.get('contrast'))
                    rows.append({
                        'cell': cell, 'endpoint': metric, 'contrast': key,
                        'name': str(c.get('name', '')),
                        'n': fint(c.get('n')),
                        'mean': fnum(c.get('mean')),
                        'hl': fmt_ci(c.get('hl'), c.get('ci_lo'),
                                     c.get('ci_hi')),
                        'agree': str(c.get('unanimous', '')),
                        'reading': verdict_word(c.get('ci_lo'),
                                                c.get('ci_hi')) + '; '
                                   + exclusion_bound_text(c.get('ci_lo')),
                        'restriction': EXCLUSION_RESTRICTIONS.get(
                            key, 'no exclusion restriction is declared for '
                                 'this contrast'),
                    })
                if missing:
                    notes.append(
                        f'{metric}/{cell}: condition(s) {", ".join(missing)} '
                        f'are absent, so every contrast that needs them is not '
                        f'computed and not approximated.')
                corr = res.get('correlations') or []
                if corr:
                    worst = max(corr, key=lambda r: abs(r.get('rho_seeds') or 0))
                    notes.append(
                        f'{metric}/{cell}: the per-seed contrast values are '
                        f'correlated (largest |rho| is '
                        f'{worst.get("a")} vs {worst.get("b")}, '
                        f'rho = {fnum(worst.get("rho_seeds"), 2)}), which is '
                        f'why the intervals come from one joint resampling '
                        f'rather than from separate two-sample tests on shared '
                        f'groups.')

    notes = [
        'Conditions (DESIGN.md 4): C0 scratch; C1 transfer from a trained '
        'same-cell source at the same seed; C2 transfer_untrained, a randomly '
        'initialised source of the same shape, which runs the same partial '
        'copy, the same reinitialised head and the same freeze window with no '
        'learned content; C2K0 the same at freeze_updates=0; C3 '
        'transfer_permuted with each transferred kernel shuffled entry-wise; '
        'C3b permuted but matched to the source layer\'s singular-value '
        'spectrum.',
        '(C2-C0) + (C3-C2) + (C1-C3) = C1-C0 is a telescoping arithmetic '
        'identity. It holds for any four numbers, nothing about it is '
        'testable, and it is shown only to fix notation. It is NOT an additive '
        'decomposition of a causal effect (DESIGN.md 4.1).',
        'Contrasts are named after what was manipulated, never after a '
        'mechanism. The last column states, per contrast, exactly what a '
        'mechanistic reading of it would have to assume -- so the assumption '
        'is visible rather than implied by the name.',
        'All intervals in this table come from ONE bootstrap of the per-seed '
        'vector (C0, C1, C2, C2K0, C3, C3b) over the listwise-complete seeds, '
        'so the contrasts and their correlations are estimated jointly. Four '
        'independent two-sample tests on shared groups would ignore that '
        'structure (ANALYSIS_PLAN.md 3).',
        'Estimation-only: no p-value appears in this table at all. C1-C0 is '
        'the one contrast here that is also the confirmatory estimand of '
        'Table 2; its p-value is reported there and only there.',
        'C3b exists because C3 does not preserve the singular-value spectrum, '
        'so C3-C2 also absorbs spectral effects. If C3 and C3b agree, the '
        'spectral caveat is empirically void; if they disagree, the '
        'disagreement is the finding (DESIGN.md 4.1).',
    ] + notes
    notes.append(SCOPE_CLAUSE)

    caption = build_caption(
        ctx,
        'The three control contrasts of DESIGN.md 4 plus the spectrum-matched '
        'control C3b, per cell, with the exclusion restriction each '
        'mechanistic reading requires',
        endpoints=tuple(sorted((stats.get('s7_controls') or {}).keys()))
        if stats else ('final_score',),
        test='Estimation-only: Hodges-Lehmann shift with a bias-corrected '
             '95% seed-level bootstrap interval from one joint resampling of '
             'the per-seed condition vector. No p-value is emitted for any row '
             '(ANALYSIS_PLAN.md 3, 7)')
    return Table('control_contrasts',
                 'Control contrasts: mechanics, weight statistics, structure',
                 caption, cols, rows, tuple(notes), frozenset(rules))


# ===========================================================================
# 11. Table 4 -- protocol_summary. The methods table whose absence made the
#     published protocol unreproducible: the manuscript described a freeze
#     schedule that was never implemented and never listed the layers it
#     copied. Everything here is read from the manifests the runs wrote, so the
#     table describes what the code did rather than what the text claimed.
# ===========================================================================

#: `per_seed.csv` records `reinitialised_layer_count` but not the layer *names*
#: or the copied/partial counts; those live in `manifest.json` under
#: `transfer.summary`. Rather than approximate them, this module opens one
#: manifest per arm. When it cannot, the cell says so and the missing per_seed
#: columns are named in the notes -- the same discipline stats.py applies to the
#: absent `convergence_slope_se`.
MISSING_PER_SEED_COLUMNS: tuple[str, ...] = (
    'layers_copied', 'layers_partial', 'layers_skipped')


def _manifest_for(run_dir: str) -> Optional[dict]:
    """Read one run's manifest, tolerating the path separators Windows wrote."""
    for candidate in (run_dir, os.path.join(_REPO, run_dir)):
        path = os.path.join(str(candidate).replace('\\', os.sep),
                            'manifest.json')
        if os.path.exists(path):
            try:
                with open(path, encoding='utf-8') as fh:
                    return json.load(fh)
            except (OSError, json.JSONDecodeError):
                return None
    return None


def _transfer_summary(g: pd.DataFrame) -> tuple[str, str, Optional[dict]]:
    """(layers c/p/r counts, layer names, raw summary) for one arm."""
    for run_dir in g['run_dir'].astype(str):
        man = _manifest_for(run_dir)
        if not man:
            continue
        summary = ((man.get('transfer') or {}).get('summary') or {})
        if not summary:
            continue
        cop = summary.get('layers_copied') or []
        par = summary.get('layers_partial') or []
        rei = summary.get('layers_reinit') or []
        skp = summary.get('layers_skipped') or []
        counts = f'{len(cop)}/{len(par)}/{len(rei)}'
        if skp:
            counts += f' (+{len(skp)} skipped)'
        names = ('copied: ' + (', '.join(cop) or 'none')
                 + '; partial: ' + (', '.join(par) or 'none')
                 + '; reinit: ' + (', '.join(rei) or 'none')
                 + ('; skipped: ' + ', '.join(skp) if skp else ''))
        return counts, names, summary
    return NOT_RECORDED, NOT_RECORDED, None


#: Columns shown only when they vary across the tabulated runs. A constant
#: column is moved to the notes with its value, which keeps the methods table
#: narrow enough to read without hiding a factor that actually moved.
OPTIONAL_PROTOCOL_COLUMNS: tuple[tuple[str, str], ...] = (
    ('aggregation', 'aggregation'),
    ('permute_kind', 'permute kind'),
    ('value_recal', 'value recal.'),
    ('lr', 'lr'),
    ('target_update', 'target update'),
    ('hidden', 'hidden'),
    ('head_units', 'head units'),
)


def _one(g: pd.DataFrame, col: str, integer: bool = False) -> str:
    """The arm's value for a column, or every value it took, comma-joined.

    Never a mean: these are protocol fields, and an arm whose protocol field
    moved is an arm `audit.py` refuses to aggregate (`DESIGN.md` 8.4). Showing
    both values is how that becomes visible instead of being averaged away.
    """
    if col not in g.columns:
        return NOT_RECORDED
    raw = g[col].dropna().tolist()
    if not raw:
        return NOT_RECORDED
    if integer:
        try:
            return ', '.join(str(v) for v in sorted({int(round(float(x)))
                                                     for x in raw}))
        except (TypeError, ValueError):
            pass
    return ', '.join(sorted({str(v) for v in raw}))


#: Fields that describe a transfer. A scratch arm has none of them, and the
#: Config default it happens to carry must not be printed as though something
#: had been transferred or frozen -- that is exactly the reading the published
#: manuscript's missing methods table invited.
TRANSFER_ONLY_FIELDS: frozenset[str] = frozenset({
    'transfer_set', 'input_policy', 'head_policy', 'freeze_group',
    'freeze_updates', 'permute_kind', 'value_recal', 'source_env',
    'params_copied', 'transferred_param_fraction'})

#: `permute_kind` describes how a transferred kernel was shuffled, so it means
#: nothing outside C3/C3b. The Config default it carries elsewhere is not
#: printed as though a permutation had happened.
PERMUTE_ONLY_FIELDS: frozenset[str] = frozenset({'permute_kind'})


def table_protocol_summary(df: pd.DataFrame, ctx: Context) -> Table:
    varying = [(c, h) for c, h in OPTIONAL_PROTOCOL_COLUMNS
               if c in df.columns and df[c].dropna().nunique() > 1]
    constant = [(c, str(df[c].dropna().unique()[0]))
                for c, _h in OPTIONAL_PROTOCOL_COLUMNS
                if c in df.columns and df[c].dropna().nunique() == 1]
    cols = (
        Col('arm', 'Arm (label)'),
        Col('cell', 'Cell'),
        Col('condition', 'Condition'),
        Col('pair', 'Source -> target'),
        Col('transfer_set', 'Transfer set'),
        Col('input_policy', 'Input policy'),
        Col('head_policy', 'Head policy'),
        Col('layers', 'Layers c/p/r', 'r'),
        Col('params_copied', 'Params copied', 'r'),
        Col('frac', 'Fraction', 'r'),
        Col('freeze_group', 'Freeze group'),
        Col('freeze_updates', 'Freeze updates', 'r'),
    ) + tuple(Col(c, h) for c, h in varying)

    rows: list[dict[str, Any]] = []
    rules: set[int] = set()
    prev_cell: Optional[str] = None
    layer_names: list[str] = []
    moved: list[str] = []
    manifests_read = 0
    n_transferring = 0
    for (env, cell, cond, label), g in arm_groups(df):
        if prev_cell is not None and cell != prev_cell:
            rules.add(len(rows))
        prev_cell = cell
        counts, names, summary = _transfer_summary(g)
        if cond != 'scratch':
            n_transferring += 1
            if summary is not None:
                manifests_read += 1
        src = _one(g, 'source_env')
        pair = (f'{env_tag(src)} -> {env_tag(env)}'
                if (src not in ('', NOT_RECORDED) and cond != 'scratch')
                else f'n/a -> {env_tag(env)}')
        transfers = cond != 'scratch'

        def field(col: str, integer: bool = False) -> str:
            if not transfers and col in TRANSFER_ONLY_FIELDS:
                return 'n/a'
            return _one(g, col, integer=integer)

        row: dict[str, Any] = {
            'arm': label or '(unlabelled)',
            'cell': cell or NOT_RECORDED,
            'condition': cond or NOT_RECORDED,
            'pair': pair,
            'transfer_set': field('transfer_set'),
            'input_policy': field('input_policy'),
            'head_policy': field('head_policy'),
            'layers': counts if transfers else 'n/a',
            'params_copied': field('params_copied', integer=True),
            'frac': fnum(_series(g, 'transferred_param_fraction').mean(), 3,
                         missing='n/a' if not transfers else NOT_RECORDED),
            'freeze_group': field('freeze_group'),
            'freeze_updates': field('freeze_updates'),
        }
        for c, _h in varying:
            if c in row:
                continue
            if not transfers and c in TRANSFER_ONLY_FIELDS:
                row[c] = 'n/a'
            elif cond != 'transfer_permuted' and c in PERMUTE_ONLY_FIELDS:
                row[c] = 'n/a'
            else:
                row[c] = _one(g, c)
        if transfers and ',' in row['freeze_updates']:
            moved.append(f'{label} (freeze_updates {row["freeze_updates"]})')
        rows.append(row)
        if names != NOT_RECORDED and cond != 'scratch':
            layer_names.append(f'{label}: {names}')

    notes = [
        'freeze_updates is a count of GRADIENT UPDATES, not episodes. Every '
        'schedule in this study is update-indexed because LunarLander episode '
        'length is strongly performance-dependent -- measured here, a random '
        'policy runs 94 steps at gravity -10 and 183 at gravity -4 -- so an '
        'episode-indexed freeze window would mean a different amount of '
        'learning in every arm (DESIGN.md 3.2).',
        'Layers c/p/r counts the layers fully copied, partially copied (an '
        'input- or output-facing shape mismatch, remainder redrawn) and '
        'reinitialised, read from transfer.summary in each run\'s '
        'manifest.json. Every freeze transition is additionally verified by a '
        'weight fingerprint and logged to events.jsonl; the manuscript\'s '
        'freeze schedule was never implemented at all, which is why '
        'verification is a run-time check rather than a claim in this table.',
        'Fraction is transferred_param_fraction. It is the treatment intensity, '
        'and it is not comparable across architectures unless it is matched: '
        'transfer_set="matched" is the primary level precisely so that both '
        'architectures sit near 97% (DESIGN.md 3.1).',
        f'per_seed.csv pins reinitialised_layer_count but not '
        f'{", ".join(MISSING_PER_SEED_COLUMNS)}, so the layer counts and names '
        f'here are read from the run manifests instead. The missing columns are '
        f'named rather than approximated. A transfer report was found for '
        f'{manifests_read} of the {n_transferring} arm(s) that transfer '
        f'anything; a scratch arm has none by construction and shows n/a.',
        'A scratch arm shows n/a for every transfer and freeze field. The '
        'Config defaults it carries are not printed as protocol, because a '
        'methods table that lists a transfer set for an arm that transferred '
        'nothing is the reading the published manuscript invited.',
    ]
    if moved:
        notes.append(
            'freeze_updates is NOT constant within arm(s) ' + '; '.join(moved)
            + '. Both values are shown rather than averaged: an arm whose '
              'freeze window moved is an arm audit.py refuses to aggregate, '
              'and stats.py suppresses its confirmatory member for the same '
              'reason (DESIGN.md 8.4). This is a defect in the input data, not '
              'a formatting choice.')
    if constant:
        notes.append(
            'Constant across every run tabulated here, and therefore not given '
            'a column: '
            + '; '.join(f'{c} = {v}' for c, v in constant)
            + '. This is the sense in which "identical hyperparameters" is a '
              'verified fact rather than a claim: audit.py refuses to '
              'aggregate runs that differ in a declared invariant '
              '(DESIGN.md 8.4).')
    if layer_names:
        head = layer_names[:12]
        notes.append('Layer names per transferring arm -- ' + ' | '.join(head)
                     + ('' if len(layer_names) == len(head)
                        else f' | ... and {len(layer_names) - len(head)} more '
                             f'arms, all recorded in the manifests'))
    legend = env_legend(list(df['env']) + list(df.get('source_env', [])))
    if legend:
        notes.append(legend)

    caption = build_caption(
        ctx,
        'Transfer protocol per arm: the transfer set, the layers copied, '
        'partially copied and reinitialised, the parameter fraction, the '
        'freeze group and the freeze window in gradient updates. This is the '
        'methods table whose absence made the published protocol '
        'unreproducible',
        test='No test: this table is the protocol, not a result',
        normalised=False)
    return Table('protocol_summary', 'Transfer protocol per arm',
                 caption, cols, rows, tuple(notes), frozenset(rules))


# ===========================================================================
# 12. Table 5 -- environments. The registry with the measured references, and
#     the difficulty-confound flag DERIVED from the measured no-op score, so
#     the caveat DESIGN.md 5.1 attaches to the gravity family cannot be lost by
#     forgetting to type it.
# ===========================================================================

def _variant_family(canonical: str) -> str:
    """Which named variant family a canonical env belongs to, if any."""
    for name in envs.VARIANT_FAMILIES:
        for _level, spec in envs.family_specs(name):
            if spec.canonical() == canonical and spec.params:
                return name
    return ''


def _interface_only_variant(spec) -> bool:
    """True when a variant differs from its base by interface wrappers alone.

    `DESIGN.md` 6.4's missing corner: dynamics identical by construction. Its
    no-op *score* still moves, because extending the action set makes the
    random-policy reference worse and the reference is the normalisation's zero
    -- so the drift there is a change in the measuring stick, not in how hard
    the task is, and flagging it as a difficulty confound would be wrong.
    """
    if not spec.params:
        return False
    try:
        base = envs.EnvSpec(spec.env_id, {})
    except (ValueError, KeyError):
        return False
    return spec.changes_interface_only(base)


def _registry_threshold(spec) -> float | None:
    """Fallback threshold from the gymnasium registry.

    Used only where no measured reference exists, so that an environment the
    design names but has not measured -- MountainCar-v0, excluded on purpose --
    still shows its registered threshold instead of a blank.
    """
    try:
        return spec.reward_threshold()
    except Exception:                                       # noqa: BLE001
        return None


def _env_role(canonical: str) -> str:
    if canonical == _canon(registry.TARGET_ENV):
        return 'target (primary)'
    if canonical == _canon(registry.SOURCE_ENV):
        return 'source (primary)'
    if canonical == _canon(registry.INTERFACE_ENV):
        return 'interface change only (C4)'
    if canonical == 'Acrobot-v1':
        return 'source (alternate)'
    if canonical == 'MountainCar-v0':
        return 'excluded from confirmatory work (DESIGN.md 6.1)'
    fam = _variant_family(canonical)
    if fam:
        return f'variant, {fam} family'
    return 'variant' if ':' in canonical else 'base'


def table_environments(df: pd.DataFrame, ctx: Context) -> Table:
    used = {_canon(e) for e in df['env'] if e}
    used |= {_canon(e) for e in df.get('source_env', []) if e}
    catalogue: set[str] = set(envs.load_references())
    catalogue |= {_canon(registry.SOURCE_ENV), _canon(registry.TARGET_ENV),
                  _canon(registry.INTERFACE_ENV)}
    catalogue |= set(envs.DESCRIPTORS)
    for name in envs.VARIANT_FAMILIES:
        catalogue |= {spec.canonical() for _l, spec in envs.family_specs(name)}
    catalogue |= used

    cols = (
        Col('env', 'Environment'),
        Col('role', 'Role'),
        Col('used', 'In data', 'c'),
        Col('obs', 'obs', 'r'),
        Col('act', 'act', 'r'),
        Col('threshold', 'threshold', 'r'),
        Col('random', 'random return', 'r'),
        Col('noop', 'no-op return', 'r'),
        Col('noop_score', 'no-op score', 'r'),
        Col('confound', 'Difficulty confound'),
    )
    base_noop: dict[str, float] = {}
    for canonical in sorted(envs.DESCRIPTORS):
        ref = env_reference(canonical)
        if ref and _isnum(ref.get('noop_score')):
            base_noop[canonical] = float(ref['noop_score'])

    rows: list[dict[str, Any]] = []
    rules: set[int] = set()
    prev_base: Optional[str] = None
    unmeasured: list[str] = []
    def sort_key(canonical: str) -> tuple:
        try:
            spec = envs.parse(canonical)
        except (ValueError, KeyError):
            return (9, canonical, canonical)
        # Rank by BASE environment, then by canonical string, so every variant
        # of one environment sits with its base rather than being scattered by
        # the reporting rank of the variant itself.
        return (_env_rank(spec.env_id)[0], spec.env_id, canonical)

    fallback_thresholds: list[str] = []
    for canonical in sorted(catalogue, key=sort_key):
        try:
            spec = envs.parse(canonical)
        except (ValueError, KeyError):
            continue
        if prev_base is not None and spec.env_id != prev_base:
            rules.add(len(rows))
        prev_base = spec.env_id
        ref = env_reference(canonical)
        if ref is None:
            unmeasured.append(canonical)
        noop_score = (ref or {}).get('noop_score')
        base = base_noop.get(spec.env_id)
        threshold = (ref or {}).get('threshold')
        if not _isnum(threshold):
            threshold = _registry_threshold(spec)
            if _isnum(threshold):
                fallback_thresholds.append(canonical)
        if not spec.params:
            confound = 'n/a (base environment)'
        elif _interface_only_variant(spec):
            confound = ('n/a (interface change only: dynamics identical by '
                        'construction, so any no-op score drift is the '
                        'random-policy reference moving, not the task getting '
                        'easier)')
        elif not _isnum(noop_score) or base is None:
            confound = NOT_RECORDED
        else:
            drift = float(noop_score) - base
            confound = (
                f'YES, no-op score {drift:+.2f} vs base: this axis changes how '
                f'hard the task is as well as how it behaves'
                if abs(drift) > NOOP_DRIFT_TOLERANCE
                else f'no, no-op score {drift:+.2f} vs base')
        rows.append({
            'env': canonical,
            'role': _env_role(canonical),
            'used': 'yes' if canonical in used else '--',
            'obs': str(spec.obs_dim), 'act': str(spec.act_dim),
            'threshold': fnum(threshold, 1),
            'random': fnum((ref or {}).get('random_return'), 1),
            'noop': fnum((ref or {}).get('noop_return'), 1),
            'noop_score': fnum(noop_score, 2),
            'confound': confound,
        })

    pairs: list[tuple[str, str]] = []
    if 'source_env' in df.columns:
        for src, tgt in {(str(a), str(b)) for a, b
                         in zip(df['source_env'], df['env']) if str(a)}:
            pairs.append((src, tgt))
    for _name, s, t in registry.ENV_PAIRS:
        if (s, t) not in pairs:
            pairs.append((s, t))
    pair_notes: list[str] = []
    for desc in envs.shift_descriptor_table(sorted(set(pairs))):
        s, t = envs.parse(desc['source']), envs.parse(desc['target'])
        # envs.shift_descriptor_table reports the BASE dimensions, so for a
        # padded/extended variant its obs/act strings understate the interface
        # change. The wrapped dimensions are taken from the EnvSpec instead,
        # because a pair labelled "interface CHANGED" beside "obs 8 -> 8" would
        # read as a contradiction.
        pair_notes.append(
            f'{desc["source"]} -> {desc["target"]}: obs '
            f'{s.obs_dim} -> {t.obs_dim}, act {s.act_dim} -> {t.act_dim}, '
            f'reward {desc["reward_density"]}, horizon {desc["horizon"]}; '
            f'interface '
            f'{"unchanged" if desc["interface_match"] else "CHANGED"}; shift '
            f'family: {desc["shift_family"]}; a scalar shift metric is '
            f'{"defined" if desc["scalar_shift_metric_defined"] else "NOT DEFINED"}.')

    notes = [
        'Random and no-op returns are measured, not remembered: 100 fixed-seed '
        'episodes per environment AND per variant, stored in '
        'experiments/reference_returns.json. Across the LunarLander gravity '
        'family the random-policy return moves from -202 to -463 and the score '
        'denominator from 402 to 663, so raw returns are not comparable across '
        'variants and a raw delta would fold a scale change into a shift '
        'effect (DESIGN.md 5.1).',
        'The threshold is the base environment\'s registered reward threshold '
        'even for a parametric variant, which is deliberate: these variants '
        'change the transition dynamics and leave the reward function '
        'untouched, so what counts as solved is unchanged while what a random '
        'policy achieves is not -- and it is the latter the normalisation '
        'corrects for.',
        f'The difficulty-confound column is DERIVED from the measured no-op '
        f'score, not asserted: a variant whose no-op score has moved by more '
        f'than {NOOP_DRIFT_TOLERANCE} from its base environment\'s has changed '
        f'task difficulty as well as dynamics. That is why the wind family is '
        f'the primary shift axis for H4 and gravity is secondary, reported with '
        f'the caveat attached (DESIGN.md 5.1, 6.2).',
        'No scalar shift metric is computed for a cross-interface pair. No '
        'distance between different state spaces is defined, and saying so is '
        'more defensible than inventing one -- the published '
        '2-Wasserstein-over-returns metric is not used, because returns are a '
        'consequence of the policy, the reward scale and the horizon cap '
        '(DESIGN.md 6.3).',
        'Source-target pairs and their structured shift descriptors: '
        + ' | '.join(pair_notes),
    ]
    if fallback_thresholds:
        notes.append(
            'Threshold taken from the gymnasium registry rather than from a '
            'measured reference for: ' + ', '.join(fallback_thresholds)
            + '. A threshold without a measured random-policy return cannot '
              'normalise anything, so these rows carry a threshold and no '
              'score.')
    if unmeasured:
        notes.append(
            'No measured reference for: ' + ', '.join(unmeasured)
            + '. envs.reference raises rather than defaulting to zero, so '
              'these environments cannot enter a normalised comparison until '
              '`python experiments/measure_references.py --env <name>` has '
              'been run. MountainCar-v0 is listed and excluded on purpose: '
              'DQN without shaping often fails it, which would reproduce the '
              'published invalid-source error.')

    caption = build_caption(
        ctx,
        'Environment registry: interface dimensions, the registered threshold, '
        'the measured random-policy and no-op references, the no-op score, and '
        'which shift axes are confounded with task difficulty',
        test='No test: this table is the measurement scale, not a result')
    return Table('environments',
                 'Environments, measured references and shift descriptors',
                 caption, cols, rows, tuple(notes), frozenset(rules))


# ===========================================================================
# 13. Table 6 -- power. Which cells are powered is a property of the observed
#     dispersion, known before any test is read, not a verdict discovered
#     afterwards. The multipliers are the pre-registered ones of
#     `ANALYSIS_PLAN.md` 6.2 and are NOT re-derived from the data: 6.4 says the
#     power table is not re-tuned after seeing results, so recomputing them
#     here would itself be a plan violation.
# ===========================================================================

def _powered_phrase(mde_holm: Any) -> str:
    """What the cell can detect, never a bare "yes".

    A binary powered/unpowered verdict invites the reading that a null in a
    "powered" cell is an absence of effect. The phrase states the smallest
    effect the cell can detect at the corrected alpha, which is the statement
    `ANALYSIS_PLAN.md` 6.3 actually licenses.
    """
    if not _isnum(mde_holm):
        return NOT_RECORDED
    mde = float(mde_holm)
    if mde >= UNPOWERED_MDE:
        return (f'NOT POWERED: MDE {mde:.3f} reaches {UNPOWERED_MDE} score '
                f'unit, the whole distance from random play to solved')
    return f'detects effects of {mde:.3f} score units or larger'


def _alpha_label(kind: str) -> str:
    return (f'{ALPHA}' if kind == 'nominal'
            else f'{ALPHA_STRICTEST:.5f} (Holm over {FAMILY_SIZE})')


def table_power(df: pd.DataFrame, ctx: Context,
                stats: Optional[dict]) -> Table:
    cols = (
        Col('scope', 'Scope'),
        Col('endpoint', 'Endpoint'),
        Col('cell', 'Cell / arm'),
        Col('n', 'n', 'r'),
        Col('sigma_delta', 'sigma(delta)', 'r'),
        Col('sigma_pooled', 'sigma(pooled)', 'r'),
        Col('observed', 'observed delta', 'r'),
        Col('p_nom', 'MDE paired, a=0.05', 'r'),
        Col('p_holm', 'MDE paired, Holm', 'r'),
        Col('u_nom', 'MDE unpaired, a=0.05', 'r'),
        Col('u_holm', 'MDE unpaired, Holm', 'r'),
        Col('powered', 'Powered?'),
    )
    rows: list[dict[str, Any]] = []
    rules: set[int] = set()
    notes: list[str] = []

    # Block A: the planning table. Pre-registered, computed before launch, and
    # available whether or not stats.py has run -- it is not a result.
    for arm, planned in PLANNING_SDS.items():
        cell, _, cond = arm.rpartition(' ')
        p_holm = MDE_MULTIPLIERS[('paired', 'holm8')] * planned
        rows.append({
            'scope': 'planning (ANALYSIS_PLAN.md 6.3)',
            'endpoint': 'final_score', 'cell': f'{cell} {cond}',
            'n': '10', 'sigma_delta': fnum(planned, 3),
            'sigma_pooled': fnum(planned, 3), 'observed': EM_DASH,
            'p_nom': fnum(MDE_MULTIPLIERS[('paired', 'nominal')] * planned, 3),
            'p_holm': fnum(p_holm, 3),
            'u_nom': fnum(MDE_MULTIPLIERS[('unpaired', 'nominal')] * planned, 3),
            'u_holm': fnum(MDE_MULTIPLIERS[('unpaired', 'holm8')] * planned, 3),
            'powered': _powered_phrase(p_holm),
        })

    # Block B: the observed table, which requires stats.py.
    rules.add(len(rows))
    if stats is None:
        notes.append(NO_STATS_NOTE)
        rows.append({
            'scope': 'observed', 'endpoint': 'final_score, auc_score',
            'cell': 'all four', 'n': NOT_RECORDED,
            'sigma_delta': 'requires --stats', 'sigma_pooled': NOT_RECORDED,
            'observed': NOT_RECORDED, 'p_nom': NOT_RECORDED,
            'p_holm': NOT_RECORDED, 'u_nom': NOT_RECORDED,
            'u_holm': NOT_RECORDED,
            'powered': 'not assessable without stats.py',
        })
    else:
        power = stats.get('s10_power', {})
        for r in power.get('per_member', []):
            if r.get('note') == 'suppressed' or 'paired_holm8' not in r:
                rows.append({
                    'scope': 'observed', 'endpoint': str(r.get('metric')),
                    'cell': str(r.get('cell')), 'n': fint(r.get('n')),
                    'sigma_delta': NOT_RECORDED,
                    'sigma_pooled': NOT_RECORDED, 'observed': NOT_RECORDED,
                    'p_nom': NOT_RECORDED, 'p_holm': NOT_RECORDED,
                    'u_nom': NOT_RECORDED, 'u_holm': NOT_RECORDED,
                    'powered': 'the member was suppressed; no dispersion to '
                               'scale the MDE by',
                })
                continue
            rows.append({
                'scope': 'observed', 'endpoint': str(r.get('metric')),
                'cell': str(r.get('cell')), 'n': fint(r.get('n')),
                'sigma_delta': fnum(r.get('sigma_delta')),
                'sigma_pooled': fnum(r.get('sigma_pooled')),
                'observed': fnum(r.get('observed_delta')),
                'p_nom': fnum(r.get('paired_nominal')),
                'p_holm': fnum(r.get('paired_holm8')),
                'u_nom': fnum(r.get('unpaired_nominal')),
                'u_holm': fnum(r.get('unpaired_holm8')),
                'powered': _powered_phrase(r.get('paired_holm8')),
            })
        comp = power.get('planning_comparison') or []
        if comp:
            notes.append(
                'Observed dispersion against the planning inputs: '
                + '; '.join(
                    f'{c.get("arm")} planned {fnum(c.get("planned_sd"))} vs '
                    f'observed {fnum(c.get("observed_sd"))} '
                    f'(ratio {fnum(c.get("ratio"), 2)})' for c in comp)
                + '. The planning SDs come from the published runs, which used '
                  'a different protocol, budget and exploration schedule, so '
                  'they are a planning input and not a prediction '
                  '(ANALYSIS_PLAN.md 6.4).')

    notes = [
        'MDE units are normalised score. The multiplier on sigma is '
        'pre-registered in ANALYSIS_PLAN.md 6.2 and is not re-derived from '
        'these data, because 6.4 forbids re-tuning the power table after '
        'seeing results: '
        + '; '.join(f'{k[0]} at alpha = {_alpha_label(k[1])}: {v} sigma'
                    for k, v in MDE_MULTIPLIERS.items())
        + '. statlib.py --self-test reproduces them from the exact null '
          'distribution.',
        'sigma(delta) is the SD of the PAIRED delta and scales the paired MDE; '
        'sigma(pooled) is the root-mean-square of the two arms\' SDs and '
        'scales the unpaired MDE. The 1.00-versus-1.39 gap in the multipliers '
        'is why the paired test is primary: at this sample size the '
        'matched-seed design is worth roughly a 40% reduction in the '
        'detectable effect (ANALYSIS_PLAN.md 6.2).',
        f'A cell is flagged NOT POWERED when its MDE at the Holm-corrected '
        f'alpha reaches {UNPOWERED_MDE} score unit(s) -- which by construction '
        f'of the normalisation is the entire distance from a random policy to '
        f'the registered threshold (DESIGN.md 5.1). Which cells are powered is '
        f'therefore known in advance from the dispersion, not discovered after '
        f'the test.',
        'A null result in an unpowered cell is a power result as much as an '
        'absence. The honest statement there is the exclusion bound of Table 2, '
        'not a null p-value (DESIGN.md 10.8, ANALYSIS_PLAN.md 4).',
        'At n=10 the exact sign-flip test cannot return a two-sided p below '
        f'{2 / 2 ** 10:.5f}, attained only when every seed moves the same way, '
        f'and the strictest Holm step is {ALPHA_STRICTEST:.5f}. A cell is '
        'therefore confirmed if and only if all ten of its seeds move in the '
        'same direction. That bar is stated before the numbers, not after '
        '(ANALYSIS_PLAN.md 2.2).',
        'The REPLICATE seed block exists so n can be doubled to 20 under a '
        'pre-registered pooling rule; the decision to run it is made on '
        'compute availability only, never on the n=10 outcome '
        '(ANALYSIS_PLAN.md 6.5).',
    ] + notes

    caption = build_caption(
        ctx,
        'Minimum detectable effects per cell at the nominal and the '
        'Holm-corrected alpha, and which cells are powered',
        endpoints=('final_score', 'auc_score'),
        test=f'MDE at 80% power. Multipliers pre-registered in '
             f'ANALYSIS_PLAN.md 6.2 (paired sign-flip and unpaired '
             f'Mann-Whitney), applied to the planning SDs of 6.3 and to the '
             f'dispersion actually observed. Family-wise alpha = {ALPHA} over '
             f'{FAMILY_SIZE} tests, strictest step {ALPHA_STRICTEST:.5f}')
    return Table('power', 'Power and minimum detectable effects',
                 caption, cols, rows, tuple(notes), frozenset(rules))


# ===========================================================================
# 14. Writing, with a provenance sidecar per table. `DESIGN.md` 9 lists "stale
#     artifacts" as a fallacy to guard: a table in the paper that no longer
#     matches the CSV it came from is undetectable without this file.
# ===========================================================================

BUILDERS: dict[str, str] = {
    'main_results': 'every arm x the four reported metrics, with intensity and '
                    'headroom',
    'inferential': 'every test and estimate, with its RQ and its family',
    'control_contrasts': 'the three contrasts plus C3b, per cell, with '
                         'exclusion restrictions',
    'protocol_summary': 'the transfer protocol per arm',
    'environments': 'the environment registry and measured references',
    'power': 'MDE per cell and which cells are powered',
}

FORMATS: tuple[str, ...] = ('latex', 'markdown')
_EXTENSION = {'latex': '.tex', 'markdown': '.md'}
_RENDER = {'latex': render_latex, 'markdown': render_markdown}


def build_table(key: str, df: pd.DataFrame, ctx: Context,
                stats: Optional[dict]) -> Table:
    if key == 'main_results':
        return table_main_results(df, ctx)
    if key == 'inferential':
        return table_inferential(df, ctx, stats)
    if key == 'control_contrasts':
        return table_control_contrasts(df, ctx, stats)
    if key == 'protocol_summary':
        return table_protocol_summary(df, ctx)
    if key == 'environments':
        return table_environments(df, ctx)
    if key == 'power':
        return table_power(df, ctx, stats)
    raise KeyError(f'unknown table {key!r}; known: {sorted(BUILDERS)}')


def write_table(t: Table, outdir: str, formats: Sequence[str],
                ctx: Context) -> dict:
    """Write one table in every requested format, plus its sidecar."""
    os.makedirs(outdir, exist_ok=True)
    written: dict[str, str] = {}
    for fmt in formats:
        path = os.path.join(outdir, t.key + _EXTENSION[fmt])
        with open(path, 'w', encoding='utf-8', newline='\n') as fh:
            fh.write(_RENDER[fmt](t))
        written[fmt] = path.replace(os.sep, '/')
    prov = {
        'tool': 'experiments/tables.py',
        'table': t.key,
        'title': t.title,
        'label': t.label,
        'what': BUILDERS[t.key],
        'columns': [c.key for c in t.cols],
        'headers': [c.header for c in t.cols],
        'rows': len(t.rows),
        'notes': list(t.notes),
        'caption': t.caption,
        'outputs': {fmt: {'path': p, 'sha': provenance.file_hash(p)}
                    for fmt, p in written.items()},
        'inputs': {
            'per_seed_csv': ctx.per_seed_path.replace(os.sep, '/'),
            'per_seed_sha': ctx.per_seed_sha,
            'per_seed_rows': ctx.n_runs,
            'stats_json': (ctx.stats_path or '').replace(os.sep, '/') or None,
            'stats_sha': ctx.stats_sha,
        },
        'arms': ctx.n_arms,
        'seeds_per_arm': {'min': ctx.min_n, 'max': ctx.max_n},
        'validation_stamp': VALIDATION_STAMP if ctx.validation else None,
        'plan_hash_in_run_data': list(ctx.plan_hash_in_data),
        'plans': ctx.plan_hashes,
        'git': ctx.git,
        'inference_constants': {
            'alpha': ALPHA,
            'alpha_strictest_holm_step': ALPHA_STRICTEST,
            'confirmatory_family_size': FAMILY_SIZE,
            'confirmatory_endpoints': list(CONFIRMATORY_ENDPOINTS),
            'equivalence_margin': EQUIVALENCE_MARGIN,
            'min_n_for_inference': MIN_N_FOR_INFERENCE,
            'mde_multipliers': {f'{k[0]}/{k[1]}': v
                                for k, v in MDE_MULTIPLIERS.items()},
            'unpowered_mde_score_units': UNPOWERED_MDE,
            'source_validity_gate': SOURCE_VALIDITY_GATE,
            'noop_drift_tolerance': NOOP_DRIFT_TOLERANCE,
        },
        'latex_unmapped_characters': dict(_UNMAPPED),
        'argv': list(ctx.argv),
        'cwd': os.getcwd(),
    }
    prov_path = os.path.join(outdir, t.key + '.provenance.json')
    with open(prov_path, 'w', encoding='utf-8', newline='\n') as fh:
        json.dump(prov, fh, indent=2, sort_keys=True)
    written['provenance'] = prov_path.replace(os.sep, '/')
    return {'table': t.key, 'rows': len(t.rows), 'files': written}


def multiplicity_ledger(stats: Optional[dict], tables: Sequence[Table]) -> str:
    """The ledger `ANALYSIS_PLAN.md` 7 requires on every invocation.

    Printed here as well as by `stats.py`, because a table is what reaches a
    reader: the count of analyses carrying no p-value has to travel with the
    artifacts, not only with the console session that made them.
    """
    lines = [
        '== multiplicity ledger (ANALYSIS_PLAN.md 7) ==',
        f'  family      : Confirmatory',
        f'  members     : {FAMILY_SIZE} (4 cells x '
        f'{len(CONFIRMATORY_ENDPOINTS)} co-primary endpoints: '
        f'{", ".join(CONFIRMATORY_ENDPOINTS)})',
        f'  procedure   : Holm-Bonferroni over {FAMILY_SIZE}',
        f'  adjusted a  : step-down from {ALPHA_STRICTEST:.5f} '
        f'(family-wise alpha {ALPHA})',
        '  screens     : Benjamini-Hochberg q, orientation only, no assertion '
        'permitted',
        '  everything else: estimation-only, no p-values emitted',
    ]
    no_p = 0
    total = 0
    for t in tables:
        if t.key != 'inferential':
            continue
        for r in t.rows:
            total += 1
            if r.get('p_raw') == EM_DASH:
                no_p += 1
    lines.append(f'  analyses carrying NO p-value, as tabulated: {no_p} of '
                 f'{total} rows in the inferential table')
    if stats is not None:
        ledger = stats.get('s11_ledger', {})
        lines.append(f'  stats.py estimation-only sections: '
                     f'{len(ledger.get("estimation_only", []))}')
        lines.append(f'  stats.py suppressed members       : '
                     f'{len(ledger.get("suppressed", []))}')
        lines.append(f'  stats.py refusals                 : '
                     f'{len(ledger.get("refusals", []))}')
        dev = stats.get('s12_deviations', {}).get('deviations', [])
        lines.append(f'  deviations recorded in stats.py    : {len(dev)}')
        for d in dev:
            lines.append(f'    - {d}')
    else:
        lines.append('  stats.py output was not supplied, so the inferential, '
                     'control-contrast and observed-power tables carry '
                     'placeholders rather than results.')
    return '\n'.join(lines)


# ===========================================================================
# 15. Self-test. Cheap assertions on the two things a table can get silently
#     wrong: the escaping (a table that does not compile, or a label whose
#     characters were dropped) and the em-dash convention (a blank read as an
#     omission). Also asserts that the compact interval verdict agrees in
#     direction with `stats.phrase_interval_verdict`, so the two cannot drift.
# ===========================================================================

def self_test(verbose: bool = True) -> int:
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = '') -> None:
        checks.append((name, bool(ok), detail))

    # --- escaping ---------------------------------------------------------
    esc = latex_escape('transfer_set & 100% $x$ #1 {a} ~b^c ' + _BS + 'd')
    check('LaTeX specials are all escaped',
          all(tok in esc for tok in (_BS + '_', _BS + '%', _BS + '$',
                                     _BS + '#', _BS + '{', _BS + '}',
                                     'textasciitilde', 'textasciicircum',
                                     'textbackslash')), esc)
    check('an unescaped raw ampersand cannot survive',
          '&' not in esc.replace(_BS + '&', ''), esc)
    check('plus-minus becomes a math command',
          latex_escape('1.0 ± 0.2') == '1.0 $' + _BS + 'pm$ 0.2',
          latex_escape('1.0 ± 0.2'))
    check('the em dash renders as three hyphens',
          latex_escape(EM_DASH) == '---', latex_escape(EM_DASH))
    check('no non-ASCII survives escaping',
          all(ord(c) < 128 for c in latex_escape(
              'σ ρ θ Δ × ≥ → § ± — “x” … 1.0')),
          latex_escape('σ ρ θ Δ × ≥ → § ± — “x” … 1.0'))
    before = dict(_UNMAPPED)
    unknown = latex_escape('☃')
    check('an unmapped character is rendered visibly and recorded',
          unknown == '[U+2603]' and '☃' in _UNMAPPED
          and '☃' not in before, unknown)
    _UNMAPPED.pop('☃', None)
    check('newlines inside a cell are collapsed, not emitted',
          '\n' not in latex_escape('a\nb') and latex_escape('a\nb') == 'a b')
    check('a markdown cell escapes the delimiter',
          markdown_escape('a|b') == 'a' + _BS + '|b')

    # --- the em-dash convention -------------------------------------------
    check('a missing p renders as an em dash, never as a blank',
          fmt_p(None) == EM_DASH and fmt_p(float('nan')) == EM_DASH
          and fmt_p('') == EM_DASH)
    check('a real p keeps five decimals',
          fmt_p(0.00195) == '0.00195' and fmt_p(0.05) == '0.05000',
          fmt_p(0.00195))
    check('a p below 1e-4 goes to scientific notation',
          fmt_p(1.08e-5) == '1.08e-05', fmt_p(1.08e-5))
    check('a missing number is n/r, which is not the em dash',
          fnum(None) == NOT_RECORDED and NOT_RECORDED != EM_DASH)

    # --- formatting -------------------------------------------------------
    check('no SD is printed at n=1', fmt_mean_sd([0.5]) == '0.500',
          fmt_mean_sd([0.5]))
    check('mean +/- sd at n>1 uses the sample SD',
          fmt_mean_sd([0.0, 1.0]) == '0.500 ± 0.707', fmt_mean_sd([0.0, 1.0]))
    check('an empty arm formats as n/r', fmt_mean_sd([]) == NOT_RECORDED)
    check('a point estimate without an interval says so',
          fmt_ci(0.1, None, None) == '0.100 [no interval]',
          fmt_ci(0.1, None, None))

    # --- verdicts agree with stats.py's prose -----------------------------
    for lo, hi in ((0.1, 0.2), (-0.2, -0.1), (-0.1, 0.1), (0.0, 0.2)):
        mine = verdict_word(lo, hi)
        theirs = statsmod.phrase_interval_verdict(lo, hi, 'x')
        agree = (('positive' in mine) == ('is positive' in theirs)
                 and ('negative' in mine) == ('is negative' in theirs)
                 and ('not distinguishable' in mine)
                 == ('not distinguishable' in theirs))
        check(f'compact verdict agrees with the prose for [{lo}, {hi}]',
              agree, f'{mine!r} vs {theirs!r}')
    check('the exclusion bound is always stated as a bound',
          exclusion_bound_text(-0.25) == 'worse than 0.250 excluded',
          exclusion_bound_text(-0.25))

    # --- rendering --------------------------------------------------------
    t = Table('probe', 'Probe', 'A caption with 100% & _underscores_.',
              (Col('a', 'A'), Col('b', 'B', 'r')),
              [{'a': 'x_1', 'b': fmt_p(None)}, {'a': 'y', 'b': '0.5'}],
              notes=('one note with a ± in it',), rules_before=frozenset({1}))
    tex = render_latex(t)
    md = render_markdown(t)
    for token in ('toprule', 'midrule', 'bottomrule', 'begin{tabular}',
                  'caption{', 'label{tab:probe}', 'begin{minipage}'):
        check(f'LaTeX contains {token}', token in tex)
    check('every LaTeX body row ends with a row terminator',
          tex.count(_BS * 2) >= 3, str(tex.count(_BS * 2)))
    check('a grouping midrule is emitted inside the body',
          tex.count(_BS + 'midrule') == 2, str(tex.count(_BS + 'midrule')))
    check('siunitx is not used', 'siunitx' not in tex and _BS + 'num{' not in tex)
    check('resizebox is not used', 'resizebox' not in tex)
    check('the markdown mirror has the same row count',
          len([ln for ln in md.splitlines()
               if ln.startswith('|')]) == 2 + len(t.rows) + 1,
          md)
    check('the markdown mirror carries the same caption',
          t.caption in md)
    check('a narrow table is not starred, a wide one is',
          not t.star and Table('w', 'w', 'c',
                               tuple(Col(str(i), str(i)) for i in range(7))
                               ).star)

    ok = all(c[1] for c in checks)
    if verbose:
        for name, passed, detail in checks:
            print(f'  {"PASS" if passed else "FAIL"}  {name}'
                  + (f'   [{detail}]' if detail and not passed else ''))
        print(f'\n{sum(1 for c in checks if c[1])}/{len(checks)} checks passed')
    return 0 if ok else 1


# ===========================================================================
# 16. CLI.
# ===========================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='tables.py',
        description='Generate the paper tables (booktabs LaTeX plus a markdown '
                    'mirror) from per_seed.csv and, for the inferential '
                    'tables, from stats.py --json output.')
    p.add_argument('--per-seed', default=os.path.join('runs', 'per_seed.csv'),
                   help='per-run table produced by aggregate.py')
    p.add_argument('--stats', default=None,
                   help='stats.py --json output. Without it the inferential, '
                        'control-contrast and observed-power tables carry an '
                        'explicit refusal instead of numbers.')
    p.add_argument('--outdir', default=os.path.join('paper', 'tables'),
                   help='directory for the .tex, .md and .provenance.json '
                        'files')
    p.add_argument('--format', dest='formats', default='latex,markdown',
                   help='comma-separated: latex, markdown')
    p.add_argument('--tables', default=None,
                   help='comma-separated subset of '
                        + ','.join(BUILDERS) + ' (default: all)')
    p.add_argument('--experiments', nargs='*', default=None,
                   help='restrict to runs belonging to these experiment ids; '
                        'an exclusion is a selection, so the count excluded is '
                        'printed')
    p.add_argument('--self-test', action='store_true',
                   help='run the escaping and formatting assertions and exit')
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return self_test()

    formats = [f.strip() for f in str(args.formats).split(',') if f.strip()]
    unknown = [f for f in formats if f not in FORMATS]
    if unknown or not formats:
        print(f'tables.py: unknown --format {unknown or "(empty)"}; '
              f'known: {",".join(FORMATS)}')
        return 1
    keys = ([k.strip() for k in str(args.tables).split(',') if k.strip()]
            if args.tables else list(BUILDERS))
    unknown_keys = [k for k in keys if k not in BUILDERS]
    if unknown_keys:
        print(f'tables.py: unknown --tables {unknown_keys}; '
              f'known: {",".join(BUILDERS)}')
        return 1

    try:
        df = load_per_seed(args.per_seed)
    except (FileNotFoundError, ValueError) as exc:
        print(f'tables.py: {exc}')
        return 1
    try:
        stats = load_stats(args.stats)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f'tables.py: {exc}')
        return 1

    if args.experiments:
        selected = set(args.experiments)
        keep = df['experiments'].fillna('').apply(
            lambda s: bool(set(str(s).split(';')) & selected))
        dropped = int((~keep).sum())
        df = df[keep].reset_index(drop=True)
        print(f'selection: {",".join(sorted(selected))} keeps {len(df)} run(s) '
              f'and excludes {dropped}. An exclusion is a selection, not a '
              f'cleaning step, so the count is printed.')
        if not len(df):
            print('tables.py: nothing left after the selection; refusing to '
                  'tabulate an empty set.')
            return 1

    # TUNE seeds may never enter a reported estimate (ANALYSIS_PLAN.md 8), and
    # a table IS a reported estimate. Dropped here with the count stated.
    if 'seed_block' in df.columns:
        tune = int((df['seed_block'] == 'TUNE').sum())
        if tune:
            df = df[df['seed_block'] != 'TUNE'].reset_index(drop=True)
            print(f'excluded {tune} run(s) on TUNE seeds: a reported estimate '
                  f'may never draw on the block hyperparameters were selected '
                  f'on (ANALYSIS_PLAN.md 8, DESIGN.md 3.4).')
        if not len(df):
            print('tables.py: every run was on TUNE seeds; nothing may be '
                  'tabulated from them.')
            return 1

    ctx = build_context(df, args.per_seed, args.stats,
                        list(sys.argv if argv is None else ['tables.py', *argv]))
    print(f'tables.py -- {ctx.n_runs} runs, {ctx.n_arms} arms, '
          f'n={ctx.min_n}-{ctx.max_n} seeds per arm')
    if ctx.validation:
        print(f'\n{VALIDATION_STAMP}')
        print('  At least one tabulated arm has n < '
              f'{MIN_N_FOR_INFERENCE}. Every caption below carries this stamp. '
              'No number in these tables may be quoted, compared, or used to '
              'choose between hypotheses (ANALYSIS_PLAN.md 9, '
              'STANDING_INSTRUCTIONS S8).')
    if len(ctx.plan_hash_in_data) > 1:
        print(f'\nWARNING: {len(ctx.plan_hash_in_data)} distinct '
              f'ANALYSIS_PLAN.md hashes across the input runs. A confirmatory '
              f'result is interpretable only against the one pre-registration '
              f'in force when it ran: {list(ctx.plan_hash_in_data)}')
    elif (ctx.plan_hash_in_data
          and ctx.plan_hash_in_data[0] != ctx.plan_hashes.get(
              'ANALYSIS_PLAN.md')):
        print('\nWARNING: the plan hash recorded in the run data differs from '
              'the current ANALYSIS_PLAN.md. Any confirmatory number in these '
              'tables is exploratory until the plan change is recorded in '
              'ANALYSIS_PLAN.md 11.')

    built: list[Table] = []
    print()
    for key in keys:
        t = build_table(key, df, ctx, stats)
        result = write_table(t, args.outdir, formats, ctx)
        built.append(t)
        files = ', '.join(v for k, v in result['files'].items()
                          if k != 'provenance')
        print(f'{key:<20} {result["rows"]:>4} rows -> {files}')
        print(f'{"":<20} {"":>4}         '
              f'{result["files"]["provenance"]}')

    print()
    print(multiplicity_ledger(stats, built))
    if _UNMAPPED:
        print()
        print('WARNING: character(s) with no LaTeX mapping were rendered as '
              '[U+XXXX] and recorded in every provenance sidecar: '
              + ', '.join(f'U+{ord(c):04X} x{n}'
                          for c, n in sorted(_UNMAPPED.items())))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
