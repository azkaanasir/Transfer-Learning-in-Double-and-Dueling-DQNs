"""Every figure the paper carries, from the two pinned CSVs and nothing else.

Why this file exists in this form
--------------------------------
ICANN reviewer concern **C8** (`REVIEW_COVERAGE.md`): the published figures were
three panels squeezed into one row, with axis labels too small to read at column
width, and a caption that said the curves were shown "after smoothing" without
ever stating the window. A reader could not tell what had been averaged, over
how many seeds, or against what reference the scores were normalised. That is
not a cosmetic failure -- an unstated smoothing window is an unstated analysis
choice, and a figure whose caption cannot be checked against it is an
unfalsifiable claim.

Five properties are therefore mechanical here, not editorial:

* **The analysis set is the same one `stats.py` reports on.** A figure and the
  table beside it must not disagree about which runs exist. Three filters are
  applied once, in `analysis_set`, and every one of them is stated in every
  caption and recorded in every provenance file: `TUNE` seeds are removed
  (`ANALYSIS_PLAN.md` §8, selection leakage), donor-only seed blocks are
  removed (`DESIGN.md` §3.4: `C4SRC` and `RESERVE` runs exist to hand over a
  checkpoint and are barred from target-side estimation), and under the default
  `--source-policy valid` every run whose source failed the §4.3 normalised
  gate is removed. That last one is not a detail: plotting an arm whose source
  never learned its task, next to a table that excluded it, is the published
  study's own error reproduced in the figures. `--source-policy pooled` draws
  the pre-declared secondary instead and says so on the canvas and in the
  caption; it is never called intent-to-treat.
* **Captions are generated from the data.** `<name>.caption.txt` is written
  beside every figure and states the seed count actually plotted, the
  evaluation protocol *read out of the manifests of the runs in the figure*,
  the interval method and the number of bootstrap resamples, the smoothing
  window (the literal word `none` when there is none), and the normalisation
  reference with its measured numbers. A caption cannot contradict its figure
  because neither is written by hand. The evaluation protocol in particular is
  read from disk rather than restated from `DESIGN.md` §5.2: the demonstration
  runs in `runs_demo/` evaluate 5 episodes at 2 checkpoints, and a caption that
  claimed the design's 100 episodes at 3 checkpoints would be exactly the C8
  defect reintroduced.
* **Provenance travels with every figure** (`DESIGN.md` §8.3).
  `<name>.provenance.json` records the content hash of each input CSV, the git
  commit and dirty flag,
  the `ANALYSIS_PLAN.md` hash, the exact argv, and the arm labels the figure
  resolved -- so a stale figure is detectable rather than plausible.
* **No inference is invented here.** Every interval comes from `stats.py`'s
  pre-registered estimators (`hodges_lehmann_paired`, `bootstrap_statistic`'s
  bias-corrected bootstrap, `kaplan_meier`, `clopper_pearson`) at its fixed
  `N_BOOT` and `BOOT_SEED`. A figure computing its own CI by its own method
  would eventually disagree with the table beside it, and the reader would have
  no way to tell which was right. The interval *method* the estimator actually
  returned travels into the caption too, because `bca_interval` falls back to a
  percentile interval on a degenerate bootstrap distribution and a caption that
  still said BCa would be the C8 defect in miniature. A degenerate interval
  (zero width, from a constant sample) yields no equivalence verdict and no
  exclusion bound at all: the cell with the least information must not produce
  the strongest claim. **No figure draws a p-value at all**, even
  for the confirmatory family: a forest of significance verdicts invites the
  "A avoids negative transfer while B does not" comparison that `DESIGN.md` §9
  and `ANALYSIS_PLAN.md` §8 forbid. Tests live in `stats.py`'s tables, where
  the Holm correction is visible next to them.
* **x is env steps, never episodes**, for every learning curve (`DESIGN.md`
  §3.2). LunarLander episode length is strongly performance-dependent -- 94
  steps for a random policy at gravity -10 against 183 at gravity -4 -- so an
  episode-indexed curve silently compares arms at different amounts of
  learning. For the same reason the freeze boundary, which is defined in
  gradient updates, is located per run from the `frozen` flag and drawn at the
  env step where the window actually ended, or not drawn at all with the
  caption saying why.

Presentation rules
------------------
Okabe-Ito colours; vermillion and green are never used together, so no figure
relies on a red/green distinction. Condition is carried by colour *and* marker
*and* dash pattern, so the figures survive greyscale printing. One panel per
cell at readable font sizes rather than a single crowded row. A figure whose
smallest arm has fewer than `stats.MIN_N_FOR_INFERENCE` seeds is still drawn --
seeing that the pipeline produced a curve is the point of a validation run --
but it is stamped `PIPELINE VALIDATION` across the canvas and its intervals are
suppressed rather than computed (`ANALYSIS_PLAN.md` §9,
`STANDING_INSTRUCTIONS.md` S8).

Usage
-----
    python experiments/plots.py --per-seed runs/per_seed.csv \
        --curves runs/curves.csv --outdir paper/figures
    python experiments/plots.py --per-seed runs/per_seed.csv \
        --curves runs/curves.csv --outdir paper/figures \
        --format pdf,png --figures learning,forest
    python experiments/plots.py --per-seed runs/per_seed.csv \
        --curves runs/curves.csv --outdir paper/figures_pooled \
        --source-policy pooled
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')                                            # noqa: E402
import matplotlib.pyplot as plt                                  # noqa: E402
from matplotlib.lines import Line2D                              # noqa: E402
from matplotlib.patches import Patch                             # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import registry                                                  # noqa: E402
import stats                                                     # noqa: E402
import tuning                                                    # noqa: E402
from src.dqn import envs, provenance                              # noqa: E402

WARN = '[WARNING]'

# ---------------------------------------------------------------------------
# 1. Presentation constants
#
# Okabe-Ito, which is distinguishable under the three common forms of colour
# blindness. Vermillion (#D55E00) and bluish green (#009E73) are deliberately
# left out of every palette below, so no distinction in any figure is carried
# by a red/green pair.
# ---------------------------------------------------------------------------
COLUMN_WIDTH = 3.4          # inches, one journal column
FULL_WIDTH = 7.0            # inches, two columns

_BLACK = '#000000'
_BLUE = '#0072B2'
_ORANGE = '#E69F00'
_PURPLE = '#CC79A7'
_SKY = '#56B4E9'
_GREY = '#7F7F7F'

#: Condition styling. The C0-C3 prefixes are `DESIGN.md` §4's names, kept on
#: the legend so a reader of the paper does not have to translate.
CONDITION_STYLE: dict[str, dict[str, Any]] = {
    'scratch': dict(colour=_BLACK, marker='o', dashes=(None, None),
                    name='C0 scratch'),
    'transfer': dict(colour=_BLUE, marker='s', dashes=(None, None),
                     name='C1 transfer'),
    'transfer_untrained': dict(colour=_ORANGE, marker='^', dashes=(4, 1.6),
                               name='C2 untrained source'),
    'transfer_permuted': dict(colour=_PURPLE, marker='D', dashes=(1.4, 1.4),
                              name='C3 permuted source'),
}

#: Cell styling, used where cells are series rather than panels.
CELL_STYLE: dict[str, dict[str, Any]] = {
    'mlp-vanilla': dict(colour=_BLUE, marker='o', dashes=(None, None)),
    'mlp-double': dict(colour=_SKY, marker='s', dashes=(4, 1.6)),
    'dueling-vanilla': dict(colour=_ORANGE, marker='^', dashes=(2, 1.4)),
    'dueling-double': dict(colour=_PURPLE, marker='D', dashes=(1.2, 1.2)),
}

#: Registry order, so cells appear in the same order in every figure and table.
CELL_ORDER: tuple[str, ...] = tuple(f'{a}-{r}' for a, r in registry.CELLS)

#: Telescoping order (`DESIGN.md` §4.1): laid out C0, C2, C3, C1 so that the
#: three adjacent gaps *are* the three named contrasts, left to right.
CONTRAST_ORDER: tuple[str, ...] = ('scratch', 'transfer_untrained',
                                   'transfer_permuted', 'transfer')

#: Short codes for the conditions, used where an annotation has to fit inside
#: a column-width panel. The long names are in the legend and the caption.
CONDITION_CODE: dict[str, str] = {
    'scratch': 'C0', 'transfer': 'C1', 'transfer_untrained': 'C2',
    'transfer_permuted': 'C3',
}

#: The endpoints `ANALYSIS_PLAN.md` §4 fixes an equivalence margin for. The
#: margin is +/-0.05 *normalised-score* units, justified as ~20 return points on
#: LunarLander, and that justification does not transfer to an area-per-env-step
#: quantity. So P2 gets no margin band, no equivalence verdict and no
#: "score units" label: an equivalence claim on a scale the plan never defined a
#: margin for would be a new analysis choice made in a plotting script.
MARGIN_ENDPOINTS: tuple[str, ...] = ('final_score',)

#: The unit each endpoint's exclusion bound is quoted in.
ENDPOINT_UNITS: dict[str, str] = {
    'final_score': 'normalised score units',
    'auc_score': 'normalised-score-per-env-step units',
}

CONTRAST_NAMES: tuple[tuple[str, str, str], ...] = (
    ('transfer_untrained', 'scratch', 'untrained-source'),
    ('transfer_permuted', 'transfer_untrained', 'permuted-source'),
    ('transfer', 'transfer_permuted', 'trained-vs-permuted'),
)

RC = {
    'figure.dpi': 150,
    'savefig.dpi': 400,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'pdf.fonttype': 42,          # embed as TrueType, not Type 3
    'ps.fonttype': 42,
    'font.size': 8.0,
    'axes.titlesize': 8.5,
    'axes.labelsize': 8.0,
    'axes.titlepad': 3.0,
    'axes.labelpad': 2.0,
    'xtick.labelsize': 7.0,
    'ytick.labelsize': 7.0,
    'legend.fontsize': 7.0,
    'legend.frameon': False,
    'legend.handlelength': 2.2,
    'legend.borderaxespad': 0.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'axes.grid': True,
    'grid.color': '#BFBFBF',
    'grid.linewidth': 0.35,
    'grid.alpha': 0.6,
    'lines.linewidth': 1.2,
    'lines.markersize': 3.2,
    'errorbar.capsize': 0.0,
    'axes.axisbelow': True,
}


# ---------------------------------------------------------------------------
# 2. Small formatting helpers
# ---------------------------------------------------------------------------
def _f(value: Any, digits: int = 3) -> str:
    """Format a number for a caption, or a two-hyphen placeholder if the value
    is missing or non-finite. Never prints `nan`, which reads as a number."""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return '--'
    if not np.isfinite(x):
        return '--'
    return f'{x:.{digits}f}'


def _i(value: Any) -> str:
    try:
        return f'{int(round(float(value))):,}'
    except (TypeError, ValueError):
        return '--'


def _seed_list(seeds: Sequence[int], limit: int = 12) -> str:
    """Seeds for a caption, always as integers.

    A single row with a missing seed used to promote the whole seed index to
    float, and captions then read `0.0, 1.0, 2.0`. Such rows no longer reach a
    figure (`analysis_set` removes them loudly), and this is the second belt:
    a seed is an integer identifier, so it is printed as one.
    """
    out = []
    for s in seeds:
        try:
            out.append(str(int(s)))
        except (TypeError, ValueError):
            out.append(str(s))
    if len(out) <= limit:
        return ', '.join(out)
    return ', '.join(out[:limit]) + f', ... (+{len(out) - limit} more)'


def _cell_label_with_n(cell: str, n_by_cell: dict[str, set]) -> str:
    """Cell name with its seed count, for a tick label.

    n lives in the tick label rather than in an annotation inside the axes,
    where a wide interval collides with it. Where the endpoints disagree on n
    -- one endpoint missing for a seed -- every value is shown rather than the
    first, because a hidden n is how a partial arm gets read as a complete one.
    """
    name = cell.replace('-', ' / ')
    values = sorted(n_by_cell.get(cell, ()))
    if not values:
        return name
    return f"{name}\n(n={'/'.join(str(v) for v in values)})"


def _bool_tokens(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """CSV booleans, plus the mask of values that were actually recognised.

    Booleans arrive from a CSV as bool, str or object depending on whether the
    column had any missing values, so the parse has to be explicit. It also has
    to be honest about failure: returning a bare `False` for an unrecognised
    token turns a *missing* censoring flag into an *observed event*, which is
    how three runs that never reached a threshold were reported as 3/3 reached.
    The second return value is therefore the recognised mask, and every caller
    that reads a flag whose absence changes a number is required to look at it.
    """
    def one(v: Any) -> tuple[bool, bool]:
        if isinstance(v, (bool, np.bool_)):
            return bool(v), True
        if isinstance(v, str):
            token = v.strip().lower()
            if token in ('true', '1', 'yes', 't'):
                return True, True
            if token in ('false', '0', 'no', 'f'):
                return False, True
            return False, False
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return False, False
        try:
            return bool(int(v)), True
        except (TypeError, ValueError):
            return False, False
    pairs = [one(v) for v in series.tolist()]
    return (np.array([p[0] for p in pairs], dtype=bool),
            np.array([p[1] for p in pairs], dtype=bool))


def _as_bool(series: pd.Series) -> np.ndarray:
    """`_bool_tokens` where an unrecognised token may safely read as False.

    Safe means: False is the conservative direction for this particular flag.
    `metrics_contiguous` is one (an unreadable integrity flag should count as a
    failed check); a censoring flag is emphatically not, and that caller uses
    `_bool_tokens` directly.
    """
    return _bool_tokens(series)[0]


def _trailing_mean(y: np.ndarray, window: int) -> np.ndarray:
    """Trailing mean over `window` points, shrinking at the start.

    Trailing rather than centred: a centred window at evaluation point *t* uses
    measurements from after *t*, which makes an early part of the curve depend
    on later training and would misplace anything read off against the freeze
    boundary.
    """
    if window <= 1:
        return y
    cum = np.concatenate(([0.0], np.cumsum(y)))
    out = np.empty_like(y)
    for i in range(len(y)):
        lo = max(0, i - window + 1)
        out[i] = (cum[i + 1] - cum[lo]) / (i + 1 - lo)
    return out


def _no_data(ax: plt.Axes, message: str, hide_x: bool = False,
             hide_y: bool = False) -> None:
    """Say that a panel is empty and why, rather than leaving blank axes.

    Only the tick *marks* and the grid go; the tick *labels* stay. Two
    mistakes are being avoided. `set_xticks([])` on a shared axis changes the
    locator for every panel in the group, so one empty panel would strip the
    tick labels off its populated neighbours. And on a shared axis the empty
    panel is often the very one that carries the group's labels -- the
    left-hand panel of a row -- so switching its labels off blanks the scale
    for the whole row. `hide_x` / `hide_y` are for the case where the axis is
    genuinely not shared with anything populated, so its scale means nothing.
    """
    ax.text(0.5, 0.5, textwrap.fill(message, 34), transform=ax.transAxes,
            ha='center', va='center', fontsize=6.6, color=_GREY, style='italic')
    ax.tick_params(which='both', bottom=False, left=False,
                   labelbottom=not hide_x, labelleft=not hide_y)
    ax.grid(False)


def _legend(fig: plt.Figure, handles: list, ncol: int, y: float = 0.0) -> None:
    if handles:
        fig.legend(handles=handles, loc='lower center', ncol=ncol,
                   bbox_to_anchor=(0.5, y))


#: How many lines of "not drawn" a panel may carry before the note defers to
#: the caption. The note has to be readable at column width and must not sit on
#: top of the data; the caption is where the full list lives, and it is
#: generated from the same record, so the two cannot disagree.
ABSENT_NOTE_LINES = 4


def note_absent(ax: plt.Axes, entries: Sequence[str],
                where: tuple[float, float] = (0.03, 0.03),
                va: str = 'bottom') -> None:
    """Name, on the canvas, the series this panel does NOT contain and why.

    `_no_data` covers the panel that is entirely empty. The panel that keeps
    its scratch arm and loses its transfer arm had nothing at all: it was drawn
    with one curve where its neighbours have two, a shorter x tick set, or a
    single lone marker with no line, and the reader was given no way to tell an
    absent arm from an arm that was measured and came out flat. On the real
    tree that is not hypothetical -- the `DESIGN.md` 4.3 source gate removes
    every dueling-vanilla transfer arm -- and it affected five of the nine
    figures. The caption named the removal in all nine, which protects a reader
    who reads the caption; a figure lifted into a slide or a draft carries no
    caption, which is the case `stamp_pooled` and `stamp_validation` were
    written for.
    """
    if not entries:
        return
    lines = textwrap.wrap('NOT DRAWN: ' + '; '.join(entries), 52)
    if len(lines) > ABSENT_NOTE_LINES:
        lines = lines[:ABSENT_NOTE_LINES - 1]
        lines.append(f'... and more; the full list is in the caption '
                     f'({len(entries)} series).')
    ax.annotate(chr(10).join(lines), xy=where, xycoords='axes fraction',
                fontsize=5.2, color='#8A6D3B', ha='left', va=va, zorder=6)


def absent_reason(ctx: Context, label: Optional[str]) -> str:
    """Why an expected arm is not in the analysis set, from the record.

    Read out of `ctx.analysis`, so the reason on the canvas is the same reason
    the caption gives and neither is typed. The forest printed " no matched
    seeds" for the gate-removed arms, which is a true statement about the
    pairing and the wrong attribution: the seeds ARE matched, the arm was
    removed by the DESIGN.md 4.3 gate before any pairing was attempted.
    """
    if not label:
        return 'arm not defined for this cell'
    # Compact by design. The canvas note has to stay off the data at column
    # width; the numbers behind each reason are in the analysis-set paragraph
    # of the caption, which is generated from this same record, so shortening
    # here cannot make the two disagree.
    named = {'invalid_source': 'removed: source failed the DESIGN.md 4.3 gate',
             'tune_seeds': 'removed: TUNE seed (ANALYSIS_PLAN.md 8)',
             'donor_only_blocks': 'removed: donor-only block (DESIGN.md 3.4)',
             'duplicate_arm_rows': 'removed: duplicated row (DESIGN.md 8.2)',
             'unusable_seed': 'removed: unusable seed'}
    for key, rec in (ctx.analysis.get('removed') or {}).items():
        if label in (rec.get('labels') or []):
            return f'{label}: {named.get(key, "removed: " + key)}'
    if not len(ctx.per_seed[ctx.per_seed['label'] == label]):
        return f'{label}: no run with this label in the supplied table'
    return f'{label}: no run in the analysis set'


def absent_notes(ctx: Context, labels: Iterable[Optional[str]],
                 fallback: str = 'no usable data for this series here'
                 ) -> list[str]:
    """One reason per absent series, in order.

    Three cases, kept apart because they need different fixes: the arm has no
    label in the catalogue at all, the arm has a label but no rows left in the
    analysis set (which is where the gate removal lands), or the arm has rows
    and this particular figure could not use them. Reporting the second as the
    third is what "no matched seeds" did.
    """
    out: list[str] = []
    for label in labels:
        if not label:
            out.append('an arm of this cell is not defined in the catalogue')
        elif arm_rows(ctx, label).empty:
            out.append(absent_reason(ctx, label))
        else:
            out.append(f'{label}: {fallback}')
    return out


# ---------------------------------------------------------------------------
# 3. Loading, and the arm labels the figures are built on
# ---------------------------------------------------------------------------
def resolve_arm_labels() -> dict[str, dict[str, str]]:
    """cell -> condition -> arm label, read out of the catalogue.

    Taken from the registry rather than hard-coded, because the label is the
    arm's identity: `DESIGN.md` §11's fourth defect was nine conditions from
    six experiments colliding onto one run directory, and the fix was to make
    the label the thing that distinguishes them. E1 supplies the scratch
    baseline and the transfer arm at the primary `transfer_set='matched'`
    intensity; E2 supplies the two controls at the protocol's freeze window.
    The secondary arms -- `transfer_set='trunk'`, the K=0 control, the
    spectrum-matched control -- are deliberately excluded: each varies a second
    factor, and folding them in would rebuild the two-variables-one-claim
    defect the study exists to correct.
    """
    protocol_k = registry.PROTOCOL['freeze_updates']
    out: dict[str, dict[str, str]] = {}
    for exp_id in ('E1', 'E2'):
        exp = registry.EXPERIMENTS.get(exp_id)
        if exp is None:
            continue
        for arm in exp.arms:
            ov = arm.overrides
            if arm.role != 'target' or arm.only_as_source:
                continue
            if ov.get('env') != registry.TARGET_ENV:
                continue
            cond = ov.get('condition')
            if cond is None or 'arch' not in ov:
                continue
            if cond != 'scratch':
                if ov.get('transfer_set') != 'matched':
                    continue
                if ov.get('freeze_updates') != protocol_k:
                    continue
                if ov.get('permute_kind', 'shuffle') != 'shuffle':
                    continue
            cell = f"{ov['arch']}-{ov['target_rule']}"
            out.setdefault(cell, {}).setdefault(cond, arm.label)
    return out


def interface_labels() -> dict[str, dict[str, str]]:
    """cell -> condition -> label for E8i, the interface-change-only corner."""
    out: dict[str, dict[str, str]] = {}
    exp = registry.EXPERIMENTS.get('E8i')
    if exp is None:
        return out
    for arm in exp.arms:
        ov = arm.overrides
        if arm.only_as_source or 'arch' not in ov:
            continue
        if ov.get('env') != registry.INTERFACE_ENV:
            continue
        cell = f"{ov['arch']}-{ov['target_rule']}"
        out.setdefault(cell, {})[ov['condition']] = arm.label
    return out


def shift_labels() -> dict[str, list[dict[str, Any]]]:
    """cell -> [{level, env, scratch, transfer}] for E8's shift families."""
    out: dict[str, list[dict[str, Any]]] = {}
    exp = registry.EXPERIMENTS.get('E8')
    if exp is None:
        return out
    per: dict[tuple[str, str], dict[str, Any]] = {}
    for arm in exp.arms:
        ov = arm.overrides
        if arm.only_as_source or 'arch' not in ov:
            continue
        env = ov.get('env')
        if env == registry.TARGET_ENV:
            continue                       # the level-0 scratch denominator
        cell = f"{ov['arch']}-{ov['target_rule']}"
        rec = per.setdefault((cell, env), {'env': env})
        rec[ov['condition']] = arm.label
    for (cell, _env), rec in per.items():
        out.setdefault(cell, []).append(rec)
    return out


#: `--source-policy` values. The same two names and the same meanings as
#: `stats.py`'s flag, deliberately: a figure whose analysis set differs from the
#: table printed beside it is precisely the disagreement this module exists to
#: make impossible.
SOURCE_POLICIES: tuple[str, ...] = ('valid', 'pooled')

#: The conditions whose source is a *trained* checkpoint and therefore has a
#: validity verdict to have. C0 has no source at all and C2's source is
#: untrained by construction (`DESIGN.md` §4), so an empty `source_valid` on
#: either is the expected value, not a missing measurement. Only C1 and C3 carry
#: the §4.3 gate, and only for them is a missing verdict worth a warning.
SCORED_SOURCE_CONDS: tuple[str, ...] = ('transfer', 'transfer_permuted')


def analysis_set(per_seed: pd.DataFrame,
                 source_policy: str = 'valid') -> tuple[pd.DataFrame, dict]:
    """The runs a figure may draw, and a full record of everything removed.

    Four filters, each of which exists because leaving it out reproduces a named
    defect. None of them is silent: the record returned here is printed to
    stdout, summarised in every caption and written into every provenance file,
    because "dropping a seed, for any reason, after it has run" is forbidden by
    `ANALYSIS_PLAN.md` §8 and the only honest alternative to dropping quietly is
    saying exactly what was dropped and why.

    1. **An unusable seed.** A row whose `seed` will not parse as an integer
       cannot be matched to its partner in any paired contrast. It used to
       vanish inside `groupby(['label', 'seed'])`, which defaults to
       `dropna=True`, so an arm quietly fell from n=3 to n=2 while stdout still
       reported the full run count. Now it is removed here, counted, and named.
    2. **`TUNE` seeds** (`DESIGN.md` §3.4, `ANALYSIS_PLAN.md` §8). No reported
       estimate may be computed on hyperparameter-selection seeds: that is
       selection leakage, and a figure is a reported estimate. Selection by arm
       label alone does not exclude them, because a TUNE run of an E1 arm
       carries the E1 label.
    3. **Donor-only seed blocks** (`DESIGN.md` §3.4). `C4SRC` runs exist to hand
       a checkpoint to a C4 arm and `RESERVE` runs to replace a rejected source;
       both are barred from target-side estimation. A C4SRC donor is a scratch
       run on the target environment, so without this filter it pools into that
       cell's scratch baseline and shifts the denominator of every delta.
    4. **Sources that failed the validity gate** (`DESIGN.md` §4.3), under the
       default `valid` policy. A source is valid when its own normalised final
       score is at least 0.6. Plotting a transfer arm whose source never learned
       its task is the published study's actual error; in P0 it is live, because
       `src-dueling-vanilla` scored 0.599 against the 0.600 gate. `pooled` is
       the pre-declared secondary and keeps them, labelled as such, never called
       intent-to-treat.
    5. **A seed counted twice** (`DESIGN.md` §8.2). See the comment at the
       filter itself: this is what makes every n in every figure a count of
       distinct seeds rather than of rows.
    """
    record: dict[str, Any] = {
        'source_policy': source_policy,
        'rows_in_table': int(len(per_seed)),
        'removed': {},
    }
    df = per_seed

    seeds = pd.to_numeric(df['seed'], errors='coerce')
    bad_seed = ~np.isfinite(seeds.to_numpy(float))
    if bool(bad_seed.any()):
        lost = df[bad_seed]
        record['removed']['unusable_seed'] = {
            'n': int(bad_seed.sum()),
            'reason': 'the seed does not parse as an integer, so the run '
                      'cannot be matched to its partner in any paired '
                      'contrast',
            'runs': sorted(str(r) for r in lost['run_dir'].tolist()),
            'labels': sorted(set(str(v) for v in lost['label'].tolist())),
        }
        df = df[~bad_seed]
        seeds = seeds[~bad_seed]
    df = df.assign(seed=seeds.astype('int64'))

    if 'seed_block' in df.columns:
        tune = df['seed_block'].astype(str) == 'TUNE'
        if bool(tune.any()):
            lost = df[tune]
            record['removed']['tune_seeds'] = {
                'n': int(tune.sum()),
                'reason': 'ANALYSIS_PLAN.md 8: no estimate may be computed on '
                          'hyperparameter-selection seeds',
                'seeds': sorted({int(s) for s in lost['seed'].tolist()}),
                'labels': sorted(set(str(v) for v in lost['label'].tolist())),
            }
            df = df[~tune]
        donor = df['seed_block'].astype(str).isin(stats.SOURCE_ONLY_BLOCKS)
        if bool(donor.any()):
            lost = df[donor]
            record['removed']['donor_only_blocks'] = {
                'n': int(donor.sum()),
                'blocks': sorted(set(str(v) for v in
                                     lost['seed_block'].tolist())),
                'reason': f'DESIGN.md 3.4: {list(stats.SOURCE_ONLY_BLOCKS)} '
                          f'exist only to donate a source checkpoint and are '
                          f'barred from target-side estimation',
                'seeds': sorted({int(s) for s in lost['seed'].tolist()}),
                'labels': sorted(set(str(v) for v in lost['label'].tolist())),
            }
            df = df[~donor]

    if source_policy == 'valid':
        invalid = df['source_valid'] == False                    # noqa: E712
        if bool(invalid.any()):
            lost = df[invalid]
            scores = pd.to_numeric(lost.get('source_final_score'),
                                   errors='coerce')
            record['removed']['invalid_source'] = {
                'n': int(invalid.sum()),
                'reason': 'DESIGN.md 4.3: the primary estimand is valid '
                          'sources only (normalised source score >= 0.6)',
                'seeds': sorted({int(s) for s in lost['seed'].tolist()}),
                'labels': sorted(set(str(v) for v in lost['label'].tolist())),
                'source_scores': sorted(
                    {round(float(v), 4) for v in scores.dropna().tolist()}),
            }
            df = df[~invalid]

    # 5. **A seed counted twice.** An arm label plus a seed names one run, so
    #    two ROWS under one (label, seed) are one observation recorded twice,
    #    not two observations. Removed here, once, so that no figure can
    #    inherit the old behaviour by reading `per_seed` directly: `arm_rows`
    #    filters on `run_dir in selected`, which keeps BOTH rows when they
    #    share a path, and `fig_km_threshold` is the one figure that used those
    #    rows without going through `_rows_to_series`. On a two-seed tree with
    #    every row duplicated it therefore printed "4/4 reached, 95% CI [0.40,
    #    1.00]" in every panel against the true 2/2 [0.16, 1.00], and n_min=4
    #    took the `PIPELINE VALIDATION` stamp off a caption that still said
    #    "n=2 distinct seeds plotted". n is a seed count everywhere in this
    #    project (`tables.n_seeds`, `stats.py`'s duplicate refusal), and this
    #    is what makes it one here.
    dupes: list[str] = []
    per_label: list[pd.DataFrame] = []
    for label, group in df.groupby('label', sort=True, dropna=False):
        per_label.append(one_row_per_seed(group, str(label), dupes))
    if per_label:
        kept = pd.concat(per_label).sort_index()
        if len(kept) < len(df):
            lost = df.loc[df.index.difference(kept.index)]
            record['removed']['duplicate_arm_rows'] = {
                'n': int(len(df) - len(kept)),
                'reason': 'DESIGN.md 8.2: more than one row for one (arm '
                          'label, seed). One run recorded twice is one '
                          'observation, and counting it twice tightens every '
                          'interval scaled by n. One row per seed is kept, '
                          'chosen by run directory then by row content so the '
                          'choice does not depend on the order of the CSV',
                'seeds': sorted({int(s) for s in lost['seed'].tolist()}),
                'labels': sorted(set(str(v) for v in lost['label'].tolist())),
                'arm_seeds': sorted(set(dupes)),
            }
        df = kept

    # An UNSCORED source is not a passed source. `DESIGN.md` §4.3 defines
    # validity as a score at or above the gate, so a transfer run whose
    # `source_valid` is empty has no verdict at all, and neither policy licenses
    # reading that as a pass. Scratch runs have no source and are expected to be
    # empty here, so the two are counted separately: the second number is the
    # one that means something is wrong upstream.
    if len(df):
        unscored = df['source_valid'].isna()
        transfer_like = df['condition'].astype(str).isin(SCORED_SOURCE_CONDS)
        record['rows_with_no_source_verdict'] = int(unscored.sum())
        record['transfer_rows_with_no_source_verdict'] = int(
            (unscored & transfer_like).sum())
        record['arms_with_no_source_verdict'] = sorted(set(
            str(v) for v in df.loc[unscored & transfer_like, 'label']))
    else:
        record['rows_with_no_source_verdict'] = 0
        record['transfer_rows_with_no_source_verdict'] = 0
        record['arms_with_no_source_verdict'] = []
    record['rows_in_analysis_set'] = int(len(df))
    return df, record


def analysis_set_sentence(record: dict) -> str:
    """The analysis set in one caption paragraph, generated from the record."""
    policy = record.get('source_policy', 'valid')
    if policy == 'valid':
        head = ('Analysis set: VALID SOURCES ONLY, the primary estimand of '
                'DESIGN.md 4.3 (a source is valid when its own normalised '
                'final score is at least 0.6).')
    else:
        head = ('Analysis set: POOLED OVER SOURCE COMPETENCE, the pre-declared '
                'SECONDARY of DESIGN.md 4.3 and NOT the primary estimand. It '
                'is not intent-to-treat: source competence is known before the '
                'target run begins, so it is not a post-randomisation '
                'compliance event. Every number in this figure pools transfer '
                'from a competent source with transfer from a source that '
                'never learned, which is the published study\'s actual error, '
                'reproduced here deliberately and labelled.')
    parts = [head,
             f"{record.get('rows_in_analysis_set', 0)} of "
             f"{record.get('rows_in_table', 0)} rows in the supplied table "
             f"enter it."]
    removed = record.get('removed') or {}
    names = {'unusable_seed': 'unusable seed',
             'tune_seeds': 'TUNE seeds (ANALYSIS_PLAN.md 8)',
             'donor_only_blocks': 'donor-only seed blocks (DESIGN.md 3.4)',
             'invalid_source': 'source failed the DESIGN.md 4.3 validity gate',
             'duplicate_arm_rows': 'a second row for one (arm label, seed), '
                                   'which is one run recorded twice and not '
                                   'two seeds (DESIGN.md 8.2)'}
    for key, rec in removed.items():
        seeds = rec.get('seeds')
        parts.append(f"Removed for {names.get(key, key)}: {rec['n']} run(s)"
                     + (f" at seed(s) {_seed_list(seeds)}" if seeds else '')
                     + (f", arms {', '.join(rec['labels'][:6])}"
                        + (' and others' if len(rec.get('labels', [])) > 6
                           else '')
                        if rec.get('labels') else '')
                     + '.')
    if not removed:
        parts.append('No run in the supplied table was removed by any of '
                     'these filters.')
    n_unscored = record.get('transfer_rows_with_no_source_verdict', 0)
    if n_unscored:
        arms = record.get('arms_with_no_source_verdict') or []
        parts.append(f"WARNING: {n_unscored} run(s) in the analysis set "
                     f"transferred from a source with NO validity verdict at "
                     f"all (source_valid is empty, not False), in arm(s) "
                     f"{', '.join(arms[:6])}"
                     + (' and others' if len(arms) > 6 else '') + '. '
                     "DESIGN.md 4.3 defines validity as a score at or above "
                     "the gate, so an unscored source is not a passed source "
                     "and neither policy licenses reading it as one. These "
                     "runs are NOT removed, because removing them would be a "
                     "filter the plan does not define; they are named here "
                     "instead.")
    return ' '.join(parts)


def print_analysis_set(record: dict) -> None:
    """The analysis set on stdout, in the shape `stats.py` prints it.

    Loud rather than tidy. Every one of these removals changes a number in
    every figure, and the previous version of this module printed a run count
    that included runs no figure could use, which is how "96 runs" sat above a
    forest built on 2 seeds per arm.
    """
    policy = record.get('source_policy', 'valid')
    print(f"  analysis set: {record.get('rows_in_analysis_set', 0)} of "
          f"{record.get('rows_in_table', 0)} runs, source policy {policy!r} "
          + ('(valid sources only, the primary estimand of DESIGN.md 4.3)'
             if policy == 'valid' else
             '(POOLED OVER SOURCE COMPETENCE: the pre-declared SECONDARY of '
             'DESIGN.md 4.3, never intent-to-treat)'))
    removed = record.get('removed') or {}
    if not removed:
        print('    nothing removed')
    for key, rec in removed.items():
        print(f"    -{rec['n']:>4} {key}: {rec['reason']}")
        if rec.get('labels'):
            shown = ', '.join(rec['labels'][:6])
            more = (f" (+{len(rec['labels']) - 6} more arms)"
                    if len(rec['labels']) > 6 else '')
            print(f"          arms: {shown}{more}")
        if rec.get('source_scores'):
            print(f"          source scores seen: {rec['source_scores']}")
    if record.get('transfer_rows_with_no_source_verdict'):
        print(f"{WARN} "
              f"{record['transfer_rows_with_no_source_verdict']} run(s) in the "
              f"analysis set transferred from a source with NO validity "
              f"verdict (source_valid empty, not False): "
              f"{', '.join((record.get('arms_with_no_source_verdict') or [])[:6])}. "
              f"DESIGN.md 4.3 does not license an unscored source as a pass. "
              f"They are plotted and named in every caption rather than "
              f"removed by a rule the plan does not define.")
    if policy == 'pooled':
        print('    !! every figure will be stamped SECONDARY ANALYSIS SET and '
              'no number in it is the primary estimand')


def row_content_key(rows: pd.DataFrame) -> pd.Series:
    """A per-row sort key built from the row's CONTENT, never its position.

    The tie-break under every de-duplication in this module. `_rows_to_series`
    sorted on `run_dir` alone with a STABLE sort and promised that "sorted by
    path so the choice is reproducible"; for two rows sharing one `run_dir` the
    sort is a no-op and the survivor is whichever row happens to come first in
    the CSV. That is not reproducibility, and it is not cosmetic: on a tree
    holding one duplicated row, emitting the two colliding rows in the opposite
    order flipped the sign of the Hodges-Lehmann shift in two members of the
    confirmatory family (mlp-double on both co-primary endpoints), while every
    provenance hash except the CSV's matched. Sorting on the content makes the
    survivor a function of the SET of rows, so any permutation of the file
    yields the same figure.
    """
    if rows.empty:
        return pd.Series(dtype=object, index=rows.index)
    cols = sorted(str(c) for c in rows.columns)
    # Built element by element rather than with a vectorised join: a frame of
    # mixed dtypes does not reliably present every cell as a string to a
    # row-wise aggregate, and one float reaching `str.join` raises. The key is
    # only ever compared to another key, so the exact rendering does not
    # matter; that it is a total function of the row's values does.
    unit = chr(31)
    values = rows[cols].to_numpy(dtype=object)
    return pd.Series([unit.join(str(v) for v in row) for row in values],
                     index=rows.index, dtype=object)


def one_row_per_seed(rows: pd.DataFrame, name: str,
                     duplicates: list[str]) -> pd.DataFrame:
    """One row per seed, chosen by path then by content, duplicates recorded.

    `stats.py` refuses a duplicated (arm, seed) outright. Refusing is not open
    to this module -- its own docstring says refusing to draw is `audit.py`'s
    job -- so the equivalent here is to keep exactly one row per seed by a rule
    that does not depend on file order, and to name every seed it happened to.
    What must never happen is the third option: counting both rows as two
    observations, which is what turned one seed into "2/2 reached" with a
    tighter Clopper-Pearson interval in the Kaplan-Meier panels.
    """
    if rows.empty or 'seed' not in rows.columns:
        return rows
    keyed = rows.assign(**{'_content_key': row_content_key(rows)})
    order = ['run_dir', '_content_key'] if 'run_dir' in keyed.columns \
        else ['_content_key']
    keyed = keyed.sort_values(order, kind='mergesort')
    dupe = keyed['seed'].duplicated(keep=False)
    if bool(dupe.any()):
        for seed in sorted(set(keyed.loc[dupe, 'seed'].tolist())):
            duplicates.append(f'{name}@s{seed}')
        keyed = keyed.drop_duplicates(subset=['seed'], keep='first')
    return keyed.drop(columns=['_content_key'])


def resolve_selection(per_seed: pd.DataFrame
                      ) -> tuple[set[str], list[dict], list[dict]]:
    """One run directory per (arm label, seed), chosen deterministically.

    An arm label plus a seed is supposed to name exactly one run: the label is
    the arm's identity, and revision 1's fourth fatal defect was nine
    conditions from six experiments collapsing onto one run directory
    (`DESIGN.md` §11). Two DISTINCT run directories under one (label, seed)
    therefore means the table mixes *configurations* inside one arm -- runs
    produced under different registry settings, or a directory left behind by
    an earlier protocol. Averaging across them would silently pool two
    treatments, which is the published study's original error.

    Two rows sharing ONE run directory is a different thing entirely, and the
    previous version reported it as the first: on a table with every row
    duplicated it printed "72 (arm label, seed) pair(s) resolve to MORE THAN
    ONE run directory ... this table mixes configurations inside one arm
    identity", and then contradicted itself on the next line with "fields that
    differ between the colliding runs: none, identical configs, duplicated on
    disk". Nothing had resolved to more than one directory and no configuration
    was mixed; a row was repeated. The two are counted and named separately
    now, and each points at the section that actually covers it: `DESIGN.md`
    §11 for a collision, §8.2 for a duplicated row.

    The figure is still drawn, because refusing to draw is `audit.py`'s job and
    not a plotting decision, but both are reported loudly, the surviving run is
    chosen deterministically, and the discarded paths go into every provenance
    record.
    """
    keep: set[str] = set()
    collisions: list[dict] = []
    duplicate_rows: list[dict] = []
    # `dropna=False`, against pandas' default: a group key that is missing must
    # produce a visible group rather than a vanished run. `analysis_set` has
    # already removed and named any row whose seed will not parse, so this is
    # the second belt, not the first.
    for (label, seed), group in per_seed.groupby(['label', 'seed'], sort=True,
                                                 dropna=False):
        run_dirs = sorted(set(str(r) for r in group['run_dir'].tolist()))
        keep.add(run_dirs[0])
        if len(run_dirs) > 1:
            varying = sorted(
                col for col in ('freeze_updates', 'num_episodes', 'lr',
                                'transfer_set', 'freeze_group', 'permute_kind',
                                'aggregation', 'value_recal', 'target_update',
                                'hidden', 'head_units', 'env', 'source_env')
                if col in group.columns
                and group[col].astype(str).nunique() > 1)
            collisions.append({'label': str(label), 'seed': int(seed),
                               'kept': run_dirs[0],
                               'discarded': run_dirs[1:],
                               'fields_that_differ': varying})
        elif len(group) > 1:
            duplicate_rows.append({'label': str(label), 'seed': int(seed),
                                   'run_dir': run_dirs[0],
                                   'rows': int(len(group)),
                                   'identical_rows': bool(
                                       group.astype(str).nunique().max() == 1)})
    return keep, collisions, duplicate_rows


# ---------------------------------------------------------------------------
# The DESIGN.md 3.3 arbitration, READ here and never re-derived.
#
# `stats.py` writes two keys onto every confirmatory member:
# `arbitration_verdict`, the verdict of DESIGN.md 3.3's arbitration between the
# common-configuration and the per-cell-tuned hyperparameter policies, and
# `asserted`, the narrower flag saying whether a conclusion may be drawn.
# `ANALYSIS_PLAN.md` 2.4 closes the loop on the consumer side: "A downstream
# artifact may not present a confirmatory conclusion from the significance flag
# alone ... `report.py`, `tables.py` and `plots.py` must read those rather than
# re-deriving licence from the p-value. A guard the consumer ignores is not a
# guard."
#
# All three modules ignored both keys and derived licence from
# `significant_holm` alone. On today's data -- where every verdict is
# `not-evaluable` because the tuned arms have not been run -- that published a
# confirmatory conclusion for every member of the family, out of a bundle whose
# own stats.json said that nothing in it was assertable.
#
# This block is duplicated verbatim in `report.py`, `tables.py` and `plots.py`,
# because it is the gate each of them has to pass and none of the three is a
# natural home for the other two. Each module's `--self-test` asserts the
# vocabulary against `stats.py`, so a drift is a test failure rather than a
# silent divergence.
# ---------------------------------------------------------------------------
#: The two keys, named once. A consumer that spelled either differently would
#: read `None` and fail OPEN, so they are constants rather than literals.
ARBITRATION_KEY = 'arbitration_verdict'
ASSERTED_KEY = 'asserted'

#: The three verdicts, taken FROM `stats.py` rather than re-listed, so the
#: vocabulary cannot drift between the module that computes it and the three
#: that consume it.
ARBITRATION_VERDICTS: tuple[str, ...] = tuple(stats.ARBITRATION_VERDICTS)
AGREES = stats.AGREES
DISAGREES = stats.DISAGREES
NOT_EVALUABLE = stats.NOT_EVALUABLE

#: `ANALYSIS_PLAN.md` 2.4: "`not-evaluable` is the default, so an unrun tuned
#: stage cannot silently license a conclusion." A missing key, an unreadable
#: verdict, a missing flag, and a member contradicting its own arbitration
#: table all land on this verdict, and every one of them blocks.
ARBITRATION_DEFAULT = NOT_EVALUABLE

#: What each fail-closed defect means, in the words the artifacts print. They
#: are named apart because "the tuned leg has not been run" and "the key is
#: missing from stats.json" are different repairs, and a reader told only that
#: something is not evaluable cannot tell which one is owed.
ARBITRATION_MISSING: dict[str, str] = {
    '': 'the tuned leg has not been run',
    'key-absent': 'stats.json carries no ' + repr(ARBITRATION_KEY)
                  + ' on this member',
    'unparseable': 'the arbitration recorded in stats.json is unreadable',
    'flag-absent': 'stats.json carries no ' + repr(ASSERTED_KEY)
                   + ' flag on this member',
    'flag-contradicts-verdict':
        'stats.json sets ' + repr(ASSERTED_KEY) + ' on a member whose verdict '
        'is not ' + repr(AGREES),
    'inconsistent': 'the member and the arbitration table in stats.json give '
                    'different verdicts for the same member',
    'withheld': 'stats.py withheld the assertion although the two policies '
                'agree',
}

#: Sentinel for "the key is not in the dict at all", which is a different state
#: from "the key is present and null". `dict.get` cannot tell them apart, and
#: the difference is the whole of the deleted-key test.
_ARB_ABSENT = object()


def _arbitration_flag(value: Any) -> Optional[bool]:
    """A JSON boolean, or None when the value is not a boolean at all.

    `stats.py` round-trips Python bools through `int` on the way into JSON, so
    `asserted` arrives as 0 or 1 rather than false or true, and both are read.
    Everything else is refused rather than coerced, because `bool('false')` is
    True and that coercion is exactly how a hand-edited stats.json would turn
    into a licence.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, float) and value in (0.0, 1.0):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ('true', '1', 'yes'):
            return True
        if text in ('false', '0', 'no'):
            return False
    return None


def read_arbitration(member: Any, row: Any = None) -> dict:
    """The arbitration state of one confirmatory member. FAILS CLOSED.

    `member` is one element of `s5_confirmatory.members`. `row` is the matching
    element of `s5_confirmatory.arbitration.rows`, which carries the prose
    reason; the row is used for that reason and for a consistency check, never
    as a substitute for the member's own two keys, because the member is what
    every consumer reads.

    The returned `blocks` is the only thing a caller needs in order to decide
    whether a conclusion may be presented. Nothing here computes a verdict:
    `ANALYSIS_PLAN.md` 2.4 requires the consumer to READ the verdict, and a
    consumer that re-derived one would be a second implementation of the
    arbitration, free to disagree with the first.
    """
    member = member if isinstance(member, dict) else {}
    row = row if isinstance(row, dict) else {}
    verdict = ARBITRATION_DEFAULT
    defect = ''
    raw = member.get(ARBITRATION_KEY, _ARB_ABSENT)
    if raw is _ARB_ABSENT:
        defect = 'key-absent'
    elif isinstance(raw, str) and raw.strip().lower() in ARBITRATION_VERDICTS:
        verdict = raw.strip().lower()
    else:
        defect = 'unparseable'

    flag_raw = member.get(ASSERTED_KEY, _ARB_ABSENT)
    parsed = None if flag_raw is _ARB_ABSENT else _arbitration_flag(flag_raw)
    asserted = bool(parsed)
    if flag_raw is _ARB_ABSENT:
        defect = defect or 'flag-absent'
    elif parsed is None:
        defect = defect or 'unparseable'

    # The member and the arbitration table have to say the same thing: they are
    # written by one function in one pass, so a disagreement means the file was
    # edited or truncated and neither half of it can then be trusted.
    row_raw = row.get('verdict')
    if (not defect and isinstance(row_raw, str)
            and row_raw.strip().lower() in ARBITRATION_VERDICTS
            and row_raw.strip().lower() != verdict):
        defect = 'inconsistent'
    if not defect and asserted and verdict != AGREES:
        defect = 'flag-contradicts-verdict'
    if not defect and verdict == AGREES and not asserted:
        defect = 'withheld'

    blocks = bool(defect) or verdict != AGREES or not asserted
    why = str(row.get('why') or '').strip()
    missing = ARBITRATION_MISSING.get(defect, ARBITRATION_MISSING[''])

    if not blocks:
        label = 'ASSERTED: both policies agree (DESIGN.md 3.3)'
        blocked_because = ''
        sentence = ('DESIGN.md 3.3 arbitration: AGREES, and stats.py set '
                    + ASSERTED_KEY + ', so ANALYSIS_PLAN.md 2.4 licenses a '
                    'conclusion from this member. The verdict is read from '
                    'stats.json, not re-derived from the p-value.')
    elif verdict == DISAGREES and defect in ('', 'flag-contradicts-verdict'):
        label = ('DISAGREEMENT between the two hyperparameter policies: THIS '
                 'IS THE FINDING')
        blocked_because = (
            'the common-configuration and per-cell-tuned policies of '
            'DESIGN.md 3.3 reach different conclusions about this member'
            + (': ' + why if why else ''))
        sentence = ('DESIGN.md 3.3 arbitration: DISAGREES. '
                    + blocked_because[0].upper() + blocked_because[1:]
                    + '. ANALYSIS_PLAN.md 2.4 makes that disagreement the '
                    'REPORTED FINDING: it may not be suppressed, averaged '
                    'away, or resolved by preferring one policy, and NO '
                    'conclusion is asserted from either leg.')
    else:
        label = 'not evaluable: ' + missing
        blocked_because = 'not evaluable: ' + missing + (
            ' (' + why + ')' if why else '')
        sentence = ('DESIGN.md 3.3 arbitration: NOT EVALUABLE, because '
                    + missing + (' (' + why + ')' if why else '')
                    + '. ANALYSIS_PLAN.md 2.4 makes not-evaluable the default '
                    'and permits a conclusion only under agrees, so nothing is '
                    'asserted from this member however small its Holm-adjusted '
                    'p.')
    return {
        'verdict': verdict,
        'asserted': bool(asserted and not blocks),
        'verdict_in_json': None if raw is _ARB_ABSENT else raw,
        'asserted_in_json': None if flag_raw is _ARB_ABSENT else flag_raw,
        'blocks': bool(blocks),
        'defect': defect,
        'missing': missing,
        'why': why,
        'label': label,
        'blocked_because': blocked_because,
        'sentence': sentence,
    }


def arbitration_index(sj: Any) -> dict:
    """(metric, cell) -> the arbitration row `stats.py` wrote for that member."""
    conf = (sj or {}).get('s5_confirmatory') if isinstance(sj, dict) else None
    rows = ((conf or {}).get('arbitration') or {}).get('rows') or []
    out: dict[tuple[str, str], dict] = {}
    for row in rows:
        if isinstance(row, dict):
            out[(str(row.get('metric')), str(row.get('cell')))] = row
    return out


def arbitration_summary(sj: Any) -> dict:
    """The arbitration over the whole confirmatory family, in one block.

    Printed in captions and written into provenance records, so that a table or
    a figure separated from its bundle still says whether anything in it was
    assertable. An input carrying no confirmatory family reports that as an
    absence rather than as nothing to declare, because an empty family is
    exactly the shape a suppressed one has.
    """
    conf = (sj or {}).get('s5_confirmatory') if isinstance(sj, dict) else None
    members = ((conf or {}).get('members') or []) if isinstance(conf, dict) \
        else []
    index = arbitration_index(sj)
    counts = {v: 0 for v in ARBITRATION_VERDICTS}
    defects: dict[str, int] = {}
    asserted = 0
    disagreements: list[str] = []
    blocked: list[str] = []
    for member in members:
        member = member if isinstance(member, dict) else {}
        key = (str(member.get('metric')), str(member.get('cell')))
        arb = read_arbitration(member, index.get(key))
        counts[arb['verdict']] = counts.get(arb['verdict'], 0) + 1
        if arb['defect']:
            defects[arb['defect']] = defects.get(arb['defect'], 0) + 1
        tag = key[0] + '/' + key[1]
        if arb['blocks']:
            blocked.append(tag + ': ' + arb['label'])
        else:
            asserted += 1
        if arb['verdict'] == DISAGREES:
            disagreements.append(tag + ': ' + (arb['why'] or arb['label']))
    total = len(members)
    if not total:
        sentence = ('DESIGN.md 3.3 arbitration: this input carries no '
                    'confirmatory family, so there is no verdict to read and '
                    'nothing here may be presented as a confirmatory '
                    'conclusion (ANALYSIS_PLAN.md 2.4).')
    else:
        sentence = ('DESIGN.md 3.3 arbitration: ' + str(asserted) + ' of '
                    + str(total) + ' confirmatory member(s) may be asserted ('
                    + str(counts.get(NOT_EVALUABLE, 0)) + ' not evaluable, '
                    + str(counts.get(DISAGREES, 0)) + ' disagreeing, '
                    + str(counts.get(AGREES, 0)) + ' agreeing). '
                    'ANALYSIS_PLAN.md 2.4 permits a conclusion only where both '
                    'hyperparameter policies agree, so the other '
                    + str(total - asserted) + ' carry their statistic and '
                    'their Holm-adjusted p but NO conclusion.')
        if disagreements:
            sentence += (' The ' + str(len(disagreements))
                         + ' DISAGREEMENT(S) are themselves the reported '
                         'finding and may not be resolved by preferring one '
                         'policy: ' + '; '.join(disagreements) + '.')
        if defects:
            sentence += (' Fail-closed defects in stats.json: '
                         + ', '.join(k + ' x' + str(v)
                                     for k, v in sorted(defects.items()))
                         + '.')
    return {
        'rule': 'DESIGN.md 3.3, ANALYSIS_PLAN.md 2.4: a conclusion is asserted '
                'only where the common-configuration and per-cell-tuned '
                'policies agree; not-evaluable is the default and blocks',
        'members': total,
        'asserted': asserted,
        'blocked': total - asserted,
        'by_verdict': counts,
        'defects': defects,
        'disagreements': disagreements,
        'blocked_members': blocked,
        'sentence': sentence,
    }


#: See the note on the same name in `report.py`. `plots.py` imports its
#: siblings absolutely, so `import tuning` here is the same module object
#: `audit` and `registry` hold, not a second copy.
_SELECTION_MODULE = tuning


#: What a consumer prints where the selection cannot be read at all. FAILS
#: CLOSED, like `read_arbitration`: the subject is never omitted, because a
#: bundle that says nothing about which pre-registration the tuned arms were
#: selected under reads as one where the question did not arise.
SELECTION_PROVENANCE_UNREAD = (
    'Tuning selection: NOT READ from this run tree, so the pre-registration '
    'behind the DESIGN.md 3.3 tuned leg is unstated (ANALYSIS_PLAN.md 1).')


def selection_plan_provenance(out_root: Any,
                              current: Optional[str] = None) -> dict:
    """Which `ANALYSIS_PLAN.md` version the tuned leg's evidence was produced under.

    `ANALYSIS_PLAN.md` 1 puts the plan hash in "every run manifest and every
    emitted table and figure" and re-labels every result produced under a
    superseded version **exploratory**. For the runs in `per_seed.csv` that is
    already handled: they carry `plan_hash` and a mismatch is stamped. The
    tuned leg's evidence is not in that table. `E3` runs on the `TUNE` block,
    which `ANALYSIS_PLAN.md` 8 bars from every reported estimate, so a
    selection computed from runs produced under a superseded plan reaches the
    tuned arms without leaving a trace in anything downstream of it. On the
    completed screen that is not hypothetical: 155 of `E3`'s 160 runs were
    produced under two superseded versions, which `ANALYSIS_PLAN.md` 11
    predicted and `audit.py` reports as `plan_hash_split`.

    Reads the artifact and never re-derives a selection. Never raises: an
    absent, unreadable or edited artifact is reported as UNREAD, which is a
    different statement from clean and is written down as one.
    """
    out = {'present': False, 'short_id': None, 'drift': False, 'split': False,
           'stale': False, 'unknown': False, 'hashes': [], 'counts': {},
           'rows_without_a_hash': 0, 'error': None, 'note': '',
           'sentence': SELECTION_PROVENANCE_UNREAD}
    if not out_root:
        return out
    try:
        selection = _SELECTION_MODULE.read_selection(
            str(out_root), required=False, warn_placeholder=False,
            warn_plan_drift=False)
    except Exception as exc:                       # noqa: BLE001 - see below
        # Deliberately broad. This is a provenance annotation on somebody
        # else's artifact, and every way of failing to read it (absent,
        # truncated, edited so it no longer hashes to its own id, written under
        # an older schema) has the same consequence here: the pre-registration
        # is unstated. Letting one of them abort a report would make an
        # unreadable selection quieter than a readable one.
        out['error'] = f'{type(exc).__name__}: {exc}'
        out['sentence'] = (SELECTION_PROVENANCE_UNREAD
                           + ' The artifact could not be read: '
                           + out['error'])
        return out
    if selection is None:
        out['sentence'] = (
            'Tuning selection: none stored in this run tree, so the DESIGN.md '
            '3.3 tuned leg has not been selected and ANALYSIS_PLAN.md 2.4 '
            'leaves every confirmatory member not-evaluable.')
        return out
    counts = dict(selection.evidence_plan_counts)
    out.update({
        'present': True,
        'short_id': selection.short_id,
        'split': bool(selection.evidence_plan_split),
        'stale': bool(selection.evidence_plan_stale(current)),
        'unknown': bool(selection.evidence_plan_unknown),
        'drift': bool(selection.evidence_plan_drift(current)),
        'hashes': sorted(counts),
        'counts': counts,
        'rows_without_a_hash': int(selection.evidence_rows_without_a_plan),
        'note': selection.evidence_plan_note(current),
    })
    if out['drift']:
        out['sentence'] = ('PRE-REGISTRATION DRIFT IN THE TUNING SELECTION: '
                           + out['note'])
    else:
        only = out['hashes'][0] if out['hashes'] else 'unrecorded'
        out['sentence'] = (
            f'Tuning selection {out["short_id"]}: its E3 evidence was produced '
            f'entirely under ANALYSIS_PLAN.md {only}, which is the plan in '
            f'force here (ANALYSIS_PLAN.md 1).')
    return out


@dataclass
class Context:
    """Everything a figure function needs, resolved once."""

    per_seed: pd.DataFrame
    curves: pd.DataFrame
    per_seed_path: str
    curves_path: Optional[str]
    outdir: str
    formats: tuple[str, ...]
    n_boot: int
    boot_seed: int
    smooth: int
    grid_points: int
    argv: list[str]
    #: 'valid' (primary) or 'pooled' (the DESIGN.md 4.3 secondary).
    source_policy: str = 'valid'
    #: What `analysis_set` removed, and why. Travels into every caption.
    analysis: dict = field(default_factory=dict)
    #: Duplicated (run_dir, episode) rows found in the curve table, if any.
    curve_integrity: dict = field(default_factory=dict)
    shift_metrics: dict[str, float] = field(default_factory=dict)
    arms: dict[str, dict[str, str]] = field(default_factory=dict)
    iface: dict[str, dict[str, str]] = field(default_factory=dict)
    shifts: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    prov: dict = field(default_factory=dict)
    hashes: dict = field(default_factory=dict)
    written: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    #: The stats.py JSON report, when one was supplied with --stats, and the
    #: DESIGN.md 3.3 arbitration read out of it. A figure whose caption is
    #: silent about the arbitration can be read as licensing exactly what the
    #: table beside it refuses, which is why ANALYSIS_PLAN.md 2.4 binds this
    #: module too. None means no stats.json was supplied, and that is reported
    #: as an unread arbitration rather than omitted.
    stats_path: Optional[str] = None
    stats_sha: Optional[str] = None
    arbitration: Optional[dict] = None
    #: The pre-registration behind the DESIGN.md 3.3 tuned leg, read off the
    #: selection artifact. On the Context for the same reason `arbitration` is:
    #: a figure travels with its caption and its sidecar and nothing else. The
    #: per_seed table cannot answer it, because E3 runs on TUNE and
    #: ANALYSIS_PLAN.md 8 keeps TUNE out of every reported estimate, so the
    #: plan hashes of the runs the tuned arms were selected from never appear
    #: in `plan_hash` here.
    selection_plan: dict = field(default_factory=dict)

    def selection_plan_sentence(self) -> str:
        return str(self.selection_plan.get('sentence')
                   or SELECTION_PROVENANCE_UNREAD)

    @property
    def selection_plan_drift(self) -> bool:
        return bool(self.selection_plan.get('drift'))
    #: One run directory per (label, seed). See `resolve_selection`.
    selected: set[str] = field(default_factory=set)
    #: (label, seed) pairs that resolved to more than one run directory.
    collisions: list[dict] = field(default_factory=list)
    #: (label, seed) pairs holding more than one ROW at ONE run directory. A
    #: different thing from a collision, with a different section of DESIGN.md
    #: behind it, and reporting the second as the first pointed a reader at the
    #: nine-conditions-one-directory defect when a row had merely been repeated.
    duplicate_rows: list[dict] = field(default_factory=list)
    _manifests: dict[str, dict] = field(default_factory=dict)
    #: The parsed stats.py report, kept beside the summary so a per-cell
    #: verdict can be looked up without re-reading the file.
    _stats_report: Optional[dict] = None

    def arbitration_sentence(self) -> str:
        """The one caption sentence, whatever state the input is in.

        A missing stats.json is not silence: `ANALYSIS_PLAN.md` 2.4 makes
        not-evaluable the default, so a figure drawn without one says that
        nothing in it may be read as a confirmatory conclusion rather than
        leaving the subject out.
        """
        if isinstance(self.arbitration, dict) and self.arbitration.get(
                'sentence'):
            return str(self.arbitration['sentence'])
        return ('DESIGN.md 3.3 arbitration: NOT READ, because no stats.json '
                'was supplied to this invocation (--stats). '
                'ANALYSIS_PLAN.md 2.4 makes not-evaluable the default, so '
                'nothing in this figure may be read as a confirmatory '
                'conclusion.')

    def arbitration_for(self, metric: str, cell: str) -> dict:
        """One member's arbitration state, or the blocking default.

        The default is what a figure gets when no stats.json was supplied, and
        it blocks: a per-cell annotation that fell silent where the verdict
        could not be read would be the figure licensing what the table refuses.
        """
        conf = (self.arbitration_source() or {}).get('s5_confirmatory') or {}
        index = arbitration_index(self.arbitration_source())
        for member in conf.get('members') or []:
            if not isinstance(member, dict):
                continue
            if (str(member.get('metric')) == str(metric)
                    and str(member.get('cell')) == str(cell)):
                return read_arbitration(member, index.get((str(metric),
                                                           str(cell))))
        return read_arbitration({}, None)

    def arbitration_source(self) -> Optional[dict]:
        return self._stats_report

    # -- label -> runs ----------------------------------------------------
    def rows_for_labels(self, labels: Iterable[str]) -> pd.DataFrame:
        # The `selected` filter is applied unconditionally, not `if
        # self.selected`. An empty selection means the analysis set is empty,
        # and a filter that switches itself off when it has nothing to keep
        # fails open: it would hand back every row in the table.
        wanted = [name for name in dict.fromkeys(labels) if name]
        rows = self.per_seed[self.per_seed['label'].isin(wanted)]
        return rows[rows['run_dir'].isin(self.selected)]

    def run_dirs_for_labels(self, labels: Iterable[str]) -> list[str]:
        return self.rows_for_labels(labels)['run_dir'].tolist()

    def intensity_for_labels(self, labels: Iterable[str]) -> dict:
        """Transfer intensity of a set of arms: which parameters were carried.

        `DESIGN.md` §3.1 names the intensity confound: two arms that differ in
        how much of the network was transferred are not comparable at fixed
        interface, and `audit.py` is required to refuse a claim that crosses
        one. `shift_gradient` puts three panels on a shared y axis whose
        transfer arms come from two experiments, and E8 declares
        `transfer_set='trunk'` where E8i declares `'matched'`, so the figure's
        own cross-panel reading crossed an unstated change in transferred
        fraction. The reading is not refused here (refusing is `audit.py`'s
        job), but it can no longer be made without seeing the change.
        """
        rows = self.rows_for_labels(labels)
        out: dict[str, Any] = {}
        if 'transfer_set' in rows.columns:
            out['transfer_set'] = sorted(
                set(str(v) for v in rows['transfer_set'].dropna()))
        if 'transferred_param_fraction' in rows.columns:
            vals = pd.to_numeric(rows['transferred_param_fraction'],
                                 errors='coerce').dropna()
            out['transferred_param_fraction'] = sorted(
                {round(float(v), 4) for v in vals})
        return out

    def seed_blocks_for(self, seeds: Iterable[int]) -> list[str]:
        """The `DESIGN.md` §3.4 blocks the plotted seeds belong to.

        Stated in every caption and every provenance record. A seed list on its
        own does not tell a reader whether a number is contaminated by
        selection: the caption used to read "n=4 distinct seeds plotted (0, 1,
        2, 200)" with nothing to say that 200 is a `TUNE` seed. `analysis_set`
        now removes those, so this line should read `CONFIRM` on a clean table,
        and anything else in it is visible rather than inferred.
        """
        if 'seed_block' not in self.per_seed.columns:
            return []
        wanted = {int(s) for s in seeds}
        rows = self.per_seed[self.per_seed['seed'].isin(wanted)]
        return sorted(set(str(v) for v in rows['seed_block'].dropna()))

    def envs_for_labels(self, labels: Iterable[str]) -> list[str]:
        if 'env' not in self.per_seed.columns:
            return []
        return list(dict.fromkeys(
            self.rows_for_labels(labels)['env'].dropna().tolist()))

    # -- manifest access, best effort ------------------------------------
    def manifest(self, run_dir: str) -> dict:
        """The manifest of one plotted run, so the caption can state the
        protocol that produced the numbers instead of restating the design."""
        if run_dir in self._manifests:
            return self._manifests[run_dir]
        found: dict = {}
        candidates = [run_dir]
        base = os.path.dirname(os.path.abspath(self.per_seed_path))
        candidates.append(os.path.join(os.path.dirname(base), run_dir))
        candidates.append(os.path.join(_ROOT, run_dir))
        for cand in candidates:
            path = os.path.join(cand, 'manifest.json')
            if os.path.isfile(path):
                try:
                    with open(path, 'r', encoding='utf-8') as fh:
                        found = json.load(fh)
                except (OSError, ValueError):
                    found = {}
                break
        self._manifests[run_dir] = found
        return found

    def protocol(self, run_dirs: Iterable[str]) -> dict:
        """Evaluation protocol of the runs in one figure, read from disk.

        Returns the *set* of distinct values found for each field, because more
        than one value means the figure mixes measurement protocols and the
        caption has to say so.
        """
        keys = ('eval_every', 'eval_episodes', 'final_eval_episodes',
                'final_eval_checkpoints')
        seen: dict[str, set] = {k: set() for k in keys}
        n_read = 0
        for run_dir in dict.fromkeys(run_dirs):
            cfg = self.manifest(run_dir).get('config') or {}
            if not cfg:
                continue
            n_read += 1
            for k in keys:
                if cfg.get(k) is not None:
                    seen[k].add(cfg[k])
        return {'manifests_read': n_read,
                **{k: sorted(v) for k, v in seen.items()}}

    def references(self, run_dirs: Iterable[str]) -> dict[str, dict]:
        """Normalisation references for the runs actually in one figure.

        Keyed on run directories rather than on environment names, and that is
        the whole point of the signature. The previous version scanned every row
        of the table with a matching `env`, so the reference printed under a
        figure could come from a run the figure does not contain, and it stopped
        at the first manifest that had a block, so any disagreement between runs
        was invisible. `DESIGN.md` §5.1 measures the random-policy reference per
        environment *and per variant*, so two runs on nominally the same env
        disagreeing about the denominator is a real possibility and a reporting
        stopper: a figure whose scores were divided by two different numbers is
        not on one scale.

        Every distinct block found among the figure's own runs is therefore
        collected, disagreement is reported rather than resolved, and the result
        is cross-checked against `reference_returns.json`, which is the file
        `audit.py` and `aggregate.py` treat as canonical.
        """
        wanted = set(str(r) for r in run_dirs)
        if 'env' not in self.per_seed.columns:
            return {}
        rows = self.per_seed[self.per_seed['run_dir'].astype(str).isin(wanted)]
        out: dict[str, dict] = {}
        for env in dict.fromkeys(rows['env'].dropna().tolist()):
            here = rows[rows['env'] == env]
            distinct: dict[tuple, dict] = {}
            n_read = 0
            for run_dir in here['run_dir'].tolist():
                block = self.manifest(run_dir).get('reference') or {}
                if block.get('random_return') is None:
                    continue
                n_read += 1
                key = (round(float(block['random_return']), 6),
                       round(float(block['threshold']), 6))
                distinct.setdefault(key, {
                    'random_return': block['random_return'],
                    'threshold': block['threshold'],
                    'noop_return': block.get('noop_return'),
                    'runs': []})['runs'].append(str(run_dir))
            canonical: dict = {}
            try:
                block = envs.reference(env)
                canonical = {'random_return': block.get('random_return'),
                             'threshold': block.get('threshold'),
                             'noop_return': block.get('noop_return')}
            except Exception:                              # noqa: BLE001
                canonical = {}
            if not distinct:
                ref = dict(canonical) if canonical else {}
                ref.update({'origin': ('reference_returns.json (no run '
                                       'manifest in this figure carried a '
                                       'reference block)')
                            if canonical else 'unavailable',
                            'manifest_blocks_read': 0,
                            'distinct_blocks': 0})
                out[str(env)] = ref
                continue
            first = next(iter(distinct.values()))
            ref = {'random_return': first['random_return'],
                   'threshold': first['threshold'],
                   'noop_return': first['noop_return'],
                   'origin': 'run manifests of the runs in this figure',
                   'manifest_blocks_read': n_read,
                   'distinct_blocks': len(distinct)}
            if len(distinct) > 1:
                ref['disagreement'] = [
                    {'random_return': v['random_return'],
                     'threshold': v['threshold'], 'runs': sorted(v['runs'])}
                    for v in distinct.values()]
            if canonical:
                agrees = (canonical.get('random_return') is not None
                          and abs(float(canonical['random_return'])
                                  - float(first['random_return'])) < 1e-6
                          and abs(float(canonical['threshold'])
                                  - float(first['threshold'])) < 1e-6)
                ref['matches_reference_returns_json'] = bool(agrees)
                if not agrees:
                    ref['reference_returns_json'] = canonical
            else:
                ref['matches_reference_returns_json'] = None
            out[str(env)] = ref
        return out


#: Columns this module indexes unconditionally, so a table lacking any of them
#: is not a per_seed table and is refused with a friendly message rather than a
#: `KeyError` traceback five call frames deep.
#:
#: `seed_block` and `source_valid` are in this list on purpose, and that is a
#: guard rather than a convenience. Both gates that keep a figure honest read
#: them: `TUNE` exclusion (`ANALYSIS_PLAN.md` §8) and the source-validity gate
#: (`DESIGN.md` §4.3). A guard written as `if 'seed_block' in df.columns` fails
#: *open*, which is how TUNE-seed leakage survived in a neighbouring module, so
#: the column is required instead of being treated as optional. `aggregate.py`
#: emits both for every run.
REQUIRED_PER_SEED_COLUMNS: tuple[str, ...] = (
    'run_dir', 'label', 'cell', 'condition', 'env', 'seed', 'seed_block',
    'source_valid', 'final_score', 'auc_score')

#: Columns a non-empty curve table must carry. `seed` is here because
#: `series_matrix` reads it to caption the seed count, and the synthesised empty
#: frame below includes it, so its absence in a real file was invisible until it
#: raised.
REQUIRED_CURVE_COLUMNS: tuple[str, ...] = ('run_dir', 'label', 'seed',
                                           'episode', 'env_steps')


def load(per_seed_path: str,
         curves_path: Optional[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_seed = pd.read_csv(per_seed_path)
    missing = [c for c in REQUIRED_PER_SEED_COLUMNS
               if c not in per_seed.columns]
    if missing:
        raise SystemExit(f'{WARN} {per_seed_path} is not a per_seed table: '
                         f'missing columns {missing}. It is produced by '
                         f'experiments/aggregate.py. seed_block and '
                         f'source_valid are required because the TUNE-seed '
                         f'exclusion (ANALYSIS_PLAN.md 8) and the '
                         f'source-validity gate (DESIGN.md 4.3) read them, and '
                         f'a gate that switches itself off when its column is '
                         f'absent is not a gate.')
    if curves_path and os.path.isfile(curves_path):
        curves = pd.read_csv(curves_path)
        if len(curves):
            missing = [c for c in REQUIRED_CURVE_COLUMNS
                       if c not in curves.columns]
            if missing:
                raise SystemExit(
                    f'{WARN} {curves_path} is not a curves table: missing '
                    f'columns {missing}. It is produced by '
                    f'experiments/aggregate.py.')
    else:
        curves = pd.DataFrame(columns=['run_dir', 'cell', 'condition', 'label',
                                       'seed', 'episode', 'env_steps',
                                       'eval_score', 'frozen'])
    return per_seed, curves


def curve_integrity(curves: pd.DataFrame) -> dict:
    """Whether `(run_dir, episode)` is unique in the curve table.

    `DESIGN.md` §8.2 names duplicated and interleaved episode rows as the
    corruption two trainers writing into one directory produces, and `B1` of
    `PRELAUNCH_FIXES.md` records that it very nearly happened in P0. `per_seed`
    carries a `metrics_contiguous` flag and this module checks it, but nothing
    checked the curve table itself, and a duplicated evaluation row is invisible
    at `--smooth 0` while silently shifting every windowed statistic once
    smoothing is on: one duplicated row moved a plotted curve's final grid mean
    by 0.04 at `--smooth 5`. Duplicates are therefore counted here, reported on
    stdout, written into every provenance record and named in the caption of
    any figure that smooths; the surviving row is the first at each
    `(run_dir, episode)` after a stable sort, so the figure is reproducible.
    """
    out = {'checked': False, 'duplicate_rows': 0, 'runs_affected': []}
    if curves.empty or not {'run_dir', 'episode'} <= set(curves.columns):
        return out
    out['checked'] = True
    dupe = curves.duplicated(subset=['run_dir', 'episode'], keep=False)
    if not bool(dupe.any()):
        return out
    affected = curves.loc[dupe, 'run_dir'].astype(str)
    out['duplicate_rows'] = int(
        curves.duplicated(subset=['run_dir', 'episode'], keep='first').sum())
    out['runs_affected'] = sorted(set(affected.tolist()))
    return out


def seeds_per_run(curves: pd.DataFrame) -> dict[str, list[int]]:
    """Run directories in the curve table that carry more than one seed.

    `series_matrix` attributes a whole run's curve to `g['seed'].iloc[0]`, so a
    run mapping to two seeds would make the caption's seed set quietly wrong.
    Nothing checked it; this does, and the caller reports it.
    """
    if curves.empty or not {'run_dir', 'seed'} <= set(curves.columns):
        return {}
    out: dict[str, list[int]] = {}
    for run_dir, g in curves.groupby('run_dir'):
        found = sorted({int(v) for v in
                        pd.to_numeric(g['seed'], errors='coerce').dropna()})
        if len(found) > 1:
            out[str(run_dir)] = found
    return out


# ---------------------------------------------------------------------------
# 4. Selection and pairing
# ---------------------------------------------------------------------------
def arm_rows(ctx: Context, label: str) -> pd.DataFrame:
    """One arm's runs, from the analysis set, one run per seed.

    The `selected` filter is unconditional for the reason given in
    `Context.rows_for_labels`: a filter that switches itself off when it has
    nothing to keep fails open.
    """
    rows = ctx.per_seed[ctx.per_seed['label'] == label]
    return rows[rows['run_dir'].isin(ctx.selected)]


@dataclass
class Paired:
    """One within-cell contrast at matched seeds, and what it had to drop."""

    seeds: list[int]
    base: np.ndarray
    treat: np.ndarray
    delta: np.ndarray
    base_label: str
    treat_label: str
    metric: str
    unmatched_base: list[int] = field(default_factory=list)
    unmatched_treat: list[int] = field(default_factory=list)
    duplicates: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.seeds)


def _rows_to_series(rows: pd.DataFrame, metric: str, name: str,
                    duplicates: list[str]) -> pd.Series:
    """seed -> metric for a set of runs, one value per seed.

    De-duplication is by seed within whatever rows the caller passes, by
    `one_row_per_seed`: sorted by path and then by row CONTENT, so the survivor
    is a function of the set of rows and not of their order in the file. The
    previous version sorted on the path alone with a stable sort, which for two
    rows sharing a `run_dir` is a no-op, so "sorted by path so the choice is
    reproducible" was false exactly where it mattered: reversing two colliding
    rows in the CSV flipped the sign of two confirmatory-family shifts.

    Callers that vary a factor pass rows already restricted to one level of it,
    so a factor level is never collapsed away here. `analysis_set` has already
    removed duplicated (label, seed) rows from `ctx.per_seed`, so on that path
    this is the second belt; it is still applied, because `treat_rows=` callers
    hand in rows they selected themselves.
    """
    if metric not in rows.columns:
        return pd.Series(dtype=float)
    rows = rows.dropna(subset=[metric])
    rows = one_row_per_seed(rows, name, duplicates)
    return rows.set_index('seed')[metric].astype(float)


def _seed_series(ctx: Context, label: str, metric: str,
                 duplicates: list[str]) -> pd.Series:
    return _rows_to_series(arm_rows(ctx, label), metric, label, duplicates)


def pair(ctx: Context, base_label: str, treat_label: str, metric: str,
         base_rows: Optional[pd.DataFrame] = None,
         treat_rows: Optional[pd.DataFrame] = None) -> Paired:
    """Matched-seed contrast. Seeds are a blocking factor (`DESIGN.md` §2.4
    RQ2), so an unmatched seed cannot enter the delta -- and it is recorded
    rather than dropped quietly, because silent seed dropping is one of the six
    published defects (`DESIGN.md` §1).

    `treat_rows` lets a caller that is varying a factor supply the runs itself,
    restricted to one level. Selecting by label alone would be wrong there: two
    levels of a factor can share an arm label, and pairing on the label would
    then hand back the same runs for both levels and draw a flat line.
    """
    dupes: list[str] = []
    b = (_rows_to_series(base_rows, metric, base_label, dupes)
         if base_rows is not None
         else _seed_series(ctx, base_label, metric, dupes))
    t = (_rows_to_series(treat_rows, metric, treat_label, dupes)
         if treat_rows is not None
         else _seed_series(ctx, treat_label, metric, dupes))
    common = sorted(set(b.index) & set(t.index))
    return Paired(
        seeds=common,
        base=b.reindex(common).to_numpy(float),
        treat=t.reindex(common).to_numpy(float),
        delta=(t.reindex(common) - b.reindex(common)).to_numpy(float),
        base_label=base_label, treat_label=treat_label, metric=metric,
        unmatched_base=sorted(set(b.index) - set(common)),
        unmatched_treat=sorted(set(t.index) - set(common)),
        duplicates=sorted(set(dupes)))


def _note_losses(lost: dict[str, list[str]], where: str, metric: str,
                 p: Paired) -> None:
    """Record what one matched-seed contrast had to leave out.

    `Paired` has always carried `unmatched_base`, `unmatched_treat` and
    `duplicates`, and exactly one figure read them. Every other paired figure
    threw them away, so an arm silently falling from n=3 to n=2 changed the
    interaction contrast in all four cells and left no trace in the figure, the
    caption or the provenance. `DESIGN.md` §1 lists silent seed dropping as one
    of the six published defects; recording it in one place makes it a fact
    every figure carries rather than a courtesy one figure extended.
    """
    if p.unmatched_base:
        lost['unmatched seeds'].append(
            f'{where}/{metric} {p.base_label} seeds '
            f'{_seed_list(p.unmatched_base)}')
    if p.unmatched_treat:
        lost['unmatched seeds'].append(
            f'{where}/{metric} {p.treat_label} seeds '
            f'{_seed_list(p.unmatched_treat)}')
    if p.duplicates:
        lost['duplicated arm rows'].append(
            f"{where}/{metric} {', '.join(p.duplicates)}")


def estimate_shift(ctx: Context, delta: np.ndarray,
                   idx: Optional[np.ndarray] = None) -> dict:
    """Hodges-Lehmann paired shift with `stats.py`'s bias-corrected bootstrap
    interval. `ANALYSIS_PLAN.md` §2: no mean-with-a-normal-interval anywhere.

    `vec=stats.hl_vec` is not an optimisation detail. Without it the Walsh
    average median is recomputed in Python once per bootstrap resample, which is
    10,000 scalar passes per contrast and several dozen contrasts per
    invocation: the forest alone took 36 s at n=3 and would take minutes at the
    confirmatory n=10, which is long enough that a figure stops being re-run
    after a change. `hl_vec` computes the whole stack at once and `stats.py`'s
    `--self-test` checks the two agree.
    """
    units = np.asarray(delta, dtype=float).reshape(-1, 1)
    if units.shape[0] == 0:
        return {'estimate': float('nan'), 'lo': float('nan'),
                'hi': float('nan'), 'n': 0, 'method': 'none', 'note': 'no data'}
    out = stats.bootstrap_statistic(
        units, lambda u: stats.hodges_lehmann_paired(u[:, 0]),
        n_boot=ctx.n_boot, seed=ctx.boot_seed, idx=idx,
        vec=lambda s: stats.hl_vec(s[..., 0]))
    out.pop('reps', None)
    return out


def estimate_mean(ctx: Context, units: np.ndarray, column: int,
                  idx: Optional[np.ndarray] = None) -> dict:
    """Seed mean of one column of a joint unit matrix, with a bootstrap
    interval from the *shared* resampling in `idx`."""
    units = np.asarray(units, dtype=float)
    if units.shape[0] == 0:
        return {'estimate': float('nan'), 'lo': float('nan'),
                'hi': float('nan'), 'n': 0, 'method': 'none'}
    out = stats.bootstrap_statistic(
        units, lambda u: float(np.mean(u[:, column])),
        n_boot=ctx.n_boot, seed=ctx.boot_seed, idx=idx)
    out.pop('reps', None)
    return out


def estimate_difference(ctx: Context, units: np.ndarray, a: int, b: int,
                        idx: Optional[np.ndarray] = None) -> dict:
    """Paired mean difference of columns a - b under the shared resampling."""
    units = np.asarray(units, dtype=float)
    if units.shape[0] == 0:
        return {'estimate': float('nan'), 'lo': float('nan'),
                'hi': float('nan'), 'n': 0, 'method': 'none'}
    out = stats.bootstrap_statistic(
        units, lambda u: float(np.mean(u[:, a] - u[:, b])),
        n_boot=ctx.n_boot, seed=ctx.boot_seed, idx=idx)
    out.pop('reps', None)
    return out


def _degenerate(est: dict) -> bool:
    """Whether an interval has collapsed to a point.

    A zero-width bootstrap interval means every resample produced the same
    value, which happens when the sample itself is constant: a floored metric, a
    duplicated row, or an arm where every seed landed on the same number. It is
    the *least* informative case, and both `bca_interval` (which falls back to a
    percentile interval there) and the sentences below have to treat it as an
    absence of information rather than as an infinitely precise measurement. The
    unguarded version read `lo = hi = 0.0` as "a degradation of any size is
    excluded" and "equivalence supported", which is the strongest possible pair
    of claims from the weakest possible evidence.
    """
    if est.get('degenerate'):
        return True                  # stats.py's own flag, where it sets one
    lo, hi = est.get('lo'), est.get('hi')
    if lo is None or hi is None:
        return False
    # Measured from the interval itself as well, so the guard does not depend
    # on an upstream flag being present in every estimator's return value.
    return bool(np.isfinite(lo) and np.isfinite(hi) and (hi - lo) <= 0.0)


def exclusion_sentence(est: dict, units: str = 'normalised score units') -> str:
    """The only licensed positive statement about a null (`DESIGN.md` §9,
    `ANALYSIS_PLAN.md` §4): what the interval excludes.

    `units` is a parameter because the bound is quoted on whatever scale the
    endpoint lives on, and calling an area-per-env-step quantity "score units"
    would misdescribe P2 in the caption of the figure that draws it.
    """
    lo = est.get('lo')
    if lo is None or not np.isfinite(lo):
        return 'no exclusion bound (interval suppressed)'
    if _degenerate(est):
        return ('no exclusion bound: the bootstrap interval has zero width, so '
                'the sample is constant and the interval carries no '
                'information about what is excluded')
    if lo >= 0:
        return (f'a degradation worse than 0.000 {units} is excluded at 95% '
                f'(the interval lies entirely at or above zero)')
    return (f'a degradation worse than {abs(lo):.3f} {units} is '
            f'excluded at 95%')


def equivalence_sentence(est: dict, sd: float, n: Optional[int] = None,
                         endpoint: str = 'final_score',
                         margin: float = stats.EQUIVALENCE_MARGIN) -> str:
    """`ANALYSIS_PLAN.md` §4 in its branches: an equivalence verdict, or the
    statement that this cell cannot support one. Never TOST, and never a null
    read as equivalence.

    Four refusals come before the verdict, and each closes a way of getting the
    strongest available claim out of the weakest available evidence:

    * **Endpoint.** §4 fixes +/-0.05 for the paired delta on the *normalised
      score*, justified as ~20 return points on LunarLander. That justification
      does not extend to `auc_score`, and inventing a margin for an endpoint the
      plan never defined one for is a new analysis choice.
    * **Sample size.** Below `MIN_N_FOR_INFERENCE` there is no interval to be
      inside anything (`ANALYSIS_PLAN.md` §9).
    * **Zero dispersion.** The feasibility gate is `sd > margin`, which at
      SD = 0 passes trivially, so a degenerate arm sailed straight through it to
      "equivalence supported".
    * **A degenerate interval.** Same reason, at the other end.
    """
    if endpoint not in MARGIN_ENDPOINTS:
        return (f'no equivalence verdict for {endpoint}: ANALYSIS_PLAN.md 4 '
                f'fixes the +/-{margin:.2f} margin for the paired delta on the '
                f'normalised score only, and no margin is pre-registered on '
                f'this scale')
    if n is not None and n < stats.MIN_N_FOR_INFERENCE:
        return (f'equivalence not assessed (n={n} < '
                f'{stats.MIN_N_FOR_INFERENCE}, ANALYSIS_PLAN.md 9)')
    if not np.isfinite(sd):
        return ('equivalence not assessed (across-seed SD undefined at this n)')
    if sd <= 0.0:
        return ('equivalence not assessed: the across-seed SD is exactly 0, so '
                'this arm is constant across seeds (a floored metric, a '
                'duplicated row, or a single distinct value) and carries no '
                'dispersion information to test against the margin')
    if sd > margin:
        return (f'equivalence untestable in this cell at this n '
                f'(across-seed SD {sd:.3f} > margin {margin:.2f})')
    lo, hi = est.get('lo'), est.get('hi')
    if lo is None or not np.isfinite(lo) or not np.isfinite(hi):
        return 'equivalence not assessed (interval suppressed)'
    if _degenerate(est):
        return ('equivalence not assessed: the bootstrap interval has zero '
                'width, and a point interval from a constant sample is an '
                'absence of information, not evidence of equivalence')
    if lo > -margin and hi < margin:
        return f'CI inside +/-{margin:.2f}: equivalence supported'
    return f'CI not inside +/-{margin:.2f}: equivalence not supported'


class MethodLog:
    """Every interval method an estimator actually returned inside one figure.

    Reviewer concern C8 is a caption that cannot be checked against its figure,
    and the `interval=` strings in this module were hand-written constants: they
    said "bias-corrected (BCa) seed-level bootstrap 95% CI" whatever
    `bca_interval` in fact returned. On a degenerate bootstrap distribution it
    returns a *percentile* interval and says so in a `note`, and that note was
    carried in the estimate dict and then thrown away. The same string also
    asserted globally that "no interval is drawn" whenever the smallest arm was
    below the inference floor, while `band()` suppresses per arm, so seven of
    eight arms could carry a band under a caption saying none did.

    So the methods are counted as they arrive and the caption states the tally.
    A figure cannot contradict itself about its own intervals if the sentence is
    generated from what happened.
    """

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}
        self.notes: dict[str, None] = {}

    def add(self, est: dict) -> dict:
        """Record one estimate's method and note; returns it for chaining."""
        method = str(est.get('method') or 'none')
        self.counts[method] = self.counts.get(method, 0) + 1
        note = str(est.get('note') or '').strip()
        if note:
            self.notes.setdefault(note, None)
        return est

    def add_method(self, method: str, note: str = '') -> None:
        self.add({'method': method, 'note': note})

    @property
    def drawn(self) -> int:
        return sum(v for k, v in self.counts.items()
                   if not k.startswith(('suppressed', 'none')))

    @property
    def suppressed(self) -> int:
        return sum(v for k, v in self.counts.items()
                   if k.startswith(('suppressed', 'none')))

    def sentence(self) -> str:
        if not self.counts:
            return ('No interval was computed anywhere in this figure: no arm '
                    'produced an estimate.')
        tally = ', '.join(f'{k} x{v}' for k, v in sorted(self.counts.items()))
        out = (f'Interval methods the estimator actually returned in this '
               f'figure: {tally} ({self.drawn} drawn, {self.suppressed} '
               f'suppressed).')
        if self.notes:
            out += ' Estimator notes: ' + ' | '.join(self.notes) + '.'
        return out

    def record(self) -> dict:
        return {'counts': dict(sorted(self.counts.items())),
                'drawn': self.drawn, 'suppressed': self.suppressed,
                'notes': sorted(self.notes)}


# ---------------------------------------------------------------------------
# 5. Curve machinery -- common env-step support, bootstrap bands, the freeze
#    boundary
# ---------------------------------------------------------------------------
def curve_rows(ctx: Context, label: str) -> pd.DataFrame:
    """The episode rows of one arm, restricted to the canonical run per seed.

    The restriction matters: without it a label carrying two configurations
    (see `resolve_selection`) would have both averaged into one band, and a
    band over two freeze windows is not a curve of either.

    The restriction is also what carries the analysis set into the curve
    figures: `ctx.selected` is built from the filtered `per_seed`, so a run
    excluded for a TUNE seed, a donor-only block or a failed source-validity
    gate has no curve drawn either. It is applied unconditionally for the reason
    in `Context.rows_for_labels`: an empty selection must yield no rows, not
    every row.
    """
    if ctx.curves.empty or 'label' not in ctx.curves.columns:
        return ctx.curves
    rows = ctx.curves[ctx.curves['label'] == label]
    if 'run_dir' not in rows.columns:
        return rows.iloc[0:0]
    return rows[rows['run_dir'].isin(ctx.selected)]


def _support(frames: Sequence[pd.DataFrame], column: str) -> tuple[float, float,
                                                                   int]:
    """The env-step interval every plotted run covers, for `column`.

    Curves are averaged on a common grid inside the *intersection* of the runs'
    env-step ranges, so no value on any band is an extrapolation and the number
    of seeds behind every point of the band is the same. A band whose n changes
    along x is the kind of artefact that reads as a narrowing of uncertainty.
    """
    los, his, runs = [], [], 0
    for frame in frames:
        if frame.empty or column not in frame.columns:
            continue
        for _run, g in frame.dropna(subset=[column]).groupby('run_dir'):
            x = g['env_steps'].to_numpy(float)
            if len(x) < 2:
                continue
            los.append(float(np.min(x)))
            his.append(float(np.max(x)))
            runs += 1
    if runs == 0:
        return (float('nan'), float('nan'), 0)
    return (max(los), min(his), runs)


def series_matrix(ctx: Context, frame: pd.DataFrame, column: str,
                  grid: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """One row per seed, interpolated onto `grid`. No extrapolation.

    Duplicated `(run_dir, episode)` rows are dropped here, deterministically and
    after a stable sort, because a duplicated evaluation point enters the
    trailing mean twice and shifts a windowed statistic without changing
    anything visible at `--smooth 0`. The count is reported by
    `curve_integrity` at load time and named in the caption; this is where the
    corruption is actually kept out of the numbers.

    A run with no usable `seed` no longer raises: it is skipped and the caption
    then rests on the seeds that were readable, which the seed count states.
    """
    rows, seeds = [], []
    if frame.empty or column not in frame.columns:
        return np.zeros((0, len(grid))), seeds
    frame = frame.dropna(subset=[column])
    if 'episode' in frame.columns:
        frame = (frame.sort_values(['run_dir', 'episode'], kind='mergesort')
                 .drop_duplicates(subset=['run_dir', 'episode'], keep='first'))
    for run_dir, g in frame.groupby('run_dir'):
        g = g.sort_values('env_steps', kind='mergesort')
        x = g['env_steps'].to_numpy(float)
        y = g[column].to_numpy(float)
        if len(x) < 2:
            continue
        seed = pd.to_numeric(g['seed'], errors='coerce').dropna() \
            if 'seed' in g.columns else pd.Series(dtype=float)
        if seed.empty:
            print(f'{WARN} series_matrix: no readable seed for run {run_dir}; '
                  f'its curve is not plotted, because a curve with no seed '
                  f'cannot be counted in the seed set the caption states')
            continue
        y = _trailing_mean(y, ctx.smooth)
        rows.append(np.interp(grid, x, y, left=np.nan, right=np.nan))
        seeds.append(int(seed.iloc[0]))
    if not rows:
        return np.zeros((0, len(grid))), seeds
    return np.vstack(rows), seeds


def band(ctx: Context, mat: np.ndarray) -> dict:
    """Mean curve across seeds with a percentile bootstrap 95% band.

    Percentile rather than BCa here, and the caption says so: BCa's
    acceleration term is estimated from a jackknife at every one of hundreds of
    grid points, which is a lot of machinery for a visual envelope. The scalar
    estimands -- the ones a claim is made about -- carry `stats.py`'s
    bias-corrected interval instead.
    """
    n = mat.shape[0]
    if n == 0:
        return {'n': 0, 'mean': None, 'lo': None, 'hi': None,
                'method': 'none'}
    mean = np.nanmean(mat, axis=0)
    if ctx.n_boot < 1:
        # Zero resamples produced an empty replicate array, `np.nanpercentile`
        # returned NaN bounds, no band was drawn, and the caption still read
        # "95% percentile bootstrap over seeds, 0 resamples". The CLI now
        # refuses n_boot < 1; this is the second belt, and it names the reason.
        return {'n': n, 'mean': mean, 'lo': None, 'hi': None,
                'method': 'suppressed (no bootstrap resamples requested)'}
    if n < stats.MIN_N_FOR_INFERENCE:
        return {'n': n, 'mean': mean, 'lo': None, 'hi': None,
                'method': f'suppressed (n={n} < {stats.MIN_N_FOR_INFERENCE})'}
    idx = stats.boot_indices(n, ctx.n_boot, ctx.boot_seed)
    reps = np.empty((ctx.n_boot, mat.shape[1]), dtype=float)
    step = 500
    for start in range(0, ctx.n_boot, step):
        block = idx[start:start + step]
        reps[start:start + len(block)] = np.nanmean(mat[block], axis=1)
    return {'n': n, 'mean': mean,
            'lo': np.nanpercentile(reps, 2.5, axis=0),
            'hi': np.nanpercentile(reps, 97.5, axis=0),
            'method': f'percentile bootstrap over seeds, {ctx.n_boot} resamples'}


def freeze_boundary(ctx: Context, frame: pd.DataFrame) -> dict:
    """Where the freeze window ended, in env steps, per run.

    The schedule is indexed in gradient updates (`DESIGN.md` §3.2) while the
    x-axis is env steps, so the boundary is not a fixed x: it is located per
    run at the first episode whose `frozen` flag is false after a frozen
    prefix, and the mean of those is drawn. A run that never leaves the window
    within its budget contributes nothing and is counted, so the caption can
    say the boundary is absent rather than the figure implying there was none.
    """
    out = {'runs': 0, 'exited': 0, 'mean_env_step': float('nan'),
           'sd_env_step': float('nan'), 'ever_frozen': 0,
           'max_updates': float('nan'), 'unreadable_frozen_flags': 0}
    if frame.empty or 'frozen' not in frame.columns:
        return out
    steps, updates_max = [], []
    for _run, g in frame.groupby('run_dir'):
        g = g.sort_values('episode', kind='mergesort')
        flag, known = _bool_tokens(g['frozen'])
        # An unreadable freeze flag is counted rather than assumed. It reads as
        # "not frozen", which can only move the boundary earlier or remove it,
        # never invent one, and the count travels into the provenance so a
        # boundary resting on guesswork is visible.
        out['unreadable_frozen_flags'] += int((~known).sum())
        out['runs'] += 1
        if 'updates' in g.columns and len(g):
            updates_max.append(float(g['updates'].max()))
        if not flag.any():
            continue
        out['ever_frozen'] += 1
        unfrozen = np.flatnonzero(~flag)
        if len(unfrozen) == 0:
            continue
        first = int(unfrozen[0])
        if first == 0:
            continue                     # never frozen at episode 0
        steps.append(float(g['env_steps'].to_numpy(float)[first]))
    out['exited'] = len(steps)
    if steps:
        out['mean_env_step'] = float(np.mean(steps))
        out['sd_env_step'] = (float(np.std(steps, ddof=1))
                              if len(steps) > 1 else 0.0)
    if updates_max:
        out['max_updates'] = float(np.max(updates_max))
    return out


def draw_boundary(ax: plt.Axes, boundary: dict, label: bool = False) -> bool:
    if boundary.get('exited', 0) <= 0 or not np.isfinite(
            boundary.get('mean_env_step', float('nan'))):
        return False
    ax.axvline(boundary['mean_env_step'], color=_GREY, linewidth=0.8,
               dashes=(3, 2), zorder=1)
    if label:
        ax.annotate('freeze window ends', xy=(boundary['mean_env_step'], 1.0),
                    xycoords=('data', 'axes fraction'),
                    xytext=(2, -6), textcoords='offset points',
                    fontsize=6.2, color=_GREY, rotation=90, va='top')
    return True


# ---------------------------------------------------------------------------
# 6. Emission -- the figure, its generated caption, its provenance
# ---------------------------------------------------------------------------
def stamp_validation(fig: plt.Figure, n: Optional[int]) -> None:
    """Mark the figure as machinery evidence, not result evidence.

    Faint on purpose: the stamp has to be unmissable in a slide deck and still
    leave the data readable, because the whole use of an n<3 figure is checking
    that the pipeline produced the right shape of curve.

    `n` may be None, meaning the figure found no data at all. That case used to
    escape the stamp entirely, because the caller stamped only `if n_min is not
    None`: an empty figure came out looking like an ordinary methods figure with
    an ordinary caption, which is the stamping rule inverted for exactly the
    case that most needs it.
    """
    size = max(16.0, min(30.0, 4.2 * fig.get_figwidth()))
    fig.text(0.5, 0.5, stats.VALIDATION_STAMP.split(' - ')[0],
             fontsize=size, color='#B4B4B4', alpha=0.30, ha='center',
             va='center', rotation=22, zorder=0,
             transform=fig.transFigure)
    detail = (f'n={n} seeds: ' if n is not None
              else 'NO DATA in the analysis set: ')
    fig.text(0.5, 0.5 - 0.055, detail + stats.VALIDATION_STAMP,
             fontsize=6.6, color='#A8A8A8', alpha=0.75, ha='center',
             va='center', rotation=22, zorder=0, transform=fig.transFigure)


def stamp_pooled(fig: plt.Figure) -> None:
    """Say on the canvas that this is the secondary analysis set.

    A pooled figure separated from its caption (dropped into a slide, pasted
    into a draft) would otherwise be indistinguishable from the primary one,
    and the difference between them is the whole of `DESIGN.md` §4.3.
    """
    fig.text(0.5, 0.985, 'SECONDARY ANALYSIS SET: POOLED OVER SOURCE '
                         'COMPETENCE (DESIGN.md 4.3), NOT THE PRIMARY '
                         'ESTIMAND',
             fontsize=6.0, color='#8A6D3B', alpha=0.95, ha='center', va='top',
             zorder=5, transform=fig.transFigure)


def protocol_sentence(protocol: dict) -> str:
    """The evaluation protocol, from the manifests of the plotted runs."""
    if not protocol.get('manifests_read'):
        return ('evaluation protocol not verifiable -- no run manifest was '
                'readable from the paths in the per-seed table, so the '
                'protocol is NOT restated here')

    def one(key: str, unit: str) -> str:
        vals = protocol.get(key) or []
        if not vals:
            return f'{key} unknown'
        if len(vals) == 1:
            return f'{vals[0]} {unit}'
        return f'MIXED {vals} {unit}'

    return (f"monitoring evaluation every {one('eval_every', 'episodes')} of "
            f"{one('eval_episodes', 'episodes')} on a separate environment "
            f"instance; final_score is the held-out greedy mean over "
            f"{one('final_eval_episodes', 'episodes')} at each of the final "
            f"{one('final_eval_checkpoints', 'checkpoints')}, averaged "
            f"(read from {protocol['manifests_read']} run manifests)")


def normalisation_sentence(refs: dict[str, dict]) -> str:
    if not refs:
        return ('normalised score = (return - random_return) / (threshold - '
                'random_return); no reference block was readable')
    parts = []
    warnings = []
    for env, ref in refs.items():
        parts.append(f"{env}: random {_f(ref.get('random_return'), 1)}, "
                     f"threshold {_f(ref.get('threshold'), 1)}, no-op "
                     f"{_f(ref.get('noop_return'), 1)} [{ref.get('origin')}]")
        if ref.get('distinct_blocks', 0) > 1:
            warnings.append(
                f"WARNING: the runs in this figure on {env} do NOT share one "
                f"normalisation reference ({ref['distinct_blocks']} distinct "
                f"blocks). Their scores are not on one scale and no contrast "
                f"between them is interpretable; the details are in the "
                f"provenance record.")
        if ref.get('matches_reference_returns_json') is False:
            warnings.append(
                f"WARNING: the reference the {env} runs were normalised by "
                f"disagrees with reference_returns.json, which audit.py treats "
                f"as canonical.")
    return ('normalised score = (return - random_return) / (threshold - '
            'random_return), so a uniform-random policy scores 0 and the '
            'registered threshold scores 1 (DESIGN.md 5.1) -- '
            + '; '.join(parts)
            + ((' ' + ' '.join(warnings)) if warnings else ''))


def smoothing_sentence(ctx: Context, extra: str = '') -> str:
    if ctx.smooth <= 1:
        return 'smoothing window: none (raw evaluation points)' + extra
    return (f'smoothing window: trailing mean over {ctx.smooth} evaluation '
            f'points, applied per run before averaging across seeds' + extra)


def n_is_a_seed_count(n_min: Optional[int], seeds: Sequence[int]) -> bool:
    """The invariant every figure's n has to satisfy, as a testable predicate.

    `n_min` is the SMALLEST per-arm count and `seeds` is the union of the seeds
    the figure plotted, so `n_min <= len(seeds)` holds for any honest pair. It
    failed in the Kaplan-Meier panels, which counted rows: n_min=4 against two
    distinct seeds. Written as a function rather than inline in `emit` so
    `--self-test` can put both a good and a bad pair in front of it; a guard
    that can only be exercised by rendering a figure is a guard nobody checks.
    """
    return not (n_min is not None and len(seeds) > 0
                and int(n_min) > len(seeds))


def emit(ctx: Context, name: str, fig: plt.Figure, body: str, *,
         seeds: Sequence[int] = (), n_min: Optional[int] = None,
         protocol: Optional[dict] = None, refs: Optional[dict] = None,
         interval: str = '', smoothing: Optional[str] = None,
         extra_lines: Sequence[str] = (),
         meta: Optional[dict] = None,
         methods: Optional[MethodLog] = None,
         dropped: Optional[dict] = None) -> None:
    """Write the figure, its generated caption and its provenance record.

    `methods` is the log of interval methods the estimators actually returned,
    so the caption states them instead of asserting a constant. `dropped` is
    what this particular figure lost that the analysis-set filters did not
    remove: unmatched seeds, duplicated arm rows, censored runs with no usable
    time. Recording it per figure closes the asymmetry where only the forest
    kept `Paired`'s bookkeeping and the other five paired figures discarded it,
    so an arm quietly falling from n=3 to n=2 left no trace at all.
    """
    os.makedirs(ctx.outdir, exist_ok=True)
    seeds = list(seeds)
    # The stamp fires when there is too little data OR none at all. `n_min is
    # None` means no arm produced an estimate, which is further below the
    # inference floor than n=1, not above it.
    stamped = n_min is None or n_min < stats.MIN_N_FOR_INFERENCE
    # One invariant across every figure: `n_min` is the SMALLEST per-arm seed
    # count and `seeds` is the union of the seeds plotted, so n_min can never
    # exceed len(seeds). It did in the Kaplan-Meier panels, because that figure
    # counted rows: n_min=4 against two distinct seeds, which is exactly how
    # the stamp came off a caption that still said "n=2 distinct seeds
    # plotted". Checked here rather than in one figure, so no figure can
    # reintroduce it quietly, and a violation forces the stamp on rather than
    # being reported and ignored.
    n_leak = not n_is_a_seed_count(n_min, seeds)
    if n_leak:
        print(f'{WARN} {name}: n_min={n_min} exceeds the {len(seeds)} distinct '
              f'seed(s) plotted, so some arm counted ROWS as observations. '
              f'Every interval scaled by that n is too narrow. The figure is '
              f'stamped {stats.VALIDATION_STAMP} whatever n_min says.')
        stamped = True
    if stamped:
        stamp_validation(fig, n_min)
    if ctx.source_policy != 'valid':
        stamp_pooled(fig)
    paths = []
    for fmt in ctx.formats:
        path = os.path.join(ctx.outdir, f'{name}.{fmt}')
        fig.savefig(path, format=fmt)
        paths.append(path)
    plt.close(fig)

    lines = [textwrap.fill(body, 92), '']
    lines.append(textwrap.fill(analysis_set_sentence(ctx.analysis), 92))
    if seeds:
        blocks = ctx.seed_blocks_for(seeds)
        lines.append(textwrap.fill(
            f'Seeds: n={len(seeds)} distinct seeds plotted '
            f'({_seed_list(seeds)}), from seed block(s) '
            f'{", ".join(blocks) or "unknown"}. '
            + ('Below the confirmatory floor of ten seeds '
               '(STANDING_INSTRUCTIONS.md S4): estimates only, and no claim in '
               'the paper may rest on this figure.'
               if len(seeds) < 10 else
               'At or above the confirmatory floor of ten seeds '
               '(STANDING_INSTRUCTIONS.md S4).'), 92))
    else:
        lines.append(textwrap.fill(
            'Seeds: none. No arm in this figure had a run in the analysis set, '
            'so nothing is plotted and there is no estimate of any kind here.',
            92))
    if n_min is None:
        lines.append(textwrap.fill(
            f'{stats.VALIDATION_STAMP}: this figure contains NO DATA. No arm '
            f'produced an estimate, so nothing here may be quoted, compared or '
            f'used to choose between hypotheses (ANALYSIS_PLAN.md 9). An empty '
            f'figure is evidence about the pipeline, never about the study.',
            92))
    elif n_min < stats.MIN_N_FOR_INFERENCE:
        lines.append(textwrap.fill(
            f'{stats.VALIDATION_STAMP}: the smallest arm has n={n_min} < '
            f'{stats.MIN_N_FOR_INFERENCE}, so no number here may be quoted, '
            f'compared or used to choose between hypotheses '
            f'(ANALYSIS_PLAN.md 9). Intervals are suppressed per arm, not '
            f'globally: which arms carry one is in the interval line below and '
            f'in the provenance record.', 92))
    if protocol is not None:
        lines.append(textwrap.fill('Evaluation: ' + protocol_sentence(protocol),
                                   92))
    lines.append(textwrap.fill('Interval: ' + (interval or 'none drawn'), 92))
    if methods is not None:
        lines.append(textwrap.fill(methods.sentence(), 92))
    losses = '; '.join(f'{k}: {v}' for k, v in sorted((dropped or {}).items())
                       if v)
    if losses:
        lines.append(textwrap.fill(
            'Runs this figure could not use, beyond the analysis-set filters '
            'above. Each is a seed that exists but does not enter the estimate '
            'beside it, so the n on the axis and the n in the table can '
            'differ, and neither is silent: ' + losses + '.', 92))
    if ctx.curve_integrity.get('duplicate_rows') and (
            smoothing is None and ctx.smooth > 1):
        lines.append(textwrap.fill(
            f"INPUT INTEGRITY: the curve table contains "
            f"{ctx.curve_integrity['duplicate_rows']} duplicated "
            f"(run_dir, episode) row(s) across "
            f"{len(ctx.curve_integrity['runs_affected'])} run(s). They are "
            f"dropped deterministically before smoothing, because a duplicated "
            f"evaluation point enters the trailing mean twice; a table needing "
            f"that repair has not passed DESIGN.md 8.2 and no window statistic "
            f"from it is trustworthy.", 92))
    lines.append(textwrap.fill(smoothing if smoothing is not None
                               else smoothing_sentence(ctx), 92))
    if refs is not None:
        lines.append(textwrap.fill('Normalisation: '
                                   + normalisation_sentence(refs), 92))
    for line in extra_lines:
        lines.append(textwrap.fill(line, 92))
    lines.append(textwrap.fill(
        'Inference: this figure draws no p-value. The confirmatory family is '
        'the 8 tests of ANALYSIS_PLAN.md 2 (4 within-cell deltas x 2 co-primary '
        'endpoints) under Holm-Bonferroni, reported by stats.py; every quantity '
        'drawn here is estimation-only.', 92))
    # In EVERY caption, not only the forest's. ANALYSIS_PLAN.md 2.4 binds this
    # module as well, and a figure separated from its bundle carries only its
    # caption: an interval drawn beside a cell whose arbitration blocks must
    # not be readable as the conclusion the table refuses.
    lines.append(textwrap.fill(
        'Assertion: ' + ctx.arbitration_sentence()
        + ' Nothing in this figure is a confirmatory conclusion, and an '
          'interval drawn here licenses one only where that verdict says it '
          'does.', 92))
    for item in (ctx.arbitration or {}).get('disagreements') or []:
        lines.append(textwrap.fill(
            'POLICY DISAGREEMENT, and under ANALYSIS_PLAN.md 2.4 the '
            'disagreement is itself the reported finding rather than a caveat '
            'on one: ' + str(item), 92))
    if ctx.selection_plan_drift:
        lines.append(textwrap.fill(
            'PRE-REGISTRATION DRIFT IN THE TUNING SELECTION: '
            + str(ctx.selection_plan.get('note')
                  or ctx.selection_plan_sentence())
            + ' Anything in this figure that the arbitration licensed from the '
              'tuned leg is EXPLORATORY on that ground (ANALYSIS_PLAN.md 1).',
            92))
    git = (ctx.prov.get('git') or {})
    lines.append(textwrap.fill(
        f"Source: {os.path.basename(ctx.per_seed_path)} "
        f"[{ctx.hashes.get('per_seed')}]"
        + (f", {os.path.basename(ctx.curves_path)} "
           f"[{ctx.hashes.get('curves')}]" if ctx.curves_path else '')
        + f"; git {str(git.get('commit'))[:12]}"
        + (' (dirty working tree)' if git.get('dirty') else '')
        + f"; ANALYSIS_PLAN.md "
          f"{(ctx.prov.get('plans') or {}).get('ANALYSIS_PLAN.md')}", 92))

    caption_path = os.path.join(ctx.outdir, f'{name}.caption.txt')
    with open(caption_path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')

    record = {
        'figure': name,
        'files': [os.path.basename(p) for p in paths],
        'tool': 'experiments/plots.py',
        'command': 'python ' + ' '.join(['experiments/plots.py'] + ctx.argv),
        'argv': list(ctx.argv),
        'cwd': os.getcwd(),
        'inputs': {
            'per_seed_csv': ctx.per_seed_path,
            'per_seed_sha': ctx.hashes.get('per_seed'),
            'curves_csv': ctx.curves_path,
            'curves_sha': ctx.hashes.get('curves'),
        },
        'git': ctx.prov.get('git'),
        'plans': ctx.prov.get('plans'),
        'selection_plan_provenance': dict(ctx.selection_plan),
        'packages': ctx.prov.get('packages'),
        'bootstrap': {'n_boot': ctx.n_boot, 'seed': ctx.boot_seed},
        'smoothing_window_eval_points': ctx.smooth,
        'seeds': [int(s) for s in seeds],
        'seed_blocks': ctx.seed_blocks_for(seeds),
        # The quotability verdict, in the record as well as on the canvas and
        # in the caption. `tables.py` writes `validation_stamp` and a
        # `seeds_per_arm` block with the unit spelled out into all six of its
        # sidecars, and not one of the nine figure sidecars carried either: a
        # figure whose caption has been re-typed by an author could not be
        # checked for quotability against its own provenance, which is the
        # whole mechanism DESIGN.md 8.3 provides for detecting a stale
        # artefact. The n recovered indirectly from `interval_methods` and
        # `figure_specific.n` was no substitute, because that n could itself be
        # a row count.
        'validation': {
            'stamped': bool(stamped),
            'stamp': stats.VALIDATION_STAMP if stamped else None,
            'n_min': None if n_min is None else int(n_min),
            'n_min_unit': 'distinct seeds in the smallest arm, not rows',
            'n_distinct_seeds_plotted': len(seeds),
            'min_n_for_inference': int(stats.MIN_N_FOR_INFERENCE),
            'n_min_exceeds_distinct_seeds': n_leak,
            'quotable': bool(not stamped),
            'rule': 'ANALYSIS_PLAN.md 9, STANDING_INSTRUCTIONS S8: below '
                    f'n={stats.MIN_N_FOR_INFERENCE} distinct seeds nothing in '
                    'the figure may be quoted, compared, or used to choose '
                    'between hypotheses',
        },
        'source_policy_stamp': (None if ctx.source_policy == 'valid'
                                else 'SECONDARY ANALYSIS SET: POOLED OVER '
                                     'SOURCE COMPETENCE (DESIGN.md 4.3), NOT '
                                     'THE PRIMARY ESTIMAND'),
        'analysis_set': ctx.analysis,
        # DESIGN.md 8.3: the sidecar is how a rendered artifact is checked
        # against what produced it. Without this a reader holding the PDF could
        # check the seed count and the plan hash but not whether the study had
        # licensed a single conclusion behind it.
        'arbitration': (ctx.arbitration if ctx.arbitration is not None else
                        {'read': False,
                         'sentence': ctx.arbitration_sentence()}),
        'stats_json': ctx.stats_path,
        'stats_sha': ctx.stats_sha,
        'source_policy': ctx.source_policy,
        'interval_methods': (methods.record() if methods is not None else None),
        'runs_not_used_by_this_figure': dropped or {},
        'curve_table_integrity': ctx.curve_integrity,
        'analyses_carrying_a_p_value': 0,
        'arm_labels': ctx.arms,
        'arm_seed_collisions': ctx.collisions,
        'duplicated_arm_rows': ctx.duplicate_rows,
        'runs_selected': len(ctx.selected),
        'evaluation_protocol': protocol,
        'normalisation_references': refs,
        'figure_specific': meta or {},
    }
    prov_path = os.path.join(ctx.outdir, f'{name}.provenance.json')
    with open(prov_path, 'w', encoding='utf-8') as fh:
        json.dump(record, fh, indent=2, sort_keys=True, default=str)
    ctx.written.extend(paths + [caption_path, prov_path])
    print(f'  wrote {name}: ' + ', '.join(os.path.basename(p) for p in paths)
          + f', {name}.caption.txt, {name}.provenance.json')


# ---------------------------------------------------------------------------
# 7. Figure 1 -- learning curves
# ---------------------------------------------------------------------------
def fig_learning_curves(ctx: Context) -> None:
    cells = [c for c in CELL_ORDER if c in ctx.arms]
    if not cells or ctx.curves.empty:
        print(f'{WARN} learning_curves: no curve data for any cell; skipped')
        return

    frames: dict[tuple[str, str], pd.DataFrame] = {}
    labels_used: list[str] = []
    for cell in cells:
        for cond in ('scratch', 'transfer'):
            label = ctx.arms.get(cell, {}).get(cond)
            if label:
                frames[(cell, cond)] = curve_rows(ctx, label)
                labels_used.append(label)
    x0, x1, n_runs = _support(list(frames.values()), 'eval_score')
    if n_runs == 0 or not np.isfinite(x0) or x1 <= x0:
        print(f'{WARN} learning_curves: no common env-step support across the '
              f'plotted runs; skipped')
        return
    grid = np.linspace(x0, x1, ctx.grid_points)

    fig, axes = plt.subplots(2, 2, figsize=(FULL_WIDTH, 4.5), sharex=True,
                             sharey=True)
    seeds: set[int] = set()
    n_min = None
    log = MethodLog()
    meta: dict[str, Any] = {'grid_env_steps': [float(x0), float(x1)],
                            'grid_points': ctx.grid_points, 'panels': {}}
    boundaries: dict[str, dict] = {}
    drew_boundary = False
    for ax, cell in zip(axes.ravel(), cells):
        panel: dict[str, Any] = {}
        for cond in ('scratch', 'transfer'):
            frame = frames.get((cell, cond))
            if frame is None or frame.empty:
                continue
            mat, sds = series_matrix(ctx, frame, 'eval_score', grid)
            if mat.shape[0] == 0:
                continue
            seeds.update(sds)
            n_min = mat.shape[0] if n_min is None else min(n_min, mat.shape[0])
            style = CONDITION_STYLE[cond]
            b = band(ctx, mat)
            log.add_method(b['method'])
            if b['lo'] is not None:
                ax.fill_between(grid, b['lo'], b['hi'], color=style['colour'],
                                alpha=0.16, linewidth=0)
            line, = ax.plot(grid, b['mean'], color=style['colour'],
                            label=style['name'])
            if style['dashes'][0] is not None:
                line.set_dashes(style['dashes'])
            panel[cond] = {'n': int(mat.shape[0]), 'band': b['method'],
                           'seeds': sorted(int(s) for s in sds),
                           'final_grid_mean': float(b['mean'][-1])}
        bnd = freeze_boundary(ctx, frames.get((cell, 'transfer'), pd.DataFrame()))
        boundaries[cell] = bnd
        drew_boundary |= draw_boundary(ax, bnd)
        ax.axhline(0.0, color=_GREY, linewidth=0.5)
        ax.axhline(1.0, color=_GREY, linewidth=0.5, dashes=(1, 2))
        ax.set_title(cell.replace('-', ' / '))
        missing = absent_notes(
            ctx, [ctx.arms.get(cell, {}).get(c) for c in ('scratch',
                                                          'transfer')
                  if c not in panel],
            fallback='rows exist but no evaluation curve on the common grid')
        if not panel:
            _no_data(ax, 'no evaluation curve for this cell'
                     + (chr(10) + '; '.join(missing) if missing else ''))
        else:
            # A panel drawn with one curve where its neighbours have two says
            # nothing about the missing one unless this line is here.
            note_absent(ax, missing)
        meta['panels'][cell] = panel
        meta.setdefault('series_absent', {})[cell] = missing
    for ax in axes.ravel()[len(cells):]:
        _no_data(ax, 'cell not present in the supplied table')
    for ax in axes[1]:
        ax.set_xlabel('environment steps')
    for ax in axes[:, 0]:
        ax.set_ylabel('normalised evaluation score')

    handles = [Line2D([], [], color=CONDITION_STYLE[c]['colour'],
                      dashes=CONDITION_STYLE[c]['dashes']
                      if CONDITION_STYLE[c]['dashes'][0] else (10, 0),
                      label=CONDITION_STYLE[c]['name'])
               for c in ('scratch', 'transfer')]
    handles.append(Patch(facecolor=_BLUE, alpha=0.16,
                         label='95% bootstrap band'))
    if drew_boundary:               # never advertise a line that is not there
        handles.append(Line2D([], [], color=_GREY, dashes=(3, 2),
                              label='freeze window ends'))
    handles.append(Line2D([], [], color=_GREY, linewidth=0.5, dashes=(1, 2),
                          label='score 1 = registered threshold'))
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    _legend(fig, handles, ncol=len(handles))

    exited = sum(b['exited'] for b in boundaries.values())
    total = sum(b['runs'] for b in boundaries.values())
    max_upd = max([b['max_updates'] for b in boundaries.values()
                   if np.isfinite(b['max_updates'])] or [float('nan')])
    transfer_labels = [ctx.arms[c]['transfer'] for c in cells
                       if 'transfer' in ctx.arms.get(c, {})]
    # `freeze_updates` is not in the required-column set, because a table
    # without it is still a per_seed table; indexing it unconditionally turned
    # its absence into a traceback rather than a missing sentence.
    trows = ctx.rows_for_labels(transfer_labels)
    ks = (sorted({int(v) for v in trows['freeze_updates'].dropna().tolist()})
          if 'freeze_updates' in trows.columns else [])
    if exited:
        bnd_text = (f'The freeze boundary is drawn per cell at the mean env '
                    f'step where the window ended ({exited} of {total} '
                    f'transfer runs left it within budget).')
    else:
        bnd_text = (f'No freeze boundary is drawn: none of the {total} transfer '
                    f'runs left the freeze window within its budget (window '
                    f'{ks or "not recorded in this table"} gradient updates '
                    f'against at most {_i(max_upd)} updates performed), so the '
                    f'entire curve is inside the window.')

    emit(ctx, 'learning_curves', fig,
         body=('Normalised evaluation score against environment steps, one '
               'panel per (architecture, Q-target rule) cell on shared axes, '
               'scratch against transfer at matched seeds. The x-axis is env '
               'steps and not episodes because LunarLander episode length is '
               'performance-dependent, so an episode index compares arms at '
               'different amounts of learning (DESIGN.md 3.2). Curves are the '
               'mean across seeds of each run interpolated onto a common '
               f'env-step grid of {ctx.grid_points} points spanning '
               f'[{_i(x0)}, {_i(x1)}] steps -- the intersection of the plotted '
               'runs\' ranges, so no point of any band is an extrapolation and '
               'every point rests on the same number of seeds. ' + bnd_text),
         seeds=sorted(seeds), n_min=n_min,
         protocol=ctx.protocol(ctx.run_dirs_for_labels(labels_used)),
         refs=ctx.references(ctx.run_dirs_for_labels(labels_used)),
         interval=(f'shaded band = 95% percentile bootstrap over seeds, '
                   f'{ctx.n_boot} resamples, fixed seed {ctx.boot_seed} '
                   f'(stats.boot_indices); suppressed per arm where n < '
                   f'{stats.MIN_N_FOR_INFERENCE}, never suppressed globally'),
         methods=log,
         extra_lines=[
             'The dashed horizontal line at 1.0 is the registered solved '
             'threshold and the solid line at 0.0 is the measured random '
             'policy, both by construction of the normalisation.'],
         meta={**meta, 'freeze_boundaries': boundaries,
               'freeze_updates_levels_in_transfer_arms': ks})


# ---------------------------------------------------------------------------
# 8. Figure 2 -- the transfer-effect forest
# ---------------------------------------------------------------------------
def fig_transfer_effect_forest(ctx: Context) -> None:
    endpoints = stats.CONFIRMATORY_ENDPOINTS
    cells = [c for c in CELL_ORDER if c in ctx.arms]
    fig, axes = plt.subplots(1, len(endpoints),
                             figsize=(FULL_WIDTH, 0.42 * len(cells) + 1.7),
                             sharey=True)
    axes = np.atleast_1d(axes)
    meta: dict[str, Any] = {'cells': {}}
    seeds: set[int] = set()
    n_min = None
    log = MethodLog()
    lost: dict[str, list[str]] = {'unmatched seeds': [],
                                  'duplicated arm rows': []}
    margin = stats.EQUIVALENCE_MARGIN
    labels_used = [lbl for cell in cells for lbl in
                   (ctx.arms[cell].get('scratch'),
                    ctx.arms[cell].get('transfer')) if lbl]
    n_by_cell: dict[str, set[int]] = {}

    for ax, endpoint in zip(axes, endpoints):
        # The margin band is drawn only where a margin exists. ANALYSIS_PLAN.md
        # 4 fixes +/-0.05 for the normalised-score delta and justifies it as ~20
        # return points on LunarLander; painting the same band on the
        # area-per-env-step panel would extend a pre-registered quantity to a
        # scale it was never defined for, by nothing more than a shared axis.
        if endpoint in MARGIN_ENDPOINTS:
            ax.axvspan(-margin, margin, color=_GREY, alpha=0.13, linewidth=0,
                       zorder=0)
        ax.axvline(0.0, color=_BLACK, linewidth=0.7, zorder=1)
        ypos = list(range(len(cells)))[::-1]
        for y, cell in zip(ypos, cells):
            labels = ctx.arms[cell]
            if 'scratch' not in labels or 'transfer' not in labels:
                continue
            p = pair(ctx, labels['scratch'], labels['transfer'], endpoint)
            # Recorded before the empty-pair exit, not after: the case where an
            # arm is missing entirely is the one whose lost seeds most need
            # naming, and it is exactly the case the early `continue` used to
            # skip past.
            _note_losses(lost, cell, endpoint, p)
            if p.n == 0:
                # Not " no matched seeds". That is true of the pairing and
                # false about the cause whenever an arm was removed before any
                # pairing was attempted: on the real tree both dueling-vanilla
                # arms are outside the analysis set because their source failed
                # the DESIGN.md 4.3 gate, and the row read as though ten seeds
                # had failed to line up.
                gone = absent_notes(
                    ctx, [labels['scratch'], labels['transfer']],
                    fallback='has rows but no value for this endpoint')
                # One short cause on the row, the arm names in the record: the
                # row already names the cell, and a line long enough to hold
                # both arm labels runs off the panel and takes the cause with
                # it. A removal outranks a missing value, because a removal is
                # the thing a reader would otherwise mistake for a null.
                causes = sorted({g.split(': ', 1)[-1] for g in gone
                                 if ': removed' in g}) \
                    or sorted({g.split(': ', 1)[-1] for g in gone})
                reason = '; '.join(causes) or 'no seed has a run in both arms'
                ax.annotate(textwrap.shorten(reason, 56),
                            xy=(0.02, y), xycoords=('axes fraction', 'data'),
                            fontsize=5.6, color='#8A6D3B', va='center',
                            ha='left', style='italic')
                meta.setdefault('cells_absent', {}).setdefault(
                    cell, {})[endpoint] = gone
                continue
            seeds.update(p.seeds)
            n_min = p.n if n_min is None else min(n_min, p.n)
            est = log.add(estimate_shift(ctx, p.delta))
            sd_base = float(np.std(p.base, ddof=1)) if p.n > 1 else float('nan')
            sd_treat = float(np.std(p.treat, ddof=1)) if p.n > 1 else float('nan')
            # `np.nanmax` over two NaNs warns on stderr and returns NaN. At n=1
            # both SDs are NaN by construction, so the project's own primary
            # table printed the warning eight times per invocation. The NaN is
            # the right answer and the equivalence branch handles it; only the
            # noise is wrong.
            sds = [v for v in (sd_base, sd_treat) if np.isfinite(v)]
            sd_cell = float(max(sds)) if sds else float('nan')
            ax.plot(p.delta, [y + 0.22] * p.n, marker='|', linestyle='none',
                    color=_GREY, markersize=4.5, markeredgewidth=0.8, zorder=2)
            if np.isfinite(est['lo']) and np.isfinite(est['hi']):
                ax.plot([est['lo'], est['hi']], [y, y], color=_BLUE,
                        linewidth=1.4, solid_capstyle='butt', zorder=3)
                for edge in (est['lo'], est['hi']):
                    ax.plot([edge], [y], marker='|', color=_BLUE,
                            markersize=5.0, markeredgewidth=1.1, zorder=3)
            ax.plot([est['estimate']], [y], marker='o', color=_BLUE,
                    markersize=4.2, zorder=4)
            n_by_cell.setdefault(cell, set()).add(p.n)
            arb = ctx.arbitration_for(endpoint, cell)
            # On the panel, not only in the record: a row whose interval sits
            # clear of zero reads as a result, and the reason it is not one has
            # to be visible on the canvas that gets pasted into a slide.
            ax.annotate(textwrap.shorten(arb['label'], 58),
                        xy=(0.02, y - 0.26), xycoords=('axes fraction', 'data'),
                        fontsize=5.2,
                        color=('#8A6D3B' if arb['blocks'] else '#3C763D'),
                        va='center', ha='left', style='italic')
            meta['cells'].setdefault(cell, {})[endpoint] = {
                'n': p.n, 'seeds': p.seeds,
                'arbitration': arb,
                'hodges_lehmann': est['estimate'], 'ci_lo': est['lo'],
                'ci_hi': est['hi'], 'ci_method': est['method'],
                'ci_note': est.get('note', ''),
                'across_seed_sd_scratch': sd_base,
                'across_seed_sd_transfer': sd_treat,
                'exclusion': exclusion_sentence(
                    est, ENDPOINT_UNITS.get(endpoint, 'units')),
                'equivalence': equivalence_sentence(
                    est, sd_cell, n=p.n, endpoint=endpoint),
                'unmatched_scratch_seeds': p.unmatched_base,
                'unmatched_transfer_seeds': p.unmatched_treat,
                'duplicate_arm_seeds': p.duplicates,
            }
        ax.set_yticks(list(range(len(cells)))[::-1])
        ax.set_yticklabels([_cell_label_with_n(c, n_by_cell) for c in cells])
        ax.set_xlabel(f'delta {endpoint} (transfer - scratch)')
        ax.set_title(f'{endpoint}  (co-primary '
                     f'{"P1" if endpoint == "final_score" else "P2"})')
        ax.set_ylim(-0.6, len(cells) - 0.35)
        ax.grid(axis='y', visible=False)

    handles = [
        Line2D([], [], color=_BLUE, marker='o', linestyle='-',
               label='Hodges-Lehmann shift, 95% CI'),
        Line2D([], [], color=_GREY, marker='|', linestyle='none',
               label='per-seed delta'),
        Patch(facecolor=_GREY, alpha=0.13,
              label=f'+/-{margin:.2f} equivalence margin (final_score panel '
                    f'only)'),
    ]
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    _legend(fig, handles, ncol=3)

    verdicts = []
    arb_lines = []
    for cell in cells:
        rec = (meta['cells'].get(cell) or {}).get('final_score')
        if rec:
            verdicts.append(f"{cell}: {rec['exclusion']}; {rec['equivalence']}")
        for endpoint in endpoints:
            row = (meta['cells'].get(cell) or {}).get(endpoint) or {}
            arb = row.get('arbitration')
            if arb:
                arb_lines.append(f"{endpoint}/{cell}: {arb['label']}")

    emit(ctx, 'transfer_effect_forest', fig,
         body=('The within-cell transfer effect on each co-primary endpoint: '
               'the paired delta transfer - scratch at matched seeds, which is '
               'the only estimand that separates transferability from '
               'target-task suitability (DESIGN.md 2.5). Marker is the '
               'Hodges-Lehmann shift, bar is its bias-corrected bootstrap 95% '
               'CI, ticks above each row are the individual per-seed deltas. '
               'The shaded band is the pre-registered +/-0.05 normalised-score '
               'equivalence margin of ANALYSIS_PLAN.md 4, drawn so a reader can '
               'see directly whether an interval lies inside it; it is not a '
               'test, and a CI that merely covers zero is not evidence of '
               'equivalence. The margin is drawn on the final_score panel '
               'ONLY: the plan fixes it in normalised-score units, justified '
               'as about 20 return points on LunarLander, and that '
               'justification does not carry over to an area-per-env-step '
               'quantity, so auc_score gets no band and no equivalence '
               'verdict.'),
         seeds=sorted(seeds), n_min=n_min,
         protocol=ctx.protocol(ctx.run_dirs_for_labels(labels_used)),
         refs=ctx.references(ctx.run_dirs_for_labels(labels_used)),
         interval=(f'Hodges-Lehmann paired shift with a bias-corrected (BCa) '
                   f'seed-level bootstrap 95% CI requested at {ctx.n_boot} '
                   f'resamples, fixed seed {ctx.boot_seed} '
                   f'(stats.bootstrap_statistic); suppressed where n < '
                   f'{stats.MIN_N_FOR_INFERENCE}. BCa is what is requested, '
                   f'not necessarily what was returned: the tally below is '
                   f'what the estimator actually produced'),
         methods=log, dropped={k: '; '.join(v) for k, v in lost.items()},
         smoothing='smoothing window: none (endpoint scalars, not curves)',
         extra_lines=[
             ('Exclusion and equivalence statements generated per cell: '
              + ' | '.join(verdicts)) if verdicts else
             'No cell had a matched scratch/transfer pair.',
             'No p-value is drawn. These four contrasts on two endpoints are '
             'the whole confirmatory family, and their Holm-adjusted p-values '
             'belong in the results table: a forest of significance verdicts '
             'invites the "A avoids negative transfer while B does not" '
             'comparison that DESIGN.md 9 and ANALYSIS_PLAN.md 8 forbid.',
             ('DESIGN.md 3.3 arbitration, per row, printed on the panel as '
              'well as here so that a row lifted into a slide carries it: '
              + ' | '.join(arb_lines)) if arb_lines else
             'DESIGN.md 3.3 arbitration: no row of this forest carries an '
             'estimate, so there is no member to read a verdict for.'],
         meta=meta)


# ---------------------------------------------------------------------------
# 9. Figure 3 -- the control decomposition
# ---------------------------------------------------------------------------
def fig_control_decomposition(ctx: Context) -> None:
    endpoint = 'final_score'
    cells = [c for c in CELL_ORDER if c in ctx.arms]
    labels_used = [lbl for cell in cells
                   for lbl in ctx.arms[cell].values()]
    meta: dict[str, Any] = {'endpoint': endpoint, 'cells': {}}
    seeds: set[int] = set()
    n_min = None
    log = MethodLog()
    lost: dict[str, list[str]] = {'seeds lost to the four-way intersection': [],
                                  'duplicated arm rows': []}
    missing: dict[str, list[str]] = {}

    # --- pass 1: estimate everything, so the y range can be shared ---------
    # The panels share a y axis, so a per-panel `set_ylim` would leave the last
    # call governing every panel and clip the others' markers. Everything is
    # therefore estimated first and the limits are set once.
    prepared: dict[str, dict] = {}
    for cell in cells:
        labels = ctx.arms[cell]
        dupes: list[str] = []
        per_cond: dict[str, pd.Series] = {}
        for cond in CONTRAST_ORDER:
            if cond not in labels:
                continue
            series = _seed_series(ctx, labels[cond], endpoint, dupes)
            if not series.empty:
                per_cond[cond] = series
        present = [c for c in CONTRAST_ORDER if c in per_cond]
        absent = [c for c in CONTRAST_ORDER if c not in present]
        missing[cell] = absent
        if not present:
            prepared[cell] = {'reason': 'no control runs for this cell',
                              'absent': absent}
            continue
        # One common seed set per cell: the contrasts come from a single
        # shared resampling (ANALYSIS_PLAN.md 3), which a ragged seed set
        # cannot support.
        common = sorted(set.intersection(*[set(v.index)
                                           for v in per_cond.values()]))
        # A seed that has a run in some conditions but not all is dropped by
        # this intersection. That is correct (the joint bootstrap needs one seed
        # set) and it must not be quiet: the loss is recorded per condition and
        # printed in the caption, because a cell silently falling from n=3 to
        # n=2 is the silent-seed-dropping defect of DESIGN.md 1.
        for cond, series in per_cond.items():
            dropped_seeds = sorted(set(series.index) - set(common))
            if dropped_seeds:
                lost['seeds lost to the four-way intersection'].append(
                    f'{cell} {cond} seeds {_seed_list(dropped_seeds)}')
        if dupes:
            lost['duplicated arm rows'].append(
                f"{cell} {', '.join(sorted(set(dupes)))}")
        if not common:
            prepared[cell] = {'reason': 'no seed has a run in every condition',
                              'absent': absent}
            continue
        units = np.column_stack([per_cond[c].reindex(common).to_numpy(float)
                                 for c in present])
        n = len(common)
        seeds.update(int(v) for v in common)
        n_min = n if n_min is None else min(n_min, n)
        idx = (stats.boot_indices(n, ctx.n_boot, ctx.boot_seed)
               if n >= stats.MIN_N_FOR_INFERENCE else None)
        levels = {cond: log.add(estimate_mean(ctx, units,
                                              present.index(cond), idx))
                  for cond in present}
        contrasts: dict[str, dict] = {}
        for hi_c, lo_c, name in CONTRAST_NAMES:
            if hi_c not in present or lo_c not in present:
                continue
            est = log.add(estimate_difference(
                ctx, units, present.index(hi_c), present.index(lo_c), idx))
            # `note` is carried, not dropped. It is where bca_interval records
            # that it fell back to a percentile interval and why, so without it
            # the reason the method changed is unrecoverable from provenance.
            contrasts[name] = {'contrast': f'{hi_c} - {lo_c}',
                               'hi_cond': hi_c, 'lo_cond': lo_c,
                               'note': est.get('note', ''),
                               **{k: est[k] for k in ('estimate', 'lo', 'hi',
                                                      'method')}}
        prepared[cell] = {'present': present, 'absent': absent, 'units': units,
                          'common': common, 'levels': levels,
                          'contrasts': contrasts, 'n': n,
                          'duplicate_arm_seeds': sorted(set(dupes))}
        meta['cells'][cell] = {
            'n': n, 'seeds': common, 'order': present,
            'levels': {k: {kk: v.get(kk) for kk in ('estimate', 'lo', 'hi',
                                                    'method', 'note')}
                       for k, v in levels.items()},
            'seeds_lost_to_intersection': {
                c: sorted(set(s.index) - set(common))
                for c, s in per_cond.items() if set(s.index) - set(common)},
            'contrasts': contrasts, 'conditions_absent': absent,
            'duplicate_arm_seeds': sorted(set(dupes))}

    # --- the shared y range, with the top of each panel reserved ----------
    values: list[float] = []
    for rec in prepared.values():
        if 'units' not in rec:
            continue
        values.extend(float(v) for v in np.asarray(rec['units']).ravel()
                      if np.isfinite(v))
        for est in rec['levels'].values():
            values.extend(float(est[k]) for k in ('lo', 'hi')
                          if np.isfinite(est[k]))
    if values:
        y_bot, y_top = min(values), max(values)
    else:
        y_bot, y_top = 0.0, 1.0
    span = max(y_top - y_bot, 1e-6)
    ylim = (y_bot - 0.08 * span, y_top + 0.58 * span)
    # Annotation rows live in the reserved band above the data, in axes
    # fractions, so they cannot collide with the panel title however the data
    # happen to fall. Adjacent gaps alternate between two heights, because two
    # labels centred over neighbouring gaps would otherwise overlap.
    arrow_fracs = (0.87, 0.68, 0.87)

    fig, axes = plt.subplots(1, max(len(cells), 1),
                             figsize=(FULL_WIDTH, 3.1), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, cell in zip(axes, cells):
        rec = prepared.get(cell, {})
        ax.set_title(cell.replace('-', ' / '))
        gone = absent_notes(
            ctx, [ctx.arms.get(cell, {}).get(c) for c in rec.get('absent', [])],
            fallback='has rows but no value for this endpoint')
        meta.setdefault('series_absent', {})[cell] = gone
        if 'units' not in rec:
            _no_data(ax, rec.get('reason', 'no runs for this cell')
                     + (chr(10) + '; '.join(gone) if gone else ''))
            continue
        # A panel showing C0 and C2 only, with an x tick set two shorter than
        # its neighbours', is otherwise indistinguishable from a cell whose
        # controls were measured and agreed.
        note_absent(ax, gone, where=(0.03, 0.02))
        present, units, n = rec['present'], rec['units'], rec['n']
        xs = list(range(len(present)))
        for x, cond in zip(xs, present):
            style = CONDITION_STYLE[cond]
            est = rec['levels'][cond]
            ax.plot([x] * n, units[:, present.index(cond)], marker='|',
                    linestyle='none', color=_GREY, markersize=4.5,
                    markeredgewidth=0.8, zorder=2)
            if np.isfinite(est['lo']):
                ax.plot([x, x], [est['lo'], est['hi']], color=style['colour'],
                        linewidth=1.3, zorder=3)
            ax.plot([x], [est['estimate']], marker=style['marker'],
                    color=style['colour'], markersize=4.8, zorder=4)
        for k, (hi_c, lo_c, name) in enumerate(CONTRAST_NAMES):
            if name not in rec['contrasts']:
                continue
            est = rec['contrasts'][name]
            xa, xb = present.index(lo_c), present.index(hi_c)
            frac = arrow_fracs[k % len(arrow_fracs)]
            ax.annotate('', xy=(xb, frac), xytext=(xa, frac),
                        xycoords=('data', 'axes fraction'),
                        textcoords=('data', 'axes fraction'),
                        arrowprops=dict(arrowstyle='<->', linewidth=0.6,
                                        color=_GREY, shrinkA=0, shrinkB=0))
            text = (f"{CONDITION_CODE[hi_c]}-{CONDITION_CODE[lo_c]}"
                    f"\n{est['estimate']:+.3f}")
            if np.isfinite(est['lo']):
                text += f"\n[{est['lo']:+.2f},{est['hi']:+.2f}]"
            ax.annotate(text, xy=((xa + xb) / 2.0, frac),
                        xycoords=('data', 'axes fraction'), xytext=(0, 2),
                        textcoords='offset points', fontsize=5.4,
                        color='#333333', ha='center', va='bottom')
        ax.set_xticks(xs)
        ax.set_xticklabels([CONDITION_CODE[c] for c in present])
        ax.set_xlim(-0.55, len(present) - 0.45)
        ax.set_title(f"{cell.replace('-', ' / ')}  (n={n})")
        ax.grid(axis='x', visible=False)
    for ax in axes[len(cells):]:
        _no_data(ax, 'cell not present in the supplied table')
    axes[0].set_ylim(*ylim)
    axes[0].set_ylabel(f'{endpoint} (normalised)')

    handles = [Line2D([], [], color=CONDITION_STYLE[c]['colour'],
                      marker=CONDITION_STYLE[c]['marker'], linestyle='none',
                      label=f"{CONDITION_CODE[c]} = "
                            f"{CONDITION_STYLE[c]['name'].split(' ', 1)[1]}")
               for c in CONTRAST_ORDER]
    handles.append(Line2D([], [], color=_GREY, marker='|', linestyle='none',
                          label='individual runs'))
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    _legend(fig, handles, ncol=5)

    absent_all = sorted({c for v in missing.values() for c in v})
    absent_text = (' Conditions with no runs in this table, and therefore no '
                   'marker: ' + ', '.join(absent_all) + '.' if absent_all
                   else ' All four conditions are present in every cell drawn.')

    emit(ctx, 'control_decomposition', fig,
         body=('The four conditions of DESIGN.md 4 per cell, ordered C0, C2, '
               'C3, C1 so that the three adjacent gaps are exactly the three '
               'named contrasts, left to right: untrained-source (C2-C0), '
               'permuted-source (C3-C2) and trained-vs-permuted (C1-C3). '
               'Markers are seed means with a bootstrap interval; ticks are '
               'individual runs. THE SUM OF THE THREE GAPS IS C1-C0 BY '
               'ARITHMETIC. That telescoping identity holds for any four '
               'numbers, is shown only to fix notation, and is not evidence of '
               'additivity, not a decomposition of a causal effect, and not '
               'testable (DESIGN.md 4.1). The contrasts are named after what '
               'was manipulated, never after a mechanism; in particular the '
               'permuted-source contrast also absorbs spectral effects, '
               'because an entry-wise shuffle preserves the Frobenius norm but '
               'not the singular-value spectrum.' + absent_text),
         seeds=sorted(seeds), n_min=n_min,
         protocol=ctx.protocol(ctx.run_dirs_for_labels(labels_used)),
         refs=ctx.references(ctx.run_dirs_for_labels(labels_used)),
         methods=log, dropped={k: '; '.join(v) for k, v in lost.items()},
         interval=(f'one joint seed-level bootstrap per cell -- a single shared '
                   f'resampling of seeds ({ctx.n_boot} resamples, fixed seed '
                   f'{ctx.boot_seed}) supplies every level and every contrast, '
                   f'so the contrasts\' correlations are respected rather than '
                   f'ignored by four separate two-sample procedures '
                   f'(ANALYSIS_PLAN.md 3). Levels and gaps are seed means, so '
                   f'the identity holds exactly for the plotted points; the '
                   f'Hodges-Lehmann estimates of the same contrasts are in '
                   f'stats.py\'s tables'),
         smoothing='smoothing window: none (endpoint scalars, not curves)',
         extra_lines=[
             'Only seeds at which every plotted condition has a run enter a '
             'cell, so all four levels rest on one seed set; the seeds used '
             'are listed in the provenance record.'],
         meta=meta)


# ---------------------------------------------------------------------------
# 10. Figure 4 -- the 2x2 interaction
# ---------------------------------------------------------------------------
def fig_interaction_2x2(ctx: Context) -> None:
    endpoints = stats.CONFIRMATORY_ENDPOINTS
    archs = ('mlp', 'dueling')
    rules = ('vanilla', 'double')
    rule_style = {'vanilla': dict(colour=_BLUE, marker='o', dashes=(None, None)),
                  'double': dict(colour=_ORANGE, marker='s', dashes=(4, 1.6))}
    fig, axes = plt.subplots(1, len(endpoints), figsize=(FULL_WIDTH, 2.9),
                             sharey=False)
    axes = np.atleast_1d(axes)
    meta: dict[str, Any] = {'cells': {}, 'interaction': {}}
    seeds: set[int] = set()
    n_min = None
    log = MethodLog()
    lost: dict[str, list[str]] = {'unmatched seeds': [],
                                  'duplicated arm rows': [],
                                  'seeds lost to the four-cell intersection':
                                      []}
    labels_used = [lbl for cell in CELL_ORDER
                   for lbl in (ctx.arms.get(cell, {}).get('scratch'),
                               ctx.arms.get(cell, {}).get('transfer')) if lbl]

    for ax, endpoint in zip(axes, endpoints):
        deltas: dict[str, pd.Series] = {}
        for cell in CELL_ORDER:
            labels = ctx.arms.get(cell, {})
            if 'scratch' not in labels or 'transfer' not in labels:
                continue
            p = pair(ctx, labels['scratch'], labels['transfer'], endpoint)
            _note_losses(lost, cell, endpoint, p)
            if p.n == 0:
                continue
            deltas[cell] = pd.Series(p.delta, index=p.seeds)
            seeds.update(p.seeds)
            n_min = p.n if n_min is None else min(n_min, p.n)
        if not deltas:
            _no_data(ax, 'no matched scratch/transfer pair in any cell')
            ax.set_title(endpoint)
            continue
        ax.axhline(0.0, color=_BLACK, linewidth=0.7)
        for rule in rules:
            xs, ys, los, his = [], [], [], []
            for j, arch in enumerate(archs):
                cell = f'{arch}-{rule}'
                if cell not in deltas:
                    continue
                est = log.add(estimate_shift(ctx, deltas[cell].to_numpy(float)))
                xs.append(j)
                ys.append(est['estimate'])
                los.append(est['lo'])
                his.append(est['hi'])
                meta['cells'].setdefault(cell, {})[endpoint] = {
                    'n': int(len(deltas[cell])),
                    'seeds': sorted(int(s) for s in deltas[cell].index),
                    'hodges_lehmann': est['estimate'], 'ci_lo': est['lo'],
                    'ci_hi': est['hi'], 'ci_method': est['method'],
                    'ci_note': est.get('note', '')}
            if not xs:
                continue
            st = rule_style[rule]
            xs_off = [x + (0.05 if rule == 'double' else -0.05) for x in xs]
            line, = ax.plot(xs_off, ys, color=st['colour'], marker=st['marker'],
                            label=f'target rule: {rule}')
            if st['dashes'][0] is not None:
                line.set_dashes(st['dashes'])
            for x, lo, hi in zip(xs_off, los, his):
                if np.isfinite(lo):
                    ax.plot([x, x], [lo, hi], color=st['colour'],
                            linewidth=1.2)
        # Every cell the 2x2 needs and does not have, on the canvas. A series
        # reduced to one lone marker with no line is otherwise
        # indistinguishable from a series that was measured at one level only.
        gone = absent_notes(
            ctx, [lbl for cell in CELL_ORDER if cell not in deltas
                  for lbl in (ctx.arms.get(cell, {}).get('scratch'),
                              ctx.arms.get(cell, {}).get('transfer'))],
            fallback='has rows but no matched pair for this endpoint')
        meta.setdefault('cells_absent', {})[endpoint] = gone
        # The interaction contrast, on the seeds common to all four cells.
        if len(deltas) != 4:
            # The caption used to promise "with the interaction contrast
            # annotated" while this branch quietly drew nothing and left
            # meta['interaction'] empty: a caption asserting content the figure
            # does not contain, which is reviewer concern C8 exactly.
            missing = [c for c in CELL_ORDER if c not in deltas]
            meta['interaction'][endpoint] = {
                'computed': False,
                'reason': f'the 2x2 interaction needs all four cells and '
                          f'{len(deltas)} are present; missing '
                          f'{", ".join(missing)}',
                'cells_absent': gone}
            # Drawn as its own annotation, not folded into the "not drawn"
            # list: that list is truncated when it is long, and the one line a
            # reader must not miss is the one saying the annotation the caption
            # used to promise is absent.
            ax.annotate(
                'INTERACTION CONTRAST NOT COMPUTED' + chr(10)
                + f'(it needs all four cells; {len(deltas)} are present)',
                xy=(0.5, 0.93), xycoords='axes fraction', fontsize=6.0,
                color='#8A6D3B', ha='center', va='top', zorder=6)
            note_absent(ax, gone, where=(0.03, 0.02))
        if len(deltas) == 4:
            common = sorted(set.intersection(*[set(s.index)
                                               for s in deltas.values()]))
            for cell, s in deltas.items():
                gone = sorted(set(s.index) - set(common))
                if gone:
                    lost['seeds lost to the four-cell intersection'].append(
                        f'{cell}/{endpoint} seeds {_seed_list(gone)}')
            if common:
                units = np.column_stack([deltas[c].reindex(common)
                                         .to_numpy(float) for c in CELL_ORDER])
                n = len(common)
                idx = (stats.boot_indices(n, ctx.n_boot, ctx.boot_seed)
                       if n >= stats.MIN_N_FOR_INFERENCE else None)
                order = list(CELL_ORDER)
                i_md = order.index('mlp-double')
                i_mv = order.index('mlp-vanilla')
                i_dd = order.index('dueling-double')
                i_dv = order.index('dueling-vanilla')

                def inter(u: np.ndarray) -> float:
                    return float(np.mean((u[:, i_dd] - u[:, i_dv])
                                         - (u[:, i_md] - u[:, i_mv])))
                est = stats.bootstrap_statistic(
                    units, inter, n_boot=ctx.n_boot, seed=ctx.boot_seed,
                    idx=idx)
                est.pop('reps', None)
                log.add(est)
                meta['interaction'][endpoint] = {
                    'computed': True,
                    'definition': '(d[dueling-double] - d[dueling-vanilla]) - '
                                  '(d[mlp-double] - d[mlp-vanilla])',
                    'n': n, 'seeds': common, **est}
                ci = ('' if not np.isfinite(est['lo'])
                      else f"\n[{est['lo']:+.3f}, {est['hi']:+.3f}]")
                ax.annotate(f"interaction {est['estimate']:+.3f}{ci}",
                            xy=(0.5, 0.02), xycoords='axes fraction',
                            fontsize=6.2, color='#333333', ha='center',
                            va='bottom')
        ax.set_xticks(list(range(len(archs))))
        ax.set_xticklabels(archs)
        ax.set_xlim(-0.4, len(archs) - 0.6)
        ax.set_xlabel('architecture')
        ax.set_ylabel(f'delta {endpoint} (transfer - scratch)')
        ax.set_title(f'{endpoint}  (co-primary '
                     f'{"P1" if endpoint == "final_score" else "P2"})')
        ax.grid(axis='x', visible=False)

    handles = [Line2D([], [], color=rule_style[r]['colour'],
                      marker=rule_style[r]['marker'],
                      dashes=rule_style[r]['dashes']
                      if rule_style[r]['dashes'][0] else (10, 0),
                      label=f'target rule: {r}') for r in rules]
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    _legend(fig, handles, ncol=2)

    # The caption says what the figure HAS, not what it was meant to have. It
    # promised "with the interaction contrast annotated" unconditionally, and
    # on the real tree the contrast is not computed at all because the
    # DESIGN.md 4.3 gate removes one of the four cells: a caption asserting
    # content the figure does not contain is reviewer concern C8.
    computed = [ep for ep, rec in meta['interaction'].items()
                if rec.get('computed')]
    not_computed = {ep: rec.get('reason', '') for ep, rec
                    in meta['interaction'].items() if not rec.get('computed')}
    if computed and not not_computed:
        contrast_clause = 'with the interaction contrast annotated'
    elif computed:
        contrast_clause = (
            'with the interaction contrast annotated on '
            + ', '.join(computed) + ' and NOT COMPUTED on '
            + ', '.join(f'{ep} ({why})' for ep, why in not_computed.items()))
    else:
        why = ('; '.join(f'{ep}: {rec}' for ep, rec in not_computed.items())
               or 'no cell had a matched scratch/transfer pair on any endpoint')
        contrast_clause = (
            'The interaction contrast is NOT DRAWN and NOT COMPUTED on any '
            'panel: ' + why
            + '. Its absence is an analysis-set exclusion, not a null result')
    emit(ctx, 'interaction_2x2', fig,
         body=('Cell means of the within-cell transfer delta, architecture on '
               'x and Q-target rule as the series. ' + contrast_clause
               + '. This is RQ3, and RQ3 is EFFECT '
               'MODIFICATION -- how a causal effect varies across cells -- not '
               '"architecture causes the difference": the cells are different '
               'algorithms, not treatments assigned to units (DESIGN.md 2.4). '
               'It is estimation-only by design and not by omission: the '
               'interaction\'s minimum detectable effect is about 2.7 sigma at '
               'n=10, larger than any plausible effect, so it carries an '
               'interval and no p-value (ANALYSIS_PLAN.md 3, 6). Non-parallel '
               'lines here are not a finding.'),
         seeds=sorted(seeds), n_min=n_min,
         protocol=ctx.protocol(ctx.run_dirs_for_labels(labels_used)),
         refs=ctx.references(ctx.run_dirs_for_labels(labels_used)),
         methods=log, dropped={k: '; '.join(v) for k, v in lost.items()},
         interval=(f'Hodges-Lehmann paired shift per cell with a '
                   f'bias-corrected bootstrap 95% CI; the interaction contrast '
                   f'is a paired mean under one joint resampling of the seeds '
                   f'common to all four cells ({ctx.n_boot} resamples, seed '
                   f'{ctx.boot_seed}), which is valid because the 2x2 is '
                   f'matched by seed -- at a given seed the mlp and dueling '
                   f'networks share their trunk initialisation and the two '
                   f'target-rule levels share everything (DESIGN.md 8.1)'),
         smoothing='smoothing window: none (endpoint scalars, not curves)',
         extra_lines=[
             'Any between-cell reading also re-admits the headroom confound: a '
             'cell whose scratch baseline is near the ceiling has less to gain '
             'and more to lose (DESIGN.md 2.5). Agreement on the '
             'headroom-adjusted scale is required before wording is used.'],
         meta=meta)


# ---------------------------------------------------------------------------
# 11. Figure 5 -- the shift gradient
# ---------------------------------------------------------------------------
def _variant_level(env: str, family: str) -> Optional[float]:
    try:
        return envs.family_level_value(family, env)
    except Exception:                                            # noqa: BLE001
        return None


def fig_shift_gradient(ctx: Context) -> None:
    endpoint = 'final_score'
    families = (('ll_wind', 'wind (primary axis for H4)', False),
                ('ll_gravity', 'gravity (secondary: confounded with difficulty)',
                 True))
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 2.9), sharey=True,
                             gridspec_kw={'width_ratios': [3, 3, 1.35]})
    meta: dict[str, Any] = {'endpoint': endpoint, 'families': {},
                            'interface_corner': {},
                            'transferred_intensity': {},
                            'x_axis': ('measured divergence from --shift-metrics'
                                       if ctx.shift_metrics
                                       else 'declared manipulated level')}
    seeds: set[int] = set()
    n_min = None
    log = MethodLog()
    lost: dict[str, list[str]] = {'unmatched seeds': [],
                                  'duplicated arm rows': []}
    labels_used: list[str] = []
    for cell in CELL_ORDER:
        for rec in ctx.shifts.get(cell, []):
            labels_used.extend(v for k, v in rec.items() if k != 'env')
        labels_used.extend(ctx.iface.get(cell, {}).values())

    for ax, (family, title, confounded) in zip(axes[:2], families):
        base_value = None
        try:
            base_value = envs.family_level_value(
                family, envs.VARIANT_FAMILIES[family]['base'])
        except Exception:                                        # noqa: BLE001
            base_value = None
        points: dict[str, list[tuple[float, dict, str]]] = {}
        for cell in CELL_ORDER:
            for rec in ctx.shifts.get(cell, []):
                env = rec.get('env')
                if 'scratch' not in rec or 'transfer' not in rec:
                    continue
                level = _variant_level(env, family)
                if level is None or base_value is None:
                    continue
                if abs(level - base_value) < 1e-12:
                    continue          # the family's own level 0, no shift arm
                if ctx.shift_metrics:
                    x = ctx.shift_metrics.get(str(env))
                    if x is None:
                        continue
                else:
                    x = abs(level - base_value)
                p = pair(ctx, rec['scratch'], rec['transfer'], endpoint)
                _note_losses(lost, f'{family}/{env}/{cell}', endpoint, p)
                if p.n == 0:
                    continue
                seeds.update(p.seeds)
                n_min = p.n if n_min is None else min(n_min, p.n)
                est = log.add(estimate_shift(ctx, p.delta))
                points.setdefault(cell, []).append((float(x), est, str(env)))
        if confounded:
            ax.set_facecolor('#F4F1EC')
            ax.annotate('shift is confounded with task difficulty:\nthe no-op '
                        'score rises 0.18 -> 0.55 across\nthese levels '
                        '(DESIGN.md 5.1)',
                        xy=(0.03, 0.03), xycoords='axes fraction', fontsize=5.8,
                        color='#6B5B3E', ha='left', va='bottom')
        if not points:
            _no_data(ax, f'no {family.split("_")[-1]}-family runs in the '
                         f'supplied table', hide_x=True)
            ax.set_title(title, fontsize=7.6)
            meta['families'][family] = {'levels': 0}
            continue
        ax.axhline(0.0, color=_BLACK, linewidth=0.7)
        rec_out = {}
        for cell, pts in points.items():
            pts.sort(key=lambda t: t[0])
            st = CELL_STYLE[cell]
            xs = [t[0] for t in pts]
            ys = [t[1]['estimate'] for t in pts]
            line, = ax.plot(xs, ys, color=st['colour'], marker=st['marker'],
                            label=cell)
            if st['dashes'][0] is not None:
                line.set_dashes(st['dashes'])
            for x, est, _env in pts:
                if np.isfinite(est['lo']):
                    ax.plot([x, x], [est['lo'], est['hi']], color=st['colour'],
                            linewidth=1.1)
            rec_out[cell] = [{'x': x, 'env': env,
                              'estimate': est['estimate'], 'lo': est['lo'],
                              'hi': est['hi'], 'method': est['method'],
                              'note': est.get('note', '')}
                             for x, est, env in pts]
        meta['families'][family] = {'levels': max(len(v) for v in
                                                  points.values()),
                                    'points': rec_out}
        fam_labels = [rec['transfer'] for cell in CELL_ORDER
                      for rec in ctx.shifts.get(cell, [])
                      if 'transfer' in rec]
        meta['transferred_intensity'][family] = \
            ctx.intensity_for_labels(fam_labels)
        ax.set_title(title, fontsize=7.6)
        ax.set_xlabel('measured divergence' if ctx.shift_metrics
                      else '|change in the manipulated variable| from source')

    # The interface-only corner: its own panel, never on the shift axis.
    ax = axes[2]
    corner = {}
    for cell in CELL_ORDER:
        labels = ctx.iface.get(cell, {})
        if 'scratch' not in labels or 'transfer' not in labels:
            continue
        p = pair(ctx, labels['scratch'], labels['transfer'], endpoint)
        _note_losses(lost, f'interface/{cell}', endpoint, p)
        if p.n == 0:
            continue
        seeds.update(p.seeds)
        n_min = p.n if n_min is None else min(n_min, p.n)
        est = log.add(estimate_shift(ctx, p.delta))
        corner[cell] = {'n': p.n, 'estimate': est['estimate'], 'lo': est['lo'],
                        'hi': est['hi'], 'method': est['method'],
                        'note': est.get('note', ''), 'seeds': p.seeds}
    if corner:
        ax.axhline(0.0, color=_BLACK, linewidth=0.7)
        for j, (cell, rec) in enumerate(corner.items()):
            st = CELL_STYLE[cell]
            ax.plot([j], [rec['estimate']], marker=st['marker'],
                    color=st['colour'], linestyle='none')
            if np.isfinite(rec['lo']):
                ax.plot([j, j], [rec['lo'], rec['hi']], color=st['colour'],
                        linewidth=1.1)
        ax.set_xticks(list(range(len(corner))))
        ax.set_xticklabels(['' for _ in corner])
        ax.set_xlim(-0.7, len(corner) - 0.3)
        ax.grid(axis='x', visible=False)
    else:
        _no_data(ax, 'no interface-only runs in the supplied table',
                 hide_x=True)
    ax.set_title('interface change,\nzero dynamics shift', fontsize=7.0)
    meta['interface_corner'] = corner
    meta['transferred_intensity']['interface_only'] = ctx.intensity_for_labels(
        [lbl for cell in CELL_ORDER
         for lbl in [ctx.iface.get(cell, {}).get('transfer')] if lbl])
    axes[0].set_ylabel(f'delta {endpoint} (transfer - scratch)')

    # Do the panels share a transfer protocol? The answer is generated, not
    # asserted, because on the registry as it stands they do not: E8's shift
    # arms declare transfer_set='trunk' and E8i's interface arms declare
    # 'matched'. The verdict is on `transfer_set`, which is the declared
    # protocol; the transferred parameter *fraction* is reported alongside but
    # does not by itself raise the warning, because it differs between mlp and
    # dueling by construction (different parameter counts under the same
    # protocol) and that is a property of the 2x2, not a protocol change.
    panels = {k: v for k, v in meta['transferred_intensity'].items()
              if v.get('transfer_set')}
    per_panel = {k: tuple(v['transfer_set']) for k, v in panels.items()}
    sets_seen = sorted({s for v in per_panel.values() for s in v})
    fracs = {k: v.get('transferred_param_fraction', [])
             for k, v in meta['transferred_intensity'].items()}
    frac_text = '; '.join(f'{k} {v or "not recorded"}'
                          for k, v in sorted(fracs.items())) or 'not recorded'
    if len(set(per_panel.values())) > 1 or len(sets_seen) > 1:
        intensity_note = (
            'PROTOCOL IS NOT FIXED ACROSS THESE PANELS. The transfer arms '
            f'drawn here do not all declare the same transfer_set '
            f'({"; ".join(f"{k}: {list(v)}" for k, v in sorted(per_panel.items()))}), '
            'so reading the panels against one another crosses an unstated '
            'change in how much of the network was carried over. That is the '
            'intensity confound of DESIGN.md 3.1: it is stated here rather '
            'than hidden, and audit.py is the point at which a claim crossing '
            'it is refused. The shared y axis is a shared scale, not a licence '
            'to compare across panels. Transferred parameter fraction per '
            f'panel: {frac_text}.')
        ax.annotate('transfer protocol differs\nfrom the shift panels',
                    xy=(0.5, 0.995), xycoords='axes fraction', fontsize=5.2,
                    color='#8A6D3B', ha='center', va='top')
    else:
        intensity_note = (
            'Transfer protocol is fixed across the panels drawn: every '
            f'transfer arm here declares transfer_set '
            f'{sets_seen or "not recorded"}, so the cross-panel reading does '
            'not cross a DESIGN.md 3.1 protocol change. Transferred parameter '
            f'fraction per panel (which varies with architecture under one '
            f'protocol, and is not itself a protocol change): {frac_text}.')

    handles = [Line2D([], [], color=CELL_STYLE[c]['colour'],
                      marker=CELL_STYLE[c]['marker'],
                      dashes=CELL_STYLE[c]['dashes']
                      if CELL_STYLE[c]['dashes'][0] else (10, 0),
                      label=c.replace('-', ' / '))
               for c in CELL_ORDER]
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    _legend(fig, handles, ncol=4)

    x_note = ('The x-axis is the measured divergence supplied by '
              '--shift-metrics.' if ctx.shift_metrics else
              'No measured-divergence table was supplied, so the x-axis is the '
              'declared manipulated level -- the absolute change in wind power '
              'or in gravity relative to the source environment -- and not a '
              'measured distance between MDPs. A scalar shift metric is '
              'refused outright for the cross-interface pairs, where no '
              'distance between different state spaces is defined '
              '(DESIGN.md 6.3).')

    emit(ctx, 'shift_gradient', fig,
         body=('The within-variant transfer delta against shift level, at '
               'fixed interface, for the two '
               'same-interface families. Wind is the primary axis for H4 '
               'because its no-op score is flat across levels; the gravity '
               'panel is shaded and annotated because weakening gravity makes '
               'the task easier as well as different, so that family confounds '
               'shift severity with difficulty and corroborates rather than '
               'carries H4 (DESIGN.md 5.1, 6.2). The right-hand panel is the '
               'corner nobody had run: the same dynamics with a changed '
               'observation and action interface, drawn as separate points and '
               'never joined to the shift curves, because there is no shift '
               'axis it belongs on. Each point is a within-variant delta '
               'against that variant\'s own scratch arm, which is what keeps a '
               'variant\'s changed return scale out of the effect. ' + x_note),
         seeds=sorted(seeds), n_min=n_min,
         protocol=ctx.protocol(ctx.run_dirs_for_labels(labels_used)),
         refs=ctx.references(ctx.run_dirs_for_labels(labels_used)),
         methods=log, dropped={k: '; '.join(v) for k, v in lost.items()},
         interval=(f'Hodges-Lehmann paired shift per level with a '
                   f'bias-corrected bootstrap 95% CI, {ctx.n_boot} resamples, '
                   f'seed {ctx.boot_seed}. The ordered-alternative trend '
                   f'statistic for H4 is reported by stats.py, not drawn here'),
         smoothing='smoothing window: none (endpoint scalars, not curves)',
         extra_lines=[intensity_note],
         meta=meta)


# ---------------------------------------------------------------------------
# 12. Figure 6 -- freeze duration
# ---------------------------------------------------------------------------
def _freeze_family(ctx: Context) -> pd.DataFrame:
    """Transfer runs that differ from the protocol only in `freeze_updates`.

    Selected on the substantive fields rather than on a label prefix, because
    E4's K=10k level and E1's transfer arm are the same configuration and
    therefore the same run directory (`registry.py`: `all_jobs` de-duplicates
    by configuration digest), so neither label alone enumerates the family.

    Three restrictions that were missing, each of which let a run into the
    figure that the scratch side of the same pairing could never contain:

    * **`ctx.selected`.** The scratch arm comes through `pair`, which restricts
      to one run per (label, seed); this side did not, so a colliding run
      directory entered the treatment side of a contrast whose baseline had
      already excluded it. An asymmetric analysis set is not a contrast.
    * **The budget.** The docstring claimed membership in E1 or E4 keeps E0's
      smoke runs out because they have "a different episode budget entirely",
      but the guard was by experiment tag, not by budget, so a 14-episode run
      tagged `E1;E4` was plotted against 1000-episode scratch runs. The budget
      is now compared to the protocol's directly, which is what the sentence
      always meant.
    * **The analysis set.** `ctx.per_seed` is already filtered for TUNE seeds,
      donor blocks and source validity, so reading it (rather than the raw
      table) is what keeps a TUNE run matching every substantive field out.
    """
    ps = ctx.per_seed
    protocol = registry.PROTOCOL
    needed = ('condition', 'env', 'source_env', 'transfer_set', 'freeze_group')
    missing = [c for c in needed if c not in ps.columns]
    if missing:
        print(f'{WARN} freeze_duration: the per-seed table has no {missing} '
              f'column(s), so the freeze-duration family cannot be identified '
              f'by configuration and the figure is drawn empty rather than '
              f'from a guess')
        return ps.iloc[0:0]
    mask = ((ps['condition'] == 'transfer')
            & (ps['env'] == registry.TARGET_ENV)
            & (ps['source_env'] == registry.SOURCE_ENV)
            & (ps['transfer_set'] == protocol['transfer_set'])
            & (ps['freeze_group'] == protocol['freeze_group'])
            & ps['run_dir'].isin(ctx.selected))
    if 'num_episodes' in ps.columns:
        budget = pd.to_numeric(ps['num_episodes'], errors='coerce')
        mask = mask & (budget == float(registry.COMMON['num_episodes']))
    if 'experiments' in ps.columns:
        member = ps['experiments'].fillna('').apply(
            lambda s: bool({'E1', 'E4'} & set(str(s).split(';'))))
        mask = mask & member
    return ps[mask]


def fig_freeze_duration(ctx: Context) -> None:
    endpoint = 'final_score'
    fam = _freeze_family(ctx)
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH * 1.55, 2.9))
    meta: dict[str, Any] = {'endpoint': endpoint, 'cells': {},
                            'levels_found': [],
                            'family_budget_episodes':
                                int(registry.COMMON['num_episodes']),
                            'family_runs': int(len(fam))}
    seeds: set[int] = set()
    n_min = None
    log = MethodLog()
    lost: dict[str, list[str]] = {'unmatched seeds': [],
                                  'duplicated arm rows': []}
    if fam.empty:
        _no_data(ax, 'no freeze-duration family runs in the supplied table')
    else:
        levels = sorted({int(v) for v in fam['freeze_updates'].dropna()})
        meta['levels_found'] = levels
        finite = [k for k in levels if k > 0]
        # 0 and "never unfrozen" (-1) have no place on a log axis, so they are
        # drawn as bracketed off-scale positions and labelled as such.
        lo_pos = (min(finite) / 4.0) if finite else 1.0
        hi_pos = (max(finite) * 4.0) if finite else 10.0
        pos = {k: (lo_pos if k == 0 else hi_pos if k < 0 else float(k))
               for k in levels}
        ax.axhline(0.0, color=_BLACK, linewidth=0.7)
        for cell in CELL_ORDER:
            base = ctx.arms.get(cell, {}).get('scratch')
            if not base:
                continue
            sub = fam[fam['cell'] == cell]
            pts = []
            for k in levels:
                rows = sub[sub['freeze_updates'] == k]
                if rows.empty:
                    continue
                # Paired on the rows at THIS window length, not on the arm
                # label: the same label can carry two window lengths, and
                # pairing by label would return one level's runs for both.
                arm_labels = ', '.join(sorted(set(rows['label'].tolist())))
                p = pair(ctx, base, f'K={k}', endpoint, treat_rows=rows)
                _note_losses(lost, f'{cell}/K={k}', endpoint, p)
                if p.n == 0:
                    continue
                seeds.update(p.seeds)
                n_min = p.n if n_min is None else min(n_min, p.n)
                est = log.add(estimate_shift(ctx, p.delta))
                pts.append((pos[k], k, arm_labels, est, p.n))
            if not pts:
                continue
            pts.sort(key=lambda t: t[0])
            st = CELL_STYLE[cell]
            line, = ax.plot([t[0] for t in pts], [t[3]['estimate'] for t in pts],
                            color=st['colour'], marker=st['marker'], label=cell)
            if st['dashes'][0] is not None:
                line.set_dashes(st['dashes'])
            for x, _k, _lab, est, _n in pts:
                if np.isfinite(est['lo']):
                    ax.plot([x, x], [est['lo'], est['hi']], color=st['colour'],
                            linewidth=1.1)
            meta['cells'][cell] = [
                {'freeze_updates': k, 'labels': lab, 'n': n,
                 'estimate': est['estimate'], 'lo': est['lo'], 'hi': est['hi'],
                 'method': est['method'], 'note': est.get('note', ''),
                 'x_position': x, 'off_scale': k <= 0}
                for x, k, lab, est, n in pts]
        # A cell with no point is dropped from the legend, so without this the
        # reader has no way to know the cell existed at all: on the real tree
        # dueling-vanilla vanishes entirely because the DESIGN.md 4.3 gate
        # removed its transfer arms, and the figure looked like a three-cell
        # design.
        gone = absent_notes(
            ctx, [ctx.arms.get(cell, {}).get('transfer')
                  for cell in CELL_ORDER if cell not in meta['cells']],
            fallback='has rows but no freeze-duration family member here')
        meta['cells_absent'] = gone
        note_absent(ax, gone, where=(0.02, 0.92), va='top')
        ax.set_xscale('log')
        ticks = [pos[k] for k in levels]
        names = ['0\n(no freeze)' if k == 0 else
                 'never\nunfrozen' if k < 0 else f'{k:,}' for k in levels]
        ax.set_xticks(ticks)
        ax.set_xticklabels(names, fontsize=6.2)
        ax.set_xticks([], minor=True)
    ax.set_xlabel('freeze window (gradient updates, log scale)')
    ax.set_ylabel(f'delta {endpoint} (transfer - scratch)')
    handles = [Line2D([], [], color=CELL_STYLE[c]['colour'],
                      marker=CELL_STYLE[c]['marker'],
                      dashes=CELL_STYLE[c]['dashes']
                      if CELL_STYLE[c]['dashes'][0] else (10, 0),
                      label=c.replace('-', ' / '))
               for c in CELL_ORDER if c in meta['cells']]
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    _legend(fig, handles, ncol=2)

    n_levels = len(meta['levels_found'])
    emit(ctx, 'freeze_duration', fig,
         body=('The within-cell transfer delta against the length of the freeze '
               'window, one series per cell. The window is measured in '
               'GRADIENT UPDATES, not episodes: LunarLander episode length is '
               'performance-dependent, so an episode-indexed window would mean '
               'a different amount of learning in every arm and the levels '
               'would not be comparable (DESIGN.md 3.2). The manuscript this '
               'revises described a freeze schedule that was never '
               'implemented; here each window is logged as an event with '
               'trainable-parameter counts and verified by weight '
               f'fingerprints. {n_levels} window level(s) are present in the '
               'supplied table. A window of 0 and a window that is never '
               'released have no position on a log axis and are drawn at '
               'bracketed off-scale ticks, labelled as such. The family is '
               'identified by configuration and restricted to the protocol '
               f"budget of {registry.COMMON['num_episodes']} episodes, so a "
               'smoke run tagged E1 or E4 at a shorter budget is not plotted '
               'against full-length scratch arms.'),
         seeds=sorted(seeds), n_min=n_min,
         protocol=ctx.protocol(fam['run_dir'].tolist() if not fam.empty
                               else []),
         refs=ctx.references(fam['run_dir'].tolist() if not fam.empty else []),
         methods=log, dropped={k: '; '.join(v) for k, v in lost.items()},
         interval=(f'Hodges-Lehmann paired shift against the same cell\'s '
                   f'scratch arm with a bias-corrected bootstrap 95% CI, '
                   f'{ctx.n_boot} resamples, seed {ctx.boot_seed}'),
         smoothing='smoothing window: none (endpoint scalars, not curves)',
         extra_lines=[
             'This is a screen (E4). ANALYSIS_PLAN.md 3 permits it to select at '
             'most one follow-up, which is then run on REPLICATE seeds and '
             'reported as a fresh estimate; no level shown here is assertable '
             'as a finding.'],
         meta=meta)


# ---------------------------------------------------------------------------
# 13. Figure 7 -- mechanism diagnostics
# ---------------------------------------------------------------------------
def fig_diagnostics(ctx: Context) -> None:
    cells = [c for c in CELL_ORDER if c in ctx.arms]
    if not cells or ctx.curves.empty:
        print(f'{WARN} diagnostics: no curve data; skipped')
        return
    stream_cols = [c for c in ('grad_norm_trunk', 'grad_norm_value',
                               'grad_norm_adv', 'grad_norm_head')
                   if c in ctx.curves.columns]
    grad_cols = stream_cols or ([c for c in ('grad_norm_global', 'grad_norm')
                                 if c in ctx.curves.columns][:1])
    rows: list[dict[str, Any]] = [
        {'key': 'streams', 'columns': ['v_abs_mean', 'a_abs_mean'],
         'ylabel': 'mean |V|, mean |A|', 'log': False,
         'dueling_only': True},
        {'key': 'gradients', 'columns': grad_cols,
         'ylabel': ('per-stream gradient norm' if stream_cols
                    else 'gradient norm (global)'), 'log': True,
         'dueling_only': False},
        {'key': 'dead_units', 'columns': ['dead_unit_frac'],
         'ylabel': 'dead trunk units (fraction)', 'log': False,
         'dueling_only': False},
        {'key': 'cka_drift', 'columns': ['cka_drift'],
         'ylabel': 'CKA(trunk ep0, trunk t)', 'log': False,
         'dueling_only': False},
        # The plasticity-loss signatures. These are here because the plasticity
        # literature supplies a complete, architecture-free rival explanation
        # for degradation after pretraining -- feature-rank collapse and
        # parameter-norm growth -- that the weight-scale control (C3) does not
        # exclude: preserving a weight multiset says nothing about the rank of
        # the features those weights produce. Instrumenting them and then not
        # plotting them would leave the rival account unaddressed in exactly the
        # figure a reviewer would look at (paper/LITERATURE.md 3.4).
        {'key': 'effective_rank', 'columns': ['effective_rank'],
         'ylabel': 'effective rank, trunk features', 'log': False,
         'dueling_only': False},
        {'key': 'param_norm', 'columns': ['param_norm_trunk',
                                          'param_norm_total'],
         'ylabel': 'parameter L2 norm', 'log': False,
         'dueling_only': False},
    ]
    rows = [r for r in rows if any(c in ctx.curves.columns for c in r['columns'])]
    if not rows:
        print(f'{WARN} diagnostics: none of the mechanism columns are in the '
              f'curve table; skipped')
        return

    frames: dict[tuple[str, str], pd.DataFrame] = {}
    labels_used: list[str] = []
    for cell in cells:
        for cond in ('scratch', 'transfer'):
            label = ctx.arms.get(cell, {}).get(cond)
            if label:
                frames[(cell, cond)] = curve_rows(ctx, label)
                labels_used.append(label)

    # `sharey='row'`, not the previous no-sharing. A reader comparing curve
    # heights across a row of four cell panels was comparing four independent y
    # scales, and on the gradient row four independent log scales, which is a
    # figure that invites the comparison it cannot support. One row is one
    # signal, so one row is one scale.
    fig, axes = plt.subplots(len(rows), len(cells),
                             figsize=(FULL_WIDTH, 1.55 * len(rows) + 0.9),
                             sharex='col', sharey='row', squeeze=False)
    seeds: set[int] = set()
    n_min = None
    log = MethodLog()
    meta: dict[str, Any] = {'rows': [r['key'] for r in rows],
                            'gradient_columns': grad_cols, 'panels': {},
                            'y_axis_shared': 'per row'}
    dash_by_col = {}
    drew_boundary = False
    for i, spec in enumerate(rows):
        for j, col in enumerate(spec['columns']):
            dash_by_col[col] = [(None, None), (4, 1.6), (1.4, 1.4),
                                (5, 1.2, 1.2, 1.2)][j % 4]
    for i, spec in enumerate(rows):
        cols = [c for c in spec['columns'] if c in ctx.curves.columns]
        for j, cell in enumerate(cells):
            ax = axes[i][j]
            drew = False
            if spec['dueling_only'] and not cell.startswith('dueling'):
                # y IS shared across the row now, so the tick labels are not
                # hidden: switching them off on a shared axis blanks the scale
                # for the populated panels beside it, which is the trap
                # `_no_data` documents.
                _no_data(ax, 'no value/advantage streams here')
            else:
                x0, x1, n_runs = _support(
                    [frames.get((cell, c), pd.DataFrame())
                     for c in ('scratch', 'transfer')], cols[0] if cols else '')
                if n_runs and np.isfinite(x0) and x1 > x0:
                    grid = np.linspace(x0, x1, max(ctx.grid_points // 2, 20))
                    for cond in ('scratch', 'transfer'):
                        frame = frames.get((cell, cond))
                        if frame is None or frame.empty:
                            continue
                        style = CONDITION_STYLE[cond]
                        for col in cols:
                            mat, sds = series_matrix(ctx, frame, col, grid)
                            if mat.shape[0] == 0:
                                continue
                            seeds.update(sds)
                            n_min = (mat.shape[0] if n_min is None
                                     else min(n_min, mat.shape[0]))
                            b = band(ctx, mat)
                            log.add_method(b['method'])
                            if b['lo'] is not None:
                                ax.fill_between(grid, b['lo'], b['hi'],
                                                color=style['colour'],
                                                alpha=0.13, linewidth=0)
                            line, = ax.plot(grid, b['mean'],
                                            color=style['colour'])
                            dashes = dash_by_col.get(col, (None, None))
                            if dashes[0] is not None:
                                line.set_dashes(dashes)
                            drew = True
                            meta['panels'].setdefault(spec['key'], {}) \
                                .setdefault(cell, {})[f'{cond}:{col}'] = {
                                    'n': int(mat.shape[0]),
                                    'seeds': sorted(int(s) for s in sds),
                                    'band': b['method'],
                                    'final': float(b['mean'][-1])}
                    drew_boundary |= draw_boundary(ax, freeze_boundary(
                        ctx, frames.get((cell, 'transfer'), pd.DataFrame())))
                if not drew:
                    _no_data(ax, 'no measurements for this signal')
            if spec['log'] and drew:
                ax.set_yscale('log')
            if i == 0:
                ax.set_title(cell.replace('-', ' / '))
            if j == 0:
                ax.set_ylabel(spec['ylabel'], fontsize=7.0)
            if i == len(rows) - 1:
                ax.set_xlabel('environment steps')

    handles = [Line2D([], [], color=CONDITION_STYLE[c]['colour'],
                      label=CONDITION_STYLE[c]['name'])
               for c in ('scratch', 'transfer')]
    multi = {c for spec in rows if len(spec['columns']) > 1
             for c in spec['columns']}
    for col, dashes in dash_by_col.items():
        if col in multi:            # a dash pattern only means something where
            handles.append(Line2D(  # one panel carries more than one signal
                [], [], color=_GREY,
                dashes=dashes if dashes[0] else (10, 0), label=col))
    if drew_boundary:
        handles.append(Line2D([], [], color=_GREY, dashes=(3, 2),
                              label='freeze window ends (vertical rule)'))
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    _legend(fig, handles, ncol=4)

    gap = ('' if stream_cols else
           ' Per-stream gradient norms are NOT in the pinned curves schema, so '
           'the gradient row falls back to the global norm; the per-stream '
           'quantities that DESIGN.md 5.5 asks for across the freeze boundary '
           'survive only as end-of-run scalars in per_seed.csv.')

    emit(ctx, 'diagnostics', fig,
         body=('Mechanism instrumentation against environment steps, transfer '
               'against scratch, with the freeze boundary marked where a run '
               'left the window. These are measured on the evaluation cadence '
               'from a fixed diagnostic state batch drawn once per run from a '
               'dedicated RNG stream, so a diagnostic cannot perturb training '
               'and the same states are used at every measurement point '
               '(DESIGN.md 5.5, 8.1). They exist to license or refuse '
               'mechanism wording, never to assert a mechanism: DESIGN.md 9 '
               'requires a mechanism claim to cite an instrumented signal, and '
               'a signal moving is not itself a claim. Value/advantage '
               'magnitudes are undefined for the mlp cells and those panels '
               'say so rather than showing an empty axis.' + gap),
         seeds=sorted(seeds), n_min=n_min,
         protocol=ctx.protocol(ctx.run_dirs_for_labels(labels_used)),
         refs=ctx.references(ctx.run_dirs_for_labels(labels_used)),
         methods=log,
         interval=(f'shaded band = 95% percentile bootstrap over seeds, '
                   f'{ctx.n_boot} resamples, seed {ctx.boot_seed}; suppressed '
                   f'per arm where n < {stats.MIN_N_FOR_INFERENCE}'),
         extra_lines=[
             'Estimation-only, no p-value: mechanism signals are not in the '
             'confirmatory family (ANALYSIS_PLAN.md 1).',
             'The y axis is shared across each row, so curve heights are '
             'comparable between the cell panels of one signal. It is not '
             'shared between rows, which measure different quantities, and the '
             'gradient row is log-scaled.'],
         meta=meta)


# ---------------------------------------------------------------------------
# 14. Figure 8 -- performance profiles
# ---------------------------------------------------------------------------
def fig_performance_profiles(ctx: Context) -> None:
    endpoint = 'final_score'
    cells = [c for c in CELL_ORDER if c in ctx.arms]
    fig, axes = plt.subplots(2, 2, figsize=(FULL_WIDTH, 4.2), sharex=True,
                             sharey=True)
    axes = axes.ravel()
    seeds: set[int] = set()
    n_min = None
    meta: dict[str, Any] = {'endpoint': endpoint, 'cells': {}}
    labels_used = [lbl for cell in cells
                   for lbl in ctx.arms[cell].values()]

    values: dict[tuple[str, str], np.ndarray] = {}
    seeds_by_arm: dict[tuple[str, str], list[int]] = {}
    # `dupes` used to be built here and then never read, so the forest recorded
    # duplicated arm rows while this figure reported n=3 with no duplicate
    # record at all: the same table, two different accounts of what it holds.
    lost: dict[str, list[str]] = {'duplicated arm rows': []}
    for cell in cells:
        for cond, label in ctx.arms.get(cell, {}).items():
            dupes: list[str] = []
            s = _seed_series(ctx, label, endpoint, dupes)
            if dupes:
                lost['duplicated arm rows'].append(
                    f"{cell}/{cond} {', '.join(sorted(set(dupes)))}")
            if not s.empty:
                values[(cell, cond)] = s.to_numpy(float)
                seeds_by_arm[(cell, cond)] = sorted(int(i) for i in s.index)
                seeds.update(int(i) for i in s.index)
    if values:
        all_v = np.concatenate(list(values.values()))
        lo = float(np.min(all_v)) - 0.02
        hi = float(np.max(all_v)) + 0.02
        taus = np.linspace(lo, hi, 400)
    else:
        taus = np.linspace(0.0, 1.0, 2)

    drawn_refs: set[float] = set()
    for ax, cell in zip(axes, cells):
        drew = False
        rec = {}
        for cond in CONTRAST_ORDER:
            v = values.get((cell, cond))
            if v is None or len(v) == 0:
                continue
            n_min = len(v) if n_min is None else min(n_min, len(v))
            style = CONDITION_STYLE[cond]
            frac = np.array([float(np.mean(v > t)) for t in taus])
            line, = ax.plot(taus, frac, color=style['colour'],
                            label=style['name'])
            if style['dashes'][0] is not None:
                line.set_dashes(style['dashes'])
            ax.plot(v, np.full(len(v), -0.045), marker='|', linestyle='none',
                    color=style['colour'], markersize=4.0,
                    markeredgewidth=0.8, clip_on=False)
            drew = True
            rec[cond] = {'n': int(len(v)),
                         'seeds': seeds_by_arm.get((cell, cond), []),
                         'min': float(np.min(v)),
                         'median': float(np.median(v)),
                         'max': float(np.max(v)),
                         'fraction_above_0': float(np.mean(v > 0.0)),
                         'fraction_above_1': float(np.mean(v > 1.0))}
        gone = absent_notes(
            ctx, [ctx.arms.get(cell, {}).get(c) for c in CONTRAST_ORDER
                  if c in ctx.arms.get(cell, {}) and c not in rec],
            fallback=f'has rows but no finite {endpoint}')
        if not drew:
            _no_data(ax, 'no runs for this cell'
                     + (chr(10) + '; '.join(gone) if gone else ''))
        else:
            # The reference lines are drawn where they fall inside the view and
            # ANNOTATED where they do not. `axvline(0.0)` on an autoscaled axis
            # dragged the x limits down to 0 on a normalised scale where every
            # arm scores above 1: the informative range, 1.02 to 1.18, occupied
            # 13% of the panel width and the four condition profiles were four
            # indistinguishable vertical steps. Keeping the reference and
            # losing the figure is the wrong trade; the reference is kept as a
            # statement instead.
            off: list[str] = []
            for value, name, dashes in ((0.0, 'score 0 (random policy)',
                                         (10, 0)),
                                        (1.0, 'score 1 (threshold)',
                                         (1, 2))):
                if taus[0] <= value <= taus[-1]:
                    ax.axvline(value, color=_GREY, linewidth=0.5,
                               dashes=dashes)
                    drawn_refs.add(value)
                else:
                    off.append(name)
            if off:
                off = [' and '.join(off) + f' off-scale, left of tau='
                                           f'{taus[0]:.2f}']
            note_absent(ax, gone + off, where=(0.03, 0.97), va='top')
            ax.set_xlim(taus[0], taus[-1])
            ax.set_ylim(-0.08, 1.04)
        ax.set_title(cell.replace('-', ' / '))
        meta['cells'][cell] = rec
        meta.setdefault('series_absent', {})[cell] = gone
    for ax in axes[len(cells):]:
        _no_data(ax, 'cell not present in the supplied table')
    for ax in axes[2:]:
        ax.set_xlabel(f'score threshold tau ({endpoint})')
    for ax in (axes[0], axes[2]):
        ax.set_ylabel('fraction of runs > tau')

    handles = [Line2D([], [], color=CONDITION_STYLE[c]['colour'],
                      dashes=CONDITION_STYLE[c]['dashes']
                      if CONDITION_STYLE[c]['dashes'][0] else (10, 0),
                      label=CONDITION_STYLE[c]['name'])
               for c in CONTRAST_ORDER
               if any((cell, c) in values for cell in cells)]
    # Never advertise a line that is not on the canvas, the discipline
    # `drew_boundary` applies in the curve figure.
    if 1.0 in drawn_refs:
        handles.append(Line2D([], [], color=_GREY, linewidth=0.5,
                              dashes=(1, 2),
                              label='score 1 = registered threshold'))
    if 0.0 in drawn_refs:
        handles.append(Line2D([], [], color=_GREY, linewidth=0.5,
                              label='score 0 = random policy'))
    meta['reference_lines_drawn'] = sorted(drawn_refs)
    meta['tau_range'] = [float(taus[0]), float(taus[-1])]
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    _legend(fig, handles, ncol=5)

    emit(ctx, 'performance_profiles', fig,
         body=('Run-score distributions per condition: for every threshold tau '
               'on the x-axis, the fraction of that arm\'s runs whose final '
               'normalised score exceeds tau. The whole distribution is shown '
               'rather than a mean with an error bar, because LunarLander '
               'returns are structurally bimodal -- a run either lands or '
               'crashes -- so a central tendency hides the shape that matters '
               'and is one reason ANALYSIS_PLAN.md 8 forbids normality-assuming '
               'summaries here. The x axis spans the observed scores '
               f'({taus[0]:.3f} to {taus[-1]:.3f}) and is NOT extended to '
               'include 0 or 1: the reference lines are drawn where they fall '
               'inside that span and named on the panel where they do not, '
               'because forcing 0 into view on a normalised scale whose arms '
               'all score above 1 compressed every curve into a thirteenth of '
               'the panel. '
               'Ticks below the axis are the individual runs, '
               'so the step size of each curve is visibly 1/n. A curve to the '
               'right of another dominates it at every threshold; crossing '
               'curves mean no arm dominates, which a difference in means '
               'would conceal.'),
         seeds=sorted(seeds), n_min=n_min,
         protocol=ctx.protocol(ctx.run_dirs_for_labels(labels_used)),
         refs=ctx.references(ctx.run_dirs_for_labels(labels_used)),
         dropped={k: '; '.join(v) for k, v in lost.items()},
         interval=('none -- this is an empirical distribution function, drawn '
                   'without a confidence band because at these seed counts a '
                   'band would be wider than the distance between the arms and '
                   'would suggest a precision the design does not have'),
         smoothing='smoothing window: none (endpoint scalars, not curves)',
         meta=meta)


# ---------------------------------------------------------------------------
# 15. Figure 9 -- Kaplan-Meier for the threshold-reaching time
# ---------------------------------------------------------------------------
def _km_arm(rows: pd.DataFrame, tcol: str, ccol: str,
            lost: dict[str, list[str]], where: str
            ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Event times and event flags for one arm, with nothing imputed and
    nothing dropped quietly.

    Two `ANALYSIS_PLAN.md` §5 rules meet here, and the previous version broke
    both in the same three lines.

    **"Never impute the budget."** `e = ~_as_bool(rows[ccol])` mapped every
    unrecognised token, NaN included, to False and therefore to an OBSERVED
    EVENT. A missing censoring flag turned three runs that never reached the
    threshold into 3/3 reached, with a Clopper-Pearson interval of [0.29, 1.00]
    and a Kaplan-Meier median, silently. An unreadable flag is now treated as
    censored, which is the direction that claims nothing (a censored
    observation contributes time at risk and no event), and the count is
    reported in the panel annotation, the caption and the provenance so that a
    reader can see the measurement failed rather than reading an invented
    event.

    **"Never drop censored runs."** `dropna(subset=[tcol])` deleted them, which
    conditions on the outcome and is the silent-seed-dropping defect again: a
    whole scratch arm vanished from a panel with no annotation anywhere.
    `aggregate.py` produces a missing time only when a censored run's own
    `total_env_steps` is missing, so the run's `env_steps` is the censoring time
    it should have carried, and using it is not imputing an event: the run is
    still recorded as not having reached the threshold. Where even that is
    unavailable the run cannot be placed on a time axis at all, so it is
    excluded, counted, named on the panel and named in the caption. Excluded
    loudly is the only honest option left; excluded quietly is the defect.
    """
    # n and k in this figure are SEED counts, and this is the line that makes
    # them so. `arm_rows` filters on `run_dir in ctx.selected`, which keeps both
    # rows of a duplicated (label, seed) because they share a path, and this is
    # the only figure that reads those rows without going through
    # `_rows_to_series`. One duplicated row therefore moved a Clopper-Pearson
    # interval from [0.025, 1.00] at k/n = 1/1 to [0.158, 1.00] at 2/2 on three
    # panels, with no new data anywhere and nothing on the curve to show it,
    # and duplicating a whole two-seed tree reported n=4 in every panel while
    # the caption below it said "n=2 distinct seeds plotted" and the
    # `PIPELINE VALIDATION` stamp had gone.
    seed_dupes: list[str] = []
    rows = one_row_per_seed(rows, where, seed_dupes)
    times = pd.to_numeric(rows[tcol], errors='coerce')
    if ccol in rows.columns:
        censored, known = _bool_tokens(rows[ccol])
    else:
        censored = np.zeros(len(rows), dtype=bool)
        known = np.zeros(len(rows), dtype=bool)
    events = ~censored
    info = {'flag_unreadable': int((~known).sum()),
            'time_recovered_from_env_steps': 0,
            'excluded_no_time': 0, 'excluded_seeds': [],
            'distinct_seeds': int(pd.to_numeric(rows['seed'], errors='coerce')
                                  .nunique()) if 'seed' in rows.columns else 0,
            'unit': 'distinct seeds, not rows',
            'duplicate_arm_rows_dropped': sorted(set(seed_dupes))}
    if seed_dupes:
        lost['rows dropped as a second row for one (label, seed)'] \
            .append(f'{where} {", ".join(sorted(set(seed_dupes)))}')
    if info['flag_unreadable']:
        events = np.where(known, events, False)
        lost['runs with an unreadable censoring flag, counted as censored'] \
            .append(f"{where} x{info['flag_unreadable']}")

    t = times.to_numpy(float)
    need = ~np.isfinite(t)
    if bool(need.any()) and 'env_steps' in rows.columns:
        fallback = pd.to_numeric(rows['env_steps'], errors='coerce') \
            .to_numpy(float)
        usable = need & np.isfinite(fallback)
        if bool(usable.any()):
            t = np.where(usable, fallback, t)
            events = np.where(usable, False, events)
            info['time_recovered_from_env_steps'] = int(usable.sum())
            lost['censored runs whose time came from env_steps'].append(
                f'{where} x{int(usable.sum())}')
        need = ~np.isfinite(t)
    if bool(need.any()):
        info['excluded_no_time'] = int(need.sum())
        info['excluded_seeds'] = sorted(
            int(s) for s in pd.to_numeric(rows.loc[need, 'seed'],
                                          errors='coerce').dropna())
        lost['runs with no usable time or censoring time, EXCLUDED'].append(
            f"{where} seeds {_seed_list(info['excluded_seeds'])}")
    keep = np.isfinite(t)
    return t[keep], np.asarray(events, dtype=bool)[keep], info


def fig_km_threshold(ctx: Context) -> None:
    cells = [c for c in CELL_ORDER if c in ctx.arms]
    levels = [(tag, value) for tag, value in stats.THRESHOLD_LEVELS
              if f'steps_to_threshold_{tag}' in ctx.per_seed.columns]
    if not levels or not cells:
        print(f'{WARN} km_threshold: no steps_to_threshold columns; skipped')
        return
    fig, axes = plt.subplots(len(levels), len(cells),
                             figsize=(FULL_WIDTH, 1.5 * len(levels) + 1.0),
                             sharex='col', sharey=True, squeeze=False)
    seeds: set[int] = set()
    n_min = None
    meta: dict[str, Any] = {'levels': {}, 'logrank': {}}
    lost: dict[str, list[str]] = {
        'runs with an unreadable censoring flag, counted as censored': [],
        'censored runs whose time came from env_steps': [],
        'rows dropped as a second row for one (label, seed)': [],
        'runs with no usable time or censoring time, EXCLUDED': []}
    labels_used = [lbl for cell in cells for lbl in
                   (ctx.arms[cell].get('scratch'),
                    ctx.arms[cell].get('transfer')) if lbl]

    for i, (tag, value) in enumerate(levels):
        tcol, ccol = f'steps_to_threshold_{tag}', f'censored_{tag}'
        for j, cell in enumerate(cells):
            ax = axes[i][j]
            drew = False
            arms_data = {}
            panel_lost: list[str] = []
            for cond in ('scratch', 'transfer'):
                label = ctx.arms.get(cell, {}).get(cond)
                if not label:
                    continue
                rows = arm_rows(ctx, label)
                if rows.empty:
                    continue
                t, e, info = _km_arm(rows, tcol, ccol, lost,
                                     f'{tag}/{cell}/{cond}')
                if info['excluded_no_time']:
                    # Said on the panel, not only in the caption: an arm that
                    # vanishes from a curve is exactly what "no censored run is
                    # dropped" must not be able to hide.
                    panel_lost.append(
                        f"{CONDITION_CODE[cond]}: {info['excluded_no_time']} "
                        f"run(s) had no usable time, excluded")
                if len(t) == 0:
                    continue
                seeds.update(int(s) for s in
                             pd.to_numeric(rows['seed'], errors='coerce')
                             .dropna().tolist())
                n_min = len(t) if n_min is None else min(n_min, len(t))
                km = stats.kaplan_meier(t, e)
                k = int(e.sum())
                cp = stats.clopper_pearson(k, len(t))
                arms_data[cond] = {'t': t, 'e': e, 'km': km, 'k': k,
                                   'n': len(t), 'cp': cp, 'info': info}
                style = CONDITION_STYLE[cond]
                xs = [0.0] + [row['t'] for row in km['curve']]
                ys = [0.0] + [1.0 - row['survival'] for row in km['curve']]
                # The two arms coincide whenever neither reaches the
                # threshold, so the baseline is drawn wider: two identical
                # curves must not look like one arm.
                line, = ax.step(xs, ys, where='post', color=style['colour'],
                                label=style['name'],
                                linewidth=2.0 if cond == 'scratch' else 1.1)
                if style['dashes'][0] is not None:
                    line.set_dashes(style['dashes'])
                cx = [row['t'] for row in km['curve'] if row['censored'] > 0]
                cy = [1.0 - row['survival'] for row in km['curve']
                      if row['censored'] > 0]
                if cx:
                    ax.plot(cx, cy, marker='|', linestyle='none',
                            color=style['colour'], markersize=5.5,
                            markeredgewidth=1.0)
                drew = True
            if drew:
                notes = chr(10).join(
                    f"{CONDITION_STYLE[c]['name'].split(' ')[0]} "
                    f"{d['k']}/{d['n']} reached, 95% CI "
                    f"[{d['cp'][0]:.2f}, {d['cp'][1]:.2f}]"
                    + (f" (+{d['info']['excluded_no_time']} with no time)"
                       if d['info']['excluded_no_time'] else '')
                    + (f" (!{d['info']['flag_unreadable']} flag unreadable)"
                       if d['info']['flag_unreadable'] else '')
                    for c, d in arms_data.items())
                if panel_lost:
                    notes += chr(10) + chr(10).join(panel_lost)
                ax.annotate(notes, xy=(0.03, 0.97), xycoords='axes fraction',
                            fontsize=5.4, color='#333333', va='top')
                ax.set_ylim(-0.05, 1.05)
            else:
                _no_data(ax, 'no threshold times for this cell'
                             + (chr(10) + chr(10).join(panel_lost)
                                if panel_lost else ''))
            if 'scratch' in arms_data and 'transfer' in arms_data:
                a, b = arms_data['scratch'], arms_data['transfer']
                lr = stats.logrank_statistic(a['t'], a['e'], b['t'], b['e'])
                meta['logrank'][f'{tag}:{cell}'] = lr
            meta['levels'].setdefault(tag, {})[cell] = {
                c: {'events': d['k'], 'n': d['n'],
                    'n_unit': 'distinct seeds placed on the time axis, '
                              'not rows',
                    'distinct_seeds_in_arm': d['info']['distinct_seeds'],
                    'n_is_a_seed_count': bool(
                        d['n'] <= d['info']['distinct_seeds']),
                    'p_reached': d['k'] / d['n'] if d['n'] else None,
                    'clopper_pearson': list(d['cp']),
                    'km_median': d['km']['median'],
                    'censoring': d['info']}
                for c, d in arms_data.items()}
            for c, d in arms_data.items():
                # The invariant, checked rather than assumed: a run that could
                # not be placed on the axis LOWERS n, so n can never exceed the
                # arm's distinct seed count. If it ever does, a row count has
                # leaked back into the binomial and the interval beside it is
                # too narrow.
                if d['n'] > d['info']['distinct_seeds']:
                    print(f'{WARN} km_threshold {tag}/{cell}/{c}: n={d["n"]} '
                          f'exceeds the arm\'s '
                          f'{d["info"]["distinct_seeds"]} distinct seed(s), '
                          f'so the Clopper-Pearson interval on this panel is '
                          f'computed over rows and is too narrow. This is the '
                          f'defect ANALYSIS_PLAN.md 9 stamps figures for.')
            if i == 0:
                ax.set_title(cell.replace('-', ' / '))
            if j == 0:
                ax.set_ylabel(f'reached score {value:g}', fontsize=7.0)
            if i == len(levels) - 1:
                ax.set_xlabel('environment steps')

    handles = [Line2D([], [], color=CONDITION_STYLE[c]['colour'],
                      dashes=CONDITION_STYLE[c]['dashes']
                      if CONDITION_STYLE[c]['dashes'][0] else (10, 0),
                      label=CONDITION_STYLE[c]['name'])
               for c in ('scratch', 'transfer')]
    handles.append(Line2D([], [], color=_GREY, marker='|', linestyle='none',
                          label='censored (budget reached first)'))
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    _legend(fig, handles, ncol=3)

    for key, entries in lost.items():
        if entries:
            print(f'{WARN} km_threshold, {key}: ' + '; '.join(entries))

    total_events = sum(
        rec.get(c, {}).get('events', 0)
        for per_cell in meta['levels'].values()
        for rec in per_cell.values() for c in ('scratch', 'transfer'))
    computed_lr = sum(1 for v in meta['logrank'].values()
                      if v.get('statistic') is not None)

    emit(ctx, 'km_threshold', fig,
         body=('Kaplan-Meier cumulative incidence of reaching each '
               'pre-declared normalised score -- the fraction of runs that '
               'have reached it by a given env step -- with censoring marks. '
               'steps_to_threshold is right-censored at the training budget, '
               'and the censoring is administrative: the same budget for every '
               'run, independent of the event time by construction. The budget '
               'is never imputed as an event time, which would bias the '
               'estimate and create a tie mass that degrades every rank '
               'statistic downstream, and no censored run is dropped, which '
               'would condition on the outcome (ANALYSIS_PLAN.md 5). The '
               'annotation in each panel is the primary summary: '
               'P(reached within budget) as k/n with an exact Clopper-Pearson '
               '95% interval, which at 0/10 is the informative statement that '
               'the probability is below about 0.31. Thresholds are '
               'pre-declared at 0.25, 0.5 and 1.0 so a metric exists even when '
               f'no run reaches "solved"; {total_events} event(s) are observed '
               'in total in this table. A run whose censoring flag will not '
               'parse is counted as CENSORED, never as an event: the flag is '
               'the only evidence that the threshold was reached, and reading '
               'a missing flag as an event manufactures the outcome. A '
               'censored run with no recorded time takes its own env-step '
               'total as its censoring time, which is a time at risk and not '
               'an imputed event; where even that is missing the run cannot be '
               'placed on the axis, and it is excluded and named rather than '
               'dropped quietly.'),
         seeds=sorted(seeds), n_min=n_min,
         protocol=ctx.protocol(ctx.run_dirs_for_labels(labels_used)),
         refs=ctx.references(ctx.run_dirs_for_labels(labels_used)),
         dropped={k: '; '.join(v) for k, v in lost.items()},
         interval=('Kaplan-Meier step estimate (stats.kaplan_meier); the '
                   'annotated interval is the exact Clopper-Pearson binomial '
                   'interval on P(reached), not a bootstrap'),
         smoothing='smoothing window: none (step function, drawn as measured)',
         extra_lines=[
             f'Log-rank comparison: computed for {computed_lr} of '
             f'{len(meta["logrank"])} panel(s). ANALYSIS_PLAN.md 5 licenses it '
             f'only when both arms have at least '
             f'{stats.LOGRANK_MIN_EVENTS} events; where it is computed, '
             f'stats.py emits the statistic without a p-value, and none is '
             f'drawn here.'],
         meta=meta)


# ---------------------------------------------------------------------------
# 16. Registry of figures, the multiplicity ledger, and the CLI
# ---------------------------------------------------------------------------
FIGURES: dict[str, Callable[[Context], None]] = {
    'learning_curves': fig_learning_curves,
    'transfer_effect_forest': fig_transfer_effect_forest,
    'control_decomposition': fig_control_decomposition,
    'interaction_2x2': fig_interaction_2x2,
    'shift_gradient': fig_shift_gradient,
    'freeze_duration': fig_freeze_duration,
    'diagnostics': fig_diagnostics,
    'performance_profiles': fig_performance_profiles,
    'km_threshold': fig_km_threshold,
}

ALIASES: dict[str, str] = {
    'learning': 'learning_curves',
    'curves': 'learning_curves',
    'forest': 'transfer_effect_forest',
    'controls': 'control_decomposition',
    'decomposition': 'control_decomposition',
    'interaction': 'interaction_2x2',
    'shift': 'shift_gradient',
    'freeze': 'freeze_duration',
    'diag': 'diagnostics',
    'profiles': 'performance_profiles',
    'km': 'km_threshold',
}


def print_ledger(n_figures: int) -> None:
    """`ANALYSIS_PLAN.md` §7 asks for the ledger on every invocation, so the
    count is a recorded fact rather than a claim. For this module the count is
    zero by construction, and printing a zero is the point."""
    print('\nMultiplicity ledger (ANALYSIS_PLAN.md 7)')
    print(f'  family: CONFIRMATORY -- members: '
          f'{stats.CONFIRMATORY_FAMILY_SIZE} '
          f'(4 cells x {len(stats.CONFIRMATORY_ENDPOINTS)} co-primary '
          f'endpoints: {", ".join(stats.CONFIRMATORY_ENDPOINTS)})')
    print(f'  procedure: Holm-Bonferroni step-down from alpha/m = '
          f'{stats.ALPHA_STRICTEST:.5f} (alpha = {stats.ALPHA})')
    print('  family: SCREENS (E3-E8, E12) -- Benjamini-Hochberg q, orientation '
          'only, no assertion permitted')
    print('  family: EVERYTHING ELSE -- estimation-only, no p-value emitted')
    print(f'  figures drawn by this invocation: {n_figures}')
    print(f'  analyses in this output carrying a p-value: 0 of {n_figures} -- '
          f'plots.py draws intervals and distributions only; the confirmatory '
          f'tests live in stats.py')


# ---------------------------------------------------------------------------
# 16b. Self-test.
#
# This module had no test of any kind: 3756 lines producing nine paper figures,
# with `validate.py` recording it as an explicit residual ("the per-figure and
# per-table provenance sidecars are written by plots.py and tables.py and are
# not produced or inspected here") and `tables.py` next door carrying a
# self-test that has repeatedly caught real defects. Three of the findings this
# file was corrected for -- rows counted as seeds in the Kaplan-Meier panels,
# the survivor of a duplicated row decided by CSV order, and a removed arm
# reported as unmatched seeds -- are each the kind a single fixture catches, and
# one of them has a direct analogue that IS tested for `stats.py`
# (`validate.test_duplicate_unit_moves_no_number`).
#
# Deliberately cheap: fixtures in memory, no matplotlib rendering, no run tree,
# so it can be run on every edit. It tests the SELECTION and COUNTING rules,
# which is where a figure lies about the data; it does not test the drawing.
# ---------------------------------------------------------------------------
def _probe_frame(dupe: bool = False, seeds: Sequence[int] = (0, 1)
                 ) -> pd.DataFrame:
    """A minimal per-seed table: one scratch arm and one transfer arm."""
    rows = []
    for i, seed in enumerate(seeds):
        for label, cond in (('scratch-x', 'scratch'), ('transfer-x',
                                                       'transfer')):
            rows.append({
                'run_dir': f'runs/{cond}/aaa/s{seed}', 'label': label,
                'arm': label, 'cell': 'x', 'condition': cond,
                'env': 'LunarLander-v3', 'seed': seed, 'seed_block': 'CONFIRM',
                'final_score': 1.0 + 0.1 * i, 'auc_score': 0.9 + 0.1 * i,
                'source_valid': None if cond == 'scratch' else True,
                'source_final_score': None if cond == 'scratch' else 0.8,
                'steps_to_threshold_p50': 1000.0 * (i + 1),
                'censored_p50': False, 'env_steps': 200000,
            })
    df = pd.DataFrame(rows)
    if dupe:
        # The same row twice at the same path: one run recorded twice, with the
        # metric moved so that counting both is visible in every statistic.
        extra = df.iloc[[0]].copy()
        extra['final_score'] = float(df.at[0, 'final_score']) * 0.5
        extra['steps_to_threshold_p50'] = float(
            df.at[0, 'steps_to_threshold_p50']) * 0.5
        df = pd.concat([df, extra], ignore_index=True)
    return df


def self_test(verbose: bool = True) -> int:
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = '') -> None:
        checks.append((name, bool(ok), detail))

    # --- the de-duplication rule does not depend on file order ------------
    frame = _probe_frame(dupe=True)
    reversed_frame = frame.iloc[::-1].reset_index(drop=True)
    key_a = sorted(row_content_key(frame).tolist())
    key_b = sorted(row_content_key(reversed_frame).tolist())
    check('the content key is a function of the rows, not of their order',
          key_a == key_b, f'{len(key_a)} keys')
    check('two rows that differ in one field get different content keys',
          row_content_key(frame).nunique() == len(frame),
          f'{row_content_key(frame).nunique()} distinct of {len(frame)}')
    d1: list[str] = []
    d2: list[str] = []
    keep_a = one_row_per_seed(frame[frame['label'] == 'scratch-x'],
                              'scratch-x', d1)
    keep_b = one_row_per_seed(
        reversed_frame[reversed_frame['label'] == 'scratch-x'], 'scratch-x', d2)
    check('the survivor of a duplicated row is the same in either file order',
          (sorted(keep_a['final_score'].tolist())
           == sorted(keep_b['final_score'].tolist())),
          f"{sorted(keep_a['final_score'].tolist())} vs "
          f"{sorted(keep_b['final_score'].tolist())}")
    check('a duplicated (label, seed) leaves one row per seed, and is named',
          len(keep_a) == keep_a['seed'].nunique() and d1 == ['scratch-x@s0'],
          f'{len(keep_a)} rows, {keep_a["seed"].nunique()} seeds, {d1}')
    clean = _probe_frame()
    d3: list[str] = []
    check('a clean arm is returned unchanged and nothing is recorded',
          len(one_row_per_seed(clean[clean['label'] == 'scratch-x'],
                               'scratch-x', d3)) == 2 and d3 == [], str(d3))

    # --- the analysis set counts seeds, not rows --------------------------
    kept, rec = analysis_set(frame)
    dup_rec = (rec.get('removed') or {}).get('duplicate_arm_rows') or {}
    check('the analysis set removes the second row of a duplicated pair',
          len(kept) == 4 and dup_rec.get('n') == 1,
          f'{len(kept)} rows kept, removal record {dup_rec.get("n")}')
    check('the removal names the arm and cites DESIGN.md 8.2',
          dup_rec.get('labels') == ['scratch-x']
          and '8.2' in str(dup_rec.get('reason')),
          str(dup_rec.get('labels')))
    check('the caption sentence states the duplicated-row removal',
          'one run recorded twice' in analysis_set_sentence(rec).lower()
          or 'recorded twice' in analysis_set_sentence(rec).lower(),
          analysis_set_sentence(rec)[-200:])
    _kept_clean, rec_clean = analysis_set(clean)
    check('a clean table loses nothing to any filter',
          not (rec_clean.get('removed') or {}),
          str(sorted((rec_clean.get('removed') or {}))))

    # --- the Kaplan-Meier arm counts seeds --------------------------------
    lost: dict[str, list[str]] = {
        'runs with an unreadable censoring flag, counted as censored': [],
        'censored runs whose time came from env_steps': [],
        'rows dropped as a second row for one (label, seed)': [],
        'runs with no usable time or censoring time, EXCLUDED': []}
    arm_dupe = frame[frame['label'] == 'scratch-x']
    arm_clean = clean[clean['label'] == 'scratch-x']
    t_d, e_d, info_d = _km_arm(arm_dupe, 'steps_to_threshold_p50',
                               'censored_p50', lost, 'probe/dupe')
    t_c, e_c, info_c = _km_arm(arm_clean, 'steps_to_threshold_p50',
                               'censored_p50', dict(lost), 'probe/clean')
    check('a duplicated row does not add an observation to the KM arm',
          len(t_d) == len(t_c) == 2 and int(e_d.sum()) == int(e_c.sum()),
          f'{len(t_d)} vs {len(t_c)} times, {int(e_d.sum())} vs '
          f'{int(e_c.sum())} events')
    check('the KM arm reports n as a seed count and says so',
          info_d['distinct_seeds'] == 2
          and info_d['unit'].startswith('distinct seeds')
          and len(t_d) <= info_d['distinct_seeds'],
          str(info_d['distinct_seeds']))
    check('a duplicated row cannot tighten the Clopper-Pearson interval',
          (stats.clopper_pearson(int(e_d.sum()), len(t_d))
           == stats.clopper_pearson(int(e_c.sum()), len(t_c))),
          f'{stats.clopper_pearson(int(e_d.sum()), len(t_d))}')
    check('the KM arm records which rows it dropped as duplicates',
          info_d['duplicate_arm_rows_dropped'] == ['probe/dupe@s0'],
          str(info_d['duplicate_arm_rows_dropped']))
    # The anchor: the interval this defect used to move really does move, so
    # the check above is testing something.
    check('1/1 and 2/2 are different Clopper-Pearson intervals, so the guard '
          'above is not vacuous',
          stats.clopper_pearson(1, 1) != stats.clopper_pearson(2, 2),
          f'{stats.clopper_pearson(1, 1)} vs {stats.clopper_pearson(2, 2)}')

    # --- a duplicated row is not a collision ------------------------------
    _sel, coll, dups = resolve_selection(frame)
    check('a repeated row is reported as a duplicated row, not as a collision',
          coll == [] and len(dups) == 1 and dups[0]['label'] == 'scratch-x',
          f'{len(coll)} collision(s), {len(dups)} duplicate(s)')
    collide = _probe_frame()
    collide.loc[0, 'run_dir'] = 'runs/scratch/bbb/s0'
    collide = pd.concat([collide, _probe_frame().iloc[[0]]], ignore_index=True)
    _sel2, coll2, dups2 = resolve_selection(collide)
    check('two DIFFERENT directories under one (label, seed) is a collision',
          len(coll2) == 1 and dups2 == [],
          f'{len(coll2)} collision(s), {len(dups2)} duplicate(s)')

    # --- n_min is a seed count -------------------------------------------
    check('n_min above the distinct seed count is caught',
          n_is_a_seed_count(2, [0, 1]) and n_is_a_seed_count(1, [0, 1])
          and not n_is_a_seed_count(4, [0, 1]),
          'n_min=4 over seeds (0, 1)')
    check('a figure with no data is not reported as a leak',
          n_is_a_seed_count(None, []) and n_is_a_seed_count(None, [0]))

    # --- a removed arm is named as removed, not as unmatched --------------
    gated = _probe_frame()
    gated.loc[gated['condition'] == 'transfer', 'source_valid'] = False
    gated.loc[gated['condition'] == 'transfer', 'source_final_score'] = 0.5992
    kept_g, rec_g = analysis_set(gated, 'valid')
    ctx = Context(per_seed=kept_g, curves=pd.DataFrame(),
                  per_seed_path='probe.csv', curves_path=None, outdir='.',
                  formats=('png',), n_boot=1, boot_seed=0, smooth=0,
                  grid_points=8, argv=[], analysis=rec_g)
    ctx.selected, ctx.collisions, ctx.duplicate_rows = \
        resolve_selection(kept_g)
    reason = absent_reason(ctx, 'transfer-x')
    check('an arm removed by the source gate says SO, not "no matched seeds"',
          'removed' in reason and '4.3' in reason
          and 'matched' not in reason, reason)
    check('an arm that is simply not in the table is distinguished from one '
          'that was removed',
          'no run with this label' in absent_reason(ctx, 'never-heard-of-it'),
          absent_reason(ctx, 'never-heard-of-it'))
    check('the scratch arm, which survived the gate, is not reported absent',
          absent_notes(ctx, ['scratch-x']) == ['scratch-x: no usable data for '
                                               'this series here']
          or not arm_rows(ctx, 'scratch-x').empty,
          str(absent_notes(ctx, ['scratch-x'])))
    check('absent_notes distinguishes an undefined arm from a removed one',
          'not defined' in absent_notes(ctx, [None])[0],
          str(absent_notes(ctx, [None])))

    # --- the panel note stays short enough to read ------------------------
    long_entries = [f'arm-{i}: removed: source failed the DESIGN.md 4.3 gate'
                    for i in range(12)]
    fig, ax = plt.subplots()
    note_absent(ax, long_entries)
    texts = [t.get_text() for t in ax.texts]
    plt.close(fig)
    check('a long absent list is truncated and defers to the caption',
          texts and len(texts[0].split(chr(10))) <= ABSENT_NOTE_LINES
          and 'caption' in texts[0], str(len(texts[0].split(chr(10)))))

    # --- the DESIGN.md 3.3 arbitration reaches the captions ---------------
    # ANALYSIS_PLAN.md 2.4 binds this module too. A figure whose caption says
    # nothing about the arbitration can be read as licensing exactly what the
    # results table refuses, and this module drew the whole confirmatory family
    # as a forest with no verdict anywhere in its caption or its sidecar.
    check('the verdict vocabulary is stats.py own, not a second copy',
          ARBITRATION_VERDICTS == tuple(stats.ARBITRATION_VERDICTS)
          and (AGREES, DISAGREES, NOT_EVALUABLE)
          == (stats.AGREES, stats.DISAGREES, stats.NOT_EVALUABLE),
          str(ARBITRATION_VERDICTS))
    check('not-evaluable is the default', ARBITRATION_DEFAULT == NOT_EVALUABLE)

    def _arb_member(verdict=AGREES, asserted=1, cell='mlp-double',
                    metric='final_score', **extra):
        member = {'metric': metric, 'cell': cell,
                  ARBITRATION_KEY: verdict, ASSERTED_KEY: asserted}
        member.update(extra)
        return member

    def _arb_stats(*members, rows=()):
        return {'s5_confirmatory': {'members': list(members),
                                    'arbitration': {'rows': list(rows)}}}

    a = read_arbitration(_arb_member())
    check('agrees plus the asserted flag does not block', not a['blocks'])
    a = read_arbitration(_arb_member(verdict=NOT_EVALUABLE, asserted=0))
    check('not-evaluable blocks and names the tuned leg',
          a['blocks']
          and a['label'] == 'not evaluable: the tuned leg has not been run',
          a['label'])
    a = read_arbitration(_arb_member(verdict=DISAGREES, asserted=0))
    check('disagrees blocks and renders as the finding',
          a['blocks'] and 'THIS IS THE FINDING' in a['label'], a['label'])
    stripped = _arb_member()
    stripped.pop(ARBITRATION_KEY)
    a = read_arbitration(stripped)
    check('a DELETED verdict key blocks and is named',
          a['blocks'] and a['defect'] == 'key-absent'
          and ARBITRATION_KEY in a['label'], str(a))
    a = read_arbitration({})
    check('a member with neither key blocks', read_arbitration({})['blocks'])

    arb_ctx = Context(
        per_seed=_probe_frame(), curves=pd.DataFrame(), per_seed_path='x',
        curves_path=None, outdir='x', formats=('png',), n_boot=10,
        boot_seed=1, smooth=0, grid_points=8, argv=[])
    check('a context with NO stats.json says the arbitration was NOT READ',
          'NOT READ' in arb_ctx.arbitration_sentence()
          and 'not-evaluable the default' in arb_ctx.arbitration_sentence(),
          arb_ctx.arbitration_sentence())
    check('a per-cell lookup with no stats.json still BLOCKS',
          arb_ctx.arbitration_for('final_score', 'mlp-double')['blocks'])

    report = _arb_stats(
        _arb_member(cell='mlp-vanilla'),
        _arb_member(cell='mlp-double', verdict=DISAGREES, asserted=0),
        _arb_member(cell='dueling-vanilla', verdict=NOT_EVALUABLE, asserted=0),
        rows=[{'metric': 'final_score', 'cell': 'mlp-double',
               'verdict': DISAGREES,
               'why': 'the common configuration concludes transfer ABOVE '
                      'scratch; the per-cell tuned configuration concludes '
                      'transfer BELOW scratch'}])
    arb_ctx = Context(
        per_seed=_probe_frame(), curves=pd.DataFrame(), per_seed_path='x',
        curves_path=None, outdir='x', formats=('png',), n_boot=10,
        boot_seed=1, smooth=0, grid_points=8, argv=[],
        arbitration=arbitration_summary(report), _stats_report=report)
    sentence = arb_ctx.arbitration_sentence()
    check('the caption sentence counts what may be asserted',
          '1 of 3 confirmatory member(s) may be asserted' in sentence,
          sentence)
    check('the caption sentence names the disagreement as the finding',
          'DISAGREEMENT' in sentence and 'ABOVE' in sentence, sentence)
    check('a per-cell lookup finds the agreeing member',
          not arb_ctx.arbitration_for('final_score', 'mlp-vanilla')['blocks'])
    check('a per-cell lookup finds the disagreeing member',
          arb_ctx.arbitration_for('final_score', 'mlp-double')['blocks'])
    check('a cell absent from the family blocks rather than passing',
          arb_ctx.arbitration_for('final_score', 'dueling-double')['blocks'])

    ok = all(c[1] for c in checks)
    if verbose:
        for name, passed, detail in checks:
            print(f'  {"PASS" if passed else "FAIL"}  {name}'
                  + (f'   [{detail}]' if detail and not passed else ''))
        print(f'\n{sum(1 for c in checks if c[1])}/{len(checks)} checks passed')
    return 0 if ok else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--per-seed', default=os.path.join('runs',
                                                           'per_seed.csv'),
                        help='per-run table from aggregate.py')
    parser.add_argument('--curves', default=os.path.join('runs', 'curves.csv'),
                        help='long-form per-episode table from aggregate.py')
    parser.add_argument('--outdir', default=os.path.join('paper', 'figures'),
                        help='directory for the figures, captions and '
                             'provenance records')
    parser.add_argument('--format', default='pdf,png',
                        help='comma-separated output formats (default pdf,png)')
    parser.add_argument('--figures', default=None,
                        help='comma-separated figures to draw; names or '
                             'aliases. Default: all. Known: '
                             + ', '.join(sorted(FIGURES)))
    parser.add_argument('--smooth', type=int, default=0,
                        help='trailing smoothing window in EVALUATION POINTS '
                             'for every curve; 0 or 1 means no smoothing. '
                             'Whatever is passed here is written into every '
                             'caption, which is the point (reviewer concern C8)')
    parser.add_argument('--grid-points', type=int, default=200,
                        help='points on the common env-step grid used to '
                             'average curves across seeds (default 200)')
    parser.add_argument('--n-boot', type=int, default=stats.N_BOOT,
                        help=f'bootstrap resamples (default {stats.N_BOOT}, '
                             f'the pre-registered value)')
    parser.add_argument('--boot-seed', type=int, default=stats.BOOT_SEED,
                        help=f'bootstrap seed (default {stats.BOOT_SEED}, the '
                             f'pre-registered value)')
    parser.add_argument('--source-policy', default='valid',
                        choices=list(SOURCE_POLICIES),
                        help='which runs enter every figure (DESIGN.md 4.3). '
                             '"valid" (default, the primary estimand) excludes '
                             'every run whose source failed the normalised '
                             'validity gate; "pooled" is the pre-declared '
                             'secondary and keeps them, stamped on the canvas '
                             'and stated in the caption. The same flag and the '
                             'same meanings as stats.py, so a figure and the '
                             'table beside it cannot disagree about which runs '
                             'exist')
    parser.add_argument('--stats', default=None,
                        help='the stats.py --json report. Read for the '
                             'DESIGN.md 3.3 arbitration ONLY, which every '
                             'caption and every provenance sidecar then '
                             'carries: ANALYSIS_PLAN.md 2.4 binds this module '
                             'as well as report.py and tables.py, and a '
                             'caption silent about the verdict can be read as '
                             'licensing what the results table refuses. No '
                             'number plotted here comes from it. Without the '
                             'flag every caption says the arbitration was not '
                             'read, which blocks rather than omits')
    parser.add_argument('--shift-metrics', default=None,
                        help='JSON mapping environment id -> measured '
                             'divergence, used as the x-axis of '
                             'shift_gradient. Without it the x-axis is the '
                             'declared manipulated level and the caption says '
                             'so')
    parser.add_argument('--self-test', action='store_true',
                        help='run the selection and counting assertions on '
                             'in-memory fixtures and exit. Cheap: no run tree, '
                             'no figure is rendered')
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()

    argv_list = list(argv if argv is not None else sys.argv[1:])
    formats = tuple(f.strip().lstrip('.') for f in args.format.split(',')
                    if f.strip())
    if not formats:
        print(f'{WARN} --format left no output format')
        return 2
    # Checked before the output directory is created, because the previous
    # version raised matplotlib's own ValueError from inside `emit` after
    # `os.makedirs` had already run, which leaves a half-written figure set
    # behind and reports the problem as a traceback rather than as a message.
    supported = sorted(plt.gcf().canvas.get_supported_filetypes())
    plt.close('all')
    unknown = [f for f in formats if f not in supported]
    if unknown:
        print(f'{WARN} --format {unknown}: not a supported output format. '
              f'This matplotlib backend writes: {", ".join(supported)}.')
        return 2
    if int(args.n_boot) < 1:
        # Zero resamples produced NaN band bounds, no band, and a caption that
        # still claimed a "95% percentile bootstrap over seeds, 0 resamples".
        # --grid-points and --smooth have floors; this one had none.
        print(f'{WARN} --n-boot {args.n_boot}: at least 1 resample is '
              f'required, and the pre-registered value is {stats.N_BOOT}. A '
              f'bootstrap with no resamples produces no interval, and a '
              f'caption describing an interval that was never computed is the '
              f'defect this module exists to prevent.')
        return 2
    if int(args.n_boot) != stats.N_BOOT:
        print(f'{WARN} --n-boot {args.n_boot} differs from the pre-registered '
              f'{stats.N_BOOT} (ANALYSIS_PLAN.md 2). Every caption and every '
              f'provenance record states the number actually used.')

    requested = list(FIGURES)
    if args.figures:
        requested = []
        for raw in args.figures.split(','):
            name = raw.strip()
            if not name:
                continue
            name = ALIASES.get(name, name)
            if name not in FIGURES:
                print(f'{WARN} unknown figure {raw.strip()!r}. Known: '
                      + ', '.join(sorted(FIGURES)) + '; aliases: '
                      + ', '.join(sorted(ALIASES)))
                return 2
            requested.append(name)

    if not os.path.isfile(args.per_seed):
        print(f'{WARN} no per-seed table at {args.per_seed}. Produce one with '
              f'`python experiments/aggregate.py --out-root runs`.')
        return 1
    if args.curves and not os.path.isfile(args.curves):
        print(f'{WARN} no curve table at {args.curves}: the curve figures will '
              f'be skipped.')

    raw_per_seed, curves = load(args.per_seed, args.curves)
    per_seed, analysis = analysis_set(raw_per_seed, args.source_policy)

    # The arbitration, read once. An unreadable or wrong-shaped file is NOT
    # treated as an absent one silently: the reason is printed and the captions
    # then say the arbitration was not read, which blocks. Failing open here
    # would put an unqualified interval in front of a reader on the strength of
    # a corrupt JSON file.
    stats_report: Optional[dict] = None
    stats_note = ''
    if args.stats:
        try:
            with open(args.stats, 'r', encoding='utf-8') as fh:
                loaded = json.load(fh)
            if not isinstance(loaded, dict):
                stats_note = (f'--stats {args.stats} does not hold a JSON '
                              f'object')
            elif 's5_confirmatory' not in loaded:
                stats_note = (f'--stats {args.stats} has no "s5_confirmatory" '
                              f'block, so it is not a stats.py report')
            else:
                stats_report = loaded
        except (OSError, ValueError) as exc:
            stats_note = f'--stats {args.stats}: {type(exc).__name__}: {exc}'
        if stats_note:
            print(f'{WARN} {stats_note}. Every caption below will record that '
                  f'the DESIGN.md 3.3 arbitration could NOT be read, which '
                  f'blocks a confirmatory reading rather than omitting the '
                  f'subject (ANALYSIS_PLAN.md 2.4).')

    with plt.rc_context(RC):
        ctx = Context(
            per_seed=per_seed, curves=curves,
            per_seed_path=args.per_seed,
            curves_path=args.curves if os.path.isfile(args.curves) else None,
            outdir=args.outdir, formats=formats,
            n_boot=int(args.n_boot), boot_seed=int(args.boot_seed),
            smooth=max(int(args.smooth), 0),
            grid_points=max(int(args.grid_points), 8),
            argv=argv_list,
            source_policy=args.source_policy, analysis=analysis,
            curve_integrity=curve_integrity(curves),
            arms=resolve_arm_labels(), iface=interface_labels(),
            shifts=shift_labels(),
            stats_path=args.stats,
            stats_sha=(provenance.file_hash(args.stats)
                       if args.stats and os.path.isfile(args.stats) else None),
            arbitration=(arbitration_summary(stats_report)
                         if stats_report is not None else None),
            _stats_report=stats_report,
            prov=provenance.snapshot(['experiments/plots.py'] + argv_list),
            hashes={'per_seed': provenance.file_hash(args.per_seed),
                    'curves': (provenance.file_hash(args.curves)
                               if os.path.isfile(args.curves) else None)})
        ctx.selected, ctx.collisions, ctx.duplicate_rows = \
            resolve_selection(per_seed)
        # plots.py takes no --runs: the run tree is the directory
        # `aggregate.py` wrote per_seed.csv into, which is where the
        # selection artifact lives. Derived rather than added as a flag,
        # so a figure can never be drawn with this switched off.
        ctx.selection_plan = selection_plan_provenance(
            os.path.dirname(os.path.abspath(args.per_seed)),
            (ctx.prov.get('plans') or {}).get('ANALYSIS_PLAN.md'))
        if args.shift_metrics:
            try:
                with open(args.shift_metrics, 'r', encoding='utf-8') as fh:
                    ctx.shift_metrics = {str(k): float(v) for k, v in
                                         json.load(fh).items()}
            except (OSError, ValueError, TypeError) as exc:
                print(f'{WARN} could not read --shift-metrics '
                      f'{args.shift_metrics}: {exc}')
                return 2

        plans = ctx.prov.get('plans') or {}
        table_plans = sorted(set(per_seed['plan_hash'].dropna().tolist())
                             if 'plan_hash' in per_seed.columns else [])
        print(f'plots.py: {len(raw_per_seed)} runs in the table, '
              f'{len(per_seed)} in the analysis set, '
              f'{len(curves)} curve rows, out -> {args.outdir}/')
        print(f'  ANALYSIS_PLAN.md hash now: {plans.get("ANALYSIS_PLAN.md")}')
        print(f'  {ctx.arbitration_sentence()}')
        for item in (ctx.arbitration or {}).get('disagreements') or []:
            print(f'  POLICY DISAGREEMENT, and that is the finding: {item}')
        print_analysis_set(analysis)
        if not len(per_seed):
            print(f'{WARN} the analysis set is EMPTY. Every figure below is '
                  f'drawn with no data and stamped '
                  f'{stats.VALIDATION_STAMP}; nothing in this output is a '
                  f'result of any kind. If that is unexpected, the removals '
                  f'listed above say why.')
        if ctx.curve_integrity.get('duplicate_rows'):
            print(f'{WARN} the curve table has '
                  f'{ctx.curve_integrity["duplicate_rows"]} duplicated '
                  f'(run_dir, episode) row(s) across '
                  f'{len(ctx.curve_integrity["runs_affected"])} run(s). '
                  f'DESIGN.md 8.2 names this as the corruption two writers '
                  f'into one directory produce. They are dropped '
                  f'deterministically before any smoothing, but no window '
                  f'statistic from this table is trustworthy.')
        multi_seed = seeds_per_run(curves)
        if multi_seed:
            print(f'{WARN} {len(multi_seed)} run(s) in the curve table carry '
                  f'more than one seed: {sorted(multi_seed)[:4]}. A run '
                  f'directory names one run, so the seed set stated in every '
                  f'curve caption is unreliable for this table.')
        if ctx.selection_plan_drift:
            print(f'{WARN} ' + str(ctx.selection_plan.get('note')
                                   or ctx.selection_plan_sentence()))
        if len(table_plans) > 1:
            print(f'{WARN} the runs in this table were produced under '
                  f'{len(table_plans)} different ANALYSIS_PLAN.md hashes '
                  f'{table_plans}: a confirmatory result may not be reported '
                  f'across pre-registrations (ANALYSIS_PLAN.md 1). Figures are '
                  f'still drawn, and every caption carries the current hash.')
        elif table_plans and table_plans[0] != plans.get('ANALYSIS_PLAN.md'):
            print(f'{WARN} the runs were produced under ANALYSIS_PLAN.md '
                  f'{table_plans[0]} but the plan is now '
                  f'{plans.get("ANALYSIS_PLAN.md")}: results built from these '
                  f'figures are exploratory until audit.py says otherwise.')
        if 'metrics_contiguous' in per_seed.columns:
            bad = int((~_as_bool(per_seed['metrics_contiguous'])).sum())
            if bad:
                print(f'{WARN} {bad} run(s) in this table failed the '
                      f'metrics-integrity check. They are plotted, because '
                      f'dropping them silently is the defect under repair, but '
                      f'no window statistic from them is trustworthy.')
        if ctx.collisions:
            print()
            print('!' * 72)
            print(f'{WARN} {len(ctx.collisions)} (arm label, seed) pair(s) '
                  f'resolve to MORE THAN ONE run directory. An arm label plus '
                  f'a seed names one run by construction, so this table mixes '
                  f'configurations inside one arm identity -- exactly the '
                  f'collision DESIGN.md 11 exists to prevent. Figures are '
                  f'drawn from the lexicographically first path per pair, '
                  f'deterministically, and every discarded path is in each '
                  f'figure provenance record. audit.py is the enforcement '
                  f'point; a result must not be reported from this table until '
                  f'it passes.')
            fields = sorted({f for c in ctx.collisions
                             for f in c['fields_that_differ']})
            labels = sorted({c['label'] for c in ctx.collisions})
            print(f'  affected arms ({len(labels)}): '
                  + ', '.join(labels[:8])
                  + (f', ... (+{len(labels) - 8} more)' if len(labels) > 8
                     else ''))
            same = ('none: the run directories differ but every compared '
                    'field agrees')
            print('  fields that differ between the colliding runs: '
                  + (', '.join(fields) if fields else same))
            print('!' * 72)
        if ctx.duplicate_rows:
            # Deliberately NOT the collision banner. Nothing here resolved to
            # more than one directory and no configuration is mixed: a row is
            # repeated, which is DESIGN.md 8.2 and not DESIGN.md 11, and the
            # two have different causes and different remedies.
            # This is the second belt. `analysis_set` removes duplicated
            # (label, seed) rows before `resolve_selection` ever sees them and
            # reports the removal in the analysis-set block above, so on the
            # normal path this branch is silent. It fires if a caller ever
            # hands `resolve_selection` an unfiltered table, which is exactly
            # when nothing else would say so.
            n_rows = sum(int(d['rows']) - 1 for d in ctx.duplicate_rows)
            labels = sorted({d['label'] for d in ctx.duplicate_rows})
            print()
            print(f'{WARN} {len(ctx.duplicate_rows)} (arm label, seed) '
                  f'pair(s) carry MORE THAN ONE ROW at ONE run directory: '
                  f'{n_rows} extra row(s) across {len(labels)} arm(s). That '
                  f'is one run recorded twice (DESIGN.md 8.2), not two '
                  f'configurations inside one arm identity (DESIGN.md 11), '
                  f'and it is not two seeds: counting the rows would tighten '
                  f'every interval scaled by n and could lift an arm above '
                  f'the n<{stats.MIN_N_FOR_INFERENCE} stamp without a single '
                  f'new run. One row per seed is kept, chosen by directory '
                  f'then by row content so the choice does not depend on the '
                  f'order of the CSV, and the removal is in every caption and '
                  f'every provenance record.')
            print(f'  affected arms ({len(labels)}): ' + ', '.join(labels[:8])
                  + (f', ... (+{len(labels) - 8} more)' if len(labels) > 8
                     else ''))
        os.makedirs(args.outdir, exist_ok=True)
        for name in requested:
            FIGURES[name](ctx)

    print_ledger(len(requested))
    print(f'\n{len(ctx.written)} file(s) written to {args.outdir}/')
    return 0


if __name__ == '__main__':
    sys.exit(main())
