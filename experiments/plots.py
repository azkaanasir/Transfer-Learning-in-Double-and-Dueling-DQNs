"""Every figure the paper carries, drawn from the two pinned CSVs and nothing else.

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

Four properties are therefore mechanical here, not editorial:

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
* **Provenance travels with every figure** (`DESIGN.md` §8.3). `<name>.provenance.json`
  records the content hash of each input CSV, the git commit and dirty flag,
  the `ANALYSIS_PLAN.md` hash, the exact argv, and the arm labels the figure
  resolved -- so a stale figure is detectable rather than plausible.
* **No inference is invented here.** Every interval comes from `stats.py`'s
  pre-registered estimators (`hodges_lehmann_paired`, `bootstrap_statistic`'s
  bias-corrected bootstrap, `kaplan_meier`, `clopper_pearson`) at its fixed
  `N_BOOT` and `BOOT_SEED`. A figure computing its own CI by its own method
  would eventually disagree with the table beside it, and the reader would have
  no way to tell which was right. **No figure draws a p-value at all**, even
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
    """Format a number for a caption, or an em dash if it is not one."""
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
    seeds = list(seeds)
    if len(seeds) <= limit:
        return ', '.join(str(s) for s in seeds)
    return (', '.join(str(s) for s in seeds[:limit])
            + f', ... (+{len(seeds) - limit} more)')


def _as_bool(series: pd.Series) -> np.ndarray:
    """CSV booleans, which arrive as bool, str or object depending on whether a
    column had any missing values. Never guess: an unrecognised token becomes
    False and is counted by the caller."""
    def one(v: Any) -> bool:
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        if isinstance(v, str):
            return v.strip().lower() in ('true', '1', 'yes')
        try:
            return bool(int(v))
        except (TypeError, ValueError):
            return False
    return np.array([one(v) for v in series.tolist()], dtype=bool)


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


def _no_data(ax: plt.Axes, message: str) -> None:
    """Say that a panel is empty and why, rather than leaving blank axes."""
    ax.text(0.5, 0.5, textwrap.fill(message, 34), transform=ax.transAxes,
            ha='center', va='center', fontsize=6.6, color=_GREY, style='italic')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


def _legend(fig: plt.Figure, handles: list, ncol: int, y: float = 0.0) -> None:
    if handles:
        fig.legend(handles=handles, loc='lower center', ncol=ncol,
                   bbox_to_anchor=(0.5, y))


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
    shift_metrics: dict[str, float] = field(default_factory=dict)
    arms: dict[str, dict[str, str]] = field(default_factory=dict)
    iface: dict[str, dict[str, str]] = field(default_factory=dict)
    shifts: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    prov: dict = field(default_factory=dict)
    hashes: dict = field(default_factory=dict)
    written: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    _manifests: dict[str, dict] = field(default_factory=dict)

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

    def references(self, envs_used: Iterable[str]) -> dict[str, dict]:
        """Normalisation references for the environments in one figure.

        The manifest's own `reference` block is preferred over
        `reference_returns.json`: it is what the plotted scores were actually
        divided by, and a figure must be described by the numbers that made it.
        """
        out: dict[str, dict] = {}
        for env in dict.fromkeys(envs_used):
            if env is None or (isinstance(env, float) and np.isnan(env)):
                continue
            rows = self.per_seed[self.per_seed['env'] == env]
            ref: dict = {}
            for run_dir in rows['run_dir'].tolist():
                block = self.manifest(run_dir).get('reference') or {}
                if block.get('random_return') is not None:
                    ref = {'random_return': block['random_return'],
                           'threshold': block['threshold'],
                           'noop_return': block.get('noop_return'),
                           'origin': 'run manifest'}
                    break
            if not ref:
                try:
                    block = envs.reference(env)
                    ref = {'random_return': block.get('random_return'),
                           'threshold': block.get('threshold'),
                           'noop_return': block.get('noop_return'),
                           'origin': 'reference_returns.json'}
                except Exception:                          # noqa: BLE001
                    ref = {'origin': 'unavailable'}
            out[str(env)] = ref
        return out


def load(per_seed_path: str, curves_path: Optional[str]) -> tuple[pd.DataFrame,
                                                                 pd.DataFrame]:
    per_seed = pd.read_csv(per_seed_path)
    missing = [c for c in ('run_dir', 'label', 'cell', 'condition', 'seed',
                           'final_score', 'auc_score')
               if c not in per_seed.columns]
    if missing:
        raise SystemExit(f'{WARN} {per_seed_path} is not a per_seed table: '
                         f'missing columns {missing}. It is produced by '
                         f'experiments/aggregate.py.')
    if curves_path and os.path.isfile(curves_path):
        curves = pd.read_csv(curves_path)
    else:
        curves = pd.DataFrame(columns=['run_dir', 'cell', 'condition', 'label',
                                       'seed', 'episode', 'env_steps',
                                       'eval_score', 'frozen'])
    return per_seed, curves


# ---------------------------------------------------------------------------
# 4. Selection and pairing
# ---------------------------------------------------------------------------
def arm_rows(ctx: Context, label: str) -> pd.DataFrame:
    return ctx.per_seed[ctx.per_seed['label'] == label]


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


def _seed_series(ctx: Context, label: str, metric: str,
                 duplicates: list[str]) -> pd.Series:
    rows = arm_rows(ctx, label)
    if metric not in rows.columns:
        return pd.Series(dtype=float)
    rows = rows[['seed', metric]].dropna(subset=[metric])
    dupe = rows['seed'].duplicated(keep=False)
    if bool(dupe.any()):
        for seed in sorted(set(rows.loc[dupe, 'seed'].tolist())):
            duplicates.append(f'{label}@s{seed}')
        rows = rows.drop_duplicates(subset=['seed'], keep='first')
    return rows.set_index('seed')[metric].astype(float)


def pair(ctx: Context, base_label: str, treat_label: str,
         metric: str) -> Paired:
    """Matched-seed contrast. Seeds are a blocking factor (`DESIGN.md` §2.4
    RQ2), so an unmatched seed cannot enter the delta -- and it is recorded
    rather than dropped quietly, because silent seed dropping is one of the six
    published defects (`DESIGN.md` §1)."""
    dupes: list[str] = []
    b = _seed_series(ctx, base_label, metric, dupes)
    t = _seed_series(ctx, treat_label, metric, dupes)
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


def estimate_shift(ctx: Context, delta: np.ndarray,
                   idx: Optional[np.ndarray] = None) -> dict:
    """Hodges-Lehmann paired shift with `stats.py`'s bias-corrected bootstrap
    interval. `ANALYSIS_PLAN.md` §2: no mean-with-a-normal-interval anywhere."""
    units = np.asarray(delta, dtype=float).reshape(-1, 1)
    if units.shape[0] == 0:
        return {'estimate': float('nan'), 'lo': float('nan'),
                'hi': float('nan'), 'n': 0, 'method': 'none', 'note': 'no data'}
    out = stats.bootstrap_statistic(
        units, lambda u: stats.hodges_lehmann_paired(u[:, 0]),
        n_boot=ctx.n_boot, seed=ctx.boot_seed, idx=idx)
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


def exclusion_sentence(est: dict) -> str:
    """The only licensed positive statement about a null (`DESIGN.md` §9,
    `ANALYSIS_PLAN.md` §4): what the interval excludes."""
    lo = est.get('lo')
    if lo is None or not np.isfinite(lo):
        return 'no exclusion bound (interval suppressed)'
    if lo >= 0:
        return 'a degradation of any size is excluded at 95%'
    return f'a degradation worse than {abs(lo):.3f} score units is excluded at 95%'


def equivalence_sentence(est: dict, sd: float,
                         margin: float = stats.EQUIVALENCE_MARGIN) -> str:
    """`ANALYSIS_PLAN.md` §4 verbatim in its two branches: an equivalence
    verdict, or the statement that dispersion makes equivalence untestable at
    this n. Never TOST, and never a null read as equivalence."""
    if np.isfinite(sd) and sd > margin:
        return (f'equivalence untestable in this cell at this n '
                f'(across-seed SD {sd:.3f} > margin {margin:.2f})')
    lo, hi = est.get('lo'), est.get('hi')
    if lo is None or not np.isfinite(lo) or not np.isfinite(hi):
        return 'equivalence not assessed (interval suppressed)'
    if lo > -margin and hi < margin:
        return f'CI inside +/-{margin:.2f}: equivalence supported'
    return f'CI not inside +/-{margin:.2f}: equivalence not supported'


# ---------------------------------------------------------------------------
# 5. Curve machinery -- common env-step support, bootstrap bands, the freeze
#    boundary
# ---------------------------------------------------------------------------
def curve_rows(ctx: Context, label: str) -> pd.DataFrame:
    if ctx.curves.empty or 'label' not in ctx.curves.columns:
        return ctx.curves
    return ctx.curves[ctx.curves['label'] == label]


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
    """One row per seed, interpolated onto `grid`. No extrapolation."""
    rows, seeds = [], []
    if frame.empty or column not in frame.columns:
        return np.zeros((0, len(grid))), seeds
    for run_dir, g in frame.dropna(subset=[column]).groupby('run_dir'):
        g = g.sort_values('env_steps')
        x = g['env_steps'].to_numpy(float)
        y = g[column].to_numpy(float)
        if len(x) < 2:
            continue
        y = _trailing_mean(y, ctx.smooth)
        rows.append(np.interp(grid, x, y, left=np.nan, right=np.nan))
        seeds.append(int(g['seed'].iloc[0]))
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
           'max_updates': float('nan')}
    if frame.empty or 'frozen' not in frame.columns:
        return out
    steps, updates_max = [], []
    for _run, g in frame.groupby('run_dir'):
        g = g.sort_values('episode')
        flag = _as_bool(g['frozen'])
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
        out['sd_env_step'] = float(np.std(steps, ddof=1)) if len(steps) > 1 else 0.0
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
def stamp_validation(fig: plt.Figure, n: int) -> None:
    fig.text(0.5, 0.5, stats.VALIDATION_STAMP.split(' - ')[0],
             fontsize=34, color='#D0D0D0', alpha=0.55, ha='center',
             va='center', rotation=24, zorder=0,
             transform=fig.transFigure)
    fig.text(0.5, 0.455, f'n={n} seeds: {stats.VALIDATION_STAMP}',
             fontsize=7.5, color='#9A9A9A', ha='center', va='center',
             rotation=24, zorder=0, transform=fig.transFigure)


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
    for env, ref in refs.items():
        parts.append(f"{env}: random {_f(ref.get('random_return'), 1)}, "
                     f"threshold {_f(ref.get('threshold'), 1)}, no-op "
                     f"{_f(ref.get('noop_return'), 1)} [{ref.get('origin')}]")
    return ('normalised score = (return - random_return) / (threshold - '
            'random_return), so a uniform-random policy scores 0 and the '
            'registered threshold scores 1 (DESIGN.md 5.1) -- '
            + '; '.join(parts))


def smoothing_sentence(ctx: Context, extra: str = '') -> str:
    if ctx.smooth <= 1:
        return 'smoothing window: none (raw evaluation points)' + extra
    return (f'smoothing window: trailing mean over {ctx.smooth} evaluation '
            f'points, applied per run before averaging across seeds' + extra)


def emit(ctx: Context, name: str, fig: plt.Figure, body: str, *,
         seeds: Sequence[int] = (), n_min: Optional[int] = None,
         protocol: Optional[dict] = None, refs: Optional[dict] = None,
         interval: str = '', smoothing: Optional[str] = None,
         extra_lines: Sequence[str] = (),
         meta: Optional[dict] = None) -> None:
    """Write the figure, its generated caption and its provenance record."""
    os.makedirs(ctx.outdir, exist_ok=True)
    if n_min is not None and n_min < stats.MIN_N_FOR_INFERENCE:
        stamp_validation(fig, n_min)
    paths = []
    for fmt in ctx.formats:
        path = os.path.join(ctx.outdir, f'{name}.{fmt}')
        fig.savefig(path, format=fmt)
        paths.append(path)
    plt.close(fig)

    seeds = list(seeds)
    lines = [textwrap.fill(body, 92), '']
    if seeds:
        lines.append(textwrap.fill(
            f'Seeds: n={len(seeds)} distinct seeds plotted ({_seed_list(seeds)}). '
            + ('Below the confirmatory floor of ten seeds '
               '(STANDING_INSTRUCTIONS.md S4): estimates only, and no claim in '
               'the paper may rest on this figure.'
               if len(seeds) < 10 else
               'At or above the confirmatory floor of ten seeds '
               '(STANDING_INSTRUCTIONS.md S4).'), 92))
    if n_min is not None and n_min < stats.MIN_N_FOR_INFERENCE:
        lines.append(textwrap.fill(
            f'{stats.VALIDATION_STAMP}: the smallest arm has n={n_min} < '
            f'{stats.MIN_N_FOR_INFERENCE}, so no interval is drawn and no '
            f'number here may be quoted, compared or used to choose between '
            f'hypotheses (ANALYSIS_PLAN.md 9).', 92))
    if protocol is not None:
        lines.append(textwrap.fill('Evaluation: ' + protocol_sentence(protocol),
                                   92))
    lines.append(textwrap.fill('Interval: ' + (interval or 'none drawn'), 92))
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
    git = (ctx.prov.get('git') or {})
    lines.append(textwrap.fill(
        f"Source: {os.path.basename(ctx.per_seed_path)} "
        f"[{ctx.hashes.get('per_seed')}]"
        + (f", {os.path.basename(ctx.curves_path)} "
           f"[{ctx.hashes.get('curves')}]" if ctx.curves_path else '')
        + f"; git {str(git.get('commit'))[:12]}"
        + (' (dirty working tree)' if git.get('dirty') else '')
        + f"; ANALYSIS_PLAN.md {(ctx.prov.get('plans') or {}).get('ANALYSIS_PLAN.md')}", 92))

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
        'packages': ctx.prov.get('packages'),
        'bootstrap': {'n_boot': ctx.n_boot, 'seed': ctx.boot_seed},
        'smoothing_window_eval_points': ctx.smooth,
        'seeds': seeds,
        'analyses_carrying_a_p_value': 0,
        'arm_labels': ctx.arms,
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
    for cell in cells:
        for cond in ('scratch', 'transfer'):
            label = ctx.arms.get(cell, {}).get(cond)
            if label:
                frames[(cell, cond)] = curve_rows(ctx, label)
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
            if b['lo'] is not None:
                ax.fill_between(grid, b['lo'], b['hi'], color=style['colour'],
                                alpha=0.16, linewidth=0)
            line, = ax.plot(grid, b['mean'], color=style['colour'],
                            label=style['name'])
            if style['dashes'][0] is not None:
                line.set_dashes(style['dashes'])
            panel[cond] = {'n': int(mat.shape[0]), 'band': b['method'],
                           'final_grid_mean': float(b['mean'][-1])}
        bnd = freeze_boundary(ctx, frames.get((cell, 'transfer'), pd.DataFrame()))
        boundaries[cell] = bnd
        drew_boundary |= draw_boundary(ax, bnd, label=(cell == cells[0]))
        ax.axhline(0.0, color=_GREY, linewidth=0.5)
        ax.axhline(1.0, color=_GREY, linewidth=0.5, dashes=(1, 2))
        ax.set_title(cell.replace('-', ' / '))
        if not panel:
            _no_data(ax, 'no evaluation curve for this cell')
        meta['panels'][cell] = panel
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
    handles.append(Line2D([], [], color=_GREY, dashes=(3, 2),
                          label='freeze window ends'))
    handles.append(Line2D([], [], color=_GREY, linewidth=0.5, dashes=(1, 2),
                          label='score 1 = registered threshold'))
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    _legend(fig, handles, ncol=5)

    exited = sum(b['exited'] for b in boundaries.values())
    total = sum(b['runs'] for b in boundaries.values())
    max_upd = max([b['max_updates'] for b in boundaries.values()
                   if np.isfinite(b['max_updates'])] or [float('nan')])
    ks = sorted({int(v) for v in
                 ctx.per_seed[ctx.per_seed['condition'] == 'transfer']
                 ['freeze_updates'].dropna().tolist()})
    if exited:
        bnd_text = (f'The freeze boundary is drawn per cell at the mean env '
                    f'step where the window ended ({exited} of {total} '
                    f'transfer runs left it within budget).')
    else:
        bnd_text = (f'No freeze boundary is drawn: none of the {total} transfer '
                    f'runs left the freeze window within its budget (window '
                    f'{ks} gradient updates against at most {_i(max_upd)} '
                    f'updates performed), so the entire curve is inside the '
                    f'window.')

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
         protocol=ctx.protocol(
             [r for f in frames.values() for r in f['run_dir'].unique()]),
         refs=ctx.references(ctx.per_seed['env'].dropna().unique()),
         interval=(f'shaded band = 95% percentile bootstrap over seeds, '
                   f'{ctx.n_boot} resamples, fixed seed {ctx.boot_seed} '
                   f'(stats.boot_indices); suppressed where n < '
                   f'{stats.MIN_N_FOR_INFERENCE}'),
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
                             figsize=(FULL_WIDTH, 0.55 * len(cells) + 1.9),
                             sharey=True)
    axes = np.atleast_1d(axes)
    meta: dict[str, Any] = {'cells': {}}
    seeds: set[int] = set()
    n_min = None
    margin = stats.EQUIVALENCE_MARGIN

    for ax, endpoint in zip(axes, endpoints):
        ax.axvspan(-margin, margin, color=_GREY, alpha=0.13, linewidth=0,
                   zorder=0)
        ax.axvline(0.0, color=_BLACK, linewidth=0.7, zorder=1)
        ypos = list(range(len(cells)))[::-1]
        for y, cell in zip(ypos, cells):
            labels = ctx.arms[cell]
            if 'scratch' not in labels or 'transfer' not in labels:
                continue
            p = pair(ctx, labels['scratch'], labels['transfer'], endpoint)
            if p.n == 0:
                ax.text(0.0, y, ' no matched seeds', fontsize=6.4,
                        color=_GREY, va='center', style='italic')
                continue
            seeds.update(p.seeds)
            n_min = p.n if n_min is None else min(n_min, p.n)
            est = estimate_shift(ctx, p.delta)
            sd_base = float(np.std(p.base, ddof=1)) if p.n > 1 else float('nan')
            sd_treat = float(np.std(p.treat, ddof=1)) if p.n > 1 else float('nan')
            sd_cell = float(np.nanmax([sd_base, sd_treat]))
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
            ax.annotate(f'n={p.n}', xy=(1.0, y), xycoords=('axes fraction',
                                                           'data'),
                        xytext=(-2, 0), textcoords='offset points',
                        fontsize=6.2, color=_GREY, ha='right', va='center')
            meta['cells'].setdefault(cell, {})[endpoint] = {
                'n': p.n, 'seeds': p.seeds,
                'hodges_lehmann': est['estimate'], 'ci_lo': est['lo'],
                'ci_hi': est['hi'], 'ci_method': est['method'],
                'across_seed_sd_scratch': sd_base,
                'across_seed_sd_transfer': sd_treat,
                'exclusion': exclusion_sentence(est),
                'equivalence': equivalence_sentence(est, sd_cell),
                'unmatched_scratch_seeds': p.unmatched_base,
                'unmatched_transfer_seeds': p.unmatched_treat,
                'duplicate_arm_seeds': p.duplicates,
            }
        ax.set_yticks(list(range(len(cells)))[::-1])
        ax.set_yticklabels([c.replace('-', ' / ') for c in cells])
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
              label=f'+/-{margin:.2f} equivalence margin'),
    ]
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    _legend(fig, handles, ncol=3)

    verdicts = []
    for cell in cells:
        rec = (meta['cells'].get(cell) or {}).get('final_score')
        if rec:
            verdicts.append(f"{cell}: {rec['exclusion']}; {rec['equivalence']}")

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
               'equivalence.'),
         seeds=sorted(seeds), n_min=n_min,
         protocol=ctx.protocol(ctx.per_seed['run_dir'].tolist()),
         refs=ctx.references(ctx.per_seed['env'].dropna().unique()),
         interval=(f'Hodges-Lehmann paired shift with a bias-corrected (BCa) '
                   f'seed-level bootstrap 95% CI, {ctx.n_boot} resamples, '
                   f'fixed seed {ctx.boot_seed} (stats.bootstrap_statistic); '
                   f'suppressed where n < {stats.MIN_N_FOR_INFERENCE}'),
         smoothing='smoothing window: none (endpoint scalars, not curves)',
         extra_lines=[
             'Exclusion and equivalence statements generated per cell: '
             + ' | '.join(verdicts) if verdicts else
             'No cell had a matched scratch/transfer pair.',
             'No p-value is drawn. These four contrasts on two endpoints are '
             'the whole confirmatory family, and their Holm-adjusted p-values '
             'belong in the results table: a forest of significance verdicts '
             'invites the "A avoids negative transfer while B does not" '
             'comparison that DESIGN.md 9 and ANALYSIS_PLAN.md 8 forbid.'],
         meta=meta)


# ---------------------------------------------------------------------------
# 9. Figure 3 -- the control decomposition
# ---------------------------------------------------------------------------
def fig_control_decomposition(ctx: Context) -> None:
    endpoint = 'final_score'
    cells = [c for c in CELL_ORDER if c in ctx.arms]
    fig, axes = plt.subplots(1, max(len(cells), 1),
                             figsize=(FULL_WIDTH, 3.0), sharey=True)
    axes = np.atleast_1d(axes)
    meta: dict[str, Any] = {'endpoint': endpoint, 'cells': {}}
    seeds: set[int] = set()
    n_min = None
    missing: dict[str, list[str]] = {}

    for ax, cell in zip(axes, cells):
        labels = ctx.arms[cell]
        present = [c for c in CONTRAST_ORDER if c in labels]
        absent = [c for c in CONTRAST_ORDER if c not in labels]
        # Restrict to seeds where every present condition has a run: the
        # contrasts are estimated from one shared resampling of seeds
        # (ANALYSIS_PLAN.md 3), which requires one common seed set.
        per_cond: dict[str, pd.Series] = {}
        dupes: list[str] = []
        for cond in list(present):
            s = _seed_series(ctx, labels[cond], endpoint, dupes)
            if s.empty:
                present.remove(cond)
                absent.append(cond)
            else:
                per_cond[cond] = s
        if not present:
            _no_data(ax, 'no control runs for this cell')
            ax.set_title(cell.replace('-', ' / '))
            missing[cell] = absent
            continue
        common = sorted(set.intersection(*[set(s.index) for s in
                                           per_cond.values()]))
        if not common:
            _no_data(ax, 'no seed has every condition')
            ax.set_title(cell.replace('-', ' / '))
            missing[cell] = absent
            continue
        units = np.column_stack([per_cond[c].reindex(common).to_numpy(float)
                                 for c in present])
        seeds.update(common)
        n = len(common)
        n_min = n if n_min is None else min(n_min, n)
        idx = (stats.boot_indices(n, ctx.n_boot, ctx.boot_seed)
               if n >= stats.MIN_N_FOR_INFERENCE else None)

        xs = list(range(len(present)))
        levels = {}
        for x, cond in zip(xs, present):
            style = CONDITION_STYLE[cond]
            col = present.index(cond)
            est = estimate_mean(ctx, units, col, idx)
            ax.plot([x] * n, units[:, col], marker='|', linestyle='none',
                    color=_GREY, markersize=4.5, markeredgewidth=0.8, zorder=2)
            if np.isfinite(est['lo']):
                ax.plot([x, x], [est['lo'], est['hi']], color=style['colour'],
                        linewidth=1.3, zorder=3)
            ax.plot([x], [est['estimate']], marker=style['marker'],
                    color=style['colour'], markersize=4.6, zorder=4)
            levels[cond] = {k: est[k] for k in ('estimate', 'lo', 'hi',
                                                'method')}
        # The three named contrasts, annotated on the adjacent gaps.
        contrasts = {}
        y_top = float(np.nanmax(units))
        y_bot = float(np.nanmin(units))
        span = max(y_top - y_bot, 1e-6)
        for k, (hi_c, lo_c, name) in enumerate(CONTRAST_NAMES):
            if hi_c not in present or lo_c not in present:
                continue
            a, b = present.index(hi_c), present.index(lo_c)
            est = estimate_difference(ctx, units, a, b, idx)
            contrasts[name] = {'contrast': f'{hi_c} - {lo_c}',
                               **{kk: est[kk] for kk in ('estimate', 'lo',
                                                         'hi', 'method')}}
            xa, xb = present.index(lo_c), present.index(hi_c)
            ylev = y_top + span * (0.12 + 0.13 * k)
            ax.annotate('', xy=(xb, ylev), xytext=(xa, ylev),
                        arrowprops=dict(arrowstyle='<->', linewidth=0.6,
                                        color=_GREY, shrinkA=0, shrinkB=0))
            ci = ('' if not np.isfinite(est['lo'])
                  else f" [{est['lo']:+.2f}, {est['hi']:+.2f}]")
            ax.annotate(f"{name}\n{est['estimate']:+.3f}{ci}",
                        xy=((xa + xb) / 2.0, ylev), xytext=(0, 2),
                        textcoords='offset points', fontsize=5.8,
                        color='#333333', ha='center', va='bottom')
        ax.set_xticks(xs)
        ax.set_xticklabels([CONDITION_STYLE[c]['name'].split(' ')[0]
                            for c in present])
        ax.set_title(f"{cell.replace('-', ' / ')}  (n={n})")
        ax.set_ylim(y_bot - span * 0.12, y_top + span * (0.2 + 0.14 *
                                                        max(len(contrasts), 1)))
        ax.grid(axis='x', visible=False)
        meta['cells'][cell] = {'n': n, 'seeds': common, 'order': present,
                               'levels': levels, 'contrasts': contrasts,
                               'conditions_absent': absent,
                               'duplicate_arm_seeds': sorted(set(dupes))}
        missing[cell] = absent
    for ax in axes[len(cells):]:
        _no_data(ax, 'cell not present in the supplied table')
    axes[0].set_ylabel(f'{endpoint} (normalised)')

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
         protocol=ctx.protocol(ctx.per_seed['run_dir'].tolist()),
         refs=ctx.references(ctx.per_seed['env'].dropna().unique()),
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

    for ax, endpoint in zip(axes, endpoints):
        deltas: dict[str, pd.Series] = {}
        for cell in CELL_ORDER:
            labels = ctx.arms.get(cell, {})
            if 'scratch' not in labels or 'transfer' not in labels:
                continue
            p = pair(ctx, labels['scratch'], labels['transfer'], endpoint)
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
                est = estimate_shift(ctx, deltas[cell].to_numpy(float))
                xs.append(j)
                ys.append(est['estimate'])
                los.append(est['lo'])
                his.append(est['hi'])
                meta['cells'].setdefault(cell, {})[endpoint] = {
                    'n': int(len(deltas[cell])),
                    'hodges_lehmann': est['estimate'], 'ci_lo': est['lo'],
                    'ci_hi': est['hi'], 'ci_method': est['method']}
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
        # The interaction contrast, on the seeds common to all four cells.
        if len(deltas) == 4:
            common = sorted(set.intersection(*[set(s.index)
                                               for s in deltas.values()]))
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
                est = stats.bootstrap_statistic(units, inter,
                                               n_boot=ctx.n_boot,
                                               seed=ctx.boot_seed, idx=idx)
                est.pop('reps', None)
                meta['interaction'][endpoint] = {
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

    emit(ctx, 'interaction_2x2', fig,
         body=('Cell means of the within-cell transfer delta, architecture on '
               'x and Q-target rule as the series, with the interaction '
               'contrast annotated. This is RQ3, and RQ3 is EFFECT '
               'MODIFICATION -- how a causal effect varies across cells -- not '
               '"architecture causes the difference": the cells are different '
               'algorithms, not treatments assigned to units (DESIGN.md 2.4). '
               'It is estimation-only by design and not by omission: the '
               'interaction\'s minimum detectable effect is about 2.7 sigma at '
               'n=10, larger than any plausible effect, so it carries an '
               'interval and no p-value (ANALYSIS_PLAN.md 3, 6). Non-parallel '
               'lines here are not a finding.'),
         seeds=sorted(seeds), n_min=n_min,
         protocol=ctx.protocol(ctx.per_seed['run_dir'].tolist()),
         refs=ctx.references(ctx.per_seed['env'].dropna().unique()),
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
                            'x_axis': ('measured divergence from --shift-metrics'
                                       if ctx.shift_metrics
                                       else 'declared manipulated level')}
    seeds: set[int] = set()
    n_min = None

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
                if p.n == 0:
                    continue
                seeds.update(p.seeds)
                n_min = p.n if n_min is None else min(n_min, p.n)
                est = estimate_shift(ctx, p.delta)
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
                         f'supplied table')
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
                              'hi': est['hi']} for x, est, env in pts]
        meta['families'][family] = {'levels': max(len(v) for v in
                                                  points.values()),
                                    'points': rec_out}
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
        if p.n == 0:
            continue
        seeds.update(p.seeds)
        n_min = p.n if n_min is None else min(n_min, p.n)
        est = estimate_shift(ctx, p.delta)
        corner[cell] = {'n': p.n, 'estimate': est['estimate'], 'lo': est['lo'],
                        'hi': est['hi'], 'method': est['method'],
                        'seeds': p.seeds}
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
        _no_data(ax, 'no interface-only runs in the supplied table')
    ax.set_title('interface change,\nzero dynamics shift', fontsize=7.0)
    meta['interface_corner'] = corner
    axes[0].set_ylabel(f'delta {endpoint} (transfer - scratch)')

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
               'fixed interface and fixed protocol, for the two '
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
         protocol=ctx.protocol(ctx.per_seed['run_dir'].tolist()),
         refs=ctx.references(ctx.per_seed['env'].dropna().unique()),
         interval=(f'Hodges-Lehmann paired shift per level with a '
                   f'bias-corrected bootstrap 95% CI, {ctx.n_boot} resamples, '
                   f'seed {ctx.boot_seed}. The ordered-alternative trend '
                   f'statistic for H4 is reported by stats.py, not drawn here'),
         smoothing='smoothing window: none (endpoint scalars, not curves)',
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
    Membership in E1 or E4 is still required, which is what keeps E0's smoke
    runs -- a different episode budget entirely -- out of the curve.
    """
    ps = ctx.per_seed
    protocol = registry.PROTOCOL
    mask = ((ps['condition'] == 'transfer')
            & (ps['env'] == registry.TARGET_ENV)
            & (ps['source_env'] == registry.SOURCE_ENV)
            & (ps['transfer_set'] == protocol['transfer_set'])
            & (ps['freeze_group'] == protocol['freeze_group']))
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
                            'levels_found': []}
    seeds: set[int] = set()
    n_min = None
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
                for label in sorted(set(rows['label'].tolist())):
                    p = pair(ctx, base, label, endpoint)
                    if p.n == 0:
                        continue
                    seeds.update(p.seeds)
                    n_min = p.n if n_min is None else min(n_min, p.n)
                    est = estimate_shift(ctx, p.delta)
                    pts.append((pos[k], k, label, est, p.n))
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
                {'freeze_updates': k, 'label': lab, 'n': n,
                 'estimate': est['estimate'], 'lo': est['lo'], 'hi': est['hi'],
                 'x_position': x, 'off_scale': k <= 0}
                for x, k, lab, est, n in pts]
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
               'bracketed off-scale ticks, labelled as such.'),
         seeds=sorted(seeds), n_min=n_min,
         protocol=ctx.protocol(fam['run_dir'].tolist() if not fam.empty
                               else []),
         refs=ctx.references(ctx.per_seed['env'].dropna().unique()),
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
    ]
    rows = [r for r in rows if any(c in ctx.curves.columns for c in r['columns'])]
    if not rows:
        print(f'{WARN} diagnostics: none of the mechanism columns are in the '
              f'curve table; skipped')
        return

    frames: dict[tuple[str, str], pd.DataFrame] = {}
    for cell in cells:
        for cond in ('scratch', 'transfer'):
            label = ctx.arms.get(cell, {}).get(cond)
            if label:
                frames[(cell, cond)] = curve_rows(ctx, label)

    fig, axes = plt.subplots(len(rows), len(cells),
                             figsize=(FULL_WIDTH, 1.55 * len(rows) + 0.9),
                             sharex='col', squeeze=False)
    seeds: set[int] = set()
    n_min = None
    meta: dict[str, Any] = {'rows': [r['key'] for r in rows],
                            'gradient_columns': grad_cols, 'panels': {}}
    dash_by_col = {}
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
                _no_data(ax, 'value/advantage streams do not exist in this '
                             'architecture')
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
                                    'final': float(b['mean'][-1])}
                    draw_boundary(ax, freeze_boundary(
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
    for col, dashes in dash_by_col.items():
        handles.append(Line2D([], [], color=_GREY,
                              dashes=dashes if dashes[0] else (10, 0),
                              label=col))
    handles.append(Line2D([], [], color=_GREY, dashes=(3, 2),
                          label='freeze window ends'))
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
         protocol=ctx.protocol(
             [r for f in frames.values() for r in f['run_dir'].unique()]),
         refs=ctx.references(ctx.per_seed['env'].dropna().unique()),
         interval=(f'shaded band = 95% percentile bootstrap over seeds, '
                   f'{ctx.n_boot} resamples, seed {ctx.boot_seed}; suppressed '
                   f'where n < {stats.MIN_N_FOR_INFERENCE}'),
         extra_lines=[
             'Estimation-only, no p-value: mechanism signals are not in the '
             'confirmatory family (ANALYSIS_PLAN.md 1).'],
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

    values: dict[tuple[str, str], np.ndarray] = {}
    for cell in cells:
        for cond, label in ctx.arms.get(cell, {}).items():
            dupes: list[str] = []
            s = _seed_series(ctx, label, endpoint, dupes)
            if not s.empty:
                values[(cell, cond)] = s.to_numpy(float)
                seeds.update(int(i) for i in s.index)
    if values:
        all_v = np.concatenate(list(values.values()))
        lo = float(np.min(all_v)) - 0.02
        hi = float(np.max(all_v)) + 0.02
        taus = np.linspace(lo, hi, 400)
    else:
        taus = np.linspace(0.0, 1.0, 2)

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
            rec[cond] = {'n': int(len(v)), 'min': float(np.min(v)),
                         'median': float(np.median(v)),
                         'max': float(np.max(v)),
                         'fraction_above_0': float(np.mean(v > 0.0)),
                         'fraction_above_1': float(np.mean(v > 1.0))}
        if not drew:
            _no_data(ax, 'no runs for this cell')
        else:
            ax.axvline(0.0, color=_GREY, linewidth=0.5)
            ax.axvline(1.0, color=_GREY, linewidth=0.5, dashes=(1, 2))
            ax.set_ylim(-0.08, 1.04)
        ax.set_title(cell.replace('-', ' / '))
        meta['cells'][cell] = rec
    for ax in axes[len(cells):]:
        _no_data(ax, 'cell not present in the supplied table')
    for ax in axes[2:]:
        ax.set_xlabel(f'score threshold tau ({endpoint})')
    for ax in (axes[0], axes[2]):
        ax.set_ylabel('fraction of runs with score > tau')

    handles = [Line2D([], [], color=CONDITION_STYLE[c]['colour'],
                      dashes=CONDITION_STYLE[c]['dashes']
                      if CONDITION_STYLE[c]['dashes'][0] else (10, 0),
                      label=CONDITION_STYLE[c]['name'])
               for c in CONTRAST_ORDER
               if any((cell, c) in values for cell in cells)]
    handles.append(Line2D([], [], color=_GREY, linewidth=0.5, dashes=(1, 2),
                          label='score 1 = registered threshold'))
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
               'summaries here. Ticks below the axis are the individual runs, '
               'so the step size of each curve is visibly 1/n. A curve to the '
               'right of another dominates it at every threshold; crossing '
               'curves mean no arm dominates, which a difference in means '
               'would conceal.'),
         seeds=sorted(seeds), n_min=n_min,
         protocol=ctx.protocol(ctx.per_seed['run_dir'].tolist()),
         refs=ctx.references(ctx.per_seed['env'].dropna().unique()),
         interval=('none -- this is an empirical distribution function, drawn '
                   'without a confidence band because at these seed counts a '
                   'band would be wider than the distance between the arms and '
                   'would suggest a precision the design does not have'),
         smoothing='smoothing window: none (endpoint scalars, not curves)',
         meta=meta)


# ---------------------------------------------------------------------------
# 15. Figure 9 -- Kaplan-Meier for the threshold-reaching time
# ---------------------------------------------------------------------------
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

    for i, (tag, value) in enumerate(levels):
        tcol, ccol = f'steps_to_threshold_{tag}', f'censored_{tag}'
        for j, cell in enumerate(cells):
            ax = axes[i][j]
            drew = False
            arms_data = {}
            for cond in ('scratch', 'transfer'):
                label = ctx.arms.get(cell, {}).get(cond)
                if not label:
                    continue
                rows = arm_rows(ctx, label).dropna(subset=[tcol])
                if rows.empty:
                    continue
                t = rows[tcol].to_numpy(float)
                e = ~_as_bool(rows[ccol]) if ccol in rows.columns else \
                    np.ones(len(t), dtype=bool)
                seeds.update(int(s) for s in rows['seed'].tolist())
                n_min = len(t) if n_min is None else min(n_min, len(t))
                km = stats.kaplan_meier(t, e)
                k = int(e.sum())
                cp = stats.clopper_pearson(k, len(t))
                arms_data[cond] = {'t': t, 'e': e, 'km': km, 'k': k,
                                   'n': len(t), 'cp': cp}
                style = CONDITION_STYLE[cond]
                xs = [0.0] + [row['t'] for row in km['curve']]
                ys = [0.0] + [1.0 - row['survival'] for row in km['curve']]
                line, = ax.step(xs, ys, where='post', color=style['colour'],
                                label=style['name'])
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
                notes = ' | '.join(
                    f"{c[:4]} {d['k']}/{d['n']} "
                    f"[{d['cp'][0]:.2f},{d['cp'][1]:.2f}]"
                    for c, d in arms_data.items())
                ax.annotate(notes, xy=(0.02, 0.96), xycoords='axes fraction',
                            fontsize=5.6, color='#333333', va='top')
                ax.set_ylim(-0.05, 1.05)
            else:
                _no_data(ax, 'no threshold times for this cell')
            if 'scratch' in arms_data and 'transfer' in arms_data:
                a, b = arms_data['scratch'], arms_data['transfer']
                lr = stats.logrank_statistic(a['t'], a['e'], b['t'], b['e'])
                meta['logrank'][f'{tag}:{cell}'] = lr
            meta['levels'].setdefault(tag, {})[cell] = {
                c: {'events': d['k'], 'n': d['n'],
                    'p_reached': d['k'] / d['n'] if d['n'] else None,
                    'clopper_pearson': list(d['cp']),
                    'km_median': d['km']['median']}
                for c, d in arms_data.items()}
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
               'in total in this table.'),
         seeds=sorted(seeds), n_min=n_min,
         protocol=ctx.protocol(ctx.per_seed['run_dir'].tolist()),
         refs=ctx.references(ctx.per_seed['env'].dropna().unique()),
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
    parser.add_argument('--shift-metrics', default=None,
                        help='JSON mapping environment id -> measured '
                             'divergence, used as the x-axis of '
                             'shift_gradient. Without it the x-axis is the '
                             'declared manipulated level and the caption says '
                             'so')
    args = parser.parse_args(argv)

    argv_list = list(argv if argv is not None else sys.argv[1:])
    formats = tuple(f.strip().lstrip('.') for f in args.format.split(',')
                    if f.strip())
    if not formats:
        print(f'{WARN} --format left no output format')
        return 2

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

    per_seed, curves = load(args.per_seed, args.curves)

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
            arms=resolve_arm_labels(), iface=interface_labels(),
            shifts=shift_labels(),
            prov=provenance.snapshot(['experiments/plots.py'] + argv_list),
            hashes={'per_seed': provenance.file_hash(args.per_seed),
                    'curves': (provenance.file_hash(args.curves)
                               if os.path.isfile(args.curves) else None)})
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
        print(f'plots.py: {len(per_seed)} runs, '
              f'{len(curves)} curve rows, out -> {args.outdir}/')
        print(f'  ANALYSIS_PLAN.md hash now: {plans.get("ANALYSIS_PLAN.md")}')
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
        if 'seed_block' in per_seed.columns:
            tuned = per_seed[per_seed['seed_block'] == 'TUNE']
            if len(tuned):
                print(f'{WARN} {len(tuned)} run(s) are on TUNE seeds. No '
                      f'reported estimate may draw on them (DESIGN.md 3.4); '
                      f'audit.py is the enforcement point.')

        os.makedirs(args.outdir, exist_ok=True)
        for name in requested:
            FIGURES[name](ctx)

    print_ledger(len(requested))
    print(f'\n{len(ctx.written)} file(s) written to {args.outdir}/')
    return 0


if __name__ == '__main__':
    sys.exit(main())
