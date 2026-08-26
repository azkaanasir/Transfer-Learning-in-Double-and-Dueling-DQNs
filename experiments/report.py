"""One command from run tree to results bundle -- and the gate on what the prose may say.

    python experiments/report.py --out-root runs --outdir paper/results
    python experiments/report.py --out-root runs --outdir paper/results --experiments E1 E2
    python experiments/report.py --out-root runs_demo --outdir paper/results \
        --seeds 0-2 --overrides num_episodes=14 freeze_updates=150 \
        --allow-audit-failure --source-policy pooled
    python experiments/report.py --self-test

The pipeline is `audit.audit_ok` -> `aggregate.py` -> `stats.py --json` ->
`plots.py` -> tables -> `REPORT.md` + `MANIFEST.json`, all of it into one dated
directory whose every file is hashed. Two properties are the point, and both are
mechanical rather than promised:

* **Nothing is reported over a failed audit without a stamp.** `DESIGN.md` 8.4:
  "Aggregation and reporting refuse to run on a failed audit unless overridden,
  and the override is stamped into the output." So the audit runs *first*, and
  without `--allow-audit-failure` this module writes nothing at all -- not a
  partial bundle, not a draft. With the flag, the failing check names go at the
  top of `REPORT.md`, into every table's provenance record, into
  `MANIFEST.json`, and into a marker file beside the figures.
* **No sentence is emitted unless the evidence supports the *kind* of claim it
  makes.** That is `claim()`, and it is the reason this file exists rather than
  being a shell script. The published study's central failure was not a wrong
  number: it was a sentence type its evidence could not carry ("positive
  transfer" from p=0.421; a between-architecture return gap read as a transfer
  effect; a narrower spread described as broader; "identical hyperparameters"
  over a 5x learning-rate gap). Each of those is now a refusal with a section
  citation attached.

What each guardrail is defending against, defect by defect
----------------------------------------------------------

* **Affirming a null.** `DESIGN.md` 9, `ANALYSIS_PLAN.md` 4. `kind='equivalence'`
  requires an equivalence *verdict* computed by `stats.py` from interval
  containment -- never a large p-value, and never the absence of one. Where the
  cell's dispersion exceeds the +/-0.05 margin the verdict is UNTESTABLE and the
  emitted sentence says so. The exclusion bound is emitted alongside every
  interval, because it is the only powered directional statement available at
  this sample size.
* **A between-group difference narrated as a within-group effect.**
  `DESIGN.md` 2.4, 9. `kind='causal'` is admitted only for the RQ2 estimand:
  the within-cell paired delta. Between-cell quantities are `associational`
  (RQ1) or `effect_modification` (RQ3), both of which refuse causal verbs and
  refuse p-values. An equivalence or exclusion statement is admitted only about
  an estimand whose interval is on the paired normalised-score scale, so a
  bound in score units cannot be asserted over a proportion, an SD ratio or a
  mechanism signal.
* **Comparing two significance verdicts and calling it a comparison.**
  `ANALYSIS_PLAN.md` 8. The sentence form "A avoids X while B does not" is
  detected by shape and refused unless the evidence carries an explicit
  between-cell contrast with an interval.
* **A mechanism claimed from prose.** `DESIGN.md` 5.5, 9. `kind='mechanism'`
  requires a named instrumented signal from the 5.5 table; there is no
  free-text mechanism slot anywhere in the template.
* **A directional adjective contradicting the numbers.** `DESIGN.md` 9. For
  `kind='dispersion'` the sentence is *generated* from the two SDs by
  `stats.phrase_dispersion`, and a caller-supplied adjective that disagrees with
  the ratio is refused rather than quietly overwritten. The same treatment is
  given to the RQ1 relative effect: the verb comes out of `phrase_relative_effect`
  as a function of theta against the 0.5 null, because a hand-written "exceeds"
  survived into shipped output over theta=0.469 and said the opposite of the
  data. Every dispersion and relative-effect sentence carries its interval, and
  the interval is read against the null the ratio has (1, or theta=0.5), not
  against zero.
* **An equivalence verdict taken on trust.** `ANALYSIS_PLAN.md` 4. The verdict
  string is not enough: `claim()` re-derives the verdict from the interval and
  the margin with the same containment rule `stats.section_equivalence` uses,
  and refuses when the two disagree. A verdict is a computation, so a caller
  that can supply a verdict can supply the wrong one.
* **A number published under its own suppression.** `DESIGN.md` 8.4,
  `ANALYSIS_PLAN.md` 9. Where `stats.py` suppressed a confirmatory member, the
  estimate columns are blanked in `REPORT.md` and in `tables/confirmatory.csv`,
  leaving the reason. "A suppression is a refusal with a reason, never a null"
  is false the moment the table hands the reader the point estimate anyway.
* **A sentence that loses its scope on the way to the manuscript.** Every
  accepted claim, of every kind, carries the `DESIGN.md` 2.1 clause; deciding
  per sentence whether it generalises is exactly the judgement that fails
  silently, and it failed here (zero of 106 emitted sentences carried it).
* **A secondary estimand mistaken for the primary one.** `DESIGN.md` 4.3.
  Under `--source-policy pooled` every emitted sentence is prefixed with the
  pooled-over-source-competence label, so a sentence copied out of the bundle
  cannot lose it.
* **A finding nothing could refute.** `STANDING_INSTRUCTIONS` S5 Q3. Every claim
  must supply what would refute it, and the four Socratic questions are printed
  against each emitted claim, with any unanswered question named rather than
  dropped.
* **A single-seed number quoted as a result.** `ANALYSIS_PLAN.md` 9,
  `STANDING_INSTRUCTIONS` S8. Below n=3 a claim is refused, and the whole bundle
  is stamped PIPELINE VALIDATION - NOT A RESULT. The n that guard sees is a
  count of *units*, never of rows: the inventory claim is bounded by the
  weakest cell rather than by the sum over cells, and RQ1's n is the seed count
  behind an arm rather than the number of rows pooled into it. Both used to be
  the larger number, which let a sentence through on a one-seed tree.
* **Stale artifacts.** `DESIGN.md` 8.3, 9. Every table records the hash of the
  CSV and the stats JSON it was built from; `MANIFEST.json` hashes every file in
  the bundle; `per_seed.csv`, `curves.csv` and `stats.json` are copied in, so the
  bundle is readable years later without the run tree. A bundle directory is
  never reused: the date alone is not a key, so a second invocation on the same
  day takes `<date>.2` rather than merging into figures drawn from a different
  run tree and hashing them as its own. A bundle carries `INCOMPLETE.txt` from
  the moment it is created until `MANIFEST.json` is written, so a run that dies
  in a middle stage leaves a directory that says what it is.

Numbers, and the one place arithmetic happens here
--------------------------------------------------

Every *endpoint* number in `REPORT.md` is read from `stats.py --json` or from
the pinned `per_seed.csv`, because `STANDING_INSTRUCTIONS` S6 forbids
hand-computing a number that appears in the paper, and because a second
definition of an endpoint is how the published paper's V.A and V.B came to
contradict each other. The single exception is arithmetic *on the sample size
itself*: the smallest attainable two-sided sign-flip p at n is 2/2^n, and the
sentence stating the confirmatory bar is generated from the n actually in the
bundle. It used to be a hand-written constant for n=10, which printed the n=10
bar over an n=3 table and told the reader a cell could be confirmed when no
cell could. Anywhere that arithmetic happens, the sentence says what it is a
function of.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from experiments import audit as audit_mod                          # noqa: E402
from experiments import registry                                    # noqa: E402
from experiments import stats as stats_mod                           # noqa: E402
from src.dqn import envs, provenance                                 # noqa: E402

def rel(path: str, start: str) -> str:
    """`os.path.relpath`, but tolerant of a bundle on a different drive.

    On Windows a results directory on another volume than the repository makes
    `relpath` raise, and a crash in the path formatting of a log line is not an
    acceptable way to lose a report.
    """
    try:
        return os.path.relpath(path, start).replace(os.sep, '/')
    except ValueError:
        return os.path.abspath(path).replace(os.sep, '/')


WARN = '[WARNING]'

# ===========================================================================
# 1. The scope clause and the vocabulary of claims.
#
# `DESIGN.md` 2.1: "Where a sentence in the report would generalise past this,
# `report.py` prefixes it with this clause or refuses to emit it." The clause is
# therefore stored verbatim and pasted into the claim *string*, not merely
# printed once at the top of the page -- a sentence copied out of this report
# into the manuscript has to carry its own scope with it.
# ===========================================================================
SCOPE_CLAUSE = (
    'Over the finite seed set actually run, for the stated (arch, target_rule) '
    'implementations at hidden=(128,128), head_units=64, Adam, the declared '
    'exploration schedule and episode budget, on the named environment pairs, '
    'and with no claim about the dueling decomposition or the double-Q update '
    'as algorithmic ideas nor about deep RL transfer in general '
    '(DESIGN.md 2.1):')

#: Inference type licensed for each estimand, and where the design says so.
#: `DESIGN.md` 2.4: "Inference type is binding on wording."
ESTIMAND_INFERENCE: dict[str, tuple[str, str]] = {
    'within_cell_delta':    ('causal', 'DESIGN.md 2.4 RQ2'),
    'control_contrast':     ('causal_component', 'DESIGN.md 2.4 RQ4, 4.1'),
    'shift_gradient':       ('causal_component', 'DESIGN.md 2.4 RQ5'),
    'budget_prefix':        ('causal_component', 'DESIGN.md 2.4 RQ6'),
    'between_cell_scratch': ('associational', 'DESIGN.md 2.4 RQ1'),
    'between_cell_delta':   ('effect_modification', 'DESIGN.md 2.4 RQ3'),
    'interaction_2x2':      ('effect_modification', 'DESIGN.md 2.4 RQ3'),
    'mechanism_signal':     ('mechanism', 'DESIGN.md 5.5'),
    'dispersion':           ('dispersion', 'DESIGN.md 5.3'),
    'censoring_proportion': ('descriptive', 'ANALYSIS_PLAN.md 5'),
    'secondary_endpoint':   ('descriptive', 'ANALYSIS_PLAN.md 1'),
    'arm_descriptive':      ('descriptive', 'ANALYSIS_PLAN.md 10.4'),
}

#: Kinds that may additionally be asserted about an estimand whose interval is
#: on the paired normalised-score scale, because they are statements *about
#: that interval* rather than about the causal structure (`ANALYSIS_PLAN.md` 4).
INTERVAL_KINDS = ('equivalence', 'exclusion')

#: Which estimands each interval kind may be asserted *about*. Without this the
#: `INTERVAL_KINDS` escape hatch admitted an exclusion bound quoted in score
#: units over a proportion, a gradient-norm signal and an SD ratio, and let
#: kind='exclusion' on `mechanism_signal` walk past the `DESIGN.md` 5.5
#: requirement to name a signal. `ANALYSIS_PLAN.md` 4 fixes the margin in
#: normalised-score units, so only a contrast measured in those units can be
#: contained in it or excluded by it. Equivalence is narrower still: the margin
#: was pre-registered for the within-cell paired delta and for nothing else.
INTERVAL_KIND_ESTIMANDS: dict[str, tuple[str, ...]] = {
    'equivalence': ('within_cell_delta',),
    'exclusion': ('within_cell_delta', 'control_contrast',
                  'between_cell_delta', 'interaction_2x2', 'shift_gradient',
                  'budget_prefix'),
}

KINDS: tuple[str, ...] = (
    'causal', 'causal_component', 'associational', 'effect_modification',
    'equivalence', 'exclusion', 'mechanism', 'dispersion', 'descriptive')

#: `DESIGN.md` 2.4 RQ2 is the only causal estimand this module will dress in the
#: word "causal": the within-cell paired delta, whose warrant is *ceteris
#: paribus* at matched seeds. Everything RQ4-RQ6 manipulates is causal with
#: respect to *that component* and gets `causal_component`, which forces the
#: manipulated thing to be named in the evidence.
CAUSAL_ESTIMANDS = ('within_cell_delta',)

#: The instrumented signals of `DESIGN.md` 5.5. Derived from the metric-role
#: table in `stats.py` so the two cannot drift apart, plus the three signals 5.5
#: names that are not per-run columns in the pinned schema.
DESIGN_5_5_SIGNALS: tuple[str, ...] = tuple(sorted(
    set(stats_mod.MECHANISM_COLUMNS)
    | {'q_max', 'effective_rank', 'param_norm'}))

#: Verdicts `stats.section_equivalence` can return. Anything else is not an
#: equivalence verdict and may not license an equivalence sentence.
#: `DEGENERATE` is listed because `stats.py` emits it for a zero-width interval
#: and a verdict this module does not recognise was refused as "not a verdict
#: from the interval-containment procedure", which is the wrong reason and
#: reads as a template fault rather than as the finding it is. What refuses the
#: sentence is `degenerate-interval` below.
EQUIVALENCE_VERDICTS = ('EQUIVALENT', 'DIFFERENT', 'INCONCLUSIVE',
                        'UNTESTABLE', 'DEGENERATE', 'no interval',
                        'suppressed')

#: Verdicts whose *availability* the `ANALYSIS_PLAN.md` 4 dispersion gate
#: governs. `stats.section_equivalence` evaluates DIFFERENT *above* its own
#: dispersion gate on purpose (stats.py: calling an interval that lies wholly
#: outside the margin "untestable" "discards a finding the data supports"), so
#: applying the gate to DIFFERENT here contradicts the module that computes the
#: verdict and withdraws the largest effects in the study: at the
#: `ANALYSIS_PLAN.md` 6.3 planning SDs, two of the four cells sit above the
#: margin. The gate limits what may be called EQUIVALENT, not what may be
#: called different.
DISPERSION_GATED_VERDICTS = ('EQUIVALENT', 'INCONCLUSIVE')

VALIDATION_STAMP = stats_mod.VALIDATION_STAMP
OVERRIDE_STAMP = ('AUDIT OVERRIDE IN FORCE -- produced with '
                  '--allow-audit-failure over a FAILED audit (DESIGN.md 8.4)')

#: Written into a bundle directory the moment it is created and removed only
#: when `MANIFEST.json` lands. A run that dies in a middle stage then leaves a
#: directory that says what it is, rather than a plausible-looking partial one.
INCOMPLETE_MARKER = 'INCOMPLETE.txt'

#: `DESIGN.md` 4.3 pre-declares pooling over source competence as a *secondary*
#: estimand that must be "labelled exactly that". Recording it in an inventory
#: table is not labelling the sentences: a sentence lifted out of the bundle
#: carries only its own words, so the label goes into the words.
POOLED_SOURCE_LABEL = (
    'SECONDARY ESTIMAND, pooled over source competence: this sentence includes '
    'transfer runs whose source failed the DESIGN.md 4.3 validity gate, and is '
    'not the primary valid-sources-only estimand (DESIGN.md 4.3).')

#: Column names that carry a p-value or an adjusted p-value. `ANALYSIS_PLAN.md`
#: 7: which tables carry one is a bookkeeping fact, so it is read off the
#: columns rather than off the table's name. Naming only `confirmatory` missed
#: the BH q column in `screens`, and the prose then contradicted the table.
PVALUE_COLUMNS: tuple[str, ...] = (
    'p_signflip', 'p_holm', 'p_wilcoxon', 'p_mannwhitney', 'p_logrank',
    'p_value', 'q', 'q_bh', 'significant_holm')

#: The confirmatory columns that carry an *estimate*. Where `stats.py` recorded
#: a suppression, these are blanked in every rendering: `ANALYSIS_PLAN.md` 9 and
#: `DESIGN.md` 8.4 make a suppression a refusal with a reason, and a table that
#: prints the point estimate beside the reason has published the number anyway.
SUPPRESSED_BLANK_COLUMNS: tuple[str, ...] = (
    'scratch_mean', 'transfer_mean', 'mean_delta', 'hl', 'ci_lo', 'ci_hi',
    'p_signflip', 'p_holm', 'significant_holm', 'p_wilcoxon', 'mannwhitney_U',
    'p_mannwhitney', 'rho_pearson', 'rho_spearman', 'unanimous')

# ---------------------------------------------------------------------------
# 1.1 Forbidden phrases. Each one was either printed in the published paper or
# was a live temptation in the design, and each names the section that forbids
# it. Matching is on the emitted sentence, whoever wrote it.
# ---------------------------------------------------------------------------
FORBIDDEN_PHRASES: tuple[tuple[str, str, str], ...] = (
    (r'positive\s+transfer', 'DESIGN.md 1, ANALYSIS_PLAN.md 4',
     'the published study claimed "positive transfer" from p=0.421; a null is '
     'never evidence of an effect, and the licensed positive statement is the '
     'exclusion bound'),
    (r'\bprove[sdn]?\b|\bproof\b|\bproven\b', 'ANALYSIS_PLAN.md 8, DESIGN.md 9',
     'no design proves anything; the licensed forms are an interval, a '
     'Holm-adjusted verdict inside the one confirmatory family, and an '
     'exclusion bound'),
    (r'demonstrates\s+that\s+architecture\s+determines',
     'DESIGN.md 2.4 RQ3, 10.1',
     'RQ3 is effect modification and is estimation-only: its MDE is ~2.7 sigma, '
     'larger than the plausible effect, so it carries an interval and no '
     'assertion'),
    (r'identical\s+baselines', 'DESIGN.md 1, 3.3',
     'the published arms differed in learning rate by 5x under a printed claim '
     'of identical hyperparameters; what is verified is invariance of the '
     'audited fields, and audit.py names the scope at which it holds'),
    (r'broader\s+spread', 'DESIGN.md 9, 5.3',
     'the published paper called a narrower spread broader; dispersion wording '
     'is generated from the two SDs by stats.phrase_dispersion, so a literal '
     'adjective may not be written by hand'),
    (r'\bidentical\s+hyperparameters\b', 'DESIGN.md 3.3, 8.4',
     'the claim is machine-checked at a declared scope, so the sentence must '
     'name that scope (audit.py INVARIANTS) rather than assert the property'),
)

#: The sentence *shape* `DESIGN.md` 9 forbids: "A avoids it, B does not", which
#: compares two significance verdicts and calls the comparison a result. It is
#: permitted only when the evidence carries the between-cell contrast itself.
#: The verbs that carry a per-cell verdict in this template. Kept separate so
#: the shapes below are built from one list rather than each naming its own
#: subset: the original patterns were bypassed by writing "degrades and does
#: not" instead of "avoids while does not", which is the same sentence.
_VERDICT_VERBS = (r'avoids?|escapes?|suffers?|degrades?|shows?|exhibits?'
                  r'|is\s+hurt|is\s+harmed|is\s+spared')

VERDICT_COMPARISON_PATTERNS: tuple[str, ...] = (
    # "A <verdict verb> ... B does not", whatever joins the two halves: while,
    # whereas, but, although, a semicolon, a comma or a bare "and".
    rf'\b(?:{_VERDICT_VERBS})\b[^.]*\b(?:does|do|did)\s+not\b',
    # "only A suffers it", which asserts the same two-verdict comparison by
    # quantifying over the cells instead of naming the second one.
    rf'\bonly\s+[\w.-]+\s+(?:{_VERDICT_VERBS})\b',
    rf'\bnone\s+of\s+[^.]*\b(?:{_VERDICT_VERBS})\b',
    r'\b(?:while|whereas)\b[^.]*\bdoes\s+not\s+(?:suffer|show|exhibit|avoid'
    r'|degrade)\b',
    r'\bis\s+(?:significant|distinguishable)\b[^.]*\b(?:while|whereas|but|;)\b'
    r'[^.]*\bis\s+not\b',
)

#: Verbs that assert causation. Refused on the kinds whose design-declared
#: inference is not causal (`DESIGN.md` 2.4). The list is long because English
#: has many ways to say it and the guard is only as good as its vocabulary:
#: "produces", "stems from", "explains" and "gives rise to" all walked past the
#: first version of this pattern in an associational sentence.
CAUSAL_VERB_PATTERN = (
    r'\bcauses?\b|\bcaused\b|\bcausing\b|\bcausal\b|\bdetermines?\b'
    r'|\bdetermined\s+by\b|\bleads?\s+to\b|\bled\s+to\b|\bdrives?\b|\bdriven'
    r'\s+by\b|\bbecause\s+of\b|\bresponsible\s+for\b|\bproduces?\b|\bproduced'
    r'\s+by\b|\bstems?\s+from\b|\bstemming\s+from\b|\bthe\s+reason\s+for\b'
    r'|\bgives?\s+rise\s+to\b|\bgiving\s+rise\s+to\b|\bexplains?\b'
    r'|\bexplained\s+by\b|\binduces?\b|\binduced\s+by\b|\bbrings?\s+about\b'
    r'|\battributable\s+to\b|\battributed\s+to\b'
    r'|\bmakes?\s+(?:\w+\s+){0,3}(?:worse|better|harder|easier)\b'
    r'|\bmade\s+(?:\w+\s+){0,3}(?:worse|better)\b|\bresponsible'
    r'\b|\bowing\s+to\b|\btriggers?\b|\bthe\s+effect\s+of\s+\w+\s+on\b')

#: Any mention of a p-value or of significance. Permitted only inside the one
#: confirmatory family (`ANALYSIS_PLAN.md` 2, 7).
PVALUE_PATTERN = (r'\bp\s*[=<>]\s*0?\.\d|\bp-value\b|\bsignifican(?:t|ce|tly)\b'
                  r'|\bHolm\b')

#: Words that affirm a null. Permitted only for kind='equivalence' carrying an
#: EQUIVALENT verdict (`ANALYSIS_PLAN.md` 4). Every alternative below was
#: verified to walk past the first version of this pattern: "the delta is nil",
#: "no impact", "perform identically", "transfer is neutral", "no evidence of a
#: shift", "the effect is absent", "no meaningful difference", "neither helps
#: nor hurts". A guard on wording is a guard on a vocabulary, so the vocabulary
#: is enumerated rather than sampled.
NULL_AFFIRMING_PATTERN = (
    r'\bequivalent\b|\bequivalence\s+holds\b'
    r'|\bno\s+(?:difference|effect|degradation|change|shift|impact|benefit'
    r'|harm|gain|loss|penalty)\b'
    r'|\bunaffected\b|\bunchanged\b|\bthe\s+same\s+as\b|\bhave\s+the\s+same\b'
    r'|\bhas\s+the\s+same\b|\bidentical(?:ly)?\b|\bindistinguishable\s+from\s+'
    r'zero,\s+therefore\b|\bis\s+nil\b|\bare\s+nil\b|\bnil\b'
    r'|\bneutral\b|\bno\s+evidence\s+(?:of|for|that)\b'
    r'|\b(?:effect|difference|delta|shift|gap)\s+is\s+absent\b'
    r'|\bno\s+(?:meaningful|detectable|appreciable|measurable|real)\s+'
    r'(?:difference|effect|change|shift)\b'
    r'|\bneither\s+\w+\s+nor\s+\w+\b|\bmakes?\s+no\s+difference\b'
    r'|\bnothing\s+(?:changed|happened)\b|\bzero\s+effect\b')

#: Directional dispersion adjectives, checked against the SD ratio.
DISPERSION_WIDER = (r'\bwider\b', r'\bbroader\b', r'\bmore\s+variable\b',
                    r'\bnoisier\b', r'\bhigher\s+variance\b')
DISPERSION_NARROWER = (r'\bnarrower\b', r'\btighter\b', r'\bless\s+variable\b',
                       r'\bquieter\b', r'\blower\s+variance\b')

#: Directional verbs for a *relative effect* (RQ1's Brunner-Munzel theta),
#: checked against theta the way the dispersion adjectives are checked against
#: the SD ratio. `DESIGN.md` 9: direction sentences are generated from the
#: data. This list exists because "exceeds" was written by hand and shipped
#: over theta=0.469 and theta=0.333, asserting the opposite of the numbers in
#: the same sentence that printed them.
RELATIVE_EFFECT_HIGHER = (r'\bexceeds?\b', r'\bbeats?\b', r'\boutperforms?\b',
                          r'\bis\s+better\s+than\b', r'\bscores?\s+above\b',
                          r'\bahead\s+of\b', r'\bsurpasses?\b',
                          r'\bis\s+higher\s+than\b', r'\bwins\s+against\b')
RELATIVE_EFFECT_LOWER = (r'\bfalls?\s+short\s+of\b', r'\btrails?\b',
                         r'\bunderperforms?\b', r'\bis\s+worse\s+than\b',
                         r'\bscores?\s+below\b', r'\bbehind\b',
                         r'\bis\s+lower\s+than\b', r'\bloses\s+to\b')

#: Every guard `claim()` can apply, keyed by the id it records in its refusals.
#: Section 11a is rendered from this table and counts each guard's refusals in
#: the invocation being reported, so a guard that cannot fire on real data is
#: visible as one rather than advertised as "exercised in this invocation".
#: Two of them could never fire here and said nothing about it: `TUNE` (no
#: production evidence dict carries a seed block, because `stats.py` drops the
#: TUNE block before this module sees a row) and `equivalence-basis` (the
#: template always passes basis='bootstrap_ci'). Both are kept, because a guard
#: that is redundant with an upstream one is defence in depth, and both now
#: name where the real enforcement is.
GUARD_CATALOGUE: tuple[tuple[str, str, str, str], ...] = (
    ('kind', 'the sentence performs one of the declared inference types',
     'report.py 1', ''),
    ('estimand', 'the estimand is in the design research-question table',
     'DESIGN.md 2.4', ''),
    ('inference-binding',
     'the wording performs the inference type the estimand licenses',
     'DESIGN.md 2.4', ''),
    ('interval-kind-estimand',
     'an equivalence or exclusion statement is made only about an estimand '
     'measured in paired normalised-score units',
     'DESIGN.md 2.4, ANALYSIS_PLAN.md 4', ''),
    ('causal-estimand',
     'estimand is the within-cell paired delta; the DESIGN.md 2.1 scope clause '
     'is prefixed automatically', 'DESIGN.md 2.4 RQ2, 2.1', ''),
    ('component-manipulation',
     'the manipulated component is named, as in DESIGN.md 4.1',
     'DESIGN.md 4.1', ''),
    ('mechanism-signal',
     'a named instrumented signal from the DESIGN.md 5.5 table',
     'DESIGN.md 5.5, 9', ''),
    ('equivalence-verdict',
     'an equivalence verdict from interval containment, never a null p',
     'ANALYSIS_PLAN.md 4', ''),
    ('equivalence-assertion',
     'the sentence asserts equivalence only under an EQUIVALENT verdict',
     'ANALYSIS_PLAN.md 4', ''),
    ('equivalence-containment',
     'the verdict agrees with containment of the interval in the margin, '
     're-derived here from the numbers', 'ANALYSIS_PLAN.md 4', ''),
    ('equivalence-feasibility',
     'an EQUIVALENT verdict carries two across-seed SDs and both are inside '
     'the margin; an UNTESTABLE one is not contradicted by them',
     'ANALYSIS_PLAN.md 4',
     'the gate governs what may be called EQUIVALENT, not what may be called '
     'DIFFERENT: stats.py evaluates an interval lying wholly outside the '
     'margin before its own dispersion gate, and refusing that verdict here '
     'withdrew the study largest effects'),
    ('degenerate-interval',
     'the interval has non-zero width; a zero-width one reports that nothing '
     'was estimable', 'ANALYSIS_PLAN.md 3, DESIGN.md 9', ''),
    ('exclusion-bound',
     'the bound is the one the computing section published, not one '
     're-derived from a bound it withheld', 'ANALYSIS_PLAN.md 4, DESIGN.md 9',
     ''),
    ('equivalence-basis', 'the verdict is not read off a p-value',
     'ANALYSIS_PLAN.md 4, 8',
     'unreachable from this template, which always supplies '
     'basis=bootstrap_ci; the real protection is that stats.py computes the '
     'verdict by containment and never from a p-value'),
    ('dispersion-evidence', 'two across-seed SDs are present',
     'DESIGN.md 9, 5.3', ''),
    ('dispersion-direction',
     'the directional word is generated from the two SDs, not written',
     'DESIGN.md 9', ''),
    ('relative-effect-direction',
     'the direction verb is generated from theta against the 0.5 null',
     'DESIGN.md 9', ''),
    ('interval',
     'a directional statement carries the interval it is read from, the SD '
     'ratio included', 'ANALYSIS_PLAN.md 3', ''),
    ('n-missing', 'the number of units is stated and is a count',
     'ANALYSIS_PLAN.md 9', ''),
    ('n<3', 'n >= 3; below it nothing is emitted',
     'ANALYSIS_PLAN.md 9, STANDING_INSTRUCTIONS S8', ''),
    ('TUNE', 'the estimate does not draw on the selection block',
     'DESIGN.md 3.4',
     'unreachable from this template: stats.py excludes seed_block=TUNE from '
     'the table before any number reaches this module, and the count it '
     'excluded is printed in section 2'),
    ('raw-return', 'the claim is on the normalised score', 'DESIGN.md 5.1', ''),
    ('forbidden-phrase',
     'none of the phrasings section 11b lists appears in the sentence',
     'DESIGN.md 1, 9', ''),
    ('verdict-comparison',
     'an "A avoids it, B does not" shape carries an explicit between-cell '
     'contrast with its interval', 'ANALYSIS_PLAN.md 8, DESIGN.md 9', ''),
    ('p-value', 'membership of the eight-test confirmatory family',
     'ANALYSIS_PLAN.md 2, 7', ''),
    ('null-affirming',
     'an EQUIVALENT verdict; otherwise "not distinguishable" plus the '
     'exclusion bound', 'DESIGN.md 9', ''),
    ('causal-verb', 'no causal verb in a non-causal sentence',
     'DESIGN.md 2.4, 9', ''),
    ('refutability', 'the evidence states what would refute the claim',
     'STANDING_INSTRUCTIONS S5 Q3', ''),
)

#: The four questions `STANDING_INSTRUCTIONS` S5 requires against every result.
SOCRATIC_QUESTIONS: tuple[tuple[str, str], ...] = (
    ('Q1', 'What is the counterfactual, and is it actually in the data?'),
    ('Q2', 'What else could produce this number? Which control excludes that?'),
    ('Q3', 'What would refute this? If nothing could, it is not a finding.'),
    ('Q4', "Is the wording's inference type the same as the design's?"),
)


# ===========================================================================
# 2. Claims.
# ===========================================================================
@dataclass
class Claim:
    """One sentence, its evidence, and whether the evidence licenses it.

    `refusals` is a tuple of `(guard, where, why)`. The `guard` id is the same
    key the catalogue in section 11a is built from, so the catalogue cannot
    advertise a guard that no longer exists and the report can state, per guard,
    whether it fired in this invocation instead of implying that all of them
    did.
    """

    text: str
    kind: str
    estimand: str
    evidence: dict
    accepted: bool
    refusals: tuple[tuple[str, str, str], ...] = ()
    socratic: tuple[tuple[str, str, str], ...] = ()
    generated: bool = False
    proposed: Optional[str] = None
    section: str = ''

    @property
    def carries_p(self) -> bool:
        return bool(self.evidence.get('confirmatory')
                    and self.evidence.get('p_holm') is not None)


@dataclass
class ClaimLog:
    """Every claim considered, accepted or refused, in emission order."""

    claims: list[Claim] = field(default_factory=list)
    echo: bool = True
    section: str = ''

    def add(self, c: Claim) -> Claim:
        c.section = c.section or self.section
        self.claims.append(c)
        if self.echo:
            for line in render_claim_lines(c):
                print(line)
        return c

    @property
    def accepted(self) -> list[Claim]:
        return [c for c in self.claims if c.accepted]

    @property
    def refused(self) -> list[Claim]:
        return [c for c in self.claims if not c.accepted]

    def with_p(self) -> list[Claim]:
        return [c for c in self.accepted if c.carries_p]

    def without_p(self) -> list[Claim]:
        return [c for c in self.accepted if not c.carries_p]


def _hits(pattern: str, text: str) -> bool:
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def _finite(v: Any) -> bool:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    return f == f and abs(f) != float('inf')


def _degenerate_interval(lo: Any, hi: Any) -> bool:
    """Whether an interval reported as [lo, hi] has zero (or negative) width.

    A zero-width bootstrap interval is not an infinitely precise measurement.
    It is what a constant arm, a duplicated run row or complete separation
    between two cells produces: every resample is identical, so nothing about
    sampling uncertainty was estimable and the width carries no information at
    all. `stats.py` says so in two places (a `DEGENERATE` equivalence verdict
    and a `degenerate` flag on its BCa intervals) and `plots.py` measures the
    same thing off the bounds; this module was the only one that read
    `[1.000, 1.000]` as decisive, and printed "the ordering is not a sampling
    artefact at this n" and "every degradation is excluded at 95%" off it.

    Both bounds must be present for the question to be asked at all: a missing
    interval is the `interval` guard's business, not this one's.
    """
    if not (_finite(lo) and _finite(hi)):
        return False
    return float(hi) - float(lo) <= 0.0


def _f(v: Any) -> float:
    """Coerce to float for a `stats.phrase_*` call, or NaN.

    The phrase functions guard themselves with `np.isfinite`, which handles NaN
    and refuses to handle `None`: `ufunc 'isfinite' not supported`. A null CI
    bound in the stats JSON therefore crashed the whole run *after* audit,
    aggregation, stats, plots and tables had all completed, leaving a bundle
    with no `REPORT.md`. A missing bound is a normal state of the data, so it
    reaches the phrase function as the NaN it means.
    """
    return float(v) if _finite(v) else float('nan')


def _int_or_none(v: Any) -> Optional[int]:
    """`int(v)` or None, never an exception.

    `int(float('nan'))` raises, and `stats.py` writes its JSON with Python's
    `json` module, which emits and parses a bare `NaN`, so a NaN n round-trips
    into the n<3 guard. A guardrail that raises instead of refusing is the one
    failure mode `claim()` must not have: the sentence would not be refused,
    the whole report would be lost.
    """
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f or abs(f) == float('inf'):
        return None
    return int(f)


#: The source policy in force, set once by `main` from `--source-policy`.
#: Module state rather than an evidence key on purpose: a per-call-site key is
#: something a template author forgets, and the failure is silent, which is
#: exactly how 106 claims came to be emitted under the pooled secondary
#: estimand with nothing in their text saying so (`DESIGN.md` 4.3).
_SOURCE_POLICY: str = 'valid'


def set_source_policy(policy: str) -> None:
    """Record the source policy every subsequent claim was estimated under."""
    global _SOURCE_POLICY
    _SOURCE_POLICY = str(policy or 'valid')


def phrase_relative_effect(name_a: str, name_b: str, theta: float,
                           lo: Any = None, hi: Any = None) -> str:
    """RQ1's sentence, with the verb chosen by theta against the 0.5 null.

    `DESIGN.md` 9: direction sentences are generated from the data. theta is
    the Brunner-Munzel relative effect P(A > B) + 0.5 P(A = B), so its null is
    0.5 and not zero, and the interval is read against 0.5 for the same reason.
    `stats.phrase_interval_verdict` reads an interval against zero, so it is the
    wrong function here and using it would have printed "excludes zero" about a
    quantity that cannot be zero.

    No statistic is computed: theta and its bounds are read from the stats JSON,
    and this function only chooses words from their signs.
    """
    if not _finite(theta):
        return f'{name_a} versus {name_b}: no relative effect estimable'
    t = float(theta)
    if abs(t - 0.5) <= 1e-12:
        head = (f'a random run of {name_a} and a random run of {name_b} are '
                f'exactly level (theta={t:.3f})')
    else:
        word = 'above' if t > 0.5 else 'below'
        head = (f'a random run of {name_a} scores {word} a random run of '
                f'{name_b} more often than not (theta={t:.3f}, against a '
                f'null value of 0.5)')
    if not (_finite(lo) and _finite(hi)):
        return f'{head}; no interval emitted, so the ordering is not read'
    if _degenerate_interval(lo, hi):
        # Complete separation between the two cells gives theta_se = 0, which
        # forces the percentile fallback, and every resample of a separated
        # pair returns the same theta, so the percentile interval collapses to
        # a point. At n=3 per arm that is the most extreme of the 20 possible
        # arrangements and its exact two-sided permutation p is 0.10: the
        # sentence "not a sampling artefact at this n" was being asserted from
        # the one interval that says nothing about sampling at all.
        return (f'{head}; the 95% interval [{float(lo):.3f}, {float(hi):.3f}] '
                f'has zero width, so no sampling uncertainty was estimable and '
                f'the ordering is not read')
    flo, fhi = float(lo), float(hi)
    if flo > 0.5:
        tail = (f'the 95% interval [{flo:.3f}, {fhi:.3f}] lies wholly above '
                f'0.5, so the ordering is not a sampling artefact at this n')
    elif fhi < 0.5:
        tail = (f'the 95% interval [{flo:.3f}, {fhi:.3f}] lies wholly below '
                f'0.5, so the ordering is not a sampling artefact at this n')
    else:
        tail = (f'the 95% interval [{flo:.3f}, {fhi:.3f}] covers 0.5, so the '
                f'ordering is not distinguishable at this n')
    return f'{head}; {tail}'


def phrase_ratio_interval(lo: Any, hi: Any, null: float = 1.0,
                          label: str = 'the ratio') -> str:
    """Whether a ratio interval excludes its null, which is 1 and not zero."""
    if not (_finite(lo) and _finite(hi)):
        return f'{label}: no interval emitted'
    if _degenerate_interval(lo, hi):
        return (f'{label} 95% interval [{float(lo):.4f}, {float(hi):.4f}] has '
                f'zero width, so no sampling uncertainty was estimable and the '
                f'direction is not read')
    flo, fhi = float(lo), float(hi)
    if flo > null:
        return (f'{label} 95% interval [{flo:.4f}, {fhi:.4f}] lies wholly '
                f'above {null:g}')
    if fhi < null:
        return (f'{label} 95% interval [{flo:.4f}, {fhi:.4f}] lies wholly '
                f'below {null:g}')
    return (f'{label} 95% interval [{flo:.4f}, {fhi:.4f}] covers {null:g}, so '
            f'the direction is not distinguishable at this n')


def implied_equivalence_verdict(lo: Any, hi: Any, margin: float) -> str:
    """The verdict the interval and the margin imply, by containment.

    The same rule `stats.section_equivalence` applies, restated here so that the
    verdict `claim()` is handed can be checked against the numbers beside it
    rather than trusted. `ANALYSIS_PLAN.md` 4 defines the procedure as
    containment of the 95% bootstrap CI in +/-margin, and a caller that can
    supply a verdict string can supply the wrong one: an interval of
    [-0.30, -0.10] was accepted as EQUIVALENT against a +/-0.05 margin, which is
    the affirmed null this module exists to refuse.

    The branch order is `stats.section_equivalence`'s own, degeneracy first.
    Without that branch this function returned EQUIVALENT for [0.0000, 0.0000],
    because a point lies inside every margin: the re-derivation written to stop
    an affirmed null was itself affirming one, and a `stats.json` written before
    `stats.py` grew its DEGENERATE branch carries exactly that row.
    """
    if not (_finite(lo) and _finite(hi)):
        return 'no interval'
    if _degenerate_interval(lo, hi):
        return 'DEGENERATE'
    flo, fhi = float(lo), float(hi)
    if flo > -margin and fhi < margin:
        return 'EQUIVALENT'
    if flo >= margin or fhi <= -margin:
        return 'DIFFERENT'
    return 'INCONCLUSIVE'


def _blank_suppressed(rows: Sequence[dict]) -> list[dict]:
    """Blank the estimate columns of any row `stats.py` suppressed.

    `DESIGN.md` 8.4 and `ANALYSIS_PLAN.md` 9: a suppression is a refusal with a
    reason, never a null. The reason and the identity of the cell stay; the
    point estimate goes, because a table that prints `mean_delta = -0.6545`
    beside "refuses to aggregate runs that differ in an invariant" has published
    the number the suppression withheld, and the reader will quote it.
    """
    out: list[dict] = []
    for row in rows:
        if not row.get('suppressed'):
            out.append(row)
            continue
        blanked = dict(row)
        for column in SUPPRESSED_BLANK_COLUMNS:
            if column in blanked:
                blanked[column] = None
        out.append(blanked)
    return out


def _socratic(kind: str, estimand: str, evidence: dict
              ) -> tuple[tuple[str, str, str], ...]:
    """Answer the four `STANDING_INSTRUCTIONS` S5 questions from the evidence.

    An unanswered question is named as unanswered. Silence would let a claim
    pass by omitting the awkward question, which is precisely the failure mode
    S5 exists to prevent.
    """
    licensed, where = ESTIMAND_INFERENCE.get(estimand, ('unknown', 'unknown'))
    n = evidence.get('n')
    cf = evidence.get('counterfactual')
    if cf:
        a1 = (f'{cf}. In the data: {n if n is not None else "unknown"} '
              f'matched unit(s)'
              + (f', seeds {evidence["seeds"]}' if evidence.get('seeds') else '')
              + '.')
    else:
        a1 = 'UNANSWERED: the evidence names no counterfactual.'
    rivals = evidence.get('rivals')
    excl = evidence.get('excluded_by')
    if rivals:
        a2 = ('Rival explanations: ' + '; '.join(rivals) + '. '
              + (f'Excluded by: {excl}.' if excl
                 else 'NOT EXCLUDED by any control in this bundle.'))
    else:
        a2 = 'UNANSWERED: the evidence names no rival explanation.'
    ref = evidence.get('refuted_by')
    a3 = ref if ref else ('UNANSWERED: nothing is stated that would refute '
                          'this, so it is not a finding (S5 Q3).')
    if licensed == 'unknown':
        a4 = f'UNANSWERED: estimand {estimand!r} is not in the design RQ table.'
    elif kind == licensed or estimand in INTERVAL_KIND_ESTIMANDS.get(kind, ()):
        a4 = (f'yes: the wording is {kind!r} and the design licenses '
              f'{licensed!r} for estimand {estimand!r} ({where}).')
    else:
        a4 = (f'NO: the wording is {kind!r} but the design licenses '
              f'{licensed!r} for estimand {estimand!r} ({where}).')
    return tuple((tag, q, a) for (tag, q), a
                 in zip(SOCRATIC_QUESTIONS, (a1, a2, a3, a4)))


def claim(text: str, kind: str, evidence: dict) -> Claim:
    """Emit a sentence, or refuse it and say which section forbids it.

    `text` is the sentence as proposed. `kind` is the inference type the
    sentence performs. `evidence` is the numbers and the provenance behind it,
    normally lifted straight out of `stats.py --json`.

    The return value is always a `Claim`: an accepted one carries the text that
    may be used, a refused one carries the reasons and each reason's section
    citation. Nothing is silently downgraded, because a silently weakened
    sentence is how the published study came to describe a null as an effect.

    Recognised keys in `evidence` (all optional unless a rule below needs one):

    ``estimand``       one of `ESTIMAND_INFERENCE`; required.
    ``n``              units behind the number; below 3 the claim is refused.
    ``ci_lo``/``ci_hi`` the interval; required for any directional kind.
    ``verdict``        the `stats.section_equivalence` verdict; required for
                       kind='equivalence', and re-derived here from the interval
                       and the margin rather than trusted.
    ``margin``         the pre-registered equivalence margin the verdict was
                       taken against; defaults to `stats.EQUIVALENCE_MARGIN`.
    ``signal``         a `DESIGN.md` 5.5 signal name; required for
                       kind='mechanism'.
    ``sd_a``/``sd_b``  the two dispersions; required for kind='dispersion', and
                       used to check an UNTESTABLE equivalence verdict against
                       the feasibility rule that produced it.
    ``theta``          a relative effect; where present the sentence is
                       generated from it and a hand-written direction verb that
                       contradicts it is refused.
    ``between_cell_contrast``  an explicit contrast with an interval; the only
                       thing that licenses an "A avoids it, B does not" shape.
    ``confirmatory``   True only for a member of the eight-test family.
    ``counterfactual``/``rivals``/``excluded_by``/``refuted_by``  the S5
                       answers; ``refuted_by`` is required for every kind whose
                       inference is causal, mechanistic or an equivalence.
    ``seed_block``     refused when it is TUNE.
    ``scale``          refused when it is 'raw_return'.
    """
    refusals: list[tuple[str, str, str]] = []
    estimand = str(evidence.get('estimand') or '')
    proposed = text
    generated = False
    out = text.strip()

    # --- vocabulary ------------------------------------------------------
    if kind not in KINDS:
        refusals.append(('kind', 'report.py 1',
                         f'unknown claim kind {kind!r}; known kinds are '
                         f'{", ".join(KINDS)}'))
    if estimand not in ESTIMAND_INFERENCE:
        refusals.append(('estimand', 'DESIGN.md 2.4',
                         f'estimand {estimand!r} is not in the design research-'
                         f'question table, so no inference type is licensed for '
                         f'it. Known: {", ".join(sorted(ESTIMAND_INFERENCE))}'))
    else:
        licensed, where = ESTIMAND_INFERENCE[estimand]
        interval_ok = estimand in INTERVAL_KIND_ESTIMANDS.get(kind, ())
        if kind != licensed and not interval_ok:
            if kind in INTERVAL_KINDS:
                # The escape hatch, closed. An exclusion bound is a statement
                # in normalised-score units about a paired contrast; asserted
                # over a proportion, an SD ratio or a gradient-norm signal it
                # is a bound in the wrong units, and it also let a sentence
                # about a mechanism signal skip the DESIGN.md 5.5 requirement
                # to name one.
                refusals.append((
                    'interval-kind-estimand', 'DESIGN.md 2.4, ANALYSIS_PLAN.md 4',
                    f'{kind!r} is a statement about a paired '
                    f'normalised-score interval and may be asserted only about '
                    f'{", ".join(INTERVAL_KIND_ESTIMANDS.get(kind, ())) or "no estimand"}; '
                    f'estimand {estimand!r} licenses {licensed!r} wording '
                    f'({where})'))
            else:
                refusals.append(('inference-binding', where,
                                 f'estimand {estimand!r} licenses {licensed!r} '
                                 f'wording; this sentence performs {kind!r}. '
                                 f'Inference type is binding on wording'))

    # --- the kind-specific evidence requirements -------------------------
    if kind == 'causal' and estimand not in CAUSAL_ESTIMANDS:
        refusals.append(('causal-estimand', 'DESIGN.md 2.4 RQ2, 9',
                         'a causal claim is licensed only for the within-cell '
                         'paired delta, whose warrant is ceteris paribus at '
                         'matched seeds; a between-cell quantity is effect '
                         'modification or an association, never a cause'))
    if kind == 'causal_component' and not evidence.get('manipulated'):
        refusals.append(('component-manipulation', 'DESIGN.md 4.1',
                         'a component claim must name what was manipulated; '
                         'the contrasts are named after the manipulation, '
                         'never after a mechanism'))
    if kind == 'mechanism':
        signal = evidence.get('signal')
        if signal not in DESIGN_5_5_SIGNALS:
            refusals.append(('mechanism-signal', 'DESIGN.md 5.5, 9',
                             f'a mechanism claim must cite an instrumented '
                             f'signal; {signal!r} is not one of '
                             f'{", ".join(DESIGN_5_5_SIGNALS)}. There is no '
                             f'free-text mechanism slot'))
    if kind == 'equivalence':
        verdict = evidence.get('verdict')
        margin = evidence.get('margin')
        margin = (float(margin) if _finite(margin)
                  else float(stats_mod.EQUIVALENCE_MARGIN))
        if verdict not in EQUIVALENCE_VERDICTS:
            refusals.append(('equivalence-verdict', 'ANALYSIS_PLAN.md 4',
                             f'an equivalence claim requires a verdict from the '
                             f'interval-containment procedure; got '
                             f'{verdict!r}. A large p-value, or the absence of '
                             f'one, is not a verdict'))
        elif verdict != 'EQUIVALENT' and _hits(r'\bequivalent\b', out):
            refusals.append(('equivalence-assertion', 'ANALYSIS_PLAN.md 4',
                             f'the verdict is {verdict!r}, so the sentence may '
                             f'not assert equivalence; the licensed statement '
                             f'is the exclusion bound'))
        # The verdict is a computation, so it is re-derived here from the two
        # numbers printed beside it. Trusting the string accepted
        # [-0.30, -0.10] as EQUIVALENT against a +/-0.05 margin: an interval
        # wholly outside the margin, excluding zero, published as an affirmed
        # null. ANALYSIS_PLAN.md 4 defines the verdict as containment, and a
        # definition that is never evaluated is a comment.
        sds = (evidence.get('sd_a'), evidence.get('sd_b'))
        worst_sd = max((float(v) for v in sds if _finite(v)),
                       default=float('nan'))
        if verdict in ('EQUIVALENT', 'DIFFERENT', 'INCONCLUSIVE'):
            implied = implied_equivalence_verdict(
                evidence.get('ci_lo'), evidence.get('ci_hi'), margin)
            if implied != verdict:
                refusals.append((
                    'equivalence-containment', 'ANALYSIS_PLAN.md 4',
                    f'the supplied verdict is {verdict!r}, but containment of '
                    f'[{f_(evidence.get("ci_lo"))}, {f_(evidence.get("ci_hi"))}] '
                    f'in +/-{margin} implies {implied!r}. The verdict is the '
                    f'containment relation, not a label attached to it'))
        # The feasibility gate, and only over the verdicts it governs. See
        # DISPERSION_GATED_VERDICTS: a DIFFERENT verdict is a difference
        # statement, which ANALYSIS_PLAN.md 4 does not gate on dispersion and
        # stats.py deliberately evaluates first. Refusing it printed "the
        # verdict may only be UNTESTABLE" directly beneath a table row reading
        # DIFFERENT, in the two cells with the largest effects.
        if verdict in DISPERSION_GATED_VERDICTS:
            if _finite(worst_sd) and worst_sd > margin:
                refusals.append((
                    'equivalence-feasibility', 'ANALYSIS_PLAN.md 4',
                    f'an across-seed SD of {worst_sd:.4f} exceeds the margin '
                    f'{margin}, so equivalence is untestable in this cell at '
                    f'this n and the verdict may only be UNTESTABLE'))
        if verdict == 'EQUIVALENT' and not all(_finite(v) for v in sds):
            # Fail CLOSED. The gate used to be reached only through
            # `if _finite(worst_sd)`, so omitting the two SDs, or passing them
            # as null or NaN, bought an unconditional pass -- and stats.py
            # produces exactly that dict, because sd() returns NaN for an arm
            # with fewer than two usable rows and its own gate is guarded the
            # same way. The result was an affirmed null in precisely the cell
            # where dispersion could not be estimated.
            refusals.append((
                'equivalence-feasibility', 'ANALYSIS_PLAN.md 4',
                f'an EQUIVALENT verdict requires the two across-seed SDs the '
                f'feasibility gate is evaluated on; got sd_a={f_(sds[0])} and '
                f'sd_b={f_(sds[1])}. A dispersion that could not be estimated '
                f'is not a dispersion below the margin'))
        if verdict == 'UNTESTABLE':
            if _finite(worst_sd) and worst_sd <= margin:
                refusals.append((
                    'equivalence-feasibility', 'ANALYSIS_PLAN.md 4',
                    f'the verdict is UNTESTABLE, but the largest across-seed '
                    f'SD supplied is {worst_sd:.4f}, inside the margin '
                    f'{margin}: the feasibility gate that produces UNTESTABLE '
                    f'did not fire on these numbers'))
        if evidence.get('basis') == 'p_value':
            refusals.append(('equivalence-basis', 'ANALYSIS_PLAN.md 4, 8',
                             'equivalence read off a p-value is the forbidden '
                             'inference; the procedure is containment of the '
                             '95% bootstrap CI in the +/-0.05 margin'))
    if kind == 'dispersion':
        sd_a, sd_b = evidence.get('sd_a'), evidence.get('sd_b')
        if not (_finite(sd_a) and _finite(sd_b)):
            refusals.append(('dispersion-evidence', 'DESIGN.md 9, 5.3',
                             'a dispersion sentence is generated from two '
                             'across-seed SDs; they are not both present'))
        else:
            generated_text = stats_mod.phrase_dispersion(
                str(evidence.get('name_a', 'arm A')), float(sd_a),
                str(evidence.get('name_b', 'arm B')), float(sd_b))
            wider = float(sd_a) > float(sd_b)
            claimed_wider = any(_hits(p, out) for p in DISPERSION_WIDER)
            claimed_narrower = any(_hits(p, out) for p in DISPERSION_NARROWER)
            if (claimed_wider and not wider) or (claimed_narrower and wider):
                refusals.append(('dispersion-direction', 'DESIGN.md 9',
                                 f'the adjective in the proposed sentence '
                                 f'contradicts the numbers (SD {float(sd_a):.4f} '
                                 f'vs {float(sd_b):.4f}); dispersion and '
                                 f'direction sentences are generated from the '
                                 f'data'))
            # ANALYSIS_PLAN.md 3 defines the dispersion endpoint as a ratio of
            # SDs *with a bootstrap CI*, so the ratio is never asserted bare.
            # The interval is read against 1, the ratio's null.
            out = (f'{generated_text}; '
                   + phrase_ratio_interval(evidence.get('ci_lo'),
                                           evidence.get('ci_hi'), 1.0,
                                           'the SD ratio') + '.')
            generated = True

    # --- a relative effect is directional, so its verb is generated -------
    # DESIGN.md 9: direction sentences come out of the data. A hand-written
    # "exceeds" shipped over theta=0.469 and theta=0.333, asserting the
    # opposite of the number in the same sentence that printed it.
    if _finite(evidence.get('theta')):
        theta = float(evidence['theta'])
        claimed_higher = any(_hits(p, out) for p in RELATIVE_EFFECT_HIGHER)
        claimed_lower = any(_hits(p, out) for p in RELATIVE_EFFECT_LOWER)
        higher = theta > 0.5
        if (claimed_higher and not higher) or (claimed_lower and higher):
            refusals.append(('relative-effect-direction', 'DESIGN.md 9',
                             f'the direction verb in the proposed sentence '
                             f'contradicts the relative effect (theta='
                             f'{theta:.3f}, against a null value of 0.5); '
                             f'direction sentences are generated from the '
                             f'data'))
        out = (f'{str(evidence.get("prefix") or "").strip()} '
               + phrase_relative_effect(
                   str(evidence.get('name_a', 'arm A')),
                   str(evidence.get('name_b', 'arm B')), theta,
                   evidence.get('ci_lo'), evidence.get('ci_hi'))).strip()
        out = out[0].upper() + out[1:] + '.' if out else out
        generated = True

    # --- interval discipline ---------------------------------------------
    # Dispersion is in this list because ANALYSIS_PLAN.md 3 makes the SD ratio
    # an interval endpoint: a ratio asserted with no interval ("2.03x
    # narrower") is a direction read off two point estimates.
    # A relative effect is directional whatever kind carries it: RQ1 is an
    # association, and all six of its sentences used to assert an ordering with
    # no interval verdict at all while [0.1311, 0.8254] straddled the theta=0.5
    # null.
    directional = (kind in ('causal', 'causal_component',
                            'effect_modification', 'exclusion', 'equivalence',
                            'dispersion')
                   or _finite(evidence.get('theta')))
    if directional and not (_finite(evidence.get('ci_lo'))
                            and _finite(evidence.get('ci_hi'))):
        if kind == 'equivalence' and evidence.get('verdict') in (
                'UNTESTABLE', 'suppressed', 'no interval'):
            pass                      # the verdict *is* that there is none
        else:
            refusals.append(('interval', 'ANALYSIS_PLAN.md 3',
                             'a directional statement needs the interval it is '
                             'read from; every estimate in this study is '
                             'reported as a point estimate with a seed-level '
                             'bootstrap interval'))
    # A present interval of zero width satisfies the rule above vacuously, and
    # that is how "the ordering is not a sampling artefact at this n" came to be
    # printed off [1.000, 1.000] and "every degradation is excluded at 95%" off
    # [+0.0000, +0.0000]. Degeneracy is an ABSENCE of information, so it is
    # refused here rather than downgraded: DESIGN.md 9's licensed statement in
    # that cell is that nothing was estimable.
    degenerate = _degenerate_interval(evidence.get('ci_lo'),
                                      evidence.get('ci_hi'))
    if degenerate:
        refusals.append((
            'degenerate-interval', 'ANALYSIS_PLAN.md 3, DESIGN.md 9',
            f'the interval [{f_(evidence.get("ci_lo"))}, '
            f'{f_(evidence.get("ci_hi"))}] has zero width: every bootstrap '
            f'replicate is identical, so no sampling uncertainty was estimable '
            f'and the interval licenses neither a direction, an exclusion '
            f'bound nor an equivalence verdict. A point is not a precise '
            f'interval'))
    if kind == 'exclusion':
        lo = evidence.get('ci_lo')
        # `exclusion_bound` is stats.py's own answer, and it is None exactly
        # where stats.py withheld the bound. Re-deriving the sentence from
        # ci_lo ignored that: a degenerate cell for which stats.py had written
        # "no equivalence or exclusion claim is available at all" was published
        # as "every degradation is excluded at 95%: the interval lies at or
        # above +0.0000 score units".
        withheld = ('exclusion_bound' in evidence
                    and evidence.get('exclusion_bound') is None)
        if withheld and _finite(lo):
            refusals.append((
                'exclusion-bound', 'ANALYSIS_PLAN.md 4, DESIGN.md 9',
                'the section that computed this interval withheld its '
                'exclusion bound, so there is no bound to state; a bound '
                're-derived here from the lower limit would contradict the '
                'module that declined to publish one'))
        if _finite(lo) and not (degenerate or withheld):
            out = f'{out.rstrip(". ")}: {stats_mod.phrase_exclusion_bound(float(lo))}.'
            generated = True
        elif degenerate or withheld:
            out = (f'{out.rstrip(". ")}: no exclusion bound is available, '
                   f'because nothing about sampling uncertainty was estimable '
                   f'in this cell.')
            generated = True

    # --- sample size, seed block, scale ----------------------------------
    n = evidence.get('n')
    n_int = _int_or_none(n)
    if n is None:
        refusals.append(('n-missing', 'ANALYSIS_PLAN.md 9',
                         'the number of units behind the claim is not stated, '
                         'so the n<3 guard cannot be applied'))
    elif n_int is None:
        # int(float('nan')) raises, and stats.py's JSON round-trips a bare NaN.
        # A guard that throws does not refuse the sentence, it loses the report.
        refusals.append(('n-missing', 'ANALYSIS_PLAN.md 9',
                         f'the stated n is {n!r}, which is not a count, so the '
                         f'n<3 guard cannot be applied and the claim is '
                         f'refused rather than the guard skipped'))
    elif n_int < stats_mod.MIN_N_FOR_INFERENCE:
        refusals.append(('n<3', 'ANALYSIS_PLAN.md 9, STANDING_INSTRUCTIONS S8',
                         f'n={n_int} < {stats_mod.MIN_N_FOR_INFERENCE}: no '
                         f'test and no interval is emitted, and such a number '
                         f'may not be quoted, compared, or used to choose '
                         f'between hypotheses'))
    if str(evidence.get('seed_block', '')).upper() == 'TUNE':
        refusals.append(('TUNE', 'DESIGN.md 3.4, ANALYSIS_PLAN.md 8',
                         'the estimate draws on the TUNE block, which is '
                         'reserved for hyperparameter selection and may never '
                         'enter a reported estimate'))
    if str(evidence.get('scale', '')) == 'raw_return':
        refusals.append(('raw-return', 'DESIGN.md 5.1, ANALYSIS_PLAN.md 8',
                         'raw returns are not comparable across environments '
                         'or variants -- the random-policy reference moves by '
                         'hundreds of points -- so a claim must be on the '
                         'normalised score'))

    # --- what the sentence itself says ------------------------------------
    for pattern, where, why in FORBIDDEN_PHRASES:
        if _hits(pattern, out):
            refusals.append(('forbidden-phrase', where,
                             f'forbidden phrasing: {why}'))
    for pattern in VERDICT_COMPARISON_PATTERNS:
        if _hits(pattern, out):
            contrast = evidence.get('between_cell_contrast') or {}
            if not (_finite(contrast.get('ci_lo'))
                    and _finite(contrast.get('ci_hi'))):
                refusals.append((
                    'verdict-comparison',
                    'ANALYSIS_PLAN.md 8, DESIGN.md 9',
                    'the sentence form "A avoids it, B does not" compares two '
                    'significance verdicts; the licensed form is the '
                    'between-cell contrast delta_X - delta_Y with its interval, '
                    'and no such contrast is in the evidence'))
            break
    if _hits(PVALUE_PATTERN, out) and not (
            evidence.get('confirmatory') and evidence.get('p_holm') is not None):
        refusals.append(('p-value', 'ANALYSIS_PLAN.md 2, 7',
                         'a p-value or a significance word appears in a '
                         'sentence that is not a member of the eight-test '
                         'confirmatory family; everything else is '
                         'estimation-only and carries no p-value'))
    if _hits(NULL_AFFIRMING_PATTERN, out) and not (
            kind == 'equivalence' and evidence.get('verdict') == 'EQUIVALENT'):
        refusals.append(('null-affirming', 'DESIGN.md 9, ANALYSIS_PLAN.md 4',
                         'the sentence affirms a null; a non-significant result '
                         'renders as "not distinguishable" and the licensed '
                         'positive statement is what the interval excludes'))
    if kind in ('associational', 'effect_modification', 'descriptive',
                'dispersion') and _hits(CAUSAL_VERB_PATTERN, out):
        refusals.append(('causal-verb', 'DESIGN.md 2.4, 9',
                         f'a causal verb appears in a {kind!r} sentence; cells '
                         f'are different algorithms rather than treatments '
                         f'assigned to units, and a between-cell contrast is '
                         f'effect modification, not causation'))

    # --- S5 Q3: a claim nothing could refute is not a finding -------------
    if kind in ('causal', 'causal_component', 'mechanism', 'equivalence') \
            and not evidence.get('refuted_by'):
        refusals.append(('refutability', 'STANDING_INSTRUCTIONS S5 Q3',
                         'the evidence does not state what would refute this '
                         'claim; a claim nothing could refute is not a finding'))

    # --- the labels a sentence has to carry out of the bundle -------------
    # DESIGN.md 4.3: the pooled-source estimand is the pre-declared *secondary*
    # one and must be labelled exactly that. Recorded in the inventory is not
    # labelled: a sentence lifted into the manuscript carries only its words.
    policy = str(evidence.get('source_policy') or _SOURCE_POLICY)
    if not refusals and policy == 'pooled':
        out = f'{POOLED_SOURCE_LABEL} {out}'
        generated = True
    # DESIGN.md 2.1: "Where a sentence in the report would generalise past
    # this, report.py prefixes it with this clause or refuses to emit it." It
    # used to be prefixed for kind='causal' alone, and a bundle with no causal
    # claim in it therefore emitted 106 sentences with no scope statement at
    # all. Deciding per sentence whether it generalises is precisely the
    # judgement that fails silently, so every accepted claim carries it.
    if not refusals:
        out = f'{SCOPE_CLAUSE} {out}'
        generated = True

    return Claim(text=out, kind=kind, estimand=estimand, evidence=evidence,
                 accepted=not refusals, refusals=tuple(refusals),
                 socratic=_socratic(kind, estimand, evidence),
                 generated=generated,
                 proposed=(proposed if proposed != out else None))


def render_claim_lines(c: Claim) -> list[str]:
    """The plain-text rendering used for stdout; `REPORT.md` mirrors it."""
    head = 'CLAIM' if c.accepted else 'REFUSED'
    lines = [f'  {head} [{c.kind} / {c.estimand}]'
             + (f' -- {c.section}' if c.section else '')]
    lines += ['    ' + l for l in textwrap.wrap(c.text, 88)]
    if not c.accepted:
        for guard, where, why in c.refusals:
            lines += ['      ' + l for l in
                      textwrap.wrap(f'refused by {where} [{guard}]: {why}', 86)]
    for tag, question, answer in c.socratic:
        lines += ['      ' + l for l in
                  textwrap.wrap(f'{tag} {question}', 86)]
        lines += ['          ' + l for l in textwrap.wrap(answer, 82)]
    return lines


def render_claim_md(c: Claim) -> list[str]:
    head = ('**CLAIM** (`%s`, estimand `%s`)' % (c.kind, c.estimand)
            if c.accepted else
            '**REFUSED** (`%s`, estimand `%s`)' % (c.kind, c.estimand))
    lines = [f'> {head}', '>', f'> {c.text}', '>']
    if c.proposed:
        lines += [f'> *Proposed text, replaced by generated wording:* '
                  f'{c.proposed}', '>']
    if not c.accepted:
        for guard, where, why in c.refusals:
            lines.append(f'> - refused by **{where}** (guard `{guard}`): {why}')
        lines.append('>')
    for tag, question, answer in c.socratic:
        lines.append(f'> - *{tag} {question}* {answer}')
    lines.append('')
    return lines


# ===========================================================================
# 3. Stage execution. The analysis modules are run as child processes so that
#    the exact command is recorded and reproducible, and so that a crash in one
#    stage cannot leave this process holding half-initialised module state.
#    `audit` is the exception: `DESIGN.md` 8.4 makes it a *gate*, and a gate is
#    a function call returning (ok, report), not a parsed exit code.
# ===========================================================================
@dataclass
class Stage:
    name: str
    command: list[str]
    exit_code: Optional[int]
    seconds: float
    log: Optional[str]
    ok: bool
    note: str = ''


def run_stage(name: str, script: str, args: Sequence[str], log_path: str,
              quiet: bool) -> Stage:
    cmd = [sys.executable, '-u', os.path.join('experiments', script),
           *[str(a) for a in args]]
    env = dict(os.environ)
    # Windows consoles default to a legacy code page; the analysis modules print
    # section signs and arrows, and a pipe would otherwise raise rather than
    # print. Recorded here rather than assumed.
    env['PYTHONIOENCODING'] = 'utf-8'
    started = time.time()
    print(f'\n== {name} ==')
    print('   ' + ' '.join(cmd))
    proc = subprocess.run(cmd, cwd=_REPO, env=env, capture_output=True)
    seconds = time.time() - started
    text = (proc.stdout or b'').decode('utf-8', errors='replace')
    err = (proc.stderr or b'').decode('utf-8', errors='replace')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'w', encoding='utf-8') as fh:
        fh.write('$ ' + ' '.join(cmd) + '\n\n')
        fh.write(text)
        if err.strip():
            fh.write('\n---- stderr ----\n')
            fh.write(err)
    tail = [l for l in text.splitlines() if l.strip()][-8:]
    if not quiet:
        for line in tail:
            print('   | ' + line[:160])
    print(f'   exit={proc.returncode}  {seconds:.1f}s  log -> '
          f'{rel(log_path, _REPO)}')
    return Stage(name=name, command=cmd, exit_code=proc.returncode,
                 seconds=seconds, log=log_path, ok=proc.returncode == 0)


def probe_options(script: str) -> set[str]:
    """Long options a sibling module advertises, so a hand-off can be built
    from what exists rather than from what this file assumes exists."""
    try:
        proc = subprocess.run(
            [sys.executable, os.path.join('experiments', script), '--help'],
            cwd=_REPO, capture_output=True, timeout=180)
    except (OSError, subprocess.SubprocessError):
        return set()
    text = (proc.stdout or b'').decode('utf-8', errors='replace')
    return set(re.findall(r'(--[a-z][a-z0-9-]+)', text))


# ===========================================================================
# 4. Small formatting helpers. No statistic is computed here.
# ===========================================================================
def f_(value: Any, nd: int = 4) -> str:
    if value is None:
        return '--'
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if value != value:
            return '--'
        return f'{value:.{nd}f}'
    if isinstance(value, (list, tuple)):
        return ','.join(str(v) for v in value) or '--'
    text = str(value)
    return text if text else '--'


def md_table(rows: Sequence[dict], columns: Sequence[str], nd: int = 4
             ) -> list[str]:
    if not rows:
        return ['*(no rows)*', '']
    head = '| ' + ' | '.join(columns) + ' |'
    rule = '|' + '|'.join('---' for _ in columns) + '|'
    out = [head, rule]
    for row in rows:
        # A pipe ends the cell and a newline ends the row, so both have to go:
        # a suppression reason or an audit finding arriving with an embedded
        # newline would otherwise silently break the table into fragments and
        # take the rest of the rows with it.
        out.append('| ' + ' | '.join(
            f_(row.get(c), nd).replace('|', '\\|')
            .replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ')
            for c in columns) + ' |')
    out.append('')
    return out


def write_csv(path: str, rows: Sequence[dict], columns: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns),
                                extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c) for c in columns})


def wrap(text: str, width: int = 88) -> list[str]:
    return textwrap.wrap(text, width) or ['']


def csv_columns(path: str) -> list[str]:
    try:
        with open(path, 'r', encoding='utf-8', newline='') as fh:
            return next(csv.reader(fh))
    except (OSError, StopIteration):
        return []


def target_scratch_seed_count(sj: dict, cell: str) -> Optional[int]:
    """Distinct seeds in a cell's scratch arms on the target task, or None.

    RQ1's `n_a` and `n_b` are row counts: on a three-seed tree they read 9 and
    6, because several labels at the same three seeds are pooled into the arm.
    The n<3 guard is about units, and the unit here is a seed, so passing the
    row count let a claim through at a seed count the design refuses. Returning
    None where the seed set cannot be recovered makes `claim()` refuse the
    sentence rather than fall back to the number that fails open.
    """
    invocation = sj.get('invocation') or {}
    target = str(invocation.get('target_env') or '').strip()
    if not target:
        return None
    seeds: set[str] = set()
    seen = False
    for arm in (sj.get('s2_inventory') or {}).get('arms') or []:
        # The same filter stats.scratch_arm applies: a C4SRC or RESERVE run
        # sits on the target environment but exists only to donate a source
        # checkpoint, and pooling it into the baseline would both inflate this
        # count and undo the independence the positive control is built on.
        if (str(arm.get('cell')) != cell
                or str(arm.get('condition')) != 'scratch'
                or str(arm.get('env')) != target
                or str(arm.get('seed_block')) in stats_mod.SOURCE_ONLY_BLOCKS):
            continue
        seen = True
        raw = arm.get('seeds')
        tokens = (raw if isinstance(raw, (list, tuple))
                  else str(raw or '').split(','))
        seeds.update(str(t).strip() for t in tokens if str(t).strip())
    return len(seeds) if seen else None


def csv_row_count(path: str) -> int:
    try:
        with open(path, 'r', encoding='utf-8', newline='') as fh:
            return max(sum(1 for _ in fh) - 1, 0)
    except OSError:
        return 0


# ===========================================================================
# 5. The tables. Each is written as CSV beside a provenance record carrying the
#    hash of the inputs it was built from (`DESIGN.md` 8.3), and mirrored into
#    `REPORT.md` from the same rows, so a table and the prose beside it cannot
#    disagree.
# ===========================================================================
@dataclass
class TableSpec:
    name: str
    title: str
    rows: list[dict]
    columns: tuple[str, ...]
    why: str


def _flatten_controls(controls: dict) -> list[dict]:
    out: list[dict] = []
    for metric, block in (controls or {}).items():
        for cell, payload in (block.get('cells') or {}).items():
            for contrast in payload.get('contrasts') or []:
                out.append({
                    'metric': metric, 'cell': cell,
                    'contrast': contrast.get('contrast'),
                    'name': contrast.get('name'), 'n': contrast.get('n'),
                    'mean': contrast.get('mean'), 'hl': contrast.get('hl'),
                    'ci_lo': contrast.get('ci_lo'),
                    'ci_hi': contrast.get('ci_hi'),
                    'unanimous': contrast.get('unanimous'),
                    'protocol_freeze_updates':
                        payload.get('protocol_freeze_updates'),
                    'missing_conditions': ','.join(payload.get('missing') or []),
                })
    return out


def _flatten_censored(censored: dict) -> list[dict]:
    out: list[dict] = []
    for level_key, block in (censored or {}).items():
        for arm in block.get('arms') or []:
            row = dict(arm)
            row['level_key'] = level_key
            row['level'] = block.get('level')
            out.append(row)
    return out


def _mechanism_rows(secondary: dict) -> list[dict]:
    return [r for r in (secondary or {}).get('rows') or []
            if str(r.get('role')) == stats_mod.MECHANISM]


def _secondary_rows(secondary: dict) -> list[dict]:
    return [r for r in (secondary or {}).get('rows') or []
            if str(r.get('role')) != stats_mod.MECHANISM]


def build_tables(sj: dict) -> list[TableSpec]:
    """Every table in the bundle, built from the stats JSON and nothing else."""
    inv = sj.get('s2_inventory', {}) or {}
    desc = sj.get('s3_descriptives', {}) or {}
    conv = sj.get('s4_convergence', {}) or {}
    conf = sj.get('s5_confirmatory', {}) or {}
    equiv = sj.get('s6_equivalence', {}) or {}
    est = sj.get('s9_estimation', {}) or {}
    power = sj.get('s10_power', {}) or {}
    ledger = sj.get('s11_ledger', {}) or {}

    specs: list[TableSpec] = [
        TableSpec('inventory_arms', 'Run inventory, per arm',
                  inv.get('arms') or [],
                  ('env', 'cell', 'condition', 'label', 'n', 'seeds',
                   'seed_block', 'transfer_set', 'freeze_updates'),
                  'DESIGN.md 9 "silent seed dropping": the seed list is '
                  'printed, not summarised.'),
        TableSpec('inventory_completeness', 'Pairing completeness per cell',
                  inv.get('completeness') or [],
                  ('cell', 'n_scratch', 'n_transfer', 'paired_seeds',
                   'scratch_only', 'transfer_only', 'transfer_labels'),
                  'A partial arm is refused rather than averaged '
                  '(DESIGN.md 8.4).'),
        TableSpec('inventory_source_validity', 'Source-validity verdicts',
                  inv.get('source_validity') or [],
                  ('cell', 'condition', 'label', 'n', 'valid', 'invalid',
                   'invalid_seeds', 'source_score_mean'),
                  'DESIGN.md 4.3: the normalised gate at 0.6, with the number '
                  'and identity of rejected source seeds reported.'),
        TableSpec('inventory_intensity_gate',
                  'Transferred-parameter fraction, cross-architecture pairs',
                  inv.get('intensity_gate') or [],
                  ('a', 'b', 'frac_a', 'frac_b', 'abs_diff', 'verdict',
                   'permitted'),
                  'DESIGN.md 3.1: an arch contrast whose transferred fractions '
                  'differ beyond tolerance is intensity-confounded.'),
        TableSpec('convergence', 'Convergence gate, per arm',
                  conv.get('per_arm') or [],
                  ('cell', 'condition', 'label', 'n', 'median_slope', 'ci_lo',
                   'ci_hi', 'frac_positive', 'cp_lo', 'cp_hi', 'still_moving'),
                  'DESIGN.md 5.2: where runs have not converged, P1 is '
                  'performance at budget, not asymptotic performance.'),
        TableSpec('confirmatory', 'The confirmatory family: 8 tests',
                  _blank_suppressed(conf.get('members') or []),
                  ('metric', 'cell', 'n', 'scratch_mean', 'transfer_mean',
                   'mean_delta', 'hl', 'ci_lo', 'ci_hi', 'p_signflip',
                   'p_holm', 'significant_holm', 'p_wilcoxon',
                   'mannwhitney_U', 'p_mannwhitney', 'rho_pearson',
                   'rho_spearman', 'unanimous', 'suppressed'),
                  'ANALYSIS_PLAN.md 2: the only family that carries a p-value.'),
        TableSpec('equivalence', 'Equivalence and exclusion, per cell',
                  equiv.get('rows') or [],
                  ('metric', 'cell', 'n', 'ci_lo', 'ci_hi', 'sd_scratch',
                   'sd_transfer', 'margin', 'verdict', 'exclusion_bound',
                   'reason'),
                  'ANALYSIS_PLAN.md 4: containment in +/-0.05, never TOST, and '
                  'the exclusion bound whatever the verdict.'),
        TableSpec('controls', 'Control contrasts (joint seed bootstrap)',
                  _flatten_controls(sj.get('s7_controls', {})),
                  ('metric', 'cell', 'contrast', 'name', 'n', 'mean', 'hl',
                   'ci_lo', 'ci_hi', 'unanimous', 'protocol_freeze_updates',
                   'missing_conditions'),
                  'DESIGN.md 4.1: contrasts named after what was manipulated, '
                  'from one resampling so their correlations are estimated.'),
        TableSpec('c4_positive_control', 'C4 positive control',
                  (sj.get('s8_c4', {}) or {}).get('rows') or [],
                  ('cell', 'n', 'n_transfer', 'n_scratch', 'hl', 'ci_lo',
                   'ci_hi', 'verdict', 'reason'),
                  'DESIGN.md 4.2: pre-registered criterion, CI lower bound '
                  'above -0.10 on the interface-change-only pair.'),
        TableSpec('rq1_between_cell_scratch',
                  'RQ1 between-cell scratch comparison (Brunner-Munzel)',
                  (est.get('rq1') or {}).get('rows') or [],
                  ('a', 'b', 'n_a', 'n_b', 'mean_a', 'mean_b', 'sd_a', 'sd_b',
                   'theta', 'ci_lo', 'ci_hi', 'note'),
                  'ANALYSIS_PLAN.md 3: relative effect, because the cells SDs '
                  'differ by up to 8x and location shift is violated.'),
        TableSpec('rq3_between_cell_delta',
                  'RQ3 between-cell delta contrasts',
                  (est.get('rq3') or {}).get('pairs') or [],
                  ('a', 'b', 'n', 'hl', 'ci_lo', 'ci_hi', 'hl_headroom_adj',
                   'adj_lo', 'adj_hi', 'headroom_a', 'headroom_b', 'scales',
                   'note'),
                  'Estimation-only: MDE ~1.96 sigma_delta, so an interval and '
                  'no p-value (ANALYSIS_PLAN.md 3, 6).'),
        TableSpec('rq6_budget', 'RQ6 budget prefixes',
                  (est.get('rq6') or {}).get('rows') or [],
                  ('cell', 'prefix', 'n', 'delta_at_prefix', 'delta_at_budget'),
                  'DESIGN.md 2.4 RQ6: a prefix is compared like with like, '
                  'single checkpoint against single checkpoint.'),
        TableSpec('dispersion', 'Across-seed dispersion, SD ratio',
                  (est.get('dispersion') or {}).get('rows') or [],
                  ('cell', 'n', 'sd_scratch', 'sd_transfer', 'sd_ratio',
                   'ci_lo', 'ci_hi', 'brown_forsythe_W'),
                  'ANALYSIS_PLAN.md 3: no dispersion p-value is interpreted; '
                  'the ratio carries a bootstrap interval.'),
        TableSpec('censored_thresholds',
                  'Censored steps-to-threshold: P(reached by budget)',
                  _flatten_censored(est.get('censored') or {}),
                  ('level_key', 'level', 'cell', 'condition', 'label', 'n',
                   'reached', 'p_reached', 'cp_lo', 'cp_hi',
                   'km_median_steps'),
                  'ANALYSIS_PLAN.md 5: never imputed, never dropped; '
                  'Clopper-Pearson exact interval.'),
        TableSpec('secondary_endpoints', 'Secondary endpoints (no p-values)',
                  _secondary_rows(est.get('secondary') or {}),
                  ('metric', 'role', 'cell', 'n', 'transfer_mean',
                   'scratch_mean', 'hl_delta', 'ci_lo', 'ci_hi'),
                  'ANALYSIS_PLAN.md 1: estimation-only by pre-registration.'),
        TableSpec('mechanism_signals', 'Mechanism signals (DESIGN.md 5.5)',
                  _mechanism_rows(est.get('secondary') or {}),
                  ('metric', 'role', 'cell', 'n', 'transfer_mean',
                   'scratch_mean', 'hl_delta', 'ci_lo', 'ci_hi'),
                  'DESIGN.md 5.5, 9: a mechanism claim must cite one of these '
                  'and may not be made from prose.'),
        TableSpec('source_competence', 'Delta versus source competence',
                  (est.get('secondary') or {}).get('source_competence') or [],
                  ('cell', 'n', 'slope', 'ci_lo', 'ci_hi'),
                  'DESIGN.md 4.3: a descriptive relationship, not a mediation '
                  'claim.'),
        TableSpec('screens', 'Ablation screens (BH q, orientation only)',
                  (est.get('screens') or {}).get('rows') or [],
                  ('experiment', 'metric', 'cell', 'level', 'n', 'hl',
                   'ci_lo', 'ci_hi', 'q'),
                  'ANALYSIS_PLAN.md 3: a screen result is never asserted; it '
                  'selects at most one follow-up.'),
        TableSpec('power_per_member', 'Power and MDE per confirmatory member',
                  power.get('per_member') or [],
                  ('metric', 'cell', 'n', 'sd_delta', 'mde_nominal',
                   'mde_holm8', 'powered', 'note'),
                  'ANALYSIS_PLAN.md 6: which cells are powered was known before '
                  'launch, not discovered later.'),
        TableSpec('power_planning', 'Planned versus observed dispersion',
                  power.get('planning_comparison') or [],
                  ('arm', 'planned_sd', 'observed_sd', 'ratio'),
                  'ANALYSIS_PLAN.md 6.4: the power table is not re-tuned after '
                  'seeing results; the comparison is printed instead.'),
        TableSpec('multiplicity_ledger', 'Multiplicity ledger',
                  ledger.get('families') or [],
                  ('family', 'members', 'procedure', 'adjusted_alpha'),
                  'ANALYSIS_PLAN.md 7: printed on every invocation so the '
                  'count is a recorded fact rather than a claim.'),
    ]
    for metric, block in (desc or {}).items():
        specs.append(TableSpec(
            f'descriptives_{metric}', f'Descriptives per arm: {metric}',
            block.get('per_arm') or [],
            ('env', 'cell', 'condition', 'label', 'seed_block', 'n', 'mean',
             'sd', 'median', 'ci_lo', 'ci_hi', 'min', 'max'),
            'ANALYSIS_PLAN.md 10.4: mean, SD, median and a bootstrap interval, '
            'with the scratch mean, the threshold and the headroom beside it.'))
        specs.append(TableSpec(
            f'headroom_{metric}', f'Headroom per cell: {metric}',
            block.get('headroom') or [],
            ('cell', 'n', 'scratch_mean', 'scratch_sd', 'threshold',
             'headroom'),
            'DESIGN.md 2.5: a cell near the ceiling has less to gain and more '
            'to lose, so headroom travels with every between-cell comparison.'))
    return specs


def write_tables(bundle: str, specs: Sequence[TableSpec], prov_base: dict
                 ) -> list[dict]:
    """Write each table plus its provenance record; return the index."""
    index: list[dict] = []
    tdir = os.path.join(bundle, 'tables')
    for spec in specs:
        path = os.path.join(tdir, f'{spec.name}.csv')
        write_csv(path, spec.rows, spec.columns)
        # ANALYSIS_PLAN.md 7: which tables carry a p-value is bookkeeping, so
        # it is read off the columns. Stamping the flag from the table's *name*
        # recorded 0 for screens.csv, which has a BH q column, while the prose
        # in section 10 said the same thing and the table said otherwise. The
        # plan permits those q values; what it does not permit is a provenance
        # record that contradicts the file it describes.
        p_cols = [c for c in spec.columns if c in PVALUE_COLUMNS]
        record = dict(prov_base)
        record.update({
            'table': spec.name, 'title': spec.title,
            'columns': list(spec.columns), 'rows': len(spec.rows),
            'why': spec.why,
            'file': os.path.basename(path),
            'file_sha': provenance.file_hash(path),
            'p_value_columns': p_cols,
            'confirmatory_family': spec.name == 'confirmatory',
            'analyses_carrying_a_p_value':
                1 if spec.name == 'confirmatory' else 0,
            'orientation_only_q': bool(p_cols)
                                  and spec.name != 'confirmatory',
        })
        prov_path = os.path.join(tdir, f'{spec.name}.provenance.json')
        with open(prov_path, 'w', encoding='utf-8') as fh:
            json.dump(record, fh, indent=2, sort_keys=True, default=str)
        index.append({'table': spec.name, 'title': spec.title,
                      'rows': len(spec.rows),
                      'p_value_columns': ','.join(p_cols) or '--',
                      'csv': rel(path, bundle),
                      'provenance': rel(prov_path, bundle)})
    return index


# ===========================================================================
# 6. REPORT.md, in the order `ANALYSIS_PLAN.md` 10 prescribes.
# ===========================================================================
def section_stamps(out: list[str], ctx: dict) -> None:
    audit_report = ctx['audit_report']
    if ctx['override']:
        failed = [c['name'] for c in audit_report.get('checks', [])
                  if c.get('status') == 'FAIL']
        out += ['> ## ' + OVERRIDE_STAMP, '>']
        out += [f'> Failing checks: **{", ".join(failed) or "none named"}** '
                f'({audit_report.get("errors")} error(s), '
                f'{audit_report.get("warnings")} warning(s)).', '>']
        out += ['> Every table provenance record, the figure directory and '
                '`MANIFEST.json` carry the same stamp. Nothing below may be '
                'reported as a confirmatory result while these checks fail: '
                'the audit is a precondition for inference, not a summary of '
                'it (DESIGN.md 8.4).', '']
        for chk in audit_report.get('checks', []):
            if chk.get('status') != 'FAIL':
                continue
            out.append(f'- **{chk["name"]}** ({chk["errors"]} error(s)) -- '
                       f'{chk["why"]}')
            for finding in chk.get('findings', [])[:6]:
                if finding.get('level') != audit_mod.ERROR:
                    continue
                out.append(f'    - `{finding.get("code")}`: '
                           f'{finding.get("message")}')
        out.append('')
    if ctx['validation_stamp']:
        out += ['> ## ' + VALIDATION_STAMP, '>',
                '> An arm in this bundle has n below '
                f'{stats_mod.MIN_N_FOR_INFERENCE}. No number here may be '
                'quoted, compared, or used to choose between hypotheses; a '
                'single-seed run shows that a run executes, never that an arm '
                'differs (ANALYSIS_PLAN.md 9, STANDING_INSTRUCTIONS S8).', '']
    if ctx['plan_drift']:
        out += ['> ## PRE-REGISTRATION DRIFT', '>',
                '> The runs were produced under a different `ANALYSIS_PLAN.md` '
                'hash than the one in force now, so every confirmatory result '
                'below is **exploratory** until the audit says otherwise '
                '(ANALYSIS_PLAN.md 1).', '']


def section_provenance(out: list[str], ctx: dict) -> None:
    prov = ctx['provenance']
    git = prov.get('git') or {}
    sj = ctx['stats']
    s1 = sj.get('s1_provenance', {}) or {}
    out += ['## 1. Provenance', '']
    out += wrap('Recorded rather than promised. The Phase 0 audit could not '
                'reconstruct which configuration produced one of four '
                'published numbers because the machine that ran it was gone '
                '(DESIGN.md 8.3).')
    out.append('')
    rows = [
        {'field': 'git commit', 'value': git.get('commit')},
        {'field': 'git branch', 'value': git.get('branch')},
        {'field': 'working tree dirty', 'value': git.get('dirty')},
        {'field': 'uncommitted files', 'value': git.get('dirty_files')},
        {'field': 'ANALYSIS_PLAN.md hash (now)',
         'value': (prov.get('plans') or {}).get('ANALYSIS_PLAN.md')},
        {'field': 'DESIGN.md hash (now)',
         'value': (prov.get('plans') or {}).get('DESIGN.md')},
        {'field': 'reference_returns.json hash (now)',
         'value': (prov.get('plans') or {}).get('reference_returns.json')},
        # The key names are stats.py's own. They were 'plan_md5_in_data' and
        # 'table_md5' here, which no version of that module ever wrote, so both
        # rows printed "--" in every bundle: a provenance table that recorded
        # nothing while stating that provenance is recorded rather than
        # promised (DESIGN.md 8.3).
        {'field': 'ANALYSIS_PLAN.md hash in the run data',
         'value': ','.join(str(h) for h in s1.get('plan_hash_in_data') or [])},
        {'field': 'ANALYSIS_PLAN.md hash seen by stats.py',
         'value': s1.get('plan_hash_current')},
        {'field': 'DESIGN.md hash seen by stats.py',
         'value': s1.get('design_hash')},
        {'field': 'per_seed.csv digest seen by stats.py',
         'value': s1.get('table_digest')},
        {'field': 'per_seed.csv hash (bundle copy)',
         'value': ctx['hashes'].get('per_seed')},
        {'field': 'curves.csv hash (bundle copy)',
         'value': ctx['hashes'].get('curves')},
        {'field': 'stats.json hash', 'value': ctx['hashes'].get('stats')},
        {'field': 'bootstrap', 'value': f'{s1.get("n_boot")} resamples, '
                                        f'seed {s1.get("boot_seed")}'},
        {'field': 'report.py argv', 'value': ' '.join(prov.get('argv') or [])},
        {'field': 'cwd', 'value': prov.get('cwd')},
        {'field': 'generated', 'value': ctx['timestamp']},
    ]
    out += md_table(rows, ('field', 'value'))
    if git.get('dirty'):
        out += wrap(f'{WARN} the working tree was dirty '
                    f'({git.get("dirty_files")} uncommitted file(s)). A result '
                    'produced from an uncommitted tree is not reproducible '
                    'from the repository, and that has to be visible in the '
                    'artifact rather than discovered later.')
        out.append('')
    out += ['### Package versions', '']
    out += md_table([{'package': k, 'version': v} for k, v
                     in sorted((prov.get('packages') or {}).items())],
                    ('package', 'version'))
    out += ['### Machine and determinism', '']
    out += md_table([{'field': k, 'value': v} for k, v
                     in sorted((prov.get('machine') or {}).items())]
                    + [{'field': k, 'value': v} for k, v
                       in sorted((prov.get('determinism') or {}).items())],
                    ('field', 'value'))
    out += ['### Pipeline stages', '']
    out += md_table([{'stage': s.name, 'exit': s.exit_code,
                      # Pre-formatted: f_ would render a 60.0 second stage as
                      # 60.0000, which reads as four-decimal precision on a
                      # wall-clock duration.
                      'seconds': f'{s.seconds:.1f}',
                      'log': (rel(s.log, ctx['bundle']) if s.log else '--'),
                      'command': ' '.join(s.command[1:])}
                     for s in ctx['stages']],
                    ('stage', 'exit', 'seconds', 'log', 'command'))


#: Files that make a directory a run directory. `manifest.json` is what
#: `audit.discover_runs` globs for, and `metrics.jsonl` is what a run writes
#: from its first episode, so a directory holding the second and not the first
#: is a run the audit cannot see. A checkpoint directory carries neither.
RUN_DIR_MARKERS = ('manifest.json', 'metrics.jsonl')


def run_dirs_on_disk(out_root: str) -> tuple[int, list[str]]:
    """Count run directories on disk, and name those with no manifest.

    `DESIGN.md` 9 forbids silent seed dropping, and section 2 used to assert
    that "an absence below is an absence in the run tree" over a counter it
    labelled "run directories discovered". That counter is `audit.py`'s, and
    `audit.discover_runs` globs for `manifest.json`: deleting only the manifest
    from a run whose metrics, checkpoints and weights are all still present
    made it vanish from every count in the bundle, with `runs_unattributed`
    still reading 0. Manifests parsed is not run directories discovered, so
    both numbers are printed and a difference is named.
    """
    root = os.path.abspath(out_root)
    total = 0
    unmanifested: list[str] = []
    for entry in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        if entry.startswith('_') or entry.startswith('.'):
            continue                  # _jobs, _index, _logs: bookkeeping
        top = os.path.join(root, entry)
        if not os.path.isdir(top):
            continue
        for current, dirs, files in os.walk(top):
            names = set(files)
            if not names.intersection(RUN_DIR_MARKERS):
                continue
            total += 1
            if 'manifest.json' not in names:
                unmanifested.append(rel(current, root))
            dirs[:] = []               # a run directory holds no run directory
    return total, unmanifested


def section_inventory(out: list[str], ctx: dict, log: ClaimLog) -> None:
    sj = ctx['stats']
    inv = sj.get('s2_inventory', {}) or {}
    audit_report = ctx['audit_report']
    out += ['## 2. Run inventory', '']
    disk_runs, unmanifested = run_dirs_on_disk(ctx['out_root'])
    out += wrap('Which runs exist, at which seeds, in which block, with which '
                'source-validity verdict and which transferred-parameter '
                'fraction. A seed is never dropped silently (DESIGN.md 9), so '
                'an absence below is an absence in the run tree *provided the '
                'two counters agree*: everything downstream of the audit sees '
                'a run through its `manifest.json`, so the directory count on '
                'disk is taken independently and printed beside the manifest '
                'count. Where they differ, the unmanifested directories are '
                'named below and they are in none of the tables.')
    out.append('')
    summary = [
        {'field': 'run tree', 'value': audit_report.get('out_root')},
        {'field': 'run directories on disk (manifest.json or metrics.jsonl)',
         'value': disk_runs},
        {'field': 'run manifests parsed by audit.py',
         'value': audit_report.get('runs_discovered')},
        {'field': 'runs in audit scope',
         'value': audit_report.get('runs_in_scope')},
        {'field': 'runs unattributed to any experiment',
         'value': audit_report.get('runs_unattributed')},
        {'field': 'experiments selected',
         'value': ', '.join(audit_report.get('experiments_selected') or [])},
        {'field': 'rows in per_seed.csv', 'value': ctx['per_seed_rows']},
        {'field': 'rows in curves.csv', 'value': ctx['curves_rows']},
        {'field': 'rows entering the analysis (TUNE excluded)',
         'value': (sj.get('s1_provenance') or {}).get('rows')},
        {'field': 'TUNE runs excluded by stats.py',
         'value': (sj.get('invocation') or {}).get('tune_runs_excluded')},
        {'field': 'source policy in force',
         'value': (sj.get('invocation') or {}).get('source_policy')},
    ]
    out += md_table(summary, ('field', 'value'))
    if unmanifested:
        out += wrap(f'{WARN} {len(unmanifested)} run director(ies) hold run '
                    f'output but no `manifest.json`, so `audit.py` never saw '
                    f'them, they are counted in no table above, and they are '
                    f'in no analysis in this bundle. This is a dropped run, '
                    f'not an absent one (DESIGN.md 9): '
                    + ', '.join(f'`{d}`' for d in unmanifested[:12])
                    + (' ...' if len(unmanifested) > 12 else '') + '.')
        out.append('')

    out += ['### 2a. Seed blocks', '']
    out += md_table(inv.get('seed_blocks') or [],
                    ('seed_block', 'runs', 'seeds'))
    out += wrap('The blocks are disjoint by construction: `TUNE` (200-204) '
                'selects hyperparameters and may never enter a reported '
                'estimate, `CONFIRM` (0-9) carries every confirmatory arm, '
                '`C4SRC` (300-309) supplies the positive control sources and '
                '`RESERVE` (400+) replaces a source the validity gate rejects '
                '(DESIGN.md 3.4).')
    out.append('')

    out += ['### 2b. Arms', '']
    out += md_table(inv.get('arms') or [],
                    ('env', 'cell', 'condition', 'label', 'n', 'seeds',
                     'seed_block', 'transfer_set', 'freeze_updates'))
    out += ['### 2c. Pairing completeness', '']
    out += md_table(inv.get('completeness') or [],
                    ('cell', 'n_scratch', 'n_transfer', 'paired_seeds',
                     'scratch_only', 'transfer_only', 'transfer_labels'))
    out += ['### 2d. Source validity (DESIGN.md 4.3, normalised gate 0.6)', '']
    out += md_table(inv.get('source_validity') or [],
                    ('cell', 'condition', 'label', 'n', 'valid', 'invalid',
                     'invalid_seeds', 'source_score_mean'))
    out += ['### 2e. Transferred-parameter fraction and the intensity gate', '']
    out += md_table([{'cell': k, 'transferred_param_fraction': v} for k, v
                     in sorted((inv.get('transferred_fraction') or {}).items())],
                    ('cell', 'transferred_param_fraction'))
    out += md_table(inv.get('intensity_gate') or [],
                    ('a', 'b', 'frac_a', 'frac_b', 'abs_diff', 'verdict',
                     'permitted'))
    out += wrap('DESIGN.md 3.1: revision 1 transferred 97% of the mlp network '
                'and 51% of the dueling network and called that the same '
                'protocol, so the arch factor was confounded with treatment '
                'intensity by a factor of about two. A cross-arch contrast '
                'beyond the tolerance is refused unless it is labelled '
                'intensity-confounded.')
    out.append('')

    out += ['### 2f. Normalisation (DESIGN.md 5.1)', '']
    refs = []
    for env_id in ctx['envs_seen']:
        try:
            ref = envs.reference(env_id)
        except (KeyError, ValueError) as exc:
            refs.append({'env': env_id, 'random_return': f'MISSING: {exc}',
                         'threshold': '--', 'noop_return': '--'})
            continue
        refs.append({'env': env_id, 'random_return': ref.get('random_return'),
                     'threshold': ref.get('threshold'),
                     'noop_return': ref.get('noop_return')})
    out += md_table(refs, ('env', 'random_return', 'threshold', 'noop_return'))
    out += wrap('score = (return - random_return) / (threshold - '
                'random_return), so a random policy scores 0 and the '
                'registered threshold scores 1 by construction. Across the '
                'LunarLander gravity family the random return moves from -202 '
                'to -463, so a raw delta would silently mix a scale change '
                'into a shift effect.')
    out.append('')

    # A descriptive claim about the inventory, subject to the same guardrails.
    comp = inv.get('completeness') or []
    # _int_or_none throughout: a NaN reaching int() raises, and stats.py's
    # JSON round-trips a bare NaN. A count that cannot be read is 0 here, and
    # the guard in claim() then sees the smaller number rather than crashing.
    paired = sum(_int_or_none(r.get('paired_seeds')) or 0 for r in comp)
    # The n behind this sentence is the *weakest cell*, not the sum. Summing
    # across cells passed the n<3 guard on a one-seed tree (4 pairs, one per
    # cell) and printed the sentence inside a bundle stamped PIPELINE
    # VALIDATION - NOT A RESULT. The docstring defines n as the units behind
    # the number, and nothing in this study is estimated across cells: the unit
    # is a matched pair within a cell, so the guard has to see the smallest
    # per-cell count.
    per_cell = [_int_or_none(r.get('paired_seeds')) or 0 for r in comp]
    weakest = min(per_cell) if per_cell else 0
    log.section = 'Section 2 (inventory)'
    c = claim(
        f'Across the {len(per_cell) or "four"} cells the run tree supplies '
        f'{paired} matched scratch/transfer seed pair(s), the weakest cell '
        f'carrying {weakest}; the per-cell counts and the identity of every '
        f'unpaired seed are printed above rather than summarised.',
        'descriptive',
        {'estimand': 'arm_descriptive', 'n': max(weakest, 0),
         'counterfactual': 'none: this is an inventory count, not a contrast',
         'rivals': ['a run present on disk but attributed to the wrong arm '
                    'by label rather than by configuration digest',
                    'a run present on disk with no readable manifest, which '
                    'every counter downstream of the audit would miss'],
         'excluded_by': ('audit.py RUN ATTRIBUTION and RUN-DIRECTORY '
                         'UNIQUENESS for the first; the independent directory '
                         'count in the table above for the second'),
         'refuted_by': 'a declared arm x seed with no run directory, which '
                       'aggregate.py --require-complete would list, or a run '
                       'directory with no manifest, which the two counters '
                       'above disagree about'})
    out += render_claim_md(log.add(c))


def signflip_bar_lines(members: Sequence[dict], alpha: float) -> list[str]:
    """The confirmatory bar, stated at the n this bundle actually has.

    This is the one place the module does arithmetic, and it is arithmetic on
    the sample size rather than on an endpoint: the exact two-sided sign-flip
    test on n paired deltas has 2^n equally likely sign assignments, so the
    smallest attainable p is 2/2^n. It was a hand-written constant for n=10,
    which printed "the smallest attainable two-sided sign-flip p is 2/1024 =
    0.00195 ... so a cell is confirmed if and only if every seed moves the same
    way" above a table whose every row read n=3. At n=3 that p is 0.25, no cell
    can be confirmed at any Holm step, and the sentence told the reader the
    opposite of the truth about its own table.
    """
    ns = sorted({v for v in (_int_or_none(m.get('n')) for m in members)
                 if v is not None and v > 0})
    if not ns:
        return [f'No confirmatory member carries an n, so the attainable '
                f'sign-flip bar is not stated. The strictest Holm step is '
                f'alpha = {alpha:g} (ANALYSIS_PLAN.md 2.2).']
    lines: list[str] = []
    for n in ns:
        p_min = min(2.0 / float(2 ** n), 1.0)
        if p_min <= alpha:
            verdict = ('so a cell at this n is confirmed if and only if every '
                       'seed moves the same way')
        else:
            verdict = (f'so **no cell at n={n} can be confirmed at all**, '
                       f'however large its effect: the smallest p the test can '
                       f'return is above the step it would have to clear, so '
                       f'the family is reported for its intervals and the '
                       f'confirmatory verdict is unavailable by construction')
        lines.append(
            f'At n={n} the smallest attainable two-sided sign-flip p is '
            f'2/2^{n} = {p_min:.5f} and the strictest Holm step is {alpha:g}, '
            f'{verdict}. That bar was stated before the numbers were seen '
            f'(ANALYSIS_PLAN.md 2.2); the value above is arithmetic on n and '
            f'on nothing else.')
    return lines


def section_confirmatory(out: list[str], ctx: dict, log: ClaimLog) -> None:
    sj = ctx['stats']
    conf = sj.get('s5_confirmatory', {}) or {}
    members = conf.get('members') or []
    out += ['## 3. The confirmatory family', '']
    out += wrap(f'Exactly {conf.get("family_size", 8)} tests: the within-cell '
                f'paired delta (transfer - scratch) for each of the four cells '
                f'on each of the two co-primary endpoints. Holm-Bonferroni '
                f'over the family, strictest step alpha = '
                f'{conf.get("alpha_strictest")}. Primary test: exact sign-flip '
                f'randomisation on the per-seed deltas, statistic the mean. '
                f'Wilcoxon signed-rank and Mann-Whitney U are reported for the '
                f'same contrast by pre-registration, not by selection after '
                f'the fact (ANALYSIS_PLAN.md 2).')
    out.append('')
    alpha = conf.get('alpha_strictest')
    alpha = float(alpha) if _finite(alpha) else float(stats_mod.ALPHA_STRICTEST)
    for line in signflip_bar_lines(members, alpha):
        out += wrap(line)
        out.append('')
    out += md_table(_blank_suppressed(members),
                    ('metric', 'cell', 'n', 'scratch_mean', 'transfer_mean',
                     'mean_delta', 'hl', 'ci_lo', 'ci_hi', 'p_signflip',
                     'p_holm', 'significant_holm', 'p_wilcoxon',
                     'mannwhitney_U', 'p_mannwhitney', 'rho_pearson',
                     'suppressed'))
    suppressed = [m for m in members if m.get('suppressed')]
    if suppressed:
        out += [f'{len(suppressed)} of {len(members)} members suppressed. A '
                f'suppression is a refusal with a reason, never a null:', '']
        for m in suppressed:
            out.append(f'- `{m["metric"]}/{m["cell"]}`: {m["suppressed"]}')
        out.append('')

    log.section = 'Section 3 (confirmatory)'
    for m in members:
        cell, metric = m.get('cell'), m.get('metric')
        rho = m.get('rho_pearson')
        pairing = ''
        if _finite(rho):
            pairing = (f' The within-seed correlation is rho={float(rho):+.3f}'
                       + ('; it is negative, so the matched-seed pairing does '
                          'not hold in this cell and the unpaired '
                          'Mann-Whitney result is given equal prominence '
                          '(ANALYSIS_PLAN.md 2.1).'
                          if float(rho) < 0 else
                          ', reported whatever its value; the paired test '
                          'stays primary by pre-registration.'))
        if m.get('suppressed'):
            body = (f'In cell {cell} the transfer protocol shifted {metric} '
                    f'relative to the same cell own scratch baseline at '
                    f'matched seeds.')
        else:
            # _f, not float: stats.py emits mean_delta for a member it
            # suppressed and blanks the interval, so the shapes are not tied
            # to each other and a null here used to raise TypeError out of
            # np.isfinite after every stage had already run.
            direction = stats_mod.phrase_direction(
                _f(m.get('mean_delta')), 'the transfer arm',
                "its own cell's scratch baseline")
            verdict = stats_mod.phrase_interval_verdict(
                _f(m.get('ci_lo')), _f(m.get('ci_hi')),
                'the paired shift (Hodges-Lehmann)')
            body = (f'In cell {cell} on {metric}, {direction}; {verdict}. '
                    f'{m.get("unanimous")}.{pairing}')
        evidence = {
            'estimand': 'within_cell_delta', 'n': m.get('n'),
            'seeds': m.get('seeds'), 'ci_lo': m.get('ci_lo'),
            'ci_hi': m.get('ci_hi'), 'confirmatory': True,
            'p_holm': m.get('p_holm'), 'scale': 'normalised_score',
            'counterfactual': ("the same cell's own scratch run at the same "
                               'seed, sharing per-layer initialisation for '
                               'every non-transferred layer, the '
                               'environment-reset sequence and the evaluation '
                               'seed streams (DESIGN.md 8.1)'),
            'rivals': ['protocol mechanics with no learned content (C2)',
                       'the transferred weights marginal distribution alone '
                       '(C3)',
                       'plasticity loss after pretraining: dead units, '
                       'parameter-norm growth, feature-rank collapse'],
            'excluded_by': ('the C2 and C3 contrasts in section 4; the '
                            'plasticity account is instrumented (DESIGN.md '
                            '5.5) but experimentally excluded only if E13 ran'),
            'refuted_by': ('a paired interval covering zero, or seeds split in '
                           'direction, in which case the licensed statement is '
                           'the exclusion bound and not an effect'),
        }
        if m.get('suppressed'):
            evidence['suppressed'] = m['suppressed']
        out += render_claim_md(log.add(claim(body, 'causal', evidence)))


def section_equivalence(out: list[str], ctx: dict, log: ClaimLog) -> None:
    sj = ctx['stats']
    eq = sj.get('s6_equivalence', {}) or {}
    rows = eq.get('rows') or []
    margin = eq.get('margin', stats_mod.EQUIVALENCE_MARGIN)
    out += ['## 4. Equivalence and exclusion', '']
    out += wrap(f'Not TOST. The margin is +/-{margin} normalised-score units, '
                f'fixed in ANALYSIS_PLAN.md 4 before any data existed, and '
                f'equivalence is assessed by whether the 95% bootstrap CI on '
                f'the paired delta lies entirely inside it. Where a cell '
                f'across-seed SD exceeds the margin, EQUIVALENCE is untestable '
                f'at this n and the report says so instead of returning a '
                f'verdict; a difference, meaning an interval lying wholly '
                f'outside the margin, is not governed by that gate and is '
                f'reported whatever the dispersion. The exclusion bound is '
                f'reported for every cell whose interval has width, because it '
                f'is the only powered directional statement available; a cell '
                f'whose interval is a point has none.')
    out.append('')
    out += md_table(rows, ('metric', 'cell', 'n', 'ci_lo', 'ci_hi',
                           'sd_scratch', 'sd_transfer', 'margin', 'verdict',
                           'exclusion_bound'))
    log.section = 'Section 4 (equivalence / exclusion)'
    for r in rows:
        verdict = r.get('verdict')
        # The margin and the two SDs travel with the verdict so that claim()
        # can re-derive it by containment instead of trusting the string
        # (ANALYSIS_PLAN.md 4).
        base = {'estimand': 'within_cell_delta', 'n': r.get('n'),
                'ci_lo': r.get('ci_lo'), 'ci_hi': r.get('ci_hi'),
                'verdict': verdict, 'basis': 'bootstrap_ci',
                'margin': r.get('margin', margin),
                'sd_a': r.get('sd_transfer'), 'sd_b': r.get('sd_scratch'),
                'scale': 'normalised_score',
                'counterfactual': ("the same cell's own scratch run at the "
                                   'same seed'),
                'rivals': ['a dispersion so wide that any margin is contained '
                           'by accident of width'],
                'excluded_by': ('the feasibility gate: a cell whose across-seed '
                                'SD exceeds the margin is declared untestable '
                                'rather than given a verdict'),
                'refuted_by': ('an interval that leaves the margin, or a cell '
                               'SD above the margin, either of which withdraws '
                               'the verdict')}
        if verdict == 'EQUIVALENT':
            text = (f'In cell {r.get("cell")} on {r.get("metric")} the paired '
                    f'delta is equivalent to zero within the pre-registered '
                    f'margin: {r.get("reason")}.')
        else:
            text = (f'In cell {r.get("cell")} on {r.get("metric")} the '
                    f'equivalence verdict is {verdict}: {r.get("reason")}.')
        out += render_claim_md(log.add(claim(text, 'equivalence', base)))
        excl = dict(base)
        excl['refuted_by'] = ('a wider interval at a larger n moving the lower '
                              'bound down')
        # stats.py's own answer, forwarded only when the row carries it: it is
        # None exactly where stats.py declined to publish a bound (a zero-width
        # interval, or none at all), and re-deriving the sentence from ci_lo
        # published "every degradation is excluded at 95%" over a cell whose
        # own reason field read "no equivalence or exclusion claim is available
        # at all". A row from an older stats.json has no such key, and its
        # absence is not a withheld bound.
        if 'exclusion_bound' in r:
            excl['exclusion_bound'] = r.get('exclusion_bound')
        out += render_claim_md(log.add(claim(
            f'For cell {r.get("cell")} on {r.get("metric")}',
            'exclusion', excl)))


def section_controls(out: list[str], ctx: dict, log: ClaimLog) -> None:
    sj = ctx['stats']
    controls = sj.get('s7_controls', {}) or {}
    out += ['## 5. Control contrasts', '']
    out += wrap('The identity (C2-C0) + (C3-C2) + (C1-C3) = C1-C0 is a '
                'telescoping arithmetic identity. It holds for any four '
                'numbers, is shown only to fix notation, and is not evidence '
                'of additivity: revision 1 called it an additive decomposition '
                'and implied an empirical finding where there is none '
                '(DESIGN.md 4.1). Estimation is joint -- the per-seed vector '
                '(C0, C1, C2, C3, C3b) is bootstrapped once, so the contrasts '
                'correlations are estimated rather than ignored.')
    out.append('')
    rows = _flatten_controls(controls)
    out += md_table(rows, ('metric', 'cell', 'contrast', 'name', 'n', 'hl',
                           'ci_lo', 'ci_hi', 'unanimous'))
    log.section = 'Section 5 (control contrasts)'
    for metric, block in controls.items():
        for cell, payload in (block.get('cells') or {}).items():
            missing = payload.get('missing') or []
            if missing:
                out += wrap(f'`{metric}/{cell}`: conditions absent from the run '
                            f'tree: {", ".join(missing)}. A contrast that needs '
                            f'them is not estimated and nothing is substituted '
                            f'for it.')
                out.append('')
            for contrast in payload.get('contrasts') or []:
                key = str(contrast.get('contrast'))
                restriction = stats_mod.CONTROL_EXCLUSION_RESTRICTIONS.get(key)
                verdict = stats_mod.phrase_interval_verdict(
                    _f(contrast.get('ci_lo')), _f(contrast.get('ci_hi')),
                    f'the {contrast.get("name")} ({key})')
                text = (f'In cell {cell} on {metric}, {verdict}. '
                        f'{contrast.get("unanimous")}.')
                out += render_claim_md(log.add(claim(text, 'causal_component', {
                    'estimand': 'control_contrast', 'n': contrast.get('n'),
                    'ci_lo': contrast.get('ci_lo'),
                    'ci_hi': contrast.get('ci_hi'),
                    'manipulated': key, 'scale': 'normalised_score',
                    'counterfactual': f'the {key.split("-")[1]} arm at the same '
                                      f'seed, differing only in the '
                                      f'manipulated component',
                    'rivals': ['the spectral change the shuffle also makes: '
                               'C3 preserves the weight multiset and the '
                               'Frobenius norm but not the row/column norms '
                               'or the singular-value spectrum'],
                    'excluded_by': ('the spectrum-matched control C3b, whose '
                                    'agreement or disagreement with C3 bounds '
                                    'the caveat'),
                    'refuted_by': (f'an interval covering zero, or a '
                                   f'disagreement between C3 and C3b, which '
                                   f'would relocate the contrast to a spectral '
                                   f'effect. The mechanistic reading further '
                                   f'assumes {restriction}'
                                   if restriction else
                                   'an interval covering zero'),
                })))
            # H2 is a magnitude comparison between two contrasts, so its
            # wording is generated from the two numbers rather than asserted.
            by_key = {str(c.get('contrast')): c
                      for c in payload.get('contrasts') or []}
            a, b = by_key.get('C1-C3'), by_key.get('C2-C0')
            if a and b:
                text = stats_mod.phrase_magnitude_comparison(
                    'the trained-vs-permuted contrast (C1-C3)',
                    _f(a.get('hl')),
                    'the untrained-source contrast (C2-C0)', _f(b.get('hl')))
                out += render_claim_md(log.add(claim(
                    f'{cell}, {metric}: {text} (H2 in DESIGN.md 2.3 is refuted '
                    f'when the trained-vs-permuted contrast is the larger of '
                    f'the two in 2 or more cells).',
                    'descriptive',
                    {'estimand': 'arm_descriptive',
                     'n': min((v for v in (_int_or_none(a.get('n')),
                                           _int_or_none(b.get('n')))
                               if v is not None), default=None),
                     'counterfactual': 'the two contrasts share the C1 and C0 '
                                       'arms, so they are compared within one '
                                       'joint bootstrap',
                     'rivals': ['both contrasts are estimated at the same n, '
                                'so a magnitude ordering can invert on '
                                'resampling'],
                     'excluded_by': 'nothing at this n; the ordering is '
                                    'reported as a description, not a test',
                     'refuted_by': 'the ordering inverting in the joint '
                                   'bootstrap, which the intervals above show'})))


def section_c4(out: list[str], ctx: dict, log: ClaimLog) -> None:
    sj = ctx['stats']
    c4 = sj.get('s8_c4', {}) or {}
    out += ['## 6. C4 positive control', '']
    out += wrap(f'The interface-change-only pair: `{c4.get("env")}`, where the '
                f'dynamics are identical by construction while the partial '
                f'copy, the head reinitialisation and the freeze window all '
                f'run exactly as in E1. Sources come from the disjoint C4SRC '
                f'block, so these deltas are independent of E1. Pre-registered '
                f'pass criterion: the Hodges-Lehmann estimate 95% CI lower '
                f'bound exceeds {stats_mod.C4_LOWER_BOUND} in normalised-score '
                f'units (DESIGN.md 4.2).')
    out.append('')
    if not c4.get('available'):
        out += ['*The interface-change-only pair is absent from this run tree, '
                'so C4 is not evaluated.*', '']
        return
    rows = c4.get('rows') or []
    out += md_table(rows, ('cell', 'n', 'n_transfer', 'n_scratch', 'hl',
                           'ci_lo', 'ci_hi', 'verdict', 'reason'))
    log.section = 'Section 6 (C4 positive control)'
    for r in rows:
        text = (f'C4 in cell {r.get("cell")}: verdict {r.get("verdict")} '
                f'against the pre-registered criterion -- {r.get("reason")}')
        out += render_claim_md(log.add(claim(text, 'exclusion', {
            'estimand': 'within_cell_delta', 'n': r.get('n'),
            'ci_lo': r.get('ci_lo'), 'ci_hi': r.get('ci_hi'),
            'scale': 'normalised_score',
            'counterfactual': 'the same cell scratch run on the same '
                              'interface-changed environment at the same seed',
            'rivals': ['the padded observation and duplicated actions '
                       'themselves making the task harder, independently of '
                       'transfer'],
            'excluded_by': 'the scratch arm on the same padded/extended '
                           'environment, which absorbs that',
            'refuted_by': 'a lower bound at or below '
                          f'{stats_mod.C4_LOWER_BOUND}, which would mean the '
                          'protocol degrades performance at zero dynamics '
                          'shift and that "negative transfer" is the wrong '
                          'name for the finding'})))
    failed = [r for r in rows if r.get('verdict') == 'FAIL']
    if failed:
        out += wrap(f'{WARN} C4 failed in {len(failed)} cell(s). DESIGN.md 4.2 '
                    f'states in advance what that means: the protocol degrades '
                    f'performance even with no dynamics shift at all, which '
                    f'does not invalidate the study but does make "negative '
                    f'transfer" the wrong name for the finding, and the paper '
                    f'must say so.')
        out.append('')


def section_estimation(out: list[str], ctx: dict, log: ClaimLog) -> None:
    sj = ctx['stats']
    est = sj.get('s9_estimation', {}) or {}
    desc = sj.get('s3_descriptives', {}) or {}
    conv = sj.get('s4_convergence', {}) or {}
    out += ['## 7. Estimation-only sections', '']
    out += wrap('Every analysis in this section carries a point estimate and a '
                'seed-level bootstrap interval and no p-value at all. That is '
                'a design decision forced by the sample size and declared '
                'before launch, not an omission (ANALYSIS_PLAN.md 3).')
    out.append('')

    out += ['### 7a. Descriptives and headroom', '']
    for metric, block in desc.items():
        out += [f'**{metric}**', '']
        out += md_table(block.get('per_arm') or [],
                        ('env', 'cell', 'condition', 'label', 'n', 'mean',
                         'sd', 'median', 'ci_lo', 'ci_hi'))
        out += md_table(block.get('headroom') or [],
                        ('cell', 'n', 'scratch_mean', 'scratch_sd',
                         'threshold', 'headroom'))

    out += ['### 7b. Convergence gate', '']
    out += md_table(conv.get('per_arm') or [],
                    ('cell', 'condition', 'label', 'n', 'median_slope',
                     'ci_lo', 'ci_hi', 'frac_positive', 'still_moving'))
    failing = conv.get('failing') or []
    if failing:
        out += wrap(f'{len(failing)} arm(s) are still moving at the end of the '
                    f'budget: {", ".join(str(f) for f in failing[:12])}'
                    + (' ...' if len(failing) > 12 else '')
                    + '. For those arms P1 is performance at budget, not '
                      'asymptotic performance (DESIGN.md 5.2).')
        out.append('')

    log.section = 'Section 7c (RQ1)'
    out += ['### 7c. RQ1 -- between-cell scratch comparison (associational)',
            '']
    out += wrap('Cells are different algorithms, not treatments assigned to '
                'units, so this is an association. It is also a sanity check '
                'rather than a novel result: the scratch ordering across three '
                'of these four cells has been published at n=100 on these '
                'environments, and our direction must agree with it '
                '(DESIGN.md 7).')
    out.append('')
    rq1 = (est.get('rq1') or {}).get('rows') or []
    out += md_table(rq1, ('a', 'b', 'n_a', 'n_b', 'mean_a', 'mean_b', 'sd_a',
                          'sd_b', 'theta', 'ci_lo', 'ci_hi', 'note'))
    for r in rq1:
        theta = r.get('theta')
        if not _finite(theta):
            continue
        # No sentence is proposed, exactly as for dispersion: the verb is a
        # function of theta and claim() generates it. The template used to
        # write "exceeds" here as a literal, which shipped over theta=0.469 and
        # theta=0.333 beside mean_a < mean_b: the published paper's own error
        # (a direction asserted against the numbers printed in the same
        # sentence) reappearing in the module written to prevent it. A
        # hand-written verb is still refused if a caller supplies one, which is
        # what --self-test asserts.
        text = ''
        # n is the number of *seeds*, not the number of rows pooled into the
        # arm. On a three-seed tree n_a reads 9 because three labels at three
        # seeds are pooled, and 9 would pass a guard that 3 is the honest
        # input to. Where the seed set cannot be recovered the claim is
        # refused rather than estimated at the row count.
        n_a = target_scratch_seed_count(sj, str(r.get('a')))
        n_b = target_scratch_seed_count(sj, str(r.get('b')))
        out += render_claim_md(log.add(claim(text, 'associational', {
            'estimand': 'between_cell_scratch',
            'n': (min(n_a, n_b) if (n_a is not None and n_b is not None)
                  else None),
            'ci_lo': r.get('ci_lo'), 'ci_hi': r.get('ci_hi'),
            'theta': theta, 'name_a': r.get('a'), 'name_b': r.get('b'),
            'prefix': 'On the target task from scratch,',
            'rows_pooled_a': r.get('n_a'), 'rows_pooled_b': r.get('n_b'),
            'scale': 'normalised_score',
            'counterfactual': 'none within a cell: the comparison is between '
                              'two algorithms on the same task, so no '
                              'counterfactual is assigned to a unit',
            'rivals': ['target-task suitability rather than anything about '
                       'transfer, which is exactly why the primary estimand '
                       'is the within-cell delta'],
            'excluded_by': 'nothing here; RQ2 in section 3 is the separation',
            'refuted_by': 'an ordering opposite to the published n=100 '
                          'comparison, which would be a finding about this '
                          'hyperparameter regime and would be reported as one'
        })))

    log.section = 'Section 7d (RQ3)'
    out += ['### 7d. RQ3 -- between-cell delta contrasts (effect modification)',
            '']
    out += wrap('How the within-cell effect varies across cells. Not '
                '"architecture causes the difference": the MDE is about 2.7 '
                'sigma for the interaction, larger than any plausible effect, '
                'so this is estimation-only by design. Both the normalised and '
                'the headroom-adjusted scales are reported, and agreement '
                'across them is required before any wording is used '
                '(DESIGN.md 2.5).')
    out.append('')
    rq3 = est.get('rq3') or {}
    out += md_table(rq3.get('pairs') or [],
                    ('a', 'b', 'n', 'hl', 'ci_lo', 'ci_hi', 'hl_headroom_adj',
                     'adj_lo', 'adj_hi', 'scales', 'note'))
    for r in rq3.get('pairs') or []:
        if not _finite(r.get('hl')):
            continue
        agree = str(r.get('scales'))
        text = (f'The within-cell delta in {r.get("a")} differs from the one in '
                f'{r.get("b")} by {f_(r.get("hl"))} '
                f'[{f_(r.get("ci_lo"))}, {f_(r.get("ci_hi"))}]; on the '
                f'headroom-adjusted scale {f_(r.get("hl_headroom_adj"))} '
                f'[{f_(r.get("adj_lo"))}, {f_(r.get("adj_hi"))}]. The two '
                f'scales {agree}.')
        out += render_claim_md(log.add(claim(text, 'effect_modification', {
            'estimand': 'between_cell_delta', 'n': r.get('n'),
            'ci_lo': r.get('ci_lo'), 'ci_hi': r.get('ci_hi'),
            'scale': 'normalised_score',
            'between_cell_contrast': {'ci_lo': r.get('ci_lo'),
                                      'ci_hi': r.get('ci_hi')},
            'counterfactual': 'each cell own scratch baseline; the contrast is '
                              'a difference of two within-cell deltas, not a '
                              'difference of two verdicts',
            'rivals': ['ceiling proximity: a cell whose scratch baseline is '
                       'near the ceiling has less headroom to gain and more to '
                       'lose', 'treatment intensity: the transferred-parameter '
                       'fraction differing across architectures'],
            'excluded_by': 'the headroom-adjusted scale in the same row, and '
                           'the intensity gate in section 2e',
            'refuted_by': 'the two scales disagreeing, or an interval covering '
                          'zero, either of which withdraws the wording'})))
    inter = rq3.get('interaction') or {}
    if inter.get('available'):
        out += ['**2x2 interaction**', '']
        out += md_table([inter], ('n', 'hl', 'ci_lo', 'ci_hi',
                                 'intensity_confounded'))
        out += render_claim_md(log.add(claim(
            f'The 2x2 interaction contrast on the within-cell deltas is '
            f'{f_(inter.get("hl"))} [{f_(inter.get("ci_lo"))}, '
            f'{f_(inter.get("ci_hi"))}].',
            'effect_modification',
            {'estimand': 'interaction_2x2', 'n': inter.get('n'),
             'ci_lo': inter.get('ci_lo'), 'ci_hi': inter.get('ci_hi'),
             'between_cell_contrast': {'ci_lo': inter.get('ci_lo'),
                                       'ci_hi': inter.get('ci_hi')},
             'counterfactual': 'the four cells own scratch baselines at matched '
                               'seeds',
             'rivals': ['the interaction being an intensity artifact if the '
                        'transferred fractions are not matched'],
             'excluded_by': 'the matched transfer set of DESIGN.md 3.1 and the '
                            'intensity gate',
             'refuted_by': 'an interval covering zero, which at this MDE is '
                           'the expected outcome and is a power result as much '
                           'as an absence'})))

    out += ['### 7e. RQ5 -- shift gradient', '']
    rq5 = est.get('rq5') or {}
    for family, block in rq5.items():
        if not block.get('available'):
            out += [f'- `{family}`: not estimable in this run tree.']
            continue
        out += [f'**{family}**', '']
        out += md_table(block.get('levels') or [],
                        ('level', 'n', 'hl', 'ci_lo', 'ci_hi'))
        out += md_table([block], ('standardised', 'ci_lo', 'ci_hi', 'note'))
    out.append('')
    out += wrap('The wind family is the primary axis: its no-op score is flat '
                'across levels, so difficulty is held roughly constant while '
                'the dynamics change. The gravity family is secondary and '
                'carries the caveat that weakening gravity makes the task '
                'easier as well as different -- the no-op score rises from 0.18 '
                'to 0.55 -- so it may not carry H4 alone (DESIGN.md 5.1, 6.2).')
    out.append('')

    out += ['### 7f. RQ6 -- budget', '']
    rq6 = est.get('rq6') or {}
    out += md_table(rq6.get('rows') or [],
                    ('cell', 'prefix', 'n', 'delta_at_prefix',
                     'delta_at_budget'))
    out += wrap('An episode prefix is a valid budget counterfactual only '
                'because the exploration schedule is a closed-form function of '
                'elapsed env steps and never reads the budget, which '
                '`validate.py` asserts. The comparison is single held-out '
                'checkpoint against single final checkpoint, never against the '
                'three-checkpoint mean (DESIGN.md 2.4 RQ6).')
    out.append('')

    log.section = 'Section 7g (dispersion)'
    out += ['### 7g. Dispersion', '']
    disp = (est.get('dispersion') or {}).get('rows') or []
    out += md_table(disp, ('cell', 'n', 'sd_scratch', 'sd_transfer',
                           'sd_ratio', 'ci_lo', 'ci_hi', 'brown_forsythe_W'))
    out += wrap('The published study conflated within-run instability with '
                'across-seed sensitivity and then described the result '
                'backwards. They are separate metrics here for that reason, '
                'and no dispersion p-value is interpreted: at the sample sizes '
                'in the table above, a variance test on an SD ratio of this '
                'size has almost no power, which is the honest explanation of '
                'the published null. Each ratio below carries the bootstrap '
                'interval from the same row (ANALYSIS_PLAN.md 3), read against '
                '1 rather than against zero.')
    out.append('')
    for r in disp:
        out += render_claim_md(log.add(claim(
            '', 'dispersion',
            {'estimand': 'dispersion', 'n': r.get('n'),
             'sd_a': r.get('sd_transfer'), 'sd_b': r.get('sd_scratch'),
             'ci_lo': r.get('ci_lo'), 'ci_hi': r.get('ci_hi'),
             'name_a': f'{r.get("cell")} transfer',
             'name_b': f'{r.get("cell")} scratch',
             'counterfactual': 'the same cell scratch arm across the same '
                               'seeds',
             'rivals': ['a single outlying seed dominating an SD at this n'],
             'excluded_by': 'the bootstrap interval on the ratio, which is '
                            'reported beside it',
             'refuted_by': 'a ratio interval covering 1'})))

    log.section = 'Section 7h (censored)'
    out += ['### 7h. Censored steps-to-threshold', '']
    out += wrap('Right-censored at the budget, administratively and identically '
                'for every run. Never imputed, never dropped: the primary '
                'summary is P(threshold reached within budget) as k/n with a '
                'Clopper-Pearson exact interval, and the log-rank test is run '
                'only where both arms have at least 3 events '
                '(ANALYSIS_PLAN.md 5).')
    out.append('')
    cens = est.get('censored') or {}
    out += md_table(_flatten_censored(cens),
                    ('level_key', 'level', 'cell', 'condition', 'label', 'n',
                     'reached', 'p_reached', 'cp_lo', 'cp_hi',
                     'km_median_steps'))
    for level_key, block in cens.items():
        logrank = block.get('logrank') or []
        if logrank:
            out += [f'**log-rank, {level_key}**', '']
            out += md_table(logrank, tuple(logrank[0].keys()))
        else:
            out += [f'- `{level_key}`: no log-rank test -- fewer than '
                    f'{stats_mod.LOGRANK_MIN_EVENTS} events in at least one '
                    f'arm, so the proportion and its interval stand alone.']
    out.append('')
    zero_event = [a for b in cens.values() for a in (b.get('arms') or [])
                  if (_int_or_none(a.get('reached')) or 0) == 0
                  and (_int_or_none(a.get('n')) or 0)
                  >= stats_mod.MIN_N_FOR_INFERENCE]
    for arm in zero_event[:8]:
        out += render_claim_md(log.add(claim(
            f'No run in arm {arm.get("label")} ({arm.get("cell")}, '
            f'{arm.get("condition")}) reached the threshold within budget: '
            f'{arm.get("reached")}/{arm.get("n")}, Clopper-Pearson '
            f'[{f_(arm.get("cp_lo"))}, {f_(arm.get("cp_hi"))}]. The upper bound '
            f'is the informative statement.',
            'descriptive',
            {'estimand': 'censoring_proportion', 'n': arm.get('n'),
             'counterfactual': 'the same arm at a larger budget, which is not '
                               'in the data',
             'rivals': ['the budget being too short for any arm, rather than '
                        'anything about this one'],
             'excluded_by': 'the same table for every other arm at the same '
                            'budget',
             'refuted_by': 'one run crossing the threshold, which moves k'})))

    log.section = 'Section 7i (secondary and mechanism)'
    out += ['### 7i. Secondary endpoints', '']
    out += md_table(_secondary_rows(est.get('secondary') or {}),
                    ('metric', 'role', 'cell', 'n', 'transfer_mean',
                     'scratch_mean', 'hl_delta', 'ci_lo', 'ci_hi'))
    out += ['### 7j. Mechanism signals (DESIGN.md 5.5)', '']
    mech = _mechanism_rows(est.get('secondary') or {})
    out += md_table(mech, ('metric', 'role', 'cell', 'n', 'transfer_mean',
                           'scratch_mean', 'hl_delta', 'ci_lo', 'ci_hi'))
    out += wrap('A mechanism claim must cite one of these signals. There is no '
                'free-text mechanism slot in this template: `claim()` refuses '
                'kind=mechanism unless the evidence names a signal from the '
                'DESIGN.md 5.5 table.')
    out.append('')
    for r in mech:
        if not _finite(r.get('hl_delta')):
            continue
        # _f: hl_delta being finite says nothing about the bounds, and a null
        # bound reaching np.isfinite raised out of the whole run.
        verdict = stats_mod.phrase_interval_verdict(
            _f(r.get('ci_lo')), _f(r.get('ci_hi')),
            f'the transfer-minus-scratch shift in {r.get("metric")}')
        out += render_claim_md(log.add(claim(
            f'In cell {r.get("cell")}, {verdict}.', 'mechanism',
            {'estimand': 'mechanism_signal', 'signal': r.get('metric'),
             'n': r.get('n'), 'ci_lo': r.get('ci_lo'),
             'ci_hi': r.get('ci_hi'),
             'counterfactual': 'the matched-seed scratch run measured on the '
                               'same fixed diagnostic state batch',
             'rivals': ['the signal moving because performance moved, rather '
                        'than performance moving because the signal did'],
             'excluded_by': 'nothing in this bundle: the direction of the '
                            'relationship is not identified, so this is an '
                            'instrumented association reported beside the '
                            'effect',
             'refuted_by': 'an interval covering zero, or the signal moving in '
                           'the same direction in the C2 untrained-source arm, '
                           'which would attribute it to protocol mechanics'})))

    out += ['### 7k. Ablation screens', '']
    screens = est.get('screens') or {}
    out += md_table(screens.get('rows') or [],
                    ('experiment', 'metric', 'cell', 'level', 'n', 'hl',
                     'ci_lo', 'ci_hi', 'q'))
    out += wrap('Benjamini-Hochberg q values, where present, are for '
                'orientation only and are never an assertion. A screen result '
                'selects at most one follow-up, which is then run on '
                'REPLICATE seeds and reported as a fresh estimate '
                '(ANALYSIS_PLAN.md 3).')
    out.append('')

    out += ['### 7l. Source competence as a covariate', '']
    out += md_table((est.get('secondary') or {}).get('source_competence') or [],
                    ('cell', 'n', 'slope', 'ci_lo', 'ci_hi'))
    out += wrap('A descriptive relationship between the delta and the source '
                'normalised score, not a mediation claim (DESIGN.md 4.3).')
    out.append('')


def section_power(out: list[str], ctx: dict) -> None:
    sj = ctx['stats']
    power = sj.get('s10_power', {}) or {}
    out += ['## 8. Power and minimum detectable effects', '']
    out += wrap('Computed before launch and, per ANALYSIS_PLAN.md 6.4, not '
                're-tuned after seeing results. Which cells are powered was '
                'therefore known in advance rather than discovered afterwards.')
    out.append('')
    # The multipliers were written into this paragraph by hand as the n=10
    # values (1.00 / 1.54 / 1.39 / 1.87) and printed unchanged above a table of
    # a different n. They are stats.py's own numbers, so they are read from it.
    mult = power.get('multipliers') or {}
    if mult:
        out += wrap('The MDE multipliers on the relevant sigma, as stats.py '
                    'reports them: '
                    + '; '.join(f'{k} = {f_(v, 2)}'
                                for k, v in mult.items())
                    + '. The paired test is primary wherever its multiplier is '
                      'the smaller one, which is a property of the design '
                      'rather than of this run.')
        out.append('')
    ns = sorted({v for v in (_int_or_none(r.get('n'))
                             for r in power.get('per_member') or [])
                 if v is not None})
    if ns:
        out += wrap('The per-member table below is at n='
                    + ', '.join(str(v) for v in ns)
                    + '; an MDE is sigma_delta times the multiplier above, and '
                      'sigma_delta is the observed across-seed SD of the '
                      'paired delta in that cell.')
        out.append('')
    out += md_table(power.get('per_member') or [],
                    ('metric', 'cell', 'n', 'sd_delta', 'mde_nominal',
                     'mde_holm8', 'powered', 'note'))
    out += ['**MDE multipliers on the relevant sigma**', '']
    out += md_table([{'test': k, 'multiplier': v} for k, v
                     in (power.get('multipliers') or {}).items()],
                    ('test', 'multiplier'))
    out += ['**Planned versus observed dispersion** (a comparison, not a '
            'recalibration)', '']
    out += md_table(power.get('planning_comparison') or [],
                    ('arm', 'planned_sd', 'observed_sd', 'ratio'))


def section_ledger(out: list[str], ctx: dict, log: ClaimLog) -> None:
    sj = ctx['stats']
    ledger = sj.get('s11_ledger', {}) or {}
    out += ['## 9. Multiplicity ledger', '']
    out += wrap('Printed on every invocation, so the count is a recorded fact '
                'rather than a claim (ANALYSIS_PLAN.md 7). Family membership '
                'is fixed by the pre-registration and read from '
                '`stats.py` own constants, never taken as an argument, which '
                'is what prevents a result from being rescued by relocating it '
                'into a family of one.')
    out.append('')
    out += md_table(ledger.get('families') or [],
                    ('family', 'members', 'procedure', 'adjusted_alpha'))
    rows = [
        {'quantity': 'confirmatory family size (fixed)',
         'count': stats_mod.CONFIRMATORY_FAMILY_SIZE},
        {'quantity': 'confirmatory tests actually computed',
         'count': len(ledger.get('confirmatory') or [])},
        {'quantity': 'confirmatory members suppressed with a reason',
         'count': len(ledger.get('suppressed') or [])},
        {'quantity': 'estimation-only analysis sections',
         'count': len(ledger.get('estimation_only') or [])},
        {'quantity': 'screen q-values reported for orientation',
         'count': ledger.get('screen_q_count')},
        {'quantity': 'refusals recorded by stats.py',
         'count': len(ledger.get('refusals') or [])},
        {'quantity': 'adjusted alpha, strictest Holm step',
         'count': stats_mod.ALPHA_STRICTEST},
        {'quantity': 'claims emitted by report.py',
         'count': len(log.accepted)},
        {'quantity': 'claims refused by report.py',
         'count': len(log.refused)},
        {'quantity': 'emitted claims carrying a p-value',
         'count': len(log.with_p())},
        {'quantity': 'emitted claims carrying no p-value',
         'count': len(log.without_p())},
    ]
    out += md_table(rows, ('quantity', 'count'))
    if ledger.get('suppressed'):
        out += ['**Suppressed confirmatory members**', '']
        for item in ledger['suppressed']:
            out.append(f'- {item}')
        out.append('')
    if ledger.get('refusals'):
        out += ['**Refusals recorded by stats.py**', '']
        for item in ledger['refusals']:
            out.append(f'- {item}')
        out.append('')


def section_no_pvalue(out: list[str], ctx: dict, log: ClaimLog) -> None:
    sj = ctx['stats']
    ledger = sj.get('s11_ledger', {}) or {}
    out += ['## 10. Analyses carrying no p-value', '']
    out += wrap('Everything in this list is estimation-only by '
                'pre-registration. The list is the point: at the sample sizes '
                'this design was planned for, the achievable MDEs are large '
                'enough that spending the error budget on more than one family '
                'would leave nothing powered at all, so the study tests very '
                'little and estimates everything (ANALYSIS_PLAN.md, preamble '
                'and 2).')
    out.append('')
    for item in ledger.get('estimation_only') or []:
        out.append(f'- {item}')
    out.append('')
    # Read off the provenance records rather than asserted. Naming only
    # confirmatory.csv here was wrong: screens.csv carries a BH q column, which
    # ANALYSIS_PLAN.md 3 permits for orientation, and the sentence denied its
    # existence while the same section's list acknowledged it.
    with_p = [t for t in ctx['table_index']
              if t.get('p_value_columns') not in (None, '', '--')]
    if with_p:
        listed = '; '.join(f'`tables/{t["table"]}.csv` ({t["p_value_columns"]})'
                           for t in with_p)
        carried = (f'Every table in this bundle carries no p-value except '
                   f'{listed}.')
        if any(t['table'] != 'confirmatory' for t in with_p):
            carried += (' Only `tables/confirmatory.csv` is a test: the '
                        'adjusted p-values elsewhere are Benjamini-Hochberg '
                        'orientation and are never asserted '
                        '(ANALYSIS_PLAN.md 3, 7).')
    else:
        carried = 'No table in this bundle carries a p-value column.'
    out += wrap(f'Plus every figure drawn by `plots.py` (which emits intervals '
                f'and distributions only), and {len(log.without_p())} of the '
                f'{len(log.accepted)} claim(s) emitted above. {carried}')
    out.append('')
    by_kind: dict[str, int] = {}
    for c in log.accepted:
        by_kind[c.kind] = by_kind.get(c.kind, 0) + 1
    out += md_table([{'claim kind': k, 'emitted': v}
                     for k, v in sorted(by_kind.items())],
                    ('claim kind', 'emitted'))


def section_guardrails(out: list[str], ctx: dict, log: ClaimLog) -> None:
    # Not "exercised in this invocation": the heading used to say that over a
    # catalogue that listed guards which cannot fire on real data at all.
    out += ['## 11. Claim guardrails, and which of them fired here', '']
    out += wrap('Every sentence in sections 2-7 above passed through '
                '`claim(text, kind, evidence)`, which refuses a sentence whose '
                'evidence does not support the kind of claim it makes and '
                'names the section that forbids it. The refusals below are '
                'output, not errors: a refused sentence is a sentence the data '
                'cannot carry.')
    out.append('')
    out += ['### 11a. The guardrail catalogue', '']
    out += wrap('The table is generated from the guard ids `claim()` records '
                'in its refusals, so it cannot advertise a guard the code no '
                'longer has, and the last column is a count of what actually '
                'happened in this invocation rather than an implication that '
                'all of them fired. A guard that refused nothing here is not '
                'thereby idle: most sentences in this template are built not '
                'to trip it. A guard marked *unreachable from this template* '
                'is a different thing, and says where the real enforcement is.')
    out.append('')
    fired: dict[str, int] = {}
    for c in log.refused:
        for guard, _where, _why in c.refusals:
            fired[guard] = fired.get(guard, 0) + 1
    rows = [{'guard': guard, 'requires': requires, 'section': section,
             'refused here': fired.get(guard, 0),
             'note': note or ('--' if fired.get(guard) else
                              'did not fire in this invocation')}
            for guard, requires, section, note in GUARD_CATALOGUE]
    out += md_table(rows, ('guard', 'requires', 'section', 'refused here',
                           'note'))
    unknown = sorted(set(fired) - {g for g, _r, _s, _n in GUARD_CATALOGUE})
    if unknown:
        out += wrap(f'{WARN} refusals were recorded under guard id(s) '
                    f'{", ".join(unknown)}, which are not in the catalogue '
                    f'above. The catalogue is incomplete.')
        out.append('')
    out += ['### 11b. Forbidden phrasings', '']
    out += md_table([{'pattern': p, 'section': s, 'why': w}
                     for p, s, w in FORBIDDEN_PHRASES]
                    + [{'pattern': p, 'section':
                        'ANALYSIS_PLAN.md 8, DESIGN.md 9',
                        'why': 'the "A avoids it, B does not" sentence form, '
                               'permitted only with an explicit between-cell '
                               'contrast and its interval'}
                       for p in VERDICT_COMPARISON_PATTERNS],
                    ('pattern', 'section', 'why'))
    if log.refused:
        out += [f'### 11c. Refused in this invocation '
                f'({len(log.refused)} of {len(log.claims)})', '']
        for c in log.refused:
            out += render_claim_md(c)
    else:
        out += ['### 11c. Refused in this invocation', '',
                'None. Every sentence the template proposed was licensed by '
                'its evidence.', '']


def section_deviations(out: list[str], ctx: dict) -> None:
    sj = ctx['stats']
    dev = sj.get('s12_deviations', {}) or {}
    out += ['## 12. Deviations, tensions and open items', '']
    items = dev.get('deviations') or []
    if items:
        out += ['**Deviations from the pre-registered plan, as recorded by '
                '`stats.py`**', '']
        for item in items:
            out.append(f'- {item}')
        out.append('')
    else:
        out += ['No deviation from `ANALYSIS_PLAN.md` was recorded.', '']
    tensions = dev.get('tensions') or []
    if tensions:
        out += ['**Tensions between the plan and what the data can support**',
                '']
        for item in tensions:
            out.append(f'- {item}')
        out.append('')
    if str(ctx.get('source_policy')) == 'pooled':
        out += ['- ' + POOLED_SOURCE_LABEL + ' Every emitted sentence in this '
                'bundle carries that label in its own text, so it survives '
                'being copied out (DESIGN.md 4.3).', '']
    if ctx['override']:
        out += ['- ' + OVERRIDE_STAMP + '.', '']
    if ctx['skipped_plots']:
        out += ['- Figures were skipped (`--skip-plots`), so this bundle '
                'carries no figure provenance records.', '']


def section_artifacts(out: list[str], ctx: dict) -> None:
    out += ['## 13. Artifacts in this bundle', '']
    out += wrap('`MANIFEST.json` lists every file with its blake2b hash. '
                '`per_seed.csv`, `curves.csv` and `stats.json` are copies, so '
                'the bundle is readable without the run tree, and every table '
                'records the hash of the inputs it was built from '
                '(DESIGN.md 8.3, 9 "stale artifacts").')
    out.append('')
    out += md_table(ctx['table_index'], ('table', 'title', 'rows', 'csv'))
    if ctx['figures']:
        out += ['### Figures', '']
        out += md_table([{'figure': f} for f in ctx['figures']], ('figure',))


def build_report(ctx: dict, log: ClaimLog) -> str:
    out: list[str] = []
    out += [f'# Results bundle {ctx["date"]}', '']
    out += wrap('Produced by `experiments/report.py` in one invocation: audit, '
                'aggregation, the pre-registered analysis, figures and tables. '
                'The order of sections below is the order ANALYSIS_PLAN.md 10 '
                'prescribes, and no number here was computed by this file: '
                'every one is read from `stats.json` or from the pinned '
                '`per_seed.csv` (STANDING_INSTRUCTIONS S6).')
    out.append('')
    section_stamps(out, ctx)
    section_provenance(out, ctx)
    section_inventory(out, ctx, log)
    section_confirmatory(out, ctx, log)
    section_equivalence(out, ctx, log)
    section_controls(out, ctx, log)
    section_c4(out, ctx, log)
    section_estimation(out, ctx, log)
    section_power(out, ctx)
    section_ledger(out, ctx, log)
    section_no_pvalue(out, ctx, log)
    section_guardrails(out, ctx, log)
    section_deviations(out, ctx)
    section_artifacts(out, ctx)
    return '\n'.join(out).rstrip() + '\n'


# ===========================================================================
# 7. The bundle manifest.
# ===========================================================================
ROLE_BY_DIR = {
    'data': 'input table copied for self-containment',
    'stats': 'pre-registered analysis output',
    'audit': 'invariant check',
    'figures': 'figure, caption or figure provenance',
    'tables': 'table or table provenance',
    'logs': 'stage log',
}


def write_manifest(bundle: str, ctx: dict, log: ClaimLog) -> str:
    # The bundle stops being incomplete here and nowhere else: the manifest is
    # the last file written, and it is what makes the directory citable.
    marker = os.path.join(bundle, INCOMPLETE_MARKER)
    if os.path.isfile(marker):
        os.remove(marker)
    entries: list[dict] = []
    for root, _dirs, files in os.walk(bundle):
        for name in sorted(files):
            if name == 'MANIFEST.json':
                continue
            path = os.path.join(root, name)
            relative = rel(path, bundle)
            entries.append({
                'path': relative,
                'bytes': os.path.getsize(path),
                'blake2b_128': provenance.file_hash(path),
                'role': ROLE_BY_DIR.get(relative.split('/')[0], 'report'),
            })
    audit_report = ctx['audit_report']
    manifest = {
        'bundle': os.path.abspath(bundle),
        'date': ctx['date'],
        'generated': ctx['timestamp'],
        'tool': 'experiments/report.py',
        'argv': list(sys.argv),
        'cwd': os.getcwd(),
        'out_root': ctx['out_root'],
        'experiments_requested': ctx['experiments'],
        'source_policy': ctx.get('source_policy'),
        'source_policy_is_secondary_estimand':
            str(ctx.get('source_policy')) == 'pooled',
        'audit': {
            'ok': audit_report.get('ok'),
            'errors': audit_report.get('errors'),
            'warnings': audit_report.get('warnings'),
            'failing_checks': [c['name'] for c in audit_report.get('checks', [])
                               if c.get('status') == 'FAIL'],
            'override_in_force': ctx['override'],
            'override_stamp': OVERRIDE_STAMP if ctx['override'] else None,
        },
        'validation_stamp': VALIDATION_STAMP if ctx['validation_stamp'] else None,
        'plan_drift': ctx['plan_drift'],
        'stages': [{'name': s.name, 'command': s.command,
                    'exit_code': s.exit_code, 'seconds': round(s.seconds, 3),
                    'ok': s.ok, 'note': s.note,
                    'log': (rel(s.log, bundle) if s.log else None)}
                   for s in ctx['stages']],
        'inputs': {
            'per_seed_csv': ctx['per_seed_src'],
            'per_seed_sha': ctx['hashes'].get('per_seed'),
            'per_seed_rows': ctx['per_seed_rows'],
            'curves_csv': ctx['curves_src'],
            'curves_sha': ctx['hashes'].get('curves'),
            'curves_rows': ctx['curves_rows'],
            'stats_json_sha': ctx['hashes'].get('stats'),
        },
        'claims': {
            'emitted': len(log.accepted),
            'refused': len(log.refused),
            'carrying_a_p_value': len(log.with_p()),
            'carrying_no_p_value': len(log.without_p()),
            # DESIGN.md 2.1 and 4.3 as counts, so the properties can be checked
            # from the manifest without reading the prose. The defect they
            # record was found exactly this way: by_kind causal=0 over 106
            # emitted claims meant no sentence carried the scope clause.
            'carrying_the_scope_clause':
                sum(1 for c in log.accepted if c.text.startswith(SCOPE_CLAUSE)),
            'carrying_the_pooled_source_label':
                sum(1 for c in log.accepted
                    if POOLED_SOURCE_LABEL in c.text),
            'refusals_by_guard': {
                guard: sum(1 for c in log.refused
                           for g, _w, _y in c.refusals if g == guard)
                for guard, _r, _s, _n in GUARD_CATALOGUE},
            'by_kind': {k: sum(1 for c in log.accepted if c.kind == k)
                        for k in KINDS},
            'refusal_sections': sorted({where for c in log.refused
                                        for _g, where, _w in c.refusals}),
        },
        'tables': ctx['table_index'],
        'figures': ctx['figures'],
        'provenance': ctx['provenance'],
        'files': entries,
        'file_count': len(entries),
    }
    path = os.path.join(bundle, 'MANIFEST.json')
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(manifest, fh, indent=2, sort_keys=False, default=str)
    return path


def write_claims_json(bundle: str, log: ClaimLog) -> str:
    payload = [{
        'section': c.section, 'kind': c.kind, 'estimand': c.estimand,
        'accepted': c.accepted, 'generated': c.generated,
        'text': c.text, 'proposed_text': c.proposed,
        'refusals': [{'guard': g, 'section': w, 'why': y}
                     for g, w, y in c.refusals],
        'socratic': [{'tag': t, 'question': q, 'answer': a}
                     for t, q, a in c.socratic],
        'carries_p_value': c.carries_p,
        'evidence': c.evidence,
    } for c in log.claims]
    path = os.path.join(bundle, 'claims.json')
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, indent=2, sort_keys=False, default=str)
    return path


# ===========================================================================
# 8. Self-test. The guardrails have to be shown to bite, because the run data
#    will not naturally contain a forbidden sentence -- this module writes the
#    sentences, so the only way to demonstrate the refusals is to try them.
# ===========================================================================
def self_test() -> int:
    failures: list[str] = []

    def check(name: str, condition: bool, detail: str = '') -> None:
        if not condition:
            failures.append(f'{name}: {detail or "failed"}')

    good = {'estimand': 'within_cell_delta', 'n': 10, 'ci_lo': -0.30,
            'ci_hi': -0.10, 'confirmatory': True, 'p_holm': 0.004,
            'counterfactual': 'the same cell scratch run at the same seed',
            'rivals': ['protocol mechanics'], 'excluded_by': 'C2',
            'refuted_by': 'an interval covering zero'}

    c = claim('the paired shift is negative', 'causal', dict(good))
    check('causal accepted', c.accepted, str(c.refusals))
    check('scope clause prefixed', c.text.startswith(SCOPE_CLAUSE), c.text[:60])
    check('four socratic questions', len(c.socratic) == 4)

    c = claim('cell A differs from cell B', 'causal',
              {**good, 'estimand': 'between_cell_delta'})
    check('causal refused off a between-cell estimand', not c.accepted)

    c = claim('the delta is negative', 'causal', {**good, 'n': 2})
    check('n<3 refused', not c.accepted)
    check('n<3 cites the plan',
          any('ANALYSIS_PLAN.md 9' in w for _g, w, _y in c.refusals))

    c = claim('the delta is negative', 'causal', {**good, 'seed_block': 'TUNE'})
    check('TUNE refused', not c.accepted)

    c = claim('the delta is negative', 'causal', {**good, 'ci_lo': None,
                                                  'ci_hi': None})
    check('no interval refused', not c.accepted)

    c = claim('the delta is negative', 'causal',
              {k: v for k, v in good.items() if k != 'refuted_by'})
    check('unrefutable refused', not c.accepted)
    check('unrefutable cites S5',
          any('S5' in w for _g, w, _y in c.refusals))

    c = claim('the arm shows positive transfer', 'causal', dict(good))
    check('positive transfer refused', not c.accepted)
    c = claim('this proves the protocol helps', 'causal', dict(good))
    check('proves refused', not c.accepted)
    c = claim('the result demonstrates that architecture determines transfer',
              'effect_modification',
              {**good, 'estimand': 'between_cell_delta', 'confirmatory': False,
               'p_holm': None,
               'between_cell_contrast': {'ci_lo': -0.3, 'ci_hi': -0.1}})
    check('architecture-determines refused', not c.accepted)
    c = claim('the arms had identical baselines', 'descriptive',
              {'estimand': 'arm_descriptive', 'n': 10,
               'refuted_by': 'an invariant that moved'})
    check('identical baselines refused', not c.accepted)
    c = claim('the transfer arm has a broader spread', 'dispersion',
              {'estimand': 'dispersion', 'n': 10, 'sd_a': 0.05, 'sd_b': 0.20,
               'ci_lo': 0.10, 'ci_hi': 0.60,
               'name_a': 'transfer', 'name_b': 'scratch'})
    check('broader spread refused', not c.accepted)
    check('broader spread cites DESIGN 9',
          any('DESIGN.md 9' in w for _g, w, _y in c.refusals))

    c = claim('mlp-double avoids the degradation while dueling-vanilla does '
              'not', 'effect_modification',
              {**good, 'estimand': 'between_cell_delta', 'confirmatory': False,
               'p_holm': None})
    check('verdict-comparison refused without a contrast', not c.accepted)
    c = claim('mlp-double avoids the degradation while dueling-vanilla does '
              'not', 'effect_modification',
              {**good, 'estimand': 'between_cell_delta', 'confirmatory': False,
               'p_holm': None,
               'between_cell_contrast': {'ci_lo': 0.10, 'ci_hi': 0.40}})
    check('verdict-comparison permitted with a contrast', c.accepted,
          str(c.refusals))

    c = claim('the two arms are equivalent', 'equivalence',
              {**good, 'verdict': 'UNTESTABLE'})
    check('equivalence refused on UNTESTABLE', not c.accepted)
    # The two SDs are part of the evidence an EQUIVALENT verdict needs, not an
    # optional extra: see the fail-closed case below.
    inside = {'sd_a': 0.02, 'sd_b': 0.03}
    c = claim('the two arms are equivalent', 'equivalence',
              {**good, 'verdict': 'EQUIVALENT', 'ci_lo': -0.02, 'ci_hi': 0.01,
               **inside})
    check('equivalence accepted on EQUIVALENT', c.accepted, str(c.refusals))
    c = claim('the arms are equivalent', 'equivalence',
              {**good, 'verdict': 'EQUIVALENT', 'ci_lo': -0.02,
               'ci_hi': 0.01, 'basis': 'p_value', **inside})
    check('equivalence from a p-value refused', not c.accepted)
    check('p-value refusal cites the basis guard',
          any(g == 'equivalence-basis' for g, _w, _y in c.refusals),
          str(c.refusals))

    c = claim('the trunk representation drifted', 'mechanism',
              {**good, 'estimand': 'mechanism_signal', 'signal': 'vibes'})
    check('unnamed mechanism signal refused', not c.accepted)
    c = claim('the trunk representation drifted', 'mechanism',
              {**good, 'estimand': 'mechanism_signal', 'signal': 'cka_drift'})
    check('named mechanism signal accepted', c.accepted, str(c.refusals))

    c = claim('the effect is significant at p < 0.01', 'associational',
              {**good, 'estimand': 'between_cell_scratch',
               'confirmatory': False, 'p_holm': None})
    check('p-value outside the family refused', not c.accepted)
    c = claim('the cells show no difference', 'associational',
              {**good, 'estimand': 'between_cell_scratch',
               'confirmatory': False, 'p_holm': None})
    check('affirming a null refused', not c.accepted)
    c = claim('the architecture causes the gap', 'associational',
              {**good, 'estimand': 'between_cell_scratch',
               'confirmatory': False, 'p_holm': None})
    check('causal verb in an associational sentence refused', not c.accepted)
    c = claim('the arm gained 40 return points', 'causal',
              {**good, 'scale': 'raw_return'})
    check('raw-return claim refused', not c.accepted)

    # The dispersion sentence is generated, and reversing the arguments has to
    # reverse the word -- the published paper's actual error.
    disp = {'estimand': 'dispersion', 'n': 10, 'ci_lo': 1.60, 'ci_hi': 9.00,
            'name_a': 'A', 'name_b': 'B'}
    c1 = claim('', 'dispersion', {**disp, 'sd_a': 0.20, 'sd_b': 0.05})
    c2 = claim('', 'dispersion', {**disp, 'sd_a': 0.05, 'sd_b': 0.20,
                                  'ci_lo': 0.11, 'ci_hi': 0.62})
    check('dispersion word generated', 'wider' in c1.text and c1.accepted,
          c1.text)
    check('dispersion word reverses', 'narrower' in c2.text and c2.accepted,
          c2.text)
    check('dispersion carries its interval',
          'SD ratio 95% interval' in c1.text, c1.text)
    c = claim('', 'dispersion', {**disp, 'sd_a': 0.20, 'sd_b': 0.05,
                                 'ci_lo': None, 'ci_hi': None})
    check('dispersion with no interval refused', not c.accepted)
    check('dispersion interval refusal cites the plan',
          any(g == 'interval' for g, _w, _y in c.refusals))
    c = claim('', 'dispersion', {**disp, 'sd_a': 0.30, 'sd_b': 0.30,
                                 'ci_lo': 0.40, 'ci_hi': 2.10})
    check('equal SDs are not an affirmed null', not c.accepted, c.text)

    # A relative effect is directional, so its verb comes out of theta. The
    # hand-written "exceeds" below is the sentence that shipped over
    # theta=0.469 while the same sentence printed the number.
    rel = {'estimand': 'between_cell_scratch', 'n': 10, 'ci_lo': 0.13,
           'ci_hi': 0.83, 'name_a': 'A', 'name_b': 'B',
           'confirmatory': False, 'p_holm': None,
           'refuted_by': 'an interval covering 0.5'}
    c = claim('a random run of A exceeds a random run of B', 'associational',
              {**rel, 'theta': 0.469})
    check('direction verb against theta refused', not c.accepted, c.text)
    check('direction refusal cites DESIGN 9',
          any(g == 'relative-effect-direction' for g, _w, _y in c.refusals))
    c = claim('a random run of A exceeds a random run of B', 'associational',
              {**rel, 'theta': 0.72, 'ci_lo': 0.55, 'ci_hi': 0.90})
    check('direction verb agreeing with theta accepted', c.accepted,
          str(c.refusals))
    check('theta sentence is generated', 'scores above' in c.text, c.text)
    check('theta interval read against 0.5', 'above 0.5' in c.text, c.text)
    c = claim('', 'associational', {**rel, 'theta': 0.33})
    check('theta below the null renders as below', 'scores below' in c.text,
          c.text)
    check('theta interval covering 0.5 says so', 'covers 0.5' in c.text,
          c.text)
    c = claim('', 'associational', {**rel, 'theta': 0.33, 'ci_lo': None,
                                    'ci_hi': None})
    check('a relative effect with no interval is refused', not c.accepted,
          c.text)

    # Equivalence: the verdict is re-derived from containment, never trusted.
    c = claim('the transfer arm is equivalent to its scratch baseline',
              'equivalence',
              {'estimand': 'within_cell_delta', 'n': 10, 'ci_lo': -0.30,
               'ci_hi': -0.10, 'verdict': 'EQUIVALENT',
               'refuted_by': 'an interval that leaves the margin'})
    check('EQUIVALENT outside the margin refused', not c.accepted, c.text)
    check('containment refusal cites the plan',
          any(g == 'equivalence-containment' for g, _w, _y in c.refusals))
    c = claim('the paired delta is not distinguishable from zero',
              'equivalence',
              {'estimand': 'within_cell_delta', 'n': 10, 'ci_lo': -0.02,
               'ci_hi': 0.01, 'verdict': 'DIFFERENT',
               'refuted_by': 'a wider interval'})
    check('DIFFERENT inside the margin refused', not c.accepted, c.text)
    c = claim('the equivalence verdict is UNTESTABLE', 'equivalence',
              {'estimand': 'within_cell_delta', 'n': 10, 'ci_lo': -0.02,
               'ci_hi': 0.01, 'verdict': 'UNTESTABLE', 'sd_a': 0.01,
               'sd_b': 0.02, 'refuted_by': 'a smaller SD'})
    check('UNTESTABLE with SDs inside the margin refused', not c.accepted,
          c.text)
    c = claim('the equivalence verdict is UNTESTABLE', 'equivalence',
              {'estimand': 'within_cell_delta', 'n': 10, 'ci_lo': -0.02,
               'ci_hi': 0.01, 'verdict': 'UNTESTABLE', 'sd_a': 0.55,
               'sd_b': 0.62, 'refuted_by': 'a smaller SD'})
    check('UNTESTABLE with a dispersion above the margin accepted',
          c.accepted, str(c.refusals))
    c = claim('the two arms are equivalent', 'equivalence',
              {'estimand': 'within_cell_delta', 'n': 10, 'ci_lo': -0.02,
               'ci_hi': 0.01, 'verdict': 'EQUIVALENT', 'sd_a': 0.60,
               'sd_b': 0.01, 'refuted_by': 'a cell SD above the margin'})
    check('EQUIVALENT over a dispersion above the margin refused',
          not c.accepted, c.text)
    check('the dispersion gate is its own guard',
          any(g == 'equivalence-feasibility' for g, _w, _y in c.refusals),
          str(c.refusals))

    # The gate governs EQUIVALENCE and not difference. ANALYSIS_PLAN.md 6.3
    # pins the dueling-vanilla scratch SD at 0.369 and the mlp-double one at
    # 0.093, both above the margin, so refusing DIFFERENT here withheld the
    # per-cell statement in exactly the two cells with the largest effects and
    # printed "the verdict may only be UNTESTABLE" beneath a table row reading
    # DIFFERENT.
    different = {'estimand': 'within_cell_delta', 'n': 10, 'ci_lo': -0.60,
                 'ci_hi': -0.40, 'verdict': 'DIFFERENT', 'margin': 0.05,
                 'refuted_by': 'an interval that re-enters the margin'}
    c = claim('the equivalence verdict is DIFFERENT', 'equivalence',
              {**different, 'sd_a': 0.10, 'sd_b': 0.369})
    check('DIFFERENT over a dispersion above the margin accepted',
          c.accepted, str(c.refusals))
    c = claim('the equivalence verdict is DIFFERENT', 'equivalence',
              {**different, 'sd_a': 0.01, 'sd_b': 0.02})
    check('DIFFERENT under a small dispersion accepted', c.accepted,
          str(c.refusals))
    for label, sds in (('absent', {}),
                       ('null', {'sd_a': None, 'sd_b': None}),
                       ('NaN', {'sd_a': float('nan'),
                                'sd_b': float('nan')})):
        c = claim('the two arms are equivalent', 'equivalence',
                  {'estimand': 'within_cell_delta', 'n': 10, 'ci_lo': -0.02,
                   'ci_hi': 0.01, 'verdict': 'EQUIVALENT',
                   'refuted_by': 'a cell SD above the margin', **sds})
        check(f'EQUIVALENT with SDs {label} refused rather than passed',
              not c.accepted and any(g == 'equivalence-feasibility'
                                     for g, _w, _y in c.refusals),
              f'{c.accepted} {c.refusals}')

    # A zero-width interval is an absence of information, and every sentence
    # form that reads an interval has to treat it as one. stats.py emits the
    # DEGENERATE verdict for it; this module used to publish an exclusion bound
    # of +0.0000 and "the ordering is not a sampling artefact at this n" from
    # the same rows.
    check('containment re-derivation calls a point degenerate',
          implied_equivalence_verdict(0.0, 0.0, 0.05) == 'DEGENERATE',
          implied_equivalence_verdict(0.0, 0.0, 0.05))
    degen = {'estimand': 'within_cell_delta', 'n': 10, 'ci_lo': 0.0,
             'ci_hi': 0.0, 'margin': 0.05, 'sd_a': 0.0, 'sd_b': 0.0,
             'refuted_by': 'an arm with across-seed variation'}
    c = claim('the equivalence verdict is DEGENERATE', 'equivalence',
              {**degen, 'verdict': 'DEGENERATE'})
    check('a DEGENERATE verdict is refused as degenerate, not as unknown',
          not c.accepted
          and any(g == 'degenerate-interval' for g, _w, _y in c.refusals)
          and not any(g == 'equivalence-verdict' for g, _w, _y in c.refusals),
          str(c.refusals))
    c = claim('the two arms are equivalent', 'equivalence',
              {**degen, 'verdict': 'EQUIVALENT'})
    check('a zero-width interval is not an equivalence', not c.accepted,
          c.text)
    c = claim('For cell A on final_score', 'exclusion',
              {**degen, 'verdict': 'DEGENERATE'})
    check('no exclusion bound off a zero-width interval',
          not c.accepted and 'excluded at 95%' not in c.text, c.text[-120:])
    c = claim('For cell A on final_score', 'exclusion',
              {**good, 'exclusion_bound': None})
    check('a withheld exclusion bound is not re-derived here',
          not c.accepted
          and any(g == 'exclusion-bound' for g, _w, _y in c.refusals),
          str(c.refusals))
    c = claim('', 'associational',
              {'estimand': 'between_cell_scratch', 'n': 10, 'theta': 1.0,
               'ci_lo': 1.0, 'ci_hi': 1.0, 'name_a': 'A', 'name_b': 'B',
               'confirmatory': False, 'p_holm': None,
               'refuted_by': 'an interval covering 0.5'})
    check('a zero-width theta interval is not decisive',
          not c.accepted and 'not a sampling artefact' not in c.text,
          c.text[-140:])
    c = claim('', 'dispersion',
              {'estimand': 'dispersion', 'n': 10, 'sd_a': 0.20, 'sd_b': 0.05,
               'ci_lo': 4.0, 'ci_hi': 4.0, 'name_a': 'A', 'name_b': 'B',
               'refuted_by': 'a wider interval'})
    check('a zero-width SD-ratio interval is not decisive',
          not c.accepted and 'lies wholly above' not in c.text,
          c.text[-140:])

    # The INTERVAL_KINDS escape hatch: an exclusion bound in score units may
    # not be asserted over a proportion, an SD ratio or a mechanism signal.
    for bad in ('censoring_proportion', 'dispersion', 'mechanism_signal',
                'between_cell_scratch', 'arm_descriptive',
                'secondary_endpoint'):
        c = claim('the degradation is bounded', 'exclusion',
                  {**good, 'estimand': bad})
        check(f'exclusion refused over {bad}', not c.accepted, c.text)
    c = claim('the degradation is bounded', 'exclusion',
              {**good, 'estimand': 'control_contrast'})
    check('exclusion accepted over a score-scale contrast', c.accepted,
          str(c.refusals))
    c = claim('the arms are equivalent', 'equivalence',
              {**good, 'estimand': 'control_contrast', 'verdict': 'EQUIVALENT'})
    check('equivalence refused off the within-cell delta', not c.accepted)

    # A guard that raises is a guard that loses the report rather than the
    # sentence. stats.py's JSON round-trips a bare NaN into n.
    for bad_n in (float('nan'), 'many', float('inf')):
        c = claim('the paired shift is negative', 'causal',
                  {**good, 'n': bad_n})
        check(f'non-count n={bad_n!r} refused rather than raised',
              not c.accepted)

    # Every accepted sentence carries its scope out of the bundle, whatever
    # its kind (DESIGN.md 2.1), and the pooled estimand carries its label.
    for kind, evidence in (
            ('causal', dict(good)),
            ('mechanism', {**good, 'estimand': 'mechanism_signal',
                           'signal': 'cka_drift'}),
            ('causal_component', {**good, 'estimand': 'control_contrast',
                                  'manipulated': 'C2-C0'}),
            ('exclusion', dict(good))):
        c = claim('the interval is below zero', kind, evidence)
        check(f'scope clause prefixed for {kind}',
              c.accepted and c.text.startswith(SCOPE_CLAUSE), c.text[:80])
    set_source_policy('pooled')
    c = claim('the paired shift is negative', 'causal', dict(good))
    check('pooled estimand labelled in the sentence',
          c.accepted and POOLED_SOURCE_LABEL in c.text, c.text[:120])
    set_source_policy('valid')
    c = claim('the paired shift is negative', 'causal', dict(good))
    check('primary estimand carries no pooled label',
          c.accepted and POOLED_SOURCE_LABEL not in c.text, c.text[:120])

    # The wording guards are guards on a vocabulary, so the vocabulary is
    # tested. Each of these walked past the first version of its pattern.
    assoc = {**good, 'estimand': 'between_cell_scratch', 'confirmatory': False,
             'p_holm': None}
    for sentence in ('the delta is nil',
                     'transfer had no impact on the score',
                     'the two cells perform identically',
                     'transfer is neutral',
                     'there is no evidence of a shift',
                     'the effect is absent',
                     'no meaningful difference was seen',
                     'transfer neither helps nor hurts'):
        c = claim(sentence, 'associational', assoc)
        check(f'null-affirming refused: {sentence!r}', not c.accepted)
    for sentence in ('the architecture produces the gap',
                     'the gap stems from the architecture',
                     'the architecture is the reason for the gap',
                     'the architecture gives rise to the gap',
                     'the architecture explains the gap',
                     'the architecture induces the gap',
                     'dueling makes the agent worse'):
        c = claim(sentence, 'associational', assoc)
        check(f'causal verb refused: {sentence!r}', not c.accepted)
    em = {**good, 'estimand': 'between_cell_delta', 'confirmatory': False,
          'p_holm': None}
    for sentence in ('mlp-double avoids the degradation; dueling-vanilla does '
                     'not',
                     'only dueling-vanilla suffers the degradation',
                     'dueling-vanilla degrades and mlp-double does not'):
        c = claim(sentence, 'effect_modification', em)
        check(f'verdict comparison refused: {sentence!r}', not c.accepted)
        c2 = claim(sentence, 'effect_modification',
                   {**em, 'between_cell_contrast': {'ci_lo': 0.1,
                                                    'ci_hi': 0.4}})
        check(f'verdict comparison permitted with a contrast: {sentence!r}',
              c2.accepted, str(c2.refusals))

    c = claim('the contrast is negative', 'causal_component',
              {**good, 'estimand': 'control_contrast'})
    check('component claim needs the manipulation named', not c.accepted)
    c = claim('the contrast is negative', 'causal_component',
              {**good, 'estimand': 'control_contrast', 'manipulated': 'C2-C0'})
    check('component claim accepted with it', c.accepted, str(c.refusals))

    c = claim('something', 'vibes', {'estimand': 'within_cell_delta', 'n': 10})
    check('unknown kind refused', not c.accepted)
    c = claim('something', 'causal', {'estimand': 'made_up', 'n': 10})
    check('unknown estimand refused', not c.accepted)

    # Every refusal must name a document section.
    for c in (claim('the arm shows positive transfer', 'causal', dict(good)),
              claim('this proves it', 'causal', dict(good))):
        for _guard, where, why in c.refusals:
            check('refusal names a section',
                  any(tok in where for tok in ('DESIGN.md', 'ANALYSIS_PLAN.md',
                                               'STANDING_INSTRUCTIONS',
                                               'report.py')),
                  where)

    check('5.5 signal set non-empty', len(DESIGN_5_5_SIGNALS) >= 13)
    check('every estimand has an inference type',
          all(v[0] in KINDS for v in ESTIMAND_INFERENCE.values()))

    print(f'report.py self-test: {len(failures)} failure(s)')
    for line in failures:
        print(f'  FAIL {line}')
    if not failures:
        print('  every guardrail refused its target sentence, and every '
              'licensed sentence was emitted.')
    return 1 if failures else 0


# ===========================================================================
# 9. CLI.
# ===========================================================================
def allocate_bundle(outdir: str, date: str) -> str:
    """A bundle directory that no previous invocation has written into.

    `DESIGN.md` 8.3, 9 ("stale artifacts"). The path used to be the date alone,
    created with `exist_ok=True`, so a second invocation on the same day merged
    into the first: a bundle built from `runs` with `--skip-plots` inherited 28
    figures drawn from `runs_demo`, hashed them into its own `MANIFEST.json`
    under the role "figure, caption or figure provenance", and printed
    "Figures were skipped, so this bundle carries no figure provenance records"
    in the same document. Every hash was correct and detected nothing, because
    hashes detect a changed file and not a foreign one. A date is not a key, so
    an occupied directory is stepped over rather than reused.
    """
    for index in range(1, 100):
        bundle = os.path.join(outdir, date if index == 1 else f'{date}.{index}')
        if os.path.exists(bundle) and not os.path.isdir(bundle):
            continue                  # a file is in the way, step over it too
        if not os.path.isdir(bundle) or not os.listdir(bundle):
            return bundle
    raise RuntimeError(f'99 bundles already exist for {date} under {outdir}')


def mark_incomplete(bundle: str, why: str) -> None:
    """Say on disk that this bundle is a partial one, until it is not.

    The docstring promises "not a partial bundle, not a draft", and a run that
    dies between the audit and `MANIFEST.json` used to leave a directory with
    `audit/`, `logs/` and `stats/` in it and nothing saying so. The marker is
    written before any stage runs and removed when the manifest lands.
    """
    with open(os.path.join(bundle, INCOMPLETE_MARKER), 'w',
              encoding='utf-8') as fh:
        fh.write('INCOMPLETE BUNDLE. report.py created this directory and did '
                 'not finish.\n')
        fh.write(f'{why}\n')
        fh.write('There is no MANIFEST.json, so nothing here is hashed and '
                 'nothing here may be cited.\n')


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out-root', default='runs',
                   help='run tree to audit and aggregate (default: runs)')
    p.add_argument('--outdir', default=os.path.join('paper', 'results'),
                   help='parent directory for the dated bundle '
                        '(default: paper/results)')
    p.add_argument('--experiments', nargs='*', default=None,
                   help='catalogue ids to report on, e.g. E1 E2 (default: '
                        'those with runs on disk)')
    p.add_argument('--skip-plots', action='store_true',
                   help='do not draw figures; the bundle then carries no '
                        'figure provenance and REPORT.md says so')
    p.add_argument('--allow-audit-failure', action='store_true',
                   help='produce the bundle over a FAILED audit. DESIGN.md 8.4 '
                        'permits this only with the override stamped into '
                        'every output, which is what this flag does')
    p.add_argument('--seeds', default=None,
                   help='seed set the runs were launched at, passed to '
                        'audit.py; reducing it is the STANDING_INSTRUCTIONS S8 '
                        'validation invocation and is recorded')
    p.add_argument('--overrides', nargs='*', default=None,
                   help='launch-level overrides that were in force, as '
                        'field=value, passed to audit.py')
    p.add_argument('--strict', action='store_true',
                   help='audit warnings count as errors (the setting a '
                        'confirmatory campaign should be audited under)')
    p.add_argument('--source-policy', choices=('valid', 'pooled'),
                   default='valid',
                   help="passed to stats.py. 'valid' is the primary estimand; "
                        "'pooled' is the pre-declared secondary of DESIGN.md "
                        "4.3 and is recorded as a deviation")
    p.add_argument('--format', default='pdf,png',
                   help='figure formats passed to plots.py (default pdf,png)')
    p.add_argument('--quiet', action='store_true',
                   help='do not echo stage output or per-claim interrogation '
                        'to stdout; REPORT.md still carries all of it')
    p.add_argument('--self-test', action='store_true',
                   help='assert that every guardrail refuses its target '
                        'sentence, and exit')
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return self_test()

    out_root = os.path.abspath(args.out_root)
    outdir = os.path.abspath(args.outdir)
    if not os.path.isdir(out_root):
        print(f'{WARN} no run tree at {out_root}. Launch runs with '
              f'`python experiments/sweep.py --experiments E1` first.')
        return 1
    exps = list(args.experiments) if args.experiments else None
    unknown = [e for e in (exps or []) if e not in registry.EXPERIMENTS]
    if unknown:
        print(f'{WARN} unknown experiment id(s) {unknown}. Known: '
              f'{", ".join(registry.EXPERIMENTS)}')
        return 2

    date = datetime.date.today().isoformat()
    bundle = allocate_bundle(outdir, date)
    set_source_policy(args.source_policy)
    stages: list[Stage] = []
    prov = provenance.snapshot(['experiments/report.py',
                                *(list(argv) if argv is not None
                                  else sys.argv[1:])])

    print('=' * 78)
    print(f'report.py  {out_root} -> {bundle}')
    if os.path.basename(bundle) != date:
        print(f'  {WARN} {os.path.join(outdir, date)} already holds a bundle. '
              f'A bundle directory is never reused, because merging two '
              f'invocations produces a MANIFEST.json that hashes another run '
              f"tree's figures as its own (DESIGN.md 8.3).")
    print(f'  experiments : {", ".join(exps) if exps else "those with runs"}')
    print(f'  source policy: {args.source_policy}'
          + (' (DESIGN.md 4.3 secondary estimand; every emitted sentence is '
             'labelled)' if args.source_policy == 'pooled' else ''))
    print(f'  plan hash   : '
          f'{(prov.get("plans") or {}).get("ANALYSIS_PLAN.md")}')
    print('=' * 78)

    # --- stage 1: the audit gate (DESIGN.md 8.4) --------------------------
    print('\n== audit (the gate) ==')
    overrides = audit_mod.parse_overrides(args.overrides)
    started = time.time()
    ok, audit_report = audit_mod.audit_ok(out_root, exps, seeds=args.seeds,
                                          strict=args.strict,
                                          overrides=overrides)
    audit_seconds = time.time() - started
    failing = [c['name'] for c in audit_report.get('checks', [])
               if c.get('status') == 'FAIL']
    print(f'   audit {"PASS" if ok else "FAIL"}: '
          f'{audit_report.get("errors")} error(s), '
          f'{audit_report.get("warnings")} warning(s), {audit_seconds:.1f}s')
    if failing:
        print(f'   failing checks: {", ".join(failing)}')
    if not ok and not args.allow_audit_failure:
        print()
        print('REFUSING TO PRODUCE A REPORT.')
        print('DESIGN.md 8.4: aggregation and reporting refuse to run on a '
              'failed audit unless')
        print('overridden, and the override is stamped into the output. '
              'Nothing has been written.')
        print('  Fix the failing checks, or re-run with '
              '--allow-audit-failure to produce a')
        print('  bundle in which every artifact carries the override stamp:')
        for name in failing:
            chk = next(c for c in audit_report['checks'] if c['name'] == name)
            print(f'    {name}: {chk["errors"]} error(s) -- {chk["why"]}')
            for finding in chk['findings'][:4]:
                if finding.get('level') == audit_mod.ERROR:
                    print(f'        {finding["code"]}: {finding["message"]}')
        print()
        print('  Full audit output: python experiments/audit.py --out-root '
              f'{args.out_root}'
              + (f' --experiments {" ".join(exps)}' if exps else '')
              + (f' --seeds {args.seeds}' if args.seeds else ''))
        return 1
    override = bool(not ok and args.allow_audit_failure)
    if override:
        print(f'   {OVERRIDE_STAMP}')

    os.makedirs(bundle, exist_ok=True)
    mark_incomplete(bundle, 'It was created for the audit output and the '
                            'pipeline stopped before the report was written.')
    for sub in ('data', 'stats', 'audit', 'tables', 'logs'):
        os.makedirs(os.path.join(bundle, sub), exist_ok=True)
    audit_txt = os.path.join(bundle, 'audit', 'audit.txt')
    with open(audit_txt, 'w', encoding='utf-8') as fh:
        fh.write(audit_mod.render(audit_report, verbose=False, notes=True))
        fh.write('\n')
    with open(os.path.join(bundle, 'audit', 'audit.json'), 'w',
              encoding='utf-8') as fh:
        json.dump(audit_report, fh, indent=2, sort_keys=False, default=str)
    stages.append(Stage(
        name='audit', command=['experiments/audit.py', '--out-root', out_root]
        + (['--experiments', *exps] if exps else [])
        + (['--seeds', args.seeds] if args.seeds else [])
        + (['--strict'] if args.strict else []),
        exit_code=0 if ok else 1, seconds=audit_seconds, log=audit_txt,
        ok=ok, note=(OVERRIDE_STAMP if override else '')))

    # --- stage 2: aggregate ----------------------------------------------
    agg_args = ['--out-root', out_root]
    if exps:
        agg_args += ['--experiments', ','.join(exps)]
    stage = run_stage('aggregate', 'aggregate.py', agg_args,
                      os.path.join(bundle, 'logs', 'aggregate.log'),
                      args.quiet)
    stages.append(stage)
    per_seed_src = os.path.join(out_root, 'per_seed.csv')
    curves_src = os.path.join(out_root, 'curves.csv')
    if not stage.ok or not os.path.isfile(per_seed_src):
        print(f'\n{WARN} aggregation produced no per-seed table at '
              f'{per_seed_src}; there is nothing to report on. See '
              f'{rel(stage.log, _REPO)}.')
        print(f'  the partial bundle at {rel(bundle, _REPO)} carries '
              f'{INCOMPLETE_MARKER} and no MANIFEST.json; nothing in it may be '
              f'cited, and the next invocation will not write into it.')
        return 1

    # --- stage 3: stats ---------------------------------------------------
    stats_json = os.path.join(bundle, 'stats', 'stats.json')
    stats_args = ['--per-seed', per_seed_src, '--json', stats_json,
                  '--source-policy', args.source_policy]
    if exps:
        stats_args += ['--experiments', *exps]
    if override:
        # stats.py runs the DESIGN.md 8.4 gate for itself and refuses a failed
        # audit unless told otherwise. The decision to proceed was already
        # taken here, by --allow-audit-failure, and it is already stamped into
        # every artifact this module writes, so it is propagated rather than
        # taken twice. Probed rather than assumed: if the sibling does not
        # advertise the flag, nothing is passed and its own refusal stands.
        stats_opts = probe_options('stats.py')
        for flag in ('--override-audit', '--allow-audit-failure'):
            if flag in stats_opts:
                stats_args.append(flag)
                break
        else:
            print('   stats.py advertises no audit-override flag; if it runs '
                  'its own audit gate it will refuse, and the report will '
                  'stop here rather than be written without an analysis.')
    stage = run_stage('stats', 'stats.py', stats_args,
                      os.path.join(bundle, 'stats', 'stats.stdout.txt'),
                      args.quiet)
    stages.append(stage)
    if not stage.ok or not os.path.isfile(stats_json):
        print(f'\n{WARN} stats.py did not produce {stats_json}; refusing to '
              f'write a report with no analysis in it. See '
              f'{rel(stage.log, _REPO)}.')
        print(f'  the partial bundle at {rel(bundle, _REPO)} carries '
              f'{INCOMPLETE_MARKER} and no MANIFEST.json; nothing in it may be '
              f'cited, and the next invocation will not write into it.')
        return 1
    with open(stats_json, 'r', encoding='utf-8') as fh:
        sj = json.load(fh)

    # --- stage 4: figures -------------------------------------------------
    figdir = os.path.join(bundle, 'figures')
    figures: list[str] = []
    if args.skip_plots:
        print('\n== plots ==\n   skipped (--skip-plots)')
        stages.append(Stage('plots', ['(skipped)'], None, 0.0, None, True,
                            'skipped by --skip-plots'))
    else:
        stage = run_stage('plots', 'plots.py',
                          ['--per-seed', per_seed_src, '--curves', curves_src,
                           '--outdir', figdir, '--format', args.format],
                          os.path.join(bundle, 'logs', 'plots.log'),
                          args.quiet)
        stages.append(stage)
        if os.path.isdir(figdir):
            figures = sorted(f for f in os.listdir(figdir)
                             if not f.endswith('.provenance.json'))
        if not stage.ok:
            stage.note = ('figures are estimation-only illustrations, so a '
                          'failure here does not block the report; it is '
                          'recorded instead')
            print(f'   {WARN} plots.py failed; the bundle carries the log and '
                  f'whatever figures were written before the failure.')

    # --- copies, so the bundle stands alone -------------------------------
    hashes: dict[str, Optional[str]] = {}
    for src, dest_name, key in (
            (per_seed_src, 'per_seed.csv', 'per_seed'),
            (curves_src, 'curves.csv', 'curves'),
            (os.path.join(out_root, 'per_seed.provenance.json'),
             'per_seed.provenance.json', 'per_seed_provenance')):
        if os.path.isfile(src):
            dest = os.path.join(bundle, 'data', dest_name)
            shutil.copy2(src, dest)
            hashes[key] = provenance.file_hash(dest)
    hashes['stats'] = provenance.file_hash(stats_json)

    # --- stage 5: tables --------------------------------------------------
    print('\n== tables ==')
    prov_base = {
        'tool': 'experiments/report.py',
        'command': 'python experiments/report.py '
                   + ' '.join(list(argv) if argv is not None else sys.argv[1:]),
        'argv': list(sys.argv), 'cwd': os.getcwd(),
        'inputs': {'per_seed_csv': per_seed_src,
                   'per_seed_sha': hashes.get('per_seed'),
                   'curves_csv': curves_src,
                   'curves_sha': hashes.get('curves'),
                   'stats_json': rel(stats_json, bundle),
                   'stats_json_sha': hashes.get('stats')},
        'git': prov.get('git'), 'plans': prov.get('plans'),
        'packages': prov.get('packages'),
        'audit_ok': ok, 'audit_override': override,
        'audit_override_stamp': OVERRIDE_STAMP if override else None,
        'audit_failing_checks': failing,
        'source_policy': args.source_policy,
        'bootstrap': {'n_boot': (sj.get('s1_provenance') or {}).get('n_boot'),
                      'seed': (sj.get('s1_provenance') or {}).get('boot_seed')},
    }
    specs = build_tables(sj)
    table_index = write_tables(bundle, specs, prov_base)
    print(f'   {len(table_index)} table(s) + provenance -> '
          f'{rel(os.path.join(bundle, "tables"), _REPO)}')
    # A sibling module owns the LaTeX rendering of the paper's tables. Hand off
    # to it if it exists, using only the options it advertises, and record the
    # outcome rather than assuming it.
    if os.path.isfile(os.path.join(_HERE, 'tables.py')):
        opts = probe_options('tables.py')
        hand: list[str] = []
        # Aliases are grouped, and the first the sibling advertises wins. A
        # flat list appended every match, so a tables.py advertising both
        # --stats and --stats-json would have been handed the same path twice
        # under two names and asked to reconcile them.
        for aliases, value in (
                (('--per-seed',), per_seed_src),
                (('--stats', '--stats-json'), stats_json),
                (('--runs', '--out-root'), out_root),
                (('--outdir',), os.path.join(bundle, 'tables_latex'))):
            for flag in aliases:
                if flag in opts:
                    hand += [flag, value]
                    break
        if override:
            # As for stats.py: the DESIGN.md 8.4 decision was taken once, here,
            # and is stamped into everything this module writes. A sibling that
            # runs the gate again would otherwise refuse the hand-off and the
            # LaTeX tables would go missing with only a stage log to say why.
            for flag in ('--allow-audit-failure', '--override-audit'):
                if flag in opts:
                    hand.append(flag)
                    break
        if hand:
            stages.append(run_stage(
                'tables.py (LaTeX hand-off)', 'tables.py', hand,
                os.path.join(bundle, 'logs', 'tables.log'), args.quiet))
        else:
            print('   tables.py is present but advertises none of the options '
                  'this hand-off needs; the CSV tables above stand alone.')

    # --- REPORT.md --------------------------------------------------------
    envs_seen: list[str] = []
    try:
        with open(per_seed_src, 'r', encoding='utf-8', newline='') as fh:
            for row in csv.DictReader(fh):
                value = (row.get('env') or '').strip()
                if value and value not in envs_seen:
                    envs_seen.append(value)
    except OSError:
        pass
    s1 = sj.get('s1_provenance') or {}
    arm_ns = [_int_or_none(a.get('n')) or 0
              for a in (sj.get('s2_inventory') or {}).get('arms') or []]
    ctx = {
        'bundle': bundle, 'date': date,
        'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
        'out_root': out_root, 'experiments': exps,
        'source_policy': args.source_policy,
        'audit_report': audit_report, 'override': override,
        'stats': sj, 'provenance': prov, 'stages': stages,
        'hashes': hashes, 'per_seed_src': per_seed_src,
        'curves_src': curves_src,
        'per_seed_rows': csv_row_count(per_seed_src),
        'curves_rows': csv_row_count(curves_src),
        'table_index': table_index, 'figures': figures,
        'skipped_plots': bool(args.skip_plots),
        'envs_seen': envs_seen,
        'validation_stamp': bool((sj.get('s12_deviations') or {})
                                 .get('validation_stamp')
                                 or (arm_ns and min(arm_ns)
                                     < stats_mod.MIN_N_FOR_INFERENCE)),
        'plan_drift': bool(s1.get('exploratory')),
    }
    log = ClaimLog(echo=not args.quiet)
    print('\n== claims (each interrogated against STANDING_INSTRUCTIONS S5) ==')
    text = build_report(ctx, log)
    report_path = os.path.join(bundle, 'REPORT.md')
    with open(report_path, 'w', encoding='utf-8') as fh:
        fh.write(text)
    claims_path = write_claims_json(bundle, log)
    if override:
        with open(os.path.join(figdir if os.path.isdir(figdir) else bundle,
                               'AUDIT_OVERRIDE.txt'), 'w',
                  encoding='utf-8') as fh:
            fh.write(OVERRIDE_STAMP + '\n')
            fh.write(f'failing checks: {", ".join(failing)}\n')
            fh.write('Every figure, table and number in this bundle was '
                     'produced over a failed audit.\n')
    manifest_path = write_manifest(bundle, ctx, log)

    print(f'\n{len(log.accepted)} claim(s) emitted, {len(log.refused)} '
          f'refused, {len(log.with_p())} carrying a p-value.')
    print(f'REPORT.md   -> {rel(report_path, _REPO)}')
    print(f'claims.json -> {rel(claims_path, _REPO)}')
    print(f'MANIFEST.json -> {rel(manifest_path, _REPO)}')
    if override:
        print(f'\n{OVERRIDE_STAMP}')
        print(f'  failing checks: {", ".join(failing)}')
    if ctx['validation_stamp']:
        print(f'\n{VALIDATION_STAMP}: an arm has n < '
              f'{stats_mod.MIN_N_FOR_INFERENCE}. No number in this bundle may '
              f'be quoted, compared, or used to choose between hypotheses '
              f'(ANALYSIS_PLAN.md 9, STANDING_INSTRUCTIONS S8).')
    return 0


if __name__ == '__main__':
    sys.exit(main())
