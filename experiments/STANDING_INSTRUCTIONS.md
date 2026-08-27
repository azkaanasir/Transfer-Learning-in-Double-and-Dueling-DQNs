# Standing Instructions

Directives given by the user for this work, recorded verbatim in substance so
that later sessions do not drift from them. **Read this before touching
anything in `experiments/` or `src/dqn/`.** If a design decision appears to
conflict with an item here, the item wins; raise the conflict rather than
resolving it silently.

Companions: [`DESIGN.md`](DESIGN.md) (what is measured) ·
[`ANALYSIS_PLAN.md`](ANALYSIS_PLAN.md) (how it is tested) ·
[`EXPERIMENTS.md`](EXPERIMENTS.md) (how it is run)

---

## S1: Build the whole pipeline, end to end

Every experiment must be runnable, and analysable, from committed code in this
repository: launch → runs → aggregation → statistics → plots → tables. No step
is allowed to live in a notebook cell, a shell history, or a person's head. The
infrastructure must cover experiments on **different environments**, on the
**same environment**, and across **variants** of both.

*Applied in:* `DESIGN.md` §7 (catalogue), §8 (infrastructure requirements).

## S2: No compromise on scientific rigour

No logical, causal, or descriptive fallacy is acceptable: not in the
experimental design, not in the inference, not in the narrative. Specifically:
a null result is never evidence of equivalence, a between-group difference is
never a within-group effect, and a mechanism is never claimed from prose.

*Applied in:* `DESIGN.md` §9 (guardrails enforced in code), §2 (inference type
declared per research question and binding on wording).

## S3: Ablations and controls must be as clean as reproducibility allows

An ablation changes exactly one thing. Its counterfactual is explicit. The
control conditions must be strong enough that a positive finding cannot be
explained by protocol mechanics, weight scale, or initialisation luck.

*Applied in:* `DESIGN.md` §4 (the C0–C4 control set and the additive
decomposition), §8.1 (per-role RNG streams, so an ablation cannot perturb the
machinery it does not touch), §8.4 (machine-checked invariants).

## S4: Statistics: at least 10 seeds, and every test must be the right test

Ten seeds is the floor for confirmatory claims. Mann–Whitney U remains the
headline test: it is appropriate at this sample size and reviewers endorsed it
- but it is not sufficient on its own: effect sizes with confidence intervals,
dispersion tests, equivalence tests, multiplicity control, and a stated
minimum detectable effect are all required. Nothing decorative; every statistic
must answer a declared question.

*Applied in:* `ANALYSIS_PLAN.md` (whole document), `DESIGN.md` §5 (metric roles).

## S5: Keep asking Socratic questions; stress-test everything, continuously

Interrogate the logic, the clarity of the research questions, the methodology,
and the results: repeatedly and adversarially, including work already
finished. Assume something is wrong and go looking for it. The method is
Socratic: for every claim, ask what it would take to be false, what it is being
compared against, what else could produce the same observation, and what is
being assumed without being measured.

The four questions to put to every result before it is written down:

1. What is the counterfactual, and is it actually in the data?
2. What else could produce this number? Which control excludes that?
3. What would refute this? If nothing could, it is not a finding.
4. Is the wording's inference type the same as the design's? (See `DESIGN.md` §2.)

*Applied in:* adversarial review passes over the design and the implementation;
`validate.py` (self-tests, determinism, resume-equivalence); the open-questions
sections of `DESIGN.md` and `ANALYSIS_PLAN.md`; and `report.py`, which prints
these four questions against each emitted claim.

## S6: Hold to the highest standards of research discipline

Pre-register before running. Record provenance. Never drop a seed. Never
hand-compute a number that appears in the paper. Report what was skipped and
why.

*Applied in:* `DESIGN.md` §8.3–8.4, `ANALYSIS_PLAN.md` §1 (pre-registration
discipline and change-log requirement).

## S7: Ground the thesis in the literature, and name the gap precisely

Work through the literature the paper already cites, then sharpen the thesis
against it. The gap must be stated as a specific, checkable deficiency in
identified prior work: not as a generic "this is under-explored".

*Applied in:* `paper/LITERATURE.md` (per-source audit and the derived gap
statement), `DESIGN.md` §2 (research questions written to close that gap).

## S8: Execution right now: one seed, to validate the machinery

Until the infrastructure is complete and validated, **run every experiment at a
single seed**. The purpose of those runs is to exercise the pipeline end to end,
not to produce evidence. The full seed set comes later, on the user's word.

Two consequences, and they must not be confused with each other:

* **Design defaults stay at 10+ seeds** (S4). The registry, the analysis plan
  and the cost model are all written for the confirmatory sample size; only the
  invocation is reduced.
* **Nothing produced at n=1 is a result.** Single-seed output is labelled
  `PIPELINE VALIDATION` by `report.py`, confirmatory tests are suppressed rather
  than computed on n=1, and no such number may be quoted in the paper or used to
  choose between hypotheses. A single seed can show that a run *executes*; it
  cannot show that an arm *differs*.

*Applied in:* `EXPERIMENTS.md` (validation invocations), `plan.py` (`--seeds 0`
cost estimates), and the n<3 guard in `stats.py`.

## S9: The eight peer reviews are paramount

The reviews from IJCNN 2026 and ICANN 2026 are the specification for this
revision, not background material. Every experiment must trace to a concern
raised in them (or to a defect the Phase 0 audit found that no reviewer caught),
and every concern must trace to something that answers it. An experiment that
answers nothing gets cut; a concern with nothing pointing at it is a gap in the
plan, not an acceptable omission.

Note the asymmetry the review digest establishes: the most-cited concerns are
the ordinary scope complaints, while the three that actually threatened the
paper's validity were each raised by a single reviewer. **Prioritising by vote
count would be a mistake.**

*Applied in:* `REVIEW_COVERAGE.md`: the two-way traceability matrix, with
per-item status and an explicit open-items list.

## S10 - No em-dashes

Do not use em-dashes anywhere: not in chat, not in documentation, not in code
comments, not in the manuscript. Use a colon, a semicolon, a comma, parentheses,
or a full stop instead, whichever the sentence actually needs.

This applies retrospectively as well: when editing a file that already contains
them, remove them.

## S11 - Keep this file current

These instructions are the reference point when direction is unclear. Add new
directives here as they arrive, with the section they affected.

---

## Hard constraints

* **Pushing requires the user to ask for it, every time.** The standing rule
  was "never push, under any circumstances"; on 2026-08-28 the user
  instructed otherwise and this line records that, rather than leaving a
  binding instruction on file that is no longer true. Committing locally
  remains right and worth doing periodically so work is recoverable.
  Publishing is still not a default: it is done when asked and not
  otherwise.
* **`experiments/REVIEW_COVERAGE.md` is never published.** It maps the eight
  peer reviews (IJCNN R1 to R4, ICANN #2, #3, #5) to their specific
  complaints, which is confidential referee material for a submission still
  under review. It is gitignored and untracked as of 2026-08-28. `paper/`
  and `memory/` are ignored for the same reason.
* **Never name Claude as an author or contributor** in a commit message, a
  commit trailer, a PR body, or the manuscript. No `Co-Authored-By`, no
  "generated with" line. This overrides the default harness behaviour. The repo
  is an authored academic record with named human authors under peer review, and
  commit provenance is part of that record.
* **`paper/` and `memory/` are gitignored and untracked**, hold the only copy of
  the review analysis, and are not backed up. Do not move or delete them.
* **No archival submission anywhere** while ICANN 2026 submission 603 is under
  review.
