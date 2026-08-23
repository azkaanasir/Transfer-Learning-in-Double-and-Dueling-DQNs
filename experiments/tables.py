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
  (`DESIGN.md` 8.3, 9 "stale artifacts").
* **A caption that contradicts its table.** The same failure `plots.py` guards
  for figures (C8). Captions here are *generated*: the seed count, the endpoint
  definition, the test, the normalisation and the provenance stamp are filled in
  from the data, so a caption cannot claim an n or an endpoint the table does
  not have.
* **A blank read as an omission.** An estimation-only row has no p-value *by
  design* -- `ANALYSIS_PLAN.md` 7 emits p-values inside exactly one family of
  eight -- so its p column carries an explicit em dash and a footnote saying so.
  A blank cell would read as a number the authors declined to report, which is
  the opposite of the claim being made.
* **A suppressed test rendered as a null result.** Where `stats.py` refused a
  confirmatory member (incomplete arm, ambiguous primary arm, an invariant that
  moved, n<3) the p columns carry the refusal, not a dash and not a number.
  Refusing and saying why is the whole point of the refusal.
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
  by the dispersion actually observed, and flags a cell as not powered on the
  plan's own reference of one score unit.
* **A single-seed number quoted as a result.** When any tabulated arm has
  n < 3 every caption carries the `PIPELINE VALIDATION - NOT A RESULT` stamp
  (`ANALYSIS_PLAN.md` 9, `STANDING_INSTRUCTIONS` S8).

Nothing here computes an inference. Tables 2, 3 and 6's estimates come from
`stats.py --json`; without `--stats` those tables are emitted with an explicit
refusal in place of their numbers rather than with numbers this module derived
on its own, because a second implementation of the plan is a second chance to
diverge from it.

Formatting is deliberately plain: booktabs rules, no siunitx, no S columns, no
resizebox, no colour, so the output compiles in a stock IEEE or LNCS template
with `\usepackage{booktabs}` and nothing else. Non-ASCII is mapped to LaTeX
commands rather than relying on `inputenc`.

    python experiments/tables.py --per-seed runs/per_seed.csv --outdir paper/tables
    python experiments/tables.py --per-seed runs/per_seed.csv --stats stats.json
    python experiments/tables.py --self-test
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from experiments import registry                                  # noqa: E402
from experiments import stats as statsmod                         # noqa: E402
from src.dqn import envs, provenance                              # noqa: E402
