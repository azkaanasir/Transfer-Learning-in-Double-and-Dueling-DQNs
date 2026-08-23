"""An idempotent, keyed metrics log.

The published training loop appended to `metrics.csv` on resume with no
truncation. Reproduced by execution during review: a 45-episode run interrupted
at episode 39 whose last checkpoint was episode 30 produced a file with 54 rows
and episodes 31-39 recorded twice. Everything downstream then computed its
window statistics over duplicated episodes, and `episodes_completed` -- which
the sweep used to decide a run was finished -- was simply wrong.

A stream of rows is the wrong primitive for something that must survive
interruption. Here the log is **keyed by episode**: writing episode *k* replaces
any existing row for *k*, and on resume everything at or after the resume point
is dropped before appending. The file's episode set is asserted to be exactly
`range(0, n)` at the end of every run, and `aggregate.py` and `audit.py` assert
it again -- three independent checks, because silent duplication is invisible in
the aggregate and corrupts every window statistic.
"""
from __future__ import annotations

import json
import os
import tempfile
from typing import Iterable


class MetricsLog:
    """Append-only-by-episode JSONL log with idempotent rewrite on resume."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        self._fh = None

    # ---- reading ---------------------------------------------------------
    def read(self) -> list[dict]:
        if not os.path.exists(self.path):
            return []
        rows = []
        with open(self.path, encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    # A torn final line is the expected artifact of a kill
                    # mid-write. Dropping it is safe precisely because the log
                    # is keyed: the episode will be rewritten on resume.
                    continue
        return rows

    def episodes(self) -> set[int]:
        return {int(r['episode']) for r in self.read() if 'episode' in r}

    # ---- writing ---------------------------------------------------------
    def truncate_from(self, episode: int) -> int:
        """Drop every row with `episode >= episode`. Returns rows removed.

        Called on resume, before any append. Uses write-to-temp-and-replace so
        an interruption during the truncation itself cannot leave a partial log.
        """
        rows = self.read()
        keep = [r for r in rows if int(r.get('episode', -1)) < episode]
        removed = len(rows) - len(keep)
        if removed:
            self._rewrite(keep)
        return removed

    def _rewrite(self, rows: Iterable[dict]) -> None:
        self.close()
        directory = os.path.dirname(self.path) or '.'
        fd, tmp = tempfile.mkstemp(dir=directory, suffix='.tmp')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8', newline='') as fh:
                for row in rows:
                    fh.write(json.dumps(row, sort_keys=True) + '\n')
            os.replace(tmp, self.path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def append(self, row: dict) -> None:
        if self._fh is None:
            self._fh = open(self.path, 'a', encoding='utf-8', newline='')
        self._fh.write(json.dumps(row, sort_keys=True) + '\n')
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    # ---- integrity -------------------------------------------------------
    def check(self, expected: int | None = None) -> dict:
        """Assert the episode index set is exactly range(0, n).

        Returned rather than raised so a caller can record the verdict in the
        manifest; `audit.py` is what turns a failure into a refusal to report.
        """
        rows = self.read()
        eps = [int(r['episode']) for r in rows if 'episode' in r]
        uniq = sorted(set(eps))
        problems = []
        if len(eps) != len(uniq):
            dupes = sorted({e for e in eps if eps.count(e) > 1})
            problems.append(f'duplicate episodes: {dupes[:12]}'
                            f'{"..." if len(dupes) > 12 else ""}')
        if uniq and uniq != list(range(len(uniq))):
            missing = sorted(set(range(uniq[-1] + 1)) - set(uniq))
            problems.append(f'gaps in episode index: missing {missing[:12]}'
                            f'{"..." if len(missing) > 12 else ""}')
        if expected is not None and len(uniq) != expected:
            problems.append(f'expected {expected} episodes, found {len(uniq)}')
        return {'rows': len(eps), 'unique_episodes': len(uniq),
                'contiguous': not problems, 'problems': problems}

    def as_dataframe(self):
        import pandas as pd
        rows = self.read()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows).sort_values('episode').reset_index(drop=True)
        return df


class JsonlLog:
    """A plain append-only JSONL sink for records that are not episode-keyed.

    Used for per-evaluation-episode returns and for freeze events, so that both
    are inspectable at full resolution instead of only as summaries -- which is
    what makes the evaluation-noise floor estimable and the freeze verifiable.
    """

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

    def append(self, row: dict) -> None:
        with open(self.path, 'a', encoding='utf-8', newline='') as fh:
            fh.write(json.dumps(row, sort_keys=True) + '\n')

    def extend(self, rows: Iterable[dict]) -> None:
        with open(self.path, 'a', encoding='utf-8', newline='') as fh:
            for row in rows:
                fh.write(json.dumps(row, sort_keys=True) + '\n')

    def read(self) -> list[dict]:
        if not os.path.exists(self.path):
            return []
        out = []
        with open(self.path, encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return out

    def drop_where(self, predicate) -> int:
        """Remove records matching `predicate`, for idempotent resume."""
        rows = self.read()
        keep = [r for r in rows if not predicate(r)]
        removed = len(rows) - len(keep)
        if removed:
            directory = os.path.dirname(self.path) or '.'
            fd, tmp = tempfile.mkstemp(dir=directory, suffix='.tmp')
            with os.fdopen(fd, 'w', encoding='utf-8', newline='') as fh:
                for row in keep:
                    fh.write(json.dumps(row, sort_keys=True) + '\n')
            os.replace(tmp, self.path)
        return removed


__all__ = ['MetricsLog', 'JsonlLog']
