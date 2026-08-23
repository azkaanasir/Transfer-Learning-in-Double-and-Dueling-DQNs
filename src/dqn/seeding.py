"""Named, independent random streams, and per-layer deterministic initialisation.

Why this is not just `np.random.seed(seed)`
-------------------------------------------
Clean ablations require that changing one thing changes only that thing. With a
single shared generator, adding a diagnostic that draws a random batch, or
raising the number of evaluation episodes, shifts every subsequent draw and
therefore perturbs the training trajectory of arms that were supposed to be
untouched. The ablation then measures its own instrumentation. Splitting the
entropy into named streams removes that coupling by construction, and
`validate.py` tests it rather than trusting it.

Streams
-------
`init`         weight initialisation
`action`       epsilon-greedy exploration draws
`buffer`       replay sampling
`env_reset`    training episode reset seeds
`eval_monitor` periodic in-training evaluation episodes
`eval_final`   the held-out terminal evaluation -- deliberately disjoint from
               `eval_monitor`, so the primary endpoint is never measured on the
               initial states that were watched during training
`diag`         the fixed diagnostic state batch
`control`      the untrained/permuted source constructions, so building a
               control condition cannot perturb the run it is a control for

Determinism where it matters, and honesty where it does not
-----------------------------------------------------------
Episode and evaluation seeds are *derived by hash* from (run seed, role,
index) rather than drawn sequentially from a generator. That is what makes
resume exact: episode 700's reset seed does not depend on how many draws
happened before it, so a run interrupted at episode 650 and resumed produces the
same episodes as one that never stopped. A sequential generator cannot promise
that without also checkpointing its position, and it silently breaks whenever
anything upstream consumes a different number of draws.

Per-layer initialisation
------------------------
Weight initialisation is seeded per layer, from a hash of (run seed, layer
name). Two consequences, both deliberate:

* An `mlp` and a `dueling` network at the same run seed receive **identical**
  `trunk_fc1` and `trunk_fc2` initialisations. The architecture contrast is
  therefore not polluted by trunk initialisation noise, and the two
  `target_rule` levels are initialised identically throughout.
* Because conditions at a given seed share their initialisation, the design is
  *matched by seed*. That is what licenses the paired analysis in
  `ANALYSIS_PLAN.md` §2, where it buys roughly a 40% reduction in the minimum
  detectable effect -- the largest single power gain available at n=10.
"""
from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np

STREAMS = ('init', 'action', 'buffer', 'env_reset', 'eval_monitor',
           'eval_final', 'diag', 'control')

_MASK32 = 0xFFFFFFFF


def _digest_int(*parts: object) -> int:
    """A stable 32-bit integer from arbitrary labels.

    `hashlib`, not the built-in `hash`: Python's string hashing is randomised
    per interpreter process unless PYTHONHASHSEED is set, so a run's seeds would
    silently differ between invocations on the same machine.
    """
    payload = '\x1f'.join(str(p) for p in parts).encode('utf-8')
    return int.from_bytes(hashlib.blake2b(payload, digest_size=4).digest(),
                          'big') & _MASK32


class Seeds:
    """The full seed derivation for one run.

    Constructed from a single integer, which is the only thing an experiment
    definition has to vary. Everything else is derived and recorded.
    """

    def __init__(self, seed: int):
        self.seed = int(seed)
        root = np.random.SeedSequence(self.seed)
        children = root.spawn(len(STREAMS))
        self._ss = dict(zip(STREAMS, children))
        self._rngs: dict[str, np.random.Generator] = {}

    # ---- streams ---------------------------------------------------------
    def rng(self, stream: str) -> np.random.Generator:
        """The generator for one named stream, created once and reused."""
        if stream not in self._ss:
            raise KeyError(f'unknown stream {stream!r}; known: {STREAMS}')
        if stream not in self._rngs:
            self._rngs[stream] = np.random.Generator(
                np.random.PCG64(self._ss[stream]))
        return self._rngs[stream]

    def value(self, stream: str) -> int:
        """A recordable integer identifying a stream, for the manifest."""
        if stream not in self._ss:
            raise KeyError(f'unknown stream {stream!r}; known: {STREAMS}')
        return int(self._ss[stream].generate_state(1, dtype=np.uint32)[0])

    def report(self) -> dict:
        """Every derived seed, for the run manifest."""
        return {'run_seed': self.seed,
                'streams': {name: self.value(name) for name in STREAMS}}

    # ---- derived, index-addressable seeds --------------------------------
    def episode_seed(self, episode: int) -> int:
        """Reset seed for one training episode. Independent of run history."""
        return _digest_int(self.seed, 'env_reset', episode)

    def eval_seed(self, stream: str, checkpoint: int, index: int) -> int:
        """Reset seed for one evaluation episode.

        `stream` must be 'eval_monitor' or 'eval_final'. Because the label is
        part of the hash, the two evaluation families never share an initial
        state, so the held-out terminal evaluation is genuinely held out.
        """
        if stream not in ('eval_monitor', 'eval_final'):
            raise ValueError("stream must be 'eval_monitor' or 'eval_final', "
                             f'got {stream!r}')
        return _digest_int(self.seed, stream, checkpoint, index)

    def layer_seed(self, layer: str) -> int:
        """Initialisation seed for one named layer."""
        return _digest_int(self.seed, 'init', layer)

    def layer_seeds(self, layers: Iterable[str]) -> dict[str, int]:
        return {name: self.layer_seed(name) for name in layers}

    def control_seed(self, kind: str) -> int:
        """Seed for a control-condition construction (untrained/permuted source)."""
        return _digest_int(self.seed, 'control', kind)

    # ---- checkpointing ---------------------------------------------------
    def rng_states(self) -> dict:
        """Serialisable state of every instantiated stream.

        Only `action` and `buffer` advance during training, so only those two
        actually need restoring -- but capturing whatever exists keeps the
        checkpoint self-describing rather than relying on that remaining true.
        """
        return {name: _jsonable(rng.bit_generator.state)
                for name, rng in self._rngs.items()}

    def restore_rng_states(self, states: dict) -> list[str]:
        """Restore stream positions from a checkpoint. Returns names restored."""
        restored = []
        for name, state in (states or {}).items():
            if name not in self._ss:
                continue
            rng = self.rng(name)
            try:
                rng.bit_generator.state = _unjsonable(state)
                restored.append(name)
            except (ValueError, KeyError, TypeError):
                # A corrupt or version-mismatched state is reported rather than
                # silently ignored: a run that resumes with a fresh exploration
                # stream is not the run it claims to continue.
                raise ValueError(
                    f'cannot restore RNG state for stream {name!r}; the '
                    f'checkpoint was written by an incompatible numpy version. '
                    f'Delete the checkpoint and restart this run rather than '
                    f'continuing with a different random stream.')
        return restored


def _jsonable(state: dict) -> dict:
    """numpy bit-generator states hold ints wider than JSON handles portably."""
    out = {}
    for key, val in state.items():
        if isinstance(val, dict):
            out[key] = {k: (str(v) if isinstance(v, int) else v)
                        for k, v in val.items()}
        elif isinstance(val, (np.integer,)):
            out[key] = int(val)
        else:
            out[key] = val
    return out


def _unjsonable(state: dict) -> dict:
    out = {}
    for key, val in state.items():
        if isinstance(val, dict):
            out[key] = {k: (int(v) if isinstance(v, str) and v.lstrip('-').isdigit()
                            else v)
                        for k, v in val.items()}
        else:
            out[key] = val
    return out


def seed_frameworks(seed: int) -> None:
    """Set global framework seeds.

    Every quantity this study depends on is drawn from an explicit stream or a
    per-layer initialiser, so this is defence in depth rather than the mechanism:
    it pins any library-internal randomness that would otherwise be unseeded.
    Calling it does not make the run depend on global state.
    """
    import random

    import tensorflow as tf
    random.seed(int(seed))
    np.random.seed(int(seed) % (2 ** 31 - 1))
    tf.random.set_seed(int(seed))


__all__ = ['Seeds', 'STREAMS', 'seed_frameworks']
