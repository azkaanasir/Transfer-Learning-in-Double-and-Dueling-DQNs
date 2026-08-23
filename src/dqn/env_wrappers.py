"""Wrappers that change an environment's *interface* without changing its dynamics.

These exist to supply the one corner of the shift factorial that nobody has run
(`DESIGN.md` §6.4): **same dynamics, changed interface**.

The published study's only pair, CartPole-v1 -> LunarLander-v3, changes the
observation dimension (4 -> 8) and the action cardinality (2 -> 4) at the same
time as the dynamics, the reward structure and the horizon. Every effect it
attributed to "domain shift" is therefore confounded with the partial-copy and
head-reinitialisation mechanics that the interface change *forces*. Separating
the two needs a pair where the dynamics are identical by construction and only
the interface moves, so that the identical protocol -- partial copy of the
input-facing layer, reinitialised output head, freeze window -- runs with no
dynamics shift at all to explain the result.

`PadObservation`
    Appends `k` extra observation dimensions. They are **uninformative but
    active**: filled with i.i.d. standard normal noise rather than zeros.
    Zeros would make the padding inert -- a linear layer maps a zero input to no
    contribution, so the freshly initialised rows of `trunk_fc1` would have no
    effect and the mechanics under test would not actually be exercised. Noise
    keeps the new input pathway live, which is the faithful analogue of the
    real case where the extra dimensions carry signal, while guaranteeing they
    carry nothing the optimal policy could use.

`DuplicateActions`
    Extends the discrete action set by `m` actions that alias existing ones
    (`a mod n`). The action cardinality the network's head must produce grows,
    while the set of achievable behaviours is exactly unchanged, so no dynamics
    or reachability difference can be confused for an interface effect.

Both wrappers are seeded from the reset seed, so a padded run is reproducible
and -- importantly -- two variants at the same seed share their underlying
episode.
"""
from __future__ import annotations

from typing import Any

import numpy as np

try:                                        # gymnasium is optional at import
    import gymnasium as gym
    _Wrapper = gym.Wrapper
    _ObsWrapper = gym.ObservationWrapper
except Exception:                            # pragma: no cover
    gym = None
    _Wrapper = object
    _ObsWrapper = object

PAD_MODES = ('noise', 'zeros')


class PadObservation(_ObsWrapper):
    """Append `pad` uninformative dimensions to a Box observation."""

    def __init__(self, env, pad: int, mode: str = 'noise', scale: float = 1.0):
        super().__init__(env)
        if pad < 1:
            raise ValueError(f'pad must be >= 1, got {pad}')
        if mode not in PAD_MODES:
            raise ValueError(f'pad mode must be one of {PAD_MODES}, got {mode!r}')
        self.pad = int(pad)
        self.mode = mode
        self.scale = float(scale)
        self._rng = np.random.default_rng(0)

        low = np.asarray(env.observation_space.low, dtype=np.float32)
        high = np.asarray(env.observation_space.high, dtype=np.float32)
        bound = np.inf if mode == 'noise' else 0.0
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([low, np.full(self.pad, -bound, dtype=np.float32)]),
            high=np.concatenate([high, np.full(self.pad, bound, dtype=np.float32)]),
            dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            # Offset so the padding noise is not the same stream the simulator
            # uses; the underlying episode stays identical to the unpadded env
            # at the same seed.
            self._rng = np.random.default_rng((int(seed) + 0x5F37) & 0xFFFFFFFF)
        return super().reset(seed=seed, options=options)

    def observation(self, observation):
        base = np.asarray(observation, dtype=np.float32)
        if self.mode == 'zeros':
            extra = np.zeros(self.pad, dtype=np.float32)
        else:
            extra = (self._rng.standard_normal(self.pad)
                     * self.scale).astype(np.float32)
        return np.concatenate([base, extra])


class DuplicateActions(_Wrapper):
    """Extend a Discrete action space by `extra` aliases of existing actions."""

    def __init__(self, env, extra: int):
        super().__init__(env)
        if extra < 1:
            raise ValueError(f'extra must be >= 1, got {extra}')
        self.base_n = int(env.action_space.n)
        self.extra = int(extra)
        self.action_space = gym.spaces.Discrete(self.base_n + self.extra)

    def action(self, action: int) -> int:
        return int(action) % self.base_n

    def step(self, action):
        return self.env.step(self.action(action))


def apply_interface_wrappers(env, pad_obs: int = 0, pad_mode: str = 'noise',
                             extra_actions: int = 0) -> tuple[Any, dict]:
    """Apply the interface-change wrappers and report what was applied."""
    applied: dict[str, Any] = {}
    if pad_obs:
        env = PadObservation(env, pad=int(pad_obs), mode=pad_mode)
        applied['pad_obs'] = int(pad_obs)
        applied['pad_mode'] = pad_mode
    if extra_actions:
        env = DuplicateActions(env, extra=int(extra_actions))
        applied['extra_actions'] = int(extra_actions)
    return env, applied


__all__ = ['PadObservation', 'DuplicateActions', 'apply_interface_wrappers',
           'PAD_MODES']
