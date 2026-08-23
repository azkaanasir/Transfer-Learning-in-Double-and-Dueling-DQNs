"""Fixed-capacity uniform replay buffer backed by preallocated arrays.

The published implementation stored transitions in a deque of tuples and rebuilt
numpy arrays on every sample. Preallocating removes that per-step allocation and
makes the buffer cheap to checkpoint, which matters because long runs get
interrupted often enough that resume has to be routine.

Two things differ from the published version, and both exist to keep ablations
clean:

* The generator is **injected**, not constructed here, so it comes from the
  run's named `buffer` stream (`seeding.Seeds`) rather than from a seed the
  buffer invents.
* `sample()` accepts an alternative generator. That is what lets a diagnostic
  draw a batch without advancing the training stream. The published loop drew
  its diagnostic batch from the training generator, so switching diagnostics on
  changed every subsequent minibatch and therefore the whole trajectory.
"""
from __future__ import annotations

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int,
                 rng: np.random.Generator | int = 0):
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.states = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int32)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.next_states = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0
        self.rng = (rng if isinstance(rng, np.random.Generator)
                    else np.random.default_rng(rng))

    def __len__(self) -> int:
        return self.size

    def add(self, state, action, reward, next_state, done) -> None:
        i = self.pos
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = float(done)
        self.pos = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int,
               rng: np.random.Generator | None = None):
        """Uniform sample. Pass `rng` to draw without touching the training stream."""
        gen = rng if rng is not None else self.rng
        idx = gen.integers(0, self.size, size=batch_size)
        return (self.states[idx], self.actions[idx], self.rewards[idx],
                self.next_states[idx], self.dones[idx])

    # ---- checkpointing ---------------------------------------------------
    def save(self, path: str) -> None:
        """Persist contents. The generator's position is *not* stored here.

        It belongs to the run's `buffer` stream and is checkpointed with the
        other stream states, so that there is exactly one place responsible for
        RNG position. The published version wrote a partial RNG state into the
        buffer archive that its own loader then ignored, which meant a resumed
        run silently sampled a different sequence.
        """
        np.savez_compressed(
            path,
            states=self.states[:self.size], actions=self.actions[:self.size],
            rewards=self.rewards[:self.size],
            next_states=self.next_states[:self.size],
            dones=self.dones[:self.size],
            pos=self.pos, size=self.size, capacity=self.capacity,
            state_dim=self.state_dim)

    def load(self, path: str) -> None:
        d = np.load(path)
        if int(d['state_dim']) != self.state_dim:
            raise ValueError(
                f'buffer checkpoint holds {int(d["state_dim"])}-dim states but '
                f'this run uses {self.state_dim}. Refusing to load.')
        if int(d['capacity']) != self.capacity:
            raise ValueError(
                f'buffer checkpoint capacity {int(d["capacity"])} != '
                f'{self.capacity}; the sampling distribution would differ.')
        n = int(d['size'])
        self.states[:n] = d['states']
        self.actions[:n] = d['actions']
        self.rewards[:n] = d['rewards']
        self.next_states[:n] = d['next_states']
        self.dones[:n] = d['dones']
        self.pos = int(d['pos'])
        self.size = n


__all__ = ['ReplayBuffer']
