"""Fixed-capacity uniform replay buffer backed by preallocated arrays.

The original implementation stored transitions in a deque of tuples and rebuilt
numpy arrays on every sample. Preallocating removes that per-step allocation and
makes the buffer cheap to checkpoint, which matters because Colab sessions are
interrupted often enough that resume has to be routine.
"""
from __future__ import annotations

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, seed: int = 0):
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.states = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int32)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.next_states = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

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

    def sample(self, batch_size: int):
        idx = self.rng.integers(0, self.size, size=batch_size)
        return (self.states[idx], self.actions[idx], self.rewards[idx],
                self.next_states[idx], self.dones[idx])

    # ---- checkpointing ---------------------------------------------------
    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            states=self.states[:self.size], actions=self.actions[:self.size],
            rewards=self.rewards[:self.size],
            next_states=self.next_states[:self.size],
            dones=self.dones[:self.size],
            pos=self.pos, size=self.size,
            rng_state=np.frombuffer(
                self.rng.bit_generator.state['state']['state'].to_bytes(16, 'little'),
                dtype=np.uint8),
        )

    def load(self, path: str) -> None:
        d = np.load(path)
        n = int(d['size'])
        self.states[:n] = d['states']
        self.actions[:n] = d['actions']
        self.rewards[:n] = d['rewards']
        self.next_states[:n] = d['next_states']
        self.dones[:n] = d['dones']
        self.pos = int(d['pos'])
        self.size = n
