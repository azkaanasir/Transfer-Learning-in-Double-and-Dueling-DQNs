"""Simple FIFO ReplayBuffer implementation."""
from collections import deque
from typing import Deque, List, Tuple, Any

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=capacity)

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        if batch_size > len(self.buffer):
            raise ValueError("batch_size larger than current buffer length")
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])
        return (
            np.array(states),
            np.array(actions, dtype=np.int32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )

    def clear(self):
        self.buffer.clear()

    def load_buffer(self, experiences: List[Tuple[Any, ...]]):
        self.clear()
        for exp in experiences:
            self.buffer.append(exp)

    def __len__(self):
        return len(self.buffer)