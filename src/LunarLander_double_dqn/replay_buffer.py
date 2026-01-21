from collections import deque
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, s, a, r, ns, d):
        self.buffer.append((s, a, r, ns, d))

    def sample(self, batch_size):
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        s, a, r, ns, d = zip(*[self.buffer[i] for i in idx])
        return (
            np.array(s),
            np.array(a),
            np.array(r, dtype=np.float32),
            np.array(ns),
            np.array(d, dtype=np.float32)
        )

    def load_buffer(self, data):
        self.buffer.clear()
        for exp in data:
            self.buffer.append(exp)

    def __len__(self):
        return len(self.buffer)
