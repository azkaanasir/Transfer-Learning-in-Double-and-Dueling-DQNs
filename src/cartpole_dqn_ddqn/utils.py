"""Utility helpers: TensorBoard helpers, seeding, etc."""
import os
from typing import List

import numpy as np
import tensorflow as tf

from config import BASE_SAVE_PATH, TB_LOG_DIR


def plot_training_results(rewards: List[float], lengths: List[int], validations: List[float], losses: List[float]):
    """Deprecated: kept for backward compatibility. Prefer using TensorBoard logs in `runs/`.
    If you still want an image, you can generate it by reading logs with tensorboard or
    calling this function manually (it uses matplotlib which may not be available everywhere).
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available. Use TensorBoard to visualize training results.")
        return

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.axhline(y=195, color='r', linestyle='--', label='Solved Threshold')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    window_size = 10
    if len(rewards) >= window_size:
        smoothed_rewards = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')
        plt.plot(smoothed_rewards)
    else:
        plt.plot(rewards)
    plt.title(f'Smoothed Rewards (Window Size: {window_size})')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.axhline(y=195, color='r', linestyle='--', label='Solved Threshold')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    val_x = list(range(0, len(rewards), 10))[:len(validations)]
    plt.plot(val_x, validations)
    plt.title('Validation Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True)

    plt.tight_layout()
    os.makedirs(BASE_SAVE_PATH, exist_ok=True)
    plt.savefig(os.path.join(BASE_SAVE_PATH, 'dueling_dqn_cartpole_results.png'))
    plt.show()


def set_global_seed(seed: int = 0):
    import random
    import numpy as _np

    random.seed(seed)
    _np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass