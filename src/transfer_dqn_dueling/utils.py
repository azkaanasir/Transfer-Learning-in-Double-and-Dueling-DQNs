# utils.py
"""Utility helpers: GPU setup, persistence, gif saving and evaluation."""

import os
import json
from typing import List, Tuple, Optional

import imageio
import numpy as np
import gymnasium as gym

# NOTE: import tensorflow inside functions to avoid early-side-effects at import time
def set_gpu_growth():
    """Attempt to enable TF GPU memory growth (safe on CPU-only machines)."""
    try:
        import tensorflow as tf
    except Exception:
        return
    try:
        gpus = tf.config.list_physical_devices('GPU')
        for g in gpus:
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except Exception:
                pass
    except Exception:
        pass


def load_training_state(path: str) -> Tuple[List[float], List[float], int, Optional[float]]:
    """
    Load training metadata saved by save_training_state.
    Returns: (all_rewards, eval_scores, ep, epsilon)
    If file missing or malformed returns ([], [], 0, None)
    """
    if not os.path.exists(path):
        return [], [], 0, None
    try:
        with open(path, 'r') as f:
            s = json.load(f)
        all_rewards = s.get('all_rewards', [])
        eval_scores = s.get('eval_scores', [])
        ep = s.get('ep', 0)
        epsilon = s.get('epsilon', None)
        print(f"[utils] Resuming from episode {ep} with epsilon={epsilon}")
        return all_rewards, eval_scores, ep, epsilon
    except Exception as e:
        print(f"[utils] Failed to load training state from {path}: {e}")
        return [], [], 0, None


def save_training_state(path: str, all_rewards: List[float], eval_scores: List[float], ep: int, epsilon: float):
    """Save training metadata (best-effort)."""
    try:
        d = {
            'all_rewards': all_rewards,
            'eval_scores': eval_scores,
            'ep': ep,
            'epsilon': epsilon
        }
        # ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(d, f)
        print(f"[utils] Training state saved to {path} at episode {ep}")
    except Exception as e:
        print(f"[utils] Failed to save training state to {path}: {e}")


def save_gif(agent, filename: str, max_steps: int = 1000):
    """
    Record a run using env.render() (render_mode='rgb_array') and save as GIF.
    Agent is expected to have agent.select_action and agent.env_id attributes.
    This is best-effort and will quietly return if rendering not supported.
    """
    try:
        env = gym.make(agent.env_id, render_mode='rgb_array')
    except Exception as e:
        print(f"[utils] Cannot create env for gif: {e}")
        return

    try:
        state, _ = env.reset()
        frames = []
        done = False
        steps = 0
        while not done and steps < max_steps:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            frame = env.render()
            # some envs return tuple (frame, info) — handle conservatively:
            if isinstance(frame, (tuple, list)):
                frame = frame[0]
            frames.append(frame)
            state = next_state
            steps += 1
        env.close()
        if frames:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            imageio.mimsave(filename, frames, fps=30)
            print(f"[utils] Saved gif -> {filename}")
    except Exception as e:
        print(f"[utils] Failed to save gif: {e}")


def evaluate(agent, env, episodes: int = 3) -> float:
    """Run greedy evaluation for `episodes` episodes and return mean reward."""
    scores = []
    for _ in range(episodes):
        try:
            state, _ = env.reset()
        except Exception:
            # fallback if env.reset() returns only state
            state = env.reset()
        done = False
        total = 0.0
        while not done:
            q = agent.model.predict(state[None, :].astype(np.float32), verbose=0)[0]
            action = int(np.argmax(q))
            next_state, r, term, trunc, _ = env.step(action)
            total += r
            done = term or trunc
            state = next_state
        scores.append(total)
    if not scores:
        return 0.0
    return float(np.mean(scores))
