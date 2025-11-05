"""Training loop, validation and TensorBoard logging utilities."""
from typing import List, Tuple, Optional
import datetime
import os
import time

import numpy as np
import tensorflow as tf

from agent import TransferDuelingDQNAgent  # keeps compatibility with original agent API
from config import TB_LOG_DIR as CONFIG_TB_LOG_DIR  # fallback if tb_log_dir not provided
# from utils import plot_training_results  # kept for backward compatibility if needed


def validate_agent(env, agent: TransferDuelingDQNAgent, num_episodes: int = 10) -> float:
    """Run deterministic evaluation (greedy policy) and return average episode reward."""
    total_rewards = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        done = False
        while not done:
            state_tensor = np.reshape(state, [1, agent.state_dim])
            q_values = agent.model.predict(state_tensor, verbose=0)[0]
            action = int(np.argmax(q_values))
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            episode_reward += float(reward)
        total_rewards.append(episode_reward)
    return float(np.mean(total_rewards)) if total_rewards else 0.0


def record_video(env, agent: TransferDuelingDQNAgent, video_path: str):
    # Placeholder for HPC-safe environments where rendering isn't available.
    print(f"[HPC] Skipping video recording. Would have saved to: {video_path}")


def _safe_timestamp_for_path() -> str:
    # Filesystem safe timestamp (no colons).
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def train_agent(
    env,
    agent: TransferDuelingDQNAgent,
    num_episodes: int = 200,
    max_steps: int = 500,
    render_every: int = 50,
    early_stop: bool = True,
    tb_log_dir: Optional[str] = None,
) -> Tuple[List[float], List[int], List[float], List[float]]:
    """
    Train loop compatible with both the original DuelingDQNAgent and the TransferDuelingDQNAgent.
    - tb_log_dir: optional override for TensorBoard logs; if None the module-level config TB_LOG_DIR is used.
    Returns: (episode_rewards, episode_lengths, validation_rewards, loss_history)
    """
    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    validation_rewards: List[float] = []

    # thresholds (kept locally so signature/behavior stays identical)
    solved_threshold = 195
    consecutive_solves = 0
    required_solves = 5

    # TensorBoard writer (allow override so transfer_main can pass its own TB path)
    base_logdir = tb_log_dir or CONFIG_TB_LOG_DIR
    logdir = os.path.join(base_logdir, _safe_timestamp_for_path())
    os.makedirs(logdir, exist_ok=True)
    writer = tf.summary.create_file_writer(logdir)

    # Try to resume training if checkpoint exists
    checkpoint_loaded, start_episode, ep_rewards, ep_lengths, val_rewards = agent.load_checkpoint()

    if checkpoint_loaded:
        print(f"Resuming from episode {start_episode}")
        episode_rewards = ep_rewards
        episode_lengths = ep_lengths
        validation_rewards = val_rewards
    else:
        start_episode = 0

    global_train_step = 0

    for episode in range(start_episode, num_episodes):
        state, _ = env.reset()
        episode_reward = 0.0

        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.replay_buffer.add(state, action, reward, next_state, done)
            loss = agent.train()

            # Log training loss per training step to TensorBoard
            if loss and loss != 0.0:
                with writer.as_default():
                    tf.summary.scalar('train/loss', loss, step=global_train_step)
                global_train_step += 1

            state = next_state
            episode_reward += float(reward)

            if done:
                break

        episode_rewards.append(episode_reward)
        episode_lengths.append(step + 1)

        # Log episode-level metrics (episode index is consistent whether resuming or fresh)
        with writer.as_default():
            tf.summary.scalar('episode/reward', episode_reward, step=episode)
            tf.summary.scalar('episode/length', step + 1, step=episode)
            tf.summary.scalar('agent/epsilon', agent.epsilon, step=episode)

        # Validation & logging every 10 episodes
        if episode % 10 == 0:
            # Epsilon decay applied on validation cadence (keeps behaviour from previous code)
            if agent.epsilon > agent.epsilon_min:
                agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

            val_reward = validate_agent(env, agent, num_episodes=5)
            validation_rewards.append(val_reward)

            print(f"Episode {episode}/{num_episodes}, Reward: {episode_reward:.2f}, Validation: {val_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

            # Log validation metric
            with writer.as_default():
                tf.summary.scalar('validation/avg_reward', val_reward, step=episode)

            # Early stopping condition (requires consecutive validation solves)
            if early_stop and val_reward >= solved_threshold:
                consecutive_solves += 1
                if consecutive_solves >= required_solves:
                    print(f"Environment solved in {episode} episodes! Avg validation reward: {val_reward:.2f}")
                    agent.save_checkpoint(episode, episode_rewards, episode_lengths, validation_rewards)
                    break
            else:
                consecutive_solves = 0
        else:
            print(f"Episode {episode}/{num_episodes}, Reward: {episode_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

        # Optional recording
        if render_every and episode % render_every == 0:
            record_video(env, agent, f"dueling_episode_{episode}.mp4")

        # Periodic checkpoint save (every 10 episodes; mirrors previous behaviour)
        if episode % 10 == 9:
            agent.save_checkpoint(episode, episode_rewards, episode_lengths, validation_rewards)

    # Final checkpoint/save at the end of training (if not saved by early stop)
    try:
        agent.save_checkpoint(episode, episode_rewards, episode_lengths, validation_rewards)
    except Exception:
        # Avoid crashing at the end if filesystem is readonly, etc.
        print("Warning: failed to save final checkpoint.")

    # Flush and close writer
    writer.flush()
    writer.close()

    return episode_rewards, episode_lengths, validation_rewards, agent.loss_history
