"""Main runner for transfer: sets up LunarLander env and trains using transferred agent."""
import argparse
import os

import tensorflow as tf

# GPU discovery & basic configuration (same style as your original main.py)
physical_devices = tf.config.list_physical_devices()
print("All Physical Devices:")
for device in physical_devices:
    print(device)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPUs available:")
    for gpu in gpus:
        print(gpu)
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("TensorFlow is set to use the first GPU.")
    except RuntimeError as e:
        print("Error setting GPU device:", e)
else:
    print("No GPU found. Using CPU.")

import gymnasium as gym
from agent import TransferDuelingDQNAgent
from config import transfer_params, TRANSFER_OPTIONS
from train import train_agent, validate_agent  # reuse your existing train.py (compatible)
from utils import set_global_seed

def parse_args():
    parser = argparse.ArgumentParser(description='Transfer Dueling DQN from CartPole -> LunarLander')
    parser.add_argument('--episodes', type=int, default=300)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--no-early-stop', dest='early_stop', action='store_false')
    parser.add_argument('--render-every', type=int, default=0)
    return parser.parse_args()

def main():
    args = parse_args()
    set_global_seed(args.seed)

    env = gym.make(transfer_params['env_id'])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = TransferDuelingDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        gamma=transfer_params['gamma'],
        epsilon=transfer_params['epsilon_start'],
        epsilon_min=transfer_params['epsilon_min'],
        epsilon_decay=transfer_params['epsilon_decay'],
        buffer_size=transfer_params['replay_buffer_size'],
        batch_size=transfer_params['batch_size'],
        target_update_freq=transfer_params['target_update_freq'],
        learning_rate=transfer_params['lr'],
        transfer_options=TRANSFER_OPTIONS,
    )

    print("Training Dueling DQN Agent (transferred weights) on LunarLander...")

    rewards, lengths, validations, losses = train_agent(
        env=env,
        agent=agent,
        num_episodes=args.episodes,
        max_steps=1000,
        render_every=args.render_every,
        early_stop=args.early_stop,
    )

    print("Final Evaluation:")
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    final_reward = 0.0
    try:
        final_reward = validate_agent(env, agent, num_episodes=20)
    except Exception:
        print("Validation failed — environment may not be available in this environment.")
    agent.epsilon = original_epsilon
    print(f"Dueling DQN (transfer) average reward: {final_reward:.2f}")

    env.close()

if __name__ == '__main__':
    main()
