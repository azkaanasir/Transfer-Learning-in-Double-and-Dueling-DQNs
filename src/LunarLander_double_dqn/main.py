import argparse
import os
import tensorflow as tf

# GPU configuration
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
from config import lunar_params
from agent import DoubleDQNAgent
from train import train_agent, validate_agent
from utils import set_global_seed

def parse_args():
    parser = argparse.ArgumentParser(description='Train Double DQN on LunarLander-v3')
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--no-early-stop', dest='early_stop', action='store_false')
    parser.add_argument('--render-every', type=int, default=0)
    return parser.parse_args()

def main():
    args = parse_args()
    set_global_seed(args.seed)
    
    env = gym.make('LunarLander-v3')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DoubleDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        gamma=lunar_params['gamma'],
        epsilon=lunar_params['epsilon_start'],
        epsilon_min=lunar_params['epsilon_min'],
        epsilon_decay=lunar_params['epsilon_decay'],
        buffer_size=lunar_params['replay_buffer_size'],
        batch_size=lunar_params['batch_size'],
        target_update_freq=lunar_params['target_update_freq'],
        learning_rate=lunar_params['lr']
    )
    
    print("Training Double DQN Agent on LunarLander...")
    rewards, lengths, validations, losses = train_agent(
        env=env,
        agent=agent,
        num_episodes=args.episodes,
        max_steps=500,
        early_stop=args.early_stop
    )
    
    print("Final Evaluation:")
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    final_reward = validate_agent(env, agent, episodes=20)
    agent.epsilon = original_epsilon
    print(f"Double DQN average reward: {final_reward:.2f}")
    
    env.close()

if __name__ == '__main__':
    main()