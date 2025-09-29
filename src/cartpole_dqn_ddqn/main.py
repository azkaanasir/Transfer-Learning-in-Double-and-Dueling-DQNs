import argparse
import os

# Import TensorFlow early so we can configure visible devices before other
# modules import tensorflow/keras. This helps avoid RuntimeError when calling
# set_visible_devices after GPUs have been initialized.
import tensorflow as tf

# GPU discovery & basic configuration --------------------------------------
# List all physical devices
physical_devices = tf.config.list_physical_devices()
print("All Physical Devices:")
for device in physical_devices:
    print(device)

# List all GPUs
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPUs available:")
    for gpu in gpus:
        print(gpu)

    try:
        # Set TensorFlow to use the first GPU (you can change the index if needed)
        tf.config.set_visible_devices(gpus[0], 'GPU')
        # Enable memory growth to avoid TF allocating all GPU memory at once
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("TensorFlow is set to use the first GPU.")
    except RuntimeError as e:
        # Visible devices must be set before GPUs are initialized
        print("Error setting GPU device:", e)
else:
    print("No GPU found. Using CPU.")

# Other imports (done after configuring TF)
import gymnasium as gym

from agent import DuelingDQNAgent
from config import cartpole_params
from train import train_agent, validate_agent
from utils import set_global_seed

def parse_args():
    parser = argparse.ArgumentParser(description='Train Dueling DQN on CartPole-v1')
    parser.add_argument('--episodes', type=int, default=300)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--no-early-stop', dest='early_stop', action='store_false')
    parser.add_argument('--render-every', type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    set_global_seed(args.seed)

    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DuelingDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        gamma=cartpole_params['gamma'],
        epsilon=cartpole_params['epsilon_start'],
        epsilon_min=cartpole_params['epsilon_min'],
        epsilon_decay=cartpole_params['epsilon_decay'],
        buffer_size=cartpole_params['replay_buffer_size'],
        batch_size=cartpole_params['batch_size'],
        target_update_freq=cartpole_params['target_update_freq'],
        learning_rate=cartpole_params['lr'],
    )

    print("Training Dueling DQN Agent on CartPole...")

    rewards, lengths, validations, losses = train_agent(
        env=env,
        agent=agent,
        num_episodes=args.episodes,
        max_steps=500,
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
    print(f"Dueling DQN average reward: {final_reward:.2f}")

    # Optionally save final checkpoint (already done periodically in training)
    env.close()


if __name__ == '__main__':
    main()