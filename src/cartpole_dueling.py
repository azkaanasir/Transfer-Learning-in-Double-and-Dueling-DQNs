#list down dependencies
import os
import json
import pickle
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from collections import deque
import matplotlib.pyplot as plt

# Define save path for Dueling DQN
DUELING_SAVE_PATH = './models/dueling_dqn_cartpole/'
os.makedirs(DUELING_SAVE_PATH, exist_ok=True)

#step 2  - replay action buffer
class DuelingCombineLayer(layers.Layer):
    def call(self, inputs):
        value, advantage = inputs
        return value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )

    def clear(self):
        self.buffer.clear()

    def load_buffer(self, experiences):
        self.clear()
        for exp in experiences:
            self.buffer.append(exp)

    def __len__(self):
        return len(self.buffer)
    

#dueling class
class DuelingDQNAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        buffer_size=10000,
        batch_size=64,
        target_update_freq=10
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.update_counter = 0

        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_network()

        self.replay_buffer = ReplayBuffer(buffer_size)
        self.loss_history = []

    def build_model(self):
        inputs = layers.Input(shape=(self.state_dim,))
        x = layers.Dense(128, activation='relu')(inputs)
        x = layers.Dense(128, activation='relu')(x)

        value = layers.Dense(64, activation='relu')(x)
        value = layers.Dense(1)(value)

        advantage = layers.Dense(64, activation='relu')(x)
        advantage = layers.Dense(self.action_dim)(advantage)

        outputs = DuelingCombineLayer()([value, advantage])

        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')
        return model

    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        state = np.reshape(state, [1, self.state_dim])
        q_values = self.model.predict(state, verbose=0)[0]
        return np.argmax(q_values)

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return 0

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        next_q_values = self.target_model.predict(next_states, verbose=0)
        max_next_q = np.max(next_q_values, axis=1)
        target_q_values = rewards + (1 - dones) * self.gamma * max_next_q

        current_q = self.model.predict(states, verbose=0)
        batch_indices = np.arange(len(states))
        current_q[batch_indices, actions] = target_q_values

        history = self.model.fit(states, current_q, verbose=0, batch_size=self.batch_size)
        loss = history.history['loss'][0]
        self.loss_history.append(loss)

        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.update_target_network()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss

    def save(self, filepath):
        self.model.save(filepath)

    def load(self, filepath):
        self.model = keras.models.load_model(filepath)
        self.update_target_network()

    def save_checkpoint(self, episode, episode_rewards, episode_lengths, validation_rewards):
        os.makedirs(DUELING_SAVE_PATH, exist_ok=True)

        self.model.save(f"{DUELING_SAVE_PATH}dueling_model_latest.h5")
        self.target_model.save(f"{DUELING_SAVE_PATH}dueling_target_model_latest.h5")

        with open(f"{DUELING_SAVE_PATH}replay_buffer_latest.pkl", 'wb') as f:
            pickle.dump(list(self.replay_buffer.buffer), f)

        metadata = {
            'epsilon': self.epsilon,
            'episode': episode,
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'validation_rewards': validation_rewards,
            'loss_history': self.loss_history,
            'update_counter': self.update_counter,
            'timestamp': time.time()
        }

        with open(f"{DUELING_SAVE_PATH}training_metadata_latest.json", 'w') as f:
            json.dump(metadata, f)

        print(f"Checkpoint saved at episode {episode}")

    def load_checkpoint(self):
        try:
            self.model = keras.models.load_model(f"{DUELING_SAVE_PATH}dueling_model_latest.h5")
            self.target_model = keras.models.load_model(f"{DUELING_SAVE_PATH}dueling_target_model_latest.h5")

            with open(f"{DUELING_SAVE_PATH}replay_buffer_latest.pkl", 'rb') as f:
                buffer_data = pickle.load(f)
                self.replay_buffer.load_buffer(buffer_data)

            with open(f"{DUELING_SAVE_PATH}training_metadata_latest.json", 'r') as f:
                metadata = json.load(f)

            self.epsilon = metadata['epsilon']
            self.update_counter = metadata.get('update_counter', 0)
            self.loss_history = metadata.get('loss_history', [])

            episode = metadata['episode']
            episode_rewards = metadata['episode_rewards']
            episode_lengths = metadata['episode_lengths']
            validation_rewards = metadata['validation_rewards']

            print(f"Loaded checkpoint from episode {episode}")
            return True, episode + 1, episode_rewards, episode_lengths, validation_rewards

        except (FileNotFoundError, OSError) as e:
            print(f"No checkpoint found or error loading checkpoint: {e}")
            print("Starting training from scratch.")
            return False, 0, [], [], []

def validate_agent(env, agent, num_episodes=10):
    total_rewards = []

    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # Use greedy policy for validation
            state_tensor = np.reshape(state, [1, agent.state_dim])
            q_values = agent.model.predict(state_tensor, verbose=0)[0]
            action = np.argmax(q_values)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state = next_state
            episode_reward += reward

        total_rewards.append(episode_reward)

    return np.mean(total_rewards)

def train_agent(env, agent, num_episodes=200, max_steps=500, render_every=50, early_stop=True):
    episode_rewards = []
    episode_lengths = []
    validation_rewards = []
    solved_threshold = 195
    consecutive_solves = 0
    required_solves = 5

    # load checkpoint using the agent's method
    checkpoint_loaded, start_episode, ep_rewards, ep_lengths, val_rewards = agent.load_checkpoint()

    if checkpoint_loaded:
        print(f"Resuming from episode {start_episode}")
        episode_rewards = ep_rewards
        episode_lengths = ep_lengths
        validation_rewards = val_rewards
    else:
        start_episode = 0

    for episode in range(start_episode, num_episodes):
        state, _ = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.replay_buffer.add(state, action, reward, next_state, done)
            loss = agent.train()

            state = next_state
            episode_reward += reward

            if done:
                break

        episode_rewards.append(episode_reward)
        episode_lengths.append(step + 1)

        if episode % 10 == 0:
            original_epsilon = agent.epsilon
            agent.epsilon = 0.0
            val_reward = validate_agent(env, agent, num_episodes=5)
            agent.epsilon = original_epsilon

            validation_rewards.append(val_reward)
            print(f"Episode {episode}/{num_episodes}, Reward: {episode_reward}, Validation: {val_reward}, Epsilon: {agent.epsilon:.3f}")

            if early_stop and val_reward >= solved_threshold:
                consecutive_solves += 1
                if consecutive_solves >= required_solves:
                    print(f"Environment solved in {episode} episodes! Avg reward: {val_reward:.2f}")
                    agent.save_checkpoint(episode, episode_rewards, episode_lengths, validation_rewards)
                    break
            else:
                consecutive_solves = 0
        else:
            print(f"Episode {episode}/{num_episodes}, Reward: {episode_reward}, Epsilon: {agent.epsilon:.3f}")

        # HPC-safe: Disable actual recording
        if render_every and episode % render_every == 0:
            record_video(env, agent, f"dueling_cartpole_episode_{episode}.mp4")

        if episode % 10 == 9:
            agent.save_checkpoint(episode, episode_rewards, episode_lengths, validation_rewards)

    return episode_rewards, episode_lengths, validation_rewards, agent.loss_history
def record_video(env, agent, video_path):
    print(f"[HPC] Skipping video recording cuz issues. Would have saved to: {video_path}")

def plot_training_results(rewards, lengths, validations, losses):
    plt.figure(figsize=(15, 10))

    # Plot rewards
    plt.subplot(2, 2, 1)
    plt.plot(rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.axhline(y=195, color='r', linestyle='--', label='Solved Threshold')
    plt.legend()
    plt.grid(True)

    # Plot smoothed rewards
    plt.subplot(2, 2, 2)
    window_size = 10
    smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    plt.plot(smoothed_rewards)
    plt.title(f'Smoothed Rewards (Window Size: {window_size})')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.axhline(y=195, color='r', linestyle='--', label='Solved Threshold')
    plt.legend()
    plt.grid(True)

    # Plot validation rewards
    plt.subplot(2, 2, 3)
    plt.plot(range(0, len(rewards), 10)[:len(validations)], validations)
    plt.title('Validation Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.grid(True)

    # Plot losses
    plt.subplot(2, 2, 4)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.yscale('log')  # Log scale for better visualization
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"{DUELING_SAVE_PATH}dueling_dqn_cartpole_results.png")
    plt.show()

if __name__ == "__main__":
    import gymnasium as gym

    # Create environment
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Initialize agent
    dueling_dqn_agent = DuelingDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        buffer_size=10000,
        batch_size=64,
        target_update_freq=10
    )

    print("Training Dueling DQN Agent on CartPole...")

    # Train agent
    rewards, lengths, validations, losses = train_agent(
        env=env,
        agent=dueling_dqn_agent,
        num_episodes=300,
        max_steps=500,
        render_every=0,  # Disable video recording
        early_stop=True
    )

    # Plot results
    plot_training_results(rewards, lengths, validations, losses)

    # Final evaluation
    print("Final Evaluation:")
    original_epsilon = dueling_dqn_agent.epsilon
    dueling_dqn_agent.epsilon = 0.0  # Greedy for eval
    final_reward = validate_agent(env, dueling_dqn_agent, num_episodes=20)
    dueling_dqn_agent.epsilon = original_epsilon
    print(f"Dueling DQN average reward: {final_reward:.2f}")

    # Save model
    final_model_path = f"{DUELING_SAVE_PATH}dueling_dqn_cartpole_final.h5"
    dueling_dqn_agent.save(final_model_path)
    print(f"Model saved to {final_model_path}")

    # Final checkpoint
    dueling_dqn_agent.save_checkpoint(
        episode=len(rewards)-1,
        episode_rewards=rewards,
        episode_lengths=lengths,
        validation_rewards=validations
    )
    print("Final checkpoint saved.")

    env.close()
