#step1 - dependencies
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import gymnasium as gym
import matplotlib.pyplot as plt
import random
from collections import deque
import os
import time
import json
import pickle
import datetime

   
# List all physical devices
physical_devices = tf.config.list_physical_devices()
print("All Physical Devices:")
for device in physical_devices:
    print(device)

# List all GPUs
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("\nGPUs available:")
    for gpu in gpus:
        print(gpu)
    
    try:
        # Set TensorFlow to use the first GPU (you can change the index if needed)
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("\nTensorFlow is set to use the first GPU.")
    except RuntimeError as e:
        print("\nError setting GPU device:", e)
else:
    print("\nNo GPU found. Using CPU.")


# Set a path for saving models and results locally
SAVE_PATH = './models/ddqn_cartpole/' 
os.makedirs(SAVE_PATH, exist_ok=True) 
print("Using save path:", SAVE_PATH) 

# Create log directory
log_dir = os.path.join("logs", "cartpole", "ddqn", datetime.datetime.now().strftime("%Y-%m-%d ---%H:%M:%S"))
train_log_dir = os.path.join(log_dir, "train")
test_log_dir = os.path.join(log_dir, "test")
train_writer = tf.summary.create_file_writer(train_log_dir)
test_writer = tf.summary.create_file_writer(test_log_dir)

#step2 - implement experience replay buffer 
class ReplayBuffer: 
    def __init__(self, capacity): 
        self.buffer = deque(maxlen=capacity) 

    def add(self, state, action, reward, next_state, done): 
        self.buffer.append((state, action, reward, next_state, done)) 

    def sample(self, batch_size): 
        # Allow replacement when buffer is smaller than batch size
        replace = len(self.buffer) < batch_size
        indices = np.random.choice(len(self.buffer), batch_size, replace=replace) 
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
        """Load experiences into buffer from a saved list""" 
        self.clear() 
        for exp in experiences: 
            self.buffer.append(exp) 

    def __len__(self): 
        return len(self.buffer) 

# DoubleDQNAgent Implementation 
class DoubleDQNAgent: 
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

        # Create main and target networks 
        self.model = self.build_model() 
        self.target_model = self.build_model() 
        self.update_target_network()  # Initial sync 

        # Create replay buffer 
        self.replay_buffer = ReplayBuffer(buffer_size) 

        # Metrics tracking 
        self.loss_history = [] 
        self.reward_history = []  # Added for tracking rewards 

    def build_model(self): 
        model = keras.Sequential([ 
            layers.Dense(128, activation='relu', input_shape=(self.state_dim,)), 
            layers.Dense(128, activation='relu'), 
            layers.Dense(self.action_dim, activation='linear') 
        ]) 

        model.compile( 
            optimizer=keras.optimizers.Adam(learning_rate=0.001), 
            loss='mse' 
        ) 

        return model 

    def update_target_network(self): 
        self.target_model.set_weights(self.model.get_weights()) 

    def select_action(self, state): 
        if np.random.rand() < self.epsilon: 
            # Exploration: select random action 
            return np.random.randint(self.action_dim) 
        else: 
            # Exploitation: select best action according to model 
            state = np.reshape(state, [1, self.state_dim]) 
            q_values = self.model.predict(state, verbose=0)[0] 
            return np.argmax(q_values) 

    def train(self): 
        # Check if we have enough samples 
        if len(self.replay_buffer) < self.batch_size: 
            return 0 

        # Sample a batch of experiences 
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size) 

        # Double DQN update 
        # 1. Select actions using online network 
        next_q_values_online = self.model.predict(next_states, verbose=0) 
        best_actions = np.argmax(next_q_values_online, axis=1) 

        # 2. Evaluate actions using target network 
        next_q_values_target = self.target_model.predict(next_states, verbose=0) 
        batch_indices = np.arange(len(next_states)) 
        next_q_values = next_q_values_target[batch_indices, best_actions] 

        # 3. Compute target Q values 
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values 

        # 4. Update online network 
        current_q = self.model.predict(states, verbose=0) 
        current_q[batch_indices, actions] = target_q_values 

        # Train the model 
        history = self.model.fit(states, current_q, verbose=0, batch_size=self.batch_size) 
        loss = history.history['loss'][0] 
        self.loss_history.append(loss) 

        # Update target network if needed 
        self.update_counter += 1 
        if self.update_counter % self.target_update_freq == 0: 
            self.update_target_network()  # Hard update 

        # Decay epsilon 
        if self.epsilon > self.epsilon_min: 
            self.epsilon *= self.epsilon_decay 

        return loss 

    def save(self, filepath): 
        self.model.save(filepath) 

    def load(self, filepath): 
        self.model = keras.models.load_model(filepath) 
        self.update_target_network() 

    # New methods for checkpointing 
    def save_checkpoint(self, episode, episode_rewards, episode_lengths, validation_rewards):
        # Create checkpoint directory if it doesn't exist
        os.makedirs(SAVE_PATH, exist_ok=True)

        # Save models
        self.model.save(f"{SAVE_PATH}ddqn_model_latest.h5")
        self.target_model.save(f"{SAVE_PATH}ddqn_target_model_latest.h5")

        # Save replay buffer
        with open(f"{SAVE_PATH}replay_buffer_latest.pkl", 'wb') as f:
            pickle.dump(list(self.replay_buffer.buffer), f)

        # Save training metadata
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

        with open(f"{SAVE_PATH}training_metadata_latest.json", 'w') as f:
            json.dump(metadata, f)

        print(f"Checkpoint saved at episode {episode}")

        # Also save milestone checkpoints
        if episode % 50 == 0:
            self.model.save(f"{SAVE_PATH}ddqn_model_ep{episode}.h5")
            self.target_model.save(f"{SAVE_PATH}ddqn_target_model_ep{episode}.h5")

            with open(f"{SAVE_PATH}training_metadata_ep{episode}.json", 'w') as f:
                json.dump(metadata, f)

    def load_checkpoint(self):
        try:
            # Load models
            self.model = keras.models.load_model(f"{SAVE_PATH}ddqn_model_latest.h5")
            self.target_model = keras.models.load_model(f"{SAVE_PATH}ddqn_target_model_latest.h5")

            # Load replay buffer
            with open(f"{SAVE_PATH}replay_buffer_latest.pkl", 'rb') as f:
                buffer_data = pickle.load(f)
                # Clear existing buffer and add loaded experiences
                self.replay_buffer.load_buffer(buffer_data)

            # Load training metadata
            with open(f"{SAVE_PATH}training_metadata_latest.json", 'r') as f:
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
    #step 4
def train_agent(env, agent, num_episodes=200, max_steps=500, early_stop=True):
    episode_rewards = []
    episode_lengths = []
    validation_rewards = []
    solved_threshold = 195
    consecutive_solves = 0
    required_solves = 3  # Number of consecutive validations above threshold

    # Try to load checkpoint using our agent's method
    # checkpoint_loaded, start_episode, ep_rewards, ep_lengths, val_rewards = agent.load_checkpoint()

    checkpoint_loaded = False

    if checkpoint_loaded:
        print(f"Resuming from episode {start_episode}")
        episode_rewards = ep_rewards
        episode_lengths = ep_lengths
        validation_rewards = val_rewards
    else:
        # Initialize from scratch if no checkpoint
        start_episode = 0
        episode_rewards = []
        episode_lengths = []
        validation_rewards = []

    for episode in range(start_episode, num_episodes):
        state, _ = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            # Select and perform action
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store experience in replay buffer
            agent.replay_buffer.add(state, action, reward, next_state, done)

            # Train the agent
            loss = agent.train()

            # Move to next state
            state = next_state
            episode_reward += reward

            if done:
                break

        # Record metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(step + 1)

        # Validate every 10 episodes
        if episode % 10 == 0:
            # Save current epsilon and force greedy policy for validation
            original_epsilon = agent.epsilon
            agent.epsilon = 0.0
            
            val_reward = validate_agent(env, agent, num_episodes=5)
            
            # Restore original epsilon
            agent.epsilon = original_epsilon
            
            validation_rewards.append(val_reward)
            print(f"Episode {episode}/{num_episodes}, Reward: {episode_reward}, Validation: {val_reward}, Epsilon: {agent.epsilon:.3f}")

            # Check for early stopping
            if early_stop and val_reward >= solved_threshold:
                consecutive_solves += 1
                if consecutive_solves >= required_solves:
                    print(f"Environment solved in {episode} episodes! Average reward: {val_reward:.2f}")
                    # Save final checkpoint
                    agent.save_checkpoint(episode, episode_rewards, episode_lengths, validation_rewards)
                    break
            else:
                consecutive_solves = 0
        else:
            print(f"Episode {episode}/{num_episodes}, Reward: {episode_reward}, Epsilon: {agent.epsilon:.3f}")

        # Save checkpoint every 10 episodes
        if episode % 10 == 9:  # Save at episodes 9, 19, 29, etc.
            agent.save_checkpoint(episode, episode_rewards, episode_lengths, validation_rewards)
            print(f"Checkpoint saved at episode {episode}")


        # Log metrics manually to TensorBoard
        with train_writer.as_default():
            tf.summary.scalar('rewards', episode_reward, step=episode)
            tf.summary.scalar('episode_lengths', episode_lengths[-1], step=episode)
            # print(loss)
            tf.summary.scalar('Loss', loss , step=episode)

        with test_writer.as_default():
            tf.summary.scalar('validation_rewards', validation_rewards[-1], step=episode)
            # tf.summary.scalar('4', test_accuracy.result(), step=episode)

        

    return episode_rewards, episode_lengths, validation_rewards, agent.loss_history

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

#step5 - visualisation
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

    # Save figure
    plt.savefig(f'{SAVE_PATH}double_dqn_cartpole_results.png')
    print(f"Results plot saved to {SAVE_PATH}double_dqn_cartpole_results.png")

    plt.show()

#step6 - main function
if __name__ == "__main__":
    # Environment setup
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]  # 4 for CartPole
    action_dim = env.action_space.n  # 2 for CartPole

    # Create Double DQN agent
    double_dqn_agent = DoubleDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        gamma=0.99,  # Discount factor
        epsilon=1.0,  # Start with full exploration
        epsilon_min=0.01,  # Minimum exploration rate
        epsilon_decay=0.995,  # Decay rate
        buffer_size=10000,  # Replay buffer size
        batch_size=64,  # Training batch size
        target_update_freq=10  # Update target network every 10 episodes
    )

    # Train the agent
    print("Training Double DQN Agent on CartPole...")
    rewards, lengths, validations, losses = train_agent(
        env=env,
        agent=double_dqn_agent,
        num_episodes=300,  # Train for 300 episodes
        max_steps=500,  # Maximum steps per episode
    )

    # Plot and save results
    plot_training_results(rewards, lengths, validations, losses)

    # Final evaluation
    print("Final Evaluation:")
    # Force greedy policy for final evaluation
    original_epsilon = double_dqn_agent.epsilon
    double_dqn_agent.epsilon = 0.0
    final_reward = validate_agent(env, double_dqn_agent, num_episodes=20)
    double_dqn_agent.epsilon = original_epsilon  # Restore epsilon
    print(f"Double DQN average reward: {final_reward:.2f}")

    # Save the model
    model_path = f"{SAVE_PATH}double_dqn_cartpole_final.h5"
    double_dqn_agent.model.save(model_path)
    print(f"Model saved to {model_path}")

    # Save checkpoint one last time
    double_dqn_agent.save_checkpoint(
        episode=len(rewards)-1,
        episode_rewards=rewards,
        episode_lengths=lengths,
        validation_rewards=validations
    )
    print("Final checkpoint saved")

    # Close the environment
    env.close()