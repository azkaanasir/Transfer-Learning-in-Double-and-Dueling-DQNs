
# import numpy as np
# import gymnasium as gym
# import tensorflow as tf
# from tensorflow import keras
# import matplotlib.pyplot as plt
# from collections import deque
# import random
# # import cv2
# import os

# # GPU Memory Growth
# physical_devices = tf.config.list_physical_devices('GPU')
# for device in physical_devices:
#     tf.config.experimental.set_memory_growth(device, True)

# # Local directory paths
# base_path = './LunarLander-DDQN'
# checkpoint_path = os.path.join(base_path, 'checkpoints')
# video_path = os.path.join(base_path, 'videos')
# log_path = os.path.join(base_path, 'logs')

# os.makedirs(checkpoint_path, exist_ok=True)
# os.makedirs(video_path, exist_ok=True)
# os.makedirs(log_path, exist_ok=True)

# # Replay Buffer
# class ReplayBuffer:
#     def __init__(self, capacity):
#         self.buffer = deque(maxlen=capacity)

#     def add(self, state, action, reward, next_state, done):
#         self.buffer.append((state, action, reward, next_state, done))

#     def sample(self, batch_size):
#         batch = random.sample(self.buffer, batch_size)
#         states, actions, rewards, next_states, dones = zip(*batch)
#         return (np.stack(states), np.array(actions), np.array(rewards),
#                 np.stack(next_states), np.array(dones))

#     def size(self):
#         return len(self.buffer)

# # Double DQN Agent
# class LunarDoubleDQNAgent:
#     def __init__(self, state_dim, action_dim,
#                  gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
#                  buffer_size=50000, batch_size=128, target_update_freq=5):

#         self.state_dim = state_dim
#         self.action_dim = action_dim
#         self.gamma = gamma
#         self.epsilon = epsilon
#         self.epsilon_min = epsilon_min
#         self.epsilon_decay = epsilon_decay
#         self.batch_size = batch_size
#         self.target_update_freq = target_update_freq
#         self.update_counter = 0

#         self.model = self.build_model()
#         self.target_model = self.build_model()
#         self.update_target_network()

#         self.replay_buffer = ReplayBuffer(buffer_size)
#         self.loss_history = []
#         self.reward_history = []

#     def build_model(self):
#         inputs = keras.Input(shape=(self.state_dim,))
#         x = keras.layers.Dense(64, activation='relu')(inputs)
#         x = keras.layers.Dense(64, activation='relu')(x)
#         outputs = keras.layers.Dense(self.action_dim)(x)
#         model = keras.Model(inputs, outputs)
#         model.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-4), loss='mse')
#         return model

#     def update_target_network(self):
#         self.target_model.set_weights(self.model.get_weights())

#     def select_action(self, state):
#         if np.random.rand() < self.epsilon:
#             return np.random.randint(self.action_dim)
#         q_values = self.model.predict(state[None], verbose=0)
#         return np.argmax(q_values[0])

#     def train(self):
#         if self.replay_buffer.size() < self.batch_size:
#             return

#         states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

#         next_actions = np.argmax(self.model.predict(next_states, verbose=0), axis=1)
#         next_q_values = self.target_model.predict(next_states, verbose=0)
#         target_q = rewards + self.gamma * next_q_values[np.arange(self.batch_size), next_actions] * (1 - dones)

#         q_values = self.model.predict(states, verbose=0)
#         q_values[np.arange(self.batch_size), actions] = target_q

#         history = self.model.fit(states, q_values, epochs=1, verbose=0, batch_size=self.batch_size)
#         self.loss_history.append(history.history['loss'][0])

#         self.update_counter += 1
#         if self.update_counter % self.target_update_freq == 0:
#             self.update_target_network()

#         self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

#     def save(self, path):
#         self.model.save(path)

#     def load(self, path):
#         self.model = keras.models.load_model(path)
#         self.update_target_network()

# # Record a video
# def record_video(agent, filename=os.path.join(video_path, "latest.mp4"), max_steps=1000):
#     env = gym.make('LunarLander-v3', render_mode='rgb_array')
#     state, _ = env.reset()
#     frame = env.render()
#     h, w, _ = frame.shape
#     out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))

#     done, total_reward, steps = False, 0, 0
#     while not done and steps < max_steps:
#         action = agent.select_action(state)
#         next_state, reward, terminated, truncated, _ = env.step(action)
#         done = terminated or truncated
#         frame = cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR)
#         out.write(frame)
#         state, total_reward = next_state, total_reward + reward
#         steps += 1

#     out.release()
#     env.close()
#     print(f"Recorded video to {filename}, total reward = {total_reward:.2f}")

# # Evaluation function
# def evaluate(agent, env, episodes=3):
#     rewards = []
#     for _ in range(episodes):
#         state, _ = env.reset()
#         total, done = 0, False
#         while not done:
#             action = agent.select_action(state)
#             state, reward, terminated, truncated, _ = env.step(action)
#             done = terminated or truncated
#             total += reward
#         rewards.append(total)
#     return np.mean(rewards)

# # Training function
# def train_agent(episodes=300, max_steps=500, solved_score=200, eval_freq=10):
#     env = gym.make('LunarLander-v3')
#     state_dim = env.observation_space.shape[0]
#     action_dim = env.action_space.n
#     agent = LunarDoubleDQNAgent(state_dim, action_dim)

#     all_rewards, eval_scores = [], []
#     best_score = -np.inf

#     for ep in range(episodes):
#         state, _ = env.reset()
#         total_reward = 0
#         for step in range(max_steps):
#             action = agent.select_action(state)
#             next_state, reward, terminated, truncated, _ = env.step(action)
#             done = terminated or truncated
#             agent.replay_buffer.add(state, action, reward, next_state, done)
#             agent.train()
#             state, total_reward = next_state, total_reward + reward
#             if done:
#                 break

#         all_rewards.append(total_reward)

#         if ep % eval_freq == 0:
#             eval_score = evaluate(agent, env)
#             eval_scores.append(eval_score)
#             print(f"Episode {ep} | Train reward: {total_reward:.2f} | Eval: {eval_score:.2f} | Eps: {agent.epsilon:.2f}")

#             if eval_score > best_score:
#                 best_score = eval_score
#                 agent.save(os.path.join(checkpoint_path, "best_model.h5"))
#                 print("New best model saved.")

#             if ep % (eval_freq * 5) == 0:
#                 record_video(agent, os.path.join(video_path, f"episode_{ep}.mp4"))

#             if np.mean(all_rewards[-100:]) >= solved_score:
#                 print(f"Environment solved at episode {ep}!")
#                 record_video(agent, os.path.join(video_path, f"solved_at_{ep}.mp4"))
#                 break

#     agent.save(os.path.join(checkpoint_path, "final_model.h5"))
#     record_video(agent, os.path.join(video_path, "final.mp4"))

#     # Plot training progress
#     plt.plot(all_rewards, label='Train Rewards')
#     plt.plot(np.arange(0, len(eval_scores)) * eval_freq, eval_scores, label='Eval Scores')
#     plt.axhline(y=solved_score, color='r', linestyle='--', label='Solved Threshold')
#     plt.legend()
#     plt.title('Training Progress')
#     plt.xlabel('Episode')
#     plt.ylabel('Reward')
#     plt.grid()
#     plt.tight_layout()
#     plt.savefig(os.path.join(log_path, "training_plot.png"))
#     plt.show()

# # Main entry point
# if __name__ == '__main__':
#     train_agent()
import numpy as np
import gymnasium as gym
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from collections import deque
import random
import os
import imageio
import json
from param import lunar_params as params  # <-- Import your parameter file

# GPU Memory Growth
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

# Local directory paths
base_path = './LunarLander-DDQN'
checkpoint_path = os.path.join(base_path, 'checkpoints')
video_path = os.path.join(base_path, 'videos')
log_path = os.path.join(base_path, 'logs')
state_file = os.path.join(checkpoint_path, "training_state.json")

os.makedirs(checkpoint_path, exist_ok=True)
os.makedirs(video_path, exist_ok=True)
os.makedirs(log_path, exist_ok=True)

# -------------------------
# Training state persistence
# -------------------------
def load_training_state(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            state = json.load(f)
        print(f"Resuming from episode {state['ep']} with epsilon={state['epsilon']}")
        return state["all_rewards"], state["eval_scores"], state["ep"], state["epsilon"]
    return [], [], 0, None

def save_training_state(path, all_rewards, eval_scores, ep, epsilon):
    state = {
        "all_rewards": all_rewards,
        "eval_scores": eval_scores,
        "ep": ep,
        "epsilon": epsilon
    }
    with open(path, "w") as f:
        json.dump(state, f)
    print(f"Training state saved at episode {ep}")

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.stack(states), np.array(actions), np.array(rewards),
                np.stack(next_states), np.array(dones))

    def size(self):
        return len(self.buffer)

# Double DQN Agent
class LunarDoubleDQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = params['gamma']
        self.epsilon = params['epsilon_start']
        self.epsilon_min = params['epsilon_min']
        self.epsilon_decay = params['epsilon_decay']
        self.batch_size = params['batch_size']
        self.target_update_freq = params['target_update_freq']
        self.update_counter = 0

        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_network()

        self.replay_buffer = ReplayBuffer(params['replay_buffer_size'])
        self.loss_history = []
        self.reward_history = []

    def build_model(self):
        inputs = keras.Input(shape=(self.state_dim,))
        x = keras.layers.Dense(64, activation='relu')(inputs)
        x = keras.layers.Dense(64, activation='relu')(x)
        outputs = keras.layers.Dense(self.action_dim)(x)
        optimizer_choice = keras.optimizers.Adam(learning_rate=params['lr']) if params['optimizer'].lower() == 'adam' else keras.optimizers.SGD(learning_rate=params['lr'])
        model = keras.Model(inputs, outputs)
        model.compile(optimizer=optimizer_choice, loss='mse')
        return model

    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        q_values = self.model.predict(state[None], verbose=0)
        return np.argmax(q_values[0])

    def train(self):
        if self.replay_buffer.size() < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        next_actions = np.argmax(self.model.predict(next_states, verbose=0), axis=1)
        next_q_values = self.target_model.predict(next_states, verbose=0)
        target_q = rewards + self.gamma * next_q_values[np.arange(self.batch_size), next_actions] * (1 - dones)

        q_values = self.model.predict(states, verbose=0)
        q_values[np.arange(self.batch_size), actions] = target_q

        history = self.model.fit(states, q_values, epochs=1, verbose=0, batch_size=self.batch_size)
        self.loss_history.append(history.history['loss'][0])

        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.update_target_network()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = keras.models.load_model(path)
        self.update_target_network()

# Save GIF
def save_gif(agent, filename, max_steps=1000):
    env = gym.make(params['env_id'], render_mode='rgb_array')
    state, _ = env.reset()
    frames = []
    done, steps = False, 0

    while not done and steps < max_steps:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        frames.append(env.render())
        state = next_state
        steps += 1

    env.close()
    imageio.mimsave(filename, frames, fps=30)
    print(f"Saved GIF to {filename}")

# Evaluation
def evaluate(agent, env, episodes=3):
    rewards = []
    for _ in range(episodes):
        state, _ = env.reset()
        total, done = 0, False
        while not done:
            action = agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += reward
        rewards.append(total)
    return np.mean(rewards)

# Training
def train_agent():
    env = gym.make(params['env_id'])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = LunarDoubleDQNAgent(state_dim, action_dim)

    # Resume from checkpoint if available
    best_model_path = os.path.join(checkpoint_path, "best_model.h5")
    all_rewards, eval_scores, start_ep, loaded_epsilon = load_training_state(state_file)

    if os.path.exists(best_model_path):
        agent.load(best_model_path)
        print("Resumed training from saved checkpoint.")

    if loaded_epsilon is not None:
        agent.epsilon = loaded_epsilon

    best_score = max(eval_scores) if eval_scores else -np.inf

    for ep in range(start_ep, params['num_episodes']):
        state, _ = env.reset()
        total_reward = 0
        for step in range(500):  # max_steps hardcoded in original
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.replay_buffer.add(state, action, reward, next_state, done)
            agent.train()
            state, total_reward = next_state, total_reward + reward
            if done:
                break

        all_rewards.append(total_reward)

        if ep % 10 == 0:
            eval_score = evaluate(agent, env)
            eval_scores.append(eval_score)
            print(f"Episode {ep} | Train reward: {total_reward:.2f} | Eval: {eval_score:.2f} | Eps: {agent.epsilon:.2f}")

            if eval_score > best_score:
                best_score = eval_score
                agent.save(best_model_path)
                print("New best model saved.")

            # Save training state here
            save_training_state(state_file, all_rewards, eval_scores, ep + 1, agent.epsilon)

        # Save GIF every 100 episodes
        if ep % 100 == 0 and ep > 0:
            save_gif(agent, os.path.join(video_path, f"episode_{ep}.gif"))

        if np.mean(all_rewards[-100:]) >= params['reward_threshold']:
            print(f"Environment solved at episode {ep}!")
            save_gif(agent, os.path.join(video_path, f"solved_at_{ep}.gif"))
            break

    agent.save(os.path.join(checkpoint_path, "final_model.h5"))
    save_gif(agent, os.path.join(video_path, "final.gif"))

    # Plot training progress
    plt.plot(all_rewards, label='Train Rewards')
    plt.plot(np.arange(0, len(eval_scores)) * 10, eval_scores, label='Eval Scores')
    plt.axhline(y=params['reward_threshold'], color='r', linestyle='--', label='Solved Threshold')
    plt.legend()
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(log_path, "training_plot.png"))
    plt.show()

if __name__ == '__main__':
    train_agent()
