# lunar_dueling_dqn.py
import os, json, random
from collections import deque

import numpy as np
import gymnasium as gym
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import imageio

from param import lunar_params as params  # <-- Import your parameter file

# -------------------------
# GPU memory growth (safe on CPU too)
# -------------------------
for d in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(d, True)
    except Exception:
        pass

# -------------------------
# Paths
# -------------------------
base_path       = './LunarLander-DuelingDQN'
checkpoint_path = os.path.join(base_path, 'checkpoints')
video_path      = os.path.join(base_path, 'videos')
log_path        = os.path.join(base_path, 'logs')
state_file      = os.path.join(checkpoint_path, "training_state.json")

os.makedirs(checkpoint_path, exist_ok=True)
os.makedirs(video_path, exist_ok=True)
os.makedirs(log_path, exist_ok=True)

# -------------------------
# Training state persistence
# -------------------------
def load_training_state(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            s = json.load(f)
        print(f"Resuming from episode {s['ep']} with epsilon={s['epsilon']:.4f}")
        return s["all_rewards"], s["eval_scores"], s["ep"], s["epsilon"]
    return [], [], 0, None

def save_training_state(path, all_rewards, eval_scores, ep, epsilon):
    with open(path, "w") as f:
        json.dump({
            "all_rewards": all_rewards,
            "eval_scores": eval_scores,
            "ep": ep,
            "epsilon": epsilon
        }, f)
    print(f"Training state saved at episode {ep}")

# -------------------------
# Replay Buffer
# -------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, s, a, r, ns, d):
        self.buffer.append((s, a, r, ns, d))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.stack(states).astype(np.float32),
                np.array(actions, dtype=np.int32),
                np.array(rewards, dtype=np.float32),
                np.stack(next_states).astype(np.float32),
                np.array(dones, dtype=np.float32))

    def size(self):
        return len(self.buffer)

# -------------------------
# Dueling DQN Agent 
# -------------------------
class DuelingDQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim  = state_dim
        self.action_dim = action_dim

        self.gamma            = params['gamma']
        self.epsilon          = params['epsilon_start']
        self.epsilon_min      = params['epsilon_min']
        self.epsilon_decay    = params['epsilon_decay']
        self.batch_size       = params['batch_size']
        self.target_update_freq = params['target_update_freq']
        self.update_counter   = 0

        self.model        = self.build_model()
        self.target_model = self.build_model()
        self.update_target_network()

        self.replay_buffer = ReplayBuffer(params['replay_buffer_size'])
        self.loss_history  = []
        self.reward_history = []

    def build_model(self):
        inputs = keras.Input(shape=(self.state_dim,), dtype=tf.float32)

        # shared trunk
        x = keras.layers.Dense(128, activation='relu')(inputs)
        x = keras.layers.Dense(128, activation='relu')(x)

        # value stream
        v = keras.layers.Dense(128, activation='relu')(x)
        v = keras.layers.Dense(1)(v)

        # advantage stream
        a = keras.layers.Dense(128, activation='relu')(x)
        a = keras.layers.Dense(self.action_dim)(a)

        # dueling combine: Q = V + (A - mean(A))
        def dueling_combine(tensors):
            value, advantage = tensors
            advantage = advantage - tf.reduce_mean(advantage, axis=1, keepdims=True)
            return value + advantage

        q_values = keras.layers.Lambda(dueling_combine)([v, a])

        opt = (keras.optimizers.Adam(learning_rate=params['lr'])
               if params.get('optimizer', 'adam').lower() == 'adam'
               else keras.optimizers.SGD(learning_rate=params['lr']))

        model = keras.Model(inputs, q_values)
        model.compile(optimizer=opt, loss='mse')
        return model

    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_dim)
        q = self.model.predict(state.reshape(1, -1).astype(np.float32), verbose=0)[0]
        return int(np.argmax(q))

    def train(self):
        if self.replay_buffer.size() < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Dueling DQN : max over target-Q for next state
        next_q = self.target_model.predict(next_states, verbose=0)
        target_q_vals = rewards + self.gamma * np.max(next_q, axis=1) * (1.0 - dones)

        q_curr = self.model.predict(states, verbose=0)
        q_curr[np.arange(self.batch_size), actions] = target_q_vals

        hist = self.model.fit(states, q_curr, epochs=1, verbose=0, batch_size=self.batch_size)
        self.loss_history.append(hist.history['loss'][0])

        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.update_target_network()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = keras.models.load_model(path, compile=True)
        self.update_target_network()

# -------------------------
# GIF saving (headless)
# -------------------------
def save_gif(agent, filename, max_steps=1000):
    if 'gif' not in params.get('video_format', []):
        return
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
    print(f"[gif] saved -> {filename}")

# -------------------------
# Evaluation 
# -------------------------
def evaluate(agent, env, episodes=3):
    scores = []
    for _ in range(episodes):
        state, _ = env.reset()
        total, done = 0.0, False
        while not done:
            # greedy at eval
            q = agent.model.predict(state[None, :].astype(np.float32), verbose=0)[0]
            action = int(np.argmax(q))
            state, r, term, trunc, _ = env.step(action)
            total += r
            done = term or trunc
        scores.append(total)
    return float(np.mean(scores))

# -------------------------
# Training 
# -------------------------
def train_agent():
    env = gym.make(params['env_id'])
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DuelingDQNAgent(state_dim, action_dim)

    # resume
    best_model_path = os.path.join(checkpoint_path, "best_model.h5")
    all_rewards, eval_scores, start_ep, loaded_eps = load_training_state(state_file)
    if os.path.exists(best_model_path):
        agent.load(best_model_path)
        print("Resumed training from saved checkpoint.")
    if loaded_eps is not None:
        agent.epsilon = loaded_eps

    best_score = max(eval_scores) if eval_scores else -np.inf

    num_episodes = params['num_episodes']
    max_steps    = params.get('max_steps', 500)

    for ep in range(start_ep, num_episodes):
        state, _ = env.reset()
        total_reward = 0.0

        for _ in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc

            agent.replay_buffer.add(state, action, reward, next_state, done)
            agent.train()

            state = next_state
            total_reward += reward
            if done:
                break

        all_rewards.append(total_reward)

        # periodic eval/checkpoint 
        if ep % 10 == 0:
            eval_score = evaluate(agent, env)
            eval_scores.append(eval_score)
            print(f"Episode {ep} | Train: {total_reward:.2f} | Eval: {eval_score:.2f} | Eps: {agent.epsilon:.3f}")

            if eval_score > best_score:
                best_score = eval_score
                agent.save(best_model_path)
                print("New best model saved.")

            save_training_state(state_file, all_rewards, eval_scores, ep + 1, agent.epsilon)

        # save GIF every 100 episodes 
        if ep % 100 == 0 and ep > 0 and 'gif' in params.get('video_format', []):
            save_gif(agent, os.path.join(video_path, f"episode_{ep}.gif"))

        # early stop when solved (mean of last 100)
        if len(all_rewards) >= 100 and np.mean(all_rewards[-100:]) >= params['reward_threshold']:
            print(f"Environment solved at episode {ep}!")
            if 'gif' in params.get('video_format', []):
                save_gif(agent, os.path.join(video_path, f"solved_at_{ep}.gif"))
            break

    # final saves
    agent.save(os.path.join(checkpoint_path, "final_model.h5"))
    if 'gif' in params.get('video_format', []):
        save_gif(agent, os.path.join(video_path, "final.gif"))

    # plot 
    plt.plot(all_rewards, label='Train Rewards')
    if eval_scores:
        plt.plot(np.arange(0, len(eval_scores)) * 10, eval_scores, label='Eval Scores')
    plt.axhline(y=params['reward_threshold'], linestyle='--', label='Solved Threshold')
    plt.legend()
    plt.title('Dueling DQN — Training Progress')
    plt.xlabel('Episode'); plt.ylabel('Reward'); plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(log_path, "training_plot.png"))
    plt.show()

if __name__ == '__main__':
    train_agent()
