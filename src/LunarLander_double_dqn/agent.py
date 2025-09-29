"""
Dueling DQN agent: builds models, performs training steps and handles checkpointing.

Fixed behaviors:
- Properly handles missing checkpoint files and starts training from scratch.
- Loads saved models with compile=False to avoid Keras3 deserialization issues (e.g. "keras.metrics.mse").
- Recompiles models after loading with explicit Keras loss/metric classes.
- Supports custom_objects for the custom DuelingCombineLayer.
- Provides clearer log messages.
"""
import json
import os
import pickle
import time
from typing import List, Tuple

import numpy as np
from tensorflow import keras

from models import build_dueling_model, DuelingCombineLayer
from replay_buffer import ReplayBuffer
from config import BASE_SAVE_PATH, MODEL_LATEST, TARGET_MODEL_LATEST, REPLAY_BUFFER, METADATA


class DuelingDQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10_000,
        batch_size: int = 64,
        target_update_freq: int = 10,
        learning_rate: float = 1e-3,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        # exploration
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # replay & learning
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.loss_history: List[float] = []

        # model & training bookkeeping
        self.target_update_freq = target_update_freq
        self.update_counter = 0
        self.learning_rate = learning_rate

        # build models (uncompiled) — build_dueling_model may compile; if so it's okay
        self.model = build_dueling_model(state_dim, action_dim, learning_rate)
        self.target_model = build_dueling_model(state_dim, action_dim, learning_rate)
        self.update_target_network()

    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())

    def select_action(self, state: np.ndarray) -> int:
        if np.random.rand() < self.epsilon:
            return int(np.random.randint(self.action_dim))
        state = np.reshape(state, [1, self.state_dim])
        q_values = self.model.predict(state, verbose=0)[0]
        return int(np.argmax(q_values))

    def train(self) -> float:
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        next_q_values = self.target_model.predict(next_states, verbose=0)
        max_next_q = np.max(next_q_values, axis=1)
        target_q_values = rewards + (1 - dones) * self.gamma * max_next_q

        current_q = self.model.predict(states, verbose=0)
        batch_indices = np.arange(len(states))
        current_q[batch_indices, actions] = target_q_values

        history = self.model.fit(states, current_q, verbose=0, batch_size=self.batch_size)
        loss = float(history.history['loss'][0])
        self.loss_history.append(loss)

        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.update_target_network()

        # if self.epsilon > self.epsilon_min:
        #     self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return loss

    # --- checkpointing -----------------------------------------------------
    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)

    def load(self, filepath: str):
        # When loading models containing custom layers, supply custom_objects
        self.model = keras.models.load_model(filepath, custom_objects={'DuelingCombineLayer': DuelingCombineLayer})
        self.update_target_network()

    def save_checkpoint(self, episode: int, episode_rewards: List[float], episode_lengths: List[int], validation_rewards: List[float]):
        os.makedirs(BASE_SAVE_PATH, exist_ok=True)

        # Save model weights/architecture
        self.model.save(os.path.join(BASE_SAVE_PATH, MODEL_LATEST))
        self.target_model.save(os.path.join(BASE_SAVE_PATH, TARGET_MODEL_LATEST))

        # Save replay buffer and metadata
        with open(os.path.join(BASE_SAVE_PATH, REPLAY_BUFFER), 'wb') as f:
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

        with open(os.path.join(BASE_SAVE_PATH, METADATA), 'w') as f:
            json.dump(metadata, f)

        print(f"Checkpoint saved at episode {episode}")

    def _recompile_loaded_models(self):
        """Helper to recompile models after loading to avoid deserialization issues."""
        try:
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                loss=keras.losses.MeanSquaredError(),
                metrics=[keras.metrics.MeanSquaredError()]
            )
        except Exception as e:
            print(f"Warning: failed to recompile main model cleanly: {e}")

        try:
            self.target_model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                loss=keras.losses.MeanSquaredError(),
                metrics=[keras.metrics.MeanSquaredError()]
            )
        except Exception as e:
            print(f"Warning: failed to recompile target model cleanly: {e}")

    def load_checkpoint(self) -> Tuple[bool, int, List[float], List[int], List[float]]:
        model_path = os.path.join(BASE_SAVE_PATH, MODEL_LATEST)
        target_model_path = os.path.join(BASE_SAVE_PATH, TARGET_MODEL_LATEST)

        # If model files don't exist, start fresh
        if not os.path.exists(model_path):
            print("No checkpoint model found. Starting training from scratch.")
            return False, 0, [], [], []

        try:
            print(f"Loading model from {model_path} (compile=False to avoid deserialization issues)...")
            # load with compile=False to avoid old-metric deserialization problems
            self.model = keras.models.load_model(model_path,
                                                custom_objects={'DuelingCombineLayer': DuelingCombineLayer},
                                                compile=False)

            if os.path.exists(target_model_path):
                self.target_model = keras.models.load_model(target_model_path,
                                                            custom_objects={'DuelingCombineLayer': DuelingCombineLayer},
                                                            compile=False)
            else:
                # fallback: copy weights into target model if target not saved
                print("No separate target model found; copying weights to target model.")
                self.update_target_network()

            # attempt to recompile with explicit Keras classes
            self._recompile_loaded_models()

            # load replay buffer if present
            try:
                with open(os.path.join(BASE_SAVE_PATH, REPLAY_BUFFER), 'rb') as f:
                    buffer_data = pickle.load(f)
                    self.replay_buffer.load_buffer(buffer_data)
            except (FileNotFoundError, EOFError, pickle.UnpicklingError):
                print("No replay buffer found or failed to load it; continuing without it.")

            # load metadata
            episode = 0
            episode_rewards = []
            episode_lengths = []
            validation_rewards = []

            try:
                with open(os.path.join(BASE_SAVE_PATH, METADATA), 'r') as f:
                    metadata = json.load(f)
                    self.epsilon = metadata.get('epsilon', self.epsilon)
                    self.update_counter = metadata.get('update_counter', 0)
                    self.loss_history = metadata.get('loss_history', [])

                    episode = metadata.get('episode', 0)
                    episode_rewards = metadata.get('episode_rewards', [])
                    episode_lengths = metadata.get('episode_lengths', [])
                    validation_rewards = metadata.get('validation_rewards', [])
            except (FileNotFoundError, json.JSONDecodeError):
                print("No metadata found or failed to parse metadata; continuing with defaults.")

            print(f"Loaded checkpoint from episode {episode}")
            return True, episode + 1, episode_rewards, episode_lengths, validation_rewards

        except Exception as e:
            # Catch any deserialization/other errors and start fresh (safer than crashing)
            print(f"Error loading checkpoint: {e}")
            print("Starting training from scratch.")
            return False, 0, [], [], []
