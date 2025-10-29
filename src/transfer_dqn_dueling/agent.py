"""Agent implementation that optionally uses a transfer model built from a pretrained CartPole model."""

import os
import pickle
import json
from typing import Tuple, List, Optional

import numpy as np
from tensorflow import keras

from models import build_dueling_model, build_transfer_dueling_model
from replay_buffer import ReplayBuffer
from config import BASE_SAVE_PATH, MODEL_LATEST, TARGET_MODEL_LATEST, REPLAY_BUFFER, METADATA, lunar_params


class DuelingDQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        params: dict,
        pretrained_path: Optional[str] = None,
        freeze_base: bool = True,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.params = params

        self.gamma = params['gamma']
        self.epsilon = params.get('epsilon_start', 1.0)
        self.epsilon_min = params.get('epsilon_min', 0.01)
        self.epsilon_decay = params.get('epsilon_decay', 0.995)
        self.batch_size = params.get('batch_size', 64)
        self.target_update_freq = params.get('target_update_freq', 1000)
        self.update_counter = 0
        self.learning_rate = params.get('lr', 1e-3)

        # Build model: use transfer builder if pretrained_path provided
        if pretrained_path:
            self.model = build_transfer_dueling_model(
                state_dim=state_dim,
                action_dim=action_dim,
                pretrained_path=pretrained_path,
                freeze_base=freeze_base,
                learning_rate=self.learning_rate
            )
        else:
            self.model = build_dueling_model(state_dim, action_dim, learning_rate=self.learning_rate)

        # Build target model and sync weights
        self.target_model = build_dueling_model(state_dim, action_dim, learning_rate=self.learning_rate)
        self.update_target_network()

        # replay buffer
        self.replay_buffer = ReplayBuffer(params['replay_buffer_size'])
        self.loss_history = []
        self.reward_history = []

    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())

    def select_action(self, state: np.ndarray) -> int:
        if np.random.rand() <= self.epsilon:
            return int(np.random.randint(self.action_dim))
        q = self.model.predict(state.reshape(1, -1).astype(np.float32), verbose=0)[0]
        return int(np.argmax(q))

    def train_step(self) -> float:
        """Perform one update (sample a batch and train). Uses train_on_batch to avoid eager/fit differences."""
        if self.replay_buffer.size() < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Double DQN target computation: choose action from online model, evaluate with target model
        next_q_online = self.model.predict(next_states, verbose=0)
        next_actions = np.argmax(next_q_online, axis=1)
        next_q_target = self.target_model.predict(next_states, verbose=0)
        target_vals = rewards + (1 - dones) * self.gamma * next_q_target[np.arange(len(next_actions)), next_actions]

        q_curr = self.model.predict(states, verbose=0)
        q_curr[np.arange(len(states)), actions] = target_vals

        # Train using train_on_batch for stability and to avoid eager conversion issues
        loss = self.model.train_on_batch(states, q_curr)
        # train_on_batch returns [loss, metrics..] when compiled with metrics; adapt accordingly
        if isinstance(loss, list) or isinstance(loss, tuple):
            loss_value = float(loss[0])
        else:
            loss_value = float(loss)

        self.loss_history.append(loss_value)

        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.update_target_network()

        return loss_value

    # persistence
    def save_checkpoint(self, episode: int, episode_rewards: List[float], episode_lengths: List[int], validation_rewards: List[float]):
        os.makedirs(BASE_SAVE_PATH, exist_ok=True)
        self.model.save(os.path.join(BASE_SAVE_PATH, MODEL_LATEST))
        self.target_model.save(os.path.join(BASE_SAVE_PATH, TARGET_MODEL_LATEST))
        with open(os.path.join(BASE_SAVE_PATH, REPLAY_BUFFER), 'wb') as f:
            pickle.dump(self.replay_buffer.save_buffer(), f)

        metadata = {
            'epsilon': self.epsilon,
            'ep': episode,
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'validation_rewards': validation_rewards,
            'loss_history': self.loss_history,
            'update_counter': self.update_counter,
        }
        with open(os.path.join(BASE_SAVE_PATH, METADATA), 'w') as f:
            json.dump(metadata, f)

    def load_checkpoint(self) -> Tuple[bool, int, List[float], List[int], List[float]]:
        model_path = os.path.join(BASE_SAVE_PATH, MODEL_LATEST)
        meta_path = os.path.join(BASE_SAVE_PATH, METADATA)
        if not (os.path.exists(model_path) and os.path.exists(meta_path)):
            return False, 0, [], [], []
        try:
            # load weights into model (compile=False then recompile) to avoid Keras3 string-deserialize issue
            loaded = keras.models.load_model(model_path, compile=False)
            self.model.set_weights(loaded.get_weights())
            self.target_model.set_weights(loaded.get_weights())
            # replay buffer
            try:
                with open(os.path.join(BASE_SAVE_PATH, REPLAY_BUFFER), 'rb') as f:
                    data = pickle.load(f)
                    self.replay_buffer.load_buffer(data)
            except Exception:
                pass
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            self.epsilon = meta.get('epsilon', self.epsilon)
            self.loss_history = meta.get('loss_history', [])
            ep = meta.get('ep', 0)
            ep_rewards = meta.get('episode_rewards', [])
            ep_lengths = meta.get('episode_lengths', [])
            val_rewards = meta.get('validation_rewards', [])
            return True, ep + 1, ep_rewards, ep_lengths, val_rewards
        except Exception as e:
            print('Failed to load checkpoint:', e)
            return False, 0, [], [], []

    def unfreeze_trunk_and_recompile(self):
        """If you want to fine-tune the frozen trunk later: call this to unfreeze and recompile."""
        for layer in self.model.layers:
            if layer.name.startswith('trunk_dense_'):
                layer.trainable = True
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=keras.losses.MeanSquaredError(),
            metrics=[keras.metrics.MeanSquaredError()]
        )
