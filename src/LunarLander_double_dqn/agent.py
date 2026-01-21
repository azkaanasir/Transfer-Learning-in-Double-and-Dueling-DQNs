
import os, json, pickle, time
import numpy as np
from tensorflow import keras

from models import build_ddqn_model
from replay_buffer import ReplayBuffer
from config import *

class DoubleDQNAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        gamma,
        epsilon,
        epsilon_min,
        epsilon_decay,
        buffer_size,
        batch_size,
        target_update_freq,
        learning_rate
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(buffer_size)

        self.update_counter = 0
        self.target_update_freq = target_update_freq
        self.loss_history = []

        self.model = build_ddqn_model(state_dim, action_dim, learning_rate)
        self.target_model = build_ddqn_model(state_dim, action_dim, learning_rate)
        self.update_target_network()

    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        q = self.model.predict(state[None], verbose=0)[0]
        return int(np.argmax(q))
    
    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        s, a, r, ns, d = self.replay_buffer.sample(self.batch_size)


        next_actions = np.argmax(self.model.predict(ns, verbose=0), axis=1)
        next_q = self.target_model.predict(ns, verbose=0)
        target = r + (1 - d) * self.gamma * next_q[np.arange(self.batch_size), next_actions]

        q_vals = self.model.predict(s, verbose=0)
        q_vals[np.arange(self.batch_size), a] = target

        history = self.model.fit(s, q_vals, verbose=0)
        loss = history.history['loss'][0]
        self.loss_history.append(loss)

        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.update_target_network()

        return float(loss)

