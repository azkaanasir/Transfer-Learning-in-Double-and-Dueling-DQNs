"""The agent: one class covering all four (arch x target_rule) cells.

Two things here are load-bearing for Phase 1.

**Speed.** The original agent called `model.predict()` three times per gradient
step and `model.fit()` once. `predict()` and `fit()` carry per-call dispatch
overhead measured in milliseconds, and a 500-episode LunarLander run performs
~150k-270k updates, so nearly all wall-clock was Keras dispatch rather than
arithmetic. Everything here runs inside `@tf.function`-compiled graphs.

**The target rule is an explicit switch.** Phase 0 found the published DDQN and
Dueling arms differed in *both* architecture and Q-target rule, confounding the
comparison the paper claimed to isolate. Here `target_rule` is a first-class
axis, so the 2x2 is exact.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras

from .config import Config
from .networks import build_q_network, build_stream_probe
from .replay import ReplayBuffer
from .transfer import apply_freeze, trainable_report


class Agent:
    def __init__(self, cfg: Config, state_dim: int, action_dim: int):
        self.cfg = cfg
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.online = build_q_network(state_dim, action_dim, cfg.arch,
                                      cfg.hidden, cfg.head_units)
        self.target = build_q_network(state_dim, action_dim, cfg.arch,
                                      cfg.hidden, cfg.head_units)
        self.target.set_weights(self.online.get_weights())
        self.probe = build_stream_probe(self.online) if cfg.log_diagnostics else None

        self.optimizer = keras.optimizers.Adam(learning_rate=cfg.lr,
                                               clipnorm=cfg.grad_clip_norm)
        # Build the optimiser against the *full* variable set now, while every
        # layer is still trainable. Keras 3 binds an optimiser to the variables
        # it first saw, so if we let it build later against a frozen subset it
        # raises "Unknown variable" the moment the freeze lifts and the trunk
        # returns. Pre-building keeps every slot allocated across the
        # freeze/unfreeze boundary, which is also what preserves Adam's moments
        # through the transition.
        self.optimizer.build(self.online.trainable_variables)
        self.buffer = ReplayBuffer(cfg.replay_capacity, state_dim, cfg.seed)

        self.epsilon = cfg.epsilon_start
        self.update_counter = 0
        self._train_fn = None          # rebuilt whenever the freeze state changes
        self._rng = np.random.default_rng(cfg.seed)

    # ---- freezing --------------------------------------------------------
    def set_frozen(self, layer_names, frozen: bool) -> dict:
        """Freeze/unfreeze layers and invalidate the compiled train step.

        Retracing is required because the set of differentiated variables
        changes. It happens at most twice per run, so the cost is negligible --
        and because the optimiser object is untouched, Adam's slot variables
        survive the transition.
        """
        apply_freeze(self.online, layer_names, frozen)
        self._train_fn = None
        return trainable_report(self.online)

    # ---- acting ----------------------------------------------------------
    @tf.function(reduce_retracing=True)
    def _greedy(self, state):
        return tf.argmax(self.online(state, training=False), axis=1,
                         output_type=tf.int32)

    def act(self, state, greedy: bool = False) -> int:
        if not greedy and self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.action_dim))
        s = tf.convert_to_tensor(state[None], dtype=tf.float32)
        return int(self._greedy(s)[0])

    # ---- learning --------------------------------------------------------
    def _build_train_fn(self):
        gamma = tf.constant(self.cfg.gamma, dtype=tf.float32)
        double = self.cfg.target_rule == 'double'
        variables = self.online.trainable_variables

        @tf.function(reduce_retracing=True)
        def step(states, actions, rewards, next_states, dones):
            next_q_target = self.target(next_states, training=False)
            if double:
                # Double DQN: select with the online net, evaluate with target.
                next_actions = tf.argmax(self.online(next_states, training=False),
                                         axis=1, output_type=tf.int32)
                idx = tf.stack([tf.range(tf.shape(next_actions)[0]),
                                next_actions], axis=1)
                next_q = tf.gather_nd(next_q_target, idx)
            else:
                next_q = tf.reduce_max(next_q_target, axis=1)

            targets = rewards + (1.0 - dones) * gamma * next_q
            targets = tf.stop_gradient(targets)

            act_idx = tf.stack([tf.range(tf.shape(actions)[0]), actions], axis=1)
            with tf.GradientTape() as tape:
                q = self.online(states, training=True)
                chosen = tf.gather_nd(q, act_idx)
                loss = tf.reduce_mean(tf.square(targets - chosen))
            grads = tape.gradient(loss, variables)
            self.optimizer.apply_gradients(zip(grads, variables))
            return loss

        return step

    def train_step(self) -> float:
        if len(self.buffer) < max(self.cfg.batch_size, self.cfg.learning_starts):
            return float('nan')
        if self._train_fn is None:
            self._train_fn = self._build_train_fn()

        s, a, r, ns, d = self.buffer.sample(self.cfg.batch_size)
        loss = self._train_fn(
            tf.convert_to_tensor(s), tf.convert_to_tensor(a),
            tf.convert_to_tensor(r), tf.convert_to_tensor(ns),
            tf.convert_to_tensor(d))

        self.update_counter += 1
        if self.cfg.target_update == 'hard':
            if self.update_counter % self.cfg.target_update_freq == 0:
                self.target.set_weights(self.online.get_weights())
        else:
            self._soft_update()
        return float(loss)

    @tf.function(reduce_retracing=True)
    def _soft_update_graph(self, tau):
        for t, o in zip(self.target.weights, self.online.weights):
            t.assign(tau * o + (1.0 - tau) * t)

    def _soft_update(self):
        self._soft_update_graph(tf.constant(self.cfg.tau, dtype=tf.float32))

    # ---- diagnostics -----------------------------------------------------
    def stream_magnitudes(self, states) -> dict:
        """Mean |V| and |A| over a batch of states; {} for non-dueling nets.

        Supplies the mechanistic evidence Phase 2 needs and three reviewers
        asked for (C6), at negligible cost during training.
        """
        if self.probe is None:
            return {}
        v, a = self.probe(tf.convert_to_tensor(states, dtype=tf.float32),
                          training=False)
        return {'v_abs_mean': float(tf.reduce_mean(tf.abs(v))),
                'a_abs_mean': float(tf.reduce_mean(tf.abs(a))),
                'a_spread': float(tf.reduce_mean(
                    tf.math.reduce_std(a, axis=1)))}

    # ---- epsilon ---------------------------------------------------------
    def decay_epsilon(self) -> None:
        self.epsilon = max(self.cfg.epsilon_min,
                           self.epsilon * self.cfg.epsilon_decay)
