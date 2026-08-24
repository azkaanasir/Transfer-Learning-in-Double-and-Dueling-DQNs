"""The agent: one class covering all four (arch x target_rule) cells.

Four things here are load-bearing.

**Speed.** The published agent called `model.predict()` three times per gradient
step and `model.fit()` once, so nearly all wall-clock was Keras dispatch rather
than arithmetic. Everything here runs inside `@tf.function`-compiled graphs.

**The target rule is an explicit switch.** Phase 0 found the published DDQN and
Dueling arms differed in *both* architecture and Q-target rule, confounding the
comparison the paper claimed to isolate.

**Gradient clipping is global, not per-tensor.** Keras's `clipnorm` clips each
gradient tensor separately. The mlp cell has 6 weight tensors and the dueling
cell has 10, with different fan-ins, so one scalar `clipnorm` imposes a
materially different constraint on each architecture -- a hidden confound on the
very axis under study. `global_clipnorm` is a single constraint on the whole
gradient and is invariant to how the parameters are partitioned.

**Diagnostics cannot perturb training.** The published loop sampled the replay
buffer to get states for its stream-magnitude diagnostic, which advanced the
buffer's generator and therefore changed every subsequent minibatch: toggling
`log_diagnostics` silently changed the training trajectory. Here every
diagnostic draws from a dedicated `diag` stream and a fixed state batch, and
`validate.py` asserts the trajectory is unchanged when diagnostics are switched
off.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf
import keras

from .config import Config
from .networks import (LAYER_GROUPS, build_feature_probe, build_q_network,
                       build_stream_probe)
from .replay import ReplayBuffer
from .seeding import Seeds
from .transfer import apply_freeze, trainable_report

# Which weight tensors belong to which stream, for the per-stream gradient
# norms that separate an optimisation mismatch from a representational one.
_STREAM_GROUPS = {
    'trunk': ('trunk_fc1', 'trunk_fc2'),
    'value': ('value_fc', 'value_out'),
    'adv': ('adv_fc', 'adv_out'),
    'head': ('q_out',),
}


class Agent:
    def __init__(self, cfg: Config, state_dim: int, action_dim: int,
                 seeds: Seeds | None = None):
        self.cfg = cfg
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.seeds = seeds or Seeds(cfg.seed)

        layer_seeds = self.seeds.layer_seeds(LAYER_GROUPS[cfg.arch]['all'])
        self.online = build_q_network(state_dim, action_dim, cfg.arch,
                                      cfg.hidden, cfg.head_units,
                                      cfg.aggregation, layer_seeds)
        # The target network's initialisation is irrelevant -- it is overwritten
        # immediately -- but it is seeded anyway so that nothing in the run
        # depends on unseeded state.
        self.target = build_q_network(state_dim, action_dim, cfg.arch,
                                      cfg.hidden, cfg.head_units,
                                      cfg.aggregation, layer_seeds)
        self.target.set_weights(self.online.get_weights())

        self.probe = build_stream_probe(self.online)
        self.feature_probe = build_feature_probe(self.online, 'trunk_fc2')

        self.optimizer = keras.optimizers.Adam(
            learning_rate=cfg.lr, global_clipnorm=cfg.grad_clip_norm)
        # Build the optimiser against the *full* variable set now, while every
        # layer is still trainable. Keras 3 binds an optimiser to the variables
        # it first saw, so building later against a frozen subset raises
        # "Unknown variable" the moment the freeze lifts. Pre-building keeps
        # every slot allocated across the freeze boundary, which is also what
        # preserves Adam's moments through the transition.
        self.optimizer.build(self.online.trainable_variables)

        self.buffer = ReplayBuffer(cfg.replay_capacity, state_dim,
                                   self.seeds.rng('buffer'))
        self.env_steps = 0
        # The episode the agent is in. Set by the trainer at the top of each
        # episode, and the index of the exploration schedule -- which is
        # episode-indexed because a step-indexed horizon is endogenous to policy
        # quality (see `Config.epsilon_at`).
        self.episode = 0
        self.update_counter = 0
        self.clip_events = 0
        self._train_fn = None            # rebuilt when the freeze state changes
        self._action_rng = self.seeds.rng('action')
        self._diag_rng = self.seeds.rng('diag')
        self._diag_states: np.ndarray | None = None

    # ---- exploration -----------------------------------------------------
    @property
    def epsilon(self) -> float:
        """Closed form in episodes: independent of the evaluation cadence,
        replayable without simulating history, and exogenous to how well the
        policy happens to be doing."""
        return self.cfg.epsilon_at(self.episode)

    # ---- freezing --------------------------------------------------------
    def set_frozen(self, layer_names, frozen: bool) -> dict:
        """Freeze/unfreeze layers and invalidate the compiled train step.

        Retracing is required because the set of differentiated variables
        changes. It happens at most twice per run, so the cost is negligible,
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

    def act(self, state, greedy: bool = False,
            rng: np.random.Generator | None = None) -> int:
        """Select an action. `rng` overrides the training stream, which is how
        evaluation draws its (unused, since evaluation is greedy) randomness
        without touching exploration."""
        if not greedy:
            gen = rng or self._action_rng
            if gen.random() < self.epsilon:
                return int(gen.integers(self.action_dim))
        s = tf.convert_to_tensor(np.asarray(state, dtype=np.float32)[None])
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

            targets = tf.stop_gradient(
                rewards + (1.0 - dones) * gamma * next_q)
            act_idx = tf.stack([tf.range(tf.shape(actions)[0]), actions], axis=1)
            with tf.GradientTape() as tape:
                q = self.online(states, training=True)
                chosen = tf.gather_nd(q, act_idx)
                td = targets - chosen
                loss = tf.reduce_mean(tf.square(td))
            grads = tape.gradient(loss, variables)
            gnorm = tf.linalg.global_norm([g for g in grads if g is not None])
            self.optimizer.apply_gradients(zip(grads, variables))
            return loss, gnorm, tf.reduce_mean(tf.abs(td)), tf.reduce_mean(q), \
                tf.reduce_max(q)

        return step

    def train_step(self) -> dict:
        """One gradient update. Returns the per-step scalars for logging."""
        if len(self.buffer) < max(self.cfg.batch_size, self.cfg.learning_starts):
            return {}
        if self._train_fn is None:
            self._train_fn = self._build_train_fn()

        s, a, r, ns, d = self.buffer.sample(self.cfg.batch_size)
        loss, gnorm, td_abs, q_mean, q_max = self._train_fn(
            tf.convert_to_tensor(s), tf.convert_to_tensor(a),
            tf.convert_to_tensor(r), tf.convert_to_tensor(ns),
            tf.convert_to_tensor(d))

        self.update_counter += 1
        gnorm = float(gnorm)
        if gnorm > self.cfg.grad_clip_norm:
            self.clip_events += 1

        if self.cfg.target_update == 'hard':
            if self.update_counter % self.cfg.target_update_freq == 0:
                self.target.set_weights(self.online.get_weights())
        else:
            self._soft_update()

        return {'loss': float(loss), 'grad_norm': gnorm,
                'td_error_abs': float(td_abs), 'q_mean': float(q_mean),
                'q_max': float(q_max)}

    @tf.function(reduce_retracing=True)
    def _soft_update_graph(self, tau):
        for t, o in zip(self.target.weights, self.online.weights):
            t.assign(tau * o + (1.0 - tau) * t)

    def _soft_update(self):
        self._soft_update_graph(tf.constant(self.cfg.tau, dtype=tf.float32))

    # ---- diagnostics -----------------------------------------------------
    def set_diagnostic_states(self, states: np.ndarray) -> None:
        """Install the fixed state batch every diagnostic is measured on.

        Fixed for the whole run and drawn from the `diag` stream, so a
        diagnostic trajectory is a comparison of the network at different times
        rather than of the network at different states.
        """
        self._diag_states = np.asarray(states, dtype=np.float32)

    def stream_magnitudes(self) -> dict:
        """Mean |V| and |A| on the fixed diagnostic batch; {} for non-dueling."""
        if self.probe is None or self._diag_states is None:
            return {}
        v, a = self.probe(self._diag_states, training=False)
        v, a = np.asarray(v), np.asarray(a)
        return {'v_abs_mean': float(np.mean(np.abs(v))),
                'a_abs_mean': float(np.mean(np.abs(a))),
                'a_spread': float(np.mean(np.std(a, axis=1)))}

    def trunk_features(self) -> np.ndarray | None:
        """`trunk_fc2` activations on the fixed diagnostic batch.

        The input to `cka_transfer_vs_scratch` and `cka_drift`. Note what is
        *not* offered: CKA between activations on two different state sets.
        That is ill-posed -- CKA compares two representations of the same
        examples -- and for a cross-interface pair the input dimensionalities
        differ, so it is not even computable.
        """
        if self._diag_states is None:
            return None
        return np.asarray(self.feature_probe(self._diag_states, training=False))

    def dead_unit_fraction(self) -> float:
        """Trunk units never active over the diagnostic batch."""
        feats = self.trunk_features()
        if feats is None:
            return float('nan')
        return float(np.mean(np.all(feats <= 0.0, axis=0)))

    def stream_gradient_norms(self) -> dict:
        """Per-stream gradient norms, computed *without* applying an update.

        Separates an optimisation mismatch from a representational one, which is
        the distinction ICANN reviewer 5's question 5 turns on. Uses a batch
        drawn with the `diag` generator, so it consumes none of the training
        stream's randomness and none of the buffer's.
        """
        if len(self.buffer) < self.cfg.batch_size:
            return {}
        s, a, r, ns, d = self.buffer.sample(self.cfg.batch_size,
                                            rng=self._diag_rng)
        gamma = self.cfg.gamma
        next_q_target = np.asarray(self.target(ns, training=False))
        if self.cfg.target_rule == 'double':
            nxt = np.asarray(self.online(ns, training=False)).argmax(axis=1)
            next_q = next_q_target[np.arange(len(nxt)), nxt]
        else:
            next_q = next_q_target.max(axis=1)
        targets = tf.constant(r + (1.0 - d) * gamma * next_q, dtype=tf.float32)

        variables = self.online.trainable_variables
        with tf.GradientTape() as tape:
            q = self.online(tf.constant(s), training=True)
            idx = tf.stack([tf.range(len(a)), tf.constant(a, dtype=tf.int32)],
                           axis=1)
            loss = tf.reduce_mean(tf.square(targets - tf.gather_nd(q, idx)))
        grads = tape.gradient(loss, variables)

        by_layer: dict[str, list] = {}
        for var, grad in zip(variables, grads):
            if grad is None:
                continue
            layer = _layer_of(var, self.online)
            by_layer.setdefault(layer, []).append(grad)

        out = {}
        for group, members in _STREAM_GROUPS.items():
            gs = [g for name in members for g in by_layer.get(name, [])]
            if gs:
                out[f'grad_norm_{group}'] = float(tf.linalg.global_norm(gs))
        allg = [g for gs in by_layer.values() for g in gs]
        if allg:
            out['grad_norm_global'] = float(tf.linalg.global_norm(allg))
        return out

    def plasticity_signals(self) -> dict:
        """Effective rank of the trunk features, and parameter norms.

        The plasticity-loss literature (primacy bias, capacity loss, loss of
        plasticity) supplies an architecture-free explanation for degradation
        after pretraining: units die, parameter norms grow, and the feature
        matrix loses rank. That is a *rival explanation* for anything this study
        would otherwise attribute to the transferred representation, and the
        weight-scale control (C3) does not exclude it -- C3 preserves the weight
        multiset, which says nothing about the rank of the features those weights
        produce. Measuring it is what makes the rival hypothesis checkable
        instead of arguable.

        `effective_rank` is the exponential of the entropy of the normalised
        singular-value spectrum, which is continuous and does not need an
        arbitrary tolerance the way a matrix-rank threshold does. `stable_rank`
        is the squared Frobenius-to-spectral norm ratio, reported alongside
        because the two disagree in informative ways.
        """
        out: dict = {}
        feats = self.trunk_features()
        if feats is not None and feats.size:
            centred = feats - feats.mean(axis=0, keepdims=True)
            sv = np.linalg.svd(centred, compute_uv=False)
            total = float(sv.sum())
            if total > 0:
                p = sv / total
                p = p[p > 0]
                out['effective_rank'] = float(np.exp(-np.sum(p * np.log(p))))
                out['stable_rank'] = float((sv ** 2).sum() / (sv[0] ** 2)
                                           if sv[0] > 0 else np.nan)
                out['feature_var_mean'] = float(centred.var(axis=0).mean())
        by_group: dict[str, list[float]] = {}
        for layer in self.online.layers:
            if not layer.weights:
                continue
            group = next((g for g, members in _STREAM_GROUPS.items()
                          if layer.name in members), 'other')
            norm = float(np.sqrt(sum(float(np.sum(np.asarray(w) ** 2))
                                     for w in layer.get_weights())))
            by_group.setdefault(group, []).append(norm)
        for group, norms in by_group.items():
            out[f'param_norm_{group}'] = float(np.sqrt(np.sum(np.square(norms))))
        if by_group:
            out['param_norm_total'] = float(np.sqrt(sum(
                n ** 2 for norms in by_group.values() for n in norms)))
        return out

    # ---- checkpointing ---------------------------------------------------
    def optimizer_state(self) -> dict:
        """Adam's slot variables and step counter.

        The published checkpoint saved weights and the replay buffer but not the
        optimiser, so a resumed run restarted Adam from zero moments with
        `iterations == 0` -- a different optimiser trajectory wearing the same
        run's name. Verified by execution during review of the previous design.
        """
        return {'values': [np.asarray(v.numpy()) for v in self.optimizer.variables],
                'shapes': [list(np.asarray(v.numpy()).shape)
                           for v in self.optimizer.variables]}

    def load_optimizer_state(self, state: dict) -> None:
        values = state.get('values') or []
        if len(values) != len(self.optimizer.variables):
            raise ValueError(
                f'optimiser checkpoint has {len(values)} variables but this '
                f'agent has {len(self.optimizer.variables)}. Refusing to resume: '
                f'a partially restored optimiser is a different training run.')
        for var, val in zip(self.optimizer.variables, values):
            var.assign(np.asarray(val, dtype=var.dtype.as_numpy_dtype)
                       if hasattr(var.dtype, 'as_numpy_dtype')
                       else np.asarray(val))


def _layer_of(var, model) -> str:
    """Map a trainable variable back to the layer that owns it."""
    for layer in model.layers:
        for w in layer.weights:
            if w is var:
                return layer.name
    # Fall back to the variable path, which for Keras 3 embeds the layer name.
    return str(getattr(var, 'path', getattr(var, 'name', 'unknown'))).split('/')[0]


__all__ = ['Agent']
