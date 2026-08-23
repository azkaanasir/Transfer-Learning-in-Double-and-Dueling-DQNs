"""Training loop: protocol setup, freeze schedule, evaluation, resume, provenance.

Design notes, each tied to a specific defect the Phase 0 audit or the
adversarial design review found:

* **The primary endpoint is a held-out evaluation, averaged over the final k
  checkpoints.** The published runs' evaluation window was never disclosed and
  mixed exploration-contaminated training returns with 5-episode monitoring
  evaluations. A single terminal snapshot is also the wrong estimator for a
  process the pilot shows oscillating (solve at episode 620, destabilise,
  re-solve once epsilon floors), so k checkpoints are averaged.

* **The freeze schedule is real, indexed in gradient updates, and verified.**
  The manuscript described freezing for 100 episodes and unfreezing; the code
  never changed `trainable` after construction. Episodes are also the wrong unit:
  LunarLander episode length varies by an order of magnitude with performance, so
  an episode-indexed window applies a performance-dependent amount of
  optimisation. Weight fingerprints at each transition make the freeze a
  checkable fact rather than a claim.

* **Resume is state-complete or refused.** The published checkpoint omitted the
  optimiser, so a resumed run restarted Adam with zero moments while claiming to
  be the same run. It also appended to the metrics file without truncating,
  duplicating episodes.

* **Nothing measured can perturb what is measured.** Evaluation uses its own
  environment instance and its own seed stream; diagnostics use a fixed state
  batch drawn from the `diag` stream; the linear probe runs on a snapshot and
  restores it.
"""
from __future__ import annotations

import json
import os
import shutil
import time

import numpy as np
import tensorflow as tf
import keras

from . import envs, provenance
from .agent import Agent
from .config import Config
from .metrics import JsonlLog, MetricsLog
from .networks import LAYER_GROUPS, build_q_network
from .seeding import Seeds, seed_frameworks
from .shift import linear_cka
from .transfer import (load_source, permute_source, recalibrate_value_head,
                       spectrum_matched_source, transfer_summary,
                       transfer_weights, trainable_report, verify_freeze,
                       weight_fingerprint)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(env, agent: Agent, episodes: int, seeds: Seeds, stream: str,
             checkpoint: int, max_steps: int) -> tuple[float, list[dict]]:
    """Greedy evaluation. Returns the mean return and one record per episode.

    Per-episode records are kept, not just the mean: they are what makes the
    evaluation-noise floor estimable, and therefore what lets `within_run_sd` be
    separated from measurement noise instead of conflated with it.
    """
    rows = []
    for i in range(episodes):
        seed = seeds.eval_seed(stream, checkpoint, i)
        state, _ = env.reset(seed=int(seed) % (2 ** 31 - 1))
        state = np.asarray(state, dtype=np.float32)
        total, steps, done = 0.0, 0, False
        while not done and steps < max_steps:
            action = agent.act(state, greedy=True)
            state, reward, term, trunc, _ = env.step(action)
            state = np.asarray(state, dtype=np.float32)
            total += float(reward)
            done = term or trunc
            steps += 1
        rows.append({'stream': stream, 'checkpoint': checkpoint, 'index': i,
                     'env_seed': int(seed), 'return': total, 'length': steps})
    return float(np.mean([r['return'] for r in rows])), rows


class Trainer:
    def __init__(self, cfg: Config, argv: list[str] | None = None):
        self.cfg = cfg
        self.seeds = Seeds(cfg.seed)
        seed_frameworks(cfg.seed)
        self.dir = cfg.run_dir()
        os.makedirs(self.dir, exist_ok=True)

        # Three environment instances. Sharing one would let evaluation and
        # diagnostics disturb the training environment's internal state, which
        # is the sort of coupling that makes an ablation measure itself.
        self.env, self.env_info = envs.make(cfg.env)
        self.eval_env, _ = envs.make(cfg.env)
        state_dim = int(self.env_info['obs_dim'])
        action_dim = int(self.env_info['act_dim'])

        self.agent = Agent(cfg, state_dim, action_dim, self.seeds)
        self.metrics = MetricsLog(os.path.join(self.dir, 'metrics.jsonl'))
        self.evals = JsonlLog(os.path.join(self.dir, 'eval_episodes.jsonl'))
        self.events = JsonlLog(os.path.join(self.dir, 'events.jsonl'))

        self.manifest: dict = {
            'identity': cfg.identity(),
            'config': cfg.to_dict(),
            'env': self.env_info,
            'seeds': self.seeds.report(),
            'provenance': provenance.snapshot(argv),
            'transfer': None,
            'freeze_events': [],
            'source': None,
            'reference': self._reference_info(),
        }

        self.start_episode = 0
        self._frozen: bool | None = None
        self._fp_at_freeze: dict | None = None
        self._diag_baseline: np.ndarray | None = None
        self._last_ckpt = time.time()

        self._install_diagnostic_states()
        if cfg.is_transfer:
            self._setup_transfer()
        self._maybe_resume()
        self._apply_freeze(initial=True)

    # ---- setup -----------------------------------------------------------
    def _reference_info(self) -> dict:
        """The normalisation constants, recorded so a score is recomputable."""
        try:
            ref = envs.reference(self.cfg.env)
            return {'env': self.cfg.env, 'random_return': ref['random_return'],
                    'threshold': ref['threshold'],
                    'noop_return': ref.get('noop_return')}
        except KeyError as exc:
            return {'env': self.cfg.env, 'error': str(exc)}

    def score(self, ret: float | None) -> float | None:
        if ret is None:
            return None
        try:
            return envs.normalised_score(self.cfg.env, ret)
        except (KeyError, ValueError):
            return None

    def _install_diagnostic_states(self) -> None:
        """A fixed state batch, drawn once, from its own stream and own env.

        The published loop drew its diagnostic states from the replay buffer,
        which advanced the training generator: turning diagnostics on changed
        every subsequent minibatch and hence the whole trajectory.
        """
        if not self.cfg.log_diagnostics or self.cfg.diag_states <= 0:
            return
        rng = self.seeds.rng('diag')
        env, _ = envs.make(self.cfg.env)
        states, ep = [], 0
        while len(states) < self.cfg.diag_states:
            state, _ = env.reset(seed=int(rng.integers(0, 2 ** 31 - 1)))
            for _ in range(self.cfg.max_steps):
                states.append(np.asarray(state, dtype=np.float32))
                if len(states) >= self.cfg.diag_states:
                    break
                state, _, term, trunc, _ = env.step(
                    int(rng.integers(self.agent.action_dim)))
                if term or trunc:
                    break
            ep += 1
            if ep > 500:                       # pathological safety valve
                break
        env.close()
        batch = np.asarray(states[:self.cfg.diag_states], dtype=np.float32)
        self.agent.set_diagnostic_states(batch)
        self.manifest['diagnostic_states'] = {
            'n': int(len(batch)),
            'policy': 'uniform_random',
            'hash': provenance.file_hash.__module__ and _array_hash(batch),
        }

    def _build_source_model(self):
        """Materialise the source network for whichever condition this is."""
        cfg = self.cfg
        src_spec = cfg.source_env_spec
        if src_spec is None:
            raise ValueError('a transfer condition requires source_env')
        src_obs, src_act = src_spec.obs_dim, src_spec.act_dim
        info: dict = {'condition': cfg.condition,
                      'source_env': src_spec.canonical(),
                      'source_obs_dim': src_obs, 'source_act_dim': src_act}

        if cfg.condition == 'transfer_untrained':
            # Distributionally equivalent to freezing a fresh initialisation,
            # which is exactly why it isolates protocol mechanics: identical
            # partial copy, identical head reinitialisation, identical freeze
            # window, no learned content.
            seed = self.seeds.control_seed('untrained')
            model = build_q_network(
                src_obs, src_act, cfg.arch, cfg.hidden, cfg.head_units,
                cfg.aggregation,
                Seeds(seed).layer_seeds(LAYER_GROUPS[cfg.arch]['all']))
            info['built'] = 'randomly initialised source'
            info['control_seed'] = int(seed)
            return model, info

        model = load_source(cfg.source_checkpoint)
        info['checkpoint'] = cfg.source_checkpoint
        info['source_fingerprint'] = weight_fingerprint(model)
        info['source_result'] = _read_source_result(cfg.source_checkpoint)

        if cfg.condition == 'transfer_permuted':
            layers = cfg.transfer_layers(src_obs, src_act,
                                         self.agent.state_dim,
                                         self.agent.action_dim)
            rng = np.random.default_rng(self.seeds.control_seed('permute'))
            if cfg.permute_kind == 'spectrum':
                model, rep = spectrum_matched_source(model, layers, rng)
            else:
                model, rep = permute_source(model, layers, rng,
                                            cfg.permute_scope)
            info['permutation'] = rep
            info['permute_kind'] = cfg.permute_kind
        return model, info

    def _setup_transfer(self) -> None:
        cfg = self.cfg
        source, info = self._build_source_model()
        src_spec = cfg.source_env_spec
        layers = cfg.transfer_layers(src_spec.obs_dim, src_spec.act_dim,
                                     self.agent.state_dim,
                                     self.agent.action_dim)
        report = transfer_weights(
            self.agent.online, source, layers,
            input_policy=cfg.input_policy, head_policy=cfg.head_policy,
            rng=np.random.default_rng(self.seeds.control_seed('embed')))
        summary = transfer_summary(report, self.agent.online)

        recal = None
        if cfg.value_recal != 'none':
            states = (self.agent._diag_states
                      if self.agent._diag_states is not None
                      else np.zeros((1, self.agent.state_dim), dtype=np.float32))
            recal = recalibrate_value_head(self.agent.online, states,
                                           cfg.value_recal)

        self.agent.target.set_weights(self.agent.online.get_weights())
        self.manifest['transfer'] = {
            'transfer_set': cfg.transfer_set,
            'layers_requested': list(layers),
            'input_policy': cfg.input_policy,
            'head_policy': cfg.head_policy,
            'report': report,
            'summary': summary,
            'value_recalibration': recal,
        }
        self.manifest['source'] = info

        # Source validity, on the normalised score of the *source* environment.
        src_result = (info.get('source_result') or {})
        src_score = src_result.get('final_score')
        gate = 0.6
        self.manifest['source']['validity'] = {
            'rule': 'normalised final score >= 0.6 on the source environment',
            'gate': gate,
            'source_final_score': src_score,
            'valid': (None if src_score is None else bool(src_score >= gate)),
            'note': ('untrained control: validity is not applicable, the source '
                     'is random by construction'
                     if cfg.condition == 'transfer_untrained' else None),
        }
        print(f'[transfer] {cfg.condition} from {info.get("source_env")} '
              f'set={cfg.transfer_set} '
              f'frac={summary["fraction_of_model_transferred"]:.3f}')
        for rec in report:
            print(f'  {rec["layer"]:11s} {rec["action"]:8s} '
                  f'{rec["params_copied"]:6d}/{rec["params_total"]:6d} '
                  f'{rec["detail"][:56]}')

    # ---- freeze schedule -------------------------------------------------
    def _should_freeze(self) -> bool:
        cfg = self.cfg
        if not cfg.is_transfer or cfg.freeze_updates == 0:
            return False
        if cfg.freeze_updates < 0:               # negative encodes "never unfreeze"
            return True
        return self.agent.update_counter < cfg.freeze_updates

    def _apply_freeze(self, initial: bool = False) -> None:
        want = self._should_freeze()
        if self._frozen == want and not initial:
            return
        layers = self.cfg.freeze_layers() if self.cfg.is_transfer else ()
        report = (self.agent.set_frozen(layers, want) if layers
                  else trainable_report(self.agent.online))
        fingerprint = weight_fingerprint(self.agent.online)

        intervention = None
        if self._frozen and not want:
            intervention = self._apply_plasticity_intervention()

        verdict = None
        if self._frozen and not want and self._fp_at_freeze is not None:
            # Leaving a frozen window: the layers declared frozen must not have
            # moved, and the layers left trainable must have. Both directions
            # matter -- an unchanged trainable layer usually means the optimiser
            # never received its gradients, which is what a positionally
            # resolved freeze produces silently.
            verdict = verify_freeze(self._fp_at_freeze, fingerprint, layers)

        # `report` carries its own per-layer dict under the key 'layers', so the
        # frozen *set* is recorded as `freeze_layers`. Splatting report over a
        # key of the same name silently replaced the set with the report dict,
        # losing the record of which layers the freeze actually targeted -- the
        # single most important fact in the event.
        event = {'kind': 'freeze', 'episode': self.start_episode,
                 'updates': self.agent.update_counter, 'frozen': want,
                 'freeze_group': self.cfg.freeze_group,
                 'freeze_layers': list(layers),
                 'verification': verdict,
                 'plasticity_intervention': intervention, **report}
        self.manifest['freeze_events'].append(event)
        self.events.append(event)
        self._frozen = want
        self._fp_at_freeze = fingerprint if want else None
        if layers:
            print(f'[freeze] updates={self.agent.update_counter} frozen={want} '
                  f'trainable={report["trainable_params"]} '
                  f'frozen_params={report["frozen_params"]}'
                  + (f' verify={verdict}' if verdict else ''))

    def _apply_plasticity_intervention(self) -> dict | None:
        """Reset the head, or shrink-and-perturb, at the unfreeze boundary.

        The plasticity-loss literature offers an architecture-free account of
        degradation after pretraining and two standard mitigations. Running them
        here is what turns that rival explanation from something the design
        merely measures into something it can test: if resetting the head
        recovers the loss, the mechanism is plasticity rather than transferred
        content, and no amount of representation-similarity argument overrides
        that.

        Fired exactly once, at the transition, and recorded in the freeze event.
        """
        cfg = self.cfg
        if not (cfg.reset_head_at_unfreeze or cfg.shrink_perturb > 0.0):
            return None
        from .networks import LAYER_GROUPS
        rng_seed = self.seeds.control_seed('plasticity')
        fresh = build_q_network(
            self.agent.state_dim, self.agent.action_dim, cfg.arch, cfg.hidden,
            cfg.head_units, cfg.aggregation,
            Seeds(rng_seed).layer_seeds(LAYER_GROUPS[cfg.arch]['all']))
        head = LAYER_GROUPS[cfg.arch]['head' if cfg.arch == 'mlp' else 'heads']
        record: dict = {'reset_head': bool(cfg.reset_head_at_unfreeze),
                        'shrink_perturb': float(cfg.shrink_perturb),
                        'layers_reset': [], 'layers_perturbed': []}
        lam = float(cfg.shrink_perturb)
        for layer in self.agent.online.layers:
            if not layer.weights:
                continue
            new = fresh.get_layer(layer.name).get_weights()
            if cfg.reset_head_at_unfreeze and layer.name in head:
                layer.set_weights(new)
                record['layers_reset'].append(layer.name)
                continue
            if lam > 0.0:
                cur = layer.get_weights()
                layer.set_weights([(1.0 - lam) * np.asarray(c) + lam * np.asarray(n)
                                   for c, n in zip(cur, new)])
                record['layers_perturbed'].append(layer.name)
        # The target network is resynchronised: leaving it holding pre-
        # intervention weights would make the TD target reference a network the
        # online net no longer resembles, which is a different experiment.
        self.agent.target.set_weights(self.agent.online.get_weights())
        self.agent._train_fn = None
        print(f'[plasticity] at unfreeze: reset={record["layers_reset"]} '
              f'perturbed={len(record["layers_perturbed"])} layers '
              f'lambda={lam}')
        return record

    # ---- checkpointing ---------------------------------------------------
    def _paths(self, sub: str = '') -> dict:
        base = os.path.join(self.dir, sub) if sub else self.dir
        return {'dir': base,
                'online': os.path.join(base, 'online.weights.h5'),
                'target': os.path.join(base, 'target.weights.h5'),
                'buffer': os.path.join(base, 'buffer.npz'),
                'optim': os.path.join(base, 'optimizer.npz'),
                'state': os.path.join(base, 'state.json')}

    def save_checkpoint(self, episode: int, sub: str = '') -> None:
        p = self._paths(sub)
        os.makedirs(p['dir'], exist_ok=True)
        self.agent.online.save_weights(p['online'])
        self.agent.target.save_weights(p['target'])
        self.agent.buffer.save(p['buffer'])
        opt = self.agent.optimizer_state()
        np.savez_compressed(p['optim'],
                            **{f'v{i}': v for i, v in enumerate(opt['values'])})
        with open(p['state'], 'w', encoding='utf-8') as fh:
            json.dump({'episode': episode,
                       'env_steps': self.agent.env_steps,
                       'update_counter': self.agent.update_counter,
                       'clip_events': self.agent.clip_events,
                       'frozen': self._frozen,
                       'trajectory_digest': self.cfg.trajectory_digest(),
                       'rng_states': self.seeds.rng_states()}, fh)
        self._last_ckpt = time.time()

    def _maybe_resume(self) -> None:
        p = self._paths()
        if not os.path.exists(p['state']):
            return
        with open(p['state'], encoding='utf-8') as fh:
            st = json.load(fh)

        stored = st.get('trajectory_digest')
        current = self.cfg.trajectory_digest()
        if stored and stored != current:
            raise RuntimeError(
                f'refusing to resume {self.dir}: the checkpoint was written '
                f'under a different training configuration '
                f'(trajectory digest {stored[:12]} != {current[:12]}). '
                f'Continuing would silently mix two configurations, which is '
                f'the error the Phase 0 audit spent days undoing. Delete the '
                f'directory to restart, or run the other configuration, which '
                f'has its own directory.')

        self.agent.online.load_weights(p['online'])
        self.agent.target.load_weights(p['target'])
        if os.path.exists(p['buffer']):
            self.agent.buffer.load(p['buffer'])
        if os.path.exists(p['optim']):
            d = np.load(p['optim'])
            self.agent.load_optimizer_state(
                {'values': [d[f'v{i}'] for i in range(len(d.files))]})
        else:
            raise RuntimeError(
                f'{self.dir}: checkpoint has no optimiser state. Resuming would '
                f'restart Adam from zero moments while claiming to continue the '
                f'same run. Delete the directory and restart.')
        self.seeds.restore_rng_states(st.get('rng_states') or {})
        self.agent.env_steps = int(st['env_steps'])
        self.agent.update_counter = int(st['update_counter'])
        self.agent.clip_events = int(st.get('clip_events', 0))
        self._frozen = st.get('frozen')
        self.start_episode = int(st['episode']) + 1

        # Idempotency: drop anything at or after the resume point before
        # appending, in both logs.
        dropped = self.metrics.truncate_from(self.start_episode)
        dropped_evals = self.evals.drop_where(
            lambda r: int(r.get('checkpoint', -10 ** 9)) >= self.start_episode)
        print(f'[resume] episode {self.start_episode} '
              f'(dropped {dropped} metric rows, {dropped_evals} eval rows)')

    # ---- linear probe ----------------------------------------------------
    def _probe_jumpstart(self) -> dict | None:
        """Fit only the output head on target transitions, then evaluate.

        With a reinitialised head the zero-shot greedy policy is an argmax over
        a random readout, so plain jumpstart sits at chance regardless of how
        good the transferred features are. This measures the quantity the
        question is actually about: whether the features carry information a
        linear readout can use. The trunk stays frozen, a separate optimiser is
        used, and the original weights are restored afterwards, so the training
        run is unaffected.
        """
        cfg = self.cfg
        if cfg.probe_steps <= 0 or cfg.probe_transitions <= 0:
            return None

        rng = self.seeds.rng('diag')
        env, _ = envs.make(cfg.env)
        S, A, R, NS, D = [], [], [], [], []
        while len(S) < cfg.probe_transitions:
            state, _ = env.reset(seed=int(rng.integers(0, 2 ** 31 - 1)))
            state = np.asarray(state, dtype=np.float32)
            for _ in range(cfg.max_steps):
                action = int(rng.integers(self.agent.action_dim))
                nxt, reward, term, trunc, _ = env.step(action)
                nxt = np.asarray(nxt, dtype=np.float32)
                S.append(state); A.append(action); R.append(float(reward))
                NS.append(nxt); D.append(float(term))
                state = nxt
                if term or trunc or len(S) >= cfg.probe_transitions:
                    break
        env.close()
        S = np.asarray(S, dtype=np.float32); NS = np.asarray(NS, dtype=np.float32)
        A = np.asarray(A, dtype=np.int32); R = np.asarray(R, dtype=np.float32)
        D = np.asarray(D, dtype=np.float32)

        snapshot = [w.copy() for w in self.agent.online.get_weights()]
        trainable_before = {l.name: l.trainable for l in self.agent.online.layers
                            if l.weights}
        head = LAYER_GROUPS[cfg.arch]['head' if cfg.arch == 'mlp' else 'heads']
        for layer in self.agent.online.layers:
            if layer.weights:
                layer.trainable = layer.name in head

        opt = keras.optimizers.Adam(learning_rate=cfg.lr)
        variables = self.agent.online.trainable_variables
        opt.build(variables)
        gamma = tf.constant(cfg.gamma, dtype=tf.float32)
        probe_rng = np.random.default_rng(self.seeds.control_seed('probe'))
        for _ in range(cfg.probe_steps):
            idx = probe_rng.integers(0, len(S), size=cfg.batch_size)
            nq = tf.reduce_max(self.agent.target(NS[idx], training=False), axis=1)
            targets = tf.stop_gradient(
                tf.constant(R[idx]) + (1.0 - tf.constant(D[idx])) * gamma * nq)
            act_idx = tf.stack([tf.range(len(idx)),
                                tf.constant(A[idx], dtype=tf.int32)], axis=1)
            with tf.GradientTape() as tape:
                q = self.agent.online(tf.constant(S[idx]), training=True)
                loss = tf.reduce_mean(
                    tf.square(targets - tf.gather_nd(q, act_idx)))
            opt.apply_gradients(zip(tape.gradient(loss, variables), variables))

        ret, _rows = evaluate(self.eval_env, self.agent,
                              min(cfg.final_eval_episodes, 30), self.seeds,
                              'eval_final', -2, cfg.max_steps)

        self.agent.online.set_weights(snapshot)
        for layer in self.agent.online.layers:
            if layer.weights:
                layer.trainable = trainable_before[layer.name]
        self.agent._train_fn = None
        return {'probe_steps': cfg.probe_steps,
                'probe_transitions': int(len(S)),
                'head_layers': list(head),
                'return': ret, 'score': self.score(ret),
                'eval_episodes': min(cfg.final_eval_episodes, 30),
                'note': 'trunk frozen, head refit, weights restored afterwards'}

    # ---- main loop -------------------------------------------------------
    def run(self) -> dict:
        cfg = self.cfg
        started = time.time()

        final_eval_episodes = {cfg.num_episodes - 1 - j * cfg.eval_every
                               for j in range(cfg.final_eval_checkpoints)}
        final_eval_episodes = {e for e in final_eval_episodes if e >= 0}
        prefix_eval_episodes = set(cfg.prefix_checkpoints)
        held_out: dict[int, float] = {}

        if self.start_episode == 0:
            ret, rows = evaluate(self.eval_env, self.agent,
                                 cfg.final_eval_episodes, self.seeds,
                                 'eval_final', -1, cfg.max_steps)
            self.evals.extend(rows)
            self.manifest['jumpstart'] = {
                'return': ret, 'score': self.score(ret),
                'episodes': cfg.final_eval_episodes,
                'interpretable': bool(
                    not cfg.is_transfer
                    or not (self.manifest.get('transfer') or {})
                    .get('summary', {}).get('layers_reinit')),
                'note': ('a reinitialised output head makes the zero-shot '
                         'greedy policy an argmax over a random readout, so '
                         'this is at chance by construction; use '
                         'probe_jumpstart instead'),
            }
            probe = self._probe_jumpstart()
            if probe:
                self.manifest['probe_jumpstart'] = probe
            feats = self.agent.trunk_features()
            self._diag_baseline = feats if feats is not None else None

        try:
            for episode in range(self.start_episode, cfg.num_episodes):
                state, _ = self.env.reset(
                    seed=int(self.seeds.episode_seed(episode)) % (2 ** 31 - 1))
                state = np.asarray(state, dtype=np.float32)
                total, steps = 0.0, 0
                losses, gnorms, tds, qs = [], [], [], []
                eps_at_start = self.agent.epsilon

                for steps in range(1, cfg.max_steps + 1):
                    action = self.agent.act(state)
                    nxt, reward, term, trunc, _ = self.env.step(action)
                    nxt = np.asarray(nxt, dtype=np.float32)
                    self.agent.buffer.add(state, action, reward, nxt, term)
                    state = nxt
                    total += float(reward)
                    self.agent.env_steps += 1

                    if self.agent.env_steps % cfg.train_every == 0:
                        out = self.agent.train_step()
                        if out:
                            losses.append(out['loss'])
                            gnorms.append(out['grad_norm'])
                            tds.append(out['td_error_abs'])
                            qs.append(out['q_mean'])
                            # Checked per update, not per episode: the boundary
                            # is defined in updates, so an episode-granular
                            # check would place it up to 1000 updates late.
                            self._apply_freeze()
                    if term or trunc:
                        break

                row = {'episode': episode, 'return': total,
                       'score': self.score(total), 'length': steps,
                       'epsilon': eps_at_start,
                       'env_steps': self.agent.env_steps,
                       'updates': self.agent.update_counter,
                       'clip_events': self.agent.clip_events,
                       'loss': float(np.mean(losses)) if losses else None,
                       'grad_norm': float(np.mean(gnorms)) if gnorms else None,
                       'td_error_abs': float(np.mean(tds)) if tds else None,
                       'q_mean': float(np.mean(qs)) if qs else None,
                       'frozen': bool(self._frozen),
                       'wall_time': round(time.time() - started, 1)}

                if episode % cfg.eval_every == 0:
                    ret, rows = evaluate(self.eval_env, self.agent,
                                         cfg.eval_episodes, self.seeds,
                                         'eval_monitor', episode, cfg.max_steps)
                    self.evals.extend(rows)
                    row['eval_return'] = ret
                    row['eval_score'] = self.score(ret)
                    if cfg.log_diagnostics:
                        row.update(self.agent.stream_magnitudes())
                        row.update(self.agent.stream_gradient_norms())
                        row['dead_unit_frac'] = self.agent.dead_unit_fraction()
                        row.update(self.agent.plasticity_signals())
                        feats = self.agent.trunk_features()
                        if feats is not None and self._diag_baseline is not None:
                            row['cka_drift'] = linear_cka(self._diag_baseline,
                                                          feats)
                    print(f'ep {episode:4d}/{cfg.num_episodes}  '
                          f'R={total:8.2f} eval={ret:8.2f} '
                          f'score={row["eval_score"] if row["eval_score"] is None else round(row["eval_score"], 3)} '
                          f'eps={eps_at_start:.3f} upd={self.agent.update_counter}',
                          flush=True)

                if episode in final_eval_episodes or episode in prefix_eval_episodes:
                    ret, rows = evaluate(self.eval_env, self.agent,
                                         cfg.final_eval_episodes, self.seeds,
                                         'eval_final', episode, cfg.max_steps)
                    self.evals.extend(rows)
                    held_out[episode] = ret
                    row['held_out_return'] = ret
                    row['held_out_score'] = self.score(ret)
                    if episode in prefix_eval_episodes:
                        self.save_checkpoint(episode, sub=f'ckpt_ep{episode}')

                self.metrics.append(row)
                if time.time() - self._last_ckpt > cfg.checkpoint_seconds:
                    self.save_checkpoint(episode)
        finally:
            self.metrics.close()

        self.save_checkpoint(cfg.num_episodes - 1)
        self.agent.online.save(os.path.join(self.dir, 'model.keras'))
        self._finalise(held_out, started, final_eval_episodes)

        if not cfg.keep_buffer:
            # ~5 MB per run, and a completed run has no use for it. At the
            # catalogue's scale this is the difference between ~10 GB and ~1 GB.
            for sub in ('', *[f'ckpt_ep{e}' for e in cfg.prefix_checkpoints]):
                path = self._paths(sub)['buffer']
                if os.path.exists(path):
                    os.remove(path)

        self.env.close()
        self.eval_env.close()
        return self.manifest

    def _finalise(self, held_out: dict, started: float,
                  final_eval_episodes: set) -> None:
        cfg = self.cfg
        integrity = self.metrics.check(expected=cfg.num_episodes)
        df = self.metrics.as_dataframe()

        finals = [held_out[e] for e in sorted(final_eval_episodes)
                  if e in held_out]
        final_return = float(np.mean(finals)) if finals else None
        prefix = {int(e): {'return': held_out[e], 'score': self.score(held_out[e])}
                  for e in sorted(cfg.prefix_checkpoints) if e in held_out}

        result = {
            'episodes_completed': int(integrity['unique_episodes']),
            'metrics_integrity': integrity,
            'env_steps': int(self.agent.env_steps),
            'updates': int(self.agent.update_counter),
            'clip_fraction': (self.agent.clip_events / self.agent.update_counter
                              if self.agent.update_counter else None),
            'final_eval_episodes': sorted(final_eval_episodes),
            'final_return': final_return,
            'final_score': self.score(final_return),
            'final_return_per_checkpoint': {int(k): v
                                            for k, v in sorted(held_out.items())
                                            if k in final_eval_episodes},
            'prefix_evaluations': prefix,
            'wall_time_s': round(time.time() - started, 1),
        }
        if not df.empty:
            result.update(_curve_summaries(df, cfg))
        self.manifest['result'] = result

        with open(os.path.join(self.dir, 'manifest.json'), 'w',
                  encoding='utf-8') as fh:
            json.dump(self.manifest, fh, indent=2, default=str)
        if not integrity['contiguous']:
            print(f'[WARNING] metrics integrity: {integrity["problems"]}')


def _curve_summaries(df, cfg: Config) -> dict:
    """AUC over env steps, convergence slope, and the dispersion metrics.

    AUC is integrated over **env steps** rather than episodes because episode
    length is performance-dependent, so an episode-indexed area silently weights
    arms differently.
    """
    ev = df.dropna(subset=['eval_score']) if 'eval_score' in df else df.iloc[0:0]
    out: dict = {}
    if len(ev) >= 2:
        x = ev['env_steps'].to_numpy(dtype=float)
        y = ev['eval_score'].to_numpy(dtype=float)
        span = x[-1] - x[0]
        out['auc_score'] = float(np.trapezoid(y, x) / span) if span > 0 else None
        tail = ev[ev['episode'] >= max(0, cfg.num_episodes - 200)]
        if len(tail) >= 3:
            slope, _ = np.polyfit(tail['episode'].to_numpy(dtype=float),
                                  tail['eval_score'].to_numpy(dtype=float), 1)
            out['convergence_slope_per_episode'] = float(slope)
            out['convergence_window_episodes'] = 200
        last10 = ev['eval_score'].tail(10).to_numpy(dtype=float)
        if len(last10) >= 2:
            out['within_run_sd'] = float(np.std(last10, ddof=1))
    if 'length' in df:
        out['episode_length_final100'] = float(df['length'].tail(100).mean())
    if 'loss' in df:
        losses = df['loss'].dropna()
        out['td_loss_final100'] = (float(losses.tail(100).mean())
                                   if len(losses) else None)
        out['td_loss_window'] = 'mean over the final 100 episodes'
    return out


def _array_hash(arr: np.ndarray) -> str:
    import hashlib
    return hashlib.blake2b(np.ascontiguousarray(arr, dtype=np.float32).tobytes(),
                           digest_size=8).hexdigest()


def _read_source_result(checkpoint: str | None) -> dict | None:
    """The source run's own result, read from the manifest beside its model."""
    if not checkpoint:
        return None
    manifest = os.path.join(os.path.dirname(checkpoint), 'manifest.json')
    try:
        with open(manifest, encoding='utf-8') as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    res = data.get('result') or {}
    return {'final_score': res.get('final_score'),
            'final_return': res.get('final_return'),
            'run_digest': (data.get('identity') or {}).get('run_digest'),
            'env': (data.get('config') or {}).get('env')}


def train(cfg: Config, argv: list[str] | None = None) -> dict:
    return Trainer(cfg, argv).run()


__all__ = ['Trainer', 'train', 'evaluate']
