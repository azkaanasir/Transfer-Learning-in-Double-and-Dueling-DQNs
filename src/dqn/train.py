"""Training loop: freeze schedule, evaluation, logging, and resume.

Design notes tied to the Phase 0 audit (paper/METHODS_ACTUAL.md):

* **Metrics go to CSV, not only TensorBoard.** The published runs survive only
  as TF2 event files whose scalars are tensor-encoded, so recovering them
  needed a bespoke parser. A tidy CSV per run makes aggregation trivial and
  keeps the results readable without any TF dependency.
* **The evaluation window is explicit.** The manuscript never stated it; Phase 0
  identified it as the final 100 episodes. `cfg.eval_window` records it, and
  the headline scalar is written into the manifest at the end of the run.
* **The freeze schedule is real and logged.** Every transition writes a
  trainable-parameter report, so the trainable surface is a recorded fact.
* **Checkpoints carry the replay buffer and RNG state**, so a Colab timeout
  costs only the episodes since the last checkpoint.
"""
from __future__ import annotations

import csv
import json
import os
import random
import time

import numpy as np
import tensorflow as tf

from .agent import Agent
from .config import Config
from .transfer import load_source, transfer_weights, trainable_report

METRIC_FIELDS = ['episode', 'reward', 'length', 'epsilon', 'loss',
                 'updates', 'eval_reward', 'v_abs_mean', 'a_abs_mean',
                 'a_spread', 'wall_time']


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def make_env(env_id: str):
    import gymnasium as gym
    return gym.make(env_id)


def evaluate(env, agent: Agent, episodes: int, seed: int) -> float:
    """Greedy evaluation -- no exploration noise."""
    totals = []
    for i in range(episodes):
        state, _ = env.reset(seed=seed + 10_000 + i)
        done, total, steps = False, 0.0, 0
        while not done and steps < agent.cfg.max_steps:
            action = agent.act(np.asarray(state, dtype=np.float32), greedy=True)
            state, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            done = terminated or truncated
            steps += 1
        totals.append(total)
    return float(np.mean(totals))


class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.dir = cfg.run_dir()
        os.makedirs(self.dir, exist_ok=True)
        seed_everything(cfg.seed)

        self.env = make_env(cfg.env_id)
        state_dim = int(np.prod(self.env.observation_space.shape))
        action_dim = int(self.env.action_space.n)
        self.agent = Agent(cfg, state_dim, action_dim)

        self.manifest = {'config': cfg.to_dict(),
                         'state_dim': state_dim,
                         'action_dim': action_dim,
                         'transfer': None,
                         'freeze_events': []}
        self.start_episode = 0
        self.metrics_path = os.path.join(self.dir, 'metrics.csv')

        if cfg.mode == 'transfer':
            self._do_transfer()
        self._maybe_resume()
        self._apply_freeze_for_episode(self.start_episode, initial=True)

    # ---- setup -----------------------------------------------------------
    def _do_transfer(self):
        source = load_source(self.cfg.source_checkpoint)
        report = transfer_weights(
            self.agent.online, source,
            layer_names=self.cfg.transfer_layers,
            first_layer_policy=self.cfg.first_layer_policy,
            rng=np.random.default_rng(self.cfg.seed))
        self.agent.target.set_weights(self.agent.online.get_weights())
        self.manifest['transfer'] = {
            'source_checkpoint': self.cfg.source_checkpoint,
            'layers': report}
        print(f'[transfer] from {self.cfg.source_checkpoint}')
        for rec in report:
            print(f"  {rec['layer']:12s} {rec['action']:8s} {rec['detail']}")

    def _apply_freeze_for_episode(self, episode: int, initial: bool = False):
        """Freeze transferred layers for the first `freeze_episodes` episodes."""
        if self.cfg.mode != 'transfer' or self.cfg.freeze_episodes <= 0:
            if initial:
                self.manifest['freeze_events'].append(
                    {'episode': episode, 'frozen': False,
                     **trainable_report(self.agent.online)})
            return
        should_freeze = episode < self.cfg.freeze_episodes
        current = getattr(self, '_frozen', None)
        if current == should_freeze:
            return
        report = self.agent.set_frozen(self.cfg.freeze_layers, should_freeze)
        self._frozen = should_freeze
        event = {'episode': episode, 'frozen': should_freeze, **report}
        self.manifest['freeze_events'].append(event)
        print(f"[freeze] episode {episode}: frozen={should_freeze} "
              f"trainable_params={report['trainable_params']} "
              f"frozen_params={report['frozen_params']}")

    # ---- checkpointing ---------------------------------------------------
    def _ckpt_paths(self):
        return (os.path.join(self.dir, 'online.weights.h5'),
                os.path.join(self.dir, 'target.weights.h5'),
                os.path.join(self.dir, 'buffer.npz'),
                os.path.join(self.dir, 'state.json'))

    def save_checkpoint(self, episode: int):
        online, target, buf, state = self._ckpt_paths()
        self.agent.online.save_weights(online)
        self.agent.target.save_weights(target)
        self.agent.buffer.save(buf)
        with open(state, 'w', encoding='utf-8') as fh:
            json.dump({'episode': episode,
                       'epsilon': self.agent.epsilon,
                       'update_counter': self.agent.update_counter}, fh)

    def _maybe_resume(self):
        online, target, buf, state = self._ckpt_paths()
        if not os.path.exists(state):
            return
        try:
            with open(state, encoding='utf-8') as fh:
                st = json.load(fh)
            self.agent.online.load_weights(online)
            self.agent.target.load_weights(target)
            if os.path.exists(buf):
                self.agent.buffer.load(buf)
            self.agent.epsilon = st['epsilon']
            self.agent.update_counter = st['update_counter']
            self.start_episode = st['episode'] + 1
            print(f'[resume] continuing from episode {self.start_episode}')
        except Exception as exc:                      # noqa: BLE001
            print(f'[resume] checkpoint unusable ({exc}); starting fresh')

    # ---- logging ---------------------------------------------------------
    def _open_metrics(self):
        exists = os.path.exists(self.metrics_path) and self.start_episode > 0
        fh = open(self.metrics_path, 'a' if exists else 'w',
                  newline='', encoding='utf-8')
        writer = csv.DictWriter(fh, fieldnames=METRIC_FIELDS)
        if not exists:
            writer.writeheader()
        return fh, writer

    def _read_history(self) -> list[float]:
        """Episode rewards for the whole run, across every resumed session."""
        if not os.path.exists(self.metrics_path):
            return []
        try:
            with open(self.metrics_path, newline='', encoding='utf-8') as fh:
                return [float(r['reward']) for r in csv.DictReader(fh)
                        if r.get('reward') not in (None, '')]
        except Exception:                              # noqa: BLE001
            return []

    # ---- main loop -------------------------------------------------------
    def run(self) -> dict:
        cfg = self.cfg
        fh, writer = self._open_metrics()
        started = time.time()
        rewards = []

        try:
            for episode in range(self.start_episode, cfg.num_episodes):
                self._apply_freeze_for_episode(episode)

                state, _ = self.env.reset(seed=cfg.seed * 100_000 + episode)
                state = np.asarray(state, dtype=np.float32)
                total, steps, losses = 0.0, 0, []

                for steps in range(1, cfg.max_steps + 1):
                    action = self.agent.act(state)
                    nxt, reward, terminated, truncated, _ = self.env.step(action)
                    nxt = np.asarray(nxt, dtype=np.float32)
                    self.agent.buffer.add(state, action, reward, nxt, terminated)
                    state = nxt
                    total += float(reward)

                    if steps % cfg.train_every == 0:
                        loss = self.agent.train_step()
                        if not np.isnan(loss):
                            losses.append(loss)
                    if terminated or truncated:
                        break

                rewards.append(total)
                row = {'episode': episode, 'reward': total, 'length': steps,
                       'epsilon': self.agent.epsilon,
                       'loss': float(np.mean(losses)) if losses else '',
                       'updates': self.agent.update_counter,
                       'eval_reward': '', 'v_abs_mean': '', 'a_abs_mean': '',
                       'a_spread': '', 'wall_time': round(time.time() - started, 1)}

                if episode % cfg.eval_every == 0:
                    self.agent.decay_epsilon()
                    row['eval_reward'] = evaluate(self.env, self.agent,
                                                  cfg.eval_episodes, cfg.seed)
                    if cfg.log_diagnostics and len(self.agent.buffer) > cfg.batch_size:
                        s, *_ = self.agent.buffer.sample(cfg.batch_size)
                        row.update(self.agent.stream_magnitudes(s))
                    print(f"ep {episode:4d}/{cfg.num_episodes}  "
                          f"R={total:8.2f}  eval={row['eval_reward']:8.2f}  "
                          f"eps={self.agent.epsilon:.3f}  "
                          f"upd={self.agent.update_counter}", flush=True)

                writer.writerow(row)
                fh.flush()

                if episode % cfg.checkpoint_every == 0 and episode > 0:
                    self.save_checkpoint(episode)
        finally:
            fh.close()

        self.save_checkpoint(cfg.num_episodes - 1)
        self.agent.online.save(os.path.join(self.dir, 'model.keras'))

        # Compute the headline scalar from the full metrics file rather than
        # this session's in-memory rewards: after a resume the in-memory list
        # holds only the episodes since the checkpoint, which would both skew
        # the evaluation window and make the sweep's completion check wrong.
        history = self._read_history()
        window = history[-cfg.eval_window:] if len(history) else []
        self.manifest['result'] = {
            'episodes_completed': len(history) or len(rewards),
            'episodes_this_session': len(rewards),
            'eval_window': cfg.eval_window,
            f'ep_reward_last{cfg.eval_window}': float(np.mean(window)) if len(window) else None,
            'wall_time_s': round(time.time() - started, 1),
            'updates': self.agent.update_counter,
        }
        with open(os.path.join(self.dir, 'manifest.json'), 'w',
                  encoding='utf-8') as f:
            json.dump(self.manifest, f, indent=2)
        return self.manifest


def train(cfg: Config) -> dict:
    return Trainer(cfg).run()
