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

import h5py
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


# ---------------------------------------------------------------------------
# Atomic checkpointing
# ---------------------------------------------------------------------------
# A checkpoint is five artefacts, and the published writer emitted them in
# sequence with neither atomicity nor a consistency marker. `open(state, 'w')`
# truncates in place, so a kill a millisecond later left an empty `state.json`
# and the next resume died inside `json.load`. That was the visible failure.
# The silent one is worse: a kill between the replay archive and the state file
# left weights, replay and optimiser from checkpoint N+1 beside a `state.json`
# from checkpoint N, and `_maybe_resume` then restored episode N into a network
# that already held N+1's parameters, replayed those episodes with advanced
# weights, and produced a wrong trajectory with nothing raised. The existing
# `trajectory_digest` guard cannot see it: it compares the *configuration*, not
# the *progress*.
#
# What replaces it is a three-phase commit over a generation-stamped set.
#
#   1. STAGE   Every artefact is written into `<dir>/.ckpt_staging` with the
#              checkpoint's GENERATION stamped inside it, and flushed to the
#              device. Nothing already committed is touched, and this is where
#              essentially all of the wall clock goes: measured on the campaign
#              machine, 1.45 s of the 1.46 s a full LunarLander checkpoint
#              costs. So a kill anywhere in phase 1 leaves the previous
#              generation exactly as it was, which is the opposite of the
#              published writer, where a kill during the write destroyed the
#              checkpoint being overwritten as well as the one being made.
#   2. COMMIT  Each staged file is moved into place with `os.replace`, which is
#              atomic on Windows and POSIX for a same-volume rename, and
#              `state.json` goes LAST. `state.json` is the pointer: until it
#              names generation N, generation N does not exist.
#   3. CLEAN   The staging directory is removed.
#
# The generation is stamped INSIDE each artefact -- an h5 root attribute for
# the two Keras weight files, an extra array in the two NumPy archives -- and
# not into a marker file beside it. A marker is a second file, so a commit
# killed between an artefact and its marker leaves one of the two stale; both
# orders fail, and a size check cannot separate them because successive
# checkpoints of the same network are byte-for-byte the same length. Stamping
# the artefact makes the artefact and its generation a single `os.replace`, and
# the set is consistent exactly when all five stamps agree.
#
# Phase 2 is the only window in which the committed set can be mixed, and it is
# five renames wide -- measured at 5.8 ms for the whole set, of which 2.5 ms is
# the 6.3 MB replay archive -- against a phase 1 of about 1.45 s. Even that
# window is not merely detectable but repairable: every artefact of generation
# N is still on disk, committed or staged, so `_recover_checkpoint` finishes
# the commit rather than discarding the checkpoint. Only a set that can be
# neither completed nor verified is refused, and it is refused loudly. One lost
# run costs about half an hour; one silently mixed run costs the result.
#
# The whole writer costs +331 ms per checkpoint against the published one
# (1.46 s against 1.13 s, medians of 13 interleaved pairs on a loaded
# machine). Almost none of that is the staging or the renames, which measure at
# the noise floor; it is the two fsyncs' and the two h5 stamps' reopening of a
# just-written file, which on this Windows volume costs about 20 ms a time
# whatever is then done with the handle. In run terms it is +1.7 s on a
# ~1900 s run, or 0.09 per cent, which is why it does not move the cadence.
#
# Disk cost: one extra copy of the checkpoint while it is staged, so the peak
# is 2x, about 13.9 MB against the 6.9 MB steady state (measured at full replay
# occupancy: 6.35 MB of it is the replay archive), released at phase 3. Six
# concurrent runs peak at about 83 MB. The catalogue's roughly 1200 stored
# checkpoints are unaffected: a staging directory never outlives a checkpoint,
# the replay archive is reaped when the run finishes, and the stamps themselves
# are eight bytes each.

#: Staged into `<dir>/.ckpt_staging`, then renamed into place one by one.
CHECKPOINT_ARTEFACTS: tuple[str, ...] = ('online', 'target', 'buffer', 'optim')
#: Deliberately not named `state.json`: `audit.py` and `aggregate.py` treat any
#: directory holding a `state.json` as a run directory, so a staging directory
#: left behind by a kill would otherwise be counted as a run of its own.
STAGED_STATE_NAME = 'state.json.staged'
STAGING_DIRNAME = '.ckpt_staging'
#: The h5 attribute and the NumPy archive key the generation is stamped under.
GENERATION_KEY = 'ckpt_generation'


def _fsync_file(path: str) -> None:
    """Force one file's bytes to the device before it is renamed into place.

    `os.replace` orders the *rename*; on its own it does not order the data
    behind it, so a power event can leave the new name pointing at bytes that
    never landed. Measured cost over the four staged artefacts: +222 ms of the
    checkpoint's 1.46 s, which is 15 per cent of a checkpoint and 0.06 per cent
    of a run. Almost all of it is the reopen rather than the flush -- `fsync`
    itself measures 0.8 ms on the small artefacts and 4.9 ms on the 6.3 MB
    replay archive, while the first open of a just-written file on this Windows
    volume costs about 20 ms whatever is done with the handle. Worth paying:
    the campaign's stated failure modes include a power event, and this is the
    only part of the writer that addresses one.

    Windows refuses `os.fsync` on a read-only descriptor, so the file is opened
    for writing.
    """
    try:
        fd = os.open(path, os.O_RDWR | getattr(os, 'O_BINARY', 0))
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _fsync_dir(path: str) -> None:
    """Flush the directory entry itself; a no-op where the platform forbids it.

    POSIX needs this for a rename to survive a power event. Windows has no
    portable equivalent and `os.open` on a directory raises there, so the
    failure is swallowed rather than reported as a checkpoint error.
    """
    try:
        fd = os.open(path, getattr(os, 'O_DIRECTORY', 0) | os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _write_json_atomic(path: str, payload, **dump_kwargs) -> None:
    """Write a JSON document that can never be observed torn or empty.

    `open(path, 'w')` truncates immediately, so a kill before the dump finishes
    leaves a zero-byte file where the caller believes a document is. The bytes
    go to a fixed temporary name in the same directory -- fixed, so that an
    interrupted write is overwritten by the next one rather than accumulating
    -- are flushed to the device, and are then moved into place by a
    same-volume `os.replace`.
    """
    directory = os.path.dirname(path) or '.'
    tmp = os.path.join(directory, f'.{os.path.basename(path)}.tmp')
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, **dump_kwargs)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    _fsync_dir(directory)


def _read_json_file(path: str) -> tuple[dict | None, str | None]:
    """Parse a JSON document, reporting *why* it could not be read.

    A torn `state.json` used to reach `json.load` and end the run with a
    `JSONDecodeError` traceback. The caller has to tell "absent" (a fresh run,
    proceed) from "present but unreadable" (refuse), so the reason is returned
    rather than raised.
    """
    if not os.path.exists(path):
        return None, 'absent'
    try:
        size = os.path.getsize(path)
        with open(path, encoding='utf-8') as fh:
            text = fh.read()
    except OSError as exc:
        return None, f'unreadable ({exc})'
    if not text.strip():
        return None, f'empty ({size} bytes)'
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f'truncated or not valid JSON at {size} bytes ({exc})'
    if not isinstance(data, dict):
        return None, f'holds a {type(data).__name__}, not an object'
    return data, None


def _stamp_weights_file(path: str, generation: int) -> None:
    """Record the generation inside a Keras weights file.

    Keras owns the format, so the stamp is added by reopening the file and
    setting a root attribute. Measured at +163 ms for the pair against a
    checkpoint of 1.46 s, nearly all of it the reopen rather than the eight
    bytes written. A content hash recorded in `state.json` would be cheaper and
    would detect more, but it could not answer the question the refusal message
    has to answer -- *which* checkpoint this file came from -- and the stamp
    also survives the file being copied out of the directory. `load_weights`
    ignores unknown root attributes; `test_resume_equivalence` is what checks
    that, since resume would not stay bitwise otherwise.
    """
    with h5py.File(path, 'a') as fh:
        fh.attrs[GENERATION_KEY] = int(generation)


def _artefact_state(kind: str, path: str) -> dict:
    """What is on disk for one artefact: its size and the generation it carries.

    Never raises. Everything reported is either a fact about the file or a
    named reason the file cannot be trusted, because the caller's job is to
    write a refusal message, not to stack a second traceback on the first. The
    two readers below are given a broad `except` for the same reason: h5py and
    NumPy signal a damaged archive with half a dozen different exception types
    and the distinction is of no use to the reader of the message.
    """
    absent = {'present': False, 'generation': None, 'bytes': None,
              'note': 'absent'}
    if not os.path.exists(path):
        return absent
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        return dict(absent, present=True, note=f'unreadable ({exc})')
    try:
        if kind in ('online', 'target'):
            with h5py.File(path, 'r') as fh:
                raw = fh.attrs.get(GENERATION_KEY)
        else:
            with np.load(path) as data:
                raw = (data[GENERATION_KEY] if GENERATION_KEY in data.files
                       else None)
    except Exception as exc:                              # noqa: BLE001
        return {'present': True, 'generation': None, 'bytes': size,
                'note': f'unreadable ({type(exc).__name__}: {exc})'}
    if raw is None:
        return {'present': True, 'generation': None, 'bytes': size,
                'note': f'carries no {GENERATION_KEY} stamp'}
    try:
        gen = int(np.asarray(raw).reshape(()).item())
    except (TypeError, ValueError):
        return {'present': True, 'generation': None, 'bytes': size,
                'note': f'{GENERATION_KEY} is {raw!r}, not an integer'}
    return {'present': True, 'generation': gen, 'bytes': size, 'note': None}


def _artefact_ok(found: dict, generation, expect_bytes) -> bool:
    """Is this copy a sound member of generation `generation`?"""
    return bool(found['present'] and found['note'] is None
                and found['generation'] == generation
                and (expect_bytes is None or found['bytes'] == expect_bytes))


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
        # The episode currently executing. The freeze schedule is indexed in
        # gradient updates, but the *event* also records the episode it fired
        # in, and recording `start_episode` there would have labelled every
        # mid-run transition with the episode the session began at.
        self._episode = 0
        self._frozen: bool | None = None
        self._fp_at_freeze: dict | None = None
        self._diag_baseline: np.ndarray | None = None
        self._last_ckpt = time.time()
        # Highest checkpoint generation committed under each checkpoint
        # directory this process has written. Consulted alongside what is on
        # disk, never instead of it: a resumed process starts with none.
        self._generations: dict = {}

        self._install_diagnostic_states()
        if cfg.is_transfer:
            self._setup_transfer()
        self._maybe_resume()
        self._episode = self.start_episode
        self.agent.episode = self.start_episode
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
        event = {'kind': 'freeze', 'episode': self._episode,
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
                'state': os.path.join(base, 'state.json'),
                'stage': os.path.join(base, STAGING_DIRNAME)}

    def _write_artefact(self, kind: str, path: str, generation: int) -> None:
        """Write one artefact to `path`, with `generation` stamped inside it.

        One dispatch, so the staged write cannot drift from the committed one,
        and so a test can make exactly one artefact fail at exactly one
        boundary. Keras insists a weights file end in `.weights.h5` and NumPy
        appends `.npz` to anything that does not already end in it, which is
        why staging is a directory of correctly named files rather than a set
        of `*.tmp` siblings.
        """
        if kind == 'online':
            self.agent.online.save_weights(path)
            _stamp_weights_file(path, generation)
        elif kind == 'target':
            self.agent.target.save_weights(path)
            _stamp_weights_file(path, generation)
        elif kind == 'buffer':
            self.agent.buffer.save(path, generation=generation)
        elif kind == 'optim':
            opt = self.agent.optimizer_state()
            np.savez_compressed(
                path,
                **{GENERATION_KEY: np.int64(generation)},
                **{f'v{i}': v for i, v in enumerate(opt['values'])})
        else:
            raise KeyError(f'unknown checkpoint artefact {kind!r}')

    def _checkpoint_state(self, episode: int, generation: int,
                          sizes: dict) -> dict:
        """The state document, which is also the checkpoint's pointer.

        It is renamed into place last and it names the generation every other
        artefact of the set must carry, so the set it describes either exists
        entirely or does not exist at all. The recorded sizes are a second,
        cheaper guard: a stamp can be read out of an archive whose payload was
        truncated after it.
        """
        return {'episode': episode,
                'env_steps': self.agent.env_steps,
                'update_counter': self.agent.update_counter,
                'clip_events': self.agent.clip_events,
                'frozen': self._frozen,
                'trajectory_digest': self.cfg.trajectory_digest(),
                'rng_states': self.seeds.rng_states(),
                'generation': generation,
                'checkpoint_set': {'generation': generation,
                                   'artefacts': dict(sizes),
                                   'buffer_reaped': False}}

    def _next_generation(self, p: dict) -> int:
        """One past the highest generation this directory has ever held.

        Read off disk as well as out of memory: a resumed process starts with
        no memory of what the previous one wrote, and a generation that went
        backwards would let a stale artefact pass for a current one.
        """
        seen = self._generations.get(p['dir'], 0)
        for path in (p['state'], os.path.join(p['stage'], STAGED_STATE_NAME)):
            doc, _why = _read_json_file(path)
            if doc is not None and isinstance(doc.get('generation'), int):
                seen = max(seen, int(doc['generation']))
        for kind in CHECKPOINT_ARTEFACTS:
            found = _artefact_state(kind, p[kind])
            if found['generation'] is not None:
                seen = max(seen, int(found['generation']))
        return seen + 1

    def save_checkpoint(self, episode: int, sub: str = '') -> None:
        """Write a checkpoint that is either wholly present or wholly absent.

        Phase 1 stages the whole generation-stamped set and touches nothing
        already committed. Phase 2 renames the set into place with `state.json`
        last. Phase 3 sweeps the staging directory up. See the module-level
        note above `CHECKPOINT_ARTEFACTS` for why, and for what a kill in each
        phase costs.
        """
        p = self._paths(sub)
        os.makedirs(p['dir'], exist_ok=True)
        generation = self._next_generation(p)

        # -- phase 1. About 99 per cent of the cost and none of the risk: a
        # kill anywhere in here loses the new checkpoint and nothing else.
        shutil.rmtree(p['stage'], ignore_errors=True)
        os.makedirs(p['stage'], exist_ok=True)
        sizes: dict = {}
        for kind in CHECKPOINT_ARTEFACTS:
            name = os.path.basename(p[kind])
            staged = os.path.join(p['stage'], name)
            self._write_artefact(kind, staged, generation)
            _fsync_file(staged)
            sizes[name] = os.path.getsize(staged)
        # Written last, and atomically: its presence with a parsable generation
        # is the whole of the evidence that phase 1 completed, and therefore
        # the whole of the licence to roll a half-finished commit forward.
        _write_json_atomic(os.path.join(p['stage'], STAGED_STATE_NAME),
                           self._checkpoint_state(episode, generation, sizes))

        # -- phases 2 and 3.
        self._commit_checkpoint(p)
        self._generations[p['dir']] = generation
        self._last_ckpt = time.time()

    def _commit_checkpoint(self, p: dict) -> None:
        """Phases 2 and 3: rename the staged set into place, `state.json` last.

        Idempotent, because this is also how an interrupted commit is finished:
        every rename whose source has already gone is a rename that already
        happened.
        """
        for kind in CHECKPOINT_ARTEFACTS:
            self._commit_one(os.path.join(p['stage'],
                                          os.path.basename(p[kind])), p[kind])
        self._commit_one(os.path.join(p['stage'], STAGED_STATE_NAME), p['state'])
        _fsync_dir(p['dir'])
        shutil.rmtree(p['stage'], ignore_errors=True)

    @staticmethod
    def _commit_one(src: str, dst: str) -> None:
        """The only way a file ever appears under a committed name."""
        if os.path.exists(src):
            os.replace(src, dst)

    def _save_model_atomic(self, path: str) -> None:
        """`model.keras` is another run's *input*, so it gets the same treatment.

        A torn source model does not break the run that wrote it; it breaks
        every transfer arm that later loads it, and it breaks them at a moment
        when the obvious suspect is the transfer code.
        """
        stage = os.path.join(os.path.dirname(path), STAGING_DIRNAME)
        shutil.rmtree(stage, ignore_errors=True)
        os.makedirs(stage, exist_ok=True)
        tmp = os.path.join(stage, os.path.basename(path))
        self.agent.online.save(tmp)
        _fsync_file(tmp)
        os.replace(tmp, path)
        _fsync_dir(os.path.dirname(path))
        shutil.rmtree(stage, ignore_errors=True)

    def _reap_buffer(self, sub: str = '') -> None:
        """Drop a finished checkpoint's replay archive without breaking its set.

        Measured at 6.35 MB for a LunarLander replay at capacity, and a
        completed run has no use for it; over roughly 1200 runs holding two
        checkpoints each that is the difference between about 15 GB and about
        1.6 GB. It cannot simply be unlinked, because the set would then
        be missing an artefact its own `state.json` declares and the next
        resume would rightly refuse. `state.json` is rewritten FIRST, at the
        same generation, declaring the archive reaped; only then is the file
        removed. A kill between the two leaves a file that nothing claims,
        which is ignorable, rather than a claim with no file, which is not.
        """
        p = self._paths(sub)
        if not os.path.exists(p['buffer']):
            return
        state, _why = _read_json_file(p['state'])
        if state is not None:
            cset = dict(state.get('checkpoint_set') or {})
            artefacts = dict(cset.get('artefacts') or {})
            artefacts.pop(os.path.basename(p['buffer']), None)
            cset['artefacts'] = artefacts
            cset['buffer_reaped'] = True
            state['checkpoint_set'] = cset
            _write_json_atomic(p['state'], state)
        os.remove(p['buffer'])

    def _locate_set(self, p: dict, state: dict) -> dict:
        """Where a sound copy of each artefact of `state`'s generation is.

        Two places, and they mean different things: a committed copy is an
        `os.replace` that already happened, a staged copy is one that had not
        happened yet when the process died.
        """
        generation = state.get('generation')
        cset = state.get('checkpoint_set') or {}
        ledger = dict(cset.get('artefacts') or {})
        reaped = bool(cset.get('buffer_reaped'))
        report: dict = {}
        for kind in CHECKPOINT_ARTEFACTS:
            if kind == 'buffer' and reaped:
                continue
            name = os.path.basename(p[kind])
            expect = ledger.get(name)
            committed = _artefact_state(kind, p[kind])
            staged = _artefact_state(kind, os.path.join(p['stage'], name))
            report[kind] = {
                'name': name,
                'committed': committed,
                'staged': staged,
                'committed_ok': _artefact_ok(committed, generation, expect),
                'staged_ok': _artefact_ok(staged, generation, expect)}
        return report

    def _recover_checkpoint(self, p: dict) -> dict | None:
        """The state of a complete, self-consistent checkpoint, or a refusal.

        `None` means there is no checkpoint here at all, which is a fresh run
        rather than a fault. Anything else is either a set whose every artefact
        carries the generation `state.json` names, or a `RuntimeError` that
        says what was found where.
        """
        staged_state, _why = _read_json_file(
            os.path.join(p['stage'], STAGED_STATE_NAME))
        if staged_state is not None:
            report = self._locate_set(p, staged_state)
            if report and all(e['committed_ok'] or e['staged_ok']
                              for e in report.values()):
                # Phase 2 was interrupted. Every artefact of this generation is
                # still on disk, committed or staged, so the commit is finished
                # rather than the checkpoint thrown away.
                self._commit_checkpoint(p)
                print(f'[checkpoint] finished an interrupted commit of '
                      f'generation {staged_state.get("generation")} in '
                      f'{p["dir"]}')
        if os.path.isdir(p['stage']):
            # Phase 1 was interrupted: an incomplete set that nothing ever
            # pointed at. Nothing committed was touched, so it is swept up and
            # the previous generation stands.
            shutil.rmtree(p['stage'], ignore_errors=True)

        state, why = _read_json_file(p['state'])
        if state is None:
            leftovers = sorted(os.path.basename(p[k])
                               for k in CHECKPOINT_ARTEFACTS
                               if os.path.exists(p[k]))
            if why == 'absent' and not leftovers:
                return None
            if why == 'absent':
                raise RuntimeError(
                    f'refusing to resume {self.dir}: the directory holds '
                    f'checkpoint artefacts {leftovers} but no state.json, so '
                    f'nothing on disk says which episode they belong to. '
                    f'Starting over would silently discard whatever training '
                    f'they represent and append episodes from 0 to a metrics '
                    f'log that may already hold them, which is the duplication '
                    f'DESIGN.md 8.2(1) exists to prevent. Delete the directory '
                    f'and restart.')
            raise RuntimeError(
                f'refusing to resume {self.dir}: state.json is {why}. A '
                f'checkpoint whose pointer cannot be read is not a checkpoint. '
                f'The published writer truncated it in place, so a kill a '
                f'millisecond later left exactly this; the writer here renames '
                f'it into position instead, so a torn one now means damage '
                f'below this process. Delete the directory and restart.')

        generation = state.get('generation')
        if not isinstance(generation, int):
            raise RuntimeError(
                f'refusing to resume {self.dir}: state.json carries no '
                f'checkpoint generation, so it was written by the pre-atomic '
                f'writer and the files beside it cannot be shown to be one '
                f'checkpoint. That writer could leave weights, replay and '
                f'optimiser from checkpoint N+1 beside a state.json from '
                f'checkpoint N, which resumes silently and trains a wrong '
                f'trajectory. Delete the directory and restart.')

        report = self._locate_set(p, state)
        if not all(entry['committed_ok'] for entry in report.values()):
            raise RuntimeError(self._mixed_set_message(p, state, report))
        return state

    def _mixed_set_message(self, p: dict, state: dict, report: dict) -> str:
        """Name the generation of every artefact found, then say what it costs.

        The message is long on purpose. Its reader is looking at a refusal in
        the middle of a 1200-run campaign and has to decide, in one glance,
        whether to delete a directory; "checkpoint mismatch" would not let
        them.
        """
        generation = state.get('generation')
        lines = []
        for kind in CHECKPOINT_ARTEFACTS:
            entry = report.get(kind)
            if entry is None:
                lines.append(f'    {os.path.basename(p[kind]):<20s} '
                             f'not part of this set (reaped after the run)')
                continue
            found = entry['committed']
            if not found['present']:
                where = 'absent'
            elif found['generation'] is None:
                where = f'generation unknown: {found["note"]}'
            else:
                where = f'generation {found["generation"]}'
                if found['note']:
                    where += f', but {found["note"]}'
            flag = '' if entry['committed_ok'] else '   <-- does not match'
            lines.append(f'    {entry["name"]:<20s} {where}{flag}')
        return (
            f'refusing to resume {self.dir}: the files here are not one '
            f'checkpoint.\n'
            f'  state.json names generation {generation}, at episode '
            f'{state.get("episode")!r}; on disk:\n'
            + '\n'.join(lines)
            + '\n  Resuming would restore that episode into a network already '
              'holding a different checkpoint\'s parameters, replay those '
              'episodes with advanced weights, and produce a wrong trajectory '
              'with nothing raised. That is worse than a crash, because it is '
              'silent. Delete the directory and restart: one lost run costs '
              'about half an hour, one silently mixed run costs the result.')

    def _maybe_resume(self) -> None:
        p = self._paths()
        st = self._recover_checkpoint(p)
        if st is None:
            return
        self._generations[p['dir']] = int(st['generation'])

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
            # By name, not by count: the archive also carries the generation
            # stamp, so `range(len(d.files))` would ask for one slot too many.
            slots = sorted((k for k in d.files
                            if k.startswith('v') and k[1:].isdigit()),
                           key=lambda k: int(k[1:]))
            self.agent.load_optimizer_state({'values': [d[k] for k in slots]})
        else:
            # Unreachable now that `_recover_checkpoint` verifies the whole set
            # before anything is loaded, and kept anyway: it is the specific
            # defect DESIGN.md 8.2 names, and a backstop costing one branch is
            # cheaper than trusting that the check above never regresses.
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
              f'(generation {st["generation"]}, dropped {dropped} metric rows, '
              f'{dropped_evals} eval rows)')

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
                self._episode = episode
                self.agent.episode = episode
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
        self._save_model_atomic(os.path.join(self.dir, 'model.keras'))
        self._finalise(held_out, started, final_eval_episodes)

        if not cfg.keep_buffer:
            for sub in ('', *[f'ckpt_ep{e}' for e in cfg.prefix_checkpoints]):
                self._reap_buffer(sub)

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
            result.update(_curve_summaries(df, cfg, result['env_steps']))
        self.manifest['result'] = result

        # `manifest.json` is what every downstream tool reads to decide a
        # run is finished and sound. Truncating it in place means a kill during
        # the dump leaves a file that parses as nothing, and a run that reads
        # as neither complete nor absent, so it goes through the same
        # write-then-rename as the checkpoint.
        _write_json_atomic(os.path.join(self.dir, 'manifest.json'),
                           self.manifest, indent=2, default=str)
        if not integrity['contiguous']:
            print(f'[WARNING] metrics integrity: {integrity["problems"]}')


def _curve_summaries(df, cfg: Config, total_env_steps: int) -> dict:
    """AUC per env step, convergence slope, and the dispersion metrics.

    AUC is integrated over **env steps** rather than episodes because episode
    length is performance-dependent, so an episode-indexed area silently weights
    arms differently.

    The divisor is the run's **total** env steps, which is what `DESIGN.md`
    5.2 and `ANALYSIS_PLAN.md` 1 define P2 to be: "divided by total env
    steps", "per step". Dividing by the span between the first and the last
    evaluation instead would report the mean score over whatever window
    happened to carry evaluations, so a run that stopped early, or one whose
    unevaluated tail is long because its episodes are long, would be credited
    for budget it never spent under evaluation. Under the specified divisor
    the steps before the first evaluation and after the last one contribute
    no area; `auc_env_step_coverage` records what fraction of the budget the
    curve actually spans, so the size of that omission is auditable rather
    than assumed (0.965 to 0.996 on the n=1 pilot tree).
    """
    ev = df.dropna(subset=['eval_score']) if 'eval_score' in df else df.iloc[0:0]
    out: dict = {}
    if len(ev) >= 2:
        x = ev['env_steps'].to_numpy(dtype=float)
        y = ev['eval_score'].to_numpy(dtype=float)
        span = x[-1] - x[0]
        total = float(total_env_steps or 0)
        out['auc_score'] = (float(np.trapezoid(y, x) / total)
                            if span > 0 and total > 0 else None)
        out['auc_env_step_coverage'] = (float(span / total)
                                        if total > 0 else None)
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
