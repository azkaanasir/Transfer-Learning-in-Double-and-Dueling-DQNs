"""Measuring domain shift, and refusing to measure it where it is undefined.

Why the published metric was rejected
-------------------------------------
The revised version of the manuscript answered IJCNN reviewer 1's demand to
quantify domain divergence with a 2-Wasserstein distance over *return* summary
statistics. ICANN reviewer 5 rejected it, correctly: returns are a consequence
of the policy, the reward scale and the horizon cap, so W2 between CartPole
returns ([0, 500], all positive) and LunarLander returns (about [-600, +320])
mostly measures the difference in reward scale. It is not a distance between
MDPs. Both reviewers were right about different things -- divergence *should* be
quantified, and *that* quantity was the wrong one.

What this module computes instead
---------------------------------
Three measurements, each with a stated scope, and one explicit refusal.

1. **Paired trajectory divergence** (`paired_trajectory_divergence`) --
   the primary measure, and only defined when the two environments share an
   interface. From an identical initial state, drive both environments with an
   *identical action sequence* and watch the trajectories separate. Because the
   policy, the initial state and the actions are all held fixed, the separation
   is attributable to the transition dynamics alone. This is a paired design,
   which is why it is far more sensitive than comparing marginal distributions.

2. **State-visitation divergence** (`state_visitation_divergence`) -- per
   dimension W2 and energy distance between the state distributions each
   environment induces under a fixed reference policy, standardised by the
   source environment's own per-dimension scale so the number is unit-free.
   Reported under two reference policies, which answer different questions:
     * `random`        -- dynamics divergence over the reachable region, with the
                          action distribution held identical in both
     * `source_greedy` -- divergence over the region the transferred policy
                          actually drives the target environment into, which is
                          what a transferred representation is evaluated on

3. **Linear CKA** (`linear_cka`, `representation_similarity`) --
   representation-level similarity between two networks' activations on **the
   same** states. CKA compares two representations of the same examples, so the
   earlier formulation -- one network on two different state sets -- was
   ill-posed and, across a change of input dimensionality, not computable. The
   two well-posed uses are transferred-versus-scratch at matched seed, and
   drift of the transferred trunk over training.

**The refusal.** For a cross-interface pair such as CartPole -> LunarLander,
there is no shared state space, so no distance between state distributions is
defined. `shift_report` returns `defined: False` with the reason, and the
qualitative descriptor from `envs.shift_descriptor_table` is what gets reported.
Saying a quantity is undefined is a stronger position than inventing one.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from . import envs


# ---------------------------------------------------------------------------
# 1. Paired trajectory divergence -- the dynamics measure
# ---------------------------------------------------------------------------
def paired_trajectory_divergence(source: str | envs.EnvSpec,
                                 target: str | envs.EnvSpec,
                                 episodes: int = 30,
                                 max_steps: int = 400,
                                 seed: int = 0,
                                 policy: Callable[[np.ndarray], int] | None = None,
                                 run_self_check: bool = True) -> dict:
    """Separation between two environments under identical states and actions.

    The action sequence is drawn once per episode from a dedicated RNG and
    replayed into both environments, so the two rollouts differ in nothing but
    their transition dynamics. Distances are in units of the source
    environment's per-dimension standard deviation, so they are comparable
    across dimensions and across environment families.

    `policy=None` uses a uniform-random action sequence, which is the correct
    choice for a *dynamics* measurement: it fixes the action distribution
    identically in both environments. Passing a policy makes the measurement
    conditional on that policy's state coverage, which is a different and also
    legitimate question -- but then the action sequence is no longer shared,
    because the policy reacts to states that have already diverged, so this
    function refuses it. Use `state_visitation_divergence` for that question.
    """
    if policy is not None:
        raise ValueError(
            'paired_trajectory_divergence requires a shared action sequence, '
            'which a state-dependent policy cannot provide once the '
            'trajectories diverge. Use state_visitation_divergence(policy=...) '
            'for a policy-conditional measurement.')
    s_spec, t_spec = envs.parse(source), envs.parse(target)
    if not envs.interfaces_match(s_spec, t_spec):
        return {'defined': False,
                'reason': f'interfaces differ ({s_spec.obs_dim},{s_spec.act_dim}) '
                          f'vs ({t_spec.obs_dim},{t_spec.act_dim}); no shared '
                          f'state space, so trajectory distance is undefined',
                'source': s_spec.canonical(), 'target': t_spec.canonical()}

    env_s, _ = envs.make(s_spec)
    env_t, _ = envs.make(t_spec)
    rng = np.random.default_rng(seed)
    n_actions = s_spec.act_dim

    per_step: list[list[np.ndarray]] = []   # [episode][step] -> per-dim difference
    reference_states: list[np.ndarray] = []
    # Which environment left the MDP first, and when. Recorded rather than
    # differenced: because the comparison stops at the first termination, any
    # "horizon gap" computed from these paired rollouts would be zero by
    # construction and would say nothing. Which side terminates first, and how
    # often, is the informative quantity.
    terminated_first = {'source': 0, 'target': 0, 'simultaneous': 0, 'neither': 0}
    stop_steps: list[int] = []

    for ep in range(episodes):
        ep_seed = int(seed * 100_000 + ep)
        actions = rng.integers(0, n_actions, size=max_steps)
        s0, _ = env_s.reset(seed=ep_seed)
        t0, _ = env_t.reset(seed=ep_seed)
        s0 = np.asarray(s0, dtype=np.float64)
        t0 = np.asarray(t0, dtype=np.float64)
        reference_states.append(s0)

        # Step 0 of the curve is the difference already present in the reset
        # observation. It is not necessarily zero even under identical seeds:
        # Gymnasium's LunarLander `reset()` advances the simulator by one step
        # before returning, so the first observation it hands back has already
        # been acted on by the variant's dynamics. Including it makes that
        # visible instead of letting it masquerade as a pairing failure. The
        # pairing itself is validated by `self_check`, which runs an
        # environment against itself and must return exactly zero.
        dists = [np.asarray(s0 - t0, dtype=np.float64)]
        s_state, t_state = s0, t0
        stop = max_steps
        for k in range(max_steps):
            s_state, _, s_term, s_trunc, _ = env_s.step(int(actions[k]))
            t_state, _, t_term, t_trunc, _ = env_t.step(int(actions[k]))
            s_state = np.asarray(s_state, dtype=np.float64)
            t_state = np.asarray(t_state, dtype=np.float64)
            reference_states.append(s_state)
            s_done, t_done = s_term or s_trunc, t_term or t_trunc
            if s_done or t_done:
                # Beyond the first termination the rollouts are no longer
                # comparable step-for-step: one of them has left the MDP.
                if s_done and t_done:
                    terminated_first['simultaneous'] += 1
                elif s_done:
                    terminated_first['source'] += 1
                else:
                    terminated_first['target'] += 1
                stop = k + 1
                break
            dists.append(np.asarray(s_state - t_state, dtype=np.float64))
        else:
            terminated_first['neither'] += 1
        stop_steps.append(stop)
        per_step.append(dists)

    env_s.close()
    env_t.close()

    scale = np.std(np.asarray(reference_states, dtype=np.float64), axis=0)
    scale[scale < 1e-8] = 1.0

    curves = []
    for dists in per_step:
        if not dists:
            curves.append(np.zeros(0))
            continue
        arr = np.asarray(dists) / scale
        curves.append(np.linalg.norm(arr, axis=1))

    # Threshold for "meaningfully separated": one standardised unit, i.e. the
    # trajectories differ by as much as the reference state distribution's own
    # spread. Stated here rather than tuned, so it cannot be chosen after
    # seeing the answer.
    threshold = 1.0
    steps_to_separate = []
    for c, dists in zip(curves, per_step):
        idx = np.argmax(c > threshold) if len(c) and (c > threshold).any() else None
        steps_to_separate.append(int(idx) if idx is not None else None)

    max_len = max((len(c) for c in curves), default=0)
    mean_curve = []
    for k in range(max_len):
        vals = [c[k] for c in curves if len(c) > k]
        mean_curve.append(float(np.mean(vals)))

    reached = [s for s in steps_to_separate if s is not None]
    result = {
        'defined': True,
        'source': s_spec.canonical(),
        'target': t_spec.canonical(),
        'episodes': episodes,
        'reference_policy': 'uniform_random_shared_action_sequence',
        'standardisation': 'per-dimension SD of source-environment states',
        'separation_threshold_sd': threshold,
        'mean_divergence_curve': mean_curve,
        'initial_divergence': mean_curve[0] if mean_curve else 0.0,
        'divergence_at_step_10': mean_curve[10] if len(mean_curve) > 10 else None,
        'divergence_at_step_50': mean_curve[50] if len(mean_curve) > 50 else None,
        'terminal_divergence': mean_curve[-1] if mean_curve else 0.0,
        'median_steps_to_separate': (float(np.median(reached)) if reached else None),
        'episodes_that_separated': len(reached),
        'terminated_first': terminated_first,
        'mean_comparable_steps': float(np.mean(stop_steps)) if stop_steps else 0.0,
    }
    if run_self_check:
        # Validates the method rather than the pair: driving one environment
        # against a second instance of *itself* with the same seeds and the same
        # actions must give exactly zero divergence. If it does not, the
        # environment carries hidden state and no divergence number from this
        # function means anything.
        control = paired_trajectory_divergence(
            s_spec, s_spec, episodes=min(episodes, 5), max_steps=max_steps,
            seed=seed, run_self_check=False)
        residual = float(control.get('terminal_divergence', float('nan')))
        result['self_check'] = {
            'identical_env_terminal_divergence': residual,
            'pairing_valid': bool(residual == 0.0),
        }
    return result


# ---------------------------------------------------------------------------
# 2. State-visitation divergence -- the coverage measure
# ---------------------------------------------------------------------------
def collect_states(spec: str | envs.EnvSpec,
                   episodes: int = 40,
                   max_steps: int = 400,
                   seed: int = 0,
                   act: Callable[[np.ndarray], int] | None = None) -> np.ndarray:
    """Sample the state distribution an environment induces under a policy."""
    es = envs.parse(spec)
    env, _ = envs.make(es)
    rng = np.random.default_rng(seed)
    out = []
    for ep in range(episodes):
        state, _ = env.reset(seed=int(seed * 100_000 + ep))
        state = np.asarray(state, dtype=np.float32)
        for _ in range(max_steps):
            out.append(state)
            action = (int(rng.integers(0, es.act_dim)) if act is None
                      else int(act(state)))
            state, _, term, trunc, _ = env.step(action)
            state = np.asarray(state, dtype=np.float32)
            if term or trunc:
                break
    env.close()
    return np.asarray(out, dtype=np.float64)


def state_visitation_divergence(source: str | envs.EnvSpec,
                                target: str | envs.EnvSpec,
                                episodes: int = 40,
                                max_steps: int = 400,
                                seed: int = 0,
                                act: Callable[[np.ndarray], int] | None = None,
                                policy_label: str = 'random') -> dict:
    """Per-dimension W2 and energy distance between two state distributions.

    Standardised by the source distribution's per-dimension SD, so the result is
    unit-free and dimensions with different physical units are comparable.
    Only defined when the two environments share an observation space.
    """
    from scipy import stats

    s_spec, t_spec = envs.parse(source), envs.parse(target)
    if s_spec.obs_dim != t_spec.obs_dim:
        return {'defined': False,
                'reason': f'observation dimensionality differs '
                          f'({s_spec.obs_dim} vs {t_spec.obs_dim}); a distance '
                          f'between these state distributions is undefined',
                'source': s_spec.canonical(), 'target': t_spec.canonical()}

    xs = collect_states(s_spec, episodes, max_steps, seed, act)
    xt = collect_states(t_spec, episodes, max_steps, seed, act)
    scale = xs.std(axis=0)
    scale[scale < 1e-8] = 1.0

    w2, energy = [], []
    for d in range(xs.shape[1]):
        a, b = xs[:, d] / scale[d], xt[:, d] / scale[d]
        w2.append(float(stats.wasserstein_distance(a, b)))
        energy.append(float(stats.energy_distance(a, b)))

    return {
        'defined': True,
        'source': s_spec.canonical(),
        'target': t_spec.canonical(),
        'reference_policy': policy_label,
        'n_source_states': int(len(xs)),
        'n_target_states': int(len(xt)),
        'standardisation': 'per-dimension SD of source-environment states',
        'w2_per_dim': w2,
        'energy_per_dim': energy,
        # Summaries. The max is reported alongside the mean because a shift
        # concentrated in one dimension and a shift spread over all of them are
        # different situations that a mean alone would hide.
        'w2_mean': float(np.mean(w2)),
        'w2_max': float(np.max(w2)),
        'w2_argmax_dim': int(np.argmax(w2)),
        'energy_mean': float(np.mean(energy)),
    }


# ---------------------------------------------------------------------------
# 3. Representation similarity
# ---------------------------------------------------------------------------
def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear centred kernel alignment between two activation matrices.

    Rows are examples, columns are units. Invariant to isotropic scaling and to
    orthogonal transformation of the features, which is what makes it the right
    similarity measure for comparing representations that were never required
    to align coordinate-wise. Returns a value in [0, 1].
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape[0] != y.shape[0]:
        n = min(x.shape[0], y.shape[0])
        x, y = x[:n], y[:n]
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    num = np.linalg.norm(x.T @ y, ord='fro') ** 2
    den = (np.linalg.norm(x.T @ x, ord='fro')
           * np.linalg.norm(y.T @ y, ord='fro'))
    return float(num / den) if den > 0 else float('nan')


def representation_similarity(features_a: np.ndarray,
                              features_b: np.ndarray) -> dict:
    """CKA between two networks' activations on the SAME states.

    This replaces an earlier, ill-posed statistic. CKA compares two
    *representations of the same examples*: given activation matrices from two
    different input sets the Gram matrices are not comparable, and for a
    cross-interface pair such as CartPole -> LunarLander the two input
    dimensionalities differ so the quantity is not even computable. The two
    well-posed questions, both on one fixed batch of target-environment states,
    are:

    * `cka_transfer_vs_scratch` -- transferred run's trunk activations against
      the matched-seed scratch run's: how different a representation the transfer
      actually produced.
    * `cka_drift` -- the trunk at episode 0 against the trunk later: how far
      fine-tuning had to move the transferred features.

    Both are calls to this function with the appropriate pair of matrices.
    """
    fa = np.asarray(features_a)
    fb = np.asarray(features_b)
    if fa.shape[0] != fb.shape[0]:
        raise ValueError(
            f'CKA needs the same examples in both matrices, got {fa.shape[0]} '
            f'and {fb.shape[0]} rows. Comparing activations on two different '
            f'state sets is not a similarity between representations.')
    dead_a = float(np.mean(np.all(fa <= 0, axis=0)))
    dead_b = float(np.mean(np.all(fb <= 0, axis=0)))
    return {
        'cka': linear_cka(fa, fb),
        'n_states': int(fa.shape[0]),
        'activation_norm_a': float(np.linalg.norm(fa, axis=1).mean()),
        'activation_norm_b': float(np.linalg.norm(fb, axis=1).mean()),
        # A unit that never activates over the batch carries no information
        # about it. For a *frozen* transferred layer this is a concrete,
        # checkable failure mode rather than a speculative one.
        'dead_unit_frac_a': dead_a,
        'dead_unit_frac_b': dead_b,
    }


# ---------------------------------------------------------------------------
# Top-level report
# ---------------------------------------------------------------------------
def shift_report(source: str | envs.EnvSpec,
                 target: str | envs.EnvSpec,
                 episodes: int = 30,
                 max_steps: int = 400,
                 seed: int = 0) -> dict:
    """Everything defensible that can be said about one source -> target pair.

    Always includes the structured qualitative descriptor. Includes the
    quantitative measures only when they are defined, and states why when they
    are not.
    """
    s_spec, t_spec = envs.parse(source), envs.parse(target)
    descriptor = envs.shift_descriptor_table([(s_spec.canonical(),
                                               t_spec.canonical())])[0]
    report: dict[str, Any] = {
        'source': s_spec.canonical(),
        'target': t_spec.canonical(),
        'qualitative': descriptor,
        'interface_match': descriptor['interface_match'],
    }
    report['trajectory'] = paired_trajectory_divergence(
        s_spec, t_spec, episodes=episodes, max_steps=max_steps, seed=seed)
    report['state_visitation'] = state_visitation_divergence(
        s_spec, t_spec, episodes=episodes, max_steps=max_steps, seed=seed,
        policy_label='random')
    if not descriptor['interface_match']:
        report['scalar_metric'] = None
        report['scalar_metric_reason'] = (
            'The observation and/or action spaces differ, so no distance '
            'between state distributions is defined. Reported qualitatively '
            'instead. A single "domain distance" number here would be an '
            'artifact of an arbitrary alignment, which is the defect that '
            'sank the 2-Wasserstein-over-returns metric in review.')
    return report


__all__ = ['paired_trajectory_divergence', 'state_visitation_divergence',
           'collect_states', 'linear_cka', 'representation_similarity',
           'shift_report']
