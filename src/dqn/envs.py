"""Environment registry, parametric variants, and interface bookkeeping.

Why this module exists at all
----------------------------
The published study used a single source->target pair, CartPole-v1 ->
LunarLander-v3, which changes the observation dimension (4->8), the action
cardinality (2->4), the dynamics, the reward structure and the horizon
*simultaneously*. Every effect it attributed to "domain shift" is therefore
confounded with the partial-copy and head-reinitialisation mechanics that the
interface change forces. No reviewer named that confound; separating it is the
point of `DESIGN.md` RQ5.

Separating it requires environments that differ in dynamics while keeping the
interface **identical**, so that the whole network transfers with no partial
copy and no reinitialised head. Gymnasium supports exactly that through
documented parameters, so a variant here is a *configuration*, never a code
fork:

    LunarLander-v3:gravity=-4                 dynamics shift, interface identical
    LunarLander-v3:enable_wind=1,wind_power=15
    CartPole-v1:length=1.0,masspole=0.2

A trap this module exists to close: CartPole keeps `total_mass` and
`polemass_length` as *derived* attributes. Setting `masspole` or `length`
without recomputing them leaves the simulation running the default physics, so
the "variant" silently is not one. `_DERIVED` handles that, and every applied
value is read back and verified before the environment is returned.

Reward thresholds are read from the Gymnasium registry rather than hard-coded,
because the published paper quoted CartPole's ">=195" success criterion, which
belongs to `CartPole-v0`; `CartPole-v1` is registered at 475. That single
mismatch is why its source agent's score of 26.94 was never recognised as a
total failure to learn.
"""
from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import Any

# Parameters that may be varied per environment, and how they reach the
# simulator: 'kwarg' goes to gym.make(), 'attr' is set on env.unwrapped after
# construction. Anything not listed here is rejected, so a typo in an
# experiment definition fails loudly instead of producing a silent default.
TUNABLE: dict[str, dict[str, str]] = {
    'LunarLander-v3': {
        'gravity': 'kwarg',
        'enable_wind': 'kwarg',
        'wind_power': 'kwarg',
        'turbulence_power': 'kwarg',
        # Interface-change parameters: applied as wrappers, so they alter the
        # observation dimension and action cardinality while leaving the
        # dynamics bit-identical. This is the same-dynamics/changed-interface
        # corner of DESIGN.md section 6.4.
        'pad_obs': 'wrapper',
        'pad_mode': 'wrapper',
        'extra_actions': 'wrapper',
    },
    'CartPole-v1': {
        'gravity': 'attr',
        'masscart': 'attr',
        'masspole': 'attr',
        'length': 'attr',
        'force_mag': 'attr',
        'tau': 'attr',
        'pad_obs': 'wrapper',
        'pad_mode': 'wrapper',
        'extra_actions': 'wrapper',
    },
    'Acrobot-v1': {
        'LINK_LENGTH_1': 'attr',
        'LINK_LENGTH_2': 'attr',
        'LINK_MASS_1': 'attr',
        'LINK_MASS_2': 'attr',
    },
    'MountainCar-v0': {
        'force': 'attr',
        'gravity': 'attr',
    },
}

# Attributes a simulator derives from others. Whenever any name on the right is
# set, the attribute on the left must be recomputed or the change is inert.
_DERIVED: dict[str, list[tuple[str, tuple[str, ...]]]] = {
    'CartPole-v1': [
        ('total_mass', ('masscart', 'masspole')),
        ('polemass_length', ('masspole', 'length')),
    ],
}


def _recompute_derived(env_id: str, unwrapped) -> list[str]:
    """Restore invariants between primitive and derived simulator attributes."""
    done = []
    for name, deps in _DERIVED.get(env_id, []):
        if name == 'total_mass':
            unwrapped.total_mass = unwrapped.masscart + unwrapped.masspole
        elif name == 'polemass_length':
            unwrapped.polemass_length = unwrapped.masspole * unwrapped.length
        else:                                    # pragma: no cover - guard
            raise NotImplementedError(f'no rule for derived attribute {name!r}')
        done.append(f'{name}={getattr(unwrapped, name)!r} (from {"+".join(deps)})')
    return done


# Qualitative shift descriptors. `DESIGN.md` section 6.3 refuses to emit a
# single scalar "domain distance" between environments with different state
# spaces, because no such quantity is defined. This is what is reported instead,
# and it is deliberately structured rather than prose so it can go straight into
# a table.
DESCRIPTORS: dict[str, dict[str, Any]] = {
    'CartPole-v1': dict(obs_dim=4, act_dim=2, reward_density='dense (+1 per step)',
                        reward_sign='non-negative', horizon=500,
                        termination='failure (pole angle / cart position)',
                        return_range='[0, 500]',
                        control='discrete push left/right'),
    'Acrobot-v1': dict(obs_dim=6, act_dim=3, reward_density='dense (-1 per step)',
                       reward_sign='non-positive', horizon=500,
                       termination='goal reached (height)',
                       return_range='[-500, ~-60]',
                       control='discrete torque -1/0/+1'),
    'LunarLander-v3': dict(obs_dim=8, act_dim=4,
                           reward_density='shaped, with terminal bonus/penalty',
                           reward_sign='mixed', horizon=1000,
                           termination='landing or crash, plus time limit',
                           return_range='~[-600, +320]',
                           control='discrete main/side engines'),
    'MountainCar-v0': dict(obs_dim=2, act_dim=3, reward_density='sparse (-1 per step)',
                           reward_sign='non-positive', horizon=200,
                           termination='goal reached (position)',
                           return_range='[-200, ~-90]',
                           control='discrete accelerate left/none/right'),
}


@dataclass(frozen=True)
class EnvSpec:
    """A base environment plus a validated set of parameter overrides."""

    env_id: str
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.env_id not in DESCRIPTORS:
            raise ValueError(f'unknown env_id {self.env_id!r}; '
                             f'known: {sorted(DESCRIPTORS)}')
        allowed = TUNABLE.get(self.env_id, {})
        unknown = sorted(set(self.params) - set(allowed))
        if unknown:
            raise ValueError(
                f'{self.env_id}: parameter(s) {unknown} are not tunable. '
                f'Tunable here: {sorted(allowed)}. Refusing rather than '
                f'silently ignoring them.')

    # ---- identity --------------------------------------------------------
    def slug(self) -> str:
        """Filesystem-safe label, readable enough to skip the manifest.

        'LunarLander-v3' -> 'lunarlander'
        'LunarLander-v3:gravity=-4.0' -> 'lunarlander_gravity-4.0'
        """
        base = self.env_id.split('-')[0].lower()
        if not self.params:
            return base
        parts = [f'{k}{_fmt(v)}' for k, v in sorted(self.params.items())]
        return base + '_' + '_'.join(parts)

    def canonical(self) -> str:
        """Round-trips through `parse`; this is what goes in the manifest."""
        if not self.params:
            return self.env_id
        kv = ','.join(f'{k}={_fmt(v)}' for k, v in sorted(self.params.items()))
        return f'{self.env_id}:{kv}'

    def is_variant(self) -> bool:
        return bool(self.params)

    # ---- interface -------------------------------------------------------
    @property
    def obs_dim(self) -> int:
        """Observation dimensionality *after* any interface wrapper.

        Transfer keys on this, so it has to be the wrapped value: a padded
        variant whose declared dimensionality was the base one would silently
        mis-resolve the shape-compatible transfer set.
        """
        return (int(DESCRIPTORS[self.env_id]['obs_dim'])
                + int(self.params.get('pad_obs', 0) or 0))

    @property
    def act_dim(self) -> int:
        return (int(DESCRIPTORS[self.env_id]['act_dim'])
                + int(self.params.get('extra_actions', 0) or 0))

    @property
    def base_obs_dim(self) -> int:
        return int(DESCRIPTORS[self.env_id]['obs_dim'])

    @property
    def base_act_dim(self) -> int:
        return int(DESCRIPTORS[self.env_id]['act_dim'])

    def changes_interface_only(self, other: 'EnvSpec') -> bool:
        """True when `self` and `other` differ only by interface wrappers.

        The predicate that identifies the same-dynamics/changed-interface pair:
        the base environment and every dynamics parameter match, and only the
        wrapper parameters differ.
        """
        wrapper_keys = {'pad_obs', 'pad_mode', 'extra_actions'}
        if self.env_id != other.env_id:
            return False
        mine = {k: v for k, v in self.params.items() if k not in wrapper_keys}
        theirs = {k: v for k, v in other.params.items() if k not in wrapper_keys}
        if mine != theirs:
            return False
        return ((self.obs_dim, self.act_dim) != (other.obs_dim, other.act_dim))

    def descriptor(self) -> dict:
        d = dict(DESCRIPTORS[self.env_id])
        d['env'] = self.canonical()
        if self.params:
            d['altered'] = dict(self.params)
        return d

    def reward_threshold(self) -> float | None:
        """The registry's own success threshold, never a remembered number."""
        from gymnasium.envs.registration import registry
        spec = registry.get(self.env_id)
        thr = getattr(spec, 'reward_threshold', None) if spec else None
        return float(thr) if thr is not None else None

    def max_episode_steps(self) -> int | None:
        from gymnasium.envs.registration import registry
        spec = registry.get(self.env_id)
        n = getattr(spec, 'max_episode_steps', None) if spec else None
        return int(n) if n is not None else None


def _fmt(v: Any) -> str:
    if isinstance(v, bool):
        return '1' if v else '0'
    if isinstance(v, float) and v == int(v):
        return str(int(v)) if abs(v) >= 1 or v == 0 else repr(v)
    return str(v)


def _coerce(text: str) -> Any:
    low = text.strip().lower()
    if low in ('true', 'yes'):
        return True
    if low in ('false', 'no'):
        return False
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def parse(spec: str | EnvSpec) -> EnvSpec:
    """Parse 'LunarLander-v3' or 'LunarLander-v3:gravity=-4,enable_wind=1'.

    A compact string form is what lets an environment variant travel through a
    CLI flag, a config field, a run directory name and a manifest without any
    of them needing to know the variant vocabulary.
    """
    if isinstance(spec, EnvSpec):
        return spec
    text = str(spec).strip()
    if ':' not in text:
        return EnvSpec(text, {})
    env_id, _, rest = text.partition(':')
    params: dict[str, Any] = {}
    for token in rest.split(','):
        token = token.strip()
        if not token:
            continue
        if '=' not in token:
            raise ValueError(f'malformed env parameter {token!r} in {spec!r}; '
                             f'expected name=value')
        key, _, val = token.partition('=')
        params[key.strip()] = _coerce(val)
    return EnvSpec(env_id.strip(), params)


def interfaces_match(a: str | EnvSpec, b: str | EnvSpec) -> bool:
    """True when a network can transfer whole, with no partial copy or reinit.

    This is the predicate that separates RQ5's clean dynamics-shift arm from
    every cross-environment pair, so it is worth having in one place rather
    than re-derived at each call site.
    """
    a, b = parse(a), parse(b)
    return a.obs_dim == b.obs_dim and a.act_dim == b.act_dim


def make(spec: str | EnvSpec, render_mode: str | None = None):
    """Construct an environment, apply overrides, and verify they took effect.

    Returns `(env, resolved)` where `resolved` records exactly what the
    simulator ended up running -- including recomputed derived attributes --
    so that a run's manifest documents the physics rather than the request.
    """
    import gymnasium as gym

    es = parse(spec)
    allowed = TUNABLE.get(es.env_id, {})
    kwargs = {k: v for k, v in es.params.items() if allowed.get(k) == 'kwarg'}
    attrs = {k: v for k, v in es.params.items() if allowed.get(k) == 'attr'}
    wrapper = {k: v for k, v in es.params.items() if allowed.get(k) == 'wrapper'}

    env = gym.make(es.env_id, render_mode=render_mode, **kwargs)
    env.reset(seed=0)                 # some simulators build state lazily

    notes = []
    for key, val in attrs.items():
        if not hasattr(env.unwrapped, key):
            env.close()
            raise AttributeError(
                f'{es.env_id}: env.unwrapped has no attribute {key!r}. '
                f'The registry claims it is tunable; the installed Gymnasium '
                f'disagrees. Refusing to run a variant that would be inert.')
        setattr(env.unwrapped, key, val)
    if attrs:
        notes = _recompute_derived(es.env_id, env.unwrapped)

    # Read back. An override that did not stick would otherwise produce a run
    # labelled as a variant while simulating the default physics -- exactly the
    # class of undetectable error this whole rebuild exists to eliminate.
    resolved: dict[str, Any] = {}
    for key, val in es.params.items():
        if allowed[key] == 'attr':
            got = getattr(env.unwrapped, key)
            if not _close(got, val):
                env.close()
                raise RuntimeError(
                    f'{es.env_id}: setting {key}={val!r} did not take effect '
                    f'(read back {got!r}).')
            resolved[key] = got
        else:
            got = getattr(env.unwrapped, key, val)
            resolved[key] = got

    if wrapper:
        from .env_wrappers import apply_interface_wrappers
        env, applied_wrappers = apply_interface_wrappers(
            env,
            pad_obs=int(wrapper.get('pad_obs', 0) or 0),
            pad_mode=str(wrapper.get('pad_mode', 'noise')),
            extra_actions=int(wrapper.get('extra_actions', 0) or 0))
        resolved.update(applied_wrappers)

    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.n)
    if (obs_dim, act_dim) != (es.obs_dim, es.act_dim):
        env.close()
        raise RuntimeError(
            f'{es.canonical()}: registry says interface is '
            f'({es.obs_dim}, {es.act_dim}) but the constructed environment is '
            f'({obs_dim}, {act_dim}). The registry is stale -- fix it before '
            f'running, since transfer keys on this.')

    return env, {
        'env': es.canonical(),
        'obs_dim': obs_dim,
        'act_dim': act_dim,
        'applied': resolved,
        'derived_recomputed': notes,
        'reward_threshold': es.reward_threshold(),
        'max_episode_steps': es.max_episode_steps(),
    }


def _close(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    try:
        return abs(float(a) - float(b)) <= 1e-9 * max(1.0, abs(float(b)))
    except (TypeError, ValueError):
        return a == b


# ---------------------------------------------------------------------------
# Named variant families used by the experiment catalogue. Kept here so that an
# experiment definition names a family rather than restating physics, and so
# that the levels used by a published figure are recoverable from one place.
# ---------------------------------------------------------------------------
VARIANT_FAMILIES: dict[str, dict] = {
    # RQ5's clean arm: interface identical, dynamics shifted monotonically.
    # Gravity is the natural ordering variable -- weaker gravity makes the
    # lander easier to hover but changes the achievable return range, which is
    # why RQ5 compares scale-free effect sizes across levels and not raw
    # deltas (DESIGN.md section 6.2).
    'll_gravity': {
        'base': 'LunarLander-v3',
        'levels': [
            ('g10', {}),                          # -10.0, the default
            ('g08', {'gravity': -8.0}),
            ('g06', {'gravity': -6.0}),
            ('g04', {'gravity': -4.0}),
        ],
        'ordering_key': 'gravity',
        'default_value': -10.0,
    },
    # A second, qualitatively different same-interface shift: stochastic
    # disturbance rather than a changed constant. If the transfer effect tracks
    # gravity but not wind, the mechanism is about dynamics scale; if it tracks
    # both, it is about dynamics mismatch generally.
    'll_wind': {
        'base': 'LunarLander-v3',
        'levels': [
            ('w00', {}),
            ('w075', {'enable_wind': True, 'wind_power': 7.5, 'turbulence_power': 1.5}),
            ('w15', {'enable_wind': True, 'wind_power': 15.0, 'turbulence_power': 1.5}),
        ],
        'ordering_key': 'wind_power',
        'default_value': 0.0,
    },
    # Cheap same-interface family for pipeline validation and for a second
    # environment's worth of evidence at low cost.
    # The missing corner of the shift factorial: dynamics identical, interface
    # changed. Level 0 is the unmodified environment, so the pair
    # (level 0 -> level k) is a pure interface change.
    'll_interface': {
        'base': 'LunarLander-v3',
        'levels': [
            ('i0', {}),
            ('i4a2', {'pad_obs': 4, 'pad_mode': 'noise', 'extra_actions': 2}),
        ],
        'ordering_key': 'pad_obs',
        'default_value': 0,
    },
    'cp_pole': {
        'base': 'CartPole-v1',
        'levels': [
            ('l050', {}),
            ('l075', {'length': 0.75, 'masspole': 0.15}),
            ('l100', {'length': 1.0, 'masspole': 0.2}),
        ],
        'ordering_key': 'length',
        'default_value': 0.5,
    },
}


def family_specs(name: str) -> list[tuple[str, EnvSpec]]:
    """(level label, EnvSpec) for a named variant family."""
    fam = VARIANT_FAMILIES[name]
    return [(label, EnvSpec(fam['base'], dict(params)))
            for label, params in fam['levels']]


def family_level_value(name: str, spec: str | EnvSpec) -> float:
    """The ordering variable's value for one level, for dose-response plots."""
    fam = VARIANT_FAMILIES[name]
    es = parse(spec)
    return float(es.params.get(fam['ordering_key'], fam['default_value']))


def shift_descriptor_table(pairs: list[tuple[str, str]]) -> list[dict]:
    """Structured qualitative shift descriptors for a set of source->target pairs.

    Emitted instead of a scalar distance for cross-interface pairs. Saying that
    a quantity is undefined across different state spaces is more defensible
    than computing one anyway -- which is what sank the published version's
    2-Wasserstein-over-returns metric with ICANN reviewer #5.
    """
    rows = []
    for src, tgt in pairs:
        s, t = parse(src), parse(tgt)
        matched = interfaces_match(s, t)
        sd, td = s.descriptor(), t.descriptor()
        rows.append({
            'source': s.canonical(),
            'target': t.canonical(),
            'interface_match': matched,
            'obs_dim': f"{sd['obs_dim']} -> {td['obs_dim']}",
            'act_dim': f"{sd['act_dim']} -> {td['act_dim']}",
            'reward_density': f"{sd['reward_density']} -> {td['reward_density']}",
            'reward_sign': f"{sd['reward_sign']} -> {td['reward_sign']}",
            'horizon': f"{sd['horizon']} -> {td['horizon']}",
            'termination': f"{sd['termination']} -> {td['termination']}",
            'scalar_shift_metric_defined': matched,
            'shift_family': ('dynamics only (interface identical)' if matched
                             else 'interface + dynamics + reward (confounded)'),
        })
    return rows


# ---------------------------------------------------------------------------
# Normalisation. Everything reported in this study is on a normalised score, not
# a raw return, and this is where that lives.
# ---------------------------------------------------------------------------
_REFERENCE_CACHE: dict | None = None
REFERENCE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'experiments', 'reference_returns.json')


def load_references(path: str | None = None) -> dict:
    """Measured random-policy and no-op reference returns, per canonical env."""
    global _REFERENCE_CACHE
    if _REFERENCE_CACHE is None or path is not None:
        target = path or REFERENCE_FILE
        try:
            with open(target, encoding='utf-8') as fh:
                data = json.load(fh)
        except FileNotFoundError:
            data = {}
        if path is None:
            _REFERENCE_CACHE = data
        else:
            return data
    return _REFERENCE_CACHE or {}


def reference(spec: str | EnvSpec) -> dict:
    """The reference points for one environment, or raise.

    Raising rather than defaulting is deliberate. A missing reference would
    otherwise be silently replaced by zero, which would put every score for that
    variant on a different scale from every other -- exactly the class of
    undetectable scale error that made the published cross-variant comparisons
    meaningless. Measure it with `experiments/measure_references.py`.
    """
    es = parse(spec)
    key = es.canonical()
    refs = load_references()
    if key in refs:
        return refs[key]
    raise KeyError(
        f'no measured reference return for {key!r}. Scores are normalised '
        f'against the random-policy return, so this must be measured before '
        f'the environment is used: run '
        f'`python experiments/measure_references.py --env "{key}"`.')


def normalised_score(spec: str | EnvSpec, ret: float) -> float:
    """(return - random_return) / (threshold - random_return).

    A uniform-random policy scores 0 and the registered threshold scores 1, by
    construction, so scores are comparable across environments and across
    variants whose return scales differ by hundreds of points.

    The threshold is taken from the *base* environment registration even for a
    parametric variant, and that is a deliberate, defensible choice rather than
    an oversight: these variants change the transition dynamics and leave the
    reward function untouched, so "what counts as solved" is unchanged, while
    "what a random policy achieves" is not -- and it is the latter that the
    normalisation corrects for.
    """
    ref = reference(spec)
    rand = float(ref['random_return'])
    thr = ref.get('threshold')
    if thr is None:
        raise KeyError(f'{parse(spec).canonical()}: no reward threshold registered')
    denom = float(thr) - rand
    if abs(denom) < 1e-9:
        raise ValueError(f'{parse(spec).canonical()}: degenerate normalisation '
                         f'(threshold equals random-policy return)')
    return (float(ret) - rand) / denom


def denormalise_score(spec: str | EnvSpec, score: float) -> float:
    """Inverse of `normalised_score`, for reporting raw returns alongside."""
    ref = reference(spec)
    rand = float(ref['random_return'])
    return rand + float(score) * (float(ref['threshold']) - rand)


__all__ = ['EnvSpec', 'parse', 'make', 'interfaces_match', 'DESCRIPTORS',
           'TUNABLE', 'VARIANT_FAMILIES', 'family_specs', 'family_level_value',
           'shift_descriptor_table', 'load_references', 'reference',
           'normalised_score', 'denormalise_score', 'REFERENCE_FILE']
