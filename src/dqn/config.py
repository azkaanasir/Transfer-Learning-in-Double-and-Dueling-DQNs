"""One configuration schema for every experimental arm, with run identity.

Phase 0 found the original four packages each carried their own copy of the
hyperparameters and had drifted apart -- most damagingly a 5x learning-rate gap
between the transfer and baseline arms, which broke the manuscript's "identical
hyperparameters" control claim. The fix is structural: there is one `Config`,
and every optimisation hyperparameter is shared by construction.

Run identity is a first-class object here, and that is not bookkeeping
pedantry. Adversarial review of revision 1 of the design demonstrated by
execution that the previous scheme -- a directory named
`<env>/<arch>-<rule>-<mode>-s<NN>` -- omitted `freeze_*`, `transfer_*`, `lr`,
`target_update`, `hidden`, `aggregation`, the environment variant and the
control condition. Nine distinct conditions drawn from six catalogue
experiments collapsed onto a single directory, and because a completed
directory was silently *resumed* rather than refused, a run declaring
`lr=1e-3, freeze=0` trained zero episodes and emitted a manifest carrying those
values over metrics byte-identical to the `lr=5e-4, freeze=100` run. Five
experiments would have been fabricated from one experiment's data, with every
invariant check passing. Hence:

    run_dir = <out_root>/<experiment>/<condition>/<run_digest12>/s<NN>

where `run_digest` covers every field that can change either the training
trajectory or the reported measurement, over an explicit versioned allowlist.
Fields are classified into exactly three sets, and a field that is added
without being classified raises at import time -- so the digest cannot silently
stop covering something.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Optional

from . import envs
from .networks import AGGREGATIONS, LAYER_GROUPS, TRANSFER_SETS

ARCHS = ('mlp', 'dueling')
TARGET_RULES = ('vanilla', 'double')
CONDITIONS = ('scratch', 'transfer', 'transfer_untrained', 'transfer_permuted')
PERMUTE_KINDS = ('shuffle', 'spectrum')
INPUT_POLICIES = ('partial', 'reinit', 'redraw_matched')
HEAD_POLICIES = ('reinit', 'partial')
VALUE_RECAL = ('none', 'center', 'center_scale')

# Bumped whenever the meaning of a digested field changes, so that old and new
# digests cannot silently collide.
DIGEST_SCHEMA = 'v2'


@dataclass
class Config:
    # ---- identity --------------------------------------------------------
    experiment: str = 'adhoc'
    # Human-readable arm name supplied by the experiment registry, e.g.
    # 'freezedur-K10k'. Deliberately excluded from the digest: renaming an arm
    # for a table must not orphan its runs.
    label: str = ''

    # ---- factors ---------------------------------------------------------
    arch: str = 'dueling'
    target_rule: str = 'double'
    condition: str = 'scratch'
    aggregation: str = 'mean'
    env: str = 'LunarLander-v3'            # EnvSpec canonical string
    source_env: Optional[str] = None       # recorded for provenance and audit
    source_checkpoint: Optional[str] = None
    seed: int = 0

    # ---- transfer protocol ----------------------------------------------
    # A *named* set, never a raw layer list: {trunk_fc1, trunk_fc2} copies 94%
    # of the mlp and 50% of the dueling network, which confounds the
    # architecture factor with treatment intensity.
    transfer_set: str = 'matched'
    input_policy: str = 'partial'
    head_policy: str = 'reinit'
    # Freezing is indexed in *gradient updates*, not episodes. LunarLander
    # episode length ranges over an order of magnitude with performance, so an
    # episode-indexed freeze window means a different amount of optimisation in
    # every arm -- and that difference is endogenous to performance.
    freeze_group: str = 'trunk'
    freeze_updates: int = 10_000
    permute_scope: str = 'all'
    permute_kind: str = 'shuffle'
    value_recal: str = 'none'
    # Plasticity interventions, applied at the freeze->unfreeze boundary. The
    # plasticity-loss literature supplies an architecture-free rival explanation
    # for degradation after pretraining -- dead units, parameter-norm growth,
    # feature-rank collapse -- and these are the standard mitigations. They are
    # here so the rival account can be *tested* rather than only measured; see
    # paper/LITERATURE.md section 3.4.
    reset_head_at_unfreeze: bool = False
    # Ash & Adams shrink-and-perturb: W <- (1 - shrink_perturb) * W
    # + shrink_perturb * W_fresh. 0.0 disables it.
    shrink_perturb: float = 0.0

    # ---- shared optimisation; do not vary these per arm -----------------
    num_episodes: int = 1000
    max_steps: int = 1000
    lr: float = 5e-4
    gamma: float = 0.99
    batch_size: int = 64
    replay_capacity: int = 100_000
    learning_starts: int = 1_000           # env steps before the first update
    train_every: int = 1                   # env steps between updates
    # A *global* norm. Keras clips each tensor separately under `clipnorm`, and
    # the mlp has 6 weight tensors against the dueling network's 10, so the same
    # scalar would impose a materially different constraint per architecture --
    # a hidden confound on the very axis under study.
    grad_clip_norm: float = 10.0

    epsilon_start: float = 1.0
    epsilon_min: float = 0.01
    # Closed form in env steps (see `epsilon_at`). The published code decayed
    # epsilon *inside* the evaluation branch, which made the exploration
    # schedule a function of `eval_every`; nothing here couples them.
    epsilon_anneal_steps: int = 300_000

    target_update: str = 'hard'            # 'hard' | 'soft'
    target_update_freq: int = 1_000        # gradient steps, when 'hard'
    tau: float = 0.005                     # when 'soft'

    hidden: tuple = (128, 128)
    head_units: int = 64

    # ---- measurement -----------------------------------------------------
    eval_every: int = 10                   # episodes, monitoring evaluation
    eval_episodes: int = 5
    final_eval_episodes: int = 100          # held-out, the primary endpoint
    final_eval_checkpoints: int = 3          # averaged, to remove oscillation
    # Immutable checkpoints for the budget analysis. Without these, RQ6 is not
    # executable: the rolling checkpoint is overwritten, so no episode-500 state
    # survives to be evaluated on the primary endpoint.
    #
    # Only episode 500 by default, and deliberately so. Each prefix costs a full
    # 100-episode held-out evaluation -- measured at about 21 s per block on a
    # 1000-episode LunarLander run, against 778 s of training -- and 500 is the
    # only prefix an actual question attaches to: it is the manuscript's
    # published budget, measured before epsilon reaches its floor at roughly
    # episode 891. Adding 250 and 750 cost 5 per cent of every run in the
    # catalogue to answer nothing that was asked.
    prefix_checkpoints: tuple = (500,)
    diag_states: int = 512
    # Linear-probe jumpstart: with a reinitialised output head the zero-shot
    # greedy policy is an argmax over a random readout, so plain jumpstart is
    # structurally at chance and comparing it across head-reinit arms would be
    # meaningless. Fitting only the head on a fixed batch of target transitions
    # measures what the question actually asks: whether the transferred
    # features carry usable information. Set probe_steps=0 to skip.
    probe_steps: int = 1_500
    probe_transitions: int = 4_000

    # ---- bookkeeping -----------------------------------------------------
    out_root: str = 'runs'
    checkpoint_seconds: int = 600           # wall-clock, so cadence is
    #                                         condition-independent
    keep_buffer: bool = False               # ~5 MB/run; deleted on success
    log_diagnostics: bool = True
    notes: str = ''

    # -----------------------------------------------------------------
    def __post_init__(self):
        if self.arch not in ARCHS:
            raise ValueError(f'arch must be one of {ARCHS}, got {self.arch!r}')
        if self.target_rule not in TARGET_RULES:
            raise ValueError(f'target_rule must be one of {TARGET_RULES}, '
                             f'got {self.target_rule!r}')
        if self.condition not in CONDITIONS:
            raise ValueError(f'condition must be one of {CONDITIONS}, '
                             f'got {self.condition!r}')
        if self.aggregation not in AGGREGATIONS:
            raise ValueError(f'aggregation must be one of {AGGREGATIONS}')
        if self.arch == 'mlp' and self.aggregation != 'mean':
            raise ValueError(
                'aggregation applies only to the dueling architecture; leave it '
                "at 'mean' for mlp so the digest does not record a factor that "
                'had no effect.')
        if self.transfer_set not in TRANSFER_SETS:
            raise ValueError(f'transfer_set must be one of {TRANSFER_SETS}')
        if self.input_policy not in INPUT_POLICIES:
            raise ValueError(f'input_policy must be one of {INPUT_POLICIES}')
        if self.head_policy not in HEAD_POLICIES:
            raise ValueError(f'head_policy must be one of {HEAD_POLICIES}')
        if self.permute_kind not in PERMUTE_KINDS:
            raise ValueError(f'permute_kind must be one of {PERMUTE_KINDS}')
        if self.value_recal not in VALUE_RECAL:
            raise ValueError(f'value_recal must be one of {VALUE_RECAL}')
        if self.target_update not in ('hard', 'soft'):
            raise ValueError('target_update must be "hard" or "soft"')
        if not 0.0 <= self.shrink_perturb <= 1.0:
            raise ValueError('shrink_perturb must lie in [0, 1], got '
                             f'{self.shrink_perturb}')
        if (self.reset_head_at_unfreeze or self.shrink_perturb) \
                and self.freeze_updates == 0:
            raise ValueError(
                'reset_head_at_unfreeze and shrink_perturb act at the '
                'freeze->unfreeze boundary, and freeze_updates=0 means there is '
                'no such boundary. The intervention would silently never fire.')
        if self.freeze_group not in LAYER_GROUPS[self.arch]:
            raise ValueError(
                f'freeze_group {self.freeze_group!r} is not a layer group for '
                f'{self.arch}; known: {sorted(LAYER_GROUPS[self.arch])}')
        if self.is_transfer and not self.source_checkpoint \
                and self.condition == 'transfer':
            raise ValueError("condition='transfer' requires a source_checkpoint")
        if self.is_transfer and self.condition != 'transfer' \
                and not (self.source_checkpoint or self.source_env):
            raise ValueError(
                f"condition={self.condition!r} needs source_env (to know the "
                f'source interface) and, for transfer_permuted, a '
                f'source_checkpoint to permute')
        if self.condition == 'transfer_permuted' and not self.source_checkpoint:
            raise ValueError("condition='transfer_permuted' permutes a *trained* "
                             'source, so it requires a source_checkpoint')
        # Normalise the environment strings through the registry so that two
        # spellings of the same variant cannot produce two run directories.
        self.env = envs.parse(self.env).canonical()
        if self.source_env:
            self.source_env = envs.parse(self.source_env).canonical()
        self.hidden = tuple(int(h) for h in self.hidden)
        self.prefix_checkpoints = tuple(sorted(int(p) for p in self.prefix_checkpoints
                                               if 0 < int(p) < self.num_episodes))

    # ---- derived ---------------------------------------------------------
    @property
    def is_transfer(self) -> bool:
        return self.condition != 'scratch'

    @property
    def env_spec(self) -> envs.EnvSpec:
        return envs.parse(self.env)

    @property
    def source_env_spec(self) -> Optional[envs.EnvSpec]:
        return envs.parse(self.source_env) if self.source_env else None

    def epsilon_at(self, env_step: int) -> float:
        """Closed-form exploration schedule, in env steps.

        Geometric interpolation from `epsilon_start` to `epsilon_min`, reaching
        the floor exactly at `epsilon_anneal_steps`. Being closed form is what
        makes it independent of the evaluation cadence, replayable at any point
        without simulating history, and -- because it never reads
        `num_episodes` -- what licenses the budget analysis in `DESIGN.md` RQ6:
        a 500-episode prefix of a longer run is exactly what a 500-episode run
        would have produced.
        """
        if self.epsilon_anneal_steps <= 0:
            return float(self.epsilon_min)
        frac = min(1.0, max(0.0, env_step / float(self.epsilon_anneal_steps)))
        ratio = self.epsilon_min / self.epsilon_start
        return float(max(self.epsilon_min, self.epsilon_start * (ratio ** frac)))

    def transfer_layers(self, src_obs: int, src_act: int,
                        tgt_obs: int, tgt_act: int) -> tuple[str, ...]:
        from .networks import transfer_set_layers
        return transfer_set_layers(self.arch, self.transfer_set,
                                   src_obs, tgt_obs, src_act, tgt_act)

    def freeze_layers(self) -> tuple[str, ...]:
        from .networks import resolve_layers
        return resolve_layers(self.arch, [self.freeze_group])

    # ---- identity --------------------------------------------------------
    def arm_id(self) -> str:
        """Canonical arm name, e.g. 'dueling-double-transfer'."""
        return f'{self.arch}-{self.target_rule}-{self.condition}'

    def digest(self, fields: tuple[str, ...]) -> str:
        """Hash over the named fields, with transfer-only fields neutralised for
        a scratch run.

        Nothing in a scratch run reads `transfer_set`, `freeze_updates`,
        `permute_kind` and the rest -- they are inert. Hashing them anyway would
        mean that scaling a freeze window for a pilot produced a *second* copy of
        every scratch baseline whose trajectory is bit-identical to the first, so
        the same quantity would be estimated twice and the catalogue's run count
        would silently inflate. Neutralising them keeps one scratch run per
        genuine configuration. The recorded config is left untouched, so the
        manifest still reflects exactly what was requested.
        """
        payload = {'schema': DIGEST_SCHEMA}
        data = self.to_dict()
        inert = TRANSFER_ONLY_FIELDS if self.condition == 'scratch' else frozenset()
        for name in fields:
            if name in inert:
                payload[name] = '<inert-for-scratch>'
                continue
            val = data[name]
            payload[name] = list(val) if isinstance(val, tuple) else val
        blob = json.dumps(payload, sort_keys=True, separators=(',', ':'))
        return hashlib.blake2b(blob.encode('utf-8'), digest_size=16).hexdigest()

    def trajectory_digest(self) -> str:
        """Covers everything that can change the numerical training trajectory.

        This is what a resume is checked against: continuing a run under changed
        training hyperparameters is the class of error Phase 0 spent days undoing.
        """
        return self.digest(TRAJECTORY_FIELDS)

    def measurement_digest(self) -> str:
        """Covers everything that changes the reported metrics but not training."""
        return self.digest(MEASUREMENT_FIELDS)

    def run_digest(self) -> str:
        return self.digest(IDENTITY_FIELDS)

    def run_id(self) -> str:
        return f'{self.arm_id()}-{self.run_digest()[:12]}-s{self.seed:02d}'

    def run_dir(self) -> str:
        """Where this run lives. Deliberately *not* keyed by experiment.

        Two catalogue experiments often request an identical configuration --
        E4's freeze level that equals E1's protocol value, E8's shift level 0
        scratch arm, E7's scratch arms. Keying the path by experiment would
        train each of them again, wasting compute and, worse, producing two
        independent estimates of the same quantity that a reader would
        reasonably assume were different arms. Identical configuration means one
        run; experiment membership is recorded separately in the run index,
        which is what `audit.py` reads.
        """
        return os.path.join(self.out_root, self.condition,
                            self.run_digest()[:12], f's{self.seed:02d}')

    # ---- serialisation ---------------------------------------------------
    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def identity(self) -> dict:
        """Everything needed to recognise, reproduce and audit this run."""
        return {
            'experiment': self.experiment,
            'label': self.label,
            'arm_id': self.arm_id(),
            'condition': self.condition,
            'seed': self.seed,
            'run_dir': self.run_dir(),
            'digest_schema': DIGEST_SCHEMA,
            'run_digest': self.run_digest(),
            'trajectory_digest': self.trajectory_digest(),
            'measurement_digest': self.measurement_digest(),
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: str) -> 'Config':
        with open(path, encoding='utf-8') as fh:
            return cls(**json.load(fh))


# ---------------------------------------------------------------------------
# Field classification. Every field belongs to exactly one set; the assertion
# below fires at import if a new field is added without being classified, so
# the digests cannot silently stop covering something.
# ---------------------------------------------------------------------------
# Fields that only affect a transfer condition. A scratch run never reads them,
# so they are excluded from its digest (see `Config.digest`).
TRANSFER_ONLY_FIELDS = frozenset({
    'source_env', 'transfer_set', 'input_policy', 'head_policy', 'freeze_group',
    'freeze_updates', 'permute_scope', 'permute_kind', 'value_recal',
    'reset_head_at_unfreeze', 'shrink_perturb',
})

TRAJECTORY_FIELDS: tuple[str, ...] = (
    'arch', 'target_rule', 'condition', 'aggregation', 'env', 'source_env',
    'seed',
    'transfer_set', 'input_policy', 'head_policy', 'freeze_group',
    'freeze_updates', 'permute_scope', 'permute_kind', 'value_recal',
    'reset_head_at_unfreeze', 'shrink_perturb',
    'num_episodes', 'max_steps', 'lr', 'gamma', 'batch_size',
    'replay_capacity', 'learning_starts', 'train_every', 'grad_clip_norm',
    'epsilon_start', 'epsilon_min', 'epsilon_anneal_steps',
    'target_update', 'target_update_freq', 'tau', 'hidden', 'head_units',
)

MEASUREMENT_FIELDS: tuple[str, ...] = (
    'eval_every', 'eval_episodes', 'final_eval_episodes',
    'final_eval_checkpoints', 'prefix_checkpoints', 'diag_states',
    'probe_steps', 'probe_transitions',
)

# Excluded from every digest, deliberately. `source_checkpoint` is a *path*:
# hashing it would mean moving the run tree changed a run's identity, while the
# source's actual content is pinned by `source_env` plus the recorded source
# digest and weight fingerprint in the manifest.
BOOKKEEPING_FIELDS: tuple[str, ...] = (
    'experiment', 'label', 'source_checkpoint', 'out_root',
    'checkpoint_seconds', 'keep_buffer', 'log_diagnostics', 'notes',
)

# Run identity is the union of what changes the trajectory and what changes the
# measurement -- and nothing else. `experiment` is bookkeeping: an experiment is
# a *set of runs*, not a property of one.
IDENTITY_FIELDS: tuple[str, ...] = tuple(
    sorted(set(TRAJECTORY_FIELDS) | set(MEASUREMENT_FIELDS)))


def _check_field_classification() -> None:
    declared = {f.name for f in dataclasses.fields(Config)}
    classified = (set(TRAJECTORY_FIELDS) | set(MEASUREMENT_FIELDS)
                  | set(BOOKKEEPING_FIELDS))
    missing = declared - classified
    spurious = classified - declared
    overlap = ((set(TRAJECTORY_FIELDS) & set(MEASUREMENT_FIELDS))
               | (set(TRAJECTORY_FIELDS) & set(BOOKKEEPING_FIELDS))
               | (set(MEASUREMENT_FIELDS) & set(BOOKKEEPING_FIELDS)))
    if missing or spurious or overlap:
        raise RuntimeError(
            'Config field classification is incomplete. Every field must be in '
            'exactly one of TRAJECTORY_FIELDS, MEASUREMENT_FIELDS or '
            'BOOKKEEPING_FIELDS, or the run digest stops covering it and '
            'distinct conditions collide onto one run directory.\n'
            f'  unclassified: {sorted(missing)}\n'
            f'  not a field:  {sorted(spurious)}\n'
            f'  in two sets:  {sorted(overlap)}')


_check_field_classification()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _add(parser: argparse.ArgumentParser, name: str, f: dataclasses.Field):
    flag = '--' + name.replace('_', '-')
    if isinstance(f.default, bool):
        parser.add_argument(flag, default=None,
                            type=lambda s: s.lower() not in ('0', 'false', 'no'),
                            help=f'(default {f.default})')
    elif isinstance(f.default, tuple):
        parser.add_argument(flag, nargs='*', default=None,
                            help=f'(default {f.default})')
    else:
        typ = type(f.default) if f.default is not None else str
        parser.add_argument(flag, type=typ, default=None,
                            help=f'(default {f.default})')


def build_parser(description: str = 'Train one DQN arm.') -> argparse.ArgumentParser:
    """A CLI covering every Config field, so nothing is un-overridable.

    Phase 0 found the published runs could not have set the learning rate at
    all -- no `--lr` flag existed -- which is why the transfer/baseline gap was
    unfixable at launch.
    """
    p = argparse.ArgumentParser(description=description)
    for f in dataclasses.fields(Config):
        _add(p, f.name, f)
    return p


def config_from_args(argv=None, **overrides) -> Config:
    """Parse argv into a Config; only flags actually passed take effect."""
    args = build_parser().parse_args(argv)
    kwargs = {k: v for k, v in vars(args).items() if v is not None}
    kwargs.update(overrides)
    for key in ('hidden', 'prefix_checkpoints'):
        if key in kwargs and isinstance(kwargs[key], list):
            kwargs[key] = tuple(int(v) for v in kwargs[key])
    return Config(**kwargs)


__all__ = ['Config', 'config_from_args', 'build_parser', 'ARCHS',
           'TARGET_RULES', 'CONDITIONS', 'VALUE_RECAL', 'PERMUTE_KINDS',
           'INPUT_POLICIES', 'HEAD_POLICIES', 'DIGEST_SCHEMA',
           'TRAJECTORY_FIELDS', 'MEASUREMENT_FIELDS', 'BOOKKEEPING_FIELDS',
           'TRANSFER_ONLY_FIELDS']
