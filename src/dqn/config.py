"""Single configuration schema for every experimental arm.

Phase 0 found that the original four packages each carried their own copy of the
hyperparameters, and that they had drifted apart -- most damagingly a 5x
learning-rate gap between the transfer and baseline arms, which broke the
manuscript's "identical hyperparameters" control claim.

The fix is structural rather than procedural: there is one Config, and an arm is
identified *only* by (arch, target_rule, mode). Every optimisation
hyperparameter is shared by construction, so the arms cannot silently drift
again. `Config.arm_id()` is the canonical name used for run directories.

See paper/METHODS_ACTUAL.md for what the original runs actually used.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import Optional

ARCHS = ('mlp', 'dueling')
TARGET_RULES = ('vanilla', 'double')
MODES = ('scratch', 'transfer')


@dataclass
class Config:
    # ---- what makes this arm different from the others -------------------
    arch: str = 'dueling'              # 'mlp' | 'dueling'
    target_rule: str = 'double'        # 'vanilla' | 'double'
    mode: str = 'scratch'              # 'scratch' | 'transfer'
    env_id: str = 'LunarLander-v3'
    seed: int = 0

    # ---- shared by every arm; do not vary these per-arm ------------------
    num_episodes: int = 500
    max_steps: int = 1000
    lr: float = 5e-4
    gamma: float = 0.99
    batch_size: int = 64
    replay_capacity: int = 100_000
    learning_starts: int = 1_000       # env steps before the first update
    train_every: int = 1               # env steps between updates
    grad_clip_norm: float = 10.0

    epsilon_start: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.95
    # The original code decayed epsilon once per validation pass, not per
    # episode. Preserved (and now explicit) so the schedule stays comparable
    # with the published runs: 0.95 every 10 episodes ~= 0.995 per episode.
    epsilon_decay_every: int = 10

    target_update: str = 'hard'        # 'hard' | 'soft'
    target_update_freq: int = 1_000    # gradient steps, when 'hard'
    tau: float = 0.005                 # when 'soft'

    hidden: tuple = (128, 128)
    head_units: int = 64               # dueling V/A stream width

    # ---- transfer protocol ----------------------------------------------
    source_checkpoint: Optional[str] = None
    # Which named layers to copy from the source. Trunk layers share names
    # across architectures, so transfer maps *by name* -- never by position.
    transfer_layers: tuple = ('trunk_fc1', 'trunk_fc2')
    # trunk_fc1's kernel is (state_dim, 128), so its shape differs between
    # CartPole (4) and LunarLander (8). 'partial' copies the overlapping rows
    # and Glorot-initialises the rest (what the published runs did);
    # 'reinit' skips it (what the manuscript describes).
    first_layer_policy: str = 'partial'   # 'partial' | 'reinit'
    # Layers held frozen for the first `freeze_episodes` episodes, then
    # released. freeze_episodes=0 disables freezing entirely.
    freeze_layers: tuple = ('trunk_fc1', 'trunk_fc2')
    freeze_episodes: int = 100

    # ---- evaluation ------------------------------------------------------
    eval_every: int = 10               # episodes
    eval_episodes: int = 5
    eval_window: int = 100             # episodes averaged for the headline scalar

    # ---- bookkeeping -----------------------------------------------------
    out_root: str = 'runs'
    checkpoint_every: int = 25         # episodes; keeps Colab timeouts survivable
    log_diagnostics: bool = True       # V/A stream magnitudes, for Phase 2
    notes: str = ''

    def __post_init__(self):
        if self.arch not in ARCHS:
            raise ValueError(f'arch must be one of {ARCHS}, got {self.arch!r}')
        if self.target_rule not in TARGET_RULES:
            raise ValueError(f'target_rule must be one of {TARGET_RULES}, '
                             f'got {self.target_rule!r}')
        if self.mode not in MODES:
            raise ValueError(f'mode must be one of {MODES}, got {self.mode!r}')
        if self.mode == 'transfer' and not self.source_checkpoint:
            raise ValueError('mode="transfer" requires --source-checkpoint')
        if self.target_update not in ('hard', 'soft'):
            raise ValueError('target_update must be "hard" or "soft"')
        if self.first_layer_policy not in ('partial', 'reinit'):
            raise ValueError('first_layer_policy must be "partial" or "reinit"')
        self.hidden = tuple(self.hidden)
        self.transfer_layers = tuple(self.transfer_layers)
        self.freeze_layers = tuple(self.freeze_layers)

    # ---- identity --------------------------------------------------------
    def arm_id(self) -> str:
        """Canonical arm name, e.g. 'dueling-double-transfer'."""
        return f'{self.arch}-{self.target_rule}-{self.mode}'

    def run_id(self) -> str:
        """Arm plus seed, e.g. 'dueling-double-transfer-s03'."""
        return f'{self.arm_id()}-s{self.seed:02d}'

    def run_dir(self) -> str:
        env_slug = self.env_id.split('-')[0].lower()
        return os.path.join(self.out_root, env_slug, self.run_id())

    # ---- serialisation ---------------------------------------------------
    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: str) -> 'Config':
        with open(path, encoding='utf-8') as fh:
            return cls(**json.load(fh))


def _add(parser: argparse.ArgumentParser, name: str, f: dataclasses.Field):
    """Expose one dataclass field as a CLI flag."""
    flag = '--' + name.replace('_', '-')
    if f.type is bool or isinstance(f.default, bool):
        parser.add_argument(flag, type=lambda s: s.lower() not in ('0', 'false', 'no'),
                            default=None, help=f'(default {f.default})')
    elif isinstance(f.default, tuple):
        parser.add_argument(flag, nargs='*', default=None,
                            help=f'(default {f.default})')
    else:
        typ = type(f.default) if f.default is not None else str
        parser.add_argument(flag, type=typ, default=None,
                            help=f'(default {f.default})')


def build_parser(description: str = 'Train one DQN arm.') -> argparse.ArgumentParser:
    """A CLI covering every Config field, so nothing is un-overridable.

    Phase 0 found the original runs could not have set the learning rate at
    all -- no `--lr` flag existed -- which is why the transfer/baseline gap was
    unfixable at launch. Every field is exposed here deliberately.
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
    for key in ('hidden', 'transfer_layers', 'freeze_layers'):
        if key in kwargs and isinstance(kwargs[key], list):
            cast = int if key == 'hidden' else str
            kwargs[key] = tuple(cast(v) for v in kwargs[key])
    return Config(**kwargs)
