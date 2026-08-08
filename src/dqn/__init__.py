"""Unified DQN implementation for the 2x2 transfer study.

Replaces the five duplicated packages audited in paper/METHODS_ACTUAL.md, whose
configs had drifted apart and whose arms differed in more variables than the
manuscript claimed. One Config, one agent, one training loop; an arm is
identified only by (arch, target_rule, mode).
"""
from .config import ARCHS, MODES, TARGET_RULES, Config, config_from_args
from .networks import build_q_network

__all__ = ['Config', 'config_from_args', 'build_q_network',
           'ARCHS', 'TARGET_RULES', 'MODES']
