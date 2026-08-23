"""Unified DQN implementation for the controlled transfer study.

Replaces the five duplicated packages audited in `paper/METHODS_ACTUAL.md`,
whose configs had drifted apart and whose arms differed in more variables than
the manuscript claimed. One `Config`, one agent, one training loop; an arm is
identified by `(arch, target_rule, condition)` plus an explicit protocol, and a
run is identified by a digest over every field that can change its trajectory or
its measurement.

Read `experiments/DESIGN.md` before changing anything here.
"""
from .config import (ARCHS, CONDITIONS, TARGET_RULES, Config,  # noqa: F401
                     config_from_args)
from .networks import (TRANSFER_SETS, build_q_network,  # noqa: F401
                       resolve_layers, transfer_set_layers)
from .seeding import STREAMS, Seeds  # noqa: F401

__all__ = ['Config', 'config_from_args', 'build_q_network', 'resolve_layers',
           'transfer_set_layers', 'Seeds', 'ARCHS', 'TARGET_RULES',
           'CONDITIONS', 'TRANSFER_SETS', 'STREAMS']
