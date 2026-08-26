"""Single-run entry point.

    python -m src.dqn.run --arch dueling --target-rule double \
        --env-id LunarLander-v3 --mode scratch --seed 0

    python -m src.dqn.run --arch dueling --target-rule double \
        --env-id LunarLander-v3 --mode transfer --seed 0 \
        --source-checkpoint runs/cartpole/dueling-double-scratch-s00/model.keras

Every Config field is exposed as a flag -- notably `--lr`, whose absence in the
original code made the transfer/baseline learning-rate gap impossible to correct
at launch (paper/METHODS_ACTUAL.md section 2).
"""
from __future__ import annotations

import json
import sys

from .config import config_from_args
from .train import train


def main(argv=None) -> int:
    cfg = config_from_args(argv)
    print(f'=== {cfg.run_id()} on {cfg.env} ===')
    print(f'    arch={cfg.arch}  target_rule={cfg.target_rule}  '
          f'mode={cfg.mode}  lr={cfg.lr}  episodes={cfg.num_episodes}')
    manifest = train(cfg)
    print('\n=== result ===')
    print(json.dumps(manifest['result'], indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
