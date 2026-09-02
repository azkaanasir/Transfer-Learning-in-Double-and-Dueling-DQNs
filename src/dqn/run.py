"""Single-run entry point.

    python -m src.dqn.run --arch dueling --target-rule double \
        --env LunarLander-v3 --condition scratch --seed 0

    python -m src.dqn.run --arch dueling --target-rule double \
        --env LunarLander-v3 --condition transfer --seed 0 \
        --source-env CartPole-v1 \
        --source-checkpoint runs/scratch/<run_digest12>/s00/model.keras

Every Config field is exposed as a flag -- notably `--lr`, whose absence in the
original code made the transfer/baseline learning-rate gap impossible to correct
at launch (paper/METHODS_ACTUAL.md section 2).

Two things in the usage above were wrong until this revision, and both of them
made the module unusable rather than merely misdocumented.

* The banner printed `cfg.mode`, and `Config` has no `mode` field. The factor
  is `condition`: scratch, transfer, transfer_untrained, transfer_permuted, and
  the three control conditions are exactly why it cannot be a two-valued
  "mode". So this entry point raised AttributeError on its second print, before
  `train()` was ever reached, and could not start a run at all. `sweep.py`
  calls `train()` directly and was unaffected, which is how it survived.
* The flag is `--env`, not `--env-id`, and a source model lives at
  `<out_root>/<condition>/<run_digest12>/s<NN>/model.keras`: runs are keyed by
  configuration digest (`Config.run_dir`), never by environment name.

`train()` raises rather than returning when a run cannot be finalised
soundly -- a checkpoint that is not one checkpoint, or a held-out evaluation
set that cannot be reconstructed after a resume. The traceback is the intended
behaviour: the process exits non-zero and no manifest is written, so the
directory reads as unfinished to `sweep.py` and `aggregate.py` rather than as a
finished run carrying a number nothing can vouch for.
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
          f'condition={cfg.condition}  lr={cfg.lr}  '
          f'episodes={cfg.num_episodes}')
    print(f'    run_dir={cfg.run_dir()}')
    manifest = train(cfg)
    print('\n=== result ===')
    print(json.dumps(manifest['result'], indent=2, default=str))
    return 0


if __name__ == '__main__':
    sys.exit(main())
