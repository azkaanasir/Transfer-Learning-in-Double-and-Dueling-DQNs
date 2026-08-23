"""Provenance capture: what code, on what machine, with what inputs.

The Phase 0 audit spent days reconstructing which configuration produced each
published number, and failed for one of four arms because the machine that ran
it was gone. Everything that reconstruction needed and could not find is
recorded here, at run time, in the run's own manifest.
"""
from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(('git', *args), cwd=_REPO, capture_output=True,
                             text=True, timeout=15)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:                                   # noqa: BLE001
        return None


def git_state() -> dict:
    """Commit, branch, and whether the tree was dirty.

    The dirty flag matters more than the commit: a result produced from an
    uncommitted tree is not reproducible from the repository, and that has to be
    visible in the artifact rather than discovered later.
    """
    status = _git('status', '--porcelain')
    return {
        'commit': _git('rev-parse', 'HEAD'),
        'branch': _git('rev-parse', '--abbrev-ref', 'HEAD'),
        'dirty': bool(status) if status is not None else None,
        'dirty_files': (len([l for l in status.splitlines() if l.strip()])
                        if status else 0),
    }


def package_versions() -> dict:
    out = {'python': sys.version.split()[0]}
    for mod in ('tensorflow', 'keras', 'gymnasium', 'numpy', 'scipy', 'pandas',
                'matplotlib', 'Box2D'):
        try:
            m = __import__(mod)
            out[mod] = str(getattr(m, '__version__', 'unknown'))
        except Exception:                               # noqa: BLE001
            out[mod] = None
    return out


def machine() -> dict:
    return {
        'platform': platform.platform(),
        'machine': platform.machine(),
        'processor': platform.processor(),
        'cpu_count': os.cpu_count(),
        'hostname': platform.node(),
    }


def determinism_env() -> dict:
    """The environment variables and framework switches that affect determinism.

    Recorded rather than promised. Bitwise reproducibility is achievable on an
    identical software and hardware stack and is *not* achievable across stacks:
    oneDNN is on by default in the pinned TensorFlow build and prints a warning
    to that effect on every import, and threading changes reduction order. The
    honest claim is "bitwise on an identical stack, statistically comparable
    across stacks", and these fields are the evidence for which one applies.
    """
    keys = ('TF_ENABLE_ONEDNN_OPTS', 'TF_DETERMINISTIC_OPS', 'PYTHONHASHSEED',
            'OMP_NUM_THREADS', 'TF_NUM_INTRAOP_THREADS',
            'TF_NUM_INTEROP_THREADS', 'CUDA_VISIBLE_DEVICES')
    out = {k: os.environ.get(k) for k in keys}
    try:
        import tensorflow as tf
        out['tf_intra_op_threads'] = tf.config.threading.get_intra_op_parallelism_threads()
        out['tf_inter_op_threads'] = tf.config.threading.get_inter_op_parallelism_threads()
        out['tf_gpus'] = len(tf.config.list_physical_devices('GPU'))
    except Exception:                                   # noqa: BLE001
        pass
    return out


def file_hash(path: str) -> str | None:
    """Content hash of a file, for pinning the analysis plan and reference data."""
    try:
        with open(path, 'rb') as fh:
            return hashlib.blake2b(fh.read(), digest_size=16).hexdigest()
    except OSError:
        return None


def plan_hashes() -> dict:
    """Hashes of the documents that govern how results may be analysed.

    `ANALYSIS_PLAN.md` is pre-registered, so a confirmatory result is only
    interpretable against the version of the plan in force when it ran. Storing
    the hash in every manifest is what lets `audit.py` detect that the plan
    changed after the fact.
    """
    base = os.path.join(_REPO, 'experiments')
    return {name: file_hash(os.path.join(base, name))
            for name in ('ANALYSIS_PLAN.md', 'DESIGN.md',
                         'reference_returns.json')}


def snapshot(argv: list[str] | None = None) -> dict:
    return {
        'git': git_state(),
        'packages': package_versions(),
        'machine': machine(),
        'determinism': determinism_env(),
        'plans': plan_hashes(),
        'argv': list(argv if argv is not None else sys.argv),
        'cwd': os.getcwd(),
    }


__all__ = ['snapshot', 'git_state', 'package_versions', 'machine',
           'determinism_env', 'plan_hashes', 'file_hash']
