"""Layer-wise weight transfer, the control conditions, and the freeze schedule.

Four defects in the published implementation are fixed here, all documented in
paper/METHODS_ACTUAL.md:

1. Transfer aligned source and target Dense layers by *position* (`zip`), so a
   depth mismatch between architectures silently dropped layers. Here transfer
   keys on names and reports every decision, including how many parameters were
   actually copied.
2. The freeze set was resolved by indexing into `model.layers`, whose order for
   a branched functional model puts `value_fc` and `adv_fc` adjacent -- so a
   rule meant to spare the trunk froze both head hiddens. Freeze sets are named
   groups now (`networks.resolve_layers`).
3. There was no episode-indexed schedule at all: `trainable` was set once at
   construction and never changed, so the manuscript's "frozen for the first
   100 episodes, then unfrozen" never happened.
4. A frozen layer was never *verified* to be frozen. `weight_fingerprint`
   makes the freeze a checkable fact: ROADMAP task 0.3 had to recover the
   published freeze map by diffing checkpoints months later, which is only
   necessary when the run does not record it.

The control conditions (`DESIGN.md` section 4) also live here. They are what
turn "we observed negative transfer" into "we measured how much of it is
learned structure and how much is protocol mechanics":

    transfer            trained source
    transfer_untrained  randomly initialised source of the same shape
    transfer_permuted   trained source with each transferred kernel shuffled

`transfer_untrained` is distributionally equivalent to freezing the target's own
fresh initialisation, which is exactly why it is interpretable: it carries the
protocol's mechanics -- the partial copy, the reinitialised head, the freeze
window -- with no learned content to carry. `transfer_permuted` keeps the
trained weights' multiset, and therefore their scale and marginal distribution,
while destroying the structure. The difference between them is the only
quantity in the study that deserves the name "transferred knowledge".
"""
from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np
import keras

from .networks import CUSTOM_OBJECTS, build_q_network, resolve_layers

CONDITIONS = ('scratch', 'transfer', 'transfer_untrained', 'transfer_permuted')
PERMUTE_SCOPES = ('all', 'units')
VALUE_RECAL = ('none', 'center', 'center_scale')


def load_source(path: str) -> keras.Model:
    return keras.models.load_model(path, custom_objects=CUSTOM_OBJECTS,
                                   compile=False)


# ---------------------------------------------------------------------------
# Control-condition source construction
# ---------------------------------------------------------------------------
def untrained_source(state_dim: int, action_dim: int, arch: str,
                     hidden, head_units: int, aggregation: str,
                     seed: int) -> keras.Model:
    """A randomly initialised stand-in for a trained source (control C2).

    Built at the *source task's* interface, not the target's, so that the copy
    it feeds into performs the identical mechanics as a real transfer: the same
    partial copy on the input-facing layer, the same head reinitialisation, the
    same freeze window. Only the training is removed.

    Seeded from a dedicated stream so that constructing this control cannot
    perturb the target network's own initialisation.
    """
    keras.utils.set_random_seed(int(seed))
    return build_q_network(state_dim, action_dim, arch, hidden, head_units,
                           aggregation)


def permute_source(source: keras.Model, layer_names: Iterable[str],
                   rng: np.random.Generator,
                   scope: str = 'all') -> tuple[keras.Model, list[dict]]:
    """Destroy learned structure while preserving weight statistics (control C3).

    scope='all'    shuffle every entry of the kernel. Preserves the multiset of
                   weights exactly -- hence the norm, the scale and the marginal
                   distribution -- and destroys all structure. This is the
                   default because it is the control that isolates *structure*.
    scope='units'  permute output columns only, preserving each unit's incoming
                   weight vector and scrambling which downstream unit reads it.
                   A deliberately weaker control: a trainable downstream layer
                   can partly undo a column permutation, so a null result here
                   means less than a null result under 'all'.

    Mutates and returns `source`, plus a record of what was permuted.
    """
    if scope not in PERMUTE_SCOPES:
        raise ValueError(f'scope must be one of {PERMUTE_SCOPES}, got {scope!r}')
    report = []
    for name in layer_names:
        try:
            layer = source.get_layer(name)
        except ValueError:
            continue
        weights = layer.get_weights()
        if not weights:
            continue
        kernel = np.array(weights[0])
        before = _fingerprint_array(kernel)
        if scope == 'all':
            flat = kernel.reshape(-1)
            kernel = rng.permutation(flat).reshape(kernel.shape)
        else:
            kernel = kernel[:, rng.permutation(kernel.shape[1])]
        new = [kernel] + [np.array(w) for w in weights[1:]]
        if len(new) > 1 and scope == 'all':
            new[1] = rng.permutation(np.array(weights[1]))
        layer.set_weights(new)
        report.append({
            'layer': name, 'scope': scope,
            'kernel_shape': list(kernel.shape),
            'fingerprint_before': before,
            'fingerprint_after': _fingerprint_array(kernel),
            # Preserved by construction under scope='all'; recorded so the claim
            # "scale is unchanged" is a measurement rather than an assertion.
            'frobenius_norm': float(np.linalg.norm(kernel)),
        })
    return source, report


def spectrum_matched_source(source: keras.Model, layer_names: Iterable[str],
                            rng: np.random.Generator
                            ) -> tuple[keras.Model, list[dict]]:
    """Replace each transferred kernel with a random matrix of the same spectrum.

    Control C3b. The entry-wise shuffle in `permute_source` preserves the
    multiset of weights -- and therefore the Frobenius norm -- but it does *not*
    preserve the singular-value spectrum, the per-row norms or the per-column
    norms. So the permuted-source contrast absorbs spectral effects along with
    the structural ones, and the interpretation "this contrast isolates learned
    structure" rests on an assumption nobody checked.

    This control removes that assumption empirically rather than by argument:
    W' = U S V^T with S the source layer's singular values and U, V drawn
    Haar-uniform. Structure is destroyed, the spectrum is preserved exactly. If
    C3 and C3b agree, the spectral caveat is void; if they disagree, the
    disagreement is reported instead of assumed away.
    """
    report = []
    for name in layer_names:
        try:
            layer = source.get_layer(name)
        except ValueError:
            continue
        weights = layer.get_weights()
        if not weights:
            continue
        kernel = np.asarray(weights[0], dtype=np.float64)
        if kernel.ndim != 2:
            continue
        u_s, sing, vt = np.linalg.svd(kernel, full_matrices=False)
        q1 = _haar(rng, kernel.shape[0], len(sing))
        q2 = _haar(rng, kernel.shape[1], len(sing))
        new = (q1 * sing) @ q2.T
        new = new.astype(np.asarray(weights[0]).dtype)
        rest = [np.asarray(w) for w in weights[1:]]
        if rest:
            rest[0] = rng.permutation(rest[0])
        layer.set_weights([new] + rest)
        report.append({
            'layer': name, 'scope': 'spectrum',
            'kernel_shape': list(new.shape),
            'singular_values_preserved': bool(np.allclose(
                np.linalg.svd(new.astype(np.float64), compute_uv=False),
                sing, rtol=1e-6, atol=1e-8)),
            'frobenius_norm': float(np.linalg.norm(new)),
            'frobenius_norm_source': float(np.linalg.norm(kernel)),
        })
    return source, report


def _haar(rng: np.random.Generator, rows: int, cols: int) -> np.ndarray:
    """A Haar-uniform matrix with orthonormal columns, via QR of a Gaussian."""
    a = rng.standard_normal((rows, cols))
    q, r = np.linalg.qr(a)
    # Sign-fix the diagonal, without which QR is not Haar-distributed.
    return q * np.sign(np.diag(r))


# ---------------------------------------------------------------------------
# Weight transfer
# ---------------------------------------------------------------------------
def transfer_weights(target: keras.Model,
                     source: keras.Model,
                     layer_names=('trunk_fc1', 'trunk_fc2'),
                     input_policy: str = 'partial',
                     head_policy: str = 'reinit',
                     rng: np.random.Generator | None = None) -> list[dict]:
    """Copy weights source -> target for the named layers.

    Three shape relationships are possible and they are *not* the same decision,
    which is why they are separate policies:

    * **exact** -- copy.
    * **input-facing mismatch** (`trunk_fc1`, whose kernel is
      (state_dim, width)): the source and target observation dimensionalities
      differ. `input_policy='partial'` keeps the overlapping row block and
      Glorot-initialises the rest, which is what the published runs did;
      `'reinit'` skips the layer, which is what the manuscript described.
    * **output-facing mismatch** (`q_out`, `adv_out`, whose kernel is
      (width, action_dim)): the action cardinalities differ.
      `head_policy='reinit'` leaves the head fresh; `'partial'` copies the
      overlapping column block. The published code conflated this with the
      input case under one flag, so a head could be partially copied without
      anyone choosing that.

    Returns one record per requested layer, including `params_copied` as a
    count, so the manifest can support a statement about *how much* of the
    network was actually transferred instead of implying all of it was.
    """
    if input_policy not in ('partial', 'reinit', 'redraw_matched'):
        raise ValueError('input_policy must be partial|reinit|redraw_matched, '
                         f'got {input_policy!r}')
    if head_policy not in ('partial', 'reinit'):
        raise ValueError(f'head_policy must be partial|reinit, got {head_policy!r}')
    rng = rng or np.random.default_rng(0)
    report: list[dict] = []

    for name in layer_names:
        rec: dict = {'layer': name, 'action': None, 'detail': '',
                     'params_copied': 0, 'params_total': 0}
        try:
            s_layer = source.get_layer(name)
            t_layer = target.get_layer(name)
        except ValueError:
            rec.update(action='skipped', detail='absent from source or target')
            report.append(rec)
            continue

        s_w, t_w = s_layer.get_weights(), t_layer.get_weights()
        if not s_w or not t_w:
            rec.update(action='skipped', detail='layer has no weights')
            report.append(rec)
            continue

        s_kernel, s_bias = np.array(s_w[0]), np.array(s_w[1])
        t_kernel, t_bias = np.array(t_w[0]), np.array(t_w[1])
        rec['params_total'] = int(t_kernel.size + t_bias.size)
        rec['source_shape'] = list(s_kernel.shape)
        rec['target_shape'] = list(t_kernel.shape)

        if s_kernel.shape == t_kernel.shape:
            if input_policy == 'redraw_matched' and name == 'trunk_fc1':
                # Protocol matching for the same-interface arm. Under a
                # cross-interface pair such as CartPole(4) -> LunarLander(8),
                # half the input rows of `trunk_fc1` are freshly drawn because
                # the source has no weights for them. A same-interface arm would
                # otherwise copy the layer intact, so it would differ from the
                # cross-interface arm in *shift and protocol at once* -- which is
                # exactly the confound the same-interface arm exists to remove.
                # Re-drawing the upper half of the rows reproduces the same
                # initialisation perturbation at the same size.
                half = t_kernel.shape[0] // 2
                limit = np.sqrt(6.0 / (t_kernel.shape[0] + t_kernel.shape[1]))
                new_kernel = np.array(s_kernel, dtype=t_kernel.dtype)
                new_kernel[half:, :] = rng.uniform(
                    -limit, limit,
                    size=(t_kernel.shape[0] - half, t_kernel.shape[1])
                ).astype(t_kernel.dtype)
                t_layer.set_weights([new_kernel, s_bias])
                rec.update(action='partial',
                           detail=f'redraw_matched: kept rows [0:{half}] of '
                                  f'{tuple(t_kernel.shape)}, redrew the rest to '
                                  f'size-match a cross-interface partial copy',
                           params_copied=int(half * t_kernel.shape[1]
                                             + s_bias.size))
                report.append(rec)
                continue
            t_layer.set_weights([s_kernel, s_bias])
            rec.update(action='copied', detail=f'exact {tuple(s_kernel.shape)}',
                       params_copied=int(s_kernel.size + s_bias.size))
            report.append(rec)
            continue

        rows_differ = s_kernel.shape[0] != t_kernel.shape[0]
        cols_differ = s_kernel.shape[1] != t_kernel.shape[1]
        policy = input_policy if rows_differ else head_policy
        which = 'input-facing' if rows_differ else 'output-facing'
        if rows_differ and cols_differ:
            # Both ends differ. Only the input policy can sensibly apply, and
            # copying into a layer whose fan-out also changed is not a
            # meaningful representation transfer, so it is refused.
            rec.update(action='skipped',
                       detail=f'both dimensions differ '
                              f'{tuple(s_kernel.shape)} -> {tuple(t_kernel.shape)}; '
                              f'not a well-defined layer correspondence')
            report.append(rec)
            continue

        if policy == 'reinit':
            rec.update(action='reinit',
                       detail=f'{which} mismatch '
                              f'{tuple(s_kernel.shape)} -> {tuple(t_kernel.shape)}, '
                              f'left at target initialisation')
            report.append(rec)
            continue

        rows = min(s_kernel.shape[0], t_kernel.shape[0])
        cols = min(s_kernel.shape[1], t_kernel.shape[1])
        limit = np.sqrt(6.0 / (t_kernel.shape[0] + t_kernel.shape[1]))
        new_kernel = rng.uniform(-limit, limit,
                                 size=t_kernel.shape).astype(t_kernel.dtype)
        new_kernel[:rows, :cols] = s_kernel[:rows, :cols]
        new_bias = np.array(t_bias, dtype=t_bias.dtype)
        keep = min(len(s_bias), len(new_bias))
        new_bias[:keep] = s_bias[:keep]
        t_layer.set_weights([new_kernel, new_bias])
        rec.update(action='partial',
                   detail=f'{which} mismatch: copied [{rows}x{cols}] of '
                          f'{tuple(t_kernel.shape)}, remainder Glorot',
                   params_copied=int(rows * cols + keep))
        report.append(rec)

    return report


def transfer_summary(report: list[dict], model: keras.Model) -> dict:
    """Aggregate a transfer report into the numbers a methods section needs."""
    copied = sum(int(r.get('params_copied', 0)) for r in report)
    total = int(sum(int(np.prod(w.shape)) for w in model.weights))
    return {
        'params_copied': copied,
        'params_in_model': total,
        'fraction_of_model_transferred': (copied / total) if total else 0.0,
        'layers_copied': [r['layer'] for r in report if r['action'] == 'copied'],
        'layers_partial': [r['layer'] for r in report if r['action'] == 'partial'],
        'layers_reinit': [r['layer'] for r in report if r['action'] == 'reinit'],
        'layers_skipped': [r['layer'] for r in report if r['action'] == 'skipped'],
    }


# ---------------------------------------------------------------------------
# Value-head recalibration probe (DESIGN.md E11)
# ---------------------------------------------------------------------------
def recalibrate_value_head(model: keras.Model, states: np.ndarray,
                           mode: str = 'center') -> dict:
    """Rescale the transferred value head to the target task's return range.

    A source-trained V(s) carries the source's return scale in its output bias.
    Under `Q = V + (A - mean A)`, adding a constant to V shifts every Q equally,
    so this intervention **does not change the greedy policy at all** -- stating
    that plainly matters, because the tempting claim "recalibration improves the
    initial policy" would be false by construction. What it changes is the
    magnitude of the TD targets relative to the target task's rewards, and hence
    the early optimisation dynamics. That is the hypothesis being tested.

    mode='center'        subtract the mean V over `states` (offset only)
    mode='center_scale'  also divide the head's kernel and bias by the observed
                         SD of V, giving unit-scale values before learning
    """
    if mode not in VALUE_RECAL:
        raise ValueError(f'mode must be one of {VALUE_RECAL}, got {mode!r}')
    if mode == 'none':
        return {'applied': False, 'mode': mode}
    try:
        layer = model.get_layer('value_out')
    except ValueError:
        return {'applied': False, 'mode': mode, 'reason': 'no value_out layer'}

    from .networks import build_stream_probe
    probe = build_stream_probe(model)
    if probe is None:
        return {'applied': False, 'mode': mode, 'reason': 'no dueling streams'}

    v_before, _ = probe(np.asarray(states, dtype=np.float32), training=False)
    v_before = np.asarray(v_before).reshape(-1)
    kernel, bias = [np.array(w) for w in layer.get_weights()]
    mean, sd = float(v_before.mean()), float(v_before.std())

    if mode == 'center_scale' and sd > 1e-8:
        kernel = kernel / sd
        bias = (bias - mean) / sd
    else:
        bias = bias - mean
    layer.set_weights([kernel, bias])

    v_after, _ = probe(np.asarray(states, dtype=np.float32), training=False)
    v_after = np.asarray(v_after).reshape(-1)
    return {'applied': True, 'mode': mode,
            'v_mean_before': mean, 'v_sd_before': sd,
            'v_mean_after': float(v_after.mean()),
            'v_sd_after': float(v_after.std()),
            'n_states': int(len(v_before)),
            'policy_invariant': True}


# ---------------------------------------------------------------------------
# Freezing
# ---------------------------------------------------------------------------
def apply_freeze(model: keras.Model, layer_names, frozen: bool) -> list[str]:
    """Set `trainable` on the named layers. Returns the names actually changed.

    Because training runs through an explicit `GradientTape` over
    `model.trainable_variables` rather than `model.fit`, this takes effect
    immediately: no recompile, and the optimiser's slot variables survive the
    transition. The caller must re-trace its `tf.function`, since the set of
    differentiated variables changes.
    """
    changed = []
    for name in layer_names:
        try:
            layer = model.get_layer(name)
        except ValueError:
            continue
        if layer.trainable != (not frozen):
            layer.trainable = not frozen
            changed.append(name)
    return changed


def trainable_report(model: keras.Model) -> dict:
    """Per-layer trainable flags and parameter counts, for the run manifest.

    Logged every time the freeze state changes, so the actual trainable surface
    is a recorded fact rather than something inferred from a config field.
    """
    layers_info, n_train, n_frozen = {}, 0, 0
    for layer in model.layers:
        if not layer.weights:
            continue
        count = int(sum(np.prod(w.shape) for w in layer.weights))
        layers_info[layer.name] = {'trainable': bool(layer.trainable),
                                   'params': count}
        if layer.trainable:
            n_train += count
        else:
            n_frozen += count
    return {'layers': layers_info,
            'trainable_params': n_train,
            'frozen_params': n_frozen}


# ---------------------------------------------------------------------------
# Freeze verification
# ---------------------------------------------------------------------------
def _fingerprint_array(arr: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(arr, dtype=np.float32)
                        .tobytes()).hexdigest()[:16]


def weight_fingerprint(model: keras.Model) -> dict[str, str]:
    """Per-layer content hash of the weights.

    Recorded at every freeze transition so that `audit.py` can *verify* a layer
    held constant across the window it was declared frozen for. Recovering the
    published study's freeze map required diffing saved checkpoints months after
    the fact; a run that records this cannot have an unrecoverable freeze map.
    """
    out = {}
    for layer in model.layers:
        if not layer.weights:
            continue
        out[layer.name] = _fingerprint_array(
            np.concatenate([np.asarray(w).reshape(-1) for w in layer.get_weights()]))
    return out


def verify_freeze(before: dict[str, str], after: dict[str, str],
                  expected_frozen: Iterable[str]) -> dict:
    """Check that declared-frozen layers are unchanged and others moved.

    Both directions matter. An unchanged trainable layer is as much a defect as
    a changed frozen one -- it usually means the optimiser never received its
    gradients, which is precisely the failure mode a `zip`-based freeze produces
    silently.
    """
    expected = set(expected_frozen)
    violations, inert = [], []
    for name, digest in after.items():
        was = before.get(name)
        if was is None:
            continue
        changed = digest != was
        if name in expected and changed:
            violations.append(name)
        if name not in expected and not changed:
            inert.append(name)
    return {'frozen_but_changed': sorted(violations),
            'trainable_but_unchanged': sorted(inert),
            'ok': not violations}


def resolve_transfer_and_freeze(arch: str, transfer_layers, freeze_layers):
    """Canonicalise both layer sets, and refuse an incoherent combination.

    Freezing a layer that was never transferred is a different experiment from
    the one the config appears to describe: it holds a *randomly initialised*
    layer fixed. That may be deliberate -- it is the shape of control C2 -- but
    it must be requested explicitly rather than arrived at by a typo, so the
    mismatch is surfaced to the caller for the manifest instead of being
    normalised away.
    """
    t = resolve_layers(arch, transfer_layers)
    f = resolve_layers(arch, freeze_layers)
    return t, f, {'frozen_without_transfer': tuple(n for n in f if n not in t),
                  'transferred_without_freeze': tuple(n for n in t if n not in f)}


__all__ = ['CONDITIONS', 'PERMUTE_SCOPES', 'VALUE_RECAL', 'load_source',
           'untrained_source', 'permute_source', 'spectrum_matched_source',
           'transfer_weights',
           'transfer_summary', 'recalibrate_value_head', 'apply_freeze',
           'trainable_report', 'weight_fingerprint', 'verify_freeze',
           'resolve_transfer_and_freeze']
