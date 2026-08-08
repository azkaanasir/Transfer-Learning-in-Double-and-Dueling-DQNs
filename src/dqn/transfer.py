"""Layer-wise weight transfer and the freeze schedule.

Three defects in the original implementation are fixed here, all documented in
paper/METHODS_ACTUAL.md:

1. Transfer aligned source and target Dense layers by *position* (`zip`), so a
   depth mismatch between architectures silently dropped layers. Here transfer
   keys on layer names and reports every decision.
2. The freeze set was resolved by indexing into `model.layers`, whose order for
   a branched functional model put `value_fc` and `adv_fc` adjacent -- freezing
   both head hiddens rather than the intended trunk. Here layers are named.
3. There was no episode-indexed schedule at all: `trainable` was set once at
   construction and never changed, so the manuscript's "frozen for the first
   100 episodes, then unfrozen" never happened. `apply_freeze` implements it.
"""
from __future__ import annotations

import numpy as np
from tensorflow import keras

from .networks import CUSTOM_OBJECTS


def load_source(path: str) -> keras.Model:
    return keras.models.load_model(path, custom_objects=CUSTOM_OBJECTS,
                                   compile=False)


def transfer_weights(target: keras.Model,
                     source: keras.Model,
                     layer_names=('trunk_fc1', 'trunk_fc2'),
                     first_layer_policy: str = 'partial',
                     rng: np.random.Generator | None = None) -> list[dict]:
    """Copy weights source -> target for the named layers.

    Returns one record per requested layer describing what happened, which the
    caller writes into the run manifest. Nothing is ever copied silently.
    """
    rng = rng or np.random.default_rng(0)
    report = []

    for name in layer_names:
        rec = {'layer': name, 'action': None, 'detail': ''}
        try:
            s_layer = source.get_layer(name)
            t_layer = target.get_layer(name)
        except ValueError:
            rec.update(action='skipped', detail='absent from source or target')
            report.append(rec)
            continue

        s_w = s_layer.get_weights()
        t_w = t_layer.get_weights()
        if not s_w or not t_w:
            rec.update(action='skipped', detail='layer has no weights')
            report.append(rec)
            continue

        s_kernel, s_bias = s_w[0], s_w[1]
        t_kernel, t_bias = t_w[0], t_w[1]

        if s_kernel.shape == t_kernel.shape:
            t_layer.set_weights([s_kernel, s_bias])
            rec.update(action='copied', detail=f'exact {s_kernel.shape}')
        elif first_layer_policy == 'reinit':
            rec.update(action='reinit',
                       detail=f'shape {s_kernel.shape} -> {t_kernel.shape}')
        else:
            # Partial copy: keep the overlapping block, Glorot-initialise the
            # rest. Only sensible for the input-facing layer, where the source
            # and target observation dimensionalities differ.
            rows = min(s_kernel.shape[0], t_kernel.shape[0])
            cols = min(s_kernel.shape[1], t_kernel.shape[1])
            limit = np.sqrt(6.0 / (t_kernel.shape[0] + t_kernel.shape[1]))
            new_kernel = rng.uniform(-limit, limit,
                                     size=t_kernel.shape).astype(t_kernel.dtype)
            new_kernel[:rows, :cols] = s_kernel[:rows, :cols]
            new_bias = np.array(t_bias, dtype=t_bias.dtype)
            new_bias[:min(len(s_bias), len(new_bias))] = \
                s_bias[:min(len(s_bias), len(new_bias))]
            t_layer.set_weights([new_kernel, new_bias])
            rec.update(action='partial',
                       detail=f'copied [{rows}x{cols}] of {t_kernel.shape}, '
                              f'remainder Glorot')
        report.append(rec)

    return report


def apply_freeze(model: keras.Model, layer_names, frozen: bool) -> list[str]:
    """Set `trainable` on the named layers. Returns the names actually changed.

    Because training runs through an explicit GradientTape over
    `model.trainable_variables` rather than `model.fit`, this takes effect
    immediately -- no recompile, and the optimiser's slot variables survive the
    transition. The caller must re-trace its tf.function, since the set of
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

    Logged every time the freeze state changes so that the actual trainable
    surface is a recorded fact rather than something inferred from config.
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
