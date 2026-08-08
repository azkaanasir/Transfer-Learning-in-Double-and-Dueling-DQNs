"""Q-network builders for the 2x2 design.

Both architectures share an identically-named trunk (`trunk_fc1`, `trunk_fc2`)
so that weight transfer can key on layer *names*. The original implementation
zipped the source and target Dense layers positionally, which silently dropped
layers whenever the two architectures had different depths -- see
paper/METHODS_ACTUAL.md section 5.1.

    mlp      state -> trunk_fc1(128) -> trunk_fc2(128) -> q_out(|A|)

    dueling  state -> trunk_fc1(128) -> trunk_fc2(128) -> value_fc(64) -> value_out(1)
                                                       \\-> adv_fc(64)  -> adv_out(|A|)
             Q = V + (A - mean_a A)
"""
from __future__ import annotations

from typing import Any, Sequence

import tensorflow as tf
from tensorflow import keras
from keras import layers

TRUNK_LAYERS = ('trunk_fc1', 'trunk_fc2')


class DuelingAggregation(layers.Layer):
    """Q(s,a) = V(s) + (A(s,a) - mean_a' A(s,a'))."""

    def call(self, inputs):
        value, advantage = inputs
        return value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))

    def get_config(self):
        return dict(super().get_config())


def build_q_network(state_dim: int,
                    action_dim: int,
                    arch: str = 'dueling',
                    hidden: Sequence[int] = (128, 128),
                    head_units: int = 64) -> keras.Model:
    """Build an uncompiled Q-network. Training uses an explicit GradientTape,
    so the model is deliberately left uncompiled -- this is what lets
    `layer.trainable` be toggled mid-run without discarding optimiser state."""
    if len(hidden) != 2:
        raise ValueError(f'expected two trunk widths, got {hidden!r}')

    inputs = layers.Input(shape=(state_dim,), name='state')
    x = layers.Dense(hidden[0], activation='relu', name='trunk_fc1')(inputs)
    x = layers.Dense(hidden[1], activation='relu', name='trunk_fc2')(x)

    if arch == 'mlp':
        outputs = layers.Dense(action_dim, name='q_out')(x)
    elif arch == 'dueling':
        v = layers.Dense(head_units, activation='relu', name='value_fc')(x)
        v = layers.Dense(1, name='value_out')(v)
        a = layers.Dense(head_units, activation='relu', name='adv_fc')(x)
        a = layers.Dense(action_dim, name='adv_out')(a)
        outputs = DuelingAggregation(name='dueling_aggregation')([v, a])
    else:
        raise ValueError(f'unknown arch {arch!r}')

    return keras.Model(inputs, outputs, name=f'{arch}_q')


def build_stream_probe(model: keras.Model) -> keras.Model | None:
    """A model exposing (V, A) for diagnostics. None for non-dueling nets.

    Phase 2 needs V/A magnitude trajectories; capturing them requires the
    intermediate tensors, which is cheap to wire up now and awkward later.
    """
    try:
        v = model.get_layer('value_out').output
        a = model.get_layer('adv_out').output
    except ValueError:
        return None
    return keras.Model(model.input, [v, a], name='stream_probe')


CUSTOM_OBJECTS: dict[str, Any] = {
    'DuelingAggregation': DuelingAggregation,
    # the original checkpoints used this name; kept so old models still load
    'DuelingCombineLayer': DuelingAggregation,
}
