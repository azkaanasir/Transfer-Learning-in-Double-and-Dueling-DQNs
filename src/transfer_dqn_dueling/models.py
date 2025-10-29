"""Model definitions and transfer-builder for safer weight copy."""

from typing import Optional, List, Tuple
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class DuelingCombineLayer(layers.Layer):
    def call(self, inputs):
        value, advantage = inputs
        advantage_mean = tf.reduce_mean(advantage, axis=1, keepdims=True)
        return value + (advantage - advantage_mean)

    def get_config(self):
        return super().get_config()


def build_dueling_model(state_dim: int, action_dim: int, learning_rate: float = 1e-3) -> keras.Model:
    """Full dueling model (no transfer)."""
    inputs = layers.Input(shape=(state_dim,), name='state_input')
    x = layers.Dense(128, activation='relu', name='dense_0')(inputs)
    x = layers.Dense(128, activation='relu', name='dense_1')(x)

    value = layers.Dense(64, activation='relu', name='value_dense')(x)
    value = layers.Dense(1, name='value_output')(value)

    advantage = layers.Dense(64, activation='relu', name='adv_dense')(x)
    advantage = layers.Dense(action_dim, name='advantage_output')(advantage)

    outputs = DuelingCombineLayer(name='dueling_combiner')([value, advantage])

    model = keras.Model(inputs=inputs, outputs=outputs, name='dueling_dqn')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanSquaredError()]
    )
    return model


# --------------------------
# Transfer builder (safe copy)
# --------------------------
def _build_trunk(state_dim: int, trunk_sizes: List[int] = [128, 128]) -> Tuple[keras.Model, layers.Layer]:
    """Return trunk model and trunk output tensor. Trunk layer names are deterministic."""
    inputs = layers.Input(shape=(state_dim,), name='trunk_input')
    x = inputs
    for i, h in enumerate(trunk_sizes):
        x = layers.Dense(h, activation='relu', name=f'trunk_dense_{i}')(x)
    trunk = keras.Model(inputs=inputs, outputs=x, name='dueling_trunk')
    return trunk, x


def _build_heads_from_tensor(trunk_output: layers.Layer, action_dim: int) -> layers.Layer:
    """Attach dueling heads to a trunk output tensor and return final output tensor."""
    v = layers.Dense(64, activation='relu', name='value_dense')(trunk_output)
    v = layers.Dense(1, name='value_output')(v)

    a = layers.Dense(64, activation='relu', name='adv_dense')(trunk_output)
    a = layers.Dense(action_dim, name='advantage_output')(a)

    out = DuelingCombineLayer(name='dueling_combiner')([v, a])
    return out


def build_transfer_dueling_model(
    state_dim: int,
    action_dim: int,
    pretrained_path: Optional[str] = None,
    freeze_base: bool = True,
    trunk_sizes: List[int] = [128, 128],
    learning_rate: float = 1e-3
) -> keras.Model:
    """
    Build a dueling model sized for (state_dim, action_dim).
    If pretrained_path is supplied, attempt to copy weights from it but only when shapes match.
    Freezes trunk layers if freeze_base True.
    """
    trunk, trunk_out = _build_trunk(state_dim, trunk_sizes)
    outputs = _build_heads_from_tensor(trunk_out, action_dim)
    model = keras.Model(inputs=trunk.input, outputs=outputs, name='dueling_transfer')

    copied = 0
    if pretrained_path:
        try:
            src = keras.models.load_model(pretrained_path, compile=False)
            # Strategy: first try exact layer name match; else try by matching weight shapes sequence.
            for src_layer in src.layers:
                try:
                    tgt_layer = model.get_layer(src_layer.name)
                except Exception:
                    tgt_layer = None

                src_w = src_layer.get_weights()
                if not src_w:
                    continue

                if tgt_layer is not None:
                    # check shapes
                    tgt_w = tgt_layer.get_weights()
                    if len(tgt_w) == len(src_w) and all(sw.shape == tw.shape for sw, tw in zip(src_w, tgt_w)):
                        try:
                            tgt_layer.set_weights(src_w)
                            copied += 1
                            continue
                        except Exception:
                            pass

                # fallback: try to find any layer in target with equal weight shapes
                for candidate in model.layers:
                    cand_w = candidate.get_weights()
                    if not cand_w or len(cand_w) != len(src_w):
                        continue
                    if all(sw.shape == cw.shape for sw, cw in zip(src_w, cand_w)):
                        try:
                            candidate.set_weights(src_w)
                            copied += 1
                            break
                        except Exception:
                            continue
            print(f"[transfer] copied weights for {copied} layers from '{pretrained_path}'")
        except Exception as e:
            print(f"[transfer] failed to load pretrained model '{pretrained_path}': {e}")

    # Freeze trunk layers if desired
    if freeze_base:
        for layer in trunk.layers:
            layer.trainable = False

    # compile with explicit losses/metrics
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanSquaredError()]
    )
    return model
