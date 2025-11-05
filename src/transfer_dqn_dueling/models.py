"""Model builder for transfer use. Same DuelingCombineLayer and build function style."""
from typing import Any, List

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

# keep same combine layer as original
class DuelingCombineLayer(layers.Layer):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def call(self, inputs):
        value, advantage = inputs
        advantage_mean = tf.reduce_mean(advantage, axis=1, keepdims=True)
        return value + (advantage - advantage_mean)

    def get_config(self):
        base = super().get_config()
        return dict(base)

def build_dueling_model(state_dim: int, action_dim: int, learning_rate: float = 1e-3) -> keras.Model:
    """Create and compile a dueling DQN network — same architecture as your model.py."""
    inputs = layers.Input(shape=(state_dim,), name='state_input')
    x = layers.Dense(128, activation='relu', name='shared_fc1')(inputs)
    x = layers.Dense(128, activation='relu', name='shared_fc2')(x)

    value = layers.Dense(64, activation='relu', name='value_fc')(x)
    value = layers.Dense(1, name='value_output')(value)

    advantage = layers.Dense(64, activation='relu', name='advantage_fc')(x)
    advantage = layers.Dense(action_dim, name='advantage_output')(advantage)

    outputs = DuelingCombineLayer(name='dueling_combiner')([value, advantage])

    model = keras.Model(inputs=inputs, outputs=outputs, name='dueling_dqn')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanSquaredError()]
    )
    return model


def _get_dense_layers(model: keras.Model) -> List[keras.layers.Layer]:
    return [l for l in model.layers if isinstance(l, keras.layers.Dense)]


def transfer_weights_from_source(
    target_model: keras.Model,
    source_model_path: str,
    partial_copy_first_dense: bool = True
):
    """
    Load source model (with custom object) and copy weights into target model where possible.
    - If shapes match exactly for a Dense layer: copy directly.
    - If shapes mismatch for the first Dense (input->units) and partial_copy_first_dense is True:
      copy overlapping portion and init remaining rows with GlorotUniform.
    """
    try:
        source = keras.models.load_model(source_model_path, custom_objects={'DuelingCombineLayer': DuelingCombineLayer})
    except Exception as e:
        raise RuntimeError(f"Failed to load source model at {source_model_path}: {e}")

    source_denses = _get_dense_layers(source)
    target_denses = _get_dense_layers(target_model)

    # Transfer by aligning the sequence of Dense layers (model architectures share same order).
    import tensorflow as _tf
    initializer = _tf.keras.initializers.GlorotUniform()

    for i, (t_layer, s_layer) in enumerate(zip(target_denses, source_denses)):
        try:
            s_wb = s_layer.get_weights()  # [kernel, bias]
            if not s_wb:
                continue
            t_wb = t_layer.get_weights()
            s_kernel, s_bias = s_wb[0], s_wb[1]
            t_kernel_shape = t_wb[0].shape
            t_bias_shape = t_wb[1].shape

            # If same shape -> direct copy
            if s_kernel.shape == t_kernel_shape:
                new_kernel = s_kernel
            else:
                # If this is the first dense and partial copy allowed, copy overlap
                if i == 0 and partial_copy_first_dense:
                    new_kernel = np.zeros(t_kernel_shape, dtype=s_kernel.dtype)
                    min_rows = min(s_kernel.shape[0], t_kernel_shape[0])
                    min_cols = min(s_kernel.shape[1], t_kernel_shape[1])
                    new_kernel[:min_rows, :min_cols] = s_kernel[:min_rows, :min_cols]
                    # initialize the remaining entries
                    extra = initializer(shape=t_kernel_shape).numpy()
                    mask = np.ones_like(new_kernel)
                    mask[:min_rows, :min_cols] = 0
                    new_kernel = new_kernel + extra * mask
                else:
                    # shapes incompatible and not allowed to partial copy -> initialize freshly
                    new_kernel = initializer(shape=t_kernel_shape).numpy()

            # biases
            if s_bias.shape == t_bias_shape:
                new_bias = s_bias
            else:
                new_bias = np.zeros(t_bias_shape, dtype=s_bias.dtype)
                min_b = min(len(s_bias), len(new_bias))
                new_bias[:min_b] = s_bias[:min_b]

            t_layer.set_weights([new_kernel, new_bias])
        except Exception as e:
            # Non-fatal: continue but report
            print(f"[transfer] skipped dense index {i} ({t_layer.name}) due to: {e}")

    print("[transfer] weight copy complete (partial copies applied where necessary).")
