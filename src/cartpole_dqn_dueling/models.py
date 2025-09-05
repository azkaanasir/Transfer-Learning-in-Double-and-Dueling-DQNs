"""Model definitions: DuelingCombineLayer and a builder for the dueling network."""
from typing import Any

import tensorflow as tf
from tensorflow import keras
from keras import layers


class DuelingCombineLayer(layers.Layer):
    """Combine value and advantage streams into Q-values.

    Q(s, a) = V(s) + (A(s,a) - mean_a A(s,a))
    """
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
    """Create and compile a dueling DQN network."""
    inputs = layers.Input(shape=(state_dim,), name='state_input')
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dense(128, activation='relu')(x)

    value = layers.Dense(64, activation='relu')(x)
    value = layers.Dense(1, name='value_output')(value)

    advantage = layers.Dense(64, activation='relu')(x)
    advantage = layers.Dense(action_dim, name='advantage_output')(advantage)

    outputs = DuelingCombineLayer(name='dueling_combiner')([value, advantage])

    model = keras.Model(inputs=inputs, outputs=outputs, name='dueling_dqn')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    return model

