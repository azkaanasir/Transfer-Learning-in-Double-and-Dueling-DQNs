from tensorflow import keras
from keras import layers


def build_ddqn_model(state_dim: int, action_dim: int, learning_rate: float):
    inputs = layers.Input(shape=(state_dim,), name="state")
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(action_dim, name="q_values")(x)

    model = keras.Model(inputs, outputs, name="ddqn")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanSquaredError()]
    )
    return model
