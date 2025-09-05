"""Configuration and constants for the Dueling DQN project."""
import os

# Path where models, checkpoints and plots are stored
BASE_SAVE_PATH = os.path.join('.', 'models', 'dueling_dqn_cartpole')
os.makedirs(BASE_SAVE_PATH, exist_ok=True)

# Default hyperparameters (can be overridden via CLI)
DEFAULTS = {
    'gamma': 0.99,
    'epsilon': 1.0,
    'epsilon_min': 0.01,
    'epsilon_decay': 0.995,
    'buffer_size': 10_000,
    'batch_size': 64,
    'target_update_freq': 10,
    'learning_rate': 1e-3,
}

# Filenames used for checkpointing
MODEL_LATEST = 'dueling_model_latest.h5'
TARGET_MODEL_LATEST = 'dueling_target_model_latest.h5'
REPLAY_BUFFER = 'replay_buffer_latest.pkl'
METADATA = 'training_metadata_latest.json'
RESULTS_PNG = 'dueling_dqn_cartpole_results.png'
TB_LOG_DIR = "cartpole_dqn_duelling_logs"
