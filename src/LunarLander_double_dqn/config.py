"""Configuration and hyperparameters."""
import os

# Path where models, checkpoints and plots are stored
BASE_SAVE_PATH = os.path.join('.', 'models', 'dueling_dqn_lunarlander')
os.makedirs(BASE_SAVE_PATH, exist_ok=True)


lunar_params = {
    'env_id': 'LunarLander-v3',
    'reward_threshold': 200,
    'num_episodes': 1000,
    'video_format': ['mp4', 'gif'],              #mp4 prolly wont work so we will work with gifs
    'lr': 0.0005,
    'gamma': 0.99,
    'batch_size': 64,
    'replay_buffer_size': 100000,
    'epsilon_start': 1.0,
    'epsilon_min': 0.01,
    'epsilon_decay': 0.95,                       #else try 0.990
    'target_update_freq': 1000,
    'soft_update_tau': 0.001,
    'optimizer': 'adam'
}

# Filenames used for checkpointing
MODEL_LATEST = 'dueling_model.h5'
TARGET_MODEL_LATEST = 'dueling_target_model.h5'
REPLAY_BUFFER = 'replay_buffer.pkl'
METADATA = 'training_metadata.json'
RESULTS_PNG = 'dueling_dqn_lunarlander_results.png'
TB_LOG_DIR = "logs/lunarlander_dqn_duelling"