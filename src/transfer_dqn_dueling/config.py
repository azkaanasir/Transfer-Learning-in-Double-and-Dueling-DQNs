
"""Configuration and hyperparameters."""
import os

# Path where models, checkpoints and plots are stored
BASE_SAVE_PATH = os.path.join('.', 'models', 'dueling_dqn_lunarlander')
os.makedirs(BASE_SAVE_PATH, exist_ok=True)

# TensorBoard log directory (top-level)
TB_LOG_DIR = "lunarlander_dqn_duelling_logs"
os.makedirs(TB_LOG_DIR, exist_ok=True)

lunar_params = {
    'env_id': 'LunarLander-v3',
    'gamma': 0.99,
    'epsilon_start': 1.0,
    'epsilon_min': 0.01,
    'epsilon_decay': 0.95,
    'batch_size': 64,
    'target_update_freq': 1000,  # in training steps
    'replay_buffer_size': 100000,
    'lr': 0.0001,
    'optimizer': 'adam',
    'num_episodes': 1000,
    'max_steps': 1000,
    'reward_threshold': 200,
    'video_format': ['gif'],
    'save_interval_episodes': 10,
    'eval_episodes': 3
}

# Filenames used for checkpointing
MODEL_LATEST = 'dueling_model.h5'
TARGET_MODEL_LATEST = 'dueling_target_model.h5'
REPLAY_BUFFER = 'replay_buffer.pkl'
METADATA = 'training_metadata.json'
RESULTS_PNG = 'dueling_dqn_lunarlander_results.png'

# Optional pretrained CartPole weights for transfer learning (set to path or None)
PRETRAINED_CARTPOLE_PATH = 'models/dueling_dqn_cartpole/dueling_model.h5'  # e.g. './models/cartpole_dueling/best_model.h5'


# Filenames used for checkpointing
MODEL_LATEST = 'dueling_model.h5'
TARGET_MODEL_LATEST = 'dueling_target_model.h5'
REPLAY_BUFFER = 'replay_buffer.pkl'
METADATA = 'training_metadata.json'
RESULTS_PNG = 'dueling_dqn_cartpole_results.png'
TB_LOG_DIR = "logs/cartpole_dqn_dueling"
