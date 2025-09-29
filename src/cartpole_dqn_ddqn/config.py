"""Configuration and constants for the Dueling DQN project."""
import os

# Path where models, checkpoints and plots are stored
BASE_SAVE_PATH = os.path.join('.', 'models', 'dueling_dqn_cartpole')
os.makedirs(BASE_SAVE_PATH, exist_ok=True)

# # Default hyperparameters (can be overridden via CLI)
# DEFAULTS = {
#     'gamma': 0.99,
#     'epsilon': 1.0,
#     'epsilon_min': 0.01,
#     'epsilon_decay': 0.995,
#     'buffer_size': 10_000,
#     'batch_size': 64,
#     'target_update_freq': 10,
#     'learning_rate': 1e-3,
# }

cartpole_params = {
  
    'env_id': 'CartPole-v1',
    'reward_threshold': 195,              # solved when avg ≥195 over 100 consecutive episodes
    'num_episodes': 1000,
    'video_format': ['mp4', 'gif'],       # mp4 might not work, fallback to gif
    'lr': 0.0001,                         # lower LR for stable updates (repo default)
    'gamma': 0.99,                        # standard discount for CartPole
    'batch_size': 64,                     # smaller batch than repo’s 256, good for CartPole
    'replay_buffer_size': 100000,         # enough for diverse transitions
    'epsilon_start': 1.0,                 # full exploration initially
    'epsilon_min': 0.01,                  # maintain some exploration
    'epsilon_decay': 0.95,               # can adjust to 0.999 if exploration is too short
    'target_update_freq': 1000,           # hard update every 1000 steps
    'soft_update_tau': 0.005,             # use if soft updates are enabled
    'optimizer': 'adam'                   # adaptive gradients, repo default
}

# Filenames used for checkpointing
MODEL_LATEST = 'dueling_model.h5'
TARGET_MODEL_LATEST = 'dueling_target_model.h5'
REPLAY_BUFFER = 'replay_buffer.pkl'
METADATA = 'training_metadata.json'
RESULTS_PNG = 'dueling_dqn_cartpole_results.png'
TB_LOG_DIR = "logs/cartpole_dqn_dueling"
