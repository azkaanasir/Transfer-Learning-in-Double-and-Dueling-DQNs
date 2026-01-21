"""Configuration and hyperparameters."""
import os

BASE_SAVE_PATH = os.path.join('.', 'models', 'double_dqn_lunarlander')
os.makedirs(BASE_SAVE_PATH, exist_ok=True)

lunar_params = {
    'env_id': 'LunarLander-v3',
    'reward_threshold': 200,
    'num_episodes': 1000,
    'lr': 5e-4,
    'gamma': 0.99,
    'batch_size': 64,
    'replay_buffer_size': 100_000,
    'epsilon_start': 1.0,
    'epsilon_min': 0.01,
    'epsilon_decay': 0.95,
    'target_update_freq': 1000,
    'optimizer': 'adam'
}

MODEL_LATEST = 'ddqn_model.h5'
TARGET_MODEL_LATEST = 'ddqn_target_model.h5'
REPLAY_BUFFER = 'replay_buffer.pkl'
METADATA = 'training_metadata.json'
TB_LOG_DIR = "logs/lunarlander_ddqn"
