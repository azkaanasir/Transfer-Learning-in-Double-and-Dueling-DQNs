"""Configuration for CartPole -> LunarLander transfer (keeps same names & layout)."""
import os

# Where to save transfer models, checkpoints, etc.
BASE_SAVE_PATH = os.path.join('.', 'models', 'dueling_transfer_cartpole_to_lunarlander')
os.makedirs(BASE_SAVE_PATH, exist_ok=True)

# Filenames used for checkpointing (same names as original)
MODEL_LATEST = 'dueling_model.h5'
TARGET_MODEL_LATEST = 'dueling_target_model.h5'
REPLAY_BUFFER = 'replay_buffer.pkl'
METADATA = 'training_metadata.json'
RESULTS_PNG = 'dueling_lunarlander_results.png'
TB_LOG_DIR = "logs/lunarlander_dqn_dueling"

# Source (CartPole) model location: assumes you used original BASE_SAVE_PATH
SOURCE_BASE = os.path.join('.', 'models', 'dueling_dqn_cartpole')
SOURCE_MODEL = os.path.join(SOURCE_BASE, MODEL_LATEST)
SOURCE_TARGET_MODEL = os.path.join(SOURCE_BASE, TARGET_MODEL_LATEST)

# Keep the same hyper-parameter names and values to avoid "parameter" changes.
# These mirror cartpole_params names/values (no changes).
transfer_params = {
    'env_id': 'LunarLander-v3',
    'reward_threshold': 200,    # (kept as a nominal value; names unchanged)
    'num_episodes': 1000,
    'video_format': ['mp4', 'gif'],
    'lr': 0.0001,
    'gamma': 0.99,
    'batch_size': 64,
    'replay_buffer_size': 100000,
    'epsilon_start': 1.0,
    'epsilon_min': 0.01,
    'epsilon_decay': 0.95,
    'target_update_freq': 1000,
    'soft_update_tau': 0.005,
    'optimizer': 'adam'
}

# Transfer-specific, modular options (can be changed later programmatically)
TRANSFER_OPTIONS = {
    # If True, copy/transfer weights for available overlapping layers from source.
    'do_weight_transfer': True,

    # Unfreeze (make trainable) the first FC layer (default True per your request).
    'unfreeze_first_fc': True,

    # Unfreeze the last N FC layers (default 2 per your request).
    'unfreeze_last_n_fc': 2,

    # If True, attempt partial-copy for first Dense layer when input dims differ
    # (cartpole input_dim=4 -> lunarlander input_dim=8). Partial copying fills the
    # overlapping portion and initializes the rest with GlorotUniform.
    'partial_copy_first_dense_on_shape_mismatch': True,
}
