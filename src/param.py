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
    'epsilon_decay': 0.995,                       #else try 0.990
    'target_update_freq': 1000,
    'soft_update_tau': 0.001,
    'optimizer': 'adam'
}
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
    'epsilon_decay': 0.995,               # can adjust to 0.999 if exploration is too short
    'target_update_freq': 1000,           # hard update every 1000 steps
    'soft_update_tau': 0.005,             # use if soft updates are enabled
    'optimizer': 'adam'                   # adaptive gradients, repo default
}

