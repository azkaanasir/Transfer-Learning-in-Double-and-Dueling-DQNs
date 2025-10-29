"""Training loop that coordinates environment, agent, persistence, evaluation and TensorBoard logging."""
import os
import datetime
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import tensorflow as tf

from config import lunar_params, BASE_SAVE_PATH, MODEL_LATEST, METADATA, TB_LOG_DIR, RESULTS_PNG, PRETRAINED_CARTPOLE_PATH
from agent import DuelingDQNAgent
from utils import load_training_state, save_training_state, save_gif, evaluate, set_gpu_growth


def train_agent(pretrained_path: str = None, freeze_base: bool = True):
    set_gpu_growth()
    env = gym.make(lunar_params['env_id'])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DuelingDQNAgent(state_dim, action_dim, lunar_params, pretrained_path, freeze_base)

    # resume state (from metadata file)
    all_rewards, eval_scores, start_ep, loaded_eps = load_training_state(os.path.join(BASE_SAVE_PATH, METADATA))
    if loaded_eps is not None:
        agent.epsilon = loaded_eps

    # try checkpoint from agent-level checkpoint
    ckpt_loaded, ckpt_ep, ep_rewards, ep_lengths, val_rewards = agent.load_checkpoint()
    if ckpt_loaded:
        print(f"Loaded checkpoint from episode {ckpt_ep-1}; resuming at {ckpt_ep}")
        start_ep = max(start_ep, ckpt_ep)
        # merge loaded histories if present
        if ep_rewards:
            all_rewards = ep_rewards
        if val_rewards:
            eval_scores = val_rewards

    best_score = max(eval_scores) if eval_scores else -np.inf

    num_episodes = lunar_params['num_episodes']
    max_steps = lunar_params.get('max_steps', 500)

    # TensorBoard setup
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    logdir = os.path.join(TB_LOG_DIR, timestamp)
    os.makedirs(logdir, exist_ok=True)
    writer = tf.summary.create_file_writer(logdir)
    global_train_step = 0

    for ep in range(start_ep, num_episodes):
        state, _ = env.reset()
        total_reward = 0.0

        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc

            agent.replay_buffer.add(state, action, reward, next_state, done)
            loss = agent.train_step()

            # Log training loss per training step to TensorBoard
            if loss and loss != 0.0:
                with writer.as_default():
                    tf.summary.scalar('train/loss', loss, step=global_train_step)
                global_train_step += 1

            state = next_state
            total_reward += float(reward)

            if done:
                break

        all_rewards.append(total_reward)

        # Log episode-level metrics
        with writer.as_default():
            tf.summary.scalar('episode/reward', total_reward, step=ep)
            tf.summary.scalar('episode/length', step + 1, step=ep)
            tf.summary.scalar('agent/epsilon', agent.epsilon, step=ep)

        # Validation & logging every save interval
        if ep % lunar_params['save_interval_episodes'] == 0:
            original_epsilon = agent.epsilon
            agent.epsilon = 0.0
            val_score = evaluate(agent, env, episodes=lunar_params.get('eval_episodes', 3))
            agent.epsilon = original_epsilon

            eval_scores.append(val_score)
            print(f"Episode {ep}/{num_episodes}, Reward: {total_reward:.2f}, Validation: {val_score:.2f}, Epsilon: {agent.epsilon:.3f}")

            # Log validation metric
            with writer.as_default():
                tf.summary.scalar('validation/avg_reward', val_score, step=ep)

            if val_score > best_score:
                best_score = val_score
                agent.save_checkpoint(ep, all_rewards, [], eval_scores)
                print("New best model saved.")

            save_training_state(os.path.join(BASE_SAVE_PATH, METADATA), all_rewards, eval_scores, ep + 1, agent.epsilon)

        else:
            print(f"Episode {ep} | Train: {total_reward:.2f} | Eps: {agent.epsilon:.3f}")

        # save GIF every 100 episodes
        if ep % 100 == 0 and ep > 0 and 'gif' in lunar_params.get('video_format', []):
            save_gif(agent, os.path.join(BASE_SAVE_PATH, f'episode_{ep}.gif'))

        # early stop when solved (mean of last 100)
        if len(all_rewards) >= 100 and np.mean(all_rewards[-100:]) >= lunar_params['reward_threshold']:
            print(f"Environment solved at episode {ep}!")
            if 'gif' in lunar_params.get('video_format', []):
                save_gif(agent, os.path.join(BASE_SAVE_PATH, f'solved_at_{ep}.gif'))
            break

        # Decay epsilon ONCE per episode (safer)
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

        # periodic checkpoint save (end of block to include last step)
        if ep % lunar_params['save_interval_episodes'] == lunar_params['save_interval_episodes'] - 1:
            agent.save_checkpoint(ep, all_rewards, [], eval_scores)

    # final saves
    agent.save_checkpoint(ep, all_rewards, [], eval_scores)
    agent.save(os.path.join(BASE_SAVE_PATH, 'final_model.h5'))
    if 'gif' in lunar_params.get('video_format', []):
        save_gif(agent, os.path.join(BASE_SAVE_PATH, 'final.gif'))

    writer.flush()
    writer.close()

    # Save plot (best-effort)
    # try:
    #     plt.plot(all_rewards, label='Train Rewards')
    #     if eval_scores:
    #         plt.plot(np.arange(0, len(eval_scores)) * lunar_params['save_interval_episodes'], eval_scores, label='Eval Scores')
    #     plt.axhline(y=lunar_params['reward_threshold'], linestyle='--', label='Solved Threshold')
    #     plt.legend()
    #     plt.title('Dueling DQN — Training Progress')
    #     plt.xlabel('Episode'); plt.ylabel('Reward'); plt.grid(True)
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(BASE_SAVE_PATH, RESULTS_PNG))
    # except Exception as e:
    #     print("Failed to save training plot:", e)

    return all_rewards, None, eval_scores, agent.loss_history
