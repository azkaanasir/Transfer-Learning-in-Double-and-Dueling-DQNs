import time, datetime
import numpy as np
import tensorflow as tf
from config import TB_LOG_DIR


def validate_agent(env, agent, episodes=5):
    rewards = []
    for _ in range(episodes):
        s,_ = env.reset()
        done = False
        total = 0
        while not done:
            a = agent.select_action(s)
            s,r,t,tr,_ = env.step(a)
            done = t or tr
            total += r
        rewards.append(total)
    return float(np.mean(rewards))


def train_agent(env, agent, num_episodes, max_steps, early_stop=True):
    rewards, lengths, validations = [], [], []

    logdir = f"{TB_LOG_DIR}/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    writer = tf.summary.create_file_writer(logdir)

    global_step = 0
    solved = 0

    for ep in range(num_episodes):
        s,_ = env.reset()
        total = 0

        for step in range(max_steps):
            a = agent.select_action(s)
            ns,r,t,tr,_ = env.step(a)
            done = t or tr

            agent.replay_buffer.add(s,a,r,ns,done)
            loss = agent.train()

            if loss:
                with writer.as_default():
                    tf.summary.scalar("train/loss", loss, global_step)
                global_step += 1

            s = ns
            total += r
            if done:
                break

        rewards.append(total)
        lengths.append(step+1)

        with writer.as_default():
            tf.summary.scalar("episode/reward", total, ep)
            tf.summary.scalar("agent/epsilon", agent.epsilon, ep)

        if ep % 10 == 0:
            agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
            val = validate_agent(env, agent)
            validations.append(val)

            with writer.as_default():
                tf.summary.scalar("validation/avg_reward", val, ep)

            print(f"Ep {ep} | Reward {total:.1f} | Val {val:.1f} | ε {agent.epsilon:.3f}")

            if val >= 195:
                solved += 1
                if solved >= 5:
                    print("Environment solved.")
                    break
            else:
                solved = 0

    writer.close()
    return rewards, lengths, validations, agent.loss_history
