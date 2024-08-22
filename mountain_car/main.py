#!/home/mzums/miniconda3/envs/ml/bin
import pickle
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym


def run(episodes, is_training=True, render=False):
    env = gym.make("MountainCar-v0", render_mode='human' if render else None)

    pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0])
    vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1])
    q = np.zeros((len(pos_space), len(vel_space), env.action_space.n))

    epsilon = 1
    epsilon_deacy = 0.0001
    learning_rate = 0.9
    discount_factor = 0.9
    rng = np.random.default_rng()
    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0]
        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], vel_space)
        terminated = False
        truncated = False
        rewards = 0

        while not terminated and not truncated:
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state, :])

            new_state, reward, terminated, truncated, _ = env.step(action)
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vel_space)
            rewards += reward



if __name__ == '__main__':
    run(1000)