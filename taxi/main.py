import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import pickle

def run(episodes, is_training=True, render=False):
    env = gym.make("Taxi.v3", render_mode='human' if render else None)

    if is_training:
        q = np.zeros(env.observation_space.n, env.action_space.n)
    else:
        f = open('taxi.pkl', 'rb')
        q = pickle.load(f)
        f.close()

    epsilon = 1
    learning_rate = 0.9
    discount_factor = 0.9
    decay = 0.001
    rng = np.random.default_rng()
    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False
        rewards = 0

        while not terminated and not truncated:
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = max(q[state, :])

            new_state, reward, terminated, truncated, _ = env.step(action)
            rewards += reward

            if is_training:
                q[state, action] = q[state, action] + learning_rate * (
                    reward + discount_factor * np.max(q[new_state, :]) - q[state, action]
                )
            