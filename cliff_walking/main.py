import gymnasium as gym
import numpy as np
import pickle

def run(episodes, is_training=True, render=False):
    env = gym.make('CliffWalking-v0', render_mode='human' if render else None)

    q = np.zeros((env.observation_space.n, env.action_space.n)) # 48 x 4 array

    epsilon = 1
    epsilon_decay = 0.0001
    learning_rate = 0.9
    discount_factor = 0.9
    rng = np.random.default_rng()

    for i in range(episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False

        while not terminated and not truncated:
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state, :])
            
            new_state,reward,terminated,truncated,_ = env.step(action)
            if is_training:
                q[state, action] = q[state, action] + learning_rate * (reward + discount_factor * np.max(q[new_state, :]) - q[state, action])

            state = new_state

        epsilon = max(epsilon - epsilon_decay, 0)
        if epsilon == 0: learning_rate = 0.0001

    env.close()

    f = open("cliff_walking.pkl", "wb")
    pickle.dump(q, f)
    f.close()


if __name__ == '__main__':
    run(1000)
