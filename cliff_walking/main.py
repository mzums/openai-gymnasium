import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def run(episodes, is_training=True, render=False):
    env = gym.make('CliffWalking-v0', render_mode='human' if render else None)

    if(is_training):
        q = np.zeros((env.observation_space.n, env.action_space.n)) # 64 x 4 array
    else:
        f = open('cliff_walking.pkl', 'rb')
        q = pickle.load(f)
        f.close()

    epsilon = 1
    epsilon_decay = 0.0001
    learning_rate = 0.9
    discount_factor = 0.9
    rng = np.random.default_rng()
    rewards_per_episode = np.zeros(episodes)

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
            rewards_per_episode[i] += reward
            if is_training:
                q[state, action] = q[state, action] + learning_rate * (reward + discount_factor * np.max(q[new_state, :]) - q[state, action])

            state = new_state

        epsilon = max(epsilon - epsilon_decay, 0)
        if epsilon == 0: learning_rate = 0.0001

    env.close()

    sum_rewards = np.zeros(episodes)
    for ep in range(episodes):
        sum_rewards[ep] = np.mean(rewards_per_episode[max(0, ep-100):(ep+1)])

    sum_rewards = sum_rewards[10:]
    plt.plot(sum_rewards)
    plt.savefig('cliff_walking.png')

    if is_training:
        f = open("cliff_walking.pkl", "wb")
        pickle.dump(q, f)
        f.close()


if __name__ == '__main__':
    #run(1000)
    run(10, is_training=False, render=True)