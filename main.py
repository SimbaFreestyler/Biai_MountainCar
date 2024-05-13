import json

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import codecs, json
from datetime import datetime


def sleep_microseconds(microseconds):
    start_time = time.time()
    while (time.time() - start_time) * 1e6 < microseconds:
        pass


date_variable1 = datetime(2002, 9, 20)
date_variable2 = datetime(1923, 12, 1)
# Random formula generator
#y\ =\ \left(\sin\left(x^{1.7232}\right)\ \cdot\ \sin\left(x^{2.488}\right)\ \cdot\ \sin\left(x^{0.824}\right)\right)^{2}
last_random_number = float(0)


def randomNumberGenerator():
    global last_random_number
    value1 = float(np.sin(np.power(datetime.now().microsecond, 1.7232)))
    value2 = float(np.sin(np.power(date_variable1.now().microsecond, 2.488)))
    value3 = float(np.sin(np.power(date_variable2.now().microsecond, 0.824)))
    value = round(float(np.power((value1 * value2 * value3), 2)), 3)
    if value != last_random_number:
        last_random_number = value
        return value
    else:
        sleep_microseconds(1)
        return randomNumberGenerator()


def run(episodes, is_training=True, render=False):
    env = gym.make('MountainCar-v0', render_mode='human' if render else None)

    # Divide position and velocity into segments
    pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 20)  # Between -1.2 and 0.6
    vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 20)  # Between -0.07 and 0.07

    if (is_training):
        q = np.zeros((len(pos_space), len(vel_space), env.action_space.n)) # init a 20x20x3 array
    else:

        json_file = open(f'mountain_car.json', 'r')
        q_list = json.load(json_file)
        q = np.array(q_list)
        json_file.close()

        #f = open('mountain_car.pkl', 'rb')
        #q = pickle.load(f)
        #f.close()

    crossing_factor = 0.9  # crossing factor
    mutation_factor = 0.9  # 1 / mutation factor.

    epsilon = 1  # exploration factor
    epsilon_decay_rate = 2 / episodes  # epsilon decay rate
    rewards_per_episode = np.zeros(episodes)
    #np.random.seed(1)
    rng = np.random.default_rng()
    for i in range(episodes):
        print(f'Episode {i + 1}/{episodes}')
        state = env.reset()[0]  # Starting position, starting velocity always 0
        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], vel_space)

        terminated = False  # True when reached goal

        rewards = 0

        while not terminated and rewards > -300:
            #randomValue = randomNumberGenerator()
            #randomValue = np.random.rand()
            randomValue = rng.random()
            #print(randomValue)
            if is_training and randomValue < epsilon:
                # Choose random action (0=drive left, 1=stay neutral, 2=drive right)
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state_p, state_v, :])

            new_state, reward, terminated, _, _ = env.step(action)
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vel_space)

            if is_training:
                q[state_p, state_v, action] = q[state_p, state_v, action] + crossing_factor * (
                        reward + mutation_factor * np.max(q[new_state_p, new_state_v, :]) - q[
                    state_p, state_v, action]
                )

            state = new_state
            state_p = new_state_p
            state_v = new_state_v

            rewards += reward

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        rewards_per_episode[i] = rewards

    env.close()

    # Save Q table to file
    if is_training:

        q_list = q.tolist()
        json_file = open(f'mountain_car.json', 'w')
        #json.dump(q_list, codecs.open(json_file, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
        json.dump(q_list, json_file)
        json_file.close()

        #f = open('mountain_car.pkl', 'wb')
        #pickle.dump(q, f)
        #f.close()

    mean_rewards = np.zeros(episodes)
    for t in range(episodes):
        mean_rewards[t] = np.mean(rewards_per_episode[max(0, t - 100):(t + 1)])
    plt.plot(mean_rewards)
    plt.xlabel('Episodes')
    plt.savefig(f'mountain_car.png')


def convert_pkl_to_txt(pkl_filename, txt_filename):
    try:
        # Open the .pkl file in read-binary mode ('rb')
        with open(pkl_filename, 'rb') as pkl_file:
            # Load the contents of the pickle file
            data = np.load(pkl_file, allow_pickle=True)

        # Open the .txt file in write mode ('w')
        with open(txt_filename, 'w') as txt_file:
            # Convert the data to a human-readable format and write it to the .txt file
            if isinstance(data, np.ndarray):
                if data.ndim == 1:
                    # If data has only one dimension, treat it as a single row
                    float_values = data[:2]
                    int_value = int(data[2])
                    txt_file.write(f"{float_values[0]:.8f} {float_values[1]:.8f} {int_value}\n")
                else:
                    # Iterate over each row if data has multiple rows
                    for row in data:
                        float_values = row[:2]
                        int_value = int(row[2])
                        txt_file.write(f"{float_values[0]:.8f} {float_values[1]:.8f} {int_value}\n")
            else:
                txt_file.write(str(data))

        print(f"Conversion successful. Data from '{pkl_filename}' has been written to '{txt_filename}'.")

    except Exception as e:
        print(f"Error occurred during conversion: {e}")


if __name__ == '__main__':
    #run(3000, is_training=True, render=False)
    #convert_pkl_to_txt('mountain_car.pkl', 'mountain_car.txt')
    run(10, is_training=False, render=True)
