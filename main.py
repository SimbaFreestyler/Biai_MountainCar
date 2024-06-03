import json

import gymnasium as gym
import numpy
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import codecs, json
import os
from datetime import datetime


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


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


def qRun(episodes, is_training=True, render=False):
    env = gym.make('MountainCar-v0', render_mode='human' if render else None)

    # Divide position and velocity into segments
    pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 20)  # Between -1.2 and 0.6
    vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 20)  # Between -0.07 and 0.07

    if (is_training):
        q = np.zeros((len(pos_space), len(vel_space), env.action_space.n))  # init a 20x20x3 array
    else:

        json_file = open(f'mountain_car.json', 'r')
        loaded = json.load(json_file)
        loadeded = json.loads(loaded)
        q = np.asarray(loadeded)
        json_file.close()

    learning_factor = 0.9
    mutation_factor = 0.9

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

        while not terminated and rewards > -1000:
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
                q[state_p, state_v, action] = q[state_p, state_v, action] + learning_factor * (
                        reward + mutation_factor * np.max(q[new_state_p, new_state_v, :]) - q[
                    state_p, state_v, action]
                )

            state = new_state
            state_p = new_state_p
            state_v = new_state_v

            rewards += reward

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        rewards_per_episode[i] = rewards
        if not is_training:
            print(f'Cycles: ' + (rewards_per_episode[i] * -1).__str__())
    env.close()

    # Save Q table to file
    if is_training:
        q_list = q.tolist()
        json_file = open(f'mountain_car.json', 'w')

        dumped = json.dumps(q, cls=NumpyEncoder)
        json.dump(dumped, json_file)
        json_file.close()

    mean_rewards = np.zeros(episodes)
    for t in range(episodes):
        mean_rewards[t] = np.mean(rewards_per_episode[max(0, t - 100):(t + 1)])
    plt.plot(mean_rewards)
    plt.xlabel('Episodes')
    plt.savefig(f'mountain_car.png')


def create_buffer(buffer_length):
    """Creates a random policy."""
    return np.random.choice([0, 1, 2], size=buffer_length)


def evaluate_buffer(policy, env):
    """Evaluates a policy by running it in the environment."""
    state = env.reset()[0]
    total_reward = 0
    for action in policy:
        state, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward


def select_parents(population, fitnesses):
    """Selects two parents using a fitness-proportionate selection."""
    total_fitness = np.sum(fitnesses)
    selection_probs = fitnesses / total_fitness
    parent_indices = np.random.choice(np.arange(len(population)), size=2, p=selection_probs)
    return population[parent_indices[0]], population[parent_indices[1]]


def crossover(parent1, parent2, crossover_rate, buffer_length):
    """Performs a single-point crossover."""
    if np.random.rand() < crossover_rate:
        crossover_point = np.random.randint(0, buffer_length)
        child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
    else:
        child1, child2 = parent1.copy(), parent2.copy()
    return child1, child2


def mutate(policy, policy_length, mutation_rate):
    """Mutates a policy by randomly changing some actions."""
    for i in range(policy_length):
        if np.random.rand() < mutation_rate:
            policy[i] = np.random.choice([0, 1, 2])
    return policy


def save_best_buffer(policy, best_policy_file):
    """Saves the best policy to a JSON file."""
    with open(best_policy_file, 'w') as file:
        json.dump(policy.tolist(), file)


def load_best_buffer(best_policy_file):
    """Loads the best policy from a JSON file."""
    if os.path.exists(best_policy_file):
        with open(best_policy_file, 'r') as file:
            policy = np.array(json.load(file))
            return policy
    else:
        return None


def genetic_algorithm(population_size, generations, mutation_rate, crossover_rate, buffer_length, best_buffer_file,
                      env):
    # Initialize the population
    population = np.array([create_buffer(buffer_length) for _ in range(population_size)])
    best_buffer = load_best_buffer(best_buffer_file)
    best_score = evaluate_buffer(best_buffer, env) if best_buffer is not None else -np.inf

    for generation in range(generations):
        # Evaluate fitness of each individual
        fitnesses = np.array([evaluate_buffer(policy, env) for policy in population])

        # Select the best policy
        max_fitness_idx = np.argmax(fitnesses)
        if fitnesses[max_fitness_idx] > best_score:
            best_score = fitnesses[max_fitness_idx]
            best_buffer = population[max_fitness_idx].copy()

        print(f'Generation {generation}: Best Score = {best_score}')

        # Selection
        new_population = []
        for _ in range(population_size // 2):
            parent1, parent2 = select_parents(population, fitnesses)
            child1, child2 = crossover(parent1, parent2, crossover_rate, buffer_length)
            new_population.append(mutate(child1, buffer_length, mutation_rate))
            new_population.append(mutate(child2, buffer_length, mutation_rate))
        population = np.array(new_population)

    return best_buffer


def run(population_size, generations, mutation_rate, crossover_rate, policy_length, best_policy_file, render,
        is_training):
    if is_training:
        env = gym.make('MountainCar-v0')
        best_policy = genetic_algorithm(population_size, generations, mutation_rate, crossover_rate, policy_length,
                                        best_policy_file, env)
        save_best_buffer(best_policy, best_policy_file)
        env.close()
        return
    best_policy = load_best_buffer(best_policy_file)
    if best_policy is None:
        print('No best policy found.')
        return
    # Test the best policy
    env = gym.make('MountainCar-v0', render_mode='human' if render else None)
    total_reward = evaluate_buffer(best_policy, env) if best_policy is not None else -np.inf
    env.close()
    print(f'Total reward of the best policy: {total_reward}')


if __name__ == '__main__':
    run(population_size=50, generations=300, mutation_rate=0.1, crossover_rate=0.5, policy_length=1000,
        best_policy_file='best_policy.json', render=True, is_training=True)

    #qRun(3000, is_training=True, render=False)
    #qRun(5, is_training=False, render=True)
