import gymnasium as gym
import pygame


def run():
    print("Hello World")
    env = gym.make('MountainCar-v0', render_mode='human')
    state = env.reset()[0]
    terminated = False

    rewards = 0

#0: Accelerate to the left 1: Don’t accelerate 2: Accelerate to the right

    while (not terminated and rewards > -1000):
        action = env.action_space.sample()
        next_state, reward, terminated, _, _ = env.step(action)

        rewards += reward
        state = next_state

    env.close()


if __name__ == '__main__':
    run()
