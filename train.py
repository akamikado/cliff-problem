from env import Model
from brain import QLearner, DoubleQLearner, TripleQLearner, QuadrupleQLearner

import matplotlib.pyplot as plt
import numpy as np
import time
import argparse

def plot_cumulative_rewards(algorithm_names, rewards_lists):
    plt.figure(figsize=(10, 6))
    
    for name, rewards in zip(algorithm_names, rewards_lists):
        plt.plot(rewards, label=name)

    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Rewards')
    plt.title('Cumulative Rewards vs Episodes')
    plt.legend()
    plt.grid()
    plt.savefig("rewards_vs_episodes.png", format="png", dpi=300)

def plot_steps_per_episode(algorithm_names, steps_list):
    plt.figure(figsize=(10, 6))
    
    for name, steps in zip(algorithm_names, steps_list):
        plt.plot(steps, label=name)

    plt.xlabel('Episodes')
    plt.ylabel('Steps per Episode')
    plt.title('Steps per Episode vs Episodes')
    plt.legend()
    plt.grid()
    plt.savefig("steps_vs_episodes.png", format="png", dpi=300)

def main(args):
    env = Model()
    learners = [QLearner(env), DoubleQLearner(env), TripleQLearner(env), QuadrupleQLearner(env)]

    episodes = args.episodes

    rewards = []
    steps = []

    for learner in learners:
        print(f"Training {learner.algo_name}")
        start_time = time.time()
        learner_rewards, learner_steps, _ = learner.train(episodes)
        end_time = time.time()
        print(f"{learner.algo_name} took {end_time - start_time:.2f} seconds to train.")
        elapsed_time = end_time - start_time

        with open('results.txt', 'a') as f:
            f.write(f"{learner.algo_name} took {elapsed_time:.2f} seconds to train.\n")

        rewards.append(learner_rewards)
        steps.append(learner_steps)

    plot_cumulative_rewards([learner.algo_name for learner in learners], rewards)
    plot_steps_per_episode([learner.algo_name for learner in learners], steps)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description= "Grid Problem")

    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes the model should run for')

    args = parser.parse_args()

    main(args)
