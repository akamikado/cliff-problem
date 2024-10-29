import matplotlib.pyplot as plt
import numpy as np
import csv

def plot_cumulative_rewards(algorithm_names, rewards_lists, dpi=300, save=True):
    plt.figure(figsize=(10, 6))
    
    for name, rewards in zip(algorithm_names, rewards_lists):
        cum_rewards = np.cumsum(rewards)
        plt.plot(cum_rewards, label=name)

    plt.xlabel('Episodes')
    plt.ylabel('Cumulative rewards')
    plt.title('Cumulative rewards vs Episodes')
    plt.legend()
    plt.grid()
    if save:
        plt.savefig("cumulative_rewards_vs_episodes.png", format="png", dpi=dpi)
    else:
        plt.show()

def plot_rewards_per_episode(algorithm_names, rewards_list, window_size=500, dpi=300, save=True):
    plt.figure(figsize=(10, 6))
    
    for name, rewards in zip(algorithm_names, rewards_list):
        smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(smoothed_rewards, label=name)

    plt.xlabel('Episodes')
    plt.ylabel('Sum of Rewards per Episode')
    plt.title(f'Sum of Rewards per Episode vs Episodes')
    plt.legend()
    plt.grid()
    if save:
        plt.savefig("rewards_per_episode_vs_epsiodes.png", format="png", dpi=dpi)
    else:
        plt.show()

def plot_steps_per_episode(algorithm_names, steps_list, window_size=500, dpi=300, save=True):
    plt.figure(figsize=(10, 6))
    
    for name, steps in zip(algorithm_names, steps_list):
        smoothed_steps = np.convolve(steps, np.ones(window_size)/window_size, mode='valid')
        plt.plot(smoothed_steps, label=name)

    plt.xlabel('Episodes')
    plt.ylabel('Steps per Episode')
    plt.title('Steps per Episode vs Episodes')
    plt.legend()
    plt.grid()
    if save:
        plt.savefig("steps_vs_episodes.png", format="png", dpi=dpi)
    else:
        plt.show()


if __name__ == "__main__":
    dpi = int(input("Enter the DPI for the plots: "))
    window_size = int(input("Enter the window size for smoothing: "))
    save = input("Do you want to save the plots? (y/n): ").lower() == 'y'
    algorithm_names = ['Q-Learning', 'Double Q-Learning', 'Triple Q-Learning', 'Quadruple Q-Learning']
    rewards_lists = []
    steps_lists = []
    file_names = ['q-learning_', 'double_q-learning_', 'triple_q-learning_', 'quadruple_q-learning_']
    for file in file_names:
        with open(file + 'rewards.csv', 'r') as f:
            csvreader = csv.reader(f)
            rewards = []
            for row in csvreader:
                rewards.append(float(row[0]))
            rewards_lists.append(rewards)
        with open(file + 'steps.csv', 'r') as f:
            csvreader = csv.reader(f)
            steps = []
            for row in csvreader:
                steps.append(float(row[0]))
            steps_lists.append(steps)

    plot_cumulative_rewards(algorithm_names, rewards_lists, dpi=dpi, save=save)
    plot_rewards_per_episode(algorithm_names, rewards_lists, window_size=window_size, dpi=dpi, save=save)
    plot_steps_per_episode(algorithm_names, steps_lists, window_size=window_size, dpi=dpi, save=save)
