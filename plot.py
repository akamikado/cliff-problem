import matplotlib.pyplot as plt
import numpy as np

def plot_cumulative_rewards(algorithm_names, rewards_lists):
    plt.figure(figsize=(10, 6))
    
    for name, rewards in zip(algorithm_names, rewards_lists):
        cum_rewards = np.cumsum(rewards)
        plt.plot(cum_rewards, label=name)

    plt.xlabel('Episodes')
    plt.ylabel('Cumulative rewards')
    plt.title('Cumulative rewards vs Episodes')
    plt.legend()
    plt.grid()
    plt.savefig("cumulative_rewards_vs_episodes.png", format="png", dpi=450)

def plot_rewards_per_episode(algorithm_names, rewards_list, window_size=500):
    plt.figure(figsize=(10, 6))
    
    for name, rewards in zip(algorithm_names, rewards_list):
        smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(smoothed_rewards, label=name)

    plt.xlabel('Episodes')
    plt.ylabel('Smoothed Rewards per Episode')
    plt.title(f'Sum of Rewards per Episode (Smoothed over {window_size} episodes)')
    plt.legend()
    plt.grid()
    plt.savefig("smoothed_rewards_per_episode.png", format="png", dpi=450)

def plot_steps_per_episode(algorithm_names, steps_list, window_size=500):
    plt.figure(figsize=(10, 6))
    
    for name, steps in zip(algorithm_names, steps_list):
        smoothed_steps = np.convolve(steps, np.ones(window_size)/window_size, mode='valid')
        plt.plot(smoothed_steps, label=name)

    plt.xlabel('Episodes')
    plt.ylabel('Steps per Episode')
    plt.title('Steps per Episode vs Episodes')
    plt.legend()
    plt.grid()
    plt.savefig("steps_vs_episodes.png", format="png", dpi=300)

