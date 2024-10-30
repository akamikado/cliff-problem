from env import Model
from brain import QLearner, DoubleQLearner, TripleQLearner, QuadrupleQLearner
from plot import plot_cumulative_rewards, plot_rewards_per_episode, plot_steps_per_episode

import time
import argparse
from datetime import datetime
import os
from concurrent.futures import ProcessPoolExecutor

def train_learner(learner, episodes, save_folder):
    print(f"Training {learner.algo_name}")
    start_time = time.time()
    learner_rewards, learner_steps, _ = learner.train(episodes)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"{learner.algo_name} took {elapsed_time:.2f} seconds to train.")

    with open(f'{save_folder}/results.txt', 'a') as f:
        f.write(f"{learner.algo_name} took {elapsed_time:.2f} seconds to train.\n")

    return learner.algo_name, learner_rewards, learner_steps

def main(args):
    save_folder = f"results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(save_folder)
    env = Model()

    learners = [
        QLearner(env, q_values=args.q_values, alpha=args.alpha, epsilon=args.epsilon, save_folder=save_folder),
        DoubleQLearner(env, q_values=args.dq_values, alpha=args.alpha, epsilon=args.epsilon, save_folder=save_folder),
        TripleQLearner(env, q_values=args.tq_values, alpha=args.alpha, epsilon=args.epsilon, save_folder=save_folder),
        QuadrupleQLearner(env, q_values=args.qq_values, alpha=args.alpha, epsilon=args.epsilon, save_folder=save_folder)
    ]

    episodes = args.episodes

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(train_learner, learner, episodes, save_folder) for learner in learners]

        results = []
        for future in futures:
            algo_name, learner_rewards, learner_steps = future.result()
            results.append((algo_name, learner_rewards, learner_steps))

    algo_names = [result[0] for result in results]
    rewards = [result[1] for result in results]
    steps = [result[2] for result in results]

    plot_cumulative_rewards(algo_names, rewards, save_folder=save_folder)
    plot_rewards_per_episode(algo_names, rewards, window_size=1000, save_folder=save_folder)
    plot_steps_per_episode(algo_names, steps, window_size=1000, save_folder=save_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grid Problem")

    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes the model should run for')
    parser.add_argument('--q-values', type=str, default=None, help='Number of Q values to use for QLearner')
    parser.add_argument('--dq-values', type=str, default=None, help='Number of Q values to use for DoubleQLearner')
    parser.add_argument('--tq-values', type=str, default=None, help='Number of Q values to use for TripleQLearner')
    parser.add_argument('--qq-values', type=str, default=None, help='Number of Q values to use for QuadrupleQLearner')
    parser.add_argument('--alpha', type=float, default=0.1, help='Alpha value')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Epsilon value')

    args = parser.parse_args()

    main(args)
