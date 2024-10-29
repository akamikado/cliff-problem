from env import Model
from brain import QLearner, DoubleQLearner, TripleQLearner, QuadrupleQLearner
from plot import plot_cumulative_rewards, plot_rewards_per_episode, plot_steps_per_episode

import time
import argparse


def main(args):
    env = Model()
    learners = [QLearner(env, q_values=args.q_values, alpha=args.alpha), DoubleQLearner(env, q_values=args.dq_values, alpha=args.alpha), TripleQLearner(env, q_values=args.tq_values, alpha=args.alpha), QuadrupleQLearner(env, q_values=args.qq_values, alpha=args.alpha)]

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
    plot_rewards_per_episode([learner.algo_name for learner in learners], rewards)
    plot_steps_per_episode([learner.algo_name for learner in learners], steps)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description= "Grid Problem")

    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes the model should run for')
    parser.add_argument('--q-values', type=str, default=None, help='Number of Q values to use for QLearner')
    parser.add_argument('--dq-values', type=str, default=None, help='Number of Q values to use for DoubleQLearner')
    parser.add_argument('--tq-values', type=str, default=None, help='Number of Q values to use for TripleQLearner')
    parser.add_argument('--qq-values', type=str, default=None, help='Number of Q values to use for QuadrupleQLearner')
    parser.add_argument('--alpha', type=int, default=0.1, help='Whether to plot the results')

    args = parser.parse_args()

    main(args)
