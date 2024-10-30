import numpy as np
import random

class QLearner:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1, epsilon_decay=0.99, q_values=None, save_folder=None):
        self.algo_name = "Q-Learning"
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.save_folder = save_folder
        if q_values is None:
            self.q1 = np.zeros((env.cols, env.rows, len(env.get_actions())))
        else:
            self.q1 = np.load(q_values)['Q1']

    def epsilon_greedy(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(range(len(self.env.get_actions())))
        else:
            q = self.q1[state]
            return np.argmax(q)

    def update_q_value(self, state, next_state, action, reward):
        self.q1[state][action] += self.alpha * (reward + self.gamma * np.max(self.q1[next_state]) - self.q1[state][action])

    def save_episode_rewards(self, rewards):
        if self.save_folder:
            with open(f'{self.save_folder}/{self.algo_name.replace(' ', '_').lower()}_rewards.csv', 'a') as f:
                f.write(f"{rewards}\n")
        else:
            with open(f'{self.algo_name.replace(' ', '_').lower()}_rewards.csv', 'a') as f:
                f.write(f"{rewards}\n")

    def save_episode_steps(self, steps):
        if self.save_folder:
            with open(f'{self.save_folder}/{self.algo_name.replace(' ', '_').lower()}_steps.csv', 'a') as f:
                f.write(f"{steps}\n")
        else:
            with open(f'{self.algo_name.replace(' ', '_').lower()}_steps.csv', 'a') as f:
                f.write(f"{steps}\n")

    def decay_epsilon(self):
        new_epsilon = self.epsilon * self.epsilon_decay
        if new_epsilon < 0.05:
            new_epsilon = 0.05
        self.epsilon = new_epsilon

    def train(self, episodes):
        save_file_name = "q_values_" + self.algo_name.replace(' ', '_').lower()
        rewards = []
        steps = []

        for i in range(episodes):
            state = self.env.reset()
            total_reward = 0
            steps_taken = 0
            done = False
            while not done:
                action_idx = self.epsilon_greedy(state)
                action = self.env.get_actions()[action_idx]
                next_state, reward, done = self.env.step(action)
                self.update_q_value(state, next_state, action_idx, reward)
                state = next_state
                total_reward += reward
                steps_taken += 1

            rewards.append(total_reward)
            steps.append(steps_taken)

            self.decay_epsilon()

            if i % 100 == 0:
                print(f"Episode {i} completed.")

            self.save_episode_rewards(total_reward)
            self.save_episode_steps(steps_taken)

            if i % 1000 == 0:
                np.savez(save_file_name, Q1=self.q1)

        return rewards, steps, [self.q1]

class DoubleQLearner(QLearner):
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1, epsilon_decay=0.99, q_values=None, save_folder=None):
        super().__init__(env, alpha, gamma, epsilon, epsilon_decay, q_values, save_folder)
        self.algo_name = "Double Q-Learning"
        if q_values is None:
            self.q2 = np.zeros((env.cols, env.rows, len(env.get_actions())))
        else:
            self.q2 = np.load(q_values)['Q2']

    def epsilon_greedy(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(range(len(self.env.get_actions())))
        else:
            q = self.q1[state] + self.q2[state]
            return np.argmax(q)

    def update_q_value(self, state, next_state, action, reward):
        if np.random.rand() < 0.5:
            self.q1[state][action] += self.alpha * (reward + self.gamma * self.q2[next_state][np.argmax(self.q1[next_state])] - self.q1[state][action])
        else:
            self.q2[state][action] += self.alpha * (reward + self.gamma * self.q1[next_state][np.argmax(self.q2[next_state])] - self.q2[state][action])

    def train(self, episodes):
        save_file_name = "q_values_" + self.algo_name.replace(' ', '_').lower()
        rewards = []
        steps = []

        for i in range(episodes):
            state = self.env.reset()
            total_reward = 0
            steps_taken = 0
            done = False
            while not done:
                action_idx = self.epsilon_greedy(state)
                action = self.env.get_actions()[action_idx]
                next_state, reward, done = self.env.step(action)
                self.update_q_value(state, next_state, action_idx, reward)
                state = next_state
                total_reward += reward
                steps_taken += 1

            rewards.append(total_reward)
            steps.append(steps_taken)

            self.decay_epsilon()
            
            if i % 100 == 0:
                print(f"Episode {i} completed.")

            self.save_episode_rewards(total_reward)
            self.save_episode_steps(steps_taken)

            if i % 1000 == 0:
                np.savez(save_file_name, Q1=self.q1, Q2=self.q2)

        return rewards, steps, [self.q1, self.q2]

class TripleQLearner(DoubleQLearner):
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1, epsilon_decay=0.99, q_values=None, save_folder=None):
        super().__init__(env, alpha, gamma, epsilon, epsilon_decay, q_values, save_folder)
        self.algo_name = "Triple Q-Learning"
        if q_values is None:
            self.q3 = np.zeros((env.cols, env.rows, len(env.get_actions())))
        else:
            self.q3 = np.load(q_values)['Q3']

    def epsilon_greedy(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(range(len(self.env.get_actions())))
        else:
            q = self.q1[state] + self.q2[state] + self.q3[state]
            return np.argmax(q)

    def update_q_value(self, state, next_state, action, reward):
        choice = random.choice(range(3))

        if choice == 0:
            best_action = np.argmax((self.q2[next_state] + self.q3[next_state]) / 2)
            self.q1[state][action] += self.alpha * (reward + self.gamma * min(self.q2[next_state][best_action], self.q3[next_state][best_action]) - self.q1[state][action])
        elif choice == 1:
            best_action = np.argmax((self.q1[next_state] + self.q3[next_state]) / 2)
            self.q2[state][action] += self.alpha * (reward + self.gamma * min(self.q1[next_state][best_action], self.q3[next_state][best_action]) - self.q2[state][action])
        else:
            best_action = np.argmax((self.q2[next_state] + self.q1[next_state]) / 2)
            self.q3[state][action] += self.alpha * (reward + self.gamma * self.q1[next_state][np.argmax(self.q3[next_state])] - self.q3[state][action])
            self.q3[state][action] += self.alpha * (reward + self.gamma * min(self.q1[next_state][best_action], self.q2[next_state][best_action]) - self.q3[state][action])

    def train(self, episodes):
        save_file_name = "q_values_" + self.algo_name.replace(' ', '_').lower()
        rewards = []
        steps = []

        for i in range(episodes):
            state = self.env.reset()
            total_reward = 0
            steps_taken = 0
            done = False
            while not done:
                action_idx = self.epsilon_greedy(state)
                action = self.env.get_actions()[action_idx]
                next_state, reward, done = self.env.step(action)
                self.update_q_value(state, next_state, action_idx, reward)
                state = next_state
                total_reward += reward
                steps_taken += 1

            rewards.append(total_reward)
            steps.append(steps_taken)

            self.decay_epsilon()

            if i % 100 == 0:
                print(f"Episode {i} completed.")

            self.save_episode_rewards(total_reward)
            self.save_episode_steps(steps_taken)

            if i % 1000 == 0:
                np.savez(save_file_name, Q1=self.q1, Q2=self.q2, Q3=self.q3)

        return rewards, steps, [self.q1, self.q2, self.q3]

class QuadrupleQLearner(TripleQLearner):
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1, epsilon_decay=0.99, q_values=None, save_folder=None):
        super().__init__(env, alpha, gamma, epsilon, epsilon_decay, q_values, save_folder)
        self.algo_name = "Quadruple Q-Learning"
        if q_values is None:
            self.q4 = np.zeros((env.cols, env.rows, len(env.get_actions())))
        else:
            self.q4 = np.load(q_values)['Q4']

    def epsilon_greedy(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(range(len(self.env.get_actions())))
        else:
            q = self.q1[state] + self.q2[state] + self.q3[state] + self.q4[state]
            return np.argmax(q)

    def update_q_value(self, state, next_state, action, reward):
        choice = random.choice(range(4))
        if choice == 0:
            best_action = np.argmax((self.q2[next_state] + self.q3[next_state] + self.q4[next_state]) / 3)
            self.q1[state][action] += self.alpha * (reward + self.gamma * self.q2[next_state][np.argmax(self.q1[next_state])] - self.q1[state][action])
            self.q1[state][action] += self.alpha * (reward + self.gamma * min(self.q2[next_state][best_action], self.q3[next_state][best_action], self.q4[next_state][best_action]) - self.q1[state][action])
        elif choice == 1:
            best_action = np.argmax((self.q1[next_state] + self.q3[next_state] + self.q4[next_state]) / 3)
            self.q2[state][action] += self.alpha * (reward + self.gamma * min(self.q1[next_state][best_action], self.q3[next_state][best_action], self.q4[next_state][best_action]) - self.q2[state][action])
        elif choice == 2:
            best_action = np.argmax((self.q1[next_state] + self.q2[next_state] + self.q4[next_state]) / 3)
            self.q3[state][action] += self.alpha * (reward + self.gamma * min(self.q1[next_state][best_action], self.q2[next_state][best_action], self.q4[next_state][best_action]) - self.q3[state][action])
        else:
            best_action = np.argmax((self.q1[next_state] + self.q2[next_state] + self.q3[next_state]) / 3)
            self.q4[state][action] += self.alpha * (reward + self.gamma * min(self.q1[next_state][best_action], self.q2[next_state][best_action], self.q3[next_state][best_action]) - self.q4[state][action])

    def train(self, episodes):
        save_file_name = "q_values_" + self.algo_name.replace(' ', '_').lower()
        rewards = []
        steps = []

        for i in range(episodes):
            state = self.env.reset()
            total_reward = 0
            steps_taken = 0
            done = False
            while not done:
                action_idx = self.epsilon_greedy(state)
                action = self.env.get_actions()[action_idx]
                next_state, reward, done = self.env.step(action)
                self.update_q_value(state, next_state, action_idx, reward)
                state = next_state
                total_reward += reward
                steps_taken += 1

            rewards.append(total_reward)
            steps.append(steps_taken)

            self.decay_epsilon()

            if i % 100 == 0:
                print(f"Episode {i} completed.")

            self.save_episode_rewards(total_reward)
            self.save_episode_steps(steps_taken)

            if i % 1000 == 0:
                np.savez(save_file_name, Q1=self.q1, Q2=self.q2, Q3=self.q3, Q4=self.q4)

        return rewards, steps, [self.q1, self.q2, self.q3, self.q4]
