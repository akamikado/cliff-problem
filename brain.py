import numpy as np
import random

class QLearner:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1, epsilon_decay=0.99):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.q1 = np.zeros((env.rows, env.cols, len(env.get_actions())))

    def epsilon_greedy(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(range(len(self.env.get_actions())))
        else:
            return np.argmax(self.q1[state])

    def update_q_value(self, state, next_state, action, reward):
        self.q1[state][action] += self.alpha * (reward + self.gamma * np.max(self.q1[next_state]) - self.q1[state][action])

    def train(self, episodes):
        rewards = []
        steps = []

        for i in range(episodes):
            state = self.env.reset()
            total_reward = 0
            steps_taken = 0
            done = False
            while not done:
                action = self.epsilon_greedy(state)
                next_state, reward, done = self.env.step(action)
                self.update_q_value(state, next_state, action, reward)
                state = next_state
                total_reward += reward
                steps_taken += 1

            rewards.append(total_reward)
            steps.append(steps_taken)

            self.epsilon *= self.epsilon_decay

            if i % 500 == 0:
                print(f"Episode {i} completed.")

        return rewards, steps, [self.q1]

class DoubleQLearner(QLearner):
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1, epsilon_decay=0.99):
        super().__init__(env, alpha, gamma, epsilon, epsilon_decay)
        self.q2 = np.zeros((env.rows, env.cols, len(env.get_actions())))

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
        rewards = []
        steps = []

        for i in range(episodes):
            state = self.env.reset()
            total_reward = 0
            steps_taken = 0
            done = False
            while not done:
                action = self.epsilon_greedy(state)
                next_state, reward, done = self.env.step(action)
                self.update_q_value(state, next_state, action, reward)
                state = next_state
                total_reward += reward
                steps_taken += 1

            rewards.append(total_reward)
            steps.append(steps_taken)

            self.epsilon *= self.epsilon_decay

            if i % 500 == 0:
                print(f"Episode {i} completed.")

        return rewards, steps, [self.q1, self.q2]

class TripleQLearner(DoubleQLearner):
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1, epsilon_decay=0.99):
        super().__init__(env, alpha, gamma, epsilon, epsilon_decay)
        self.q3 = np.zeros((env.rows, env.cols, len(env.get_actions())))

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
        rewards = []
        steps = []

        for i in range(episodes):
            state = self.env.reset()
            total_reward = 0
            steps_taken = 0
            done = False
            while not done:
                action = self.epsilon_greedy(state)
                next_state, reward, done = self.env.step(action)
                self.update_q_value(state, next_state, action, reward)
                state = next_state
                total_reward += reward
                steps_taken += 1

            rewards.append(total_reward)
            steps.append(steps_taken)

            self.epsilon *= self.epsilon_decay

            if i % 500 == 0:
                print(f"Episode {i} completed.")

        return rewards, steps, [self.q1, self.q2, self.q3]

class QuadrupleQLearner(TripleQLearner):
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1, epsilon_decay=0.99):
        super().__init__(env, alpha, gamma, epsilon, epsilon_decay)
        self.q4 = np.zeros((env.rows, env.cols, len(env.get_actions())))

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
        rewards = []
        steps = []

        for i in range(episodes):
            state = self.env.reset()
            total_reward = 0
            steps_taken = 0
            done = False
            while not done:
                action = self.epsilon_greedy(state)
                next_state, reward, done = self.env.step(action)
                self.update_q_value(state, next_state, action, reward)
                state = next_state
                total_reward += reward
                steps_taken += 1

            rewards.append(total_reward)
            steps.append(steps_taken)

            self.epsilon *= self.epsilon_decay

            if i % 500 == 0:
                print(f"Episode {i} completed.")

        return rewards, steps, [self.q1, self.q2, self.q3, self.q4]
