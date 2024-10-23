import numpy as np
import random
import math

class Model:
    up = {'x': -1, 'y': 0}
    down = {'x': 1, 'y': 0}
    left = {'x': 0, 'y': -1}
    right = {'x': 0, 'y': 1}

    def __init__(self, X=50, Y=30):
        self.Y = Y
        self.X = X
        self.grid = np.zeroes((X, Y))
        self.bottom_cliff = set()
        self.top_cliff = set()
        for i in range(2, X-2):
            self.top_cliff.add((i, 0))
            self.bottom_cliff.add((i, Y-1))
        self.goal = random.choice([(i, j) for j in range(Y) for i in range(X-2, X)])
        self.agent_pos = (0, Y//2)
        self.agent_dist = math.sqrt((self.agent_pos[0]-self.goal[0])**2 + (self.agent_pos[1]-self.goal[1])**2)

    def move(self, action):
        new_pos = (self.agent[0] + action['x'], self.agent[1] + action['y'])
        will_slip = random.random() < 0.3

        if will_slip:
            new_pos = (self.agent[0], self.agent[1] + random.choice([-1, 1]))

        if new_pos[0] < 0 or new_pos[0] >= self.X or new_pos[1] < 0 or new_pos[1] >= self.Y:
            reward = -100
            done = True
        elif new_pos in self.top_cliff or new_pos in self.bottom_cliff:
            reward = -100
            done = True
        elif self.is_path_blocked():
            reward = -100
            done = True
        elif new_pos == self.goal:
            reward = 100
            done = True
        else:
            new_pos_dist = math.sqrt((new_pos[0]-self.goal[0])**2 + (new_pos[1]-self.goal[1])**2)
            reward = -1 if new_pos_dist > self.agent_dist else 1
            done = False

        self.agent = new_pos

        return reward, done

    def is_path_blocked(self):
        path_blocked = True
        if self.agent[0] >= self.X-3:
            return False
        for i in range(self.agent[0]+1, self.X-3):
            for j in range(self.Y):
                if (i, j) in self.top_cliff or (i, j) in self.bottom_cliff:
                    path_blocked = True
                else:
                    path_blocked = False
                    break
            if path_blocked:
                return True
        return False

    def grow_cliff(self):
        if self.agent[0] >= self.X-3:
            return
        top_possible_boxes = []
        bottom_possible_boxes = []
        for i in range(self.agent[0]+1, self.X-3):
            for j in range(self.Y):
                if (i, j) in self.top_cliff and (i, j+1) not in self.top_cliff:
                    top_possible_boxes.append((i, j+1))
                elif (i, j) in self.bottom_cliff and (i, j-1) not in self.bottom_cliff:
                    bottom_possible_boxes.append((i, j-1))

        self.top_cliff.add(random.choice(top_possible_boxes))
        self.bottom_cliff.add(random.choice(bottom_possible_boxes))

    def move_goal(self):
        if self.agent[0] >= self.X-3:
            return
        self.goal = random.choice([(i, j) for j in range(self.Y) for i in range(self.X-2, self.X)])
        self.agent_dist = math.sqrt((self.agent[0]-self.goal[0])**2 + (self.agent[1]-self.goal[1])**2)

    def reset(self):
        self.goal = random.choice([(i, j) for j in range(self.Y) for i in range(self.X-2, self.X)])
        self.agent_pos = (0, self.Y//2)
        self.agent_dist = math.sqrt((self.agent_pos[0]-self.goal[0])**2 + (self.agent_pos[1]-self.goal[1])**2)
        self.top_cliff = set()
        self.bottom_cliff = set()
        for i in range(2, self.X-2):
            self.top_cliff.add((i, 0))
            self.bottom_cliff.add((i, self.Y-1))
