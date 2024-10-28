import numpy as np
import random
import math

class Model:
    up = {'x': 1, 'y': 0}
    down = {'x': -1, 'y': 0}
    left = {'x': 0, 'y': -1}
    right = {'x': 0, 'y': 1}

    def __init__(self, cols=100, rows=15):
        self.rows = rows
        self.cols = cols
        self.grid = np.zeros((cols, rows))
        self.bottom_cliff = set()
        self.top_cliff = set()
        for i in range(2, cols-2):
            self.bottom_cliff.add((i, 0))
            self.top_cliff.add((i, rows-1))
        self.goal = random.choice([(i, j) for j in range(rows) for i in range(cols-2, cols)])
        self.agent_pos = random.choice([(i, j) for j in range(rows) for i in range(0, 2)])
        self.agent_dist = math.sqrt((self.agent_pos[0]-self.goal[0])**2 + (self.agent_pos[1]-self.goal[1])**2)

    def get_actions(self):
        return [self.up, self.down, self.left, self.right]

    def step(self, action):
        new_pos = (self.agent_pos[0] + action['x'], self.agent_pos[1] + action['y'])
        will_slip = random.random() < 0.3

        if will_slip:
            slip = random.choice([-1, 1])
            new_pos = (new_pos[0], new_pos[1] + slip)

        new_pos = (max(0, min(self.cols-1, new_pos[0])), max(0, min(self.rows - 1, new_pos[1])))

        if new_pos in self.top_cliff or new_pos in self.bottom_cliff:
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
            reward = -0.5 if new_pos_dist >= self.agent_dist else 0.25
            done = False

        self.agent_pos = new_pos

        self.grow_cliff()

        return self.agent_pos, reward, done

    def is_path_blocked(self):
        path_blocked = True
        if self.agent_pos[0] >= self.cols-3:
            return False
        for i in range(self.agent_pos[0]+1, self.cols-3):
            for j in range(self.rows):
                if (i, j) in self.top_cliff or (i, j) in self.bottom_cliff:
                    path_blocked = True
                else:
                    path_blocked = False
                    break
            if path_blocked:
                return True
        return False

    def grow_cliff(self):
        if self.agent_pos[0] >= self.cols-3:
            return
        top_possible_boxes = []
        bottom_possible_boxes = []
        for i in range(self.agent_pos[0]+1, self.cols-3):
            for j in range(self.rows):
                if (i, j) in self.top_cliff and (i, j-1) not in self.top_cliff:
                    top_possible_boxes.append((i, j-1))
                elif (i, j) in self.bottom_cliff and (i, j+1) not in self.bottom_cliff:
                    bottom_possible_boxes.append((i, j+1))

        if top_possible_boxes:
            self.top_cliff.add(random.choice(top_possible_boxes))
        if bottom_possible_boxes:
            self.bottom_cliff.add(random.choice(bottom_possible_boxes))

    def reset(self):
        self.goal = random.choice([(i, j) for j in range(self.rows) for i in range(self.cols-2, self.cols)])
        self.agent_pos = random.choice([(i, j) for j in range(self.rows) for i in range(0, 2)])
        self.agent_dist = math.sqrt((self.agent_pos[0]-self.goal[0])**2 + (self.agent_pos[1]-self.goal[1])**2)
        self.top_cliff = set()
        self.bottom_cliff = set()
        for i in range(2, self.cols-2):
            self.bottom_cliff.add((i, 0))
            self.top_cliff.add((i, self.rows-1))

        return self.agent_pos

    def render(self):
        grid_display = np.full((self.cols, self.rows), '.', dtype=str)

        # Mark the goal
        goal_x, goal_y = self.goal
        grid_display[goal_x, goal_y] = 'G'

        # Mark the cliffs
        for (x, y) in self.top_cliff:
            grid_display[x, y] = '#'
        for (x, y) in self.bottom_cliff:
            grid_display[x, y] = '#'

        # Mark the agent's position
        agent_x, agent_y = self.agent_pos
        grid_display[agent_x, agent_y] = 'A'

        # Print grid display
        for row in grid_display.T:
            print(' '.join(row))
        print("\n")
