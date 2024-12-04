# agent.py

import random
import numpy as np
import pickle
from settings import (ACTIONS, REWARDS, LEARNING_RATE, DISCOUNT_FACTOR,
                      EXPLORATION_DECAY, MIN_EXPLORATION_RATE, TILE_SIZE,
                      GRID_WIDTH, GRID_HEIGHT, EXPLORATION_RATE)

class Agent:
    def __init__(self, exploration_rate=EXPLORATION_RATE):
        self.q_table = {}
        self.exploration_rate = exploration_rate

    def get_state(self, snake, food):
        head_x, head_y = snake.body[0]
        point_l = snake.get_next_position(self.turn_left(snake.direction))
        point_r = snake.get_next_position(self.turn_right(snake.direction))
        point_s = snake.get_next_position(snake.direction)

        # Danger straight, left, right
        danger_straight = self.is_danger(snake, point_s)
        danger_left = self.is_danger(snake, point_l)
        danger_right = self.is_danger(snake, point_r)

        # Food direction
        food_direction_x = 0
        if food.position[0] < head_x:
            food_direction_x = -1
        elif food.position[0] > head_x:
            food_direction_x = 1

        food_direction_y = 0
        if food.position[1] < head_y:
            food_direction_y = -1
        elif food.position[1] > head_y:
            food_direction_y = 1

        state = (
            danger_straight,
            danger_left,
            danger_right,
            snake.direction == 'LEFT',
            snake.direction == 'RIGHT',
            snake.direction == 'UP',
            snake.direction == 'DOWN',
            food_direction_x,
            food_direction_y
        )
        return state

    def is_danger(self, snake, point):
        x, y = point
        # Check wall collision
        if x < 0 or x >= GRID_WIDTH * TILE_SIZE or y < 0 or y >= GRID_HEIGHT * TILE_SIZE:
            return 1
        # Check self-collision
        if [x, y] in snake.body:
            return 1
        return 0

    def turn_left(self, direction):
        directions = ['UP', 'LEFT', 'DOWN', 'RIGHT']
        idx = directions.index(direction)
        return directions[(idx + 1) % 4]

    def turn_right(self, direction):
        directions = ['UP', 'RIGHT', 'DOWN', 'LEFT']
        idx = directions.index(direction)
        return directions[(idx + 1) % 4]

    def choose_action(self, state):
        if random.uniform(0, 1) < self.exploration_rate:
            return random.choice(ACTIONS)
        else:
            if state in self.q_table:
                return ACTIONS[np.argmax(self.q_table[state])]
            else:
                self.q_table[state] = np.zeros(len(ACTIONS))
                return random.choice(ACTIONS)

    def learn(self, state, action, reward, next_state, done):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(ACTIONS))
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(len(ACTIONS))

        action_idx = ACTIONS.index(action)
        old_value = self.q_table[state][action_idx]
        next_max = np.max(self.q_table[next_state])

        # Q-learning formula
        self.q_table[state][action_idx] = old_value + LEARNING_RATE * (
            reward + DISCOUNT_FACTOR * next_max - old_value)

    def update_exploration_rate(self):
        if self.exploration_rate > MIN_EXPLORATION_RATE:
            self.exploration_rate *= EXPLORATION_DECAY

    def save_q_table(self, filename='q_table.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"Q-table saved to {filename}")

    def load_q_table(self, filename='q_table.pkl'):
        with open(filename, 'rb') as f:
            self.q_table = pickle.load(f)
        print(f"Q-table loaded from {filename}")
