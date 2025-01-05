# sarsa_agent.py

import random
import numpy as np
import pickle
from settings import (
    ACTIONS, LEARNING_RATE, DISCOUNT_FACTOR, EXPLORATION_RATE,
    EXPLORATION_DECAY, MIN_EXPLORATION_RATE, TILE_SIZE, GRID_WIDTH, GRID_HEIGHT,
    # We'll assume you have S1..S5 in STATE_SPACES (if you want to reference them)
)

class SarsaAgent:
    def __init__(self, state_space, exploration_rate=EXPLORATION_RATE):
        """
        Args:
            state_space: A list of features (e.g. STATE_SPACES["S5"]).
            exploration_rate: Epsilon for epsilon-greedy strategy.
        """
        self.q_table = {}
        self.state_space = state_space
        self.exploration_rate = exploration_rate

    # ----------------------
    # Epsilon-greedy Action
    # ----------------------
    def choose_action(self, state):
        if random.random() < self.exploration_rate:
            return random.choice(ACTIONS)
        else:
            if state not in self.q_table:
                self.q_table[state] = np.zeros(len(ACTIONS))
                return random.choice(ACTIONS)
            return ACTIONS[np.argmax(self.q_table[state])]

    # ----------------
    # SARSA Update
    # ----------------
    def sarsa_update(self, state, action, reward, next_state, next_action):
        """
        SARSA update rule:
          Q(s,a) ← Q(s,a) + α [r + γ * Q(s', a') - Q(s,a)]
        """
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(ACTIONS))
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(len(ACTIONS))

        a_idx = ACTIONS.index(action)
        na_idx = ACTIONS.index(next_action)

        current_q = self.q_table[state][a_idx]
        next_q = self.q_table[next_state][na_idx]
        
        new_q = current_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * next_q - current_q)
        self.q_table[state][a_idx] = new_q

    def sarsa_update_terminal(self, state, action, reward):
        """
        If the episode ends (terminal), there's no Q(s',a') to consider (it's 0).
        So the update is Q(s,a) += α [r - Q(s,a)].
        """
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(ACTIONS))

        a_idx = ACTIONS.index(action)
        old_val = self.q_table[state][a_idx]
        new_val = old_val + LEARNING_RATE * (reward - old_val)
        self.q_table[state][a_idx] = new_val

    # --------------------------
    # Exploration Rate Decay
    # --------------------------
    def update_exploration_rate(self):
        if self.exploration_rate > MIN_EXPLORATION_RATE:
            self.exploration_rate *= EXPLORATION_DECAY
            if self.exploration_rate < MIN_EXPLORATION_RATE:
                self.exploration_rate = MIN_EXPLORATION_RATE

    # ---------------------
    # Q-Table Persistence
    # ---------------------
    def save_q_table(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"SARSA Q-table saved to {filename}")

    def load_q_table(self, filename):
        with open(filename, 'rb') as f:
            self.q_table = pickle.load(f)
        print(f"SARSA Q-table loaded from {filename}")

    # ---------------------
    # Construct State Tuple
    # ---------------------
    def get_state(self, snake, food):
        # Convert self.state_space to a set for comparison
        #current_features = set(self.state_space)

        if set(self.state_space) == {
            'danger_straight', 'danger_left', 'danger_right',
            'snake_direction_left', 'snake_direction_right',
            'snake_direction_up', 'snake_direction_down',
            'food_direction_x', 'food_direction_y'
        }:
            return self.get_state_s5(snake, food)
        if set(self.state_space) == {'danger_straight', 
            'danger_left', 
            'danger_right',
            'snake_direction_up', 
            'snake_direction_down',
            'snake_direction_left', 
            'snake_direction_right',

            # Manhattan distance to food in x, y
            'food_dist_x', 
            'food_dist_y',

            # Distances to walls (how many tiles until hitting a wall)
            'wall_dist_up',
            'wall_dist_down',
            'wall_dist_left',
            'wall_dist_right'}:
            return self.get_state_s4(snake, food)
        if set(self.state_space) == {'wall_straight','wall_left','wall_right','relative_food','relative_tail'}:
            return self.get_state_s1(snake, food)
        elif set(self.state_space) == {
            'danger_straight','danger_left','danger_right',
            'moving_left','moving_right','moving_up','moving_down',
            'food_left','food_up','food_down'
        }:
            return self.get_state_s2(snake, food)
        else:
            # We'll assume it's the 8-direction S3
            return self.get_state_s3(snake, food)

    # --------------------------------
    # S1 Implementation
    # {wall_straight, wall_left, wall_right, relative_food, relative_tail}
    # --------------------------------
    def get_state_s1(self, snake, food):
        head = snake.body[0]
        direction = snake.direction

        ws = self.is_wall_ahead(snake, direction)
        wl = self.is_wall_ahead(snake, self.turn_left(direction))
        wr = self.is_wall_ahead(snake, self.turn_right(direction))

        # relative_food (qf)
        fx, fy = food.position
        qf_x = 0
        if fx < head[0]:
            qf_x = -1
        elif fx > head[0]:
            qf_x = 1
        qf_y = 0
        if fy < head[1]:
            qf_y = -1
        elif fy > head[1]:
            qf_y = 1
        qf = (qf_x, qf_y)

        # relative_tail (qt)
        tail = snake.body[-1]
        qt_x = 0
        if tail[0] < head[0]:
            qt_x = -1
        elif tail[0] > head[0]:
            qt_x = 1
        qt_y = 0
        if tail[1] < head[1]:
            qt_y = -1
        elif tail[1] > head[1]:
            qt_y = 1
        qt = (qt_x, qt_y)

        return (ws, wl, wr, qf, qt)

    # --------------------------------
    # S2 Implementation
    # [danger_straight, danger_left, danger_right,
    #  moving_left, moving_right, moving_up, moving_down,
    #  food_left, food_up, food_down]
    # --------------------------------
    def get_state_s2(self, snake, food):
        head = snake.body[0]
        direction = snake.direction

        danger_straight = self.is_danger(snake, snake.get_next_position(direction))
        danger_left = self.is_danger(snake, snake.get_next_position(self.turn_left(direction)))
        danger_right = self.is_danger(snake, snake.get_next_position(self.turn_right(direction)))

        moving_left = int(direction == 'LEFT')
        moving_right = int(direction == 'RIGHT')
        moving_up = int(direction == 'UP')
        moving_down = int(direction == 'DOWN')

        fx, fy = food.position
        food_left = int(fx < head[0])
        food_up = int(fy < head[1])
        food_down = int(fy > head[1])
        # The original list doesn't mention 'food_right', so we skip it.

        return (
            danger_straight, danger_left, danger_right,
            moving_left, moving_right, moving_up, moving_down,
            food_left, food_up, food_down
        )

    # --------------------------------
    # S3 Implementation
    # 8 directions, each with [see_body, see_wall, dist_body, dist_food, dist_wall]
    # => 40 features total
    # --------------------------------
    def get_state_s3(self, snake, food):
        head = snake.body[0]
        # directions_8 in (dx, dy) form
        directions_8 = [
            (0, -1),   # Up
            (1, -1),   # UpRight
            (1, 0),    # Right
            (1, 1),    # DownRight
            (0, 1),    # Down
            (-1, 1),   # DownLeft
            (-1, 0),   # Left
            (-1, -1)   # UpLeft
        ]

        features = []
        for d in directions_8:
            dx, dy = d
            see_body, see_wall, dist_body, dist_food, dist_wall = self.explore_direction(head, dx, dy, snake, food)
            features.extend([see_body, see_wall, dist_body, dist_food, dist_wall])

        return tuple(features)

    # --------------------------------
    # S4 Implementation
    # [danger_straight, danger_left, danger_right,
    #  snake_direction_up, snake_direction_down, snake_direction_left, snake_direction_right,
    #  food_dist_x, food_dist_y,
    #  wall_dist_up, wall_dist_down, wall_dist_left, wall_dist_right]
    # --------------------------------
    def get_state_s4(self, snake, food):
        head_x, head_y = snake.body[0]
        direction = snake.direction

        point_s = snake.get_next_position(direction)
        point_l = snake.get_next_position(self.turn_left(direction))
        point_r = snake.get_next_position(self.turn_right(direction))
        danger_straight = self.is_danger(snake, point_s)
        danger_left = self.is_danger(snake, point_l)
        danger_right = self.is_danger(snake, point_r)

        dir_up = int(direction == 'UP')
        dir_down = int(direction == 'DOWN')
        dir_left = int(direction == 'LEFT')
        dir_right = int(direction == 'RIGHT')

        fx, fy = food.position
        food_dist_x = abs(fx - head_x) // TILE_SIZE
        food_dist_y = abs(fy - head_y) // TILE_SIZE

        wall_dist_up = head_y // TILE_SIZE
        wall_dist_down = (GRID_HEIGHT * TILE_SIZE - head_y) // TILE_SIZE
        wall_dist_left = head_x // TILE_SIZE
        wall_dist_right = (GRID_WIDTH * TILE_SIZE - head_x) // TILE_SIZE

        return (
            danger_straight, danger_left, danger_right,
            dir_up, dir_down, dir_left, dir_right,
            food_dist_x, food_dist_y,
            wall_dist_up, wall_dist_down, wall_dist_left, wall_dist_right
        )

    # --------------------------------
    # S5 Implementation
    # [danger_straight, danger_left, danger_right,
    #  snake_direction_left, snake_direction_right,
    #  snake_direction_up, snake_direction_down,
    #  food_direction_x, food_direction_y]
    # --------------------------------
    def get_state_s5(self, snake, food):
        head_x, head_y = snake.body[0]
        direction = snake.direction

        point_s = snake.get_next_position(direction)
        point_l = snake.get_next_position(self.turn_left(direction))
        point_r = snake.get_next_position(self.turn_right(direction))

        danger_straight = self.is_danger(snake, point_s)
        danger_left = self.is_danger(snake, point_l)
        danger_right = self.is_danger(snake, point_r)

        snake_dir_left = int(direction == 'LEFT')
        snake_dir_right = int(direction == 'RIGHT')
        snake_dir_up = int(direction == 'UP')
        snake_dir_down = int(direction == 'DOWN')

        fx, fy = food.position
        if fx < head_x:
            fd_x = -1
        elif fx > head_x:
            fd_x = 1
        else:
            fd_x = 0

        if fy < head_y:
            fd_y = -1
        elif fy > head_y:
            fd_y = 1
        else:
            fd_y = 0

        return (
            danger_straight, danger_left, danger_right,
            snake_dir_left, snake_dir_right, snake_dir_up, snake_dir_down,
            fd_x, fd_y
        )

    # ---------------------------------------
    # Helper: Check if Next Position is Wall
    # ---------------------------------------
    def is_wall_ahead(self, snake, direction):
        nxt = snake.get_next_position(direction)
        x, y = nxt
        if x < 0 or x >= GRID_WIDTH * TILE_SIZE or y < 0 or y >= GRID_HEIGHT * TILE_SIZE:
            return 1
        return 0

    # ---------------------------------
    # Danger if wall or snake body
    # ---------------------------------
    def is_danger(self, snake, point):
        x, y = point
        if x < 0 or x >= GRID_WIDTH * TILE_SIZE or y < 0 or y >= GRID_HEIGHT * TILE_SIZE:
            return 1
        if [x, y] in snake.body:
            return 1
        return 0

    # ---------------------------------
    # Directions Turn
    # ---------------------------------
    def turn_left(self, direction):
        directions = ['UP', 'LEFT', 'DOWN', 'RIGHT']
        idx = directions.index(direction)
        return directions[(idx + 1) % 4]

    def turn_right(self, direction):
        directions = ['UP', 'RIGHT', 'DOWN', 'LEFT']
        idx = directions.index(direction)
        return directions[(idx + 1) % 4]

    # --------------------------------
    # S3 "Explore" function
    # stepping outward along (dx, dy)
    # --------------------------------
    def explore_direction(self, head, dx, dy, snake, food):
        """
        For S3: we scan outward until we exit the grid or see something.
        Returns (see_body, see_wall, dist_body, dist_food, dist_wall).
        """
        see_body = 0
        see_wall = 0
        dist_body = 0
        dist_food = 0
        dist_wall = 0

        steps = 0
        cur_x, cur_y = head
        while True:
            steps += 1
            cur_x += dx * TILE_SIZE
            cur_y += dy * TILE_SIZE

            # Check wall
            if (cur_x < 0 or cur_x >= GRID_WIDTH * TILE_SIZE or
                cur_y < 0 or cur_y >= GRID_HEIGHT * TILE_SIZE):
                # Found the wall
                see_wall = 1
                dist_wall = steps
                break

            # Check body
            if [cur_x, cur_y] in snake.body and see_body == 0:
                see_body = 1
                dist_body = steps
                # We can continue searching for wall or food beyond the body if desired

            # Check food
            if (cur_x, cur_y) == tuple(food.position) and dist_food == 0:
                dist_food = steps

            # Keep stepping until we find a wall or exit
            # If you want 'first object' logic, you'd break on the first body or food,
            # but we won't break until we see the wall.

        return (see_body, see_wall, dist_body, dist_food, dist_wall)
