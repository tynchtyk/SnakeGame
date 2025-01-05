# agent.py

import random
import numpy as np
import pickle
from settings import (
    ACTIONS, LEARNING_RATE, DISCOUNT_FACTOR, EXPLORATION_RATE,
    EXPLORATION_DECAY, MIN_EXPLORATION_RATE, TILE_SIZE, GRID_WIDTH, GRID_HEIGHT
)

class Agent:
    def __init__(self, state_space, exploration_rate=EXPLORATION_RATE):
        """
        state_space: e.g. STATE_SPACES["S1"], STATE_SPACES["S2"], or STATE_SPACES["S3"]
        """
        self.q_table = {}
        self.state_space = state_space
        self.exploration_rate = exploration_rate

    def choose_action(self, state):
        """
        Epsilon-greedy strategy
        """
        if random.random() < self.exploration_rate:
            return random.choice(ACTIONS)
        else:
            if state not in self.q_table:
                self.q_table[state] = np.zeros(len(ACTIONS))
                return random.choice(ACTIONS)
            else:
                return ACTIONS[np.argmax(self.q_table[state])]

    def learn(self, state, action, reward, next_state, done):
        """
        Q-learning update
        """
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(ACTIONS))
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(len(ACTIONS))

        action_idx = ACTIONS.index(action)
        old_value = self.q_table[state][action_idx]
        next_max = 0.0 if done else np.max(self.q_table[next_state])

        new_value = old_value + LEARNING_RATE * (
            reward + DISCOUNT_FACTOR * next_max - old_value
        )
        self.q_table[state][action_idx] = new_value

    def update_exploration_rate(self):
        """
        Decays epsilon but won't go below MIN_EXPLORATION_RATE
        """
        if self.exploration_rate > MIN_EXPLORATION_RATE:
            self.exploration_rate *= EXPLORATION_DECAY
            if self.exploration_rate < MIN_EXPLORATION_RATE:
                self.exploration_rate = MIN_EXPLORATION_RATE

    def save_q_table(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"Q-table saved to {filename}")

    def load_q_table(self, filename):
        with open(filename, 'rb') as f:
            self.q_table = pickle.load(f)
        print(f"Q-table loaded from {filename}")

    def get_state(self, snake, food):
        """
        Build a feature tuple based on which state space is in self.state_space.
        We'll branch logic for S1, S2, S3 for clarity.
        """
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
        

    # -----------------------------
    # S1: { ws, wl, wr, qf, qt }
    # -----------------------------
    def get_state_s1(self, snake, food):
        head = snake.body[0]
        direction = snake.direction

        # 1) ws, wl, wr: is there a wall if we go straight, left, right?
        ws = self.is_wall_ahead(snake, direction)
        wl = self.is_wall_ahead(snake, self.turn_left(direction))
        wr = self.is_wall_ahead(snake, self.turn_right(direction))

        # 2) qf: relative position of food
        #    We'll do a simple approach: 
        #    qf = (dx, dy) where dx, dy in {-1, 0, +1} if it's left/right or up/down
        #    Or you might do a single integer encoding. We'll do a small 2D approach:
        fx, fy = food.position
        qf_x = 0
        if fx < head[0]: qf_x = -1
        elif fx > head[0]: qf_x = 1
        qf_y = 0
        if fy < head[1]: qf_y = -1
        elif fy > head[1]: qf_y = 1
        qf = (qf_x, qf_y)

        # 3) qt: relative position of tail -> last segment in snake.body
        tail = snake.body[-1]
        qt_x = 0
        if tail[0] < head[0]: qt_x = -1
        elif tail[0] > head[0]: qt_x = 1
        qt_y = 0
        if tail[1] < head[1]: qt_y = -1
        elif tail[1] > head[1]: qt_y = 1
        qt = (qt_x, qt_y)

        return (ws, wl, wr, qf, qt)

    # -----------------------------
    # S2: 10 features
    # [danger_straight, danger_left, danger_right,
    #  moving_left, moving_right, moving_up, moving_down,
    #  food_left, food_up, food_down]
    # -----------------------------
    def get_state_s2(self, snake, food):
        head = snake.body[0]
        direction = snake.direction

        danger_straight = self.is_danger(snake, snake.get_next_position(direction))
        danger_left = self.is_danger(snake, snake.get_next_position(self.turn_left(direction)))
        danger_right = self.is_danger(snake, snake.get_next_position(self.turn_right(direction)))

        # moving_{...}
        moving_left = int(direction == 'LEFT')
        moving_right = int(direction == 'RIGHT')
        moving_up = int(direction == 'UP')
        moving_down = int(direction == 'DOWN')

        # food_{...} (binary indicators)
        fx, fy = food.position
        food_left = int(fx < head[0])
        food_right = int(fx > head[0])  # not explicitly in your list, but we might want it
        food_up = int(fy < head[1])
        food_down = int(fy > head[1])

        # If you only want {food_left, food_up, food_down} and not 'food_right',
        # you can omit 'food_right'.
        # We'll do exactly what's stated: "food_left, food_up, food_down"
        # So let's define them carefully:

        return (
            danger_straight,
            danger_left,
            danger_right,
            moving_left,
            moving_right,
            moving_up,
            moving_down,
            food_left,
            food_up,
            food_down
        )

    # -----------------------------
    # S3: 8 directions, each with 5 features:
    # [see_body, see_wall, dist_body, dist_food, dist_wall]
    # => 40 features total
    # -----------------------------
    def get_state_s3(self, snake, food):
        # We'll define 8 directions as: 
        # 0 = Up, 1 = UpRight, 2 = Right, 3=DownRight, 4=Down, 5=DownLeft, 6=Left, 7=UpLeft
        # For each direction, we compute:
        #   see_body (0/1), see_wall (0/1), dist_body, dist_food, dist_wall
        # This is a placeholder; real logic can be more complicated.

        head = snake.body[0]
        directions_8 = [
            (0, -1),  # Up
            (1, -1),  # UpRight
            (1, 0),   # Right
            (1, 1),   # DownRight
            (0, 1),   # Down
            (-1, 1),  # DownLeft
            (-1, 0),  # Left
            (-1, -1)  # UpLeft
        ]

        features = []
        for d in directions_8:
            dx, dy = d
            (see_body, see_wall, dist_body, dist_food, dist_wall) = self.explore_direction(head, dx, dy, snake, food)
            features.append(see_body)
            features.append(see_wall)
            features.append(dist_body)
            features.append(dist_food)
            features.append(dist_wall)

        return tuple(features)
    def get_state_s4(self, snake, food):
        head_x, head_y = snake.body[0]
        direction = snake.direction

        # 1) Danger detection
        point_straight = snake.get_next_position(direction)
        point_left = snake.get_next_position(self.turn_left(direction))
        point_right = snake.get_next_position(self.turn_right(direction))
        danger_straight = self.is_danger(snake, point_straight)
        danger_left = self.is_danger(snake, point_left)
        danger_right = self.is_danger(snake, point_right)

        # 2) Snake direction (one-hot)
        dir_up = int(direction == 'UP')
        dir_down = int(direction == 'DOWN')
        dir_left = int(direction == 'LEFT')
        dir_right = int(direction == 'RIGHT')

        # 3) Food distance (Manhattan)
        fx, fy = food.position
        food_dist_x = abs(fx - head_x) // TILE_SIZE
        food_dist_y = abs(fy - head_y) // TILE_SIZE

        # 4) Wall distances (in tiles)
        wall_dist_up = head_y // TILE_SIZE
        wall_dist_down = (GRID_HEIGHT * TILE_SIZE - head_y) // TILE_SIZE
        wall_dist_left = head_x // TILE_SIZE
        wall_dist_right = (GRID_WIDTH * TILE_SIZE - head_x) // TILE_SIZE

        return (
            danger_straight, 
            danger_left, 
            danger_right,
            dir_up, 
            dir_down, 
            dir_left, 
            dir_right,
            food_dist_x, 
            food_dist_y,
            wall_dist_up,
            wall_dist_down,
            wall_dist_left,
            wall_dist_right
        )
    def get_state_s5(self, snake, food):
        head_x, head_y = snake.body[0]
        
        # 1) Danger checks
        point_straight = snake.get_next_position(snake.direction)
        point_left = snake.get_next_position(self.turn_left(snake.direction))
        point_right = snake.get_next_position(self.turn_right(snake.direction))
        danger_straight = self.is_danger(snake, point_straight)
        danger_left = self.is_danger(snake, point_left)
        danger_right = self.is_danger(snake, point_right)

        # 2) Snake direction (one-hot)
        snake_dir_left = int(snake.direction == 'LEFT')
        snake_dir_right = int(snake.direction == 'RIGHT')
        snake_dir_up = int(snake.direction == 'UP')
        snake_dir_down = int(snake.direction == 'DOWN')

        # 3) Food direction in x, y
        #    -1 if food is left/up, 0 if same coordinate, +1 if right/down
        fx, fy = food.position

        # food_direction_x
        if fx < head_x:
            fd_x = -1
        elif fx > head_x:
            fd_x = 1
        else:
            fd_x = 0

        # food_direction_y
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



    # -----------
    # HELPER FUNCS
    # -----------
    def is_wall_ahead(self, snake, direction):
        """
        Returns 1 if the next position in `direction` is outside the grid, else 0.
        """
        next_pos = snake.get_next_position(direction)
        x, y = next_pos
        if x < 0 or x >= GRID_WIDTH * TILE_SIZE or y < 0 or y >= GRID_HEIGHT * TILE_SIZE:
            return 1
        return 0

    def is_danger(self, snake, point):
        """
        Danger means either a wall or self-collision.
        """
        x, y = point
        # Check wall
        if x < 0 or x >= GRID_WIDTH * TILE_SIZE or y < 0 or y >= GRID_HEIGHT * TILE_SIZE:
            return 1
        # Check body
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

    def explore_direction(self, head, dx, dy, snake, food):
        """
        For S3: we step outward from the head along (dx, dy):
          - see_body: 1 if the first object is snake's body
          - see_wall: 1 if the first object is a wall
          - dist_body: how many steps until we see body (0 if we don't see it)
          - dist_food: how many steps until food (0 if we don't see it)
          - dist_wall: how many steps until wall
        We keep stepping until we hit something or go outside the grid.
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

            # check if out of bounds (wall)
            if (cur_x < 0 or cur_x >= GRID_WIDTH * TILE_SIZE or
                cur_y < 0 or cur_y >= GRID_HEIGHT * TILE_SIZE):
                # We found the wall
                dist_wall = steps
                see_wall = 1
                break

            # Otherwise, check if body
            if [cur_x, cur_y] in snake.body:
                if see_body == 0:  # first time we see it
                    see_body = 1
                    dist_body = steps
                # We keep going to see if there's also a wall further
                # But the first object found is typically body, so we might break here
                # If you prefer to stop once you see body, you can break here.

            # Check if this is the food
            if (cur_x, cur_y) == tuple(food.position):
                if dist_food == 0:  # first time
                    dist_food = steps
                # Similarly, you could choose to break if you want "first object" logic.

            # If we want the "first object" approach, we might break as soon as we see something.
            # But let's assume we want to continue to find the wall distance.
            # So we only break once we see the wall.
            # Continue stepping...

        return (see_body, see_wall, dist_body, dist_food, dist_wall)
