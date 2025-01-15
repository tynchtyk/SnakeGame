# settings.py

# ------------------------------
# GLOBAL GAME & RENDER SETTINGS
# ------------------------------
TILE_SIZE = 40
GRID_WIDTH = 30
GRID_HEIGHT = 30
SCREEN_WIDTH = GRID_WIDTH * TILE_SIZE
SCREEN_HEIGHT = GRID_HEIGHT * TILE_SIZE
FPS = 15

# ------------------------------
# ACTIONS
# ------------------------------
ACTIONS = ['STRAIGHT', 'LEFT', 'RIGHT']

# ------------------------------
# STATE SPACE DEFINITIONS
# ------------------------------

# 1) State #1 => { ws, wl, wr, qf, qt }
#    - ws, wl, wr are wall indicators (straight, left, right)
#    - qf, qt are relative positions of food and tail
STATE_1 = [
    'wall_straight',    # ws
    'wall_left',        # wl
    'wall_right',       # wr
    'relative_food',    # qf (you'll encode how to interpret this)
    'relative_tail'     # qt
]

# 2) State #2 => [danger_straight, danger_left, danger_right, 
#                 moving_left, moving_right, moving_up, moving_down, 
#                 food_left, food_up, food_down]
STATE_2 = [
    'danger_straight',
    'danger_left',
    'danger_right',
    'moving_left',
    'moving_right',
    'moving_up',
    'moving_down',
    'food_left',
    'food_up',
    'food_down'
]

# 3) State #3 => x(8), for 8 directions, each with [see_body, see_wall, dist_body, dist_food, dist_wall]
# We'll store 40 features total: 5 features * 8 directions
# For naming, let's do something like:
STATE_3 = []
for i in range(8):
    # i might represent direction index, e.g., 0=Up, 1=UpRight, 2=Right, etc.
    # We'll name them generically: dir0_see_body, dir0_see_wall, ...
    STATE_3.append(f'dir{i}_see_body')
    STATE_3.append(f'dir{i}_see_wall')
    STATE_3.append(f'dir{i}_dist_body')
    STATE_3.append(f'dir{i}_dist_food')
    STATE_3.append(f'dir{i}_dist_wall')

STATE_4 = [
    # Danger flags
    'danger_straight', 
    'danger_left', 
    'danger_right',

    # One-hot snake direction
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
    'wall_dist_right'
]

STATE_5 = [
    'danger_straight', 'danger_left', 'danger_right',
    'snake_direction_left', 'snake_direction_right',
    'snake_direction_up', 'snake_direction_down',
    'food_direction_x', 'food_direction_y'
]

# Combine into a dictionary for easy reference:
STATE_SPACES = {
    "S1": STATE_1,
    "S2": STATE_2,
    "S3": STATE_3,
    "S4": STATE_4,
    "S5": STATE_5
}




# ------------------------------
# REWARD SETTINGS
# ------------------------------

# 1) R1: eat food +500, hit wall -100, hit snake -100, else -10
REWARD_1 = {
    'food': 500,
    'hit_wall': -100,
    'hit_snake': -100,
    'step': -10
}

# 2) R2: eat food +10, hit wall -10, hit snake -10, else 0
REWARD_2 = {
    'food': 10,
    'hit_wall': -10,
    'hit_snake': -10,
    'step': 0
}

# 3) R3: eat food +20, hit wall -24, hit snake -24, get close +0.3, else -0.5
# We'll treat "get close" as a separate key 'closer_to_food' that you handle in environment logic
REWARD_3 = {
    'food': 20,
    'hit_wall': -24,
    'hit_snake': -24,
    'closer_to_food': 0.3,
    'step': -0.5
}

REWARD_4 = {
    'food': 50,            # Big reward for eating food
    'death': -100,         # Big penalty for hitting a wall or itself
    'step': -1,            # Small penalty each step, encourages shorter paths
    'closer_to_food': 0.5  # Bonus if the snake moves closer to the food this step
}


REWARD_SETTINGS = {
    "R1": REWARD_1,
    "R2": REWARD_2,
    "R3": REWARD_3,
    "R4": REWARD_4
}

# --------------------------------
# LEARNING HYPERPARAMETERS
# --------------------------------
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.95
EXPLORATION_RATE = 1.0
EXPLORATION_DECAY = 0.995
MIN_EXPLORATION_RATE = 0.01

# --------------------------------
# MISCELLANEOUS
# --------------------------------
NUM_EPISODES = 2000
MAX_STEPS_PER_EPISODE = 2000

# Colors, Fonts, etc.
COLORS = {
    'white': (255, 255, 255),
    'green': (50, 205, 50),
    'dark_green': (0, 150, 0),
    'red': (220, 20, 60),
    'black': (0, 0, 0),
    'light_gray': (60, 60, 60)
}

FONT_NAME = 'arial'
FONT_SIZE = 20
