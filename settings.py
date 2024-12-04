# settings.py

# Game settings
TILE_SIZE = 40  # Size of each grid cell
GRID_WIDTH = 30  # Number of columns
GRID_HEIGHT = 20  # Number of rows
SCREEN_WIDTH = GRID_WIDTH * TILE_SIZE
SCREEN_HEIGHT = GRID_HEIGHT * TILE_SIZE
FPS = 1000  # Frames per second

# Define possible actions
ACTIONS = ['STRAIGHT', 'LEFT', 'RIGHT']

# Define rewards
REWARDS = {
    'food': 10,
    'death': -100,
    'step': -1  # Penalty for each step to encourage shorter paths
}

# Learning parameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.95
EXPLORATION_RATE = 1.0
EXPLORATION_DECAY = 0.995
MIN_EXPLORATION_RATE = 0.01

# State definitions (you can customize this)
STATE_SPACE = {
    'danger_straight': [0, 1],  # 0: Safe, 1: Danger
    'danger_left': [0, 1],
    'danger_right': [0, 1],
    'snake_direction_left': [0, 1],
    'snake_direction_right': [0, 1],
    'snake_direction_up': [0, 1],
    'snake_direction_down': [0, 1],
    'food_direction_x': [-1, 0, 1],  # -1: Left, 0: Same, 1: Right
    'food_direction_y': [-1, 0, 1]   # -1: Up, 0: Same, 1: Down
}


# Colors
COLORS = {
    'white': (255, 255, 255),
    'green': (50, 205, 50),
    'dark_green': (0, 150, 0),
    'red': (220, 20, 60),
    'black': (0, 0, 0),
    'light_gray': (60, 60, 60)
}

# Font settings
FONT_NAME = 'arial'
FONT_SIZE = 25

# Number of training episodes
NUM_EPISODES = 2000
MAX_STEPS_PER_EPISODE = 1000
