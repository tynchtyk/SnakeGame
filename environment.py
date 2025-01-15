# environment.py

import random
import pygame
from settings import TILE_SIZE, GRID_WIDTH, GRID_HEIGHT, COLORS

class Snake:
    def __init__(self):
        self.size = TILE_SIZE
        start_x = GRID_WIDTH // 2 * TILE_SIZE
        start_y = GRID_HEIGHT // 2 * TILE_SIZE
        self.body = [[start_x, start_y]]
        self.direction = random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT'])
        self.growing = False
    
    def move(self, action):
        # Update direction based on action
        if action == 'LEFT':
            self.direction = self.turn_left(self.direction)
        elif action == 'RIGHT':
            self.direction = self.turn_right(self.direction)
        # If action is 'STRAIGHT', keep direction

        head = self.body[0][:]
        if self.direction == 'UP':
            head[1] -= self.size
        elif self.direction == 'DOWN':
            head[1] += self.size
        elif self.direction == 'LEFT':
            head[0] -= self.size
        elif self.direction == 'RIGHT':
            head[0] += self.size

        self.body.insert(0, head)  # new head
        if not self.growing:
            self.body.pop()
        else:
            self.growing = False

    def get_next_position(self, direction):
        head = self.body[0][:]
        if direction == 'UP':
            head[1] -= self.size
        elif direction == 'DOWN':
            head[1] += self.size
        elif direction == 'LEFT':
            head[0] -= self.size
        elif direction == 'RIGHT':
            head[0] += self.size
        return head

    def turn_left(self, current_direction):
        directions = ['UP', 'LEFT', 'DOWN', 'RIGHT']
        idx = directions.index(current_direction)
        return directions[(idx + 1) % 4]

    def turn_right(self, current_direction):
        directions = ['UP', 'RIGHT', 'DOWN', 'LEFT']
        idx = directions.index(current_direction)
        return directions[(idx + 1) % 4]

    def grow(self):
        self.growing = True

    def draw(self, surface):
        for i, segment in enumerate(self.body):
            color = COLORS['dark_green'] if i == 0 else COLORS['green']
            rect = pygame.Rect(segment[0], segment[1], self.size, self.size)
            pygame.draw.rect(surface, color, rect, border_radius=5)

class Food:
    def __init__(self):
        self.size = TILE_SIZE
        self.position = self.random_position()

    def random_position(self):
        return [
            random.randrange(0, GRID_WIDTH) * TILE_SIZE,
            random.randrange(0, GRID_HEIGHT) * TILE_SIZE
        ]

    def draw(self, surface):
        rect = pygame.Rect(self.position[0], self.position[1],
                           self.size, self.size)
        pygame.draw.rect(surface, COLORS['red'], rect, border_radius=5)

class Environment:
    def __init__(self, rewards):
        """
        'rewards' is a dictionary, e.g.:
         {
            'food': 500,
            'hit_wall': -100,
            'hit_snake': -100,
            'step': -10,
            'closer_to_food': 0.3  # optional
         }
        """
        self.rewards = rewards
        self.reset()

    def reset(self):
        self.snake = Snake()
        self.food = Food()
        self.score = 0

    def step(self, action):
        """
        1. Move the snake based on the action.
        2. Check collisions (wall, self).
        3. Check if food is eaten.
        4. Possibly check 'closer to food' reward if defined.
        5. Update self.score with the reward.
        6. Return (reward, done).
        """
        old_distance = self.distance_to_food()

        # 2) Move
        self.snake.move(action)

        # 3) Initialize reward
        reward = 0
        done = False

        # 4) Check collision with walls
        head_x, head_y = self.snake.body[0]
        if (head_x < 0 or head_x >= GRID_WIDTH * TILE_SIZE or
            head_y < 0 or head_y >= GRID_HEIGHT * TILE_SIZE):
            reward += self.rewards.get('hit_wall', 0)
            done = True

        # 5) Check collision with itself
        if not done:  # only check if we're not already done
            if self.snake.body[0] in self.snake.body[1:]:
                reward += self.rewards.get('hit_snake', 0)
                done = True

        # 6) Check if food is eaten
        if not done and self.snake.body[0] == self.food.position:
            reward += self.rewards.get('food', 0)
            self.snake.grow()
            # Re-spawn food in a valid position
            while True:
                self.food = Food()
                if self.food.position not in self.snake.body:
                    break

        # 7) If not done, check 'closer_to_food' or 'step' penalty
        if not done:
            new_distance = self.distance_to_food()
            if 'closer_to_food' in self.rewards:
                if new_distance < old_distance:
                    reward += self.rewards['closer_to_food']
                else:
                    reward += self.rewards.get('step', 0)
            else:
                # just do step penalty
                reward += self.rewards.get('step', 0)

        self.score += reward
        return reward, done

    def distance_to_food(self):
        """Simple Euclidean or Manhattan distance from snake head to food."""
        head_x, head_y = self.snake.body[0]
        fx, fy = self.food.position
        return abs(head_x - fx) + abs(head_y - fy)

    def draw(self, surface, font, episode):
        surface.fill(COLORS['black'])
        self.draw_grid(surface)
        self.snake.draw(surface)
        self.food.draw(surface)

        text = f"Score: {self.score:.1f} | Episode: {episode}"
        text_surface = font.render(text, True, COLORS['white'])
        surface.blit(text_surface, (10, 10))

    def draw_grid(self, surface):
        for x in range(0, GRID_WIDTH * TILE_SIZE, TILE_SIZE):
            pygame.draw.line(surface, COLORS['light_gray'], (x, 0), (x, GRID_HEIGHT * TILE_SIZE))
        for y in range(0, GRID_HEIGHT * TILE_SIZE, TILE_SIZE):
            pygame.draw.line(surface, COLORS['light_gray'], (0, y), (GRID_WIDTH * TILE_SIZE, y))