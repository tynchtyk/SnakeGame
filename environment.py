# environment.py

import pygame
import random
from settings import TILE_SIZE, ACTIONS, COLORS, GRID_WIDTH, GRID_HEIGHT, REWARDS

class Snake:
    def __init__(self):
        self.size = TILE_SIZE
        start_x = GRID_WIDTH // 2 * TILE_SIZE
        start_y = GRID_HEIGHT // 2 * TILE_SIZE
        self.body = [[start_x, start_y]]
        self.direction = random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT'])
        self.growing = False
    
    def move(self):
        head = self.body[0][:]
        if self.direction == 'UP':
            head[1] -= self.size
        elif self.direction == 'DOWN':
            head[1] += self.size
        elif self.direction == 'LEFT':
            head[0] -= self.size
        elif self.direction == 'RIGHT':
            head[0] += self.size

        self.body.insert(0, head)
        if not self.growing:
            self.body.pop()
        else:
            self.growing = False
        
    def move(self, action):
        # Update the direction based on the action
        if action == 'LEFT':
            self.direction = self.turn_left(self.direction)
        elif action == 'RIGHT':
            self.direction = self.turn_right(self.direction)
        # If action is 'STRAIGHT', direction remains the same

        head = self.body[0][:]
        if self.direction == 'UP':
            head[1] -= self.size
        elif self.direction == 'DOWN':
            head[1] += self.size
        elif self.direction == 'LEFT':
            head[0] -= self.size
        elif self.direction == 'RIGHT':
            head[0] += self.size

        self.body.insert(0, head)
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
    def __init__(self):
        self.snake = Snake()
        self.food = Food()
        self.score = 0

    def reset(self):
        self.snake = Snake()
        self.food = Food()
        self.score = 0

    def step(self, action):
        self.snake.move(action)

        reward = REWARDS['step']
        done = False

        # Check collision with food
        if self.snake.body[0] == self.food.position:
            self.snake.grow()
            reward += REWARDS['food']
            # Place new food
            while True:
                self.food = Food()
                if self.food.position not in self.snake.body:
                    break

        # Check collision with walls or self
        head_x, head_y = self.snake.body[0]
        if (head_x < 0 or head_x >= GRID_WIDTH * TILE_SIZE or
            head_y < 0 or head_y >= GRID_HEIGHT * TILE_SIZE or
            self.snake.body[0] in self.snake.body[1:]):
            reward += REWARDS['death']
            done = True

        self.score += reward
        return reward, done

    def draw(self, surface, font, episode):
        surface.fill(COLORS['black'])
        self.draw_grid(surface)
        self.snake.draw(surface)
        self.food.draw(surface)
        # Draw score and episode
        score_text = f'Score: {self.score} | Episode: {episode}'
        score_surface = font.render(score_text, True, COLORS['white'])
        surface.blit(score_surface, (10, 10))

    def draw_grid(self, surface):
        for x in range(0, GRID_WIDTH * TILE_SIZE, TILE_SIZE):
            pygame.draw.line(surface, COLORS['light_gray'], (x, 0),
                             (x, GRID_HEIGHT * TILE_SIZE))
        for y in range(0, GRID_HEIGHT * TILE_SIZE, TILE_SIZE):
            pygame.draw.line(surface, COLORS['light_gray'], (0, y),
                             (GRID_WIDTH * TILE_SIZE, y))
