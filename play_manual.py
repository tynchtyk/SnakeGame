# play_manual.py

import pygame
import sys
from settings import (SCREEN_WIDTH, SCREEN_HEIGHT, FONT_NAME, FONT_SIZE, FPS)
from environment import Environment

def turn_left(direction):
    directions = ['UP', 'LEFT', 'DOWN', 'RIGHT']
    idx = directions.index(direction)
    return directions[(idx + 1) % 4]

def turn_right(direction):
    directions = ['UP', 'RIGHT', 'DOWN', 'LEFT']
    idx = directions.index(direction)
    return directions[(idx + 1) % 4]

def get_relative_action(current_direction, desired_direction):
    if desired_direction == current_direction:
        return 'STRAIGHT'
    elif desired_direction == turn_left(current_direction):
        return 'LEFT'
    elif desired_direction == turn_right(current_direction):
        return 'RIGHT'
    else:
        # Opposite direction or invalid turn; continue straight
        return 'STRAIGHT'

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption('Snake Game - Manual Play')
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(FONT_NAME, FONT_SIZE)

    env = Environment()
    done = False
    episode = 1

    action = 'STRAIGHT'

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            # Movement control
            elif event.type == pygame.KEYDOWN:
                current_direction = env.snake.direction
                if event.key == pygame.K_UP:
                    desired_direction = 'UP'
                elif event.key == pygame.K_DOWN:
                    desired_direction = 'DOWN'
                elif event.key == pygame.K_LEFT:
                    desired_direction = 'LEFT'
                elif event.key == pygame.K_RIGHT:
                    desired_direction = 'RIGHT'
                else:
                    desired_direction = current_direction  # No change

                # Determine the relative action
                action = get_relative_action(current_direction, desired_direction)

        # Move the snake based on the action
        env.snake.move(action)

        # Reset action to 'STRAIGHT' after moving
        action = 'STRAIGHT'

        # Check collision with food
        if env.snake.body[0] == env.food.position:
            env.snake.grow()
            env.score += 1
            # Place new food
            while True:
                env.food = env.food.__class__()
                if env.food.position not in env.snake.body:
                    break

        # Check collision with walls or self
        head_x, head_y = env.snake.body[0]
        if (head_x < 0 or head_x >= SCREEN_WIDTH or
            head_y < 0 or head_y >= SCREEN_HEIGHT or
            env.snake.body[0] in env.snake.body[1:]):
            print(f"Game Over! Score: {env.score}")
            done = True
            break

        # Draw everything
        env.draw(screen, font, episode)
        pygame.display.update()
        clock.tick(FPS / 5)

    # Close the game
    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()
