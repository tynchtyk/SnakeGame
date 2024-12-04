# play.py

import pygame
import sys
from settings import (SCREEN_WIDTH, SCREEN_HEIGHT, FONT_NAME, FONT_SIZE, FPS)
from environment import Environment
from agent import Agent

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption('Snake Game - Play with Trained Agent')
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(FONT_NAME, FONT_SIZE)

    # Create an agent with exploration_rate = 0
    agent = Agent(exploration_rate=0.0)
    agent.load_q_table('q_table.pkl')

    env = Environment()
    state = agent.get_state(env.snake, env.food)
    total_reward = 0
    done = False
    episode = 1

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        action = agent.choose_action(state)
        reward, done = env.step(action)
        next_state = agent.get_state(env.snake, env.food)
        state = next_state
        total_reward += reward

        env.draw(screen, font, episode)
        pygame.display.update()
        clock.tick(FPS)

        if done:
            print(f"Game Over! Score: {env.score}")
            break

    # Close the game
    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()
