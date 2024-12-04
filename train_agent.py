# main.py

import pygame
import sys
from settings import (SCREEN_WIDTH, SCREEN_HEIGHT, FONT_NAME, FONT_SIZE,
                      NUM_EPISODES, FPS, MAX_STEPS_PER_EPISODE)
from environment import Environment
from agent import Agent

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption('Snake Game with Q-Learning')
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(FONT_NAME, FONT_SIZE)

    agent = Agent()
    scores = []

    for episode in range(1, NUM_EPISODES + 1):
        env = Environment()
        state = agent.get_state(env.snake, env.food)
        total_reward = 0
        done = False
        step = 0
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            action = agent.choose_action(state)
            reward, done = env.step(action)
            next_state = agent.get_state(env.snake, env.food)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            env.draw(screen, font, episode)
            pygame.display.update()
            clock.tick(FPS)

            if done:
                scores.append(env.score)
                agent.update_exploration_rate()
                break
            
            step += 1
            if step >= MAX_STEPS_PER_EPISODE:
                break

    # After training, save the Q-table
    agent.save_q_table('q_table.pkl')

    # Display training results
    print(f"Training completed over {NUM_EPISODES} episodes")
    print(f"Average score: {sum(scores) / len(scores)}")

    # Close the game
    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()
