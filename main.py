# main.py

import pygame
import sys
from settings import SCREEN_WIDTH, SCREEN_HEIGHT, FONT_NAME, FONT_SIZE, FPS
from settings import STATE_SPACES, REWARD_SETTINGS, NUM_EPISODES

from environment import Environment
from agent import Agent
from experiments import run_all_experiments, run_all_experiments2
from experiments_sarsa import run_all_experiments_sarsa

def interactive_play():
    """
    Example function to play the game interactively with a trained (or random) agent.
    For demonstration after experiments are done.
    """
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption('Snake Game - Interactive Play')
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(FONT_NAME, FONT_SIZE)

    # Just load a Q-table if you want your agent to use the learned behavior
    # agent = Agent(state_space=STATE_SPACES["advanced"])  # or "basic", "danger_only"
    # agent.load_q_table("q_tables/q_table_advanced_standard.pkl")
    agent = Agent(state_space=[])  # empty = random policy

    env = Environment(rewards={})
    done = False
    episode = 1

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        state = agent.get_state(env.snake, env.food)
        action = agent.choose_action(state)  # random or Q-based
        _, done = env.step(action)

        # Render
        env.draw(screen, font, episode)
        pygame.display.update()
        clock.tick(FPS)

        if done:
            episode += 1
            env.reset()

def main():
    # Run experiments (train agent with different states & rewards)
    results = run_all_experiments()
    print("All Q learning experiments finished. Results saved and plotted.")


    results = run_all_experiments_sarsa()
    print("All Sarsa learning experiments finished. Results saved and plotted.")

    # Optionally, play a quick interactive session
    # interactive_play()

if __name__ == "__main__":
    main()
