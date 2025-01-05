import argparse
import pygame
import sys
import os
from agent import Agent           # Q-learning agent
from sarsa_agent import SarsaAgent  # SARSA agent
from environment import Environment
from settings import (
    STATE_SPACES,
    REWARD_SETTINGS,
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    FONT_NAME,
    FONT_SIZE,
    FPS
)

def parse_state_reward(filename):
    """
    Extracts state (e.g., S1, S5) and reward (e.g., R1, R3) from the Q-table filename.
    Assumes filenames like:
      'q_table_S5_R2.pkl' or 'sarsa_qtable_S1_R3.pkl'
    Returns (state, reward).
    """
    base = os.path.basename(filename)
    base = base.replace('.pkl', '')
    parts = base.split('_')

    state, reward = None, None
    for p in parts:
        if p in STATE_SPACES:
            state = p
        if p in REWARD_SETTINGS:
            reward = p
    if not state or not reward:
        raise ValueError(f"Unable to parse state or reward from filename: {filename}")
    return state, reward

def play_agent(qtable_path, agent_type):
    """
    Loads the specified Q-table and plays the Snake game until the agent dies.

    Args:
        qtable_path: Path to the Q-table file.
        agent_type: 'Q' for Q-learning or 'SARSA' for SARSA agent.
    """
    # Parse state and reward from the filename
    state, reward = parse_state_reward(qtable_path)
    print(f"Playing agent with State: {state}, Reward: {reward}")

    # Create the agent and load the Q-table
    if agent_type.upper() == 'Q':
        agent_class = Agent
    elif agent_type.upper() == 'SARSA':
        agent_class = SarsaAgent
    else:
        raise ValueError("Invalid agent type. Must be 'Q' or 'SARSA'.")

    agent = agent_class(state_space=STATE_SPACES[state], exploration_rate=0.0)
    agent.load_q_table(qtable_path)

    # Set up the environment
    env = Environment(rewards=REWARD_SETTINGS[reward])

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption(f"Snake Game - {agent_type} Agent")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(FONT_NAME, FONT_SIZE)

    state = agent.get_state(env.snake, env.food)
    done = False
    total_reward = 0
    steps = 0

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Agent chooses an action
        action = agent.choose_action(state)

        # Step the environment
        reward, done = env.step(action)
        next_state = agent.get_state(env.snake, env.food)
        state = next_state
        total_reward += reward
        steps += 1

        # Render the environment
        env.draw(screen, font, episode=1)
        pygame.display.update()
        clock.tick(FPS)

    print(f"Game Over! Total Steps: {steps}, Total Reward: {total_reward}, Snake Length: {len(env.snake.body)}")
    pygame.quit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Play Snake game using a trained agent.")
    parser.add_argument('--agent_type', type=str, required=True, help="Agent type: 'Q' for Q-learning, 'SARSA' for SARSA")
    parser.add_argument('--qtable', type=str, required=True, help="Path to the Q-table file")

    args = parser.parse_args()
    play_agent(qtable_path=args.qtable, agent_type=args.agent_type)
