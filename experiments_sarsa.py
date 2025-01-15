# run_experiment_sarsa.py
import numpy as np

import pygame
import sys
import matplotlib.pyplot as plt
import os

from environment import Environment
from sarsa_agent import SarsaAgent
from settings import FPS
import matplotlib.pyplot as plt
import os
from settings import (
    STATE_SPACES, REWARD_SETTINGS, NUM_EPISODES, MAX_STEPS_PER_EPISODE
)
from environment import Environment
import pygame
import sys
from environment import Environment
from settings import (
    SCREEN_WIDTH, SCREEN_HEIGHT, FONT_NAME, FONT_SIZE, FPS,
    MAX_STEPS_PER_EPISODE
)


def run_experiment_sarsa(state_space, rewards, num_episodes=1000, show_game=False):
    agent = SarsaAgent(state_space=state_space)
    env = Environment(rewards=rewards)

    total_rewards = []
    lengths = []

    if show_game:
        pygame.init()
        screen = pygame.display.set_mode((800, 600))
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("arial", 20)

    for episode in range(1, num_episodes + 1):
        env.reset()

        # Initial state & action
        state = agent.get_state(env.snake, env.food)
        action = agent.choose_action(state)

        ep_reward = 0
        done = False

        while not done:
            # Step in environment
            reward, done = env.step(action)
            next_state = agent.get_state(env.snake, env.food)

            if not done:
                next_action = agent.choose_action(next_state)
                # SARSA update
                agent.sarsa_update(state, action, reward, next_state, next_action)
                # Advance state
                state, action = next_state, next_action
            else:
                # Terminal update
                agent.sarsa_update_terminal(state, action, reward)

            ep_reward += reward

            if show_game:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                env.draw(screen, font, episode)
                pygame.display.update()
                clock.tick(FPS)

        agent.update_exploration_rate()
        total_rewards.append(ep_reward)
        lengths.append(len(env.snake.body))

    if show_game:
        pygame.quit()

    return total_rewards, lengths, agent

def moving_average(data, window_size=50):
    """
    Compute the moving average of a list.
    """
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def plot_results(results, step=30, window_size=50):
    """
    Plot the results with moving averages.
    """
    plt.figure(figsize=(12, 5))

    # -- Subplot 1: Total Reward --
    plt.subplot(1, 2, 1)
    for reward_name, state_dict in results.items():
        for state_name, (total_rewards, lengths) in state_dict.items():
            avg_rewards = moving_average(total_rewards, window_size)
            episodes = range(1, len(avg_rewards) + 1)
            plt.plot(
                [e for e in episodes][::step],
                [r for r in avg_rewards][::step],
                label=f"{state_name}+{reward_name}"
            )
    plt.title("Total Reward (Moving Average)")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.legend()

    # -- Subplot 2: Snake Length --
    plt.subplot(1, 2, 2)
    for reward_name, state_dict in results.items():
        for state_name, (total_rewards, lengths) in state_dict.items():
            avg_lengths = moving_average(lengths, window_size)
            episodes = range(1, len(avg_lengths) + 1)
            plt.plot(
                [e for e in episodes][::step],
                [l for l in avg_lengths][::step],
                label=f"{state_name}+{reward_name}"
            )
    plt.title("Snake Length (Moving Average)")
    plt.xlabel("Episode")
    plt.ylabel("Average Length")
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_results_by_reward(results, step=30, window_size=50):
    for reward_name, state_dict in results.items():
        plt.figure(figsize=(12, 5))

        # -- Subplot 1: Total Reward --
        plt.subplot(1, 2, 1)
        for state_name, (total_rewards, lengths) in state_dict.items():
            #avg_rewards = moving_average(total_rewards, window_size)
            episodes = range(1, len(total_rewards) + 1)
            plt.plot(
                [e for e in episodes][::step],
                [r for r in total_rewards][::step],
                label=state_name
            )
        plt.title(f"Sarsa Learning Reward Graph: {reward_name}")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()

        # -- Subplot 2: Snake Length --
        plt.subplot(1, 2, 2)
        for state_name, (total_rewards, lengths) in state_dict.items():
            #avg_lengths = moving_average(lengths, window_size)
            episodes = range(1, len(lengths) + 1)
            plt.plot(
                [e for e in episodes][::step],
                [l for l in lengths][::step],
                label=state_name
            )
        plt.title(f"Sarsa Learning Length Graph: {reward_name}")
        plt.xlabel("Episode")
        plt.ylabel("Length")
        plt.legend()

        plt.tight_layout()
        plt.show()

def run_all_experiments_sarsa():
    """
    Runs SARSA training for all combinations of state spaces (S1..S5)
    and rewards (R1..?), collects results, and plots them.
    """

    # results[reward_name][state_name] = (total_rewards, lengths)
    results = {}

    # Create a directory for Q-tables if you want to save them
    os.makedirs("q_tables_sarsa", exist_ok=True)

    # 1. Run over each reward configuration
    for reward_name, rewards in REWARD_SETTINGS.items():
        results[reward_name] = {}

        # 2. For each state space
        for state_name, state_space in STATE_SPACES.items():
            print(f"=== SARSA: Training {state_name} with {reward_name} ===")

            # Run the experiment (no rendering for faster training)
            total_rewards, lengths, agent = run_experiment_sarsa(
                state_space=state_space,
                rewards=rewards,
                num_episodes=NUM_EPISODES,
                show_game=False
            )

            # Optionally save the SARSA Q-table
            q_table_filename = f"q_tables_sarsa/sarsa_qtable_{state_name}_{reward_name}.pkl"
            agent.save_q_table(q_table_filename)

            # Store results
            results[reward_name][state_name] = (total_rewards, lengths)

    plot_results_by_reward(results)

    return results