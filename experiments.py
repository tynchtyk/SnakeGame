# experiments.py

import matplotlib.pyplot as plt
import os
from settings import (
    STATE_SPACES, REWARD_SETTINGS, NUM_EPISODES, MAX_STEPS_PER_EPISODE
)
from environment import Environment
from agent import Agent

import pygame
import sys
from environment import Environment
from agent import Agent
from settings import (
    SCREEN_WIDTH, SCREEN_HEIGHT, FONT_NAME, FONT_SIZE, FPS,
    MAX_STEPS_PER_EPISODE
)

def run_experiment(state_space, rewards, num_episodes=1000, show_game=False):
    agent = Agent(state_space=state_space)
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
        state = agent.get_state(env.snake, env.food)
        done = False
        episode_reward = 0

        while not done:
            action = agent.choose_action(state)
            reward, done = env.step(action)
            next_state = agent.get_state(env.snake, env.food)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            if show_game:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                env.draw(screen, font, episode)
                pygame.display.update()
                clock.tick(15)

        agent.update_exploration_rate()
        total_rewards.append(episode_reward)
        lengths.append(len(env.snake.body))

    if show_game:
        pygame.quit()

    return total_rewards, lengths, agent





def run_all_experiments():
    """
    Runs all combinations of state spaces and reward settings,
    then creates a separate plot for each reward to compare different state spaces.
    """
    # Structure to hold data:
    # results[reward_name][state_name] = (total_rewards, lengths)
    results = {}

    # Initialize the nested dictionary
    for reward_name in REWARD_SETTINGS:
        results[reward_name] = {}

    # 1. Run all combinations
    for reward_name, rewards in REWARD_SETTINGS.items():
        for state_name, state_space in STATE_SPACES.items():
            experiment_key = f"{state_name}_{reward_name}"
            print(f"=== Running {experiment_key} ===")

            # Run the experiment
            total_rewards, lengths, agent = run_experiment(
                state_space=state_space,
                rewards=rewards,
                num_episodes=NUM_EPISODES,
                show_game=False
            )

            # Store in nested dict
            results[reward_name][state_name] = (total_rewards, lengths)

            # Save the Q-table
            q_table_filename = f"q_tables/q_table_{experiment_key}.pkl"
            agent.save_q_table(q_table_filename)

    # 2. Plot each reward separately
    step = 20  # downsample step (plot every 20th point)
    for reward_name, state_dict in results.items():
        plt.figure(figsize=(12, 5))

        # ---------- Subplot 1: total rewards (learning curve) ----------
        plt.subplot(1, 2, 1)
        for state_name, (total_rewards, lengths) in state_dict.items():
            episodes = range(1, len(total_rewards) + 1)
            # Plot every 'step'-th point
            plt.plot(
                [e for e in episodes][::step],
                [r for r in total_rewards][::step],
                label=state_name
            )
        plt.title(f"Learning Curve - Reward: {reward_name}")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.legend()

        # ---------- Subplot 2: final snake length ----------
        plt.subplot(1, 2, 2)
        for state_name, (total_rewards, lengths) in state_dict.items():
            episodes = range(1, len(lengths) + 1)
            plt.plot(
                [e for e in episodes][::step],
                [l for l in lengths][::step],
                label=state_name
            )
        plt.title(f"Final Snake Length - Reward: {reward_name}")
        plt.xlabel("Episode")
        plt.ylabel("Snake Length")
        plt.legend()

        plt.tight_layout()
        plt.show()

    return results

def run_all_experiments2():
    os.makedirs("q_tables", exist_ok=True)

    # results[reward_name][state_name] = (total_rewards, lengths)
    results = {"R1": {}, "R2": {}, "R3": {}, "R4": {}}

    # ---------- 1) State S1 + Reward R1 ----------
    total_rewards, lengths, agent = run_experiment(
        state_space=STATE_SPACES["S1"],
        rewards=REWARD_SETTINGS["R1"],
        num_episodes=NUM_EPISODES,
        show_game=False
    )
    agent.save_q_table("q_tables/q_table_S1_R1.pkl")
    results["R1"]["S1"] = (total_rewards, lengths)

    # ---------- 2) State S2 + Reward R2 ----------
    total_rewards, lengths, agent = run_experiment(
        state_space=STATE_SPACES["S2"],
        rewards=REWARD_SETTINGS["R2"],
        num_episodes=NUM_EPISODES,
        show_game=False
    )
    agent.save_q_table("q_tables/q_table_S2_R2.pkl")
    results["R2"]["S2"] = (total_rewards, lengths)

    # ---------- 3) State S3 + Reward R3 ----------
    total_rewards, lengths, agent = run_experiment(
        state_space=STATE_SPACES["S3"],
        rewards=REWARD_SETTINGS["R3"],
        num_episodes=NUM_EPISODES,
        show_game=False
    )
    agent.save_q_table("q_tables/q_table_S3_R3.pkl")
    results["R3"]["S3"] = (total_rewards, lengths)

    # ---------- 3) State S4 + Reward R4 ----------
    total_rewards, lengths, agent = run_experiment(
        state_space=STATE_SPACES["S4"],
        rewards=REWARD_SETTINGS["R4"],
        num_episodes=NUM_EPISODES,
        show_game=False
    )
    agent.save_q_table("q_tables/q_table_S4_R4.pkl")
    results["R4"]["S4"] = (total_rewards, lengths)

    # ---------- 3) State S5 + Reward R5 ----------
    total_rewards, lengths, agent = run_experiment(
        state_space=STATE_SPACES["S5"],
        rewards=REWARD_SETTINGS["R2"],  # or whichever reward you want
        num_episodes=500,
        show_game=False
    )
    agent.save_q_table("q_tables/q_table_S5.pkl")
    
    # ---------- Plot All Together on One Figure ----------
    step = 20  # downsample step (plot every 20th point)
    plt.figure(figsize=(12, 5))

    # -- Subplot 1: Total Reward --
    plt.subplot(1, 2, 1)
    for reward_name, state_dict in results.items():
        for state_name, (total_rewards, lengths) in state_dict.items():
            episodes = range(1, len(total_rewards) + 1)
            # Plot every 'step'-th point
            plt.plot(
                [e for e in episodes][::step],
                [r for r in total_rewards][::step],
                label=f"{state_name}+{reward_name}"
            )
    plt.title("Total Reward (Downsampled)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()

    # -- Subplot 2: Snake Length --
    plt.subplot(1, 2, 2)
    for reward_name, state_dict in results.items():
        for state_name, (total_rewards, lengths) in state_dict.items():
            episodes = range(1, len(lengths) + 1)
            plt.plot(
                [e for e in episodes][::step],
                [l for l in lengths][::step],
                label=f"{state_name}+{reward_name}"
            )
    plt.title("Snake Length (Downsampled)")
    plt.xlabel("Episode")
    plt.ylabel("Length")
    plt.legend()

    plt.tight_layout()
    plt.show()

    return results