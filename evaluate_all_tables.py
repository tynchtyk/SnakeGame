import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from agent import Agent           # Q-learning agent
from sarsa_agent import SarsaAgent  # SARSA agent
from environment import Environment
from settings import STATE_SPACES, REWARD_SETTINGS


def evaluate_agent(qtable_path, agent_class, state_space, rewards, num_episodes=1000, max_steps=1000):
    """
    Evaluates the agent's performance in the environment.

    Args:
        qtable_path (str): Path to the Q-table file.
        agent_class (class): Agent class (Agent or SarsaAgent).
        state_space (list): Feature list for the state space.
        rewards (dict): Reward settings for the environment.
        num_episodes (int): Number of episodes for evaluation.
        max_steps (int): Maximum steps per episode to prevent infinite loops.

    Returns:
        tuple: (best_length, worst_length, avg_length).
    """
    agent = agent_class(state_space=state_space, exploration_rate=0.0)
    agent.load_q_table(qtable_path)

    lengths = []

    for _ in range(num_episodes):
        env = Environment(rewards=rewards)
        state = agent.get_state(env.snake, env.food)
        done = False
        steps = 0

        while not done:
            action = agent.choose_action(state)
            reward, done = env.step(action)
            state = agent.get_state(env.snake, env.food)

            steps += 1
            if steps >= max_steps:  # Terminate the episode if step limit is reached
                break

        lengths.append(len(env.snake.body))

    return max(lengths), min(lengths), sum(lengths) / len(lengths)


def parse_state_reward(filename):
    """
    Parses the state and reward from the Q-table filename.

    Args:
        filename (str): Filename of the Q-table.

    Returns:
        tuple: (state, reward) or (None, None) if parsing fails.
    """
    base = os.path.basename(filename).replace('.pkl', '')
    parts = base.split('_')
    state, reward = None, None
    for p in parts:
        if p in STATE_SPACES:
            state = p
        if p in REWARD_SETTINGS:
            reward = p
    return state, reward


def evaluate_all_tables(num_episodes=1000, max_steps=1000):
    """
    Evaluates all Q-tables (Q-learning and SARSA) and returns a sorted table of results.

    Args:
        num_episodes (int): Number of episodes for evaluation.
        max_steps (int): Maximum steps per episode.

    Returns:
        pandas.DataFrame: Table of results sorted by average length.
    """
    q_files = glob.glob("q_tables/*.pkl")
    sarsa_files = glob.glob("q_tables_sarsa/*.pkl")
    results = []

    # Evaluate Q-learning Q-tables
    for qfile in q_files:
        state, reward = parse_state_reward(qfile)
        if not state or not reward:
            print(f"Skipping file: {qfile} (could not parse state or reward)")
            continue
        print(f"Evaluating Q-learning for {state} + {reward}...")
        best, worst, avg = evaluate_agent(
            qtable_path=qfile,
            agent_class=Agent,
            state_space=STATE_SPACES[state],
            rewards=REWARD_SETTINGS[reward],
            num_episodes=num_episodes,
            max_steps=max_steps
        )
        results.append({
            "State": state,
            "Reward": reward,
            "Agent": "Q-Learning",
            "Best Length": best,
            "Worst Length": worst,
            "Average Length": avg
        })

    # Evaluate SARSA Q-tables
    for sfile in sarsa_files:
        state, reward = parse_state_reward(sfile)
        if not state or not reward:
            print(f"Skipping file: {sfile} (could not parse state or reward)")
            continue
        print(f"Evaluating SARSA for {state} + {reward}...")
        best, worst, avg = evaluate_agent(
            qtable_path=sfile,
            agent_class=SarsaAgent,
            state_space=STATE_SPACES[state],
            rewards=REWARD_SETTINGS[reward],
            num_episodes=num_episodes,
            max_steps=max_steps
        )
        results.append({
            "State": state,
            "Reward": reward,
            "Agent": "SARSA",
            "Best Length": best,
            "Worst Length": worst,
            "Average Length": avg
        })

    df = pd.DataFrame(results)
    df = df.sort_values(by="Average Length", ascending=False)
    return df


def plot_results(df, num_episodes, max_steps):
    """
    Plots the evaluation results as a sorted bar chart.

    Args:
        df (pandas.DataFrame): DataFrame of results.
        num_episodes (int): Number of episodes.
        max_steps (int): Maximum steps per episode.
    """
    labels = [f"{row['Agent']} ({row['State']}+{row['Reward']})" for _, row in df.iterrows()]
    best_lengths = df["Best Length"]
    worst_lengths = df["Worst Length"]
    avg_lengths = df["Average Length"]

    x = range(len(labels))

    plt.figure(figsize=(12, 6))

    # Plot bars for Best, Worst, and Average lengths
    plt.bar(x, best_lengths, label="Best Length", alpha=0.7)
    plt.bar(x, worst_lengths, label="Worst Length", alpha=0.7)
    plt.plot(x, avg_lengths, label="Average Length", color="red", marker="o", linewidth=2)

    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Snake Length")
    plt.title(f"Snake Evaluation Results (Episodes: {num_episodes}, Max Steps: {max_steps})")
    plt.legend()

    plt.tight_layout()
    plt.show()


def main():
    num_episodes = 10000
    max_steps = 5000  # Limit each episode to 1000 steps to prevent infinite loops

    # Evaluate and get the results as a DataFrame
    results_table = evaluate_all_tables(num_episodes=num_episodes, max_steps=max_steps)

    # Print the table in the console
    print("==== Evaluation Results ====")
    print(results_table)

    # Plot the sorted results
    plot_results(results_table, num_episodes, max_steps)

    # Save the table to a CSV file for further analysis
    results_table.to_csv("evaluation_results.csv", index=False)
    print("Results saved to evaluation_results.csv")


if __name__ == "__main__":
    main()
