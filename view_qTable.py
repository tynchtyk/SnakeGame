# view_q_table.py

import pickle
import pandas as pd
import numpy as np
from settings import ACTIONS

def main():
    # Load the Q-table
    with open('q_table.pkl', 'rb') as f:
        q_table = pickle.load(f)
    
    # Check the type of q_table
    print(f"Type of q_table: {type(q_table)}")
    
    # Inspect the number of entries
    print(f"Number of entries in Q-table: {len(q_table)}")
    
    # If q_table is mapping states to arrays of Q-values
    if isinstance(next(iter(q_table.values())), (list, tuple, np.ndarray)):
        # Convert Q-table to DataFrame
        df = q_table_to_dataframe(q_table)
        
        # Display sample of the DataFrame
        print("\nSample of the Q-table data:")
        print(df.head())
        
        # Visualize the Q-table as an HTML file with color-coding
        visualize_q_table(df)
    else:
        print("Visualization for state-action pair Q-tables is not implemented.")

def q_table_to_dataframe(q_table):
    # Prepare data for DataFrame
    data = []
    for state, q_values in q_table.items():
        state_features = {
            'danger_straight': state[0],
            'danger_left': state[1],
            'danger_right': state[2],
            'moving_left': int(state[3]),
            'moving_right': int(state[4]),
            'moving_up': int(state[5]),
            'moving_down': int(state[6]),
            'food_dir_x': state[7],
            'food_dir_y': state[8],
        }
        for action_idx, q_value in enumerate(q_values):
            entry = state_features.copy()
            entry['action'] = ACTIONS[action_idx]
            entry['q_value'] = q_value
            data.append(entry)
    
    df = pd.DataFrame(data)
    return df

def visualize_q_table(df):
    # Pivot the DataFrame to have actions as columns
    pivot_df = df.pivot_table(
        index=['danger_straight', 'danger_left', 'danger_right', 'moving_left', 'moving_right', 'moving_up', 'moving_down', 'food_dir_x', 'food_dir_y'],
        columns='action',
        values='q_value',
        aggfunc='mean'
    )
    
    # Reset index to turn multi-index into columns
    pivot_df = pivot_df.reset_index()
    
    # Sample the data if it's too large
    if len(pivot_df) > 500:
        pivot_df = pivot_df.sample(n=500, random_state=42)
        print("Data is large; sampling 500 entries for visualization.")
    
    # Apply styling to the DataFrame
    styled_df = pivot_df.style.background_gradient(cmap='viridis', subset=ACTIONS)
    
    # Save to HTML file
    styled_df.to_html('q_table_visualization.html')
    print("Q-table visualization saved to 'q_table_visualization.html'.")

if __name__ == '__main__':
    main()
