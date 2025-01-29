# Author: Branson Byers
# Summer 2024
'''
Summary: 
* Code to do planning in a shape world, over all possible goals.
* Saves Q-tables as optimal value functions. 
* These Q-tables will be used to compute an optimal goal selection policy. 
'''

##################################################
# IMPORTS
##################################################

# Core imports
import sys
import os
import numpy as np
import pandas as pd
from typing import Sequence
from tqdm import tqdm

# Custom imports
from rllib.shapeworld import ShapeWorld, State, Shape, Action
from rllib.mdp import ValueIteration

##################################################
# VALUE ITERATION FOR A SINGLE GOAL
##################################################

def run_value_iteration(goal_index: int, discount_rate: float = 0.95) -> tuple[dict, State]:
    """Run value iteration for a specific goal state.
    
    Args:
        goal_index: Index of the goal state in state space
        discount_rate: Discount factor for future rewards
        
    Returns:
        tuple: (value_function, goal_state)
    """
    # Initialize temporary environment to get state space
    initial_state = State(
        shape1=Shape(sides='circle', shade='low', texture='plain'),
        shape2=Shape(sides='circle', shade='low', texture='plain'),
        shape3=Shape(sides='circle', shade='low', texture='plain')
    )
    temp_env = ShapeWorld(initial_state, discount_rate)
    state_space = temp_env.get_state_space()
    
    # Get goal state and create environment
    try:
        goal_state = state_space[goal_index]
    except IndexError:
        raise ValueError(f"Goal index {goal_index} is out of range. Max index is {len(state_space)-1}")
    
    env = ShapeWorld(goal_state, discount_rate)
    
    # Run value iteration
    value_it = ValueIteration(
        mdp=env, 
        initial_value=0.0, 
        threshold=1e-6, 
        verbose=False,
        max_iterations=10000
    )
    
    # Run value iteration with progress tracking
    pbar = tqdm(desc=f"Value Iteration (Goal {goal_index})")
    last_delta = float('inf')
    
    while not value_it.has_converged() and value_it.iterations < value_it.max_iterations:
        value_it.value_iteration()
        current_delta = value_it.delta
        
        # Update progress bar
        pbar.update(1)
        pbar.set_postfix({
            'delta': f'{current_delta:.6f}',
            'iterations': value_it.iterations
        })
        
        # Check if delta is not changing significantly
        if abs(current_delta - last_delta) < 1e-8:
            pbar.write("Warning: Delta change is very small, may be stuck")
        last_delta = current_delta
    
    pbar.close()
    
    # Validate results
    value_function = value_it.get_value_function()
    
    # The goal state should have value 0
    if abs(value_function[goal_state]) > 1e-6:
        print(f"Warning: Goal state value is {value_function[goal_state]}, expected 0")
    
    # All other states should have negative values
    non_goal_values = [v for s, v in value_function.items() if s != goal_state]
    if any(v > 0 for v in non_goal_values):
        print("Warning: Some non-goal states have positive values")
        
    if not value_it.has_converged():
        print("Warning: Value iteration did not converge to specified threshold")
    
    return value_function, goal_state

def state_to_dict(state: State) -> dict:
    """Convert a state to a dictionary for DataFrame creation."""
    return {
        'shape1_sides': state.shape1.sides,
        'shape1_shade': state.shape1.shade,
        'shape1_texture': state.shape1.texture,
        'shape2_sides': state.shape2.sides,
        'shape2_shade': state.shape2.shade,
        'shape2_texture': state.shape2.texture,
        'shape3_sides': state.shape3.sides,
        'shape3_shade': state.shape3.shade,
        'shape3_texture': state.shape3.texture
    }

def save_results(value_function: dict, goal_state: State, goal_index: int, 
                output_dir: str = './value-iteration-results') -> None:
    """Save value function results to a single tidy CSV file.
    
    Args:
        value_function: Computed value function
        goal_state: Goal state used
        goal_index: Index of goal state for filename
        output_dir: Directory to save results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert state values to DataFrame
    results = []
    for state, value in value_function.items():
        state_dict = state_to_dict(state)
        state_dict['value'] = value
        state_dict['is_goal'] = (state == goal_state)
        state_dict['goal_index'] = goal_index
        state_dict['metric'] = np.nan  # Add nan for non-summary rows
        state_dict['metric_value'] = np.nan  # Add nan for non-summary rows
        results.append(state_dict)
    
    df = pd.DataFrame(results)
    
    # Calculate summary statistics
    summary_stats = {
        'total_states': len(df),
        'goal_value': df[df['is_goal']]['value'].iloc[0],
        'max_non_goal_value': df[~df['is_goal']]['value'].max(),
        'min_value': df['value'].min(),
        'mean_value': df['value'].mean(),
        'median_value': df['value'].median()
    }
    
    # Add summary stats as rows with a special identifier
    for metric, value in summary_stats.items():
        summary_row = {
            'shape1_sides': 'SUMMARY',
            'shape1_shade': 'SUMMARY_ROW',
            'shape1_texture': 'SUMMARY_ROW',
            'shape2_sides': 'SUMMARY_ROW',
            'shape2_shade': 'SUMMARY_ROW',
            'shape2_texture': 'SUMMARY_ROW',
            'shape3_sides': 'SUMMARY_ROW',
            'shape3_shade': 'SUMMARY_ROW',
            'shape3_texture': 'SUMMARY_ROW',
            'value': value,
            'is_goal': False,
            'goal_index': goal_index,
            'metric': metric,
            'metric_value': str(value)
        }
        results.append(summary_row)
    
    # Create final DataFrame and sort
    final_df = pd.DataFrame(results)
    # First sort by whether it's a summary row (SUMMARY first), then by value descending
    final_df = final_df.sort_values(
        ['shape1_sides', 'value'], 
        ascending=[True, False],  # True for shape1_sides puts SUMMARY first since it comes before letters
        na_position='first'  # Ensures summary rows (with NaN values) stay together
    )
    
    # Save to CSV
    csv_filename = os.path.join(output_dir, f'value_function_goal_{goal_index}.csv')
    final_df.to_csv(csv_filename, index=False)
    print(f"Results saved to {csv_filename}")

def main():
    """Main execution function."""
    if len(sys.argv) != 2:
        print("Usage: python value_iteration.py <goal_index>")
        sys.exit(1)
        
    try:
        goal_index = int(sys.argv[1])
    except ValueError:
        print("Error: goal_index must be an integer")
        sys.exit(1)
        
    value_function, goal_state = run_value_iteration(goal_index)
    save_results(value_function, goal_state, goal_index)

if __name__ == "__main__":
    main()

##################################################