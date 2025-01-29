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

def run_value_iteration(goal_index: int, discount_rate: float = 0.95) -> tuple[dict, State]: # thinking of changing discount rate to 0.9
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
    
    # Run value iteration with progress tracking
    value_it = ValueIteration(
        mdp=env, 
        initial_value=0.0, 
        threshold=1e-6, 
        verbose=False,  # Disable default printing
        max_iterations=10000
    )
    
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
    
    if not value_it.has_converged():
        print("Warning: Value iteration did not converge to specified threshold")
    
    return value_it.get_value_function(), goal_state

def save_results(value_function: dict, goal_state: State, goal_index: int) -> None:
    """Save value function and goal state to file.
    
    Args:
        value_function: Computed value function
        goal_state: Goal state used
        goal_index: Index of goal state for filename
    """
    filename = f'./value-iteration-results/value_function_goal_{goal_index}.pkl'
    pd.to_pickle((value_function, goal_state), filename)
    print(f"Value function saved as {filename}")

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