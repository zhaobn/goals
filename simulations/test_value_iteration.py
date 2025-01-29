from rllib.shapeworld import ShapeWorld, State, Shape, Action
from rllib.mdp import ValueIteration
import pandas as pd
import os

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

def save_results_to_csv(value_function: dict, goal_state: State, output_dir: str = 'value_iteration_results'):
    """Save value function results to CSV.
    
    Args:
        value_function: Dictionary mapping states to values
        goal_state: The goal state used
        output_dir: Directory to save results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert results to DataFrame
    results = []
    for state, value in value_function.items():
        state_dict = state_to_dict(state)
        state_dict['value'] = value
        state_dict['is_goal'] = (state == goal_state)
        results.append(state_dict)
    
    df = pd.DataFrame(results)
    
    # Sort by value to easily see highest/lowest values
    df = df.sort_values('value', ascending=False)
    
    # Save to CSV
    filename = os.path.join(output_dir, 'value_iteration_results.csv')
    df.to_csv(filename, index=False)
    print(f"\nResults saved to {filename}")
    
    # Save summary statistics
    summary_df = pd.DataFrame({
        'Metric': [
            'Total States',
            'Goal State Value',
            'Max Non-Goal Value',
            'Min Value',
            'Mean Value',
            'Median Value'
        ],
        'Value': [
            len(df),
            df[df['is_goal']]['value'].iloc[0],
            df[~df['is_goal']]['value'].max(),
            df['value'].min(),
            df['value'].mean(),
            df['value'].median()
        ]
    })
    
    summary_filename = os.path.join(output_dir, 'value_iteration_summary.csv')
    summary_df.to_csv(summary_filename, index=False)
    print(f"Summary statistics saved to {summary_filename}")

def test_value_iteration():
    """Test value iteration with a simple goal state."""
    
    # Create a simple goal state
    goal_state = State(
        shape1=Shape(sides='circle', shade='low', texture='plain'),
        shape2=Shape(sides='square', shade='medium', texture='stripes'),
        shape3=Shape(sides='triangle', shade='high', texture='dots')
    )
    
    # Initialize environment
    env = ShapeWorld(goal_state, discount_rate=0.95)
    
    # Run value iteration
    value_it = ValueIteration(
        mdp=env,
        initial_value=0.0,
        threshold=1e-6,
        verbose=True,
        max_iterations=1000
    )
    
    print("\nRunning value iteration...")
    value_it.value_iteration()
    
    # Get value function and optimal policy
    value_function = value_it.get_value_function()
    optimal_policy = value_it.get_optimal_policy()
    
    # Save results to CSV
    save_results_to_csv(value_function, goal_state)
    
    # Validate results
    print("\nValidating results...")
    
    # 1. Check goal state value
    goal_value = value_function[goal_state]
    print(f"Goal state value: {goal_value:.6f} (should be 0)")
    
    # 2. Check that all non-goal states have negative values
    print("\nAnalyzing state values:")
    print(f"Total states: {len(value_function)}")
    
    # More explicit check for non-goal states
    non_goal_values = []
    for state, value in value_function.items():
        if state == goal_state:
            print(f"Goal state value: {value:.6f}")
        else:
            non_goal_values.append(value)
            if value >= 0:  # Debug any non-negative values
                print(f"Warning: Found non-negative value {value:.6f} for non-goal state")
    
    max_non_goal = max(non_goal_values)
    print(f"Number of non-goal states: {len(non_goal_values)}")
    print(f"Maximum non-goal state value: {max_non_goal:.6f} (should be negative)")
    
    # 3. Test policy execution
    print("\nTesting policy execution...")
    
    # Start from a different state
    current_state = State(
        shape1=Shape(sides='square', shade='low', texture='plain'),
        shape2=Shape(sides='circle', shade='low', texture='plain'),
        shape3=Shape(sides='circle', shade='low', texture='plain')
    )
    
    path = [current_state]
    max_steps = 20
    steps = 0
    
    while steps < max_steps and current_state != goal_state:
        if current_state not in optimal_policy:
            print("Error: No optimal action found for state")
            break
            
        action = optimal_policy[current_state]
        next_state = env.next_state_sample(current_state, action)
        
        print(f"\nStep {steps + 1}:")
        print(f"Action: {action}")
        print(f"State value: {value_function[current_state]:.6f}")
        
        current_state = next_state
        path.append(current_state)
        steps += 1
    
    # Report results
    if current_state == goal_state:
        print(f"\nSuccess! Reached goal state in {steps} steps")
    else:
        print(f"\nFailed to reach goal state in {max_steps} steps")
    
    print(f"Path length: {len(path)}")
    print(f"Final state value: {value_function[current_state]:.6f}")

if __name__ == "__main__":
    test_value_iteration() 