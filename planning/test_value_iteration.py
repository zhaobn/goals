import numpy as np
from rllib.shapeworld import ShapeWorld, State, Shape, Action
from rllib.mdp import ValueIteration

def test_simple_goal():
    """Test value iteration with a simple goal state."""
    
    # Create a simple goal state
    goal_state = State(
        shape1=Shape(sides='circle', shade='low', texture='plain'),
        shape2=Shape(sides='square', shade='medium', texture='stripes'),
        shape3=Shape(sides='triangle', shade='high', texture='dots')
    )
    
    # Initialize ShapeWorld
    env = ShapeWorld(goal_state, discount_rate=0.9)
    
    # Run value iteration
    vi = ValueIteration(
        mdp=env,
        initial_value=0.0,
        threshold=1e-6,
        verbose=True,
        max_iterations=1000
    )
    
    vi.value_iteration()
    value_function = vi.get_value_function()
    
    # Basic checks
    print("\nBasic Checks:")
    print(f"Converged: {vi.has_converged()}")
    print(f"Number of iterations: {vi.iterations}")
    print(f"Final delta: {vi.delta}")
    
    # Check goal state value
    goal_value = value_function[goal_state]
    print(f"\nGoal state value: {goal_value}")
    assert goal_value == 0.0, "Goal state should have value 0"
    
    # Get and check optimal policy
    policy = vi.get_optimal_policy()
    
    # Sample a few states and their values
    print("\nSample State Values:")
    sample_states = list(value_function.keys())[:5]
    for s in sample_states:
        print(f"\nState: {s}")
        print(f"Value: {value_function[s]}")
        if s in policy:
            print(f"Optimal action: {policy[s]}")

def test_value_propagation():
    """Test if values propagate correctly from goal state."""
    
    # Create a goal state
    goal_state = State(
        shape1=Shape(sides='circle', shade='low', texture='plain'),
        shape2=Shape(sides='circle', shade='low', texture='plain'),
        shape3=Shape(sides='circle', shade='low', texture='plain')
    )
    
    env = ShapeWorld(goal_state, discount_rate=0.9)
    vi = ValueIteration(mdp=env, threshold=1e-6, max_iterations=1000)
    vi.value_iteration()
    value_function = vi.get_value_function()
    
    # Get states one action away from goal
    print("\nChecking Value Propagation:")
    near_goal_states = []
    for s in vi.states:
        for a in vi.action_map[s]:
            next_states = env.get_possible_next_states(s, a)
            if goal_state in next_states:
                near_goal_states.append(s)
                break
    
    print(f"\nFound {len(near_goal_states)} states one action away from goal")
    if near_goal_states:
        print("\nSample near-goal state values:")
        for s in near_goal_states[:3]:  # Show first 3
            print(f"State: {s}")
            print(f"Value: {value_function[s]}")

def main():
    print("Testing Value Iteration Implementation")
    print("\n1. Testing with simple goal state...")
    test_simple_goal()
    
    print("\n2. Testing value propagation...")
    test_value_propagation()
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main() 