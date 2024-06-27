import os
import pandas as pd
import numpy as np

# Assuming the same State namedtuple definition is available
# from collections import namedtuple
# Shape = namedtuple('Shape',['sides', 'shade', 'texture'])
# State = namedtuple('State',['shape1', 'shape2', 'shape3'])

def load_value_functions(directory):
    """
    Load all value function files from the specified directory.
    """
    value_functions = {}
    for filename in os.listdir(directory):
        if filename.endswith('.pkl'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'rb') as file:
                value_function, goal_state = pd.read_pickle(file)
                value_functions[goal_state] = value_function
    return value_functions

def calculate_goal_value_functions(value_functions):
    """
    Calculate the average value for each goal state across all state values under that goal.
    """
    goal_value_function = {}
    for goal_state, value_function in value_functions.items():
        average_value = np.mean(list(value_function.values()))
        goal_value_function[goal_state] = average_value
    return goal_value_function

# Load value functions from the directory where they are saved
directory = './value-iteration-results'
value_functions = load_value_functions(directory)

# Calculate the goal value function
goal_value_function = calculate_goal_value_functions(value_functions)

# Optionally, save or print the goal value function
print(goal_value_function)