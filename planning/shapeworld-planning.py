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

# Custom class imports
from rllib.shapeworld import ShapeWorld
from rllib.tools import isclose
from rllib.mdp import MarkovDecisionProcess, MDPPolicy, QLearner, BellmanUpdater
# from rllib.distributions import DiscreteDistribution
# from rllib.gymwrap import GymWrapper
#from rllib.simulation_result import TDLearningSimulationResult
# function for running an RL simulation
#from rllib.tools import simulation_loop

# Other libraries
import numpy as np
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass
from typing import Sequence, Hashable, TypeVar, Generic, Container, Any, Union, Counter
from collections import namedtuple
from itertools import product
import random
from random import Random
from collections import defaultdict, Counter
# import gymnasium as gym
from tqdm import tqdm

# Plotting libraries
import matplotlib.pyplot as plt
from matplotlib import patheffects

# Define the needed shape objects
Shape = namedtuple('Shape',['sides', 'shade', 'texture'])
State = namedtuple('State',['shape1', 'shape2', 'shape3'])
Action = namedtuple('Action',['actor','recipient'])

##################################################
# VALUE ITERATION FOR ALL GOALS
##################################################

# create an environment so we can get the state space
discount_rate = 0.5
goal_state = State(
    Shape('circle', 'low', 'striped'),
    Shape('circle', 'low', 'striped'),
    Shape('circle', 'low', 'striped')
)
env = ShapeWorld(goal_state, discount_rate)

# Loop through every state in the ShapeWorld object env
all_states = env.get_state_space()
value_functions = {}  # Dictionary to store the value functions
discount_rate = 0.5

for state in tqdm(all_states):
    # Create a new environment with the current state as the goal state
    goal_state = state
    env = ShapeWorld(goal_state, discount_rate)
    
    # Initialize the Bellman updater for the new environment
    bellman_updater = BellmanUpdater(mdp=env, initial_value=0, threshold=1e-6, verbose=False)
    
    # Run value iteration
    bellman_updater.value_iteration()
    
    # Retrieve the value of the goal state
    value_function = bellman_updater.get_value(goal_state)
    
    # Store the value function in the dictionary
    value_functions[state] = value_function

# Save the value functions as a file
filename = '/jukebox/niv/branson/goals/planning/value_functions.pkl'
pd.to_pickle(value_functions, filename)
print(f"Value functions saved as {filename}")

# Calculate average value for each goal state
goal_value_function = {}
for goal_state, value_function in value_functions.items():
    average_value = np.mean(list(value_function.values()))
    goal_value_function[goal_state] = average_value

# Save the goal value function as a pickle file
goal_value_function_filename = '/jukebox/niv/branson/goals/planning/goal_value_function.pkl'
pd.to_pickle(goal_value_function, goal_value_function_filename)
print(f"Goal value function saved as {goal_value_function_filename}")


##################################################