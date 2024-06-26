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
from rllib.gymwrap import GymWrapper
from rllib.simulation_result import TDLearningSimulationResult
# function for running an RL simulation
from rllib.tools import simulation_loop

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
import gymnasium as gym
from tqdm import tqdm

# Plotting libraries
import matplotlib.pyplot as plt
from matplotlib import patheffects

# Define the needed shape objects
Shape = namedtuple('Shape',['sides', 'shade', 'texture'])
State = namedtuple('State',['shape1', 'shape2', 'shape3'])
Action = namedtuple('Action',['actor','recipient'])

##################################################
# ENVIRONMENT SPECIFICATION
##################################################
'''
# Specify the goal state for this simulation
shape1 = Shape(sides='circle', shade='medium', texture='present')
shape2 = Shape(sides='square', shade='low', texture='present')
shape3 = Shape(sides='square', shade='low', texture='present')
goal = State(shape1, shape1, shape1)

# Create the shape world environment
shape_world = ShapeWorld(goal=goal, discount_rate=0.8) # MDP object
shape_env = GymWrapper(shape_world) # uses MDP object for simulation object
'''
##################################################
# LEARNER SPECIFICATION
##################################################
'''
qlearner = QLearner(
    discount_rate = shape_world.discount_rate,
    learning_rate = 0.1,
    initial_value= 0, # default value for all states
    epsilon=0.2, # uses epsilon greedy
    action_space=shape_world.action_space
)
'''
##################################################
# SIMULATION SPECIFICATION
##################################################
'''
results = simulation_loop(
    env = shape_env,
    policy = qlearner,
    n_episodes=100,
    max_steps=1000,
    seed = None
)
'''
##################################################
# VALUE ITERATION FOR A SINGLE GOAL
##################################################
'''
# Define the ShapeWorld MDP (assume it's correctly implemented and imported)
discount_rate = 0.5
goal_state = State(
    Shape('circle', 'low', 'present'),
    Shape('circle', 'low', 'present'),
    Shape('circle', 'low', 'present')
    #Shape('square', 'medium', 'not_present'),
    #Shape('triangle', 'high', 'present')
)
env = ShapeWorld(goal_state, discount_rate)

# Initialize the Bellman updater
bellman_updater = BellmanUpdater(mdp = env, 
                                 initial_value=0,
                                 threshold=1e-6)

# Run value iteration
bellman_updater.value_iteration()

# Retrieve the value of a specific state
state_value = bellman_updater.get_value(goal_state)
print(f"Value of the goal state: {state_value}")
print(f"Converged in {bellman_updater.iterations} iterations")
'''
##################################################
# VALUE ITERATION FOR ALL GOALS
##################################################

# create an environment so we can get the state space
discount_rate = 0.5
goal_state = State(
    Shape('circle', 'low', 'present'),
    Shape('circle', 'low', 'present'),
    Shape('circle', 'low', 'present')
    #Shape('square', 'medium', 'not_present'),
    #Shape('triangle', 'high', 'present')
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
filename = '/Users/jbbyers/Research/Goals as Concepts/goals/value_functions.pkl'
pd.to_pickle(value_functions, filename)
print(f"Value functions saved as {filename}")

# Calculate average value for each goal state
goal_value_function = {}
for goal_state, value_function in value_functions.items():
    average_value = np.mean(list(value_function.values()))
    goal_value_function[goal_state] = average_value

# Save the goal value function as a pickle file
goal_value_function_filename = '/Users/jbbyers/Research/Goals as Concepts/goals/goal_value_function.pkl'
pd.to_pickle(goal_value_function, goal_value_function_filename)
print(f"Goal value function saved as {goal_value_function_filename}")


##################################################