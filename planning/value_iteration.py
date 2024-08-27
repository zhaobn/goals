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
from rllib.mdp import MarkovDecisionProcess, MDPPolicy, QLearner, ValueIteration
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
from tqdm import tqdm
#import os

# make sure working directory is as expected
#print(os.getcwd())

# Plotting libraries
import matplotlib.pyplot as plt
from matplotlib import patheffects
import sys

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

# Define the needed shape objects
Shape = namedtuple('Shape',['sides', 'shade', 'texture'])
State = namedtuple('State',['shape1', 'shape2', 'shape3'])
Action = namedtuple('Action',['actor','recipient'])

##################################################
# VALUE ITERATION FOR A SINGLE GOAL
##################################################

# Create shapeworld object for indexing goal state
discount_rate = 0.5
temporary_goal_state = State(
    Shape('circle', 'low', 'present'),
    Shape('circle', 'low', 'present'),
    Shape('circle', 'low', 'present')
)
temp_env = ShapeWorld(temporary_goal_state, discount_rate)
state_space = temp_env.get_state_space()

# Use command line arg to select goal state
goal_index = int(sys.argv[1])
goal_state = state_space[goal_index]

# Create a new environment with the goal state
env = ShapeWorld(goal_state, discount_rate)

# Value Iteration
value_it = ValueIteration(mdp=env, initial_value=0, threshold=1e-6, verbose=True)
value_it.value_iteration()
value_function = value_it.get_value_function()

# Save the converged value function
filename = f'./value-iteration-results/value_function_goal_{goal_index}.pkl'
pd.to_pickle((value_function, goal_state), filename) # note the ordered pair formatting
print(f"Value function saved as {filename}")

##################################################