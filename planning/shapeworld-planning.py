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
from rllib.mdp import MarkovDecisionProcess, MDPPolicy, QLearner
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

# Specify the goal state for this simulation
shape1 = Shape(sides='circle', shade='medium', texture='present')
shape2 = Shape(sides='square', shade='low', texture='present')
shape3 = Shape(sides='square', shade='low', texture='present')
goal = State(shape1, shape1, shape1)

# Create the shape world environment
shape_world = ShapeWorld(goal=goal, discount_rate=0.8) # MDP object
shape_env = GymWrapper(shape_world) # uses MDP object for simulation object

##################################################
# LEARNER SPECIFICATION
##################################################

qlearner = QLearner(
    discount_rate = shape_world.discount_rate,
    learning_rate = 0.1,
    initial_value= 0, # default value for all states
    epsilon=0.2, # uses epsilon greedy
    action_space=shape_world.action_space
)

##################################################
# SIMULATION SPECIFICATION
##################################################

results = simulation_loop(
    env = shape_env,
    policy = qlearner,
    n_episodes=100,
    max_steps=1000,
    seed = None
)