from collections import namedtuple, defaultdict
from typing import Sequence, Tuple, Dict
from itertools import product
import random

import numpy as np

from .mdp import MarkovDecisionProcess
# from .distributions import Distribution, DiscreteDistribution, Uniform

Shape = namedtuple('Shape',['sides', 'shade', 'texture'])
State = namedtuple('State',['shape1', 'shape2', 'shape3'])
Action = namedtuple('Action',['actor','recipient'])

# The action space. a = actor, r = recipient
a1r2 = Action(0, 1)
a1r3 = Action(0, 2)
a2r1 = Action(1, 0)
a2r3 = Action(1, 2)
a3r1 = Action(2, 0)
a3r2 = Action(2, 1)

class ShapeWorld(MarkovDecisionProcess[State, Action]):
    GOAL = None
    GOAL_REWARD = 0 # we let this be zero because of averaging across Q-tables we do later
    STEP_COST = -1
    SHAPE_LIST = tuple(['circle','square','triangle'])
    SHADE_LIST = tuple(['low','medium','high'])
    TEXTURE_LIST = tuple(['present','not_present'])
    SHAPE_TRANSITION_PROB = 0.8

    def __init__(self, goal : State, discount_rate):
        self.discount_rate = discount_rate
        self.GOAL = goal

        # set up shapeworld shape space
        shape_space : Sequence[Shape] = tuple(
            Shape(sides, shade, texture) 
                for (sides, shade, texture) 
                    in product(self.SHAPE_LIST,self.SHADE_LIST,self.TEXTURE_LIST)
        )

        # set up state space
        self.state_space : Sequence[State] = tuple(
            State(shape1, shape2, shape3) for(shape1, shape2, shape3) in product(shape_space, shape_space, shape_space)
        )

        # set up state action space
        self.action_space = tuple([a1r2, a1r3, a2r1, a2r3, a3r1, a3r2])
        self.state_action_space = tuple(product(self.state_space, self.action_space))

    def actions(self, s : State) -> Sequence[Action]:
        '''Return the action space.'''
        return self.action_space
    
    def next_state_sample(
            self,
            s : State,
            a : Action,
            rng : random.Random = random
    ) -> State:
        # determine if recipient shape changes
        if rng.random() < self.SHAPE_TRANSITION_PROB:
            sides = s[a.actor].sides
        else:
            sides = s[a.recipient].sides
        
        # determine if the recipient shade changes
        shade = None

        if self._is_darker(s[a.actor].shade, s[a.recipient].shade):
            shade = self._get_darker_shade(s[a.recipient].shade)

        elif self._is_lighter(s[a.actor].shade, s[a.recipient].shade):
            shade = self._get_lighter_shade(s[a.recipient].shade)

        else:
            shade = s[a.recipient].shade

        # determine recipient texture change
        texture = 'not_present' if s[a.recipient].texture == 'present' else 'present'

        # instantiate next state
        new_state_list  = [s[0], s[1], s[2]]
        new_state_list[a.recipient] = Shape(sides, shade, texture)
        new_state = tuple(new_state_list)

        return new_state
    
    def reward(self, s: State, a : Action, ns : State) -> float:
        reward = self.STEP_COST
        if self._is_goal(ns):
            reward += self.GOAL_REWARD
        return reward

    def is_absorbing(self, s: State) -> bool:
        '''
        Check to see if state is absorbing. 

        For this conetext, the goal state is our absorbing state, and the simulation should be terminated if reached.

        Returns:
            bool: Whether the state is an absorbing state or not.
        '''
        is_absorbing = np.nan
        if self.GOAL == s:
            is_absorbing = True
        else:
            is_absorbing = False
        
        return is_absorbing

    def plot(self, ax=None):
        raise NotImplementedError
    
    # helper functions
    def _is_goal(self, ns : State):
        return self.GOAL == ns

    def _is_darker(self, shade1, shade2):
        return self.SHADE_LIST.index(shade1) > self.SHADE_LIST.index(shade2)
    
    def _is_lighter(self, shade1, shade2):
        return self.SHADE_LIST.index(shade1) < self.SHADE_LIST.index(shade2)
    
    def _get_darker_shade(self, shade):
        current_idx = self.SHADE_LIST.index(shade)
        return self.SHADE_LIST[current_idx + 1] if current_idx > 0 else shade
    
    def _get_lighter_shade(self, shade):
        current_idx = self.SHADE_LIST.index(shade)
        return self.SHADE_LIST[current_idx - 1] if current_idx < len(self.SHADE_LIST) else shade