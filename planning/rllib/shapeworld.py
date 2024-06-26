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
    GOAL_REWARD = 100 # we let this be zero because of averaging across Q-tables we do later
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
        '''
        Given a state and action, return a possible next state.
        '''
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
        new_state = State(*new_state_list)

        return new_state
    
    def get_possible_next_states(self, s: State, a: Action) -> Sequence[State]:
        """Return the possible next states given a state and action."""
        
        # either the recipient shape takes on parent shape or doesn't
        possible_sides = [s[a.actor].sides, s[a.recipient].sides]
        # recipient becomes lighter, darker, or same
        possible_shades = [
            s[a.recipient].shade,
            self._get_lighter_shade(s[a.recipient].shade),
            self._get_darker_shade(s[a.recipient].shade)]
        # texture always changes
        possible_texture = ['not_present' if s[a.recipient].texture == 'present' else 'present']
        possible_states = set()

        # create all possible next states by replacing recipient state
        for side in possible_sides:
            for shade in possible_shades:
                for texture in possible_texture:
                    new_shape = Shape(side, shade, texture)
                    new_state_list = [s[0], s[1], s[2]] # shapes of current state
                    new_state_list[a.recipient] = new_shape
                    new_state = State(*new_state_list) # unpack new state list
                    possible_states.add(new_state) # using a set ensures uniqueness

        return list(possible_states)
    
    def transition_probability(self, s: State, a: Action, ns: State) -> float:
        """Return the transition probability from state s to state ns given action a."""
        # determine if recipient shape changes
        if ns[a.recipient].sides != s[a.actor].sides:
            return 1 - self.SHAPE_TRANSITION_PROB
        else:
            # determine if the recipient shade changes
            if self._is_darker(s[a.actor].shade, s[a.recipient].shade) and ns[a.recipient].shade == self._get_darker_shade(s[a.recipient].shade):
                return self.SHAPE_TRANSITION_PROB
            elif self._is_lighter(s[a.actor].shade, s[a.recipient].shade) and ns[a.recipient].shade == self._get_lighter_shade(s[a.recipient].shade):
                return self.SHAPE_TRANSITION_PROB
            elif s[a.recipient].shade == ns[a.recipient].shade:
                # determine recipient texture change
                if s[a.recipient].texture == 'present' and ns[a.recipient].texture == 'not_present':
                    return self.SHAPE_TRANSITION_PROB
                elif s[a.recipient].texture == 'not_present' and ns[a.recipient].texture == 'present':
                    return self.SHAPE_TRANSITION_PROB
                elif s[a.recipient].texture == ns[a.recipient].texture:
                    return 1
            else:
                return 0
    
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
        # TODO: I should implement some form of visualization to make sure that everything is working as expected.
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
        # if shade cannot go lower
        if current_idx == len(self.SHADE_LIST) - 1:
            return shade
        else:
            return self.SHADE_LIST[current_idx + 1]
    def _get_lighter_shade(self, shade):
        current_idx = self.SHADE_LIST.index(shade)
        # if shade cannot go lower
        if current_idx - 1 < 0:
            return shade
        else:
            return self.SHADE_LIST[current_idx - 1]
        