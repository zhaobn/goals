import random
from typing import Union, Any
import numpy as np
import gymnasium as gym
from .mdp import MarkovDecisionProcess
from typing import Sequence, Hashable, TypeVar, Generic, Container, Any, Union, Counter
from collections import namedtuple

ObsType = Union[int, np.ndarray, dict[str, Any]]
ActType = Union[int, np.ndarray, dict[str, Any]]

# temporarily here for setting state transitions
Shape = namedtuple('Shape',['sides', 'shade', 'texture'])
State = namedtuple('State',['shape1', 'shape2', 'shape3'])
Action = namedtuple('Action',['actor','recipient'])

class GymWrapper(gym.Env):
    '''
    A convenient class for creating an environment for performing simulations. 
    '''
    def __init__(self, mdp : MarkovDecisionProcess):
        self.mdp = mdp
        self.current_state = None
        self.rng : random.Random = random
        self.action_space = gym.spaces.Discrete(len(mdp.action_space))
        assert not isinstance(next(iter(mdp.action_space)), int), \
            "To avoid ambiguity with gym action encoding, action space must not be an integer"
        self.observation_space = gym.spaces.Discrete(len(mdp.state_space))
        self.reward_range = (float('-inf'), float('inf'))

    def step(self, action : ActType) -> tuple[ObsType, float, bool, dict[str, Any]]:
        '''
        Returns: 
            obs, reward, terminated, info
        '''
        assert isinstance(action, int), 'Input is the action index'
        action = self.mdp.action_space[action]
        next_state = self.mdp.next_state_sample(self.current_state, action, rng=self.rng)
        next_state_idx = self.mdp.state_space.index(next_state)
        reward = self.mdp.reward(self.current_state, action, next_state)
        terminated = self.mdp.is_absorbing(next_state)
        self.current_state = next_state
        info = dict(
            state=self.current_state,
            action=action,
            next_state=next_state,
            reward=reward
        )
        return next_state_idx, reward, terminated, info
    
    def reset(self, *, seed : int = None, options : dict[str, Any] = None) -> tuple[ObsType, dict[str, Any]]:
        '''
        Resets the starting state for when starting another episode.
        '''
        super().reset(seed=seed)
        self.rng = random.Random(seed)
        # TODO: fix this to where the starts really are random
        '''
        shape1 = Shape(sides='circle', shade='medium', texture='present')
        shape2 = Shape(sides='square', shade='low', texture='present')
        shape3 = Shape(sides='triangle', shade='high', texture='present')
        '''
        
        # using this starting state to make sure that it's easy to achieve the goal
        shape1 = Shape(sides='circle', shade='medium', texture='not_present')

        s0 = State(shape1, shape1, shape1)
        self.current_state = s0 #self.mdp.initial_state_dist().sample(rng=self.rng)
        return self.mdp.state_space.index(self.current_state), dict(state=self.current_state)
    
    def render(self) -> None:
        raise NotImplementedError