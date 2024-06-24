from typing import Sequence, Hashable, TypeVar, Generic, Container
import random
from random import Random

# from .distributions import Distribution, DiscreteDistribution, Uniform, Gaussian

# We want actions to be hashable so that we can use them as dictionary keys for Q-learning later
Action = TypeVar('Action', bound=Hashable)
State = TypeVar('State', bound=Hashable)

class MarkovDecisionProcess(Generic[State, Action]):
    '''A general class for many MDPs.'''
    discount_rate : float
    action_space : Sequence[Action]
    state_space : Sequence[State]
    state_action_space : Sequence[tuple[State, Action]]

    def actions(self, s : State) -> Sequence[Action]:
        '''Return the action space.'''
        return self.action_space
    
    def next_state_sample(self, s : State, a : Action, rng : Random = random) -> State:
        '''Sample the next state.'''
        raise NotImplementedError
    
    def reward(self, s : State, a : Action, ns : State) -> float:
        raise NotImplementedError
    
    def is_absorbing(self, s : State) -> bool:
        raise NotImplementedError