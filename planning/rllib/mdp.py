from typing import Sequence, Hashable, TypeVar, Generic, Container
import random
from random import Random
from collections import defaultdict, Counter
from tqdm import tqdm

# from .distributions import Distribution, DiscreteDistribution, Uniform, Gaussian

# We want actions to be hashable so that we can use them as dictionary keys for Q-learning later
Action = TypeVar('Action', bound=Hashable)
State = TypeVar('State', bound=Hashable)

class MarkovDecisionProcess(Generic[State, Action]):
    '''A general class for many MDPs.'''
    discount_rate : float
    action_space : Sequence[Action]
    state_space : Sequence[State]
    state_action_space : Sequence['tuple[State, Action]']

    #TODO: change function name to get_actions()
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
    
class MDPPolicy(Generic[State, Action]):
    '''
    A very general class for an MDP policy.
    '''
    DISCOUNT_RATE : float

    def action_sample(self, s : State, rng : Random = random) -> Action:
        '''Sample which actions to take.'''
        raise NotImplementedError
    
    def state_value(self, s : State) -> float:
        '''Retrieve the value of a state: s.'''
        raise NotImplementedError
    
    def update(self, s : State, a : Action, r : float, ns : State) -> None:
        '''Update the value of a state action pair.'''
        raise NotImplementedError
    
    def end_episode(self) -> None:
        raise NotImplementedError
    
    def reset(self) -> None:
        raise NotImplementedError

class BaseLearner(MDPPolicy):
    def __init__(self, discount_rate, learning_rate, initial_value, epsilon, action_space):
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        self.initial_value = initial_value
        self.epsilon = epsilon
        self.action_space = action_space

    def reset(self):
        '''
        Using default dict allows us to set a default value for 
        items that do not have a value assigned to them yet. 
        Useful for updating values for state-action pairs for states
        that have not been encountered yet. 
        '''
        self.estimated_state_action_values = defaultdict(
            lambda : {a: self.initial_value for a in self.action_space}
        )

    def state_value(self, s) -> float:
        return max(self.estimated_state_action_values[s].values())
    
    def action_sample(self, s, rng = random):
        # e-greedy action selection
        if rng.random() < self.epsilon:
            return rng.choice(self.action_space)
        astar = max(self.estimated_state_action_values[s].items(),
                          key = lambda x: x[1])[0]
        return astar
    
    def end_episode(self) -> None:
        raise NotImplementedError

class QLearner(BaseLearner):
    def end_episode(self):
        pass

    def update(self, s, a, r, ns):
        # TODO: will this ever update the actual goal state value?
        max_ns_q = max(self.estimated_state_action_values[ns].values())
        td_target = r + self.discount_rate * max_ns_q
        td_error = td_target - self.estimated_state_action_values[s][a]
        self.estimated_state_action_values[s][a] += self.learning_rate * td_error

class BellmanUpdater(Generic[State, Action]):
    def __init__(self, mdp: MarkovDecisionProcess[State, Action], 
                 initial_value: float = 0.0,
                 threshold: float = 1e-6,
                 verbose: bool = True):
        self.mdp = mdp
        self.threshold = threshold
        self.value_function = defaultdict(lambda: initial_value)
        self.iterations = 0
        self.delta = float('inf')
        self.verbose = verbose
        self.initial_value = initial_value
    
    def value_iteration(self):
        while self.delta > self.threshold:
            self.delta = 0
            new_value_function = defaultdict(float)

            for s in self.mdp.state_space:
                if self.mdp.is_absorbing(s):
                    new_value_function[s] = 0.0
                else:
                    q_values = []

                    for a in self.mdp.actions(s):
                        q_value = 0.0
                        for ns in self.mdp.get_possible_next_states(s, a):
                            reward = self.mdp.reward(s, a, ns)
                            prob = self.mdp.transition_probability(s, a, ns)
                            q_value += prob * (reward + self.mdp.discount_rate * self.value_function[ns])
                        q_values.append(q_value)

                    new_value_function[s] = max(q_values)
                    self.delta = max(self.delta, abs(new_value_function[s] - self.value_function[s]))

            self.value_function = new_value_function
            self.iterations += 1
            if self.verbose:
                print(f"Iteration {self.iterations}, Delta: {self.delta}")

    def get_value(self, s: State) -> float:
        return self.value_function.get(s, 0.0)
    
    def get_value_function(self) -> dict:
        return dict(self.value_function)
    
    def reset(self):
        self.value_function = defaultdict(lambda: self.initial_value)
        self.iterations = 0
        self.delta = float('inf')

    def has_converged(self) -> bool:
        return self.iterations > 0 and self.delta <= self.threshold
