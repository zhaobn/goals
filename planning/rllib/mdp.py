from typing import Sequence, Hashable, TypeVar, Generic, Container
import random
import numpy as np
from random import Random, sample
from collections import defaultdict, Counter
from tqdm import tqdm
from typing import TypeVar
from math import log
import pandas as pd

# from .distributions import Distribution, DiscreteDistribution, Uniform, Gaussian

# We want actions to be hashable so that we can use them as dictionary keys for Q-learning later
Action = TypeVar('Action', bound=Hashable)
State = TypeVar('State', bound=Hashable)
Likelihood = TypeVar('Likelihood', float, int)

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

    def sample_action(self, s : State, rng : Random = random) -> tuple[Action, Likelihood]:
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
    
    def sample_action(self, s, rng = random):
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

class ValueIteration(Generic[State, Action]):
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

class GoalSelectionPolicy(Generic[State, Action]):
    def __init__(self, mdp: MarkovDecisionProcess[State, Action]):
        '''Initialize the policy with the MDP.'''
        self.mdp = mdp
        self.state_space = mdp.state_space

    def sample_action(self, s : State, rng : Random = random) -> tuple[Action, Likelihood]:
        '''Uses some policy to select which goal to select.'''
        raise NotImplementedError
    
    def reset(self):
        raise NotImplementedError

class OptimalGoalPolicy(GoalSelectionPolicy[State, Action]):
    '''Takes the optimal value function computed from value iteration to select goals.'''

    def __init__(self, mdp: MarkovDecisionProcess[State, Action], value_function: dict[State, float]):
        super().__init__(mdp)
        # we will use the value function from value iteration to select the goal
        self.value_function = value_function

    def sample_action(self, rng: random.Random = random) -> tuple[State, float]:
        '''Use softmax to sample a goal state based on the value function.'''
        # Assuming 'actions' here are actually states for goal selection
        states = list(self.value_function.keys())
        state_values = np.array([self.value_function[state] for state in states])

        # Compute softmax probabilities
        exp_values = np.exp(state_values - np.max(state_values))  # for numerical stability
        probabilities = exp_values / np.sum(exp_values)

        # Sample a state based on these probabilities
        selected_state = rng.choices(states, weights=probabilities)[0]

        # Calculate the negative log likelihood of the selected state
        nll = -np.log(probabilities[states.index(selected_state)])

        return (selected_state, nll)
    
    def calc_log_lik(self, state: State) -> float:
        '''Calculate the log likelihood of selecting a particular state.'''
        state_values = np.array([self.value_function[s] for s in self.state_space])
        exp_values = np.exp(state_values - np.max(state_values))
        probabilities = exp_values / np.sum(exp_values)
        return -np.log(probabilities[self.state_space.index(state)])
    
    def calc_log_likelihood_all(self) -> dict[State, float]:
        '''Calculate the log likelihood of all states in the state space.'''
        # get all the value for the states and convert them to probabilities
        state_values = np.array([self.value_function[state] for state in self.state_space])
        exp_values = np.exp(state_values - np.max(state_values))
        probabilities = exp_values / np.sum(exp_values)
        # print(f"Length of probabilities: {len(probabilities)}")
        log_likelihoods = {s: -np.log(probabilities[self.state_space.index(s)]) for s in self.state_space}
        return log_likelihoods
    
    def calc_log_likelihood_all_df(self) -> pd.DataFrame:
        '''Return log lik as a dataframe.'''
        log_likelihood = self.calc_log_likelihood_all()
        return pd.DataFrame(log_likelihood.items(), columns=['State', 'Log Likelihood'])
    
    def reset(self):
        pass

class RandomGoalPolicy(GoalSelectionPolicy[State, Action]):
    
    def __init__(self, mdp: MarkovDecisionProcess[State, Action]):
        super().__init__(mdp)
    
    def select_goal(self, s: State) -> State:
        return random.choice(self.mdp.state_space)
    
class PCFGGoalPolicy(GoalSelectionPolicy[State, Action]):

    def __init__(self, mdp: MarkovDecisionProcess[State, Action], p_rules, cap=10):
        super().__init__(mdp)
        self.NON_TERMINALS = [x[0] for x in p_rules]
        self.PRODUCTIONS = {}
        self.CAP = cap
        for rule in p_rules:
            self.PRODUCTIONS[rule[0]] = rule[1]

    def generate_tree(self, logging=True, tree_str='S', log_prob=0., depth=0):
        current_nt_indices = [tree_str.find(nt) for nt in self.NON_TERMINALS]
        # Sample a non-terminal for generation
        to_gen_idx = sample([idx for idx, el in enumerate(current_nt_indices) if el > -1], 1)[0]
        to_gen_nt = self.NON_TERMINALS[to_gen_idx]
        # Do generation
        leaf = sample(self.PRODUCTIONS[to_gen_nt], 1)[0]
        to_gen_tree_idx = tree_str.find(to_gen_nt)
        tree_str = tree_str[:to_gen_tree_idx] + leaf + tree_str[(to_gen_tree_idx+1):]
        # Update production log prob
        log_prob += log(1/len(self.PRODUCTIONS[to_gen_nt]))
        # Increase depth count
        depth += 1

        # Recursively rewrite string
        if any (nt in tree_str for nt in self.NON_TERMINALS) and depth <= self.CAP:
            return self.generate_tree(logging, tree_str, log_prob, depth)
        elif any (nt in tree_str for nt in self.NON_TERMINALS):
            if logging:
                print('====DEPTH EXCEEDED!====')
            return None
        else:
            if logging:
                print(tree_str, log_prob)
            return tree_str, log_prob
        
    