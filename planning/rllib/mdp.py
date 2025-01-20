from typing import Sequence, Hashable, TypeVar, Generic
from random import Random, sample
import random
import numpy as np
from collections import defaultdict
import pandas as pd
from dataclasses import dataclass
from typing import Literal
from math import log
from tqdm import tqdm

# Define generic type variables for any state/action types
S = TypeVar('S', bound=Hashable)  # Generic State type
A = TypeVar('A', bound=Hashable)  # Generic Action type

class MarkovDecisionProcess(Generic[S, A]):
    """Base class for Markov Decision Processes.
    
    Attributes:
        discount_rate: Discount factor for future rewards (0 to 1)
        action_space: Sequence of all possible actions
        state_space: Sequence of all possible states
    """
    discount_rate: float
    action_space: Sequence[A]
    state_space: Sequence[S]
    
    def get_state_space(self) -> Sequence[S]:
        '''Return the state space.'''
        return self.state_space
    
    def get_actions(self, s: S) -> Sequence[A]:
        '''Return the available actions in state s.'''
        return self.action_space
    
    def next_state_sample(self, s: S, a: A, rng: Random = random) -> S:
        '''Sample the next state.'''
        raise NotImplementedError
    
    def reward(self, s: S, a: A, ns: S) -> float:
        '''Calculate reward for transitioning from s to ns with action a.'''
        raise NotImplementedError
    
    def is_absorbing(self, s: S) -> bool:
        '''Check if state s is absorbing.'''
        raise NotImplementedError
    
    def get_possible_next_states(self, s: S, a: A) -> Sequence[S]:
        '''Return all possible next states from state s taking action a.'''
        raise NotImplementedError
    
    def transition_probability(self, s: S, a: A, ns: S) -> float:
        '''Return probability of transitioning from s to ns with action a.'''
        raise NotImplementedError

class MDPPolicy(Generic[S, A]):
    '''A very general class for an MDP policy.'''
    DISCOUNT_RATE: float

    def sample_action(self, s: S, rng: Random = random) -> tuple[A, float]:
        '''Sample which actions to take.'''
        raise NotImplementedError
    
    def state_value(self, s: S) -> float:
        '''Retrieve the value of a state: s.'''
        raise NotImplementedError
    
    def update(self, s: S, a: A, r: float, ns: S) -> None:
        '''Update the value of a state action pair.'''
        raise NotImplementedError
    
    def end_episode(self) -> None:
        raise NotImplementedError
    
    def reset(self) -> None:
        raise NotImplementedError

class BaseLearner(MDPPolicy[S, A]):
    def __init__(self, discount_rate: float, 
                 learning_rate: float,
                 initial_value: float,
                 epsilon: float,
                 action_space: Sequence[A]):
        # Add type hints
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        self.initial_value = initial_value
        self.epsilon = epsilon
        self.action_space = action_space
        self.reset()  # Initialize values

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
    
    def sample_action(self, s: S, rng: Random = random) -> tuple[A, float]:
        # Fix return type to match parent class
        if rng.random() < self.epsilon:
            action = rng.choice(self.action_space)
            return action, -np.log(1.0/len(self.action_space))  # Return likelihood
        
        astar = max(self.estimated_state_action_values[s].items(),
                   key=lambda x: x[1])[0]
        return astar, 0.0  # Deterministic choice has 0 negative log likelihood
    
    def end_episode(self) -> None:
        raise NotImplementedError

class QLearner(BaseLearner[S, A]):
    def end_episode(self):
        pass

    def update(self, s, a, r, ns):
        # TODO: will this ever update the actual goal state value?
        max_ns_q = max(self.estimated_state_action_values[ns].values())
        td_target = r + self.discount_rate * max_ns_q
        td_error = td_target - self.estimated_state_action_values[s][a]
        self.estimated_state_action_values[s][a] += self.learning_rate * td_error

class ValueIteration(Generic[S, A]):
    """Value Iteration algorithm for solving MDPs."""
    
    def __init__(self, mdp: MarkovDecisionProcess[S, A], 
                 initial_value: float = 0.0,
                 threshold: float = 1e-6,
                 verbose: bool = True,
                 max_iterations: int = 1000):  # Add maximum iterations
        """Initialize Value Iteration solver."""
        if not isinstance(mdp, MarkovDecisionProcess):
            raise TypeError("mdp must be an instance of MarkovDecisionProcess")
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        self.mdp = mdp
        self.threshold = threshold
        self.value_function = defaultdict(lambda: initial_value)
        self.iterations = 0
        self.delta = float('inf')
        self.verbose = verbose
        self.initial_value = initial_value
        self.max_iterations = max_iterations
        
        # Cache state and action spaces to avoid repeated calls
        self.states = list(mdp.get_state_space())
        self.action_map = {s: list(mdp.get_actions(s)) for s in self.states}
        
        # Pre-compute transition dynamics where possible
        self.transitions = {}
        for s in self.states:
            self.transitions[s] = {}
            for a in self.action_map[s]:
                next_states = mdp.get_possible_next_states(s, a)
                self.transitions[s][a] = [
                    (ns, mdp.reward(s, a, ns), mdp.transition_probability(s, a, ns))
                    for ns in next_states
                ]
    
    def value_iteration(self):
        """Run the value iteration algorithm until convergence."""
        pbar = tqdm(total=self.max_iterations, desc="Value Iteration")
        
        while self.delta > self.threshold and self.iterations < self.max_iterations:
            self.delta = 0
            new_values = {}

            # Update values
            for s in self.states:
                if self.mdp.is_absorbing(s):
                    new_values[s] = 0.0
                    continue

                # Calculate Q-values for all actions at once
                q_values = []
                for a in self.action_map[s]:
                    q_value = sum(
                        prob * (r + self.mdp.discount_rate * self.value_function[ns])
                        for ns, r, prob in self.transitions[s][a]
                    )
                    q_values.append(q_value)

                if q_values:
                    new_value = max(q_values)
                    new_values[s] = new_value
                    self.delta = max(self.delta, abs(new_value - self.value_function[s]))
                else:
                    new_values[s] = self.value_function[s]

            # Batch update value function
            self.value_function = new_values
            self.iterations += 1
            
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({'delta': f'{self.delta:.6f}'})
            
            if self.verbose and self.iterations % 10 == 0:
                print(f"Iteration {self.iterations}, Delta: {self.delta:.6f}")

        pbar.close()
        if self.iterations >= self.max_iterations:
            print("Warning: Value iteration reached maximum iterations without converging")

    def get_optimal_policy(self) -> dict[S, A]:
        """Return the optimal policy based on the computed value function.
        
        Returns:
            Dictionary mapping states to optimal actions
        """
        policy = {}
        for s in self.states:
            if self.mdp.is_absorbing(s):
                continue
                
            best_value = float('-inf')
            best_action = None
            
            for a in self.action_map[s]:
                value = sum(
                    prob * (r + self.mdp.discount_rate * self.value_function[ns])
                    for ns, r, prob in self.transitions[s][a]
                )
                if value > best_value:
                    best_value = value
                    best_action = a
                    
            if best_action is not None:
                policy[s] = best_action
                
        return policy

    def get_value(self, s: S) -> float:
        """Get the value of a specific state."""
        return self.value_function.get(s, 0.0)
    
    def get_value_function(self) -> dict[S, float]:
        """Get the complete value function."""
        return dict(self.value_function)
    
    def has_converged(self) -> bool:
        """Check if value iteration has converged."""
        return self.iterations > 0 and self.delta <= self.threshold
    
    def reset(self):
        """Reset the value iteration to initial state."""
        self.value_function = defaultdict(lambda: self.initial_value)
        self.iterations = 0
        self.delta = float('inf')
        # Don't reset cached transitions since they remain valid

class GoalSelectionPolicy(Generic[S, A]):
    def __init__(self, mdp: MarkovDecisionProcess[S, A]):
        '''Initialize the policy with the MDP.'''
        self.mdp = mdp
        self.state_space = mdp.state_space

    def sample_action(self, s : S, rng : Random = random) -> tuple[A, float]:
        '''Uses some policy to select which goal to select.'''
        raise NotImplementedError
    
    def reset(self):
        raise NotImplementedError

class OptimalGoalPolicy(GoalSelectionPolicy[S, A]):
    '''Takes the optimal value function computed from value iteration to select goals.'''

    def __init__(self, mdp: MarkovDecisionProcess[S, A], 
                 value_function: dict[S, float],
                 temperature: float = 1.0):  # Add temperature parameter
        super().__init__(mdp)
        self.value_function = value_function
        self.temperature = temperature
        
        # Cache state values and compute them once
        self.states = list(value_function.keys())
        self.state_values = np.array([value_function[s] for s in self.states])
        
    def sample_action(self, rng: random.Random = random) -> tuple[S, float]:
        # Use temperature in softmax
        scaled_values = self.state_values / self.temperature
        exp_values = np.exp(scaled_values - np.max(scaled_values))
        probabilities = exp_values / np.sum(exp_values)
        
        selected_state = rng.choices(self.states, weights=probabilities)[0]
        nll = -np.log(probabilities[self.states.index(selected_state)])
        
        return (selected_state, nll)
    
    def calc_log_lik(self, state: S) -> float:
        '''Calculate the log likelihood of selecting a particular state.'''
        state_values = np.array([self.value_function[s] for s in self.state_space])
        exp_values = np.exp(state_values - np.max(state_values))
        probabilities = exp_values / np.sum(exp_values)
        return -np.log(probabilities[self.state_space.index(state)])
    
    def calc_log_likelihood_all(self) -> dict[S, float]:
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

class RandomGoalPolicy(GoalSelectionPolicy[S, A]):
    
    def __init__(self, mdp: MarkovDecisionProcess[S, A]):
        super().__init__(mdp)
    
    def select_goal(self, s: S) -> S:
        return random.choice(self.mdp.state_space)
    
class PCFGGoalPolicy(GoalSelectionPolicy[S, A]):

    def __init__(self, mdp: MarkovDecisionProcess[S, A], p_rules, cap=10):
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
        
    