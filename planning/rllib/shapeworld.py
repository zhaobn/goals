from collections import namedtuple, defaultdict
from typing import Sequence, Tuple, Dict
from itertools import product
import random
from random import Random
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from dataclasses import dataclass
from typing import Literal
from .mdp import MarkovDecisionProcess, S, A
# from .distributions import Distribution, DiscreteDistribution, Uniform

@dataclass(frozen=True)
class Shape:
    sides: Literal['circle', 'square', 'triangle']
    shade: Literal['low', 'medium', 'high']
    texture: Literal['plain', 'stripes', 'dots']

    def __eq__(self, other) -> bool:
        """Compare two shapes for equality.
        
        Shapes are equal if all their attributes match.
        
        Args:
            other: Another Shape object to compare with
            
        Returns:
            bool: True if shapes are equal, False otherwise
        """
        if not isinstance(other, Shape):
            return False
        
        return (self.sides == other.sides and 
                self.shade == other.shade and 
                self.texture == other.texture)
    
    def __hash__(self):
        """Make Shape hashable for use as dictionary key.
        
        Returns:
            int: Hash value based on the attributes
        """
        return hash((self.sides, self.shade, self.texture))

    def __post_init__(self):
        # Additional validation if needed
        if self.sides not in ['circle', 'square', 'triangle']:
            raise ValueError(f'Invalid shape: {self.sides}')
        if self.shade not in ['low', 'medium', 'high']:
            raise ValueError(f'Invalid shade: {self.shade}')
        if self.texture not in ['plain', 'stripes', 'dots']:
            raise ValueError(f'Invalid texture: {self.texture}')

@dataclass(frozen=True)
class State:
    shape1: Shape
    shape2: Shape
    shape3: Shape

    def __eq__(self, other) -> bool:
        """Compare two states for equality.
        
        States are equal if all their shapes are equal in corresponding positions.
        
        Args:
            other: Another State object to compare with
            
        Returns:
            bool: True if states are equal, False otherwise
        """
        if not isinstance(other, State):
            return False
        
        return (self.shape1 == other.shape1 and 
                self.shape2 == other.shape2 and 
                self.shape3 == other.shape3)
    
    def __hash__(self):
        """Make State hashable for use as dictionary key.
        
        Returns:
            int: Hash value based on the shapes
        """
        return hash((self.shape1, self.shape2, self.shape3))

@dataclass(frozen=True)
class Action:
    actor: Literal[1, 2, 3]
    recipient: Literal[1, 2, 3]
    
    def __post_init__(self):
        if self.actor == self.recipient:
            raise ValueError("Actor and recipient must be different")
        if not (1 <= self.actor <= 3 and 1 <= self.recipient <= 3):
            raise ValueError("Actor and recipient must be between 1 and 3")

class ShapeWorld(MarkovDecisionProcess[State, Action]):
    '''A world with shapes that can be manipulated.'''
    
    # Feature space constants
    SHAPE_LIST = ('circle', 'square', 'triangle')
    SHADE_LIST = ('low', 'medium', 'high')
    TEXTURE_LIST = ('plain', 'stripes', 'dots')
    
    # Transition probability constants
    SHAPE_TRANSITION_PROB = 0.9
    TEXTURE_TRANSITION_PROB = 1.0
    SHADE_CYCLE_PROB = 0.1  # you changed this to 10%
    
    def __init__(self, goal: State, discount_rate: float):
        '''Initialize the ShapeWorld with a goal state and discount rate.'''
        self.GOAL = goal
        self.discount_rate = discount_rate
        
        # Constants for rewards
        self.GOAL_REWARD = 0  # zero because of averaging across Q-tables we do later
        self.STEP_COST = -1
        
        # Set up shape space using class constants
        shape_space = [
            Shape(sides=sides, shade=shade, texture=texture)
            for sides, shade, texture in product(
                self.SHAPE_LIST,
                self.SHADE_LIST,
                self.TEXTURE_LIST
            )
        ]
        
        # Set up state space
        self.state_space = [
            State(shape1=shape1, shape2=shape2, shape3=shape3)
            for shape1, shape2, shape3 in product(shape_space, shape_space, shape_space)
        ]
        
        # Set up action space
        self.action_space = [
            Action(actor=1, recipient=2),  # a1r2
            Action(actor=1, recipient=3),  # a1r3
            Action(actor=2, recipient=1),  # a2r1
            Action(actor=2, recipient=3),  # a2r3
            Action(actor=3, recipient=1),  # a3r1
            Action(actor=3, recipient=2),  # a3r2
        ]

    def next_state_sample(self, s: State, a: Action, rng: Random = random) -> State:
        '''
        Given a state and action, return a possible next state.
        
        Note on indexing:
        - Action.actor and Action.recipient are 1-based (1,2,3)
        - Python list indexing is 0-based (0,1,2)
        - State attributes are accessed by name (shape1, shape2, shape3)
        '''
        # Get shapes using proper attribute names
        actor_shape = getattr(s, f'shape{a.actor}')
        recipient_shape = getattr(s, f'shape{a.recipient}')
        
        # 1. Determine recipient's new shape
        if rng.random() < self.SHAPE_TRANSITION_PROB:
            sides = actor_shape.sides
        else:
            available_shapes = [shape for shape in self.SHAPE_LIST if shape != actor_shape.sides]
            sides = rng.choice(available_shapes)
        
        # 2. Determine recipient's new texture
        current_texture_idx = self.TEXTURE_LIST.index(recipient_shape.texture)
        next_texture_idx = (current_texture_idx + 1) % len(self.TEXTURE_LIST)
        texture = self.TEXTURE_LIST[next_texture_idx]
        
        # 3. Determine recipient's new shade
        actor_shade_idx = self.SHADE_LIST.index(actor_shape.shade)
        recipient_shade_idx = self.SHADE_LIST.index(recipient_shape.shade)
        
        if actor_shade_idx == recipient_shade_idx:
            # Same shade case
            if recipient_shape.shade in ['low', 'high']:
                if rng.random() < self.SHADE_CYCLE_PROB:
                    # Equal chance to cycle around or go to medium
                    if rng.random() < 0.5:
                        shade = 'high' if recipient_shape.shade == 'low' else 'low'
                    else:
                        shade = 'medium'
                else:
                    shade = recipient_shape.shade
            elif recipient_shape.shade == 'medium':
                if rng.random() < self.SHADE_CYCLE_PROB:
                    shade = rng.choice(['low', 'high'])
                else:
                    shade = 'medium'
        else:
            # Different shade case: move one step towards actor's shade
            if actor_shade_idx > recipient_shade_idx:
                shade = self.SHADE_LIST[recipient_shade_idx + 1]
            else:
                shade = self.SHADE_LIST[recipient_shade_idx - 1]
        
        # Create new state using proper attribute names
        new_shapes = {
            'shape1': s.shape1,
            'shape2': s.shape2,
            'shape3': s.shape3
        }
        new_shapes[f'shape{a.recipient}'] = Shape(sides=sides, shade=shade, texture=texture)
        
        return State(
            shape1=new_shapes['shape1'],
            shape2=new_shapes['shape2'],
            shape3=new_shapes['shape3']
        )

    def reward(self, s: State, a: Action, ns: State) -> float:
        """Calculate the reward for a state transition.
        
        Args:
            s: Current state
            a: Action taken
            ns: Next state
        
        Returns:
            float: STEP_COST (-1) for each action, plus GOAL_REWARD (0) if goal reached
        
        Note:
            GOAL_REWARD is 0 because we average across Q-tables later.
            This means the reward structure is purely driven by path length to goal.
        """
        reward = self.STEP_COST
        if self._is_goal(ns):
            reward += self.GOAL_REWARD
        return reward

    def is_absorbing(self, s: State) -> bool:
        '''
        Check to see if state is absorbing. 

        For this context, the goal state is our absorbing state, and the simulation 
        should be terminated if reached.

        Args:
            s: The state to check

        Returns:
            bool: Whether the state is an absorbing state (goal state) or not.
        '''
        return self.GOAL == s

    def get_possible_next_states(self, s: State, a: Action) -> Sequence[State]:
        """Return the possible next states given a state and action.
        
        Args:
            s: Current state
            a: Action taken
            
        Returns:
            Sequence[State]: List of all possible next states
        """
        # Get shapes using proper attribute names
        actor_shape = getattr(s, f'shape{a.actor}')
        recipient_shape = getattr(s, f'shape{a.recipient}')
        
        # 1. Determine possible sides (80% actor's shape, 20% different shape)
        possible_sides = [actor_shape.sides]  # 80% chance
        other_sides = [side for side in self.SHAPE_LIST if side != actor_shape.sides]
        possible_sides.extend(other_sides)  # 20% chance split among other shapes
        
        # 2. Determine possible shades
        if actor_shape.shade == recipient_shape.shade:
            # Same shade case
            if recipient_shape.shade in ['low', 'high']:
                possible_shades = [
                    recipient_shape.shade,  # 90% chance stay same
                    'high' if recipient_shape.shade == 'low' else 'low',  # 5% chance to cycle
                    'medium'  # 5% chance to go to medium
                ]
            elif recipient_shape.shade == 'medium':
                possible_shades = [
                    'medium',  # 90% chance stay same
                    'low',    # 5% chance
                    'high'    # 5% chance
                ]
        else:
            # Different shade case - move one step towards actor's shade
            recipient_idx = self.SHADE_LIST.index(recipient_shape.shade)
            actor_idx = self.SHADE_LIST.index(actor_shape.shade)
            step = 1 if actor_idx > recipient_idx else -1
            possible_shades = [self.SHADE_LIST[recipient_idx + step]]
        
        # 3. Determine possible textures (cycles through plain -> stripes -> dots)
        current_texture_idx = self.TEXTURE_LIST.index(recipient_shape.texture)
        next_texture_idx = (current_texture_idx + 1) % len(self.TEXTURE_LIST)
        possible_textures = [self.TEXTURE_LIST[next_texture_idx]]
        
        # Generate all possible combinations
        possible_states = set()
        for sides in possible_sides:
            for shade in possible_shades:
                for texture in possible_textures:
                    # Create new shape with the combination
                    new_shape = Shape(sides=sides, shade=shade, texture=texture)
                    
                    # Create new state, only updating the recipient shape
                    if a.recipient == 1:
                        new_state = State(shape1=new_shape, shape2=s.shape2, shape3=s.shape3)
                    elif a.recipient == 2:
                        new_state = State(shape1=s.shape1, shape2=new_shape, shape3=s.shape3)
                    else:  # recipient == 3
                        new_state = State(shape1=s.shape1, shape2=s.shape2, shape3=new_shape)
                    
                    possible_states.add(new_state)
        
        return list(possible_states)
    
    def transition_probability(self, s: State, a: Action, ns: State) -> float:
        """Return the transition probability from state s to state ns given action a.
        
        Args:
            s: Current state
            a: Action taken
            ns: Next state
            
        Returns:
            float: Probability of transitioning from s to ns given action a
        """
        # Get shapes using proper attribute access
        actor_shape = getattr(s, f'shape{a.actor}')
        recipient_shape = getattr(s, f'shape{a.recipient}')
        next_recipient_shape = getattr(ns, f'shape{a.recipient}')
        
        # Verify that only the recipient shape changed
        if (getattr(ns, f'shape{a.actor}') != actor_shape or
            any(getattr(ns, f'shape{i}') != getattr(s, f'shape{i}') 
                for i in [1,2,3] if i != a.recipient)):
            return 0.0
        
        # 1. Check shape transition
        shape_prob = 0.0
        if next_recipient_shape.sides == actor_shape.sides:
            shape_prob = self.SHAPE_TRANSITION_PROB
        elif next_recipient_shape.sides != actor_shape.sides:
            available_shapes = [shape for shape in self.SHAPE_LIST if shape != actor_shape.sides]
            shape_prob = (1 - self.SHAPE_TRANSITION_PROB) / len(available_shapes)
        else:
            return 0.0

        # 2. Check texture transition
        texture_prob = 0.0
        current_texture_idx = self.TEXTURE_LIST.index(recipient_shape.texture)
        expected_texture_idx = (current_texture_idx + 1) % len(self.TEXTURE_LIST)
        if next_recipient_shape.texture == self.TEXTURE_LIST[expected_texture_idx]:
            texture_prob = 1.0
        else:
            return 0.0

        # 3. Check shade transition
        shade_prob = 0.0
        if actor_shape.shade == recipient_shape.shade:
            # Same shade case
            if recipient_shape.shade in ['low', 'high']:
                if next_recipient_shape.shade == recipient_shape.shade:
                    shade_prob = 1 - self.SHADE_CYCLE_PROB
                elif (recipient_shape.shade == 'low' and next_recipient_shape.shade == 'high' or
                    recipient_shape.shade == 'high' and next_recipient_shape.shade == 'low'):
                    shade_prob = self.SHADE_CYCLE_PROB
            elif recipient_shape.shade == 'medium':
                if next_recipient_shape.shade == 'medium':
                    shade_prob = 1 - self.SHADE_CYCLE_PROB
                elif next_recipient_shape.shade in ['low', 'high']:
                    shade_prob = self.SHADE_CYCLE_PROB / 2  # Equal chance of low or high
        else:
            # Different shade case - move one step towards actor's shade
            recipient_idx = self.SHADE_LIST.index(recipient_shape.shade)
            actor_idx = self.SHADE_LIST.index(actor_shape.shade)
            step = 1 if actor_idx > recipient_idx else -1
            if next_recipient_shape.shade == self.SHADE_LIST[recipient_idx + step]:
                shade_prob = 1.0

        # Return combined probability
        return shape_prob * texture_prob * shade_prob

    def plot_state_transitions(self, state: State, action: Action, n_samples: int = 10000):
        """Visualize state transitions from a given state and action.
        
        Args:
            state: Starting state
            action: Action to take
            n_samples: Number of samples to generate
        """
        
        # Get all possible next states and their theoretical probabilities
        possible_states = self.get_possible_next_states(state, action)
        theoretical_probs = {
            ns: self.transition_probability(state, action, ns)
            for ns in possible_states
        }
        
        # Generate samples
        samples = [self.next_state_sample(state, action) for _ in range(n_samples)]
        empirical_counts = Counter(samples)
        empirical_probs = {s: c/n_samples for s, c in empirical_counts.items()}
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Compare theoretical vs empirical probabilities
        states = list(set(possible_states) | set(empirical_counts.keys()))
        x = range(len(states))
        theoretical = [theoretical_probs.get(s, 0) for s in states]
        empirical = [empirical_probs.get(s, 0) for s in states]
        
        width = 0.35
        ax1.bar([i - width/2 for i in x], theoretical, width, label='Theoretical', alpha=0.6)
        ax1.bar([i + width/2 for i in x], empirical, width, label='Empirical', alpha=0.6)
        ax1.set_ylabel('Probability')
        ax1.set_title('Transition Probabilities')
        ax1.legend()
        
        # Plot 2: Feature changes
        feature_changes = {
            'sides': 0,
            'shade': 0,
            'texture': 0
        }
        
        recipient_idx = action.recipient
        recipient_shape = getattr(state, f'shape{recipient_idx}')
        
        for ns in samples:
            new_recipient = getattr(ns, f'shape{recipient_idx}')
            if new_recipient.sides != recipient_shape.sides:
                feature_changes['sides'] += 1
            if new_recipient.shade != recipient_shape.shade:
                feature_changes['shade'] += 1
            if new_recipient.texture != recipient_shape.texture:
                feature_changes['texture'] += 1
        
        # Convert to percentages
        for k in feature_changes:
            feature_changes[k] = (feature_changes[k] / n_samples) * 100
        
        ax2.bar(feature_changes.keys(), feature_changes.values())
        ax2.set_ylabel('Percentage Changed (%)')
        ax2.set_title('Feature Change Frequencies')
        
        # Add percentage labels on top of bars
        for i, v in enumerate(feature_changes.values()):
            ax2.text(i, v + 1, f'{v:.1f}%', ha='center')
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print("\nSummary of transitions:")
        print(f"Number of possible next states: {len(possible_states)}")
        print(f"Number of observed next states: {len(empirical_counts)}")
        print("\nFeature change frequencies:")
        for feature, pct in feature_changes.items():
            print(f"{feature.capitalize()}: {pct:.1f}%")

    def get_state_space(self) -> Sequence[State]:
        '''Return the state space.'''
        return self.state_space
    
    def get_actions(self, s: State) -> Sequence[Action]:
        '''Return the action space.'''
        return self.action_space

    # helper functions
    def _is_goal(self, ns: State):
        return self.GOAL == ns

class GoalWorld(MarkovDecisionProcess[State, State]):
    '''A world where the agent selects goals as actions, using ShapeWorld parameters.
    
    This class represents a meta-level decision problem where the agent chooses
    which goals to pursue in ShapeWorld, rather than low-level shape manipulation actions.
    '''
    
    def __init__(self, shape_world: ShapeWorld):
        '''Initialize GoalWorld using parameters from a ShapeWorld instance.
        
        Args:
            shape_world: A ShapeWorld instance to inherit parameters from
        '''
        self.shape_world = shape_world
        
        # Inherit state space from ShapeWorld
        self.state_space = shape_world.state_space
        
        # In GoalWorld, the actions are selecting states as goals
        self.action_space = self.state_space
        
        # Track number of states for convenience
        self.num_states = len(self.state_space)
        
        # Inherit constants
        self.GOAL_REWARD = shape_world.GOAL_REWARD
        self.STEP_COST = shape_world.STEP_COST
        self.DISCOUNT_RATE = shape_world.discount_rate

    def get_state_space(self) -> Sequence[State]:
        '''Return the state space (inherited from ShapeWorld).'''
        return self.state_space
    
    def get_actions(self, s: State) -> Sequence[State]:
        '''Return the action space (all possible goal states).
        
        In GoalWorld, actions are selecting which state to make the goal.
        '''
        return self.action_space

    def next_state_sample(self, s: State, a: State, rng: Random = random) -> State:
        '''Sample the next state. Not needed for goal selection.'''
        raise NotImplementedError("GoalWorld does not implement state transitions")

    def reward(self, s: State, a: State, ns: State) -> float:
        '''Calculate reward. Not needed for goal selection.'''
        raise NotImplementedError("GoalWorld does not implement rewards")

    def is_absorbing(self, s: State) -> bool:
        '''Check if state is absorbing. Not needed for goal selection.'''
        raise NotImplementedError("GoalWorld does not implement absorbing states")
        
