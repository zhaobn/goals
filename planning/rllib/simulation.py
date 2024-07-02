import matplotlib.pyplot as plt
import random
from matplotlib import patheffects
import numpy as np
import pandas as pd
from collections import Counter
# from .bandit import MultiArmedBandit, BanditPolicy
from .mdp import MDPPolicy, MarkovDecisionProcess
from .shapeworld import ShapeWorld

class Simulation:
    def __init__(
            self,
            trajectory,
            state_values,
            policy : MDPPolicy,
            mdp : MarkovDecisionProcess
    ):
        self.trajectory = trajectory
        self.state_values = state_values
        self.policy = policy
        self.mdp = mdp # shape world

    def plot_timestep(self, timestep):
        raise NotImplementedError

class TDLearningSimulationResult:
    def __init__(
            self,
            trajectory,
            state_values,
            policy : MDPPolicy,
            sw : ShapeWorld
    ):
        self.trajectory = trajectory
        self.state_values = state_values
        self.policy = policy
        self.sw = sw # shape world

    def plot_timestep(self, timestep):
        raise NotImplementedError
        # TODO: Create plotting code to visualize what the learner is doing.
        timestep = timestep if timestep >= 0 else len(self.trajectory) + timestep
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        states_visited = Counter(
            [s for s, _, _, _, _ in self.trajectory[:timestep]]
        )
        gwp = self.sw.plot(ax=axes[0])
        gwp.plot_location_map(states_visited)
        gwp.ax.set_title(f"States Visitation Counts at Timestep {timestep}")
        gwp = self.sw.plot(ax=axes[1])
        gwp.plot_location_map(self.state_values[timestep])
        gwp.ax.set_title(f"State Values at Timestep {timestep}")

    def rewards(self):
        '''
        Get the rewards at each time step. Recall that trajectory has format:
        trajectory.append((state, action, reward, new_state, done))
        '''
        return np.array([r for _, _, r, _, _ in self.trajectory])

    def plot_reward_rate(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(5, 3), dpi=200)
        _ = ax.plot(self.rewards().cumsum() / np.arange(1, len(self.rewards()) + 1))
        ax.set_title("Reward Rate")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Reward Rate")