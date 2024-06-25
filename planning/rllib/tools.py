from .simulation_result import TDLearningSimulationResult
from .mdp import MDPPolicy
from .gymwrap import GymWrapper
import random
from tqdm import tqdm

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def simulation_loop(
        *,
        env: GymWrapper,
        policy : MDPPolicy,
        n_episodes : int,
        max_steps : int,
        seed: int,
) -> TDLearningSimulationResult:
    '''
    Runs a RL simulation and conveniently stores the results.

    Returns:
        TDLearningSimulationResult
    '''
    
    # both initialize and reset the Q values
    policy.reset()
    trajectory = []
    state_values = []
    rng = random.Random(seed)

    # episodes loop
    for _ in tqdm(range(n_episodes), desc="Episodes"):
        # TODO: finish implementing the reset function
        state_idx, _ = env.reset(seed=rng.randint(0, 2**32 - 1))
        state = env.mdp.state_space[state_idx]

        # steps loop
        for _ in range(max_steps):
            action = policy.action_sample(state, rng=rng)
            action_idx = env.mdp.action_space.index(action)
            new_state_idx, reward, done, _ = env.step(action_idx)
            new_state = env.mdp.state_space[new_state_idx]
            policy.update(
                s=state,
                a=action,
                r=reward,
                ns=new_state
            )

            trajectory.append((state, action, reward, new_state, done))
            state_values.append({
                s: policy.state_value(s)
                for s in env.mdp.state_space
            })

            # if we reach the termination condition (an absorbing state), simulation is done. 
            if done:
                break
            state = new_state
        policy.end_episode()
    return TDLearningSimulationResult(trajectory, state_values, policy, env.mdp)