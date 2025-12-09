"""
Python Low-Level API Reward Calculation Example

This script demonstrates how rewards are calculated in Python using the
low-level ML-Agents API. Rewards are computed based on observations received
from Unity, implementing the same reward logic that was previously in agent.cs.
"""

import numpy as np
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import \
    EngineConfigurationChannel


class PythonRewardCalculator:
    """
    Calculates rewards based on observations from Unity.

    This replaces the reward logic that was previously in agent.cs.
    """

    def __init__(self):
        # Reward parameters
        self.reward_goal_reached = 1.0
        # Increased from 0.01 to 1.0 to provide meaningful feedback against time penalty
        self.reward_distance_improvement = 0.001
        self.reward_wall_collision = -0.15
        self.reward_wall_time_penalty = -0.01

        # Track previous observation for reward shaping
        self.previous_obs = None

    def calculate_reward(self, obs, max_steps, current_step):
        """
        Calculate reward based on current observation.

        Observation structure (from agent.cs):
        - obs[0]: relative goal X position (normalized by 5f)
        - obs[1]: relative goal Z position (normalized by 5f)
        - obs[2]: current distance to goal (normalized by 10f)
        - obs[3]: previous distance to goal (normalized by 10f)
        - obs[4]: has hit wall (1.0 = true, 0.0 = false)
        - obs[5]: has reached goal (1.0 = true, 0.0 = false)
        - obs[6]: time spent in wall collision
        - obs[7]: agent velocity magnitude (normalized by 5f)

        Args:
            obs: Observation array from Unity
            max_steps: Maximum steps per episode
            current_step: Current step number

        Returns:
            float: Calculated reward
        """
        reward = 0.0

        # Check if goal reached
        if len(obs) > 5 and obs[5] > 0.5:
            reward += self.reward_goal_reached

        # Reward shaping: distance improvement/worsening
        # We calculate this manually using previous_obs to ensure consistency in Python
        if (
            len(obs) > 2
            and self.previous_obs is not None
            and len(self.previous_obs) > 2
        ):
            current_distance = (
                obs[2] * 10.0
            )  # Denormalize (assuming normalization by 10)
            previous_distance = self.previous_obs[2] * 10.0

            distance_delta = previous_distance - current_distance  # Positive = closer
            reward += distance_delta * self.reward_distance_improvement

        # Wall collision penalty
        if len(obs) > 4 and obs[4] > 0.5:
            reward += self.reward_wall_collision

            # Additional penalty for time spent in wall
            if len(obs) > 6:
                time_in_wall = obs[6]
                reward += time_in_wall * self.reward_wall_time_penalty

        # Time penalty (encourage faster completion)
        # Normalized so that if max_steps is taken, total penalty is -2.0 (unless goal reached)
        reward += -2.0 / max_steps

        # Update previous observation
        self.previous_obs = obs.copy()

        return reward

    def reset(self):
        """Reset the calculator state."""
        self.previous_obs = None


def run_low_level_api_example(unity_env_path=None, worker_id=0):
    """
    Example of using low-level API with Python reward calculation.

    Args:
        unity_env_path: Path to Unity executable (None for Editor)
        worker_id: Worker ID for parallel environments
    """
    # Create engine configuration channel
    engine_config_channel = EngineConfigurationChannel()

    # Initialize Unity environment
    env = UnityEnvironment(
        file_name=unity_env_path,
        worker_id=worker_id,
        no_graphics=True,
        side_channels=[engine_config_channel],
        timeout_wait=60,
    )

    # Set time scale
    engine_config_channel.set_configuration_parameters(time_scale=20.0)

    # Reset environment
    env.reset()

    # Get behavior name
    behavior_names = list(env.behavior_specs.keys())
    if len(behavior_names) == 0:
        raise ValueError("No behavior specs found!")

    behavior_name = behavior_names[0]
    spec = env.behavior_specs[behavior_name]
    max_steps = spec.max_step if spec.max_step > 0 else 1000

    print(f"Using behavior: {behavior_name}")
    print(f"Observation shape: {spec.observation_specs[0].shape}")
    print(f"Max steps: {max_steps}")

    # Create reward calculator
    reward_calc = PythonRewardCalculator()

    # Run a few episodes
    for episode in range(3):
        env.reset()
        reward_calc.reset()

        episode_reward = 0.0
        step_count = 0

        decision_steps, terminal_steps = env.get_steps(behavior_name)

        while len(terminal_steps) == 0 and step_count < max_steps:
            # Get current observation
            if len(decision_steps) > 0:
                obs = decision_steps.obs[0][0]
                agent_id = decision_steps.agent_id[0]

                # Calculate reward in Python
                reward = reward_calc.calculate_reward(obs, max_steps, step_count)
                episode_reward += reward

                # Take random action
                action = np.random.randint(0, spec.action_spec.discrete_branches[0])
                discrete_actions = np.array([[action]], dtype=np.int32)
                action_tuple = ActionTuple(discrete=discrete_actions)

                env.set_actions(behavior_name, action_tuple)
                env.step()

                decision_steps, terminal_steps = env.get_steps(behavior_name)
                step_count += 1
            else:
                break

        print(
            f"Episode {episode + 1}: Steps = {step_count}, Total Reward = {episode_reward:.2f}"
        )

    env.close()
    print("\nLow-level API example completed!")


if __name__ == "__main__":
    print("Python Low-Level API Reward Calculation Example")
    print("=" * 50)

    # Run example (connect to Unity Editor)
    run_low_level_api_example(unity_env_path=None, worker_id=0)
