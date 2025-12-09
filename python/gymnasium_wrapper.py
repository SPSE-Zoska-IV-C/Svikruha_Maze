"""
Gymnasium Wrapper for Unity ML-Agents Maze Environment

This wrapper allows you to use the Unity ML-Agents maze environment
with Gymnasium API, making it compatible with standard RL libraries
like Stable-Baselines3.
"""

import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.exception import UnityTimeOutException
from mlagents_envs.side_channel.engine_configuration_channel import \
    EngineConfigurationChannel


class UnityMazeGymWrapper(gym.Env):
    """
    Gymnasium wrapper for Unity ML-Agents Maze environment.

    Args:
        unity_env_path (str): Path to the Unity executable. Use None for Unity Editor.
        worker_id (int): Worker ID for parallel environments (default: 0).
        no_graphics (bool): Run Unity without graphics (faster training, default: True).
        time_scale (float): Unity time scale (higher = faster, default: 20.0).
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self, unity_env_path=None, worker_id=0, no_graphics=True, time_scale=20.0, max_steps=None
    ):
        super().__init__()

        # Create engine configuration channel for speed control
        self.engine_config_channel = EngineConfigurationChannel()

        # Initialize Unity environment with error handling
        try:
            self.unity_env = UnityEnvironment(
                file_name=unity_env_path,
                worker_id=worker_id,
                no_graphics=no_graphics,
                side_channels=[self.engine_config_channel],
                timeout_wait=60,  # Wait up to 60 seconds for Unity to connect
            )
        except Exception as e:
            error_msg = (
                f"Failed to connect to Unity environment.\n"
                f"Error: {e}\n"
                f"Make sure Unity is running with the scene open, "
                f"or provide a valid path to a Unity build executable."
            )
            raise RuntimeError(error_msg) from e

        # Set time scale for faster training
        self.engine_config_channel.set_configuration_parameters(time_scale=time_scale)

        # Reset the environment to get behavior specs
        self.unity_env.reset()

        # Get behavior name (should be the agent behavior)
        self.behavior_names = list(self.unity_env.behavior_specs.keys())
        if len(self.behavior_names) == 0:
            raise ValueError("No behavior specs found in Unity environment!")

        self.behavior_name = self.behavior_names[0]
        print(f"Using behavior: {self.behavior_name}")

        # Get behavior spec
        self.spec = self.unity_env.behavior_specs[self.behavior_name]

        # Define observation space
        # Base observations (8): goal position, distance, wall collision, velocity
        # Ray observations (optional, 16 with 8 rays): distances and types to obstacles
        # Total: 8 (base) or 24 (with ray observations)
        # The actual shape is determined dynamically from Unity's behavior spec
        # ML-Agents 4.x uses observation_specs list
        if len(self.spec.observation_specs) > 0:
            obs_spec = self.spec.observation_specs[0]
            obs_shape = obs_spec.shape
        else:
            raise ValueError("No observation specs found!")

        # Define observation space
        # Based on agent.cs: observations are normalized relative goal positions
        # Values are divided by 5f, so they should be roughly in [-1, 1] range
        # But we use wider bounds to be safe
        self.observation_space = spaces.Box(
            low=-10.0,  # More reasonable bounds than -inf
            high=10.0,  # Based on maze size (10x10 * scale 6 = 60 units, normalized by 5)
            shape=obs_shape,
            dtype=np.float32,
        )

        # Define action space (discrete: 0=nothing, 1=forward, 2=left, 3=right)
        if self.spec.action_spec.discrete_size > 0:
            self.action_space = spaces.Discrete(
                self.spec.action_spec.discrete_branches[0]
            )
        else:
            raise ValueError("Expected discrete action space!")

        print(f"Observation space: {self.observation_space}")
        print(f"Action space: {self.action_space}")

        self.current_step = 0

        # Handle missing max_step in newer ML-Agents versions
        if max_steps is not None:
            self.max_steps = max_steps
        elif hasattr(self.spec, "max_step") and self.spec.max_step > 0:
            self.max_steps = self.spec.max_step
        else:
            self.max_steps = 1000

        # Store previous observation for reward calculation
        self.previous_obs = None

        # Reward parameters (can be adjusted)
        self.reward_goal_reached = 1.0
        self.reward_distance_improvement = (
            0.001  # Reduced for maze: prevents local optima
        )
        self.reward_wall_collision = -0.15
        self.reward_wall_time_penalty = -0.01
        self.reward_time_penalty = -2.0 / self.max_steps

    def reset(self, seed=None, options=None):
        """Reset the environment and return initial observation."""
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)
            # Note: Unity ML-Agents doesn't support seed propagation directly
            # The seed mainly affects Python-side randomness

        # Try to get steps from running environment first (avoids double reset)
        needs_reset = True
        try:
            decision_steps, terminal_steps = self.unity_env.get_steps(self.behavior_name)
            if len(decision_steps) > 0:
                needs_reset = False
        except:
            needs_reset = True

        if needs_reset:
            # Reset Unity environment with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self.unity_env.reset()
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise RuntimeError(
                            f"Failed to reset Unity environment after {max_retries} attempts: {e}"
                        )
                    # Wait a bit before retrying
                    time.sleep(0.1)

        # Get initial observation with retry logic
        decision_steps, terminal_steps = self.unity_env.get_steps(self.behavior_name)

        if len(decision_steps) > 0:
            obs = decision_steps.obs[0][0]  # Get first agent's observation
        elif len(terminal_steps) > 0:
            # Agent terminated immediately, reset again
            self.unity_env.reset()
            decision_steps, terminal_steps = self.unity_env.get_steps(
                self.behavior_name
            )
            if len(decision_steps) > 0:
                obs = decision_steps.obs[0][0]
            else:
                raise RuntimeError("Failed to get initial observation after reset")
        else:
            raise RuntimeError("No agents found after reset!")

        self.current_step = 0
        self.previous_obs = obs.copy()  # Store for reward calculation

        info = {"max_steps": self.max_steps}

        return obs.astype(np.float32), info

    def step(self, action):
        """Execute action and return observation, reward, done, info."""
        # Validate action
        if not self.action_space.contains(action):
            raise ValueError(
                f"Invalid action {action}. Action must be in {self.action_space}"
            )

        # Convert action to Unity format (ML-Agents 4.x compatible)
        discrete_actions = np.array([[action]], dtype=np.int32)
        action_tuple = ActionTuple(discrete=discrete_actions)

        # Set action for all agents with this behavior
        self.unity_env.set_actions(self.behavior_name, action_tuple)

        # Step the environment
        try:
            self.unity_env.step()
        except UnityTimeOutException as e:
            print(f"\nERROR: Unity environment timed out during step().")
            print(f"Possible causes:")
            print(f"1. The Unity Editor is paused or crashed.")
            print(
                f"2. 'Run In Background' is not enabled in Unity (Edit > Project Settings > Player > Resolution and Presentation)."
            )
            print(
                f"3. The Time Scale is too high (currently {self.engine_config_channel.time_scale if hasattr(self, 'engine_config_channel') else 'unknown'})."
            )
            raise e

        # Get results
        decision_steps, terminal_steps = self.unity_env.get_steps(self.behavior_name)

        self.current_step += 1

        # Check if max steps reached (truncation)
        max_steps_reached = self.current_step >= self.max_steps

        # Check if episode terminated
        if len(terminal_steps) > 0:
            # Agent terminated (reached goal or hit max steps in Unity)
            agent_id = list(terminal_steps.agent_id)[0]
            obs = terminal_steps.obs[0][0]
            # If max steps reached, it's truncation; otherwise natural termination
            terminated = not max_steps_reached
            truncated = max_steps_reached

        elif len(decision_steps) > 0:
            # Agent is still running
            obs = decision_steps.obs[0][0]
            terminated = False
            truncated = max_steps_reached

        else:
            # No agents found (shouldn't happen)
            raise RuntimeError("No agents found in environment!")

        # Calculate reward in Python based on observations
        # Observation structure (from agent.cs):
        # obs[0]: relative goal X position (normalized by 5f)
        # obs[1]: relative goal Z position (normalized by 5f)
        # obs[2]: current distance to goal (normalized by 10f)
        # obs[3]: previous distance to goal (normalized by 10f)
        # obs[4]: has hit wall (1.0 = true, 0.0 = false)
        # obs[5]: has reached goal (1.0 = true, 0.0 = false)
        # obs[6]: time spent in wall collision
        # obs[7]: agent velocity magnitude (normalized by 5f)
        #
        # If ray observations are enabled (default: 8 rays):
        # obs[8+i*2]: distance to obstacle in direction i (normalized, 0-1)
        # obs[9+i*2]: obstacle type (1.0 = wall, 0.5 = other, 0.0 = none)
        # Total with rays: 8 + (num_rays * 2) = 24 observations (with 8 rays)

        reward = self._calculate_reward(obs, terminated)

        # Store current observation for next step
        self.previous_obs = obs.copy()

        info = {
            "current_step": self.current_step,
            "max_steps": self.max_steps,
            "episode_length": self.current_step,
        }

        return obs.astype(np.float32), float(reward), terminated, truncated, info

    def _calculate_reward(self, obs, terminated):
        """
        Calculate reward based on observations.

        This implements the reward logic that was previously in Unity agent.cs:
        - Goal reached: +1.0
        - Distance improvement: +0.01 per unit closer
        - Distance worsening: -0.01 per unit farther
        - Wall collision: -0.15
        - Time in wall: -0.01 per second
        - Time penalty: -2.0 / max_steps per step
        """
        reward = 0.0

        # Check if goal reached
        if len(obs) > 5 and obs[5] > 0.5:  # has_reached_goal
            reward += self.reward_goal_reached

        # Reward shaping: distance improvement/worsening
        if (
            len(obs) > 3
            and self.previous_obs is not None
            and len(self.previous_obs) > 2
        ):
            current_distance = obs[2] * 10.0  # Denormalize (was normalized by 10f)
            previous_distance = (
                self.previous_obs[2] * 10.0
                if len(self.previous_obs) > 2
                else current_distance
            )

            distance_delta = previous_distance - current_distance  # Positive = closer
            reward += distance_delta * self.reward_distance_improvement

        # Wall collision penalty
        if len(obs) > 4 and obs[4] > 0.5:  # has_hit_wall
            reward += self.reward_wall_collision

            # Additional penalty for time spent in wall
            if len(obs) > 6:
                time_in_wall = obs[6]
                reward += time_in_wall * self.reward_wall_time_penalty

        # Time penalty (encourage faster completion)
        reward += self.reward_time_penalty

        return reward

    def render(self):
        """
        Render is handled by Unity.

        Returns:
            None: Rendering is handled by Unity environment.
        """
        # Rendering is handled by Unity, so we return None
        # If you want to add screenshot functionality, you could implement it here
        return None

    def close(self):
        """Close the Unity environment."""
        if hasattr(self, "unity_env") and self.unity_env is not None:
            try:
                self.unity_env.close()
                print("Unity environment closed.")
            except Exception as e:
                print(f"Warning: Error closing Unity environment: {e}")
            finally:
                self.unity_env = None


def make_unity_maze_env(
    unity_env_path=None, worker_id=0, no_graphics=True, time_scale=20.0, max_steps=None
):
    """
    Factory function to create Unity Maze Gymnasium environment.

    Args:
        unity_env_path (str): Path to Unity build executable. Use None for Unity Editor.
        worker_id (int): Worker ID for parallel training.
        no_graphics (bool): Run without graphics for faster training.
        time_scale (float): Speed up Unity simulation.
        max_steps (int): Override max steps per episode.

    Returns:
        UnityMazeGymWrapper: Gymnasium-compatible environment.

    Example:
        >>> env = make_unity_maze_env(unity_env_path="./build/MazeAgent.exe")
        >>> obs, info = env.reset()
        >>> for _ in range(1000):
        ...     action = env.action_space.sample()
        ...     obs, reward, terminated, truncated, info = env.step(action)
        ...     if terminated or truncated:
        ...         obs, info = env.reset()
        >>> env.close()
    """
    return UnityMazeGymWrapper(
        unity_env_path=unity_env_path,
        worker_id=worker_id,
        no_graphics=no_graphics,
        time_scale=time_scale,
        max_steps=max_steps,
    )


if __name__ == "__main__":
    # Test the wrapper
    print("Testing Unity Maze Gymnasium Wrapper...")

    # Create environment (connects to Unity Editor by default)
    env = make_unity_maze_env(unity_env_path=None, no_graphics=False, time_scale=1.0)

    print("\nRunning random agent for 5 episodes...")
    for episode in range(5):
        obs, info = env.reset()
        episode_reward = 0
        steps = 0

        while True:
            # Random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            steps += 1

            if terminated or truncated:
                print(
                    f"Episode {episode + 1}: Steps = {steps}, Total Reward = {episode_reward:.2f}"
                )
                break

    env.close()
    print("\nTest completed!")
