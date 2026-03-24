"""
Gymnasium Wrapper for Unity ML-Agents Maze Environment

This wrapper allows you to use the Unity ML-Agents maze environment
with Gymnasium API, making it compatible with standard RL libraries
like Stable-Baselines3.

Supports parallel training with multiple Unity build instances via SubprocVecEnv.
"""

import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.exception import UnityTimeOutException
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

# Default base port for ML-Agents communication
DEFAULT_BASE_PORT = 5004


class UnityMazeGymWrapper(gym.Env):
    """
    Gymnasium wrapper for Unity ML-Agents Maze environment.

    Args:
        unity_env_path (str): Path to the Unity executable. Use None for Unity Editor.
        worker_id (int): Worker ID for parallel environments (default: 0).
        no_graphics (bool): Run Unity without graphics (faster training, default: True).
        time_scale (float): Unity time scale (higher = faster, default: 20.0).
        max_steps (int): Maximum steps per episode (default: None, uses Unity setting).
        base_port (int): Base port for ML-Agents communication (default: 5005).
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        unity_env_path=None,
        worker_id=0,
        no_graphics=True,
        time_scale=20.0,
        max_steps=None,
        base_port=DEFAULT_BASE_PORT,
    ):
        super().__init__()

        self.worker_id = worker_id
        self.unity_env_path = unity_env_path

        # Create engine configuration channel for speed control
        self.engine_config_channel = EngineConfigurationChannel()

        # Initialize Unity environment with retry logic for parallel startup
        max_retries = 3
        retry_delay = 2.0  # seconds between retries

        for attempt in range(max_retries):
            try:
                self.unity_env = UnityEnvironment(
                    file_name=unity_env_path,
                    worker_id=worker_id,
                    no_graphics=no_graphics,
                    side_channels=[self.engine_config_channel],
                    timeout_wait=120,  # Increased timeout for parallel startup
                    base_port=base_port,
                )
                break  # Success, exit retry loop
            except Exception as e:
                if attempt < max_retries - 1:
                    print(
                        f"Worker {worker_id}: Connection attempt {attempt + 1} failed, "
                        f"retrying in {retry_delay}s... ({e})"
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 1.5  # Exponential backoff
                else:
                    error_msg = (
                        f"Failed to connect to Unity environment after {max_retries} attempts.\n"
                        f"Worker ID: {worker_id}, Port: {base_port + worker_id}\n"
                        f"Error: {e}\n"
                        f"Make sure Unity is running with the scene open, "
                        f"or provide a valid path to a Unity build executable.\n"
                        f"For parallel training, you MUST use a Unity build (.exe), not the Editor."
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
            self.action_space = spaces.Discrete(self.spec.action_spec.discrete_branches[0])
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

        # Reward parameters (optimized for random maze navigation)
        # KEY INSIGHT: Standing still must be MUCH worse than exploring with wall collisions!
        # Old values caused agent to learn standing still (-50) is safer than exploring (-55+)
        self.reward_goal_reached = 50.0  # Very high reward - must dominate all penalties
        self.reward_wall_collision_initial = -0.2  # Mild penalty - walls are expected in a maze!
        self.reward_wall_collision_per_second = -0.02  # Mild continuous penalty
        self.reward_time_penalty = -30.0 / self.max_steps  # Time pressure (but not dominant)
        self.reward_movement_bonus = 0.02  # Meaningful bonus for moving (velocity > 0)
        self.reward_standing_still = -0.1  # STRONG penalty for not moving - prevents "stand still" local optimum
        self.distance_reward_scale = 0.0  # Disabled - Euclidean distance is misleading in maze (pulls agent into walls)

    def reset(self, seed=None, options=None):
        """Reset the environment and return initial observation."""
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        # OPRAVA: Vždy resetujeme Unity, aby sa _currentStep vynuloval
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
            decision_steps, terminal_steps = self.unity_env.get_steps(self.behavior_name)
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
            raise ValueError(f"Invalid action {action}. Action must be in {self.action_space}")

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

        # Check if Python-side max steps reached (truncation)
        max_steps_reached = self.current_step >= self.max_steps

        # Check if episode terminated
        if len(terminal_steps) > 0:
            obs = terminal_steps.obs[0][0]

            # ML-Agents provides `interrupted` flag:
            #   False = natural end (goal reached via EndEpisode())
            #   True  = truncated (Unity's MaxStep limit hit)
            unity_interrupted = bool(terminal_steps.interrupted[0])

            if unity_interrupted or max_steps_reached:
                # Timeout — either Unity's MaxStep or Python's max_steps
                terminated = False
                truncated = True
            else:
                # Agent genuinely reached the goal
                terminated = True
                truncated = False

        elif len(decision_steps) > 0:
            obs = decision_steps.obs[0][0]
            terminated = False
            truncated = max_steps_reached

        else:
            raise RuntimeError("No agents found in environment!")

        # Calculate reward in Python based on observations
        # Observation structure (from agent.cs):
        # obs[0]: relative goal X position (normalized by 5f)
        # obs[1]: relative goal Z position (normalized by 5f)
        # obs[2]: current distance to goal (normalized by 10f)
        # obs[3]: previous distance to goal (normalized by 10f)
        # obs[4]: has hit wall (1.0 = true, 0.0 = false)
        # obs[5]: has reached goal (1.0 = true, 0.0 = false) - NOTE: May be stale due to timing!
        # obs[6]: time spent in wall collision
        # obs[7]: agent velocity magnitude (normalized by 5f)
        #
        # If ray observations are enabled (default: 8 rays):
        # obs[8+i*2]: distance to obstacle in direction i (normalized, 0-1)
        # obs[9+i*2]: obstacle type (1.0 = wall, 0.5 = other, 0.0 = none)
        # Total with rays: 8 + (num_rays * 2) = 24 observations (with 8 rays)

        reward = self._calculate_reward(obs, terminated, truncated)

        # Store current observation for next step
        self.previous_obs = obs.copy()

        info = {
            "current_step": self.current_step,
            "max_steps": self.max_steps,
            "episode_length": self.current_step,
            "goal_reached": bool(len(obs) > 5 and obs[5] > 0.5),
        }

        return obs.astype(np.float32), float(reward), terminated, truncated, info

    def _calculate_reward(self, obs, terminated, truncated):
        """
        Calculate reward based on observations.

        Reward structure (optimized to prevent standing-still local optimum):
        - Goal reached: +50.0 (very high reward to dominate all penalties)
        - Movement bonus: +0.02 per step when agent is moving (velocity > 0)
        - Standing still penalty: -0.1 per step when not moving (prevents local optimum!)
        - Wall collision (initial): -0.2 (mild - walls are expected in a maze)
        - Wall collision (continuous): -0.02 per second while touching wall
        - Time penalty: -30.0 / max_steps per step
        - Distance improvement: disabled (Euclidean misleading in maze)
        """
        reward = 0.0

        # Goal detection: use the direct observation from Unity instead of
        # the terminated flag. obs[5] is _hasReachedGoal set in GoalReached()
        # BEFORE EndEpisode() is called, so the terminal observation always
        # contains the correct value (1.0 when goal is reached).
        # This is more reliable than inferring goal from terminated/truncated
        # because it comes straight from the physics trigger in Unity.
        goal_reached = len(obs) > 5 and obs[5] > 0.5
        if goal_reached:
            reward += self.reward_goal_reached

        # Movement bonus OR standing still penalty
        # obs[7] = velocity magnitude (normalized by 5f in Unity)
        if len(obs) > 7:
            velocity = obs[7] * 5.0  # Denormalize
            if velocity > 0.01:  # Agent is moving (not standing still)
                reward += self.reward_movement_bonus
            else:  # Agent is standing still - PENALIZE to prevent local optimum!
                reward += self.reward_standing_still

        # Reward shaping: scaled Δd / d (distance improvement normalized by current distance)
        # Scaled down because Euclidean distance is misleading in a maze
        # (optimal path often goes away from goal to navigate around walls)
        if len(obs) > 3 and self.previous_obs is not None and len(self.previous_obs) > 2:
            current_distance = obs[2] * 10.0  # Denormalize (was normalized by 10f)
            previous_distance = (
                self.previous_obs[2] * 10.0 if len(self.previous_obs) > 2 else current_distance
            )

            distance_delta = previous_distance - current_distance  # Δd (positive = closer)
            # Reward = scale * Δd / d - normalized distance improvement
            if current_distance > 0.1:  # Avoid division by very small numbers
                reward += self.distance_reward_scale * (distance_delta / current_distance)

        # Wall collision penalty
        # obs[4] = has_hit_wall (1.0 if touching wall, 0.0 otherwise)
        # obs[6] = time_in_wall (seconds spent touching wall)
        if len(obs) > 6 and obs[4] > 0.5:  # has_hit_wall
            # Check if this is a NEW collision (wasn't touching wall in previous step)
            was_touching_wall = (
                self.previous_obs is not None
                and len(self.previous_obs) > 4
                and self.previous_obs[4] > 0.5
            )
            
            if not was_touching_wall:
                # Initial collision penalty (-3.0)
                reward += self.reward_wall_collision_initial
            
            # Continuous penalty based on time in wall (-0.1 per second)
            time_in_wall = obs[6]  # Time spent touching wall in seconds
            reward += self.reward_wall_collision_per_second * time_in_wall

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
    unity_env_path=None,
    worker_id=0,
    no_graphics=True,
    time_scale=20.0,
    max_steps=None,
    base_port=DEFAULT_BASE_PORT,
):
    """
    Factory function to create Unity Maze Gymnasium environment.

    Args:
        unity_env_path (str): Path to Unity build executable. Use None for Unity Editor.
        worker_id (int): Worker ID for parallel training (each worker uses base_port + worker_id).
        no_graphics (bool): Run without graphics for faster training.
        time_scale (float): Speed up Unity simulation.
        max_steps (int): Override max steps per episode.
        base_port (int): Base port for ML-Agents communication (default: 5005).

    Returns:
        UnityMazeGymWrapper: Gymnasium-compatible environment.

    Example (single environment):
        >>> env = make_unity_maze_env(unity_env_path="./build/MazeAgent.exe")
        >>> obs, info = env.reset()
        >>> for _ in range(1000):
        ...     action = env.action_space.sample()
        ...     obs, reward, terminated, truncated, info = env.step(action)
        ...     if terminated or truncated:
        ...         obs, info = env.reset()
        >>> env.close()

    Example (parallel environments with SubprocVecEnv):
        >>> from stable_baselines3.common.vec_env import SubprocVecEnv
        >>> def make_env(worker_id):
        ...     def _init():
        ...         return make_unity_maze_env("./build/MazeAgent.exe", worker_id=worker_id)
        ...     return _init
        >>> env = SubprocVecEnv([make_env(i) for i in range(4)])
    """
    return UnityMazeGymWrapper(
        unity_env_path=unity_env_path,
        worker_id=worker_id,
        no_graphics=no_graphics,
        time_scale=time_scale,
        max_steps=max_steps,
        base_port=base_port,
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
