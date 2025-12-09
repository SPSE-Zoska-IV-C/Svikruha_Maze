"""
Train Unity Maze Agent using Stable-Baselines3

This script demonstrates how to train the Unity maze agent using
popular RL algorithms from Stable-Baselines3 library.
"""

import glob
import os

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import A2C, DQN, PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from gymnasium_wrapper import make_unity_maze_env


def get_latest_log_dir(log_base_path, algorithm_prefix):
    """
    Find the latest tensorboard log directory for the given algorithm.

    Args:
        log_base_path: Base path where tensorboard logs are stored
        algorithm_prefix: Algorithm prefix (e.g., "PPO", "DQN", "A2C", "SAC")

    Returns:
        Name of the latest log directory (e.g., "PPO_12")
    """
    pattern = os.path.join(log_base_path, f"{algorithm_prefix}_*")
    log_dirs = glob.glob(pattern)

    if not log_dirs:
        return f"{algorithm_prefix}_1"

    # Sort by modification time to get the most recent
    log_dirs.sort(key=os.path.getmtime)
    return os.path.basename(log_dirs[-1])


def get_final_checkpoint_name(log_base_path, algorithm_prefix, total_steps):
    """
    Generate final checkpoint name based on latest log directory and total steps.

    Args:
        log_base_path: Base path where tensorboard logs are stored
        algorithm_prefix: Algorithm prefix (e.g., "PPO", "DQN", "A2C", "SAC")
        total_steps: Total number of steps trained

    Returns:
        Checkpoint name (e.g., "PPO_12_30000_steps")
    """
    log_name = get_latest_log_dir(log_base_path, algorithm_prefix)
    return f"{log_name}_{total_steps}_steps"


class DiscreteToBoxWrapper(gym.ActionWrapper):
    """
    Wraps a discrete environment to accept continuous actions (Box).
    Useful for using algorithms like SAC on discrete environments.
    """

    def __init__(self, env):
        super().__init__(env)
        # Assumes a single discrete action dimension
        self.n_actions = env.action_space.n
        # Create a continuous action space [-1, 1]
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

    def action(self, action):
        # Map [-1, 1] to [0, n_actions-1]
        # Normalize from [-1, 1] to [0, 1]
        # Clip input to ensure it's within bounds
        act = np.clip(action[0], -1, 1)
        normalized_action = (act + 1) / 2.0
        # Scale to [0, n_actions]
        scaled_action = normalized_action * self.n_actions
        # Floor to get integer index, clamp to [0, n_actions-1]
        discrete_action = int(np.clip(np.floor(scaled_action), 0, self.n_actions - 1))
        return discrete_action


def make_env(unity_env_path, worker_id, force_continuous=False):
    """Create a single environment instance."""

    def _init():
        env = make_unity_maze_env(
            unity_env_path=unity_env_path,
            worker_id=worker_id,
            no_graphics=True,
            time_scale=5.0,
            max_steps=10000,
        )
        if force_continuous and isinstance(env.action_space, spaces.Discrete):
            env = DiscreteToBoxWrapper(env)
        env = Monitor(env)
        return env

    return _init


def train_ppo(
    unity_env_path=None,
    total_timesteps=2000000,
    n_envs=1,
    save_dir="./models",
    load_path=None,
):
    """
    Train agent using PPO (Proximal Policy Optimization).

    Args:
        unity_env_path (str): Path to Unity build. None for Unity Editor.
        total_timesteps (int): Total training steps.
        n_envs (int): Number of parallel environments.
        save_dir (str): Directory to save models.
        load_path (str): Path to a saved model to continue training from.
    """
    print(f"Training PPO agent for {total_timesteps} timesteps...")

    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/logs", exist_ok=True)

    # Create vectorized environments
    if n_envs > 1:
        print(f"Creating {n_envs} parallel environments...")
        env = SubprocVecEnv([make_env(unity_env_path, i) for i in range(n_envs)])
    else:
        env = DummyVecEnv([make_env(unity_env_path, 0)])

    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000 // n_envs,
        save_path=f"{save_dir}/checkpoints",
        name_prefix="ppo_maze",
    )

    # Create PPO model with improved hyperparameters
    # Learning rate schedule: linear decay from 3e-4 to 1e-5
    def lr_schedule(progress_remaining):
        """Linear learning rate decay."""
        return 3e-4 * progress_remaining + 1e-5 * (1 - progress_remaining)

    if load_path is not None and os.path.exists(load_path):
        print(f"Loading model from {load_path}...")
        # Override learning rate with the schedule for the new training phase
        model = PPO.load(
            load_path, env=env, custom_objects={"learning_rate": lr_schedule}
        )
        print("Model loaded with reset learning rate schedule.")
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=lr_schedule,  # Learning rate with decay
            n_steps=2048 // n_envs,  # More steps for better gradient estimates
            batch_size=64,  # Larger batch for stability
            n_epochs=10,  # More epochs per update
            gamma=0.99,  # Higher discount for longer-term planning in maze
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,  # Slightly higher entropy for exploration
            vf_coef=0.5,  # Value function coefficient
            max_grad_norm=0.5,  # Gradient clipping for stability
            policy_kwargs=dict(
                net_arch=dict(pi=[128, 128], vf=[128, 128])  # Separate networks, larger
            ),
            verbose=1,
            tensorboard_log=f"{save_dir}/logs",
        )

    # Train the model
    print("Starting training...")
    print("Press Ctrl+C in PowerShell to stop training and save checkpoint.")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            progress_bar=False,  # Disabled progress bar to avoid dependency issues
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user (Ctrl+C)!")

    # Save final checkpoint with log name and actual steps trained
    actual_steps = model.num_timesteps
    checkpoint_name = get_final_checkpoint_name(f"{save_dir}/logs", "PPO", actual_steps)
    final_model_path = f"{save_dir}/checkpoints/{checkpoint_name}"
    model.save(final_model_path)
    print(f"Final checkpoint saved to {final_model_path} ({actual_steps} steps)")

    env.close()
    return model


def train_dqn(
    unity_env_path=None, total_timesteps=500000, save_dir="./models", load_path=None
):
    """
    Train agent using DQN (Deep Q-Network).

    Args:
        unity_env_path (str): Path to Unity build. None for Unity Editor.
        total_timesteps (int): Total training steps.
        save_dir (str): Directory to save models.
        load_path (str): Path to a saved model to continue training from.
    """
    print(f"Training DQN agent for {total_timesteps} timesteps...")

    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/logs", exist_ok=True)

    # Create environment
    env = DummyVecEnv([make_env(unity_env_path, 0)])

    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000, save_path=f"{save_dir}/checkpoints", name_prefix="dqn_maze"
    )

    # Create DQN model
    if load_path is not None and os.path.exists(load_path):
        print(f"Loading model from {load_path}...")
        model = DQN.load(load_path, env=env)
    else:
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=1e-4,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=32,
            tau=1.0,
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=1000,
            exploration_fraction=0.1,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            verbose=1,
            tensorboard_log=f"{save_dir}/logs",
        )

    # Train the model
    print("Starting training...")
    print("Press Ctrl+C in PowerShell to stop training and save checkpoint.")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            progress_bar=False,  # Disabled progress bar to avoid dependency issues
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user (Ctrl+C)!")

    # Save final checkpoint with log name and actual steps trained
    actual_steps = model.num_timesteps
    checkpoint_name = get_final_checkpoint_name(f"{save_dir}/logs", "DQN", actual_steps)
    final_model_path = f"{save_dir}/checkpoints/{checkpoint_name}"
    model.save(final_model_path)
    print(f"Final checkpoint saved to {final_model_path} ({actual_steps} steps)")

    env.close()
    return model


def train_sac(
    unity_env_path=None, total_timesteps=500000, save_dir="./models", load_path=None
):
    """
    Train agent using SAC (Soft Actor-Critic).

    Args:
        unity_env_path (str): Path to Unity build. None for Unity Editor.
        total_timesteps (int): Total training steps.
        save_dir (str): Directory to save models.
        load_path (str): Path to a saved model to continue training from.
    """
    print(f"Training SAC agent for {total_timesteps} timesteps...")

    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/logs", exist_ok=True)

    # Create environment (force continuous for SAC)
    env = DummyVecEnv([make_env(unity_env_path, 0, force_continuous=True)])

    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000, save_path=f"{save_dir}/checkpoints", name_prefix="sac_maze"
    )

    # Create SAC model
    if load_path is not None and os.path.exists(load_path):
        print(f"Loading model from {load_path}...")
        model = SAC.load(load_path, env=env)
    else:
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=100000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",
            verbose=1,
            tensorboard_log=f"{save_dir}/logs",
        )

    # Train the model
    print("Starting training...")
    print("Press Ctrl+C in PowerShell to stop training and save checkpoint.")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            progress_bar=False,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user (Ctrl+C)!")

    # Save final checkpoint with log name and actual steps trained
    actual_steps = model.num_timesteps
    checkpoint_name = get_final_checkpoint_name(f"{save_dir}/logs", "SAC", actual_steps)
    final_model_path = f"{save_dir}/checkpoints/{checkpoint_name}"
    model.save(final_model_path)
    print(f"Final checkpoint saved to {final_model_path} ({actual_steps} steps)")

    env.close()
    return model


def train_a2c(
    unity_env_path=None,
    total_timesteps=500000,
    n_envs=1,
    save_dir="./models",
    load_path=None,
):
    """
    Train agent using A2C (Advantage Actor-Critic).

    A2C is a synchronous version of A3C, good for environments where
    parallelization provides significant speedup.

    Args:
        unity_env_path (str): Path to Unity build. None for Unity Editor.
        total_timesteps (int): Total training steps.
        n_envs (int): Number of parallel environments.
        save_dir (str): Directory to save models.
        load_path (str): Path to a saved model to continue training from.
    """
    print(f"Training A2C agent for {total_timesteps} timesteps...")

    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/logs", exist_ok=True)

    # Create vectorized environments
    if n_envs > 1:
        print(f"Creating {n_envs} parallel environments...")
        env = SubprocVecEnv([make_env(unity_env_path, i) for i in range(n_envs)])
    else:
        env = DummyVecEnv([make_env(unity_env_path, 0)])

    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000 // n_envs,
        save_path=f"{save_dir}/checkpoints",
        name_prefix="a2c_maze",
    )

    # Create A2C model
    # Learning rate schedule for A2C
    def lr_schedule(progress_remaining):
        """Linear learning rate decay."""
        return 7e-4 * progress_remaining + 1e-5 * (1 - progress_remaining)

    if load_path is not None and os.path.exists(load_path):
        print(f"Loading model from {load_path}...")
        model = A2C.load(
            load_path, env=env, custom_objects={"learning_rate": lr_schedule}
        )
        print("Model loaded with reset learning rate schedule.")
    else:
        model = A2C(
            "MlpPolicy",
            env,
            learning_rate=lr_schedule,  # Standard A2C learning rate with decay
            n_steps=5,  # A2C typically uses fewer steps than PPO
            gamma=0.99,  # Discount factor for long-term rewards
            gae_lambda=1.0,  # GAE lambda (1.0 = no GAE, just returns)
            ent_coef=0.01,  # Entropy coefficient for exploration
            vf_coef=0.5,  # Value function loss coefficient
            max_grad_norm=0.5,  # Gradient clipping for stability
            use_rms_prop=True,  # Use RMSprop optimizer (standard for A2C)
            normalize_advantage=True,  # Normalize advantages for stability
            policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
            verbose=1,
            tensorboard_log=f"{save_dir}/logs",
        )

    # Train the model
    print("Starting training...")
    print("Press Ctrl+C in PowerShell to stop training and save checkpoint.")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            progress_bar=False,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user (Ctrl+C)!")

    # Save final checkpoint with log name and actual steps trained
    actual_steps = model.num_timesteps
    checkpoint_name = get_final_checkpoint_name(f"{save_dir}/logs", "A2C", actual_steps)
    final_model_path = f"{save_dir}/checkpoints/{checkpoint_name}"
    model.save(final_model_path)
    print(f"Final checkpoint saved to {final_model_path} ({actual_steps} steps)")

    env.close()
    return model


def evaluate_model(model_path, unity_env_path=None, n_episodes=10):
    """
    Evaluate a trained model.

    Args:
        model_path (str): Path to saved model.
        unity_env_path (str): Path to Unity build.
        n_episodes (int): Number of episodes to evaluate.
    """
    print(f"Evaluating model: {model_path}")

    # Determine if we need continuous action space (for SAC)
    is_sac = "sac" in model_path.lower()

    # Create environment
    env = make_unity_maze_env(
        unity_env_path=unity_env_path, no_graphics=False, time_scale=1.0
    )

    if is_sac and isinstance(env.action_space, spaces.Discrete):
        print("Wrapping environment for SAC (Discrete -> Box)")
        env = DiscreteToBoxWrapper(env)

    # Load model
    if "ppo" in model_path.lower():
        model = PPO.load(model_path)
    elif "dqn" in model_path.lower():
        model = DQN.load(model_path)
    elif "a2c" in model_path.lower():
        model = A2C.load(model_path)
    elif "sac" in model_path.lower():
        model = SAC.load(model_path)
    else:
        raise ValueError(f"Unknown model type in {model_path}")

    # Run evaluation
    episode_rewards = []
    episode_lengths = []

    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        steps = 0

        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            steps += 1

            if terminated or truncated:
                episode_rewards.append(episode_reward)
                episode_lengths.append(steps)
                print(
                    f"Episode {episode + 1}: Steps = {steps}, Reward = {episode_reward:.2f}"
                )
                break

    env.close()

    # Print statistics
    print("\n--- Evaluation Results ---")
    print(
        f"Mean Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}"
    )
    print(
        f"Mean Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}"
    )

    return episode_rewards, episode_lengths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train Unity Maze Agent with Stable-Baselines3"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "evaluate"],
        help="Mode: train or evaluate",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="ppo",
        choices=["ppo", "dqn", "a2c", "sac"],
        help="RL algorithm to use",
    )
    parser.add_argument(
        "--unity-env",
        type=str,
        default=None,
        help="Path to Unity build executable (None for Unity Editor)",
    )
    parser.add_argument(
        "--timesteps", type=int, default=30000, help="Total timesteps for training"
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help="Number of parallel environments (PPO only)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model for evaluation OR continuation of training",
    )
    parser.add_argument(
        "--save-dir", type=str, default="./models", help="Directory to save models"
    )

    args = parser.parse_args()

    if args.mode == "train":
        if args.algorithm == "ppo":
            train_ppo(
                unity_env_path=args.unity_env,
                total_timesteps=args.timesteps,
                n_envs=args.n_envs,
                save_dir=args.save_dir,
                load_path=args.model_path,
            )
        elif args.algorithm == "dqn":
            train_dqn(
                unity_env_path=args.unity_env,
                total_timesteps=args.timesteps,
                save_dir=args.save_dir,
                load_path=args.model_path,
            )
        elif args.algorithm == "sac":
            train_sac(
                unity_env_path=args.unity_env,
                total_timesteps=args.timesteps,
                save_dir=args.save_dir,
                load_path=args.model_path,
            )
        elif args.algorithm == "a2c":
            train_a2c(
                unity_env_path=args.unity_env,
                total_timesteps=args.timesteps,
                n_envs=args.n_envs,
                save_dir=args.save_dir,
                load_path=args.model_path,
            )

    elif args.mode == "evaluate":
        if args.model_path is None:
            print("Error: --model-path required for evaluation mode")
        else:
            evaluate_model(
                model_path=args.model_path, unity_env_path=args.unity_env, n_episodes=10
            )
