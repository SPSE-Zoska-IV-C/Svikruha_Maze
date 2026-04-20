"""
Train Unity Maze Agent using Stable-Baselines3

This script demonstrates how to train the Unity maze agent using
popular RL algorithms from Stable-Baselines3 library.

Supports parallel training with multiple Unity build instances via SubprocVecEnv.
For parallel training (n_envs > 1), you MUST provide a Unity build executable.
Unity Editor only supports single environment connections.

Usage:
    # Single environment (Unity Editor or build)
    python train_sb3.py --algorithm ppo --timesteps 100000

    # Parallel training (requires Unity build)
    python train_sb3.py --algorithm ppo --timesteps 500000 --n-envs 4 --unity-env ./build/MazeAgent.exe
"""

import glob
import os
import sys

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from gymnasium_wrapper import make_unity_maze_env
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize


def verify_unity_build(unity_env_path, n_envs):
    """
    Verify Unity build exists and is valid for the requested number of environments.

    Args:
        unity_env_path (str): Path to Unity build executable.
        n_envs (int): Number of parallel environments requested.

    Returns:
        bool: True if valid, raises error otherwise.

    Raises:
        ValueError: If parallel training requested without Unity build.
        FileNotFoundError: If Unity build path doesn't exist.
    """
    if n_envs > 1:
        if unity_env_path is None:
            raise ValueError(
                "\n" + "=" * 70 + "\n"
                "ERROR: Parallel training requires a Unity BUILD executable!\n"
                "=" * 70 + "\n"
                "Unity Editor only supports ONE environment connection at a time.\n"
                "To use parallel training (n_envs > 1), you must:\n\n"
                "1. Build your Unity project:\n"
                "   - Open Unity Editor\n"
                "   - Go to File > Build Settings\n"
                "   - Add your scene (Assets/Scenes/SampleScene.unity)\n"
                "   - Click 'Build' and save to a folder (e.g., ./build/MazeAgent/)\n\n"
                "2. Run training with the build path:\n"
                "   python train_sb3.py --algorithm ppo --n-envs 4 "
                "--unity-env ./build/MazeAgent/MazeAgent.exe\n"
                "=" * 70
            )
        if not os.path.exists(unity_env_path):
            raise FileNotFoundError(
                f"\nUnity build not found at: {unity_env_path}\n"
                f"Please build your Unity project first or check the path."
            )
        print(f"[OK] Unity build verified: {unity_env_path}")
        print(f"[OK] Parallel training with {n_envs} environments enabled")
    else:
        if unity_env_path is None:
            print("[INFO] Using Unity Editor (single environment mode)")
            print("[INFO] Make sure Unity Editor is running with the scene open!")
        else:
            if not os.path.exists(unity_env_path):
                raise FileNotFoundError(f"Unity build not found at: {unity_env_path}")
            print(f"[OK] Unity build: {unity_env_path}")

    return True


def get_latest_log_dir(log_base_path, algorithm_prefix):
    """
    Find the latest tensorboard log directory for the given algorithm.

    Args:
        log_base_path: Base path where tensorboard logs are stored
        algorithm_prefix: Algorithm prefix (e.g., "PPO", "SAC")

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
        algorithm_prefix: Algorithm prefix (e.g., "PPO", "SAC")
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


def make_env(unity_env_path, worker_id, force_continuous=False, time_scale=2.5, startup_delay=0.0, max_steps=2000):
    """
    Create a single environment instance for vectorized training.

    Args:
        unity_env_path (str): Path to Unity build executable or None for Editor.
        worker_id (int): Worker ID for parallel environments (affects port assignment).
        force_continuous (bool): Wrap discrete actions as continuous (for SAC).
        time_scale (float): Unity simulation speed multiplier.
        startup_delay (float): Delay before creating environment (for staggered parallel startup).
        max_steps (int): Maximum steps per episode (default: 2000).

    Returns:
        Callable: Function that creates the environment when called.
    """
    import time as time_module  # Local import to avoid conflicts

    def _init():
        # Staggered startup to avoid port conflicts during parallel initialization
        if startup_delay > 0:
            time_module.sleep(startup_delay)

        env = make_unity_maze_env(
            unity_env_path=unity_env_path,
            worker_id=worker_id,
            no_graphics=True,
            time_scale=time_scale,
            max_steps=max_steps,
        )
        if force_continuous and isinstance(env.action_space, spaces.Discrete):
            env = DiscreteToBoxWrapper(env)
        env = Monitor(env)
        return env

    return _init


def create_parallel_envs(unity_env_path, n_envs, time_scale=20.0, force_continuous=False, max_steps=2000):
    """
    Create parallel vectorized environments with proper startup sequencing.

    Args:
        unity_env_path (str): Path to Unity build executable.
        n_envs (int): Number of parallel environments.
        time_scale (float): Unity simulation speed multiplier.
        force_continuous (bool): Wrap discrete actions as continuous (for SAC).
        max_steps (int): Maximum steps per episode (default: 2000).

    Returns:
        SubprocVecEnv or DummyVecEnv: Vectorized environment.
    """
    print(f"  Max steps per episode: {max_steps}")
    if n_envs > 1:
        print(f"Creating {n_envs} parallel Unity environments...")
        print(f"  Time scale: {time_scale}x")
        print(f"  This may take a moment as Unity instances start up...")

        # Staggered startup: delay each worker slightly to avoid port conflicts
        startup_delay_per_worker = 1.0  # seconds between each worker start

        env_fns = [
            make_env(
                unity_env_path,
                worker_id=i,
                force_continuous=force_continuous,
                time_scale=time_scale,
                startup_delay=i * startup_delay_per_worker,
                max_steps=max_steps,
            )
            for i in range(n_envs)
        ]
        env = SubprocVecEnv(env_fns)
        print(f"[OK] All {n_envs} environments created successfully!")
    else:
        env = DummyVecEnv(
            [make_env(unity_env_path, 0, force_continuous=force_continuous, time_scale=time_scale, max_steps=max_steps)]
        )
    return env


def train_ppo(
    unity_env_path=None,
    total_timesteps=2000000,
    n_envs=1,
    save_dir="./models",
    load_path=None,
    max_steps=2000,
):
    """
    Train agent using PPO (Proximal Policy Optimization).

    Args:
        unity_env_path (str): Path to Unity build. None for Unity Editor.
        total_timesteps (int): Total training steps.
        n_envs (int): Number of parallel environments (requires Unity build for n_envs > 1).
        save_dir (str): Directory to save models.
        load_path (str): Path to a saved model to continue training from.
        max_steps (int): Maximum steps per episode (default: 2000).
    """
    print(f"\n{'='*60}")
    print(f"PPO Training Configuration")
    print(f"{'='*60}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Parallel environments: {n_envs}")

    # Verify Unity build for parallel training
    verify_unity_build(unity_env_path, n_envs)

    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/logs", exist_ok=True)

    # Create vectorized environments with faster time_scale for PPO
    ppo_time_scale = 20.0  # Faster simulation for PPO training
    print(f"Time scale: {ppo_time_scale}x")
    base_env = create_parallel_envs(unity_env_path, n_envs, time_scale=ppo_time_scale, max_steps=max_steps)

    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(10000 // n_envs, 1000),
        save_path=f"{save_dir}/checkpoints",
        name_prefix="ppo_maze",
    )

    # Create PPO model with improved hyperparameters
    # Fixed learning rate for more stable training
    FIXED_LR = 3e-4

    # Force CPU training for PPO (better stability, GPU overhead not worth it for small networks)
    device = "cpu"
    print(f"Training device: {device}")

    # PPO uses raw observations directly (no VecNormalize)
    # Observations are already normalized in Unity (positions /5f, distances /10f, rays 0-1)
    env = base_env

    if load_path is not None and os.path.exists(load_path):
        print(f"Loading model from {load_path}...")
        # Load model temporarily to check observation space
        temp_model = PPO.load(load_path)
        model_obs_shape = temp_model.observation_space.shape
        env_obs_shape = env.observation_space.shape
        
        if model_obs_shape != env_obs_shape:
            print(f"\n{'='*70}")
            print(f"WARNING: Observation space mismatch!")
            print(f"  Model expects:     {model_obs_shape}")
            print(f"  Environment has:   {env_obs_shape}")
            print(f"{'='*70}")
            print(f"You cannot continue training a model with different observation space.")
            print(f"Options:")
            print(f"  1. Start fresh training (remove --model-path)")
            print(f"  2. Use a model trained with {env_obs_shape[0]} observations")
            print(f"{'='*70}\n")
            env.close()
            raise ValueError(f"Observation space mismatch: model {model_obs_shape} vs env {env_obs_shape}")
        
        del temp_model  # Clean up temporary model
        
        # Override learning rate with fixed value for stability
        model = PPO.load(load_path, env=env, device=device, custom_objects={"learning_rate": FIXED_LR})
        print(f"Model loaded with fixed learning rate: {FIXED_LR}, device: {device}")
        print(f"Observation space verified: {model_obs_shape}")
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=FIXED_LR,  # Fixed learning rate (no decay)
            n_steps=2048,  # Steps per update PER environment (SB3 handles n_envs internally)
            batch_size=256,  # Larger batch for stability (2048/256 = 8 minibatches)
            n_epochs=10,  # More epochs per update
            gamma=0.99,  # Higher discount for longer-term planning in maze
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.02,  # Higher entropy to encourage exploration and prevent standing-still convergence
            vf_coef=0.5,  # Value function coefficient
            max_grad_norm=0.5,  # Gradient clipping for stability
            policy_kwargs=dict(
                net_arch=dict(pi=[128, 128], vf=[128, 128])  # Separate networks, larger
            ),
            device=device,  # Force CPU training
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


def train_sac(
    unity_env_path=None,
    total_timesteps=500000,
    n_envs=1,
    save_dir="./models",
    load_path=None,
    max_steps=2000,
):
    """
    Train agent using SAC (Soft Actor-Critic) with GPU support.

    Note: SAC is an off-policy algorithm and doesn't benefit as much from
    parallel environments as on-policy algorithms (PPO). However,
    parallel envs can still speed up data collection.

    Args:
        unity_env_path (str): Path to Unity build. None for Unity Editor.
        total_timesteps (int): Total training steps.
        n_envs (int): Number of parallel environments (requires Unity build for n_envs > 1).
        save_dir (str): Directory to save models.
        load_path (str): Path to a saved model to continue training from.
        max_steps (int): Maximum steps per episode (default: 2000).
    """
    print(f"\n{'='*60}")
    print(f"SAC Training Configuration")
    print(f"{'='*60}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Parallel environments: {n_envs}")

    if n_envs > 1:
        print("[NOTE] SAC is off-policy; parallel envs provide moderate speedup")

    # Check for GPU availability
    if torch.cuda.is_available():
        device = "cuda"
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"Training on GPU (CUDA)")
    else:
        device = "cpu"
        print("No GPU detected, training on CPU")

    # Verify Unity build for parallel training
    verify_unity_build(unity_env_path, n_envs)

    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/logs", exist_ok=True)

    # Create environment (force continuous for SAC)
    sac_time_scale = 20.0
    print(f"Time scale: {sac_time_scale}x")
    base_env = create_parallel_envs(
        unity_env_path, n_envs, time_scale=sac_time_scale, force_continuous=True, max_steps=max_steps
    )

    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(10000 // n_envs, 1000),
        save_path=f"{save_dir}/checkpoints",
        name_prefix="sac_maze",
    )

    # Create SAC model with GPU support and VecNormalize wrapper
    if load_path is not None and os.path.exists(load_path):
        print(f"Loading model from {load_path}...")
        # Check observation space compatibility first
        temp_model = SAC.load(load_path)
        model_obs_shape = temp_model.observation_space.shape
        env_obs_shape = base_env.observation_space.shape
        
        if model_obs_shape != env_obs_shape:
            print(f"\n{'='*70}")
            print(f"WARNING: Observation space mismatch!")
            print(f"  Model expects:     {model_obs_shape}")
            print(f"  Environment has:   {env_obs_shape}")
            print(f"{'='*70}")
            base_env.close()
            raise ValueError(f"Observation space mismatch: model {model_obs_shape} vs env {env_obs_shape}")
        
        del temp_model
        print(f"Observation space verified: {model_obs_shape}")
        
        # Try to load VecNormalize stats if they exist
        vec_normalize_path = load_path.replace(".zip", "_vecnormalize.pkl")
        if not vec_normalize_path.endswith("_vecnormalize.pkl"):
            vec_normalize_path = f"{load_path}_vecnormalize.pkl"
        if os.path.exists(vec_normalize_path):
            print(f"Loading VecNormalize stats from {vec_normalize_path}...")
            env = VecNormalize.load(vec_normalize_path, base_env)
        else:
            # No saved stats, create new VecNormalize
            env = VecNormalize(base_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        model = SAC.load(load_path, env=env, device=device)
    else:
        # Add VecNormalize for better training stability
        env = VecNormalize(base_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
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
            device=device,  # Use GPU if available
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

    # Save VecNormalize statistics
    vec_normalize_path = f"{save_dir}/checkpoints/{checkpoint_name}_vecnormalize.pkl"
    env.save(vec_normalize_path)
    print(f"Final checkpoint saved to {final_model_path} ({actual_steps} steps)")
    print(f"VecNormalize stats saved to {vec_normalize_path}")

    env.close()
    return model


def evaluate_model(model_path, unity_env_path=None, n_episodes=10, max_steps=2000):
    """
    Evaluate a trained model.

    Args:
        model_path (str): Path to saved model.
        unity_env_path (str): Path to Unity build.
        n_episodes (int): Number of episodes to evaluate.
        max_steps (int): Maximum steps per episode (default: 2000).
    """
    print(f"Evaluating model: {model_path}")

    # Determine if we need continuous action space (for SAC)
    is_sac = "sac" in model_path.lower()

    # Create environment with matching max_steps
    base_env = make_unity_maze_env(unity_env_path=unity_env_path, no_graphics=False, time_scale=1.0, max_steps=max_steps)

    if is_sac and isinstance(base_env.action_space, spaces.Discrete):
        print("Wrapping environment for SAC (Discrete -> Box)")
        base_env = DiscreteToBoxWrapper(base_env)

    vec_env = DummyVecEnv([lambda: base_env])

    # PPO models don't use VecNormalize; SAC does
    is_ppo = "ppo" in model_path.lower()

    if is_ppo:
        env = vec_env
        print("[OK] PPO evaluation without VecNormalize")
    else:
        vec_normalize_path = model_path.replace(".zip", "_vecnormalize.pkl")
        if not vec_normalize_path.endswith("_vecnormalize.pkl"):
            vec_normalize_path = f"{model_path}_vecnormalize.pkl"

        if os.path.exists(vec_normalize_path):
            print(f"Loading VecNormalize stats from {vec_normalize_path}...")
            env = VecNormalize.load(vec_normalize_path, vec_env)
        else:
            print("Warning: VecNormalize stats not found!")
            print("  The agent may perform poorly without normalization stats.")
            env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

        env.training = False
        env.norm_reward = False

    # Load the correct model class
    if is_ppo:
        model = PPO.load(model_path)
    elif "sac" in model_path.lower():
        model = SAC.load(model_path)
    else:
        raise ValueError(f"Unknown model type in {model_path}. Supported: PPO, SAC")

    # Run evaluation (all models use VecEnv via VecNormalize now)
    episode_rewards = []
    episode_lengths = []

    for episode in range(n_episodes):
        # VecEnv reset returns only obs (no info)
        obs = env.reset()
        episode_reward = 0
        steps = 0

        while True:
            action, _states = model.predict(obs, deterministic=True)
            
            # VecEnv step returns (obs, reward, done, info) - old Gym API
            obs, reward, done, info = env.step(action)
            # Extract scalar values from arrays
            episode_reward += reward[0]
            steps += 1

            if done[0]:
                episode_rewards.append(episode_reward)
                episode_lengths.append(steps)
                print(f"Episode {episode + 1}: Steps = {steps}, Reward = {episode_reward:.2f}")
                break

    env.close()

    # Print statistics
    print("\n--- Evaluation Results ---")
    print(f"Mean Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Mean Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")

    return episode_rewards, episode_lengths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train Unity Maze Agent with Stable-Baselines3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single environment with Unity Editor
  python train_sb3.py --algorithm ppo --timesteps 100000

  # Parallel training with Unity build (4 environments)
  python train_sb3.py --algorithm ppo --timesteps 500000 --n-envs 4 --unity-env ./build/MazeAgent.exe

  # Continue training from checkpoint
  python train_sb3.py --algorithm ppo --timesteps 200000 --model-path ./models/checkpoints/ppo_maze_100000_steps.zip

Note: Parallel training (--n-envs > 1) requires a Unity BUILD executable.
      Unity Editor only supports single environment connections.
        """,
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
        choices=["ppo", "sac"],
        help="RL algorithm to use",
    )
    parser.add_argument(
        "--unity-env",
        type=str,
        default=None,
        help="Path to Unity build executable (None for Unity Editor). "
        "REQUIRED for parallel training (--n-envs > 1)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=30000,
        help="Total timesteps for training (default: 30000)",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help="Number of parallel environments. Requires Unity build for n_envs > 1. "
        "Recommended: 2-8 depending on CPU cores. (default: 1)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model for evaluation OR continuation of training",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./models",
        help="Directory to save models (default: ./models)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=2000,
        help="Maximum steps per episode. Increase for larger mazes (default: 2000)",
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
                max_steps=args.max_steps,
            )
        elif args.algorithm == "sac":
            train_sac(
                unity_env_path=args.unity_env,
                total_timesteps=args.timesteps,
                n_envs=args.n_envs,
                save_dir=args.save_dir,
                load_path=args.model_path,
                max_steps=args.max_steps,
            )

    elif args.mode == "evaluate":
        if args.model_path is None:
            print("Error: --model-path required for evaluation mode")
            sys.exit(1)
        else:
            evaluate_model(model_path=args.model_path, unity_env_path=args.unity_env, n_episodes=10, max_steps=args.max_steps)
