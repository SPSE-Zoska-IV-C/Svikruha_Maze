"""
Watch Agent - Visualize trained agent in Unity

Use this script to watch your trained agent play and record videos.
Runs at time_scale=1.0 for smooth visualization.

Usage:
    python watch_agent.py                           # Watch latest checkpoint
    python watch_agent.py --model path/to/model     # Watch specific model
    python watch_agent.py --episodes 5              # Watch 5 episodes
    python watch_agent.py --time-scale 0.5          # Slow motion for detailed analysis
"""

import argparse
import glob
import os
import time

import numpy as np
from gymnasium import spaces
from gymnasium_wrapper import make_unity_maze_env


class DiscreteToBoxWrapper:
    """Simple wrapper for SAC models that expect continuous actions."""

    def __init__(self, env):
        self.env = env
        self.n_actions = env.action_space.n
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = env.observation_space

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        act = np.clip(action[0], -1, 1)
        normalized = (act + 1) / 2.0
        discrete = int(np.clip(np.floor(normalized * self.n_actions), 0, self.n_actions - 1))
        return self.env.step(discrete)

    def close(self):
        self.env.close()


def find_latest_checkpoint(models_dir="./models/checkpoints"):
    """Find the most recently modified checkpoint."""
    pattern = os.path.join(models_dir, "*.zip")
    checkpoints = glob.glob(pattern)

    if not checkpoints:
        return None

    # Sort by modification time
    checkpoints.sort(key=os.path.getmtime, reverse=True)
    return checkpoints[0]


def load_model(model_path):
    """Load model based on filename."""
    from stable_baselines3 import A2C, DQN, PPO, SAC

    model_path_lower = model_path.lower()

    if "ppo" in model_path_lower:
        return PPO.load(model_path), "PPO"
    elif "dqn" in model_path_lower:
        return DQN.load(model_path), "DQN"
    elif "a2c" in model_path_lower:
        return A2C.load(model_path), "A2C"
    elif "sac" in model_path_lower:
        return SAC.load(model_path), "SAC"
    else:
        # Try to load as PPO by default
        try:
            return PPO.load(model_path), "PPO"
        except:
            raise ValueError(f"Cannot determine model type from: {model_path}")


def watch_agent(model_path, n_episodes=10, time_scale=1.0, unity_env_path=None):
    """
    Watch a trained agent play.

    Args:
        model_path: Path to saved model (.zip file)
        n_episodes: Number of episodes to watch
        time_scale: Unity time scale (1.0 = real-time, 0.5 = slow motion)
        unity_env_path: Path to Unity executable (None = Editor)
    """
    print("=" * 60)
    print("🎬 WATCH AGENT - Visualization Mode")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Episodes: {n_episodes}")
    print(f"Time Scale: {time_scale}x")
    print()
    print("💡 TIP: Use OBS Studio or Unity Recorder to capture video")
    print("💡 TIP: Use --time-scale 0.5 for slow motion analysis")
    print("=" * 60)
    print()

    # Load model
    model, algo_name = load_model(model_path)
    print(f"✅ Loaded {algo_name} model")

    # Create environment with graphics enabled
    print("🔌 Connecting to Unity... (make sure Unity Editor is running)")
    env = make_unity_maze_env(
        unity_env_path=unity_env_path,
        no_graphics=False,  # Graphics ON for visualization
        time_scale=time_scale,  # Smooth playback
    )

    # Wrap for SAC if needed
    is_sac = algo_name == "SAC"
    if is_sac and isinstance(env.action_space, spaces.Discrete):
        print("🔄 Wrapping environment for SAC")
        env = DiscreteToBoxWrapper(env)

    print(f"✅ Connected to Unity")
    print()

    # Statistics
    episode_rewards = []
    episode_lengths = []
    goals_reached = 0

    print("🎮 Starting visualization...")
    print("-" * 40)

    try:
        for episode in range(n_episodes):
            obs, info = env.reset()
            episode_reward = 0
            steps = 0

            while True:
                # Get action from trained model
                action, _ = model.predict(obs, deterministic=True)

                # Take step
                obs, reward, terminated, truncated, info = env.step(action)

                episode_reward += reward
                steps += 1

                # Check if episode ended
                if terminated or truncated:
                    episode_rewards.append(episode_reward)
                    episode_lengths.append(steps)

                    # Check if goal was reached (positive final reward usually means goal)
                    reached_goal = reward > 1.0  # Goal reward is typically large
                    if reached_goal:
                        goals_reached += 1
                        status = "🎯 GOAL!"
                    else:
                        status = "⏱️ Timeout" if truncated else "❌ Failed"

                    print(
                        f"Episode {episode + 1:3d}: {status:12s} | Steps: {steps:5d} | Reward: {episode_reward:8.2f}"
                    )
                    break

        print("-" * 40)
        print()

        # Print final statistics
        print("📊 STATISTICS")
        print("=" * 40)
        print(f"Episodes:      {n_episodes}")
        print(f"Goals Reached: {goals_reached}/{n_episodes} ({100*goals_reached/n_episodes:.1f}%)")
        print(f"Mean Reward:   {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"Mean Steps:    {np.mean(episode_lengths):.0f} ± {np.std(episode_lengths):.0f}")
        print(f"Best Episode:  {max(episode_rewards):.2f} reward")
        print(f"Worst Episode: {min(episode_rewards):.2f} reward")
        print("=" * 40)

    except KeyboardInterrupt:
        print("\n\n⏹️ Stopped by user (Ctrl+C)")

    finally:
        env.close()
        print("\n✅ Unity environment closed")


def main():
    parser = argparse.ArgumentParser(
        description="Watch trained agent play in Unity (for recording videos)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python watch_agent.py                              # Watch latest checkpoint
  python watch_agent.py --model ./models/best.zip   # Watch specific model
  python watch_agent.py --episodes 3 --time-scale 0.5  # 3 episodes in slow motion
        """,
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="Path to model file (.zip). If not specified, uses latest checkpoint.",
    )

    parser.add_argument(
        "--episodes", "-e", type=int, default=5, help="Number of episodes to watch (default: 5)"
    )

    parser.add_argument(
        "--time-scale",
        "-t",
        type=float,
        default=1.0,
        help="Unity time scale. 1.0=real-time, 0.5=slow motion, 2.0=fast (default: 1.0)",
    )

    parser.add_argument(
        "--unity-env",
        type=str,
        default=None,
        help="Path to Unity executable. None=connect to Editor (default: None)",
    )

    args = parser.parse_args()

    # Find model
    if args.model is None:
        model_path = find_latest_checkpoint()
        if model_path is None:
            print("❌ No checkpoints found in ./models/checkpoints/")
            print("   Train a model first with: python train_sb3.py --mode train")
            return
        print(f"📁 Using latest checkpoint: {model_path}")
    else:
        model_path = args.model
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            return

    # Watch agent
    watch_agent(
        model_path=model_path,
        n_episodes=args.episodes,
        time_scale=args.time_scale,
        unity_env_path=args.unity_env,
    )


if __name__ == "__main__":
    main()
