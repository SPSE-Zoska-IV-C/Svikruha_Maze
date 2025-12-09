# Unity ML-Agents Maze - Gymnasium Wrapper

This project provides a Gymnasium wrapper for your Unity ML-Agents 4.x maze environment, enabling training with popular RL libraries like Stable-Baselines3.

**Compatible with Unity ML-Agents 4.0.0+**

## 🚀 Quick Start

### 1. Install Dependencies

First, install the required Python packages:

```bash
pip install -r requirements.txt
```

**Note for ML-Agents 4.x:**
- Requires Python 3.9 - 3.11 (Python 3.12+ not yet supported)
- Requires numpy < 2.0
- If you encounter installation issues, try:
  ```bash
  pip install torch torchvision  # Install PyTorch first
  pip install -r requirements.txt
  ```

### 2. Configure Unity for Communication

In Unity Editor:
1. Open your scene (`Assets/Scenes/SampleScene.unity`)
2. Select your agent GameObject
3. In the **Behavior Parameters** component, ensure:
   - **Behavior Type** is set to **Default**
   - **Vector Observation Space Size** is **2** (relative goal X and Z)
   - **Actions** → **Discrete Branches** → **Branch 0 Size** is **4** (0=nothing, 1=forward, 2=left, 3=right)

### 3. Test the Wrapper

Test with Unity Editor (no build needed):

```bash
python gymnasium_wrapper.py
```

This will run 5 episodes with random actions to verify the connection works.

## 📚 Training with Stable-Baselines3

### Train Using PPO (Recommended)

**Option 1: Train with Unity Editor** (for testing/debugging)
```bash
python train_sb3.py --algorithm ppo --timesteps 500000
```

**Option 2: Train with Unity Build** (much faster)
```bash
# First, build your Unity project (File → Build Settings → Build)
# Then train with the executable:
python train_sb3.py --algorithm ppo --timesteps 500000 --unity-env "./build/MazeAgent.exe"
```

**Option 3: Train with Multiple Parallel Environments** (fastest)
```bash
python train_sb3.py --algorithm ppo --timesteps 1000000 --n-envs 4 --unity-env "./build/MazeAgent.exe"
```

### Train Using DQN

```bash
python train_sb3.py --algorithm dqn --timesteps 500000 --unity-env "./build/MazeAgent.exe"
```

### Evaluate a Trained Model

```bash
python train_sb3.py --mode evaluate --model-path "./models/ppo_maze_final.zip" --unity-env "./build/MazeAgent.exe"
```

## 🎮 Using the Gymnasium Wrapper Directly

You can also use the wrapper in your own Python scripts:

```python
from gymnasium_wrapper import make_unity_maze_env

# Create environment
env = make_unity_maze_env(
    unity_env_path="./build/MazeAgent.exe",  # or None for Unity Editor
    worker_id=0,
    no_graphics=True,  # Set to False to see visualization
    time_scale=20.0    # Speed up simulation
)

# Standard Gymnasium API
obs, info = env.reset()

for step in range(1000):
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

## 📊 Monitor Training Progress

View training metrics in real-time with TensorBoard:

```bash
tensorboard --logdir ./models/logs
```

Then open your browser to http://localhost:6006

## 🔧 Custom Training

You can customize training by modifying `train_sb3.py`:

```python
from gymnasium_wrapper import make_unity_maze_env
from stable_baselines3 import PPO

# Create environment
env = make_unity_maze_env(unity_env_path="./build/MazeAgent.exe")

# Create custom PPO model
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    verbose=1
)

# Train
model.learn(total_timesteps=1000000)

# Save
model.save("my_custom_model")

env.close()
```

## 🏗️ Building Unity Executable

For faster training, build your Unity project:

1. In Unity Editor: **File → Build Settings**
2. Select **Windows/Mac/Linux**
3. Click **Build**
4. Save to a folder (e.g., `./build/`)
5. Use the executable path in training scripts

## 🎯 Environment Details

### Observation Space
- **Type**: Box(2,)
- **Values**:
  - `obs[0]`: Relative goal position X (left/right)
  - `obs[1]`: Relative goal position Z (forward/back)

### Action Space
- **Type**: Discrete(4)
- **Actions**:
  - `0`: Do nothing
  - `1`: Move forward
  - `2`: Rotate left
  - `3`: Rotate right

### Rewards
- `+1.0`: Reaching the goal
- `-0.05`: Hitting a wall
- `-0.01 * dt`: Continuous penalty for staying against wall
- `0.01 * distance_delta`: Reward shaping for getting closer to goal
- `-2.0 / MaxStep`: Time penalty per step

## 🐛 Troubleshooting

### "No behavior specs found"
- Make sure Unity is running and your agent has a **Behavior Parameters** component
- Check that **Behavior Type** is set to **Default** (not Heuristic Only)

### Training is slow
- Build your Unity project instead of using the Editor
- Use multiple parallel environments with `--n-envs`
- Increase `time_scale` parameter
- Set `no_graphics=True`

### Connection timeout
- Make sure no other Unity ML-Agents environments are running
- Try a different `worker_id` parameter
- Check firewall settings

## 📝 Tips for Better Training

1. **Start with Unity Editor** for debugging, then switch to build
2. **Use PPO** for most cases (works well for discrete actions)
3. **Monitor TensorBoard** to track progress
4. **Tune hyperparameters** based on your specific maze complexity
5. **Use parallel environments** for faster convergence

## 🤝 Integration with Other RL Libraries

The Gymnasium wrapper is compatible with any library that supports the Gymnasium API:

- ✅ Stable-Baselines3
- ✅ RLlib (Ray)
- ✅ CleanRL
- ✅ Tianshou
- ✅ Custom implementations

## 📄 License

This wrapper is provided as-is for your ML-Agents project.

---

**Need help?** Check the Unity ML-Agents documentation: https://github.com/Unity-Technologies/ml-agents

