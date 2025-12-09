# ML-Agents 4.x Compatibility Notes

## ✅ What's Updated for ML-Agents 4.x

All files have been updated to work with **Unity ML-Agents 4.0.0** specifically:

### Key Changes from Older Versions

1. **ActionTuple API**
   - Old (v1.x): `action_tuple.add_discrete(...)`
   - New (v4.x): `ActionTuple(discrete=...)`

2. **Observation Specs**
   - ML-Agents 4.x uses `observation_specs` list
   - Better handling of multiple observation types

3. **Python Dependencies**
   - Requires Python 3.9 - 3.11 (not 3.12+)
   - Requires numpy < 2.0
   - Compatible with latest Gymnasium API

## 🚀 Installation

### Quick Install (Windows)
```bash
install.bat
```

### Quick Install (Linux/Mac)
```bash
chmod +x install.sh
./install.sh
```

### Manual Install
```bash
# Make sure you have Python 3.9-3.11
python --version

# Install PyTorch first (optional but recommended)
pip install torch torchvision

# Install all dependencies
pip install -r requirements.txt
```

## 🔧 Unity Configuration for ML-Agents 4.x

### Required Settings in Unity Editor

1. **Agent Behavior Parameters Component:**
   - **Behavior Name**: Any name (e.g., "MazeAgent")
   - **Behavior Type**: ⚠️ Must be set to **Default** (not Heuristic Only!)
   - **Vector Observation Space**: 8 (updated for Python reward calculation)
     - obs[0-1]: Relative goal position (X, Z) - for agent decision making
     - obs[2-7]: Additional state for Python reward calculation:
       - obs[2]: Current distance to goal (normalized)
       - obs[3]: Previous distance to goal (normalized)
       - obs[4]: Has hit wall (1.0/0.0)
       - obs[5]: Has reached goal (1.0/0.0)
       - obs[6]: Time spent in wall collision
       - obs[7]: Agent velocity magnitude (normalized)
   - **Actions**:
     - Discrete Branches: 1
     - Branch 0 Size: 4 (actions: 0=nothing, 1=forward, 2=left, 3=right)
   - **Model**: Leave empty for training (will be set later)

2. **Decision Requester Component** (should be added automatically):
   - Decision Period: 5 (or adjust as needed)
   - Take Actions Between Decisions: Checked

### Testing the Connection

```bash
# Test with Unity Editor (Unity must be running with scene open)
python gymnasium_wrapper.py
```

You should see:
```
Using behavior: MazeAgent
Observation space: Box(2,)
Action space: Discrete(4)
Running random agent for 5 episodes...
```

## 📊 Training Recommendations for ML-Agents 4.x

### For Best Performance:

1. **Build Your Unity Project** (much faster than Editor)
   - File → Build Settings → Build
   - Save to `./build/` folder

2. **Use Time Acceleration**
   ```python
   env = make_unity_maze_env(
       unity_env_path="./build/MazeAgent.exe",
       time_scale=20.0  # 20x speed
   )
   ```

3. **Use Multiple Parallel Environments** (fastest)
   ```bash
   python train_sb3.py --algorithm ppo --n-envs 4 --unity-env "./build/MazeAgent.exe"
   ```

4. **Monitor with TensorBoard**
   ```bash
   tensorboard --logdir ./models/logs
   ```

## 🐛 Common Issues with ML-Agents 4.x

### Issue: "No behavior specs found"
**Solution:**
- Make sure Unity is running
- Check Behavior Type is set to **Default** (not Heuristic Only)
- Verify agent has Behavior Parameters component

### Issue: "numpy 2.0 not compatible"
**Solution:**
```bash
pip install "numpy<2.0"
```

### Issue: Python 3.12 installation fails
**Solution:**
- Use Python 3.9, 3.10, or 3.11
- ML-Agents 4.x doesn't support Python 3.12+ yet

### Issue: Port already in use
**Solution:**
```python
# Use different worker_id
env = make_unity_maze_env(worker_id=1)  # Try 1, 2, 3, etc.
```

### Issue: Training is slow
**Solutions:**
1. Build Unity project instead of using Editor
2. Set `no_graphics=True`
3. Increase `time_scale` (e.g., 20.0)
4. Use multiple parallel environments

## 📦 Package Versions (Tested and Working)

```
mlagents==4.0.0
mlagents-envs==4.0.0
gymnasium>=0.29.0
stable-baselines3>=2.0.0
numpy>=1.21.0,<2.0.0
protobuf>=3.20.0,<5.0.0
torch>=1.13.0
```

## 🎯 Example Training Commands

### Basic Training (with Unity Editor)
```bash
python train_sb3.py --algorithm ppo --timesteps 500000
```
"""
python train_sb3.py --algorithm ppo --model-path ./models/checkpoints/ppo_maze_10000_steps.zip
"""
### Fast Training (with Build)
```bash
python train_sb3.py --algorithm ppo --timesteps 1000000 --unity-env "./build/MazeAgent.exe"
```

### Fastest Training (Build + Multiple Envs)
```bash
python train_sb3.py --algorithm ppo --timesteps 1000000 --n-envs 4 --unity-env "./build/MazeAgent.exe"
```

### Evaluation
```bash
python train_sb3.py --mode evaluate --model-path "./models/ppo_maze_final.zip"
```

## 📝 Code Example

```python
from gymnasium_wrapper import make_unity_maze_env
from stable_baselines3 import PPO

# Create ML-Agents 4.x environment
env = make_unity_maze_env(
    unity_env_path="./build/MazeAgent.exe",  # or None for Editor
    worker_id=0,
    no_graphics=True,
    time_scale=20.0
)

# Create and train PPO model
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=500000)

# Save model
model.save("ppo_maze_model")

# Close environment
env.close()
```

## 🔗 Useful Links

- [Unity ML-Agents GitHub](https://github.com/Unity-Technologies/ml-agents)
- [ML-Agents 4.x Release Notes](https://github.com/Unity-Technologies/ml-agents/releases)
- [Stable-Baselines3 Docs](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)

---

**Last Updated:** Compatible with ML-Agents 4.0.0

