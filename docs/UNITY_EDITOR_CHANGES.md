# Unity Editor Changes Required

After updating `agent.cs` to use Python-based reward calculation with ray observations, you need to make the following changes in Unity Editor:

## 🔧 Required Changes

### 1. **Agent Behavior Parameters Component**

**Location:** Select your Agent GameObject → Inspector → Behavior Parameters component

**Changes needed:**

1. **Vector Observation Space Size**
   - ❌ **OLD:** `2` or `8` (without ray observations)
   - ✅ **NEW:** `24` (with ray observations enabled - default)
   
   **Calculation:**
   - Base observations: `8`
   - Ray observations: `8 rays × 2 values each = 16`
   - **Total: `8 + 16 = 24`**
   
   **How to change:**
   - Select your Agent GameObject in Hierarchy
   - In Inspector, find the **Behavior Parameters** component
   - Change **Vector Observation → Space Size** to `24`

### 2. **Agent Component - Ray Observation Settings**

**Location:** Select your Agent GameObject → Inspector → Agent (Script) component

**New configurable fields:**

| Field | Default | Description |
|-------|---------|-------------|
| **Use Ray Observations** | ✓ Enabled | Toggle ray-based wall detection |
| **Num Rays** | `8` | Number of rays cast around agent |
| **Ray Length** | `10` | Maximum ray detection distance |
| **Ray Start Height** | `0.5` | Height offset for ray origin |
| **Ray Layer Mask** | Everything | Layers to detect with rays |

**If you disable ray observations:**
- Set **Vector Observation Space Size** to `8` instead of `24`

### 3. **Verify Other Settings**

Make sure these settings are correct:

- ✅ **Behavior Type:** `Default` (NOT "Heuristic Only")
- ✅ **Vector Observation → Space Size:** `24` ← **CHANGE THIS!**
- ✅ **Actions → Discrete Branches:** `1`
- ✅ **Actions → Branch 0 Size:** `4` (0=nothing, 1=forward, 2=left, 3=right)
- ✅ **Decision Requester → Decision Period:** `5` (or your preferred value)
- ✅ **Decision Requester → Take Actions Between Decisions:** ✓ Checked

## 📋 Observation Structure (for reference)

### Base Observations (indices 0-7)

| Index | Observation | Description | Normalization |
|-------|-------------|-------------|---------------|
| 0 | Relative Goal X | Goal position relative to agent (left/right) | ÷ 5.0 |
| 1 | Relative Goal Z | Goal position relative to agent (forward/back) | ÷ 5.0 |
| 2 | Current Distance | Current distance to goal | ÷ 10.0 |
| 3 | Previous Distance | Previous distance to goal | ÷ 10.0 |
| 4 | Has Hit Wall | 1.0 if colliding with wall, 0.0 otherwise | - |
| 5 | Has Reached Goal | 1.0 if goal reached, 0.0 otherwise | - |
| 6 | Time In Wall | Time spent in wall collision | seconds |
| 7 | Velocity Magnitude | Agent's velocity magnitude | ÷ 5.0 |

### Ray Observations (indices 8-23, with 8 rays)

For each ray `i` (0 to 7), the observations are:

| Index | Observation | Description | Value Range |
|-------|-------------|-------------|-------------|
| 8 + i×2 | Ray Distance | Normalized distance to obstacle | 0.0 - 1.0 |
| 9 + i×2 | Obstacle Type | What the ray hit | 0.0, 0.5, or 1.0 |

**Obstacle Type Values:**
- `1.0` = Wall detected
- `0.5` = Other obstacle detected
- `0.0` = No obstacle (max range)

**Ray Directions (starting from forward, clockwise):**
- Ray 0: Forward (0°)
- Ray 1: Forward-Right (45°)
- Ray 2: Right (90°)
- Ray 3: Back-Right (135°)
- Ray 4: Back (180°)
- Ray 5: Back-Left (225°)
- Ray 6: Left (270°)
- Ray 7: Forward-Left (315°)

## ⚠️ Important Notes

1. **If you don't change Vector Observation Space:**
   - Unity will throw errors about observation size mismatch
   - Training will fail to start
   - The Python reward calculation won't work correctly

2. **Ray observations require walls tagged as "wall":**
   - Make sure all wall objects in your scene have the tag `wall`
   - The ray system detects walls vs other obstacles differently

3. **After making changes:**
   - Save the scene (Ctrl+S)
   - The changes take effect immediately
   - No need to rebuild the project

4. **Debug visualization:**
   - Ray observations are visualized in Scene view during Play mode
   - Green rays = no obstacle
   - Yellow rays = non-wall obstacle
   - Red rays = wall detected

## 🎯 Quick Checklist

- [ ] Open Unity Editor
- [ ] Select Agent GameObject in Hierarchy
- [ ] Find Behavior Parameters component in Inspector
- [ ] Change Vector Observation Space Size to `24` (or `8` if rays disabled)
- [ ] Verify Behavior Type is set to `Default`
- [ ] (Optional) Adjust ray settings in Agent script component
- [ ] Ensure walls are tagged as "wall"
- [ ] Save scene (Ctrl+S)
- [ ] Test connection with Python

## 🔍 How to Verify the Change

After making the change, you can verify it worked:

1. **In Unity Editor:**
   - Select Agent GameObject
   - Check Behavior Parameters → Vector Observation → Space Size shows `24`
   - Check Agent script shows ray observation settings

2. **In Python:**
   ```bash
   python gymnasium_wrapper.py
   ```
   Should output:
   ```
   Observation space: Box(24,)
   ```
   NOT:
   ```
   Observation space: Box(2,)  # ❌ Wrong - old version
   Observation space: Box(8,)  # ❌ Wrong - rays disabled but not configured
   ```

3. **In Unity Scene View (Play mode):**
   - You should see colored debug rays around the agent
   - This confirms ray observations are working

## 📝 Configuration Summary

| Configuration | Observation Size | Use When |
|---------------|------------------|----------|
| Rays enabled (default) | `24` | Full wall detection capabilities |
| Rays disabled | `8` | Simpler environment without walls |

**Default setup (recommended):**
- **Vector Observation Space Size:** `24`
- **Use Ray Observations:** ✓ Enabled
- **Num Rays:** `8`

## 🔄 Changing Number of Rays

If you change `Num Rays` in the Agent component, update the observation space:

| Num Rays | Ray Observations | Total Observations |
|----------|------------------|-------------------|
| 4 | 8 | 16 |
| 6 | 12 | 20 |
| 8 (default) | 16 | 24 |
| 12 | 24 | 32 |
| 16 | 32 | 40 |

**Formula:** `Total = 8 (base) + (num_rays × 2)`
