# Unity ML-Agents Maze - Gymnasium Wrapper

Gymnasium wrapper pre Unity ML-Agents 4.x bludiskove prostredie, umoznujuci trening s RL kniznicami ako Stable-Baselines3.

**Kompatibilne s Unity ML-Agents 4.0.0**

## Rychly start

### 1. Instalacia zavislosti

```bash
pip install -r requirements.txt
```

Poziadavky:
- Python 3.9 - 3.11 (Python 3.12+ nie je podporovany)
- NumPy < 2.0
- Ak mas problemy s instalaciou:
  ```bash
  pip install torch torchvision
  pip install -r requirements.txt
  ```

### 2. Konfiguracia Unity pre komunikaciu

V Unity Editore:
1. Otvor scenu `Assets/Scenes/SampleScene.unity`
2. Vyber Agent GameObject
3. V komponente **Behavior Parameters** nastav:
   - **Behavior Type**: **Default** (nie Heuristic Only)
   - **Vector Observation Space Size**: **24** (8 zakladnych + 16 ray observacii)
   - **Actions** > **Discrete Branches** > **Branch 0 Size**: **4** (0=nic, 1=dopredu, 2=vlavo, 3=vpravo)

Ak su ray observacie vypnute, nastav Vector Observation Space Size na **8**.

### 3. Test wrappera

Test s Unity Editorom (bez buildu):

```bash
cd python
python gymnasium_wrapper.py
```

Spusti 5 epizod s nahodnymi akciami pre overenie spojenia.

## Trening so Stable-Baselines3

### PPO (odporucany)

**S Unity Editorom** (na testovanie/debugging):
```bash
python train_sb3.py --algorithm ppo --timesteps 500000
```

**S Unity buildom** (ovela rychlejsie):
```bash
python train_sb3.py --algorithm ppo --timesteps 500000 --unity-env "./build/MazeAgent.exe"
```

**S paralelnym treningom** (najrychlejsie, vyzaduje build):
```bash
python train_sb3.py --algorithm ppo --timesteps 1000000 --n-envs 4 --unity-env "./build/MazeAgent.exe"
```

### SAC

SAC vyzaduje spojity akcny priestor, wrapper automaticky konvertuje diskretne akcie.

```bash
python train_sb3.py --algorithm sac --timesteps 500000 --unity-env "./build/MazeAgent.exe"
```

### Pokracovanie treningu z checkpointu

```bash
python train_sb3.py --algorithm ppo --timesteps 200000 --model-path ./models/checkpoints/ppo_maze_100000_steps.zip
```

### Evaluacia natrenovaneho modelu

```bash
python train_sb3.py --mode evaluate --model-path "./models/checkpoints/ppo_maze_500000_steps.zip"
```

## Vizualizacia natrenovaneho agenta

```bash
python watch_agent.py                                         # Najnovsi checkpoint
python watch_agent.py --model ./models/checkpoints/best.zip   # Konkretny model
python watch_agent.py --episodes 5 --time-scale 0.5           # Spomaleny pohyb
```

## Pouzitie Gymnasium Wrappera priamo

```python
from gymnasium_wrapper import make_unity_maze_env

env = make_unity_maze_env(
    unity_env_path="./build/MazeAgent.exe",  # alebo None pre Unity Editor
    worker_id=0,
    no_graphics=True,
    time_scale=20.0
)

obs, info = env.reset()

for step in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

## Sledovanie treningu cez TensorBoard

```bash
tensorboard --logdir ./models/logs
```

Otvor prehliadac na http://localhost:6006

## Vlastny trening

```python
from gymnasium_wrapper import make_unity_maze_env
from stable_baselines3 import PPO

env = make_unity_maze_env(unity_env_path="./build/MazeAgent.exe")

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    ent_coef=0.02,
    policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
    verbose=1
)

model.learn(total_timesteps=1000000)
model.save("my_custom_model")
env.close()
```

## Buildovanie Unity spustitelneho suboru

Pre rychlejsi trening buildni Unity projekt:

1. V Unity Editore: **File > Build Settings**
2. Pridaj scenu `Assets/Scenes/SampleScene.unity`
3. Vyber platformu (Windows/Mac/Linux)
4. Klikni **Build**
5. Uloz do zlozky (napr. `./build/`)
6. Pouzi cestu k spustitelnemu suboru v trenovacich skriptoch

## Detaily prostredia

### Observation Space

Velkost zavisi od konfiguracie ray observacii:

**S ray observaciami (default) - Box(24,):**

| Index | Popis | Rozsah |
|-------|-------|--------|
| 0 | Relativna pozicia ciela X (vlavo/vpravo) | ~ [-1, 1] |
| 1 | Relativna pozicia ciela Z (dopredu/dozadu) | ~ [-1, 1] |
| 2 | Aktualna vzdialenost k cielu (normalizovana) | [0, ~6] |
| 3 | Predchadzajuca vzdialenost k cielu (normalizovana) | [0, ~6] |
| 4 | Kolizia so stenou | 0.0 alebo 1.0 |
| 5 | Dosiahol ciel | 0.0 alebo 1.0 |
| 6 | Cas straveny v kolizii so stenou | sekundy |
| 7 | Rychlost agenta (normalizovana) | [0, ~1] |
| 8-23 | Ray observacie (8 lucov x 2 hodnoty: vzdialenost + typ prekazky) | [0, 1] |

**Bez ray observacii - Box(8,):**

Iba indexy 0-7.

### Action Space - Discrete(4)

| Akcia | Popis |
|-------|-------|
| 0 | Nic nerob |
| 1 | Pohyb dopredu |
| 2 | Otoc vlavo |
| 3 | Otoc vpravo |

### Reward struktura (pocitana v Pythone)

| Odmena | Hodnota | Popis |
|--------|---------|-------|
| Dosiahnutie ciela | +50.0 | Vysoka odmena pre dominanciu nad penaltami |
| Bonus za pohyb | +0.02 / krok | Ked sa agent pohybuje (velocity > 0) |
| Statie na mieste | -0.1 / krok | Silna penalta pre prevenciu lokalneho optima |
| Kolizia so stenou (initial) | -0.2 | Pri zaciatku kolizie |
| Kolizia so stenou (continuous) | -0.02 * cas_v_kolizii | Kontinualna penalta |
| Casova penalta | -30.0 / max_steps | Tlak na rychle dokoncenie |
| Zmena vzdialenosti | vypnute (0.0) | Euklidovska vzdialenost je zavádzajúca v bludisku |

## Riesenie problemov

### "No behavior specs found"
- Uisti sa, ze Unity bezi a agent ma komponent **Behavior Parameters**
- Skontroluj, ze **Behavior Type** je nastaveny na **Default** (nie Heuristic Only)

### Trening je pomaly
- Buildni Unity projekt namiesto pouzivania Editora
- Pouzi paralelne prostredia s `--n-envs`
- Zvy parameter `time_scale`
- Nastav `no_graphics=True`

### Connection timeout
- Uisti sa, ze nebezi ine Unity ML-Agents prostredie
- Skus iny `worker_id` parameter
- Skontroluj nastavenia firewallu
- Skontroluj, ze pouzivas rovnaky port (default: 5004)

### Observation space mismatch
- Skontroluj, ci Vector Observation Space Size v Unity zodpoveda konfigurácii ray observácii
- S ray observaciami (default): **24**
- Bez ray observacii: **8**

## Tipy pre lepsi trening

1. **Zacni s Unity Editorom** na debugging, potom prepni na build
2. **Pouzi PPO** pre vacsinu pripadov (funguje dobre pre diskretne akcie)
3. **Sleduj TensorBoard** na sledovanie progresu
4. **Paralelne prostredia** pre rychlejsiu konvergenciu
5. **Zvys max_steps** pre vacsie bludiská: `--max-steps 3000`
6. **PPO nepotrebuje VecNormalize** (observacie su normalizovane v Unity), ostatne algoritmy (DQN, A2C, SAC) ho pouzivaju automaticky

## Kompatibilita s inymi RL kniznicami

Gymnasium wrapper je kompatibilny s lubovolnou kniznicou podporujucou Gymnasium API:

- Stable-Baselines3
- RLlib (Ray)
- CleanRL
- Tianshou
- Vlastne implementacie

## Uzitocne linky

- [Unity ML-Agents GitHub](https://github.com/Unity-Technologies/ml-agents)
- [Stable-Baselines3 Docs](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
