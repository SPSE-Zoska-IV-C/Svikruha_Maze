# ML-Agents 4.x - Poznamky ku kompatibilite

## Co je aktualizovane pre ML-Agents 4.x

Vsetky subory su kompatibilne s **Unity ML-Agents 4.0.0**.

### Klucove zmeny oproti starsim verziam

1. **ActionTuple API**
   - Stare (v1.x): `action_tuple.add_discrete(...)`
   - Nove (v4.x): `ActionTuple(discrete=...)`

2. **Observation Specs**
   - ML-Agents 4.x pouziva `observation_specs` zoznam
   - Lepsie spracovanie viacerych typov observacii

3. **Python zavislosti**
   - Vyzaduje Python 3.9 - 3.11 (nie 3.12+)
   - Vyzaduje NumPy < 2.0
   - Kompatibilne s najnovsim Gymnasium API

## Instalacia

### Manualna instalacia
```bash
# Over verziu Pythonu (3.9-3.11)
python --version

# Instaluj PyTorch (volitelne ale odporucane)
pip install torch torchvision

# Instaluj vsetky zavislosti
pip install -r requirements.txt
```

## Konfiguracia Unity pre ML-Agents 4.x

### Potrebne nastavenia v Unity Editore

1. **Behavior Parameters komponent na Agent GameObjecte:**
   - **Behavior Name**: lubovolne meno (napr. "MazeAgent")
   - **Behavior Type**: **Default** (nie Heuristic Only!)
   - **Vector Observation Space Size**: **24** (s ray observaciami) alebo **8** (bez ray observacii)
   - **Actions**:
     - Discrete Branches: 1
     - Branch 0 Size: 4 (akcie: 0=nic, 1=dopredu, 2=vlavo, 3=vpravo)
   - **Model**: Nechaj prazdne pocas treningu

2. **Decision Requester komponent:**
   - Decision Period: 5 (alebo podla potreby)
   - Take Actions Between Decisions: zaciarkni

### Struktura observacii (24 hodnot s ray observaciami)

**Zakladne observacie (index 0-7):**

| Index | Observacia | Popis | Normalizacia |
|-------|-----------|-------|--------------|
| 0 | Relativna pozicia ciela X | Vlavo/vpravo od agenta | / 5.0 |
| 1 | Relativna pozicia ciela Z | Dopredu/dozadu od agenta | / 5.0 |
| 2 | Aktualna vzdialenost | Vzdialenost k cielu | / 10.0 |
| 3 | Predchadzajuca vzdialenost | Predchadzajuca vzdialenost k cielu | / 10.0 |
| 4 | Kolizia so stenou | 1.0 ak koliduje, 0.0 inak | - |
| 5 | Dosiahol ciel | 1.0 ak dosiahol, 0.0 inak | - |
| 6 | Cas v kolizii | Cas straveny v kolizii so stenou | sekundy |
| 7 | Rychlost | Velkost rychlosti agenta | / 5.0 |

**Ray observacie (index 8-23, pri 8 lucoch):**

Pre kazdy luc `i` (0 az 7):

| Index | Observacia | Rozsah |
|-------|-----------|--------|
| 8 + i*2 | Normalizovana vzdialenost k prekazke | 0.0 - 1.0 |
| 9 + i*2 | Typ prekazky (1.0=stena, 0.5=ine, 0.0=nic) | 0.0, 0.5, 1.0 |

**Vzorec: Total = 8 (zaklad) + (pocet_lucov * 2)**

### Testovanie spojenia

```bash
cd python
python gymnasium_wrapper.py
```

Ocakavany vystup:
```
Using behavior: MazeAgent
Observation space: Box(24,)
Action space: Discrete(4)
Running random agent for 5 episodes...
```

## Odporucania pre trening s ML-Agents 4.x

### Pre najlepsi vykon:

1. **Buildni Unity projekt** (ovela rychlejsie nez Editor)
   - File > Build Settings > Build

2. **Pouzi akceleraciu casu**
   ```python
   env = make_unity_maze_env(
       unity_env_path="./build/MazeAgent.exe",
       time_scale=20.0  # 20x rychlost
   )
   ```

3. **Pouzi paralelne prostredia** (najrychlejsie)
   ```bash
   python train_sb3.py --algorithm ppo --n-envs 4 --unity-env "./build/MazeAgent.exe"
   ```

4. **Sleduj s TensorBoardom**
   ```bash
   tensorboard --logdir ./models/logs
   ```

## Caste problemy s ML-Agents 4.x

### "No behavior specs found"
- Uisti sa, ze Unity bezi
- Skontroluj Behavior Type = **Default** (nie Heuristic Only)
- Over, ze agent ma Behavior Parameters komponent

### "numpy 2.0 not compatible"
```bash
pip install "numpy<2.0"
```

### Python 3.12 instalacia zlyhava
- Pouzi Python 3.9, 3.10, alebo 3.11
- ML-Agents 4.x este nepodporuje Python 3.12+

### Port uz sa pouziva
```python
env = make_unity_maze_env(worker_id=1)  # Skus 1, 2, 3, atd.
```

### Trening je pomaly
1. Buildni Unity projekt namiesto pouzivania Editora
2. Nastav `no_graphics=True`
3. Zvys `time_scale` (napr. 20.0)
4. Pouzi paralelne prostredia

### Observation space mismatch pri nacitani modelu
- Model natrenovany s inou velkostou observation space nie je kompatibilny
- Skontroluj ci ray observacie su zapnute/vypnute rovnako ako pri treningu
- S ray: 24 observacii, bez ray: 8 observacii

## Verzie balikov (pouzivane v projekte)

Aktualne verzie v `requirements.txt`:

```
gymnasium==1.2.2
mlagents_envs==1.2.0.dev0
numpy==1.23.5
stable_baselines3==2.7.0
```

Unity strana:
```
com.unity.ml-agents: 4.0.0
Unity: 6000.2.6f2
```

**Poznamka:** Python balicek `mlagents_envs==1.2.0.dev0` je kompatibilny s Unity ML-Agents package 4.0.0. Verzie nemusia mat rovnake cisla - dolezita je kompatibilita komunikacneho protokolu.

## Priklady trenovacich prikazov

### Zakladny trening (s Unity Editorom)
```bash
python train_sb3.py --algorithm ppo --timesteps 500000
```

### Pokracovanie treningu z checkpointu
```bash
python train_sb3.py --algorithm ppo --timesteps 200000 --model-path ./models/checkpoints/ppo_maze_100000_steps.zip
```

### Rychly trening (s buildom)
```bash
python train_sb3.py --algorithm ppo --timesteps 1000000 --unity-env "./build/MazeAgent.exe"
```

### Najrychlejsi trening (build + paralelne prostredia)
```bash
python train_sb3.py --algorithm ppo --timesteps 1000000 --n-envs 4 --unity-env "./build/MazeAgent.exe"
```

### Evaluacia
```bash
python train_sb3.py --mode evaluate --model-path "./models/checkpoints/ppo_maze_500000_steps.zip"
```

### Vizualizacia
```bash
python watch_agent.py --model ./models/checkpoints/ppo_maze_500000_steps.zip --episodes 5
```

## Priklad kodu

```python
from gymnasium_wrapper import make_unity_maze_env
from stable_baselines3 import PPO

env = make_unity_maze_env(
    unity_env_path="./build/MazeAgent.exe",
    worker_id=0,
    no_graphics=True,
    time_scale=20.0
)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=500000)
model.save("ppo_maze_model")
env.close()
```

## Uzitocne linky

- [Unity ML-Agents GitHub](https://github.com/Unity-Technologies/ml-agents)
- [ML-Agents Release Notes](https://github.com/Unity-Technologies/ml-agents/releases)
- [Stable-Baselines3 Docs](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
