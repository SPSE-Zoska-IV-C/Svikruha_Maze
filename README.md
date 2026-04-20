# Svikruha_Maze

Maturitny projekt - AI agent navigujuci proceduralnym bludisko pomocou reinforcement learningu.

## Popis projektu

Projekt kombinuje **Unity 6** (URP) s **ML-Agents 4.0** na strane simulacie a **Python** s **Stable-Baselines3** + **Gymnasium** na strane treningu. Agent sa uci navigovat nahodne generovanym bludisko (maze) pomocou diskretnych akcii (dopredu, otoc vlavo, otoc vpravo, nic nerob).

### Klucove vlastnosti

- **Proceduralne bludisko** - generovane rekurzivnym DFS algoritmom s kontrolou konektivity
- **Ray-based observacie** - 8 smerovych lucov na detekciu stien (volitelne)
- **Python-side reward** - odmeny su pocitane v Pythone na zaklade pozorovani z Unity
- **Paralelny trening** - podpora viacerych Unity buildov sucasne cez `SubprocVecEnv`
- **2 RL algoritmy** - PPO, SAC

## Technologie

| Komponent | Verzia |
|-----------|--------|
| Unity | 6000.2.6f2 (URP) |
| ML-Agents (Unity package) | 4.0.0 |
| Python | 3.9 - 3.11 |
| mlagents_envs | 1.2.0.dev0 |
| Gymnasium | 1.2.2 |
| Stable-Baselines3 | 2.7.0 |
| NumPy | 1.23.5 |

## Struktura projektu

```
Svikruha_Maze/
├── ENVs/                          # Unity projekt
│   ├── Assets/
│   │   ├── Scenes/                # SampleScene.unity (hlavna scena)
│   │   ├── scripts/
│   │   │   ├── agent.cs           # ML-Agent - observacie, akcie, spawn logika
│   │   │   ├── maze.cs            # Proceduralna generacia bludiska
│   │   │   ├── TrainingManager.cs # Optimalizacia GPU pocas treningu
│   │   │   ├── GUI_Agent.cs       # Debug HUD (epizoda, reward, kolizie)
│   │   │   ├── camera.cs          # Ortograficka kamera na bludisko
│   │   │   └── Extensions.cs      # Shuffle extension metoda
│   │   ├── materials/             # Materialy (steny, agent, ciel)
│   │   └── ML-Agents/Timers/     # Profiler data
│   ├── ProjectSettings/           # Unity nastavenia
│   └── results/                   # Ulozene run konfiguracie (YAML)
├── python/
│   ├── gymnasium_wrapper.py       # Gymnasium wrapper pre Unity prostredie
│   ├── train_sb3.py               # Trenovaci skript (PPO, DQN, A2C, SAC)
│   └── watch_agent.py             # Vizualizacia natrenovaneho agenta
├── docs/
│   ├── README_GYMNASIUM.md        # Dokumentacia Gymnasium wrappera
│   ├── MLAGENTS_4X_NOTES.md       # Poznamky k ML-Agents 4.x kompatibilite
│   └── UNITY_EDITOR_CHANGES.md    # Nastavenia v Unity Editore
├── requirements.txt               # Python zavislosti
└── README.md                      # Tento subor
```

## Rychly start

### 1. Instalacia Python zavislosti

```bash
pip install -r requirements.txt
```

### 2. Otvorenie Unity projektu

Otvor zlozku `ENVs/` v Unity Hub (Unity 6000.2.6f2). Hlavna scena je `Assets/Scenes/SampleScene.unity`.

### 3. Spustenie treningu

```bash
cd python

# Trening s Unity Editorom (najprv spusti Play v Unity)
python train_sb3.py --algorithm ppo --timesteps 500000

# Trening s buildom (rychlejsie)
python train_sb3.py --algorithm ppo --timesteps 1000000 --unity-env "./build/MazeAgent.exe"

# Paralelny trening (najrychlejsie, vyzaduje build)
python train_sb3.py --algorithm ppo --timesteps 1000000 --n-envs 4 --unity-env "./build/MazeAgent.exe"
```

### 4. Sledovanie treningu

```bash
tensorboard --logdir ./models/logs
```

### 5. Vizualizacia natrenovaneho agenta

```bash
cd python
python watch_agent.py --model ./models/checkpoints/ppo_maze_500000_steps.zip
```

## Agent - observacie a akcie

### Observation Space (24 hodnot s ray observaciami)

| Index | Popis | Normalizacia |
|-------|-------|--------------|
| 0-1 | Relativna pozicia ciela (X, Z) | / 5.0 |
| 2 | Aktualna vzdialenost k cielu | / 10.0 |
| 3 | Predchadzajuca vzdialenost k cielu | / 10.0 |
| 4 | Kolizia so stenou (0.0 / 1.0) | - |
| 5 | Dosiahol ciel (0.0 / 1.0) | - |
| 6 | Cas v kolizii so stenou | sekundy |
| 7 | Rychlost agenta | / 5.0 |
| 8-23 | Ray observacie (8 lucov x 2 hodnoty) | 0.0 - 1.0 |

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
| Bonus za pohyb | +0.02 / krok | Ked sa agent pohybuje |
| Statie na mieste | -0.1 / krok | Penalta za necinnost |
| Kolizia so stenou (initial) | -0.2 | Mierna penalta (steny su ocakavane v bludisku) |
| Kolizia so stenou (continuous) | -0.02 * cas | Pokracujuca penalta |
| Casova penalta | -30.0 / max_steps / krok | Tlak na rychle dokoncenie |

## Dalsie dokumentacia

- [Gymnasium Wrapper](docs/README_GYMNASIUM.md) - Pouzivanie Gymnasium API a SB3
- [ML-Agents 4.x poznamky](docs/MLAGENTS_4X_NOTES.md) - Kompatibilita a instalacia
- [Unity Editor nastavenia](docs/UNITY_EDITOR_CHANGES.md) - Konfiguracia v Unity
