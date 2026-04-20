# Unity Editor - potrebne nastavenia

Nastavenia v Unity Editore pre spravne fungovanie agenta s Python-side reward vypoctom a ray observaciami.

## Potrebne zmeny

### 1. Behavior Parameters komponent

**Umiestnenie:** Vyber Agent GameObject > Inspector > Behavior Parameters

**Nastavenia:**

1. **Vector Observation Space Size**
   - S ray observaciami (default): **24**
   - Bez ray observacii: **8**

   **Vypocet:**
   - Zakladne observacie: `8`
   - Ray observacie: `8 lucov x 2 hodnoty = 16`
   - **Celkom: `8 + 16 = 24`**

   **Postup zmeny:**
   - Vyber Agent GameObject v Hierarchy
   - V Inspectore najdi **Behavior Parameters** komponent
   - Zmen **Vector Observation > Space Size** na `24`

### 2. Agent komponent - nastavenia ray observacii

**Umiestnenie:** Agent GameObject > Inspector > Agent (Script) komponent

**Konfigurovatelne polia:**

| Pole | Default | Popis |
|------|---------|-------|
| **Use Ray Observations** | zapnute | Prepinac ray detekcie stien |
| **Num Rays** | `8` | Pocet lucov okolo agenta |
| **Ray Length** | `10` | Maximalny dosah detekcie |
| **Ray Start Height** | `0.5` | Vyskovy offset pociatku luca |
| **Ray Layer Mask** | Everything | Vrstvy pre detekciu |

**Ak vypnes ray observacie:**
- Nastav **Vector Observation Space Size** na `8` namiesto `24`

### 3. Overenie ostatnych nastaveni

Skontroluj tieto nastavenia:

- **Behavior Type:** `Default` (NIE "Heuristic Only")
- **Vector Observation > Space Size:** `24`
- **Actions > Discrete Branches:** `1`
- **Actions > Branch 0 Size:** `4` (0=nic, 1=dopredu, 2=vlavo, 3=vpravo)
- **Decision Requester > Decision Period:** `5` (alebo podla potreby)
- **Decision Requester > Take Actions Between Decisions:** zaciarkni

### 4. Maze komponent

Ak pouzivas proceduralne bludisko, nastav na Maze GameObjecte:

| Pole | Default | Popis |
|------|---------|-------|
| **Width** | `10` | Sirka bludiska (pocet buniek) |
| **Height** | `10` | Vyska bludiska (pocet buniek) |
| **Scale** | `6` | Velkost kazdej bunky vo svete |
| **Wall Material** | - | Material pre steny (nastav!) |

### 5. Tagy

Uisti sa, ze tieto tagy existuju v projekte (Project Settings > Tags and Layers):

- **wall** - priradeny vsetkym stenam bludiska
- **goal** - priradeny cielovemu objektu

### 6. TrainingManager (volitelne)

Pre optimalizaciu pocas treningu pridaj `TrainingManager` skript na lubovolny GameObject:

| Pole | Default | Popis |
|------|---------|-------|
| **Target Frame Rate** | `60` | Limituje FPS renderovania |
| **Disable VSync** | zapnute | Pre funkcnost targetFrameRate |
| **Optimize For Training** | zapnute | Vypne tiene, AA, atd. |
| **Fixed Timestep** | `0.02` | Fyzikalny timestep |
| **Maximum Delta Time** | `0.1` | Prevencia "spiral of death" |

## Struktura observacii

### Zakladne observacie (index 0-7)

| Index | Observacia | Popis | Normalizacia |
|-------|-----------|-------|--------------|
| 0 | Relativna pozicia ciela X | Vlavo/vpravo od agenta | / 5.0 |
| 1 | Relativna pozicia ciela Z | Dopredu/dozadu od agenta | / 5.0 |
| 2 | Aktualna vzdialenost | Vzdialenost k cielu | / 10.0 |
| 3 | Predchadzajuca vzdialenost | Predchadzajuca vzdialenost k cielu | / 10.0 |
| 4 | Kolizia so stenou | 1.0 ak koliduje, 0.0 inak | - |
| 5 | Dosiahol ciel | 1.0 ak dosiahol, 0.0 inak | - |
| 6 | Cas v kolizii | Cas straveny pri stene | sekundy |
| 7 | Rychlost agenta | Velkost vektora rychlosti | / 5.0 |

### Ray observacie (index 8-23, pri 8 lucoch)

Pre kazdy luc `i` (0 az 7):

| Index | Observacia | Rozsah |
|-------|-----------|--------|
| 8 + i*2 | Vzdialenost k prekazke | 0.0 - 1.0 (normalizovana) |
| 9 + i*2 | Typ prekazky | 1.0 = stena, 0.5 = ina prekazka, 0.0 = nic |

**Smery lucov (od smeru dopredu, v smere hodinovych ruciciek):**
- Luc 0: Dopredu (0°)
- Luc 1: Dopredu-vpravo (45°)
- Luc 2: Vpravo (90°)
- Luc 3: Dozadu-vpravo (135°)
- Luc 4: Dozadu (180°)
- Luc 5: Dozadu-vlavo (225°)
- Luc 6: Vlavo (270°)
- Luc 7: Dopredu-vlavo (315°)

## Dolezite poznamky

1. **Ak nezmenis Vector Observation Space Size:**
   - Unity hodi chybu o nesuladu velkosti observacii
   - Trening sa nespusti
   - Python reward vypocet nebude fungovat spravne

2. **Ray observacie vyzaduju tag "wall":**
   - Vsetky steny v scene musia mat tag `wall`
   - Lucovy system rozlisuje steny od inych prekazok

3. **Maze regeneracia:**
   - Bludisko sa regeneruje kazych 3000 epizod pre lepsiu generalizaciu
   - Toto je konfigurovatelne v `agent.cs` (podmienka `_currentEpisode % 3000 == 0`)

4. **Spawn logika:**
   - Agent sa spawnuje na nahodnej prazdnej pozicii v bludisku
   - Ciel sa umiestni na najvzdialenjsiu prazdnu poziciu od agenta
   - Toto zabezpecuje maximalnu narocnost navigacie

5. **Debug vizualizacia:**
   - Ray observacie su vizualizovane v Scene pohlade pocas Play modu
   - Zeleny luc = ziadna prekazka
   - Zlty luc = ina prekazka
   - Cerveny luc = detekovana stena

## Rychly kontrolny zoznam

- [ ] Otvor Unity Editor
- [ ] Vyber Agent GameObject v Hierarchy
- [ ] Najdi Behavior Parameters v Inspectore
- [ ] Nastav Vector Observation Space Size na `24` (alebo `8` bez lucov)
- [ ] Over Behavior Type = `Default`
- [ ] Over Actions > Discrete Branches = 1, Branch 0 Size = 4
- [ ] Skontroluj ray nastavenia v Agent skript komponente
- [ ] Over tagy: steny = "wall", ciel = "goal"
- [ ] Over Maze komponent (width, height, scale, material)
- [ ] Uloz scenu (Ctrl+S)
- [ ] Otestuj spojenie s Pythonom: `python gymnasium_wrapper.py`

## Overenie

### V Unity Editore
- Vyber Agent GameObject
- Behavior Parameters > Vector Observation > Space Size = `24`
- Agent skript ukazuje ray observation nastavenia

### V Pythone
```bash
cd python
python gymnasium_wrapper.py
```
Ocakavany vystup:
```
Observation space: Box(24,)
```

### V Unity Scene pohlade (Play mod)
- Mali by byt viditelne farebne debug luce okolo agenta

## Tabulka konfigurácii

| Konfiguracia | Observation Size | Pouzitie |
|--------------|------------------|----------|
| Luče zapnute (default) | `24` | Plna detekcia stien |
| Luče vypnute | `8` | Jednoduchsie prostredie |

**Zmena poctu lucov:**

| Pocet lucov | Ray observacii | Celkove observacie |
|-------------|---------------|-------------------|
| 4 | 8 | 16 |
| 6 | 12 | 20 |
| 8 (default) | 16 | 24 |
| 12 | 24 | 32 |
| 16 | 32 | 40 |

**Vzorec:** `Celkom = 8 (zaklad) + (pocet_lucov x 2)`
