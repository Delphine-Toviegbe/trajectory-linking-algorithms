# Trajectory Linking Algorithm

Algorithme de liaison de trajectoires fragmentées basé sur des critères **spatio-temporels**.  
Ce projet se décompose en **deux parties indépendantes** :

1. **Liaison + visualisation** — relie les trajectoires et produit une vidéo animée (`link_trajectories.py` + `visualize_trajectories.py`)
2. **Comparaison d'algorithmes** — benchmark de trois stratégies d'affectation avec analyse de complexité (`compare_algorithms.py`)

---

## Table des matières

- [Installation](#installation)
- [Démarrage rapide](#démarrage-rapide)
- [Structure du projet](#structure-du-projet)
- [Format des données](#format-des-données)
- [Partie 1 — Liaison et visualisation](#partie-1--liaison-et-visualisation)
  - [link_trajectories.py](#1-link_trajectoriespy)
  - [visualize_trajectories.py](#2-visualize_trajectoriespy)
- [Partie 2 — Comparaison des algorithmes](#partie-2--comparaison-des-algorithmes)
  - [compare_algorithms.py](#3-compare_algorithmspy)
  - [visualize_trajectories_compare.py](#4-visualize_trajectories_comparepy)
- [Analyse : pourquoi le glouton suffit ici](#analyse--pourquoi-le-glouton-suffit-ici)
- [Points forts et limites de chaque algorithme](#points-forts-et-limites-de-chaque-algorithme)
- [Résultats sur le dataset fourni](#résultats-sur-le-dataset-fourni)

---

## Installation

**Prérequis : Python 3.10+**

```bash
# Cloner le dépôt
git clone https://github.com/<votre-utilisateur>/trajectory-linking-algorithms.git
cd trajectory-linking-algorithms

# Installer les dépendances
pip install -r requirements.txt
```

`requirements.txt` :

```
numpy>=1.24
scipy>=1.11
opencv-python>=4.8
```

---

## Démarrage rapide

Pour exécuter l'ensemble du pipeline en une fois :

```bash
# 1. Liaisons texte
python link_trajectories.py --input trajectories_dataset.json

# 2. Vidéo de visualisation
python visualize_trajectories.py --input trajectories_dataset.json --output trajectories_linked.mp4

# 3. Benchmark des 3 algorithmes
python compare_algorithms.py --input trajectories_dataset.json

# 4. Vidéos comparatives (une par algorithme)
python visualize_trajectories_compare.py \
  --input trajectories_dataset.json \
  --algo greedy

python visualize_trajectories_compare.py \
  --input trajectories_dataset.json \
  --algo hungarian

python visualize_trajectories_compare.py \
  --input trajectories_dataset.json \
  --algo bipartite
```

---

## Structure du projet

```
.
├── link_trajectories.py              # Algorithme de liaison (importable + CLI)
├── visualize_trajectories.py         # Rendu vidéo OpenCV
├── compare_algorithms.py             # Benchmark des 3 algorithmes
├── visualize_trajectories_compare.py # Vidéos comparatives (glouton / hongrois / biparti)
├── trajectories_dataset.json         # Données d'entrée
├── trajectories_linked.mp4           # Vidéo produite (hongrois, défaut)
├── trajectories_linked_greedy.mp4    # Vidéo produite (glouton)
├── trajectories_linked_hungarian.mp4 # Vidéo produite (hongrois)
├── trajectories_linked_bipartite.mp4 # Vidéo produite (biparti + fantômes)
├── requirements.txt
└── README.md
```

---

## Format des données

```json
{
  "trajectories": [
    {
      "id": "traj_001",
      "startTime": "2026-03-17T10:00:00Z",
      "endTime":   "2026-03-17T10:00:27.25Z",
      "points": [x0, y0, x1, y1, ...]
    }
  ]
}
```

Les points sont stockés comme une liste plate de flottants : `[x0, y0, x1, y1, …]`.

---

## Partie 1 — Liaison et visualisation

### Principe

Les trajectoires sont des fragments d'un mouvement continu, segmentés par exemple lors d'occlusions ou de sorties de champ.
L'objectif est de reconnecter ces fragments en **chaînes cohérentes** en s'appuyant sur deux critères :

| Critère | Condition | Paramètre |
|---|---|---|
| **Temporel** | `0 ≤ start(B) − end(A) ≤ MAX_GAP` | `--max_gap` (défaut : 10 s) |
| **Spatial** | `dist(last(A), first(B)) ≤ MAX_DIST` | `--max_dist` (défaut : 50 px) |

Quand plusieurs candidats satisfont les deux critères, le meilleur est sélectionné via un score normalisé :

```
score = w_dist × (dist / MAX_DIST) + w_time × (gap / MAX_GAP)
```

L'affectation est réalisée par l'**algorithme Hongrois** (`scipy.optimize.linear_sum_assignment`), qui garantit l'optimum global en minimisant le coût total de l'ensemble des liaisons simultanément. Les paires liées sont ensuite enchaînées en séquences : `traj_003 → traj_004 → traj_005 → …`

---

### 1. `link_trajectories.py`

Calcule les liaisons entre trajectoires et affiche les résultats dans le terminal.

**Commande minimale :**

```bash
python link_trajectories.py --input trajectories_dataset.json
```

**Sortie attendue :**

```
=======================================================
  Trajectories : 56
  Links found  : 21
  Chains       : 35  (of which 15 multi-segment)
=======================================================

Pairwise links:
  traj_006 -> traj_007   gap=2.02s   dist=2.2px
  traj_030 -> traj_031   gap=2.16s   dist=2.7px
  ...

Chains:
  Chain 01 (3 segs): traj_026 -> traj_027 -> traj_028
  Chain 02 (3 segs): traj_010 -> traj_011 -> traj_012
  ...
```

**Toutes les options :**

```bash
python link_trajectories.py \
  --input trajectories_dataset.json \
  --max_gap 10.0 \
  --max_dist 50.0 \
  --w_dist 1.0 \
  --w_time 2.0
```

| Option | Type | Défaut | Description |
|---|---|---|---|
| `--input` | str | *(requis)* | Chemin vers le fichier JSON |
| `--max_gap` | float | 10.0 | Gap temporel max en secondes |
| `--max_dist` | float | 50.0 | Distance spatiale max en pixels |
| `--w_dist` | float | 1.0 | Poids du terme spatial dans le score |
| `--w_time` | float | 2.0 | Poids du terme temporel dans le score |

Le module est aussi **importable** dans un autre script Python :

```python
from link_trajectories import build_links_hungarian, build_chains

with open("trajectories_dataset.json") as f:
    data = json.load(f)

links  = build_links_hungarian(data["trajectories"])
chains = build_chains(data["trajectories"], links)
```

---

### 2. `visualize_trajectories.py`

Génère une vidéo MP4 animant les trajectoires et leurs liaisons dans le temps.

**Commande minimale :**

```bash
python visualize_trajectories.py --input trajectories_dataset.json
```

Produit le fichier `trajectories_linked.mp4` dans le répertoire courant.

**Commande complète avec toutes les options :**

```bash
python visualize_trajectories.py \
  --input trajectories_dataset.json \
  --output trajectories_linked.mp4 \
  --max_gap 10.0 \
  --max_dist 50.0 \
  --fps 30 \
  --width 1280 \
  --height 720
```

| Option | Type | Défaut | Description |
|---|---|---|---|
| `--input` | str | *(requis)* | Chemin vers le fichier JSON |
| `--output` | str | trajectories_linked.mp4 | Chemin de la vidéo de sortie |
| `--max_gap` | float | 10.0 | Gap temporel max en secondes |
| `--max_dist` | float | 50.0 | Distance spatiale max en pixels |
| `--fps` | int | 30 | Images par seconde |
| `--width` | int | 1280 | Largeur en pixels |
| `--height` | int | 720 | Hauteur en pixels |

**Contenu de la vidéo :**
- chaque chaîne animée dans une couleur distincte
- les liaisons entre segments représentées en **lignes pointillées**
- une barre de progression temporelle en bas de l'image
- une légende des chaînes multi-segments en haut à gauche

> **Note :** la durée de rendu est d'environ 30 secondes pour ce dataset à 30 fps et 1280×720.

---

## Partie 2 — Comparaison des algorithmes

### 3. `compare_algorithms.py`

Compare trois algorithmes de liaison sur la **même matrice de coûts** et affiche un tableau de métriques : nombre de liaisons, coût total, coût moyen, coût max, nombre de chaînes et temps d'exécution.

**Commande minimale :**

```bash
python compare_algorithms.py --input trajectories_dataset.json
```

**Sortie attendue :**

```
============================================================
  Dataset : 56 trajectoires
  Seuils  : gap ≤ 10.0s, dist ≤ 50.0px
============================================================

Matrice de coûts (56×56) : 21 paires valides   [2.06 ms]

Algorithme                   Liaisons   Coût total   Coût moy   Coût max  Chaînes multi   Temps (ms)
----------------------------------------------------------------------------------------------------
1. Glouton          O(n²)          21      18.8814     0.8991     1.4342             15        0.606
2. Hongrois         O(n³)          21      18.8814     0.8991     1.4342             15        0.117
3. Biparti+fantômes O(n³)          14      10.6681     0.7620     0.9779             12        0.490
----------------------------------------------------------------------------------------------------

Différences entre solutions :
  Glouton et Hongrois : solutions IDENTIQUES
  ...

→ Gain coût total Hongrois vs Glouton : +0.00%
→ Recommandation : Hongrois (scipy.optimize.linear_sum_assignment)
```

**Commande complète avec toutes les options :**

```bash
python compare_algorithms.py \
  --input trajectories_dataset.json \
  --max_gap 10.0 \
  --max_dist 50.0 \
  --ghost 0.5
```

| Option | Type | Défaut | Description |
|---|---|---|---|
| `--input` | str | *(requis)* | Chemin vers le fichier JSON |
| `--max_gap` | float | 10.0 | Gap temporel max en secondes |
| `--max_dist` | float | 50.0 | Distance spatiale max en pixels |
| `--ghost` | float | 0.5 | Coût de rejet pour l'algorithme biparti |

Pour tester l'effet du seuil de rejet du biparti, par exemple :

```bash
# Biparti très permissif (accepte presque tout)
python compare_algorithms.py --input trajectories_dataset.json --ghost 2.0

# Biparti très conservateur (rejette beaucoup)
python compare_algorithms.py --input trajectories_dataset.json --ghost 0.1
```

---

### 4. `visualize_trajectories_compare.py`

Génère trois vidéos distinctes — une par algorithme — pour comparer visuellement les différences de liaison.

**Commande minimale :**

```bash
python visualize_trajectories_compare.py --input trajectories_dataset.json
```

Produit trois fichiers dans le répertoire courant :

```
trajectories_linked_greedy.mp4      # liaisons du glouton
trajectories_linked_hungarian.mp4   # liaisons de l'algorithme Hongrois
trajectories_linked_bipartite.mp4   # liaisons du biparti + fantômes
```

**Commande complète :**

```bash
python visualize_trajectories_compare.py \
  --input trajectories_dataset.json \
  --max_gap 10.0 \
  --max_dist 50.0 \
  --ghost 0.5 \
  --fps 30 \
  --width 1280 \
  --height 720
```

| Option | Type | Défaut | Description |
|---|---|---|---|
| `--input` | str | *(requis)* | Chemin vers le fichier JSON |
| `--max_gap` | float | 10.0 | Gap temporel max en secondes |
| `--max_dist` | float | 50.0 | Distance spatiale max en pixels |
| `--ghost` | float | 0.5 | Coût de rejet pour l'algorithme biparti |
| `--fps` | int | 30 | Images par seconde |
| `--width` | int | 1280 | Largeur en pixels |
| `--height` | int | 720 | Hauteur en pixels |

---

## Analyse : pourquoi le glouton suffit ici

Sur ce dataset, le glouton et l'algorithme Hongrois produisent **exactement la même solution**. Ce n'est pas une coïncidence, c'est la conséquence directe de la structure des données.

L'algorithme Hongrois est utile quand plusieurs trajectoires sont candidates à se lier à la même cible — il faut alors arbitrer globalement. Cela se produit uniquement lorsque le **graphe de candidats est dense** : beaucoup de paires valides en compétition les unes avec les autres.

Ici, sur une matrice 56×56 = 3 136 paires possibles, **seulement 21 sont valides** (0,67 % de remplissage). Le graphe est extrêmement creux. Chaque trajectoire a au plus un ou deux candidats de liaison, sans chevauchement. Il n'y a donc aucun conflit d'affectation à résoudre, et le glouton tombe naturellement sur l'optimum global.

En pratique, l'Hongrois devient indispensable dès que les trajectoires sont plus nombreuses, plus rapprochées dans l'espace, ou que les seuils sont plus larges — situations dans lesquelles plusieurs paires valides entrent en compétition pour la même cible.

---

## Points forts et limites de chaque algorithme

### 1. Glouton — O(n²)

Trie toutes les paires valides par score croissant et les affecte dans l'ordre, en garantissant qu'une trajectoire n'apparaît qu'une fois comme source et une fois comme cible.

**Points forts**
- Simple à implémenter et à comprendre
- Très rapide en pratique sur des graphes creux
- Résultat optimal quand il n'y a pas de conflits d'affectation (cas de ce dataset)

**Limites**
- Ne garantit pas l'optimum global : si deux sources veulent la même cible, il prend la moins coûteuse sans vérifier si cela bloque une meilleure liaison en aval
- Fragile sur les datasets denses où les conflits sont fréquents

---

### 2. Algorithme Hongrois (Munkres) — O(n³)

Résout le problème d'affectation linéaire en minimisant le coût total de toutes les liaisons simultanément. Implémenté via `scipy.optimize.linear_sum_assignment`.

**Points forts**
- **Optimum global garanti** : minimise la somme totale des coûts de liaison
- Référence industrielle pour le suivi multi-objets (MOT challenge, radar, vision)
- 0.12 ms sur ce dataset — négligeable même à grande échelle pour n < 1 000

**Limites**
- Complexité O(n³) : devient lent pour n > 5 000 trajectoires sans optimisation
- Force une affectation 1-à-1 complète : toutes les trajectoires doivent être assignées, même si aucune liaison pertinente n'existe — il faut filtrer les liaisons à coût infini en post-traitement
- Ne modélise pas explicitement la possibilité qu'un objet disparaisse (pas de coût de rejet natif)

---

### 3. Graphe biparti + nœuds fantômes — O(n³)

Variante inspirée des trackers temps réel SORT et ByteTrack. On augmente la matrice de coûts avec des **nœuds fantômes** : chaque source peut se lier à un fantôme (l'objet disparaît) plutôt qu'être forcée vers une cible réelle sous-optimale. Le coût du fantôme est un seuil de rejet explicite.

**Points forts**
- Modélise naturellement les **naissances et morts d'objets** : une trajectoire peut légitimement ne pas être liée si aucun candidat ne justifie le coût
- Plus conservateur que l'Hongrois pur : rejette les liaisons douteuses plutôt que de forcer une affectation
- Adapté aux scènes où des objets entrent et sortent du champ fréquemment

**Limites**
- Le paramètre `ghost_cost` est un seuil à calibrer manuellement : trop bas et l'algorithme refuse toutes les liaisons, trop haut et il se comporte comme l'Hongrois pur
- Sur ce dataset, il rejette 7 liaisons valides car leur coût dépasse 0.5 — comportement souhaité ou non selon le contexte
- Complexité O((2n)³) : constante 8× plus grande que l'Hongrois, même classe asymptotique

---

## Résultats sur le dataset fourni

Paramètres : `max_gap=10s`, `max_dist=50px`, `ghost_cost=0.5`

| Algorithme | Complexité | Liaisons | Coût total | Coût moyen | Chaînes multi | Temps |
|---|---|---|---|---|---|---|
| Glouton | O(n²) | 21 | 18.88 | 0.90 | 15 | 0.6 ms |
| Hongrois | O(n³) | 21 | 18.88 | 0.90 | 15 | 0.1 ms |
| Biparti + fantômes | O(n³) | 14 | 10.67 | 0.76 | 12 | 0.5 ms |

**56 trajectoires** → **21 liaisons** → **35 chaînes** (dont 15 multi-segments)

Chaînes les plus longues reconstruites :

```
traj_026 → traj_027 → traj_028
traj_010 → traj_011 → traj_012
traj_003 → traj_004 → traj_005
traj_023 → traj_024 → traj_025
traj_029 → traj_030 → traj_031
```

Les liaisons présentent des distances sub-pixel (< 5 px) et des gaps inférieurs à 6 s, confirmant que les fragments appartiennent bien aux mêmes objets en mouvement.