# Trajectory Linking Algorithm

Algorithme de liaison de trajectoires fragmentées basé sur des critères **spatio-temporels**.

## Principe de l'algorithme

Les trajectoires sont des fragments d'un mouvement continu qui ont été segmentés (occlusions, sorties de champ, etc.). L'objectif est de reconnecter ces fragments en **chaînes** cohérentes.

### Critères de liaison

Pour relier une trajectoire **A** à une trajectoire **B** :

| Critère | Condition | Paramètre |
|---|---|---|
| **Temporel** | `0 ≤ start(B) − end(A) ≤ MAX_GAP` | `--max_gap` (défaut : 10 s) |
| **Spatial** | `dist(last(A), first(B)) ≤ MAX_DIST` | `--max_dist` (défaut : 50 px) |

### Score combiné (optimisation)

Quand plusieurs candidats satisfont les deux critères, le meilleur est choisi via :

```
score = w_dist × (dist / MAX_DIST) + w_time × (gap / MAX_GAP)
```

Un score faible signifie une transition spatiale et temporelle courte → meilleure liaison.

### Affectation (greedy)

Les liaisons sont triées par score croissant et affectées de façon gloutonne, en garantissant que chaque trajectoire apparaît **au plus une fois** comme source et **au plus une fois** comme cible.

### Reconstruction des chaînes

Les paires liées sont enchaînées en séquences : `traj_003 → traj_004 → traj_005 → …`

---

## Installation

```bash
# Python 3.10+
pip install opencv-python numpy
```

## Utilisation

### 1. Afficher les liaisons (texte)

```bash
python link_trajectories.py --input trajectories_dataset.json
```

Options disponibles :

```
--max_gap   float   Gap temporel max en secondes        [défaut: 10.0]
--max_dist  float   Distance spatiale max en pixels     [défaut: 50.0]
--w_dist    float   Poids du terme spatial dans le score [défaut: 1.0]
--w_time    float   Poids du terme temporel dans le score [défaut: 2.0]
```

### 2. Générer la vidéo

```bash
python visualize_trajectories.py --input trajectories_dataset.json --output trajectories_linked.mp4
```

Options supplémentaires :

```
--output    str     Chemin de la vidéo de sortie        [défaut: trajectories_linked.mp4]
--fps       int     Images par seconde                   [défaut: 30]
--width     int     Largeur vidéo en pixels             [défaut: 1280]
--height    int     Hauteur vidéo en pixels             [défaut: 720]
```

---

## Structure du projet

```
.
├── link_trajectories.py       # Algorithme de liaison (importable + CLI)
├── visualize_trajectories.py  # Rendu vidéo OpenCV
├── trajectories_dataset.json  # Données d'entrée
└── README.md
```

## Format des données d'entrée

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

## Résultats sur le dataset fourni

Avec les paramètres par défaut (`max_gap=10s`, `max_dist=50px`) :

- **56 trajectoires** en entrée
- **~26 liaisons** trouvées
- **~30 chaînes** reconstruites (dont plusieurs multi-segments)

Les liaisons présentent des distances sub-pixel (< 5 px) et des gaps < 10 s, ce qui indique que les fragments appartiennent bien aux mêmes objets en mouvement.
