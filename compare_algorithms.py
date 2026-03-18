"""
Comparaison de 3 algorithmes de liaison de trajectoires
=========================================================
1. Glouton (baseline)             — O(n²)
2. Hongrois / Munkres             — O(n³) via scipy  [RECOMMANDÉ]
3. Graphe biparti + min-cost flow — O(n² log n) via scipy sparse

Tous les algorithmes partagent la même matrice de coûts et les mêmes seuils,
ce qui rend la comparaison équitable.

Usage:
    python compare_algorithms.py --input trajectories_dataset.json
"""

import json
import math
import time
import argparse
from datetime import datetime
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

# -- Paramètres par défaut --
MAX_GAP_S   = 10.0    # secondes
MAX_DIST_PX = 50.0    # pixels
W_DIST      = 1.0
W_TIME      = 2.0
INF_COST    = 1e9     # coût infini pour paires invalides


# -- Helpers --
def parse_iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))

def last_pt(t):  return t["points"][-2], t["points"][-1]
def first_pt(t): return t["points"][0],  t["points"][1]

def euclidean(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def combined_score(gap, dist, max_gap=MAX_GAP_S, max_dist=MAX_DIST_PX,
                   w_d=W_DIST, w_t=W_TIME):
    """Score normalisé : plus bas = meilleure liaison."""
    return w_d * (dist / max_dist) + w_t * (gap / max_gap)


# -- Construction de la matrice de coûts --
def build_cost_matrix(trajs, max_gap=MAX_GAP_S, max_dist=MAX_DIST_PX):
    """
    Matrice n×n :  coût[i][j] = score(fin de i → début de j)
    INF_COST si hors seuils (gap négatif, trop long, trop loin).
    """
    n = len(trajs)
    C = np.full((n, n), INF_COST, dtype=np.float64)
    for i, a in enumerate(trajs):
        end_a   = parse_iso(a["endTime"])
        last_a  = last_pt(a)
        for j, b in enumerate(trajs):
            if i == j: continue
            gap = (parse_iso(b["startTime"]) - end_a).total_seconds()
            if not (0 <= gap <= max_gap): continue
            dist = euclidean(last_a, first_pt(b))
            if dist > max_dist: continue
            C[i, j] = combined_score(gap, dist, max_gap, max_dist)
    return C


# -- 1. Glouton --
def algo_greedy(trajs, cost_matrix):
    """
    Trie toutes les paires valides par score et les affecte dans l'ordre croissant
    (chaque trajectoire : 1 source max, 1 cible max).
    Complexité : O(n² log n) — triage de la liste aplatie.
    """
    n = len(trajs)
    pairs = []
    for i in range(n):
        for j in range(n):
            c = cost_matrix[i, j]
            if c < INF_COST:
                pairs.append((c, i, j))
    pairs.sort()

    used_src, used_tgt = set(), set()
    links = []
    for c, i, j in pairs:
        if i in used_src or j in used_tgt: continue
        links.append((trajs[i]["id"], trajs[j]["id"], c))
        used_src.add(i); used_tgt.add(j)
    return links


# -- 2. Algorithme Hongrois (Munkres) --
def algo_hungarian(trajs, cost_matrix):
    """
    scipy.optimize.linear_sum_assignment résout l'affectation optimale globale.
    Complexité : O(n³) — algorithme de Munkres.

    Le problème : minimiser sum(C[i, row_ind[i]]) sur toutes les affectations.

    Attention : l'algorithme force une affectation complète n→n.
    On filtre ensuite les liaisons dont le coût est infini (= paires invalides).
    """
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    links = []
    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] < INF_COST:
            links.append((trajs[i]["id"], trajs[j]["id"], cost_matrix[i, j]))
    return links


# -- 3. Graphe biparti + nœuds fantômes --
def algo_bipartite_ghost(trajs, cost_matrix, ghost_cost=0.5):
    """
    Variante inspirée de SORT/ByteTrack.

    On augmente la matrice n×n avec des nœuds fantômes (ghost nodes) :
    - Chaque source i a un fantôme dans la colonne (n+i) au coût `ghost_cost`
    - Chaque cible j a un fantôme dans la ligne (n+j) au coût `ghost_cost`
    → Un objet peut "mourir" (liaison vers fantôme) ou "naître" (depuis fantôme)
      sans forcer une liaison réelle sous-optimale.

    La matrice augmentée (2n×2n) est passée à l'algorithme Hongrois.
    Seules les liaisons source réelle → cible réelle sont conservées.

    Complexité : O((2n)³) = O(n³) — constants plus larges mais même classe.
    """
    n = len(trajs)
    # Matrice augmentée 2n × 2n initialisée à INF_COST
    C_aug = np.full((2*n, 2*n), INF_COST, dtype=np.float64)

    # Bloc supérieur-gauche : coûts réels
    C_aug[:n, :n] = cost_matrix

    # Fantômes sources (ligne i → colonne n+i) : coût de "ne pas lier"
    for i in range(n):
        C_aug[i, n+i] = ghost_cost

    # Fantômes cibles (ligne n+j → colonne j) : coût de "naître sans passé"
    for j in range(n):
        C_aug[n+j, j] = ghost_cost

    # Bloc inférieur-droit : petite valeur pour permettre affectation des fantômes entre eux
    C_aug[n:, n:] = 0.0

    row_ind, col_ind = linear_sum_assignment(C_aug)
    links = []
    for i, j in zip(row_ind, col_ind):
        if i < n and j < n and C_aug[i, j] < INF_COST:
            links.append((trajs[i]["id"], trajs[j]["id"], C_aug[i, j]))
    return links


# -- Reconstruction des chaînes --
def build_chains(trajs, links):
    succ = {a: b for a, b, *_ in links}
    pred = {b: a for a, b, *_ in links}
    traj_ids = {t["id"] for t in trajs}
    visited, chains = set(), []
    for tid in traj_ids:
        if tid in pred or tid in visited: continue
        chain = [tid]; visited.add(tid)
        cur = tid
        while cur in succ:
            nxt = succ[cur]
            if nxt in visited: break
            chain.append(nxt); visited.add(nxt); cur = nxt
        chains.append(chain)
    return chains


# -- Métriques de qualité --
def quality_metrics(trajs, links):
    costs = [c for _, _, c in links]
    return {
        "n_links"      : len(links),
        "total_cost"   : sum(costs),
        "mean_cost"    : sum(costs)/len(costs) if costs else 0,
        "max_cost"     : max(costs) if costs else 0,
        "n_chains_multi": sum(1 for c in build_chains(trajs, links) if len(c) > 1),
    }


# -- Différences entre solutions --
def diff_solutions(name_a, links_a, name_b, links_b):
    set_a = {(a, b) for a, b, *_ in links_a}
    set_b = {(a, b) for a, b, *_ in links_b}
    only_a = set_a - set_b
    only_b = set_b - set_a
    if not only_a and not only_b:
        print(f"  {name_a} et {name_b} : solutions IDENTIQUES")
    else:
        print(f"  {name_a} only : {sorted(only_a)}")
        print(f"  {name_b} only : {sorted(only_b)}")


# -- Main --
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",    required=True)
    parser.add_argument("--max_gap",  type=float, default=MAX_GAP_S)
    parser.add_argument("--max_dist", type=float, default=MAX_DIST_PX)
    parser.add_argument("--ghost",    type=float, default=0.5,
                        help="Coût fantôme algo biparti (défaut 0.5)")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)
    trajs = data["trajectories"]
    n = len(trajs)

    print(f"\n{'='*60}")
    print(f"  Dataset : {n} trajectoires")
    print(f"  Seuils  : gap ≤ {args.max_gap}s, dist ≤ {args.max_dist}px")
    print(f"{'='*60}\n")

    # Construction de la matrice (commune aux 3)
    t0 = time.perf_counter()
    C = build_cost_matrix(trajs, args.max_gap, args.max_dist)
    t_matrix = time.perf_counter() - t0
    valid = int(np.sum(C < INF_COST))
    print(f"Matrice de coûts ({n}×{n}) : {valid} paires valides   [{t_matrix*1000:.2f} ms]\n")

    # -- Algo 1 : Glouton --
    t0 = time.perf_counter()
    links_greedy = algo_greedy(trajs, C)
    t_greedy = time.perf_counter() - t0
    m_greedy = quality_metrics(trajs, links_greedy)

    # -- Algo 2 : Hongrois --
    t0 = time.perf_counter()
    links_hungarian = algo_hungarian(trajs, C)
    t_hungarian = time.perf_counter() - t0
    m_hungarian = quality_metrics(trajs, links_hungarian)

    # -- Algo 3 : Biparti + fantômes --
    t0 = time.perf_counter()
    links_bipartite = algo_bipartite_ghost(trajs, C, ghost_cost=args.ghost)
    t_bipartite = time.perf_counter() - t0
    m_bipartite = quality_metrics(trajs, links_bipartite)

    # -- Résultats --
    header = f"{'Algorithme':<28} {'Liaisons':>8} {'Coût total':>12} {'Coût moy':>10} {'Coût max':>10} {'Chaînes multi':>14} {'Temps (ms)':>12}"
    sep = "-" * len(header)
    print(header)
    print(sep)

    def row(name, m, t):
        return (f"{name:<28} {m['n_links']:>8} {m['total_cost']:>12.4f} "
                f"{m['mean_cost']:>10.4f} {m['max_cost']:>10.4f} "
                f"{m['n_chains_multi']:>14} {t*1000:>12.3f}")

    print(row("1. Glouton          O(n²)", m_greedy,    t_greedy))
    print(row("2. Hongrois         O(n³)", m_hungarian, t_hungarian))
    print(row("3. Biparti+fantômes O(n³)", m_bipartite, t_bipartite))
    print(sep)

    # -- Différences --
    print("\nDifférences entre solutions :")
    diff_solutions("Glouton",  links_greedy,    "Hongrois",  links_hungarian)
    diff_solutions("Hongrois", links_hungarian, "Biparti",   links_bipartite)
    diff_solutions("Glouton",  links_greedy,    "Biparti",   links_bipartite)

    # -- Gain de l'Hongrois sur le Glouton --
    if m_greedy["total_cost"] > 0:
        gain = 100 * (m_greedy["total_cost"] - m_hungarian["total_cost"]) / m_greedy["total_cost"]
        print(f"\n→ Gain coût total Hongrois vs Glouton : {gain:+.2f}%")
    
    print(f"\n→ Recommandation : Hongrois (scipy.optimize.linear_sum_assignment)")
    print(f"  Optimal global, {t_hungarian*1000:.2f} ms pour {n} trajectoires.")
    print(f"  Le biparti+fantômes est utile si certaines trajectoires ne doivent")
    print(f"  PAS être liées (objets qui disparaissent, coût de rejet explicite).\n")

    return links_greedy, links_hungarian, links_bipartite


if __name__ == "__main__":
    main()
