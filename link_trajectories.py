"""
Trajectory Linking Algorithm
=============================
Connects fragmented trajectories using spatio-temporal criteria:
  - Temporal gap between end of A and start of B must be within [0, MAX_GAP_SECONDS]
  - Spatial distance between last point of A and first point of B must be <= MAX_DIST_PX
  - Among all candidates, the best link minimises a combined score: w_d * dist + w_t * gap

Usage:
    python link_trajectories.py --input trajectories_dataset.json [options]
    python link_trajectories.py --help
"""

import json
import math
import argparse
from datetime import datetime, timezone
from collections import defaultdict


# -- Default thresholds --
DEFAULT_MAX_GAP   = 10.0   # seconds
DEFAULT_MAX_DIST  = 50.0   # pixels
DEFAULT_W_DIST    = 1.0    # weight for spatial term in combined score
DEFAULT_W_TIME    = 2.0    # weight for temporal term in combined score


# -- Helpers --
def parse_iso(ts: str) -> datetime:
    """Parse ISO-8601 timestamp (with or without trailing Z)."""
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def last_point(traj: dict) -> tuple[float, float]:
    pts = traj["points"]
    return pts[-2], pts[-1]


def first_point(traj: dict) -> tuple[float, float]:
    pts = traj["points"]
    return pts[0], pts[1]


def euclidean(p1: tuple, p2: tuple) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# -- Core algorithm --
def build_links(
    trajectories: list[dict],
    max_gap: float = DEFAULT_MAX_GAP,
    max_dist: float = DEFAULT_MAX_DIST,
    w_dist: float = DEFAULT_W_DIST,
    w_time: float = DEFAULT_W_TIME,
) -> list[tuple[str, str, float, float]]:
    """
    For each trajectory A, find the best successor B such that:
      1. start(B) > end(A)                      — B starts after A ends
      2. gap = start(B) - end(A) <= max_gap     — temporal proximity
      3. dist(last(A), first(B)) <= max_dist    — spatial proximity
      4. score = w_dist * dist + w_time * gap   — combined optimality

    Returns a list of (id_A, id_B, gap_s, dist_px) tuples,
    ensuring each trajectory appears at most once as a source and once as a target.
    """
    # Normalise thresholds for scoring (avoid division-by-zero)
    norm_d = max_dist if max_dist > 0 else 1.0
    norm_t = max_gap  if max_gap  > 0 else 1.0

    # Pre-compute endpoints and times
    meta = {}
    for t in trajectories:
        meta[t["id"]] = {
            "end_t":   parse_iso(t["endTime"]),
            "start_t": parse_iso(t["startTime"]),
            "last_pt": last_point(t),
            "first_pt": first_point(t),
        }

    # Gather all valid candidates (a -> b)
    candidates: dict[str, list] = defaultdict(list)  # key = source id
    for a in trajectories:
        ma = meta[a["id"]]
        for b in trajectories:
            if a["id"] == b["id"]:
                continue
            mb = meta[b["id"]]
            gap = (mb["start_t"] - ma["end_t"]).total_seconds()
            if not (0 <= gap <= max_gap):
                continue
            dist = euclidean(ma["last_pt"], mb["first_pt"])
            if dist > max_dist:
                continue
            # Normalised combined score (lower = better)
            score = w_dist * (dist / norm_d) + w_time * (gap / norm_t)
            candidates[a["id"]].append((score, gap, dist, b["id"]))

    # Greedy assignment: each trajectory is used at most once per role
    used_as_source = set()
    used_as_target = set()
    links = []

    # Sort all potential links globally by score so best ones are assigned first
    all_links = []
    for src_id, cands in candidates.items():
        for score, gap, dist, tgt_id in cands:
            all_links.append((score, gap, dist, src_id, tgt_id))
    all_links.sort()

    for score, gap, dist, src_id, tgt_id in all_links:
        if src_id in used_as_source or tgt_id in used_as_target:
            continue
        links.append((src_id, tgt_id, gap, dist))
        used_as_source.add(src_id)
        used_as_target.add(tgt_id)

    return links


# -- Chain reconstruction --
def build_chains(trajectories: list[dict], links: list[tuple]) -> list[list[str]]:
    """
    From pairwise links, reconstruct full chains (sequences of connected trajectories).
    """
    succ = {a: b for a, b, *_ in links}
    pred = {b: a for a, b, *_ in links}

    traj_ids = {t["id"] for t in trajectories}
    visited = set()
    chains = []

    for tid in traj_ids:
        # Start a chain only from a trajectory that has no predecessor
        if tid in pred or tid in visited:
            continue
        chain = [tid]
        visited.add(tid)
        current = tid
        while current in succ:
            nxt = succ[current]
            if nxt in visited:
                break
            chain.append(nxt)
            visited.add(nxt)
            current = nxt
        chains.append(chain)

    # Isolated trajectories (singletons) are already included above
    return chains


# -- CLI entry point --
def main():
    parser = argparse.ArgumentParser(description="Trajectory linking algorithm")
    parser.add_argument("--input",    required=True, help="Path to JSON dataset")
    parser.add_argument("--max_gap",  type=float, default=DEFAULT_MAX_GAP,
                        help=f"Max temporal gap in seconds (default {DEFAULT_MAX_GAP})")
    parser.add_argument("--max_dist", type=float, default=DEFAULT_MAX_DIST,
                        help=f"Max spatial distance in pixels (default {DEFAULT_MAX_DIST})")
    parser.add_argument("--w_dist",   type=float, default=DEFAULT_W_DIST,
                        help=f"Weight for spatial term (default {DEFAULT_W_DIST})")
    parser.add_argument("--w_time",   type=float, default=DEFAULT_W_TIME,
                        help=f"Weight for temporal term (default {DEFAULT_W_TIME})")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)
    trajectories = data["trajectories"]

    links = build_links(
        trajectories,
        max_gap=args.max_gap,
        max_dist=args.max_dist,
        w_dist=args.w_dist,
        w_time=args.w_time,
    )

    chains = build_chains(trajectories, links)

    print(f"\n{'='*55}")
    print(f"  Trajectories : {len(trajectories)}")
    print(f"  Links found  : {len(links)}")
    print(f"  Chains       : {len(chains)}  "
          f"(of which {sum(1 for c in chains if len(c)>1)} multi-segment)")
    print(f"{'='*55}\n")

    print("Pairwise links:")
    for src, tgt, gap, dist in sorted(links, key=lambda x: x[2]):
        print(f"  {src} -> {tgt}   gap={gap:.2f}s   dist={dist:.1f}px")

    print("\nChains:")
    for i, chain in enumerate(sorted(chains, key=len, reverse=True)):
        print(f"  Chain {i+1:02d} ({len(chain)} segs): {' -> '.join(chain)}")

    return links, chains


if __name__ == "__main__":
    main()
