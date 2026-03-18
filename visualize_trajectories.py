"""
Trajectory Visualisation — OpenCV
===================================
Renders an MP4 video showing:
  - Each trajectory drawn in its chain colour, animated over time
  - Dashed lines representing links between consecutive trajectory segments
  - A timeline bar at the bottom
  - Legend with chain IDs

Usage:
    python visualize_trajectories.py --input trajectories_dataset.json [options]
"""

import json
import math
import argparse
import colorsys
from datetime import datetime

import cv2
import numpy as np

# Local import
from link_trajectories import build_links, build_chains, parse_iso


# -- Colour helpers --
def chain_palette(n: int) -> list[tuple[int, int, int]]:
    """Generate n visually distinct BGR colours."""
    colours = []
    for i in range(n):
        hue = i / max(n, 1)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
        colours.append((int(b * 255), int(g * 255), int(r * 255)))  # BGR
    return colours


def draw_dashed_line(img, p1, p2, colour, thickness=2, dash=12, gap=8):
    """Draw a dashed line between two points."""
    x1, y1 = p1
    x2, y2 = p2
    length = math.hypot(x2 - x1, y2 - y1)
    if length == 0:
        return
    dx, dy = (x2 - x1) / length, (y2 - y1) / length
    pos = 0.0
    drawing = True
    while pos < length:
        seg = dash if drawing else gap
        end = min(pos + seg, length)
        if drawing:
            pa = (int(x1 + dx * pos),  int(y1 + dy * pos))
            pb = (int(x1 + dx * end),  int(y1 + dy * end))
            cv2.line(img, pa, pb, colour, thickness, cv2.LINE_AA)
        pos = end
        drawing = not drawing


# -- Main rendering --
def render_video(
    trajectories: list[dict],
    links: list[tuple],
    chains: list[list[str]],
    output_path: str = "trajectories_linked.mp4",
    fps: int = 30,
    margin: int = 60,
    width: int = 1280,
    height: int = 720,
    trail_alpha: float = 0.35,
):
    # -- Coordinate mapping --
    all_pts = []
    for t in trajectories:
        pts = t["points"]
        for i in range(0, len(pts), 2):
            all_pts.append((pts[i], pts[i + 1]))

    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    TIMELINE_H = 50   # pixels reserved at bottom for timeline bar

    def world_to_screen(x, y):
        sx = int(margin + (x - x_min) / (x_max - x_min) * (width  - 2 * margin))
        sy = int(margin + (y - y_min) / (y_max - y_min) * (height - 2 * margin - TIMELINE_H))
        return sx, sy

    # -- Temporal mapping --
    t_start_global = min(parse_iso(t["startTime"]) for t in trajectories)
    t_end_global   = max(parse_iso(t["endTime"])   for t in trajectories)
    total_duration = (t_end_global - t_start_global).total_seconds()

    # -- Index structures --
    traj_by_id = {t["id"]: t for t in trajectories}

    # Assign each trajectory to a chain index
    traj_chain_idx = {}
    for ci, chain in enumerate(chains):
        for tid in chain:
            traj_chain_idx[tid] = ci

    # Build link lookup: src_id -> (tgt_id, last_pt_src, first_pt_tgt)
    link_map = {}
    for src, tgt, gap, dist in links:
        t_src = traj_by_id[src]
        t_tgt = traj_by_id[tgt]
        p_src = (t_src["points"][-2], t_src["points"][-1])
        p_tgt = (t_tgt["points"][0],  t_tgt["points"][1])
        link_map[src] = (tgt, p_src, p_tgt,
                         parse_iso(t_src["endTime"]),
                         parse_iso(t_tgt["startTime"]))

    colours = chain_palette(len(chains))

    # -- Build frame list --
    # Each "animation step" = 0.25 s of real time
    TIME_STEP   = 0.25   # seconds per animation frame
    PAUSE_END   = int(fps * 2)  # 2-second hold at the end

    n_steps = int(math.ceil(total_duration / TIME_STEP)) + 1
    video_frames = n_steps + PAUSE_END

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps,
                             (width, height + TIMELINE_H))

    # Precompute per-trajectory screen point arrays
    screen_pts = {}
    for t in trajectories:
        pts = t["points"]
        arr = []
        for i in range(0, len(pts), 2):
            arr.append(world_to_screen(pts[i], pts[i + 1]))
        screen_pts[t["id"]] = arr

    print(f"Rendering {video_frames} frames at {fps} fps …")

    for frame_idx in range(video_frames):
        sim_time = min(frame_idx * TIME_STEP, total_duration)
        canvas = np.zeros((height + TIMELINE_H, width, 3), dtype=np.uint8)
        canvas[:] = (20, 20, 25)   # dark background

        overlay = canvas.copy()

        for t in trajectories:
            t_s = (parse_iso(t["startTime"]) - t_start_global).total_seconds()
            t_e = (parse_iso(t["endTime"])   - t_start_global).total_seconds()

            if sim_time < t_s:
                continue   # not yet started

            ci    = traj_chain_idx[t["id"]]
            colour = colours[ci]
            spts  = screen_pts[t["id"]]
            n_pts = len(spts)

            # How many points to draw
            if sim_time >= t_e:
                draw_count = n_pts
            else:
                frac = (sim_time - t_s) / max(t_e - t_s, 1e-9)
                draw_count = max(2, int(frac * n_pts))

            draw_count = min(draw_count, n_pts)

            # Trail (faded)
            if draw_count > 1:
                pts_array = np.array(spts[:draw_count], dtype=np.int32)
                cv2.polylines(overlay, [pts_array], False,
                              tuple(int(c * trail_alpha) for c in colour),
                              2, cv2.LINE_AA)

            # Active segment (bright)
            recent = max(0, draw_count - 10)
            if draw_count - recent > 1:
                pts_array = np.array(spts[recent:draw_count], dtype=np.int32)
                cv2.polylines(canvas, [pts_array], False, colour, 3, cv2.LINE_AA)

            # Head dot
            hx, hy = spts[draw_count - 1]
            cv2.circle(canvas, (hx, hy), 5, colour, -1, cv2.LINE_AA)
            #cv2.circle(canvas, (hx, hy), 7, (255, 255, 255), 1, cv2.LINE_AA)
            traj_id = t["id"]
            cv2.putText(
                canvas,
                traj_id,
                (hx + 8, hy - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                colour,
                1,
                cv2.LINE_AA
            )

        # Blend trail overlay
        canvas = cv2.addWeighted(canvas, 1.0, overlay, 0.6, 0)

        # -- Draw links (dashed lines) when both endpoints are reached --
        for src_id, (tgt_id, p_src, p_tgt, t_end_src, t_start_tgt) in link_map.items():
            t_end_rel   = (t_end_src   - t_start_global).total_seconds()
            t_start_rel = (t_start_tgt - t_start_global).total_seconds()
            if sim_time < t_end_rel:
                continue
            ci = traj_chain_idx[src_id]
            colour = colours[ci]
            ps = world_to_screen(*p_src)
            pt = world_to_screen(*p_tgt)
            # Fade-in the dashed link
            elapsed = sim_time - t_end_rel
            alpha = min(1.0, elapsed / max(t_start_rel - t_end_rel, 0.5))
            faded = tuple(int(c * (0.3 + 0.7 * alpha)) for c in colour)
            draw_dashed_line(canvas, ps, pt, faded, thickness=2)

        # -- Timeline bar --
        bar_y     = height + TIMELINE_H // 2
        bar_x0    = margin
        bar_x1    = width - margin
        bar_len   = bar_x1 - bar_x0
        progress  = sim_time / max(total_duration, 1)
        cur_x     = int(bar_x0 + progress * bar_len)

        cv2.line(canvas, (bar_x0, bar_y), (bar_x1, bar_y), (80, 80, 80), 3)
        cv2.line(canvas, (bar_x0, bar_y), (cur_x,  bar_y), (200, 200, 200), 3)
        cv2.circle(canvas, (cur_x, bar_y), 6, (255, 255, 255), -1)

        # Timestamp
        elapsed_s = int(sim_time)
        ts_str = f"{elapsed_s // 60:02d}:{elapsed_s % 60:02d}"
        cv2.putText(canvas, ts_str, (cur_x - 18, bar_y - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

        # -- Legend (top-left) --
        multi_chains = [c for c in chains if len(c) > 1]
        for li, chain in enumerate(multi_chains[:12]):
            ci = chains.index(chain)
            col = colours[ci]
            lx, ly = 12, 20 + li * 18
            cv2.rectangle(canvas, (lx, ly - 10), (lx + 14, ly + 2), col, -1)
            label = " -> ".join(chain)
            if len(label) > 50:
                label = label[:47] + "…"
            cv2.putText(canvas, label, (lx + 18, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (210, 210, 210), 1, cv2.LINE_AA)

        # Title
        cv2.putText(canvas, "Trajectory Linking", (width - 220, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)

        writer.write(canvas)

        if frame_idx % 50 == 0:
            pct = 100 * frame_idx / video_frames
            print(f"  {pct:5.1f}%  (frame {frame_idx}/{video_frames})")

    writer.release()
    print(f"\n Video saved -> {output_path}")


# -- CLI --
def main():
    parser = argparse.ArgumentParser(description="Visualise linked trajectories")
    parser.add_argument("--input",    required=True)
    parser.add_argument("--output",   default="trajectories_linked.mp4")
    parser.add_argument("--max_gap",  type=float, default=10.0)
    parser.add_argument("--max_dist", type=float, default=50.0)
    parser.add_argument("--fps",      type=int,   default=30)
    parser.add_argument("--width",    type=int,   default=1280)
    parser.add_argument("--height",   type=int,   default=720)
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)
    trajectories = data["trajectories"]

    links  = build_links(trajectories, max_gap=args.max_gap, max_dist=args.max_dist)
    chains = build_chains(trajectories, links)

    render_video(
        trajectories, links, chains,
        output_path=args.output,
        fps=args.fps,
        width=args.width,
        height=args.height,
    )


if __name__ == "__main__":
    main()
