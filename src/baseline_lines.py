"""Baseline: LSD + filters + graph construction."""
import cv2
import numpy as np
import os
import time
from pathlib import Path

from utils.pipe_detection.data_structures import Point, LineSegment
from utils.pipe_detection.line_detector import LineDetector
from utils.pipe_detection.graph_builder import GraphBuilder
from utils.pipe_detection.visualizer import PipeVisualizer


def detect_text_regions(gray, padding=10):
    """Use pytesseract to get text bounding boxes."""
    try:
        import pytesseract
    except ImportError:
        return []

    old_tmpdir = os.environ.get("TMPDIR")
    try:
        fallback_tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".tmp")
        os.makedirs(fallback_tmp, exist_ok=True)
        os.environ["TMPDIR"] = fallback_tmp
        data = pytesseract.image_to_data(gray, config="--oem 3 --psm 6", output_type=pytesseract.Output.DICT)
    except Exception:
        return []
    finally:
        if old_tmpdir is not None:
            os.environ["TMPDIR"] = old_tmpdir
        elif "TMPDIR" in os.environ:
            del os.environ["TMPDIR"]

    text_bboxes = []
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        conf = int(data["conf"][i])
        if not text or conf < 40 or len(text) < 2:
            continue
        x, y, bw, bh = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        text_bboxes.append((x - padding, y - padding, x + bw + padding, y + bh + padding))

    return text_bboxes


def line_inside_text(lx1, ly1, lx2, ly2, text_bboxes):
    """Check if both endpoints are inside any text bbox."""
    for tx1, ty1, tx2, ty2 in text_bboxes:
        p1_in = tx1 <= lx1 <= tx2 and ty1 <= ly1 <= ty2
        p2_in = tx1 <= lx2 <= tx2 and ty1 <= ly2 <= ty2
        if p1_in and p2_in:
            return True
    return False


def load_yolo_bboxes(label_path, img_w, img_h):
    bboxes = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, cx, cy, w, h = (float(p) for p in parts[:5])
            x1 = int((cx - w / 2) * img_w)
            y1 = int((cy - h / 2) * img_h)
            x2 = int((cx + w / 2) * img_w)
            y2 = int((cy + h / 2) * img_h)
            bboxes.append((x1, y1, x2, y2))
    return bboxes


def is_frame_line(x1, y1, x2, y2, length, img_w, img_h, edge_pct=0.01, max_length_pct=0.5):
    edge_x = img_w * edge_pct
    edge_y = img_h * edge_pct
    near_left   = x1 < edge_x and x2 < edge_x
    near_right  = x1 > img_w - edge_x and x2 > img_w - edge_x
    near_top    = y1 < edge_y and y2 < edge_y
    near_bottom = y1 > img_h - edge_y and y2 > img_h - edge_y
    if near_left or near_right or near_top or near_bottom:
        return True
    if length > img_w * max_length_pct or length > img_h * max_length_pct:
        return True
    return False


def find_rectangles(lines, corner_tolerance=30):
    """Find groups of 4 lines forming closed rectangles.
    Returns set of line indices that are part of a rectangle.
    """
    h_lines = [(i, l) for i, l in enumerate(lines)
               if abs(l[1] - l[3]) < corner_tolerance]  # horizontal: y1 ≈ y2
    v_lines = [(i, l) for i, l in enumerate(lines)
               if abs(l[0] - l[2]) < corner_tolerance]  # vertical: x1 ≈ x2

    rect_indices = set()

    for hi1, (i1, h1) in enumerate(h_lines):
        h1_xmin, h1_xmax = min(h1[0], h1[2]), max(h1[0], h1[2])
        h1_y = (h1[1] + h1[3]) / 2

        for hi2 in range(hi1 + 1, len(h_lines)):
            i2, h2 = h_lines[hi2]
            h2_xmin, h2_xmax = min(h2[0], h2[2]), max(h2[0], h2[2])
            h2_y = (h2[1] + h2[3]) / 2

            # Similar x-range?
            if abs(h1_xmin - h2_xmin) > corner_tolerance:
                continue
            if abs(h1_xmax - h2_xmax) > corner_tolerance:
                continue
            # Different y
            if abs(h1_y - h2_y) < corner_tolerance:
                continue

            # Look for left vertical
            left_v = None
            right_v = None
            for vi, (iv, v) in enumerate(v_lines):
                v_x = (v[0] + v[2]) / 2
                v_ymin, v_ymax = min(v[1], v[3]), max(v[1], v[3])
                y_min, y_max = min(h1_y, h2_y), max(h1_y, h2_y)

                if abs(v_ymin - y_min) > corner_tolerance:
                    continue
                if abs(v_ymax - y_max) > corner_tolerance:
                    continue

                if abs(v_x - h1_xmin) < corner_tolerance:
                    left_v = iv
                elif abs(v_x - h1_xmax) < corner_tolerance:
                    right_v = iv

            if left_v is not None and right_v is not None:
                rect_indices.update([i1, i2, left_v, right_v])

    return rect_indices


def bridge_collinear_gaps(segments, min_pipe_length=150, max_gap=300, perp_tolerance=20):
    """Bridge gaps between long collinear pipe segments.
    Two segments are bridged if:
    - Both are longer than min_pipe_length
    - Same orientation (both H or both V)
    - Perpendicular distance < perp_tolerance
    - Endpoint gap < max_gap
    """
    if len(segments) < 2:
        return segments

    used = [False] * len(segments)
    result = []

    h_segs = [(i, s) for i, s in enumerate(segments) if s.is_horizontal and s.length >= min_pipe_length]
    v_segs = [(i, s) for i, s in enumerate(segments) if s.is_vertical and s.length >= min_pipe_length]

    for group in [h_segs, v_segs]:
        for a_idx in range(len(group)):
            i, seg_a = group[a_idx]
            if used[i]:
                continue
            for b_idx in range(a_idx + 1, len(group)):
                j, seg_b = group[b_idx]
                if used[j]:
                    continue

                perp_dist = min(seg_a.perpendicular_distance(seg_b.midpoint),
                                seg_b.perpendicular_distance(seg_a.midpoint))
                if perp_dist > perp_tolerance:
                    continue

                gap = seg_a.endpoint_gap(seg_b)
                if gap > max_gap:
                    continue

                # Merge into one spanning segment
                all_pts = [seg_a.start, seg_a.end, seg_b.start, seg_b.end]
                if seg_a.is_horizontal:
                    all_pts.sort(key=lambda p: p.x)
                    avg_y = np.mean([p.y for p in all_pts])
                    merged = LineSegment(Point(all_pts[0].x, avg_y), Point(all_pts[-1].x, avg_y))
                else:
                    all_pts.sort(key=lambda p: p.y)
                    avg_x = np.mean([p.x for p in all_pts])
                    merged = LineSegment(Point(avg_x, all_pts[0].y), Point(avg_x, all_pts[-1].y))

                used[i] = True
                used[j] = True
                result.append(merged)
                break

    for i, seg in enumerate(segments):
        if not used[i]:
            result.append(seg)

    return result


def endpoint_near_symbol(x, y, bbox, padding=30):
    sx1, sy1, sx2, sy2 = bbox
    return (sx1 - padding) <= x <= (sx2 + padding) and (sy1 - padding) <= y <= (sy2 + padding)


def find_symbol_connections(lines, bboxes, padding=30, coord_tolerance=25):
    """For each symbol, find lines with endpoints on opposite sides.
    Returns set of line indices that connect through at least one symbol,
    and list of (symbol_idx, line_idx_a, line_idx_b) connections.
    """
    connected_lines = set()
    connections = []

    for si, (sx1, sy1, sx2, sy2) in enumerate(bboxes):
        left, right, top, bottom = [], [], [], []

        for li, (lx1, ly1, lx2, ly2, _) in enumerate(lines):
            for px, py in [(lx1, ly1), (lx2, ly2)]:
                if not ((sx1 - padding) <= px <= (sx2 + padding) and
                        (sy1 - padding) <= py <= (sy2 + padding)):
                    continue
                if px < sx1 and (sx1 - px) <= padding:
                    left.append(li)
                elif px > sx2 and (px - sx2) <= padding:
                    right.append(li)
                if py < sy1 and (sy1 - py) <= padding:
                    top.append(li)
                elif py > sy2 and (py - sy2) <= padding:
                    bottom.append(li)

        # Horizontal connections
        for li_l in left:
            for li_r in right:
                if li_l == li_r:
                    continue
                ly_l = (lines[li_l][1] + lines[li_l][3]) / 2
                ly_r = (lines[li_r][1] + lines[li_r][3]) / 2
                if abs(ly_l - ly_r) <= coord_tolerance:
                    connected_lines.add(li_l)
                    connected_lines.add(li_r)
                    connections.append((si, li_l, li_r))

        # Vertical connections
        for li_t in top:
            for li_b in bottom:
                if li_t == li_b:
                    continue
                lx_t = (lines[li_t][0] + lines[li_t][2]) / 2
                lx_b = (lines[li_b][0] + lines[li_b][2]) / 2
                if abs(lx_t - lx_b) <= coord_tolerance:
                    connected_lines.add(li_t)
                    connected_lines.add(li_b)
                    connections.append((si, li_t, li_b))

    return connected_lines, connections


def expand_connected(connected_indices, lines, snap_dist=40):
    """Expand: also keep lines whose endpoints touch a connected line's endpoints."""
    expanded = set(connected_indices)
    changed = True
    while changed:
        changed = False
        for li, (lx1, ly1, lx2, ly2, _) in enumerate(lines):
            if li in expanded:
                continue
            for ci in list(expanded):
                cx1, cy1, cx2, cy2, _ = lines[ci]
                for px, py in [(lx1, ly1), (lx2, ly2)]:
                    for qx, qy in [(cx1, cy1), (cx2, cy2)]:
                        if np.hypot(px - qx, py - qy) < snap_dist:
                            expanded.add(li)
                            changed = True
                            break
                    if li in expanded:
                        break
                if li in expanded:
                    break
    return expanded


def run_baseline(img_path, label_path, output_dir, min_length=80, max_width=5):
    img = cv2.imread(img_path)
    if img is None:
        return

    name = Path(img_path).stem
    out = Path(output_dir) / name
    out.mkdir(parents=True, exist_ok=True)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    thick = max(3, int(w / 1500))

    bboxes = load_yolo_bboxes(label_path, w, h) if label_path else []

    print(f"\n{'='*60}")
    print(f"Image: {name} ({w}x{h}), {len(bboxes)} symbols")
    print(f"{'='*60}")

    # --- LSD ---
    t0 = time.time()
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines_raw, widths, _, _ = lsd.detect(gray)
    dt_lsd = time.time() - t0

    if lines_raw is None:
        print("No lines detected")
        return

    # --- Filter: H/V + min length + max width + frame + inside symbol ---
    hv_lines = []
    frame_lines = []
    inside_symbol = []
    for i, line in enumerate(lines_raw):
        lx1, ly1, lx2, ly2 = line[0]
        length = np.hypot(lx2 - lx1, ly2 - ly1)
        lw = widths[i][0] if widths is not None else 1
        angle = abs(np.degrees(np.arctan2(ly2 - ly1, lx2 - lx1))) % 180
        is_hv = angle < 10 or angle > 170 or 80 < angle < 100
        if length < min_length or lw > max_width or not is_hv:
            continue
        if is_frame_line(lx1, ly1, lx2, ly2, length, w, h):
            frame_lines.append((lx1, ly1, lx2, ly2, length))
            continue
        # Check if both endpoints are inside any symbol bbox
        both_inside = False
        for sx1, sy1, sx2, sy2 in bboxes:
            p1_in = sx1 <= lx1 <= sx2 and sy1 <= ly1 <= sy2
            p2_in = sx1 <= lx2 <= sx2 and sy1 <= ly2 <= sy2
            if p1_in and p2_in:
                both_inside = True
                break
        if both_inside:
            inside_symbol.append((lx1, ly1, lx2, ly2, length))
        else:
            hv_lines.append((lx1, ly1, lx2, ly2, length))

    # --- Filter: rectangles (equipment outlines) ---
    rect_indices = find_rectangles(hv_lines, corner_tolerance=40)
    rect_lines = [hv_lines[i] for i in rect_indices]
    hv_lines = [l for i, l in enumerate(hv_lines) if i not in rect_indices]

    print(f"LSD: {len(lines_raw)} raw → {len(hv_lines)} H/V (removed {len(frame_lines)} frame, {len(inside_symbol)} in symbols, {len(rect_lines)} rectangles) [{dt_lsd:.2f}s]")

    # --- Convert to LineSegment objects ---
    segments = []
    for lx1, ly1, lx2, ly2, _ in hv_lines:
        segments.append(LineSegment(Point(lx1, ly1), Point(lx2, ly2)))

    # --- Bridge across symbols (connect pipes through symbols directly) ---
    t = time.time()
    line_cfg = {
        "merge_angle_thresh": 5,
        "merge_dist_thresh": 20,
        "merge_gap_thresh": 60,
        "min_segment_length": min_length,
        "bridge_max_gap": 80,
    }
    detector = LineDetector(line_cfg)
    bridged = detector.bridge_symbol_gaps(segments, bboxes)
    bridges_added = len(bridged) - len(segments)
    dt_bridge = time.time() - t
    print(f"Bridged {bridges_added} gaps across symbols [{dt_bridge:.2f}s]")

    # --- Snap + merge ---
    t = time.time()
    snapped = detector.snap_to_orthogonal(bridged)
    merged = detector.merge_collinear_segments(snapped)

    # --- Bridge collinear gaps (long pipes with gaps between them) ---
    before_bridge = len(merged)
    merged = bridge_collinear_gaps(merged, min_pipe_length=150, max_gap=300, perp_tolerance=20)
    dt_merge = time.time() - t
    print(f"Merged {len(bridged)} → {before_bridge} → {len(merged)} segments (bridged {before_bridge - len(merged)} collinear gaps) [{dt_merge:.2f}s]")

    # --- Build graph ---
    t = time.time()
    graph_cfg = {
        "junction_snap_dist": 40,
        "min_pipe_length": min_length,
    }
    builder = GraphBuilder(graph_cfg)
    graph = builder.build_graph(merged)
    graph = builder.simplify_graph(graph)
    dt_graph = time.time() - t
    print(f"Graph: {len(graph.junctions)} junctions, {len(graph.pipe_segments)} pipes [{dt_graph:.2f}s]")

    # --- Visualize + save ---
    visualizer = PipeVisualizer(str(out))
    visualizer.draw_pipe_graph(img, graph, "01_pipe_graph")

    graph.save_json(str(out / "pipe_graph.json"))

    total = time.time() - t0
    print(f"{graph.summary()}")
    print(f"Total: {total:.2f}s → {out}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", default="../results/stage_3/baseline_lsd")
    parser.add_argument("--min_length", type=int, default=80)
    parser.add_argument("--max_width", type=float, default=5)
    args = parser.parse_args()

    from utils.helpers import get_files
    input_path = Path(args.input)
    labels_path = Path(args.labels)
    files = get_files(input_path, [".jpg", ".jpeg", ".png", ".tif", ".tiff"]) if input_path.is_dir() else [input_path]

    for f in files:
        lbl = labels_path / f"{f.stem}.txt"
        if not lbl.exists():
            lbl = labels_path / "labels" / f"{f.stem}.txt"
        run_baseline(str(f), str(lbl) if lbl.exists() else None, args.output, args.min_length, args.max_width)
