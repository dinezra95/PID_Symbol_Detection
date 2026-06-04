import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from .data_structures import Point, LineSegment
import logging

logger = logging.getLogger(__name__)


class LineDetector:
    """Detect and refine pipe line segments from a binary mask."""

    def __init__(self, config: Dict):
        self.config = config
        self.hough_rho = config.get("hough_rho", 1)
        self.hough_theta = np.pi / 180 * config.get("hough_theta_deg", 1)
        self.hough_threshold = config.get("hough_threshold", 40)
        self.hough_min_line_length = config.get("hough_min_line_length", 25)
        self.hough_max_line_gap = config.get("hough_max_line_gap", 15)
        self.angle_tolerance = config.get("angle_tolerance", 8)
        self.merge_angle_thresh = config.get("merge_angle_thresh", 5)
        self.merge_dist_thresh = config.get("merge_dist_thresh", 8)
        self.merge_gap_thresh = config.get("merge_gap_thresh", 30)
        self.bridge_max_gap = config.get("bridge_max_gap", 60)
        self.min_segment_length = config.get("min_segment_length", 10)

    def detect_lines(self, pipe_mask: np.ndarray) -> List[LineSegment]:
        """Run Probabilistic Hough Transform on pipe mask."""
        lines_raw = cv2.HoughLinesP(
            pipe_mask,
            rho=self.hough_rho,
            theta=self.hough_theta,
            threshold=self.hough_threshold,
            minLineLength=self.hough_min_line_length,
            maxLineGap=self.hough_max_line_gap,
        )

        if lines_raw is None:
            logger.warning("No lines detected by Hough Transform")
            return []

        segments = []
        for line in lines_raw:
            x1, y1, x2, y2 = line[0]
            seg = LineSegment(Point(x1, y1), Point(x2, y2))
            if seg.length >= self.min_segment_length:
                segments.append(seg)

        logger.info(f"Hough Transform detected {len(segments)} line segments")
        return segments

    def filter_near_orthogonal(self, segments: List[LineSegment]) -> List[LineSegment]:
        """Keep only segments that are near horizontal or vertical."""
        filtered = []
        for seg in segments:
            a = seg.angle
            is_h = a < self.angle_tolerance or a > (180 - self.angle_tolerance)
            is_v = abs(a - 90) < self.angle_tolerance
            if is_h or is_v:
                filtered.append(seg)

        logger.info(
            f"Orthogonal filter: {len(filtered)}/{len(segments)} segments kept "
            f"(tolerance={self.angle_tolerance}°)"
        )
        return filtered

    def snap_to_orthogonal(self, segments: List[LineSegment]) -> List[LineSegment]:
        """Snap near-horizontal lines to exact horizontal, near-vertical to exact vertical."""
        snapped = []
        for seg in segments:
            a = seg.angle
            is_h = a < self.angle_tolerance or a > (180 - self.angle_tolerance)
            if is_h:
                mid_y = (seg.start.y + seg.end.y) / 2
                snapped.append(LineSegment(
                    Point(seg.start.x, mid_y),
                    Point(seg.end.x, mid_y),
                ))
            else:
                mid_x = (seg.start.x + seg.end.x) / 2
                snapped.append(LineSegment(
                    Point(mid_x, seg.start.y),
                    Point(mid_x, seg.end.y),
                ))
        return snapped

    def merge_collinear_segments(self, segments: List[LineSegment]) -> List[LineSegment]:
        """Merge collinear, nearby segments into longer ones.

        Two segments are merged if:
        1. Their angles are within merge_angle_thresh
        2. Their perpendicular distance is within merge_dist_thresh
        3. Their endpoint gap is within merge_gap_thresh
        """
        if not segments:
            return []

        used = [False] * len(segments)
        merged = []

        h_segs = [(i, s) for i, s in enumerate(segments) if s.is_horizontal]
        v_segs = [(i, s) for i, s in enumerate(segments) if s.is_vertical]

        for group in [h_segs, v_segs]:
            group_indices = [i for i, _ in group]
            clusters = self._cluster_collinear(
                [s for _, s in group], group_indices, used
            )
            for cluster in clusters:
                merged_seg = self._merge_cluster(cluster)
                if merged_seg:
                    merged.append(merged_seg)

        for i, seg in enumerate(segments):
            if not used[i]:
                merged.append(seg)

        logger.info(f"Merged {len(segments)} segments into {len(merged)}")
        return merged

    def _cluster_collinear(
        self,
        segs: List[LineSegment],
        indices: List[int],
        used: List[bool],
    ) -> List[List[LineSegment]]:
        """Group collinear segments into clusters using union-find."""
        n = len(segs)
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i in range(n):
            for j in range(i + 1, n):
                if self._should_merge(segs[i], segs[j]):
                    union(i, j)

        groups: Dict[int, List[int]] = {}
        for i in range(n):
            root = find(i)
            groups.setdefault(root, []).append(i)

        clusters = []
        for members in groups.values():
            cluster = []
            for m in members:
                used[indices[m]] = True
                cluster.append(segs[m])
            clusters.append(cluster)

        return clusters

    def _should_merge(self, a: LineSegment, b: LineSegment) -> bool:
        angle_diff = abs(a.angle - b.angle)
        if angle_diff > 90:
            angle_diff = 180 - angle_diff
        if angle_diff > self.merge_angle_thresh:
            return False

        perp_dist = min(
            a.perpendicular_distance(b.midpoint),
            b.perpendicular_distance(a.midpoint),
        )
        if perp_dist > self.merge_dist_thresh:
            return False

        gap = a.endpoint_gap(b)
        if gap > self.merge_gap_thresh:
            return False

        return True

    def _merge_cluster(self, cluster: List[LineSegment]) -> Optional[LineSegment]:
        """Merge a cluster of collinear segments into one spanning segment."""
        if not cluster:
            return None

        all_points = []
        for seg in cluster:
            all_points.append(seg.start)
            all_points.append(seg.end)

        if cluster[0].is_horizontal:
            all_points.sort(key=lambda p: p.x)
            avg_y = np.mean([p.y for p in all_points])
            return LineSegment(
                Point(all_points[0].x, avg_y),
                Point(all_points[-1].x, avg_y),
            )
        else:
            all_points.sort(key=lambda p: p.y)
            avg_x = np.mean([p.x for p in all_points])
            return LineSegment(
                Point(avg_x, all_points[0].y),
                Point(avg_x, all_points[-1].y),
            )

    def bridge_symbol_gaps(
        self,
        segments: List[LineSegment],
        symbol_bboxes: List[Tuple[int, int, int, int]],
    ) -> List[LineSegment]:
        """Bridge gaps in pipes caused by masked-out symbols.

        For each symbol bbox, find pairs of line endpoints on opposite sides
        that are collinear and connect them through the symbol.
        """
        if not symbol_bboxes or not segments:
            return segments

        bridged = list(segments)
        new_bridges = []

        for sx1, sy1, sx2, sy2 in symbol_bboxes:
            cx = (sx1 + sx2) / 2
            cy = (sy1 + sy2) / 2
            margin = self.bridge_max_gap

            nearby_h = []
            nearby_v = []
            for seg in bridged:
                for pt in [seg.start, seg.end]:
                    dist_to_center = pt.distance_to(Point(cx, cy))
                    box_diag = np.hypot(sx2 - sx1, sy2 - sy1)
                    if dist_to_center > box_diag + margin:
                        continue
                    if seg.is_horizontal:
                        nearby_h.append((pt, seg))
                    elif seg.is_vertical:
                        nearby_v.append((pt, seg))

            new_bridges.extend(
                self._try_bridge_pairs(nearby_h, sx1, sy1, sx2, sy2, horizontal=True)
            )
            new_bridges.extend(
                self._try_bridge_pairs(nearby_v, sx1, sy1, sx2, sy2, horizontal=False)
            )

        if new_bridges:
            bridged.extend(new_bridges)
            logger.info(f"Bridged {len(new_bridges)} gaps across symbols")

        return bridged

    def _try_bridge_pairs(
        self,
        candidates: List[Tuple[Point, LineSegment]],
        sx1: int, sy1: int, sx2: int, sy2: int,
        horizontal: bool,
    ) -> List[LineSegment]:
        """Try to bridge pairs of endpoints across a symbol bbox."""
        bridges = []
        if horizontal:
            left = [(pt, seg) for pt, seg in candidates if pt.x <= sx1]
            right = [(pt, seg) for pt, seg in candidates if pt.x >= sx2]
            for lpt, lseg in left:
                for rpt, rseg in right:
                    if lseg is rseg:
                        continue
                    y_diff = abs(lpt.y - rpt.y)
                    if y_diff < self.merge_dist_thresh:
                        avg_y = (lpt.y + rpt.y) / 2
                        bridges.append(LineSegment(
                            Point(lpt.x, avg_y), Point(rpt.x, avg_y)
                        ))
        else:
            top = [(pt, seg) for pt, seg in candidates if pt.y <= sy1]
            bottom = [(pt, seg) for pt, seg in candidates if pt.y >= sy2]
            for tpt, tseg in top:
                for bpt, bseg in bottom:
                    if tseg is bseg:
                        continue
                    x_diff = abs(tpt.x - bpt.x)
                    if x_diff < self.merge_dist_thresh:
                        avg_x = (tpt.x + bpt.x) / 2
                        bridges.append(LineSegment(
                            Point(avg_x, tpt.y), Point(avg_x, bpt.y)
                        ))
        return bridges

    def detect_and_refine(
        self,
        pipe_mask: np.ndarray,
    ) -> List[LineSegment]:
        """Full line detection pipeline: detect → filter → snap → merge."""
        raw_lines = self.detect_lines(pipe_mask)
        if not raw_lines:
            return []

        orthogonal = self.filter_near_orthogonal(raw_lines)
        snapped = self.snap_to_orthogonal(orthogonal)
        merged = self.merge_collinear_segments(snapped)

        logger.info(f"Final line count after all refinement: {len(merged)}")
        return merged
