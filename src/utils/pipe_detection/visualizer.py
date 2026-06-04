import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from .data_structures import Point, LineSegment, PipeSegment, Junction, PipeGraph
import logging

logger = logging.getLogger(__name__)

COLORS = {
    "pipe": (255, 100, 0),      # orange
    "junction_T": (0, 255, 0),  # green
    "junction_L": (0, 200, 200),# cyan
    "junction_cross": (255, 0, 255),  # magenta
    "junction_endpoint": (0, 0, 255), # red
    "symbol_mask": (200, 200, 200),   # light gray
    "label": (255, 255, 0),     # yellow
    "bridge": (0, 255, 255),    # cyan
}


class PipeVisualizer:
    """Visualize pipe detection results at each pipeline stage."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_mask(self, mask: np.ndarray, name: str):
        path = self.output_dir / f"{name}.png"
        cv2.imwrite(str(path), mask)
        logger.debug(f"Saved mask: {path}")

    def draw_lines_on_image(
        self,
        image: np.ndarray,
        segments: List[LineSegment],
        name: str,
        color: Tuple[int, int, int] = COLORS["pipe"],
        thickness: int = 2,
    ) -> np.ndarray:
        vis = image.copy()
        for seg in segments:
            pt1 = seg.start.as_tuple()
            pt2 = seg.end.as_tuple()
            cv2.line(vis, pt1, pt2, color, thickness)
        path = self.output_dir / f"{name}.png"
        cv2.imwrite(str(path), vis)
        return vis

    def draw_pipe_graph(
        self,
        image: np.ndarray,
        graph: PipeGraph,
        name: str = "pipe_graph",
    ) -> np.ndarray:
        vis = image.copy()

        for pid, pipe in graph.pipe_segments.items():
            for seg in pipe.segments:
                cv2.line(vis, seg.start.as_tuple(), seg.end.as_tuple(), COLORS["pipe"], 2)

        # Draw symbol-connection edges as dashed-style lines
        for u, v, data in graph.graph.edges(data=True):
            if data.get("type") == "symbol_connection":
                pos_u = graph.graph.nodes[u].get("pos")
                pos_v = graph.graph.nodes[v].get("pos")
                if pos_u and pos_v:
                    cv2.line(vis, pos_u, pos_v, COLORS["bridge"], 2)

        # Draw symbol nodes
        for sid, sym in graph.symbol_nodes.items():
            x1, y1, x2, y2 = sym.bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 220, 0), 2)
            cx, cy = sym.center.as_tuple()
            cv2.putText(
                vis, sid,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 0), 1,
            )

        for jid, junction in graph.junctions.items():
            color_key = f"junction_{junction.junction_type}"
            color = COLORS.get(color_key, (128, 128, 128))
            pos = junction.position.as_tuple()
            radius = 6 if junction.junction_type == "endpoint" else 8
            cv2.circle(vis, pos, radius, color, -1)
            cv2.putText(
                vis, f"{junction.junction_type}",
                (pos[0] + 10, pos[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1,
            )

        for tr in graph.text_regions:
            x, y, w, h = tr.bbox
            cv2.rectangle(vis, (x, y), (x + w, y + h), COLORS["label"], 1)
            cv2.putText(
                vis, tr.text,
                (x, y - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLORS["label"], 1,
            )

        path = self.output_dir / f"{name}.png"
        cv2.imwrite(str(path), vis)
        logger.info(f"Saved graph visualization: {path}")
        return vis

    def draw_debug_composite(
        self,
        image: np.ndarray,
        pipe_mask: np.ndarray,
        symbol_mask: np.ndarray,
        segments: List[LineSegment],
        graph: PipeGraph,
        name: str = "debug_composite",
    ) -> np.ndarray:
        """Create a 2x2 debug composite image."""
        h, w = image.shape[:2]
        scale = 0.5
        sh, sw = int(h * scale), int(w * scale)

        panel1 = cv2.resize(image, (sw, sh))

        mask_vis = cv2.cvtColor(pipe_mask, cv2.COLOR_GRAY2BGR)
        sym_overlay = np.zeros_like(mask_vis)
        sym_overlay[symbol_mask > 0] = [0, 0, 200]
        mask_vis = cv2.addWeighted(mask_vis, 1.0, sym_overlay, 0.4, 0)
        panel2 = cv2.resize(mask_vis, (sw, sh))

        lines_vis = image.copy()
        for seg in segments:
            cv2.line(lines_vis, seg.start.as_tuple(), seg.end.as_tuple(), COLORS["pipe"], 2)
        panel3 = cv2.resize(lines_vis, (sw, sh))

        graph_vis = self.draw_pipe_graph(image.copy(), graph, name="__temp_graph")
        panel4 = cv2.resize(graph_vis, (sw, sh))

        top = np.hstack([panel1, panel2])
        bottom = np.hstack([panel3, panel4])
        composite = np.vstack([top, bottom])

        labels = ["Original", "Pipe Mask + Symbols", "Detected Lines", "Pipe Graph"]
        for i, label in enumerate(labels):
            x = (i % 2) * sw + 10
            y = (i // 2) * sh + 25
            cv2.putText(composite, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)

        path = self.output_dir / f"{name}.png"
        cv2.imwrite(str(path), composite)
        logger.info(f"Saved debug composite: {path}")
        return composite
