from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import numpy as np
import networkx as nx
import json


@dataclass
class Point:
    x: float
    y: float

    def distance_to(self, other: "Point") -> float:
        return np.hypot(self.x - other.x, self.y - other.y)

    def as_tuple(self) -> Tuple[int, int]:
        return (int(round(self.x)), int(round(self.y)))

    def __hash__(self):
        return hash((round(self.x, 1), round(self.y, 1)))


@dataclass
class LineSegment:
    start: Point
    end: Point

    @property
    def length(self) -> float:
        return self.start.distance_to(self.end)

    @property
    def angle(self) -> float:
        dx = self.end.x - self.start.x
        dy = self.end.y - self.start.y
        return np.degrees(np.arctan2(dy, dx)) % 180

    @property
    def midpoint(self) -> Point:
        return Point((self.start.x + self.end.x) / 2, (self.start.y + self.end.y) / 2)

    @property
    def is_horizontal(self) -> bool:
        a = self.angle
        return a < 10 or a > 170

    @property
    def is_vertical(self) -> bool:
        a = self.angle
        return 80 < a < 100

    def perpendicular_distance(self, point: Point) -> float:
        x0, y0 = point.x, point.y
        x1, y1 = self.start.x, self.start.y
        x2, y2 = self.end.x, self.end.y
        num = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        den = np.hypot(y2 - y1, x2 - x1)
        if den == 0:
            return point.distance_to(self.start)
        return num / den

    def endpoint_gap(self, other: "LineSegment") -> float:
        gaps = [
            self.end.distance_to(other.start),
            self.end.distance_to(other.end),
            self.start.distance_to(other.start),
            self.start.distance_to(other.end),
        ]
        return min(gaps)


@dataclass
class PipeSegment:
    id: int
    segments: List[LineSegment]
    label: Optional[str] = None
    diameter: Optional[str] = None
    pipe_type: Optional[str] = None

    @property
    def total_length(self) -> float:
        return sum(s.length for s in self.segments)

    @property
    def endpoints(self) -> Tuple[Point, Point]:
        all_points = []
        for seg in self.segments:
            all_points.extend([seg.start, seg.end])
        if len(all_points) < 2:
            return (all_points[0], all_points[0])
        max_dist = 0
        p1, p2 = all_points[0], all_points[-1]
        for i, a in enumerate(all_points):
            for b in all_points[i + 1:]:
                d = a.distance_to(b)
                if d > max_dist:
                    max_dist = d
                    p1, p2 = a, b
        return (p1, p2)


@dataclass
class Junction:
    id: int
    position: Point
    junction_type: str  # "T", "L", "cross", "endpoint", "unknown"
    connected_pipe_ids: List[int] = field(default_factory=list)

    @property
    def degree(self) -> int:
        return len(self.connected_pipe_ids)


@dataclass
class SymbolNode:
    id: str
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    symbol_class: str = "symbol"

    @property
    def center(self) -> Point:
        x1, y1, x2, y2 = self.bbox
        return Point((x1 + x2) / 2, (y1 + y2) / 2)


@dataclass
class TextRegion:
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    text: str
    confidence: float = 0.0

    @property
    def center(self) -> Point:
        x, y, w, h = self.bbox
        return Point(x + w / 2, y + h / 2)


class PipeGraph:
    def __init__(self):
        self.graph = nx.Graph()
        self.junctions: Dict[int, Junction] = {}
        self.pipe_segments: Dict[int, PipeSegment] = {}
        self.symbol_nodes: Dict[str, SymbolNode] = {}
        self.text_regions: List[TextRegion] = []

    def add_junction(self, junction: Junction):
        self.junctions[junction.id] = junction
        self.graph.add_node(
            junction.id,
            pos=junction.position.as_tuple(),
            type=junction.junction_type,
        )

    def add_symbol_node(self, symbol: SymbolNode):
        self.symbol_nodes[symbol.id] = symbol
        self.graph.add_node(
            symbol.id,
            pos=symbol.center.as_tuple(),
            type="symbol",
            symbol_class=symbol.symbol_class,
            bbox=symbol.bbox,
        )

    def add_pipe(self, pipe: PipeSegment, from_junction_id: int, to_junction_id: int):
        self.pipe_segments[pipe.id] = pipe
        self.graph.add_edge(
            from_junction_id,
            to_junction_id,
            pipe_id=pipe.id,
            length=pipe.total_length,
            label=pipe.label,
            diameter=pipe.diameter,
            pipe_type=pipe.pipe_type,
        )

    def get_connected_pipes(self, junction_id: int) -> List[PipeSegment]:
        pipe_ids = []
        for _, _, data in self.graph.edges(junction_id, data=True):
            pipe_ids.append(data["pipe_id"])
        return [self.pipe_segments[pid] for pid in pipe_ids if pid in self.pipe_segments]

    def to_dict(self) -> dict:
        nodes = []
        for jid, j in self.junctions.items():
            nodes.append({
                "id": jid,
                "x": j.position.x,
                "y": j.position.y,
                "type": j.junction_type,
                "degree": j.degree,
                "connected_pipes": j.connected_pipe_ids,
            })
        for sid, s in self.symbol_nodes.items():
            nodes.append({
                "id": sid,
                "x": s.center.x,
                "y": s.center.y,
                "type": "symbol",
                "symbol_class": s.symbol_class,
                "bbox": list(s.bbox),
            })
        edges = []
        for u, v, data in self.graph.edges(data=True):
            if data.get("type") == "symbol_connection":
                edges.append({
                    "from": u,
                    "to": v,
                    "type": "connected_through_symbol",
                    "confidence": data.get("confidence", 0.0),
                })
            else:
                edges.append({
                    "from": u,
                    "to": v,
                    "type": "pipe",
                    "pipe_id": data.get("pipe_id"),
                    "length": data.get("length"),
                    "label": data.get("label"),
                    "diameter": data.get("diameter"),
                    "pipe_type": data.get("pipe_type"),
                })
        return {
            "nodes": nodes,
            "edges": edges,
            "num_junctions": len(self.junctions),
            "num_symbols": len(self.symbol_nodes),
            "num_pipes": len(self.pipe_segments),
            "text_regions": [
                {"bbox": t.bbox, "text": t.text, "confidence": t.confidence}
                for t in self.text_regions
            ],
        }

    def save_json(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    def summary(self) -> str:
        return (
            f"PipeGraph: {len(self.junctions)} junctions, "
            f"{len(self.symbol_nodes)} symbol nodes, "
            f"{len(self.pipe_segments)} pipe segments, "
            f"{nx.number_connected_components(self.graph)} connected components, "
            f"{len(self.text_regions)} text labels"
        )
