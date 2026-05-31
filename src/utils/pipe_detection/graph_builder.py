import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Optional
from scipy.spatial import KDTree
from .data_structures import Point, LineSegment, PipeSegment, Junction, PipeGraph
import logging

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Build a connectivity graph from detected line segments."""

    def __init__(self, config: Dict):
        self.config = config
        self.junction_snap_dist = config.get("junction_snap_dist", 15)
        self.min_pipe_length = config.get("min_pipe_length", 15)

    def build_graph(self, segments: List[LineSegment]) -> PipeGraph:
        """Convert line segments into a PipeGraph with junctions and pipes."""
        if not segments:
            logger.warning("No segments to build graph from")
            return PipeGraph()

        endpoints = self._collect_endpoints(segments)
        junction_positions = self._cluster_endpoints(endpoints)
        junctions = self._classify_junctions(junction_positions, segments)

        graph = PipeGraph()
        for j in junctions:
            graph.add_junction(j)

        pipe_id = 0
        for seg_idx, seg in enumerate(segments):
            if seg.length < self.min_pipe_length:
                continue

            from_j = self._find_nearest_junction(seg.start, junctions)
            to_j = self._find_nearest_junction(seg.end, junctions)

            if from_j is None or to_j is None:
                continue
            if from_j.id == to_j.id:
                continue

            pipe = PipeSegment(id=pipe_id, segments=[seg])
            graph.add_pipe(pipe, from_j.id, to_j.id)

            from_j.connected_pipe_ids.append(pipe_id)
            to_j.connected_pipe_ids.append(pipe_id)
            pipe_id += 1

        self._reclassify_junctions(graph)

        logger.info(graph.summary())
        return graph

    def _collect_endpoints(
        self, segments: List[LineSegment]
    ) -> List[Tuple[Point, int, str]]:
        """Collect all endpoints with their segment index and side (start/end)."""
        endpoints = []
        for i, seg in enumerate(segments):
            endpoints.append((seg.start, i, "start"))
            endpoints.append((seg.end, i, "end"))
        return endpoints

    def _cluster_endpoints(
        self, endpoints: List[Tuple[Point, int, str]]
    ) -> List[Point]:
        """Cluster nearby endpoints into junction positions using KDTree."""
        if not endpoints:
            return []

        coords = np.array([[p.x, p.y] for p, _, _ in endpoints])
        tree = KDTree(coords)

        visited = set()
        junction_positions = []

        for i in range(len(coords)):
            if i in visited:
                continue

            neighbors = tree.query_ball_point(coords[i], self.junction_snap_dist)
            cluster = [j for j in neighbors if j not in visited]
            if not cluster:
                continue

            for j in cluster:
                visited.add(j)

            cluster_coords = coords[cluster]
            centroid = cluster_coords.mean(axis=0)
            junction_positions.append(Point(centroid[0], centroid[1]))

        logger.info(
            f"Clustered {len(endpoints)} endpoints into {len(junction_positions)} junctions"
        )
        return junction_positions

    def _classify_junctions(
        self, positions: List[Point], segments: List[LineSegment]
    ) -> List[Junction]:
        """Create Junction objects with initial type classification."""
        junctions = []
        for i, pos in enumerate(positions):
            touching = self._count_touching_segments(pos, segments)
            jtype = self._junction_type_from_degree(touching)
            junctions.append(Junction(id=i, position=pos, junction_type=jtype))
        return junctions

    def _count_touching_segments(self, point: Point, segments: List[LineSegment]) -> int:
        """Count how many segments have an endpoint near this point."""
        count = 0
        for seg in segments:
            if (point.distance_to(seg.start) <= self.junction_snap_dist or
                    point.distance_to(seg.end) <= self.junction_snap_dist):
                count += 1
        return count

    def _junction_type_from_degree(self, degree: int) -> str:
        if degree <= 1:
            return "endpoint"
        elif degree == 2:
            return "L"
        elif degree == 3:
            return "T"
        elif degree >= 4:
            return "cross"
        return "unknown"

    def _find_nearest_junction(
        self, point: Point, junctions: List[Junction]
    ) -> Optional[Junction]:
        """Find the junction nearest to a point within snap distance."""
        best = None
        best_dist = float("inf")
        for j in junctions:
            d = point.distance_to(j.position)
            if d < best_dist:
                best_dist = d
                best = j
        if best_dist > self.junction_snap_dist * 2:
            return None
        return best

    def _reclassify_junctions(self, graph: PipeGraph):
        """Reclassify junction types based on actual graph connectivity."""
        for jid, junction in graph.junctions.items():
            degree = graph.graph.degree(jid)
            junction.junction_type = self._junction_type_from_degree(degree)

    def simplify_graph(self, graph: PipeGraph) -> PipeGraph:
        """Remove degree-2 pass-through junctions, merging their pipes.

        A degree-2 L-junction where both pipes are collinear is just a
        continuation — merge the two pipes and remove the junction.
        """
        to_remove = []
        for jid, junction in graph.junctions.items():
            if graph.graph.degree(jid) != 2:
                continue

            neighbors = list(graph.graph.neighbors(jid))
            if len(neighbors) != 2:
                continue

            edge1_data = graph.graph.edges[jid, neighbors[0]]
            edge2_data = graph.graph.edges[jid, neighbors[1]]

            pid1 = edge1_data.get("pipe_id")
            pid2 = edge2_data.get("pipe_id")

            if pid1 is None or pid2 is None:
                continue

            pipe1 = graph.pipe_segments.get(pid1)
            pipe2 = graph.pipe_segments.get(pid2)
            if pipe1 is None or pipe2 is None:
                continue

            both_h = all(s.is_horizontal for s in pipe1.segments + pipe2.segments)
            both_v = all(s.is_vertical for s in pipe1.segments + pipe2.segments)
            if not (both_h or both_v):
                continue

            merged_pipe = PipeSegment(
                id=pid1,
                segments=pipe1.segments + pipe2.segments,
                label=pipe1.label or pipe2.label,
                diameter=pipe1.diameter or pipe2.diameter,
                pipe_type=pipe1.pipe_type or pipe2.pipe_type,
            )

            to_remove.append((jid, neighbors[0], neighbors[1], pid1, pid2, merged_pipe))

        for jid, n1, n2, pid1, pid2, merged_pipe in to_remove:
            if jid not in graph.graph:
                continue
            graph.graph.remove_node(jid)
            del graph.junctions[jid]
            if pid2 in graph.pipe_segments:
                del graph.pipe_segments[pid2]
            graph.pipe_segments[pid1] = merged_pipe
            if not graph.graph.has_edge(n1, n2):
                graph.graph.add_edge(
                    n1, n2,
                    pipe_id=merged_pipe.id,
                    length=merged_pipe.total_length,
                    label=merged_pipe.label,
                    diameter=merged_pipe.diameter,
                    pipe_type=merged_pipe.pipe_type,
                )

        if to_remove:
            logger.info(f"Simplified graph: removed {len(to_remove)} pass-through junctions")
            self._reclassify_junctions(graph)

        return graph

    def filter_to_labeled_components(self, graph: PipeGraph) -> PipeGraph:
        """Keep only connected components that contain at least one labeled pipe."""
        labeled_nodes = set()
        for u, v, data in graph.graph.edges(data=True):
            pid = data.get("pipe_id")
            if pid is not None and pid in graph.pipe_segments:
                pipe = graph.pipe_segments[pid]
                if pipe.label or pipe.diameter:
                    labeled_nodes.add(u)
                    labeled_nodes.add(v)

        if not labeled_nodes:
            logger.warning("No labeled pipes found — returning empty graph")
            return PipeGraph()

        keep_nodes = set()
        for comp in nx.connected_components(graph.graph):
            if comp & labeled_nodes:
                keep_nodes.update(comp)

        remove_nodes = set(graph.graph.nodes) - keep_nodes
        filtered = PipeGraph()

        for jid in keep_nodes:
            if jid in graph.junctions:
                filtered.add_junction(graph.junctions[jid])

        for u, v, data in graph.graph.edges(data=True):
            if u in keep_nodes and v in keep_nodes:
                pid = data.get("pipe_id")
                if pid is not None and pid in graph.pipe_segments:
                    filtered.add_pipe(graph.pipe_segments[pid], u, v)

        filtered.text_regions = graph.text_regions

        removed_pipes = len(graph.pipe_segments) - len(filtered.pipe_segments)
        removed_junctions = len(graph.junctions) - len(filtered.junctions)
        logger.info(
            f"Label filter: kept {len(filtered.pipe_segments)}/{len(graph.pipe_segments)} pipes, "
            f"removed {removed_junctions} junctions, "
            f"{nx.number_connected_components(filtered.graph)} components remain"
        )
        return filtered
