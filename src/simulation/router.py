"""
Pathfinding router for the road network.

Uses Dijkstra's algorithm on the intersection graph, weighted by
street length, to find the shortest route between two intersections.

A Route is an ordered list of intersection IDs the vehicle must visit.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field

from src.simulation.models.city import City


@dataclass
class Route:
    """
    The planned path for a vehicle.

    intersection_ids : ordered list of intersections from origin to destination.
    current_index    : pointer to the NEXT intersection the vehicle is heading toward.
    """
    intersection_ids: list[str] = field(default_factory=list)
    current_index: int = 0

    @property
    def destination_id(self) -> str | None:
        if not self.intersection_ids:
            return None
        return self.intersection_ids[-1]

    @property
    def next_intersection_id(self) -> str | None:
        if self.current_index >= len(self.intersection_ids):
            return None
        return self.intersection_ids[self.current_index]

    @property
    def is_finished(self) -> bool:
        return self.current_index >= len(self.intersection_ids)

    @property
    def remaining_steps(self) -> int:
        return max(0, len(self.intersection_ids) - self.current_index)

    def advance(self) -> str | None:
        """Mark the current intersection as reached; return the new next target."""
        self.current_index += 1
        return self.next_intersection_id

    def __repr__(self) -> str:
        total = len(self.intersection_ids)
        idx = self.current_index
        return f"Route(steps={total}, at={idx}/{total})"


class Router:
    """
    Builds shortest-path routes on the city's road graph using Dijkstra.
    Caches the adjacency list for fast repeated queries.
    """

    def __init__(self, city: City):
        self.city = city
        # adjacency: intersection_id → list of (neighbor_id, weight, street_id)
        self._adj: dict[str, list[tuple[str, float, str]]] = {}
        self._build_adjacency()

    def _build_adjacency(self) -> None:
        self._adj.clear()
        for iid in self.city.intersections:
            self._adj[iid] = []

        for street in self.city.streets.values():
            sid = street.start_intersection_id
            eid = street.end_intersection_id
            w = street.length
            if sid in self._adj:
                self._adj[sid].append((eid, w, street.id))
            if street.is_bidirectional and eid in self._adj:
                self._adj[eid].append((sid, w, street.id))

    def find_route(self, origin_id: str, destination_id: str) -> Route | None:
        """
        Dijkstra shortest path from origin to destination.
        Returns a Route or None if no path exists.
        """
        if origin_id == destination_id:
            return Route(intersection_ids=[origin_id], current_index=0)

        if origin_id not in self._adj or destination_id not in self._adj:
            return None

        # dist[node] = (cost, previous_node)
        dist: dict[str, float] = {origin_id: 0.0}
        prev: dict[str, str | None] = {origin_id: None}
        # priority queue: (cost, node_id)
        pq: list[tuple[float, str]] = [(0.0, origin_id)]

        while pq:
            cost, node = heapq.heappop(pq)
            if node == destination_id:
                break
            if cost > dist.get(node, float("inf")):
                continue
            for neighbor, weight, _ in self._adj.get(node, []):
                new_cost = cost + weight
                if new_cost < dist.get(neighbor, float("inf")):
                    dist[neighbor] = new_cost
                    prev[neighbor] = node
                    heapq.heappush(pq, (new_cost, neighbor))

        if destination_id not in prev:
            return None

        # Reconstruct path
        path: list[str] = []
        current: str | None = destination_id
        while current is not None:
            path.append(current)
            current = prev.get(current)
        path.reverse()

        # current_index = 1 because index 0 is the origin (already there)
        return Route(intersection_ids=path, current_index=1)

    def street_between(self, from_id: str, to_id: str) -> str | None:
        """Find the street connecting two adjacent intersections."""
        for neighbor, _, street_id in self._adj.get(from_id, []):
            if neighbor == to_id:
                return street_id
        return None
