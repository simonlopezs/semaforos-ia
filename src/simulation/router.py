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
    destination_street_id : the street where the final destination point is.
    destination_progress  : 0.0–1.0, exact point on the destination street.
    on_final_street       : True when the vehicle has entered the destination street
                            and is heading toward destination_progress.
    """
    intersection_ids: list[str] = field(default_factory=list)
    current_index: int = 0
    destination_street_id: str | None = None
    destination_progress: float = 0.5
    on_final_street: bool = False
    total_distance: float = 0.0      # meters, full route distance
    optimal_time: float = 0.0        # seconds, assuming all green + speed limit

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

    def compute_route_metrics(self, route: Route) -> None:
        """
        Calculate total_distance and optimal_time for a route.
        Optimal time assumes all green lights and driving at each street's speed limit.
        """
        total_dist = 0.0
        total_time = 0.0

        # Sum distance/time for each street segment in the route
        for i in range(len(route.intersection_ids) - 1):
            from_id = route.intersection_ids[i]
            to_id = route.intersection_ids[i + 1]
            street_id = self.street_between(from_id, to_id)
            if street_id:
                street = self.city.get_street(street_id)
                if street:
                    total_dist += street.length
                    speed = street.effective_max_speed_ms
                    if speed > 0:
                        total_time += street.length / speed

        # Add the final leg on the destination street
        if route.destination_street_id:
            dest_street = self.city.get_street(route.destination_street_id)
            if dest_street:
                # Distance from the entry intersection to the destination point
                last_iid = route.intersection_ids[-1] if route.intersection_ids else None
                if last_iid == dest_street.start_intersection_id:
                    leg_dist = route.destination_progress * dest_street.length
                elif last_iid == dest_street.end_intersection_id:
                    leg_dist = (1.0 - route.destination_progress) * dest_street.length
                else:
                    leg_dist = route.destination_progress * dest_street.length
                total_dist += leg_dist
                speed = dest_street.effective_max_speed_ms
                if speed > 0:
                    total_time += leg_dist / speed

        route.total_distance = total_dist
        route.optimal_time = total_time

    def street_between(self, from_id: str, to_id: str) -> str | None:
        """Find the street connecting two adjacent intersections."""
        for neighbor, _, street_id in self._adj.get(from_id, []):
            if neighbor == to_id:
                return street_id
        return None
