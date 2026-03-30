"""
City model — the top-level container for the entire simulation world.

The City acts as the registry for all simulation entities.
It does NOT run the simulation loop (that belongs to the Simulator class,
to be built later); it only holds and organises the data.

Lookup methods return None rather than raise KeyError to simplify
caller code in the hot simulation path.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from src.simulation.models.geometry import Point2D
from src.simulation.models.street import Street
from src.simulation.models.intersection import Intersection
from src.simulation.models.traffic_light import TrafficLight
from src.simulation.models.vehicle import Vehicle
from src.simulation.models.spawn_point import SpawnPoint, DespawnPoint


@dataclass
class CityBounds:
    """Axis-aligned bounding box of the simulated area."""
    min_x: float = 0.0
    min_y: float = 0.0
    max_x: float = 1000.0
    max_y: float = 1000.0

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        return self.max_y - self.min_y

    @property
    def center(self) -> Point2D:
        return Point2D((self.min_x + self.max_x) / 2, (self.min_y + self.max_y) / 2)

    def contains(self, point: Point2D) -> bool:
        return (
            self.min_x <= point.x <= self.max_x
            and self.min_y <= point.y <= self.max_y
        )


@dataclass
class City:
    """
    The complete model of a simulated city.

    All entity registries are dicts keyed by entity ID for O(1) lookup.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Unnamed City"
    bounds: CityBounds = field(default_factory=CityBounds)

    # Road network
    streets: dict[str, Street] = field(default_factory=dict)
    intersections: dict[str, Intersection] = field(default_factory=dict)

    # Traffic control
    traffic_lights: dict[str, TrafficLight] = field(default_factory=dict)

    # Moving entities
    vehicles: dict[str, Vehicle] = field(default_factory=dict)

    # Boundary sources / sinks
    spawn_points: dict[str, SpawnPoint] = field(default_factory=dict)
    despawn_points: dict[str, DespawnPoint] = field(default_factory=dict)

    # ---------------------------------------------------------------------------
    # Registration helpers
    # ---------------------------------------------------------------------------

    def add_street(self, street: Street) -> Street:
        self.streets[street.id] = street
        return street

    def add_intersection(self, intersection: Intersection) -> Intersection:
        self.intersections[intersection.id] = intersection
        return intersection

    def add_traffic_light(self, light: TrafficLight) -> TrafficLight:
        self.traffic_lights[light.id] = light
        # Register in the intersection as well
        if light.intersection_id and light.intersection_id in self.intersections:
            self.intersections[light.intersection_id].add_traffic_light(light.id)
        return light

    def add_vehicle(self, vehicle: Vehicle) -> Vehicle:
        self.vehicles[vehicle.id] = vehicle
        return vehicle

    def remove_vehicle(self, vehicle_id: str) -> Vehicle | None:
        return self.vehicles.pop(vehicle_id, None)

    def add_spawn_point(self, sp: SpawnPoint) -> SpawnPoint:
        self.spawn_points[sp.id] = sp
        return sp

    def add_despawn_point(self, dp: DespawnPoint) -> DespawnPoint:
        self.despawn_points[dp.id] = dp
        return dp

    # ---------------------------------------------------------------------------
    # Lookup helpers
    # ---------------------------------------------------------------------------

    def get_street(self, street_id: str) -> Street | None:
        return self.streets.get(street_id)

    def get_intersection(self, intersection_id: str) -> Intersection | None:
        return self.intersections.get(intersection_id)

    def get_traffic_light(self, light_id: str) -> TrafficLight | None:
        return self.traffic_lights.get(light_id)

    def get_vehicle(self, vehicle_id: str) -> Vehicle | None:
        return self.vehicles.get(vehicle_id)

    def streets_at_intersection(self, intersection_id: str) -> list[Street]:
        node = self.intersections.get(intersection_id)
        if not node:
            return []
        return [self.streets[sid] for sid in node.street_ids if sid in self.streets]

    def lights_at_intersection(self, intersection_id: str) -> list[TrafficLight]:
        node = self.intersections.get(intersection_id)
        if not node:
            return []
        return [
            self.traffic_lights[lid]
            for lid in node.traffic_light_ids
            if lid in self.traffic_lights
        ]

    def neighbors(self, intersection_id: str) -> list[Intersection]:
        """All intersections reachable in one street hop."""
        result: list[Intersection] = []
        for street in self.streets_at_intersection(intersection_id):
            other_id = (
                street.end_intersection_id
                if street.start_intersection_id == intersection_id
                else street.start_intersection_id
            )
            if other_id and other_id in self.intersections:
                result.append(self.intersections[other_id])
        return result

    # ---------------------------------------------------------------------------
    # Stats / summary
    # ---------------------------------------------------------------------------

    @property
    def stats(self) -> dict:
        return {
            "name": self.name,
            "streets": len(self.streets),
            "intersections": len(self.intersections),
            "traffic_lights": len(self.traffic_lights),
            "vehicles": len(self.vehicles),
            "spawn_points": len(self.spawn_points),
            "despawn_points": len(self.despawn_points),
        }

    def __repr__(self) -> str:
        s = self.stats
        return (
            f"City(name='{self.name}', "
            f"streets={s['streets']}, "
            f"intersections={s['intersections']}, "
            f"lights={s['traffic_lights']}, "
            f"vehicles={s['vehicles']})"
        )
