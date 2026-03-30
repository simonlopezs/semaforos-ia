"""
Spawn and despawn points for vehicles.

SpawnPoint  → vehicles enter the simulation here.
DespawnPoint → vehicles exit the simulation here.

Both are attached to a street; their position is usually at the boundary
of the simulated area (city edge, freeway on-ramp, etc.).
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from src.simulation.enums import VehicleType
from src.simulation.models.geometry import Point2D


@dataclass
class SpawnPoint:
    """
    A source of new vehicles entering the simulation.

    Vehicle types are drawn according to `vehicle_type_weights`.
    `spawn_rate` vehicles are generated per minute on average (Poisson process).

    `heading` is the initial bearing of spawned vehicles (degrees).
    `target_despawn_ids` optionally restricts which despawn points are
    assigned as destination — if empty, any despawn point is valid.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    position: Point2D = field(default_factory=lambda: Point2D(0.0, 0.0))

    # Which street/lane vehicles spawn onto
    street_id: str = ""
    lane_index: int = 0

    # Initial bearing of spawned vehicles (degrees, 0°=East)
    heading: float = 0.0

    # Vehicles per minute (Poisson rate)
    spawn_rate: float = 10.0

    # Probability weights per vehicle type (need not sum to 1; will be normalised)
    vehicle_type_weights: dict[str, float] = field(default_factory=lambda: {
        VehicleType.CAR.value:         0.70,
        VehicleType.MOTORCYCLE.value:  0.15,
        VehicleType.VAN.value:         0.08,
        VehicleType.BUS.value:         0.04,
        VehicleType.TRUCK.value:       0.03,
    })

    # Destination restriction (empty = any DespawnPoint)
    target_despawn_ids: list[str] = field(default_factory=list)

    # Time-of-day multipliers: maps hour (0–23) → rate multiplier
    # e.g. {8: 2.5, 17: 2.0} doubles rate at 8 AM and 5 PM
    hourly_multipliers: dict[int, float] = field(default_factory=dict)

    # Is this spawn point currently active?
    is_active: bool = True

    def effective_rate(self, hour: int) -> float:
        """Spawn rate adjusted for time-of-day."""
        multiplier = self.hourly_multipliers.get(hour, 1.0)
        return self.spawn_rate * multiplier

    def __repr__(self) -> str:
        return (
            f"SpawnPoint(name='{self.name}', "
            f"rate={self.spawn_rate:.1f}/min, "
            f"pos={self.position})"
        )


@dataclass
class DespawnPoint:
    """
    A sink where vehicles exit the simulation.

    When a vehicle reaches its assigned DespawnPoint, it is removed from
    the simulation and its statistics are recorded.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    position: Point2D = field(default_factory=lambda: Point2D(0.0, 0.0))

    # Street and lane where vehicles "exit"
    street_id: str = ""
    lane_index: int = 0

    # Capture radius: vehicle is despawned when its front is within this distance
    capture_radius: float = 5.0    # meters

    is_active: bool = True

    def __repr__(self) -> str:
        return f"DespawnPoint(name='{self.name}', pos={self.position})"
