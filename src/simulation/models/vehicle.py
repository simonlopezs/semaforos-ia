"""
Vehicle model.

Units used throughout:
  - Distances:     meters (m)
  - Speeds:        meters/second (m/s)  — helper properties convert to km/h
  - Accelerations: meters/second² (m/s²)
  - Angles:        degrees (0° = East, CCW positive)
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.simulation.enums import VehicleState, VehicleType
from src.simulation.models.geometry import Point2D, Vector2D
from src.simulation.models.driver import Driver

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Physical specs per vehicle type (realistic defaults)
# ---------------------------------------------------------------------------

@dataclass
class VehicleSpec:
    """
    Static physical specification of a vehicle category.
    Actual instances may vary ±10 % around these values.
    """
    vehicle_type: VehicleType

    # Dimensions (meters)
    length: float          # bumper-to-bumper
    width: float

    # Dynamics (m/s, m/s²)
    max_speed: float       # governed / design top speed
    max_acceleration: float
    max_deceleration: float   # absolute value; comfortable braking
    emergency_deceleration: float  # absolute value; hard braking

    # Mass (kg) — influences braking distance on poor pavement
    mass: float


VEHICLE_SPECS: dict[VehicleType, VehicleSpec] = {
    VehicleType.MOTORCYCLE: VehicleSpec(
        vehicle_type=VehicleType.MOTORCYCLE,
        length=2.2,   width=0.8,
        max_speed=33.3,          # 120 km/h
        max_acceleration=4.0,
        max_deceleration=7.0,
        emergency_deceleration=10.0,
        mass=200,
    ),
    VehicleType.CAR: VehicleSpec(
        vehicle_type=VehicleType.CAR,
        length=4.5,   width=1.9,
        max_speed=38.9,          # 140 km/h
        max_acceleration=3.0,
        max_deceleration=6.5,
        emergency_deceleration=9.0,
        mass=1400,
    ),
    VehicleType.VAN: VehicleSpec(
        vehicle_type=VehicleType.VAN,
        length=5.5,   width=2.1,
        max_speed=33.3,          # 120 km/h
        max_acceleration=2.2,
        max_deceleration=5.5,
        emergency_deceleration=8.0,
        mass=2500,
    ),
    VehicleType.BUS: VehicleSpec(
        vehicle_type=VehicleType.BUS,
        length=12.0,  width=2.5,
        max_speed=25.0,          # 90 km/h
        max_acceleration=1.2,
        max_deceleration=4.0,
        emergency_deceleration=6.5,
        mass=12000,
    ),
    VehicleType.TRUCK: VehicleSpec(
        vehicle_type=VehicleType.TRUCK,
        length=16.5,  width=2.55,
        max_speed=22.2,          # 80 km/h
        max_acceleration=0.8,
        max_deceleration=3.5,
        emergency_deceleration=5.5,
        mass=25000,
    ),
}


# ---------------------------------------------------------------------------
# Vehicle instance
# ---------------------------------------------------------------------------

@dataclass
class Vehicle:
    """
    A single vehicle moving through the simulation.

    Position and heading describe the **front center** of the vehicle.
    Route is an ordered list of intersection IDs the vehicle will visit.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    vehicle_type: VehicleType = VehicleType.CAR

    # Physical characteristics (may differ slightly from spec defaults)
    length: float = 4.5          # meters
    width: float = 1.9           # meters
    mass: float = 1400.0         # kg

    # Dynamics limits (m/s, m/s²)
    max_speed: float = 38.9
    max_acceleration: float = 3.0
    max_deceleration: float = 6.5
    emergency_deceleration: float = 9.0

    # Current kinematic state
    position: Point2D = field(default_factory=lambda: Point2D(0.0, 0.0))
    heading: float = 0.0         # degrees; direction the front faces
    speed: float = 0.0           # m/s, always >= 0
    acceleration: float = 0.0   # m/s², signed (+ = speeding up, - = braking)

    # Behavioral state
    state: VehicleState = VehicleState.STOPPED
    driver: Driver = field(default_factory=Driver)

    # Navigation
    current_street_id: str | None = None
    current_lane_index: int = 0
    # Progress along the current street's polyline [0.0 – 1.0]
    street_progress: float = 0.0
    # Planned route (set by Router — typed as Any to avoid circular import)
    planned_route: object | None = None
    # Destination despawn point ID
    destination_id: str | None = None

    # Simulation bookkeeping
    distance_traveled: float = 0.0   # meters since spawn
    time_alive: float = 0.0          # seconds since spawn
    time_waiting: float = 0.0        # seconds spent stopped/waiting

    # ---------------------------------------------------------------------------
    # Convenience properties
    # ---------------------------------------------------------------------------

    @property
    def speed_kmh(self) -> float:
        return self.speed * 3.6

    @property
    def velocity_vector(self) -> Vector2D:
        return Vector2D.from_angle(self.heading, self.speed)

    @property
    def is_stopped(self) -> bool:
        return self.speed < 0.01

    @property
    def rear_position(self) -> Point2D:
        """Center of the rear bumper."""
        import math
        rad = math.radians(self.heading)
        return Point2D(
            self.position.x - math.cos(rad) * self.length,
            self.position.y - math.sin(rad) * self.length,
        )

    @property
    def center_position(self) -> Point2D:
        import math
        rad = math.radians(self.heading)
        return Point2D(
            self.position.x - math.cos(rad) * self.length / 2,
            self.position.y - math.sin(rad) * self.length / 2,
        )

    # ---------------------------------------------------------------------------
    # Factory
    # ---------------------------------------------------------------------------

    @classmethod
    def from_spec(
        cls,
        spec: VehicleSpec,
        driver: Driver | None = None,
        position: Point2D | None = None,
        heading: float = 0.0,
        length_jitter: float = 0.05,
        seed: int | None = None,
    ) -> Vehicle:
        """
        Instantiate a vehicle from a VehicleSpec with optional randomisation.
        `length_jitter` adds ± jitter*length variation to dimensions/mass.
        """
        import random
        rng = random.Random(seed)

        def vary(base: float) -> float:
            return base * (1 + rng.uniform(-length_jitter, length_jitter))

        return cls(
            vehicle_type=spec.vehicle_type,
            length=vary(spec.length),
            width=vary(spec.width),
            mass=vary(spec.mass),
            max_speed=vary(spec.max_speed),
            max_acceleration=vary(spec.max_acceleration),
            max_deceleration=vary(spec.max_deceleration),
            emergency_deceleration=vary(spec.emergency_deceleration),
            position=position or Point2D(0.0, 0.0),
            heading=heading,
            driver=driver or Driver.random(seed=rng.randint(0, 2**31)),
        )

    def __repr__(self) -> str:
        return (
            f"Vehicle(type={self.vehicle_type.value}, "
            f"speed={self.speed_kmh:.1f}km/h, "
            f"state={self.state.value}, "
            f"pos={self.position})"
        )
