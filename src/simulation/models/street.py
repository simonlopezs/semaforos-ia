"""
Street and Lane models.

A Street is a directed polyline of Point2D nodes that connects two intersections.
If `is_bidirectional` is True, the street supports travel in both directions
and has lanes split between them.

Geometry:
  - `nodes` define the shape of the street's centerline.
  - Lane offsets are computed perpendicular to each segment.
  - All distances in meters, speeds in m/s (stored) / km/h (display).

Pavement condition affects:
  - `speed_factor`: multiplier on the effective speed limit (0.6–1.0)
  - `friction_factor`: affects braking distance (0.5–1.0)
"""
from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field

from src.simulation.enums import LaneDirection, PavementCondition
from src.simulation.models.geometry import Point2D, Segment2D


# ---------------------------------------------------------------------------
# Pavement lookup tables
# ---------------------------------------------------------------------------

_PAVEMENT_SPEED_FACTOR: dict[PavementCondition, float] = {
    PavementCondition.EXCELLENT: 1.00,
    PavementCondition.GOOD:      0.95,
    PavementCondition.FAIR:      0.85,
    PavementCondition.POOR:      0.70,
    PavementCondition.DAMAGED:   0.55,
}

_PAVEMENT_FRICTION_FACTOR: dict[PavementCondition, float] = {
    PavementCondition.EXCELLENT: 1.00,
    PavementCondition.GOOD:      0.95,
    PavementCondition.FAIR:      0.85,
    PavementCondition.POOR:      0.70,
    PavementCondition.DAMAGED:   0.50,
}


# ---------------------------------------------------------------------------
# Lane
# ---------------------------------------------------------------------------

@dataclass
class Lane:
    """
    A single lane within a street.

    `index` is the lane number counting from the road centerline outward.
    `direction` indicates whether it runs forward (start→end) or backward.
    `width` is in meters.
    """
    index: int
    direction: LaneDirection
    width: float = 3.5   # meters; standard urban lane ~3.0–3.75 m

    @property
    def label(self) -> str:
        dir_char = "F" if self.direction == LaneDirection.FORWARD else "B"
        return f"L{dir_char}{self.index}"


# ---------------------------------------------------------------------------
# Street
# ---------------------------------------------------------------------------

@dataclass
class Street:
    """
    A road segment connecting two intersections, described by a polyline.

    The street's canonical direction is from `start_intersection_id` toward
    `end_intersection_id` following `nodes` in order.

    For bidirectional streets, forward lanes run start→end and backward
    lanes run end→start.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""

    # Geometry: ordered list of points defining the centerline
    nodes: list[Point2D] = field(default_factory=list)

    # Connectivity
    start_intersection_id: str = ""
    end_intersection_id: str = ""

    # Traffic rules
    is_bidirectional: bool = True
    lanes_per_direction: int = 1     # number of lanes in each direction
    max_speed_kmh: float = 50.0      # posted speed limit in km/h
    has_bike_lane: bool = False
    has_bus_lane: bool = False

    # Physical condition
    pavement_condition: PavementCondition = PavementCondition.GOOD

    # Optional metadata
    zone: str = ""    # e.g. "residential", "downtown", "highway", "school"

    # ---------------------------------------------------------------------------
    # Derived / computed properties
    # ---------------------------------------------------------------------------

    @property
    def max_speed_ms(self) -> float:
        """Posted limit converted to m/s."""
        return self.max_speed_kmh / 3.6

    @property
    def speed_factor(self) -> float:
        """Multiplier on max speed due to pavement quality."""
        return _PAVEMENT_SPEED_FACTOR[self.pavement_condition]

    @property
    def friction_factor(self) -> float:
        """Multiplier on deceleration capacity due to pavement quality."""
        return _PAVEMENT_FRICTION_FACTOR[self.pavement_condition]

    @property
    def effective_max_speed_ms(self) -> float:
        """Actual achievable top speed considering pavement."""
        return self.max_speed_ms * self.speed_factor

    @property
    def effective_max_speed_kmh(self) -> float:
        return self.effective_max_speed_ms * 3.6

    @property
    def segments(self) -> list[Segment2D]:
        """The polyline broken into directed segments."""
        return [
            Segment2D(self.nodes[i], self.nodes[i + 1])
            for i in range(len(self.nodes) - 1)
        ]

    @property
    def length(self) -> float:
        """Total arc length of the street in meters."""
        return sum(s.length for s in self.segments)

    @property
    def total_lanes(self) -> int:
        factor = 2 if self.is_bidirectional else 1
        return self.lanes_per_direction * factor

    @property
    def lanes(self) -> list[Lane]:
        """All lanes in this street, ordered centerline→kerb in each direction."""
        result: list[Lane] = []
        for i in range(self.lanes_per_direction):
            result.append(Lane(index=i, direction=LaneDirection.FORWARD))
        if self.is_bidirectional:
            for i in range(self.lanes_per_direction):
                result.append(Lane(index=i, direction=LaneDirection.BACKWARD))
        return result

    @property
    def bearing_deg(self) -> float:
        """Approximate overall bearing of the street (first→last node)."""
        if len(self.nodes) < 2:
            return 0.0
        return self.nodes[0].bearing_to(self.nodes[-1])

    # ---------------------------------------------------------------------------
    # Geometry helpers
    # ---------------------------------------------------------------------------

    def point_at_distance(self, distance: float) -> tuple[Point2D, float]:
        """
        Return the position and heading at `distance` meters from the start.
        Returns (point, heading_deg).
        """
        remaining = distance
        for seg in self.segments:
            if remaining <= seg.length:
                t = remaining / seg.length if seg.length > 0 else 0.0
                return seg.point_at(t), seg.bearing_deg
            remaining -= seg.length
        # Past end → clamp to last point
        last_seg = self.segments[-1] if self.segments else None
        if last_seg:
            return last_seg.end, last_seg.bearing_deg
        return self.nodes[-1], 0.0

    def lane_offset(self, lane: Lane, lane_width: float = 3.5) -> float:
        """
        Lateral offset in meters from the centerline for a given lane.
        Positive = left side (when facing direction of travel).
        """
        # Forward lanes sit to the right of center (negative offset in left-hand coords)
        # Backward lanes sit to the left
        center_offset = (lane.index + 0.5) * lane_width
        if lane.direction == LaneDirection.FORWARD:
            return -center_offset
        else:
            return center_offset

    def closest_segment(self, point: Point2D) -> tuple[Segment2D, float]:
        """
        Find the segment closest to `point` and the distance to it.
        Returns (segment, distance).
        """
        best_seg = None
        best_dist = float("inf")
        for seg in self.segments:
            closest = seg.closest_point(point)
            dist = point.distance_to(closest)
            if dist < best_dist:
                best_dist = dist
                best_seg = seg
        return best_seg, best_dist  # type: ignore[return-value]

    def __repr__(self) -> str:
        bidir = "↔" if self.is_bidirectional else "→"
        return (
            f"Street(name='{self.name}', {bidir}, "
            f"len={self.length:.0f}m, "
            f"limit={self.max_speed_kmh:.0f}km/h, "
            f"pavement={self.pavement_condition.value})"
        )
