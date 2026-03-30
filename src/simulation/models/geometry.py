"""
2D geometry primitives used throughout the simulation.
All coordinates are in meters. Angles are in degrees (0° = East, CCW positive).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Point2D:
    x: float
    y: float

    def distance_to(self, other: Point2D) -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def bearing_to(self, other: Point2D) -> float:
        """Angle in degrees from this point toward other (0° = East, CCW)."""
        return math.degrees(math.atan2(other.y - self.y, other.x - self.x))

    def translate(self, dx: float, dy: float) -> Point2D:
        return Point2D(self.x + dx, self.y + dy)

    def move_toward(self, other: Point2D, distance: float) -> Point2D:
        """Return a new point moved `distance` meters toward `other`."""
        d = self.distance_to(other)
        if d == 0:
            return Point2D(self.x, self.y)
        ratio = distance / d
        return Point2D(
            self.x + (other.x - self.x) * ratio,
            self.y + (other.y - self.y) * ratio,
        )

    def __iter__(self):
        yield self.x
        yield self.y

    def __repr__(self) -> str:
        return f"Point2D({self.x:.2f}, {self.y:.2f})"


@dataclass
class Vector2D:
    x: float
    y: float

    @classmethod
    def from_angle(cls, angle_deg: float, magnitude: float = 1.0) -> Vector2D:
        rad = math.radians(angle_deg)
        return cls(math.cos(rad) * magnitude, math.sin(rad) * magnitude)

    @property
    def magnitude(self) -> float:
        return math.sqrt(self.x ** 2 + self.y ** 2)

    @property
    def angle_deg(self) -> float:
        return math.degrees(math.atan2(self.y, self.x))

    def normalized(self) -> Vector2D:
        m = self.magnitude
        if m == 0:
            return Vector2D(0.0, 0.0)
        return Vector2D(self.x / m, self.y / m)

    def scale(self, factor: float) -> Vector2D:
        return Vector2D(self.x * factor, self.y * factor)

    def dot(self, other: Vector2D) -> float:
        return self.x * other.x + self.y * other.y

    def __add__(self, other: Vector2D) -> Vector2D:
        return Vector2D(self.x + other.x, self.y + other.y)


@dataclass
class Segment2D:
    """A directed segment between two points."""
    start: Point2D
    end: Point2D

    @property
    def length(self) -> float:
        return self.start.distance_to(self.end)

    @property
    def direction(self) -> Vector2D:
        return Vector2D(
            self.end.x - self.start.x,
            self.end.y - self.start.y,
        ).normalized()

    @property
    def bearing_deg(self) -> float:
        return self.start.bearing_to(self.end)

    def point_at(self, t: float) -> Point2D:
        """Point at fraction t (0=start, 1=end) along the segment."""
        t = max(0.0, min(1.0, t))
        return Point2D(
            self.start.x + (self.end.x - self.start.x) * t,
            self.start.y + (self.end.y - self.start.y) * t,
        )

    def closest_point(self, p: Point2D) -> Point2D:
        """Closest point on segment to external point p."""
        dx = self.end.x - self.start.x
        dy = self.end.y - self.start.y
        length_sq = dx * dx + dy * dy
        if length_sq == 0:
            return Point2D(self.start.x, self.start.y)
        t = ((p.x - self.start.x) * dx + (p.y - self.start.y) * dy) / length_sq
        return self.point_at(t)
