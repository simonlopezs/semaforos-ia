"""
Intersection model.

An intersection is a node in the road graph where streets meet.
It holds references to the streets that connect through it, and to the
traffic lights that regulate movements at that node.

Signal groups: a SignalGroup clusters the lights that are green at the
same time (conflicting movements must never share a group).
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from src.simulation.models.geometry import Point2D


@dataclass
class SignalGroup:
    """
    A set of traffic light IDs that display the same state simultaneously.
    Used to model coordinated phases at complex intersections.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""                       # e.g. "NS_through", "EW_left_turn"
    traffic_light_ids: list[str] = field(default_factory=list)


@dataclass
class Intersection:
    """
    A point where two or more streets meet.

    Attributes
    ----------
    position        : geometric center of the intersection box
    street_ids      : IDs of all streets that start or end here
    traffic_light_ids : IDs of all traffic lights at this intersection
    signal_groups   : optional grouping of lights into coordinated phases
    stop_line_distance: how far back from center vehicles stop (meters)
    is_roundabout   : if True, yield-at-entry rules apply instead of signals
    has_stop_sign   : if True and no signals, vehicles yield in priority order
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    position: Point2D = field(default_factory=lambda: Point2D(0.0, 0.0))

    # Road graph connectivity
    street_ids: list[str] = field(default_factory=list)

    # Traffic control
    traffic_light_ids: list[str] = field(default_factory=list)
    signal_groups: list[SignalGroup] = field(default_factory=list)

    # Geometry / rules
    stop_line_distance: float = 3.0    # meters from center to stop line
    is_roundabout: bool = False
    has_stop_sign: bool = False        # all-way stop if True and no lights

    # ---------------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------------

    @property
    def is_signalised(self) -> bool:
        return len(self.traffic_light_ids) > 0

    @property
    def is_unsignalised(self) -> bool:
        return not self.is_signalised

    def connects_streets(self, street_id_a: str, street_id_b: str) -> bool:
        return street_id_a in self.street_ids and street_id_b in self.street_ids

    def add_street(self, street_id: str) -> None:
        if street_id not in self.street_ids:
            self.street_ids.append(street_id)

    def add_traffic_light(self, light_id: str) -> None:
        if light_id not in self.traffic_light_ids:
            self.traffic_light_ids.append(light_id)

    def add_signal_group(self, group: SignalGroup) -> None:
        self.signal_groups.append(group)

    def __repr__(self) -> str:
        ctrl = "signalised" if self.is_signalised else (
            "stop-sign" if self.has_stop_sign else "uncontrolled"
        )
        if self.is_roundabout:
            ctrl = "roundabout"
        return (
            f"Intersection(name='{self.name}', "
            f"streets={len(self.street_ids)}, "
            f"control={ctrl}, "
            f"pos={self.position})"
        )
