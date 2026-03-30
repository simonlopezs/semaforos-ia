"""
Traffic light model.

A TrafficLight controls one approach direction at an intersection.
It cycles through a list of TrafficLightPhase objects.

Timing is in seconds. The simulation advances `time_in_current_phase`
each tick and calls `.advance(dt)` to auto-transition.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from src.simulation.enums import TrafficLightState
from src.simulation.models.geometry import Point2D


@dataclass
class TrafficLightPhase:
    """A single phase in a traffic light cycle."""
    state: TrafficLightState
    duration: float    # seconds this phase lasts


# ---------------------------------------------------------------------------
# Standard phase patterns (helpers)
# ---------------------------------------------------------------------------

def standard_phases(
    green_duration: float = 30.0,
    yellow_duration: float = 4.0,
    red_duration: float = 30.0,
    all_red_duration: float = 1.0,
) -> list[TrafficLightPhase]:
    """
    Typical 4-phase sequence used in many signalised intersections.
    all_red provides an inter-green clearance interval.
    """
    return [
        TrafficLightPhase(TrafficLightState.GREEN,  green_duration),
        TrafficLightPhase(TrafficLightState.YELLOW, yellow_duration),
        TrafficLightPhase(TrafficLightState.RED,    red_duration),
        TrafficLightPhase(TrafficLightState.RED,    all_red_duration),  # clearance
    ]


def pedestrian_phases(
    green_duration: float = 20.0,
    yellow_duration: float = 3.0,
    red_duration: float = 40.0,
) -> list[TrafficLightPhase]:
    """Shorter green, longer red — typical near schools/crossings."""
    return [
        TrafficLightPhase(TrafficLightState.GREEN,  green_duration),
        TrafficLightPhase(TrafficLightState.YELLOW, yellow_duration),
        TrafficLightPhase(TrafficLightState.RED,    red_duration),
    ]


# ---------------------------------------------------------------------------
# TrafficLight
# ---------------------------------------------------------------------------

@dataclass
class TrafficLight:
    """
    A traffic signal head controlling one movement (one approach or turn).

    `orientation_deg` is the compass bearing of the traffic approaching
    this light (e.g., 90° = vehicles coming from the West heading East).
    Multiple TrafficLight instances per intersection form a signal group.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Location
    position: Point2D = field(default_factory=lambda: Point2D(0.0, 0.0))

    # Orientation: bearing of the controlled movement (degrees, 0°=East, CCW)
    orientation_deg: float = 0.0

    # Which intersection this light belongs to
    intersection_id: str = ""

    # Which street/lane approach this light governs
    controlled_street_id: str = ""

    # Optional: specific lane indices this head applies to (empty = all lanes)
    controlled_lane_indices: list[int] = field(default_factory=list)

    # Phase cycle
    phases: list[TrafficLightPhase] = field(default_factory=lambda: standard_phases())

    # Offset into the cycle (seconds) — used to coordinate multiple lights
    # at the same intersection so they start at different points.
    phase_offset: float = 0.0

    # Runtime state
    current_phase_index: int = 0
    time_in_current_phase: float = 0.0
    is_active: bool = True

    # ---------------------------------------------------------------------------
    # Properties
    # ---------------------------------------------------------------------------

    @property
    def current_phase(self) -> TrafficLightPhase:
        return self.phases[self.current_phase_index]

    @property
    def current_state(self) -> TrafficLightState:
        if not self.is_active:
            return TrafficLightState.OFF
        return self.current_phase.state

    @property
    def time_remaining_in_phase(self) -> float:
        return max(0.0, self.current_phase.duration - self.time_in_current_phase)

    @property
    def cycle_duration(self) -> float:
        return sum(p.duration for p in self.phases)

    def is_green(self) -> bool:
        return self.current_state == TrafficLightState.GREEN

    def is_red(self) -> bool:
        return self.current_state in (
            TrafficLightState.RED, TrafficLightState.FLASHING_RED
        )

    def is_yellow(self) -> bool:
        return self.current_state in (
            TrafficLightState.YELLOW, TrafficLightState.FLASHING_YELLOW
        )

    # ---------------------------------------------------------------------------
    # Simulation update
    # ---------------------------------------------------------------------------

    def apply_offset(self) -> None:
        """
        Fast-forward the light state to account for phase_offset.
        Call once after construction, before the simulation loop starts.
        """
        remaining = self.phase_offset % self.cycle_duration
        while remaining > 0:
            phase_duration = self.phases[self.current_phase_index].duration
            if remaining >= phase_duration:
                remaining -= phase_duration
                self.current_phase_index = (
                    (self.current_phase_index + 1) % len(self.phases)
                )
            else:
                self.time_in_current_phase = remaining
                remaining = 0

    def advance(self, dt: float) -> TrafficLightState | None:
        """
        Advance the light by `dt` seconds.
        Returns the new state if a phase transition occurred, else None.
        """
        if not self.is_active:
            return None

        self.time_in_current_phase += dt
        transitioned = None

        while self.time_in_current_phase >= self.current_phase.duration:
            self.time_in_current_phase -= self.current_phase.duration
            self.current_phase_index = (self.current_phase_index + 1) % len(self.phases)
            transitioned = self.current_state

        return transitioned

    def force_state(self, state: TrafficLightState, duration: float = 60.0) -> None:
        """Override the light to a fixed state (e.g., emergency pre-emption)."""
        self.phases = [TrafficLightPhase(state, duration)]
        self.current_phase_index = 0
        self.time_in_current_phase = 0.0

    def set_flashing(self, flashing_yellow: bool = True) -> None:
        """Enter flashing mode (fault / night mode)."""
        state = (
            TrafficLightState.FLASHING_YELLOW
            if flashing_yellow
            else TrafficLightState.FLASHING_RED
        )
        self.phases = [TrafficLightPhase(state, 9999.0)]
        self.current_phase_index = 0
        self.time_in_current_phase = 0.0

    def __repr__(self) -> str:
        return (
            f"TrafficLight(state={self.current_state.value}, "
            f"orientation={self.orientation_deg:.0f}°, "
            f"remaining={self.time_remaining_in_phase:.1f}s)"
        )
