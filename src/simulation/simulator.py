"""
Core simulation engine.

Responsibilities:
  1. Advance traffic lights each tick.
  2. Spawn vehicles at SpawnPoints according to traffic_level.
  3. Move vehicles along streets (street speed, not per-vehicle yet).
  4. Brake at red/yellow lights.
  5. Basic following distance (don't overlap the vehicle ahead).
  6. Transfer vehicles between streets at intersections (random walk).
  7. Despawn vehicles that reach a DespawnPoint or leave the map.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from src.simulation.config import SimulationConfig
from src.simulation.enums import VehicleState, VehicleType, TrafficLightState
from src.simulation.models.city import City
from src.simulation.models.driver import Driver
from src.simulation.models.geometry import Point2D
from src.simulation.models.street import Street
from src.simulation.models.intersection import Intersection
from src.simulation.models.traffic_light import TrafficLight, TrafficLightPhase, standard_phases
from src.simulation.models.vehicle import Vehicle, VEHICLE_SPECS
from src.simulation.models.spawn_point import SpawnPoint, SpawnKind


def randomize_traffic_lights(city: City, seed: int = 0) -> None:
    """
    Give each intersection its own random green durations while keeping
    opposing movements coordinated (NS green ↔ EW red and vice-versa).
    """
    rng = random.Random(seed)

    # Group lights by intersection
    by_intersection: dict[str, list[TrafficLight]] = {}
    for light in city.traffic_lights.values():
        by_intersection.setdefault(light.intersection_id, []).append(light)

    for iid, lights in by_intersection.items():
        green_ns = rng.uniform(15, 45)
        green_ew = rng.uniform(15, 45)
        yellow = rng.uniform(3, 5)

        for light in lights:
            # N-S approaches: orientation ≈ 90° or 270°
            is_ns = abs(light.orientation_deg % 180 - 90) < 45
            if is_ns:
                light.phases = [
                    TrafficLightPhase(TrafficLightState.GREEN, green_ns),
                    TrafficLightPhase(TrafficLightState.YELLOW, yellow),
                    TrafficLightPhase(TrafficLightState.RED, green_ew + yellow),
                ]
                light.phase_offset = 0.0
            else:
                light.phases = [
                    TrafficLightPhase(TrafficLightState.GREEN, green_ew),
                    TrafficLightPhase(TrafficLightState.YELLOW, yellow),
                    TrafficLightPhase(TrafficLightState.RED, green_ns + yellow),
                ]
                light.phase_offset = green_ns + yellow

            light.current_phase_index = 0
            light.time_in_current_phase = 0.0
            light.apply_offset()


class Simulator:
    def __init__(self, city: City, config: SimulationConfig, seed: int = 42):
        self.city = city
        self.config = config
        self.rng = random.Random(seed)
        self.time: float = 0.0
        self.total_spawned: int = 0
        self.total_despawned: int = 0

        # Accumulator per spawn point for Poisson process
        self._spawn_accum: dict[str, float] = {
            sp_id: 0.0 for sp_id in city.spawn_points
        }

        # Randomize traffic light timings
        randomize_traffic_lights(city, seed=seed)

    # ------------------------------------------------------------------
    # Main tick
    # ------------------------------------------------------------------

    def tick(self, dt: float) -> None:
        self.time += dt
        self._advance_lights(dt)
        self._spawn_vehicles(dt)
        self._update_vehicles(dt)
        self._despawn_vehicles()

    # ------------------------------------------------------------------
    # 1. Traffic lights
    # ------------------------------------------------------------------

    def _advance_lights(self, dt: float) -> None:
        for light in self.city.traffic_lights.values():
            light.advance(dt)

    # ------------------------------------------------------------------
    # 2. Spawn
    # ------------------------------------------------------------------

    def _spawn_vehicles(self, dt: float) -> None:
        if len(self.city.vehicles) >= self.config.max_vehicles:
            return

        for sp in self.city.spawn_points.values():
            if not sp.is_active:
                continue

            rate = sp.spawn_rate * self.config.traffic_level  # veh/min
            expected = rate * dt / 60.0

            self._spawn_accum[sp.id] += expected
            while self._spawn_accum[sp.id] >= 1.0:
                self._spawn_accum[sp.id] -= 1.0
                self._try_spawn(sp)

                if len(self.city.vehicles) >= self.config.max_vehicles:
                    return

    def _try_spawn(self, sp: SpawnPoint) -> None:
        street = self.city.get_street(sp.street_id)
        if not street:
            return

        # Pick vehicle type from weights
        types = list(sp.vehicle_type_weights.keys())
        weights = list(sp.vehicle_type_weights.values())
        vtype_str = self.rng.choices(types, weights=weights, k=1)[0]
        vtype = VehicleType(vtype_str)
        spec = VEHICLE_SPECS.get(vtype, VEHICLE_SPECS[VehicleType.CAR])

        driver = Driver.random(seed=self.rng.randint(0, 2**31))
        vehicle = Vehicle.from_spec(
            spec,
            driver=driver,
            seed=self.rng.randint(0, 2**31),
        )

        # Determine direction on street
        if sp.kind == SpawnKind.CITY_BORDER:
            # Border: enter from the intersection closest to this spawn
            start_inter = self.city.get_intersection(street.start_intersection_id)
            end_inter = self.city.get_intersection(street.end_intersection_id)
            if start_inter and sp.position.distance_to(start_inter.position) < 5:
                forward = True
                vehicle.street_progress = 0.0
            elif end_inter and sp.position.distance_to(end_inter.position) < 5:
                forward = False
                vehicle.street_progress = 1.0
            else:
                forward = self.rng.choice([True, False])
                vehicle.street_progress = sp.street_progress
        else:
            # Mid-block (residential, etc): random direction
            forward = self.rng.choice([True, False])
            vehicle.street_progress = sp.street_progress

        vehicle.current_street_id = street.id
        vehicle.heading = street.bearing_deg if forward else (street.bearing_deg + 180) % 360
        vehicle.speed = 0.0
        vehicle.state = VehicleState.ACCELERATING

        # Store forward flag in a simple way: lane_index 0=forward, 1=backward
        vehicle.current_lane_index = 0 if forward else 1

        # Update position from street geometry
        pos, _ = street.point_at_distance(vehicle.street_progress * street.length)
        vehicle.position = pos

        self.city.add_vehicle(vehicle)
        self.total_spawned += 1

    # ------------------------------------------------------------------
    # 3. Vehicle update
    # ------------------------------------------------------------------

    def _update_vehicles(self, dt: float) -> None:
        # Build lookup: street_id → sorted vehicles for following distance
        street_vehicles: dict[str, list[Vehicle]] = {}
        for v in self.city.vehicles.values():
            if v.current_street_id:
                street_vehicles.setdefault(v.current_street_id, []).append(v)

        # Sort by progress (forward vehicles ascending, backward descending)
        for sid, vehs in street_vehicles.items():
            vehs.sort(key=lambda v: v.street_progress if v.current_lane_index == 0 else -v.street_progress)

        to_transfer: list[Vehicle] = []

        for v in list(self.city.vehicles.values()):
            street = self.city.get_street(v.current_street_id) if v.current_street_id else None
            if not street:
                continue

            is_forward = v.current_lane_index == 0
            target_speed = street.effective_max_speed_ms

            # Distance to the intersection ahead
            if is_forward:
                dist_to_end = (1.0 - v.street_progress) * street.length
                target_intersection_id = street.end_intersection_id
            else:
                dist_to_end = v.street_progress * street.length
                target_intersection_id = street.start_intersection_id

            # --- Check traffic light ---
            should_stop = False
            light = self._find_light_for_approach(target_intersection_id, v.heading)
            if light and dist_to_end < 100:  # only check within 100m
                if light.current_state == TrafficLightState.RED:
                    should_stop = True
                elif light.current_state == TrafficLightState.YELLOW:
                    # If we can't clear the intersection before red, brake
                    time_to_red = light.time_remaining_in_phase
                    time_to_reach = dist_to_end / max(v.speed, 0.1)
                    if time_to_reach > time_to_red:
                        should_stop = True

            # --- Check vehicle ahead (following distance) ---
            vehs_on_street = street_vehicles.get(v.current_street_id, [])
            gap_ahead = self._gap_to_vehicle_ahead(v, vehs_on_street, street)

            # --- Decide acceleration ---
            if should_stop:
                stop_dist = dist_to_end - self.config.stop_margin
                if stop_dist <= 0:
                    # Already at/past stop line
                    v.speed = 0.0
                    v.acceleration = 0.0
                    v.state = VehicleState.WAITING_LIGHT
                else:
                    # Kinematic braking: a = -v² / (2d)
                    if v.speed > 0.1:
                        needed_decel = v.speed ** 2 / (2 * max(stop_dist, 0.5))
                        v.acceleration = -min(needed_decel, self.config.default_deceleration * 2)
                        v.state = VehicleState.DECELERATING
                    else:
                        v.speed = 0.0
                        v.acceleration = 0.0
                        v.state = VehicleState.STOPPED
            elif gap_ahead is not None and gap_ahead < self.config.min_following_gap + v.length:
                # Too close to vehicle ahead → brake
                v.acceleration = -self.config.default_deceleration
                v.state = VehicleState.DECELERATING
            elif v.speed < target_speed:
                # Accelerate toward street speed
                v.acceleration = self.config.default_acceleration
                v.state = VehicleState.ACCELERATING
            else:
                # Cruising
                v.acceleration = 0.0
                v.speed = target_speed
                v.state = VehicleState.CRUISING

            # --- Integrate motion ---
            v.speed = max(0.0, min(v.speed + v.acceleration * dt, target_speed * 1.1))

            progress_delta = v.speed * dt / street.length if street.length > 0 else 0
            if is_forward:
                v.street_progress += progress_delta
            else:
                v.street_progress -= progress_delta

            # Update world position
            clamped_progress = max(0.0, min(1.0, v.street_progress))
            pos, heading = street.point_at_distance(clamped_progress * street.length)
            v.position = pos
            if not is_forward:
                heading = (heading + 180) % 360
            v.heading = heading

            # Bookkeeping
            v.distance_traveled += v.speed * dt
            v.time_alive += dt
            if v.speed < 0.1:
                v.time_waiting += dt

            # --- Check if reached end of street ---
            reached_end = (
                (is_forward and v.street_progress >= 1.0) or
                (not is_forward and v.street_progress <= 0.0)
            )
            if reached_end:
                to_transfer.append(v)

        # Transfer vehicles to next street
        for v in to_transfer:
            self._transfer_vehicle(v)

    def _find_light_for_approach(
        self, intersection_id: str, vehicle_heading: float
    ) -> TrafficLight | None:
        """Find the traffic light at an intersection that matches a given approach heading."""
        lights = self.city.lights_at_intersection(intersection_id)
        if not lights:
            return None

        best_light = None
        best_diff = 360.0
        for light in lights:
            diff = abs(((vehicle_heading - light.orientation_deg + 180) % 360) - 180)
            if diff < best_diff:
                best_diff = diff
                best_light = light

        return best_light if best_diff < 60 else None

    def _gap_to_vehicle_ahead(
        self, vehicle: Vehicle, vehicles_on_street: list[Vehicle], street: Street
    ) -> float | None:
        """
        Distance in meters to the next vehicle ahead on the same street/direction.
        Returns None if no vehicle ahead.
        """
        is_forward = vehicle.current_lane_index == 0
        min_gap = None

        for other in vehicles_on_street:
            if other.id == vehicle.id:
                continue
            if other.current_lane_index != vehicle.current_lane_index:
                continue

            if is_forward:
                if other.street_progress > vehicle.street_progress:
                    gap = (other.street_progress - vehicle.street_progress) * street.length
                    gap -= other.length  # measure to rear of vehicle ahead
                    if min_gap is None or gap < min_gap:
                        min_gap = gap
            else:
                if other.street_progress < vehicle.street_progress:
                    gap = (vehicle.street_progress - other.street_progress) * street.length
                    gap -= other.length
                    if min_gap is None or gap < min_gap:
                        min_gap = gap

        return min_gap

    # ------------------------------------------------------------------
    # 4. Transfer & Despawn
    # ------------------------------------------------------------------

    def _transfer_vehicle(self, vehicle: Vehicle) -> None:
        """Move vehicle from one street to the next at an intersection."""
        street = self.city.get_street(vehicle.current_street_id)
        if not street:
            self.city.remove_vehicle(vehicle.id)
            return

        is_forward = vehicle.current_lane_index == 0
        intersection_id = (
            street.end_intersection_id if is_forward
            else street.start_intersection_id
        )
        intersection = self.city.get_intersection(intersection_id)
        if not intersection:
            self.city.remove_vehicle(vehicle.id)
            self.total_despawned += 1
            return

        # Check if this is a despawn point
        for dp in self.city.despawn_points.values():
            if dp.is_active and vehicle.position.distance_to(dp.position) < dp.capture_radius:
                self.city.remove_vehicle(vehicle.id)
                self.total_despawned += 1
                return

        # Pick next street (not the one we came from)
        candidates = [
            sid for sid in intersection.street_ids
            if sid != street.id
        ]
        if not candidates:
            # Dead end, U-turn
            candidates = [street.id]

        next_street_id = self.rng.choice(candidates)
        next_street = self.city.get_street(next_street_id)
        if not next_street:
            self.city.remove_vehicle(vehicle.id)
            self.total_despawned += 1
            return

        # Determine direction on new street
        if next_street.start_intersection_id == intersection_id:
            vehicle.current_lane_index = 0  # forward
            vehicle.street_progress = 0.01
        elif next_street.end_intersection_id == intersection_id:
            vehicle.current_lane_index = 1  # backward
            vehicle.street_progress = 0.99
        else:
            self.city.remove_vehicle(vehicle.id)
            self.total_despawned += 1
            return

        vehicle.current_street_id = next_street.id

        # Update heading
        is_forward = vehicle.current_lane_index == 0
        base_heading = next_street.bearing_deg
        vehicle.heading = base_heading if is_forward else (base_heading + 180) % 360

    def _despawn_vehicles(self) -> None:
        """Remove vehicles that are stuck too long or out of bounds."""
        to_remove: list[str] = []
        for v in self.city.vehicles.values():
            # Remove vehicles alive > 5 minutes
            if v.time_alive > 300:
                to_remove.append(v.id)
            # Remove out of bounds
            elif not self.city.bounds.contains(v.position):
                to_remove.append(v.id)

        for vid in to_remove:
            self.city.remove_vehicle(vid)
            self.total_despawned += 1
