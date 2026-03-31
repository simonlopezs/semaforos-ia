"""
Core simulation engine.

Responsibilities:
  1. Advance traffic lights each tick.
  2. Spawn vehicles at SpawnPoints with a route to a random DespawnPoint.
  3. Move vehicles along streets at street speed.
  4. Brake at red/yellow lights — vehicles queue behind each other with 1m gap.
  5. Follow the planned route at each intersection.
  6. Despawn vehicles when they reach their destination.
"""
from __future__ import annotations

import random

from src.simulation.config import SimulationConfig
from src.simulation.enums import VehicleState, VehicleType, TrafficLightState
from src.simulation.models.city import City
from src.simulation.models.driver import Driver
from src.simulation.models.geometry import Point2D
from src.simulation.models.street import Street
from src.simulation.models.traffic_light import TrafficLight, TrafficLightPhase
from src.simulation.models.vehicle import Vehicle, VEHICLE_SPECS
from src.simulation.models.spawn_point import SpawnPoint, SpawnKind
from src.simulation.physics import emergency_braking
from src.simulation.router import Router, Route


# ---------------------------------------------------------------------------
# Traffic-light randomisation helper
# ---------------------------------------------------------------------------

def randomize_traffic_lights(city: City, seed: int = 0) -> None:
    rng = random.Random(seed)
    by_intersection: dict[str, list[TrafficLight]] = {}
    for light in city.traffic_lights.values():
        by_intersection.setdefault(light.intersection_id, []).append(light)

    for iid, lights in by_intersection.items():
        green_ns = rng.uniform(15, 45)
        green_ew = rng.uniform(15, 45)
        yellow = rng.uniform(3, 5)
        all_red = 2.0  # clearance interval — both directions red

        # Full cycle sequence (shared by all lights at this intersection):
        #   NS_GREEN → NS_YELLOW → ALL_RED → EW_GREEN → EW_YELLOW → ALL_RED
        #
        # NS sees: GREEN, YELLOW, RED,   RED,     RED,     RED
        # EW sees: RED,   RED,    RED,   GREEN,   YELLOW,  RED
        #
        # We express this as explicit phase lists with NO offsets,
        # so both directions share the exact same clock.

        for light in lights:
            is_ns = abs(light.orientation_deg % 180 - 90) < 45
            if is_ns:
                light.phases = [
                    TrafficLightPhase(TrafficLightState.GREEN, green_ns),
                    TrafficLightPhase(TrafficLightState.YELLOW, yellow),
                    TrafficLightPhase(TrafficLightState.RED, all_red),       # clearance
                    TrafficLightPhase(TrafficLightState.RED, green_ew),      # EW has green
                    TrafficLightPhase(TrafficLightState.RED, yellow),        # EW has yellow
                    TrafficLightPhase(TrafficLightState.RED, all_red),       # clearance
                ]
            else:
                light.phases = [
                    TrafficLightPhase(TrafficLightState.RED, green_ns),      # NS has green
                    TrafficLightPhase(TrafficLightState.RED, yellow),        # NS has yellow
                    TrafficLightPhase(TrafficLightState.RED, all_red),       # clearance
                    TrafficLightPhase(TrafficLightState.GREEN, green_ew),
                    TrafficLightPhase(TrafficLightState.YELLOW, yellow),
                    TrafficLightPhase(TrafficLightState.RED, all_red),       # clearance
                ]

            light.phase_offset = 0.0
            light.current_phase_index = 0
            light.time_in_current_phase = 0.0


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class Simulator:
    def __init__(self, city: City, config: SimulationConfig, seed: int = 42):
        self.city = city
        self.config = config
        self.rng = random.Random(seed)
        self.time: float = 0.0
        self.total_spawned: int = 0
        self.total_despawned: int = 0

        # Optimality tracking: ratio of optimal_time / actual_time for completed trips
        self.completed_trips: int = 0
        self.optimality_sum: float = 0.0  # sum of (optimal / actual) per trip

        self._spawn_accum: dict[str, float] = {
            sp_id: 0.0 for sp_id in city.spawn_points
        }

        randomize_traffic_lights(city, seed=seed)
        self.router = Router(city)

        # Pre-spawn vehicles so the city isn't empty at start
        self._pre_spawn()

    @property
    def avg_optimality(self) -> float:
        """Average ratio of optimal_time / actual_time across all completed trips."""
        if self.completed_trips == 0:
            return 0.0
        return self.optimality_sum / self.completed_trips

    def _record_trip(self, vehicle: Vehicle) -> None:
        """Record optimality metric when a vehicle completes its trip."""
        route: Route | None = vehicle.planned_route  # type: ignore[assignment]
        if route and route.optimal_time > 0 and vehicle.time_alive > 0.1:
            optimality = route.optimal_time / vehicle.time_alive
            self.optimality_sum += min(optimality, 1.0)
            self.completed_trips += 1

    def _pre_spawn(self) -> None:
        """Spawn an initial batch of vehicles so the city isn't empty."""
        target = int(self.config.max_vehicles * self.config.traffic_level * 0.3)
        spawn_list = [sp for sp in self.city.spawn_points.values() if sp.is_active]
        if not spawn_list:
            return
        attempts = 0
        while self.total_spawned < target and attempts < target * 3:
            sp = self.rng.choice(spawn_list)
            self._try_spawn(sp)
            attempts += 1

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

            rate = sp.spawn_rate * self.config.traffic_level
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

        # Pick vehicle type
        types = list(sp.vehicle_type_weights.keys())
        weights = list(sp.vehicle_type_weights.values())
        vtype = VehicleType(self.rng.choices(types, weights=weights, k=1)[0])
        spec = VEHICLE_SPECS.get(vtype, VEHICLE_SPECS[VehicleType.CAR])

        driver = Driver.random(seed=self.rng.randint(0, 2**31))
        vehicle = Vehicle.from_spec(spec, driver=driver, seed=self.rng.randint(0, 2**31))

        # Determine direction on street and origin intersection
        if sp.kind == SpawnKind.CITY_BORDER:
            start_inter = self.city.get_intersection(street.start_intersection_id)
            end_inter = self.city.get_intersection(street.end_intersection_id)
            if start_inter and sp.position.distance_to(start_inter.position) < 5:
                forward = True
                vehicle.street_progress = 0.01
                origin_intersection_id = street.start_intersection_id
            elif end_inter and sp.position.distance_to(end_inter.position) < 5:
                forward = False
                vehicle.street_progress = 0.99
                origin_intersection_id = street.end_intersection_id
            else:
                forward = self.rng.choice([True, False])
                vehicle.street_progress = sp.street_progress
                origin_intersection_id = (
                    street.end_intersection_id if forward
                    else street.start_intersection_id
                )
        else:
            forward = self.rng.choice([True, False])
            vehicle.street_progress = sp.street_progress
            origin_intersection_id = (
                street.end_intersection_id if forward
                else street.start_intersection_id
            )

        # Pick a random point on a random street as destination
        all_streets = list(self.city.streets.values())
        if not all_streets:
            return

        dest_street = self.rng.choice(all_streets)
        dest_progress = self.rng.uniform(0.15, 0.85)  # avoid extremes near intersections

        # Route to the intersection closest to the destination point
        # Try both ends — pick the one that gives a valid route
        route: Route | None = None
        for dest_iid in (dest_street.start_intersection_id, dest_street.end_intersection_id):
            if dest_iid == origin_intersection_id and dest_street.id == street.id:
                continue  # skip routing to ourselves
            candidate = self.router.find_route(origin_intersection_id, dest_iid)
            if candidate and not candidate.is_finished:
                candidate.destination_street_id = dest_street.id
                candidate.destination_progress = dest_progress
                route = candidate
                break

        if not route:
            return  # can't route, skip

        # Calculate optimal travel time (all green, speed limit)
        self.router.compute_route_metrics(route)

        vehicle.planned_route = route
        vehicle.destination_id = dest_street.id

        # Temporarily set lane to check room
        vehicle.current_lane_index = 0 if forward else 1

        # Find the entry intersection for this direction
        entry_intersection = (
            street.start_intersection_id if forward
            else street.end_intersection_id
        )

        # Use _place_on_street to position behind existing queue
        placed = self._place_on_street(vehicle, street, entry_intersection)
        if not placed:
            return  # street is full, skip this spawn

        vehicle.state = VehicleState.ACCELERATING

        pos, _ = street.point_at_distance(
            max(0.0, min(1.0, vehicle.street_progress)) * street.length
        )
        vehicle.position = pos

        self.city.add_vehicle(vehicle)
        self.total_spawned += 1

    # ------------------------------------------------------------------
    # 3. Vehicle update
    # ------------------------------------------------------------------

    def _update_vehicles(self, dt: float) -> None:
        # Build per-street, per-direction sorted vehicle lists for queuing
        # Key: (street_id, lane_index) → vehicles sorted by progress toward end
        lane_vehicles: dict[tuple[str, int], list[Vehicle]] = {}
        for v in self.city.vehicles.values():
            if v.current_street_id:
                key = (v.current_street_id, v.current_lane_index)
                lane_vehicles.setdefault(key, []).append(v)

        # Sort: vehicle closest to the end of the street first
        for key, vehs in lane_vehicles.items():
            _, lane_idx = key
            if lane_idx == 0:  # forward: higher progress = closer to end
                vehs.sort(key=lambda v: -v.street_progress)
            else:  # backward: lower progress = closer to end
                vehs.sort(key=lambda v: v.street_progress)

        to_transfer: list[Vehicle] = []

        for v in list(self.city.vehicles.values()):
            street = self.city.get_street(v.current_street_id) if v.current_street_id else None
            if not street:
                continue

            is_forward = v.current_lane_index == 0
            target_speed = street.effective_max_speed_ms

            # Distance from front of vehicle to the intersection ahead
            if is_forward:
                dist_to_end = (1.0 - v.street_progress) * street.length
                target_intersection_id = street.end_intersection_id
            else:
                dist_to_end = v.street_progress * street.length
                target_intersection_id = street.start_intersection_id

            # --- Stop line distance from intersection geometry ---
            target_inter = self.city.get_intersection(target_intersection_id)
            stop_margin = target_inter.stop_line_distance if target_inter else self.config.stop_margin

            # --- Check traffic light ---
            must_stop_for_light = False
            light = self._find_light_for_approach(target_intersection_id, v.heading)
            if light and dist_to_end < 120:
                if light.current_state == TrafficLightState.RED:
                    must_stop_for_light = True
                elif light.current_state == TrafficLightState.YELLOW:
                    # Can we physically stop before the intersection?
                    stop_dist = dist_to_end - stop_margin
                    braking = emergency_braking(
                        speed_ms=v.speed,
                        mass_kg=v.mass,
                        distance_available=max(0.0, stop_dist),
                        weather=self.config.weather,
                        pavement=street.pavement_condition,
                        reaction_time=v.driver.reaction_time * 0.3,  # reduced: already alert
                    )
                    if braking.can_stop:
                        # Physics says we CAN stop → brake
                        must_stop_for_light = True
                    # else: can't stop safely → commit to crossing

            # --- Calculate stop target from traffic light ---
            # The stop point is at the edge of the intersection square.
            light_stop_dist: float | None = None
            if must_stop_for_light:
                light_stop_dist = dist_to_end - stop_margin

            # --- Calculate stop target from vehicle ahead ---
            # gap = distance from my front to the rear of the vehicle ahead
            key = (v.current_street_id, v.current_lane_index)
            vehs_sorted = lane_vehicles.get(key, [])
            ahead_stop_dist: float | None = None
            gap_ahead = self._gap_to_vehicle_ahead(v, vehs_sorted, street)
            if gap_ahead is not None:
                # We want to keep exactly 1 meter behind the rear of the vehicle ahead
                desired_gap = 1.0
                # Distance we can travel before hitting the desired gap
                ahead_stop_dist = gap_ahead - desired_gap
                if ahead_stop_dist < 0:
                    ahead_stop_dist = 0.0

            # --- Calculate stop target from destination (final street) ---
            dest_stop_dist: float | None = None
            route_obj: Route | None = v.planned_route  # type: ignore[assignment]
            if route_obj and route_obj.on_final_street:
                if is_forward:
                    dest_stop_dist = (route_obj.destination_progress - v.street_progress) * street.length
                else:
                    dest_stop_dist = (v.street_progress - route_obj.destination_progress) * street.length
                if dest_stop_dist < 0:
                    dest_stop_dist = 0.0

            # --- Pick the more restrictive stop target ---
            effective_stop_dist: float | None = None
            for candidate in (light_stop_dist, ahead_stop_dist, dest_stop_dist):
                if candidate is not None:
                    if effective_stop_dist is None or candidate < effective_stop_dist:
                        effective_stop_dist = candidate

            # --- Decide acceleration ---
            # Compute physics-based max deceleration for this vehicle + conditions
            from src.simulation.physics import compute_effective_friction, G
            import math

            mu_eff = compute_effective_friction(
                self.config.weather, street.pavement_condition, v.mass
            )
            max_decel = mu_eff * G  # physics-limited deceleration
            accel = self.config.default_acceleration

            if effective_stop_dist is not None and effective_stop_dist <= 0.05:
                # Already at stop point — full stop
                v.speed = 0.0
                v.acceleration = 0.0
                v.state = (
                    VehicleState.WAITING_LIGHT if must_stop_for_light
                    else VehicleState.STOPPED
                )
            elif effective_stop_dist is not None:
                # Max speed at which we can still brake in time:
                #   v_max = sqrt(2 × max_decel × distance)
                max_safe_speed = math.sqrt(
                    2.0 * max_decel * max(effective_stop_dist, 0.01)
                )
                max_safe_speed = min(max_safe_speed, target_speed)

                if v.speed > max_safe_speed + 0.1:
                    # Above safe speed — brake (capped to physics limit)
                    needed_decel = v.speed ** 2 / (2.0 * max(effective_stop_dist, 0.1))
                    v.acceleration = -min(needed_decel, max_decel)
                    v.state = VehicleState.DECELERATING
                elif v.speed < max_safe_speed - 0.1:
                    # Below safe speed — can still accelerate toward stop point
                    v.acceleration = accel * min(1.0, effective_stop_dist / 10.0)
                    v.state = VehicleState.ACCELERATING
                else:
                    v.acceleration = 0.0
                    v.state = VehicleState.CRUISING
            elif v.speed < target_speed:
                v.acceleration = accel
                v.state = VehicleState.ACCELERATING
            else:
                v.acceleration = 0.0
                v.speed = target_speed
                v.state = VehicleState.CRUISING

            # --- Integrate motion ---
            new_speed = v.speed + v.acceleration * dt
            v.speed = max(0.0, min(new_speed, target_speed * 1.1))

            progress_delta = v.speed * dt / street.length if street.length > 0 else 0
            if is_forward:
                v.street_progress += progress_delta
            else:
                v.street_progress -= progress_delta

            # --- Hard clamp: front must not pass destination point ---
            if route_obj and route_obj.on_final_street and street.length > 0:
                if is_forward and v.street_progress > route_obj.destination_progress:
                    v.street_progress = route_obj.destination_progress
                    v.speed = 0.0
                    v.acceleration = 0.0
                elif not is_forward and v.street_progress < route_obj.destination_progress:
                    v.street_progress = route_obj.destination_progress
                    v.speed = 0.0
                    v.acceleration = 0.0

            # --- Hard clamp: front must not pass the stop line ---
            if must_stop_for_light and street.length > 0:
                if is_forward:
                    stop_progress = 1.0 - stop_margin / street.length
                    if v.street_progress > stop_progress:
                        v.street_progress = stop_progress
                        v.speed = 0.0
                        v.acceleration = 0.0
                        v.state = VehicleState.WAITING_LIGHT
                else:
                    stop_progress = stop_margin / street.length
                    if v.street_progress < stop_progress:
                        v.street_progress = stop_progress
                        v.speed = 0.0
                        v.acceleration = 0.0
                        v.state = VehicleState.WAITING_LIGHT

            clamped_progress = max(0.0, min(1.0, v.street_progress))
            pos, heading = street.point_at_distance(clamped_progress * street.length)
            v.position = pos
            if not is_forward:
                heading = (heading + 180) % 360
            v.heading = heading

            v.distance_traveled += v.speed * dt
            v.time_alive += dt
            if v.speed < 0.1:
                v.time_waiting += dt

            # --- Check if reached destination on final street ---
            if route_obj and route_obj.on_final_street:
                reached_dest = False
                if is_forward and v.street_progress >= route_obj.destination_progress - 0.005:
                    reached_dest = True
                elif not is_forward and v.street_progress <= route_obj.destination_progress + 0.005:
                    reached_dest = True
                if reached_dest and v.speed < 0.5:
                    self._record_trip(v)
                    self.city.remove_vehicle(v.id)
                    self.total_despawned += 1
                    continue

            # --- Check if reached end of street ---
            reached_end = (
                (is_forward and v.street_progress >= 1.0) or
                (not is_forward and v.street_progress <= 0.0)
            )
            if reached_end:
                to_transfer.append(v)

        # --- Enforce no-overlap: push vehicles apart post-integration ---
        self._resolve_overlaps(lane_vehicles)

        for v in to_transfer:
            self._transfer_vehicle(v)

    def _resolve_overlaps(
        self, lane_vehicles: dict[tuple[str, int], list[Vehicle]]
    ) -> None:
        """
        After integration, ensure no two vehicles overlap.
        Walk from the front of the queue backward, pushing each vehicle
        so it sits at least 1m behind the rear of the one ahead.
        """
        gap_m = 1.0  # minimum bumper-to-bumper gap

        for (street_id, lane_idx), vehs in lane_vehicles.items():
            if len(vehs) < 2:
                continue
            street = self.city.get_street(street_id)
            if not street or street.length <= 0:
                continue

            # vehs is already sorted: first element is closest to the END of the street
            for i in range(1, len(vehs)):
                ahead = vehs[i - 1]
                behind = vehs[i]

                if lane_idx == 0:  # forward
                    # ahead has higher progress
                    # Rear of ahead = ahead.progress - ahead.length / street.length
                    rear_ahead = ahead.street_progress - ahead.length / street.length
                    # Front of behind should be at most rear_ahead - gap/street.length
                    max_progress = rear_ahead - gap_m / street.length
                    if behind.street_progress > max_progress:
                        behind.street_progress = max(0.0, max_progress)
                        behind.speed = min(behind.speed, ahead.speed)
                else:  # backward
                    rear_ahead = ahead.street_progress + ahead.length / street.length
                    min_progress = rear_ahead + gap_m / street.length
                    if behind.street_progress < min_progress:
                        behind.street_progress = min(1.0, min_progress)
                        behind.speed = min(behind.speed, ahead.speed)

                # Update world position
                clamped = max(0.0, min(1.0, behind.street_progress))
                pos, heading = street.point_at_distance(clamped * street.length)
                behind.position = pos
                if lane_idx == 1:
                    heading = (heading + 180) % 360
                behind.heading = heading

    def _find_light_for_approach(
        self, intersection_id: str, vehicle_heading: float
    ) -> TrafficLight | None:
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
        self, vehicle: Vehicle, vehicles_sorted: list[Vehicle], street: Street
    ) -> float | None:
        """
        Distance in meters from the FRONT of `vehicle` to the REAR of the
        nearest vehicle ahead in the same lane.
        Returns None if no vehicle ahead.
        """
        is_forward = vehicle.current_lane_index == 0
        best_gap: float | None = None

        for other in vehicles_sorted:
            if other.id == vehicle.id:
                continue

            if is_forward:
                if other.street_progress <= vehicle.street_progress:
                    continue
                front_to_front = (other.street_progress - vehicle.street_progress) * street.length
            else:
                if other.street_progress >= vehicle.street_progress:
                    continue
                front_to_front = (vehicle.street_progress - other.street_progress) * street.length

            # Gap = front-to-front distance minus the length of the vehicle ahead
            gap = front_to_front - other.length
            if best_gap is None or gap < best_gap:
                best_gap = gap

        return best_gap

    # ------------------------------------------------------------------
    # 4. Transfer & Despawn
    # ------------------------------------------------------------------

    def _transfer_vehicle(self, vehicle: Vehicle) -> None:
        street = self.city.get_street(vehicle.current_street_id)
        if not street:
            self.city.remove_vehicle(vehicle.id)
            self.total_despawned += 1
            return

        is_forward = vehicle.current_lane_index == 0
        arrived_at_id = (
            street.end_intersection_id if is_forward
            else street.start_intersection_id
        )

        # Advance route
        route: Route | None = vehicle.planned_route  # type: ignore[assignment]
        if route:
            if route.next_intersection_id == arrived_at_id:
                route.advance()

            if route.is_finished:
                # Route intersections done — enter destination street for final leg
                if (
                    route.destination_street_id
                    and not route.on_final_street
                ):
                    dest_street = self.city.get_street(route.destination_street_id)
                    if dest_street:
                        placed = self._place_on_street(
                            vehicle, dest_street, arrived_at_id
                        )
                        if placed:
                            route.on_final_street = True
                            return
                        # Can't enter — wait
                        vehicle.speed = 0.0
                        vehicle.state = VehicleState.STOPPED
                        route.current_index = max(0, route.current_index - 1)
                        if is_forward:
                            vehicle.street_progress = 0.99
                        else:
                            vehicle.street_progress = 0.01
                        return

                # No destination street or already on it — despawn
                self._record_trip(vehicle)
                self.city.remove_vehicle(vehicle.id)
                self.total_despawned += 1
                return

            # Follow route: find the street to the next intersection
            next_target = route.next_intersection_id
            if next_target:
                street_id = self.router.street_between(arrived_at_id, next_target)
                if street_id:
                    next_street = self.city.get_street(street_id)
                    if next_street:
                        placed = self._place_on_street(vehicle, next_street, arrived_at_id)
                        if not placed:
                            # Street is full — stay at end of current street, stopped
                            vehicle.speed = 0.0
                            vehicle.acceleration = 0.0
                            vehicle.state = VehicleState.STOPPED
                            # Undo the route advance so we retry next tick
                            route.current_index = max(0, route.current_index - 1)
                            # Clamp progress so we don't re-trigger transfer
                            if is_forward:
                                vehicle.street_progress = 0.99
                            else:
                                vehicle.street_progress = 0.01
                            return
                        return

        # Fallback: no route or route broken — despawn
        self.city.remove_vehicle(vehicle.id)
        self.total_despawned += 1

    def _place_on_street(
        self, vehicle: Vehicle, street: Street, from_intersection_id: str
    ) -> bool:
        """
        Place a vehicle on a new street coming from a given intersection.
        Positions it behind the last vehicle in the queue (1m gap).
        Returns False if the entry is blocked (no room).
        """
        if street.start_intersection_id == from_intersection_id:
            lane_idx = 0  # forward
        elif street.end_intersection_id == from_intersection_id:
            lane_idx = 1  # backward
        else:
            return False

        # Find the last (rearmost) vehicle in this lane to position behind it
        entry_progress = 0.0 if lane_idx == 0 else 1.0
        min_gap_m = 1.0  # 1 meter gap

        rearmost_progress: float | None = None
        rearmost_length: float = 0.0
        for other in self.city.vehicles.values():
            if other.current_street_id != street.id:
                continue
            if other.current_lane_index != lane_idx:
                continue

            if lane_idx == 0:
                # Forward: entry is at progress ~0. Find vehicle with lowest progress.
                if rearmost_progress is None or other.street_progress < rearmost_progress:
                    rearmost_progress = other.street_progress
                    rearmost_length = other.length
            else:
                # Backward: entry is at progress ~1. Find vehicle with highest progress.
                if rearmost_progress is None or other.street_progress > rearmost_progress:
                    rearmost_progress = other.street_progress
                    rearmost_length = other.length

        if rearmost_progress is not None and street.length > 0:
            # Need to place behind the rearmost vehicle
            needed_space_m = rearmost_length + min_gap_m + vehicle.length
            needed_progress = needed_space_m / street.length

            if lane_idx == 0:
                # The rearmost vehicle's rear is at progress - length/street_length
                rear_of_rearmost = rearmost_progress - rearmost_length / street.length
                my_progress = rear_of_rearmost - (min_gap_m + vehicle.length) / street.length
                if my_progress < -0.05:
                    return False  # no room
                entry_progress = max(0.001, my_progress)
            else:
                rear_of_rearmost = rearmost_progress + rearmost_length / street.length
                my_progress = rear_of_rearmost + (min_gap_m + vehicle.length) / street.length
                if my_progress > 1.05:
                    return False  # no room
                entry_progress = min(0.999, my_progress)
        else:
            entry_progress = 0.01 if lane_idx == 0 else 0.99

        vehicle.current_street_id = street.id
        vehicle.current_lane_index = lane_idx
        vehicle.street_progress = entry_progress

        is_forward = lane_idx == 0
        new_heading = (
            street.bearing_deg if is_forward
            else (street.bearing_deg + 180) % 360
        )

        # Reduce speed based on turn angle (sharper turn = more speed loss)
        import math
        angle_diff = abs(((new_heading - vehicle.heading + 180) % 360) - 180)
        # 0° = straight → keep 90%, 90° = right angle → keep 40%, 180° = U-turn → keep 10%
        speed_factor = max(0.1, 1.0 - 0.6 * (angle_diff / 180.0))
        vehicle.speed *= speed_factor

        vehicle.heading = new_heading

        # Update world position immediately so vehicle doesn't blink
        clamped = max(0.0, min(1.0, entry_progress))
        pos, _ = street.point_at_distance(clamped * street.length)
        vehicle.position = pos

        return True

    def _despawn_vehicles(self) -> None:
        to_remove: list[str] = []
        for v in self.city.vehicles.values():
            if v.time_alive > 300:
                to_remove.append(v.id)
            elif not self.city.bounds.contains(v.position):
                to_remove.append(v.id)

        for vid in to_remove:
            self.city.remove_vehicle(vid)
            self.total_despawned += 1
