"""
Grid city factory.

Builds a rectangular grid of city blocks for initial iteration and testing.

Layout (block_size=100m, 3 columns × 3 rows):
  Intersections at every grid node.
  Horizontal streets run West→East.
  Vertical streets run South→North.
  All streets are bidirectional, 1 lane per direction, 50 km/h limit.
  Signalised intersections at every interior node.
  Spawn/despawn points at every border node.

Coordinate system: origin at bottom-left corner, meters.

         N
         |
  W ---- + ---- E
         |
         S

  (0,0) is SW corner; x increases East, y increases North.
"""
from __future__ import annotations

from src.simulation.enums import PavementCondition, TrafficLightState, VehicleType
from src.simulation.models.city import City, CityBounds
from src.simulation.models.geometry import Point2D
from src.simulation.models.intersection import Intersection
from src.simulation.models.spawn_point import SpawnPoint, DespawnPoint, SpawnKind
from src.simulation.models.street import Street
from src.simulation.models.traffic_light import TrafficLight, standard_phases


def build_grid_city(
    cols: int = 4,
    rows: int = 4,
    block_size: float = 100.0,      # meters per block
    max_speed_kmh: float = 50.0,
    green_ns: float = 30.0,         # green duration for N-S traffic
    green_ew: float = 30.0,         # green duration for E-W traffic
    yellow: float = 4.0,
    spawn_rate: float = 8.0,        # vehicles / minute per spawn point
) -> City:
    """
    Build a `cols × rows` block grid city.
    Node count: (cols+1) × (rows+1).
    """
    city = City(
        name=f"Grid {cols}×{rows}",
        bounds=CityBounds(
            min_x=0,
            min_y=0,
            max_x=cols * block_size,
            max_y=rows * block_size,
        ),
    )

    # ------------------------------------------------------------------
    # 1. Create intersections at every grid node
    # ------------------------------------------------------------------
    # intersections[col][row]
    nodes: list[list[Intersection]] = []
    for col in range(cols + 1):
        col_nodes: list[Intersection] = []
        for row in range(rows + 1):
            x = col * block_size
            y = row * block_size
            is_border = (col == 0 or col == cols or row == 0 or row == rows)
            node = Intersection(
                name=f"I({col},{row})",
                position=Point2D(x, y),
                stop_line_distance=3.0,
                # Border nodes are unsignalised entry/exit points
                # Interior nodes will get signals below
            )
            city.add_intersection(node)
            col_nodes.append(node)
        nodes.append(col_nodes)

    # ------------------------------------------------------------------
    # 2. Create streets
    # ------------------------------------------------------------------

    # Horizontal streets (West→East) — one per row, spanning each block
    for row in range(rows + 1):
        for col in range(cols):
            west_node = nodes[col][row]
            east_node = nodes[col + 1][row]
            street = Street(
                name=f"H({col},{row})→({col+1},{row})",
                nodes=[Point2D(west_node.position.x, west_node.position.y),
                       Point2D(east_node.position.x, east_node.position.y)],
                start_intersection_id=west_node.id,
                end_intersection_id=east_node.id,
                is_bidirectional=True,
                lanes_per_direction=1,
                max_speed_kmh=max_speed_kmh,
                pavement_condition=PavementCondition.GOOD,
                zone="urban_grid",
            )
            city.add_street(street)
            west_node.add_street(street.id)
            east_node.add_street(street.id)

    # Vertical streets (South→North)
    for col in range(cols + 1):
        for row in range(rows):
            south_node = nodes[col][row]
            north_node = nodes[col][row + 1]
            street = Street(
                name=f"V({col},{row})→({col},{row+1})",
                nodes=[Point2D(south_node.position.x, south_node.position.y),
                       Point2D(north_node.position.x, north_node.position.y)],
                start_intersection_id=south_node.id,
                end_intersection_id=north_node.id,
                is_bidirectional=True,
                lanes_per_direction=1,
                max_speed_kmh=max_speed_kmh,
                pavement_condition=PavementCondition.GOOD,
                zone="urban_grid",
            )
            city.add_street(street)
            south_node.add_street(street.id)
            north_node.add_street(street.id)

    # ------------------------------------------------------------------
    # 3. Add traffic lights at interior intersections
    #    Each interior node gets 4 lights (one per approach).
    #    N-S approaches: green_ns first; E-W approaches offset by green_ns+yellow.
    # ------------------------------------------------------------------
    red_ns = green_ew + yellow          # red for N-S = green+yellow of E-W
    red_ew = green_ns + yellow          # red for E-W = green+yellow of N-S

    for col in range(1, cols):
        for row in range(1, rows):
            node = nodes[col][row]
            offset_ew = green_ns + yellow  # E-W lights start after N-S green+yellow

            # N-S approaches (bearings ≈ 90° northbound / 270° southbound)
            for bearing_deg, label in [(90.0, "northbound"), (270.0, "southbound")]:
                light = TrafficLight(
                    position=node.position,
                    orientation_deg=bearing_deg,
                    intersection_id=node.id,
                    phases=standard_phases(
                        green_duration=green_ns,
                        yellow_duration=yellow,
                        red_duration=red_ns,
                    ),
                    phase_offset=0.0,
                )
                city.add_traffic_light(light)

            # E-W approaches (bearings ≈ 0° eastbound / 180° westbound)
            for bearing_deg, label in [(0.0, "eastbound"), (180.0, "westbound")]:
                light = TrafficLight(
                    position=node.position,
                    orientation_deg=bearing_deg,
                    intersection_id=node.id,
                    phases=standard_phases(
                        green_duration=green_ew,
                        yellow_duration=yellow,
                        red_duration=red_ew,
                    ),
                    phase_offset=offset_ew,
                )
                city.add_traffic_light(light)

            # Apply offsets so lights start at the right phase
            for lid in node.traffic_light_ids:
                city.traffic_lights[lid].apply_offset()

    # ------------------------------------------------------------------
    # 4. Spawn / Despawn points
    #
    # Two kinds:
    #   a) CITY_BORDER — at border nodes (traffic entering/exiting the map)
    #   b) RESIDENTIAL — mid-block on interior streets (houses, garages)
    # ------------------------------------------------------------------

    # 4a. Border spawn/despawn
    border_nodes: list[tuple[Intersection, float]] = []  # (node, entry_heading)

    # South edge: heading = 90° (northbound entry)
    for col in range(cols + 1):
        border_nodes.append((nodes[col][0], 90.0))

    # North edge: heading = 270° (southbound entry)
    for col in range(cols + 1):
        border_nodes.append((nodes[col][rows], 270.0))

    # West edge: heading = 0° (eastbound entry)
    for row in range(1, rows):
        border_nodes.append((nodes[0][row], 0.0))

    # East edge: heading = 180° (westbound entry)
    for row in range(1, rows):
        border_nodes.append((nodes[cols][row], 180.0))

    for node, heading in border_nodes:
        street_id = node.street_ids[0] if node.street_ids else ""

        sp = SpawnPoint(
            name=f"Spawn@{node.name}",
            position=Point2D(node.position.x, node.position.y),
            kind=SpawnKind.CITY_BORDER,
            street_id=street_id,
            street_progress=0.0,
            heading=heading,
            spawn_rate=spawn_rate,
        )
        city.add_spawn_point(sp)

        dp = DespawnPoint(
            name=f"Despawn@{node.name}",
            position=Point2D(node.position.x, node.position.y),
            kind=SpawnKind.CITY_BORDER,
            street_id=street_id,
            street_progress=1.0,
            capture_radius=8.0,
        )
        city.add_despawn_point(dp)

    # 4b. Mid-block residential spawn/despawn on some interior streets.
    #     These simulate houses/garages: low spawn rate, mostly cars.
    import random as _random
    rng = _random.Random(42)  # deterministic layout

    residential_weights = {
        VehicleType.CAR.value:        0.85,
        VehicleType.MOTORCYCLE.value: 0.10,
        VehicleType.VAN.value:        0.05,
    }

    all_streets = list(city.streets.values())
    # Pick ~30 % of interior horizontal streets for residential spawn points
    interior_streets = [
        s for s in all_streets
        if s.zone == "urban_grid"
    ]
    chosen = rng.sample(interior_streets, k=max(1, len(interior_streets) // 3))

    for street in chosen:
        # Place a spawn point at a random position between 20%–80% of the street
        progress = rng.uniform(0.2, 0.8)
        pos, heading = street.point_at_distance(progress * street.length)

        sp = SpawnPoint(
            name=f"House@{street.name}:{progress:.0%}",
            position=pos,
            kind=SpawnKind.RESIDENTIAL,
            street_id=street.id,
            street_progress=progress,
            heading=heading,
            spawn_rate=rng.uniform(0.5, 2.0),  # much lower than border
            vehicle_type_weights=residential_weights,
        )
        city.add_spawn_point(sp)

        dp = DespawnPoint(
            name=f"Home@{street.name}:{progress:.0%}",
            position=pos,
            kind=SpawnKind.RESIDENTIAL,
            street_id=street.id,
            street_progress=progress,
            capture_radius=5.0,
        )
        city.add_despawn_point(dp)

    return city
