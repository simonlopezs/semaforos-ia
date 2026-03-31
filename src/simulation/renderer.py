"""
Pygame-based renderer for the traffic simulation.

Streets are drawn as two parallel lanes (forward + backward).
Vehicles are drawn with their FRONT at the position point, body extending backward.
Vehicles are laterally offset to their lane (right side of the road).
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pygame

from src.simulation.config import SimulationConfig
from src.simulation.enums import TrafficLightState, VehicleType
from src.simulation.models.city import City
from src.simulation.models.geometry import Point2D
from src.simulation.router import Route

if TYPE_CHECKING:
    from src.simulation.simulator import Simulator


VEHICLE_COLORS: dict[VehicleType, tuple[int, int, int]] = {
    VehicleType.CAR:        (79, 195, 247),
    VehicleType.MOTORCYCLE: (255, 241, 118),
    VehicleType.VAN:        (129, 199, 132),
    VehicleType.BUS:        (255, 138, 101),
    VehicleType.TRUCK:      (229, 115, 115),
}

LIGHT_COLORS: dict[TrafficLightState, tuple[int, int, int]] = {
    TrafficLightState.GREEN:           (76, 175, 80),
    TrafficLightState.YELLOW:          (255, 235, 59),
    TrafficLightState.RED:             (244, 67, 54),
    TrafficLightState.FLASHING_RED:    (244, 67, 54),
    TrafficLightState.FLASHING_YELLOW: (255, 235, 59),
    TrafficLightState.OFF:             (80, 80, 80),
}

# Lane offset from centerline in meters (each lane is ~3.5m wide, offset by half)
LANE_OFFSET_M = 2.0


class Renderer:
    def __init__(self, config: SimulationConfig):
        pygame.init()
        self.config = config
        self.screen = pygame.display.set_mode(
            (config.window_width, config.window_height)
        )
        pygame.display.set_caption("Semáforos IA — Simulación de Tráfico")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 14)
        self.font_big = pygame.font.SysFont("monospace", 16, bold=True)

        self.map_margin = 40
        self.hud_width = 200
        self.map_w = config.window_width - self.hud_width - self.map_margin * 2
        self.map_h = config.window_height - self.map_margin * 2

        # Alpha surface for semi-transparent route lines
        self._route_surface = pygame.Surface(
            (config.window_width, config.window_height), pygame.SRCALPHA
        )

        # Toggle state
        self.show_routes = False

        # Cache scale once we know the city
        self._scale: float = 1.0
        self._offset_x: float = 0.0
        self._offset_y: float = 0.0
        self._bounds_set = False

    def _update_transform(self, city: City) -> None:
        b = city.bounds
        if b.width == 0 or b.height == 0:
            return
        scale_x = self.map_w / b.width
        scale_y = self.map_h / b.height
        self._scale = min(scale_x, scale_y)
        self._offset_x = self.map_margin + (self.map_w - b.width * self._scale) / 2
        self._offset_y = self.map_margin + (self.map_h - b.height * self._scale) / 2
        self._bounds_set = True

    def world_to_screen(self, point: Point2D, city: City) -> tuple[int, int]:
        if not self._bounds_set:
            self._update_transform(city)
        b = city.bounds
        sx = self._offset_x + (point.x - b.min_x) * self._scale
        sy = self._offset_y + (b.max_y - point.y) * self._scale
        return int(sx), int(sy)

    def world_scale(self, meters: float) -> float:
        return meters * self._scale

    # ------------------------------------------------------------------
    # Offset helpers
    # ------------------------------------------------------------------

    def _offset_point(
        self, point: Point2D, bearing_deg: float, lateral_m: float
    ) -> Point2D:
        """
        Offset a point perpendicular to the bearing direction.
        positive lateral_m = right side (when facing bearing direction).
        """
        # Right perpendicular = bearing - 90°
        perp_rad = math.radians(bearing_deg - 90)
        return Point2D(
            point.x + math.cos(perp_rad) * lateral_m,
            point.y + math.sin(perp_rad) * lateral_m,
        )

    def _offset_polyline(
        self, nodes: list[Point2D], lateral_m: float
    ) -> list[Point2D]:
        """Offset an entire polyline laterally. Uses segment bearings."""
        if len(nodes) < 2:
            return list(nodes)
        result: list[Point2D] = []
        for i in range(len(nodes)):
            if i == 0:
                bearing = nodes[0].bearing_to(nodes[1])
            elif i == len(nodes) - 1:
                bearing = nodes[-2].bearing_to(nodes[-1])
            else:
                # Average bearing of segments before and after this node
                b1 = nodes[i - 1].bearing_to(nodes[i])
                b2 = nodes[i].bearing_to(nodes[i + 1])
                # Average angles correctly
                bearing = math.degrees(math.atan2(
                    (math.sin(math.radians(b1)) + math.sin(math.radians(b2))) / 2,
                    (math.cos(math.radians(b1)) + math.cos(math.radians(b2))) / 2,
                ))
            result.append(self._offset_point(nodes[i], bearing, lateral_m))
        return result

    # ------------------------------------------------------------------
    # Main render
    # ------------------------------------------------------------------

    def render(self, sim: Simulator) -> None:
        city = sim.city
        self._update_transform(city)
        self.screen.fill(self.config.bg_color)

        self._draw_streets(city)
        self._draw_direction_arrows(city)
        self._draw_intersections(city)
        self._draw_traffic_lights(city)
        if self.show_routes:
            self._draw_routes(city)
        self._draw_vehicles(city)
        self._draw_hud(sim)

        pygame.display.flip()

    # ------------------------------------------------------------------
    # Drawing layers
    # ------------------------------------------------------------------

    def _draw_streets(self, city: City) -> None:
        lane_w = max(2, int(self.world_scale(3.0)))  # lane width in pixels
        divider_color = (40, 40, 55)

        for street in city.streets.values():
            if len(street.nodes) < 2:
                continue

            offset = LANE_OFFSET_M

            # Forward lane (right side of the road direction)
            fwd_nodes = self._offset_polyline(street.nodes, offset)
            fwd_pts = [self.world_to_screen(n, city) for n in fwd_nodes]
            if len(fwd_pts) >= 2:
                pygame.draw.lines(self.screen, self.config.street_color, False, fwd_pts, lane_w)

            if street.is_bidirectional:
                # Backward lane (left side)
                bwd_nodes = self._offset_polyline(street.nodes, -offset)
                bwd_pts = [self.world_to_screen(n, city) for n in bwd_nodes]
                if len(bwd_pts) >= 2:
                    pygame.draw.lines(self.screen, self.config.street_color, False, bwd_pts, lane_w)

                # Center divider line (thin, dashed look via thin line)
                center_pts = [self.world_to_screen(n, city) for n in street.nodes]
                if len(center_pts) >= 2:
                    pygame.draw.lines(self.screen, divider_color, False, center_pts, 1)

    def _draw_direction_arrows(self, city: City) -> None:
        """Draw small arrows alongside each lane at mid-block to show direction."""
        arrow_color = (90, 90, 110)
        arrow_len_px = 8
        arrow_head_px = 4

        for street in city.streets.values():
            if len(street.nodes) < 2:
                continue

            # Midpoint of the street centerline
            mid_dist = street.length / 2.0
            mid_pt, bearing = street.point_at_distance(mid_dist)

            # Offset further outside the lane (LANE_OFFSET_M + extra)
            arrow_offset_m = LANE_OFFSET_M + 3.5

            # Forward arrow (right side of road direction)
            fwd_world = self._offset_point(mid_pt, bearing, arrow_offset_m)
            fwd_center = self.world_to_screen(fwd_world, city)
            self._draw_arrow(fwd_center, bearing, arrow_len_px, arrow_head_px, arrow_color)

            if street.is_bidirectional:
                # Backward arrow (left side, opposite direction)
                bwd_world = self._offset_point(mid_pt, bearing, -arrow_offset_m)
                bwd_center = self.world_to_screen(bwd_world, city)
                self._draw_arrow(bwd_center, bearing + 180, arrow_len_px, arrow_head_px, arrow_color)

    def _draw_arrow(
        self,
        center: tuple[int, int],
        bearing_deg: float,
        length: int,
        head_size: int,
        color: tuple[int, int, int],
    ) -> None:
        """Draw a small directional arrow at screen coordinates."""
        # Arrow direction in screen space (Y flipped)
        rad = -math.radians(bearing_deg)
        dx = math.cos(rad)
        dy = math.sin(rad)

        # Tail and tip
        half = length / 2
        tx = center[0] - dx * half
        ty = center[1] - dy * half
        hx = center[0] + dx * half
        hy = center[1] + dy * half

        # Draw shaft
        pygame.draw.line(self.screen, color, (int(tx), int(ty)), (int(hx), int(hy)), 1)

        # Draw arrowhead (two small lines from tip)
        for angle_offset in (150, -150):
            a = rad + math.radians(angle_offset)
            ax = hx + math.cos(a) * head_size
            ay = hy + math.sin(a) * head_size
            pygame.draw.line(self.screen, color, (int(hx), int(hy)), (int(ax), int(ay)), 1)

    def _draw_intersections(self, city: City) -> None:
        # Draw intersection as a filled square sized by stop_line_distance
        for inter in city.intersections.values():
            half_size = max(4, int(self.world_scale(inter.stop_line_distance)))
            sx, sy = self.world_to_screen(inter.position, city)
            rect = pygame.Rect(sx - half_size, sy - half_size, half_size * 2, half_size * 2)
            pygame.draw.rect(self.screen, self.config.intersection_color, rect)

    def _draw_traffic_lights(self, city: City) -> None:
        for light in city.traffic_lights.values():
            color = LIGHT_COLORS.get(light.current_state, (80, 80, 80))

            # Position light at the lane edge near the intersection
            offset_m = LANE_OFFSET_M + 2.5  # slightly outside the lane
            rad = math.radians(light.orientation_deg)
            light_world = Point2D(
                light.position.x + math.cos(rad) * offset_m,
                light.position.y + math.sin(rad) * offset_m,
            )
            pos = self.world_to_screen(light_world, city)
            pygame.draw.circle(self.screen, color, pos, 3)

    def _draw_routes(self, city: City) -> None:
        """Draw a straight line from each vehicle to its destination + X marker."""
        self._route_surface.fill((0, 0, 0, 0))
        line_alpha = 50
        marker_alpha = 160

        for v in city.vehicles.values():
            route: Route | None = v.planned_route  # type: ignore[assignment]
            if not route or not route.destination_street_id:
                continue

            dest_street = city.get_street(route.destination_street_id)
            if not dest_street or dest_street.length <= 0:
                continue

            color = VEHICLE_COLORS.get(v.vehicle_type, (200, 200, 200))

            # Vehicle screen position
            lateral = LANE_OFFSET_M
            offset_pos = self._offset_point(v.position, v.heading, lateral)
            veh_screen = self.world_to_screen(offset_pos, city)

            # Destination screen position
            dest_pos, _ = dest_street.point_at_distance(
                route.destination_progress * dest_street.length
            )
            dest_screen = self.world_to_screen(dest_pos, city)

            # Straight line from vehicle to destination
            line_color = (color[0], color[1], color[2], line_alpha)
            pygame.draw.line(self._route_surface, line_color, veh_screen, dest_screen, 1)

            # X marker at destination
            mx, my = dest_screen
            m = 4
            marker_color = (color[0], color[1], color[2], marker_alpha)
            pygame.draw.line(self._route_surface, marker_color, (mx - m, my - m), (mx + m, my + m), 2)
            pygame.draw.line(self._route_surface, marker_color, (mx - m, my + m), (mx + m, my - m), 2)

        self.screen.blit(self._route_surface, (0, 0))

    def _draw_vehicles(self, city: City) -> None:
        for v in city.vehicles.values():
            color = VEHICLE_COLORS.get(v.vehicle_type, (200, 200, 200))

            # Both directions use their RIGHT side (right-hand traffic).
            # _offset_point with positive lateral gives right side of heading.
            lateral = LANE_OFFSET_M

            # Offset position perpendicular to heading
            offset_pos = self._offset_point(v.position, v.heading, lateral)
            sx, sy = self.world_to_screen(offset_pos, city)

            length_px = max(4, int(self.world_scale(v.length)))
            width_px = max(2, int(self.world_scale(v.width)))
            hw = width_px / 2

            # Vehicle heading in screen space (Y flipped)
            rad = -math.radians(v.heading)
            cos_a = math.cos(rad)
            sin_a = math.sin(rad)

            # Front is at (sx, sy); body extends BACKWARD by length_px
            # Local coords: front = (0,0), rear = (-length_px, 0)
            corners = []
            for lx, ly in [(0, -hw), (0, hw), (-length_px, hw), (-length_px, -hw)]:
                rx = sx + lx * cos_a - ly * sin_a
                ry = sy + lx * sin_a + ly * cos_a
                corners.append((rx, ry))

            pygame.draw.polygon(self.screen, color, corners)

    def _draw_hud(self, sim: Simulator) -> None:
        cfg = self.config
        x = cfg.window_width - self.hud_width + 10
        y = 20
        line_h = 20

        lines = [
            ("SIMULACIÓN", self.font_big, (220, 220, 240)),
            ("", self.font, (160, 160, 180)),
            (f"Tiempo: {sim.time:.1f}s", self.font, (180, 180, 200)),
            (f"Vehículos: {len(sim.city.vehicles)}", self.font, (180, 180, 200)),
            (f"Spawned: {sim.total_spawned}", self.font, (140, 140, 160)),
            (f"Despawned: {sim.total_despawned}", self.font, (140, 140, 160)),
            ("", self.font, (160, 160, 180)),
            (f"Tráfico: {cfg.traffic_level:.1f}", self.font, (180, 180, 200)),
            (f"Clima: {cfg.weather.value}", self.font, (180, 180, 200)),
            (f"Velocidad: {cfg.time_scale:.1f}x", self.font, (180, 180, 200)),
            ("", self.font, (160, 160, 180)),
            ("EFICIENCIA", self.font_big, (220, 220, 240)),
            ("", self.font, (160, 160, 180)),
            (f"Optimalidad: {sim.avg_optimality:.0%}", self.font, (180, 180, 200)),
            (f"Viajes comp: {sim.completed_trips}", self.font, (140, 140, 160)),
            ("", self.font, (160, 160, 180)),
            ("CONTROLES", self.font_big, (220, 220, 240)),
            ("", self.font, (160, 160, 180)),
            ("↑↓  Tráfico", self.font, (140, 140, 160)),
            ("+/- Velocidad", self.font, (140, 140, 160)),
            ("T   Rutas " + ("ON" if self.show_routes else "OFF"), self.font, (140, 140, 160)),
            ("P   Pausa", self.font, (140, 140, 160)),
            ("R   Reset", self.font, (140, 140, 160)),
            ("ESC Salir", self.font, (140, 140, 160)),
        ]

        for text, font, color in lines:
            if text:
                surf = font.render(text, True, color)
                self.screen.blit(surf, (x, y))
            y += line_h

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def tick(self) -> float:
        return self.clock.tick(self.config.fps) / 1000.0

    def destroy(self) -> None:
        pygame.quit()
