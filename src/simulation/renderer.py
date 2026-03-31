"""
Pygame-based renderer for the traffic simulation.

Draws streets, intersections, traffic lights, and vehicles in real time.
Includes a simple HUD with stats.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pygame

from src.simulation.config import SimulationConfig
from src.simulation.enums import TrafficLightState, VehicleType
from src.simulation.models.city import City
from src.simulation.models.geometry import Point2D

if TYPE_CHECKING:
    from src.simulation.simulator import Simulator


# Vehicle colors by type
VEHICLE_COLORS: dict[VehicleType, tuple[int, int, int]] = {
    VehicleType.CAR:        (79, 195, 247),   # light blue
    VehicleType.MOTORCYCLE: (255, 241, 118),   # yellow
    VehicleType.VAN:        (129, 199, 132),   # green
    VehicleType.BUS:        (255, 138, 101),   # orange
    VehicleType.TRUCK:      (229, 115, 115),   # red
}

LIGHT_COLORS: dict[TrafficLightState, tuple[int, int, int]] = {
    TrafficLightState.GREEN:           (76, 175, 80),
    TrafficLightState.YELLOW:          (255, 235, 59),
    TrafficLightState.RED:             (244, 67, 54),
    TrafficLightState.FLASHING_RED:    (244, 67, 54),
    TrafficLightState.FLASHING_YELLOW: (255, 235, 59),
    TrafficLightState.OFF:             (80, 80, 80),
}


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

        # Map area (leave margin for HUD on the right)
        self.map_margin = 40
        self.hud_width = 200
        self.map_w = config.window_width - self.hud_width - self.map_margin * 2
        self.map_h = config.window_height - self.map_margin * 2

    # ------------------------------------------------------------------
    # Coordinate transform
    # ------------------------------------------------------------------

    def world_to_screen(self, point: Point2D, city: City) -> tuple[int, int]:
        b = city.bounds
        if b.width == 0 or b.height == 0:
            return (0, 0)

        # Uniform scaling (keep aspect ratio)
        scale_x = self.map_w / b.width
        scale_y = self.map_h / b.height
        scale = min(scale_x, scale_y)

        # Center the map
        offset_x = self.map_margin + (self.map_w - b.width * scale) / 2
        offset_y = self.map_margin + (self.map_h - b.height * scale) / 2

        sx = offset_x + (point.x - b.min_x) * scale
        sy = offset_y + (b.max_y - point.y) * scale  # flip Y
        return int(sx), int(sy)

    def world_scale(self, meters: float, city: City) -> float:
        b = city.bounds
        scale_x = self.map_w / b.width if b.width else 1
        scale_y = self.map_h / b.height if b.height else 1
        return meters * min(scale_x, scale_y)

    # ------------------------------------------------------------------
    # Main render
    # ------------------------------------------------------------------

    def render(self, sim: Simulator) -> None:
        city = sim.city
        cfg = self.config
        self.screen.fill(cfg.bg_color)

        self._draw_streets(city)
        self._draw_intersections(city)
        self._draw_traffic_lights(city)
        self._draw_vehicles(city)
        self._draw_hud(sim)

        pygame.display.flip()

    # ------------------------------------------------------------------
    # Drawing layers
    # ------------------------------------------------------------------

    def _draw_streets(self, city: City) -> None:
        for street in city.streets.values():
            if len(street.nodes) < 2:
                continue
            points = [self.world_to_screen(n, city) for n in street.nodes]
            pygame.draw.lines(
                self.screen,
                self.config.street_color,
                False,
                points,
                self.config.street_width_px,
            )

    def _draw_intersections(self, city: City) -> None:
        r = max(3, self.config.street_width_px // 2 + 2)
        for inter in city.intersections.values():
            pos = self.world_to_screen(inter.position, city)
            pygame.draw.circle(self.screen, self.config.intersection_color, pos, r)

    def _draw_traffic_lights(self, city: City) -> None:
        for light in city.traffic_lights.values():
            color = LIGHT_COLORS.get(light.current_state, (80, 80, 80))

            # Offset the light dot slightly in the approach direction
            offset_m = 6.0  # meters from intersection center
            rad = math.radians(light.orientation_deg)
            offset_point = Point2D(
                light.position.x + math.cos(rad) * offset_m,
                light.position.y + math.sin(rad) * offset_m,
            )
            pos = self.world_to_screen(offset_point, city)
            pygame.draw.circle(self.screen, color, pos, 4)

    def _draw_vehicles(self, city: City) -> None:
        for v in city.vehicles.values():
            color = VEHICLE_COLORS.get(v.vehicle_type, (200, 200, 200))
            pos = self.world_to_screen(v.position, city)

            # Draw as a small oriented rectangle
            length_px = max(4, int(self.world_scale(v.length, city)))
            width_px = max(2, int(self.world_scale(v.width, city)))

            # Use half-size for cleaner look
            hl = length_px / 2
            hw = width_px / 2

            rad = -math.radians(v.heading)  # screen Y is flipped
            cos_a = math.cos(rad)
            sin_a = math.sin(rad)

            # Rectangle corners (centered on position)
            corners = []
            for lx, ly in [(-hl, -hw), (hl, -hw), (hl, hw), (-hl, hw)]:
                rx = pos[0] + lx * cos_a - ly * sin_a
                ry = pos[1] + lx * sin_a + ly * cos_a
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
            (f"Tráfico: {cfg.traffic_level:.0%}", self.font, (180, 180, 200)),
            (f"Clima: {cfg.weather.value}", self.font, (180, 180, 200)),
            (f"Velocidad: {cfg.time_scale:.1f}x", self.font, (180, 180, 200)),
            ("", self.font, (160, 160, 180)),
            ("CONTROLES", self.font_big, (220, 220, 240)),
            ("", self.font, (160, 160, 180)),
            ("↑↓  Tráfico", self.font, (140, 140, 160)),
            ("+/- Velocidad", self.font, (140, 140, 160)),
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
        """Call once per frame. Returns dt in seconds."""
        return self.clock.tick(self.config.fps) / 1000.0

    def destroy(self) -> None:
        pygame.quit()
