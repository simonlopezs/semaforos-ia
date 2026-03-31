"""
Global simulation configuration.

traffic_level  → multiplier on all spawn rates (0.0 = empty, 1.0 = max)
weather        → placeholder for future: rain reduces friction, fog reduces visibility
stress_level   → placeholder for future: rush-hour aggressiveness modifier
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Weather(Enum):
    CLEAR = "clear"
    CLOUDY = "cloudy"
    RAIN = "rain"
    HEAVY_RAIN = "heavy_rain"
    FOG = "fog"


@dataclass
class SimulationConfig:
    # --- World knobs ---
    traffic_level: float = 0.5       # 0.0–1.0, multiplier on spawn rates
    weather: Weather = Weather.CLEAR  # no effect yet
    stress_level: float = 0.3        # no effect yet

    # --- Simulation ---
    time_scale: float = 1.0          # >1 = fast-forward
    fps: int = 60

    # --- Physics defaults (used when vehicle spec isn't consulted yet) ---
    default_acceleration: float = 2.5    # m/s²  comfortable accel
    default_deceleration: float = 3.5    # m/s²  comfortable braking
    stop_margin: float = 5.0             # meters before intersection stop line
    min_following_gap: float = 1.0       # meters bumper-to-bumper minimum

    # --- Rendering ---
    window_width: int = 1100
    window_height: int = 900
    bg_color: tuple[int, int, int] = (22, 22, 38)
    street_color: tuple[int, int, int] = (60, 60, 75)
    intersection_color: tuple[int, int, int] = (45, 45, 58)
    street_width_px: int = 8

    # --- Vehicle spawning ---
    max_vehicles: int = 200
