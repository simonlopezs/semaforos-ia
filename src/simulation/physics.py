"""
Braking physics model.

Calculates emergency braking distance considering:
  - Vehicle speed (m/s)
  - Vehicle mass (kg) — heavier vehicles need more force to stop
  - Tire-road friction coefficient (μ) — depends on weather/pavement
  - Road grade (future: uphill helps, downhill hurts)
  - Driver reaction time (seconds before brakes are applied)

Core formula (flat road):
  Braking force = μ × m × g
  Deceleration  = μ × g                    (mass cancels out!)
  Braking distance = v² / (2 × μ × g)
  Total stopping distance = v × t_reaction + v² / (2 × μ × g)

BUT mass still matters indirectly:
  - Heavier vehicles have longer mechanical response time (brake lag)
  - Tire deformation under load reduces effective μ slightly
  - Brake fade: sustained braking heats brakes, reducing effectiveness
    for heavier vehicles more than lighter ones.

We model this with a mass correction factor on μ.

Friction coefficients (μ) for rubber on asphalt:
  Dry excellent:   0.80
  Dry good:        0.70
  Dry fair:        0.60
  Wet (rain):      0.40 – 0.50
  Wet heavy rain:  0.30 – 0.40
  Ice/snow:        0.15 – 0.25   (future)
"""
from __future__ import annotations

from dataclasses import dataclass

from src.simulation.config import Weather
from src.simulation.enums import PavementCondition

# Gravitational acceleration (m/s²)
G = 9.81

# ---------------------------------------------------------------------------
# Friction lookup tables
# ---------------------------------------------------------------------------

# Base friction coefficient by weather
_WEATHER_FRICTION: dict[Weather, float] = {
    Weather.CLEAR:      0.75,
    Weather.CLOUDY:     0.73,   # slightly lower — dust/moisture
    Weather.RAIN:       0.45,
    Weather.HEAVY_RAIN: 0.33,
    Weather.FOG:        0.55,   # damp surface
}

# Pavement condition multiplier on friction (stacks with weather)
_PAVEMENT_FRICTION_MULT: dict[PavementCondition, float] = {
    PavementCondition.EXCELLENT: 1.00,
    PavementCondition.GOOD:      0.95,
    PavementCondition.FAIR:      0.85,
    PavementCondition.POOR:      0.70,
    PavementCondition.DAMAGED:   0.55,
}

# ---------------------------------------------------------------------------
# Mass correction
# ---------------------------------------------------------------------------

# Reference mass for "ideal" braking (a standard car)
_REFERENCE_MASS_KG = 1400.0

# Heavier vehicles suffer a penalty on effective μ due to:
#   - Brake lag (pneumatic brakes on trucks take ~0.3-0.5s to engage)
#   - Tire load deformation
#   - Brake fade under sustained use
# The correction: μ_eff = μ × (ref_mass / vehicle_mass) ^ exponent
# With exponent ~0.08, a 25-ton truck gets ~0.80× the friction of a car.
_MASS_EXPONENT = 0.08

# Mechanical brake lag by mass tier (seconds added to reaction time)
# Light vehicles (< 2 ton): 0.0s
# Medium (2-5 ton):         0.1s
# Heavy (5-15 ton):         0.25s
# Very heavy (> 15 ton):    0.4s
def _brake_lag(mass_kg: float) -> float:
    if mass_kg < 2000:
        return 0.0
    elif mass_kg < 5000:
        return 0.1
    elif mass_kg < 15000:
        return 0.25
    else:
        return 0.4


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class BrakingResult:
    """Result of an emergency braking calculation."""
    can_stop: bool               # True if vehicle can stop in the available distance
    braking_distance: float      # meters needed to decelerate to 0 (excluding reaction)
    reaction_distance: float     # meters traveled during reaction + brake lag
    total_stopping_distance: float  # braking + reaction
    effective_deceleration: float   # m/s² achievable under current conditions
    effective_friction: float       # μ_eff used in the calculation
    brake_lag: float                # seconds of mechanical brake lag


def compute_effective_friction(
    weather: Weather,
    pavement: PavementCondition,
    mass_kg: float,
) -> float:
    """
    Compute the effective tire-road friction coefficient.
    Considers weather, pavement quality, and vehicle mass.
    """
    mu_base = _WEATHER_FRICTION[weather]
    mu_pavement = mu_base * _PAVEMENT_FRICTION_MULT[pavement]

    # Mass correction: heavier → slightly lower effective friction
    mass_ratio = _REFERENCE_MASS_KG / max(mass_kg, 100.0)
    mu_eff = mu_pavement * (mass_ratio ** _MASS_EXPONENT)

    return max(0.1, min(1.0, mu_eff))


def emergency_braking(
    speed_ms: float,
    mass_kg: float,
    distance_available: float,
    weather: Weather,
    pavement: PavementCondition,
    reaction_time: float = 0.0,
) -> BrakingResult:
    """
    Calculate whether an emergency stop is possible within the available distance.

    Parameters
    ----------
    speed_ms           : current vehicle speed in m/s
    mass_kg            : vehicle mass in kg
    distance_available : meters available before the stop line
    weather            : current weather condition
    pavement           : road surface condition
    reaction_time      : driver's cognitive reaction time in seconds
                         (time from seeing yellow to deciding to brake)

    Returns
    -------
    BrakingResult with all computed values.
    """
    if speed_ms <= 0.01:
        return BrakingResult(
            can_stop=True,
            braking_distance=0.0,
            reaction_distance=0.0,
            total_stopping_distance=0.0,
            effective_deceleration=0.0,
            effective_friction=0.0,
            brake_lag=0.0,
        )

    mu_eff = compute_effective_friction(weather, pavement, mass_kg)
    lag = _brake_lag(mass_kg)

    # Deceleration: a = μ × g
    decel = mu_eff * G

    # Distance during reaction + mechanical brake lag (constant speed)
    total_reaction = reaction_time + lag
    reaction_dist = speed_ms * total_reaction

    # Kinematic braking distance: d = v² / (2a)
    braking_dist = (speed_ms ** 2) / (2.0 * decel)

    total = reaction_dist + braking_dist

    return BrakingResult(
        can_stop=total <= distance_available,
        braking_distance=braking_dist,
        reaction_distance=reaction_dist,
        total_stopping_distance=total,
        effective_deceleration=decel,
        effective_friction=mu_eff,
        brake_lag=lag,
    )
