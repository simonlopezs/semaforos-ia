"""
Driver model.

The core concept is `impulsivity` (0.0–1.0): a single scalar that captures
how reckless/impatient a driver is. All other behavioral parameters can be
derived from it via a DriverProfile preset, or set manually for full control.

Impulsivity effects:
  - Higher impulsivity → shorter reaction time
  - Higher impulsivity → more likely to run yellow/red lights
  - Higher impulsivity → drives above speed limit
  - Higher impulsivity → follows closer (smaller gap acceptance)
  - Higher impulsivity → accelerates more aggressively
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum


class DriverProfile(Enum):
    """
    Preset profiles that map to impulsivity ranges and personality traits.
    These are starting points; individual drivers vary within ranges.
    """
    CAUTIOUS = "cautious"         # elderly, learner, anxious
    NORMAL = "normal"             # average commuter
    ASSERTIVE = "assertive"       # confident, slightly aggressive
    AGGRESSIVE = "aggressive"     # tailgates, speeds, cuts lanes
    RECKLESS = "reckless"         # dangerous — runs lights, races


# Impulsivity ranges per profile (min, max)
_PROFILE_IMPULSIVITY: dict[DriverProfile, tuple[float, float]] = {
    DriverProfile.CAUTIOUS:    (0.00, 0.20),
    DriverProfile.NORMAL:      (0.20, 0.45),
    DriverProfile.ASSERTIVE:   (0.45, 0.65),
    DriverProfile.AGGRESSIVE:  (0.65, 0.85),
    DriverProfile.RECKLESS:    (0.85, 1.00),
}


@dataclass
class Driver:
    """
    Behavioral model of a vehicle operator.

    All float attributes in [0.0, 1.0] unless noted otherwise.
    """

    # --- Core personality axis ---
    impulsivity: float = 0.3
    """
    Main behavioral scalar.
    0.0 = maximally cautious, 1.0 = maximally reckless.
    """

    # --- Derived / overridable attributes ---

    reaction_time: float = 1.0
    """
    Seconds between perceiving a stimulus and acting on it.
    Range: ~0.3s (alert/impulsive) → ~2.5s (distracted/cautious).
    """

    speed_compliance: float = 0.9
    """
    Probability (per road segment) of respecting the posted speed limit.
    1.0 = never exceeds limit; 0.0 = always ignores it.
    """

    speed_excess_factor: float = 1.0
    """
    When non-compliant, driver's target speed = limit × speed_excess_factor.
    Range: 1.0 (at limit) → 1.5+ (50 % over limit).
    """

    traffic_light_compliance: float = 0.95
    """
    Probability of stopping at a red light.
    Values below ~0.7 model habitual red-light runners.
    """

    yellow_light_behavior: float = 0.5
    """
    0.0 = always stops on yellow; 1.0 = always accelerates through yellow.
    Intermediate values are probabilistic.
    """

    gap_acceptance: float = 0.5
    """
    Minimum acceptable gap (relative). Lower = willing to merge/turn into
    smaller gaps. 0.0 = extremely aggressive gap acceptance.
    """

    following_distance_factor: float = 1.0
    """
    Multiplier on the minimum safe following distance.
    < 1.0 = tailgating; > 1.5 = very cautious spacing.
    """

    attention: float = 0.9
    """
    How attentive the driver is. Low attention increases reaction_time
    dynamically and may cause the vehicle to drift or brake late.
    Range: 0.3 (heavily distracted) → 1.0 (fully focused).
    """

    aggression: float = 0.3
    """
    Tendency to cut lanes, honk, prevent merges.
    Independent of impulsivity — a cautious driver can still be aggressive.
    """

    profile: DriverProfile = DriverProfile.NORMAL

    # --- Factory methods ---

    @classmethod
    def from_profile(cls, profile: DriverProfile, seed: int | None = None) -> Driver:
        """
        Create a driver whose attributes are sampled to match a preset profile.
        `seed` allows reproducible generation.
        """
        rng = random.Random(seed)
        lo, hi = _PROFILE_IMPULSIVITY[profile]
        impulsivity = rng.uniform(lo, hi)
        return cls._from_impulsivity(impulsivity, profile, rng)

    @classmethod
    def random(cls, seed: int | None = None) -> Driver:
        """
        Randomly sample a driver from a realistic population distribution.
        ~60 % normal, ~20 % cautious/assertive, ~15 % aggressive, ~5 % reckless.
        """
        rng = random.Random(seed)
        weights = [0.15, 0.45, 0.20, 0.15, 0.05]
        profile = rng.choices(list(DriverProfile), weights=weights, k=1)[0]
        return cls.from_profile(profile, seed=rng.randint(0, 2**31))

    @classmethod
    def _from_impulsivity(
        cls, impulsivity: float, profile: DriverProfile, rng: random.Random
    ) -> Driver:
        """Derive all behavioral attributes from an impulsivity score."""
        i = impulsivity  # shorthand

        def jitter(base: float, sigma: float = 0.05) -> float:
            return max(0.0, min(1.0, base + rng.gauss(0, sigma)))

        # reaction time: 0.4s (i=1) → 2.0s (i=0)
        reaction_time = 2.0 - 1.6 * i + rng.gauss(0, 0.1)
        reaction_time = max(0.3, min(2.5, reaction_time))

        # speed compliance: decreases with impulsivity
        speed_compliance = jitter(1.0 - 0.7 * i)

        # how much over limit when non-compliant: 1.0x (i=0) → 1.5x (i=1)
        speed_excess_factor = 1.0 + 0.5 * i + rng.gauss(0, 0.05)
        speed_excess_factor = max(1.0, speed_excess_factor)

        # red light compliance
        traffic_light_compliance = jitter(1.0 - 0.6 * i)

        # yellow: 0 (stop) → 1 (go) increases with impulsivity
        yellow_light_behavior = jitter(0.2 + 0.7 * i)

        # gap acceptance: lower = more aggressive merging
        gap_acceptance = jitter(1.0 - 0.8 * i)

        # following distance: less with more impulsivity
        following_distance_factor = max(0.3, 1.5 - 1.2 * i + rng.gauss(0, 0.1))

        # attention (loosely correlated with impulsivity but not directly)
        attention = jitter(0.95 - 0.4 * i)

        # aggression: increases with impulsivity but with more variance
        aggression = jitter(0.1 + 0.8 * i, sigma=0.1)

        return cls(
            impulsivity=i,
            reaction_time=reaction_time,
            speed_compliance=speed_compliance,
            speed_excess_factor=speed_excess_factor,
            traffic_light_compliance=traffic_light_compliance,
            yellow_light_behavior=yellow_light_behavior,
            gap_acceptance=gap_acceptance,
            following_distance_factor=following_distance_factor,
            attention=attention,
            aggression=aggression,
            profile=profile,
        )

    def effective_speed_limit(self, posted_limit_ms: float, rng: random.Random) -> float:
        """
        Return the speed this driver will actually target on a segment.
        posted_limit_ms is in m/s.
        """
        if rng.random() < self.speed_compliance:
            return posted_limit_ms
        return posted_limit_ms * self.speed_excess_factor

    def will_stop_at_red(self, rng: random.Random) -> bool:
        return rng.random() < self.traffic_light_compliance

    def will_run_yellow(self, rng: random.Random) -> bool:
        return rng.random() < self.yellow_light_behavior

    def __repr__(self) -> str:
        return (
            f"Driver(profile={self.profile.value}, "
            f"impulsivity={self.impulsivity:.2f}, "
            f"reaction={self.reaction_time:.2f}s)"
        )
