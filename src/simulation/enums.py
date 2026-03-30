from enum import Enum, auto


class VehicleType(Enum):
    MOTORCYCLE = "motorcycle"
    CAR = "car"
    VAN = "van"
    BUS = "bus"
    TRUCK = "truck"


class VehicleState(Enum):
    PARKED = "parked"
    ACCELERATING = "accelerating"
    CRUISING = "cruising"
    DECELERATING = "decelerating"
    STOPPED = "stopped"           # stopped by traffic or light
    WAITING_LIGHT = "waiting_light"


class TrafficLightState(Enum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"
    FLASHING_RED = "flashing_red"    # all-way stop mode (fault/night)
    FLASHING_YELLOW = "flashing_yellow"  # caution / yield mode
    OFF = "off"


class PavementCondition(Enum):
    """Affects friction, speed factor and accident probability."""
    EXCELLENT = "excellent"   # new asphalt, smooth
    GOOD = "good"             # minor wear
    FAIR = "fair"             # some potholes / cracks
    POOR = "poor"             # frequent potholes
    DAMAGED = "damaged"       # severely broken


class LaneDirection(Enum):
    FORWARD = "forward"   # matches street orientation
    BACKWARD = "backward" # opposite to street orientation (bidirectional streets)


class CardinalDirection(Enum):
    """Used to express signal orientation and approach direction."""
    NORTH = 0
    NORTHEAST = 45
    EAST = 90
    SOUTHEAST = 135
    SOUTH = 180
    SOUTHWEST = 225
    WEST = 270
    NORTHWEST = 315
