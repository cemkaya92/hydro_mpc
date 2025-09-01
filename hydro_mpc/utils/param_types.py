# param_types.py

from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Union, List, Sequence

Repeat = Literal["none", "loop", "pingpong"]
MissionType = Literal[
    "line_to", "straight", "arc", "rounded_rectangle", "racetrack_capsule"
]

@dataclass
class VehicleParams:
    mass: float                         # Kg
    arm_length: float                   # m
    inertia: Tuple[float, float, float] # (Ix, Iy, Iz) Kg.m^2
    gravity: float                      # m/s^2
    num_of_arms: int
    moment_constant: float              # m
    thrust_constant: float              # N.s^2/rad^2
    max_rotor_speed: float              # rad/s
    omega_to_pwm_coefficient: Tuple[float, float, float] # (x_2, x_1, x_0)
    PWM_MIN: float
    PWM_MAX: float
    input_scaling: float
    zero_position_armed: float

@dataclass
class ControlParams:
    horizon: float
    N: int
    frequency: float
    Q: List[float]  # length should be NX (e.g., 12)
    R: List[float]  # length should be NU (e.g., 4)


# -------- MISSION RELATED DATA CLASSES ---------
# ---------------- Shared blocks ----------------
@dataclass
class StartPose:
    use_current: bool = True
    x: float = 0.0
    y: float = 0.0
    psi: float = 0.0  # radians

@dataclass
class Common:
    repeat: Repeat = "none"
    start: StartPose = StartPose()
    speed: Optional[float] = None  # m/s (may be unused by some types)

# ---------------- Variants ----------------
@dataclass
class LineTo:
    type: Literal["line_to"] = "line_to"
    common: Common = Common()
    goal_xypsi: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    duration: float = 0.0  # seconds

@dataclass
class Straight:
    type: Literal["straight"] = "straight"
    common: Common = Common()
    segment_distance: float = 0.0  # m, 0 => unbounded

@dataclass
class Arc:
    type: Literal["arc"] = "arc"
    common: Common = Common()
    radius: float = 1.0
    angle: Optional[float] = None   # radians; provide angle OR yaw_rate
    yaw_rate: Optional[float] = None # rad/s
    cw: bool = True

@dataclass
class RoundedRectangle:
    type: Literal["rounded_rectangle"] = "rounded_rectangle"
    common: Common = Common()
    width: float = 1.0
    height: float = 1.0
    corner_radius: float = 0.1
    cw: bool = True

@dataclass
class RacetrackCapsule:
    type: Literal["racetrack_capsule"] = "racetrack_capsule"
    common: Common = Common()
    straight_length: float = 1.0
    radius: float = 0.5
    cw: bool = True

Mission = Union[LineTo, Straight, Arc, RoundedRectangle, RacetrackCapsule]

@dataclass
class TakeoffParams:
    waypoint: Tuple[float, float, float] = (0.0, 0.0, -2.0)  # NED z up (negative)
    speed: float = 1.0                                       # m/s

@dataclass
class LoiterParams:
    center: Tuple[float, float, float] = (0.0, 0.0, -2.0)
    radius: float = 1.5
    omega: float = 0.5                                       # rad/s

@dataclass
class LandingParams:
    final_altitude: float = -0.1                             # NED z at touchdown
    trigger_radius: float = 0.6

@dataclass
class TargetParams:
    timeout: float = 0.6                                     # seconds

@dataclass
class TrajParams:
    v_max: Tuple[float, float, float] = (2.0, 2.0, 1.0)      # m/s
    a_max: Tuple[float, float, float] = (2.0, 2.0, 1.0)      # m/s^2
    segment_duration: float = 3.0                            # s
    yawrate_max: float = 1.2                                 # rad/s
    yawacc_max: float = 3.0                                  # rad/s^2

@dataclass
class MissionConfig:
    mission: Mission
    takeoff: TakeoffParams
    loiter: LoiterParams
    landing: LandingParams
    target: TargetParams
    traj: TrajParams

