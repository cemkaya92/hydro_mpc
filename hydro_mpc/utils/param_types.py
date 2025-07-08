# param_types.py

from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class UAVParams:
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
class MPCParams:
    horizon: float
    N: int
    frequency: float
    Q: List[float]  # length should be NX (e.g., 12)
    R: List[float]  # length should be NU (e.g., 4)
