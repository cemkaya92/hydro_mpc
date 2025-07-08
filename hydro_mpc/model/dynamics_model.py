#!/usr/bin/env python3

import numpy as np
from casadi import SX, vertcat, horzcat, Function, diag, sin, cos, tan


# ==================== CasADi MPC Model ====================
def build_casadi_model(params):
    """
    Builds the CasADi symbolic model of the drone.
    The model now has 12 states and 4 control inputs (Thrust + Torques).
    """
    MASS = params.mass
    ARM_LENGTH = params.arm_length
    IX, IY, IZ = params.inertia
    GRAV = params.gravity

    # === States: 12 states ===
    # North - East - Down Coordinate Frame Convention is Used
    # pos: x, y, z
    # vel: vx, vy, vz
    # rpy: roll, pitch, yaw
    # omega: p, q, r (body-frame angular velocities)
    x_pos = SX.sym('x_pos', 3)
    x_vel = SX.sym('x_vel', 3)
    x_rpy = SX.sym('x_rpy', 3)  
    x_omega = SX.sym('x_omega', 3)
    x = vertcat(x_pos, x_vel, x_rpy, x_omega)

    # === Controls: 4 inputs ===
    # u0: Total Thrust (T)
    # u1: Body-x torque (tau_phi)
    # u2: Body-y torque (tau_theta)
    # u3: Body-z torque (tau_psi)
    u = SX.sym('u', 4)

    # Extract states for clarity
    roll, pitch, yaw = x_rpy[0], x_rpy[1], x_rpy[2]
    p, q, r = x_omega[0], x_omega[1], x_omega[2]

    # === Dynamics Equations ===
    # 1. Rotation matrix from body to world frame
    cφ, sφ = cos(roll), sin(roll)
    cθ, sθ = cos(pitch), sin(pitch)
    cψ, sψ = cos(yaw), sin(yaw)
    R = vertcat(
        horzcat(cθ*cψ, sφ*sθ*cψ - cφ*sψ, cφ*sθ*cψ + sφ*sψ),
        horzcat(cθ*sψ, sφ*sθ*sψ + cφ*cψ, cφ*sθ*sψ - sφ*cψ),
        horzcat(-sθ,   sφ*cθ,            cφ*cθ)
    )


    # 2. Translational Dynamics (in world frame)
    # Acceleration = gravity + rotated body thrust
    f_thrust = vertcat(0, 0, -u[0]) # Total thrust is along body z-axis
    accel = vertcat(0, 0, GRAV) + (R @ f_thrust) / MASS
    pos_dot = x_vel
    vel_dot = accel

    # 3. Rotational Dynamics (in body frame)
    # Euler's equations of motion
    omega_dot = vertcat(
        (u[1] - (IZ - IY) * q * r) / IX,
        (u[2] - (IX - IZ) * p * r) / IY,
        (u[3] - (IY - IX) * p * q) / IZ
    )

    # 4. Attitude Kinematics
    # Transformation from body rates (p,q,r) to Euler angle rates (d(rpy)/dt)
    # Using tan(pitch) can lead to singularity at +/- 90 degrees
    W_rpy = vertcat(
        horzcat(1, sφ*tan(pitch), cφ*tan(pitch)),
        horzcat(0, cφ, -sφ),
        horzcat(0, sφ/cθ, cφ/cθ)
    )
    rpy_dot = W_rpy @ x_omega

    # Full state derivative vector
    xdot = vertcat(pos_dot, vel_dot, rpy_dot, omega_dot)

    # CasADi function
    f = Function('f', [x, u], [xdot], ['x', 'u'], ['xdot'])
    return f, x.size1(), u.size1()

