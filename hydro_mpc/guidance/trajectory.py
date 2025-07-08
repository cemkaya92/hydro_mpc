#!/usr/bin/env python3

import numpy as np

# ==================== Trajectory Generation ====================
def min_jerk(t, T, p0, pf):
    """
    Generates a minimum-jerk trajectory segment.
    Quintic polynomial interpolation for position and velocity.
    """
    # Ensure time is within the segment duration
    tau = np.clip(t/T, 0, 1)
    # Quintic polynomial for smooth position
    pos = p0 + (pf - p0) * (10*tau**3 - 15*tau**4 + 6*tau**5)
    # Derivative for smooth velocity
    vel = (pf - p0) * (30*tau**2 - 60*tau**3 + 30*tau**4) / T
    return pos, vel

def eval_traj_docking(t):
    """
    Generates a trajectory to take off and dock at a target.
    The trajectory consists of:
    1. Takeoff to a safe altitude.
    2. Traverse to a staging point above the target.
    3. A slow, final descent onto the docking target.
    """
    # === Define Trajectory Parameters ===
    CRUISING_ALTITUDE = -3.0
    DOCK_TARGET = np.array([2.0, 2.0, -0.30])

    T_takeoff = 4.0      # Time to climb to cruising altitude
    T_hover_initial = 3.0  # Time to stabilize after takeoff
    T_traverse = 6.0     # Time to fly to the staging point
    T_hover_staging = 4.0  # Time to stabilize before final descent
    T_descent = 5.0      # Time for slow final descent

    # === Define Key Points in the Trajectory ===
    P_ground = np.array([0., 0., 0.])
    P_takeoff_hover = np.array([0., 0., CRUISING_ALTITUDE])
    P_staging = np.array([DOCK_TARGET[0], DOCK_TARGET[1], CRUISING_ALTITUDE])
    P_dock = DOCK_TARGET
    ZERO_VEL = np.zeros(3)

    # === Timekeeping for each segment ===
    t_current = 0.0

    # --- 1. Takeoff (Ground -> Cruising Altitude) ---
    t_segment_end = t_current + T_takeoff
    if t <= t_segment_end:
        return min_jerk(t - t_current, T_takeoff, P_ground, P_takeoff_hover)
    t_current = t_segment_end

    # --- 2. Initial Hover ---
    t_segment_end = t_current + T_hover_initial
    if t <= t_segment_end:
        return P_takeoff_hover, ZERO_VEL
    t_current = t_segment_end

    # --- 3. Traverse (Takeoff Point -> Staging Point) ---
    t_segment_end = t_current + T_traverse
    if t <= t_segment_end:
        return min_jerk(t - t_current, T_traverse, P_takeoff_hover, P_staging)
    t_current = t_segment_end

    # --- 4. Hover at Staging Point (above target) ---
    t_segment_end = t_current + T_hover_staging
    if t <= t_segment_end:
        return P_staging, ZERO_VEL
    t_current = t_segment_end

    # --- 5. Final Descent (Staging Point -> Docking Target) ---
    t_segment_end = t_current + T_descent
    if t <= t_segment_end:
        return min_jerk(t - t_current, T_descent, P_staging, P_dock)
    t_current = t_segment_end

    # --- 6. Docked (Hold position at target) ---
    # After the sequence is complete, hover at the dock position indefinitely.
    return P_dock, ZERO_VEL

def eval_traj(t):
    """
    Generates a 1x1 meter square trajectory in the XY plane at Z=1m.
    Includes hover periods at each corner to ensure stability.
    """
    # === Define Trajectory Parameters ===
    SIDE_LENGTH = 1.0
    ALTITUDE = -3.0
    T_takeoff = 4.0  # Time for takeoff
    T_side = 5.0     # Time to fly one side of the square
    T_hover = 3.0    # Time to hover at each corner

    # === Define Corner Points ===
    P_start = np.array([0., 0., 0.])
    P1 = np.array([0., 0., ALTITUDE])
    P2 = np.array([SIDE_LENGTH, 0., ALTITUDE])
    P3 = np.array([SIDE_LENGTH, SIDE_LENGTH, ALTITUDE])
    P4 = np.array([0., SIDE_LENGTH, ALTITUDE])
    ZERO_VEL = np.zeros(3)

    # === Timekeeping for each segment ===
    t_current = 0.0

    # --- 1. Takeoff ---
    t_segment_end = t_current + T_takeoff
    if t <= t_segment_end:
        return min_jerk(t - t_current, T_takeoff, P_start, P1)
    t_current = t_segment_end

    # --- 2. Hover at Corner 1 ---
    t_segment_end = t_current + T_hover
    if t <= t_segment_end:
        return P1, ZERO_VEL
    t_current = t_segment_end

    # --- 3. Fly Side 1 (P1 -> P2) ---
    t_segment_end = t_current + T_side
    if t <= t_segment_end:
        return min_jerk(t - t_current, T_side, P1, P2)
    t_current = t_segment_end

    # --- 4. Hover at Corner 2 ---
    t_segment_end = t_current + T_hover
    if t <= t_segment_end:
        return P2, ZERO_VEL
    t_current = t_segment_end

    # --- 5. Fly Side 2 (P2 -> P3) ---
    t_segment_end = t_current + T_side
    if t <= t_segment_end:
        return min_jerk(t - t_current, T_side, P2, P3)
    t_current = t_segment_end

    # --- 6. Hover at Corner 3 ---
    t_segment_end = t_current + T_hover
    if t <= t_segment_end:
        return P3, ZERO_VEL
    t_current = t_segment_end

    # --- 7. Fly Side 3 (P3 -> P4) ---
    t_segment_end = t_current + T_side
    if t <= t_segment_end:
        return min_jerk(t - t_current, T_side, P3, P4)
    t_current = t_segment_end

    # --- 8. Hover at Corner 4 ---
    t_segment_end = t_current + T_hover
    if t <= t_segment_end:
        return P4, ZERO_VEL
    t_current = t_segment_end

    # --- 9. Fly Side 4 (P4 -> P1, return to start) ---
    t_segment_end = t_current + T_side
    if t <= t_segment_end:
        return min_jerk(t - t_current, T_side, P4, P1)
    t_current = t_segment_end

    # --- 10. Final Hover ---
    # After completing the square, just hover at the starting corner indefinitely
    return P1, ZERO_VEL