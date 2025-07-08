# File: hydro_mpc/control/control_allocator.py

import numpy as np

class ControlAllocator:
    @staticmethod
    def compute_allocation_matrices(num_of_arms=4, thrust_constant=5.84e-6, moment_constant=0.06, arm_length=0.25):
        if num_of_arms != 4:
            raise ValueError("Unknown UAV parameter num_of_arms. Only quadcopters are supported.")

        k_deg_to_rad = np.pi / 180.0
        kS = np.sin(45.0 * k_deg_to_rad)

        # Geometry matrix: unitless rotor layout
        rotor_layout = np.array([
            [ kS, -kS, -kS,  kS],  # roll
            [ kS,  kS, -kS, -kS],  # pitch
            [-1.0, 1.0, -1.0, 1.0], # yaw
            [1.0,  1.0,  1.0, 1.0]  # thrust
        ])

        # Gain matrix (diagonal)
        k_diag = np.diag([
            thrust_constant * arm_length,
            thrust_constant * arm_length,
            moment_constant * thrust_constant,
            thrust_constant
        ])

        # Control allocation matrix
        rotor_velocities_to_torques_and_thrust = k_diag @ rotor_layout
        torques_and_thrust_to_rotor_velocities = np.linalg.pinv(rotor_velocities_to_torques_and_thrust)

        return rotor_velocities_to_torques_and_thrust, torques_and_thrust_to_rotor_velocities

    @staticmethod
    def generate_mixing_matrices(arm_length, kf, km, angles_deg, spin_dirs):
        """
        Generates a normalized mixing matrix and its pseudoinverse.
        
        Returns:
            mixing_matrix          : shape (4,4), wrench components as rows
            mixing_matrix_pinv     : shape (4,4), throttle mapping for desired wrench
        """
        mixing_matrix = []
        for theta_deg, spin in zip(angles_deg, spin_dirs):
            theta = np.radians(theta_deg)
            roll = arm_length * np.cos(theta) * kf
            pitch = arm_length * np.sin(theta) * kf
            yaw = spin * km * kf
            thrust = kf
            mixing_matrix.append([roll, pitch, yaw, thrust])

        mixing_matrix = np.array(mixing_matrix)
        mixing_matrix_pinv = np.linalg.pinv(mixing_matrix)
        return mixing_matrix, mixing_matrix_pinv
