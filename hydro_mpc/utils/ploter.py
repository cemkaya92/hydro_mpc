import os
import matplotlib.pyplot as plt

class Logger:
    def __init__(self):
        self.t = []
        self.x = []; self.y = []; self.z = []
        self.vx = []; self.vy = []; self.vz = []
        self.roll = []; self.pitch = []; self.yaw = []
        self.x_cmd = []; self.y_cmd = []; self.z_cmd = []
        self.vx_cmd = []; self.vy_cmd = []; self.vz_cmd = []
        self.roll_cmd = []; self.pitch_cmd = []; self.yaw_cmd = []
        self.thrust_cmd = []

    def log(self, t_sim, pos, vel, rpy, p_ref, v_ref, u_mpc):
        self.t.append(t_sim)

        self.x.append(pos[0]); self.y.append(pos[1]); self.z.append(pos[2])
        self.vx.append(vel[0]); self.vy.append(vel[1]); self.vz.append(vel[2])
        self.roll.append(rpy[0]); self.pitch.append(rpy[1]); self.yaw.append(rpy[2])

        self.x_cmd.append(p_ref[0]); self.y_cmd.append(p_ref[1]); self.z_cmd.append(p_ref[2])
        self.vx_cmd.append(v_ref[0]); self.vy_cmd.append(v_ref[1]); self.vz_cmd.append(v_ref[2])

        self.roll_cmd.append(u_mpc[1]); self.pitch_cmd.append(u_mpc[2]); self.yaw_cmd.append(u_mpc[3])
        self.thrust_cmd.append(u_mpc[0])

    def plot_logs(self):
        output_dir = os.path.expanduser("~/mpc_logs")
        os.makedirs(output_dir, exist_ok=True)
        t = self.t

        # Torque and Thrust
        fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
        axs[0].plot(t, self.roll_cmd, label='Roll Cmd'); axs[0].plot(t, self.roll, label='Roll'); axs[0].legend(); axs[0].grid(); axs[0].set_ylabel('Roll [rad]')
        axs[1].plot(t, self.pitch_cmd, label='Pitch Cmd'); axs[1].plot(t, self.pitch, label='Pitch'); axs[1].legend(); axs[1].grid(); axs[1].set_ylabel('Pitch [rad]')
        axs[2].plot(t, self.yaw_cmd, label='Yaw Cmd'); axs[2].plot(t, self.yaw, label='Yaw'); axs[2].legend(); axs[2].grid(); axs[2].set_ylabel('Yaw [rad]')
        axs[3].plot(t, self.thrust_cmd, label='Thrust Cmd'); axs[3].legend(); axs[3].grid(); axs[3].set_ylabel('Norm Thrust'); axs[3].set_xlabel('Time [s]')
        plt.suptitle("Torque and Thrust"); plt.tight_layout(); plt.savefig(os.path.join(output_dir, "thrust_torque_plot.png"))

        # Position
        fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        axs[0].plot(t, self.x_cmd, label='X Cmd'); axs[0].plot(t, self.x, label='X'); axs[0].legend(); axs[0].grid(); axs[0].set_ylabel('X [m]')
        axs[1].plot(t, self.y_cmd, label='Y Cmd'); axs[1].plot(t, self.y, label='Y'); axs[1].legend(); axs[1].grid(); axs[1].set_ylabel('Y [m]')
        axs[2].plot(t, self.z_cmd, label='Z Cmd'); axs[2].plot(t, self.z, label='Z'); axs[2].legend(); axs[2].grid(); axs[2].set_ylabel('Z [m]')
        plt.suptitle("Position"); plt.tight_layout(); plt.savefig(os.path.join(output_dir, "position_plot.png"))

        # Velocity
        fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        axs[0].plot(t, self.vx_cmd, label='VX Cmd'); axs[0].plot(t, self.vx, label='VX'); axs[0].legend(); axs[0].grid(); axs[0].set_ylabel('VX [m/s]')
        axs[1].plot(t, self.vy_cmd, label='VY Cmd'); axs[1].plot(t, self.vy, label='VY'); axs[1].legend(); axs[1].grid(); axs[1].set_ylabel('VY [m/s]')
        axs[2].plot(t, self.vz_cmd, label='VZ Cmd'); axs[2].plot(t, self.vz, label='VZ'); axs[2].legend(); axs[2].grid(); axs[2].set_ylabel('VZ [m/s]')
        plt.suptitle("Velocity"); plt.tight_layout(); plt.savefig(os.path.join(output_dir, "velocity_plot.png"))

        print(f"[Logger] Logs saved to: {output_dir}")
