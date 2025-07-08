#!/usr/bin/env python3
"""
plot_states.py

Reads a ROS 2 bag containing PX4 messages, deserialises vehicle-odometry and
actuator-motor topics, and plots

    • position          (x, y, z)
    • linear velocity   (vx, vy, vz)
    • Euler angles      (roll, pitch, yaw)
    • angular velocity  (p, q, r)
    • motor commands    (u₁ … u₄)

over time.
"""

from pathlib import Path
from math     import atan2, asin

import numpy as np
import matplotlib.pyplot as plt

from rosbags.highlevel import AnyReader
from rosbags.typesys   import Stores, get_typestore, get_types_from_msg


# ───── User-configurable topic names ────────────────────────────────────────────
ODOM_TOPIC  = '/fmu/out/vehicle_odometry'
MOTOR_TOPIC = '/fmu/in/actuator_motors'

# ───── Helper: build a typestore that knows PX4 custom messages ────────────────
def build_typestore(msg_dir: Path):
    ts = get_typestore(Stores.ROS2_FOXY)
    custom_defs = {}
    for msg_file in msg_dir.glob('*.msg'):
        text = msg_file.read_text(encoding='utf-8')
        pkg  = msg_file.parent.parent.name                               # px4_msgs
        rel  = f'{msg_file.parent.name}/{msg_file.stem}'                 # msg/VehicleOdometry
        ros2_type = f'{pkg}/{rel}'
        custom_defs.update(get_types_from_msg(text, ros2_type))
    ts.register(custom_defs)
    return ts


# ───── Helper: quaternion → Euler (returns roll, pitch, yaw) ───────────────────
def quat_to_euler(q):
    """q = (w, x, y, z)  as in PX4 messages (note the order!)."""
    w, x, y, z = q  # PX4 stores w first
    # standard aerospace (x-forward, y-right, z-down) 3-2-1 sequence
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll  = atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = asin(np.clip(sinp, -1.0, 1.0))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw   = atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def main():
    # ─── Paths (adapt if your workspace differs) ───────────────────────────────
    px4_msg_dir = Path.home() / 'ros2_ws/src/px4_msgs/msg'
    bag_path    = Path.home() / 'ros2_ws/src/hydro_mpc/hydro_mpc/visualization/mpc_run_1749852230'

    ts = build_typestore(px4_msg_dir)

    # ─── Containers ────────────────────────────────────────────────────────────
    t_odom            = []
    pos, vel          = [], []
    euler, ang_vel    = [], []

    t_motors, motors  = [], []

    # ─── Read bag ──────────────────────────────────────────────────────────────
    with AnyReader([bag_path], default_typestore=ts) as reader:
        print("Available topics:", {c.topic for c in reader.connections})

        for c, stamp, raw in reader.messages():
            topic = c.topic
            msg   = reader.deserialize(raw, c.msgtype)
            t     = stamp * 1e-9  # ns → s

            if topic == ODOM_TOPIC:
                # PX4 VehicleOdometry fields
                t_odom.append(t)

                pos.append(msg.position)               # [x,y,z]
                vel.append(msg.velocity)               # [vx,vy,vz]

                # msg.q is [w,x,y,z] float32[4]
                euler.append(quat_to_euler(msg.q))

                ang_vel.append(msg.angular_velocity)   # [p,q,r]

            elif topic == MOTOR_TOPIC:
                t_motors.append(t)
                motors.append(msg.control)             # float32[4]

    # --- to numpy.
    t_odom   = np.asarray(t_odom)
    pos      = np.asarray(pos)
    vel      = np.asarray(vel)
    euler    = np.asarray(euler)
    ang_vel  = np.asarray(ang_vel)

    t_motors = np.asarray(t_motors)
    motors   = np.asarray(motors)

    # ─── Plot helpers ──────────────────────────────────────────────────────────
    def quick_plot(t, data, labels, title, ylab):
        plt.figure(figsize=(12, 5))
        plt.title(title, fontsize=17)
        for i, lab in enumerate(labels):
            plt.plot(t, data[:, i], label=lab, linewidth=2)
        plt.xlabel('Time [s]', fontsize=13)
        plt.ylabel(ylab, fontsize=13)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

    # ─── Plots ─────────────────────────────────────────────────────────────────
    if pos.size:
        quick_plot(t_odom, pos,   ['x', 'y', 'z'],            'Position',          'm')
        quick_plot(t_odom, vel,   ['vx', 'vy', 'vz'],         'Linear velocity',   'm/s')
        quick_plot(t_odom, euler, ['roll φ', 'pitch θ', 'yaw ψ'], 'Euler angles', 'rad')
        quick_plot(t_odom, ang_vel, ['p', 'q', 'r'],          'Body rates',        'rad/s')
    else:
        print(f'No odometry on {ODOM_TOPIC}')

    if motors.size:
        quick_plot(t_motors, motors, [f'M{i+1}' for i in range(4)],
                   'Actuator motor commands', 'normalised throttle')
    else:
        print(f'No motor commands on {MOTOR_TOPIC}')

    plt.show()


if __name__ == '__main__':
    main()
