#!/usr/bin/env python3
# hydro_mpc/visualization/state_plotter_node.py
# Realtime plotting of vehicle state, target state, and trajectory setpoints (6DoF)
#
# Conventions/Style follow the hydro_mpc nodes:
#  - Topics and params loaded via ParamLoader + package share config
#  - QoS choices mirror navigator/trajectory_publisher
#  - Uses px4_msgs.VehicleOdometry, geometry_msgs/PoseWithCovarianceStamped,
#    and px4_msgs.TrajectorySetpoint6dof ("commanded trajectory")
#
from __future__ import annotations
import os
import math
import time
from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from ament_index_python.packages import get_package_share_directory

from px4_msgs.msg import VehicleOdometry, TrajectorySetpoint6dof as Traj6
from geometry_msgs.msg import PoseWithCovarianceStamped

from hydro_mpc.utils.param_loader import ParamLoader
from hydro_mpc.utils.helper_functions import quat_to_eul

# Matplotlib backend: allow user to override; default to interactive if available
import matplotlib
if os.environ.get("MPLBACKEND") is None:
    try:
        import matplotlib.pyplot as plt  # noqa: F401 (probe)
    except Exception:
        os.environ["MPLBACKEND"] = "Agg"
import matplotlib.pyplot as plt


# ----------------- Helpers -----------------

def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi

class _Series:
    __slots__ = ("t","x","y","z","yaw")
    def __init__(self, t: float, x: float, y: float, z: float, yaw: float) -> None:
        self.t = float(t); self.x = float(x); self.y = float(y); self.z = float(z); self.yaw = float(yaw)


class StatePlotterNode(Node):
    def __init__(self) -> None:
        super().__init__('state_plotter_node')

        # --------- Parameters (match hydro_mpc style) ---------
        package_dir = get_package_share_directory('hydro_mpc')
        self.declare_parameter('sitl_param_file', 'sitl_params.yaml')
        self.declare_parameter('window_sec', 30.0)
        self.declare_parameter('plot_rate_hz', 20.0)
        self.declare_parameter('save_png', False)
        self.declare_parameter('save_every_sec', 2.0)
        self.declare_parameter('fig_title', 'hydro_mpc • States & Setpoints')
        # UI/behavior
        self.declare_parameter('run_headless', False)
        self.declare_parameter('show_window', True)
        self.declare_parameter('autoraise', False)
        self.declare_parameter('out_dir', '')

        sitl_param_file = self.get_parameter('sitl_param_file').get_parameter_value().string_value
        self.win = float(self.get_parameter('window_sec').get_parameter_value().double_value)
        self.rate_hz = float(self.get_parameter('plot_rate_hz').get_parameter_value().double_value)
        self.save_png = bool(self.get_parameter('save_png').get_parameter_value().bool_value)
        self.save_every = float(self.get_parameter('save_every_sec').get_parameter_value().double_value)
        self.fig_title = self.get_parameter('fig_title').get_parameter_value().string_value

        self.run_headless = bool(self.get_parameter('run_headless').get_parameter_value().bool_value)
        self.show_window = bool(self.get_parameter('show_window').get_parameter_value().bool_value)
        self.autoraise = bool(self.get_parameter('autoraise').get_parameter_value().bool_value)
        self.out_dir = self.get_parameter('out_dir').get_parameter_value().string_value
        if self.out_dir:
            os.makedirs(self.out_dir, exist_ok=True)

        # Decide backend and import pyplot now (after params)
        if self.run_headless or not self.show_window:
            try:
                matplotlib.use('Agg', force=True)
            except Exception:
                pass
        global plt
        import matplotlib.pyplot as plt # late import so we can pick backend
        if not self.autoraise:
            try:
                matplotlib.rcParams['figure.raise_window'] = False
            except Exception:
                pass

        sitl_yaml_path = os.path.join(package_dir, 'config', 'sitl', sitl_param_file)
        sitl_yaml = ParamLoader(sitl_yaml_path)

        # Topics (mirror existing names used by navigator/trajectory publisher)
        odom_topic         = sitl_yaml.get_topic("odometry_topic")
        target_topic       = sitl_yaml.get_topic("target_state_topic")
        commanded_traj_top = sitl_yaml.get_topic("command_traj_topic")   # Traj6 from TrajectoryManager

        # --------- QoS (consistent with other nodes) ---------
        qos_odom = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        qos_target = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=4,
        )
        qos_traj = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # --------- State ---------
        self._t0 = time.time()
        self._last_png = 0.0
        self._buf_lock = None  # keep simple; deque ops are fast and we only touch from ROS thread/timer

        self.veh_buf: Deque[_Series] = deque()
        self.tgt_buf: Deque[_Series] = deque()
        self.sp_buf:  Deque[_Series] = deque()

        # --------- Subscriptions ---------
        self.create_subscription(VehicleOdometry, odom_topic, self._on_odom, qos_odom)
        self.create_subscription(PoseWithCovarianceStamped, target_topic, self._on_target, qos_target)
        self.create_subscription(Traj6, commanded_traj_top, self._on_traj6, qos_traj)

        # --------- Figure ---------
        self.fig = plt.figure(figsize=(15, 7), constrained_layout=True)
        if self.show_window:
            try:
                self.fig.canvas.manager.set_window_title(self.fig_title)
            except Exception:
                pass
        gs = self.fig.add_gridspec(2, 3)
        self.ax_xy   = self.fig.add_subplot(gs[:, 0])  # big XY
        self.ax_x    = self.fig.add_subplot(gs[0, 1])  # x vs t
        self.ax_y    = self.fig.add_subplot(gs[1, 1])  # y vs t
        self.ax_z    = self.fig.add_subplot(gs[0, 2])  # z vs t
        self.ax_yaw  = self.fig.add_subplot(gs[1, 2])  # yaw vs t

        (self.l_xy_veh,) = self.ax_xy.plot([], [], label='veh XY')
        (self.l_xy_tgt,) = self.ax_xy.plot([], [], linestyle='--', label='tgt XY')
        (self.l_xy_sp,)  = self.ax_xy.plot([], [], linestyle=':', label='sp XY')
        self.ax_xy.set_xlabel('Y [m]'); self.ax_xy.set_ylabel('X [m]'); self.ax_xy.set_title('Planar track (NED)')
        self.ax_xy.grid(True); self.ax_xy.legend(loc='best'); self.ax_xy.set_aspect('equal', adjustable='box')

        (self.l_x_veh,) = self.ax_x.plot([], [], label='x veh')
        (self.l_x_sp,)  = self.ax_x.plot([], [], linestyle=':', label='x sp')
        self.ax_x.set_ylabel('X [m]'); self.ax_x.grid(True); self.ax_x.legend(loc='best')

        (self.l_y_veh,) = self.ax_y.plot([], [], label='y veh')
        (self.l_y_sp,)  = self.ax_y.plot([], [], linestyle=':', label='y sp')
        self.ax_y.set_ylabel('Y [m]'); self.ax_y.grid(True); self.ax_y.legend(loc='best')
        self.ax_y.set_xlabel('t [s] rel'); 

        (self.l_z_veh,) = self.ax_z.plot([], [], label='z veh')
        (self.l_z_sp,)  = self.ax_z.plot([], [], linestyle=':', label='z sp')
        self.ax_z.set_ylabel('Z [m]'); self.ax_z.grid(True); self.ax_z.legend(loc='best')

        (self.l_yaw_veh,) = self.ax_yaw.plot([], [], label='yaw veh')
        (self.l_yaw_tgt,) = self.ax_yaw.plot([], [], linestyle='--', label='yaw tgt')
        (self.l_yaw_sp,)  = self.ax_yaw.plot([], [], linestyle=':', label='yaw sp')
        self.ax_yaw.set_xlabel('t [s] rel'); self.ax_yaw.set_ylabel('yaw [rad]'); self.ax_yaw.grid(True); self.ax_yaw.legend(loc='best')

        # Timer
        self.timer = self.create_timer(1.0 / max(1e-3, self.rate_hz), self._tick)
        if (not self.run_headless) and self.show_window:
            plt.ion()
        self.get_logger().info("StatePlotter ready.")

    # ------------- Callbacks -------------
    def _on_odom(self, msg: VehicleOdometry) -> None:
        t = time.time() - self._t0
        x, y, z = float(msg.position[0]), float(msg.position[1]), float(-msg.position[2])
        r, p, yaw = quat_to_eul(msg.q)
        self._append(self.veh_buf, _Series(t, x, y, z, float(yaw)))

    def _on_target(self, msg: PoseWithCovarianceStamped) -> None:
        t = time.time() - self._t0
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        # geometry_msgs uses (x,y,z,w); helper expects (w,x,y,z)
        yaw = quat_to_eul(np.array([q.w, q.x, q.y, q.z], dtype=float))[2]
        self._append(self.tgt_buf, _Series(t, float(p.x), float(p.y), float(-p.z), float(yaw)))

    def _on_traj6(self, msg: Traj6) -> None:
        t = time.time() - self._t0
        x, y, z = float(msg.position[0]), float(msg.position[1]), float(-msg.position[2])
        # yaw from quaternion [w,x,y,z]
        yaw = quat_to_eul(np.array([msg.quaternion[0], msg.quaternion[1], msg.quaternion[2], msg.quaternion[3]], dtype=float))[2]
        self._append(self.sp_buf, _Series(t, x, y, z, float(yaw)))

    # ------------- Internals -------------
    def _append(self, buf: Deque[_Series], s: _Series) -> None:
        buf.append(s)
        tmin = (time.time() - self._t0) - self.win
        while buf and buf[0].t < tmin:
            buf.popleft()

    def _xs(self, buf: Deque[_Series]):
        ts = [s.t for s in buf]
        xs = [s.x for s in buf] 
        ys = [s.y for s in buf] 
        zs = [s.z for s in buf]
        ysaw = [s.yaw for s in buf]
        return ts, xs, ys, zs, ysaw

    def _autoscale_xy(self, X, Y) -> None:
        if not X or not Y:
            return
        xmin, xmax = min(X), max(X)
        ymin, ymax = min(Y), max(Y)
        dx = max(1.0, 0.2 * (xmax - xmin))
        dy = max(1.0, 0.2 * (ymax - ymin))
        self.ax_xy.set_xlim(xmin - dx, xmax + dx)
        self.ax_xy.set_ylim(ymin - dy, ymax + dy)

    def _autoscale_axis(self, ax, t_all, y_all, pad_t=0.1, pad_y=0.2, min_pad_y=0.5, min_span_t=5.0):
        if not t_all:
            return
        tmin, tmax = min(t_all), max(t_all)
        dt = max(min_span_t, (tmax - tmin) * pad_t)
        ax.set_xlim(tmin - dt, tmax + dt)
        if y_all:
            ymin, ymax = min(y_all), max(y_all)
            dy = max(min_pad_y, (ymax - ymin) * pad_y)
            ax.set_ylim(ymin - dy, ymax + dy)

    # ------------- Plot tick -------------
    def _tick(self) -> None:
        
        vt, vx, vy, vz, vyaw = self._xs(self.veh_buf)
        tt, tx, ty, tz, tyaw = self._xs(self.tgt_buf)
        st, sx, sy, sz, syaw = self._xs(self.sp_buf)

        # XY
        self.l_xy_veh.set_data(vy, vx)
        self.l_xy_tgt.set_data(ty, tx)
        self.l_xy_sp.set_data(sy, sx)
        self._autoscale_xy([*vy, *ty, *sy], [*vx, *tx, *sx])

        # X vs t
        self.l_x_veh.set_data(vt, vx)
        self.l_x_sp.set_data(st, sx)
        self._autoscale_axis(self.ax_x, [*(vt or []), *(st or [])], [*(vx or []), *(sx or [])], min_span_t=5.0)

        # Y vs t
        self.l_y_veh.set_data(vt, vy)
        self.l_y_sp.set_data(st, sy)
        self._autoscale_axis(self.ax_y, [*(vt or []), *(st or [])], [*(vy or []), *(sy or [])], min_span_t=5.0)

        # Z vs t
        self.l_z_veh.set_data(vt, vz)
        self.l_z_sp.set_data(st, sz)
        self._autoscale_axis(self.ax_z, [*(vt or []), *(st or [])], [*(vz or []), *(sz or [])], min_span_t=5.0)

        # yaw vs t
        self.l_yaw_veh.set_data(vt, vyaw)
        self.l_yaw_tgt.set_data(tt, tyaw)
        self.l_yaw_sp.set_data(st, syaw)
        self._autoscale_axis(self.ax_yaw, [*(vt or []), *(tt or []), *(st or [])], [*(vyaw or []), *(tyaw or []), *(syaw or [])], min_span_t=5.0)

        # draw/flush & optional save
        self.fig.suptitle(self.fig_title)
        self.fig.canvas.draw_idle()
        try:
            self.fig.canvas.flush_events()
        except Exception:
            pass

        if self.save_png:
            now = time.time()
            if now - self._last_png >= self.save_every:
                self._last_png = now
                try:
                    self.fig.savefig(f"state_plot_{int(now)}.png", dpi=120)
                except Exception as e:
                    self.get_logger().warn(f"Failed to save png: {e}")


# ----------------- Main -----------------

def main(args=None) -> None:
    rclpy.init(args=args)
    node = StatePlotterNode()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            plt.pause(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        try:
            if (not node.run_headless) and node.show_window:
                plt.ioff(); plt.close('all')
        except Exception:
            pass


if __name__ == '__main__':
    main()
