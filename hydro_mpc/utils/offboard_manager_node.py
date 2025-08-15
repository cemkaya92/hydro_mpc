# offboard_manager_node.py
from __future__ import annotations
import os
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from px4_msgs.msg import (
    VehicleCommand, OffboardControlMode, VehicleOdometry
)
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray

from ament_index_python.packages import get_package_share_directory

# Reuse your helpers (arm/offboard) and your ParamLoader + topics
from hydro_mpc.utils.vehicle_command_utils import (
    create_arm_command, create_offboard_mode_command
)
from hydro_mpc.utils.param_loader import ParamLoader

# ----------------------- small safety monitor -----------------------
def _bad(x) -> bool:
    if x is None:
        return True
    arr = np.asarray(x)
    return not np.all(np.isfinite(arr))

class SafetyLimits:
    def __init__(
        self,
        max_roll_pitch_deg: float = 35.0,
        max_rate_rad_s: float = 4.0,
        max_pos_err_m: float = 2.0,
        max_vel_err_mps: float = 2.0,
        odom_timeout_s: float = 0.25,
        cmd_bounds: tuple[float, float] | None = None,
    ):
        self.max_roll_pitch_deg = max_roll_pitch_deg
        self.max_rate_rad_s = max_rate_rad_s
        self.max_pos_err_m = max_pos_err_m
        self.max_vel_err_mps = max_vel_err_mps
        self.odom_timeout_s = odom_timeout_s
        self.cmd_bounds = cmd_bounds

class SafetyHysteresis:
    def __init__(self, trip_after_bad_cycles=2, clear_after_good_cycles=20):
        self.trip_after_bad_cycles = trip_after_bad_cycles
        self.clear_after_good_cycles = clear_after_good_cycles

class SafetyMonitor:
    def __init__(self, limits: SafetyLimits, hyst: SafetyHysteresis):
        self.lim = limits
        self.hyst = hyst
        self._bad_count = 0
        self._good_count = 0
        self._tripped = False
        self.last_odom_stamp_s: float | None = None

    def note_odom_stamp(self, t_s: float):
        self.last_odom_stamp_s = t_s

    def evaluate(
        self,
        t_now_s: float,
        rpy_rad: np.ndarray,
        omega_rad_s: np.ndarray,
        pos_m: np.ndarray, vel_mps: np.ndarray,
        p_ref_m: np.ndarray, v_ref_mps: np.ndarray,
        u_cmd: np.ndarray | None,
    ) -> tuple[bool, str | None]:
        reason = None
        if _bad([rpy_rad, omega_rad_s, pos_m, vel_mps, p_ref_m, v_ref_mps]):
            reason = "nan_or_inf_in_state_or_ref"

        if reason is None and u_cmd is not None:
            if _bad(u_cmd):
                reason = "nan_or_inf_in_command"
            elif self.lim.cmd_bounds:
                lo, hi = self.lim.cmd_bounds
                if np.any(u_cmd < lo) or np.any(u_cmd > hi):
                    reason = "command_out_of_bounds"

        if reason is None:
            rp_deg = np.abs(np.rad2deg(rpy_rad[:2]))
            if np.any(rp_deg > self.lim.max_roll_pitch_deg):
                reason = "excess_tilt"

        if reason is None and np.linalg.norm(omega_rad_s) > self.lim.max_rate_rad_s:
            reason = "excess_rate"

        if reason is None:
            if np.linalg.norm(p_ref_m - pos_m) > self.lim.max_pos_err_m:
                reason = "position_error_too_large"
            elif np.linalg.norm(v_ref_mps - vel_mps) > self.lim.max_vel_err_mps:
                reason = "velocity_error_too_large"

        if reason is None and self.last_odom_stamp_s is not None:
            if (t_now_s - self.last_odom_stamp_s) > self.lim.odom_timeout_s:
                reason = "odom_stale"

        if reason is None:
            self._good_count += 1
            self._bad_count = 0
            if self._tripped and self._good_count >= self.hyst.clear_after_good_cycles:
                self._tripped = False
            return (not self._tripped), None
        else:
            self._bad_count += 1
            self._good_count = 0
            if (not self._tripped) and self._bad_count >= self.hyst.trip_after_bad_cycles:
                self._tripped = True
            return (not self._tripped), reason

# ----------------------- Offboard Manager Node -----------------------
class OffboardManagerNode(Node):
    """
    Central manager:
      - Publishes OffboardControlMode keepalive
      - Decides failsafe vs offboard based on state/refs/cmds
      - If safe: set OFFBOARD + ARM
      - If unsafe: switch to POSCTL (and optionally disarm)
    """
    def __init__(self):
        super().__init__("offboard_manager")

        # params
        self.declare_parameter('vehicle_param_file', 'crazyflie_param.yaml')
        self.declare_parameter('sitl_param_file', 'sitl_param.yaml')
        self.declare_parameter('disarm_on_trip', False)

        package_dir = get_package_share_directory('hydro_mpc')
        vehicle_param_file = self.get_parameter('vehicle_param_file').get_parameter_value().string_value
        sitl_param_file = self.get_parameter('sitl_param_file').get_parameter_value().string_value
        self.disarm_on_trip = bool(self.get_parameter('disarm_on_trip').get_parameter_value().bool_value)

        sitl_yaml_path = os.path.join(package_dir, 'config', 'sitl', sitl_param_file)
        vehicle_yaml_path = os.path.join(package_dir, 'config', 'vehicle_parameters', vehicle_param_file)

        sitl_yaml = ParamLoader(sitl_yaml_path)
        # vehicle_yaml = ParamLoader(vehicle_yaml_path)  # not strictly needed here

        # topics
        odom_topic = sitl_yaml.get_topic("odometry_topic")
        status_topic = sitl_yaml.get_topic("status_topic")
        target_state_topic = sitl_yaml.get_topic("target_state_topic")
        control_cmd_topic = sitl_yaml.get_topic("control_command_topic")
        self.sys_id = sitl_yaml.get_nested(["sys_id"], 1)

        # pubs
        self.cmd_pub = self.create_publisher(VehicleCommand, sitl_yaml.get_topic("vehicle_command_topic"), 10)
        self.offboard_ctrl_pub = self.create_publisher(OffboardControlMode, sitl_yaml.get_topic("offboard_control_topic"), 10)

        # subs
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.create_subscription(VehicleOdometry, odom_topic, self._odom_cb, qos)
        self.create_subscription(Odometry, target_state_topic, self._target_cb, qos)
        self.create_subscription(Float32MultiArray, control_cmd_topic, self._cmd_cb, 10)

        # state
        self.t0 = None
        self.t_sim = 0.0
        self.px4_timestamp_us = 0
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.q = np.array([1.0, 0.0, 0.0, 0.0])  # w,x,y,z
        self.rpy = np.zeros(3)
        self.omega_body = np.zeros(3)
        self.target_pos = np.zeros(3)
        self.target_vel = np.zeros(3)
        self.have_target = True
        self.have_odom = False
        self.last_cmd: np.ndarray | None = None

        # control flags
        self.offboard_set = False
        self.armed_once = False
        self.failsafe_activated = False

        # safety
        limits = SafetyLimits(
            max_roll_pitch_deg=35.0,
            max_rate_rad_s=4.0,
            max_pos_err_m=2.0,
            max_vel_err_mps=2.0,
            odom_timeout_s=0.25,
            cmd_bounds=None,   # e.g. set to (-1.5, 1.5) if you want to bound torque/thrust inputs
        )
        self.mon = SafetyMonitor(limits, SafetyHysteresis())
        self.get_logger().info("OffboardManagerNode initialized")

        # timers
        self.keepalive_timer = self.create_timer(0.2, self._publish_offboard_keepalive)  # 5 Hz
        self.safety_timer = self.create_timer(0.1, self._safety_tick)                   # 10 Hz

    # ---------- callbacks ----------
    def _timing(self, stamp_us):
        t = stamp_us * 1e-6
        if self.t0 is None:
            self.t0 = t
        return t - self.t0


    def _odom_cb(self, msg: VehicleOdometry):
        self.px4_timestamp_us = msg.timestamp
        self.t_sim = self._timing(msg.timestamp)
        self.pos = np.array(msg.position, dtype=float)
        self.vel = np.array(msg.velocity, dtype=float)
        self.q = np.array([msg.q[0], msg.q[1], msg.q[2], msg.q[3]], dtype=float)  # w,x,y,z
        self.rpy = self._quat_to_eul(self.q)
        self.omega_body = np.array(msg.angular_velocity, dtype=float)
        self.mon.note_odom_stamp(msg.timestamp * 1e-6)
        if not self.have_odom:
            self.get_logger().info("First odom received.")
            self.have_odom = True

    def _target_cb(self, msg: Odometry):
        self.target_pos = np.array([msg.pose.pose.position.x,
                                    msg.pose.pose.position.y,
                                    msg.pose.pose.position.z], dtype=float)
        self.target_vel = np.array([msg.twist.twist.linear.x,
                                    msg.twist.twist.linear.y,
                                    msg.twist.twist.linear.z], dtype=float)
        self.have_target = True

    def _cmd_cb(self, msg: Float32MultiArray):
        self.last_cmd = np.asarray(msg.data, dtype=float)

    # ---------- timers ----------
    def _publish_offboard_keepalive(self):

        if not self.failsafe_activated:
                
            now_us = int(self.get_clock().now().nanoseconds / 1000)
            offboard = OffboardControlMode()
            offboard.timestamp = now_us
            offboard.position = False
            offboard.velocity = False
            offboard.acceleration = False
            offboard.attitude = False
            offboard.body_rate = False
            offboard.thrust_and_torque = True
            offboard.direct_actuator = False
            self.offboard_ctrl_pub.publish(offboard)

            # If we haven't set Offboard yet, and things look ok, do it here
            if not self.offboard_set and self.have_odom:
                # require target + (optionally) a first cmd sample to avoid arming on junk
                if self.have_target:
                    self._set_offboard_and_arm(now_us)

    def _safety_tick(self):
        if not (self.have_odom and self.have_target):
            return

        u_cmd = self.last_cmd if self.last_cmd is not None else None
        safe, reason = self.mon.evaluate(
            self.t_sim, self.rpy, self.omega_body,
            self.pos, self.vel,
            self.target_pos, self.target_vel,
            u_cmd
        )

        if safe:
            # if we lost Offboard due to a previous trip and things are good long enough,
            # you can choose to auto-reenter; here we keep it manual (offboard_set stays as is)
            
            # self.failsafe_activated = False
            return

        # Unsafe -> switch to POSCTL and optionally disarm
        self.failsafe_activated = True
        now_us = int(self.get_clock().now().nanoseconds / 1000)
        self._send_posctl_mode(now_us)
        if self.disarm_on_trip:
            self._send_disarm(now_us)
        self.offboard_set = False  # require re-enable

        if reason:
            self.get_logger().error(f"Failsafe trip: {reason}")

    # ---------- mode/arm helpers ----------
    def _set_offboard_and_arm(self, now_us: int):

        self.get_logger().info("sending _set_offboard_and_arm.")
        if self.offboard_set:
            return
        
        self.cmd_pub.publish(create_offboard_mode_command(now_us, self.sys_id))
        self.cmd_pub.publish(create_arm_command(now_us, self.sys_id))
        self.offboard_set = True
        self.armed_once = True
        self.get_logger().info("Sent OFFBOARD and ARM")

    def _send_posctl_mode(self, now_us: int):
        msg = VehicleCommand()
        msg.timestamp = now_us
        msg.command = VehicleCommand.VEHICLE_CMD_DO_SET_MODE
        msg.param1 = 1.0   # PX4 custom mode enabled
        msg.param2 = 4.0   # PX4_CUSTOM_MAIN_MODE_POSCTL
        msg.target_system = self.sys_id
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self.cmd_pub.publish(msg)
        self.get_logger().warn("Switching to POSCTL (leaving Offboard)")

    def _send_disarm(self, now_us: int):
        msg = VehicleCommand()
        msg.timestamp = now_us
        msg.param1 = 0.0
        msg.command = VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM
        msg.target_system = self.sys_id
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self.cmd_pub.publish(msg)
        self.get_logger().warn("Disarm sent")

    # ---------- utils ----------
    @staticmethod
    def _quat_to_eul(q_wxyz: np.ndarray) -> np.ndarray:
        # PX4 gives [w, x, y, z]
        from scipy.spatial.transform import Rotation as R
        r = R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
        return r.as_euler('ZYX', degrees=False)  # [yaw, pitch, roll]

def main(args=None):
    rclpy.init(args=args)
    node = OffboardManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
