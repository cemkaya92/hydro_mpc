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
from std_srvs.srv import SetBool, Trigger

from ament_index_python.packages import get_package_share_directory

# Reuse your helpers (arm/offboard) and your ParamLoader + topics
from hydro_mpc.utils.vehicle_command_utils import (
    create_arm_command, create_disarm_command, create_offboard_mode_command, create_posctl_mode_command
)
from hydro_mpc.utils.param_loader import ParamLoader
from hydro_mpc.safety.safety_monitor import SafetyMonitor



# ----------------------- Offboard Manager Node -----------------------
class OffboardManagerNode(Node):
    """
    Central manager:
      - Publishes OffboardControlMode keepalive
      - Decides failsafe vs offboard based on state/refs/cmds
      - If safe: set OFFBOARD + ARM
      - If unsafe: switch to POSCTL (and optionally disarm) and LATCH (manual re-enable)
    """
    def __init__(self):
        super().__init__("offboard_manager")

        # params
        self.declare_parameter('vehicle_param_file', 'crazyflie_param.yaml')
        self.declare_parameter('sitl_param_file', 'sitl_param.yaml')
        self.declare_parameter('disarm_on_trip', False)
        self.declare_parameter('auto_reenter_after_trip', False)  # (default: NO auto re-entry)

        package_dir = get_package_share_directory('hydro_mpc')
        vehicle_param_file = self.get_parameter('vehicle_param_file').get_parameter_value().string_value
        sitl_param_file = self.get_parameter('sitl_param_file').get_parameter_value().string_value
        self.disarm_on_trip = bool(self.get_parameter('disarm_on_trip').get_parameter_value().bool_value)
        self.auto_reenter_after_trip = bool(self.get_parameter('auto_reenter_after_trip').get_parameter_value().bool_value)

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

        # services (to control the latch)  <-- NEW
        self.srv_enable = self.create_service(SetBool, 'offboard_manager/enable_offboard', self._srv_enable_offboard)
        self.srv_clear  = self.create_service(Trigger, 'offboard_manager/clear_trip', self._srv_clear_trip)


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
        self.have_target = True # HARD-CODED FOR TESTING =======================================================
        self.have_odom = False
        self.last_cmd: np.ndarray | None = None

        # control flags
        self.offboard_set = False
        self.armed_once = False

        # LATCH flags (block offboard re-entry after trip)  
        self.trip_latched = False         # set True on failsafe trip
        self.offboard_blocked = False     # global disable (also set on trip)

        # safety
        self.mon = SafetyMonitor()
        self.get_logger().info("OffboardManagerNode initialized")

        # timers
        self.keepalive_timer = self.create_timer(0.2, self._publish_offboard_keepalive)  # 5 Hz
        self.safety_timer = self.create_timer(0.1, self._safety_tick)                   # 10 Hz

    # ---------- services ----------
    def _srv_enable_offboard(self, req: SetBool.Request, res: SetBool.Response):
        if req.data:
            self.offboard_blocked = False
            self.trip_latched = False
            res.success = True
            res.message = "Offboard enabled; latch cleared."
            self.get_logger().info(res.message)
        else:
            self.offboard_blocked = True
            res.success = True
            res.message = "Offboard disabled."
            self.get_logger().warn(res.message)
        return res

    def _srv_clear_trip(self, req: Trigger.Request, res: Trigger.Response):
        self.trip_latched = False
        res.success = True
        res.message = "Trip latch cleared."
        self.get_logger().info(res.message)
        return res
    
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

        # Always safe to publish keepalive (it does not switch modes by itself)       
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

        # Only auto-set Offboard+Arm if not blocked/latched
        if (not self.offboard_set
            and not self.offboard_blocked
            and not self.trip_latched
            and self.have_odom
            and self.have_target):
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

            return

        # Unsafe -> switch to POSCTL and (optionally) disarm; then LATCH
        now_us = int(self.get_clock().now().nanoseconds / 1000)
        self._set_posctl_mode(now_us)
        if self.disarm_on_trip:
            self._send_disarm(now_us)
        self.offboard_set = False  # require re-enable

        # Latch and block re-entry unless user explicitly enables
        self.trip_latched = True
        if not self.auto_reenter_after_trip:
            self.offboard_blocked = True

        if reason:
            self.get_logger().error(f"Failsafe trip: {reason} (Offboard re-entry blocked; call /offboard_manager/enable_offboard True to re-enable)")


    # ---------- mode/arm helpers ----------
    def _set_offboard_and_arm(self, now_us: int):

        self.get_logger().info("sending _set_offboard_and_arm.")
        if self.offboard_set or self.offboard_blocked or self.trip_latched:
            return
        
        self.cmd_pub.publish(create_offboard_mode_command(now_us, self.sys_id))
        self.cmd_pub.publish(create_arm_command(now_us, self.sys_id))
        self.offboard_set = True
        self.armed_once = True
        self.get_logger().info("Sent OFFBOARD and ARM")

    def _set_posctl_mode(self, now_us: int):
        self.cmd_pub.publish(create_posctl_mode_command(now_us, self.sys_id))
        self.get_logger().warn("Switching to POSCTL (leaving Offboard)")


    def _send_disarm(self, now_us: int):
        self.cmd_pub.publish(create_disarm_command(now_us, self.sys_id))
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
