# offboard_failsafe.py (ROS 2 adapter for PX4)
import rclpy
from rclpy.node import Node
from px4_msgs.msg import VehicleCommand
from safety_monitor import SafetyMonitor, SafetyLimits, SafetyHysteresis
import numpy as np
from typing import Optional

PX4_CUSTOM_MAIN_MODE_MANUAL   = 1.0
PX4_CUSTOM_MAIN_MODE_ACRO     = 2.0
PX4_CUSTOM_MAIN_MODE_ALTCTL   = 3.0
PX4_CUSTOM_MAIN_MODE_POSCTL   = 4.0
PX4_CUSTOM_MAIN_MODE_AUTO     = 5.0
PX4_CUSTOM_MAIN_MODE_OFFBOARD = 6.0

class OffboardFailsafe:
    """
    Glue class you can embed in your node.
    Call:
      - note_odom_stamp() on each odom,
      - check_and_maybe_disengage(...) each control tick before publishing u.
    """
    def __init__(self, node: Node,
                 limits: Optional[SafetyLimits]=None,
                 hyst: Optional[SafetyHysteresis]=None,
                 cmd_bounds: Optional[tuple[float,float]] = None):
        self.node = node
        if limits is None: limits = SafetyLimits()
        if cmd_bounds is not None:
            limits.cmd_bounds = cmd_bounds
        self.mon = SafetyMonitor(limits, hyst or SafetyHysteresis())
        self.cmd_pub = node.create_publisher(VehicleCommand, "/fmu/in/vehicle_command", 10)
        self._last_reason: Optional[str] = None

    def note_odom_stamp(self, stamp_us: int):
        self.mon.note_odom_stamp(stamp_us * 1e-6)

    def _send_mode_posctl(self):
        now = int(self.node.get_clock().now().nanoseconds / 1000)
        msg = VehicleCommand()
        msg.timestamp = now
        msg.param1 = 1.0                          # base mode (MAV_MODE_FLAG_CUSTOM_MODE_ENABLED)
        msg.param2 = PX4_CUSTOM_MAIN_MODE_POSCTL  # switch to Position mode
        msg.command = VehicleCommand.VEHICLE_CMD_DO_SET_MODE
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self.cmd_pub.publish(msg)
        self.node.get_logger().warn("Failsafe: switching to PX4 Position mode (POSCTL)")

    def _disarm(self):
        now = int(self.node.get_clock().now().nanoseconds / 1000)
        msg = VehicleCommand()
        msg.timestamp = now
        msg.param1 = 0.0
        msg.command = VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self.cmd_pub.publish(msg)
        self.node.get_logger().warn("Failsafe: disarm sent")

    def check_and_maybe_disengage(
        self,
        t_now_s: float,
        rpy_rad: np.ndarray,
        omega_rad_s: np.ndarray,
        pos_m: np.ndarray,
        vel_mps: np.ndarray,
        p_ref_m: np.ndarray,
        v_ref_mps: np.ndarray,
        u_cmd: Optional[np.ndarray] = None,
        disarm_on_trip: bool = False
    ) -> bool:
        """
        Returns True if it's safe to remain in Offboard; False if it requested POSCTL.
        """
        safe, reason = self.mon.evaluate(t_now_s, rpy_rad, omega_rad_s, pos_m, vel_mps, p_ref_m, v_ref_mps, u_cmd)
        if not safe:
            if reason and reason != self._last_reason:
                self.node.get_logger().error(f"Failsafe trip: {reason}")
                self._last_reason = reason
            self._send_mode_posctl()
            if disarm_on_trip:
                self._disarm()
            return False
        return True
