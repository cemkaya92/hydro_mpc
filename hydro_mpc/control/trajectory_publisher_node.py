import rclpy
from rclpy.node import Node
import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from px4_msgs.msg import TrajectorySetpoint6dof, TrajectorySetpoint
from std_msgs.msg import UInt8

from hydro_mpc.navigation.state_machine import NavState

from hydro_mpc.utils.param_loader import ParamLoader

from ament_index_python.packages import get_package_share_directory
import os
import math




class TrajectoryPublisherNode(Node):
    def __init__(self):
        super().__init__('trajectory_publisher_node')

        package_dir = get_package_share_directory('hydro_mpc')
        
        # Declare param with default
        self.declare_parameter('sitl_param_file', 'sitl_param.yaml')
        self.declare_parameter('publish_rate_hz', 100.0)               
        self.declare_parameter('cmd_timeout_ms', 250)               # fail to neutral if no cmd within this window


        sitl_param_file = self.get_parameter('sitl_param_file').get_parameter_value().string_value

        self.rate = float(self.get_parameter('publish_rate_hz').value)

        sitl_yaml_path = os.path.join(package_dir, 'config', 'sitl', sitl_param_file)
        
        # Load parameters
        sitl_yaml = ParamLoader(sitl_yaml_path)

        # Topics
        trajectory_setpoint_topic = sitl_yaml.get_topic("trajectory_setpoint_topic")
        commanded_traj_topic = sitl_yaml.get_topic("command_traj_topic")
        nav_state_topic = sitl_yaml.get_topic("nav_state_topic")

        # QOS Options
        qos_traj = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        qos_nav_state = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,  # <-- latch last message
        )

        
        # sub
        self.sub_traj = self.create_subscription(
            TrajectorySetpoint6dof, commanded_traj_topic, self._traj_sub_cb, qos_traj)
        self.create_subscription(UInt8, nav_state_topic, self._on_nav_state, qos_nav_state)

        # pub
        self.pub_traj_sp  = self.create_publisher(TrajectorySetpoint, trajectory_setpoint_topic, 10)

        # State
        self.last_traj = None
        self.armed = False
        self.offboard_started = False

        self._last_cmd_time_us = 0

        self.allow_commands = False  # start in IDLE

        self.nav_state = NavState.IDLE
        
        # Timer
        self.timer = self.create_timer(1.0/self.rate, self._tick)
        self.get_logger().info("Trajectory Publisher with Offboard control started")


    # ---------- callbacks ----------
    def _on_nav_state(self, msg: UInt8):
        #is_idle = int(self.get_parameter('idle_nav_state').get_parameter_value().integer_value)
        self.nav_state = int(msg.data)
        # 1=IDLE, 2=HOLD, 3=TAKEOFF, 4=LOITER, 5=FOLLOW_TARGET, 6=MISSION, 7=LANDING, 8=EMERGENCY, 9=MANUAL 

        blocked = {8, 9}                       # EMERGENCY, MANUAL
        self.allow_commands = (self.nav_state not in blocked)

        # self.get_logger().info(f"nav_state: {self.nav_state} | allow_commands {self.allow_commands} ")

            
    def _traj_sub_cb(self, msg: TrajectorySetpoint6dof):
        self.last_traj = msg
        self._last_cmd_time_us = self._now_us()


    # -------------- main loop -----------------
    def _tick(self):

        # watchdog: no fresh cmd -> neutral
        timeout_ms = int(self.get_parameter('cmd_timeout_ms').get_parameter_value().integer_value)
        is_timeout = (self._now_us() - self._last_cmd_time_us > timeout_ms * 1000)

        self.get_logger().info(f"Just before idle, state: {self.nav_state}")
        # Always stream a safe keepalive in IDLE so PX4 accepts Offboard later
        if (self.nav_state == NavState.IDLE or self.nav_state == NavState.MANUAL):
            self.get_logger().info(f"sending idle setpoints")
            self.pub_traj_sp.publish(self._safe_idle_setpoint())
            return
        
        if not self.allow_commands:
            self.get_logger().info(f"do not allow commands")
            return
        
        # If we have a trajectory, convert & publish TrajectorySetpoint
        if self.last_traj is not None:
            traj_sp = self._convert_to_px4_ts(self.last_traj)
            self.pub_traj_sp.publish(traj_sp)
        else:
            # No upstream trajectory yet -> publish safe hold to keep Offboard alive
            self.pub_traj_sp.publish(self._safe_idle_setpoint())




    def _convert_to_px4_ts(self, src: TrajectorySetpoint6dof) -> TrajectorySetpoint:
        dst = TrajectorySetpoint()
        dst.timestamp = self._now_us()

        # Your message fields (assumes arrays length 3); adjust if named differently:
        p = np.array(src.position, dtype=float)
        v = np.array(src.velocity, dtype=float)
        a = np.array(src.acceleration, dtype=float)

        # PX4 expects NED: x forward, y right, z down
        # If your world is ENU, map: x=x, y= -y, z= -z
        dst.position = [ float(p[0]), float(p[1]), float(p[2]) ]
        dst.velocity = [ float(v[0]), float(v[1]), float(v[2]) ]
        dst.acceleration = [ float(a[0]), float(a[1]), float(a[2]) ]

        # yaw: radians. If your yaw is ENU, flip sign for NED

        dst.yaw = 0.0
        dst.yawspeed = 0.0
            

        # If you don’t fill fields, set them to NaN to “ignore” in PX4
        # import math; dst.jerk = [math.nan]*3

        return dst


    def _now_us(self) -> int:
        return int(self.get_clock().now().nanoseconds/1000)
    

    def _safe_idle_setpoint(self) -> TrajectorySetpoint:
        """Zero-velocity hold; all other fields ignored (NaN)."""
        ts = TrajectorySetpoint()
        ts.timestamp = self._now_us()
        
        # Ignore position/accel by setting NaN (PX4 treats NaN as 'unused'
        # ts.position = [math.nan, math.nan, math.nan]
        # ts.acceleration = [math.nan, math.nan, math.nan]
        ts.position = [0.0, 0.0, 0.0]
        ts.acceleration = [0.0, 0.0, 0.0]

        # Command zero velocity (safe hold)
        ts.velocity = [0.0, 0.0, 0.0]

        # Yaw/yawspeed: ignore yaw, zero yaw rate
        ts.yaw = 0.0
        ts.yawspeed = 0.0
        return ts


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryPublisherNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
