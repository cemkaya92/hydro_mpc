import rclpy
from rclpy.node import Node
import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from px4_msgs.msg import TrajectorySetpoint6dof, TrajectorySetpoint
from std_msgs.msg import UInt8

from std_srvs.srv import Trigger

from hydro_mpc.navigation.state_machine import NavState

from hydro_mpc.utils.param_loader import ParamLoader

from hydro_mpc.utils.helper_functions import quat_to_eul

from ament_index_python.packages import get_package_share_directory
import os
import math




class TrajectoryPublisherNode(Node):
    def __init__(self):
        super().__init__('trajectory_publisher_node')

        package_dir = get_package_share_directory('hydro_mpc')
        
        # Declare param with default
        self.declare_parameter('sitl_param_file', 'sitl_param.yaml')
        self.declare_parameter('publish_rate_hz', 50.0)               
        self.declare_parameter('cmd_timeout_ms', 250)               # fail to neutral if no cmd within this window
        self.declare_parameter('prime_offboard_ms', 1800)


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
            depth=1,
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

        self._prime_until_us = 0
        
        # Timer
        self.timer = self.create_timer(1.0/self.rate, self._tick)


        self.create_service(Trigger, 'prime_offboard', self._srv_prime_offboard)


        self.get_logger().info("Trajectory Publisher with Offboard control started")



    # ---------- callbacks ----------
    def _on_nav_state(self, msg: UInt8):
        # 1=IDLE, 2=HOLD, 3=TAKEOFF, 4=LOITER, 5=FOLLOW_TARGET, 6=MISSION, 7=LANDING, 8=EMERGENCY, 9=MANUAL 

        raw = int(msg.data)
        prev = getattr(self, "nav_state", NavState.UNKNOWN)

        # Normalize to enum
        try:
            state = NavState(raw)
        except ValueError:
            self.get_logger().warn(f"Unknown nav_state {raw}; treating as UNKNOWN")
            state = NavState.UNKNOWN

        self.nav_state = state


        self.allow_commands = (state not in {NavState.MANUAL, NavState.EMERGENCY})


        if prev != state:
            self.get_logger().info(
                f"nav_state: {state.name} ({state.value}) | allow_commands={self.allow_commands}"
            )
            
            
    def _traj_sub_cb(self, msg: TrajectorySetpoint6dof):
        self.last_traj = msg
        self._last_cmd_time_us = self._now_us()


    def _srv_prime_offboard(self, req, res):
        prime_ms = int(self.get_parameter('prime_offboard_ms').get_parameter_value().integer_value)
        self._prime_until_us = self._now_us() + prime_ms * 1000
        res.success = True
        res.message = f"Primed for {prime_ms} ms"
        self.get_logger().info(f"Primed for {prime_ms} ms")
        return res


    # -------------- main loop -----------------
    def _tick(self):

        # watchdog: no fresh cmd -> neutral
        timeout_ms = int(self.get_parameter('cmd_timeout_ms').get_parameter_value().integer_value)
        is_timeout = (self._now_us() - self._last_cmd_time_us > timeout_ms * 1000)


        state = self.nav_state # already an enum now
        # self.get_logger().info(f"Just before idle, state: {state.name} ({state.value})")

        # Always stream a safe keepalive in IDLE/MANUAL so PX4 accepts Offboard later
        if state == NavState.IDLE:
            # self.get_logger().info("sending idle setpoints")
            self.pub_traj_sp.publish(self._safe_idle_setpoint())
            return
        
        if state == NavState.MANUAL:
            if self._now_us() < self._prime_until_us:
                # publish either last_traj or a safe idle setpoint
                if self.last_traj is not None and not is_timeout:
                    self.pub_traj_sp.publish(self._convert_to_px4_ts(self.last_traj))
                else:
                    self.pub_traj_sp.publish(self._safe_idle_setpoint())
            return

        if not self.allow_commands:
            # self.get_logger().info("do not allow commands → idle keepalive")
            # self.pub_traj_sp.publish(self._safe_idle_setpoint())
            return

        if self.last_traj is not None:
            traj_sp = self._convert_to_px4_ts(self.last_traj)
            self.pub_traj_sp.publish(traj_sp)




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
        _, _, dst.yaw  = quat_to_eul(src.quaternion)
        dst.yawspeed = float(src.angular_velocity[2])
            

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
        ts.position = [math.nan, math.nan, math.nan]
        ts.acceleration = [0.0, 0.0, 0.0]

        # Command zero velocity (safe hold)
        ts.velocity = [0.0, 0.0, 0.0]
        # ts.velocity = [math.nan, math.nan, math.nan]

        # Yaw/yawspeed: ignore yaw, zero yaw rate
        ts.yaw = math.nan
        ts.yawspeed = math.nan
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
