# hydro_mpc/navigation/navigator_node.py
from __future__ import annotations
import os, math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from typing import Optional

from px4_msgs.msg import VehicleOdometry
from nav_msgs.msg import Odometry
from ament_index_python.packages import get_package_share_directory

from hydro_mpc.utils.param_loader import ParamLoader
from hydro_mpc.navigation.state_machine import NavState, NavStateMachine, NavEvents
from hydro_mpc.navigation.trajectory_manager import TrajectoryManager
from hydro_mpc.safety.rate_limiter import SafetyRateLimiter, RateLimitConfig
from hydro_mpc.navigation.trajectory_manager import TrajMsg

class NavigatorNode(Node):
    def __init__(self):
        super().__init__("navigator")

        # ------- params -------
        self.declare_parameter('sitl_param_file', 'sitl_params.yaml')
        self.declare_parameter('mission_param_file', 'mission.yaml')
        self.declare_parameter('command_traj_topic', '/navigator/trajectory_setpoint')
        self.declare_parameter('control_frequency', 50.0)
        self.declare_parameter('auto_start', True)

        package_dir = get_package_share_directory('hydro_mpc')

        sitl_param_file = self.get_parameter('sitl_param_file').get_parameter_value().string_value
        mission_param_file = self.get_parameter('mission_param_file').get_parameter_value().string_value

        sitl_yaml_path = os.path.join(package_dir, 'config', 'sitl', sitl_param_file)
        mission_yaml_path = os.path.join(package_dir, 'config', 'mission', mission_param_file)

        sitl_yaml = ParamLoader(sitl_yaml_path)
        mission_yaml = ParamLoader(mission_yaml_path)

        self.freq = float(self.get_parameter('control_frequency').get_parameter_value().double_value)
        self.Ts = 1.0 / self.freq
        self.auto_start = bool(self.get_parameter('auto_start').get_parameter_value().bool_value)

        # SITL topics
        traj_topic = sitl_yaml.get_topic("command_traj_topic")
        odom_topic = sitl_yaml.get_topic("odometry_topic")
        target_topic = sitl_yaml.get_topic("target_state_topic", default="/target/odom")

        # Mission config
        takeoff_wp = np.array(mission_yaml.get_nested(["takeoff","waypoint"], [0.0,0.0,-2.0]), float)
        self.takeoff_speed = float(mission_yaml.get_nested(["takeoff","speed"], 1.0))
        loiter_center = np.array(mission_yaml.get_nested(["loiter","center"], [0.0,0.0,-2.0]), float)
        loiter_radius = float(mission_yaml.get_nested(["loiter","radius"], 1.5))
        loiter_omega  = float(mission_yaml.get_nested(["loiter","omega"], 0.5))
        self.landing_z = float(mission_yaml.get_nested(["landing","final_altitude"], -0.1))
        self.landing_trigger_radius = float(mission_yaml.get_nested(["landing","trigger_radius"], 0.6))
        self.target_timeout = float(mission_yaml.get_nested(["target","timeout"], 0.6))
        a_max = np.array(mission_yaml.get_nested(["traj","a_max"], [2.0,2.0,1.0]), float)
        default_T = float(mission_yaml.get_nested(["traj","segment_duration"], 3.0))

        # ------- components -------
        self.sm = NavStateMachine()
        self.tm = TrajectoryManager(a_max, default_T, loiter_center, loiter_radius, loiter_omega)
        self.takeoff_wp = takeoff_wp

        limiter_cfg = RateLimitConfig(
            err_pos_cap=np.array([0.6,0.6,0.6]),
            err_vel_cap=np.array([1.0,1.0,0.6]),
            ref_v_cap=np.array([1.0,1.0,0.6]),
            ref_a_cap=np.array([2.0,2.0,1.0]),
        )
        self.limiter = SafetyRateLimiter(limiter_cfg)

        # ------- state -------
        self.t0 = None
        self.t_sim = 0.0
        self.odom_ready = False
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.rpy = np.zeros(3)

        self.target_pos: Optional[np.ndarray] = None
        self.target_vel: Optional[np.ndarray] = None
        self.t_last_target: Optional[float] = None

        # ------- IO -------
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(VehicleOdometry, odom_topic, self._odom_cb, qos)
        self.create_subscription(Odometry, target_topic, self._target_cb, qos)

        # publisher lives in TrajectoryManager.publish_traj(); create here:
        # (we import the class there to resolve msg type)
        
        self.traj_pub = self.create_publisher(TrajMsg, traj_topic, 10)

        # timers
        self.timer = self.create_timer(self.Ts, self._tick)
        self.get_logger().info("NavigatorNode ready.")

    # ---------- callbacks ----------
    def _odom_cb(self, msg: VehicleOdometry):
        self.t_sim = self._clock_to_t(msg.timestamp)
        self.pos = np.array(msg.position, float)
        self.vel = np.array(msg.velocity, float)
        self.rpy = self._quat_to_eul(np.array([msg.q[0],msg.q[1],msg.q[2],msg.q[3]], float))
        if not self.odom_ready:
            self.odom_ready = True
            self.get_logger().info("First odometry received.")

    def _target_cb(self, msg: Odometry):
        self.target_pos = np.array([msg.pose.pose.position.x,
                                    msg.pose.pose.position.y,
                                    msg.pose.pose.position.z], float)
        self.target_vel = np.array([msg.twist.twist.linear.x,
                                    msg.twist.twist.linear.y,
                                    msg.twist.twist.linear.z], float)
        self.t_last_target = self.t_sim

    # ---------- main loop ----------
    def _tick(self):
        if not self.odom_ready:
            return

        # events
        target_fresh = (self.t_last_target is not None) and ((self.t_sim - self.t_last_target) <= self.target_timeout)
        at_takeoff = (np.linalg.norm(self.pos - self.takeoff_wp) < 0.2) and (np.linalg.norm(self.vel) < 0.5)
        dist_to_target = np.linalg.norm(self.pos - self.target_pos) if (self.target_pos is not None) else 1e9
        landing_needed = (self.sm.state == NavState.FOLLOW_TARGET and
                          target_fresh and (dist_to_target < self.landing_trigger_radius) and
                          ((self.pos[2] - self.landing_z) > 0.15))
        landing_done = (self.sm.state == NavState.LANDING and
                        (self.pos[2] <= self.landing_z + 0.05) and (np.linalg.norm(self.vel) < 0.3))

        ev = NavEvents(
            have_odom=self.odom_ready,
            auto_start=self.auto_start,
            target_fresh=bool(target_fresh),
            at_takeoff_wp=bool(at_takeoff),
            landing_needed=bool(landing_needed),
            landing_done=bool(landing_done),
        )
        prev = self.sm.state
        state = self.sm.step(ev)

        # plan on transitions
        if prev != state:
            self.get_logger().info(f"State: {prev.name} -> {state.name}")
            if state == NavState.TAKEOFF:
                self.tm.plan_min_jerk(self.t_sim, self.pos, self.vel, self.takeoff_wp, np.zeros(3))
                self.limiter.reset()
            elif state == NavState.LOITER:
                self.limiter.reset()
            elif state == NavState.FOLLOW_TARGET and target_fresh:
                self.tm.plan_min_jerk(self.t_sim, self.pos, self.vel, self.target_pos, self.target_vel)
                self.limiter.reset()
            elif state == NavState.LANDING:
                xy = self.target_pos[:2] if (self.target_pos is not None) else self.pos[:2]
                p_goal = np.array([xy[0], xy[1], self.landing_z], float)
                self.tm.plan_min_jerk(self.t_sim, self.pos, self.vel, p_goal, np.zeros(3))
                self.limiter.reset()

        # select references per state
        if state == NavState.IDLE:
            p_ref, v_ref, a_ref = self.pos.copy(), np.zeros(3), np.zeros(3)
        elif state == NavState.TAKEOFF:
            p_ref, v_ref, a_ref = self._plan_or_hold()
        elif state == NavState.LOITER:
            p_ref, v_ref, a_ref = self.tm.loiter_ref(self.t_sim)
        elif state == NavState.FOLLOW_TARGET:
            p_ref, v_ref, a_ref = self._plan_or_hold()
            # replan periodically / when far
            if target_fresh and (dist_to_target > 0.8):
                self.tm.plan_min_jerk(self.t_sim, self.pos, self.vel, self.target_pos, self.target_vel)
        elif state == NavState.LANDING:
            p_ref, v_ref, a_ref = self._plan_or_hold()
        else:
            p_ref, v_ref, a_ref = self.pos.copy(), np.zeros(3), np.zeros(3)

        # apply limiter and publish
        p_cmd, v_cmd = self.limiter.limit(p_ref, v_ref, self.pos, self.vel, self.Ts)
        yaw = self._face_velocity_yaw(v_cmd)
        now_us = int(self.get_clock().now().nanoseconds / 1000)
        self.tm.publish_traj(self.traj_pub, now_us, p_cmd, v_cmd, a_ref, yaw=yaw)

    def _plan_or_hold(self):
        p, v, a = self.tm.get_plan_ref(self.t_sim)
        if p is None:  # plan finished
            return self.pos.copy(), np.zeros(3), np.zeros(3)
        return p, v, a

    # ---------- utils ----------
    def _clock_to_t(self, stamp_us: int) -> float:
        t = stamp_us * 1e-6
        if self.t0 is None:
            self.t0 = t
        return t - self.t0

    @staticmethod
    def _quat_to_eul(q_wxyz: np.ndarray) -> np.ndarray:
        from scipy.spatial.transform import Rotation as R
        r = R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
        return r.as_euler('ZYX', degrees=False)

    @staticmethod
    def _face_velocity_yaw(v: np.ndarray) -> float:
        vx, vy = float(v[0]), float(v[1])
        if abs(vx)+abs(vy) < 1e-3:
            return 0.0
        return math.atan2(vy, vx)

def main(args=None):
    rclpy.init(args=args)
    node = NavigatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
