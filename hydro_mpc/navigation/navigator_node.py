# hydro_mpc/navigation/navigator_node.py
from __future__ import annotations
import os, math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from typing import Optional

from px4_msgs.msg import VehicleOdometry, VehicleStatus, VehicleCommandAck
from nav_msgs.msg import Odometry
from std_srvs.srv import Trigger
from std_msgs.msg import UInt8
from custom_offboard_msgs.msg import SafetyTrip

from ament_index_python.packages import get_package_share_directory

from hydro_mpc.utils.param_loader import ParamLoader
from hydro_mpc.navigation.state_machine import NavState, NavStateMachine, NavEvents
from hydro_mpc.safety.rate_limiter import SafetyRateLimiter, RateLimitConfig
from hydro_mpc.guidance.trajectory_manager import TrajectoryManager, TrajMsg

from hydro_mpc.utils.param_types import (
    LineTo, Straight, Arc, RoundedRectangle, RacetrackCapsule
)


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
        status_topic = sitl_yaml.get_topic("status_topic")
        command_ack_topic = sitl_yaml.get_topic("vehicle_command_ack_topic")
        nav_state_topic = sitl_yaml.get_topic("nav_state_topic")

        # Mission config
        self.mission = mission_yaml.get_full_config()

        ok, msg = mission_yaml.validate_mission()
        self.mission_valid = bool(ok)
        if not ok:
            self.get_logger().warn(f"[Navigator] Mission invalid: {msg}")
        else:
            self.get_logger().info(f"[Navigator] Mission valid: {msg}")

        # ------- components -------
        self.sm = NavStateMachine()
        self.tm = TrajectoryManager(
            v_max=np.array(self.mission.traj.v_max),
            a_max=np.array(self.mission.traj.a_max),
            yawrate_max=float(self.mission.traj.yawrate_max),
            yawacc_max=float(self.mission.traj.yawacc_max),
            default_T=float(self.mission.traj.segment_duration),
        )


        limiter_cfg = RateLimitConfig(
            err_pos_cap=np.array([1.5,1.5,1.0]),
            err_vel_cap=np.array([3.0,3.0,1.0]),
            ref_v_cap=np.array([5.0,5.0,3.0]),
            ref_a_cap=np.array([5.0,5.0,3.0]),
        )
        self.limiter = SafetyRateLimiter(limiter_cfg)

        # ------- state -------
        self.t0 = None
        self.t_sim = 0.0
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.rpy = np.zeros(3)
        self.omega_body = np.zeros(3)

        self.target_pos: Optional[np.ndarray] = None
        self.target_vel: Optional[np.ndarray] = None
        self.t_last_target: Optional[float] = None

        self.got_odom = False
        self.nav_offboard = False
        self.armed = False
        self.last_ack_ok = False
        self.start_requested = False  # set by service to allow IDLE->MISSION

        self.last_stamp_us: int | None = None

        self.emergency_latched: bool = False
        self.trip_reason: str | None = None

        # mission bookkeeping
        self.plan_created = False
        self.trajectory_fresh = False
        self.at_destination = False
        self.halt_condition = False

        # takeoff states
        self.takeoff_ok_counter = 0
        self.takeoff_ok_dwell_ticks = max(1, int(0.6 / self.Ts))  # require ~0.6 s stable
        self.takeoff_started_sec = None
        self.takeoff_timeout_sec = 20.0 

        # offboard states
        self.allow_offboard_output = True        # master gate for Offboard setpoints
        self.in_offboard_last = None             # for logging transitions (optional)
        self.suppress_plan_output = False

        # hold position states
        self._hold_p4 = None          # latched hover target [x,y,z,psi]
        self._hold_mode_prev = False  # tracks rising-edge into hold mode

        # ------- IO -------
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        trip_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST, 
            depth=1
        )

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        
        self.create_subscription(VehicleOdometry, odom_topic, self._odom_cb, qos)
        self.create_subscription(Odometry, target_topic, self._target_cb, qos)
        self.create_subscription(SafetyTrip, 'safety/trip', self._on_trip, trip_qos)
        self.create_subscription(VehicleStatus, status_topic, self._on_status, sensor_qos)
        self.create_subscription(VehicleCommandAck, command_ack_topic, self._on_ack, 10)

        # publisher lives in TrajectoryManager.publish_traj(); create here:
        # (we import the class there to resolve msg type)
        
        self.traj_pub = self.create_publisher(TrajMsg, traj_topic, 10)
        self.nav_state_pub = self.create_publisher(UInt8, nav_state_topic, trip_qos)

        # Services
        self.start_srv = self.create_service(Trigger, 'navigator/start_mission', self._srv_start_mission)
        self.halt_srv  = self.create_service(Trigger, 'navigator/halt_mission',  self._srv_halt_mission)

        # timers
        self.timer = self.create_timer(self.Ts, self._tick)
        self.get_logger().info("NavigatorNode ready.")

    # ---------- callbacks ----------
    def _odom_cb(self, msg: VehicleOdometry):
        self.t_sim = self._clock_to_t(msg.timestamp)
        self.last_stamp_us = int(msg.timestamp)
        self.pos = np.array(msg.position, float)
        self.vel = np.array(msg.velocity, float)
        self.rpy[2],self.rpy[1],self.rpy[0] = self._quat_to_eul(np.array([msg.q[0],msg.q[1],msg.q[2],msg.q[3]], float))
        
        #self.get_logger().info(f"self.rpy: {self.rpy}")
        # Body-frame Angular Velocity
        self.omega_body = np.array(msg.angular_velocity, float)

        if not self.got_odom:
            self.got_odom = True
            self.get_logger().info("First odometry received.")

    def _target_cb(self, msg: Odometry):
        self.target_pos = np.array([msg.pose.pose.position.x,
                                    msg.pose.pose.position.y,
                                    msg.pose.pose.position.z], float)
        self.target_vel = np.array([msg.twist.twist.linear.x,
                                    msg.twist.twist.linear.y,
                                    msg.twist.twist.linear.z], float)
        self.t_last_target = self.t_sim

    def _on_status(self, msg: VehicleStatus):
        self.nav_offboard = (msg.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD)
        self.armed = (msg.arming_state == VehicleStatus.ARMING_STATE_ARMED)

        if self.nav_offboard != self.in_offboard_last:
            self.get_logger().info(f"[Navigator] FCU mode changed: {'OFFBOARD' if self.nav_offboard else 'NOT-OFFBOARD'}")
            self.in_offboard_last = self.nav_offboard

    def _on_ack(self, ack: VehicleCommandAck):
        # VEHICLE_RESULT_ACCEPTED = 0
        VEHICLE_RESULT_ACCEPTED = 0
        if ack.result == VEHICLE_RESULT_ACCEPTED:
            self.last_ack_ok = True

    # ---- safety trip -> EMERGENCY ----
    def _on_trip(self, msg: SafetyTrip):
        if msg.tripped:
            self.emergency_latched = True
            self.trip_reason = msg.reason
            self.sm.reset(NavState.EMERGENCY)

            # 1) Stop streaming Offboard setpoints
            if self.allow_offboard_output:
                self.get_logger().warn(f"[Navigator] EMERGENCY: stopping Offboard outputs now: {self.trip_reason}")
            self.allow_offboard_output = False

            # 2) Publish nothing further (do NOT “hold” in Offboard—stop outputs)
            # If you still want a final “hold” before stopping, do it ONCE before flipping the gate.

            # 3) (Optional) request a mode change through your offboard manager, if available
            # self._request_leave_offboard()

            self._publish_nav_state()

            self.auto_start = False
            self.start_requested = False

            self.suppress_plan_output = True
            self._clear_active_plan()

        else:
            # optional: clear EMERGENCY latch if you want automatic recovery logic here
            self.emergency_latched = False
            self.trip_reason = None


    # ---------- main loop ----------
    def _tick(self):
        
        if not self.got_odom:
            self._publish_nav_state()
            return
        
        # EMERGENCY: stop sending new trajectories; hold position
        if self.emergency_latched:
            # Hold current position & yaw
            p4 = np.array([self.pos[0], self.pos[1], self.pos[2], float(self.rpy[2])], float)
            v4 = np.zeros(4, float)
            a4 = np.zeros(4, float)
            self._publish_cmd4(p4, v4, a4)
            self._publish_nav_state()
            return
        
        
        # Ensure a plan exists (so trajectory_fresh can gate IDLE->MISSION)
        # if self.auto_start and not self.plan_created:
        #     # legacy behavior: plan and start immediately
        #     self._plan_mission()
        #     self.plan_created = True
        #     self.trajectory_fresh = True
        #     self.get_logger().info(f"self.plan_created: {self.plan_created} ")

        # events
        target_fresh = (self.t_last_target is not None) and ((self.t_sim - self.t_last_target) <= self.mission.target.timeout)
        at_takeoff = self._at_takeoff_wp()
        dist_to_target = np.linalg.norm(self.pos - self.target_pos) if (self.target_pos is not None) else 1e9
        landing_needed = (self.sm.state == NavState.FOLLOW_TARGET and
                          target_fresh and (dist_to_target < self.mission.landing.trigger_radius) and
                          ((self.pos[2] - self.mission.landing.final_altitude) > 0.15))
        landing_done = (self.sm.state == NavState.LANDING and
                        (self.pos[2] <= self.mission.landing.final_altitude + 0.05) and (np.linalg.norm(self.vel) < 0.3))

        #self.get_logger().info(f"State: {self.sm.state} | at_takeoff: {at_takeoff}: {np.linalg.norm(self.pos - self.mission.takeoff.waypoint)}: {np.linalg.norm(self.vel)}")

        ev = NavEvents(
            have_odom=bool(self.got_odom),
            auto_start=bool(self.auto_start),
            target_fresh=bool(target_fresh),
            trajectory_fresh=bool(self.trajectory_fresh),
            at_takeoff_wp=bool(at_takeoff),
            at_destination=bool(self.at_destination),
            landing_needed=bool(landing_needed),
            landing_done=bool(landing_done),
            start_requested=bool(self.start_requested),
            halt_condition=bool(self.halt_condition),
            mission_valid=bool(self.mission_valid),
        )
        prev = self.sm.state
        new = self.sm.step(ev)

        # plan on transitions
        if new != prev:
            self._on_state_change(prev, new)

        
        # select references per state
        if new == NavState.IDLE:
            p_ref, v_ref, a_ref = self._plan_or_hold()  

        elif new == NavState.TAKEOFF:
            p_ref, v_ref, a_ref = self._plan_or_hold()

        elif new == NavState.LOITER:
            p_ref, v_ref, a_ref = self._plan_or_hold()  

        elif new == NavState.FOLLOW_TARGET:
            p_ref, v_ref, a_ref = self._plan_or_hold()
            # replan periodically / when far
            if target_fresh and (dist_to_target > 0.8):
                state0_12 = np.hstack([self._current_p4(), self._current_v4(), np.zeros(4)])
                target0_12 = np.hstack([
                    self.target_pos, float(self.rpy[2]), # or face target heading if you prefer
                    self.target_vel, 0.0,
                    np.zeros(4)
                ])
                self.tm.plan_min_jerk_pose_to(self.t_sim, state0_12, target0_12, duration=None, repeat="none")
        
        elif new == NavState.LANDING:
            p_ref, v_ref, a_ref = self._plan_or_hold()

        else:
            p_ref, v_ref, a_ref = self._plan_or_hold()

        # apply limiter and publish
        # p_ref, v_ref, a_ref are 4D: [x,y,z,psi], [vx,vy,vz,psi_dot], [ax,ay,az,psi_ddot]
        # 1) limit only XYZ using your limiter
        p_cmd_xyz, v_cmd_xyz = self.limiter.limit(
            p_ref[:3], v_ref[:3],
            self.pos[:3], self.vel[:3],
            self.Ts
        )

        # 2) reassemble 4D with yaw from the plan
        p_cmd = np.array([p_cmd_xyz[0], p_cmd_xyz[1], p_cmd_xyz[2], p_ref[3]], float)
        v_cmd = np.array([v_cmd_xyz[0], v_cmd_xyz[1], v_cmd_xyz[2], v_ref[3]], float)
        a_cmd = np.array([a_ref[0],     a_ref[1],     a_ref[2],     0.0    ], float)  # or keep a_ref[3] if you compute ψ̈

        self._publish_cmd4(p_cmd, v_cmd, a_cmd)

        #self._publish_cmd4(p_ref, v_ref, a_ref)

            

    def _on_state_change(self, prev, state):

        self.get_logger().info(f"State: {prev.name} -> {state.name}")

        if state in (NavState.TAKEOFF, NavState.LOITER, NavState.FOLLOW_TARGET, NavState.MISSION):
            self.suppress_plan_output = False

        if state == NavState.TAKEOFF:
            self._plan_takeoff()
            self.takeoff_started_sec = self.get_clock().now().seconds_nanoseconds()[0]
            self.takeoff_ok_counter = 0

        elif state == NavState.MISSION:
            self._plan_mission()
            self.limiter.reset()

        elif state == NavState.LOITER:
            speed = float(self.mission.loiter.radius * self.mission.loiter.omega)
            # Build a short arc segment that loops; generator will repeat it
            state0_12 = np.hstack([self._current_p4(), self._current_v4(), np.zeros(4)])
            self.tm.plan_arc_by_rate_3d(
                t_now=self.t_sim,
                state0_12=state0_12,
                speed=speed,
                yaw_rate=self.mission.loiter.omega,   # same angular speed
                duration=2.0*np.pi/max(1e-3, abs(self.mission.loiter.omega)),  # one full lap
                repeat="loop",
                z_mode="hold",
                yaw_mode="follow_heading",
            )
            self.limiter.reset()

        elif state == NavState.FOLLOW_TARGET:# and target_fresh:
            target0_12 = np.hstack([
                self.target_pos, float(self.rpy[2]), # or face target heading if you prefer
                self.target_vel, 0.0,
                np.zeros(4)
            ])
            state0_12 = np.hstack([self._current_p4(), self._current_v4(), np.zeros(4)])
            self.tm.plan_min_jerk_pose_to(self.t_sim, state0_12, target0_12, duration=None, repeat="none")
            self.limiter.reset()

        elif state == NavState.LANDING:
            self._plan_landing()

        elif state == NavState.IDLE:
            self.suppress_plan_output = True   # hold in IDLE
            self._clear_active_plan()          # drop any plan still in TM
            self.halt_condition = False        # you already had this line

        # one-shot start trigger consumed once we leave IDLE
        if prev == NavState.IDLE and state != NavState.IDLE:
            self.start_requested = False
        # halt condition is sticky only for the transition; clear it after we’re in IDLE
        if state == NavState.IDLE:
            self.halt_condition = False


        self._publish_nav_state()


    def _plan_or_hold(self):
        """Return plan ref; if we're in hold, return a latched hover target."""
        in_hold = (self.suppress_plan_output or self.sm.state == NavState.IDLE)

        # Rising edge: just entered hold → latch the current pose once
        if in_hold and not self._hold_mode_prev:
            self._hold_p4 = self._current_p4().copy()
            self.limiter.reset()          # optional: avoid step-limited drift
            # self.get_logger().info(f"[Navigator] HOLD latched at {self._hold_p4}")

        # Update edge tracker
        self._hold_mode_prev = in_hold

        if in_hold:
            # Use the latched pose every tick (do NOT refresh it)
            p = self._hold_p4 if self._hold_p4 is not None else self._current_p4()
            v = np.zeros(4, dtype=float)
            a = np.zeros(4, dtype=float)
            return p.copy(), v, a

        # Not in hold → clear the latch (so next entry will re-latch)
        self._hold_p4 = None

        # Normal: pull from active plan; if none, fall back to a latched-or-current hold
        p, v, a = self.tm.get_plan_ref(self.t_sim)
        if p is None:
            if self._hold_p4 is None:
                self._hold_p4 = self._current_p4().copy()
            return self._hold_p4.copy(), np.zeros(4, float), np.zeros(4, float)

        return p, v, a


    # -------------------- planning --------------------
    def _plan_mission(self):
        """Plan according to mission params. Called once when odom first arrives."""
        m = self.mission.mission    # variant: LineTo/Straight/Arc/...
        rep = getattr(m.common, "repeat", "loop")
        spd = float(getattr(m.common, "speed", 1.0))


        # Seed start pose/yaw
        if m.common.start.use_current:
            x0, y0, psi0 = float(self.pos[0]), float(self.pos[1]), float(self.rpy[2])
        else:
            x0, y0, psi0 = float(m.common.start.x), float(m.common.start.y), float(m.common.start.psi)

        z0 = float(self.pos[2])   # keep current altitude for horizontal mission tracks

        # Build piecewise segment list (2D), then the manager will lift to 3D/4D
        segs = []

        # Convert this *2D* mission to **piecewise 4D** segments and call
        # self.tm.plan_piecewise_track_3d(...). For example, Straight:

        if isinstance(m, LineTo):
            # Move to an XY ψ goal in 'duration' at constant speed along a straight track.
            gx, gy, gpsi = m.goal_xypsi
            dist = max(1e-6, np.hypot(gx - x0, gy - y0))
            T = float(m.duration) if getattr(m, "duration", None) else dist / max(0.05, spd)
            segs.append({
                "type": "straight",
                "speed": spd,
                "duration": T,
                "end_heading": gpsi,
            })

        elif isinstance(m, Straight):
            # Single straight segment of fixed distance at speed
            dist = float(m.segment_distance)
            T = dist / max(0.05, spd) # avoid zero-speed divisions
            segs.append({
                "type": "straight",
                "speed": spd,
                "duration": T,
            })

        elif isinstance(m, Arc):
            R = float(m.radius)
            cw = bool(getattr(m, "cw", True))
            # yaw rate sign: cw negative, ccw positive (convention)
            if getattr(m, "angle", None) is not None:
                angle = float(m.angle)               # radians, signed by cw
                omega = np.sign(-1 if cw else 1) * (spd / max(0.05, R))
                T = abs(angle) / max(1e-3, abs(omega))
            else:
                # if yaw_rate is provided in params, honor it and still pass speed
                omega = float(m.yaw_rate) * (-1 if cw else 1)
                T = max(0.1, (2.0 * np.pi) / max(1e-3, abs(omega))) * 0.25  # quarter lap default
            segs.append({
                "type": "arc",
                "radius": R,
                "yaw_rate": omega,
                "speed": spd,
                "duration": T,
                "cw": cw,
            })

        elif isinstance(m, RoundedRectangle):
            W, H, Rc = float(m.width), float(m.height), float(m.corner_radius)
            cw = bool(m.cw)
            # 2 straights + 2 rounded corners (x2) = 4 arcs, 4 straights
            # Long edge
            T_long  = (W - 2*Rc) / max(0.05, spd)
            # Short edge
            T_short = (H - 2*Rc) / max(0.05, spd)
            # Each corner is a quarter circle
            omega = (spd / max(0.05, Rc)) * (-1 if cw else 1)
            T_corner = (0.5 * np.pi) / max(1e-3, abs(omega))
            for _ in range(2):
                segs.append({"type": "straight", "speed": spd, "duration": T_long})
                segs.append({"type": "arc", "speed": spd, "radius": Rc, "yaw_rate": omega, "duration": T_corner, "cw": cw})
                segs.append({"type": "straight", "speed": spd, "duration": T_short})
                segs.append({"type": "arc", "speed": spd, "radius": Rc, "yaw_rate": omega, "duration": T_corner, "cw": cw})

        elif isinstance(m, RacetrackCapsule):
            L, R, cw = float(m.straight_length), float(m.radius), bool(m.cw)
            # Two straights + two half-circles
            T_st = L / max(0.05, spd)
            omega = (spd / max(0.05, R)) * (-1 if cw else 1)
            T_half = np.pi / max(1e-3, abs(omega))
            # straight → half-circle → straight → half-circle
            segs += [
                {"type": "straight", "speed": spd, "duration": T_st},
                {"type": "arc", "speed": spd, "radius": R, "yaw_rate": omega, "duration": T_half, "cw": cw},
                {"type": "straight", "speed": spd, "duration": T_st},
                {"type": "arc", "speed": spd, "radius": R, "yaw_rate": omega, "duration": T_half, "cw": cw},
            ]
            
        else:
            self.get_logger().warn(f"Unknown mission.type='{self.mission.mission_type}', staying IDLE.")
            return

        # Plan in 4-D using piecewise segments (manager handles lifting & yaw mode)
        state0_12 = np.hstack([np.array([x0, y0, z0, psi0], dtype=float),
                            self._current_v4(),
                            np.zeros(4, dtype=float)])
        
        self.get_logger().info(f"[Navigator] piecewise segments = {segs}")

        self.tm.plan_piecewise_track_3d(
            t_now=self.t_sim,
            state0_12=state0_12,
            segments=segs,         # list of {"type": "straight"/"arc", ...}
            repeat=rep,            # "none" | "loop" | "pingpong"
            name="mission_track",
        )
        self.limiter.reset()


    def _plan_takeoff(self):
        # If your odometry is ENU (ROS default), Z up is positive.
        x_t, y_t, z_t = self.mission.takeoff.waypoint
        p4_target = np.array([float(x_t), float(y_t), float(z_t), float(self.rpy[2])], dtype=float)

        state0_12 = np.hstack([self._current_p4(), self._current_v4(), np.zeros(4, dtype=float)])
        self.tm.plan_min_jerk_pose_to(
            t_now=self.t_sim,
            state0_12=state0_12,
            target0_12=p4_target,
            duration=None,
            repeat="none",
        )
        self.limiter.reset()
        self.get_logger().info(f"[Navigator] TAKEOFF plan to NED {p4_target}")
        

    def _plan_landing(self):
        x_t, y_t = self._landing_xy()
        z_t = float(self.mission.landing.final_altitude)
        p4_target = np.array([x_t, y_t, z_t, float(self.rpy[2])], dtype=float)

        # Choose a reasonable duration from vertical distance and a descent rate.
        # Use takeoff speed as a proxy if you like; otherwise ~0.6 m/s is gentle.
        dz = abs(self.pos[2] - z_t)
        descent_rate = float(getattr(self.mission.takeoff, "speed", 0.6)) or 0.6
        T = max(20.0, dz / max(0.2, descent_rate))

        state0_12 = np.hstack([self._current_p4(), self._current_v4(), np.zeros(4, dtype=float)])
        self.tm.plan_min_jerk_pose_to(
            t_now=self.t_sim,
            state0_12=state0_12,
            p4_target=p4_target,
            duration=T,
            repeat="none",
        )
        self.limiter.reset()
        self.get_logger().info(f"[Navigator] LANDING plan → ENU {p4_target} (T={T:.2f}s)")
        
    


    # ---------- services ----------
    def _srv_start_mission(self, req, resp):
        if not self.got_odom:
            resp.success = False
            resp.message = "Cannot start: no odometry yet."
            return resp

        self.start_requested = True
        self.halt_condition = False

        resp.success = True
        resp.message = "Start requested."
        return resp

    def _srv_halt_mission(self, req, resp):
        # Ask the SM to go back to IDLE; we'll hold position there
        self.halt_condition = True
        # Clear any outstanding start request so we don't immediately re-enter
        self.start_requested = False
        # (optional) mark current plan as consumed
        self.trajectory_fresh = False
        self.auto_start = False            # don’t immediately leave IDLE again
        self.suppress_plan_output = True   # force HOLD outputs
        self._clear_active_plan()          # purge any active mission plan
        resp.success = True
        resp.message = "Mission halt requested."
        return resp


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
    
    def _current_p4(self) -> np.ndarray:
        return np.array([self.pos[0], self.pos[1], self.pos[2], float(self.rpy[2])], float)

    def _current_v4(self) -> np.ndarray:
        # If you don’t track ψ̇, leave it 0
        return np.array([self.vel[0], self.vel[1], self.vel[2], self.omega_body[2]], float)

    def _at_takeoff_wp(self,
                   pos_xy_tol=0.35,  # m
                   pos_z_tol=0.32,   # m
                   vz_tol=0.15,      # m/s
                   roll_pitch_tol_deg=6.0):
        x_t, y_t, z_t = self._takeoff_target_ned()
        # If start.use_current, ignore XY requirement (we're taking off in place)
        use_current_xy = getattr(self.mission.mission.common.start, "use_current", True)
        dist_xy = float(np.hypot(self.pos[0] - x_t, self.pos[1] - y_t))
        xy_ok = use_current_xy or (dist_xy <= pos_xy_tol)

        z_ok  = abs(self.pos[2] - z_t) <= pos_z_tol
        vz_ok = abs(self.vel[2]) <= vz_tol

        rp = np.abs(self.rpy[:2])  # roll, pitch
        rp_ok = (rp[0] <= np.deg2rad(roll_pitch_tol_deg)) and (rp[1] <= np.deg2rad(roll_pitch_tol_deg))

        cond = xy_ok and z_ok and vz_ok and rp_ok

        # dwell filter to avoid transient spikes
        if cond:
            self.takeoff_ok_counter += 1
        else:
            self.takeoff_ok_counter = 0

        return self.takeoff_ok_counter >= self.takeoff_ok_dwell_ticks
    
    def _takeoff_guard(self):
        if self.takeoff_started_sec is None:
            return
        now_s = self.get_clock().now().seconds_nanoseconds()[0]
        if (now_s - self.takeoff_started_sec) > self.takeoff_timeout_sec:
            self.get_logger().warn("[Navigator] Takeoff timeout—replanning takeoff.")
            self._plan_takeoff()
            self.takeoff_started_sec = now_s
            self.takeoff_ok_counter = 0
    
    def _takeoff_target_ned(self):
        x_t, y_t, z_t = self.mission.takeoff.waypoint
        return float(x_t), float(y_t), float(z_t)
    
    def _landing_xy(self):
        # Prefer the last known/valid target XY if available; else hold current XY
        if hasattr(self, "target_pos") and self.target_pos is not None:
            return float(self.target_pos[0]), float(self.target_pos[1])
        return float(self.pos[0]), float(self.pos[1])
    
    def _publish_nav_state(self):
        msg = UInt8(); 
        msg.data = self.sm.state.value  # IDLE=1, TAKEOFF=2, LOITER,  
        self.nav_state_pub.publish(msg)

    def _publish_cmd4(self, p4, v4, a4):
        if not self.allow_offboard_output:
            return
        now_us = int(self.get_clock().now().nanoseconds / 1000)
        self.tm.publish_traj(self.traj_pub, now_us, p4, v4, a4)

    def _clear_emergency_and_resume(self):
        self.emergency_latched = False
        # … reset state machine, plans, limiter, etc.
        self.allow_offboard_output = True

    def _clear_active_plan(self):
        """
        Best-effort: clear any active plan in the TrajectoryManager if the API exists.
        Otherwise mark our bookkeeping so we won't rely on a stale plan.
        """
        for fn in ("clear", "clear_plan", "reset"):
            if hasattr(self.tm, fn):
                try:
                    getattr(self.tm, fn)()
                    break
                except Exception:
                    pass
        # local bookkeeping
        self.plan_created = False
        self.trajectory_fresh = False
        self.at_destination = False


def main(args=None):
    rclpy.init(args=args)
    node = NavigatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
