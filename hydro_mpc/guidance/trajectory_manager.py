# hydro_mpc/navigation/trajectory_manager.py
from __future__ import annotations
import math
import numpy as np
import rclpy
from typing import Dict, List, Optional, Tuple, Callable

from px4_msgs.msg import TrajectorySetpoint6dof as TrajMsg

from hydro_mpc.guidance.trajectory_generator import TrajectoryGenerator


class TrajectoryManager:
    """
    Quadrotor trajectory manager:
      - Plans with a 4D TrajectoryGenerator (x,y,z,psi).
      - Stores start time, exposes get_plan_ref(t_now).
      - Publishes PX4 TrajectorySetpoint (duck-typed).
    """
    def __init__(
        self,
        v_max: np.ndarray = np.array([3.0, 3.0, 2.0]),
        a_max: np.ndarray = np.array([2.5, 2.5, 2.0]),
        yawrate_max: float = 1.2,
        yawacc_max: float = 3.0,
        default_T: float = 4.0,
    ):
        self.gen = TrajectoryGenerator(
            v_max=np.asarray(v_max, float),
            a_max=np.asarray(a_max, float),
            yawrate_max=float(yawrate_max),
            yawacc_max=float(yawacc_max),
        )
        self.default_T = float(default_T)
        self.plan_active: bool = False
        self._plan_start_time: float = 0.0
        self._plan_type: Optional[str] = None
        self._plan_meta: Dict = {}



    # =========================
    # Planning APIs (quad 4D)
    # =========================
    def plan_min_jerk_pose_to(
        self,
        t_now: float,
        state0_12: np.ndarray,            # [p(4), v(4), a(4)]
        target0_12: np.ndarray,          # (4,) or full (12,)
        duration: Optional[float] = None, # None => auto-T search
        repeat: str = "none",
        post_behavior: str = "hold",
    ) -> None:
        self.gen.generate_minimum_jerk_pose_to(
            state_current=np.asarray(state0_12, float).reshape(-1),
            target_state=np.asarray(target0_12, float).reshape(-1),
            duration=duration,
            repeat=repeat,
            post_behavior=post_behavior,
        )
        self._activate_plan(t_now, "min_jerk4d", repeat, duration)

    def plan_arc_by_rate_3d(
        self,
        t_now: float,
        state0_12: np.ndarray,   # [p(4), v(4), a(4)]
        speed: float,
        yaw_rate: float,
        duration: Optional[float] = None,
        repeat: str = "loop",
        z_mode: str = "hold",    # "hold" or "rate"
        z_rate: float = 0.0,
        yaw_mode: str = "follow_heading",  # "follow_heading" | "rate" | "hold"
    ) -> None:
        self.gen.generate_circular_trajectory(
            state_current=np.asarray(state0_12, float).reshape(-1),
            altitude=None,                    # keep current z unless z_mode says otherwise
            speed=float(speed),
            yaw_rate=yaw_rate,
            duration=duration,
            repeat=repeat,
            z_mode=z_mode,
            z_rate=float(z_rate),
            yaw_mode=yaw_mode,
        )
        self._activate_plan(t_now, "arc3d", repeat, duration)

    def plan_arc_by_radius_3d(
        self,
        t_now: float,
        state0_12: np.ndarray,
        speed: float,
        radius: float,
        cw: bool = True,
        angle: float | None = None,
        duration: float | None = None,
        repeat: str = "loop",
        z_mode: str = "hold",
        z_rate: float = 0.0,
        yaw_mode: str = "follow_heading",
    ) -> None:
        # Build a single arc segment and send it through the piecewise API
        seg = {"type":"arc", "speed": float(speed), "radius": float(radius), "cw": bool(cw)}
        if angle is not None and duration is None:
            # derive T from angle
            omega = (speed / max(0.05, abs(radius))) * (-1.0 if cw else +1.0)
            seg["duration"] = abs(float(angle)) / max(1e-3, abs(omega))
        else:
            seg["duration"] = float(self.default_T) if duration is None else float(duration)
        seg["yaw_mode"] = yaw_mode
        seg["z_rate"] = float(z_rate)

        self.gen.generate_piecewise_track_3d(
            state_start=np.asarray(state0_12, float).reshape(-1),
            segments=[seg],
            repeat=repeat,
            name="arc_by_radius",
        )
        self._activate_plan(t_now, "arc3d", repeat, seg["duration"])

    def plan_straight_3d(
        self,
        t_now: float,
        state0_12: np.ndarray,
        speed: float,
        duration: Optional[float] = None,  # if None => generator will pick default (e.g., 5s)
        repeat: str = "none",
        z_rate: float = 0.0,
        yaw_mode: str = "follow_heading",
        yaw_rate: float = 0.0,
    ) -> None:
        self.gen.generate_circular_trajectory(  # straight path is the w≈0 branch
            state_current=np.asarray(state0_12, float).reshape(-1),
            altitude=None,
            speed=float(speed),
            yaw_rate=float(0.0),
            duration=duration,
            repeat=repeat,
            z_mode=("rate" if abs(z_rate) > 1e-9 else "hold"),
            z_rate=float(z_rate),
            yaw_mode=("rate" if abs(yaw_rate) > 1e-9 else yaw_mode),
        )
        self._activate_plan(t_now, "straight3d", repeat, duration)

    def plan_hover_turn(
        self,
        t_now: float,
        state0_12: np.ndarray,
        yaw_rate: float,
        duration: float,
        repeat: str = "none",
    ) -> None:
        segs = [{'type': 'hover_turn', 'yaw_rate': float(yaw_rate), 'duration': float(duration)}]
        self.gen.generate_piecewise_track_3d(
            state_start=np.asarray(state0_12, float).reshape(-1),
            segments=segs,
            repeat=repeat,
            name="hover_turn",
        )
        self._activate_plan(t_now, "hover_turn", repeat, duration)

    def plan_climb(
        self,
        t_now: float,
        state0_12: np.ndarray,
        z_rate: float,
        duration: float,
        repeat: str = "none",
    ) -> None:
        segs = [{'type': 'climb', 'rate': float(z_rate), 'duration': float(duration)}]
        self.gen.generate_piecewise_track_3d(
            state_start=np.asarray(state0_12, float).reshape(-1),
            segments=segs,
            repeat=repeat,
            name="climb",
        )
        self._activate_plan(t_now, "climb", repeat, duration)

    def plan_piecewise_track_3d(
        self,
        t_now: float,
        state0_12: np.ndarray,
        segments: List[Dict],
        repeat: str = "loop",
        name: str = "piecewise3d",
    ) -> None:
        self.gen.generate_piecewise_track_3d(
            state_start=np.asarray(state0_12, float).reshape(-1),
            segments=segments,
            repeat=repeat,
            name=name,
        )
        self._activate_plan(t_now, name, repeat, self.gen._duration)


    # =========================
    # References + publish
    # =========================
    def get_plan_ref(self, t_now: float):
        """Return (p(4), v(4), a(4)). Deactivates when plan finishes (non-repeat)."""
        if not self.plan_active:
            return None, None, None
        tau = float(t_now) - float(self._plan_start_time)
        p, v, a = self.gen.get_ref_at_time(tau)
        if p is None:
            self.plan_active = False
            return None, None, None
        return np.asarray(p, float), np.asarray(v, float), np.asarray(a, float)

    def publish_traj(self, pub, clock_now_us: int, p4, v4, a4) -> None:
        """
        p4: [x, y, z, psi]           (meters, radians)   NED
        v4: [vx, vy, vz, psi_dot]    (m/s,   rad/s)      NED
        a4: [ax, ay, az, psi_ddot]   (m/s^2, rad/s^2)    NED
        Publishes px4_msgs/TrajectorySetpoint6dof (NED).
        """

        p = np.asarray(p4, dtype=np.float32).reshape(4,)
        v = np.asarray(v4, dtype=np.float32).reshape(4,)
        a = np.asarray(a4, dtype=np.float32).reshape(4,)

        msg = TrajMsg()

        # timestamp (μs since boot)
        if hasattr(msg, "timestamp"):
            msg.timestamp = int(clock_now_us)

        # position / velocity / acceleration [float32[3]]
        msg.position[0] = float(p[0])
        msg.position[1] = float(p[1])
        msg.position[2] = float(p[2])

        msg.velocity[0] = float(v[0])
        msg.velocity[1] = float(v[1])
        msg.velocity[2] = float(v[2])

        msg.acceleration[0] = float(a[0])
        msg.acceleration[1] = float(a[1])
        msg.acceleration[2] = float(a[2])

        # quaternion [w, x, y, z] for yaw-only rotation (roll=pitch=0)
        cy = np.cos(0.5 * float(p[3]))
        sy = np.sin(0.5 * float(p[3]))
        msg.quaternion[0] = float(cy)  # w
        msg.quaternion[1] = 0.0        # x
        msg.quaternion[2] = 0.0        # y
        msg.quaternion[3] = float(sy)  # z

        # angular velocity [p, q, r] (rad/s) — yaw rate only
        msg.angular_velocity[0] = 0.0
        msg.angular_velocity[1] = 0.0
        msg.angular_velocity[2] = float(v[3])  # psi_dot

        pub.publish(msg)



    # =========================
    # Internals
    # =========================
    def _activate_plan(self, t_now: float, plan_type: str, repeat: str, duration: Optional[float]):
        self.plan_active = True
        self._plan_start_time = float(t_now)
        self._plan_type = str(plan_type)
        self._plan_meta = {
            "repeat": str(repeat),
            "duration": (float("inf") if duration is None else float(duration)),
        }