# hydro_mpc/navigation/trajectory_manager.py
from __future__ import annotations
import math
import numpy as np

# px4 message compatibility: prefer 6DoF, fall back to TrajectorySetpoint
try:
    from px4_msgs.msg import TrajectorySetpoint6DoF as TrajMsg
except Exception:
    from px4_msgs.msg import TrajectorySetpoint as TrajMsg

from hydro_mpc.guidance.min_jerk_trajectory_generator import MinJerkTrajectoryGenerator

class TrajectoryManager:
    """
    Handles segment planning (min-jerk), loiter reference, and publishing.
    Does NOT decide states — only provides references / planning.
    """
    def __init__(self, a_max: np.ndarray, default_T: float,
                 loiter_center: np.ndarray, loiter_radius: float, loiter_omega: float):
        self.gen = MinJerkTrajectoryGenerator(np.asarray(a_max, float))
        self.default_T = float(default_T)
        self.loiter_center = np.asarray(loiter_center, float).reshape(3,)
        self.loiter_radius = float(loiter_radius)
        self.loiter_omega  = float(loiter_omega)

        self.plan_active = False
        self._plan_start_time = 0.0

    # ---------- planning ----------
    def plan_min_jerk(self, t_now: float, p0: np.ndarray, v0: np.ndarray,
                      p1: np.ndarray, v1: np.ndarray, T: float | None = None) -> None:
        p0, v0 = np.asarray(p0,float).reshape(3,), np.asarray(v0,float).reshape(3,)
        p1, v1 = np.asarray(p1,float).reshape(3,), np.asarray(v1,float).reshape(3,)
        T = float(T if T is not None else self.default_T)
        x0 = np.concatenate([p0, v0, np.zeros(3)])
        x1 = np.concatenate([p1, v1, np.zeros(3)])
        self.gen.generate_tracking_trajectory(x0, x1)
        self.plan_active = True
        self._plan_start_time = t_now

    # ---------- references ----------
    def get_plan_ref(self, t_now: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.plan_active:
            return None, None, None
        p, v, a = self.gen.get_ref_at_time(t_now)
        if p is None or v is None or a is None:
            self.plan_active = False
            return None, None, None
        return np.asarray(p,float), np.asarray(v,float), np.asarray(a,float)

    def loiter_ref(self, t_now: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        R, w = self.loiter_radius, self.loiter_omega
        cx, cy, cz = self.loiter_center
        x = cx + R*math.cos(w*t_now)
        y = cy + R*math.sin(w*t_now)
        z = cz
        vx = -R*w*math.sin(w*t_now)
        vy =  R*w*math.cos(w*t_now)
        vz = 0.0
        ax = -R*w*w*math.cos(w*t_now)
        ay = -R*w*w*math.sin(w*t_now)
        az = 0.0
        return np.array([x,y,z]), np.array([vx,vy,vz]), np.array([ax,ay,az])

    # ---------- publish ----------
    @staticmethod
    def publish_traj(pub, clock_now_us: int, p: np.ndarray, v: np.ndarray, a: np.ndarray,
                     yaw: float | None = None) -> None:
        p = np.asarray(p,float).reshape(3,)
        v = np.asarray(v,float).reshape(3,)
        a = np.asarray(a,float).reshape(3,)
        msg = TrajMsg()
        if hasattr(msg, "timestamp"):
            msg.timestamp = int(clock_now_us)

        # attempt common fields
        def set_triplet(obj, names, vec):
            ok = True
            for n, val in zip(names, vec):
                if hasattr(obj, n):
                    setattr(obj, n, float(val))
                else:
                    ok = False
            return ok

        # Try array fields: position / velocity / acceleration
        for field, vec in (("position", p), ("velocity", v), ("acceleration", a)):
            if hasattr(msg, field):
                arr = getattr(msg, field)
                try:
                    arr[:] = np.asarray(vec, float).tolist()
                    continue
                except Exception:
                    pass
            # fallbacks
            if field == "position":
                set_triplet(msg, ("x","y","z"), vec)
            elif field == "velocity":
                set_triplet(msg, ("vx","vy","vz"), vec)
            else:
                set_triplet(msg, ("ax","ay","az"), vec)

        if yaw is not None:
            if hasattr(msg, "yaw"):
                msg.yaw = float(yaw)
            elif hasattr(msg, "heading"):
                msg.heading = float(yaw)

        pub.publish(msg)
