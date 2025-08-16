# hydro_mpc/safety/rate_limiter.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class RateLimitConfig:
    err_pos_cap: np.ndarray  # (3,)
    err_vel_cap: np.ndarray  # (3,)
    ref_v_cap:  np.ndarray   # (3,)  max delta on position ref per tick
    ref_a_cap:  np.ndarray   # (3,)  max delta on velocity ref per tick

class SafetyRateLimiter:
    """
    Reusable reference limiter: clamp error vs current state and slew-limit
    change vs last command. Independent of ROS; reuse in your MPC too.
    """
    def __init__(self, cfg: RateLimitConfig):
        self.cfg = cfg
        self._prev_p_cmd: np.ndarray | None = None
        self._prev_v_cmd: np.ndarray | None = None

    def reset(self):
        self._prev_p_cmd = None
        self._prev_v_cmd = None

    def limit(self, p_ref: np.ndarray, v_ref: np.ndarray,
              pos: np.ndarray, vel: np.ndarray, Ts: float) -> tuple[np.ndarray, np.ndarray]:
        p_ref = np.asarray(p_ref, float).reshape(3,)
        v_ref = np.asarray(v_ref, float).reshape(3,)
        pos   = np.asarray(pos, float).reshape(3,)
        vel   = np.asarray(vel, float).reshape(3,)

        # 1) clamp error vs current state
        e_p = np.clip(p_ref - pos, -self.cfg.err_pos_cap, self.cfg.err_pos_cap)
        e_v = np.clip(v_ref - vel, -self.cfg.err_vel_cap, self.cfg.err_vel_cap)
        p_cmd = pos + e_p
        v_cmd = vel + e_v

        # 2) slew-limit vs previous command
        if self._prev_p_cmd is None:
            self._prev_p_cmd = pos.copy()
            self._prev_v_cmd = vel.copy()

        dp_max = self.cfg.ref_v_cap * Ts
        dv_max = self.cfg.ref_a_cap * Ts

        p_cmd = self._prev_p_cmd + np.clip(p_cmd - self._prev_p_cmd, -dp_max, dp_max)
        v_cmd = self._prev_v_cmd + np.clip(v_cmd - self._prev_v_cmd, -dv_max, dv_max)

        self._prev_p_cmd = p_cmd
        self._prev_v_cmd = v_cmd
        return p_cmd, v_cmd
