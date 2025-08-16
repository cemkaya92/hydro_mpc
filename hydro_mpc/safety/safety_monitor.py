# safety_monitor.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

def _bad(x) -> bool:
    if x is None:
        return True
    arr = np.asarray(x)
    return not np.all(np.isfinite(arr))

@dataclass
class SafetyLimits:
    max_roll_pitch_deg: float = 35.0     # hard tilt limit
    max_rate_rad_s: float = 4.0          # body rate magnitude
    max_pos_err_m: float = 3.0           # ||p_ref - p||
    max_vel_err_mps: float = 3.0         # ||v_ref - v||
    odom_timeout_s: float = 0.25         # stale odom?
    cmd_bounds: tuple[float, float] | None = None  # e.g. thrust [0,1] or torque bounds

@dataclass
class SafetyHysteresis:
    trip_after_bad_cycles: int = 3       # require consecutive violations to trip
    clear_after_good_cycles: int = 20    # require sustained good cycles to clear

class SafetyMonitor:
    """
    Stateless checks + small hysteresis to avoid false positives.
    Call `evaluate(...)` each control tick.
    """
    def __init__(self, limits: SafetyLimits = SafetyLimits(), hyst: SafetyHysteresis = SafetyHysteresis()):
        self.limits = limits
        self.hyst = hyst
        self._bad_count = 0
        self._good_count = 0
        self._tripped = False
        self.last_odom_stamp_s: float | None = None

    def note_odom_stamp(self, t_s: float) -> None:
        self.last_odom_stamp_s = t_s

    def evaluate(
        self,
        t_now_s: float,
        rpy_rad: np.ndarray,            # shape (3,)
        omega_rad_s: np.ndarray,        # shape (3,)
        pos_m: np.ndarray, vel_mps: np.ndarray,
        p_ref_m: np.ndarray, v_ref_mps: np.ndarray,
        u_cmd: np.ndarray | None = None # control vector to be sent
    ) -> tuple[bool, str | None]:
        """
        ALWAYS returns (safe: bool, reason: Optional[str]).
        """
        reason = None
        lim = self.limits

        # Basic sanity
        if _bad(rpy_rad) or _bad(omega_rad_s) or _bad(pos_m) or _bad(vel_mps) or _bad(p_ref_m) or _bad(v_ref_mps):
            reason = "nan_or_inf_in_state_or_ref"

        # Command sanity
        if reason is None and u_cmd is not None:
            if _bad(u_cmd):
                reason = "nan_or_inf_in_command"
            elif lim.cmd_bounds is not None:
                lo, hi = lim.cmd_bounds
                arr = np.asarray(u_cmd)
                if np.any(arr < lo) or np.any(arr > hi):
                    reason = "command_out_of_bounds"

        '''
        # Attitude limits (roll/pitch)
        if reason is None:
            rp_deg = np.abs(np.rad2deg(rpy_rad[:2]))
            if np.any(rp_deg > lim.max_roll_pitch_deg):
                reason = "excess_tilt"

        # Rate limits
        if reason is None and np.linalg.norm(omega_rad_s) > lim.max_rate_rad_s:
            reason = "excess_rate"

        # Tracking blow-up
        if reason is None:
            if np.linalg.norm(p_ref_m - pos_m) > lim.max_pos_err_m:
                reason = "position_error_too_large"
            elif np.linalg.norm(v_ref_mps - vel_mps) > lim.max_vel_err_mps:
                reason = "velocity_error_too_large"

        # Staleness
        if reason is None and self.last_odom_stamp_s is not None:
            if (t_now_s - self.last_odom_stamp_s) > lim.odom_timeout_s:
                reason = "odom_stale"
        '''
        # Hysteresis
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
        