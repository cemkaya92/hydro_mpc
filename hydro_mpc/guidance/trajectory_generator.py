#!/usr/bin/env python3

import numpy as np
from typing import Optional, Tuple, Callable, List, Dict

# ==================== Trajectory Generation ====================
class TrajectoryGenerator:
    """
    State layout (quadrotor, shape (12,)):
        p = [x, y, z, psi]
        v = [xd, yd, zd, psid]
        a = [xdd, ydd, zdd, psidd]
    Units: meters, radians, seconds (SI)
    """
    def __init__(self,
                 v_max: np.ndarray = np.array([2.0, 2.0, 2.0]),
                 a_max: np.ndarray = np.array([2.0, 2.0, 2.0]),
                 yawrate_max: float = 1.5,
                 yawacc_max: float = 3.0):

        self._v_max = v_max
        self._a_max = a_max
        self._yawrate_max = float(yawrate_max)
        self._yawacc_max  = float(yawacc_max)

        self._generated: bool = False
        self._duration: Optional[float] = None
        self._coeffs = None     # for minimum-jerk: list of 4 arrays [a0..a5] (x,y,z,psi)
        self._ref_func: Optional[
            Callable[[float], Tuple[np.ndarray, np.ndarray, np.ndarray]]
        ] = None
        self._traj_name: Optional[str] = None
        self._repeat_mode = "none"     # "none", "loop", "pingpong"
        self._post_behavior = "hold"   # currently used for non-repeating polynomials
        self._segments: Optional[List[Tuple[float, float, Callable[[float], Tuple[np.ndarray,np.ndarray,np.ndarray]]]]] = None  # [(t0,t1,ref_tau)]


    # -------------------- Public API --------------------

    # -------------------- Planar primitives, lifted to 3D + yaw --------------------
    def generate_circular_trajectory(self,
                                     state_current: np.ndarray,
                                     altitude: Optional[float],
                                     speed: float,
                                     yaw_rate: float,
                                     duration=None,
                                     repeat="loop",
                                     z_mode: str = "hold",
                                     z_rate: float = 0.0,
                                     yaw_mode: str = "follow_heading"):
        """
        XY circular/straight motion with optional vertical motion and yaw scheduling.
          - z_mode: "hold" keeps constant altitude (use altitude if not None else current z),
                    "rate" uses constant climb rate z_rate.
          - yaw_mode: "follow_heading" => yaw=heading(t), "rate" => yaw(t)=yaw0+clamped(yaw_rate)*t,
                      "hold" => yaw(t)=yaw0.
        """
        # Parse state (support legacy (9,) [x,y,psi] too; fall back gracefully)
        if state_current.shape[0] >= 12:
            p0 = state_current[0:4].astype(float)
        else:
            # legacy rover-style start: [x,y,psi]
            p2d = state_current[0:3].astype(float)
            p0 = np.array([p2d[0], p2d[1], (altitude if altitude is not None else 0.0), p2d[2]], dtype=float)

        x0, y0, z0, psi0 = p0.tolist()
        if altitude is not None:
            z0 = float(altitude)

        v = float(np.clip(speed, -float(self._v_max[0]), float(self._v_max[0])))
        w = float(np.clip(yaw_rate, -self._yawrate_max, self._yawrate_max))

        # Default duration
        if duration is None:
            T = (2.0*np.pi/abs(w)) if (abs(w) > 1e-6 and yaw_mode != "hold") else 5.0
        else:
            T = float(duration)

        # Build a reference function f(t)->(p(4), v(4), a(4))
        if abs(v) < 1e-9:
            # Hover / in-place yaw (or hold)
            def _ref(t: float):
                psi = psi0
                psid = 0.0
                if yaw_mode == "rate":
                    psid = w
                    psi = psi0 + w * t
                x = x0; y = y0
                xd = yd = xdd = ydd = 0.0
                # Altitude profile
                if z_mode == "rate":
                    z = z0 + z_rate * t; zd = z_rate; zdd = 0.0
                else:
                    z = z0; zd = zdd = 0.0
                psidd = 0.0
                return (np.array([x, y, z, psi]),
                        np.array([xd, yd, zd, psid]),
                        np.array([xdd, ydd, zdd, psidd]))
        else:
            if abs(w) < 1e-6:
                # Straight line; yaw holds heading
                cpsi, spsi = np.cos(psi0), np.sin(psi0)
                def _ref(t: float):
                    x  = x0 + v * t * cpsi
                    y  = y0 + v * t * spsi
                    xd = v * cpsi
                    yd = v * spsi
                    xdd = ydd = 0.0
                    # z
                    if z_mode == "rate":
                        z = z0 + z_rate * t; zd = z_rate; zdd = 0.0
                    else:
                        z = z0; zd = zdd = 0.0
                    # yaw
                    if yaw_mode == "rate":
                        psid = w; psi = psi0 + w * t
                    elif yaw_mode == "hold":
                        psid = 0.0; psi = psi0
                    else:
                        # follow_heading
                        psi = psi0; psid = 0.0
                    psidd = 0.0
                    return (np.array([x, y, z, psi]),
                            np.array([xd, yd, zd, psid]),
                            np.array([xdd, ydd, zdd, psidd]))
            else:
                # Constant-twist XY arc
                R = (v / w) if abs(w) > 1e-9 else 1e9
                def _ref(t: float):
                    psi = psi0 + (w * t if yaw_mode in ("rate", "follow_heading") else 0.0)
                    # integrate planar
                    x = x0 + R * (np.sin(psi) - np.sin(psi0)) if abs(w) > 1e-9 else x0 + v*t*np.cos(psi0)
                    y = y0 - R * (np.cos(psi) - np.cos(psi0)) if abs(w) > 1e-9 else y0 + v*t*np.sin(psi0)
                    xd = v * np.cos(psi if yaw_mode == "follow_heading" else (psi0 + w*t))
                    yd = v * np.sin(psi if yaw_mode == "follow_heading" else (psi0 + w*t))
                    xdd = -v * w * np.sin(psi) if abs(w) > 1e-9 else 0.0
                    ydd =  v * w * np.cos(psi) if abs(w) > 1e-9 else 0.0
                    # z
                    if z_mode == "rate":
                        z = z0 + z_rate * t; zd = z_rate; zdd = 0.0
                    else:
                        z = z0; zd = zdd = 0.0
                    # yaw selection
                    if yaw_mode == "follow_heading":
                        psid = w;            # same turn rate as heading
                    elif yaw_mode == "rate":
                        psid = w
                    else:
                        psid = 0.0
                        psi  = psi0
                    psidd = 0.0
                    return (np.array([x, y, z, psi]),
                            np.array([xd, yd, zd, psid]),
                            np.array([xdd, ydd, zdd, psidd]))

        self._duration = T
        self._repeat_mode = repeat
        self._post_behavior = "hold"
        self._ref_func = _ref
        self._coeffs = None
        self._traj_name = "arc3d" if abs(w) > 1e-6 or yaw_mode != "hold" else "straight3d"
        self._generated = True

    # -------------------- New: 4D minimum-jerk pose-to --------------------
    def generate_minimum_jerk_pose_to(self,
                                      state_current: np.ndarray,
                                      target_state: np.ndarray,
                                      duration: Optional[float] = None,
                                      repeat: str = "none",
                                      post_behavior: str = "hold"):
        """
        Quintic min-jerk from current (p,v,a) -> target (p,v,a) in 4D (x,y,z,psi).
        Accepts:
          - target_state shape (4,)  => final p only, with zero final v,a
          - target_state shape (12,) => full final [p v a]
        Yaw is wrapped to the nearest equivalent angle.
        """
        # Current state (support legacy shapes)
        if state_current.shape[0] >= 12:
            p0 = state_current[0:4].astype(float)
            v0 = state_current[4:8].astype(float)
            a0 = state_current[8:12].astype(float)
        else:
            # legacy: treat yaw & its rates as zero
            p0 = np.array([state_current[0], state_current[1], 0.0, state_current[2]], dtype=float)
            v0 = np.zeros(4); a0 = np.zeros(4)

        # Target interpretation
        if target_state.size == 4:
            pf = target_state.astype(float)
            pf[3] = p0[3] + self._wrap_to_pi(pf[3] - p0[3])
            vf = np.zeros(4); af = np.zeros(4)
        elif target_state.size == 12:
            pf = target_state[0:4].astype(float)
            vf = target_state[4:8].astype(float)
            af = target_state[8:12].astype(float)
            pf[3] = p0[3] + self._wrap_to_pi(pf[3] - p0[3])
        else:
            raise ValueError("target_state must be shape (4,) or (12,)")

        # === pick duration ===
        if duration is None:
            T = self._auto_duration_minjerk_4d(p0, v0, a0, pf, vf, af)
        else:
            T = float(duration)

        coeffs = []
        # Plan each axis independently (x,y,z,psi)
        # For yaw, we do a standard quintic but we ensured the Δψ is wrapped.
        for i in range(4):
            coeffs.append(self._plan_mj(p0[i], v0[i], a0[i], pf[i], vf[i], af[i], T))

        self._coeffs = coeffs
        self._duration = T
        self._repeat_mode = repeat
        self._post_behavior = post_behavior
        self._ref_func = None
        self._traj_name = "minimum-jerk 4D pose-to"
        self._generated = True


    def get_ref_at_time(self, t: float):
        """Return (p(4), v(4), a(4)) for clamped time in [0, T]."""
        if not self._generated:
            return None, None, None

        # Piecewise track takes precedence
        if self._segments is not None and len(self._segments) > 0:
            Ttot = self._segments[-1][1]
            tt = max(0.0, float(t))
            if np.isfinite(Ttot):
                if self._repeat_mode == "loop" and Ttot > 1e-9:
                    tt = tt % Ttot
                elif self._repeat_mode == "none":
                    tt = min(tt, Ttot)
            # find active segment
            for (t0, t1, ref_tau) in self._segments:
                if tt <= t1 or (t1 == Ttot and tt >= t1):
                    tau = tt - t0
                    return ref_tau(tau)
            # fallback: last segment end
            t0, t1, ref_tau = self._segments[-1]
            return ref_tau(t1 - t0)

        # Otherwise, polynomial or single primitive
        _tau = max(0.0, t)
        _tau_clamped = min(_tau, self._duration) if (self._duration is not None and np.isfinite(self._duration)) else _tau
        if self._ref_func is not None:
            return self._ref_func(_tau_clamped)
        # MJ poly: now 4D (x,y,z,psi)
        p = np.array([self._mj_eval(self._coeffs[i], _tau_clamped, 0)[0] for i in range(4)])
        v = np.array([self._mj_eval(self._coeffs[i], _tau_clamped, 0)[1] for i in range(4)])
        a = np.array([self._mj_eval(self._coeffs[i], _tau_clamped, 0)[2] for i in range(4)])
        return p, v, a


    # -------------------- Piecewise 3D + yaw builder --------------------
    def generate_piecewise_track_3d(
        self,
        state_start: np.ndarray,
        segments: List[Dict],
        repeat: str = "loop",
        name: str = "piecewise3d",
    ):
        """
        Concatenate 3D + yaw segments into a single reference (p(4), v(4), a(4)).
        Supported segment dicts:
          - {'type':'straight',   'speed': v,              'duration': T}     # forward along current yaw
          - {'type':'arc',        'speed': v, 'radius': R, 'duration': T, 'cw': bool}   # constant-twist in XY
          - {'type':'climb',      'rate':  z_rate,           'duration': T}   # change Z at constant rate
          - {'type':'hover_turn', 'yaw_rate': w,             'duration': T}   # spin in place
          - {'type':'hold',                               'duration': T}     # hold pose
        Notes:
          * straight/arc accept optional 'z_rate' (default 0) and yaw policy via:
              seg.get('yaw_mode', 'follow_heading'|'rate'|'hold'), seg.get('yaw_rate', 0.0)
        """
        assert state_start.shape[0] >= 4, "state_start must include at least [x,y,z,psi]"
        # Current pose/vel/acc (gracefully handle shorter legacy shapes)
        if state_start.shape[0] >= 12:
            p_cur = state_start[0:4].astype(float)
            v_cur = state_start[4:8].astype(float)
            a_cur = state_start[8:12].astype(float)
        else:
            p_cur = np.zeros(4, dtype=float); v_cur = np.zeros(4, dtype=float); a_cur = np.zeros(4, dtype=float)
            p_cur[0:2] = state_start[0:2].astype(float)
            p_cur[3]   = float(state_start[2])
            # leave z at 0 by default

        seg_list: List[Tuple[float,float,Callable[[float],Tuple[np.ndarray,np.ndarray,np.ndarray]]]] = []
        t_cursor = 0.0

        def _mk_hold(p0, T):
            def _f(tau: float):
                tt = min(max(0.0, tau), T)
                p = p0.copy(); v = np.zeros(4); a = np.zeros(4)
                return p, v, a
            return _f, T

        def _mk_straight(p0, v_fwd, T, z_rate=0.0, yaw_mode="follow_heading", yaw_rate=0.0):
            x0,y0,z0,psi0 = p0.tolist()
            cpsi, spsi = np.cos(psi0), np.sin(psi0)
            def _f(tau: float):
                tt = min(max(0.0, tau), T)
                x  = x0 + v_fwd * tt * cpsi
                y  = y0 + v_fwd * tt * spsi
                z  = z0 + z_rate * tt
                xd = v_fwd * cpsi
                yd = v_fwd * spsi
                zd = z_rate
                xdd = ydd = zdd = 0.0
                if yaw_mode == "rate":
                    psi = psi0 + yaw_rate * tt; psid = yaw_rate; psidd = 0.0
                elif yaw_mode == "hold":
                    psi = psi0; psid = psidd = 0.0
                else:
                    psi = psi0; psid = 0.0; psidd = 0.0  # follow current heading
                return np.array([x,y,z,psi]), np.array([xd,yd,zd,psid]), np.array([xdd,ydd,zdd,psidd])
            return _f, T

        def _mk_arc(p0, v_lin, R, T=None, z_rate=0.0, yaw_mode="follow_heading", cw=True, angle=None):
            x0,y0,z0,psi0 = p0.tolist()
            v = float(abs(v_lin))
            sgn = +1.0 if bool(cw) else  -1.0
            R = float(abs(R))
            
            if T is not None:
                # If T is specified, sweep either full circle or 'angle' in T
                sweep = (2.0*np.pi if angle is None else abs(float(angle)))
                w = sgn * sweep / max(1e-3, float(T))   # rad/s
                v = abs(w) * R                           # ensure closure
                T_eff = float(T)
            else:
                # No T: set ω from v/R and derive T (full circle or given angle)
                w = sgn * (v / max(0.05, R))
                sweep = (2.0*np.pi if angle is None else abs(float(angle)))
                T_eff = sweep / max(1e-3, abs(w))

            # Circle center (unicycle model)
            xc = x0 - R * np.sin(psi0)
            yc = y0 + R * np.cos(psi0)

            def wrap_pi(a):
                return (a + np.pi) % (2.0*np.pi) - np.pi
            
            def _f(tau: float):
                tt = float(np.clip(tau, 0.0, T_eff))

                # Unwrapped path heading used for geometry/derivatives
                heading = psi0 + w * tt

                # Position from center
                x = xc + R * np.sin(heading)
                y = yc - R * np.cos(heading)

                # Vel/acc along the tangent (match geometry exactly)
                xd  = v * np.cos(heading)
                yd  = v * np.sin(heading)
                xdd = -v * w * np.sin(heading)
                ydd =  v * w * np.cos(heading)

                # Altitude
                z   = z0 + z_rate * tt
                zd  = z_rate
                zdd = 0.0

                # Yaw command
                if yaw_mode == "follow_heading":
                    yaw_cmd = wrap_pi(heading)
                    psid    = w
                elif yaw_mode == "rate":
                    yaw_cmd = wrap_pi(psi0 + w * tt)
                    psid    = w
                else:  # "hold"
                    yaw_cmd = wrap_pi(psi0)
                    psid    = 0.0

                return (np.array([x, y, z, yaw_cmd]),
                        np.array([xd, yd, zd, psid]),
                        np.array([xdd, ydd, zdd, 0.0]))

            return _f, T_eff

        def _mk_climb(p0, z_rate, T):
            x0,y0,z0,psi0 = p0.tolist()
            def _f(tau: float):
                tt = min(max(0.0, tau), T)
                x,y = x0,y0
                z   = z0 + z_rate * tt
                return (np.array([x,y,z,psi0]),
                        np.array([0.0,0.0,z_rate,0.0]),
                        np.array([0.0,0.0,0.0,0.0]))
            return _f, T

        def _mk_hover_turn(p0, w, T):
            x0,y0,z0,psi0 = p0.tolist()
            def _f(tau: float):
                tt = min(max(0.0, tau), T)
                psi = psi0 + w * tt
                return (np.array([x0,y0,z0,psi]),
                        np.array([0.0,0.0,0.0,w]),
                        np.array([0.0,0.0,0.0,0.0]))
            return _f, T

        # Build segments
        for seg in segments:
            st = seg["type"].lower()
            if st == "straight":
                v = float(seg["speed"])
                T = float(seg["duration"])
                z_rate = float(seg.get("z_rate", 0.0))
                yaw_mode = seg.get("yaw_mode","follow_heading")
                yaw_rate = float(seg.get("yaw_rate", 0.0))
                f,T = _mk_straight(p_cur, v, T, z_rate, yaw_mode, yaw_rate)
            elif st == "arc":
                v = float(seg["speed"])
                T = float(seg["duration"])
                z_rate = float(seg.get("z_rate", 0.0))
                yaw_mode = seg.get("yaw_mode","follow_heading")
                if "radius" in seg:
                    R = float(seg["radius"])
                    # direction can be given as 'cw' (bool) or 'direction' in {"cw","ccw"} or 'dir_sign' in {+1,-1}
                    if "cw" in seg:
                        cw = bool(seg["cw"])
                    elif "direction" in seg:
                        cw = (str(seg["direction"]).lower() == "cw")
                    elif "dir_sign" in seg:
                        cw = (float(seg["dir_sign"]) < 0.0)  # negative sign → cw
                    else:
                        cw = True  # default

                    # If duration wasn't derived upstream, you can auto-fill when angle is given:
                    if "angle" in seg and not np.isfinite(T):
                        omega = (v / max(0.05, abs(R))) * (+1.0 if cw else -1.0)
                        T = abs(float(seg["angle"])) / max(1e-3, abs(omega))

                    f, T = _mk_arc(p_cur, v, R, T, z_rate, yaw_mode, cw)

                else:
                    # Legacy path: use provided yaw_rate and back-compute R
                    w = float(seg["yaw_rate"])
                    R = (v / w) if abs(w) > 1e-9 else 1e9
                    cw = (w < 0.0)  # keep convention consistent
                    f, T = _mk_arc(p_cur, v, R, T, z_rate, yaw_mode, cw)
            elif st == "climb":
                z_rate = float(seg["rate"])
                T = float(seg["duration"])
                f,T = _mk_climb(p_cur, z_rate, T)
            elif st == "hover_turn":
                w = float(seg["yaw_rate"])
                T = float(seg["duration"])
                f,T = _mk_hover_turn(p_cur, w, T)
            elif st == "hold":
                T = float(seg["duration"])
                f,T = _mk_hold(p_cur, T)
            else:
                raise ValueError(f"Unknown segment type: {st}")

            # record with absolute time bounds
            seg_list.append((t_cursor, t_cursor+T, f))
            # propagate start pose for next segment using this segment’s end value
            p_end, v_end, a_end = f(T)
            p_cur = p_end.copy()
            t_cursor += T

        # Commit piecewise plan
        self._segments = seg_list
        self._duration = t_cursor
        self._repeat_mode = repeat
        self._post_behavior = "hold"
        self._ref_func = None
        self._coeffs = None
        self._traj_name = name
        self._generated = True



    
    # // Private Functions
    # Minimum Jerk Trajectory Utilities from jinays.py
    def _mj_coeffs(self, t0, tf, p0, pf, v0=0, vf=0, a0=0, af=0):
        T = tf - t0
        if T <= 1e-5:
            return np.zeros(6)
        A = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0, 0.0, 0.0],
            [1.0, T, T**2, T**3, T**4, T**5],
            [0.0, 1.0, 2.0*T, 3.0*T**2, 4.0*T**3, 5.0*T**4],
            [0.0, 0.0, 2.0, 6.0*T, 12.0*T**2, 20.0*T**3]
        ])
        b = np.array([p0, v0, a0, pf, vf, af])
        return np.linalg.lstsq(A, b, rcond=None)[0]

    def _mj_eval(self,_coeffs, t, t0):
        dt = t - t0
        p = _coeffs[0] + _coeffs[1]*dt + _coeffs[2]*dt**2 + _coeffs[3]*dt**3 + _coeffs[4]*dt**4 + _coeffs[5]*dt**5
        v = _coeffs[1] + 2*_coeffs[2]*dt + 3*_coeffs[3]*dt**2 + 4*_coeffs[4]*dt**3 + 5*_coeffs[5]*dt**4
        a = 2*_coeffs[2] + 6*_coeffs[3]*dt + 12*_coeffs[4]*dt**2 + 20*_coeffs[5]*dt**3
        return p, v, a

    def _find_tf_nonzero(self,p0, pf, v0, vf, a0, af, a_max, t_min=0.1, t_max=15.0, tol=1e-3): # t_min=0.1
        _tf_low, _tf_high = t_min, t_max
        for _ in range(30):
            _tf = 0.5 * (_tf_low + _tf_high)
            if _tf <= 0:
                _tf_low = tol
                continue
            _coeffs = self._mj_coeffs(0, _tf, p0, pf, v0, vf, a0, af)
            _tvec = np.linspace(0, _tf, 50)
            _acc = np.array([self._mj_eval(_coeffs, t, 0)[2] for t in _tvec])
            if np.any(np.isnan(_acc)):
                _tf_low = _tf
                continue
            _acc_max = np.max(np.abs(_acc))
            if _acc_max > a_max:
                _tf_low = _tf
            else:
                _tf_high = _tf
            if _tf_high - _tf_low < tol:
                break
        return _tf_high
    
    # -------------------- Duration search (auto-T) --------------------
    def _auto_duration_minjerk_4d(self, p0, v0, a0, pf, vf, af,
                                  t_min: float = 0.15,
                                  t_max: float = 50.0,
                                  samples: int = 40) -> float:
        """
        Find the smallest T in [t_min, t_max] such that per-axis velocity/acceleration
        limits are respected for a 4D quintic (x,y,z,psi).
        Strategy:
          1) Try to reuse a class-provided _find_tf_nonzero for the 3 position axes.
          2) For yaw (and as a fallback for all axes), do a robust sampler-based
             compliance check + bisection on T.
        """
        # helper: max |v|, |a| over trajectory at given T (sampled)
        def _axis_peaks(p0_, v0_, a0_, pf_, vf_, af_, T_):
            coeff = self._plan_mj(p0_, v0_, a0_, pf_, vf_, af_, T_)
            ts = np.linspace(0.0, T_, max(10, samples))
            vmax = 0.0; amax = 0.0
            for t in ts:
                _, v, a = self._mj_eval(coeff, t, 0)  # assumes signature -> (p,v,a)
                vmax = max(vmax, abs(float(v)))
                amax = max(amax, abs(float(a)))
            return vmax, amax

        # 1) Try per-axis times using provided _find_tf_nonzero for positions
        T_candidates = []
        if hasattr(self, "_find_tf_nonzero"):
            for i in range(3):
                try:
                    Ti = float(self._find_tf_nonzero(p0[i], pf[i], v0[i], vf[i], a0[i], af[i],
                                                     float(self._a_max[i]), t_min, t_max))
                    if np.isfinite(Ti) and Ti > 0.0:
                        T_candidates.append(Ti)
                except Exception:
                    pass

        # 2) Yaw axis—respect yawrate & yawacc
        #    (and positions too, if step 1 didn't yield anything reliable)
        def _search_axis(v_lim, a_lim, p0_, v0_, a0_, pf_, vf_, af_):
            lo = t_min
            hi = max(t_min*2.0, 1.0)
            # grow 'hi' until feasible or bound by t_max
            while hi < t_max:
                vpk, apk = _axis_peaks(p0_, v0_, a0_, pf_, vf_, af_, hi)
                if (vpk <= v_lim + 1e-6) and (apk <= a_lim + 1e-6):
                    break
                hi *= 1.6
            hi = min(hi, t_max)
            # if still infeasible at t_max, just return t_max
            vpk, apk = _axis_peaks(p0_, v0_, a0_, pf_, vf_, af_, hi)
            if not ((vpk <= v_lim + 1e-6) and (apk <= a_lim + 1e-6)):
                return hi
            # bisection to tighten
            for _ in range(32):
                mid = 0.5*(lo+hi)
                vpk, apk = _axis_peaks(p0_, v0_, a0_, pf_, vf_, af_, mid)
                if (vpk <= v_lim + 1e-6) and (apk <= a_lim + 1e-6):
                    hi = mid
                else:
                    lo = mid
                if (hi-lo) < 1e-3:
                    break
            return hi

        # search positions if needed
        if len(T_candidates) < 3:
            for i in range(3):
                Ti = _search_axis(float(self._v_max[i]), float(self._a_max[i]),
                                  p0[i], v0[i], a0[i], pf[i], vf[i], af[i])
                T_candidates.append(Ti)

        # yaw search
        Tyaw = _search_axis(self._yawrate_max, self._yawacc_max,
                            p0[3], v0[3], a0[3], pf[3], vf[3], af[3])
        T_candidates.append(Tyaw)

        # overall duration is the max across axes
        T = max(T_candidates) if len(T_candidates) > 0 else 1.0
        return float(np.clip(T, t_min, t_max))
    

    # ===== Rover utilities ported & adapted to 4D =====
    @staticmethod
    def _wrap_to_pi(angle: float) -> float:
        a = (angle + np.pi) % (2.0*np.pi) - np.pi
        if a <= -np.pi:
            a += 2.0*np.pi
        return a

    @staticmethod
    def _plan_mj(p0, v0, a0, pf, vf, af, T) -> np.ndarray:
        """Quintic a0..a5 for p(0)=p0, p'(0)=v0, p''(0)=a0 and same at T."""
        T1 = T
        T2 = T1*T1
        T3 = T2*T1
        T4 = T3*T1
        T5 = T4*T1
        A0, A1, A2 = p0, v0, 0.5*a0
        c0 = pf - (A0 + A1*T1 + A2*T2)
        c1 = vf - (A1 + 2.0*A2*T1)
        c2 = af - (2.0*A2)
        M = np.array([[T3,  T4,   T5],
                      [3*T2,4*T3, 5*T4],
                      [6*T1,12*T2,20*T3]], dtype=float)
        b = np.array([c0, c1, c2], dtype=float)
        A3, A4, A5 = np.linalg.solve(M, b)
        return np.array([A0, A1, A2, A3, A4, A5], dtype=float)
    
