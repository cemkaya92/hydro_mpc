import yaml
import os
from typing import Sequence, Tuple
import numpy as np  # optional; only if you like validating shapes

from hydro_mpc.utils.param_types import (
    VehicleParams, ControlParams, Mission, LineTo, Straight, Arc, RoundedRectangle, RacetrackCapsule,
    Common, StartPose, TrajParams, TakeoffParams, LoiterParams, LandingParams, TargetParams
)

def _to_f(x, name):
    if x is None: return None
    if isinstance(x, (int, float)): return float(x)
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("", "none", "null", "nan"): return None
        return float(s)
    raise ValueError(f"Mission YAML: '{name}' must be a number or null, got {type(x).__name__}")


def _to_cw(val, omega=None):
    # prefer explicit boolean or "cw"/"ccw" string; fall back to omega sign
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        s = val.strip().lower()
        if s in ("cw", "clockwise"): return True
        if s in ("ccw", "counterclockwise", "anti-clockwise", "anticlockwise"): return False
    if isinstance(val, (int, float)):
        # 1 → cw, 0/negative → ccw (if someone used 1/0/-1)
        return float(val) > 0.0
    if omega is not None:
        return float(omega) < 0.0
    # default to cw if nothing else is given
    return True


class ParamLoader:
    def __init__(self, yaml_path):
        self.yaml_path = yaml_path
        self.params = self._load_yaml(yaml_path)

    def _load_yaml(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"YAML file not found: {path}")
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return data.get('/**', {}).get('ros__parameters', {})
    
    def _first(self, paths, default=None):
        """
        Return the first non-None from a list of nested key paths.
        Each path is a list of keys, e.g., ["mission","goal","x"].
        """
        for ks in paths:
            v = self.get_nested(ks, None)
            if v is not None:
                return v
        return default

    def get(self, key, default=None):
        """Generic access to a top-level parameter."""
        return self.params.get(key, default)

    def get_nested(self, keys, default=None):
        """Access nested parameter with list of keys."""
        ref = self.params
        try:
            for key in keys:
                ref = ref[key]
            return ref
        except (KeyError, TypeError):
            return default

    def get_topic(self, topic_key, default=None):
        """Access topic name from 'topics_names' section."""
        return self.get_nested(["topics_names", topic_key], default)

    def get_all_topics(self):
        """Return entire topics dictionary if available."""
        return self.get("topics_names", {})

    def get_control_gains(self):
        """Optional helper if using control_gains block."""
        return self.get("control_gains", {})

    def get_control_params(self) -> ControlParams:

        return ControlParams(
            horizon=self.get("control_parameters", {}).get("horizon", 1.5),
            N=self.get("control_parameters", {}).get("N", 20),
            frequency=self.get("control_parameters", {}).get("frequency", 100.0),
            Q=self.get("control_parameters", {}).get("Q", [40.0, 40.0, 40.0, 4.0, 4.0, 4.0, 2.0, 2.0, 2.0, 0.5, 0.5, 0.5]),
            R=self.get("control_parameters", {}).get("R", [0.1, 1.0, 1.0, 1.0])
        )
    
    def get_vehicle_params(self) -> VehicleParams:

        params = self.get("vehicle_parameters", {})

        required_fields = [
            'mass', 'arm_length', 'gravity', 'input_scaling',
            ('inertia', 'x'), ('inertia', 'y'), ('inertia', 'z'),
            'num_of_arms', 'moment_constant', 'thrust_constant',
            'max_rotor_speed', 'PWM_MIN', 'PWM_MAX', 'zero_position_armed', 
            ('omega_to_pwm_coefficient', 'x_2'), ('omega_to_pwm_coefficient', 'x_1'), ('omega_to_pwm_coefficient', 'x_0')
        ]

        for field in required_fields:
            if isinstance(field, tuple):
                group, subfield = field
                if group not in params or subfield not in params[group]:
                    raise ValueError(f"Missing required parameter: '{group}.{subfield}' in YAML file.")
            else:
                if field not in params:
                    raise ValueError(f"Missing required parameter: '{field}' in YAML file.")

        return VehicleParams(
            mass=params['mass'],
            arm_length=params['arm_length'],
            gravity=params['gravity'],
            inertia=[
                params['inertia']['x'],
                params['inertia']['y'],
                params['inertia']['z']
            ],
            num_of_arms=params['num_of_arms'],
            moment_constant=params['moment_constant'],
            thrust_constant=params['thrust_constant'],
            max_rotor_speed=params['max_rotor_speed'],
            PWM_MIN=params['PWM_MIN'],
            PWM_MAX=params['PWM_MAX'],
            input_scaling=params['input_scaling'],
            zero_position_armed=params['zero_position_armed'],
            omega_to_pwm_coefficient=[
                params['omega_to_pwm_coefficient']['x_2'],
                params['omega_to_pwm_coefficient']['x_1'],
                params['omega_to_pwm_coefficient']['x_0']
            ]
        )
    
    def get_mission(self) -> Mission:
        """Parse modern nested mission schema -> Mission variant."""
        mroot = self.get("mission", {}) or {}
        mtype = str(mroot.get("type", "line_to")).lower()
        common = mroot.get("common", {}) or {}
        start  = (common.get("start", {}) or {})
        params = mroot.get("params", {}) or {}

        c = Common(
            repeat=str(common.get("repeat", "none")).lower(),  # none|loop|pingpong
            start=StartPose(
                bool(start.get("use_current", True)),
                float(start.get("x", 0.0)),
                float(start.get("y", 0.0)),
                float(start.get("psi", 0.0)),
            ),
            speed=None if common.get("speed") is None else float(common["speed"]),
        )

        if mtype == "line_to":
            g = params.get("goal_xypsi", [0.0, 0.0, 0.0])
            return LineTo(common=c, goal_xypsi=(float(g[0]), float(g[1]), float(g[2])),
                          duration=float(params.get("duration", 0.0)))

        if mtype == "straight":
            return Straight(common=c, segment_distance=float(params.get("segment_distance", 0.0)))

        if mtype == "arc":
            ang = _to_f(params.get("angle"), name="angle")
            yr  = _to_f(params.get("yaw_rate"), name="yaw_rate")
            R   = _to_f(params.get("radius"), name="radius")
            if ang is None and yr is None and R is None:
                raise ValueError("arc mission: provide one of {angle, yaw_rate, radius}")
            return Arc(common=c, radius=R, angle=ang, yaw_rate=yr,
                       cw=bool(params.get("cw", True)))

        if mtype == "rounded_rectangle":
            W = float(params["width"]); H = float(params["height"]); r = float(params["corner_radius"])
            if W <= 2*r or H <= 2*r:
                raise ValueError("rounded_rectangle: width/height must be > 2*corner_radius")
            return RoundedRectangle(common=c, width=W, height=H, corner_radius=r, cw=bool(params.get("cw", True)))

        if mtype == "racetrack_capsule":
            L = float(params["straight_length"]); r = float(params["radius"])
            if r <= 0.0:
                raise ValueError("racetrack_capsule: radius must be > 0")
            return RacetrackCapsule(common=c, straight_length=L, radius=r, cw=bool(params.get("cw", True)))

        raise ValueError(f"Unknown mission type: {mtype}")
    


    
    def _as_xyz(self, seq, name: str) -> Tuple[float, float, float]:
        if not isinstance(seq, (list, tuple)) or len(seq) != 3:
            raise ValueError(f"{name}: expected length-3 list/tuple [x,y,z], got {seq}")
        return (float(seq[0]), float(seq[1]), float(seq[2]))

    def get_takeoff_params(self):
        tk = self.get_nested(["mission", "takeoff"], {}) or {}
        wp = self._as_xyz(tk.get("waypoint", [0.0, 0.0, -2.0]), "mission.takeoff.waypoint")
        sp = float(tk.get("speed", 1.0))
        return TakeoffParams(waypoint=wp, speed=sp)

    def get_loiter_params(self):
        lo = self.get_nested(["mission", "loiter"], {}) or {}
        center = self._as_xyz(lo.get("center", [0.0, 0.0, -2.0]), "mission.loiter.center")
        radius = float(lo.get("radius", 1.5))
        omega  = float(lo.get("omega", 0.5))
        return LoiterParams(center=center, radius=radius, omega=omega)

    def get_landing_params(self):
        ld = self.get_nested(["mission", "landing"], {}) or {}
        return LandingParams(
            final_altitude=float(ld.get("final_altitude", -0.1)),
            trigger_radius=float(ld.get("trigger_radius", 0.6)),
        )

    def get_target_params(self):
        tg = self.get_nested(["mission", "target"], {}) or {}
        return TargetParams(timeout=float(tg.get("timeout", 0.6)))

    def get_traj_params(self):
        tr = self.get_nested(["mission", "traj"], {}) or {}
        # accept both list/tuple or scalars → coerce to 3-tuple
        def _3(x, default):
            if x is None:
                return default
            if isinstance(x, (list, tuple)) and len(x) == 3:
                return (float(x[0]), float(x[1]), float(x[2]))
            # allow scalar to broadcast (not recommended but convenient)
            x = float(x)
            return (x, x, x)

        
        v_max = _3(tr.get("v_max", [2.0, 2.0, 1.0]), (2.0, 2.0, 1.0))
        a_max = _3(tr.get("a_max", [2.0, 2.0, 1.0]), (2.0, 2.0, 1.0))
        Tseg  = float(tr.get("segment_duration", 3.0))
        yawrate_max  = float(tr.get("yawrate_max", 1.2))
        yawacc_max  = float(tr.get("yawacc_max", 3.0))
        return TrajParams(v_max=v_max, a_max=a_max, segment_duration=Tseg,
                          yawrate_max=yawrate_max, yawacc_max=yawacc_max)

    def get_full_config(self):
        """
        Return a single object bundling mission path + quad phases + traj.
        """
        from hydro_mpc.utils.param_types import MissionConfig
        return MissionConfig(
            mission=self.get_mission(),
            takeoff=self.get_takeoff_params(),
            loiter=self.get_loiter_params(),
            landing=self.get_landing_params(),
            target=self.get_target_params(),
            traj=self.get_traj_params(),
        )

    
    def validate_mission(self) -> tuple[bool, str]:
        """
        Try parsing the mission; return (ok, message).
        Keeps get_full_config() unchanged — just a fast preflight check.
        """
        try:
            m = self.get_mission()
            # optional guardrails:
            if (m.common.speed is not None) and (m.common.speed < 0):
                return False, "common.speed must be >= 0"
            return True, f"OK ({type(m).__name__})"
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"



    def as_dict(self):
        """Return full parameter dictionary."""
        return self.params
    

    
