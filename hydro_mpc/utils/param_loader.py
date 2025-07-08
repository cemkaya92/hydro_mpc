import yaml
import os

from hydro_mpc.utils.param_types import UAVParams, MPCParams

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

    def get_mpc_params(self) -> MPCParams:

        return MPCParams(
            horizon=self.get("mpc_parameters", {}).get("horizon", 1.5),
            N=self.get("mpc_parameters", {}).get("N", 20),
            frequency=self.get("mpc_parameters", {}).get("frequency", 100.0),
            Q=self.get("mpc_parameters", {}).get("Q", [40.0, 40.0, 40.0, 4.0, 4.0, 4.0, 2.0, 2.0, 2.0, 0.5, 0.5, 0.5]),
            R=self.get("mpc_parameters", {}).get("R", [0.1, 1.0, 1.0, 1.0])
        )
    
    def get_uav_params(self) -> UAVParams:

        params = self.get("uav_parameters", {})

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

        return UAVParams(
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
    

    def as_dict(self):
        """Return full parameter dictionary."""
        return self.params
