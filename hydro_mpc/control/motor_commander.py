import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from std_msgs.msg import Float32MultiArray, UInt8
from px4_msgs.msg import VehicleThrustSetpoint, VehicleTorqueSetpoint 

from hydro_mpc.utils.param_loader import ParamLoader
from hydro_mpc.control.control_allocator import ControlAllocator

from ament_index_python.packages import get_package_share_directory
import os
import signal, time


def vec3(x):
    a = np.asarray(x, dtype=float).ravel()
    if a.size == 1:
        a = np.full(3, float(a))
    elif a.size != 3:
        raise ValueError(f"vec3 expects scalar or length-3, got shape {a.shape}")
    return a  # shape (3,)

class MotorCommander(Node):
    def __init__(self):
        super().__init__('motor_commander')

        package_dir = get_package_share_directory('hydro_mpc')
        
        # Declare param with default
        self.declare_parameter('vehicle_param_file', 'crazyflie_param.yaml')
        self.declare_parameter('sitl_param_file', 'sitl_params.yaml')
        self.declare_parameter('cmd_timeout_ms', 250)               # fail to neutral if no cmd within this window
        self.declare_parameter('hover_thrust', 0.6)                 # normalized hover thrust magnitude (0..1)
        self.declare_parameter('thrust_limits', [0.0, 1.0])         # [min,max] magnitude before sign application
        self.declare_parameter('torque_limits', [0.3, 0.3, 0.3])    # per-axis absolute max (normalized)
        self.declare_parameter('thrust_slew_per_s', 1.5)            # max Δ per second (normalized)
        self.declare_parameter('torque_slew_per_s', 3.0)            # max Δ per second (normalized)
        self.declare_parameter('idle_nav_state', 1)             # keep your current gating by default


        vehicle_param_file = self.get_parameter('vehicle_param_file').get_parameter_value().string_value
        sitl_param_file = self.get_parameter('sitl_param_file').get_parameter_value().string_value

        sitl_yaml_path = os.path.join(package_dir, 'config', 'sitl', sitl_param_file)
        vehicle_yaml_path = os.path.join(package_dir, 'config', 'vehicle_parameters', vehicle_param_file)

        # Load parameters
        sitl_yaml = ParamLoader(sitl_yaml_path)
        vehicle_yaml = ParamLoader(vehicle_yaml_path)

        # Vehicle parameters
        self.vehicle_params = vehicle_yaml.get_vehicle_params()

        # Topics
        nav_state_topic = sitl_yaml.get_topic("nav_state_topic")
        control_command_topic = sitl_yaml.get_topic("control_command_topic")
        thrust_setpoints_topic = sitl_yaml.get_topic("thrust_setpoints_topic")
        torque_setpoints_topic = sitl_yaml.get_topic("torque_setpoints_topic")

        # QOS Options
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,  # <-- latch last message
        )

        
        # sub
        self.create_subscription(Float32MultiArray, control_command_topic, self.control_cmd_callback, 10)
        self.create_subscription(UInt8, nav_state_topic, self._on_nav_state, qos)

        # pub
        self.thrust_pub = self.create_publisher(VehicleThrustSetpoint, thrust_setpoints_topic, 1)
        self.torque_pub = self.create_publisher(VehicleTorqueSetpoint, torque_setpoints_topic, 1)


        # initial states
        self.latest_motor_cmd = [0.0, 0.0, 0.0, 0.0]

        self.latest_thrust_cmd = VehicleThrustSetpoint()
        self.latest_torque_cmd = VehicleTorqueSetpoint()

        self.normalized_torque_and_thrust = [0.0, 0.0, 0.0, 0.0]

        self._last_cmd_time_us = 0

        self._last_ntt = np.zeros(4, dtype=float)  # [Tx, Ty, Tz, thrust], shape (4,)

        self.allow_commands = False  # start in IDLE

        self.nav_state = 1 # start in IDLE

        
        
        # static allocation matrices
        self.rotor_velocities_to_torques_and_thrust, self.torques_and_thrust_to_rotor_velocities = \
        ControlAllocator.compute_allocation_matrices(self.vehicle_params.num_of_arms, self.vehicle_params.thrust_constant, self.vehicle_params.moment_constant, self.vehicle_params.arm_length)

        # Typical X-configuration
        angles_deg = [135, 45, 315, 225]
        spin_dirs = [-1, 1, -1, 1]

        _, self.throttles_to_normalized_torques_and_thrust = ControlAllocator.generate_mixing_matrices(
            0.225, 1.00, 1.00, angles_deg, spin_dirs
        )

        # self.get_logger().info("[mixing_matrix] =\n" + np.array2string(mixing_matrix, precision=10, suppress_small=True))
        # self.get_logger().info("[mixing_matrix_inv] =\n" + np.array2string(self.throttles_to_normalized_torques_and_thrust, precision=10, suppress_small=True))

        # Timers
        self.dt = 0.01
        self.motor_command_timer = self.create_timer(self.dt, self.motor_command_timer_callback) # 100 Hz

        self.get_logger().info("MotorCommander with Offboard control started")


     # ---------- callbacks ----------
    def _on_nav_state(self, msg: UInt8):
        #is_idle = int(self.get_parameter('idle_nav_state').get_parameter_value().integer_value)
        self.nav_state = int(msg.data)
        self.allow_commands = True #(self.nav_state != 1)
        if not self.allow_commands:
             self._set_latest_to_neutral()

    # ---------- timers ----------
    def motor_command_timer_callback(self):
        now_us = int(self.get_clock().now().nanoseconds / 1000)

        # watchdog: no fresh cmd -> neutral
        timeout_ms = int(self.get_parameter('cmd_timeout_ms').get_parameter_value().integer_value)
        is_timeout = (now_us - self._last_cmd_time_us > timeout_ms * 1000)

        run_motors = self.allow_commands #and is_timeout 


        if not run_motors:
            
            self._set_latest_to_neutral()
            if is_timeout and not (self.nav_state == 1) :
                self.get_logger().warn("control stream timeout -> neutral", throttle_duration_sec=1.0)

        self.thrust_pub.publish(self.latest_thrust_cmd)
        self.torque_pub.publish(self.latest_torque_cmd)

        
    def control_cmd_callback(self, msg):

        if not self.allow_commands:
            return

        now_us = int(self.get_clock().now().nanoseconds / 1000)
        self._last_cmd_time_us = now_us

        # Expect [thrust, Tx, Ty, Tz]
        if len(msg.data) < 4:
            self.get_logger().error("control_cmd too short; need 4 floats [thrust,Tx,Ty,Tz]")
            return
        if not np.all(np.isfinite(msg.data[:4])):
            self.get_logger().error("control_cmd contains non-finite values, ignoring")
            return

        thrust_cmd = float(msg.data[0])
        torque_cmd = np.array(msg.data[1:4], dtype=float)


        #self.get_logger().info(f"thrust= {thrust_cmd} | torque= {torque_cmd}")

        torque_thrust_vec = np.concatenate((torque_cmd, [thrust_cmd])).reshape((4, 1))
        omega_sq = self.torques_and_thrust_to_rotor_velocities @ torque_thrust_vec
        omega_sq = np.clip(omega_sq, 0.0, None)  # Ensures non-negative values
        # self.get_logger().info(f"torque_thrust_vec= {torque_thrust_vec}")
        # self.get_logger().info(f"omega_sq= {omega_sq}")

        omega = np.sqrt(omega_sq)

        throttles = np.clip(omega / self.vehicle_params.max_rotor_speed, 0.0, 1.0)

        # self.get_logger().info(f"omega= {omega} | throttles= {throttles}")

        ntt = (self.throttles_to_normalized_torques_and_thrust @ throttles).astype(float).reshape(4)  # <-- (4,)

        ntt = np.asarray(ntt, dtype=float).ravel()   # shape (4,)

        # clamp & slew
        tmax = np.asarray(self.get_parameter('torque_limits').get_parameter_value().double_array_value or [0.3,0.3,0.3], dtype=float)

        # tmax can be scalar or len-3; normalize to (3,)
        tmax_arr = np.asarray(tmax, dtype=float).ravel()
        if tmax_arr.size == 1:
            tmax_vec = np.repeat(tmax_arr, 3)
        elif tmax_arr.size == 3:
            tmax_vec = tmax_arr
        else:
            raise ValueError("torque_limits must be scalar or length-3")

        ntt[0:3] = np.clip(ntt[0:3], -tmax_vec, tmax_vec)   # 1-D slice

  
        hmin, hmax = self.get_parameter('thrust_limits').get_parameter_value().double_array_value or [0.0, 1.0]
        ntt[3] = float(np.clip(ntt[3], hmin, hmax))

        # slew limiting
        t_slew = float(self.get_parameter('torque_slew_per_s').get_parameter_value().double_value)
        h_slew = float(self.get_parameter('thrust_slew_per_s').get_parameter_value().double_value)

        max_step_t = t_slew * self.dt
        max_step_t_arr = np.asarray(max_step_t, dtype=float).ravel()
        if max_step_t_arr.size == 1:
            max_step_t_vec = np.repeat(max_step_t_arr, 3)
        elif max_step_t_arr.size == 3:
            max_step_t_vec = max_step_t_arr
        else:
            raise ValueError("torque_slew_per_s must be scalar or length-3")

        lb = self._last_ntt[0:3] - max_step_t_vec          # all (3,)
        ub = self._last_ntt[0:3] + max_step_t_vec
        ntt[0:3] = np.clip(ntt[0:3], lb, ub)               # 1-D slice

        max_step_h = h_slew * self.dt


        ntt[3] = float(np.clip(ntt[3], self._last_ntt[3] - max_step_h, self._last_ntt[3] + max_step_h))


        # --- Store for next tick & for messages ---
        self._last_ntt = ntt.copy()                        # (4,)
        self.normalized_torque_and_thrust = ntt            # (4,)


        #self.get_logger().info(f"normalized_torque_and_thrust= {self.normalized_torque_and_thrust} ")


        # Prepare thrust message
        self.latest_thrust_cmd.timestamp = now_us
        self.latest_thrust_cmd.xyz[0] = 0.0
        self.latest_thrust_cmd.xyz[1] = 0.0
        # PX4 NED convention: up-thrust is negative along body z
        self.latest_thrust_cmd.xyz[2] = -float(self.normalized_torque_and_thrust[3])


        # Prepare torque message
        self.latest_torque_cmd.timestamp = now_us
        self.latest_torque_cmd.xyz[0] = float(self.normalized_torque_and_thrust[0])
        self.latest_torque_cmd.xyz[1] = float(self.normalized_torque_and_thrust[1])
        self.latest_torque_cmd.xyz[2] = float(self.normalized_torque_and_thrust[2])


    # ---------- neutral helpers ----------
    def _set_latest_to_neutral(self):

        hover = float(self.get_parameter('hover_thrust').get_parameter_value().double_value)
        self.latest_thrust_cmd.xyz[0] = 0.0
        self.latest_thrust_cmd.xyz[1] = 0.0
        self.latest_thrust_cmd.xyz[2] = -hover

        self.latest_torque_cmd.xyz[0] = 0.0
        self.latest_torque_cmd.xyz[1] = 0.0
        self.latest_torque_cmd.xyz[2] = 0.0
        

    def _publish_neutral_once(self):
        # call only while the context is still valid
        now = int(self.get_clock().now().nanoseconds / 1000)
        self.latest_thrust_cmd.timestamp = now
        self.latest_torque_cmd.timestamp = now
        
        self.thrust_pub.publish(self.latest_thrust_cmd)
        self.torque_pub.publish(self.latest_torque_cmd)


    


def main(args=None):
    rclpy.init(args=args)
    node = MotorCommander()

    exe = SingleThreadedExecutor()
    exe.add_node(node)

    # trap signals so rclpy doesn't tear the context down before we burst neutral
    shutdown = {"req": False}
    def _sig_handler(signum, frame):
        shutdown["req"] = True
    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    try:
        # spin manually so we can intercept shutdown
        while rclpy.ok() and not shutdown["req"]:
            exe.spin_once(timeout_sec=0.1)
    finally:
        # 1) immediately switch desired command to neutral
        node._set_latest_to_neutral()

        # 2) publish a short neutral burst while the context is STILL valid
        deadline = time.time() + 0.3  # ~300 ms
        while time.time() < deadline and rclpy.ok():
            node._publish_neutral_once()
            exe.spin_once(timeout_sec=0.0)  # flush any pending work
            time.sleep(0.02)               # ~50 Hz

        # 3) clean teardown
        exe.shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()