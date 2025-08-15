import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import numpy as np

from std_msgs.msg import Float32MultiArray
from px4_msgs.msg import VehicleOdometry, TimesyncStatus
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped

from hydro_mpc.utils.mpc_solver import MPCSolver
from hydro_mpc.guidance.min_jerk_trajectory_generator import MinJerkTrajectoryGenerator


from hydro_mpc.utils.param_loader import ParamLoader

from ament_index_python.packages import get_package_share_directory
import os

import signal




class MpcControllerNode(Node):
    def __init__(self):
        super().__init__('mpc_controller_node')
        
        self.declare_parameter('vehicle_param_file', 'crazyflie_param.yaml')
        self.declare_parameter('controller_param_file', 'mpc_crazyflie.yaml')
        self.declare_parameter('sitl_param_file', 'sitl_param.yaml')
        self.declare_parameter('world_frame', 'map')
        self.declare_parameter('trajectory_topic', '/trajectory') 

        vehicle_param_file = self.get_parameter('vehicle_param_file').get_parameter_value().string_value
        controller_param_file = self.get_parameter('controller_param_file').get_parameter_value().string_value
        sitl_param_file = self.get_parameter('sitl_param_file').get_parameter_value().string_value
        self.world_frame = self.get_parameter('world_frame').get_parameter_value().string_value
        trajectory_topic = self.get_parameter('trajectory_topic').get_parameter_value().string_value
        
        package_dir = get_package_share_directory('hydro_mpc')
        
        sitl_yaml_path = os.path.join(package_dir, 'config', 'sitl', sitl_param_file)
        vehicle_yaml_path = os.path.join(package_dir, 'config', 'vehicle_parameters', vehicle_param_file)
        controller_yaml_path = os.path.join(package_dir, 'config', 'controller', controller_param_file)


        # Load parameters
        sitl_yaml = ParamLoader(sitl_yaml_path)
        vehicle_yaml = ParamLoader(vehicle_yaml_path)
        controller_yaml = ParamLoader(controller_yaml_path)

        # Topic names
        odom_topic = sitl_yaml.get_topic("odometry_topic")
        timesync_topic = sitl_yaml.get_topic("status_topic")
        control_cmd_topic = sitl_yaml.get_topic("control_command_topic")
        target_state_topic = sitl_yaml.get_topic("target_state_topic")

        # Controller parameters
        self.control_params = controller_yaml.get_control_params()
        # UAV parameters
        self.vehicle_params = vehicle_yaml.get_vehicle_params()



        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Sub
        self.sub_odom = self.create_subscription(
            VehicleOdometry, 
            odom_topic, 
            self._odom_callback, qos_profile)
        
        self.sub_sync = self.create_subscription(
            TimesyncStatus, 
            timesync_topic, 
            self._sync_callback, qos_profile)
        
        self.sub_target_state = self.create_subscription(
            Odometry, 
            target_state_topic, 
            self._target_state_callback, qos_profile)
        
        # Pub
        self.motor_cmd_pub = self.create_publisher(
            Float32MultiArray, 
            control_cmd_topic, 
            10)
                        
        self.trajectory_pub = self.create_publisher(
            Path, 
            trajectory_topic, 
            10)

        # State variables
        self.t0 = None
        self.t_sim = 0.0
        self.px4_timestamp = 0
        self.odom_ready = False

        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.q = np.zeros(4)
        self.rpy = np.zeros(3)
        self.omega_body = np.zeros(3) # Angular velocity in body frame

        # MPC predicted path 
        self.last_pred_X = None     # np.ndarray, shape (NX, N+1), world frame
        self.last_ref_X  = None     # np.ndarray, shape (NX, N+1), world frame

        # Trajectory Generator
        a_max = np.array([2.0, 2.0, 1.0])
        self.traj_generator = MinJerkTrajectoryGenerator(a_max)

        self.target_pos = None
        self.target_vel = None
        self.target_rpy = None
        self.target_omega = None

        self.target_pos = np.array([0.0, 0.0, -2.0])
        self.target_vel = np.array([0.0, 0.0, 0.0])

        self._target_state_received = False

        self.Ts = 1.0 / self.control_params.frequency
        
        # MPC + Timer
        self.mpc = MPCSolver(self.control_params, self.vehicle_params, debug=False)
        self.timer_mpc = self.create_timer(self.Ts, self._control_loop)  

        #self.timer_traj_update = self.create_timer(1.0, self._update_ref_trajectory)

        self.get_logger().info("MPC Controller Node initialized.")


        # Safety caps (per-axis)
        self.err_pos_cap = np.array([0.5, 0.5, 0.5])   # meters of allowed position error into MPC
        self.err_vel_cap = np.array([1.0, 1.0, 1.0])   # m/s  of allowed velocity error into MPC

        # Rate limits (per-axis)
        self.ref_v_cap   = np.array([1.0, 1.0, 0.5])   # m/s   max change of position reference
        self.ref_a_cap   = np.array([2.0, 2.0, 1.0])   # m/s^2 max change of velocity reference

        # Running state for slew limiter
        self.prev_p_cmd = None
        self.prev_v_cmd = None

        self.generated = False


    def _timing(self, stamp_us):
        t = stamp_us * 1e-6
        if self.t0 is None:
            self.t0 = t
        return t - self.t0
    
    def _reset_t0(self, t_now):
        self.t0 = t_now

    def _odom_callback(self, msg):
        self.px4_timestamp = msg.timestamp
        t_now = self._timing(msg.timestamp)
        self.t_sim = t_now

        # Position
        self.pos = np.array(msg.position)

        # Attitude (Quaternion to Euler)
        self.q[0],self.q[1],self.q[2],self.q[3] = msg.q
        self.rpy[2],self.rpy[1],self.rpy[0] = self._quat_to_eul(self.q)
        
        #rot_mat = eul2rotm_py(self.rpy)
        
        # World-frame Velocity (transform from body frame)
        self.vel = np.array(msg.velocity)

        # Body-frame Angular Velocity
        self.omega_body = np.array(msg.angular_velocity)

        if not self.odom_ready:
            self.get_logger().warn("First odometry message is received...")
            self.odom_ready = True
            return
        
        self.odom_ready = True

    def _quat_to_eul(self, q_xyzw):
        # PX4: [w, x, y, z]
        from scipy.spatial.transform import Rotation as R
        _r = R.from_quat([q_xyzw[1], q_xyzw[2], q_xyzw[3], q_xyzw[0]])
        return _r.as_euler('ZYX', degrees=False)

    def _sync_callback(self, msg):
        self.px4_timestamp = msg.timestamp

    def _target_state_callback(self, msg):
        #msg_time = msg.header.stamp
        #target_frame_id = msg.header.frame_id

        # Position
        self.target_pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])
        # Linear Velocity
        self.target_vel = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z])
        # Angular Velocity
        self.target_omega = np.array([msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z])

        # Attitude (Quaternion to Euler)
        _q = np.zeros(4)
        _q[0] = msg.pose.pose.orientation.x
        _q[1] = msg.pose.pose.orientation.y
        _q[2] = msg.pose.pose.orientation.z
        _q[3] = msg.pose.pose.orientation.w

        self.target_rpy[2],self.target_rpy[1],self.target_rpy[0] = self._quat_to_eul(_q)


        self._target_state_received = True

    def _safe_ref(self, p_ref: np.ndarray, v_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Clamp error vs current state and slew-limit vs previous commanded reference.
        Returns (p_cmd, v_cmd) to pass into MPC.
        """
        # 1) Error clamp vs current state
        e_p = p_ref - self.pos
        e_v = v_ref - self.vel
        e_p = np.clip(e_p, -self.err_pos_cap, self.err_pos_cap)
        e_v = np.clip(e_v, -self.err_vel_cap, self.err_vel_cap)

        p_cmd = self.pos + e_p
        v_cmd = self.vel + e_v

        # 2) Slew limit vs last command
        if self.prev_p_cmd is None:
            # First call: start from current state (no jump)
            self.prev_p_cmd = self.pos.copy()
            self.prev_v_cmd = self.vel.copy()

        dp_max = self.ref_v_cap * self.Ts         # meters per tick
        dv_max = self.ref_a_cap * self.Ts         # m/s per tick

        dp = p_cmd - self.prev_p_cmd
        dv = v_cmd - self.prev_v_cmd

        p_cmd = self.prev_p_cmd + np.clip(dp, -dp_max, dp_max)
        v_cmd = self.prev_v_cmd + np.clip(dv, -dv_max, dv_max)

        # store for next tick
        self.prev_p_cmd = p_cmd
        self.prev_v_cmd = v_cmd
        return p_cmd, v_cmd
    

    def _update_ref_trajectory(self):


        if all(v is not None for v in [self.target_pos, self.target_vel]):

            # Construct the 12-dimensional state vector
            _x0 = np.concatenate([self.pos, self.vel, self.rpy, self.omega_body])

            _current_state = np.concatenate([_x0[0:6], np.zeros(3)])
            _target_state = np.concatenate([self.target_pos, self.target_vel, np.zeros(3)])

            self.traj_generator.generate_tracking_trajectory(_current_state, _target_state)


            self._reset_t0(self.t_sim)

    
    def _control_loop(self):
        """
        Runs at a slower rate, solving the optimization problem.
        """
        if not self.odom_ready:
            self.get_logger().warn("Waiting for odometry...")
            return

        
        # Construct the 12-dimensional state vector
        _x0 = np.concatenate([self.pos, self.vel, self.rpy, self.omega_body])

        
        if not self.generated:
            self._update_ref_trajectory()
            self.generated = True

        
        # Get reference trajectory point
        #p_ref, v_ref = eval_traj_docking(self.t_sim)
        p_ref, v_ref, _ = self.traj_generator.get_ref_at_time(self.t_sim) 

        # Apply rate limiter safety 
        p_cmd, v_cmd = self._safe_ref(p_ref, v_ref)

        # Build reference vector for MPC (setpoint-tracking variant)
        _xref_h = np.concatenate([p_cmd, v_cmd, np.zeros(6)])

        # Solve the MPC problem
        _u_mpc, _X_opt, _ = self.mpc.solve(_x0, _xref_h)  # [thrust, tau_phi, tau_theta, tau_psi]

        self.last_pred_X = _X_opt

        _u_mpc[1] = -_u_mpc[1]

        # self.get_logger().info(f"p_ref= {p_ref} | diff= {np.array(p_ref) - self.pos}")
        # self.get_logger().info(f"v_ref= {p_ref} | diff= {np.array(v_ref) - self.vel}")

        # self.get_logger().info(f"roll= {self.rpy[0]*180/np.pi} | diff= {(0.0 - self.rpy[0])*180/np.pi}")
        # self.get_logger().info(f"pitch= {self.rpy[1]*180/np.pi} | diff= {(0.0 - self.rpy[1])*180/np.pi}")
        # self.get_logger().info(f"yaw= {self.rpy[2]*180/np.pi} | diff= {(0.0 - self.rpy[2])*180/np.pi}")
   
        # Publish
        msg = Float32MultiArray()
        msg.data = _u_mpc.tolist()
        self.motor_cmd_pub.publish(msg)

        self._publish_trajectory()




    def _publish_trajectory(self):
        # Decide what to draw: prefer predicted trajectory, else reference
        X = self.last_pred_X 
        if X is None:
            return

        xs = X[0, :]
        ys = -X[1, :]
        zs = -X[2, :]

        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.world_frame

        poses = []
        for x, y, z in zip(xs, ys, zs):
            p = PoseStamped()
            p.header = msg.header
            p.pose.position.x = float(x)
            p.pose.position.y = float(y)
            p.pose.position.z = float(z)

            p.pose.orientation.x = 0.0
            p.pose.orientation.y = 0.0
            p.pose.orientation.z = 0.0
            p.pose.orientation.w = 1.0

            poses.append(p)

        msg.poses = poses
        self.trajectory_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = MpcControllerNode()

    def signal_handler(sig, frame):
        node.get_logger().info("Shutdown signal received. Cleaning up...")
        node.destroy_node()
        rclpy.shutdown()

    # Register the signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()