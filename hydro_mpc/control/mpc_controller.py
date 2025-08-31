import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import numpy as np

from std_msgs.msg import Float32MultiArray, UInt8
from px4_msgs.msg import VehicleOdometry, TimesyncStatus, TrajectorySetpoint6dof
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

from hydro_mpc.utils.mpc_solver import MPCSolver


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
        self.declare_parameter('mpc_trajectory_topic', '/trajectory') 

        vehicle_param_file = self.get_parameter('vehicle_param_file').get_parameter_value().string_value
        controller_param_file = self.get_parameter('controller_param_file').get_parameter_value().string_value
        sitl_param_file = self.get_parameter('sitl_param_file').get_parameter_value().string_value
        self.world_frame = self.get_parameter('world_frame').get_parameter_value().string_value
        mpc_trajectory_topic = self.get_parameter('mpc_trajectory_topic').get_parameter_value().string_value
        
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
        trajectory_sub_topic = sitl_yaml.get_topic("command_traj_topic")
        nav_state_topic = sitl_yaml.get_topic("nav_state_topic")

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

        plan_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,  # <-- ask for the stored last sample
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
        
        self.sub_commanded_trajectory = self.create_subscription(
            TrajectorySetpoint6dof, 
            trajectory_sub_topic, 
            self._trajectory_callback, 10)
        
        self.sub_nav_state = self.create_subscription(
            UInt8, 
            nav_state_topic, 
            self._on_nav_state, plan_qos)
        
        # Pub
        self.motor_cmd_pub = self.create_publisher(
            Float32MultiArray, 
            control_cmd_topic, 
            10)
                        
        self.trajectory_pub = self.create_publisher(
            Path, 
            mpc_trajectory_topic, 
            10)

        # State variables
        self.t0 = None
        self.t_sim = 0.0
        self.px4_timestamp = 0
        self._odom_ready = False
        self._trajectory_ready = False

        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.q = np.zeros(4)
        self.rpy = np.zeros(3)
        self.omega_body = np.zeros(3) # Angular velocity in body frame

        # MPC predicted path 
        self.last_pred_X = None     # np.ndarray, shape (NX, N+1), world frame
        self.last_ref_X  = None     # np.ndarray, shape (NX, N+1), world frame

        # Trajectory 
        self._x_ref = None

        self.plan_type = None
        self.plan_t0_us = None
        self.plan_data = {}     # dict: coeffs/state0/speed/yaw_rate/heading/duration/repeat/distance
 
        self.nav_state = 1  # IDLE by default
        
        # MPC + Timer
        self.Ts = 1.0 / self.control_params.frequency
        self.mpc = MPCSolver(self.control_params, self.vehicle_params, debug=False)
        self.timer_mpc = self.create_timer(self.Ts, self._control_loop)  


        self.get_logger().info("MPC Controller Node initialized.")

    
    # ---------- callbacks ----------
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

        if not self._odom_ready:
            self.get_logger().warn("First odometry message is received...")
            self._odom_ready = True
            return
        
        self._odom_ready = True

    def _sync_callback(self, msg):
        self.px4_timestamp = msg.timestamp

    def _trajectory_callback(self, msg):

        #last_traj_received_stamp = msg.timestamp

        _pos = msg.position
        _vel = msg.velocity
        #_acc = msg.acceleration

        self._x_ref = np.concatenate([_pos, _vel, np.zeros(3), np.zeros(3)]) # pos, vel, att, omega

        self._trajectory_ready = True

    def _on_nav_state(self, msg: UInt8):
        self.nav_state = int(msg.data)
        if self.nav_state == 0:          # 0 = IDLE
            self._trajectory_ready = False
            

    # ---------- main loop ----------
    def _control_loop(self):
        """
        Runs at a slower rate, solving the optimization problem.
        """
        if not self._odom_ready:
            self.get_logger().warn("Waiting for odometry...")
            return

        
        # Construct the 12-dimensional state vector
        _x0 = np.concatenate([self.pos, self.vel, self.rpy, self.omega_body])

        
        if not self._trajectory_ready and self.nav_state != 2:
            # publish hold/zero command and bail
            msg = Float32MultiArray(); 
            msg.data = [0.0, 0.0, 0.0, 0.0]
            self.motor_cmd_pub.publish(msg)
            return

        #self.get_logger().info(f"_x0= {_x0} | _x_ref= {self._x_ref}")

        # Solve the MPC problem
        _u_mpc, _X_opt, _ = self.mpc.solve(_x0, self._x_ref)  # [thrust, tau_phi, tau_theta, tau_psi]

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

        # If state includes attitude (roll, pitch, yaw), pull them out
        have_att = X.shape[0] >= 9
        if have_att:
            rolls  = X[6, :]
            pitchs = X[7, :]
            yaws   = X[8, :]

        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.world_frame

        poses = []

        if have_att:
            it = zip(xs, ys, zs, rolls, pitchs, yaws)
        else:
            it = zip(xs, ys, zs)

        for item in it:
            if have_att:
                x, y, z, r, p, yy = item
            else:
                x, y, z = item

            p = PoseStamped()
            p.header = msg.header
            p.pose.position.x = float(x)
            p.pose.position.y = float(y)
            p.pose.position.z = float(z)

            if have_att:
                qx, qy, qz, qw = self._rpy_to_quat_map(r, p, yy)
                p.pose.orientation.x = float(qx)
                p.pose.orientation.y = float(qy)
                p.pose.orientation.z = float(qz)
                p.pose.orientation.w = float(qw)
            else:
                # Fallback: identity orientation
                p.pose.orientation.x = 0.0
                p.pose.orientation.y = 0.0
                p.pose.orientation.z = 0.0
                p.pose.orientation.w = 1.0

            poses.append(p)

        msg.poses = poses
        self.trajectory_pub.publish(msg)


    # ---------- helpers ----------
    def _timing(self, stamp_us):
        t = stamp_us * 1e-6
        if self.t0 is None:
            self.t0 = t
        return t - self.t0
    
    def _reset_t0(self, t_now):
        self.t0 = t_now

    def _quat_to_eul(self, q_xyzw):
        # PX4: [w, x, y, z]
        from scipy.spatial.transform import Rotation as R
        _r = R.from_quat([q_xyzw[1], q_xyzw[2], q_xyzw[3], q_xyzw[0]])
        return _r.as_euler('ZYX', degrees=False)
    
    def _rpy_to_quat_map(self, roll: float, pitch: float, yaw: float):
        """
        Convert MPC RPY to a quaternion consistent with the position mapping:
        positions use (x, -y, -z), which equals a 180° rotation about X.
        Under this change of basis: roll -> roll, pitch -> -pitch, yaw -> -yaw.
        Returns [x,y,z,w] for geometry_msgs orientation fields.
        """
        from scipy.spatial.transform import Rotation as R
        r = float(roll)
        p = float(-pitch)
        y = float(-yaw)
        return R.from_euler('ZYX', [y, p, r]).as_quat()  # SciPy returns [x,y,z,w]

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