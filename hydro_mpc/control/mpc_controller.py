import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import numpy as np

from std_msgs.msg import Float32MultiArray
from px4_msgs.msg import VehicleOdometry, TimesyncStatus

from hydro_mpc.utils.mpc_solver import MPCSolver
from hydro_mpc.guidance.trajectory import eval_traj_docking
from hydro_mpc.utils.first_order_filter import FirstOrderFilter

from hydro_mpc.utils.ploter import Logger

from hydro_mpc.utils.param_loader import ParamLoader

from ament_index_python.packages import get_package_share_directory
import os

import signal




class MpcControllerNode(Node):
    def __init__(self):
        super().__init__('mpc_controller_node')
        
        self.declare_parameter('uav_param_file', 'crazyflie_param.yaml')
        self.declare_parameter('mpc_param_file', 'mpc_crazyflie.yaml')

        uav_param_file = self.get_parameter('uav_param_file').get_parameter_value().string_value
        mpc_param_file = self.get_parameter('mpc_param_file').get_parameter_value().string_value

        package_dir = get_package_share_directory('hydro_mpc')
        
        sitl_yaml_path = os.path.join(package_dir, 'config', 'sitl', 'sitl_params.yaml')
        uav_yaml_path = os.path.join(package_dir, 'config', 'uav_parameters', uav_param_file)
        mpc_yaml_path = os.path.join(package_dir, 'config', 'controller', mpc_param_file)


        # Load parameters
        sitl_yaml = ParamLoader(sitl_yaml_path)
        uav_yaml = ParamLoader(uav_yaml_path)
        mpc_yaml = ParamLoader(mpc_yaml_path)

        # Topic names
        odom_topic = sitl_yaml.get_topic("odometry_topic")
        timesync_topic = sitl_yaml.get_topic("status_topic")
        mpc_cmd_topic = sitl_yaml.get_topic("mpc_command_topic")

        # Controller parameters
        mpc_params = mpc_yaml.get_mpc_params()
        # UAV parameters
        uav_params = uav_yaml.get_uav_params()



        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Sub/Pub
        self.sub_odom = self.create_subscription(
            VehicleOdometry, 
            odom_topic, 
            self.odom_callback, qos_profile)
        
        self.sub_sync = self.create_subscription(
            TimesyncStatus, 
            timesync_topic, 
            self.sync_callback, qos_profile)
        #self.pub_motor = self.create_publisher(ActuatorMotors, '/fmu/in/actuator_motors', 10)
        self.motor_cmd_pub = self.create_publisher(
            Float32MultiArray, 
            mpc_cmd_topic, 
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


        # First-order filters for smoother control outputs
        self.thrust_filter = FirstOrderFilter(alpha=0.8)
        self.roll_torque_filter = FirstOrderFilter(alpha=0.5)
        self.pitch_torque_filter = FirstOrderFilter(alpha=0.5)
        self.yaw_torque_filter = FirstOrderFilter(alpha=0.9)

        # Data logging
        self.logger = Logger()
        
        # MPC + Timer
        self.mpc = MPCSolver(mpc_params, uav_params, debug=False)
        self.timer_mpc = self.create_timer(1.0 / mpc_params.frequency, self.mpc_loop)  # 100 Hz

        self.get_logger().info("MPC Controller Node initialized.")


    def _timing(self, stamp_us):
        t = stamp_us * 1e-6
        if self.t0 is None:
            self.t0 = t
        return t - self.t0

    def odom_callback(self, msg):
        self.px4_timestamp = msg.timestamp
        t_now = self._timing(msg.timestamp)
        self.t_sim = t_now

        # Position
        self.pos = np.array(msg.position)

        # Attitude (Quaternion to Euler)
        self.q[0],self.q[1],self.q[2],self.q[3] = msg.q
        self.rpy[2],self.rpy[1],self.rpy[0] = self.quat_to_eul(self.q)
        
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

    def quat_to_eul(self, q_xyzw):
        # PX4: [w, x, y, z]
        from scipy.spatial.transform import Rotation as R
        r = R.from_quat([q_xyzw[1], q_xyzw[2], q_xyzw[3], q_xyzw[0]])
        return r.as_euler('ZYX', degrees=False)

    def sync_callback(self, msg):
        self.px4_timestamp = msg.timestamp

    def mpc_loop(self):
        """
        Runs at a slower rate, solving the optimization problem.
        """
        if not self.odom_ready:
            self.get_logger().warn("Waiting for odometry...")
            return
        
        # Construct the 12-dimensional state vector
        x0 = np.concatenate([self.pos, self.vel, self.rpy, self.omega_body])

        # Get reference trajectory point
        p_ref, v_ref = eval_traj_docking(self.t_sim)


        # Reference for attitude and angular rates is zero
        x_ref = np.concatenate([p_ref, v_ref, np.zeros(6)])

        # Solve the MPC problem
        u_mpc = self.mpc.solve(x0, x_ref)  # [thrust, tau_phi, tau_theta, tau_psi]

        # u_mpc = np.array([0.0, 0.0, 0.0, 0.0])

        u_mpc[1] = -u_mpc[1]
        # u_mpc[2] = 0.0
        # u_mpc[3] = 0.0
                         
        # roll_command =  self.roll_p_gain*(0.0 - self.rpy[0]) + self.roll_d_gain * (0.0 - self.omega_body[0])
        # pitch_command =  self.pitch_p_gain*(0.0 - self.rpy[1]) + self.pitch_d_gain * (0.0 - self.omega_body[1])          
        # yaw_command =  self.yaw_p_gain*(0.0 - self.rpy[2]) + self.yaw_d_gain * (0.0 - self.omega_body[2])

        # self.get_logger().info(f"p_ref= {p_ref} | diff= {np.array(p_ref) - self.pos}")
        # self.get_logger().info(f"v_ref= {p_ref} | diff= {np.array(v_ref) - self.vel}")

        # self.get_logger().info(f"roll= {self.rpy[0]*180/np.pi} | diff= {(0.0 - self.rpy[0])*180/np.pi}")
        # self.get_logger().info(f"pitch= {self.rpy[1]*180/np.pi} | diff= {(0.0 - self.rpy[1])*180/np.pi}")
        # self.get_logger().info(f"yaw= {self.rpy[2]*180/np.pi} | diff= {(0.0 - self.rpy[2])*180/np.pi}")
        #u_mpc[0] = (-np.sqrt(u_mpc[0] / (4 * KF_SIM))) / MAX_OMEGA_SIM
        # yaw_command = (yaw_command / self.max_torque)

        # roll_command = (roll_command / self.max_torque)
        # pitch_command = (pitch_command / self.max_torque)

        # thrust_command = (-np.sqrt(0.027*9.8066 / (4 * KF_SIM)) / MAX_OMEGA_SIM )
    

        # u_mpc[0] = self.thrust_filter.filter(thrust_command)
        # u_mpc[1] = self.roll_torque_filter.filter(roll_command)
        # u_mpc[2] = self.pitch_torque_filter.filter(pitch_command)
        # u_mpc[3] = self.yaw_torque_filter.filter(yaw_command)

        self.logger.log(self.t_sim, self.pos, self.vel, self.rpy, p_ref, v_ref, u_mpc)

        # Normalize thrust by max_thrust
        #u_mpc[0] /= MAX_OMEGA_SIM
        # Normalize torques by max_thrust
        # u_mpc[1] /= self.max_torque
        # u_mpc[2] /= self.max_torque
        # u_mpc[3] /= self.max_torque_yaw

        # u_mpc[1] = 0*u_mpc[1]
        # u_mpc[2] = 0*u_mpc[2]
        # u_mpc[3] = 0*u_mpc[3]

        # Clip all values between -1 and 1
        # u_mpc = np.clip(u_mpc, -1.0, 1.0)

        #u_mpc = np.array([-0.027*9.81, 0.0, 0.0, 0.0])
        #u_mpc = np.array([0.0, 0.0, 0.0, -0.0001])

        # Publish
        msg = Float32MultiArray()
        msg.data = u_mpc.tolist()
        self.motor_cmd_pub.publish(msg)

        # msg = VehicleRatesSetpoint()
        # msg.timestamp = self.px4_timestamp
        # msg.roll = 0.0
        # msg.pitch = 0.0
        # msg.yaw = 0.0
        # msg.thrust_body = [0.0,0.0,-0.5]
        #self.rates_cmd_pub.publish(msg)

        #self.get_logger().info(f"t={self.t_sim:.2f} | pos={self.pos.round(2)} | u={np.round(u_mpc, 4)}")
        # self.get_logger().info(f"t={self.t_sim:.2f} | u={np.round(u_mpc, 4)}")
        # self.get_logger().info(f"x={x0}")
        # self.get_logger().info(f"x_ref={x_ref}")
        # self.get_logger().info(f"x_dif={x_ref-x0}")


def main(args=None):
    rclpy.init(args=args)
    node = MpcControllerNode()

    def signal_handler(sig, frame):
        node.get_logger().info("Shutdown signal received. Cleaning up...")
        node.logger.plot_logs()
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