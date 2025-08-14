import rclpy
from rclpy.node import Node
import numpy as np
from std_msgs.msg import Float32MultiArray
from px4_msgs.msg import ActuatorMotors, OffboardControlMode, VehicleCommand, VehicleThrustSetpoint, VehicleTorqueSetpoint 
from hydro_mpc.utils.vehicle_command_utils import create_arm_command, create_offboard_mode_command

from hydro_mpc.utils.param_loader import ParamLoader
from hydro_mpc.control.control_allocator import ControlAllocator

from ament_index_python.packages import get_package_share_directory
import os


class MotorCommander(Node):
    def __init__(self):
        super().__init__('motor_commander')

        package_dir = get_package_share_directory('hydro_mpc')
        
        # Declare param with default
        self.declare_parameter('vehicle_param_file', 'crazyflie_param.yaml')
        self.declare_parameter('sitl_param_file', 'sitl_params.yaml')

        vehicle_param_file = self.get_parameter('vehicle_param_file').get_parameter_value().string_value
        sitl_param_file = self.get_parameter('sitl_param_file').get_parameter_value().string_value

        sitl_yaml_path = os.path.join(package_dir, 'config', 'sitl', sitl_param_file)
        vehicle_yaml_path = os.path.join(package_dir, 'config', 'vehicle_parameters', vehicle_param_file)

        # Load parameters
        sitl_yaml = ParamLoader(sitl_yaml_path)
        vehicle_yaml = ParamLoader(vehicle_yaml_path)

        # UAV parameters
        self.vehicle_params = vehicle_yaml.get_vehicle_params()

        # pub / sub
        self.motor_pub = self.create_publisher(ActuatorMotors, sitl_yaml.get_topic("actuator_control_topic"), 10)
        self.offboard_ctrl_pub = self.create_publisher(OffboardControlMode, sitl_yaml.get_topic("offboard_control_topic"), 10)
        self.cmd_pub = self.create_publisher(VehicleCommand, sitl_yaml.get_topic("vehicle_command_topic"), 10)
        self.thrust_pub = self.create_publisher(VehicleThrustSetpoint, sitl_yaml.get_topic("thrust_setpoints_topic"), 1)
        self.torque_pub = self.create_publisher(VehicleTorqueSetpoint, sitl_yaml.get_topic("torque_setpoints_topic"), 1)
        
        self.create_subscription(Float32MultiArray, sitl_yaml.get_topic("mpc_command_topic"), self.mpc_cmd_callback, 10)

        self.sys_id = sitl_yaml.get_nested(["sys_id"],1)

        self.get_logger().info(f"sys_id= {self.sys_id}")

        # initial states
        self.latest_motor_cmd = [0.0, 0.0, 0.0, 0.0]

        self.latest_thrust_cmd = VehicleThrustSetpoint()
        self.latest_torque_cmd = VehicleTorqueSetpoint()

        self.normalized_torque_and_thrust = [0.0, 0.0, 0.0, 0.0]
        
        # static allocation matrices
        self.rotor_velocities_to_torques_and_thrust, self.torques_and_thrust_to_rotor_velocities = \
        ControlAllocator.compute_allocation_matrices(self.vehicle_params.num_of_arms, self.vehicle_params.thrust_constant, self.vehicle_params.moment_constant, self.vehicle_params.arm_length)

        # Typical X-configuration
        angles_deg = [135, 45, 315, 225]
        spin_dirs = [-1, 1, -1, 1]

        _, self.throttles_to_normalized_torques_and_thrust = ControlAllocator.generate_mixing_matrices(
            1.0, 1.0, 1.5, angles_deg, spin_dirs
        )

        # self.get_logger().info("[mixing_matrix] =\n" + np.array2string(mixing_matrix, precision=10, suppress_small=True))
        # self.get_logger().info("[mixing_matrix_inv] =\n" + np.array2string(self.throttles_to_normalized_torques_and_thrust, precision=10, suppress_small=True))

        # Timers
        self.motor_command_timer = self.create_timer(0.01, self.motor_command_timer_callback) # 100 Hz
        self.offboard_timer = self.create_timer(0.2, self.publish_offboard_control_mode)  # 5 Hz
        self.offboard_set = False
        self.get_logger().info("MotorCommander with Offboard control started")

    def motor_command_timer_callback(self):
        now_us = int(self.get_clock().now().nanoseconds / 1000)

        motors_msg = ActuatorMotors()
        motors_msg.timestamp = now_us
        # #motors_msg.control = [0.90, 0.0, 0.0, 0.0] + [0.0] * 8
        motors_msg.control[0:4] = self.latest_motor_cmd
        # self.motor_pub.publish(motors_msg)

        self.thrust_pub.publish(self.latest_thrust_cmd)
        self.torque_pub.publish(self.latest_torque_cmd)


    def publish_offboard_control_mode(self):
        now_us = int(self.get_clock().now().nanoseconds / 1000)

        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp = now_us
        offboard_msg.position = False
        offboard_msg.velocity = False
        offboard_msg.acceleration = False
        offboard_msg.attitude = False
        offboard_msg.body_rate = False
        offboard_msg.thrust_and_torque = True
        offboard_msg.direct_actuator = False
        self.offboard_ctrl_pub.publish(offboard_msg)

        # Start Offboard + Arm only after receiving first valid control command
        # if not self.offboard_set and any([abs(x) > 1e-3 for x in self.latest_motor_cmd]):
        if not self.offboard_set:
            self.cmd_pub.publish(create_offboard_mode_command(now_us, self.sys_id))
            self.cmd_pub.publish(create_arm_command(now_us, self.sys_id))
            self.offboard_set = True
            self.get_logger().info("Sent OFFBOARD and ARM command") 

    
        
        
    def mpc_cmd_callback(self, msg):

        now_us = int(self.get_clock().now().nanoseconds / 1000)

        thrust_cmd = msg.data[0]
        torque_cmd = msg.data[1:4]
        #self.get_logger().info(f"thrust= {thrust_cmd} | torque= {torque_cmd}")

        torque_thrust_vec = np.concatenate((torque_cmd, [thrust_cmd])).reshape((4, 1))
        omega_sq = self.torques_and_thrust_to_rotor_velocities @ torque_thrust_vec
        omega_sq = np.clip(omega_sq, 0.0, None)  # Ensures non-negative values
        # self.get_logger().info(f"torque_thrust_vec= {torque_thrust_vec}")
        # self.get_logger().info(f"omega_sq= {omega_sq}")

        omega = np.sqrt(omega_sq)

        throttles = omega / self.vehicle_params.max_rotor_speed

        # self.get_logger().info(f"omega= {omega} | throttles= {throttles}")

        self.normalized_torque_and_thrust = self.throttles_to_normalized_torques_and_thrust @ throttles

        #self.get_logger().info(f"normalized_torque_and_thrust= {self.normalized_torque_and_thrust} ")


        # Prepare thrust message
        self.latest_thrust_cmd.timestamp = now_us
        self.latest_thrust_cmd.xyz[0] = 0.0
        self.latest_thrust_cmd.xyz[1] = 0.0
        self.latest_thrust_cmd.xyz[2] = -self.normalized_torque_and_thrust[3] 


        # Prepare torque message
        self.latest_torque_cmd.timestamp = now_us
        self.latest_torque_cmd.xyz[0] = self.normalized_torque_and_thrust[0]
        self.latest_torque_cmd.xyz[1] = self.normalized_torque_and_thrust[1]
        self.latest_torque_cmd.xyz[2] = self.normalized_torque_and_thrust[2]



def main(args=None):
    rclpy.init(args=args)
    node = MotorCommander()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()