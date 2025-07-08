from launch import LaunchDescription
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


uav_params_path = PathJoinSubstitution([
    FindPackageShare('hydro_mpc'),
    'config',
    'uav_parameters',
    'crazyflie.yaml'
])

sitl_params_path = PathJoinSubstitution([
    FindPackageShare('hydro_mpc'),
    'config',
    'sitl',
    'sitl_params.yaml'
])

mpc_params_path = PathJoinSubstitution([
    FindPackageShare('hydro_mpc'),
    'config',
    'controller',
    'mpc_crazyflie.yaml'
])

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'uav_param_file',
            default_value='crazyflie_param.yaml',
            description='UAV param file inside config/uav_parameters/'
        ),
        DeclareLaunchArgument(
            'mpc_param_file',
            default_value='mpc_crazyflie.yaml',
            description='MPC parameter file inside config/controller/'
        ),

        Node(
            package='hydro_mpc',
            executable='motor_commander',
            name='motor_commander',
            output='screen',
            parameters=[{
                'uav_param_file': LaunchConfiguration('uav_param_file')
            }]
        ),
        Node(
            package='hydro_mpc',
            executable='mpc_controller',
            name='mpc_controller',
            output='screen',
            parameters=[{
                'uav_param_file': LaunchConfiguration('uav_param_file'),
                'mpc_param_file': LaunchConfiguration('mpc_param_file')
            }]
        )
    ])

