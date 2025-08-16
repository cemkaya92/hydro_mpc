from launch import LaunchDescription
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


namePackage = 'hydro_mpc'

uav_params_path = PathJoinSubstitution([
    FindPackageShare(namePackage),
    'config',
    'vehicle_parameters',
    'f450_param.yaml'
])

sitl_params_path = PathJoinSubstitution([
    FindPackageShare(namePackage),
    'config',
    'sitl',
    'sitl_coop_params.yaml'
])

mpc_params_path = PathJoinSubstitution([
    FindPackageShare(namePackage),
    'config',
    'controller',
    'mpc_f450.yaml'
])

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'vehicle_param_file',
            default_value='f450_param.yaml',
            description='Vehicle param file inside config/vehicle_parameters/'
        ),

        DeclareLaunchArgument(
            'sitl_param_file',
            default_value='sitl_coop_params.yaml',
            description='SITL param file inside config/sitl/'
        ),

        DeclareLaunchArgument(
            'controller_param_file',
            default_value='mpc_f450.yaml',
            description='Controller parameter file inside config/controller/'
        ),

        Node(
            package=namePackage,
            executable='motor_commander',
            name='motor_commander',
            output='screen',
            parameters=[{
                'vehicle_param_file': LaunchConfiguration('vehicle_param_file'),
                'sitl_param_file': LaunchConfiguration('sitl_param_file')
            }]
        ),

        Node(
            package=namePackage,
            executable='mpc_controller',
            name='mpc_controller',
            output='screen',
            parameters=[{
                'vehicle_param_file': LaunchConfiguration('vehicle_param_file'),
                'sitl_param_file': LaunchConfiguration('sitl_param_file'),
                'controller_param_file': LaunchConfiguration('controller_param_file'),
                'trajectory_topic': '/drone/trajectory',
                'world_frame': 'map'
            }]
        ),

        Node(
            package=namePackage,
            executable='offboard_manager_node',
            name='offboard_manager_node',
            output='screen',
            parameters=[{
                'vehicle_param_file': LaunchConfiguration('vehicle_param_file'),
                'sitl_param_file': LaunchConfiguration('sitl_param_file'),
                'disarm_on_trip': False
            }]
        ),

        Node(
            package=namePackage,
            executable='navigator_node',
            name='navigator_node',
            output='screen',
            parameters=[{
                'mission_param_file': 'mission.yaml',
                'control_frequency': 50.0,
                'auto_start': True,
            }],
        )

    ])

