import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import Command, PathJoinSubstitution
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource


namePackage = 'hydro_mpc'



def generate_launch_description():

    ns_drone  = LaunchConfiguration('ns_drone')
    sys_id  = LaunchConfiguration('sys_id')
    mission_param_file  = LaunchConfiguration('mission_param_file')
    ns_camera = LaunchConfiguration('camera_ns')

    yolo_yaml = PathJoinSubstitution([
        FindPackageShare('hydro_mpc'),
        'config', 
        'perception', 
        'yolo.yaml'
    ])
    aruco_yaml = PathJoinSubstitution([
        FindPackageShare('hydro_mpc'),
        'config', 
        'perception', 
        'aruco.yaml'
    ])
    est_yaml = PathJoinSubstitution([
        FindPackageShare('hydro_mpc'),
        'config', 
        'perception', 
        'estimator.yaml'
    ])


    venv_py = "/home/jetson/vision/bin/python"   # your venv Python
    apt_env = {"PYTHONNOUSERSITE": "1"}  # ignore ~/.local wheels for apt-based nodes



    return LaunchDescription([

        DeclareLaunchArgument('ns_drone',   default_value=''),
        DeclareLaunchArgument('sys_id',   default_value='1'),
        DeclareLaunchArgument('camera_ns', default_value='/camera'),

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

        DeclareLaunchArgument(
            'mission_param_file',
            default_value='utari_demo_mission_params.yaml',
            description='Mission parameter file inside config/mission/'
        ),

        Node(
               package="v4l2_camera",
               executable="v4l2_camera_node",
               name="usb_camera",
               namespace=ns_camera,
               output="screen",
               parameters=[{
                    "video_device": "/dev/video0",   # Change if needed
                    "image_size": [640, 480],        # Resolution
                    "frame_rate": 30.0,              # FPS
                    "pixel_format": "YUYV",          # or "MJPG" if supported
                    "camera_name": "usb_cam",        # Name used in /camera_info
                         "output_encoding": "yuv422_yuy2",      # v4l2_camera publishes mono8 without conversion
               }]
        ),

        Node(
            package=namePackage,
            executable='offboard_manager_node',
            name='offboard_manager_node',
            namespace=ns_drone,
            output='screen',
            parameters=[{
                'vehicle_param_file': LaunchConfiguration('vehicle_param_file'),
                'sitl_param_file': LaunchConfiguration('sitl_param_file'),
                'disarm_on_trip': False,
                'auto_reenter_after_trip': False,
                'sys_id': sys_id
            }]
        ),

        Node(
            package=namePackage,
            executable='navigator_node',
            name='navigator_node',
            namespace=ns_drone,
            output='screen',
            parameters=[{
                'sitl_param_file': LaunchConfiguration('sitl_param_file'),
                'mission_param_file': LaunchConfiguration('mission_param_file'),
                'control_frequency': 50.0,
                'auto_start': False,
            }],
        ),

        Node(
            package=namePackage,
            executable='trajectory_publisher_node',
            name='trajectory_publisher_node',
            namespace=ns_drone,
            output='screen',
            parameters=[{
                'sitl_param_file': LaunchConfiguration('sitl_param_file')
            }]
        ),

        Node(
            package='hydro_mpc', 
            executable='aruco_detector', 
            name='aruco_detector',
            namespace=ns_drone,
            parameters=[aruco_yaml],
            prefix=venv_py
        ),

        Node(
            package='hydro_mpc', 
            executable='ekf_fuser',
            name='ekf_fuser', 
            namespace=ns_drone,
            parameters=[est_yaml],
            prefix=venv_py
        ),

        


    ])

