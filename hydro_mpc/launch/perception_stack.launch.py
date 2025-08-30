from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration as C
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
import os

def generate_launch_description():
     ns = C('uav_ns')
     pkg_share = get_package_share_directory('hydro_mpc')
     yolo_yaml = os.path.join(pkg_share, 'config', 'perception', 'yolo.yaml')
     aruco_yaml = os.path.join(pkg_share, 'config', 'perception', 'aruco.yaml')
     est_yaml  = os.path.join(pkg_share, 'config', 'perception', 'estimator.yaml')

     venv_py = "/home/asl-simulation/venvs/vision/bin/python"   # your venv Python
     apt_env = {"PYTHONNOUSERSITE": "1"}  # ignore ~/.local wheels for apt-based nodes


     return LaunchDescription([
          DeclareLaunchArgument('uav_ns', default_value='drone'),

          Node(package='hydro_mpc', executable='yolo_detector', namespace=ns,
               parameters=[yolo_yaml],
               prefix=venv_py),

          Node(package='hydro_mpc', executable='aruco_detector', namespace=ns,
               parameters=[aruco_yaml],
               prefix=venv_py),

          Node(package='hydro_mpc', executable='ekf_fuser', namespace=ns,
               parameters=[est_yaml],
               prefix=venv_py),

          Node(
               package='tf2_ros', executable='static_transform_publisher',
               namespace=ns,
               arguments=['0','0','0',  '-1.570796','0','-1.570796',
                         'camera_link','camera_optical_frame'],
          )
     ])
