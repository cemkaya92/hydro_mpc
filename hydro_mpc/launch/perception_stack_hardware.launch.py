from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration as C
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
import os

def generate_launch_description():
     ns = C('uav_ns')
     ns_camera = C('camera_ns')
     pkg_share = get_package_share_directory('hydro_mpc')
     yolo_yaml = os.path.join(pkg_share, 'config', 'perception', 'yolo.yaml')
     aruco_yaml = os.path.join(pkg_share, 'config', 'perception', 'aruco.yaml')
     est_yaml  = os.path.join(pkg_share, 'config', 'perception', 'estimator.yaml')

     venv_py = "/home/jetson/vision/bin/python"   # your venv Python
     apt_env = {"PYTHONNOUSERSITE": "1"}  # ignore ~/.local wheels for apt-based nodes


     return LaunchDescription([
          DeclareLaunchArgument('uav_ns', default_value='drone'),
	  DeclareLaunchArgument('camera_ns', default_value='drone/camera'),
          
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
	  
	  Node(package='hydro_mpc', executable='aruco_detector', namespace=ns,
               parameters=[aruco_yaml],
               prefix=venv_py),


          Node(package='hydro_mpc', executable='ekf_fuser', namespace=ns,
               parameters=[est_yaml],
               prefix=venv_py)		   

     ])
