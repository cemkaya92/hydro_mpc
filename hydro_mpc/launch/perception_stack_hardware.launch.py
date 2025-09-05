from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration as C
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution

namePackage = 'hydro_mpc'

def generate_launch_description():
     ns = C('uav_ns')
     ns_camera = C('camera_ns')

     
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
          'aruco_exp.yaml'
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
	  
          Node(
               package=namePackage, 
               executable='aruco_detector', namespace=ns,
               parameters=[aruco_yaml],
               prefix=venv_py
          ),


          Node(
               package=namePackage, 
               executable='ekf_fuser', 
               namespace=ns,
               parameters=[est_yaml],
               prefix=venv_py
          )		   

     ])
