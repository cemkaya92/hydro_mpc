# File: hydro_mpc/gui/ros_interface.py

import rclpy
from rclpy.node import Node
from threading import Thread
from px4_msgs.msg import VehicleOdometry
from std_msgs.msg import Float32MultiArray
import numpy as np

from rclpy.qos import QoSProfile, ReliabilityPolicy

from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import PoseArray, Pose

class ROSInterface:
    def __init__(self):

        if not rclpy.ok():
            rclpy.init()  # ✅ Ensure ROS 2 is initialized before creating node

        self.pose = np.array([0.0, 0.0, 0.0])
        self.node = rclpy.create_node("waypoint_gui_ros_node")

        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)

        # Subscriber: Vehicle state
        self.node.create_subscription(
            VehicleOdometry,
            "/fmu/out/vehicle_odometry",
            self.odom_callback,
            qos
        )

        # Publisher: Waypoints
        self.waypoint_pub = self.node.create_publisher(Float32MultiArray, "/waypoint_array", 10)

        self.pose_pub = self.node.create_publisher(PoseArray, '/visualization_waypoints', 10)

        # Spin in background thread
        self.running = True
        
        self.executor = MultiThreadedExecutor()
        self.executor.add_node(self.node)

    def odom_callback(self, msg):
        self.pose = np.array([msg.position[0], msg.position[1], msg.position[2]])

    def get_current_pose(self):
        return self.pose.copy()

    def publish_waypoints(self, waypoint_list):
        flat = [val for wp in waypoint_list for val in wp]
        msg = Float32MultiArray()
        msg.data = flat
        self.waypoint_pub.publish(msg)

    def publish_pose_array(self, waypoints):
        msg = PoseArray()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = 'map'  # Or 'odom' or whatever frame you're using
        for wp in waypoints:
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = wp[:3]
            msg.poses.append(pose)
        self.pose_pub.publish(msg)

    def spin_loop(self):
        self.executor.spin()

    def shutdown(self):
        self.running = False
        self.executor.shutdown()
        self.node.destroy_node()
        rclpy.shutdown()
