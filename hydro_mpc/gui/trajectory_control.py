# File: hydro_mpc/gui/trajectory_control.py

from std_msgs.msg import String

class TrajectoryController:
    def __init__(self, ros_interface, waypoint_manager):
        self.ros = ros_interface
        self.waypoints = waypoint_manager
        self.control_pub = self.ros.node.create_publisher(String, "/traj_control_cmd", 10)

    def start(self):
        msg = String()
        msg.data = "start"
        self.control_pub.publish(msg)
        print("▶️ Trajectory START command sent.")

    def stop(self):
        msg = String()
        msg.data = "stop"
        self.control_pub.publish(msg)
        print("⏹ Trajectory STOP command sent.")

    def reset(self):
        msg = String()
        msg.data = "reset"
        self.control_pub.publish(msg)
        print("🔄 Trajectory RESET command sent.")
