# File: hydro_mpc/gui/waypoint_manager.py

from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem
import numpy as np

class WaypointManager:
    def __init__(self, ros_interface):
        self.ros = ros_interface
        self.current_pose = np.array([0.0, 0.0, 0.0])
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["X", "Y", "Z"])

    def get_table_widget(self):
        return self.table

    def set_current_pose(self, pose):
        self.current_pose = pose

    def add_waypoint(self):
        row = self.table.rowCount()
        self.table.insertRow(row)
        for col in range(3):
            self.table.setItem(row, col, QTableWidgetItem("0.0"))

    def get_user_waypoints(self):
        waypoints = []
        for row in range(self.table.rowCount()):
            wp = []
            for col in range(3):
                item = self.table.item(row, col)
                try:
                    val = float(item.text()) if item else 0.0
                except ValueError:
                    val = 0.0
                wp.append(val)
            waypoints.append(wp)
        return waypoints

    def publish_waypoints(self):
        waypoints = [self.current_pose.tolist()]  # Always start with current pose
        waypoints += self.get_user_waypoints()
        self.ros.publish_waypoints(waypoints)
        print(f"✅ Published {len(waypoints)} waypoints.")
