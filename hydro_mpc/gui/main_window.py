# File: hydro_mpc/gui/main_window.py

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QTableWidget, QHBoxLayout
from PyQt5.QtCore import QTimer
from .ros_interface import ROSInterface
from .waypoint_manager import WaypointManager
from .trajectory_control import TrajectoryController


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HydroMPC Waypoint Planner")

        # Core Components
        self.ros = ROSInterface()
        self.waypoints = WaypointManager(self.ros)
        self.trajectory = TrajectoryController(self.ros, self.waypoints)

        # Layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.status_label = QLabel("Current Pose: [0.0, 0.0, 0.0]")
        self.layout.addWidget(self.status_label)

        # Waypoint Table
        self.layout.addWidget(self.waypoints.get_table_widget())

        # Buttons
        button_layout = QHBoxLayout()
        btn_add = QPushButton("Add Waypoint")
        btn_add.clicked.connect(self.waypoints.add_waypoint)
        btn_publish = QPushButton("Publish Waypoints")
        btn_publish.clicked.connect(self.waypoints.publish_waypoints)
        btn_start = QPushButton("Start Trajectory")
        btn_start.clicked.connect(self.trajectory.start)

        btn_stop = QPushButton("Stop")
        btn_stop.clicked.connect(self.trajectory.stop)

        btn_reset = QPushButton("Reset")
        btn_reset.clicked.connect(self.trajectory.reset)

        button_layout.addWidget(btn_add)
        button_layout.addWidget(btn_publish)
        button_layout.addWidget(btn_start)
        button_layout.addWidget(btn_stop)
        button_layout.addWidget(btn_reset)
        self.layout.addLayout(button_layout)
        

        # Update Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_gui)
        self.timer.start(500)

    def update_gui(self):
        pose = self.ros.get_current_pose()
        self.status_label.setText(f"Current Pose: {pose.round(2).tolist()}")
        self.waypoints.set_current_pose(pose)
