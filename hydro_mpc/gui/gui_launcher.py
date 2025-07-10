# File: hydro_mpc/gui/gui_launcher.py

import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
from .main_window import MainWindow
from .ros_interface import ROSInterface


def main():
    app = QApplication(sys.argv)

    ros = ROSInterface()
    gui = MainWindow()
    gui.show()

    # Periodically spin ROS 2 to handle callbacks
    timer = QTimer()
    timer.timeout.connect(lambda: None)  # Keeps Qt's event loop going
    timer.start(100)

    exit_code = app.exec_()

    # Clean shutdown
    ros.shutdown()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
