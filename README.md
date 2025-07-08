# HydroMPC

A modular ROS 2 package for MPC-based control and motor command allocation in PX4-based drones.

## Features
- Modular motor command architecture
- Support for UAV-specific YAML config
- Real-time MPC using ROS 2 timers

## How to Run

```bash
ros2 launch hydro_mpc control_stack.launch.py uav_param_file:=crazyflie_param.yaml mpc_param_file:=mpc_crazyflie.yaml
