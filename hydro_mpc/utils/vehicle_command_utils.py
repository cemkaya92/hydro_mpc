
from px4_msgs.msg import VehicleCommand

def create_arm_command(timestamp):
    cmd = VehicleCommand()
    cmd.timestamp = timestamp
    cmd.param1 = 1.0
    cmd.command = VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM
    cmd.target_system = 1
    cmd.target_component = 1
    cmd.source_system = 1
    cmd.source_component = 1
    cmd.from_external = True
    return cmd

def create_offboard_mode_command(timestamp):
    cmd = VehicleCommand()
    cmd.timestamp = timestamp
    cmd.command = VehicleCommand.VEHICLE_CMD_DO_SET_MODE
    cmd.param1 = 1.0  # PX4_CUSTOM_MAIN_MODE = OFFBOARD
    cmd.param2 = 6.0  # PX4_CUSTOM_SUB_MODE = OFFBOARD
    cmd.target_system = 1
    cmd.target_component = 1
    cmd.source_system = 1
    cmd.source_component = 1
    cmd.from_external = True
    return cmd