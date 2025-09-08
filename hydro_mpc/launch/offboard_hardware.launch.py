from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription, DeclareLaunchArgument, ExecuteProcess,
    LogInfo, RegisterEventHandler, OpaqueFunction, SetLaunchConfiguration
)
from launch.event_handlers import OnProcessIO, OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

PROBE_READY_MARKER = 'Telemetry received AND command subscribers discovered.'
XRCE_READY_MARKER = 'participant created'   # seen in Agent logs


namePackage = 'hydro_mpc'


def generate_launch_description():

    init_flag = SetLaunchConfiguration('stack_started', '0')

    # --- Arguments for XRCE Agent ---
    xrce_dev_arg = DeclareLaunchArgument(
        'xrce_dev', default_value='/dev/px4',
        description='Serial device for Micro XRCE-DDS Agent (use /dev/px4 if you set a udev rule)'
    )
    xrce_baud_arg = DeclareLaunchArgument(
        'xrce_baud', default_value='921600',
        description='Baudrate for Micro XRCE-DDS Agent'
    )
    xrce_ns_arg = DeclareLaunchArgument(
        'xrce_ns', default_value='',
        description='Namespace for Micro XRCE-DDS Agent'
    )


    # Deep readiness probe: confirms PX4 pubs/subs & first msgs
    px4_probe = Node(
        package=namePackage,
        executable='wait_px4_ready',
        name='wait_px4_ready',
        # pass just the raw ns name, not '/ns'
        arguments=['--ns', LaunchConfiguration('xrce_ns'), '--timeout', '40.0'],
        output='screen'
    )

    drone_launch = PathJoinSubstitution([
        FindPackageShare(namePackage),
        'launch',
        'drone_offboard_hardware.launch.py'
    ])

    # --- Start Micro XRCE-DDS Agent ---
    # (No sudo needed if you fixed permissions)
    agent_proc = ExecuteProcess(
        cmd=[
            'MicroXRCEAgent', 'serial',
            '--dev', LaunchConfiguration('xrce_dev'),
            '-b',   LaunchConfiguration('xrce_baud'),
            '-n',   LaunchConfiguration('xrce_ns')   # <--- namespace arg
        ],
        output='screen'
    )


    start_drone = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(drone_launch)
    )


    def _start_probe(_ctx, *a, **k):
        return [LogInfo(msg='[XRCE] Agent detected, running PX4 readiness probe...'), px4_probe]

    def _start_stack(_ctx, *a, **k):
        return [LogInfo(msg='[XRCE] PX4 ready. Launching drone...'), start_drone]


    def _start_stack_once(context, *a, **k):
        # run only once even if multiple handlers fire
        started = LaunchConfiguration('stack_started').perform(context)
        if started == '1':
            return []
        # mark as started
        return [
            SetLaunchConfiguration('stack_started', '1'),
            LogInfo(msg=f'[XRCE] PX4 ready. Launching drone...'),
            # bring up relays/bridge if you have them, then the rest:
            start_drone,
        ]
    
    # 1) When Agent prints the ready marker, run the probe
    on_agent_seen = RegisterEventHandler(
        OnProcessIO(
            target_action=agent_proc,
            on_stdout=lambda ev: [OpaqueFunction(function=_start_probe)]
                if XRCE_READY_MARKER in ev.text.decode(errors='ignore') else [],
            on_stderr=lambda ev: [OpaqueFunction(function=_start_probe)]
                if XRCE_READY_MARKER in ev.text.decode(errors='ignore') else [],
        )
    )

    # Fire the rest of the stack as soon as the probe prints the READY line
    on_probe_ready = RegisterEventHandler(
        OnProcessIO(
            target_action=px4_probe,
            on_stdout=lambda ev: [OpaqueFunction(function=_start_stack_once)]
                if PROBE_READY_MARKER in ev.text.decode(errors='ignore') else [],
            on_stderr=lambda ev: [OpaqueFunction(function=_start_stack_once)]
                if PROBE_READY_MARKER in ev.text.decode(errors='ignore') else [],
        )
    )

    # Optional: if Agent dies, bring down launch
    on_agent_exit = RegisterEventHandler(
        OnProcessExit(
            target_action=agent_proc,
            on_exit=[LogInfo(msg='[XRCE] Agent exited — shutting down.')]
        )
    )


    return LaunchDescription([
        xrce_dev_arg, xrce_baud_arg, xrce_ns_arg,
        init_flag,
        agent_proc,
        on_agent_seen,
        on_probe_ready,     # <--- NEW: react to probe's READY log
        on_agent_exit,
    ])

