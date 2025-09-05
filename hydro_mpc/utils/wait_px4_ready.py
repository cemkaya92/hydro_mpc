#!/usr/bin/env python3
import argparse, time, sys, os
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import VehicleStatus, VehicleOdometry

def qos_sensor():
    q = QoSProfile(depth=10)
    q.reliability = ReliabilityPolicy.BEST_EFFORT
    q.history = HistoryPolicy.KEEP_LAST
    q.durability = DurabilityPolicy.VOLATILE
    return q

class Px4Ready(Node):
    def __init__(self, ns: str, timeout: float):
        super().__init__('wait_px4_ready')
        self.ns = ('' if not ns or ns == '/' else f'/{ns.strip("/")}')
        self.deadline = time.time() + timeout

        self.got_any_telemetry = False
        self.have_offboard_sinks = False

        # Subscribe to either of these; one is enough to prove telemetry is flowing
        self.sub_status = self.create_subscription(
            VehicleStatus, f'{self.ns}/fmu/out/vehicle_status',
            lambda _msg: self._mark_telemetry(), qos_sensor())
        self.sub_odom = self.create_subscription(
            VehicleOdometry, f'{self.ns}/fmu/out/vehicle_odometry',
            lambda _msg: self._mark_telemetry(), qos_sensor())

        self.create_timer(0.25, self._tick)

        self.get_logger().info(f'[PX4 READY] Watching "{self.ns or "/"}/fmu/*" for pubs/subs...')

    def _mark_telemetry(self):
        self.got_any_telemetry = True

    def _tick(self):
        # Graph query: does *any* node subscribe to our command sinks?
        offboard_infos = self.get_subscriptions_info_by_topic(f'{self.ns}/fmu/in/offboard_control_mode')
        traj_infos     = self.get_subscriptions_info_by_topic(f'{self.ns}/fmu/in/trajectory_setpoint')
        self.have_offboard_sinks = bool(offboard_infos) and bool(traj_infos)

        if self.got_any_telemetry and self.have_offboard_sinks:
            self.get_logger().info('[PX4 READY] Telemetry received AND command subscribers discovered.')
            # Flush logs and exit *without* waiting on executor shutdown,
            # avoids occasional hang when calling rclpy.shutdown() inside a timer.
            sys.stdout.flush(); sys.stderr.flush()
            os._exit(0)

        if time.time() > self.deadline:
            self.get_logger().error(f'[PX4 READY] Timeout waiting for PX4 pubs/subs. namespace: {self.ns}')
            rclpy.shutdown(); sys.exit(2)

    

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ns', default='', help='namespace (e.g., rover1)')
    ap.add_argument('--timeout', type=float, default=40.0)
    args, _ = ap.parse_known_args()
    rclpy.init()
    node = Px4Ready(args.ns, args.timeout)
    try:
        rclpy.spin(node)
    except SystemExit as e:
        raise e
    except Exception as ex:
        print(f'[PX4 READY] Unexpected error: {ex}', file=sys.stderr); sys.exit(3)

if __name__ == '__main__':
    main()
