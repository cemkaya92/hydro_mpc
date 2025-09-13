#!/usr/bin/env python3
import math, numpy as np, rclpy, cv2
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import Header, String
import time

class ArucoDetectorNode(Node):
    def __init__(self):
        super().__init__('aruco_detector')
        ns = 'perception.aruco_detector'
        self.ns = ns
        self.declare_parameters('', [
            (f'{ns}.tag_dictionary', 'DICT_4X4_250'),
            (f'{ns}.tag_ids', [0]),
            (f'{ns}.plate_size_m', 0.50),
            (f'{ns}.publish_debug', True),
            (f'{ns}.camera_optical_frame', 'uav/camera_optical_frame'),
            (f'{ns}.parent_frame', 'uav/base_link'),
            (f'{ns}.child_frame', 'rover/base_link'),
            (f'{ns}.overlay.base_is_frd', True),
            (f'{ns}.overlay.show', False)
        ])

        self.bridge = CvBridge()
        self.K = None; self.dist = None
        # self.K = np.array([[1.04953655e+03, 0.00000000e+00, 5.76017562e+02],[0.00000000e+00, 1.04706728e+03, 3.08646671e+02],[0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        # self.dist = np.array([-0.49527971,  0.28734881,  0.00506379,  0.00353585, -0.05903241])

        # self.K = np.array([[678.70790409, 0.0, 289.10897615], [0.0, 675.90522502, 217.38481548], [0.0, 0.0, 1.0]])
        # self.dist = np.array([-0.47523875, 0.19233214, 0.00119413, 0.00175783, 0.20134492])



        self.dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.P('tag_dictionary')))
        self.params = cv2.aruco.DetectorParameters()
        p = self.params
        # p.adaptiveThreshWinSizeMin = 3
        # p.adaptiveThreshWinSizeMax = 23
        # p.adaptiveThreshWinSizeStep = 10
        p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE  # off = faster
        # # helpful if tags occupy a known size range at 640x480:
        # p.minMarkerPerimeterRate = 0.03
        # p.maxMarkerPerimeterRate = 0.8
        self.detector = cv2.aruco.ArucoDetector(self.dict, self.params)

        qos = QoSProfile(depth=2)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        qos.history = HistoryPolicy.KEEP_LAST
        qos.durability = DurabilityPolicy.VOLATILE
        # Subscriptions to EKF outputs
        qos_rel = QoSProfile(depth=4)
        qos_rel.reliability = ReliabilityPolicy.BEST_EFFORT
        qos_rel.history = HistoryPolicy.KEEP_LAST
        qos_rel.durability = DurabilityPolicy.TRANSIENT_LOCAL

        qos_dbg = QoSProfile(depth=1)
        qos_dbg.reliability = ReliabilityPolicy.BEST_EFFORT
        qos_dbg.history = HistoryPolicy.KEEP_LAST
        qos_dbg.durability = DurabilityPolicy.VOLATILE

        qos_status = QoSProfile(depth=1)
        qos_status.reliability = ReliabilityPolicy.BEST_EFFORT
        qos_status.history = HistoryPolicy.KEEP_LAST
        qos_status.durability = DurabilityPolicy.VOLATILE

        self.sub_img = self.create_subscription(Image, 'camera/image_raw', self.cb_img, qos)
        self.sub_info = self.create_subscription(CameraInfo, 'camera/camera_info', self.cb_info, qos)

        self.pub_pose = self.create_publisher(PoseWithCovarianceStamped, 'perception/aruco/relative_pose', 10)
        self.pub_dbg = self.create_publisher(Image, 'perception/debug/aruco_overlay', 10)
        self.status_pub = self.create_publisher(String, 'perception/aruco/status', qos_status)
        self.sub_fused = self.create_subscription(
            PoseWithCovarianceStamped, 'perception/relative_pose_fused', self.cb_fused, qos_rel)
                
        self.latest_fused = None
        self.rel_flag = False
        self.meas_age = float('nan')

        self.tag_set = set(int(x) for x in self.P('tag_ids'))    # cache once
        self.publish_debug = bool(self.P('publish_debug'))
        self.overlay_show = bool(self.P('overlay.show'))
        self.plate_size_m = float(self.P('plate_size_m'))
        self.camera_optical_frame = self.P('camera_optical_frame')

    def P(self, key): return self.get_parameter(f'{self.ns}.{key}').value

    def _to_flu(self, p, base_is_frd: bool):
        # Convert FLU -> FRD if needed, else pass-through
        if not base_is_frd:
            return np.array([p[0], -p[1], -p[2]], dtype=float)
        return np.array(p, dtype=float)

    def cb_fused(self, msg: PoseWithCovarianceStamped):
        self.latest_fused = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ], dtype=float)

    def cb_info(self, msg: CameraInfo):
        self.K = np.array(msg.k, dtype=np.float64).reshape(3,3)
        self.dist = np.array(msg.d, dtype=np.float64).reshape((-1,))

    def cb_img(self, msg: Image):
        if self.K is None:
            return

        enc = (msg.encoding or "").lower()
        cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

        # Convert to grayscale exactly once, based on actual encoding
        if enc in ('yuv422_yuy2', 'yuyv'):
            gray = cv2.cvtColor(cv, cv2.COLOR_YUV2GRAY_YUY2)
        elif enc == 'mono8':
            gray = cv  # already 1-channel
        elif enc == 'bgr8':
            gray = cv2.cvtColor(cv, cv2.COLOR_BGR2GRAY)
        elif enc == 'rgb8':
            gray = cv2.cvtColor(cv, cv2.COLOR_RGB2GRAY)
        else:
            # fallback if something unexpected shows up
            gray = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')


        # small = cv2.resize(gray, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        # corners_s, ids, _ = self.detector.detectMarkers(small)  # heavy step (faster on small)


        corners, ids, _ = self.detector.detectMarkers(gray)
        
        # Find first desired tag
        idx = None
        if ids is not None:
            flat = ids.flatten()
            for i, mid in enumerate(flat):
                if int(mid) in self.tag_set:
                    idx = i
                    # status is optional; keep it lightweight
                    self.status_pub.publish(String(data='ok'))
                    break

        if idx is None:
            self.status_pub.publish(String(data='no-detection'))
            return

        # scale corners back up to 640x480 coordinates
        # corners = [ (corners_s[idx] * 2.0).astype(np.float32) ]

        # Pose for the chosen tag (use idx, not i)
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
            [corners[idx]], self.plate_size_m, self.K, self.dist
        )
        rvec = np.asarray(rvec, dtype=float).reshape(3,)
        tvec = np.asarray(tvec, dtype=float).reshape(3,)

        # Fast scalar quaternion from rvec
        x, y, z = float(rvec[0]), float(rvec[1]), float(rvec[2])
        theta2 = x*x + y*y + z*z
        if theta2 < 1e-12:
            qx = qy = qz = 0.0; qw = 1.0
        else:
            theta = math.sqrt(theta2)
            inv_theta = 1.0 / theta
            ax, ay, az = x*inv_theta, y*inv_theta, z*inv_theta
            half = 0.5 * theta
            sh = math.sin(half)
            qw = math.cos(half)
            qx, qy, qz = ax*sh, ay*sh, az*sh
            # normalize once
            inv_norm = 1.0 / math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
            qx *= inv_norm; qy *= inv_norm; qz *= inv_norm; qw *= inv_norm

        # Covariance roughness
        pose = PoseWithCovarianceStamped()
        pose.header = Header()
        pose.header.frame_id = self.camera_optical_frame
        pose.header.stamp = msg.header.stamp
        pose.pose.pose.position.x = float(tvec[0])
        pose.pose.pose.position.y = float(tvec[1])
        pose.pose.pose.position.z = float(tvec[2])
        # Orientation from R:
        pose.pose.pose.orientation.w = qw
        pose.pose.pose.orientation.x = qx
        pose.pose.pose.orientation.y = qy
        pose.pose.pose.orientation.z = qz
        self.pub_pose.publish(pose)

        # self.get_logger().info(f"ArUco hit dict={self.P('tag_dictionary')} id={marker_id}")

        # Optional overlay (keep OFF for perf)
        if self.publish_debug and self.overlay_show:
            annotated = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.aruco.drawDetectedMarkers(annotated, [corners[idx]])
            cv2.drawFrameAxes(annotated, self.K, self.dist, rvec, tvec, 0.05)
            try:
                dbg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
                dbg.header = pose.header
                self.pub_dbg.publish(dbg)
            except Exception:
                pass

        

def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()