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
            (f'{ns}.tag_dictionary', 'DICT_6X6_250'),
            (f'{ns}.tag_ids', [0]),
            (f'{ns}.plate_size_m', 0.50),
            (f'{ns}.publish_debug', True),
            (f'{ns}.camera_optical_frame', 'uav/camera_optical_frame'),
            (f'{ns}.parent_frame', 'uav/base_link'),
            (f'{ns}.child_frame', 'rover/base_link'),
            (f'{ns}.overlay.base_is_frd', True),
            (f'{ns}.overlay.show', True)
        ])

        self.bridge = CvBridge()
        self.K = None; self.dist = None

        self.dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.P('tag_dictionary')))
        self.params = cv2.aruco.DetectorParameters()
        # self.detector_params.cornerRefinementMaxIterations = 50
        # self.detector_params.cornerRefinementMinAccuracy = 0.01
        self.detector = cv2.aruco.ArucoDetector(self.dict, self.params)

        qos = QoSProfile(depth=2)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        qos.history = HistoryPolicy.KEEP_LAST
        qos.durability = DurabilityPolicy.VOLATILE
        # Subscriptions to EKF outputs
        qos_rel = QoSProfile(depth=10)

        self.sub_img = self.create_subscription(Image, 'camera/image_raw', self.cb_img, qos)
        self.sub_info = self.create_subscription(CameraInfo, 'camera/camera_info', self.cb_info, qos)

        self.pub_pose = self.create_publisher(PoseWithCovarianceStamped, 'perception/aruco/relative_pose', 10)
        self.pub_dbg = self.create_publisher(Image, 'perception/debug/aruco_overlay', 1)
        self.status_pub = self.create_publisher(String, 'perception/aruco/status', 1)
        self.sub_fused = self.create_subscription(
            PoseWithCovarianceStamped, 'perception/relative_pose_fused', self.cb_fused, qos_rel)
                
        self.latest_fused = None
        self.rel_flag = False
        self.meas_age = float('nan')

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
        a = 1
        #self.K = np.array(msg.k, dtype=np.float64).reshape(3,3)
        #self.dist = np.array(msg.d, dtype=np.float64).reshape((-1,))

    def cb_img(self, msg: Image):

        #if self.K is None: 
        #    self.get_logger().info(f"self.K: {self.K}")
        #    return

        cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        height, width = cv.shape[:2]
        #self.get_logger().info(f"Image width: {width}, height: {height}")

        corners, ids, _ = self.detector.detectMarkers(cv)
        found = False
        if ids is not None and len(ids) > 0:
            tag_set = set(int(x) for x in self.P('tag_ids'))
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id not in tag_set: continue
                found = True

                marker_size_meters = 0.50

                obj_points = np.array([
                    [-marker_size_meters / 2, marker_size_meters / 2, 0],
                    [ marker_size_meters / 2, marker_size_meters / 2, 0],
                    [ marker_size_meters / 2,-marker_size_meters / 2, 0],
                    [-marker_size_meters / 2,-marker_size_meters / 2, 0]
                ], dtype=np.float32)

                self.K = np.array([[678.70790409, 0.0, 289.10897615], [0.0, 675.90522502, 217.38481548], [0.0, 0.0, 1.0]])
                self.dist = np.array([-0.47523875, 0.19233214, 0.00119413, 0.00175783, 0.20134492])

                #self.K = np.load("camera_matrix.npy")
                #self.dist = np.load("dist_coeffs.npy")

                # img_points = corners[i].reshape(4, 2).astype(np.float32)
                img_points = corners[i].reshape(4, 2)
                success, rvec, tvec = cv2.solvePnP(
                    obj_points, img_points, self.K, self.dist        )

                self.get_logger().info(f"K: {self.K}")
                self.get_logger().info(f"dist: {self.dist}")
                #self.get_logger().info(f"{rvec}")
                self.get_logger().info(f"{tvec}")

                
                #if success:
                    
                 #   cv2.drawFrameAxes(cv, self.K, self.dist, rvec, tvec, marker_size_meters / 2)
                    # Draw coordinate axes for visualization.
                    
                    #print(f"Target Marker ID: {marker_id}, Translation (m): {tvec.flatten()}, Rotation: {rvec.flatten()}")
                    # Output pose for potential drone control.

                #else:

                    #print(f"Pose estimation failed for Marker ID: {marker_id}")


                # --- Robust rvec -> quaternion (geometry_msgs order: x,y,z,w) ---
                theta = float(np.linalg.norm(rvec))
                if theta < 1e-12:
                    qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
                else:
                    axis = (rvec / theta)
                    half = 0.5 * theta
                    s = math.cos(half)  # careful: OpenCV uses right-handed axis-angle
                    qw = math.cos(half)
                    sin_half = math.sin(half)
                    qx = float(axis[0] * sin_half)
                    qy = float(axis[1] * sin_half)
                    qz = float(axis[2] * sin_half)

                # (Optionally normalize to be extra safe)
                norm = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
                qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
                # Covariance roughness
                pose = PoseWithCovarianceStamped()
                pose.header = Header()
                pose.header.frame_id = self.P('camera_optical_frame')
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

                # if self.P('publish_debug') and bool(self.get_parameter(f'{self.ns}.overlay.show').value):
                #     annotated = cv.copy()

                #     # 1) If we have a fused pose, show normalized vector and altitude
                #     col_ok = (0, 200, 0)
                #     col_bad = (0, 0, 255)
                #     col = col_ok if self.rel_flag else col_bad

                #     cv2.aruco.drawDetectedMarkers(annotated, [corners[i]])
                #     cv2.drawFrameAxes(annotated, self.K, self.dist, rvec, tvec, 0.05)
                    
                #     if self.latest_fused is not None:

                #         text1 = f"Rover Pos: X={self.latest_fused[0]:.2f}, Y={self.latest_fused[1]:.2f}, Z={self.latest_fused[2]:.2f}"
                        
                #         #cv2.putText(annotated, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2, cv2.LINE_AA)
                        
                #     # 2) Also show camera-space measurement (quick sanity)
                #     text3 = f"Cam Pos: x={float(tvec[0]):.2f} y={float(tvec[1]):.2f} z={float(tvec[2]):.2f} m"
                #     #cv2.putText(annotated, text3, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 200, 0), 2, cv2.LINE_AA)


                #     try:
                #         dbg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
                #         dbg.header = pose.header
                #         self.pub_dbg.publish(dbg)
                #     except Exception:
                #         pass

        self.status_pub.publish(String(data= 'ok' if found else 'no-detection'))

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
