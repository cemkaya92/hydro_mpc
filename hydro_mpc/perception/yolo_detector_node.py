#!/usr/bin/env python3
import cv2, math, time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import Header, String
from hydro_mpc.perception.yolo_backends import make_backend
import torch

class YoloDetectorNode(Node):
    def __init__(self):
        super().__init__('yolo_detector')
        ns = 'perception.yolo_detector'
        self.declare_parameters('', [
            (f'{ns}.backend', 'torch'),
            (f'{ns}.model_path', 'models/gz_rover.pt'),
            (f'{ns}.class_name', 'landing_plate'),
            (f'{ns}.score_thresh', 0.5),
            (f'{ns}.nms_thresh', 0.45),
            (f'{ns}.input_size', [1920, 1080]),
            (f'{ns}.plate_size_m', 0.50),
            (f'{ns}.yolo_depth_scale', 1.0),
            (f'{ns}.camera_optical_frame', 'uav/camera_optical_frame'),
            (f'{ns}.publish_debug', True),
            (f'{ns}.overlay.base_is_frd', True),
            (f'{ns}.overlay.show', True),
            (f'{ns}.overlay.thickness', 2),
            (f'{ns}.overlay.font_scale', 0.7),
            (f'{ns}.torch_device', 'auto'),   # 'auto'|'cpu'|'cuda:0'
            (f'{ns}.torch_half', True),       # use half on CUDA
            (f'{ns}.imgsz', 640),             # model inference size
            (f'{ns}.max_det', 3),             # keep NMS small
        ])

        self.ns = ns
        self.bridge = CvBridge()
        self.K = None

        qos = QoSProfile(depth=2)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        qos.history = HistoryPolicy.KEEP_LAST
        qos.durability = DurabilityPolicy.VOLATILE

        qos_rel = QoSProfile(depth=10)

        self.sub_img = self.create_subscription(Image, 'camera/image_raw', self.cb_img, qos)
        self.sub_info = self.create_subscription(CameraInfo, 'camera/camera_info', self.cb_info, qos)
        self.sub_fused = self.create_subscription(
            PoseWithCovarianceStamped, 'perception/relative_pose_fused', self.cb_fused, qos_rel)
                
        self.pub_pose = self.create_publisher(PoseWithCovarianceStamped, 'perception/yolo/relative_pose', 10)
        self.status_pub = self.create_publisher(String, 'perception/yolo/status', 1)
        self.pub_dbg = self.create_publisher(Image, 'perception/debug/yolo_overlay', 1)

        self.backend = make_backend(self)

        # NEW: pick a device/precision now
        ready = False
        try:
            if hasattr(self.backend, "warmup"):
                ready = bool(self.backend.warmup())
        except Exception as e:
            self.get_logger().warn(f"YOLO warmup failed: {e}")
       
        # now log AFTER warmup
        try:
            bi = self.backend.info() if hasattr(self.backend, "info") else {"backend": type(self.backend).__name__}
            self.get_logger().info(f"YOLO backend info: {bi}")
            if bi.get("valid", False) and not bi.get("ready", False):
                self.get_logger().warn("YOLO backend not ready yet; will resolve device on first frame.")
            elif not bi.get("valid", False):
                self.get_logger().error("YOLO backend is not valid. Check backend param and model_path.")
        except Exception as e:
            self.get_logger().warn(f"Could not retrieve backend info: {e}")

        self.latest_fused = None
        self.rel_flag = False
        self.meas_age = float('nan')

        self.get_logger().info(f"torch.cuda.is_available={torch.cuda.is_available()}")


    def P(self, key):
        return self.get_parameter(f'{self.ns}.{key}').value
    
    def _to_flu(self, p, base_is_frd: bool):
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

    def cb_img(self, msg: Image):

        if self.K is None: return
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge error: {e}'); return

        dets = []
        try:
            dets = self.backend.infer(img, msg.header.stamp)
            #self.get_logger().info(f'Yolo model infers: {dets}')
        except Exception as e:
            self.get_logger().error(f'backend inference error: {e}')


        cls = self.P('class_name'); score_th = float(self.P('score_thresh'))
        fx, fy, cx, cy = self.K[0,0], self.K[1,1], self.K[0,2], self.K[1,2]
        plate = float(self.P('plate_size_m'))

        # before the loop, if you plan to draw:
        do_overlay = bool(self.P('publish_debug')) and bool(self.get_parameter(f'{self.ns}.overlay.show').value)
        if do_overlay:
            annotated = img.copy()
            H_img, W_img = annotated.shape[:2]
        thick = int(self.get_parameter(f'{self.ns}.overlay.thickness').value)
        fscale = float(self.get_parameter(f'{self.ns}.overlay.font_scale').value)


        had = False
        for d in dets:

            if (d.class_name != cls) or (d.score < score_th): 
                continue

            
            had = True
            u, v = d.cx, d.cy
            yaw = np.arctan2(u - cx, fx)
            pitch = np.arctan2(v - cy, fy)
            

            # depth from shorter side
            px_w, px_h = d.w_px, d.h_px
            px_side = min(px_w, px_h)

            # pick the matching focal for that axis
            f_side = fx if px_side == px_w else fy

            rng = float('nan')

            if px_side > 1.0:
                rng = (f_side * plate) / float(px_side)

            # Apply correction scale
            rng *= float(self.P('yolo_depth_scale'))

            # If no reliable depth, skip publishing 3D point
            if not math.isfinite(rng):
                continue
            
            xyz = self._bearingrange_xyz(pitch,yaw,rng)

            pose = PoseWithCovarianceStamped()
            pose.header = Header()
            pose.header.frame_id = self.P('camera_optical_frame')
            pose.header.stamp = msg.header.stamp

            pose.pose.pose.position.x = float(xyz[0])
            pose.pose.pose.position.y = float(xyz[1])
            pose.pose.pose.position.z = float(xyz[2])
            # Orientation from R:
            pose.pose.pose.orientation.w = 1.0
            pose.pose.pose.orientation.x = 0.0
            pose.pose.pose.orientation.y = 0.0
            pose.pose.pose.orientation.z = 0.0
            self.pub_pose.publish(pose)


            if do_overlay:
                x1 = int(round(d.cx - d.w_px / 2))
                y1 = int(round(d.cy - d.h_px / 2))
                x2 = int(round(d.cx + d.w_px / 2))
                y2 = int(round(d.cy + d.h_px / 2))
                # clamp to image bounds
                x1 = max(0, min(W_img - 1, x1)); y1 = max(0, min(H_img - 1, y1))
                x2 = max(0, min(W_img - 1, x2)); y2 = max(0, min(H_img - 1, y2))

                # color: green for target class, orange if anything else
                color = (0, 200, 0)

                # rectangle
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thick)

                # label text: "<class> <score> Z=..m"
                z_txt = ("-" if not math.isfinite(rng) else f"{rng:.2f}m")
                label = f"{d.class_name} {d.score:.2f}  Z={z_txt}"

                # draw a filled text background for readability
                (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fscale, max(1, thick-1))
                tx, ty = x1, max(0, y1 - 5)
                cv2.rectangle(annotated, (tx, ty - th - 4), (tx + tw + 4, ty + 2), (0, 0, 0), -1)
                cv2.putText(annotated, label, (tx + 2, ty - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, fscale, (255, 255, 255), max(1, thick-1), cv2.LINE_AA)
                
                if self.latest_fused is not None:

                    text1 = f"Rover Pos: X={self.latest_fused[0]:.2f}, Y={self.latest_fused[1]:.2f}, Z={self.latest_fused[2]:.2f}"
                    
                    cv2.putText(annotated, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 2, cv2.LINE_AA)
                    
                # 2) Also show camera-space measurement (quick sanity)
                text3 = f"Cam Pos: x={float(xyz[0]):.2f} y={float(xyz[1]):.2f} z={float(xyz[2]):.2f} m"
                cv2.putText(annotated, text3, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 200, 0), 2, cv2.LINE_AA)


                try:
                    dbg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
                    dbg.header = pose.header
                    self.pub_dbg.publish(dbg)
                except Exception:
                    pass

        self.status_pub.publish(String(data= 'ok' if had else 'no-detection'))

    def _bearingrange_xyz(self, pitch: float, yaw: float, range: float):

        # 3D position in camera optical frame
        x = np.tan(yaw) * range
        y = np.tan(pitch) * range

        xyz = np.array([x, y, range], dtype=float)

        return xyz


def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()