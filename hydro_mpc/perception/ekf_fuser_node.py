#!/usr/bin/env python3
import math, numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import Header
from geometry_msgs.msg import TransformStamped, PoseWithCovarianceStamped, PointStamped
from tf2_ros import TransformBroadcaster
from hydro_mpc.utils.perception_tf_utils import mat, vec, se3
from builtin_interfaces.msg import Time as TimeMsg


def R_from_quat(qx, qy, qz, qw):
    x,y,z,w = qx,qy,qz,qw
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)],
    ], dtype=float)

def wrap_pi(a):
    # wrap to (-pi, pi]
    return (a + np.pi) % (2*np.pi) - np.pi

class EkfFuserNode(Node):
    def __init__(self):
        super().__init__('ekf_fuser')
        ns = 'perception.ekf_fuser'
        self.ns = ns
        self.declare_parameters('', [
            (f'{ns}.q_pos', 0.02), (f'{ns}.q_vel', 0.10), (f'{ns}.q_yaw', 0.05),
            (f'{ns}.gate_NIS', 9.21), (f'{ns}.publish_tf', True),
            (f'{ns}.parent_frame', 'uav/base_link'),
            (f'{ns}.child_frame', 'rover/base_link'),
            (f'{ns}.camera_optical_frame', 'uav/camera_optical_frame'),
            (f'{ns}.extrinsics.T_uav_base_cam.R', [1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0]),
            (f'{ns}.extrinsics.T_uav_base_cam.t', [0.0,0.0,0.0]),
            (f'{ns}.use_tag_yaw', True),
            (f'{ns}.yaw_meas_std', 0.10),
            (f'{ns}.base_is_frd', True),
            (f'{ns}.extrinsics.T_tag_rover.R', [1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0]),
            (f'{ns}.extrinsics.T_tag_rover.t', [0.0,0.0,0.0]),
            (f'{ns}.publish_rate_hz', 30.0),
            (f'{ns}.predict_when_idle', True),
            (f'{ns}.prediction_timeout_sec', 3.0),
            (f'{ns}.stop_publish_after_timeout', True),
            (f'{ns}.reset_on_timeout', True),
            # covariance inflation & floors (variances = σ²)
            (f'{ns}.cov_inflation_rate_per_s', 0.5),   # scales P ∝ (1 + γ·age)^2
            (f'{ns}.pos_var_floor', 0.04),             # min pos var (m²) when “fresh”  (e.g., 0.2 m σ → 0.04 m²)
            (f'{ns}.yaw_var_floor', 0.03),             # min yaw var (rad²) fresh      (~10° σ → 0.03 rad²)
            (f'{ns}.unreliable_pos_var', 2.25),        # pos var when timed out (m²)    (e.g., 1.5 m σ → 2.25 m²)
            (f'{ns}.unreliable_yaw_var', 0.62),        # yaw var when timed out (rad²)  (~45° σ → 0.62 rad²)
            # cap the age used for inflation
            (f'{ns}.max_inflation_age_sec', 3.0),      # usually same as timeout
            # Optional global cap on diagonal variances (keeps numbers sane)
            (f'{ns}.max_variance_cap', 1e3)            # m² or rad² caps

        ])

        # State: [px,py,pz,vx,vy,vz,yaw]
        self.x = np.zeros((7,1)); self.P = np.eye(7)*1.0; self.t_last = None
        self.pub_fused = self.create_publisher(PoseWithCovarianceStamped, 'perception/relative_pose_fused', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        # Choose groups
        self.state_group = MutuallyExclusiveCallbackGroup()  # protects EKF state
        self.timer_group    = ReentrantCallbackGroup()          # lightweight I/O ok to overlap


        qos_rel = QoSProfile(depth=10); qos_rel.reliability = ReliabilityPolicy.RELIABLE
        qos_rel.history = HistoryPolicy.KEEP_LAST; qos_rel.durability = DurabilityPolicy.VOLATILE

        self.sub_aruco = self.create_subscription(
            PoseWithCovarianceStamped,
            'perception/aruco/relative_pose',
            self.cb_aruco_pose,
            qos_rel,
            callback_group=self.state_group)

        self.sub_yolo = self.create_subscription(
            PoseWithCovarianceStamped,
            'perception/yolo/relative_pose',
            self.cb_yolo_pose,
            qos_rel,
            callback_group=self.state_group)
        
        R = mat(self.get_parameter(f'{ns}.extrinsics.T_uav_base_cam.R').value)
        t = vec(self.get_parameter(f'{ns}.extrinsics.T_uav_base_cam.t').value)
        self.T_uav_base_cam = se3(R,t)

        R_tr = mat(self.get_parameter(f'{ns}.extrinsics.T_tag_rover.R').value)
        self.R_tag_rover = R_tr 

        self.is_estimation_reliable = False
        self._timed_out_prev = False   # edge-detect timeout → only reset once per timeout window

        self._infl_age_prev = 0.0

        # internal clocks
        self.t_last = None  # seconds since epoch (node clock)
        self.t_last_update = None   # last time we propagated state (seconds)
        self.t_last_meas   = None   # last time we incorporated a measurement (seconds)


        rate = float(self.get_parameter(f'{ns}.publish_rate_hz').value)
        # Timer that predicts/publishes → same group to avoid races with subs
        self.timer = self.create_timer(1.0 / max(1.0, rate), self._on_timer, callback_group=self.timer_group)
        
        
    def _sec_from_stamp(self, stamp):
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def _predict_to(self, t_now):
        # First call just initializes the clock
        if self.t_last is None:
            self.t_last = t_now
            return
        dt = t_now - self.t_last
        if dt <= 0.0:
            return
        # Clamp dt to keep the model stable after long gaps
        dt = max(1e-3, min(0.2, dt))
        self.predict(dt)
        self.t_last = t_now


    def Pget(self, key): return self.get_parameter(f'{self.ns}.{key}').value

    def predict(self, dt):
        self._ensure_P_finite()
        F = np.eye(7); F[0,3]=dt; F[1,4]=dt; F[2,5]=dt
        q_pos = float(self.Pget('q_pos')); q_vel = float(self.Pget('q_vel')); q_yaw = float(self.Pget('q_yaw'))
        
        scale_z = 0.4   # z process noise 40% of xy (tune 0.3–0.6)
        Q = np.diag([q_pos, q_pos, q_pos*scale_z,
                    q_vel, q_vel, q_vel*scale_z,
                    q_yaw])

        self.x = F @ self.x; self.P = F @ self.P @ F.T + Q
        self._symmetrize_clip()
        self._ensure_P_finite()

    
    def _set_reliability(self, reliable: bool, stamp):
        self.is_estimation_reliable = reliable

    def _apply_covariance_inflation(self, age_sec: float, unreliable: bool):
        """Age-aware covariance handling:
        - Soft inflation grows P with age using a simple scale.
        - Floors keep P from being unrealistically tiny.
        - When unreliable, bump to conservative variances.
        """
        # If age is not finite (e.g., no measurement yet), treat as timed-out
        if not np.isfinite(age_sec):
            unreliable = True
            age_sec = float(self.get_parameter(f'{self.ns}.prediction_timeout_sec').value)

        # Cap the age used for smooth inflation
        age_cap = float(self.get_parameter(f'{self.ns}.max_inflation_age_sec').value)
        age_now = max(0.0, min(age_sec, age_cap))
        
        # incremental factor since last call
        gamma = float(self.get_parameter(f'{self.ns}.cov_inflation_rate_per_s').value)
        s_prev = 1.0 + gamma * float(getattr(self, "_infl_age_prev", 0.0))
        s_now  = 1.0 + gamma * age_now
        inc = (s_now / max(1e-6, s_prev))**2
        self.P *= inc
        self._infl_age_prev = age_now

        # Floors (keep P realistic even if tiny)
        pos_floor = float(self.get_parameter(f'{self.ns}.pos_var_floor').value)
        yaw_floor = float(self.get_parameter(f'{self.ns}.yaw_var_floor').value)
        for i in range(3):
            self.P[i, i] = max(self.P[i, i], pos_floor)
        self.P[6, 6] = max(self.P[6, 6], yaw_floor)

        # Hard bump when unreliable (past timeout or never saw a meas)
        if unreliable:
            pos_var_unrel = float(self.get_parameter(f'{self.ns}.unreliable_pos_var').value)
            yaw_var_unrel = float(self.get_parameter(f'{self.ns}.unreliable_yaw_var').value)
            for i in range(3):
                self.P[i, i] = max(self.P[i, i], pos_var_unrel)
            self.P[6, 6] = max(self.P[6, 6], yaw_var_unrel)

        self._symmetrize_clip()
        self._ensure_P_finite()

    def cb_aruco_pose(self, msg: PoseWithCovarianceStamped):
        t = self._sec_from_stamp(msg.header.stamp)
        self._predict_to(t)
        p_cam = np.array([[msg.pose.pose.position.x],[msg.pose.pose.position.y],[msg.pose.pose.position.z]])
        p_uav = self.T_uav_base_cam[:3,:3] @ p_cam + self.T_uav_base_cam[:3,3:4]
        z = p_uav.reshape(3,1)
        H = np.zeros((3,7)); H[0,0]=1; H[1,1]=1; H[2,2]=1
        Rm = np.diag([0.10,0.10,0.25])

        self._nis_gate_and_update(z, H, Rm, t, msg.header.stamp, label="aruco_pose")
        
        if self.get_parameter(f'{self.ns}.use_tag_yaw').value:
            # Build camera->tag rotation from ArUco quaternion
            q = msg.pose.pose.orientation
            R_cam_tag = R_from_quat(q.x, q.y, q.z, q.w)

            # Rotate into base frame: base<-cam * cam<-tag (= base<-tag)
            R_base_cam = self.T_uav_base_cam[:3,:3]
            R_base_tag = R_base_cam @ R_cam_tag

            # If tag frame is not rover frame, apply static tag->rover rotation
            R_base_rover = R_base_tag @ self.R_tag_rover

            # Extract yaw about base Z (FLU convention)
            yaw_meas = math.atan2(R_base_rover[1,0], R_base_rover[0,0])

            # If your base_link is FRD, the yaw extraction is the same numerically,
            # but you can flip sign if you’ve defined yaw differently. Usually not needed.

            # Wrap the innovation to keep EKF stable
            yaw_pred = float(self.x[6,0])
            yaw_adj  = yaw_pred + wrap_pi(yaw_meas - yaw_pred)

            zpsi = np.array([[yaw_adj]])
            Hpsi = np.zeros((1,7)); Hpsi[0,6] = 1.0
            Rpsi = np.array([[float(self.get_parameter(f'{self.ns}.yaw_meas_std').value)**2]])

            self.update(zpsi, Hpsi, Rpsi)
            
        self.publish(msg.header.stamp)

    def cb_yolo_pose(self, msg: PoseWithCovarianceStamped):
        t = self._sec_from_stamp(msg.header.stamp)
        self._predict_to(t)
        p_cam = np.array([[msg.pose.pose.position.x],[msg.pose.pose.position.y],[msg.pose.pose.position.z]])
        p_uav = self.T_uav_base_cam[:3,:3] @ p_cam + self.T_uav_base_cam[:3,3:4]
        z = p_uav.reshape(3,1)
        H = np.zeros((3,7)); H[0,0]=1; H[1,1]=1; H[2,2]=1
        Rm = np.diag([0.2,0.2,0.4])  # rougher noise

        self._nis_gate_and_update(z, H, Rm, t, msg.header.stamp, label="yolo_pose")


    def update(self, z, H, R):
        # Joseph form: P = (I-KH)P(I-KH)^T + K R K^T
        self._ensure_P_finite()
        S = H @ self.P @ H.T + R
        
        # Regularize S if near-singular
        eps = 1e-9
        tries = 0
        while True:
            try:
                # K = P H^T S^{-1}  → solve(S, (H @ P).T).T
                HPt = H @ self.P
                K = np.linalg.solve(S, HPt).T
                break
            except np.linalg.LinAlgError:
                if tries >= 3:  # give up after a few nudges
                    self.get_logger().warn("EKF: S not SPD even after regularization; skipping update.")
                    return
                S = S + np.eye(S.shape[0]) * (eps * (10**tries))
                tries += 1

        y = z - H @ self.x
        self.x = self.x + K @ y
        I = np.eye(7); IKH = I - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T
        self._symmetrize_clip(); self._ensure_P_finite()

    def _on_timer(self):
        # predict to "now"
        t_now = self.get_clock().now().nanoseconds * 1e-9
        # predict forward if enabled
        if bool(self.get_parameter(f'{self.ns}.predict_when_idle').value):
            self._predict_to(t_now)

        # Compute age since last measurement
        if self.t_last_meas is None:
            age = float('nan')   # signals “no meas yet”; inflation will treat as unreliable with timeout
        else:
            age = max(0.0, t_now - self.t_last_meas)

        timeout = float(self.get_parameter(f'{self.ns}.prediction_timeout_sec').value)
        reliable = (np.isfinite(age) and age <= timeout)

        # Inflate covariance according to age & reliability
        self._apply_covariance_inflation(age, not reliable)

        # Publish status
        # Create a stamp for the pose equal to "now"
        sec = int(t_now); nsec = int((t_now - sec) * 1e9)
        stamp_now = TimeMsg(sec=sec, nanosec=nsec)
        self._set_reliability(reliable, stamp_now)

        # On first detection of timeout, optionally reset the EKF and hold invalid flag
        if (not reliable):
            if bool(self.get_parameter(f'{self.ns}.reset_on_timeout').value) and (not self._timed_out_prev):
                self._reset_filter("no fresh measurements (timeout)")
            self._timed_out_prev = True
        else:
            # back to healthy streaming; clear the edge flag
            self._timed_out_prev = False

        # Optionally stop publishing pose after timeout
        stop_after = bool(self.get_parameter(f'{self.ns}.stop_publish_after_timeout').value)
        if (not reliable) and stop_after:
            return  # don't publish stale pose

        # Otherwise, keep publishing predicted pose (flagged reliable=False when timed out)
        self.publish(stamp_now)

    def publish(self, stamp):
        fused = PoseWithCovarianceStamped()
        fused.header = Header()
        fused.header.frame_id = self.Pget('parent_frame')
        fused.header.stamp = stamp
        fused.pose.pose.position.x = float(self.x[0,0])
        fused.pose.pose.position.y = float(self.x[1,0])
        fused.pose.pose.position.z = float(self.x[2,0])
        # yaw only
        cy = math.cos(self.x[6,0]*0.5); sy = math.sin(self.x[6,0]*0.5)
        fused.pose.pose.orientation.w = cy
        fused.pose.pose.orientation.z = sy

        # --- DEBUG: print Euler angles in degrees ---
        # Only yaw is valid here (roll, pitch are undefined / huge covariance)
        yaw_deg = math.degrees(self.x[6,0])
        self.get_logger().info(f"Detected rover Euler angles [deg]: yaw={yaw_deg:.2f}")


        # Build 6x6 covariance: [x y z R P Y]
        cov = np.zeros((6,6), dtype=float)
        cov[0:3, 0:3] = self.P[0:3, 0:3]            # position block
        cov[5, 5] = self.P[6, 6]                    # yaw variance
        cov[3, 3] = cov[4, 4] = 1e6                 # roll/pitch unknown -> very large
        fused.pose.covariance = cov.reshape(-1).tolist()
        
        self.pub_fused.publish(fused)

        if self.Pget('publish_tf'):
            tf = TransformStamped()
            tf.header = fused.header
            tf.child_frame_id = self.Pget('child_frame')
            tf.transform.translation.x = fused.pose.pose.position.x
            tf.transform.translation.y = fused.pose.pose.position.y
            tf.transform.translation.z = fused.pose.pose.position.z
            tf.transform.rotation = fused.pose.pose.orientation
            self.tf_broadcaster.sendTransform(tf)

    def _nis_gate_and_update(self, z, H, R, t, stamp, label=""):
        # innovation and S
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        try:
            nis = float(y.T @ np.linalg.solve(S, y))
        except np.linalg.LinAlgError:
            self.get_logger().warn(f"EKF {label}: S not solvable; skipping")
            return
        gate = float(self.get_parameter(f'{self.ns}.gate_NIS').value)
        if not np.isfinite(nis) or nis > gate:
            self.get_logger().warn(f"EKF {label}: NIS reject {nis:.2f} > {gate:.2f}")
            return

        self.update(z, H, R)
        self.t_last_meas = t
        self._infl_age_prev = 0.0
        self._set_reliability(True, stamp)
        self._timed_out_prev = False  # we have a fresh accepted measurement again
        self.publish(stamp)
        
    def _symmetrize_clip(self):
        # keep P symmetric and non-negative on diagonal
        self.P = 0.5 * (self.P + self.P.T)
        max_var = float(self.get_parameter(f'{self.ns}.max_variance_cap').value)
        # Cap diagonal
        diag = np.diag(self.P).copy()
        diag = np.clip(diag, 1e-12, max_var)
        np.fill_diagonal(self.P, diag)
        # Also softly cap off-diagonals to ±max_var
        self.P = np.clip(self.P, -max_var, max_var)

    def _ensure_P_finite(self):
        if not np.all(np.isfinite(self.P)):
            self.get_logger().warn('EKF: covariance became non-finite; resetting to safe diag.')
            # Reset to conservative diagonal (position & yaw large, velocity modest)
            pos_var_unrel = float(self.get_parameter(f'{self.ns}.unreliable_pos_var').value)
            yaw_var_unrel = float(self.get_parameter(f'{self.ns}.unreliable_yaw_var').value)
            self.P = np.diag([pos_var_unrel, pos_var_unrel, pos_var_unrel,
                            1.0, 1.0, 1.0,
                            yaw_var_unrel])
            self._symmetrize_clip()

    def _reset_filter(self, reason: str = "timeout"):
        """Reset state & covariance to a conservative baseline."""
        self.get_logger().warn(f"EKF reset due to {reason}.")
        # zero state, conservative covariance (unreliable variances for pos/yaw)
        pos_var_unrel = float(self.get_parameter(f'{self.ns}.unreliable_pos_var').value)
        yaw_var_unrel = float(self.get_parameter(f'{self.ns}.unreliable_yaw_var').value)
        self.x = np.zeros((7, 1))
        self.P = np.diag([pos_var_unrel, pos_var_unrel, pos_var_unrel,
                          1.0, 1.0, 1.0,
                          yaw_var_unrel])
        self._symmetrize_clip()
        self._infl_age_prev = 0.0
        # mark estimate invalid until we accept a fresh measurement
        self.is_estimation_reliable = False
        # treat as if we never had a valid measurement
        self.t_last_meas = None

def main(args=None):
    rclpy.init(args=args)
    node = EkfFuserNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()