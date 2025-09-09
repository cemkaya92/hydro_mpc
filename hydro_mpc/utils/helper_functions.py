
from scipy.spatial.transform import Rotation as R

def quat_to_eul(q_xyzw):
    # PX4: [w, x, y, z]
    r = R.from_quat([q_xyzw[1], q_xyzw[2], q_xyzw[3], q_xyzw[0]])
    yaw, pitch, roll = r.as_euler('ZYX', degrees=False)
    return float(roll), float(pitch), float(yaw)

def rpy_to_quat_map(roll: float, pitch: float, yaw: float):
    """
    Convert MPC RPY to a quaternion consistent with the position mapping:
    positions use (x, -y, -z), which equals a 180° rotation about X.
    Under this change of basis: roll -> roll, pitch -> -pitch, yaw -> -yaw.
    Returns [x,y,z,w] for geometry_msgs orientation fields.
    """
    r = float(roll)
    p = float(-pitch)
    y = float(-yaw)
    return R.from_euler('ZYX', [y, p, r]).as_quat()  # SciPy returns [x,y,z,w]