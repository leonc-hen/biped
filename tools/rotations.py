"""All rotation conversions. Our quaternion order is w, x, y, z."""
import numpy as np
from scipy.spatial.transform import Rotation as R


def wxyz_to_xyzw(q):
    q = np.asarray(q, dtype=float)
    return np.array([q[1], q[2], q[3], q[0]])


def xyzw_to_wxyz(q):
    q = np.asarray(q, dtype=float)
    return np.array([q[3], q[0], q[1], q[2]])


def canonical(q_wxyz):
    """Normalize and make w >= 0 (q and -q are the same rotation)."""
    q = np.asarray(q_wxyz, dtype=float)
    q = q / np.linalg.norm(q)
    return -q if q[0] < 0 else q


def quat_to_matrix(q_wxyz):
    return R.from_quat(wxyz_to_xyzw(q_wxyz)).as_matrix()


def matrix_to_quat(m):
    return canonical(xyzw_to_wxyz(R.from_matrix(m).as_quat()))


def euler_to_quat(yaw, pitch, roll):
    q_xyzw = R.from_euler("ZYX", [yaw, pitch, roll]).as_quat()
    return canonical(xyzw_to_wxyz(q_xyzw))


def quat_to_euler(q_wxyz):
    yaw, pitch, roll = R.from_quat(wxyz_to_xyzw(q_wxyz)).as_euler("ZYX")
    return yaw, pitch, roll


def imu_to_body(vec_imu, R_body_from_imu):
    """Rotate an IMU vector (accel or gyro) into the body frame."""
    return np.asarray(R_body_from_imu, dtype=float) @ np.asarray(vec_imu, dtype=float)
