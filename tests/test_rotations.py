import numpy as np

from tools.rotations import (
    euler_to_quat,
    imu_to_body,
    matrix_to_quat,
    quat_to_euler,
    quat_to_matrix,
)


def test_wxyz_order():
    # 90 degrees about z, written with w first
    q = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
    m = quat_to_matrix(q)
    assert np.allclose(m @ [1, 0, 0], [0, 1, 0])


def test_round_trip():
    yaw, pitch, roll = 0.3, 0.2, -0.1
    q = euler_to_quat(yaw, pitch, roll)
    assert np.allclose(quat_to_euler(q), [yaw, pitch, roll])
    assert np.allclose(matrix_to_quat(quat_to_matrix(q)), q)


def test_positive_pitch_leans_forward():
    m = quat_to_matrix(euler_to_quat(0.0, 0.1, 0.0))
    top = m @ np.array([0.0, 0.0, 1.0])
    assert top[0] > 0


def test_positive_roll_leans_right():
    m = quat_to_matrix(euler_to_quat(0.0, 0.0, 0.1))
    left_side = m @ np.array([0.0, 1.0, 0.0])
    assert left_side[2] > 0


def test_positive_yaw_turns_left():
    m = quat_to_matrix(euler_to_quat(0.1, 0.0, 0.0))
    forward = m @ np.array([1.0, 0.0, 0.0])
    assert forward[1] > 0


def test_imu_gravity_example():
    # Example only: an IMU mounted with its x axis pointing up,
    # y axis pointing left, so its z axis points back.
    R_body_from_imu = np.array([
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    accel_at_rest_imu = np.array([9.81, 0.0, 0.0])
    accel_body = imu_to_body(accel_at_rest_imu, R_body_from_imu)
    assert np.allclose(accel_body, [0.0, 0.0, 9.81])