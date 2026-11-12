"""Simulated IMU (gyro and accelerometer) and wheel encoders for the planar model."""
import numpy as np
from model import deriv

def read_sensors(p, s, tau, rng):
    """Simulated IMU and encoders at the instant the state is s and the wheel torque is tau.
    Returns (gyro, ax_body, az_body, wheel_speed_relative_to_body)."""
    _, xd, th, thd = s
    h, g, r = p.imu_height_m, p.gravity_ms2, p.wheel_radius_m
    _, xdd, _, thdd = deriv(p, s, tau)
    # World acceleration of the IMU point (planar: x forward, y up)
    ax = xdd + h * np.cos(th) * thdd - h * np.sin(th) * thd**2
    ay = -h * np.sin(th) * thdd - h * np.cos(th) * thd**2
    # Specific force (what an accelerometer feels) is acceleration minus gravity
    fx_w, fy_w = ax, ay + g
    # Body axes: forward axis (cos th, -sin th), up axis (sin th, cos th)
    fx = fx_w * np.cos(th) - fy_w * np.sin(th)
    fz = fx_w * np.sin(th) + fy_w * np.cos(th)
    gyro = thd + p.gyro_bias_rad_s + p.gyro_noise_rad_s * rng.standard_normal()
    fx += p.accel_noise_ms2 * rng.standard_normal()
    fz += p.accel_noise_ms2 * rng.standard_normal()
    w_rel = xd / r - thd + p.encoder_noise_rad_s * rng.standard_normal()
    return gyro, fx, fz, w_rel
