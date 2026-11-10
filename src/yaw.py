import numpy as np
from actuator import torque_limit

def yaw_inertia(p):
    """Yaw inertia about the vertical axis: body part plus the wheels swinging around it."""
    d = p.track_width_m
    return p.yaw_body_inertia_kgm2 + p.m_eff * d**2 / 4.0

def yaw_gain(p):
    """Yaw acceleration (rad/s^2) per N m of differential torque (tau_R - tau_L)."""
    d, r = p.track_width_m, p.wheel_radius_m
    return d / (2.0 * r * yaw_inertia(p))

def design_rate_pi(p, wn, zeta):
    """PI gains for the yaw rate loop. Plant: yaw_accel = K * delta."""
    K = yaw_gain(p)
    return 2.0 * zeta * wn / K, wn**2 / K     # Kp, Ki

def wheel_speeds(p, s, yaw_rate):
    """Relative wheel speeds (wheel minus body) for the right and left wheel."""
    r, d = p.wheel_radius_m, p.track_width_m
    w_R = s[1] / r + yaw_rate * d / (2 * r) - s[3]
    w_L = s[1] / r - yaw_rate * d / (2 * r) - s[3]
    return w_R, w_L

def split_torque(p, w_R, w_L, u_bal, delta):
    """Split balance torque (total) and differential torque into right and left
    motor torques. Balance has priority. Returns (tau_R, tau_L, scale)."""
    hi_R, lo_R = torque_limit(p, w_R, +1), -torque_limit(p, w_R, -1)
    hi_L, lo_L = torque_limit(p, w_L, +1), -torque_limit(p, w_L, -1)
    half_b = float(np.clip(u_bal / 2.0, max(lo_R, lo_L), min(hi_R, hi_L)))
    hd = delta / 2.0
    s = 1.0
    if hd != 0.0:
        for c, lo, hi in ((hd, lo_R, hi_R), (-hd, lo_L, hi_L)):
            margin = (hi - half_b) / c if c > 0 else (lo - half_b) / c
            s = min(s, margin)
    s = max(s, 0.0)
    return half_b + s * hd, half_b - s * hd, s