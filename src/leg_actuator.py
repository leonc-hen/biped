"""Leg actuator model: clipped PD with a speed dependent torque envelope and
battery scaling. The single definition shared by the training environment (to
be ported to JAX verbatim) and the firmware spec. Pure numpy, no state; the
command delay ring buffer lives in the environment, not here.
"""
import numpy as np


def torque_envelope(qvel, peak_Nm, max_speed_rad_s, v_scale=1.0):
    """Available driving torque shrinks linearly with speed in the direction of
    motion (back EMF line) and stays at peak for braking. v_scale multiplies the
    envelope for battery sag, 1.0 at nominal voltage."""
    frac = np.clip(1.0 - np.abs(qvel) / max_speed_rad_s, 0.0, 1.0)
    drive = peak_Nm * frac * v_scale
    brake = peak_Nm * v_scale
    return drive, brake


def pd_torque(q, qvel, q_target, kp, kd, peak_Nm, max_speed_rad_s, v_scale=1.0):
    """tau = kp (q_target - q) - kd qvel, clipped to the envelope. Arrays of any
    matching shape."""
    tau = kp * (np.asarray(q_target) - np.asarray(q)) - kd * np.asarray(qvel)
    drive, brake = torque_envelope(qvel, peak_Nm, max_speed_rad_s, v_scale)
    driving = tau * np.asarray(qvel) > 0.0
    limit = np.where(driving, drive, brake)
    return np.clip(tau, -limit, limit)