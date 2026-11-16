"""JAX port of src/leg_actuator.py. Must match it exactly; step16 check 2 enforces
the parity. Edit both files together or not at all."""
import jax.numpy as jnp


def torque_envelope(qvel, peak_Nm, max_speed_rad_s, v_scale=1.0):
    frac = jnp.clip(1.0 - jnp.abs(qvel) / max_speed_rad_s, 0.0, 1.0)
    return peak_Nm * frac * v_scale, peak_Nm * v_scale


def pd_torque(q, qvel, q_target, kp, kd, peak_Nm, max_speed_rad_s, v_scale=1.0):
    tau = kp * (q_target - q) - kd * qvel
    drive, brake = torque_envelope(qvel, peak_Nm, max_speed_rad_s, v_scale)
    limit = jnp.where(tau * qvel > 0.0, drive, brake)
    return jnp.clip(tau, -limit, limit)