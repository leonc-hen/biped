"""Wheel motor model: torque and voltage command modes with current, voltage and friction limits."""
import numpy as np

def rel_speed(p, s):
    """Wheel speed relative to the body, rad/s. This is what the motor shaft sees."""
    return s[1] / p.wheel_radius_m - s[3]

def torque_limit(p, w_rel, direction):
    """Largest torque magnitude one motor can deliver in `direction` (+1 or -1)."""
    Kt, Ke, R = p.motor_kt_nm_per_a, p.motor_ke_vs_per_rad, p.winding_resistance_ohm
    V, I = p.battery_voltage_v, p.current_limit_a
    if direction * w_rel > 0:                      # motoring: back EMF reduces the available voltage
        i_volt = max((V - Ke * abs(w_rel)) / R, 0.0)
        return Kt * min(I, i_volt)
    return Kt * I                                  # braking: current limit only

def torque_mode(p, s, tau_cmd_total):
    """Torque commanded mode. Returns (applied total torque, current per motor)."""
    n = p.motor_count
    w = rel_speed(p, s)
    tau_m = tau_cmd_total / n
    d = 1.0 if tau_m >= 0 else -1.0
    tau_m = d * min(abs(tau_m), torque_limit(p, w, d))
    current = tau_m / p.motor_kt_nm_per_a
    tau_out = tau_m - p.viscous_friction_nm_s_per_rad * w
    return n * tau_out, current

def voltage_mode(p, s, volts):
    """Voltage commanded mode (brushed motor with H bridge)."""
    n = p.motor_count
    w = rel_speed(p, s)
    V = float(np.clip(volts, -p.battery_voltage_v, p.battery_voltage_v))
    current = (V - p.motor_ke_vs_per_rad * w) / p.winding_resistance_ohm
    current = float(np.clip(current, -p.current_limit_a, p.current_limit_a))
    tau_m = p.motor_kt_nm_per_a * current
    tau_out = tau_m - p.viscous_friction_nm_s_per_rad * w
    return n * tau_out, current
