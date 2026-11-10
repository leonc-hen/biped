import numpy as np
from params import load_params
from model import linear_AB
from actuator import torque_limit, torque_mode, voltage_mode

p = load_params()
n, r = p.motor_count, p.wheel_radius_m
Kt, Ke, R = p.motor_kt_nm_per_a, p.motor_ke_vs_per_rad, p.winding_resistance_ohm
bf = p.viscous_friction_nm_s_per_rad
A, B = linear_AB(p)

print("Check 1: torque limit per motor (N m)")
print("  w_rel (rad/s)   motoring   braking")
for w in [0, 50, 100, 140, 200, 240]:
    print(f"  {w:8d}      {torque_limit(p, w, +1):7.3f}   {torque_limit(p, w, -1):7.3f}")
print("  no load speed (rad/s):", p.battery_voltage_v / Ke, " stall current (A):", p.battery_voltage_v / R)

print("Check 2: saturation. Command 2.0 N m total at rest")
tau, cur = torque_mode(p, np.zeros(4), 2.0)
print("  applied total torque:", tau, " current per motor:", cur)

row = np.array([[0, 1 / r, 0, -1.0]])   # w_rel = xdot/r - thetadot

A_t = A + B @ (-n * bf * row)
print("Check 3: open loop poles")
print("  torque mode (with friction): ", np.round(np.linalg.eigvals(A_t), 3))
c = n * (Kt * Ke / R + bf)
A_v = A + B @ (-c * row)
print("  voltage mode, V = 0:         ", np.round(np.linalg.eigvals(A_v), 3))
print("  voltage mode damping c =", c, "N m s/rad")

print("Check 4: voltage mode at rest, 24 V vs torque mode limit")
tau_v, cur_v = voltage_mode(p, np.zeros(4), 24.0)
print("  voltage mode applied total torque:", tau_v, " current per motor:", cur_v)