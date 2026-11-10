import numpy as np
from params import load_params
from yaw import yaw_inertia, yaw_gain, design_rate_pi, split_torque

p = load_params()
d, r = p.track_width_m, p.wheel_radius_m

# ---- Check 1: yaw inertia from first principles ----
M_w, I_w = p.wheel_mass_total_kg, p.wheel_inertia_total_kgm2
# Each wheel: mass M_w/2 at d/2 from the axis, spin inertia I_w/2 spinning at (d/2)/r times yaw rate
ke_per_unit = 2 * (0.5 * (M_w / 2) * (d / 2) ** 2 + 0.5 * (I_w / 2) * (d / (2 * r)) ** 2)
I_wheels_direct = 2 * ke_per_unit            # because KE = 0.5 * I * yawrate^2
print("Check 1: wheel part of yaw inertia")
print("  direct from kinetic energy:", I_wheels_direct, " formula M_eff*d^2/4:", p.m_eff * d**2 / 4)
print("  total yaw inertia:", yaw_inertia(p), "kg m^2")
print("  yaw gain K:", yaw_gain(p), "rad/s^2 per N m")

# ---- Check 2: yaw rate loop step response (linear, no saturation) ----
wn, zeta = 15.0, 1.0
Kp, Ki = design_rate_pi(p, wn, zeta)
K = yaw_gain(p)
print("Check 2: yaw rate PI, wn =", wn, "zeta =", zeta, " Kp =", round(Kp, 4), "Ki =", round(Ki, 4))
dt, ref = 1e-3, 2.0
rate, integ, peak_delta, hist = 0.0, 0.0, 0.0, []
for k in range(int(1.5 / dt)):
    err = ref - rate
    integ += err * dt
    delta = Kp * err + Ki * integ
    peak_delta = max(peak_delta, abs(delta))
    rate += K * delta * dt
    hist.append(rate)
hist = np.array(hist)
t = np.arange(len(hist)) * dt
settle = t[np.where(np.abs(hist - ref) > 0.02 * ref)[0][-1]] + dt
print("  overshoot (percent):", 100 * (hist.max() - ref) / ref)
print("  2 percent settling time (s):", settle)
print("  peak differential torque for a 2 rad/s step (N m):", peak_delta)

# ---- Check 3: balance priority when a motor is near its limit ----
print("Check 3: priority. Balance 0.9 N m total, wanted differential 0.4 N m, wheels at rest")
tR, tL, s = split_torque(p, 0.0, 0.0, 0.9, 0.4)
print("  tau_R =", tR, " tau_L =", tL, " differential scale =", s)
print("  balance torque delivered (tau_R + tau_L):", tR + tL)
tR, tL, s = split_torque(p, 0.0, 0.0, 1.4, 0.4)
print("  balance 1.4 total (over the 1.0 limit): tau_R =", tR, " tau_L =", tL, " scale =", s)