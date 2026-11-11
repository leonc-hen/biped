import numpy as np
from params import load_params
from control import design_lqr
from sim import simulate

p = load_params()
K, _ = design_lqr(p, [5, 1, 50, 1], 100)
RATES = [50, 100, 200, 500, 1000]

def recovered(theta0, rate_hz, extra_delay_s=0.0, latency_samples=0):
    """Recovered = did not fall and is back near upright after 4 s.
    (The slow velocity poles near -1 rad/s need a few seconds to finish settling.)"""
    r = simulate(p, K, theta0=theta0, rate_hz=rate_hz, delay_s=extra_delay_s,
                 latency_samples=latency_samples, t_end=4.0, dt=2e-4)
    return (not r["fell"]) and abs(r["final_theta"]) < 0.05

print("Check 1: delay margin versus loop rate (extra delay on top of the sample hold)")
print("  largest extra delay in ms, scanned 0 to 30 ms in 2 ms steps, stops at first failure")
delays = [i * 0.002 for i in range(16)]
for th0 in [0.3, 1.0]:
    print("  start tilt", th0, "rad")
    for hz in RATES:
        margin = -1.0
        for d in delays:
            if recovered(th0, float(hz), d):
                margin = d
            else:
                break
        print(f"    {hz:5d} Hz   {margin * 1000:3.0f} ms   (sample period {1000 / hz:.0f} ms)")

print("Check 2: one full sample of latency (command used one tick after the measurement)")
print("  largest recovered start tilt in rad, scan 0.1 to 1.2")
tilts = [round(0.1 * i, 1) for i in range(1, 13)]
for hz in RATES:
    best = 0.0
    for th in tilts:
        if recovered(th, float(hz), 0.0, 1):
            best = th
        else:
            break
    print(f"    {hz:5d} Hz   {best:.1f}")