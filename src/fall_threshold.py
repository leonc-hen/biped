import numpy as np
from params import load_params

# ----------------------------------------------------------------------------
# FILL THIS IN FROM YOUR CAD.
# Measure each point of the robot body that could hit the ground first, relative
# to the AXLE CENTER, in the body frame when the robot stands upright:
#   xb = distance forward of the axle (negative means behind it), in meters
#   zb = height above the axle (negative means below it), in meters
# Include: the corners of the body and battery, the arm and gripper tips, and
# anything else that sticks out. Do NOT include the wheels.
# The values below are EXAMPLES ONLY.
# ----------------------------------------------------------------------------
POINTS = {
    "front top corner":   ( 0.06, 0.25),
    "front bottom corner": ( 0.06, 0.03),
    "rear top corner":    (-0.06, 0.25),
    "rear bottom corner":  (-0.06, 0.03),
    "arm tip (stowed)":   ( 0.10, 0.28),
}
MARGIN_RAD = 0.0          # optional safety margin to subtract

p = load_params()
r = p.wheel_radius_m

def contact_angle(xb, zb, direction):
    """Tilt magnitude (rad) at which the point first touches the ground.
    direction +1 is leaning forward, -1 is leaning backward. Returns None if never within 90 degrees.
    Height above ground = r + zb*cos(theta) - xb*sin(theta)."""
    for th in np.arange(0.0, np.pi / 2, 0.0005):
        t = direction * th
        if r + zb * np.cos(t) - xb * np.sin(t) <= 0.0:
            return float(th)
    return None

print(f"Wheel radius {r} m. Contact tilt angles (rad, and degrees):")
print(f"  {'point':22s} {'forward':>16s} {'backward':>16s}")
worst = np.pi / 2
for name, (xb, zb) in POINTS.items():
    f, b = contact_angle(xb, zb, +1), contact_angle(xb, zb, -1)
    fmt = lambda v: "never" if v is None else f"{v:6.3f} ({np.degrees(v):4.0f} deg)"
    print(f"  {name:22s} {fmt(f):>16s} {fmt(b):>16s}")
    for v in (f, b):
        if v is not None:
            worst = min(worst, v)

threshold = worst - MARGIN_RAD
print(f"\nFirst contact at {worst:.3f} rad ({np.degrees(worst):.0f} deg). Suggested value for params.yaml:")
print(f"fall_threshold_rad: {threshold:.3f}")