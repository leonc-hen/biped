# Conventions

These rules never change. Any change needs an entry in decisions.md.

## Units
SI only: meters, kilograms, seconds, radians, newton meters, volts, amps.
Degrees appear only on plots, never in code or logs.

## Frames
Right handed. x forward, y left, z up.
Body frame origin: midpoint between the two hip joints.
World frame: same axes, origin on the ground where the robot starts.

## Body angles
Euler order: yaw, then pitch, then roll (intrinsic ZYX).

| Angle | Axis | Positive means |
|---|---|---|
| Yaw | +z | turning left (counterclockwise seen from above) |
| Pitch | +y | leaning forward (top of the robot moves toward +x) |
| Roll | +x | leaning right (left side rises) |

## Joints
Every joint axis is +y in the body frame. Positive rotation follows the right hand rule.

| Joint | Zero position | Positive direction |
|---|---|---|
| Hip pitch | thigh pointing straight down | foot swings backward |
| Knee pitch | shank in line with thigh (straight leg) | foot swings backward relative to the thigh |
| Wheel | n/a | robot rolls forward (v = omega * wheel radius) |

Left and right joints use the same sign in all code (no negation).
Physical motor mirroring is handled only in firmware, at the hardware boundary.
Sketch: docs/figures/joint_conventions.png
![Joint conventions] (figures/joint_conventions.png)

## Quaternions
Order inside our code: w, x, y, z (the MuJoCo order).
SciPy uses x, y, z, w. Convert only inside tools/rotations.py.
Quaternions are stored with w >= 0.

## IMU
The accelerometer reads specific force, so at rest and level it reads +g along the up axis.
Each IMU has a matrix R_body_from_imu stored in params.yaml once the part is chosen.
All IMU data is converted to the body frame in tools/rotations.py, nowhere else.

## Time
Physics step: 2 ms (500 Hz).
Policy and control rate: 10 ms (100 Hz), so 5 physics steps per policy step.
Logs are written at the policy rate. Motor inner loops run faster, in the drivers.

## Torque and current
A positive torque command produces positive torque about the joint's +y axis.
For the wheels, positive torque pushes the robot forward.
Torque = torque constant * current, and the sign is never flipped in between.