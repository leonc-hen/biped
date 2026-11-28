# Policy interface, v0.5

Single definition for training, export and firmware. Any change bumps the
version and is a decisions.md entry. Order, units and scaling are normative.

## Rates
- Policy: 100 Hz (R11). 200 Hz is an open experiment, decide at D06 review.
- PD inner loop: every physics step in sim (500 Hz equivalent on hardware).

## Observations (actor, real robot measurable only), float32, this order
| # | Name | Dim | Units | Scale |
|---|---|---|---|---|
| 0 | gravity direction, body frame | 3 | unit vector | 1 |
| 1 | gyro, body frame | 3 | rad/s | 0.25 |
| 2 | hip_l, knee_l, hip_r, knee_r position | 4 | rad | 1 |
| 3 | leg joint velocities, same order | 4 | rad/s | 0.05 |
| 4 | wheel_l, wheel_r velocity | 2 | rad/s | 0.025 |
| 5 | previous action | 6 | action units | 1 |
| 6 | commands: vx, yaw rate, hip height | 3 | m/s, rad/s, m | 1, 0.33, 4 |
| 7 | mode flags: jump trigger (1 for 1.5 s after a jump command), get up (1 in get up mode, also set at reset) | 2 | 0 or 1 | 1 |
Total: 27. Scales multiply the raw value. History length 1 for now; adding
history is a versioned change.

## Privileged critic extras (training only)
True base velocity (3), base height (1), contact flags (2), push force (2),
randomized parameter vector (as configured). Never in the actor.

## Actions, float32, this order
| # | Name | Dim | Mapping |
|---|---|---|---|
| 0 | hip_l, knee_l, hip_r, knee_r offset | 4 | target = IK(commanded hip height) + 0.5 * y, rad |
| 1 | wheel_l, wheel_r torque | 2 | tau = peak_torque_Nm * tanh(action) |

Leg filter: y = 0.5 * tanh(action) + 0.5 * y_prev, per leg joint, at the policy
rate. IK for the symmetric pose: q_hip = -acos((h - r_wheel) / (l_thigh + l_shank)),
q_knee = -2 * q_hip. The "previous action" observation is the applied action,
with the leg part filtered. Height command range: 0.15 to 0.30 m.

Leg targets feed the PD defined in src/leg_actuator.py, identically in sim
and firmware.

## Leg drive (v0.3, D20 coaxial belt)
Motor 1 turns the hip joint. Motor 2 sits at the hip and turns the shank through a
belt, so its angle is hip + knee. Firmware maps joint targets to motor targets
(m1 = hip, m2 = hip + knee), runs one PD loop per motor on motor angle and speed
inside that motor's torque speed envelope, and reports joint angles back to the
policy (hip = m1, knee = m2 - m1). Observations and actions stay in joint space.

Target clamp (v0.4): leg joint targets are clamped to the joint limits minus 0.05 rad
before mapping to motor targets. Firmware applies the same clamp.

## Get up mode (v0.5, D25)
The get up policy is a separate network with this same observation and action
layout. The get up flag (row 7, observation index 26) is 1 while it runs and
is also set at reset; the locomotion policy runs with it at 0.

Fall handling, identical in sim and firmware:
1. On fall detection the legs go limp until the body is at rest.
2. Get up mode starts: flag to 1, the get up policy runs.
3. Handoff to the locomotion policy (s3v9, D24) when all three hold: tilt
   < 20 deg, torso height > 0.15 m, body rate < 2 rad/s (env config
   handoff_tilt, handoff_height, handoff_rate). Flag back to 0.
Open for firmware: the fall detection and "at rest" tests must match the
ones used in training (src/env_s1.py).

## Conventions
As docs/conventions.md: x forward, joint axes +y, positive hip and knee swing
the foot backward, left before right everywhere.