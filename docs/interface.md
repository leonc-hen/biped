# Policy interface, v0.1

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
| 7 | mode flags: jump trigger, get up | 2 | 0 or 1 | 1 |
Total: 27. Scales multiply the raw value. History length 1 for now; adding
history is a versioned change.

## Privileged critic extras (training only)
True base velocity (3), base height (1), contact flags (2), push force (2),
randomized parameter vector (as configured). Never in the actor.

## Actions, float32, this order
| # | Name | Dim | Mapping |
|---|---|---|---|
| 0 | hip_l, knee_l, hip_r, knee_r offset | 4 | target = stand pose + 0.5 * tanh-clipped action, rad |
| 1 | wheel_l, wheel_r torque | 2 | tau = peak_torque_Nm * tanh-clipped action |
Leg targets feed the PD defined in src/leg_actuator.py, identically in sim
and firmware.

## Conventions
As docs/conventions.md: x forward, joint axes +y, positive hip and knee swing
the foot backward, left before right everywhere.