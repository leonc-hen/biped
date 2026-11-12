# Architecture decisions

Format: ID, status, decision, reasons, rejected alternatives, revisit trigger.
Status is Decided, or Deferred with a deadline gate.
Requirements can change only at a gate (see requirements.md). The same applies to these decisions.

## Decided now

### D01 Leg degrees of freedom
Status: Decided
Decision: two joints per leg (hip pitch and knee pitch) plus one drive wheel per leg.
Reasons: matches the four leg actuators in the budget; keeps mass and cost low; enough range for height change and jumping.
Rejected: one joint per leg (no height range, no jump); three joints per leg (adds roll but costs mass, money and wiring).
Known risk: planar legs may limit roll and fall recovery.
Fallbacks, in order: (1) add a third joint per leg; (2) change the body shape so it self rights; (3) use the arm as a prop.
Revisit trigger: if R08 (fall recovery) is below 90 percent under randomization in early Phase 2 tests.

### D02 Compute
Status: Decided
Decision: an STM32 microcontroller with a hardware floating point unit and a CAN peripheral runs the low level control and the policy. Starting recommendation: an STM32G4 series part (for example the G474, which has FDCAN, a Cortex-M4F core at 170 MHz, and math accelerators). Verify the exact part and a dev board before buying. A UART port is reserved for an optional companion computer (for example a Raspberry Pi) for vision later, using the same policy interface.
Reasons: a two layer, 64 unit network is tiny and runs easily at 100 to 200 Hz on a core with a floating point unit; hard real time timing with hardware timers and DMA gives more predictable latency for R11 and R12; FDCAN gives faster bus rates; matches the embedded C++ learning goal; large ecosystem and a clear upgrade path (an H7 part if more compute is needed).
Rejected: ESP32-S3 (built in CAN and wireless, but less deterministic timing and a weaker real time toolchain); Raspberry Pi as the main controller (not real time, jittery latency).
Tradeoffs accepted: no built in WiFi or Bluetooth, so wireless telemetry needs a separate module or the companion computer; a steeper setup with CubeMX and the toolchain; a dev board costs more than an ESP32 board.
Revisit trigger: if firmware timing cannot meet R11 and R12, move to a higher performance STM32 (H7 series).

### D03 Actuator bus
Status: Decided
Decision: CAN. The STM32 CAN peripheral (FDCAN on G4 parts) needs an external 3.3 V CAN transceiver chip, so add one to the parts list (verify the exact part before buying). CubeMars AK actuators support servo and MIT control modes.
Bus load estimate (my rough calculation, verify): six actuators, one command frame and one reply frame each per 10 ms policy step, about 130 bits per frame, gives about 1560 bits per step, or about 16 percent of a 1 Mbps bus.
Rejected: UART or RS485 per actuator (more wiring, lower rates); hobby PWM servos (no torque feedback).
Revisit trigger: if bus load, wiring noise or frame loss shows up in bring up.

### D04 Control layering
Status: Decided
Decision: the policy outputs position targets for the four leg joints, with fixed gains in the actuator loops, and a torque (current) command for each wheel.
Reasons: position targets with set gains usually transfer from sim to real more easily; the actuators' own fast loops handle the rest; wheel torque matches the LQR output.
Rejected: torque control for the legs (more flexible, harder to transfer); wheel velocity command.
Constraint for Phase 3: the wheel motor and driver must support current control.
Revisit trigger: if jumping or landing needs leg torque control.

### D05 Simulation and control rates
Status: Decided
Decision: physics at 500 Hz (2 ms), policy at 100 Hz (10 ms), so five physics steps per policy step. Firmware uses the same policy rate. Inner motor loops in the drivers run much faster.
Latency definition for R12: time from the IMU sample to the motor command being sent, 10 ms or less, measured in firmware. Simulation randomizes control delay over at least 0 to 2 policy steps.
Rejected: 200 Hz policy (revisit if balance margin or latency is poor at Gate A).

### D06 Interface contract
Status: Decided
Decision: the observation vector and action vector are defined once, in docs/interface.md and a Python module, with fixed order, units and scaling. Simulation, training and firmware all use that definition.
Reasons: mismatched ordering and scaling between sim and firmware is a common source of sim to real failures.
Revisit trigger: any change needs a version bump and a log entry.

### D07 Simulation effects are written once
Status: Provisional, confirm in Phase 2
Decision: motor model, sensor noise, delay, battery and randomization are pure functions (arrays in, arrays out, state passed explicitly), written with jax.numpy so the same code runs in plain MuJoCo debugging and in the GPU training environment.
Reasons: one implementation means no drift between debugging and training; pure functions are easy to test.
Rejected: separate numpy and JAX versions (they drift apart).
Revisit trigger: if the Playground environment structure makes this awkward when you read it in Phase 2.

2026-11-12  R08  old: back or side  new: back or front, side falls prevented  see D08
### D08 Side fall recovery route (upholds R08 as written)
Status: Decided (recorded 2026-11-12; ratify in the Gate A report)
Decision: R08 keeps "up from back or side within 10 s, 90 percent".
Side recovery is achieved by D01 fallback 2, a self righting body shape:
the shell cross section is rounded or chamfered so that side lying is
unstable, and the policy rolls the body onto its front or back using leg
pumping and wheel torque, then runs the normal get up. The shell profile
becomes a Phase 2 design variable in params.yaml, verified in simulation.
Reasons: fallback 1 (third joint per leg) adds roughly 240 g and two
actuators, which breaks R01 on top of D09; fallback 3 (arm as prop) makes
a core requirement depend on a stretch item.
Rejected: narrowing R08 to front and back only (user decision, 2026-11-12).
Revisit trigger: if S4 side recovery stays under 90 percent at Level 3
after at least two shell profile iterations, escalate to fallback 1 and
reopen the R01 mass cap at a recorded gate decision.

### D09 Leg actuator class
Status: Decided
Decision: the CubeMars AK MIT command format (position, velocity, kp, kd,
feedforward torque) stays as the control interface reference for D03 and
D04, but AK actuators are not purchase candidates. Candidates must be
150 g or less per actuator with roughly 1 to 3 N m peak torque. Shortlist
directions for the 2.9 sizing passes: small gimbal motor plus planetary
quasi direct drive units (MyActuator RMD L small sizes, GIM style), DJI
M2006 with C610 driver (about 90 g, CAN, current control), and a serial
bus servo tier (Feetech STS or Dynamixel XL) as the budget fallback,
accepting degraded torque control and likely giving up R06.
Reason: four AK class actuators at roughly 300 g each is about 1.2 kg,
which breaks R01 (1.5 kg target) before the body, battery, wheels or arm
exist.
Accepted variation to D04: if the chosen driver has no internal position
loop (the M2006 route), the fixed gain PD runs in STM32 firmware at the
inner loop rate instead of in the driver. The policy interface is
unchanged.
Revisit trigger: if no sub 150 g candidate clears the 2.9 sizing targets
with 30 percent headroom, raise the mass cap question at Gate A before
raising the actuator class.

### D10 Pinned simulation stack
Status: Decided
Decision: exact versions of mujoco, mujoco-mjx, jax, mujoco_playground and
the NVIDIA driver are recorded in the build log and committed in
requirements.lock.txt. No upgrades mid phase. Any upgrade is a new
decision entry and requires rerunning the 2.5 parity test before further
training.
Reason: closes the open Phase 0 gate item (exact versions were never
recorded) and prevents silent physics changes under trained policies.
Known risk carried from Phase 0: JAX on the RTX 5060 (Blackwell, sm_120)
is unverified; JAX must report a gpu backend before any training run
counts.

## Deferred to Phase 2 data (decide at Gate A)

Leg topology (knee motor at the knee, or both motors at the hip with a linkage to keep the leg light); exact actuator models and gear ratios; leg lengths; wheel size; battery capacity.
Deadline: Gate A.

## Deferred to Phase 3 (decide at Gate B)

Frame materials; connectors; wiring layout; exact wheel motor and driver (must support current control, see D04); gripper design.
Deadline: Gate B.


“Model: torque input with body reaction (τ/r on x, −τ on θ). Verified by sympy, energy conservation, and linear vs nonlinear comparison.” 
Yaw angle ψ is positive counterclockwise seen from above, which is a left turn. The right wheel driving faster than the left produces a positive yaw rate.
   Yaw: psi positive counterclockwise (left turn). I_psi = I_z + M_eff*d^2/4
   (corrected from d^2/2). Yaw loop: PI on rate (wn 15, zeta 1) with P on
   heading (Kh 4). Balance torque has priority over differential torque.

   Controller: LQR on [z, v, theta, thetadot], Q = diag(5,1,50,1), R = 100.
   Large R chosen for delay tolerance (R <= 10 fails near 8 ms). R = 1000 too weak
   to recover 0.2 rad. Designed on the frictionless model, tested with friction.
   Position hold uses the same matrices with z = x. Clamp the integral in velocity mode.
   Velocity tracking is slow (about 4 s). Tune Q later.

Latency budget (target: 5 ms total, measured margin about 14 ms)

IMU sample and read time              ___ ms
Estimator and filter compute          ___ ms
Controller compute                    ___ ms
CAN message to the motor driver       ___ ms
Driver current loop response          ___ ms
Average sample hold (half the period) ___ ms
----------------------------------------------
Total                                 ___ ms

   Loop rate: inner balance loop at 200 Hz minimum, 500 Hz target. 50 Hz rejected.
   Delay: measured margin about 12 to 20 ms (idealized, no noise or estimator).
   Latency target: 5 ms total or less. Budget table to be filled when hardware is chosen.
   Fall threshold of 1.3 rad is a placeholder, set it from the CAD geometry.
   My loop rate numbers differ from the original Phase 1 notes (those had 0.6 rad at 50 Hz);
   likely they included noise or an estimator.

   Estimator: complementary filter, tau 0.5 s, encoder acceleration compensation
   (20 Hz low pass), gyro bias calibrated at startup. Plain 0.5 s filter rejected:
   acceleration contamination ruins velocity tracking. 2 s filter rejected: bias error
   grows as bias * tau. Kalman filter planned as an upgrade.
   Sensor noise values are placeholders. Randomize IMU position in Phase 2.
   My numbers are smaller than the original notes (those had 7.6 deg for the plain filter).

   Actuator demands (balance alone, R = 100, placeholder params):
   0.3 rad recovery: 0.43 N m total, 2.1 A per motor, 2.4 W, 16 rad/s relative wheel speed.
   Driving 1 m/s needs 33 rad/s at the wheel (318 rpm). Standing still: negligible torque.
   Preliminary motor target with 2 to 3x margin: 0.4 to 0.6 N m per motor, 100 to 150 rad/s.
   NOT final. Jumping and landing in Phase 2 set the real requirement.

      Sweeps (nominal gains, R = 100): stable and full recovery for mass +/-20%, CoM height
   0.08 to 0.20 m, inertia x0.5 to x2, wheel radius +/-20%, torque gain x0.5 to x2.
   Redesigned tilt gain varies only -1.34 to -1.61, so no gain scheduling.
   Risks, in order: viscous friction (3x placeholder cuts recovery to 0.3 rad), forward
   payload offset (robot leans back and creeps about 0.7 m; compensate arm CoM), then Kt.
   Delay margin with estimator about 22 ms. Keep the 5 ms latency target.
   Not yet run: Coulomb friction, backlash, slope, combined sweeps.

      Baseline: baseline_lqr.json holds 7 metrics for LQR R=100 with estimator, noise,
   limits, 500 Hz. RL must be scored under identical conditions. Metrics 1 and 2 are
   limited by the placeholder 1.3 rad fall threshold. Set it from CAD and regenerate.
   Metric 7 is dominated by placeholder friction. Regenerate after any params change.

      MuJoCo cross check passed (MuJoCo 4.0.1, implicitfast, 0.5 ms, solref 0.001 1,
   solimp 0.99 0.999 0.0001, friction 1.5). Open loop tilt error 0.3 to 0.9 percent over 0.5 s,
   torque step 0.4 percent, closed loop LQR tilt error 0.17 percent (contact model).
   Removing the reaction torque gives 26 percent error, so the check is sensitive to it.
   Use the contact model as the closed loop reference. The constraint model's velocity
   error (7 percent) is constraint softness and shrinks with timestep.