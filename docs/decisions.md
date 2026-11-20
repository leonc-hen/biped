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
Status: Superseded by D19
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

### D13 Interface v0.2: residual leg targets and leg action filter
Status: Decided
Decision: docs/interface.md v0.2. Leg targets are residuals around the
symmetric IK pose for the commanded hip height, target = IK(h) + 0.5 * y,
where y is the leg action passed through a first order filter with alpha 0.5
at the policy rate. The "previous action" observation is the applied,
filtered action.
Reason: the full height command range fits within the action scale, and the
filter stops the bang-bang leg torque seen in s1_v2.
Firmware impact: one acos and one filter per leg joint per policy tick.

### D19 Leg actuator design basis: CubeMars AK40-10
Status: Decided (draft wording from repo results, pending review). Supersedes D09.
Decision: the CubeMars AK40-10 (4.1 N m peak, 1.3 N m rated, 435 rpm no load
at 24 V, 10:1, 195 g with driver) is the leg actuator design basis.
params_legged.yaml leg_motor carries its numbers.
Shortlist (tools/mass_budget.py: D20 coaxial belt leg, 155 mm legs, budget from
docs/mass_budget.csv plus 10 percent contingency on unmeasured items; jump is
the scripted rail jump, CoM rise above standing / wheel clearance; standing
load is the static torque per leg motor at the nominal pose, percent of rated;
speed is no load at 22.2 V):

| Candidate | Peak N m | rad/s | g | Total kg | R01 | Jump cm | Standing load | USD per unit |
|---|---|---|---|---|---|---|---|---|
| CubeMars AK40-10 | 4.1 | 42.1 | 195 | 1.924 | under target | 17.5 / 10.2 | 77 % | 136-163 |
| CubeMars AKE40-6 + driver | 5.5 | 32.0 | 170 | 1.824 | under target | 18.8 / 12.5 | 53 % | not listed |
| SteadyWin GIM4305-10 | 3.47 | 35.5 | 150 | 1.744 | under target | 13.8 / 8.0 | 91 % | 70-132 |
| SteadyWin GIM4310-10 V1.1 | 5.6 | 22.1 | 227 | 2.052 | under cap | 17.7 / 9.2 | 52 % | 82 + driver |
| DJI M2006 + C610 | 1.8 | 48.4 | 107 | 1.572 | under target | 8.3 / 3.2 | 82 % | 41-95 + ESC |

Numbers are at the pre-17a placeholder geometry; rerun at the 17a parameters
they move by at most 0.3 cm and 2 points.
Reasons: keeps the robot under the R01 target with a listed price and an
integrated driver, and clears the 10 cm jump target (R06 as amended, D18) with
margin. The AKE40-6 scores better on every column but has no listed price and
an assumed driver mass; the GIM4310 pushes the total over the 2.0 kg target;
the GIM4305 runs at 91 percent of rated just standing; the M2006 misses the
jump target.
Costs: standing load is 77 percent of rated at 1.924 kg, more near the
2.25 kg cap, and it sits on both motors of each leg (D20). Heavier than D09's
150 g cap and above its 1 to 3 N m band, hence the supersession. The rotor
inertia (540 g cm2) is assumed to be output side; confirm in the CubeMars manual.
Rejected: the other shortlist rows, for the reasons above.
Revisit trigger: the AKE40-6 gets a listed price and driver, or bench tests
show the standing load overheating the AK40-10.

### D20 Coaxial belt knee
Status: Decided (draft wording from repo results, pending review)
Decision: both leg motors sit on the torso at the hip. Motor 1 turns the hip
joint. Motor 2 turns the shank through a 1:1 belt along the thigh, so its
angle is hip + knee. Each motor runs its own PD on motor angle and speed inside
its own torque speed envelope (interface v0.3). In sim the belt is a fixed
tendon over hip and knee, driven by the knee actuator in torque mode
(model/build_mjcf.py).
Reasons:
- Jump (Step 15, src/actuator_candidates.py, 155 mm legs, 2.0 kg, pre-17a
  geometry, CoM rise / wheel clearance in cm, serial then coaxial): AK40-10
  11.1 / 4.9 and 14.9 / 7.6; AKE40-6 11.9 / 5.6 and 16.8 / 9.4; GIM4310-10
  8.3 / 2.4 and 14.4 / 7.1. Both motors work during the push, and the belt
  motor turns at about half the knee speed, so speed limited motors stay in
  the high torque part of their envelope.
- Mass (Step 16, tools/mass_budget.py): with all four actuators on the torso
  the legs carry only links and the belt (65 g thigh, 30 g shank), and the
  AK40-10 build totals 1.924 kg, under the R01 target (D19).
Costs:
1. Standing, the hip motor holds a torque equal and opposite to the belt
   motor's to keep the torso level, so both motors of a leg carry the full
   static knee torque (77 percent of rated for the AK40-10, D19).
2. Belt mass, about 5 percent belt loss and belt compliance are not modelled,
   so the jump numbers are slightly optimistic.
3. Firmware maps joint targets to motor targets (m1 = hip, m2 = hip + knee)
   and motor angles back to joint angles, and the per motor PD couples the
   joints: the hip sees the stiffness of both motors.
Known model gap: the torque speed envelope does not brake a motor that is
driven past its no load speed. Under full drive on an airborne leg the belt
motor drags the hip motor to 72.6 rad/s against a 45.6 limit; the scripted
jumps overshoot by at most 2 percent.
Rejected: serial leg with the knee motor at the knee (lower jump and wheel
clearance, more mass at the knee).
Revisit trigger: belt slip, stretch or backlash in bring up, or a measured
belt loss well above 5 percent.

## Deferred to Phase 2 data (decide at Gate A)

Exact wheel motor model and gear ratio; leg lengths; wheel size; battery capacity. Leg topology is decided in D20 and the leg actuator in D19.
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