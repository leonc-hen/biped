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
### D02 Compute
Status: Decided
Decision: an STM32 microcontroller with a hardware floating point unit and a CAN peripheral runs the low level control and the policy. Starting recommendation: an STM32G4 series part (for example the G474, which has FDCAN, a Cortex-M4F core at 170 MHz, and math accelerators). Verify the exact part and a dev board before buying. A UART port is reserved for an optional companion computer (for example a Raspberry Pi) for vision later, using the same policy interface.
Reasons: a two layer, 64 unit network is tiny and runs easily at 100 to 200 Hz on a core with a floating point unit; hard real time timing with hardware timers and DMA gives more predictable latency for R11 and R12; FDCAN gives faster bus rates; matches the embedded C++ learning goal; large ecosystem and a clear upgrade path (an H7 part if more compute is needed).
Rejected: ESP32-S3 (built in CAN and wireless, but less deterministic timing and a weaker real time toolchain); Raspberry Pi as the main controller (not real time, jittery latency).
Tradeoffs accepted: no built in WiFi or Bluetooth, so wireless telemetry needs a separate module or the companion computer; a steeper setup with CubeMX and the toolchain; a dev board costs more than an ESP32 board.
Revisit trigger: if firmware timing cannot meet R11 and R12, move to a higher performance STM32 (H7 series).

### D03 Actuator bus
Status: Decided
Decision: CAN. The STM32 CAN peripheral (FDCAN on G4 parts) needs an external 3.3 V CAN transceiver chip, so add one to the parts list (verify the exact part before buying).CubeMars AK actuators support servo and MIT control modes.
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

## Deferred to Phase 2 data (decide at Gate A)

Leg topology (knee motor at the knee, or both motors at the hip with a linkage to keep the leg light); exact actuator models and gear ratios; leg lengths; wheel size; battery capacity.
Deadline: Gate A.

## Deferred to Phase 3 (decide at Gate B)

Frame materials; connectors; wiring layout; exact wheel motor and driver (must support current control, see D04); gripper design.
Deadline: Gate B.