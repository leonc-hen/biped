# Requirements

Rules for this file are at the bottom. Every row has an ID, a number, and a way to verify it.
Values marked (start) are starting guesses to be refined with Phase 2 data.

## 1. Requirements

| ID | Requirement | Target | Verified by | Where |
|---|---|---|---|---|
| R01 | Total mass, including battery, excluding payload | 2.0 kg target, 2.25 kg hard cap | Kitchen scale | Hardware |
| R02 | Hip height range | 15 to 30 cm (start) | CAD, then measurement | CAD, hardware |
| R03 | Wheel diameter | 50 to 70 mm | CAD | CAD |
| R04 | Forward speed | Up to 1.0 m/s (start) | Sim, then logged hardware run | Both |
| R05 | Turn rate | Up to 3 rad/s (start) | Sim, then logged hardware run | Both |
| R06 | Jump | Center of mass rises 25 cm from the standing pose (start) | Sim under randomization, then video | Both |
| R07 | Push recovery | Survives an initial 20 degree tilt and a sudden 0.4 m/s velocity change | Sim success rate | Sim, then hardware |
| R08 | Fall recovery | Gets up from lying on its back or side within 8 s, 90 percent success across randomized sims | Sim success rate | Sim |
| R09 | Payload | Carries 100 to 200 g without losing balance | Hardware test | Hardware |
| R10 | Runtime | At least 15 minutes of active use | Battery test | Hardware |
| R11 | Policy rate | 100 Hz, inner motor loops much faster | Firmware timing log | Hardware |
| R12 | Latency, sensor to actuator | 10 ms or less | Oscilloscope or logged timestamps | Hardware |
| R13 | Budget cap | $1,000 USD including shipping, tax and spares; tools tracked separately | Receipts | Hardware |
| R14 | Navigation accuracy (stretch goal) | Reach a waypoint within 10 cm | Hardware test | Hardware |

Reference numbers for motor sizing: wheel speed at 1.0 m/s is about 40 rad/s for a 50 mm wheel,
33 rad/s for 60 mm, and 28.6 rad/s for 70 mm.

## 2. Pass thresholds per phase (proposed, adjust at each gate)

Phase 1, dynamics and LQR
- The derivation is written up and every symbol maps to an entry in params.yaml.
- The linear model matches the nonlinear sim for tilts under 7.5 degrees over 1 s.
- LQR balances the Level 0 model from a 15 degree tilt and settles within 2 s.
- Q and R choices are documented, and the actuators do not saturate in the nominal case.

Phase 2, simulation and RL (Gate A)
- Under full randomization (Level 3), R07 succeeds in at least 95 percent of trials.
- R08 succeeds in at least 90 percent of trials.
- R06 is met in at least 90 percent of trials.
- Speed and turn commands (R04, R05) are tracked within 10 percent.
- Control delay is randomized over at least 0 to 2 policy steps.
- The actuator specification is frozen and the budget is approved.

Phase 3, CAD and design (Gate B)
- CAD mass is within 10 percent of the mass in params.yaml and under the R01 cap.
- Link lengths in CAD match params.yaml within 1 mm.
- The full bill of materials fits the R13 budget with a 10 percent contingency.
- Every part has a link, a price and a lead time.

Phase 4, hardware (Gate C)
- The robot balances untethered for 60 s.
- Real logs match sim logs within agreed tolerances (decide the numbers at the start of Phase 4).
- R01, R04, R05, R09, R10, R11, R12 and R13 are verified on hardware.

## 3. Not in scope

Uneven terrain, outdoor use, stairs, rain or dust, running with no tether during early tests,
and anything that needs a requirement not listed above.
(Add the pickup arm here if you decided it is not part of version one.)

## 4. Rule for changing requirements

A requirement can change only at a gate (A, B or C). Record the date, the old value, the new value,
and the reason in decisions.md. Do not change a requirement because a test is failing.