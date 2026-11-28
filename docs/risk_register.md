# Risk register

| ID | Risk | Evidence | Mitigation | Check at | Status |
|---|---|---|---|---|---|
| RK01 | Leg gearbox shock at landing | Landing leg torque reaches the 4.5 N m envelope top on almost every jump (D24) | Get the AK40-10 shock rating (CAD-10); soft tires; softer landing pose | CAD-10, Gate A | Open |
| RK02 | Layout A fold clearance | Fold blocked from knee 2.25 rad with the D08 shell; needs >= 0.26 m leg spacing with today's rails (D30) | CAD-2 rule; fold_clearance.py; strict RUN-7 gate (0.7 every rest pose) | RUN-2, RUN-7 | Open |
| RK03 | Mass near the R01 cap | About 2.18 kg with RMD-L-5010 wheels and the 150 g sensor bay, cap 2.25 kg | Real masses in CAD-9; shrink the bay first | CAD-9 | Open |
| RK04 | Leg motor standing load | 77 percent of rated on both motors of each leg at 1.924 kg, more near the cap (D19, D20) | Bench thermal test | Bring up | Open |
| RK05 | Belt slip, stretch or backlash | About 5 percent belt loss and compliance not modelled (D20) | Measure belt loss in bring up | Bring up | Open |
| RK06 | Control delay cliff | Failures jump between 12 and 14 ms delay (D23); R12 is 10 ms | Latency budget in firmware; measure IMU to command | Firmware | Open |
| RK07 | Wheel to torso or top shell contact not simulated | Wheels collide with the rails only (D27) | Check reach on the new shell | RUN-2 | Open |
| RK08 | Motor overspeed not braked in sim | Belt motor drags the hip motor to 72.6 rad/s against 45.6 on an airborne leg (D20) | Scripted jumps overshoot at most 2 percent; recheck after retrain | RUN-6 | Open |
