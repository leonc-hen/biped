# Gates and exit checklists

Rules

1. A phase is finished only when every box in its exit checklist is ticked.
2. Gates A, B and C add extra pass criteria. If a criterion fails, either fix it and retest, or change the requirement with a recorded reason (see the change rule in requirements.md). Never move on with a failed criterion and no record.
3. Requirements and architecture decisions can change only at a gate. Record the date, the old value, the new value and the reason in decisions.md.
4. Gate review procedure: fill in the result and date next to each criterion, write a BUILDLOG.md entry titled with the gate name, create a git tag (gate-a, gate-b, gate-c), and push the tag.

Thresholds below match requirements.md section 2. If one changes, change both files at the same time.

## Phase 0 exit (handoff to Phase 1)

- [ ] The environment checks pass: MuJoCo and viewer, JAX backend is gpu, PyTorch CUDA is True with sm_120, and the CartpoleBalance training test finishes
- [ ] Exact versions of MuJoCo, JAX and PyTorch are in BUILDLOG.md and the lock file is committed
- [ ] The repo is on GitHub and the local folder matches it
- [ ] conventions.md, requirements.md, decisions.md, safety.md, gates.md, risk_register.md, tools.md and log_format.md exist
- [ ] params.yaml generates an MJCF that loads and falls under gravity, and all tests pass
- [ ] The five reading notes and papers.md are done and no TO CONFIRM marker is left unexplained
- [ ] The tools access plan is written and the budget sheet has Tools and Consumables rows
- [ ] python -m tools.handoff_check passes

## Phase 1 exit (dynamics and LQR)

- [ ] The derivation is written up with every step shown and every assumption listed
- [ ] Every symbol maps to an entry in params.yaml
- [ ] The linear model matches the nonlinear sim for tilts under 7.5 degrees over 1 s
- [ ] LQR balances the Level 0 model from a 15 degree tilt and settles within 2 s
- [ ] The Q and R choices are documented
- [ ] The actuators do not saturate in the nominal case
- [ ] Plots are saved in docs/figures with dated names
- [ ] BUILDLOG.md entry written and everything committed and pushed

## Phase 2 exit and Gate A (simulation and RL)

Phase 2 exit checklist

- [ ] Fidelity Levels 1, 2 and 3 are built (see requirements.md section 5), each with its test
- [ ] Every training run has its config, seed and result recorded
- [ ] An ablation table exists (for example, removing delay randomization and the change in success rate)
- [ ] Failure videos are saved
- [ ] interface.md (observation and action definitions) is final

Gate A pass criteria (all at Level 3, full randomization)

| Criterion | Threshold | Result | Date |
|---|---|---|---|
| R07 push recovery | 95 percent or more of trials succeed | | |
| R08 fall recovery | 90 percent or more of trials succeed | | |
| R06 jump | 90 percent or more of trials succeed | | |
| R04 and R05 command tracking | within 10 percent | | |
| Control delay randomized | at least 0 to 2 policy steps | | |
| Actuator specification frozen | models, gear ratios and leg lengths written in params.yaml | | |
| Budget approved | BOM estimate fits R13 | | |
| Gate A report written | pass or fail against every row | | |

If R08 fails: go to the fallbacks in decisions.md D01 in order before buying anything.

## Phase 3 exit and Gate B (CAD and design)

Phase 3 exit checklist

- [ ] CAD assembly is complete and params.yaml is updated from the CAD values
- [ ] Weight budget is up to date
- [ ] Wheel motor and driver chosen (the driver supports current control, decisions.md D04)
- [ ] Connectors, wiring layout and frame materials chosen
- [ ] Lab access and battery rules confirmed (tools.md questions answered)
- [ ] BUILDLOG.md entry written and everything committed and pushed

Gate B pass criteria

| Criterion | Threshold | Result | Date |
|---|---|---|---|
| CAD mass | within 10 percent of params.yaml and under the R01 cap | | |
| CAD link lengths | match params.yaml within 1 mm | | |
| Bill of materials | fits R13 with a 10 percent contingency | | |
| Every part | has a link, a price and a lead time | | |
| Safety review | kill switch, current limits, fuse and connector ratings planned (safety.md) | | |
| Parts ordered | long lead items ordered first | | |

## Phase 4 exit and Gate C (hardware)

Phase 4 exit checklist

- [ ] Bring up checklist ticked and dated
- [ ] Real logs use the same format as sim logs
- [ ] Sim versus hardware overlay plots saved
- [ ] Videos of tests and failures saved
- [ ] Everything that broke is recorded with its fix

Gate C pass criteria

| Criterion | Threshold | Result | Date |
|---|---|---|---|
| Balance | untethered for 60 s | | |
| Real logs versus sim | within tolerances written in docs/tolerances.md at the start of Phase 4 | | |
| R01 mass | 2.0 kg target, 2.25 kg hard cap, measured | | |
| R04 and R05 | speed and turn rate verified in logged runs | | |
| R09 payload | 100 to 200 g carried without losing balance | | |
| R10 runtime | 15 minutes or more of active use | | |
| R11 policy rate | 100 Hz in the firmware timing log | | |
| R12 latency | 10 ms or less, measured with GPIO toggles and a logic analyzer | | |
| R13 budget | receipts total at or under the cap | | |