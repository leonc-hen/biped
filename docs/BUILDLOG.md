# Build log

## 2026-11-19, Step 18b: reward v0.5 (S2R), interface v0.4
- Reward v0.5, S2R only (S1 and S2 weights are zero): two posture costs on the
  measured leg joints q. Posture, w_posture 0.3 times the sum over the four
  joints of (q - IK(commanded height))^2. Hip symmetry, w_hip_sym 1.0 times
  (hip_l - hip_r)^2. Roll correction needs only about 0.09 rad of hip difference
  (2 cm of leg length), so it stays nearly free; a 1 rad hip stagger costs about
  1.0 per step. Builds on v0.4 (roll and pitch rate cost 0.1, sharp velocity
  kernel 0.5).
- Interface v0.4: leg joint targets are clamped to the joint limits minus
  0.05 rad (TARGET_MARGIN in src/env_s1.py) before mapping to motor targets,
  in every env config. Firmware applies the same clamp.

## 2026-11-22, Step 18: S2 closed on the real design (s2r6_e)
- s2r6_e closes S2: standing lean 1.54 deg, stagger 8.2 mm, recovers 0.6 rad
  forward and 0.7 back, pushes 1.6 forward, 2.0 back, 0.5 to 0.6 lateral m/s,
  tracks 0.3 and 1.0 m/s, heights within 0.3 mm. Random direction 0.4 m/s
  pushes 100 percent (n 1000), training distribution falls 10 percent,
  1.43 J/m at 1 m/s. Gate A R07 passed under randomization.
- On the way: s2r6_c within spec falls 13.4 percent; s2r6_d with 100 rad/s
  wheels (D22) 3.4 percent. Fixes: wheel damping, heading frame velocity,
  limit cycle, staggered stance, numerical ringing, upside down stable flat
  top, delay cliff (D23), within spec gate (D17).

## 2026-11-26, Step 19: S3 jump closed (s3v9)
- Scripted jump primitive plus residual RL, pipeline stages s3_a and s3_b.
  s3v9 (rerun): jump 93.7 percent (CI 92.0 to 95.1), CoM rise median 13.5 cm,
  R07 99.4 percent (98.3 against the unpushed twin), within spec falls
  1.5 percent. 0.15 m height stagger 10.1 mm against a 10 mm bar, accepted as
  noise (D24).
- On the way: s3v3_b jumped implausibly high, so physics settings were
  checked; s3v7_b fixed backward recovery; s3v8 jump 88.8 percent.
- Landing leg torque reaches the top of the envelope (4.5 N m) on almost
  every jump: gearbox shock risk RK01. render_policy.py added.

## 2026-11-27, Step 20: S4 get up not met, D01 fallback 2 (D25 to D30)
- Separate get up policy with handoff to s3v9 (D25). Five approaches fell
  short of R08 from the rest poses: PPO at knee 2.4 and 2.85 (D26), fall
  library (D28, 56 percent from 28 to 57 deg, 0 past flat), demo starts
  (D29, 0 percent), open loop search with s3v9 catching (25 / 0 / 0
  percent).
- Root cause (fold_clearance.py): the wheels sit inside the shell rails, so
  the knee fold is blocked from about 2.25 rad. Self collision added (D27).
- Decision D30: change the body shape (layout A, legs outboard). Next: CAD
  skeleton, then retrain.

## 2026-11-28, Step 20 close out: docs and housekeeping
- Killed a search that had run 10.5 h beside the last pipeline (PID 1026).
  Wrote D24 to D30. Interface v0.5 (get up mode, handoff, limp rule).
  Created docs/risk_register.md.
- Checks: selfcol_check.py shows 0 explicit pairs (collision bits only) and
  -1.4 mm deepest overlap; the 4.5 N m clip is kt_leg randomization
  (0.9 to 1.1), not a sim error. Stale files in model/generated/ (OPS-6).
