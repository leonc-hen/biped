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
