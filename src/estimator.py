"""Complementary filter tilt and velocity estimator, and startup gyro bias calibration."""
import numpy as np

class ComplementaryEstimator:
    """Tilt and velocity estimator.
    tau_c       : filter time constant in seconds (accelerometer trust)
    compensate  : subtract the wheel acceleration (from encoders) before computing tilt
    """
    def __init__(self, p, Ts, tau_c, compensate, gyro_bias_est=0.0, fc_acc=20.0):
        self.r = p.wheel_radius_m
        self.Ts = Ts
        self.tau_c = tau_c
        self.compensate = compensate
        self.bias = gyro_bias_est
        self.alpha = Ts / (Ts + 1.0 / (2 * np.pi * fc_acc))
        self.theta = 0.0
        self.v_prev = 0.0
        self.a_est = 0.0
        self.first = True

    def update(self, gyro, fx, fz, w_rel):
        Ts = self.Ts
        thd = gyro - self.bias
        v_enc = self.r * (w_rel + thd)           # forward speed from encoders plus gyro
        if self.first:
            self.v_prev, self.first = v_enc, False
        a_raw = (v_enc - self.v_prev) / Ts
        self.a_est += self.alpha * (a_raw - self.a_est)
        self.v_prev = v_enc
        if self.compensate:
            fx = fx - self.a_est * np.cos(self.theta)
            fz = fz - self.a_est * np.sin(self.theta)
        th_acc = np.arctan2(-fx, fz)
        self.theta += Ts * (thd + (th_acc - self.theta) / self.tau_c)
        return self.theta, thd, v_enc

def calibrate_gyro_bias(p, rng, n=500):
    """Average the gyro while the robot is held still. Returns the bias estimate."""
    samples = p.gyro_bias_rad_s + p.gyro_noise_rad_s * rng.standard_normal(n)
    return float(np.mean(samples))
