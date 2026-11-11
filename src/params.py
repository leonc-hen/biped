import yaml
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Params:
    wheel_mass_total_kg: float
    wheel_radius_m: float
    body_mass_kg: float
    axle_to_com_m: float
    body_pitch_inertia_kgm2: float
    motor_kt_nm_per_a: float
    winding_resistance_ohm: float
    battery_voltage_v: float
    current_limit_a: float
    track_width_m: float
    gravity_ms2: float
    wheel_inertia_total_kgm2: float
    motor_ke_vs_per_rad: float
    viscous_friction_nm_s_per_rad: float
    yaw_body_inertia_kgm2: float
    motor_count: int
    imu_height_m: float
    gyro_noise_rad_s: float
    gyro_bias_rad_s: float
    accel_noise_ms2: float
    encoder_noise_rad_s: float

    # Derived quantities, used by the model in step 2
    @property
    def m_eff(self):
        return self.wheel_mass_total_kg + self.wheel_inertia_total_kgm2 / self.wheel_radius_m**2

    @property
    def a(self):
        return self.m_eff + self.body_mass_kg

    @property
    def b(self):
        return self.body_mass_kg * self.axle_to_com_m

    @property
    def J(self):
        return self.body_pitch_inertia_kgm2 + self.body_mass_kg * self.axle_to_com_m**2

    @property
    def D(self):
        return self.a * self.J - self.b**2

def load_params(path="params.yaml") -> Params:
    with open(Path(path)) as f:
        raw = yaml.safe_load(f)
    p = Params(**raw)          # fails loudly if a key is missing or extra
    validate(p)
    return p

def validate(p: Params):
    for name, value in p.__dict__.items():
        assert value >= 0, f"{name} must not be negative"
    assert p.wheel_radius_m > 0 and p.axle_to_com_m > 0 and p.body_mass_kg > 0
    assert p.D > 0, "D = a*J - b^2 must be positive for a physical robot"