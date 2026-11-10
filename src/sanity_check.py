from params import load_params
import math

p = load_params()
print("M_eff", p.m_eff, "a", p.a, "b", p.b, "J", p.J, "D", p.D)

# Linearized unstable pole with zero input:
m, l, g = p.body_mass_kg, p.axle_to_com_m, p.gravity_ms2
w2 = m * g * l / (p.J - p.b**2 / p.a)
print("open loop pole:", math.sqrt(w2), "rad/s")