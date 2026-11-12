"""Loader for the legged robot description (params_legged.yaml), used by the MJCF
builder and the Phase 2 code. The Phase 1 planar model reads params.yaml, which is
derived from this file by src/planar_from_legged.py."""
import math
from pathlib import Path

import yaml

PARAMS_PATH = Path(__file__).resolve().parent.parent / "params_legged.yaml"


def load_params(path=PARAMS_PATH):
    with open(path) as f:
        return yaml.safe_load(f)


def total_mass(p):
    return (
        p["body"]["mass_kg"]
        + p["shell"]["mass_kg"]
        + p["arm"]["mass_kg"]
        + 2 * (p["thigh"]["mass_kg"] + p["shank"]["mass_kg"] + p["wheel"]["mass_kg"])
    )


def hip_height_straight(p):
    return p["thigh"]["length_m"] + p["shank"]["length_m"] + p["wheel"]["radius_m"]


def leg_ik(p, hip_height_m):
    """Hip and knee angles that put the wheel axle directly under the hip at the
    given hip height above the floor. Convention (docs/conventions.md): positive hip
    and knee swing the foot backward, so a crouch uses a negative hip angle and a
    positive knee angle, q_knee = -2 * q_hip. Assumes equal thigh and shank lengths."""
    l1, l2 = p["thigh"]["length_m"], p["shank"]["length_m"]
    if abs(l1 - l2) > 1e-9:
        raise ValueError("leg_ik assumes equal thigh and shank lengths")
    c = (hip_height_m - p["wheel"]["radius_m"]) / (l1 + l2)
    if not 0.0 < c <= 1.0:
        raise ValueError(f"hip height {hip_height_m} m is out of reach")
    q_hip = -math.acos(c)
    q_knee = -2.0 * q_hip
    lo_h, hi_h = p["joints"]["hip"]["range_rad"]
    lo_k, hi_k = p["joints"]["knee"]["range_rad"]
    if not (lo_h <= q_hip <= hi_h and lo_k <= q_knee <= hi_k):
        raise ValueError(f"pose ({q_hip:.3f}, {q_knee:.3f}) rad is outside the joint ranges")
    return q_hip, q_knee


def hip_height_from_angles(p, q_hip, q_knee):
    l1, l2 = p["thigh"]["length_m"], p["shank"]["length_m"]
    return l1 * math.cos(q_hip) + l2 * math.cos(q_hip + q_knee) + p["wheel"]["radius_m"]


def height_range(p):
    """Reachable hip height range with the axle under the hip, under the joint limits."""
    lo_h, _ = p["joints"]["hip"]["range_rad"]
    _, hi_k = p["joints"]["knee"]["range_rad"]
    q_min = max(lo_h, -hi_k / 2.0)  # the most crouched legal pose
    return hip_height_from_angles(p, q_min, -2 * q_min), hip_height_straight(p)