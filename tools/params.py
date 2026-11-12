"""Loader for the Phase 0 robot description (phase0_params.yaml), used by the MJCF builder."""
from pathlib import Path

import yaml

PARAMS_PATH = Path(__file__).resolve().parent.parent / "phase0_params.yaml"


def load_params(path=PARAMS_PATH):
    with open(path) as f:
        return yaml.safe_load(f)


def total_mass(p):
    return p["body"]["mass_kg"] + 2 * (
        p["thigh"]["mass_kg"] + p["shank"]["mass_kg"] + p["wheel"]["mass_kg"]
    )


def hip_height_straight(p):
    return p["thigh"]["length_m"] + p["shank"]["length_m"] + p["wheel"]["radius_m"]
