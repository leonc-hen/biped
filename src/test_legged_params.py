from tools.params import (
    load_params,
    total_mass,
    leg_ik,
    hip_height_from_angles,
    height_range,
)


def test_total_mass_includes_shell_and_arm():
    p = load_params()
    assert abs(total_mass(p) - 1.72) < 1e-9  # update together with the mass budget


def test_ik_round_trip():
    p = load_params()
    for h in [0.15, 0.19, 0.24]:
        q1, q2 = leg_ik(p, h)
        assert abs(hip_height_from_angles(p, q1, q2) - h) < 1e-9
        assert abs(q2 + 2 * q1) < 1e-12


def test_nominal_pose_inside_joint_ranges():
    p = load_params()
    leg_ik(p, p["pose"]["nominal_hip_height_m"])


def test_height_range_reaches_the_requirement_floor():
    p = load_params()
    lo, hi = height_range(p)
    assert lo <= 0.15
    assert hi >= 0.24