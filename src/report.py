"""One command summary for a failed gate: the newest eval JSON for a run, its key
numbers, and the posture diagnostic. Run from the repo root:
python src/report.py --run s2r5_a2
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KEYS = ("standing_tilt_mean_deg", "standing_tilt_std_deg", "standing_wobble_hz", "standing_mean_v_m_s",
        "standing_wheel_stagger_mm", "max_recovered_tilt_rad", "max_recovered_tilt_back_rad",
        "max_push_fwd_m_s", "max_push_back_m_s", "max_push_lat_m_s", "max_push_lat_neg_m_s",
        "track_0.3_mean_v", "track_1.0_mean_v", "track_1.0_settle_s",
        "height_0.150_signed_mm", "height_0.250_signed_mm", "height_0.300_signed_mm",
        "push_rand_0.4_success", "dr_push_0.4_success", "dr_push_0.4_success_vs_prepush",
        "dr_nopush_drift_deg", "dr_push_fail_reasons", "within_spec_fall",
        "train_dist_fall", "first_failures",
        "jump_nominal_success_10cm", "jump_dr_success_10cm", "jump_dr_fell", "jump_dr_com_rise_cm",
        "jump_dr_wheel_clearance_cm_median", "jump_dr_landing_leg_torque_Nm",
        "jump_dr_fall_time_after_cmd_s", "jump_dr_max_tilt_in_window_deg",
        "jump_nominal_com_rise_cm", "jump_nominal_fell", "jump_nominal_wheel_clearance_cm_median",
        "jump_nominal_landing_leg_torque_Nm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    files = sorted((ROOT / "results").glob(f"evalb_*_{args.run}*.json"),
                   key=lambda f: f.stat().st_mtime)
    if not files:
        print(f"no eval results for {args.run}")
        return 1
    print(f"== {files[-1].name}")
    r = json.load(open(files[-1]))
    for k in KEYS:
        if k in r:
            print(f"{k}: {r[k]}")
    if "gate" in r:
        print("\n== gate (this evaluation)")
        for title, checks in r["gate"].items():
            print(title)
            for name, passed in checks.items():
                print(f"  {'PASS' if passed else 'FAIL'}  {name}")
        print(f"\nGATE {'PASS' if r['gate_pass'] else 'FAIL'}")
    print("\n== posture")
    subprocess.run([sys.executable, "src/diag_saturation.py", "--run", args.run], cwd=ROOT)
    return 0


if __name__ == "__main__":
    sys.exit(main())