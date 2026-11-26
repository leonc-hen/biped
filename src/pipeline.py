"""Overnight training pipeline: preflight checks, then curriculum stages back to
back, each followed by its gate. Stops at the first failure, so a bad stage never
wastes the next. Log in results/pipeline.txt. Run from the repo root:
python src/pipeline.py [--start s3v9]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
PREFLIGHT = [["-m", "pytest", "tests/", "-q"], ["src/step16_checks.py"], ["src/lqr_on_legged.py"]]
STAGES = [
    dict(name="s3v9", gate="exit", eval=[],
         train=["--env", "s3", "--dr_scale", "1.0", "--steps", "150000000", "--init_from", "s3v8_b"]),
]

def run(args, log):
    log.write(f"\n$ python {' '.join(args)}\n")
    log.flush()
    return subprocess.run([PY, *args], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode


def s1_gate(name):
    r = json.load(open(ROOT / "results" / f"evalb_s1_{name}_Aonly.json"))
    return {
        "tracks 1.0 m/s": r.get("track_1.0_settle_s") is not None,
        "creep under 0.03 m/s": abs(r["standing_mean_v_m_s"]) < 0.03,
        "standing wobble under 0.1 deg": r["standing_tilt_std_deg"] < 0.1,
        "tilt 0.35 rad both ways":
            min(r["max_recovered_tilt_rad"], r["max_recovered_tilt_back_rad"]) >= 0.35,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=None, help="skip stages before this one")
    args = ap.parse_args()
    names = [st["name"] for st in STAGES]
    if args.start is not None and args.start not in names:
        print(f"unknown stage '{args.start}'; stages are: {', '.join(names)}")
        return 1
    with open(ROOT / "results" / "pipeline.txt", "a") as log:
        for cmd in PREFLIGHT:
            if run(cmd, log) != 0:
                print(f"PREFLIGHT FAILED: {' '.join(cmd)} (see results/pipeline.txt)")
                return 1
        started = args.start is None
        ran = 0
        for st in STAGES:
            started = started or st["name"] == args.start
            if not started:
                continue
            print(f"training {st['name']}", flush=True)
            ran += 1
            if run(["src/train_s1.py", "--name", st["name"], *st["train"]], log) != 0:
                print(f"TRAINING CRASHED: {st['name']}")
                return 1
            code = run(["src/eval_batch.py", "--run", st["name"], *st["eval"]], log)
            if st["gate"] == "s1":
                checks = s1_gate(st["name"])
                passed = all(checks.values())
                for k, v in checks.items():
                    print(f"  {'PASS' if v else 'FAIL'}  {k}")
            else:
                passed = code == 0
            print(f"{st['name']}: {'GATE PASS' if passed else 'GATE FAIL, stopping'}", flush=True)
            if not passed:
                return 1
    if ran == 0:
        print("NOTHING RAN")
        return 1
    print(f"ALL {ran} STAGES PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())