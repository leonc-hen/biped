import math
import random

from tools.logger import RunLogger
from tools.run_record import save_run_record


def main():
    seed = 0
    random.seed(seed)
    config = {"note": "dummy run", "policy_rate_hz": 100}
    with RunLogger("sim", "dummy") as log:
        for i in range(100):
            t = i / 100
            log.log(
                t_s=t,
                pitch_rad=0.1 * math.sin(2 * math.pi * t),
                battery_V=24.0,
            )
    print("wrote", log.path)
    print("wrote", save_run_record(log.path, seed, config))


if __name__ == "__main__":
    main()
