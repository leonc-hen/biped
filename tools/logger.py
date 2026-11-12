"""CSV run logger. Same columns for simulation and real hardware."""
import csv
from datetime import datetime
from pathlib import Path

LOG_COLUMNS = [
    "t_s", "pitch_rad", "pitch_rate_rad_s", "yaw_rad", "yaw_rate_rad_s",
    "wheel_pos_l_rad", "wheel_pos_r_rad", "wheel_vel_l_rad_s", "wheel_vel_r_rad_s",
    "hip_pos_l_rad", "hip_pos_r_rad", "knee_pos_l_rad", "knee_pos_r_rad",
    "wheel_torque_l_Nm", "wheel_torque_r_Nm", "hip_torque_l_Nm", "hip_torque_r_Nm",
    "knee_torque_l_Nm", "knee_torque_r_Nm", "battery_V",
]


class RunLogger:
    def __init__(self, kind, name, log_root="logs"):
        if kind not in ("sim", "real"):
            raise ValueError("kind must be 'sim' or 'real'")
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        folder = Path(log_root) / kind
        folder.mkdir(parents=True, exist_ok=True)
        self.path = folder / f"{stamp}_{name}.csv"
        self._file = open(self.path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=LOG_COLUMNS)
        self._writer.writeheader()

    def log(self, **values):
        unknown = set(values) - set(LOG_COLUMNS)
        if unknown:
            raise KeyError(f"Unknown log columns: {sorted(unknown)}")
        self._writer.writerow({c: values.get(c, "") for c in LOG_COLUMNS})

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
