"""Saves the seed, config, and code version next to each log."""
import json
import subprocess
from datetime import datetime
from pathlib import Path


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def save_run_record(csv_path, seed, config):
    record = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "seed": seed,
        "git_commit": git_commit(),
        "config": config,
    }
    out = Path(csv_path).with_suffix(".json")
    out.write_text(json.dumps(record, indent=2))
    return out