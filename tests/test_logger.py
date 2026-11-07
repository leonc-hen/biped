import csv

import pytest

from tools.logger import LOG_COLUMNS, RunLogger


def test_header_and_rows(tmp_path):
    with RunLogger("sim", "test", log_root=tmp_path) as log:
        log.log(t_s=0.0, pitch_rad=0.1)
        log.log(t_s=0.01, pitch_rad=0.09)
    with open(log.path) as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == LOG_COLUMNS
    assert len(rows) == 2
    assert float(rows[0]["pitch_rad"]) == 0.1


def test_unknown_column_rejected(tmp_path):
    with RunLogger("sim", "test", log_root=tmp_path) as log:
        with pytest.raises(KeyError):
            log.log(pitch_deg=5)