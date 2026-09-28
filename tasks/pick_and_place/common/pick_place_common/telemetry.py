"""Optional per-tick CSV telemetry. Off unless MANIPOPT_TELEMETRY_DIR is
set; then each node writes <dir>/<name>.csv, one row per tick, for judging a
live run by per-tick signals rather than log lines. Rows are buffered and
flushed about once a second.
"""
import csv
import os
import time
from pathlib import Path

FLUSH_EVERY_ROWS = 50


class Telemetry:
    def __init__(self, name: str, columns: list[str]):
        out_dir = os.environ.get("MANIPOPT_TELEMETRY_DIR")
        self.enabled = bool(out_dir)
        self._rows = []
        if not self.enabled:
            return
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        self._file = open(Path(out_dir) / f"{name}.csv", "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["wall_t", *columns])

    def row(self, *values):
        if not self.enabled:
            return
        self._rows.append((time.time(), *values))
        if len(self._rows) >= FLUSH_EVERY_ROWS:
            self.flush()

    def flush(self):
        if not self.enabled or not self._rows:
            return
        self._writer.writerows(self._rows)
        self._file.flush()
        self._rows = []
