"""The scenario table of mobile jobs (job_run.sh runs): from each run's monitors (the
totals they log when the run ends) and its telemetry (job_time.py): boxes placed, the
job's time and the time per box, the distance driven, the closest anyone came to the
chassis and to the arm (and a held box), protective-field intrusions and contacts,
the arm's holds and their causes, the localization error and the drives' motion.
usage: python job_results.py <name>=<run_dir> [...] [--md out.md]
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import job_time  # noqa: E402

ROWS = [
    ("boxes placed", "boxes"),
    ("job time (min)", "minutes"),
    ("time per box (s)", "per_box"),
    ("distance driven (m)", "distance"),
    ("drives docked / parked", "arrived"),
    ("docking error max: along / lateral (mm), yaw (deg)", "dock"),
    ("closest person to the chassis while it moved (mm)", "base_moving"),
    ("...also people walking up to it (mm)", "base_any"),
    ("protective-field intrusions, contacts", "intrusions"),
    ("closest person to the arm while it worked (mm)", "arm_work"),
    ("...to the arm or box while the base closed on them (mm)", "arm_closing"),
    ("...at any time while not docked (mm)", "arm_any"),
    ("...while the base moved with the arm out (mm)", "arm_overlap"),
    ("arm holds: a person near / perception / unexplained", "holds"),
    ("people tracked within 5 m (%)", "people"),
    ("localization error p50 / p95 / max (mm)", "loc"),
    ("wiggles (largest swing deg), oscillations, turn-backs", "motion"),
    ("both moving / base waiting for the arm (s)", "overlap"),
]


def _num(pattern, text, cast=float, default=None):
    m = re.search(pattern, text)
    return cast(m.group(1)) if m else default


def dist(v):
    """A distance in mm for the table: '-' for nobody measured, '> N' at the measure's
    cutoff (2 m before 2026-10-03, 3 m since)."""
    if v in (None, "inf", "-"):
        return "-"
    return f"> {v}" if float(v) in (2000.0, 3000.0) else v


def parse(run):
    log = (run / "launch.log").read_text()
    r = {}
    nav = next((ln for ln in log.splitlines() if "navigation over the run" in ln), "")
    arm = next((ln for ln in log.splitlines() if "arm and people over the run" in ln), "")
    loc = next((ln for ln in log.splitlines() if "localization over the run" in ln), "")
    ppl = next((ln for ln in log.splitlines() if "people_monitor" in ln and "over the run" in ln), "")
    r["boxes"] = _num(r"job done: (\d+) boxes", log, int, _num(r"boxes placed (\d+)", arm, int))
    t = job_time.summary(run)
    r["minutes"] = f"{t['total'] / 60:.1f}"
    r["per_box"] = f"{t['total'] / r['boxes']:.0f}" if r["boxes"] else "-"
    r["overlap"] = f"{t['overlap']:.0f} / {t['gated']:.1f}"
    legs = [float(x) for x in re.findall(r"drive +\d+ to \w+: \w+ +[\d.]+ s +([\d.]+) m", log)]
    r["distance"] = f"{sum(legs):.0f}"
    r["arrived"] = (re.search(r"arrived (\d+/\d+)", nav) or [None, "-"])[1]
    r["dock"] = "{} / {}, {}".format(_num(r"\|along\| max (\d+) mm", nav, int),
                                     _num(r"\|lateral\| max (\d+) mm", nav, int), _num(r"\|yaw\| max ([\d.]+) deg", nav))
    r["base_moving"] = dist(_num(r"closest person while moving (\w+) mm", nav, str))
    r["base_any"] = dist(_num(r"walking up: (\w+) mm", nav, str))
    r["intrusions"] = f"{_num(r'field intrusions (\d+)', nav, int)}, {_num(r'contacts (\d+)', nav, int)}"
    r["arm_work"] = dist(_num(r"while the arm worked (\w+) mm", arm, str))
    r["arm_closing"] = dist(_num(r"closing on them (\w+) mm", arm, str))
    r["arm_any"] = dist(_num(r", any (\w+) mm", arm, str))
    r["arm_overlap"] = dist(_num(r"with the arm out (\w+) mm", arm, str, "-"))
    r["holds"] = "{} / {} / {}".format(_num(r"within [\d.]+ m (\d+)", arm, int), _num(r"perception (\d+)", arm, int),
                                       _num(r"unexplained (\d+)", arm, int))
    r["people"] = _num(r"\(([\d.]+)%\)", ppl, str, "-")
    m = re.search(r"p50/p95/max (\d+)/(\d+)/(\d+) mm", loc)
    r["loc"] = " / ".join(m.groups()) if m else "-"
    r["motion"] = "{} ({}), {}, {}".format(_num(r"wiggles (\d+)", nav, int), _num(r"swing <= (\d+) deg", nav, int),
                                           _num(r"oscillations (\d+)", nav, int), _num(r"turn-backs (\d+)", nav, int))
    return r


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--md")]
    out = sys.argv[sys.argv.index("--md") + 1] if "--md" in sys.argv else None
    if out:
        args.remove(out)
    runs = [a.split("=", 1) for a in args]
    results = [(name, parse(Path(path))) for name, path in runs]
    lines = ["| | " + " | ".join(n for n, _ in results) + " |", "|---|" + "---|" * len(results)]
    for label, key in ROWS:
        lines.append(f"| {label} | " + " | ".join(str(r.get(key, "-")) for _, r in results) + " |")
    text = "\n".join(lines)
    print(text)
    if out:
        Path(out).write_text(text + "\n")


if __name__ == "__main__":
    main()
