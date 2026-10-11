"""Docks taught at commissioning: the robot parked at each station by hand (a technician
with a joystick), its own localized pose saved in the map frame, next to the map
(<map>_docks.yaml, a line "name: [x, y, yaw]" per dock). Framework-free.

See docs/implementation_notes.md (Docking facing the pick table).
"""
from pathlib import Path


def path(map_path):
    """The docks file of a map (its path without the suffix)."""
    map_path = Path(map_path)
    return map_path.with_name(map_path.name + "_docks.yaml")


def save(map_path, docks, note=""):
    lines = ["# Docks taught at commissioning: the robot's own pose (map frame) parked at each station."]
    if note:
        lines.append(f"# {note}")
    lines += [f"{name}: [{x:.5f}, {y:.5f}, {yaw:.6f}]" for name, (x, y, yaw) in docks.items()]
    path(map_path).write_text("\n".join(lines) + "\n")


def load(map_path):
    """{name: (x, y, yaw)}; FileNotFoundError if the map's docks were never taught."""
    p = path(map_path)
    if not p.exists():
        raise FileNotFoundError(f"{p}: no docks taught for this map (scripts/dev/nav_sim.py --teach)")
    docks = {}
    for line in p.read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            name, values = line.split(":", 1)
            docks[name.strip()] = tuple(float(v) for v in values.strip(" []\n").split(","))
    return docks
