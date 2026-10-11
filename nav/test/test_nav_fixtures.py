"""The navigation's offline fixtures (the RL plan's layer 1, step 0): nav_sim on two fixed
runs must reproduce the committed trajectory hash and drive scores exactly, so a change
meant to leave `rules` alone (a refactor, a new mode switched off) is checked to. Exact
on the machine and environment that wrote them; another BLAS or MuJoCo build differs in
the last bits; --fast must drive the same. About two minutes. After an intended change of behaviour:
`python nav/test/test_nav_fixtures.py --write`."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
COMMON = ["--arm", "carry", "--box", "--fusion", "ekf", "--seed", "0", "--drives", "4"]
RUNS = {"quiet": ["--crowd", "0"], "job4": ["--people", "job", "--crowd", "4"]}
KEYS = ("arrived", "drive_scores", "trajectory_sha256")


def run(name, out, extra=()):
    cmd = [sys.executable, str(REPO / "scripts/dev/nav_sim.py"), *RUNS[name], *COMMON, *extra, "--fixture", str(out)]
    p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-2000:]
    return json.loads(Path(out).read_text())


@pytest.mark.parametrize("name", sorted(RUNS))
def test_fixture(name, tmp_path):
    want = json.loads((FIXTURES / f"nav_sim_{name}.json").read_text())
    got = run(name, tmp_path / "out.json")
    for i, (w, g) in enumerate(zip(want["drive_scores"], got["drive_scores"])):
        assert g == w, f"drive {i + 1}: " + ", ".join(f"{k} {w[k]} -> {g.get(k)}" for k in w if g.get(k) != w[k])
    assert {k: got[k] for k in KEYS} == {k: want[k] for k in KEYS}


def test_fast_drives_the_same(tmp_path):
    """--fast leaves out the estimator that does not drive; the drive is unchanged."""
    want = json.loads((FIXTURES / "nav_sim_job4.json").read_text())
    got = run("job4", tmp_path / "out.json", ["--fast"])
    assert {k: got[k] for k in KEYS} == {k: want[k] for k in KEYS}


if __name__ == "__main__":
    if "--write" in sys.argv:
        FIXTURES.mkdir(exist_ok=True)
        for name in RUNS:
            run(name, FIXTURES / f"nav_sim_{name}.json")
            print(f"wrote {FIXTURES / f'nav_sim_{name}.json'}")
