"""The suction cup alone (suction.py) on a kinematic wrist: hold, slip and peel against the
analytic limits (docs/implementation_notes.md#suctionpy)."""
import mujoco
import numpy as np
import pytest

from pick_place_common import suction

G = 9.81
DT = 0.002
BENCH = """
<mujoco>
  <option timestep="0.002" integrator="implicitfast" impratio="10"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="carriage">
      <joint name="hx" type="slide" axis="1 0 0"/>
      <joint name="hy" type="slide" axis="0 1 0"/>
      <joint name="hz" type="slide" axis="0 0 1"/>
      <inertial mass="1000" pos="0 0 0" diaginertia="10 10 10"/>
      <body name="attachment" quat="0 1 0 0">
        <geom name="tool" type="cylinder" pos="0 0 0.0485" size="0.03 0.0485" mass="0" group="3"/>
      </body>
    </body>
    <body name="box" pos="{x} {y} {hz}">
      <freejoint/>
      <inertial mass="{mass}" pos="0 0 0" diaginertia="{ix} {iy} {iz}"/>
      <geom type="box" size="{hx} {hy} {hz}" condim="4" friction="0.6 0.01 0.001"/>
    </body>
  </worldbody>
</mujoco>
"""


class Bench:
    """A box on the floor under the cup; the wrist is moved kinematically (its joints set
    each step, so contacts see its velocity: friction acts)."""

    def __init__(self, mass=2.0, half=(0.06, 0.06, 0.06), offset=(0.0, 0.0)):
        hx, hy, hz = half
        m = mass / 3.0
        spec = mujoco.MjSpec.from_string(BENCH.format(
            x=offset[0], y=offset[1], hx=hx, hy=hy, hz=hz, mass=mass,
            ix=m * (hy * hy + hz * hz), iy=m * (hx * hx + hz * hz), iz=m * (hx * hx + hy * hy)))
        suction.add_cup(spec)
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
        self.cup = suction.SuctionCup(self.m)
        self.box = self.m.body("box").id
        self.hz = hz
        self.pos = np.array([0.0, 0.0, 2 * hz + suction.LIP_END_M + 0.01])
        self.vel = np.zeros(3)
        self._place()
        mujoco.mj_forward(self.m, self.d)

    def _place(self):
        self.d.qpos[:3] = self.pos
        self.d.qvel[:3] = self.vel

    def run(self, seconds, accel=(0.0, 0.0, 0.0)):
        for _ in range(round(seconds / DT)):
            self.vel = self.vel + np.asarray(accel) * DT
            self.pos = self.pos + self.vel * DT
            self._place()
            self.cup.step(self.m, self.d, DT)
            mujoco.mj_step(self.m, self.d)

    def shake(self, a, ramp=0.15, hold=0.3):
        """Sideways: acceleration ramped up to a (jerk-limited, as the arm's and the base's are),
        held, ramped back to zero (it coasts on)."""
        n = round(ramp / DT)
        for k in range(n):
            self.run(DT, accel=(a * (k + 1) / n, 0.0, 0.0))
        self.run(hold, accel=(a, 0.0, 0.0))
        for k in range(n):
            self.run(DT, accel=(a * (n - k - 1) / n, 0.0, 0.0))

    def move(self, dz, speed=0.03, accel=2.0):
        """Straight up or down, accelerating and braking at accel (an arm's, not a step)."""
        t_acc = speed / accel
        a = np.array([0.0, 0.0, np.sign(dz) * accel])
        self.run(t_acc, accel=a)
        self.run(abs(dz) / speed - t_acc)
        self.run(t_acc, accel=-a)
        self.vel = np.zeros(3)

    def lip_force(self):
        f, total = np.zeros(6), 0.0
        for i, c in enumerate(self.d.contact[:self.d.ncon]):
            if c.geom1 in self.cup.lips or c.geom2 in self.cup.lips:
                mujoco.mj_contactForce(self.m, self.d, i, f)
                total += f[0]
        return total

    def grip(self, contact_n=3.0, wait=0.5):
        """Down at the skill's touch speed until the lip carries contact_n (the skill's
        guarded descent), then the vacuum on."""
        self.vel = np.array([0.0, 0.0, -0.03])
        while self.lip_force() < contact_n:
            self.run(DT)
        self.vel = np.zeros(3)
        self.cup.valve = True
        self.run(wait)

    def box_in_hand(self):
        """The box's centre in the wrist's frame (m) and its tilt (deg)."""
        rot = self.d.xmat[self.m.body("attachment").id].reshape(3, 3)
        p = rot.T @ (self.d.xpos[self.box] - self.d.xpos[self.m.body("attachment").id])
        tilt = np.degrees(np.arccos(np.clip(rot[:, 2] @ -self.d.xmat[self.box].reshape(3, 3)[:, 2], -1, 1)))
        return p, tilt

    def held(self):
        return self.d.xpos[self.box][2] > self.hz + 0.02


def test_vacuum_reached():
    assert suction.vacuum_reached(1.0) == pytest.approx(1.0)
    assert 0.3 < suction.vacuum_reached(7 / 8) < 0.5
    assert suction.vacuum_reached(0.0) < 0.02


def test_hold_force():
    assert suction.HOLD_FORCE_MAX_N == pytest.approx(75.4, abs=0.1)


def test_hangs_still():
    b = Bench(mass=2.0)
    b.grip()
    assert b.cup.sealed == 1.0
    assert b.cup.vacuum_kpa > 0.9 * suction.VACUUM_MAX_KPA
    b.move(0.1, speed=0.1)
    b.run(1.0)
    p0, tilt = b.box_in_hand()
    assert b.held() and tilt < 1.0
    b.run(1.0)
    assert np.linalg.norm(b.box_in_hand()[0] - p0) < 2e-4  # no creep


def test_too_heavy_drops():
    b = Bench(mass=suction.HOLD_FORCE_MAX_N / G * 1.15)
    b.grip()
    b.move(0.1, speed=0.1)
    b.run(1.0)
    assert not b.held()


def test_released_with_blowoff():
    b = Bench(mass=0.2)
    b.grip()
    b.move(0.05, speed=0.1)
    b.cup.valve = False
    b.run(0.1)
    assert b.cup.vacuum_kpa < 2.0
    b.run(0.5)
    assert not b.held()


def test_no_seal_on_edge():
    # The cup's centre on the box's edge: 3 of its 8 lip points open, little vacuum.
    b = Bench(mass=0.2, half=(0.06, 0.06, 0.06), offset=(0.06, 0.0))
    b.grip()
    assert b.cup.sealed <= 5 / 8
    assert b.cup.vacuum_kpa < 0.1 * suction.VACUUM_MAX_KPA


def _peel_accel(mass, hz, offset=0.0):
    """Analytic: the box tips about the far side of the lip ring when the inertial and
    offset moments exceed the net press's: m (a hz + g e) > (F - m g) r."""
    f = suction.HOLD_FORCE_MAX_N
    return ((f - mass * G) * suction.LIP_RING_M - mass * G * offset) / (mass * hz)


@pytest.mark.parametrize("offset", [0.0, 0.01])
def test_peel_against_analytic(offset):
    mass, hz = 2.0, 0.06
    a_peel = _peel_accel(mass, hz, offset)
    for a, held in ((0.7 * a_peel, True), (1.4 * a_peel, False)):
        b = Bench(mass=mass, half=(0.07, 0.07, hz), offset=(offset, 0.0))
        b.grip()
        b.move(0.15, speed=0.1)
        b.run(0.5)
        b.shake(a)
        b.run(0.5)
        assert b.held() == held, f"a {a:.1f} m/s^2 (peel at {a_peel:.1f})"


def test_hardest_stop_holds():
    # The base's protective stop (4.5 m/s^2) with the heaviest box, centred within 5 mm.
    b = Bench(mass=2.0, half=(0.075, 0.06, 0.06), offset=(0.005, 0.0))
    b.grip()
    b.move(0.15, speed=0.1)
    b.run(0.5)
    b.run(1.5, accel=(1.0, 0.0, 0.0))
    b.run(0.3)
    p0, _ = b.box_in_hand()
    b.run(1.5 / 4.5, accel=(-4.5, 0.0, 0.0))
    b.vel = np.zeros(3)
    b.run(0.5)
    p, tilt = b.box_in_hand()
    assert b.held() and np.linalg.norm(p - p0) < 0.002 and tilt < 2.0


@pytest.mark.parametrize("deg, seals", [(2.0, True), (6.0, False)])
def test_tilt(deg, seals):
    # The lip conforms to a small tilt; past the tool's face's own tilt it leaks.
    b = Bench(mass=1.0)
    q, out = np.zeros(4), np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, np.array([1.0, 0.0, 0.0]), np.radians(deg))
    att = b.m.body("attachment").id
    mujoco.mju_mulQuat(out, q, b.m.body_quat[att].copy())
    b.m.body_quat[att] = out
    b.grip()
    assert (b.cup.vacuum_kpa > 0.9 * suction.VACUUM_MAX_KPA) == seals
