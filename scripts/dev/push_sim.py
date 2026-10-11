"""Offline push of a released box by the tool, the scene's contact settings: the box
(0.2 kg, class "box"), the tray floor and one side wall, the tool cylinder (default
friction) on a compliant mount (stiffness per axis as the live arm showed during
pushes: ~1200 N/m along the push, ~750 sideways, stiff in height). The tool pushes along
-x at 0.05 m/s from mid-height or lower, leaning across the push. Per case: how far the
box moved, the peak push force, the box's largest tilt and its yaw, and whether it
ended against the side wall.
usage: python push_sim.py [push_mm=46] [side_gap_mm=6.5] [--rise mm]
"""
import sys

import mujoco
import numpy as np

ARGS = [a for i, a in enumerate(sys.argv[1:], 1) if not a.startswith("--") and sys.argv[i - 1] != "--rise"]
PUSH = float(ARGS[0]) / 1e3 if ARGS else 0.046
SIDE_GAP = float(ARGS[1]) / 1e3 if len(ARGS) > 1 else 0.0065
HX, HY, HZ = 0.045, 0.060, 0.0375  # cbox_3 turned: 90 x 120 x 75 mm
R, HALF_LEN, TIP_ABOVE_TCP = 0.03, 0.0485, 0.003
K = (1200.0, 750.0, 10000.0)  # N/m, x (push), y (sideways), z
K_PITCH = 150.0  # Nm/rad, the tool leaning back about the flange (~2 deg at 40 N live)
FLANGE_ABOVE_TCP = 0.15
RISE = 0.0  # m the TCP rises over the push (--rise)
SPEED = 0.05
FORCE_MAX = 40.0


def model(tilt):
    axis = np.array([0.0, -np.sin(tilt), -np.cos(tilt)])  # flange -> tip
    c = -axis * (TIP_ABOVE_TCP + HALF_LEN)  # cylinder centre from the TCP
    q = np.zeros(4)
    mujoco.mju_quatZ2Vec(q, -axis)
    acts = "".join(f'<position joint="t{a}" kp="{k}" kv="{2 * np.sqrt(k * 1.0):.1f}"/>' for a, k in zip("xyz", K))
    xml = f"""<mujoco><option timestep="0.002" integrator="implicitfast" impratio="10"/>
<default><default class="box"><geom condim="4" friction="0.6 0.01 0.001"/></default></default>
<worldbody>
  <geom name="floor" type="plane" size="1 1 0.1"/>
  <geom name="wall" type="box" pos="0 {-(HY + SIDE_GAP + 0.005)} 0.11" size="0.3 0.005 0.11"/>
  <body name="box" pos="0 0 {HZ}"><freejoint/>
    <geom class="box" type="box" size="{HX} {HY} {HZ}" mass="0.2"/></body>
  <body name="tool" pos="0 0 0">
    <joint name="tx" type="slide" axis="1 0 0"/><joint name="ty" type="slide" axis="0 1 0"/>
    <joint name="tz" type="slide" axis="0 0 1"/>
    <inertial pos="0 0 0" mass="1.0" diaginertia="0.01 0.01 0.01"/>
    <body name="lean" pos="{-axis[0] * FLANGE_ABOVE_TCP} {-axis[1] * FLANGE_ABOVE_TCP} {-axis[2] * FLANGE_ABOVE_TCP}">
      <joint name="pitch" type="hinge" axis="0 1 0" stiffness="{K_PITCH}" damping="2"/>
      <inertial pos="0 0 0" mass="0.1" diaginertia="0.001 0.001 0.001"/>
      <geom name="tool" type="cylinder" size="{R} {HALF_LEN}"
        pos="{c[0] + axis[0] * FLANGE_ABOVE_TCP} {c[1] + axis[1] * FLANGE_ABOVE_TCP} {c[2] + axis[2] * FLANGE_ABOVE_TCP}"
        quat="{q[0]} {q[1]} {q[2]} {q[3]}"/>
    </body>
  </body>
</worldbody><actuator>{acts}</actuator></mujoco>"""
    return mujoco.MjModel.from_xml_string(xml)


def run(tilt_deg, height):
    m = model(np.radians(tilt_deg))
    d = mujoco.MjData(m)
    start = np.array([HX + R + 0.01, 0.0, height])
    d.qpos[7:10] = start  # the tool joints after the box's free joint
    d.ctrl[:] = start
    mujoco.mj_forward(m, d)
    for _ in range(250):  # settle
        mujoco.mj_step(m, d)
    box = m.body("box").id
    p0 = d.xpos[box].copy()
    peak, tilt_max, travel = 0.0, 0.0, PUSH + R + 0.01 + 0.02
    n = int(travel / SPEED / m.opt.timestep)
    for i in range(n):
        d.ctrl[0] = start[0] - SPEED * m.opt.timestep * (i + 1)
        d.ctrl[2] = start[2] + RISE * (i + 1) / n
        mujoco.mj_step(m, d)
        f = -K[0] * (d.ctrl[0] - d.qpos[7])  # the mount's spring: the push force
        peak = max(peak, f)
        r = d.xmat[box].reshape(3, 3)
        tilt_max = max(tilt_max, np.degrees(np.arccos(np.clip(r[2, 2], -1, 1))))
        if f > FORCE_MAX:
            break
    r = d.xmat[box].reshape(3, 3)
    moved = p0[0] - d.xpos[box][0]
    side = d.xpos[box][1] - HY - (-(HY + SIDE_GAP))
    return dict(moved=moved, peak=peak, tilt=tilt_max, yaw=np.degrees(np.arctan2(r[1, 0], r[0, 0])),
                side=side, dy_tool=d.qpos[8], stopped=peak > FORCE_MAX)


if __name__ == "__main__":
    print(f"push {1e3 * PUSH:.0f} mm along -x, box {2e3 * HX:.0f} x {2e3 * HY:.0f} x {2e3 * HZ:.0f} mm, "
          f"{1e3 * SIDE_GAP:.1f} mm from the side wall")
    if "--rise" in sys.argv:
        RISE = float(sys.argv[sys.argv.index("--rise") + 1]) / 1e3
    for tilt in (0.0, 10.0, 25.0):
        for h in (HZ, 0.025, 0.015):
            lowest = R * np.sin(np.radians(tilt)) + TIP_ABOVE_TCP
            if h - lowest < 0.004:
                continue
            r = run(tilt, h)
            print(f"tilt {tilt:4.1f} deg, TCP {1e3 * h:4.1f} mm up: moved {1e3 * r['moved']:5.1f} mm, peak "
                  f"{r['peak']:5.1f} N{' (stopped)' if r['stopped'] else ''}, box tilt max {r['tilt']:4.1f} deg, "
                  f"yaw {r['yaw']:+5.1f}, side gap {1e3 * r['side']:+5.1f} mm, tool sideways {1e3 * r['dy_tool']:+5.1f} mm")
