# Design notes

Rationale kept here in detail; README links to this for the "why" behind
each non-obvious choice.

## Why shooting + adjoint (acados) instead of direct collocation

Prior background here is direct collocation via CasADi/IPOPT, and a
generalized-alpha multibody integrator (ODIN) for constrained,
index-3-DAE biomechanical systems, where the specialized integrator earns
its complexity because the dynamics genuinely are a DAE.

This manipulator is an open kinematic chain: no closed loops, and the OCP
never needs to solve for a contact/constraint force explicitly. So its
dynamics are a plain second-order ODE,

    q_ddot = M(q)^-1 (tau - h(q, q_dot))

which removes any reason to bring in generalized-alpha's DAE handling.
acados' built-in IRK (Gauss-Legendre) shooting integrator, plus its
automatic forward/adjoint sensitivity propagation through both the
objective and the dynamics, replaces the entire hand-derived
adjoint-equation step that a from-scratch ODIN-style pipeline would need.

acados' SQP_RTI, not IPOPT, is the solver: RTI does one QP solve per call
with a fixed computational footprint, which is what makes worst-case solve
time predictable enough to run inside a hard real-time control loop, versus
IPOPT's iteration count varying with how far the current iterate is from a
solution.

## Why not port generalized-alpha to acados

Not needed: the manipulator problem is already an ODE, so there is no DAE
for generalized-alpha to help with. Porting it would add integrator
complexity that buys nothing here.

## Why not reuse ODIN's source directly

License terms are unclear, it is a general-purpose multibody code that
resists cleanly extracting just the relevant piece, and the
documentation/build burden would consume days this project doesn't have.
The ideas (single-shooting + adjoint-based OCP) carry over; the
implementation is a much smaller, acados-native rebuild.

## Why not RL (PPO/SAC)

Feasible in principle, but: (1) it is the standard Isaac Lab tutorial
example, so it demonstrates little beyond following a tutorial; (2) training
cost is large relative to a 5-day budget; (3) no explainable safety guarantee
comparable to "constraint margin was >= 0 at every solve", which matters
here; (4) sim-to-real gap. MPC's per-step optimality certificate and
legible failure modes are a better fit for this project's goals.

## Why not target a paper venue

No vision contribution (rules out CVPR), and the pipeline itself (capsule
proxies + soft constraints + online obstacle parameters + SQP_RTI) is a
well-established combination in the MPC-for-manipulation literature, not a
novel contribution for ICRA/IROS. Scoped explicitly as an applied,
well-engineered system instead.

## Collision-avoidance formulation

- Checked per-link, not just at the end-effector: 3-4 sphere proxies along
  the arm (see `core/collision.py`, `PROXY_FRAMES` in
  `mpc_controller_node.py`), because a mid-link or elbow collision is a
  realistic failure mode a single end-effector check would miss.
- Constraint is a *surface-to-surface* clearance: `dist(centers) - (r_proxy
  + r_obstacle) - margin >= 0`, not a threshold on center distance — a
  center-distance-only check can under-report risk once radii differ.
- Proxy radii must be an over-approximation of the real link mesh; any
  clearance the OCP believes it has is only real in PhysX if the proxy
  fully contains the actual geometry.
- `margin` bundles two terms: a fixed safety margin, and (if per-step
  travel is non-negligible relative to proxy size) a tunneling allowance —
  discretizing time coarsely can let two consecutive shooting nodes
  straddle an obstacle without either node registering a violation. The
  alternative to a bigger margin is a smaller `dt`; both are tunable in
  `MPCConfig`.
- Soft (slacked) constraints: a suddenly-appearing obstacle can make the
  previous instant's trajectory momentarily infeasible under a hard
  constraint, which would make SQP_RTI fail to return a step exactly when a
  step is most needed. acados' `zl/zu`/`Zl/Zu` slack penalties let the QP
  degrade smoothly instead.

## Why MuJoCo instead of Isaac Sim

Isaac Sim was the original plant choice, but two independent hardware
findings ruled it out for this timeline:
1. Isaac Sim's renderer (Omniverse/Kit, Vulkan on Linux) does not run under
   WSL2 — confirmed via NVIDIA forums and multiple upstream issues, failing
   at `VkResult: ERROR_INCOMPATIBLE_DRIVER` regardless of VRAM. That forces
   Isaac Sim onto native Windows, splitting the stack across an OS boundary
   from ROS2/acados (which stay in WSL2/Linux).
2. Even natively on Windows (where Isaac Sim uses D3D12, sidestepping the
   Vulkan/WSL2 problem), the Isaac Sim Compatibility Checker reported this
   machine's 8.55GB VRAM below Isaac Sim's 10GB minimum -- driver, CPU,
   RAM, and OS all passed; only VRAM failed.

MuJoCo is the direct substitute for this exact situation: no GPU/Vulkan
dependency (runs natively in WSL2, closing the OS-boundary problem too),
first-class Python bindings, and it occupies the same niche as Isaac
Sim/Isaac Lab in the robot-learning and control literature -- fast,
GPU-capable-when-available rigid-body physics for robot control and
learning, MJCF instead of USD as the scene format. It is the standard
alternative cited whenever Isaac Sim's GPU requirement can't be met, and
NVIDIA's own Isaac Lab has since added a MuJoCo-backend option, underscoring
that the two occupy the same role rather than being unrelated tools.

The ROS2<->MuJoCo bridge (`mujoco_sim_node.py`) is hand-written rather than
using an existing wrapper: reading qpos/qvel out of `mjData` into a
`sensor_msgs/JointState` at the control loop's rate (not the physics
engine's much faster internal step rate), reading obstacle body poses
directly out of `mjData` as ground truth, and writing the commanded torque
into `mjData.ctrl` before advancing exactly enough substeps to cover one
control period. Writing this by hand, instead of using Gazebo's built-in
ROS2 integration, is deliberate: it demonstrates the same understanding a
canned sim<->ROS2 bridge would otherwise hide.

`mujoco_sim_node.py` also owns an optional live viewer (on by default, see
its `render` ROS param), which is why it's meant to be treated as a
separate, longer-lived process from the other three nodes: the plant/scene
only needs reloading (and the viewer window only needs restarting) when the
robot/scene model itself changes, not when tuning `mpc_controller`'s OCP --
edit and restart just that node, and watch the same running viewer respond.

## Why Python (rclpy), not C++ (rclcpp), for this project

acados already separates into two layers: `acados_template` (Python) builds
the OCP and generates C code; the generated solver itself is a compiled C
shared library with a plain C API, called via a thin ctypes wrapper. The QP
math runs at C speed regardless of which language calls `solver.solve()` --
so at this project's 20-50ms control period, Python/rclpy's overhead
(message (de)serialization, callback dispatch, the GIL) is negligible next
to the solve itself. Python end-to-end is also simply faster to build,
debug, and iterate on inside a 5-day budget, and matches how most
MPC-for-manipulation research code is actually written.

If this were ever deployed on real hardware requiring harder real-time
guarantees (sub-millisecond jitter, no GC pauses), the correct change is
not a rewrite of the OCP: it's swapping `mpc_controller_node`'s Python/rclpy
shell for a C++/rclcpp node that calls the same acados-generated C solver
library directly. The OCP formulation and generated solver code are
unaffected; only the thin ROS2 wrapper around them changes. Scoped out here
as a named future extension, not attempted in this project.

## Perception scope

The manipulator's own joint feedback (q, q_dot) is real, continuous sim
feedback, standing in for encoders. Two perception pipelines sit on top,
deliberately separate (`handover_notes/obstacle-handling-spec.md`, sec. 2):

- the **wrist camera** (`container_cam`) owns pick and place: heightmap,
  box segmentation, grasp and slot choice. It is not a safety sensor and
  only sees anything when the arm is over the pile or tray;
- **people sensing** around the cell: two planar
  safety-lidar stand-ins at leg height by default, the world-fixed overhead
  **ceiling camera** (`workspace_cam`) optional. Details in the next section
  and in "Realistic cell" below.

What is still simulated rather than sensed: the grasp itself (a weld that
takes hold only on contact) and, for the obstacle channel, the whole
certified-safety layer (below). The box's height comes from the touch-down
at the tray, its mass from the wrist load cell.

## Workspace obstacle sensing and handling

Goal: while the pick-and-place cycle runs, something enters the cell. A
person is never steered around with a payload: the arm slows as they
approach, stops and waits while they are too close, and carries on once
they have gone. Something that is not a person and stays still is treated
as an object: plan a detour and carry on. The demo scenario (default
`visit` mode) is a person walking up to the tray to check on it, standing
there, and leaving: they come in from outside the camera's view and stop
beside the pallet on the side the arm comes in from, right on its carry
arc, face the pallet for 6 s, and walk back out. Like a real person they
never walk into the robot: if the arm is already stopped in their way they
stop short of it (the actor reads the arm's true pose; it is the
environment) and check from there.

**Scale** (superseded: the cell is now full size, a 1.75 m
person walking 1.2 m/s and the standard's 1.6 m/s; see "Realistic cell"
below). The Panda reaches 0.855 m; the arms people actually share
space with in palletizing reach about twice that (UR20: 1.75 m; fenced
industrial palletizers such as the ABB IRB 460, FANUC M-410 or KUKA KR
QUANTEC PA reach 2.4-3.2 m). The cell is therefore treated as a half-scale
model: the person is 0.84 m tall (legs, torso, arms, head; ~0.29 m across
the arms), walks at 0.55 m/s (a normal 1.1 m/s at full scale), and the
supervisor assumes 0.8 m/s (the standard's conservative 1.6 m/s, halved);
the "anything this tall is a person" threshold is 0.5 m. The boxes were
scaled up 1.5x (6-15 cm) and the destination is a tray on the floor, like a
pallet.

**Truth is separated from belief.** `dynamic_obstacle_node` is the
environment actor: it moves a person model or a carton-sized sphere, and
publishes ground truth on
`/env/dynamic_obstacle`. Only the plant (to move the body) and the monitoring
node read that topic. The controller learns about the obstacle exclusively
through the camera, so sensing error, latency and occlusion are real effects
in the results, not assumptions.

```
lidars (15 Hz)          /env/lidar_scan       ──▶ lidar_detection_node  ──▶ /perception/lidar_detections
workspace_cam (optional) /env/workspace_depth ──▶ camera_detection_node ──▶ /perception/camera_detections
  (sim_sensors_node: raw data only)                (perception/*_detection.py)        │
obstacle_supervisor_node: tracker + classifier (perception/obstacle_tracking.py)
  │   person-class ──▶ protective distance vs 4 arm points ──▶ /mpc/speed_scale, /mpc/hold
  │   static       ──▶ latched position/radius             ──▶ /mpc/static_obstacle
  ▼
task_node: hold / speed scale act on its Ruckig reference; static obstacle goes
           into reserved OCP obstacle slot 2 and drives detour planning
```

**Detection** (`perception/obstacle_detection.py`, no ROS/MuJoCo): a pixel is
foreground if it unprojects to a point above the floor plane, so the static
reference is calibration (`FLOOR_Z`), not a captured image. That only works
because the pile and tray, the two regions that change over a run, are masked
out by region and left to the wrist-camera pipeline. The arm and the held
box were first removed by rendering segmentation ids; now they are
masked kinematically, as a real cell would: the method's own robot model at the
measured joint angles and the held box from its sensed size, projected into the
calibrated camera (`camera_detection_node`). The obstacle's
own id is never whitelisted, that would be ground truth leaking in. The
pile and tray masks only reach up to just above their contents, so a person
leaning over them is still seen. The camera is at ceiling height (3.5 m):
at 2 m a standing person's head nearly touches the lens.

Every detection is described as a **column**: XY centre, radius (the
farthest visible point from the centre, so anything standing at an angle is
fully enclosed), and top height. A camera looking down cannot see under
anything, so the space below what it sees must be assumed occupied; a
person is a column from the floor to their head, not a ball at chest
height. Blobs within a few cm of each other are merged into one object
(seen from above, a person's arms only just touch the torso and otherwise
come out as separate small blobs). A sphere (centre one radius below the
top) is still produced for the OCP's sphere-only obstacle slots.

**Tracking and classification without ML.** Nearest-centroid gated
association, alpha-beta filter. Anything 1 m tall or taller, and anything
that has ever walked, is a `person` for good: a person standing still to
look at the tray is still a person, and must never become a detour target.
Otherwise a track is `person` unless it has been seen for `CONFIRM_S` (1 s)
essentially motionless; anything uncertain is `person`, the fail-safe
direction. A confirmed `static` track reverts only on sustained motion
(6 fast frames) or 15 cm displacement, because the arm passing over an object
occludes it from the overhead camera and the blob shrinks and shifts by
several cm; it reports the position and radius latched at confirmation and is
remembered for 10 s when hidden.

**Speed and separation monitoring.** For every person-class track the gap
from their column (floor to top) to four arm points (link3, link5, link7,
TCP) is compared with a computed
protective distance, ISO 13855 style: `human_speed * (latency + t_stop) +
v_robot * latency + v_robot^2 / (2 a_max) + uncertainty`. Latency is the
measured capture-to-receipt delay plus one 100 ms frame period (a walker
moves 0.16 m between frames) plus two control ticks. `speed_scale` is a
monotone function of the margin (gap minus required): 0 at or below zero,
a 15% crawl just above it, 100% one ramp (0.6 m) beyond; a 5 cm hysteresis
avoids chatter on release. The uncertainty pad (5 cm), human speed and
`a_max` are defaults, not tuned or measured values.

**Level carry and how the arm swings.** The gripper's straight-down
orientation cost used to be switched off on the lateral travel legs; the
wrist then rolled freely during the carry (measured up to ~50 deg of tilt
and ~150 deg of spin of the held box on every pile-to-tray move). It is now
on for every leg (at most ~2.6 deg of tilt per carry); this also removed a
fast joint swing after each tray scan, caused by the cost switching back on.

Making the whole move work with the box level took three more changes,
each found by a stall in a live run:
- *Arc, not chord.* From the pile to the pallet the base swings ~155 deg;
  the straight Cartesian line between them passes ~0.1 m from the base
  axis. Moves that swing more than 50 deg now go through points on an arc
  (40 deg apart, >= 0.40 m from the base axis) and flow through them
  without stopping (`task_node._next_arc_point`).
- *The box turns with the base.* Holding one world heading through that
  swing needs joint 7 about 10 deg past its limit, which made the
  controller refuse to turn the base at all and reach the pallet by bending
  back over itself. So the heading turns smoothly with the base, in
  proportion to how far round it has come, and the box arrives turned 180
  deg (same footprint, still square to the pallet). Changing it in steps
  per arc point tipped the box by up to 11 deg.
- *The base faces where it is going.* The OCP's posture term keeps joints
  near mid-range, which for the base means "face +x". `mpc_controller` now
  sets that term's reference for joint 1 to the goal's azimuth every tick
  (through the stage-cost yref, no recompile).

**Hold lives in `task_node`.** Hold switches the Ruckig reference to its
velocity interface with a zero target: it brakes at the full acceleration
limit (the braking capability the protective distance assumes) with no
jump, and the waypoint target is untouched, so release just switches back
and the same move continues. The speed scale multiplies max velocity only,
not acceleration or jerk. `mpc_controller` is unchanged: it keeps tracking
whatever goal it is given. A controller settle event only counts as an
arrival if the reference is actually at its waypoint, otherwise a hold would
fire pick/place early. If the supervisor goes silent for 1 s `task_node`
holds; if detections stop for 0.5 s the supervisor holds.

**Detour planning.** The OCP's obstacle constraint is soft and local, so an
obstacle sitting on the straight line to the goal traps the MPC in front of
it (measured ~2.5 s stalls). `task_node._plan_via` therefore inserts one
via-point when the straight TCP path is blocked, and re-plans if the obstacle
appears mid-leg: candidates to the left, right and over the top, each grown
until both new legs clear the obstacle by its radius plus the flange proxy,
the OCP margin and the held box, restricted to the arm's reach annulus and
to legs that do not cut through the base; the shortest path wins. It is
deliberately simple (see `known_issues.md`, C4).

**Scope honesty.** There is no analog in this simulation of the spec's
certified safety scanners, PLC or safety-rated monitored stop. The
supervisor is a software stand-in and is not a safety function; the "vision
requests, scanner overrides" split is not exercised. The mobile-base parts of
the spec do not apply. Occlusion is a real limit of one fixed camera.

**Measured results** (headless, MuJoCo, WSL2; per-tick logging with ground
truth used only for evaluation; numbers depend on this environment, see
`known_issues.md` B4):

| Check | Result |
|---|---|
| No obstacle, full 8-box cycle (arm and held-box self-motion) | 1300-1800 frames per run, 0 detections, 0 tracks; ~18 s per box |
| Person visits the pallet 3 times (half scale; walks in, checks for 6 s, leaves), 8 boxes | all 8 placed. The arm slowed as they came (to 46%, 19%, ...) and was stopped before they arrived; it waited 3.4 s, 6.0 s and 6.8 s and resumed when they left. On the third visit the arm was already stopped part-way, so the person stopped short of it. Minimum true clearance between the person's body and the arm's proxy spheres 0.03 m, never inside. One track per visit, never reclassified. |
| Held box during carry | tilt at most 2.6 deg; the box turns 180 deg with the base, smoothly |
| Person detection, standing | position error ~16 mm, top height error ~1 mm, radius 0.144 m vs true 0.145 m |
| Person detection, entering at the image edge | partial views (legs only) until they are well inside the image, see `known_issues.md` E4 |
| Detection latency | 13-16 ms mean (max 45-89 ms with sim stalls); perception cycle ~12-15 ms |
| *Earlier layout* (small boxes, tray ~105 deg away): carton (r 0.07) on the transit | 8 of 8 placed via detours. **Not re-measured on the current layout**, see `known_issues.md` E5. |

## Real-time risk and mitigations

- SQP_RTI's one-QP-per-call structure keeps solve time close to constant
  per call; sub-millisecond to a few ms is typical for this DOF count and
  a short horizon (N=10-20), but this must be benchmarked on the actual
  laptop CPU under actual contention with the simulator, not assumed.
- What drives solve time up: number of collision constraints (proxies x
  obstacle slots), horizon length N, warm-start quality, and CPU
  contention from MuJoCo's physics stepping sharing the same machine (CPU,
  not GPU, contention now that the plant is MuJoCo rather than Isaac Sim).
- RTI's preparation/feedback split: precompute (linearize, condense)
  before the fresh state arrives, so the latency between "state arrives"
  and "command goes out" is only the cheap feedback-phase solve.
- Optional further mitigation: delay compensation, i.e. feed the solver a
  state extrapolated forward by the expected solve time, rather than the
  literal last-measured state.
- Fallback when the budget is missed or acados reports a solver failure:
  ramp torque toward zero (decel-to-stop), not "replay the last command".
  The last command was computed from a state that is now known-stale,
  which is precisely most dangerous when an obstacle has just appeared.
- Primary lever is still problem sizing (N, constraint count) chosen so the
  worst case fits the budget by design; the fallback exists for the
  residual risk, not as the main plan.
- Confirmed empirically, not just theoretically: once a per-stage
  end-effector tracking cost was added (MPCConfig.w_ee_track, see
  implementation_notes.md#ocppy), a single ~20ms tick of round-trip dead time in
  the control loop was enough to turn a clean, tightly-converging
  controller into one that oscillates rather than settling. Simulating
  that exact delay against a direct MuJoCo-physics + solver loop
  reproduced it precisely, confirming the mechanism rather than guessing
  at it.
- This is why mpc_controller owns the plant-facing command write directly
  instead of relaying through a separate node over its own ROS2 topic (an
  earlier version of this project used a separate trajectory_executor
  node for exactly that): every hop between "compute a command" and
  "apply it" is a hop with its own independently-timed ROS2 timer and no
  phase coordination with the others, which is exactly where that dead
  time came from. Removing one hop (folding the relay into
  mpc_controller) measurably raised how aggressive w_ee_track could be
  before the live pipeline destabilized again -- from roughly 100-300
  with the relay node, to roughly 5,000-20,000 without it, at otherwise
  the same tuning.
- This project targets simulation, not real hardware, but the control
  loop is deliberately structured as if it might be ported: read state,
  solve, write the command, all inside one tightly-coupled loop, with
  ROS2 topics reserved for slower, non-critical traffic (goal updates,
  obstacle params, diagnostics) rather than the hot path. That's not
  optional polish -- a real torque-controlled arm (e.g. a Franka Panda's
  FCI) drops out of active control if it doesn't get a fresh command
  within about 1ms, and ROS2 pub/sub over DDS has no hard real-time
  delivery guarantee to lean on for that. Porting to real hardware means
  swapping mpc_controller's plant-facing lines for a direct call into the
  robot's own real-time interface, not restructuring the node graph.
  Residual dead time (sensor sampling, the solve itself) would still
  exist even in a single tightly-coupled loop on real hardware, just
  orders of magnitude smaller than an inter-node ROS2 relay -- proper
  delay compensation (extrapolate the measured state forward by that
  much smaller, known latency) remains the right tool for whatever's
  left, and is still a named future extension here, not implemented.

### Lockstep plant, one clock for the task

Folding the relay into mpc_controller did not remove the last source of
dead time. mujoco_sim_node and mpc_controller still ran two independent
20 ms wall-clock timers, and the phase between them was set by chance at
every launch. Per-tick telemetry (the plant stamps each published state
with its step index, the controller echoes it on the command) showed
25-35% of physics steps applying a torque computed from a state one step
old, varying from launch to launch. An offline lockstep replay of a
recorded run (`scripts/dev/replay_offline.py`) reproduced the rest: with
no dead time the same goal stream tracks cleanly; with 30% of ticks late
the joint-speed "shake" rises ~4x; with most ticks late the first move
(a large goal step) either oscillates at 8-10 rad/s or fails with
ACADOS_MINSTEP. That was the flaky start and the good-run/bad-run
variance.

What changed:
- **Lockstep** (superseded by real-time stepping, see "Realistic
  cell"; kept as `lockstep:=true` for debugging). The controller solves when a state arrives (no own timer);
  the plant waits for the command computed from the state it just
  published (up to 50 ms, timed from when the state went out or the
  plant thread came back from a render stall), paced to real time.
  State and command subscriptions have depth 1. Command staleness is
  counted in plant steps, not wall time.
- **Task on the plant's clock.** task_node advances its Ruckig reference
  once per published state and stamps each goal (and heading) with the
  step it is for; the controller uses the goal for exactly the state it
  solves from. On its own timer, the goal stood still and then jumped
  double on ~4% of moving ticks, which was most of the remaining wrist
  chatter. If the stamped goal has not arrived yet (plant catching up
  after a stall, ~1 per run), the controller extrapolates it from the two
  newest ones.
- **No hand-off at rest.** The MPC keeps solving at the goal; "settled"
  only signals arrival. The PD-hold hand-off switched control laws after
  every stop and resumed from a stale warm start.
- **Clean start.** The solver's guess is initialised from the measured
  state (and gravity torque) before the first solve and after a failure;
  task_node holds its reference on the measured TCP until the controller
  starts (the old constant `HOME_EE_POSITION` was 10 cm above the real
  TCP, and the reference had long finished its first move before ENTER).
- **Motion shape.** Pass-through points (Ruckig target velocity) instead of
  a stop at every waypoint; the box heading eased in and out along a
  swing; the pile/tray hulls grow in behind the arm instead of switching
  on around it.
- **Swings planned round the base.** With pass-through arc
  points in x/y/z, the reference still slowed at every 40 deg corner (0.5
  -> 0.25-0.36 m/s about every 0.7 s): a ~1.4 Hz sway of the whole arm
  (joints 2 and 4 most), which the fast-jitter number did not see. The
  Ruckig reference is now planned in cylindrical coordinates (azimuth,
  radius, height), so a swing is one arc with one speed profile; x/y/z and
  arc points remain only while a static-obstacle detour is needed (the
  detour check assumes straight segments). Pass-through points are passed
  along the incoming direction at cos(half the turn angle) of the cruise
  speed, capped at what is reachable in the distance left. The two
  departing 90 deg corners (top of the lift, box in hand; top of the
  lift-off, empty) are blended: the swing starts 5 cm before the top, so
  rising and turning overlap (the corner rounds off at ~0.3 m/s instead of
  nearly stopping). Arriving corners are passed exactly, so nothing is
  lowered early over a stack.
- **Resume after a person leaves.** On release of a hold the target is
  re-planned from where the reference stopped; a pass-through point that is
  within 3 cm or was already passed while braking is skipped; the pass speed
  scales with the supervisor's speed limit instead of being clipped per
  axis; and a reference close to a pass-through point and moving away
  from it counts as having passed it. Before, a restart sped up, nearly
  stopped at the next corner and sped up again, hunted while the speed
  limit rose in 0.1 s steps, or (once) orbited the lift point for 13 s.
- **Reference preview for the MPC.** Every stage used to get the same goal
  (one point 2 ticks ahead), so the MPC planned as if the target stood still
  and trailed a moving reference by ~3 cm, partly sideways (5-8 mm off the
  line on every descent into the tray). task_node now publishes the
  reference's next 0.3 s (one point and heading per stage); offline replay:
  sideways error on descents 4.2/6.7 -> 0.3/0.9 mm, tracking p99 2.5 -> 1.0 cm.
- **Placement** (superseded: compact packing, then touching perimeter, see
  "Realistic cell"). Floor placements come from an online packing rule
  (`pick_place_common/packing.py`: two rows along the long walls, each box
  flush after the previous one in its row, the row chosen looking ahead over
  the boxes the pile scan can see), checked
  against the sensed heightmap, with the heightmap search as fallback;
  snapped to 3 mm from walls/neighbours; the box centre is targeted using
  the in-hand offset; the tray floor height is the real one. See
  known_issues.md H.
- **Monotonic timeouts.** WSL2 steps the system clock forward ~1.5 s about
  every 34 s. In-process timeouts (supervisor perception fail-safe,
  task_node's supervisor-silence check) now use the monotonic clock; the
  periodic false HOLD is gone.

Measured with the viewer and dashboards on (same launch script, one run
each; `scripts/dev/bench_analyze.py` columns): joint-speed shake (RMS of
qdot minus its 100 ms average) 0.76-1.11 -> 0.02 rad/s, joint speed max
8-10.5 -> 2.4 rad/s, joint acceleration p99 219-281 -> 11 rad/s^2, carried
box tilt max 13 -> 0.5 deg, stops per box 11 -> 4, full task 162-197 ->
133 s.

## Realistic cell and a genuine pipeline

Plan: `handover_notes/realistic_cell_plan.md`; reasons per module:
`implementation_notes.md`; measurements: `system_overview.md` section 10.

- **A genuine perception, plan, action pipeline.** The rule for the method:
  use only what a real robot could sense or a real integrator would teach
  (the two zones, sensor calibration, the robot's own model, a box spec).
  Applied, it removed the last shortcuts: the grasp now takes hold only when
  the tool is on a box (the method lowers until the wrist load cell feels
  contact and checks the weight after the lift), and the wrist camera's pose
  comes from the measured joint angles and the hand-eye calibration. Section 14
  of `system_overview.md` lists what is sensed, what is taught and what is
  still a stand-in.
- **Real time, not lockstep.** Lockstep removed dead time but also hid late
  answers: a real arm does not wait for its controller. The plant now steps on
  its own clock and switches to the newest command 14 ms into each tick; the
  controller predicts its state to that moment with an EKF; states and commands
  go best-effort, newest only. Drawing moved to a separate sensors process so
  it never starves the physics or the solver.
- **Realistic sensors.** Encoder and load-cell noise and bias, depth noise, a
  wrist camera on the wrist housing's side tab (the arm in view, the tool
  masked), and a full-size cell: base and tables at 0.75 m, a 1.75 m person.
- **Lidars for people.** Two 270 deg planar scanners at the base's free
  corners, 0.20 m above the floor, as a real cobot cell would use; leg
  clusters plus the ISO 13855 intrusion allowance (1.12 m) make the protective
  radius. The ceiling camera stays optional (it can tell a static object, which
  the lidar cannot, so the detour planner needs it), with a kinematic
  self-filter instead of segmentation ids.
- **Packing by touching perimeter** (*superseded*: block packing and
  the guarded slide, below). The smallest-group-rectangle rule left
  boxes off the walls, and a spot the robot could not finish was dropped
  outright. The rule now prefers the most edge contact (with look-ahead,
  filling from the far corner) and asks the robot where the box would really
  end up. Offline: all eight boxes on the floor in 76% of random orders, vs 29%.
- **Motion quality measured, not eyeballed.** `motion_check.py` measures
  sweeping (self-motion, back-and-forth of turning joints, detours, creep)
  next to `bench_analyze.py`'s stutter measures. It found the wrist winding
  up and back on every carry (the heading set per leg), a wrist spin before
  tiny pushes, elbow creep during slow set-downs and a jolt when a push
  stalled; all fixed (the heading turns in step with the base over a run of
  legs, a 1.5 rad/s turn-rate cap, no pushes under 6 mm, a stiffer posture
  term, the push force eased off).

## Mobile manipulator, step 1: the room and a mobile base

Plan: `handover_notes/mobile_manipulator_plan.md` (local); reasons per module:
`implementation_notes.md` (scene.py, frames.py, base_drive.py,
base_odometry.py, base_node.py).

- **The world is the room; the arm task keeps the arm's frame.** The plant's
  world became a 10 x 8 m room (now 7.0 x 5.6 m), and the robot a
  differential-drive base with
  the arm on a pedestal. The method's arm task (perception, packing, the
  reference, the MPC) stays in the arm's base frame, which is what a real
  mobile manipulator's arm software sees; the plant converts what its sensors
  report into the frames real drivers would use. Rather than move every
  constant, the old cell is included in the room inside a frame at its pose,
  and that pose is deliberately not the identity (yaw 180 deg), so any
  leftover "robot at the origin" assumption fails visibly. Offline tools that
  work in the arm frame load the same scene re-expressed with the parked arm at
  the origin.
- **The arm file stays robot-only.** Pinocchio builds the controller's model
  from `panda_robot.xml` alone, so the plant attaches it to the chassis at load
  (MjSpec) instead of the file growing a base.
- **Physical wheels, not a kinematic base.** Slip and odometry error have to
  come from the physics for SLAM to be tested honestly, so the wheels roll on
  the floor by friction. Getting that believable took sprung drive wheels, an
  explicit wheel/chassis contact exclusion, a rounded tread, hard wheel-floor
  contact pairs, the drives' speed loop moved out of MuJoCo's implicitly
  integrated velocity actuator into the plant (every substep), and, parked, a
  brake on the wheels plus static friction at their contact points (a pin that
  lets go above the friction limit). Each fixed a measured artefact: wheels
  lifting in turns, a flickering contact, scrub in turns, the chassis outrunning
  its wheels by 23 mm per start, and a parked base creeping (3 deg during one
  push into the tray; then still 0.1-0.3 deg per run from the arm's swings,
  enough to put a false person into the lidars' view). The global `noslip`
  solver also stopped the creep but made the arm's joint friction perfectly
  sticky, which changed the arm plant; it was not kept.
- **Odometry is the method's, from encoders and a gyro.** The plant gives no
  pose; the method integrates quantized wheel angles and a biased gyro with the
  taught wheel radius and track, against wheels that are 0.5% off. The gyro
  carries the heading (encoder-only heading was off by up to 6 deg per loop),
  its bias learnt whenever the wheels stand still.
- **Docking head-on.** An offline reach check (`dock_reach_check.py`) over both
  zones from candidate docking poses: with the table straight ahead and 5 cm
  between the chassis and the table, the arm reaches 92% of the pick zone and
  96% of the place zone grid (tool down, box tops to the scan height), more than
  from the old fixed cell (91% and 90%); docking beside the table reaches
  77-83%. The final layout (tables in opposite corners) will dock head-on.

## Mobile manipulator, step 2: SLAM and localization, two ways

Reasons per module: `implementation_notes.md` (slam).

- **Two systems on the same input, compared on the same recordings.**
  slam_toolbox is the ROS 2 standard (Karto scan matching, a Ceres pose graph);
  ours is a small graph SLAM written here (point-to-line ICP, an SE(2) pose graph,
  an occupancy grid) so every step can be measured and changed. Both take the two
  lidars and the base's odometry, run offline on recorded drives (played to
  slam_toolbox in simulated time) and live, and are scored against the true pose
  the plant keeps for validation only.
- **Map once, then localize.** A commissioning drive (two laps of the room)
  builds the map, anchored at the dock where the robot starts; jobs localize in it
  from the dock. Both systems' maps are saved in the ROS map_server format.
- **Ours is the default** (`slam:=own`; `slam:=toolbox` stays): it was the more
  accurate in mapping (2-3 mm median against 14-19 mm) and localizing (2-3 mm
  against 11-13 mm), uses less CPU when localizing (2.5 against 18-24 ms per scan)
  and traces people out of the map. slam_toolbox is well inside the 5 cm / 2 deg
  gate too, and is the one to reach for in larger or harder spaces (its scan
  matcher searches correlatively, ours trusts the odometry's prediction).
- **What it took**: thinning dense local maps before matching, a larger normal
  neighbourhood for saved maps (their walls are two cells thick), matching against
  the cells' mean hit positions rather than a distance field (no cell
  quantization), scans stamped with the plant state's time, and each scan paired
  with the odometry of its own tick (a tick off was 1 deg at 0.8 rad/s).
- **The driver is a stand-in.** The commissioning and test drives are steered on
  the true pose by a scenario actor (a technician with a joystick) until the
  navigation of step 4 drives the robot on its own estimate.

## Mobile manipulator, step 3: people on a moving base

Reasons per module: `implementation_notes.md` (map_people.py, people_tracker.py).

- **The final layout first.** The two tables moved to opposite walls (the stations
  layout) before the people work, so the people walk the room the job runs in and
  the room is mapped once for it. Each table keeps its pose relative to the docked
  arm, so the arm task's taught zones and tuning hold at each station; the cell
  layout stays as the fixed-base regression.
- **Foreground against the map, not against a background per beam.** On a moving
  base the per-beam background is meaningless; with the robot localized, anything
  15 cm or more off the mapped surfaces is foreground. The leg and pairing rules are
  the parked cell's, shared.
- **Tracks in the map frame with velocity**, from a Kalman filter, because step 4's
  base MPC predicts people over its horizon and the safety fields need their speed;
  standing or moving per track. Negative evidence (the lidars saw through the
  predicted spot) ends a track at once, so a person turning leaves no ghost.
- **A crowd that is not polite.** Six people: some give way to the robot, some walk
  straight through its path, one stands then walks off, one stands by the pick
  station; they wait for each other. The test is the worst case for the detector
  (people close to each other, occluding each other) and later for the navigation.

## Mobile manipulator, step 4: navigation

Reasons per module: `implementation_notes.md` (planner.py, base_mpc.py, safety.py,
navigator.py).

- **Three layers, as for the arm.** A kinematic route planner (Hybrid A\* with the
  base's own moves on the saved map), a base MPC (acados, unicycle) that follows it
  and keeps people at a distance over its horizon, and a separate safety layer
  (protective and warning fields in the manner of ISO 3691-4 / R15.08, and
  watchdogs on the scans, odometry, people and localization) between the MPC and
  the drives. The planner knows the room, the MPC the people's motion, the safety
  layer only what the lidars see now.
- **Docking is its own sequence.** Route to a pre-dock pose 0.9 m before the table,
  align on the dock's axis (turn, straight, turn: a differential drive cannot step
  sideways; now only where no forward curve onto the line fits, below),
  approach straight and slowly, check, back out and retry if outside 2 cm
  / 1.5 deg. In the docking moves people are left to the safety layer: stop and wait,
  no swerving next to a table.
- **Industrial manners round people.** Keep the lane and slow down rather than
  weave (lateral error dear, lag cheap; a speed limit near people); never close in
  on anyone, and do not flee either (a person walking at the robot is stopped for,
  not run from); reverse only where the plan does; people standing in the way for 4 s
  get a new route round them. Measured: the swerves through a crowd of six that does
  not give way fell from 22 to 3-8 in 20 drives, oscillations to 0-1, turn-backs to
  0-1.
- **The protective field is the stop.** The chassis swept along its current arc until
  it can stop, plus a margin, and only what it is closing on; nothing at a standstill.
  A box at a standstill kept a robot and a person facing each other for good, and the
  whole corners' circle for a turn in place held it beside a table leg.
- **Scanners on the corners, seeing from the window.** Each corner scanner's optical
  centre on the chassis's corner, so its 270 deg runs along both faces and the two see
  all round and nothing of the robot; no minimum range (a leg against a housing was
  invisible). Outside the corners the housings scraped the pick table's legs; inside
  them the chassis would hide the sides.
- **Scored on the truth, like the arm.** The people's true legs against the
  chassis, people inside the field of the actual motion, and the motion measured as
  `motion_check.py` measures the arm's (detour, extra turning, wiggles,
  oscillations, turn-backs). 20 drives each in a quiet room and in two seeds of the
  crowd: all arrived, docked within 7 mm and 0.31 deg, no field intrusion, no
  contact; live the same on 1 + 1 runs.
- **The crowd had to learn manners too.** People who are blocked by the robot now
  step aside after 3 s, and people waiting for each other walk on after 2 s; before,
  mutual waits (robot and person, two people) never ended.

## Mobile manipulator, step 5: the mobile pick and place

Reasons per module: `implementation_notes.md` (the mobile job).

- **The cell's task, unchanged at each station.** The arm still scans, picks, places
  and pushes as it did beside the parked cell, in its own frame; what is new is
  between the stations: the arm tucks into a carry pose, the task sends the base to
  the next station and waits for it to dock. The zones stay as taught: the docks put
  the pile and the tray where the cell had them, within the zones.
- **A carry pose inside the footprint.** Tool down behind the arm's base over the rear
  deck, the box low and near the base's middle: nothing sticks out while driving, the
  box is clear of the pedestal and the arm's joints are far from their limits. The arm
  goes in across at height and comes out straight up, so a box never swings low over
  a tray wall.
- **The tray is found again at every visit.** Docking repeats to a few mm and tenths of
  a degree; the tray's walls are found again in the scan taken with the box held (one
  wall per axis is enough, the box hides one) and everything placed moves with them.
  A tray not where it should be means docking again.
- **One people perception for both.** The people seen against the map serve the base's
  navigation and, moved into the arm's frame, the arm's speed-and-separation
  supervisor, which keeps its input format. A localization error cannot hide anyone
  from the arm, only add false people.
- **Two safety regimes, by mode.** While the arm works the base is docked and the
  arm's supervisor holds it for people (ISO 13855 separation); while the base drives
  the arm is still in the carry pose and the base's fields stop it (ISO 3691-4 style).
  The validation measures them apart.
- **The crowd for the job comes and goes.** The step-3 crowd has a bystander standing
  for good 1.3 m from the pick station's arm: under speed and separation monitoring
  that holds the arm for ever, and the job waited, correctly. The job's crowd is the
  same except that this bystander stands there 30 s at a time.

## Mobile manipulator, step 6: the arm and the base at once

Reasons per module: `implementation_notes.md` (arm and base together, wb_ocp.py).

- **What there was to gain.** Of the quiet job's 20 min, the base drove 15 (the docking
  approach 3, the undock 2), the arm folded and unfolded for 2 more, and the carry was
  already steady (the tool within 6 mm of its pose while driving). Folding while the
  base backs out, unfolding and reaching the station's first pose while it docks, is
  the time there was to save; carrying more steadily was not needed.
- **Three ways tried offline, the plainest kept.** The arm's legs as they are, only
  started while the base moves ("overlap"); the arm's goals held fixed in the room
  while the base moves, through the two controllers as they are ("split"); the plan's
  whole-body MPC, base and arm in one OCP (core/wb_ocp.py). Docking and tracking came
  out the same in all three (within 1 mm, the tool within 5 mm of its reference);
  overlap was the fastest (a pick-and-place cycle's four dockings 34.6 s, split 36.5,
  whole-body 36.0, one after the other 50.0), because the station's first poses are
  scan poses high over the tables: nothing is gained by holding them still in the
  room while the base moves, and doing so waits until they are within reach. The
  whole-body OCP stays in the repository as the evaluated alternative.
- **Safety while both move.** The base's fields watch the chassis, the arm's
  supervisor the arm. Off the docking line (a route, an alignment turn) the base moves
  only with the arm stowed; while the arm is out and the base is docking or undocking,
  a hold by the arm's supervisor pauses the base too. The supervisor now tracks the
  lidar's people in the odometry frame and adds the base's motion to the arm's speed
  toward them: on a turning base a coasting track in the arm frame had drifted 0.3 m
  and held the arm for nobody.
- **Scans wait for docking.** The arm may reach its scan pose before the base stops;
  the picture is taken once it has docked, so what the arm sees is where it will work.

## Mobile manipulator, step 7: the scenario set

Reasons per module: `implementation_notes.md` (the scenario set).

- **Situations, not seeds.** Five rooms for the same job: empty; three walkers; the
  crowd of six; someone standing in the pick station's docking line for 40 s at a time
  (in the way of the approach and of the back-out); someone waiting beside the route
  who steps onto it as the robot comes. Each is a different test of the same rules:
  the base stops and waits for what stands in its way, slows and steers round what
  walks; the arm holds for whoever comes near it.
- **One live job each, plus offline drives.** A live job takes 15-35 min of real time
  on this laptop; one per scenario shows the whole system together, and the offline
  navigation (20 drives, two seeds per scenario, the arm in the carry pose with the
  largest box) gives the drives' spread.
- **Mutual waits are the failure to look for.** The robot waits for people by design
  (the base for whoever stands in its way, the arm for whoever is within its separation
  distance), so a person who in turn waits for the robot stalls both. The scenarios
  found three: a scenario person whose spot was under the robot; one cornered between
  a wall, a table and the robot; and the robot itself, pausing the base for an arm
  still over its own deck, or keeping the base for an arm a few centimetres short of
  its carry pose. The people were made to move on; the robot now pauses only for an
  arm out beyond the chassis and leaves with the arm over the carry pose.
- **The same measures everywhere.** Boxes, time per box, distance; the closest anyone
  came to the chassis and to the arm and box (on the truth, the people's whole bodies);
  protective-field entries and contacts; the arm's holds and their causes; the
  localization error; the drives' motion. `job_results.py` reads them off the
  monitors.

## The room at half its floor area

Reasons: `implementation_notes.md` (the room at half its floor area).

- **Things keep their distance to their own walls.** The stations, home and the
  fixtures were placed by the wall they belong to, not scaled: the docking lines, the
  tables' gaps to their walls and the room for the chassis to turn at home and at the
  pre-dock poses stay as they were taught and validated.
- **The parked cell turned, not moved apart.** Backing out of the cell and the cell's
  visitor coming in need 5.2 m in a line; the cell was turned to lay that line along
  the room's longer side.
- **The map made again, not edited.** The robot's map is what its own SLAM saw; a
  smaller room means a new mapping drive, as at commissioning.
- **Fewer people.** The same six in half the floor made the crowd a crush (one walked
  into a nearly stopped base, 18 mm off); four in the crowd and two walkers keep the
  density about as it was.

## Speed

Reasons: `implementation_notes.md` (Speed).

- **The top speed is a setting, the rest follows it.** One number (`speed:=`, 1.0
  m/s for the job) sets the routes' cruise, the turn rate, the wheel-rim limit and the
  speed allowed near people beyond 1.8 m. The MPC's limits are set on the built
  solver, so trying a speed costs no rebuild.
- **Slow only where it matters.** The docking line is driven at 0.3 m/s and only its
  last 0.25 m, by the table, at the old 0.1 m/s; undocking and aligning at 0.3 m/s.
  Docking stayed within 7 mm and 0.4 deg.
- **The warning caps graded, not flat.** A flat box beyond the stop capped the speed
  at 0.3 m/s for anything in it, which at 1 m/s held whole corridors and every turn at
  a station. The cap is now what still stops short of the nearest point, with a
  buffer; the protective field, which decides, is unchanged.
- **Turning bounds the gain.** At 1 m/s a third of the route time is turning in
  place; 1.5 m/s gained nothing more offline, so 1.0 is the default.
- **The arm faster only where nothing is near.** Into and out of the carry pose and
  the first move at a station (high over everything) at 0.8 m/s, the corners there
  blended; the pause after a scan shortened. The rest of the arm's motion, and the
  parked cell, keep their validated limits.
- **Stepping back, only after a route round someone.** Faster arrivals brought the
  base up against someone standing near its route; backing off 0.4 m lets the route
  round them start. Where no route round them exists (someone on the goal itself)
  the base waits on its route as before: stepping back there made it shuttle.

## Safety settings

Reasons: `implementation_notes.md` (Safety settings).

- **Separate knobs for the base and the arm.** Their safety works differently (a
  protective field for the base's motion, speed and separation monitoring for the
  arm), and one may want the base to pass people closely while the arm stays cautious.
- **Margins scale, physics does not.** The stopping distances (reaction, braking) are
  what the robot can do, not a choice; a setting only moves the buffers on top.
- **A floor at a person's overhang.** The base and the arm see people's legs; the
  torso and arms reach 0.135 m beyond them. Below 0.14 m a margin would let the robot
  stop only after reaching the upper body, so the lowest settings work closer but
  still stop short of it.
- **Assumed and actual speed apart.** The arm's safety assumes a walking speed (ISO
  13855: 1.6 m/s); the scenario people walk at their own. Two knobs, so a setting can
  assume less than people actually walk, as a test.

## Localization by sensor fusion

Reasons and numbers: `implementation_notes.md` (Localization by sensor fusion).

- **Weigh, don't switch.** The localizer replaced its pose with each good scan match
  and kept the odometry's prediction otherwise: a predictor and a corrector already,
  but a hard switch, so the pose jumped by the match's own noise (11-16 mm in a tick)
  at every scan. An extended Kalman filter weighs the wheels, the gyro and each match
  by their noise; the match's weight comes from ICP's own information matrix, which
  the localizer computed and threw away.
- **Gyro for the turn rate, wheels for the speed.** The wheels' turn rate is the
  difference of two nearly equal arcs, off systematically by tyre tolerance; fused as
  noise it drags the gyro's bias estimate. It checks for slip instead.
- **Slip from what the base cannot do.** Wheels that disagree with the gyro, or that
  change speed faster than the drives can move the chassis, have let go of the floor.
  A skid is an episode as long as the worst floor would let it last; through it the
  scans, not the wheels, carry the pose. The first, one-tick version let a wet stop
  run 81 mm off with the scans rejected.
- **Same data, both estimators (shadow mode).** Driving with one changes the
  trajectory the other would have seen; running both on every tick and scan compares
  them on identical data, offline and live, and makes the switch a parameter.
- **Conditions that make the difference visible.** The simulated floor barely slips
  and the room has walls everywhere. A wet patch, a worn tyre, a drifting gyro and
  scan dropouts are plant-side and selectable; crowding round the robot was tried and
  hides too little wall to matter.
- **Consistency against the room, not the map.** The filter is consistent with its
  map; the map is a few millimetres off the room in places, which no estimator in the
  map frame can see. The covariance it reports adds that (2.5 mm, 0.04 deg).
- **Downstream unchanged.** The filter publishes map > odom like the localizer did;
  base_node's odometry, the navigation and the safety layer are untouched. The
  covariance goes out on its own topic, for the navigation to use later.

## Docking facing the pick table

Reasons and numbers: `implementation_notes.md` (Docking facing the pick table).

- **Face both tables.** The pick dock kept the old cell's pose (the table beside the
  chassis) so the arm's taught zone and tuning carried over; the cost was the base's: it
  lined up along the table from its south end, turning almost all the way round on every
  drive to it. Facing the table, as at the tray's, the docking line points into the room;
  the arm's pick zone moves in front of it (reach 99.5%, better than side-on), and joint 1
  swings further from the carry pose (about 1.3 s per box, mostly while docking).
- **Arrive lined up, don't stop to turn.** The route joins the docking line on an arc and
  rolls on into the slow straight approach; the stop-turn-settle at a waypoint stays only
  as the fallback when people pushed the base off the line. After placing a box: back
  out, one turn, straight across, steer onto the line, in.
- **Docks taught by driving, not read from the layout.** The method's docks were the
  simulator's table placement moved into the map: a stand-in. Now, as on a real robot, a
  technician parks the robot at each station once and it saves its own localized pose
  next to the map; the method never reads a station or dock pose from the layout. The
  technician's parking is steered on the truth, as the mapping drive's joystick is.
- **The tables did not move.** Only where the robot parks and how it gets there changed.

## Tighter packing, placed right first time

- **Measure the placement error before tightening the gap.** Validation-only truth lines
  and `place_budget.py` split each landing exactly into the arm's tracking, the in-hand
  offset and the release. The arm's tracking dominated (1.6 mm sd) and repeated with the
  pose: the offset-free bias reset whenever the goal moved, so on a straight-down set-down
  the sideways error was never removed. Keeping the sideways bias through vertical moves
  (and slow pushes) took it to 0.2 mm without new sensing.
- **One block, across the width first.** Growing the boxes as one block from the far
  corner, measured by how far it reaches along the tray (`strip`), keeps the free space a
  rectangle; ranking by the block's bounding rectangle left an L that the big boxes,
  arriving last and unseen, did not fit (82-86% vs 97% of orders with all 8 on the floor).
- **Slide the held box, as industrial cells do** (the user's choice, against the earlier
  rule of never sliding a held box): after touch-down, lift clear, slide into the planned
  neighbour or wall until the wrist feels 1 N (under what drags a light neighbour), back
  off, release. It replaces pushing a loose box, which can wedge (40 N) when the tool
  slides along its face; pushes stay only where the wrist cannot reach flush.
- **The first box's corner is a heading question, not an accuracy one.** The wrist
  reaches 88 mm past the tool on one side: which way the tool faces at the far corner
  decides 2 mm or 46 mm off flush (known_issues W1).

## Docking without turning on the spot

- **Misses of a few cm are the approach's to take up.** In a crowd the route ended 2-5 cm
  and 1-2 deg off the docking line and the base turned on the spot to fix it; the 0.9 m
  straight approach takes up 3 cm / 3 deg, a larger miss joins the line on one forward
  curve (a cubic Hermite, its bend no tighter than 0.3 m). Parking at home keeps 2 cm /
  1 deg: no approach follows it.
- **Weaving round people standing is left as it is.** Slowing for walkers and planning
  round people standing (with early replans) were measured and gave nothing; the turning
  only moved into longer routes (known_issues W2).

## Two methods, one plant: never run them concurrently

`pick_place_mpc` (acados MPC) and `pick_place_moveit` (MoveIt Task
Constructor + a FollowJointTrajectory-to-torque bridge) share the same
plant node, scene and sensing (`pick_place_common`, `perception/`), but
each method's `demo.launch.py` starts its **own** `mujoco_sim_node`. Both
methods drive the arm by publishing `/sim/joint_command`, so running them
against one live plant would have two controllers fighting over the same
torques. The Phase 5 benchmark harness runs each method as a separate
process for the same reason.

## Debugging journal: problems found and how they were fixed

A running log of the non-obvious failures hit while getting this pipeline
from "scaffolded but never run" to reliably working, and what actually
fixed each one -- kept because the failure mode and root cause are often
more informative than the final working state alone.

- **Model/plant dynamics mismatch (missing joint damping).** The internal
  OCP dynamics (`core/dynamics.py`, Pinocchio/CasADi) called
  `cpin.aba()` without subtracting joint damping torque, while the MuJoCo
  plant simulates it. This was the single most consequential bug in the
  whole project: the controller's belief about the robot's dynamics
  disagreed with the plant's actual behavior enough to cause large
  tracking error and instability that looked like a tuning problem but
  wasn't. Fixed by computing `qddot = cpin.aba(cmodel, cdata, q, qdot, tau
  - cmodel.damping * qdot)` instead of passing `tau` straight through.
- **Arm locking onto joint limits / local minima.** Root-caused to three
  compounding gaps, all now fixed: (1) the dynamics mismatch above, (2) no
  joint *position* limits in the OCP (added as hard box constraints,
  `core/ocp.py`), (3) nothing biasing redundancy resolution for the
  arm's 4 leftover degrees of freedom (added `MPCConfig.w_q_center`, a
  cost pulling toward each joint's mid-range).
- **Receding-horizon "stall": arm gets close but never quite arrives.**
  With only a *terminal*-stage cost pulling toward the goal, the first
  applied step of a near-converged plan was nearly zero displacement --
  each individual solve "looked" fine, but real per-tick progress
  stalled a few cm short. Fixed by adding a per-stage end-effector
  tracking cost (`MPCConfig.w_ee_track`) so every stage in the horizon,
  not just the last one, is pulled toward the goal.
- **Startup free-fall.** Zero torque was commanded while acados was
  still cold-compiling the solver, so the arm just fell under gravity.
  Fixed with a stale-command fallback in `mujoco_sim_node.py`: gravity
  compensation plus PD toward a captured hold pose, engaged whenever no
  fresh command has arrived recently.
- **"Hold" wasn't actually holding.** The first version of that fallback
  was gravity-compensation only, with no restoring force -- so each joint
  drifted slightly rather than truly holding, looking like an unstable
  jitter. Fixed by adding the PD term toward a pose captured at the
  moment the hold engages (see previous item).
- **Slow iteration: every weight tweak forced a ~1-2 min acados
  rebuild.** Added a structural-fingerprint cache (`core/ocp.py`):
  the solver is only rebuilt from scratch when something *structural*
  changes (dimensions, expressions, or the source file itself, via a
  content hash -- so silent staleness can't happen); pure numeric changes
  (weights, bounds) are pushed into an already-compiled solver at runtime
  via `cost_set`/`constraints_set` instead.
- **Solver-noise jitter once genuinely at the goal.** SQP_RTI
  re-linearizes fresh every tick even at rest, so a tracking gain strong
  enough for sub-cm precision showed up as small persistent jitter right
  at convergence. Fixed with settle-and-hold (`mpc_controller_node.py`):
  once truly converged (within tolerance, at rest, for a dwell period),
  stop re-solving and hand off to the existing stale-command hold instead
  of continuing to fight solver noise.
- **Redundant-joint "rubbing" at the target.** A separate, slower-to-spot
  issue from the jitter above: because the task only constrains
  end-effector *position* (3 numbers) on a 7-DOF arm, the 4 leftover
  degrees of freedom could keep drifting through their null space for a
  couple of seconds after the tip had already reached the target -- correct
  per the settle logic's own definition (real joint motion, not solver
  noise), but visually looked like the arm "rubbing" the target sphere.
  Fixed by raising `MPCConfig.w_q_center` from `5e-2` to `1.0` so the
  null space is pulled to a fixed configuration much faster; safe to
  raise now that hard joint velocity limits (added for the transit-speed
  fix below) bound how far a stronger pull can swing the arm before those
  limits catch it -- the same weight had caused joint-limit lockups
  earlier in the project, before those limits existed.
- **Real-time dead-time destabilizing goal-tracking.** See "Real-time
  risk and mitigations" above for the full account: an inter-node ROS2
  relay (a separate `trajectory_executor` node) added enough round-trip
  latency to turn clean convergence into sustained oscillation; fixed by
  folding the plant-facing command write directly into `mpc_controller`,
  removing that hop.
- **Abrupt, non-constant-speed motion between targets.** The OCP was
  chasing a target that jumped straight to a far waypoint, producing a
  time-optimal but visually bang-bang response (saturate to accelerate,
  saturate to stop) -- not how real industrial arms are driven for
  ordinary point-to-point moves. Fixed by generating a jerk-limited
  reference trajectory online with Ruckig (the task node, then
  `goal_publisher_node.py` for the original single-box demo, now
  `tasks/pick_and_place/mpc/pick_place_mpc/task_node.py` -- same library
  Franka's own `libfranka` uses) and having the OCP simply track
  wherever that reference currently is, each tick.
- **The moving reference was less robust to timing jitter than a static
  goal had been.** A continuously-changing reference is more exposed to
  round-trip ROS2 dead time than a fixed target: by the time a command
  computed against an earlier reference value lands, the reference has
  already moved on. Fixed with look-ahead compensation
  (`task_node.py`'s `LOOKAHEAD_TICKS`): since this node
  *generates* the reference rather than measuring it, its value a few
  ticks in the future is already known exactly, so publishing it early
  costs nothing and cancels that component of the dead time.
- **Real Panda joint velocity limits were missing entirely.** Nothing
  bounded joint *velocity*, so a strong tracking gain could swing the arm
  far faster than any real robot -- and those large per-tick state swings
  were exactly what broke SQP_RTI's linearization validity under real
  ROS2 timing jitter, showing up as occasional solve failures during fast
  transitions. Fixed by adding hard velocity-limit box constraints
  sourced from Franka's official datasheet (`MPCConfig.qdot_max`).
- **MuJoCo viewer's render thread destabilizing the control loop.** The
  passive viewer's render thread was spinning unbounded -- no vsync cap
  on a machine with no GPU passthrough (WSL2, software-rendered via
  Mesa's `llvmpipe`) -- and this was disruptive enough to the
  single-threaded control loop's timing to occasionally prevent settling,
  even with plenty of idle CPU cores free (a `ps` reading of "419% CPU"
  looked alarming but was actually only ~26% of this 16-core machine's
  total capacity, so raw CPU headroom wasn't the mechanism). An earlier
  attempt to fix this by throttling how often the code called
  `viewer.sync()` did not work, because the render thread runs
  independently of how often `sync()` is called. The actual fix was
  capping the render thread's own rate at the source: setting
  `vblank_mode=1` (a standard Mesa/GLX env var enabling vsync) before the
  viewer's GL context is created, in `mujoco_sim_node.py`.
- **Pipeline getting progressively worse across repeated restarts in the
  same session ("first run jitters, second is worse, third is
  catastrophic").** Root-caused to a real shutdown bug: `mujoco_sim_node`
  called `rclpy.shutdown()` unconditionally in its `finally` block, but
  rclpy's own SIGINT handler had already shut the context down before
  `rclpy.spin()` returned (that's *why* `spin()` returns at all on
  Ctrl+C) -- so that second call threw. Empirically, this was enough to
  occasionally crash the viewer's real-time-priority render thread during
  teardown (a segfault, confirmed via `ros2 launch` reporting exit code
  -11), leaving a thread stuck in an uninterruptible kernel wait that
  survives Ctrl+C entirely -- a zombie process still holding its ROS2/DDS
  participant and GL context, fighting the *next* launch's fresh set of
  nodes. Each additional zombie across repeated restarts compounds the
  problem, matching the reported "gets worse each time" pattern exactly.
  Fixed two ways: (1) guarded the redundant shutdown call
  (`if rclpy.ok(): rclpy.shutdown()`), which empirically also stopped the
  crash itself (a 3-cycle launch/SIGINT stress test reliably hit the hang
  before this fix, 5+ consecutive cycles were clean after); (2) added
  `scripts/run_demo.sh`, which force-kills any stragglers before every
  launch regardless, since the crash is in third-party C++ code
  (MuJoCo's/GLFW's viewer teardown) that can't be fully guaranteed
  against from here.
