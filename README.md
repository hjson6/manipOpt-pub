# manipOpt

https://github.com/user-attachments/assets/13064bec-c554-4c52-b199-ff36c97a8bc3

Real-time obstacle-avoidance MPC for a manipulator, running as a
receding-horizon acados OCP fed by a MuJoCo simulation's live physics state
over ROS2 — one continuous control loop from start to goal, where an
obstacle appearing is a parameter update, not a mode switch.

Connects a background in direct-collocation trajectory optimization
(CasADi/IPOPT, and a generalized-alpha-integrator OCP pipeline for
constrained multibody systems) to a real-time robotics stack: ROS2, a
real-time QP-based solver, and a physics simulator standing in for the
plant.

Structured as `task / method`, not as one monolithic pipeline: a task
(currently one — container pick-and-place) is solved by one or more
methods that share the same scene, sensing, and plant, differing only in
how they actually move the arm. MPC is the main method; a second one
(MoveIt 2 + MoveIt Task Constructor) exists but is parked. The point of
the split is to make the second method an honest comparison against the
same task, not a rewrite.

## Why this design

Full rationale in [`docs/design_notes.md`](docs/design_notes.md):
why shooting + adjoint sensitivities (acados/SQP_RTI) replace direct
collocation here, why the manipulator's open-chain dynamics need no
generalized-alpha-style DAE integrator, why MuJoCo replaced the originally
planned Isaac Sim (WSL2's Vulkan/RTX support and this machine's VRAM both
ruled it out), how workspace obstacle sensing and handling works and where it stops short, and how
the collision-avoidance constraints are formulated (capsule/sphere proxies,
surface-to-surface margins, soft constraints).

## Architecture

**MuJoCo** is the plant, not a viewer: it computes the manipulator's actual
dynamics and collisions, and `mujoco_sim_node` (a hand-written bridge, not
a canned integration) publishes joint state (q, q_dot) over ROS2 and
renders the wrist-mounted depth camera on demand, taking joint torque
commands back in. It's method-agnostic — it has no idea an MPC is on the
other end of the topics it publishes to/subscribes from, which is what
lets a second method reuse it unchanged.

**Control stack** is a separate ROS2 process — the simulation node's only
job is stepping physics and moving data across the ROS2 boundary, so the
same controller/task nodes could point at a real robot's driver instead,
unchanged.

```
mujoco_sim_node (plant, method-agnostic)
  │  joint state, on-demand depth scans, workspace-camera detections
  │                                         │
  │                                obstacle_supervisor_node ──▶ /mpc/hold,
  │                                (track, classify, separation)  /mpc/speed_scale,
  ▼                                                               /mpc/static_obstacle
task_node       ──▶ sensed pick/place decisions, jerk-limited reference
  │                  (hold / speed scale / detour), /mpc/obstacle_params
  │
mpc_controller  ──▶ solves acados OCP every tick and applies u0 directly
  │                  (this same node also owns the decel-to-stop fallback --
  │                   see docs/design_notes.md "Real-time risk" for why keeping
  │  joint torque      the hot loop to one hop matters for real-hardware timing
  │  command            too, not just this simulation)
  ▼
mujoco_sim_node (plant)
```

See [`tasks/pick_and_place`](tasks/pick_and_place) for the node
implementations and [`core`](core) for the framework-agnostic
dynamics/collision/OCP code the controller node calls into. Note that
`core/dynamics.py` (Pinocchio/CasADi, used inside the OCP) and
`mujoco_sim_node.py` (MuJoCo, the actual plant) are independent models of
the same robot — the OCP's belief about the dynamics is not forced to
match what it's actually controlling, same as a real deployment.

## Repo layout

```
core/                              framework-agnostic: CasADi dynamics (via Pinocchio),
                                    collision-avoidance constraints, acados OCP builder
perception/                        framework-agnostic: dense-heightmap sensing/segmentation,
                                    workspace obstacle detection, tracking and classification,
                                    method-agnostic (any task's methods can use it)
tasks/pick_and_place/
  common/                          ROS2 pkg pick_place_common: the MuJoCo plant node,
                                    scene models, and scene/sensing constants every
                                    method must agree on to be solving the same task
  mpc/                             ROS2 pkg pick_place_mpc: this task's MPC method --
                                    task_node (sensed decisions) + mpc_controller
  moveit/                          ROS2 pkg pick_place_moveit (C++): the MoveIt/MTC method
                                    (parked) -- decision node, MTC executor, torque bridge
  interfaces/                      ROS2 pkg pick_place_interfaces: MoveTo/Pick/Place actions
scripts/                           launch scripts, env.sh; dev/ holds offline replay,
                                    batch runs and measurement tools
benchmarks/                        cross-method comparison harnesses and results (empty)
docs/system_overview.md            how the whole system works, end to end
docs/design_notes.md               design rationale (the "why" behind each choice)
docs/implementation_notes.md       per-module details and the reasons behind them
docs/setup.md                      local WSL setup: conda env, acados build, ROS2, MuJoCo assets
known_issues.md                    open problems, evidence and next steps
figures/                           plots and renders referenced from the docs
handover_notes/                    working notes and plans between sessions (not maintained docs)
```
The MoveIt method is a separate package alongside `mpc/`, importing the
same `common/` and `perception/` — not a copy of the scene or the sensing.

## Status

**Container pick-and-place, MPC method** (`bash scripts/container_pickplace.sh`) —
depalletizes a stacked pile of boxes of varying size, with no predefined
pick points, place points, or box dimensions. What's actually sensed
rather than hardcoded:
- **Where and what to pick**: a wrist-mounted depth camera renders a dense
  heightmap of the source pile on demand; it's segmented into discrete
  exposed boxes (connected-component clustering by height, not a search
  for one known footprint size), and the largest-footprint one is picked
  first — the standard decreasing-size bin-packing heuristic, since
  placing big items while destination space is still open avoids
  fragmenting it down to gaps only small items fit.
  A box is only picked once nothing rests on it.
- **What size the box is**: its footprint and the grasp's in-hand offset
  come from the pick scan's depth pixels. Its height isn't visible to a
  top-down camera before it's grasped (a real limitation, not simulated),
  so it's read from a force-sensed touch-down at the wrist, and never used
  to decide where or which box to pick, only how to place it.
- **Where to place it**: a depth scan of the destination finds a flat spot
  sized to the box actually in hand. Packing is compact, with an optional
  90 deg turn at the tray, and only picks spots the robot can actually
  finish. Every tray scan re-measures the boxes already placed; a blocked
  place lifts, rescans and re-plans.
- **Place-then-push**: the wrist can't set a box flush against a tray wall
  without hitting it, so wall spots are set down clear and pushed back with
  the tool tilted, the lean chosen from the scan. Push points are never
  predefined.
- **Physics, not scripting**: boxes and tray have contacts and rest by
  gravity; the only weld is the grasp (a suction-cup stand-in). Both
  cameras get depth noise.
- **Model mismatch**: the plant differs from the controller's model
  (joint friction, damping, armature, link and box masses; seeded). The
  OCP carries the payload using the mass from the wrist load cell, and a
  goal-bias integrator removes steady offsets once the reference stops.
- **Level carry**: the gripper points straight down for the whole cycle
  apart from pushes, carry legs included (measured: 0.1-0.6 deg of tilt).
  Long moves swing round the base on an arc, and the box turns with the
  base like on a palletizing robot, arriving square to the pallet.
- **Orientation**: a second gripper-axis constraint (not just "point
  straight down") keeps every grasped box's edges aligned with the
  destination tray's walls, so packing is meaningful rather than
  landing at an arbitrary yaw.
- **Collision avoidance** is phase-gated: a bounding-sphere obstacle
  around the remaining source stack activates only on the leg genuinely
  far from it (carrying a box toward the destination), and the
  destination tray gets its own independent obstacle on the return leg
  — active exactly when each is a real hazard, not uniformly.
- **Live decision dashboards**: two windows show what the wrist camera
  sees at each scan (RGB and depth) with the chosen pick footprint or
  placement spot outlined, drawn at its actually sensed height.

Brings everything up (viewer, acados codegen/compile if needed) and holds
there until you press ENTER, rather than lurching into motion mid-setup,
and recovers cleanly from solver failures or a slow tick via the
gravity-compensated position hold in `mujoco_sim_node`. See
`docs/setup.md` to bring up the local environment, then
`docs/design_notes.md` for the "why" behind what's here (including a full
debugging journal of every non-obvious problem hit getting here and how it
was fixed).

**Obstacle scenario, MPC method** (`ros2 launch pick_place_mpc demo_obstacle.launch.py`,
or `bash scripts/container_pickplace_obstacle.sh [args]`; args `obstacle:=visit|walk|static|none`,
`trigger`, `delay_s`, `speed`, `passes`, `pause_s`, `dwell_s`, `duration_s`, `static_position`) —
by default a person walks up beside the pallet to check on it, right where the
arm delivers, stands there, and leaves, while the cycle runs. The cell is a
half-scale model (the Panda is about half the size of a palletizing cobot
such as the UR20), so the person is 0.84 m tall and walks at 0.55 m/s. A fixed
ceiling depth camera (separate from the wrist camera) detects it; a supervisor
tracks it and decides: **moving means a person** (slow down, then stop and wait
until clear, never dodged), **still means an object** (a detour via-point is
planned and the object goes into the OCP's reserved obstacle slot, an online
parameter update, not a mode switch). Anything uncertain is treated as a
person. The controller never sees ground truth; a monitor node logs sensed vs
true pose, error and latency. There is no certified-scanner analog in the
simulation, so this is a software stand-in, not a safety function. Design,
measured results and limits: `docs/design_notes.md`; open problems:
[`known_issues.md`](known_issues.md).

**MoveIt method, parked** (`ros2 launch pick_place_moveit demo.launch.py`;
needs the `~/mtc_ws` overlay, see `docs/setup.md`) — MoveIt 2 + MoveIt Task
Constructor (C++), driven by a decision node that makes the same sensed
decisions as the MPC `task_node`, executed on the same MuJoCo plant through a
FollowJointTrajectory-to-torque bridge. It completed full autonomous runs on
an earlier big-box scene; on the current scene it has no push, touch-down or
tray re-measure, still reads the grasped box size from the simulator, and
hasn't been re-validated since contacts were switched on. Never run it at the
same time as the MPC launch: each starts its own plant. Pick-up notes:
`handover_notes/moveit_handover_note.md`.

**Next**: a baseline measurement of the MPC method under noise and model
mismatch (`handover_notes/realism_plan.md`, step 5), then the benchmarking
below.

## Results (to fill in after benchmarking)

- Solve-time distribution (ms) under MuJoCo-stepping CPU contention —
  planned: acados/SQP_RTI vs. the same OCP solved with IPOPT, same
  dynamics/constraints/horizon, to put a number on the design tradeoff
  in `docs/design_notes.md` rather than leaving it as prose
- Success rate over N trials, across obstacle position/timing variations
- MPC vs. the MoveIt method, side by side on the same task
  (`benchmarks/`) — packing quality, completion time, and reaction latency
  in the obstacle scenario
