# Implementation notes

The reasons and measurements behind specific pieces of code, too long to
keep in the source comments. One section per source file, in repository
order. For what the system does and how, see `system_overview.md`; for the
larger design decisions, `design_notes.md`; for open problems,
`../known_issues.md`.

## ocp.py

`core/ocp.py`

The formulation, weights and solver are described in `system_overview.md`,
section 5. Why dead time broke the controller is in `design_notes.md`,
"Real-time risk and mitigations".

**Per-stage goal pull (`w_ee_track`).** With the goal only in the terminal
constraint, nothing rewards progress at the early stages. From rest, the
cheapest plan eases into motion, and its first step (the only one applied
before re-planning) barely moves, since early displacement grows with dt².
Every tick re-plans nearly the same plan, and the arm stalls a few cm short
of the goal even though each plan reaches it at stage N. Changing the
terminal slack weight did not move the stall. A direct pull at every stage
fixes it.

**Why 5000 is stable.** At first, weights above ~100-300 oscillated in the
live loop (one tick of dead time through a relay node), and 5000 later
oscillated even without the relay while every solve reported success.
Dropping to 1000 gave margin but cost precision (~5 mm vs ~1 mm). The root
cause was that nothing bounded joint velocity: a strong tracking weight
swung the arm far faster than a real Panda, and large per-tick state changes
break the RTI linearisation under timing jitter. With the datasheet velocity
limits (`qdot_max`) as hard constraints, 5000 settled cleanly and precisely.

**Posture term (`w_q_center`).** A 7-joint arm reaching a point is
redundant, and SQP_RTI only refines locally, so it cannot jump to a
different family of configurations once in one. Without a posture term an
early aggressive step could lock onto a configuration drifting towards the
joint limits. At 5e-2 the redundant joints kept drifting after the tool had
converged (the arm visibly "rubbed" at the target), so it was raised to 1.

**Rebuild cache.** Code generation and compilation take a minute or two, so
the solver is reused when a fingerprint of its structure is unchanged. The
fingerprint includes this file's source bytes, because the expressions are
built in code; so any edit to `ocp.py`, even a comment, triggers one rebuild.

## heightmap.py

`perception/heightmap.py`

Depth to height, parallax correction, segmentation and the shadow check are
described in `system_overview.md`, section 6.

**Grasping tolerates outliers, placing does not.** `find_best_footprint` and
`find_topmost_boxes` take `inlier_frac`. Placement keeps the default 1.0:
a window that disagrees anywhere overlaps something, and accepting it would
drop a box on a neighbour's edge. Grasping uses 0.85: real depth has a few
bad cells at surface edges, and with a strict test one such cell vetoed an
otherwise good box top, splitting it into slivers and moving the grasp
off centre. A window straddling two different surfaces disagrees with its
median in about half its cells, so 0.85 still rejects it.

**The probe window.** A perspective camera sees an off-axis box's side walls
near its top edges, and they read as intermediate heights. Measured: a
0.06 m top read flat over only ~0.045 m. So the flatness window may be
smaller than the box, and `adjacency_cells` gives the true footprint for the
contact score.

**Contact score for placement** (`mode="lowest"`). Among the lowest
candidates, each side of the footprint scores `wall_bonus` if it is against
the edge of the map (a tray wall), or `neighbor_bonus` times the fraction of
the adjacent ring that is occupied. It is proportional to the contact
length: with a flat bonus for any contact, a spot grazing a neighbour's
corner scored like one flush along its whole side, and left slivers of floor
no box could use. "Occupied" means any box is there, not one of matching
height, which would also reward touching empty floor. `neighbor_bonus` is
twice `wall_bonus`, so a spot flush against a box and a wall beats a fresh
corner, and boxes cluster instead of scattering to the corners. Floor
placement now normally comes from `packing.py`; this search is the fallback.

**Speed.** The parallax correction and the window search are vectorised. As
per-point / per-window Python loops they cost ~30 ms and tens of ms per scan,
which stalled the plant at the time.

## mujoco_sim_node.py

`tasks/pick_and_place/common/pick_place_common/mujoco_sim_node.py`

The lockstep with the controller, the viewer's vsync cap and the shutdown
fix are in `design_notes.md`. The workspace camera's self-filter and hull
rendering are in "Workspace obstacle sensing and handling".

**Plant and model.** This node runs full MuJoCo physics (contact, armature,
damping). The controller's model (`core/dynamics.py`) is built from the same
`panda_robot.xml`, but through Pinocchio's MJCF importer, so the two need not
agree. The plant/model mismatch is real.

**Grasping with a weld, resting by contact.** Boxes, tray, arm and a tool
cylinder (the suction cup) have contacts; boxes rest on the floor, the tray
and each other by gravity and friction. The grasp is one shared `box_carry`
weld, repointed at whichever box is picked (`model.eq_obj2id` can be changed
after compiling) and switched off at release, when the box settles. It gets
its relative pose from the current poses first, so the box never snaps to a
stale offset. `pick_at` gives only a sensed pose, not a box
identity; the node resolves the nearest box in 3D (boxes stack, so z matters)
with a z tolerance that covers the tallest box's half-height, because the
target is the box's top and a body's position is its centre.

**What the wrist camera must not see.** The arm's visual meshes are hidden:
with the camera this close to the gripper, the gripper produced false heights
near the camera. The held box is hidden during a tray scan: it hangs between
the camera and the tray, and without the mask it made the cells under it read
full; in one run all 8 boxes went into 2 of 4 spots. Both are things a real
system knows from its own kinematics.

**Stale-command hold.** If the latest command came from a state more than
`COMMAND_STALE_STEPS` old (before the controller starts, or if it dies), the
plant holds position: gravity compensation plus a PD pull to where the hold
began. Zero torque would drop the arm, and gravity compensation alone lets
residual velocity coast.

## scene.py

`tasks/pick_and_place/common/pick_place_common/scene.py`

**Why the scene is its own module.** Everything two methods must agree on
to be solving the same problem lives here and is imported, never copied:
scene dimensions, where the cameras look, the heightmap resolution, what
counts as flat, and the scan vantage points. If each method kept its own
copy, the copies would drift, and a benchmark between methods would measure
the drift as much as the methods. What stays in a method package is how that
method moves: solver or planner settings, reference generation, cost
tuning. The plant node (`mujoco_sim_node`) also reads from here, so it never
imports from a method package.

**Scan resolution.** The footprint search returns a window aligned to grid
cells, so a box's position is quantized to within half a cell. Measured
against rendered depth, a 1 cm grid put the grasp 7 mm off the box centre,
which shows up as visible gaps once boxes are packed edge to edge. 5 mm
halves that for a few ms more per scan. Scans run on demand, not every
control tick, so the extra time does not matter next to the depth render.

**Grasp tolerance.** Grasping (`find_topmost_boxes`) accepts a top where 85%
of the cells agree; placement (`find_best_footprint`) stays strict. With
strict grasping, sensor noise split box tops into slivers and the grasp point
landed ~14 mm off centre on a 60 mm box. At 0.85 it landed on the centre.

**Pile detection thresholds.** `SOURCE_MIN_FOOTPRINT_CELLS = 30` sits
between sensor edge-noise blobs (9-12 cells) and the smallest box top
(0.06 x 0.06 m, ~144 cells). `SOURCE_MIN_FILL_FRAC` rejects L shapes and
scattered cells.

**Heights.** `LIFT_HEIGHT` (0.45 m) has to clear the tallest thing under the
carried box (two stacked boxes in the tray, ~0.26 m, or the pile top,
0.195 m) plus the box itself (up to 0.12 m). It is kept as low as that
allows, because the Panda's horizontal reach shrinks quickly as the wrist
rises: at 0.62 m the far side of the tray was out of reach. `SCAN_HEIGHT` is
5 cm higher so the wrist camera frames the whole pile. At 0.72 m the arm
could not reach the tray scan point with the box held level.

**Tray.** Given by its wall faces, floor top and wall top (a fixture); no
slots or spots are predefined, placement comes from the destination scan.
`DEST_FLOOR_Z` must be the real floor top, because the heightmap reports
empty cells at that height. The tray has contacts, so the arm and the boxes
really collide with it. `DEST_SCAN_BOUNDS` is not inset by the box size: the footprint
search already returns whole footprints inside the bounds, and insetting as
well left the first box 30 mm off the wall.

**Tray hull flag.** The tray's bounding sphere has its own on/off flag
(`Waypoint.dest_hull_active`). A hull must be off on any leg whose target is
inside it, or the QP is infeasible, so this one is on only on the way back
to `PARK_POSITION`.

**Choosing the tray position.** Whether the arm can settle at a tray corner
depends on that corner's distance and direction from the robot base, not on
the tray's own shape, so a tray position that works cannot simply be
translated. Earlier positions failed at the farthest corner in two ways:
- The arm settled 7 mm short while holding the gripper pointing down. That
  was just outside the settle tolerance at the time, so it held the box
  forever.
- Another position stopped 33 mm short. A position-only target converged to
  under 1 mm in 3 s, so the limit was reaching that corner with the gripper
  pointing down, not reach as such.

Moving the tray closer to the base reduced the error at every corner. Before
trusting a new tray position, run the farthest corner in a standalone
closed-loop test (acados and MuJoCo, no ROS) for ~30 simulated seconds and
check that it settles well inside `SETTLE_DIST_TOL_M`.

## depth_noise.py

Both simulated cameras (wrist and ceiling) pass their depth images through
`add_depth_noise` unless `mujoco_sim_node`'s `depth_noise` parameter is false
(`noise_seed` makes runs repeatable). Per-pixel Gaussian noise grows with depth
squared (about 0.7 mm at the wrist camera's 0.65 m, 6.6 mm at the ceiling
camera's 3.5 m), each frame gets a small common offset, and pixels go invalid
(0, as real depth cameras report no reading): half of those along a depth edge
and 0.2% at random. Consumers: the heightmap inference fills invalid pixels
from the nearest valid one first; `top_extent` counts only flat top pixels, so
invalid ones drop out; obstacle detection ignores them (a 0 would unproject to
the camera's own height). Measured offline: box footprints from the depth
pixels 0.7 mm mean / 1.5 mm worst with noise; 0 false obstacle blobs in 100
noisy frames of the empty cell.

## plant_mismatch.py

`mujoco_sim_node` changes the plant once at load unless its `plant_mismatch`
parameter is false (`mismatch_seed` picks the draw); the controller keeps the
nominal `panda_robot.xml`. Joint friction 0.5-1.5 Nm on joints 1-4 and
0.2-0.5 Nm on 5-7, damping +-30%, armature (rotor inertia) +-15%, link masses
and inertias +-10% with ~5 mm centre-of-mass shifts, and box masses 0.2-2 kg.
The armature range is narrower than the others on purpose: a datasheet value,
and the MPC chatters on joint 7 below ~0.75x (known_issues I8).

Compensation in the controller, as a real integration would have it: the
payload's weight is in the OCP's dynamics (`core/dynamics.py`: J^T of a point
5 cm below the TCP times -m g, the mass an online parameter), with the mass
taken from the wrist load cell (`mpc_controller_node`, filtered, 0 with an
empty hand); and offset-free tracking, a bias on the goal integrated only while
the whole reference stands still and the arm is within 1 cm and nearly at rest
(so never against a moving or blocked reference), clamped to 2 cm. With both,
the settle tolerance is back to 5 mm. Without them, seed 0 stalled on the first
1.78 kg box (8.2 mm sag at the tray scan point); with them, arrivals are within
0.3 mm median, 1.6 mm worst.

## task_node.py

`tasks/pick_and_place/mpc/pick_place_mpc/task_node.py`

Reference shaping (pass-through points, blending, cylindrical planning,
resume after a hold, the per-stage preview, the heading turning with the
base) and placement are described in `design_notes.md`, "Lockstep plant,
one clock for the task". Detours are in "Workspace obstacle sensing and
handling".

**What is sensed.** The node knows the pile's and the tray's footprints in
advance, but not what is in them. The wrist camera sees the pile only from
`PARK_POSITION` and the tray only from `DEST_SCAN_POSITION`, so every scan
follows a move there. A box's footprint comes from the depth image. Its
height cannot be seen from above, so the arm grasps at the sensed top and
reads the box's size from the sim once the gripper has closed on it
(`/sim/grasped_box_size`, like a contact reading or a barcode lookup on
contact). That is the only ground-truth read, and it never affects which
box is picked or where.

**Pick order.** `find_topmost_boxes` returns one region per box whose top is
visible. A box fully covered by another has no visible top, but one that is
only partly covered does, and is then lifted through the box on it
(`known_issues.md` H5). The largest footprint is picked first
(decreasing-size bin-packing heuristic, ties broken by height), so big boxes
go into the tray while it still has large free areas.

**Which hull is on, per leg.** A hull must be off on any leg whose target is
inside it. With the target in the keep-out, tracking and avoidance fight
every tick and the QP fails (ACADOS_MINSTEP, seen on both the approach and
the return legs).

| Leg | Pile hull (slot 0) | Tray hull (slot 1) |
|---|---|---|
| Approach, pick, lift | off | off |
| Pile to tray scan point | on | off |
| Release-above, place, lift-off | off | off |
| Tray back to `PARK_POSITION` | off | on |

The legs that start inside a hull are safe because the hull grows in behind
the arm (`_send_obstacle_params`).

**Held box and the static obstacle.** The OCP has no proxy for a carried box,
and a held box clipped an obstacle in a live run. The static obstacle's
radius is therefore grown by the box's half-diagonal, which has the same
effect on the margin as growing the proxy, without recompiling the OCP. This
applies only on the pile-to-tray leg. Applied for the whole time the box was
held, it left the arm a few cm from its place target with too little margin
for a clean SQP_RTI step, and it never settled.
