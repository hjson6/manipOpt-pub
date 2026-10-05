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
At 1, joints 1, 3 and 5 (the self-motion that leaves the tool where it is)
still drifted up to 13.5 deg during a slow 3 s set-down, from the plant's
torque mismatch with nothing but the posture term to hold that direction, and
partly snapped back on lift-off; at 5 the worst drift was 3.9 deg (seed 2) with
the same tracking (p99 19 mm).

**Rebuild cache.** Code generation and compilation take a minute or two, so
the solver is reused when a fingerprint of its structure is unchanged. The
fingerprint includes this file's source bytes, because the expressions are
built in code; so any edit to `ocp.py`, even a comment, triggers one rebuild.

## heightmap.py

**Any camera pose.** `infer_heights_parallax_corrected` takes a cell's height
from the point on its pixel's ray at the measured depth (depth is along the
optical axis), not `cam_z - depth`; for a camera looking straight down the two
are the same. `top_extent` and the tray detection unproject every pixel.

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

**The room and the robot, composed at load.** `load_scene_model` reads
`room_scene.xml` (the room, the mobile base, the person) with the cell
(`cell_container.xml`, or the tall pile's file, `cell_file:=`) included inside a
`<frame>` at the cell's pose, and attaches `panda_robot.xml`'s link0 at the
base's `arm_mount` site with MjSpec (empty prefix: every joint, actuator and
site keeps its name; the arm's `home` keyframe gets the base's and the boxes'
declared poses for the rest). `panda_robot.xml` is unchanged: Pinocchio loads it
alone for the controller's model, and its MJCF parser takes no extra bodies. The
room's option block repeats the arm file's (`implicitfast`, `impratio` 10, 2 ms):
when attaching, the parent's options win. The loader checks the cell's and the
base's poses in the XML against `scene.CELL_POSE`/`BASE_PARK_POSE`.
`world="arm"` re-expresses the same scene with the parked arm's base at the
origin (the room in a frame at the cell pose's inverse): offline tools that
work in the arm frame (wrist camera, packing, lidar and layout checks, the
replay) use it unchanged; the plant always uses the room.

**What leaves the plant, in which frame.** Joint states, wheel encoders and the
IMU are in the sensors' own terms; the wrist load cell in the arm frame's axes
(its mount on the arm is the robot's own model); the lidars' ranges from their
sites on the moving chassis; the wrist camera's pose (not used by the method)
in the arm frame. `/sim/base_truth` (base_link's true pose in the room) is for
validation only: the person actor and the detection monitor use it, the method
never. `sim.csv` logs it per tick (`base_x` .. `base_qz`) with the wheel
commands and speeds; the person's pose there is in the room frame now.

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

**Wrist camera mount.** `container_cam` is a camera in the `attachment` body
of `panda_robot.xml` (Pinocchio's MJCF parser ignores it), so it turns with the
tool; the plant checks it against the calibrated mount in `scene.py`
(`CONTAINER_CAM_POS_TCP/ROT_TCP`) and renders the arm and the tool
(`tool_visual`). The mount: on the side tab of the wrist housing just above the
flange, 10 cm from the tool axis and 3 cm above the flange face (13 cm above the
TCP), looking straight down along the tool; the tool is out of its view. Found
offline (`scripts/dev/wrist_cam_check.py`, scratch sweeps over offset, height
and side):
- The camera has to be 0.5-0.7 m above what it scans to fit a zone in one
  picture, so every centimetre it sits above the TCP is reach the arm does not
  need. A camera level with the flange and on the tool's +-x sides put the first
  scans out of reach; raising it brings the tool into view (the plan's 7 cm
  off, 3 cm back, 12 deg in: the tool across the middle of the image, 10-17% of
  the scanned area behind it). A 15 cm x 15 cm bracket reached everything and
  ran live, but is an awkward arm.
- On the tab's side the arm leans less to put the camera over a zone: everyday
  scans have 18-20 cm of reach to spare. The two first scans (nothing measured
  yet) do not fit one picture: the first tray scan takes one picture per half of
  the place zone (`_halves`; the tray is found from both, `tray_detection` takes
  several frames), then a normal scan of the found tray; the pick zone's height
  limit is 0.35 m (was 0.40; the tall test pile is 0.35 m).
- With a box held, the box hangs beside the camera's line of sight and hides
  about 70% of the tray in a rescan (the old virtual camera: 93-97%, J1).

**What the wrist camera must not see.** The tool: `task_node` projects the
tool cylinder into the image once (rigid mount, so constant), dilated 3 px,
and zeroes those pixels in every frame, like a depth camera's dropouts; the
scan poses keep the scanned area above the mask's top row. With the tab mount
the mask is empty (kept for other mounts). The held box: the
method masks its own projection (task_node.py). Other arm links are seen at
the image edges; at the scan poses none reaches the scanned area. With the
heading turned 90 degrees (a box placed turned) the camera looked across the
robot's base, and the base's edge reached the tray area, so the tray scan
always uses the aligned heading; the leg to it is slowed so that turning back
keeps joint 7 under 1.5 rad/s (a turn back on this short leg at full speed
hit joint 7's speed limit before). Offline, rendered with and without the
robot at every scan pose, no heightmap cell differs by more than 5 mm.

**Benchmark telemetry.** With `MANIPOPT_TELEMETRY_DIR` set, `sim.csv` also
carries the person's mocap pose (`person_x/y/z/yaw`; z = -3 when absent) so the
true person-to-arm clearance can be computed offline from geometry
(`scripts/dev/results_summary.py`), and `mpc.csv` carries the solver's x1, the
goal bias and the payload estimate. The model-error metric does not use x1:
after one SQP_RTI iteration x1 is the linearised QP's prediction, not the
model's, so the summary integrates the controller's own ODE one tick from the
measured state with the torque the plant applied. `plant_mismatch:=false`
(launch argument) gives the oracle runs.

**Joint and load-cell noise.** With `sensor_noise` (default on, drawn from
`noise_seed` on its own stream, so depth noise draws are unchanged), joint
positions get 5e-5 rad Gaussian noise (encoder class), and joint velocities are
the finite difference of two noisy encoder reads 8 ms apart (the last 4 physics
substeps; about 9e-3 rad/s noise, 4 ms lag), as a driver computes them at its
servo rate, not MuJoCo's qvel. The difference over the whole 20 ms tick (10 ms
lag) makes the MPC chatter on joint 7 at the torque limit from the first second
(live, seed 2); offline replay (`replay_offline.py --enc-noise 5e-5 --fd-vel
--fd-substeps N`): noise alone changes nothing (hf 0.039 vs 0.038 rad/s),
windows of 2-12 ms are smooth (hf 0.08-0.12, torque step p99 4.5-8.6 Nm vs 2.3),
14 ms starts to shake and 16 ms and more chatters (hf 0.26, 15 Nm; 20 ms: 2.8,
45 Nm). Like I8, joint 7 has little margin: a real integration needs a velocity
estimate that lags less than about 6 ms. The wrist force gets
0.2 N per axis plus a per-run bias of up to 0.5 N per axis (tare drift). The
published force goes into `sim.csv` (`fx/fy/fz`), so a run with the noise off can
be replayed with noise added (`scripts/dev/force_noise_check.py`).

**Stale-command hold.** If the latest command came from a state more than
`COMMAND_STALE_STEPS` old (before the controller starts, or if it dies), the
plant holds position: gravity compensation plus a PD pull to where the hold
began. Zero torque would drop the arm, and gravity compensation alone lets
residual velocity coast.

**Real-time stepping (default; `lockstep:=true` restores the old behaviour).**
A thread steps the plant every 20 ms of wall time. The command computed from
state k starts 14 ms (7 substeps) into the next tick if it has arrived, else the
previous torque runs on and the tick counts as late (`late` and `cmd_ms`, the
command's arrival after its state went out, in `sim.csv`; a summary every 30 s
and at shutdown). The state message's `effort` is the torque in effect, so the
controller knows what ran. Everything else that touches the simulation takes a
lock; camera renders work on a copy of the state (`_snapshot`) and stay on the
executor thread (their GL contexts belong to it); commands arrive on their own
node and thread (`mujoco_sim_commands`), so they never wait behind a render.
- Why 14 ms: the controller answers about 5 ms after a state (solve) but its
  command reached the plant after 8-10 ms (ROS and Python on both sides; p99
  9.4 ms once commands had their own thread, 18 ms before, behind the 18 ms
  overhead-camera frames). At 10 ms most ticks were late; a late command then
  takes effect a tick later than the controller planned (30 ms in effect), and
  joint 7 went unstable (21 rad/s). At 14 ms: about 0.05% of ticks late.
- Why not a whole tick: offline (`replay_offline.py --delay --delay-substeps N
  --predict --ekf`), a 20 ms delay with the EKF left the joints twice as jittery
  (hf 0.18 vs 0.08 rad/s); 14 ms: hf 0.066, jerk p99 359 (lockstep 858), torque
  step p99 3.3 Nm (5.1), path error 2.9 mm RMS at the best time shift (2.3).
- The windows. At first they shared the plant's process, and with them open
  18-30% of commands came late (headless 0.05%) and the arm wobbled into a false
  "box on an edge" abort. Three causes, all fixed: the drawing shared the
  plant's Python process (now sim_sensors_node); OpenCV's worker threads span
  between calls in the window process, ~14 cores busy (now one thread); Qt6 on
  WSLg's Wayland took ~2-3 cores per window (now X11, `QT_QPA_PLATFORM=xcb`).
  With every window open: 4 late ticks in 6206, plant step p99 3.3 ms, solve
  p99 4.8 ms. All rendering is on the CPU (WSL's OpenGL is llvmpipe);
  `GALLIUM_DRIVER=d3d12` uses the GPU but reading frames back through D3D12
  made the overhead camera 4x slower. The sensors node caps llvmpipe at 2
  threads (`LP_NUM_THREADS`).

## sim_sensors_node.py

`tasks/pick_and_place/common/pick_place_common/sim_sensors_node.py`

The plant's cameras and windows in their own process: the wrist camera's scans,
the overhead camera and its obstacle detection, the decision and obstacle
windows, and the MuJoCo viewer. It keeps its own copy of the scene from
`/sim/scene_state` (every tick: step, held box, sphere radius, qpos, mocap poses),
rendered from when a scan is asked for or a frame is due. The model is loaded
the same way (with the same tray shift); masses and the arm's mismatch do not
matter for rendering. A scan renders the latest scene state (at most a tick old;
the arm is at rest at a scan).

The safety lidars are ray cast here too (`lidar_sim.py`, `mj_multiRay`, 0.2 ms
for both), on the scene-state thread with its own copy of the scene, so camera
renders on the main executor never delay a scan. Since the mobile base, each
scan starts at its `lidar_*` site on the chassis and fans out in the site's x-y
plane (it tilts with the base), skipping the chassis's own body; the scan
message still carries the calibrated mount (`LIDAR_SCANNERS`), which the
plant checks against the sites at start. The scan has no detection, as a
driver's would not: `/env/lidar_scan` carries raw ranges (nan: no return).

The scene state comes in on its own node and thread: on the main executor the
camera renders held it up for 60-350 ms at a time, and the viewer, redrawing a
stale pose, played like a low-frame-rate video. The viewer now redraws every
20 ms on a fixed clock, one frame per plant tick.

## lidar_detection.py

**Two scanners on the base's free corners.** The plan put them front-left and
rear-right; in the cell the base's (+x, +y) and (-x, -y) corners are under the pick
and place tables, whose legs would cut the view. They sit on the other two with their
optical centres on the chassis's corners, so each 270 deg sector runs exactly along
both faces: together they see all round the outline and nothing of the robot (the
chassis is left out of the rays; no beam enters it). Two other mounts were tried:
3 cm outside both faces, the 4 cm housings scraped the pick table's legs
while docking; recessed into the corners, the chassis would hide both sides from each
scanner (180 deg each instead of 270) and, through the chassis in the model, the
beams found the drive wheels (4220 in 20 drives). **No minimum range**: like safety
scanners, which measure from their front window, the model returns anything from 0 m;
with a 5 cm minimum a leg against a scanner's housing was invisible to both, and once
the base crept within 50 mm of one. Offline, in the cell: of 933 grid poses out to 5 m,
1 was hidden (behind a table leg), all others detected; no false person in 200 empty
scans.

**Background per beam.** The median of the first 20 scans of the empty cell
(beams with returns in fewer than half: none). A beam is foreground when it
returns 8 cm (4 sigma of the 2 cm noise) nearer. Single foreground beams are
dropped (MIN_POINTS 2): 0 false positives in 100 empty scans per scanner.

**Leg centre.** A leg shows the scanner an arc; its points' centroid lies about
2/pi of the radius in front of the leg's centre, so the centroid is moved away
from the scanner by 2/pi of the arc's half width. Legs seen by both scanners
(within 10 cm) are averaged; legs within 0.5 m are one person, centred between
them. Offline centre error: median 24 mm, p95 120 mm (one leg hidden behind the
other: the centre is that leg's).

**Radius.** The detection's radius holds the legs seen (about 0.15 m), not the
person's arm span: the supervisor adds the ISO 13855 allowance (step 6).

## obstacle_supervisor_node.py and camera_detection_node.py (people sensing)

**Sources.** `lidar:=` (default on) and `ceiling_camera:=` (default off) pick
what the supervisor reads. One tracker per source; every track of every source
is checked each frame and the most cautious result counts, so with both on the
two never need associating (a person seen by both is checked twice). A track's
age is the time since its source's last capture (at least that source's usual
latency), plus one frame period and two control ticks. Any enabled source silent
for 0.5 s, or never heard from (the lidar learning its background at start),
holds.

**Lidar person.** Radius = the legs' radius + ISO 13855's C = 1.2 m - 0.4 H
(H = 0.20 m: 1.12 m), top 2.0 m above the floor, always a person: the lidar
cannot tell a static object, so the detour planner (`obstacle:=static`) needs the
camera and the launch refuses it without.

**Kinematic self-filter.** The ceiling camera's detection left the plant: the
plant publishes raw depth (`/env/workspace_depth`) and `camera_detection_node`
masks the robot with the method's own robot model at the measured joint angles
nearest the capture time (each collision hull's vertices projected, their convex
outline filled), plus the held box from its sensed size and in-hand offset
(`/task/held_box`), grown 3 px (about 6 cm on the floor). Offline over 43
recorded poses it covered every robot pixel the plant's segmentation marked
(about 39 extra pixels per frame, 3.6 ms). No segmentation ids are used.

**Reliable depth.** The 0.6 MB depth frames went best effort at first: most
were lost in fragments (2 Hz arrived of 10) and the supervisor held for "camera
down" over and over. Keep-last-1 reliable delivers all of them.

## state_estimator.py

`core/state_estimator.py`, used by `mpc_controller_node` when not in lockstep.

The controller solves from the state 14 ms ahead (when its torque will start),
predicted with its own model and the torque the plant reports running. Predicted
from the measured state, the joints shook (offline hf 1.35 rad/s): the driver's
velocity (8 ms difference of noisy encoders) is noisy and late, and predicting
ahead amplifies both. `DelayEKF` estimates the velocity from the joint positions
alone (the driver's velocity comes from the same encoder reads): the mean is
predicted with the full model over the split tick (old torque 14 ms, new torque
6 ms), the covariance with q' = q + dt qdot (the model's Jacobian made an update
7 ms; the simple form 0.3 ms, same result offline). Settings, from offline
replays and headless runs:
- Position noise assumed 5e-4 rad (encoders: 5e-5): trusting the model more is
  smoother; at 5e-5 the wrist jittered at rest (0.03 rad/s) and never settled.
- Velocity process noise 0.03 rad/s per tick: at 0.3 the arm jittered at rest
  (joints 2 and 6) and legs timed out; a per-joint mix did not help.
- The solve starts from the measured q with the estimated velocity: the
  friction-free model runs ahead of an arm held back by a box (a push) or by
  friction at rest; positions are measured well. Settling and the goal bias use
  the measured state.
- Tried and dropped: a torque-disturbance state (random walk 0.005-0.5 Nm per
  tick): no better at best, shaking when fast.
- The goals are those for the next step (the torque starts 14 ms after its
  state): without that the arm followed the path 20 ms late (time-aligned path
  error 6.2 mm RMS, 3.0 mm median; with it 4.1 / 2.8 mm; lockstep 5.8 / 1.3 mm).

## scene.py

`tasks/pick_and_place/common/pick_place_common/scene.py`

**Two frames (mobile manipulator, step 1).** MuJoCo's world is now the room
(7.0 x 5.6 m now, 10 x 8 m before; origin at the south-west corner,
floor z = 0); the robot is a mobile
base with the arm on a pedestal. Everything the method uses for the arm task is
in the arm's base frame (link0): with the base parked at the cell, that is the
old cell's frame, so every task constant kept its value. `CELL_POSE` is where
the cell stands in the room ((3.0, 2.8), yaw 90 deg; (5.0, 3.5), 180 deg in the larger
room; deliberately not the
identity, so a forgotten frame change shows up as a wrong position, not as
nothing); `BASE_PARK_POSE` is base_link's pose there, derived from the arm's
mount on the chassis (`BASE_ARM_MOUNT`: 0.20 m ahead of the wheel axle, 0.75 m
up, turned 90 deg). The chassis's footprint in the arm frame (`CHASSIS_BOUNDS`)
replaces the old 0.60 x 0.50 m base box as the robot's own furniture; the
lidars' calibration is given in base_link (`LIDAR_MOUNTS_BASE`, the chassis's
front-left and rear-right corners) and derived in the arm frame
(`LIDAR_SCANNERS`), which came out within 5 cm of the old mounts. Why the arm
sits forward and turned: the old tables are 5 cm from the chassis on two sides,
so only a chassis that extends behind the arm, away from both tables, fits
(0.80 x 0.56 m, MiR-like).

**Two layouts.** `layout:=cell` (the arm task, step 1): the parked cell in the middle
of the room. `layout:=stations` (the mobile job, from step 3): the cell's two tables
apart, each a station with its load (`station_pick*.xml`, `station_place.xml`, in the
cell's coordinates) at its own pose in the room (`PICK_STATION_POSE`,
`PLACE_STATION_POSE`): the pile's table against the west wall, the tray's against the
east wall, 5 cm off, and the robot starts at home by the south wall
(`BASE_HOME_POSE`), where its map is anchored. Each table keeps its pose relative to
the arm docked there, so the arm task's zones hold at each station; the pick table is
docked 6 cm farther off than in the cell (`PICK_DOCK_OFFSET_M`): its legs pass 9.5 cm
beside the chassis and 5.5 cm beside the corner scanner on the way in. At 3 cm the
legs' noisy returns (2 cm) reached into the docking field 17-45 times per approach.
The arm still reaches 98% of the pile's footprint with a 3 cm margin (99% at 3 cm,
`dock_reach_check.py`).

**The cell's layout.** In the arm frame the tables have their tops at z = 0
(`FLOOR_Z`, where the zones are), the room floor is at z = -0.75
(`ROOM_FLOOR_Z`). The furniture's footprints are taught like the zones (`*_BOUNDS`); obstacle
detection ignores them up to 3 cm above their tops (depth noise at 3.5 m pushed
top pixels above z = 0: ~2 false blobs per frame of the empty cell before). The
overhead camera stays 3.5 m above the room floor. The person is full size (1.75
m, 6 cm legs, 0.6 m across the arms, 1.2 m/s); the tracker's "person-tall" rule
is 1.0 m above the floor (was 0.5 m, half scale) and the supervisor assumes 1.6
m/s (ISO 13855, was 0.8). The visit stand is at the tray table's corner, clear
of the table and the base by 3 cm; the walk goes along y = -1.2, south of the
tray table. Offline (`scripts/dev/layout_check.py`): the arm stays 38 mm or more
from the base and tables over a recorded run's poses; 0 blobs in 20 noisy frames
of the empty cell; the person seen along both paths (13-20 mm at the stand and
on the walk, up to 14 cm at the image's edge, where only part of them is in
view).

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
between sensor edge-noise blobs (9-12 cells) and the smallest top the box
spec allows (0.05 x 0.05 m, 100 cells). `SOURCE_MIN_FILL_FRAC` rejects L
shapes and scattered cells.

**What is given, what is sensed (`handover_notes/sensed_cell_plan.md`).**
Given, as an integrator would teach it at installation: the two zones (a
0.5 x 0.5 m pick zone and a place zone of the nominal tray +- 8 cm, each with
the highest thing it takes), the box spec (sides 0.05-0.16 m, heights
0.07-0.13 m), camera calibration, margins. Nothing in this file is read from
the XML pile: the pile top for the pile scan's parallax guess and the ceiling
camera's mask comes from the method's last pile scan (`/task/pile_top`;
the zone's maximum before the first scan), and pads use the spec's largest
box. The zone edges moved the 5 mm grid by 3.75 mm against the old
pile-fitted bounds, which changed the pick order on seed 2 (`known_issues.md`
J2).

**Heights.** `LIFT_HEIGHT`/`SCAN_HEIGHT` (0.45/0.50 m) are nominal values the
MoveIt method still uses; the MPC method computes its heights from each scan
(task_node.py below). The reach limit that set them is real: the Panda's
horizontal reach shrinks quickly as the wrist rises (the far tray corner is
out of reach at 0.51 m with the tool down), so the MPC method checks every
pose with IK instead.

**Tray.** `DEST_TRAY_*` is the nominal tray: the plant's geoms, the MoveIt
method and the person's script use it. The MPC method finds the tray in the
place zone at its first tray scan (tray_detection.py) and uses only that; the
plant can shift it (`tray_seed`). The tray's heightmap is not inset by the
box size: the footprint search already returns whole footprints inside the
bounds, and insetting as well left the first box 30 mm off the wall.

**Tray hull flag.** The tray's bounding sphere has its own on/off flag
(`Waypoint.dest_hull_active`). A hull must be off on any leg whose target is
inside it, or the QP is infeasible, so this one is on only on the way back
to the pile scan.

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

## frames.py

`tasks/pick_and_place/common/pick_place_common/frames.py`

Plant and offline side only: the arm base's true pose (link0 in MuJoCo's world)
and the room/arm conversions the plant needs to hand the method what a real
sensor or driver would give it in the robot's own frame: the wrist load cell's
axes, the wrist camera's pose in the depth message (the method computes its own
from the joint angles anyway), the box-overlap log. `person_in_arm` turns a
recorded person pose (room frame since step 1) into the arm frame with the base
pose logged next to it in `sim.csv`, for `person_clearance.py` and
`results_summary.py`; rows from older runs (no base columns) are used as they
are.

## base_drive.py

`tasks/pick_and_place/common/pick_place_common/base_drive.py`

The base's motor controllers and sensor drivers, plant side.

**Speed loop in the plant, every substep.** The wheels are torque motors with
the geared motor's rotor inertia as the joint's armature (0.1 kg m^2 at the
wheel); a P speed loop (40 Nm per rad/s, 30 Nm limit) runs before every 2 ms
physics substep, as a motor controller's inner loop would. A MuJoCo `velocity`
actuator was tried first: under the `implicitfast` integrator its damping is
integrated implicitly while the contact solver does not see it, and the chassis
outran its wheels by 23 mm over a 0.5 m/s^2 start, giving it back when braking
(Euler and RK4 rolled exactly; switching the arm's integrator was not an
option). With the explicit loop the chassis and the wheel arc agree to 0.1 mm
over the same start. The speed setpoints are set once per tick (the commanded
speeds, or the standstill hold below); a missing command for 0.2 s stops the
wheels (the driver's watchdog).

**Standstill: brake and static friction.** Commanded zero for 0.2 s, the
drive brakes: the wheels' angles are held with 10000 Nm/rad (damped, within the
30 Nm limit), the stiffness of a motor brake through the gearbox, and each
wheel's contact point is pinned to the floor (a `connect` equality anchored where
the wheel touches down). The pin is static friction, which MuJoCo's soft
contacts do not have: it lets go when its horizontal force would exceed the
friction coefficient times the wheel's normal load (the wheel slips, and sticks
again where it stops) or when the drive moves. Why both: in the step-1 gate the
parked base turned 0.09 deg (plain) and 0.33 deg (with a person) over a run, in
0.06-0.09 deg steps at each swing of joint 1 (up to 20 Nm of reaction); the
lidars then saw the place table's front leg and grazing beams along the far
walls move against their learnt background, and one of five holds had no person
behind it. Offline (`arm_on_moving_base`-style swings under the MPC, the seed-2
wheel mismatch) the parked base ratcheted 0.0045 deg per swing: the soft hold let
the wheels roll and the contacts micro-slipped under the cyclic load. A stiffer
hold alone halved it; pins alone did not help (the wheels rolled about their
pinned contact points); both together: the base twists 0.01-0.02 deg and springs
back, no ratchet over 40 s of swings, 0 mm and 0.004 deg under a steady 30 N push.

**Encoders and IMU.** Encoder angles at 16384 counts per wheel revolution (no
noise beyond the quantization: encoders do not drift; slip and the wheel-radius
mismatch are the errors that matter). The gyro gives the mean rate over the last
tick (a delta-angle output: the rotation between the IMU's orientations at the
two ticks, divided by 20 ms) plus a bias drawn once per run (up to 0.005 rad/s,
0.3 deg/s) and 0.002 rad/s of noise; the accelerometer reads MuJoCo's
`accelerometer` sensor with 0.03 m/s^2 noise. Both are stamped with the plant's
step, like the joint states, so the method integrates over exact 20 ms ticks.

**The base in the scene (`room_scene.xml`).** Chassis 110 kg (the pedestal
included; 132 kg with the arm), drive wheels r 0.10 m on a 0.50 m track at the
chassis's centre, four ball casters (r 0.04 m, ball joints) at the corners. What
it took for believable rolling:
- The drive wheels are sprung (slide joint, 40 kN/m, about 400 N preload each,
  damped): on a rigid chassis resting on six points the wheels lost contact in
  turns (turns in place reached 26 deg of a commanded 69). Real AMRs spring
  their drive wheels for the same reason.
- The wheels are one body (the suspension) away from the chassis, so MuJoCo's
  parent-child contact exclusion no longer covers them: without an explicit
  `exclude` the wheel brushed the chassis box intermittently, which looked like
  a flickering floor contact and a 12-17 deg yaw on some starts.
- A rounded tread (an ellipsoid, 0.10 x 0.025 x 0.10 m): one contact point, no
  scrub in turns. A flat cylinder (two contacts across its width) made
  wheel-only odometry's heading 4-8% wrong per turn.
- Wheel on floor as explicit contact pairs with a near-hard impedance
  (`solimp` 0.999 0.9999): MuJoCo's soft friction let a parked wheel creep
  sideways under a steady load (1.2 mm and 0.25 deg per 5 s at 30 N on the arm
  mount; in a live run a 4.5 s push into the tray turned the parked base 3 deg
  and moved it 13 mm, and every placed box then appeared shifted by ~9 mm).
  With the pairs: 0 mm sideways, 0.035 deg; the brake and the pins above did
  the rest.
  `noslip_iterations` did the same, but it also makes the arm's joint friction
  perfectly sticky (0.5 Nm against 1 Nm of friction: 0.016 rad of creep in 5 s
  without it, none with it), which left the arm 10.6 mm short of the pile's scan
  point, outside the settle window; the arm's plant must not change in step 1.
- Wheel contact `solref` 0.005 (stiffer than the default 0.02).
The parked base settles with the arm's mount at 0.74999 m and 0.001 deg of
tilt.

## base_odometry.py

`core/base_odometry.py`

Distance from the encoders (mean of the two wheel arcs), heading from the gyro,
integrated at the arc's midpoint heading. Encoder-only heading is the classic
weak point of a differential drive (it is the difference of two nearly equal
arcs, divided by the track, so 0.5% of radius mismatch is degrees per loop);
the gyro's own error is its bias, so the bias is learnt whenever both encoders
have not moved for 0.3 s (running mean over up to 500 samples). Taught: the
nominal wheel radius and track (`scene.BASE_WHEEL_*`); the plant's wheels are
0.5% off in radius each and 5 mm off in track (`plant_mismatch.py`).

Offline (`scripts/dev/odom_check.py`, seeds 0-4, the base's own drives and
sensors, arm held): out 3 m and back, a 2 m square with turns in place, a 1 m
circle, 0.5 m/s, 0.5 m/s^2:

| | wheels + gyro | wheels only |
|---|---|---|
| end position error | 0-9 mm (0.00-0.11% of the distance) | 0-157 mm (up to 2%) |
| worst error along the way | 3-12 mm | |
| heading error at the end | 0.01-0.43 deg | 0-6.3 deg |
| gyro bias learnt standing still | within 0.22 mrad/s of the true bias | |

`figures/base/odom_check.png`. Real AMR odometry is worse (1-2% is typical);
the plant has no floor unevenness or tyre deformation, so this is optimistic,
which SLAM (step 2) has to cope with either way.

## tray_detection.py

`perception/tray_detection.py`

**From the depth pixels, not the heightmap.** The walls are 1 cm thick. In
the parallax-corrected heightmap, cells on a wall seen at an angle re-project
onto its inner face and do not converge to the top in three passes, so two
of the four walls read low and the rim search found none of them. Detection
unprojects the depth pixels instead: the rim is the highest surface (90th
percentile of the points above the floor), and a histogram of rim points
along x and along y (5 mm bins, summed over three since a wall's points
split over two) has the walls as its two densest bands. Refinement takes,
per 5 mm along each wall (world coordinates, so any camera heading), the rim
pixel nearest the inside, the median over the wall away from the corners,
plus half a pixel.

**Accuracy** (`scripts/dev/tray_detect_check.py`, 50 random shifts within
+-5 cm, one noisy frame each, the arm at the first tray scan pose): found in
50/50 from the two half-zone pictures, wall faces 1.04 mm mean, 1.85 mm worst
(the virtual mount before: 1.25 mean, 2.3 worst), all measured inside the true face (edge pixels drop out half
the time, so the extreme rim pixel is often one further in); wall top 2.1 mm
worst. The inward bias makes the tray read slightly small, and
`PLACE_CLEARANCE_M` (5 mm) covers it. Tray rotation is not sensed (assumed
squared by guide rails).

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
pixels 0.7 mm mean / 1.5 mm worst with noise (virtual mount); with the camera
on the arm (`footprint_accuracy.py --noise`, 6 noise draws) 1.0 mm mean,
2.0-2.7 mm worst, about one pixel (2.5 mm at the pile scan). The worst case is
edge quantization, not the mount: moving the scan height by 1-4 cm takes one
box from +2.2 to -0.4 or -0.7 mm. Two tops of the same height 7.5 mm apart
(cbox_0 and cbox_4 once cbox_6 is gone) merge in the heightmap in most noise
draws at some camera heights (the floor seen through the gap is about one
pixel, and edge dropouts fill it), but the footprint from the depth pixels,
the flat patch connected to the top's centre, is right in 10 of 10 (152 x 120
mm for 150 x 120). `PLACE_CLEARANCE_M` (5 mm)
covers it. 0 false obstacle blobs in 100 noisy frames of the empty cell.

## plant_mismatch.py

`mujoco_sim_node` changes the plant once at load (`mismatch_seed` picks the
draw); the controller keeps the nominal `panda_robot.xml`. The base's draw
(`apply_base_mismatch`: each wheel's radius +-0.5%, the track +-5 mm per side,
the chassis's mass +-10%) comes from its own random stream (`[seed, 2]`), so
the arm's and the boxes' draws of a seed are those of the earlier runs; the
wheel's mount is moved with its radius so the chassis stays level. With the
`plant_mismatch` parameter false the arm stays nominal but the boxes still get
the seed's masses (the arm draws are made and discarded), so an oracle run is
the same task as the mismatch run of that seed. Joint friction 0.5-1.5 Nm on joints 1-4 and
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

**What is sensed.** The node knows the two zones and the box spec, not what
is in them. A box's footprint comes from the pile scan's depth pixels, its
in-hand offset from the sensed top and the arm's pose at the grasp, its
height from the wrist load at touch-down. The tray is found at the first
tray scan (tray_detection.py); every tray value after that (walls, floor,
wall top, heightmap grid, hull sphere, push heights) comes from that `Tray`.

**Touch rules with a noisy load cell.** The sideways load that aborts a place
("the box is on something's edge", 1 N plus friction on the load taken) is taken
relative to the filtered reading at the hover, and both touch rules use the mean
of the last 3 readings. Replaying the noise-free seed-2 run with the plant's
noise (200 draws of noise and bias, 8 touch legs): the untared rule fired
falsely on 372 of 1600 legs, because a 0.5 N bias on both axes alone is 0.7 N
of the 1 N limit. Tared, 1 of 1600 (plain noise crossing 1 N); tared and
averaged, 0, also with every leg rescaled to the lightest spec box (0.2 kg),
where the unaveraged contact rule fired falsely on 3 of 1600. The average costs
about one tick (0.4 mm of descent at 2 cm/s). During carries the raw sideways
load exceeds 1 N on about a quarter of the ticks (acceleration), which is why
the rule is only armed on the touch leg.

**Floor packing: touching perimeter.** The packing (`packing.py`, plan_compact)
ranks the flush spots by: most of the upcoming boxes (up to 3) still fitting, most
of this box's perimeter touching walls or placed boxes, nearest flush, smallest
group rectangle, nearest the tray corner farthest from the robot. It used to rank
by the group rectangle first; boxes then lined up with each other and not the
walls (one 24 mm off the back wall). Offline over 200 random pick orders with a
wrist model: all 8 on the floor 76% (was 29%); looking ahead at 0 boxes: 38%;
filling from the near corner: 58%. The robot's side (`_floor_spot_end`) says
where the box really ends up: flush, where it is left when a push back is not
possible (scored again there), or nowhere; rejections are logged by reason
(scan, overhang, robot). Before, a spot the robot could not reach flush was
dropped, and when all four corners were (a 76 x 105 mm box first) the box went to
any free floor spot.

**The arm-wall check leaves out the flange.** ARM_CHECK_FRAMES stops at link7;
the tool and link7's housing are checked by _push_clearance, with their real
extents (the wrist camera is not a collision part). The flange as a 7 cm ball read
-32 mm to the front wall with the tool straight down over a box against that
wall, where the tool and wrist cleared it by 13 mm: every push along a wall was
refused. The corners on the robot's side still refuse pushes towards the robot's
wall: there the forearm hangs over it.

**Stacking when the floor is full.** The packing rule fills the floor; when it
finds no spot, `heightmap.find_resting_footprint` takes the lowest spot where the
box rests stably, trying it both ways round: it rests on the highest cells under
it, those within 1 cm of that height carry it, and it must be carried on at least
half its footprint and on at least 15% of each quarter round its centre (so it
cannot tip). A gap between two boxes of the same height and a small overhang are
fine; an unknown cell (a held-box rescan's hidden area) rules a spot out. The old
fallback wanted the whole footprint flat within 1 cm, so a 150 x 120 mm box could
not go on a 135 x 135 mm one or across two boxes, and runs stopped with the tray
"full" (I10). A stacked box is not slid flush to a neighbour afterwards (that
could take it off its support). The stable spots are tried in that order (up to
20, 1 cm apart) and the first the robot can place into wins: set down and pushed
back without the wrist or the arm hitting anything (`_place_options`, as the
packing rule's spots). Without that check a small box got a floor spot in a
corner where the wrist could not reach: the set-down shift away from the wall
was blocked by a neighbour, it was placed flush, and link7 pressed on the wall
(three tries, 45-65 N, then a stop).

**Push tilt: straight down if it fits.** A push first tries the tool straight
down; if the tool, the wrist or the arm would not fit beside the box, the
smallest tilt that fits (5-40 deg in 5 deg steps, either way in the plane of the
pushed face); if none fits, the push is skipped and the gap left. "Fits": the
tool and link7 against the walls and the scan, and the whole arm against the
tray walls: IK for the pose (seeded from the measured joint angles, as the MPC
moves from there), then points from link3 to the flange and between them as
7 cm spheres, 2 cm clear of the wall slabs. Before, a push took whichever of
straight, or 30 deg either way, gave the tool and the wrist the most room, so it
tilted even when straight fitted, and the forearm was not checked: on the tall
pile, link5 pressed on the tray's east wall for 12 s at up to 156 N during a
30 deg push. Offline, the new check reads -15 mm at that pose (MuJoCo: link5
0.2 mm into the wall); the tall pile then ran 8/8 with no arm-wall contact (a
push at 5 deg, re-planned at 15 deg after touch-down). The MPC may still choose
another elbow posture than the check's IK (the arm is redundant); the margin
covers small differences.

**Stopping above a spot near a wall.** The point above the spot is passed through
(the path rounds the corner and starts down early) unless the box's half-diagonal
plus 2 cm reaches a wall from the spot; then the arm stops above it first. A big
box turned 90 deg for a spot 5 mm from the east wall started down while still
over the wall and caught on its outside (390 steps, 97 N; the controller then
failed to solve and the run stalled).

**Clear of overhangs.** A spot keeps PLACE_CLEARANCE_M from every placed box
whose bottom is above the spot's surface and below the held box's top: a stacked
box can overhang its base, and the scan check ignores one cell round the
footprint. A 60 mm box once went 4.5 mm (planned) from the corner of a 135 mm box
stacked on a 120 mm one; its top met the overhang's underside and three
set-downs failed. Boxes standing on or below the surface are left to the scan
check: applying the rule to them too left a 135 mm box no spot at all.

**The heading turns with the base, over a run of legs.** joint7 ~ joint1 -
heading - 135 deg, so a heading in step with the base keeps joint 7 still. The
heading used to be set per leg, towards the aligned heading at that leg's end,
eased with a smoothstep: on a carry the zone-exit leg kept the heading while the
base swung 60 deg, then the next leg turned it 180 deg; joint 7 went -7, -70, +9
deg (2.4 rad of back and forth, every carry, in all 10 runs of a batch), and
the smoothstep alone made a single leg's wrist overshoot by about 25 deg. Now a
run of pass-through legs up to the next stop turns as one, towards the aligned
heading at that stop, shared out by each leg's base swing and linear in it, so
joint 7 moves one way at a steady rate (-7 to +9 deg). A turn more than twice
the swing follows the distance instead (an 83 deg turn on a 2 deg swing, rising
to the tray scan: in step with the base it spun the wrist, and under the rate cap
it crawled at 2 cm/s for 16 s). The smoothstep stays only for a turn by distance
from rest to rest.

**Turn rate cap.** Every leg is slowed so the tool turns (heading and tilt) no
faster than HEADING_TURN_RATE_MAX. A 2-3 mm push after a set-down started a few
cm away with the heading changed by about 80 deg and the tool tilted 10 deg: the
wrist spun 78 deg in 0.5 s, the joints reached 4.8 rad/s and the TCP left its
reference by 96 mm beside the box just placed (3 of 10 runs).

**No push under 6 mm.** PUSH_TRY_M went from 2 mm to PUSH_MIN_M (6 mm): a gap
that small is already accepted when a push is not possible, and it is about the
spacing between boxes; the short pushes were the ones that spun the wrist.

**Picking by contact.** The pick used to stop at the sensed top and the plant
welded whichever box was nearest the requested point: a pick that stopped short
still "gripped". Now the plant grips only the box the tool is on, and the
method hovers 15 mm above the sensed top (the load cell tares), lowers at 3 cm/s
aiming 20 mm below it, and grips when the wrist load changes by 3 N (load-cell
noise 0.2 N, bias up to 0.5 N); after the lift it needs 2 N of weight in the
wrist. No contact by the end of the leg, or an empty lift: back above the pile
and scan again (3 times in a row, then stop).

**Which way to turn the wrist (joint 7).** Each leg's heading goal is one of
psi + 2 pi k: the one nearest to turning with the base whose joint 7 at the goal
(from the measured joint 7, plus the swing, minus the heading turn) stays within
150 deg. Nearest alone once tied at 180 deg on a carry's last leg and picked joint
7 at -244 deg: the wrist stopped at its -166 deg limit, the arm reached back over
its shoulder instead (joint 2 at -101 deg) and the controller stopped solving. A
box turned 90 deg can be set down at either of two headings: the one reached
without turning the wrist more than 90 deg past the base's swing is preferred.

**Guarded pushes.** A push aims 2 cm past its planned end and stops when the
box does: the reference at least 8 mm ahead of the TCP, the TCP advancing less
than 5 mm/s over 0.2 s, and the wrist feeling the push (tared at the start of
the push leg; above 2 N), for 3 ticks, and only once the TCP is past the planned
end; or at 40 N. Then the reference freezes, the tool backs off 1 cm from where
it stopped and lifts. With the real-time plant the push to a precise end fell
short (the box stopped 19-26 mm from the wall, lockstep 6-11 mm): the model
thinks the arm moves on while the box holds it back. Stopping on "not advancing"
before the planned end also stopped a heavy box (1.8 kg) that had not started
sliding at 20 N, or was creeping in stick-slip. Headless, seed 2: real time
4.3 mm to the wall at 18 N; lockstep 0.0 mm at 31-40 N (the stiffer lockstep arm
reaches the cap). The `pushed` report comes after the back-off (pressing, the box
leans on what stopped it). The wall-clearance check still uses the planned end. The reference is not reset to the tool when the box stops: it had run up to 31
mm ahead, and resetting it let the pressing arm spring back in one tick (3.5
rad/s); the back-off leg now starts from where the reference is, and the force
eases off.

**Task order (empty-hand tray scans).** The tray is scanned before the first
pick and again after each place, with the hand empty. The spot for the next
box is chosen at its grasp (its footprint is known from the pick scan) from
the last tray heightmap, and the arm carries straight to it. Scanning with
the box held put the box right under the camera; the simulator used to hide
it. The node builds the tray heightmap from the depth frame itself.

**Held-box rescans.** A blocked place, or no spot in the last scan, still
needs a scan with the box held. The box is projected into the image (sensed
footprint and offset, measured height or the spec's tallest, pose from the
kinematics, grown 1 cm) and its pixels made invalid; a cell whose line of
sight from the floor to the wall top crosses them is unknown (NaN) unless it
reads back its own height elsewhere. The camera is only 0.25 m above the box
top, so the box hides 93% of the tray (49% for the smallest box). Taken
literally ("hidden is unknown, never free"), the rescan could never find a
spot. Hidden cells therefore keep the last empty-hand scan's heights, except
round the blocked set-down (+-2 cm), which stay unknown: nothing else in the
tray moves while the robot holds the box, and the thing that blocked it is
under that footprint (`known_issues.md` J1). NaN makes every search window
over it invalid and a push pose over it a hit.

**Heights from the scans.** Constants are margins only:
- approach a pick: pile top + 5 cm;
- lift: the rest of the pile (the pile scan outside the picked top) + the
  spec's tallest box + 3 cm + `BLEND_M` (the swing starts 5 cm below a lift
  point), and at least the tray entry height;
- enter the tray: tray top (wall tops, last scan) + held box height (the
  spec's tallest until touch-down) + 3 cm; lift-off: tray top or the placed
  box top + 5 cm;
- carry: the pile's height until the swing leaves the pick zone's azimuth
  range by the held box's half-diagonal + 5 cm (one pass-through point), then
  the tray's. The plan said the higher of the two all the way, but with a
  0.35 m stack that is 0.56 m, and the far half of the tray is out of reach
  there; the space between the zones is assumed free;
- scan poses: the lowest TCP height at which the 55 deg camera sees the whole
  zone (the tray once found) at its sensed top, beyond the depth camera's
  0.30 m near limit, and at least pile top + 5 cm (the way back); a held-box
  scan also clears the tray's contents and the pile;
- the tray scan's parallax reference is half the last tray top.
Every tool-down pose is checked with damped least-squares IK (seeds: the
current joints and home turned to the target's azimuth, joint limits less
0.02 rad) before it is used; out of reach stops the task. Test scene:
`panda_scene_container_tall.xml` (`scene_file:=`): with the fixed 0.45 m
lift, the first box carried hit the 0.35 m tower (13 N, the top box knocked
off 0.3 m); with these heights, no contact.

**Wrist at a pick.** link7 hangs 0.10-0.155 m above the TCP and up to 88 mm
sideways. Next to a taller neighbour (the tall scene's tower beside cbox_1)
it hit the neighbour at 150 N going down. A top is skipped in the pick order
while any pile cell under link7's footprint (+2 cm sideways) is higher than
the top + 0.10 m - 8 mm. The vertical margin is the tray's wrist margin, not
2 cm: the baseline picked cbox_1 beside a 0.195 m top with 1 cm to spare, 0
link contacts in 60 runs.

**Heading after a turned placement.** A box placed turned 90 deg leaves the
tool 90 deg off its aligned heading. Turned back over the lift-off (5-12 cm
now), joint 7 hit its speed limit (2.66 rad/s, with joints 1, 3, 5 near
2 rad/s and a 45 mm tracking error). The heading is now kept through the
lift-off and the tray scan and turns back over the swing to the pile. (With
the old 0.45 m lift-off it reached 2.66 rad/s too, less often.)

**Empty pile.** "Container empty" needs the pile scan to show nothing above
half the spec's lowest box. With the new grid and the lower scan, a 60 mm
stacked top failed detection in 1.5% of noisy frames and the run parked with
two boxes left; such a scan is repeated up to 3 times.

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
| Carry from the pile (zone exit, else release-above) | on | off |
| Release-above, place, lift-off, tray scan | off | off |
| Tray scan back to the pile scan | off | on |

The legs that start inside a hull are safe because the hull grows in behind
the arm (`_send_obstacle_params`).

**Held box and the static obstacle.** The OCP has no proxy for a carried box,
and a held box clipped an obstacle in a live run. The static obstacle's
radius is therefore grown by the box's half-diagonal, which has the same
effect on the margin as growing the proxy, without recompiling the OCP. This
applies only on the pile-to-tray leg. Applied for the whole time the box was
held, it left the arm a few cm from its place target with too little margin
for a clean SQP_RTI step, and it never settled.

## base_node.py

`tasks/pick_and_place/mpc/pick_place_mpc/base_node.py`

The base's driver on the method side: odometry (base_odometry.py) on the wheel
encoders and the gyro, paired by the plant's step stamp, published as `/odom`
and the `odom > base_link` transform; body velocity commands (`/base/cmd_vel`,
clipped to 1 m/s and 1.5 rad/s) turned into wheel speeds with the taught radius
and track, scaled together (same curvature) under 15 rad/s, sent every tick; no
command for 0.25 s means zero speed. In step 1 nothing sends `cmd_vel`, so the
base stays parked (the drives hold the wheels) and the odometry reads still.

## slam

`slam/` (framework-free: numpy, scipy), `pick_place_mpc/slam_node.py`,
`scan_merger_node.py`, `localization_monitor_node.py`, `config/slam_toolbox_*.yaml`.
Two SLAM options run on the same input (`slam:=own|toolbox`); ours is the default.

**Input.** Both lidars' raw scans with their taught mounts on base_link
(`scene.LIDAR_MOUNTS_BASE`) and the base's odometry. Ours takes the two scanners'
points directly (and each point's scanner, for tracing free space);
slam_toolbox takes one 360 deg scan (`/scan`, 720 beams, frame base_scan), the nearest
point per 0.5 deg seen from base_link's origin (`scan.merge_scan`,
`scan_merger_node`), as a MiR merges its scanners. The plant stamps each scan with the
time of the state it ray-cast (`/sim/scene_state` carries it): stamped with the
sensors process's clock instead, a scan was up to a tick older than its stamp.

**Scan matching** (`icp.py`): point-to-line ICP (Censi's PL-ICP), Gauss-Newton with
Huber weights (5 cm), the match distance shrinking from 0.5 to 0.08 m. Target normals
by PCA over each point's neighbours; a point counts as on a line if the smaller
spread is under 0.15 of the larger and its neighbours are within 25 cm. Scans and
the mapper's local maps are thinned to one point per 5 cm cell first: twelve
keyframes of 2 cm range noise made each wall a dense band in which six neighbours
no longer looked like a line, and matching fell back to odometry on 1779 of 1855
scans. A saved map's walls are about two cells thick, so its targets use twelve
neighbours.

**Mapper** (`mapper.py`): each scan is matched against the last twelve keyframes
from the odometry's prediction (kept if the fitness, the share of points within 5
cm of a line, is under 0.35 or the correction over 0.3 m). A keyframe every 0.3 m or
15 deg joins the pose graph with a 2 cm / 1 deg step edge. Loop closures: the three
nearest keyframes at least 25 keyframes back and within 3 m are each matched (a
local map of seven keyframes around them); accepted at fitness 0.55, RMS 3.5 cm and
a jump under 1 m, then the graph is optimized (`pose_graph.py`, SE(2) Gauss-Newton
on sparse matrices, the first pose fixed). CPU per scan 3-4 ms median, 7.6 ms mean,
p95 about 40 ms (the optimizations; Python).

**Map** (`grid.py`): log-odds (+0.85 hit, -0.4 free, clamped), free space traced
along each ray from its own scanner, 5 cm cells, saved in the map_server format
(PGM + YAML) so either system can read it. Each cell also keeps the mean of the hits
that fell in it (`.npz` next to the PGM): a wall's position is then not quantized
to the cells. Occupied cell centres lie about half a cell behind the surfaces
(the cell in front is mostly traced free), so the walls of a perfect-pose map read
16-30 mm outward; the evaluation compares with that reference map. Building a map
from 187 keyframes takes 3.8 s, so the live node draws each keyframe once for
display and rebuilds properly only when saving (`/slam/save_map`).

**Localization** (`localizer.py`): each scan matched by the same ICP against the
map's occupied cells' mean hit positions, from the odometry's prediction. A first
version matched the scan against a distance field of the occupied cells: right on
average but with 2 cm of scatter, since each wall was only representable at cell
faces.

**Live node** (`slam_node.py`): odometry interpolated to each scan's time; a scan
that arrives before its tick's odometry waits for it (taking the previous tick's
odometry gave 1 deg yaw errors during turns in place). Publishes map > odom.

**slam_toolbox**: the package's shipped parameter files (online sync for mapping,
localization) with base_link, a 10 m range, scans added every 0.2 m or 0.2 rad and
no interactive mode. Fed live by `scan_merger_node`, offline by
`scripts/dev/toolbox_replay.py` (the recording played in simulated time).

**Evaluation** (`scripts/dev/record_drive.py`, `slam_eval.py`, `slam_compare.py`):
recorded commissioning drives (two laps of the room, a figure-eight across it;
quiet or with four people walking, one standing), the true pose kept for scoring
only. The map frame is anchored at the dock, so the truth is moved there.
Offline (seeds 1-4):

| | own | slam_toolbox |
|---|---|---|
| mapping: position error p50 / max | 2-3 / 10-19 mm | 14-19 / 31-41 mm |
| mapping: walls against the room | within 30 mm (the grid's half cell) | within 33 mm |
| localizing (start 7 cm / 2 deg off): p50 / max | 2 / 9-12 mm | 12-13 / 22-32 mm |
| yaw, max | 0.17 deg | 0.43 deg |
| CPU per scan, mapping | 7.6 ms mean | 8-13 ms |
| CPU per scan, localizing | 2.5 ms | 18-24 ms |
| people in the map | traced out (free space) | a few traces kept |

Live (headless, real time, three people walking): mapping own 2/6/12 mm
(p50/p95/max), slam_toolbox 17/31/34 mm; localizing own 3/6/11 mm, slam_toolbox
11/28/36 mm; yaw within 0.31 deg; the live maps' walls within 20 mm (own) and 29 mm
(slam_toolbox) of the room's. `figures/slam/compare.png`.

## map_people.py

`perception/map_people.py`, `pick_place_mpc/people_node.py`, `people_monitor_node.py`

People seen from the moving base. The per-beam background of `lidar_detection.py`
needs a still scanner; on the move everything is foreground. Instead each beam's end
point is put in the map frame with the localized pose (map > odom from whichever
SLAM runs, times the odometry at the scan's time) and is foreground if it lies more
than 15 cm from every mapped surface (the map's occupied cells' mean hit positions,
a KD-tree). The leg and pairing rules are `lidar_detection.py`'s, now shared
functions (`legs_from_scan`, `group_people`): runs of foreground beams along each
scan with at least two beams are legs, legs within 0.5 m are one person. 15 cm is
well above the localization error (a few mm) and the walls' scatter in the map, and
below a leg's distance from anything it stands next to; a 3 cm / 0.5 deg pose error
leaves the walls background (unit test). An object that was not in the map (a cart
left in the aisle) counts as a person, as anything the lidars see did before:
uncertain means person.

Offline (`scripts/dev/people_eval.py`, two recorded drives in the stations room,
six people: looping, crossing without giving way, standing, standing then walking,
a slow diagonal walker, one near the place station): of the people within 5 m that
the lidars saw (3 beams or more on them), 99.3-99.6% had a confirmed track within
0.5 m in the same scan, each person at least 98%, and no one was missed for more
than 5 scans (0.33 s) in a row; position error 22-23 mm median (119-122 mm p95,
mostly one leg seen), speed error 0.03-0.04 m/s median for walking people; no false
person within 1.4 m of a wall, table or fixture (2 single-scan ones in the open in
one drive, none in the other). Live (headless, the crossing route, six people): 97.8%
of the people within 5 m tracked (occlusion is not known live, so a lower bound), 23
mm median error, no false tracks. `figures/people/`.

## people_tracker.py

`perception/people_tracker.py`

A constant-velocity Kalman filter per person in the map frame (2 m/s^2 of
acceleration noise, 5 cm measurements), detections assigned by optimal matching
within 0.6 m of each prediction, confirmed after three matches (0.2 s at 15 Hz),
standing after 1 s under 0.2 m/s. A track coasts through occlusions for up to 1 s,
except that it is dropped at once if the lidars saw through its predicted spot
(`MapPeopleDetector.seen_empty`: a beam at that bearing reached 30 cm past it):
without that, a person turning a corner left a ghost track walking on in the old
direction for a second (12-13 false people per drive, all of this kind; none after).
The old `obstacle_tracking.py` (alpha-beta, base frame) stays for the parked cell's
supervisor.

## planner.py

`nav/planner.py`

Routes on the saved map for the differential-drive base. **Costmap**: the distance
from each 5 cm cell to the nearest occupied or unknown cell (a Euclidean distance
transform). The chassis (0.80 x 0.56 m, the arm tucked inside) is three circles of
0.31 m along its axis (-0.27, 0, 0.27 m); a pose is free if every circle clears the
nearest blocked cell by 8 cm.

**Hybrid A\*** over (x, y, heading) binned at 0.1 m and 5 deg, with the base's own
moves: 0.2 m forward, straight or on a 0.8 m radius arc either way; 0.2 m straight
back (three times the cost); 15 deg turns in place (0.15 m each). Clearance below
0.35 m costs extra, so routes keep to the middle of the free space. Each change of
steering (straight, left, right, turn, back) between moves costs 0.3 m: without it the
pick-to-place route on open floor changed its turn direction four times, and the base
weaved along it; with it, none. The heuristic is a 2D Dijkstra from the goal over the
cells where the middle circle fits (it knows the walls and tables), never below the
straight-line distance. From any node within 4 m of the goal (otherwise every 50
expansions) a rotate-straight-rotate shot to the goal is tried and taken if free.
Within 5 cm of the goal the shot is the turn alone, and up to 0.5 m it may back to the
goal: a shot from a node 3 cm beside the goal otherwise turned 90 deg to face it,
crept 3 cm and turned back. Routes go from pre-dock pose to pre-dock pose (0.9 m
before a dock): the dock itself lies within the table's clearance and would read as a
collision. Planning takes 0.1-0.7 s for the room's legs (Python).

## base_mpc.py

`nav/base_mpc.py`

**Model and limits.** Unicycle kinematics with the accelerations as inputs (x, y,
heading, v, omega; a, alpha), 20 stages of 0.1 s, SQP_RTI with HPIPM, as the arm's
controller. Speed -0.15 to 0.5 m/s, turn rate 1 rad/s, acceleration -1 to 0.5 m/s^2,
1.5 rad/s^2, and each wheel's rim under 0.8 m/s (|v| + 0.25 |omega|). 2.2 ms per
solve live (p99 2.8 ms). The speed, turn, acceleration and rim limits are set on the
built solver, so a top speed setting needs no rebuild: they follow nav_node's `speed`
(see Speed below; the mobile job's default is 1.0 m/s).

**Tracking cost.** The reference (`reference.py`) is the route resampled every 2 cm /
2 deg with a time profile (0.5 m/s, slower on arcs, 0.6 rad/s turns in place, braking
to the end) and is sampled over the horizon from the robot's progress, which only
moves forward. The position error is split along the reference's heading at each
stage (it is a stage parameter): lag along the route weighs 5, lateral error 40 (the
heading 4, speed 2, turn rate 1, the inputs 0.5 and 1; the terminal stage double).
With one weight for both, slowing down for a person cost as much as stepping
sideways, and the base weaved through the crowd (22 left-right-left swerves in 20
drives; 9 with the split, 3-8 with the speed limit below). Where the reference stops
or turns in place the lag weighs as much as the lateral error, so stops land on the
spot: with the cheap lag, routes into the pick station stopped 2 cm past the turn
point, which the 90 deg turn made a lateral error, and four drives in seven
realigned.

**Reversing** only where the reference reverses (undocking, backing out, short
shots), or creeping back at up to 5 cm/s onto a stop or a turn in place; otherwise the
speed bound only lets it brake out of reversing (v0 + a_max t). Allowed freely, the MPC
backed away from people walking at it, toward whoever was behind.

**People.** Each of the six nearest tracks is predicted at constant velocity (up to 3
s) and kept at its radius plus the robot's (0.45 + 0.45 m between centres) by a soft
constraint per stage (L1 300, L2 3000), so the QP stays feasible when someone steps
close. The radius is capped at the distance the robot would keep standing still (from
its current position, less 2 cm): it never closes in on anyone, and gains nothing by
fleeing. With the full radius always, it turned 50 deg away on the spot from someone
walking at it, then back. **Speed near people**: the speed bound falls from 0.5 m/s
(nearest person's centre 1.8 m away) to 0.2 m/s (0.9 m), reached braking at full
deceleration if need be; this halved the swerves left. Above 0.5 m/s it rises on to
the top speed at 3 m.

## safety.py

`nav/safety.py`, `pick_place_mpc/safety_node.py`

The base's safety layer, separate from the planner and the MPC, in the manner of ISO
3691-4 / ANSI/RIA R15.08 protective and warning fields (a software stand-in, not a
certified function). **The protective field follows the motion**: it is the chassis
swept along its current arc for as long as the stop takes (0.15 s of reaction at the
current speeds, then braking the chassis's fastest point, at most |v| + 0.49 |omega|,
at 1 m/s^2), sampled every centimetre of that point's travel, plus a 10 cm margin. A
point there stops the base if the robot is getting closer to it: something beside the
chassis does not stop it driving past, nor anything it turns away from. Driving
straight this is a box ahead as long as the stopping distance; turning, the stretch of
their circles the corners cover before they stop. Standing still there is none.
Warning caps beyond it: the speed is capped to what would stop 0.5 m short of the nearest
point in the corridor ahead (the chassis's width, wider when turning), the turn in
place to what would stop the corners 0.15 m short of the nearest point round them;
never below 0.3 m/s and 0.5 rad/s, where the protective field decides. (They were flat
boxes: 0.3 m/s anywhere within 1 m beyond the stop, 0.3 rad/s within 0.3 m beyond the
corners' stop; at 1 m/s a flat box held the base at 0.3 m/s for whole corridors, and
turns at the stations at 0.3 rad/s.) Docking switches the
margin to 2 cm, so the table 5 cm ahead is not an obstacle at 0.1 m/s. A check of both
scans takes under 0.3 ms.

The field is checked for the command after the warning caps and for the measured
motion (odometry): a stop commanded while the base still rolls keeps the field of the
speed it has. The first version had a box ahead (wider when turning) and, for a turn in
place, the whole circle the corners sweep. With the box also at a standstill, a person
at the front corner held the base even from turning away, and they waited for each
other for good. The circle stopped any turn near a table leg however slow: setting off
from standstill beside the pick table (the first command 0.05 m/s, a turn in place by
the old rule) the base stopped for good, the leg 15 cm off its corner but inside the
circle. The swept stop covers what the base can actually reach. It is tighter than the
old widened box ahead of a turn: once, a person walking at 1.2 m/s into the front
corner of a slowly turning base came within 34 mm (it was already stopping; no
contact).

**People get 10 cm more.** The scanners see legs 0.2 m above the floor; a person's
torso and arms overhang their legs by up to 0.135 m (the person model's arms side-on),
and the folded arm and a carried box reach to within 4-8 cm of the chassis's edge (the
elbow over the front, the box over the rear deck). With the 10 cm margin the elbow
came within 72 mm of a person's upper body live (known_issues Q1). Scan points within
0.15 m of the people detector's foreground (the scan against the map, which
`people_node` also publishes in the base frame, `/perception/foreground_base`) get a
margin of 0.20 m; everything else keeps 0.10 m. A wider margin for every point would
stop the base beside the pick table's legs (9.5 cm off the chassis on the way into
the dock), the live-lock the swept stop removed. The foreground may be a round (67
ms) behind the scan; the 0.15 m match covers a person and the base closing at up to
2.2 m/s over it. In 20 drives through a crowd with the arm in the carry pose and the
largest box held (`nav_sim.py --arm carry --box`): the upper body's closest approach
while the robot closed on someone went from 129-168 mm to 219-307 mm, the legs' from
38-120 mm to 131-141 mm (body_clearance.py measures the arm, the box and the people's
whole bodies on the truth); the navigation's own results did not change (the table
below is with it).

**safety_node** sits between `nav_node` (`/nav/cmd_vel`) and `base_node`
(`/base/cmd_vel`) and checks each command and each round of both lidars' raw scans (in
base_link with the taught mounts), with the docking margins while `/nav/status` says
docking. Watchdogs stop the base when the scans (0.25 s), the odometry (0.1 s), the
people tracks (0.5 s) or the localization (0.5 s: our SLAM's pose, otherwise the map
> odom transform) are late; a command older than 0.3 s is zero. Live, past the
start-up (0.6 s until every input is in), faults came only with stalls of the plant
itself (WSL, up to 0.5 s; none in the final runs).

## navigator.py

`nav/navigator.py`, `pick_place_mpc/nav_node.py`, `mobile_scenario_node.py`
(`route:=nav`), `nav_monitor_node.py`, `pick_place_common/nav_scoring.py`,
`scripts/dev/nav_sim.py`

**Goals by name**: pick and place (the taught dock poses, `scene.PICK_DOCK_BASE` and
`PLACE_DOCK_BASE`, moved to the map frame anchored at home) and home. **Sequence**:
undock (straight back on its own heading to 0.9 m, at 0.3 m/s) > route (Hybrid A\* to the pre-dock
pose, followed by the MPC) > align (turn, straight, turn onto the pre-dock pose,
backwards if it lies behind; until within 2 cm of the dock's axis and 1 deg, up to
three times) > approach (straight along the dock's axis at 0.3 m/s, the last 0.25 m at
0.1 m/s) > docked if within
2 cm along and across and 1.5 deg, else back out straight and align again (twice) >
failed. Home: route, align (2 cm, 1 deg), parked. A goal after a failed docking
backs out first: planning from inside the docking corridor found no free start.

**Docking moves leave people to the safety layer**: in undock, align, approach and
back-out the MPC has no people constraints (stop and wait, no swerving by the table).
Before, a person standing beside the corridor pushed the approach 4-8 cm sideways
and the docking failed three times. **Blocked**: no progress along the route for 4 s
with someone standing in the way: plan again with the standing people as 0.45 m
discs, each shrunk where needed so the robot's own pose stays free (otherwise the
plan found no start and the robot waited for good). **Stepping back**: blocked again on
such a route, someone standing within 1 m ahead, the base backs straight off 0.4 m
(0.2 m/s) and holds there, planning again every second (the discs now also keep the
pose it stepped back from free, so the route round them found there is found again).
At 1 m/s the base crept up to someone standing near home until their legs were 0.2 m
off its front corner; the route round them began with a turn towards them that the
safety layer refused, and it planned the same route every 4 s for good. Only after a
route round them: someone standing on the goal itself (the docking line's person) gives
no such route, and the base waits on its route as before (stepping back there made it
shuttle back and forth, and the person waiting for it never left).

**nav_node** runs the navigator at 10 Hz on the localized pose (the SLAM's map > odom
times the odometry) and the people tracks; publishes `/nav/cmd_vel`, `/nav/status`
("state goal docking|normal") and `/nav/plan` (the first plan's length and turning,
for the scoring). `mobile_scenario_node` with `route:=nav` sends the goals in turn.
**Planning in the background**: in `nav_node` the Hybrid A\* runs in a worker thread.
A goal's first route is waited for standing still (state "planning", a few ticks); a
new route round people standing in the way is made while the base drives on the
current one. Planning inside the tick held the 10 Hz up for up to 142 ms live; in a
thread the ticks stayed within 109 ms of each other, the longest step 38 ms. A plan
that comes back once the base has gone on to align or dock is dropped: at 1 m/s a
route round someone, asked for on the way, came back during the approach and replaced
it with a route starting 0.65 m behind the base, which as a route (the table ahead an
obstacle, not a dock) the safety layer stopped for good. `nav_sim.py` plans in line,
so its runs repeat exactly (and it never saw that).

**Evaluation** (`nav_sim.py`, no ROS): the plant with the seed's mismatch and the
crowd, and the method's chain at its rates (odometry, localization and people at the
lidars' 15 Hz, the navigator at 10 Hz, the safety layer), 20 drives pick > place >
home in turn; scored on the truth (`nav_scoring.py`, also the live
`nav_monitor_node`): the distance from the chassis to each person's legs (the person
model's two legs, 6 cm radius), legs inside the protective field of the base's actual
motion for over 0.3 s while it moves (an intrusion), contacts, and the motion as
`motion_check.py` measures the arm's: the detour (distance over the first plan's),
the extra turning (the true heading's travel beyond the plan's), wiggles (a turn
reversed and reversed again within 2 s: one swerve), oscillations (wiggles in a
row), turn-backs (turning in place one way, then back) and realignments at a dock.

| 20 drives each | quiet room | crowd of 6, seed 0 | crowd of 6, seed 1 |
|---|---|---|---|
| arrived | 20/20 | 20/20 | 20/20 |
| docked: lateral / along / yaw, max | 7 / 4 mm / 0.30 deg | 6 / 5 mm / 0.26 deg | 5 / 5 mm / 0.31 deg |
| parked at home, max | 3 / 4 mm / 1.03 deg | 3 / 3 mm / 1.02 deg | 3 / 2 mm / 0.99 deg |
| protective-field intrusions, contacts | 0, 0 | 0, 0 | 0, 0 |
| closest person while closing on them | - | 34 mm (walking at it) | 116 mm |
| detour max | 1.01 | 1.12 | 1.07 |
| extra turning max | +11 deg | +491 deg | +304 deg |
| wiggles (largest swing), oscillations, turn-backs | 0, 0, 0 | 7 (18 deg), 0, 0 | 10 (10 deg), 1, 0 |
| realignments at a dock | 0 | 0 | 0 |
| safety stops on the way into the pick dock | 0 | 0 | 0 |
| lidar beams on the robot itself | 0 | 0 | 0 |

The extra turning in the crowd is avoiding: turning out round people and back, and
the new plans round people who stand in the way (the first plan is the reference);
in the quiet room it stays at +11 deg. Every close approach to someone standing kept
109 mm or more; the 34 mm was a person walking at 1.2 m/s into the front corner of a
slowly turning base that was already stopping. Live (headless, real time, own SLAM
localizing in the stations map, the arm tucked, pick > place > home): quiet 3/3,
docked within 6 mm and 0.17 deg, home within 3 mm and 0.85 deg, extra turning at most
+9 deg, no stop on the way into the pick dock; with the crowd 3/3, docked within 6 mm
and 0.11 deg, home within 3 mm and 0.87 deg, closest person 145 mm, no intrusions or
contacts, one swerve (16 deg), no oscillations or turn-backs; localization 4/8/13 mm
(p50/p95/max). The navigation step takes 2.2 ms median, 3.1 ms p99, at most 7.5 ms
(planning in its thread), the ticks at most 106 ms apart. The arm task in the cell with
the scanners on the corners: 8/8 placed, every hold with a person.
`figures/nav/nav_sim.png` (seed 0's paths).

**Crowd model** (`mobile_scenarios.py`, plant side): someone blocked by the robot's
body steps 0.9 m aside after 3 s and walks on round it; someone waiting for another
person walks on past them after 2 s, for 1.5 s. Without the second rule two people
waiting for each other crept one tick every 2 s, the blocked counter of one of them
kept restarting, and robot and person faced each other for minutes. Someone whose next
spot the robot stands on (within 0.6 m of its centre) goes on to the spot after it; if
that is where they stand already (a path of two spots), to a spot just clear of the
robot beside the taken one instead (they had stood on for good, the robot waiting for
them).

## The mobile job (task_node.py, mobile:=true)

`pick_place_mpc/task_node.py`, `launch/mobile_job.launch.py`, `people_node.py`
(arm_detections), `job_monitor_node.py`, `perception/tray_detection.py` (reanchor),
`scripts/dev/job_run.sh`, `scripts/dev/tray_reanchor_check.py`

**Sequence.** The arm's task runs as in the cell at each station; between them the
base drives with the arm in the carry pose. Start at home: to the place station, find
the tray (empty hand, the cell's two-frame first scan); to the pick station: scan the
pile, pick, lift; to the place station: scan the tray with the box held, plan the
place there, place, scan with the empty hand; back to the pick station; when the pile
is empty, home. The drives are navigation goals (`/nav/goal`), done when `/nav/status`
says docked (or parked) for that goal after it has been seen taken up; a failed
drive is tried once more. The base only drives with the arm in the carry pose.

**The carry pose** (`scene.CARRY_TCP_POSITION`, `ARM_CARRY_Q`): the tool straight down
0.40 m behind the arm's base and 0.22 m above its mount (arm frame), over the rear
deck: the largest box (0.13 m tall, 0.13 m half-diagonal) hangs inside the footprint
and above the pedestal, every link over the robot (0.39 rad from every joint limit).
The tucked pose of the navigation tests (`ARM_TUCKED_Q`) had the tool tilted 20 deg at
the front edge: a box would have stuck out. Into the carry pose the arm goes across at
its height first, then down; out of it, straight up to the next leg's height first: a
long swing that rises as it goes would carry a low box into the tray's walls.

**Azimuths.** The task plans in cylindrical coordinates round the arm's base and went
the shorter way round; the carry pose (+90 deg) and the tray (about -125 deg) are 145
deg apart the short way, across the back, where joint 1 cannot follow (+-166 deg): the
arm stuck 11 mm from the first tray scan pose, contorted. In the mobile job azimuths
stay within +-180 deg (joint 1's range) and the heading profile measures the swing the
same way: from the carry pose to the tray the arm swings over the robot's left side
and front (215 deg), joint 7 picked to stay in range.

**The zones stay as taught.** At the pick dock (6 cm farther from the table than in
the cell) the pile still lies inside the cell's pick zone (boxes at x 0.38-0.63 m in a
zone 0.19-0.69 m), and the place dock puts the tray where it was in the cell.

**The tray across visits.** The tray, the boxes placed in it and its last heightmap
are kept in the arm frame of the last visit; the base docks a few mm and tenths of a
degree differently each time. At each later visit the tray scan (box held, its pixels
blanked) finds it again (`tray_detection.reanchor`): the inner faces within 3 cm of
the known walls; the tray's size is known, so one wall per axis is enough (the held
box hides the wall nearest the arm), and where both are seen the size must not have
changed. The shift moves the placed boxes and the heightmap (resampled) with it; a
tray not found again sends the base to dock again (twice, then stop). Offline
(`tray_reanchor_check.py`, dockings up to 15 mm and 0.4 deg off): 20 of 20 found again,
walls within 1.8 mm, with the box held or not; requiring all four walls refused every
held scan. Live the tray moved 0.2-3.5 mm between visits.

**Obstacle hulls per station.** The pile's and the tray's hulls (MPC obstacles) are in
the arm frame of their station; in the mobile job each is on only at its own station.

**People at the arm.** The arm's supervisor keeps its input format: `people_node`
(arm_detections:=true) publishes each scan's people, found against the map, moved into
the arm frame (`/perception/lidar_detections`, as the parked cell's lidar node gave
them). The arm-frame positions come out as the raw scan geometry (the localized pose
in and out), so a localization error cannot hide a person from the arm; it would only
make walls look like people, and the arm hold. A silent source holds the arm (the
supervisor's watchdog); the base's safety layer has its own watchdogs.

**Validation** (`job_monitor_node`, truth only): the plant's model at the base's true
pose, the arm's joint angles and the people's true poses; the smallest distance
between the arm's collision geoms (and a held box's) and any part of a person within 3
m (`body_clearance.py`: legs, torso, arms, head), apart while the arm works (docked,
the arm moving) and while the base drives: then also apart when the base moves and its
nearest point closes on that person (its speed from the truth; people walking up to
a robot standing still count only in "any"); each supervisor hold with the nearest
person's distance (a hold with nobody within 2.5 m is unexplained). `nav_monitor_node` scores the drives, `localization_monitor_node` the
pose. `job_run.sh` starts the job (`/mpc/go` once the controller is ready) and waits
for "job done".

**Results** (live, headless, real time; the full job: 8 boxes from the pile's table to
the tray's, then home). Quiet room: 8/8 placed in 20 min; 19/19 drives docked within
7 mm and 0.9 deg; the tray found again on every visit (moved 0.2-3.5 mm); boxes 1.4-10
mm from the walls and each other; localization 4/9/14 mm (p50/p95/max). With the job's
crowd of six: 8/8 placed in 32 min; 78 arm holds, every one with a person within 2.5 m
of the arm (none for perception, none unexplained); the closest anyone came while the
arm worked 0.93 m; while the base drove, a person's upper body came within 72 mm of the
folded arm's elbow (link3, 1.4 m up near the chassis's front edge; the base's fields
kept the legs 82-110 mm or more from the chassis, known_issues Q1); 19/19 drives docked
within 7 mm and 0.9 deg, no protective-field intrusion, no contact; people tracked
97.5% within 5 m; localization 4/9/16 mm. With people kept 10 cm farther by the
base's fields (safety.py): 8/8 in 32.9 min; the arm or box to a person's body while
the base drove 192 mm closing on them, 127 mm at any time; 95 holds, 94 with a person
within 2.5 m, one of 0.2 s with nobody near while the base drove (the arm's supervisor
tracks in the arm frame, which turns with the base: a coasting track drifted onto the
folded arm); 19/19 docked within 7 mm and 0.9 deg, no intrusion, no contact; people
97.6%; localization 4/9/15 mm. On the way: the step-3 crowd's bystander
standing for good 1.3 m from the pick station's arm held it for ever (the job's crowd
lets the bystander come and go); a Ruckig numerical failure in the task's step crashed it once
(now taken again from zero acceleration); a single `/mpc/go` can be lost before the
controller is discovered (`job_run.sh` repeats it until the controller starts). The
parked cell's demo with a person, after the changes: 8/8, every hold with a person.

## Arm and base together (step 6)

`task_node.py` (mobile), `nav/navigator.py` (arm_ready, paused), `nav_node.py`,
`obstacle_supervisor_node.py`, `people_node.py` (arm_in_odom), `job_monitor_node.py`,
`scripts/dev/dock_reach_sim.py`, `scripts/dev/job_time.py`

**What there was to gain.** The quiet step-5 job (8 boxes, 20.2 min, 152 s a box,
`job_time.py`): the base drove 906 s (route 563, the docking approach 187, the undock
132, alignment 18), the arm tucked 98 s and unfolded 33 s, and worked 176 s. While
driving, the tool stayed within 1.1 mm (p95) and 6.4 mm (max) of the carry pose: the
carry needed nothing. The approach and the undock (0.9 m at 0.10 and 0.15 m/s, a
docking table 5 cm off) are where the arm can move at the same time.

**Three ways, offline** (`dock_reach_sim.py`: the plant at the stations, lockstep, the
poses of the live job; the time until the base has docked or undocked and the arm has
settled): overlap (the arm's legs as the task plans them, in its own frame, started as
the base sets off backwards or starts its approach), split (the station's first pose
held fixed in the room while the base moves: the arm's MPC given goals moved into the
arm frame along the base MPC's prediction) and wb (the whole-body MPC below drives
both).

| one docking each | in sequence | overlap | split | wb |
|---|---|---|---|---|
| pick, in (carry pose to the pile scan pose) | 12.7 s | 10.0 s | 10.0 s | 9.6 s |
| pick, out (the lift to the carry pose, a box held) | 9.7 s | 7.2 s | 7.2 s | 7.2 s |
| place, in (a box held, carry to the tray scan pose) | 15.2 s | 10.1 s | 12.0 s | 12.1 s |
| place, out (the tray scan pose to carry) | 12.4 s | 7.3 s | 7.3 s | 7.0 s |
| a cycle's four | 50.0 s | 34.6 s | 36.5 s | 36.0 s |

All docked within 1 mm and 0.3 deg, the tool within 5 mm of its reference, the arm
and box as far from the stations as in sequence (55-288 mm), no contact; the solves
2.7 ms (the arm's MPC) and 3.6 ms (whole-body) at the median. Overlap is the fastest:
the station's first poses are scan poses well above anything on the tables, so holding
them still in the room while the base moves gains nothing, and it waits until they are
within reach (split and wb start the reach only within 0.8 m of the shoulder; with the
base 0.9 m off, the tray's scan pose is not). The heading turns the way that keeps
joint 7 in range (joint 7 ~ joint 1 - heading - 135 deg), as the task's own heading
profile does; the shorter way drove joint 7 into its limit and the arm stuck.

**Live (overlap).** The task sends the next goal as the arm starts to tuck: the base
backs out of the dock while the arm folds (`/task/arm_stowed` false); off the docking
line (planning, a route, an alignment turn) the navigator waits until the arm is
stowed, which includes the tool over the carry pose at any height (within 5 cm in
plan: a person standing 1.3 m off held the arm there, the base waited for the arm,
and the person waited beside the spot the robot stood on; leaving frees it). On the straight approach (`/nav/status` "approach") the task starts the
station's first move (out of the carry pose straight up, then to the scan pose); a scan
asked for before the base has docked waits for the docked status, so what the camera
sees is where the arm will work. A docking retry (backout, align) or a failed drive with
the arm out sends it back to the carry pose first. While the arm is out and the base is
not docked, a hold by the arm's supervisor pauses the base too (`/task/pause_base`),
until the hold has been off for 1 s (at the threshold the hold flickers: in the first
crowd run the base paused and went on several times within a second), and only while
the tool is beyond the chassis outline (5 cm in): held over the deck the arm is under
the base's fields, and pausing there had the base wait at the start of its approach for
someone standing 1.5 m off, on the spot that person wanted back, for good.
Quiet room, the full job: 8/8 in 17.2 min (20.2 min in sequence), 129 s a box (152);
the arm was out while the base docked or undocked for 286 s, and the base waited for
it off the docking line 0.8 s in all (the tuck at the pick station takes 2-4 s, the
undock 7); the scans began as the base docked (within 0.1 s; 3.3-6.4 s after it in
sequence) but for the first visit to each station;
19/19 docked within 7 mm and 0.9 deg; localization 4/8/14 mm. The first quiet run
stopped after 4 boxes at a blocked place (below), not at anything the overlap did.
With the job's crowd of six: 8/8 in 30.2 min, 227 s a box (245 in sequence, the Q1
run); the docking approaches and undocks took longer than in sequence (261 and 244 s
against 187 and 133), the base paused for people near the unfolded arm; 120 holds,
every one with a person within 1.6 m of the arm (none unexplained: the supervisor's
tracks no longer drift); the arm or box to anyone while the base drove 206 mm closing
on them, 155 mm at any time; 19/19 docked within 6 mm and 0.93 deg, no intrusion, no
contact; people 97.3%; localization 4/8/15 mm. Again with the pause kept 1 s past
the hold and the monitor's "arm out" test on the tool's position (the live carry pose
sits 0.11 rad off `ARM_CARRY_Q` in joint 1, so the joint test had counted the carry as
out): 8/8 in 31.4 min, 236 s a box; the base paused 43 times for 169 s in all (0.7-10.5
s each); while the base moved with the arm out nobody came nearer the arm than 1.14 m;
114 holds, all with a person near; 19/19 docked within 6 mm and 0.75 deg, no intrusion,
no contact. In a crowd the pauses take most of the overlap's gain back: a person
within the supervisor's distance of the unfolded arm stops the base as it stops the
arm (a held arm on a moving base still moves toward them). With the final rules
(the base pauses only for an arm held beyond the chassis, step 7): 8/8 in 27.8 min,
209 s a box; nobody nearer the arm than 1.0 m while the base moved with it out; 146
holds, all with a person near.

**The supervisor on a moving base.** The arm's supervisor tracked the lidar's people
in the arm frame, which turns with the base: a person standing still seemed to move,
and a coasting track kept going (offline: 0.31 m off after 0.8 s of coasting while the
base turned at 0.6 rad/s; live once a 0.2 s hold for nobody). `people_node` now also
publishes the arm frame's pose in the odometry frame at each scan
(`/perception/arm_in_odom`); the supervisor tracks there (0 m off), takes the tracks
back into the arm frame for the separation, and adds the base's motion (`/odom`) to
the arm points' speed toward a person. On the parked cell (no odometry) it is as
before.

**A blocked place, seen again.** After a place blocked on a neighbour's edge (Q3) the
held rescan came from the usual scan pose, where the held box hides 29-80% of a blocked
spot on the tray's far side; the blocked area is unknown until seen, so no spot fitted
and 30 rescans from the same pose ended the job with the box held (once, live). The
rescan is now taken with the wrist camera just past the blocked spot and the held box
beyond it (`_blocked_view_pos`): 0% hidden at 12 positions over the tray (offline
render). The parked cell after steps 6 and 7 (the supervisor and the task are
shared): 8/8 plain and 8/8 with the person visiting the tray, every hold for them.

## wb_ocp.py

`core/wb_ocp.py` (not used live; `dock_reach_sim.py --strategy wb`)

The whole-body MPC of the plan: one acados OCP over the base (unicycle: pose, v, omega;
inputs the two accelerations) and the arm (the torque dynamics of `core/ocp.py`, fixed
base), in a station frame W (the arm frame of the nominal dock), 15 stages of 20 ms
like the arm's controller. The tool's goal per stage is in W or in the arm's own frame
(a selector parameter): `sel (T(base) fk(q) - g) + (1 - sel) (fk(q) - g)`, the tool
axes likewise; the base tracks a reference along its docking line with the lag and
lateral error weighted apart (as `nav/base_mpc.py`); joint 1's posture reference is
the goal's azimuth in the arm frame at that stage's base pose. Obstacle spheres (the
stations' hulls) are in W and apply to every proxy; people are vertical cylinders in W,
every proxy kept their radius plus 10 cm away (soft). The base's velocity, acceleration
and wheel-rim limits are as the base MPC's, slower (0.2 m/s, docking only). 19 states,
9 inputs, 20 soft constraints; 3.6 ms per solve at the median. It docked and tracked as
well as the two controllers apart (above) and was not faster for this job; the arm's
inertia under the base's acceleration is left out (as in the arm's own MPC: the carry
holds within 6 mm while driving).

## The scenario set (step 7)

`pick_place_common/mobile_scenarios.py` (CROWDS), `launch/mobile_job.launch.py`
(crowd:=, people:=), `scripts/dev/job_results.py`, `scripts/dev/job_figure.py`

**Scenarios** (plant side; the method sees only the lidars): quiet (nobody); walkers
(three of the crowd: one looping the room's middle and giving way, one crossing the
room and never giving way, one on a slow diagonal never giving way); the crowd of six
of the job; dock block (someone standing on the pick station's docking line, 0.35 m
behind the docked robot: in the way of the approach and of the back-out, 40 s there,
40 s two metres off, never giving way); step in (someone waiting 1 m beside the
diagonal route between the stations who steps onto it when the robot is 2.5 m from the
spot, stands 6 s, steps back, and does it again once the robot has gone; since the
results below, the crowd's person who stands, then walks off starts away from home: 0.5
m beside the first route, they held the start back for 45 s). Each person
in the plant also stops for the robot's body in front of them and steps round it after
3 s.

**Offline navigation** (`nav_sim.py`, 20 drives pick > place > home, the arm in the
carry pose with the largest box, two seeds each; the 10 x 8 m room):

| 20 drives, seeds 0 / 1 | quiet | walkers | crowd | dock block | step in |
|---|---|---|---|---|---|
| arrived | 20/20 | 20/20, 20/20 | 20/20, 20/20 | 20/20, 20/20 | 20/20, 20/20 |
| docked, max lateral / yaw | 6 mm / 0.27 deg | 7, 5 mm / 0.38 deg | 6, 5 mm / 0.23 deg | 7, 6 mm / 0.31 deg | 6, 5 mm / 0.37 deg |
| closest person, chassis moving (mm) | - | 188, 139 | 111, 86 | 250, 225 | 74, 254 |
| ...also walking up to it (mm) | - | 90, 98 | 98, 86 | 131, 136 | 68, 248 |
| protective-field intrusions, contacts | 0, 0 | 0, 0 | 0, 0 | 0, 0 | 0, 0 |
| arm or box to a body, closing / any (mm) | - | 278, 240 / 225, 118 | 250, 230 / 185, 138 | 382, 310 / 269, 270 | 226, 327 / 123, 327 |
| wiggles (largest swing), oscillations | 0, 0 | 2 (26 deg), 9 (27); 0, 3 | 4 (21), 2 (7); 0, 0 | 0, 0 | 0, 0 |
| detour max | 1.02 | 1.01, 1.13 | 1.05, 1.20 | 1.06, 1.05 | 1.01, 1.01 |
| simulated time of the 20 drives | 738 s | 887, 939 s | 1036, 1042 s | 1207, 1147 s | 917, 792 s |

The person in the docking line holds the base at the end of its route until they walk
off (the docking moves leave people to the safety layer). The one stepping in stops
the base (no detour: it waits the 6 s and goes on); walking into its path they came
within 74 mm of a base already braking (P10). The scenario people were made less
stubborn twice on the way (plant side): a spot under the robot is skipped, and a
person blocked by the robot steps aside the free way nearest their way on (cornered
between the wall, the pick table and the robot, both perpendicular sidesteps had
pointed at the robot, and one stood on the pre-dock pose for good).

**Live** (the full job of 8 boxes each, headless, real time, the final system, in the
10 x 8 m room with six people in the crowd and three walkers; `job_results.py`,
`job_figure.py` > `figures/mobile/scenarios.png`):

| | quiet | walkers | crowd | dock block | step in |
|---|---|---|---|---|---|
| boxes placed | 8 | 8 | 8 | 8 | 8 |
| job time (min) | 17.2 | 21.2 | 27.8 | 18.9 | 20.9 |
| time per box (s) | 129 | 159 | 209 | 142 | 157 |
| distance driven (m) | 208 | 211 | 211 | 210 | 209 |
| drives docked / parked | 19/19 | 19/19 | 19/19 | 19/19 | 19/19 |
| docking error max: along / lateral (mm), yaw (deg) | 6 / 7, 0.69 | 6 / 7, 0.66 | 5 / 6, 0.85 | 6 / 7, 0.96 | 6 / 7, 0.88 |
| closest person to the chassis while it moved (mm) | - | 102 | 160 | 195 | 208 |
| ...also people walking up to it (mm) | - | 92 | 98 | 195 | 204 |
| protective-field intrusions, contacts | 0, 0 | 0, 0 | 0, 0 | 0, 0 | 0, 0 |
| closest person to the arm while it worked (mm) | - | 1461 | 927 | 1451 | 1579 |
| ...to the arm or box while the base closed on them (mm) | - | 177 | 231 | 219 | 289 |
| ...at any time while not docked (mm) | - | 87 | 99 | 115 | 119 |
| ...while the base moved with the arm out (mm) | - | 1562 | 1007 | 359 | - |
| arm holds: a person near / perception / unexplained | 0 / 1 / 0 | 40 / 0 / 0 | 146 / 0 / 0 | 61 / 0 / 0 | 18 / 0 / 0 |
| people tracked within 5 m (%) | - | 98.2 | 97.7 | 100.0 | 100.0 |
| localization error p50 / p95 / max (mm) | 4 / 9 / 14 | 4 / 8 / 14 | 4 / 9 / 15 | 4 / 8 / 13 | 4 / 8 / 13 |
| wiggles (largest swing deg), oscillations, turn-backs | 0 (0), 0, 0 | 5 (26), 0, 1 | 9 (29), 0, 1 | 0 (0), 0, 0 | 0 (0), 0, 0 |
| both moving / base waiting for the arm (s) | 284 / 1.5 | 281 / 1.5 | 402 / 18.5 | 315 / 115.1 | 286 / 3.9 |

Every job placed its 8 boxes and every drive docked; no protective-field intrusion, no
contact; every arm hold had a person near it (the quiet run's one hold is the
supervisor's own at start-up, before the first detections). People came within
87-99 mm of the arm or the box only by walking up to a robot standing for them; while
the robot closed on someone the nearest was 177 mm, and while the base moved with the
arm out 359 mm (the dock-blocking person, by the robot backing out) or a metre and
more. The time per box follows the people more than anything: 129 s quiet, 142-159 s
with one to three people, 209 s in the crowd. The docking line's person cost the base
115 s of waiting in all, standing within the arm's separation distance while the arm
went into its carry pose (R4). Getting there took fixes on both sides: the scenario
people (a spot under the robot is skipped; blocked, they step aside the free way
nearest their way on) and the method (the base pauses only for an arm held beyond
the chassis; an arm over the carry pose counts as stowed, whatever its height); with
the earlier rules the dock-block run stalled three times, each a mutual wait between
the robot and one person.

## The room at half its floor area

`scene.py` (ROOM_SIZE, ROOM_FIXTURES, CELL_POSE, the stations, home), `room_scene.xml`,
`mobile_scenarios.py`, `scripts/dev/room_layout.py`, the stations map

The room went from 10 x 8 m to 7.0 x 5.6 m (49% of the floor). Everything keeps its
distance to its own walls, so what was taught near a wall still holds: the pick
station (its table backing onto the west wall, the docking line 1.1 m from the south
wall) did not move; the place station keeps its distances to the east and north walls,
(6.25, 4.35); home is the middle of the south wall, (3.5, 1.0). The pillar went into
the north-west corner and the shelf to the north wall's middle, both clear of the
docking lines and of the mapping loop round the room. The parked cell is turned 90 deg
((3.0, 2.8), yaw 90 deg): the base backs 1.7 m out of it (the scripted drives) and its
visiting person enters from 3 m the other way, 5.2 m in all, which fits along the
room's 7 m but not across its 5.6 (the scripted back-out now follows the base's
heading). `room_layout.py` checks it all on the compiled scene (each table 5 cm off its
wall; the chassis turning on the spot at home, the pre-dock poses and the routes'
corners, 0.11 m or more from anything; routes and people's paths clear of tables and
fixtures) and draws `figures/mobile/room_layout.png`.

The stations map was made again the same way (our SLAM, two laps of the loop, nobody
in the room: 119 keyframes, 218 loop closures; the old one kept as `stations_10x8`):
its wall cells lie within 3 cm of the true walls (`figures/mobile/stations_map_7x5.6.png`).
The people's paths were laid out again in the smaller room, and the crowds are smaller
(4 in the job's crowd, 2 walkers: six people in half the floor had one walk into a
nearly stopped base's corner, 18 mm off). The scenario people were made to give up a
spot only when it is within 0.6 m of the robot's centre (0.8 m kept one of them on
the docking line for good: the robot, waiting for them, stood 0.77 m from their other
spot).

Home and the pick pre-dock pose are now 2.65 m apart, both facing north: a drive from
home to the pick station arrives heading west, turns on the spot and ends about 3 cm
off the docking line, more than the approach absorbs, so it aligns again (turn,
straight, turn: about 7 s). The job never drives home > pick (home > place first).

Checked offline (`nav_sim.py`, 20 drives each, two seeds: the folded arm quiet and
with the crowd of four; the carry pose with a box with the job's crowd, two walkers,
the docking line's person and the one stepping in): 200 of 200 drives arrived, docked
within 7 mm and 0.39 deg, no protective-field intrusion, no contact, the arm or box 126
mm or more from anyone (183 mm while the robot closed on them). Live, the full job:
quiet 8/8 in 15.8 min, 119 s a box (129 s in the larger room; the docking moves do not
get shorter), docked within 6 mm and 0.9 deg, localization 4/8/13 mm in the new map;
with the job's crowd of four 8/8 in 27.4 min, 206 s a box, no intrusion, no contact,
146 of 147 arm holds with a person near (the other the supervisor's start-up hold). The
parked cell in its new place: 8/8 plain and 8/8 with the person visiting the tray.

## Speed

`nav/navigator.py`, `nav/base_mpc.py`, `nav/reference.py`, `nav/safety.py`,
`pick_place_mpc/nav_node.py` (`speed`), `launch/mobile_job.launch.py` (`speed:=`,
default 1.0), `scripts/mobile_pickplace.sh` (`speed=`), `pick_place_mpc/task_node.py`
(transit legs), `scripts/dev/nav_sim.py` (`--speed`, `--accel`, the time per state),
`pick_place_common/mobile_scenarios.py`

**Where the time went.** The quiet job took 119 s a box: the base drove at 0.5 m/s,
docked the last 0.9 m at 0.1 m/s and undocked at 0.15 m/s, turned at 0.6 rad/s, and
the flat warning boxes held it at 0.3 m/s (0.3 rad/s) whenever anything was within 1
m beyond its stop, which in the smaller room is often. Offline (20 drives, the carry
pose, quiet): 684 s simulated; the faster docking alone 585 s; 1.0 m/s 473 s; with
the graded warning caps (`safety.py` above) 436 s. At 1.0 m/s a third of the route
time is turning in place (the flat turn cap had been on for a quarter of it); 1.5 m/s
with 0.8 m/s^2 was not faster (455 s).

**The base.** One setting, the top speed (`speed`, the job's default 1.0 m/s): the
routes' cruise, the turn rate (1.2 rad/s per m/s, within 0.6-1.2 rad/s), the rim limit
(|v| + 0.25 |omega| up to 1.4 m/s) and the speed near people (0.2 m/s at 0.9 m, 0.5 at
1.8 m, the top speed from 3 m). The approach drives the docking line at 0.3 m/s and
its last 0.25 m at 0.1 m/s (the reference brakes into it); undocking and aligning
at 0.3 m/s.

**The arm (the mobile job only).** Into and out of the carry pose and the first move
at a station, high over everything, at 0.8 m/s instead of 0.5 (offline,
`dock_reach_sim.py`: the tool within 5 mm of its reference, the clearances to the
station as before; the move to the tray scan 5.1 > 3.8 s); the tuck's move across and
the unfold's rise blend into the next leg instead of settling at the corner; the pause
after a scan 0.2 s instead of 1 s. The parked cell keeps its limits.

**Two fixes the speed brought out.** The step back above, and dropping a plan that
comes back once the base has gone on to dock (`navigator.py` above). In the scenario
model, the step-in person stepped in only once in the smaller room (the place dock is
3.0 m from their spot, re-arming needed 4.0 m; now 2.9 m), and someone whose other
spot the robot stands on goes beside it instead of staying put (crowd model above).

**Checked offline** (`nav_sim.py`, 20 drives each, seeds 0 and 1 where people walk:
the folded arm quiet and with the crowd of four; the carry pose with a box quiet, with
two walkers, the job's crowd, the docking line's person and the one stepping in): 240
of 240 drives arrived, docked within 7 mm and 0.35 deg, no protective-field
intrusion, no contact; the closest anyone came to the moving chassis 73 mm (a walker
walking into it), the arm or box 125 mm (207 mm while the robot closed on them). 20
drives, simulated: quiet 438 s (684 s before), the crowd of four 683-698 s (874-941 s).
Someone standing near home, as the step-in person did before the fix: 20/20, a drive
home past them 48 s (the base waited for good before the step back).

| the full job, live | quiet, before | quiet | crowd of 4, before | crowd of 4 |
|---|---|---|---|---|
| boxes placed | 8 | 8 | 8 | 8 |
| job time (min) | 15.8 | 10.5 | 27.4 | 18.8 |
| time per box (s) | 119 | 79 | 206 | 141 |
| drives docked | 19/19 | 19/19 | 19/19 | 19/19 |
| docking error max: along / lateral (mm), yaw (deg) | 4 / 6, 0.89 | 4 / 7, 0.59 | 7 / 5, 0.9 | 4 / 7, 0.93 |
| protective-field intrusions, contacts | 0, 0 | 0, 0 | 0, 0 | 0, 0 |
| closest person to the moving chassis (mm) | - | - | 74 | 140 |
| closest person to the arm or box while not docked (mm) | - | - | 121 | 97 |
| arm holds: a person near / other | 0 / 0 | 0 / 1 | 146 / 1 | 101 / 1 |
| localization p50 / p95 / max (mm) | 4 / 8 / 13 | 4 / 8 / 15 | 4 / 7 / 13 | 4 / 7 / 12 |
| swerves (largest, deg), oscillations | 0, 0 | 0, 0 | 8 (24), 1 | 16 (30), 5 |

(The other hold is the supervisor's start-up hold, before the job starts.) Where the
quiet job's time goes now (`job_time.py`): the base drove 271 s on routes (460
before), approached 113 s (187), undocked 82 s (133); the arm tucked 73 s (103) and
unfolded 11 s (31). With the crowd, the base waited 118 s off the docking line for the
arm (67 s before): the arm's supervisor holds the tuck while someone is near, and the
base leaves the line only with the arm stowed. The first crowd run stopped for good on
the late plan (fixed above; that run is not in the table). The parked cell, whose
paths in `task_node.py` did not change: 8/8 plain (188 s; 175 s before, one box placed
turned and pushed with a tilt) and 8/8 with the person visiting the tray (199 s; 203
s).

## Safety settings

`scripts/mobile_pickplace.sh` (`base_safety=`, `arm_safety=`, `assumed_speed=`,
`people_speed=`, or `key:=value` after the scenario), `launch/mobile_job.launch.py`,
`nav/navigator.py` and `nav_node.py` (`safety`), `nav/safety.py` and `safety_node.py`
(`person_extra`), `obstacle_supervisor_node.py` (`arm_safety`, `human_speed`),
`mobile_scenarios.walking` and `mobile_scenario_node.py` (`people_speed`),
`job_monitor_node.py`, `nav_sim.py` (`--base-safety`, `--people-speed`)

Settings for trying the robot closer to people or more cautious, and people faster or
slower. What scales is the margins on top of the physics: the stopping distances
(reaction and braking, of the base and of the arm) never do.

- **base_safety** (x, default 1): the distances where the base slows for the nearest
  person (0.9 / 1.8 / 3.0 m), the room the MPC keeps from people (0.15 m beyond a
  person's own 0.3 m; also the discs round people standing in the way), and a person's
  whole margin in the protective field (0.20 m), which never goes below 0.14 m: the
  scanners see legs, and a person's torso and arms reach 0.135 m beyond them (with
  0.10 m the arm came within 72 mm of a person's upper body, known_issues Q1).
- **arm_safety** (x, default 1): the supervisor's allowance for a person seen by the
  lidar (ISO 13855 C for a scan plane 0.2 m up, 1.12 m; never below 0.14 m), its
  position pad (0.05 m) and the band over which the arm's speed recovers past the
  required distance (0.6 m). The job monitor counts a hold as explained by a person
  within 2.5 m x the larger of 1 and arm_safety (and of assumed_speed / 1.6).
- **assumed_speed** (m/s, default 1.6, ISO 13855's walking speed): the speed the arm's
  supervisor assumes a person approaches at, over its reaction and stopping time.
- **people_speed** (x, default 1): the scenario people's walking speeds (0.8-1.3 m/s).
  The plant only: the method is not told, it senses them.

The base's warning caps (0.5 m beyond its stop, for anything) do not scale: they are
for all obstacles, not people.

Offline (`nav_sim.py`, the carry pose with a box, 20 drives, seeds 0 and 1 for the
job's crowd of four): every drive arrived, docked within 7 mm and 0.39 deg, no
protective-field intrusion, no contact.

| 20 drives, simulated | base_safety 0.5 | 1.0 | 1.5 | people 1.5x faster |
|---|---|---|---|---|
| the job's crowd of four | 526-537 s | 695-703 s | 803-837 s | 680-720 s |
| closest person to the moving chassis | 123 mm | 121 mm | 121 mm | 84 mm |
| arm or box, the robot closing on them | 179 mm | 207 mm | 256 mm | 180 mm |
| the docking line's person / the one stepping in | 615 / 467 s | 640 / 561 s | 691 / 806 s | |

The closest approaches to the moving chassis hardly change with base_safety: they are
people walking into a base that has stopped. Below about 0.7 the stop margin is at
its floor; what lower values still change is where the base slows and how near it
plans past people.

Live, the full job with the crowd of four (1+1 runs, besides the default run of the
speed section):

| | close: base 0.6, arm 0.6 | default | cautious: base 1.5, arm 1.5, assumed 2.0 m/s |
|---|---|---|---|
| boxes placed | 8 | 8 | 8 |
| job time, time per box | 14.1 min, 106 s | 18.8 min, 141 s | 29.4 min to the last box, 221 s |
| drives docked | 19/19 | 19/19 | 18/18 (the drive home cut short) |
| protective-field intrusions, contacts | 0, 0 | 0, 0 | 0, 0 |
| closest person to the moving chassis | 148 mm | 140 mm | 144 mm |
| arm or box, the robot closing on them | 190 mm | 216 mm | 235 mm |
| arm, anyone walking up while not docked | 83 mm | 97 mm | 48 mm |
| arm holds, all with a person near | 58 | 101 | 117 |
| the base waiting off the docking line for the arm | 4 s | 118 s | 178 s |

The settings did what they say: the close run's arm held half as often and the base
hardly waited for its tuck; the cautious run held more, waited longer and closed on
nobody nearer than 235 mm. The closest approaches to the arm in every run are someone
walking into the folded elbow of a robot that was not moving towards them (the
crowd's crossing walker, who never gives way); the cautious robot spends longer
in the room, so they did more often. The cautious run was stopped by the test
harness's time limit on its drive home, after the last box, so its monitors logged no
totals; its row is from their per-drive and per-minute lines.

## Localization by sensor fusion

Plan: `handover_notes/ekf_plan.md` (local only). Today's localizer replaces its estimate
with each good scan match; this work adds an extended Kalman filter that weighs the
wheels, the gyro and the scan match by their noise, and conditions in the plant that
make the difference visible.

### Plant conditions (step 1)

`pick_place_common/plant_conditions.py`, `base_drive.py` (`conditions`),
`plant_mismatch.apply_base_mismatch` (`worn`), `mujoco_sim_node` and `sim_sensors_node`
(parameter `conditions`), `conditions:=` in `mobile.launch.py` and
`mobile_job.launch.py`, `odom_check.py --conditions`, `nav_sim.py --conditions`.

The simulated floor barely slips and the room has walls everywhere, so the scan match
alone does well. Each condition is plant side; the method is never told.

- **spill**: friction 0.2 instead of 1.0 (a wet floor) on a 0.9 x 1.4 m patch in front
  of the place table, where the robot stops, turns in place and starts. Each tick the
  wheel-floor contact pairs take the friction under each wheel, and so does the braked
  wheel's static friction. Driving at the navigation's limits (0.5 m/s^2, 1.5 rad/s^2)
  hardly slips on it (3.5 mm more over a stop, a turn and a start); a protective stop
  does: the drives brake with up to 30 Nm per wheel (4.5 m/s^2 on a dry floor), the wheels
  lock and the base slides. At 0.25 friction the stop slid as far, the smooth driving
  half as much.
- **worn_tyre**: the left tyre 2% smaller (on top of the 0.5% mismatch).
- **gyro_drift**: the gyro's bias wanders (first-order Gauss-Markov: 60 s time constant,
  0.005 rad/s steady state, the datasheet's range) instead of staying fixed per run.
- **dropout**: the localization's scans lost for 1-2 s at a time, every 8-20 s: a
  network hiccup between the scanners and the navigation computer. The sensors node
  publishes a lossy copy (`/env/lidar_scan_lossy`) and the launch remaps slam_node to it;
  the safety layer and the people detection keep the full feed (on a real robot the
  safety fields run in the scanners). The safety layer's localization watchdog (no
  pose for 0.5 s) then stops the base, as it would live.
- **crowding** (people round the robot so few wall points match) was tried and dropped:
  four visitors walking beside and behind the robot at 1.1-1.3 m took the scan match's
  fitness (the share of points on a mapped wall) only from 0.79 to 0.76 (p5) and trapped
  the robot by the pick table. Thin people hide little of the walls; the job's crowd
  of four stays among the normal conditions.

Checks (`odom_check.py --conditions ...`, seeds 0-2; the profiles now in the stations
layout's open middle: since the room was halved, the square and the circle ran into
the parked cell, metres of error):

| profile | normal | with the condition |
|---|---|---|
| a 0.5 m/s protective stop on the patch (`wet_stop`): odometry error | 1-2 mm | 81-92 mm (the slide) |
| square, circle: wheels-only heading error | 1.6-5.6 deg | 4.5-20.4 deg (worn_tyre) |
| straight, square, circle: wheels + gyro, worst error | 3-12 mm | 11-37 mm (worn_tyre), 8-42 mm (gyro_drift) |
| the true gyro bias over a profile | fixed | moves by up to 3.3 mrad/s (gyro_drift) |

In `nav_sim.py` the job's drives rarely stop hard on the patch (the warning field slows
the base first), so with `spill` a protective stop is injected the first time in each
drive the base crosses the patch above 0.3 m/s, for 1 s, as if someone stepped in; the
wheels then slip up to 7 mm in a tick (2 mm on a dry floor).

`nav_sim.py` also scores the localization on the truth every tick
(`pick_place_common/loc_scoring.py`): position and heading error, pose jumps (the
estimate's change per tick beyond the true motion) and, given a covariance, NEES; and
counts lost, dropped and slipping scans.

### Baseline: today's localizer under each condition (step 2)

`nav_sim.py` (20 drives each, the carry pose with a box, seed 0 unless named; the job's
crowd of four), `results/ekf/baseline/*.json`, tabled by `scripts/dev/loc_compare.py`:

| run | position p50 / p95 / max mm | pose jumps max mm | docked lateral / along mm, yaw deg |
|---|---|---|---|
| quiet | 3.9 / 7.2 / 11.7 | 14.2 | 5.9 / 4.5, 0.18 |
| crowd, seeds 0 / 1 | 3.5 / 6.7 / 12.6-13.0 | 11.4-14.4 | 3.6-5.8 / 3.5-4.7, 0.28-0.35 |
| spill, quiet / crowd | 3.9 / 7.2 / 17.6, 3.5 / 6.7 / 12.4 | 16.1, 12.7 | 7.8 / 4.1, 0.61; 6.2 / 4.7, 0.35 |
| worn_tyre, quiet / crowd | 3.2-3.8 / 6.4-7.0 / 11.3-14.9 | 11.2-11.5 | 18.9-19.9 / 3.9-4.8, 0.46-0.55 |
| gyro_drift, quiet / crowd | 3.3-3.8 / 6.6-7.1 / 12.3-15.6 | 11.3-13.0 | 9.1-14.4 / 3.5-4.2, 0.98-1.37 |
| dropout, quiet / crowd | 4.0 / 7.5 / 49.2, 3.5 / 6.7 / 11.7 | 45.1, 11.3 | 6.5 / 17.2, 0.23; 5.6 / 3.9, 0.31 |

Every drive arrived; no scan was lost (the match's fitness never fell under 0.4).
What there is to fix:
- **Jumps**: each good match replaces the pose, so the pose jumps by the match's own
  noise, up to 11-16 mm in a tick, in every run.
- **Dropouts in the quiet room**: the localization watchdog stops the base from full
  speed; the stops slide on the dry floor too (up to 7 mm in a tick) and the odometry
  does not see it, so the pose is 49 mm off when the scans come back, and jumps 45 mm.
  With the crowd the base drives slower and the same gaps cost nothing.
- **The spill**: the slides of the 13 injected stops show the same way, 17.6 mm.
- **The worn tyre and the drifting gyro do not reach the localization** (each scan
  corrects the heading and the position), but they reach the docking: the worn tyre
  leaves the base 19-20 mm to the side of the docks (the tolerance is 20 mm), the
  drifting bias 1.0-1.4 deg off their heading (tolerance 1.5 deg). Both come through the
  navigation's own use of the odometry's turn rate and wheel speeds, not the pose; a
  filter that only replaces the map > odom transform does not change them.

### The filter: `slam/base_ekf.py` (step 3)

`slam/base_ekf.py`, `slam/test/test_base_ekf.py`, `slam/localizer.py` (`match`, the scan
match from any prior, leaving the localizer as it was).

State in the map frame: x, y, heading, forward speed v, turn rate omega, the gyro's
bias. Each 50 Hz tick (the plant's step, wheels and gyro of the same step):
1. The rates' uncertainty grows by the base's acceleration limits over the tick
   (0.5 m/s^2, 1.5 rad/s^2) and the bias by a slow random walk (2e-4 rad/s per sqrt(s)).
2. The wheels' speed (the mean of the two arcs over the tick) updates v, with the
   encoder's resolution (one count per tick) and 1% of the speed (tyre tolerance and
   slip). Standing still (encoders unmoved 0.3 s): v = 0 and omega = 0, tightly, so
   the gyro's reading is all bias (as the odometry learns it).
3. The gyro (its mean rate over the tick) updates omega + bias with its datasheet noise.
4. The pose is integrated over the tick with the updated rates (heading at the midpoint,
   as the odometry), and its covariance grows by 7 mm per sqrt(m) driven: the wheels'
   scale error between scans. Updating the rates before integrating makes them the
   tick's own (as measured); integrating with the previous tick's rates first would
   leave the pose one tick's travel behind after a start (20 mm at 1 m/s) until the
   scans pulled it back.

The wheels' turn rate is not fused, only compared: with tyre tolerance it is off by up to
0.02 rad/s at 1 m/s (four times the gyro's largest bias), systematically, and as white
noise it would drag the bias estimate. Gyro for the turn rate and wheels for the speed is
the usual set-up on wheeled robots. Slip: the wheels' turn rate off the gyro's by more
than three standard deviations of their tolerance (the wheels' speed then counts only as
far as it agrees, and v's uncertainty grows by the disagreement), or the wheels' speed
changing faster than 6 m/s^2: the drives can brake the chassis at 4.5 m/s^2 at most;
faster, the wheels have let go of the floor, a skid. A skid is an episode, as long as
the slide could last on the worst floor (friction 0.1, 1 m/s^2: v / 1 m/s^2, at least
0.2 s): at its start the position's uncertainty along the heading grows by the slide
that floor would allow (v^2 / 2 m/s^2) and v's by v itself; through it the wheels'
speed counts with 0.5 m/s of noise (locked wheels read zero, on a dry floor they are
nearly right), v's uncertainty grows at 6 m/s^2, and standing still is not trusted, so
the scans carry the pose through the slide (step 4 has why each part is needed).

Scans: the match starts from the filter's pose at the scan's capture time (from a 1 s
history of the ticks); its covariance is the inverse of ICP's information matrix plus a
floor (3 mm, 0.1 deg: a match's error beyond its information). Rejected if the match's
fitness is under 0.4, as before, or if the innovation fails a chi-square gate (99%, 3
degrees of freedom) instead of the fixed 0.3 m cap. Five good matches rejected in a row:
the covariance was too small; it is widened to the innovation and the match taken (a
reset). A late scan is applied at its capture time and the stored ticks re-run to now.

The covariance it reports adds the map's own error against the room (2.5 mm, 0.04 deg;
the commissioning evaluation measured the mapping error at 2-3 mm p50): the filter's
covariance is against its map, the map is a few millimetres off the room in places, and
no estimator in the map frame can see that (checked in step 4: without it the NEES is
8.4, with it 2.5).

Tests: the motion Jacobian against a numerical one; the bias learnt standing still and
while turning (with scans); an outlier scan rejected and a poor match refused; a late
scan giving the same state and covariance as the same scan in order; a skid widening
the covariance; the scans followed through a slide on locked wheels. Cost (Python): 0.02 ms per tick, the prediction 0.007 ms of it; 0.2 ms
per scan besides the match itself, with the re-run of two late ticks.

### Offline: shadow mode in nav_sim (step 4)

`scripts/dev/nav_sim.py` (`--fusion icp|ekf`: the estimator that drives, both run on
the same data every tick; `--scan-latency`, default 0.02 s: a scan reaches them a tick
after its capture, as live, where slam_node took them 24 ms p50 and 38 ms p95 after
capture; `--dump`; `--plot` also draws `localization.png`), `results/ekf/offline/*.json`
(`<run>_<driver>`), `scripts/dev/loc_compare.py`.

Tuning, seed 0 only (quiet and the crowd of four): the noise values above as first
chosen, from the datasheets and the robot's limits, gave 3.8 / 5.6 / 8.1 mm against the
scan matcher's 3.9 / 7.2 / 12.9 on the same data, but a NEES of 8.4 (66% in the 95% band).
The error had a part that moved with the place in the room (0.5 m cells' mean errors
spread 2.6 mm in x, up to 6.4 mm) and a -0.04 deg heading offset: the map's own error
against the room, which both estimators share. Adding it to the reported covariance
(2.5 mm, 0.04 deg) gave 95-96% in the band. Nothing else was tuned.

Found on the way, both in the filter now (step 3's text):
- **A skid is an episode.** The first version widened the covariance for the one tick
  the wheels locked; the next scan shrank it again, then the locked wheels said "zero
  speed" with their usual small noise while the chassis slid on, and the gate refused
  the scans that showed the slide: 81 mm off, 12 scans rejected, until a reset 0.4 s
  later (the scan matcher alone: 23 mm). Leaving the wheels out for the slide's length
  fixed that, but then a watchdog stop during a dropout (a skid on the dry floor, with
  no scans) coasted on at the old speed: 282 mm. The wheels counting with 0.5 m/s of
  noise through the skid does both: 19 mm and 29 mm (the scan matcher 23 and 32).
- **The tick's length from the plant's step, not the stamps.** Live, the samples'
  stamps are wall time; under load a 20 ms step looked like 5 or 40 ms, the wheels'
  speed jumped and 8 skids showed in the first seconds of a quiet drive. base_node
  integrates by the step count already; the filter now takes dt from it too (the stamps
  only line up the scans).

Results (20 drives each, the carry pose with a box, seed 0 unless named; "drives" is
the estimator that drove, both scored on the same data):

| run | the scan matcher drives: icp / ekf, max mm | the filter drives: icp / ekf, max mm | ekf p95 mm, NEES in band | docked, the filter driving |
|---|---|---|---|---|
| quiet | 12.8 / 8.1 | 12.9 / 8.1 | 5.6, 95% | 5.2 / 3.3 mm, 0.20 deg |
| crowd | 11.4 / 6.9 | 15.2 / 7.0 | 5.0, 96% | 5.8 / 3.4 mm, 0.25 deg |
| crowd, seed 1 | 12.0 / 6.6 | 12.6 / 7.1 | 4.9, 96% | 3.6 / 3.0 mm, 0.19 deg |
| spill, quiet / crowd | 24.7 / 20.3, 14.9 / 9.0 | 24.3 / 21.0, 21.3 / 13.6 | 6.0, 4.9; 95-97% | 7.5 / 2.8 mm, 0.44 deg; 6.5 / 3.0, 0.39 |
| worn_tyre, quiet / crowd | 11.7 / 10.4, 11.1 / 9.1 | 12.3 / 10.3, 13.5 / 9.5 | 6.9, 5.8; 98-99% | 19.6 / 3.8 mm, 0.35 deg; 19.9 / 3.9, 0.59 |
| gyro_drift, quiet / crowd | 12.5 / 8.0, 13.3 / 7.1 | 10.9 / 9.1, 11.3 / 6.9 | 5.7, 5.1; 90-91% | 13.8 / 3.3 mm, 1.32 deg; 12.4 / 3.7, 1.34 |
| dropout, quiet / crowd | 32.4 / 28.9, 14.2 / 7.0 | 45.9 / 43.0, 11.9 / 7.7 | 5.9, 4.9; 94-96% | 6.4 / 16.5 mm, 0.29 deg; 5.4 / 3.9, 0.21 |

- Every drive arrived, in every run; no scan was rejected or lost, no reset.
- Pose jumps: the filter's largest 3.8-5.1 mm in a tick (p99 1.3-2.0 mm) against the
  scan matcher's 11.9-15.4 (p99 6.2-6.5) in normal conditions; in the spill and
  dropout runs both jump when a slide comes to light (the filter 11-41, the matcher
  13-46 mm).
- On the same data the filter's worst position error is smaller in all 22 runs. The
  quiet dropout runs differ between drivers (43.0 against the matcher-driven run's
  32.4 mm) because the gaps fell at different moments of different drives: in each
  run the filter was ahead.
- **The drifting gyro's heading**: the filter's worst heading error is 0.26-0.30 deg
  against the matcher's 0.22-0.24, and NEES 90-91% in the band. Its bias walk (2e-4
  rad/s per sqrt(s)) is slower than this drift (the condition's is 9e-4); not tuned to
  it (the plan: tuned on seed 0's normal conditions only). The position does not
  suffer (the worst 6.9-9.1 mm against 10.9-13.3).
- The worn tyre and the drifting gyro's docking errors (19-20 mm lateral, 1.3 deg)
  are the same with either driving: they come through the navigation's own use of the
  odometry (step 2), which the filter does not change.
- Cost (Python): 0.06 ms per tick (the prediction 0.007 ms of it), 0.17-0.22 ms per scan
  besides the match; the match itself (2.9-3.6 ms) runs twice in shadow mode.


### Live wiring (step 5)

`slam_node.py` (parameter `fusion`), `localization_monitor_node.py`, `fusion:=` in
`mobile.launch.py` and `mobile_job.launch.py`, `localization=` in
`scripts/mobile_pickplace.sh` (`localization:=ekf|icp` on its command line).

- **slam_node, localization mode**: both estimators run on every scan and every plant
  step. The filter takes the wheels and the gyro (`/sim/wheel_states`, `/sim/imu`)
  paired by the plant's step, as base_node does, with the step's own 20 ms as the tick;
  a scan waits until both the odometry and the filter have reached its capture time.
  With `fusion:=ekf` the map > odom transform is the filter's pose times the inverse of
  the odometry at the same step (still sent at 20 Hz), so nothing downstream changes;
  with `fusion:=icp` it is the scan matcher's, as before. `/slam/pose` is the driving
  estimator's, once per scan, as before (the safety layer's localization watchdog
  reads it). New: `/slam/covariance` (PoseWithCovarianceStamped, the filter's pose and
  covariance every step, for logging and later the navigation) and `/slam/estimates`
  (validation: both estimates and the covariance every step). The start of each slip
  episode and every reset or late scan is logged. Mapping is unchanged.
- **localization_monitor_node** still scores the pose the stack uses (map > odom times
  the odometry) and now also both estimators from `/slam/estimates` against the truth
  (`loc_scoring.py`, with the filter's NEES): `estimates.csv`, and a line for each at
  the end of the run.
- Cost: each scan is matched twice (2.5-3 ms each); the filter adds 0.06 ms per step.

### Live runs and the default (steps 6 and 7)

The full job (8 boxes), real time, windows on, the filter driving and the scan matcher
in shadow; `scripts/dev/loc_live.py` scores both from the run's telemetry
(`results/ekf/live/*.json`, `figures/ekf/live_crowd.png`):

| run | boxes, time | drives docked, worst | icp p50 / p95 / max mm | ekf p50 / p95 / max mm | jumps max icp / ekf mm | ekf NEES in band |
|---|---|---|---|---|---|---|
| quiet | 8, 10.9 min | 19/19, 7 mm, 0.43 deg | 4.2 / 7.5 / 13.2 | 3.9 / 5.6 / 7.9 | 13.6 / 4.2 | 98% |
| crowd (two people) | 8, 20.4 min | 19/19, 8 mm, 0.47 deg | 3.8 / 7.2 / 11.3 | 3.4 / 5.0 / 7.5 | 14.7 / 4.2 | 98% |
| quiet, scan dropouts | 8, 11.4 min | 19/19, 7 mm, 0.34 deg | 4.2 / 7.5 / 52.1 | 3.9 / 5.5 / 49.1 | 49.2 / 46.0 | 97% |

(Docked worst: lateral and heading at the docks; parked at home 0.48-1.65 deg off,
known_issues T1.) No protective-field intrusion or contact; every scan accepted; scans
reached slam_node 17-21 ms p50, 27-30 ms p95 after capture. With dropouts the
localization watchdog stopped the base 45 times; the worst error in both is the slide
of such a stop, unseen until the scans came back.

The gates (all in `handover_notes/ekf_plan.md`), the filter against today's scan
matcher:
- Normal conditions no worse: p95 4.9-5.6 mm against 6.7-7.5 (offline and live), worst
  7.0-8.1 mm (gate 15), every drive arrived. Docked within 5.8 mm and 0.25 deg offline;
  live one place docking at 7.8 mm against the gate's 7. At the docks both estimators
  sit the same 1-3 mm off the truth (the map's own offset there), and runs with the scan
  matcher driving docked up to 7-8 mm before; taken as within today's.
- Smoother: the largest jump 3.8-5.1 mm against 11.9-15.4 (p99 0.6-2.0 against 6.2-6.8).
- Degraded conditions: a smaller worst position error in every run on the same data,
  offline and live. The worst heading error under a drifting gyro is the exception
  (0.26-0.30 deg against 0.22-0.24, known_issues U1); taken as passing on position, the
  error the gates are stated in elsewhere.
- Consistent: 90-99% of ticks in the NEES band (gate 90%), 97-98% live.
- Cheap: 0.06 ms per tick (the prediction 0.007 ms), 0.2 ms per scan besides the match.

So the filter is the default now (`localization=ekf` in `mobile_pickplace.sh`,
`fusion:=ekf` in the launches); `localization:=icp` brings back the scan matcher alone,
which keeps running in shadow either way.

## Docking facing the pick table

Plan: `handover_notes/pick_dock_plan.md` (local only). The pick dock was side-on (the
cell's pose: the base heading along the pick table, the table on its left): out of
the place dock the base turned about 180 deg, then lined up along the pick table from
its south end (a turn of 100-135 deg at a waypoint beside it) and drove in along it.
Live, the drives to the pick dock turned 130-215 deg more than planned with people.
Now the robot faces the pick table as it faces the tray's; the tables stay.

### The dock and the pick zone there (step 1)

`scene.py` (`PICK_DOCK_BASE`, `PICK_DOCK_ARM`, `MOBILE_PICK_ZONE_BOUNDS`, `pick_zone()`),
`task_node.py` and `sim_sensors_node.py` (the zone by layout), `mujoco_sim_node.load_scene_model`
(`dock=`: the stations layout in the arm frame at a dock, for offline tools),
`wrist_cam_check.py --mobile`, `mobile_scenarios.DOCK_BLOCK`.

- **The dock**: base_link at (0.94, 1.95), heading west: the chassis 5 cm off the pick
  table's edge (as at the tray table), centred on the pick zone. From the table's taught
  footprint.
- **The pick zone there** (the mobile job's; the parked cell keeps its own): the table's
  top 3 cm in from its edges and the old zone's width along them, in the docked arm's
  frame: x -0.25..0.25, y -0.66..-0.28 m (ahead of the arm). The pile lies centred 0.31-0.47
  m ahead.
- **Reach** (`dock_reach_check` machinery, tool down at every working height): 99.5%
  of the zone, all of where the boxes lie; from the side-on dock 90.6%.
- **The wrist camera**: the pile scans see the whole zone, 4-24 heightmap cells on the
  robot (the cell's pile scan: 1882-2185, the chassis under its zone's edge). The first
  scan, planned for the tallest pile the zone allows, was 12 mm out of reach at 0.35 m;
  the zone takes piles up to 0.30 m there (the pile is 0.21 m).
- **dock_block** moved into the new docking line (1.9, 1.95).

Offline (`nav_sim.py`, 20 drives between the tables, the carry pose with a box):

| | side-on: to pick / to place | facing: to pick / to place | per box |
|---|---|---|---|
| quiet | 30.2 s, 7.9 m / 23.7 s, 7.1 m | 24.8 s, 6.1 m / 22.8 s, 5.9 m | -6.3 s, -3.0 m |
| the job's crowd of four | 44.8 s / 37.0 s | 37.7 s / 39.6 s | -4.5 s |
| dock_block | 57.1 s / 28.7 s | 55.7 s / 24.4 s | -5.7 s |

Every drive docked, within 7 mm; no intrusion or contact. With the crowd the drives to
the tray table got slower: the middle loop's walker turns at (2.2, 2.0), where the base
now backs out of the pick dock and turns (the crowd's paths were drawn for the old
layout; kept).

### Docks taught by driving (step 2)

`nav/docks.py`, `nav_sim.py --teach`, `nav_node.py`; `data/maps/<map>_docks.yaml`.

Before, the method's docks came from `scene.py`'s layout (the simulator's own table
placement) moved into the map frame: a stand-in for teaching. Now they are taught as on
a real robot: at commissioning a technician parks the robot at each station by eye, and
the robot saves its own localized pose (the filter's, map frame) as the dock, next to
the map. `nav_sim.py --teach` is that drive offline: the navigation steered on the true
pose (the technician's eyes, as the mapping drive's joystick stand-in) to each station's
spot, docked within 1-4 mm of it, the filter's pose there saved. `nav_node` and
`nav_sim` load only the taught docks (an untaught map is an error, never a fallback to
the layout); home is the map's origin (the map is drawn from home). The method side no
longer reads any station or dock pose from the layout: only the plant, the scenario
actors and the validation monitors do.

With the taught docks (20 drives each): quiet, docked within 6 mm and 0.42 deg of the
technician's spots (the taught pose carries where the technician stopped and what the
filter made of it, -0.2 deg at the pick table), the drives as in step 1; the crowd of
four, seeds 0 and 1, within 7 mm, a box's two drives 74-77 s.

### The arm at the facing dock (step 3)

`dock_reach_sim.py` (its pick scenarios carried to the facing dock: the pile scan by
task_node's rule over the new zone for a 0.20 m pile, the lift over the same table spot
as measured live at the side-on dock). The arm's moves under its MPC while the base
docks or undocks (no contact, no solver failure, the tool within 6 mm of its reference):

| | side-on dock | facing dock |
|---|---|---|
| carry pose to the pile scan, unfolding on the approach | 10.0 s | 11.3 s |
| the same, docked first | 12.7 s | 14.8 s |
| a box from the lift to the carry pose, backing out | 7.2 s | 7.3 s |
| the same, the arm first | 9.7 s | 11.1 s |
| the arm's closest to the table and the pile (unfolding / with a box) | 286 / 55 mm | 155 / 55 mm |

Joint 1 swings about 170 deg from the carry pose to the pile (60 before), over the
robot's left side; the overlap with docking absorbs most of it: about 1.3 s more per
box against the base's 6 s less. The tray side is unchanged.

### The way in: an arc onto the docking line, rolling on (step 4)

`nav/planner.py` (`rotate_straight_arc`, `plan(entry=)`), `nav/reference.py` (`v_end`),
`nav/navigator.py`.

With the facing dock the route still went to the pre-dock point, stopped, turned in
place onto the docking line (35-40 deg, the hook in the paths), settled half a second
and started the approach. Now, for a station goal, the planner's shot to the pre-dock
pose is first rotate-straight-arc: one turn in place, straight, then a 0.6 m radius arc
that ends tangent to the docking line (on it, or 0.3-1.0 m before the pre-dock point,
the shortest that is free, with at most 150 deg of arc), so the robot arrives lined up;
rotate-straight-rotate stays the fallback. The route keeps the approach's speed (0.3
m/s) at its end instead of braking to a stop, and arriving within the alignment
tolerance (2 cm, 1 deg) it rolls straight on into the approach, which takes its first
command in the same control step. Not lined up (people pushed it off): the align step as
before. From the tray table to the pick table that is: back out, one turn of about 140
deg, straight across, an arc onto the line, straight in.

Offline (20 drives between the tables, the carry pose with a box; every drive docked, no
protective-field intrusion or contact):

| | side-on | facing, stop and turn (steps 1-2) | facing, arc and roll on |
|---|---|---|---|
| quiet: to pick / to place | 30.2 / 23.7 s | 25.0 / 22.9 s | 21.4 / 19.3 s |
| the crowd of four, seeds 0 / 1, a box's two drives | 81.8 / - s | 77.3 / 73.9 s | 77.6 / 74.6 s |
| walkers, dock_block, step_in: a box's two drives | - | - | 90.2, 80.8, 66.3 s |
| docked within (the technician's spot) | 6 mm | 6-7 mm | 6 mm quiet, 5-9 mm with people |

In the quiet room a box's two drives take 40.7 s instead of 53.9. With the crowd the
total is as in steps 1-2: the drives to the pick table got slower (45 s, one 86 s) and
those to the tray table faster: the middle loop's walker turns at (2.2, 2.0), where the
docking line starts, and when they stop there to give way the robot goes round them
(146 protective stops and twice the planned length in the 86 s drive); kept, a fair case.

### Live (step 5)

The full job, real time, windows on, the filter localizing:

| | side-on (earlier today) | facing, arc and roll on |
|---|---|---|
| quiet: boxes, time, per box | 8, 10.9 min, 82 s | 8, 8.8 min, 66 s |
| quiet: drives to pick / to place | 30-33 / 24 s | 21.0-21.4 / 19.7-20.0 s |
| quiet: docked (19 drives), align steps | within 7 mm, 0.43 deg | within 6 mm, 0.56 deg; none |
| the two-person crowd | 8, 20.4 min | stalled after four drives (one box), stopped by hand; with the considerate crowd 8, 12.9 min |

Every pick at the facing dock grasped first time (sensed tops 1.2-5.2 mm from the TCP).
The crowd run stalled on a mutual wait (known_issues V1): the middle loop's walker stood
by the pick dock's line about 1 m from the arm as the robot backed out; the arm's
supervisor held the arm, which pauses the base, and the walker waited for the robot.


## Considerate people in the job's crowd

`mobile_scenarios.py` (`JOB_CROWD`, `PAUSE_S`, `WAIT_M`, `PARKED_S`, `PATIENCE_S`).

The scenario people were written to test the robot's safety: two never gave way, those
who did stood in front of it up to 3 s and then walked on into it, one stood by the pick
table 30 s at a time. With the pick dock's line pointing into the room, a walker stopped
about 1 m from the arm as the robot backed out; the arm's supervisor held the arm, which
pauses the base, and the walker waited for the robot: the live crowd run stalled.

Now the job's crowd is considerate, as people at work are, without fleeing the robot:
- They walk their own paths and pay a moving robot no attention; if its body is right in
  their way they step aside after 1 s (the obstructive people after 3 s) and walk round.
- They stand 6 s at each of their points (they move less), and the bystander by the pick
  table's second spot is away from the docks.
- They do not stand within 2.5 m of a parked robot (it moved under 5 cm for 3 s: docked,
  working; 2.5 m is beyond the arm's holds): they skip such a spot of theirs, or leave it
  once the robot has parked there, waiting no nearer than that if every spot is.

A first version kept people 1.5 m from the robot's centre at all times, walking round it
wide and stepping back whenever it came towards them: the robot drove as in a quiet room
(protective stops 5-9 per 20 drives), but people visibly fled it; dropped. The
obstructive people stay as tests: `walkers` (two of its three never give way),
`dock_block`, `step_in`, and the stations crowd of the people-detection evaluations.

Offline (20 drives between the tables, the carry pose with a box; every drive docked, no
intrusion or contact; quiet: 21.4 / 19.3 s):

| the job's crowd | to pick / to place | protective stops | closest to the chassis |
|---|---|---|---|
| four people, seeds 0 / 1, as before | 45.0 / 32.6 s, 42.3 / 32.3 s | 425, 430 | 106, 96 mm |
| four people, seeds 0 / 1, considerate | 35.1 / 29.0 s, 38.7 / 30.0 s | 94, 150 | 101, 147 mm |
| two people (the script's crowd), seeds 0 / 1 | 37.4 / 27.5 s, 39.0 / 28.5 s | 78, 107 | 110, 133 mm |

Live, the full job with the script's two-person crowd and the facing pick dock: 8 boxes
in 12.9 min (97 s per box; side-on with the old crowd this morning, 20.4 min), 19/19
drives docked within 7 mm, no protective-field intrusion or contact, people walking past
the moving robot as close as 90 mm, 44 arm holds (43 with a person within 2.5 m).

## Tighter packing, placed right first time

Plan: `handover_notes/tight_packing_plan.md`. Boxes in the tray were not closely packed:
every box is aimed 5 mm from its neighbours and the walls, and lands a few mm off. Wanted:
set down right first time in a tighter spot, pushes only where lowering straight down
cannot work.

### Error budget (step 1)

Validation lines, read by no node: the sim logs, at each release, the released box's,
the TCP's and every box's true pose in the arm frame, and the released box's pose once
settled (`truth at release`, `released ... settled: ... at`); the task node logs what it
aimed at (`placement aim`: the box's spot, the TCP target and heading, the sensed in-hand
offset, the floor boxes as the packer had them). `scripts/dev/place_budget.py` splits
each set-down's landing error exactly into the arm's tracking (true TCP - aimed TCP), the
in-hand offset (true - sensed, sensed once at the grasp from the pile scan) and the shift
at release, and compares the neighbours, sizes and gaps with the truth.

Live, the mobile job, 1+1 (quiet: 8/8 on the floor, 1 push; crowd: 7 on the floor and one
stacked, 2 pushes), 16 set-downs, mm (sd x / y unless said):

| source | error |
|---|---|
| landing, total | 1.4 / 1.9 (p95 2.2 / 2.8) |
| the arm's tracking at set-down | 1.6 / 1.7 (p95 3.1 / 2.7) |
| in-hand offset | 1.0 / 0.5 |
| shift at release | 0.0 / 0.1 |
| yaw off square | 0.5 deg (tool 0.2, in hand 0.4): 0.8 mm at the far corner, p95 1.7 |
| neighbours as the packer had them | 2.9 / 1.3 (median 1.0 / 0.7, p95 5.1 / 2.4) |
| sensed footprint sides (pile scan) | 1.5 / 1.4 |
| box-to-box gap, true - planned (planned 5) | +1.3, sd 1.8, p95 4.6 (true median 5.9) |

- **The arm's tracking dominates, and it repeats**: the same box at the same spot had the
  same TCP error in both runs (+1.5/+2.2 and +1.5/+2.3 mm; -2.7 and -2.7 mm in y). The
  controller's own TCP error is a steady 2-4 mm in the last 3 s before release. Its
  offset-free correction (`mpc_controller_node._update_goal_bias`) integrates only while
  the whole reference stands still and resets when the goal moves by over 1 mm, so on the
  way down (the goal moving in z) the sideways offset is never removed; at touch-down
  friction holds the box where it is.
- Release, the tool's yaw and the in-hand yaw are small (0.5 deg at most, under 2 mm at a
  far corner).
- **A push turns the box and leaves its record off**: one push turned a box 3.3 deg; the
  re-measure then had it 5 mm off, and the next box planned against it landed 6.3 mm
  away. The neighbours' p95 is these records.
- The in-hand offset is the second term. The wrist camera cannot range the held box (its
  top is 0.13 m from the camera, the depth near limit 0.30 m); its colour image shows the
  box's far top edge and both side edges at ~0.57 mm a pixel, for boxes over ~64 mm
  along the camera's offset (the near edge is under the tool): a colour-edge measurement
  is possible for the larger boxes only (`wrist_cam_check.py`, `tray_held`).

Removing the tracking term leaves the in-hand offset, the neighbours and the sizes:
about 1.6 mm sd on a box-to-box gap, short of the ~1.4 mm the plan's gates need together
(gap 3 mm, under 2% brushing). Step 2 first removes the tracking offset, then measures.

### The arm's sideways offset kept on the way down (step 2)

`mpc_controller_node._update_goal_bias` now splits the offset-free bias: the sideways
part (x/y) integrates while the reference's x/y stands still and the arm is within 1 cm
of it, also while the reference moves straight down (a set-down, a pick's descent), and
resets only when the x/y goal moves on; gain 4/s (the offset changes with the arm's pose
along the descent, ~1 mm/s: at 1.5/s it lagged ~0.7 mm). The height part keeps the old
rule (the whole reference still, the arm nearly at rest). The controller's own TCP (from
the encoders) matched the true TCP within 0.1 mm at every release, so the error is the
controller's to remove, no truth needed. Contact ends a descent within ~60 ms (the
wrist-load rule) and the reference then freezes at the measured TCP: the bias cannot
wind up against friction.

Live, 1+1 each (mm, sd x / y; 16 set-downs each):

| | before | x/y bias kept, 1.5/s (quiet only) | x/y bias kept, 4/s |
|---|---|---|---|
| tracking at set-down | 1.6 / 1.7 | 0.4 / 0.4 (mean +0.4 / +0.6) | 0.2 / 0.2 |
| landing | 1.4 / 1.9 | 1.0 / 1.3 | 0.9 / 0.9 (median 0.3 / 0.7) |
| in-hand offset | 1.0 / 0.5 | 0.7 / 1.2 | 0.9 / 1.0 |
| time per box, quiet / crowd | 63 / 103 s | 63 s | 62 / 102 s |

No wiggles in the quiet runs, no retries or sideways loads at set-down. The in-hand
offset (the pile scan's edges, 2-4 mm off on some boxes) is now the landing error.

The box-to-box gap did not tighten (true - planned sd 1.8 mm): the neighbours' records
are off. The first box (set down 46 mm from flush so the wrist clears the wall, then
pushed back) stops 39 mm short of the push's aim at 40 N in all three runs that placed
it there: the tool, tilted 10 deg, drags the box into the next wall and it jams. It is
left ~30 mm from flush, but its record says flush (`_clearance_and_pushes` adds the
whole push), and the re-measure (6 mm window, 15 mm at most) corrects it ~9 mm a scan:
the next two boxes are planned against a record 10-20 mm off.

### Pushes: the record where the box stopped, no sideways drag (step 2)

- `task_node._record_push_end`: when a push leg ends (stalled or not), the pushed box's
  record along the push is set from where the tool stopped, its side on the box's face
  (the tool's section along the push is its radius at any lean, the lean being across
  the push), no longer from the planned push length; the next tray scan re-measures that
  box once with a wider window (15 mm, corrections up to 25 mm believed).
- `mpc_controller_node._update_goal_bias`: the sideways bias is per axis, and an axis
  standing still is also held while the reference moves slowly (under 0.06 m/s, a push)
  along the other. The jam was this: in contact the tool slid 7 mm sideways along the
  pushed face (the controller's model has no contact), and the friction dragged the box
  into the next wall and pressed it down (-7 N sideways, -11 to -21 N down at the wrist),
  stopping it at 40 N.

Live 1+1 (quiet / crowd, 8/8 each, 62 / 107 s per box): four pushes, all to their end,
stopping at 9-20 N (before: 40 N, 18 of 46 mm); records of the neighbours as the packer
had them sd 1.2 / 0.9 mm, p95 3.2 / 1.9 (before 4.4 / 1.6, p95 10.7 / 4.2), the pushed
box's record within ~3 mm after the push (the re-measure's correction). Both runs picked
a different first box, so the 46 mm push with the tool tilted 10 deg did not recur; the
steep (25 deg) pushes turned their box ~2 deg.

What is left on a box-to-box gap (true - planned, sd 2.2 mm, mean +1.0), split exactly
per pair: the landing (sd 0.9), the neighbour's record (sd ~1.2) and both boxes' sensed
sizes (sd ~0.9 each; the pile scan reads them ~0.5-1 mm large, so the gaps come out wider
than planned): the depth scans now set the spread, not the arm.

### Block packing at a 4 mm gap (step 3)

`packing.plan_compact(group_first=True, strip=True, needs_push=..., push_area=...)`: the
boxes grow as one block from the tray's far corner, measured by how far the block reaches
from the far short wall (it fills the tray's width first, so the free space stays one
rectangle), and a spot needing a wall push (`task_node._needs_push`: the wrist cannot
lower the box flush at any heading that turns it so) counts 50 cm2 more block. The
planned gap (`PLACE_CLEARANCE_M`) is 4 mm. Ranking by the block's bounding rectangle
instead grew it square and left an L of free space: all 8 on the floor in 82-86% of
pick orders; a free square reserved for the largest box the spec allows helped little
(to 88%); dropped.

Offline, 201 pick orders, the measured landing error (`packing_compare.py 200 3 --land
0.0009 --clear 0.004 --strip --push-area 0.005`):

| rule | all 8 on the floor | pushes per job | pushes without room for the tool | largest free space |
|---|---|---|---|---|
| today's (corners first, 5 mm) | 92% | 3.5 | 1.08 | 111 cm2 |
| block by strip, 4 mm | 100% | 3.0 | 1.12 | 126 cm2 |
| ... with the push penalty | 97% | 1.9 | 0.24 | 121 cm2 |

The layouts (`figures/packing/block_strip.png`) look only a little tighter: these 8 boxes
cover 69% of the tray and two of them cannot sit side by side across its width, so
every rule spans most of its length. Live (quiet): the true box-to-box gap median 4.3 mm
(6.5 before), 62 s per box.

### The held box slid into place before release

The user's choice, as industrial cells do: a guarded sideways move of the
held box. After touch-down (`_touch_down`, the box height measured) `_slide_legs` lifts
the box 5 mm (2 mm left it on the floor: the arm sags a few mm under it, and the drag
read as a touch), slides it at 8 mm/s towards each side it was planned flush with (a
wall or a placed floor box; one side per axis) up to the planned gap plus 6 mm (the pile
scan reads boxes ~1 mm large: 3 mm reached the wall just as the slide ended) and within
the wrist's limits, and `_check_slide_contact` stops it when the wrist's sideways load
along the slide (tared from the 3-tick average at its start) holds 1 N for 3 ticks (under
the ~2 N that drags a 0.2 kg neighbour); it backs off 1 mm (0.5 mm left it rubbing the
face it touched, and the friction on the next slide read as a touch), lowers to the
touch-down height and releases. The record moves by where the box really went. Not
along an axis the wrist set the box down off, and not at all for a box to be pushed
(pressed against a wall, the push dragged it along it).

The controller's sideways bias got two guards on the way: it integrates only while the
arm follows the reference in height (a descent blocked on a neighbour's edge wound it to
16 mm, and the arm never settled), and an error that unwinds it always counts. A box
beside a taller placed box now settles above the spot and goes straight down
(`RELEASE_BOX_MARGIN_M`, as near walls): the fast pass-through descent lagged sideways and
caught a neighbour's top edge twice in a run.

Live (quiet; mm, true poses): the box-to-box gap at each set-down median 0.8 (p5 0.2,
p95 5.9), in the final tray median 1.2 (most 0.2-1.6); 64 s per box (62 before); no
blocked set-down; no box disturbed. The wrist-limited corner push still jams (40 N, the
box 30-40 mm short; `scripts/dev/push_sim.py` does not reproduce it), and its box takes
the space one more box needed: 7 of 8 on the floor.

## Docking without turning on the spot

Plan: `handover_notes/nav_crowd_turning_plan.md`. The user, watching the crowd job: the base
turns on the spot to line up with the pick table's dock.

Measured (nav_sim, the job's crowd of four, carry pose with the largest box, seeds 0/1,
20 drives each; `nav_scoring.turning_summary`, the heading travel by navigator state and
the navigator's counts): extra turning per drive median +121/+122 deg, max +307/+330;
981/1108 deg of it in `align` (21/24 turn-straight-turn aligns, never in a quiet room),
5245/5522 deg on routes (3618 quiet: ~1700 extra), no replans, no step-backs.

- **Aligns were for tiny misses**: a route bent round people ends 2-5 cm and 1-2 deg off
  the docking line (`align_from`), just over the old tolerance (2 cm, 1 deg); the align
  then turned on the spot, drove, turned again, and often missed the 1 deg once more.
  The pre-dock tolerance is now 3 cm and 3 deg (the 0.9 m straight approach takes up the
  rest: docked within 7 mm in the quiet runs, 12 mm on the short first drive from the
  start pose in the crowd, inside the 20 mm dock tolerance), and a larger miss joins the
  line on one forward curve (`planner.curve_onto_line`: a cubic Hermite from the robot to
  the line, its bend no tighter than 0.3 m, joining it 0.35 m or more before the dock,
  checked free on the costmap) rolling straight on into the approach; turn-straight-turn
  only where no curve fits.
- Offline after: no turn-straight-turn align in 40 crowd drives (2-3 curves per 20),
  extra turning median +98/+110 deg, drives 29.4 s (31.8), no intrusion or contact; quiet
  unchanged (+13 deg, docked within 7 mm).
- **Route weaving, not changed**: labelling the route's turning by the nearest person
  within 1.5 m (6 drives): beside someone standing 1349 deg, walking 357, nobody near 236.
  The job's people stand 6 s at points by the tables and the MPC steers round them. Tried
  and dropped: slowing (0.1 m/s) or stopping for a walking person predicted across the
  route (median +110 vs +112, drives slower); planning round people standing from the
  first plan and replanning when one stands near the route ahead (+148 vs +161, 38
  replans in 6 drives, longer detours: the turning moved into the routes).

Parking at home keeps 2 cm / 1 deg (`PARK_TOL_*`): no approach follows it, and the live
crowd run had parked 2.0 deg off with the docks' 3 deg (offline after: within 1.46 deg,
as T1).

Live, the mobile job (crowd of four): turning in aligns 37 deg per job (783-1049 before),
extra turning per drive median +89 deg, max +201 (+113-116, max 296-336), drives 32.9 s
(34.6-37.2), docked within 5 mm, 104 s per box, no intrusion or contact; quiet: docked
within 7 mm, +12-18 deg per drive, as before.

## Holding the mobile job until ENTER

`mobile_pickplace.sh` started on its own since 832ac75 (`AUTO_START=1`); it holds for
ENTER again, like the cell's scripts. Even holding, the job looked started: the launch
log kept scrolling over the prompt (92 lines in 40 s, mostly the arm supervisor's holds
for people walking near), and the people walked from launch. Now `_run_scenario.sh`
stops showing the log while it waits (it resumes from that point after ENTER), drops
keys typed during start-up (a stray ENTER had started it), and the job's
`mobile_scenario_node` (`wait_for_go`, set in `mobile_job.launch.py`) keeps the people
standing at their starts until `/mpc/go`. `job_run.sh` sends `/mpc/go` itself, so the
batch runs are unchanged; `mobile.launch.py` (no arm controller) leaves it off.
