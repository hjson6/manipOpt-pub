# System overview: a mobile manipulator that navigates among people and packs boxes

Current state: a Franka Panda arm on a differential-drive mobile base does the
whole job in a 7.0 x 5.6 m room with people walking about: it maps the room with its own
lidar SLAM, localizes with a filter fusing wheels, gyro and scan matching, tracks the
people, drives between the pile's table and the tray's (Hybrid A\* and a base MPC, a
separate safety layer) at up to 1.0 m/s, docks at each table to a few millimetres
without turning on the spot, and moves 8 boxes from the pile into the tray, packed as
one block about a millimetre apart; the arm, driven by torques from its own MPC, folds
and unfolds while the base undocks and docks. The same arm task also runs with the base
parked between the tables (the cell). Source material for the README and a
presentation: what the system does, how each part works (intuition first, then the
technical detail and the maths), which libraries it uses, and what was measured.
Design history and rejected alternatives are in [`design_notes.md`](design_notes.md);
per-module reasons in [`implementation_notes.md`](implementation_notes.md); open
problems in [`../known_issues.md`](../known_issues.md).

---

## 1. What it does, in one minute

A simulated Franka Panda arm on a differential-drive mobile base empties a pile of
eight mixed-size boxes and packs them into a tray, in a 7.0 x 5.6 m room with people
walking about, driving between the pile's table and the tray's on opposite walls (the
mobile job). The same arm task also runs with the base parked between the two tables
(the cell).

- **The mobile job**: the robot maps the room once with its own lidar SLAM, then
  localizes in that map (a filter fusing wheels, gyro and scan matches); it plans a
  route (Hybrid A\*), follows it with a second MPC that slows near people and keeps its
  distance, joins each table's docking line without turning on the spot and docks to a
  few millimetres, and has a separate safety layer that stops the base for anything it
  is about to hit. The arm folds the box over the base's rear deck while it drives.
- **Nothing about the boxes is given to the robot.** A camera on the wrist
  looks at the pile, works out which boxes are on top and how big they are,
  and picks the biggest one first. The pick lowers until the wrist's load cell
  feels the box, and the box's weight is checked after the lift. The tray is
  scanned again before each placement.
- **The arm is driven by torques from a model-predictive controller (MPC)**
  that re-plans the next 0.3 s of motion 50 times a second, with the arm's
  dynamics, joint limits, torque limits and collision constraints inside the
  optimisation. The simulated arm differs from the controller's model, and its
  sensors are noisy, as a real arm's would be.
- **Real time.** The physics runs on its own clock and never waits for the
  controller; a late answer is simply late.
- **Boxes are carried level** and turn with the base like on a palletising
  robot, so they arrive square to the tray.
- **Boxes are packed as one block from the tray's far corner**, 4 mm apart
  as planned, with a look-ahead at the boxes the camera can already see; the
  held box is slid into its neighbour or wall until the wrist feels it, then
  released; spots the wrist cannot reach flush are set down clear and pushed home.
- **People are sensed by two safety-lidar stand-ins** at the base (a ceiling
  camera is optional), tracked, and kept at a protective distance computed
  the ISO 13855 way: the arm slows as a person approaches, stops while they
  are too close, and resumes when they leave.

The mobile job: 64-65 s a box in an empty room, 104 s with four people walking, the
true gap between packed boxes 0.7-0.8 mm (median; section 10). A full cycle of eight
boxes in the cell takes about 190 s; with three visits from a person about 200 s.

---

## 2. The system at a glance

Ten ROS 2 nodes (Python, `rclpy`), plus framework-free libraries in `core/`
and `perception/` that the nodes call. The tenth, `base_node`, is the mobile
base's driver (odometry from the wheel encoders and the gyro, `/base/cmd_vel`
to wheel speeds). The mobile base's own stack (`mobile.launch.py`, sections 9a and
9b) adds `slam_node`, `people_node`, `nav_node` and `safety_node`; the mobile job
(section 9c) runs the arm's task on top of it. The plant is two processes (physics, and sensors and
windows); everything below the dashed line is the method.

```
 dynamic_obstacle_node ── /env/dynamic_obstacle (ground truth: the scenario, the monitor) ──┐
        │ (moves the person)                                                                 │
 mujoco_sim_node: PLANT physics ── /sim/joint_states (noisy encoders), /sim/wrist_force       │
        │      (noisy load cell), /sim/scene_state ──▶ sim_sensors_node: PLANT sensors        │
        ▲                                               wrist camera depth (/sim/*_depth),     │
        │ /sim/joint_command (torques)                  ceiling camera depth (/env/workspace_depth),
        │                                               lidar scans (/env/lidar_scan), windows │
 - - - -│- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - │
 mpc_controller ◀── /mpc/goal, /mpc/orientation_goal ── task_node ◀── /mpc/hold, /mpc/speed_scale
 (EKF + acados OCP)    (16-point reference horizon)     (perception, decisions,  ▲
                                                         Ruckig reference)       │
 lidar_detection_node ── /perception/lidar_detections ──▶ obstacle_supervisor_node
 camera_detection_node ─ /perception/camera_detections ─▶ (tracks, separation)
 (optional; kinematic robot mask)                       detection_monitor_node: sensed vs truth ◀┘
```

**One control tick (20 ms):**

1. The plant publishes the arm state (measured joint angles and speeds),
   stamped with its step number.
2. The controller updates a state estimate and predicts it to the moment its
   torque will start (14 ms into the next tick), solves the MPC problem
   (about 3 ms) and sends the first torque of the plan.
3. `task_node`, which also received the state, advances its reference and
   publishes the next 0.3 s of it.
4. The plant keeps stepping physics (2 ms sub-steps) on its own clock and
   switches to the newest torque 14 ms into the tick; if none has come, the
   old one runs on and the tick is counted late (2-5 late ticks in about
   10 000 per run).

---

## 3. The plant: `mujoco_sim_node` and `sim_sensors_node` (MuJoCo)

**Intuition.** This is the "real robot" and the "real cell" of the
simulation. It knows the true physics and the true positions of everything;
the rest of the system only sees what its sensors report.

**Technical.**

- **Room and cell**: MuJoCo's world is the room (7.0 x 5.6 m, walls 2 m high, a
  pillar and a shelf; origin at a corner, floor z = 0). The cell (two tables,
  the pile, the tray, the ceiling camera) stands in it at `CELL_POSE`. The arm
  is on a pedestal of a mobile base, its mount 0.75 m above the floor, level
  with the tables' tops; parked at the cell, the arm's base frame is the old
  cell's frame, which is the frame of everything the method does for the arm
  task. A full-size person (1.75 m, walking 1.2 m/s) is a mocap body moved by
  the scenario (room frame).
- **Mobile base** (MiR-like, 0.80 x 0.56 x 0.30 m, 110 kg): two driven wheels
  (r 0.10 m, track 0.50 m) on sprung mounts at the chassis's centre and four
  ball casters, rolling on the floor by contact and friction; the arm mounted
  0.20 m ahead of the wheel axle. The drives are torque motors with a speed
  loop run every physics substep; parked, a brake holds the wheels and their
  contact points stick to the floor until a force beyond friction; the wheel radii and
  track differ from the taught values by the seed (0.5%, 5 mm). Sensors:
  wheel encoders (16384 counts per revolution) and an IMU (a gyro with bias
  and noise, an accelerometer). The plant never publishes the base's pose to
  the method (`/sim/base_truth` is for validation only).
- **Physics**: MuJoCo (`mujoco` Python bindings), `implicitfast` integrator,
  2 ms timestep. The Panda model is MuJoCo Menagerie's with plain torque
  motors (limits 87 Nm on joints 1-4, 12 Nm on 5-7). The plant's masses,
  friction, damping and armature differ from the controller's model (seeded
  draw, `plant_mismatch.py`). Boxes and the tray have contacts and rest by
  gravity.
- **Timing**: physics on its own thread at real time; commands arrive on a
  separate thread and the newest one is switched in 14 ms into each tick.
  `lockstep:=true` (the plant waits for each command) remains for debugging
  only.
- **Torque interface**: `ctrl = τ` from the controller; with no command for 10
  steps (controller absent or dead) it falls back to a gravity-compensated PD
  hold, $\tau = g(q) + K_p (q_{hold} - q) - K_d \dot q$.
- **Sensors**: encoders with seeded noise (speeds by finite differences), a
  wrist load cell with noise and a bias (in the arm frame's axes), depth noise
  on both cameras, lidar range noise and dropouts; the base's wheel encoders
  and IMU (`/sim/wheel_states`, `/sim/imu`), stamped with the step.
- **Grasping** is a weld constraint (MuJoCo equality) between the gripper
  flange and the box, re-anchored at the current relative pose when engaged,
  so there is no snap. It engages only on the box whose top the tool is on
  (contact), never on the box nearest a requested point.
- **In-hand sensing**: after a grasp the plant publishes the box's true size and
  offset from the gripper point, for its log only; the method does not use them
  (it senses both, section 14).
- **The sensors process** (`sim_sensors_node`) keeps its own copy of the scene
  from `/sim/scene_state` (every tick) and renders from it, so drawing never
  delays the physics:
  - **Wrist camera** (`container_cam`, 320×240, 55° field of view, on the
    wrist housing's side tab 10 cm from the tool axis, looking straight
    down): depth on request.
  - **Ceiling camera** (`workspace_cam`, 3.5 m above the floor, 70°, 320×240,
    10 Hz; on with `ceiling_camera:=true`): raw depth only.
  - **Safety lidars**: two 270° scanners at the chassis's front-left and
    rear-right corners, 0.20 m above the floor, 0.5° steps, 15 Hz, ray cast
    from their sites on the moving base (`lidar_sim.py`): raw ranges only.
  - **MuJoCo viewer**, redrawn once per plant tick (50 Hz).
- **Decision window** ("pick & place decisions", OpenCV): pick-up on top,
  placement below. Each half shows the camera frame of that side's last scan,
  its depth image, and the decision `task_node` actually made, outlined at
  its sensed height; redrawn only when a scan or decision comes in.
- **Obstacle window** ("obstacle detection (live)", OpenCV; off with
  `obstacle_view:=false`): a top-down lidar panel (returns grey, foreground
  orange, people red, the supervisor's protective radius and its status:
  clear / slowed to N% / HOLD), plus, with the ceiling camera on, the
  detector's view of the camera image and the supervisor's tracks on its
  depth image. Windows are shown by a separate process
  (`window_process.py`).
- **Telemetry**: with `MANIPOPT_TELEMETRY_DIR` set, a CSV row per step (sim
  time, step period, late ticks, joint speeds, torques, angles, the person's
  pose) and snapshots of both windows.

---

## 4. Robot model and dynamics: `core/dynamics.py` (Pinocchio + CasADi)

**Intuition.** The controller needs its own mathematical copy of the robot to
predict what a torque will do. It is built from the same robot file as the
plant, but by a different library, just as a real controller's model is not
the real robot.

**Technical.** Pinocchio parses the MJCF; `pinocchio.casadi` builds a
symbolic Articulated-Body Algorithm (ABA), giving the forward dynamics as a
CasADi expression:

$$\ddot q = M(q)^{-1}\big(\tau - D\dot q - C(q,\dot q)\dot q - g(q)\big),$$

with $D$ the joint damping, and the armature included in $M$ (verified: the
Pinocchio and MuJoCo mass matrices agree to 3e-8). State
$x = [q;\dot q] \in \mathbb R^{14}$, control $u = \tau \in \mathbb R^{7}$.
Forward kinematics of the tool point (`tcp_site`, 0.10 m past the flange),
of two tool axes (for orientation) and of the collision-proxy frames are
CasADi functions too, so acados can differentiate everything exactly.

The carried box is in the model as a payload at the tool point; its mass is
the wrist load cell's reading (filtered), so nothing about the box is given.

---

## 5. The controller: MPC with acados (`core/ocp.py`, `mpc_controller_node`)

**Intuition.** Every 20 ms the controller asks: *"which torques over the next
0.3 s keep the gripper on the planned path, the box level, and every link
clear of obstacles, without exceeding what the motors and joints can do?"*
It applies only the first torque, then asks again with the new measured
state. That is receding-horizon (model-predictive) control.

**The optimal control problem** (N = 15 steps of 20 ms, horizon 0.3 s):

$$\min_{x_{0..N},\,u_{0..N-1},\,s}\ \sum_{k=0}^{N-1} \big\| y(x_k,u_k,p_k) - y^{ref}_k \big\|^2_{W}
 + \big\|\dot q_N\big\|^2_{W_N} + \text{slack penalties}$$

subject to

- dynamics: $x_{k+1} = F(x_k, u_k)$, the Section 4 ODE integrated by an
  implicit Runge-Kutta (IRK) scheme over each 20 ms step;
- initial state: $x_0$ = the state estimate (an EKF on the measured joint
  angles) predicted 14 ms ahead, to when the torque will start;
- torque limits $|\tau_i| \le 87$ / $12$ Nm; joint position limits (from the
  robot file); joint speed limits 2.175 / 2.61 rad/s (the Panda datasheet), all
  hard;
- collision (soft, section 5.2):
  $\|p_j(q_k) - c_o\| - r_j - r_o - 0.03 \ge -s$;
- terminal goal (soft): $\mathrm{FK}(q_N) = g_N$.

The stage cost vector (nonlinear least squares) is

$$y = \big[\ \tau,\ \dot q,\ q - q_c,\ \mathrm{FK}(q) - g_k,\ \alpha(z_{tool}(q) - z^*),\ \alpha(x_{tool}(q) - x^*_k)\ \big]$$

with weights 1e-3 (torque), 1e-2 (joint speed), 5 (posture), **5000**
(position tracking), **2000** (both orientation terms); terminal joint speed
0.1; terminal-goal slack 1e4 (L1) / 1e5 (L2); collision slack 1e3 / 1e4.

- $g_k$ is the **reference position for the moment of stage k**: `task_node`
  sends the next 16 reference points, one per stage (the "reference
  preview"). Giving every stage the same current goal made the MPC plan as if
  the target stood still, and it trailed a moving reference by ~3 cm (5-8 mm
  of it sideways on every descent into the tray).
- The orientation terms keep the tool's z-axis pointing straight down (box
  level) and its x-axis at the planned heading $x^*_k$ (box square to the
  tray); $\alpha$ switches them on.
- The posture term pulls joints to mid-range, except joint 1, whose target is
  set every tick to the azimuth of that stage's goal ("the base faces where
  it is going", no recompilation needed since it is a cost reference).
- Online parameters $p_k$ (changed every solve without recompiling): three
  obstacle spheres, the stage goal, the two orientation targets and the
  orientation switch.

**Solver.** acados, SQP with real-time iteration (`SQP_RTI`: one Gauss-Newton
QP per tick, warm-started from the previous solution), partial condensing
into HPIPM (an interior-point QP solver), exact sensitivities from CasADi.
Code-generated C, cached by a fingerprint of the problem structure, so
changing only a weight never forces a rebuild. Solve time: median 3.0 ms,
p99 about 4.4 ms (10 runs, real time).

**Around the solve** (`mpc_controller_node`):

- Solves as soon as a new state arrives (no timer of its own), always from
  the newest state (best-effort, keep-last-1 delivery both ways), with the
  goal and orientation horizon for the step its torque will act in;
  extrapolated linearly from the two newest if that one has not arrived.
- Offset-free tracking: the remaining tool error is integrated into a small
  bias on the goal. Sideways (x, y, per axis) while the reference stands still
  or creeps along that axis, also while it moves straight up or down (a
  set-down) or slowly along the other axis (a push), and only while the arm
  follows in height (a descent blocked on an edge would wind it up); height
  once the whole reference stops. It removes a 1.6 mm (sd) sideways error at
  set-down that repeats with the arm's pose.
- Initial guess: the whole horizon set to "stay here, holding against
  gravity" ($x_k = x_0$, $u_k = g(q_0)$) at start and after any failure. The
  default all-zero guess puts joint 4 outside its range, a meaningless point
  to linearise at.
- Keeps solving at rest; "settled" (within 8 mm and 0.05 rad/s for 0.5 s) is
  only a signal to `task_node` that the arm has arrived.
- Fallback on a failed or late (> 50 ms) solve: ramp the last good torque
  down over 25 ticks, then let the plant's hold take over.

### 5.2 Collision avoidance: `core/collision.py`

**Intuition.** The arm is wrapped in four invisible spheres; obstacles are
spheres too. The optimiser must keep every arm sphere a margin away from
every obstacle sphere over the whole predicted motion.

**Technical.** Proxy spheres on link 3 (r 0.10 m), link 5 (0.09), link 7
(0.09) and the flange (0.08); three obstacle slots; 12 constraints per stage,
surface-to-surface distance minus a 3 cm margin. They are **soft** (slack
variables with L1 + L2 penalties): if an obstacle suddenly makes the current
plan infeasible, the QP still has a solution and the arm is pushed away
instead of the solver failing.

The three slots, filled by `task_node`:

1. **Pile hull**: one sphere around the sensed remaining pile, active while
   carrying a box away from it.
2. **Tray hull**: one sphere around the tray, active on the way back.
3. **Static obstacle** from the supervisor (an object in the way), used with
   a detour via-point.

The hulls **grow in** rather than switching on: a hull's radius never exceeds
the arm's current clearance from its centre (computed with the same proxy
spheres and margin), and never shrinks within a leg. Switched on at full
size while the arm was still above the pile, the violated constraint slammed
joints 2 and 4 to their torque limits for one tick on every box.

---

## 6. Perception: the wrist camera (`perception/heightmap.py`)

**Intuition.** Seen from above, a pile of boxes is a height map. Each flat
patch at one height is the top of a box that nothing rests on, so that box
can be lifted.

**Technical** (numpy only, no ROS or MuJoCo; unit-tested):

- **Depth to height.** The camera looks straight down, so height = camera
  height minus depth (MuJoCo's depth is along the optical axis). A dense grid
  of sample points every 5 mm covers the pile (80 × 83) or the tray (57 × 87).
- **Pinhole projection** of a world point into the image:
  $u = \frac{W}{2} + f\frac{x_c}{-z_c},\ v = \frac{H}{2} - f\frac{y_c}{-z_c}$,
  $f = \frac{H/2}{\tan(\text{fovy}/2)}$, with $(x_c,y_c,z_c)$ the point in the
  camera frame.
- **Parallax correction.** Which pixel a world (x, y) falls on depends on its
  height, which is what is being measured. So each point is projected at a
  reference height, its height read, re-projected at that height and read
  again (3 passes, vectorised, same frame). Without it, the flat part of a
  6 cm box top shrank to 4.5 cm and sat off-centre.
- **Segmentation** (`find_topmost_boxes`): connected components over
  4-neighbours that join only if their heights differ by ≤ 5 mm, so a short
  box next to a tall one stays two regions. A region counts as a box if it has
  ≥ 30 cells and fills ≥ 85% of its bounding box; its height is a robust
  inlier median. A visible top face is by construction an exposed box.
- **Pick rule**: the biggest footprint first (ties: the tallest), the classic
  decreasing-size heuristic; a top that touches a higher one waits, and so does
  one where the wrist would hit a taller neighbour. The grasp point is the
  centre of the sensed top face; the tool hovers above it, lowers until the
  wrist load shows contact, grips, and checks the weight after the lift. The
  box's height cannot be seen from above; it comes from the touch-down at the
  tray (the wrist load drops when the tray takes the box).
- **Camera pose** at every scan: from the measured joint angles and the
  hand-eye calibration. The tool's own pixels are masked from its known
  geometry.
- **Shadows.** A cell hidden behind a tall box never converges between floor
  and box top; a final consistency check reports unseen cells as floor.

---

## 7. Placement: block packing and a guarded slide (`pick_place_common/packing.py`)

**Intuition.** Pack the tray the way a person would: start in the far corner
and grow one block from it, filling the tray's width before moving along it,
glancing at the boxes still waiting on the pile so the next ones still fit.
Ask the robot whether it can really put the box there, and score the spot
where it would really end up. Then set the box down beside its neighbour and
slide it over until it touches.

**Technical.**

- **Candidates**: spots 4 mm (`PLACE_CLEARANCE_M`) from a wall or a placed box
  on both axes, the box either way round (0 or 90°).
- **Ranking**, lexicographic: most of the next three sensed boxes still fitting
  (each placed greedily by the same rule) → the smallest block, measured by
  how far it reaches from the far short wall (`strip`), a spot needing a wall
  push counting 50 cm² more (`needs_push`, `push_area`) → most of this box's
  perimeter touching walls or placed boxes (1 cm steps) → nearest to flush →
  nearest the tray corner farthest from the robot. Only the current box is
  committed; the choice is made again, with a fresh scan, for the next one.
- **The guarded slide** (`task_node._slide_legs`): after touch-down the box is
  lifted 5 mm and slid at 8 mm/s towards each side it was planned flush with
  (one per axis, within the wrist's limits), stopped when the wrist's sideways
  load holds 1 N for 3 ticks (under what drags a 0.2 kg neighbour), backed off
  1 mm, lowered and released; the record moves with it. Not along an axis the
  wrist set the box down off, nor for a box to be pushed. Beside a taller box
  the set-down settles above the spot and goes straight down.
- **Checked against the scan and the robot**, best first (`where`): the
  sensed tray heightmap must show the footprint clear down to the floor (inset
  one 5 mm sample), clear of any stacked box's overhang, and the robot must be
  able to get the box there: set down where the wrist clears the walls, then
  pushed home with the tool upright or at the smallest tilt that fits, with
  the tool, wrist and whole arm checked against the walls and the scan. Where
  a push is not possible the robot reports where the box would really be left,
  and that spot is scored again there.
- **Pushes**: a wall spot is set down clear and pushed back; a push stops
  when the box stops (guarded), then backs off with the force easing off, and
  the box's record is set from where the tool stopped (re-measured wider on the
  next scan). Shifts under 6 mm are left as a gap.
- **When the floor is full**: the lowest stable resting spot in the scan (on
  boxes, bridging boxes of one height), either way round, checked the same
  way.
- **Placing the box, not the gripper:** the target is the box centre,
  converted to a gripper target with the in-hand offset rotated by the
  gripper's heading at the tray: $p_{tcp} = c_{box} - R(\psi)\,o_{box}$. The
  box goes down until the wrist load drops (the tray takes it), which also
  measures its height.

Offline over 200 random pick orders (`scripts/dev/packing_compare.py`, with a
wrist model and the measured landing error): all eight boxes on the floor in
97% of orders (92% with the earlier corners-first rule), 1.9 pushes a job (3.5).
Live, the true gap between neighbours: median 0.7-0.8 mm (6.5 mm before).

---

## 8. Motion and timing: `task_node` (Ruckig) and real-time stepping

### 8.1 The reference: where the gripper should be, moment by moment

**Intuition.** The MPC tracks a moving target. `task_node` moves that target
along a smooth path with limited speed, acceleration and jerk, stops only
where the task needs it (grasp, release, the camera scans), and turns the box
with the base.

**Technical.**

- **Ruckig** (online, time-optimal, jerk-limited trajectory generation):
  0.5 m/s, 3 m/s², 20 m/s³ per axis. Each tick it advances one step from the
  current reference state to the current target.
- **Planned round the base, in cylindrical coordinates** $(\varphi, r, z)$
  instead of $(x, y, z)$: a swing from the pile to the tray is a straight line
  in these coordinates, i.e. a true arc with one accelerate/cruise/brake
  profile. The azimuth limits are the Cartesian ones divided by the leg's
  larger radius, so the gripper never exceeds them. State conversion:
  $\dot r = v\cdot e_r,\ \dot\varphi = v\cdot e_t / r,\
  \ddot r = a\cdot e_r + r\dot\varphi^2,\
  \ddot\varphi = (a\cdot e_t - 2\dot r\dot\varphi)/r$.
  The x/y/z arc-and-via-point path is kept only while a detour around a static
  obstacle is needed.
- **Pass-through waypoints**: the points above the box, the lift, the point
  above the slot and the lift-off are passed at speed (Ruckig target velocity)
  instead of stopping. Pass speed $0.5\cos(\theta/2)$ m/s for a turn of
  $\theta$. Arriving corners are passed moving straight down; departing ones
  are **blended** (the swing starts 5 cm before the top).
- **Heading of the box**: joint 7 ≈ joint 1 − heading − 135°, so a heading
  that turns in step with the base keeps the wrist still. A run of
  pass-through legs up to the next stop turns as one, towards the aligned
  heading at that stop (±90°, or the turned box's), shared out by each leg's
  base swing and linear in it; a turn more than twice the swing follows the
  distance instead. Among the equivalent headings ($\psi + 2\pi k$) the one
  nearest to turning with the base that keeps joint 7 within ±150° is used.
  Every leg is slowed so the tool turns (heading and tilt) no faster than
  1.5 rad/s.
- **Reference preview**: every tick, one Ruckig trajectory calculation on a
  copy of the input, sampled at 0, 20, …, 300 ms gives the 16 stage goals,
  and the heading at each.
- **People**: the supervisor's hold switches Ruckig to its velocity interface
  with a zero target (braking at the full 3 m/s², the stopping performance the
  separation distance assumes); its speed scale multiplies the velocity limit.
  On release the target is re-planned from where the reference stopped.

### 8.2 The task sequence

A state machine per box. First, once: the tray scan points → two pictures of
the place zone with an empty hand → the tray. Then per box: the pile's scan
point → scan → above the box (pass) → hover 15 mm above the sensed top (the
load cell tares) → down at 3 cm/s until the wrist load changes by 3 N →
grip (the spot in the tray is chosen now, from the last tray scan) → lift
(blend) → weight check (≥ 2 N, else re-scan the pile) → swing, holding the
pile's clearance height until clear of the pick zone → above the spot →
hover → down until the wrist load drops → release → pushes if planned →
lift-off → the tray scan point → scan with an empty hand (re-measures the
placed boxes) → back to the pile's scan point. Every height comes from the
last scans. A blocked place (no touch-down, or a sideways load: the box on a
neighbour's edge) lifts, rescans the tray with the box held (its silhouette
masked) and re-plans. It stops when the pile is empty, a pose is out of
reach, three picks in a row fail, or the tray is full.

### 8.3 Timing: real time

**Intuition.** On a real arm, time does not wait for the controller. The
simulation now behaves the same way.

**History.** First the plant and controller each ran their own 20 ms timer,
with a random phase between them; on some launches 25-35% of torques were a
step old and the start was flaky. Lockstep (the plant waits for each answer)
fixed that, but hid late answers, so it is now a debugging option only.

**What it is now:**

- The plant steps physics on its own thread at real time and switches to the
  newest command 14 ms into each tick; commands come in on a separate
  thread. A missing one leaves the old torque on and counts the tick late
  (2-5 per 10 000 ticks in the latest batch).
- The controller estimates the state with an EKF on the measured joint
  angles (the full model for the mean) and solves from it predicted to the
  moment its torque will start.
- States and commands go best-effort, keep-last-1: always the newest.
- Cameras, lidars and windows run in the separate sensors process; OpenCV is
  single-threaded and the software GL limited to two threads, so drawing
  never starves the plant or the controller (WSL renders on the CPU).
- In-process timeouts use a monotonic clock: WSL2 steps the system clock
  forward ~1.5 s every ~34 s.

---

## 9. Obstacle handling: people and objects (lidars, optional ceiling camera)

**Intuition.** Two scanners at the robot's base sweep a plane 20 cm above the
floor and see people's legs all round the cell. Anyone seen is a person: the
robot slows down and waits for them, never steering a box around them. With
the ceiling camera switched on, something small that stays still can also be
recognised as an object and planned around.

**Truth vs belief.** `dynamic_obstacle_node` moves the person model (full
size, 1.75 m, walking 1.2 m/s) and publishes ground truth. Only the plant
(to move the model) and the validation monitor read it. The robot learns
about people only through its sensors, so detection error, latency and
occlusion are real effects in the results.

**Lidar detection** (`perception/lidar_detection.py`, numpy;
`lidar_detection_node`): a background range per beam learnt from the first
scans of the empty cell; beams 8 cm nearer than it are foreground;
consecutive foreground beams within 10 cm make a leg, its centre the visible
arc's centroid moved back by the arc's mean depth; legs seen by both
scanners are merged, legs within 0.5 m make one person. Offline: 0 false
positives in 100 empty scans per scanner; detected wherever not occluded out
to 5 m; centre error median 24 mm.

**Camera detection** (optional, `ceiling_camera:=true`;
`perception/obstacle_detection.py`, `camera_detection_node`): every depth
pixel is unprojected to 3D; a pixel is foreground if above the floor (5 cm),
outside the masked pile, tray and furniture, and not the robot: the robot is
masked kinematically, the method's own robot model at the measured joint
angles (and the held box from its sensed size and offset) projected into the
calibrated camera. Foreground pixels form blobs, each a **column** (x, y,
radius, top height).

**Tracking and classification** (`perception/obstacle_tracking.py`, no ML),
one tracker per source: gated nearest-centroid association, alpha-beta
filter. Lidar tracks are always persons. A camera track is a **person** if it
moves, has ever moved, is ≥ 1.0 m tall, or is not yet confirmed; **static**
only after 1 s motionless. Uncertain means person.

**Speed and separation monitoring** (`obstacle_supervisor_node`): for each
person the gap from their column to four points on the arm is compared with
the protective distance (ISO 13855 style)

$$S = v_h (T + t_s) + v_r T + \frac{v_r^2}{2a} + C,\qquad t_s = \frac{v_r}{a}$$

with $v_h$ = 1.6 m/s, $v_r$ the arm's speed toward the person, $T$ = the
detection's age (at least its usual latency) + one frame period + two control
ticks, $a$ = 3 m/s² (the reference's braking) and $C$ = 5 cm. A lidar person's
radius is its legs' plus the standard's intrusion allowance for a scan plane
at height H, $1.2 - 0.4H$ = 1.12 m. Speed scale = 0 (hold) if the margin
$d = \text{gap} - S \le 0$, else $0.15 + 0.85\min(d/0.6\,\text{m}, 1)$, with
5 cm hysteresis on release; with both sources on, the most cautious counts.
Fail-safes: any enabled source silent for 0.5 s → hold; supervisor silent for
1 s → `task_node` holds. A static object (camera) goes into obstacle slot 3
and `task_node` plans a single detour via-point; `obstacle:=static` refuses to
start without the camera.

This is a software stand-in: the simulation has no certified safety scanner.

---

## 9a. Mapping and localization (mobile base)

**Intuition.** The two lidars at the chassis's corners see the room's walls, the
pillar, the shelf and the table legs. Driven round the room once at commissioning,
the robot stitches its scans into a map; afterwards it finds itself in that map by
matching each new scan against it, with the wheels' odometry as the starting guess.

**Technical.** Two options on the same input (`slam:=own|toolbox`, ours the
default). Both take the lidars' scans (slam_toolbox as one merged 360 deg scan) and
the odometry, and publish the map > odom transform (TF tree: map > odom > base_link
> arm_base, lidar_0/1, base_scan).

- **Ours** (`slam/`, `slam_node`): each scan matched by point-to-line ICP against
  the latest keyframes, starting from the odometry's prediction; a keyframe every
  0.3 m or 15 deg; loop closures checked by ICP against older keyframes nearby, then
  the SE(2) pose graph optimized by Gauss-Newton. The map is a 5 cm occupancy grid
  (log-odds, free space traced from each scanner) that also keeps each cell's mean
  hit position. Localization matches each scan by ICP against those positions.
- **slam_toolbox** (2.6.10, robostack): its shipped mapping (online sync) and
  localization parameters adapted to this robot.
- **Map once, localize after**: a commissioning drive builds and saves the map,
  anchored at the dock; jobs localize in it starting from the dock.
- **Localization by sensor fusion** (`slam/base_ekf.py`, the default):
  an extended Kalman filter in the map frame (pose, speed, turn rate, the gyro's
  bias). The wheels' speed and the gyro's turn rate predict at every plant step
  (50 Hz); each scan's ICP match corrects at 15 Hz, weighted by ICP's own information
  matrix and gated (chi-square, 99%). Slip shows where the wheels disagree with the
  gyro or decelerate faster than the drives can; through a skid the scans carry the
  pose. Scans arrive about a tick after capture and are applied at their capture time
  (the filter re-runs the ticks since). The scan matcher alone (each good match
  replaces the pose; `localization:=icp`) runs beside it in shadow, scored on the same
  data. Outputs: map > odom as before, `/slam/covariance`.

**People on the moving base** (`perception/map_people.py`, `people_tracker.py`,
`people_node`): the lidars' points more than 15 cm from the map's surfaces (at the
localized pose) are foreground; legs and people as in section 9; tracks in the map
frame by a constant-velocity Kalman filter (position, velocity, standing or moving),
dropped at once where the lidars see through. Offline with six people (some not giving
way): 99.3-99.6% of the people within 5 m in view tracked, nobody missed for more
than 0.33 s, 22 mm median error, no false person from walls, tables or fixtures; live
97.8% within 5 m (occlusions included), no false tracks.

**Layouts**: `layout:=cell` (the arm task: the parked cell in the middle of the room)
and `layout:=stations` (the mobile job: the pile's table on the west wall, the tray's
on the east wall, the robot's home by the south wall).

Measured against the true pose (recorded drives offline, and live; people walking):
ours 2-3 mm median, at most 19 mm, mapping or localizing; slam_toolbox 11-19 mm
median, at most 41 mm; yaw within 0.43 deg; walls within 3 cm of the room's. Details
and CPU: `implementation_notes.md` (slam), `figures/slam/compare.png`.

The filter against the scan matcher alone, on the same data (the full job live; 22
offline batches of 20 drives with a wet patch, a worn tyre, a drifting gyro and scan
dropouts as plant conditions): p95 5.0-5.6 mm against 7.2-7.5 live, the worst 7.5-7.9
against 11.3-13.2, the pose's largest jump in a tick 4.2 mm against 13.6-14.7; under
every condition a smaller worst position error; its covariance consistent with the
errors (97-98% of ticks in the NEES 95% band live). `figures/ekf/live_crowd.png`;
`implementation_notes.md` (Localization by sensor fusion).

---

## 9b. Navigation (mobile base)

**Intuition.** Asked to go to a table, the robot backs out of where it is docked,
turns once, plans a route on its map that steers onto the line straight in front of the
table, follows it while watching the people (slowing near them, never closing in on
anyone) and rolls on in, facing the table, at both tables. A separate safety layer, like a safety
scanner's fields, stops the base whenever anything is in the zone it would need to
stop, whatever the planner and controller think.

**Technical.** Framework-free modules in `nav/`, wrapped by `nav_node` and
`safety_node`:

- **Route** (`planner.py`): Hybrid A\* over (x, y, heading) on the saved map's
  distance field, the chassis as three circles, with the base's own moves (0.2 m
  straight or on 0.8 m arcs, 15 deg turns in place, costly reversing), a cost per
  change of steering, a 2D Dijkstra heuristic and a shot to the goal: for a docking
  line, turn, straight and an arc that ends on the line; else rotate-straight-rotate.
- **Base MPC** (`base_mpc.py`, acados, 2.2 ms): unicycle kinematics with
  acceleration inputs over 2 s, speed, acceleration and wheel-rim limits; tracking
  with lateral error dear and lag cheap (keep the lane, slow down); each nearby
  person predicted at constant velocity and kept 0.9 m away by a soft constraint
  capped at the standing-still distance (never close in, never flee); a speed limit
  falling to 0.2 m/s near people; reversing only where planned. The top speed is a
  setting (`speed:=`, the mobile job's default 1.0 m/s; the turn rate follows it).
- **Docking** (`navigator.py`): both docks face their table, taught at commissioning
  (the robot parked there once, its own pose saved with the map). Undock straight back,
  route onto the dock's line, arriving within 3 cm / 3 deg of it and rolling on into
  the approach (else onto the line on one forward curve, rolling on; turn, straight,
  turn only where no curve fits), approach straight at 0.3 m/s (the last 0.25 m at 0.1
  m/s), check 2 cm / 1.5 deg, back out and retry if outside. Home is parked within
  2 cm / 1 deg (turn, straight, turn: no approach follows). People in the docking moves
  are the safety layer's.
- **Safety layer** (`safety.py`): the protective field is the chassis swept along its
  current arc until it can stop (0.15 s reaction, 1 m/s^2 braking) plus 10 cm, and
  stops the base for anything it is closing on; beyond it the speed and the turn are
  capped to what stops 0.5 m (turning in place 0.15 m) short of the nearest point
  (never below 0.3 m/s, 0.5 rad/s); checked for the command and the measured motion;
  watchdogs on the scans, odometry, people tracks and localization. The two corner
  scanners see from their window, their sectors along the chassis's faces.

Measured on the truth (20 drives pick > place > home each, quiet and with a crowd of
six in two seeds, offline; 1 + 1 live): all arrived; docked within 7 mm and 0.31 deg,
home within 4 mm and 1.03 deg; no protective-field intrusion and no contact; detour
at most 1.12; 7-10 swerves per 20 drives in the crowd, none in a quiet room. Details:
`implementation_notes.md` (navigator.py), `figures/nav/nav_sim.png`.

---

## 9c. The mobile pick and place (mobile_job.launch.py)

**Intuition.** The robot starts at home, drives to the tray's table and looks for the
tray, then shuttles: to the pile's table, scan, pick, fold the arm back with the box
over the robot's rear deck, drive to the tray, find the tray again, place, look, drive
back, until the pile is empty; then home. People walk the room: the base stops and
slows for them while driving, the arm holds for them while working.

**Technical.** `task_node` (mobile:=true) runs the cell's task at each station in the
arm's frame and between stations sends navigation goals. The carry pose: the tool down
0.40 m behind the arm's base, the box inside the footprint. Azimuths are planned within
joint 1's range, never across the back; into and out of the carry pose the arm moves
across at height and vertically. At each later visit the tray is found again in the
held-box scan (one wall per axis) and the placed boxes move with it; a tray not where it
was means docking again. The arm's supervisor takes the people seen against the map,
moved into the arm frame (`people_node`); the base's safety layer, its own watchdogs.

Measured (live, the full job of 8 boxes): quiet 8/8 in 20 min; with six people walking
8/8 in 32 min, every arm hold with a person within 2.5 m, nobody nearer than 0.93 m
while the arm worked, no protective-field intrusion or contact while driving, docked
within 7 mm and 0.9 deg. Details: `implementation_notes.md` (the mobile job).

## 9d. The arm and the base at once (step 6)

**Intuition.** The robot no longer waits for itself: it backs out of a station while
the arm folds, and unfolds the arm while it makes its final approach, so the camera
is over the pile or the tray as the base stops. It never leaves the docking line with
the arm out, and if someone comes near the unfolded arm, the base stops too.

**Technical.** `task_node` sends the next navigation goal as the arm starts to tuck and
starts the station's first move when the navigation reports the straight approach; a
scan waits for the docked status. The navigator drives a route or an alignment turn
only with the arm stowed (`/task/arm_stowed`, the tool over the carry pose) and stops
while the task pauses it (`/task/pause_base`, the arm's supervisor holding the arm out
beyond the chassis). The
supervisor tracks the lidar's people in the odometry frame and adds the base's motion
to the arm's speed toward them. Three ways were compared offline (the arm's legs as
they are; goals held still in the room through the two controllers; a whole-body MPC
over base and arm, `core/wb_ocp.py`): docking and tracking the same, the plainest the
fastest, and kept.

Measured (live, the full job): quiet 8/8 in 17.2 min, 129 s a box (152 s in sequence),
docked within 7 mm and 0.9 deg; with six people 8/8 in 28-31 min, 209-236 s a box (245
in sequence: the base pauses for people near the unfolded arm), nobody within 1.0 m of
the arm while the base moved with it out, no intrusion or contact. Details:
`implementation_notes.md` (arm and base together).

## 10. Measured results

**The mobile job, latest** (live, real time, the full job of 8 boxes, the
7.0 x 5.6 m room, the base at 1.0 m/s; block packing with the held slide, docking
without turning on the spot):

| | quiet | crowd of 4 |
|---|---|---|
| job time, time per box | 8.5-8.6 min, 64-65 s | 13.8 min, 104 s |
| drives docked, max error | 19/19, 7 mm 0.5-0.8 deg | 19/19, 5-8 mm 0.9-2.0 deg (2.0: parked at home, since kept to 1 deg) |
| true gap between neighbouring boxes, median | 0.8 mm | 0.7 mm |
| boxes on the floor (the rest stacked) | 6-7 of 8 | 7-8 of 8 |
| closest person to the moving chassis | - | 73-95 mm |
| closest to the arm or box, robot closing on them | - | 199-258 mm |
| protective-field intrusions, contacts | 0, 0 | 0, 0 |
| arm holds (all with a person near) | 0 | 28-36 |
| people within 5 m tracked | - | 98.8% |
| extra turning per drive, median / max | 12-18 deg | 89 / 201 deg |

Boxes on the floor vary with the pick order: when the first box in the far corner stands
off its wall (`known_issues.md` W1) the last one or two find no floor spot and are
stacked. The tables below are earlier states, kept for the record.

**The mobile job, five scenarios** (live, real time, the full job of 8
boxes each, in the 10 x 8 m room with a crowd of six; `figures/mobile/scenarios.png`):

| | quiet | walkers (3) | crowd (6) | dock block (1) | step in (1) |
|---|---|---|---|---|---|
| boxes placed, time per box | 8, 129 s | 8, 159 s | 8, 209 s | 8, 142 s | 8, 157 s |
| drives docked, max error | 19/19, 7 mm 0.7 deg | 19/19, 7 mm 0.7 deg | 19/19, 6 mm 0.9 deg | 19/19, 7 mm 1.0 deg | 19/19, 7 mm 0.9 deg |
| closest person, chassis moving | - | 102 mm | 160 mm | 195 mm | 208 mm |
| closest to the arm or box, robot closing on them | - | 177 mm | 231 mm | 219 mm | 289 mm |
| protective-field intrusions, contacts | 0, 0 | 0, 0 | 0, 0 | 0, 0 | 0, 0 |
| arm holds (all with a person near) | 1 (start-up) | 40 | 146 | 61 | 18 |
| localization error p95 / max | 9 / 14 mm | 8 / 14 mm | 9 / 15 mm | 8 / 13 mm | 8 / 13 mm |

Offline, the same scenarios' navigation (20 drives, two seeds each, the carry pose
with a box): 160/160 arrived, docked within 7 mm, no intrusion or contact, the arm
or box at least 226 mm from anyone the robot closed on. Details:
`implementation_notes.md` (the scenario set).

**Speed** (the 7.0 x 5.6 m room, the base at 1.0 m/s, the faster docking
and arm transits; live, the full job):

| | quiet, 0.5 m/s | quiet, 1.0 m/s | crowd of 4, 0.5 m/s | crowd of 4, 1.0 m/s |
|---|---|---|---|---|
| job time, time per box | 15.8 min, 119 s | 10.5 min, 79 s | 27.4 min, 206 s | 18.8 min, 141 s |
| drives docked, max error | 19/19, 6 mm 0.9 deg | 19/19, 7 mm 0.6 deg | 19/19, 7 mm 0.9 deg | 19/19, 7 mm 0.9 deg |
| protective-field intrusions, contacts | 0, 0 | 0, 0 | 0, 0 | 0, 0 |
| closest person, chassis moving | - | - | 74 mm | 140 mm |

Offline at 1.0 m/s, the scenario set and the gate (12 batches of 20 drives): 240/240
arrived, docked within 7 mm and 0.35 deg, no intrusion or contact. Details:
`implementation_notes.md` (Speed).

**Safety settings** (`base_safety=`, `arm_safety=`, `assumed_speed=`, `people_speed=` in
`scripts/mobile_pickplace.sh`; the margins on top of the stopping distances scale,
never below 0.14 m): live with the crowd of four, close (0.6 / 0.6) 106 s a box,
default 141 s, cautious (1.5 / 1.5, assuming 2.0 m/s) 221 s; 8/8 each, no intrusion
or contact. Details: `implementation_notes.md` (Safety settings).

**Latest batch**: the obstacle scenario (a person visiting the
tray three times), seeds 1-10, real time, viewer and windows on, people
sensed by the lidars, after the motion fixes of section 8.1 (before the
contact pick of 8.2, which passed its own seed-2 gate: 8/8 plain and
obstacle):

| | 10 runs |
|---|---|
| boxes placed | 80 / 80 |
| solver failures | 0 |
| late ticks per run | 2-5 of about 10 000 |
| solve time p50 / p99 / max | 3.0 / 4.4 / 11 ms |
| joint speed, max | 2.5-2.9 rad/s |
| jitter (RMS of $\dot q$ minus its 100 ms average) | 0.04-0.06 rad/s |
| tool off its reference while moving, p99 / max | 13-19 / 22-55 mm |
| carried box tilt, max | 1.0-2.0° |
| sweeping (`motion_check.py`): self-motion / worst back-and-forth | none / 0.55 rad |
| time per run (8 boxes, 3 visits) | 197-226 s |

**Final test** (step 7 of the realistic cell, after the contact pick; fresh
seeds 11-15, three visits and two walks across the cell, lidars, real time,
all windows): 40 / 40 boxes placed, no failed pick, no solver failure, 2-3
late ticks per run, solve p50 / p99 2.8 / 4.4 ms, top joint speed 2.5-2.7
rad/s, no self-motion, worst back-and-forth 0.52 rad, tool off its reference
at most 49 mm, closest true person-arm distance while the arm moved
938-1321 mm, every hold with a person in the cell, 185-220 s per run.

**Mobile manipulator, step 1** (the cell inside the room, the arm on
the mobile base parked at the cell; seed 2, real time, lidars):

| | plain | obstacle (3 visits) |
|---|---|---|
| boxes placed | 8 / 8 | 8 / 8 |
| time | 182 s | 205 s |
| late ticks | 8 of 8969 | 9 of 10071 |
| solve p50 / p99 | 4.0 / 4.7 ms | 4.0 / 4.7 ms |
| joint speed (norm), max | 3.3 rad/s (the grip jolt, M7) | 2.6 rad/s |
| self-motion / worst back-and-forth | none / 0.38 rad | none / 0.36 rad |
| base while parked | 0.25 mm, ±0.06°, ends 0.000° | 0.05 mm, ±0.014°, ends 0.000° |
| holds / without a person | - | 3 / 0 |
| closest true person-arm distance while moving | - | 1009 mm |

Solve times were 2.7-2.8 ms at the start of the session and 4.0 ms on the same
code later (an A/B run without the base's brake gave 4.0 ms too: the machine,
not the change). One grip-and-release jolt with a 1.7 kg box in the plain run
(`known_issues.md` M7). The first gate runs, before the base's brake and static
friction (`implementation_notes.md`, base_drive.py), also placed 8/8 twice, but
the parked base turned 0.09° and 0.33° over the run and the lidars saw a false
person at a table leg.

The motion fixes, same seeds, before → after: worst back-and-forth of a
turning joint 2.5 → 0.55 rad (the wrist wound up and back on every carry),
moves flagged as sweeping 18-22 → 0-3 per run, the arm swinging with the tool
still in 3 runs → none, worst detour 4.2× → 1.07×, top joint speed 4.9 → 2.9
rad/s.

People (seed 2): the closest true distance between the arm and the person
while the arm moved was 940-1361 mm with the lidars (344 mm with the ceiling
camera only); every hold had a person in the cell.

Earlier (control-smoothness work, before this cell; lockstep, half-scale
cell):

| | before | after |
|---|---|---|
| launches with a failed start | 30-50% | 0 in 76 launches |
| jitter | 0.76-1.11 rad/s | 0.016-0.017 |
| joint speed max | 8-10.5 rad/s | 2.8 |
| carried box tilt max | 13° | 0.1° |

---

## 11. Tools for measuring

All in `scripts/dev/`, runnable with the conda environment (`source
scripts/env.sh`):

| script | what it does |
|---|---|
| `bench_run.sh <tag> <plain|obstacle>` | one launch of the real demo script with telemetry; stops when the task parks (`HEADLESS=1`: no windows) |
| `bench_batch.sh <prefix> N` | N launches of each demo, then a summary table |
| `bench_analyze.py <runs>` | one line of numbers per run: solve times, late ticks, jitter, jerk, torque chatter, sway, tilt, tracking wobble |
| `motion_check.py [--plot dir] <runs>` | sweeping motion: self-motion, back-and-forth of turning joints, detours, creep, tracking; one plot per run |
| `person_clearance.py <runs>` | true person-arm clearance while the arm moves, and holds without a person |
| `lidar_check.py [--draw]` | offline lidar detection check: grid, paths, empty scans |
| `layout_check.py <run>` | arm clearance to the furniture over a run; ceiling-camera person detection |
| `odom_check.py [--plot dir]` | the base's odometry against truth on scripted drives (no ROS) |
| `dock_reach_check.py` | IK reach over both zones from candidate docking poses |
| `arm_on_moving_base.py` | the arm's MPC holding a pose while the base drives and turns (no ROS) |
| `record_drive.py <name>` | records a drive (lidars, encoders, IMU, odometry; truth for scoring), no ROS |
| `slam_eval.py <recording>` | our SLAM on a recording, or slam_toolbox's result: trajectory, walls, map, CPU; `--localize` |
| `toolbox_replay.py <recording>` | plays a recording to slam_toolbox in simulated time and collects its result |
| `slam_compare.py` | the comparison figure of both SLAM options |
| `mobile_run.sh <tag> <own|toolbox> <mapping|localization> <map>` | one live drive with SLAM (mobile.launch.py), the map saved; `route:=nav` drives the goals with the navigation |
| `nav_sim.py [--drives N] [--crowd N] [--seed S] [--arm carry --box] [--people job]` | closed-loop navigation offline (plant, odometry, localization, people, navigator, safety; no ROS), scored on the truth; with the arm in the carry pose (a box held) also the arm and box against the people's whole bodies |
| `job_run.sh <tag> [timeout_s] [launch args...]` | one mobile job (mobile_job.launch.py) with telemetry, until the task is done |
| `dock_reach_sim.py [--scenario S] [--strategy seq|overlap|split|wb]` | the arm moving while the base docks or undocks, four ways, offline (lockstep): time, docking, tracking, clearance to the stations |
| `job_time.py <runs>` | a mobile job's time per box, by navigation and arm state; the arm out while the base moves; the base waiting for the arm |
| `room_layout.py` | the room's layout checked on the compiled scene (tables to walls, the chassis turning at home, docks and route corners, people's paths) and drawn |
| `job_results.py name=<run> ...` | the scenario table of mobile jobs from the monitors' totals |
| `tray_reanchor_check.py [n] [--held]` | the tray found again after dockings a few mm and tenths of a degree apart, offline |
| `people_eval.py <recording>` | people detection and tracking on a crowd recording against the truth (no ROS) |
| `packing_compare.py` | offline packing comparison over random pick orders (`--strip --push-area`, `--land`, `--clear`) |
| `place_budget.py <run_dir> [...] [--rows]` | placement error budget from a run's validation lines: landing split into the arm's tracking, the in-hand offset and the release, the yaw, the neighbours' records, the sensed sizes and the box-to-box gaps against the truth |
| `push_sim.py [push_mm] [side_gap_mm]` | offline push of a released box by a compliant tool, the scene's contacts, at several heights and leans |
| `replay_offline.py <run>` | replays a recorded run's goals through the same OCP and MuJoCo, no ROS |
| `place_check.py` / `place_depth.py` | sideways error during each descent into the tray; depth below target |

Per-tick telemetry: `MANIPOPT_TELEMETRY_DIR=<dir>` makes the plant, the
controller and `task_node` each write a CSV (`sim.csv`, `mpc.csv`,
`task.csv`) and the sensors process save both windows.

---

## 12. Libraries

| library | used for |
|---|---|
| **ROS 2 Humble** (`rclpy`, `launch_ros`, `tf2_ros`, `nav_msgs`) | nodes, topics, launch files, the odometry transform |
| **slam_toolbox** (2.6.10) | the second SLAM option (mapping, localization) |
| **MuJoCo** (+ Menagerie Panda) | physics, rendering (depth, RGB), ray casting (lidars), viewer |
| **acados** (`acados_template`, HPIPM, BLASFEO) | code-generated OCP solver: SQP_RTI, IRK, partial condensing, soft constraints |
| **CasADi** | symbolic dynamics, kinematics and costs, exact derivatives for acados |
| **Pinocchio** (+ `pinocchio.casadi`) | robot model from MJCF, ABA dynamics, forward kinematics, gravity torque |
| **Ruckig** (community) | online jerk-limited trajectory generation |
| **NumPy / SciPy** | heightmaps, segmentation, detection, tracking, packing |
| **OpenCV** | the decision and obstacle windows |
| **MoveIt 2 + MoveIt Task Constructor** (second method, C++) | sampling-based planning, with a computed-torque trajectory-following bridge |

---

## 13. The second method: MoveIt (brief)

Same task, same plant, same sensing and decisions (`decision_node.py` reuses
`perception/heightmap.py`); only the motion differs. MoveIt Task Constructor
plans each pick and place as a stage graph (approach, attach, lift, connect,
lower, detach, retreat) with OMPL, and a C++ `bridge_node` follows the
trajectories with a computed-torque law
$\tau = \mathrm{RNEA}(q,\dot q,\ddot q_{ref} + K_p e + K_d \dot e + K_i\!\int e)$,
with the feedback scaled by the true inertia so every joint is critically
damped at every pose. The pile is put into the planning scene from each
scan as 2 cm columns. It was not re-tested after the realistic-cell changes
(layout, real time, sensor noise, the contact grasp, the packing rule); it
still reads the grasped box's size from the simulator (known_issues.md G2).

---

## 14. What is sensed, what is given (honesty section)

Per `handover_notes/realistic_cell_plan.md`, MPC method. The
MoveIt method still uses the nominal constants and reads the grasped box's
size from the simulator.

**Sensed or planned online, no ground truth and no recorded paths:**
which box to pick and where to grasp it, and its footprint (wrist-camera
depth); that the tool is on it (wrist load) and that it was gripped (its
weight after the lift); its offset in the gripper (sensed top vs the arm's pose at the
grasp); its height (wrist load at touch-down); the pile's height; the tray's
position, walls, floor and wall height (found in the first tray scan); what
is in the tray (a scan with an empty hand after every place); where to place
(block packing with look-ahead, checked against the scan and the robot) and
how far to slide the held box (until the wrist feels it touch); every height:
approach, lift, carry, tray entry, lift-off and both scan poses, each checked
for reach with IK; every reference trajectory (Ruckig); every torque (the
MPC solve); the carried box's mass (wrist load); people (two safety lidars;
the ceiling camera optional, which can also tell a static object).

**Given in advance, as an integrator would teach it at installation:** the
robot's model (the base's nominal wheel radius and track included); both cameras' and both lidars' calibration and the wrist
camera's near limit;
the two zones (a pick zone where piles are put and a place zone where the
tray is put, each with the highest thing it takes); a box spec (sides
0.05-0.16 m, heights 0.07-0.13 m); speed limits, margins and tuned weights;
the packing rule and the heading rule (an empirical joint-7 relation);
separation-distance parameters (ISO 13855 values).

**Read from the simulator or not modelled (stand-ins):**
1. The grasp is a weld that grips only with contact: on `pick_at` the plant
   attaches the box whose top the tool is actually on (TCP within 4 mm above it
   or up to 15 mm into it, inside its top face); otherwise nothing. The method
   lowers until the wrist load shows contact, grips, and after the lift checks
   the box's weight in the wrist; a miss or an empty lift re-scans the pile. No
   suction model, so no slip once gripped.
2. The wrist camera's pose comes from the measured joint angles and the
   hand-eye calibration (the pose the plant puts in the depth message is not
   used); the method masks the tool and a held box itself.
3. Calibration is exact; pile boxes are never rotated; the tray is never
   rotated (only its position is sensed).
4. The plant runs in real time by default and never waits for a late solve
   (lockstep stays only as a debugging option).
5. The ceiling camera's robot self-filter is kinematic: the method's own robot
   model at the measured joint angles and the held box from its sensed size and
   offset, projected into the calibrated camera. People are sensed by two
   safety-lidar stand-ins by default (ray cast, with noise and dropouts).
6. Joint encoders and the wrist load cell have seeded noise (and the load cell
   a bias); the wrist camera stays pointing straight down whatever the tool's
   tilt (the tool is level at every scan).
7. The floor height is given (the robot's mounting), and the zones and box
   spec were chosen to cover this demo's pile and tray.
8. The plant still publishes a grasped box's true size and offset for its own
   log; the method does not subscribe to them.
9. The mobile manipulator (steps 1-2): the base is parked at the cell for the
   arm task, so the zones (taught in the arm's frame) and the ceiling camera's
   calibration (relative to the arm) hold as taught; once the base drives to the
   tables, the zones will come from the map through the localized pose. The
   base's odometry, the map and the pose in it are the method's own (encoders,
   gyro, lidars). The commissioning drive is steered on the true pose by a scenario
   actor standing in for a technician with a joystick; the navigation (step 4)
   drives on the method's own estimate, to docks taught by driving: at commissioning
   the robot is parked at each station once (by the same joystick stand-in, steered on
   the truth) and saves its own localized pose next to the map (`nav/docks.py`); the
   method reads no station or dock pose from the layout. The pick zone at the pick dock
   is taught relative to the docked arm, as the zones always were.

---

## 15. Limits and open problems (see `known_issues.md`)

- Packing is online with at most 3 boxes of look-ahead and sensed sizes only;
  with this pile's sizes some orders still leave one box to stack (97% of
  random orders fit all eight on the floor offline). The pick order is the
  pile's (biggest visible top first), not chosen for the packing.
- The first box in the far corner can stand off its wall: the set-down heading
  is fixed per footprint, and at the usual one the wrist's long side faces the
  wall (46 mm off for one box), and that long push jams (known_issues W1).
- In a crowd the base still steers round people standing by the tables (~90 deg
  extra turning a drive; known_issues W2).
- The corners on the robot's side of the tray refuse pushes towards the
  robot's wall (the forearm hangs over it); boxes there may stay a few mm off
  flush.
- Slow set-downs still let joints 1, 3 and 5 drift a few degrees (the posture
  term is the only thing holding that direction); one-tick speed spikes of up
  to about 3 rad/s remain at some pass-through corners.
- Obstacle handling is a software stand-in, not a safety function; the lidar
  sees legs only (hence the 1.12 m allowance), the ceiling camera is blind in
  the pile and tray regions; the protective-distance parameters are the
  standard's defaults, not measured stopping performance.
- WSL2: the system clock jumps ~1.5 s every ~34 s (handled); all rendering is
  on the CPU (llvmpipe).
- The base's localization filter does not see a slide while the scans are lost (a
  protective stop during a dropout: 29-52 mm off until they return; the IMU's
  accelerometer would), its heading lags a fast-drifting gyro bias by a few hundredths
  of a degree more than the scan matcher alone, and the navigation does not use its
  covariance yet (known_issues U).
- At 1.0 m/s with four people the base swerves more (16 swerves, the largest 30 deg,
  in a live job); with people near a station the base waits for the arm's held fold
  (118 s in a live job); people walking into the robot have come within 48-97 mm of
  the folded arm's elbow (no contact; the robot was not moving towards them).
- The base in a crowd still swerves a few times per 20 drives (largest heading
  swing 18 deg) and turns up to 500 deg more than its plan avoiding people; the
  2D lidars see tables as their legs; the base's safety layer is a software
  stand-in too. The fields keep people's legs 10 cm farther than other things, for
  the upper body the scanners do not see (the folded arm 127 mm or more from anyone
  live); people near a station hold the arm and slow the job. While the base docks or
  undocks the arm may be out, beyond what the base's fields watch: the arm's
  supervisor covers it, and its hold stops the base too.

---

## 16. Suggested storyline for a presentation

1. **The task**: an arm, a pile, a tray, a person. Nothing about the boxes is
   given to the robot (show the decision window).
2. **Sense**: the depth image becomes a height map, then boxes; biggest first.
   Figure: the heightmap with the chosen box outlined.
3. **Plan the motion**: jerk-limited reference, swings as arcs round the base,
   the box turning with the base. Figure: a reference speed profile before
   (dips at every arc point) and after (one smooth profile).
4. **Control**: MPC on joint torques, the OCP in one slide (cost, dynamics,
   limits, soft collision constraints), acados SQP_RTI at about 3 ms per solve.
5. **Why timing matters**: dead time from two unsynchronised timers → lockstep
   → real time with a state estimator predicting to when the torque starts.
   Figure: joint speed before/after (plots from `sim.csv`), and the replay
   experiment (lag injected → instability).
6. **Reference preview**: the MPC gets the future path; the sideways
   tracking error drops from 5-8 mm to under 1 mm.
7. **Pack**: one block from the far corner with a look-ahead, checked against
   what the robot can really do, then the held box slid against its neighbour
   until the wrist feels it (gaps 6.5 → 0.8 mm). Figure: `figures/packing/`
   layouts.
8. **People**: lidar legs (and optionally the ceiling camera), track,
   speed and separation monitoring (the formula, the 1.12 m allowance), hold
   and smooth resume. Figure: gap vs required distance
   and the speed scale over a visit.
9. **Numbers**: the before/after table.
10. **The mobile base**: SLAM (map once, localize), people as the points the map does
    not explain, Hybrid A\* and the base MPC, the protective field as the chassis's way
    to a stop. Figure: `figures/mobile/scenarios.png`.
11. **The mobile job**: the arm folding while the base undocks, unfolding while it
    docks; speed (79 s a box quiet) and the safety settings (106 to 221 s a box in a
    crowd, from close to cautious).
12. **Limits and next steps**: the navigation slowing while localization is unsure
    (the filter's covariance), packing-aware pick
    order, a learning-based MPC against this baseline, certified safety is out of
    scope, the MoveIt comparison.
