# System overview: sensed pick-and-place with a torque-level MPC

State as of 2026-09-28. Source material for the README and a presentation:
what the system does, how each part works (intuition first, then the
technical detail and the maths), which libraries it uses, and what was
measured. Design history and rejected alternatives are in
[`design_notes.md`](design_notes.md); open problems in
[`../known_issues.md`](../known_issues.md).

---

## 1. What it does, in one minute

A simulated Franka Panda arm empties a pile of eight mixed-size boxes and
packs them into a tray on the floor behind it, while a person may walk up to
the tray at any time.

- **Nothing about the boxes is given to the robot.** A camera on the wrist
  looks at the pile, works out which boxes are on top and how big they are,
  and picks the biggest one first. It looks at the tray again before each
  placement.
- **The arm is driven by torques from a model-predictive controller (MPC)**
  that re-plans the next 0.3 s of motion 50 times a second, with the arm's
  real dynamics, joint limits, torque limits and collision constraints
  inside the optimisation.
- **Boxes are carried level** (tilt at most 0.1-0.6 deg) and turn with the
  base like on a palletising robot, so they arrive square to the tray.
- **Boxes are packed in two tidy rows**, each box 3 mm from its neighbours
  and the tray walls, planned with a short look-ahead at the boxes the camera
  can already see.
- **People are detected by a ceiling camera**, tracked and classified. The
  arm slows as a person approaches, stops while they are too close, and
  resumes when they leave, with a separation distance computed the way the
  safety standard (ISO 13855 style) does it. A static object in the way is
  planned around instead.
- **Smooth and repeatable**: every launch behaves the same, no start-up
  failures in 76 launches since the timing fixes, joint motion free of
  jitter (numbers in section 9).

A full cycle of eight boxes takes about 133 s (156 s with three visits from
a person).

---

## 2. The system at a glance

Six ROS 2 nodes (Python, `rclpy`), plus framework-free libraries in `core/`
and `perception/` that the nodes call.

```
                      ┌──────────────────────── ground truth (environment only) ─────────────┐
 dynamic_obstacle_node ─ /env/dynamic_obstacle ─▶ mujoco_sim_node (moves the person model)    │
                                                    │                                         │
            ┌─────────────── mujoco_sim_node: the PLANT (MuJoCo physics, cameras) ───────────┐│
            │  /sim/joint_states (q, q̇, step stamp) ─────────────┬───────────┬──────────────┐ ││
            │  /env/workspace_detections (ceiling camera blobs)  │           │              │ ││
            │  /sim/container_occupancy, /sim/destination_occupancy (wrist-camera heightmaps)│ ││
            │  /sim/grasped_box_size, /sim/grasped_box_offset (in-hand sensing)              │ ││
            └───────────▲───────────────────────────────────────┼───────────┼──────────────┘ ││
                        │ /sim/joint_command (torques,          │           │                ││
                        │  stamped with the state they answer)  │           │                ││
        mpc_controller ─┘◀── /mpc/goal, /mpc/orientation_goal ── task_node ◀── /mpc/hold,     ││
        (acados OCP,          (16-point reference horizon)     (decisions,    /mpc/speed_scale,
         solves on every      /mpc/obstacle_params             Ruckig          /mpc/static_obstacle
         new state)           /mpc/settled ─────────────────▶  reference)      ▲              ││
                                                                               │              ││
                                        obstacle_supervisor_node ──────────────┘              ││
                                        (track, classify, separation monitoring)              ││
                                        detection_monitor_node: sensed vs truth (validation) ◀┘│
```

**One control tick (20 ms of simulated time):**

1. The plant publishes the arm state (joint angles and speeds) stamped with
   its step number.
2. The controller receives it and immediately solves the MPC problem (about
   2.5 ms) using the reference points `task_node` stamped for that step, and
   sends back the first torque of the plan, stamped with the same step.
3. `task_node`, which also received the state, advances its reference by one
   step and publishes the next 0.3 s of it for the following solve.
4. The plant waits for the torque that answers its latest state (up to
   50 ms), applies it, and advances physics by 20 ms (10 MuJoCo sub-steps of
   2 ms).

The plant, the controller and the reference generator therefore run in
**lockstep**, on one clock: the plant's step counter.

---

## 3. The plant: `mujoco_sim_node` (MuJoCo)

**Intuition.** This is the "real robot" of the simulation. It knows the true
physics and the true positions of everything; the rest of the system only
sees what its sensors report.

**Technical.**

- **Physics**: MuJoCo (`mujoco` Python bindings), `implicitfast` integrator,
  2 ms timestep, 10 sub-steps per 20 ms control period. The Panda model is
  MuJoCo Menagerie's with plain torque motors (limits 87 Nm on joints 1-4,
  12 Nm on 5-7), joint damping 1 N·m·s/rad and armature 0.1 kg·m² per joint.
- **Torque interface**: `ctrl = τ` from the controller; if no answer arrived
  for 10 steps (controller absent or dead) it falls back to a
  gravity-compensated PD hold, $\tau = g(q) + K_p (q_{hold} - q) - K_d \dot q$.
- **Lockstep** (see section 8): each step waits for the command computed from
  the state it just published, up to 50 ms, paced to real time, catching up
  at most 100 ms after a stall.
- **Grasping** is a weld constraint (MuJoCo equality) between the gripper
  flange and the box, re-anchored at the current relative pose when engaged,
  so there is no snap. Each box also has a "rest" weld to the world, swapped
  with the carry weld at pick and place. Boxes have collisions disabled,
  so the plant also measures overlap and gaps at every release and while
  carrying, and logs them.
- **In-hand sensing** (stand-in for a gripper's contact sensing): after a
  grasp it publishes the box's size and the box centre's offset from the
  gripper point in the gripper frame. These are the only two ground-truth
  reads the task uses, and neither decides which box to pick or where to put
  it.
- **Wrist camera** (`container_cam`, 320×240, 55° field of view, mounted
  0.15 m above the flange): renders depth on request, turns it into a
  heightmap (section 6), and masks the arm and the held box out of its own
  image.
- **Ceiling camera** (`workspace_cam`, 3.5 m high, 70°, 320×240, 10 Hz):
  depth + segmentation renders for obstacle detection (section 7).
- **Decision window** ("pick & place decisions", OpenCV): pick-up on top,
  placement below. Each half shows the camera frame of that side's last scan,
  its depth image, and the decision `task_node` actually made, outlined at
  its true height (perspective projection with the pinhole model); it is
  redrawn only when a scan or decision comes in.
- **Obstacle window** ("obstacle detection (live)", OpenCV, 2 Hz,
  `obstacle_view_hz`; off with `obstacle_view:=false`): obstacle handling
  from the ceiling camera, zoomed
  to the central 2.8 × 2.1 m: left, what the detector made of the image
  (height in grey, ignored pile/tray/aisle regions blue, the robot's own
  pixels green, obstacle pixels red, each blob as the column it reports with
  its top height); right, the depth image with the supervisor's tracks
  (class, speed, velocity arrow), the worst gap vs the required protective
  distance and the arm point it is measured at, and a status bar (clear /
  slowed to N% / HOLD). The supervisor publishes this on `/supervisor/tracks`
  for display only. Both windows are composed on a thread of the plant and
  shown by a separate process (`window_process.py`): putting it on screen
  inside the plant's process, or often, slowed the simulation (see
  known_issues.md H7).
- **Telemetry**: with `MANIPOPT_TELEMETRY_DIR` set, a CSV row per step (sim
  time, step period, dead time of the applied torque, joint speeds, torques,
  angles) and a snapshot of the decision window.

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

Not in the model: the carried box (0.2 kg). The arm sags about 1-4 mm under
it, which is why "arrived" allows 8 mm.

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
- initial state: $x_0 = x_{measured}$;
- torque limits $|\tau_i| \le 87$ / $12$ Nm; joint position limits (from the
  robot file); joint speed limits 2.175 / 2.61 rad/s (the Panda datasheet), all
  hard;
- collision (soft, section 5.2):
  $\|p_j(q_k) - c_o\| - r_j - r_o - 0.03 \ge -s$;
- terminal goal (soft): $\mathrm{FK}(q_N) = g_N$.

The stage cost vector (nonlinear least squares) is

$$y = \big[\ \tau,\ \dot q,\ q - q_c,\ \mathrm{FK}(q) - g_k,\ \alpha(z_{tool}(q) - z^*),\ \alpha(x_{tool}(q) - x^*_k)\ \big]$$

with weights 1e-3 (torque), 1e-2 (joint speed), 1 (posture), **5000**
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
changing only a weight never forces a rebuild. Solve time: median 2.5 ms,
p99 about 8 ms.

**Around the solve** (`mpc_controller_node`):

- Solves as soon as a new state arrives (no timer of its own) and stamps the
  torque with the state's step (lockstep, section 8).
- Picks the goal and orientation horizon stamped for that exact step; if it
  has not arrived yet (the plant catching up after a stall), it extrapolates
  linearly from the two newest ones rather than reusing a stale one.
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
  decreasing-size heuristic. The grasp point is the centre of the sensed top
  face. The box's height cannot be seen from above; it comes from in-hand
  sensing after the grasp.
- **Shadows.** A cell hidden behind a tall box never converges between floor
  and box top; a final consistency check reports unseen cells as floor.

---

## 7. Placement: two-row ("shelf") packing (`pick_place_common/packing.py`)

**Intuition.** Pack the tray the way a person would: two rows along the long
walls, each box pushed against the previous one. The only real decision is
which row a box goes into, and it is made by glancing at the boxes still
waiting on the pile.

**Technical.**

- Tray inner size 430 × 280 mm, rows along the long side: the back row flush
  against the far wall, the front row against the near wall, both filled from
  the same end; each box goes 3 mm after the last box in its row
  (`PLACE_CLEARANCE_M`, which covers the ~1.6 mm release error of this box and
  its neighbour).
- **Look-ahead (receding horizon, like the MPC):** the current box and the up
  to three boxes the last pile scan could also see (their sensed footprints)
  are tried in every combination of rows ($2^{k+1}$, at most 16). Choose,
  lexicographically: most boxes fitting → the shorter longest row (balanced
  rows) → most touching perimeter. Only the current box is committed; the
  choice is made again, with a fresh scan, for the next box.
- **Verification:** the sensed tray heightmap must show the chosen footprint
  clear down to the floor (inset by one 5 mm sample), so a foreign object
  still blocks a spot.
- **Fallback:** a box that fits in neither row goes to the heightmap search
  (lowest flat spot, contact-scored), then is snapped flush (3 mm) against the
  nearest wall or placed box.
- **Placing the box, not the gripper:** the target is the box centre, then
  converted to a gripper target with the in-hand offset rotated by the
  gripper's heading at the tray:
  $p_{tcp} = c_{box} - R(\psi)\,o_{box}$. The box is set down 2 mm above the
  sensed surface (the arm sits ~1.2 mm low with a box in hand).

Why rows: free-form rules (most contact, fewest unusable slivers, most compact
from a corner, largest free rectangle, with and without look-ahead and
90° rotation) were compared offline (`scripts/dev/packing_sim.py`,
`tray_layout.py`). All left holes and slivers between groups of boxes; the
user chose rows. Even a full-knowledge beam search cannot make these eight
sizes into one gap-free block.

Measured: every box 2.6-3.6 mm from its facing neighbours and walls, zero
overlap at release and on the way down, bottoms 1-1.7 mm above the tray
floor.

---

## 8. Motion and timing: `task_node` (Ruckig) and the lockstep

### 8.1 The reference: where the gripper should be, moment by moment

**Intuition.** The MPC tracks a moving target. `task_node` moves that target
along a smooth path with limited speed, acceleration and jerk, stops only
where the task needs it (grasp, release, the two camera scans), and turns the
box with the base.

**Technical.**

- **Ruckig** (online, time-optimal, jerk-limited trajectory generation):
  0.5 m/s, 3 m/s², 20 m/s³ per axis. Each tick it advances one step from the
  current reference state to the current target.
- **Planned round the base, in cylindrical coordinates** $(\varphi, r, z)$
  instead of $(x, y, z)$: a swing from the pile to the tray (~155° round the
  base) is a straight line in these coordinates, i.e. a true arc with one
  accelerate/cruise/brake profile. The azimuth limits are the Cartesian ones
  divided by the leg's larger radius, so the gripper never exceeds them.
  State conversion (position, velocity, acceleration):
  $\dot r = v\cdot e_r,\ \dot\varphi = v\cdot e_t / r,\
  \ddot r = a\cdot e_r + r\dot\varphi^2,\
  \ddot\varphi = (a\cdot e_t - 2\dot r\dot\varphi)/r$.
  As a chain of 40° points in x/y/z the reference slowed at every corner (a
  1.4 Hz sway of the whole arm). The old x/y/z arc-and-via-point path is kept
  only while a detour around a static obstacle is needed, because the detour
  clearance check assumes straight segments.
- **Pass-through waypoints**: the points above the box, the lift, the point
  above the slot and the lift-off are passed at speed (Ruckig target velocity)
  instead of stopping. Pass speed $0.5\cos(\theta/2)$ m/s for a turn of
  $\theta$, capped at what is reachable in the distance left. Arriving
  corners (above the box or slot) are passed moving straight down, so the
  descent into the tray has no sideways speed. Departing corners (top of the
  lift, top of the lift-off) are **blended**: the swing starts 5 cm before
  the top, so rising and turning overlap. The carried box still clears the
  pile by ≥ 8 cm there.
- **Heading of the box**: the tool's x-axis heading goes from its start value
  to the aligned heading at the goal (±90°, whichever keeps joint 7 near
  mid-range: $q_7 \approx q_1 - \psi - 135°$), in proportion to how far round
  the base the reference has come, eased with a smoothstep $3f^2 - 2f^3$ so
  its rate starts and ends at zero. It arrives turned 180°, which gives the
  same footprint.
- **Reference preview**: every tick, one Ruckig trajectory calculation on a
  copy of the input, sampled at 0, 20, …, 300 ms (continuing at the pass
  speed past a pass-through point) gives the 16 stage goals, and the heading
  at each.
- **People**: the supervisor's hold switches Ruckig to its velocity interface
  with a zero target (braking at the full 3 m/s², the stopping performance the
  separation distance assumes); its speed scale multiplies the velocity limit
  (and the pass speed). On release the target is re-planned from where the
  reference stopped, and a pass-through point that is within 3 cm or was
  already passed while braking is skipped (no hesitation or turning back on
  restart).

### 8.2 The task sequence

A state machine per box: go to the pile's scan point → scan → pause 1 s →
above the box (pass) → down to the top face → grasp → lift (blend) → swing to
the tray's scan point → scan (box in hand, masked from the camera) → pause →
above the slot (pass, straight down) → down → release → lift-off (blend) →
back to the pile's scan point. It stops when the pile is empty or the tray is
full, and gives up (box held over the tray) after 30 scans with no room. It
advances on the controller's "settled" signal at stops, and on the reference
itself at pass-through points.

### 8.3 Timing: why lockstep, and what it fixed

**Intuition.** A feedback controller is only as good as the freshness of what
it acts on. If the torque applied now was computed from where the arm was
20 ms ago, a strong controller starts to oscillate.

**What was wrong:** the plant and the controller each ran their own 20 ms
timer, and the phase between them was set by chance at every launch. On some
launches 25-35% of torques were a step old. That was the flaky start (30-50%
of launches: solver failures `ACADOS_MINSTEP` or 4-7 rad/s oscillation) and
the "some runs fine, some shaky" behaviour. An offline lockstep replay
reproduced both by injecting the delay.

**What it is now:**

- The controller solves on each state; the plant steps only when the answer to
  its latest state has arrived (max wait 50 ms, timed from when the state was
  sent or the plant thread came back from a stall), paced to real time.
- State and command queues hold one message (never solve for an old state).
- `task_node` ticks on the plant's states and stamps its goals with the step
  they are for.
- In-process timeouts use a monotonic clock: WSL2 steps the system clock
  forward ~1.5 s every ~34 s, which used to trip the supervisor's "perception
  down" fail-safe.

---

## 9. Obstacle handling: people and objects (ceiling camera)

**Intuition.** A fixed camera above the cell sees anything that sticks up
from the floor outside the pile and tray. Anything that moves, or is as tall
as a person, is treated as a person: the robot slows down and waits for them,
never steering a box around them. Something small that stays still is an
object, and the robot plans around it.

**Truth vs belief.** `dynamic_obstacle_node` moves the person model (a
half-scale 0.84 m body: legs, torso, arms, head) and publishes ground truth.
Only the plant (to move the model) and the validation monitor read it. The
controller learns about people only through the camera, so detection error,
latency and occlusion are real effects in the results.

**Detection** (`perception/obstacle_detection.py`, numpy/scipy): every depth
pixel is unprojected to 3D. A pixel is foreground if its point is above the
floor plane (5 cm), outside the masked pile/tray/aisle regions, and not the
robot (segmentation ids of the arm and the held box are masked: a kinematic
self-filter). Foreground pixels form blobs (≥ 25 px, merged if a few cm
apart, since arms seen from above only touch the torso). Each is described as
a **column** (x, y, radius, top height), because a camera looking down cannot
see under anything.

**Tracking and classification** (`perception/obstacle_tracking.py`, no ML):
gated nearest-centroid association (gate 0.15 m + 2.5 m/s × dt), alpha-beta
filter for position and velocity. A track is a **person** if it moves
(> 0.25 m/s for 2 frames), has ever moved, is ≥ 0.5 m tall, or is not yet
confirmed. It is **static** only after 1 s essentially motionless. Uncertain
means person, the fail-safe side. A static object latches its position at
confirmation (the arm passing over it hides it) and is remembered 10 s when
occluded.

**Speed and separation monitoring** (`obstacle_supervisor_node`): for each
person the gap from their column to four points on the arm is compared with
the protective distance (ISO 13855 style)

$$S = v_h (T + t_s) + v_r T + \frac{v_r^2}{2a} + C,\qquad t_s = \frac{v_r}{a}$$

with $v_h$ = 0.8 m/s (the standard's 1.6 m/s at half scale), $v_r$ the arm's
speed toward the person, $T$ = measured camera latency + one frame (100 ms) +
two control ticks, $a$ = 3 m/s² (the reference's braking) and $C$ = 5 cm.
Speed scale = 0 (hold) if the margin $d = \text{gap} - S \le 0$, else
$0.15 + 0.85\min(d/0.6\,\text{m}, 1)$, with 5 cm hysteresis on release.
Fail-safes: no detections for 0.5 s → hold; supervisor silent for 1 s →
`task_node` holds. A static object goes into obstacle slot 3 and `task_node`
plans a single detour via-point (left, right or over), checked for clearance
and reach.

This is a software stand-in: the simulation has no certified safety scanner.

---

## 10. Measured results

Viewer and decision window on, launched with the demo scripts exactly as a
user runs them, one run each (the last validation of the final code):

| | plain demo | with a person visiting 3 times |
|---|---|---|
| boxes placed / on the tray floor | 8 / 8 | 8 / 8 |
| overlap with walls/boxes at release | 0 mm | 0 mm |
| torque from a stale state (dead time) | 0% of steps | 0% |
| solver failures | 0 | 0 |
| joint speed max | 2.8 rad/s | 2.8 rad/s |
| jitter (RMS of $\dot q$ minus its 100 ms average) | 0.017 rad/s | 0.016 rad/s |
| wobble around the path (1-5 Hz tracking error, RMS / p99) | 0.4 / 1.1 mm | 0.4 / 1.2 mm |
| carried box tilt, max | 0.1° | 0.1° |
| holds for the person / false "perception down" holds | – | 3 / 0 |
| total time | 133 s | 156 s |

Before the control-smoothness work (same scripts and metrics):

| | before | now |
|---|---|---|
| launches with a failed start | 30-50% | 0 (0 solver failures in 76 launches since the fix) |
| jitter | 0.76-1.11 rad/s | 0.016-0.017 |
| joint speed max | 8-10.5 rad/s | 2.8 |
| joint acceleration p99 | 219-281 rad/s² | 11 |
| wobble around the path (RMS) | 9.7 mm | 0.4 mm |
| carried box tilt max | 13° | 0.1° |
| times the arm stops per box | 11 | 4 (grasp, scan, release, scan) |
| boxes on the tray floor, gap to neighbours | 7 of 8, gaps 5-300 mm, overlaps up to 7 mm | 8 of 8 in two rows, 3 mm, no overlap |
| cycle time | 162-197 s | 133 s |

---

## 11. Tools for measuring

All in `scripts/dev/`, runnable with the conda environment (`source
scripts/env.sh`):

| script | what it does |
|---|---|
| `bench_run.sh <tag> <plain|obstacle>` | one launch of the real demo script with telemetry; stops when the task parks |
| `bench_batch.sh <prefix> N` | N launches of each demo, then a summary table |
| `bench_analyze.py <runs>` | one line of numbers per run (the columns in section 10) |
| `replay_offline.py <run>` | replays a recorded run's goals through the same OCP and MuJoCo in lockstep, no ROS; can inject dead time and change weights, or give the MPC the recorded future (`--preview`) |
| `packing_sim.py` / `tray_layout.py` | offline packing comparison; final tray layout of a run with its unusable gaps |
| `place_check.py` / `place_depth.py` | sideways error during each descent into the tray; depth below target |

Per-tick telemetry: `MANIPOPT_TELEMETRY_DIR=<dir>` makes the plant, the
controller and `task_node` each write a CSV (`sim.csv`, `mpc.csv`,
`task.csv`) and the plant save the decision window.

---

## 12. Libraries

| library | used for |
|---|---|
| **ROS 2 Humble** (`rclpy`, `launch_ros`) | nodes, topics, launch files |
| **MuJoCo** (+ Menagerie Panda) | physics, rendering (depth, RGB, segmentation), viewer |
| **acados** (`acados_template`, HPIPM, BLASFEO) | code-generated OCP solver: SQP_RTI, IRK, partial condensing, soft constraints |
| **CasADi** | symbolic dynamics, kinematics and costs, exact derivatives for acados |
| **Pinocchio** (+ `pinocchio.casadi`) | robot model from MJCF, ABA dynamics, forward kinematics, gravity torque |
| **Ruckig** (community) | online jerk-limited trajectory generation |
| **NumPy / SciPy** | heightmaps, segmentation, detection, tracking, packing |
| **OpenCV** | the decision window |
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
scan as 2 cm columns. It was not re-tested after this round's plant changes,
and it does not use the new row packing or the new decision window
(known_issues.md G2, H7).

---

## 14. What is sensed, what is given (honesty section)

**Sensed or planned online, no ground truth and no recorded paths:**
which box to pick and where to grasp it (wrist-camera heightmap); where to
place it (row choice with look-ahead over the sensed pile, checked against
the sensed tray); every reference trajectory (Ruckig, computed live between
points chosen from sensing); every torque (the MPC solve); people and
objects (ceiling camera only; the controller and supervisor never read the
actor's true pose).

**Given in advance, as a real cell would be set up:** the pile area, the
tray's walls and floor height, the ceiling camera's masks for the pile, tray
and aisle, and both cameras' calibration; fixed vantage points (the two scan
positions) and heights (lift 0.45 m, scan 0.5 m); the packing rule (compact,
flush against walls and placed boxes, a 90° turn allowed); the ±90° heading
rule (an empirical joint-7 relation); tuned weights, margins and separation-distance
parameters.

**Read from the simulator (stand-ins for sensing or hardware that is not
modelled):**
1. The grasped box's size and its offset in the gripper
   (`/sim/grasped_box_size`, `/sim/grasped_box_offset`): a top-down camera
   cannot see a box's height; a real cell would measure both in hand.
2. The grasp is a weld: on `pick_at` the plant attaches the true box nearest
   the sensed grasp point (within 2 cm sideways, 8 cm vertically). No gripper,
   contact, friction or slip, so a grasp within tolerance always succeeds.
3. Boxes have collisions disabled and rest on welds (the pile is not
   physically stacked). Nothing physical stops a bad placement or a lift
   through a neighbour; overlap and gaps are measured and logged instead.
4. The simulator steps in lockstep with the controller: if a solve is late,
   simulated time waits. "No dead time" is a property of this setup; on
   hardware the loop would have to meet its deadline (median solve 2.5 ms,
   p99 ~8 ms, against a 20 ms tick).

---

## 15. Limits and open problems (see `known_issues.md`)

- The carried box is not in the controller's model (it sags 1-4 mm).
- Packing is online with at most 3 boxes of look-ahead; with mixed sizes,
  about 18% of the tray ends up as gaps no box could use. A box that fits
  neither row goes to a free spot or on top.
- In the pile, cbox_6 rests partly on cbox_0, which is picked first and lifted
  through it (boxes have no collisions).
- The dashboard renders pause the plant thread (~85 ms per scan); with
  lockstep this only pauses the simulation.
- Obstacle handling is a software stand-in, not a safety function; one fixed
  camera; the pile and tray regions are blind to it; the protective-distance
  parameters are defaults, not measured stopping performance.
- WSL2: the system clock jumps ~1.5 s every ~34 s (handled), software
  rendering only.

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
   limits, soft collision constraints), acados SQP_RTI at 2.5 ms per solve.
5. **Why timing matters**: dead time from two unsynchronised timers →
   lockstep. Figure: joint speed before/after (plots from `sim.csv`), and the
   replay experiment (lag injected → instability).
6. **Reference preview**: the MPC gets the future path; the sideways
   tracking error drops from 5-8 mm to under 1 mm.
7. **Pack**: two rows, 3 mm apart, planned with a look-ahead. Figure: tray
   layout before (scattered, overlaps) and after.
8. **People**: detect, track, classify, speed and separation monitoring
   (the formula), hold and smooth resume. Figure: gap vs required distance
   and the speed scale over a visit.
9. **Numbers**: the before/after table.
10. **Limits and next steps**: payload in the model, better look-ahead,
    certified safety is out of scope, the MoveIt comparison.
