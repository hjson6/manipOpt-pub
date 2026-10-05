# Known issues and things to look at later

Things noticed while building the obstacle-sensing work that are
weird, fragile or unfinished, and that are **not fixed** (unless marked
*Fixed*). Each entry says what was seen, the evidence, and a suggested next
step. Add to this file rather than letting things live in chat history.

## A. Pre-existing behaviour (seen with obstacle sensing switched off too)

**A1. First-move flake.** After `/mpc/go`, roughly a third to a half of my
launches ended with `acados solve failed (status=4)` spam or joint speed
oscillating at ~6.5 rad/s that never settles. A restart clears it.
- Evidence: seen on ~15 of ~35 starts. An earlier A/B on vanilla code (see
  `handover_notes/obstacle_sensing_handover_note.md`) says it is not caused by the obstacle
  work; I did not measure whether sensing changes the rate.
- Next: find why the first solve from home fails (initial guess / warm start /
  home pose against the orientation cost). A start-up retry is a workaround,
  not a fix.
- *Resolved*: dead time from two free-running 20 ms timers (plant
  and controller) whose phase was set by chance at each launch; the first
  move (a 27 cm goal step, since the reference had already run before
  ENTER) is where it blew up first. Offline replay reproduces it: 100% late
  ticks -> ACADOS_MINSTEP, ~50% -> 8-10 rad/s oscillation. Fixed by the
  plant/controller lockstep, the reference starting from the measured TCP
  when the controller starts, and a solver guess initialised from the
  measured state. See design_notes.md "Lockstep plant, one clock for the
  task". Also found: with a second `/mpc/go` subscriber, the start script's
  `ros2 topic pub --once` (waits for one subscriber) could miss the
  controller; task_node now starts on the controller's first solve instead.

**A2. Post-scan swing on the local move.** At the start of the local move after
the destination scan (arm leaves the tray scan pose for the release-above
point) the TCP swings 15-20 cm sideways for ~0.3 s at ~5 rad/s, once per box.
- Evidence: qdot spikes of 5-6 rad/s at the same point of every cycle, also with
  sensing off. Suspect: the orientation cost switching on at the
  `hull_active=False` transition.
- Why it matters: with a static carton near the transit it brings the padded
  TCP proxy up to 7 cm into the obstacle for ~0.3 s once per cycle. It is not a
  person problem (the robot is stopped then).
- Decision: documented, not chased.
- *Resolved*: it was the orientation cost switching back on after a
  travel leg with it off. With the orientation held on every leg (level carry)
  the swing is gone: peak joint speed 7.6 -> 4.4 rad/s.

**A3. Blocked or full tray was retried for ever.** *Fixed*: `task_node` now gives
up after 30 consecutive no-spot scans (`DEST_NO_SPOT_MAX_RETRIES`), logs an
error and stops with the box held over the tray. Still open: it does not
recover if the tray is cleared afterwards (needs a restart).

## B. Simulation and real-time

**B1. Perception and physics share one executor thread.** *Fixed*: cameras, lidars and windows run in their own process (`sim_sensors_node`); the plant steps physics on its own thread in real time. The workspace camera
cycle (~12 ms at 10 Hz) and the wrist-camera dashboard render (~85 ms) run in
the same single-threaded node as the 20 ms physics timer. Seen: a 60 ms stall
made one sensed frame look 0.27 m off in the monitor; the first version of the
camera (visual meshes, ~42 ms/cycle) slowed the sim clock (fixed by rendering
collision hulls).
- Next: render on its own thread or process, or drop the dashboard for
  benchmark runs.
- *Mitigated*: with the lockstep a render stall pauses the
  simulation instead of disturbing control (staleness is counted in steps,
  the plant waits for the controller's answer after a stall, and task_node's
  reference waits too). Still true: the viewer freezes for the render
  (~85 ms per scan), and after a stall the plant catches up at most 100 ms.

**B4. Periodic ~0.8 s system-wide pause, about every 31.5 s.** Seen in every long
run: detections, `/mpc/speed_scale` and `/sim/joint_states` each show a ~0.8-0.9 s
arrival gap, in independent processes, with capture timestamps contiguous
(capture gap 0.10 s). It is not the sim's frame timer, not the supervisor's loop,
not reliable-QoS retransmission (best-effort changed nothing) and not shared-
memory transport (UDP only changed nothing). Most likely the WSL2 VM/host
pausing. Effect: the supervisor's perception fail-safe (0.5 s) trips, the arm
gets a HOLD for ~40 ms while moving (visible as `perception down` errors and
`supervisor HOLD/RESUME` lines with no track), and any tighter timing claim in
this environment is suspect. Kept the 0.5 s timeout: a real 0.8 s data gap is
a valid reason to stop. Left the detections topic on best-effort QoS (right for
a sensor stream anyway).
- Next: check on a native Linux machine; if it disappears it is the VM. Worth
  checking whether the same pause explains A1 or A2.
- *Resolved*: not a pause. The system clock (time.time(), ROS
  time) jumps forward ~1.5 s every ~33.7 s while the monotonic clock shows
  the processes running normally (20-30 ms between physics steps across
  the jump): WSL2 re-syncing a slow guest clock. Anything timing durations
  on the ROS clock saw a fake 1.5 s gap. The supervisor's perception
  fail-safe and task_node's supervisor-silence check now use the monotonic
  clock (0 perception-down HOLDs in a full obstacle run, was one per ~31
  s); the plant's stale-command check counts steps. Cross-process
  timestamps (capture time, latency) are still ROS time, and so still see
  the jump once per 34 s (e.g. one "frame timer late" warning).

**B2. Renderer quirks.** A second `mujoco.Renderer` of a different size in the
same process breaks segmentation output (IndexError in `render()`); comparing an
int array to the pybind enum object is ~100x slower than to `int(...)`. Both
worked around; keep in mind if the perception code is refactored.

**B3. Obstacle pose lags the published truth by ~20 ms.** The mocap pose is
applied one tick after the actor publishes it, so the sensed-vs-truth error at
1.6 m/s (~29 mm mean in x) is almost all this lag (residual ~9 mm). It is not a
perception error, but a tighter accuracy claim needs a truth read from inside
the plant.

## C. Obstacle handling: decisions and limits

**C1. A person who stops became "static" after ~3 s.** *Fixed*: a
track that has ever walked, or anything 1 m tall or taller, is a person for
good. Side effect to keep in mind: a carton that was carried in and set down
stays "person" (the robot waits instead of detouring) until its track is lost.

**C2. Hold/resume chatter with a person who keeps re-entering.** A ping-pong
sweep makes hold/resume alternate about every 0.5 s and the arm creep forward
(3 boxes in 120 s). Safe, but a minimum hold time would calm it.

**C3. Resume is delayed ~1 s after the person leaves** (track coast timeout).

**C4. The detour planner is deliberately simple.** One via-point, straight
segments, lateral or over-the-top. Limits: the outer lateral detour reaches
~0.7-0.77 m (the Panda's limit at carry height), so in this tight cell only a
small carton (r ~0.07) has a detour; "over" never wins with a single via (it
needs a rise-pass-descend pair); elbow and wrist links are left to the soft OCP
constraint; via legs ignore the pile/tray hulls; a tall object is described by
its top-cap sphere only.

**C4b. Soft, local OCP avoidance stalls in front of an obstacle on the straight
line** (the goal pulls straight through it). This is why the planner exists;
without it the arm sat ~2.5 s in front of the carton.

**C5. Single fixed camera.** *The ceiling camera is now optional; people are sensed by two lidars by default (L12).* It is overhead, so the arm partly hides an object it
passes over (the blob shrinks to r~0.05 and drifts 4-6 cm). Handled for static
objects by latching the estimate at confirmation; a person hidden behind the arm
is not handled. The pile and tray regions are masked, so an intruder there is
invisible to this channel. A second camera is the fallback the spec names.

**C6. No geometry-level collision check.** "Gap" in my tests is the distance
between the true obstacle and the supervisor's proxy spheres (the TCP proxy is
0.10 to include a held box). Real overlap between gripper/box meshes and the
obstacle was never measured.

**C7. The supervisor-death fail-safe** (`task_node` holds if `/mpc/hold` goes
silent for 1 s) was exercised once by an accidental start-up race, never by
killing the supervisor on purpose.

**C8. Protective-distance numbers are untuned.** `uncertainty_m = 0.05` is a
guess; human speed 1.6 m/s and `a_max = 3.0` are defaults (spec / Ruckig
limits), not measured stopping performance of the MPC-tracked arm.

## D. Repo and tooling

**D1. My test harness is not in the repo.** The start-with-retry script, the
per-tick watcher (true gap, hold, qdot, goal speed) and the CSV analysers lived
in the session scratchpad. The watcher is worth turning into a script under
`scripts/`.

**D2. `pytest` fails to start in the conda env** (a `launch_testing` plugin hook
error). Run with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.

**D3. Wrong script for the obstacle scenario.** `scripts/container_pickplace.sh`
runs the plain demo (no obstacle at all). *Fixed*: the obstacle demo
is `bash scripts/container_pickplace_obstacle.sh [launch args]` (the old
handover note claimed it existed; it did not). Launch args (`ros2 launch
pick_place_mpc demo_obstacle.launch.py`): `obstacle:=walk|
static|none`, `trigger`, `delay_s`, `speed`, `passes`, `pause_s`, `duration_s`,
`static_position`, `csv_path`, `sup_csv_path`, `workspace_sensing`.

**D4. My cleanup script missed `obstacle_supervisor_node`** (its process-name list
predated that node), so every supervisor from every run since phase 2 stayed
alive, running old code and publishing `/mpc/hold`, `/mpc/speed_scale` and
`/mpc/static_obstacle` next to the live one. Phase 3 and 4 results were
therefore re-measured after killing them all (they matched). If results ever look
odd, first check `ps aux | grep manipOpt/install` for leftovers. Related: a
cleanup command that greps process names kills a shell whose own command line
contains those names; keep it in a script file.

## E. Geometry and scenario

**E1. With the tray ~155 deg from the pile the transit is a chord over the base.**
*Resolved* by arc transit points (see design notes); kept for the
history.
The pile-to-tray reference is a straight Cartesian line, and for a ~155 deg
swing it passes within ~0.11 m of the base axis (measured: minimum TCP radius
0.05 m while z > 0.5), so joint 1 swings fast with the TCP going over the base,
not around it on an arc. Consequences: (a) a person cannot "walk through" the
TCP path itself, only past the arm's outer links (the actor walks a tangent
line at r = 0.5 m and the arm holds on every pass, min clearance 0.10 m); (b) a
carton "on the transit" would have to be next to the base, so the earlier
static-carton detour test does not carry over: `_plan_via` finds no detour for
cartons on the pile side of the chord and none that is sensible near the base.
The carton scenario (`obstacle:=static`, default position (0.32, -0.35, 0.55))
is therefore **not validated** on this geometry.
- Likely fix: an explicit transit via-point on an arc at r ~0.5 m (fly-by), so
  the working path really is a corridor in front of the robot. That changes
  the arm's motion for the whole task, so it needs a decision.

**E2. Actor speed was conflated with the safety assumption.** The first
person actor paced back and forth over 0.45 m at 1.6 m/s (the supervisor's
conservative human speed for the protective distance), which looked frantic.
Now separate: the actor walks at 1.1 m/s (`speed:=`), the supervisor still
assumes 1.6 m/s (`human_speed`).

**E3. The person was a 0.10 m-radius sphere at z = 0.55.** *Fixed*:
a 1.68 m body model (legs, torso, arms, head, facing its walking/looking
direction); detections and the supervisor's gap model are floor-to-top
columns. Still simplified: rigid, no swinging arms or bending over; the
person's route and standing spot are hard-coded in `dynamic_obstacle_node`.

**E4. Camera coverage of a standing person.** At 3.5 m with a 70 deg field of
view the full height of a person is in view only within ~1.2-1.6 m of the
image centre (0.0, -0.20); further out only legs/lower body are seen, so a
person entering the cell is at first reported up to 1.2 m too short and with
a small radius. Harmless here (they are far from the arm then), but a wider
cell needs a wider lens or a second camera.

**E5. Carton scenario not re-validated since the tray moved and the boxes were
scaled up**, and the carton is still a floating sphere at carry height, not
something on the floor or on a cart. The detour planner (C4) works on the
straight segment to the next arc point, which it was not designed for.

**E6. The arm does not plan around a person, it only waits.** In the visit
scenario it can hold with a box in the gripper for as long as the person
stands there (6 s by default). By design (spec: never squeeze past a person
with a payload), but there is no timeout or "go and do something else"
behaviour, e.g. picking the next box while the tray is occupied.

## F. After the 1.5x box scale-up and the half-scale person

**F1. The carried box is not in the controller's model.** *Fixed* (G1). It sags the arm
~4 mm below target (5.8 mm total at the tray scan point), just outside the
old 5 mm "arrived" tolerance, so the arm never counted as arrived. Worked
around by raising `SETTLE_DIST_TOL_M` to 8 mm. Proper fix: payload
compensation (known box mass from the grasp, gravity feed-forward or a
payload term in the OCP model), as real robot controllers are configured.

**F2. Heightmap bugs exposed by the bigger boxes (fixed).** (a) Box tops were
grouped into 1 cm height bins by rounding; a top lying on a bin boundary
(0.195 m) split in two and was rejected. Now neighbouring cells join if their
heights differ by at most 5 mm. (b) The parallax-corrected height of a floor
cell hidden behind a tall box never converged (alternated floor / box top),
and the last pass's parity decided; half the shadow was painted onto the box
top. Two boxes were never detected and grasps were up to 18 mm off centre.
Now a final consistency check reports unseen cells as floor; grasps within
~4 mm. Unit tests for these two cases are still to be written.

**F3. Everything fits on the first layer.** The tray is 0.43 x 0.28 m and the
eight 1.5x boxes all fit on its floor, so the second-layer stacking path is
not exercised. A 2 x 2 tray (0.28 x 0.28) would force stacking but may leave
no flat spot for some size combinations (see A3's timeout).

**F4. The first-move flake got more frequent.** In one start 5 attempts in a
row flaked. Not investigated (see A1). *Resolved with A1.*

**F5. `walk` mode is untested since the rescale.** The crossing line now meets
the arm's carry arc; unlike `visit`, the walking person does not stop short
of the robot, so they can walk into a stopped arm.

**F6. The robot's orientation cost uses one weight for tilt and heading.** A
stronger tilt weight than heading weight would be the natural next step if
tilt must stay below ~1 deg while the box turns.

## G. Control smoothness work

**G1. Payload still not in the controller's model (F1 unchanged).** The 8 mm
settle tolerance stays. Payload compensation is still the proper fix.
*Fixed*: payload mass in the OCP from the wrist load cell, plus
offset-free tracking; the tolerance is back to 5 mm (implementation notes,
plant_mismatch.py).

**G2. The MoveIt method's bridge_node does not echo the state stamp** on its
commands. The plant takes an unstamped command as the answer to the latest
state (the bridge runs one step per state), which keeps it working, but it
is not strictly locked. Echoing `header.frame_id` in `PublishCommand` would
make it exact. Not re-tested live after the plant change.

**G3. Simulated time can fall behind wall time** after a long stall (the
plant catches up at most 100 ms). task_node now runs on plant steps, so the
task is consistent with the arm; only the wall-clock duration of a run
stretches.

**G4. Tooling.** `scripts/dev/bench_run.sh` (one launch of the real demo
script with per-tick telemetry, `MANIPOPT_TELEMETRY_DIR`),
`bench_batch.sh` (N launches of each script), `bench_analyze.py` (one line
of numbers per run), `replay_offline.py` (lockstep acados + MuJoCo replay of
a recorded goal stream, with optional injected dead time).
`start_until_good.sh` is no longer needed.

## H. Placement and packing

**H1. Boxes swung into the tray walls/neighbours on the way down** (3-4 cm
sideways below the wall tops) after pass-through waypoints were introduced.
*Fixed*: arriving corners are passed moving straight down, and the MPC now
gets the reference's future per stage (task_node.HORIZON_STEPS), which
removed a steady 5-8 mm sideways lag on every descent. Measured per place by
mujoco_sim_node's "placed" log line (overlap at release, worst while
carried, gap on each side) and `scripts/dev/place_check.py`.

**H2. Boxes were set down 2 cm above the tray floor.** *Fixed*:
`scene.DEST_FLOOR_Z` was the old layer-0 constant (0.020), not the floor's
real top (0.0). Also shared with the MoveIt method's decision_node (not
re-tested there).

**H3. Visible gaps / small overlaps at walls and neighbours.** *Fixed*:
placement is snapped flush (3 mm) against the known walls and the boxes this
node placed, and targets the box centre using the in-hand offset
(/sim/grasped_box_offset; the grasp point from a 5 mm heightmap leaves the
box a few mm off centre). Now every facing side is 2.6-3.6 mm, no overlap.
/sim/grasped_box_offset is a ground-truth read standing in for in-hand
sensing, like /sim/grasped_box_size.

**H4. Placement is two rows ("shelf") with a short look-ahead.** *Superseded*: compact packing (I), then touching-perimeter packing (L6). Chosen
with the user after comparing free-form rules (most contact, fewest slivers,
compact from a corner, largest free rectangle), which left holes and slivers
between groups of boxes (`scripts/dev/packing_sim.py`, `tray_layout.py`).
Each box goes flush after the last one in the back or front row; the row is
chosen looking ahead over the boxes the pile scan can see (at most 3). A box
that fits in neither row goes to the heightmap search (floor if there is
room, else on top). With mixed sizes and an unknown pick order there is no
gap-free layout. Rotating boxes 90 deg was simulated for the free-form rules
(contact 1.44 -> 1.50 m, no gain on the demo's order) and not done.

**H5. A pile box is lifted through the box resting on it.** cbox_6 overhangs
cbox_0 by 1.5 cm in the pile; cbox_0 is picked first (its visible top is the
larger), so it passes 15 mm through cbox_6 on the way up (boxes have no
collisions). The pick order should skip a box another box rests on.
*Fixed*: `heightmap.pick_order` (used by task_node and the MoveIt
decision_node) puts a top that touches a higher detected top after all others,
since the camera cannot tell "rests on it" from "stands beside it". Pile order
is now 3, 5, 1, 6, 0, 4, 7, 2 (cbox_6 before cbox_0); verified offline on the
scene geometry and in one live obstacle run.

**H8. Stall at a place point after a hold during the final descent.** Seen
once: a HOLD braked the reference 4 mm above the place point, the
arm settled there (pulse correctly ignored, not at target), and after RESUME
the reference crept the last 4 mm at < 1 mm per tick. The controller re-armed
settle detection only on a > 1 mm jump between ticks, so it never pulsed
again and the box was never released. *Fixed*: while settled, the goal is
compared with the goal it settled at. Not yet reproduced live with the fix
(the next run's holds did not fall in a descent).

**H6. `panda_scene_container.xml` has "--" inside a comment** (line 11), which
strict XML parsers reject (MuJoCo accepts it). *Fixed* in the comment cleanup;
the file now parses with a strict parser.

**H7. One decision window, fed by the decisions actually made.** The pick-up
and placement windows are now one ("pick & place decisions"), drawing what
task_node publishes on /task/decision (they used to redo the decision for
display, and the placement one still ran the old heightmap search). The
MoveIt method's decision_node does not publish /task/decision yet, so with
that method the window shows the scans with "deciding..." only.
A second, live window ("obstacle detection (live)") shows the ceiling
camera's obstacle detection (pixel classes and blobs) and the supervisor's
tracks and decision (/supervisor/tracks, display only), redrawn at 2 Hz. In the plain demo no
supervisor runs, so its half says "no data".
Cost of the window, measured with the plant alone and the viewer on: the
event pump used to run on every physics step (~2 ms each, ~10% of real
time); at 5 Hz the live row slowed the ceiling camera's cycle from 13 to
51 ms and physics steps to up to 200 ms. Drawing costs ~2 ms; showing the
image is what hurts, and not only inside the plant's process: shown from a
separate process it still interacts with the MuJoCo viewer through the WSLg
compositor (camera cycle 43 ms at 5 Hz with the viewer, 20 ms without).
Now: composed on a thread, shown by its own process, at 2 Hz (camera cycle
24 ms, physics steps as regular as with no window), and in its own 640x240
window so the 2 Hz updates no longer resend the four wrist-camera panels
(that window is redrawn only on scans and decisions). The GPU (d3d12 in WSL2,
RTX 5060 / Arc 140T) is no help here: small offscreen renders with readback
are 5-6x slower through it than in software, and its depth images differ at
edges by up to 0.77 m from the software renders the perception is tuned on.

## I. Contacts, place-then-push, compact packing, sensed boxes

What changed is in `handover_notes/realism_plan.md` (steps 1-3) and the
implementation notes; what is still open or worth knowing. The packing
strategies (rows, compact, compact with a 90 deg turn) are compared in
`figures/packing_compare.png` (`scripts/dev/packing_compare.py`).

**I1. Tray walls vs wrist and forearm.** With contacts on, the wrist (link 7,
up to 8.8 cm past the TCP) lands on the 22 cm walls when a box is set flush
against one, and at pushing height the forearm (links 5-6, ~14 cm above the tip,
reaching toward the base) lands on the walls on the robot's side. Handled by
setting the box down clear of the wall and pushing it back with the tool tilted
30 deg in the plane of the pushed face (IK search: 27 mm clearance at a
back-row push), the lean chosen from the scan, and by the packer only choosing
spots the robot can finish. Wrist extents are a fixed table
(`scene.WRIST_EXTENT_TOOL`); the forearm is a rule (flange >= 9 cm from a wall
on the robot's side), not a model. The forearm-on-wall contact:
`figures/issue2_forearm_wall.png`.

**I2. Final gaps vary (0.2-6 mm, target 3 mm).** A box measured large is
pushed too little, one measured small too far; the re-measure from each tray
scan keeps the records right to ~1-2 mm, but the push itself uses the sensed
size. The tool touched a wall once for one step (27 N) at the end of a push.

**I3. Figure-3 packing is not reachable with what the camera sees.** Planned
with the true next boxes, all 8 fit on the floor; with the visible tops only
(cbox_1 is under cbox_5, cbox_2 comes last) the last box is stacked. More
information (a max box size spec, a box list) would change that.

**I4. The heightmap projection uses pixel index, not pixel centre.** Fixed in
`top_extent` (it cost ~1.4 mm, opposite signs in x and y), not yet in
`infer_heights_parallax_corrected` / `project_world_point`: heightmap cells are
~1.4 mm off. Grasp point and sizes now come from `top_extent`, so the effect is
small; fix and re-tune together.

**I5. The MoveIt method still reads the simulator** for the grasped box size
(`/sim/grasped_box_size`, kept for it) and has no push, touch-down or tray
re-measure. Not re-validated since contacts were switched on.

**I6. Blocked places recover by re-planning** (lift, rescan, footprint grown by
2.5 mm per side, at most 2 retries, then stop with the box held). Not yet
triggered in a live run.

**I7. Found by the depth-noise step (fixed).** (a) A box top's height was the
max of its cells, biased ~2 mm up by noise, and the grasp stopped that far
above the top (inflating the touch-down height): now the median. (b) The H5
pick-order check only looked at detected tops; under noise cbox_6's top once
failed detection, cbox_0 was picked with cbox_6 on it and cbox_6 rode along
243 mm. Any higher cells touching a top now count. (c) The side-force guard
also fired on the floor's friction at touch-down; it now only counts while the
wrist still carries >= 80% of the box's weight. (d) With noise the tray
re-measure is within ~3 mm, so the packing gap went from 3 to 5 mm.

**I8. The MPC is sensitive to the wrist's rotor inertia.** With joint 7's
armature at 0.7x the model's value (the plant ~30% livelier than predicted),
the controller chatters on joint 7 at the torque limit every tick (offline
replay: 99.5% of ticks; 0.8x and 0.9x are fine). The mismatch range for
armature is therefore +-15% (a datasheet value), not the +-30% used for
damping. A robustness target for later work (e.g. a torque-rate penalty on
the wrist, or learning). Joint 7 at 0.72x vs 0.86x:
`figures/armature_30_vs_15.png`.

**I9. Pushes are open-loop.** *Partly fixed (414bc5d)*: the pushed box's record is now
set from where the tool stopped and re-measured wider on the next scan; the push itself
still aims at the plan. A push aims at where the box was planned to be
set down, not at where it settled after release (settling moves it ~1-2 mm, so
it works). A quick look before the push would close the loop at a few seconds
per push. Push points are never predefined: side, distance, start point,
height, lean and centring are all computed per box from the scan, the
measured box and the robot's geometry.

**I10. Tray full before the largest box is free (obstacle seed 1).
Fixed:** when the floor is full, the box now goes where it rests
stably (heightmap.find_resting_footprint): on a box, bridging boxes of the same
height, or with a small overhang, either way round. On the tray the step-4
obstacle run stopped with, the old search found nothing; the new one puts
cbox_0 (150 x 120 mm) on cbox_1 (135 x 135 mm), 84% supported, and dropped there
in MuJoCo it settles level (0.0 deg) without moving the others. History:
The largest box (cbox_1, 135 x 135 mm) is under others, so it is picked last;
by then the floor layout (cbox_5 placed turned 90 deg) left no 135 mm spot and
the run stopped after 30 no-spot scans with 6 boxes moved. The plain run on the
same seed finished 8/8. In the baseline batch (`results/mpc/`, 60 runs) this is
the only way a run failed: 9 runs, on seeds 1, 6 and 8 with mismatch (both
scenarios) and seed 10 in the oracle; whether a seed hits it depends on the
run's timing. Next: let the placement reserve room for boxes still under
others (their footprint is partly visible), or allow stacking a big box on a
flat group of smaller ones.

## J. Sensed cell (`handover_notes/sensed_cell_plan.md`)

**J1. A tray scan with the box held sees almost nothing.** The wrist camera is
0.25 m above the held box's top, so the box hides 93-97% of the tray (49% for
the 60 mm box). Hidden cells keep the last empty-hand scan's heights, except
round the blocked set-down (+-2 cm), which are unknown; the plan said hidden
cells are always unknown, which would make every such rescan find nothing. The
30 no-spot retries in that state are pointless (the same cells are hidden each
time). Next: a fixed camera over the tray, or a rescan pose that puts the spot
outside the box's shadow.

**J2. The pick zone's grid phase changed the pick order.** The new zone edges
moved the 5 mm grid by 3.75 mm. The "touches a higher top" test (H5) uses a
one-cell ring, so a real 7 mm gap between cbox_1 and the higher cbox_2 reads
as touching or not depending on the phase; with the new phase cbox_1 goes last
on seed 2, and seed 2 hit I10 (tray full) in the fix 1 and fix 2 checks (5-7
of 8), where the baseline finished 4/4. Later checks on seed 2 finished 8/8
(other heights, other timing). I10 is still the main way a run fails.

**J3. A small stacked top can fail detection at the lower pile scan.** 60 mm
cbox_7 on cbox_2: 1.5% of noisy frames found no top (0% at the old 0.5 m
scan and grid). A scan that sees something but no top is now repeated up to 3
times before the task parks.

**J4. Wrist vs taller neighbours at a pick is a skip, not a fix.** A top whose
link7 footprint would hit a taller neighbour is skipped; if every top is
blocked, it picks the first anyway with a warning. No grasp shift or heading
choice for wrist clearance.

**J5. Joint 6 reaches ~2.5 rad/s on some pushes** (once per run on seed 2,
~1.35 s after a push starts, while the tool tilts). Not investigated; the
joint-7 whip after turned placements is fixed (implementation_notes,
task_node.py).

**J6. Facing gaps ~1 mm wider after fix 3** (median 7.0-7.2 mm vs 5.7-6.3 on
seed 2). Cause not looked at; the tray is now scanned from lower (0.30-0.40 m
TCP instead of 0.50 m).

**J7. Wall touches with a shifted tray.** Tray seed 8 (+49 mm towards the
base): one run had a held box pressed on the south wall rim (376 steps, 41 N,
a blocked place that recovered), the other a 1-step tool tap (72 N). In the
baseline's range (0-2 wall contacts per run).

**J8. "boxes disturbed" after an aborted place counts the held box itself**
(e.g. "cbox_4 1032 mm"): the plant's report does not exclude it.

## K. Realistic cell (`handover_notes/realistic_cell_plan.md`)

**K1. Sensor noise slows settling.** With encoder noise (5e-5 rad) and the
driver-style velocity (difference over 8 ms), the arm comes to rest later after
the reference stops: settle time median 0.50 -> 0.58 s (plain) and 0.60 s
(obstacle), p90 0.75-0.84 -> 1.14-1.22 s (seed 2, one run each). Arrival error
is unchanged (0.3 mm median); the true joint speed takes longer to fall below
the 0.05 rad/s settle tolerance. Offline replay: the 4 ms velocity lag alone
raises the joint-speed jitter (`hf`, bench_analyze) from 0.038 to 0.063 rad/s, the
encoder noise to 0.081. Candidates: a model-based velocity estimate in the
controller (the MPC's own one-step prediction fused with the measurement), or
a settle rule on a filtered speed. The MPC tolerates less than about 6 ms of
velocity lag before joint 7 chatters (implementation_notes, mujoco_sim_node.py).

**K2. Costs of the wrist camera on the arm.** (a) Box footprints are about one
pixel coarser at worst: 2.0-2.7 mm (was 1.5 mm with the virtual camera 15 cm
above the flange), edge quantization at 2.5 mm pixels, inside the 5 mm placement
clearance. (b) The first tray scan takes two pictures and the tray scan leg is
slowed when the heading turns back: time per box +1.5 s (plain) and +3.0 s
(obstacle), seed 2. (c) A tray rescan with a box held sees about 30% of the tray
(J1: the virtual camera saw 3-7%). (d) The pick zone takes piles up to 0.35 m
(was 0.40): the first pile scan has to fit one picture within the arm's reach.
(e) Two same-height tops a few mm apart can merge in the heightmap (the gap is
about one pixel); the footprint from the depth pixels still finds the right box
(10/10 offline), but the pick order then counts both tops' area.

**K3. Heavy boxes are not always pushed flush in real time.** The guarded push
(stop when the box stops, past the planned end, or at 40 N) leaves light and
medium boxes 1-3 mm from the wall; the heaviest (1.8-1.95 kg, seed 2) stopped
15-23 mm short, one at the 40 N cap, one at 20.6 N. In lockstep the same pushes
reached the cap with the boxes already flush. 40 N is well above sliding
friction (0.6 x 19 N); the tool may also press the box down or into a neighbour.
Next: look at the push force direction and the cap with the box spec's heaviest.
*Seen again with the light boxes (W1)*: the far-corner push stops at 40 N
30-40 mm short; the tool sliding sideways along the face was one cause (fixed, 414bc5d),
the rest is not reproduced offline (`push_sim.py`).

**K4. The tracking metric reads high in real time.** `mpc.csv`'s `ee_err` is
the TCP against the controller's goal, which in real time is one step ahead (the
torque starts 14 ms after its state), so it reads ~7 mm at 0.5 m/s. Aligned in
time (`tcp(k+1)` against that goal), real time tracks at 4.1 mm RMS, 2.8 mm
median (lockstep 5.8 / 1.3 mm, seed 2, headless). `results_summary.py` still
uses `ee_err`.

## L. Motion quality, packing and a genuine pipeline

Measured with `scripts/dev/motion_check.py` (sweeping) and `bench_analyze.py`
(stutter) over 10 obstacle runs, seeds 1-10, real time.

**L1. The wrist wound up and back on every carry.** *Fixed (308bc52)*: the
heading was set per leg, so on a carry joint 7 went -7, -70, +9 deg (2.4 rad of
back and forth, all 10 runs). A run of pass-through legs now turns as one, in
step with the base swing (implementation_notes, task_node.py).

**L2. A short push spun the wrist.** *Fixed (308bc52)*: a 2-3 mm push after a
set-down needed a new heading and a 10 deg tilt over a few cm: 78 deg of wrist
in 0.5 s, 4.8 rad/s, the TCP 96 mm off its reference beside the placed box (3
of 10 runs). Pushes under 6 mm are skipped and every leg is slowed so the tool
turns no faster than 1.5 rad/s.

**L3. Elbow creep during slow set-downs.** *Mostly fixed (17ed658)*: joints 1,
3 and 5 drifted up to 13.5 deg during a 3 s set-down (the plant's torque
mismatch, only the posture term holding that direction); posture weight 1 -> 5
left 3.9 deg. A few degrees remain; a slower creep of 0.15-0.18 rad over ~1.5 s
shows once per run at 15-23 s, harmless.

**L4. Jolt at the end of a stalled push.** *Fixed (17ed658)*: the reference
had run up to 31 mm ahead and was reset to the tool in one tick (3.5 rad/s for
20 ms); the back-off now starts from the reference.

**L5. One-tick speed spikes at some pass-through corners** (e.g. the approach
to a pick turning into the descent): up to ~3 rad/s for 20 ms. Not
investigated.

**L6. Packing.** *Fixed (dc9eb53)*: the first box (76 x 105 mm) could reach no
corner flush, so the old packing fell back to any free spot; the arm-wall check
treated the flange as a 7 cm ball and refused every push along a wall. Now the
packing ranks by touching perimeter with look-ahead and scores a spot where the
robot really leaves the box; the flange is left to the precise wrist check.
Offline: all 8 on the floor in 76% of random orders (was 29%). Still open: the
corners on the robot's side refuse pushes towards the robot's wall (the
forearm), and the pick order is the pile's, not chosen for the packing.

**L7. Joint-7 limit jam at turned placements.** *Fixed (2bd0f09, dba3ae4)*: the
heading's 2 pi branch was chosen nearest to turning with the base, which tied
at 180 deg and drove joint 7 into its -166 deg limit; the arm reached back over
its shoulder and the solver failed. The branch now has to keep joint 7 within
150 deg.

**L8. A box caught under a stacked box's overhang.** *Fixed (dba3ae4)*: spots
keep 5 mm from any placed box whose bottom is above the spot's surface.

**L9. The viewer looked like a low-frame-rate video.** *Fixed (dba3ae4)*:
camera renders on the sensors process's executor held up the scene state for
60-350 ms; it now has its own thread and the viewer redraws every tick.

**L10. Shortcuts removed.** *Fixed (00093a7)*: the plant's grasp welded the box
nearest the requested point, touching or not, and the method never checked it
held anything; the wrist camera's pose came from the plant. Now the grasp needs
contact, the method picks by touch and checks the weight, and the camera pose
comes from the measured joint angles. The miss and empty-lift paths (re-scan
the pile) have not been exercised live yet: no pick has failed.

**L11. Ceiling-camera depth frames were lost.** *Fixed (1583312)*: 0.6 MB
frames over best-effort delivery mostly did not arrive (2 of 10 Hz); now
reliable, keep-last-1.

**L12. Lidar false positive at the visit entry.** One per run at most: the
actor's truth says "gone" a frame before the scene moves the person away; a
monitor artefact, not a detection error. Frames between an appear/leave pair
of truth samples are not scored.

**L13. Camera timer late warnings** (0.3-0.7 s) in the sensors process with all
windows on; present before the lidars (which add about 9 ms of work per
second). The supervisor's per-source watchdog holds if a source is silent for
0.5 s.

## M. Mobile manipulator, step 1: room and mobile base

**M1. The ceiling camera's calibration holds only while parked at the cell.**
`WORKSPACE_CAM_POS` is taught relative to the arm with the base parked at the
cell; the camera is fixed in the room. Once the base drives, it needs the
localized pose (or stays a cell-only sensor). The lidars move with the base and
are the main people sensor anyway.

**M2. Odometry is optimistic.** 0.00-0.11% of the distance with the gyro
(`odom_check.py`), against 1-2% typical on a real AMR: the floor is perfectly
flat, the tyres do not deform, the casters do not shimmy. Wheel-only heading is
realistic (up to 6 deg per loop from the 0.5% radius mismatch). Consider floor
unevenness or tyre deformation before judging SLAM on these numbers.

**M3. Wheel contacts are near-hard pairs, and a parked wheel is pinned.** The
wheel-floor pairs use `solimp` 0.999 0.9999 and a braked wheel's contact point is
pinned to the floor until its force exceeds friction (implementation_notes,
base_drive.py): MuJoCo's soft friction has no static friction, and the parked
base crept up to 0.33 deg per run, which disturbed the lidars' background (a
false person at the place table's leg, grazing beams along the far walls). Watch
for contact chatter at high speeds or on impacts; the pins are a plant model of
stiction, not something the method knows about. The arm's
joint friction still creeps under a steady torque (the plant as before);
MuJoCo's `noslip` would make it stick like real Coulomb friction, but that
changes the arm plant and was left for later.

**M4. Encoder-only odometry heading.** A flat (cylinder) tread scrubbed in
turns, 4-8% heading error per turn; the rounded tread fixed it. If the tread
changes, rerun `odom_check.py`.

**M5. `bench_analyze.py` counted 0 boxes.** *Fixed*: it counted a "scanning
destination" log line that `task_node` stopped printing at b0eec68, so the
per-box columns (stops, dips) were per run, not per box, since then. It now
counts the plant's "placed" lines.

**M6. A one-tick Ruckig failure in the reference preview.** Once in 4 of 11
step-1 runs (with and without the brake; e.g. the gate's plain run, seed 2, the
carry of cbox_7); the fallback that holds the reference has been in task_node
since 20651f2, so it may well predate step 1 unlogged. Example: "reference horizon Ruckig solve
failed ... error in step 2 in dof 0", azimuth -2.156 rad at -0.09 rad/s and
2.6 rad/s^2 towards a target 2 mrad away; `task_node` held the reference for
that tick and the place went on normally. Not related to the base (task_node is
unchanged and works in the arm frame). Next: replay the logged input in Ruckig offline; a target that close with
that much acceleration may need the target nudged or the step retried with the
velocity interface.

**M7. A jolt at grip or release of a heavy box.** Seen once in seven step-1
runs (final plain verification, seed 2, cbox_4, 1.70 kg): right after the grip
the wrist load jumped to -24 N and joints 1-2 hit their 87 Nm limits for a tick
(joint 1 at 2 rad/s for ~40 ms); right after the release the load cell read 0
while the filtered payload estimate still held 1.2 kg, and the tool jumped 23 mm
up. Jitter 0.082 rad/s for that run (0.04-0.05 in the others). The payload path
(wrist load, filter, the OCP's payload parameter) is unchanged by step 1 and the
base stayed still through both (yaw within 0.06 deg); it depends on when the
grip or release falls in the tick. Next: reset the payload estimate at release
(the method knows when it lets go) and ramp it in at the grip, then replay
(`replay_offline.py`) or rerun seed 2.

**M8. A stale packing test.** *Fixed*: `test_packing.py` still called
`plan_compact(is_free=...)`, which dc9eb53 replaced by `where=...` (where the
box would really end up, or None); the test failed since then.

## N. Mobile manipulator, step 2: SLAM and localization

**N1. The test drives are steered on the true pose.** `mobile_scenario_node` (and
`record_drive.py`) drive the routes as a technician with a joystick would; the
method only localizes. Since step 4, `route:=nav` drives on the method's own
estimate; the commissioning (mapping) drive stays a technician's.

**N2. Our mapper's CPU spikes at loop closures.** 7.6 ms mean per scan, p95 about
40 ms, up to 140 ms (Python Gauss-Newton on the whole graph after each closure;
about 300 closures per two laps). Fine at 15 Hz in this room; a larger map would
want the optimization off the scan thread, or fewer closures.

**N3. Scans are instantaneous.** The ray-cast scan is taken at one instant; a real
scanner sweeps over 66 ms, so a turning base distorts its scan (de-skewing). Not
modelled.

**N4. The maps are of the step-1 room.** The cell stands in the middle; when the
tables move to the room's corners (steps 4-5), drive the commissioning route
again.

**N5. The lidar people detector assumes a still base.** It learns a background
range per beam; on a moving base everything is foreground. Step 3 replaces it with
foreground against the map.

**N6. slam_toolbox keeps traces of people** in its map (a few occupied cells where
someone stood); ours traces them out with the free space behind. Both localize
through them.

**N7. Installing slam_toolbox** (robostack) also pulled SuiteSparse and some CUDA
libraries into the environment; the arm's solver was unaffected (2.55 ms per
solve).

## O. Mobile manipulator, step 3: people on a moving base

**O1. Live detection rates include occlusion.** The live monitor cannot tell a
hidden person from a missed one (97.8% within 5 m); the offline evaluation counts
the lidar beams on each person and scores only the visible (99.3-99.6%). Publishing
the beam counts from the plant would make the live figure comparable.

**O2. Anything new on the floor is a person.** A cart or box left where the map
has free floor is foreground with legs-sized clusters (or a big cluster clipped to
leg size): the supervisor and the navigation will treat it as a person. Conservative
by design; a static, non-leg-shaped object could be told apart later.

**O3. One leg seen** puts the person's centre up to ~12 cm off (the p95 error).
Fine for safety distances, which add the leg radius and the ISO allowance.

**O4. The crowd's people pass through each other** (mocap bodies without contacts);
they only wait for someone just ahead, and for at most 2 s. Since step 4 they stop
for the robot's body and step round it after 3 s.

## P. Mobile manipulator, step 4: navigation

**P1. The corner scanners' 5 cm minimum range.** *Fixed*: a leg right at a
corner scanner was in neither scanner's view (the lidar model dropped returns under 5
cm), and once the base crept within 50 mm of a person standing there. The scanners now
see from their front window (no minimum range). That showed their 4 cm housings, 3 cm
outside the chassis, scraping the pick table's legs on the way into the dock; the
scanners' optical centres moved onto the chassis's corners (each 270 deg sector runs
along both faces: all round seen, nothing of the robot) and the pick dock 3 cm
farther from the table (the legs 9.5 cm beside the chassis). Recessing them inside the
outline was tried and rejected: the chassis would hide the sides from both. Offline
the closest approach to anyone standing is now 109 mm or more; no safety stop on the
way into the pick dock (17-45 before).

**P2. Swerves and turning in a crowd.** In 20 drives through six people who do not
all give way the base still swerves 7-10 times (left-right-left within 2 s, heading
swings up to 18 deg), and once in 40 drives two swerves ran together (an
oscillation). Its total turning is up to 500 deg above the plan's (avoiding and
replanning); in a quiet room 11 deg. The track velocities of people turning
and passing each other are noisy (they drive the predictions); a prediction that
knows people walk round each other, or a hysteresis on the side to pass on, would
help.

**P3. Two people passing close swap tracks.** A walker passing within 0.4 m of a
standing person can pull the standing person's track along for half a second; the
MPC then sees free space and the safety layer stops the base (seen once, 4 cm of
creep).

**P4. Parking at home is within 1 deg, not better.** (Kept so when the
docks' pre-dock tolerance went to 3 deg: `PARK_TOL_*`.) The align step accepts 1 deg
and there is no approach after it; docking reaches 0.2-0.4 deg on the approach.
Turning in place by less than a degree is slow to settle (the drives' stiction).

**P5. Unmapped static obstacles are people.** The MPC avoids only tracked people;
anything else new on the floor is either a person (O2) or seen only by the safety
layer, which stops for it; the planner does not route round it.

**P6. Replanning took up to 0.7 s inside a navigation tick.** *Fixed*: the
planner runs in a worker thread in `nav_node`; the base waits for a goal's first route
(a few ticks in "planning") and drives on its current route while a new one round
people is made. Live the ticks stayed within 109 ms of each other (142 ms steps
before). `nav_sim.py` still plans in line, so its runs repeat exactly.

**P7. Real-time stalls trip the watchdogs.** The plant itself stalls for up to 0.5
s under WSL in some runs (six times in one live run, none in another); the scans,
odometry and localization then arrive late and the safety layer holds the base for
that long.


**P8. The map sees the tables as their legs.** The lidars scan 20 cm above the floor;
a table is four leg sections in the map (`figures/nav/nav_sim.png`). Routes keep
off the tables only because their legs stand closer together than the chassis is
wide; a wider table, or a shelf with a low gap, would look passable to the planner
and the safety layer alike. Taught keep-out zones on the map (as industrial fleets
use) would close it.

**P9. Tracks coast on past people walking away.** A track keeps going for up to 1 s
(the tracker's coast) when its person turns away or leaves view at the edge of range;
the people monitor counts it as a false track while it is more than 0.5 m from anyone
(about 36 scan rounds per 3-drive live run, 37 offline in 6 drives). All were 5.4 m or
more from the robot, beyond what the navigation reacts to.

**P10. The protective field is the stop, no more.** It covers what the chassis can
reach before stopping plus 10 cm; a person walking fast into a slowly turning base's
corner can still come close (34 mm once in 60 drives, the base already stopping, no
contact). The old fields (a box widened ahead of a turn, the whole circle for a turn
in place) were more generous there, but the circle held the base for good beside a
table leg.


## Q. Mobile manipulator, step 5: the mobile pick and place

**Q1. While the base drives, people's upper bodies come nearer the arm than their legs
to the chassis.** *Fixed*: the base's fields watch legs 0.2 m above the
floor; a person's torso and arms overhang their legs (up to 0.135 m), and the folded
arm and a carried box reach to within 4-8 cm of the chassis's edge, so passing close
a person's upper body came within 72 mm of the elbow (link3) once. The scan points
that are people (the people detector's foreground against the map) now get 10 cm more
margin in the protective field; walls and tables keep theirs (more for everything
would stop the base beside the pick table's legs). Offline (carry pose, the largest
box held, crowd of 6): the upper body's closest approach while the robot closed on
someone went from 129-168 mm to 219-307 mm. Live with the job's crowd: 192 mm while
closing on someone, 127 mm at any time (people walking up to a robot standing still
for them), the navigation's results unchanged (19/19 docked within 7 mm, no
intrusion, no contact).

**Q2. People at a station slow the job.** The arm holds whenever someone is within
about 1.2-1.4 m of it (speed and separation with leg-only sensing); a person looping
near the place station held the arm two thirds of the time in one run. The job waits
as long as someone stays; there is no escalation (asking them to move, working on
the side away from them).

**Q3. A place now and then is blocked on the way down.** Once in the crowd runs the box
did not reach its hover point (a few mm onto a neighbour); the cell's retry lifted it
and re-planned with the footprint grown, and it was placed. Once (step 6's first quiet
run) the retry found no spot: from the usual scan pose the held box hid the blocked
spot (29-80% hidden for spots on the tray's far side), the blocked area stays unknown
until seen, and 30 rescans from the same pose stopped the job with the box held. The
rescan after a blocked place is now taken with the wrist camera just past the blocked
spot and the box beyond it (0% hidden at 12 positions, offline); not yet met live.

**Q4. The tray's rotation is not re-measured.** The tray is found again at each visit
by its walls, axis-aligned in the arm's frame; a docking turned 0.3 deg leaves about
1 mm at the walls' ends.

**Q5. The job's crowd differs from step 3's.** The bystander by the pick station comes
and goes (30 s there, 30 s away); standing there for good holds the arm for ever.

**Q6. Ruckig's rare numerical failure** (time synchronization near the target with the
limits just scaled) crashed the task once; the step is now taken again from zero
acceleration, or the reference holds for a tick.


## R. Mobile manipulator, step 6: the arm and the base at once

**R1. The unfolded arm is outside what the base's fields watch.** While the base docks
or undocks (0.10-0.15 m/s) the arm may be out over a table or beside the robot; the
base's protective field covers the chassis only. People near the arm are the arm's
supervisor's (speed and separation, now with the base's motion in the robot's speed),
and its hold pauses the base. A person reaching the arm faster than the supervisor's
1.6 m/s allowance, from a side the lidars cannot see (above a table), is not covered.

**R2. The whole-body MPC is evaluated, not used.** `core/wb_ocp.py` docked and tracked
as well as the two controllers apart and was not faster for this job (the station
poses are scan poses, high over the tables); it leaves out the arm's inertia under the
base's acceleration, and its people constraints were not tried against a crowd. A
task that must hold the tool still in the room while the base moves (reaching while
driving past) would need it.

**R3. The arm's first move at a station is planned in its own frame.** It starts on
the straight approach and reaches the scan pose relative to the base, not the table;
should the base stop short (someone in the way), the arm waits there, and the scan
waits for the docked status. Harmless for scans; a first move that touched the station
would have to wait for docking.

**R4. A person standing near the arm can stall the robot where it stands.** The arm's
supervisor holds the arm while anyone is within its separation distance (about 1.4 m
for leg-height sensing, Q2), and the base waits for the arm: at a station, during the
tuck until the arm is over the chassis, and on the approach while the unfolded arm is
out. A person who stays (the scenario people were made to move on when their spot is
under the robot, or when cornered by it) holds the job for as long as they stay; there
is no escalation (asking them to step back, folding away from them).


## S. The room at half its floor area

**S1. From home to the pick station the base aligns again.** Home and the pick
pre-dock pose are 2.65 m apart, both facing north; the route arrives heading west,
turns on the spot and ends about 3 cm off the docking line, more than the approach
absorbs, so the base aligns again (turn, straight, turn, about 7 s; 12 turn reversals
in 20 offline drives). The job never drives home > pick. Since 0e13d73 a dock's
pre-dock tolerance is 3 cm / 3 deg and a larger miss joins on a curve; not re-measured
for home > pick.

**S2. People walk closer past the base in the smaller room.** With four people in half
the floor, people walking at 1.0-1.2 m/s past the front corner of a slow or stopping
base came within 49-58 mm of the chassis offline (no contact, no field intrusion; P10).


## T. Speed

**T1. Parked at home 1.3-1.75 deg off.** At 1.0 m/s the base parks at home up to 1.75
deg off on the truth offline (it was within 1.07 deg at 0.5 m/s; the docks stay within
0.35 deg: the straight approach takes the heading up). Not the alignment's turn rate
(the same at 0.6 rad/s). Home is only where the job ends.

**T2. The step-in person stepped in only once.** In the smaller room the place dock
is 3.0 m from their step-in spot, and they stepped in again only once the robot had
been 4.0 m away; after the first time they stood beside the route. The room-shrink
results for that scenario are of someone standing there. Now 2.9 m.

**T3. With people near, the base waits for the arm's tuck.** In the live crowd run
the base stood off the docking line 118 s in all (67 s at 0.5 m/s) for the arm: the
supervisor holds the tuck while someone is near the arm, and the base may leave the
line only with the arm stowed. Now the largest wait in the crowd.

**T4. More swerves at 1 m/s in the crowd.** Live with the crowd of four: 16 swerves
(the largest 30 deg) and 5 oscillations, against 8 (24 deg) and 1 at 0.5 m/s; offline
up to 15 swerves in 20 drives. The speed near people is unchanged below 1.8 m.

**T5. Someone came within 97 mm of the folded arm while the base drove.** Live with
the crowd (the elbow, link 3; 121 mm at 0.5 m/s), counting people walking up to it;
while the base closed on someone, 216 mm. No contact.

**T6. People walk into the folded elbow, more so with cautious settings.** Live with
the crowd and base_safety / arm_safety 1.5, the crossing walker (who never gives way)
came within 48 mm of the folded arm's elbow while the robot was not moving towards
them (235 mm at the closest while it was). The robot keeps its distance; it cannot
keep a person who walks up to it away, and the cautious robot spends longer in the
room. No contact.

## U. Localization by sensor fusion

**U1. A drifting gyro: the filter's heading lags.** With the gyro bias wandering
(plant condition `gyro_drift`, 9e-4 rad/s per sqrt(s)) the filter's worst heading error
is 0.26-0.30 deg against the scan matcher's 0.22-0.24, and its NEES is at 90-91% in the
95% band (96-98% otherwise). Its bias random walk (2e-4) is the datasheet-like value it
was tuned with; a faster one would follow the drift but trust the gyro less everywhere.
The position does not suffer (worst 6.9-9.1 mm against 10.9-13.3).

**U2. Docking errors that are not the localization's.** With a worn tyre the base
docks 19-20 mm to the side (the tolerance is 20 mm), with a drifting gyro up to 1.4
deg off its heading (tolerance 1.5 deg), whichever estimator drives: the navigation
uses the odometry's turn rate and wheel speeds directly. At the docks both estimators
sit the same 1-3 mm off the truth: the map's own offset there.

**U3. Slides nobody sees without scans.** A protective stop from full speed slides up
to 7 mm in a tick even on the dry floor; during a scan dropout neither estimator sees
it (29-46 mm off when the scans return, offline). The IMU's accelerometer would.

**U4. Shadow mode matches every scan twice** (2.5-3.5 ms each, Python). Cheap enough
here; drop the shadow when the comparison is no longer wanted.

**U5. The wet patch is tested offline only.** In the job the warning field slows the
base before most stops, so stops on the patch are rare and slow; `nav_sim` injects a
protective stop there instead. The filter's covariance is published but not yet used
by the navigation (slowing down while localization is unsure is the next step).

## V. Docking facing the pick table

**V1. A walker turns where the pick docking line starts, and the two can wait for each
other for ever.** The job's crowd's middle loop turns at (2.2, 2.0), where the robot now
backs out of the pick dock and joins its line. Offline one drive in 20 took 86 s going
round them. Live with the two-person crowd the job stalled after four drives: the walker
stood about 1 m from the arm while the robot backed out of the pick dock; the arm's
supervisor held the arm short of its carry pose, which pauses the base, and the walker
waited for the robot (a mutual wait, as fixed for the old layout in step 7). The run was
stopped by hand. Fixed with a considerate job crowd (implementation_notes, Considerate
people): live 8 boxes in 12.9 min.

**V2. Docks must be taught per map.** `nav_node` and `nav_sim` load
`data/maps/<map>_docks.yaml` (git-ignored, like the maps) and refuse to run without it;
after mapping, `python scripts/dev/nav_sim.py --teach --drives 3 --crowd 0 --goals pick
place home --arm carry --box`. The taught pick dock carries the teaching's own
offset (+4 mm lateral, -0.2 deg against the technician's spot); docking repeats around it.

**V3. Pile height at the facing dock.** The first pile scan there is planned for the
tallest pile the zone takes; above 0.30 m its pose is out of reach (0.35 m in the cell).
The pile is 0.21 m; a taller one would need the dock a little farther off the table.

**V4. wrist_cam_check reports robot cells in the cell's pile scan** (1882-2185 cells over
the chassis under the zone's edge, since the mobile base was added), so its overall
verdict fails for the cell; at the facing dock 4-24 cells.

## W. Tighter packing and docking without turning

**W1. The first box in the far corner can stand off its wall.** The set-down heading is
fixed per footprint (unturned: the aligned heading), and at it the wrist's 88 mm side
faces the west wall: cbox_3 is set down 46 mm off and its push jams at 40 N with the box
30-40 mm short (5 of 6 runs it went first; it predates this work, K3). cbox_5 first ends
1.1 mm off the west wall but 8.5 mm off the south (a 5 mm offset, under `PUSH_TRY_M`).
`scripts/dev/push_sim.py` (the scene's contacts, a compliant tool) does not reproduce the
jam. The task node's wrist model puts cbox_3 at +2/+46/+30/+12 mm off flush for the
headings 0/90/180/270 (footprint 92 x 123). Noted with the user for later
(handover_notes/tight_packing_plan.md, last section): price pushes by length and take the
footprint and joint-7-feasible heading with the least offset (12 mm here); then choose
the grasp heading at the pick from the planned spot (2 mm: no push).

**W2. In a crowd the base still steers round people standing by the tables.** After the
docking change the extra turning is ~90 deg a drive (quiet: 13-18), most of it while
someone stands within 1.5 m (the job's people stand 6 s at points by the tables).
Slowing or stopping for walkers predicted across the route, and planning round standing
people from the first plan with early replans, were tried offline and gave nothing
(implementation_notes, "Docking without turning on the spot").

**W3. The short first drive docks up to 12 mm off** (offline, from the start pose): with
the 3 cm / 3 deg pre-dock tolerance the 0.9 m approach does not take it all up. Inside
the 20 mm dock tolerance; every other drive within 7 mm.

**W4. The in-hand offset is now the landing error.** With the arm's sideways error removed
(0.2 mm sd at set-down) the pile scan's in-hand offset (~1 mm sd, up to 3 mm when an edge
is misread) sets where the box lands; the held slide then closes the gap anyway. In-hand
sensing (a closer second look, colour edges) was put off by the user.

**W5. The held box brushed the robot-side wall once** (77 N, 18 steps, a 150 x 120 box
placed beside it at a 4 mm gap; once in 40 runs before). Not fixed on one sighting.

**W6. The sim's side-gap report misreads boxes that nearly touch** (it needs one axis
apart, the others overlapping: a gap of 0.4 mm read as 113 mm to the far wall). The
validation uses the true poses instead (`scripts/dev/place_budget.py`).

**W7. A run stopped part-way can leave the next launch deaf to `/mpc/go`.** After killing a
job mid-run the next `job_run.sh` published the start command for minutes unheard; a
second `stop_all.sh` and `ros2 daemon stop` before launching cleared it.
