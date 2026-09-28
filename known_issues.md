# Known issues and things to look at later

Things noticed while building the obstacle-sensing work (2026-09-26) that are
weird, fragile or unfinished, and that are **not fixed** (unless marked
*Fixed*). Each entry says what was seen, the evidence, and a suggested next
step. Add to this file rather than letting things live in chat history.

## A. Pre-existing behaviour (seen with obstacle sensing switched off too)

**A1. First-move flake.** After `/mpc/go`, roughly a third to a half of my
launches ended with `acados solve failed (status=4)` spam or joint speed
oscillating at ~6.5 rad/s that never settles. A restart clears it.
- Evidence: seen on ~15 of ~35 starts. An earlier A/B on vanilla code (see
  `obstacle_sensing_handover_note.md`) says it is not caused by the obstacle
  work; I did not measure whether sensing changes the rate.
- Next: find why the first solve from home fails (initial guess / warm start /
  home pose against the orientation cost). A start-up retry is a workaround,
  not a fix.
- *Resolved 2026-09-27*: dead time from two free-running 20 ms timers (plant
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
- Decision (2026-09-26): documented, not chased.
- *Resolved 2026-09-27*: it was the orientation cost switching back on after a
  travel leg with it off. With the orientation held on every leg (level carry)
  the swing is gone: peak joint speed 7.6 -> 4.4 rad/s.

**A3. Blocked or full tray was retried for ever.** *Fixed*: `task_node` now gives
up after 30 consecutive no-spot scans (`DEST_NO_SPOT_MAX_RETRIES`), logs an
error and stops with the box held over the tray. Still open: it does not
recover if the tray is cleared afterwards (needs a restart).

## B. Simulation and real-time

**B1. Perception and physics share one executor thread.** The workspace camera
cycle (~12 ms at 10 Hz) and the wrist-camera dashboard render (~85 ms) run in
the same single-threaded node as the 20 ms physics timer. Seen: a 60 ms stall
made one sensed frame look 0.27 m off in the monitor; the first version of the
camera (visual meshes, ~42 ms/cycle) slowed the sim clock (fixed by rendering
collision hulls).
- Next: render on its own thread or process, or drop the dashboard for
  benchmark runs.
- *Mitigated 2026-09-27*: with the lockstep a render stall pauses the
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
- *Resolved 2026-09-27*: not a pause. The system clock (time.time(), ROS
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

**C1. A person who stops became "static" after ~3 s.** *Fixed 2026-09-27*: a
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

**C5. Single fixed camera.** It is overhead, so the arm partly hides an object it
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
runs the plain demo (no obstacle at all). *Fixed 2026-09-27*: the obstacle demo
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

## E. Geometry and scenario (2026-09-27)

**E1. With the tray ~155 deg from the pile the transit is a chord over the base.**
*Resolved 2026-09-27* by arc transit points (see design notes); kept for the
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

**E3. The person was a 0.10 m-radius sphere at z = 0.55.** *Fixed 2026-09-27*:
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

## F. After the 1.5x box scale-up and the half-scale person (2026-09-27)

**F1. The carried box is not in the controller's model.** It sags the arm
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

## G. Control smoothness work (2026-09-27)

**G1. Payload still not in the controller's model (F1 unchanged).** The 8 mm
settle tolerance stays. Payload compensation is still the proper fix.
*Fixed 2026-09-28*: payload mass in the OCP from the wrist load cell, plus
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

## H. Placement and packing (2026-09-28)

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

**H4. Placement is two rows ("shelf") with a short look-ahead.** Chosen
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
*Fixed 2026-09-28*: `heightmap.pick_order` (used by task_node and the MoveIt
decision_node) puts a top that touches a higher detected top after all others,
since the camera cannot tell "rests on it" from "stands beside it". Pile order
is now 3, 5, 1, 6, 0, 4, 7, 2 (cbox_6 before cbox_0); verified offline on the
scene geometry and in one live obstacle run.

**H8. Stall at a place point after a hold during the final descent.** Seen
once (2026-09-28): a HOLD braked the reference 4 mm above the place point, the
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

## I. Contacts, place-then-push, compact packing, sensed boxes (2026-09-28)

What changed is in `realism_plan.md` (steps 1-3) and the implementation notes;
what is still open or worth knowing:

**I1. Tray walls vs wrist and forearm.** With contacts on, the wrist (link 7,
up to 8.8 cm past the TCP) lands on the 22 cm walls when a box is set flush
against one, and at pushing height the forearm (links 5-6, ~14 cm above the tip,
reaching toward the base) lands on the walls on the robot's side. Handled by
setting the box down clear of the wall and pushing it back with the tool tilted
30 deg in the plane of the pushed face (IK search: 27 mm clearance at a
back-row push), the lean chosen from the scan, and by the packer only choosing
spots the robot can finish. Wrist extents are a fixed table
(`scene.WRIST_EXTENT_TOOL`); the forearm is a rule (flange >= 9 cm from a wall
on the robot's side), not a model.

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
the wrist, or learning).

**I9. Pushes are open-loop.** A push aims at where the box was planned to be
set down, not at where it settled after release (settling moves it ~1-2 mm, so
it works). A quick look before the push would close the loop at a few seconds
per push. Push points are never predefined: side, distance, start point,
height, lean and centring are all computed per box from the scan, the
measured box and the robot's geometry.
