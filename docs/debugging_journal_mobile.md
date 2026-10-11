# Debugging journal: the mobile manipulator

The non-obvious problems hit while putting the arm on a mobile base (the plant's base,
SLAM, people tracking, navigation, the mobile job, speed, sensor fusion, docking,
tighter packing), and how each was found and fixed, plus the pitfalls of measuring the
fixed-arm MPC baseline (section 11) and of the realistic cell just before (section 12). The arm's own journal is in
[`design_notes.md`](design_notes.md) (Debugging journal). Details and numbers:
[`implementation_notes.md`](implementation_notes.md) and
[`../known_issues.md`](../known_issues.md).

Each story: **symptom** (what was seen), **cause** (what was really going on), **fix**,
and **takeaway**.

## The ones worth telling first

1. **The navigation and the crowd model waiting for each other** (deadlocks, 5.3):
   three different mutual waits between the robot and one simulated person, each
   fixed by looking at both sides, not by tuning the robot.
2. **A background plan that hijacked the docking** (4.10): a race only the live,
   threaded system could show; the offline simulator planned in line and never saw it.
3. **A skid the Kalman filter refused to believe** (6.2): the filter rejected the very
   scans that showed the robot sliding, because it trusted locked wheels saying "zero".
4. **The parked base that crept** (1.3): MuJoCo's soft contacts have no static friction;
   a parked robot turned 3 deg under a push and every box placed after it was 9 mm off.
5. **The protective field that held the robot for ever beside a table leg** (4.4):
   a safety field drawn too generously is a liveness bug.
6. **SLAM falling back to odometry on 96% of scans** (2.1): denser data made the scan
   matcher worse, because walls stopped looking like lines.
7. **A correction that never ran where it mattered** (8.1): the arm's offset-free bias
   reset whenever its goal moved, so on every straight-down set-down a repeatable 1.6 mm
   sideways error was never removed; the fix then wound up and hung a run (8.2).
8. **Every friction reads as a touch** (8.6): a force-guarded slide stopped early three
   ways (a lagging tare, the box dragging on the floor, rubbing the face it had just
   touched) before it stopped only on contact.

---

## 1. The base in the simulator (plant)

**1.1 The chassis outran its own wheels.**
- Symptom: over a 0.5 m/s^2 start the chassis got 23 mm ahead of the distance its
  wheels had rolled, and gave it back when braking.
- Cause: the wheels were MuJoCo `velocity` actuators. Under the `implicitfast`
  integrator their damping is integrated implicitly, but the contact solver does not
  see it, so wheel and floor disagreed. (Euler and RK4 rolled exactly; changing the
  integrator was not an option because the arm depends on it.)
- Fix: torque motors with an explicit speed loop run before every physics substep, as
  a real motor controller's inner loop. Chassis and wheel arc then agree to 0.1 mm.
- Takeaway: a simulator's built-in actuator can quietly break the physics you are
  testing; check rolling against kinematics before trusting odometry results.

**1.2 Turns in place reached 26 deg of a commanded 69.**
- Symptom: the base barely turned on the spot.
- Cause: a rigid chassis resting on six points (two drive wheels, four casters): in a
  turn the drive wheels lost contact with the floor.
- Fix: sprung drive wheels (40 kN/m, about 400 N preload), as real AMRs have. That
  raised a second bug: the wheels, now one body away from the chassis, were no longer
  covered by MuJoCo's parent-child contact exclusion and brushed the chassis
  intermittently, which looked like a flickering floor contact and gave 12-17 deg of
  yaw on some starts. Fixed with an explicit contact exclusion.
- Takeaway: one fix (springs) changed the body tree and broke an implicit assumption
  (contact exclusion); re-check what the change touched.

**1.3 The parked base crept, and every box after it was off.**
- Symptom: in a live run a 4.5 s push of a box into the tray turned the parked base 3
  deg and moved it 13 mm; every box placed afterwards appeared shifted by ~9 mm. In
  other runs the parked base turned 0.09-0.33 deg in small steps at each swing of the
  arm, and the lidars then saw a table leg move against their learnt background (a
  false person).
- Cause: MuJoCo's contact friction is soft: there is no true static friction, so a
  wheel under a steady sideways load creeps. The arm's swings ratcheted it.
- Fix: three things together. Near-hard wheel-floor contact pairs; a braked wheel's
  angle held stiffly (a motor brake through a gearbox); and each braked wheel's contact
  point pinned to the floor until the force exceeds friction times load (a model of
  stiction that lets go and re-sticks). Pins alone did not help (the wheels rolled about
  the pins), a stiffer brake alone only halved it. MuJoCo's global `noslip` option fixed
  the base too but made the arm's joint friction perfectly sticky and left the arm 10.6
  mm short of a scan pose, so it was rejected.
- Takeaway: a millimetre-level plant artefact surfaced as a perception problem (false
  person) and a placement problem (boxes off); trace effects back to physics.

**1.4 Wheel-only odometry was 4-8% wrong per turn.**
- Symptom: heading from the wheels drifted badly in turns.
- Cause: a flat cylinder tread has two contact points across its width and scrubs in
  turns.
- Fix: a rounded (ellipsoid) tread: one contact point, no scrub.

## 2. SLAM and localization

**2.1 The scan matcher fell back to odometry on 1779 of 1855 scans.**
- Symptom: mapping barely used the lidar.
- Cause: the scan matcher (point-to-line ICP) needs each target point's neighbours to
  look like a line. Twelve keyframes with 2 cm of range noise made each wall a dense
  band; six neighbours in a band are not a line, so almost no points qualified.
- Fix: thin scans and local maps to one point per 5 cm cell before matching; for saved
  maps (walls about two cells thick) use twelve neighbours.
- Takeaway: more data made the algorithm worse, because it violated an assumption
  (thin surfaces).

**2.2 Localization against a distance field scattered by 2 cm.**
- Cause: matching against a distance field of occupied cells can only place a wall at
  a cell face.
- Fix: keep the mean of the hits in each occupied cell and match against those points.

**2.3 1 deg yaw errors during turns in place.**
- Cause: a scan sometimes arrived before the odometry of its own tick, and the node used
  the previous tick's odometry.
- Fix: a scan waits for its tick's odometry (interpolated to its capture time). Scans
  are also stamped with the time of the plant state they were ray-cast from, not the
  sensor process's clock, which had made them up to a tick older than their stamp.
- Takeaway: at 1 rad/s, 20 ms of timestamp error is over a degree.

**2.4 Our SLAM against slam_toolbox.** Not a bug, a comparison worth telling: on the
same data ours localized at 2-3 mm median against slam_toolbox's 11-17 mm, partly
because the robot's simulated lidars are good and our matcher uses both scanners'
raw points instead of one merged scan.

## 3. People on a moving base

**3.1 On a moving base everything was a person.**
- Cause: the parked cell's detector learnt a background range per lidar beam; that only
  works while the scanner stands still.
- Fix: foreground against the map instead: each beam's end point placed in the map
  with the localized pose, and foreground if more than 15 cm from every mapped surface.

**3.2 Ghost people walking on round corners.**
- Symptom: 12-13 false people per drive.
- Cause: when a person turned a corner and left the lidar's view, their Kalman track
  coasted on in the old direction for its 1 s coast time.
- Fix: drop a track at once if the lidar sees through its predicted position (a beam
  reaching 30 cm past it). None after.

## 4. Navigation and the safety layer

**4.1 The base weaved through the crowd (22 swerves in 20 drives).**
- Cause: the base MPC weighed position error the same in every direction, so slowing
  down for a person cost as much as stepping sideways, and it chose sideways.
- Fix: split the error into lag (along the route, cheap) and lateral (across, dear),
  plus a speed limit near people. 9 swerves, then 3-8.

**4.2 Routes stopped 2 cm past their turning points.**
- Cause: with cheap lag the MPC overshot where the route stopped to turn in place; the
  90 deg turn then made that a lateral error, and 4 drives in 7 had to realign at the
  dock.
- Fix: where the reference stops or turns in place, lag weighs as much as lateral.

**4.3 The MPC fled people, toward others.**
- Symptom: allowed to reverse freely, the base backed away from someone walking at it,
  toward whoever was behind; with the full person radius always enforced it turned 50
  deg away on the spot from someone, then back.
- Fix: reverse only where the route reverses; cap the person constraint at the distance
  the robot would keep standing still (never close in, never flee).

**4.4 A safety field that held the robot for ever beside a table leg.**
- Symptom: setting off from standstill beside the pick table, the base stopped and never
  moved again.
- Cause: the first field for a turn in place was the whole circle the corners sweep,
  however slow the turn. The first command (0.05 m/s, a turn by the rules then) put a
  table leg 15 cm off the corner inside that circle. A box-shaped field at standstill
  also made the robot and a person at its front corner wait for each other.
- Fix: the protective field is the chassis swept along its actual arc for as long as
  the stop takes, and it only stops for points the robot is getting closer to.
- Takeaway: a too-generous safety field is not "safer", it is a liveness bug.

**4.5 Corner scanners blind at their own corner.**
- Symptom: a leg right at a corner scanner was in neither scanner's view; once the base
  crept within 50 mm of a person standing there.
- Cause: the lidar model dropped returns under 5 cm. Removing that showed the scanners'
  housings, 3 cm outside the chassis, scraping the pick table's legs.
- Fix: scanners' optical centres moved onto the chassis corners (each 270 deg sector
  runs along both faces), the dock 3 cm farther from the table. Recessing them inside
  the outline was tried and rejected: the chassis hid the sides from both.

**4.6 The planner made the base weave on open floor.**
- Cause: Hybrid A* found routes that changed turn direction four times between the
  tables; all "optimal" by distance.
- Fix: a cost for each change of steering.

**4.7 Replanning froze the control loop for up to 0.7 s.**
- Fix: the planner moved to a worker thread; the base drives on its current route while
  a new one is made. (This later caused 4.10.)

**4.8 People's upper bodies came within 72 mm of the folded arm.**
- Cause: the scanners see legs 20 cm above the floor; a person's torso and arms overhang
  their legs by up to 13.5 cm, and the folded arm reaches near the chassis edge.
- Fix: scan points that are people get 10 cm more margin; walls and tables keep theirs
  (more for everything would stop the base beside the pick table's legs, 4.4 again).

**4.9 Stuck next to someone standing (found when the base went to 1 m/s).**
- Symptom: the base crept up to a person standing near its route home, stopped with
  their legs 0.2 m off its corner, and planned the same route every 4 s for good.
- Cause: the new route round them started with a turn in place towards them, which the
  safety layer (correctly) refused that close.
- Fix: blocked again on a route round someone standing within 1 m, back off 0.4 m and
  plan from there, keeping the pose it backed off from free in the plan.
  The first version backed off whenever blocked; with someone standing on the dock's
  pre-dock pose it shuttled back and forth, and its shuffling kept resetting the
  simulated person's "blocked by the robot" timer, so they never left. Backing off
  only after a route round someone fixed both.
- Takeaway: a recovery behaviour needs a precise trigger, or it creates the next
  deadlock.

**4.10 A late background plan hijacked the docking.**
- Symptom: live, in a crowd run, the base sat in front of the pick table for 9 minutes,
  commanding 5 cm/s while the safety layer held it.
- Cause: a route round someone, asked for on the way, came back from the planner thread
  after the base had already aligned and started its final approach. It replaced the
  approach with a route starting 0.65 m behind the base. As a route (not docking), the
  table ahead was an obstacle, and the safety layer stopped it for good.
- Fix: a finished plan is only taken while the navigator is still in a route,
  planning or holding state. A unit test reproduces the race; it fails without the fix.
- Takeaway: the offline simulator plans in line, so it could never show this; threading
  bugs need live runs (or a test that forces the interleaving).

**4.11 The robot turned on the spot to line up with every dock in a crowd.**
- Cause: routes bent round people ended 2-5 cm and 1-2 deg off the docking line, just
  over the 2 cm / 1 deg tolerance, so the robot did a turn-straight-turn alignment and
  often missed 1 deg again.
- Fix: a 3 cm / 3 deg tolerance (the straight approach absorbs it) and, for larger
  misses, one forward curve onto the line. Alignment turning per job went from about
  1000 deg to 37.

## 5. The mobile job: arm and base together

**5.1 The job stopped with a box held after 30 identical rescans.**
- Cause: after a blocked place the retry rescanned from the usual pose, where the held
  box hid 29-80% of the spot it needed to see; the spot stayed unknown, so it rescanned
  again from the same place.
- Fix: rescan with the camera just past the blocked spot and the box beyond it (0%
  hidden offline).

**5.2 The base paused and went on several times a second.**
- Cause: the arm's people supervisor flickers at its threshold; each hold paused the
  base.
- Fix: keep the pause 1 s past the last hold.

**5.3 Three different deadlocks between the robot and one simulated person.**
Each a mutual wait, each fixed on both sides:
- A person's next spot was under the robot: they waited for it for ever. Fix (plant):
  skip a spot within 0.6 m of the robot (0.8 m kept one of them on the docking line).
- A person cornered between a wall, a table and the robot: both perpendicular sidesteps
  pointed at the robot. Fix (plant): sidestep in whichever of 16 directions is free.
- The base paused for an arm held over its own deck, and waited for "stowed", which
  meant all the way down: a person standing 1.3 m off held the arm there, the base waited
  for the arm, the person waited for the base. Fix (method): pause only for an arm held
  beyond the chassis, and an arm over the carry pose counts as stowed at any height.
- Takeaway: a "test person" written to be stubborn finds liveness bugs, but some of the
  deadlocks were the test person's fault; fix the side that is unrealistic.

**5.4 The arm's wrist hit its joint limit when moving while docking.**
- Cause: turning the tool heading the "shorter" way drove joint 7 past its limit.
- Fix: turn the way that keeps joint 7 in range (joint 7 ~ joint 1 - heading - 135 deg).

**5.5 A crash from a numpy boolean.** `Bool(data=np.bool_)` raised in a ROS message
constructor and killed the task node; wrapped in `bool()`.

**5.6 Halving the room.** The parked cell's layout did not fit along the shorter wall
(backing out and the visitor need 5.2 m in a line): the cell was turned 90 deg. Six
people in half the floor made a crush (one walked into a nearly stopped base, 18 mm
off): four in the crowd.

**5.7 A person stuck near the arm, the robot waiting for the person (facing dock).**
- Cause: the new docking line pointed into the room, where a walker's loop turned; the
  walker stopped 1 m from the arm as the robot backed out, the supervisor held the arm,
  which pauses the base, and the walker waited for the robot.
- Fix (plant): a considerate crowd: people step aside after 1 s and do not stand within
  2.5 m of a parked robot. A first version kept people 1.5 m away at all times; the robot
  drove as in an empty room, but people visibly fled it, so it was dropped.

**5.8 The considerate crowd froze twice before it worked.**
- A walker stood still 1.5 m from the arm for minutes (live). Cause: when the robot sat
  on their turning point, the old model's "go to a spot beside the robot" (0.9 m away)
  pulled them in while the new 1.5 m courtesy pushed them out: they balanced at 1.5 m,
  close enough to hold the arm. Fix: a considerate person whose spot the robot is at
  waits where they are, 2.5 m off or more.
- A bystander never moved. Cause: both of their spots were near the robot; "spot taken,
  take the next" flipped the target every tick and each flip skipped that tick's step.
  Fix: take the next free spot in the same tick; if none is free, wait 2.5 m off.
- Takeaway: reproduce a scenario person offline with the robot parked where it stalled;
  both freezes showed in seconds.

## 6. Sensor fusion (EKF localization)

**6.1 Baseline first: what was there to fix?** Today's localizer replaced its pose with
each good scan match. Measured under new plant conditions (a wet patch, a worn tyre, a
drifting gyro, scan dropouts): every good match made the pose jump 11-16 mm; a stop
during a dropout slid unseen, 49 mm off. A worn tyre and a drifting gyro did not hurt
the pose at all (each scan corrects it) but put the docks 19-20 mm sideways and 1.4 deg
off: those came through the navigator's own use of the odometry's speeds, which no
pose estimator changes. (A "crowding" condition, people round the robot, was tried and
dropped: thin people hide little wall.)

**6.2 The filter rejected the scans that showed the robot sliding.**
- Symptom: on the wet patch the filter ended 81 mm off, 12 scans rejected in a row,
  against the plain scan matcher's 23 mm.
- Cause: the first version widened its uncertainty for the one tick the wheels locked.
  The next scan shrank it again; then the locked wheels kept saying "zero speed" with
  their usual small noise while the chassis slid on, and the chi-square gate refused
  the scans that showed the slide.
- Fix: a skid is an episode, as long as the slide could last; through it the wheels
  count with 0.5 m/s of noise. (Leaving the wheels out entirely fixed the slide but then
  a stop during a scan dropout coasted on at the old speed: 282 mm.) Result: 19 and 29
  mm, better than the scan matcher's 23 and 32.
- Takeaway: an overconfident filter rejects the evidence that it is wrong.

**6.3 Eight skids in the first seconds of a quiet live drive.**
- Cause: the filter took each tick's length from wall-clock stamps; under load (WSL) a
  20 ms step looked like 5 or 40 ms, so the wheel speed jumped and looked like a skid.
- Fix: the tick length from the plant's step count, as the odometry already did.

**6.4 The filter's error was larger than its own covariance said (NEES 8.4).**
- Cause: part of the error moved with the place in the room: the map's own error
  against the room (2.6-6.4 mm), which no estimator in the map frame can see.
- Fix: add the map's error (2.5 mm, 0.04 deg) to the reported covariance: 95-96% in the
  band. Nothing else was tuned.

**6.5 Pose jumps.** Result worth quoting: the filter's largest jump per tick 3.8-5.1 mm
against the scan matcher's 11.9-15.4 mm, and a smaller worst error in every run on the
same data. It became the default; the scan matcher still runs in shadow.

**6.6 Shadow mode.** Both estimators run on identical data every tick while one drives,
so the comparison is fair; without it, closed-loop runs diverge and differences could
be timing.

**6.7 Scan dropouts froze the lidar thread.**
- Symptom: in the first live smoke test the localizer processed no scans and the robot
  never moved.
- Cause: the dropout schedule started at time 0, but scan stamps are wall-clock epoch
  seconds; the first call looped through 10^8 gaps to catch up, inside the lidar thread.
- Fix: start the schedule at the first scan's stamp. Takeaway: a short live smoke test
  before any batch catches what offline time (which starts at 0) hides.

**6.8 Tests that did not test.**
- The odometry check's square and circle showed metres of error with no condition at
  all: since the room was halved, their paths ran into the parked cell's tables. Moved
  to the open floor; the check had been silently broken since the halving.
- The wet patch changed nothing in the job's drives: the warning field slows the base
  before any stop, so it never braked hard on the patch. A protective stop is injected
  there offline instead, and the check reports wheel slip per tick to prove it acted.
- A live dock 7.8 mm off looked like the filter's fault; both estimators sat the same
  distance off the truth there (the map's own offset). Compare estimators at the same spot
  before blaming one.

## 7. Speed and tooling

**7.1 The step-in person stepped in only once** (test bug). After the room was halved,
the place dock was 3.0 m from their step-in spot and they re-armed only once the robot
was 4.0 m away, so after the first time they just stood beside the route: the
earlier results for that scenario were of someone standing.

**7.2 Parallel offline runs crashed loading a half-built solver.** Four simulations
started at once; one rebuilt the base MPC's solver (its cache fingerprint had changed)
while another loaded the half-written library. A test-harness race; rerun once built.

**7.3 Making speed limits settable without rebuilding the solver.** The base MPC's
speed and acceleration limits were moved out of the compiled solver's fingerprint and
set at runtime, so trying a speed costs no 1-2 min rebuild.

**7.4 Flat warning boxes held the base at 0.3 m/s for whole corridors** at 1 m/s and
turns at stations at 0.3 rad/s; graded caps (what still stops short of the nearest
point) fixed it. At 1.0 m/s a third of the route time is still turning in place, which
is why 1.5 m/s was not faster.

## 8. Placing and packing (2026-10-04)

**8.1 The arm's sideways error at set-down was never corrected.**
- Symptom: boxes landed 1.4-1.9 mm (sd) off their spot; the same box at the same spot
  had the same error in two runs (+1.5/+2.2 mm and +1.5/+2.3).
- Cause: found by splitting each landing exactly into the arm's tracking, the in-hand
  offset and the release (validation-only truth lines, `scripts/dev/place_budget.py`):
  tracking dominated. The controller's offset-free bias integrated only while the whole
  reference stood still and reset when the goal moved by over 1 mm, so on a straight-down
  set-down (the goal moving in z) the sideways error was never removed; at touch-down
  friction locked it in. The controller's own kinematics saw the error (within 0.1 mm of
  the truth): it was the controller's to remove.
- Fix: the sideways bias per axis, kept while the reference's x/y stands still (also
  during vertical moves), gain 4/s (1.5/s lagged the pose-dependent error by ~0.7 mm).
  Tracking at set-down 0.2 mm sd.
- Takeaway: a correction that resets on "the goal moved" never works along a ramp;
  measure where the error is before tightening anything.

**8.2 The fix wound up and the job hung.**
- Symptom: a box landed on a neighbour's edge, the arm lifted, then stood 14 mm off its
  goal for minutes; the task waited for it to settle for ever.
- Cause: during the blocked descent the sideways bias kept integrating against the
  contact, to 15.7 mm; once free, the error (14 mm) was beyond the 10 mm window in which
  it integrates, so it could not unwind.
- Fix: integrate sideways only while the arm follows the reference in height (a blocked
  descent lags it), and always when the error unwinds the bias.
- Takeaway: every integrator needs to know when it is blocked, and a way back.

**8.3 The bias froze during pushes.**
- Symptom: in a push along x the tool drifted 5-8 mm in y; the y bias stayed constant.
- Cause: "standing still" was tested as an exact zero spread of the reference over the
  horizon; the task's cylindrical interpolation makes a straight push wobble ~1 mm
  sideways, so the test never passed.
- Fix: still means creeping slower than 10 mm/s.

**8.4 A pushed box's record was 10-20 mm off.**
- Symptom: the box placed after the first one landed on its edge or 10 mm away.
- Cause: a push that stopped short (a jam) still recorded the box at the planned end;
  the tray re-measure, limited to a 6 mm window and 15 mm corrections, caught up ~9 mm a
  scan.
- Fix: record the box from where the tool stopped (its side on the face), and re-measure
  that box once with a wider window.

**8.5 The far-corner push jams at 40 N (open).**
- Symptom: the first box, set down 46 mm off the west wall, is pushed 17-24 mm and stops
  at 40 N, 30-40 mm short (5 of 6 runs it came first).
- Cause, part: in contact the tool slid sideways along the face (the controller's model
  has no contact) and the friction dragged the box into the next wall and pressed it
  down; fixed with 8.3. The rest is not reproduced by an offline push with the scene's
  contacts and a compliant tool (`scripts/dev/push_sim.py`).
- The real cause of the long push: which way the tool faces. The wrist reaches 88 mm
  past the tool on one side; at the usual heading that side faces the wall (46 mm off),
  at another heading 2 mm. Noted for later (known_issues W1).
- Takeaway: before tuning the push, ask why a push is needed at all.

**8.6 The held slide stopped on things that were not neighbours.**
- Symptom: the slide (lift, move sideways until the wrist feels 1 N, release) reported
  touches after 0-2.5 mm while the true gap was 2.8-6 mm.
- Causes, one after the other: (a) the tare came from a slow filter still settling
  after the lift; (b) a 2 mm lift left the box on the floor (the arm sags 2-3 mm under
  it; the wrist's vertical load read 10-12 N instead of 17), and floor friction read as
  a touch; (c) after a real touch a 0.5 mm back-off left the box rubbing that face, and
  the friction on the next slide read as a touch ahead.
- Fix: tare from the 3-tick average at the slide's start and 3 ticks to confirm; lift 5
  mm; back off 1 mm.
- Takeaway: a force guard reads any friction against the motion as contact; check the
  vertical load too.

**8.7 Slides ran to their limit without touching.** The pile scan reads boxes ~1 mm
large, so true gaps are wider than planned; a slide of the planned gap + 3 mm reached a
wall just as it ended. Allowance raised to + 6 mm (the force still stops it).

**8.8 A short box caught a taller neighbour's top edge.**
- Cause: the hover point is set for the tallest box the spec allows, measured at the
  held box's top; a short box hovering there has its bottom below a tall neighbour's
  top, and the fast pass-through descent lagged sideways into its edge.
- Fix: beside a taller placed box, settle above the spot and go straight down (as near
  walls already).

**8.9 Packing as one block left an L.** Ranking spots by the block's bounding rectangle
grew it square and left an L-shaped free space that the big boxes, arriving last and
unseen, did not fit (all 8 on the floor in 82-86% of orders, worse than before);
measuring the block by how far it reaches along the tray (fill the width first) gave
97%.

**8.10 Validation pitfalls.** The simulator's side-gap report needs one axis apart and
the others overlapping, so two boxes 0.4 mm apart read as 113 mm to a far wall; and
sorting the true sizes longest-first lost the boxes' axes in a gap script. Gaps are now
computed from the true poses.

**8.11 The packing looked worse on the mobile robot; it was not.** A live camera picture
of the mobile tray was first compared with a drawing of the cell's planned spots, not
like with like. Drawing both from their logs showed the same fill and the same 5 mm
gaps: the packing rule, not the mobile job. The comparison
tool also read the boxes from a model file that no longer held them and planned with a
3 mm gap where the task uses 5 mm, so its "98% of orders fit" was optimistic.

## 9. Docking without turning on the spot (2026-10-04)

**9.1 The base turned on the spot for misses of a few centimetres.**
- Symptom: in a crowd 40-360 deg of extra turning per drive; ~1/3 of it in the
  pre-dock align (turn, straight, turn), never in a quiet room.
- Cause: routes bent round people ended 2-5 cm and 1-2 deg off the docking line, just
  over the 2 cm / 1 deg tolerance; the align then often missed 1 deg again and aligned
  twice.
- Fix: 3 cm / 3 deg (the 0.9 m approach takes up the rest) and a forward curve onto the
  line for larger misses. Live: on-the-spot turning 783-1049 deg a job to 37.

**9.2 The curve was rejected for small misses.** A fixed curve length bent too tightly
over a 4.5 cm offset, so it fell back to turning on the spot; the length now grows with
the square root of the offset.

**9.3 Two fixes for the weaving that changed nothing.** Slowing or stopping for walkers
predicted across the route, and planning round standing people early, left the extra
turning where it was (or moved it into longer routes). Labelling the route's turning by
the nearest person showed why: most of it is beside people standing by the tables, and
neither fix was aimed at that. Takeaway: label the metric by cause before writing the
fix. (Open: known_issues W2.)

**9.4 A tolerance meant for docks loosened parking at home.** The 3 deg pre-dock
tolerance also applied at home, where no approach follows: parked 2.0 deg off live.
Home keeps 2 cm / 1 deg. Found only by sweeping the docs for what the change touched.

**9.5 The facing pick dock's first pile scan was out of reach.** The first scan is
planned for the tallest pile the zone allows (0.35 m); from the facing dock that camera
pose is 12 mm beyond the arm's reach, and the task stops on an
unreachable pose. Found offline with the wrist-camera check run at the new dock; the
zone there takes piles up to 0.30 m (the pile is 0.21 m).

**9.6 Rolling on into the approach braked for a tick.** On arriving lined up the
navigator switched to the approach but returned zero speed for that control step, a
brief brake mid-roll. It now solves the approach's first command in the same step.

## 10. Tooling (2026-10-04)

**10.1 Commands that killed themselves.** `pkill -f "<pattern>"` and a wait "until no
process matches <pattern>" both matched the shell running them (the pattern is in its
own command line): one killed itself before its edits ran, the other waited for ever.
Match with `grep "[n]av_sim"`-style patterns.

**10.4 A monitor that reported nothing.** A live-run monitor piped the log through
`grep --line-buffered | sed -u | cut`; `cut` buffers when writing to a pipe, so no event
arrived until the run had long progressed. Every stage must flush per line (`awk` with
`fflush()`).

**10.5 Parallel offline arm runs truncated a shared solver.** Several docking simulations
started at once rebuilt the same generated solvers in the repository; one loaded a
library another was still writing ("file too short"). Run solver-building tools one at a
time (or build first), as 7.2 for the base's solver.

**10.2 A launch deaf to its start command.** After stopping a job part-way, the next
`job_run.sh` published `/mpc/go` for minutes unheard; `stop_all.sh` and `ros2 daemon
stop` before launching cleared it.

**10.3 The job that "started on its own".** The ENTER prompt was there, but the launch
log scrolled ~90 lines over it while waiting, the people walked from launch, and an
ENTER typed during start-up was kept for the prompt. Now the log pauses while waiting,
early keys are dropped, and the people wait for the start too.


## 11. Measuring the fixed-arm baseline (2026-09-29)

The MPC baseline for the learning-based MPC comparison (`results/mpc/`,
`scripts/dev/results_*.py`). Pitfalls in the measuring, not the robot.

**11.1 The solver's prediction is not the model's.**
- Symptom: the one-step model error looked large (0.17 rad/s joint velocity).
- Cause: with SQP_RTI the logged next state x1 is one linearised QP iteration, so it
  mixes solver error with model error.
- Fix: roll the controller's own ODE forward offline, one tick from the measured state
  with the torque the plant applied. Same 0.17 rad/s, so it was model error, but now
  known rather than assumed.
- Takeaway: measure model error with the model, not with the solver's output.

**11.2 Alignment found by trying every offset.**
- Symptom: unclear which logged plant step a prediction should be compared to.
- Fix: compared at offsets -1 to +2 steps; only +1 gave a small error (1.8 mrad
  against 20+).
- Takeaway: check alignment against the data before building metrics on it.

**11.3 The oracle solved an easier task.**
- Symptom: the perfect-model oracle (no plant/model mismatch) used less effort per box.
- Cause: switching the mismatch off also skipped the box-mass draws, so its boxes
  weighed 0.2 kg instead of the seeded 0.2-2 kg.
- Fix: make the arm's random draws and discard them, so the random sequence and the box
  masses stay the same as the mismatch run of that seed.
- Takeaway: "everything else equal" needs the random draws checked too.

**11.4 Same seed, different run.**
- Symptom: two runs of seed 1 differed: 8/8 boxes and 11.6 mm tracking against 6/8 and
  17.8 mm.
- Cause: lockstep waits and render stalls change the timing, and from there the packing
  layout.
- Fix: two runs per seed, averaged per seed; the summary reports the SD within a seed
  next to the SD between seeds (for tracking they are about equal, ~3 mm).
- Takeaway: a seed fixes the inputs, not the run; measure the noise floor before
  claiming an improvement.

**11.5 The 1.4 cm near miss that was not one.**
- Symptom: minimum person-to-arm clearance of 1.4-2 cm.
- Cause: every close approach was the person walking up to an arm that was already
  held; sampling every 5th step also overstated the minimum.
- Fix: report the clearance while the arm moves separately (0.117-0.134 m), sampled
  every 2nd step.
- Takeaway: a safety minimum without the robot's state can tell the wrong story.

**11.6 A grep that blamed the solver.** "solve failed" matched a Ruckig warning from the
reference generator, not the MPC; the run had stopped on a full tray. Match the node
name too, not only a phrase.

**11.7 Failed runs dropped out of the timing.** Time per box was computed only for
finished runs, so every failure vanished from the cycle-time comparison. Now it runs to
the last placement for every run.

## 12. The realistic cell, before the base moved (2026-09-30 to 10-01)

The fixed arm in its realistic cell (real time, sensor noise, lidars, the packing
rework), measured with `motion_check.py` (sweeping), `bench_analyze.py` (stutter) and
`person_clearance.py`. Several of these were only found because a metric was written
first and the run judged by it, not by watching.

**12.1 A wrapped angle that hid a joint limit.**
- Symptom: placing a box turned 90 deg at a spot behind the base, the arm reached back
  over its shoulder (joint 2 at -101 deg, joint 7 at its -166 deg limit) and the solver
  failed for good.
- Cause: the placement checked joint 7 with the angle wrapped to +-180 deg, so heading
  0 read as joint 7 at +116 deg; the wrist really turns continuously and would have
  needed -244. After that was fixed, it came back: the carry is several legs, and on
  the last one the "nearest to turning with the base" heading tied at exactly 180 deg
  between two branches, one of them out of range.
- Fix: predict joint 7 the way the heading profile really turns it (unwrapped), and
  choose the heading's 2 pi branch that keeps joint 7 within 150 deg, nearest to
  turning with the base (the mobile version is in 5.4).
- Takeaway: a joint that turns continuously must be checked unwrapped, and a
  "nearest" rule needs a tie-breaker that knows the limits.

**12.2 The viewer that played like a low-frame-rate video.**
- Symptom: smooth motion looked jerky in the MuJoCo viewer, with and without the
  obstacle window.
- Cause: not the motion (the plant was on time). The sensors process received the
  scene state on the same single-threaded executor that rendered the cameras, so
  updates stalled for 60-350 ms and the viewer redrew a stale pose, then jumped.
- Fix: the scene state on its own node and thread; the viewer on a fixed 20 ms clock.
  A first version died after 5 s: `time.sleep(next - time.monotonic())` with the
  clock read twice could go negative.
- Takeaway: measure what the display actually received before blaming the motion.

**12.3 A 7 cm ball that refused every push.**
- Symptom: the first box (76 x 105 mm) never reached a corner flush; the packing fell
  back to "any free spot", and after a rework took a spot 5 cm off the wall.
- Cause: the whole-arm wall check modelled the flange as a 7 cm sphere; with the tool
  over a box against a wall it "touched" the wall (-32 mm) while the precise tool and
  wrist check on the same pose cleared it by 13 mm. Every push along a wall was refused.
- Fix: the flange left to the precise check (the wrist camera is not a collision part);
  the packing scores a spot where the robot really leaves the box and prefers the one
  nearest flush.
- Takeaway: a conservative proxy is a liveness bug when it is wrong by more than the
  margin it protects.

**12.4 A box caught under an overhang.**
- Symptom: a 60 mm box failed three set-downs in a row ("the wrist load never
  dropped").
- Cause: its spot was 4.5 mm (planned) from the corner of a 135 mm box stacked on a
  120 mm one; the packing only kept clear of floor boxes, and the scan check ignores one
  cell round the footprint. The held box's top met the overhang's underside.
- Fix: keep 5 mm from any placed box whose bottom is above the spot's surface. A first
  version applied it to every box and left a 135 mm box no spot at all (exact 5 mm
  gaps failed a strict `<` by rounding, and taller neighbours are the scan's job).
- Takeaway: new geometric rules need a tolerance and a scope, or they fail the cases
  the existing rules already handle.

**12.5 Depth frames lost in transit.**
- Symptom: with the ceiling camera on, the supervisor held again and again for
  "camera down"; 272 frames arrived in 120 s instead of 1200.
- Cause: the camera rendered at 10 Hz, but the 0.6 MB frames went best-effort and were
  lost in fragments.
- Fix: reliable, keep-last-1.
- Takeaway: best effort is right for small, frequent messages, not for big ones.

**12.6 The wrist that wound up and back on every carry.**
- Symptom: `motion_check` flagged joint 7 travelling 2.4 rad back and forth on every
  pile-to-tray carry, in all 10 runs; the return trip swung about 25 deg each way.
- Cause: the box heading was set per leg. The zone-exit leg held it while the base swung
  60 deg, then the next leg turned it 180 deg: joint 7 went -7, -70, +9 deg. Within a
  leg, the smoothstep shape alone made the wrist overshoot.
- Fix: a run of pass-through legs turns as one, linear in the base swing.
- Takeaway: if a joint is the difference of two motions (joint 7 ~ joint 1 - heading),
  plan the two in step, not leg by leg.

**12.7 A 3 mm push that spun the wrist.**
- Symptom: in 3 of 10 runs the arm whipped right after a set-down: joint 7 turned 78 deg
  in 0.5 s, joints at 4.8 rad/s, the tool 96 mm off its path beside the box just placed.
- Cause: a 2-3 mm push needed another heading and a 10 deg tilt, squeezed into a few cm
  of travel; nothing limited the turn rate.
- Fix: no pushes under 6 mm, and every leg slowed so the tool turns no faster than
  1.5 rad/s. That cap then made one leg crawl at 2 cm/s for 16 s (an 83 deg turn on a
  2 deg base swing, tied to the swing); a turn more than twice the swing now follows the
  distance instead.
- Takeaway: a rate cap exposes every place the motion was tied to the wrong variable.

**12.8 Creep during slow set-downs, and a jolt when a push stalled.**
- Symptom: joints 1, 3 and 5 drifted up to 13.5 deg over a 3 s set-down and snapped
  back on lift-off; a stalled push ended with a one-tick 3.5 rad/s spike.
- Cause: the self-motion that leaves the tool still was held only by the posture term
  (weight 1 against 5000 for tracking), so the plant's torque mismatch drifted it; the
  push's reference had run 31 mm ahead and was reset to the tool in one tick.
- Fix: posture weight 5 (worst drift 3.9 deg, same tracking); the back-off starts from
  the reference, so the force eases off.

**12.9 Shortcuts found by asking "could a real robot know this?"**
- Symptom: none in the runs; found in an audit after the user's genuine-pipeline rule.
- Cause: the plant's grasp welded the box nearest the requested point whether the tool
  touched it or not, and the method never checked it held anything; the wrist camera's
  pose came from the simulator's message.
- Fix: the grasp takes hold only on contact; the method hovers, lowers until the wrist
  load changes by 3 N, grips and checks the weight after the lift; the camera pose
  comes from the measured joint angles and the hand-eye calibration.
- Takeaway: shortcuts do not show as failures; audit every input of the method side.

**12.10 Measuring pitfalls.**
- A clearance "while moving" of 343 mm was measured during a hold: encoder noise pushed
  single ticks over the 0.05 rad/s threshold. Now motion is averaged over 0.2 s
  (940 mm).
- The lidar "false positives" and the one miss at the visit entry were the truth
  topic saying "gone" a frame before the scene moved the person; the monitor no longer
  scores frames straddling an appear/leave.
- Two background run loops overlapped: starting the second called `stop_all.sh` and cut
  the first's run short (7 of 8 boxes). Never start a batch while another is running.
- A scratch folder was cleaned between sessions and with it the camera-only baseline
  for a timing comparison; keep comparison runs somewhere that survives.
- A large in-file replacement deleted the lidar functions that sat inside the replaced
  block; lint (`pyflakes`: unused imports) caught it before the build.
