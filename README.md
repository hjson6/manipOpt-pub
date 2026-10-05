# manipOpt

https://github.com/user-attachments/assets/5d20ed4a-0f24-43bf-97c5-697f5f58c288

A mobile manipulator in simulation: a Franka Panda arm on a differential-drive mobile
base that does a whole pick-and-pack job on its own, in a 7.0 x 5.6 m room with people
walking about. It maps the room with its own lidar SLAM, localizes in that map, tracks
the people, drives between the pile's table and the tray's with an acados MPC and a
separate safety layer, docks at each table to a few millimetres, and moves eight
mixed-size boxes from a pile into a tray, packed about a millimetre apart. The arm is
driven by torques from a second acados MPC; it folds while the base undocks and unfolds
while it docks.

Everything runs in real time on the robot's own sensors (noisy encoders, a wrist load
cell, a wrist depth camera, two safety lidars, wheel encoders, a gyro), against a plant
that differs from the controllers' models. Nothing about the boxes, the tray's position
or the people is given; the simulator's truth is used only to score the robot.

Background: direct-collocation trajectory optimization (CasADi/IPOPT, and a
generalized-alpha-integrator OCP pipeline for constrained multibody systems), carried
over to a real-time robotics stack: ROS 2, real-time QP-based MPC (acados SQP_RTI), and
a physics simulator (MuJoCo) as the plant.

## What it does today

**Get around the room** (sections 9a-9b of the [system overview](docs/system_overview.md))
- **Map once, then localize**: our own 2D graph SLAM (`slam/`: point-to-line ICP, an
  SE(2) pose graph, an occupancy grid) maps the room on a commissioning drive; in the job
  an extended Kalman filter fuses wheel odometry, the gyro and the scan matches (p95
  about 5-9 mm live, with people walking).
- **People**: the lidar points the map does not explain become legs, then people,
  tracked in the map frame with their velocity (99% of those within 5 m).
- **Navigate**: a Hybrid A\* route on the map, followed by a base MPC (acados, unicycle
  model) that keeps its lane, slows near people and keeps its distance from them (never
  closing in, never fleeing), at up to 1.0 m/s.
- **Dock**: both docks face their table, taught at commissioning by parking there once.
  The base joins the docking line without stopping to turn (a small miss is taken up on
  the straight approach, a larger one on a curve), approaches slowly and checks 2 cm /
  1.5 deg; live it docks within 5-8 mm.
- **Stay safe**: a separate safety layer stops the base for anything inside the
  chassis's own way to a stop (an ISO 3691-4-style protective field), slows it in a
  warning field, and watches the scans, odometry, people and localization.

**Pick and pack** (sections 5-8)
- **Torque-level MPC on the arm**: acados SQP_RTI on the arm's full dynamics
  (Pinocchio/CasADi), joint and torque limits and soft collision constraints, re-planning
  0.3 s ahead 50 times a second; an EKF predicts the state to when each torque starts;
  offset-free tracking removes the steady error.
- **Sensed picking**: the wrist camera's heightmap of the pile is segmented into boxes;
  the largest free one is picked by touch (the load cell feels the top), its weight
  checked after the lift.
- **Sensed placing**: the tray is found in the first scan and re-measured every visit;
  the boxes grow as one block from its far corner, 4 mm apart as planned, with a
  look-ahead at the boxes still on the pile. After touch-down (which measures the box's
  height) the held box is slid into its neighbour or the wall until the wrist feels it,
  then released: live, the true gap between boxes is 0.7-0.8 mm (median). Spots the
  wrist cannot reach flush are set down clear and pushed home. When the floor is full
  it stacks.
- **Arm and base at once**: the arm folds the box over the base's rear deck while the
  base backs out, and unfolds over the table while it docks.

**Work beside people** (section 9)
- The arm keeps an ISO 13855 protective distance from the people the lidars see
  (slows, holds, resumes; never dodges); the base's and the arm's distances to people
  are settings (`base_safety`, `arm_safety`, `assumed_speed`).

There is no certified scanner in a simulation: the safety functions are software
stand-ins, not safety functions.

## Run it

Setup (conda env, acados, ROS 2, MuJoCo assets): [`docs/setup.md`](docs/setup.md). On a
fresh clone the room's map and taught docks have to be made once (`data/` is not in the
repository; setup.md section 5).

**The mobile job** (`bash scripts/mobile_pickplace.sh [scenario] [settings]`): everything
comes up and holds, the people standing at their starts, until you press ENTER (keys
typed during start-up are ignored; `AUTO_START=1` before the command skips the prompt); then from home to the tray's table, then between the pile and the tray until
all 8 boxes are moved, then home. Scenarios: `crowd` (default: the job's crowd, two
people; `people:=4` for four), `quiet`, `walkers`, `dock_block` (someone standing in the
pick dock's line), `step_in` (someone stepping into the route). Settings, in the script
or after the scenario: `speed` (m/s), `base_safety`, `arm_safety`, `assumed_speed`, `people_speed`,
`localization=ekf|icp`, `conditions:=spill,worn_tyre,gyro_drift,dropout` (plant
conditions that make localization harder). For example:
`bash scripts/mobile_pickplace.sh crowd people:=4 base_safety:=0.7`.

**The parked cell** (`bash scripts/container_pickplace.sh`): the same arm task with the
base parked between the two tables; `bash scripts/container_pickplace_obstacle.sh` adds
a person who walks up to the tray, stands there and leaves (`obstacle:=visit|walk|static`,
`ceiling_camera:=true` to tell a still object apart and plan round it).

**The base alone** (`ros2 launch pick_place_mpc mobile.launch.py slam:=own
slam_mode:=mapping|localization map:=stations route:=nav people:=4`): mapping, or
navigation between the stations with people. Offline, without ROS:
`python scripts/dev/nav_sim.py --people job --crowd 4 --arm carry --box` (20 drives,
scored on the truth).

**The MoveIt method** (parked): `ros2 launch pick_place_moveit demo.launch.py`; see
below.

## Results

**The mobile job** (live, real time, 8 boxes, the base at 1.0 m/s):

| | quiet | crowd of 4 |
|---|---|---|
| boxes placed | 8 | 8 |
| time per box | 64-65 s | 104 s |
| drives docked, max error | 19/19, 7 mm | 19/19, 5-8 mm |
| true gap between neighbouring boxes, median | 0.8 mm | 0.7 mm |
| boxes on the floor (the rest stacked) | 6-7 of 8 | 7-8 of 8 |
| protective-field intrusions, contacts | 0, 0 | 0, 0 |
| closest person to the moving chassis | - | 73-95 mm |
| arm holds, all with a person near | 0 | 28-36 |
| people within 5 m tracked | - | 98.8% |

Safety settings, live with the crowd of four (before the faster docking and
the tighter packing): close (`base_safety=0.6 arm_safety=0.6`) 106 s a box, default 141
s, cautious (1.5 / 1.5, assuming 2.0 m/s walking) 221 s; 8/8 each, no intrusion or
contact. The scenario set (earlier room and speed): all five scenarios 8/8,
every drive docked within 7 mm, no intrusion or contact. Offline, the navigation's gate
and scenarios: 240/240 drives arrived. Section 10 of the
[system overview](docs/system_overview.md) has every table.

**The arm's MPC** (the parked cell, under plant/model mismatch and depth noise, seeds
1-10 with two runs each, plus an oracle whose arm matches the controller's model;
2026-09, before the realistic cell, [`results/mpc/`](results/mpc/README.md)):

| | plain | obstacle | oracle |
|---|---|---|---|
| runs finished | 16/20 | 16/20 | 19/20 |
| time per placed box [s] | 18.8 | 21.6 | 19.5 |
| one-step model error, qdot median [rad/s] | 0.175 | 0.170 | 0.0009 |
| TCP tracking while moving, RMS [mm] | 14.1 | 11.5 | 12.2 |
| arrival error, median [mm] | 0.43 | 0.45 | 0.20 |
| solve time p50 / p99 [ms] | 2.7 / 14.8 | 2.7 / 14.9 | 2.7 / 14.7 |

The mismatch is almost all of the model's one-step error, yet a perfect arm model changes
tracking, cycle time and effort by nothing measurable; only arrival (sub-millimetre)
improves. Since then, in the realistic cell: 80/80 boxes over seeds 1-10 with a person
visiting, 40/40 on fresh seeds 11-15, 2-5 late ticks in about 10 000 per run, solve p50 /
p99 2.8-3.0 / 4.4 ms.

## Architecture

**MuJoCo is the plant**, not a viewer: it computes the robot's dynamics and contacts in
real time, on its own clock, and never waits for the controllers. `mujoco_sim_node` (a
hand-written bridge) publishes the arm's noisy encoders and wrist load cell, the wheels'
encoders and the gyro, and takes joint torques and wheel speeds back; `sim_sensors_node`
renders the wrist depth camera, two ray-cast safety lidars, an optional ceiling camera
and the windows. Both publish raw sensor data, as drivers would.

**The method** is separate ROS 2 processes over framework-free libraries (`core/`,
`slam/`, `nav/`, `perception/`), so the same nodes could point at a real robot's drivers:

```
mujoco_sim_node (plant: physics)        sim_sensors_node (plant: cameras, lidars, windows)
  │ joint state, wrist load               │ wrist depth on request, lidar scans
  │ wheel encoders, gyro                  │
  ▼                                       ▼
base_node (odometry) ──▶ slam_node (localize: EKF on wheels, gyro, scan matches)
                         people_node (points the map does not explain: people, tracked)
                         nav_node (Hybrid A* route, base MPC, docking) ──▶ safety_node
                           (protective and warning fields, watchdogs) ──▶ wheel speeds
task_node (perception, pick and place decisions, packing, Ruckig reference; the job's
  stations: /nav/goal; holds and speed scale from obstacle_supervisor_node, the people
  against the ISO 13855 distance)
  ──▶ mpc_controller (EKF state estimate, acados OCP every 20 ms) ──▶ joint torques
monitors (nav, people, localization, job): truth-scored logs, never read by the method
```

**Task and method**: a task (container pick-and-place) is solved by methods that share
the scene, sensing and plant and differ only in how they move the arm. The MPC is the
main method; a MoveIt 2 + MoveIt Task Constructor method exists but is parked (it ran
full jobs on an earlier scene; on the current one it has no push, touch-down or tray
re-measure and reads the grasped box's size from the simulator). Never run both at once:
each starts its own plant.

`core/dynamics.py` (Pinocchio/CasADi, inside the OCP) and `mujoco_sim_node.py` (MuJoCo,
the plant) are independent models of the same robot, so the controllers' belief is not
forced to match what they control, as in a real deployment.

Why each piece is the way it is (shooting + adjoint sensitivities instead of
collocation, MuJoCo instead of Isaac Sim on WSL2, the collision constraints, the
people handling, the navigation and docking, the packing): [`docs/design_notes.md`](docs/design_notes.md).

## Repo layout

```
core/                              framework-agnostic: CasADi dynamics (via Pinocchio),
                                    collision-avoidance constraints, acados OCP builders
slam/                              our own 2D graph SLAM (scan merging, point-to-line ICP,
                                    pose graph, occupancy grid, localization, the EKF)
nav/                               the base's navigation (Hybrid A* route, acados base MPC,
                                    docking sequence, safety fields)
perception/                        dense-heightmap sensing and segmentation, tray detection,
                                    lidar and camera people detection, tracking
tasks/pick_and_place/
  common/                          ROS 2 pkg pick_place_common: the MuJoCo plant and sensors
                                    nodes, scene models, packing, scoring, the constants
                                    every method must agree on
  mpc/                             ROS 2 pkg pick_place_mpc: task_node, mpc_controller, the
                                    base's nodes (base, slam, people, nav, safety), people
                                    detection, obstacle supervisor, monitors, launch files
  moveit/                          ROS 2 pkg pick_place_moveit (C++): the parked MoveIt/MTC
                                    method
  interfaces/                      ROS 2 pkg pick_place_interfaces: MoveTo/Pick/Place actions
scripts/                           launch scripts, env.sh; dev/ holds offline simulation,
                                    batch runs and measurement tools
benchmarks/                        cross-method comparison harnesses and results (empty)
docs/system_overview.md            how the whole system works, end to end, with every result
docs/design_notes.md               design rationale (the "why" behind each choice)
docs/implementation_notes.md       per-module details, measurements and their reasons
docs/setup.md                      local WSL setup: conda env, acados, ROS 2, MuJoCo assets
known_issues.md                    open problems, evidence and next steps
figures/                           plots and renders referenced from the docs
results/                           batch results of the arm's MPC
handover_notes/                    working plans between sessions (local only, not in the
                                    repository)
```

## How it got here

Each step is in [`docs/implementation_notes.md`](docs/implementation_notes.md) with its
measurements.

1. **The arm's MPC in a cell** (2026-09): torque-level acados MPC fed by MuJoCo over ROS 2,
   obstacle avoidance as a parameter update; then real time with a state estimator,
   sensed boxes, contacts and pushes, people sensed by lidars (ISO 13855), a realistic
   cell with model mismatch and sensor noise.
2. **The arm on a mobile base**: the room and a MiR-like
   differential-drive base in the plant; our own SLAM (beside `slam_toolbox`); people
   seen from the moving base; navigation (Hybrid A\*, base MPC, docking, safety layer);
   the mobile pick and place; arm and base moving at once (a whole-body MPC was built
   and compared, the simpler split kept); five people scenarios; the room halved; 1.0
   m/s; safety settings; localization by sensor fusion; docks facing both tables, taught
   by driving; people in the job who are considerate of the robot.
3. **Tighter packing and smoother docking**: the placement error measured
   and split; the arm's sideways tracking error at set-down removed; one block 4 mm
   apart; the held box slid into place until the wrist feels it (gaps 6.5 mm to 0.7-0.8
   mm); pushes that record where the box stopped; the docking line joined without
   turning on the spot (in a crowd, 783-1049 deg of on-the-spot turning a job to 37).

## Limits and next steps

Open problems, with evidence: [`known_issues.md`](known_issues.md). The main ones:
- The first box in the tray's far corner can stand off its wall: at the usual set-down
  heading the wrist's long side faces the wall, and the long push that follows can jam
  (W1). Next: choose the heading (and later the grasp) for the least offset.
- In a crowd the base still steers round people standing by the tables (about 90 deg of
  extra turning a drive; W2).
- Packing is online with at most 3 boxes of look-ahead and sensed sizes; the pick order
  is the pile's, not chosen for the packing.

Still planned:
- The navigation slowing while localization is unsure (the filter's covariance).
- A mecanum base as an option beside the differential drive, the same scenarios on both.
- Solve times under MuJoCo's CPU load, acados SQP_RTI against the same OCP in IPOPT.
- Success rates over many trials and obstacle variations.
- The MPC against the MoveIt method on the same task (`benchmarks/`): packing quality,
  completion time, reaction latency.
