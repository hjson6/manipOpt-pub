# MPC baseline (step 5): results and findings

This is the "MPC alone" baseline a learning-based MPC is
compared against, seed by seed.

- Protocol, metric definitions: [`handover_notes/baseline_benchmark_plan.md`](../../handover_notes/baseline_benchmark_plan.md)
- Full table and noise table (generated): [`summary.md`](summary.md); one row per run: [`metrics.csv`](metrics.csv)
- Time per box by phase: [`time_breakdown.md`](time_breakdown.md)
- Paired test, plain vs oracle: [`compare_plain_vs_oracle.md`](compare_plain_vs_oracle.md)
- Raw runs: `runs/<scenario>_s<seed>[_r2]/` (not in git; ~5 MB each)
- Reproduce: `bash scripts/dev/results_batch.sh` (runs already on disk are kept);
  tables only: `python scripts/dev/results_summary.py results/mpc`; a paired
  comparison: `python scripts/dev/results_compare.py <a.csv>:<scenario> <b.csv>:<scenario>`
  (after `source scripts/env.sh`, with the repo root and
  `tasks/pick_and_place/common` on `PYTHONPATH`)

Runs: plain, obstacle (person visits) and oracle on test seeds 1-10, two runs
per seed, 60 runs. Oracle = the arm has no plant/model mismatch; the boxes get
the same seeded masses (0.2-2 kg) as the plain run of that seed, so oracle
seed s is the same task as plain seed s.

## Headline numbers

| | plain | obstacle | oracle |
|---|---|---|---|
| runs finished | 16/20 | 16/20 | 19/20 |
| time per placed box [s] | 18.8 | 21.6 | 19.5 |
| one-step model error, qdot median [rad/s] | 0.175 | 0.170 | 0.0009 |
| one-step model error, qdot RMS [rad/s] | 0.193 | 0.186 | 0.071 |
| TCP tracking error while moving, RMS [mm] | 14.1 | 11.5 | 12.2 |
| arrival error, median [mm] | 0.43 | 0.45 | 0.20 |
| settle time, median [s] (0.5 s is the dwell) | 0.52 | 0.52 | 0.69 |
| solve time p50 / p99 [ms] | 2.7 / 14.8 | 2.7 / 14.9 | 2.7 / 14.7 |
| min person clearance while the arm moves [m] | - | 0.117 | - |

Medians of the seed means; ranges and all other metrics in `summary.md`.

## Findings

1. **The plant/model mismatch is almost all of the model error.** One-step
   velocity error, median: 0.0009 rad/s with the arm equal to the model, 0.175
   rad/s with mismatch (joint friction the model does not have, plus damping,
   armature and mass errors); lower on all 10 seeds, p = 0.002. A large, clean
   target for a learned residual dynamics model. The oracle's RMS (0.071) is
   carried by a few percent of ticks: about half of them while the payload
   estimate is still settling after a grasp or release (0.2 s filter; boxes up
   to 2 kg), the rest not yet explained.

2. **A perfect arm model buys almost nothing in control performance.** Paired
   plain vs oracle over 10 seeds: no significant difference in tracking error
   (14.1 vs 12.2 mm, p = 0.32), time per box, effort, jerk, torque steps or
   solve time. Significant only: arrival error (0.43 -> 0.20 mm), goal bias,
   facing gaps (6.1 -> 5.6 mm), and settle time, which is *longer* with the
   perfect model (0.52 -> 0.69 s; 0/10 seeds faster, p = 0.002; cause not
   investigated, plausibly the plant's joint friction damping the final
   approach). The goal-bias integrator already removes the offset the mismatch
   causes. This is the upper bound for a learned dynamics model on this task:
   expect a large gain in prediction error and at most sub-millimetre arrival
   gains. Claims should be made on those metrics; any other gain (shorter
   horizon, lower effort) has to be shown, it does not follow from a better model.

3. **Every failed run is a packing failure, not a control failure.** All 9
   unfinished runs of 60 stopped with the tray full before the last, largest
   box was free (known_issues I10): seeds 1, 6 and 8 with mismatch, seed 10 in the oracle. No MPC
   solver failure ended a run (one failed tick in 60 runs). Completion (80% with
   mismatch) is a placement-policy metric: the clearest headroom in the system.

4. **A seed does not fully determine a run, and for tracking the noise is as
   large as the seed spread.** Within-seed SD vs between-seed SD: tracking RMS
   3.1 vs 2.8 mm, time per box 1.0 vs 0.9 s, jerk p99 620 vs 540 (plain). Model
   error (0.006 vs 0.026) and arrival are well resolved. So a learned method
   needs a tracking or cycle-time difference of several mm / ~1 s per box to be
   distinguishable with 10 seeds x 2 runs; smaller claims need more runs.

5. **Real-time headroom.** Solves take 2.7 ms median and 14.8 ms p99 against
   the 20 ms budget; ~0.3% of ticks go over. A learned model evaluated inside
   the OCP must keep p99 under 20 ms; this is the guardrail most likely to
   regress.

6. **Obstacle scenario.** 65 holds over 20 runs, all resumed; time per box
   +15% against plain. The arm was never closer than 0.117 m to the person
   while moving. The overall minimum (5 mm) is the person walking up to an arm
   that was already held. Detection error: 20 mm median.

## Where the time per box goes (plain, 18.9 s)

| phase | s / box | share | set by |
|---|---|---|---|
| fast motion (reference >= 0.05 m/s) | 10.3 | 55% | task_node's reference limits (0.5 m/s, 3 m/s^2, 20 m/s^3) |
| slow motion (touch-down 0.02, push 0.05 m/s) | 2.9 | 15% | task_node's contact speed caps |
| at goal: settle dwell, grasp/release | 3.0 | 16% | 0.5 s dwell per stop, task logic |
| pause after each scan | 2.0 | 10% | `SCAN_PAUSE_TICKS` (1 s, for log readability) |
| converging: reference stopped, arm not there yet | 0.7 | 4% | **the controller** |

(`scripts/dev/time_breakdown.py`; the scans themselves pause the sim and cost
0.2 s of wall time per box.) Only 4% of the cycle is time the controller
itself decides; a better controller shortens the cycle only by letting the
reference go faster. So a learning-based MPC is worth measuring as: the same
task at a higher reference speed/acceleration, with tracking, box tilt,
clearance and solve time held to this baseline's levels. The perfect-model
oracle converges slower (1.6 s), matching its longer settle time. Without any
learning, the 1 s scan pauses (2 s per box) could go.

## Caveats

- WSL2, viewer and dashboards on (the normal configuration); absolute timings
  are for this machine.
- The obstacle scenario has no oracle; its model-error and tracking numbers
  match plain.
- Oracle runs had 4 push legs reach their 6 s limit (TCP within 2 mm of the
  push target) and 2 re-planned places; none in the mismatch runs. Not
  investigated.
