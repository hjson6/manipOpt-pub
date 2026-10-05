# results/mpc/metrics.csv:oracle vs results/mpc/metrics.csv:plain

Seed means; diff = B - A per seed. p: Wilcoxon signed-rank, two-sided. diff/noise: median diff over A's within-seed SD.

| metric | A median | B median | median diff | B lower (seeds) | p | diff/noise |
|---|---|---|---|---|---|---|
| boxes | 8 | 8 | +0 | 1/10 | 0.375 | +0 |
| done | 1 | 1 | +0 | 1/10 | 0.5 | +nan |
| time_per_box_s | 18.8 | 19.5 | +0.62 | 2/10 | 0.232 | +0.62 |
| pred_q_rms_mrad | 1.98 | 0.739 | -1.35 | 10/10 | 0.00195 | +nan |
| pred_qd_rms | 0.193 | 0.0712 | -0.137 | 10/10 | 0.00195 | -23 |
| pred_qd_med | 0.175 | 0.000863 | -0.174 | 10/10 | 0.00195 | +nan |
| track_rms_mm | 14.1 | 12.2 | -1.12 | 6/10 | 0.322 | -0.37 |
| track_p99_mm | 57.5 | 46.8 | -8.47 | 6/10 | 0.557 | +nan |
| tsway_rms_mm | 6.9 | 6.45 | +0 | 5/10 | 0.643 | +0 |
| arrive_med_mm | 0.425 | 0.2 | -0.225 | 10/10 | 0.00195 | -4.5 |
| settle_med_s | 0.518 | 0.688 | +0.17 | 0/10 | 0.00195 | +10 |
| bias_mean_mm | 0.742 | 0.362 | -0.454 | 10/10 | 0.00195 | +nan |
| solve_p50_ms | 2.68 | 2.69 | +0.0125 | 3/10 | 0.389 | +nan |
| solve_p99_ms | 14.8 | 14.7 | +0.0925 | 4/10 | 0.922 | +0.16 |
| over_budget | 24.2 | 27.2 | +1.75 | 4/10 | 0.791 | +nan |
| jerk_p99 | 1.28e+03 | 1.39e+03 | -8 | 5/10 | 0.77 | -0.013 |
| dtau_p99 | 9.73 | 10.2 | -0.845 | 6/10 | 0.77 | +nan |
| effort_per_box | 2.86e+04 | 2.72e+04 | -1.63e+03 | 6/10 | 0.432 | -0.62 |
| tilt_p99_deg | 1.67 | 1.38 | +0.025 | 5/10 | 0.9 | +nan |
| gap_med_mm | 6.08 | 5.6 | -0.725 | 9/10 | 0.00391 | +nan |
