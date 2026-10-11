# The route's policy `tuned` against the rules and the hybrid (paired by seed)

From `results/rl/baseline_rules.json` (commit d83c551-dirty). Change per drive, mean over seeds [95% interval over seeds] (relative to the reference).

## job2

| metric | against the rules | against the hybrid |
|---|---|---|
| drive s | -6.45 [-6.76, -6.12] (-21%) | - |
| stop onsets on routes | -0.71 [-0.95, -0.47] (-44%) | - |
| swerves | -0.08 [-0.20, +0.05] (-11%) | - |
| extra turning deg | -24.21 [-28.23, -19.98] (-31%) | - |
| warned s on routes | -0.04 [-0.10, +0.01] (-3%) | - |

Hits (intrusions + contacts) 0 (the rules 0); arrived 400/400; closest while closing, p5, chassis / arm or box: job2 140 mm / 253 mm; job2_hybrid - / -; job2_tuned 138 mm / 181 mm

## job4

| metric | against the rules | against the hybrid |
|---|---|---|
| drive s | -6.57 [-7.04, -6.12] (-21%) | +0.56 [+0.14, +1.00] (+2%) |
| stop onsets on routes | -1.14 [-1.45, -0.85] (-53%) | -0.08 [-0.34, +0.19] (-7%) |
| swerves | -0.19 [-0.32, -0.08] (-25%) | -0.15 [-0.29, -0.01] (-20%) |
| extra turning deg | -29.96 [-36.35, -23.75] (-33%) | +9.69 [+2.31, +16.91] (+19%) |
| warned s on routes | -0.01 [-0.07, +0.06] (-0%) | +0.05 [-0.01, +0.12] (+3%) |

Hits (intrusions + contacts) 0 (the rules 0); arrived 400/400; closest while closing, p5, chassis / arm or box: job4 141 mm / 255 mm; job4_hybrid 137 mm / 175 mm; job4_tuned 144 mm / 179 mm

## walkers3

| metric | against the rules | against the hybrid |
|---|---|---|
| drive s | -10.39 [-11.10, -9.64] (-27%) | +0.25 [-0.21, +0.73] (+1%) |
| stop onsets on routes | -4.53 [-5.09, -3.95] (-49%) | +0.29 [-0.18, +0.75] (+6%) |
| swerves | +0.29 [+0.14, +0.46] (+30%) | +0.40 [+0.27, +0.55] (+46%) |
| extra turning deg | -22.45 [-30.43, -14.41] (-19%) | +24.88 [+15.24, +33.52] (+34%) |
| warned s on routes | +0.10 [+0.04, +0.16] (+7%) | -0.12 [-0.19, -0.06] (-7%) |

Hits (intrusions + contacts) 0 (the rules 0); arrived 400/400; closest while closing, p5, chassis / arm or box: walkers3 109 mm / 211 mm; walkers3_hybrid 127 mm / 220 mm; walkers3_tuned 117 mm / 213 mm

## stations6

| metric | against the rules | against the hybrid |
|---|---|---|
| drive s | -9.78 [-10.56, -8.87] (-25%) | -0.51 [-1.21, +0.16] (-2%) |
| stop onsets on routes | -4.64 [-5.42, -3.90] (-49%) | -0.52 [-1.20, +0.19] (-10%) |
| swerves | +0.50 [+0.32, +0.69] (+37%) | +0.71 [+0.55, +0.90] (+63%) |
| extra turning deg | +6.62 [-3.33, +17.45] (+5%) | +37.58 [+26.86, +48.08] (+37%) |
| warned s on routes | +0.33 [+0.23, +0.44] (+20%) | -0.05 [-0.13, +0.03] (-3%) |

Hits (intrusions + contacts) 0 (the rules 0); arrived 400/400; closest while closing, p5, chassis / arm or box: stations6 97 mm / 207 mm; stations6_hybrid 121 mm / 218 mm; stations6_tuned 110 mm / 217 mm

## dock_block

| metric | against the rules | against the hybrid |
|---|---|---|
| drive s | -5.73 [-7.70, -3.99] (-14%) | - |
| stop onsets on routes | -1.52 [-1.91, -1.12] (-40%) | - |
| swerves | -0.05 [-0.11, +0.01] (-10%) | - |
| extra turning deg | -10.12 [-13.05, -7.43] (-37%) | - |
| warned s on routes | +0.04 [+0.00, +0.08] (+4%) | - |

Hits (intrusions + contacts) 0 (the rules 0); arrived 400/400; closest while closing, p5, chassis / arm or box: dock_block 219 mm / 184 mm; dock_block_hybrid - / -; dock_block_tuned 172 mm / 187 mm

## step_in

| metric | against the rules | against the hybrid |
|---|---|---|
| drive s | -4.42 [-4.58, -4.28] (-16%) | - |
| stop onsets on routes | +0.00 [+0.00, +0.00] (+nan%) | - |
| swerves | -0.91 [-0.95, -0.86] (-98%) | - |
| extra turning deg | -25.21 [-25.45, -24.95] (-59%) | - |
| warned s on routes | +0.20 [+0.19, +0.21] (+17%) | - |

Hits (intrusions + contacts) 0 (the rules 0); arrived 400/400; closest while closing, p5, chassis / arm or box: step_in 357 mm / 322 mm; step_in_hybrid - / -; step_in_tuned 353 mm / 317 mm

## random6

| metric | against the rules | against the hybrid |
|---|---|---|
| drive s | -6.09 [-7.89, -4.46] (-18%) | -0.37 [-1.18, +0.49] (-1%) |
| stop onsets on routes | -1.06 [-2.57, +0.47] (-24%) | -0.99 [-2.17, +0.31] (-23%) |
| swerves | +0.05 [-0.17, +0.25] (+7%) | +0.10 [-0.06, +0.27] (+14%) |
| extra turning deg | -15.88 [-27.82, -4.12] (-19%) | +12.98 [+4.43, +22.59] (+23%) |
| warned s on routes | +0.05 [-0.05, +0.15] (+4%) | -0.14 [-0.21, -0.06] (-8%) |

Hits (intrusions + contacts) 0 (the rules 0); arrived 400/400; closest while closing, p5, chassis / arm or box: random6 127 mm / 226 mm; random6_hybrid 131 mm / 226 mm; random6_tuned 116 mm / 218 mm

## quiet

| metric | against the rules | against the hybrid |
|---|---|---|
| drive s | -0.39 [-0.41, -0.37] (-2%) | - |
| stop onsets on routes | +0.00 [+0.00, +0.00] (+nan%) | - |
| swerves | +0.00 [+0.00, +0.00] (+nan%) | - |
| extra turning deg | +1.08 [+0.99, +1.18] (+9%) | - |
| warned s on routes | -0.13 [-0.13, -0.12] (-12%) | - |

Hits (intrusions + contacts) 0 (the rules 0); arrived 400/400; closest while closing, p5, chassis / arm or box: quiet - / -; quiet_hybrid - / -; quiet_tuned - / -

## Safety gates

| crowd | hits | closest p5 (no smaller than the hybrid's, 10 mm slack) | every drive arrived |
|---|---|---|---|
| job2 | ok | ok | ok |
| job4 | ok | ok | ok |
| walkers3 | ok | ok | ok |
| stations6 | ok | FAIL | ok |
| dock_block | ok | ok | ok |
| step_in | ok | ok | ok |
| random6 | ok | FAIL | ok |
| quiet | ok | ok | ok |
