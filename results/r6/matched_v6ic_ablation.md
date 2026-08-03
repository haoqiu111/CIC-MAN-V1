# CIC-MAN R6 matched v6ic ablation

Three seeds (42, 2025, 2026), three strict leave-one-dataset-out targets, identity+HP800 training union, last-epoch checkpoint, and recording-level majority vote with probability tie-breaking.

| Configuration | Mean recording macro-F1 | Delta vs. full |
|---|---:|---:|
| Full CIC-MAN v6ic | 0.542 +/- 0.015 | -- |
| Without intervention-consistent reliability prior | 0.355 +/- 0.017 | -0.187 |
| Without health/domain disentanglement | 0.534 +/- 0.049 | -0.008 |
| Without consensus loss | 0.538 +/- 0.029 | -0.004 |
| Without view dropout | 0.515 +/- 0.015 | -0.027 |
| Without consensus/uncertainty router evidence | 0.471 +/- 0.045 | -0.071 |

These values are the R6 Table 4 values. They are not the older v4 ablation that existed under the same historical table filename.
