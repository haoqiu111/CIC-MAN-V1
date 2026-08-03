# v2 Cross-Dataset Main Matrix (official: last-epoch, majority vote)

3-class leave-one-dataset-out, strict target-free, recording-level macro-F1.
The recording metric is freshly re-inferred from checkpoints; mean and worst-target aggregate over seeds.

## Per target (mean +- sample std over available seeds)

| Model | Target | Rec Macro-F1 | Win Macro-F1 | Seeds |
|---|---|---|---|---:|
| cicman_v4 | hust | 0.519 +- 0.087 | 0.500 +- 0.062 | 5 |
| cicman_v4 | ottawa | 0.509 +- 0.222 | 0.453 +- 0.054 | 5 |
| cicman_v4 | paderborn | 0.372 +- 0.020 | 0.369 +- 0.020 | 5 |
| cicman_v6ic | hust | 0.467 +- 0.039 | 0.451 +- 0.019 | 5 |
| cicman_v6ic | ottawa | 0.829 +- 0.045 | 0.549 +- 0.026 | 5 |
| cicman_v6ic | paderborn | 0.345 +- 0.014 | 0.347 +- 0.010 | 5 |
| dann | hust | 0.266 +- 0.034 | 0.265 +- 0.023 | 5 |
| dann | ottawa | 0.210 +- 0.147 | 0.255 +- 0.081 | 5 |
| dann | paderborn | 0.188 +- 0.042 | 0.190 +- 0.038 | 5 |
| dg_coral | hust | 0.367 +- 0.051 | 0.387 +- 0.033 | 5 |
| dg_coral | ottawa | 0.211 +- 0.067 | 0.345 +- 0.024 | 5 |
| dg_coral | paderborn | 0.362 +- 0.034 | 0.350 +- 0.025 | 5 |
| dg_groupdro | hust | 0.461 +- 0.042 | 0.476 +- 0.033 | 5 |
| dg_groupdro | ottawa | 0.317 +- 0.130 | 0.394 +- 0.052 | 5 |
| dg_groupdro | paderborn | 0.335 +- 0.017 | 0.338 +- 0.012 | 5 |
| dg_irm | hust | 0.374 +- 0.069 | 0.404 +- 0.054 | 5 |
| dg_irm | ottawa | 0.258 +- 0.236 | 0.326 +- 0.123 | 5 |
| dg_irm | paderborn | 0.350 +- 0.017 | 0.345 +- 0.010 | 5 |
| dg_mmd | hust | 0.417 +- 0.107 | 0.428 +- 0.080 | 5 |
| dg_mmd | ottawa | 0.345 +- 0.321 | 0.395 +- 0.113 | 5 |
| dg_mmd | paderborn | 0.342 +- 0.015 | 0.339 +- 0.018 | 5 |
| ensemble | hust | 0.392 +- 0.059 | 0.407 +- 0.036 | 5 |
| ensemble | ottawa | 0.387 +- 0.080 | 0.401 +- 0.037 | 5 |
| ensemble | paderborn | 0.350 +- 0.035 | 0.350 +- 0.028 | 5 |
| moe | hust | 0.483 +- 0.034 | 0.494 +- 0.012 | 5 |
| moe | ottawa | 0.332 +- 0.101 | 0.404 +- 0.076 | 5 |
| moe | paderborn | 0.334 +- 0.031 | 0.335 +- 0.025 | 5 |
| single_env_order | hust | 0.532 +- 0.028 | 0.512 +- 0.025 | 5 |
| single_env_order | ottawa | 0.645 +- 0.061 | 0.525 +- 0.024 | 5 |
| single_env_order | paderborn | 0.342 +- 0.011 | 0.361 +- 0.009 | 5 |
| single_raw | hust | 0.299 +- 0.030 | 0.299 +- 0.036 | 5 |
| single_raw | ottawa | 0.295 +- 0.097 | 0.288 +- 0.040 | 5 |
| single_raw | paderborn | 0.165 +- 0.017 | 0.167 +- 0.018 | 5 |

## Mean over targets and strict worst target (per seed, then aggregated)

| Model | Mean Rec Macro-F1 | Strict worst-target Rec Macro-F1 | Seeds |
|---|---|---|---:|
| cicman_v4 | 0.467 +- 0.076 | 0.333 +- 0.075 | 5 |
| cicman_v6ic | 0.547 +- 0.013 | 0.345 +- 0.014 | 5 |
| dann | 0.221 +- 0.036 | 0.137 +- 0.049 | 5 |
| dg_coral | 0.313 +- 0.012 | 0.211 +- 0.067 | 5 |
| dg_groupdro | 0.371 +- 0.040 | 0.274 +- 0.064 | 5 |
| dg_irm | 0.327 +- 0.088 | 0.188 +- 0.142 | 5 |
| dg_mmd | 0.368 +- 0.127 | 0.214 +- 0.142 | 5 |
| ensemble | 0.376 +- 0.046 | 0.332 +- 0.048 | 5 |
| moe | 0.383 +- 0.040 | 0.285 +- 0.060 | 5 |
| single_env_order | 0.506 +- 0.016 | 0.342 +- 0.011 | 5 |
| single_raw | 0.253 +- 0.042 | 0.165 +- 0.017 | 5 |
