# Paper-wide metric protocol

Frozen official recording metric:

- checkpoint: fixed-budget **last epoch**;
- aggregation: **true majority vote** over per-window argmax predictions within each recording;
- tie rule: among vote-tied classes, select the class with the largest cumulative per-window probability; if those sums also tie exactly, select the smallest class id;
- primary statistic: recording-level macro-F1, averaged over the three target datasets per seed and then over the five seeds;
- robustness statistic: strict worst-target macro-F1, the minimum of the three target values within each seed, then averaged over seeds;
- sensitivity analysis: probability-sum aggregation (sum per-window class probabilities, then argmax), with best-epoch and last-epoch variants retained in the audit.

No original `metrics.json` file is overwritten. Fresh checkpoint inference is stored in `recording_aggregation_audit.csv`/`.json`; the official summary is `v2_multiseed_main_matrix.md`.
