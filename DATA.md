# Data availability and local layout

The R6 experiments use three third-party bearing datasets: Paderborn, HUST, and the Ottawa variable-speed bearing dataset. Their raw files are not part of this repository.

Users must obtain each dataset from its official distributor, follow its license or terms of use, and place a local copy under `data/paper1_cicman/raw_copy/`. The preparation scripts generate manifests, splits, window indices, and six-view caches from those local files.

The task is three-class diagnosis with dataset-specific labels mapped to normal, inner-race fault, and outer-race fault. Splits are recording-disjoint. In the cross-dataset leave-one-dataset-out protocol, the target dataset is excluded from training, source validation, reliability-prior construction, model selection, and threshold selection.

Generated caches can be large and are ignored by Git. Before sharing a derived cache, check whether the source dataset license permits redistribution.

For a new machine, copy `configs/data/paths.example.json`, replace the example placeholders with local paths, and run the preparation scripts with `--help` first. The repository does not contain credentials, private download links, or raw recordings.
