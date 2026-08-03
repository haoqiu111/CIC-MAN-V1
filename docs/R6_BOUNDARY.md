# R6 reconstruction boundary

## What this package represents

This repository is aligned to the R6 manuscript and its R6 result archive. It was produced after the manuscript work had continued to R7/R8, while the original workspace did not contain a usable Git history. Consequently, this package is a semantic reconstruction rather than a byte-identical historical snapshot.

The reconstruction was performed by copying the current code to a separate directory, leaving the current R7/R8 workspace untouched, and removing later-only experiments and implementation branches.

## Excluded post-R6 work

- R7 matched-ablation five-seed extension and R7/R8 evidence-manifest scripts.
- R8 CWRU external audit and all-source CWRU training.
- R8 `v6ic_pair_guard` preset.
- R8 paired identity/HP800 prediction-consistency loss.
- R8 source-calibrated conservative routing fallback.

## Added packaging-only work

- relative/environment-configurable paths for primary entry points;
- `README.md`, `DATA.md`, `pyproject.toml`, `.gitignore`, tests, citation metadata, and license notice;
- a corrected R6 Table 4 summary replacing an older v4 table that shared a historical filename.

These additions do not create new target results.

## Evidence limitations

The compact archive contains summary tables, selected per-seed shortcut-reversal reports, hashes, and environment provenance. Raw datasets, full view caches, and model checkpoints are not included. The provenance file records the environment used for R6 audit/re-inference and explicitly does not infer the unrecorded historical training environment of older checkpoints.
