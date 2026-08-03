# CIC-MAN R6 reproduction code

This repository packages the code and compact evidence archive corresponding to the R6 manuscript:

> CIC-MAN: Causal intervention-consistent multi-agent for target-free cross-dataset bearing fault diagnosis

CIC-MAN represents each vibration window through six signal-processing views, separates health and rig information, and routes the views using a source-only reliability prior evaluated on identity and HP800 counterfactual data.

## R6 headline result

Under strict target-free leave-one-dataset-out evaluation on Paderborn, HUST, and Ottawa, using five seeds, last-epoch checkpoints, and recording-level majority voting, CIC-MAN v6ic obtained:

| Metric | R6 value |
|---|---:|
| Mean recording macro-F1 | 0.547 +/- 0.013 |
| HUST | 0.467 +/- 0.039 |
| Ottawa | 0.829 +/- 0.045 |
| Paderborn / strict worst target | 0.345 +/- 0.014 |

The mean advantage is concentrated on Ottawa. The Paderborn result remains close to three-class chance level and the method is not presented as deployment-ready.

The matched R6 ablation is in [results/r6/matched_v6ic_ablation.md](results/r6/matched_v6ic_ablation.md). The full compact result archive is under [results/r6](results/r6).

The R6 revision evidence added 75 runs: 45 matched v6ic ablations, 15 equal-backbone CCN reimplementation runs, and 15 five-seed shortcut-reversal runs.

## Version boundary

This is an R6-compatible reconstruction, not a byte-for-byte historical Git checkout. The original experiment workspace had no usable commit history. The package was reconstructed from the later working tree and checked against the R6 manuscript and R6 evidence files.

R7/R8-only experiment routes are excluded, including:

- the R7 five-seed extension of the matched ablation;
- the R8 CWRU external-target experiments;
- the R8 `v6ic_pair_guard` configuration;
- the R8 explicit identity-HP800 paired loss;
- the R8 conservative single-view fallback.

Portable paths, packaging metadata, and tests were added for this public archive and do not define new manuscript experiments. See [docs/R6_BOUNDARY.md](docs/R6_BOUNDARY.md).

## Repository layout

```text
configs/                 data and experiment examples
recording_protocol/      canonical recording-level aggregation
results/r6/              compact R6 tables and environment provenance
scripts/prepare_data/    manifests, splits, windows, and view caches
scripts/train/           CIC-MAN and baseline training entry points
scripts/evaluate/        recording and perturbation evaluation
scripts/analysis/        tables, figures, and mechanism audits
src/cicman/              model, views, data, and training implementation
tests/                   lightweight protocol/version tests
```

Large public datasets, cached views, and checkpoints are intentionally not included.

## Installation

Python 3.10 or newer is recommended. Install PyTorch separately for the CUDA or CPU build appropriate to your machine, then install this repository:

```bash
python -m venv .venv
# activate the environment using your shell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e ".[test]"
```

The verified R6 audit machine reported PyTorch `2.13.0+cu132` and CUDA build `13.2`. This is preserved as observed provenance, not imposed as a portable dependency pin. See [results/r6/environment_provenance_20260721.json](results/r6/environment_provenance_20260721.json) and the archived `pip freeze` file beside it.

## Data layout

Prepare the licensed/downloaded datasets outside Git. The primary scripts expect this structure under `--project-root`:

```text
<project-root>/
  data/paper1_cicman/
    raw_copy/
      paderborn/
      hust/
      ottawa/
    manifests/
    splits/
    cache/
```

Dataset acquisition and label-policy notes are in [DATA.md](DATA.md). Do not commit raw recordings or derived caches unless their licenses explicitly permit redistribution.

## Main reproduction flow

The exact commands depend on where the three datasets were downloaded. A typical sequence is:

```bash
python scripts/prepare_data/build_manifests.py --help
python scripts/prepare_data/build_splits.py --help
python scripts/prepare_data/build_view_cache.py --help
python scripts/prepare_data/build_source_mixed_cross_dataset_windows.py --help
```

Run one R6 model/target/seed combination:

```bash
python scripts/train/run_v2_pilot.py \
  --project-root /path/to/workspace \
  --output-root ./outputs \
  --models cicman_v6ic \
  --targets paderborn \
  --seed 42 \
  --epochs 40 \
  --isolate
```

The `cicman_v6ic` preset uses the six views, the source-only intervention-consistent reliability prior, the identity+HP800 training union, and the final fixed training budget. Use `python scripts/train/train_cicman_v2.py --help` to inspect all R6-era presets. The fixed three-seed matched component audit is launched by `python scripts/train/run_v6ic_ablation_batch.py`.

## Evaluation protocol

The official metric is fixed in [recording_protocol/aggregation.py](recording_protocol/aggregation.py):

1. use the last-epoch checkpoint;
2. obtain one argmax vote per window;
3. select the majority class per recording;
4. break a vote tie by cumulative probability among tied classes;
5. break an exact probability tie by the smallest class id;
6. compute recording-level macro-F1.

Seeds measure optimization variability within a target rig. They are not independent test rigs and must not be promoted to 15 independent cross-rig samples.

## Tests

```bash
python -m pytest -q
python -m compileall -q src scripts recording_protocol tests
```

## Citation and license

The manuscript was submitted concurrently with the FMG-MATA companion study. Replace the provisional metadata in [CITATION.cff](CITATION.cff) after a DOI or preprint identifier is assigned.

No open-source license was selected on the author's behalf. The current [LICENSE](LICENSE) is all-rights-reserved; replace it before publication if you want to permit reuse, modification, or redistribution.
