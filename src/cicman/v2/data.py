"""Dataset over the precomputed multi-view window cache (views_v2)."""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np

ALL_VIEWS = ["raw", "denoise", "env_spec", "env_order", "stft", "cwt"]


class ViewCache:
    """Memory-mapped access to the global view cache."""

    _shared: dict[tuple[str, tuple[str, ...], bool], tuple[dict, np.ndarray, dict]] = {}

    def __init__(self, cache_dir: str | Path, views: list[str] | None = None):
        self.cache_dir = Path(cache_dir)
        self.views = views or ALL_VIEWS
        preload = os.environ.get("CICMAN_PRELOAD_CACHE", "0") == "1"
        key = (str(self.cache_dir.resolve()), tuple(self.views), preload)
        if key in self._shared:
            self.arrays, self.feats, self.key_to_row = self._shared[key]
            return
        self.arrays = {
            v: (np.load(self.cache_dir / f"{v}.npy", mmap_mode="r").copy()
                if preload else np.load(self.cache_dir / f"{v}.npy", mmap_mode="r"))
            for v in self.views
        }
        feat_map = np.load(self.cache_dir / "feats.npy", mmap_mode="r")
        self.feats = feat_map.copy() if preload else feat_map
        self.key_to_row: dict[tuple[str, str, int], int] = {}
        with (self.cache_dir / "master.csv").open(newline="", encoding="utf-8") as f:
            for i, row in enumerate(csv.DictReader(f)):
                self.key_to_row[(row["dataset_id"], row["recording_id"], int(row["window_index"]))] = i
        self._shared[key] = (self.arrays, self.feats, self.key_to_row)


class CachedWindowDataset:
    """Joins a split window-index CSV against the view cache."""

    def __init__(
        self,
        index_csv: str | Path,
        cache: ViewCache,
        *,
        label_column: str = "label_id",
        domain_column: str = "dataset_id",
        dataset_filter: set[str] | None = None,
        domain_filter: set[str] | None = None,
        recording_filter: set[str] | None = None,
    ):
        self.cache = cache
        self.rows = []
        missing = 0
        with Path(index_csv).open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if dataset_filter is not None and row["dataset_id"] not in dataset_filter:
                    continue
                if recording_filter is not None and row["recording_id"] not in recording_filter:
                    continue
                if domain_filter is not None:
                    domain_value = row.get(domain_column, row["dataset_id"]) or row["dataset_id"]
                    if domain_value not in domain_filter:
                        continue
                key = (row["dataset_id"], row["recording_id"], int(row["window_index"]))
                cache_row = cache.key_to_row.get(key)
                if cache_row is None:
                    missing += 1
                    continue
                self.rows.append(
                    {
                        "cache_row": cache_row,
                        "label": int(row[label_column]),
                        "dataset_id": row["dataset_id"],
                        "domain": row.get(domain_column, row["dataset_id"]) or row["dataset_id"],
                        "recording_id": row["recording_id"],
                        "window_index": int(row["window_index"]),
                    }
                )
        if missing:
            print(f"warning: {missing} index rows not found in view cache ({index_csv})")
        self.datasets = sorted({r["dataset_id"] for r in self.rows})
        self.domains = sorted({r["domain"] for r in self.rows})
        self.domain_to_id = {name: i for i, name in enumerate(self.domains)}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        import torch

        row = self.rows[index]
        i = row["cache_row"]
        views = {
            v: torch.from_numpy(np.array(self.cache.arrays[v][i], dtype=np.float32, copy=True, order="C"))
            for v in self.cache.views
        }
        feats = torch.from_numpy(np.array(self.cache.feats[i], dtype=np.float32, copy=True, order="C"))
        return {
            "views": views,
            "feats": feats,
            "y": torch.tensor(row["label"], dtype=torch.long),
            "domain": torch.tensor(self.domain_to_id.get(row["domain"], 0), dtype=torch.long),
            "index": index,
        }

    def fetch_batch(self, indices):
        """Vectorized equivalent of collating ``__getitem__`` results.

        NumPy advanced indexing performs one contiguous read per view instead
        of thousands of small Python-level memmap reads.  Values and ordering
        are identical to the default DataLoader path.
        """
        import torch

        indices = [int(i) for i in indices]
        rows = [self.rows[i] for i in indices]
        cache_rows = [r["cache_row"] for r in rows]
        views = {
            v: torch.from_numpy(np.ascontiguousarray(self.cache.arrays[v][cache_rows], dtype=np.float32))
            for v in self.cache.views
        }
        feats = torch.from_numpy(np.ascontiguousarray(self.cache.feats[cache_rows], dtype=np.float32))
        return {
            "views": views,
            "feats": feats,
            "y": torch.tensor([r["label"] for r in rows], dtype=torch.long),
            "domain": torch.tensor([self.domain_to_id.get(r["domain"], 0) for r in rows], dtype=torch.long),
            "index": torch.tensor(indices, dtype=torch.long),
        }

    def balanced_sample_weights(self) -> np.ndarray:
        """Inverse domain-x-class frequency weights for WeightedRandomSampler."""
        return _balanced_weights(self.rows)


class UnionWindowDataset:
    """Concatenation of CachedWindowDatasets (e.g. measurement-intervention
    variants of the same windows) with a unified domain mapping."""

    def __init__(self, parts: list[CachedWindowDataset]):
        self.parts = parts
        self.rows = [r for p in parts for r in p.rows]
        self.datasets = sorted({r["dataset_id"] for r in self.rows})
        self.domains = sorted({r["domain"] for r in self.rows})
        self.domain_to_id = {name: i for i, name in enumerate(self.domains)}
        self._offsets = []
        total = 0
        for p in parts:
            self._offsets.append(total)
            total += len(p)
        self._total = total

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, index: int):
        for part, offset in zip(reversed(self.parts), reversed(self._offsets)):
            if index >= offset:
                item = part[index - offset]
                row = part.rows[index - offset]
                import torch

                item["domain"] = torch.tensor(self.domain_to_id.get(row["domain"], 0), dtype=torch.long)
                return item
        raise IndexError(index)

    def fetch_batch(self, indices):
        """Vectorized batch fetch preserving the union's requested order."""
        import torch

        indices = [int(i) for i in indices]
        locations = []
        for index in indices:
            found = None
            for part_i in range(len(self.parts) - 1, -1, -1):
                offset = self._offsets[part_i]
                if index >= offset:
                    found = (part_i, index - offset)
                    break
            if found is None:
                raise IndexError(index)
            locations.append(found)

        by_part: dict[int, list[tuple[int, int]]] = {}
        for out_pos, (part_i, local_i) in enumerate(locations):
            by_part.setdefault(part_i, []).append((out_pos, local_i))

        batches = {}
        for part_i, pairs in by_part.items():
            local = [local_i for _, local_i in pairs]
            batches[part_i] = (pairs, self.parts[part_i].fetch_batch(local))

        views = {}
        for view in self.parts[0].cache.views:
            sample_shape = batches[next(iter(batches))][1]["views"][view].shape[1:]
            out = torch.empty((len(indices), *sample_shape), dtype=torch.float32)
            for pairs, batch in batches.values():
                pos = torch.tensor([p for p, _ in pairs], dtype=torch.long)
                out[pos] = batch["views"][view]
            views[view] = out
        feat_dim = self.parts[0].cache.feats.shape[1]
        feats = torch.empty((len(indices), feat_dim), dtype=torch.float32)
        labels = torch.empty(len(indices), dtype=torch.long)
        domains = torch.empty(len(indices), dtype=torch.long)
        for part_i, (pairs, batch) in batches.items():
            pos = torch.tensor([p for p, _ in pairs], dtype=torch.long)
            feats[pos] = batch["feats"]
            labels[pos] = batch["y"]
            local_rows = [self.parts[part_i].rows[local_i] for _, local_i in pairs]
            domains[pos] = torch.tensor(
                [self.domain_to_id.get(r["domain"], 0) for r in local_rows], dtype=torch.long
            )
        return {
            "views": views,
            "feats": feats,
            "y": labels,
            "domain": domains,
            "index": torch.tensor(indices, dtype=torch.long),
        }

    def balanced_sample_weights(self) -> np.ndarray:
        return _balanced_weights(self.rows)


def _balanced_weights(rows) -> np.ndarray:
    counts: dict[tuple[str, int], int] = {}
    for r in rows:
        key = (r["domain"], r["label"])
        counts[key] = counts.get(key, 0) + 1
    return np.array([1.0 / counts[(r["domain"], r["label"])] for r in rows], dtype=np.float64)
