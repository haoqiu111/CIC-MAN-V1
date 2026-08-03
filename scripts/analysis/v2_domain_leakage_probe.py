#!/usr/bin/env python3
"""Domain-leakage probes on frozen CIC-MAN v2/v6ic representations.

For a trained checkpoint, extracts per-view health (z_h) and domain (z_d)
embeddings plus the router-fused health embedding on SOURCE validation
windows, then fits linear probes predicting the source dataset id.
Low accuracy on z_h and high accuracy on z_d = successful disentanglement.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


def add_src_to_path() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


add_src_to_path()

from cicman.v2.data import CachedWindowDataset, ViewCache  # noqa: E402
from cicman.v2.model import CICMANv2  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--model", default="cicman_v6ic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--which", choices=["best", "last"], default="last")
    parser.add_argument("--max-windows", type=int, default=8000)
    args = parser.parse_args()

    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    root = args.project_root
    windows_root = root / "data/paper1_cicman/cache/windows/cross_dataset_task3_source_mixed"
    cache_root = root / "data/paper1_cicman/cache/views_v2"
    ckpt_root = root / "outputs/checkpoints"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    results = []
    for target in ["hust", "ottawa", "paderborn"]:
        ckpt = torch.load(ckpt_root / f"v2_{args.model}_target_{target}_seed{args.seed}" / f"{args.which}.pt", map_location="cpu")
        views = ckpt["views"]
        state = ckpt["model"]
        model = CICMANv2(num_classes=3, views=views,
                         num_domains=state["domain_adv_head.2.weight"].shape[0],
                         router_mode="causal", router_prior=[1.0 / len(views)] * len(views))
        model.load_state_dict(state, strict=False)
        model = model.to(device).eval()

        cache = ViewCache(cache_root, views=views)
        val_ds = CachedWindowDataset(windows_root / f"target_dataset_{target}" / "val_windows.csv", cache)
        idx = np.arange(len(val_ds))
        if len(idx) > args.max_windows:
            idx = rng.choice(idx, size=args.max_windows, replace=False)

        zh_fused, zh_all, zd_all, doms = [], [], [], []
        from torch.utils.data import DataLoader, Subset

        loader = DataLoader(Subset(val_ds, idx.tolist()), batch_size=256, shuffle=False)
        with torch.no_grad():
            for batch in loader:
                x = {v: batch["views"][v].to(device) for v in views}
                out = model(x, batch["feats"].to(device))
                w = out["router_weights"].unsqueeze(-1)
                zh_fused.append((out["z_h"] * w).sum(1).cpu().numpy())
                zh_all.append(out["z_h"].mean(1).cpu().numpy())
                zd_all.append(out["z_d"].mean(1).cpu().numpy())
                doms.append(batch["domain"].numpy())
        zh_fused = np.concatenate(zh_fused)
        zh_all = np.concatenate(zh_all)
        zd_all = np.concatenate(zd_all)
        doms = np.concatenate(doms)

        def probe(feats):
            if len(np.unique(doms)) < 2:
                return float("nan"), float("nan")
            Xtr, Xte, ytr, yte = train_test_split(feats, doms, test_size=0.3, random_state=args.seed, stratify=doms)
            sc = StandardScaler().fit(Xtr)
            clf = LogisticRegression(max_iter=1000).fit(sc.transform(Xtr), ytr)
            acc = float(clf.score(sc.transform(Xte), yte))
            majority = float(np.max(np.bincount(yte)) / len(yte))
            return acc, majority

        acc_h, maj = probe(zh_fused)
        acc_hm, _ = probe(zh_all)
        acc_d, _ = probe(zd_all)
        results.append({
            "target": target, "model": args.model,
            "domain_acc_on_fused_health": round(acc_h, 4),
            "domain_acc_on_mean_health": round(acc_hm, 4),
            "domain_acc_on_domain_repr": round(acc_d, 4),
            "majority_baseline": round(maj, 4),
            "num_windows": int(len(doms)),
        })
        print(results[-1], flush=True)

    out = root / "outputs/tables/v2_domain_leakage.md"
    with out.open("w", encoding="utf-8") as f:
        f.write(f"# Domain-Leakage Probes ({args.model}, seed {args.seed}, {args.which} checkpoint)\n\n")
        f.write("Linear probes predicting source dataset id from frozen representations on source val windows.\n")
        f.write("Health close to majority baseline = low leakage; domain repr high = information routed correctly.\n\n")
        f.write("| Target task | Fused z_h | Mean z_h | z_d | Majority |\n|---|---:|---:|---:|---:|\n")
        for r in results:
            f.write(f"| {r['target']} | {r['domain_acc_on_fused_health']} | {r['domain_acc_on_mean_health']} "
                    f"| {r['domain_acc_on_domain_repr']} | {r['majority_baseline']} |\n")
    (root / "outputs/tables/v2_domain_leakage.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
