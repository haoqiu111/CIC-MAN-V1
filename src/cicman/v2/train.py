"""Training/evaluation loop for CIC-MAN v2 and its baseline family."""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

_WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

from recording_protocol import OFFICIAL_AGGREGATION, RecordingAccumulator


def set_seed(seed: int):
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if os.environ.get("CICMAN_FAST_CUDNN", "0") == "1":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    f1s = []
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1s))


def evaluate(model, loader, device, num_classes: int, views: list[str], recording_aggregation: str = OFFICIAL_AGGREGATION):
    import torch

    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    per_dataset_cm: dict[str, np.ndarray] = defaultdict(lambda: np.zeros((num_classes, num_classes), dtype=np.int64))
    recording = RecordingAccumulator(num_classes)
    router_sum = None
    n = 0

    with torch.no_grad():
        for batch in loader:
            x = {v: batch["views"][v].to(device, non_blocking=True) for v in views}
            feats = batch["feats"].to(device, non_blocking=True)
            y = batch["y"].numpy()
            out = model(x, feats)
            probs = torch.softmax(out["logits"], dim=1).cpu().numpy()
            pred = probs.argmax(1)
            for t, p in zip(y, pred):
                cm[t, p] += 1
            w = out["router_weights"].cpu().numpy()
            router_sum = w.sum(0) if router_sum is None else router_sum + w.sum(0)
            n += len(y)
            ds = loader.dataset
            for j, idx in enumerate(batch["index"].numpy()):
                row = ds.rows[idx]
                rid = f"{row['dataset_id']}::{row['recording_id']}"
                recording.add(rid, int(row["label"]), probs[j])
                per_dataset_cm[row["dataset_id"]][y[j], pred[j]] += 1

    rec_metrics = recording.metrics(recording_aggregation)
    rec_cm = np.asarray(rec_metrics["recording_confusion_matrix"], dtype=np.int64)

    by_dataset = {
        name: {
            "macro_f1": macro_f1_from_confusion(m),
            "accuracy": float(np.trace(m) / max(m.sum(), 1)),
            "num_windows": int(m.sum()),
        }
        for name, m in per_dataset_cm.items()
    }
    return {
        "window_accuracy": float(np.trace(cm) / max(cm.sum(), 1)),
        "window_macro_f1": macro_f1_from_confusion(cm),
        "recording_accuracy": float(np.trace(rec_cm) / max(rec_cm.sum(), 1)),
        "recording_macro_f1": macro_f1_from_confusion(rec_cm),
        "recording_aggregation": recording_aggregation,
        "recording_vote_ties": int(rec_metrics["vote_ties"]),
        "recording_probability_tiebreaks": int(rec_metrics["probability_tiebreaks"]),
        "num_windows": int(cm.sum()),
        "num_recordings": int(rec_cm.sum()),
        "confusion_matrix": cm.tolist(),
        "recording_confusion_matrix": rec_cm.tolist(),
        "by_dataset": by_dataset,
        "worst_dataset_macro_f1": min((v["macro_f1"] for v in by_dataset.values()), default=0.0),
        "mean_router_weights": (router_sum / max(n, 1)).tolist() if router_sum is not None else None,
    }


def train_model(
    *,
    model,
    train_dataset,
    val_dataset,
    test_dataset,
    output_dir: Path,
    views: list[str],
    num_classes: int,
    epochs: int = 40,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cuda",
    num_workers: int = 0,
    seed: int = 42,
    lambda_view_cls: float = 0.3,
    lambda_consensus: float = 0.1,
    lambda_adversarial: float = 0.1,
    lambda_domain_cls: float = 0.1,
    lambda_orthogonal: float = 0.05,
    lambda_balance: float = 0.01,
    lambda_route_prior: float = 0.0,
    focal_gamma: float = 0.0,
    dg_method: str | None = None,  # coral | mmd | irm | groupdro | ccn
    lambda_dg: float = 1.0,
    label_smoothing: float = 0.05,
    balanced_sampling: bool = True,
    recording_aggregation: str = OFFICIAL_AGGREGATION,
    log_fn=print,
):
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    set_seed(seed)
    torch.set_num_threads(2)  # bound CPU-side memory/thread overhead on the shared workstation
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    model = model.to(device)

    if balanced_sampling:
        weights = train_dataset.balanced_sample_weights()
        sampler = WeightedRandomSampler(torch.as_tensor(weights), num_samples=len(train_dataset), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=num_workers, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=num_workers, pin_memory=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    ce = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    ce_plain = torch.nn.CrossEntropyLoss()

    def coral_loss(feats_by_domain):
        losses = []
        for i in range(len(feats_by_domain)):
            for j in range(i + 1, len(feats_by_domain)):
                a, b = feats_by_domain[i], feats_by_domain[j]
                if len(a) < 2 or len(b) < 2:
                    continue
                ca = (a - a.mean(0)).T @ (a - a.mean(0)) / (len(a) - 1)
                cb = (b - b.mean(0)).T @ (b - b.mean(0)) / (len(b) - 1)
                losses.append(((ca - cb) ** 2).sum() / (4 * a.size(1) ** 2))
        return sum(losses) / max(len(losses), 1) if losses else torch.zeros((), device=device)

    def mmd_loss(feats_by_domain):
        def rbf(x, y, sigmas=(1.0, 5.0, 10.0)):
            d = torch.cdist(x, y) ** 2
            return sum(torch.exp(-d / (2 * s**2)) for s in sigmas).mean()

        losses = []
        for i in range(len(feats_by_domain)):
            for j in range(i + 1, len(feats_by_domain)):
                a, b = feats_by_domain[i], feats_by_domain[j]
                if len(a) < 2 or len(b) < 2:
                    continue
                losses.append(rbf(a, a) + rbf(b, b) - 2 * rbf(a, b))
        return sum(losses) / max(len(losses), 1) if losses else torch.zeros((), device=device)

    def irm_penalty(logits, y):
        scale = torch.ones((), device=device, requires_grad=True)
        loss = ce_plain(logits * scale, y)
        grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
        return grad**2

    def ccn_loss(fused_h, logits, labels, domains):
        """Li et al. CCN objective on the controlled fused-health backbone.

        Causal consistency uses 1-|cos(z, class prototype)|, matching Eq. 13.
        Collaborative training uses the variance of per-domain CE, matching
        Eq. 17.  Published weights lambda1=10 and lambda2=1 are retained.
        """
        con_terms = []
        for cls in torch.unique(labels):
            mask = labels == cls
            if mask.sum() < 2:
                continue
            z = fused_h[mask]
            proto = z.mean(0, keepdim=True)
            cos = torch.nn.functional.cosine_similarity(z, proto.expand_as(z), dim=1)
            con_terms.append((1.0 - cos.abs()).mean())
        loss_con = (sum(con_terms) / len(con_terms)) if con_terms else torch.zeros((), device=device)

        domain_losses = []
        for domain in torch.unique(domains):
            mask = domains == domain
            if mask.any():
                domain_losses.append(ce_plain(logits[mask], labels[mask]))
        if len(domain_losses) > 1:
            stacked = torch.stack(domain_losses)
            loss_col = ((stacked - stacked.mean()) ** 2).mean()
        else:
            loss_col = torch.zeros((), device=device)
        return 10.0 * loss_con + loss_col

    groupdro_weights = None

    best = {"metric": -1.0, "epoch": -1}
    history = []
    total_steps = epochs * max(len(train_loader), 1)
    step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        agg = defaultdict(float)
        n_batches = 0
        for batch in train_loader:
            step += 1
            p = step / total_steps
            adv_alpha = (2.0 / (1.0 + np.exp(-10 * p)) - 1.0) if lambda_adversarial > 0 else 0.0

            x = {v: batch["views"][v].to(device, non_blocking=True) for v in views}
            feats = batch["feats"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            dom = batch["domain"].to(device, non_blocking=True)

            out = model(x, feats, adv_alpha=adv_alpha)
            k = out["logits_per_view"].size(1)

            if focal_gamma > 0:
                # hard-sample adaptive reweighting (R1 revision): down-weight
                # easy samples so the gradient budget concentrates on hard ones
                ce_i = torch.nn.functional.cross_entropy(
                    out["logits"], y, reduction="none", label_smoothing=label_smoothing)
                with torch.no_grad():
                    p_true = torch.softmax(out["logits"], dim=1).gather(1, y.unsqueeze(1)).squeeze(1)
                    w = (1 - p_true).pow(focal_gamma)
                    w = w / w.mean().clamp_min(1e-6)
                loss_cls = (w * ce_i).mean()
            else:
                loss_cls = ce(out["logits"], y)
            loss_view = sum(ce(out["logits_per_view"][:, i], y) for i in range(k)) / k

            z_h = out["z_h"]
            w_detached = out["router_weights"].detach()
            center = (z_h * w_detached.unsqueeze(-1)).sum(1, keepdim=True).detach()
            # blueprint eq. 6.2: router-weighted consensus, so unreliable views
            # are not forced onto the health consensus center
            loss_cons = (w_detached * ((z_h - center) ** 2).sum(-1)).sum(1).mean()

            loss_adv = torch.zeros((), device=device)
            if out["adv_logits"] is not None:
                adv_target = dom if out["adv_logits"].size(0) == dom.size(0) else dom.repeat_interleave(k)
                loss_adv = ce_plain(out["adv_logits"], adv_target)
            # domain ids can exceed the head size when the model is built with
            # fewer domains than the sampler sees (e.g. probe models); only
            # evaluate the CE when the loss actually contributes
            if lambda_domain_cls > 0:
                loss_dom = ce_plain(out["domain_logits"], dom.repeat_interleave(k))
            else:
                loss_dom = torch.zeros((), device=device)

            zh_n = torch.nn.functional.normalize(z_h, dim=-1)
            zd_n = torch.nn.functional.normalize(out["z_d"], dim=-1)
            dmin = min(zh_n.size(-1), zd_n.size(-1))
            loss_orth = (zh_n[..., :dmin] * zd_n[..., :dmin]).sum(-1).pow(2).mean()

            mean_w = out["router_weights"].mean(0)
            loss_balance = (mean_w * torch.log(mean_w.clamp_min(1e-9) * k)).sum()

            loss_prior = torch.zeros((), device=device)
            if lambda_route_prior > 0 and getattr(model, "router_log_prior", None) is not None:
                w = out["router_weights"]
                loss_prior = (w * (torch.log(w.clamp_min(1e-9)) - model.router_log_prior)).sum(1).mean()

            loss_dg = torch.zeros((), device=device)
            if dg_method:
                fused_h = (out["z_h"] * out["router_weights"].unsqueeze(-1)).sum(1)
                if dg_method in ("coral", "mmd"):
                    groups = [fused_h[dom == d] for d in torch.unique(dom)]
                    loss_dg = coral_loss(groups) if dg_method == "coral" else mmd_loss(groups)
                elif dg_method == "irm":
                    penalties = [irm_penalty(out["logits"][dom == d], y[dom == d]) for d in torch.unique(dom) if (dom == d).sum() > 1]
                    loss_dg = sum(penalties) / max(len(penalties), 1) if penalties else loss_dg
                elif dg_method == "groupdro":
                    n_dom = int(dom.max().item()) + 1
                    if groupdro_weights is None or len(groupdro_weights) < n_dom:
                        groupdro_weights = torch.ones(n_dom, device=device) / n_dom
                    group_losses = torch.stack([
                        ce_plain(out["logits"][dom == d], y[dom == d]) if (dom == d).any() else torch.zeros((), device=device)
                        for d in range(n_dom)
                    ])
                    groupdro_weights = groupdro_weights * torch.exp(0.01 * group_losses.detach())
                    groupdro_weights = groupdro_weights / groupdro_weights.sum()
                    loss_dg = (groupdro_weights * group_losses).sum() - loss_cls  # replaces plain CE emphasis
                elif dg_method == "ccn":
                    loss_dg = ccn_loss(fused_h, out["logits"], y, dom)

            loss = (
                loss_cls
                + lambda_dg * loss_dg
                + lambda_view_cls * loss_view
                + lambda_consensus * loss_cons
                + lambda_adversarial * loss_adv
                + lambda_domain_cls * loss_dom
                + lambda_orthogonal * loss_orth
                + lambda_balance * loss_balance
                + lambda_route_prior * loss_prior
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            agg["loss"] += float(loss.detach())
            agg["cls"] += float(loss_cls.detach())
            agg["cons"] += float(loss_cons.detach())
            agg["adv"] += float(loss_adv.detach())
            n_batches += 1
        scheduler.step()

        import gc

        gc.collect()
        val_metrics = evaluate(model, val_loader, device, num_classes, views, recording_aggregation)
        sel = val_metrics["worst_dataset_macro_f1"] if len(val_metrics["by_dataset"]) > 1 else val_metrics["window_macro_f1"]
        history.append(
            {
                "epoch": epoch,
                "train": {key: value / max(n_batches, 1) for key, value in agg.items()},
                "val": {key: val_metrics[key] for key in ("window_accuracy", "window_macro_f1", "worst_dataset_macro_f1")},
                "selection_metric": sel,
                "seconds": time.time() - t0,
            }
        )
        log_fn(
            f"epoch {epoch:3d}/{epochs} loss={agg['loss']/max(n_batches,1):.4f} "
            f"val_f1={val_metrics['window_macro_f1']:.4f} worst_src={val_metrics['worst_dataset_macro_f1']:.4f} "
            f"({time.time()-t0:.0f}s)"
        )
        if sel > best["metric"]:
            best = {"metric": sel, "epoch": epoch}
            torch.save({"model": model.state_dict(), "epoch": epoch, "views": views}, output_dir / "best.pt")

    # Last-epoch checkpoint used by the strict reporting protocol.
    torch.save({"model": model.state_dict(), "epoch": epochs, "views": views}, output_dir / "last.pt")
    test_metrics_last = evaluate(model, test_loader, device, num_classes, views, recording_aggregation)

    ckpt = torch.load(output_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    test_metrics = evaluate(model, test_loader, device, num_classes, views, recording_aggregation)
    val_metrics = evaluate(model, val_loader, device, num_classes, views, recording_aggregation)

    result = {
        "best_epoch": best["epoch"],
        "selection_metric": best["metric"],
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "test_metrics_last": test_metrics_last,
        "history": history,
        "views": views,
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
