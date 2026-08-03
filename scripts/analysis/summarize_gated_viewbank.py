#!/usr/bin/env python3
"""Summarize conservative gated view-bank CIC-MAN runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def target_from_name(path: Path) -> str:
    for prefix in [
        "cic_man_gated_viewbank_conservative_source_mixed_cross_dataset_task3_target_",
        "cic_man_gated_viewbank_v2_gate_source_mixed_cross_dataset_task3_target_",
        "cic_man_gated_viewbank_disentangle_source_mixed_cross_dataset_task3_target_",
        "cic_man_gated_viewbank_gate_tuned_source_mixed_cross_dataset_task3_target_",
        "cic_man_gated_viewbank_orthogonal_source_mixed_cross_dataset_task3_target_",
        "cic_man_gated_viewbank_orthogonal_aux_source_mixed_cross_dataset_task3_target_",
        "cic_man_gated_viewbank_health_gate_source_mixed_cross_dataset_task3_target_",
    ]:
        if path.name.startswith(prefix):
            return path.name.removeprefix(prefix)
    return path.name


def rows(root: Path) -> list[dict[str, object]]:
    out = []
    patterns = [
        "cic_man_gated_viewbank_conservative_source_mixed_cross_dataset_task3_target_*",
        "cic_man_gated_viewbank_v2_gate_source_mixed_cross_dataset_task3_target_*",
        "cic_man_gated_viewbank_disentangle_source_mixed_cross_dataset_task3_target_*",
        "cic_man_gated_viewbank_gate_tuned_source_mixed_cross_dataset_task3_target_*",
        "cic_man_gated_viewbank_orthogonal_source_mixed_cross_dataset_task3_target_*",
        "cic_man_gated_viewbank_orthogonal_aux_source_mixed_cross_dataset_task3_target_*",
        "cic_man_gated_viewbank_health_gate_source_mixed_cross_dataset_task3_target_*",
    ]
    checkpoint_dirs = []
    for pattern in patterns:
        checkpoint_dirs.extend(sorted(root.glob(pattern)))
    for checkpoint_dir in checkpoint_dirs:
        metrics_path = checkpoint_dir / "metrics.json"
        recording_path = checkpoint_dir / "recording_metrics.json"
        diagnostics_path = checkpoint_dir / "diagnostics" / "cic_man_diagnostics_test.json"
        if not metrics_path.exists():
            continue
        metrics = load_json(metrics_path)
        test = metrics["test_metrics"]
        recording = load_json(recording_path)["mean_logits"] if recording_path.exists() else {}
        diagnostics = load_json(diagnostics_path) if diagnostics_path.exists() else {}
        router = diagnostics.get("router", {})
        gate_weights = router.get("mean_weights", [])[4:]
        out.append(
            {
                "target": target_from_name(checkpoint_dir),
                "variant": (
                    "disentangle"
                    if "gated_viewbank_disentangle" in checkpoint_dir.name
                    else (
                        "health_gate"
                        if "gated_viewbank_health_gate" in checkpoint_dir.name
                        else (
                            "orthogonal_aux"
                            if "gated_viewbank_orthogonal_aux" in checkpoint_dir.name
                            else (
                                "orthogonal"
                                if "gated_viewbank_orthogonal" in checkpoint_dir.name
                                else (
                                    "gate_tuned"
                                    if "gated_viewbank_gate_tuned" in checkpoint_dir.name
                                    else (
                                        "v2_gate"
                                        if "gated_viewbank_v2_gate" in checkpoint_dir.name
                                        else "conservative"
                                    )
                                )
                            )
                        )
                    )
                ),
                "best_epoch": metrics["best_epoch"],
                "window_accuracy": test["accuracy"],
                "window_macro_f1": test["macro_f1"],
                "recording_accuracy": recording.get("accuracy", ""),
                "recording_macro_f1": recording.get("macro_f1", ""),
                "mean_optional_gate_mass": sum(float(value) for value in gate_weights),
                "effective_agents": router.get("effective_agents_from_mean_weights", ""),
                "gate_target_rate_last_epoch": metrics["history"][-1]["train"].get("gate_target_rate", 0.0),
                "gate_mean_last_epoch": metrics["history"][-1]["train"].get("filterbank_gate_mean", 0.0),
                "view_bank_views": ";".join(metrics["config"].get("view_bank_views", [])),
            }
        )
    return out


def write_csv(items: list[dict[str, object]], path: Path) -> None:
    if not items:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(items[0].keys()))
        writer.writeheader()
        writer.writerows(items)


def fmt(value: object) -> str:
    if value == "":
        return ""
    return f"{float(value):.6f}"


def write_markdown(items: list[dict[str, object]], path: Path) -> None:
    lines = [
        "# Conservative Gated View-Bank CIC-MAN Summary",
        "",
        "| Variant | Target | Views | Window F1 | Recording F1 | Optional Gate Mass | Gate Target Rate | Effective Agents |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in items:
        lines.append(
            f"| {row['variant']} | {row['target']} | {row['view_bank_views']} | {fmt(row['window_macro_f1'])} | "
            f"{fmt(row['recording_macro_f1'])} | {fmt(row['mean_optional_gate_mass'])} | "
            f"{fmt(row['gate_target_rate_last_epoch'])} | {fmt(row['effective_agents'])} |"
        )
    for variant in sorted({str(row["variant"]) for row in items}):
        variant_rows = [row for row in items if row["variant"] == variant]
        lines.extend(
            [
                "",
                f"{variant} mean window Macro-F1: `{sum(float(row['window_macro_f1']) for row in variant_rows) / len(variant_rows):.6f}`",
                f"{variant} mean recording Macro-F1: `{sum(float(row['recording_macro_f1']) for row in variant_rows) / len(variant_rows):.6f}`",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    items = rows(args.checkpoint_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(items, args.output_dir / "gated_viewbank_conservative_summary.csv")
    write_markdown(items, args.output_dir / "gated_viewbank_conservative_summary.md")
    print(f"Wrote {len(items)} rows to {args.output_dir / 'gated_viewbank_conservative_summary.md'}")


if __name__ == "__main__":
    main()
