"""Domain-generalization split builders for Paper 1."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SplitResult:
    name: str
    train: list[dict[str, str]]
    val: list[dict[str, str]]
    test: list[dict[str, str]]
    protocol: str
    target: str
    label_column: str
    notes: str = ""


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(rows: list[dict[str, str]], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def filter_label(rows: list[dict[str, str]], label_column: str) -> list[dict[str, str]]:
    return [row for row in rows if row.get(label_column, "exclude") != "exclude"]


def split_source_validation(
    source_rows: list[dict[str, str]],
    *,
    group_column: str,
    label_column: str,
    preferred_group: str | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Create a source-only validation split.

    The validation set is deliberately small and deterministic. It never uses the
    target domain because these are strict target-free DG experiments.
    """

    if not source_rows:
        return [], []

    groups = sorted({row.get(group_column, "") for row in source_rows if row.get(group_column, "")})
    val_group = preferred_group if preferred_group in groups else (groups[0] if groups else "")
    if val_group:
        val = [row for row in source_rows if row.get(group_column, "") == val_group]
        train = [row for row in source_rows if row.get(group_column, "") != val_group]
        if train and val:
            return train, val

    # Fallback: deterministic per-label holdout by recording id.
    by_label: dict[str, list[dict[str, str]]] = {}
    for row in sorted(source_rows, key=lambda r: r["recording_id"]):
        by_label.setdefault(row[label_column], []).append(row)
    train: list[dict[str, str]] = []
    val: list[dict[str, str]] = []
    for _, label_rows in sorted(by_label.items()):
        n_val = max(1, len(label_rows) // 5)
        val.extend(label_rows[:n_val])
        train.extend(label_rows[n_val:])
    return train, val


def add_split_column(rows: list[dict[str, str]], split_name: str) -> list[dict[str, str]]:
    output = []
    for row in rows:
        item = dict(row)
        item["split"] = split_name
        output.append(item)
    return output


def summarize_split(result: SplitResult) -> dict[str, object]:
    def counts(rows: list[dict[str, str]], column: str) -> dict[str, int]:
        return dict(sorted(Counter(row.get(column, "") for row in rows).items()))

    return {
        "name": result.name,
        "protocol": result.protocol,
        "target": result.target,
        "label_column": result.label_column,
        "notes": result.notes,
        "sizes": {
            "train": len(result.train),
            "val": len(result.val),
            "test": len(result.test),
        },
        "labels": {
            "train": counts(result.train, result.label_column),
            "val": counts(result.val, result.label_column),
            "test": counts(result.test, result.label_column),
        },
        "datasets": {
            "train": counts(result.train, "dataset_id"),
            "val": counts(result.val, "dataset_id"),
            "test": counts(result.test, "dataset_id"),
        },
        "conditions": {
            "train": counts(result.train, "condition_id"),
            "val": counts(result.val, "condition_id"),
            "test": counts(result.test, "condition_id"),
        },
        "speed_profiles": {
            "train": counts(result.train, "speed_profile_id"),
            "val": counts(result.val, "speed_profile_id"),
            "test": counts(result.test, "speed_profile_id"),
        },
        "bearing_types": {
            "train": counts(result.train, "bearing_type_id"),
            "val": counts(result.val, "bearing_type_id"),
            "test": counts(result.test, "bearing_type_id"),
        },
    }


def write_split_result(result: SplitResult, output_root: Path, fieldnames: list[str]) -> dict[str, object]:
    split_dir = output_root / result.protocol / result.name
    split_fieldnames = list(fieldnames)
    if "split" not in split_fieldnames:
        split_fieldnames.append("split")
    write_rows(add_split_column(result.train, "train"), split_dir / "train.csv", split_fieldnames)
    write_rows(add_split_column(result.val, "val"), split_dir / "val.csv", split_fieldnames)
    write_rows(add_split_column(result.test, "test"), split_dir / "test.csv", split_fieldnames)

    summary = summarize_split(result)
    lines = [
        f"# Split Summary: {result.name}",
        "",
        f"- Protocol: `{result.protocol}`",
        f"- Target: `{result.target}`",
        f"- Label column: `{result.label_column}`",
        f"- Notes: {result.notes}",
        "",
        "## Sizes",
        "",
        f"- Train: `{len(result.train)}`",
        f"- Val: `{len(result.val)}`",
        f"- Test: `{len(result.test)}`",
        "",
        "## Label Counts",
        "",
        f"- Train: `{summary['labels']['train']}`",
        f"- Val: `{summary['labels']['val']}`",
        f"- Test: `{summary['labels']['test']}`",
        "",
        "## Domain Counts",
        "",
        f"- Train datasets: `{summary['datasets']['train']}`",
        f"- Val datasets: `{summary['datasets']['val']}`",
        f"- Test datasets: `{summary['datasets']['test']}`",
        f"- Train conditions: `{summary['conditions']['train']}`",
        f"- Val conditions: `{summary['conditions']['val']}`",
        f"- Test conditions: `{summary['conditions']['test']}`",
        f"- Train speed profiles: `{summary['speed_profiles']['train']}`",
        f"- Val speed profiles: `{summary['speed_profiles']['val']}`",
        f"- Test speed profiles: `{summary['speed_profiles']['test']}`",
        f"- Train bearing types: `{summary['bearing_types']['train']}`",
        f"- Val bearing types: `{summary['bearing_types']['val']}`",
        f"- Test bearing types: `{summary['bearing_types']['test']}`",
        "",
    ]
    (split_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def paderborn_leave_condition(rows: list[dict[str, str]]) -> list[SplitResult]:
    rows = filter_label(rows, "task3_label")
    conditions = sorted({row["condition_id"] for row in rows})
    results = []
    for target in conditions:
        test = [row for row in rows if row["condition_id"] == target]
        source = [row for row in rows if row["condition_id"] != target]
        preferred_val = next((condition for condition in conditions if condition != target), None)
        train, val = split_source_validation(source, group_column="condition_id", label_column="task3_label", preferred_group=preferred_val)
        results.append(
            SplitResult(
                name=f"target_{target}",
                train=train,
                val=val,
                test=test,
                protocol="paderborn_leave_condition",
                target=target,
                label_column="task3_label",
                notes="Paderborn task3 Normal/Inner/Outer; KB compound excluded; validation uses source condition only.",
            )
        )
    return results


def paderborn_leave_bearing(rows: list[dict[str, str]]) -> list[SplitResult]:
    rows = filter_label(rows, "task3_label")
    bearings = sorted({row["bearing_id"] for row in rows})
    results = []
    for target in bearings:
        test = [row for row in rows if row["bearing_id"] == target]
        source = [row for row in rows if row["bearing_id"] != target]
        train, val = split_source_validation(source, group_column="bearing_id", label_column="task3_label")
        results.append(
            SplitResult(
                name=f"target_{target}",
                train=train,
                val=val,
                test=test,
                protocol="paderborn_leave_bearing",
                target=target,
                label_column="task3_label",
                notes="Paderborn task3 leave-one-bearing; validation uses source bearing only.",
            )
        )
    return results


def ottawa_leave_speed(rows: list[dict[str, str]]) -> list[SplitResult]:
    rows = filter_label(rows, "task4_label")
    speeds = sorted({row["speed_profile_id"] for row in rows})
    results = []
    for target in speeds:
        test = [row for row in rows if row["speed_profile_id"] == target]
        source = [row for row in rows if row["speed_profile_id"] != target]
        preferred_val = next((speed for speed in speeds if speed != target), None)
        train, val = split_source_validation(source, group_column="speed_profile_id", label_column="task4_label", preferred_group=preferred_val)
        results.append(
            SplitResult(
                name=f"target_speed_{target}",
                train=train,
                val=val,
                test=test,
                protocol="ottawa_leave_speed",
                target=target,
                label_column="task4_label",
                notes="Ottawa task4 H/I/O/B; compound C excluded; validation uses source speed profile only.",
            )
        )
    return results


def hust_leave_bearing_type(rows: list[dict[str, str]]) -> list[SplitResult]:
    rows = filter_label(rows, "task4_label")
    bearing_types = sorted({row["bearing_type_id"] for row in rows})
    results = []
    for target in bearing_types:
        test = [row for row in rows if row["bearing_type_id"] == target]
        source = [row for row in rows if row["bearing_type_id"] != target]
        preferred_val = next((bearing for bearing in bearing_types if bearing != target), None)
        train, val = split_source_validation(source, group_column="bearing_type_id", label_column="task4_label", preferred_group=preferred_val)
        results.append(
            SplitResult(
                name=f"target_bearing_type_{target}",
                train=train,
                val=val,
                test=test,
                protocol="hust_leave_bearing_type",
                target=target,
                label_column="task4_label",
                notes="HUST task4 N/I/O/B; compound IB/IO/OB excluded; validation uses source bearing type only.",
            )
        )
    return results


def cross_dataset_task3(all_rows: list[dict[str, str]]) -> list[SplitResult]:
    rows = filter_label(all_rows, "task3_label")
    datasets = sorted({row["dataset_id"] for row in rows})
    results = []
    for target in datasets:
        test = [row for row in rows if row["dataset_id"] == target]
        source = [row for row in rows if row["dataset_id"] != target]
        preferred_val = next((dataset for dataset in datasets if dataset != target), None)
        train, val = split_source_validation(source, group_column="dataset_id", label_column="task3_label", preferred_group=preferred_val)
        results.append(
            SplitResult(
                name=f"target_dataset_{target}",
                train=train,
                val=val,
                test=test,
                protocol="cross_dataset_task3",
                target=target,
                label_column="task3_label",
                notes="Cross-dataset shared task3 Normal/Inner/Outer; compounds and rolling excluded.",
            )
        )
    return results


def cross_dataset_task4_ottawa_hust(ottawa_rows: list[dict[str, str]], hust_rows: list[dict[str, str]]) -> list[SplitResult]:
    rows = filter_label(ottawa_rows + hust_rows, "task4_label")
    datasets = sorted({row["dataset_id"] for row in rows})
    results = []
    for target in datasets:
        test = [row for row in rows if row["dataset_id"] == target]
        source = [row for row in rows if row["dataset_id"] != target]
        train, val = split_source_validation(source, group_column="dataset_id", label_column="task4_label")
        results.append(
            SplitResult(
                name=f"target_dataset_{target}",
                train=train,
                val=val,
                test=test,
                protocol="cross_dataset_task4_ottawa_hust",
                target=target,
                label_column="task4_label",
                notes="Cross-dataset task4 uses Ottawa and HUST only because Paderborn lacks rolling-element class.",
            )
        )
    return results

