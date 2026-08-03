"""Recording-level manifest builders for the Paper 1 datasets."""

from __future__ import annotations

import csv
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


COMMON_COLUMNS = [
    "dataset_id",
    "source_file",
    "archive_file",
    "recording_id",
    "fault_code",
    "fault_label",
    "fault_label_id",
    "task3_label",
    "task3_label_id",
    "task4_label",
    "task4_label_id",
    "is_compound",
    "bearing_id",
    "bearing_type_id",
    "condition_id",
    "speed_profile_id",
    "trial_id",
    "measurement_id",
    "sensor_modality",
    "signal_key",
    "speed_key",
    "sampling_rate",
    "duration_sec",
    "num_samples",
    "rotation_speed_rpm",
    "load_torque_nm",
    "radial_force_n",
    "notes",
]


LABEL_IDS = {
    "normal": 0,
    "inner": 1,
    "outer": 2,
    "rolling": 3,
    "compound": 4,
    "unknown": -1,
}

TASK3_IDS = {
    "normal": 0,
    "inner": 1,
    "outer": 2,
    "exclude": -1,
}

TASK4_IDS = {
    "normal": 0,
    "inner": 1,
    "outer": 2,
    "rolling": 3,
    "exclude": -1,
}


PADERBORN_CONDITIONS = {
    "N15_M07_F10": {"rpm": 1500, "torque": 0.7, "force": 1000},
    "N09_M07_F10": {"rpm": 900, "torque": 0.7, "force": 1000},
    "N15_M01_F10": {"rpm": 1500, "torque": 0.1, "force": 1000},
    "N15_M07_F04": {"rpm": 1500, "torque": 0.7, "force": 400},
}

PADERBORN_HEALTHY = {"K001", "K002", "K003", "K004", "K005", "K006"}
PADERBORN_OUTER = {
    "KA01",
    "KA03",
    "KA04",
    "KA05",
    "KA06",
    "KA07",
    "KA08",
    "KA09",
    "KA15",
    "KA16",
    "KA22",
    "KA30",
}
PADERBORN_INNER = {
    "KI01",
    "KI03",
    "KI04",
    "KI05",
    "KI07",
    "KI08",
    "KI14",
    "KI16",
    "KI17",
    "KI18",
    "KI21",
}
PADERBORN_COMPOUND = {"KB23", "KB24", "KB27"}
PADERBORN_UNREADABLE_RECORDINGS = {
    "N15_M01_F10_KA08_2",
}

OTTAWA_LABELS = {
    "H": ("normal", "normal"),
    "I": ("inner", "inner"),
    "O": ("outer", "outer"),
    "B": ("rolling", "rolling"),
    "C": ("compound", "compound"),
}

HUST_LABELS = {
    "N": ("normal", "normal"),
    "I": ("inner", "inner"),
    "O": ("outer", "outer"),
    "B": ("rolling", "rolling"),
    "IB": ("compound", "inner_ball"),
    "IO": ("compound", "inner_outer"),
    "OB": ("compound", "outer_ball"),
}


@dataclass
class ManifestRow:
    dataset_id: str
    source_file: str
    archive_file: str = ""
    recording_id: str = ""
    fault_code: str = ""
    fault_label: str = "unknown"
    fault_label_id: int = -1
    task3_label: str = "exclude"
    task3_label_id: int = -1
    task4_label: str = "exclude"
    task4_label_id: int = -1
    is_compound: int = 0
    bearing_id: str = ""
    bearing_type_id: str = ""
    condition_id: str = ""
    speed_profile_id: str = ""
    trial_id: str = ""
    measurement_id: str = ""
    sensor_modality: str = "vibration"
    signal_key: str = ""
    speed_key: str = ""
    sampling_rate: int = 0
    duration_sec: float = 0.0
    num_samples: int = 0
    rotation_speed_rpm: str = ""
    load_torque_nm: str = ""
    radial_force_n: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        return {key: data.get(key, "") for key in COMMON_COLUMNS}


def label_fields(label: str, *, supports_task3: bool = True, supports_task4: bool = True) -> dict[str, object]:
    is_compound = int(label == "compound")

    if label in {"normal", "inner", "outer"} and supports_task3:
        task3_label = label
    else:
        task3_label = "exclude"

    if label in {"normal", "inner", "outer", "rolling"} and supports_task4:
        task4_label = label
    else:
        task4_label = "exclude"

    return {
        "fault_label": label,
        "fault_label_id": LABEL_IDS.get(label, -1),
        "task3_label": task3_label,
        "task3_label_id": TASK3_IDS.get(task3_label, -1),
        "task4_label": task4_label,
        "task4_label_id": TASK4_IDS.get(task4_label, -1),
        "is_compound": is_compound,
    }


def write_manifest(rows: list[ManifestRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COMMON_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())


def summarize_rows(rows: Iterable[ManifestRow]) -> dict[str, object]:
    rows = list(rows)
    by_label = Counter(row.fault_label for row in rows)
    by_task3 = Counter(row.task3_label for row in rows)
    by_task4 = Counter(row.task4_label for row in rows)
    by_condition = Counter(row.condition_id for row in rows if row.condition_id)
    by_speed = Counter(row.speed_profile_id for row in rows if row.speed_profile_id)
    by_bearing_type = Counter(row.bearing_type_id for row in rows if row.bearing_type_id)
    by_bearing = Counter(row.bearing_id for row in rows if row.bearing_id)
    return {
        "rows": len(rows),
        "fault_label_counts": dict(sorted(by_label.items())),
        "task3_label_counts": dict(sorted(by_task3.items())),
        "task4_label_counts": dict(sorted(by_task4.items())),
        "condition_counts": dict(sorted(by_condition.items())),
        "speed_profile_counts": dict(sorted(by_speed.items())),
        "bearing_type_counts": dict(sorted(by_bearing_type.items())),
        "bearing_count": len(by_bearing),
        "sampling_rates": sorted({row.sampling_rate for row in rows if row.sampling_rate}),
    }


def build_paderborn_manifest(raw_root: Path) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    rar_files = sorted(raw_root.glob("*.rar"))
    pattern = re.compile(r"(?P<condition>N\d+_M\d+_F\d+)_(?P<bearing>[A-Z0-9]+)_(?P<measurement>\d+)\.mat$")

    for rar_path in rar_files:
        try:
            archive_listing = list_archive_members(rar_path)
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            raise RuntimeError(f"Could not list Paderborn archive {rar_path}") from exc

        for item in sorted(archive_listing.splitlines()):
            if not item.endswith(".mat"):
                continue
            match = pattern.search(Path(item).name)
            if not match:
                continue

            bearing_id = match.group("bearing")
            condition_id = match.group("condition")
            measurement_id = match.group("measurement")
            recording_id = Path(item).stem
            if recording_id in PADERBORN_UNREADABLE_RECORDINGS:
                continue
            if bearing_id in PADERBORN_HEALTHY:
                label = "normal"
                damage_kind = "healthy"
            elif bearing_id in PADERBORN_OUTER:
                label = "outer"
                damage_kind = "outer"
            elif bearing_id in PADERBORN_INNER:
                label = "inner"
                damage_kind = "inner"
            elif bearing_id in PADERBORN_COMPOUND:
                label = "compound"
                damage_kind = "multiple"
            else:
                label = "unknown"
                damage_kind = "unknown"

            condition = PADERBORN_CONDITIONS.get(condition_id, {})
            fields = label_fields(label, supports_task3=True, supports_task4=True)
            row = ManifestRow(
                dataset_id="paderborn",
                archive_file=str(rar_path),
                source_file=item,
                recording_id=recording_id,
                fault_code=bearing_id,
                bearing_id=bearing_id,
                condition_id=condition_id,
                measurement_id=measurement_id,
                sensor_modality="vibration",
                signal_key="vibration_1",
                speed_key="speed",
                sampling_rate=64000,
                duration_sec=4.0,
                num_samples=256000,
                rotation_speed_rpm=str(condition.get("rpm", "")),
                load_torque_nm=str(condition.get("torque", "")),
                radial_force_n=str(condition.get("force", "")),
                notes=f"damage_kind={damage_kind}; source_is_rar_member; paderborn_has_no_dedicated_rolling_class",
                **fields,
            )
            rows.append(row)

    return sorted(rows, key=lambda r: (r.bearing_id, r.condition_id, int(r.measurement_id or 0)))


def list_archive_members(archive_path: Path) -> str:
    if shutil.which("bsdtar"):
        return subprocess.check_output(["bsdtar", "-tf", str(archive_path)], text=True)
    if shutil.which("tar"):
        return subprocess.check_output(["tar", "-tf", str(archive_path)], text=True)
    if shutil.which("7z"):
        output = subprocess.check_output(["7z", "l", "-ba", str(archive_path)], text=True)
        members = []
        for line in output.splitlines():
            parts = line.split()
            if len(parts) >= 6:
                members.append(parts[-1])
        return "\n".join(members)
    raise FileNotFoundError("Listing Paderborn .rar files requires bsdtar, tar, or 7z on PATH.")


def build_ottawa_manifest(raw_root: Path) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    pattern = re.compile(r"(?P<fault>[A-Z])-(?P<speed>[A-Z])-(?P<trial>\d+)\.mat$")

    for mat_path in sorted(raw_root.rglob("*.mat")):
        match = pattern.match(mat_path.name)
        if not match:
            continue
        fault_code = match.group("fault")
        speed_profile = match.group("speed")
        trial_id = match.group("trial")
        label, subtype = OTTAWA_LABELS.get(fault_code, ("unknown", "unknown"))
        fields = label_fields(label, supports_task3=True, supports_task4=True)
        row = ManifestRow(
            dataset_id="ottawa",
            source_file=str(mat_path),
            recording_id=mat_path.stem,
            fault_code=fault_code,
            speed_profile_id=speed_profile,
            trial_id=trial_id,
            sensor_modality="vibration",
            signal_key="Channel_1",
            speed_key="Channel_2",
            sampling_rate=200000,
            duration_sec=10.0,
            num_samples=2_000_000,
            notes=f"fault_subtype={subtype}; Channel_2_is_encoder_or_speed_pulse",
            **fields,
        )
        rows.append(row)

    return sorted(rows, key=lambda r: (r.fault_code, r.speed_profile_id, int(r.trial_id or 0)))


def build_hust_manifest(raw_root: Path) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    dataset_dir = raw_root / "HUST bearing dataset"
    pattern = re.compile(r"(?P<fault>IB|IO|OB|N|I|O|B)(?P<suffix>\d{3})\.mat$")

    for mat_path in sorted(dataset_dir.glob("*.mat")):
        match = pattern.match(mat_path.name)
        if not match:
            continue
        fault_code = match.group("fault")
        suffix = match.group("suffix")
        bearing_type_id = suffix[0]
        condition_id = suffix[1:]
        label, subtype = HUST_LABELS.get(fault_code, ("unknown", "unknown"))
        fields = label_fields(label, supports_task3=True, supports_task4=True)
        row = ManifestRow(
            dataset_id="hust",
            source_file=str(mat_path),
            recording_id=mat_path.stem,
            fault_code=fault_code,
            bearing_type_id=bearing_type_id,
            condition_id=condition_id,
            sensor_modality="vibration",
            signal_key="data",
            speed_key="fs",
            sampling_rate=51200,
            duration_sec=0.0,
            num_samples=0,
            notes=f"fault_subtype={subtype}; fs_field_is_rotation_frequency_proxy_not_sampling_rate; length_checked_later",
            **fields,
        )
        rows.append(row)

    return sorted(rows, key=lambda r: (r.fault_code, r.bearing_type_id, r.condition_id))


def write_summary_markdown(summary: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Manifest Summary", ""]
    for dataset, stats in summary.items():
        lines.extend([f"## {dataset}", ""])
        lines.append(f"- Rows: `{stats['rows']}`")
        lines.append(f"- Sampling rates: `{stats['sampling_rates']}`")
        lines.append(f"- Bearing count: `{stats['bearing_count']}`")
        for key in [
            "fault_label_counts",
            "task3_label_counts",
            "task4_label_counts",
            "condition_counts",
            "speed_profile_counts",
            "bearing_type_counts",
        ]:
            values = stats.get(key, {})
            if values:
                lines.append(f"- {key}: `{values}`")
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")
