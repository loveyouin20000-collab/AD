from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    image_path: str
    mask_path: str
    category: str
    label: int


def load_samples_from_meta(root: Path, mode: str = "train") -> list[SampleRecord]:
    meta_path = Path(root) / "meta.json"
    raw: dict[str, Any] = json.loads(meta_path.read_text())
    if mode not in raw:
        raise KeyError(f"mode {mode!r} missing from {meta_path}")
    records: list[SampleRecord] = []
    for _cls_name, items in raw[mode].items():
        for item in items:
            image_path = str(item["img_path"])
            records.append(
                SampleRecord(
                    sample_id=image_path,
                    image_path=image_path,
                    mask_path=str(item.get("mask_path") or ""),
                    category=str(item["cls_name"]),
                    label=int(item["anomaly"]),
                )
            )
    return records


def build_source_split(
    samples: Sequence[SampleRecord],
    calibration_fraction: float = 0.2,
    seed: int = 111,
) -> list[dict[str, Any]]:
    if not 0.0 <= calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be in [0, 1)")

    by_stratum: dict[tuple[str, int], list[SampleRecord]] = defaultdict(list)
    for sample in samples:
        by_stratum[(sample.category, sample.label)].append(sample)

    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for key in sorted(by_stratum.keys()):
        group = list(by_stratum[key])
        rng.shuffle(group)
        n = len(group)
        if n <= 1:
            n_cal = 0
        else:
            n_cal = min(n - 1, max(0, int(round(n * calibration_fraction))))
        for idx, sample in enumerate(group):
            split = "calibration" if idx < n_cal else "train"
            row = asdict(sample)
            row["split"] = split
            rows.append(row)

    rows.sort(key=lambda r: (r["split"], r["category"], r["label"], r["sample_id"]))
    return rows


def write_manifest(path: Path, rows: Sequence[dict[str, Any]], *, force: bool = False) -> None:
    path = Path(path)
    if path.exists() and not force:
        raise FileExistsError(f"manifest already exists: {path} (pass force=True to overwrite)")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
