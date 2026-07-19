"""VisA adapter with metadata normalization (PIL only; no preprocessing)."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from rad.data.adapters.types import EvaluationRecord
from rad.errors import DatasetIntegrityError


def _as_mapping(raw: Any) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise DatasetIntegrityError("VisA meta entry must be a mapping")
    return raw


def _normalize_split_block(meta: Mapping[str, Any], split: str) -> Mapping[str, Any]:
    candidates = [split, split.lower(), split.capitalize(), split.upper()]
    for key in candidates:
        block = meta.get(key)
        if isinstance(block, Mapping):
            return block
    raise DatasetIntegrityError(
        f"VisA meta.json missing split '{split}' (keys={sorted(meta.keys())})"
    )


def _coerce_anomaly(raw: Mapping[str, Any]) -> int:
    if "anomaly" in raw:
        value = raw["anomaly"]
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"0", "normal", "good", "ok"}:
                return 0
            if lowered in {"1", "anomaly", "abnormal", "defect"}:
                return 1
            return int(value)
        return 1 if int(value) != 0 else 0
    if "label" in raw:
        label = str(raw["label"]).strip().lower()
        if label in {"normal", "good", "ok", "0"}:
            return 0
        if label in {"anomaly", "abnormal", "defect", "1"}:
            return 1
        raise DatasetIntegrityError(f"unsupported VisA label: {raw['label']!r}")
    raise DatasetIntegrityError(f"VisA record missing anomaly/label: {raw}")


def _coerce_image_path(raw: Mapping[str, Any]) -> str:
    for key in ("img_path", "image", "img"):
        if key in raw and raw[key]:
            return str(raw[key]).replace("\\", "/").lstrip("./")
    raise DatasetIntegrityError(f"VisA record missing image path: {raw}")


def _coerce_mask_path(raw: Mapping[str, Any], *, is_anomaly: bool) -> str | None:
    value: Any = None
    for key in ("mask_path", "mask"):
        if key in raw:
            value = raw[key]
            break
    if value is None or value == "":
        if is_anomaly:
            raise DatasetIntegrityError(
                f"anomalous VisA sample missing mask path: {raw}"
            )
        return None
    return str(value).replace("\\", "/").lstrip("./")


def _coerce_category(raw: Mapping[str, Any], fallback: str) -> str:
    for key in ("cls_name", "category", "object", "class"):
        if key in raw and raw[key]:
            return str(raw[key])
    return fallback


def _assert_unique_records(records: Sequence[EvaluationRecord]) -> None:
    seen_ids: set[str] = set()
    for record in records:
        if record.sample_id in seen_ids:
            raise DatasetIntegrityError(f"duplicate sample_id: {record.sample_id}")
        seen_ids.add(record.sample_id)


class VisAAdapter:
    """VisA adapter backed by normalized VisualAD-style meta.json."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise DatasetIntegrityError(f"VisA root is not a directory: {self.root}")
        self.meta_path = self.root / "meta.json"
        if not self.meta_path.is_file():
            raise DatasetIntegrityError(f"VisA meta.json not found: {self.meta_path}")

    def records(self, split: str = "test") -> Sequence[EvaluationRecord]:
        meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        if not isinstance(meta, Mapping):
            raise DatasetIntegrityError("VisA meta.json must be a JSON object")

        split_block = _normalize_split_block(meta, split)
        records: list[EvaluationRecord] = []
        for category in sorted(split_block.keys()):
            items = split_block[category]
            if not isinstance(items, list):
                raise DatasetIntegrityError(
                    f"VisA meta split '{split}' category '{category}' must be a list"
                )
            for raw_item in items:
                raw = _as_mapping(raw_item)
                rel_image = _coerce_image_path(raw)
                image_label = _coerce_anomaly(raw)
                rel_mask = _coerce_mask_path(raw, is_anomaly=image_label == 1)
                image_path = self.root / rel_image
                if not image_path.is_file():
                    raise DatasetIntegrityError(f"missing image: {image_path}")
                mask_path: Path | None
                if rel_mask is None:
                    mask_path = None
                else:
                    mask_path = self.root / rel_mask
                    if not mask_path.is_file():
                        raise DatasetIntegrityError(f"missing mask: {mask_path}")
                records.append(
                    EvaluationRecord(
                        sample_id=rel_image,
                        dataset="visa",
                        category=_coerce_category(raw, fallback=str(category)),
                        image_path=image_path,
                        mask_path=mask_path,
                        image_label=image_label,
                        split=split.lower(),
                    )
                )

        records.sort(key=lambda r: r.sample_id)
        _assert_unique_records(records)
        return tuple(records)

    def open_image(self, record: EvaluationRecord) -> Image.Image:
        if not record.image_path.is_file():
            raise DatasetIntegrityError(f"missing image: {record.image_path}")
        return Image.open(record.image_path).convert("RGB")

    def open_mask(self, record: EvaluationRecord) -> Image.Image | None:
        if record.image_label == 0 or record.mask_path is None:
            return None
        if not record.mask_path.is_file():
            raise DatasetIntegrityError(f"missing mask: {record.mask_path}")
        return Image.open(record.mask_path).convert("L")
