"""MVTec AD filesystem adapter (layout + PIL only; no preprocessing)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from PIL import Image

from rad.data.adapters.types import EvaluationRecord
from rad.errors import DatasetIntegrityError

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
_NORMAL_SPECIES = frozenset({"good"})


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES


def _assert_unique_records(records: Sequence[EvaluationRecord]) -> None:
    seen_ids: set[str] = set()
    seen_resolved: dict[Path, str] = {}
    for record in records:
        if record.sample_id in seen_ids:
            raise DatasetIntegrityError(f"duplicate sample_id: {record.sample_id}")
        seen_ids.add(record.sample_id)
        resolved = record.image_path.resolve()
        if resolved in seen_resolved:
            raise DatasetIntegrityError(
                f"duplicate sample path for {record.sample_id} "
                f"(also {seen_resolved[resolved]})"
            )
        seen_resolved[resolved] = record.sample_id


def _resolve_mvtec_mask(category_dir: Path, specie: str, image_path: Path) -> Path:
    gt_dir = category_dir / "ground_truth" / specie
    stem = image_path.stem
    candidates = [
        gt_dir / f"{stem}_mask{image_path.suffix}",
        gt_dir / f"{stem}_mask.png",
        gt_dir / image_path.name,
        gt_dir / f"{stem}.png",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise DatasetIntegrityError(
        f"missing anomalous mask for image {image_path} under {gt_dir}"
    )


class MVTecAdapter:
    """Deterministic MVTec AD adapter over the classic filesystem layout."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise DatasetIntegrityError(f"MVTec root is not a directory: {self.root}")

    def records(
        self,
        split: str = "test",
        *,
        categories: Sequence[str] | None = None,
    ) -> Sequence[EvaluationRecord]:
        records: list[EvaluationRecord] = []
        selected_categories = set(categories) if categories is not None else None
        for category_dir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            if (
                selected_categories is not None
                and category_dir.name not in selected_categories
            ):
                continue
            split_dir = category_dir / split
            if not split_dir.is_dir():
                continue
            category = category_dir.name
            for specie_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
                specie = specie_dir.name
                is_anomaly = specie not in _NORMAL_SPECIES
                for image_path in sorted(p for p in specie_dir.iterdir() if _is_image(p)):
                    if not image_path.is_file():
                        raise DatasetIntegrityError(f"missing image: {image_path}")
                    rel = image_path.relative_to(self.root).as_posix()
                    if is_anomaly:
                        mask_path = _resolve_mvtec_mask(category_dir, specie, image_path)
                    else:
                        mask_path = None
                    records.append(
                        EvaluationRecord(
                            sample_id=rel,
                            dataset="mvtec",
                            category=category,
                            image_path=image_path,
                            mask_path=mask_path,
                            image_label=1 if is_anomaly else 0,
                            split=split,
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
