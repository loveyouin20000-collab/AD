from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from rad.data.cache_schema import SCHEMA_VERSION, CacheManifestError, load_shard


class TeacherCacheDataset(Dataset):
    """Read versioned teacher-output shards via a Parquet index."""

    def __init__(self, cache_dir: Path | str) -> None:
        self.cache_dir = Path(cache_dir)
        meta_path = self.cache_dir / "meta.json"
        index_path = self.cache_dir / "index.parquet"
        if not meta_path.is_file():
            raise FileNotFoundError(f"missing cache meta: {meta_path}")
        if not index_path.is_file():
            raise FileNotFoundError(f"missing cache index: {index_path}")

        self.meta: dict[str, Any] = json.loads(meta_path.read_text())
        if int(self.meta.get("schema_version", -1)) != SCHEMA_VERSION:
            raise CacheManifestError(
                f"schema_version mismatch in meta: got {self.meta.get('schema_version')}, "
                f"expected {SCHEMA_VERSION}"
            )
        table = pq.read_table(index_path)
        self._index = table.to_pylist()
        self._shard_cache: dict[str, list[dict[str, Any]]] = {}

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self._index[idx]
        shard_name = str(row["shard_name"])
        index_in_shard = int(row["index_in_shard"])
        if shard_name not in self._shard_cache:
            # Keep only the active shard resident to bound RAM on large caches.
            self._shard_cache.clear()
            self._shard_cache[shard_name] = load_shard(self.cache_dir / shard_name)
        record = self._shard_cache[shard_name][index_in_shard]
        if record["sample_id"] != row["sample_id"]:
            raise CacheManifestError(
                f"index/sample mismatch: index={row['sample_id']} shard={record['sample_id']}"
            )
        return record

    def get_by_sample_id(self, sample_id: str) -> dict[str, Any]:
        for i, row in enumerate(self._index):
            if row["sample_id"] == sample_id:
                return self[i]
        raise KeyError(sample_id)
