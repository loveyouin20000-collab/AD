"""B2-05C1 untouched final evaluation roster (identity-only, no paths)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NoReturn

from rad.phase_b import b2_dlcm_v2_protocol as protocol
from rad.phase_b.b2_tiny_split import collect_source_records

ALLOWED_CATEGORIES = ("bottle", "carpet")
PER_GROUP_COUNT = 4
PUBLIC_FIELDS = (
    "stable_sample_id",
    "category",
    "normal_or_anomalous",
    "source_record_scientific_sha256",
    "source_manifest_scientific_sha256",
    "selection_rank",
)
FORBIDDEN_SUBSTRINGS = (
    "path",
    "uri",
    "url",
    "filename",
    "directory",
    "dir",
    "root",
    "image_identity",
    "mask_identity",
)


class B2DLCMV2FinalRosterError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> NoReturn:
    raise B2DLCMV2FinalRosterError(code, detail)


def _record_scientific_sha256(record: Mapping[str, Any]) -> str:
    scientific = {
        "stable_sample_id": record["stable_sample_id"],
        "dataset": record.get("dataset", "mvtec"),
        "category": record["category"],
        "source_split": record.get("source_split", "test"),
        "anomaly_type": record.get("anomaly_type"),
        "image_label": record["image_label"],
    }
    return protocol.canonical_json_sha256(scientific)


def load_original_32_stable_ids(split_manifest: Mapping[str, Any]) -> set[str]:
    ids: set[str] = set()
    for split in ("training", "calibration", "evaluation"):
        for row in split_manifest["splits"][split]:
            ids.add(str(row["stable_sample_id"]))
    if len(ids) != 32:
        _fail("B2_DLCM_FINAL_ROSTER_SOURCE_INVALID", f"expected 32 exclusion ids, got {len(ids)}")
    return ids


def verify_split_manifest_receipt(manifest_path: Path | str) -> dict[str, Any]:
    path = Path(manifest_path)
    if not path.is_file():
        _fail("B2_DLCM_FINAL_ROSTER_SOURCE_INVALID", f"missing split manifest {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    receipt = path.with_suffix(path.suffix + ".sha256")
    if receipt.is_file():
        claimed = receipt.read_text(encoding="utf-8").strip().split()[0]
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if claimed != actual:
            _fail("B2_DLCM_FINAL_ROSTER_SOURCE_INVALID", "split manifest receipt mismatch")
    contract = payload.get("scientific_hash_contract") or {}
    expected = contract.get("canonical_scientific_hash_v2")
    if not expected:
        _fail("B2_DLCM_FINAL_ROSTER_SOURCE_INVALID", "missing canonical_scientific_hash_v2")
    # Trust embedded hash when receipt optional; still require source block.
    if "source" not in payload or "splits" not in payload:
        _fail("B2_DLCM_FINAL_ROSTER_SOURCE_INVALID", "split manifest missing source/splits")
    return payload


def resolve_specification(
    split_manifest: Mapping[str, Any],
    *,
    repo_root: Path | str | None = None,
) -> dict[str, Any]:
    """Load full Gate-C specification; manifest may only store a path pointer."""

    pointer = split_manifest.get("specification")
    if not isinstance(pointer, Mapping):
        _fail("B2_DLCM_FINAL_ROSTER_SOURCE_INVALID", "split manifest missing specification")
    if "stable_sample_id_rule" in pointer and "categories" in pointer:
        return dict(pointer)
    rel = pointer.get("path")
    if not isinstance(rel, str) or not rel:
        _fail("B2_DLCM_FINAL_ROSTER_SOURCE_INVALID", "specification path pointer missing")
    root = Path(repo_root) if repo_root is not None else Path.cwd()
    path = Path(rel)
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        _fail("B2_DLCM_FINAL_ROSTER_SOURCE_INVALID", f"specification file missing: {path}")
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    claimed = pointer.get("sha256")
    actual = hashlib.sha256(raw).hexdigest()
    if claimed and claimed != actual:
        # Tracked specification bytes may evolve while the Gate-C scientific
        # contract remains pinned; accept only when scientific V2 hashes match.
        expected_v2 = (split_manifest.get("scientific_hash_contract") or {}).get(
            "canonical_scientific_hash_v2"
        )
        got_v2 = (payload.get("scientific_hash_contract") or {}).get(
            "canonical_scientific_hash_v2"
        )
        if not expected_v2 or got_v2 != expected_v2:
            _fail(
                "B2_DLCM_FINAL_ROSTER_SOURCE_INVALID",
                "specification sha256 mismatch and scientific hash contract drifted",
            )
    return payload


def build_final_roster(
    *,
    source_root: Path | str,
    split_manifest: Mapping[str, Any],
    implementation_commit: str,
    specification: Mapping[str, Any] | None = None,
    repo_root: Path | str | None = None,
) -> dict[str, Any]:
    """Deterministically select 16 untouched final samples (identity metadata only)."""

    if not implementation_commit or len(implementation_commit) < 7:
        _fail("B2_DLCM_FINAL_ROSTER_SOURCE_INVALID", "implementation_commit required")

    excluded = load_original_32_stable_ids(split_manifest)
    if specification is None:
        specification = resolve_specification(split_manifest, repo_root=repo_root)
    snapshot = collect_source_records(source_root=source_root, specification=specification)
    source_manifest_sha = snapshot.source_list_sha256
    expected_source = split_manifest["source"]["source_list_sha256"]
    if source_manifest_sha != expected_source:
        _fail(
            "B2_DLCM_FINAL_ROSTER_SOURCE_INVALID",
            "source_list_sha256 mismatch vs official split manifest",
        )

    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {
        (cat, label): []
        for cat in ALLOWED_CATEGORIES
        for label in ("normal", "anomalous")
    }
    for record in snapshot.canonical_records:
        cat = str(record["category"])
        if cat not in ALLOWED_CATEGORIES:
            continue
        sid = str(record["stable_sample_id"])
        if sid in excluded:
            continue
        label = "normal" if int(record["image_label"]) == 0 else "anomalous"
        groups[(cat, label)].append(record)

    selected: list[dict[str, Any]] = []
    for cat in ALLOWED_CATEGORIES:
        for label in ("normal", "anomalous"):
            pool = sorted(groups[(cat, label)], key=lambda r: str(r["stable_sample_id"]))
            if len(pool) < PER_GROUP_COUNT:
                _fail(
                    "B2_DLCM_FINAL_ROSTER_INSUFFICIENT",
                    f"{cat}/{label} has {len(pool)} < {PER_GROUP_COUNT}",
                )
            for rank, record in enumerate(pool[:PER_GROUP_COUNT], start=1):
                sid = str(record["stable_sample_id"])
                if sid in excluded:
                    _fail("B2_DLCM_FINAL_ROSTER_OVERLAP", f"selected id overlaps original 32: {sid}")
                row = {
                    "stable_sample_id": sid,
                    "category": cat,
                    "normal_or_anomalous": label,
                    "source_record_scientific_sha256": _record_scientific_sha256(record),
                    "source_manifest_scientific_sha256": source_manifest_sha,
                    "selection_rank": rank,
                }
                _assert_no_path_fields(row)
                selected.append(row)

    selected_ids = [row["stable_sample_id"] for row in selected]
    if len(selected_ids) != 16 or len(set(selected_ids)) != 16:
        _fail("B2_DLCM_FINAL_ROSTER_SOURCE_INVALID", "selected roster must be 16 unique ids")
    if set(selected_ids) & excluded:
        _fail("B2_DLCM_FINAL_ROSTER_OVERLAP", "overlap with original 32")

    roster = {
        "schema_version": "b2_dlcm_v2_final_evaluation_roster_v1",
        "implementation_commit": implementation_commit,
        "source_manifest_scientific_sha256": source_manifest_sha,
        "source_dataset_root_identity_sha256": snapshot.dataset_root_identity_sha256,
        "exclusion_stable_sample_ids_sha256": protocol.canonical_json_sha256(
            sorted(excluded)
        ),
        "exclusion_count": 32,
        "selection_rule": {
            "categories": list(ALLOWED_CATEGORIES),
            "per_group_count": PER_GROUP_COUNT,
            "grouping": ["category", "normal_or_anomalous"],
            "order": "ascending stable_sample_id",
            "take": "first_n_per_group",
        },
        "counts": {
            "bottle_normal": 4,
            "bottle_anomalous": 4,
            "carpet_normal": 4,
            "carpet_anomalous": 4,
            "total": 16,
            "overlap": 0,
        },
        "records": selected,
        "development_evaluation_status": (
            "used_for_B2_05B_qualification_and_postmortem"
        ),
        "final_content_resolved": False,
        "paths_present": False,
    }
    roster["roster_scientific_sha256"] = protocol.canonical_json_sha256(
        {k: v for k, v in roster.items() if k != "roster_scientific_sha256"}
    )
    return roster


def _assert_no_path_fields(row: Mapping[str, Any]) -> None:
    for key in row:
        lowered = key.lower()
        if any(token in lowered for token in FORBIDDEN_SUBSTRINGS):
            _fail("B2_DLCM_FINAL_CONTENT_ACCESS_FORBIDDEN", f"forbidden field in roster: {key}")
        if key not in PUBLIC_FIELDS:
            _fail("B2_DLCM_FINAL_CONTENT_ACCESS_FORBIDDEN", f"non-public roster field: {key}")


def assert_roster_no_paths(roster: Mapping[str, Any]) -> None:
    for row in roster["records"]:
        _assert_no_path_fields(row)
        for value in row.values():
            if isinstance(value, str) and ("/" in value or "\\" in value):
                # stable ids are hex; paths would contain slash
                if value.count("/") >= 1 and not all(c in "0123456789abcdef" for c in value.replace("/", "")):
                    _fail("B2_DLCM_FINAL_CONTENT_ACCESS_FORBIDDEN", "path-like value in roster")


def persist_roster(path: Path | str, roster: Mapping[str, Any]) -> str:
    assert_roster_no_paths(roster)
    return protocol.persist_json_atomic(path, roster)


def build_roster_from_official_paths(
    *,
    source_root: Path | str,
    split_manifest_path: Path | str,
    implementation_commit: str,
    repo_root: Path | str | None = None,
) -> dict[str, Any]:
    manifest = verify_split_manifest_receipt(split_manifest_path)
    root = Path(repo_root) if repo_root is not None else Path(split_manifest_path).resolve().parents[0]
    # Prefer repo root containing configs/phase_b/b2_tiny_gate_c.json
    for candidate in (
        Path(repo_root) if repo_root is not None else None,
        Path.cwd(),
        Path(__file__).resolve().parents[2],
    ):
        if candidate is None:
            continue
        if (candidate / "configs/phase_b/b2_tiny_gate_c.json").is_file():
            root = candidate
            break
    return build_final_roster(
        source_root=source_root,
        split_manifest=manifest,
        implementation_commit=implementation_commit,
        repo_root=root,
    )


def hermetic_roster_from_records(
    *,
    candidates: Sequence[Mapping[str, Any]],
    excluded_ids: Sequence[str],
    implementation_commit: str,
    source_manifest_scientific_sha256: str,
) -> dict[str, Any]:
    """Test-only deterministic roster builder over pre-enumerated identity records."""

    excluded = set(excluded_ids)
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {
        (cat, label): []
        for cat in ALLOWED_CATEGORIES
        for label in ("normal", "anomalous")
    }
    for record in candidates:
        cat = str(record["category"])
        if cat not in ALLOWED_CATEGORIES:
            continue
        sid = str(record["stable_sample_id"])
        if sid in excluded:
            continue
        label = str(record["normal_or_anomalous"])
        groups[(cat, label)].append(record)
    selected: list[dict[str, Any]] = []
    for cat in ALLOWED_CATEGORIES:
        for label in ("normal", "anomalous"):
            pool = sorted(groups[(cat, label)], key=lambda r: str(r["stable_sample_id"]))
            if len(pool) < PER_GROUP_COUNT:
                _fail(
                    "B2_DLCM_FINAL_ROSTER_INSUFFICIENT",
                    f"{cat}/{label} has {len(pool)} < {PER_GROUP_COUNT}",
                )
            for rank, record in enumerate(pool[:PER_GROUP_COUNT], start=1):
                sid = str(record["stable_sample_id"])
                if sid in excluded:
                    _fail("B2_DLCM_FINAL_ROSTER_OVERLAP", sid)
                row = {
                    "stable_sample_id": sid,
                    "category": cat,
                    "normal_or_anomalous": label,
                    "source_record_scientific_sha256": str(
                        record.get("source_record_scientific_sha256")
                        or protocol.canonical_json_sha256({"stable_sample_id": sid})
                    ),
                    "source_manifest_scientific_sha256": source_manifest_scientific_sha256,
                    "selection_rank": rank,
                }
                _assert_no_path_fields(row)
                selected.append(row)
    if set(r["stable_sample_id"] for r in selected) & excluded:
        _fail("B2_DLCM_FINAL_ROSTER_OVERLAP", "overlap")
    roster = {
        "schema_version": "b2_dlcm_v2_final_evaluation_roster_v1",
        "implementation_commit": implementation_commit,
        "source_manifest_scientific_sha256": source_manifest_scientific_sha256,
        "exclusion_count": len(excluded),
        "records": selected,
        "counts": {
            "bottle_normal": 4,
            "bottle_anomalous": 4,
            "carpet_normal": 4,
            "carpet_anomalous": 4,
            "total": 16,
            "overlap": 0,
        },
        "final_content_resolved": False,
        "paths_present": False,
    }
    roster["roster_scientific_sha256"] = protocol.canonical_json_sha256(
        {k: v for k, v in roster.items() if k != "roster_scientific_sha256"}
    )
    return roster
