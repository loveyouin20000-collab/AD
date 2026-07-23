"""Hermetic helpers for portable B2 CPU unit tests (no AutoDL paths/tags)."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
B2_SPLIT_FIXTURE = REPO_ROOT / "tests" / "rad" / "fixtures" / "b2_gate_c_split_manifest.json"
EXPECTED_SPLIT_V2 = (
    "91570da1fed6d7859d407196b10403581832ae0ff677a1ea7657ca76b91471f0"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "97bd461163efb96e36cddb1c3adf677e4c4fc2daabb2521021689f30e799b4f4"
)
EXPECTED_B1_TAG = "b1-strict-independent-v1"
EXPECTED_B1_COMMIT = "3a751b2784a50eb0a08ed49e1db2df0b53608ccc"
EXPECTED_B2_COMMIT = "18bac047227754c975b23b46842458a5b41d5e2a"
EXPECTED_CONTRACT_COMMIT = "ec2f3eaf9a8e393e2f662dbcd8ec1c2c3437a024"


def load_b2_split_fixture() -> dict[str, Any]:
    payload = json.loads(B2_SPLIT_FIXTURE.read_text(encoding="utf-8"))
    digest = payload["scientific_hash_contract"]["canonical_scientific_hash_v2"]
    if digest != EXPECTED_SPLIT_V2:
        raise AssertionError("tracked B2 split fixture drifted from approved V2 hash")
    return payload


def write_b2_split_fixture(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(B2_SPLIT_FIXTURE, destination)
    return destination


def write_hermetic_checkpoint(destination: Path) -> tuple[Path, str]:
    """Write deterministic bytes; return path and *fixture* SHA-256."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = b"b2-hermetic-checkpoint-fixture-v1\n"
    destination.write_bytes(payload)
    return destination, hashlib.sha256(payload).hexdigest()


def synthetic_teacher_cache_identity(
    *,
    head_commit: str,
    worktree_clean: bool = True,
    head_is_descendant: bool = True,
) -> dict[str, Any]:
    return {
        "b2_tag_commit": EXPECTED_B2_COMMIT,
        "contract_tag_commit": EXPECTED_CONTRACT_COMMIT,
        "head_commit": head_commit,
        "head_is_descendant": head_is_descendant,
        "worktree_clean": worktree_clean,
    }


def synthetic_tiny_split_identity(
    *,
    head_commit: str,
    worktree_clean: bool = True,
    worktree_path: str,
    branch: str = "hermetic-b2",
) -> dict[str, Any]:
    return {
        "b1_base_tag": EXPECTED_B1_TAG,
        "b1_base_commit": EXPECTED_B1_COMMIT,
        "generation_git_commit": head_commit,
        "generation_branch": branch,
        "worktree_clean": worktree_clean,
        "worktree_path": worktree_path,
    }


def populate_controlled_mvtec(root: Path) -> Path:
    """Minimal MVTec-like tree for portable tiny-split CLI tests."""

    for category in ("bottle", "carpet"):
        for anomaly_type in ("good", "crack"):
            for index in range(8):
                image = root / category / "test" / anomaly_type / f"{index:03d}.png"
                image.parent.mkdir(parents=True, exist_ok=True)
                image.write_bytes(b"controlled-source-image")
                if anomaly_type != "good":
                    mask = (
                        root
                        / category
                        / "ground_truth"
                        / anomaly_type
                        / f"{index:03d}_mask.png"
                    )
                    mask.parent.mkdir(parents=True, exist_ok=True)
                    mask.write_bytes(b"controlled-source-mask")
    return root


def populate_b1_task_level_roots(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create MVTec bottle + VisA candle trees plus a rejected mvtec/sample fixture."""

    mvtec = tmp_path / "mvtec"
    visa = tmp_path / "Visa"
    sample = mvtec / "sample"
    # MVTec bottle
    for anomaly, label in (("good", 0), ("broken_large", 1), ("contamination", 1)):
        for index in range(3):
            image = mvtec / "bottle" / "test" / anomaly / f"{index:03d}.png"
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(b"mvtec-bottle")
            if label == 1:
                mask = (
                    mvtec
                    / "bottle"
                    / "ground_truth"
                    / anomaly
                    / f"{index:03d}_mask.png"
                )
                mask.parent.mkdir(parents=True, exist_ok=True)
                mask.write_bytes(b"mask")
    # Rejected flat fixture directory
    sample.mkdir(parents=True, exist_ok=True)
    (sample / "000.png").write_bytes(b"fixture")
    # VisA candle (adapter requires meta.json)
    candle_entries: list[dict[str, Any]] = []
    for kind, count in (("Normal", 3), ("Anomaly", 3)):
        folder = visa / "candle" / "Data" / "Images" / kind
        folder.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            rel_image = f"candle/Data/Images/{kind}/{index:04d}.JPG"
            (folder / f"{index:04d}.JPG").write_bytes(b"visa-candle")
            entry: dict[str, Any] = {
                "cls_name": "candle",
                "img_path": rel_image,
                "anomaly": 0 if kind == "Normal" else 1,
            }
            if kind == "Anomaly":
                masks = visa / "candle" / "Data" / "Masks" / "Anomaly"
                masks.mkdir(parents=True, exist_ok=True)
                rel_mask = f"candle/Data/Masks/Anomaly/{index:04d}.png"
                (masks / f"{index:04d}.png").write_bytes(b"visa-mask")
                entry["mask_path"] = rel_mask
            candle_entries.append(entry)
    (visa / "meta.json").write_text(
        json.dumps({"test": {"candle": candle_entries}}, indent=2),
        encoding="utf-8",
    )
    return mvtec, visa, sample
