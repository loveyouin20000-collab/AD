"""Portability / leakage guards for B2-05A."""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PRODUCTION = [
    REPO / "rad/phase_b/b2_dlcm.py",
    REPO / "rad/phase_b/b2_dlcm_training.py",
    REPO / "rad/phase_b/b2_dlcm_deployment.py",
    REPO / "tools/train_b2_dlcm.py",
    REPO / "tools/verify_b2_dlcm_artifacts.py",
]


def test_production_modules_do_not_import_fixtures() -> None:
    for path in PRODUCTION:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                assert "b2_dlcm_fixtures" not in mod
                assert "tests.rad" not in mod
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "b2_dlcm_fixtures" not in alias.name


def test_scope_exclusions_in_docs() -> None:
    arch = (REPO / "docs/phase_b/b2_05a_dlcm_training_architecture.md").read_text(encoding="utf-8")
    for phrase in (
        "real DLCM training not started",
        "evaluation not unlocked",
        "residual-gain",
        "LSE",
        "early-exit",
    ):
        assert phrase.lower() in arch.lower() or phrase in arch


def test_no_teacher_or_visa_imports_in_new_modules() -> None:
    banned = ("visa", "VisA", "load_teacher", "VisualAD_lib")
    for path in PRODUCTION:
        text = path.read_text(encoding="utf-8")
        for token in banned:
            assert token not in text, f"{path} contains banned token {token}"
