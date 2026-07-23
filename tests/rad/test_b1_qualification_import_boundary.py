"""B1-05: qualification tools must not import executable logic from tests.*."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
QUALIFY_TOOL = REPO_ROOT / "tools" / "qualify_b1_cuda_equivalence.py"
DIAGNOSE_TOOL = REPO_ROOT / "tools" / "diagnose_staged_cuda_divergence.py"
QUAL_MODULE = REPO_ROOT / "rad" / "qualification" / "b1_cuda_equivalence.py"


def _imported_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.append(node.module)
    return modules


def _imports_tests_package(path: Path) -> list[str]:
    hits: list[str] = []
    for module in _imported_modules(path):
        if module == "tests" or module.startswith("tests."):
            hits.append(module)
    return hits


def test_qualify_tool_does_not_import_tests_package() -> None:
    hits = _imports_tests_package(QUALIFY_TOOL)
    assert hits == [], f"{QUALIFY_TOOL.name} imports tests.*: {hits}"


def test_diagnose_tool_does_not_import_tests_package() -> None:
    hits = _imports_tests_package(DIAGNOSE_TOOL)
    assert hits == [], f"{DIAGNOSE_TOOL.name} imports tests.*: {hits}"


def test_reusable_b1_cuda_logic_lives_outside_tests() -> None:
    assert QUAL_MODULE.is_file(), (
        "shared B1 CUDA qualification logic must live in "
        "rad/qualification/b1_cuda_equivalence.py"
    )
    source = QUAL_MODULE.read_text(encoding="utf-8")
    assert "def apply_deterministic_cuda_settings" in source
    assert "def run_equivalence_protocol" in source
    assert "def load_task_level_category_samples" in source
