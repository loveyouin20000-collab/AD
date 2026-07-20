#!/usr/bin/env python3
"""Diagnose the first CUDA divergence between official and staged VisualAD execution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rad.qualification.b1_cuda_equivalence import (  # noqa: E402
    B1_ACCEPTED_CHECKPOINT,
    B1_ACCEPTED_CHECKPOINT_SHA256,
    collect_environment,
    diagnose_four_path_divergence,
    deterministic_synthetic,
    git_sha,
    load_preprocessed_image,
    load_teacher_production,
    set_seed,
    tensor_fingerprint,
    validate_checkpoint,
)


def _scan_staged_mutations() -> list[dict[str, str]]:
    text = (REPO_ROOT / "VisualAD_lib" / "VisualAD.py").read_text(encoding="utf-8")
    keys = (
        ".clone(",
        ".detach(",
        ".contiguous(",
        ".to(",
        ".float(",
        ".half(",
        ".permute(",
        ".transpose(",
        ".reshape(",
        ".view(",
        "copy_",
    )
    hits: list[dict[str, str]] = []
    for line_no, line in enumerate(text.splitlines(), 1):
        if any(token in line for token in ("run_to", "prepare_stage", "forward_staged", "StageCache")):
            for key in keys:
                if key in line:
                    hits.append({"line": str(line_no), "code": line.strip(), "keyword": key})
    return hits


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument(
        "--sample",
        default="/root/autodl-tmp/data/mvtec/bottle/test/good/000.png",
        help="Real image path for deterministic sample.",
    )
    parser.add_argument("--sample-id", default="mvtec_sample_000")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--max-block", type=int, default=12)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPO_ROOT / "docs" / "phase_b" / "b1_cuda_divergence_diagnosis.json",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not torch.cuda.is_available():
        print("CUDA is required for staged divergence diagnosis.", file=sys.stderr)
        return 2

    checkpoint = validate_checkpoint(args.checkpoint, args.expected_sha256)
    output_json = args.output_json
    if args.output_dir is not None:
        output_json = args.output_dir / "b1_cuda_divergence_diagnosis.json"

    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "checkpoint": str(checkpoint),
                    "expected_sha256": args.expected_sha256,
                    "sample_id": args.sample_id,
                    "synthetic": args.synthetic,
                    "max_block": args.max_block,
                    "seed": args.seed,
                    "output_json": str(output_json),
                },
                indent=2,
            )
        )
        return 0

    device = torch.device("cuda:0")
    set_seed(args.seed)
    bundle = load_teacher_production(checkpoint, args.expected_sha256, device)
    if args.synthetic:
        image = deterministic_synthetic(device)
        sample_id = "synthetic_seed111"
    else:
        sample_path = Path(args.sample)
        if not sample_path.is_file():
            print(f"sample missing: {sample_path}", file=sys.stderr)
            return 2
        image = load_preprocessed_image(str(sample_path), device)
        sample_id = args.sample_id

    diagnosis = diagnose_four_path_divergence(
        bundle,
        image,
        sample_id=sample_id,
        max_block=args.max_block,
    )
    payload: dict[str, Any] = {
        "schema_version": 2,
        "phase": "B1",
        "status": "failed",
        "git_sha": git_sha(),
        "environment": collect_environment(),
        "checkpoint": {"path": str(checkpoint), "sha256": args.expected_sha256},
        "seed": args.seed,
        "dry_run": False,
        "staged_mutation_scan": _scan_staged_mutations(),
        "input_image_fingerprint": tensor_fingerprint(image),
        **diagnosis,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(diagnosis["comparison_summary_table"], indent=2))
    print(f"Root cause: {diagnosis['root_cause_classification']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
