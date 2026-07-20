from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rad.config import ExperimentConfig  # noqa: E402
from rad.data.cache_dataset import TeacherCacheDataset  # noqa: E402
from rad.models.descriptors import (  # noqa: E402
    CheckpointContextExtractor,
    DescriptorNormalizer,
    LayerDescriptorExtractor,
)
from rad.models.dlcm import DLCM, sum_preserving_fusion  # noqa: E402
from rad.models.lse import LSE  # noqa: E402
from rad.models.selector_signals import (  # noqa: E402
    SelectorSignalLayout,
    apply_selector_signal_mask,
    build_default_selector_signal_layout,
    parse_enabled_signals,
    selector_signal_provenance,
)
from rad.trainers.lse_trainer import LSETrainer  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LSE on residual-gain targets")
    p.add_argument("--config", type=str, default="configs/rad/lse.yaml")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--limit-train", type=int, default=None)
    p.add_argument("--limit-cal", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def load_gain_index(path: Path) -> dict[str, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu")
    out: dict[str, dict[str, Any]] = {}
    for rec in payload["records"]:
        out[rec["sample_id"]] = rec
    return out


@torch.no_grad()
def build_states_for_sample(
    *,
    sample: dict[str, Any],
    dlcm: DLCM,
    layer_extractor: LayerDescriptorExtractor,
    context_extractor: CheckpointContextExtractor,
    normalizer: DescriptorNormalizer | None,
    candidate_layers: tuple[int, ...],
    early_depths: tuple[int, ...],
    device: torch.device,
    enabled_signals: dict[str, bool] | None = None,
    selector_layout: SelectorSignalLayout | None = None,
) -> dict[int, torch.Tensor]:
    """Return depth -> state [state_dim] using mean-pooled layer desc + context (18+8)."""
    signals = parse_enabled_signals(enabled_signals)
    layout = selector_layout or build_default_selector_signal_layout()
    prev_fused: torch.Tensor | None = None
    states: dict[int, torch.Tensor] = {}
    for depth in sorted(early_depths):
        avail = [x for x in candidate_layers if x <= depth]
        stacked = torch.stack([sample["maps"][depth][layer] for layer in avail], dim=0)
        maps = stacked.unsqueeze(0).unsqueeze(2).to(device)  # [1,L,1,H,W]
        layer_ids = torch.tensor([avail], dtype=torch.long, device=device)
        b, l = maps.shape[:2]
        valid = torch.ones(b, l, dtype=torch.bool, device=device)
        maps_4d = maps.squeeze(2)
        layer_desc = layer_extractor(maps_4d, valid_mask=valid)
        if normalizer is not None:
            flat = layer_desc.reshape(b * l, -1)
            flat = normalizer.transform(flat)
            layer_desc = flat.view(b, l, -1)
        ctx = context_extractor(
            maps_4d,
            valid_mask=valid,
            layer_ids=layer_ids,
            prev_fused=prev_fused,
        )
        weights = dlcm(layer_desc, ctx, layer_ids, valid)
        fused = sum_preserving_fusion(maps, weights, valid)
        # Primary scientific ablation: mask during training materialization (A).
        lse_desc = apply_selector_signal_mask(
            layer_desc, layout=layout, enabled_signals=signals
        )
        state = torch.cat([lse_desc.mean(dim=1), ctx], dim=-1)[0].cpu()
        states[int(depth)] = state
        prev_fused = fused.detach()
    return states


class LSEFeatureDataset(Dataset):
    """Precomputed (state, depth, targets) rows for one split."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


def collate_lse(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sample_id": [b["sample_id"] for b in batch],
        "state": torch.stack([b["state"] for b in batch], dim=0),
        "depth_id": torch.stack([b["depth_id"] for b in batch], dim=0),
        "target_gain": torch.stack([b["target_gain"] for b in batch], dim=0),
        "target_sufficient": torch.stack([b["target_sufficient"] for b in batch], dim=0),
    }


def materialize_rows(
    *,
    cache: TeacherCacheDataset,
    gains: dict[str, dict[str, Any]],
    dlcm: DLCM,
    layer_extractor: LayerDescriptorExtractor,
    context_extractor: CheckpointContextExtractor,
    normalizer: DescriptorNormalizer | None,
    candidate_layers: tuple[int, ...],
    early_depths: tuple[int, ...],
    device: torch.device,
    limit: int | None,
    enabled_signals: dict[str, bool] | None = None,
    selector_layout: SelectorSignalLayout | None = None,
) -> list[dict[str, Any]]:
    n = len(cache) if limit is None else min(len(cache), limit)
    rows: list[dict[str, Any]] = []
    missing = 0
    for i in tqdm(range(n), desc="materialize"):
        sample = cache[i]
        sid = sample["sample_id"]
        rec = gains.get(sid)
        if rec is None:
            missing += 1
            continue
        states = build_states_for_sample(
            sample=sample,
            dlcm=dlcm,
            layer_extractor=layer_extractor,
            context_extractor=context_extractor,
            normalizer=normalizer,
            candidate_layers=candidate_layers,
            early_depths=early_depths,
            device=device,
            enabled_signals=enabled_signals,
            selector_layout=selector_layout,
        )
        for depth in early_depths:
            g = rec["gains"][depth]
            if hasattr(g, "reshape"):
                g = float(g.reshape(-1)[0].item())
            else:
                g = float(g)
            s = rec["sufficient"][depth]
            if hasattr(s, "reshape"):
                s = float(s.reshape(-1)[0].item())
            else:
                s = float(s)
            rows.append(
                {
                    "sample_id": sid,
                    "state": states[depth],
                    "depth_id": torch.tensor(depth, dtype=torch.long),
                    "target_gain": torch.tensor(g, dtype=torch.float32),
                    "target_sufficient": torch.tensor(s, dtype=torch.float32),
                }
            )
    if missing:
        print(f"warning: skipped {missing} samples without gain targets")
    return rows


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "sample_id": batch.get("sample_id"),
        "state": batch["state"].to(device),
        "depth_id": batch["depth_id"].to(device),
        "target_gain": batch["target_gain"].to(device),
        "target_sufficient": batch["target_sufficient"].to(device),
    }


def main() -> int:
    args = parse_args()
    raw = yaml.safe_load(Path(args.config).read_text())
    cfg = ExperimentConfig.from_yaml(args.config)
    lse_cfg = dict(raw.get("lse", {}))
    fusion_cfg = raw.get("fusion", {})

    seed = args.seed if args.seed is not None else cfg.seed
    torch.manual_seed(seed)
    device = torch.device(args.device or raw.get("device", cfg.device))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    early_depths = tuple(int(x) for x in lse_cfg.get("early_depths", [12, 18]))
    candidate_layers = tuple(cfg.backbone.candidate_layers)
    epsilon_gain = float(lse_cfg.get("epsilon_gain", 0.05))
    epochs = args.epochs if args.epochs is not None else int(lse_cfg.get("epochs", 30))
    patience = args.patience if args.patience is not None else int(lse_cfg.get("patience", 10))
    batch_size = int(lse_cfg.get("batch_size", 32))
    lr = float(lse_cfg.get("lr", 1e-3))
    state_dim = int(lse_cfg.get("state_dim", 26))  # 18 + 8

    train_cache_dir = Path(lse_cfg.get("train_cache", fusion_cfg.get("train_cache")))
    cal_cache_dir = Path(
        lse_cfg.get("calibration_cache", fusion_cfg.get("calibration_cache"))
    )
    if not train_cache_dir.is_absolute():
        train_cache_dir = REPO_ROOT / train_cache_dir
    if not cal_cache_dir.is_absolute():
        cal_cache_dir = REPO_ROOT / cal_cache_dir

    train_gains_path = Path(
        lse_cfg.get("train_gain_targets", "artifacts/targets/gain/mvtec_train.pt")
    )
    cal_gains_path = Path(
        lse_cfg.get("calibration_gain_targets", "artifacts/targets/gain/mvtec_calibration.pt")
    )
    if not train_gains_path.is_absolute():
        train_gains_path = REPO_ROOT / train_gains_path
    if not cal_gains_path.is_absolute():
        cal_gains_path = REPO_ROOT / cal_gains_path

    ckpt_path = Path(lse_cfg.get("dlcm_checkpoint", fusion_cfg.get("output_dir", "") + "/seed_222/dlcm.pt"))
    if not ckpt_path.is_absolute():
        ckpt_path = REPO_ROOT / ckpt_path
    stats_path = Path(lse_cfg.get("descriptor_stats", fusion_cfg.get("descriptor_stats", "")))
    if stats_path and not stats_path.is_absolute():
        stats_path = REPO_ROOT / stats_path

    output_dir = args.output_dir or Path(lse_cfg.get("train_output_dir", "artifacts/checkpoints/lse"))
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir

    config_hash = sha256_file(Path(args.config))
    sha = git_sha()
    checkpoint_hash = sha256_file(ckpt_path) if ckpt_path.is_file() else "missing"

    print(f"config: {args.config}")
    print(f"config_hash: {config_hash}")
    print(f"git_sha: {sha}")
    print(f"seed: {seed}")
    print(f"device: {device}")
    print(f"dlcm_checkpoint: {ckpt_path}")
    print(f"checkpoint_hash: {checkpoint_hash}")
    print(f"train_gains: {train_gains_path}")
    print(f"cal_gains: {cal_gains_path}")
    print(f"output_dir: {output_dir}")
    print(f"epochs: {epochs} patience: {patience}")
    print(f"early_depths: {early_depths} state_dim: {state_dim}")

    selector_cfg = dict(raw.get("selector", {}))
    selector_layout = build_default_selector_signal_layout()
    enabled_signals = parse_enabled_signals(selector_cfg.get("signals"))
    mask_mode = str(selector_cfg.get("mask_mode", "train_and_infer"))
    if mask_mode not in {"train_and_infer", "infer_only"}:
        raise SystemExit(f"unsupported selector.mask_mode: {mask_mode}")
    # Primary scientific ablation uses train_and_infer (A). infer_only is stress-only.
    train_time_signals = (
        enabled_signals if mask_mode == "train_and_infer" else parse_enabled_signals(None)
    )
    selector_prov = selector_signal_provenance(
        enabled_signals=enabled_signals,
        layout=selector_layout,
        mask_applied=True,
    )
    print(f"selector_mask_mode: {mask_mode}")
    print(f"selector_signals: {json.dumps(enabled_signals)}")
    print(f"selector_signal_layout_hash: {selector_prov['selector_signal_layout_hash']}")

    if args.dry_run:
        return 0

    if not train_gains_path.is_file():
        raise SystemExit(f"missing train gain targets: {train_gains_path}")
    if not cal_gains_path.is_file():
        raise SystemExit(
            f"missing calibration gain targets: {cal_gains_path}\n"
            "Generate with: python tools/generate_gain_targets.py "
            f"--config {args.config} --cache {cal_cache_dir} --output {cal_gains_path}"
        )
    if not ckpt_path.is_file():
        raise SystemExit(f"missing DLCM checkpoint: {ckpt_path}")

    train_cache = TeacherCacheDataset(train_cache_dir)
    cal_cache = TeacherCacheDataset(cal_cache_dir)
    train_gains = load_gain_index(train_gains_path)
    cal_gains = load_gain_index(cal_gains_path)
    print(f"split_manifest_hash(train_cache): {train_cache.meta.get('split_hash')}")

    normalizer = DescriptorNormalizer.load(stats_path) if stats_path.is_file() else None
    dlcm_ckpt = torch.load(ckpt_path, map_location="cpu")
    dlcm = DLCM(max_layer_id=max(candidate_layers), alpha=0.0)
    dlcm.load_state_dict(dlcm_ckpt["dlcm"])
    dlcm.eval()
    dlcm.to(device)
    for p in dlcm.parameters():
        p.requires_grad_(False)

    layer_extractor = LayerDescriptorExtractor()
    context_extractor = CheckpointContextExtractor(backbone_depth=cfg.backbone.depth)

    train_rows = materialize_rows(
        cache=train_cache,
        gains=train_gains,
        dlcm=dlcm,
        layer_extractor=layer_extractor,
        context_extractor=context_extractor,
        normalizer=normalizer,
        candidate_layers=candidate_layers,
        early_depths=early_depths,
        device=device,
        limit=args.limit_train,
        enabled_signals=train_time_signals,
        selector_layout=selector_layout,
    )
    cal_rows = materialize_rows(
        cache=cal_cache,
        gains=cal_gains,
        dlcm=dlcm,
        layer_extractor=layer_extractor,
        context_extractor=context_extractor,
        normalizer=normalizer,
        candidate_layers=candidate_layers,
        early_depths=early_depths,
        device=device,
        limit=args.limit_cal,
        enabled_signals=train_time_signals,
        selector_layout=selector_layout,
    )
    if train_rows:
        print(f"tensor_shapes state={tuple(train_rows[0]['state'].shape)}")
    print(f"train_rows={len(train_rows)} cal_rows={len(cal_rows)}")

    train_loader = DataLoader(
        LSEFeatureDataset(train_rows),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_lse,
        num_workers=0,
    )
    cal_loader = DataLoader(
        LSEFeatureDataset(cal_rows),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_lse,
        num_workers=0,
    )

    model = LSE(state_dim=state_dim, early_depths=early_depths).to(device)
    trainer = LSETrainer(
        model=model,
        early_depths=early_depths,
        epsilon_gain=epsilon_gain,
        sufficiency_weight=float(lse_cfg.get("sufficiency_weight", 0.5)),
    ).to(device)
    opt = torch.optim.Adam(trainer.parameters(), lr=lr)

    output_dir.mkdir(parents=True, exist_ok=True)
    best_nll = float("inf")
    best_path = output_dir / "lse_best.pt"
    stale = 0
    history: list[dict[str, Any]] = []

    for epoch in range(epochs):
        trainer.train()
        epoch_losses: list[float] = []
        for batch in train_loader:
            batch = move_batch(batch, device)
            metrics = trainer.training_step(batch, opt)
            epoch_losses.append(metrics["loss"])
        train_loss = float(sum(epoch_losses) / max(len(epoch_losses), 1))

        cal_batches = [move_batch(b, device) for b in cal_loader]
        report = trainer.evaluate(cal_batches)
        cal_nll = float(report["nll"])
        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "cal_nll": cal_nll,
            "per_depth": {str(d): report[d] for d in early_depths},
        }
        history.append(entry)
        print(json.dumps(entry, indent=2))

        if cal_nll + 1e-8 < best_nll:
            best_nll = cal_nll
            stale = 0
            torch.save(
                {
                    "seed": seed,
                    "lse": model.state_dict(),
                    "state_dim": state_dim,
                    "early_depths": list(early_depths),
                    "epsilon_gain": epsilon_gain,
                    "cal_nll": cal_nll,
                    "cal_metrics": {str(d): report[d] for d in early_depths},
                    "config_hash": config_hash,
                    "git_sha": sha,
                    "checkpoint_hash": checkpoint_hash,
                    "split_manifest_hash": train_cache.meta.get("split_hash"),
                    "selector_mask_mode": mask_mode,
                    **selector_prov,
                },
                best_path,
            )
            # prediction table at best
            (output_dir / "cal_predictions.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in report["predictions"])
            )
            (output_dir / "cal_metrics.json").write_text(
                json.dumps({str(d): report[d] for d in early_depths} | {"nll": cal_nll}, indent=2)
                + "\n"
            )
        else:
            stale += 1
            if stale >= patience:
                print(f"early stop at epoch {epoch} (patience={patience})")
                break

    (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    summary = {
        "best_checkpoint": str(best_path),
        "best_cal_nll": best_nll,
        "epochs_ran": len(history),
        "config_hash": config_hash,
        "git_sha": sha,
        "checkpoint_hash": checkpoint_hash,
        "seed": seed,
        "selector_mask_mode": mask_mode,
        **selector_prov,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
