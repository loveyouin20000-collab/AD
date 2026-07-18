from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class EfficiencyTrace:
    """Per-sample efficiency record."""

    sample_id: str
    latency_ms: float
    selector_overhead_ms: float
    peak_memory_mb: float
    backbone_ms: float = 0.0
    maps_ms: float = 0.0
    dlcm_ms: float = 0.0
    lse_ms: float = 0.0


def aggregate_efficiency(traces: Sequence[EfficiencyTrace]) -> dict[str, Any]:
    if not traces:
        return {
            "n": 0,
            "mean_latency_ms": float("nan"),
            "throughput_img_s": float("nan"),
            "mean_selector_overhead_ms": float("nan"),
            "peak_memory_mb": float("nan"),
        }
    latencies = [float(t.latency_ms) for t in traces]
    overheads = [float(t.selector_overhead_ms) for t in traces]
    mems = [float(t.peak_memory_mb) for t in traces]
    mean_lat = sum(latencies) / len(latencies)
    return {
        "n": len(traces),
        "mean_latency_ms": mean_lat,
        "median_latency_ms": float(sorted(latencies)[len(latencies) // 2]),
        "throughput_img_s": (1000.0 / mean_lat) if mean_lat > 0 else float("nan"),
        "mean_selector_overhead_ms": sum(overheads) / len(overheads),
        "peak_memory_mb": max(mems),
        "mean_backbone_ms": sum(t.backbone_ms for t in traces) / len(traces),
        "mean_maps_ms": sum(t.maps_ms for t in traces) / len(traces),
        "mean_dlcm_ms": sum(t.dlcm_ms for t in traces) / len(traces),
        "mean_lse_ms": sum(t.lse_ms for t in traces) / len(traces),
    }


def write_efficiency_traces(
    traces: Sequence[EfficiencyTrace],
    *,
    output_dir: Path | str,
) -> dict[str, Any]:
    """Write per-sample efficiency traces before aggregate summary."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    trace_path = out / "efficiency_traces.jsonl"
    with trace_path.open("w", encoding="utf-8") as f:
        for t in traces:
            f.write(json.dumps(asdict(t)) + "\n")
    summary = aggregate_efficiency(traces)
    (out / "efficiency_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
