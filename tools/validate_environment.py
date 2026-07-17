from __future__ import annotations

import json
import platform
from typing import Any

import torch


def collect_environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "gpu_count": torch.cuda.device_count(),
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }


if __name__ == "__main__":
    print(json.dumps(collect_environment(), indent=2))
