#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="rad-visualad"
conda create -y -n "${ENV_NAME}" python=3.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
python -m pip install --upgrade pip
python -m pip install torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu118
grep -vE '^(torch|torchvision)==' requirements.txt > /tmp/visualad-no-torch.txt
python -m pip install -r /tmp/visualad-no-torch.txt
python -m pip install -r requirements-dev.txt
python tools/validate_environment.py
