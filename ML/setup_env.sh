#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="${ROOT_DIR}/env"

python3 -m venv "${ENV_DIR}"
source "${ENV_DIR}/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "${ROOT_DIR}/requirements.txt"

echo "Environment ready. Activate with: source ${ENV_DIR}/bin/activate"
