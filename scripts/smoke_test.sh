#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${WIND_PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
PYTHON_BIN=${PYTHON_BIN:-python3}
cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" -m compileall -q src_2

if command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    "${PYTHON_BIN}" - <<'PY'
from pathlib import Path

required = [
    Path("README.md"),
    Path("requirements-plot.txt"),
    Path("src_2/comparison.py"),
    Path("src_2/drawing_temporal_spatial_consistency.py"),
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit(f"Missing release files: {missing}")
print("Source smoke test passed")
PY
fi
