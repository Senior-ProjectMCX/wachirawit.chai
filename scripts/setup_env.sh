#!/usr/bin/env bash
# Create .venv and install dependencies, building pmcxcl from source.
# PyPI's pmcxcl wheels are tagged universal2 but ship an x86_64-only binary,
# so on Apple Silicon the extension silently fails to load.
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-/opt/homebrew/opt/python@3.11/bin/python3.11}"

command -v cmake >/dev/null || brew install cmake

[ -d .venv ] || "$PYTHON" -m venv .venv
source .venv/bin/activate    # cmake must find the venv's ninja on PATH
pip install --upgrade pip
pip install -r requirements.txt ninja

if [ ! -d build_src/mcxcl ]; then
    mkdir -p build_src
    git clone --recurse-submodules https://github.com/fangq/mcxcl.git build_src/mcxcl
fi
pip install --no-build-isolation --force-reinstall --no-deps ./build_src/mcxcl/pmcxcl

python - <<'EOF'
import glob, subprocess, pmcxcl
so = glob.glob(".venv/lib/python*/site-packages/_pmcxcl*.so")[0]
print(subprocess.run(["lipo", "-info", so], capture_output=True, text=True).stdout.strip())
pmcxcl.gpuinfo()
EOF
