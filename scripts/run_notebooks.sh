#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root_dir"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install uv first: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

# Ensure Jupyter/IPython uses writable, project-local directories.
export IPYTHONDIR="$root_dir/.ipython"
export JUPYTER_RUNTIME_DIR="$root_dir/.jupyter/runtime"
export JUPYTER_DATA_DIR="$root_dir/.jupyter/data"
mkdir -p "$IPYTHONDIR" "$JUPYTER_RUNTIME_DIR" "$JUPYTER_DATA_DIR"

echo "Syncing dev dependencies (jupyter, etc.)..."
uv sync --group dev >/dev/null

notebooks=(
  "notebooks/normal.ipynb"
  "notebooks/lognormal.ipynb"
  "notebooks/uniform.ipynb"
  "notebooks/exponential.ipynb"
)

echo "Executing notebooks with nbconvert..."
for nb in "${notebooks[@]}"; do
  echo "==> $nb"
  uv run jupyter nbconvert --execute --to notebook --inplace "$nb"
done

echo "All notebooks executed successfully."

