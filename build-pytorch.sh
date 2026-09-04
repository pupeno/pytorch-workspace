#!/usr/bin/env bash
#
# Build PyTorch, then restore clangd's filtered compile database.
#
# The build regenerates both compile_commands.json files: cmake rewrites
# build/compile_commands.json, and PyTorch's own merge_compile_commands ALL
# target (cmake/PostBuildSteps.cmake) rewrites the one at the repo root. Both
# come back listing all 3808 translation units, which sends clangd's background
# index off to crawl the entire codebase again -- hours of CPU for third_party
# sources, CUDA kernels it cannot parse, and C++ tests.
#
# Re-filtering afterwards keeps that from happening. It runs even if the build
# fails, because a failed build can still have regenerated the databases.
#
# Any arguments are passed through to pip.

set -euo pipefail

workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

refilter() {
  echo
  echo "==> restoring clangd compile-database filter"
  python3 "$workspace/scratch/clangd/filter-compile-commands.py"
}
trap refilter EXIT

cd "$workspace/pytorch"
python -m pip install -e . -v --no-build-isolation "$@"
