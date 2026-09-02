#!/usr/bin/env bash

# Fail fast, fail early, fail loud.
set -euo pipefail

workspace_dir="$PWD"

echo "==> Upgrading packages"
sudo apt-get update
sudo apt-get upgrade --yes

echo "==> Installing PyTorch's build dependencies"
sudo apt-get install --yes build-essential cmake ninja-build python3 python3-pip python3-dev python3-venv libopenblas-dev

echo "==> Installing ripgrep"
sudo apt-get install --yes ripgrep

echo "==> Installing Starship"
sudo apt-get install starship --yes
grep -qxF 'eval "$(starship init bash)"' "$HOME/.bashrc" || echo 'eval "$(starship init bash)"' >> "$HOME/.bashrc"
grep -qxF 'eval "$(starship init zsh)"' "$HOME/.zshrc" || echo 'eval "$(starship init zsh)"' >> "$HOME/.zshrc"
mkdir -p "$HOME/.config"
if [ ! -f "$HOME/.config/starship.toml" ]; then
    cp "$workspace_dir/.devcontainer/starship.toml" "$HOME/.config/starship.toml"
fi

echo "==> Installing Codex and Claude Code"
npm config set allow-scripts=@anthropic-ai/claude-code --location=user
npm install -g @openai/codex @anthropic-ai/claude-code

echo "==> Installing uv"
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "==> Initializing the PyTorch submodule"
if [ ! -e "$workspace_dir/pytorch/.git" ]; then
    git -C "$workspace_dir" submodule sync -- pytorch
    git -C "$workspace_dir" submodule update --init pytorch
fi

echo "==> Fetching PyTorch's git submodules"
cd "$workspace_dir/pytorch"
git submodule sync --recursive
git submodule update --init --recursive

echo "==> Setting up PyTorch's Python venv"
uv venv --allow-existing
source .venv/bin/activate
uv pip install -r requirements.txt

echo "==> Setting up lintrunner"
make setup-lint

echo "==> Configuring the build environment"
for shell_rc in "$HOME/.bashrc" "$HOME/.zshrc"; do
    grep -qxF 'export CUDA_HOME=/usr/local/cuda' "$shell_rc" || echo 'export CUDA_HOME=/usr/local/cuda' >> "$shell_rc"
    grep -qxF 'export PATH="$CUDA_HOME/bin:$PATH"' "$shell_rc" || echo 'export PATH="$CUDA_HOME/bin:$PATH"' >> "$shell_rc"
    grep -qxF 'export CMAKE_PREFIX_PATH=/usr/local' "$shell_rc" || echo 'export CMAKE_PREFIX_PATH=/usr/local' >> "$shell_rc"
    grep -qxF 'export LDFLAGS="-L/usr/local/cuda/lib64/ $LDFLAGS"' "$shell_rc" || echo 'export LDFLAGS="-L/usr/local/cuda/lib64/ $LDFLAGS"' >> "$shell_rc"
    # Cap parallel build jobs so PyTorch compiles don't exhaust RAM.
    grep -qxF 'export MAX_JOBS=8' "$shell_rc" || echo 'export MAX_JOBS=8' >> "$shell_rc"
    grep -qxF "source $workspace_dir/pytorch/.venv/bin/activate" "$shell_rc" || echo "source $workspace_dir/pytorch/.venv/bin/activate" >> "$shell_rc"
done

# `scratch/` deliberately has no environment of its own: experiments there must
# exercise the PyTorch built from `pytorch/`, which the venv activated above
# provides. Installing `torch` for `scratch/` would pull a release from PyPI and
# silently shadow the local build.
