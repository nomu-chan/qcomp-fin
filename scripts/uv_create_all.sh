#!/bin/bash
set -e 

echo "Starting Creation of Dual-Venv Sync..."

# Clear the 'Active' environment inherited from your terminal
unset VIRTUAL_ENV

# 1. ARCHITECT (.venv-qam)
echo "------------------------------------------------"
echo "> Creating ARCHITECT (.venv-qam)..."
uv venv .venv-qam --python 3.12
UV_PROJECT_ENVIRONMENT=.venv-qam uv sync --extra modeling

# 2. ENGINE (.venv-gpu)
echo "------------------------------------------------"
echo "> Creating ENGINE (.venv-gpu)..."
uv venv .venv-gpu --python 3.12
UV_PROJECT_ENVIRONMENT=.venv-gpu uv sync --extra gpu

echo "------------------------------------------------"
echo "Environments Made!"