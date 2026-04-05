#!/bin/bash
set -e 

echo "Starting Clean Dual-Venv Sync..."

# Clear the 'Active' environment inherited from your terminal
unset VIRTUAL_ENV

# 1. ARCHITECT (.venv-qam)
echo "------------------------------------------------"
echo "> Syncing ARCHITECT (.venv-qam)..."
UV_PROJECT_ENVIRONMENT=.venv-qam uv sync --extra modeling

# 2. ENGINE (.venv-gpu)
echo "------------------------------------------------"
echo "> Syncing ENGINE (.venv-gpu)..."
UV_PROJECT_ENVIRONMENT=.venv-gpu uv sync --extra gpu

echo "------------------------------------------------"
echo "Environments Synchronized!"