#!/bin/bash
# CPU-only installation script for research-agent
# This script installs PyTorch CPU-only first, then installs the rest of the requirements

set -e  # Exit on error

echo "Installing PyTorch CPU-only version..."
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo "Installing remaining requirements..."
uv pip install -r requirements-cpu.txt

echo ""
echo "Installation complete!"
echo ""
echo "Verifying CPU-only installation..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

