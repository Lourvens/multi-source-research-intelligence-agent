# CPU-Only Installation Guide

This guide explains how to install the project dependencies without GPU/NVIDIA packages.

## Problem

By default, `sentence-transformers` will pull in PyTorch with CUDA support, which includes NVIDIA GPU dependencies that you may not want on a CPU-only system.

## Solution: Install PyTorch CPU-First

### Method 1: Use the CPU-only requirements file (Recommended)

```bash
pip install -r requirements-cpu.txt
```

This file installs PyTorch CPU-only first, then installs the rest of the packages.

### Method 2: Manual two-step installation

**Step 1:** Install PyTorch CPU-only first:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Step 2:** Install the rest of the requirements:
```bash
pip install -r requirements.txt
```

### Method 3: Using conda (Alternative)

```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
pip install -r requirements.txt
```

## Verify CPU-Only Installation

After installation, verify that PyTorch is CPU-only:

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")  # Should be False
```

If `torch.cuda.is_available()` returns `False`, you've successfully installed the CPU-only version.

## Why This Works

When you install `sentence-transformers` after PyTorch CPU-only is already installed, pip will detect that PyTorch is already satisfied and won't pull in the CUDA version. The `--index-url` flag tells pip to use PyTorch's CPU-only wheel repository, ensuring no GPU dependencies are installed.

## Notes

- The CPU-only version is sufficient for most use cases, especially for inference
- CPU inference will be slower than GPU, but works fine for development and smaller datasets
- If you later need GPU support, you can uninstall PyTorch and reinstall with CUDA support


