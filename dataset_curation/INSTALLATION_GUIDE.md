# 🔧 Installation & Setup Guide

## Prerequisites Check

Before starting, verify you have:

### 1. NVIDIA GPU (Required for Training)
```bash
# Check if GPU is detected
nvidia-smi

# Expected output: NVIDIA GPU info with CUDA version
```

If command not found, [install NVIDIA drivers](https://www.nvidia.com/Download/driverDetails.aspx)

---

## Installation Steps

### Step 1: Python Environment Setup

**Option A: Using Virtual Environment (Recommended)**
```bash
# Create virtual environment
python -m venv uad_env

# Activate it
# Windows:
uad_env\Scripts\activate

# Linux/Mac:
source uad_env/bin/activate

# Verify activation (should show "uad_env" in prompt)
```

**Option B: Using Conda**
```bash
# Create conda environment
conda create -n uad_env python=3.10 -y

# Activate
conda activate uad_env
```

### Step 2: Install Core Dependencies

**All at Once:**
```bash
pip install ultralytics==8.0.220 \
            torch==2.1.0 \
            torchvision==0.16.0 \
            torchaudio==2.1.0 \
            scikit-learn==1.3.2 \
            pandas==2.1.3 \
            matplotlib==3.8.2 \
            seaborn==0.13.0 \
            pillow==10.1.0 \
            numpy==1.24.3
```

**Or Install Step-by-Step:**

#### Install PyTorch with GPU Support

```bash
# For CUDA 11.8 (Most compatible)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1 (Newer GPUs)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify installation
python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

#### Install YOLO
```bash
pip install ultralytics
```

#### Install Data Science Stack
```bash
pip install scikit-learn pandas matplotlib seaborn
```

#### Install Image Processing
```bash
pip install pillow
```

### Step 3: Verify Installation

Run diagnostics script:

```python
# save as check_setup.py
import sys
import subprocess

def check_package(name, import_name=None):
    if import_name is None:
        import_name = name
    
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'installed')
        print(f"✓ {name}: {version}")
        return True
    except ImportError:
        print(f"✗ {name}: NOT INSTALLED")
        return False

def check_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  Compute Capability: {torch.cuda.get_device_capability()}")
            return True
        else:
            print("✗ GPU: CUDA NOT AVAILABLE")
            return False
    except Exception as e:
        print(f"✗ GPU Check Failed: {e}")
        return False

print("=" * 50)
print("Urban Asset Detection - Setup Verification")
print("=" * 50)

packages = [
    ('PyTorch', 'torch'),
    ('TorchVision', 'torchvision'),
    ('YOLO (Ultralytics)', 'ultralytics'),
    ('Scikit-learn', 'sklearn'),
    ('Pandas', 'pandas'),
    ('Matplotlib', 'matplotlib'),
    ('Seaborn', 'seaborn'),
    ('Pillow', 'PIL'),
    ('NumPy', 'numpy'),
]

print("\n📦 Package Installation Status:")
all_good = True
for name, import_name in packages:
    if not check_package(name, import_name):
        all_good = False

print("\n🔌 GPU Status:")
gpu_ok = check_gpu()

print("\n" + "=" * 50)
if all_good and gpu_ok:
    print("✅ All systems ready for training!")
else:
    print("❌ Some packages missing or GPU not available")
    print("   Run: pip install -r requirements.txt")
```

Run it:
```bash
python check_setup.py
```

---

## Using train_quick_start.py (Automatic)

The `train_quick_start.py` script automatically:
1. Checks for missing packages
2. Installs missing ones
3. Asks for training configuration
4. Starts training

```bash
python train_quick_start.py
# Follow the interactive prompts
```

---

## CUDA Installation (If Not Already Installed)

### Windows

1. **Download CUDA Toolkit:**
   - Visit: https://developer.nvidia.com/cuda-downloads
   - Select: Windows, x86_64, 11.8/12.1
   - Download installer

2. **Install:**
   - Run installer with default settings
   - Restart computer after installation

3. **Verify:**
   ```bash
   nvcc --version  # Should show CUDA version
   ```

### Linux (Ubuntu 22.04)
```bash
# Remove old CUDA
sudo apt remove cuda nvidia-* -y

# Install CUDA 12.1
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda-repo-ubuntu2204-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo apt-get update
sudo apt-get -y install cuda

# Add to PATH (add to ~/.bashrc)
export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
```

---

## Requirements.txt Installation

Create `requirements.txt`:

```
ultralytics==8.0.220
torch==2.1.0
torchvision==0.16.0
torchaudio==2.1.0
scikit-learn==1.3.2
pandas==2.1.3
matplotlib==3.8.2
seaborn==0.13.0
pillow==10.1.0
numpy==1.24.3
```

Install:
```bash
pip install -r requirements.txt
```

---

## Optional Dependencies

### For REST API Deployment
```bash
pip install fastapi==0.104.1 uvicorn==0.24.0 python-multipart==0.0.6
```

### For ONNX Export (Faster Inference)
```bash
pip install onnx==1.14.1 onnxruntime-gpu==1.16.3
```

### For TensorRT (NVIDIA GPU Optimization)
```bash
# Linux (Ubuntu)
sudo apt install libnvinfer8 libnvonnxparsers8 libnvinfer-plugin8

# Windows: Download from https://developer.nvidia.com/tensorrt
```

### For Jupyter Notebooks
```bash
pip install jupyter jupyterlab ipykernel
```

---

## Troubleshooting Installation

### Issue: "No module named 'torch'"
```bash
# Reinstall PyTorch
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue: "CUDA is not available"
```bash
# Check CUDA installation
nvcc --version

# Check if CUDA in PATH
python -c "import torch; print(torch.version.cuda)"

# If None, reinstall PyTorch:
pip install torch torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/cu118
```

### Issue: "Permission denied" on Linux
```bash
# Install with user flag
pip install --user ultralytics

# Or use venv
python -m venv uad_env
source uad_env/bin/activate
pip install ultralytics
```

### Issue: "Broken dependencies" or "Conflicting versions"
```bash
# Fresh install in virtual environment
deactivate  # if in venv
rm -rf uad_env
python -m venv uad_env
source uad_env/bin/activate  # or uad_env\Scripts\activate on Windows
pip install -r requirements.txt
```

---

## Environment Variables (Optional)

### For Performance Tuning

Create `.env` file in project root:
```
# PyTorch settings
TORCH_HOME=./models
OMP_NUM_THREADS=8
MKL_NUM_THREADS=8

# CUDA settings
CUDA_LAUNCH_BLOCKING=1
CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=0

# YOLO settings
YOLO_CONFIG_DIR=./yolo_config
```

Then load in scripts:
```python
from dotenv import load_dotenv
load_dotenv()
```

---

## Docker Setup (Optional)

For reproducible environment:

Create `Dockerfile`:
```dockerfile
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git

# Set working directory
WORKDIR /workspace

# Copy requirements
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Set command
CMD ["python", "train_quick_start.py"]
```

Build and run:
```bash
# Build image
docker build -t uad:latest .

# Run container (with GPU)
docker run --gpus all -it -v $(pwd):/workspace uad:latest

# Or on Windows
docker run --gpus all -it -v %cd%:/workspace uad:latest
```

---

## Version Compatibility

### Tested Configurations

| Component | Version | Status |
|-----------|---------|--------|
| Python | 3.9+ | ✓ Tested |
| PyTorch | 2.0+ | ✓ Tested |
| Ultralytics | 8.0+ | ✓ Tested |
| CUDA | 11.8+ | ✓ Tested |
| cuDNN | 8.6+ | ✓ Tested |

### Known Working Combinations

**Configuration 1 (Recommended for Latest GPUs)**
- Python 3.10
- PyTorch 2.1.0 + CUDA 12.1
- Ultralytics 8.0+
- NVIDIA Driver 535+

**Configuration 2 (Broad Compatibility)**
- Python 3.9
- PyTorch 2.0 + CUDA 11.8
- Ultralytics 8.0+
- NVIDIA Driver 520+

---

## Disk Space Requirements

| Component | Space |
|-----------|-------|
| Python environment | 2GB |
| PyTorch + dependencies | 5GB |
| unified_dataset/ | 3-4GB |
| yolo_dataset/ | 3-4GB |
| Training artifacts | 1-2GB |
| **Total** | **~20GB** |

Ensure you have **30GB free** for comfortable operation.

---

## Memory Requirements

### RAM (CPU Memory)
- **Minimum:** 8GB
- **Recommended:** 16GB
- **Optimal:** 32GB+

### VRAM (GPU Memory)
Per GPU:
- YOLOv8n: 2-3GB
- YOLOv8s: 3-4GB
- YOLOv8m: 6-7GB
- YOLOv8l: 10-12GB
- YOLOv8x: 16+GB

---

## Verification Checklist

- [ ] Python 3.9+ installed
- [ ] Virtual environment activated
- [ ] PyTorch installed with GPU support
- [ ] `nvidia-smi` shows GPU
- [ ] All packages from `requirements.txt` installed
- [ ] `python check_setup.py` shows all ✓
- [ ] Disk space >30GB available
- [ ] CUDA Toolkit installed (CUDA 11.8 or 12.1)
- [ ] Dataset files present in `datasets/` folder
- [ ] Can run: `python validate_unified_dataset.py`

---

## Quick Setup Script (Automated)

Save as `setup.sh` (Linux/Mac) or `setup.bat` (Windows):

**Linux/Mac (`setup.sh`):**
```bash
#!/bin/bash
set -e

echo "🔧 Urban Asset Detection - Setup Script"
echo "========================================"

# Create venv
python3 -m venv uad_env
source uad_env/bin/activate

# Install CUDA PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other packages
pip install -r requirements.txt

# Verify
python check_setup.py

echo "✅ Setup complete! Activate with: source uad_env/bin/activate"
```

**Windows (`setup.bat`):**
```batch
@echo off
echo.
echo 🔧 Urban Asset Detection - Setup Script
echo ========================================

REM Create venv
python -m venv uad_env
call uad_env\Scripts\activate.bat

REM Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

REM Install requirements
pip install -r requirements.txt

REM Verify
python check_setup.py

echo.
echo ✅ Setup complete! Activate with: uad_env\Scripts\activate.bat
pause
```

Run:
```bash
# Linux/Mac
chmod +x setup.sh
./setup.sh

# Windows
setup.bat
```

---

## Getting Help

### If Installation Fails

1. **Check Python version:**
   ```bash
   python --version  # Should be 3.9+
   ```

2. **Check GPU:**
   ```bash
   nvidia-smi
   ```

3. **Try fresh environment:**
   ```bash
   pip install --upgrade pip
   pip cache purge
   pip install -r requirements.txt --force-reinstall
   ```

4. **Check for conflicts:**
   ```bash
   pip check
   ```

5. **Try specific PyTorch version:**
   ```bash
   pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
   ```

---

## Next Steps

After successful installation:

1. Run verification: `python check_setup.py`
2. Validate data: `python validate_unified_dataset.py`
3. Start training: `python train_quick_start.py`

For detailed training guide, see: [README_TRAINING.md](README_TRAINING.md)

