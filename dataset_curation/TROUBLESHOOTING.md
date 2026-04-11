# 🔧 Troubleshooting & Advanced Guide

## Problem Diagnosis Flowchart

```
Training Issues?
├─ GPU Related?
│  ├─ CUDA not found → [Fix #1]
│  ├─ Out of memory → [Fix #2]
│  └─ GPU not detected → [Fix #3]
├─ Data Related?
│  ├─ data.yaml not found → [Fix #4]
│  ├─ Image/label mismatch → [Fix #5]
│  └─ No valid images → [Fix #6]
├─ Model Training?
│  ├─ Very slow training → [Fix #7]
│  ├─ Loss not decreasing → [Fix #8]
│  ├─ Accuracy low → [Fix #9]
│  └─ Training crashes → [Fix #10]
└─ Output Related?
   ├─ Metrics not generated → [Fix #11]
   └─ Confusion matrix empty → [Fix #12]
```

---

## GPU Issues

### ⚠️ Fix #1: CUDA Not Found

**Error Message:**
```
RuntimeError: CUDA is not available
OR
No CUDA GPUs detected
```

**Diagnosis:**
```bash
# Check GPU availability
nvidia-smi  # If command not found, GPU driver not installed

# Check PyTorch CUDA support
python -c "import torch; print(torch.cuda.is_available())"
```

**Solutions:**

1. **Install NVIDIA GPU Drivers:**
   - Visit: https://www.nvidia.com/Download/driverDetails.aspx
   - Select your GPU model
   - Download and install latest driver

2. **Install CUDA Toolkit (Windows):**
   ```bash
   # Download from: https://developer.nvidia.com/cuda-downloads
   # Select: Windows, x86_64, 11.8 or 12.1
   # Install with default settings
   ```

3. **Reinstall PyTorch with CUDA:**
   ```bash
   # Uninstall current
   pip uninstall torch torchvision torchaudio -y
   
   # Install for CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # OR for CUDA 12.1
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

4. **Verify Installation:**
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
   ```

---

### ⚠️ Fix #2: Out of Memory (OOM)

**Error Messages:**
```
RuntimeError: CUDA out of memory
OR
torch.cuda.OutOfMemoryError
```

**Diagnosis:**
```bash
# Check GPU memory usage
nvidia-smi

# Check available memory
python -c "import torch; print(f'Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"
```

**Solutions (in order of effectiveness):**

1. **Reduce Batch Size** (Most effective):
   ```bash
   # Instead of --batch 32
   python train_yolo_model.py \
       --model yolov8m \
       --batch 16  # Or even 8
   ```

2. **Use Smaller Model:**
   ```bash
   # Instead of yolov8m
   python train_yolo_model.py \
       --model yolov8s  # Or yolov8n
       --batch 32
   ```

3. **Reduce Image Size:**
   ```bash
   python train_yolo_model.py \
       --model yolov8m \
       --batch 32 \
       --imgsz 512  # Instead of 640
   ```

4. **Disable Other GPU Tasks:**
   ```bash
   # Close other GPU-intensive apps (Chrome, Discord, etc.)
   # Restart Python kernel
   ```

5. **Use Mixed Precision** (Advanced):
   ```python
   # Edit train_yolo_model.py, in train() method:
   model = YOLO(self.model_name)
   model.train(
       data=self.data_yaml,
       epochs=self.epochs,
       batch=self.batch,
       device=self.device,
       half=True  # Add this line
   )
   ```

**GPU Memory Estimation:**
| Model | 640x640 Batch 32 | 512x512 Batch 32 |
|-------|------------------|-----------------|
| yolov8n | 2-3 GB | 1-2 GB |
| yolov8s | 3-4 GB | 2-3 GB |
| yolov8m | 6-7 GB | 4-5 GB |
| yolov8l | 10-12 GB | 7-9 GB |
| yolov8x | 16+ GB | 12-14 GB |

---

### ⚠️ Fix #3: GPU Device Not Found / Multi-GPU Issues

**Error:**
```
RuntimeError: Invalid device ordinal
OR
No GPU device found
```

**Solutions:**

1. **Specify Correct Device ID:**
   ```bash
   # List available GPUs
   nvidia-smi
   
   # Use device 0 (first GPU)
   python train_yolo_model.py \
       --device 0  # Must be 0 if only one GPU
   ```

2. **For Multi-GPU Training:**
   ```python
   # Edit train_yolo_model.py:
   # Change
   model.train(device=self.device, ...)
   # To
   model.train(device=[0, 1], ...)  # Multiple GPUs
   ```

3. **Use CPU (Fallback - Slow):**
   ```bash
   python train_yolo_model.py \
       --device cpu
   ```

---

## Data Issues

### ⚠️ Fix #4: data.yaml Not Found

**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'yolo_dataset/data.yaml'
```

**Solution:**
```bash
# Ensure YOLO dataset was created
python create_yolo_dataset.py

# Verify file exists
ls -la yolo_dataset/data.yaml  # Linux/Mac
dir yolo_dataset\data.yaml     # Windows

# Check file contents
cat yolo_dataset/data.yaml     # Should show paths and class names
```

**Expected content of data.yaml:**
```yaml
path: C:\...\yolo_dataset
train: images/train
val: images/val
test: images/test

nc: 5
names: ['building', 'road', 'water_body', 'vegetation', 'vehicle']
```

---

### ⚠️ Fix #5: Image-Label Mismatch

**Error:**
```
RuntimeError: Image and label mismatch
OR
Images or labels not found in dataset
```

**Diagnosis:**
```bash
# Run validation script
python validate_unified_dataset.py

# Check directory structure
ls -la yolo_dataset/images/train/     # Images present?
ls -la yolo_dataset/labels/train/     # Labels present?
```

**Solutions:**

1. **Verify File Counts Match:**
   ```bash
   # Count images
   ls -1 yolo_dataset/images/train/ | wc -l
   
   # Count labels
   ls -1 yolo_dataset/labels/train/ | wc -l
   
   # Should be equal!
   ```

2. **Check for Special Characters in Filenames:**
   ```bash
   # Look for problematic names
   ls yolo_dataset/images/train/ | grep -E "[^a-zA-Z0-9._-]"
   ```

3. **Regenerate YOLO Dataset:**
   ```bash
   rm -rf yolo_dataset
   python create_yolo_dataset.py
   ```

---

### ⚠️ Fix #6: No Valid Images in Dataset

**Error:**
```
No images found in training set
OR
Empty dataset error
```

**Diagnosis:**
```bash
# Check if unified_dataset exists
ls -la unified_dataset/

# Count images in unified dataset
find unified_dataset -name "*.jpg" -o -name "*.png" | wc -l

# Check if YOLO dataset has files
find yolo_dataset -name "*.jpg" -o -name "*.png" | wc -l
```

**Solutions:**

1. **Run Dataset Creation:**
   ```bash
   python sample_and_convert_datasets.py
   python validate_unified_dataset.py
   python create_yolo_dataset.py
   ```

2. **Check Unified Dataset Structure:**
   ```
   unified_dataset/
   ├── landcover/
   │   ├── images/ (should have 800+ .jpg files)
   │   └── labels/ (should have 800+ .txt files)
   ├── road_extraction/
   ├── dota/
   └── semantic_buildings/
   ```

---

## Training Issues

### ⚠️ Fix #7: Training Very Slow

**Symptoms:**
- Each epoch takes >30 min even on GPU
- GPU utilization low (< 50%)
- CPU at 100%

**Diagnosis:**
```bash
# Monitor during training
nvidia-smi -l 1  # Refresh every second - watch GPU %

# Check if CPU bound
# Windows: Task Manager → Performance → CPU
# Linux: htop
```

**Solutions (in order):**

1. **Ensure Using GPU (not CPU):**
   ```python
   # Add to train_yolo_model.py before training
   import torch
   print(f"Using device: {torch.cuda.get_device_name(self.device)}")
   ```

2. **Increase Workers for Data Loading:**
   ```python
   # Edit train_yolo_model.py, in train() method:
   model.train(
       data=self.data_yaml,
       epochs=self.epochs,
       batch=self.batch,
       device=self.device,
       workers=8  # Increase from default 4
   )
   ```

3. **Increase Batch Size** (if GPU memory allows):
   ```bash
   python train_yolo_model.py \
       --model yolov8m \
       --batch 64  # Increase from 32
   ```

4. **Use Smaller Image Size:**
   ```bash
   python train_yolo_model.py \
       --imgsz 512  # From 640
   ```

5. **Monitor System Resources:**
   ```bash
   # Windows PowerShell (continuous monitoring)
   while($true) { nvidia-smi; Start-Sleep -Seconds 2; cls }
   ```

---

### ⚠️ Fix #8: Loss Not Decreasing

**Symptom:**
- Loss values stay constant (e.g., ~10.5 for first 10+ epochs)
- mAP stays near 0
- Model not learning

**Diagnosis:**

```python
# Check if data is being loaded
import sys
sys.path.insert(0, 'yolo_dataset')
from pathlib import Path
images = list(Path('yolo_dataset/images/train').glob('*.jpg'))
print(f"Found {len(images)} training images")
```

**Solutions:**

1. **Check Data is Valid:**
   ```bash
   # Run validation
   python validate_unified_dataset.py
   ```

2. **Restart from Scratch:**
   ```bash
   # Delete results folder if it exists
   rm -rf training_results
   
   # Run with more verbose output
   python train_yolo_model.py \
       --model yolov8n \
       --epochs 50 \
       --batch 64
   ```

3. **Increase Learning Rate** (Advanced):
   ```python
   # Edit train_yolo_model.py:
   model.train(
       data=self.data_yaml,
       epochs=self.epochs,
       batch=self.batch,
       device=self.device,
       lr0=0.01,  # Increase from default 0.01
       lrf=0.1    # Final LR ratio
   )
   ```

4. **Use Different Architecture:**
   ```bash
   python train_yolo_model.py \
       --model yolov8n  # Start with nano model
       --epochs 100
   ```

---

### ⚠️ Fix #9: Low Accuracy

**Symptom:**
- mAP50 < 0.30
- Precision/Recall low
- Model predicts same class for everything

**Causes & Solutions:**

1. **Class Imbalance** (Vehicle = 65% of data):
   ```python
   # Add class weights to train_yolo_model.py
   model.train(
       data=self.data_yaml,
       epochs=self.epochs,
       batch=self.batch,
       device=self.device,
       class_weights=[1.0, 10.0, 10.0, 3.0, 0.3]  # Adjust minority classes
   )
   ```

2. **Too Few Epochs:**
   ```bash
   python train_yolo_model.py \
       --model yolov8m \
       --epochs 150  # Or 200
   ```

3. **Model Too Small:**
   ```bash
   python train_yolo_model.py \
       --model yolov8l  # From yolov8m
   ```

4. **Data Issues:**
   ```bash
   # Ensure training data has enough objects
   python validate_unified_dataset.py
   ```

5. **Image Size Mismatch:**
   ```bash
   python train_yolo_model.py \
       --imgsz 640  # Ensure it matches training
   ```

---

### ⚠️ Fix #10: Training Crashes Mid-Training

**Error Examples:**
```
RuntimeError: CUDA ran out of memory
OR
Segmentation fault
OR
KeyboardInterrupt
```

**Solutions:**

1. **For CUDA OOM:**
   - See [Fix #2: Out of Memory](#fix-2-out-of-memory-oom)

2. **Enable Automatic Mixed Precision:**
   ```python
   # Edit train_yolo_model.py:
   model.train(
       data=self.data_yaml,
       epochs=self.epochs,
       batch=self.batch,
       device=self.device,
       half=True  # Use FP16 instead of FP32
   )
   ```

3. **Resume Training from Checkpoint:**
   ```python
   # If model.pt exists from previous training
   model = YOLO('training_results/model/weights/last.pt')
   model.train(resume=True)
   ```

4. **Increase Early Stopping Patience:**
   ```bash
   python train_yolo_model.py \
       --model yolov8m \
       --patience 50  # From default 20
   ```

---

## Output Issues

### ⚠️ Fix #11: Metrics Not Generated

**Problem:**
- training_metrics.png not created
- confusion_matrix.png missing

**Diagnosis:**
```bash
# Check if results folder was created
ls -la training_results/
ls -la training_results/YYYYMMDD_HHMMSS/
```

**Solutions:**

1. **Check Training Completed:**
   ```bash
   # Verify training finished without errors
   # Look for "Training complete" message in console
   ```

2. **Manual Metric Generation:**
   ```python
   from train_yolo_model import YOLOTrainer
   
   trainer = YOLOTrainer(
       data_yaml='yolo_dataset/data.yaml',
       model_name='yolov8m',
       epochs=100,
       batch=32,
       device=0
   )
   
   # Generate metrics from existing model
   trainer.results_df = trainer.model.results  # Requires results.csv
   trainer.plot_training_metrics()
   ```

---

### ⚠️ Fix #12: Confusion Matrix Empty

**Problem:**
- Confusion matrix generated but shows no data
- All zeros

**Causes & Solutions:**

1. **Model didn't evaluate on test set:**
   ```bash
   # Run evaluation manually
   python -c "
   from train_yolo_model import YOLOTrainer
   trainer = YOLOTrainer('yolo_dataset/data.yaml')
   trainer.model = trainer.load_model()
   results = trainer.evaluate_test_set()
   print(results)
   "
   ```

2. **Test set doesn't exist:**
   ```bash
   # Check test set
   ls yolo_dataset/images/test/
   ls yolo_dataset/labels/test/
   ```

3. **Wrong test set path in data.yaml:**
   ```yaml
   # Check this in yolo_dataset/data.yaml
   test: images/test  # Must exist!
   ```

---

## Advanced Optimization

### Custom Training Configuration

```python
# Edit train_yolo_model.py, modify train() method:

model.train(
    data=self.data_yaml,
    epochs=self.epochs,
    batch=self.batch,
    device=self.device,
    
    # Augmentation
    augment=True,
    hsv_h=0.015,      # HSV hue aug
    hsv_s=0.7,        # HSV saturation
    hsv_v=0.4,        # HSV value
    degrees=10.0,     # Rotation
    translate=0.1,    # Translation
    scale=0.5,        # Scale
    flipud=0.5,       # Flip vertically
    fliplr=0.5,       # Flip horizontally
    mosaic=1.0,       # Mosaic aug
    
    # Regularization
    weight_decay=5e-4,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    
    # Learning rate
    lr0=0.01,         # Initial LR
    lrf=0.01,         # Final LR ratio
    
    # Class weights (for imbalance)
    class_weights=[1.0, 10.0, 10.0, 3.0, 0.3],
    
    # Early stopping
    patience=20,
    
    # Data loading
    workers=8,        # Data loader workers
    
    # Optimization
    optimizer='SGD',  # Or 'Adam'
    momentum=0.937,
    
    # Format
    imgsz=640,
    save_period=10,
    val=True,
    imgsz=640,
)
```

### Class Balancing Strategy

```python
# For highly imbalanced data like ours:
# Vehicle: 65.6%, Building: 17.3%, Vegetation: 11.6%, 
# Water Body: 2.7%, Road: 2.6%

class_weights = [
    1.0,    # Building - present (25% reduction from vehicle weight)
    4.0,    # Road - very rare (upgrade this)
    4.0,    # Water Body - very rare (upgrade this)
    2.0,    # Vegetation - moderate (slight upgrade)
    0.3     # Vehicle - dominant (downweight heavily)
]

# Calculate inverse frequency weights
total_objects = 96915
class_counts = [17039, 2557, 2636, 11218, 63465]
class_weights = [total_objects / (5 * count) for count in class_counts]
# Results: [1.14, 7.54, 7.29, 1.72, 0.31]
```

---

## Performance Tuning Checklist

- [ ] GPU available and detected: `nvidia-smi`
- [ ] Sufficient GPU memory for batch size
- [ ] All required packages installed: `pip list | grep -i torch`
- [ ] YOLO dataset created: `ls yolo_dataset/data.yaml`
- [ ] Data validated: `python validate_unified_dataset.py`
- [ ] Batch size optimized for GPU memory
- [ ] Image size appropriate (640 for most cases)
- [ ] Initial learning rate tested: start with
lr0=0.01`
- [ ] Epochs set reasonably: 100+ for good results
- [ ] Early stopping patience set: 20-50
- [ ] Workers increased for faster data loading: 4-8
- [ ] Output directory writable
- [ ] Disk space available (>10GB for training artifacts)

---

## Emergency Recovery

### If Everything Breaks

```bash
# 1. Start fresh with validation
python validate_unified_dataset.py

# 2. Recreate YOLO dataset
rm -rf yolo_dataset
python create_yolo_dataset.py

# 3. Test training with minimal config
python train_yolo_model.py \
    --model yolov8n \
    --epochs 5 \
    --batch 64 \
    --device 0

# 4. If that works, scale up
python train_yolo_model.py \
    --model yolov8m \
    --epochs 100 \
    --batch 32 \
    --device 0
```

### Debug Mode

```python
# Add to train_yolo_model.py for detailed logging:
import logging
logging.basicConfig(level=logging.DEBUG)

# In train() method:
print(f"GPU Device: {torch.cuda.current_device()}")
print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.1f} / {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"Data path: {self.data_yaml}")
print(f"Model: {self.model_name}")
print(f"Config: {self.epochs} epochs, batch {self.batch}")
```

---

## When to Contact Support

Document for support:
- Error message (full traceback)
- Command you ran
- Your GPU model: `nvidia-smi`
- PyTorch version: `python -c "import torch; print(torch.__version__)"`
- Dataset status: `python validate_unified_dataset.py` output
- System info: GPU, CPU, RAM

