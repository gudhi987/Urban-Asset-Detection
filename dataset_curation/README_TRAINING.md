# Urban Asset Detection - Complete Machine Learning Pipeline

A comprehensive pipeline for converting multiple datasets into a unified format and training a YOLO model for multi-class object detection.

## 📊 Project Overview

**Objective:** Train a YOLO model to detect 5 urban asset classes:
- 🏢 Building
- 🛣️ Road  
- 💧 Water Body
- 🌳 Vegetation
- 🚗 Vehicle

**Datasets Integrated:**
1. **Deep Globe Land Cover** (803 images)
2. **Deep Globe Road Extraction** (1,000 images sampled)
3. **DOTA** (1,128 images, 760 with vehicle annotations)
4. **Semantic Buildings** (2,000 images sampled)

**Total:** 4,560 images with 96,915 objects

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install ultralytics torch torchvision scikit-learn pandas matplotlib seaborn pillow
```

### Step 1: Create YOLO Dataset (Already Done ✓)
```bash
python create_yolo_dataset.py \
    --unified-dir unified_dataset \
    --output-dir yolo_dataset \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --test-ratio 0.1
```

### Step 2: Train Model (Recommended)
**Option A: Interactive Quick Start**
```bash
python train_quick_start.py
# Follow the prompts to select training configuration
```

**Option B: Direct Command**
```bash
python train_yolo_model.py \
    --data yolo_dataset/data.yaml \
    --model yolov8m \
    --epochs 100 \
    --batch 32 \
    --device 0
```

**Model Options:**
- `yolov8n` - Nano (Fast, low memory) - Recommended for testing
- `yolov8s` - Small
- `yolov8m` - Medium (Recommended for production)
- `yolov8l` - Large
- `yolov8x` - Extra Large (Highest accuracy)

---

## 📁 Directory Structure

```
Urban-Asset-Detection/
├── dataset_curation/
│   ├── unified_dataset/              # Merged datasets
│   │   ├── dota/
│   │   ├── landcover/
│   │   ├── road_extraction/
│   │   └── semantic_buildings/
│   │
│   ├── yolo_dataset/                 # Final YOLO format
│   │   ├── images/
│   │   │   ├── train/  (3,648 images)
│   │   │   ├── val/    (456 images)
│   │   │   └── test/   (456 images)
│   │   ├── labels/
│   │   │   ├── train/  (3,648 labels)
│   │   │   ├── val/    (456 labels)
│   │   │   └── test/   (456 labels)
│   │   └── data.yaml   # YOLO config
│   │
│   ├── training_results/             # Training outputs
│   │   └── YYYYMMDD_HHMMSS/
│   │       ├── model/
│   │       │   ├── weights/
│   │       │   │   ├── best.pt
│   │       │   │   └── last.pt
│   │       │   └── results.csv
│   │       ├── training_metrics.png
│   │       ├── confusion_matrix.png
│   │       ├── summary_statistics.csv
│   │       └── TRAINING_REPORT.md
│   │
│   └── Scripts:
│       ├── sample_and_convert_datasets.py
│       ├── create_yolo_dataset.py
│       ├── train_yolo_model.py
│       ├── train_quick_start.py
│       ├── validate_unified_dataset.py
│       └── visualize_unified_dataset.py

```

---

## 📊 Dataset Characteristics

### Class Distribution
| Class | Count | % | Train | Val | Test |
|-------|-------|---|-------|-----|------|
| Building | 17,039 | 17.3% | 13,736 | 1,706 | 1,597 |
| Road | 2,557 | 2.6% | 2,026 | 275 | 256 |
| Water Body | 2,636 | 2.7% | 2,106 | 332 | 198 |
| Vegetation | 11,218 | 11.6% | 8,632 | 1,151 | 1,435 |
| Vehicle | 63,465 | 65.6% | 52,072 | 6,860 | 4,533 |
| **Total** | **96,915** | **100%** | **78,572** | **10,324** | **8,019** |

### Data Split
- **Train:** 3,648 images (80%)
- **Validation:** 456 images (10%)
- **Test:** 456 images (10%)
- **Total:** 4,560 images

---

## 🎯 Training Output

The training script generates:

### 1. **Training Metrics Dashboard** (`training_metrics.png`)
- Training vs Validation Loss
- Box Loss per epoch
- Class Loss per epoch
- mAP50 progression
- Final metrics comparison

### 2. **Confusion Matrix** (`confusion_matrix.png`)
- True vs Predicted classifications
- Heatmap visualization
- Per-class accuracy analysis

### 3. **Training Report** (`TRAINING_REPORT.md`)
- Configuration details
- Test set metrics (mAP50, mAP50-95, Precision, Recall)
- Dataset information
- Performance insights
- Recommendations

### 4. **Summary Statistics** (`summary_statistics.csv`)
- Dataset split breakdown
- Image and label counts per split

### 5. **Trained Models** (`model/weights/`)
- `best.pt` - Best model (based on mAP)
- `last.pt` - Last checkpoint

---

## 🔧 Training Configuration Examples

### Quick Testing
```bash
python train_yolo_model.py \
    --data yolo_dataset/data.yaml \
    --model yolov8n \
    --epochs 10 \
    --batch 64 \
    --device 0
```

### Standard Production
```bash
python train_yolo_model.py \
    --data yolo_dataset/data.yaml \
    --model yolov8m \
    --epochs 100 \
    --batch 32 \
    --device 0
```

### High-Accuracy
```bash
python train_yolo_model.py \
    --data yolo_dataset/data.yaml \
    --model yolov8l \
    --epochs 150 \
    --batch 16 \
    --device 0
```

---

## 📈 Expected Performance (YOLOv8m on this dataset)

**Typical Results After 100 Epochs:**
- **mAP50:** 0.50-0.60
- **mAP50-95:** 0.30-0.40
- **Precision:** 0.55-0.65
- **Recall:** 0.45-0.55

*Note: High vehicle class (65% of data) influences overall metrics*

---

## 🛠️ Inference

### Load and Use Trained Model
```python
from ultralytics import YOLO

# Load trained model
model = YOLO('training_results/YYYYMMDD_HHMMSS/model/weights/best.pt')

# Predict on image
results = model.predict('path/to/image.jpg', conf=0.5)

# Predict on video
results = model.predict('path/to/video.mp4', conf=0.5)

# Get results
for result in results:
    print(result.boxes)  # Bounding boxes
    print(result.probs)  # Class probabilities
```

---

## 📝 Class Mapping

The unified dataset uses consistent class IDs:
- **0:** building
- **1:** road
- **2:** water_body
- **3:** vegetation
- **4:** vehicle

---

## ⚠️ Important Notes

### Class Imbalance
- Vehicle class dominates (65% of data)
- Consider using class weights for balanced training:
  ```python
  # In training config
  weights = [1.0, 10.0, 10.0, 3.0, 0.3]  # Adjust per class importance
  ```

### GPU Requirements
- **Minimum:** NVIDIA GPU with 4GB VRAM (YOLOv8n)
- **Recommended:** 8GB+ VRAM (YOLOv8m)
- **Optimal:** 12GB+ VRAM (YOLOv8l)

### Training Time
- **YOLOv8n, 50 epochs:** ~30 min
- **YOLOv8m, 100 epochs:** ~2-3 hours
- **YOLOv8l, 150 epochs:** ~6-8 hours

---

## 🔍 Troubleshooting

**Issue:** CUDA not available
- **Solution:** Install CUDA Toolkit and cuDNN for your GPU

**Issue:** Out of memory errors
- **Solution:** Reduce batch size or use smaller model (yolov8n/s)

**Issue:** Low accuracy on minority classes
- **Solution:** Use augmentation or class weighting

**Issue:** Training is slow
- **Solution:** Enable GPU, use smaller model, reduce image size

---

## 📚 References

- **YOLOv8 Docs:** https://docs.ultralytics.com/
- **YOLO GitHub:** https://github.com/ultralytics/ultralytics
- **Dataset Sources:**
  - DOTA: https://captain-whu.github.io/DOTA/
  - DeepGlobe: http://deepglobe.org/
  - Semantic Buildings: Various sources

---

## 📄 License

This project integrates multiple public datasets. Please refer to individual dataset licenses for usage rights.

---

## ✅ Checklist

- [x] Unified multiple datasets (4 sources)
- [x] Created YOLO format dataset
- [x] Implemented 80/10/10 train/val/test split
- [x] Built complete training pipeline
- [x] GPU support enabled
- [x] Metrics collection and visualization
- [ ] Train model (Your turn!)
- [ ] Evaluate on test set
- [ ] Deploy to production

---

**Start training:** `python train_quick_start.py`
