# 🚀 Quick Reference Guide

## All Available Scripts

### 1. **Data Curation Pipeline**

#### `sample_and_convert_datasets.py`
Converts 4 raw datasets to unified format.

**What it does:**
- Extracts images from Deep Globe Land Cover, Road, DOTA, Semantic Buildings
- Maps classes to 5-class schema: building, road, water_body, vegetation, vehicle
- Converts to YOLO normalized format (class_id cx cy w h in [0,1])
- Handles image suffix issues (e.g., `_sat` suffix)

**Usage:**
```bash
python sample_and_convert_datasets.py
```

**Output:**
```
unified_dataset/
├── landcover/      (800 images)
├── road_extraction/ (1,000 images)
├── dota/           (760 images)
└── semantic_buildings/ (2,000 images)
```

**Key Fix:** Strips `_sat` suffix from satellite imagery before label matching

---

#### `validate_unified_dataset.py`
Quality checks the unified dataset.

**What it validates:**
- All images have corresponding labels
- YOLO format compliance
- Bounding box normalization (0-1 range)
- Class IDs in valid range (0-4)
- Image-label correspondence

**Usage:**
```bash
python validate_unified_dataset.py
```

**Expected Output:**
```
✓ Total images: 4,560
✓ Total objects: 96,915
✓ Validation errors: 0
✓ All checks passed!
```

---

#### `visualize_unified_dataset.py`
Displays sample images with annotations.

**What it shows:**
- Random sample of 5 images from each dataset
- Bounding boxes overlaid on images
- Class labels with colors
- Side-by-side comparison

**Usage:**
```bash
python visualize_unified_dataset.py
```

**Output:** Plot windows showing annotated images

---

### 2. **YOLO Dataset Creation**

#### `create_yolo_dataset.py`
Converts unified dataset to YOLO train/val/test structure.

**What it does:**
- Collects all images and labels from unified_dataset/
- Random shuffle and 80/10/10 split
- Creates directory structure for training
- Generates data.yaml config file

**Usage:**
```bash
python create_yolo_dataset.py \
    --unified-dir unified_dataset \
    --output-dir yolo_dataset \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --test-ratio 0.1
```

**Output:**
```
yolo_dataset/
├── images/
│   ├── train/  (3,648 images)
│   ├── val/    (456 images)
│   └── test/   (456 images)
├── labels/
│   ├── train/  (3,648 labels)
│   ├── val/    (456 labels)
│   └── test/   (456 labels)
└── data.yaml   (YOLO config)
```

---

### 3. **Training & Evaluation**

#### `train_quick_start.py`
Interactive training launcher with dependency management.

**What it does:**
- Checks for required packages (auto-installs if missing)
- Presents training preset options
- Calls train_yolo_model.py with selected config

**Usage:**
```bash
python train_quick_start.py
```

**Menu Options:**
```
1. Quick     → yolov8n, 50 epochs, batch 64
2. Standard  → yolov8m, 100 epochs, batch 32
3. Advanced  → yolov8l, 150 epochs, batch 16
4. Custom    → Specify your own parameters
```

---

#### `train_yolo_model.py`
Complete GPU-accelerated training pipeline.

**Core Class:** `YOLOTrainer`

**Methods:**
- `train()` - Train model on full dataset
- `evaluate_test_set()` - Calculate metrics (mAP, precision, recall)
- `generate_confusion_matrix()` - Create confusion matrix from test predictions
- `plot_training_metrics()` - Generate 3x2 dashboard plot
- `generate_report()` - Create markdown report with insights
- `create_summary_table()` - Output dataset statistics

**Direct Usage:**
```bash
python train_yolo_model.py \
    --data yolo_dataset/data.yaml \
    --model yolov8m \
    --epochs 100 \
    --batch 32 \
    --device 0
```

**Command-line Arguments:**
```
--data        Path to data.yaml (required)
--model       Model size: yolov8n/s/m/l/x (default: yolov8m)
--epochs      Number of training epochs (default: 100)
--batch       Batch size (default: 32)
--device      GPU device ID (default: 0)
--patience    Early stopping patience (default: 20)
--imgsz       Image size (default: 640)
```

**Output:**
```
training_results/YYYYMMDD_HHMMSS/
├── model/
│   ├── weights/
│   │   ├── best.pt
│   │   └── last.pt
│   └── results.csv
├── training_metrics.png
├── confusion_matrix.png
├── summary_statistics.csv
└── TRAINING_REPORT.md
```

---

## 📊 Output Files Explained

### `training_metrics.png`
3x2 dashboard showing:
- **Top-left:** Train vs Val Loss (decreasing over time = good)
- **Top-right:** Box Loss (bounding box coordinate accuracy)
- **Middle-left:** Class Loss (classification accuracy)
- **Middle-right:** mAP50 (detection accuracy at IoU=0.5)
- **Bottom-left:** Final metrics bar chart
- **Bottom-right:** Legend with key metrics

### `confusion_matrix.png`
- **Columns:** True classes
- **Rows:** Predicted classes
- **Values:** Number of predictions
- **Diagonal:** Correct predictions (want high here!)
- **Off-diagonal:** Misclassifications (want low here!)

### `TRAINING_REPORT.md`
Contains:
- Training configuration
- Test set performance metrics
- Per-class analysis
- Insights and recommendations
- Class distribution info

### `summary_statistics.csv`
Dataset breakdown:
```
Split,Images,Objects,Avg_Objects_Per_Image
Train,3648,78572,21.5
Validation,456,10324,22.6
Test,456,8019,17.6
Total,4560,96915,21.2
```

---

## 🔄 Complete Workflow

```
Step 1: Data Preparation
└─→ python sample_and_convert_datasets.py
    └─→ Output: unified_dataset/ (4,560 images)

Step 2: Validation
└─→ python validate_unified_dataset.py
    └─→ Output: Validation report (0 errors)

Step 3: Visualization (Optional)
└─→ python visualize_unified_dataset.py
    └─→ Output: Sample image plots

Step 4: YOLO Dataset Creation
└─→ python create_yolo_dataset.py
    └─→ Output: yolo_dataset/ (train/val/test split)

Step 5: Training
└─→ python train_quick_start.py  (OR)
    └─→ python train_yolo_model.py --model yolov8m --epochs 100
    └─→ Output: training_results/ (metrics, plots, models)

Step 6: Analysis
└─→ Review training_metrics.png
└─→ Review confusion_matrix.png
└─→ Read TRAINING_REPORT.md
```

---

## 🎯 When to Use Each Model Size

| Model | Speed | Accuracy | Memory | Best For |
|-------|-------|----------|--------|----------|
| yolov8n | ⚡⚡⚡ Fast | ⭐⭐ | 2GB | Testing, edge devices |
| yolov8s | ⚡⚡ Fast | ⭐⭐⭐ | 3GB | Mobile, embedded |
| yolov8m | ⚡ Medium | ⭐⭐⭐⭐ | 6GB | **Recommended** |
| yolov8l | Slow | ⭐⭐⭐⭐⭐ | 10GB | High accuracy needed |
| yolov8x | ⚠️ Slow | ⭐⭐⭐⭐⭐ | 16GB | Research, max accuracy |

---

## 💾 Training on Different Configurations

### Fast Prototyping (10 min)
```bash
python train_yolo_model.py \
    --model yolov8n \
    --epochs 10 \
    --batch 64
```

### Quick Test (30 min)
```bash
python train_quick_start.py
# Choose: Quick
```

### Production Standard (2-3 hours)
```bash
python train_quick_start.py
# Choose: Standard
```

### High Accuracy (6-8 hours)
```bash
python train_quick_start.py
# Choose: Advanced
```

### Custom Configuration (Your choice)
```bash
python train_yolo_model.py \
    --model yolov8l \
    --epochs 200 \
    --batch 16 \
    --patience 30
```

---

## 📌 Key Metrics to Watch

### During Training
- **Box Loss**: Should decrease (train model to predict bounding boxes)
- **Class Loss**: Should decrease (train model to classify objects)
- **Train Loss vs Val Loss**: Should both decrease together (not overfitting)

### After Training
- **mAP50**: >0.50 is good, >0.60 is great
- **mAP50-95**: >0.30 is acceptable, >0.40 is excellent
- **Precision**: What % of positive predictions are correct
- **Recall**: What % of actual targets are found

---

## ⚠️ Common Issues & Fixes

| Issue | Solution |
|-------|----------|
| CUDA not found | Install PyTorch with CUDA: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118` |
| Out of memory | Reduce batch size: `--batch 16` or use smaller model `yolov8n` |
| Training stuck | Check GPU: `nvidia-smi` or reduce image size: `--imgsz 512` |
| Low accuracy on class X | High class imbalance (vehicle is 65% of data). Try class weighting or augmentation |
| Can't find data.yaml | Ensure yolo_dataset/data.yaml exists (run create_yolo_dataset.py first) |

---

## 🎓 Understanding the Dataset

**Class Distribution:**
- **Vehicle: 65.6%** (dominant class)
- **Building: 17.3%** (well-represented)
- **Vegetation: 11.6%** (reasonable)
- **Water Body: 2.7%** (minority)
- **Road: 2.6%** (minority)

**Implication:** Model may favor vehicle detection. Consider:
- Class weighting to balance training
- Data augmentation for minority classes
- Separate models for different classes

---

## 🚀 Next Steps After Training

1. **Evaluate Results**
   ```bash
   # Check best.pt performance on test set
   # Review confusion_matrix.png for error patterns
   ```

2. **If Accuracy is Low:**
   - Rerun with more epochs (150+)
   - Try larger model (yolov8l)
   - Apply class weighting
   - Use image augmentation

3. **If Accuracy is Good:**
   - Export model for deployment
   - Setup inference pipeline
   - Batch process on production data

4. **Deploy Model**
   ```python
   from ultralytics import YOLO
   model = YOLO('training_results/YYYYMMDD_HHMMSS/model/weights/best.pt')
   results = model.predict('image.jpg')
   ```

---

**Ready to train?** Run: `python train_quick_start.py`
