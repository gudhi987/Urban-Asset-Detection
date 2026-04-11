# 🚀 Model Deployment & Inference Guide

## Overview

After training your YOLO model, you'll have:
- **best.pt** - Best trained model (use this!)
- **last.pt** - Final checkpoint
- **results.csv** - Training metrics
- **training_metrics.png** - Loss curves
- **confusion_matrix.png** - Error analysis

---

## Quick Start Inference

### Single Image Prediction

```python
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt

# Load trained model
model = YOLO('training_results/YYYYMMDD_HHMMSS/model/weights/best.pt')

# Predict on image
results = model.predict('path/to/image.jpg', conf=0.5)

# Access results
result = results[0]
boxes = result.boxes  # Bounding boxes
probs = result.probs  # Class probabilities

# Visualize
img = Image.open('path/to/image.jpg')
annotated = result.plot()
plt.imshow(annotated[..., ::-1])  # BGR to RGB
plt.show()

# Print detections
for box in result.boxes:
    print(f"Class: {int(box.cls)}, Confidence: {box.conf:.2f}")
    print(f"Box: {box.xyxy}")
```

### Batch Processing

```python
from ultralytics import YOLO
from pathlib import Path

model = YOLO('training_results/YYYYMMDD_HHMMSS/model/weights/best.pt')

# Process all images in folder
image_folder = Path('path/to/images')
results = model.predict(source=image_folder, conf=0.5, save=True)

# Results will be saved to runs/detect/predict/
```

### Video Inference

```python
from ultralytics import YOLO

model = YOLO('training_results/YYYYMMDD_HHMMSS/model/weights/best.pt')

# Process video (saves annotated output)
results = model.predict(
    source='video.mp4',
    conf=0.5,
    save=True,
    save_txt=True,
    device=0  # GPU device
)
# Output saved to runs/detect/predict/
```

---

## Understanding Model Output

### Class IDs
```python
CLASS_NAMES = {
    0: 'building',
    1: 'road',
    2: 'water_body',
    3: 'vegetation',
    4: 'vehicle'
}

# From results
for box in result.boxes:
    class_id = int(box.cls)
    class_name = CLASS_NAMES[class_id]
    confidence = float(box.conf)
    print(f"{class_name}: {confidence:.2%}")
```

### Bounding Box Formats

```python
result = results[0]

# Different box formats:
print(result.boxes.xyxy)   # x1, y1, x2, y2 (pixels)
print(result.boxes.xywh)   # x, y, w, h (pixels)
print(result.boxes.xyxyn)  # x1, y1, x2, y2 (normalized 0-1)
print(result.boxes.xywhn)  # x, y, w, h (normalized 0-1)

# Full boxes object
print(result.boxes.conf)   # Confidence scores
print(result.boxes.cls)    # Class IDs
```

---

## Production Deployment

### 1. Organize Model for Deployment

```bash
# Create deployment package
mkdir -p deployment/models
mkdir -p deployment/inference
mkdir -p deployment/results

# Copy model
cp training_results/YYYYMMDD_HHMMSS/model/weights/best.pt deployment/models/

# Copy inference script
cp inference_batch.py deployment/
```

### 2. Batch Inference Script

Create `inference_batch.py`:

```python
"""
Batch inference on multiple images
Usage: python inference_batch.py --input ./images --output ./results
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import json
from datetime import datetime

class YOLOInference:
    def __init__(self, model_path, conf=0.5, device=0):
        self.model = YOLO(model_path)
        self.conf = conf
        self.device = device
        self.class_names = {
            0: 'building',
            1: 'road', 
            2: 'water_body',
            3: 'vegetation',
            4: 'vehicle'
        }
    
    def predict_single(self, image_path):
        """Predict on single image"""
        results = self.model.predict(
            source=image_path,
            conf=self.conf,
            device=self.device,
            verbose=False
        )
        return results[0]
    
    def extract_detections(self, result):
        """Extract detections from result object"""
        detections = []
        
        for box in result.boxes:
            detection = {
                'class_id': int(box.cls),
                'class_name': self.class_names[int(box.cls)],
                'confidence': float(box.conf),
                'xyxy': box.xyxy.tolist()[0],  # [x1, y1, x2, y2]
                'xywh': box.xywh.tolist()[0]   # [x, y, w, h]
            }
            detections.append(detection)
        
        return detections
    
    def process_batch(self, image_folder, output_folder):
        """Process all images in folder"""
        image_folder = Path(image_folder)
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        results_summary = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'model': self.model.model_name,
                'confidence_threshold': self.conf,
                'device': self.device
            },
            'images': {}
        }
        
        # Process images
        for image_file in image_folder.glob('*.jpg') + image_folder.glob('*.png'):
            print(f"Processing: {image_file.name}")
            
            try:
                result = self.predict_single(str(image_file))
                detections = self.extract_detections(result)
                
                # Save annotated image
                img_with_boxes = result.plot()
                output_image = output_folder / f"annotated_{image_file.name}"
                result.save(str(output_image))
                
                # Save detections JSON
                detections_json = output_folder / f"{image_file.stem}_detections.json"
                with open(detections_json, 'w') as f:
                    json.dump(detections, f, indent=2)
                
                # Add to summary
                results_summary['images'][image_file.name] = {
                    'num_objects': len(detections),
                    'detections_file': str(detections_json),
                    'annotated_file': str(output_image),
                    'detections': detections
                }
                
                print(f"  → Found {len(detections)} objects")
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
                results_summary['images'][image_file.name] = {'error': str(e)}
        
        # Save summary
        summary_file = output_folder / 'summary.json'
        with open(summary_file, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        return results_summary

def main():
    parser = argparse.ArgumentParser(description='Batch YOLO inference')
    parser.add_argument('--input', type=str, required=True,
                       help='Input image folder')
    parser.add_argument('--output', type=str, required=True,
                       help='Output folder for results')
    parser.add_argument('--model', type=str, 
                       default='best.pt',
                       help='Path to model weights')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Confidence threshold (0-1)')
    parser.add_argument('--device', type=int, default=0,
                       help='GPU device ID (0 for first GPU)')
    
    args = parser.parse_args()
    
    # Run inference
    inference = YOLOInference(args.model, conf=args.conf, device=args.device)
    summary = inference.process_batch(args.input, args.output)
    
    print(f"\n✓ Processing complete!")
    print(f"Results saved to: {args.output}")
    print(f"Summary: {len(summary['images'])} images processed")

if __name__ == '__main__':
    main()
```

**Usage:**
```bash
python inference_batch.py \
    --input ./images \
    --output ./results \
    --model training_results/YYYYMMDD_HHMMSS/model/weights/best.pt \
    --conf 0.5 \
    --device 0
```

---

### 3. REST API Deployment (FastAPI)

Create `app.py`:

```python
"""
FastAPI web service for YOLO inference
Usage: uvicorn app:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import tempfile

# Initialize FastAPI
app = FastAPI(
    title="YOLO Urban Asset Detection API",
    description="API for detecting buildings, roads, water, vegetation, and vehicles",
    version="1.0.0"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model (on startup)
MODEL_PATH = 'models/best.pt'
model = YOLO(MODEL_PATH)

CLASS_NAMES = {
    0: 'building',
    1: 'road',
    2: 'water_body',
    3: 'vegetation',
    4: 'vehicle'
}

@app.get("/")
def read_root():
    """Health check"""
    return {"status": "healthy"}

@app.get("/api/info")
def get_model_info():
    """Get model information"""
    return {
        "model": "YOLOv8",
        "classes": CLASS_NAMES,
        "input_size": 640,
        "format": "PyTorch"
    }

@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    conf: float = Query(0.5, ge=0, le=1)
):
    """
    Predict objects in image
    
    Parameters:
    - file: Image file (JPG, PNG)
    - conf: Confidence threshold (0-1)
    
    Returns:
    - Detections with class, confidence, coordinates
    """
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Run inference
        results = model.predict(img, conf=conf, verbose=False)
        result = results[0]
        
        # Extract detections
        detections = []
        for box in result.boxes:
            detection = {
                'class_id': int(box.cls),
                'class_name': CLASS_NAMES[int(box.cls)],
                'confidence': float(box.conf),
                'x1': float(box.xyxy[0, 0]),
                'y1': float(box.xyxy[0, 1]),
                'x2': float(box.xyxy[0, 2]),
                'y2': float(box.xyxy[0, 3]),
            }
            detections.append(detection)
        
        return {
            'success': True,
            'image_name': file.filename,
            'num_objects': len(detections),
            'detections': detections,
            'timestamp': datetime.now().isoformat()
        }
    
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={'success': False, 'error': str(e)}
        )

@app.post("/api/predict_annotated")
async def predict_annotated(
    file: UploadFile = File(...),
    conf: float = Query(0.5, ge=0, le=1)
):
    """
    Predict and return annotated image
    """
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Run inference
        results = model.predict(img, conf=conf, verbose=False)
        result = results[0]
        
        # Draw boxes
        annotated = result.plot()
        
        # Save temp file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        cv2.imwrite(temp_file.name, annotated)
        
        return FileResponse(temp_file.name)
    
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={'success': False, 'error': str(e)}
        )

# Run with: uvicorn app:app --host 0.0.0.0 --port 8000
```

**Setup:**
```bash
pip install fastapi uvicorn python-multipart pillow

# Run server
uvicorn app:app --host 0.0.0.0 --port 8000

# Test in browser: http://localhost:8000/docs
```

**API Usage Examples:**

```bash
# Test health
curl http://localhost:8000/

# Get model info
curl http://localhost:8000/api/info

# Predict on image
curl -X POST "http://localhost:8000/api/predict?conf=0.5" \
     -F "file=@image.jpg"

# Get annotated image
curl -X POST "http://localhost:8000/api/predict_annotated?conf=0.5" \
     -F "file=@image.jpg" \
     > annotated.jpg
```

---

## Performance Optimization

### 1. Export to ONNX (Faster Inference)

```python
from ultralytics import YOLO

model = YOLO('best.pt')

# Export to ONNX
model.export(format='onnx')  # Creates best.onnx

# Use ONNX model
onnx_model = YOLO('best.onnx')
results = onnx_model.predict('image.jpg')
```

### 2. Export to TensorRT (NVIDIA)

```python
from ultralytics import YOLO

model = YOLO('best.pt')

# Export to TensorRT (fastest on NVIDIA GPUs)
model.export(format='engine')  # Creates best.engine

# Use TensorRT model
trt_model = YOLO('best.engine')
results = trt_model.predict('image.jpg')
```

### 3. Multiprocessing Inference

```python
from ultralytics import YOLO
from multiprocessing import Pool
from pathlib import Path

def infer_image(args):
    model_path, image_path = args
    model = YOLO(model_path)
    results = model.predict(image_path, verbose=False)
    return image_path, results[0]

# Batch process with multiprocessing
images = list(Path('images').glob('*.jpg'))
model_path = 'best.pt'

with Pool(4) as pool:  # 4 processes
    results = pool.map(infer_image, [(model_path, img) for img in images])

for image_path, result in results:
    print(f"{image_path}: Found {len(result.boxes)} objects")
```

---

## Confidence Thresholding

### Adjust per class

```python
from ultralytics import YOLO

model = YOLO('best.pt')
results = model.predict('image.jpg', conf=0.5)

result = results[0]

# Filter by class and confidence
for box in result.boxes:
    class_id = int(box.cls)
    confidence = float(box.conf)
    
    # Custom threshold per class
    thresholds = {
        0: 0.5,  # Building
        1: 0.7,  # Road (stricter)
        2: 0.6,  # Water
        3: 0.5,  # Vegetation
        4: 0.4   # Vehicle (more lenient)
    }
    
    if confidence >= thresholds[class_id]:
        print(f"Detected {CLASS_NAMES[class_id]} with {confidence:.2%} confidence")
```

---

## Integration Examples

### 1. GIS Data Analysis

```python
from ultralytics import YOLO
import rasterio
import numpy as np

model = YOLO('best.pt')

# Read satellite image
with rasterio.open('satellite_image.tif') as src:
    image = src.read()
    profile = src.profile

# Predict
results = model.predict(image)
result = results[0]

# Map results back to geospatial coordinates
for box in result.boxes:
    # box.xyxy contains pixel coordinates
    # Convert to geo-coordinates using rasterio
    pass
```

### 2. Drone Video Processing

```python
from ultralytics import YOLO
import cv2

model = YOLO('best.pt')

# Open drone video
cap = cv2.VideoCapture('drone_footage.mp4')

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Inference every N frames (skip expensive frames)
    if frame_count % 5 == 0:
        results = model.predict(frame, verbose=False)
        result = results[0]
        
        # Draw on frame
        annotated = result.plot()
        
        # Display
        cv2.imshow('Detection', annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
```

---

## Monitoring & Logging

```python
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f'inference_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# In inference loop
logger.info(f"Processing image: {image_path}")
logger.info(f"Found {len(detections)} objects")
logger.info(f"Class distribution: {class_distribution}")
```

---

## Next Steps

1. ✅ Train model
2. ✅ Evaluate on test set
3. **→ Deploy using one of these methods:**
   - Batch inference script
   - REST API
   - Edge deployment (Jetson, etc.)
4. Monitor performance
5. Retrain periodically with new data

