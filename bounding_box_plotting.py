# Add bounding boxes to some sample images
import random
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import os 
import numpy as np
from PIL import Image

def visualize_samples_with_bboxes(split_dir, pipeline, num_samples=4):
    images_dir = os.path.join(split_dir, "images")
    masks_dir = os.path.join(split_dir, "masks")
    labels_dir = os.path.join(split_dir, "labels")
    
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    image_files = random.sample(image_files, min(num_samples, len(image_files)))
    
    rows = len(image_files)
    cols = 3  # image, mask, bbox image
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)
    
    for row_idx, image_file in enumerate(image_files):
        base_name = os.path.splitext(image_file)[0].split("_")[0]  # Remove suffix if present
        
        image_path = os.path.join(images_dir, image_file)
        mask_path = os.path.join(masks_dir, base_name + "_mask.png")
        label_path = os.path.join(labels_dir, base_name + ".txt")
        
        # Load image & mask
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("RGB"))
        
        h, w = image.shape[:2]
        
        # --- Column 1: Original Image ---
        axes[row_idx, 0].imshow(image)
        axes[row_idx, 0].set_title(f"Image\n{image_file}")
        axes[row_idx, 0].axis("off")
        
        # --- Column 2: Mask ---
        axes[row_idx, 1].imshow(mask)
        axes[row_idx, 1].set_title("Mask")
        axes[row_idx, 1].axis("off")
        
        # --- Column 3: Image + BBoxes ---
        axes[row_idx, 2].imshow(image)
        axes[row_idx, 2].set_title("BBoxes")
        axes[row_idx, 2].axis("off")
        
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                lines = f.readlines()
            
            for line in lines:
                class_id, x_center, y_center, bw, bh = map(float, line.strip().split())
                
                # Convert YOLO → pixel coords
                x_center *= w
                y_center *= h
                bw *= w
                bh *= h
                
                x_min = x_center - bw / 2
                y_min = y_center - bh / 2
                
                rect = patches.Rectangle(
                    (x_min, y_min),
                    bw,
                    bh,
                    linewidth=2,
                    edgecolor='red',
                    facecolor='none'
                )
                axes[row_idx, 2].add_patch(rect)
                
                # Optional: class label
                class_name = pipeline.class_names.get(int(class_id), str(int(class_id)))
                axes[row_idx, 2].text(
                    x_min, y_min - 5,
                    class_name,
                    color='yellow',
                    fontsize=8,
                    backgroundcolor='black'
                )
        else:
            axes[row_idx, 2].text(10, 10, "No labels", color="red")
    
    plt.tight_layout()
    plt.show()