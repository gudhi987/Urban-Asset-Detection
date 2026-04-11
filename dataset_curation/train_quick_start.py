"""
Quick start training with recommended settings.
Requires: pip install ultralytics torch torchvision scikit-learn pandas matplotlib seaborn pillow
"""

import subprocess
import sys
from pathlib import Path


def check_dependencies():
    """Check if required packages are installed"""
    required = ['ultralytics', 'torch', 'torchvision', 'scikit-learn', 'pandas', 'matplotlib', 'seaborn']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print(f"\nInstalling missing packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
        print("[OK] Packages installed\n")


def main():
    check_dependencies()
    
    print("\n" + "=" * 70)
    print("YOLO Training - Quick Start")
    print("=" * 70 + "\n")
    
    # Check if YOLO dataset exists
    dataset_path = Path('yolo_dataset')
    if not dataset_path.exists():
        print("Error: yolo_dataset not found!")
        print("Please create the YOLO dataset first using create_yolo_dataset.py")
        return 1
    
    print("Training Options:")
    print("\n1. Quick Training (yolov8n, 50 epochs, batch 64)")
    print("2. Standard Training (yolov8m, 100 epochs, batch 32)")
    print("3. Advanced Training (yolov8l, 150 epochs, batch 16)")
    print("4. Custom Settings")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    configs = {
        '1': {
            'model': 'yolov8n',
            'epochs': 50,
            'batch': 64,
            'name': 'Quick'
        },
        '2': {
            'model': 'yolov8m',
            'epochs': 100,
            'batch': 32,
            'name': 'Standard'
        },
        '3': {
            'model': 'yolov8l',
            'epochs': 150,
            'batch': 16,
            'name': 'Advanced'
        }
    }
    
    if choice in configs:
        config = configs[choice]
        print(f"\nSelected: {config['name']} Training")
        print(f"Model: {config['model']}")
        print(f"Epochs: {config['epochs']}")
        print(f"Batch: {config['batch']}")
    elif choice == '4':
        model = input("Model (yolov8n/s/m/l/x) [default: yolov8m]: ").strip() or 'yolov8m'
        epochs = int(input("Epochs [default: 100]: ").strip() or 100)
        batch = int(input("Batch size [default: 32]: ").strip() or 32)
        config = {'model': model, 'epochs': epochs, 'batch': batch, 'name': 'Custom'}
        print(f"\nSelected: {config['name']} Training")
    else:
        print("Invalid option")
        return 1
    
    # Run training
    print("\n" + "=" * 70)
    print("Starting Training...")
    print("=" * 70 + "\n")
    
    cmd = [
        sys.executable,
        'train_yolo_model.py',
        '--data', str(dataset_path / 'data.yaml'),
        '--model', config['model'],
        '--epochs', str(config['epochs']),
        '--batch', str(config['batch']),
        '--device', '0'
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n" + "=" * 70)
        print("[OK] Training completed successfully!")
        print("=" * 70)
        return result.returncode
    except KeyboardInterrupt:
        print("\n[!] Training interrupted by user")
        return 1
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Training failed with code {e.returncode}")
        return e.returncode


if __name__ == "__main__":
    exit(main())
