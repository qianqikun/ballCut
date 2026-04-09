import os
import yaml
import torch
from ultralytics import YOLO

# Add project root to sys.path to import config
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from config import DATA_DIR, BASE_DIR

def get_device():
    """Automatically select device -> mps(Mac) -> cuda -> cpu"""
    if torch.cuda.is_available():
        return '0'  # Use first GPU
    elif torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

def train_model(epochs=50, imgsz=640, batch=16):
    dataset_path = os.path.join(DATA_DIR, 'train_dataset')
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        print("Please run scripts/export_training_data.py first.")
        return

    # 1. Create data.yaml
    data_yaml = {
        'path': dataset_path,
        'train': 'images/train',
        'val': 'images/val',
        'names': {0: 'ball'}
    }
    
    yaml_path = os.path.join(dataset_path, 'ball_data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"Generated config at: {yaml_path}")
    
    # 2. Load model (use the best.pt from models dir as starting point)
    initial_model_path = os.path.join(BASE_DIR, 'models', 'best.pt')
    if not os.path.exists(initial_model_path):
        print(f"Initial model not found at {initial_model_path}, using yolov8n.pt")
        initial_model_path = 'yolov8n.pt'
    
    model = YOLO(initial_model_path)
    
    # 3. Start training
    device = get_device()
    print(f"Starting training on device: {device}")
    
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=os.path.join(DATA_DIR, 'runs'),
        name='ball_refinement'
    )
    
    print("\nTraining Finished!")
    final_path = os.path.join(DATA_DIR, 'runs', 'ball_refinement', 'weights', 'best.pt')
    if os.path.exists(final_path):
        print(f"New best model saved at: {final_path}")
        print(f"To use it, copy it to: {os.path.join(BASE_DIR, 'models', 'best.pt')}")
    else:
        print("Could not find the new best.pt. Check the runs/ directory.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train YOLO model on exported dataset")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs (default: 50)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    
    args = parser.parse_args()
    train_model(epochs=args.epochs, imgsz=args.imgsz, batch=args.batch)
