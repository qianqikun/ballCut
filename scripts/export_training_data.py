import os
import json
import cv2
import shutil
import random
import argparse
from pathlib import Path

# Add project root to sys.path to import config
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from config import SESSION_DIR, DATA_DIR

def export_data(val_split=0.2):
    """
    Exports confirmed scoring events from sessions into a YOLOv8 dataset.
    """
    export_root = os.path.join(DATA_DIR, 'train_dataset')
    
    # Prepare directories
    for split in ['train', 'val']:
        os.makedirs(os.path.join(export_root, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(export_root, 'labels', split), exist_ok=True)
    
    print(f"Scanning sessions in: {SESSION_DIR}")
    
    samples = []
    
    # 1. Collect all confirmed samples
    for session_file in os.listdir(SESSION_DIR):
        if not session_file.endswith('.json'):
            continue
            
        session_path = os.path.join(SESSION_DIR, session_file)
        try:
            with open(session_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading {session_file}: {e}")
            continue
            
        video_path = data.get('video_path')
        if not video_path or not os.path.exists(video_path):
            print(f"Video not found for session {session_file}: {video_path}")
            continue
            
        scores = data.get('scores', [])
        for score in scores:
            # We only export confirmed scores that have a recorded bounding box
            if score.get('confirmed') and 'yolo_bbox' in score:
                samples.append({
                    'video_path': video_path,
                    'frame_idx': score['frame'],
                    'bbox': score['yolo_bbox'],
                    'score_id': score['id'],
                    'video_id': data.get('video_id', 'unknown')
                })
    
    if not samples:
        print("No confirmed samples with bounding boxes found. Try reviewing some videos first!")
        return

    print(f"Found {len(samples)} valid samples. Splitting into train/val...")
    
    # 2. Shuffle and split
    random.shuffle(samples)
    val_count = int(len(samples) * val_split)
    
    # 3. Process and save
    stats = {'train': 0, 'val': 0}
    
    current_video_path = None
    cap = None
    
    for i, sample in enumerate(samples):
        split = 'val' if i < val_count else 'train'
        
        # Open video only when path changes
        if sample['video_path'] != current_video_path:
            if cap: cap.release()
            cap = cv2.VideoCapture(sample['video_path'])
            current_video_path = sample['video_path']
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, sample['frame_idx'])
        ret, frame = cap.read()
        
        if not ret:
            print(f"Failed to extract frame {sample['frame_idx']} from {sample['video_path']}")
            continue
            
        # File names
        file_base = f"{sample['video_id']}_{sample['frame_idx']}_{sample['score_id']}"
        img_path = os.path.join(export_root, 'images', split, f"{file_base}.jpg")
        lbl_path = os.path.join(export_root, 'labels', split, f"{file_base}.txt")
        
        # Save image
        cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # Save label (Class 0 for basketball)
        # Bbox is [x_center, y_center, w, h] normalized
        bbox = sample['bbox']
        with open(lbl_path, 'w') as f:
            f.write(f"0 {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")
            
        stats[split] += 1
        if (i + 1) % 10 == 0 or (i + 1) == len(samples):
            print(f"Progress: {i+1}/{len(samples)} images exported...")

    if cap: cap.release()
    
    print("\nExport Complete!")
    print(f"Total exported: {stats['train'] + stats['val']}")
    print(f"Train: {stats['train']} | Val: {stats['val']}")
    print(f"Dataset path: {export_root}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export confirmed scores to YOLO dataset")
    parser.add_argument("--split", type=float, default=0.2, help="Validation split ratio (default: 0.2)")
    args = parser.parse_args()
    
    export_data(val_split=args.split)
