"""
Score Detector - Detects basketball scoring events using YOLOv8 and trajectory analysis.
"""

import cv2
import numpy as np
import os
import math
import logging
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)

def get_device():
    """Automatically select device -> mps(Mac) -> cpu"""
    if torch.cuda.is_available():
        return 'cuda'
    # Use mps on apple silicon for performance
    elif torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

def clean_ball_pos(ball_pos, current_frame):
    # ball_pos: array of ((x, y), frame, w, h, conf)
    if len(ball_pos) > 1:
        w1, h1 = ball_pos[-2][2], ball_pos[-2][3]
        w2, h2 = ball_pos[-1][2], ball_pos[-1][3]
        x1, y1 = ball_pos[-2][0]
        x2, y2 = ball_pos[-1][0]
        f1 = ball_pos[-2][1]
        f2 = ball_pos[-1][1]
        
        f_dif = f2 - f1
        dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        max_dist = 4 * math.sqrt(w1**2 + h1**2)
        
        # Ball should not move > 4x its diameter in 5 frames
        if (dist > max_dist) and (f_dif < 5):
            ball_pos.pop()
        # Ball should be relatively square
        elif (w2*1.8 < h2) or (h2*1.8 < w2):
            ball_pos.pop()

    # Remove points older than 30 frames
    if len(ball_pos) > 0:
        if current_frame - ball_pos[0][1] > 30:
            ball_pos.pop(0)

    return ball_pos

def check_score(ball_pos, hoop_rect):
    h_cx, h_cy, h_w, h_h = hoop_rect
    # rim_height defines the vertical plane of the rim
    rim_height = h_cy - 0.2 * h_h
    
    x = []
    y = []
    
    for i in reversed(range(len(ball_pos))):
        if ball_pos[i][0][1] < rim_height:
            x.append(ball_pos[i][0][0])
            y.append(ball_pos[i][0][1])
            if i + 1 < len(ball_pos):
                x.append(ball_pos[i + 1][0][0])
                y.append(ball_pos[i + 1][0][1])
            break
            
    if len(x) > 1:
        m, b = np.polyfit(x, y, 1)
        # Avoid division by zero if line is perfectly horizontal
        if m == 0:
            return False
            
        predicted_x = (rim_height - b) / m
        
        rim_x1 = h_cx - 0.4 * h_w
        rim_x2 = h_cx + 0.4 * h_w
        
        if rim_x1 < predicted_x < rim_x2:
            return True
        hoop_rebound_zone = 10
        if rim_x1 - hoop_rebound_zone < predicted_x < rim_x2 + hoop_rebound_zone:
            return True
            
    return False

def detect_up(ball_pos, hoop_rect):
    h_cx, h_cy, h_w, h_h = hoop_rect
    x1 = h_cx - 4 * h_w
    x2 = h_cx + 4 * h_w
    y1 = h_cy - 2 * h_h
    y2 = h_cy

    bx, by = ball_pos[-1][0]
    # Check if ball is in upper region
    if x1 < bx < x2 and y1 < by < y2 - 0.5 * h_h:
        return True
    return False

def detect_down(ball_pos, hoop_rect):
    h_cx, h_cy, h_w, h_h = hoop_rect
    by = ball_pos[-1][0][1]
    y = h_cy + 0.5 * h_h
    if by > y:
        return True
    return False

def in_hoop_region(center, hoop_rect):
    h_cx, h_cy, h_w, h_h = hoop_rect
    x, y = center
    
    x1 = h_cx - 2 * h_w
    x2 = h_cx + 2 * h_w
    y1 = h_cy - 2 * h_h
    y2 = h_cy + 1.5 * h_h
    
    if x1 < x < x2 and y1 < y < y2:
        return True
    return False


class ScoreDetector:
    """Detects scoring events by analyzing YOLOv8 object trajectories and validating them against hoop constraints."""

    def __init__(self, hoop_region, frame_skip=3, min_interval=5.0, sensitivity=50):
        self.hoop_region = tuple(int(v) for v in hoop_region)
        self.frame_skip = max(1, frame_skip)
        self.min_interval = min_interval
        self.sensitivity = int(sensitivity)
        
        # We will load the model lazily in detect to avoid blocking app startup
        self.model = None

    def detect(self, video_path, video_id, thumbnail_dir, progress_callback=None, debug_video=False):
        """
        Process video and detect scoring events.
        """
        logger.info("Initializing YOLOv8 model instance.")
        device = get_device()
        
        # Load best.pt from models dir
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'best.pt')
        if not os.path.exists(model_path):
            # Fallback to yolov8n.pt if best.pt is missing
            model_path = 'yolov8n.pt'
            logger.warning(f"best.pt not found, using {model_path}")
            
        if self.model is None:
            self.model = YOLO(model_path)
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        hx1, hy1, hx2, hy2 = self.hoop_region
        hx1, hy1 = max(0, hx1), max(0, hy1)
        hx2, hy2 = min(width, hx2), min(height, hy2)
        
        h_w = hx2 - hx1
        h_h = hy2 - hy1
        h_cx = hx1 + h_w / 2
        h_cy = hy1 + h_h / 2
        hoop_rect = (h_cx, h_cy, h_w, h_h)
        
        # Adjust frame skip based on sensitivity (1-100)
        # Low (0-40) -> skip 3-4, High (80-100) -> skip 1
        adj_skip = self.frame_skip
        if self.sensitivity >= 80:
            adj_skip = 1
        elif self.sensitivity >= 50:
            adj_skip = 2
        
        # Confidence threshold based on sensitivity
        conf_thresh = 0.4 if self.sensitivity < 30 else (0.25 if self.sensitivity < 70 else 0.15)
        
        logger.info(f"YOLO Processing video: {video_path} on {device}. Frame skip: {adj_skip}, Sensitivity: {self.sensitivity}")
        
        # Setup Debug Video Writer
        video_writer = None
        if debug_video:
            debug_path = os.path.join(os.path.dirname(thumbnail_dir), f"debug_{video_id}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(debug_path, fourcc, fps / adj_skip, (width, height))
            logger.info(f"Debug video enabled, saving to: {debug_path}")

        ball_pos = []
        scores_metadata = []
        
        up = False
        down = False
        up_frame = 0
        down_frame = 0
        last_score_time = -self.min_interval
        
        frame_idx = 0
        score_count = 0
        
        os.makedirs(thumbnail_dir, exist_ok=True)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % adj_skip == 0:
                results = self.model(frame, stream=True, device=device, verbose=False)
                
                debug_frame = None
                if debug_video:
                    debug_frame = frame.copy()
                    # Draw hoop region
                    cv2.rectangle(debug_frame, (int(hx1), int(hy1)), (int(hx2), int(hy2)), (255, 255, 0), 2)
                    cv2.putText(debug_frame, "HOOP", (int(hx1), int(hy1)-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                # Check for objects in the frame
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0]
                        w = x2 - x1
                        h = y2 - y1
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        
                        center = (int(x1 + w / 2), int(y1 + h / 2))
                        
                        # Basketball detection (COCO class 32 for sports ball)
                        target_cls = 0 if 'best.pt' in model_path else 32
                        if cls == target_cls:
                            # Use adaptive threshold inside the hoop region
                            if conf > conf_thresh or (in_hoop_region(center, hoop_rect) and conf > (conf_thresh * 0.6)):
                                ball_pos.append((center, frame_idx, w, h, conf))
                        
                        if debug_video:
                            color = (0, 255, 0) if cls == target_cls else (0, 0, 255)
                            cv2.rectangle(debug_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                            cv2.putText(debug_frame, f"{conf:.2f}", (int(x1), int(y1)-5),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                # Clean up tracks
                ball_pos = clean_ball_pos(ball_pos, frame_idx)
                
                if len(ball_pos) > 0:
                    if not up:
                        up = detect_up(ball_pos, hoop_rect)
                        if up:
                            up_frame = ball_pos[-1][1]
                    
                    if up and not down:
                        down = detect_down(ball_pos, hoop_rect)
                        if down:
                            down_frame = ball_pos[-1][1]
                            
                    # Trigger condition: passing up then down
                    if up and down and up_frame < down_frame:
                        timestamp = frame_idx / fps
                        
                        # Throttle detections
                        if timestamp - last_score_time >= self.min_interval:
                            if check_score(ball_pos, hoop_rect):
                                last_score_time = timestamp
                                score_count += 1
                                logger.info(f"SCORE DETECTED at {timestamp:.2f}s")
                                if debug_video:
                                    cv2.putText(debug_frame, "!!! SCORE !!!", (width // 2 - 100, height // 2),
                                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

                                # Draw target indicator for thumbnail
                                thumb = frame.copy()
                                cv2.rectangle(thumb, (int(hx1), int(hy1)), (int(hx2), int(hy2)), (0, 255, 0), 2)
                                for p in ball_pos:
                                    cv2.circle(thumb, p[0], 4, (0, 0, 255), -1)
                                    
                                thumb_filename = f"score_{score_count:04d}.jpg"
                                thumb_path = os.path.join(thumbnail_dir, thumb_filename)
                                cv2.imwrite(thumb_path, thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])
                                
                                scores_metadata.append({
                                    'id': score_count - 1,
                                    'frame': frame_idx,
                                    'timestamp': round(timestamp, 2),
                                    'timestamp_str': self._format_time(timestamp),
                                    'motion_score': round(float(ball_pos[-1][4]), 2),  # Use confidence as score
                                    'thumbnail': thumb_filename,
                                    'player': '',
                                    'confirmed': True
                                })
                        
                        # Reset for next detection after a trigger attempt
                        up = False
                        down = False
                
                if video_writer:
                    video_writer.write(debug_frame)

            frame_idx += 1
            if progress_callback and frame_idx % 30 == 0:
                progress_callback(frame_idx / total_frames)
                
        cap.release()
        if video_writer:
            video_writer.release()
        logger.info(f"Found {len(scores_metadata)} scoring events using YOLO.")
        return scores_metadata
        
    @staticmethod
    def _format_time(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        return f"{minutes}:{secs:02d}"


def extract_first_frame(video_path, output_path):
    """Extract the first frame of a video and save as JPEG."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError("Cannot read first frame")
    cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    h, w = frame.shape[:2]
    return {'width': w, 'height': h}
