"""
Score Detector - Detects basketball scoring events using YOLOv8 and trajectory analysis.
"""

import cv2
import numpy as np
import os
import math
import logging
import shutil
import threading
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# The model is shared by all detection runs. Its predictor keeps per-call state, so
# loading and inference are serialized through one lock.
_model_lock = threading.Lock()
_model = None
_model_key = None


def get_model():
    """Load the detection model once per process. The caller must hold _model_lock."""
    global _model, _model_key
    # Load best.pt from models dir
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'best.pt')
    if not os.path.exists(model_path):
        # Fallback to yolov8n.pt if best.pt is missing
        model_path = 'yolov8n.pt'
        logger.warning(f"best.pt not found, using {model_path}")

    # Reload when the weights file is replaced (e.g. by scripts/run_evolution.sh)
    key = (model_path, os.path.getmtime(model_path)) if os.path.exists(model_path) else model_path
    if _model is None or key != _model_key:
        logger.info("Initializing YOLOv8 model instance.")
        _model = YOLO(model_path)
        _model_key = key

    # Basketball detection (COCO class 32 for sports ball)
    target_cls = 0 if 'best.pt' in model_path else 32
    return _model, target_cls


def get_device():
    """Automatically select device -> mps(Mac) -> cpu"""
    if torch.cuda.is_available():
        return 'cuda'
    # Use mps on apple silicon for performance
    elif torch.backends.mps.is_available():
        return 'mps'
    if shutil.which('nvidia-smi'):
        # The torch that pip installs by default on Windows / Linux has no CUDA support
        logger.warning("An NVIDIA GPU is present but this PyTorch build has no CUDA support: "
                       "detection runs on the CPU. Install a CUDA build of torch to use the GPU.")
    return 'cpu'


# Where the ball is tracked, relative to the hoop box (centre h_cx, h_cy and size h_w, h_h):
# two hoop widths to each side, from two hoop heights above the centre to 1.5 below it.
ZONE_X = 2.0
ZONE_TOP = 2.0
ZONE_BOTTOM = 1.5


def in_hoop_region(center, hoop_rect):
    h_cx, h_cy, h_w, h_h = hoop_rect
    x, y = center

    return (abs(x - h_cx) < ZONE_X * h_w
            and h_cy - ZONE_TOP * h_h < y < h_cy + ZONE_BOTTOM * h_h)


def hoop_roi(hoop_rect):
    """Crop (x1, y1, x2, y2) around the hoop region that is sent to the model instead of the whole frame.

    Every decision is made next to the hoop, and shrinking a full HD frame down to the model's input
    size leaves the ball only a few pixels wide there. The margin keeps objects whole at the edges.
    """
    h_cx, h_cy, h_w, h_h = hoop_rect
    margin = h_w

    return (h_cx - (ZONE_X * h_w + margin), h_cy - (ZONE_TOP * h_h + margin),
            h_cx + (ZONE_X * h_w + margin), h_cy + (ZONE_BOTTOM * h_h + margin))


def is_ball_candidate(center, conf, hoop_rect, conf_thresh):
    """Ball boxes that may enter the track: inside the hoop region and confident enough.

    The threshold is relaxed to 60% there: the ball is small, blurred and partly hidden by rim and
    net exactly where it matters.
    """
    return conf > conf_thresh * 0.6 and in_hoop_region(center, hoop_rect)


class ShotTracker:
    """Follows the ball around the hoop and reports made shots.

    The track holds at most one point per frame, picked by continuity, so a stray detection elsewhere
    cannot slip in between the points that matter. A score needs the ball to be seen above the rim
    plane near the hoop and, not much later, twice inside the net or below it close to the centre.
    The two sightings need not be continuous: on the rim and inside the net the ball is small, blurred
    and partly hidden, and routinely stays undetected for half a second or more.
    """

    MAX_GAP = 0.35          # seconds the ball may be invisible before the track is dropped and may restart
    HISTORY = 2.0           # seconds of track kept, for thumbnails and the debug video
    TAKEOVER_CONF = 0.6     # a detection far from the track replaces it only when at least this confident ...
    TAKEOVER_MARGIN = 0.25  # ... and clearly stronger than what the track has been seeing

    APPROACH_REACH = 1.0    # "near the hoop" above the rim: up to this many hoop widths left/right of the centre
    APPROACH_HEIGHT = 1.5   # ... and up to this many hoop heights above the rim plane
    APPROACH_WINDOW = 1.0   # seconds after which a sighting above the rim no longer counts
    NET_HALF = 0.25         # inside the net: within this many hoop widths of the centre, rim to net bottom
    BELOW_HALF = 0.4        # below the net: within this many hoop widths of the centre
    MIN_DESCENT = 2.0       # px per frame the ball must drop between the two sightings in the net

    def __init__(self, hoop_rect, fps, min_interval):
        h_cx, h_cy, h_w, h_h = hoop_rect
        self.hoop_rect = hoop_rect
        self.fps = fps
        self.min_interval = min_interval
        self.rim_y = h_cy - 0.2 * h_h                 # height of the rim plane
        self.net_bottom = h_cy + 0.5 * h_h
        self.approach_reach = self.APPROACH_REACH * h_w
        self.approach_top = self.rim_y - self.APPROACH_HEIGHT * h_h
        self.net_half = self.NET_HALF * h_w
        self.below_half = self.BELOW_HALF * h_w
        self.max_gap = max(2, round(self.MAX_GAP * fps))
        self.approach_window = max(2, round(self.APPROACH_WINDOW * fps))
        self.history = max(self.max_gap, round(self.HISTORY * fps))

        self.points = []           # (frame, x, y, w, h, conf): one per frame, oldest first
        self.confirmed = False     # a (re)started track only counts once a second point continues it
        self.approach = None       # last point seen above the rim plane near the hoop
        self.last_score_time = -min_interval

    def step(self, frame, candidates):
        """Feed the ball candidates (x, y, w, h, conf) of one frame. Returns a score event or None."""
        new = self._advance(frame, candidates)

        while self.points and frame - self.points[0][0] > self.history:
            self.points.pop(0)

        if new is not None:
            return self._update(frame, new)
        return None

    def _advance(self, frame, candidates):
        """Attach one of this frame's candidates to the track. Returns the new point, if any."""
        if not candidates:
            return None
        best = max(candidates, key=lambda c: c[4])

        points = self.points
        if points and frame - points[-1][0] <= self.max_gap:
            last = points[-1]
            # How far the ball can have moved since its last point: 2.5 ball widths per frame, 5 at most
            gate = min(5 * last[3], 2.5 * last[3] * (frame - last[0]))
            near = [c for c in candidates if math.hypot(c[0] - last[1], c[1] - last[2]) <= gate]
            cont = max(near, key=lambda c: c[4]) if near else None

            # A detection far from the track replaces it only when it is strong on its own and clearly
            # stronger than what the track has been seeing; otherwise the track stays where it is.
            if best is not cont:
                if best[4] >= self.TAKEOVER_CONF and best[4] >= (cont[4] if cont else 0) + self.TAKEOVER_MARGIN:
                    self.points = [(frame,) + best]
                    self.confirmed = False
                    return None
                if cont is None:
                    return None

            points.append((frame,) + cont)
            self.confirmed = True
            return points[-1]

        # (Re)start on the most confident candidate; it only counts once a second point continues it
        self.points = [(frame,) + best]
        self.confirmed = False
        return None

    def in_net(self, p):
        return self.rim_y <= p[2] <= self.net_bottom and abs(p[1] - self.hoop_rect[0]) <= self.net_half

    def below_net(self, p):
        return p[2] >= self.net_bottom and abs(p[1] - self.hoop_rect[0]) <= self.below_half

    def _update(self, frame, new):
        """Check the track for a made shot. Returns a score event or None."""
        prev = self.points[-2]

        # Seen above the rim plane near the hoop: the ball is on its way in
        if new[2] < self.rim_y:
            if new[2] >= self.approach_top and abs(new[1] - self.hoop_rect[0]) <= self.approach_reach:
                self.approach = new
            return None

        # Seen twice in a row coming down through the net or out below it
        approach = self.approach
        if (approach is not None and approach[0] < prev[0] and frame - approach[0] <= self.approach_window
                and (self.in_net(prev) or self.below_net(prev))
                and (self.in_net(new) or self.below_net(new))
                and new[2] - prev[2] >= self.MIN_DESCENT * (frame - prev[0])
                and 0.6 <= new[3] / prev[3] <= 1.6):
            self.approach = None
            timestamp = frame / self.fps
            if timestamp - self.last_score_time >= self.min_interval:
                self.last_score_time = timestamp
                return {'frame': frame, 'timestamp': timestamp}

        return None


class ScoreDetector:
    """Detects scoring events by analyzing YOLOv8 object trajectories and validating them against hoop constraints."""

    def __init__(self, hoop_region, frame_skip=3, min_interval=5.0, sensitivity=50):
        self.hoop_region = tuple(int(v) for v in hoop_region)
        self.frame_skip = max(1, frame_skip)
        self.min_interval = min_interval
        self.sensitivity = int(sensitivity)

    def detect(self, video_path, video_id, thumbnail_dir, progress_callback=None, debug_video=False,
               pause_check=None, score_callback=None, start_frame=0, existing_scores=None):
        """
        Process video and detect scoring events.

        Args:
            pause_check: callable() -> bool, returns True if should pause. When True,
                         blocks until it returns False.
            score_callback: callable(scores_metadata) called each time a score is detected.
            start_frame: frame index to start/resume detection from.
            existing_scores: list of pre-existing score dicts when resuming.
        """
        device = get_device()
        with _model_lock:
            model, target_cls = get_model()

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
        if h_w <= 0 or h_h <= 0:
            raise ValueError(f"Invalid hoop region: {self.hoop_region}")
        hoop_rect = (hx1 + h_w / 2, hy1 + h_h / 2, h_w, h_h)

        # Analysis frame rate is fixed to video frame rate (1.0x).
        adj_skip = 1

        # Sensitivity (0-100) directly maps to Model Confidence Threshold.
        # Higher sensitivity -> Lower threshold (more inclusive).
        # Range: Sensitivity 100 => 0.05 conf, Sensitivity 0 => 0.50 conf.
        conf_thresh = 0.5 - (self.sensitivity / 100.0) * 0.45
        # Ensure it stays within reasonable bounds
        conf_thresh = max(0.05, min(0.5, conf_thresh))

        # YOLO drops boxes below its own confidence limit (0.25 by default), so ours has to be passed down
        model_conf = max(0.05, conf_thresh * 0.6)

        rx1, ry1, rx2, ry2 = (int(round(v)) for v in hoop_roi(hoop_rect))
        rx1, ry1 = max(0, rx1), max(0, ry1)
        rx2, ry2 = min(width, rx2), min(height, ry2)

        logger.info(f"YOLO Processing video: {video_path} on {device}. Frame skip: {adj_skip} (Full FPS), "
                    f"Sensitivity: {self.sensitivity} (Conf: {conf_thresh:.3f}), "
                    f"analysed area: ({rx1}, {ry1})-({rx2}, {ry2}) of {width}x{height}")

        # Setup Debug Video Writer
        video_writer = None
        if debug_video:
            debug_path = os.path.join(os.path.dirname(thumbnail_dir), f"debug_{video_id}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(debug_path, fourcc, fps / adj_skip, (width, height))
            logger.info(f"Debug video enabled, saving to: {debug_path}")

        tracker = ShotTracker(hoop_rect, fps, self.min_interval)
        scores_metadata = list(existing_scores) if existing_scores else []

        # If resuming, continue the numbering and the throttle from the existing scores
        if scores_metadata:
            tracker.last_score_time = max(s['timestamp'] for s in scores_metadata)
        score_count = len(scores_metadata)

        frame_idx = 0

        os.makedirs(thumbnail_dir, exist_ok=True)

        # If resuming, skip to start_frame
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frame_idx = start_frame
            logger.info(f"Resuming detection from frame {start_frame}")

        while cap.isOpened():
            # Check for pause
            if pause_check and pause_check():
                logger.info(f"Detection paused at frame {frame_idx}")
                return scores_metadata, frame_idx  # Return current results and position

            ret, frame = cap.read()
            if not ret:
                break

            with _model_lock:
                results = model(frame[ry1:ry2, rx1:rx2], conf=model_conf, device=device, verbose=False)

            candidates = []
            boxes = []
            for r in results:
                for box in r.boxes:
                    # Plain floats: torch tensors cannot go through round() or json.dump below
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    # Back to full-frame coordinates
                    x1, y1, x2, y2 = x1 + rx1, y1 + ry1, x2 + rx1, y2 + ry1
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                    boxes.append((x1, y1, x2, y2, conf, cls))
                    if cls == target_cls and is_ball_candidate((cx, cy), conf, hoop_rect, conf_thresh):
                        candidates.append((cx, cy, x2 - x1, y2 - y1, conf))

            event = tracker.step(frame_idx, candidates)

            if event:
                score_count += 1
                timestamp = event['timestamp']
                logger.info(f"SCORE DETECTED at {timestamp:.2f}s")

                # Draw target indicator for thumbnail
                thumb = frame.copy()
                cv2.rectangle(thumb, (int(hx1), int(hy1)), (int(hx2), int(hy2)), (0, 255, 0), 2)
                for p in tracker.points:
                    cv2.circle(thumb, (int(p[1]), int(p[2])), 4, (0, 0, 255), -1)

                thumb_filename = f"score_{score_count:04d}.jpg"
                thumb_path = os.path.join(thumbnail_dir, thumb_filename)
                cv2.imwrite(thumb_path, thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])

                last = tracker.points[-1]
                scores_metadata.append({
                    'id': score_count - 1,
                    'frame': frame_idx,
                    'timestamp': round(timestamp, 2),
                    'timestamp_str': self._format_time(timestamp),
                    'motion_score': round(float(last[5]), 2),  # Use confidence as score
                    'thumbnail': thumb_filename,
                    'player': '',
                    'confirmed': True,
                    'yolo_bbox': [
                        round(last[1] / width, 6),
                        round(last[2] / height, 6),
                        round(last[3] / width, 6),
                        round(last[4] / height, 6)
                    ]
                })

                # Notify caller of new score
                if score_callback:
                    score_callback(scores_metadata)

            if video_writer:
                debug_frame = frame.copy()
                # Draw hoop region and analysed area
                cv2.rectangle(debug_frame, (rx1, ry1), (rx2, ry2), (255, 255, 255), 1)
                cv2.rectangle(debug_frame, (int(hx1), int(hy1)), (int(hx2), int(hy2)), (255, 255, 0), 2)
                cv2.putText(debug_frame, "HOOP", (int(hx1), int(hy1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                # Where a score is decided: sightings in the net and below it must be close to the centre
                h_cx, h_cy = hoop_rect[0], hoop_rect[1]
                for half, top, bottom in ((tracker.net_half, tracker.rim_y, tracker.net_bottom),
                                          (tracker.below_half, tracker.net_bottom, h_cy + ZONE_BOTTOM * h_h)):
                    cv2.rectangle(debug_frame, (int(h_cx - half), int(top)), (int(h_cx + half), int(bottom)),
                                  (255, 0, 255), 1)

                for (x1, y1, x2, y2, conf, cls) in boxes:
                    color = (0, 255, 0) if cls == target_cls else (0, 0, 255)
                    cv2.rectangle(debug_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(debug_frame, f"{conf:.2f}", (int(x1), int(y1) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                for p in tracker.points:
                    cv2.circle(debug_frame, (int(p[1]), int(p[2])), 3, (0, 0, 255), -1)

                if event:
                    cv2.putText(debug_frame, "!!! SCORE !!!", (width // 2 - 100, height // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

                video_writer.write(debug_frame)

            frame_idx += 1
            if progress_callback and frame_idx % 30 == 0:
                progress_callback(frame_idx / (total_frames or 1))

        cap.release()
        if video_writer:
            video_writer.release()
        logger.info(f"Found {len(scores_metadata)} scoring events using YOLO.")
        return scores_metadata, frame_idx  # Always return tuple (scores, last_frame)

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
