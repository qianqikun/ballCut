import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
UPLOAD_DIR = os.path.join(DATA_DIR, 'uploads')
THUMBNAIL_DIR = os.path.join(DATA_DIR, 'thumbnails')
OUTPUT_DIR = os.path.join(DATA_DIR, 'output')
SESSION_DIR = os.path.join(DATA_DIR, 'sessions')
TEMP_DIR = os.path.join(DATA_DIR, 'temp')

# Ensure directories exist
for d in [DATA_DIR, UPLOAD_DIR, THUMBNAIL_DIR, OUTPUT_DIR, SESSION_DIR, TEMP_DIR]:
    os.makedirs(d, exist_ok=True)

# Detection settings
FRAME_SKIP = 3          # Process every Nth frame (higher = faster but less accurate)
MIN_SCORE_INTERVAL = 5.0  # Minimum seconds between detected scores
CLIP_BEFORE = 5.0       # Seconds before score to include in clip
CLIP_AFTER = 3.0        # Seconds after score to include in clip

# Flask settings
MAX_CONTENT_LENGTH = 10 * 1024 * 1024 * 1024  # 10 GB
SECRET_KEY = 'ballcut-secret-key-change-me'
