"""
BallCut - Basketball Scoring Highlight Generator
Flask Web Application
"""

import os
import uuid
import json
import threading
import logging
from flask import (Flask, render_template, request, jsonify,
                   send_from_directory, redirect, url_for, send_file)

from config import (SECRET_KEY, MAX_CONTENT_LENGTH, THUMBNAIL_DIR, UPLOAD_DIR,
                    OUTPUT_DIR, SESSION_DIR, TEMP_DIR, FRAME_SKIP,
                    MIN_SCORE_INTERVAL, CLIP_BEFORE, CLIP_AFTER)
from src.detector import ScoreDetector, extract_first_frame
from src.editor import generate_highlights, check_ffmpeg

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# In-memory task tracking
tasks = {}  # video_id -> {progress, status, error}


def get_session_path(video_id):
    return os.path.join(SESSION_DIR, f'{video_id}.json')


def load_session(video_id):
    path = get_session_path(video_id)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def save_session(video_id, data):
    os.makedirs(SESSION_DIR, exist_ok=True)
    path = get_session_path(video_id)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ─── Pages ────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/review/<video_id>')
def review(video_id):
    session = load_session(video_id)
    if not session:
        return redirect(url_for('index'))
    return render_template('review.html', video_id=video_id, session=session)


@app.route('/highlights/<video_id>')
def highlights(video_id):
    session = load_session(video_id)
    if not session:
        return redirect(url_for('index'))
    return render_template('highlights.html', video_id=video_id, session=session)


# ─── API ──────────────────────────────────────────────

from werkzeug.utils import secure_filename

@app.route('/api/upload-video', methods=['POST'])
def api_upload_video():
    """Upload video from browser."""
    if 'video' not in request.files:
        return jsonify({'error': '未找到视频文件'}), 400
        
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': '未选择视频文件'}), 400
        
    if file:
        video_id = uuid.uuid4().hex[:12]
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        
        filename = secure_filename(file.filename)
        if not filename:
            filename = f"video_{video_id}.mp4"
            
        video_path = os.path.join(UPLOAD_DIR, f"{video_id}_{filename}")
        file.save(video_path)

        # Extract first frame
        thumb_dir = os.path.join(THUMBNAIL_DIR, video_id)
        os.makedirs(thumb_dir, exist_ok=True)
        first_frame_path = os.path.join(thumb_dir, 'first_frame.jpg')

        try:
            frame_info = extract_first_frame(video_path, first_frame_path)
        except Exception as e:
            if os.path.exists(video_path):
                os.remove(video_path)
            return jsonify({'error': f'无法读取视频: {str(e)}'}), 400

        session = {
            'video_id': video_id,
            'video_path': video_path,
            'frame_width': frame_info['width'],
            'frame_height': frame_info['height'],
            'hoop_region': None,
            'scores': [],
            'highlights': {},
            'status': 'ready'
        }
        save_session(video_id, session)

        return jsonify({
            'video_id': video_id,
            'width': frame_info['width'],
            'height': frame_info['height']
        })



@app.route('/api/first-frame/<video_id>')
def api_first_frame(video_id):
    thumb_dir = os.path.join(THUMBNAIL_DIR, video_id)
    return send_from_directory(thumb_dir, 'first_frame.jpg')


@app.route('/api/detect', methods=['POST'])
def api_detect():
    """Start scoring detection in background."""
    data = request.get_json()
    video_id = data.get('video_id')
    hoop_region = data.get('hoop_region')  # [x1, y1, x2, y2]
    sensitivity = data.get('sensitivity', 50)

    session = load_session(video_id)
    if not session:
        return jsonify({'error': '会话不存在'}), 404

    session['hoop_region'] = hoop_region
    save_session(video_id, session)

    # Init task tracking
    tasks[video_id] = {'progress': 0.0, 'status': 'processing', 'error': None}

    # Run detection in background
    thread = threading.Thread(
        target=_run_detection,
        args=(video_id, session['video_path'], hoop_region, sensitivity),
        daemon=True
    )
    thread.start()

    return jsonify({'status': 'started'})


def _run_detection(video_id, video_path, hoop_region, sensitivity):
    """Background detection task."""
    try:
        def progress_cb(p):
            tasks[video_id]['progress'] = round(p, 3)

        thumb_dir = os.path.join(THUMBNAIL_DIR, video_id)
        detector = ScoreDetector(
            hoop_region=hoop_region,
            frame_skip=FRAME_SKIP,
            min_interval=MIN_SCORE_INTERVAL,
            sensitivity=sensitivity
        )
        scores = detector.detect(video_path, video_id, thumb_dir, progress_cb)

        session = load_session(video_id)
        session['scores'] = scores
        session['status'] = 'detected'
        save_session(video_id, session)

        tasks[video_id]['status'] = 'done'
        tasks[video_id]['progress'] = 1.0
        logger.info(f"Detection complete for {video_id}: {len(scores)} events")

    except Exception as e:
        logger.exception(f"Detection failed for {video_id}")
        tasks[video_id]['status'] = 'error'
        tasks[video_id]['error'] = str(e)


@app.route('/api/status/<video_id>')
def api_status(video_id):
    task = tasks.get(video_id, {'progress': 0, 'status': 'unknown', 'error': None})
    return jsonify(task)


@app.route('/api/scores/<video_id>')
def api_scores(video_id):
    session = load_session(video_id)
    if not session:
        return jsonify({'error': '会话不存在'}), 404
    return jsonify({'scores': session.get('scores', [])})


@app.route('/api/update-scores', methods=['POST'])
def api_update_scores():
    """Update scores with player assignments and confirmations."""
    data = request.get_json()
    video_id = data.get('video_id')
    scores = data.get('scores', [])

    session = load_session(video_id)
    if not session:
        return jsonify({'error': '会话不存在'}), 404

    session['scores'] = scores
    save_session(video_id, session)
    return jsonify({'status': 'ok'})


@app.route('/api/generate', methods=['POST'])
def api_generate():
    """Generate highlight videos."""
    data = request.get_json()
    video_id = data.get('video_id')

    if not check_ffmpeg():
        return jsonify({'error': '未找到 FFmpeg，请先安装: brew install ffmpeg'}), 400

    session = load_session(video_id)
    if not session:
        return jsonify({'error': '会话不存在'}), 404

    confirmed = [s for s in session['scores'] if s.get('confirmed', True)]
    if not confirmed:
        return jsonify({'error': '没有已确认的进球'}), 400

    clip_before = data.get('clip_before', CLIP_BEFORE)
    clip_after = data.get('clip_after', CLIP_AFTER)

    tasks[video_id] = {'progress': 0.0, 'status': 'generating', 'error': None}

    thread = threading.Thread(
        target=_run_generate,
        args=(video_id, session, float(clip_before), float(clip_after)),
        daemon=True
    )
    thread.start()

    return jsonify({'status': 'started'})


def _run_generate(video_id, session, clip_before, clip_after):
    """Background highlight generation task."""
    try:
        def progress_cb(p):
            tasks[video_id]['progress'] = round(p, 3)

        output_dir = os.path.join(OUTPUT_DIR, video_id)
        temp_dir = os.path.join(TEMP_DIR, video_id)

        confirmed = [s for s in session['scores'] if s.get('confirmed', True)]

        results = generate_highlights(
            video_path=session['video_path'],
            scores=confirmed,
            output_dir=output_dir,
            temp_dir=temp_dir,
            clip_before=clip_before,
            clip_after=clip_after,
            progress_callback=progress_cb
        )

        session['highlights'] = results
        session['status'] = 'complete'
        save_session(video_id, session)

        tasks[video_id]['status'] = 'done'
        tasks[video_id]['progress'] = 1.0

    except Exception as e:
        logger.exception(f"Generation failed for {video_id}")
        tasks[video_id]['status'] = 'error'
        tasks[video_id]['error'] = str(e)


@app.route('/api/video/<video_id>')
def api_video(video_id):
    session = load_session(video_id)
    if not session:
        return jsonify({'error': '会话不存在'}), 404
        
    video_path = session['video_path']
    directory = os.path.dirname(video_path)
    filename = os.path.basename(video_path)
    return send_from_directory(directory, filename)


@app.route('/api/thumbnail/<video_id>/<filename>')
def api_thumbnail(video_id, filename):
    thumb_dir = os.path.join(THUMBNAIL_DIR, video_id)
    return send_from_directory(thumb_dir, filename)


@app.route('/api/download/<video_id>/<filename>')
def api_download(video_id, filename):
    output_dir = os.path.join(OUTPUT_DIR, video_id)
    return send_from_directory(output_dir, filename, as_attachment=True)


@app.route('/api/download_folder/<video_id>/<folder>')
def api_download_folder(video_id, folder):
    """Download a player's shot folder as a ZIP."""
    player_dir = os.path.join(OUTPUT_DIR, video_id, folder)
    if not os.path.exists(player_dir):
        return "Folder not found", 404
        
    temp_zip = f"{player_dir}.zip"
    import shutil
    # Create zip from the folder
    shutil.make_archive(player_dir, 'zip', player_dir)
    
    return send_file(temp_zip, as_attachment=True, download_name=f"{folder}_shots.zip")


if __name__ == '__main__':
    # Ensure directories exist
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(THUMBNAIL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SESSION_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    app.run(host='0.0.0.0', port=8080, debug=True)
