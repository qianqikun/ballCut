"""
BallCut - Basketball Scoring Highlight Generator
Flask Web Application
"""

import os
import re
import uuid
import json
import math
import shutil
import logging
import threading
from datetime import datetime

import cv2
from flask import (Flask, render_template, request, jsonify, Response,
                   send_from_directory, redirect, url_for, send_file)

from config import (SECRET_KEY, MAX_CONTENT_LENGTH, THUMBNAIL_DIR, UPLOAD_DIR,
                    OUTPUT_DIR, SESSION_DIR, TEMP_DIR, FRAME_SKIP,
                    MIN_SCORE_INTERVAL, CLIP_BEFORE, CLIP_AFTER)
from src.detector import ScoreDetector, extract_first_frame
from src.editor import generate_highlights, check_ffmpeg, keyframe_interval, EXPORT_MODES

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# In-memory task tracking
tasks = {}  # video_id -> {progress, status, error, paused, resume_event, paused_frame, score_count}

# Session files are read-modify-written by the detection thread (new scores, checkpoints) and by
# requests from the review page at the same time, so those operations are serialized.
_session_lock = threading.RLock()

# A detection can be continued from a session in one of these states: paused by the user, or
# interrupted while it was running (server restart, crash) or failed with an error.
RESUMABLE_STATUSES = ('paused', 'processing', 'error')

# Two detections less than this apart in the video are the same make
SCORE_MATCH_SECONDS = 2.0
VIDEO_ID_PATTERN = re.compile(r'[0-9a-f]{12}')


def get_session_path(video_id):
    return os.path.join(SESSION_DIR, f'{video_id}.json')


def format_time(seconds):
    """Seconds -> 'm:ss' or 'h:mm:ss'."""
    return ScoreDetector._format_time(seconds)


def parse_time(text):
    """'1:23:45', '23:45' or a number of seconds -> seconds; None when it cannot be read."""
    if text is None:
        return None
    parts = str(text).strip().replace('：', ':').split(':')
    if not 1 <= len(parts) <= 3:
        return None
    try:
        values = [float(p) for p in parts]
    except ValueError:
        return None
    if any(v < 0 or math.isnan(v) or math.isinf(v) for v in values):
        return None
    seconds = 0.0
    for value in values:
        seconds = seconds * 60 + value
    return seconds


def video_info(video_path):
    """(fps, number of frames) of a video; nothing is decoded."""
    cap = cv2.VideoCapture(video_path)
    try:
        return cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def is_inside(folder, path):
    """True when path lies in folder; neither of them has to exist."""
    try:
        return os.path.commonpath([os.path.realpath(folder), os.path.realpath(path)]) == os.path.realpath(folder)
    except ValueError:  # e.g. on different drives
        return False


def get_runs(session):
    """Every detection made for this video, oldest first."""
    runs = session.get('runs')
    if runs:
        return runs
    if not session.get('hoop_region') and not session.get('scores'):
        return []
    # Recorded before runs were kept: everything came from one detection of the whole video
    return [{'run': 1, 'start': None, 'end': None,
             'sensitivity': session.get('sensitivity', 50), 'hoop_region': session.get('hoop_region')}]


def merge_scores(scores, found, run_info):
    """Add what one detection run found to the scores of the video.

    A found score less than SCORE_MATCH_SECONDS away from an existing one is the same make: the existing
    entry stays as it is, labels included. The others are added with the run they came from. The list is
    kept in video order. Returns (scores, added, duplicates).
    """
    scores = list(scores)
    added, duplicates = [], []
    next_id = max((s.get('id', -1) for s in scores), default=-1) + 1
    for item in found:
        same_make = next((s for s in scores
                          if abs(s.get('timestamp', 0) - item.get('timestamp', 0)) < SCORE_MATCH_SECONDS), None)
        if same_make is not None:
            duplicates.append(item)
            # Found again by this run, so it is not "not found again" any more
            same_make.pop('missed_by', None)
            continue
        entry = dict(item)
        entry['id'] = next_id
        next_id += 1
        entry['run'] = run_info['run']
        entry['sensitivity'] = run_info['sensitivity']
        entry.pop('missed_by', None)
        scores.append(entry)
        added.append(entry)
    scores.sort(key=lambda s: (s.get('timestamp', 0), s.get('frame', 0)))
    return scores, added, duplicates


def mark_not_refound(scores, found, run, runs):
    """After a detection of one part of the video: flag the scores inside that part it did not find again.

    That is worth a look only when the comparison is fair: the hoop was not re-selected (the old score may
    be on another basket) and the sensitivity is at least the one the old score was found with.
    """
    start = run['start']
    end = run.get('end') if run.get('end') is not None else math.inf
    found_at = [s.get('timestamp', 0) for s in found]
    for score in scores:
        if not start <= score.get('timestamp', 0) <= end:
            continue
        if any(abs(t - score.get('timestamp', 0)) < SCORE_MATCH_SECONDS for t in found_at):
            score.pop('missed_by', None)
            continue
        origin = runs[score.get('run', 1) - 1] if 0 < score.get('run', 1) <= len(runs) else {}
        if (origin.get('hoop_region') == run.get('hoop_region')
                and run['sensitivity'] >= origin.get('sensitivity', 0)):
            score['missed_by'] = run['run']
    return sum(1 for s in scores if s.get('missed_by') == run['run'])


def load_session(video_id):
    path = get_session_path(video_id)
    if os.path.exists(path):
        with _session_lock:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
    return None


def save_session(video_id, data):
    os.makedirs(SESSION_DIR, exist_ok=True)
    path = get_session_path(video_id)
    with _session_lock:
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
    running = tasks.get(video_id, {}).get('status') == 'processing'

    if session.get('duration') is None:
        fps, total_frames = video_info(session['video_path'])
        if fps and total_frames:
            session['duration'] = total_frames / fps

    # What the "detect a range again" panel starts from: the settings of the last detection
    runs = get_runs(session)
    last = runs[-1] if runs else {}
    last_sensitivity = int(last.get('sensitivity', 50))
    if not runs:
        default_sensitivity = 70
    elif len(runs) == 1:
        # The whole video has been detected once: look harder than that for what it missed
        default_sensitivity = last_sensitivity + 20
    else:
        # A part was detected again before: start from what was used then, so that it does not climb by itself
        default_sensitivity = last_sensitivity
    duration = session.get('duration')
    range_defaults = {
        'sensitivity': max(10, min(90, default_sensitivity)),
        'hoop': last.get('hoop_region'),
        'video_hoop': session.get('hoop_region'),
        'duration': duration,
        # For picking a time range without typing: where the last detection of a part started and
        # ended, and which parts of the video have been detected again already
        'last_start': last.get('start'),
        'last_end': last.get('end'),
        'covered': [[r['start'], r.get('end') if r.get('end') is not None else duration]
                    for r in runs if r.get('start') is not None],
    }
    # How clips are exported: what was chosen last time, and how far a copied clip can start early
    export_mode = session.get('export_mode')
    export_mode = export_mode if export_mode in EXPORT_MODES else 'lossless'
    try:
        keyframe_gap = keyframe_interval(session['video_path'])
    except Exception as e:   # only a hint for the page: nothing depends on it
        logger.warning(f"Could not read the keyframe interval of {video_id}: {e}")
        keyframe_gap = None
    runs_view = [{
        'run': r['run'],
        'range': '整段' if r.get('start') is None
        else f"{format_time(r['start'])} – {format_time(r['end']) if r.get('end') is not None else '视频结尾'}",
        'sensitivity': r.get('sensitivity'),
        'reselected': bool(r.get('hoop_region')) and r.get('hoop_region') != session.get('hoop_region'),
        'added': r.get('added'),
    } for r in runs]

    return render_template('review.html', video_id=video_id, session=session,
                           clip_before=CLIP_BEFORE, clip_after=CLIP_AFTER, running=running,
                           range_defaults=range_defaults, runs_view=runs_view,
                           job=session.get('active_run') if running else None,
                           export_mode=export_mode, keyframe_gap=keyframe_gap)


@app.route('/highlights/<video_id>')
def highlights(video_id):
    session = load_session(video_id)
    if not session:
        return redirect(url_for('index'))
    return render_template('highlights.html', video_id=video_id, session=session)


# ─── API ──────────────────────────────────────────────

from werkzeug.utils import secure_filename, safe_join

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



@app.route('/api/sessions')
def api_sessions():
    """List all saved sessions."""
    if not os.path.exists(SESSION_DIR):
        return jsonify([])
        
    sessions = []
    for filename in os.listdir(SESSION_DIR):
        if filename.endswith('.json'):
            video_id = filename.replace('.json', '')
            session_path = os.path.join(SESSION_DIR, filename)
            
            try:
                # Use os.path.getmtime for sorting
                mtime = os.path.getmtime(session_path)
                
                with _session_lock:
                    with open(session_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                # Extract relevant info
                video_path = data.get('video_path', '')
                orig_filename = os.path.basename(video_path)
                # Remove the uuid prefix from filename if present
                if '_' in orig_filename and len(orig_filename.split('_')[0]) == 12:
                    orig_filename = '_'.join(orig_filename.split('_')[1:])
                
                sessions.append({
                    'video_id': video_id,
                    'filename': orig_filename,
                    'status': data.get('status', 'unknown'),
                    # 'processing' but not running means the detection was interrupted, e.g. by a server restart
                    'running': tasks.get(video_id, {}).get('status') == 'processing',
                    'error': data.get('error'),
                    'score_count': len(data.get('scores', [])),
                    'size': os.path.getsize(video_path) if video_path and os.path.exists(video_path) else 0,
                    'mtime': mtime,
                    'timestamp': os.path.getctime(session_path) if hasattr(os.path, 'getctime') else mtime
                })
            except Exception as e:
                logger.error(f"Error loading session {filename}: {e}")
                
    # Sort by mtime descending
    sessions.sort(key=lambda x: x['mtime'], reverse=True)
    return jsonify(sessions)


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
    debug_mode = data.get('debug_mode', False)

    session = load_session(video_id)
    if not session:
        return jsonify({'error': '会话不存在'}), 404
    task = tasks.get(video_id)
    if task and task.get('status') in ('processing', 'generating'):
        return jsonify({'error': '这个视频正在处理中，请等它结束再重新检测'}), 400

    session['hoop_region'] = hoop_region
    session['sensitivity'] = sensitivity
    session['debug_enabled'] = debug_mode
    session['status'] = 'processing'
    # A fresh run starts from the first frame and replaces what an earlier one found
    for key in ('paused_frame', 'paused_frame_ms', 'progress', 'error', 'active_run'):
        session.pop(key, None)
    session['scores'] = []
    session['highlights'] = {}
    session['runs'] = [{'run': 1, 'start': None, 'end': None,
                        'sensitivity': sensitivity, 'hoop_region': hoop_region}]
    save_session(video_id, session)

    # Init task tracking
    resume_event = threading.Event()
    resume_event.set()  # Not paused initially
    tasks[video_id] = {
        'progress': 0.0,
        'status': 'processing',
        'error': None,
        'paused': False,
        'resume_event': resume_event,
        'paused_frame': 0,
        'score_count': 0
    }

    # Run detection in background
    thread = threading.Thread(
        target=_run_detection,
        args=(video_id, session['video_path'], hoop_region, sensitivity, debug_mode),
        daemon=True
    )
    thread.start()

    return jsonify({'status': 'started'})


@app.route('/api/pause/<video_id>', methods=['POST'])
def api_pause(video_id):
    """Pause an ongoing detection."""
    task = tasks.get(video_id)
    if not task or task['status'] != 'processing':
        return jsonify({'error': '没有正在进行的检测任务'}), 400
    
    task['paused'] = True
    logger.info(f"Pause requested for {video_id}")
    return jsonify({'status': 'pausing'})


@app.route('/api/resume/<video_id>', methods=['POST'])
def api_resume(video_id):
    """Continue a detection from where it left off: paused, interrupted by a restart, or failed.

    That is the detection of the whole video, or the one of a part of it that was started last.
    """
    with _session_lock:
        session = load_session(video_id)
        if not session:
            return jsonify({'error': '会话不存在'}), 404

        task = tasks.get(video_id)
        if task and task['status'] in ('processing', 'generating'):
            return jsonify({'error': '没有已暂停的检测任务'}), 400
        paused_task = bool(task and task['status'] == 'paused')
        if not paused_task and session.get('status') not in RESUMABLE_STATUSES:
            return jsonify({'error': '没有已暂停的检测任务'}), 400

        active = session.get('active_run')
        if active:
            # The detection of a part of the video continues with its own sensitivity, hoop and range
            range_run = active
            start_frame = active.get('checkpoint_frame') or active['start_frame']
            resume_pos_ms = active['checkpoint_ms'] if active.get('checkpoint_frame') else active['start_pos_ms']
            sensitivity, hoop_region = active['sensitivity'], active['hoop_region']
            end_frame, end_pos_ms = active.get('end_frame'), active.get('end_pos_ms')
            keep_between = tuple(active['keep_between'])
            existing_scores = []
        else:
            range_run = end_frame = end_pos_ms = keep_between = None
            sensitivity, hoop_region = session.get('sensitivity', 50), session.get('hoop_region')
            if paused_task:
                # tasks only lives in memory: after a restart fall back to the position saved in the session
                start_frame = task.get('paused_frame', 0)
            else:
                start_frame = session.get('paused_frame', 0)
            # Never replay a score of this detection that is already saved
            last_score_frame = max((s.get('frame', -1) for s in session.get('scores', [])
                                    if s.get('run', 1) == 1), default=-1)
            start_frame = max(start_frame, last_score_frame + 1)
            resume_pos_ms = session.get('paused_frame_ms') if start_frame == session.get('paused_frame') else None
            # Scores up to here continue the numbering and the minimum interval. Ones further on (found by
            # a detection of a part of the video) are left to merge_scores, which keeps them from being
            # counted twice.
            existing_scores = [dict(s) for s in session.get('scores', []) if s.get('frame', 0) <= start_frame]

        session['status'] = 'processing'
        session.pop('error', None)
        save_session(video_id, session)

        run_no = active['run'] if active else 1
        # Shown while the model is still loading: where the detection continues from, not zero
        progress = ((active.get('progress') or active['start_frame'] / active['total_frames']) if active
                    else (session.get('progress') or 0.0))

    debug_mode = session.get('debug_enabled', False)

    # Reset task state for resuming
    if not task:
        task = {'progress': 0.0, 'paused': False}
        tasks[video_id] = task
    task['status'] = 'processing'
    task['paused'] = False
    task['error'] = None
    task['paused_frame'] = start_frame
    task['run'] = run_no
    task['score_count'] = len(session.get('scores', []))
    task['new_count'] = sum(1 for s in session.get('scores', []) if s.get('run', 1) == run_no)
    task['progress'] = max(task.get('progress') or 0.0, progress)

    # Run detection in background, resuming from start_frame
    thread = threading.Thread(
        target=_run_detection,
        args=(video_id, session['video_path'], hoop_region, sensitivity, debug_mode,
              start_frame, existing_scores, resume_pos_ms),
        kwargs={'end_frame': end_frame, 'end_pos_ms': end_pos_ms, 'keep_between': keep_between,
                'range_run': range_run},
        daemon=True
    )
    thread.start()

    return jsonify({'status': 'resumed', 'start_frame': start_frame, 'run': run_no})


def normalized_hoop(box, session):
    """[x1, y1, x2, y2] as integers inside the frame, or None when the box cannot be used."""
    try:
        x1, y1, x2, y2 = (int(round(float(v))) for v in box)
    except (TypeError, ValueError):
        return None
    width, height = session.get('frame_width') or 10 ** 9, session.get('frame_height') or 10 ** 9
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
    if x2 - x1 < 20 or y2 - y1 < 20:
        return None
    return [x1, y1, x2, y2]


@app.route('/api/detect-range', methods=['POST'])
def api_detect_range():
    """Detect one part of the video again, with its own sensitivity and hoop, and merge the result."""
    data = request.get_json()
    video_id = data.get('video_id')

    # Nothing or a blank for the start time: from the beginning of the video
    start = (0.0 if data.get('start') is None or str(data.get('start')).strip() == ''
             else parse_time(data.get('start')))
    if start is None:
        return jsonify({'error': '开始时间看不懂，请按 1:23:45、23:45 或秒数填写'}), 400
    end = None
    if data.get('end') is not None and str(data.get('end')).strip() != '':
        end = parse_time(data.get('end'))
        if end is None:
            return jsonify({'error': '结束时间看不懂，请按 1:23:45、23:45 或秒数填写'}), 400
        if end <= start:
            return jsonify({'error': '结束时间需要晚于开始时间'}), 400
    try:
        sensitivity = int(data.get('sensitivity'))
    except (TypeError, ValueError):
        return jsonify({'error': '灵敏度需要是 0 到 100 的整数'}), 400
    if not 0 <= sensitivity <= 100:
        return jsonify({'error': '灵敏度需要是 0 到 100 的整数'}), 400

    with _session_lock:
        session = load_session(video_id)
        if not session:
            return jsonify({'error': '会话不存在'}), 404
        task = tasks.get(video_id)
        if task and task['status'] in ('processing', 'generating'):
            return jsonify({'error': '这个视频正在处理中，请等它结束再重新检测'}), 400

        fps, total_frames = video_info(session['video_path'])
        if not fps or not total_frames:
            return jsonify({'error': '无法读取视频长度'}), 400
        duration = total_frames / fps
        if start >= duration:
            return jsonify({'error': f'开始时间超出了视频长度（{format_time(duration)}）'}), 400

        # Without a new box the last detection's is used
        runs = get_runs(session)
        box = data.get('hoop_region') or (runs[-1].get('hoop_region') if runs else None) or session.get('hoop_region')
        hoop_region = normalized_hoop(box, session)
        if hoop_region is None:
            return jsonify({'error': '请先框选篮筐区域'}), 400

        # An earlier run of a part that is still unfinished counts too (it is in active_run)
        run_no = max(len(runs), (session.get('active_run') or {}).get('run', 0)) + 1
        run = {'run': run_no, 'start': start, 'end': end, 'sensitivity': sensitivity,
               'hoop_region': hoop_region, 'at': datetime.now().isoformat(timespec='seconds')}
        # The part is analysed a little past its end, so that a shot still in progress there is not cut off
        grace = ScoreDetector.RANGE_GRACE_SECONDS
        analyse_until = None if end is None or end + grace >= duration else end + grace
        start_frame = int(round(start * fps))
        # A little before the start is analysed too, so that a shot already on its way in is not cut off
        rewind = min(start_frame, round(fps * ScoreDetector.RESUME_REWIND_SECONDS))

        session['runs'] = runs + [run]
        session['duration'] = duration
        # The detection of the whole video keeps its own state (position, status): it is put back when
        # this run is over. Until then everything about this run is in active_run, so that it can be continued.
        # When an earlier run of a part is still unfinished, what has to be put back is what it remembered.
        unfinished = session.get('active_run') or {}
        session['active_run'] = dict(
            run,
            prev_status=unfinished.get('prev_status', session.get('status')),
            prev_error=unfinished.get('prev_error', session.get('error')),
            start_frame=start_frame, from_frame=start_frame - rewind, start_pos_ms=start * 1000,
            end_frame=int(round(analyse_until * fps)) if analyse_until is not None else None,
            end_pos_ms=analyse_until * 1000 if analyse_until is not None else None,
            keep_between=[start, end],      # end None: up to the end of the video
            total_frames=total_frames, checkpoint_frame=None, checkpoint_ms=None, progress=None)
        session['status'] = 'processing'
        session.pop('error', None)
        save_session(video_id, session)

        active = session['active_run']
        tasks[video_id] = {
            'progress': active['start_frame'] / total_frames, 'status': 'processing', 'error': None,
            'paused': False, 'paused_frame': active['start_frame'],
            'score_count': len(session.get('scores', [])), 'run': run_no
        }

    thread = threading.Thread(
        target=_run_detection,
        args=(video_id, session['video_path'], hoop_region, sensitivity,
              session.get('debug_enabled', False), active['start_frame'], [], active['start_pos_ms']),
        kwargs={'end_frame': active['end_frame'], 'end_pos_ms': active['end_pos_ms'],
                'keep_between': tuple(active['keep_between']), 'range_run': active},
        daemon=True
    )
    thread.start()

    return jsonify({'status': 'started', 'run': run_no})


def save_position(session, frame, pos_ms, progress=None):
    """Remember where the running detection is, so that it can be continued after an interruption.

    The detection of the whole video and the one of a part of it each keep their own position.
    """
    target = session['active_run'] if session.get('active_run') else session
    if target is session:
        session['paused_frame'], session['paused_frame_ms'] = frame, pos_ms
    else:
        target['checkpoint_frame'], target['checkpoint_ms'] = frame, pos_ms
    if progress is not None:
        target['progress'] = progress


def drop_unused_thumbnails(thumb_dir, prefix, scores):
    """Remove the thumbnails a run made for scores that were not kept (the make was already there)."""
    if not os.path.isdir(thumb_dir):
        return
    used = {s.get('thumbnail') for s in scores}
    for name in os.listdir(thumb_dir):
        if name.startswith(prefix + '_') and name not in used:
            try:
                os.remove(os.path.join(thumb_dir, name))
            except OSError as e:
                logger.warning(f"Could not remove thumbnail {name}: {e}")


def _run_detection(video_id, video_path, hoop_region, sensitivity, debug_mode,
                   start_frame=0, existing_scores=None, resume_pos_ms=None,
                   end_frame=None, end_pos_ms=None, keep_between=None, range_run=None):
    """Background detection task: of the whole video, or with range_run of one part of it."""
    # Thumbnails are named after the run, so that one run never overwrites another run's
    run_no = range_run['run'] if range_run else 1
    prefix = 'score' if range_run is None else f"score_r{run_no}"
    run_info = {'run': run_no, 'sensitivity': sensitivity}
    thumb_dir = os.path.join(THUMBNAIL_DIR, video_id)

    try:
        def progress_cb(p):
            tasks[video_id]['progress'] = round(p, 3)

        def pause_check():
            """Returns True if detection should pause."""
            return tasks.get(video_id, {}).get('paused', False)

        def save_found(found):
            """Keep the video's scores up to date with what this run has found so far."""
            with _session_lock:
                session = load_session(video_id)
                if not session:
                    return
                scores, _added, _duplicates = merge_scores(session.get('scores', []), found, run_info)
                session['scores'] = scores
                save_session(video_id, session)
            task = tasks.setdefault(video_id, {})
            task['score_count'] = len(scores)
            task['new_count'] = sum(1 for s in scores if s.get('run', 1) == run_no)

        def checkpoint_cb(frame, pos_ms):
            """Called every few seconds while detection runs, so that a restart can continue from here."""
            try:
                with _session_lock:
                    session = load_session(video_id)
                    if session:
                        save_position(session, frame, pos_ms, tasks.get(video_id, {}).get('progress'))
                        save_session(video_id, session)
            except OSError as e:
                # Not being able to save the position must not stop the detection
                logger.warning(f"Could not save the detection position for {video_id}: {e}")

        detector = ScoreDetector(
            hoop_region=hoop_region,
            frame_skip=FRAME_SKIP,
            min_interval=MIN_SCORE_INTERVAL,
            sensitivity=sensitivity
        )
        # Enable debug video to help troubleshoot missed shots if requested
        scores, last_frame = detector.detect(
            video_path, video_id, thumb_dir,
            progress_callback=progress_cb,
            debug_video=debug_mode,
            pause_check=pause_check,
            score_callback=save_found,
            start_frame=start_frame,
            existing_scores=existing_scores,
            checkpoint_callback=checkpoint_cb,
            resume_pos_ms=resume_pos_ms,
            end_frame=end_frame,
            end_pos_ms=end_pos_ms,
            keep_between=keep_between,
            thumbnail_prefix=prefix
        )

        paused = tasks.get(video_id, {}).get('paused', False)
        all_scores = list(scores)
        with _session_lock:
            session = load_session(video_id)
            if session:
                # Scores that coincide with existing ones are the same makes: the existing entries stay
                all_scores, _added, _duplicates = merge_scores(session.get('scores', []), scores, run_info)
                session['scores'] = all_scores

                if paused:
                    session['status'] = 'paused'
                    save_position(session, last_frame, getattr(detector, 'paused_pos_ms', None),
                                  tasks[video_id].get('progress'))
                elif range_run:
                    runs = get_runs(session)
                    own_times = [s['timestamp'] for s in all_scores if s.get('run', 1) == run_no]
                    missed = mark_not_refound(all_scores, scores, range_run, runs)
                    record = next((r for r in runs if r.get('run') == run_no), None)
                    if record is not None:
                        record.update(
                            added=len(own_times),
                            matched=sum(1 for s in scores
                                        if not any(abs(t - s['timestamp']) < SCORE_MATCH_SECONDS for t in own_times)),
                            missed=missed)
                    # Back to how the detection of the whole video stood
                    session['status'] = (range_run.get('prev_status') if range_run.get('prev_status') in RESUMABLE_STATUSES
                                         else 'detected')
                    if session['status'] == 'error' and range_run.get('prev_error'):
                        session['error'] = range_run['prev_error']
                    session.pop('active_run', None)
                else:
                    session['status'] = 'detected'
                    for key in ('paused_frame', 'paused_frame_ms', 'progress', 'error'):
                        session.pop(key, None)
                save_session(video_id, session)

        if paused:
            tasks[video_id]['status'] = 'paused'
            tasks[video_id]['paused_frame'] = last_frame
            logger.info(f"Detection paused for {video_id} at frame {last_frame}: {len(all_scores)} scores so far")
            return

        if range_run:
            drop_unused_thumbnails(thumb_dir, prefix, all_scores)

        tasks[video_id]['status'] = 'done'
        tasks[video_id]['progress'] = 1.0
        tasks[video_id]['score_count'] = len(all_scores)
        logger.info(f"Detection complete for {video_id} (run {run_no}): {len(all_scores)} scores")

    except Exception as e:
        logger.exception(f"Detection failed for {video_id}")
        tasks[video_id]['status'] = 'error'
        tasks[video_id]['error'] = str(e)
        # Keep what was found and the last saved position, so that the run can be continued
        try:
            with _session_lock:
                session = load_session(video_id)
                if session:
                    session['status'] = 'error'
                    session['error'] = str(e)
                    save_session(video_id, session)
        except OSError as save_error:
            logger.warning(f"Could not record the detection error for {video_id}: {save_error}")


@app.route('/api/status/<video_id>')
def api_status(video_id):
    task = tasks.get(video_id, {'progress': 0, 'status': 'unknown', 'error': None})
    return jsonify({
        'progress': task.get('progress', 0),
        'status': task.get('status', 'unknown'),
        'error': task.get('error'),
        'paused': task.get('paused', False),
        'score_count': task.get('score_count', 0),
        # Scores found by the detection that is running (the whole video's, or the part's)
        'new_count': task.get('new_count', 0),
        'run': task.get('run', 1)
    })


@app.route('/api/scores/<video_id>')
def api_scores(video_id):
    session = load_session(video_id)
    if not session:
        return jsonify({'error': '会话不存在'}), 404
    return jsonify({'scores': session.get('scores', [])})


@app.route('/api/update-scores', methods=['POST'])
def api_update_scores():
    """Save the labels (player, confirmed, clip range) the review page changed on each score."""
    data = request.get_json()
    video_id = data.get('video_id')
    scores = data.get('scores', [])

    with _session_lock:
        session = load_session(video_id)
        if not session:
            return jsonify({'error': '会话不存在'}), 404

        # Applied to the scores as they are on disk, one by one. The page may not know scores that a
        # detection found in the meantime, and replacing the whole list would drop them.
        current = {s.get('id'): s for s in session.get('scores', [])}
        for change in scores:
            score = current.get(change.get('id'))
            if score is None:
                continue
            for key in ('player', 'confirmed', 'clip_before', 'clip_after'):
                if key in change:
                    score[key] = change[key]
        save_session(video_id, session)
    return jsonify({'status': 'ok'})


@app.route('/api/frame/<video_id>')
def api_frame(video_id):
    """One frame of the video, at the time given in ?t= (seconds)."""
    session = load_session(video_id)
    if not session:
        return jsonify({'error': '会话不存在'}), 404
    try:
        seconds = max(0.0, float(request.args.get('t', 0)))
    except ValueError:
        return jsonify({'error': '时间格式不对'}), 400

    cap = cv2.VideoCapture(session['video_path'])
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        return jsonify({'error': '无法读取这一帧'}), 404
    ok, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return jsonify({'error': '无法编码这一帧'}), 500
    return Response(buffer.tobytes(), mimetype='image/jpeg', headers={'Cache-Control': 'no-store'})


@app.route('/api/sessions/<video_id>', methods=['DELETE'])
def api_delete_session(video_id):
    """Delete a record with everything the app made for that video, the uploaded copy included."""
    if not VIDEO_ID_PATTERN.fullmatch(video_id):
        return jsonify({'error': '会话不存在'}), 404

    failed = []
    with _session_lock:
        task = tasks.get(video_id)
        if task and task.get('status') in ('processing', 'generating'):
            return jsonify({'error': '这个视频正在处理中，请等它结束再删除'}), 400

        session = load_session(video_id)
        if not session:
            return jsonify({'error': '会话不存在'}), 404

        def remove(path, is_dir=False):
            if not os.path.exists(path):
                return
            try:
                if is_dir:
                    shutil.rmtree(path)
                else:
                    os.remove(path)
            except OSError as e:
                failed.append(os.path.basename(path))
                logger.warning(f"Could not delete {path}: {e}")

        # Only what lies in the app's own data folders is ever touched: the uploaded copy, and what
        # was made from it. The record goes last, so that it is still there to retry from.
        video_path = session.get('video_path') or ''
        if os.path.basename(video_path).startswith(video_id + '_') and is_inside(UPLOAD_DIR, video_path):
            remove(video_path)
        for root in (THUMBNAIL_DIR, OUTPUT_DIR, TEMP_DIR):
            remove(os.path.join(root, video_id), is_dir=True)
        remove(os.path.join(THUMBNAIL_DIR, f"debug_{video_id}.mp4"))
        if not failed:
            remove(get_session_path(video_id))

    if failed:
        return jsonify({'error': '部分文件没有删掉，请重试', 'failed': failed}), 500
    tasks.pop(video_id, None)
    return jsonify({'status': 'deleted'})


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

    busy = tasks.get(video_id, {}).get('status')
    if busy == 'processing':
        return jsonify({'error': '检测还在进行中，请等检测结束后再生成'}), 400
    if busy == 'generating':
        return jsonify({'error': '集锦正在生成中，请稍候'}), 400

    confirmed = [s for s in session['scores'] if s.get('confirmed', True)]
    if not confirmed:
        return jsonify({'error': '没有已确认的进球'}), 400

    clip_before = data.get('clip_before', CLIP_BEFORE)
    clip_after = data.get('clip_after', CLIP_AFTER)

    # How the clips are made: copied without re-encoding (the default), or encoded again for exact
    # starts. What is chosen is remembered for the next time.
    mode = data.get('mode') or session.get('export_mode') or 'lossless'
    if mode not in EXPORT_MODES:
        return jsonify({'error': '未知的导出方式'}), 400
    with _session_lock:
        latest = load_session(video_id)
        if latest:
            latest['export_mode'] = mode
            save_session(video_id, latest)

    tasks[video_id] = {'progress': 0.0, 'status': 'generating', 'error': None,
                       'score_count': len(session.get('scores', []))}

    thread = threading.Thread(
        target=_run_generate,
        args=(video_id, session, float(clip_before), float(clip_after), mode),
        daemon=True
    )
    thread.start()

    return jsonify({'status': 'started', 'mode': mode})


def _run_generate(video_id, session, clip_before, clip_after, mode):
    """Background highlight generation task."""
    try:
        def progress_cb(p):
            tasks[video_id]['progress'] = round(p, 3)

        output_dir = os.path.join(OUTPUT_DIR, video_id)
        temp_dir = os.path.join(TEMP_DIR, video_id)

        confirmed = [s for s in session['scores'] if s.get('confirmed', True)]

        results, export_info = generate_highlights(
            video_path=session['video_path'],
            scores=confirmed,
            output_dir=output_dir,
            temp_dir=temp_dir,
            clip_before=clip_before,
            clip_after=clip_after,
            progress_callback=progress_cb,
            mode=mode
        )

        # Into the session as it is now: labels may have been changed while the clips were made
        with _session_lock:
            latest = load_session(video_id)
            if latest:
                latest['highlights'] = results
                latest['export_info'] = export_info
                latest['status'] = 'complete'
                save_session(video_id, latest)

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
    # safe_join returns None when either segment escapes OUTPUT_DIR (e.g. "..")
    player_dir = safe_join(OUTPUT_DIR, video_id, folder)
    if not player_dir or not os.path.isdir(player_dir):
        return "Folder not found", 404
        
    temp_zip = f"{player_dir}.zip"
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
    
    # Listen on localhost by default; set HOST=0.0.0.0 to reach the app from other devices on the LAN
    app.run(host=os.environ.get('HOST', '127.0.0.1'), port=8080, debug=True)
