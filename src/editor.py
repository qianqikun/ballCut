"""
Video Editor - Clips and concatenates video segments using FFmpeg.

A clip is taken out of the video in one of two ways:

- lossless: copied as it is, without re-encoding. Picture and sound stay exactly those of the video,
  and it takes seconds. A copy can only start at a keyframe, so a clip starts a little earlier than
  asked (less than one keyframe interval, which depends on the camera); its end is exact.
- precise: encoded again, so that the clip starts exactly where it should. It keeps the codec, the bit
  depth and the colour information of the video, and uses a hardware encoder where there is one.

The clips of a highlight are always joined by copying.
"""

import json
import logging
import os
import shutil
import subprocess
import threading

logger = logging.getLogger(__name__)

EXPORT_MODES = ('lossless', 'precise')

_lock = threading.Lock()
_cache = {}  # what probing and trying out found, so that it is done once per process


# ── running ffmpeg / ffprobe ───────────────────────────────────────────────────

def run_tool(cmd, timeout):
    """Runs ffmpeg or ffprobe and returns the finished process; its output is text."""
    return subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace',
                          timeout=timeout)


def tool_path(name):
    """Where ffmpeg's tool `name` is; ffprobe is also looked for next to ffmpeg."""
    path = shutil.which(name)
    if path:
        return path
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg:
        sibling = os.path.join(os.path.dirname(ffmpeg), name + ('.exe' if os.name == 'nt' else ''))
        if os.path.exists(sibling):
            return sibling
    return None


def check_ffmpeg():
    """Check if ffmpeg is available."""
    try:
        return run_tool(['ffmpeg', '-version'], 5).returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def _stamp(path):
    """Identifies the content of a file for the cache: it may have been replaced since."""
    try:
        stat = os.stat(path)
        return stat.st_mtime_ns, stat.st_size
    except OSError:
        return None


def _keyframes_of(video_path):
    """Seek points of the video found so far: (pts, dts) of each, with the ranges they were looked for in."""
    key = ('keyframes', video_path, _stamp(video_path))
    with _lock:
        if key not in _cache:
            _cache[key] = ([], [])   # (points, [range start, range end] looked at)
        return _cache[key]


# ── finding out about the video ────────────────────────────────────────────────

def probe_video(video_path):
    """Codec, bit depth and colour information of the video stream (cached per file)."""
    key = ('probe', video_path, _stamp(video_path))
    with _lock:
        if key in _cache:
            return _cache[key]

    info = {'vcodec': '', 'pix_fmt': '', 'bit_depth': 8, 'colours': [], 'start_time': 0.0}
    ffprobe = tool_path('ffprobe')
    if ffprobe:
        try:
            result = run_tool([ffprobe, '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                               'stream=codec_name,pix_fmt,color_primaries,color_transfer,color_space',
                               '-show_entries', 'format=start_time', '-of', 'json', video_path], 60)
            data = json.loads(result.stdout)
            stream = (data.get('streams') or [{}])[0]
            info['vcodec'] = stream.get('codec_name') or ''
            info['pix_fmt'] = stream.get('pix_fmt') or ''
            info['bit_depth'] = 10 if '10' in info['pix_fmt'] else 8
            # What players need to show the colours right, e.g. HDR (HLG) videos
            for name, option in (('color_primaries', '-color_primaries'), ('color_transfer', '-color_trc'),
                                 ('color_space', '-colorspace')):
                value = stream.get(name)
                if value and value not in ('unknown', 'unspecified', 'reserved'):
                    info['colours'] += [option, value]
            try:
                info['start_time'] = float(data.get('format', {}).get('start_time') or 0.0)
            except ValueError:
                pass
        except (subprocess.TimeoutExpired, OSError, ValueError) as e:
            logger.warning(f"Could not read the video format of {video_path}: {e}")

    with _lock:
        _cache[key] = info
    return info


def seek_point_at_or_before(video_path, seconds):
    """Where a copy of the video can start at or before `seconds`: (pts, dts, pts of the next seek point).

    Seek points are the keyframes the file registers as such (ffmpeg only seeks to them). The next one
    is None when there is none within the ten seconds after.
    """
    ffprobe = tool_path('ffprobe')
    if not ffprobe:
        return None
    points, looked = _keyframes_of(video_path)
    offset = probe_video(video_path)['start_time']
    ahead = 10.0     # looked at this far past the wanted time, to know the next seek point

    # Looked for in a window before the wanted time: 30 s is plenty for cameras (they put a seek point
    # every second or so), and only a video with far fewer gets a second look further back.
    for span in (30.0, 300.0):
        start = max(0.0, seconds - span)
        if any(a <= start and seconds + ahead <= b for a, b in looked):
            break
        try:
            result = run_tool([ffprobe, '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                               'packet=pts_time,dts_time,flags', '-of', 'csv=p=0',
                               '-read_intervals', f'{start:.3f}%+{span + ahead:.3f}', video_path], 120)
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.warning(f"Could not look for keyframes in {video_path}: {e}")
            return None
        found = []
        for line in result.stdout.splitlines():
            fields = line.split(',')
            if len(fields) < 3 or 'K' not in fields[2]:
                continue
            try:
                found.append((float(fields[0]) - offset, float(fields[1]) - offset))   # times count from the start of the file
            except ValueError:
                continue
        with _lock:
            points.extend(found)
            looked.append([start, seconds + ahead])
        if any(pts <= seconds + 0.05 for pts, _dts in found) or start <= 0:
            break

    with _lock:
        ordered = sorted(points)
    before = [p for p in ordered if p[0] <= seconds + 0.05]
    if not before:
        return None
    point = before[-1]
    following = [pts for pts, _dts in ordered if pts > point[0]]
    return point[0], point[1], (following[0] if following else None)


def clip_duration(path):
    """Length of the video stream of a file in seconds, or None."""
    ffprobe = tool_path('ffprobe')
    if not ffprobe:
        return None
    try:
        result = run_tool([ffprobe, '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                           'stream=duration', '-of', 'csv=p=0', path], 60)
        return float(result.stdout.strip().rstrip(','))
    except (subprocess.TimeoutExpired, OSError, ValueError):
        return None


def keyframe_interval(video_path):
    """How far apart the keyframes of a video usually are (seconds), or None when that is not known."""
    key = ('interval', video_path, _stamp(video_path))
    with _lock:
        if key in _cache:
            return _cache[key]
    interval = None
    ffprobe = tool_path('ffprobe')
    if ffprobe:
        try:
            result = run_tool([ffprobe, '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                               'packet=pts_time,flags', '-of', 'csv=p=0',
                               '-read_intervals', '%+120', video_path], 120)
            times = sorted(float(fields[0]) for fields in (line.split(',') for line in result.stdout.splitlines())
                           if len(fields) >= 2 and 'K' in fields[1])
            gaps = sorted(b - a for a, b in zip(times, times[1:]))
            if gaps:
                interval = round(gaps[len(gaps) // 2], 1)   # the median
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.warning(f"Could not read the keyframe interval of {video_path}: {e}")
    with _lock:
        _cache[key] = interval
    return interval


# ── taking one clip out of the video ───────────────────────────────────────────

def _copy(video_path, output_path, seek, duration):
    cmd = [
        'ffmpeg', '-y',
        '-ss', f'{seek:.3f}',
        '-i', video_path,
        '-t', f'{duration:.3f}',
        '-map', '0:v:0', '-map', '0:a:0?',
        '-c', 'copy',
        '-avoid_negative_ts', 'make_zero',
        '-movflags', '+faststart',
        output_path
    ]
    try:
        result = run_tool(cmd, 300)
    except (subprocess.TimeoutExpired, OSError) as e:
        raise RuntimeError(f"FFmpeg failed: {e}")
    if result.returncode != 0:
        logger.error(f"FFmpeg copy error: {result.stderr[-500:]}")
        raise RuntimeError(f"FFmpeg failed: {result.stderr[-200:]}")


def clip_lossless(video_path, output_path, start_time, end_time):
    """Copies [start, end] out of the video without re-encoding. Returns the time the clip starts at.

    That is the seek point at or before start_time, so the clip starts a little early. Its end is a
    few frames past end_time: a copy can only stop between pictures in the order they are stored, and
    the last pictures shown need the ones stored after them.
    """
    logger.info(f"Copying {start_time:.2f}s - {end_time:.2f}s -> {output_path}")
    point = seek_point_at_or_before(video_path, start_time)
    if point is None:   # no ffprobe, or no seek point could be found: let ffmpeg seek to the wanted time
        _copy(video_path, output_path, start_time + 0.01, end_time - start_time + 0.02)
        return start_time

    keyframe, keyframe_dts, next_keyframe = point
    keyframe = min(keyframe, start_time)
    delay = max(0.0, keyframe - keyframe_dts)     # pictures are shown a little after they are stored

    # ffmpeg seeks to the last seek point before its target, except that a target within about one
    # such delay past a seek point still lands before it. So the target lies a little further on, well
    # before the next seek point; the copy starts at the seek point all the same. -t counts from the
    # target. The length of the result tells whether the seek landed where it should; if not, try again.
    wanted = end_time - keyframe
    margin = delay + 0.05
    for _attempt in range(3):
        margin = min(margin, (next_keyframe - keyframe) / 2 if next_keyframe is not None else 0.12)
        target = keyframe + margin
        _copy(video_path, output_path, target, end_time - target + 0.02)
        made = clip_duration(output_path)
        if made is None:
            break
        if made < wanted - 0.1:                   # started after the seek point: come closer to it
            margin /= 2
        elif made > wanted + delay + 0.15:        # started before the seek point: go further past it
            margin = margin * 2 + 0.05
        else:
            break
    return keyframe


def _encoding_candidates(info):
    """Ways to encode the video again, best first: the codec and the bit depth stay as they are."""
    hevc, ten_bit = info['vcodec'] == 'hevc', info['bit_depth'] == 10
    nvdec = ['-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda']
    vtdec = ['-hwaccel', 'videotoolbox', '-hwaccel_output_format', 'videotoolbox']
    deep = (['-profile:v', 'main10'] if ten_bit else [])   # 10-bit HEVC has a profile of its own; H.264 and 8-bit have not
    pix_fmt = ('p010le' if ten_bit else 'yuv420p') if hevc else 'yuv420p'

    if hevc:
        high = ['-preset', 'p6', '-tune', 'hq', '-rc', 'vbr', '-cq', '19', '-b:v', '0',
                '-spatial-aq', '1', '-temporal-aq', '1', '-rc-lookahead', '30']
        old = ['-preset', 'slow', '-rc', 'vbr_hq', '-cq', '19', '-b:v', '0']       # before ffmpeg 4.4
        cpus = [
            ('libx265', 'CPU 编码（x265）', ['-c:v', 'libx265', '-preset', 'fast', '-crf', '16',
                                          '-x265-params', 'log-level=error'],
             'yuv420p10le' if ten_bit else 'yuv420p'),
        ]
    else:
        high = ['-preset', 'p6', '-tune', 'hq', '-rc', 'vbr', '-cq', '18', '-b:v', '0',
                '-spatial-aq', '1', '-temporal-aq', '1', '-rc-lookahead', '30']
        old = ['-preset', 'slow', '-rc', 'vbr_hq', '-cq', '18', '-b:v', '0']
        cpus = [
            ('libx264', 'CPU 编码（x264）', ['-c:v', 'libx264', '-preset', 'fast', '-crf', '15'], 'yuv420p'),
        ]
    # The quality knob of VideoToolbox works differently per bit depth. These levels measured, against
    # the video, at SSIM 0.994 (8-bit) and 0.996 (10-bit) with about the video's own bitrate and twice it
    vt_quality = '70' if ten_bit else '85'
    codec = 'hevc' if hevc else 'h264'
    nvenc_label, vt_label = '显卡编码（NVENC）', '硬件编码（VideoToolbox）'
    return [
        # The GPU decodes too, which takes the work off the CPU altogether
        dict(name=f'{codec}_nvenc', label=nvenc_label, decoder=nvdec,
             encoder=[f'-c:v', f'{codec}_nvenc', *high, *deep], pix_fmt=pix_fmt),
        dict(name=f'{codec}_nvenc', label=nvenc_label, decoder=[],
             encoder=[f'-c:v', f'{codec}_nvenc', *high, *deep], pix_fmt=pix_fmt),
        # The way these options were spelled before ffmpeg 4.4
        dict(name=f'{codec}_nvenc', label=nvenc_label, decoder=nvdec,
             encoder=[f'-c:v', f'{codec}_nvenc', *old, *deep], pix_fmt=pix_fmt),
        dict(name=f'{codec}_nvenc', label=nvenc_label, decoder=[],
             encoder=[f'-c:v', f'{codec}_nvenc', *old, *deep], pix_fmt=pix_fmt),
        # The hardware encoder of Apple silicon (the Mac)
        dict(name=f'{codec}_videotoolbox', label=vt_label, decoder=vtdec,
             encoder=[f'-c:v', f'{codec}_videotoolbox', '-q:v', vt_quality, *deep], pix_fmt=pix_fmt),
        dict(name=f'{codec}_videotoolbox', label=vt_label, decoder=[],
             encoder=[f'-c:v', f'{codec}_videotoolbox', '-q:v', vt_quality, *deep], pix_fmt=pix_fmt),
        *[dict(name=name, label=label, decoder=[], encoder=encoder, pix_fmt=fmt) for name, label, encoder, fmt in cpus],
    ]


def _encoder_works(candidate):
    """Whether this ffmpeg can encode with these settings (tried on a black frame, once)."""
    test = (tuple(candidate['encoder']), candidate['pix_fmt'])
    with _lock:
        if ('encoder', test) in _cache:
            return _cache[('encoder', test)]
    cmd = ['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=black:s=256x256:r=30:d=0.2',
           *candidate['encoder'], '-pix_fmt', candidate['pix_fmt'], '-an', '-f', 'null', '-']
    try:
        ok = run_tool(cmd, 60).returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        ok = False
    with _lock:
        _cache[('encoder', test)] = ok
    return ok


def usable_encoders(video_path):
    """The encoders of _encoding_candidates that this ffmpeg can really use, best first."""
    return [c for c in _encoding_candidates(probe_video(video_path)) if _encoder_works(c)]


def clip_precise(video_path, output_path, start_time, end_time, choice):
    """Encodes [start, end] of the video again: the clip starts exactly at start_time.

    `choice` carries the encoders from one clip to the next: {'encoders': [...], 'current': index}.
    An encoder that fails on a real clip (e.g. the GPU decoder of the first candidate) is dropped.
    """
    if 'encoders' not in choice:
        choice['encoders'], choice['current'] = usable_encoders(video_path), 0
    if not choice['encoders']:
        raise RuntimeError("这个 ffmpeg 没有可用的编码器")

    colours = probe_video(video_path)['colours']
    while choice['current'] < len(choice['encoders']):
        candidate = choice['encoders'][choice['current']]
        cmd = ['ffmpeg', '-y', *candidate['decoder'], '-ss', f'{start_time:.3f}', '-i', video_path,
               '-t', f'{end_time - start_time:.3f}',
               '-map', '0:v:0', '-map', '0:a:0?', *candidate['encoder']]
        if not candidate['decoder']:
            # With a hardware decoder the frames keep their format; otherwise the encoder is told it
            cmd += ['-pix_fmt', candidate['pix_fmt']]
        cmd += ['-c:a', 'copy', *colours, '-vsync', '0', '-movflags', '+faststart', output_path]

        logger.info(f"Encoding {start_time:.2f}s - {end_time:.2f}s with {candidate['name']}"
                    f"{' (hardware decoding)' if candidate['decoder'] else ''} -> {output_path}")
        try:
            result = run_tool(cmd, 900)
        except (subprocess.TimeoutExpired, OSError) as e:
            result = None
            logger.error(f"FFmpeg with {candidate['name']} did not finish: {e}")
        if result is not None and result.returncode == 0:
            return candidate
        if result is not None:
            logger.error(f"FFmpeg with {candidate['name']} failed: {result.stderr[-500:]}")
        choice['current'] += 1
        logger.warning(f"{candidate['name']} does not work on this clip, trying the next encoder")
        try:
            os.remove(output_path)
        except OSError:
            pass

    raise RuntimeError("没有能用的编码器，请改用“无损直切”导出")


# ── joining the clips ──────────────────────────────────────────────────────────

def concat_videos(clip_paths, output_path, temp_dir):
    """
    Concatenate multiple video clips using ffmpeg concat demuxer.
    """
    if not clip_paths:
        raise ValueError("No clips to concatenate")

    if len(clip_paths) == 1:
        # Just copy the single clip
        shutil.copy2(clip_paths[0], output_path)
        return output_path

    # Create concat list file
    concat_file = os.path.join(temp_dir, "concat_list.txt")
    with open(concat_file, "w", encoding="utf-8") as f:
        for path in clip_paths:
            # Escape single quotes in path
            escaped = path.replace("'", "'\\''")
            # Convert backslashes to forward slashes for Windows ffmpeg paths
            escaped = escaped.replace('\\', '/')
            f.write(f"file '{escaped}'\n")

    cmd = [
        'ffmpeg', '-y',
        '-f', 'concat',
        '-safe', '0',
        '-i', concat_file,
        '-c', 'copy',
        '-movflags', '+faststart',
        output_path
    ]

    logger.info(f"Concatenating {len(clip_paths)} clips -> {output_path}")
    result = run_tool(cmd, 600)

    # Cleanup concat file
    if os.path.exists(concat_file):
        os.remove(concat_file)

    if result.returncode != 0:
        logger.error(f"FFmpeg concat error: {result.stderr[-500:]}")
        raise RuntimeError(f"FFmpeg concat failed: {result.stderr[-200:]}")

    return output_path


# ── the highlight videos ───────────────────────────────────────────────────────

def generate_highlights(video_path, scores, output_dir, temp_dir,
                        clip_before=5.0, clip_after=3.0, progress_callback=None, mode='lossless'):
    """
    Generate highlight videos grouped by player.

    Args:
        video_path: path to source video
        scores: list of confirmed score dicts with 'player' and 'timestamp'
        output_dir: directory to save highlight videos
        temp_dir: directory for temporary clip files
        clip_before: seconds before scoring moment
        clip_after: seconds after scoring moment
        progress_callback: function(progress: float) called with 0.0-1.0
        mode: 'lossless' (clips are copied, see the top of this file) or 'precise' (encoded again)

    Returns:
        (dict mapping player_name -> info about its highlight video, dict about how it was exported)
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    # Group scores by player
    player_scores = {}
    for score in scores:
        if not score.get('confirmed', True):
            continue
        player = score.get('player', '').strip()
        if not player:
            player = '未标注'
        if player not in player_scores:
            player_scores[player] = []
        player_scores[player].append(score)

    # Sort each player's scores by timestamp
    for player in player_scores:
        player_scores[player].sort(key=lambda s: s['timestamp'])

    total_clips = sum(len(v) for v in player_scores.values())
    clips_done = 0
    results = {}
    # How the clips were made, for the page that offers the downloads
    export = {'mode': mode if mode in EXPORT_MODES else 'lossless', 'encoder': None,
              'start_shift': 0.0, 'clips': 0}
    choice = {}

    for player, p_scores in player_scores.items():
        clip_paths = []
        clip_scores = []  # score for each entry in clip_paths (failed clips are skipped)

        for i, score in enumerate(p_scores):
            cb = score.get('clip_before', clip_before)
            ca = score.get('clip_after', clip_after)
            start = max(0, score['timestamp'] - cb)
            # Not cut at the length reported for the video: it is an estimate, and past the real end
            # ffmpeg simply stops
            end = score['timestamp'] + ca

            clip_filename = f"clip_{clips_done:04d}.mp4"
            clip_path = os.path.join(temp_dir, clip_filename)

            try:
                if export['mode'] == 'precise':
                    used = clip_precise(video_path, clip_path, start, end, choice)
                    export['encoder'] = used['label']
                else:
                    cut = clip_lossless(video_path, clip_path, start, end)
                    export['start_shift'] = max(export['start_shift'], round(start - cut, 2))
                clip_paths.append(clip_path)
                clip_scores.append(score)
                export['clips'] += 1
            except Exception as e:
                logger.error(f"Failed to clip {score.get('timestamp_str', score['timestamp'])}: {e}")

            clips_done += 1
            if progress_callback:
                progress_callback(clips_done / total_clips)

        if clip_paths:
            # Sanitize player name for folder/filename (allow alphanumeric including Chinese)
            safe_name = "".join(c for c in player if (c.isalnum() or c in ('_', '-', ' ')))
            safe_name = safe_name.strip() or 'unknown'

            # Create a directory for individual clips
            player_dir = os.path.join(output_dir, safe_name)
            os.makedirs(player_dir, exist_ok=True)

            # The clips of an earlier generation: their numbering is not valid any more, and some of
            # them may not belong to the shots of this generation (found later, or rejected since)
            for old in os.listdir(player_dir):
                if old.startswith(safe_name + '_') and old.endswith('.mp4'):
                    try:
                        os.remove(os.path.join(player_dir, old))
                    except OSError as e:
                        logger.warning(f"Could not remove old clip {old}: {e}")

            # Move/Copy clips to player directory with descriptive names
            for i, cp in enumerate(clip_paths):
                # Format: Player_01_00-12-30.mp4
                time_str = clip_scores[i]['timestamp_str'].replace(':', '-')
                shot_filename = f"{safe_name}_{i+1:03d}_{time_str}.mp4"
                shot_path = os.path.join(player_dir, shot_filename)

                try:
                    shutil.copy2(cp, shot_path)
                except Exception as e:
                    logger.error(f"Failed to save individual shot {shot_filename}: {e}")

            output_filename = f"highlight_{safe_name}.mp4"
            output_path = os.path.join(output_dir, output_filename)

            try:
                concat_videos(clip_paths, output_path, temp_dir)
                results[player] = {
                    'path': output_path,
                    'filename': output_filename,
                    'count': len(clip_paths),
                    'folder': safe_name
                }
            except Exception as e:
                logger.error(f"Failed to concat clips for {player}: {e}")

            # Cleanup temp clips (we already copied them to player_dir)
            for cp in clip_paths:
                if os.path.exists(cp):
                    os.remove(cp)

    # What an earlier generation left for players that have no clips in this one (all rejected since,
    # or renamed): their folders with the single clips, the ZIPs made from them, and their highlight videos
    kept_folders = {info['folder'] for info in results.values()}
    kept_files = {info['filename'] for info in results.values()}
    for name in os.listdir(output_dir):
        path = os.path.join(output_dir, name)
        stale = (name not in kept_folders) if os.path.isdir(path) else (
            (name.endswith('.zip') and name[:-4] not in kept_folders)
            or (name.startswith('highlight_') and name.endswith('.mp4') and name not in kept_files))
        if not stale:
            continue
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
        except OSError as e:
            logger.warning(f"Could not remove old output {name}: {e}")

    if export['clips'] == 0:
        raise RuntimeError("一个片段都没有导出成功")
    return results, export
