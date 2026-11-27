"""
Video Editor - Clips and concatenates video segments using FFmpeg.

A clip is taken out of the video by encoding it again, as H.264 with AAC sound in an MP4: the format
every phone, computer, TV and chat app plays without effort. So that the clips begin and end exactly
where they should, and that a highlight joined from them runs on without a jolt at the joins, a
hardware encoder is used where there is one.

(Copying the video stream without re-encoding was tried and dropped: a copy can only start at a
keyframe, and the pictures some cameras store right after one cannot be decoded once the video is cut
there, which made every join stutter.)

The clips of a highlight are joined by copying.
"""

import json
import logging
import os
import shutil
import subprocess
import threading

logger = logging.getLogger(__name__)

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


# ── finding out about the video ────────────────────────────────────────────────

def probe_video(video_path):
    """What the video is like (cached per file): codec, size, frame rate, pixel format, colours, sound."""
    key = ('probe', video_path, _stamp(video_path))
    with _lock:
        if key in _cache:
            return _cache[key]

    info = {'vcodec': '', 'pix_fmt': '', 'width': 0, 'height': 0, 'rate': '',
            'space': '', 'primaries': '', 'transfer': '', 'sound': True}
    ffprobe = tool_path('ffprobe')
    if ffprobe:
        try:
            result = run_tool([ffprobe, '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                               'stream=codec_name,pix_fmt,width,height,r_frame_rate,'
                               'color_primaries,color_transfer,color_space', '-of', 'json', video_path], 60)
            stream = (json.loads(result.stdout).get('streams') or [{}])[0]
            info.update(vcodec=stream.get('codec_name') or '', pix_fmt=stream.get('pix_fmt') or '',
                        width=stream.get('width') or 0, height=stream.get('height') or 0,
                        rate=stream.get('r_frame_rate') or '',
                        space=stream.get('color_space') or '', primaries=stream.get('color_primaries') or '',
                        transfer=stream.get('color_transfer') or '')
        except (subprocess.TimeoutExpired, OSError, ValueError) as e:
            logger.warning(f"Could not read the video format of {video_path}: {e}")
        try:
            result = run_tool([ffprobe, '-v', 'error', '-select_streams', 'a:0', '-show_entries',
                               'stream=codec_type', '-of', 'csv=p=0', video_path], 60)
            info['sound'] = bool(result.stdout.strip())
        except (subprocess.TimeoutExpired, OSError):
            pass

    unknown = ('', 'unknown', 'unspecified', 'reserved')
    colours = []
    if _swscale_matrix(info['space']) is None:
        # Left as it is; what else is known about its colours is passed on (e.g. a BT.601 video that
        # names its transfer)
        for name, option in (('primaries', '-color_primaries'), ('transfer', '-color_trc')):
            if info[name] not in unknown:
                colours += [option, info[name]]
    else:
        # In BT.709, or converted to it by _picture_filters: that is what the clips say they are
        converted = info['space'] != 'bt709'
        colours = ['-colorspace', 'bt709']
        for name, option in (('primaries', '-color_primaries'), ('transfer', '-color_trc')):
            colours += [option, 'bt709' if converted or info[name] == 'bt709' else info[name]]
    info['colours'] = colours

    with _lock:
        _cache[key] = info
    return info


# How ffmpeg's scale filter names the colour matrices (None: one it cannot convert from)
_SWS_MATRICES = {'bt709': 'bt709', 'bt2020': 'bt2020', 'bt2020nc': 'bt2020', 'bt2020_ncl': 'bt2020',
                 'bt2020c': 'bt2020', 'bt2020_cl': 'bt2020', 'smpte170m': 'smpte170m', 'bt470bg': 'smpte170m',
                 'fcc': 'fcc', 'smpte240m': 'smpte240m'}


def _swscale_matrix(space):
    return _SWS_MATRICES.get(space)


def _frame_rate(rate):
    """The frame rate the clips are encoded at (for "-r"): that of the video, but at most 60, which is
    as fast as every phone plays H.264 smoothly. '' when the video does not say."""
    try:
        num, den = rate.split('/')
        fps = float(num) / float(den)
    except (AttributeError, ValueError, ZeroDivisionError):
        return ''
    if fps > 60:
        return '60'
    return rate if fps > 0 else ''


def first_frame_time(video_path, start_time):
    """When the first picture shown at or after start_time is shown (seconds); None when not found.

    A wanted start lies between two pictures now and then. Starting the clip on the picture after it
    makes that picture the one at 0 s of the clip; otherwise the clip would begin with a gap of one
    picture, which shows as a jolt when it is joined to the clip before it.
    """
    ffprobe = tool_path('ffprobe')
    if not ffprobe:
        return None
    # Packets only (nothing is decoded), from the seek point before the time up to a little past it
    try:
        result = run_tool([ffprobe, '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                           'packet=pts_time', '-of', 'csv=p=0',
                           '-read_intervals', f'{max(0.0, start_time - 0.5):.3f}%{start_time + 0.3:.3f}',
                           video_path], 120)
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.warning(f"Could not look for the pictures near {start_time:.2f}s of {video_path}: {e}")
        return None
    shown = []
    for line in result.stdout.split():
        try:
            time = float(line.rstrip(','))
        except ValueError:
            continue
        if time >= start_time - 0.002:
            shown.append(time)
    return min(shown) if shown else None


def picture_end(path):
    """When the last picture of a video file has been shown (seconds), or None when not known."""
    ffprobe = tool_path('ffprobe')
    if not ffprobe:
        return None
    try:
        result = run_tool([ffprobe, '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                           'stream=start_time,duration', '-of', 'csv=p=0', path], 60)
        start, duration = (float(value) for value in result.stdout.strip().split(',')[:2])
    except (subprocess.TimeoutExpired, OSError, ValueError):
        return None
    return start + duration


# ── taking one clip out of the video ───────────────────────────────────────────

def _picture_filters(info):
    """Ways to turn the pictures of the video into what H.264 clips are made of, best first: 8-bit
    4:2:0, at most 1080p, in BT.709. [''] when there is nothing to do (then the pictures can stay on
    the GPU from decoding to encoding).

    Whatever the pictures are converted with, they end up marked as the ordinary BT.709 that they
    are now, without the HDR information they came with: ffmpeg takes the colour information for the
    file from the pictures themselves, and an HDR mark on ordinary pictures makes HDR phones show
    them washed out."""
    bigger = info['width'] > 1920 or info['height'] > 1080
    resize = ['scale=-2:1080'] if bigger else []
    matrix = _swscale_matrix(info['space'])
    other = matrix is not None and matrix != 'bt709'
    deep = info['pix_fmt'] != 'yuv420p'
    if not (bigger or other or deep):
        return ['']
    plain_mark = 'setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709,sidedata=mode=delete'
    chains = []
    if info['transfer'] == 'smpte2084':      # PQ (HDR10) has to be tone-mapped; zimg can
        chains.append(','.join(resize + [
            'zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,'
            'tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p', plain_mark]))
    if other:
        # Phones record HDR (HLG) in BT.2020. The picture values themselves are kept: HLG is made to
        # look right on an ordinary screen too. Only the range of colours is narrowed to what such a
        # screen shows. zimg does that exactly, and fast
        chains.append(','.join(resize + ['zscale=m=bt709:p=bt709', 'format=yuv420p', plain_mark]))
    # What every ffmpeg can do (its own scale filter): slower, and it only changes the matrix, not
    # the range of colours, which come out a little less saturated
    plain = resize + ([f'scale=iw:ih:in_color_matrix={matrix}:out_color_matrix=bt709'] if other else [])
    chains.append(','.join(plain + ['format=yuv420p', plain_mark]))
    return chains


def _encoding_candidates(info):
    """Ways to encode the pictures again as H.264, best first. That is the video format every phone,
    computer, TV and chat app plays without effort; H.265 in 10 bits, what phones record in HDR, is not.

    `filters` is one way of _picture_filters. `decoder` says how the video is decoded: on the GPU
    (staying there for the encoder too, when the pictures need nothing done to them), on the GPU with
    the pictures handed to the CPU, or on the CPU."""
    nv_gpu = ['-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda']
    nv_cpu = ['-hwaccel', 'cuda']
    vt_gpu = ['-hwaccel', 'videotoolbox', '-hwaccel_output_format', 'videotoolbox']
    vt_cpu = ['-hwaccel', 'videotoolbox']
    high = ['-preset', 'p6', '-tune', 'hq', '-rc', 'vbr', '-cq', '19', '-b:v', '0',
            '-spatial-aq', '1', '-temporal-aq', '1', '-rc-lookahead', '30', '-profile:v', 'high']
    old = ['-preset', 'slow', '-rc', 'vbr_hq', '-cq', '19', '-b:v', '0', '-profile:v', 'high']   # before ffmpeg 4.4
    # The quality knob of VideoToolbox is not a bitrate. Measured on the user's video against a
    # lossless copy of the converted pictures: 70 gives 16 Mbps at SSIM 0.988; x264 at crf 18 gives
    # 9 Mbps at SSIM 0.990
    apple = ['-c:v', 'h264_videotoolbox', '-q:v', '70', '-profile:v', 'high']
    x264 = ['-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-profile:v', 'high']
    nvenc_label, vt_label = '显卡编码（NVENC）', '硬件编码（VideoToolbox）'
    candidates = []
    for filters in _picture_filters(info):
        kept = ['kept'] if filters == '' else []     # "kept": the pictures stay on the GPU
        for spelling in (high, old):
            for decoder in ([nv_gpu] if kept else []) + [nv_cpu, []]:
                candidates.append(dict(name='h264_nvenc', label=nvenc_label, filters=filters, decoder=decoder,
                                       encoder=['-c:v', 'h264_nvenc', *spelling]))
        for decoder in ([vt_gpu] if kept else []) + [vt_cpu, []]:
            candidates.append(dict(name='h264_videotoolbox', label=vt_label, filters=filters, decoder=decoder,
                                   encoder=apple))
        candidates.append(dict(name='libx264', label='CPU 编码（x264）', filters=filters, decoder=[], encoder=x264))
    return candidates


def _encoder_works(candidate, info):
    """Whether this ffmpeg can convert and encode like this: tried once on a black frame that is
    given the video's pixel format and colours. (Hardware decoding is not part of that, nothing is
    decoded there: a candidate whose hardware decoding fails on a real clip is dropped there.)"""
    test = (tuple(candidate['encoder']), candidate['filters'], info['pix_fmt'])
    with _lock:
        if ('encoder', test) in _cache:
            return _cache[('encoder', test)]
    source = f"color=black:s=256x256:r=30:d=0.2,format={info['pix_fmt'] or 'yuv420p'}"
    named = [f'{option}={info[name]}' for name, option in (('space', 'colorspace'), ('primaries', 'color_primaries'),
                                                           ('transfer', 'color_trc')) if info[name]]
    if named:
        source += ',setparams=' + ':'.join(named)
    cmd = ['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', source]
    if candidate['filters']:
        cmd += ['-vf', candidate['filters']]
    cmd += [*candidate['encoder'], '-pix_fmt', 'yuv420p', '-an', '-f', 'null', '-']
    try:
        ok = run_tool(cmd, 60).returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        ok = False
    with _lock:
        _cache[('encoder', test)] = ok
    return ok


def usable_encoders(video_path):
    """The candidates of _encoding_candidates that this ffmpeg can really use, best first."""
    info = probe_video(video_path)
    return [c for c in _encoding_candidates(info) if _encoder_works(c, info)]


def _remove(path):
    if not path:
        return
    try:
        os.remove(path)
    except OSError:
        pass


def encode_pictures(video_path, output_path, start_time, end_time, choice):
    """Encodes the pictures of [start, end] of the video as H.264, without sound.

    The clip begins with the picture that is shown first at or after start_time, at 0 s of the clip,
    at one steady frame rate. Returns (the candidate used, the moment of the video the clip begins
    with, when its last picture has been shown).

    `choice` carries the encoders from one clip to the next: {'encoders': [...], 'current': index}.
    An encoder that fails on a real clip (e.g. one whose hardware decoding is not there) is dropped.
    """
    if 'encoders' not in choice:
        choice['encoders'], choice['current'] = usable_encoders(video_path), 0
    if not choice['encoders']:
        raise RuntimeError("这个 ffmpeg 没有可用的编码器")
    info = probe_video(video_path)

    # Start on a picture, so that the clip begins with its first picture at 0 s (see first_frame_time);
    # a hair before it, so that the seeking cannot land past that picture
    shown = first_frame_time(video_path, start_time)
    if shown is not None and shown < end_time:
        start_time = shown - 0.000001

    used = None
    while choice['current'] < len(choice['encoders']):
        candidate = choice['encoders'][choice['current']]
        cmd = ['ffmpeg', '-y', *candidate['decoder'], '-ss', f'{start_time:.6f}', '-i', video_path,
               '-t', f'{end_time - start_time:.6f}', '-map', '0:v:0', '-an', *candidate['encoder']]
        if candidate['filters']:
            cmd += ['-vf', candidate['filters']]
        # With a hardware decoder the frames keep their format; otherwise the encoder is told it
        if not candidate['decoder'] or '-hwaccel_output_format' not in candidate['decoder']:
            cmd += ['-pix_fmt', 'yuv420p']
        rate = _frame_rate(info['rate'])
        if rate:
            cmd += ['-r', rate]     # one steady rate: some players stumble over uneven ones
        cmd += [*info['colours'], output_path]

        logger.info(f"Encoding {start_time:.2f}s - {end_time:.2f}s with {candidate['name']}"
                    f"{' (hardware decoding)' if candidate['decoder'] else ''} -> {output_path}")
        try:
            result = run_tool(cmd, 900)
        except (subprocess.TimeoutExpired, OSError) as e:
            result = None
            logger.error(f"FFmpeg with {candidate['name']} did not finish: {e}")
        if result is not None and result.returncode == 0 and picture_end(output_path) is not None:
            used = candidate
            break
        if result is not None:
            logger.error(f"FFmpeg with {candidate['name']} failed: {result.stderr[-500:]}")
        choice['current'] += 1
        logger.warning(f"{candidate['name']} does not work on this clip, trying the next encoder")
        _remove(output_path)
    if used is None:
        raise RuntimeError("没有能用的编码器，请更新 FFmpeg 后重试")
    return used, start_time, picture_end(output_path)


SOUND_RATE = 48000        # the rate the sound of the clips is in


def extract_sound(video_path, output_path, start_time, length):
    """The sound of the video from start_time for `length` seconds, as plain samples in a WAV file:
    48 kHz, stereo, exactly that long (cut, or made up with silence at the end of the video).

    Plain samples, not packets: AAC sound comes in packets of about 21 ms, so joined as packets it
    leaves a hole of up to one packet at every join. Made one track sample by sample, it runs on
    without one."""
    cmd = ['ffmpeg', '-v', 'error', '-y', '-ss', f'{start_time:.6f}', '-i', video_path, '-t', f'{length:.6f}',
           '-map', '0:a:0', '-c:a', 'pcm_s16le', '-ar', str(SOUND_RATE), '-ac', '2', output_path]
    logger.info(f"Sound of {start_time:.2f}s - {start_time + length:.2f}s -> {output_path}")
    try:
        result = run_tool(cmd, 300)
    except (subprocess.TimeoutExpired, OSError) as e:
        result = None
        logger.error(f"FFmpeg did not finish the sound of {output_path}: {e}")
    if result is None or result.returncode != 0:
        if result is not None:
            logger.error(f"FFmpeg sound error: {result.stderr[-500:]}")
        _remove(output_path)
        raise RuntimeError("FFmpeg failed: could not read the sound")
    if not _fit_wave(output_path, round(length * SOUND_RATE)):
        raise RuntimeError("could not read the sound")


def _wave_data_at(path):
    """Where the samples of a WAV file begin, or None if it is not one. Leaves the file open at that
    point."""
    f = open(path, 'rb')
    head = f.read(12)
    if len(head) < 12 or head[:4] != b'RIFF' or head[8:12] != b'WAVE':
        f.close()
        return None
    while True:
        header = f.read(8)
        if len(header) < 8:
            f.close()
            return None
        kind, size = header[:4], int.from_bytes(header[4:8], 'little')
        if kind == b'data':
            return f
        f.seek(size + (size & 1), 1)


def _fit_wave(path, samples):
    """Cuts or pads a 48 kHz stereo 16-bit WAV file to exactly `samples` frames (a frame is both
    channels). False when the file is not such a WAV."""
    try:
        f = _wave_data_at(path)
        if f is None:
            return False
        data_at, have = f.tell(), os.path.getsize(path) - f.tell()
        f.close()
        wanted = samples * 4
        with open(path, 'r+b') as f:
            if have > wanted:
                f.seek(data_at + wanted)
                f.truncate()
            elif have < wanted:
                f.seek(0, 2)
                f.write(b'\0' * (wanted - have))
            f.seek(4)
            f.write((36 + wanted).to_bytes(4, 'little'))     # the sizes in the header follow
            f.seek(data_at - 4)
            f.write(wanted.to_bytes(4, 'little'))
        return True
    except OSError as e:
        logger.error(f"Could not make the sound file {path} the right length: {e}")
        return False


def _wave_header(samples):
    """A WAV header for 48 kHz stereo 16-bit, `samples` frames."""
    return (b'RIFF' + (36 + samples * 4).to_bytes(4, 'little') + b'WAVEfmt ' + (16).to_bytes(4, 'little')
            + (1).to_bytes(2, 'little') + (2).to_bytes(2, 'little') + SOUND_RATE.to_bytes(4, 'little')
            + (SOUND_RATE * 4).to_bytes(4, 'little') + (4).to_bytes(2, 'little') + (16).to_bytes(2, 'little')
            + b'data' + (samples * 4).to_bytes(4, 'little'))


def join_waves(paths, lengths, output_path):
    """Puts the sound of clips one after the other into one WAV file, each `length` seconds long; a
    clip without a sound file is silence. The result is exactly as long as the pictures joined."""
    lengths = [round(length * SOUND_RATE) * 4 for length in lengths]
    with open(output_path, 'wb') as out:
        out.write(_wave_header(sum(lengths) // 4))
        for path, wanted in zip(paths, lengths):
            f = _wave_data_at(path) if path and os.path.exists(path) else None
            if f is None:
                out.write(b'\0' * wanted)
                continue
            with f:
                left = wanted
                while left > 0:
                    chunk = f.read(min(1 << 20, left))
                    if not chunk:
                        break
                    out.write(chunk)
                    left -= len(chunk)
                out.write(b'\0' * left)


def put_sound(video_path, wave_path, output_path):
    """An MP4 with the pictures of a video file (copied) and, when there is a sound file, that sound
    encoded as AAC."""
    cmd = ['ffmpeg', '-y', '-i', video_path]
    if wave_path:
        cmd += ['-i', wave_path]
    cmd += ['-map', '0:v:0']
    if wave_path:
        cmd += ['-map', '1:a:0', '-c:a', 'aac', '-b:a', '192k']
    cmd += ['-c:v', 'copy', '-movflags', '+faststart', output_path]
    logger.info(f"Putting sound to {os.path.basename(video_path)} -> {output_path}")
    try:
        result = run_tool(cmd, 300)
    except (subprocess.TimeoutExpired, OSError) as e:
        result = None
        logger.error(f"FFmpeg did not finish {output_path}: {e}")
    if result is None or result.returncode != 0:
        if result is not None:
            logger.error(f"FFmpeg error: {result.stderr[-500:]}")
        raise RuntimeError("FFmpeg failed: could not put the sound to the pictures")


# ── joining the clips ──────────────────────────────────────────────────────────

def concat_videos(clip_paths, output_path, temp_dir, durations=None):
    """
    Concatenate video clips using the ffmpeg concat demuxer.

    `durations` says how long each clip is shown (picture_end of the clip). Given, the next clip
    begins exactly where the previous one ends; without it ffmpeg estimates from what the files say,
    which is off by a few milliseconds now and then (a picture that comes late at the join).
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
        for i, path in enumerate(clip_paths):
            # Escape single quotes in path
            escaped = path.replace("'", "'\\''")
            # Convert backslashes to forward slashes for Windows ffmpeg paths
            escaped = escaped.replace('\\', '/')
            f.write(f"file '{escaped}'\n")
            if durations and durations[i] is not None:
                f.write(f"duration {durations[i]:.6f}\n")

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

UNLABELLED = '未标注'
GOAL, ASSIST = 'goal', 'assist'
ROLE_IN_FILENAME = {GOAL: '进球', ASSIST: '助攻'}

# Until 2026-11-26 the defaults were 5 s before and 3 s after, and the review page saved exactly
# that pair with every score whose clip was opened. Such a pair follows the defaults: it was not
# set by hand.
DEFAULTS_SET_BEFORE = (5.0, 3.0)


def clean_name(value):
    """A name as it is compared and shown: nothing around it, single spaces inside."""
    if value is None:
        return ''
    return ' '.join(str(value).split())


def participants(score):
    """Who is in the clip of one score and how: [(name, 'goal' or 'assist')], the scorer first.

    An assist named twice, or who is the scorer as well, counts once, as the scorer.
    """
    scorer = clean_name(score.get('player'))
    found = [(scorer, GOAL)] if scorer else []
    assists = score.get('assists')
    for value in assists if isinstance(assists, list) else []:
        name = clean_name(value)
        if name and all(name != other for other, _ in found):
            found.append((name, ASSIST))
    return found


def clip_range(score, clip_before, clip_after):
    """(seconds before, seconds after) the clip of one score reaches: its own, or the defaults."""
    try:
        saved = (float(score['clip_before']), float(score['clip_after']))
    except (KeyError, TypeError, ValueError):
        return clip_before, clip_after
    if score.get('clip_custom') or saved != DEFAULTS_SET_BEFORE:
        return saved
    return clip_before, clip_after


def file_name(name):
    """A name as the name of a file or a folder: letters and digits (Chinese included), '_', '-', ' '."""
    safe = ''.join(c for c in name if c.isalnum() or c in ('_', '-', ' '))
    return safe.strip() or 'unknown'


def generate_highlights(video_path, scores, output_dir, temp_dir,
                        clip_before=6.0, clip_after=2.0, progress_callback=None):
    """
    Generate one highlight video per person: the makes they scored and the ones they assisted.

    Args:
        video_path: path to source video
        scores: score dicts with 'player', 'assists' (a list of names), 'timestamp' and 'confirmed'
        output_dir: directory to save highlight videos
        temp_dir: directory for temporary clip files
        clip_before: seconds before scoring moment
        clip_after: seconds after scoring moment
        progress_callback: function(progress: float) called with 0.0-1.0

    Returns:
        (dict mapping person -> info about its highlight video, dict about how it was exported)
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    # Which clips are needed, in video order, and who is in each of them. A clip that is one
    # person's goal and another person's assist is made once and goes into both highlights.
    clips = []      # [(score, [(name, role), ...])]
    member = {}     # name -> [(number of the clip, role)]
    for score in sorted(scores, key=lambda s: s.get('timestamp', 0)):
        if not score.get('confirmed', True):
            continue
        who = participants(score) or [(UNLABELLED, GOAL)]
        for name, role in who:
            member.setdefault(name, []).append((len(clips), role))
        clips.append((score, who))

    info = probe_video(video_path)
    results = {}
    # How the clips were made, for the page that offers the downloads
    export = {'encoder': None, 'clips': 0, 'colour': ''}
    choice = {}
    paths = [os.path.join(temp_dir, f"clip_{i:04d}.mp4") for i in range(len(clips))]   # pictures only
    sounds = [os.path.join(temp_dir, f"clip_{i:04d}.wav") for i in range(len(clips))]
    made = [None] * len(clips)    # the clip files that were made; None where it failed
    begins = [0.0] * len(clips)   # the moment of the video each clip begins with
    ends = [None] * len(clips)    # when the last picture of each clip has been shown

    for i, (score, _) in enumerate(clips):
        before, after = clip_range(score, clip_before, clip_after)
        start = max(0, score['timestamp'] - before)
        # Not cut at the length reported for the video: it is an estimate, and past the real end
        # ffmpeg simply stops
        end = score['timestamp'] + after

        try:
            used, begins[i], ends[i] = encode_pictures(video_path, paths[i], start, end, choice)
            export['encoder'] = used['label']
            # How the colours were converted, when the video is not in the ordinary BT.709: exactly
            # (zimg), or by ffmpeg's own scale filter, which leaves them a little less saturated
            export['colour'] = ('zimg' if 'zscale' in used['filters'] else 'basic') if used['filters'] else ''
            made[i] = paths[i]
            export['clips'] += 1
            if info['sound']:
                try:
                    extract_sound(video_path, sounds[i], begins[i], ends[i])
                except Exception as e:
                    # The clip is silent then; its place in a highlight is kept, with silence
                    logger.error(f"Could not read the sound of {score.get('timestamp_str', score['timestamp'])}: {e}")
        except Exception as e:
            logger.error(f"Failed to clip {score.get('timestamp_str', score['timestamp'])}: {e}")

        if progress_callback:
            progress_callback((i + 1) / max(1, len(clips)))

    # The people by how many clips they are in; the makes that nobody was named in last
    people = sorted(member, key=lambda name: (name == UNLABELLED, -len(member[name])))
    folders = set()
    for name in people:
        entries = [(i, role) for i, role in member[name] if made[i] is not None]
        if not entries:
            continue
        # Names that are alike as folder names ("A·B" and "AB") get their own: the clips of the one
        # would be taken for the earlier clips of the other and removed
        safe_name = file_name(name)
        folder, n = safe_name, 2
        while folder.lower() in folders:
            folder, n = f"{safe_name}_{n}", n + 1
        folders.add(folder.lower())

        def sound_of(number):
            return sounds[number] if info['sound'] and os.path.exists(sounds[number]) else None

        # The single clips of the person, with sound
        person_dir = os.path.join(output_dir, folder)
        os.makedirs(person_dir, exist_ok=True)

        # The clips of an earlier generation: their numbering is not valid any more, and some of
        # them may not belong to the shots of this generation (found later, rejected since)
        for old in os.listdir(person_dir):
            if old.startswith(folder + '_') and old.endswith('.mp4'):
                try:
                    os.remove(os.path.join(person_dir, old))
                except OSError as e:
                    logger.warning(f"Could not remove old clip {old}: {e}")

        numbers = [number for number, _ in entries]
        goals = 0
        for i, (number, role) in enumerate(entries):
            goals += role == GOAL
            # Format: Player_001_进球_00-12-30.mp4
            time_str = clips[number][0]['timestamp_str'].replace(':', '-')
            shot_filename = f"{folder}_{i+1:03d}_{ROLE_IN_FILENAME[role]}_{time_str}.mp4"
            try:
                put_sound(made[number], sound_of(number), os.path.join(person_dir, shot_filename))
            except Exception as e:
                logger.error(f"Failed to save individual shot {shot_filename}: {e}")

        # The pictures of the clips joined, and their sound as one track, in one file
        lengths = [ends[number] for number in numbers]
        if len(numbers) == 1:
            joined = made[numbers[0]]
        else:
            joined = os.path.join(temp_dir, f"join_{len(results):04d}.mp4")
        wave = os.path.join(temp_dir, f"sound_{len(results):04d}.wav") if info['sound'] else None

        output_filename = f"highlight_{folder}.mp4"
        output_path = os.path.join(output_dir, output_filename)
        try:
            if len(numbers) > 1:
                concat_videos([made[number] for number in numbers], joined, temp_dir, lengths)
            if wave:
                join_waves([sound_of(number) for number in numbers], lengths, wave)
            put_sound(joined, wave, output_path)
            results[name] = {
                'path': output_path,
                'filename': output_filename,
                'count': len(entries),
                'goals': goals,
                'assists': len(entries) - goals,
                'folder': folder
            }
        except Exception as e:
            logger.error(f"Failed to join the clips for {name}: {e}")
        finally:
            if len(numbers) > 1:
                _remove(joined)
            _remove(wave)

    # Cleanup temp clips (the persons' directories have their copies), those of failed clips too
    for path in paths + sounds:
        _remove(path)

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
