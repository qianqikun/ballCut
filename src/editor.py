"""
Video Editor - Clips and concatenates video segments using FFmpeg.
Uses stream copy (no re-encoding) for maximum speed.
"""

import subprocess
import os
import logging

logger = logging.getLogger(__name__)


def check_ffmpeg():
    """Check if ffmpeg is available."""
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def clip_video(input_path, output_path, start_time, end_time):
    """
    Extract a video clip using ffmpeg.

    Uses re-encoding for precise cuts (not just keyframe boundaries).
    """
    duration = end_time - start_time
    cmd = [
        'ffmpeg', '-y',
        '-ss', f'{start_time:.2f}',
        '-i', input_path,
        '-t', f'{duration:.2f}',
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '23',
        '-c:a', 'aac',
        '-b:a', '128k',
        '-avoid_negative_ts', 'make_zero',
        '-movflags', '+faststart',
        output_path
    ]

    logger.info(f"Clipping: {start_time:.1f}s - {end_time:.1f}s -> {output_path}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if result.returncode != 0:
        logger.error(f"FFmpeg clip error: {result.stderr[-500:]}")
        raise RuntimeError(f"FFmpeg failed: {result.stderr[-200:]}")

    return output_path


def concat_videos(clip_paths, output_path, temp_dir):
    """
    Concatenate multiple video clips using ffmpeg concat demuxer.
    """
    if not clip_paths:
        raise ValueError("No clips to concatenate")

    if len(clip_paths) == 1:
        # Just copy the single clip
        import shutil
        shutil.copy2(clip_paths[0], output_path)
        return output_path

    # Create concat list file
    concat_file = os.path.join(temp_dir, 'concat_list.txt')
    with open(concat_file, 'w') as f:
        for path in clip_paths:
            # Escape single quotes in path
            escaped = path.replace("'", "'\\''")
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
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    # Cleanup concat file
    if os.path.exists(concat_file):
        os.remove(concat_file)

    if result.returncode != 0:
        logger.error(f"FFmpeg concat error: {result.stderr[-500:]}")
        raise RuntimeError(f"FFmpeg concat failed: {result.stderr[-200:]}")

    return output_path


def generate_highlights(video_path, scores, output_dir, temp_dir,
                        clip_before=5.0, clip_after=3.0,
                        progress_callback=None):
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

    Returns:
        dict mapping player_name -> output_video_path
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

    # Get video duration
    import cv2
    cap = cv2.VideoCapture(video_path)
    video_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    total_clips = sum(len(v) for v in player_scores.values())
    clips_done = 0
    results = {}

    for player, p_scores in player_scores.items():
        clip_paths = []

        for i, score in enumerate(p_scores):
            cb = score.get('clip_before', clip_before)
            ca = score.get('clip_after', clip_after)
            start = max(0, score['timestamp'] - cb)
            end = min(video_duration, score['timestamp'] + ca)

            clip_filename = f"clip_{player}_{i:04d}.mp4"
            clip_path = os.path.join(temp_dir, clip_filename)

            try:
                clip_video(video_path, clip_path, start, end)
                clip_paths.append(clip_path)
            except Exception as e:
                logger.error(f"Failed to clip {score['timestamp_str']}: {e}")

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

            # Move/Copy clips to player directory with descriptive names
            final_clip_paths = []
            import shutil
            for i, cp in enumerate(clip_paths):
                # Format: Player_01_00-12-30.mp4
                time_str = p_scores[i]['timestamp_str'].replace(':', '-')
                shot_filename = f"{safe_name}_{i+1:03d}_{time_str}.mp4"
                shot_path = os.path.join(player_dir, shot_filename)
                
                try:
                    shutil.copy2(cp, shot_path)
                    final_clip_paths.append(shot_path)
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

    return results
