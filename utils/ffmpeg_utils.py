import subprocess
import os
import re
import shutil
import math
from typing import Tuple

from utils.gpu_backend import (
    is_nvidia, hevc_encoder, av1_encoder,
    encoder_args, hwaccel_input_args, speed_presets,
)


# Quality preset mapping (backend-agnostic)
QUALITY_PRESETS = {'ultra': '18', 'high': '24', 'normal': '26', 'low': '28'}


def SPEED_PRESETS_get() -> dict[str, str]:
    return speed_presets()


# Backwards-compatible name: resolved lazily at attribute access.
# Some callers do `from ffmpeg_utils import SPEED_PRESETS`.
class _SpeedPresetsProxy(dict):
    def __getitem__(self, key):
        return speed_presets()[key]
    def keys(self):
        return speed_presets().keys()
    def __iter__(self):
        return iter(speed_presets())
    def __contains__(self, key):
        return key in speed_presets()


SPEED_PRESETS = _SpeedPresetsProxy()


def parse_ffmpeg_progress(line: str) -> str:
    """Only extract useful fields from ffmpeg progress line"""
    parts = []
    for field in ['time=', 'elapsed=', 'speed=']:
        match = re.search(rf'{field}(\S+)', line)
        if match:
            parts.append(f"{field}{match.group(1)}")

    # \033[K clears from cursor to end of line to prevent leftover chars
    return (' '.join(parts) + '\033[K') if parts else line.strip()


def run_ffmpeg_with_progress(
    cmd: list[str],
    progress_prefix: str = "",
    cwd: str | None = None
) -> tuple[int, str]:
    """Run an FFmpeg subprocess with real time progress display"""
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=cwd)
    stderr_lines = []
    for line in process.stderr:
        stderr_lines.append(line)
        if 'frame=' in line:
            print(f"\r{progress_prefix}{parse_ffmpeg_progress(line)}", end='', flush=True)
    process.wait()
    print()
    return process.returncode, "".join(stderr_lines)


def get_video_info(video_path: str) -> Tuple[int, int, float, float]:
    """
    Get video dimensions, framerate, and duration
    Returns: width, height, fps, duration_seconds
    """
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,r_frame_rate:format=duration',
        '-of', 'csv=p=0',
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"❌ ffprobe failed: {result.stderr}")

    lines = result.stdout.strip().split('\n')
    w, h, fps_str = lines[0].split(',')
    duration = float(lines[1]) if len(lines) > 1 else 0

    # parse fps fraction (60/1)
    if '/' in fps_str:
        num, den = fps_str.split('/')
        fps = float(num) / float(den)
    else:
        fps = float(fps_str)

    return int(w), int(h), fps, duration


def is_10bit(video_path: str) -> bool:
    """Check if video uses a 10-bit pixel format (like AV1)"""
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=pix_fmt',
        '-of', 'csv=p=0',
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return False
    return '10' in result.stdout.strip()


def get_video_codec(video_path: str) -> str:
    """Return the codec name of the first video stream (av1, hevc, h264)"""
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=codec_name',
        '-of', 'csv=p=0',
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return ''
    return result.stdout.strip()


_av1_encode_supported: bool | None = None

def supports_av1_encode() -> bool:
    """Check if the GPU supports AV1 encoding"""
    global _av1_encode_supported
    if _av1_encode_supported is not None:
        return _av1_encode_supported

    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
        '-f', 'lavfi', '-i', 'nullsrc=s=256x256:d=0.04:r=25',
        '-c:v', av1_encoder(), '-frames:v', '1',
        '-f', 'null', '-'
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    _av1_encode_supported = result.returncode == 0
    return _av1_encode_supported


def get_output_encoder(video_path: str) -> str:
    """Return the AV1 encoder if input is AV1 and GPU supports it, otherwise HEVC."""
    if get_video_codec(video_path) == 'av1' and supports_av1_encode():
        return av1_encoder()
    return hevc_encoder()


def concatenate_videos(video_list: list[str], output_path: str) -> str:
    """
    Concatenate multiple videos. Handles mixed formats (grayscale, different codecs etc)
    Uses a temp file for filter_complex to avoid cmd length limits
    """
    BATCH_SIZE = 50
    n = len(video_list)

    # process in batches to avoid ffmpeg OOM with hundreds of inputs
    if n > BATCH_SIZE:
        base_path, ext = os.path.splitext(os.path.abspath(output_path))
        temp_batches = []
        try:
            for i in range(0, n, BATCH_SIZE):
                batch_files = video_list[i:i + BATCH_SIZE]
                batch_out = f"{base_path}_batch_{i//BATCH_SIZE}{ext}"
                temp_batches.append(batch_out)
                concatenate_videos(batch_files, batch_out)
            return concatenate_videos(temp_batches, output_path)
        finally:
            for tb in temp_batches:
                if os.path.exists(tb):
                    try:
                        os.remove(tb)
                    except OSError:
                        pass

    # find common directory for relative paths
    abs_videos = [os.path.abspath(v) for v in video_list]
    abs_output = os.path.abspath(output_path)
    common_dir = os.path.commonpath(abs_videos)
    if not os.path.isdir(common_dir):
        common_dir = os.path.dirname(common_dir)

    # verify files and build relative inputs
    rel_videos = []
    for video in abs_videos:
        if not os.path.exists(video):
            raise RuntimeError(f"❌ File missing: {video}")
        rel_videos.append(os.path.relpath(video, common_dir))

    rel_output = os.path.relpath(abs_output, common_dir)

    # build inputs
    inputs = []
    for rel in rel_videos:
        inputs.extend(['-i', rel])

    # build filter_complex
    n = len(video_list)
    filter_parts = []
    for i in range(n):
        filter_parts.append(f"[{i}:v]format=yuv420p[v{i}]")
    concat_inputs = "".join(f"[v{i}]" for i in range(n))
    filter_parts.append(f"{concat_inputs}concat=n={n}:v=1:a=0[outv]")
    filter_complex = ";".join(filter_parts)

    # write filter_complex to temp file to avoid cmd length limits
    filter_file = os.path.join(common_dir, "_concat_filter.txt")
    with open(filter_file, 'w') as f:
        f.write(filter_complex)

    try:
        cmd = [
            'ffmpeg', '-y',
            *inputs,
            '-/filter_complex', '_concat_filter.txt',
            '-map', '[outv]',
            '-c:v', hevc_encoder(), *encoder_args(cq='18', extras=False),
            rel_output
        ]

        rc, _ = run_ffmpeg_with_progress(cmd, cwd=common_dir)

        if rc != 0:
            raise RuntimeError(f"❌ FFmpeg concatenation failed")

        if not os.path.exists(abs_output):
            raise RuntimeError(f"❌ Concat output not created: {output_path}")
    finally:
        if os.path.exists(filter_file):
            os.remove(filter_file)

    return output_path


def sync_mask_to_video(mask_path: str, fps: float, frame_offset: int = 0) -> str:
    """
    Re-encode a mask video so its timeline matches the target video:

    - 0 frame_offset: forces CFR at the target fps
    - positive frame_offset: trim frames from the start (mask catches up / plays earlier)
    - negative frame_offset: pad black frames at the start (mask delayed / plays later)

    Creates a _synced file next to the original and returns path to the synced file
    """
    frame_duration = abs(frame_offset) / fps

    if frame_offset > 0:
        # positive = trim from start so mask content arrives earlier
        vf = f"trim=start={frame_duration},setpts=PTS-STARTPTS"
    elif frame_offset < 0:
        # negative = pad black at start so mask content arrives later
        vf = f"tpad=start_duration={frame_duration}:color=black"
    else:
        vf = "null"

    base, ext = os.path.splitext(mask_path)
    synced_path = f"{base}_synced{ext}"

    cmd = [
        'ffmpeg', '-y',
        '-i', mask_path,
        '-r', f"{fps}", '-vsync', 'cfr',
        '-vf', vf,
        '-c:v', hevc_encoder(), *encoder_args(cq='18', extras=False),
        synced_path,
    ]

    rc, stderr_text = run_ffmpeg_with_progress(cmd)

    if rc != 0:
        error_msg = "".join(stderr_text.splitlines(True)[-20:])
        raise RuntimeError(f"❌ Mask sync failed: {error_msg}")

    return synced_path


def extract_left_eye_frames(
    video_path: str,
    timestamps: list[float],
    output_dir: str,
    orig_height: int,
) -> list[str]:
    """
    Extract left-eye (1:1 square) frames from a stereo video at given timestamps
    Returns list of output file paths
    """
    output_paths = []
    eye_size = orig_height
    crop_filter = f"crop={eye_size}:{eye_size}:0:0"

    if is_nvidia():
        # 10-bit (e.g. AV1) needs GPU format conversion before hwdownload
        if is_10bit(video_path):
            dl_filter = f"scale_cuda=format=nv12,hwdownload,format=nv12,{crop_filter}"
        else:
            dl_filter = f"hwdownload,format=nv12,{crop_filter}"
    else:
        dl_filter = crop_filter

    for ts in timestamps:
        out_path = os.path.join(output_dir, f"frame_{ts:.0f}s.png")
        cmd = [
            'ffmpeg', '-y', '-hide_banner',
            *hwaccel_input_args(),
            '-ss', str(ts),
            '-i', video_path,
            '-vf', dl_filter,
            '-frames:v', '1', '-compression_level', '1',
            out_path
        ]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        process.wait()
        if process.returncode == 0 and os.path.exists(out_path):
            output_paths.append(out_path)

    return output_paths


def extract_segment_with_frames(
    stereo_video: str,
    start: float,
    end: float,
    fps: float,
    orig_height: int,
    target_height: int,
    left_frame_out: str,
    right_frame_out: str,
    left_video_out: str,
    right_video_out: str,
    center_crop: bool = False,
    crop_ratio: float = 0.8,
    progress_prefix: str = ""
) -> tuple[str, str, str, str]:

    # frame-accurate seeking
    start_frame = math.floor(start * fps)
    end_frame = math.floor(end * fps)
    frames = end_frame - start_frame
    if frames <= 0:
        raise RuntimeError(f"❌ Invalid segment: {start=} {end=} {fps=} -> {frames} frames")

    aligned_start = start_frame / fps
    keyframe_seek = max(0.0, aligned_start - 2.0)
    fine_seek = aligned_start - keyframe_seek
    seg_dur = frames / fps

    # geometry
    orig_eye = orig_height
    target_eye = target_height

    # full-res crops for .png frames
    if center_crop:
        crop_size = int(orig_eye * crop_ratio)
        if crop_size % 2 != 0:
            crop_size -= 1
        crop_offset = (orig_eye - crop_size) // 2
        frame_left_crop = f"crop={crop_size}:{crop_size}:{crop_offset}:{crop_offset}"
        frame_right_crop = f"crop={crop_size}:{crop_size}:{orig_eye + crop_offset}:{crop_offset}"
    else:
        frame_left_crop = f"crop={orig_eye}:{orig_eye}:0:0"
        frame_right_crop = f"crop={orig_eye}:{orig_eye}:{orig_eye}:0"

    # crops for video segments
    if center_crop:
        scaled_crop_size = int(target_eye * crop_ratio)
        if scaled_crop_size % 2 != 0:
            scaled_crop_size -= 1
        scaled_crop_offset = (target_eye - scaled_crop_size) // 2
        video_left_crop = f"crop={scaled_crop_size}:{scaled_crop_size}:{scaled_crop_offset}:{scaled_crop_offset}"
        video_right_crop = f"crop={scaled_crop_size}:{scaled_crop_size}:{target_eye + scaled_crop_offset}:{scaled_crop_offset}"
    else:
        video_left_crop = f"crop={target_eye}:{target_eye}:0:0"
        video_right_crop = f"crop={target_eye}:{target_eye}:{target_eye}:0"

    scale_w = target_height * 2
    scale_h = target_height

    if is_nvidia():
        # scale stereo -> target size on GPU -> crop L/R on CPU -> upload for NVENC
        # 10-bit needs GPU format conversion before hwdownload
        _10bit = is_10bit(stereo_video)
        full_dl = "scale_cuda=format=nv12,hwdownload,format=nv12" if _10bit else "hwdownload,format=nv12"
        scale_fmt = (
            f"scale_cuda={scale_w}:{scale_h}:interp_algo=bicubic:format=nv12"
            if _10bit else
            f"scale_cuda={scale_w}:{scale_h}:interp_algo=bicubic"
        )

        filter_complex = (
            f"[0:v]trim=start={fine_seek}:duration={seg_dur},setpts=PTS-STARTPTS,split=2[full][toscale];"

            # full resolution .png frames (cpu path)
            f"[full]{full_dl},split=2[fullL][fullR];"
            f"[fullL]select=eq(n\\,0),{frame_left_crop}[frame_left];"
            f"[fullR]select=eq(n\\,0),{frame_right_crop}[frame_right];"

            # segment videos (resize on gpu, then crops on cpu at target size)
            f"[toscale]{scale_fmt},"
            f"hwdownload,format=nv12[scaled_cpu];"
            f"[scaled_cpu]split=2[sL][sR];"
            f"[sL]{video_left_crop},hwupload_cuda[video_left];"
            f"[sR]{video_right_crop},hwupload_cuda[video_right]"
        )
    else:
        # All CPU filtering
        scale_fmt = f"scale={scale_w}:{scale_h}:flags=bicubic"
        filter_complex = (
            f"[0:v]trim=start={fine_seek}:duration={seg_dur},setpts=PTS-STARTPTS,split=2[full][toscale];"

            f"[full]split=2[fullL][fullR];"
            f"[fullL]select=eq(n\\,0),{frame_left_crop}[frame_left];"
            f"[fullR]select=eq(n\\,0),{frame_right_crop}[frame_right];"

            f"[toscale]{scale_fmt},split=2[sL][sR];"
            f"[sL]{video_left_crop}[video_left];"
            f"[sR]{video_right_crop}[video_right]"
        )

    cmd = [
        "ffmpeg", "-y",
        "-hide_banner",
    ]

    if is_nvidia():
        is_av1 = get_video_codec(stereo_video) == 'av1'
        # decode and scaling on gpu. limit threads and omit extra_hw_frames for AV1 to avoid NVDEC surface crash
        cmd.extend([
            "-threads", "1" if is_av1 else "2",
            "-hwaccel", "cuda",
            "-hwaccel_output_format", "cuda",
        ])
        if not is_av1:
            cmd.extend(["-extra_hw_frames", "16"])

    cmd.extend([
        # fast seek to near-aligned_start
        "-ss", str(keyframe_seek),
        "-i", stereo_video,

        "-filter_complex", filter_complex,

        # output 1/2: pngs
        "-map", "[frame_left]", "-frames:v", "1", "-compression_level", "1", left_frame_out,
        "-map", "[frame_right]", "-frames:v", "1", "-compression_level", "1", right_frame_out,

        # output 3/4: mp4 segments
        "-map", "[video_left]", "-frames:v", str(frames),
        "-c:v", hevc_encoder(), *encoder_args(cq='18', extras=False), left_video_out,
        "-map", "[video_right]", "-frames:v", str(frames),
        "-c:v", hevc_encoder(), *encoder_args(cq='18', extras=False), right_video_out,
    ])

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stderr_lines = []
    for line in process.stderr:
        stderr_lines.append(line)
        if "frame=" in line:
            print(f"\r{progress_prefix}{parse_ffmpeg_progress(line)}", end="", flush=True)

    process.wait()
    if process.returncode != 0:
        tail = "".join(stderr_lines[-60:])
        raise RuntimeError(f"❌ Segment extraction failed.\n\nFFmpeg tail:\n{tail}")

    return left_frame_out, right_frame_out, left_video_out, right_video_out


def stitch_stereo_videos(
    left_video: str,
    right_video: str,
    output_path: str,
    center_cropped: bool = False,
    full_size: int | None = None,
) -> str:
    """
    Combine two square videos side-by-side into a stereo video preserving input fps

    Args:
        left_video: left eye video path
        right_video: right eye video path
        output_path: output stereo video path
        center_cropped: if True, pad the center-cropped masks to full_size before stitching
        full_size: target size per eye (required if center_cropped=True)

    Returns:
        output_path
    """
    if center_cropped and full_size:
        # pad center crops back to full square size, then stack. calculates padding (equal on all sides)
        filter_complex = (
            f"[0:v]pad={full_size}:{full_size}:(ow-iw)/2:(oh-ih)/2:black[left];"
            f"[1:v]pad={full_size}:{full_size}:(ow-iw)/2:(oh-ih)/2:black[right];"
            f"[left][right]hstack=inputs=2[out]"
        )
    else:
        # standard stack without padding
        filter_complex = "[0:v][1:v]hstack=inputs=2[out]"

    cmd = [
        'ffmpeg', '-y',
        '-i', left_video,
        '-i', right_video,
        '-filter_complex', filter_complex,
        '-map', '[out]',
        '-c:v', hevc_encoder(), *encoder_args(cq='18', extras=False),
        output_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"❌ Stereo stitching failed: {result.stderr}")

    return output_path


def generate_solid_mask(
    color: str,
    width: int,
    height: int,
    start_time: float,
    end_time: float,
    fps: float,
    output_path: str,
) -> str:
    """Generate a solid color mask video (white for intro, black for transitions)"""
    start_frame = math.floor(start_time * fps)
    end_frame = math.floor(end_time * fps)
    frames = end_frame - start_frame

    cmd = [
        'ffmpeg', '-y',
        '-f', 'lavfi',
        '-i', f'color=c={color}:s={width}x{height}:r={fps}',
        '-vf', 'format=gray',
        '-frames:v', str(frames),
        '-c:v', hevc_encoder(), *encoder_args(cq='18', extras=False),
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"❌ {color} mask generation failed: {result.stderr}")
    return output_path


def apply_fade_overlay(
    input_path: str,
    output_path: str,
    fade_in: float = 0.0,
    fade_out: float = 0.0,
    fade_in_white: bool = False,
) -> str:
    """
    Apply fade overlay to a video segment

    fadeout: at the end of the video, multiply pixel values towards black
    fadein: at the start of the video, fade from black or white to full

    Args:
        fade_in_white: if True, fade in from white (after intro). if False, fade from black
    """

    if fade_in == 0 and fade_out == 0:
        shutil.copy(input_path, output_path)
        return output_path

    # get video duration
    probe_cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        input_path
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"❌ ffprobe failed: {result.stderr}")
    duration = float(result.stdout.strip())

    # build fade filter
    filters = []
    if fade_in > 0:
        color = "white" if fade_in_white else "black"
        filters.append(f"fade=t=in:st=0:d={fade_in}:c={color}")
    if fade_out > 0:
        fade_start = duration - fade_out
        filters.append(f"fade=t=out:st={fade_start}:d={fade_out}")

    filter_str = ",".join(filters) if filters else "null"

    cmd = [
        'ffmpeg', '-y',
        '-i', input_path,
        '-vf', filter_str,
        '-c:v', hevc_encoder(), *encoder_args(cq='18', extras=False),
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"❌ Fade overlay failed: {result.stderr}")

    return output_path
