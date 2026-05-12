"""
Fade Transition Detection
Detects fade-to-black transitions using blackdetect
"""

import re
from typing import List, Tuple
from dataclasses import dataclass
from utils.ffmpeg_utils import run_ffmpeg_with_progress, is_10bit
from utils.gpu_backend import is_nvidia, hwaccel_input_args

# detection defaults
BLACK_THRESHOLD = 0.10      # pixel darkness threshold (0-1)
BLACKPX_PCT = 0.75          # percent of frame that must be dark
MIN_BLACK_DURATION = 0.1    # minimum black segment duration (seconds)
MIN_GAP = 2.0               # minimum seconds between transitions
IGNORE_LAST_SECONDS = 10    # ignore fades in last N seconds (outro)


@dataclass
class Transition:
    """A detected fade transition with timing info"""
    timestamp: float        # center of black segment
    type: str               # always 'fade'
    fade_start: float = 0   # when black segment starts
    fade_end: float = 0     # when black segment ends


def detect_fade_transitions(
    video_path: str,
    black_threshold: float = BLACK_THRESHOLD,
    blackpx_pct: float = BLACKPX_PCT,
    min_black_duration: float = MIN_BLACK_DURATION,
) -> List[Tuple[float, float, float]]:
    """
    Detect fade-to-black transitions using blackdetect filter
    
    Returns: List of (start, end, center) tuples for each black segment
    """
    if is_nvidia():
        # 10-bit (e.g. AV1) needs explicit format conversion on GPU before hwdownload
        scale_fmt = (
            "scale_cuda=64:32:interp_algo=nearest:format=nv12"
            if is_10bit(video_path) else
            "scale_cuda=64:32:interp_algo=nearest"
        )
        filter_graph = (
            f"{scale_fmt},hwdownload,format=nv12,"
            f"blackdetect=d={min_black_duration}:pic_th={blackpx_pct}:pix_th={black_threshold}"
        )
    else:
        filter_graph = (
            f"scale=64:32:flags=neighbor,"
            f"blackdetect=d={min_black_duration}:pic_th={blackpx_pct}:pix_th={black_threshold}"
        )

    cmd = [
        'ffmpeg',
        *hwaccel_input_args(),
        '-skip_frame', 'noref',
        '-i', video_path,
        '-vf', filter_graph,
        '-an',
        '-f', 'null', '-'
    ]
    
    _, stderr_text = run_ffmpeg_with_progress(cmd)
    
    # parse black frames from blackdetect
    fade_pattern = r'black_start:(\d+\.?\d*)\s+black_end:(\d+\.?\d*)'
    fades = []
    for match in re.finditer(fade_pattern, stderr_text):
        start = float(match.group(1))
        end = float(match.group(2))
        center = (start + end) / 2
        fades.append((start, end, center))
    
    return fades


def merge_transitions(
    fades: List[Tuple[float, float, float]],
    video_duration: float,
    intro_skip: float,
    min_gap: float = MIN_GAP,
) -> List[Transition]:
    """
    Filter and merge fade detections
    - Ignores fades before intro_skip
    - Ignores fades in last IGNORE_LAST_SECONDS seconds (outro)
    - Clusters nearby transitions
    """
    transitions = []
    
    # add fades that are within valid range
    for start, end, center in fades:
        if center > intro_skip and center < (video_duration - IGNORE_LAST_SECONDS):
            transitions.append(Transition(timestamp=center, type='fade', fade_start=start, fade_end=end))
    
    # sort by timestamp
    transitions.sort(key=lambda t: t.timestamp)
    
    # cluster nearby transitions (keeps first)
    if not transitions:
        return []
    
    result = [transitions[0]]
    for t in transitions[1:]:
        if t.timestamp - result[-1].timestamp > min_gap:
            result.append(t)
    
    return result