"""
Accurate Video Cutter
Cuts and joins video segments with frame-accurate timing

Usage:
    python accurate_cut.py video.mp4 cuts.txt

cuts.txt format (comma-separated start, end timestamps):
    00:00:00, 00:00:35
    00:10:00, 00:11:10
"""

import math
import argparse
import subprocess
import os
import sys
from pathlib import Path
import tempfile
import shutil
from utils.time_utils import parse_timestamp, format_timestamp
from utils.ffmpeg_utils import get_video_info, get_output_encoder


def parse_cuts_file(cuts_path: str) -> list[tuple[float, float]]:
    """Parse cuts .txt file into list of (start, end) tuples in seconds"""
    cuts = []
    with open(cuts_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split(',')
            if len(parts) >= 2:
                start = parse_timestamp(parts[0])
                end = parse_timestamp(parts[1])
                cuts.append((start, end))
    return cuts


def cut_segment_accurate(
    input_path: str,
    start: float,
    end: float,
    output_path: str,
    fps: float,
    encoder: str = 'hevc_nvenc',
) -> None:
    """
    Cut a segment with frame-accurate timing and exact fps preservation (uses frame count instead of duration)
    """
    # convert timestamps to frame indices (consistent rule: floor boundaries)
    start_frame = math.floor(start * fps)
    end_frame   = math.floor(end * fps)
    frame_count = max(0, end_frame - start_frame)

    if frame_count <= 0:
        raise ValueError(f"❌ Non-positive frame_count for cut {start}–{end} at {fps}fps")

    # frame-aligned timestamp for the first frame
    aligned_start = start_frame / fps

    # two-step seek: coarse to a keyframe, then fine seek
    keyframe_seek = max(0.0, aligned_start - 2.0)
    fine_seek = aligned_start - keyframe_seek

    cmd = [
        'ffmpeg', '-y',
        '-hwaccel', 'cuda',
        '-hwaccel_output_format', 'cuda',

        # coarse seek
        '-ss', str(keyframe_seek),
        '-i', input_path,

        # fine seek inside the decoded stream
        '-ss', str(fine_seek),

        # cut by exact frame count
        '-frames:v', str(frame_count),

        # keep source fps – no extra -r frame-rate conversion
        '-c:v', encoder,
        '-preset', 'p4',
        '-cq', '24',
        '-temporal-aq', '1',
        '-spatial-aq', '1',
        '-rc-lookahead', '32',

        '-c:a', 'aac',
        '-b:a', '192k',

        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"❌ FFmpeg cut failed: {result.stderr[-500:]}")


def concatenate_segments(segment_paths: list[str], output_path: str) -> None:
    """Concatenate segments using concat demuxer (no re-encode)"""
    concat_file = output_path + ".concat.txt"
    with open(concat_file, 'w') as f:
        for path in segment_paths:
            abs_path = os.path.abspath(path).replace('\\', '/')
            f.write(f"file '{abs_path}'\n")
    
    cmd = [
        'ffmpeg', '-y',
        '-f', 'concat',
        '-safe', '0',
        '-i', concat_file,
        '-c', 'copy',
        output_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    os.remove(concat_file)
    
    if result.returncode != 0:
        raise RuntimeError(f"❌ FFmpeg concat failed: {result.stderr[-500:]}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Accurately cut and join video segments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
cuts.txt format:
  00:00:00, 00:00:35
  00:10:00, 00:11:10
        """
    )
    parser.add_argument('video', help='Input video path')
    parser.add_argument('cuts', nargs='?', default='cuts.txt', help='Cuts file (default: cuts.txt)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"❌ Video not found: {args.video}")
        return 1
    
    if not os.path.exists(args.cuts):
        if args.cuts == 'cuts.txt':
            print("❌ cuts.txt was not found")
        else:
            print(f"❌ Cuts file not found: {args.cuts}")
        return 1
    
    # parse cuts
    cuts = parse_cuts_file(args.cuts)
    if not cuts:
        print("❌ No valid cuts found in file")
        return 1
    
    base = os.path.splitext(args.video)[0]
    output_path = f"{base}_cut.mp4"
    
    _, _, fps, _ = get_video_info(args.video)
    encoder = get_output_encoder(args.video)
    
    print("="*60)
    print("FFMPEG Video Cutter")
    print("="*60)
    print(f"Input: {args.video}")
    print(f"Output: {output_path}")
    print(f"FPS: {fps:.6f}")
    print(f"Encoder: {encoder}")
    print(f"Segments:")
    
    total_duration = 0
    total_frames = 0
    for i, (start, end) in enumerate(cuts):
        dur = end - start
        frames = math.floor(end * fps) - math.floor(start * fps)
        total_duration += dur
        total_frames += frames
        print(f"  [{i+1}] {format_timestamp(start)} → {format_timestamp(end)} ({dur:.1f}s, {frames} frames)")
    
    print(f"\nTotal: {format_timestamp(total_duration)} ({total_frames} frames)")
    print()
    
    # create temp directory for segments
    temp_dir = Path(tempfile.mkdtemp(prefix="accurate_cut_"))
    segment_paths = []
    
    try:
        # cut each segment
        print("Cutting segments...")
        for i, (start, end) in enumerate(cuts):
            seg_path = str(temp_dir / f"seg{i:03d}.mp4")
            print(f"  [{i+1}/{len(cuts)}] Cutting {format_timestamp(start)} → {format_timestamp(end)}")
            cut_segment_accurate(args.video, start, end, seg_path, fps, encoder=encoder)
            segment_paths.append(seg_path)
        
        print()
        
        # concatenate
        print("Concatenating segments...")
        concatenate_segments(segment_paths, output_path)
        
        # verify
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', output_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print()
        print("✂️ Cutting completed")
        print(f"🎬 Output: {output_path}")
        
    finally:
        shutil.rmtree(temp_dir)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())