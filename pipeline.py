"""
VR Video Masking Pipeline

Automated pipeline for VR video masking:
1. Intro and transition detection
2. Left/Right split/crop, video segmenting and first frames extraction
3. SAM3 first frames mask generation
4. MatAnyone mask generation
5. Mask concatenation and CFR normalization

Usage:
    python pipeline.py video.mp4
    python pipeline.py video.mp4 --mask-height 1280 --center-crop
"""

import argparse
import os
import sys
import shutil
from enum import Enum
from pathlib import Path
from typing import List
from dataclasses import dataclass
from utils.scene_detector import detect_fade_transitions, merge_transitions, Transition
from utils.time_utils import parse_timestamp, format_timestamp
from utils.ffmpeg_utils import (
    get_video_info, concatenate_videos, sync_mask_to_video,
    stitch_stereo_videos, generate_solid_mask, 
    apply_fade_overlay, extract_segment_with_frames
)
from utils.matanyone_runner import run_matanyone_inference
from utils.sam3_utils import run_sam3_batch, detect_intro_end


BLACK_GAP = 1.8         # total black gap duration centered on transition (half before + half after)
FADE_DURATION = 0.3     # duration for fadeout/fadein overlays
CENTER_CROP_RATIO = 0.8 # center crop keeps 80% of each eye (10% margin each side)


class SegmentType(Enum):
    INTRO = 'intro'
    MASK = 'mask'
    BLACK = 'black'


@dataclass
class SegmentInfo:
    """Information about a segment"""
    index: int
    start_time: float
    end_time: float
    seg_type: SegmentType
    left_frame_path: str = ""   # left eye first frame
    right_frame_path: str = ""  # right eye first frame
    left_mask_path: str = ""    # left eye SAM3 mask
    right_mask_path: str = ""   # right eye SAM3 mask
    video_path: str = ""        # final output (stereo)
    fade_out: float = 0.0       # duration of fadeout at end (0 = no fade)
    fade_in: float = 0.0        # duration of fadein at start (0 = no fade)
    fade_in_white: bool = False # True = fade from white (after intro), False = fade from black


def calculate_segments(
    intro_end: float,
    transitions: List[Transition],
    video_duration: float,
    max_segment_length: float = 5.0,
) -> List[SegmentInfo]:
    """
    Calculate segments based on fade transitions
    
    Segment types:
    - intro: 0 -> intro_end (white mask)
    - mask: MatAnyone masked segment (may have fade_out/fade_in overlays)
    - black: Solid black gap during scene transitions
    
    For fades: mask ends with fadeout overlay -> black gap -> next mask starts with fadein overlay
    """
    segments = []
    seg_idx = 0
    current_time = 0.0
    pending_fadein = False          # track if next mask needs fadein (from black)
    pending_fadein_white = False    # track if next mask needs fadein from white (after intro)
    
    # intro segment
    if intro_end > 0:
        segments.append(SegmentInfo(index=seg_idx, start_time=0.0, end_time=intro_end, seg_type=SegmentType.INTRO))
        seg_idx += 1
        current_time = intro_end
        pending_fadein_white = True # first mask after intro fades from white
    
    for trans in transitions:
        # fade: mask extends to BLACK_GAP/2 before transition, then black gap, then next mask
        # fadeout/fadein overlays are applied during stitching
        half_gap = BLACK_GAP / 2
        black_start = trans.timestamp - half_gap
        black_end = trans.timestamp + half_gap
        
        # segment before black gap, with fadeout at end
        if current_time < black_start:
            seg = SegmentInfo(
                index=seg_idx,
                start_time=current_time,
                end_time=black_start,
                seg_type=SegmentType.MASK,
                fade_in=FADE_DURATION if (pending_fadein or pending_fadein_white) else 0.0,
                fade_in_white=pending_fadein_white,
                fade_out=FADE_DURATION  # this mask ends with fadeout
            )
            segments.append(seg)
            seg_idx += 1
            pending_fadein = False
            pending_fadein_white = False
        
        # black gap segment
        segments.append(SegmentInfo(index=seg_idx, start_time=black_start, end_time=black_end, seg_type=SegmentType.BLACK))
        seg_idx += 1
        
        current_time = black_end
        pending_fadein = True   # next mask needs fadein (from black)
    
    # final segment
    if current_time < video_duration:
        segments.append(SegmentInfo(
            index=seg_idx,
            start_time=current_time,
            end_time=video_duration,
            seg_type=SegmentType.MASK,
            fade_in=FADE_DURATION if (pending_fadein or pending_fadein_white) else 0.0,
            fade_in_white=pending_fadein_white
        ))
    
    # split long mask segments into uniform chunks of max_segment_length
    split_segments = []
    new_idx = 0
    for seg in segments:
        if seg.seg_type == SegmentType.MASK:
            seg_duration = seg.end_time - seg.start_time
            
            if seg_duration > max_segment_length:
                chunk_start = seg.start_time
                chunk_num = 0
                while chunk_start < seg.end_time:
                    chunk_end = min(chunk_start + max_segment_length, seg.end_time)
                    
                    # if remaining duration after this chunk would be < 1s, extend this chunk to the end
                    remaining = seg.end_time - chunk_end
                    if 0 < remaining < 1.0:
                        chunk_end = seg.end_time
                    
                    is_first = (chunk_num == 0)
                    is_last = (chunk_end >= seg.end_time - 0.01)
                    split_segments.append(SegmentInfo(
                        index=new_idx,
                        start_time=chunk_start,
                        end_time=chunk_end,
                        seg_type=SegmentType.MASK,
                        fade_in=seg.fade_in if is_first else 0.0,
                        fade_in_white=seg.fade_in_white if is_first else False,
                        fade_out=seg.fade_out if is_last else 0.0
                    ))
                    new_idx += 1
                    chunk_start = chunk_end
                    chunk_num += 1
            else:
                seg.index = new_idx
                split_segments.append(seg)
                new_idx += 1
        else:
            # intro/black segments: keep as is
            seg.index = new_idx
            split_segments.append(seg)
            new_idx += 1
    
    return split_segments


# =============================================================================
# Pipeline step functions
# =============================================================================

def step_detect_intro_and_transitions(
    args: argparse.Namespace,
    orig_h: int,
    duration: float,
) -> tuple[float, List[Transition]]:
    print("STEP 1: INTRO AND TRANSITION DETECTION")

    if args.intro_end is not None:
        intro_end = parse_timestamp(args.intro_end)
        print(f"  Intro end: {format_timestamp(intro_end)}")
    else:
        print("Detecting intro end...")
        intro_end = detect_intro_end(args.video, orig_h, duration, prompt=args.prompt)
        print(f"  Intro end: {format_timestamp(intro_end)}")

    print()
    print("Detecting fade transitions...")
    fades = detect_fade_transitions(args.video)
    transitions = merge_transitions(fades, duration, intro_end)

    print(f"  Found {len(transitions)} fade transitions")
    for t in transitions:
        print(f"  FADE @ {format_timestamp(t.timestamp)}")
    print()

    return intro_end, transitions


def step_extract_segments(
    args: argparse.Namespace,
    segments: List[SegmentInfo],
    mask_segments: List[SegmentInfo],
    fps: float,
    orig_h: int,
    frames_dir: Path,
    segments_dir: Path,
) -> None:
    print("STEP 2: SEGMENT AND FIRST FRAMES EXTRACTION")

    print(f"Total: {len(segments)} segments")
    for seg in segments:
        dur = seg.end_time - seg.start_time
        print(f"  [{seg.index}] {seg.seg_type.value.upper():5} {format_timestamp(seg.start_time)} → {format_timestamp(seg.end_time)} ({dur:.1f}s)")
    print()

    print("Extracting L/R segments and first frames...")
    for i, seg in enumerate(mask_segments):
        # output paths
        left_frame = str(frames_dir / f"seg{seg.index:02d}_left.png")
        right_frame = str(frames_dir / f"seg{seg.index:02d}_right.png")
        seg_left_video = str(segments_dir / f"seg{seg.index:02d}_left.mp4")
        seg_right_video = str(segments_dir / f"seg{seg.index:02d}_right.mp4")

        # single-pass: extracts full-res frames + scaled videos
        extract_segment_with_frames(
            stereo_video=args.video,
            start=seg.start_time,
            end=seg.end_time,
            fps=fps,
            orig_height=orig_h,
            target_height=args.mask_height,
            left_frame_out=left_frame,
            right_frame_out=right_frame,
            left_video_out=seg_left_video,
            right_video_out=seg_right_video,
            center_crop=args.center_crop,
            crop_ratio=CENTER_CROP_RATIO,
            progress_prefix=f"[{i+1}/{len(mask_segments)}] "
        )

        seg.left_frame_path = left_frame
        seg.right_frame_path = right_frame
    print()
    print()


def step_generate_sam3_masks(
    mask_segments: List[SegmentInfo],
    frames_dir: Path,
    masks_dir: Path,
    mask_square_size: int,
    prompt: str | None,
) -> None:
    print("STEP 3: SAM3 MASK GENERATION")
    print("Running SAM3 on first segment frames...")
    run_sam3_batch(str(frames_dir), output_size=mask_square_size, prompt=prompt)

    for seg in mask_segments:
        # left eye mask
        if seg.left_frame_path:
            base = os.path.splitext(os.path.basename(seg.left_frame_path))[0]
            mask_src = frames_dir / f"{base}_mask.png"
            if mask_src.exists():
                mask_name = f"seg{seg.index:02d}_left_mask.png"
                final_mask_path = str(masks_dir / mask_name)
                shutil.move(str(mask_src), final_mask_path)
                seg.left_mask_path = final_mask_path

        # right eye mask
        if seg.right_frame_path:
            base = os.path.splitext(os.path.basename(seg.right_frame_path))[0]
            mask_src = frames_dir / f"{base}_mask.png"
            if mask_src.exists():
                mask_name = f"seg{seg.index:02d}_right_mask.png"
                final_mask_path = str(masks_dir / mask_name)
                shutil.move(str(mask_src), final_mask_path)
                seg.right_mask_path = final_mask_path

    print()


def step_run_matanyone(
    segments: List[SegmentInfo],
    segments_dir: Path,
    mask_square_size: int,
    full_square_size: int,
    stereo_w: int,
    stereo_h: int,
    args: argparse.Namespace,
    fps: float,
) -> None:
    print("STEP 4: MATANYONE MASK GENERATION")
    matanyone_output_dir = str(segments_dir / "matanyone_output")
    os.makedirs(matanyone_output_dir, exist_ok=True)

    mask_segments = [s for s in segments if s.seg_type == SegmentType.MASK]
    total_ops = len(mask_segments) * 2

    # first pass: generate intro/black solid masks, and build MASK jobs
    jobs = []
    for seg in segments:
        seg_duration = seg.end_time - seg.start_time

        if seg.seg_type == SegmentType.INTRO:
            output = str(segments_dir / f"seg{seg.index:02d}_intro.mp4")
            generate_solid_mask('white', stereo_w, stereo_h, seg.start_time, seg.end_time, fps, output)
            seg.video_path = output

        elif seg.seg_type == SegmentType.BLACK:
            output = str(segments_dir / f"seg{seg.index:02d}_black.mp4")
            generate_solid_mask('black', stereo_w, stereo_h, seg.start_time, seg.end_time, fps, output)
            seg.video_path = output

        elif seg.seg_type == SegmentType.MASK:
            if not seg.left_mask_path or not seg.right_mask_path:
                raise RuntimeError(f"Segment [{seg.index}] missing SAM3 masks")

            seg_left_video = str(segments_dir / f"seg{seg.index:02d}_left.mp4")
            seg_right_video = str(segments_dir / f"seg{seg.index:02d}_right.mp4")

            jobs.append({
                'input_path': seg_left_video,
                'mask_path': seg.left_mask_path,
                'output_path': matanyone_output_dir,
                'max_size': mask_square_size,
                'warmup': args.warmup,
                'erode': args.erode,
                'dilate': args.dilate,
                'fps': fps,
                'op_num': len(jobs) + 1,
                'total_ops': total_ops,
                'label': f"seg{seg.index:02d}_left",
                'duration': seg_duration
            })

            jobs.append({
                'input_path': seg_right_video,
                'mask_path': seg.right_mask_path,
                'output_path': matanyone_output_dir,
                'max_size': mask_square_size,
                'warmup': args.warmup,
                'erode': args.erode,
                'dilate': args.dilate,
                'fps': fps,
                'op_num': len(jobs) + 1,
                'total_ops': total_ops,
                'label': f"seg{seg.index:02d}_right",
                'duration': seg_duration
            })

    # run all MatAnyone jobs in a single batched subprocess
    if jobs:
        completed_paths = run_matanyone_inference(jobs)
        if len(completed_paths) != len(jobs):
            raise RuntimeError(f"Not all jobs completed successfully. Expected {len(jobs)}, got {len(completed_paths)}")

    # stitch the left and right masks for each MASK segment
    for seg in mask_segments:
        left_basename = os.path.splitext(os.path.basename(f"seg{seg.index:02d}_left.mp4"))[0]
        right_basename = os.path.splitext(os.path.basename(f"seg{seg.index:02d}_right.mp4"))[0]
        
        left_pha = os.path.join(matanyone_output_dir, f"{left_basename}_pha.mp4")
        right_pha = os.path.join(matanyone_output_dir, f"{right_basename}_pha.mp4")
        
        if not os.path.exists(left_pha) or not os.path.exists(right_pha):
             raise RuntimeError(f"Could not find generated masks for segment {seg.index}")

        # stitch left and right masks together, with padding if center-cropped
        stereo_output = str(segments_dir / f"seg{seg.index:02d}_stereo.mp4")
        stitch_stereo_videos(left_pha, right_pha, stereo_output, center_cropped=args.center_crop, full_size=full_square_size)
        seg.video_path = stereo_output
        
    print()


def step_concatenate_and_finalize(
    segments: List[SegmentInfo],
    video_name: str,
    video_path: str,
    fps: float,
) -> str:
    print("STEP 5: FULL LENGTH MASK CONCATENATION")

    faded_segments = [s for s in segments if s.seg_type == SegmentType.MASK and (s.fade_in > 0 or s.fade_out > 0)]
    if faded_segments:
        print(f"Applying fade overlays to {len(faded_segments)} segments...")
        for seg in faded_segments:
            if not seg.video_path or not os.path.exists(seg.video_path):
                raise RuntimeError(f"Faded segment [{seg.index}] missing: {seg.video_path}")
            # apply fade to the MatAnyone output
            faded_path = seg.video_path.replace('.mp4', '_faded.mp4')
            apply_fade_overlay(seg.video_path, faded_path, seg.fade_in, seg.fade_out, seg.fade_in_white)
            seg.video_path = faded_path
        print()

    print("Concatenating mask segments...")
    segment_videos = []
    for seg in sorted(segments, key=lambda s: s.index):
        if seg.video_path and os.path.exists(seg.video_path):
            segment_videos.append(seg.video_path)
        else:
            raise RuntimeError(f"Segment [{seg.index}] missing")

    if not segment_videos:
        raise RuntimeError("No segments to concatenate")

    output_dir = os.path.dirname(video_path) or '.'
    concat_raw = os.path.join(output_dir, f"{video_name}_mask_raw.mp4")
    output_mask = os.path.join(output_dir, f"{video_name}_mask.mp4")

    concatenate_videos(segment_videos, concat_raw)
    print()

    print("Normalizing mask frame rate (CFR)...")
    synced_path = sync_mask_to_video(concat_raw, fps=fps, frame_offset=0)   # no frame shifting, just re-encode to CFR
    os.replace(synced_path, output_mask)
    os.remove(concat_raw)
    print()

    return output_mask



def main() -> int:
    parser = argparse.ArgumentParser(
        description='VR Video Masking Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('video', help='Input video path')
    parser.add_argument(
        "--mask-height",
        type=int,
        default=1280,
        help="Height for mask processing (default: 1280)",
    )
    parser.add_argument(
        "--segment-length",
        type=float,
        default=10,
        help="Masked segments length (default: 10s)",
    )
    parser.add_argument(
        "--center-crop",
        action="store_true",
        default=False,
        help="10%% crop on all sides (+70%% masking speed, -55%% VRAM usage)",
    )
    parser.add_argument(
        "--intro-end",
        default=None,
        help="Intro end in seconds (default: auto-detect)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=15,
        help="MatAnyone warmup frames (default: 15, use 8-20)",
    )
    parser.add_argument(
        "--erode",
        type=int,
        default=3,
        help="Mask erosion kernel (default: 3, use 2-8)",
    )
    parser.add_argument(
        "--dilate",
        type=int,
        default=6,
        help="Mask dilation kernel (default: 6, use 4-10)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="woman",
        help="SAM3 prompt for initial masks (default: woman) (⚠️ only change it if using base weights)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"❌ Video not found: {args.video}")
        return 1

    # validate mask height range and alignment
    if args.mask_height < 640 or args.mask_height > 1600:
        print(f"❌ Mask height must be between 640 and 1600 (got {args.mask_height})")
        return 1

    # validate mask height is divisible by 16
    if args.mask_height % 16 != 0:
        print(f"❌ Mask height must be divisible by 16 (try --mask-height 640, 800, 960, 1120, 1280, 1440, 1600)")
        return 1

    # validate center-cropped size is also divisible by 16
    if args.center_crop:
        if int(args.mask_height * CENTER_CROP_RATIO) % 16 != 0:
            print(f"❌ Center-cropped height must be divisible by 16 (try --mask-height 640, 800, 960, 1120, 1280, 1440, 1600)")
            return 1

    video_name = os.path.splitext(os.path.basename(args.video))[0]

    # temp directory
    temp_dir = Path('temp_pipeline')
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    frames_dir = temp_dir / "frames"
    masks_dir = temp_dir / "masks"
    segments_dir = temp_dir / "segments"
    for d in [frames_dir, masks_dir, segments_dir]:
        d.mkdir(parents=True, exist_ok=True)

    orig_w, orig_h, fps, duration = get_video_info(args.video)

    print("="*60)
    print("VR Video Masking Pipeline")
    print("="*60)
    print(f"File: {args.video}")
    print(f"Specs: {orig_w}x{orig_h}, {fps:.2f}fps, {format_timestamp(duration)}")
    print(f"Mask height: {args.mask_height}px")
    print(f"Fade duration: {BLACK_GAP + FADE_DURATION*2}s")
    print()

    # determine masking dimensions
    full_square_size = args.mask_height  # full size per eye
    if args.center_crop:
        mask_square_size = int(full_square_size * CENTER_CROP_RATIO)
        if mask_square_size % 2 != 0:
            mask_square_size -= 1
    else:
        mask_square_size = full_square_size

    # stereo dimensions for intro/black masks
    stereo_w = full_square_size * 2
    stereo_h = full_square_size

    try:
        # Step 1: Intro and transition detection
        intro_end, transitions = step_detect_intro_and_transitions(args, orig_h, duration)

        # Step 2: Left/Right split/crop, video segmenting and first frames extraction
        segments = calculate_segments(intro_end, transitions, duration, args.segment_length)
        mask_segments = [s for s in segments if s.seg_type == SegmentType.MASK]
        step_extract_segments(args, segments, mask_segments, fps, orig_h, frames_dir, segments_dir)

        # Step 3: SAM3 first frames mask generation
        step_generate_sam3_masks(mask_segments, frames_dir, masks_dir, mask_square_size, args.prompt)

        # Step 4: MatAnyone mask generation
        step_run_matanyone(segments, segments_dir, mask_square_size, full_square_size, stereo_w, stereo_h, args, fps)

        # Step 5: Mask concatenation and CFR normalization
        output_mask = step_concatenate_and_finalize(segments, video_name, args.video, fps)

    except RuntimeError as e:
        print(f"❌ {e}")
        return 1

    # Done
    print("="*60)
    print("Pipeline completed")
    print("="*60)
    faded_count = len([s for s in segments if s.fade_in > 0 or s.fade_out > 0])
    print(f"✂️ Segments: {len(segments)} ({len(mask_segments)} masks, {faded_count} with fades)")
    print(f"🎬 Output: {output_mask}")
    print()
    print(f"Alpha pack with: python alpha_packer.py {output_mask} {args.video}")

    # save info
    with open(temp_dir / "segments.txt", 'w') as f:
        f.write(f"# {video_name}\n")
        for seg in segments:
            f.write(f"{seg.index},{seg.seg_type.value},{seg.start_time:.3f},{seg.end_time:.3f},{seg.fade_in},{seg.fade_out},{seg.video_path}\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())