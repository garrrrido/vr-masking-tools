"""
DeoVR Alpha Packer
Packs alpha mask into FISHEYE190 VR videos

Usage: python alpha_packer.py mask.mp4 video.mp4

Layout:

- Left eye: Split into 2 semicircles, placed at top-center and bottom-center
- Right eye: Split into 4 quadrants, placed in the 4 corners
"""

import subprocess
import argparse
import os
from utils.ffmpeg_utils import (
    get_video_info, run_ffmpeg_with_progress, sync_mask_to_video,
    get_output_encoder, SPEED_PRESETS, QUALITY_PRESETS,
)
from utils.gpu_backend import is_nvidia, encoder_args, default_preset


def _ceil_to(n: int, base: int) -> int:
    return ((n + base - 1) // base) * base


def get_circle_mask(size: int) -> str:
    import tempfile
    from pathlib import Path

    tmp_dir = Path(tempfile.gettempdir())
    mask_path = tmp_dir / f"circle_mask_{size}.png"

    try:
        from PIL import Image, ImageDraw, ImageFilter

        scale = 4
        size_hr = size * scale
        circle_img = Image.new("L", (size_hr, size_hr), 0)

        draw = ImageDraw.Draw(circle_img)
        draw.ellipse([0, 0, size_hr - 1, size_hr - 1], fill=255)

        circle_img = circle_img.resize((size, size), Image.LANCZOS)
        circle_img = circle_img.filter(ImageFilter.GaussianBlur(radius=1))
        circle_img.save(str(mask_path))

    except ImportError:
        cmd = [
            "ffmpeg", "-y", "-f", "lavfi", "-i",
            f"color=c=white:s={size}x{size}:d=1,format=gray",
            "-vf",
            "geq=lum='if(lte(pow(X-W/2,2)+pow(Y-H/2,2),pow(min(W,H)/2,2)),255,0)'",
            "-frames:v", "1",
            str(mask_path),
        ]
        subprocess.run(cmd, capture_output=True, text=True)

    return str(mask_path)


def _build_filter_complex_nvidia(out_w: int, out_h: int, overlay_size: int, half_overlay: int,
                                  sigma: float, erosion_filter: str, contrast: float, gamma: float) -> str:
    """NVENC/CUDA path: GPU decode, CUDA filters, overlay_cuda."""
    parts: list[str] = [
        # force main to aligned output size on gpu
        f"[0:v]scale_cuda=w={out_w}:h={out_h}:format=yuv420p[vid_gpu]",

        # split stereo mask into left/right eyes
        "[1:v]split=2[mask1][mask2]",

        # circle mask, split for left and right eye processing
        "[2:v]format=gray,split=2[circle_l][circle_r]",

        # left eye
        (
            f"[mask1]crop=ih:ih:0:0,"
            f"scale={overlay_size}:{overlay_size}:flags=bicubic,"
            f"{erosion_filter}"
            f"gblur=sigma={sigma},eq=contrast={contrast}:gamma={gamma},"
            "format=gbrp[left_scaled]"
        ),
        "[left_scaled][circle_l]alphamerge,format=rgba[left_circle]",

        # right eye
        (
            f"[mask2]crop=ih:ih:iw-ih:0,"
            f"scale={overlay_size}:{overlay_size}:flags=bicubic,"
            f"{erosion_filter}"
            f"gblur=sigma={sigma},eq=contrast={contrast}:gamma={gamma},"
            "format=gbrp[right_scaled]"
        ),
        "[right_scaled][circle_r]alphamerge,format=rgba[right_circle]",

        # split left circle into top/bottom semicircles
        "[left_circle]split=2[left_for_top][left_for_bottom]",
        f"[left_for_top]crop={overlay_size}:{half_overlay}:0:0,format=yuva420p[left_top_sw]",
        f"[left_for_bottom]crop={overlay_size}:{half_overlay}:0:{half_overlay},format=yuva420p[left_bottom_sw]",

        # split right circle into 4 quadrants
        "[right_circle]split=4[r1][r2][r3][r4]",
        f"[r1]crop={half_overlay}:{half_overlay}:0:0,format=yuva420p[right_tl_sw]",
        f"[r2]crop={half_overlay}:{half_overlay}:{half_overlay}:0,format=yuva420p[right_tr_sw]",
        f"[r3]crop={half_overlay}:{half_overlay}:0:{half_overlay},format=yuva420p[right_bl_sw]",
        f"[r4]crop={half_overlay}:{half_overlay}:{half_overlay}:{half_overlay},format=yuva420p[right_br_sw]",

        # upload tiles
        "[left_top_sw]hwupload=extra_hw_frames=16[left_top_gpu]",
        "[left_bottom_sw]hwupload=extra_hw_frames=16[left_bottom_gpu]",
        "[right_tl_sw]hwupload=extra_hw_frames=16[right_tl_gpu]",
        "[right_tr_sw]hwupload=extra_hw_frames=16[right_tr_gpu]",
        "[right_bl_sw]hwupload=extra_hw_frames=16[right_bl_gpu]",
        "[right_br_sw]hwupload=extra_hw_frames=16[right_br_gpu]",

        # overlays
        "[vid_gpu][left_top_gpu]overlay_cuda=x=(main_w-overlay_w)/2:y=main_h-overlay_h[v1]",
        "[v1][left_bottom_gpu]overlay_cuda=x=(main_w-overlay_w)/2:y=0[v2]",
        "[v2][right_tl_gpu]overlay_cuda=x=main_w-overlay_w:y=main_h-overlay_h[v3]",
        "[v3][right_tr_gpu]overlay_cuda=x=0:y=main_h-overlay_h[v4]",
        "[v4][right_bl_gpu]overlay_cuda=x=main_w-overlay_w:y=0[v5]",
        "[v5][right_br_gpu]overlay_cuda=x=0:y=0[out]",
    ]
    return ";".join(parts)


def _build_filter_complex_cpu(out_w: int, out_h: int, overlay_size: int, half_overlay: int,
                               sigma: float, erosion_filter: str, contrast: float, gamma: float) -> str:
    """CPU path (used with AMF encoder): software decode, CPU filters, alpha-aware CPU overlay."""
    parts: list[str] = [
        # force main to aligned output size. CPU overlay is used because these
        # packed tiles rely on their alpha channel to avoid covering fisheye pixels.
        f"[0:v]scale={out_w}:{out_h}:flags=bicubic,format=yuv420p[vid]",

        # split stereo mask into left/right eyes
        "[1:v]split=2[mask1][mask2]",

        # circle mask split for left/right eye (single frame; the -loop input is bounded by -t)
        "[2:v]format=gray,split=2[circle_l][circle_r]",

        # left eye
        (
            f"[mask1]crop=ih:ih:0:0,"
            f"scale={overlay_size}:{overlay_size}:flags=bicubic,"
            f"{erosion_filter}"
            f"gblur=sigma={sigma},eq=contrast={contrast}:gamma={gamma},"
            "format=gbrp[left_scaled]"
        ),
        "[left_scaled][circle_l]alphamerge,format=rgba[left_circle]",

        # right eye
        (
            f"[mask2]crop=ih:ih:iw-ih:0,"
            f"scale={overlay_size}:{overlay_size}:flags=bicubic,"
            f"{erosion_filter}"
            f"gblur=sigma={sigma},eq=contrast={contrast}:gamma={gamma},"
            "format=gbrp[right_scaled]"
        ),
        "[right_scaled][circle_r]alphamerge,format=rgba[right_circle]",

        # split left circle into top/bottom semicircles (CPU crop)
        "[left_circle]split=2[left_for_top][left_for_bottom]",
        f"[left_for_top]crop={overlay_size}:{half_overlay}:0:0[left_top]",
        f"[left_for_bottom]crop={overlay_size}:{half_overlay}:0:{half_overlay}[left_bottom]",

        # split right circle into 4 quadrants (CPU crop)
        "[right_circle]split=4[r1][r2][r3][r4]",
        f"[r1]crop={half_overlay}:{half_overlay}:0:0[right_tl]",
        f"[r2]crop={half_overlay}:{half_overlay}:{half_overlay}:0[right_tr]",
        f"[r3]crop={half_overlay}:{half_overlay}:0:{half_overlay}[right_bl]",
        f"[r4]crop={half_overlay}:{half_overlay}:{half_overlay}:{half_overlay}[right_br]",

        # Alpha-aware CPU overlays. The swapped destinations are intentional:
        # alpha fragments are packed into the black area opposite their source edge.
        f"[vid][left_top]overlay=x={(out_w - overlay_size) // 2}:y={out_h - half_overlay}:format=auto[v1]",
        f"[v1][left_bottom]overlay=x={(out_w - overlay_size) // 2}:y=0:format=auto[v2]",
        f"[v2][right_tl]overlay=x={out_w - half_overlay}:y={out_h - half_overlay}:format=auto[v3]",
        f"[v3][right_tr]overlay=x=0:y={out_h - half_overlay}:format=auto[v4]",
        f"[v4][right_bl]overlay=x={out_w - half_overlay}:y=0:format=auto[v5]",
        "[v5][right_br]overlay=x=0:y=0:format=auto,format=yuv420p[out]",
    ]
    return ";".join(parts)


def create_alpha_pack_command(
    video_path: str,
    mask_path: str,
    output_path: str,
    video_dims: tuple[int, int],
    preset: str | None = None,
    cq: str = '24',
    encoder: str | None = None,
    duration: float | None = None,
    fps: float | None = None,
) -> list[str]:
    """
    Generate FFmpeg command to pack alpha mask into fisheye video.

    NVIDIA backend: GPU decode (NVDEC) and overlay_cuda for all overlays. Mask
    pre-processing stays on CPU then is uploaded to CUDA frames.

    AMD backend: CPU decode + CPU overlays (overlay_opencl does not blend the
    RGBA alpha edges correctly here and can turn transparent regions into
    visible green/gray blocks), AMF encoder for output.
    """
    if encoder is None:
        encoder = get_output_encoder(video_path)
    if preset is None:
        preset = default_preset()

    video_w, video_h = video_dims

    # align output canvas to avoid encoder padded coded size artifacts
    out_h = _ceil_to(video_h, 32)

    # keep 2:1 if input is 2:1, otherwise fall back to aligning width too
    if video_w == 2 * video_h:
        out_w = 2 * out_h
    else:
        out_w = _ceil_to(video_w, 32)

    if (out_w, out_h) != (video_w, video_h):
        print(f"Encoder-aligned output: {out_w}x{out_h}")

    # calculate overlay dimensions (40% of height for fisheye corners)
    overlay_size = int(out_h * 0.4)
    overlay_size = (overlay_size // 4) * 4
    half_overlay = overlay_size // 2

    # resolution-dependent parameters (smaller videos use softer processing)
    if video_h <= 2400:
        erosion_threshold = 32768
        contrast = 2.0
        gamma = 1.2
    else:
        erosion_threshold = 65535
        contrast = 2.5
        gamma = 1.4

    sigma = 1.8

    erosion_filter = f"erosion=threshold0={erosion_threshold}:coordinates=255,"

    print(f"Mask Gen Params: gblur={sigma:.1f}, erosion={erosion_threshold}, contrast={contrast}, gamma={gamma}")

    circle_mask = get_circle_mask(overlay_size)

    if is_nvidia():
        filter_complex = _build_filter_complex_nvidia(
            out_w, out_h, overlay_size, half_overlay, sigma, erosion_filter, contrast, gamma,
        )
    else:
        filter_complex = _build_filter_complex_cpu(
            out_w, out_h, overlay_size, half_overlay, sigma, erosion_filter, contrast, gamma,
        )

    cmd: list[str] = [
        "ffmpeg", "-y",

        # threading
        "-filter_threads", "0",
        "-threads", "0",
    ]

    if is_nvidia():
        cmd += [
            # CUDA device for filters + encoder
            "-init_hw_device", "cuda=cuda",
            "-filter_hw_device", "cuda",

            # main video: NVDEC to CUDA frames
            "-hwaccel", "cuda",
            "-hwaccel_output_format", "cuda",
            "-i", video_path,

            # mask input (cpu)
            "-i", mask_path,

            # circle mask (looped)
            "-loop", "1",
            "-i", circle_mask,
        ]
    else:
        cmd += [
            # main video
            "-i", video_path,

            # mask input
            "-i", mask_path,

            # circle mask (looped; bounded by -t to match main video duration)
            "-loop", "1",
            *(["-framerate", f"{fps}"] if fps else []),
            *(["-t", f"{duration}"] if duration else []),
            "-i", circle_mask,
        ]

    cmd += [
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-map", "0:a?",

        # keep duration aligned to shortest stream
        "-shortest",

        # video encoding
        "-c:v", encoder, *encoder_args(preset=preset, cq=cq),

        # audio passthrough
        "-c:a", "copy",

        output_path,
    ]

    return cmd


def pack_video(
    video_path: str,
    mask_path: str,
    output_path: str | None = None,
    sync_frames: int = None,
    preset: str | None = None,
    cq: str = '24',
) -> int:
    """Pack a single video with the mask. Optionally sync mask first"""
    if not output_path:
        base, ext = os.path.splitext(video_path)
        output_path = f"{base}_alpha{ext}"

    print(f"{'='*60}")
    print(f"Alpha packing: {os.path.basename(video_path)}")
    print(f"{'='*60}")

    # optional mask sync
    actual_mask = mask_path
    synced_tmp = None
    if sync_frames is not None:
        _, _, fps, _ = get_video_info(mask_path)
        print(f"Syncing mask by {sync_frames} frame(s)...")
        synced_tmp = sync_mask_to_video(mask_path, fps=fps, frame_offset=sync_frames)
        actual_mask = synced_tmp

    video_w, video_h, video_fps, video_duration = get_video_info(video_path)
    mask_w, mask_h, *_ = get_video_info(actual_mask)

    encoder = get_output_encoder(video_path)

    print(f"Video: {video_w}x{video_h}")
    print(f"Encoder: {encoder}")
    print(f"Mask: {mask_w}x{mask_h}")

    cmd = create_alpha_pack_command(
        video_path, actual_mask, output_path, (video_w, video_h),
        preset=preset, cq=cq, encoder=encoder,
        duration=video_duration, fps=video_fps,
    )

    rc, _ = run_ffmpeg_with_progress(cmd)

    # clean up synced temp file
    if synced_tmp and os.path.exists(synced_tmp):
        os.remove(synced_tmp)

    if rc == 0:
        print(f"🎬 Output: {output_path}")
        return 0

    print(f"❌ Failed to pack {video_path}")
    return 1

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pack alpha mask into FISHEYE190 VR videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("mask", help="Alpha mask video path")
    parser.add_argument("video", help="Video to use alpha mask on")
    parser.add_argument(
        "--quality",
        choices=QUALITY_PRESETS.keys(),
        default="high",
        help="Encoding quality (higher = better quality and bigger file) (default: high)"
    )
    parser.add_argument(
        "--speed",
        choices=SPEED_PRESETS.keys(),
        default="normal",
        help="Encoding speed (slower = better quality) (default: normal)"
    )
    parser.add_argument(
        "--sync",
        type=int,
        default=None,
        metavar="N",
        help="Shift the mask by N frames before packing. Positive = mask catches up (plays earlier), negative = mask delayed (plays later) (default: disabled)"
    )
    args = parser.parse_args()

    preset = SPEED_PRESETS[args.speed]
    cq = QUALITY_PRESETS[args.quality]

    if not os.path.exists(args.mask):
        print(f"❌ Mask not found: {args.mask}")
        return 1

    if not os.path.isfile(args.video):
        print(f"❌ Video not found: {args.video}")
        return 1

    return pack_video(args.video, args.mask, sync_frames=args.sync, preset=preset, cq=cq)


if __name__ == "__main__":
    raise SystemExit(main())
