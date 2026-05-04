import os
import sys
import shutil
import tempfile
import argparse
import numpy as np
from utils.ffmpeg_utils import (
    get_video_info, run_ffmpeg_with_progress,
    get_output_encoder, SPEED_PRESETS, QUALITY_PRESETS,
)


def save_raw_f32_le(filename: str, data_f32: np.ndarray) -> None:
    data_f32.astype("<f4", copy=False).tofile(filename)


def generate_maps_f32_norm(
    dest_w: int,
    dest_h: int,
    src_w: int,
    src_h: int,
    x_map_file: str,
    y_map_file: str,
) -> None:
    """
    Float maps for remap_opencl:
    - store pixel_coord/65535.0 (normalized)
    - unmapped pixels use 1.0 (equivalent to 65535 after scaling)
    """
    print(f"Generating float maps ({dest_w}x{dest_h})...")

    y, x = np.mgrid[0:dest_h, 0:dest_w].astype(np.float32)

    half_w = dest_w // 2
    cx_left = half_w / 2.0
    cx_right = half_w + (half_w / 2.0)
    cy = dest_h / 2.0

    dy = y - cy
    left_mask = x < half_w
    right_mask = ~left_mask

    dx = np.empty_like(x, dtype=np.float32)
    dx[left_mask] = x[left_mask] - cx_left
    dx[right_mask] = x[right_mask] - cx_right

    r = np.hypot(dx, dy)

    max_radius_pix = min(half_w, dest_h) / 2.0
    dest_fov_deg = 190.0
    dest_max_theta = np.deg2rad(dest_fov_deg / 2.0)

    src_fov_deg = 180.0
    stretch_factor = src_fov_deg / dest_fov_deg

    theta_dest = r * (dest_max_theta / max_radius_pix)
    theta_src = theta_dest * stretch_factor
    phi = np.arctan2(dy, dx)

    sp_x = np.sin(theta_src) * np.cos(phi)
    sp_y = np.sin(theta_src) * np.sin(phi)
    sp_z = np.cos(theta_src)

    lon = np.arctan2(sp_x, sp_z)
    lat = -np.arcsin(sp_y)

    u = (lon + (np.pi / 2.0)) / np.pi
    v = 1.0 - ((lat + (np.pi / 2.0)) / np.pi)

    src_half_w = src_w // 2
    valid_mask = (theta_dest <= dest_max_theta)

    # unmapped pixel coordinate => 65535 (1.0 after normalization)
    map_x_pix = np.full((dest_h, dest_w), 65535.0, dtype=np.float32)
    map_y_pix = np.full((dest_h, dest_w), 65535.0, dtype=np.float32)

    # left eye
    mask_l = valid_mask & left_mask
    map_x_pix[mask_l] = u[mask_l] * src_half_w
    map_y_pix[mask_l] = v[mask_l] * src_h

    # right eye
    mask_r = valid_mask & right_mask
    map_x_pix[mask_r] = (u[mask_r] * src_half_w) + src_half_w
    map_y_pix[mask_r] = v[mask_r] * src_h

    inv65535 = np.float32(1.0 / 65535.0)
    map_x = map_x_pix * inv65535
    map_y = map_y_pix * inv65535

    save_raw_f32_le(x_map_file, map_x)
    save_raw_f32_le(y_map_file, map_y)


def run_ffmpeg_conversion(
    input_video: str,
    output_video: str,
    x_map_raw: str,
    y_map_raw: str,
    map_w: int,
    map_h: int,
    fps: float,
    target_w: int,
    target_h: int,
    interp: str = "lanczos",
    preset: str = "p4",
    cq: str = "24",
    encoder: str = "hevc_nvenc",
) -> None:
    print(f"Starting FISHEYE190 conversion...")
    print(f"Encoder: {encoder}")

    filter_complex = (
        f"[0:v]format=yuv444p,hwupload[v_big];"
        f"[1:v]format=grayf32le,hwupload[xm];"
        f"[2:v]format=grayf32le,hwupload[ym];"
        f"[v_big][xm][ym]remap_opencl=fill=black[out_big];"
        f"[out_big]hwdownload,format=yuv444p,hwupload_cuda,"
        f"scale_cuda={target_w}:{target_h}:interp_algo={interp}:format=nv12[out]"
    )

    cmd = [
        "ffmpeg", "-y",
        "-hide_banner",
        "-stats",
        "-nostdin",

        "-init_hw_device", "opencl=ocl",
        "-init_hw_device", "cuda=cud",
        "-filter_hw_device", "ocl",

        "-hwaccel", "cuda",
        "-i", input_video,

        # x map: one raw frame
        "-f", "rawvideo",
        "-pix_fmt", "grayf32le",
        "-s", f"{map_w}x{map_h}",
        "-r", f"{fps}",
        "-i", x_map_raw,

        # y map: one raw frame
        "-f", "rawvideo",
        "-pix_fmt", "grayf32le",
        "-s", f"{map_w}x{map_h}",
        "-r", f"{fps}",
        "-i", y_map_raw,

        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-map", "0:a?", "-c:a", "copy",

        "-c:v", encoder,
        "-preset", preset,
        "-cq", cq,
        "-b:v", "0",
        "-temporal-aq", "1",
        "-spatial-aq", "1",
        "-rc-lookahead", "32",

        output_video
    ]

    rc, _ = run_ffmpeg_with_progress(cmd)

    if rc != 0:
        raise RuntimeError(f"❌ FFmpeg failed with exit code {rc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert VR180 to FISHEYE190")
    parser.add_argument("input", help="Input VR180 video")
    parser.add_argument(
        "--interp",
        choices=["bilinear", "bicubic", "lanczos"],
        default="lanczos",
        help="Interpolation (default: lanczos)",
    )
    parser.add_argument(
        "--quality",
        choices=QUALITY_PRESETS.keys(),
        default="high",
        help="Encoding quality (higher = better quality and bigger file) (default: high)",
    )
    parser.add_argument(
        "--speed",
        choices=SPEED_PRESETS.keys(),
        default="normal",
        help="Encoding speed (slower = better quality) (default: normal)",
    )
    args = parser.parse_args()
    
    preset = SPEED_PRESETS[args.speed]
    cq = QUALITY_PRESETS[args.quality]

    input_file = args.input
    filename, ext = os.path.splitext(input_file)
    output_file = f"{filename}_FISHEYE190{ext}"

    temp_dir = tempfile.mkdtemp(prefix="fisheye190_")
    x_map_file = os.path.join(temp_dir, "map_x.f32")
    y_map_file = os.path.join(temp_dir, "map_y.f32")

    try:
        src_w, src_h, fps, _ = get_video_info(input_file)
        map_w, map_h = src_w, src_h

        generate_maps_f32_norm(map_w, map_h, src_w, src_h, x_map_file, y_map_file)

        encoder = get_output_encoder(input_file)

        run_ffmpeg_conversion(
            input_file, output_file,
            x_map_file, y_map_file,
            map_w, map_h, fps,
            src_w, src_h,
            interp=args.interp,
            preset=preset,
            cq=cq,
            encoder=encoder
        )

        print("✅ Conversion finished")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())