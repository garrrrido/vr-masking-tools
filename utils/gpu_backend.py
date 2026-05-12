"""
GPU backend selection for FFmpeg encode/decode/filter paths.

Two backends are supported:
- 'nvidia': NVENC/NVDEC + CUDA filters (scale_cuda, overlay_cuda, hwupload/hwdownload)
- 'amd':    AMF encoder, software decode and software filters

Selection order:
1. Env var GPU_BACKEND ('nvidia'/'cuda'/'nvenc' or 'amd'/'amf'/'rocm')
2. Auto-probe ffmpeg encoders (prefer NVIDIA when both are available)
3. Fallback to 'nvidia'
"""

import os
import subprocess


_BACKEND: str | None = None


def _probe_encoder(encoder: str) -> bool:
    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
        '-f', 'lavfi', '-i', 'nullsrc=s=256x256:d=0.04:r=25',
        '-c:v', encoder, '-frames:v', '1', '-f', 'null', '-',
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return r.returncode == 0
    except Exception:
        return False


def get_backend() -> str:
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND

    env = os.environ.get('GPU_BACKEND', '').strip().lower()
    if env in ('nvidia', 'cuda', 'nvenc'):
        _BACKEND = 'nvidia'
    elif env in ('amd', 'amf', 'rocm'):
        _BACKEND = 'amd'
    else:
        if _probe_encoder('hevc_nvenc'):
            _BACKEND = 'nvidia'
        elif _probe_encoder('hevc_amf'):
            _BACKEND = 'amd'
        else:
            _BACKEND = 'nvidia'
    return _BACKEND


def is_nvidia() -> bool:
    return get_backend() == 'nvidia'


def is_amd() -> bool:
    return get_backend() == 'amd'


# --- Encoder names ---

def hevc_encoder() -> str:
    return 'hevc_nvenc' if is_nvidia() else 'hevc_amf'


def av1_encoder() -> str:
    return 'av1_nvenc' if is_nvidia() else 'av1_amf'


# --- Preset / speed mappings ---

_SPEED_PRESETS_NV = {'slow': 'p6', 'normal': 'p4', 'fast': 'p2'}
_SPEED_PRESETS_AMD = {'slow': 'quality', 'normal': 'balanced', 'fast': 'speed'}


def speed_presets() -> dict[str, str]:
    return _SPEED_PRESETS_NV if is_nvidia() else _SPEED_PRESETS_AMD


def default_preset() -> str:
    return 'p4' if is_nvidia() else 'balanced'


# --- Encoder argument blocks ---

def encoder_args(preset: str | None = None, cq: str = '24', extras: bool = True) -> list[str]:
    """
    Return ffmpeg encoder arguments to follow `-c:v <encoder>`.

    `preset` is in backend-native naming (e.g. 'p4' for NVENC, 'balanced' for AMF).
    Pass None to use the backend default.
    `extras` enables NVENC quality knobs (temporal-aq / spatial-aq / rc-lookahead); ignored on AMD.
    """
    if preset is None:
        preset = default_preset()

    if is_nvidia():
        args = ['-preset', preset, '-cq', cq]
        if extras:
            args += ['-temporal-aq', '1', '-spatial-aq', '1', '-rc-lookahead', '32']
        return args
    else:
        return ['-quality', preset, '-rc', 'cqp', '-qp_i', cq, '-qp_p', cq]


# --- Hardware accel for input demux/decode ---

def hwaccel_input_args() -> list[str]:
    """ffmpeg flags to place before `-i` to enable hardware decode + GPU frames."""
    return ['-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda'] if is_nvidia() else []
