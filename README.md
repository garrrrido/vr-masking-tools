# VR Masking Tools

Automated pipeline to generate alpha matte masks of a subject and pack them into FISHEYE190 VR video. Uses MatAnyone2 for masking, and finetuned SAM3 for automated person detection and accurate first-frame mask generation

<img width="1430" height="360" alt="banner_gif" src="https://github.com/user-attachments/assets/35a51c45-79e1-4901-bdf1-5f0f722e2749" />


## Installation

You can run all scripts on Linux and WSL2, except for `fisheye190_converter.py`, which runs on Linux and Windows, but not WSL2. A separate repo for the fisheye converter is [available here](https://github.com/garrrrido/fisheye-converter) in case you want a Windows standalone installation for it

### Prerequisites

- 16GB of RAM and a GPU with at least 8GB of VRAM (NVIDIA recommended; AMD Radeon is also supported via ROCm on Windows — see [AMD ROCm setup](#amd-rocm-setup-windows))

- Have the newest NVIDIA drivers installed and CUDA Toolkit 13.1

- Pip and venv:
  ```bash
  sudo apt install -y python3-pip python3-venv
  ```

- FFmpeg 8 with NVENC/NVDEC/CUDA and OpenCL support

  <details>
  <summary>FFmpeg 8 Linux installation commands</summary>

  ```bash
  sudo apt update
  sudo apt install -y build-essential pkg-config git nasm yasm cmake libtool \
    libssl-dev zlib1g-dev libnuma-dev ocl-icd-opencl-dev \
    libx264-dev libx265-dev libvpx-dev libmp3lame-dev libopus-dev libvorbis-dev \
    libass-dev libfreetype6-dev libfdk-aac-dev

  mkdir -p ~/ffmpeg_sources && cd ~/ffmpeg_sources

  git clone --depth 1 https://github.com/FFmpeg/nv-codec-headers.git
  cd nv-codec-headers && sudo make install && cd ..

  git clone -b release/8.0 https://github.com/FFmpeg/FFmpeg.git ffmpeg
  cd ffmpeg

  CUDA_HOME=/usr/local/cuda
  CCAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '. ')

  ./configure --prefix=/usr/local --bindir=/usr/local/bin \
    --enable-gpl --enable-version3 --enable-openssl \
    --enable-nonfree --enable-libfdk-aac \
    --enable-libx264 --enable-libx265 --enable-libvpx \
    --enable-libmp3lame --enable-libopus --enable-libvorbis \
    --enable-libass --enable-libfreetype \
    --enable-cuda-nvcc --enable-nvenc --enable-nvdec --enable-cuvid \
    --enable-opencl \
    --extra-cflags="-I${CUDA_HOME}/include" \
    --extra-ldflags="-L${CUDA_HOME}/lib64" \
    --nvccflags="-gencode arch=compute_${CCAP},code=sm_${CCAP} -O2"

  make -j"$(nproc)"
  sudo make install
  sudo ldconfig
  hash -r
  ```

  </details>

### Instructions

Go to your install directory and run the following commands:

```bash
git clone https://github.com/garrrrido/vr-masking-tools.git
cd vr-masking-tools
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### AMD ROCm setup (Windows)

The scripts also run on AMD Radeon GPUs on Windows via ROCm + the AMF FFmpeg encoder. The GPU backend is selected at runtime:

1. **Auto-detect (default)**: the first `ffmpeg` encoder probe wins. `hevc_nvenc` → NVIDIA path; otherwise `hevc_amf` → AMD path.
2. **Force a backend**: set the `GPU_BACKEND` environment variable to `nvidia` or `amd` before running any script.

**Prerequisites for the AMD path:**

- Python 3.12 (required by the ROCm wheels)
- AMD Adrenalin / Radeon driver 26.2.2 or newer
- FFmpeg built with AMF support (`hevc_amf`, `av1_amf`)

**Installation** (see the [AMD docs](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/install-pytorch.html) for details):

Install the ROCm SDK first (provides the runtime PyTorch links against), then the Python dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install --no-cache-dir `
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl `
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl `
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl `
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm-7.2.1.tar.gz

pip install -r requirements-rocm.txt
```

To force the AMD path at runtime (skip auto-detection):

```powershell
$env:GPU_BACKEND = "amd"
```

## Usage

https://github.com/user-attachments/assets/a7d56c29-3daa-4a41-996f-2cd392c2e65f

In order to pack an alpha mask into a video, the video must be in FISHEYE190 format. If it's in VR180 format, you can convert it with `fisheye190_converter.py`. Then, generate a mask for that FISHEYE190 video with `pipeline.py`. Use `alpha_packer.py` to pack the alpha mask into the edges of the video so that you can view it on VR players like DeoVR. Since masking takes a long time, I included `accurate_cut.py` so you can cut a shorter segment first to use it for testing

- `pipeline.py`: Generates a mask for a given video
- `alpha_packer.py`: Packs alpha masks into FISHEYE190 video
- `fisheye190_converter.py`: Converts VR180 video to FISHEYE190 for packing the alpha mask
- `accurate_cut.py`: Cuts videos

### pipeline.py

Generate a mask of a person from a FISHEYE190 VR video

```bash
python pipeline.py video.mp4
```

**Flags:**

- `--center-crop` : (default: disabled) Crops 10% from each side of the video before masking. I **highly recommend** using this flag. It results in **70% faster masking** and **-50% VRAM usage**. I would only not enable it if the video's subject often gets close to the up/down left/right edges of the circles

- `--mask-height` : (default: 1280, min: 640, max: 1600) Height of the mask in pixels. It accepts multiples of 16. My recommendation is leaving it at the default 1280. Below that, you'll get faster masking but lower mask quality. Above that, you'll get marginal improvements at significantly slower speeds

- `--intro-end` : (default: auto) The pipeline automatically detects the end of the intro when the subject enters the frame. If your intro always finishes at the same time and you want to skip that part, you can set the intro end time in seconds with this flag

- `--prompt` : (default: "woman") SAM3 prompt for automatic subject detection. Since this model is finetuned for women, **you shouldn't use this flag** unless you've replaced the `sam3_finetuned.pth` weights with SAM3's base weights. From my own experience however, the results for other prompts with the base weights aren't consistent enough for VR video

On an RTX 3070, with `--mask-height 1280` and `--center-crop` enabled, expect masking speeds of ~6 fps. With all the other steps, this results in ~2 hours of masking for a 10 minute 60fps video. With 8GB of VRAM, you can't go above 1280px mask height with `--center-crop` enabled, or 1120px with it disabled

On an RTX 5090 or RTX 6000 Ada, you can expect masking speeds of ~15 fps at 1280px and `--center-crop` enabled. This results in ~50 minutes of masking for a 10 minute 60fps video

### alpha_packer.py

Pack your generated alpha mask into your FISHEYE190 video

```bash
python alpha_packer.py mask.mp4 video.mp4
```

**Flags:**

- `--quality` : (default: high) FFmpeg cq/qp settings for video encoding. Higher = better quality, larger file size. `ultra=18`, `high=24`, `normal=26`, `low=28`

- `--speed` : (default: normal) FFmpeg preset for video encoding. Slower = slower encoding, better quality. NVENC: `slow=p6`, `normal=p4`, `fast=p2`. AMF: `slow=quality`, `normal=balanced`, `fast=speed`

- `--sync` : (default: disabled) Sync shifts the mask by N frames before packing. Only change it if the resulting mask isn't perfectly synced with the video. Positive = mask catches up (plays earlier), negative = mask delayed (plays later)

### fisheye190_converter.py

Convert VR180 video into FISHEYE190. Since it relies on `remap_opencl`, you can use it on Windows and Linux, but you **can't use it on WSL2**. Here's a separate [repo for it](https://github.com/garrrrido/fisheye-converter) if you want a standalone Windows install

```bash
python fisheye190_converter.py video.mp4
```

**Flags:**

- `--quality` : (default: high) FFmpeg cq/qp settings for video encoding. Higher = better quality, larger file size. `ultra=18`, `high=24`, `normal=26`, `low=28`

- `--speed` : (default: normal) FFmpeg preset for video encoding. Slower = slower encoding, better quality. NVENC: `slow=p6`, `normal=p4`, `fast=p2`. AMF: `slow=quality`, `normal=balanced`, `fast=speed`

- `--interp` : (default: lanczos) Interpolation algorithm. Lanczos is the highest quality. You can use bilinear or bicubic too, but the speeds don't change much

- `--ocl-device` : (default: first OpenCL device ffmpeg finds) OpenCL device used by `remap_opencl`, in `<platform>.<device>` form (e.g. `0.0`, `1.0`). Useful if your machine has multiple OpenCL devices (e.g. a CPU runtime listed before the GPU, or an integrated GPU alongside a discrete one) and ffmpeg picks the wrong one. You can also set the `OCL_DEVICE` environment variable instead. To list available devices:
  ```bash
  ffmpeg -hide_banner -v verbose -init_hw_device opencl -f lavfi -i nullsrc -f null -
  ```

### accurate_cut.py

Cut and concatenate segments from a video. Great for testing the other scripts with shorter scenes

Create a `cuts.txt` file in the script's directory with the following format. The content between the left and right timestamps of every row will be included in the output video

```
00:05:10, 00:05:30
00:10:00, 00:12:30
```

```bash
python accurate_cut.py video.mp4
```

### Troubleshooting

Check that your system has NVIDIA drivers:
```bash
nvidia-smi
```

Check that your system has CUDA Toolkit 13 or up:
```bash
nvcc --version
```

Check that you have FFmpeg 8 installed:
```bash
ffmpeg -version
```


Check that your FFmpeg has the required encoders:
```bash
# NVIDIA backend
ffmpeg -encoders | grep nvenc
# AMD backend
ffmpeg -encoders | grep amf
```

If both encoders are available and you want to force one, set the `GPU_BACKEND` environment variable to `nvidia` or `amd` before running a script.

Check that your FFmpeg has OpenCL support (only needed for VR180 to FISHEYE190 conversion):
```bash
ffmpeg -filters | grep opencl
```

## License

This repository is **multi-licensed**:

* All original code in this repository is licensed under the [MIT License](LICENSE).
* The `MatAnyone2/` directory and the downloaded `.pth` pretrained model weights are licensed under the **S-Lab License 1.0**, which restricts use to **non-commercial research purposes only**.

The MatAnyone2 components are **not covered by the MIT License**. If you intend to use the MatAnyone2 components for commercial purposes, you must obtain permission from the original authors. See `MatAnyone2/LICENSE.txt` for full details.
