# VR Masking Tools

Automated pipeline to generate alpha matte masks of a subject and pack them into FISHEYE190 VR video. Uses MatAnyone2 for masking, and finetuned SAM3 for automated person detection and accurate first-frame mask generation

<img width="1430" height="360" alt="banner_gif" src="https://github.com/user-attachments/assets/35a51c45-79e1-4901-bdf1-5f0f722e2749" />

## Structure

- `pipeline.py`: Generates a mask for a given video
- `alpha_packer.py`: Packs alpha masks into FISHEYE190 video
- `fisheye190_converter.py`: Converts VR180 video to FISHEYE190 for packing the alpha mask
- `accurate_cut.py`: Cuts videos

## Installation

You can run this code on Linux and WSL2, except for `fisheye190_converter.py`, which runs on Linux and Windows, but not WSL2. A separate repo for the fisheye converter is available here (link) in case you want a Windows standalone installation for it

### Prerequisites

- 16GB of RAM and an NVIDIA GPU with at least 8GB of VRAM

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
ffmpeg -encoders | grep nvenc
```

Check that your FFmpeg has OpenCL support (only needed for VR180 to FISHEYE190 conversion):
```bash
ffmpeg -filters | grep opencl
```

## Docker

--- HERE DOCKER INSTRUCTIONS ---

## Usage

<video src="https://github.com/user-attachments/assets/a7d56c29-3daa-4a41-996f-2cd392c2e65f" controls="controls" style="max-width: 100%;">

In order to pack an alpha mask into a video, the video must be in FISHEYE190 format, so if your video is in VR180 format, the first thing you should do is convert it with `fisheye190_converter.py`. Then, create a mask for that FISHEYE190 video with `pipeline.py`. And finally, use `alpha_packer.py` to pack the alpha mask into the edges of the video so that you can view it on VR players like DeoVR. Since masking takes a long time, I included `accurate_cut.py` so you can cut a shorter segment first to use it for testing

### pipeline.py

Use it to generate a mask of a person from a FISHEYE190 VR video

```bash
python pipeline.py video.mp4
```

**Flags:**

- `--center-crop` : (default: disabled) Crops 10% from each side of the video before masking. Since FISHEYE190 video is 2 circles, I **highly recommend** using this flag. It results in **70% faster masking** and **-50% VRAM usage**. I would only not enable it if the video's subject often gets close to the up/down left/right edges of the circle

- `--mask-height` : (default: 1280, min: 640, max: 1600) Height of the mask in pixels. It accepts multiples of 16. My recommendation is leaving it at the default 1280. Below that, you'll get faster masking but lower mask quality. Above that, you'll get marginal improvements at significantly slower speeds

- `--intro-end` : (default: auto) The pipeline automatically detects the end of the intro when the subject enters the frame. If your intro always finishes at the same time and you want to skip that part, you can set the intro end time in seconds with this flag

- `--prompt` : (default: "woman") SAM3 prompt for automatic subject detection. Since this model is finetuned for women, **you shouldn't use this flag** unless you've replaced the `sam3_finetuned.pth` weights with SAM3's base weights. From my own experience however, the results for other prompts with the base weights aren't consistent enough for VR video

Masking speed is dependent on the size of the mask. I recommend 960, 1120, 1280 or 1440 for mask size, and enabling `--center-crop`.

On an RTX 3070, with `--mask-height 1280` and `--center-crop` enabled, you can expect masking speeds of ~6 fps. When you add the initial segment cutting, resizing, SAM3 first-frame mask generation, and final concat, this results in ~2 hours of masking for a 10-minute 60 fps video. With 8GB of VRAM, you can't go above 1280px mask height with `--center-crop` enabled, or 1120px with it disabled.

On an RTX 5090 or RTX 6000 Ada, you can expect masking speeds of ~15 fps at 1280px and `--center-crop` enabled. This results in ~50 minutes of masking for a 10 minute 60 fps video

### alpha_packer.py

Use it to pack your generated alpha mask into your FISHEYE190 video

```bash
python alpha_packer.py mask.mp4 video.mp4
```

**Flags:**

- `--quality` : (default: high) FFmpeg cq settings for video encoding. Higher means better quality but larger file size. `ultra=18`, `high=24`, `normal=26`, `low=28`

- `--speed` : (default: normal) FFmpeg preset settings for video encoding. Slower means slower encoding but better quality. `slow=p6`, `normal=p4`, `fast=p2`

- `--sync` : (default: disabled) Sync shifts the mask by N frames before packing. Only change it if the resulting mask isn't perfectly synced with the video. Positive = mask catches up (plays earlier), negative = mask delayed (plays later)

### fisheye190_converter.py

Use it to convert VR180 video into FISHEYE190. Since it relies on `remap_opencl`, you can use it on Windows and Linux, but you **can't use it on WSL2**. Here's a separate repo for it (link) if you want a standalone Windows install

```bash
python fisheye190_converter.py video.mp4
```

**Flags:**

- `--quality` : (default: high) FFmpeg cq settings for video encoding. Higher means better quality but larger file size. `ultra=18`, `high=24`, `normal=26`, `low=28`

- `--speed` : (default: normal) FFmpeg preset settings for video encoding. Slower means slower encoding but better quality. `slow=p6`, `normal=p4`, `fast=p2`

- `--interp` : (default: lanczos) FFmpeg remap interpolation algorithm. Lanczos is the highest quality. You can use bilinear or bicubic too, but the speeds don't change much

### accurate_cut.py

Use it to cut and concatenate segments from a video. Great for testing the other scripts with shorter scenes

Create a `cuts.txt` file in the script's directory with the following format. The content between the left and right timestamps of every row will be included in the output video

```
00:05:10, 00:05:30
00:10:00, 00:12:30
```

```bash
python accurate_cut.py video.mp4
```

## License

This repository is **multi-licensed**:

* All original code in this repository is licensed under the [MIT License](LICENSE).
* The `MatAnyone2/` directory and the downloaded `.pth` pretrained model weights are licensed under the **S-Lab License 1.0**, which restricts use to **non-commercial research purposes only**.

The MatAnyone2 components are **not covered by the MIT License**.

If you intend to use the MatAnyone2 components for commercial purposes, you must obtain permission from the original authors. See `MatAnyone2/LICENSE.txt` for full details.
