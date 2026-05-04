import os
import gc
import sys
import json
import cv2
import tqdm
import imageio
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from matanyone2.utils.inference_utils import gen_dilate, gen_erosion, read_frame_from_videos
from matanyone2.inference.inference_core import InferenceCore
from matanyone2.utils.get_default_model import get_matanyone2_model
from matanyone2.utils.device import get_default_device, safe_autocast_decorator

import warnings
warnings.filterwarnings("ignore")

device = get_default_device()


# terminal status display (same ANSI cursor logic as pipeline.py's _update_status)
_is_first_status = True
_tqdm_lines = 1

def _update_status(op_num, total_ops, label, duration):
    """Print status line above tqdm area, overwriting previous status + tqdm"""
    global _is_first_status
    if not _is_first_status:
        sys.stderr.write(f"\033[{1 + _tqdm_lines}A")
    sys.stderr.write(f"\r[{op_num}/{total_ops}] {label} ({duration:.1f}s)\033[K\n")
    for i in range(_tqdm_lines):
        sys.stderr.write(f"\r\033[K")
        if i < _tqdm_lines - 1:
            sys.stderr.write("\n")
    sys.stderr.flush()
    _is_first_status = False


@torch.inference_mode()
@safe_autocast_decorator()
def process_segment(matanyone2, job):
    """Process a single segment using the already-loaded model"""
    input_path = job['input_path']
    mask_path = job['mask_path']
    output_path = job['output_path']
    n_warmup = int(job.get('warmup', 10))
    r_erode = int(job.get('erode', 10))
    r_dilate = int(job.get('dilate', 10))
    suffix = job.get('suffix', '')
    max_size = int(job.get('max_size', -1))
    fps_override = job.get('fps', None)

    # create fresh processor (cheap — just Python objects, no weight loading)
    processor = InferenceCore(matanyone2, cfg=matanyone2.cfg)

    # load input frames
    vframes, fps, length, video_name = read_frame_from_videos(input_path)

    # use explicit fps if provided (for sync with original video)
    if fps_override is not None:
        fps = float(fps_override)
    repeated_frames = vframes[0].unsqueeze(0).repeat(n_warmup, 1, 1, 1) # repeat the first frame for warmup
    vframes = torch.cat([repeated_frames, vframes], dim=0).float()
    length += n_warmup  # update length

    # resize if needed
    if max_size > 0:
        h, w = vframes.shape[-2:]
        min_side = min(h, w)
        if min_side > max_size:
            new_h = int(h / min_side * max_size)
            new_w = int(w / min_side * max_size)
            vframes = F.interpolate(vframes, size=(new_h, new_w), mode="area")
        else:
            new_h, new_w = h, w  # no resize needed, use original dimensions
        
    # set output paths
    os.makedirs(output_path, exist_ok=True)
    if suffix != "":
        video_name = f'{video_name}_{suffix}'

    # load the first-frame mask
    mask = Image.open(mask_path).convert('L')
    mask = np.array(mask)

    objects = [1]

    # [optional] erode & dilate
    if r_dilate > 0:
        mask = gen_dilate(mask, r_dilate, r_dilate)
    if r_erode > 0:
        mask = gen_erosion(mask, r_erode, r_erode)

    mask = torch.from_numpy(mask).float().to(device)

    if max_size > 0:  # resize needed
        mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(new_h, new_w), mode="nearest")
        mask = mask[0,0]

    # inference start
    phas = []
    for ti in tqdm.tqdm(range(length)):
        # load the image as RGB; normalization is done within the model
        image = vframes[ti]

        image = (image / 255.).float().to(device)  # for network input

        if ti == 0:
            output_prob = processor.step(image, mask, objects=objects)  # encode given mask
            output_prob = processor.step(image, first_frame_pred=True)  # first frame for prediction
        else:
            if ti <= n_warmup:
                output_prob = processor.step(image, first_frame_pred=True)  # reinit as the first frame for prediction
            else:
                output_prob = processor.step(image)

        # convert output probabilities to alpha matte
        mask = processor.output_prob_to_mask(output_prob)

        # save results (skip warmup frames)
        if ti > (n_warmup-1):
            pha = mask.unsqueeze(2).cpu().numpy()
            pha = np.round(np.clip(pha * 255.0, 0, 255)).astype(np.uint8)
            phas.append(pha)

    phas = np.array(phas)

    output_file = f'{output_path}/{video_name}_pha.mp4'
    imageio.mimwrite(output_file, phas, fps=fps, quality=7)

    # teardown: free all segment-specific memory
    del processor, vframes, phas, mask
    torch.cuda.empty_cache()
    gc.collect()

    return output_file


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobs', type=str, required=True, help='Path to JSON file with list of segment jobs')
    args = parser.parse_args()

    # load jobs
    with open(args.jobs, 'r') as f:
        jobs = json.load(f)

    # download ckpt if needed, load model once
    pretrain_model_url = "https://github.com/pq-yang/MatAnyone2/releases/download/v1.0.0/matanyone2.pth"
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "pretrained_models"))
    os.makedirs(model_dir, exist_ok=True)
    ckpt_path = os.path.join(model_dir, "matanyone2.pth")
    if not os.path.exists(ckpt_path):
        sys.stderr.write(f"⬇️ Downloading MatAnyone2 weights...\n")
        sys.stderr.flush()
        torch.hub.download_url_to_file(pretrain_model_url, ckpt_path, progress=False)
        
    matanyone2 = get_matanyone2_model(ckpt_path, device)

    # process all segments
    for job in jobs:
        # display status
        _update_status(job['op_num'], job['total_ops'], job['label'], job['duration'])

        output_file = process_segment(matanyone2, job)

        # signal completion to the runner (stdout is captured by runner)
        sys.stdout.write(f"DONE:{output_file}\n")
        sys.stdout.flush()

    sys.stderr.write("\n")