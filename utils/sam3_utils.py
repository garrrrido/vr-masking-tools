"""
SAM3 utility functions for pipeline integration
Handles SAM3 batch execution and intro detection natively via PyTorch.
"""

import shutil
import gc
import os
import logging
import warnings
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from utils.ffmpeg_utils import extract_left_eye_frames

INTRO_FRAME_INTERVAL = 2.0      # seconds between intro detection frames
INTRO_BATCH_SIZE = 15           # number of frames per intro detection batch
INTRO_MIN_WHITE_RATIO = 0.003   # 0.3% of pixels must be white to count as detection

SAM3_REPO_ID = "garrrrido/sam3-finetuned"


def _run_sam3_inference(frames_dir: str, output_size: int | None = None, is_intro: bool = False, prompt: str = "woman") -> None:
    """Internal function to load SAM3 and process frames"""
    warnings.filterwarnings("ignore", message=".*pkg_resources.*")
    
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    import huggingface_hub
    from huggingface_hub import snapshot_download
        
    try:
        # try to load from local cache first
        repo_path = snapshot_download(repo_id=SAM3_REPO_ID, token=False, local_files_only=True)
    except Exception:
        # if not cached, download from network
        print("⬇️ Downloading finetuned SAM3 weights...")
        try:
            # suppress huggingface_hub warnings and progress bars
            logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
            huggingface_hub.utils.disable_progress_bars()
            
            repo_path = snapshot_download(repo_id=SAM3_REPO_ID, token=False)
        except Exception as e:
            raise RuntimeError(f"Failed to download SAM3 finetuned weights: {e}")
            
    model_path = os.path.join(repo_path, "sam3_finetuned.pth")

    model = build_sam3_image_model(load_from_HF=False)
    checkpoint = torch.load(model_path, weights_only=False, map_location='cpu')
    model.load_state_dict(checkpoint["model_state_dict"])
    
    processor = Sam3Processor(model, confidence_threshold=0.2)
    
    folder = Path(frames_dir)
    image_files = list(folder.glob("*.png")) + list(folder.glob("*.jpg"))
    image_files = [f for f in image_files if "_mask" not in f.stem]

    if not image_files:
        return

    SAM3_MAX = 1600

    with torch.inference_mode():
        for frame_path in sorted(image_files):
            output_path = frame_path.parent / f"{frame_path.stem}_mask.png"
            
            raw = Image.open(frame_path)
            image = raw.convert("RGB")
            raw.close()
            original_size = (image.width, image.height)
            
            if image.width > SAM3_MAX or image.height > SAM3_MAX:
                full = image
                image = full.resize((SAM3_MAX, SAM3_MAX), Image.LANCZOS)
                full.close()
                
            inference_state = processor.set_image(image)
            output = processor.set_text_prompt(state=inference_state, prompt=prompt)
            
            masks = output["masks"]
            scores = output["scores"]
            
            if len(masks) == 0:
                best_mask = np.zeros((image.height, image.width), dtype=np.uint8)
            else:
                if isinstance(scores, torch.Tensor):
                    scores = scores.cpu().numpy()
                if isinstance(masks, torch.Tensor):
                    masks = masks.cpu().numpy()
                
                best_idx = np.argmax(scores)
                best_mask = masks[best_idx]
                
                if len(best_mask.shape) == 3:
                    best_mask = best_mask[0]
                    
                best_mask = (best_mask * 255).astype(np.uint8)

            del inference_state, output
            image.close()
            
            mask_image = Image.fromarray(best_mask)
            target_size = output_size if output_size else original_size[0]
            if mask_image.width != target_size or mask_image.height != target_size:
                mask_image = mask_image.resize((target_size, target_size), Image.NEAREST)
            
            mask_image.save(output_path)
            
            gc.collect()
            torch.cuda.empty_cache()

    del processor, model, checkpoint
    gc.collect()
    torch.cuda.empty_cache()


def run_sam3_batch(
    frames_dir: str,
    output_size: int | None = None,
    prompt: str = "woman",
    quiet: bool = False,
) -> None:
    """Run SAM3 batch on all frames"""
    _run_sam3_inference(frames_dir, output_size=output_size, is_intro=quiet, prompt=prompt)


def detect_intro_end(
    video_path: str,
    orig_height: int,
    duration: float,
    prompt: str = None,
) -> float:
    """
    Auto-detect intro end by finding 2 consecutive SAM3 detections
    Processes frames in batches of INTRO_BATCH_SIZE, every INTRO_FRAME_INTERVAL seconds
    """
    intro_dir = Path('temp_pipeline') / 'intro_frames'
    
    batch_start = 0.0
    prev_detected = False   # whether the last frame of the previous batch had a detection
    prev_timestamp = 0.0    # timestamp of the last frame of the previous batch
    
    while batch_start < duration:
        # generate timestamps for this batch
        timestamps = []
        for i in range(INTRO_BATCH_SIZE):
            ts = batch_start + i * INTRO_FRAME_INTERVAL
            if ts >= duration:
                break
            timestamps.append(ts)
        
        if not timestamps:
            break
        
        # clean and recreate intro_frames dir for this batch
        if intro_dir.exists():
            shutil.rmtree(intro_dir)
        intro_dir.mkdir(parents=True, exist_ok=True)
        
        # extract frames
        extracted = extract_left_eye_frames(video_path, timestamps, str(intro_dir), orig_height)
        
        if not extracted:
            raise RuntimeError("Frame extraction failed during intro detection")
        
        # mask with SAM3
        _run_sam3_inference(str(intro_dir), is_intro=True, prompt=prompt)
        
        # review masks in order
        for ts in timestamps:
            mask_path = intro_dir / f"frame_{ts:.0f}s_mask.png"
            if not mask_path.exists():
                prev_detected = False
                continue
            
            mask = np.array(Image.open(str(mask_path)).convert('L'))
            total_pixels = mask.shape[0] * mask.shape[1]
            white_pixels = np.count_nonzero(mask > 128)
            white_ratio = white_pixels / total_pixels
            detected = white_ratio >= INTRO_MIN_WHITE_RATIO
            
            if detected and prev_detected:
                intro_end = prev_timestamp
                if intro_dir.exists():
                    shutil.rmtree(intro_dir)
                return intro_end
            
            prev_detected = detected
            prev_timestamp = ts
        
        batch_start += INTRO_BATCH_SIZE * INTRO_FRAME_INTERVAL
    
    raise RuntimeError("No subject found in video")